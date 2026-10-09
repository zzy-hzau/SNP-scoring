#!/usr/bin/env python3


"LOCO coding proxy-label builder with fold-specific train-only N_e."

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import pickle
import re
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


BASE2INT = {"A": 0, "C": 1, "G": 2, "T": 3}
BASES = ("A", "C", "G", "T")
GROUP = "Maize"

BASE_REQUIRED = [
    "Chr", "Pos", "Ref", "Alt", "Y_Label",
    "chr_num", "Sg_code", "Ctx_code", "Alt_code",
]

ID_CANDIDATES = ["snp_id", "SNP_ID"]

OPTIONAL_KEEP_COLS = [
    "snp_id", "SNP_ID",
    "GeneID", "gene_id",
    "Mutation", "aa_pos", "ref_aa", "alt_aa",
    "observed_in_vcf", "AN_obs", "callable_n", "callrate_hit",
    "AC", "AN", "AF",
]

OUTPUT_COLS = [
    "snp_id", "SNP_ID",
    "Chr", "Pos", "Ref", "Alt",
    "GeneID", "gene_id", "Mutation", "aa_pos", "ref_aa", "alt_aa",
    "chr_num", "Sg_code", "Ctx_code", "Alt_code",
    "Y_Label", "observed_in_vcf",
    "AC", "AN", "AF", "AN_obs", "callable_n", "callrate_hit",
    "P_mut_fold", "p_base_eff_fold",
    "p_base_eff_bin", "callable_n_bin", "match_group",
    "new_label", "fold", "split",
]


def log(msg: str):
    print(msg, flush=True)


def parse_folds(spec: str) -> List[int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def val_chr_for_test(test_chr: int, n_chr: int) -> int:
    return test_chr + 1 if test_chr < n_chr else 1


def infer_chr_num(chrom: str, n_chr: int = 10) -> int:
    m = re.search(r"(\d+)", str(chrom))
    if not m:
        return -1
    x = int(m.group(1))
    if 1 <= x <= n_chr:
        return x
    return -1


def stable_hash01(x: str, seed: int = 17) -> float:
    return (zlib.crc32(f"{seed}|{x}".encode("utf-8")) & 0xFFFFFFFF) / 2**32


def stable_keep(keys, probs, seed: int) -> np.ndarray:
    keys = np.asarray(keys, dtype=object)
    probs = np.asarray(probs, dtype=np.float64)
    if len(keys) == 0:
        return np.zeros(0, dtype=bool)
    vals = np.fromiter((stable_hash01(str(k), seed) for k in keys), dtype=np.float64, count=len(keys))
    return vals < probs


def make_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise RuntimeError("Cannot build edges from empty values.")
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        mn, mx = float(values.min()), float(values.max())
        if mn == mx:
            edges = np.array([mn - 1e-12, mx + 1e-12], dtype=np.float64)
        else:
            edges = np.linspace(mn, mx, min(n_bins + 1, 6))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def digitize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, values, side="right") - 1


def ctx_code_from_seq(seq: str) -> int:
    if len(seq) != 3:
        return -1
    vals = [BASE2INT.get(b, -1) for b in seq.upper()]
    if min(vals) < 0:
        return -1
    return vals[0] * 16 + vals[1] * 4 + vals[2]


def alt_code_from_base(alt: str) -> int:
    return BASE2INT.get(str(alt).upper(), -1)


def table_key_to_codes(k: str) -> Optional[Tuple[int, int, int]]:
    parts = str(k).split("_")
    if len(parts) != 3:
        return None
    sg, ctx, alt = parts

    if sg in ("Maize", "Single", "M", "S", "G0", "0", "At", "A"):
        sgc = 0
    elif sg in ("Dt", "D", "G1", "1"):
        sgc = 1
    else:
        try:
            sgc = int(sg)
        except Exception:
            return None

    if len(ctx) == 3 and all(b in BASE2INT for b in ctx):
        cc = ctx_code_from_seq(ctx)
    else:
        try:
            cc = int(ctx)
        except Exception:
            return None

    if alt in BASE2INT:
        ac = BASE2INT[alt]
    else:
        try:
            ac = int(alt)
        except Exception:
            return None

    return sgc, cc, ac


def build_rate_array(rate_table: dict, fallback: float) -> np.ndarray:
    arr = np.full((4, 64, 4), np.nan, dtype=np.float64)
    for k, v in rate_table.items():
        codes = table_key_to_codes(k)
        if codes is None:
            continue
        sg, ctx, alt = codes
        if 0 <= sg < 4 and 0 <= ctx < 64 and 0 <= alt < 4:
            arr[sg, ctx, alt] = float(v)
    return np.nan_to_num(arr, nan=fallback, posinf=fallback, neginf=fallback)


def load_pickle_or_joblib(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        import joblib
        return joblib.load(path)


def subtract_counter(dst: Counter, src: dict):
    for k, v in src.items():
        dst[k] -= v
        if dst[k] <= 0:
            dst.pop(k, None)


def subtract_mut(dst: dict, src: dict):
    for ctx, alt_dict in src.items():
        for alt, v in alt_dict.items():
            dst[ctx][alt] -= v
            if dst[ctx][alt] <= 0:
                dst[ctx].pop(alt, None)
        if not dst[ctx]:
            dst.pop(ctx, None)


def build_rate_table_excluding_chrnums(
    genome_counts_total: dict,
    genome_counts_by_chrnum: dict,
    mutation_counts_total: dict,
    mutation_counts_by_chrnum: dict,
    exclude_chrs: Iterable[int],
    mu_base: float,
    pseudo_count: float,
    group: str = GROUP,
) -> dict:
    exclude_chrs = set(int(x) for x in exclude_chrs)

    if group not in genome_counts_total:

        group = list(genome_counts_total.keys())[0]

    cur_gen = Counter(genome_counts_total.get(group, {}))
    cur_mut = defaultdict(lambda: defaultdict(int))

    for ctx, alt_dict in mutation_counts_total.get(group, {}).items():
        for alt, count in alt_dict.items():
            cur_mut[ctx][alt] = int(count)

    for chr_num in exclude_chrs:
        if chr_num in genome_counts_by_chrnum:
            subtract_counter(cur_gen, genome_counts_by_chrnum[chr_num].get(group, {}))
        if chr_num in mutation_counts_by_chrnum:
            subtract_mut(cur_mut, mutation_counts_by_chrnum[chr_num].get(group, {}))

    sum_gen = sum(cur_gen.values())
    sum_mut = sum(sum(alt_dict.values()) for alt_dict in cur_mut.values())
    avg_obs_rate = sum_mut / sum_gen if sum_gen > 0 else 1e-10
    avg_obs_rate = max(avg_obs_rate, 1e-12)

    out = {}
    for ctx, denom in cur_gen.items():
        if denom <= 0 or len(ctx) != 3 or any(b not in BASE2INT for b in ctx):
            continue
        ref = ctx[1]
        for alt in BASES:
            if alt == ref:
                continue
            mut = cur_mut.get(ctx, {}).get(alt, 0)
            obs_rate = (mut + pseudo_count) / (denom + pseudo_count * 3.0)
            relative_rate = obs_rate / avg_obs_rate
            out[f"{GROUP}_{ctx}_{alt}"] = float(mu_base * relative_rate)
    return out


def ensure_columns(dataset: ds.Dataset):
    cols = set(dataset.schema.names)
    missing = [c for c in BASE_REQUIRED if c not in cols]
    if missing:
        raise RuntimeError(f"Input coding universe missing required columns: {missing}")
    if not any(c in cols for c in ID_CANDIDATES):
        raise RuntimeError("Input coding universe must contain snp_id or SNP_ID")


def scanner_cols(dataset: ds.Dataset) -> List[str]:
    want = list(dict.fromkeys(BASE_REQUIRED + OPTIONAL_KEEP_COLS))
    return [c for c in want if c in dataset.schema.names]


def normalize_batch(df: pd.DataFrame, args) -> pd.DataFrame:
    df = df.copy()

    if "snp_id" not in df.columns and "SNP_ID" in df.columns:
        df["snp_id"] = df["SNP_ID"].astype(str)
    elif "snp_id" in df.columns:
        df["snp_id"] = df["snp_id"].astype(str)
    else:
        df["snp_id"] = (
            df["Chr"].astype(str) + "_" +
            df["Pos"].astype(str) + "_" +
            df["Ref"].astype(str) + "_" +
            df["Alt"].astype(str)
        )

    if "SNP_ID" not in df.columns:
        df["SNP_ID"] = df["snp_id"]

    df["Chr"] = df["Chr"].astype(str)
    df["Pos"] = pd.to_numeric(df["Pos"], errors="coerce").fillna(-1).astype(np.int64)
    df["Ref"] = df["Ref"].astype(str).str.upper().str[:1]
    df["Alt"] = df["Alt"].astype(str).str.upper().str[:1]

    df["Y_Label"] = pd.to_numeric(df["Y_Label"], errors="coerce").fillna(0).astype(np.int8)

    if "observed_in_vcf" not in df.columns:
        df["observed_in_vcf"] = df["Y_Label"]
    df["observed_in_vcf"] = pd.to_numeric(df["observed_in_vcf"], errors="coerce").fillna(0).astype(np.int8)

    if "chr_num" not in df.columns or df["chr_num"].isna().any():
        df["chr_num"] = df["Chr"].map(lambda x: infer_chr_num(x, args.n_chr)).astype(np.int16)
    else:
        df["chr_num"] = pd.to_numeric(df["chr_num"], errors="coerce").fillna(-1).astype(np.int16)

    if "Sg_code" not in df.columns or df["Sg_code"].isna().any():
        df["Sg_code"] = 0
    else:
        df["Sg_code"] = pd.to_numeric(df["Sg_code"], errors="coerce").fillna(0).astype(np.int16)

    df["Ctx_code"] = pd.to_numeric(df["Ctx_code"], errors="coerce").fillna(-1).astype(np.int16)

    if "Alt_code" not in df.columns or df["Alt_code"].isna().any():
        df["Alt_code"] = df["Alt"].map(alt_code_from_base).astype(np.int16)
    else:
        df["Alt_code"] = pd.to_numeric(df["Alt_code"], errors="coerce").fillna(-1).astype(np.int16)

    if "callable_n" not in df.columns:
        if "AN_obs" in df.columns:
            df["callable_n"] = pd.to_numeric(df["AN_obs"], errors="coerce").fillna(0) / 2.0
        else:
            df["callable_n"] = 0
    df["callable_n"] = pd.to_numeric(df["callable_n"], errors="coerce").fillna(0).astype(np.float32)

    if "AN_obs" not in df.columns:
        df["AN_obs"] = df["callable_n"] * 2.0
    df["AN_obs"] = pd.to_numeric(df["AN_obs"], errors="coerce").fillna(0).astype(np.float32)

    if "callrate_hit" not in df.columns:
        df["callrate_hit"] = (df["callable_n"] > 0).astype(np.int8)
    df["callrate_hit"] = pd.to_numeric(df["callrate_hit"], errors="coerce").fillna(0).astype(np.int8)

    for c in ["GeneID", "gene_id", "Mutation", "ref_aa", "alt_aa"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype(str)

    if "aa_pos" not in df.columns:
        df["aa_pos"] = -1
    df["aa_pos"] = pd.to_numeric(df["aa_pos"], errors="coerce").fillna(-1).astype(np.int32)

    for c in ["AC", "AN", "AF"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float32)

    return df


def attach_pmut(df: pd.DataFrame, rate_arr: np.ndarray, ne: float, args) -> pd.DataFrame:
    sg = pd.to_numeric(df["Sg_code"], errors="coerce").fillna(-1).to_numpy(np.int16)
    ctx = pd.to_numeric(df["Ctx_code"], errors="coerce").fillna(-1).to_numpy(np.int16)
    alt = pd.to_numeric(df["Alt_code"], errors="coerce").fillna(-1).to_numpy(np.int16)

    mu = np.full(len(df), args.fallback_mu, dtype=np.float64)
    ok = (sg >= 0) & (sg < rate_arr.shape[0]) & (ctx >= 0) & (ctx < 64) & (alt >= 0) & (alt < 4)
    if np.any(ok):
        mu[ok] = rate_arr[sg[ok], ctx[ok], alt[ok]]

    cn = pd.to_numeric(df["callable_n"], errors="coerce").fillna(0).to_numpy(np.float64)
    exposure = np.clip(cn / float(max(args.max_callable_n, 1)), 0.0, 1.0)

    df["P_mut_fold"] = mu
    df["p_base_eff_fold"] = -np.expm1(-float(ne) * mu * exposure)
    return df.replace([np.inf, -np.inf], np.nan)


def candidate_masks(df: pd.DataFrame, p_threshold: float, args):
    y = pd.to_numeric(df["Y_Label"], errors="coerce").fillna(0).astype(np.int8)
    cn = pd.to_numeric(df["callable_n"], errors="coerce").fillna(0).astype(np.float64)
    p = pd.to_numeric(df["p_base_eff_fold"], errors="coerce").fillna(-np.inf).astype(np.float64)
    common = (df["callrate_hit"].astype(np.int8) == 1) & (cn >= args.min_callable_n)
    is0 = common & (y == 1)
    is1 = common & (y == 0) & (p >= p_threshold)
    return is0, is1


def attach_bins_and_group(df: pd.DataFrame, p_edges: np.ndarray, cn_edges: np.ndarray, args) -> pd.DataFrame:
    df["p_base_eff_bin"] = digitize(df["p_base_eff_fold"].to_numpy(np.float64), p_edges).astype(np.int16)
    df["callable_n_bin"] = digitize(df["callable_n"].to_numpy(np.float64), cn_edges).astype(np.int16)

    group = (
        df["Sg_code"].astype(str) + "|" +
        df["Ctx_code"].astype(str) + "|" +
        df["Alt_code"].astype(str) + "|" +
        df["p_base_eff_bin"].astype(str) + "|" +
        df["callable_n_bin"].astype(str)
    )

    if args.match_aa_change:
        group = group + "|" + df["ref_aa"].astype(str) + ">" + df["alt_aa"].astype(str)

    df["match_group"] = group
    return df


def calibrate_ne_for_train_chrs(dataset, train_chrs, args, rate_arr):
    log("[Ne] calibrating strict train-only LOCO N_e ...")
    train_chrs = set(int(x) for x in train_chrs)
    code_base = int(args.max_callable_n) + 1
    cols = scanner_cols(dataset)

    count_by_key = defaultdict(int)
    obs_by_key = defaultdict(float)

    scanned = 0
    kept = 0
    bad_code = 0

    for rb in dataset.scanner(columns=cols, batch_size=args.batch_size).to_batches():
        df = normalize_batch(rb.to_pandas(), args)
        df = df[df["chr_num"].astype(int).isin(train_chrs)]
        if len(df) == 0:
            continue
        scanned += len(df)

        y = df["Y_Label"].to_numpy(np.int8)
        sg = df["Sg_code"].to_numpy(np.int16)
        ctx = df["Ctx_code"].to_numpy(np.int16)
        alt = df["Alt_code"].to_numpy(np.int16)
        cn = np.rint(df["callable_n"].to_numpy(np.float64)).astype(np.int32)
        cn = np.clip(cn, 0, int(args.max_callable_n))
        hit = df["callrate_hit"].to_numpy(np.int8)

        ok_code = (
            (sg >= 0) & (sg < rate_arr.shape[0]) &
            (ctx >= 0) & (ctx < 64) &
            (alt >= 0) & (alt < 4)
        )
        ok = (hit == 1) & (cn >= args.min_callable_n) & ok_code
        bad_code += int(((hit == 1) & (cn >= args.min_callable_n) & (~ok_code)).sum())

        if not np.any(ok):
            continue

        sg_ok = sg[ok].astype(np.int64)
        ctx_ok = ctx[ok].astype(np.int64)
        alt_ok = alt[ok].astype(np.int64)
        cn_ok = cn[ok].astype(np.int64)
        y_ok = y[ok].astype(np.float64)

        key = ((((sg_ok * 64) + ctx_ok) * 4 + alt_ok) * code_base + cn_ok).astype(np.int64)
        uniq, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv).astype(np.int64)
        obs = np.bincount(inv, weights=y_ok).astype(np.float64)

        for k, c, o in zip(uniq.tolist(), cnt.tolist(), obs.tolist()):
            count_by_key[int(k)] += int(c)
            obs_by_key[int(k)] += float(o)

        kept += int(ok.sum())

    if not count_by_key:
        raise RuntimeError("No callable train rows for N_e calibration.")

    keys = np.array(list(count_by_key.keys()), dtype=np.int64)
    counts = np.array([count_by_key[int(k)] for k in keys], dtype=np.float64)
    obs = np.array([obs_by_key[int(k)] for k in keys], dtype=np.float64)

    cn = keys % code_base
    tmp = keys // code_base
    alt = tmp % 4
    tmp = tmp // 4
    ctx = tmp % 64
    sg = tmp // 64

    mu = rate_arr[sg, ctx, alt].astype(np.float64)
    exposure = cn.astype(np.float64) / float(max(args.max_callable_n, 1))
    x = mu * exposure

    finite = np.isfinite(x) & (x > 0)
    x = x[finite]
    counts = counts[finite]
    obs = obs[finite]

    target = float(obs.sum())
    total = float(counts.sum())
    sum_x = float(np.sum(counts * x))

    if target <= 0:
        raise RuntimeError("N_e calibration failed: observed train count is zero.")
    if target >= total:
        raise RuntimeError(f"N_e calibration failed: observed >= total ({target} >= {total}).")

    def expected(ne):
        return float(np.sum(counts * (-np.expm1(-float(ne) * x))))

    lo = 0.0
    hi = max(1.0, target / max(sum_x, 1e-300))
    if args.ne_seed and args.ne_seed > 0:
        hi = max(hi, float(args.ne_seed))

    max_hi = float(args.ne_calibration_max)
    e_hi = expected(hi)
    while e_hi < target and hi < max_hi:
        hi *= 2.0
        e_hi = expected(hi)

    if e_hi < target:
        raise RuntimeError(
            f"N_e calibration failed: expected({hi:.3e})={e_hi:.3e} < observed={target:.3e}."
        )

    for _ in range(args.ne_calibration_iter):
        mid = 0.5 * (lo + hi)
        if expected(mid) < target:
            lo = mid
        else:
            hi = mid

    ne = 0.5 * (lo + hi)
    exp_ne = expected(ne)
    info = {
        "ne": float(ne),
        "train_chrs": sorted(train_chrs),
        "scanned_train_rows": int(scanned),
        "callable_rows_used": int(kept),
        "aggregation_groups": int(len(keys)),
        "groups_used": int(len(x)),
        "observed_sum": float(target),
        "expected_sum_at_ne": float(exp_ne),
        "relative_error": float((exp_ne - target) / max(target, 1.0)),
        "linear_ne_initial_estimate": float(target / max(sum_x, 1e-300)),
        "skipped_bad_code_rows": int(bad_code),
        "formula": "sum_train(1-exp(-Ne*P_mut_train*callable_n/max_callable_n)) = sum_train(Y_Label)",
    }
    log(f"[Ne] calibrated Ne={ne:.8g} observed={target:.3f} expected={exp_ne:.3f} relerr={info['relative_error']:.3e} rows={kept:,}")
    return float(ne), info


def collect_train_edges(dataset, train_chrs, args, rate_arr, ne):
    log("[edges] collecting train-only p_base_eff/callable_n edges ...")
    train_chrs = set(int(x) for x in train_chrs)
    cols = scanner_cols(dataset)

    pvals = []
    cnvals = []
    got = 0

    for rb in dataset.scanner(columns=cols, batch_size=args.batch_size).to_batches():
        df = normalize_batch(rb.to_pandas(), args)
        df = df[df["chr_num"].astype(int).isin(train_chrs)]
        if len(df) == 0:
            continue

        df = attach_pmut(df, rate_arr, ne, args)
        common = (df["callrate_hit"] == 1) & (df["callable_n"] >= args.min_callable_n)
        df = df[common]
        if len(df) == 0:
            continue

        if args.edge_sample_fraction >= 1.0:
            sub = df
        else:
            keep = stable_keep(df["snp_id"].to_numpy(), np.full(len(df), args.edge_sample_fraction), args.seed + 909)
            sub = df.loc[keep]

        if len(sub):
            pvals.append(sub["p_base_eff_fold"].to_numpy(np.float64))
            cnvals.append(sub["callable_n"].to_numpy(np.float64))
            got += len(sub)

        if got >= args.sample_rows_for_bins:
            break

    if not pvals:
        raise RuntimeError("No train values collected for edges.")

    p = np.concatenate(pvals)[:args.sample_rows_for_bins]
    cn = np.concatenate(cnvals)[:args.sample_rows_for_bins]

    p_edges = make_edges(p, args.opportunity_bins)
    cn_edges = make_edges(cn, args.callable_n_bins)
    p_thr = float(np.quantile(p[np.isfinite(p)], args.opportunity_min_quantile))

    log(f"[edges] p_threshold q={args.opportunity_min_quantile}: {p_thr:.8e}; sample_n={len(p):,}")
    return p_edges, cn_edges, p_thr


def count_groups_for_split(dataset, chr_keep, args, rate_arr, ne, p_edges, cn_edges, p_thr):
    chr_keep = set(int(x) for x in chr_keep)
    cols = scanner_cols(dataset)

    count0, count1 = Counter(), Counter()
    cand0 = cand1 = scanned = 0

    for rb in dataset.scanner(columns=cols, batch_size=args.batch_size).to_batches():
        df = normalize_batch(rb.to_pandas(), args)
        df = df[df["chr_num"].astype(int).isin(chr_keep)]
        if len(df) == 0:
            continue
        scanned += len(df)

        df = attach_pmut(df, rate_arr, ne, args)
        df = attach_bins_and_group(df, p_edges, cn_edges, args)

        is0, is1 = candidate_masks(df, p_thr, args)

        if is0.any():
            vc = df.loc[is0, "match_group"].value_counts()
            count0.update(vc.to_dict())
            cand0 += int(is0.sum())

        if is1.any():
            vc = df.loc[is1, "match_group"].value_counts()
            count1.update(vc.to_dict())
            cand1 += int(is1.sum())

    return count0, count1, cand0, cand1, scanned


def decide_sampling_probs(count0: Counter, count1: Counter, args):
    probs0, probs1 = {}, {}
    rows = []

    for g in sorted(set(count0) & set(count1)):
        n0, n1 = int(count0[g]), int(count1[g])
        if n0 < args.min_group_label0 or n1 < args.min_group_label1:
            continue

        t0 = min(n0, args.max_label0_per_group)
        t1 = min(n1, int(math.ceil(t0 * args.label1_per_label0)))
        t0 = min(t0, int(math.floor(t1 / max(args.label1_per_label0, 1e-12))))

        if t0 < args.min_group_label0 or t1 < args.min_group_label1:
            continue

        probs0[g] = t0 / n0
        probs1[g] = t1 / n1

        rows.append({
            "match_group": g,
            "count0": n0,
            "count1": n1,
            "target0": t0,
            "target1": t1,
            "prob0": probs0[g],
            "prob1": probs1[g],
        })

    return probs0, probs1, pd.DataFrame(rows)


def write_selected_split(dataset, out_path, chr_keep, split, fold, args, rate_arr, ne, p_edges, cn_edges, p_thr, probs0, probs1):
    chr_keep = set(int(x) for x in chr_keep)
    cols = scanner_cols(dataset)

    writer = None
    stats = {"scanned": 0, "written0": 0, "written1": 0}

    for rb in dataset.scanner(columns=cols, batch_size=args.batch_size).to_batches():
        df = normalize_batch(rb.to_pandas(), args)
        df = df[df["chr_num"].astype(int).isin(chr_keep)]
        if len(df) == 0:
            continue
        stats["scanned"] += len(df)

        df = attach_pmut(df, rate_arr, ne, args)
        df = attach_bins_and_group(df, p_edges, cn_edges, args)

        is0, is1 = candidate_masks(df, p_thr, args)

        keep = np.zeros(len(df), dtype=bool)
        lab = np.full(len(df), -1, dtype=np.int8)

        idx0 = np.flatnonzero(is0.to_numpy())
        if len(idx0):
            sub = df.iloc[idx0]
            p0 = sub["match_group"].map(probs0).fillna(0.0).to_numpy(np.float64)
            k = stable_keep(sub["snp_id"].to_numpy(), p0, args.seed + fold * 1000 + 11)
            sel = idx0[k]
            keep[sel] = True
            lab[sel] = 0

        idx1 = np.flatnonzero(is1.to_numpy())
        if len(idx1):
            sub = df.iloc[idx1]
            p1 = sub["match_group"].map(probs1).fillna(0.0).to_numpy(np.float64)
            k = stable_keep(sub["snp_id"].to_numpy(), p1, args.seed + fold * 1000 + 101)
            sel = idx1[k]
            keep[sel] = True
            lab[sel] = 1

        out = df.loc[keep].copy()
        if len(out) == 0:
            continue

        out["new_label"] = lab[keep]
        out["fold"] = int(fold)
        out["split"] = split

        for c in OUTPUT_COLS:
            if c not in out.columns:
                out[c] = np.nan
        out = out[OUTPUT_COLS]

        table = pa.Table.from_pandas(out, preserve_index=False)
        if writer is None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(str(out_path), table.schema, compression="snappy")
        else:
            table = table.cast(writer.schema, safe=False)

        writer.write_table(table)
        stats["written0"] += int((out["new_label"] == 0).sum())
        stats["written1"] += int((out["new_label"] == 1).sum())

    if writer is not None:
        writer.close()

    stats["written"] = stats["written0"] + stats["written1"]
    return stats


def export_lm_targets(out_dir: Path):
    rows = []
    for p in sorted(out_dir.glob("fold*/**/*.parquet")):
        if p.name not in {"train.parquet", "val.parquet", "test.parquet"}:
            continue
        pf = pq.ParquetFile(p)
        cols = [c for c in ["snp_id", "SNP_ID", "Chr", "Pos", "Ref", "Alt", "GeneID", "gene_id", "Mutation", "aa_pos", "ref_aa", "alt_aa"] if c in pf.schema.names]
        rows.append(pf.read(columns=cols).to_pandas())

    lm_dir = out_dir / "lm_targets"
    lm_dir.mkdir(exist_ok=True)
    out_path = lm_dir / "coding_lm_targets.tsv.gz"

    if not rows:
        pd.DataFrame(columns=["snp_id", "Chr", "Pos", "Ref", "Alt"]).to_csv(out_path, sep="\t", index=False, compression="gzip")
        return str(out_path), 0

    df = pd.concat(rows, ignore_index=True).drop_duplicates("snp_id")
    df.to_csv(out_path, sep="\t", index=False, compression="gzip")
    return str(out_path), int(len(df))


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", required=True, help="coding_preLM_universe.parquet")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rate-cache", required=True, help="maize_rate_tables_cache.pkl from maize_calc_pmut_fast.py")
    ap.add_argument("--n-chr", type=int, default=10)
    ap.add_argument("--folds", default="1-10")
    ap.add_argument("--max-callable-n", type=int, required=True)

    ap.add_argument("--fallback-mu", type=float, default=4.56e-9)
    ap.add_argument("--mu-base", type=float, default=4.56e-9)
    ap.add_argument("--pmut-pseudo-count", type=float, default=0.5)

    ap.add_argument("--min-callable-n", type=int, default=5)
    ap.add_argument("--opportunity-min-quantile", type=float, default=0.20)
    ap.add_argument("--opportunity-bins", type=int, default=10)
    ap.add_argument("--callable-n-bins", type=int, default=10)

    ap.add_argument("--label1-per-label0", type=float, default=1.0)
    ap.add_argument("--max-label0-per-group", type=int, default=20000)
    ap.add_argument("--min-group-label0", type=int, default=3)
    ap.add_argument("--min-group-label1", type=int, default=3)

    ap.add_argument("--batch-size", type=int, default=300000)
    ap.add_argument("--sample-rows-for-bins", type=int, default=50000000)
    ap.add_argument("--edge-sample-fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=17)

    ap.add_argument("--ne-seed", type=float, default=None)
    ap.add_argument("--ne-calibration-iter", type=int, default=80)
    ap.add_argument("--ne-calibration-max", type=float, default=1e12)

    ap.add_argument("--match-aa-change", action="store_true", help="Optional stricter matching by ref_aa>alt_aa. Default off to avoid over-matching away protein-effect signal.")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ds.dataset(args.input, format="parquet")
    ensure_columns(dataset)

    log("=" * 80)
    log("[coding proxy labels strict LOCO N_e]")
    log(f"input={args.input}")
    log(f"out_dir={out_dir}")
    log(f"rate_cache={args.rate_cache}")
    log(f"max_callable_n={args.max_callable_n}")
    log(f"match_aa_change={args.match_aa_change}")
    log("=" * 80)

    rate_cd = load_pickle_or_joblib(args.rate_cache)
    genome_total = rate_cd["genome_total"]
    genome_by_chrnum = rate_cd["genome_by_chrnum"]
    mut_total = rate_cd["mut_total"]
    mut_by_chrnum = rate_cd["mut_by_chrnum"]

    manifest_all = {
        "input": args.input,
        "out_dir": str(out_dir),
        "rate_cache": args.rate_cache,
        "n_chr": args.n_chr,
        "max_callable_n": args.max_callable_n,
        "label_semantics": {
            "0": "Y_Label=1 wild observed callable missense SNP; tolerated proxy",
            "1": "Y_Label=0 high-opportunity callable wild-unobserved missense SNP; depletion/proxy-deleterious",
        },
        "strict_loco": {
            "P_mut_train": "exclude test_chr and val_chr",
            "P_mut_test": "exclude test_chr",
            "Ne": "calibrated per fold using train_chrs only",
            "threshold_bins": "estimated from train_chrs only",
        },
        "matching_group": "Sg_code|Ctx_code|Alt_code|p_base_eff_bin|callable_n_bin" + ("|ref_aa>alt_aa" if args.match_aa_change else ""),
        "model_feature_note": "Do not use Y_Label/P_mut_fold/p_base_eff_fold/callability/bins/match_group/fold/split as model features.",
        "folds": [],
    }

    for fold in parse_folds(args.folds):
        test_chr = int(fold)
        val_chr = val_chr_for_test(test_chr, args.n_chr)
        train_chrs = [c for c in range(1, args.n_chr + 1) if c not in (test_chr, val_chr)]

        fold_dir = out_dir / f"fold{fold:02d}"
        fold_dir.mkdir(exist_ok=True)

        log("=" * 80)
        log(f"FOLD {fold:02d}: train={train_chrs}, val={val_chr}, test={test_chr}")
        log("=" * 80)

        train_rate = build_rate_table_excluding_chrnums(
            genome_total, genome_by_chrnum, mut_total, mut_by_chrnum,
            exclude_chrs=[test_chr, val_chr],
            mu_base=args.mu_base,
            pseudo_count=args.pmut_pseudo_count,
        )
        test_rate = build_rate_table_excluding_chrnums(
            genome_total, genome_by_chrnum, mut_total, mut_by_chrnum,
            exclude_chrs=[test_chr],
            mu_base=args.mu_base,
            pseudo_count=args.pmut_pseudo_count,
        )

        train_arr = build_rate_array(train_rate, args.fallback_mu)
        test_arr = build_rate_array(test_rate, args.fallback_mu)

        fold_ne, ne_calibration = calibrate_ne_for_train_chrs(dataset, train_chrs, args, train_arr)
        p_edges, cn_edges, p_thr = collect_train_edges(dataset, train_chrs, args, train_arr, fold_ne)

        split_defs = [
            ("train", train_chrs, train_arr),
            ("val", [val_chr], train_arr),
            ("test", [test_chr], test_arr),
        ]

        probs_by_split = {}
        split_manifest = {}

        for split, chrs, arr in split_defs:
            c0, c1, cand0, cand1, scanned = count_groups_for_split(
                dataset, chrs, args, arr, fold_ne, p_edges, cn_edges, p_thr
            )
            probs0, probs1, plan = decide_sampling_probs(c0, c1, args)
            plan_path = fold_dir / f"{split}.sampling_plan.tsv"
            plan.to_csv(plan_path, sep="\t", index=False)
            probs_by_split[split] = (probs0, probs1)

            split_manifest[split] = {
                "scanned_rows": int(scanned),
                "candidate0": int(cand0),
                "candidate1": int(cand1),
                "groups0": int(len(c0)),
                "groups1": int(len(c1)),
                "groups_overlap": int(len(set(c0) & set(c1))),
                "kept_groups": int(len(plan)),
                "planned0": int(plan["target0"].sum()) if len(plan) else 0,
                "planned1": int(plan["target1"].sum()) if len(plan) else 0,
                "sampling_plan": str(plan_path),
            }
            log(f"[fold {fold:02d} {split}] {json.dumps(split_manifest[split], ensure_ascii=False)}")

        write_stats = {}
        for split, chrs, arr in split_defs:
            probs0, probs1 = probs_by_split[split]
            write_stats[split] = write_selected_split(
                dataset, fold_dir / f"{split}.parquet",
                chrs, split, fold, args, arr, fold_ne,
                p_edges, cn_edges, p_thr, probs0, probs1
            )
            log(f"[fold {fold:02d} {split} write] {json.dumps(write_stats[split], ensure_ascii=False)}")

        fold_manifest = {
            "fold": int(fold),
            "train_chrs": train_chrs,
            "val_chr": int(val_chr),
            "test_chr": int(test_chr),
            "ne": float(fold_ne),
            "ne_calibration": ne_calibration,
            "p_base_eff_threshold": float(p_thr),
            "p_base_eff_edges": p_edges.tolist(),
            "callable_n_edges": cn_edges.tolist(),
            "split_count_manifest": split_manifest,
            "write_stats": write_stats,
        }

        with open(fold_dir / "manifest.json", "w") as f:
            json.dump(fold_manifest, f, indent=2)

        manifest_all["folds"].append(fold_manifest)
        with open(out_dir / "manifest_all.partial.json", "w") as f:
            json.dump(manifest_all, f, indent=2)

    target_path, target_n = export_lm_targets(out_dir)
    manifest_all["lm_targets"] = target_path
    manifest_all["lm_target_n"] = target_n

    with open(out_dir / "manifest_all.json", "w") as f:
        json.dump(manifest_all, f, indent=2)

    log("[DONE]")
    log(f"lm_targets={target_path}")
    log(f"lm_target_n={target_n}")


if __name__ == "__main__":
    main()
