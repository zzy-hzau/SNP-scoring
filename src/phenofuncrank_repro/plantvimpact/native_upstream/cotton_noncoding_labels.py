#!/usr/bin/env python3

"Build LOCO CADD-like proxy labels for NONCODING SNPs before LM scoring."

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

try:
    import joblib
except Exception:  
    joblib = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("noncoding_proxy_labels_preLM_strict")

BASE2INT = {"A": 0, "C": 1, "G": 2, "T": 3}

REQUIRED_BASE_COLS = [
    "snp_id", "Chr", "Pos", "Ref", "Alt",
    "region_type", "Y_Label", "chr_num", "Sg_code", "Ctx_code", "Alt_code",
]

OPTIONAL_KEEP_COLS = [
    "GeneID", "dist_to_TSS", "dist_to_gene",
    "promoter_core_flag", "promoter_proximal_flag", "promoter_distal_flag",
    "donor_offset", "acceptor_offset", "canonical_splice_core_flag",
    "homoeolog_expr_bias", "has_homolog",
]

OUTPUT_COLS = [
    "snp_id", "Chr", "Pos", "Ref", "Alt", "GeneID",
    "region_type", "region_subtype", "dist_to_TSS", "dist_to_gene",
    "promoter_core_flag", "promoter_proximal_flag", "promoter_distal_flag",
    "donor_offset", "acceptor_offset", "canonical_splice_core_flag",
    "homoeolog_expr_bias", "has_homolog",
    "chr_num", "Sg_code", "Alt_code", "Ctx_code",
    "Y_Label", "AN_obs", "callable_n", "callrate_hit",
    "P_mut_fold", "p_base_eff_fold", "p_base_eff_bin", "callable_n_bin",
    "match_group", "new_label", "fold", "split",
]

SPLIT_PARQUET_NAMES = ("train.parquet", "val.parquet", "test.parquet")


def atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def load_fold_manifest_if_complete(fold_dir: Path) -> Optional[dict]:
    fold_dir = Path(fold_dir)
    manifest_path = fold_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(f"Skipping fold with unreadable manifest during global export: {manifest_path}; {e}")
        return None

    missing = [name for name in SPLIT_PARQUET_NAMES if not (fold_dir / name).exists()]
    if missing:
        logger.warning(f"Skipping fold with incomplete split outputs during global export: {fold_dir}; missing={missing}")
        return None
    return manifest


def collect_completed_fold_manifests(out_dir: Path) -> List[dict]:
    manifests = []
    for fold_dir in sorted(Path(out_dir).glob("fold*")):
        if not fold_dir.is_dir():
            continue
        manifest = load_fold_manifest_if_complete(fold_dir)
        if manifest is not None:
            manifests.append(manifest)
    return sorted(manifests, key=lambda x: int(x.get("fold", 0)))


def build_rate_table_excluding(genome_total, genome_by_chrnum, mut_total, mut_by_chrnum, exclude_chrs, min_mu: float, pseudo_count: float):
    keys = set(genome_total) | set(mut_total)
    for c in exclude_chrs:
        keys |= set(genome_by_chrnum.get(c, {})) | set(mut_by_chrnum.get(c, {}))
    out = {}
    for k in keys:
        den = float(genome_total.get(k, 0)) - sum(float(genome_by_chrnum.get(c, {}).get(k, 0)) for c in exclude_chrs)
        mut = float(mut_total.get(k, 0)) - sum(float(mut_by_chrnum.get(c, {}).get(k, 0)) for c in exclude_chrs)
        out[k] = min_mu if den <= 0 else max((mut + pseudo_count) / (den + pseudo_count * 4.0), min_mu)
    return out


def attach_region_subtype(df: pd.DataFrame) -> pd.DataFrame:
    r = df["region_type"].astype(str).to_numpy()
    subtype = np.array(r, dtype=object)


    

    
    subtype[r == "promoter"] = "promoter_other"
    subtype[r == "splice_near"] = "splice_near_other"
    if "promoter_core_flag" in df.columns:
        core = pd.to_numeric(df["promoter_core_flag"], errors="coerce").fillna(0).to_numpy()
        prox = pd.to_numeric(df["promoter_proximal_flag"], errors="coerce").fillna(0).to_numpy()
        dist = pd.to_numeric(df["promoter_distal_flag"], errors="coerce").fillna(0).to_numpy()
        m = r == "promoter"
        subtype[m & (core > 0)] = "promoter_core"
        subtype[m & (core <= 0) & (prox > 0)] = "promoter_proximal"
        subtype[m & (core <= 0) & (prox <= 0) & (dist > 0)] = "promoter_distal"
        subtype[m & (core <= 0) & (prox <= 0) & (dist <= 0)] = "promoter_other"
    if "canonical_splice_core_flag" in df.columns:
        sp = pd.to_numeric(df["canonical_splice_core_flag"], errors="coerce").fillna(0).to_numpy()
        m = r == "splice_near"
        subtype[m & (sp > 0)] = "splice_core"
        subtype[m & (sp <= 0)] = "splice_near_other"
    df["region_subtype"] = subtype.astype(str)
    return df


def calibrate_ne_for_train_chrs(dataset, train_chrs, args, call_idx, rate_arr):
    logger.info("Calibrating fold-specific N_e on train chromosomes only ...")
    train_chrs = set(int(x) for x in train_chrs)
    max_cn = int(max(args.max_callable_n, 1))
    code_base = max_cn + 1

    count_by_key = defaultdict(np.int64)
    obs_by_key = defaultdict(np.float64)

    scanned = 0
    kept = 0
    skipped_bad_code = 0

    cols = list(lean_scanner_cols(dataset, mode="count"))
    for _subtype_col in (
        "promoter_core_flag",
        "promoter_proximal_flag",
        "promoter_distal_flag",
        "canonical_splice_core_flag",
    ):
        if _subtype_col in dataset.schema.names and _subtype_col not in cols:
            cols.append(_subtype_col)
    for rb in dataset.scanner(columns=cols, batch_size=args.batch_size).to_batches():
        df = rb.to_pandas()
        df = normalize_batch(df, args)
        df = df[df["chr_num"].astype(int).isin(train_chrs)]
        if len(df) == 0:
            continue

        scanned += len(df)
        df = attach_callrate(df, call_idx)

        y = pd.to_numeric(df["Y_Label"], errors="coerce").fillna(0).to_numpy(np.int8)
        sg = pd.to_numeric(df["Sg_code"], errors="coerce").fillna(-1).to_numpy(np.int16)
        ctx = pd.to_numeric(df["Ctx_code"], errors="coerce").fillna(-1).to_numpy(np.int16)
        alt = pd.to_numeric(df["Alt_code"], errors="coerce").fillna(-1).to_numpy(np.int16)
        cn = pd.to_numeric(df["callable_n"], errors="coerce").fillna(0).to_numpy(np.float64)
        hit = pd.to_numeric(df["callrate_hit"], errors="coerce").fillna(0).to_numpy(np.int8)

        cn_int = np.rint(cn).astype(np.int32)
        cn_int = np.clip(cn_int, 0, max_cn)

        ok_code = (
            (sg >= 0) & (sg < rate_arr.shape[0]) &
            (ctx >= 0) & (ctx < 64) &
            (alt >= 0) & (alt < 4)
        )
        ok = (
            (hit == 1) &
            (cn_int >= int(args.min_callable_n)) &
            ok_code
        )

        skipped_bad_code += int(((hit == 1) & (cn_int >= int(args.min_callable_n)) & (~ok_code)).sum())

        if not np.any(ok):
            continue

        sg_ok = sg[ok].astype(np.int64, copy=False)
        ctx_ok = ctx[ok].astype(np.int64, copy=False)
        alt_ok = alt[ok].astype(np.int64, copy=False)
        cn_ok = cn_int[ok].astype(np.int64, copy=False)
        y_ok = y[ok].astype(np.float64, copy=False)

        key = ((((sg_ok * 64) + ctx_ok) * 4 + alt_ok) * code_base + cn_ok).astype(np.int64)
        uniq, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv).astype(np.int64)
        obs = np.bincount(inv, weights=y_ok).astype(np.float64)

        for k, c, o in zip(uniq.tolist(), cnt.tolist(), obs.tolist()):
            count_by_key[int(k)] += int(c)
            obs_by_key[int(k)] += float(o)

        kept += int(ok.sum())

    if not count_by_key:
        raise RuntimeError("N_e calibration failed: no callable train rows after filtering.")

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
    exposure = cn.astype(np.float64) / float(max_cn)
    x = mu * exposure

    finite = np.isfinite(x) & (x > 0) & np.isfinite(counts)
    if not np.any(finite):
        raise RuntimeError("N_e calibration failed: no finite positive P_mut exposure groups.")

    x = x[finite]
    counts = counts[finite]
    obs = obs[finite]

    target_obs = float(obs.sum())
    total_rows = float(counts.sum())
    sum_x = float(np.sum(counts * x))

    if target_obs <= 0:
        raise RuntimeError("N_e calibration failed: train observed count is zero.")
    if target_obs >= total_rows:
        raise RuntimeError(
            f"N_e calibration failed: observed count >= total rows ({target_obs} >= {total_rows})."
        )

    def expected(ne):
        return float(np.sum(counts * (-np.expm1(-float(ne) * x))))

    lo = 0.0
    hi = float(args.ne) if args.ne is not None and float(args.ne) > 0 else max(1.0, target_obs / max(sum_x, 1e-300))


    e_hi = expected(hi)
    max_hi = float(args.ne_calibration_max)
    while e_hi < target_obs and hi < max_hi:
        hi *= 2.0
        e_hi = expected(hi)

    if e_hi < target_obs:
        raise RuntimeError(
            f"N_e calibration failed: expected({hi:.6e})={e_hi:.6e} < observed={target_obs:.6e}; "
            f"increase --ne-calibration-max."
        )

    for _ in range(int(args.ne_calibration_iter)):
        mid = 0.5 * (lo + hi)
        if expected(mid) < target_obs:
            lo = mid
        else:
            hi = mid

    ne = 0.5 * (lo + hi)
    exp_at_ne = expected(ne)
    info = {
        "ne": float(ne),
        "train_chrs": sorted(int(x) for x in train_chrs),
        "scanned_train_rows": int(scanned),
        "callable_rows_used": int(kept),
        "aggregation_groups": int(len(keys)),
        "groups_used": int(len(x)),
        "observed_sum": float(target_obs),
        "expected_sum_at_ne": float(exp_at_ne),
        "relative_error": float((exp_at_ne - target_obs) / max(target_obs, 1.0)),
        "sum_x_linear_approx": float(sum_x),
        "linear_ne_initial_estimate": float(target_obs / max(sum_x, 1e-300)),
        "skipped_bad_code_rows": int(skipped_bad_code),
        "formula": "sum(1-exp(-Ne*P_mut_fold*callable_n/max_callable_n)) = sum(Y_Label) on train chromosomes only",
    }

    logger.info(
        "LOCO N_e calibrated: "
        f"Ne={ne:.8g}, observed={target_obs:.3f}, expected={exp_at_ne:.3f}, "
        f"used_rows={kept:,}, groups={len(x):,}, relerr={info['relative_error']:.3e}"
    )
    if skipped_bad_code:
        logger.warning(f"N_e calibration skipped rows with invalid Sg/Ctx/Alt code: {skipped_bad_code:,}")

    return float(ne), info


def count_groups_for_split(dataset, chr_keep, args, call_idx, rate_arr, ne, p_edges, cn_edges, p_thr):
    chr_keep = set(int(x) for x in chr_keep)
    count0, count1 = Counter(), Counter()
    cand0 = cand1 = scanned = 0
    cols = list(lean_scanner_cols(dataset, mode="count"))
    for _subtype_col in (
        "promoter_core_flag",
        "promoter_proximal_flag",
        "promoter_distal_flag",
        "canonical_splice_core_flag",
    ):
        if _subtype_col in dataset.schema.names and _subtype_col not in cols:
            cols.append(_subtype_col)
    batch_no = 0
    t0 = time.time()
    for rb in dataset.scanner(columns=cols, batch_size=args.batch_size).to_batches():
        df = rb.to_pandas()
        df = normalize_batch(df, args)
        df = df[df["chr_num"].astype(int).isin(chr_keep)]
        if len(df) == 0:
            continue
        scanned += len(df)
        batch_no += 1
        if batch_no % max(1, args.progress_every) == 0:
            logger.info(f"[count progress] chrs={sorted(chr_keep)} batch={batch_no} scanned={scanned:,} elapsed={time.time()-t0:.1f}s")
        df = prepare_batch(df, args, call_idx, rate_arr, ne)
        df = df.dropna(subset=["Sg_code", "Ctx_code", "Alt_code", "region_type", "p_base_eff_fold"])
        df = attach_bins_and_group(df, p_edges, cn_edges)
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


def write_selected_split(dataset, out_path: Path, chr_keep, split, fold, args, call_idx, rate_arr, ne, p_edges, cn_edges, p_thr, probs0, probs1):
    chr_keep = set(int(x) for x in chr_keep)
    writer = None
    stats = dict(scanned=0, written0=0, written1=0)
    cols = list(lean_scanner_cols(dataset, mode="write"))
    for _subtype_col in (
        "promoter_core_flag",
        "promoter_proximal_flag",
        "promoter_distal_flag",
        "canonical_splice_core_flag",
    ):
        if _subtype_col in dataset.schema.names and _subtype_col not in cols:
            cols.append(_subtype_col)
    batch_no = 0
    t0 = time.time()
    for rb in dataset.scanner(columns=cols, batch_size=args.batch_size).to_batches():
        df = rb.to_pandas()
        df = normalize_batch(df, args)
        df = df[df["chr_num"].astype(int).isin(chr_keep)]
        if len(df) == 0:
            continue
        stats["scanned"] += len(df)
        batch_no += 1
        if batch_no % max(1, args.progress_every) == 0:
            logger.info(f"[write progress] split={split} chrs={sorted(chr_keep)} batch={batch_no} scanned={stats['scanned']:,} written={stats['written0']+stats['written1']:,} elapsed={time.time()-t0:.1f}s")
        df = prepare_batch(df, args, call_idx, rate_arr, ne)
        df = df.dropna(subset=["Sg_code", "Ctx_code", "Alt_code", "region_type", "p_base_eff_fold"])
        if len(df) == 0:
            continue
        df = attach_bins_and_group(df, p_edges, cn_edges)
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
        writer.write_table(table)
        stats["written0"] += int((out["new_label"] == 0).sum())
        stats["written1"] += int((out["new_label"] == 1).sum())
    if writer is not None:
        writer.close()
    stats["written"] = stats["written0"] + stats["written1"]
    return stats


def count_groups_for_all_splits(dataset, split_defs, args, call_idx, ne, p_edges, cn_edges, p_thr):
    """Count matched groups for train/val/test in one dataset scan.

    This is result-equivalent to calling count_groups_for_split three times:
    - same chromosome membership per split
    - same split-specific rate array
    - same p_base_eff threshold
    - same matching group definition
    """
    meta = {}
    for split, chrs, rate_arr, _out_path in split_defs:
        meta[split] = dict(
            chrs=set(int(x) for x in chrs),
            rate_arr=rate_arr,
            count0=Counter(),
            count1=Counter(),
            cand0=0,
            cand1=0,
            scanned=0,
        )

    cols = list(lean_scanner_cols(dataset, mode="count"))
    for _subtype_col in (
        "promoter_core_flag",
        "promoter_proximal_flag",
        "promoter_distal_flag",
        "canonical_splice_core_flag",
    ):
        if _subtype_col in dataset.schema.names and _subtype_col not in cols:
            cols.append(_subtype_col)
    batch_no = 0
    t0 = time.time()

    for rb in dataset.scanner(columns=cols, batch_size=args.batch_size).to_batches():
        base = prepare_base_batch_fast(rb.to_pandas(), args, call_idx)
        if len(base) == 0:
            continue

        chr_series = base["chr_num"].astype(int)
        batch_no += 1

        if batch_no % max(1, int(args.progress_every)) == 0:
            total_scanned = sum(int(m["scanned"]) for m in meta.values())
            logger.info(
                f"[count-all progress] batch={batch_no} split_scanned={total_scanned:,} "
                f"elapsed={time.time() - t0:.1f}s"
            )

        for split, m in meta.items():
            mask = chr_series.isin(m["chrs"]).to_numpy()
            n_mask = int(mask.sum())
            if n_mask == 0:
                continue

            m["scanned"] += n_mask
            df = base.loc[mask].copy()
            df = attach_pmut(df, m["rate_arr"], ne, args.max_callable_n, args.fallback_mu)
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=["Sg_code", "Ctx_code", "Alt_code", "region_type", "p_base_eff_fold"])
            if len(df) == 0:
                continue

            df = attach_bins_and_group(df, p_edges, cn_edges)
            is0, is1 = candidate_masks(df, p_thr, args)

            if is0.any():
                vc = df.loc[is0, "match_group"].value_counts()
                m["count0"].update(vc.to_dict())
                m["cand0"] += int(is0.sum())

            if is1.any():
                vc = df.loc[is1, "match_group"].value_counts()
                m["count1"].update(vc.to_dict())
                m["cand1"] += int(is1.sum())

    return {
        split: (m["count0"], m["count1"], m["cand0"], m["cand1"], m["scanned"])
        for split, m in meta.items()
    }


def write_selected_all_splits(dataset, split_defs, fold, args, call_idx, ne, p_edges, cn_edges, p_thr, probs_by_split):
    """Write train/val/test selected rows in one dataset scan.

    This is result-equivalent to calling write_selected_split three times:
    - same split-specific chromosome filter
    - same split-specific P_mut rate array
    - same match_group probabilities
    - same stable_keep keys and seeds
    """
    meta = {}
    split_order = []
    for split, chrs, rate_arr, out_path in split_defs:
        split_order.append(split)
        meta[split] = dict(
            chrs=set(int(x) for x in chrs),
            rate_arr=rate_arr,
            out_path=Path(out_path),
            writer=None,
            stats=dict(scanned=0, written0=0, written1=0),
        )

    cols = list(lean_scanner_cols(dataset, mode="write"))
    for _subtype_col in (
        "promoter_core_flag",
        "promoter_proximal_flag",
        "promoter_distal_flag",
        "canonical_splice_core_flag",
    ):
        if _subtype_col in dataset.schema.names and _subtype_col not in cols:
            cols.append(_subtype_col)
    batch_no = 0
    t0 = time.time()

    try:
        for rb in dataset.scanner(columns=cols, batch_size=args.batch_size).to_batches():
            base = prepare_base_batch_fast(rb.to_pandas(), args, call_idx)
            if len(base) == 0:
                continue

            chr_series = base["chr_num"].astype(int)
            batch_no += 1

            if batch_no % max(1, int(args.progress_every)) == 0:
                total_written = sum(
                    int(m["stats"]["written0"]) + int(m["stats"]["written1"])
                    for m in meta.values()
                )
                total_scanned = sum(int(m["stats"]["scanned"]) for m in meta.values())
                logger.info(
                    f"[write-all progress] batch={batch_no} split_scanned={total_scanned:,} "
                    f"written={total_written:,} elapsed={time.time() - t0:.1f}s"
                )

            for split in split_order:
                m = meta[split]
                mask = chr_series.isin(m["chrs"]).to_numpy()
                n_mask = int(mask.sum())
                if n_mask == 0:
                    continue

                m["stats"]["scanned"] += n_mask
                df = base.loc[mask].copy()
                df = attach_pmut(df, m["rate_arr"], ne, args.max_callable_n, args.fallback_mu)
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.dropna(subset=["Sg_code", "Ctx_code", "Alt_code", "region_type", "p_base_eff_fold"])
                if len(df) == 0:
                    continue

                df = attach_bins_and_group(df, p_edges, cn_edges)
                is0, is1 = candidate_masks(df, p_thr, args)

                keep = np.zeros(len(df), dtype=bool)
                lab = np.full(len(df), -1, dtype=np.int8)
                probs0, probs1 = probs_by_split[split]

                idx0 = np.flatnonzero(is0.to_numpy())
                if len(idx0):
                    sub0 = df.iloc[idx0]
                    p0 = sub0["match_group"].map(probs0).fillna(0.0).to_numpy(np.float64)
                    k0 = stable_keep(sub0["snp_id"].to_numpy(), p0, args.seed + fold * 1000 + 11)
                    sel0 = idx0[k0]
                    keep[sel0] = True
                    lab[sel0] = 0

                idx1 = np.flatnonzero(is1.to_numpy())
                if len(idx1):
                    sub1 = df.iloc[idx1]
                    p1 = sub1["match_group"].map(probs1).fillna(0.0).to_numpy(np.float64)
                    k1 = stable_keep(sub1["snp_id"].to_numpy(), p1, args.seed + fold * 1000 + 101)
                    sel1 = idx1[k1]
                    keep[sel1] = True
                    lab[sel1] = 1

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
                if m["writer"] is None:
                    m["out_path"].parent.mkdir(parents=True, exist_ok=True)
                    m["writer"] = pq.ParquetWriter(str(m["out_path"]), table.schema, compression="snappy")

                m["writer"].write_table(table)
                m["stats"]["written0"] += int((out["new_label"] == 0).sum())
                m["stats"]["written1"] += int((out["new_label"] == 1).sum())

    finally:
        for m in meta.values():
            if m["writer"] is not None:
                m["writer"].close()

    out_stats = {}
    for split, m in meta.items():
        st = dict(m["stats"])
        st["written"] = int(st["written0"] + st["written1"])
        out_stats[split] = st
    return out_stats


def export_lm_targets(out_dir: Path):
    logger.info("Exporting unique noncoding DNA-LM targets ...")
    rows = []
    completed = collect_completed_fold_manifests(out_dir)
    completed_folds = {int(m["fold"]) for m in completed if "fold" in m}
    logger.info(f"LM target export sees {len(completed_folds)} completed folds: {sorted(completed_folds)}")
    for fold_dir in sorted(Path(out_dir).glob("fold*")):
        if not fold_dir.is_dir():
            continue
        manifest = load_fold_manifest_if_complete(fold_dir)
        if manifest is None:
            continue
        for name in SPLIT_PARQUET_NAMES:
            p = fold_dir / name
            pf = pq.ParquetFile(p)
            cols = [c for c in ["snp_id", "Chr", "Pos", "Ref", "Alt", "region_type", "GeneID"] if c in pf.schema.names]
            rows.append(pf.read(columns=cols).to_pandas())
    lm_dir = out_dir / "lm_targets"
    lm_dir.mkdir(exist_ok=True)
    out_path = lm_dir / "noncoding_dna_lm_targets.tsv.gz"
    if not rows:
        pd.DataFrame(columns=["snp_id", "Chr", "Pos", "Ref", "Alt"]).to_csv(out_path, sep="\t", index=False, compression="gzip")
        return str(out_path), 0
    df = pd.concat(rows, ignore_index=True).drop_duplicates("snp_id")
    df.to_csv(out_path, sep="\t", index=False, compression="gzip")
    logger.info(f"LM targets saved: {out_path}; n={len(df):,}")
    return str(out_path), int(len(df))


def _is_cotton_nested_rate_cache(genome_total):
    """
    Cotton calc_pmut_fast.py cache uses nested At/Dt structure:
      genome_total["At"][context]
      genome_total["Dt"][context]
      mut_total["At"][context][alt]
      mut_total["Dt"][context][alt]
    """
    if not isinstance(genome_total, dict):
        return False
    return (
        ("At" in genome_total and isinstance(genome_total.get("At"), dict)) or
        ("Dt" in genome_total and isinstance(genome_total.get("Dt"), dict))
    )


def _cotton_build_rate_table_excluding_nested(
    genome_counts_total,
    genome_counts_by_chrnum,
    mutation_counts_total,
    mutation_counts_by_chrnum,
    exclude_chrs,
    mu_base,
    pseudo_count,
):
    """
    Build At/Dt-aware P_mut table after excluding one or more chromosome numbers.

    Output keys:
      At_ACA_G
      Dt_TTG_A

    Formula matches cotton calc_pmut_fast.py:
      obs_rate = (mut_count + pseudo_count) / (context_count + pseudo_count * 3)
      relative_rate = obs_rate / avg_obs_rate
      P_mut = mu_base * relative_rate
    """
    from collections import Counter, defaultdict
    import copy

    groups = [g for g in ("At", "Dt") if g in genome_counts_total]
    if not groups:
        groups = list(genome_counts_total.keys())

    exclude_chrs = set(int(x) for x in exclude_chrs)

    cur_gen = {}
    cur_mut = {}

    for sg in groups:
        cur_gen[sg] = Counter(genome_counts_total.get(sg, {}))
        cur_mut[sg] = defaultdict(lambda: defaultdict(int))
        for ctx, alt_dict in mutation_counts_total.get(sg, {}).items():
            for alt, count in alt_dict.items():
                cur_mut[sg][ctx][alt] = int(count)

    for chr_num in exclude_chrs:
        by_gen = genome_counts_by_chrnum.get(chr_num, {})
        by_mut = mutation_counts_by_chrnum.get(chr_num, {})

        for sg in groups:
            for ctx, count in by_gen.get(sg, {}).items():
                cur_gen[sg][ctx] -= int(count)
                if cur_gen[sg][ctx] <= 0:
                    cur_gen[sg].pop(ctx, None)

            for ctx, alt_dict in by_mut.get(sg, {}).items():
                for alt, count in alt_dict.items():
                    cur_mut[sg][ctx][alt] -= int(count)
                    if cur_mut[sg][ctx][alt] <= 0:
                        cur_mut[sg][ctx].pop(alt, None)
                if not cur_mut[sg].get(ctx):
                    cur_mut[sg].pop(ctx, None)

    sum_gen = sum(sum(c.values()) for c in cur_gen.values())
    sum_mut = 0
    for sg in groups:
        for alt_dict in cur_mut[sg].values():
            sum_mut += sum(alt_dict.values())

    avg_obs_rate = sum_mut / sum_gen if sum_gen > 0 else 1e-10
    avg_obs_rate = max(avg_obs_rate, 1e-12)

    out = {}
    bases = ["A", "C", "G", "T"]

    for sg in groups:
        for context, denom in cur_gen[sg].items():
            context = str(context)
            if denom <= 0 or len(context) != 3:
                continue
            ref = context[1]
            if ref not in bases:
                continue

            for alt in bases:
                if alt == ref:
                    continue
                mut = cur_mut[sg].get(context, {}).get(alt, 0)
                obs_rate = (float(mut) + float(pseudo_count)) / (float(denom) + float(pseudo_count) * 3.0)
                relative_rate = obs_rate / avg_obs_rate
                out[f"{sg}_{context}_{alt}"] = float(float(mu_base) * relative_rate)

    return out


def build_rate_table_excluding(
    genome_total,
    genome_by_chrnum,
    mut_total,
    mut_by_chrnum,
    exclude_chrs,
    min_mu,
    pseudo_count,
):
    """
    Override original function.

    Supports both:
      1. flat internal cache: key -> count
      2. cotton nested At/Dt cache: genome_total["At"][ctx], mut_total["At"][ctx][alt]
    """
    if _is_cotton_nested_rate_cache(genome_total):
        return _cotton_build_rate_table_excluding_nested(
            genome_total,
            genome_by_chrnum,
            mut_total,
            mut_by_chrnum,
            exclude_chrs=exclude_chrs,
            mu_base=min_mu,
            pseudo_count=pseudo_count,
        )


    keys = set(genome_total) | set(mut_total)
    for c in exclude_chrs:
        keys |= set(genome_by_chrnum.get(c, {})) | set(mut_by_chrnum.get(c, {}))

    out = {}
    for k in keys:
        den = float(genome_total.get(k, 0)) - sum(float(genome_by_chrnum.get(c, {}).get(k, 0)) for c in exclude_chrs)
        mut = float(mut_total.get(k, 0)) - sum(float(mut_by_chrnum.get(c, {}).get(k, 0)) for c in exclude_chrs)
        out[k] = min_mu if den <= 0 else max((mut + pseudo_count) / (den + pseudo_count * 4.0), min_mu)
    return out


from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_functions
from phenofuncrank_repro.plantvimpact.native_upstream import noncoding_kernels as _source_kernels
_bind_functions(globals(), _source_kernels, ['alt_code', 'attach_bins_and_group', 'attach_callrate', 'attach_pmut', 'build_rate_array', 'build_train_rate_from_cache_loco', 'candidate_masks', 'collect_train_edges', 'ctx_code_from_seq', 'decide_sampling_probs', 'digitize', 'ensure_columns', 'get_rate_tables', 'import_calc_pmut_fast', 'infer_chr_num', 'infer_subgenome_code', 'internal_build_rate_tables', 'lean_scanner_cols', 'load_callrate_index', 'load_fasta_dict', 'make_edges', 'normalize_batch', 'open_parquet_dataset_only', 'open_text', 'parse_4d_sites', 'parse_folds', 'prepare_base_batch_fast', 'prepare_batch', 'stable_hash01', 'stable_keep', 'table_key_to_codes', 'val_chr_for_test'])


from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_identical
from phenofuncrank_repro.common import shared_cotton_label_entry as _profile_kernel_4
_bind_identical(globals(), _profile_kernel_4, ['main'])

if __name__ == "__main__":
    main()
