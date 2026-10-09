"Common source computations; profile dispatch lives in leaf modules."
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

def alt_code(alt: str) -> int:
    return BASE2INT.get(str(alt).upper(), -1)

def attach_bins_and_group(df: pd.DataFrame, p_edges: np.ndarray, cn_edges: np.ndarray) -> pd.DataFrame:
    df["p_base_eff_bin"] = digitize(df["p_base_eff_fold"].to_numpy(np.float64), p_edges).astype(np.int16)
    df["callable_n_bin"] = digitize(df["callable_n"].to_numpy(np.float64), cn_edges).astype(np.int16)
    df["match_group"] = (
        df["region_type"].astype(str) + "|" +
        df["region_subtype"].astype(str) + "|" +
        df["Sg_code"].astype(str) + "|" +
        df["Ctx_code"].astype(str) + "|" +
        df["Alt_code"].astype(str) + "|" +
        df["p_base_eff_bin"].astype(str) + "|" +
        df["callable_n_bin"].astype(str)
    )
    return df

def attach_callrate(df: pd.DataFrame, call_idx: dict) -> pd.DataFrame:

    
    if call_idx is None:
        if "callable_n" not in df.columns:
            raise RuntimeError("--trust-input-callrate requires callable_n in input universe.")
        if "AN_obs" not in df.columns:
            df["AN_obs"] = pd.to_numeric(df["callable_n"], errors="coerce").fillna(0).astype(np.float32) * 2.0
        if "callrate_hit" not in df.columns:
            df["callrate_hit"] = (pd.to_numeric(df["callable_n"], errors="coerce").fillna(0) > 0).astype(np.int8)
        df["AN_obs"] = pd.to_numeric(df["AN_obs"], errors="coerce").fillna(0).astype(np.float32)
        df["callable_n"] = pd.to_numeric(df["callable_n"], errors="coerce").fillna(0).astype(np.float32)
        df["callrate_hit"] = pd.to_numeric(df["callrate_hit"], errors="coerce").fillna(0).astype(np.int8)
        return df

    n = len(df)
    an = np.zeros(n, dtype=np.int32)
    cn = np.zeros(n, dtype=np.int32)
    hit = np.zeros(n, dtype=bool)
    chr_arr = df["Chr"].astype(str).to_numpy()
    pos_arr = df["Pos"].to_numpy(np.int64, copy=False)
    for chrom in pd.unique(chr_arr):
        tup = call_idx.get(str(chrom))
        if tup is None:
            continue
        pos_sorted, an_sorted, cn_sorted = tup
        loc = np.flatnonzero(chr_arr == str(chrom))
        q = pos_arr[loc]
        j = np.searchsorted(pos_sorted, q)
        ok = j < len(pos_sorted)
        eq = np.zeros(len(q), dtype=bool)
        eq[ok] = pos_sorted[j[ok]] == q[ok]
        hit[loc] = eq
        if np.any(eq):
            an[loc[eq]] = an_sorted[j[eq]]
            cn[loc[eq]] = cn_sorted[j[eq]]
    df["AN_obs"] = an
    df["callable_n"] = cn
    df["callrate_hit"] = hit.astype(np.int8)
    return df

def attach_pmut(df: pd.DataFrame, rate_arr: np.ndarray, ne: float, max_callable_n: int, fallback_mu: float) -> pd.DataFrame:
    sg = pd.to_numeric(df["Sg_code"], errors="coerce").fillna(-1).to_numpy(np.int16)
    ctx = pd.to_numeric(df["Ctx_code"], errors="coerce").fillna(-1).to_numpy(np.int16)
    alt = pd.to_numeric(df["Alt_code"], errors="coerce").fillna(-1).to_numpy(np.int16)
    mu = np.full(len(df), fallback_mu, dtype=np.float64)
    ok = (sg >= 0) & (sg < rate_arr.shape[0]) & (ctx >= 0) & (ctx < 64) & (alt >= 0) & (alt < 4)
    if np.any(ok):
        mu[ok] = rate_arr[sg[ok], ctx[ok], alt[ok]]
    cn = pd.to_numeric(df["callable_n"], errors="coerce").fillna(0).to_numpy(np.float64)
    exposure = np.clip(cn / float(max(max_callable_n, 1)), 0.0, 1.0)
    df["P_mut_fold"] = mu
    df["p_base_eff_fold"] = -np.expm1(-float(ne) * mu * exposure)
    return df

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

def build_train_rate_from_cache_loco(rate_cd: dict, exclude_chrs, args) -> dict:
    """Build LOCO train-rate table directly from the loaded rate cache.

    This avoids importing an old cotton calc_pmut_fast.py that expects At/Dt.
    Supports:
      1) maize nested cache:
           genome_total["Maize"][context]
           genome_by_chrnum[chr]["Maize"][context]
           mut_total["Maize"][context][alt]
           mut_by_chrnum[chr]["Maize"][context][alt]
      2) internal flat cache:
           key = "sg_ctx_alt"
    """
    genome_total = rate_cd["genome_total"]
    genome_by_chrnum = rate_cd["genome_by_chrnum"]
    mut_total = rate_cd["mut_total"]
    mut_by_chrnum = rate_cd["mut_by_chrnum"]

    exclude_chrs = set(int(x) for x in exclude_chrs)


    if isinstance(genome_total, dict) and "Maize" in genome_total:
        group = "Maize"

        cur_gen = Counter(dict(genome_total.get(group, {})))

        cur_mut = defaultdict(lambda: defaultdict(int))
        for ctx, alt_dict in dict(mut_total.get(group, {})).items():
            for alt, v in dict(alt_dict).items():
                cur_mut[ctx][alt] = int(v)

        for chr_num in exclude_chrs:
            gd = genome_by_chrnum.get(chr_num, {})
            if group in gd:
                for ctx, v in dict(gd[group]).items():
                    cur_gen[ctx] -= int(v)
                    if cur_gen[ctx] <= 0:
                        cur_gen.pop(ctx, None)

            md = mut_by_chrnum.get(chr_num, {})
            if group in md:
                for ctx, alt_dict in dict(md[group]).items():
                    for alt, v in dict(alt_dict).items():
                        cur_mut[ctx][alt] -= int(v)
                        if cur_mut[ctx][alt] <= 0:
                            cur_mut[ctx].pop(alt, None)
                    if not cur_mut[ctx]:
                        cur_mut.pop(ctx, None)

        sum_gen = float(sum(cur_gen.values()))
        sum_mut = float(sum(sum(dict(alt_dict).values()) for alt_dict in cur_mut.values()))
        avg_obs_rate = max(sum_mut / sum_gen if sum_gen > 0 else 1e-12, 1e-12)

        out = {}
        bases = ("A", "C", "G", "T")
        mu_base = float(args.fallback_mu)
        pc = float(args.pmut_pseudo_count)

        for ctx, denom in cur_gen.items():
            if denom <= 0 or len(str(ctx)) != 3 or any(b not in BASE2INT for b in str(ctx)):
                continue
            ref = str(ctx)[1]
            for alt in bases:
                if alt == ref:
                    continue
                mut = int(cur_mut.get(ctx, {}).get(alt, 0))
                obs_rate = (mut + pc) / (float(denom) + pc * 3.0)
                relative_rate = obs_rate / avg_obs_rate
                out[f"Maize_{ctx}_{alt}"] = float(mu_base * relative_rate)

        logger.info(
            f"Built Maize LOCO train_rate directly from cache: "
            f"exclude_chrs={sorted(exclude_chrs)}, keys={len(out)}, "
            f"sum_gen={sum_gen:.0f}, sum_mut={sum_mut:.0f}"
        )
        return out


    return build_rate_table_excluding(
        genome_total, genome_by_chrnum, mut_total, mut_by_chrnum,
        exclude_chrs=exclude_chrs,
        min_mu=args.fallback_mu,
        pseudo_count=args.pmut_pseudo_count,
    )

def candidate_masks(df: pd.DataFrame, p_threshold: float, args):
    y = pd.to_numeric(df["Y_Label"], errors="coerce").fillna(0).astype(np.int8)
    cn = pd.to_numeric(df["callable_n"], errors="coerce").fillna(0).astype(np.int32)
    p = pd.to_numeric(df["p_base_eff_fold"], errors="coerce").fillna(-np.inf).astype(np.float64)
    common = (df["callrate_hit"].astype(np.int8) == 1) & (cn >= args.min_callable_n)
    is0 = common & (y == 1)
    is1 = common & (y == 0) & (p >= p_threshold)
    return is0, is1

def collect_train_edges(dataset, train_chrs, args, call_idx, rate_arr, ne):
    logger.info("Collecting train-only p_base_eff/callable_n edges ...")
    train_chrs = set(int(x) for x in train_chrs)
    pvals, cnvals = [], []
    got = 0
    rng = np.random.default_rng(args.seed)
    cols = lean_scanner_cols(dataset, mode="edges")
    batch_no = 0
    t0 = time.time()
    for rb in dataset.scanner(columns=cols, batch_size=args.batch_size).to_batches():
        df = rb.to_pandas()
        df = normalize_batch(df, args)
        df = df[df["chr_num"].astype(int).isin(train_chrs)]
        if len(df) == 0:
            continue
        df = prepare_batch(df, args, call_idx, rate_arr, ne)
        df = df[(df["callrate_hit"] == 1) & (df["callable_n"] >= args.min_callable_n)]
        if len(df) == 0:
            continue
        sub = df.loc[rng.random(len(df)) < args.edge_sample_fraction]
        if len(sub):
            pvals.append(sub["p_base_eff_fold"].to_numpy(np.float64))
            cnvals.append(sub["callable_n"].to_numpy(np.float64))
            got += len(sub)
        if got >= args.sample_rows_for_bins:
            break
    if not pvals:
        raise RuntimeError("No train values collected for noncoding edges. Check input/callrate filters.")
    p = np.concatenate(pvals)[:args.sample_rows_for_bins]
    cn = np.concatenate(cnvals)[:args.sample_rows_for_bins]
    p_edges = make_edges(p, args.opportunity_bins)
    cn_edges = make_edges(cn, args.callable_n_bins)
    p_thr = float(np.quantile(p[np.isfinite(p)], args.opportunity_min_quantile))
    logger.info(f"p_base_eff threshold q={args.opportunity_min_quantile}: {p_thr:.6e}")
    return p_edges, cn_edges, p_thr

def ctx_code_from_seq(seq: str) -> int:
    if len(seq) != 3:
        return -1
    vals = [BASE2INT.get(b, -1) for b in seq.upper()]
    if min(vals) < 0:
        return -1
    return vals[0] * 16 + vals[1] * 4 + vals[2]

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
        p0, p1 = t0 / n0, t1 / n1
        probs0[g], probs1[g] = p0, p1
        rows.append(dict(match_group=g, count0=n0, count1=n1, target0=t0, target1=t1, prob0=p0, prob1=p1))
    return probs0, probs1, pd.DataFrame(rows)

def digitize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, values, side="right") - 1

def ensure_columns(dataset: ds.Dataset, required: List[str]):
    missing = [c for c in required if c not in dataset.schema.names]
    if missing:
        raise ValueError(f"Input noncoding universe missing required columns: {missing}")

def get_rate_tables(args):
    cache = Path(args.rate_cache) if args.rate_cache else Path(args.out_dir) / "rate_tables_cache.joblib"
    if cache.exists():
        if joblib is None:
            raise RuntimeError("joblib required to load rate cache")
        logger.info(f"Loading P_mut rate cache: {cache}")
        return joblib.load(cache), "cache"
    build_all, _ = import_calc_pmut_fast()
    if build_all is not None and not args.force_internal_pmut:
        logger.info("Building P_mut rate tables via calc_pmut_fast.py ...")
        rate_tables, genome_total, genome_by_chrnum, mut_total, mut_by_chrnum = build_all(
            fasta_path=args.fasta, vcf_path=args.wild_vcf, sites_4d_path=args.sites_4d, n_chr=args.n_chr
        )
        cd = dict(rate_tables=rate_tables, genome_total=genome_total, genome_by_chrnum=genome_by_chrnum,
                  mut_total=mut_total, mut_by_chrnum=mut_by_chrnum)
    else:
        logger.warning("calc_pmut_fast.py unavailable or --force-internal-pmut set; using internal slower fallback.")
        genome_total, genome_by_chrnum, mut_total, mut_by_chrnum = internal_build_rate_tables(
            args.fasta, args.wild_vcf, args.sites_4d, args
        )
        rate_tables = {}
        for fold in range(1, args.n_chr + 1):
            rate_tables[fold] = build_rate_table_excluding(
                genome_total, genome_by_chrnum, mut_total, mut_by_chrnum,
                exclude_chrs=[fold], min_mu=args.fallback_mu, pseudo_count=args.pmut_pseudo_count,
            )
        cd = dict(rate_tables=rate_tables, genome_total=genome_total, genome_by_chrnum=genome_by_chrnum,
                  mut_total=mut_total, mut_by_chrnum=mut_by_chrnum)
    if joblib is not None:
        joblib.dump(cd, cache, compress=3)
        logger.info(f"Saved P_mut rate cache: {cache}")
    return cd, "calc_pmut_fast" if build_all is not None and not args.force_internal_pmut else "internal"

def import_calc_pmut_fast():
    try:
        from calc_pmut_fast import build_all_fold_rate_tables_fast, build_rate_table_excluding_chrnums
        return build_all_fold_rate_tables_fast, build_rate_table_excluding_chrnums
    except Exception:
        return None, None

def infer_chr_num(chrom: str, n_chr: int) -> int:
    s = str(chrom)
    m = re.search(r"(\d+)$", s) or re.search(r"(\d+)", s)
    if not m:
        return -1
    x = int(m.group(1))
    return x if 1 <= x <= n_chr else ((x - 1) % n_chr) + 1

def infer_subgenome_code(chrom: str, mode: str = "auto") -> int:
    if mode == "single":
        return 0
    s = str(chrom).upper()
    if re.search(r"A\d+", s):
        return 0
    if re.search(r"D\d+", s):
        return 1
    return 0

def internal_build_rate_tables(fasta_path: str, vcf_path: str, sites_4d_path: str, args):
    genome = load_fasta_dict(fasta_path)
    sites, denom_total, denom_by_chr = parse_4d_sites(sites_4d_path, genome, args)
    mut_total = Counter()
    mut_by_chr = defaultdict(Counter)
    logger.info("Counting 4D SNVs in wild VCF for internal P_mut fallback ...")
    with open_text(vcf_path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            chrom, pos_s, _id, ref, alts = parts[:5]
            ref = ref.upper()
            if len(ref) != 1 or ref not in BASE2INT:
                continue
            rec = sites.get((chrom, int(pos_s)))
            if rec is None:
                continue
            sg, cc, ref4, chrn = rec
            if ref != ref4:
                continue
            for alt in alts.split(","):
                alt = alt.upper()
                if len(alt) != 1 or alt not in BASE2INT or alt == ref4:
                    continue
                key = f"{sg}_{cc}_{BASE2INT[alt]}"
                mut_total[key] += 1
                mut_by_chr[chrn][key] += 1
    return denom_total, dict(denom_by_chr), mut_total, dict(mut_by_chr)

def lean_scanner_cols(dataset, mode: str = "count"):
    """Read only columns required for each stage.

    mode='calibrate'/'edges'/'count' avoids heavy metadata columns.
    mode='write' keeps optional metadata for final parquet output.
    """
    base = [
        "snp_id", "Chr", "Pos", "Ref", "Alt",
        "region_type", "region_subtype",
        "Y_Label", "observed_in_vcf",
        "chr_num", "Sg_code", "Ctx_code", "Alt_code",
        "AN_obs", "callable_n", "callrate_hit",
    ]
    if mode == "write":
        base = list(dict.fromkeys(base + OPTIONAL_KEEP_COLS))
    cols = set(dataset.schema.names)
    return [c for c in dict.fromkeys(base) if c in cols]

def load_callrate_index(path: str, batch_size: int = 2_000_000) -> dict:
    logger.info(f"Loading callrate index: {path}")
    dataset = ds.dataset(path, format="parquet")
    cols = dataset.schema.names
    if not {"Chr", "Pos"}.issubset(cols):
        raise ValueError("Callrate parquet must contain Chr and Pos")
    an_col = "AN_obs" if "AN_obs" in cols else None
    cn_col = "callable_n" if "callable_n" in cols else None
    if cn_col is None:
        raise ValueError("Noncoding callrate must contain callable_n")
    read_cols = ["Chr", "Pos", cn_col] + ([an_col] if an_col else [])
    buckets = defaultdict(lambda: defaultdict(list))
    n = 0
    for rb in dataset.scanner(columns=read_cols, batch_size=batch_size).to_batches():
        df = rb.to_pandas()
        df["Chr"] = df["Chr"].astype(str)
        df["Pos"] = pd.to_numeric(df["Pos"], errors="coerce").fillna(-1).astype(np.int64)
        df["callable_n"] = pd.to_numeric(df[cn_col], errors="coerce").fillna(0).astype(np.int32)
        if an_col:
            df["AN_obs"] = pd.to_numeric(df[an_col], errors="coerce").fillna(0).astype(np.int32)
        else:
            df["AN_obs"] = df["callable_n"].astype(np.int32)
        for chrom, sub in df.groupby("Chr", sort=False):
            buckets[chrom]["Pos"].append(sub["Pos"].to_numpy(np.int64, copy=False))
            buckets[chrom]["AN_obs"].append(sub["AN_obs"].to_numpy(np.int32, copy=False))
            buckets[chrom]["callable_n"].append(sub["callable_n"].to_numpy(np.int32, copy=False))
        n += len(df)
        if n % 20_000_000 < batch_size:
            logger.info(f"  loaded callrate rows: {n:,}")
    out = {}
    for chrom, d in buckets.items():
        pos = np.concatenate(d["Pos"])
        an = np.concatenate(d["AN_obs"])
        cn = np.concatenate(d["callable_n"])
        order = np.argsort(pos, kind="mergesort")
        pos, an, cn = pos[order], an[order], cn[order]
        keep = np.ones(len(pos), dtype=bool)
        if len(pos) > 1:
            keep[1:] = pos[1:] != pos[:-1]
        out[chrom] = (pos[keep], an[keep], cn[keep])
    logger.info(f"Callrate index loaded: chroms={len(out)}, positions={sum(len(v[0]) for v in out.values()):,}")
    return out

def load_fasta_dict(path: str) -> Dict[str, str]:
    logger.info(f"Loading FASTA for internal P_mut fallback: {path}")
    genome = {}
    with open_text(path) as f:
        name = None
        buf = []
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    genome[name] = "".join(buf).upper()
                name = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if name is not None:
            genome[name] = "".join(buf).upper()
    return genome

def make_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise RuntimeError("Cannot make bin edges from empty values.")
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        mn, mx = float(np.min(values)), float(np.max(values))
        if mn == mx:
            edges = np.array([mn - 1e-12, mx + 1e-12], dtype=np.float64)
        else:
            edges = np.linspace(mn, mx, min(n_bins + 1, 6))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges

def normalize_batch(df: pd.DataFrame, args) -> pd.DataFrame:
    df = df.copy()
    df["snp_id"] = df["snp_id"].astype(str)
    df["Chr"] = df["Chr"].astype(str)
    df["Pos"] = pd.to_numeric(df["Pos"], errors="coerce").astype(np.int64)
    df["Ref"] = df["Ref"].astype(str).str.upper().str[:1]
    df["Alt"] = df["Alt"].astype(str).str.upper().str[:1]
    df["region_type"] = df["region_type"].astype(str)
    df["Y_Label"] = pd.to_numeric(df["Y_Label"], errors="coerce").fillna(0).astype(np.int8)
    if "chr_num" not in df.columns or df["chr_num"].isna().any():
        df["chr_num"] = df["Chr"].map(lambda x: infer_chr_num(x, args.n_chr)).astype(np.int16)
    else:
        df["chr_num"] = pd.to_numeric(df["chr_num"], errors="coerce").fillna(-1).astype(np.int16)
    if "Sg_code" not in df.columns or df["Sg_code"].isna().any():
        df["Sg_code"] = df["Chr"].map(lambda x: infer_subgenome_code(x, args.subgenome_mode)).astype(np.int16)
    else:
        df["Sg_code"] = pd.to_numeric(df["Sg_code"], errors="coerce").fillna(-1).astype(np.int16)
    if "Alt_code" not in df.columns or df["Alt_code"].isna().any():
        df["Alt_code"] = df["Alt"].map(alt_code).astype(np.int16)
    else:
        df["Alt_code"] = pd.to_numeric(df["Alt_code"], errors="coerce").fillna(-1).astype(np.int16)
    df["Ctx_code"] = pd.to_numeric(df["Ctx_code"], errors="coerce").fillna(-1).astype(np.int16)
    for c in OPTIONAL_KEEP_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df

def open_parquet_dataset_only(path):
    """Open a parquet dataset while ignoring manifest.json, _SUCCESS, and other metadata files."""
    from pathlib import Path
    pp = Path(path)
    if pp.is_dir():
        files = sorted(str(x) for x in pp.glob("*.parquet"))
        if not files:
            raise RuntimeError(f"No *.parquet files found in dataset directory: {pp}")
        logger.info(f"Opening parquet dataset from {len(files)} parquet files under: {pp}")
        return ds.dataset(files, format="parquet")
    logger.info(f"Opening parquet dataset from file: {pp}")
    return ds.dataset(str(pp), format="parquet")

def open_text(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")

def parse_4d_sites(path: str, genome: Dict[str, str], args):
    sites = {}
    denom_total = Counter()
    denom_by_chr = defaultdict(Counter)
    with open_text(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        h = {c: i for i, c in enumerate(header)}
        chr_col = h.get("Chr", 0)
        end_col = h.get("End_1based", h.get("End", 2))
        ref_col = h.get("RefBase", 5)
        sg_col = h.get("SubGenome", None)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) <= max(chr_col, end_col, ref_col):
                continue
            chrom = p[chr_col]
            pos = int(p[end_col])
            ref = p[ref_col].upper()
            if chrom not in genome or pos <= 1 or pos >= len(genome[chrom]) or ref not in BASE2INT:
                continue
            ctx = genome[chrom][pos - 2:pos + 1].upper()
            if len(ctx) != 3 or any(b not in BASE2INT for b in ctx) or genome[chrom][pos - 1].upper() != ref:
                continue
            chrn = infer_chr_num(chrom, args.n_chr)
            if sg_col is not None and len(p) > sg_col:
                sg_raw = p[sg_col]
                sg = 0 if sg_raw in ("At", "A", "0", "G0") else 1 if sg_raw in ("Dt", "D", "1") else infer_subgenome_code(chrom, args.subgenome_mode)
            else:
                sg = infer_subgenome_code(chrom, args.subgenome_mode)
            cc = ctx_code_from_seq(ctx)
            sites[(chrom, pos)] = (sg, cc, ref, chrn)
            for alt in "ACGT":
                if alt == ref:
                    continue
                key = f"{sg}_{cc}_{BASE2INT[alt]}"
                denom_total[key] += 1
                denom_by_chr[chrn][key] += 1
    logger.info(f"4D sites parsed: {len(sites):,}; denominator keys={len(denom_total)}")
    return sites, denom_total, denom_by_chr

def parse_folds(spec: str) -> List[int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]

def prepare_base_batch_fast(df: pd.DataFrame, args, call_idx) -> pd.DataFrame:
    df = normalize_batch(df, args)
    df = attach_callrate(df, call_idx)
    df = attach_region_subtype(df)
    return df.replace([np.inf, -np.inf], np.nan)

def prepare_batch(df: pd.DataFrame, args, call_idx, rate_arr, ne):
    df = normalize_batch(df, args)
    df = attach_callrate(df, call_idx)
    df = attach_pmut(df, rate_arr, ne, args.max_callable_n, args.fallback_mu)
    df = attach_region_subtype(df)
    return df.replace([np.inf, -np.inf], np.nan)

def stable_hash01(x: str, seed: int = 17) -> float:
    return (zlib.crc32(f"{seed}|{x}".encode("utf-8")) & 0xFFFFFFFF) / 2**32

def stable_keep(keys: Sequence[str], probs: np.ndarray, seed: int) -> np.ndarray:
    if len(keys) == 0:
        return np.zeros(0, dtype=bool)
    vals = np.fromiter((stable_hash01(str(k), seed) for k in keys), dtype=np.float64, count=len(keys))
    return vals < probs

def table_key_to_codes(k: str) -> Optional[Tuple[int, int, int]]:
    parts = str(k).split("_")
    if len(parts) != 3:
        return None
    sg, ctx, alt = parts


    if sg in ("At", "A", "Maize", "M", "Single", "S", "G0", "0"):
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

def val_chr_for_test(test_chr: int, n_chr: int) -> int:
    return test_chr + 1 if test_chr < n_chr else 1
