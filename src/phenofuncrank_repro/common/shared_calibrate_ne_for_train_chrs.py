"""Identical source function; caller module supplies its recorded policy bindings."""
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
logger = logging.getLogger("noncoding_proxy_labels_preLM_strict")
BASE2INT = {"A": 0, "C": 1, "G": 2, "T": 3}
REQUIRED_BASE_COLS = [
    "snp_id", "Chr", "Pos", "Ref", "Alt",
    "region_type", "Y_Label", "chr_num", "Sg_code", "Ctx_code", "Alt_code",
]
OPTIONAL_KEEP_COLS = [
    "region_subtype",
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
from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_functions

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

    cols = lean_scanner_cols(dataset, mode="count")
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
