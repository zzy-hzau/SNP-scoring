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

    cols = lean_scanner_cols(dataset, mode="count")
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
