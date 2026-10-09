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

def count_groups_for_split(dataset, chr_keep, args, call_idx, rate_arr, ne, p_edges, cn_edges, p_thr):
    chr_keep = set(int(x) for x in chr_keep)
    count0, count1 = Counter(), Counter()
    cand0 = cand1 = scanned = 0
    cols = lean_scanner_cols(dataset, mode="count")
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
