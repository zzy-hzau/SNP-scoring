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

def write_selected_split(dataset, out_path: Path, chr_keep, split, fold, args, call_idx, rate_arr, ne, p_edges, cn_edges, p_thr, probs0, probs1):
    chr_keep = set(int(x) for x in chr_keep)
    writer = None
    stats = dict(scanned=0, written0=0, written1=0)
    cols = lean_scanner_cols(dataset, mode="write")
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
