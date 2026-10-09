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

    cols = lean_scanner_cols(dataset, mode="write")
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
