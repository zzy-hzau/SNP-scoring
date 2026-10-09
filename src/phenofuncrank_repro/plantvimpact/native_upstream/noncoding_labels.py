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


def export_lm_targets(out_dir: Path):
    logger.info("Exporting unique noncoding DNA-LM targets ...")
    rows = []
    for p in sorted(out_dir.glob("fold*/**/*.parquet")):
        if p.name not in {"train.parquet", "val.parquet", "test.parquet"}:
            continue
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


from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_functions
from phenofuncrank_repro.plantvimpact.native_upstream import noncoding_kernels as _source_kernels
_bind_functions(globals(), _source_kernels, ['alt_code', 'attach_bins_and_group', 'attach_callrate', 'attach_pmut', 'build_rate_array', 'build_train_rate_from_cache_loco', 'candidate_masks', 'collect_train_edges', 'ctx_code_from_seq', 'decide_sampling_probs', 'digitize', 'ensure_columns', 'get_rate_tables', 'import_calc_pmut_fast', 'infer_chr_num', 'infer_subgenome_code', 'internal_build_rate_tables', 'lean_scanner_cols', 'load_callrate_index', 'load_fasta_dict', 'make_edges', 'normalize_batch', 'open_parquet_dataset_only', 'open_text', 'parse_4d_sites', 'parse_folds', 'prepare_base_batch_fast', 'prepare_batch', 'stable_hash01', 'stable_keep', 'table_key_to_codes', 'val_chr_for_test'])


from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_identical
from phenofuncrank_repro.common import shared_diploid_label_entry as _profile_kernel_8
_bind_identical(globals(), _profile_kernel_8, ['main'])
from phenofuncrank_repro.common import shared_calibrate_ne_for_train_chrs as _profile_kernel_10
_bind_identical(globals(), _profile_kernel_10, ['calibrate_ne_for_train_chrs'])
from phenofuncrank_repro.common import shared_count_groups_for_split as _profile_kernel_11
_bind_identical(globals(), _profile_kernel_11, ['count_groups_for_split'])
from phenofuncrank_repro.common import shared_write_selected_split as _profile_kernel_12
_bind_identical(globals(), _profile_kernel_12, ['write_selected_split'])
from phenofuncrank_repro.common import shared_count_groups_for_all_splits as _profile_kernel_13
_bind_identical(globals(), _profile_kernel_13, ['count_groups_for_all_splits'])
from phenofuncrank_repro.common import shared_write_selected_all_splits as _profile_kernel_14
_bind_identical(globals(), _profile_kernel_14, ['write_selected_all_splits'])

if __name__ == "__main__":
    main()
