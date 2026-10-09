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


def import_calc_pmut_fast():
    try:
        from phenofuncrank_repro.plantvimpact.labels_loco.pmut import build_all_fold_rate_tables_fast, build_rate_table_excluding_chrnums
        return build_all_fold_rate_tables_fast, build_rate_table_excluding_chrnums
    except Exception:
        return None, None


from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_profile
from phenofuncrank_repro.plantvimpact.native_upstream import maize_noncoding_labels as _native_profile
_bind_profile(globals(), _native_profile, ['alt_code', 'attach_bins_and_group', 'attach_callrate', 'attach_pmut', 'attach_region_subtype', 'build_rate_array', 'build_rate_table_excluding', 'build_train_rate_from_cache_loco', 'calibrate_ne_for_train_chrs', 'candidate_masks', 'collect_train_edges', 'count_groups_for_all_splits', 'count_groups_for_split', 'ctx_code_from_seq', 'decide_sampling_probs', 'digitize', 'ensure_columns', 'export_lm_targets', 'get_rate_tables', 'infer_chr_num', 'infer_subgenome_code', 'internal_build_rate_tables', 'lean_scanner_cols', 'load_callrate_index', 'load_fasta_dict', 'make_edges', 'normalize_batch', 'open_parquet_dataset_only', 'open_text', 'parse_4d_sites', 'parse_folds', 'prepare_base_batch_fast', 'prepare_batch', 'stable_hash01', 'stable_keep', 'table_key_to_codes', 'val_chr_for_test', 'write_selected_all_splits', 'write_selected_split'])


from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_identical
from phenofuncrank_repro.common import shared_diploid_label_entry as _profile_kernel_8
_bind_identical(globals(), _profile_kernel_8, ['main'])

if __name__ == "__main__":
    main()
