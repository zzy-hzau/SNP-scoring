"""Identical source function; caller module supplies its recorded policy bindings."""
from __future__ import annotations
import argparse
import gzip
import json
import logging
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, Iterable, Tuple
import numpy as np
import pandas as pd
logger = logging.getLogger("calc_pmut_fast")
GROUP = "Maize"
BASES = ("A", "C", "G", "T")
BASE2INT = {"A": 0, "C": 1, "G": 2, "T": 3}
MU_BASE_DEFAULT = 4.56e-9

def load_4d_counts_total_and_by_chrnum(sites_4d_path: str, genome_dict: Dict[str, str], n_chr: int = 10):
    """
    Count 3-mer backgrounds at 4D sites.
    Return:
      total_counts[GROUP][context] -> count
      by_chrnum[chr_num][GROUP][context] -> count
    """
    df = pd.read_csv(
        sites_4d_path,
        sep="\t",
        usecols=["Chr", "Start_0based", "RefBase"],
        dtype={"Chr": "category", "Start_0based": "int64", "RefBase": "category"},
    )
    df["Chr"] = df["Chr"].astype(str)
    df = df[df["Chr"].isin(genome_dict.keys())].copy()
    df = df[df["Chr"].map(lambda x: is_main_chr(x, n_chr))]
    df = df[df["Start_0based"] >= 1]

    total_counts = {GROUP: Counter()}
    by_chrnum = defaultdict(lambda: {GROUP: Counter()})

    for chrom, grp in df.groupby("Chr", observed=True):
        chr_num = get_chr_num(str(chrom))
        if chr_num < 1:
            continue
        seq = genome_dict[str(chrom)]
        seq_len = len(seq)
        pos_arr = grp["Start_0based"].to_numpy(dtype=np.int64)
        ref_arr = grp["RefBase"].astype(str).str.upper().to_numpy()

        valid = (pos_arr >= 1) & (pos_arr + 1 < seq_len)
        pos_arr = pos_arr[valid]
        ref_arr = ref_arr[valid]
        if len(pos_arr) == 0:
            continue

        seq_bytes = np.frombuffer(seq.encode("ascii"), dtype="S1")
        prev_b = seq_bytes[pos_arr - 1]
        curr_b = seq_bytes[pos_arr]
        next_b = seq_bytes[pos_arr + 1]
        ref_b = np.array([r.encode("ascii") for r in ref_arr], dtype="S1")

        n_byte = b"N"
        ok = (prev_b != n_byte) & (curr_b != n_byte) & (next_b != n_byte) & (curr_b == ref_b)
        if not np.any(ok):
            continue

        contexts = np.char.add(np.char.add(prev_b[ok], curr_b[ok]), next_b[ok]).astype(str)
        ctx_list = contexts.tolist()
        total_counts[GROUP].update(ctx_list)
        by_chrnum[chr_num][GROUP].update(ctx_list)

    logger.info(f"4D context count complete: {sum(total_counts[GROUP].values()):,} sites, {len(total_counts[GROUP])} contexts")
    return total_counts, dict(by_chrnum)
