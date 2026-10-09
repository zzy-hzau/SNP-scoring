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

def build_rate_table_excluding_chrnums(
    genome_counts_total: dict,
    genome_counts_by_chrnum: dict,
    mutation_counts_total: dict,
    mutation_counts_by_chrnum: dict,
    exclude_chrs: Optional[Iterable[int]] = None,
    exclude_chr: Optional[int] = None,
    mu_base: float = MU_BASE_DEFAULT,
    pseudo_count: float = 0.5,
) -> dict:
    """Build Maize_<context>_<ALT> table after excluding held-out chromosome(s)."""
    if exclude_chrs is None:
        exclude_chrs = []
    exclude_chrs = set(int(x) for x in exclude_chrs)
    if exclude_chr is not None and int(exclude_chr) > 0:
        exclude_chrs.add(int(exclude_chr))

    cur_gen = {GROUP: Counter(genome_counts_total.get(GROUP, {}))}
    cur_mut = {GROUP: defaultdict(int)}
    for ctx, alt_dict in mutation_counts_total.get(GROUP, {}).items():
        cur_mut[GROUP][ctx] = defaultdict(int, dict(alt_dict))

    for chr_num in exclude_chrs:
        if chr_num in genome_counts_by_chrnum:
            _subtract_counter(cur_gen[GROUP], genome_counts_by_chrnum[chr_num].get(GROUP, Counter()))
        if chr_num in mutation_counts_by_chrnum:
            _subtract_mut(cur_mut[GROUP], mutation_counts_by_chrnum[chr_num].get(GROUP, {}))

    sum_gen = sum(cur_gen[GROUP].values())
    sum_mut = sum(sum(alt_dict.values()) for alt_dict in cur_mut[GROUP].values())
    avg_obs_rate = sum_mut / sum_gen if sum_gen > 0 else 1e-10
    avg_obs_rate = max(avg_obs_rate, 1e-12)

    rate_table = {}
    for context, count in cur_gen[GROUP].items():
        if count <= 0 or len(context) != 3:
            continue
        ref_base = context[1]
        for alt in BASES:
            if alt == ref_base:
                continue
            mut_count = cur_mut[GROUP].get(context, {}).get(alt, 0)
            obs_rate = (mut_count + pseudo_count) / (count + pseudo_count * 3.0)
            relative_rate = obs_rate / avg_obs_rate
            rate_table[f"{GROUP}_{context}_{alt}"] = float(mu_base * relative_rate)
    return rate_table
