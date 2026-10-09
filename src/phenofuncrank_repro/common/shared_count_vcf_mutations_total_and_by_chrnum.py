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

def count_vcf_mutations_total_and_by_chrnum(vcf_path: str, genome_dict: Dict[str, str], sites_4d_set: set, n_chr: int = 10):
    """
    Count observed SNP mutations at 4D sites, grouped by context+ALT.
    Multi-ALT records are split. Non-SNP and '*' alleles are skipped.
    """
    total_mut = _group_to_ctx()
    by_chrnum = defaultdict(_group_to_ctx)
    chrom_cache = {}

    def get_chrom_info(chrom: str):
        if chrom not in chrom_cache:
            if chrom not in genome_dict or not is_main_chr(chrom, n_chr):
                chrom_cache[chrom] = None
            else:
                seq = genome_dict[chrom]
                chrom_cache[chrom] = (np.frombuffer(seq.encode("ascii"), dtype="S1"), len(seq))
        return chrom_cache[chrom]

    total_records = 0
    matched_4d = 0
    with smart_open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            total_records += 1
            if total_records % 10_000_000 == 0:
                logger.info(f"VCF processed={total_records:,}, matched_4D={matched_4d:,}")

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            if len(parts) > 6 and parts[6] not in {"PASS", "."}:
                continue

            chrom = parts[0]
            chr_num = get_chr_num(chrom)
            if chr_num < 1 or chr_num > n_chr:
                continue

            pos1 = int(parts[1])
            pos0 = pos1 - 1
            ref = parts[3].strip().upper()
            if len(ref) != 1 or ref not in BASES:
                continue
            if (chrom, pos0) not in sites_4d_set:
                continue

            info = get_chrom_info(chrom)
            if info is None:
                continue
            seq_bytes, seq_len = info
            if pos0 < 1 or pos0 + 1 >= seq_len:
                continue

            prev_b = seq_bytes[pos0 - 1]
            curr_b = seq_bytes[pos0]
            next_b = seq_bytes[pos0 + 1]
            if b"N" in (prev_b, curr_b, next_b):
                continue
            if curr_b != ref.encode("ascii"):
                continue

            matched_4d += 1
            context = (prev_b + curr_b + next_b).decode("ascii")
            for alt in parts[4].strip().upper().split(","):
                if len(alt) != 1 or alt not in BASES or alt == ref:
                    continue
                total_mut[GROUP][context][alt] += 1
                by_chrnum[chr_num][GROUP][context][alt] += 1

    logger.info(f"VCF mutation count complete: records={total_records:,}, 4D_SNP={matched_4d:,}")
    return total_mut, dict(by_chrnum)
