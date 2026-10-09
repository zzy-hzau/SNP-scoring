#!/usr/bin/env python3

"maize_calc_pmut_fast.py"

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("maize_calc_pmut_fast")

GROUP = "Maize"
BASES = ("A", "C", "G", "T")
BASE2INT = {"A": 0, "C": 1, "G": 2, "T": 3}
MU_BASE_DEFAULT = 4.56e-9


def _ctx_to_alt():
    return defaultdict(int)


def _group_to_ctx():
    return {GROUP: defaultdict(_ctx_to_alt)}


def smart_open(path: str, mode: str = "rt"):
    return gzip.open(path, mode) if str(path).endswith(".gz") else open(path, mode)


def get_chr_num(chrom: str) -> int:
    """Return chromosome number for maize names such as 1, Chr1, chr01, chromosome_1."""
    s = str(chrom).strip()
    m = re.search(r"(?:^|chr|chromosome[_-]?)(\d+)$", s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else -1


def is_main_chr(chrom: str, n_chr: int = 10) -> bool:
    c = get_chr_num(chrom)
    return 1 <= c <= int(n_chr)


def get_subgenome(chrom: str) -> str:
    return GROUP if is_main_chr(chrom, 10) else "Unknown"


def load_genome_once(fasta_path: str, n_chr: int = 10, keep_non_main: bool = False) -> Dict[str, str]:
    genome: Dict[str, str] = {}
    with smart_open(fasta_path, "rt") as f:
        chrom, seqs = None, []
        for line in f:
            if line.startswith(">"):
                if chrom is not None and (keep_non_main or is_main_chr(chrom, n_chr)):
                    genome[chrom] = "".join(seqs).upper()
                chrom = line.strip()[1:].split()[0]
                seqs = []
            else:
                if chrom is not None and (keep_non_main or is_main_chr(chrom, n_chr)):
                    seqs.append(line.strip().upper())
        if chrom is not None and (keep_non_main or is_main_chr(chrom, n_chr)):
            genome[chrom] = "".join(seqs).upper()
    logger.info(f"Loaded genome sequences: {len(genome)} | {sorted(genome.keys(), key=lambda x: get_chr_num(x))}")
    return genome


def load_4d_site_set(sites_4d_path: str) -> set[tuple[str, int]]:
    sites = set()
    with smart_open(sites_4d_path, "rt") as f:
        header = next(f, None)
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            sites.add((parts[0], int(parts[1])))
    logger.info(f"Loaded 4D site set: {len(sites):,}")
    return sites


def _subtract_counter(dst: Counter, src: Counter):
    for k, v in src.items():
        dst[k] -= v
        if dst[k] <= 0:
            dst.pop(k, None)


def _subtract_mut(dst, src):
    for ctx, alt_dict in src.items():
        for alt, v in alt_dict.items():
            dst[ctx][alt] -= v
            if dst[ctx][alt] <= 0:
                dst[ctx].pop(alt, None)
        if not dst[ctx]:
            dst.pop(ctx, None)


def build_all_fold_rate_tables_fast(fasta_path: str, vcf_path: str, sites_4d_path: str, n_chr: int = 10):
    genome = load_genome_once(fasta_path, n_chr=n_chr)
    sites_4d_set = load_4d_site_set(sites_4d_path)
    genome_total, genome_by_chrnum = load_4d_counts_total_and_by_chrnum(sites_4d_path, genome, n_chr=n_chr)
    mut_total, mut_by_chrnum = count_vcf_mutations_total_and_by_chrnum(vcf_path, genome, sites_4d_set, n_chr=n_chr)

    rate_tables = {}
    for chr_num in range(1, n_chr + 1):
        rate_tables[chr_num] = build_rate_table_excluding_chrnums(
            genome_total, genome_by_chrnum, mut_total, mut_by_chrnum, exclude_chrs=[chr_num]
        )
    return rate_tables, genome_total, genome_by_chrnum, mut_total, mut_by_chrnum


def main():
    parser = argparse.ArgumentParser(description="Build maize global/fold P_mut rate tables from 4D sites and wild VCF")
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--sites-4d", required=True)
    parser.add_argument("--n-chr", type=int, default=10)
    parser.add_argument("--out", default="maize_rate_tables_cache.pkl")
    parser.add_argument("--global-json", default="maize_global_rate_table.json")
    args = parser.parse_args()

    rate_tables, genome_total, genome_by_chrnum, mut_total, mut_by_chrnum = build_all_fold_rate_tables_fast(
        args.fasta, args.vcf, args.sites_4d, n_chr=args.n_chr
    )
    global_table = build_rate_table_excluding_chrnums(
        genome_total, genome_by_chrnum, mut_total, mut_by_chrnum, exclude_chrs=[]
    )

    out = Path(args.out)
    with open(out, "wb") as f:
        pickle.dump({
            "rate_tables": rate_tables,
            "genome_total": genome_total,
            "genome_by_chrnum": genome_by_chrnum,
            "mut_total": mut_total,
            "mut_by_chrnum": mut_by_chrnum,
            "global_rate_table": global_table,
        }, f)
    with open(args.global_json, "w") as f:
        json.dump(global_table, f, indent=2)

    vals = np.array(list(global_table.values()), dtype=np.float64)
    logger.info(f"Saved: {out}")
    logger.info(f"Saved global json: {args.global_json}")
    logger.info(f"Global P_mut n={len(vals)}, min={vals.min():.3e}, median={np.median(vals):.3e}, max={vals.max():.3e}")


from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_identical
from phenofuncrank_repro.common import shared_load_4d_counts_total_and_by_chrnum as _profile_kernel_5
_bind_identical(globals(), _profile_kernel_5, ['load_4d_counts_total_and_by_chrnum'])
from phenofuncrank_repro.common import shared_count_vcf_mutations_total_and_by_chrnum as _profile_kernel_6
_bind_identical(globals(), _profile_kernel_6, ['count_vcf_mutations_total_and_by_chrnum'])
from phenofuncrank_repro.common import shared_build_rate_table_excluding_chrnums as _profile_kernel_7
_bind_identical(globals(), _profile_kernel_7, ['build_rate_table_excluding_chrnums'])

if __name__ == "__main__":
    main()
