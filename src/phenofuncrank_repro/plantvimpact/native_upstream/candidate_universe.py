#!/usr/bin/env python3

"Build pre-LM CODING and NONCODING candidate-universe tables directly from raw inputs."

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("build_preLM_universe_from_raw")

BASES = np.array(["A", "C", "G", "T"], dtype=object)
BASE2INT = {"A": 0, "C": 1, "G": 2, "T": 3}
RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "C", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

CODON_TABLE["CGC"] = "R"


def main():
    ap = argparse.ArgumentParser(description="Build pre-LM coding/noncoding SNP universe from raw FASTA/GFF/VCF/callrate")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--gff", required=True)
    ap.add_argument("--wild-vcf", required=True)
    ap.add_argument("--callrate", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", default="both", choices=["coding", "noncoding", "both"])
    ap.add_argument("--n-chr", type=int, default=13)
    ap.add_argument("--subgenome-mode", default="auto", choices=["auto", "single"])
    ap.add_argument("--chrom-regex", default=None, help="Optional regex to keep only main chromosomes in FASTA/VCF")
    ap.add_argument("--max-callable-n", type=int, default=None, help="Used to derive callable_rate from callable_n if needed")
    ap.add_argument("--promoter-upstream", type=int, default=2000)
    ap.add_argument("--promoter-downstream", type=int, default=200)
    ap.add_argument("--splice-window", type=int, default=8)
    ap.add_argument("--chunk-rows", type=int, default=500000)
    ap.add_argument("--position-chunk-size", type=int, default=200000)
    ap.add_argument("--max-rows-per-mode", type=int, default=None, help="Debug only; do not use for final runs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genome = load_fasta(args.genome, args.chrom_regex)
    gff_obj = parse_gff(args.gff, set(genome))
    vcf_idx = load_vcf_snv_index(args.wild_vcf, args.chrom_regex)
    call_idx = load_callrate(args.callrate, args.max_callable_n, args.chrom_regex)

    manifest = {
        "genome": args.genome,
        "gff": args.gff,
        "wild_vcf": args.wild_vcf,
        "callrate": args.callrate,
        "mode": args.mode,
        "n_chr": args.n_chr,
        "subgenome_mode": args.subgenome_mode,
        "promoter_upstream": args.promoter_upstream,
        "promoter_downstream": args.promoter_downstream,
        "splice_window": args.splice_window,
        "fasta_chroms": list(genome.keys()),
        "gff_feature_counts": gff_obj["feature_counts"],
        "outputs": {},
    }

    if args.mode in {"coding", "both"}:
        path = out_dir / "coding_preLM_universe.parquet"
        logger.info("Building CODING missense pre-LM universe...")
        n = write_iterator_to_parquet(iter_coding_missense(gff_obj, genome, vcf_idx, call_idx, args), path, args.chunk_rows, args.max_rows_per_mode)
        manifest["outputs"]["coding_preLM_universe"] = str(path)
        manifest["outputs"]["coding_rows"] = n
        logger.info(f"CODING pre-LM universe rows: {n:,}")

    if args.mode in {"noncoding", "both"}:
        path = out_dir / "noncoding_preLM_universe.parquet"
        logger.info("Building NONCODING pre-LM universe...")
        n = write_iterator_to_parquet(iter_noncoding(gff_obj, genome, vcf_idx, call_idx, args), path, args.chunk_rows, args.max_rows_per_mode)
        manifest["outputs"]["noncoding_preLM_universe"] = str(path)
        manifest["outputs"]["noncoding_rows"] = n
        logger.info(f"NONCODING pre-LM universe rows: {n:,}")

    (out_dir / "preLM_universe_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Done. Manifest: {out_dir / 'preLM_universe_manifest.json'}")


from phenofuncrank_repro.plantvimpact.native_upstream.candidate_kernels import ParquetChunkWriter
from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_functions
from phenofuncrank_repro.plantvimpact.native_upstream import candidate_kernels as _source_kernels
_bind_functions(globals(), _source_kernels, ['_exact_chroms_from_regex', '_runs_from_bool_mask', 'build_cds_map', 'callrate_lookup', 'clean_gene_id', 'collect_noncoding_intervals', 'ctx_code', 'first_num', 'gene_id_from_feature', 'infer_chr_num', 'infer_region_subtype', 'infer_sg', 'intervals_from_blocks', 'iter_coding_missense', 'load_callrate', 'load_fasta', 'load_vcf_snv_index', 'merge_intervals', 'open_text', 'parse_attrs', 'parse_gff', 'parse_info', 'subtract_intervals', 'vcf_annot', 'write_iterator_to_parquet'])


from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_identical
from phenofuncrank_repro.common import shared_iter_noncoding as _profile_kernel_9
_bind_identical(globals(), _profile_kernel_9, ['iter_noncoding'])

if __name__ == "__main__":
    main()
