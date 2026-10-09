"""Identical source function; caller module supplies its recorded policy bindings."""
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
from phenofuncrank_repro.plantvimpact.native_upstream.candidate_kernels import ParquetChunkWriter
from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_functions

def iter_noncoding(gff_obj: dict, genome: Dict[str, str], vcf_idx: dict, call_idx: dict, args) -> Iterator[dict]:
    intervals_by_chr = collect_noncoding_intervals(gff_obj, genome, args)
    total_intervals = sum(len(v) for v in intervals_by_chr.values())
    done = 0
    for chrom, intervals in sorted(intervals_by_chr.items()):
        seq = genome.get(chrom)
        if seq is None:
            continue
        sg_str, sg_code = infer_sg(chrom, args.subgenome_mode)
        chr_num = infer_chr_num(chrom, args.n_chr)
        vcf_chrom = vcf_idx.get(chrom, {})
        for reg in intervals:
            done += 1
            a, b = int(reg["start"]), int(reg["end"])
            if b < a:
                continue

            for start in range(a, b + 1, args.position_chunk_size):
                end = min(b, start + args.position_chunk_size - 1)
                pos_arr = np.arange(start, end + 1, dtype=np.int64)
                cr = callrate_lookup(call_idx, chrom, pos_arr)
                for k, pos in enumerate(pos_arr):
                    ref = seq[int(pos) - 1].upper()
                    if ref not in BASE2INT:
                        continue
                    tri, ccode = ctx_code(seq, int(pos))
                    if ccode < 0:
                        continue
                    tss = reg.get("tss")
                    gs, ge = reg.get("gene_start"), reg.get("gene_end")
                    dist_tss = int(pos - tss) if tss is not None else 10**9
                    if gs is None or ge is None:
                        dist_gene = 10**9
                    elif gs <= pos <= ge:
                        dist_gene = 0
                    else:
                        dist_gene = min(abs(int(pos) - int(gs)), abs(int(pos) - int(ge)))

                    donor_offset = 999999
                    acceptor_offset = 999999
                    if reg["region_type"] == "splice_near":
                        donor_offset = min(abs(int(pos) - int(reg["start"])), abs(int(pos) - int(reg["end"])))
                        acceptor_offset = donor_offset
                    subtype = infer_region_subtype(reg["region_type"], dist_tss, donor_offset, acceptor_offset)
                    for alt in ["A", "C", "G", "T"]:
                        if alt == ref:
                            continue
                        obs, ac, an, af = vcf_annot(vcf_chrom, int(pos), ref, alt)
                        promoter_core = int(reg["region_type"] == "promoter" and abs(dist_tss) <= 500)
                        promoter_prox = int(reg["region_type"] == "promoter" and 500 < abs(dist_tss) <= 1000)
                        promoter_dist = int(reg["region_type"] == "promoter" and abs(dist_tss) > 1000)
                        yield {
                            "snp_id": f"{chrom}_{int(pos)}_{ref}_{alt}",
                            "SNP_ID": f"{chrom}_{int(pos)}_{ref}_{alt}",
                            "Chr": chrom, "Pos": int(pos), "Ref": ref, "Alt": alt,
                            "GeneID": reg["GeneID"], "gene_id": reg["GeneID"],
                            "region_type": reg["region_type"], "region_subtype": subtype,
                            "dist_to_TSS": dist_tss, "dist_to_gene": dist_gene,
                            "promoter_core_flag": promoter_core,
                            "promoter_proximal_flag": promoter_prox,
                            "promoter_distal_flag": promoter_dist,
                            "donor_offset": donor_offset,
                            "acceptor_offset": acceptor_offset,
                            "canonical_splice_core_flag": int(reg["region_type"] == "splice_near" and min(abs(donor_offset), abs(acceptor_offset)) <= 2),
                            "homoeolog_expr_bias": 0.0, "has_homolog": 0,
                            "chr_num": chr_num, "subgenome": sg_str, "Sg_code": sg_code,
                            "subgenome_code": sg_code, "Context": tri, "Ctx_code": ccode, "Alt_code": BASE2INT[alt],
                            "observed_in_vcf": obs, "Y_Label": obs, "AC": ac, "AN": an, "AF": af,
                            "AN_obs": float(cr["AN_obs"][k]), "callable_n": float(cr["callable_n"][k]),
                            "call_rate": float(cr["call_rate"][k]), "callable_rate": float(cr["callable_rate"][k]),
                            "callrate_hit": int(cr["callrate_hit"][k]),
                        }
            if done % 10000 == 0:
                logger.info(f"  noncoding intervals processed: {done:,}/{total_intervals:,}")
