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
    ap.add_argument("--intron-selected", default=None, help="Optional parquet from annotation-first intron top1%% selection; if provided, noncoding intron positions are restricted to this whitelist.")
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


def load_intron_selected_pos_map_for_preLM(parquet_path, chrom_regex=None):
    """
    Load annotation-first intron selected sites.
    Return:
      pos_map[Chr] = sorted unique np.ndarray(int64)
    """
    if parquet_path is None:
        return None

    import pyarrow.parquet as _pq
    import numpy as _np
    import pandas as _pd
    import re as _re

    tbl = _pq.read_table(parquet_path, columns=["Chr", "Pos"])
    df = tbl.to_pandas()
    df["Chr"] = df["Chr"].astype(str)
    df["Pos"] = _pd.to_numeric(df["Pos"], errors="coerce").fillna(-1).astype("int64")
    df = df[df["Pos"] > 0].copy()

    if chrom_regex:
        pat = _re.compile(chrom_regex)
        df = df[df["Chr"].map(lambda x: bool(pat.search(str(x))))].copy()

    pos_map = {}
    for chrom, sub in df.groupby("Chr", sort=False):
        arr = _np.unique(sub["Pos"].to_numpy(dtype=_np.int64, copy=False))
        arr.sort()
        pos_map[str(chrom)] = arr

    logger.info(
        f"Loaded intron-selected whitelist: path={parquet_path}, "
        f"chroms={len(pos_map)}, total_sites={sum(len(v) for v in pos_map.values()):,}"
    )
    return pos_map


def slice_sorted_positions_for_preLM(sorted_pos_arr, a, b):
    """Return sorted positions within closed interval [a,b]."""
    if sorted_pos_arr is None or len(sorted_pos_arr) == 0:
        return np.empty(0, dtype=np.int64)
    l = np.searchsorted(sorted_pos_arr, int(a), side="left")
    r = np.searchsorted(sorted_pos_arr, int(b), side="right")
    return sorted_pos_arr[l:r]


def iter_noncoding(gff_obj: dict, genome: Dict[str, str], vcf_idx: dict, call_idx: dict, args) -> Iterator[dict]:
    if not hasattr(args, "intron_selected_pos_map"):
        args.intron_selected_pos_map = load_intron_selected_pos_map_for_preLM(
            getattr(args, "intron_selected", None),
            chrom_regex=getattr(args, "chrom_regex", None),
        )

    intron_pos_map = getattr(args, "intron_selected_pos_map", None)

    intervals_by_chr = collect_noncoding_intervals(gff_obj, genome, args)
    total_intervals = sum(len(v) for v in intervals_by_chr.values())
    done = 0
    intron_intervals_seen = 0
    intron_intervals_kept = 0
    intron_positions_emitted = 0

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

            region_type = reg["region_type"]


            if region_type == "intron" and intron_pos_map is not None:
                intron_intervals_seen += 1
                selected_pos = slice_sorted_positions_for_preLM(intron_pos_map.get(str(chrom)), a, b)
                if selected_pos.size == 0:
                    continue
                intron_intervals_kept += 1
                intron_positions_emitted += int(selected_pos.size)

                pos_chunks = (
                    selected_pos[i:i + args.position_chunk_size]
                    for i in range(0, len(selected_pos), args.position_chunk_size)
                )
            else:
                pos_chunks = (
                    np.arange(start, min(b, start + args.position_chunk_size - 1) + 1, dtype=np.int64)
                    for start in range(a, b + 1, args.position_chunk_size)
                )

            for pos_arr in pos_chunks:
                if pos_arr.size == 0:
                    continue

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
                    if region_type == "splice_near":
                        donor_offset = min(abs(int(pos) - int(reg["start"])), abs(int(pos) - int(reg["end"])))
                        acceptor_offset = donor_offset

                    subtype = infer_region_subtype(region_type, dist_tss, donor_offset, acceptor_offset)

                    for alt in ["A", "C", "G", "T"]:
                        if alt == ref:
                            continue

                        obs, ac, an, af = vcf_annot(vcf_chrom, int(pos), ref, alt)

                        promoter_core = int(region_type == "promoter" and abs(dist_tss) <= 500)
                        promoter_prox = int(region_type == "promoter" and 500 < abs(dist_tss) <= 1000)
                        promoter_dist = int(region_type == "promoter" and abs(dist_tss) > 1000)

                        yield {
                            "snp_id": f"{chrom}_{int(pos)}_{ref}_{alt}",
                            "SNP_ID": f"{chrom}_{int(pos)}_{ref}_{alt}",
                            "Chr": chrom,
                            "Pos": int(pos),
                            "Ref": ref,
                            "Alt": alt,
                            "GeneID": reg["GeneID"],
                            "gene_id": reg["GeneID"],
                            "region_type": region_type,
                            "region_subtype": subtype,
                            "dist_to_TSS": dist_tss,
                            "dist_to_gene": dist_gene,
                            "promoter_core_flag": promoter_core,
                            "promoter_proximal_flag": promoter_prox,
                            "promoter_distal_flag": promoter_dist,
                            "donor_offset": donor_offset,
                            "acceptor_offset": acceptor_offset,
                            "canonical_splice_core_flag": int(region_type == "splice_near" and min(abs(donor_offset), abs(acceptor_offset)) <= 2),
                            "homoeolog_expr_bias": 0.0,
                            "has_homolog": 0,
                            "chr_num": chr_num,
                            "subgenome": sg_str,
                            "Sg_code": sg_code,
                            "subgenome_code": sg_code,
                            "Context": tri,
                            "Ctx_code": ccode,
                            "Alt_code": BASE2INT[alt],
                            "observed_in_vcf": obs,
                            "Y_Label": obs,
                            "AC": ac,
                            "AN": an,
                            "AF": af,
                            "AN_obs": float(cr["AN_obs"][k]),
                            "callable_n": float(cr["callable_n"][k]),
                            "call_rate": float(cr["call_rate"][k]),
                            "callable_rate": float(cr["callable_rate"][k]),
                            "callrate_hit": int(cr["callrate_hit"][k]),
                        }

            if done % 10000 == 0:
                logger.info(f"  noncoding intervals processed: {done:,}/{total_intervals:,}")

    if intron_pos_map is not None:
        logger.info(
            f"Intron-selected restriction summary: "
            f"intron_intervals_seen={intron_intervals_seen:,}, "
            f"intron_intervals_kept={intron_intervals_kept:,}, "
            f"unique_position_hits_before_alt_expansion={intron_positions_emitted:,}"
        )


from phenofuncrank_repro.plantvimpact.native_upstream.candidate_kernels import ParquetChunkWriter
from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_functions
from phenofuncrank_repro.plantvimpact.native_upstream import candidate_kernels as _source_kernels
_bind_functions(globals(), _source_kernels, ['_exact_chroms_from_regex', '_runs_from_bool_mask', 'build_cds_map', 'callrate_lookup', 'clean_gene_id', 'collect_noncoding_intervals', 'ctx_code', 'first_num', 'gene_id_from_feature', 'infer_chr_num', 'infer_region_subtype', 'infer_sg', 'intervals_from_blocks', 'iter_coding_missense', 'load_callrate', 'load_fasta', 'load_vcf_snv_index', 'merge_intervals', 'open_text', 'parse_attrs', 'parse_gff', 'parse_info', 'subtract_intervals', 'vcf_annot', 'write_iterator_to_parquet'])


from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_identical
from phenofuncrank_repro.common import shared_iter_noncoding as _profile_kernel_9
_bind_identical(globals(), _profile_kernel_9, ['iter_noncoding'])

if __name__ == "__main__":
    main()
