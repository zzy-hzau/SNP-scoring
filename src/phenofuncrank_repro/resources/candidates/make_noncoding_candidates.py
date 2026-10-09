#!/usr/bin/env python3
"Generate noncoding candidate positions from a reference FASTA + GFF3."

import argparse
import os
import random
import sys
from collections import defaultdict


from phenofuncrank_repro.resources.utils.common import *


def parse_args(argv=None):
    """Build the argparse argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate noncoding candidate positions (promoter, UTR, intron, "
            "splice-proximal) from a reference FASTA + GFF3."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fasta", required=True, help="Path to the reference FASTA file."
    )
    parser.add_argument(
        "--gff", required=True, help="Path to the GFF3 annotation file."
    )
    parser.add_argument(
        "--species", required=True, help="Species identifier for the output."
    )
    parser.add_argument(
        "--output", required=True, help="Path to the output TSV file."
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Cap on the number of output rows (toy/test mode).",
    )
    parser.add_argument(
        "--promoter-window",
        type=int,
        default=2000,
        help="Promoter window size upstream of gene start.",
    )
    parser.add_argument(
        "--splice-window",
        type=int,
        default=5,
        help="Window around splice junctions classified as splice_proximal.",
    )
    parser.add_argument(
        "--chromosomes",
        default=None,
        help="Comma-separated list of chromosomes to restrict to.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML configuration file with 'species' overrides.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed used when downsampling with --max-positions.",
    )
    return parser.parse_args(argv)


def load_yaml_config(path):
    """Load a YAML configuration file if PyYAML is available."""
    if not path:
        return {}
    try:
        import yaml  
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to use --config but is not installed."
        ) from exc
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def collect_gene_records(gff_path, allowed_chroms):
    """Return per-gene features: gene boundaries, mRNA, exons, CDS, UTRs."""
    genes = {}  
    by_parent = defaultdict(lambda: {
        "exons": [], "cds": [], "utr5": [], "utr3": [],
        "boundary": None,
    })

    with open_text(gff_path, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            seqid = fields[0]
            if allowed_chroms is not None and seqid not in allowed_chroms:
                continue
            ftype = fields[2]
            try:
                start = int(fields[3])
                end = int(fields[4])
            except ValueError:
                continue
            strand = fields[6]
            attrs = parse_gff_attrs(fields[8])

            if ftype == "gene":
                gid = attrs.get("ID", attrs.get("Name", ""))
                if isinstance(gid, list):
                    gid = gid[0] if gid else ""
                genes[gid] = (seqid, start, end, strand)
            elif ftype in ("exon", "CDS", "five_prime_UTR", "three_prime_UTR"):
                parent = attrs.get("Parent", attrs.get("ID", ""))
                if isinstance(parent, list):
                    parent = parent[0] if parent else ""
                if ftype == "exon":
                    by_parent[parent]["exons"].append((seqid, start, end, strand))
                elif ftype == "CDS":
                    by_parent[parent]["cds"].append((seqid, start, end, strand))
                elif ftype == "five_prime_UTR":
                    by_parent[parent]["utr5"].append((seqid, start, end, strand))
                elif ftype == "three_prime_UTR":
                    by_parent[parent]["utr3"].append((seqid, start, end, strand))

    return genes, by_parent


def emit_positions(seqid, start, end, fasta, region_type, species, out_handle,
                   state, splice_windows=None):
    """Emit SNV candidates for a [start, end] (1-based, inclusive) interval."""
    chrom = fasta.get(seqid, "")
    if not chrom or start < 1 or end > len(chrom) or start > end:
        return
    splice_windows = splice_windows or set()
    for pos in range(start, end + 1):
        ref_base = chrom[pos - 1].upper()
        if ref_base not in BASES:
            continue
        if pos in splice_windows:
            actual_region = "splice_proximal"
        else:
            actual_region = region_type
        for alt_base in BASES:
            if alt_base == ref_base:
                continue
            snp_id = f"{seqid}:{pos}:{ref_base}:{alt_base}"
            out_handle.write(
                "\t".join([
                    species, "noncoding", snp_id, seqid, str(pos),
                    ref_base, alt_base, actual_region,
                ]) + "\n"
            )
            state["n"] += 1
            if state.get("max") and state["n"] >= state["max"]:
                return True
    return False


def compute_splice_windows(exons, splice_window):
    """Return the set of positions within `splice_window` bp of a junction."""
    windows = set()
    for seqid, start, end, _strand in exons:
        for pos in range(start, start + splice_window):
            windows.add((seqid, pos))
        for pos in range(end - splice_window + 1, end + 1):
            windows.add((seqid, pos))
    return windows


def main(argv=None):
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    if args.species is None:
        args.species = config.get("species")
    if args.random_seed is None and "random_seed" in config:
        args.random_seed = config["random_seed"]

    if args.random_seed is not None:
        random.seed(args.random_seed)

    allowed_chroms = None
    if args.chromosomes:
        allowed_chroms = set(c.strip() for c in args.chromosomes.split(",") if c.strip())
    elif config.get("chromosomes"):
        allowed_chroms = set(config["chromosomes"])

    species_tag = f"[{args.species}] " if args.species else ""
    log(f"{species_tag}Building noncoding candidates from {args.gff}")

    try:
        fasta = load_fasta(args.fasta)
    except Exception as exc:  
        log(f"ERROR loading FASTA: {exc}")
        return 1

    if allowed_chroms is not None:
        normalized = {norm_chrom(c): c for c in fasta}
        allowed_chroms = {
            normalized.get(norm_chrom(c), c) for c in allowed_chroms
        }

    genes, by_parent = collect_gene_records(args.gff, allowed_chroms)

    try:
        out_handle = open_text(args.output, "wt")
    except OSError as exc:
        log(f"ERROR: cannot open output {args.output}: {exc}")
        return 1

    state = {"n": 0, "max": args.max_positions}
    header = [
        "species", "kind", "snp_id", "Chr", "Pos", "Ref", "Alt", "region_type",
    ]

    try:
        out_handle.write("\t".join(header) + "\n")


        for gid, (seqid, gstart, gend, strand) in genes.items():
            if allowed_chroms is not None and seqid not in allowed_chroms:
                continue
            chrom = fasta.get(seqid, "")
            if not chrom:
                continue


            if strand == "-":
                prom_end = gend + args.promoter_window
                prom_start = gend + 1
            else:
                prom_start = max(1, gstart - args.promoter_window)
                prom_end = gstart - 1
            if prom_start <= prom_end:
                if emit_positions(seqid, prom_start, prom_end, fasta,
                                  "promoter", args.species, out_handle, state):
                    return 0


        for parent, feats in by_parent.items():
            if not feats["exons"] and not feats["cds"]:
                continue
            seqid = None
            for key in ("exons", "cds", "utr5", "utr3"):
                for (s, _start, _end, _strand) in feats[key]:
                    seqid = s
                    break
                if seqid:
                    break
            if seqid is None:
                continue
            if allowed_chroms is not None and seqid not in allowed_chroms:
                continue


            for region_key, region_type in (
                ("utr5", "utr5"), ("utr3", "utr3"),
            ):
                for (s, start, end, _strand) in feats[region_key]:
                    if s != seqid:
                        continue
                    if emit_positions(s, start, end, fasta, region_type,
                                      args.species, out_handle, state):
                        return 0


            exons = sorted(feats["exons"], key=lambda e: e[1])
            splice_windows = compute_splice_windows(exons, args.splice_window)
            splice_pos_set = {
                p for (s, p) in splice_windows if s == seqid
            }
            for i in range(len(exons) - 1):
                _s, _start_a, end_a, _strand = exons[i]
                _s, start_b, _end_b, _strand = exons[i + 1]
                if end_a + 1 < start_b:
                    if emit_positions(seqid, end_a + 1, start_b - 1, fasta,
                                      "intron", args.species, out_handle,
                                      state, splice_pos_set):
                        return 0
    finally:
        out_handle.close()

    log(
        f"{species_tag}Done. noncoding_candidates_written={state['n']} "
        f"genes_seen={len(genes)} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
