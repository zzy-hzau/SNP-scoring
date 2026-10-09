#!/usr/bin/env python3
"Verify FASTA / GFF3 seqid consistency."

import argparse
import os
import sys


from phenofuncrank_repro.resources.utils.common import *


def parse_args(argv=None):
    """Build the argparse argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Verify that FASTA and GFF3 use consistent seqids and write a "
            "TSV report of seqid presence/absence and feature counts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Path to the reference FASTA file.",
    )
    parser.add_argument(
        "--gff",
        required=True,
        help="Path to the GFF3 annotation file (may be gzip-compressed).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output TSV report.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML configuration file. May provide a 'species' "
        "field and a 'normalize_seqids' boolean flag.",
    )
    parser.add_argument(
        "--species",
        default=None,
        help="Optional species identifier (used for logging only).",
    )
    parser.add_argument(
        "--normalize-seqids",
        action="store_true",
        help="Normalize seqids with utils.norm_chrom before comparison.",
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


def iter_gff_records(handle):
    """Yield (seqid, feature_type) tuples from a GFF3 file."""
    for line in handle:
        if not line or line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 9:
            continue
        seqid = fields[0]
        feature = fields[2]
        yield seqid, feature


def main(argv=None):
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    if args.species is None:
        args.species = config.get("species")
    if config.get("normalize_seqids") and not args.normalize_seqids:
        args.normalize_seqids = True

    species_tag = f"[{args.species}] " if args.species else ""
    log(f"{species_tag}Checking FASTA/GFF seqid consistency")


    fasta_lengths = {}
    try:
        for header, seq in fasta_iter(args.fasta):
            seqid = header.split()[0]
            if args.normalize_seqids:
                seqid = norm_chrom(seqid)
            seq = "".join(seq.split())
            fasta_lengths[seqid] = len(seq)
    except OSError as exc:
        log(f"ERROR: cannot read FASTA {args.fasta}: {exc}")
        return 1


    gff_feature_counts = {}
    try:
        with open_text(args.gff, "rt") as handle:
            for seqid, _feature in iter_gff_records(handle):
                if args.normalize_seqids:
                    seqid = norm_chrom(seqid)
                gff_feature_counts[seqid] = gff_feature_counts.get(seqid, 0) + 1
    except OSError as exc:
        log(f"ERROR: cannot read GFF {args.gff}: {exc}")
        return 1

    all_seqids = set(fasta_lengths) | set(gff_feature_counts)

    try:
        out_handle = open_text(args.output, "wt")
    except OSError as exc:
        log(f"ERROR: cannot open output {args.output}: {exc}")
        return 1

    try:
        out_handle.write("seqid\tin_fasta\tin_gff\tfasta_length\tgff_features\n")
        for seqid in sorted(all_seqids):
            in_fasta = seqid in fasta_lengths
            in_gff = seqid in gff_feature_counts
            length = fasta_lengths.get(seqid, 0)
            n_feat = gff_feature_counts.get(seqid, 0)
            out_handle.write(
                f"{seqid}\t{in_fasta}\t{in_gff}\t{length}\t{n_feat}\n"
            )
    finally:
        out_handle.close()

    only_fasta = set(fasta_lengths) - set(gff_feature_counts)
    only_gff = set(gff_feature_counts) - set(fasta_lengths)
    log(
        f"{species_tag}fasta_seqids={len(fasta_lengths)} "
        f"gff_seqids={len(gff_feature_counts)} "
        f"only_in_fasta={len(only_fasta)} only_in_gff={len(only_gff)}"
    )
    if only_fasta:
        log("  seqids only in FASTA: " + ", ".join(sorted(only_fasta)[:10]))
    if only_gff:
        log("  seqids only in GFF: " + ", ".join(sorted(only_gff)[:10]))
    log(f"{species_tag}Report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
