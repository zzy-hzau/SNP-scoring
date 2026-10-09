#!/usr/bin/env python3
"Verify VCF REF alleles match the reference FASTA at each position."

import argparse
import os
import sys


from phenofuncrank_repro.resources.utils.common import *


def parse_args(argv=None):
    """Build the argparse argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Verify that VCF REF alleles match the reference FASTA at each "
            "variant position."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vcf", required=True, help="Path to the input VCF file."
    )
    parser.add_argument(
        "--fasta", required=True, help="Path to the reference FASTA file."
    )
    parser.add_argument(
        "--output", required=True, help="Path to the output TSV report."
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Cap on the number of variant records checked (toy/test mode).",
    )
    parser.add_argument(
        "--chromosomes",
        default=None,
        help="Comma-separated list of main chromosomes to check.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML configuration file with 'species' and "
        "'chromosomes' fields.",
    )
    parser.add_argument(
        "--species",
        default=None,
        help="Optional species identifier (used for logging and report).",
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


def is_main_chrom(chrom, allowed_chroms):
    """Return True if chrom should be considered a main chromosome."""
    if allowed_chroms is None:

        s = str(chrom)
        if s.lower().startswith("chr"):
            s = s[3:]
        if s.isdigit() or s in {"X", "Y", "Mt", "Pt", "M", "C", "chloroplast", "mitochondria"}:
            return True

        if s in {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
                 "XI", "XII"}:
            return True
        return False
    return chrom in allowed_chroms


def iter_vcf_records(handle):
    """Yield (chrom, pos, ref, alt) tuples from a VCF file."""
    for line in handle:
        if not line or line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 5:
            continue
        yield fields[0], int(fields[1]), fields[3], fields[4]


def main(argv=None):
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    if args.species is None:
        args.species = config.get("species")

    allowed_chroms = None
    if args.chromosomes:
        allowed_chroms = set(c.strip() for c in args.chromosomes.split(",") if c.strip())
    elif config.get("chromosomes"):
        allowed_chroms = set(config["chromosomes"])

    species_tag = f"[{args.species}] " if args.species else ""
    log(f"{species_tag}Checking VCF REF against FASTA: {args.vcf} vs {args.fasta}")

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

    try:
        in_handle = open_text(args.vcf, "rt")
    except OSError as exc:
        log(f"ERROR: cannot open VCF {args.vcf}: {exc}")
        return 1

    try:
        out_handle = open_text(args.output, "wt")
    except OSError as exc:
        log(f"ERROR: cannot open output {args.output}: {exc}")
        in_handle.close()
        return 1

    try:
        out_handle.write("Chr\tPos\tvcf_ref\tfasta_ref\tmatch\n")
        n_checked = 0
        n_match = 0
        n_mismatch = 0
        for chrom, pos, ref, _alt in iter_vcf_records(in_handle):
            if not is_main_chrom(chrom, allowed_chroms):
                continue
            chrom_seq = fasta.get(chrom)
            if chrom_seq is None:

                norm_map = {norm_chrom(c): c for c in fasta}
                actual = norm_map.get(norm_chrom(chrom))
                if actual is None:
                    continue
                chrom_seq = fasta[actual]
            if pos < 1 or pos > len(chrom_seq):
                continue
            fasta_ref = chrom_seq[pos - 1].upper()
            vcf_ref = ref.upper()
            match = (fasta_ref == vcf_ref)
            out_handle.write(
                f"{chrom}\t{pos}\t{vcf_ref}\t{fasta_ref}\t{match}\n"
            )
            n_checked += 1
            if match:
                n_match += 1
            else:
                n_mismatch += 1
            if args.max_positions and n_checked >= args.max_positions:
                break
    finally:
        in_handle.close()
        out_handle.close()

    log(
        f"{species_tag}Done. checked={n_checked} match={n_match} "
        f"mismatch={n_mismatch} -> {args.output}"
    )
    return 0 if n_mismatch == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
