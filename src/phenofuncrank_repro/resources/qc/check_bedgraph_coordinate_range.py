#!/usr/bin/env python3
"Check bedGraph coordinates against a FASTA .fai index."

import argparse
import os
import sys
from collections import defaultdict


from phenofuncrank_repro.resources.utils.common import *


def parse_args(argv=None):
    """Build the argparse argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Check bedGraph coordinates against a FASTA .fai index. "
            "Reports total intervals, out-of-range counts, clipped "
            "counts, and max coordinate per chromosome."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bedgraph",
        required=True,
        help="Path to the input bedGraph file.",
    )
    parser.add_argument(
        "--fai",
        required=True,
        help="Path to the reference FASTA .fai index file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output TSV report.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML configuration file with 'species'.",
    )
    parser.add_argument(
        "--species",
        default=None,
        help="Optional species identifier (used for logging and report).",
    )
    parser.add_argument(
        "--chromosomes",
        default=None,
        help="Comma-separated list of chromosomes to restrict reporting to.",
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


def load_fai_lengths(fai_path):
    """Return a dict mapping seqid -> length from a .fai file."""
    lengths = {}
    with open_text(fai_path, "rt") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            seqid = fields[0]
            try:
                length = int(fields[1])
            except ValueError:
                continue
            lengths[seqid] = length
    return lengths


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
    log(
        f"{species_tag}Checking bedGraph {args.bedgraph} against "
        f"{args.fai}"
    )

    try:
        fai_lengths = load_fai_lengths(args.fai)
    except OSError as exc:
        log(f"ERROR: cannot read .fai {args.fai}: {exc}")
        return 1

    normalized_lengths = {norm_chrom(c): l for c, l in fai_lengths.items()}

    try:
        in_handle = open_text(args.bedgraph, "rt")
    except OSError as exc:
        log(f"ERROR: cannot open bedGraph {args.bedgraph}: {exc}")
        return 1

    n_total = 0
    n_out_of_range = 0
    n_clipped = 0
    max_coord = defaultdict(int)
    chrom_seen = defaultdict(int)
    missing_chrom = defaultdict(int)

    try:
        for line in in_handle:
            if not line or line.startswith("#") or line.startswith("track") \
                    or line.startswith("browser"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            chrom = fields[0]
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError:
                continue


            actual_chrom = chrom
            chrom_len = fai_lengths.get(chrom)
            if chrom_len is None:
                chrom_len = normalized_lengths.get(norm_chrom(chrom))
                if chrom_len is not None:

                    for k, v in fai_lengths.items():
                        if norm_chrom(k) == norm_chrom(chrom):
                            actual_chrom = k
                            break

            if allowed_chroms is not None and chrom not in allowed_chroms \
                    and norm_chrom(chrom) not in {
                        norm_chrom(c) for c in allowed_chroms}:
                continue

            n_total += 1
            chrom_seen[chrom] += 1
            if end > max_coord[chrom]:
                max_coord[chrom] = end

            if chrom_len is None:
                missing_chrom[chrom] += 1
                continue

            out_of_range = end > chrom_len or start < 0
            if out_of_range:
                n_out_of_range += 1

            if start < chrom_len < end:
                n_clipped += 1
            elif start >= chrom_len:
                n_clipped += 1
    finally:
        in_handle.close()

    try:
        out_handle = open_text(args.output, "wt")
    except OSError as exc:
        log(f"ERROR: cannot open output {args.output}: {exc}")
        return 1

    try:
        out_handle.write("metric\tvalue\n")
        out_handle.write(f"species\t{args.species or ''}\n")
        out_handle.write(f"total_intervals\t{n_total}\n")
        out_handle.write(f"out_of_range_intervals\t{n_out_of_range}\n")
        out_handle.write(f"clipped_intervals\t{n_clipped}\n")
        out_handle.write("chrom\tfai_length\tbedgraph_intervals\tmax_coord\tmissing_chrom\n")
        all_chroms = set(fai_lengths) | set(chrom_seen)
        for chrom in sorted(all_chroms, key=lambda c: (
            0, int(c[3:]) if c.lower().startswith("chr") and c[3:].isdigit() else 1, c
        )):
            fai_len = fai_lengths.get(chrom, 0)
            out_handle.write(
                f"{chrom}\t{fai_len}\t{chrom_seen.get(chrom, 0)}\t"
                f"{max_coord.get(chrom, 0)}\t{missing_chrom.get(chrom, 0)}\n"
            )
    finally:
        out_handle.close()

    log(
        f"{species_tag}Done. total={n_total} out_of_range={n_out_of_range} "
        f"clipped={n_clipped} -> {args.output}"
    )
    return 0 if n_out_of_range == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
