#!/usr/bin/env python3
"Standardize a reference FASTA file."

import argparse
import os
import sys


from phenofuncrank_repro.resources.utils.common import *


def parse_args(argv=None):
    """Build the argparse argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Standardize a reference FASTA: rewrap to 60 bp lines, convert "
            "to uppercase, and remove internal whitespace."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input FASTA file (may be gzip-compressed if it "
        "ends with .gz).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the standardized output FASTA file.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1000,
        help="Minimum sequence length to retain. Shorter records are skipped.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML configuration file. May provide 'min_length' "
        "and 'species' overrides.",
    )
    parser.add_argument(
        "--species",
        default=None,
        help="Optional species identifier (used for logging only).",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=60,
        help="Number of bases per line in the output FASTA.",
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


def standardize_sequence(raw_seq, line_width):
    """Uppercase the sequence, strip internal whitespace, rewrap to line_width."""
    cleaned = "".join(raw_seq.split()).upper()
    if line_width <= 0:
        return cleaned
    return "\n".join(
        cleaned[i:i + line_width] for i in range(0, len(cleaned), line_width)
    )


def main(argv=None):
    args = parse_args(argv)
    config = load_yaml_config(args.config)


    if "min_length" in config and args.min_length == 1000:
        args.min_length = int(config["min_length"])
    if "line_width" in config and args.line_width == 60:
        args.line_width = int(config["line_width"])
    if args.species is None:
        args.species = config.get("species")

    species_tag = f"[{args.species}] " if args.species else ""
    log(f"{species_tag}Standardizing FASTA: {args.input} -> {args.output}")

    n_total = 0
    n_skipped = 0
    total_bases = 0

    try:
        out_handle = open_text(args.output, "wt")
    except OSError as exc:
        log(f"ERROR: cannot open output FASTA {args.output}: {exc}")
        return 1

    try:
        try:
            records = list(fasta_iter(args.input))
        except OSError as exc:
            log(f"ERROR: cannot read input FASTA {args.input}: {exc}")
            return 1
        for header, seq in records:
            n_total += 1
            cleaned_seq = "".join(seq.split()).upper()
            if len(cleaned_seq) < args.min_length:
                n_skipped += 1
                log(
                    f"  skipping {header}: length {len(cleaned_seq)} "
                    f"< min_length {args.min_length}"
                )
                continue
            total_bases += len(cleaned_seq)
            wrapped = standardize_sequence(seq, args.line_width)
            out_handle.write(f">{header}\n{wrapped}\n")
    finally:
        out_handle.close()

    log(
        f"{species_tag}Done. sequences={n_total} kept={n_total - n_skipped} "
        f"skipped={n_skipped} total_bases={total_bases}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
