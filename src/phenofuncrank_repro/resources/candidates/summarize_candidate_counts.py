#!/usr/bin/env python3
"Summarize candidate counts by species, kind, and variant_type."

import argparse
import os
import sys
from collections import defaultdict


from phenofuncrank_repro.resources.utils.common import *


def parse_args(argv=None):
    """Build the argparse argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Summarize candidate positions by species, kind, and "
            "variant_type and write a summary TSV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True, help="Input candidate positions TSV file."
    )
    parser.add_argument(
        "--output", required=True, help="Output summary TSV file."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML configuration file (currently unused).",
    )
    parser.add_argument(
        "--species",
        default=None,
        help="Optional species filter (only this species will be summarized).",
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


def find_column(cols, names, default=None):
    """Return the index of the first column name in `names` found in `cols`."""
    for name in names:
        if name in cols:
            return cols.index(name)
    return default


def main(argv=None):
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    if args.species is None:
        args.species = config.get("species")

    species_tag = f"[{args.species}] " if args.species else ""
    log(f"{species_tag}Summarizing candidate counts from {args.input}")

    try:
        in_handle = open_text(args.input, "rt")
    except OSError as exc:
        log(f"ERROR: cannot open input {args.input}: {exc}")
        return 1

    header = in_handle.readline().rstrip("\n")
    if not header:
        log("ERROR: empty input file")
        in_handle.close()
        return 1
    cols = header.split("\t")

    species_idx = find_column(cols, ["species"])
    kind_idx = find_column(cols, ["kind"])
    vtype_idx = find_column(cols, ["variant_type", "region_type"])

    counts = defaultdict(int)
    n_total = 0
    for line in in_handle:
        line = line.rstrip("\n")
        if not line:
            continue
        fields = line.split("\t")
        species_val = (
            fields[species_idx] if species_idx is not None
            and species_idx < len(fields) else ""
        )
        if args.species and species_val != args.species:
            continue
        kind_val = (
            fields[kind_idx] if kind_idx is not None
            and kind_idx < len(fields) else ""
        )
        vtype_val = (
            fields[vtype_idx] if vtype_idx is not None
            and vtype_idx < len(fields) else "NA"
        )
        counts[(species_val, kind_val, vtype_val)] += 1
        n_total += 1
    in_handle.close()

    try:
        out_handle = open_text(args.output, "wt")
    except OSError as exc:
        log(f"ERROR: cannot open output {args.output}: {exc}")
        return 1

    try:
        out_handle.write("species\tkind\tvariant_type\tcount\n")
        for (species_val, kind_val, vtype_val), count in sorted(
            counts.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])
        ):
            out_handle.write(
                f"{species_val}\t{kind_val}\t{vtype_val}\t{count}\n"
            )
    finally:
        out_handle.close()

    log(
        f"{species_tag}Done. rows_read={n_total} groups={len(counts)} -> "
        f"{args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
