#!/usr/bin/env python3
"Build a sorted index TSV from candidate positions."

import argparse
import os
import sys


from phenofuncrank_repro.resources.utils.common import *


def parse_args(argv=None):
    """Build the argparse argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a sorted index TSV from candidate positions. Adds an "
            "snp_id column (Chr:Pos:Ref:Alt) if missing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True, help="Input candidate positions TSV file."
    )
    parser.add_argument(
        "--output", required=True, help="Output sorted TSV file."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML configuration file (currently unused).",
    )
    parser.add_argument(
        "--species",
        default=None,
        help="Optional species identifier (used for logging only).",
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


def chrom_sort_key(chrom):
    """Return a sort key that places chr1, chr2 before chr10, chr11."""
    s = str(chrom)

    stripped = s[3:] if s.lower().startswith("chr") else s
    if stripped.isdigit():
        return (0, int(stripped), "")

    return (1, 0, s)


def main(argv=None):
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    if args.species is None:
        args.species = config.get("species")

    species_tag = f"[{args.species}] " if args.species else ""
    log(f"{species_tag}Building candidate index from {args.input}")

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

    try:
        chr_idx = cols.index("Chr")
    except ValueError:
        log("ERROR: input TSV must have a 'Chr' column")
        in_handle.close()
        return 1
    try:
        pos_idx = cols.index("Pos")
    except ValueError:
        log("ERROR: input TSV must have a 'Pos' column")
        in_handle.close()
        return 1

    if "snp_id" in cols:
        snp_idx = cols.index("snp_id")
    else:
        snp_idx = None
        try:
            ref_idx = cols.index("Ref")
            alt_idx = cols.index("Alt")
        except ValueError:
            log("ERROR: input TSV must have 'Ref' and 'Alt' columns if 'snp_id' is absent")
            in_handle.close()
            return 1
        cols = ["snp_id"] + cols
        snp_idx = 0

    rows = []
    for line in in_handle:
        line = line.rstrip("\n")
        if not line:
            continue
        fields = line.split("\t")
        while len(fields) < len(cols) - 1:
            fields.append("")
        try:
            pos = int(fields[pos_idx])
        except (ValueError, IndexError):
            continue
        chrom = fields[chr_idx]
        if snp_idx is None:
            ref = fields[ref_idx]
            alt = fields[alt_idx]
            snp_id = f"{chrom}:{pos}:{ref}:{alt}"
            new_row = [snp_id] + fields
        else:
            new_row = fields
        rows.append((chrom_sort_key(chrom), pos, chrom, new_row))
    in_handle.close()

    rows.sort(key=lambda r: (r[0], r[1]))

    try:
        out_handle = open_text(args.output, "wt")
    except OSError as exc:
        log(f"ERROR: cannot open output {args.output}: {exc}")
        return 1

    try:
        out_handle.write("\t".join(cols) + "\n")
        for _key, _pos, _chrom, row in rows:

            if len(row) < len(cols):
                row = row + [""] * (len(cols) - len(row))
            elif len(row) > len(cols):
                row = row[:len(cols)]
            out_handle.write("\t".join(str(x) for x in row) + "\n")
    finally:
        out_handle.close()

    log(
        f"{species_tag}Done. rows_indexed={len(rows)} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
