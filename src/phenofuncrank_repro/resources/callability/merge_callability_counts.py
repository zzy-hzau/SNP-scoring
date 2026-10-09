#!/usr/bin/env python3
"Merge per-chromosome callability TSV files."

import argparse
import glob
import os
import sys


from phenofuncrank_repro.resources.utils.common import *


def parse_args(argv=None):
    """Build the argparse argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple callability TSV files into one, sorted by "
            "Chr, Pos and deduplicated."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inputs",
        required=True,
        help="Comma-separated list of input TSV paths, or a glob pattern.",
    )
    parser.add_argument(
        "--output", required=True, help="Output merged TSV path."
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
    """Sort key placing chr1, chr2 before chr10, chr11."""
    s = str(chrom)
    stripped = s[3:] if s.lower().startswith("chr") else s
    if stripped.isdigit():
        return (0, int(stripped), "")
    return (1, 0, s)


def resolve_inputs(spec):
    """Resolve a comma-separated list or glob pattern into a file list."""
    if "," in spec:
        paths = [p.strip() for p in spec.split(",") if p.strip()]
    else:
        paths = sorted(glob.glob(spec))
        if not paths:
            paths = [spec]
    return paths


def main(argv=None):
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    if args.species is None:
        args.species = config.get("species")

    species_tag = f"[{args.species}] " if args.species else ""

    paths = resolve_inputs(args.inputs)
    if not paths:
        log(f"ERROR: no input files matched {args.inputs!r}")
        return 1

    log(f"{species_tag}Merging {len(paths)} callability TSVs -> {args.output}")

    header = None
    rows = {}  
    for path in paths:
        try:
            in_handle = open_text(path, "rt")
        except OSError as exc:
            log(f"  WARNING: cannot open {path}: {exc}")
            continue
        try:
            local_header = in_handle.readline().rstrip("\n")
            if not local_header:
                continue
            if header is None:
                header = local_header
            cols = header.split("\t")
            try:
                chr_idx = cols.index("Chr")
                pos_idx = cols.index("Pos")
            except ValueError:
                log(f"  WARNING: {path} missing Chr/Pos columns; skipping")
                continue
            for line in in_handle:
                line = line.rstrip("\n")
                if not line:
                    continue
                fields = line.split("\t")
                while len(fields) < len(cols):
                    fields.append("")
                try:
                    pos = int(fields[pos_idx])
                except ValueError:
                    continue
                key = (fields[chr_idx], pos)
                if key not in rows:
                    rows[key] = fields
        finally:
            in_handle.close()

    sorted_keys = sorted(rows.keys(), key=lambda k: (chrom_sort_key(k[0]), k[1]))

    try:
        out_handle = open_text(args.output, "wt")
    except OSError as exc:
        log(f"ERROR: cannot open output {args.output}: {exc}")
        return 1

    try:
        if header:
            out_handle.write(header + "\n")
        for key in sorted_keys:
            out_handle.write("\t".join(rows[key]) + "\n")
    finally:
        out_handle.close()

    log(
        f"{species_tag}Done. merged_rows={len(sorted_keys)} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
