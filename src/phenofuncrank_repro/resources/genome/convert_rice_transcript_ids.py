#!/usr/bin/env python3
"Convert rice transcript-level IDs to gene-level IDs."

import argparse
import os
import re
import sys


from phenofuncrank_repro.resources.utils.common import *


def parse_args(argv=None):
    """Build the argparse argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert rice transcript-level IDs in a TSV to gene-level IDs "
            "by stripping isoform suffixes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input TSV file with a column containing gene/transcript IDs.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output TSV file with the converted IDs.",
    )
    parser.add_argument(
        "--id-column",
        default="gene_id",
        help="Name of the column containing the IDs to convert.",
    )
    parser.add_argument(
        "--strip-pattern",
        default=r"\.\d+$",
        help="Regex pattern matching the isoform suffix to strip.",
    )
    parser.add_argument(
        "--normalize-rice",
        action="store_true",
        help="Normalize RAP OsXXt identifiers to OsXXg and strip supported version suffixes.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML configuration file with a 'species' field.",
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


def convert_id(raw, strip_re, normalize_rice):
    """Strip isoform suffix and (optionally) normalize rice gene IDs."""
    if raw is None:
        return ""
    raw = str(raw).strip()
    stripped = strip_re.sub("", raw)
    if normalize_rice:
        cleaned = clean_gene_id(stripped)

        match = re.fullmatch(r"(Os\d{2})t(\d{7})(?:-\d+)?", cleaned)
        if match:
            return match.group(1) + "g" + match.group(2)
        return cleaned
    return stripped


def main(argv=None):
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    if args.species is None:
        args.species = config.get("species")
    if config.get("normalize_rice") and not args.normalize_rice:
        args.normalize_rice = True

    species_tag = f"[{args.species}] " if args.species else ""
    log(f"{species_tag}Converting IDs in {args.input} -> {args.output}")

    try:
        strip_re = re.compile(args.strip_pattern)
    except re.error as exc:
        log(f"ERROR: invalid --strip-pattern regex: {exc}")
        return 1

    try:
        in_handle = open_text(args.input, "rt")
    except OSError as exc:
        log(f"ERROR: cannot open input {args.input}: {exc}")
        return 1

    try:
        out_handle = open_text(args.output, "wt")
    except OSError as exc:
        log(f"ERROR: cannot open output {args.output}: {exc}")
        in_handle.close()
        return 1

    try:
        header = in_handle.readline().rstrip("\n")
        if not header:
            log("ERROR: empty input file")
            return 1
        cols = header.split("\t")
        try:
            id_idx = cols.index(args.id_column)
        except ValueError:
            log(
                f"ERROR: id column {args.id_column!r} not found in "
                f"header: {cols}"
            )
            return 1

        out_handle.write("\t".join(cols) + "\n")
        n_rows = 0
        for line in in_handle:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            while len(fields) < len(cols):
                fields.append("")
            if id_idx < len(fields):
                fields[id_idx] = convert_id(
                    fields[id_idx], strip_re, args.normalize_rice
                )
            out_handle.write("\t".join(fields) + "\n")
            n_rows += 1
    finally:
        in_handle.close()
        out_handle.close()

    log(f"{species_tag}Done. rows_converted={n_rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
