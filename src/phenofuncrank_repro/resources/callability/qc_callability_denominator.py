#!/usr/bin/env python3
"QC the AN_obs / callable_n ratio from a callability TSV."

import argparse
import math
import os
import statistics
import sys


from phenofuncrank_repro.resources.utils.common import *


def parse_args(argv=None):
    """Build the argparse argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "QC the AN_obs / callable_n ratio from a callability TSV. "
            "Reports mean, std, min, max, and flags outliers."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True, help="Input callability TSV file."
    )
    parser.add_argument(
        "--ploidy", type=int, default=2, help="Expected ploidy per sample."
    )
    parser.add_argument(
        "--output", required=True, help="Output QC report path (TSV)."
    )
    parser.add_argument(
        "--outlier-z",
        type=float,
        default=3.0,
        help="Positions whose ratio deviates from the mean by more than "
        "this many standard deviations are flagged as outliers.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML configuration file with 'ploidy' and 'species'.",
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


def main(argv=None):
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    if args.species is None:
        args.species = config.get("species")
    if "ploidy" in config and args.ploidy == 2:
        args.ploidy = int(config["ploidy"])

    species_tag = f"[{args.species}] " if args.species else ""
    log(f"{species_tag}QC'ing callability denominators from {args.input}")

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
        callable_idx = cols.index("callable_n")
        an_idx = cols.index("AN_obs")
    except ValueError:
        log("ERROR: input must have 'callable_n' and 'AN_obs' columns")
        in_handle.close()
        return 1

    ratios = []
    zero_callable = 0
    for line in in_handle:
        line = line.rstrip("\n")
        if not line:
            continue
        fields = line.split("\t")
        try:
            callable_n = int(fields[callable_idx])
            an_obs = int(fields[an_idx])
        except (ValueError, IndexError):
            continue
        if callable_n <= 0:
            zero_callable += 1
            continue
        ratios.append(an_obs / callable_n)
    in_handle.close()

    if not ratios:
        log("ERROR: no ratio observations; nothing to summarize")
        return 1

    mean_ratio = statistics.fmean(ratios)
    if len(ratios) > 1:
        std_ratio = statistics.stdev(ratios)
    else:
        std_ratio = 0.0
    min_ratio = min(ratios)
    max_ratio = max(ratios)
    expected_ratio = float(args.ploidy)

    if std_ratio > 0:
        outliers = sum(
            1 for r in ratios
            if abs(r - mean_ratio) > args.outlier_z * std_ratio
        )
    else:
        outliers = 0

    pct_within_5pct = (
        sum(
            1 for r in ratios
            if abs(r - expected_ratio) <= 0.05 * expected_ratio
        ) / len(ratios)
    )

    try:
        out_handle = open_text(args.output, "wt")
    except OSError as exc:
        log(f"ERROR: cannot open output {args.output}: {exc}")
        return 1

    try:
        out_handle.write("metric\tvalue\n")
        out_handle.write(f"species\t{args.species or ''}\n")
        out_handle.write(f"ploidy\t{args.ploidy}\n")
        out_handle.write(f"expected_ratio\t{expected_ratio}\n")
        out_handle.write(f"n_positions\t{len(ratios)}\n")
        out_handle.write(f"zero_callable_positions\t{zero_callable}\n")
        out_handle.write(f"mean_ratio\t{mean_ratio:.6f}\n")
        out_handle.write(f"std_ratio\t{std_ratio:.6f}\n")
        out_handle.write(f"min_ratio\t{min_ratio:.6f}\n")
        out_handle.write(f"max_ratio\t{max_ratio:.6f}\n")
        out_handle.write(f"outlier_z_threshold\t{args.outlier_z}\n")
        out_handle.write(f"outlier_count\t{outliers}\n")
        out_handle.write(
            f"pct_within_5pct_of_expected\t{pct_within_5pct:.6f}\n"
        )

        passed = (
            abs(mean_ratio - expected_ratio) <= 0.10 * expected_ratio
            and std_ratio <= 0.25 * expected_ratio
        )
        out_handle.write(f"qc_pass\t{passed}\n")
    finally:
        out_handle.close()

    status = "PASS" if passed else "FAIL"
    log(
        f"{species_tag}QC {status}: mean_ratio={mean_ratio:.4f} "
        f"std={std_ratio:.4f} outliers={outliers} -> {args.output}"
    )
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
