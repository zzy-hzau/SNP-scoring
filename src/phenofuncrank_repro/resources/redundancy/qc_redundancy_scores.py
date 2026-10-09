#!/usr/bin/env python3
"QC for redundancy score tables."
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent

from phenofuncrank_repro.resources.utils.common import log


VALID_MIN = -1.0
VALID_MAX = 1.0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, type=Path,
                    help="TSV with redundancy score columns.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output QC report (TSV).")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Process only the first N rows (toy/test mode).")
    return ap.parse_args(argv)


def find_score_columns(cols):
    """Return columns whose name ends in 'redundancy'."""
    return [c for c in cols if c.lower().endswith("redundancy")]


def score_stats(series):
    """Compute distribution stats for a numeric series."""
    vals = pd.to_numeric(series, errors="coerce")
    n_total = len(vals)
    n_nan = int(vals.isna().sum())
    finite = vals.dropna()
    n_finite = len(finite)
    if n_finite == 0:
        return {
            "n_total": n_total,
            "n_finite": 0,
            "min": "NA",
            "max": "NA",
            "mean": "NA",
            "std": "NA",
            "fraction_negative": "NA",
            "fraction_zero": "NA",
            "fraction_positive": "NA",
            "nan_count": n_nan,
            "out_of_range_count": 0,
        }
    arr = finite.to_numpy(dtype=float)
    n_neg = int((arr < 0).sum())
    n_zero = int((arr == 0).sum())
    n_pos = int((arr > 0).sum())
    n_oor = int(((arr < VALID_MIN) | (arr > VALID_MAX)).sum())
    return {
        "n_total": n_total,
        "n_finite": n_finite,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "fraction_negative": n_neg / n_finite,
        "fraction_zero": n_zero / n_finite,
        "fraction_positive": n_pos / n_finite,
        "nan_count": n_nan,
        "out_of_range_count": n_oor,
    }


def fmt(v):
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def main(argv=None):
    args = parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"[ERROR] input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, sep="\t", dtype=str,
                     nrows=args.max_rows)
    score_cols = find_score_columns(df.columns)
    if not score_cols:
        raise SystemExit(
            f"[ERROR] no '*redundancy' columns found in input; "
            f"got {list(df.columns)}"
        )
    has_species = "species" in df.columns

    lines = []
    lines.append("# QC report: redundancy scores")
    lines.append(f"# input\t{args.input}")
    lines.append(f"# rows\t{len(df)}")
    lines.append(f"# score_columns\t{','.join(score_cols)}")
    lines.append("scope\tscore_column\tmetric\tvalue")


    for col in score_cols:
        stats = score_stats(df[col])
        for metric, value in stats.items():
            lines.append(f"overall\t{col}\t{metric}\t{fmt(value)}")


    if has_species:
        for sp, sub in df.groupby("species"):
            for col in score_cols:
                stats = score_stats(sub[col])
                for metric, value in stats.items():
                    lines.append(f"species={sp}\t{col}\t{metric}\t{fmt(value)}")

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(tmp, "w") as out:
        out.write("\n".join(lines) + "\n")
    tmp.replace(args.output)

    log(f"QC report written to {args.output}")
    log(f"  rows={len(df)} score_columns={score_cols} species_column={has_species}")


    problems = 0
    for col in score_cols:
        stats = score_stats(df[col])
        if stats["out_of_range_count"] > 0:
            problems += 1
            log(f"WARNING: column '{col}' has {stats['out_of_range_count']} "
                f"values outside [{VALID_MIN}, {VALID_MAX}]")
        if stats["nan_count"] > 0:
            log(f"NOTE: column '{col}' has {stats['nan_count']} NaN values "
                f"(expected 0 for genes without paralogs which are set to 0)")
    if problems == 0:
        log("Range check OK: all redundancy scores within [-1, 1]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
