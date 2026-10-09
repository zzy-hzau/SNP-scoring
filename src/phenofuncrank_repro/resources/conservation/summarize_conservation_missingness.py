#!/usr/bin/env python3
"Summarize missingness rates for conservation feature tables."
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

from phenofuncrank_repro.resources.utils.common import log, open_text


MISSING_THRESHOLD = 0.50


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, type=Path,
                    help="TSV with one or more <track>_missing columns.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output TSV missingness summary.")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Process only the first N rows (toy/test mode).")
    return ap.parse_args(argv)


def find_missing_columns(header_cols):
    """Return list of (track_name, column_index) for columns ending in '_missing'."""
    out = []
    for idx, col in enumerate(header_cols):
        if col.endswith("_missing"):
            track = col[: -len("_missing")]
            out.append((track, idx))
    return out


def summarize(path, max_rows=None):
    """Stream the TSV once, accumulating per-track totals and missing counts."""
    with open_text(path) as fh:
        header = fh.readline().rstrip("\n")
        cols = header.split("\t")
        miss_cols = find_missing_columns(cols)
        if not miss_cols:
            raise SystemExit(
                f"[ERROR] no '*_missing' columns found in header: {cols}"
            )

        totals = {track: 0 for track, _ in miss_cols}
        missing = {track: 0 for track, _ in miss_cols}
        n_rows = 0

        for line in fh:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            n_rows += 1
            for track, idx in miss_cols:
                totals[track] += 1
                if idx < len(parts):
                    val = parts[idx].strip()
                    if val in ("1", "True", "true", "TRUE"):
                        missing[track] += 1
            if max_rows is not None and n_rows >= max_rows:
                break

    summary = []
    for track, _ in miss_cols:
        total = totals[track]
        miss = missing[track]
        rate = miss / total if total else 0.0
        flag = 1 if rate > MISSING_THRESHOLD else 0
        summary.append((track, total, miss, rate, flag))
    return summary, n_rows


def main(argv=None):
    args = parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"[ERROR] input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    log(f"Summarizing conservation missingness from {args.input}")
    summary, n_rows = summarize(args.input, args.max_rows)

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(tmp, "w") as out:
        out.write("track\ttotal_positions\tmissing_count\tmissing_rate\t"
                  "high_missingness_flag\n")
        for track, total, miss, rate, flag in summary:
            out.write(f"{track}\t{total}\t{miss}\t{rate:.6f}\t{flag}\n")
    tmp.replace(args.output)

    log(f"Processed {n_rows} rows, {len(summary)} tracks; wrote {args.output}")
    flagged = [(t, r) for (t, _, _, r, f) in summary if f]
    if flagged:
        log(f"WARNING: {len(flagged)} track(s) exceed {MISSING_THRESHOLD:.0%} "
            f"missingness and need review:")
        for t, r in flagged:
            log(f"  - {t}: {r:.4f}")
    else:
        log("No tracks exceed the missingness threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
