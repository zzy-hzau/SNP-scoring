#!/usr/bin/env python3
"QC: verify that the reference amino acid in the MSA matches the annotation."
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

from phenofuncrank_repro.resources.utils.common import log, open_text


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--variant-msa-map", required=True, type=Path,
                    help="TSV from map_variant_to_msa_column.py.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output QC report (TSV).")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Process only the first N rows (toy/test mode).")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not args.variant_msa_map.exists():
        raise SystemExit(
            f"[ERROR] variant-msa-map not found: {args.variant_msa_map}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_matched = 0
    n_mismatched = 0
    n_unmapped = 0
    mismatches = []
    header_cols = []

    with open_text(args.variant_msa_map) as fh:
        header = fh.readline().rstrip("\n")
        header_cols = header.split("\t")
        col_idx = {c: i for i, c in enumerate(header_cols)}
        if "ref_aa_match" not in col_idx:
            raise SystemExit(
                f"[ERROR] variant-msa-map must have a 'ref_aa_match' column; "
                f"got {header_cols}"
            )
        ridx = col_idx["ref_aa_match"]

        for line in fh:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            n_total += 1
            val = parts[ridx].strip() if ridx < len(parts) else "NA"
            if val == "NA":
                n_unmapped += 1
            elif val in ("1", "True", "true", "TRUE"):
                n_matched += 1
            elif val in ("0", "False", "false", "FALSE"):
                n_mismatched += 1
                if len(mismatches) < 20:
                    mismatches.append(parts)
            else:
                n_unmapped += 1
            if args.max_rows is not None and n_total >= args.max_rows:
                break

    match_rate = n_matched / n_total if n_total else 0.0

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(tmp, "w") as out:
        out.write("# QC report: ref_aa_match verification\n")
        out.write(f"# input\t{args.variant_msa_map}\n")
        out.write("metric\tvalue\n")
        out.write(f"total_variants\t{n_total}\n")
        out.write(f"matched\t{n_matched}\n")
        out.write(f"mismatched\t{n_mismatched}\n")
        out.write(f"unmapped\t{n_unmapped}\n")
        out.write(f"match_rate\t{match_rate:.6f}\n")
        out.write("\n")
        out.write(f"# First {min(20, len(mismatches))} mismatches\n")
        if header_cols:
            out.write("\t".join(header_cols) + "\n")
        for row in mismatches:
            out.write("\t".join(row) + "\n")
    tmp.replace(args.output)

    log(f"QC report written to {args.output}")
    log(f"  total={n_total} matched={n_matched} "
        f"mismatched={n_mismatched} unmapped={n_unmapped} "
        f"match_rate={match_rate:.4f}")
    if n_mismatched > 0:
        log(f"WARNING: {n_mismatched} reference AA mismatches detected "
            f"(first {len(mismatches)} listed in report)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
