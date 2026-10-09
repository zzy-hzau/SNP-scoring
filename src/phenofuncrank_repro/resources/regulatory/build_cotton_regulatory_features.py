#!/usr/bin/env python3
"Build the complete cotton regulatory feature table."
from __future__ import annotations

import argparse
import sys
from pathlib import Path


_HERE = Path(__file__).resolve().parent

from phenofuncrank_repro.resources.regulatory.intersect_variants_with_peaks import load_peaks, build_interval_index, is_in_peak, iter_variants
from phenofuncrank_repro.resources.regulatory.distance_to_nearest_peak import build_sorted_index, distance_to_nearest
from phenofuncrank_repro.resources.utils.common import log


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--variants", required=True, type=Path,
                    help="TSV with Chr and Pos columns.")
    ap.add_argument("--atac-peaks", required=True, type=Path,
                    help="BED file of ATAC-seq peaks.")
    ap.add_argument("--h3k27ac-peaks", required=True, type=Path,
                    help="BED file of H3K27ac peaks.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output TSV with combined regulatory features.")
    ap.add_argument("--max-positions", type=int, default=None,
                    help="Process only the first N variants (toy/test mode).")
    return ap.parse_args(argv)


def build_track(peaks_path):
    """Load peaks and build both the interval (overlap) and sorted (distance)
    indices for one track."""
    peaks = load_peaks(peaks_path)
    interval_index = build_interval_index(peaks)
    sorted_index = build_sorted_index(peaks)
    n_peaks = sum(len(v) for v in peaks.values())
    n_chroms = len(peaks)
    return interval_index, sorted_index, n_peaks, n_chroms


def main(argv=None):
    args = parse_args(argv)

    if not args.variants.exists():
        raise SystemExit(f"[ERROR] variants file not found: {args.variants}")
    if not args.atac_peaks.exists():
        raise SystemExit(f"[ERROR] atac-peaks file not found: {args.atac_peaks}")
    if not args.h3k27ac_peaks.exists():
        raise SystemExit(
            f"[ERROR] h3k27ac-peaks file not found: {args.h3k27ac_peaks}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    log("Loading ATAC peaks")
    atac_interval, atac_sorted, n_atac, n_atac_chroms = build_track(args.atac_peaks)
    log(f"  ATAC: {n_atac} peaks across {n_atac_chroms} chromosomes")

    log("Loading H3K27ac peaks")
    h3k_interval, h3k_sorted, n_h3k, n_h3k_chroms = build_track(args.h3k27ac_peaks)
    log(f"  H3K27ac: {n_h3k} peaks across {n_h3k_chroms} chromosomes")

    n_total = 0
    n_atac_in = 0
    n_h3k_in = 0

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(tmp, "w") as out:
        out.write("Chr\tPos\tatac_in_peak\tatac_dist_to_peak\t"
                  "h3k27ac_in_peak\th3k27ac_dist_to_peak\n")
        for chrom, pos in iter_variants(args.variants, args.max_positions):
            atac_in = 1 if is_in_peak(atac_interval, chrom, pos) else 0
            atac_dist = distance_to_nearest(atac_sorted, chrom, pos)
            atac_dist_str = "NA" if atac_dist is None else str(atac_dist)

            h3k_in = 1 if is_in_peak(h3k_interval, chrom, pos) else 0
            h3k_dist = distance_to_nearest(h3k_sorted, chrom, pos)
            h3k_dist_str = "NA" if h3k_dist is None else str(h3k_dist)

            out.write(f"{chrom}\t{pos}\t{atac_in}\t{atac_dist_str}\t"
                      f"{h3k_in}\t{h3k_dist_str}\n")
            n_total += 1
            n_atac_in += atac_in
            n_h3k_in += h3k_in
    tmp.replace(args.output)

    atac_rate = n_atac_in / n_total if n_total else 0.0
    h3k_rate = n_h3k_in / n_total if n_total else 0.0

    log(f"Wrote {n_total} variants to {args.output}")
    log(f"  atac_in_peak={n_atac_in} (rate={atac_rate:.4f})")
    log(f"  h3k27ac_in_peak={n_h3k_in} (rate={h3k_rate:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
