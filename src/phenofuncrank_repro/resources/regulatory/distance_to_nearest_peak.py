#!/usr/bin/env python3
"Compute the distance from each variant to the nearest regulatory peak."
from __future__ import annotations

import argparse
import bisect
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

from phenofuncrank_repro.resources.utils.common import log, open_text, norm_chrom


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--variants", required=True, type=Path,
                    help="TSV with Chr and Pos columns.")
    ap.add_argument("--peaks", required=True, type=Path,
                    help="BED file of peaks (chrom, start, end, ...).")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output TSV with distance to nearest peak.")
    ap.add_argument("--max-positions", type=int, default=None,
                    help="Process only the first N variants (toy/test mode).")
    return ap.parse_args(argv)


def build_sorted_index(peaks):
    """Build, per chromosome, sorted starts, matching ends, and prefix-max ends.

    Returns ``{chrom: (starts_list, ends_list, max_ends_list)}``.
    """
    index = {}
    for chrom, ivs in peaks.items():
        ivs_sorted = sorted(ivs, key=lambda x: x[0])
        starts = [s for s, _ in ivs_sorted]
        ends = [e for _, e in ivs_sorted]
        max_ends = []
        cur_max = -1
        for e in ends:
            if e > cur_max:
                cur_max = e
            max_ends.append(cur_max)
        index[chrom] = (starts, ends, max_ends)
    return index


def distance_to_nearest(index, chrom, pos):
    """Distance from 1-based ``pos`` to the nearest peak on ``chrom``.

    Returns 0 if the variant is inside a peak, or ``None`` if the chromosome has
    no peaks at all.
    """
    if chrom not in index:
        return None
    starts, ends, max_ends = index[chrom]
    pos0 = pos - 1  


    i = bisect.bisect_right(starts, pos0) - 1


    
    if i >= 0 and max_ends[i] > pos0:
        return 0

    best = None


    
    if i >= 0:
        left_end = ends[i]

        

        left_boundary = max_ends[i] - 1  
        gap = pos0 - left_boundary
        if best is None or gap < best:
            best = gap


    j = i + 1
    if j < len(starts):
        gap = starts[j] - pos0
        if best is None or gap < best:
            best = gap

    return best if best is not None else 0


def main(argv=None):
    args = parse_args(argv)

    if not args.variants.exists():
        raise SystemExit(f"[ERROR] variants file not found: {args.variants}")
    if not args.peaks.exists():
        raise SystemExit(f"[ERROR] peaks file not found: {args.peaks}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    log(f"Loading peaks from {args.peaks}")
    peaks = load_peaks(args.peaks)
    index = build_sorted_index(peaks)
    log(f"Loaded {sum(len(v) for v in peaks.values())} peaks across "
        f"{len(peaks)} chromosomes")

    n_total = 0
    n_inside = 0
    n_no_peaks = 0
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(tmp, "w") as out:
        out.write("Chr\tPos\tdist_to_nearest_peak\n")
        for chrom, pos in iter_variants(args.variants, args.max_positions):
            d = distance_to_nearest(index, chrom, pos)
            if d is None:
                out.write(f"{chrom}\t{pos}\tNA\n")
                n_no_peaks += 1
            else:
                out.write(f"{chrom}\t{pos}\t{d}\n")
                if d == 0:
                    n_inside += 1
            n_total += 1
    tmp.replace(args.output)

    log(f"Wrote {n_total} variants to {args.output} "
        f"(inside_peak={n_inside}, no_peaks_on_chrom={n_no_peaks})")
    return 0


from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_identical
from phenofuncrank_repro.common import shared_iter_variants as _profile_kernel_16
_bind_identical(globals(), _profile_kernel_16, ['iter_variants'])
from phenofuncrank_repro.common import shared_load_peaks as _profile_kernel_17
_bind_identical(globals(), _profile_kernel_17, ['load_peaks'])

if __name__ == "__main__":
    sys.exit(main())
