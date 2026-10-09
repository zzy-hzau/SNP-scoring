#!/usr/bin/env python3
"Compute binary overlap of variants with regulatory peaks."
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
                    help="Output TSV with in_peak flag.")
    ap.add_argument("--max-positions", type=int, default=None,
                    help="Process only the first N variants (toy/test mode).")
    return ap.parse_args(argv)


def build_interval_index(peaks):
    """Build, per chromosome, sorted starts and the prefix-max of ends.

    Returns ``{chrom: (starts_list, max_ends_list)}`` where both lists are
    sorted by ``start``.  ``max_ends_list[i]`` is the maximum ``end`` among the
    first ``i+1`` intervals, enabling O(log n) interval-stabbing queries.
    """
    index = {}
    for chrom, ivs in peaks.items():
        ivs_sorted = sorted(ivs, key=lambda x: x[0])
        starts = [s for s, _ in ivs_sorted]
        max_ends = []
        cur_max = -1
        for _, e in ivs_sorted:
            if e > cur_max:
                cur_max = e
            max_ends.append(cur_max)
        index[chrom] = (starts, max_ends)
    return index


def is_in_peak(index, chrom, pos):
    """Return True if 1-based ``pos`` falls inside any peak on ``chrom``."""
    if chrom not in index:
        return False
    starts, max_ends = index[chrom]
    pos0 = pos - 1  

    i = bisect.bisect_right(starts, pos0) - 1
    if i < 0:
        return False

    return max_ends[i] > pos0


def iter_variants(path, max_positions=None):
    """Yield (chrom, pos, raw_line) from a variant TSV."""
    with open_text(path) as fh:
        header = fh.readline().rstrip("\n")
        cols = header.split("\t")
        try:
            chr_idx = cols.index("Chr")
            pos_idx = cols.index("Pos")
        except ValueError:
            raise SystemExit(
                f"[ERROR] variants file must have 'Chr' and 'Pos' columns; "
                f"got {cols}"
            )
        n = 0
        for line in fh:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            chrom = norm_chrom(parts[chr_idx])
            try:
                pos = int(parts[pos_idx])
            except ValueError:
                continue
            yield chrom, pos
            n += 1
            if max_positions is not None and n >= max_positions:
                break


def main(argv=None):
    args = parse_args(argv)

    if not args.variants.exists():
        raise SystemExit(f"[ERROR] variants file not found: {args.variants}")
    if not args.peaks.exists():
        raise SystemExit(f"[ERROR] peaks file not found: {args.peaks}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    log(f"Loading peaks from {args.peaks}")
    peaks = load_peaks(args.peaks)
    index = build_interval_index(peaks)
    log(f"Loaded {sum(len(v) for v in peaks.values())} peaks across "
        f"{len(peaks)} chromosomes")

    n_total = 0
    n_in = 0
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(tmp, "w") as out:
        out.write("Chr\tPos\tin_peak\n")
        for chrom, pos in iter_variants(args.variants, args.max_positions):
            hit = 1 if is_in_peak(index, chrom, pos) else 0
            out.write(f"{chrom}\t{pos}\t{hit}\n")
            n_total += 1
            n_in += hit
    tmp.replace(args.output)

    rate = n_in / n_total if n_total else 0.0
    log(f"Wrote {n_total} variants to {args.output} "
        f"(in_peak={n_in}, rate={rate:.4f})")
    return 0


from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_identical
from phenofuncrank_repro.common import shared_load_peaks as _profile_kernel_17
_bind_identical(globals(), _profile_kernel_17, ['load_peaks'])

if __name__ == "__main__":
    sys.exit(main())
