#!/usr/bin/env python3
"Build 26-state one-hot encoded PCSD chromatin-state features."
from __future__ import annotations

import argparse
import bisect
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

from phenofuncrank_repro.resources.utils.common import log, open_text, norm_chrom


N_STATES = 26
STATE_COLS = [f"pcsd_E{i}" for i in range(1, N_STATES + 1)]
_STATE_NUM = re.compile(r"(\d+)\s*$")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--variants", required=True, type=Path,
                    help="TSV with Chr and Pos columns.")
    ap.add_argument("--pcsd-bed", required=True, type=Path,
                    help="BED with 4th column = state label (e.g. E1).")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output TSV with one-hot state features.")
    ap.add_argument("--max-positions", type=int, default=None,
                    help="Process only the first N variants (toy/test mode).")
    return ap.parse_args(argv)


def parse_state_label(label):
    """Extract the 1-based state index from a label like ``E1``, ``state_3``, ``7``.

    Returns an int in [1, N_STATES] or ``None`` when unparseable / out of range.
    """
    if label is None:
        return None
    m = _STATE_NUM.search(str(label).strip())
    if not m:
        return None
    n = int(m.group(1))
    if 1 <= n <= N_STATES:
        return n
    return None


def load_pcsd(path):
    by_chrom = {}
    n_loaded = 0
    n_bad_state = 0
    with open_text(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#") or \
                    line.startswith("track") or line.startswith("browser"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                parts = line.rstrip("\n").split()
            if len(parts) < 4:
                continue
            chrom = norm_chrom(parts[0])
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue
            if end <= start:
                continue
            state_idx = parse_state_label(parts[3])
            if state_idx is None:
                n_bad_state += 1
                continue
            by_chrom.setdefault(chrom, []).append((start, end, state_idx))
            n_loaded += 1

    for chrom in by_chrom:
        by_chrom[chrom].sort(key=lambda x: x[0])
    return by_chrom, n_loaded, n_bad_state


def build_index(pcsd):
    """Build ``{chrom: (starts_list, segments_list)}`` for binary search."""
    index = {}
    for chrom, segs in pcsd.items():
        starts = [s for s, _, _ in segs]
        index[chrom] = (starts, segs)
    return index


def lookup_state(index, chrom, pos):
    """Return the 1-based state index overlapping 1-based ``pos``, or ``None``.

    Uses ``bisect`` to find the rightmost segment with ``start <= pos0``; since
    PCSD segments partition the genome (non-overlapping), at most one segment can
    contain ``pos0``.
    """
    if chrom not in index:
        return None
    starts, segs = index[chrom]
    pos0 = pos - 1  
    i = bisect.bisect_right(starts, pos0) - 1
    if i < 0:
        return None
    s, e, state_idx = segs[i]
    if s <= pos0 < e:
        return state_idx
    return None


def main(argv=None):
    args = parse_args(argv)

    if not args.variants.exists():
        raise SystemExit(f"[ERROR] variants file not found: {args.variants}")
    if not args.pcsd_bed.exists():
        raise SystemExit(f"[ERROR] pcsd-bed file not found: {args.pcsd_bed}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    log(f"Loading PCSD segments from {args.pcsd_bed}")
    pcsd, n_loaded, n_bad_state = load_pcsd(args.pcsd_bed)
    index = build_index(pcsd)
    log(f"Loaded {n_loaded} segments across {len(pcsd)} chromosomes "
        f"(skipped {n_bad_state} with unparseable state)")

    n_total = 0
    n_missing = 0
    state_counts = [0] * (N_STATES + 1)  

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(tmp, "w") as out:
        out.write("Chr\tPos\t" + "\t".join(STATE_COLS) + "\tpcsd26_state_missing\n")
        for chrom, pos in iter_variants(args.variants, args.max_positions):
            state_idx = lookup_state(index, chrom, pos)
            onehot = [0] * N_STATES
            if state_idx is None:
                missing = 1
                n_missing += 1
            else:
                missing = 0
                onehot[state_idx - 1] = 1
                state_counts[state_idx] += 1
            out.write(f"{chrom}\t{pos}\t" + "\t".join(str(v) for v in onehot) +
                      f"\t{missing}\n")
            n_total += 1
    tmp.replace(args.output)

    miss_rate = n_missing / n_total if n_total else 0.0
    log(f"Wrote {n_total} variants to {args.output} "
        f"(missing={n_missing}, rate={miss_rate:.4f})")
    for i in range(1, N_STATES + 1):
        if state_counts[i]:
            log(f"  E{i}: {state_counts[i]} variants")
    return 0


from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_identical
from phenofuncrank_repro.common import shared_iter_variants as _profile_kernel_16
_bind_identical(globals(), _profile_kernel_16, ['iter_variants'])

if __name__ == "__main__":
    sys.exit(main())
