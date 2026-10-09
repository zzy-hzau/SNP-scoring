#!/usr/bin/env python3
"Map bedGraph conservation tracks (phastCons / phyloP) to candidate SNP positions."
from __future__ import annotations

import argparse
import re
import sys
from collections import deque
from pathlib import Path

_HERE = Path(__file__).resolve().parent

from phenofuncrank_repro.resources.utils.common import log, open_text, norm_chrom


def norm_track_chrom(value: str) -> str:
    value = norm_chrom(value)
    if value.isdigit():
        return str(int(value))
    return value


def chromosome_order_key(value: str):
    text = norm_track_chrom(value)
    if text.isdigit():
        return ("", int(text), "")
    matched = re.fullmatch(r"([A-Za-z_]+)(\d+)(.*)", text)
    if matched:
        return (matched.group(1), int(matched.group(2)), matched.group(3))
    return (text, -1, "")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--snps", required=True, type=Path,
                    help="TSV with Chr and Pos columns (sorted by chrom, pos).")
    ap.add_argument("--bedgraph", required=True, type=Path,
                    help="bedGraph file: chrom, start, end, score (sorted).")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output TSV with mapped scores.")
    ap.add_argument("--track-name", required=True,
                    help='Track name used to name output columns, e.g. "phastCons".')
    ap.add_argument("--max-positions", type=int, default=None,
                    help="Process only the first N SNPs (toy/test mode).")
    return ap.parse_args(argv)


def iter_bedgraph(path):
    """Yield (chrom, start, end, score) tuples from a bedGraph file."""
    with open_text(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                parts = line.rstrip("\n").split()
            if len(parts) < 4:
                continue
            chrom = norm_track_chrom(parts[0])
            try:
                start = int(parts[1])
                end = int(parts[2])
                score = float(parts[3])
            except ValueError:
                continue
            yield chrom, start, end, score


def iter_snps(path, max_positions=None):
    """Yield (chrom, pos, raw_line) tuples from a SNP TSV.

    ``pos`` is the 1-based coordinate; bedGraph is 0-based half-open, so a SNP at
    1-based ``pos`` lies in an interval [start, end) when ``start <= pos-1 < end``.
    """
    with open_text(path) as fh:
        header = fh.readline().rstrip("\n")
        cols = header.split("\t")
        try:
            chr_idx = cols.index("Chr")
            pos_idx = cols.index("Pos")
        except ValueError:
            raise SystemExit(
                f"[ERROR] SNP file must have 'Chr' and 'Pos' columns; "
                f"got {cols}"
            )
        n = 0
        for line in fh:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            chrom = norm_track_chrom(parts[chr_idx])
            try:
                pos = int(parts[pos_idx])
            except ValueError:
                continue
            yield chrom, pos, line.rstrip("\n")
            n += 1
            if max_positions is not None and n >= max_positions:
                break


def map_snps_to_bedgraph(snps_path, bedgraph_path, track_name, max_positions=None):
    """Streaming sorted-merge of SNPs onto a bedGraph track.

    Returns a list of (chrom, pos, score_str, missing_int) tuples.  When multiple
    active intervals overlap a SNP, their scores are averaged.
    """
    bg_iter = iter_bedgraph(bedgraph_path)
    active = deque()
    current = None
    exhausted = False

    def advance_to(chrom, pos0):
        """Advance the bedGraph iterator so that all intervals overlapping
        ``pos0`` (0-based) on ``chrom`` are in ``active``."""
        nonlocal current, exhausted
        target_key = chromosome_order_key(chrom)


        while active and (chromosome_order_key(active[0][0]) < target_key or
                          (active[0][0] == chrom and active[0][2] <= pos0)):
            active.popleft()

        if exhausted:
            return

        while True:
            if current is None:
                try:
                    current = next(bg_iter)
                except StopIteration:
                    exhausted = True
                    return

            c, s, e, val = current


            if chromosome_order_key(c) < target_key or (c == chrom and e <= pos0):
                current = None
                continue


            if c == chrom and s <= pos0 < e:
                active.append(current)
                current = None
                continue


            break

    results = []
    for chrom, pos, _raw in iter_snps(snps_path, max_positions):
        pos0 = pos - 1  
        advance_to(chrom, pos0)

        vals = [v for (c, s, e, v) in active
                if c == chrom and s <= pos0 < e]
        if vals:
            score = sum(vals) / len(vals)
            results.append((chrom, pos, f"{score:.6g}", 0))
        else:
            results.append((chrom, pos, "NA", 1))
    return results


def main(argv=None):
    args = parse_args(argv)

    if not args.snps.exists():
        raise SystemExit(f"[ERROR] SNPs file not found: {args.snps}")
    if not args.bedgraph.exists():
        raise SystemExit(f"[ERROR] bedGraph file not found: {args.bedgraph}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    log(f"Mapping bedGraph '{args.bedgraph}' (track={args.track_name}) "
        f"to SNPs in '{args.snps}'")

    results = map_snps_to_bedgraph(
        args.snps, args.bedgraph, args.track_name, args.max_positions
    )

    score_col = f"{args.track_name}_score"
    miss_col = f"{args.track_name}_missing"

    n_total = len(results)
    n_missing = sum(1 for _, _, _, m in results if m)
    miss_rate = n_missing / n_total if n_total else 0.0

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(tmp, "w") as out:
        out.write(f"Chr\tPos\t{score_col}\t{miss_col}\n")
        for chrom, pos, score, miss in results:
            out.write(f"{chrom}\t{pos}\t{score}\t{miss}\n")
    tmp.replace(args.output)

    log(f"Wrote {n_total} positions to {args.output} "
        f"(missing={n_missing}, rate={miss_rate:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
