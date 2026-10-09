#!/usr/bin/env python3
"QC: check PCSD BED coordinates against a reference FASTA index (.fai)."
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

from phenofuncrank_repro.resources.utils.common import log, open_text, norm_chrom


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pcsd-bed", required=True, type=Path,
                    help="PCSD BED file (Chr, Start, End, [State]).")
    ap.add_argument("--fai", required=True, type=Path,
                    help="Reference FASTA index (.fai).")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output QC report (TSV).")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Process only the first N PCSD segments (toy/test mode).")
    return ap.parse_args(argv)


def load_fai(path):
    """Load a .fai file into ``{chrom: length}``."""
    lengths = {}
    with open_text(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            chrom = norm_chrom(parts[0])
            try:
                length = int(parts[1])
            except ValueError:
                continue
            lengths[chrom] = length
    return lengths


def stream_pcsd(path):
    """Yield (chrom, start, end) for each PCSD segment."""
    with open_text(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#") or \
                    line.startswith("track") or line.startswith("browser"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                parts = line.rstrip("\n").split()
            if len(parts) < 3:
                continue
            chrom = norm_chrom(parts[0])
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue
            yield chrom, start, end


def main(argv=None):
    args = parse_args(argv)

    if not args.pcsd_bed.exists():
        raise SystemExit(f"[ERROR] pcsd-bed not found: {args.pcsd_bed}")
    if not args.fai.exists():
        raise SystemExit(f"[ERROR] fai not found: {args.fai}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    log(f"Loading reference lengths from {args.fai}")
    fai = load_fai(args.fai)
    log(f"  {len(fai)} chromosomes in .fai")

    n_total = 0
    n_out_of_range = 0

    per_chrom = {}

    for chrom, start, end in stream_pcsd(args.pcsd_bed):
        n_total += 1
        rec = per_chrom.setdefault(chrom, {"segment_count": 0, "max_end": 0})
        rec["segment_count"] += 1
        if end > rec["max_end"]:
            rec["max_end"] = end

        chrom_len = fai.get(chrom)
        if chrom_len is None:
            n_out_of_range += 1
        elif start < 0 or end > chrom_len:
            n_out_of_range += 1

        if args.max_rows is not None and n_total >= args.max_rows:
            break

    n_chroms_pcsd = len(per_chrom)
    n_chroms_only_pcsd = sum(1 for c in per_chrom if c not in fai)

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(tmp, "w") as out:
        out.write("# QC report: PCSD coordinate compatibility\n")
        out.write(f"# pcsd_bed\t{args.pcsd_bed}\n")
        out.write(f"# fai\t{args.fai}\n")
        out.write("metric\tvalue\n")
        out.write(f"total_segments\t{n_total}\n")
        out.write(f"out_of_range_segments\t{n_out_of_range}\n")
        out.write(f"n_chroms_in_pcsd\t{n_chroms_pcsd}\n")
        out.write(f"n_chroms_in_fai\t{len(fai)}\n")
        out.write(f"n_chroms_only_in_pcsd\t{n_chroms_only_pcsd}\n")
        out.write("\n")
        out.write("chrom\tfai_length\tsegment_count\tmax_end\tmax_end_in_range\n")
        for chrom in sorted(per_chrom, key=lambda c: (c.isdigit(), c)):
            rec = per_chrom[chrom]
            chrom_len = fai.get(chrom)
            if chrom_len is None:
                fai_len_str = "NA"
                in_range = "NA"
            else:
                fai_len_str = str(chrom_len)
                in_range = "1" if rec["max_end"] <= chrom_len else "0"
            out.write(f"{chrom}\t{fai_len_str}\t{rec['segment_count']}\t"
                      f"{rec['max_end']}\t{in_range}\n")
    tmp.replace(args.output)

    log(f"QC report written to {args.output}")
    log(f"  total_segments={n_total} out_of_range={n_out_of_range} "
        f"chroms_in_pcsd={n_chroms_pcsd} chroms_only_in_pcsd={n_chroms_only_pcsd}")
    if n_out_of_range > 0:
        log(f"WARNING: {n_out_of_range} PCSD segments are out of range relative "
            f"to the reference .fai")
    else:
        log("All PCSD segments are within reference bounds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
