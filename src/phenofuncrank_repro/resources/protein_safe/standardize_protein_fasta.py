#!/usr/bin/env python3
"Standardize a protein FASTA file."
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

from phenofuncrank_repro.resources.utils.common import log, open_text, fasta_iter


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, type=Path,
                    help="Input protein FASTA (may be gzipped).")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output cleaned protein FASTA.")
    ap.add_argument("--valid-aas", default="ACDEFGHIKLMNPQRSTVWY*",
                    help="Allowed amino-acid alphabet (default includes stop '*').")
    ap.add_argument("--max-nonstandard-frac", type=float, default=0.05,
                    help="Discard sequences whose non-standard AA fraction "
                         "exceeds this value (default 0.05).")
    ap.add_argument("--log-file", type=Path, default=None,
                    help="Optional file listing removed sequences and reasons.")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Process only the first N sequences (toy/test mode).")
    return ap.parse_args(argv)


def clean_header(header):
    h = header[1:].strip() if header.startswith(">") else header.strip()
    if not h:
        return ""
    return h.split()[0]


def nonstandard_fraction(seq, valid_set):
    if not seq:
        return 1.0
    bad = sum(1 for aa in seq if aa not in valid_set)
    return bad / len(seq)


def main(argv=None):
    args = parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"[ERROR] input FASTA not found: {args.input}")

    if not (0.0 <= args.max_nonstandard_frac <= 1.0):
        raise SystemExit(
            f"[ERROR] --max-nonstandard-frac must be in [0, 1], got "
            f"{args.max_nonstandard_frac}"
        )

    valid_set = set(args.valid_aas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)

    log(f"Standardizing protein FASTA: {args.input} -> {args.output}")
    log(f"Valid AAs: {args.valid_aas} "
        f"(max non-standard frac = {args.max_nonstandard_frac})")

    seen_seqs = set()
    seen_ids = set()
    removed = []
    n_in = 0
    n_out = 0
    n_dup_seq = 0
    n_dup_id = 0
    n_nonstd = 0
    n_empty = 0

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(tmp, "w") as out:
        for header, seq in fasta_iter(args.input):
            n_in += 1
            seq_id = clean_header(header)
            cleaned = seq.upper().strip()

            if args.max_rows is not None and n_in > args.max_rows:
                break

            if not cleaned:
                n_empty += 1
                removed.append((seq_id or "<empty>", "empty_sequence"))
                continue

            frac = nonstandard_fraction(cleaned, valid_set)
            if frac > args.max_nonstandard_frac:
                n_nonstd += 1
                removed.append((seq_id, f"nonstandard_fraction={frac:.4f}"))
                continue

            if cleaned in seen_seqs:
                n_dup_seq += 1
                removed.append((seq_id, "duplicate_sequence"))
                continue

            if seq_id in seen_ids:
                n_dup_id += 1
                removed.append((seq_id, "duplicate_id"))
                continue

            seen_seqs.add(cleaned)
            seen_ids.add(seq_id)
            out.write(f">{seq_id}\n{cleaned}\n")
            n_out += 1

    tmp.replace(args.output)

    if args.log_file is not None:
        with open(args.log_file, "w") as lf:
            lf.write("seq_id\treason\n")
            for seq_id, reason in removed:
                lf.write(f"{seq_id}\t{reason}\n")

    log(f"Input sequences:  {n_in}")
    log(f"Output sequences: {n_out}")
    log(f"Removed (non-standard AAs): {n_nonstd}")
    log(f"Removed (duplicate sequence): {n_dup_seq}")
    log(f"Removed (duplicate ID): {n_dup_id}")
    log(f"Removed (empty): {n_empty}")
    if args.log_file is not None:
        log(f"Removal log written to {args.log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
