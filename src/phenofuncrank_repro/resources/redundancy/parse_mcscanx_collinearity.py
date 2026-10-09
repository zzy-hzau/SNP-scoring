#!/usr/bin/env python3
"Parse an MCScanX ``.collinearity`` file into a tidy TSV of syntenic gene pairs."
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

from phenofuncrank_repro.resources.utils.common import log, open_text


_INDEX_ONE_TOKEN = re.compile(r"^\d+-\d+:?$")

_INDEX_FIRST_HALF = re.compile(r"^\d+-$")


_INT_TOKEN = re.compile(r"^\d+:?$")

_BLOCK_HEADER = re.compile(r"^###\s+(\d+)\s*$")
_BLOCK_HEADER_ALT = re.compile(r"^##\s+Alignment\s+(\d+)\b")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--collinearity", required=True, type=Path,
                    help="MCScanX .collinearity output file.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output TSV of syntenic gene pairs.")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Emit only the first N pairs (toy/test mode).")
    return ap.parse_args(argv)


def parse_pair_line(tokens):
    if not tokens:
        return None, None, None, None


    start = 0
    if _INDEX_ONE_TOKEN.match(tokens[0]):
        start = 1
    elif _INDEX_FIRST_HALF.match(tokens[0]) and len(tokens) >= 2 and \
            _INT_TOKEN.match(tokens[1]):
        start = 2

    if len(tokens) < start + 2:
        return None, None, None, None

    gene_a = tokens[start]
    gene_b = tokens[start + 1]
    e_value = tokens[start + 2] if len(tokens) > start + 2 else "NA"
    score = tokens[start + 3] if len(tokens) > start + 3 else "NA"


    if e_value != "NA":
        try:
            float(e_value)
        except ValueError:
            e_value = "NA"
    if score != "NA":
        try:
            float(score)
        except ValueError:
            score = "NA"

    return gene_a, gene_b, e_value, score


def main(argv=None):
    args = parse_args(argv)

    if not args.collinearity.exists():
        raise SystemExit(
            f"[ERROR] collinearity file not found: {args.collinearity}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_blocks = 0
    n_pairs = 0

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(tmp, "w") as out:
        out.write("pair_id\tgene_a\tgene_b\te_value\tscore\n")
        pair_id = 0
        with open_text(args.collinearity) as fh:
            for line in fh:
                if not line.strip():
                    continue

                m = _BLOCK_HEADER.match(line)
                if m:
                    n_blocks += 1
                    continue
                m2 = _BLOCK_HEADER_ALT.match(line)
                if m2:
                    n_blocks += 1
                    continue

                if line.startswith("#"):
                    continue

                tokens = line.split()
                gene_a, gene_b, e_value, score = parse_pair_line(tokens)
                if gene_a is None or gene_a == gene_b:
                    continue


                a, b = (gene_a, gene_b) if gene_a <= gene_b else (gene_b, gene_a)

                pair_id += 1
                out.write(f"{pair_id}\t{a}\t{b}\t{e_value}\t{score}\n")
                n_pairs += 1

                if args.max_rows is not None and n_pairs >= args.max_rows:
                    break

    tmp.replace(args.output)

    log(f"Parsed {args.collinearity}: {n_blocks} blocks, {n_pairs} gene pairs")
    log(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
