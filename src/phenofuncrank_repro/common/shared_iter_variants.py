"""Identical source function; caller module supplies its recorded policy bindings."""
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

def iter_variants(path, max_positions=None):
    """Yield (chrom, pos) from a variant TSV."""
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
