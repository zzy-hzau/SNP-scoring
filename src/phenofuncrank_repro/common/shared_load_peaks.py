"""Identical source function; caller module supplies its recorded policy bindings."""
from __future__ import annotations
import argparse
import bisect
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
from phenofuncrank_repro.resources.utils.common import log, open_text, norm_chrom

def load_peaks(path):
    """Load a BED file into ``{chrom: [(start, end), ...]}``."""
    peaks = {}
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
            if end <= start:
                continue
            peaks.setdefault(chrom, []).append((start, end))
    return peaks
