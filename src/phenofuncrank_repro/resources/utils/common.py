#!/usr/bin/env python3
"Shared utility functions for the plant-cadd-supplementary-methods pipeline."

from __future__ import annotations

import gzip
import os
import re
import sys
from datetime import datetime
from typing import Dict, Iterator, Optional, Tuple

import yaml


BASES = set("ACGT")


RCMAP: Dict[str, str] = {
    "A": "T", "C": "G", "G": "C", "T": "A",
    "a": "t", "c": "g", "g": "c", "t": "a",
    "N": "N", "n": "n",
    "U": "A", "u": "a",
    "R": "Y", "Y": "R", "S": "S", "W": "W",
    "K": "M", "M": "K", "B": "V", "V": "B",
    "D": "H", "H": "D",
    "r": "y", "y": "r", "s": "s", "w": "w",
    "k": "m", "m": "k", "b": "v", "v": "b",
    "d": "h", "h": "d",
}


CODON_TABLE: Dict[str, str] = {

    "TTT": "F", "TTC": "F",

    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",

    "ATT": "I", "ATC": "I", "ATA": "I",

    "ATG": "M",

    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",

    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",

    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",

    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",

    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",

    "TAT": "Y", "TAC": "Y",

    "TAA": "*", "TAG": "*", "TGA": "*",

    "CAT": "H", "CAC": "H",

    "CAA": "Q", "CAG": "Q",

    "AAT": "N", "AAC": "N",

    "AAA": "K", "AAG": "K",

    "GAT": "D", "GAC": "D",

    "GAA": "E", "GAG": "E",

    "TGT": "C", "TGC": "C",

    "TGA": "*",

    "TGG": "W",

    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",

    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",

    "AGT": "S", "AGC": "S",
}


def reverse_complement(seq: str) -> str:
    if not seq:
        return ""
    out = []
    for base in reversed(seq):
        rc = RCMAP.get(base)
        if rc is None:

            out.append("N" if base.isupper() else "n")
        else:
            out.append(rc)
    return "".join(out)


def log(msg: str, file=None) -> None:
    """Emit a timestamped log line to ``stderr`` (or ``file``).

    The timestamp follows ISO 8601 with the local timezone offset so that
    log files are unambiguous across machines. Output is flushed
    immediately to remain visible when piped.

    Parameters
    ----------
    msg:
        Human-readable log message.
    file:
        Optional file-like object. Defaults to ``sys.stderr``.
    """
    stream = file if file is not None else sys.stderr
    ts = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"[{ts}] {msg}", file=stream, flush=True)


def open_text(path: str, mode: str = "rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def fasta_iter(path: str) -> Iterator[Tuple[str, str]]:
    """Iterate over FASTA records as ``(header, sequence)`` tuples.

    The header is returned without the leading ``">"`` character. Sequence
    lines are concatenated and upper-cased to simplify downstream codon
    translation. Multi-line records are supported.

    Parameters
    ----------
    path:
        Path to a FASTA file (optionally gzip-compressed).

    Yields
    ------
    tuple of (str, str)
        ``(header, sequence)`` for each record.
    """
    header: Optional[str] = None
    chunks: list = []
    with open_text(path) as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, "".join(chunks)


def load_fasta(path: str, min_len: int = 0) -> Dict[str, str]:
    """Load a FASTA file into a ``{header: sequence}`` dictionary.

    Sequences shorter than ``min_len`` are skipped; this is used by the
    genome-standardization QC step to drop tiny contigs before
    annotation processing.

    Parameters
    ----------
    path:
        Path to a FASTA file (optionally gzip-compressed).
    min_len:
        Minimum sequence length to retain. Defaults to ``0`` (keep all).

    Returns
    -------
    dict
        Mapping from FASTA header to upper-cased sequence.
    """
    out: Dict[str, str] = {}
    for header, seq in fasta_iter(path):
        if len(seq) < min_len:
            continue
        out[header] = seq.upper()
    return out


def parse_gff_attrs(attr: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not attr:
        return out
    for field in attr.strip().split(";"):
        field = field.strip()
        if not field:
            continue
        if "=" in field:
            key, _, value = field.partition("=")
            out[key] = value
    return out


_CHROM_PREFIX_RE = re.compile(r"^(?:chr|Chr|CHR|scaffold|Scaffold)(.+)$")


def norm_chrom(x: str) -> str:
    """Normalize a chromosome name for cross-source comparison.

    Strips a leading ``chr``/``Chr``/``CHR`` prefix and returns the
    remainder. ``chr01`` becomes ``01`` and ``chr1`` becomes ``1``.
    Scaffold names are left untouched (the regex only strips the literal
    ``scaffold`` prefix if present). ``None`` or empty input returns an
    empty string to keep downstream lookups safe.

    Parameters
    ----------
    x:
        Raw chromosome identifier.

    Returns
    -------
    str
        Normalized chromosome identifier.
    """
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    m = _CHROM_PREFIX_RE.match(s)
    if m:
        return m.group(1)
    return s


_TRANSCRIPT_SUFFIX_RE = re.compile(r"(?:[._]T\d+|\.t\d+|\.\d+)+$")


def clean_gene_id(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""

    
    s = re.sub(r"_T\d+$", "", s)
    s = re.sub(r"\.t\d+$", "", s, flags=re.IGNORECASE)

    

    s = re.sub(r"(\.\d+)+$", "", s)
    return s


def load_yaml_config(path: str) -> dict:
    """Load a YAML config file into a Python dictionary.

    Uses ``yaml.safe_load`` to avoid arbitrary-object construction from
    untrusted config files. A missing file raises ``FileNotFoundError``
    rather than silently returning an empty dict, so config errors are
    surfaced early.

    Parameters
    ----------
    path:
        Path to a YAML file.

    Returns
    -------
    dict
        Parsed YAML content. Returns ``{}`` for an empty file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if data is not None else {}
