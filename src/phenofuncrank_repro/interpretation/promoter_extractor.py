"Promoter sequence extraction for motif scanning."

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from phenofuncrank_repro.interpretation.logging_utils import get_logger
from phenofuncrank_repro.interpretation.resources import reference_fasta_path

logger = get_logger(__name__)


@lru_cache(maxsize=16)
def _fasta_handle(path: str) -> Any:
    """Open and memoise a ``pysam.FastaFile`` handle.  ``None`` on failure."""
    try:
        import pysam
    except ImportError:
        logger.warning("pysam is not installed; promoter extraction disabled")
        return None
    try:
        return pysam.FastaFile(path)
    except Exception as exc:
        logger.warning("FastaFile open failed for %s: %s", path, exc)
        return None


def clear_cache() -> None:
    """Reset memoised FASTA handles (e.g. between tests)."""
    _fasta_handle.cache_clear()


def _normalise_chrom(chrom: str, references: tuple[str, ...]) -> str:
    """Map a variant ``chrom`` value to a contig present in the FASTA index.

    Tries several common naming conventions:

    * as-given (e.g. ``chr01``, ``Chr1``, ``1``, ``A01``)
    * stripped of any ``chr`` / ``Chr`` prefix
    * lower-cased, zero-padded to two digits with a ``chr`` prefix
    * letter-prefixed cotton / arabidopsis subgenome style (``A1`` → ``A01``)

    The first candidate that resolves to a real contig is returned; an empty
    string is returned if no candidate matches.
    """
    if not chrom:
        return ""
    contigs = set(references)
    raw = str(chrom).strip()
    stripped = raw.replace("Chr", "").replace("chr", "").lstrip("0") or "0"

    candidates = [
        raw,                            
        raw.lower(),                    
        raw.upper(),                    
        stripped,                       
        stripped.zfill(2),              
        f"chr{stripped.zfill(2)}",      
        f"Chr{stripped.zfill(2)}",      
        f"chr{stripped}",               
        f"Chr{stripped}",               
    ]


    m = re.match(r"^([A-Za-z]+)(\d+)$", raw)
    if m:
        prefix, digits = m.group(1), m.group(2)
        candidates.extend([
            f"{prefix}{int(digits):02d}",
            f"{prefix.upper()}{int(digits):02d}",
            f"{prefix.lower()}{int(digits):02d}",
            f"chr{prefix.lower()}{int(digits):02d}",
            f"Chr{prefix.upper()}{int(digits):02d}",
        ])

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in contigs:
            return candidate


    
    def _canon(value: str) -> str:
        text = value.strip().lower().replace("chr", "").lstrip("0") or "0"
        try:
            return str(int(float(text)))
        except ValueError:
            pass
        match = re.search(r"(?:^|\.)(\d*)(?:ch)?(\d+)$", text)
        return match.group(2).lstrip("0") or "0" if match else text

    canon_in = _canon(raw)
    for contig in references:
        if _canon(contig) == canon_in:
            return contig
    return ""


def extract_promoter(
    species: str,
    chrom: str,
    pos: int,
    window_bp: int = 2000,
) -> str | None:
    if pos <= 0:
        return None
    fasta_path = reference_fasta_path(species)
    if not fasta_path:
        return None
    handle = _fasta_handle(fasta_path)
    if handle is None:
        return None

    chrom_resolved = _normalise_chrom(chrom, tuple(handle.references))
    if not chrom_resolved or chrom_resolved not in handle.references:
        logger.debug("chrom %r could not be resolved in %s", chrom, fasta_path)
        return None

    contig_len = handle.get_reference_length(chrom_resolved)
    if pos > contig_len:
        return None

    start_0based = max(0, pos - 1 - window_bp)
    end_0based = pos - 1
    if end_0based <= start_0based:
        return None

    try:
        seq = handle.fetch(chrom_resolved, start_0based, end_0based)
    except Exception as exc:
        logger.warning(
            "FASTA fetch failed (%s:%d-%d): %s",
            chrom_resolved, start_0based, end_0based, exc,
        )
        return None
    return seq.upper()


def extract_flanking(
    species: str,
    chrom: str,
    pos: int,
    upstream_bp: int = 1000,
    downstream_bp: int = 1000,
) -> str | None:
    """Return a window of ``[pos - upstream_bp, pos + downstream_bp)``.

    Used for any future generic context extraction (e.g. path-agnostic
    motif search).  Behaviour matches :func:`extract_promoter` for failure
    modes.
    """
    if pos <= 0:
        return None
    fasta_path = reference_fasta_path(species)
    if not fasta_path:
        return None
    handle = _fasta_handle(fasta_path)
    if handle is None:
        return None
    chrom_resolved = _normalise_chrom(chrom, tuple(handle.references))
    if not chrom_resolved or chrom_resolved not in handle.references:
        return None
    contig_len = handle.get_reference_length(chrom_resolved)
    if pos > contig_len:
        return None
    start_0based = max(0, pos - 1 - upstream_bp)
    end_0based = min(contig_len, pos - 1 + downstream_bp)
    if end_0based <= start_0based:
        return None
    try:
        seq = handle.fetch(chrom_resolved, start_0based, end_0based)
    except Exception as exc:
        logger.warning("FASTA flanking fetch failed: %s", exc)
        return None
    return seq.upper()
