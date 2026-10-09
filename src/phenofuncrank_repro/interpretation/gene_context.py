"A2 GFF parsing and nearest-gene lookup."
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

def _norm_chrom(chrom: str) -> str:
    """Canonical chromosome key for matching VCF contigs to GFF/FASTA."""
    text = str(chrom).strip().lower().replace("chr", "").lstrip("0") or "0"
    try:
        return str(int(float(text)))
    except ValueError:
        pass

    match = re.search(r"(?:^|\.)(\d*)(?:ch)?(\d+)$", text)
    return match.group(2).lstrip("0") or "0" if match else text

def _open_gff(path: str):
    """Open a GFF file, handling gzip compression."""
    p = Path(path)
    if p.suffix == ".gz" or path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")

def _load_gff_genes(gff_path: str) -> dict[str, list[dict[str, Any]]]:
    """Parse a GFF3/gzip file and return {chrom: [genes]} keyed by chromosome.

    Each gene is a dict with: gene_id, gene_name, start, end, strand.
    """
    genes: dict[str, list[dict[str, Any]]] = {}
    with _open_gff(gff_path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2].lower() != "gene":
                continue
            chrom = _norm_chrom(parts[0])
            attrs = _parse_gff_attrs(parts[8])
            gene_id = attrs.get("ID") or attrs.get("gene_id") or attrs.get("Name") or ""
            gene_name = attrs.get("Name") or attrs.get("gene") or gene_id
            try:
                start = int(parts[3])
                end = int(parts[4])
            except (ValueError, IndexError):
                continue
            genes.setdefault(chrom, []).append({
                "gene_id": gene_id,
                "gene_name": gene_name,
                "start": start,
                "end": end,
                "strand": parts[6],
            })

    for chrom in genes:
        genes[chrom].sort(key=lambda g: g["start"])
    return genes

def _parse_gff_attrs(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in text.strip().split(";"):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif " " in item:
            key, value = item.split(" ", 1)
            value = value.strip('"')
        else:
            continue
        out[key.strip()] = value.strip()
    return out

def _find_nearest_gene(
    chrom: str,
    pos: int,
    genes: dict[str, list[dict[str, Any]]],
    window_bp: int = 500_000,
) -> dict[str, Any] | None:
    """Find the nearest gene to a given position."""
    chrom_key = _norm_chrom(chrom)
    candidates = genes.get(chrom_key, [])
    if not candidates:
        return None

    best = None
    best_dist = float("inf")
    for gene in candidates:
        if pos < gene["start"] - window_bp:
            break  
        if pos > gene["end"] + window_bp:
            continue
        if gene["start"] <= pos <= gene["end"]:
            return {**gene, "distance": 0, "overlapping": True}
        dist = min(abs(pos - gene["start"]), abs(pos - gene["end"]))
        if dist < best_dist:
            best_dist = dist
            best = {**gene, "distance": dist, "overlapping": False}
    return best

GENE_ANNOTATION_MAX_DIST_BP = 2000
