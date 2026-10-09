"Pathway enrichment over the local Plant Reactome gene–pathway mappings."

from __future__ import annotations

import gzip
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from scipy.stats import hypergeom


_REACTOME_ROOT = Path(os.environ.get("PHENO_REACTOME_DIR", "resources/plant_reactome"))


@lru_cache(maxsize=1)
def _load_identifier_mappings() -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    files = (
        _REACTOME_ROOT / "UniProt2PlantReactome.txt",
        _REACTOME_ROOT / "NCBI2PlantReactome.txt",
        _REACTOME_ROOT / "Ensembl2PlantReactome.txt",
        _REACTOME_ROOT / "UniProt2Reactome.txt",
    )
    for path in files:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2 or not parts[0] or not parts[1]:
                    continue
                mapping.setdefault(parts[0].strip().lower(), set()).add(parts[1])
    return mapping


@lru_cache(maxsize=1)
def _species_pathway_counts() -> dict[str, int]:
    """Plant Reactome pathway count per species (from UniProt map)."""
    counts: dict[str, int] = {}
    path = _REACTOME_ROOT / "UniProt2PlantReactome.txt"
    if not path.is_file():
        return counts
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 6 and parts[5]:
                counts[parts[5]] = counts.get(parts[5], 0) + 1
    return counts


@lru_cache(maxsize=1)
def _load_pathway_names() -> dict[str, str]:
    names: dict[str, str] = {}
    path = _REACTOME_ROOT / "ReactomePathways.txt"
    if not path.is_file():
        return names
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                names[parts[0]] = parts[1]
    return names


@lru_cache(maxsize=8)
def _go_index(species: str) -> dict[str, set[tuple[str, str]]]:
    path = Path(os.environ.get("PHENO_UNIPROT_GO_DIR", "resources/uniprot_xref")) / f"{species.lower()}.go.tsv.gz"
    if not path.is_file():
        return {}
    index: dict[str, set[tuple[str, str]]] = {}
    process_pattern = re.compile(
        r"process|biosynthesis|metabolism|catabolism|response to|signaling|signalling|"
        r"biosynthetic|catabolic|metabolic|homeostasis|development|differentiation|"
        r"localization|transport|regulation of",
        re.I,
    )
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        try:
            col_acc = header.index("Entry")
            col_gene = header.index("Gene Names")
            col_go = header.index("Gene Ontology (GO)")
            col_refseq = header.index("RefSeq")
            col_ensembl = header.index("Ensembl")
        except ValueError:
            return {}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(col_acc, col_gene, col_go, col_refseq, col_ensembl):
                continue
            accession = parts[col_acc].strip().lower()
            go_text = parts[col_go] if len(parts) > col_go else ""
            terms: list[tuple[str, str]] = []
            for token in go_text.split(";"):
                match = re.match(r"^\s*(.*?)\s*\[(GO:\d+)\]", token)
                if not match:
                    continue
                name, go_id = match.group(1).strip(), match.group(2)
                if process_pattern.search(name):
                    terms.append((go_id, name))
            if not terms or not accession:
                continue
            identifiers = [accession]
            for field in (parts[col_gene], parts[col_refseq], parts[col_ensembl]):
                for token in re.split(r"[;,\s]+", field or ""):
                    token = token.strip().lower()
                    if len(token) >= 3:
                        identifiers.append(token)
            for identifier in identifiers:
                index.setdefault(identifier, set()).update(terms)

                

                unversioned = re.sub(r"\.\d+(\.\d+)?$", "", identifier)
                if unversioned and unversioned != identifier:
                    index.setdefault(unversioned, set()).update(terms)


    

    supplement = Path(os.environ.get("PHENO_NCBI_GO_DIR", "resources/go_ncbi")) / f"{species.lower()}.go.tsv"
    if supplement.is_file():
        with supplement.open(encoding="utf-8", errors="replace") as handle:
            handle.readline()  
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                ident, go_id, go_name = parts[0].strip().lower(), parts[1].strip(), parts[2].strip()
                if not ident or not go_id or not go_name:
                    continue
                if process_pattern.search(go_name):
                    terms = {(go_id, go_name)}
                    index.setdefault(ident, set()).update(terms)
                    unversioned = re.sub(r"\.\d+(\.\d+)?$", "", ident)
                    if unversioned != ident:
                        index.setdefault(unversioned, set()).update(terms)
    return index


def _pathway_gene_sets() -> dict[str, set[str]]:
    """Plant Reactome pathway stId → set of mapped identifiers."""
    gene_sets: dict[str, set[str]] = {}
    for identifier, pathways in _load_identifier_mappings().items():
        for pathway in pathways:
            gene_sets.setdefault(pathway, set()).add(identifier)
    return gene_sets


def run_pathway_enrichment(
    uniprot_ids: list[str],
    *,
    min_genes_for_test: int = 5,
    gene_ids: list[str] | None = None,
    refseq_ids: list[str] | None = None,
    species: str = "",
) -> dict[str, Any]:
    """Map candidate identifiers to Plant Reactome pathways.

    Returns a structured result suitable for the report:
      status, query_genes, background_genes, hits, note
    """
    mapping = _load_identifier_mappings()
    gene_sets = _pathway_gene_sets()
    names = _load_pathway_names()
    go_index = _go_index(species) if species else {}

    candidates: list[str] = []
    for value in (list(uniprot_ids or []), list(gene_ids or []), list(refseq_ids or [])):
        for identifier in value:
            key = str(identifier).strip().lower()
            if key.startswith(("gene:", "rna:", "transcript:")):
                key = key.split(":", 1)[1]
            if key and key not in candidates:
                candidates.append(key)
            unversioned = re.sub(r"\.\d+(\.\d+)?$", "", key)
            if unversioned and unversioned not in candidates:
                candidates.append(unversioned)
    genes = [g for g in candidates if g in mapping]
    go_genes = [g for g in candidates if g in go_index]
    species_counts = _species_pathway_counts()
    rice_count = species_counts.get("Oryza sativa", 0)
    if not genes and not go_genes:
        return {
            "status": "unavailable",
            "reason": (
                f"candidate identifiers checked against local Plant Reactome and GO "
                f"({len(candidates)} ids; none present; Plant Reactome Oryza coverage: {rice_count})"
            ),
            "query_genes": candidates,
            "background_genes": len(mapping),
            "hits": [],
            "note": (
                "No candidate gene could be mapped to a local Plant Reactome pathway or GO biological process."
            ),
        }

    background = len(mapping)
    counts: dict[str, set[str]] = {}
    for identifier in genes:
        for pathway in mapping[identifier]:
            counts.setdefault(pathway, set()).add(identifier)

    go_counts: dict[tuple[str, str], set[str]] = {}
    for identifier in go_genes:
        for go_id, name in go_index[identifier]:
            go_counts.setdefault((go_id, name), set()).add(identifier)

    hits: list[dict[str, Any]] = []
    for pathway, gene_set in sorted(counts.items(), key=lambda item: (-len(item[1]), item[0])):
        gene_count = len(gene_set)
        pvalue = None
        hits.append({
            "pathway_id": pathway,
            "pathway_name": names.get(pathway, pathway),
            "genes": sorted(gene_set),
            "gene_count": gene_count,
            "pvalue": pvalue,
            "source": "PlantReactome",
        })
    for (go_id, go_name), gene_set in sorted(
        go_counts.items(), key=lambda item: (-len(item[1]), item[0][1])
    ):
        hits.append({
            "pathway_id": go_id,
            "pathway_name": go_name,
            "genes": sorted(gene_set),
            "gene_count": len(gene_set),
            "pvalue": None,
            "source": "GO",
        })

    mapped_total = len(genes) + len(go_genes)
    if mapped_total >= min_genes_for_test:
        note = (
            f"{mapped_total} candidate identifier(s) mapped; {len(hits)} pathway/process "
            f"hit(s) across Plant Reactome and GO."
        )
    else:
        note = (
            f"Only {mapped_total} candidate identifier(s) mapped; below the {min_genes_for_test}-identifier "
            "threshold, so pathway hits are reported without significance tests."
        )

    return {
        "status": "success",
        "query_genes": genes,
        "background_genes": None,
        "hits": hits[:10],
        "note": "Candidate pathway/process annotations; SNP enrichment is computed with matched genomic backgrounds.",
        "analysis_type": "pathway_annotation",
    }
