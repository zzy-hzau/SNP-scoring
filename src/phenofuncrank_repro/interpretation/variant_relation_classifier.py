"Deterministic relation classification for one quoted evidence passage."

from __future__ import annotations

import re

from phenofuncrank_repro.interpretation.schemas.literature import CandidateContext
from phenofuncrank_repro.interpretation.variant_mention_parser import parse_variant_mentions


def _norm(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def _contains(text: str, term: str) -> bool:
    "Match a lexical signal without substring false positives."
    term = _norm(term)
    if not term:
        return False
    if len(term) <= 4 and " " not in term:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))
    return term in text


def classify_supported_relations(text: str, candidate: CandidateContext) -> list[str]:
    lower = _norm(text)
    gene = _norm(candidate.gene)
    has_gene = bool(gene and gene in lower)
    mentions = parse_variant_mentions(text, gene=candidate.gene)
    variant_words = (
        "allele", "allelic", "haplotype", "variant", "snp", "polymorphism", "mutation",
        "substitution", "insertion", "deletion", "editing", "edited", "crispr",
        "transposable", "retrotransposon", "mutant",
    )
    has_variant = bool(mentions) or any(_contains(lower, term) for term in variant_words)
    expression_terms = (
        "expression", "transcript", "mrna", "eqtl", "ase", "reporter",
        "transcript abundance", "promoter activity", "transcription",
        "repression", "chromatin", "methylation", "epigenetic",
    )
    trait_terms = set(re.findall(r"[a-z0-9]+", _norm(candidate.trait)))
    trait_terms.difference_update({
        "variant", "functional", "function", "effect", "evidence", "molecular",
        "mechanism", "specific", "level", "direct", "indirect",
    })
    trait_terms.update({
        "height", "dwarf", "phenotype", "yield", "flowering", "tolerance",
        "shattering", "biomass", "root", "seed", "fruit", "carotenoid",
        "grain", "size", "locule", "salt", "hypersensitivity", "cold",
        "drought", "number", "leaf",
    })
    mechanism_terms = (
        "pathway", "signaling", "signalling", "biochemical", "enzyme",
        "protein interaction", "regulat", "biosynthesis", "mechanism",
    )
    effect_terms = (
        "functional", "activity", "effect", "loss-of-function", "knockout",
        "overexpression", "complementation", "rescue", "mutant", "associated",
        "association", "significant", "cause", "caus", "impair", "substitution",
        "catalytic", "protein", "amino acid", "residue", "non-synonymous",
        "responsible", "control", "confers", "govern", "increase", "decrease",
        "change", "result", "role", "determining", "editing", "edited",
        "reduce", "reduced",
    )
    has_expression = any(
        re.search(r"\btranscripts?\b", lower)
        if term == "transcript" else _contains(lower, term)
        for term in expression_terms
    )
    has_trait = any(_contains(lower, term) for term in trait_terms)
    has_mechanism = any(_contains(lower, term) for term in mechanism_terms)
    has_effect = any(_contains(lower, term) for term in effect_terms)
    has_explicit_variant_effect = any(
        mention.exact_coordinate and (mention.ref_alt or mention.amino_acid_change)
        for mention in mentions
    ) or any(_contains(lower, term) for term in (
        "non-synonymous", "residue change", "amino acid substitution",
    ))

    relations: list[str] = []
    if has_variant and has_gene and has_expression:
        relations.append("variant_expression")
    has_molecular_effect = any(_contains(lower, term) for term in (
        "protein", "amino acid", "residue", "non-synonymous", "catalytic",
        "activity", "functional",
    ))
    if has_variant and has_gene and has_molecular_effect and has_explicit_variant_effect:
        relations.append("variant_functional_effect")
    if has_variant and has_gene and has_trait and has_effect:
        relations.append("variant_trait")
    if has_variant and (has_explicit_variant_effect or has_effect) and (
        has_gene or has_explicit_variant_effect
    ):
        if has_mechanism and any(_contains(lower, term) for term in ("protein", "enzyme", "activity")):
            relations.append("variant_protein_function")
        else:
            relations.append("variant_functional_effect")
    if has_variant and has_gene and has_mechanism:
        relations.append("variant_mechanism")
    if has_gene and has_trait and has_effect:
        relations.append("gene_trait")
    if has_gene and has_mechanism:
        relations.append("gene_mechanism")
    return list(dict.fromkeys(relations))


def classify_supported_relation(text: str, candidate: CandidateContext) -> str:
    """Return the highest-priority atomic relation supported by ``text``."""
    relations = classify_supported_relations(text, candidate)
    return relations[0] if relations else ""


__all__ = ["classify_supported_relation", "classify_supported_relations"]
