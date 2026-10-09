"A2.3 Evidence Extractor (SOP §4.3, test 7)."

from __future__ import annotations

import hashlib
import html
import re
from typing import Any

from phenofuncrank_repro.interpretation.evidence_type_registry import canonical_evidence_type, is_registered
from phenofuncrank_repro.interpretation.llm_utils import complete_json
from phenofuncrank_repro.interpretation.literature_retrieval import RELATION_REQUIRED_TYPES, canonical_relation_name, classify_evidence_type, evidence_aware_score
from phenofuncrank_repro.interpretation.logging_utils import increment
from phenofuncrank_repro.interpretation.settings import settings
from phenofuncrank_repro.interpretation.schemas.literature import CandidateContext, ChunkCitation, EvidenceGapIntent, EvidenceProposal, PaperVariantMention, RetrievalHit
from phenofuncrank_repro.interpretation.variant_identity_reconciliation import parse_variant_mention
from phenofuncrank_repro.interpretation.variant_mention_parser import select_variant_mention
from phenofuncrank_repro.interpretation.variant_mention_normalizer import select_normalized_variant_mention
from phenofuncrank_repro.interpretation.variant_relation_classifier import classify_supported_relation, classify_supported_relations

EXTRACT_DEBUG_DROPS: list[dict[str, Any]] = []


EXTRACT_SYSTEM = (
    "You are an evidence extractor for plant variant interpretation. Given "
    "one evidence gap and text chunks from papers, extract evidence proposals "
    "that DIRECTLY support the gap. Rules: (1) each proposal must be a single "
    "claim faithfully paraphrasing ONE chunk; (2) you must include the exact "
    "supporting quote from that chunk (<=300 chars); (3) never combine "
    "information across chunks; (4) evidence_type must be one of the allowed "
    "types; (5) do not infer causal relationships the paper does not state; "
    "(6) if a chunk has no usable evidence, skip it; (7) supported_relation "
    "must be exactly one canonical relation ID from the supplied gap ontology "
    "(for example variant_expression, variant_trait, gene_trait, or "
    "gene_mechanism), never a free-form claim; (8) for a variant_* relation, "
    "variant_or_allele must identify the tested allele/variant specifically, "
    "not merely repeat the gene name or the generic word allele. "
    'Return JSON only: {"proposals":[{"supported_relation":"...",'
    '"relation_match":"exact|partial|none","claim":"concise claim <=300 chars",'
    '"evidence_type":"...","support":"direct|partial|indirect",'
    '"directness":"direct|partial|indirect|unknown",'
    '"confidence":0.0-1.0,"direction":"up|down|neutral|unknown",'
    '"species_match":"same|ortholog|other|unknown", "species":"...",'
    '"gene":"...","variant_or_allele":"...",'
    '"variant_mention":{"exact_coordinate":"","ref_alt":"",'
    '"rs_id":"","vg_id":"","snp_id":"",'
    '"amino_acid_change":"","named_allele":"","haplotype":"",'
    '"mutant_line":"","structural_variant":"",'
    '"identity_basis":[]},"trait":"...",'
    '"quote":"...","chunk_id":"..."}]}.'
)

_SUPPORT_DOWNGRADE = {"direct": "partial", "partial": "indirect", "indirect": "indirect"}


def _proposal_id(gap_id: str, candidate_id: str, chunk_id: str, claim: str) -> str:
    seed = "|".join([gap_id, candidate_id, chunk_id, claim.strip()[:200]])
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]


def _normalize(text: str) -> str:

    

    
    plain = html.unescape(str(text or ""))
    plain = re.sub(r"<[^>]+>", " ", plain)
    return re.sub(r"\s+", " ", plain).strip().lower()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _normalize(text))


def _quote_grounded(quote: str, source: str) -> bool:
    if not quote or not source:
        return False
    if _normalize(quote) in _normalize(source):
        return True
    compact_quote = _compact(quote)
    return bool(compact_quote) and compact_quote in _compact(source)


_RICE_SPECIES_TERMS = {
    "rice", "oryza", "oryza sativa", "oryza glaberrima", "oryza barthii",
    "african rice",
}


def _species_compatible(left: str, right: str) -> bool:
    left = _normalize(left)
    right = _normalize(right)
    if not left or not right:
        return False
    if left == right:
        return True

    def group(value: str) -> str:
        if value in _RICE_SPECIES_TERMS or "oryza" in value or "rice" in value:
            return "rice"
        return value

    return group(left) == group(right)


def _distinct_tokens(text: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-z0-9]+", _normalize(text))
        if len(token) >= 3
    ]


def _resolve_grounded_quote(
    quote: str,
    source_text: str,
    candidate: CandidateContext,
    target_relation: str,
) -> str:
    """Recover an exact source sentence when the model paraphrases the quote.

    The model quote is still required to be source-grounded.  When its exact
    normalized form is absent, we search the same chunk for a sentence that
    (a) contains the target gene for variant-level gaps, (b) independently
    supports the target atomic relation via the deterministic classifier, and
    (c) shares enough distinctive tokens with the model quote to be the same
    statement.  The returned quote is always an exact sentence from the
    source; no paraphrase is admitted.
    """
    if _quote_grounded(quote, source_text):
        return quote
    quote_tokens = _distinct_tokens(quote)
    if not quote_tokens:
        return quote
    threshold = max(4, int(len(quote_tokens) * 0.3))
    gene = _normalize(candidate.gene)
    best = ""
    best_score = -1
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", source_text):
        sentence = sentence.strip()
        if not sentence:
            continue
        normalized_sentence = _normalize(sentence)
        if target_relation.startswith("variant_") and gene and gene not in normalized_sentence:
            continue
        if target_relation not in classify_supported_relations(sentence, candidate):
            continue
        score = sum(1 for token in quote_tokens if token in normalized_sentence)
        if score >= threshold and score > best_score:
            best = sentence
            best_score = score
    return best[:300] if best else quote


def _quote_recovery_candidates(
    quote: str,
    source_text: str,
    candidate: CandidateContext,
    target_relation: str,
) -> list[dict[str, Any]]:
    """Diagnostic-only: enumerate sentences that could ground a quote."""
    quote_tokens = _distinct_tokens(quote)
    gene = _normalize(candidate.gene)
    candidates: list[dict[str, Any]] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", source_text):
        sentence = sentence.strip()
        if not sentence:
            continue
        normalized_sentence = _normalize(sentence)
        relations = classify_supported_relations(sentence, candidate)
        score = sum(1 for token in quote_tokens if token in normalized_sentence)
        candidates.append({
            "sentence": sentence[:300],
            "normalized_sentence": normalized_sentence[:300],
            "contains_target_gene": bool(gene and gene in normalized_sentence),
            "supports_target_relation": target_relation in relations,
            "supported_relations": relations,
            "token_overlap": score,
            "quote_token_count": len(quote_tokens),
        })
    return sorted(
        candidates,
        key=lambda item: (
            item["supports_target_relation"],
            item["contains_target_gene"],
            item["token_overlap"],
        ),
        reverse=True,
    )


def _entity_grounded_quote(
    source_text: str,
    quote: str,
    candidate: CandidateContext,
) -> str:
    "Select an exact source window containing target gene and variant."
    source = str(source_text or "")
    current = str(quote or "")
    gene = str(candidate.gene or "").strip()
    if not source or not gene or _normalize(gene) in _normalize(current):
        return current
    gene_match = re.search(re.escape(gene), source, re.I)
    if gene_match is None:
        return current
    positions = [gene_match.start(), gene_match.end()]
    for pattern in (
        r"(?:chr(?:omosome)?\s*)?[0-9]+\s*[:\-]\s*[0-9]{3,}",
        r"\b[0-9]{5,}\s*(?:[ACGTN]+\s*(?:>|/|→|to)\s*[ACGTN]+)\b",
    ):
        variant_match = re.search(pattern, source, re.I)
        if variant_match is not None:
            positions.extend((variant_match.start(), variant_match.end()))
    left = max(0, min(positions) - 150)
    right = min(len(source), max(positions) + 150)
    if right - left > 300:
        left = max(0, gene_match.start() - 150)
        right = min(len(source), left + 300)
    return source[left:right].strip()[:300]


def _variant_notation_normalize(text: str) -> str:
    """Normalize short REF/ALT notation without changing ordinary prose."""
    value = _normalize(text)
    return re.sub(
        r"\b([acgtn]{1,2})\s*(?:>|/|→|⟶|to)\s*([acgtn]{1,2})\b",
        r"\1>\2", value, flags=re.I,
    )


_VARIANT_MENTION_FIELDS = (
    "exact_coordinate", "ref_alt", "rs_id", "vg_id", "snp_id",
    "amino_acid_change", "named_allele", "haplotype", "mutant_line",
    "structural_variant",
)
_GENERIC_VARIANT_VALUES = {
    "allele", "variant", "mutation", "mutant", "snp", "polymorphism",
    "natural allele", "edited allele",
}


def _paper_variant_mention(
    raw: dict[str, Any],
    candidate: CandidateContext,
    source_text: str = "",
) -> PaperVariantMention:
    "Normalize the extractor's structured identity."
    raw_mention = raw.get("variant_mention")
    if isinstance(raw_mention, dict):
        try:
            mention = PaperVariantMention.model_validate(raw_mention)
        except Exception:
            mention = PaperVariantMention()
    else:
        mention = PaperVariantMention()

    

    mention = _sanitize_paper_variant_mention(mention, source_text)
    legacy = str(raw.get("variant_or_allele") or "").strip()
    if legacy and _normalize(legacy) not in _GENERIC_VARIANT_VALUES:
        parsed = parse_variant_mention(legacy, gene=candidate.gene)
        values = {
            field: getattr(parsed, field, "")
            for field in _VARIANT_MENTION_FIELDS
        }
        values["identity_basis"] = list(parsed.identity_basis)
        legacy_mention = _sanitize_paper_variant_mention(
            PaperVariantMention(**values), source_text,
        )
        if not mention.informative_values():
            mention = legacy_mention

    normalized = select_normalized_variant_mention(
        source_text,
        candidate_pos=candidate.pos,
        candidate_chrom=candidate.chrom,
        candidate_ref=candidate.ref,
        candidate_alt=candidate.alt,
        variant_description=candidate.variant_description,
        gene=candidate.gene,
    )
    if not normalized.informative_values():
        return mention

    

    
    normalized_mention = normalized.to_paper_variant_mention()
    merged = normalized_mention.model_dump()
    for field in _VARIANT_MENTION_FIELDS:
        normalized_value = str(getattr(normalized_mention, field, "") or "").strip()
        model_value = str(getattr(mention, field, "") or "").strip()
        if not normalized_value and model_value:
            merged[field] = model_value
        elif field in {"named_allele", "haplotype", "mutant_line"} and model_value:
            if _normalize(model_value) in _normalize(source_text):
                merged[field] = model_value
    merged["identity_basis"] = sorted(set(
        list(normalized_mention.identity_basis) + list(mention.identity_basis)
    ))
    return _sanitize_paper_variant_mention(
        PaperVariantMention(**merged), source_text,
    )


def _sanitize_paper_variant_mention(
    mention: PaperVariantMention,
    source_text: str = "",
) -> PaperVariantMention:
    source = _normalize(source_text)
    updates: dict[str, Any] = {}
    coord_re = re.compile(
        r"^(?:(?:chr(?:omosome)?\s*)?[A-Za-z0-9]+\s*:\s*\d[\d,]*|\d[\d,]{4,})"
        r"(?:\s*[ACGTN]+\s*(?:>|/|→|to)\s*[ACGTN]+)?$",
        re.I,
    )
    allele_re = re.compile(r"^[ACGTN]+\s*(?:>|/|→|to)\s*[ACGTN]+$", re.I)
    id_patterns = {
        "rs_id": re.compile(r"^rs\d+$", re.I),
        "vg_id": re.compile(r"^vg[A-Za-z0-9_.-]+$", re.I),
        "snp_id": re.compile(r"^snp[_-]?[A-Za-z0-9_.-]+$", re.I),
    }
    aa_re = re.compile(
        r"^(?:p\.)?(?:[A-Z][a-z]{2}|[A-Z])\d+(?:[A-Z][a-z]{2}|[A-Z*])$"
    )
    for field in _VARIANT_MENTION_FIELDS:
        value = str(getattr(mention, field, "") or "").strip()
        if not value:
            continue
        normalized = _normalize(value)
        if normalized in _GENERIC_VARIANT_VALUES:
            updates[field] = ""
            continue
        if source and normalized not in source and _compact(value) not in _compact(source):
            if field == "ref_alt" and _variant_notation_normalize(value) in _variant_notation_normalize(source_text):
                continue
            updates[field] = ""
            continue
        if field == "exact_coordinate" and not coord_re.fullmatch(value):
            updates[field] = ""
        elif field == "ref_alt" and not allele_re.fullmatch(value):
            updates[field] = ""
        elif field in id_patterns and not id_patterns[field].fullmatch(value):
            updates[field] = ""
        elif field == "amino_acid_change" and not aa_re.fullmatch(value):
            updates[field] = ""
    if updates:
        return mention.model_copy(update=updates)
    return mention


def _mention_is_quote_grounded(mention: PaperVariantMention, source_text: str) -> bool:
    source = _normalize(source_text)
    values = [
        value for value in mention.informative_values()
        if _normalize(value) not in _GENERIC_VARIANT_VALUES
    ]
    source_compact = _compact(source_text)
    return bool(values) and any(
        _normalize(value) in source
        or _compact(value) in source_compact
        or _variant_notation_normalize(value) in _variant_notation_normalize(source_text)
        for value in values
    )


def _variant_identity_match(
    mention: PaperVariantMention,
    candidate: CandidateContext,
    source_text: str = "",
) -> bool:
    returned_values = [
        _normalize(value) for value in mention.informative_values()
        if _normalize(value) not in _GENERIC_VARIANT_VALUES
    ]
    if not returned_values:
        return False
    informative = {
        _normalize(candidate.snp_id),
        _normalize(candidate.variant_description),
        _normalize(f"{candidate.chrom}:{candidate.pos}")
        if candidate.chrom and candidate.pos is not None else "",
        _normalize(f"{candidate.ref}>{candidate.alt}")
        if candidate.ref and candidate.alt else "",
        _normalize(f"{candidate.ref} to {candidate.alt}")
        if candidate.ref and candidate.alt else "",
    }
    informative = {value for value in informative if value and len(value) > 1}
    if any(
        any(value in returned or returned in value for value in informative)
        for returned in returned_values
    ):
        if source_text and not _mention_is_quote_grounded(mention, source_text):
            return False
        return True
    candidate_description = _compact(candidate.variant_description)
    if candidate_description:
        for returned in returned_values:
            compact_returned = _compact(returned)
            if len(compact_returned) >= 3 and (
                compact_returned in candidate_description
                or candidate_description in compact_returned
            ):
                if source_text and not _mention_is_quote_grounded(mention, source_text):
                    return False
                return True

    

    tokens = [
        token for token in re.findall(r"[a-z0-9]+", _normalize(candidate.variant_description))
        if len(token) >= 2 and token not in {"allele", "variant", "mutation", "snp", "the", "and", "versus"}
    ]
    overlap = max(
        sum(token in returned for token in set(tokens))
        for returned in returned_values
    ) if returned_values else 0
    if overlap >= 2:
        if source_text and not _mention_is_quote_grounded(mention, source_text):
            return False
        return True

    

    
    gene = _normalize(candidate.gene)
    variant_terms = {
        "snp", "polymorphism", "mutation", "insertion", "deletion", "haplotype",
        "editing", "edited", "target", "site", "promoter", "regulatory",
        "missense", "retrotransposon", "tandem", "repeat", "allele-specific",
    }
    return bool(
        gene and any(gene in returned for returned in returned_values)
        and any(
            term in returned
            for returned in returned_values
            for term in variant_terms
        )
        and (not source_text or _mention_is_quote_grounded(mention, source_text))
    )


def _infer_supported_relation(text: str, candidate: CandidateContext) -> str:
    """Infer a canonical relation from one quote, independently of the gap.

    This is intentionally conservative and lexical.  It is used only by the
    offline fallback to avoid copying ``gap.missing_link`` into a proposal;
    the target gap is compared with this result afterwards.
    """
    lower = _normalize(text)
    gene = _normalize(candidate.gene)
    has_gene = bool(gene and gene in lower)
    variant_terms = (
        "allele", "haplotype", "variant", "snp", "polymorphism", "mutation",
        "editing", "crispr", "allele-specific", "allele specific",
    )
    expression_terms = (
        "expression", "transcript", "mrna", "eqtl", "ase", "reporter",
        "transcript abundance", "promoter activity",
    )
    trait_terms = (
        "height", "dwarf", "phenotype", "yield", "flowering", "tolerance",
        "shattering", "biomass",
    )
    mechanism_terms = (
        "pathway", "signaling", "signalling", "biochemical", "enzyme",
        "protein interaction", "regulat", "biosynthesis", "mechanism",
    )
    functional_terms = (
        "functional", "activity", "effect", "loss-of-function", "knockout",
        "overexpression", "complementation", "rescue", "mutant", "associated",
        "association", "significant",
    )
    has_variant = any(term in lower for term in variant_terms)
    has_expression = any(term in lower for term in expression_terms)
    has_trait = any(term in lower for term in trait_terms)
    has_mechanism = any(term in lower for term in mechanism_terms)
    has_function = any(term in lower for term in functional_terms)
    if has_variant and has_expression and has_gene:
        return "variant_expression"
    if has_variant and has_trait and has_gene and has_function:
        return "variant_trait"
    if has_variant and has_mechanism and has_gene:
        if any(term in lower for term in ("protein", "enzyme", "activity")):
            return "variant_protein_function"
        return "variant_mechanism"
    if has_variant and has_function and has_gene:
        return "variant_functional_effect"
    if has_gene and has_trait and has_function:
        return "gene_trait"
    if has_gene and has_mechanism:
        return "gene_mechanism"
    return ""


def extract_proposals(
    hits: list[RetrievalHit],
    gap: EvidenceGapIntent,
    candidate: CandidateContext,
    *,
    max_chunks: int | None = None,
) -> tuple[list[EvidenceProposal], dict[str, int]]:
    """Extract proposals from top chunks with hard constraints."""
    max_chunks = max(1, int(max_chunks or settings.extractor_context_k))
    stats = {
        "extracted": 0,
        "dropped_whitelist": 0,
        "dropped_quote": 0,
        "dropped_length": 0,
        "dropped_unknown_type": 0,
        "llm_failures": 0,
        "ortholog_discounted": 0,
        "dropped_relation": 0,
        "dropped_entity": 0,
        "variant_identity_unresolved": 0,
        "deterministic_fallback": 0,
    }
    top = sorted(hits, key=lambda h: h.rerank_score, reverse=True)[:max_chunks]
    if not top:
        return [], stats
    chunks = [
        {
            "chunk_id": h.chunk_id or h.hit_id,
            "section": h.section,
            "span": list(h.span),
            "text": h.content[:1500],
        }
        for h in top
    ]
    data = complete_json(
        EXTRACT_SYSTEM,
        {
            "gap": {
                "layer": gap.layer,
                "missing_link": gap.missing_link,
                "query_intent": gap.query_intent,
                "allowed_evidence_types": sorted(gap.required_evidence_types),
            },
            "candidate": {
                "species": candidate.species,
                "trait": candidate.trait,
                "gene": candidate.gene,
                "snp_id": candidate.snp_id,

                

                

                "variant_description": candidate.variant_description,
                "chrom": candidate.chrom,
                "pos": candidate.pos,
                "ref": candidate.ref,
                "alt": candidate.alt,
            },
            "chunks": chunks,
        },
        max_tokens=1500,
        extra_body={"reasoning_effort": "low"},
    )
    raw_proposals = (data or {}).get("proposals", []) if data else []
    if not raw_proposals:
        stats["llm_failures"] = 1 if data is None else 0
        if canonical_relation_name(gap.missing_link) in {"phenotype", "trait_evidence"}:
            return [], stats

        

        
        fallback, fallback_stats = _deterministic_fallback(
            hits, gap, candidate, max_chunks=max_chunks,
        )
        stats["deterministic_fallback"] = len(fallback)
        stats["extracted"] = len(fallback)
        for key, value in fallback_stats.items():
            stats[key] = stats.get(key, 0) + value
        increment("literature_engine_proposals_extracted", float(len(fallback)))
        return fallback, stats

    allowed = {canonical_evidence_type(value) for value in gap.required_evidence_types}
    chunk_text_by_id = {c["chunk_id"]: c["text"] for c in chunks}
    per_chunk_counts: dict[str, int] = {}
    proposals: list[EvidenceProposal] = []
    for raw in raw_proposals:
        claim = str(raw.get("claim") or "").strip()
        evidence_type = canonical_evidence_type(str(raw.get("evidence_type") or "").strip())
        quote = str(raw.get("quote") or "").strip()
        chunk_id = str(raw.get("chunk_id") or "")
        if not claim or len(claim) > 300:
            stats["dropped_length"] += 1
            continue
        supported_relation = canonical_relation_name(
            str(raw.get("supported_relation") or "").strip()
        )
        target_relation = canonical_relation_name(gap.missing_link)
        declared_match = str(raw.get("relation_match") or "none").strip().lower()
        if declared_match not in {"exact", "partial", "none"}:
            declared_match = "none"

        

        
        classified_relations = [
            canonical_relation_name(value)
            for value in classify_supported_relations(quote, candidate)
        ]
        classified_relations = [value for value in classified_relations if value]
        if target_relation in classified_relations:
            classified_relation = target_relation
        elif classified_relations:
            classified_relation = classified_relations[0]
        else:
            classified_relation = ""
        if classified_relation:
            supported_relation = classified_relation
            stats["deterministic_relation_overrides"] = (
                stats.get("deterministic_relation_overrides", 0)
                + int(classified_relation != canonical_relation_name(
                    str(raw.get("supported_relation") or "").strip()
                ))
            )

        

        if not supported_relation:
            stats["dropped_relation"] += 1
            continue
        if classified_relation == target_relation:
            relation_match = "exact"
        elif supported_relation == target_relation and declared_match == "exact":
            relation_match = "exact"
        elif supported_relation == target_relation and declared_match == "partial":
            relation_match = "partial"
        else:
            relation_match = "none"
        if evidence_type not in allowed or not is_registered(evidence_type):
            stats["dropped_whitelist"] += 1
            continue
        source_text = chunk_text_by_id.get(chunk_id, "")
        quote = _resolve_grounded_quote(
            quote, source_text, candidate, target_relation,
        )
        if not _quote_grounded(quote, source_text):
            EXTRACT_DEBUG_DROPS.append({
                "case_id": gap.candidate_id,
                "gap_id": gap.gap_id,
                "target_relation": target_relation,
                "model_quote": quote,
                "normalized_quote": _normalize(quote),
                "compact_quote": _compact(quote),
                "source_text": source_text,
                "candidate_gene": candidate.gene,
                "candidate_species": candidate.species,
                "recovery_candidates": _quote_recovery_candidates(
                    quote, source_text, candidate, target_relation,
                ),
                "reason": "quote_not_grounded_after_recovery",
            })
            stats["dropped_quote"] += 1
            continue
        variant_mention = _paper_variant_mention(raw, candidate, source_text)
        expected_species = _normalize(candidate.species)
        expected_gene = _normalize(candidate.gene)
        expected_trait = _normalize(candidate.trait)
        returned_species = _normalize(str(raw.get("species") or ""))
        returned_gene = _normalize(str(raw.get("gene") or ""))
        returned_trait = _normalize(str(raw.get("trait") or ""))
        entity_mismatch = False
        if returned_species and expected_species:
            if not _species_compatible(returned_species, expected_species):
                entity_mismatch = True
            elif returned_species != expected_species:
                raw["species"] = candidate.species
        if returned_gene and expected_gene and expected_gene not in returned_gene and returned_gene not in expected_gene:
            entity_mismatch = True
        if returned_trait and expected_trait and not (
            expected_trait in returned_trait or returned_trait in expected_trait
        ):
            entity_mismatch = True
        if target_relation.startswith("variant_"):
            if not _mention_is_quote_grounded(variant_mention, source_text):
                entity_mismatch = True
            elif not _variant_identity_match(variant_mention, candidate, source_text):
                stats["variant_identity_unresolved"] += 1
            quote = _entity_grounded_quote(source_text, quote, candidate)
        if entity_mismatch:
            relation_match = "none"
            stats["dropped_entity"] += 1
        if per_chunk_counts.get(chunk_id, 0) >= 2:
            stats["dropped_length"] += 1
            continue
        per_chunk_counts[chunk_id] = per_chunk_counts.get(chunk_id, 0) + 1
        support = str(raw.get("support") or "indirect")
        if support not in {"direct", "partial", "indirect"}:
            support = "indirect"
        confidence = max(0.0, min(1.0, float(raw.get("confidence") or 0.0)))
        direction = str(raw.get("direction") or "unknown")
        if direction not in {"up", "down", "neutral", "unknown"}:
            direction = "unknown"
        species_match = str(raw.get("species_match") or "unknown")
        if species_match not in {"same", "ortholog", "other", "unknown"}:
            species_match = "unknown"
        if species_match == "ortholog":
            confidence = round(confidence * 0.7, 4)
            support = _SUPPORT_DOWNGRADE.get(support, "indirect")
            stats["ortholog_discounted"] += 1
        directness = str(raw.get("directness") or support or "unknown")
        if directness not in {"direct", "partial", "indirect", "unknown"}:
            directness = "unknown"
        if relation_match != "exact":
            support = "partial" if relation_match == "partial" else "indirect"
        chunk = next((h for h in top if (h.chunk_id or h.hit_id) == chunk_id), None)
        if chunk is None:
            stats["dropped_unknown_type"] += 1
            continue
        raw_variant = str(raw.get("variant_or_allele") or "").strip()
        variant_text = raw_variant
        if not variant_text or _normalize(variant_text) in _GENERIC_VARIANT_VALUES:
            variant_text = next(iter(variant_mention.informative_values()), "")

        

        

        
        proposal_gene = str(raw.get("gene") or "")
        if expected_gene and expected_gene in _normalize(source_text):
            proposal_gene = candidate.gene
        proposals.append(EvidenceProposal(
            proposal_id=_proposal_id(gap.gap_id, gap.candidate_id, chunk_id, claim),
            gap_id=gap.gap_id,
            candidate_id=gap.candidate_id,
            evidence_type=evidence_type,
            claim=claim[:200],
            support=support,
            confidence=confidence,
            direction=direction,
            species_match=species_match,
            supported_relation=supported_relation,
            relation_match=relation_match,
            species=str(raw.get("species") or ""),
            gene=proposal_gene,
            variant_or_allele=variant_text,
            variant_mention=variant_mention,
            trait=str(raw.get("trait") or ""),
            directness=directness,
            source={
                "pmid": chunk.candidate.pmid,
                "doi": chunk.candidate.doi,
                "title": chunk.candidate.title,
            },
            chunk=ChunkCitation(
                article_id=chunk.article_id,
                section=chunk.section,
                start_char=chunk.span[0],
                end_char=chunk.span[1],
                quote=quote[:300],
            ),
        ))

    

    

    
    stats["extracted"] = len(proposals)
    increment("literature_engine_proposals_extracted", float(len(proposals)))
    increment("literature_engine_proposals_dropped", float(sum(
        stats[k] for k in (
            "dropped_whitelist", "dropped_quote", "dropped_length",
        "dropped_unknown_type",
        "dropped_relation",
        "dropped_entity",
    )
)))
    return proposals, stats


def _deterministic_fallback(
    hits: list[RetrievalHit], gap: EvidenceGapIntent, candidate: CandidateContext,
    *, max_chunks: int,
) -> tuple[list[EvidenceProposal], dict[str, int]]:
    "extraction when the extractor call is unavailable."
    stats = {"dropped_relation": 0, "dropped_whitelist": 0, "dropped_quote": 0}
    allowed = {canonical_evidence_type(value) for value in gap.required_evidence_types}
    target_relation = canonical_relation_name(gap.missing_link)
    proposals: list[EvidenceProposal] = []
    for hit in sorted(hits, key=lambda value: value.rerank_score, reverse=True)[:max_chunks]:
        parts = evidence_aware_score(
            hit.content, gap, candidate,
            text_status=hit.text_status or hit.candidate.text_status,
            retrieval_relevance=hit.retrieval_relevance,
        )

        

        
        relation_is_explicit = target_relation in classify_supported_relations(
            hit.content, candidate,
        )
        if parts["relation_match"] < 0.85 and not relation_is_explicit:
            stats["dropped_relation"] += 1
            continue
        if target_relation.startswith("variant_") and parts["variant_match"] < 0.8:
            stats["dropped_relation"] += 1
            continue
        if target_relation.startswith("gene_") and parts["gene_match"] < 0.8:
            stats["dropped_relation"] += 1
            continue
        evidence_type = canonical_evidence_type(classify_evidence_type(hit.content) or "")
        if (
            not evidence_type
            and target_relation.startswith("variant_")
            and relation_is_explicit
        ):

            

            evidence_type = "variant_specific"
        if not evidence_type or not is_registered(evidence_type) or evidence_type not in allowed:
            stats["dropped_whitelist"] += 1
            continue
        sentences = [
            sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", hit.content)
            if sentence.strip()
        ]
        quote = next(
            (sentence[:300] for sentence in sentences if _normalize(sentence) and (
                _normalize(candidate.gene) in _normalize(sentence)
                or target_relation.startswith("gene_")
            )),
            (sentences[0][:300] if sentences else hit.content[:300]),
        )
        if not _quote_grounded(quote, hit.content):
            stats["dropped_quote"] += 1
            continue

        

        classified_relations = classify_supported_relations(quote, candidate)
        relation = (
            target_relation if target_relation in classified_relations
            else (classified_relations[0] if classified_relations else "")
        )
        relation = relation or _infer_supported_relation(quote, candidate)
        if relation not in RELATION_REQUIRED_TYPES:
            stats["dropped_relation"] += 1
            continue

        
        # cannot license a quote about a generic gene-level variation.
        quote_parts = evidence_aware_score(
            quote, gap, candidate,
            text_status=hit.text_status or hit.candidate.text_status,
            retrieval_relevance=hit.retrieval_relevance,
        )
        quote_relation_is_explicit = target_relation in classify_supported_relations(
            quote, candidate,
        )
        if quote_parts["relation_match"] < 0.85 and not quote_relation_is_explicit:
            stats["dropped_relation"] += 1
            continue
        if target_relation.startswith("variant_") and quote_parts["variant_match"] < 0.8:
            stats["dropped_relation"] += 1
            continue
        variant_mention = PaperVariantMention()
        if target_relation.startswith("variant_"):
            variant_mention = select_variant_mention(
                quote,
                candidate_pos=candidate.pos,
                candidate_chrom=candidate.chrom,
                candidate_ref=candidate.ref,
                candidate_alt=candidate.alt,
                variant_description=candidate.variant_description,
                gene=candidate.gene,
            )
            if not _variant_identity_match(variant_mention, candidate, quote):

                
                stats["dropped_relation"] += 1
                continue
        evidence_type = canonical_evidence_type(classify_evidence_type(quote) or "")
        if (
            not evidence_type
            and target_relation.startswith("variant_")
            and quote_relation_is_explicit
        ):
            evidence_type = "variant_specific"
        if not evidence_type or not is_registered(evidence_type) or evidence_type not in allowed:
            stats["dropped_whitelist"] += 1
            continue
        direct = quote_parts["relation_match"] >= 0.95 and quote_parts["experimental_directness"] >= 0.8
        proposals.append(EvidenceProposal(
            proposal_id=_proposal_id(gap.gap_id, gap.candidate_id, hit.chunk_id or hit.hit_id, quote),
            gap_id=gap.gap_id, candidate_id=gap.candidate_id,
            evidence_type=evidence_type, claim=quote,
            support="direct" if direct else "partial",
            confidence=round(min(0.95, max(0.7, quote_parts["evidence_score"])), 4),
            direction="unknown", species_match="unknown",
            supported_relation=relation,
            relation_match="exact" if relation == target_relation else "none",
            species=candidate.species, gene=candidate.gene,
            variant_or_allele=(
                next(iter(variant_mention.informative_values()), "")
                if relation.startswith("variant_") else ""
            ),
            variant_mention=variant_mention,
            trait=candidate.trait,
            directness="direct" if direct else "partial",
            source={"pmid": hit.candidate.pmid, "doi": hit.candidate.doi, "title": hit.candidate.title},
            chunk=ChunkCitation(
                article_id=hit.article_id, section=hit.section,
                start_char=hit.span[0], end_char=hit.span[1], quote=quote,
            ),
        ))
    return proposals, stats
