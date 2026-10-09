"A2.2 Literature Retrieval Engine (SOP §4.2)."

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from xml.etree import ElementTree as ET

from phenofuncrank_repro.interpretation.embedding import get_embedding_service
from phenofuncrank_repro.interpretation.evidence_store import EvidenceStore, literature_display_title
from phenofuncrank_repro.interpretation.evidence_type_registry import GAP_LAYER_TEMPLATES, GAP_LAYER_TYPES, canonical_evidence_type, is_registered
from phenofuncrank_repro.interpretation.live_apis import _concurrent_run, _deduplicate_by_doi, _europepmc_raw, epmc_citations, epmc_references, europepmc_core_abstract, _http_get_json, _http_get_text, multi_query_literature_search, openalex_work_by_doi, openalex_cited_by, openalex_references, openalex_search, semantic_scholar_citations, source_discovery_union_search
from phenofuncrank_repro.interpretation.literature_chunker import chunk_text
from phenofuncrank_repro.interpretation.llm_utils import complete_json
from phenofuncrank_repro.interpretation.logging_utils import increment
from phenofuncrank_repro.interpretation.schemas.literature import CandidateContext, EvidenceGapIntent, RetrievalBudget, RetrievalCandidate, RetrievalHit, RetrievalResult
from phenofuncrank_repro.interpretation.settings import settings
from phenofuncrank_repro.interpretation.source_discovery import build_source_discovery_terms

logger = logging.getLogger(__name__)


_ENRICH_ABSTRACT_CACHE: dict[str, str] = {}


RELATION_REQUIRED_TYPES: dict[str, list[str]] = {

    

    "variant_expression": ["eqtl", "allele_specific", "regulatory"],
    "variant_trait": ["variant_specific", "allele_specific", "trait_association", "mutant_phenotype"],
    "gene_trait": ["mutant_phenotype", "functional_validation", "complementation", "rescue", "transgenic"],
    "gene_mechanism": ["molecular_mechanism", "pathway", "protein_interaction", "expression", "regulatory"],

    

    
    "variant_functional_effect": ["variant_specific", "allele_specific", "functional_validation"],
    "variant_mechanism": ["variant_specific", "molecular_mechanism", "regulatory"],
    "variant_protein_function": ["variant_specific", "molecular_mechanism", "functional_validation"],
    "contradiction_resolution": ["trait_association", "mutant_phenotype", "population_evidence"],
}


RELATION_QUERY_TEMPLATES: dict[str, list[str]] = {
    "variant_expression": [
        "{variant_description} {gene} allele-specific expression {species}",
        "{variant_description} {gene} ASE eQTL transcript abundance {species}",
        "{gene} {ref}>{alt} cis-regulatory expression reporter assay {species}",
        "{gene} eQTL allele-specific expression regulatory assay {species}",
    ],
    "variant_trait": [
        "{variant_description} {gene} natural allele phenotype {species}",
        "{gene} allele haplotype natural variation trait functional validation {species}",
        "{gene} allele replacement genome editing CRISPR phenotype {species}",
    ],
    "gene_trait": [
        "{gene} {trait} knockout mutant loss-of-function {species}",
        "{gene} {trait} overexpression complementation rescue transgenic {species}",
        "{gene} mutant phenotype functional validation {species}",
    ],
    "gene_mechanism": [
        "{gene} pathway biochemical assay protein interaction {species}",
        "{gene} enzyme activity signaling transcriptional regulation molecular mechanism {species}",
        "{gene} pathway mechanism expression regulation {species}",
    ],
    "variant_functional_effect": [
        "{variant_description} {gene} allele functional assay {species}",
        "{variant_description} {gene} natural allele haplotype effect {species}",
        "{gene} allele replacement genome editing CRISPR functional validation {species}",
    ],
    "variant_mechanism": [
        "{variant_description} {gene} variant molecular mechanism {species}",
        "{variant_description} {gene} biochemical assay protein interaction {species}",
        "{gene} allele regulatory assay pathway effect {species}",
    ],
    "variant_protein_function": [
        "{variant_description} {gene} protein activity functional assay {species}",
        "{variant_description} {gene} enzyme activity allele effect {species}",
        "{gene} variant protein function biochemical validation {species}",
    ],
    "contradiction_resolution": [
        "{variant_description} {gene} {trait} conflicting results replication {species}",
        "{gene} {trait} no effect contradictory association independent study {species}",
        "{gene} {trait} condition dependent allele effect validation {species}",
    ],
}


REWRITE_SYSTEM = (
    "You are a plant genetics literature query rewriter. Given one evidence "
    "gap intent and the failed queries, return at most 3 rewritten keyword "
    "queries that are more likely to surface functional or phenotypic "
    "evidence. You may only expand gene synonyms, phenotype synonyms, variant "
    "terminology, experimental terminology, or reformulate the query. You "
    "must NOT invent papers, PMIDs, DOIs, citations, or biological "
    "conclusions. Return JSON only: "
    '{"queries":["q1","q2","q3"],"intent_statement":"why these queries"}.'
)

RCS_SYSTEM = (
    "You are a chunk relevance rater for plant genetics evidence. Given one "
    "evidence gap and up to 10 text chunks, return a 1-10 score for each chunk "
    "id. Scores are used only as a tie-breaker after biological relevance; "
    "never rate a chunk above a chunk whose biological relevance (experimental "
    "phenotype evidence) is stronger. "
    'Return JSON only: {"scores":{"<chunk_id>": 7, ...}}. '
    "Use a distinct static system prefix (no dynamic data)."
)


_TRAIT_TERM_MAP = {
    "plant_height": ["height", "dwarf", "semi-dwarf", "tall"],
    "flowering_time": ["flowering", "heading date", "maturity"],
    "yield": ["yield", "grain", "seed"],
    "ear_width": ["ear width", "kernel"],
    "fiber": ["fiber", "lint", "cotton"],
    "seed_weight": ["seed weight", "100-seed", "grain weight"],
}


def _trait_terms(trait: str) -> list[str]:
    trait = (trait or "").strip()
    terms = _TRAIT_TERM_MAP.get(trait, [])
    if trait and not terms:
        terms = [trait.replace("_", " ").lower()]
    return terms


def _query_hash(query: str) -> str:
    return hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:12]


def _parse_snp_key(snp_key: str) -> dict[str, str]:
    parts = str(snp_key or "").split("|")
    if len(parts) == 6:
        return {
            "species": parts[0], "assembly": parts[1], "chrom": parts[2],
            "pos": parts[3], "ref": parts[4], "alt": parts[5],
        }
    return {"species": "", "assembly": "", "chrom": "", "pos": "", "ref": "", "alt": ""}


def _format_template(template: str, ctx: dict[str, str]) -> str:
    return template.format(**ctx).strip()


_RELATION_ALIASES: dict[str, str] = {
    "variant_to_gene": "variant_functional_effect",
    "variant_gene": "variant_functional_effect",
    "variant_gene_function": "variant_functional_effect",
    "variant_to_expression": "variant_expression",
    "variant_to_trait": "variant_trait",
    "variant_to_phenotype": "variant_trait",
    "gene_to_trait": "gene_trait",
    "gene_to_phenotype": "gene_trait",
    "gene_to_mechanism": "gene_mechanism",
    "gene_to_pathway": "gene_mechanism",
}


def canonical_relation_name(value: str) -> str:
    raw = str(value or "").strip().lower()
    normalized = re.sub(r"[→➝⟶]+|->", "_to_", raw)
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if normalized in RELATION_REQUIRED_TYPES:
        return normalized
    if normalized in _RELATION_ALIASES:
        return _RELATION_ALIASES[normalized]
    if "variant" in normalized and "expression" in normalized:
        return "variant_expression"
    if "variant" in normalized and any(term in normalized for term in ("trait", "phenotype")):
        return "variant_trait"
    if "variant" in normalized and any(term in normalized for term in ("mechan", "pathway")):
        return "variant_mechanism"
    if "variant" in normalized and any(term in normalized for term in ("protein", "function", "effect")):
        return "variant_functional_effect"
    if "gene" in normalized and any(term in normalized for term in ("trait", "phenotype")):
        return "gene_trait"
    if "gene" in normalized and any(term in normalized for term in ("mechan", "pathway")):
        return "gene_mechanism"
    return str(value or "").strip()


def build_gap_intents(
    plan: dict[str, Any],
    *,
    gene: str,
    species: str,
    trait: str,
    snp_id: str,
    is_tier1: bool = False,
    explicit_gaps: list[dict[str, Any]] | None = None,
    variant_description: str = "",
    chrom: str = "",
    pos: int | None = None,
    ref: str = "",
    alt: str = "",
) -> list[EvidenceGapIntent]:
    """A2.1 → A2.2: turn typed gaps into query intents.

    ``explicit_gaps`` is the A1 handoff for a critic-driven retrieval round.
    It takes precedence over GraphRAG's broad profile gaps so a feedback loop
    searches the requested atomic relation rather than silently reverting to
    generic gene/trait retrieval.
    """
    gaps = explicit_gaps if explicit_gaps is not None else (plan.get("gaps") or [])
    if not gaps and explicit_gaps is None and plan.get("evidence_layer_gaps"):
        gaps = plan["evidence_layer_gaps"]
    snp_parts = _parse_snp_key(plan.get("resolved_snp_key") or plan.get("snp_key") or snp_id)
    ctx = {
        "gene": gene or "",
        "species": species or "",
        "trait": trait or "",
        "snp_id": snp_id or "",
        "pos": snp_parts.get("pos") or "",
        "ref": snp_parts.get("ref") or "",
        "alt": snp_parts.get("alt") or "",
        "variant_description": variant_description or (
            f"{chrom}:{pos} {ref}>{alt}" if chrom or pos else snp_id
        ),
        "trait_term": (_trait_terms(trait) or ["phenotype"])[0],
    }
    intents: list[EvidenceGapIntent] = []
    for gap in gaps:
        layer = str(gap.get("layer") or "")
        if not layer:

            link = str(gap.get("missing_link") or "")
            link_lower = link.lower()
            if "expression" in link_lower or "regulatory" in link_lower:
                layer = "regulatory"
            elif link_lower in {"variant_trait", "variant_phenotype"}:
                layer = "variant_specific"
            elif any(k in link_lower for k in ("function", "功能")):
                layer = "gene_function"
            elif any(k in link.lower() for k in ("phenotype", "trait", "性状", "表型")):
                layer = "trait_evidence"
            elif any(k in link.lower() for k in ("regulat", "promoter", "调控", "启动子")):
                layer = "regulatory"
        relation = canonical_relation_name(str(gap.get("missing_link") or "").strip())
        templates = RELATION_QUERY_TEMPLATES.get(relation) or GAP_LAYER_TEMPLATES.get(layer, [])
        required = list(RELATION_REQUIRED_TYPES.get(relation) or GAP_LAYER_TYPES.get(layer, []))
        if not templates:
            continue

        
        if layer in {"gene_function", "pathway", "gene_pathway"} and not gene:
            continue
        queries = [_format_template(t, ctx) for t in templates if _format_template(t, ctx).strip()]

        

        for source_query in build_source_discovery_terms(
            gene=gene,
            species=species,
            trait=trait,
            variant_description=variant_description,
            max_terms=4,
        ):
            if source_query.lower() not in {query.lower() for query in queries}:
                queries.append(source_query)

        

        
        registered_requested = []
        for value in (gap.get("recommended_evidence_types") or []):
            canonical = canonical_evidence_type(str(value))
            if is_registered(canonical):
                registered_requested.append(canonical)
        descriptive_variant = re.sub(r"\s+", " ", str(variant_description or "")).strip()
        coordinate_only = bool(re.fullmatch(
            r"(?:chr)?[^\s:]+:\d+\s+[A-Za-zN.-]+>[A-Za-zN.-]+",
            descriptive_variant,
            flags=re.IGNORECASE,
        ))
        if (
            relation in {"gene_trait", "gene_mechanism"}
            and descriptive_variant
            and not coordinate_only
            and descriptive_variant.lower() not in {str(snp_id or "").lower(), str(gene or "").lower()}
        ):
            identity_query = " ".join(
                part for part in (descriptive_variant[:180], gene, trait, species) if part
            ).strip()
            if identity_query and identity_query.lower() not in {
                str(query).lower() for query in queries
            }:
                queries.append(identity_query)
        intents.append(EvidenceGapIntent(
            gap_id=str(gap.get("gap_id") or f"gap_{layer}"),
            candidate_id=str(gap.get("candidate_id") or snp_id),
            layer=layer,
            missing_link=str(gap.get("missing_link") or ""),
            priority=str(gap.get("priority") or "important"),
            query_intent=f"target relation {relation or layer} evidence for {gene or snp_id}",
            required_evidence_types=sorted(set(required) | set(registered_requested)),
            query_templates=queries,
            budget={
                "max_queries": min(3 + 3, settings.max_queries_per_candidate),
                "max_fulltext_papers": 2 if not is_tier1 else 3,
                "max_rerank_chunks": settings.max_rerank_chunks_per_candidate,
            },
            stop_conditions=["≥1 条 direct 支持且 confidence ≥ 0.8"],
        ))
    return intents


def _to_candidate(article: dict[str, Any], *, gap_id: str, query_hash: str) -> RetrievalCandidate:
    abstract = str(article.get("abstract_snippet") or "")
    return RetrievalCandidate(
        title=str(article.get("title") or ""),
        doi=str(article.get("doi") or ""),
        pmid=str(article.get("pmid") or ""),
        pmcid=str(article.get("pmcid") or ""),
        year=str(article.get("year") or ""),
        journal=str(article.get("journal") or ""),
        authors=str(article.get("authors") or ""),
        abstract_snippet=abstract[: settings.abstract_max_chars],
        source=str(article.get("source") or ""),
        relevance_score=float(article.get("relevance_score") or 0.0),
        discovery_branch=str(article.get("discovery_branch") or ""),
        discovery_provider=str(article.get("discovery_provider") or ""),
        provider_rank=(
            int(article["provider_rank"])
            if article.get("provider_rank") is not None else None
        ),
        normalized_metadata=str(article.get("normalized_metadata") or ""),
        abstract_source=str(article.get("abstract_source") or ""),
        abstract_char_count=len(abstract),
        query_hash=query_hash,
        gap_id=gap_id,
    )


def metadata_search_for_gap(
    gap: EvidenceGapIntent,
    candidate: CandidateContext,
    *,
    rewritten_terms: list[str] | None = None,
    source_discovery_only: bool = False,
    source_terms_override: list[str] | None = None,
    source_discovery_mode: str = "legacy_epmc",
    alias_terms_override: list[str] | None = None,
    source_providers: tuple[str, ...] = ("europepmc", "openalex"),
    document_normalization: bool = False,
    source_pool_k: int | None = None,
) -> tuple[list[RetrievalCandidate], dict[str, int], int]:
    """Run one metadata sweep for a Europe PMC/OpenAlex gap (fail-soft)."""
    source_providers = tuple(
        provider for provider in source_providers
        if str(provider).strip().lower() != "pubmed"
    )
    source_queries = build_source_discovery_terms(
        gene=candidate.gene,
        species=candidate.species,
        trait=candidate.trait,
        variant_description=candidate.variant_description,
        max_terms=4,
    )
    discovery_terms = (
        list(source_terms_override)
        if source_terms_override is not None
        else source_queries
    )
    if source_discovery_only:
        extra_terms = list(dict.fromkeys(discovery_terms))
    else:
        extra_terms = list(dict.fromkeys(source_queries + list(gap.query_templates) + [gap.query_intent]))
    if rewritten_terms:
        extra_terms = list(dict.fromkeys(list(rewritten_terms) + extra_terms))
    if source_discovery_mode == "multi_source_union":
        alias_terms = list(alias_terms_override or [])
        if source_terms_override is not None:
            canonical_terms = list(source_terms_override)
        else:

            

            alias_keys = {str(t).strip().lower() for t in source_queries}
            canonical_terms = [
                str(t).strip() for t in gap.query_templates
                if str(t).strip()
                and str(t).strip().lower() not in alias_keys
                and str(t).strip() != str(gap.query_intent).strip()
            ][:4]
            if not canonical_terms:
                canonical_terms = [
                    " ".join(part for part in (candidate.gene, candidate.trait, candidate.species) if part)
                ]
        lit = source_discovery_union_search(
            canonical_terms=canonical_terms,
            alias_terms=alias_terms,
            gene_name=candidate.gene,
            species=candidate.species,
            trait=candidate.trait,
            variant_description=candidate.variant_description,
            identifiers=[candidate.gene, candidate.snp_id],
            providers=source_providers,
            provider_top_k=10,
            use_alias_branch=bool(alias_terms),
            document_normalization=document_normalization,
        )
        articles = (lit or {}).get("_all_deduped", []) or []
        errors = (lit or {}).get("_source_errors") or {}
        query_count = int((lit or {}).get("_query_count") or 0)
    elif source_discovery_only:
        lit = multi_query_literature_search(
            gene_name=candidate.gene,
            species=candidate.species,
            trait=candidate.trait,
            snp_id=candidate.snp_id or None,
            extra_terms=extra_terms[:8],
            source_discovery_only=True,
            citation_traversal=False,
        )
        articles = (lit or {}).get("_all_deduped", []) or []
        errors = (lit or {}).get("_source_errors") or {}
        query_count = len(extra_terms)
    else:

        
        lit = multi_query_literature_search(
            gene_name=candidate.gene,
            species=candidate.species,
            trait=candidate.trait,
            snp_id=candidate.snp_id or None,
            extra_terms=extra_terms[:8],
        )
        articles = (lit or {}).get("_all_deduped", []) or []
        errors = (lit or {}).get("_source_errors") or {}
        query_count = len(extra_terms)

    

    def _sweep(template: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        try:
            out.extend(_europepmc_raw(template, page_size=5))
        except Exception as exc:
            logger.warning("classic-term sweep EPMC failed for %r: %s", template, exc)
        try:
            out.extend(openalex_search(template, filter_species=None, max_results=5))
        except Exception as exc:
            logger.warning("classic-term sweep OpenAlex failed for %r: %s", template, exc)
        return out

    sweep_terms = (
        discovery_terms[:4]
        if source_discovery_only
        else list(dict.fromkeys(source_queries + list(gap.query_templates)))[:5]
    )
    if source_discovery_mode != "multi_source_union" and not source_discovery_only:
        for part in _concurrent_run([
            lambda t=t: _sweep(t) for t in sweep_terms
        ]):
            if isinstance(part, list):
                articles.extend(part)
    articles = _deduplicate_by_doi(articles)
    if source_discovery_mode != "multi_source_union":
        articles = articles[:40]
    elif source_pool_k is not None:
        articles = articles[:max(1, int(source_pool_k))]
    qh = _query_hash("|".join(extra_terms[:4]))
    pool_k = max(1, int(source_pool_k or settings.metadata_candidate_pool_k))
    candidates = [
        _to_candidate(a, gap_id=gap.gap_id, query_hash=qh)
        for a in articles[:pool_k]
    ]
    for rank, cand in enumerate(candidates, 1):
        cand.metadata_rank = rank
        cand.metadata_score = cand.relevance_score
        cand.text_status = (
            "abstract_available" if (cand.abstract_snippet or "").strip()
            else "metadata_only"
        )


    

    
    def _canonicalize(cand: RetrievalCandidate) -> RetrievalCandidate:
        if not cand.pmid and cand.doi:
            try:
                rows = _europepmc_raw(f'DOI:"{cand.doi}"', page_size=1)
                if rows and rows[0].get("pmid"):
                    cand.pmid = str(rows[0]["pmid"])
            except Exception:
                pass
        elif not cand.doi and cand.pmid:
            try:
                rows = _europepmc_raw(f"EXT_ID:{cand.pmid}", page_size=1)
                if rows and rows[0].get("doi"):
                    cand.doi = str(rows[0]["doi"]).lower()
            except Exception:
                pass
        return cand

    missing_ids = [] if source_discovery_only else [
        c for c in candidates
        if (not c.pmid and c.doi) or (not c.doi and c.pmid)
    ]
    if missing_ids:
        _concurrent_run([
            lambda c=c: _canonicalize(c) for c in missing_ids
        ])
    return candidates, dict(errors), int(query_count)


def llm_rewrite_queries(
    gap: EvidenceGapIntent, *, round_no: int = 1,
) -> list[str]:
    """Single low-reasoning rewrite pass (SOP §4.2.1)."""
    if not settings.literature_engine_enabled or not settings.llm_timeout_seconds:
        return []
    data = complete_json(
        REWRITE_SYSTEM,
        {
            "gap": {"layer": gap.layer, "missing_link": gap.missing_link,
                    "query_intent": gap.query_intent},
            "failed_queries": gap.query_templates[:3],
            "round": round_no,
        },
        max_tokens=300,
        extra_body={"reasoning_effort": "low"},
    )
    queries = [str(q) for q in (data or {}).get("queries", []) if str(q).strip()][:3]
    return queries


def _cache_abstract(
    store: EvidenceStore, cand: RetrievalCandidate, abstract: str, source: str,
) -> None:
    """Persist one enriched abstract in literature_articles for re-use."""
    article = store.literature_article_by_doi(cand.doi) if cand.doi else None
    if article is None and cand.pmid:
        article = store.literature_article_by_provider_id(f"PMID:{cand.pmid}")
    if article is None:
        article_id = store.upsert_literature_article(
            provider=cand.source or "literature",
            provider_id=(
                f"PMID:{cand.pmid}" if cand.pmid
                else (f"DOI:{cand.doi}" if cand.doi else None)
            ),
            doi=cand.doi or None,
            title=literature_display_title(cand.title, doi=cand.doi, pmid=cand.pmid),
            metadata={
                "title": cand.title, "doi": cand.doi, "pmid": cand.pmid,
                "year": cand.year, "authors": cand.authors,
                "journal": cand.journal, "trait": "",
            },
        )
    else:
        article_id = str(article["id"])
    store.set_literature_abstract(article_id, abstract, source=source)


def _fetch_enrichment(cand: RetrievalCandidate) -> tuple[str, str]:
    """Remote abstract fetch: EuropePMC core (PMID) then OpenAlex (DOI)."""
    if cand.pmid:
        abstract = europepmc_core_abstract(cand.pmid)
        if abstract:
            return abstract, "europepmc_core"
    if cand.doi:
        work = openalex_work_by_doi(cand.doi)
        if work:
            abstract = str(work.get("abstract_snippet") or "")
            if abstract:
                return abstract, "openalex_work"
    return "", ""


def _apply_enriched(
    cand: RetrievalCandidate,
    abstract: str,
    source: str,
    store: EvidenceStore | None,
    stats: dict[str, int],
) -> None:
    cand.abstract_snippet = abstract[: settings.abstract_max_chars]
    cand.abstract_char_count = len(abstract)
    cand.text_status = "abstract_enriched"
    cand.abstract_source = source
    cand.enrichment_success = True
    stats["success"] += 1
    if store is not None:
        try:
            _cache_abstract(store, cand, abstract, source)
        except Exception as exc:
            logger.warning("abstract cache write failed: %s", exc)


def _needs_abstract_enrichment(cand: RetrievalCandidate) -> bool:
    "Detect non-empty EuropePMC-lite abstracts that may be truncated."
    abstract = (cand.abstract_snippet or "").strip()
    if not abstract or not cand.pmid:
        return False
    source = (cand.abstract_source or "").lower()

    

    if source and source not in {"europepmc_lite", "europepmc"}:
        return False
    return len(abstract) < max(1, int(settings.abstract_enrichment_min_chars))


def enrich_metadata_candidates(
    candidates: list[RetrievalCandidate],
    store: EvidenceStore | None = None,
) -> dict[str, int]:
    stats = {
        "attempted": 0,
        "success": 0,
        "unavailable": 0,
        "cache_hit": 0,
        "short_abstract_detected": 0,
    }
    if not settings.metadata_enrichment_enabled:
        for cand in candidates:
            if not (cand.abstract_snippet or "").strip():
                cand.text_status = "text_unavailable"
                cand.drop_reason = "text_unavailable"
        stats["unavailable"] = sum(
            1 for c in candidates if c.text_status == "text_unavailable"
        )
        return stats

    pending: list[RetrievalCandidate] = []
    for cand in candidates:
        existing_abstract = (cand.abstract_snippet or "").strip()
        refresh_abstract = _needs_abstract_enrichment(cand)
        if existing_abstract and not refresh_abstract:
            cand.text_status = "abstract_available"
            continue
        cand.enrichment_attempted = True
        stats["attempted"] += 1
        if refresh_abstract:
            stats["short_abstract_detected"] += 1
        abstract, source = "", ""
        if store is not None:
            hit = store.literature_article_abstract(doi=cand.doi, pmid=cand.pmid)
            if hit and (not existing_abstract or len(hit[0]) > len(existing_abstract)):
                abstract, source = hit
                stats["cache_hit"] += 1
        if not abstract:
            key = cand.pmid or cand.doi or ""
            abstract = _ENRICH_ABSTRACT_CACHE.get(key, "")
            if abstract:
                source = "memory_cache"
                stats["cache_hit"] += 1
        if abstract and (not existing_abstract or len(abstract) > len(existing_abstract)):
            _apply_enriched(cand, abstract, source, store, stats)
        else:
            pending.append(cand)

    if pending:
        with ThreadPoolExecutor(max_workers=min(8, len(pending))) as ex:
            futures = {ex.submit(_fetch_enrichment, c): c for c in pending}
            for fut in as_completed(futures):
                cand = futures[fut]
                abstract, source = fut.result()
                if abstract and (not cand.abstract_snippet or len(abstract) > len(cand.abstract_snippet)):
                    key = cand.pmid or cand.doi or ""
                    _ENRICH_ABSTRACT_CACHE[key] = abstract
                    _apply_enriched(cand, abstract, source, store, stats)
                else:
                    if cand.abstract_snippet:
                        cand.text_status = "abstract_available"
                        cand.enrichment_error = "core abstract unavailable"
                    else:
                        cand.text_status = "text_unavailable"
                        cand.enrichment_error = "no abstract/fulltext source"
                        cand.drop_reason = "text_unavailable"
                        stats["unavailable"] += 1
    return stats


def _safe_url(url: str) -> bool:
    if not url or not url.lower().startswith("https://"):
        return False
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in settings.fulltext_whitelist_hosts)


def _strip_xml(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _xml_local_name(tag: str) -> str:
    return str(tag or "").rsplit("}", 1)[-1].lower()


def _xml_text(element: ET.Element) -> str:
    return re.sub(r"\s+", " ", " ".join(element.itertext())).strip()


def _xml_section_label(value: str) -> str:
    """Map JATS titles/attributes to headings understood by the chunker."""
    raw = re.sub(r"\s+", " ", str(value or "")).strip()
    lower = raw.lower()
    if "abstract" in lower:
        return "Abstract"
    if "result" in lower:
        return "Results"
    if "method" in lower or "material" in lower:
        return "Methods"
    if "discussion" in lower:
        return "Discussion"
    if "conclusion" in lower:
        return "Conclusions"
    if "introduction" in lower:
        return "Introduction"
    if "figure" in lower or lower in {"fig", "figures"}:
        return "Figure legends"
    if "supplement" in lower:
        return "Supplementary materials"
    if "acknowledg" in lower:
        return "Acknowledgements"
    if "reference" in lower:
        return "References"
    return raw or "Full text"


def _xml_to_sectioned_text(text: str) -> str:
    """Convert JATS fullTextXML to plain text while retaining section heads.

    Flattening XML with a tag regex loses the Results/Figure legend boundary,
    which makes the downstream section-aware chunker operate as if the paper
    were an unstructured abstract.  This renderer keeps the evidence-bearing
    structure and remains fail-soft for malformed/non-JATS responses.
    """
    try:
        root = ET.fromstring(text or "")
    except (ET.ParseError, TypeError, ValueError):
        return _strip_xml(text)

    lines: list[str] = []

    def append_block(value: str) -> None:
        value = re.sub(r"\s+", " ", value or "").strip()
        if value:
            lines.append(value)

    def render_section(sec: ET.Element) -> None:
        title = next(
            (child for child in list(sec) if _xml_local_name(child.tag) == "title"),
            None,
        )
        raw_label = _xml_text(title) if title is not None else str(sec.attrib.get("sec-type") or "")
        append_block(_xml_section_label(raw_label))
        for child in list(sec):
            name = _xml_local_name(child.tag)
            if name == "title":
                continue
            if name == "sec":
                render_section(child)
                continue
            if name in {"fig", "fig-group"}:
                append_block("Figure legends")
                append_block(_xml_text(child))
                continue
            if name in {"table-wrap", "table-wrap-group", "supplementary-material"}:
                append_block("Supplementary materials")
                append_block(_xml_text(child))
                continue
            append_block(_xml_text(child))


    for element in root.iter():
        if _xml_local_name(element.tag) == "abstract":
            append_block("Abstract")
            append_block(_xml_text(element))

    body = next((element for element in root.iter() if _xml_local_name(element.tag) == "body"), None)
    if body is not None:
        for child in list(body):
            name = _xml_local_name(child.tag)
            if name == "sec":
                render_section(child)
            elif name in {"fig", "fig-group"}:
                append_block("Figure legends")
                append_block(_xml_text(child))
            elif name in {"table-wrap", "table-wrap-group", "supplementary-material"}:
                append_block("Supplementary materials")
                append_block(_xml_text(child))
            else:
                append_block(_xml_text(child))


    back = next((element for element in root.iter() if _xml_local_name(element.tag) == "back"), None)
    if back is not None:
        for child in list(back):
            if _xml_local_name(child.tag) == "sec":
                render_section(child)

    rendered = "\n\n".join(lines).strip()
    return rendered or _strip_xml(text)


def _fetch_oa_text(url: str, *, timeout: float = 30) -> str:
    """SSRF-safe fetch of one OA URL; returns stripped text or ''."""
    if not _safe_url(url):
        return ""
    resp = _http_get_text(url, timeout=timeout)
    if not resp.ok or len(resp.content) >= settings.fulltext_max_bytes:
        return ""
    text = _strip_xml(resp.text)
    return text[: settings.fulltext_max_bytes]


def _unpaywall_url(doi: str) -> str | None:
    """Resolve an OA URL through Unpaywall (SOP §4.2.3 route 3)."""
    try:
        data = _http_get_json(
            f"https://api.unpaywall.org/v2/{doi}?email={settings.unpaywall_email}",
            timeout=20,
        )
        loc = (data or {}).get("best_oa_location") or {}
        return (loc.get("url_for_pdf") or loc.get("url") or "") or None
    except Exception as exc:
        logger.warning("Unpaywall lookup failed for %s: %s", doi, exc)
        return None


def _pmc_oa_url(pmid: str) -> str | None:
    """Resolve a full-text URL through the PMC OA API (SOP §4.2.3 route 2)."""
    try:
        rows = _europepmc_raw(f"EXT_ID:{pmid}", page_size=1)
        pmcid = str((rows[0] if rows else {}).get("pmcid") or "").strip()
        if not pmcid:
            return None

        

        if not pmcid.upper().startswith("PMC"):
            pmcid = f"PMC{pmcid}"
        resp = _http_get_text(
            f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}",
            timeout=20,
        )
        if not resp.ok:
            return None
        for href in re.findall(r'href="([^"]+)"', resp.text):
            if href.lower().endswith((".xml", ".txt", ".pdf", ".html")):
                if _safe_url(href):
                    return href
        return None
    except Exception as exc:
        logger.warning("PMC OA lookup failed for %s: %s", pmid, exc)
        return None


def fetch_fulltext(candidate: RetrievalCandidate) -> tuple[str, str]:
    """Fetch full text with SSRF-safe https + size caps; returns (status, text)."""
    if not settings.fulltext_enabled:
        return "skipped", ""

    

    if candidate.pmid:
        pmcid = str(getattr(candidate, "pmcid", "") or "").strip()
        if not pmcid:
            try:
                rows = _europepmc_raw(f"EXT_ID:{candidate.pmid}", page_size=1)
                pmcid = str((rows[0] if rows else {}).get("pmcid") or "").strip()
            except Exception as exc:
                logger.warning("EPMC metadata lookup failed for %s: %s", candidate.pmid, exc)
        if pmcid and not pmcid.upper().startswith("PMC"):
            pmcid = f"PMC{pmcid}"

        epmc_sources = []
        if pmcid:
            epmc_sources.append(pmcid)
        epmc_sources.append(f"MED/{candidate.pmid}")
        for source_id in epmc_sources:
            url = (
                "https://www.ebi.ac.uk/europepmc/webservices/rest/"
                f"{source_id}/fullTextXML"
            )
            last_error = "unknown"
            for attempt in range(3):
                try:
                    resp = _http_get_text(url, timeout=30)
                    if resp.ok and len(resp.content) < settings.fulltext_max_bytes:
                        cleaned = _xml_to_sectioned_text(resp.text)
                        candidate.fulltext_source = "europepmc_fulltext_xml"
                        candidate.fulltext_char_count = len(cleaned)
                        return "ok", cleaned[: settings.fulltext_max_bytes]
                    if attempt < 2:
                        time.sleep(min(2.0, 0.5 * (2 ** attempt)))
                        continue
                    last_error = f"status={getattr(resp, 'status_code', 'unknown')}"
                except Exception as exc:
                    last_error = str(exc)
                    if attempt < 2:
                        time.sleep(min(2.0, 0.5 * (2 ** attempt)))
                        continue
            logger.warning("EPMC fullTextXML failed for %s: %s", source_id, last_error)

    if candidate.doi:
        try:
            mailto = os.environ.get(
                "RICE_PH_AGENT_OPENALEX_MAILTO", "rice-ph-agent@example.com",
            )
            work = _http_get_json(
                f"https://api.openalex.org/works/https://doi.org/{candidate.doi}"
                f"?mailto={mailto}",
                timeout=20,
            )
            oa_url = ((work or {}).get("open_access") or {}).get("oa_url") or ""
            if _safe_url(oa_url):
                text = _fetch_oa_text(oa_url)
                if text:
                    candidate.fulltext_source = "openalex_oa"
                    candidate.fulltext_char_count = len(text)
                    return "ok", text
        except Exception as exc:
            logger.warning("OpenAlex OA fetch failed for %s: %s", candidate.doi, exc)

    if candidate.doi:
        url = _unpaywall_url(candidate.doi)
        if url:
            text = _fetch_oa_text(url)
            if text:
                candidate.fulltext_source = "unpaywall_oa"
                candidate.fulltext_char_count = len(text)
                return "ok", text

    if candidate.pmid:
        url = _pmc_oa_url(candidate.pmid)
        if url:
            text = _fetch_oa_text(url)
            if text:
                candidate.fulltext_source = "pmc_oa"
                candidate.fulltext_char_count = len(text)
                return "ok", text
    return "failed", ""


def ingest_fulltext(
    candidate: RetrievalCandidate,
    text: str,
    store: EvidenceStore,
    *,
    text_status: str = "fulltext_available",
) -> dict[str, Any]:
    article = None
    if candidate.doi:
        article = store.literature_article_by_doi(candidate.doi.lower())
    if article is None:

        article_id = store.upsert_literature_article(
            provider=candidate.source or "literature",
            provider_id=candidate.pmid or (f"DOI:{candidate.doi}" if candidate.doi else None),
            doi=candidate.doi or None,
            title=literature_display_title(candidate.title, doi=candidate.doi, pmid=candidate.pmid, pmcid=candidate.pmcid),
            metadata={
                "title": candidate.title, "doi": candidate.doi, "pmid": candidate.pmid,
                "pmcid": candidate.pmcid,
                "year": candidate.year, "authors": candidate.authors,
                "journal": candidate.journal, "trait": "",
                "text_status": text_status,
                "abstract_source": candidate.abstract_source,
                "fulltext_source": (
                    candidate.fulltext_source if text_status == "fulltext_available" else ""
                ),
                "fulltext_char_count": (
                    candidate.fulltext_char_count or len(text)
                    if text_status == "fulltext_available" else 0
                ),
            },
        )
    else:
        article_id = article["id"] if article else None
    if not article_id:
        return {"status": "no_article"}
    store.update_literature_metadata(
        article_id,
        {
            "pmid": candidate.pmid,
            "pmcid": candidate.pmcid,
            "text_status": text_status,
            "fulltext_source": (
                candidate.fulltext_source or "fulltext"
                if text_status == "fulltext_available" else ""
            ),
            "fulltext_char_count": len(text) if text_status == "fulltext_available" else 0,
            "abstract_source": candidate.abstract_source,
            "abstract_char_count": candidate.abstract_char_count,
        },
    )
    if text_status.startswith("abstract") and candidate.abstract_snippet:
        try:
            store.set_literature_abstract(
                article_id,
                candidate.abstract_snippet,
                source=candidate.abstract_source or "retrieval_abstract",
            )
        except Exception as exc:
            logger.warning("abstract cache write during ingest failed: %s", exc)
    store.set_literature_fulltext(article_id, text)
    chunked = chunk_text(text)
    embeddings: list[bytes] | None = None
    service = get_embedding_service()
    if chunked.chunks and service.available():
        vectors = service.embed_texts([c.content for c in chunked.chunks])
        if vectors:
            embeddings = [
                struct.pack(f"<{len(v)}f", *v) for v in vectors
            ]
    stats = store.replace_literature_chunks(article_id, chunked.chunks, embeddings=embeddings)
    candidate.article_id = article_id
    return {"status": "ok", "chunks": len(chunked.chunks), "truncated": chunked.chunks_truncated}


_FUNCTIONAL_KEYWORDS = (
    "mutant", "knockout", "overexpression", "complementation", "rescue",
    "transgenic", "crispr", "tilling", "rna interference", "loss-of-function",
    "gain-of-function",
)
_ASSOCIATION_KEYWORDS = (
    "gwas", "qtl", "association", "associated", "mapping", "genome-wide", "population",
)
_COMPUTATIONAL_KEYWORDS = ("in silico", "predicted", "prediction", "computational", "homolog")
_PHENOTYPE_KEYWORDS = ("phenotype", "height", "dwarf", "tall", "steril", "yield", "flowering", "architecture")
_MECHANISM_KEYWORDS = ("signaling", "pathway", "biosynthesis", "enzyme", "oxidase", "kinase", "regulat")
_PROXIMITY_BASE = {1: 1.0, 2: 0.85, 3: 0.65, 4: 0.45}


def classify_evidence_type(text: str) -> str | None:
    """Deterministic evidence-type classification for one chunk (registry vocab)."""
    t = (text or "").lower()
    if any(k in t for k in ("eqtl", "expression qtl")):
        return "eqtl"
    if any(k in t for k in ("rna-seq", "rnaseq", "transcriptome", "rna sequencing")):
        return "rna_seq"
    if any(k in t for k in ("expression", "transcription", "mrna", "transcript level")):
        return "expression"
    if any(k in t for k in ("promoter", "motif", "tfbs", "cis-regulatory", "binding site")):
        return "promoter_motif"
    if any(k in t for k in ("gwas", "genome-wide association")):
        return "gwas"
    if any(k in t for k in ("qtl", "quantitative trait locus")):
        return "qtl"
    variant_terms = (
        "allele", "allele-specific", "allele specific", "haplotype", "variant", "snp",
        "editing", "eqtl",
    )
    effect_terms = (
        "height", "phenotype", "dwarf", "trait", "yield", "effect",
        "variation", "expression", "architecture", "grain", "seed",
        "tiller", "sterility", "flowering", "resistance", "mutation",
    )
    if any(k in t for k in variant_terms) and any(k in t for k in effect_terms):
        return "variant_specific"
    if any(k in t for k in ("mutant phenotype", "showed", "plants", "phenotype", "reduced height", "increased height")):
        return "mutant_phenotype"
    if any(k in t for k in ("mutant", "knockout", "complementation", "rescue", "transgenic", "crispr")):
        return "functional_validation"
    if any(k in t for k in _MECHANISM_KEYWORDS):
        return "molecular_mechanism"
    if any(k in t for k in ("pathway", "go annotation", "gene ontology")):
        return "pathway"
    if any(k in t for k in ("allele frequency", "selection", "population")):
        return "population_evidence"
    return None


def _norm_alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def gap_match_score(text: str, gap: EvidenceGapIntent, gene: str = "") -> float:
    t = (text or "").lower()
    layer = gap.layer
    base = 0.2
    if layer in {"trait_evidence", "phenotype"}:
        if any(k in t for k in _PHENOTYPE_KEYWORDS):
            base = 1.0
    elif layer in {"gene_function", "gene_pathway", "pathway"}:
        if any(k in t for k in ("function", "activity", "enzyme", "domain")):
            base = 1.0
        elif any(k in t for k in _MECHANISM_KEYWORDS):
            base = 0.5
    elif layer == "regulatory":
        if any(k in t for k in ("regulat", "promoter", "expression", "eqtl", "binding")):
            base = 1.0
    elif layer == "variant_specific":
        if any(k in t for k in ("variant", "allele", "snp", "mutation")):
            base = 1.0
    else:
        link = (gap.missing_link or "").lower()
        if link and any(k in t for k in link.split()[:4]):
            base = 0.8

    gene_norm = _norm_alnum(gene)
    if gene_norm and gene_norm not in _norm_alnum(text):
        base = min(base, 0.3)
    return base


def evidence_proximity_score(
    text: str, gap: EvidenceGapIntent, *, missing_text: bool = False,
) -> float:
    if missing_text:
        if settings.missing_text_policy == "unknown":
            return settings.missing_text_neutral_prior
        return 0.0
    from phenofuncrank_repro.interpretation.evidence_type_registry import EVIDENCE_PROXIMITY, EVIDENCE_TYPES

    cls = classify_evidence_type(text)
    if cls is None:
        return 0.0
    level = EVIDENCE_PROXIMITY.get(cls, 4)
    base = _PROXIMITY_BASE[level]
    if cls in gap.required_evidence_types:
        return base

    cls_cat = EVIDENCE_TYPES[cls].gold_category if cls in EVIDENCE_TYPES else None
    same_family = any(
        EVIDENCE_TYPES.get(rt) and EVIDENCE_TYPES[rt].gold_category == cls_cat
        for rt in gap.required_evidence_types
    )
    return round(base * 0.5, 4) if same_family else 0.0


def experimental_type_score(text: str) -> float:
    t = (text or "").lower()
    if any(k in t for k in _FUNCTIONAL_KEYWORDS):
        return 1.0
    if any(k in t for k in _ASSOCIATION_KEYWORDS):
        return 0.7
    if any(k in t for k in _COMPUTATIONAL_KEYWORDS):
        return 0.4
    return 0.2


def species_match_score(text: str, species: str) -> float:
    t = (text or "").lower()
    species_l = (species or "").lower()
    if species_l and species_l in t:
        return 1.0
    if "ortholog" in t or "homolog" in t:
        return 0.5
    if not species_l:
        return 0.4
    return 0.4


def biological_score(
    text: str, gap: EvidenceGapIntent, species: str, gene: str = "",
    *,
    weights: tuple[float, float, float, float] = (0.35, 0.30, 0.20, 0.15),
    missing_text: bool = False,
) -> tuple[float, float, float, float, float]:
    gm = gap_match_score(text, gap, gene=gene)
    prox = evidence_proximity_score(text, gap, missing_text=missing_text)
    ex = experimental_type_score(text)
    sm = species_match_score(text, species)
    score = weights[0] * gm + weights[1] * prox + weights[2] * ex + weights[3] * sm
    return round(score, 4), gm, prox, ex, sm


def _canonical_relation(gap: EvidenceGapIntent) -> str:
    relation = canonical_relation_name(gap.missing_link)

    

    

    
    if relation == str(gap.missing_link or "").strip() and gap.layer in {
        "trait_evidence", "phenotype",
    }:
        return "gene_trait"
    return relation


def _contains_entity(text: str, value: str) -> float:
    text_l = (text or "").lower()
    value_l = (value or "").strip().lower()
    if not value_l:
        return 0.0
    if value_l in text_l:
        return 1.0

    

    stopwords = {"to", "of", "in", "a", "an", "the", "and", "or", "for", "by", "on", "at", "is"}
    tokens = [
        t for t in re.findall(r"[a-z0-9]+", value_l)
        if len(t) >= 2 and t not in stopwords
    ]
    if tokens and all(t in text_l for t in tokens):
        return 0.8
    return 0.0


def _variant_match_score(text: str, gap: EvidenceGapIntent, candidate: CandidateContext) -> float:
    relation = _canonical_relation(gap)
    if not relation.startswith("variant_"):

        
        return 0.5
    descriptions = [
        candidate.snp_id,
        candidate.variant_description,
        f"{candidate.chrom}:{candidate.pos}" if candidate.chrom and candidate.pos else "",
        f"{candidate.ref}>{candidate.alt}" if candidate.ref and candidate.alt else "",
        f"{candidate.ref} to {candidate.alt}" if candidate.ref and candidate.alt else "",
    ]
    exact = max((_contains_entity(text, value) for value in descriptions if value), default=0.0)
    lower = (text or "").lower()
    variant_words = (
        "allele", "haplotype", "variant", "snp", "polymorphism", "mutation",
        "editing", "crispr", "tandem", "repeat", "insertion", "deletion",
        "transposon", "retrotransposon", "structural",
    )
    if exact:
        return exact
    if any(word in lower for word in variant_words):
        return 0.35
    return 0.0


def _trait_match_score(text: str, candidate: CandidateContext) -> float:
    terms = _trait_terms(candidate.trait)
    if not terms:
        return 0.5
    lower = (text or "").lower()
    if any(term in lower for term in terms):
        return 1.0

    
    if candidate.trait.lower() in {"plant_height", "height"} and any(
        term in lower for term in ("dwarf", "internode elongation", "stature")
    ):
        return 0.75
    return 0.0


def _relation_match_score(text: str, gap: EvidenceGapIntent, candidate: CandidateContext) -> float:
    lower = (text or "").lower()
    relation = _canonical_relation(gap)
    variant_words = (
        "allele", "haplotype", "variant", "snp", "polymorphism", "mutation",
        "editing", "crispr", "tandem", "repeat", "insertion", "deletion",
        "transposon", "retrotransposon", "structural",
    )
    expression_words = (
        "expression", "transcript", "mrna", "eqtl", "ase", "allele-specific",
        "allele specific", "reporter", "repression", "chromatin", "methylation",
        "epigenetic", "transcription", "derepress",
    )
    trait_words = tuple(dict.fromkeys(_trait_terms(candidate.trait) + [
        "phenotype", "dwarf", "height", "yield", "flowering", "tolerance", "shattering",
    ]))
    mechanism_words = ("pathway", "signaling", "signalling", "biochemical", "enzyme", "protein interaction", "regulat", "biosynthesis")
    has_variant = any(word in lower for word in variant_words)
    has_expression = any(word in lower for word in expression_words)
    has_trait = any(word in lower for word in trait_words)
    has_gene = _contains_entity(lower, candidate.gene) > 0
    has_experiment = any(word in lower for word in _FUNCTIONAL_KEYWORDS + _ASSOCIATION_KEYWORDS)

    if relation == "variant_expression":
        if has_variant and has_expression and has_gene:
            return 1.0
        if has_expression and has_variant:
            return 0.85
        if has_expression:
            return 0.35
        return 0.0
    if relation == "variant_trait":
        if has_variant and has_trait and has_gene:
            return 1.0
        if has_variant and has_trait:
            return 0.85
        if has_trait and has_experiment:
            return 0.3
        return 0.0
    if relation == "gene_trait":
        if has_gene and has_trait and has_experiment:
            return 1.0
        if has_gene and has_trait:
            return 0.7
        return 0.0
    if relation == "gene_mechanism":
        if has_gene and any(word in lower for word in mechanism_words):
            return 1.0
        if any(word in lower for word in mechanism_words):
            return 0.55
        return 0.0
    return 0.0


def evidence_aware_score(
    text: str,
    gap: EvidenceGapIntent,
    candidate: CandidateContext,
    *,
    text_status: str = "",
    retrieval_relevance: float = 0.0,
) -> dict[str, float]:
    "Compute an target-edge score for a retrieved chunk."
    text_l = text or ""
    variant = _variant_match_score(text_l, gap, candidate)
    gene = _contains_entity(text_l, candidate.gene)
    trait = _trait_match_score(text_l, candidate)
    relation = _relation_match_score(text_l, gap, candidate)
    detected_type = canonical_evidence_type(classify_evidence_type(text_l) or "")
    allowed = {canonical_evidence_type(v) for v in gap.required_evidence_types}
    type_match = 1.0 if detected_type and detected_type in allowed else 0.0
    if not type_match and relation >= 0.85:

        
        type_match = 0.5
    directness = 1.0 if relation >= 0.85 and any(
        word in text_l.lower() for word in _FUNCTIONAL_KEYWORDS + ("ase", "eqtl", "reporter", "allele-specific")
    ) else (0.6 if relation >= 0.55 else 0.0)
    status = (text_status or "").lower()
    provenance = 1.0 if status == "fulltext_available" else 0.75 if "abstract" in status else 0.25 if status == "metadata_only" else 0.0
    relation = _relation_match_score(text_l, gap, candidate)
    target_relation = _canonical_relation(gap)

    

    trait_weight = 0.10 if target_relation in {"variant_trait", "gene_trait"} else 0.0
    score = (
        0.10 * variant
        + 0.10 * gene
        + trait_weight * trait
        + 0.30 * relation
        + 0.15 * type_match
        + 0.15 * directness
        + 0.05 * provenance
        + 0.05 * min(1.0, max(0.0, retrieval_relevance))
    )
    return {
        "variant_match": round(variant, 4),
        "gene_match": round(gene, 4),
        "trait_match": round(trait, 4),
        "relation_match": round(relation, 4),
        "evidence_type_match": round(type_match, 4),
        "experimental_directness": round(directness, 4),
        "provenance_quality": round(provenance, 4),
        "evidence_score": round(score, 4),
    }


def rrf_fusion(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _candidate_identity(candidate: RetrievalCandidate) -> str:
    """Stable identity for unioning shortlist channels before ingestion."""
    if candidate.doi:
        return f"doi:{candidate.doi.lower().strip()}"
    if candidate.pmid:
        return f"pmid:{candidate.pmid.strip()}"
    if candidate.article_id:
        return f"article:{candidate.article_id}"
    return f"title:{candidate.title.strip().lower()}"


def select_gap_candidate_shortlist(
    candidates: list[RetrievalCandidate],
    gap: EvidenceGapIntent,
    context: CandidateContext,
    *,
    metadata_k: int,
) -> tuple[list[RetrievalCandidate], dict[str, int]]:
    "Union metadata and relation-aware shortlists."
    if not candidates:
        return [], {"metadata_shortlist": 0, "relation_shortlist": 0, "relation_shortlist_added": 0}
    k = max(1, min(int(metadata_k), len(candidates)))
    metadata_ranked = sorted(
        candidates,
        key=lambda item: (
            item.metadata_rank if item.metadata_rank is not None else 10**6,
            -item.relevance_score,
        ),
    )
    metadata_top = metadata_ranked[:k]

    def relation_key(item: RetrievalCandidate) -> tuple[float, float, float]:
        text = (item.abstract_snippet or item.title or "")[:2000]
        parts = evidence_aware_score(
            text, gap, context,
            text_status=item.text_status,
            retrieval_relevance=min(1.0, max(0.0, item.relevance_score / 5.0)),
        )
        return (
            parts["evidence_score"],
            parts["relation_match"],
            min(1.0, max(0.0, item.relevance_score / 5.0)),
        )

    relation_top = sorted(candidates, key=relation_key, reverse=True)[:k]
    selected: list[RetrievalCandidate] = []
    seen: set[str] = set()
    for item in [*metadata_top, *relation_top]:
        key = _candidate_identity(item)
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
    return selected, {
        "metadata_shortlist": len(metadata_top),
        "relation_shortlist": len(relation_top),
        "relation_shortlist_added": max(0, len(selected) - len(metadata_top)),
    }


def _variant_candidate_identity_signal(
    text: str, candidate: CandidateContext,
) -> float:
    "Score variant identity signals in metadata/abstract text."
    lowered = (text or "").lower()
    compact_text = re.sub(r"[^a-z0-9]", "", lowered)
    exact_forms = [
        candidate.snp_id,
        f"{candidate.chrom}:{candidate.pos}" if candidate.chrom and candidate.pos else "",
        f"{candidate.ref}>{candidate.alt}" if candidate.ref and candidate.alt else "",
        f"{candidate.ref} to {candidate.alt}" if candidate.ref and candidate.alt else "",
    ]
    for form in exact_forms:
        normalized = str(form or "").strip().lower()
        compact = re.sub(r"[^a-z0-9]", "", normalized)
        if normalized and normalized in lowered or compact and compact in compact_text:
            return 1.0

    description = str(candidate.variant_description or "")
    description_compact = re.sub(r"[^a-z0-9]", "", description.lower())

    

    alias_tokens = re.findall(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)+|[A-Za-z]{2,}\d+[A-Za-z0-9_-]*", description)
    for token in alias_tokens:
        token_lower = token.lower()
        if token_lower == str(candidate.gene or "").lower():
            continue
        token_compact = re.sub(r"[^a-z0-9]", "", token_lower)
        if len(token_compact) >= 3 and (
            token_lower in lowered or token_compact in compact_text
        ):
            return 0.85
    variant_words = (
        "allele", "haplotype", "variant", "snp", "polymorphism", "mutation",
        "mutant", "insertion", "deletion", "editing", "crispr", "eqtl", "ase",
    )
    if any(word in lowered for word in variant_words):
        return 0.35
    return 0.0


def prefilter_variant_candidates(
    candidates: list[RetrievalCandidate],
    gap: EvidenceGapIntent,
    context: CandidateContext,
    *,
    alias_limit: int,
) -> tuple[list[RetrievalCandidate], dict[str, int]]:
    "Reduce alias-only candidates before ingestion while preserving baseline."
    relation = canonical_relation_name(gap.missing_link)
    if not relation.startswith("variant_"):
        return candidates, {
            "variant_prefilter_input": 0,
            "variant_prefilter_output": 0,
            "variant_prefilter_alias_input": 0,
            "variant_prefilter_alias_output": 0,
            "variant_prefilter_dropped": 0,
        }
    aliases = [item for item in candidates if item.discovery_branch == "alias"]
    baseline = [item for item in candidates if item.discovery_branch != "alias"]
    if not aliases or alias_limit <= 0:
        return candidates, {
            "variant_prefilter_input": len(candidates),
            "variant_prefilter_output": len(baseline),
            "variant_prefilter_alias_input": len(aliases),
            "variant_prefilter_alias_output": 0,
            "variant_prefilter_dropped": len(aliases),
        }

    def score(item: RetrievalCandidate) -> tuple[float, float, float]:
        text = " ".join(
            part for part in (
                item.title, item.abstract_snippet, item.normalized_metadata,
            ) if part
        )[:4000]
        identity = _variant_candidate_identity_signal(text, context)
        relation_parts = evidence_aware_score(
            text, gap, context,
            text_status=item.text_status,
            retrieval_relevance=min(1.0, max(0.0, item.relevance_score / 5.0)),
        )
        return (
            round(0.60 * identity + 0.30 * relation_parts["relation_match"] + 0.10 * min(1.0, max(0.0, item.relevance_score / 5.0)), 6),
            identity,
            relation_parts["relation_match"],
        )

    ranked_aliases = sorted(
        aliases,
        key=lambda item: score(item),
        reverse=True,
    )
    kept_aliases = ranked_aliases[:max(0, int(alias_limit))]
    return baseline + kept_aliases, {
        "variant_prefilter_input": len(candidates),
        "variant_prefilter_output": len(baseline) + len(kept_aliases),
        "variant_prefilter_alias_input": len(aliases),
        "variant_prefilter_alias_output": len(kept_aliases),
        "variant_prefilter_dropped": max(0, len(aliases) - len(kept_aliases)),
    }


def _fts_query(text: str) -> str:
    """Robust FTS5 MATCH query: alphanumeric terms joined by OR."""
    terms = re.findall(r"[a-zA-Z0-9]{2,}", (text or "").lower())
    seen: list[str] = []
    for t in terms:
        if t not in seen:
            seen.append(t)
        if len(seen) >= 8:
            break
    return " OR ".join(f'"{t}"' for t in seen)


def retrieve_hybrid(
    gap: EvidenceGapIntent,
    candidate: CandidateContext,
    store: EvidenceStore,
    *,
    top_k: int = 30,
    _trace: dict[str, Any] | None = None,
) -> list[RetrievalHit]:
    """BM25 + dense (when available) with RRF fusion and biological rerank."""
    query = f"{candidate.gene} {gap.query_intent} {' '.join(gap.query_templates[:3])}"
    bm25_rows = store.search_literature_chunks(_fts_query(query), limit=top_k)
    bm25_ids = [r["id"] for r in bm25_rows]

    dense_ids: list[str] = []
    dense_hits: list[dict[str, Any]] = []
    service = get_embedding_service()
    if _trace is not None:
        _trace["dense_requested"] = bool(service.available())
    if service.available():
        vecs = service.embed_texts([query])
        if vecs:
            dense_hits = _dense_search(store, vecs[0], top_k=top_k)
            dense_ids = [h["id"] for h in dense_hits]
    if _trace is not None:
        _trace["bm25_candidates"] = len(bm25_ids)
        _trace["dense_candidates"] = len(dense_ids)

    fused = rrf_fusion([bm25_ids, dense_ids])
    chunk_ids = sorted(fused, key=fused.get, reverse=True)[:top_k]
    if _trace is not None:
        _trace["rrf_candidates"] = len(chunk_ids)
        _trace["dense_active"] = bool(dense_ids)
    rows_by_id = {r["id"]: r for r in bm25_rows}
    dense_by_id = {h["id"]: h for h in dense_hits}

    hits: list[RetrievalHit] = []
    for cid in chunk_ids:
        row = rows_by_id.get(cid) or dense_by_id.get(cid) or {}
        if not row.get("content"):
            stored = store.get_literature_chunks([cid])
            row = stored[0] if stored else row
        article = _article_for_chunk(store, cid)
        if not article:
            continue
        text = str(row.get("content") or "")
        hybrid = fused.get(cid, 0.0)
        ret_rel = round(0.7 * hybrid + 0.3 * (article.get("relevance_score") or 0.0) / 5.0, 4)
        text_status = str(article.get("text_status") or "")
        if text_status not in {
            "metadata_only", "abstract_available", "abstract_enriched",
            "fulltext_available", "text_unavailable",
        }:
            text_status = (
                "abstract_available"
                if (article.get("abstract") or "").strip()
                else "text_unavailable"
            )
        evidence_parts: dict[str, float] = {}
        if settings.bio_rerank_enabled:
            bio, gm, prox, ex, sm = biological_score(
                text, gap, candidate.species, gene=candidate.gene,
            )
            evidence_parts = evidence_aware_score(
                text, gap, candidate, text_status=text_status,
                retrieval_relevance=ret_rel,
            )
            rerank = round(
                settings.retrieval_weight * ret_rel
                + settings.bio_weight * evidence_parts["evidence_score"], 4,
            )
        else:
            bio = gm = prox = ex = sm = 0.0
            rerank = ret_rel
        cand = RetrievalCandidate(
            article_id=str(article.get("id") or ""),
            title=str(article.get("title") or ""),
            doi=str(article.get("doi") or ""),
            pmid=str(article.get("pmid") or ""),
            pmcid=str(article.get("pmcid") or ""),
            year=str(article.get("year") or ""),
            journal=str(article.get("journal") or ""),
            authors=str(article.get("authors") or ""),
            abstract_snippet=str(article.get("abstract_snippet") or "")[:1500],
            source="fulltext",
            relevance_score=float(article.get("relevance_score") or 0.0),
            text_status=text_status,
            abstract_source=str(article.get("abstract_source") or ""),
            fulltext_source=str(article.get("fulltext_source") or ""),
            fulltext_char_count=int(article.get("fulltext_char_count") or 0),
            query_hash=_query_hash(query),
            gap_id=gap.gap_id,
            fulltext_status="ok",
        )
        hits.append(RetrievalHit(
            hit_id=f"{cid}_{gap.gap_id[:8]}",
            candidate=cand,
            chunk_id=cid,
            article_id=str(article.get("id") or ""),
            section=str(row.get("section") or ""),
            span=(int(row.get("start_char") or 0), int(row.get("end_char") or 0)),
            content=text[:2000],
            hybrid_score=round(hybrid, 4),
            retrieval_relevance=ret_rel,
            gap_match=gm,
            evidence_proximity=prox,
            experimental_type=ex,
            species_match=sm,
            biological_score=bio,
            variant_match=evidence_parts.get("variant_match", 0.0),
            gene_match=evidence_parts.get("gene_match", 0.0),
            trait_match=evidence_parts.get("trait_match", 0.0),
            relation_match=evidence_parts.get("relation_match", 0.0),
            evidence_type_match=evidence_parts.get("evidence_type_match", 0.0),
            experimental_directness=evidence_parts.get("experimental_directness", 0.0),
            provenance_quality=evidence_parts.get("provenance_quality", 0.0),
            evidence_score=evidence_parts.get("evidence_score", 0.0),
            rerank_score=rerank,
            query_hash=_query_hash(query),
            gap_id=gap.gap_id,
        ))
    hits.sort(key=lambda h: h.rerank_score, reverse=True)
    return hits[:top_k]


def _dense_search(store: EvidenceStore, vector: list[float], *, top_k: int) -> list[dict[str, Any]]:
    """Cosine search over stored chunk embeddings (naive, corpus is small)."""
    import math

    with store.connect() as con:
        rows = con.execute(
            "SELECT id, embedding FROM literature_chunks WHERE embedding IS NOT NULL",
        ).fetchall()
    scored: list[dict[str, Any]] = []
    for row in rows:
        blob = row["embedding"]
        if not blob:
            continue
        dim = len(blob) // 4
        if dim == 0:
            continue
        vec = struct.unpack(f"<{dim}f", blob)
        dot = sum(a * b for a, b in zip(vector, vec))
        n1 = math.sqrt(sum(v * v for v in vector)) or 1.0
        n2 = math.sqrt(sum(v * v for v in vec)) or 1.0
        scored.append({"id": row["id"], "similarity": dot / (n1 * n2)})
    scored.sort(key=lambda r: r["similarity"], reverse=True)
    return scored[:top_k]


def _article_for_chunk(store: EvidenceStore, chunk_id: str) -> dict[str, Any] | None:
    rows = store.get_literature_chunks([chunk_id])
    if not rows:
        return None
    article_id = rows[0]["article_id"]
    with store.connect() as con:
        row = con.execute("SELECT * FROM literature_articles WHERE id=?", (article_id,)).fetchone()
    if row is None:
        return None
    article = dict(row)
    try:
        metadata = json.loads(article.get("metadata_json") or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        metadata = {}
    article.update({
        "doi": metadata.get("doi") or article.get("doi") or "",
        "pmid": metadata.get("pmid") or "",
        "pmcid": metadata.get("pmcid") or "",
        "year": metadata.get("year") or "",
        "journal": metadata.get("journal") or "",
        "authors": metadata.get("authors") or "",
        "text_status": metadata.get("text_status") or "",
        "abstract_source": metadata.get("abstract_source") or "",
        "fulltext_source": metadata.get("fulltext_source") or "",
        "fulltext_char_count": metadata.get("fulltext_char_count") or 0,
    })
    return article


def citation_traverse(
    hits: list[RetrievalHit],
    gap: EvidenceGapIntent,
    *,
    max_papers: int = 10,
    candidate: CandidateContext | None = None,
) -> list[RetrievalHit]:
    """1-degree forward/backward traversal from high-scored hits (SOP §4.2.6)."""
    if not hits:
        return []
    if settings.traversal_trigger_mode == "topk":
        expand = [
            h for h in hits[: settings.traversal_topk]
            if h.rerank_score >= settings.traversal_threshold and h.candidate.doi
        ]
    else:
        expand = [
            h for h in hits[:3]
            if h.rerank_score >= settings.rerank_threshold and h.candidate.doi
        ]
    out: list[RetrievalHit] = []
    seen_dois: set[str] = set()
    for h in expand:
        doi = h.candidate.doi.lower()
        pmid = h.candidate.pmid
        articles: list[dict[str, Any]] = []
        try:
            articles += openalex_cited_by(doi, max_results=5)
        except Exception as exc:
            logger.warning("forward traversal failed for %s: %s", doi, exc)
        try:
            articles += openalex_references(doi, max_results=5)
        except Exception as exc:
            logger.warning("backward traversal failed for %s: %s", doi, exc)
        try:
            articles += semantic_scholar_citations(doi, "citations", max_results=5)
            articles += semantic_scholar_citations(doi, "references", max_results=5)
        except Exception as exc:
            logger.warning("Semantic Scholar traversal failed for %s: %s", doi, exc)
        if pmid:
            try:
                articles += epmc_references(pmid, max_results=8)
                articles += epmc_citations(pmid, max_results=8)
            except Exception as exc:
                logger.warning("EPMC traversal failed for %s: %s", pmid, exc)
        for art in articles:
            if len(out) >= max_papers:
                break
            art_doi = (art.get("doi") or "").lower()
            if art_doi and art_doi in seen_dois:
                continue
            if art_doi:
                seen_dois.add(art_doi)
            cand = _to_candidate(art, gap_id=gap.gap_id, query_hash=_query_hash(f"traverse:{doi}"))
            cand.fulltext_status = "skipped"
            text = (art.get("abstract_snippet") or art.get("title") or "")[:2000]
            ret_rel = min(1.0, float(art.get("relevance_score") or 0.0) / 5.0)
            ctx = candidate or CandidateContext(
                species="", trait="", gene="", snp_id="",
            )
            if settings.bio_rerank_enabled:
                bio, gm, prox, ex, sm = biological_score(
                    text, gap, ctx.species, gene=ctx.gene,
                    missing_text=not (art.get("abstract_snippet") or "").strip(),
                )
                evidence_parts = evidence_aware_score(
                    text, gap, ctx,
                    text_status=cand.text_status,
                    retrieval_relevance=ret_rel,
                )
                rerank = round(
                    settings.retrieval_weight * ret_rel
                    + settings.bio_weight * evidence_parts["evidence_score"]
                    + 0.1,  
                    4,
                )
            else:
                bio = gm = prox = ex = sm = 0.0
                evidence_parts = {}
                rerank = round(ret_rel + 0.1, 4)
            out.append(RetrievalHit(
                hit_id=f"trav_{len(out)}_{gap.gap_id[:8]}",
                candidate=cand,
                section="traversal",
                span=(0, len(text)),
                content=text,
                retrieval_relevance=round(ret_rel, 4),
                gap_match=gm,
                evidence_proximity=prox,
                experimental_type=ex,
                species_match=sm,
                biological_score=bio,
                variant_match=evidence_parts.get("variant_match", 0.0),
                gene_match=evidence_parts.get("gene_match", 0.0),
                trait_match=evidence_parts.get("trait_match", 0.0),
                relation_match=evidence_parts.get("relation_match", 0.0),
                evidence_type_match=evidence_parts.get("evidence_type_match", 0.0),
                experimental_directness=evidence_parts.get("experimental_directness", 0.0),
                provenance_quality=evidence_parts.get("provenance_quality", 0.0),
                evidence_score=evidence_parts.get("evidence_score", 0.0),
                rerank_score=rerank,
                query_hash=_query_hash(f"traverse:{doi}"),
                gap_id=gap.gap_id,
                traversal_depth=1,
            ))
    out.sort(key=lambda h: h.rerank_score, reverse=True)
    return out


def _citation_traverse_with_context(
    hits: list[RetrievalHit], gap: EvidenceGapIntent,
    *, candidate: CandidateContext,
) -> list[RetrievalHit]:
    try:
        return citation_traverse(hits, gap, candidate=candidate)
    except TypeError as exc:
        if "unexpected keyword argument 'candidate'" not in str(exc):
            raise
        return citation_traverse(hits, gap)


def rcs_tiebreak(hits: list[RetrievalHit], gap: EvidenceGapIntent) -> list[RetrievalHit]:
    if not settings.rcs_enabled or not hits:
        return hits
    top = hits[:10]
    data = complete_json(
        RCS_SYSTEM,
        {"gap": {"layer": gap.layer, "missing_link": gap.missing_link},
         "chunks": [{"id": h.hit_id, "text": h.content[:800]} for h in top]},
        max_tokens=300,
        extra_body={"reasoning_effort": "low"},
    )
    if not data:
        return hits
    scores = {
        str(h.hit_id): float(data.get("scores", {}).get(h.hit_id, 0.0))
        for h in top
    }

    grouped: dict[float, list[RetrievalHit]] = {}
    for h in top:
        grouped.setdefault(h.biological_score, []).append(h)
    for bio, group in grouped.items():
        if len(group) > 1:
            group.sort(key=lambda x: scores.get(x.hit_id, 0.0), reverse=True)
            for idx, h in enumerate(group):
                if h.hit_id in scores and scores[h.hit_id] > h.rerank_score:
                    h.rerank_score = round(scores[h.hit_id], 4)
    return hits


def retrieve_for_gaps(
    intents: list[EvidenceGapIntent],
    candidate: CandidateContext,
    *,
    store: EvidenceStore,
    budget: RetrievalBudget | None = None,
    llm_rewrite: bool = True,
    candidates_override: list[RetrievalCandidate] | None = None,
    source_discovery_mode: str | None = None,
    source_discovery_use_alias: bool | None = None,
    source_discovery_providers: tuple[str, ...] | None = None,
) -> RetrievalResult:
    "A2.2 main entry: gap-driven, budget-retrieval."
    budget = budget or RetrievalBudget(
        max_queries=settings.max_queries_per_candidate,
        max_fulltext_papers=settings.max_fulltext_papers_per_candidate,
        max_rerank_chunks=settings.max_rerank_chunks_per_candidate,
    )
    started = time.time()
    result = RetrievalResult()
    result.execution_trace = {
        "pipeline": [
            "gap_intent", "query_expansion", "hybrid_candidate_retrieval",
            "evidence_aware_reranking", "chunk_output",
        ],
        "old_literature_path_active": False,
        "literature_engine_active": True,
        "bm25_active": True,
        "dense_requested": False,
        "dense_active": False,
        "retrieval_pool_k": max(1, int(settings.retrieval_pool_k)),
        "fulltext_enabled": bool(settings.fulltext_enabled),
        "xml_fulltext_attempts": 0,
        "xml_fulltext_success": 0,
        "abstract_fallback": False,
        "abstract_fallback_count": 0,
        "abstract_enrichment_attempts": 0,
        "gaps": [],
    }
    used = {
        "queries": 0, "fulltext_papers": 0, "abstract_papers": 0,
        "fulltext_attempted": 0, "llm_calls": 0, "wallclock_s": 0.0,
    }
    fetched_articles: dict[str, str] = {}  
    admission_k = max(1, int(settings.abstract_enrichment_k))
    abstract_limit = max(
        admission_k,
        min(int(settings.metadata_candidate_pool_k), admission_k * 2),
    )
    fulltext_k = max(0, min(int(settings.fulltext_fetch_k), budget.max_fulltext_papers))

    for gap in intents:
        gap_trace: dict[str, Any] = {
            "gap_id": gap.gap_id,
            "target_relation": gap.missing_link,
            "query_templates": list(gap.query_templates),
            "queries": [],
            "candidates": 0,
            "bm25_candidates": 0,
            "dense_candidates": 0,
            "rrf_candidates": 0,
            "retrieval_pool_k": max(1, int(settings.retrieval_pool_k)),
            "abstract_chunks": 0,
            "fulltext_chunks": 0,
            "xml_fulltext_attempted": 0,
            "xml_fulltext_success": 0,
            "abstract_fallback": False,
            "metadata_shortlist": 0,
            "relation_shortlist": 0,
            "relation_shortlist_added": 0,
            "source_pool_candidates": 0,
            "variant_prefilter_input": 0,
            "variant_prefilter_output": 0,
            "variant_prefilter_alias_input": 0,
            "variant_prefilter_alias_output": 0,
            "variant_prefilter_dropped": 0,
            "admitted_candidates": 0,
            "reranked_hits": 0,
        }
        result.execution_trace["gaps"].append(gap_trace)
        if time.time() - started > budget.max_wallclock_seconds:
            result.budget_used["budget_exhausted"] = "wallclock"
            increment("literature_engine_budget_exhausted", 1, reason="wallclock")
            break
        if used["queries"] >= budget.max_queries:
            result.budget_used["budget_exhausted"] = "queries"
            increment("literature_engine_budget_exhausted", 1, reason="queries")
            break

        discovery_mode = source_discovery_mode or settings.source_discovery_mode
        if candidates_override is not None:
            candidates = [c.model_copy(deep=True) for c in candidates_override]
            errors, q_used = {}, 0
            for rank, cand in enumerate(candidates, 1):
                if cand.metadata_rank is None:
                    cand.metadata_rank = rank
                    cand.metadata_score = cand.relevance_score
        else:
            use_alias = (
                settings.source_discovery_use_alias
                if source_discovery_use_alias is None else source_discovery_use_alias
            )
            providers = source_discovery_providers or tuple(
                value.strip() for value in settings.source_discovery_providers.split(",")
                if value.strip() and value.strip().lower() != "pubmed"
            )
            alias_terms = (
                build_source_discovery_terms(
                    gene=candidate.gene, species=candidate.species,
                    trait=candidate.trait,
                    variant_description=candidate.variant_description,
                    max_terms=4,
                )
                if use_alias else []
            )
            try:
                candidates, errors, q_used = metadata_search_for_gap(
                    gap, candidate,
                    source_discovery_mode=discovery_mode,
                    alias_terms_override=alias_terms,
                    source_providers=providers,
                )
            except TypeError as exc:

                

                if "unexpected keyword argument" not in str(exc):
                    raise
                candidates, errors, q_used = metadata_search_for_gap(gap, candidate)
            used["queries"] += q_used
            increment("literature_engine_queries", q_used, gap=gap.layer)
            result.query_log.append({
                "gap_id": gap.gap_id, "action": "metadata_search",
                "queries": q_used, "candidates": len(candidates),
                "best_relevance": max((c.relevance_score for c in candidates), default=0.0),
            })
            gap_trace["queries"] = list(gap.query_templates)
            for src, n in errors.items():
                result.source_errors[src] = result.source_errors.get(src, 0) + int(n)

            if llm_rewrite:
                max_rounds = max(1, int(settings.llm_rewrite_rounds))
                for round_no in range(1, max_rounds + 1):
                    best_rel = max(
                        (c.relevance_score for c in candidates), default=0.0,
                    )
                    if best_rel >= 0.5:
                        break  
                    if used["llm_calls"] >= 2 or used["queries"] + 1 > budget.max_queries:
                        break
                    rewritten = llm_rewrite_queries(gap, round_no=round_no)
                    if not rewritten:
                        break
                    used["llm_calls"] += 1
                    more, err2, q2 = metadata_search_for_gap(
                        gap, candidate, rewritten_terms=rewritten,
                    )
                    used["queries"] += q2
                    result.query_log.append({
                        "gap_id": gap.gap_id, "action": "llm_rewrite",
                        "round": round_no, "rewritten": rewritten,
                        "candidates": len(more),
                    })
                    for src, n in err2.items():
                        result.source_errors[src] = result.source_errors.get(src, 0) + int(n)
                    seen = {c.doi.lower() for c in candidates if c.doi}
                    candidates += [
                        c for c in more if not c.doi or c.doi.lower() not in seen
                    ]
                    candidates.sort(key=lambda c: c.relevance_score, reverse=True)
                    for rank, cand in enumerate(candidates, 1):
                        cand.metadata_rank = rank
                        cand.metadata_score = cand.relevance_score
                        if not (cand.abstract_snippet or "").strip():
                            cand.text_status = "metadata_only"


        

        

        
        gap_trace["source_pool_candidates"] = len(candidates)
        if (
            settings.variant_relation_prefilter_enabled
            and discovery_mode == "multi_source_union"
        ):
            candidates, prefilter_stats = prefilter_variant_candidates(
                candidates,
                gap,
                candidate,
                alias_limit=settings.variant_relation_prefilter_alias_k,
            )
            gap_trace.update(prefilter_stats)


        

        
        if settings.pre_ingest_bio_filter_enabled:
            def _bio_key(cand: RetrievalCandidate) -> float:
                text = (cand.abstract_snippet or cand.title or "")[:2000]
                return biological_score(
                    text, gap, candidate.species, gene=candidate.gene,
                    missing_text=not (cand.abstract_snippet or "").strip(),
                )[0]

            candidates.sort(key=_bio_key, reverse=True)
        else:
            candidates.sort(
                key=lambda c: (
                    c.metadata_rank if c.metadata_rank is not None else 10**6,
                    -c.relevance_score,
                ),
            )
        admitted, shortlist_stats = select_gap_candidate_shortlist(
            candidates, gap, candidate, metadata_k=admission_k,
        )
        gap_trace.update(shortlist_stats)
        gap_trace["admitted_candidates"] = len(admitted)
        for rank, cand in enumerate(admitted, 1):
            cand.admission_rank = rank
        admitted_ids = {_candidate_identity(cand) for cand in admitted}
        for cand in candidates:
            if _candidate_identity(cand) in admitted_ids:
                continue
            cand.drop_stage = "admission"
            cand.drop_reason = "admission_cap"


        enrich_stats = enrich_metadata_candidates(admitted, store)
        for k, v in enrich_stats.items():
            result.enrichment[k] = result.enrichment.get(k, 0) + int(v)
        gap_trace["candidates"] = len(candidates)
        gap_trace["abstract_enrichment_attempts"] = int(enrich_stats.get("attempted", 0))
        gap_trace["abstract_enrichment_success"] = int(enrich_stats.get("success", 0))


        for cand in admitted:
            if not cand.doi and not cand.pmid:
                cand.drop_stage = "ingest"
                cand.drop_reason = "source_error"
                continue
            key = (cand.doi or cand.pmid or "").lower()
            if key in fetched_articles:
                cand.fulltext_status = fetched_articles[key]
                continue
            if (
                settings.fulltext_enabled
                and used["fulltext_attempted"] < fulltext_k
            ):
                used["fulltext_attempted"] += 1
                gap_trace["xml_fulltext_attempted"] += 1
                status, text = fetch_fulltext(cand)
                fetched_articles[key] = status
                cand.fulltext_status = status
                if status == "ok":
                    gap_trace["xml_fulltext_success"] += 1
                    ingest_stats = ingest_fulltext(
                        cand, text, store, text_status="fulltext_available",
                    )
                    cand.text_status = "fulltext_available"
                    used["fulltext_papers"] += 1
                    increment("literature_engine_fulltext_ok", 1, source=cand.source)
                    increment(
                        "literature_engine_chunks_created",
                        float(ingest_stats.get("chunks") or 0),
                    )
                    gap_trace["fulltext_chunks"] += int(ingest_stats.get("chunks") or 0)
            else:
                cand.fulltext_status = "skipped"
                fetched_articles[key] = "skipped"

            if cand.text_status != "fulltext_available":
                abstract = (cand.abstract_snippet or "").strip()
                if not abstract:
                    cand.drop_stage = "ingest"
                    cand.drop_reason = "text_unavailable"
                elif used["abstract_papers"] < abstract_limit:
                    gap_trace["abstract_fallback"] = True

                    

                    

                    ingest_stats = ingest_fulltext(
                        cand, f"Abstract\n\n{cand.title}\n\n{abstract}", store,
                        text_status=cand.text_status or "abstract_available",
                    )
                    used["abstract_papers"] += 1
                    cand.fulltext_status = "abstract"
                    increment("literature_engine_abstract_ok", 1, source=cand.source)
                    increment(
                        "literature_engine_chunks_created",
                        float(ingest_stats.get("chunks") or 0),
                    )
                    gap_trace["abstract_chunks"] += int(ingest_stats.get("chunks") or 0)
                else:
                    cand.drop_stage = "ingest"
                    cand.drop_reason = "budget_exhausted"


        hits = retrieve_hybrid(
            gap, candidate, store,
            top_k=max(1, int(settings.retrieval_pool_k)),
            _trace=gap_trace,
        )
        if not hits and candidates:

            
            pseudo: list[RetrievalHit] = []
            for cand in candidates[:3]:
                text = (cand.abstract_snippet or cand.title or "")[:2000]
                ret_rel = min(1.0, cand.relevance_score / 5.0)
                if settings.bio_rerank_enabled:
                    bio, gm, prox, ex, sm = biological_score(
                        text, gap, candidate.species, gene=candidate.gene,
                        missing_text=not (cand.abstract_snippet or "").strip(),
                    )
                    evidence_parts = evidence_aware_score(
                        text, gap, candidate,
                        text_status=cand.text_status,
                        retrieval_relevance=ret_rel,
                    )
                    pseudo_score = round(
                        settings.retrieval_weight * ret_rel
                        + settings.bio_weight * evidence_parts["evidence_score"], 4,
                    )
                else:
                    bio = gm = prox = ex = sm = 0.0
                    evidence_parts = {}
                    pseudo_score = ret_rel
                pseudo.append(RetrievalHit(
                    hit_id=f"cand_{len(pseudo)}_{gap.gap_id[:8]}",
                    candidate=cand,
                    section="metadata",
                    span=(0, len(text)),
                    content=text,
                    retrieval_relevance=round(ret_rel, 4),
                    gap_match=gm,
                    evidence_proximity=prox,
                    experimental_type=ex,
                    species_match=sm,
                    biological_score=bio,
                    variant_match=evidence_parts.get("variant_match", 0.0),
                    gene_match=evidence_parts.get("gene_match", 0.0),
                    trait_match=evidence_parts.get("trait_match", 0.0),
                    relation_match=evidence_parts.get("relation_match", 0.0),
                    evidence_type_match=evidence_parts.get("evidence_type_match", 0.0),
                    experimental_directness=evidence_parts.get("experimental_directness", 0.0),
                    provenance_quality=evidence_parts.get("provenance_quality", 0.0),
                    evidence_score=evidence_parts.get("evidence_score", 0.0),
                    rerank_score=pseudo_score,
                    query_hash=_query_hash(f"candidate:{cand.pmid or cand.doi}"),
                    gap_id=gap.gap_id,
                    text_status=cand.text_status,
                ))
            hits += _citation_traverse_with_context(pseudo, gap, candidate=candidate)
        hybrid_hits = list(hits)
        for rank, h in enumerate(hybrid_hits, 1):
            h.chunk_rank = rank
            h.rerank_rank = rank
            h.text_status = h.candidate.text_status
        hits = rcs_tiebreak(hits, gap)
        degree = max(1, int(settings.traversal_degree))
        frontier = list(hits)
        seen_trav: set[str] = set()
        traversal_added = 0
        for hop in range(degree):
            extra = _citation_traverse_with_context(frontier, gap, candidate=candidate)
            extra = [h for h in extra if h.hit_id not in seen_trav]
            for h in extra:
                seen_trav.add(h.hit_id)
                h.traversal_depth = hop + 1
            if not extra:
                break
            traversal_added += len(extra)
            hits = hits + extra
            frontier = extra[:3]
        used["traversal_expanded"] = used.get("traversal_expanded", 0) + traversal_added
        final_hits = hits[:budget.max_rerank_chunks]
        for rank, h in enumerate(final_hits, 1):
            h.rerank_rank = rank
            h.final_rank = rank
            if h.traversal_depth == 0 and h.chunk_rank is None:
                h.chunk_rank = rank
            h.text_status = h.candidate.text_status
        result.gap_results[gap.gap_id] = final_hits
        result.candidates.extend(candidates)
        gap_trace["reranked_hits"] = len(final_hits)
        gap_trace["top_evidence_scores"] = [
            {
                "hit_id": h.hit_id,
                "evidence_score": h.evidence_score,
                "relation_match": h.relation_match,
                "evidence_type_match": h.evidence_type_match,
                "text_status": h.text_status,
            }
            for h in final_hits[:5]
        ]

    used["wallclock_s"] = round(time.time() - started, 2)
    used["metadata_pool"] = len(result.candidates)
    used["enrichment_attempted"] = int(result.enrichment.get("attempted", 0))
    used["enrichment_success"] = int(result.enrichment.get("success", 0))
    result.budget_used.update(used)
    result.execution_trace["dense_requested"] = any(
        bool(trace.get("dense_requested")) for trace in result.execution_trace["gaps"]
    )
    result.execution_trace["dense_active"] = any(
        bool(trace.get("dense_active")) for trace in result.execution_trace["gaps"]
    )
    result.execution_trace["xml_fulltext_attempts"] = int(used.get("fulltext_attempted", 0))
    result.execution_trace["xml_fulltext_success"] = sum(
        int(trace.get("xml_fulltext_success", 0))
        for trace in result.execution_trace["gaps"]
    )
    result.execution_trace["abstract_fallback_count"] = int(used.get("abstract_papers", 0))
    result.execution_trace["abstract_fallback"] = bool(
        result.execution_trace["abstract_fallback_count"]
    )
    result.execution_trace["abstract_enrichment_attempts"] = int(used.get("abstract_papers", 0))
    result.execution_trace["abstract_ingest_limit"] = int(abstract_limit)
    result.execution_trace["budget_used"] = dict(used)
    return result
