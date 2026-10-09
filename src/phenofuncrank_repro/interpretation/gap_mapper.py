"""Deterministic mapping from A4 gap relations to A1/A2 retrieval intents."""

from __future__ import annotations

from typing import Any

from phenofuncrank_repro.interpretation.schemas.critique import EvidenceGap, EvidenceGapIntent, GAP_INTENT_RELATIONS


GAP_EVIDENCE_POLICY: dict[str, dict[str, Any]] = {
    "variant_functional_effect": {
        "level": "L1",
        "types": ["natural_allele", "precise_editing", "haplotype", "variant_specific_phenotype"],
    },
    "variant_trait": {
        "level": "L1",
        "types": ["natural_allele", "precise_editing", "haplotype", "variant_specific_phenotype"],
    },
    "variant_expression": {
        "level": "L1",
        "types": ["variant_linked_eqtl", "allele_specific_expression", "reporter_assay", "regulatory_assay"],
    },
    "variant_mechanism": {
        "level": "L1",
        "types": ["precise_editing", "variant_functional_assay", "biochemical_assay"],
    },
    "variant_protein_function": {
        "level": "L1",
        "types": ["precise_editing", "variant_functional_assay", "protein_activity_assay"],
    },
    "gene_trait": {
        "level": "L2",
        "types": ["knockout", "overexpression", "complementation", "rescue", "mutant_phenotype"],
    },
    "gene_mechanism": {
        "level": "L3",
        "types": ["biochemical_assay", "protein_interaction", "pathway_experiment", "molecular_mechanism"],
    },
    "contradiction_resolution": {
        "level": "L2",
        "types": ["independent_replication", "opposing_effect_study", "condition_stratified_assay"],
    },
}


def map_gap_intent(
    intent: EvidenceGapIntent | dict[str, Any],
    *,
    candidate_id: str,
) -> EvidenceGap | None:
    """Convert one canonical gap relation into an A2-consumable EvidenceGap."""
    raw = intent.model_dump() if isinstance(intent, EvidenceGapIntent) else dict(intent)
    relation = str(raw.get("gap_relation") or "")
    if relation not in GAP_INTENT_RELATIONS or relation not in GAP_EVIDENCE_POLICY:
        return None
    policy = GAP_EVIDENCE_POLICY[relation]
    priority = {"high": "critical", "medium": "important", "low": "optional"}.get(
        str(raw.get("priority") or "medium"), "important",
    )
    variant = str(raw.get("target_variant") or candidate_id)
    reason = str(raw.get("gap_reason") or f"No trusted evidence establishes {relation}.")[:1200]
    return EvidenceGap(
        gap_id=f"gap_intent_{candidate_id}_{relation}",
        candidate_id=candidate_id,
        missing_link=relation,
        gap_reason=reason,
        required_evidence_level=policy["level"],
        target_variant=variant,
        target_gene=str(raw.get("target_gene") or ""),
        target_trait=str(raw.get("target_trait") or ""),
        priority=priority,  
        recommended_evidence_types=list(policy["types"]),

        
        suggested_tools=["multi_query_literature_search"],
    )


def map_gap_intents(
    intents: list[EvidenceGapIntent | dict[str, Any]], *, candidate_id: str,
) -> list[EvidenceGap]:
    """Map, deduplicate, and preserve the deterministic policy order."""
    output: list[EvidenceGap] = []
    seen: set[str] = set()
    for raw in intents:
        raw_dict = raw.model_dump() if isinstance(raw, EvidenceGapIntent) else dict(raw)
        target_candidate = str(raw_dict.get("target_variant") or candidate_id)
        gap = map_gap_intent(raw_dict, candidate_id=target_candidate)
        if gap is None or gap.missing_link in seen:
            continue
        seen.add(f"{gap.candidate_id}:{gap.missing_link}")
        output.append(gap)
    return output
