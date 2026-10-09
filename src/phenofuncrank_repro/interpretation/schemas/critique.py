"Critic and evidence-gap schemas (Phase 2/3 of the four-agent refactor)."

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SEMANTIC_ISSUE_TYPES = {
    "direction_mismatch",
    "trait_scope_mismatch",
    "variant_identity_mismatch",
    "allele_identity_mismatch",
    "ortholog_species_extrapolation",
    "mechanism_over_specific",
    "unverified_mediation",
    "tissue_stage_mismatch",
    "association_to_mechanism_leap",
    "contradiction_unresolved",
    "directness_overstated",
    "evidence_gap_misidentified",
}

SEMANTIC_SUPPORT_STATES = {
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
    "CONTRADICTED",
}

SEMANTIC_ACTIONS = {
    "ACCEPT",
    "REVISE",
    "RETRIEVE_MORE_EVIDENCE",
    "HUMAN_REVIEW",
}

GAP_INTENT_RELATIONS = {
    "variant_functional_effect",
    "variant_trait",
    "variant_expression",
    "variant_mechanism",
    "variant_protein_function",
    "gene_trait",
    "gene_mechanism",
    "contradiction_resolution",
}

GAP_RESOLUTION_MODES = {
    "REWRITE_ONLY",
    "RETRIEVE_EVIDENCE",
    "HUMAN_REVIEW",
    "NO_ACTION",
}


class SemanticClaimReview(BaseModel):

    claim_id: str
    support_state: Literal[
        "SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED",
    ]
    semantic_issues: list[str] = Field(default_factory=list)
    decision: Literal["ACCEPT", "REVISE", "RETRIEVE_MORE_EVIDENCE", "HUMAN_REVIEW"]
    allowed_rewrite_level: Literal[
        "fact", "mechanistic_hypothesis", "remove_claim", "none",
    ] | None = None
    evidence_ids_checked: list[str] = Field(default_factory=list)
    rationale: str = ""


class EvidenceGap(BaseModel):
    """A concrete missing link in a candidate's evidence chain."""

    gap_id: str
    candidate_id: str
    missing_link: str
    gap_reason: str = ""
    required_evidence_level: Literal["L1", "L2", "L3", "L4"] | None = None
    target_variant: str = ""
    target_gene: str = ""
    target_trait: str = ""
    priority: Literal["critical", "important", "optional"] = "important"
    recommended_evidence_types: list[str] = Field(default_factory=list)
    suggested_tools: list[str] = Field(default_factory=list)


class EvidenceGapIntent(BaseModel):

    gap_relation: str
    gap_reason: str = ""
    required_evidence_level: Literal["L1", "L2", "L3", "L4"]
    preferred_evidence_types: list[str] = Field(default_factory=list)
    target_variant: str = ""
    target_gene: str = ""
    target_trait: str = ""
    priority: Literal["high", "medium", "low"] = "medium"


class CriticFeedback(BaseModel):
    "Structured output of the optional semantic critic."

    enabled: bool = False
    provider: str = "none"
    model: str = ""
    status: Literal["not_run", "success", "invalid_output", "error"] = "not_run"
    resolution_mode: Literal[
        "REWRITE_ONLY", "RETRIEVE_EVIDENCE", "HUMAN_REVIEW", "NO_ACTION",
    ] = "NO_ACTION"
    next_action: Literal[
        "ACCEPT", "REVISE", "RETRIEVE_MORE_EVIDENCE", "HUMAN_REVIEW",
    ] = "ACCEPT"
    claim_reviews: list[SemanticClaimReview] = Field(default_factory=list)
    gap_intents: list[EvidenceGapIntent] = Field(default_factory=list)
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    error: str | None = None


class ClaimReview(BaseModel):
    """Per-claim verdict produced by A4."""

    claim_id: str
    claim_kind: Literal["supported_fact", "hypothesis", "legacy_combined"] = "legacy_combined"
    relation: str = ""
    evidence_state_level: Literal["L1", "L2", "L3", "L4"] | None = None
    gate_decision: Literal["PASS", "PARTIAL", "BLOCK", "UNKNOWN"] = "UNKNOWN"
    status: Literal[
        "supported",
        "partially_supported",
        "unsupported",
        "contradicted",
    ]
    evidence_ids_checked: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    allowed_revision: str | None = None
    required_evidence_types: list[str] = Field(default_factory=list)
    covered_evidence_types: list[str] = Field(default_factory=list)


class CritiqueReport(BaseModel):
    """Complete A4 output; consumed only by the Supervisor."""

    decision: Literal[
        "ACCEPT",
        "REVISE_WITH_EXISTING_EVIDENCE",
        "RETRIEVE_MORE_EVIDENCE",
        "INSUFFICIENT_EVIDENCE",
        "HUMAN_REVIEW",
    ]
    claim_reviews: list[ClaimReview] = Field(default_factory=list)
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    causal_language_violations: list[str] = Field(default_factory=list)
    factual_consistency_errors: list[str] = Field(default_factory=list)
    evidence_reference_errors: list[str] = Field(default_factory=list)
    accepted_hypothesis_ids: list[str] = Field(default_factory=list)

    
    hybrid_feedback: CriticFeedback | None = None
