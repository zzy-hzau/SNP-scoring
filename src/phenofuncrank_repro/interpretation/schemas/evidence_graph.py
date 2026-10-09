"Evidence graph and integration schemas."

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from phenofuncrank_repro.interpretation.schemas.messages import ArtifactRef
from phenofuncrank_repro.interpretation.schemas.evidence_state import MISSING_LINK_RELATIONS


class EvidenceNode(BaseModel):
    """One node in the evidence graph (variant, gene, pathway, trait, etc.)."""

    node_id: str
    node_type: Literal[
        "variant",
        "ld_component",
        "haplotype",
        "gene",
        "transcript",
        "protein",
        "regulatory_element",
        "pathway",
        "trait",
        "breeding_action",
    ]
    label: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class EvidenceEdge(BaseModel):
    """Directed edge connecting two evidence nodes."""

    edge_id: str
    source: str
    target: str
    relation: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    directness: str = "unknown"
    caveats: list[str] = Field(default_factory=list)


class AtomicClaim(BaseModel):
    "One biological proposition in the A3 output."

    claim_id: str
    relation: str
    text: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    status: Literal[
        "supported", "partial", "hypothesis_only", "blocked", "gap_disclosure",
    ] = "hypothesis_only"
    evidence_level: Literal["L1", "L2", "L3", "L4"] = "L4"
    causal_strength: str = "none"
    missing_links: list[str] = Field(default_factory=list)
    gate: dict[str, Any] = Field(default_factory=dict)

    @field_validator("missing_links")
    @classmethod
    def validate_missing_link_relations(cls, values: list[str]) -> list[str]:
        invalid = sorted({str(value) for value in values if str(value) not in MISSING_LINK_RELATIONS})
        if invalid:
            raise ValueError(
                "missing_links must contain canonical relation IDs only; "
                f"invalid={invalid}"
            )
        return sorted(set(values))


class CandidateConclusion(BaseModel):
    """Per-candidate integration conclusion with evidence chain."""

    variant_id: str
    ranking_summary: dict[str, Any] = Field(default_factory=dict)
    proposed_mechanism: str = ""
    evidence_level: Literal["high", "moderate", "low", "insufficient", "A", "B", "C", "D"] = "low"
    confidence: float | None = Field(default=None, ge=0, le=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    conflicting_evidence_ids: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    causal_claim_allowed: bool = False
    recommended_validation: list[str] = Field(default_factory=list)

    alternative_hypotheses: list[str] = Field(default_factory=list)

    claim_type: str = "variant_interpretation"
    inference_status: Literal[
        "supported", "partially_supported", "hypothesis", "hypothesis_only", "unsupported",
    ] = "hypothesis"
    causal_strength: Literal[
        "none", "gene_level", "mechanistic_hypothesis", "variant_level", "variant_causal",
    ] = "none"
    missing_links: list[str] = Field(default_factory=list)
    claim_gate: dict[str, Any] = Field(default_factory=dict)

    
    supported_claims: list[AtomicClaim] = Field(default_factory=list)
    mechanistic_hypotheses: list[AtomicClaim] = Field(default_factory=list)
    max_supported_level: Literal["L1", "L2", "L3", "L4"] = "L4"
    max_supported_conclusion: str = "insufficient_evidence"


class IntegrationResult(BaseModel):
    """Complete result from Biological Evidence Integration Agent."""

    status: Literal["success", "partial", "blocked", "error"] = "success"
    evidence_graph: ArtifactRef | None = None
    conclusions_table: ArtifactRef | None = None
    candidate_conclusions: list[CandidateConclusion] = Field(default_factory=list)
    global_limitations: list[str] = Field(default_factory=list)
    report_facts: list[str] = Field(default_factory=list)
    report_hypotheses: list[str] = Field(default_factory=list)
    mechanism_narrative_source: Literal["llm", "deterministic"] = "deterministic"
    pathway_enrichment: dict[str, Any] | None = None
    evidence_state_version: str = ""
