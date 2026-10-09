"""Machine-readable evidence state shared through the Supervisor."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


EvidenceRelation = Literal[
    "variant_functional_effect",
    "variant_trait",
    "variant_expression",
    "variant_mechanism",
    "variant_protein_function",
    "gene_trait",
    "gene_mechanism",
    "contradiction_resolution",
]


MISSING_LINK_RELATIONS = frozenset({
    "variant_functional_effect",
    "variant_trait",
    "variant_expression",
    "variant_mechanism",
    "variant_protein_function",
    "gene_trait",
    "gene_mechanism",
    "contradiction_resolution",
})


def canonical_missing_links(values: object) -> list[str]:
    """Return only valid, deterministic missing-link relation IDs."""
    if not isinstance(values, (list, tuple, set, frozenset)):
        return []
    return sorted({str(value) for value in values if str(value) in MISSING_LINK_RELATIONS})


class EvidenceStateEdge(BaseModel):
    edge_id: str
    source: str
    relation: EvidenceRelation
    target: str
    status: Literal["supported", "predicted", "missing", "forbidden"]
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_levels: list[str] = Field(default_factory=list)
    directness: list[str] = Field(default_factory=list)
    rationale: str = ""


class EvidenceState(BaseModel):
    """A1-owned deterministic state for one candidate variant."""

    contract_version: str = "evidence-state.v1.1"
    generated_by: str = "supervisor"
    variant_id: str
    gene_id: str = ""
    trait: str = ""
    variant_context: dict[str, Any] = Field(default_factory=dict)
    toolkit_evidence: list[dict[str, Any]] = Field(default_factory=list)
    trusted_evidence_items: list[dict[str, Any]] = Field(default_factory=list)
    provisional_evidence_items: list[dict[str, Any]] = Field(default_factory=list)
    discarded_evidence_items: list[dict[str, Any]] = Field(default_factory=list)
    supported_edges: list[EvidenceStateEdge] = Field(default_factory=list)
    predicted_edges: list[EvidenceStateEdge] = Field(default_factory=list)
    missing_edges: list[EvidenceStateEdge] = Field(default_factory=list)

    
    supported_relations: list[str] = Field(default_factory=list)
    predicted_relations: list[str] = Field(default_factory=list)
    missing_relations: list[str] = Field(default_factory=list)
    contradictory_evidence_ids: list[str] = Field(default_factory=list)
    contradictory_evidence_notes: list[str] = Field(default_factory=list)
    forbidden_conclusions: list[str] = Field(default_factory=list)
    max_supported_level: Literal["L1", "L2", "L3", "L4"] = "L4"
    max_supported_conclusion: str = "insufficient_evidence"
    admission_summary: dict[str, int] = Field(default_factory=dict)


class EvidenceStateBundle(BaseModel):
    """The only evidence handoff A1 exposes to A3/A4."""

    contract_version: str = "evidence-state-bundle.v1.1"
    generated_by: str = "supervisor"
    candidates: dict[str, EvidenceState] = Field(default_factory=dict)
    totals: dict[str, int] = Field(default_factory=dict)
