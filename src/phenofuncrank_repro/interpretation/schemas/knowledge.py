"Knowledge retrieval schemas."

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from phenofuncrank_repro.interpretation.schemas.messages import ArtifactRef, ErrorDetail
from phenofuncrank_repro.interpretation.schemas.plans import EvidenceQueryPlan


class EvidenceItem(BaseModel):
    """One piece of externally retrieved biological evidence."""

    evidence_id: str
    subject_type: Literal[
        "variant", "gene", "protein", "regulatory_element", "pathway"
    ]
    subject_id: str
    evidence_type: str
    source_name: str
    source_record_id: str | None = None
    source_version: str | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    species: str
    reference_assembly: str | None = None
    statement: str
    directness: Literal[
        "direct",
        "same_species_indirect",
        "orthology_inference",
        "computational_prediction",
    ]
    evidence_level: Literal["A", "B", "C", "D"]
    supports: bool
    stance: Literal["support", "contradict", "neutral", "uncertain"] = "support"
    limitations: list[str] = Field(default_factory=list)
    raw_result: ArtifactRef | None = None

    trait: str | None = None

    
    payload: dict[str, Any] = Field(default_factory=dict)

    

    
    claim: dict[str, Any] = Field(default_factory=dict)

    
    query_hash: str | None = None
    tool_call_id: str | None = None
    variant_key: str | None = None

    source_type: Literal[
        "literature", "database", "computational", "population",
        "prediction", "annotation", "unknown",
    ] = "unknown"
    proximity_level: Literal["L1", "L2", "L3", "L4"] = "L4"
    claim_scope: str = ""
    claim_support: Literal[
        "direct", "partial", "unsupported", "uncertain", "not_checked",
    ] = "not_checked"
    provenance_status: Literal[
        "valid", "invalid", "uncertain", "not_checked",
    ] = "not_checked"
    admission_status: Literal["trusted", "provisional", "discarded"] = "provisional"
    supports_variant: bool | None = None
    supports_gene_trait: bool | None = None
    supports_mechanism: bool | None = None

    provenance: dict[str, Any] = Field(default_factory=dict)


class EvidenceRetrievalAgentResult(BaseModel):
    """Complete result from Knowledge Retrieval Agent."""

    status: Literal["success", "partial", "blocked", "error"]
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    unresolved_entities: list[str] = Field(default_factory=list)
    source_failures: list[ErrorDetail] = Field(default_factory=list)
    evidence_table: ArtifactRef | None = None
    query_log: ArtifactRef | None = None
    llm_evidence_summary: list[dict[str, Any]] = Field(default_factory=list)
    llm_evidence_gaps: list[str] = Field(default_factory=list)
    llm_summary_source: Literal["llm", "deterministic"] = "deterministic"

    query_plans: list[EvidenceQueryPlan] = Field(default_factory=list)
    requested_gap_ids: list[str] = Field(default_factory=list)
    evidence_staged_count: int = Field(default=0, ge=0)
    evidence_source_id: str | None = None

    
    graph_context: dict[str, Any] = Field(default_factory=dict)
    graph_rebuilt: bool = False

    
    tool_call_counts: dict[str, int] = Field(default_factory=dict)
    tool_failures: dict[str, int] = Field(default_factory=dict)
    react_steps: list[dict[str, Any]] = Field(default_factory=list)
    trusted_evidence_items: list[EvidenceItem] = Field(default_factory=list)
    provisional_evidence_items: list[EvidenceItem] = Field(default_factory=list)
    discarded_evidence_items: list[EvidenceItem] = Field(default_factory=list)
    admission_summary: dict[str, int] = Field(default_factory=dict)
    literature_engine_trace: dict[str, Any] = Field(default_factory=dict)

    

    

    literature_records: list[dict[str, Any]] = Field(default_factory=list)
