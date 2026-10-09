"Planning schemas for the four-agent refactor (Phase 2)."

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolCallPlan(BaseModel):
    """One deterministic tool invocation approved by the Supervisor."""

    plan_id: str
    task: str
    component: str
    input_refs: list[str] = Field(default_factory=list)
    expected_output_schema: str
    params: dict[str, Any] = Field(default_factory=dict)


class EvidenceQueryPlan(BaseModel):
    """Per-candidate knowledge retrieval plan produced by A2.

    ``selected_tools`` uses the registered tool names from
    ``tool_registry.AGENT_WHITELIST["evidence_retrieval"]``; unregistered names
    are rejected by the registry contract test.
    """

    candidate_id: str
    biological_question: str
    selected_tools: list[str] = Field(default_factory=list)
    query_order: list[str] = Field(default_factory=list)
    stop_conditions: list[str] = Field(default_factory=list)
    expected_evidence_types: list[str] = Field(default_factory=list)
    budget: dict[str, int | float] = Field(default_factory=dict)

    
    graph_context: str = ""
    graph_covered_layers: dict[str, Any] = Field(default_factory=dict)
    graph_gap_ids: list[str] = Field(default_factory=list)
