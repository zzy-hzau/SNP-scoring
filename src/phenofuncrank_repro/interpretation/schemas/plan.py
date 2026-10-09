"Execution plan types: PlanStep and ExecutionPlan."

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


TaskType = Literal[
    "validate_request",
    "standardize_vcf",
    "run_phenotype_prediction",
    "run_allele_swap",
    "run_cadd",
    "compute_ld",
    "compute_population_metrics",
    "retrieve_knowledge",
    "integrate_evidence",
    "render_report",
]

AgentName = Literal[
    "supervisor",
    "vcf_agent",
    "model_agent",
    "evidence_retrieval",
    "hypothesis_reasoning",
    "report_agent",
    "renderer",  
]

OnFailure = Literal["abort", "interrupt", "degrade", "skip"]


class PlanStep(BaseModel):
    """One step in the execution plan with dependencies and failure strategy."""

    step_id: str
    task_type: TaskType
    assigned_agent: AgentName
    depends_on: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    expected_output_schema: str
    required: bool = True
    retry_policy: str = "none"
    on_failure: OnFailure = "abort"


class ExecutionPlan(BaseModel):
    """Full execution plan produced by Supervisor."""

    plan_id: str
    request_id: str
    steps: list[PlanStep]
    terminal_condition: str
