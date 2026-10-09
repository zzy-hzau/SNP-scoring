"Core message-passing types: ArtifactRef, ErrorDetail, MessageEnvelope."

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ArtifactRef(BaseModel):

    artifact_id: str
    artifact_type: str
    uri: str
    sha256: str
    size_bytes: int = Field(ge=0)
    media_type: str
    schema_version: str
    created_by: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    immutable: bool = True


class ErrorDetail(BaseModel):
    """Structured error reported by an Agent or Gate."""

    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


Sender = Literal[
    "supervisor",

    "evidence_retrieval",
    "hypothesis_reasoning",
    "evidence_critic",

    "tool_router",
    "renderer",
]
Recipient = Sender


class MessageEnvelope(BaseModel):

    message_id: str
    trace_id: str
    run_id: str
    task_id: str
    correlation_id: str | None = None
    sender: Sender
    recipient: Sender
    message_type: str
    schema_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    idempotency_key: str
    status: Literal["success", "partial", "blocked", "error"]

    payload: dict[str, Any] = Field(default_factory=dict)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[ErrorDetail] = Field(default_factory=list)
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
