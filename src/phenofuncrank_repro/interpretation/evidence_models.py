"""Versioned, tool-independent domain models for SNP evidence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


def canonical_json(value: Any) -> str:
    """Stable JSON encoding used for cache keys and provenance hashes."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    STALE = "stale"


class EvidenceLevel(str, Enum):
    E0 = "E0"  
    E1 = "E1"  
    E2 = "E2"  
    E3 = "E3"  
    E4 = "E4"  
    E5 = "E5"  
    E6 = "E6"  


class EvidenceScope(str, Enum):
    """The biological level at which an evidence record is directly valid."""

    SNP = "snp_level"
    GENE = "gene_level"
    PATHWAY = "pathway_level"
    DATABASE = "database_level"


EVIDENCE_SCOPE_BY_TYPE = {
    "input_identity": EvidenceScope.SNP,
    "model_effect": EvidenceScope.SNP,
    "functional_score": EvidenceScope.SNP,
    "functional_annotation": EvidenceScope.SNP,
    "variant_context": EvidenceScope.SNP,
    "ld_proxy": EvidenceScope.SNP,
    "gene_link": EvidenceScope.GENE,
    "gene_alias": EvidenceScope.DATABASE,
    "gene_database": EvidenceScope.DATABASE,
    "homology": EvidenceScope.GENE,
    "literature": EvidenceScope.GENE,
    "pathway_membership": EvidenceScope.PATHWAY,
    "pathway_projection": EvidenceScope.PATHWAY,
}


@dataclass(frozen=True)
class SNPIdentity:
    species: str
    assembly: str
    chrom: str
    pos: int
    ref: str
    alt: str
    strand: str = "forward"

    def __post_init__(self) -> None:
        object.__setattr__(self, "species", self.species.strip().lower())
        object.__setattr__(self, "assembly", self.assembly.strip())
        object.__setattr__(self, "chrom", str(self.chrom).strip())
        object.__setattr__(self, "pos", int(self.pos))
        object.__setattr__(self, "ref", self.ref.strip().upper())
        object.__setattr__(self, "alt", self.alt.strip().upper())
        if not self.species or not self.assembly or not self.chrom or self.pos < 1:
            raise ValueError("SNP identity requires species, assembly, chromosome, and positive position")
        if self.ref not in {"A", "C", "G", "T"} or self.alt not in {"A", "C", "G", "T"} or self.ref == self.alt:
            raise ValueError("SNP identity supports only distinct biallelic A/C/G/T alleles")
        if self.strand not in {"forward", "reverse"}:
            raise ValueError("strand must be forward or reverse")

    @property
    def canonical_key(self) -> str:
        return "|".join((self.species, self.assembly, self.chrom, str(self.pos), self.ref, self.alt))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "canonical_key": self.canonical_key}

    @classmethod
    def from_canonical_key(cls, key: str) -> "SNPIdentity":
        parts = key.split("|")
        if len(parts) != 6:
            raise ValueError("canonical SNP key must have six pipe-separated fields")
        return cls(parts[0], parts[1], parts[2], int(parts[3]), parts[4], parts[5])


@dataclass(frozen=True)
class ToolRunSpec:
    tool_name: str
    tool_version: str
    input_hash: str
    params: dict[str, Any]
    resource_hash: str
    resource_class: str = "cpu"

    @property
    def params_hash(self) -> str:
        return content_hash(self.params)

    @property
    def run_key(self) -> str:
        return content_hash({
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "input_hash": self.input_hash,
            "params_hash": self.params_hash,
            "resource_hash": self.resource_hash,
        })


@dataclass(frozen=True)
class EvidenceRecord:
    snp_key: str
    evidence_type: str
    subject: str
    predicate: str
    object: str
    evidence_level: EvidenceLevel
    evidence_scope: EvidenceScope | None = None
    tool_run_id: str | None = None
    value: float | str | None = None
    unit: str | None = None
    direction: str | None = None
    confidence: float | None = None
    scope: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.evidence_scope is None:
            object.__setattr__(self, "evidence_scope", EVIDENCE_SCOPE_BY_TYPE.get(self.evidence_type, EvidenceScope.DATABASE))


@dataclass(frozen=True)
class Claim:
    snp_key: str
    claim_type: str
    statement: str
    evidence_tier: EvidenceLevel
    status: str
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)
