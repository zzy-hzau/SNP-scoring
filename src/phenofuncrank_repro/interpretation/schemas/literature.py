"Literature retrieval engine contracts (SOP §5.1–5.3)."

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EvidenceGapIntent(BaseModel):
    """A2.1 output: what to search for, why, within which budget."""

    gap_id: str
    candidate_id: str
    layer: str
    missing_link: str
    priority: Literal["critical", "important", "optional"] = "important"
    query_intent: str
    required_evidence_types: list[str] = Field(default_factory=list)
    query_templates: list[str] = Field(default_factory=list)
    budget: dict[str, float | int] = Field(default_factory=dict)
    stop_conditions: list[str] = Field(default_factory=list)
    reuse_evidence_ids: list[str] = Field(default_factory=list)
    reuse_claim_ids: list[str] = Field(default_factory=list)


class CandidateContext(BaseModel):
    """Per-candidate context consumed by the retrieval engine."""

    species: str
    trait: str
    gene: str
    snp_id: str
    assembly: str = ""
    region: str = ""
    consequence: str = ""
    variant_description: str = ""
    chrom: str = ""
    pos: int | None = None
    ref: str = ""
    alt: str = ""
    is_tier1: bool = False
    trait_terms: list[str] = Field(default_factory=list)


class RetrievalBudget(BaseModel):
    """Global per-candidate budget (SOP §6.3)."""

    max_queries: int = 12
    max_fulltext_papers: int = 3
    max_rerank_chunks: int = 60
    max_extract_llm_calls: int = 10
    max_support_llm_calls: int = 8
    max_wallclock_seconds: float = 180.0
    cost_cap_cny: float = 0.5


class RetrievalCandidate(BaseModel):
    """One metadata-level article candidate from any source."""

    article_id: str | None = None
    title: str = ""
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    year: str = ""
    journal: str = ""
    authors: str = ""
    abstract_snippet: str = ""
    source: str = ""
    relevance_score: float = 0.0
    metadata_rank: int | None = None
    metadata_score: float = 0.0
    discovery_branch: Literal["canonical", "alias", ""] = ""
    discovery_provider: str = ""
    provider_rank: int | None = None
    normalized_metadata: str = ""
    admission_rank: int | None = None
    text_status: Literal[
        "metadata_only",
        "abstract_available",
        "abstract_enriched",
        "fulltext_available",
        "text_unavailable",
    ] = "metadata_only"
    abstract_source: str = ""
    abstract_char_count: int = 0
    fulltext_source: str = ""
    fulltext_char_count: int = 0
    enrichment_attempted: bool = False
    enrichment_success: bool = False
    enrichment_error: str = ""
    drop_stage: str = ""
    drop_reason: str = ""
    query_hash: str = ""
    gap_id: str = ""
    fulltext_status: Literal["none", "ok", "failed", "skipped"] = "none"


class RetrievalHit(BaseModel):
    """One chunk-level hit with retrieval + biological relevance components."""

    hit_id: str
    candidate: RetrievalCandidate
    chunk_id: str | None = None
    article_id: str = ""
    section: str = ""
    span: tuple[int, int] = (0, 0)
    content: str = ""
    hybrid_score: float = 0.0
    retrieval_relevance: float = 0.0
    gap_match: float = 0.0
    evidence_proximity: float = 0.0
    experimental_type: float = 0.0
    species_match: float = 0.0
    biological_score: float = 0.0
    variant_match: float = 0.0
    gene_match: float = 0.0
    trait_match: float = 0.0
    relation_match: float = 0.0
    evidence_type_match: float = 0.0
    experimental_directness: float = 0.0
    provenance_quality: float = 0.0
    evidence_score: float = 0.0
    rerank_score: float = 0.0
    chunk_rank: int | None = None
    rerank_rank: int | None = None
    final_rank: int | None = None
    drop_stage: str = ""
    drop_reason: str = ""
    text_status: str = ""
    query_hash: str = ""
    gap_id: str = ""
    traversal_depth: int = 0


class RetrievalResult(BaseModel):
    "A2.2 output: per-gap hits plus calculation trails."

    gap_results: dict[str, list[RetrievalHit]] = Field(default_factory=dict)
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    query_log: list[dict[str, Any]] = Field(default_factory=list)
    source_errors: dict[str, int] = Field(default_factory=dict)
    budget_used: dict[str, float | int] = Field(default_factory=dict)
    cache_hits: dict[str, int] = Field(default_factory=dict)
    enrichment: dict[str, Any] = Field(default_factory=dict)
    execution_trace: dict[str, Any] = Field(default_factory=dict)


class ChunkCitation(BaseModel):
    """Exact chunk provenance for one evidence proposal."""

    article_id: str
    section: str
    start_char: int
    end_char: int
    quote: str


class PaperVariantMention(BaseModel):
    "paper-side variant identity extracted from one quote."

    exact_coordinate: str = ""
    ref_alt: str = ""
    rs_id: str = ""
    vg_id: str = ""
    snp_id: str = ""
    amino_acid_change: str = ""
    named_allele: str = ""
    haplotype: str = ""
    mutant_line: str = ""
    structural_variant: str = ""
    identity_basis: list[str] = Field(default_factory=list)
    mention_type: str = "NONE"
    raw_mention: str = ""
    coordinate_text: str = ""
    chromosome: str = ""
    position: int | None = None
    protein_change: str = ""
    structural_event: str = ""
    gene: str = ""
    relation: str = ""
    source_quote: str = ""

    def informative_values(self) -> list[str]:
        values = (
            self.exact_coordinate, self.ref_alt, self.rs_id, self.vg_id,
            self.snp_id, self.amino_acid_change, self.named_allele,
            self.haplotype, self.mutant_line, self.structural_variant,
            self.mention_type if self.mention_type != "NONE" else "",
            self.raw_mention, self.coordinate_text, self.protein_change,
            self.structural_event,
        )
        return [str(value).strip() for value in values if str(value or "").strip()]


class EvidenceProposal(BaseModel):
    """A2.3 output: a verifiable, locatable, typed evidence proposal."""

    proposal_id: str
    gap_id: str
    candidate_id: str
    evidence_type: str
    claim: str
    support: Literal["direct", "partial", "indirect"]
    confidence: float = Field(ge=0, le=1)
    direction: Literal["up", "down", "neutral", "unknown"] = "unknown"
    species_match: Literal["same", "ortholog", "other", "unknown"] = "unknown"

    
    supported_relation: str = ""
    relation_match: Literal["exact", "partial", "none"] = "none"
    species: str = ""
    gene: str = ""
    variant_or_allele: str = ""

    

    variant_mention: PaperVariantMention = Field(default_factory=PaperVariantMention)
    trait: str = ""
    directness: Literal["direct", "partial", "indirect", "unknown"] = "unknown"
    source: dict[str, str] = Field(default_factory=dict)
    chunk: ChunkCitation
    verification_status: Literal["pending", "passed", "reconciled", "refuted"] = "pending"
