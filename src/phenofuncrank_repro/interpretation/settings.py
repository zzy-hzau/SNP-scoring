"Model and resource settings for interpretation."
from dataclasses import dataclass
from pathlib import Path
import os

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Settings:
    extractor_context_k: int = int(os.environ.get("RICE_PH_AGENT_EXTRACTOR_CONTEXT_K", "10"))
    literature_engine_enabled: bool = os.environ.get('RICE_PH_AGENT_LITERATURE_ENGINE_ENABLED', '0').lower() in {'1', 'true', 'yes', 'on'}
    fulltext_enabled: bool = os.environ.get('RICE_PH_AGENT_FULLTEXT_ENABLED', '0').lower() in {'1', 'true', 'yes', 'on'}
    bio_rerank_enabled: bool = os.environ.get('RICE_PH_AGENT_BIO_RERANK_ENABLED', '1').lower() in {'1', 'true', 'yes', 'on'}
    rcs_enabled: bool = os.environ.get('RICE_PH_AGENT_RCS_ENABLED', '0').lower() in {'1', 'true', 'yes', 'on'}
    retrieval_weight: float = float(os.environ.get('RICE_PH_AGENT_RERANK_RETRIEVAL_WEIGHT', '0.4'))
    bio_weight: float = float(os.environ.get('RICE_PH_AGENT_RERANK_BIO_WEIGHT', '0.6'))
    max_fulltext_papers_per_candidate: int = int(os.environ.get('RICE_PH_AGENT_MAX_FULLTEXT_PAPERS', '3'))
    metadata_candidate_pool_k: int = int(os.environ.get('RICE_PH_AGENT_METADATA_POOL_K', '30'))
    abstract_enrichment_k: int = int(os.environ.get('RICE_PH_AGENT_ABSTRACT_ENRICHMENT_K', '10'))
    abstract_enrichment_min_chars: int = int(os.environ.get('RICE_PH_AGENT_ABSTRACT_ENRICHMENT_MIN_CHARS', '900'))
    abstract_max_chars: int = int(os.environ.get('RICE_PH_AGENT_ABSTRACT_MAX_CHARS', '6000'))
    retrieval_pool_k: int = int(os.environ.get('RICE_PH_AGENT_RETRIEVAL_POOL_K', '100'))
    fulltext_fetch_k: int = int(os.environ.get('RICE_PH_AGENT_FULLTEXT_FETCH_K', '5'))
    metadata_enrichment_enabled: bool = os.environ.get('RICE_PH_AGENT_METADATA_ENRICHMENT_ENABLED', '1').lower() in {'1', 'true', 'yes', 'on'}
    pre_ingest_bio_filter_enabled: bool = os.environ.get('RICE_PH_AGENT_PRE_INGEST_BIO_FILTER_ENABLED', '0').lower() in {'1', 'true', 'yes', 'on'}
    llm_rewrite_rounds: int = int(os.environ.get('RICE_PH_AGENT_LLM_REWRITE_ROUNDS', '2'))
    missing_text_policy: str = os.environ.get('RICE_PH_AGENT_MISSING_TEXT_POLICY', 'unknown')
    missing_text_neutral_prior: float = float(os.environ.get('RICE_PH_AGENT_MISSING_TEXT_NEUTRAL_PRIOR', '0.5'))
    max_queries_per_candidate: int = int(os.environ.get('RICE_PH_AGENT_MAX_QUERIES', '12'))
    source_discovery_mode: str = os.environ.get('RICE_PH_AGENT_SOURCE_DISCOVERY_MODE', 'legacy_epmc')
    source_discovery_use_alias: bool = os.environ.get('RICE_PH_AGENT_SOURCE_DISCOVERY_USE_ALIAS', '0').lower() in {'1', 'true', 'yes', 'on'}
    source_discovery_providers: str = os.environ.get('RICE_PH_AGENT_SOURCE_DISCOVERY_PROVIDERS', 'europepmc,openalex')
    variant_relation_prefilter_enabled: bool = os.environ.get('RICE_PH_AGENT_VARIANT_RELATION_PREFILTER_ENABLED', '1').lower() in {'1', 'true', 'yes', 'on'}
    variant_relation_prefilter_alias_k: int = int(os.environ.get('RICE_PH_AGENT_VARIANT_RELATION_PREFILTER_ALIAS_K', '8'))
    max_rerank_chunks_per_candidate: int = int(os.environ.get('RICE_PH_AGENT_MAX_RERANK_CHUNKS', '60'))
    chunk_size_chars: int = int(os.environ.get('RICE_PH_AGENT_CHUNK_SIZE', '1500'))
    chunk_overlap_chars: int = int(os.environ.get('RICE_PH_AGENT_CHUNK_OVERLAP', '150'))
    fulltext_max_bytes: int = int(os.environ.get('RICE_PH_AGENT_FULLTEXT_MAX_BYTES', str(2 * 1024 * 1024)))
    rerank_threshold: float = float(os.environ.get('RICE_PH_AGENT_RERANK_THRESHOLD', '0.6'))
    traversal_degree: int = int(os.environ.get('RICE_PH_AGENT_TRAVERSAL_DEGREE', '1'))
    traversal_trigger_mode: str = os.environ.get('RICE_PH_AGENT_TRAVERSAL_TRIGGER_MODE', 'threshold')
    traversal_topk: int = int(os.environ.get('RICE_PH_AGENT_TRAVERSAL_TOPK', '3'))
    traversal_threshold: float = float(os.environ.get('RICE_PH_AGENT_TRAVERSAL_THRESHOLD', '0.6'))
    embedding_model: str = os.environ.get('RICE_PH_AGENT_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    unpaywall_email: str = os.environ.get('RICE_PH_AGENT_UNPAYWALL_EMAIL', 'rice-ph-agent@example.com')
    fulltext_whitelist_hosts: tuple[str, ...] = ('www.ebi.ac.uk', 'ebi.ac.uk', 'www.ncbi.nlm.nih.gov', 'ncbi.nlm.nih.gov', 'api.unpaywall.org', 'api.openalex.org', 'api.semanticscholar.org')
    database_url: str | None = os.environ.get("RICE_PH_AGENT_DATABASE_URL")
    resource_file: Path = Path(os.environ.get("RICE_PH_AGENT_RESOURCE_FILE", "configs/interpretation/graph_plant_registry.json"))
    enable_graph_guided_retrieval: bool = os.environ.get("RICE_PH_AGENT_GRAPH_GUIDED_RETRIEVAL", "1").lower() in {"1", "true", "yes", "on"}
    parallel_evidence_prefetch_max_candidates: int = int(os.environ.get("RICE_PH_AGENT_PARALLEL_EVIDENCE_PREFETCH_MAX_CANDIDATES", "20"))
    graph_stats_max_bytes: int = int(os.environ.get("RICE_PH_AGENT_GRAPH_STATS_MAX_BYTES", str(256 * 1024 ** 2)))
    graph_ld_max_neighbors: int = int(os.environ.get("RICE_PH_AGENT_GRAPH_LD_MAX_NEIGHBORS", "50"))
    graph_ld_max_distance_bp: int = int(os.environ.get("RICE_PH_AGENT_GRAPH_LD_MAX_DISTANCE_BP", "500000"))
    root: Path = Path(os.environ.get("RICE_PH_AGENT_ROOT", str(PACKAGE_ROOT / "runtime")))
    workspace: Path = Path(os.environ.get("RICE_PH_AGENT_WORKSPACE", str(PACKAGE_ROOT.parent)))
    environment: str = os.environ.get("RICE_PH_AGENT_ENV", "development")
    llm_timeout_seconds: float = float(os.environ.get("RICE_PH_AGENT_LLM_TIMEOUT", "90"))
    agent_max_output_tokens: int = int(os.environ.get("RICE_PH_AGENT_MAX_OUTPUT_TOKENS", "16384"))
    llm_max_prompt_chars: int = int(os.environ.get("RICE_PH_AGENT_LLM_MAX_PROMPT_CHARS", "300000"))
    llm_synthesis_max_tokens: int = int(os.environ.get("RICE_PH_AGENT_LLM_SYNTHESIS_MAX_TOKENS", "32000"))
    hybrid_a4_enabled: bool = os.environ.get("RICE_PH_AGENT_HYBRID_A4_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
    hybrid_a4_model: str = os.environ.get("RICE_PH_AGENT_LLM_MODEL", "deepseek-v4-flash")
    hybrid_a4_base_url: str = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    hybrid_a4_max_tokens: int = int(os.environ.get("RICE_PH_AGENT_HYBRID_A4_MAX_TOKENS", "12000"))
    hybrid_a4_max_evidence_items: int = int(os.environ.get("RICE_PH_AGENT_HYBRID_A4_MAX_EVIDENCE_ITEMS", "80"))
    hybrid_a4_max_claims: int = int(os.environ.get("RICE_PH_AGENT_HYBRID_A4_MAX_CLAIMS", "40"))

    @property
    def db_path(self) -> Path:
        return self.root / "jobs.sqlite3"
    @property
    def evidence_dir(self) -> Path:
        return self.root / "evidence"
    @property
    def artifacts_dir(self) -> Path:
        return self.evidence_dir / "artifacts"

settings = Settings()
