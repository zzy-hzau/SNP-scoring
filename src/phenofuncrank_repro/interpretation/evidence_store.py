"""SQLite-backed provenance, evidence, and content-addressed artifact storage."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from phenofuncrank_repro.interpretation.evidence_models import Claim, EvidenceRecord, RunStatus, SNPIdentity, ToolRunSpec, canonical_json
from phenofuncrank_repro.interpretation.settings import settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return canonical_json(value)


def literature_display_title(title: Any = "", *, doi: Any = "", pmid: Any = "", pmcid: Any = "", provider_id: Any = "") -> str:
    """Return a truthful non-empty literature label for UI/storage."""
    clean = str(title or "").strip()
    if clean:
        return clean
    for label, value in (("PMID", pmid), ("DOI", doi), ("PMCID", pmcid), ("Article", provider_id)):
        value = str(value or "").strip()
        if value:
            return f"{label}:{value}"
    return "Literature record (title unavailable)"


_ANNOTATION_SUMMARY_TYPES = frozenset({
    "protein_domain", "protein_structure", "protein_structure_quality",
    "pathway_annotation", "pathway_membership", "pathway_projection",
    "go_annotation", "motif_scan", "tfbs_overlap", "tfbs_density",
    "tf_family_inference", "tf_family", "regulatory_annotation",
    "gene_context", "gene_link", "sequence_homology",
    "sequence_context_alignment", "rna_structure", "uniprot_mapping",
    "known_gene_proximity", "cds_translation", "functional_annotation",
})

_EVIDENCE_LEVEL_RANK = {"E6": 6, "E5": 5, "E4": 4, "E3": 3, "E2": 2, "E1": 1, "E0": 0}
_EVIDENCE_LEVEL_WEIGHT = {"E6": 1.0, "E5": 0.95, "E4": 0.85, "E3": 0.7, "E2": 0.55, "E1": 0.4, "E0": 0.2}


class ArtifactStore:
    """Content-addressed local artifact store; job directories only keep references."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or settings.artifacts_dir

    def put_file(self, source: Path, *, media_type: str | None = None) -> dict[str, Any]:
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        sha256 = digest.hexdigest()
        target_dir = self.root / sha256[:2] / sha256[2:4] / sha256
        target = target_dir / source.name
        target_dir.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        manifest = target_dir / "manifest.json"
        if not manifest.exists():
            manifest.write_text(_json({
                "sha256": sha256,
                "filename": source.name,
                "bytes": size,
                "media_type": media_type,
                "created_at": _now(),
            }) + "\n", encoding="utf-8")
        return {"sha256": sha256, "path": str(target), "bytes": size, "media_type": media_type}


class EvidenceStore:

    bulk_mode: bool = False

    def __new__(cls, db_path: Path | None = None, artifact_store: ArtifactStore | None = None):
        if cls is EvidenceStore and db_path is None and settings.database_url and settings.database_url.startswith(("postgresql://", "postgres://")):
            from phenofuncrank_repro.interpretation.postgres_store import PostgresEvidenceStore
            return PostgresEvidenceStore(settings.database_url, artifact_store=artifact_store)
        return super().__new__(cls)

    def __init__(self, db_path: Path | None = None, artifact_store: ArtifactStore | None = None) -> None:
        self.db_path = db_path or settings.db_path
        self.artifacts = artifact_store or ArtifactStore()
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    @property
    def graph_root(self) -> Path:
        """Per-species graph databases live next to the main jobs DB."""
        return self.db_path.parent / "graphs"

    def graph_db_path(self, species: str) -> Path:
        return self.graph_root / f"{str(species).strip().lower()}.sqlite3"

    @contextmanager
    def connect_graph(self, species: str) -> Iterator[sqlite3.Connection]:
        """Open the species-scoped graph database (graph_nodes/graph_edges only)."""
        path = self.graph_db_path(species)
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(path), timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=30000")
        if self.__class__.bulk_mode:
            con.execute("PRAGMA synchronous=OFF")
        con.executescript("""
        CREATE TABLE IF NOT EXISTS graph_nodes (
            node_id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            attributes_json TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            hit_count INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(node_type);
        CREATE TABLE IF NOT EXISTS graph_edges (
            edge_id TEXT PRIMARY KEY,
            source TEXT NOT NULL REFERENCES graph_nodes(node_id),
            predicate TEXT NOT NULL,
            target TEXT NOT NULL REFERENCES graph_nodes(node_id),
            properties_json TEXT NOT NULL,
            evidence_ids_json TEXT NOT NULL,
            confidence REAL,
            directness TEXT,
            version INTEGER NOT NULL DEFAULT 1,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            hit_count INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source);
        CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target);
        CREATE INDEX IF NOT EXISTS idx_graph_edges_predicate ON graph_edges(predicate);
        CREATE TABLE IF NOT EXISTS graph_edge_versions (
            edge_key TEXT PRIMARY KEY,
            version INTEGER NOT NULL DEFAULT 0
        );
        """)
        graph_edges_columns = {
            row["name"]
            for row in con.execute("PRAGMA table_info(graph_edges)").fetchall()
        }
        if "version" not in graph_edges_columns:
            con.execute(
                "ALTER TABLE graph_edges ADD COLUMN version INTEGER NOT NULL DEFAULT 1"
            )
        con.commit()
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=30000")
        if self.__class__.bulk_mode:
            con.execute("PRAGMA synchronous=OFF")
        if not self._schema_ready:
            with self._schema_lock:
                if not self._schema_ready:
                    self._create_schema(con)
                    self._schema_ready = True
        con.commit()
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    @staticmethod
    def _create_schema(con: sqlite3.Connection) -> None:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            component TEXT PRIMARY KEY, version INTEGER NOT NULL, applied_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS snps (
            canonical_key TEXT PRIMARY KEY,
            species TEXT NOT NULL, assembly TEXT NOT NULL, chrom TEXT NOT NULL,
            pos INTEGER NOT NULL, ref TEXT NOT NULL, alt TEXT NOT NULL, strand TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tool_runs (
            id TEXT PRIMARY KEY, run_key TEXT NOT NULL UNIQUE,
            tool_name TEXT NOT NULL, tool_version TEXT NOT NULL,
            input_hash TEXT NOT NULL, params_hash TEXT NOT NULL, params_json TEXT NOT NULL,
            resource_hash TEXT NOT NULL, resource_class TEXT NOT NULL,
            status TEXT NOT NULL, artifact_id TEXT, error TEXT,
            created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
            lease_owner TEXT, lease_expires_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_tool_runs_status ON tool_runs(status, resource_class, created_at);
        CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY, sha256 TEXT NOT NULL UNIQUE, path TEXT NOT NULL,
            bytes INTEGER NOT NULL, media_type TEXT, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS evidence_records (
            id TEXT PRIMARY KEY, snp_key TEXT NOT NULL REFERENCES snps(canonical_key),
            tool_run_id TEXT REFERENCES tool_runs(id), artifact_id TEXT REFERENCES artifacts(id),
            evidence_type TEXT NOT NULL, subject TEXT NOT NULL, predicate TEXT NOT NULL,
            object TEXT NOT NULL, value_json TEXT, unit TEXT, direction TEXT,
            confidence REAL, evidence_level TEXT NOT NULL, scope_json TEXT NOT NULL,
            payload_json TEXT NOT NULL, external_id TEXT, created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_evidence_snp ON evidence_records(snp_key, evidence_type);
        CREATE INDEX IF NOT EXISTS idx_snps_coord ON snps(species, chrom, pos);
        CREATE TABLE IF NOT EXISTS claims (
            id TEXT PRIMARY KEY, snp_key TEXT NOT NULL REFERENCES snps(canonical_key),
            claim_type TEXT NOT NULL, statement TEXT NOT NULL, evidence_tier TEXT NOT NULL,
            status TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS claim_evidence (
            claim_id TEXT NOT NULL REFERENCES claims(id), evidence_id TEXT NOT NULL REFERENCES evidence_records(id),
            relation TEXT NOT NULL CHECK(relation IN ('supports','contradicts')),
            PRIMARY KEY(claim_id, evidence_id, relation)
        );
        CREATE TABLE IF NOT EXISTS evidence_relations (
            source_evidence_id TEXT NOT NULL REFERENCES evidence_records(id),
            target_evidence_id TEXT NOT NULL REFERENCES evidence_records(id),
            relation TEXT NOT NULL, reason TEXT, created_at TEXT NOT NULL,
            PRIMARY KEY(source_evidence_id, target_evidence_id, relation)
        );
        CREATE TABLE IF NOT EXISTS literature_articles (
            id TEXT PRIMARY KEY, provider TEXT NOT NULL, provider_id TEXT,
            doi TEXT, title TEXT, metadata_json TEXT NOT NULL, content TEXT,
            content_hash TEXT,
            abstract TEXT, abstract_source TEXT, abstract_hash TEXT, abstract_fetched_at TEXT,
            fetched_at TEXT NOT NULL, UNIQUE(provider, provider_id), UNIQUE(doi)
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS literature_fts USING fts5(
            article_id UNINDEXED, title, content, tokenize='unicode61'
        );
        CREATE TABLE IF NOT EXISTS literature_chunks (
            id TEXT PRIMARY KEY,
            article_id TEXT NOT NULL REFERENCES literature_articles(id),
            chunk_index INTEGER NOT NULL,
            section TEXT,
            start_char INTEGER, end_char INTEGER,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            embedding BLOB,
            created_at TEXT NOT NULL,
            UNIQUE(article_id, chunk_index)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_article ON literature_chunks(article_id);
        CREATE VIRTUAL TABLE IF NOT EXISTS literature_chunk_fts USING fts5(
            chunk_id UNINDEXED, article_id UNINDEXED, section, content, tokenize='unicode61'
        );
        CREATE TABLE IF NOT EXISTS annotation_summary (
            key TEXT PRIMARY KEY,
            species TEXT NOT NULL, assembly TEXT NOT NULL DEFAULT '',
            gene_id TEXT, snp_key TEXT,
            annotation_type TEXT NOT NULL,
            subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL,
            source_name TEXT, evidence_level TEXT,
            payload_json TEXT NOT NULL,
            first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL,
            hit_count INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_annotation_gene ON annotation_summary(species, gene_id, annotation_type);
        CREATE INDEX IF NOT EXISTS idx_annotation_snp ON annotation_summary(snp_key, annotation_type);
        CREATE TABLE IF NOT EXISTS active_topics (
            topic_key TEXT PRIMARY KEY,
            kind TEXT NOT NULL DEFAULT 'gene',
            species TEXT NOT NULL, trait TEXT NOT NULL DEFAULT '',
            gene TEXT, snp TEXT,
            last_refreshed_at TEXT, last_error TEXT,
            refresh_count INTEGER NOT NULL DEFAULT 0,
            first_seen_at TEXT NOT NULL, hit_count INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_topics_due ON active_topics(last_refreshed_at, refresh_count);
        CREATE TABLE IF NOT EXISTS knowledge_evolution (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            component TEXT NOT NULL, action TEXT NOT NULL,
            detail_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_evolution_created ON knowledge_evolution(created_at);
        CREATE TABLE IF NOT EXISTS graph_nodes (
            node_id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            attributes_json TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            hit_count INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(node_type);
        CREATE TABLE IF NOT EXISTS graph_edges (
            edge_id TEXT PRIMARY KEY,
            source TEXT NOT NULL REFERENCES graph_nodes(node_id),
            predicate TEXT NOT NULL,
            target TEXT NOT NULL REFERENCES graph_nodes(node_id),
            properties_json TEXT NOT NULL,
            evidence_ids_json TEXT NOT NULL,
            confidence REAL,
            directness TEXT,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            hit_count INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source);
        CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target);
        CREATE INDEX IF NOT EXISTS idx_graph_edges_predicate ON graph_edges(predicate);
        CREATE TABLE IF NOT EXISTS graph_edge_versions (
            edge_key TEXT PRIMARY KEY,
            version INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS provisional_sources (
            source_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL UNIQUE,
            species TEXT NOT NULL,
            trait TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'user_run',
            population_label TEXT NOT NULL DEFAULT 'user_panel',
            input_path TEXT,
            vcf_result_json TEXT,
            qc_status TEXT NOT NULL DEFAULT 'pending',
            gate_results_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            evaluated_at TEXT,
            promoted_at TEXT,
            rejected_at TEXT,
            error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_provisional_sources_status
            ON provisional_sources(qc_status, created_at);
        CREATE TABLE IF NOT EXISTS provisional_evidence (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES provisional_sources(source_id) ON DELETE CASCADE,
            snp_key TEXT NOT NULL,
            tool_run_id TEXT,
            artifact_id TEXT,
            evidence_type TEXT NOT NULL,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            value_json TEXT,
            unit TEXT,
            direction TEXT,
            confidence REAL,
            evidence_level TEXT NOT NULL,
            scope_json TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            external_id TEXT,
            created_at TEXT NOT NULL,
            qc_status TEXT NOT NULL DEFAULT 'pending',
            promoted_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_provisional_evidence_source
            ON provisional_evidence(source_id, evidence_type);
        CREATE INDEX IF NOT EXISTS idx_provisional_evidence_snp
            ON provisional_evidence(snp_key);
        CREATE TABLE IF NOT EXISTS population_layers (
            species TEXT NOT NULL,
            chrom TEXT NOT NULL,
            pos INTEGER NOT NULL,
            ref TEXT NOT NULL,
            alt TEXT NOT NULL,
            snp_key TEXT NOT NULL,
            population TEXT NOT NULL,
            source TEXT NOT NULL,
            af REAL,
            ac INTEGER,
            an INTEGER,
            callable_rate REAL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (snp_key, population, source)
        );
        CREATE INDEX IF NOT EXISTS idx_population_layers_coord
            ON population_layers(species, chrom, pos);
        CREATE INDEX IF NOT EXISTS idx_population_layers_snp
            ON population_layers(snp_key);
        """)

        
        columns = {row["name"] for row in con.execute("PRAGMA table_info(evidence_records)").fetchall()}
        if "external_id" not in columns:
            con.execute("ALTER TABLE evidence_records ADD COLUMN external_id TEXT")
            con.execute("CREATE INDEX IF NOT EXISTS idx_evidence_external ON evidence_records(external_id)")
            try:
                con.execute("""UPDATE evidence_records SET external_id = json_extract(payload_json, '$.evidence_id')
                    WHERE external_id IS NULL AND payload_json LIKE '%"evidence_id"%'""")
            except sqlite3.Error:
                pass
        con.execute("CREATE INDEX IF NOT EXISTS idx_evidence_external ON evidence_records(external_id)")

        provisional_columns = {
            row["name"]
            for row in con.execute("PRAGMA table_info(provisional_sources)").fetchall()
        }
        if provisional_columns and "source" not in provisional_columns:
            con.execute(
                "ALTER TABLE provisional_sources ADD COLUMN source TEXT NOT NULL DEFAULT 'user_run'"
            )
        provisional_columns = {
            row["name"]
            for row in con.execute("PRAGMA table_info(provisional_sources)").fetchall()
        }
        if "population_label" not in provisional_columns:
            con.execute(
                "ALTER TABLE provisional_sources ADD COLUMN population_label TEXT NOT NULL DEFAULT 'user_panel'"
            )

        

        
        provisional_sources_sql = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='provisional_sources'"
        ).fetchone()
        if provisional_sources_sql and (
            "run_id TEXT NOT NULL UNIQUE" in (provisional_sources_sql["sql"] or "")
            or "UNIQUE(run_id)" in (provisional_sources_sql["sql"] or "")
        ):
            con.execute("PRAGMA foreign_keys=OFF")
            con.execute("DROP INDEX IF EXISTS idx_provisional_sources_status")
            con.executescript("""
            CREATE TABLE provisional_sources_new (
                source_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                species TEXT NOT NULL,
                trait TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'user_run',
                population_label TEXT NOT NULL DEFAULT 'user_panel',
                input_path TEXT,
                vcf_result_json TEXT,
                qc_status TEXT NOT NULL DEFAULT 'pending',
                gate_results_json TEXT NOT NULL DEFAULT '{}',
                verification_json TEXT,
                created_at TEXT NOT NULL,
                evaluated_at TEXT,
                promoted_at TEXT,
                rejected_at TEXT,
                error TEXT
            );
            """)
            old_provisional_columns = {
                row["name"]
                for row in con.execute("PRAGMA table_info(provisional_sources)").fetchall()
            }
            columns = [
                "source_id", "run_id", "species", "trait", "source", "population_label", "input_path",
                "vcf_result_json", "qc_status", "gate_results_json", "verification_json", "created_at",
                "evaluated_at", "promoted_at", "rejected_at", "error",
            ]
            columns = [c for c in columns if c in old_provisional_columns]
            con.execute(
                f"INSERT INTO provisional_sources_new ({', '.join(columns)}) "
                f"SELECT {', '.join(columns)} FROM provisional_sources",
            )
            con.execute("DROP TABLE provisional_sources")
            con.execute("ALTER TABLE provisional_sources_new RENAME TO provisional_sources")
            con.execute(
                "CREATE INDEX idx_provisional_sources_status ON provisional_sources(qc_status, created_at)"
            )
            con.execute("PRAGMA foreign_keys=ON")

        

        
        graph_edges_sql = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='graph_edges'"
        ).fetchone()
        if graph_edges_sql and "UNIQUE(source, predicate, target)" in (graph_edges_sql["sql"] or ""):
            con.execute("DROP INDEX IF EXISTS idx_graph_edges_source")
            con.execute("DROP INDEX IF EXISTS idx_graph_edges_target")
            con.execute("DROP INDEX IF EXISTS idx_graph_edges_predicate")
            con.execute("ALTER TABLE graph_edges RENAME TO graph_edges_legacy")
            con.executescript("""
            CREATE TABLE graph_edges (
                edge_id TEXT PRIMARY KEY,
                source TEXT NOT NULL REFERENCES graph_nodes(node_id),
                predicate TEXT NOT NULL,
                target TEXT NOT NULL REFERENCES graph_nodes(node_id),
                properties_json TEXT NOT NULL,
                evidence_ids_json TEXT NOT NULL,
                confidence REAL,
                directness TEXT,
                version INTEGER NOT NULL DEFAULT 1,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 1
            );
            """)
            columns = [
                "edge_id", "source", "predicate", "target", "properties_json",
                "evidence_ids_json", "confidence", "directness",
                "first_seen_at", "last_seen_at", "hit_count",
            ]
            con.execute(
                f"INSERT INTO graph_edges ({', '.join(columns)}) "
                f"SELECT {', '.join(columns)} FROM graph_edges_legacy",
            )
            con.execute("DROP TABLE graph_edges_legacy")
            con.execute("CREATE INDEX idx_graph_edges_source ON graph_edges(source)")
            con.execute("CREATE INDEX idx_graph_edges_target ON graph_edges(target)")
            con.execute("CREATE INDEX idx_graph_edges_predicate ON graph_edges(predicate)")
        graph_edges_columns = {
            row["name"]
            for row in con.execute("PRAGMA table_info(graph_edges)").fetchall()
        }
        if "version" not in graph_edges_columns:
            con.execute(
                "ALTER TABLE graph_edges ADD COLUMN version INTEGER NOT NULL DEFAULT 1"
            )
        literature_columns = {
            row["name"]
            for row in con.execute("PRAGMA table_info(literature_articles)").fetchall()
        }
        if "content" not in literature_columns:
            con.execute("ALTER TABLE literature_articles ADD COLUMN content TEXT")

        if "abstract" not in literature_columns:
            con.execute("ALTER TABLE literature_articles ADD COLUMN abstract TEXT")
        if "abstract_source" not in literature_columns:
            con.execute("ALTER TABLE literature_articles ADD COLUMN abstract_source TEXT")
        if "abstract_hash" not in literature_columns:
            con.execute("ALTER TABLE literature_articles ADD COLUMN abstract_hash TEXT")
        if "abstract_fetched_at" not in literature_columns:
            con.execute("ALTER TABLE literature_articles ADD COLUMN abstract_fetched_at TEXT")
        provisional_columns_v6 = {
            row["name"]
            for row in con.execute("PRAGMA table_info(provisional_sources)").fetchall()
        }
        if "verification_json" not in provisional_columns_v6:
            con.execute("ALTER TABLE provisional_sources ADD COLUMN verification_json TEXT")
        con.execute("""INSERT INTO schema_migrations VALUES ('evidence', 4, ?)
            ON CONFLICT(component) DO UPDATE SET version=4, applied_at=excluded.applied_at""", (_now(),))

    def upsert_snp(self, snp: SNPIdentity) -> None:
        self.upsert_snps([snp])

    def upsert_snps(self, snps: list[SNPIdentity]) -> None:
        if not snps:
            return
        with self.connect() as con:
            con.executemany("""INSERT INTO snps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_key) DO NOTHING""", [
                (snp.canonical_key, snp.species, snp.assembly, snp.chrom, snp.pos,
                 snp.ref, snp.alt, snp.strand, _now()) for snp in snps
            ])

    def acquire_tool_run(self, spec: ToolRunSpec, *, lease_owner: str | None = None) -> tuple[dict[str, Any], str]:
        """Create a run once, or return cache_hit/in_flight for an existing run."""
        now = _now()
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT * FROM tool_runs WHERE run_key = ?", (spec.run_key,)).fetchone()
            if row is not None:
                state = dict(row)
                if state["status"] == RunStatus.SUCCEEDED.value:
                    return state, "cache_hit"
                if state["status"] == RunStatus.RUNNING.value:
                    return state, "in_flight"
                return state, "retryable"
            run_id = uuid.uuid4().hex
            con.execute("""INSERT INTO tool_runs (
                id, run_key, tool_name, tool_version, input_hash, params_hash, params_json,
                resource_hash, resource_class, status, created_at, lease_owner
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                run_id, spec.run_key, spec.tool_name, spec.tool_version, spec.input_hash,
                spec.params_hash, _json(spec.params), spec.resource_hash, spec.resource_class,
                RunStatus.QUEUED.value, now, lease_owner,
            ))
            state = dict(con.execute("SELECT * FROM tool_runs WHERE id = ?", (run_id,)).fetchone())
            return state, "created"

    def start_tool_run(self, run_id: str, *, lease_owner: str | None = None) -> None:
        lease_expires_at = (datetime.now(timezone.utc) + timedelta(seconds=1800)).isoformat()
        with self.connect() as con:
            con.execute("""UPDATE tool_runs SET status=?, started_at=?, lease_owner=?, lease_expires_at=?
                WHERE id=?""", (RunStatus.RUNNING.value, _now(), lease_owner, lease_expires_at, run_id))

    def claim_resource(self, run_id: str, *, resource_class: str, lease_owner: str, limit: int, lease_seconds: int = 1800) -> bool:
        """Atomically start one queued run if its single-machine resource has capacity."""
        now = datetime.now(timezone.utc)
        expires = (now + timedelta(seconds=lease_seconds)).isoformat()
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute("""UPDATE tool_runs SET status=? WHERE status=? AND lease_expires_at IS NOT NULL AND lease_expires_at < ?""", (
                RunStatus.STALE.value, RunStatus.RUNNING.value, now.isoformat(),
            ))
            row = con.execute("SELECT status FROM tool_runs WHERE id=?", (run_id,)).fetchone()
            if row is None or row["status"] not in {RunStatus.QUEUED.value, RunStatus.FAILED.value, RunStatus.STALE.value}:
                return False
            active = con.execute("SELECT COUNT(*) AS n FROM tool_runs WHERE resource_class=? AND status=?", (resource_class, RunStatus.RUNNING.value)).fetchone()["n"]
            if int(active) >= limit:
                return False
            con.execute("""UPDATE tool_runs SET status=?, started_at=?, lease_owner=?, lease_expires_at=? WHERE id=?""", (
                RunStatus.RUNNING.value, now.isoformat(), lease_owner, expires, run_id,
            ))
            return True

    def put_artifact(self, source: Path, *, media_type: str | None = None) -> str:
        stored = self.artifacts.put_file(source, media_type=media_type)
        artifact_id = stored["sha256"]
        with self.connect() as con:
            con.execute("""INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(sha256) DO NOTHING""", (
                artifact_id, stored["sha256"], stored["path"], stored["bytes"], stored["media_type"], _now(),
            ))
        return artifact_id

    def complete_tool_run(self, run_id: str, *, artifact_id: str | None = None) -> None:
        with self.connect() as con:
            con.execute("""UPDATE tool_runs SET status=?, artifact_id=?, finished_at=?, error=NULL, lease_expires_at=NULL
                WHERE id=?""", (RunStatus.SUCCEEDED.value, artifact_id, _now(), run_id))

    def tool_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as con:
            row = con.execute("SELECT * FROM tool_runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row is not None else None

    def artifact_path(self, artifact_id: str) -> Path | None:
        with self.connect() as con:
            row = con.execute("SELECT path FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        return Path(row["path"]) if row is not None else None

    def artifact_path_for_run(self, run_id: str) -> Path | None:
        with self.connect() as con:
            row = con.execute("""SELECT artifacts.path FROM tool_runs
                JOIN artifacts ON artifacts.id=tool_runs.artifact_id WHERE tool_runs.id=?""", (run_id,)).fetchone()
        return Path(row["path"]) if row is not None else None

    def fail_tool_run(self, run_id: str, error: str, *, unavailable: bool = False) -> None:
        status = RunStatus.UNAVAILABLE.value if unavailable else RunStatus.FAILED.value
        with self.connect() as con:
            con.execute("UPDATE tool_runs SET status=?, error=?, finished_at=?, lease_expires_at=NULL WHERE id=?", (status, error[:4000], _now(), run_id))

    def invalidate_tool_run(self, run_id: str, *, reason: str = "cache_ttl_expired") -> None:
        with self.connect() as con:
            con.execute("UPDATE tool_runs SET status=?, error=?, artifact_id=NULL, lease_expires_at=NULL WHERE id=?", (RunStatus.STALE.value, reason, run_id))

    def invalidate_tool_cache(self, *, tool_name: str | None = None, resource_hash: str | None = None) -> int:

        

        

        invalidatable = (
            RunStatus.QUEUED.value,
            RunStatus.SUCCEEDED.value,
            RunStatus.FAILED.value,
            RunStatus.UNAVAILABLE.value,
        )
        clauses = ["status IN (?, ?, ?, ?)"]
        values: list[Any] = list(invalidatable)
        if tool_name:
            clauses.append("tool_name=?"); values.append(tool_name)
        if resource_hash:
            clauses.append("resource_hash=?"); values.append(resource_hash)
        with self.connect() as con:
            cursor = con.execute(
                f"UPDATE tool_runs SET status=?, error='manual_cache_invalidation', "
                f"artifact_id=NULL, lease_expires_at=NULL WHERE {' AND '.join(clauses)}",
                [RunStatus.STALE.value, *values],
            )
            return cursor.rowcount

    def add_evidence(self, record: EvidenceRecord) -> str:
        return self.add_evidence_batch([record])[0]

    def add_evidence_batch(self, records: list[EvidenceRecord]) -> list[str]:
        if not records:
            return []
        evidence_ids = [uuid.uuid4().hex for _ in records]
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.executemany("""INSERT INTO evidence_records (
                id, snp_key, tool_run_id, artifact_id, evidence_type, subject, predicate,
                object, value_json, unit, direction, confidence, evidence_level,
                scope_json, payload_json, external_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (evidence_id, record.snp_key, record.tool_run_id, record.artifact_id,
                 record.evidence_type, record.subject, record.predicate, record.object,
                 _json(record.value) if record.value is not None else None, record.unit,
                 record.direction, record.confidence, record.evidence_level.value,
                 _json({**record.scope, "evidence_scope": record.evidence_scope.value}),
                 _json(record.payload),
                 (record.payload or {}).get("evidence_id"),
                 _now())
                for evidence_id, record in zip(evidence_ids, records)
            ])

            

            affected_snps: set[str] = set()
            for evidence_id, record in zip(evidence_ids, records):
                payload = record.payload or {}
                if record.evidence_type == "literature":
                    self._upsert_literature_article_locked(
                        con,
                        provider=str(payload.get("provider") or record.scope.get("source") or "literature"),
                        provider_id=payload.get("provider_id") or record.scope.get("source_record_id"),
                        doi=payload.get("doi"),
                        title=literature_display_title(
                            payload.get("title") or record.object,
                            doi=payload.get("doi"), pmid=payload.get("pmid"),
                            pmcid=payload.get("pmcid"), provider_id=payload.get("provider_id"),
                        ),
                        metadata={
                            "title": literature_display_title(
                                payload.get("title") or record.object,
                                doi=payload.get("doi"), pmid=payload.get("pmid"),
                                pmcid=payload.get("pmcid"), provider_id=payload.get("provider_id"),
                            ),
                            "year": str(payload.get("year") or ""),
                            "authors": str(payload.get("authors") or ""),
                            "journal": str(payload.get("journal") or ""),
                            "doi": str(payload.get("doi") or ""),
                            "pmid": str(payload.get("pmid") or ""),
                            "trait": str(payload.get("trait") or record.scope.get("trait") or ""),
                        },
                        content_hash=payload.get("content_hash"),
                    )
                if record.evidence_type in _ANNOTATION_SUMMARY_TYPES:
                    self._upsert_annotation_summary_locked(con, record)
                self._register_active_topic_locked(con, record, payload)
                self._link_evidence_to_claims_locked(con, record, evidence_id)
                affected_snps.add(record.snp_key)
            for snp_key in affected_snps:
                self._recompute_claim_tiers_locked(con, snp_key)
        return evidence_ids


    @staticmethod
    def _upsert_literature_article_locked(
        con: sqlite3.Connection,
        *,
        provider: str,
        provider_id: str | None,
        doi: str | None,
        title: str,
        metadata: dict[str, Any],
        content_hash: str | None = None,
    ) -> str:
        """Upsert one article into literature_articles + FTS (caller holds txn)."""
        article_id = uuid.uuid4().hex
        doi = (doi or "").strip().lower() or None
        provider_id = (provider_id or "").strip() or None
        if provider_id and provider_id.upper().startswith("DOI:") and not doi:
            doi = provider_id[4:].strip().lower() or None
        existing = None
        if provider_id:
            existing = con.execute(
                "SELECT id FROM literature_articles WHERE provider=? AND provider_id=?",
                (provider, provider_id),
            ).fetchone()
        if existing is None and doi:
            existing = con.execute("SELECT id FROM literature_articles WHERE doi=?", (doi,)).fetchone()
        is_new = existing is None
        if existing is not None:
            article_id = str(existing["id"])
            con.execute(
                """UPDATE literature_articles SET title=?, metadata_json=?, content_hash=?, fetched_at=?
                   WHERE id=?""",
                (title, _json(metadata), content_hash, _now(), article_id),
            )
        else:
            con.execute(
                """INSERT INTO literature_articles
                   (id, provider, provider_id, doi, title, metadata_json,
                    content, content_hash, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (article_id, provider, provider_id, doi, title, _json(metadata),
                 None, content_hash, _now()),
            )
        con.execute("DELETE FROM literature_fts WHERE article_id=?", (article_id,))
        searchable = " ".join(
            str(metadata.get(field) or "") for field in ("title", "journal", "authors", "year", "doi", "pmid")
        )
        con.execute(
            "INSERT INTO literature_fts(article_id, title, content) VALUES (?, ?, ?)",
            (article_id, title or "", searchable),
        )
        con.execute(
            """INSERT INTO knowledge_evolution(component, action, detail_json, created_at)
               VALUES ('literature_cache', 'upsert', ?, ?)""",
            (_json({
                "article_id": article_id, "provider": provider, "doi": doi,
                "title": title[:200], "trait": metadata.get("trait") or "",
                "new": is_new,
            }), _now()),
        )
        return article_id

    @staticmethod
    def _upsert_annotation_summary_locked(con: sqlite3.Connection, record: EvidenceRecord) -> None:
        """Aggregate reusable gene/SNP annotations into annotation_summary."""
        scope = record.scope or {}
        species = str(scope.get("species") or "").strip().lower()
        assembly = str(scope.get("assembly") or "").strip()
        gene_id = None
        if record.evidence_scope is not None and record.evidence_scope.value in (
            "gene_level", "pathway_level", "database_level",
        ):
            gene_id = record.subject
        key = hashlib.sha256("|".join([
            species, assembly, gene_id or "", record.snp_key or "",
            record.evidence_type, record.object or "",
        ]).encode("utf-8")).hexdigest()
        now = _now()
        con.execute(
            """INSERT INTO annotation_summary (
                key, species, assembly, gene_id, snp_key, annotation_type, subject,
                predicate, object, source_name, evidence_level, payload_json,
                first_seen_at, last_seen_at, hit_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(key) DO UPDATE SET
                last_seen_at=excluded.last_seen_at,
                hit_count=hit_count + 1,
                source_name=excluded.source_name,
                evidence_level=excluded.evidence_level,
                payload_json=excluded.payload_json""",
            (
                key, species, assembly, gene_id, record.snp_key, record.evidence_type,
                record.subject, record.predicate, record.object,
                scope.get("source") or "", record.evidence_level.value,
                _json({"evidence_id": (record.payload or {}).get("evidence_id"),
                       "statement": record.object[:500]}),
                now, now,
            ),
        )
        hit = con.execute("SELECT hit_count FROM annotation_summary WHERE key=?", (key,)).fetchone()
        con.execute(
            """INSERT INTO knowledge_evolution(component, action, detail_json, created_at)
               VALUES ('annotation_summary', 'upsert', ?, ?)""",
            (_json({
                "key": key, "species": species, "gene_id": gene_id,
                "snp_key": record.snp_key, "annotation_type": record.evidence_type,
                "hit_count": int(hit["hit_count"]) if hit else 1,
            }), now),
        )

    @staticmethod
    def _register_active_topic_locked(
        con: sqlite3.Connection, record: EvidenceRecord, payload: dict[str, Any],
    ) -> None:
        """Track an active gene/trait topic so the background refresher can re-query it."""
        scope = record.scope or {}
        species = str(scope.get("species") or "").strip().lower()
        if not species or record.evidence_type not in {"literature", "gene_link", "gene_context"}:
            return
        trait = str(payload.get("trait") or scope.get("trait") or "").strip()
        gene = str(record.subject or "").strip()
        if not gene:
            return
        snp = str(record.snp_key or "").strip()
        topic_key = hashlib.sha256("|".join([species, trait, gene, snp]).encode("utf-8")).hexdigest()
        now = _now()
        con.execute(
            """INSERT INTO active_topics (topic_key, kind, species, trait, gene, snp, first_seen_at, hit_count)
               VALUES (?, 'gene', ?, ?, ?, ?, ?, 1)
               ON CONFLICT(topic_key) DO UPDATE SET hit_count=hit_count + 1""",
            (topic_key, species, trait, gene, snp or None, now),
        )

    @staticmethod
    def _link_evidence_to_claims_locked(
        con: sqlite3.Connection, record: EvidenceRecord, evidence_id: str,
    ) -> None:
        """Attach newly stored evidence to existing claims for the same SNP."""
        claims = con.execute("SELECT id FROM claims WHERE snp_key=?", (record.snp_key,)).fetchall()
        if not claims:
            return
        relation = record.predicate if record.predicate in ("supports", "contradicts") else "supports"
        for claim in claims:
            con.execute(
                "INSERT OR IGNORE INTO claim_evidence VALUES (?, ?, ?)",
                (claim["id"], evidence_id, relation),
            )

    @staticmethod
    def _recompute_claim_tiers_locked(con: sqlite3.Connection, snp_key: str) -> None:
        """Event-driven recomputation: re-derive each claim tier from all linked evidence."""
        claims = con.execute("SELECT id FROM claims WHERE snp_key=?", (snp_key,)).fetchall()
        if not claims:
            return
        for claim in claims:
            rows = con.execute(
                """SELECT claim_evidence.relation, evidence_records.evidence_level
                   FROM claim_evidence
                   JOIN evidence_records ON evidence_records.id = claim_evidence.evidence_id
                   WHERE claim_evidence.claim_id = ?""",
                (claim["id"],),
            ).fetchall()
            support_w = sum(
                _EVIDENCE_LEVEL_WEIGHT.get(str(row["evidence_level"]), 0.1)
                for row in rows if row["relation"] == "supports"
            )
            contradict_w = sum(
                _EVIDENCE_LEVEL_WEIGHT.get(str(row["evidence_level"]), 0.1)
                for row in rows if row["relation"] == "contradicts"
            )
            support_levels = [str(row["evidence_level"]) for row in rows if row["relation"] == "supports"]
            base = max(support_levels, key=lambda lv: _EVIDENCE_LEVEL_WEIGHT.get(lv, 0.1)) if support_levels else "E0"
            if support_w <= 0:
                tier = "E0"
            elif contradict_w > 0 and contradict_w >= 0.75 * support_w:
                tier = "E2" if _EVIDENCE_LEVEL_RANK.get(base, 0) > _EVIDENCE_LEVEL_RANK["E2"] else base
            elif contradict_w > 0 and contradict_w >= 0.4 * support_w:
                tier = "E3" if _EVIDENCE_LEVEL_RANK.get(base, 0) > _EVIDENCE_LEVEL_RANK["E3"] else base
            else:
                tier = base
            existing = con.execute("SELECT payload_json FROM claims WHERE id=?", (claim["id"],)).fetchone()
            payload: dict[str, Any] = {}
            try:
                payload = json.loads(existing["payload_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = {}
            recompute = payload.setdefault("recompute", {})
            recompute.update({
                "support_weight": round(support_w, 4),
                "contradiction_weight": round(contradict_w, 4),
                "supporting_count": sum(1 for row in rows if row["relation"] == "supports"),
                "contradicting_count": sum(1 for row in rows if row["relation"] == "contradicts"),
                "computed_tier": tier,
                "rule_version": "v1",
                "recomputed_at": _now(),
            })
            con.execute(
                "UPDATE claims SET evidence_tier=?, payload_json=? WHERE id=?",
                (tier, _json(payload), claim["id"]),
            )
            con.execute(
                """INSERT INTO knowledge_evolution(component, action, detail_json, created_at)
                   VALUES ('claims', 'recompute', ?, ?)""",
                (_json({"claim_id": claim["id"], "snp_key": snp_key, "tier": tier}), _now()),
            )

    def evidence_for_snp(self, snp_key: str) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute("""SELECT evidence_records.*, tool_runs.tool_name, tool_runs.tool_version
                FROM evidence_records LEFT JOIN tool_runs ON tool_runs.id=evidence_records.tool_run_id
                WHERE snp_key=? ORDER BY evidence_records.created_at""", (snp_key,)).fetchall()
        return [dict(row) for row in rows]

    def search_snps(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Resolve canonical/coordinate/gene evidence without rerunning tools."""
        token = query.strip()
        like = f"%{token}%"
        with self.connect() as con:
            rows = con.execute("""SELECT DISTINCT snps.* FROM snps
                LEFT JOIN evidence_records ON evidence_records.snp_key=snps.canonical_key
                WHERE snps.canonical_key LIKE ? OR evidence_records.object LIKE ?
                   OR evidence_records.payload_json LIKE ?
                ORDER BY snps.created_at DESC LIMIT ?""", (like, like, like, limit)).fetchall()
        return [dict(row) for row in rows]

    def evidence_for_tool_run(self, tool_run_id: str) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute("SELECT * FROM evidence_records WHERE tool_run_id=? ORDER BY created_at", (tool_run_id,)).fetchall()
        return [dict(row) for row in rows]

    def delete_evidence_for_tool_run(self, tool_run_id: str) -> None:
        """Remove incomplete normalized rows before a retry; artifacts remain immutable."""
        with self.connect() as con:
            affected = [
                row["snp_key"] for row in con.execute(
                    "SELECT DISTINCT snp_key FROM evidence_records WHERE tool_run_id=?", (tool_run_id,)
                ).fetchall()
            ]
            con.execute("DELETE FROM claim_evidence WHERE evidence_id IN (SELECT id FROM evidence_records WHERE tool_run_id=?)", (tool_run_id,))
            con.execute("DELETE FROM evidence_relations WHERE source_evidence_id IN (SELECT id FROM evidence_records WHERE tool_run_id=?) OR target_evidence_id IN (SELECT id FROM evidence_records WHERE tool_run_id=?)", (tool_run_id, tool_run_id))
            con.execute("DELETE FROM evidence_records WHERE tool_run_id=?", (tool_run_id,))
            for snp_key in affected:
                self._recompute_claim_tiers_locked(con, snp_key)

    def add_evidence_relation(self, source_evidence_id: str, target_evidence_id: str, relation: str, reason: str | None = None) -> None:
        if source_evidence_id == target_evidence_id:
            return
        with self.connect() as con:
            con.execute("""INSERT OR IGNORE INTO evidence_relations VALUES (?, ?, ?, ?, ?)""", (
                source_evidence_id, target_evidence_id, relation, reason, _now(),
            ))

    def evidence_relations(self, snp_key: str) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute("""SELECT evidence_relations.* FROM evidence_relations
                JOIN evidence_records ON evidence_records.id=evidence_relations.source_evidence_id
                WHERE evidence_records.snp_key=? ORDER BY evidence_relations.created_at""", (snp_key,)).fetchall()
        return [dict(row) for row in rows]

    def claims_for_snp(self, snp_key: str) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute("SELECT * FROM claims WHERE snp_key=? ORDER BY created_at", (snp_key,)).fetchall()
        return [dict(row) for row in rows]

    def claim_evidence(self, claim_id: str) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute("""SELECT evidence_records.*,claim_evidence.relation,tool_runs.tool_name,tool_runs.tool_version
                FROM claim_evidence JOIN evidence_records ON evidence_records.id=claim_evidence.evidence_id
                LEFT JOIN tool_runs ON tool_runs.id=evidence_records.tool_run_id
                WHERE claim_evidence.claim_id=? ORDER BY evidence_records.created_at""", (claim_id,)).fetchall()
        return [dict(row) for row in rows]

    def add_claim(self, claim: Claim) -> str:
        """Persist one claim deterministically (same claim → same id, updates in place)."""
        claim_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            "|".join([claim.snp_key, claim.claim_type, claim.statement.strip()[:500]]),
        ))
        with self.connect() as con:
            supporting_rows: list[dict[str, Any]] = []
            if claim.supporting_evidence_ids:
                placeholders = ",".join("?" for _ in claim.supporting_evidence_ids)
                rows = con.execute(f"""SELECT evidence_records.*,tool_runs.tool_name,tool_runs.tool_version
                    FROM evidence_records LEFT JOIN tool_runs ON tool_runs.id=evidence_records.tool_run_id
                    WHERE evidence_records.id IN ({placeholders})""", claim.supporting_evidence_ids).fetchall()
                supporting_rows = [dict(row) for row in rows]
            from phenofuncrank_repro.interpretation.provenance import validate_claim_support_rows
            errors = [issue for issue in validate_claim_support_rows(claim, supporting_rows) if issue["severity"] == "error"]
            if errors:
                raise ValueError(f"claim evidence policy rejected the claim: {errors}")
            existing = con.execute("SELECT id FROM claims WHERE id=?", (claim_id,)).fetchone()
            if existing is not None:
                con.execute(
                    "UPDATE claims SET evidence_tier=?, status=?, payload_json=? WHERE id=?",
                    (claim.evidence_tier.value, claim.status, _json(claim.payload), claim_id),
                )
            else:
                con.execute("INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
                    claim_id, claim.snp_key, claim.claim_type, claim.statement,
                    claim.evidence_tier.value, claim.status, _json(claim.payload), _now(),
                ))
            con.execute("DELETE FROM claim_evidence WHERE claim_id=?", (claim_id,))
            for evidence_id in claim.supporting_evidence_ids:
                con.execute("INSERT OR IGNORE INTO claim_evidence VALUES (?, ?, 'supports')", (claim_id, evidence_id))
            for evidence_id in claim.contradicting_evidence_ids:
                con.execute("INSERT OR IGNORE INTO claim_evidence VALUES (?, ?, 'contradicts')", (claim_id, evidence_id))
            self._recompute_claim_tiers_locked(con, claim.snp_key)
        return claim_id

    def upsert_literature_article(self, *, provider: str, provider_id: str | None, doi: str | None, title: str | None, metadata: dict[str, Any], content_hash: str | None = None) -> str:
        """Cache provider metadata independently from any individual SNP request."""
        with self.connect() as con:
            return self._upsert_literature_article_locked(
                con, provider=provider, provider_id=provider_id, doi=doi,
                title=title or "", metadata=metadata, content_hash=content_hash,
            )

    def search_literature(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute("""SELECT literature_articles.* FROM literature_fts
                JOIN literature_articles ON literature_articles.id=literature_fts.article_id
                WHERE literature_fts MATCH ? ORDER BY bm25(literature_fts) LIMIT ?""", (query, limit)).fetchall()
        return [dict(row) for row in rows]


    def literature_article_by_doi(self, doi: str) -> dict[str, Any] | None:
        doi = (doi or "").strip().lower()
        if not doi:
            return None
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM literature_articles WHERE doi=?", (doi,),
            ).fetchone()
        return dict(row) if row else None

    def literature_article_by_provider_id(self, provider_id: str) -> dict[str, Any] | None:
        """Look up one article by provider id (e.g. ``PMID:12345678``)."""
        provider_id = (provider_id or "").strip()
        if not provider_id:
            return None
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM literature_articles WHERE provider_id=?", (provider_id,),
            ).fetchone()
        return dict(row) if row else None

    def literature_article_abstract(
        self, *, doi: str = "", pmid: str = "",
    ) -> tuple[str, str] | None:
        """Return (abstract, source) cached for a DOI/PMID, or None."""
        with self.connect() as con:
            row = None
            if doi:
                row = con.execute(
                    "SELECT abstract, abstract_source FROM literature_articles WHERE doi=?",
                    (doi.strip().lower(),),
                ).fetchone()
            if row is None and pmid:
                row = con.execute(
                    "SELECT abstract, abstract_source FROM literature_articles "
                    "WHERE provider_id=? OR provider_id=?",
                    (f"PMID:{pmid}", pmid),
                ).fetchone()
            if row is None:
                return None
            abstract = str(row["abstract"] or "")
            source = str(row["abstract_source"] or "")
            return (abstract, source) if abstract else None

    def set_literature_abstract(
        self, article_id: str, abstract: str, *, source: str = "",
    ) -> None:
        """Persist an enriched abstract for one article (cache for re-use)."""
        import hashlib

        abstract = abstract or ""
        abstract_hash = (
            hashlib.sha256(abstract.encode("utf-8")).hexdigest()[:32] if abstract else ""
        )
        with self.connect() as con:
            con.execute(
                "UPDATE literature_articles SET abstract=?, abstract_source=?, "
                "abstract_hash=?, abstract_fetched_at=? WHERE id=?",
                (abstract, source, abstract_hash, _now(), article_id),
            )

    def set_literature_fulltext(self, article_id: str, content: str) -> None:
        """Persist full text + hash and refresh the article FTS row."""
        import hashlib

        content = content or ""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]
        with self.connect() as con:
            con.execute(
                "UPDATE literature_articles SET content=?, content_hash=? WHERE id=?",
                (content, content_hash, article_id),
            )
            con.execute("DELETE FROM literature_fts WHERE article_id=?", (article_id,))
            row = con.execute("SELECT * FROM literature_articles WHERE id=?", (article_id,)).fetchone()
            metadata = {}
            if row is not None:
                try:
                    metadata = json.loads(row["metadata_json"] or "{}")
                except (TypeError, ValueError, json.JSONDecodeError):
                    metadata = {}
            searchable = " ".join(
                str(metadata.get(field) or "") for field in ("title", "journal", "authors", "year", "doi", "pmid")
            )
            con.execute(
                "INSERT INTO literature_fts(article_id, title, content) VALUES (?, ?, ?)",
                (article_id, str(metadata.get("title") or ""), searchable + "\n" + content),
            )
            con.execute(
                """INSERT INTO knowledge_evolution(component, action, detail_json, created_at)
                   VALUES ('literature_cache', 'fulltext_update', ?, ?)""",
                (_json({"article_id": article_id, "content_chars": len(content),
                        "content_hash": content_hash}), _now()),
            )

    def update_literature_metadata(
        self, article_id: str, updates: dict[str, Any],
    ) -> None:
        "Merge retrieval metadata without replacing cached text."
        if not updates:
            return
        with self.connect() as con:
            row = con.execute(
                "SELECT metadata_json FROM literature_articles WHERE id=?",
                (article_id,),
            ).fetchone()
            if row is None:
                return
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
            metadata.update(updates)
            con.execute(
                "UPDATE literature_articles SET metadata_json=? WHERE id=?",
                (_json(metadata), article_id),
            )

    def replace_literature_chunks(
        self,
        article_id: str,
        chunks: list[Any],
        embeddings: list[bytes] | None = None,
    ) -> dict[str, Any]:
        """Replace all chunks for one article (deterministic ids, FTS sync)."""
        import struct

        stats = {"inserted": 0, "truncated": 0, "embedding_dim": 0}
        with self.connect() as con:
            existing_ids = [
                row["id"] for row in con.execute(
                    "SELECT id FROM literature_chunks WHERE article_id=?", (article_id,),
                ).fetchall()
            ]
            if existing_ids:
                placeholders = ",".join("?" for _ in existing_ids)
                con.execute(f"DELETE FROM literature_chunk_fts WHERE chunk_id IN ({placeholders})", existing_ids)
                con.execute("DELETE FROM literature_chunks WHERE article_id=?", (article_id,))
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{article_id[:16]}_{idx}"
                emb = None
                if embeddings and idx < len(embeddings):
                    emb = embeddings[idx]
                    if emb:
                        stats["embedding_dim"] = len(emb) // 4
                con.execute(
                    """INSERT INTO literature_chunks
                       (id, article_id, chunk_index, section, start_char, end_char,
                        content, content_hash, embedding, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (chunk_id, article_id, idx, chunk.section, chunk.start_char,
                     chunk.end_char, chunk.content, chunk.content_hash, emb, _now()),
                )
                con.execute(
                    "INSERT INTO literature_chunk_fts(chunk_id, article_id, section, content) VALUES (?, ?, ?, ?)",
                    (chunk_id, article_id, chunk.section, chunk.content),
                )
                stats["inserted"] += 1
            con.execute(
                """INSERT INTO knowledge_evolution(component, action, detail_json, created_at)
                   VALUES ('literature_cache', 'chunks_update', ?, ?)""",
                (_json({"article_id": article_id, **stats}), _now()),
            )
        return stats

    def get_literature_chunks(
        self, chunk_ids: list[str], *, include_embedding: bool = False,
    ) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self.connect() as con:
            rows = con.execute(
                f"SELECT * FROM literature_chunks WHERE id IN ({placeholders})", chunk_ids,
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            if not include_embedding:
                item.pop("embedding", None)
            out.append(item)
        return out

    def search_literature_chunks(self, query: str, *, limit: int = 30) -> list[dict[str, Any]]:
        """BM25 chunk-level search over literature_chunk_fts."""
        query = (query or "").strip()
        if not query:
            return []
        with self.connect() as con:
            rows = con.execute(
                """SELECT literature_chunks.*, bm25(literature_chunk_fts) AS bm25_rank
                   FROM literature_chunk_fts
                   JOIN literature_chunks ON literature_chunks.id=literature_chunk_fts.chunk_id
                   WHERE literature_chunk_fts MATCH ?
                   ORDER BY bm25(literature_chunk_fts) LIMIT ?""",
                (query, limit),
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item.pop("embedding", None)
            out.append(item)
        return out


    def write_claim_chain(
        self,
        *,
        snp_key: str,
        gene: str,
        claim_text: str,
        claim_type: str,
        causal_claim_allowed: bool,
        evidence_level: str,
        critic_verdict: str,
        required_evidence_types: list[str],
        covered_evidence_types: list[str],
        evidence_id: str,
        chunk_id: str,
        article_id: str,
        trait: str = "",
    ) -> str:
        """Persist Variant→Claim→Evidence chunk→Paper provenance chain.

        Writes the claims row, claim_evidence mapping and typed graph
        nodes/edges.  All edges carry the evidence id (no evidence-free edges).
        """
        from phenofuncrank_repro.interpretation.evidence_models import EvidenceLevel

        valid_levels = {e.value for e in EvidenceLevel}
        level = (
            EvidenceLevel(str(evidence_level).upper())
            if str(evidence_level).upper() in valid_levels
            else EvidenceLevel.E4
        )
        claim = Claim(
            snp_key=snp_key,
            claim_type=claim_type,
            statement=claim_text,
            evidence_tier=level,
            status=str(critic_verdict or "pending"),
            supporting_evidence_ids=(evidence_id,),
            payload={
                "gene": gene,
                "trait": trait,
                "causal_claim_allowed": causal_claim_allowed,
                "required_evidence_types": required_evidence_types,
                "covered_evidence_types": covered_evidence_types,
                "provenance_chain": ["variant", "claim", "evidence_record", "chunk", "paper"],
            },
        )
        claim_id = self.add_claim(claim)

        nodes = [
            (snp_key, "variant", {"label": snp_key}),
            (f"gene:{gene}", "gene", {"label": gene}),
            (f"claim:{claim_id}", "claim", {"label": claim_text[:120]}),
            (f"evidence:{evidence_id}", "evidence_record", {"label": evidence_id}),
            (f"chunk:{chunk_id}", "literature_chunk", {"label": chunk_id}),
            (f"article:{article_id}", "literature_article", {"label": article_id}),
        ]
        if trait:
            nodes.append((f"trait:{trait}", "trait", {"label": trait}))
        edges = [
            (snp_key, "supports", f"claim:{claim_id}"),
            (f"claim:{claim_id}", "supported_by", f"evidence:{evidence_id}"),
            (f"evidence:{evidence_id}", "extracted_from", f"chunk:{chunk_id}"),
            (f"chunk:{chunk_id}", "from", f"article:{article_id}"),
            (snp_key, "in_gene", f"gene:{gene}"),
        ]
        if trait:
            edges.append((f"gene:{gene}", "associated_with", f"trait:{trait}"))

        with self.connect() as con:
            for node_id, node_type, attrs in nodes:
                con.execute(
                    """INSERT INTO graph_nodes(node_id, node_type, attributes_json, first_seen_at, last_seen_at, hit_count)
                       VALUES (?, ?, ?, ?, ?, 1)
                       ON CONFLICT(node_id) DO UPDATE SET
                           last_seen_at=excluded.last_seen_at,
                           hit_count=graph_nodes.hit_count + 1""",
                    (node_id, node_type, _json(attrs), _now(), _now()),
                )
            for source, predicate, target in edges:
                edge_id = str(uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    "|".join([source, predicate, target, evidence_id]),
                ))
                con.execute(
                    """INSERT INTO graph_edges(
                           edge_id, source, predicate, target, properties_json,
                           evidence_ids_json, confidence, directness,
                           first_seen_at, last_seen_at, hit_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                       ON CONFLICT(edge_id) DO UPDATE SET
                           last_seen_at=excluded.last_seen_at,
                           evidence_ids_json=excluded.evidence_ids_json,
                           hit_count=graph_edges.hit_count + 1""",
                    (edge_id, source, predicate, target,
                     _json({"chain": "variant->claim->chunk->paper"}),
                     _json([evidence_id]), 0.9, "direct", _now(), _now()),
                )
            con.execute(
                """INSERT INTO knowledge_evolution(component, action, detail_json, created_at)
                   VALUES ('evidence_graph', 'claim_chain_written', ?, ?)""",
                (_json({
                    "claim_id": claim_id, "snp_key": snp_key, "gene": gene,
                    "evidence_id": evidence_id, "chunk_id": chunk_id,
                    "article_id": article_id, "trait": trait,
                }), _now()),
            )
        return claim_id

    def reuse_claims_for_gene(
        self,
        *,
        gene: str,
        required_evidence_types: list[str],
        trait: str = "",
    ) -> dict[str, list[str]]:
        """Claim-level reuse: claims for *gene* supported by required types."""
        if not gene or not required_evidence_types:
            return {"reuse_claim_ids": [], "reuse_evidence_ids": []}
        placeholders = ",".join("?" for _ in required_evidence_types)
        with self.connect() as con:
            rows = con.execute(
                f"""SELECT DISTINCT c.id AS claim_id, ce.evidence_id,
                           er.evidence_type, er.scope_json
                    FROM claims c
                    JOIN claim_evidence ce ON ce.claim_id = c.id
                    JOIN evidence_records er ON er.id = ce.evidence_id
                    WHERE json_extract(c.payload_json, '$.gene') = ?
                      AND er.evidence_type IN ({placeholders})""",
                [gene, *required_evidence_types],
            ).fetchall()
        claim_ids: set[str] = set()
        evidence_ids: set[str] = set()
        for row in rows:
            if trait:
                try:
                    scope = json.loads(row["scope_json"] or "{}")
                except (TypeError, ValueError, json.JSONDecodeError):
                    scope = {}
                row_trait = str(scope.get("trait") or "")
                if row_trait and row_trait != trait:
                    continue
            claim_ids.add(str(row["claim_id"]))
            evidence_ids.add(str(row["evidence_id"]))
        return {
            "reuse_claim_ids": sorted(claim_ids),
            "reuse_evidence_ids": sorted(evidence_ids),
        }

    def set_source_verification(
        self, source_id: str, verification: list[dict[str, Any]],
    ) -> None:
        """Attach ①②③ verification records to a provisional source (SOP §5.6)."""
        with self.connect() as con:
            con.execute(
                "UPDATE provisional_sources SET verification_json=? WHERE source_id=?",
                (_json(verification), source_id),
            )


    def annotations_for_gene(
        self,
        *,
        species: str,
        gene_id: str,
        assembly: str | None = None,
        annotation_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Reusable gene-level annotations accumulated from previous runs."""
        clauses = ["species=?", "gene_id=?"]
        values: list[Any] = [str(species or "").strip().lower(), gene_id]
        if assembly:
            clauses.append("assembly=?")
            values.append(assembly)
        if annotation_types:
            placeholders = ",".join("?" for _ in annotation_types)
            clauses.append(f"annotation_type IN ({placeholders})")
            values.extend(sorted(annotation_types))
        with self.connect() as con:
            rows = con.execute(
                f"""SELECT * FROM annotation_summary WHERE {' AND '.join(clauses)}
                    ORDER BY hit_count DESC, last_seen_at DESC""",
                values,
            ).fetchall()
        return [dict(row) for row in rows]

    def annotations_for_snp(
        self, snp_key: str, annotation_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["snp_key=?"]
        values: list[Any] = [snp_key]
        if annotation_types:
            placeholders = ",".join("?" for _ in annotation_types)
            clauses.append(f"annotation_type IN ({placeholders})")
            values.extend(sorted(annotation_types))
        with self.connect() as con:
            rows = con.execute(
                f"""SELECT * FROM annotation_summary WHERE {' AND '.join(clauses)}
                    ORDER BY hit_count DESC, last_seen_at DESC""",
                values,
            ).fetchall()
        return [dict(row) for row in rows]


    def resolve_external_evidence_ids(self, external_ids: list[str]) -> dict[str, str]:
        """Map retrieval-item evidence IDs (lit_xxx / gene_xxx) to stored row UUIDs."""
        unique = list(dict.fromkeys(eid for eid in (external_ids or []) if eid))
        if not unique:
            return {}
        placeholders = ",".join("?" for _ in unique)
        with self.connect() as con:
            rows = con.execute(
                f"SELECT id, external_id FROM evidence_records WHERE external_id IN ({placeholders})",
                unique,
            ).fetchall()
        return {str(row["external_id"]): str(row["id"]) for row in rows}

    def recompute_claim_tiers_for_snp(self, snp_key: str) -> None:
        with self.connect() as con:
            self._recompute_claim_tiers_locked(con, snp_key)

    def recompute_all_claim_tiers(self) -> int:
        with self.connect() as con:
            rows = con.execute("SELECT DISTINCT snp_key FROM claims").fetchall()
            for row in rows:
                self._recompute_claim_tiers_locked(con, str(row["snp_key"]))
            return len(rows)


    def active_topics(
        self, *, due_before: str | None = None, limit: int | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM active_topics"
        values: list[Any] = []
        if due_before is not None:
            sql += " WHERE last_refreshed_at IS NULL OR last_refreshed_at < ?"
            values.append(due_before)
        sql += " ORDER BY hit_count DESC, first_seen_at ASC"
        if limit is not None:
            sql += " LIMIT ?"
            values.append(limit)
        with self.connect() as con:
            rows = con.execute(sql, values).fetchall()
        return [dict(row) for row in rows]

    def mark_topic_refreshed(
        self, topic_key: str, *, error: str | None = None, article_count: int = 0,
    ) -> None:
        with self.connect() as con:
            if error:
                con.execute(
                    """UPDATE active_topics SET last_refreshed_at=?, last_error=?, refresh_count=refresh_count + 1
                       WHERE topic_key=?""",
                    (_now(), error[:500], topic_key),
                )
            else:
                con.execute(
                    """UPDATE active_topics SET last_refreshed_at=?, last_error=NULL, refresh_count=refresh_count + 1
                       WHERE topic_key=?""",
                    (_now(), topic_key),
                )


    def log_evolution(self, component: str, action: str, detail: dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute(
                """INSERT INTO knowledge_evolution(component, action, detail_json, created_at)
                   VALUES (?, ?, ?, ?)""",
                (component, action, _json(detail), _now()),
            )

    def evolution_log(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT * FROM knowledge_evolution ORDER BY id DESC LIMIT ?", (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def knowledge_stats(self) -> dict[str, Any]:
        with self.connect() as con:
            articles = con.execute("SELECT COUNT(*) AS n FROM literature_articles").fetchone()["n"]
            topics = con.execute("SELECT COUNT(*) AS n FROM active_topics").fetchone()["n"]
            claims = con.execute("SELECT COUNT(*) AS n FROM claims").fetchone()["n"]
            annotations = con.execute("SELECT COUNT(*) AS n FROM annotation_summary").fetchone()["n"]
            evolution = con.execute("SELECT COUNT(*) AS n FROM knowledge_evolution").fetchone()["n"]
            graph_nodes, graph_edges = self._graph_counts()
            provisional_sources = con.execute(
                "SELECT COUNT(*) AS n FROM provisional_sources"
            ).fetchone()["n"]
            provisional_evidence = con.execute(
                "SELECT COUNT(*) AS n FROM provisional_evidence"
            ).fetchone()["n"]
            last_refresh = con.execute(
                "SELECT MAX(last_refreshed_at) AS ts FROM active_topics WHERE last_refreshed_at IS NOT NULL",
            ).fetchone()["ts"]
        return {
            "literature_articles": int(articles),
            "active_topics": int(topics),
            "claims": int(claims),
            "annotation_summary_rows": int(annotations),
            "evolution_events": int(evolution),
            "last_topic_refresh": last_refresh,
            "graph_nodes": int(graph_nodes),
            "graph_edges": int(graph_edges),
            "provisional_sources": int(provisional_sources),
            "provisional_evidence": int(provisional_evidence),
        }

    def _graph_counts(self) -> tuple[int, int]:
        """Count graph nodes/edges across species-scoped graph databases."""
        species_files = sorted(self.graph_root.glob("*.sqlite3")) if self.graph_root.is_dir() else []
        if species_files:
            total_nodes = 0
            total_edges = 0
            for path in species_files:
                try:
                    con = sqlite3.connect(str(path))
                    total_nodes += int(con.execute("SELECT COUNT(*) AS n FROM graph_nodes").fetchone()[0])
                    total_edges += int(con.execute("SELECT COUNT(*) AS n FROM graph_edges").fetchone()[0])
                    con.close()
                except Exception:
                    pass
            return total_nodes, total_edges

        with self.connect() as con:
            nodes = int(con.execute("SELECT COUNT(*) AS n FROM graph_nodes").fetchone()["n"])
            edges = int(con.execute("SELECT COUNT(*) AS n FROM graph_edges").fetchone()["n"])
        return nodes, edges

    def memory_documents(self) -> list[dict[str, str]]:
        """High-value durable text for dense retrieval; raw bulk numeric evidence is excluded."""
        documents: list[dict[str, str]] = []
        with self.connect() as con:
            articles = con.execute("SELECT id, title, metadata_json FROM literature_articles").fetchall()
            claims = con.execute("SELECT id, snp_key, statement, evidence_tier, status FROM claims").fetchall()
            annotations = con.execute("""SELECT id, snp_key, object, payload_json FROM evidence_records
                WHERE evidence_type IN ('functional_annotation', 'gene_link', 'literature')""").fetchall()
            annotation_rows = con.execute(
                """SELECT key, gene_id, snp_key, annotation_type, object
                   FROM annotation_summary WHERE hit_count >= 2
                   ORDER BY hit_count DESC LIMIT 2000""",
            ).fetchall()
        for row in articles:
            metadata = json.loads(row["metadata_json"])
            text = " ".join(str(metadata.get(key) or "") for key in ("title", "authors", "journal", "year", "doi", "pmid"))
            documents.append({"id": f"article:{row['id']}", "kind": "literature_article", "text": text})
        for row in claims:
            documents.append({"id": f"claim:{row['id']}", "kind": "claim", "text": f"{row['snp_key']} {row['evidence_tier']} {row['status']} {row['statement']}"})
        for row in annotations:
            payload = json.loads(row["payload_json"])
            documents.append({"id": f"evidence:{row['id']}", "kind": "functional_evidence", "text": f"{row['snp_key']} {row['object']} {json.dumps(payload, ensure_ascii=False)}"})
        for row in annotation_rows:
            anchor = row["gene_id"] or row["snp_key"] or ""
            documents.append({
                "id": f"annotation:{row['key']}",
                "kind": "annotation_summary",
                "text": f"{anchor} {row['annotation_type']} {row['object']}",
            })

        
        with self.connect() as con:
            numeric = con.execute("""SELECT snp_key, evidence_type, value_json, direction, scope_json
                FROM evidence_records WHERE evidence_type IN ('model_effect', 'functional_score')""").fetchall()
        summaries: dict[str, dict[str, object]] = {}
        for row in numeric:
            summary = summaries.setdefault(row["snp_key"], {"effect": None, "direction": None, "cadd": None, "scope": None})
            try:
                value = float(json.loads(row["value_json"])) if row["value_json"] is not None else None
            except (TypeError, ValueError, json.JSONDecodeError):
                value = None
            if row["evidence_type"] == "model_effect" and value is not None:
                prior = summary["effect"]
                if prior is None or abs(value) > abs(float(prior)):
                    summary["effect"] = value
                    summary["direction"] = row["direction"]
                    summary["scope"] = row["scope_json"]
            elif row["evidence_type"] == "functional_score" and value is not None:
                prior = summary["cadd"]
                if prior is None or value > float(prior):
                    summary["cadd"] = value
        ranked = sorted(summaries.items(), key=lambda item: (abs(float(item[1]["effect"] or 0)), float(item[1]["cadd"] or 0)), reverse=True)
        max_snp_docs = int(os.environ.get("RICE_PH_AGENT_VECTOR_MAX_SNP_DOCS", "5000"))
        for snp_key, summary in ranked[:max_snp_docs]:
            documents.append({
                "id": f"snp:{snp_key}",
                "kind": "snp_evidence_summary",
                "text": f"{snp_key} DPC predicted effect {summary['effect']} direction {summary['direction']} CADD PHRED {summary['cadd']} scope {summary['scope']}",
            })
        return [document for document in documents if document["text"].strip()]


    def bulk_upsert_population_layers(self, rows: list[dict[str, Any]]) -> int:
        """Insert/update full-population AF rows (cultivar memmap / wild VCF)."""
        if not rows:
            return 0
        now = _now()
        with self.connect() as con:
            con.executemany(
                """INSERT INTO population_layers (
                    species, chrom, pos, ref, alt, snp_key, population, source,
                    af, ac, an, callable_rate, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snp_key, population, source) DO UPDATE SET
                    af=excluded.af, ac=excluded.ac, an=excluded.an,
                    callable_rate=excluded.callable_rate, updated_at=excluded.updated_at""",
                [
                    (
                        str(r["species"]), str(r["chrom"]), int(r["pos"]),
                        str(r["ref"]), str(r["alt"]), str(r["snp_key"]),
                        str(r["population"]), str(r["source"]),
                        r.get("af"), r.get("ac"), r.get("an"),
                        r.get("callable_rate"), r.get("updated_at") or now,
                    )
                    for r in rows
                ],
            )
        return len(rows)

    def population_layer_for_snp(self, snp_key: str) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute(
                """SELECT * FROM population_layers
                   WHERE snp_key=? ORDER BY population, source""",
                (snp_key,),
            ).fetchall()
        return [dict(r) for r in rows]
