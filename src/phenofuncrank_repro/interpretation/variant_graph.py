"Build a variant-centered evidence graph."

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from phenofuncrank_repro.interpretation.evidence_models import SNPIdentity
from phenofuncrank_repro.interpretation.evidence_store import EvidenceStore
from phenofuncrank_repro.interpretation.settings import settings


NODE_TYPES = {
    "variant", "gene", "protein", "population", "pathway", "trait", "model", "evidence",
}

EDGE_PREDICATES = {
    "LD_WITH",
    "LOCATED_IN",
    "REGULATES",
    "OBSERVED_IN",
    "DIFFERENT_FROM",
    "SCORED_BY",
    "PARTICIPATES_IN",
    "RELATED_TO",
    "ASSOCIATED_WITH",
    "AFFECTS",
    "SUPPORTED_BY",
}

_GENE_ANCHOR_TYPES = frozenset({"gene_context", "known_gene_proximity", "gene_link"})
_REGULATES_TYPES = frozenset({
    "motif_scan", "tfbs_overlap", "tfbs_density", "regulatory_annotation",
    "rna_structure",
})
_PATHWAY_TYPES = frozenset({
    "pathway_annotation", "pathway_membership", "pathway_projection",
    "go_annotation", "kegg_pathway", "kegg_pathway_annotation",
    "kegg_pathway_membership",
})
_MODEL_EFFECT_TYPES = frozenset({"model_effect", "functional_score"})
_LITERATURE_TYPES = frozenset({"literature"})
_PROTEIN_TYPES = frozenset({
    "protein_domain", "protein_structure", "protein_structure_quality",
    "cds_translation", "sequence_homology", "uniprot_mapping",
})

_EVIDENCE_LEVEL_WEIGHT = {
    "E6": 1.0, "E5": 0.95, "E4": 0.85, "E3": 0.7, "E2": 0.55, "E1": 0.4, "E0": 0.2,
}

_ASSEMBLIES = {
    "rice": "IRGSP-1.0",
    "maize": "AGPv3.25",
    "cotton": "TM-1_CRI",
    "tomato": "SL5.0",
    "chickpea": "CDC_Frontier",
}

_SPECIES_ALIASES = {
    "oryza sativa": "rice",
    "oryza_sativa": "rice",
    "rice": "rice",
    "maize": "maize",
    "zea mays": "maize",
    "cotton": "cotton",
    "gossypium": "cotton",
    "tomato": "tomato",
    "solanum lycopersicum": "tomato",
    "chickpea": "chickpea",
    "cicer arietinum": "chickpea",
}


def normalize_species(value: Any) -> str:
    key = str(value or "").strip().lower()
    return _SPECIES_ALIASES.get(key, key)


def normalize_chrom(value: Any) -> str:
    """Strip ``chr`` prefixes and leading zeros for numeric chromosomes."""
    text = str(value or "").strip()
    lowered = text.lower()
    if lowered.startswith("chr"):
        text = text[3:]
    if text.isdigit() and len(text) > 1:
        text = str(int(text))
    return text


def canonical_key_from_parts(species: str, assembly: str, chrom: Any, pos: Any, ref: Any, alt: Any) -> str | None:
    try:
        return SNPIdentity(
            normalize_species(species), str(assembly or ""), normalize_chrom(chrom),
            int(pos), str(ref), str(alt),
        ).canonical_key
    except (TypeError, ValueError):
        return None


def _stable_id(parts: Iterable[str]) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LDRow:
    snp_a: str
    snp_b: str
    r2: float
    population: str = "submitted_panel"
    distance_bp: int | None = None
    sample_size: int | None = None
    source: str = "ld"
    evidence_ids: tuple[str, ...] = ()


@dataclass
class PopulationRow:
    snp_key: str
    population: str
    af: float | None = None
    ac: int | None = None
    an: int | None = None
    callable_rate: float | None = None
    sample_size: int | None = None
    source: str = "vcf"
    trait: str = ""
    fst: float | None = None
    selection_score: float | None = None


def load_static_ld_table(path: Path | None = None, *, species: str | None = None) -> list[LDRow]:
    """Load the curated LD proxy table into population-scoped LD rows."""
    path = path or Path(os.environ.get("PHENO_GRAPH_LD_PROXY_TABLE", "resources/ld_proxy_table.tsv"))
    if not path.is_file():
        return []
    import csv
    rows: list[LDRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            record_species = normalize_species(record.get("species", ""))
            if species and record_species != normalize_species(species):
                continue
            assembly = str(record.get("reference_build") or _ASSEMBLIES.get(record_species, "unknown"))
            a = _parse_ld_variant_id(str(record.get("query_variant", "")))
            b = _parse_ld_variant_id(str(record.get("lead_snp", "")))
            if not a or not b:
                continue
            key_a = canonical_key_from_parts(record_species, assembly, a[0], a[1], a[2], a[3])
            key_b = canonical_key_from_parts(record_species, assembly, b[0], b[1], b[2], b[3])
            if not key_a or not key_b:
                continue
            try:
                r2 = float(record.get("r2") or 0)
            except (TypeError, ValueError):
                r2 = 0.0
            rows.append(LDRow(
                snp_a=key_a,
                snp_b=key_b,
                r2=r2,
                population=str(record.get("ld_panel") or "curated_ld_panel"),
                source="ld_proxy_table",
            ))
    return rows


def _parse_ld_variant_id(value: str) -> tuple[str, int, str, str] | None:
    """Parse ``chr03_13261184_G_A`` (or ``chr3:13261184:G:A``) into parts."""
    text = str(value or "").strip()
    match = re.match(r"^(?:chr)?([^_:]+)[_:](\d+)[_:]([ACGTacgt])[_:]([ACGTacgt])$", text)
    if not match:
        return None
    chrom, pos, ref, alt = match.groups()
    return normalize_chrom(chrom), int(pos), ref.upper(), alt.upper()


def load_runtime_graph_sources(
    runtime_root: Path | None = None,
    *,
    max_rows: int | None = None,
    species: str | None = None,
) -> tuple[list[LDRow], list[PopulationRow]]:
    import pandas as pd

    root = Path(runtime_root or Path(__file__).resolve().parents[1] / "runtime")
    model_root = root / "model_runs"
    if not model_root.is_dir():
        return [], []


    

    from phenofuncrank_repro.interpretation.provisional import staged_user_run_ids
    staged_runs = staged_user_run_ids(root)

    species_pattern = re.compile(r"(rice|maize|cotton|tomato|chickpea)", re.IGNORECASE)
    ld_rows: list[LDRow] = []
    population_rows: list[PopulationRow] = []

    for run_dir in sorted(model_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name in staged_runs:
            continue
        match = species_pattern.search(run_dir.name)
        if not match:
            continue
        run_species = normalize_species(match.group(1))
        if species and run_species != normalize_species(species):
            continue
        assembly = _ASSEMBLIES.get(run_species, "unknown")


        ld_path = run_dir / "_ld_expanded.tsv"
        if ld_path.is_file() and (max_rows is None or len(ld_rows) < max_rows):
            try:
                ld_df = pd.read_csv(ld_path, sep="\t", low_memory=False)
                id_cols = [c for c in ("user_snp_id", "snp_id") if c in ld_df.columns]
                core_map: dict[str, tuple[str, int, str, str]] = {}
                if id_cols:
                    core_mask = ld_df.get("is_core", pd.Series(False, index=ld_df.index)).astype(str).str.lower().isin({"1", "true", "yes"})
                    for row in ld_df[core_mask].itertuples(index=False):
                        cid = str(getattr(row, id_cols[0], "") or "").strip()
                        key = canonical_key_from_parts(
                            run_species, assembly, getattr(row, "chrom", ""), getattr(row, "pos", ""),
                            getattr(row, "ref", ""), getattr(row, "alt", ""),
                        )
                        if cid and key:
                            core_map[cid] = tuple(key.split("|")[2:])
                r2_col = "r2_to_core" if "r2_to_core" in ld_df.columns else "r2"
                for row in ld_df.itertuples(index=False):
                    if max_rows is not None and len(ld_rows) >= max_rows:
                        break
                    proxy_key = canonical_key_from_parts(
                        run_species, assembly, getattr(row, "chrom", ""), getattr(row, "pos", ""),
                        getattr(row, "ref", ""), getattr(row, "alt", ""),
                    )
                    core_id = str(getattr(row, "core_snp_id", "") or "").strip()
                    core_parts = core_map.get(core_id)
                    if not proxy_key or not core_parts:
                        continue
                    core_key = "|".join((run_species, assembly, *core_parts))
                    try:
                        r2 = float(getattr(row, r2_col))
                    except (TypeError, ValueError):
                        continue
                    if not (0 < r2 < 1):
                        continue
                    ld_rows.append(LDRow(
                        snp_a=core_key,
                        snp_b=proxy_key,
                        r2=r2,
                        population="submitted_panel",
                        source=f"model_run:{run_dir.name}",
                    ))
            except Exception:
                continue


        pop_path = run_dir / "_population_metrics.tsv"
        if pop_path.is_file() and (max_rows is None or len(population_rows) < max_rows):
            try:
                pop_df = pd.read_csv(pop_path, sep="\t", low_memory=False)
                has_wild = "AF_wild" in pop_df.columns
                for row in pop_df.itertuples(index=False):
                    if max_rows is not None and len(population_rows) >= max_rows:
                        break
                    key = canonical_key_from_parts(
                        run_species, assembly, getattr(row, "chrom", ""), getattr(row, "pos", ""),
                        getattr(row, "ref", ""), getattr(row, "alt", ""),
                    )
                    if not key:
                        continue
                    try:
                        af_cultivar = float(getattr(row, "AF_cultivar")) if getattr(row, "AF_cultivar") == getattr(row, "AF_cultivar") else None
                    except (TypeError, ValueError):
                        af_cultivar = None
                    population_rows.append(PopulationRow(
                        snp_key=key,
                        population="cultivar",
                        af=af_cultivar,
                        source=f"model_run:{run_dir.name}",
                    ))
                    if has_wild:
                        try:
                            af_wild = float(getattr(row, "AF_wild")) if getattr(row, "AF_wild") == getattr(row, "AF_wild") else None
                        except (TypeError, ValueError):
                            af_wild = None
                        population_rows.append(PopulationRow(
                            snp_key=key,
                            population="wild",
                            af=af_wild,
                            source=f"model_run:{run_dir.name}",
                        ))
            except Exception:
                continue

    return ld_rows[:max_rows], population_rows[:max_rows]


class VariantGraphBuilder:
    """Deterministically materialize the variant-centric evidence graph."""

    def __init__(self, store: EvidenceStore, species: str | None = None) -> None:
        self.store = store
        self.species = normalize_species(species) if species else None

    def _graph_conn(self):
        if self.species:
            return self.store.connect_graph(self.species)
        return self.store.connect()

    def rebuild(
        self,
        *,
        ld_rows: list[LDRow] | None = None,
        population_rows: list[PopulationRow] | None = None,
        include_static_ld: bool = True,
        include_runtime_sources: bool = True,
    ) -> dict[str, Any]:
        """Clear and rebuild the graph from evidence + external sources."""
        all_ld = list(ld_rows or [])
        all_pop = list(population_rows or [])
        if include_static_ld:
            all_ld.extend(load_static_ld_table(species=self.species))
        if include_runtime_sources:
            runtime_ld, runtime_pop = load_runtime_graph_sources(species=self.species)
            all_ld.extend(runtime_ld)
            all_pop.extend(runtime_pop)

        with self._graph_conn() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute("DELETE FROM graph_edges")
            con.execute("DELETE FROM graph_nodes")
            try:
                con.execute("DELETE FROM sqlite_sequence WHERE name IN ('graph_nodes','graph_edges')")
            except Exception:
                pass  

            self._build_from_evidence(con, snp_keys=None)
            self._ingest_ld_rows(con, all_ld)
            self._ingest_population_rows(con, all_pop)
            self._build_population_comparisons(con)
            self._ingest_population_layers_into_graph(con)

            nodes = con.execute("SELECT COUNT(*) AS n FROM graph_nodes").fetchone()["n"]
            edges = con.execute("SELECT COUNT(*) AS n FROM graph_edges").fetchone()["n"]
            counts = {
                "nodes": int(nodes),
                "edges": int(edges),
                "node_types": {
                    str(row["node_type"]): int(row["n"])
                    for row in con.execute(
                        "SELECT node_type, COUNT(*) AS n FROM graph_nodes GROUP BY node_type"
                    ).fetchall()
                },
                "edge_predicates": {
                    str(row["predicate"]): int(row["n"])
                    for row in con.execute(
                        "SELECT predicate, COUNT(*) AS n FROM graph_edges GROUP BY predicate"
                    ).fetchall()
                },
            }
            self._log_evolution("variant_graph", "rebuild", counts)
        return counts

    def _ingest_population_layers_into_graph(self, con: Any) -> int:
        """Re-materialize OBSERVED_IN edges from the full population_layers table.

        Rebuilds are idempotent with respect to the full population layer: the
        cultivar-memmap / wild-VCF AF table is the source of truth, and graph
        edges are always recreated from it after a clear+rebuild.
        """
        if self.species is None:
            return 0
        now = _now()
        inserted = 0
        seen_populations: set[str] = set()
        with self.store.connect() as src:
            cursor = src.execute(
                "SELECT * FROM population_layers WHERE species=? ORDER BY population, source",
                (self.species,),
            )
            while True:
                batch = cursor.fetchmany(100_000)
                if not batch:
                    break
                node_values: list[tuple] = []
                edge_values: list[tuple] = []
                for row in batch:
                    snp_key = str(row["snp_key"])
                    parts = snp_key.split("|")
                    if len(parts) != 6:
                        continue
                    population = str(row["population"])
                    source = str(row["source"])
                    population_node = f"population:{self.species}:{population}"
                    if population_node not in seen_populations:
                        con.execute(
                            """INSERT INTO graph_nodes (
                                node_id, node_type, attributes_json, first_seen_at, last_seen_at, hit_count
                            ) VALUES (?, 'population', ?, ?, ?, 1)
                            ON CONFLICT(node_id) DO UPDATE SET last_seen_at=excluded.last_seen_at""",
                            (population_node, _json({
                                "species": self.species, "population": population,
                                "source": source,
                            }), now, now),
                        )
                        seen_populations.add(population_node)
                    node_values.append((
                        snp_key, "variant",
                        _json({
                            "species": self.species, "assembly": parts[1],
                            "chrom": parts[2], "pos": int(parts[3]),
                            "ref": parts[4], "alt": parts[5],
                        }),
                        now, now,
                    ))
                    edge_id = _stable_id((snp_key, "OBSERVED_IN", population_node))
                    edge_values.append((
                        edge_id, snp_key, "OBSERVED_IN", population_node,
                        _json({
                            "AF": row["af"], "AC": row["ac"], "AN": row["an"],
                            "callable_rate": row["callable_rate"],
                            "sample_size": None, "trait": "", "source": source,
                        }),
                        "[]", None, None, 1, now, now,
                    ))
                for i in range(0, len(node_values), 20_000):
                    con.executemany(
                        """INSERT OR IGNORE INTO graph_nodes (
                            node_id, node_type, attributes_json, first_seen_at, last_seen_at, hit_count
                        ) VALUES (?, ?, ?, ?, ?, 1)""",
                        node_values[i:i + 20_000],
                    )
                for i in range(0, len(edge_values), 20_000):
                    con.executemany(
                        """INSERT INTO graph_edges (
                            edge_id, source, predicate, target, properties_json,
                            evidence_ids_json, confidence, directness, version,
                            first_seen_at, last_seen_at, hit_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                        ON CONFLICT(edge_id) DO NOTHING""",
                        edge_values[i:i + 20_000],
                    )
                inserted += len(edge_values)
        if inserted:
            self._log_evolution("variant_graph", "rebuild_population_layers", {
                "species": self.species, "edges": inserted,
            })
        return inserted

    def materialize_population_edges_for_snps(self, snp_keys: list[str]) -> dict[str, Any]:
        """Materialize OBSERVED_IN edges only for the given SNP set.

        The hybrid population strategy: full AF lives in population_layers;
        graph edges are created for exactly the SNPs that matter (promoted
        candidates / model core + LD-expanded sets), plus on-demand lookups.
        """
        if self.species is None:
            raise ValueError("materialize_population_edges_for_snps requires a species-scoped builder")
        snp_keys = [str(k) for k in dict.fromkeys(snp_keys) if str(k).strip()]
        if not snp_keys:
            return {"species": self.species, "edges_inserted": 0}
        now = _now()
        inserted = 0
        seen_populations: set[str] = set()
        with self._graph_conn() as con:
            con.execute("BEGIN IMMEDIATE")
            for start in range(0, len(snp_keys), 5000):
                keys = snp_keys[start:start + 5000]
                placeholders = ",".join("?" for _ in keys)
                with self.store.connect() as src:
                    rows = src.execute(
                        f"SELECT * FROM population_layers WHERE snp_key IN ({placeholders})",
                        keys,
                    ).fetchall()
                node_values: list[tuple] = []
                edge_values: list[tuple] = []
                for row in rows:
                    snp_key = str(row["snp_key"])
                    parts = snp_key.split("|")
                    if len(parts) != 6:
                        continue
                    population = str(row["population"])
                    source = str(row["source"])
                    population_node = f"population:{self.species}:{population}"
                    if population_node not in seen_populations:
                        con.execute(
                            """INSERT OR IGNORE INTO graph_nodes (
                                node_id, node_type, attributes_json, first_seen_at, last_seen_at, hit_count
                            ) VALUES (?, 'population', ?, ?, ?, 1)""",
                            (population_node, _json({
                                "species": self.species, "population": population,
                                "source": source,
                            }), now, now),
                        )
                        seen_populations.add(population_node)
                    node_values.append((
                        snp_key, "variant",
                        _json({
                            "species": self.species, "assembly": parts[1],
                            "chrom": parts[2], "pos": int(parts[3]),
                            "ref": parts[4], "alt": parts[5],
                        }),
                        now, now,
                    ))
                    edge_id = _stable_id((snp_key, "OBSERVED_IN", population_node))
                    edge_values.append((
                        edge_id, snp_key, "OBSERVED_IN", population_node,
                        _json({
                            "AF": row["af"], "AC": row["ac"], "AN": row["an"],
                            "callable_rate": row["callable_rate"],
                            "sample_size": None, "trait": "", "source": source,
                        }),
                        "[]", None, None, 1, now, now,
                    ))
                for i in range(0, len(node_values), 20_000):
                    con.executemany(
                        """INSERT OR IGNORE INTO graph_nodes (
                            node_id, node_type, attributes_json, first_seen_at, last_seen_at, hit_count
                        ) VALUES (?, ?, ?, ?, ?, 1)""",
                        node_values[i:i + 20_000],
                    )
                for i in range(0, len(edge_values), 20_000):
                    con.executemany(
                        """INSERT INTO graph_edges (
                            edge_id, source, predicate, target, properties_json,
                            evidence_ids_json, confidence, directness, version,
                            first_seen_at, last_seen_at, hit_count
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                        ON CONFLICT(edge_id) DO NOTHING""",
                        edge_values[i:i + 20_000],
                    )
                inserted += len(edge_values)
        self._log_evolution("variant_graph", "materialize_population_edges", {
            "species": self.species, "snps": len(snp_keys), "edges": inserted,
        })
        return {"species": self.species, "snps": len(snp_keys), "edges_inserted": inserted}

    def materialize_population_edges_from_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Materialize OBSERVED_IN edges directly from population-layer rows.

        Used by the candidate API to attach edges under the graph's own SNP key
        (assembly labels can drift between IRGSP-1.0 and unknown).
        """
        if self.species is None:
            raise ValueError("materialize_population_edges_from_rows requires a species-scoped builder")
        rows = [dict(r) for r in rows if str(r.get("snp_key") or "").strip()]
        if not rows:
            return {"species": self.species, "edges_inserted": 0}
        now = _now()
        inserted = 0
        seen_populations: set[str] = set()
        with self._graph_conn() as con:
            con.execute("BEGIN IMMEDIATE")
            node_values: list[tuple] = []
            edge_values: list[tuple] = []
            for row in rows:
                snp_key = str(row["snp_key"])
                parts = snp_key.split("|")
                if len(parts) != 6:
                    continue
                population = str(row["population"])
                source = str(row["source"])
                population_node = f"population:{self.species}:{population}"
                if population_node not in seen_populations:
                    con.execute(
                        """INSERT OR IGNORE INTO graph_nodes (
                            node_id, node_type, attributes_json, first_seen_at, last_seen_at, hit_count
                        ) VALUES (?, 'population', ?, ?, ?, 1)""",
                        (population_node, _json({
                            "species": self.species, "population": population,
                            "source": source,
                        }), now, now),
                    )
                    seen_populations.add(population_node)
                node_values.append((
                    snp_key, "variant",
                    _json({
                        "species": self.species, "assembly": parts[1],
                        "chrom": parts[2], "pos": int(parts[3]),
                        "ref": parts[4], "alt": parts[5],
                    }),
                    now, now,
                ))
                edge_id = _stable_id((snp_key, "OBSERVED_IN", population_node))
                edge_values.append((
                    edge_id, snp_key, "OBSERVED_IN", population_node,
                    _json({
                        "AF": row.get("af"), "AC": row.get("ac"), "AN": row.get("an"),
                        "callable_rate": row.get("callable_rate"),
                        "sample_size": None, "trait": "", "source": source,
                    }),
                    "[]", None, None, 1, now, now,
                ))
            for i in range(0, len(node_values), 20_000):
                con.executemany(
                    """INSERT OR IGNORE INTO graph_nodes (
                        node_id, node_type, attributes_json, first_seen_at, last_seen_at, hit_count
                    ) VALUES (?, ?, ?, ?, ?, 1)""",
                    node_values[i:i + 20_000],
                )
            for i in range(0, len(edge_values), 20_000):
                con.executemany(
                    """INSERT INTO graph_edges (
                        edge_id, source, predicate, target, properties_json,
                        evidence_ids_json, confidence, directness, version,
                        first_seen_at, last_seen_at, hit_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(edge_id) DO NOTHING""",
                    edge_values[i:i + 20_000],
                )
            inserted += len(edge_values)
        self._log_evolution("variant_graph", "materialize_population_edges", {
            "species": self.species, "snps": len({r["snp_key"] for r in rows}),
            "edges": inserted,
        })
        return {"species": self.species, "edges_inserted": inserted}

    def update_for_snps(
        self,
        snp_keys: list[str],
        *,
        include_global_counts: bool | None = None,
    ) -> dict[str, Any]:
        snp_keys = [str(k) for k in dict.fromkeys(snp_keys) if str(k).strip()]
        if not snp_keys:
            return self.stats()
        if self.species is None:
            raise ValueError("update_for_snps requires a species-scoped builder")
        graph_path = self.store.graph_db_path(self.species)
        if include_global_counts is None:
            try:
                wal_path = Path(str(graph_path) + "-wal")
                graph_bytes = graph_path.stat().st_size + (wal_path.stat().st_size if wal_path.exists() else 0)
                include_global_counts = graph_bytes <= settings.graph_stats_max_bytes
            except OSError:
                include_global_counts = True
        with self._graph_conn() as con:
            con.execute("BEGIN IMMEDIATE")
            affected = self._affected_neighborhood(con, snp_keys, hops=1)
            for key in affected:
                con.execute(
                    "DELETE FROM graph_edges WHERE source=? OR target=?",
                    (key, key),
                )
                con.execute(
                    """DELETE FROM graph_edges
                       WHERE predicate='DIFFERENT_FROM'
                         AND json_extract(properties_json, '$.snp')=?""",
                    (key,),
                )
            self._build_from_evidence(con, snp_keys=affected)
            self._build_population_comparisons(con, snp_keys=affected)
            counts = {
                "species": self.species,
                "snps_requested": len(snp_keys),
                "snps_updated": len(affected),
            }
            if include_global_counts:
                counts.update({
                    "nodes": int(con.execute("SELECT COUNT(*) AS n FROM graph_nodes").fetchone()["n"]),
                    "edges": int(con.execute("SELECT COUNT(*) AS n FROM graph_edges").fetchone()["n"]),
                    "edge_predicates": {
                        str(row["predicate"]): int(row["n"])
                        for row in con.execute(
                            "SELECT predicate, COUNT(*) AS n FROM graph_edges GROUP BY predicate"
                        ).fetchall()
                    },
                })
            else:
                counts["global_counts_deferred"] = True
            self._log_evolution("variant_graph", "update_for_snps", counts)
        return counts

    def _affected_neighborhood(
        self, con: Any, snp_keys: list[str], *, hops: int = 1,
    ) -> list[str]:
        """Expand an update set to its 1-hop variant neighborhood."""
        affected = set(snp_keys)
        frontier = set(snp_keys)
        for _ in range(max(1, hops)):
            next_frontier: set[str] = set()
            for key in frontier:
                rows = con.execute(
                    "SELECT source, target FROM graph_edges WHERE source=? OR target=?",
                    (key, key),
                ).fetchall()
                for row in rows:
                    for node_id in (row["source"], row["target"]):
                        node_id = str(node_id)
                        if "|" in node_id and node_id not in affected:
                            affected.add(node_id)
                            next_frontier.add(node_id)
            frontier = next_frontier
            if not frontier:
                break

        
        keys = sorted(affected)
        placeholders = ",".join("?" for _ in keys)
        with self.store.connect() as src:
            rows = src.execute(
                f"""SELECT DISTINCT snp_key FROM evidence_records
                    WHERE evidence_type='ld_proxy' AND object IN ({placeholders})""",
                keys,
            ).fetchall()
        for row in rows:
            affected.add(str(row["snp_key"]))
        return sorted(affected)

    def _log_evolution(self, component: str, action: str, detail: dict[str, Any]) -> None:
        try:
            self.store.log_evolution(component, action, detail)
        except Exception:
            pass


    def _upsert_node(
        self, con: Any, *, node_id: str, node_type: str, attributes: dict[str, Any],
    ) -> None:
        now = _now()
        con.execute(
            """INSERT INTO graph_nodes (node_id, node_type, attributes_json, first_seen_at, last_seen_at, hit_count)
               VALUES (?, ?, ?, ?, ?, 1)
               ON CONFLICT(node_id) DO UPDATE SET
                   node_type=excluded.node_type,
                   attributes_json=excluded.attributes_json,
                   last_seen_at=excluded.last_seen_at,
                   hit_count=graph_nodes.hit_count + 1""",
            (node_id, node_type, _json(attributes), now, now),
        )

    def _upsert_edge(
        self,
        con: Any,
        *,
        source: str,
        predicate: str,
        target: str,
        properties: dict[str, Any],
        evidence_ids: Iterable[str] = (),
        confidence: float | None = None,
        directness: str | None = None,
        edge_key_suffix: str | None = None,
    ) -> str:
        if predicate == "LD_WITH" and source > target:
            source, target = target, source
        key_parts = (source, predicate, target)
        if edge_key_suffix:
            key_parts = (*key_parts, edge_key_suffix)
        edge_id = _stable_id(key_parts)
        now = _now()
        existing = con.execute(
            "SELECT evidence_ids_json, confidence, directness, version FROM graph_edges WHERE edge_id=?",
            (edge_id,),
        ).fetchone()
        merged: list[str] = []
        if existing is not None:
            try:
                merged = json.loads(existing["evidence_ids_json"] or "[]")
            except (TypeError, ValueError, json.JSONDecodeError):
                merged = []
        for eid in evidence_ids:
            if eid and eid not in merged:
                merged.append(eid)
        merged_conf = confidence if confidence is not None else (existing["confidence"] if existing else None)
        merged_directness = directness or (existing["directness"] if existing else None)

        try:
            con.execute(
                "INSERT OR IGNORE INTO graph_edge_versions(edge_key, version) VALUES (?, 0)",
                (edge_id,),
            )
            con.execute(
                "UPDATE graph_edge_versions SET version=version+1 WHERE edge_key=?",
                (edge_id,),
            )
            version_row = con.execute(
                "SELECT version FROM graph_edge_versions WHERE edge_key=?", (edge_id,),
            ).fetchone()
            version = int(version_row["version"]) if version_row is not None else 1
        except Exception:
            version = (int(existing["version"]) + 1) if existing is not None else 1
        con.execute(
            """INSERT INTO graph_edges (
                   edge_id, source, predicate, target, properties_json,
                   evidence_ids_json, confidence, directness, version,
                   first_seen_at, last_seen_at, hit_count
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
               ON CONFLICT(edge_id) DO UPDATE SET
                   properties_json=excluded.properties_json,
                   evidence_ids_json=excluded.evidence_ids_json,
                   confidence=excluded.confidence,
                   directness=excluded.directness,
                   version=excluded.version,
                   last_seen_at=excluded.last_seen_at,
                   hit_count=graph_edges.hit_count + 1""",
            (
                edge_id, source, predicate, target, _json(properties),
                _json(sorted(merged)), merged_conf, merged_directness, version, now, now,
            ),
        )
        return edge_id


    def _build_from_evidence(self, con: Any, *, snp_keys: list[str] | None = None) -> None:

        

        

        if self.species:
            with self.store.connect() as src:
                if snp_keys:
                    placeholders = ",".join("?" for _ in snp_keys)
                    rows = src.execute(
                        f"""SELECT evidence_records.*, tool_runs.tool_name, tool_runs.tool_version
                            FROM evidence_records
                            JOIN snps ON snps.canonical_key=evidence_records.snp_key
                            LEFT JOIN tool_runs ON tool_runs.id=evidence_records.tool_run_id
                            WHERE snps.species=? AND evidence_records.snp_key IN ({placeholders})
                            ORDER BY evidence_records.created_at""",
                        [self.species, *snp_keys],
                    ).fetchall()
                    snps = src.execute(
                        f"SELECT * FROM snps WHERE species=? AND canonical_key IN ({placeholders})",
                        [self.species, *snp_keys],
                    ).fetchall()
                else:
                    rows = src.execute(
                        """SELECT evidence_records.*, tool_runs.tool_name, tool_runs.tool_version
                           FROM evidence_records
                           JOIN snps ON snps.canonical_key=evidence_records.snp_key
                           LEFT JOIN tool_runs ON tool_runs.id=evidence_records.tool_run_id
                           WHERE snps.species=?
                           ORDER BY evidence_records.created_at""",
                        (self.species,),
                    ).fetchall()
                    snps = src.execute(
                        "SELECT * FROM snps WHERE species=?", (self.species,),
                    ).fetchall()
        elif snp_keys:
            placeholders = ",".join("?" for _ in snp_keys)
            rows = con.execute(
                f"""SELECT evidence_records.*, tool_runs.tool_name, tool_runs.tool_version
                    FROM evidence_records LEFT JOIN tool_runs ON tool_runs.id=evidence_records.tool_run_id
                    WHERE evidence_records.snp_key IN ({placeholders})
                    ORDER BY evidence_records.created_at""",
                snp_keys,
            ).fetchall()
            snps = con.execute(
                f"SELECT * FROM snps WHERE canonical_key IN ({placeholders})", snp_keys,
            ).fetchall()
        else:
            rows = con.execute(
                """SELECT evidence_records.*, tool_runs.tool_name, tool_runs.tool_version
                   FROM evidence_records LEFT JOIN tool_runs ON tool_runs.id=evidence_records.tool_run_id
                   ORDER BY evidence_records.created_at""",
            ).fetchall()
            snps = con.execute("SELECT * FROM snps").fetchall()
        for snp in snps:
            self._upsert_node(con, node_id=snp["canonical_key"], node_type="variant", attributes={
                "species": snp["species"],
                "assembly": snp["assembly"],
                "chrom": snp["chrom"],
                "pos": snp["pos"],
                "ref": snp["ref"],
                "alt": snp["alt"],
            })

        records: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            try:
                record["scope"] = json.loads(record["scope_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                record["scope"] = {}
            try:
                record["payload"] = json.loads(record["payload_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                record["payload"] = {}
            records.append(record)

        snp_genes: dict[str, list[str]] = {}
        for record in records:
            evidence_type = str(record["evidence_type"] or "")
            if evidence_type not in _GENE_ANCHOR_TYPES:
                continue
            species = normalize_species(record["scope"].get("species", ""))
            gene_id = str(record["scope"].get("source_record_id") or "").strip()
            if not gene_id and evidence_type == "gene_link":
                gene_id = str(record["object"] or "").strip()
            if not gene_id:
                match = re.search(r"\bgene (\S+)(?: \((\S+)\))?", str(record["object"] or ""))
                if match:
                    gene_id = match.group(2) or match.group(1)
            if not gene_id:
                continue
            gene_node = f"gene:{species}:{gene_id}"
            self._upsert_node(con, node_id=gene_node, node_type="gene", attributes={
                "species": species, "gene_id": gene_id,
            })
            snp_genes.setdefault(record["snp_key"], []).append(gene_node)
            distance: int | None = None
            region_type = "nearby"
            if evidence_type == "gene_context":
                distance = 0
                region_type = _region_type_from_statement(str(record["object"] or ""))
            else:
                match = re.search(r"is (\d+)bp from", str(record["object"] or ""))
                if match:
                    distance = int(match.group(1))
                if evidence_type == "gene_link":
                    region_type = "functional_annotation"
            properties: dict[str, Any] = {
                "distance_bp": distance,
                "region_type": region_type,
                "annotation_source": record["scope"].get("source") or evidence_type,
                "evidence_type": evidence_type,
            }
            if record["scope"].get("directness"):
                properties["directness"] = record["scope"]["directness"]
            self._upsert_edge(
                con,
                source=record["snp_key"],
                predicate="LOCATED_IN",
                target=gene_node,
                properties=properties,
                evidence_ids=[record["id"]],
                confidence=_confidence(record),
                directness=record["scope"].get("directness"),
            )
            trait = str(record["scope"].get("trait") or record["payload"].get("trait") or "").strip()
            if trait:
                self._link_gene_trait(con, gene_node, trait, record)


        for record in records:
            if str(record["evidence_type"] or "") not in _REGULATES_TYPES:
                continue
            species = normalize_species(record["scope"].get("species", ""))
            target_genes: list[str] = []
            subject = str(record["subject"] or "").strip()
            subject_gene = f"gene:{species}:{subject}"
            if subject and subject_gene in {
                str(row["node_id"]) for row in con.execute(
                    "SELECT node_id FROM graph_nodes WHERE node_type='gene'"
                ).fetchall()
            }:
                target_genes.append(subject_gene)
            else:

                
                target_genes.extend(snp_genes.get(record["snp_key"], []))
            for gene_node in dict.fromkeys(target_genes):
                self._upsert_edge(
                    con,
                    source=record["snp_key"],
                    predicate="REGULATES",
                    target=gene_node,
                    properties={
                        "evidence_type": record["evidence_type"],
                        "source": record["scope"].get("source") or "",
                        "directness": record["scope"].get("directness") or "",
                    },
                    evidence_ids=[record["id"]],
                    confidence=_confidence(record),
                    directness=record["scope"].get("directness"),
                )


        for record in records:
            if str(record["evidence_type"] or "") not in _PROTEIN_TYPES:
                continue
            species = normalize_species(record["scope"].get("species", ""))
            accession = str(
                record["scope"].get("source_record_id")
                or record["payload"].get("accession")
                or record["payload"].get("uniprot_acc")
                or record["subject"]
                or ""
            ).strip()
            if not accession:
                continue
            protein_node = f"protein:{species}:{accession}"
            self._upsert_node(con, node_id=protein_node, node_type="protein", attributes={
                "species": species,
                "accession": accession,
                "evidence_type": record["evidence_type"],
                "source": record["scope"].get("source") or "",
            })
            self._upsert_edge(
                con,
                source=record["snp_key"],
                predicate="AFFECTS",
                target=protein_node,
                properties={
                    "evidence_type": record["evidence_type"],
                    "source": record["scope"].get("source") or "",
                    "trait": record["scope"].get("trait") or "",
                },
                evidence_ids=[record["id"]],
                confidence=_confidence(record),
            )


        for record in records:
            if str(record["evidence_type"] or "") not in _PATHWAY_TYPES:
                continue
            species = normalize_species(record["scope"].get("species", ""))
            pathway_name = str(record["object"] or "").strip()
            if not pathway_name:
                continue
            pathway_node = f"pathway:{species}:{pathway_name}"
            self._upsert_node(con, node_id=pathway_node, node_type="pathway", attributes={
                "species": species, "pathway": pathway_name,
                "source": record["scope"].get("source") or record["evidence_type"],
            })
            genes = _resolve_gene_nodes(con, record, species, snp_genes)
            for gene_node in genes:
                self._upsert_edge(
                    con,
                    source=gene_node,
                    predicate="PARTICIPATES_IN",
                    target=pathway_node,
                    properties={
                        "source": record["scope"].get("source") or record["evidence_type"],
                        "evidence_type": record["evidence_type"],
                    },
                    evidence_ids=[record["id"]],
                    confidence=_confidence(record),
                    directness=record["scope"].get("directness"),
                )
            trait = str(record["scope"].get("trait") or record["payload"].get("trait") or "").strip()
            if trait:
                trait_node = f"trait:{trait}"
                self._upsert_node(con, node_id=trait_node, node_type="trait", attributes={"trait": trait})
                self._upsert_edge(
                    con,
                    source=pathway_node,
                    predicate="RELATED_TO",
                    target=trait_node,
                    properties={"source": record["scope"].get("source") or ""},
                    evidence_ids=[record["id"]],
                    confidence=_confidence(record),
                )


        for record in records:
            if str(record["evidence_type"] or "") not in _MODEL_EFFECT_TYPES:
                continue
            species = normalize_species(record["scope"].get("species", ""))
            model_name = _model_name(record)
            model_node = f"model:{species}:{model_name}"
            self._upsert_node(con, node_id=model_node, node_type="model", attributes={
                "species": species,
                "model_name": model_name,
                "version": record.get("tool_version") or "",
            })
            value = record["value_json"]
            try:
                value = json.loads(value) if value is not None else None
            except (TypeError, ValueError, json.JSONDecodeError):
                value = None
            properties: dict[str, Any] = {
                "score": value,
                "unit": record.get("unit") or "",
                "direction": record.get("direction") or "",
                "confidence": record.get("confidence"),
                "tool_run_id": record.get("tool_run_id"),
                "evidence_level": record.get("evidence_level"),
                "source": record["scope"].get("source") or "",
                "trait": record["scope"].get("trait") or "",
                "p_value": (
                    record["payload"].get("dpc_empirical_p_matched")
                    or record["payload"].get("pvalue_label")
                ),
                "percentile": (
                    record["payload"].get("percentile")
                    or record["payload"].get("cadd_percentile")
                ),
            }
            self._upsert_edge(
                con,
                source=record["snp_key"],
                predicate="SCORED_BY",
                target=model_node,
                properties=properties,
                evidence_ids=[record["id"]],
                confidence=_confidence(record),
            )


        for record in records:
            if str(record["evidence_type"] or "") != "ld_proxy":
                continue
            try:
                r2 = float(json.loads(record["value_json"])) if record["value_json"] is not None else None
            except (TypeError, ValueError, json.JSONDecodeError):
                r2 = None
            if r2 is None:
                continue
            population = str(
                record["scope"].get("population")
                or record["scope"].get("ld_panel")
                or ("submitted_panel" if record["scope"].get("within_submitted_panel") else "unknown")
            )
            self._upsert_edge(
                con,
                source=record["snp_key"],
                predicate="LD_WITH",
                target=str(record["object"] or ""),
                properties={
                    "r2": r2,
                    "population": population,
                    "unit": record.get("unit") or "r2",
                    "source": record["scope"].get("source") or "evidence",
                    "trait": record["scope"].get("trait") or "",
                },
                evidence_ids=[record["id"]],
                confidence=_confidence(record),
                edge_key_suffix=population or "unknown",
            )


        for record in records:
            if str(record["evidence_type"] or "") != "population_metric":
                continue
            species = normalize_species(record["scope"].get("species", ""))
            population = str(record["scope"].get("population") or "").strip()
            if not population:
                continue
            try:
                af = float(json.loads(record["value_json"])) if record["value_json"] is not None else None
            except (TypeError, ValueError, json.JSONDecodeError):
                af = None
            payload = record["payload"]
            self._ingest_population_row(con, PopulationRow(
                snp_key=record["snp_key"],
                population=population,
                af=af,
                ac=_safe_int(payload.get("ac")),
                an=_safe_int(payload.get("an")),
                callable_rate=_safe_float(payload.get("callable_rate")),
                sample_size=_safe_int(payload.get("sample_size")),
                source=str(payload.get("source") or record["scope"].get("source") or "evidence"),
                trait=str(record["scope"].get("trait") or ""),
                fst=_safe_float(payload.get("fst")),
                selection_score=_safe_float(payload.get("selection_score")),
            ), evidence_ids=[record["id"]])


        for record in records:
            if str(record["evidence_type"] or "") not in _LITERATURE_TYPES:
                continue
            evidence_id = str(
                record["payload"].get("evidence_id")
                or record.get("external_id")
                or _stable_id(("evidence", record["snp_key"], str(record["object"] or "")[:200]))
            )
            evidence_node = f"evidence:{evidence_id}"
            species = normalize_species(record["scope"].get("species", ""))
            self._upsert_node(con, node_id=evidence_node, node_type="evidence", attributes={
                "evidence_type": record["scope"].get("source") or "literature",
                "source": record["scope"].get("source") or "",
                "confidence": record.get("confidence"),
                "publication": {
                    "title": record["payload"].get("title") or "",
                    "year": record["payload"].get("year") or "",
                    "journal": record["payload"].get("journal") or "",
                    "doi": record["payload"].get("doi") or "",
                    "pmid": record["payload"].get("pmid") or "",
                },
                "claim": str(record["object"] or "")[:500],
            })

            subject = str(record["subject"] or "").strip()
            subject_gene = f"gene:{species}:{subject}"
            exists = con.execute(
                "SELECT 1 FROM graph_nodes WHERE node_id=? AND node_type='gene'", (subject_gene,),
            ).fetchone()
            if exists:
                self._upsert_edge(
                    con,
                    source=subject_gene,
                    predicate="SUPPORTED_BY",
                    target=evidence_node,
                    properties={
                        "source": record["scope"].get("source") or "",
                        "trait": record["scope"].get("trait") or "",
                    },
                    evidence_ids=[record["id"]],
                    confidence=_confidence(record),
                )

            self._upsert_edge(
                con,
                source=record["snp_key"],
                predicate="SUPPORTED_BY",
                target=evidence_node,
                properties={
                    "source": record["scope"].get("source") or "",
                    "trait": record["scope"].get("trait") or "",
                },
                evidence_ids=[record["id"]],
                confidence=_confidence(record),
            )


        for record in records:
            if str(record["evidence_type"] or "") not in {
                "model_effect", "population_metric", "literature", "ld_proxy",
            }:
                continue
            try:
                scope = json.loads(record["scope_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                scope = {}
            trait = str(scope.get("trait") or "").strip()
            if not trait:
                continue
            species = normalize_species(scope.get("species", ""))
            trait_node = f"trait:{trait}"
            self._upsert_node(con, node_id=trait_node, node_type="trait", attributes={"trait": trait})
            self._upsert_edge(
                con,
                source=record["snp_key"],
                predicate="ASSOCIATED_WITH",
                target=trait_node,
                properties={
                    "trait": trait,
                    "species": species,
                    "evidence_type": record["evidence_type"],
                    "source": scope.get("source") or "",
                },
                evidence_ids=[record["id"]],
                confidence=_confidence(record),
            )


    def _link_gene_trait(self, con: Any, gene_node: str, trait: str, record: dict[str, Any]) -> None:
        trait_node = f"trait:{trait}"
        self._upsert_node(con, node_id=trait_node, node_type="trait", attributes={"trait": trait})
        self._upsert_edge(
            con,
            source=gene_node,
            predicate="RELATED_TO",
            target=trait_node,
            properties={"source": record["scope"].get("source") or ""},
            evidence_ids=[record["id"]],
            confidence=_confidence(record),
        )

    def _ingest_population_row(
        self, con: Any, row: PopulationRow, *, evidence_ids: Iterable[str] = (),
    ) -> None:
        parts = row.snp_key.split("|")
        if len(parts) != 6:
            return
        species, assembly, chrom, pos, ref, alt = parts
        variant_node = row.snp_key
        self._upsert_node(con, node_id=variant_node, node_type="variant", attributes={
            "species": species, "assembly": assembly, "chrom": chrom,
            "pos": int(pos), "ref": ref, "alt": alt,
        })
        population_node = f"population:{species}:{row.population}"
        self._upsert_node(con, node_id=population_node, node_type="population", attributes={
            "species": species,
            "population": row.population,
            "source": row.source,
        })
        self._upsert_edge(
            con,
            source=variant_node,
            predicate="OBSERVED_IN",
            target=population_node,
            properties={
                "AF": row.af,
                "AC": row.ac,
                "AN": row.an,
                "callable_rate": row.callable_rate,
                "sample_size": row.sample_size,
                "trait": row.trait,
                "source": row.source,
            },
            evidence_ids=evidence_ids,
        )

    def _ingest_ld_rows(self, con: Any, rows: list[LDRow]) -> None:
        for row in rows:
            self._upsert_variant_from_key(con, row.snp_a)
            self._upsert_variant_from_key(con, row.snp_b)
            self._upsert_edge(
                con,
                source=row.snp_a,
                predicate="LD_WITH",
                target=row.snp_b,
                properties={
                    "r2": row.r2,
                    "population": row.population,
                    "distance_bp": row.distance_bp,
                    "sample_size": row.sample_size,
                    "source": row.source,
                    "trait": "",
                },
                evidence_ids=row.evidence_ids,
                confidence=max(0.0, min(1.0, float(row.r2))),
                edge_key_suffix=row.population or "unknown",
            )

    def ingest_ld_rows(self, rows: list[LDRow]) -> dict[str, Any]:
        """Upsert LD_WITH edges without clearing the rest of the graph.

        Used by the full-panel/model-driven universe builders: every retained
        core→proxy pair (r²>=0.8) is kept, with no neighbor cap.
        """
        if not rows:
            return {"species": self.species, "ld_edges": 0}
        with self._graph_conn() as con:
            con.execute("BEGIN IMMEDIATE")
            self._ingest_ld_rows(con, rows)
        self._log_evolution("variant_graph", "ingest_ld_rows", {
            "species": self.species, "ld_edges": len(rows),
        })
        return {"species": self.species, "ld_edges": len(rows)}

    def _ingest_population_rows(self, con: Any, rows: list[PopulationRow]) -> None:
        for row in rows:
            self._ingest_population_row(con, row)

    def bulk_ingest_population_edges(
        self,
        rows: list[dict[str, Any]],
        *,
        population: str,
        source: str,
    ) -> dict[str, Any]:
        """Bulk-materialize OBSERVED_IN edges for a full population layer.

        Used by the population ingestion pipeline (cultivar memmap / wild VCF);
        rows carry snp_key/chrom/pos/ref/alt/af/ac/an/callable_rate.
        """
        if self.species is None:
            raise ValueError("bulk_ingest_population_edges requires a species-scoped builder")
        if not rows:
            return {"species": self.species, "edges_inserted": 0}
        species = self.species
        population_node = f"population:{species}:{population}"
        now = _now()
        with self._graph_conn() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                """INSERT INTO graph_nodes (node_id, node_type, attributes_json, first_seen_at, last_seen_at, hit_count)
                   VALUES (?, 'population', ?, ?, ?, 1)
                   ON CONFLICT(node_id) DO UPDATE SET last_seen_at=excluded.last_seen_at, hit_count=graph_nodes.hit_count+1""",
                (population_node, _json({"species": species, "population": population, "source": source}), now, now),
            )
            node_values = []
            edge_values = []
            for row in rows:
                snp_key = str(row["snp_key"])
                parts = snp_key.split("|")
                if len(parts) != 6:
                    continue
                assembly, chrom, pos, ref, alt = parts[1], parts[2], parts[3], parts[4], parts[5]
                node_values.append((
                    snp_key, "variant",
                    _json({"species": species, "assembly": assembly, "chrom": chrom,
                           "pos": int(pos), "ref": ref, "alt": alt}),
                    now, now,
                ))
                edge_id = _stable_id((snp_key, "OBSERVED_IN", population_node))
                properties = {
                    "AF": row.get("af"),
                    "AC": row.get("ac"),
                    "AN": row.get("an"),
                    "callable_rate": row.get("callable_rate"),
                    "sample_size": row.get("sample_size"),
                    "trait": row.get("trait") or "",
                    "source": source,
                }
                edge_values.append((
                    edge_id, snp_key, "OBSERVED_IN", population_node,
                    _json(properties), "[]", None, None, 1, now, now,
                ))
            for i in range(0, len(node_values), 20000):
                con.executemany(
                    """INSERT OR IGNORE INTO graph_nodes (
                        node_id, node_type, attributes_json, first_seen_at, last_seen_at, hit_count
                    ) VALUES (?, ?, ?, ?, ?, 1)""",
                    node_values[i:i + 20000],
                )
            for i in range(0, len(edge_values), 20000):
                con.executemany(
                    """INSERT INTO graph_edges (
                        edge_id, source, predicate, target, properties_json,
                        evidence_ids_json, confidence, directness, version,
                        first_seen_at, last_seen_at, hit_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(edge_id) DO NOTHING""",
                    edge_values[i:i + 20000],
                )
            counts = {
                "species": species,
                "population": population,
                "source": source,
                "edges_inserted": len(edge_values),
            }
            self._log_evolution("variant_graph", "bulk_population_ingest", counts)
        return counts

    def bulk_ingest_variant_nodes(self, rows: list[dict[str, Any]]) -> int:
        """Insert variant nodes for the union of all population SNPs (dedup)."""
        if self.species is None:
            raise ValueError("bulk_ingest_variant_nodes requires a species-scoped builder")
        if not rows:
            return 0
        now = _now()
        node_values = []
        for row in rows:
            snp_key = str(row["snp_key"])
            parts = snp_key.split("|")
            if len(parts) != 6:
                continue
            node_values.append((
                snp_key, "variant",
                _json({
                    "species": self.species, "assembly": parts[1],
                    "chrom": parts[2], "pos": int(parts[3]),
                    "ref": parts[4], "alt": parts[5],
                }),
                now, now,
            ))
        with self._graph_conn() as con:
            con.execute("BEGIN IMMEDIATE")
            for i in range(0, len(node_values), 50_000):
                con.executemany(
                    """INSERT OR IGNORE INTO graph_nodes (
                        node_id, node_type, attributes_json, first_seen_at, last_seen_at, hit_count
                    ) VALUES (?, ?, ?, ?, ?, 1)""",
                    node_values[i:i + 50_000],
                )
        return len(node_values)

    def _build_population_comparisons(self, con: Any, *, snp_keys: list[str] | None = None) -> None:
        """Create DIFFERENT_FROM edges between populations sharing a SNP."""
        if snp_keys:
            placeholders = ",".join("?" for _ in snp_keys)
            rows = con.execute(
                f"""SELECT source, target, properties_json, evidence_ids_json
                    FROM graph_edges WHERE predicate='OBSERVED_IN' AND source IN ({placeholders})""",
                snp_keys,
            ).fetchall()
        else:
            rows = con.execute(
                """SELECT source, target, properties_json, evidence_ids_json
                   FROM graph_edges WHERE predicate='OBSERVED_IN'""",
            ).fetchall()
        per_snp: dict[str, dict[str, dict[str, Any]]] = {}
        for row in rows:
            try:
                props = json.loads(row["properties_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                props = {}
            population = row["target"]
            per_snp.setdefault(row["source"], {})[population] = {
                "props": props,
                "evidence_ids": json.loads(row["evidence_ids_json"] or "[]"),
            }
        for snp_key, populations in per_snp.items():
            keys = sorted(populations)
            for i, left in enumerate(keys):
                for right in keys[i + 1:]:
                    af_left = populations[left]["props"].get("AF")
                    af_right = populations[right]["props"].get("AF")
                    delta = None
                    if af_left is not None and af_right is not None:
                        delta = round(abs(float(af_left) - float(af_right)), 6)
                    fst = populations[left]["props"].get("fst") or populations[right]["props"].get("fst")
                    selection = populations[left]["props"].get("selection_score") or populations[right]["props"].get("selection_score")
                    if delta is None and fst is None and selection is None:
                        continue
                    trait = (
                        populations[left]["props"].get("trait")
                        or populations[right]["props"].get("trait")
                        or ""
                    )
                    self._upsert_edge(
                        con,
                        source=left,
                        predicate="DIFFERENT_FROM",
                        target=right,
                        properties={
                            "delta_AF": delta,
                            "Fst": fst,
                            "selection_score": selection,
                            "snp": snp_key,
                            "trait": trait,
                        },
                        evidence_ids=populations[left]["evidence_ids"] + populations[right]["evidence_ids"],
                        edge_key_suffix=snp_key,
                    )

    def _upsert_variant_from_key(self, con: Any, snp_key: str) -> None:
        parts = snp_key.split("|")
        if len(parts) != 6:
            return
        species, assembly, chrom, pos, ref, alt = parts
        self._upsert_node(con, node_id=snp_key, node_type="variant", attributes={
            "species": species, "assembly": assembly, "chrom": chrom,
            "pos": int(pos), "ref": ref, "alt": alt,
        })


    def ego_graph(
        self,
        snp_key: str,
        *,
        target_population: str | None = None,
        hops: int = 2,
        max_nodes: int = 250,
        trait: str | None = None,
        ld_max_neighbors: int | None = None,
        ld_max_distance_bp: int | None = None,
    ) -> dict[str, Any]:
        """Return the SNP-centered subgraph (BFS, population-filtered)."""
        if ld_max_neighbors is None:
            ld_max_neighbors = settings.graph_ld_max_neighbors
        if ld_max_distance_bp is None:
            ld_max_distance_bp = settings.graph_ld_max_distance_bp
        with self._graph_conn() as con:
            start = con.execute(
                "SELECT node_id FROM graph_nodes WHERE node_id=? AND node_type='variant'", (snp_key,),
            ).fetchone()
            if start is None:
                return {"snp": snp_key, "found": False, "nodes": [], "edges": [], "hops": 0}

            
            allowed_populations: set[str] | None = None
            if target_population:
                pop_rows = con.execute(
                    "SELECT node_id FROM graph_nodes WHERE node_type='population'",
                ).fetchall()
                allowed_populations = {
                    str(r["node_id"]) for r in pop_rows
                    if str(r["node_id"]).rsplit(":", 1)[-1] == target_population
                }
                if allowed_populations:
                    placeholders = ",".join("?" for _ in allowed_populations)
                    cmp_rows = con.execute(
                        f"""SELECT source, target FROM graph_edges
                            WHERE predicate='DIFFERENT_FROM'
                              AND (source IN ({placeholders}) OR target IN ({placeholders}))""",
                        list(allowed_populations) * 2,
                    ).fetchall()
                    for cmp_row in cmp_rows:
                        allowed_populations.add(str(cmp_row["source"]))
                        allowed_populations.add(str(cmp_row["target"]))
            frontier = [snp_key]
            visited = {snp_key}
            nodes: dict[str, dict[str, Any]] = {}
            edges: dict[str, dict[str, Any]] = {}

            def edge_allowed(row: Any) -> bool:
                try:
                    props = json.loads(row["properties_json"] or "{}")
                except (TypeError, ValueError, json.JSONDecodeError):
                    props = {}
                if target_population:
                    if row["predicate"] == "OBSERVED_IN":
                        return allowed_populations is not None and row["target"] in allowed_populations
                    if row["predicate"] == "LD_WITH":
                        return not props.get("population") or props["population"] == target_population
                    if row["predicate"] == "DIFFERENT_FROM":
                        if props.get("snp") != snp_key:
                            return False
                        return (
                            allowed_populations is not None
                            and row["source"] in allowed_populations
                            and row["target"] in allowed_populations
                        )
                if trait:

                    

                    if row["predicate"] in {"RELATED_TO", "ASSOCIATED_WITH"}:
                        return row["target"] == f"trait:{trait}"
                    if row["predicate"] in {
                        "SCORED_BY", "OBSERVED_IN", "LD_WITH",
                        "SUPPORTED_BY", "DIFFERENT_FROM",
                    }:
                        edge_trait = str(props.get("trait") or "")
                        if edge_trait and edge_trait != trait:
                            return False
                return True

            ld_counts: dict[str, int] = {}

            def ld_ok(row: Any, frontier: list[str] | None) -> bool:
                if row["predicate"] != "LD_WITH":
                    return True
                try:
                    props = json.loads(row["properties_json"] or "{}")
                except (TypeError, ValueError, json.JSONDecodeError):
                    props = {}
                if ld_max_distance_bp is not None:
                    distance = props.get("distance_bp")
                    if distance is not None:
                        try:
                            distance = int(distance)
                        except (TypeError, ValueError):
                            distance = None
                        if distance is not None and distance > ld_max_distance_bp:
                            return False
                if frontier and row["source"] in frontier:
                    expander = row["source"]
                elif frontier and row["target"] in frontier:
                    expander = row["target"]
                else:
                    expander = row["source"]
                if ld_counts.get(expander, 0) >= ld_max_neighbors:
                    return False
                ld_counts[expander] = ld_counts.get(expander, 0) + 1
                return True

            for _hop in range(max(1, hops)):
                next_frontier: list[str] = []
                placeholders = ",".join("?" for _ in frontier)
                rows = con.execute(
                    f"""SELECT * FROM graph_edges
                        WHERE source IN ({placeholders}) OR target IN ({placeholders})
                        ORDER BY hit_count DESC, last_seen_at DESC""",
                    [*frontier, *frontier],
                ).fetchall()
                for row in rows:
                    source, target = row["source"], row["target"]
                    if not edge_allowed(row):
                        continue
                    if not ld_ok(row, frontier):
                        continue
                    if source in visited and target in visited:
                        edges[row["edge_id"]] = _edge_dict(row)
                        continue
                    if len(nodes) + len(visited) >= max_nodes:
                        continue
                    edges[row["edge_id"]] = _edge_dict(row)
                    for node_id in (source, target):
                        if node_id not in visited:
                            visited.add(node_id)

                            

                            

                            if "|" in node_id or node_id.startswith("gene:"):
                                next_frontier.append(node_id)
                if not next_frontier:
                    break
                frontier = next_frontier

            
            visited_list = list(visited)
            for start in range(0, len(visited_list), 200):
                chunk = visited_list[start:start + 200]
                placeholders = ",".join("?" for _ in chunk)
                source_index = "INDEXED BY idx_graph_edges_source" if isinstance(con, sqlite3.Connection) else ""
                rows = con.execute(
                    f"""SELECT * FROM graph_edges {source_index}
                        WHERE source IN ({placeholders}) AND target IN ({placeholders})""",
                    [*chunk, *chunk],
                ).fetchall()
                for row in rows:
                    if edge_allowed(row) and ld_ok(row, None):
                        edges[row["edge_id"]] = _edge_dict(row)
            node_placeholders = ",".join("?" for _ in visited)
            node_rows = con.execute(
                f"SELECT * FROM graph_nodes WHERE node_id IN ({node_placeholders})", list(visited),
            ).fetchall()
            for node_row in node_rows:
                try:
                    attributes = json.loads(node_row["attributes_json"] or "{}")
                except (TypeError, ValueError, json.JSONDecodeError):
                    attributes = {}
                nodes[node_row["node_id"]] = {
                    "node_id": node_row["node_id"],
                    "node_type": node_row["node_type"],
                    "attributes": attributes,
                }
        return {
            "snp": snp_key,
            "found": True,
            "target_population": target_population,
            "hops": hops,
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    def shared_pathways(self, snp_keys: list[str]) -> list[dict[str, Any]]:
        """Return pathways shared by two or more query SNPs (semantic convergence)."""
        snp_keys = [k for k in snp_keys if k]
        if len(snp_keys) < 2:
            return []
        per_snp: dict[str, set[str]] = {snp: set() for snp in snp_keys}
        with self._graph_conn() as con:
            for snp_key in snp_keys:
                gene_rows = con.execute(
                    """SELECT target AS gene FROM graph_edges
                       WHERE source=? AND predicate IN ('LOCATED_IN','REGULATES')""",
                    (snp_key,),
                ).fetchall()
                genes = [str(r["gene"]) for r in gene_rows]
                if not genes:
                    continue
                placeholders = ",".join("?" for _ in genes)
                pathway_rows = con.execute(
                    f"""SELECT target AS pathway FROM graph_edges
                        WHERE source IN ({placeholders}) AND predicate='PARTICIPATES_IN'""",
                    genes,
                ).fetchall()
                per_snp[snp_key] = {str(r["pathway"]) for r in pathway_rows}
        shared: set[str] = set.intersection(*per_snp.values()) if per_snp else set()
        result: list[dict[str, Any]] = []
        for pathway in sorted(shared):
            genes: set[str] = set()
            snps_in: list[str] = []
            with self._graph_conn() as con:
                for snp_key in snp_keys:
                    if pathway not in per_snp.get(snp_key, set()):
                        continue
                    snps_in.append(snp_key)
                    gene_rows = con.execute(
                        """SELECT source AS gene FROM graph_edges
                           WHERE predicate='PARTICIPATES_IN' AND target=?""",
                        (pathway,),
                    ).fetchall()
                    genes.update(str(r["gene"]) for r in gene_rows)
            result.append({
                "pathway": pathway,
                "genes": sorted(genes),
                "snps": snps_in,
            })
        return result

    def shared_literature(self, snp_keys: list[str]) -> list[str]:
        """Evidence nodes (papers) shared by two or more query SNPs via genes."""
        snp_keys = [k for k in snp_keys if k]
        if len(snp_keys) < 2:
            return []
        per_snp: dict[str, set[str]] = {}
        with self._graph_conn() as con:
            for snp_key in snp_keys:
                gene_rows = con.execute(
                    """SELECT target AS gene FROM graph_edges
                       WHERE source=? AND predicate IN ('LOCATED_IN','REGULATES')""",
                    (snp_key,),
                ).fetchall()
                genes = [str(r["gene"]) for r in gene_rows]
                if not genes:
                    per_snp[snp_key] = set()
                    continue
                placeholders = ",".join("?" for _ in genes)
                evidence_rows = con.execute(
                    f"""SELECT DISTINCT target AS evidence FROM graph_edges
                        WHERE predicate='SUPPORTED_BY' AND source IN ({placeholders})""",
                    genes,
                ).fetchall()
                per_snp[snp_key] = {str(r["evidence"]) for r in evidence_rows}
        shared: set[str] = set.intersection(*per_snp.values()) if per_snp else set()
        return sorted(shared)

    def stats(self) -> dict[str, Any]:
        with self._graph_conn() as con:
            return {
                "nodes": int(con.execute("SELECT COUNT(*) AS n FROM graph_nodes").fetchone()["n"]),
                "edges": int(con.execute("SELECT COUNT(*) AS n FROM graph_edges").fetchone()["n"]),
                "node_types": {
                    str(r["node_type"]): int(r["n"])
                    for r in con.execute(
                        "SELECT node_type, COUNT(*) AS n FROM graph_nodes GROUP BY node_type"
                    ).fetchall()
                },
                "edge_predicates": {
                    str(r["predicate"]): int(r["n"])
                    for r in con.execute(
                        "SELECT predicate, COUNT(*) AS n FROM graph_edges GROUP BY predicate"
                    ).fetchall()
                },
            }


def _confidence(record: dict[str, Any]) -> float | None:
    if record.get("confidence") is not None:
        try:
            return float(record["confidence"])
        except (TypeError, ValueError):
            return None
    return _EVIDENCE_LEVEL_WEIGHT.get(str(record.get("evidence_level") or ""))


def _region_type_from_statement(statement: str) -> str:
    text = str(statement or "").lower()
    for token in ("cds", "exon", "intron", "promoter", "utr", "5' utr", "3' utr", "splice", "intergenic"):
        if token in text:
            return token
    return "gene"


def _model_name(record: dict[str, Any]) -> str:
    tool = str(record.get("tool_name") or "").lower()
    if "dpc" in tool:
        return "DPCformer"
    if "cadd" in tool:
        return "PlantVImpact"
    hint = str((record.get("scope") or {}).get("model_name") or "").lower()
    if "dpc" in hint:
        return "DPCformer"
    if "cadd" in hint:
        return "PlantVImpact"
    return str(record.get("tool_name") or hint or "model")


def _resolve_gene_nodes(
    con: Any,
    record: dict[str, Any],
    species: str,
    snp_genes: dict[str, list[str]],
) -> list[str]:
    subject = str(record["subject"] or "").strip()
    subject_gene = f"gene:{species}:{subject}"
    exists = con.execute(
        "SELECT 1 FROM graph_nodes WHERE node_id=? AND node_type='gene'", (subject_gene,),
    ).fetchone()
    if exists:
        return [subject_gene]
    return list(dict.fromkeys(snp_genes.get(record["snp_key"], [])))


def _edge_dict(row: Any) -> dict[str, Any]:
    try:
        properties = json.loads(row["properties_json"] or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        properties = {}
    try:
        evidence_ids = json.loads(row["evidence_ids_json"] or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        evidence_ids = []
    return {
        "edge_id": row["edge_id"],
        "source": row["source"],
        "predicate": row["predicate"],
        "target": row["target"],
        "properties": properties,
        "evidence_ids": evidence_ids,
        "confidence": row["confidence"],
        "directness": row["directness"],
        "version": row["version"] if "version" in row.keys() else None,
    }


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value != value:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value != value:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def ensure_graph(
    store: EvidenceStore, *, species: str | None = None, force: bool = False,
) -> dict[str, Any]:
    """Rebuild the graph when it is empty (or when explicitly forced)."""
    builder = VariantGraphBuilder(store, species=species)
    with builder._graph_conn() as con:
        count = int(con.execute("SELECT COUNT(*) AS n FROM graph_nodes").fetchone()["n"])
    if count and not force:
        return builder.stats()
    return builder.rebuild()
