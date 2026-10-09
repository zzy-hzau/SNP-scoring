"Graph-guided retrieval planning (Variant GraphRAG loop)."

from __future__ import annotations

import json
from typing import Any

from phenofuncrank_repro.interpretation.settings import settings


GENE_ANCHOR_TOOLS = ["gene_context_resolver", "gff_annotation_parser"]
REGULATORY_TOOLS = [
    "fimo_motif_scan",
    "rnafold_structure_prediction",
    "jaspar_motif_scan",
    "plantregmap_search",
    "plantregmap_genome_tfbs_overlap",
    "promoter_tfbs_density",
]
PROTEIN_TOOLS = [
    "interpro_scan",
    "pfam_domain_for_protein",
    "reactome_pathways_for_protein",
    "resolve_to_uniprot",
    "fetch_protein_sequence",
]
PATHWAY_TOOLS = [
    "interpro_scan",
    "reactome_pathways_for_protein",
    "pfam_domain_for_protein",
]
LITERATURE_TOOLS = ["multi_query_literature_search"]


_TYPE_TO_LAYER: dict[str, str] = {

    "model_effect": "variant_allele_effect",
    "variant_context": "variant_allele_effect",
    "functional_score": "variant_functional_score",
    "motif_scan": "variant_regulatory_effect",
    "tfbs_overlap": "variant_regulatory_effect",
    "tfbs_density": "variant_regulatory_effect",
    "regulatory_annotation": "variant_regulatory_effect",
    "rna_structure": "variant_regulatory_effect",
    "plantregmap_search": "variant_regulatory_effect",
    "protein_domain": "variant_protein_impact",
    "protein_structure": "variant_protein_impact",
    "protein_structure_quality": "variant_protein_impact",
    "cds_translation": "variant_protein_impact",
    "sequence_homology": "variant_protein_impact",
    "functional_annotation": "variant_protein_impact",

    "gene_context": "gene_function",
    "gene_link": "gene_function",
    "known_gene_proximity": "gene_function",
    "uniprot_mapping": "gene_function",
    "gene_alias": "gene_function",
    "gene_database": "gene_function",
    "pathway_annotation": "gene_pathway",
    "pathway_membership": "gene_pathway",
    "pathway_projection": "gene_pathway",
    "go_annotation": "gene_pathway",
    "kegg_pathway": "gene_pathway",
    "kegg_pathway_annotation": "gene_pathway",
    "kegg_pathway_membership": "gene_pathway",
    "literature": "gene_literature",

    "population_metric": "population_evidence",
    "ld_proxy": "variant_ld",
}

_FUNCTIONAL_VALIDATION_HINTS = (
    "mutant", "knockout", "overexpression", "crispr", "cas9", "tilling",
    "rna interference", "rnai", "allele-specific", "complementation",
    "transgenic", "rescue", "loss-of-function", "gain-of-function",
    "functional validation", "experimental validation", "protein-truncating",
)
_ASSOCIATION_HINTS = (
    "gwas", "qtl", "association", "co-expression", "correlation",
    "genome-wide", "mapping", "linkage", "meta-analysis", "population",
)

_LAYER_LABELS = {
    "variant_allele_effect": "variant allele effect (DPC)",
    "variant_functional_score": "variant functional score (PlantVImpact)",
    "variant_regulatory_effect": "variant regulatory effect (motif/TFBS/RNA)",
    "variant_protein_impact": "variant protein impact",
    "variant_ld": "variant LD proxies",
    "gene_function": "gene function/context",
    "gene_pathway": "gene pathway/GO",
    "gene_literature": "gene literature",
    "population_evidence": "population AF/selection",
    "trait_evidence": "trait/phenotype",
}


class GraphRAGPlanner:
    """Evidence-profile-based planner over the trusted variant graph."""

    def __init__(self, store: Any | None = None) -> None:
        if store is None:
            from phenofuncrank_repro.interpretation.evidence_store import EvidenceStore
            store = EvidenceStore()
        self.store = store


    def ego_graph(
        self, snp_key: str, *, hops: int = 2, target_population: str | None = None,
        trait: str | None = None,
    ) -> dict[str, Any]:
        if not snp_key:
            return {"found": False, "nodes": [], "edges": []}
        from phenofuncrank_repro.interpretation.variant_graph import VariantGraphBuilder
        species = str(snp_key.split("|", 1)[0]).strip().lower() if "|" in snp_key else None
        builder = VariantGraphBuilder(self.store, species=species)
        graph = builder.ego_graph(
            snp_key, hops=hops, target_population=target_population, trait=trait,
        )
        if graph.get("found"):
            return graph
        resolved = self._resolve_variant_key(snp_key)
        if resolved and resolved != snp_key:
            graph = builder.ego_graph(
                resolved, hops=hops, target_population=target_population, trait=trait,
            )
            if graph.get("found"):
                graph["resolved_snp_key"] = resolved
        return graph

    def _resolve_variant_key(self, snp_key: str) -> str | None:
        parts = str(snp_key).split("|")
        if len(parts) != 6:
            return None
        species, _assembly, chrom, pos, ref, alt = parts
        from phenofuncrank_repro.interpretation.variant_graph import normalize_chrom, normalize_species
        from phenofuncrank_repro.interpretation.evidence_normalizers import assembly_for_species
        wanted = (
            normalize_species(species), normalize_chrom(chrom),
            str(int(pos)), str(ref).upper(), str(alt).upper(),
        )
        canonical_assembly = assembly_for_species(normalize_species(species))
        exact_key: str | None = None
        with self.store.connect() as con:
            exact = con.execute(
                """SELECT canonical_key FROM snps
                   WHERE species=? AND chrom=? AND pos=? AND ref=? AND alt=?
                   LIMIT 1""",
                (wanted[0], wanted[1], int(pos), str(ref).upper(), str(alt).upper()),
            ).fetchone()
            if exact is not None:
                exact_key = str(exact["canonical_key"])

                expected_assembly = _assembly if _assembly and _assembly != "unknown" else canonical_assembly
                if str(exact_key).split("|")[1] == expected_assembly:
                    return exact_key

        
        chrom_aliases = {chrom, wanted[1], "chr" + wanted[1], "Chr" + wanted[1]}
        if wanted[1].isdigit():
            padded = f"{int(wanted[1]):02d}"
            chrom_aliases.update({padded, "chr" + padded, "Chr" + padded})
        assemblies = {_assembly, "unknown"}
        if not _assembly or _assembly == "unknown": assemblies.add(canonical_assembly)
        candidates = [snp_key]
        if exact_key and str(exact_key).split("|")[1] in assemblies: candidates.append(exact_key)
        candidates.extend("|".join((wanted[0], assembly, alias, wanted[2], wanted[3], wanted[4]))
                          for assembly in sorted(assemblies) for alias in sorted(chrom_aliases))
        candidates = list(dict.fromkeys(candidates))
        graph_con = (
            self.store.connect_graph(normalize_species(species))
            if species else self.store.connect()
        )
        with graph_con as con:
            marks = ",".join("?" for _ in candidates)
            rows = con.execute(
                f"SELECT node_id, attributes_json FROM graph_nodes WHERE node_id IN ({marks}) AND node_type='variant'",
                candidates,
            ).fetchall()
        by_id = {row["node_id"]: row for row in rows}
        for node_id in candidates:
            row = by_id.get(node_id)
            if row is None: continue
            try:
                attrs = json.loads(row["attributes_json"] or "{}")
            except Exception:
                continue
            current = (
                normalize_species(str(attrs.get("species") or "")),
                normalize_chrom(str(attrs.get("chrom") or "")),
                str(int(attrs.get("pos") or 0)),
                str(attrs.get("ref") or "").upper(),
                str(attrs.get("alt") or "").upper(),
            )
            if current == wanted:
                return str(row["node_id"])
        return exact_key if exact_key and str(exact_key).split("|")[1] in assemblies else None

    def plan(
        self,
        snp_key: str,
        *,
        region: str = "",
        consequence: str = "",
        gene_name: str = "",
        trait: str = "",
        hops: int = 2,
    ) -> dict[str, Any]:
        """Return evidence profile, layer gaps, priorities, and LLM context."""
        graph = self.ego_graph(snp_key, hops=hops, trait=trait or None)
        found = bool(graph.get("found"))
        resolved = str(graph.get("resolved_snp_key") or snp_key) if found else snp_key
        profile = self._evidence_profile(resolved, trait=trait) if found else {}
        if not found:
            profile = self._empty_profile()
        layer_gaps = self._layer_gaps(profile, region=region, consequence=consequence)
        decision = self._decision(profile, layer_gaps)
        covered = self._covered_layers(profile)
        skip = self._skip_tools(covered, layer_gaps)
        prioritized = self._prioritized_tools(layer_gaps)
        context = self.context_text(graph, profile=profile, decision=decision)

        gaps = [
            {
                "gap_id": f"graph_{resolved}_{gap['layer']}",
                "candidate_id": resolved,
                "missing_link": gap["missing_link"],
                "priority": gap["priority"],
                "recommended_evidence_types": gap["recommended_evidence_types"],
                "suggested_tools": gap["suggested_tools"],
                "graph_derived": True,
            }
            for gap in layer_gaps
        ]
        return {
            "snp_key": snp_key,
            "resolved_snp_key": resolved if found else "",
            "found": found,
            "evidence_profile": profile,
            "evidence_layer_gaps": layer_gaps,
            "decision": decision,
            "gaps": gaps,
            "covered_layers": covered,
            "prioritized_tools": prioritized,
            "skip_tools": skip,
            "context": context,
            "edge_count": len(graph.get("edges") or []),
            "node_count": len(graph.get("nodes") or []),
        }

    @staticmethod
    def build_evidence_state(
        evidence_items: list[dict[str, Any]] | list[Any],
        *,
        candidate_ref: dict[str, Any] | None = None,
        species: str = "",
        trait: str = "",
    ) -> dict[str, Any]:
        "Build the A1-owned runtime EvidenceState from admitted A2 items."
        from phenofuncrank_repro.interpretation.evidence_contract import build_evidence_state

        return build_evidence_state(
            evidence_items,
            candidate_ref=candidate_ref,
            species=species,
            trait=trait,
        )


    @staticmethod
    def _empty_profile() -> dict[str, Any]:
        layers = {
            "variant_allele_effect", "variant_functional_score",
            "variant_regulatory_effect", "variant_protein_impact", "variant_ld",
            "gene_function", "gene_pathway", "gene_literature",
            "population_evidence", "trait_evidence",
        }
        return {
            layer: {"count": 0, "assessment": "missing"}
            for layer in sorted(layers)
        }

    def _evidence_profile(self, snp_key: str, *, trait: str = "") -> dict[str, Any]:
        profile: dict[str, Any] = {}
        for layer in sorted(_LAYER_LABELS):
            profile[layer] = {"count": 0, "assessment": "missing"}
        with self.store.connect() as con:
            rows = con.execute(
                """SELECT evidence_records.*, tool_runs.tool_name
                   FROM evidence_records
                   LEFT JOIN tool_runs ON tool_runs.id=evidence_records.tool_run_id
                   WHERE evidence_records.snp_key=? ORDER BY evidence_records.created_at""",
                (snp_key,),
            ).fetchall()
            if not rows:

                
                parts = str(snp_key).split("|")
                if len(parts) == 6:
                    try:
                        rows = con.execute(
                            """SELECT evidence_records.*, tool_runs.tool_name
                               FROM evidence_records
                               JOIN snps ON snps.canonical_key=evidence_records.snp_key
                               LEFT JOIN tool_runs ON tool_runs.id=evidence_records.tool_run_id
                               WHERE snps.species=? AND snps.chrom=?
                                 AND snps.pos=? AND snps.ref=? AND snps.alt=?
                               ORDER BY evidence_records.created_at""",
                            (parts[0], parts[2], int(parts[3]), parts[4], parts[5]),
                        ).fetchall()
                    except (TypeError, ValueError):
                        rows = []
        values_by_layer: dict[str, list[float]] = {}
        for row in rows:
            record = dict(row)
            evidence_type = str(record.get("evidence_type") or "")
            scope = {}
            try:
                scope = json.loads(record.get("scope_json") or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                scope = {}
            record_trait = str(scope.get("trait") or "")
            if trait and record_trait and record_trait != trait:

                
                continue
            layer = _TYPE_TO_LAYER.get(evidence_type)
            if layer is None:
                continue
            entry = profile.setdefault(layer, {"count": 0, "assessment": "missing"})
            entry["count"] = int(entry.get("count", 0)) + 1
            entry.setdefault("evidence_types", {})
            type_counts = entry["evidence_types"]
            type_counts[evidence_type] = type_counts.get(evidence_type, 0) + 1
            try:
                value = json.loads(record.get("value_json") or "null")
                value = float(value) if value is not None else None
            except (TypeError, ValueError, json.JSONDecodeError):
                value = None
            if value is not None:
                values_by_layer.setdefault(layer, []).append(value)
            if evidence_type == "literature":
                self._classify_literature(entry, record, species=str(record.get("snp_key") or "").split("|")[0])
            text = " ".join([
                str(record.get("object") or ""),
                str(scope.get("trait") or ""),
                str(record.get("direction") or ""),
            ])
            if trait and trait.lower() in text.lower():
                trait_entry = profile.setdefault("trait_evidence", {"count": 0, "assessment": "missing"})
                trait_entry["count"] = int(trait_entry.get("count", 0)) + 1
        self._assess_layers(profile, values_by_layer)
        return profile

    @staticmethod
    def _classify_literature(entry: dict[str, Any], record: dict[str, Any], *, species: str) -> None:
        payload = {}
        try:
            payload = json.loads(record.get("payload_json") or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        scope = {}
        try:
            scope = json.loads(record.get("scope_json") or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            scope = {}
        text = " ".join([
            str(record.get("object") or ""),
            str(payload.get("title") or ""),
            str(payload.get("abstract_snippet") or ""),
            str(payload.get("statement") or ""),
        ]).lower()
        categories = entry.setdefault("categories", {
            "functional_validation": 0, "gene_trait_association": 0,
            "variant_specific": 0, "species_specific": 0, "curated": 0,
        })
        source = str(scope.get("source") or payload.get("provider") or "").lower()
        if "curated" in source:
            categories["curated"] = categories.get("curated", 0) + 1
        if any(hint in text for hint in _FUNCTIONAL_VALIDATION_HINTS):
            categories["functional_validation"] += 1
        if any(hint in text for hint in _ASSOCIATION_HINTS):
            categories["gene_trait_association"] += 1
        snp_parts = str(record.get("snp_key") or "").split("|")
        if len(snp_parts) == 6 and (
            f"{snp_parts[2]}:{snp_parts[3]}" in text
            or f"{snp_parts[3]}" in text and f"{snp_parts[4]}>{snp_parts[5]}" in text
            or f"{snp_parts[4]}>{snp_parts[5]}" in text
        ):
            categories["variant_specific"] += 1
        if species and species.lower() in text:
            categories["species_specific"] += 1

    @staticmethod
    def _assess_layers(profile: dict[str, Any], values_by_layer: dict[str, list[float]]) -> None:
        for layer, entry in profile.items():
            count = int(entry.get("count", 0))
            values = values_by_layer.get(layer) or []
            if layer == "variant_allele_effect":
                entry["assessment"] = "strong" if count and max(abs(v) for v in values) >= 0.01 else (
                    "medium" if count else "missing")
            elif layer == "variant_functional_score":
                phred = max(values) if values else 0.0
                entry["assessment"] = "strong" if phred >= 20 else (
                    "medium" if phred >= 10 else ("weak" if count else "missing"))
            elif layer == "gene_literature":
                categories = entry.get("categories", {})
                entry["assessment"] = (
                    "sufficient"
                    if (
                        (
                            categories.get("functional_validation", 0) >= 1
                            or categories.get("variant_specific", 0) >= 1
                        )
                        and (
                            categories.get("curated", 0) >= 1
                            or entry.get("count", 0) >= 2
                        )
                    )
                    else ("insufficient" if count else "missing")
                )
            elif layer in {
                "variant_regulatory_effect", "variant_protein_impact",
                "gene_function", "gene_pathway", "population_evidence",
                "trait_evidence", "variant_ld",
            }:
                entry["assessment"] = (
                    "strong" if count >= 3 else ("medium" if count >= 1 else "missing"))
            else:
                entry["assessment"] = "strong" if count >= 1 else "missing"


    @staticmethod
    def _layer_gaps(profile: dict[str, Any], *, region: str, consequence: str) -> list[dict[str, Any]]:
        gaps: list[dict[str, Any]] = []
        gene_function = profile.get("gene_function", {}).get("assessment", "missing")
        if gene_function in {"missing"}:
            gaps.append({
                "layer": "gene_function",
                "assessment": "missing",
                "missing_link": "no gene anchor/function evidence",
                "priority": "critical",
                "recommended_evidence_types": ["gene_context", "gene_link"],
                "suggested_tools": list(GENE_ANCHOR_TOOLS),
            })
        literature = profile.get("gene_literature", {})
        if literature.get("assessment") in {"missing", "insufficient"}:
            categories = literature.get("categories", {})
            gaps.append({
                "layer": "gene_literature",
                "assessment": literature.get("assessment"),
                "missing_link": (
                    f"gene literature {literature.get('count', 0)} paper(s): "
                    f"{categories.get('functional_validation', 0)} functional validation, "
                    f"{categories.get('gene_trait_association', 0)} trait association, "
                    f"{categories.get('variant_specific', 0)} variant-specific"
                ),
                "priority": "important",
                "recommended_evidence_types": ["literature"],
                "suggested_tools": list(LITERATURE_TOOLS),
            })
        region_l = (region or "").lower()
        cons_l = (consequence or "").lower()
        coding = region_l == "cds" or any(
            hint in cons_l for hint in ("missense", "nonsense", "frameshift", "stop", "coding")
        )
        regulatory = profile.get("variant_regulatory_effect", {}).get("assessment", "missing")
        if not coding and regulatory in {"missing", "weak"}:
            gaps.append({
                "layer": "variant_regulatory_effect",
                "assessment": regulatory,
                "missing_link": "variant→gene regulatory mechanism (motif/TFBS/RNA) missing",
                "priority": "important",
                "recommended_evidence_types": ["motif_scan", "tfbs_overlap", "rna_structure"],
                "suggested_tools": list(REGULATORY_TOOLS),
            })
        protein = profile.get("variant_protein_impact", {}).get("assessment", "missing")
        if coding and protein in {"missing", "weak"}:
            gaps.append({
                "layer": "variant_protein_impact",
                "assessment": protein,
                "missing_link": "variant protein-impact evidence (domain/structure/homology) missing",
                "priority": "important",
                "recommended_evidence_types": ["protein_domain", "protein_structure", "sequence_homology"],
                "suggested_tools": list(PROTEIN_TOOLS),
            })
        pathway = profile.get("gene_pathway", {}).get("assessment", "missing")
        if pathway == "missing":
            gaps.append({
                "layer": "gene_pathway",
                "assessment": "missing",
                "missing_link": "gene pathway/GO membership missing",
                "priority": "important",
                "recommended_evidence_types": ["pathway_annotation", "go_annotation"],
                "suggested_tools": list(PATHWAY_TOOLS),
            })
        population = profile.get("population_evidence", {}).get("assessment", "missing")
        if population == "missing":
            gaps.append({
                "layer": "population_evidence",
                "assessment": "missing",
                "missing_link": "population AF/selection evidence missing (Model Agent layer)",
                "priority": "optional",
                "recommended_evidence_types": ["population_metric"],
                "suggested_tools": [],
            })
        trait_layer = profile.get("trait_evidence", {}).get("assessment", "missing")
        if trait_layer == "missing":
            gaps.append({
                "layer": "trait_evidence",
                "assessment": "missing",
                "missing_link": "trait/phenotype link evidence missing",
                "priority": "optional",
                "recommended_evidence_types": ["literature", "pathway_annotation"],
                "suggested_tools": [],
            })
        return gaps

    @staticmethod
    def _decision(profile: dict[str, Any], layer_gaps: list[dict[str, Any]]) -> str:
        if not layer_gaps:
            return "evidence_layers_sufficient"
        critical = [g for g in layer_gaps if g["priority"] == "critical"]
        if critical:
            return "need_gene_anchor"
        layers = {g["layer"] for g in layer_gaps}
        gene_layers_ok = (
            profile.get("gene_function", {}).get("assessment") in {"strong", "medium"}
            and profile.get("gene_literature", {}).get("assessment") == "sufficient"
        )
        if gene_layers_ok and ({"variant_regulatory_effect", "variant_protein_impact"} & layers):
            return "need_variant_gene_mechanism"
        if "gene_literature" in layers:
            return "need_gene_literature_validation"
        return "retrieve_missing_layers"

    @staticmethod
    def _covered_layers(profile: dict[str, Any]) -> dict[str, Any]:
        gene_function = profile.get("gene_function", {}).get("assessment", "missing")
        literature = profile.get("gene_literature", {}).get("assessment", "missing")
        pathway = profile.get("gene_pathway", {}).get("assessment", "missing")
        regulatory = profile.get("variant_regulatory_effect", {}).get("assessment", "missing")
        return {
            "gene_anchored": gene_function in {"strong", "medium"},
            "literature_edge_count": int(profile.get("gene_literature", {}).get("count", 0)),
            "literature_curated_count": int(
                (profile.get("gene_literature", {}).get("categories") or {}).get("curated", 0)
            ),
            "literature_covered": literature == "sufficient",
            "literature_sufficient": literature == "sufficient",
            "regulatory_covered": regulatory in {"strong", "medium"},
            "pathway_covered": pathway in {"strong", "medium"},
            "pathway_edge_count": int(profile.get("gene_pathway", {}).get("count", 0)),
            "ld_proxy_count": int(profile.get("variant_ld", {}).get("count", 0)),
            "model_score_count": int(
                profile.get("variant_allele_effect", {}).get("count", 0)
                + profile.get("variant_functional_score", {}).get("count", 0)
            ),
            "variant_in_graph": any(
                profile.get(layer, {}).get("count", 0) > 0
                for layer in (
                    "variant_allele_effect", "variant_functional_score",
                    "variant_regulatory_effect", "variant_protein_impact",
                    "variant_ld", "gene_function",
                )
            ),
        }

    @staticmethod
    def _skip_tools(covered: dict[str, Any], layer_gaps: list[dict[str, Any]]) -> list[str]:
        """Tools redundant because their layer is already sufficient."""
        gap_layers = {gap["layer"] for gap in layer_gaps}
        skip: list[str] = []
        if covered.get("literature_covered") and "gene_literature" not in gap_layers:
            skip.extend(LITERATURE_TOOLS)
        if covered.get("regulatory_covered") and "variant_regulatory_effect" not in gap_layers:
            skip.extend(REGULATORY_TOOLS)
        if covered.get("pathway_covered") and "gene_pathway" not in gap_layers:
            skip.extend(PATHWAY_TOOLS)
        return list(dict.fromkeys(skip))

    @staticmethod
    def _prioritized_tools(layer_gaps: list[dict[str, Any]]) -> list[str]:
        priority_rank = {"critical": 0, "important": 1, "optional": 2}
        ordered: list[str] = []
        for gap in sorted(layer_gaps, key=lambda g: priority_rank.get(str(g.get("priority")), 9)):
            for tool in gap.get("suggested_tools") or []:
                if tool and tool not in ordered:
                    ordered.append(tool)
        return ordered


    @staticmethod
    def context_text(graph: dict[str, Any], *, profile: dict[str, Any] | None = None, decision: str = "") -> str:
        if not graph.get("found"):
            return ""
        snp_key = str(graph.get("snp") or "")
        lines = [f"[graph] {snp_key}"]
        if profile:
            lines.append(f"[layers] decision={decision or 'unknown'}")
            for layer in sorted(_LAYER_LABELS):
                entry = profile.get(layer) or {}
                if not entry.get("count"):
                    continue
                assessment = entry.get("assessment", "missing")
                parts = [f"{layer}={assessment}"]
                if layer == "gene_literature":
                    categories = entry.get("categories", {})
                    parts.append(
                        f"validation={categories.get('functional_validation', 0)} "
                        f"assoc={categories.get('gene_trait_association', 0)} "
                        f"variant_specific={categories.get('variant_specific', 0)} "
                        f"species={categories.get('species_specific', 0)} "
                        f"curated={categories.get('curated', 0)}"
                    )
                lines.append("  " + " ".join(parts))
        for edge in (graph.get("edges") or [])[:40]:
            source = str(edge.get("source") or "")
            predicate = str(edge.get("predicate") or "")
            target = str(edge.get("target") or "")
            props = edge.get("properties") or {}
            parts = [f"{source} -{predicate}-> {target}"]
            for key in ("r2", "AF", "score", "distance_bp", "region_type", "source"):
                if key in props and props[key] is not None:
                    parts.append(f"{key}={props[key]}")
            lines.append("  " + " ".join(parts))
        remaining = max(0, len(graph.get("edges") or []) - 40)
        if remaining:
            lines.append(f"  ... {remaining} more edge(s)")
        return "\n".join(lines)
