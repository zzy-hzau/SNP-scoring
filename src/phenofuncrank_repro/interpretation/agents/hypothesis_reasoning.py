"Hypothesis Reasoning Agent (A3) — evidence chain construction."

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from phenofuncrank_repro.interpretation.logging_utils import get_logger
from phenofuncrank_repro.interpretation.llm_utils import complete_json
from phenofuncrank_repro.interpretation.schemas.messages import MessageEnvelope, ArtifactRef
from phenofuncrank_repro.interpretation.schemas.evidence_graph import AtomicClaim, CandidateConclusion, EvidenceNode, EvidenceEdge, IntegrationResult
from phenofuncrank_repro.interpretation.schemas.evidence_state import MISSING_LINK_RELATIONS, canonical_missing_links
from phenofuncrank_repro.interpretation.settings import settings

logger = get_logger(__name__)


_STATE_LEVEL_TO_EVIDENCE = {"L1": "A", "L2": "B", "L3": "C", "L4": "D"}
_RELATION_TO_LEVEL = {
    "variant_functional_effect": "L1", "variant_trait": "L1",
    "variant_expression": "L1", "variant_mechanism": "L1",
    "variant_protein_function": "L1", "gene_trait": "L2",
    "gene_mechanism": "L3",
}


def _edge_text(edge: dict[str, Any], variant_id: str, trait: str, gene: str) -> str:
    """Render one supported relation without adding an untracked detail."""
    relation = str(edge.get("relation") or "")
    source = str(edge.get("source") or gene or variant_id)
    target = str(edge.get("target") or trait or "the trait")
    return {
        "variant_functional_effect": f"Variant {source} has a supported variant-level functional relation to {target}.",
        "variant_trait": f"Variant {source} is directly linked to {target} in the supplied evidence.",
        "variant_expression": f"Variant {source} has a supported variant-linked expression relation.",
        "variant_mechanism": f"Variant {source} has a supported variant-linked molecular relation to {target}.",
        "variant_protein_function": f"Variant {source} has a supported variant-linked protein-function relation.",
        "gene_trait": f"Gene {source} has supported evidence for an effect on {target}.",
        "gene_mechanism": f"Gene {source} has supported evidence for the molecular/pathway relation described in the Evidence Pack.",
    }.get(relation, f"Supported relation: {source} → {target}.")


def _state_summary(state: dict[str, Any], variant_id: str, trait: str) -> str:
    facts = state.get("supported_edges") or []
    if facts:
        return " ".join(_edge_text(edge, variant_id, trait, state.get("gene_id", "")) for edge in facts[:3])
    return "No evidence-supported biological relation is established for this candidate."


def _state_missing_links(state: dict[str, Any]) -> list[str]:
    return canonical_missing_links(
        list(state.get("missing_relations") or [])
        + [edge.get("relation") for edge in (state.get("missing_edges") or [])]
    )


def _state_claim_slots(
    state: dict[str, Any], variant_id: str, trait: str,
) -> tuple[list[AtomicClaim], list[AtomicClaim]]:
    """Create deterministic supported-fact and hypothesis slots from state.

    The slots are intentionally created before any LLM wording pass.  This
    makes the evidence boundary observable and prevents a prose rewrite from
    changing what is scientifically supported.
    """
    level = str(state.get("max_supported_level") or "L4")
    evidence_level = level if level in _STATE_LEVEL_TO_EVIDENCE else "L4"
    supported: list[AtomicClaim] = []
    for idx, edge in enumerate(state.get("supported_edges") or [], 1):
        relation = str(edge.get("relation") or "")
        if not relation:
            continue
        supported.append(AtomicClaim(
            claim_id=f"{variant_id}:fact:{relation}:{idx}",
            relation=relation,
            text=_edge_text(edge, variant_id, trait, str(state.get("gene_id") or "")),
            supporting_evidence_ids=[str(x) for x in edge.get("evidence_ids", []) if str(x)],
            status="supported",
            evidence_level=_RELATION_TO_LEVEL.get(relation, evidence_level),
            causal_strength="gene_level" if relation == "gene_trait" else "none",
            gate={"decision": "PASS", "source": "EvidenceState"},
        ))

    hypotheses: list[AtomicClaim] = []
    supported_relations = {str(edge.get("relation")) for edge in (state.get("supported_edges") or [])}
    missing = _state_missing_links(state)

    
    if "variant_trait" not in supported_relations and (
        "gene_trait" in supported_relations or "gene_mechanism" in supported_relations
    ):
        refs = sorted({
            str(eid)
            for edge in (state.get("supported_edges") or [])
            if edge.get("relation") in {"gene_trait", "gene_mechanism"}
            for eid in edge.get("evidence_ids", [])
            if str(eid)
        })
        hypotheses.append(AtomicClaim(
            claim_id=f"{variant_id}:hypothesis:variant_trait",
            relation="variant_trait",
            text=f"The candidate variant may influence {trait} through the supported gene-level evidence; the exact variant-to-trait effect remains unresolved.",
            supporting_evidence_ids=refs,
            status="hypothesis_only",
            evidence_level="L1",
            causal_strength="mechanistic_hypothesis",
            missing_links=[link for link in missing if link == "variant_functional_effect"],
            gate={"decision": "BLOCK", "support_status": "insufficient", "source": "EvidenceState"},
        ))
    elif "variant_functional_effect" in supported_relations and "variant_trait" not in supported_relations:
        hypotheses.append(AtomicClaim(
            claim_id=f"{variant_id}:hypothesis:variant_trait",
            relation="variant_trait",
            text=f"The supported variant-level functional relation may be relevant to {trait}; a direct variant-to-trait conclusion remains unresolved.",
            supporting_evidence_ids=sorted({
                str(eid) for edge in (state.get("supported_edges") or [])
                if edge.get("relation") == "variant_functional_effect"
                for eid in edge.get("evidence_ids", []) if str(eid)
            }),
            status="hypothesis_only",
            evidence_level="L1",
            causal_strength="mechanistic_hypothesis",
            missing_links=[link for link in missing if link in {"gene_trait", "variant_trait"}],
            gate={"decision": "BLOCK", "support_status": "insufficient", "source": "EvidenceState"},
        ))
    return supported, hypotheses


CHAIN_TEMPLATE = [
    ("variant",            "variant",           "variant_to_gene"),
    ("gene",               "gene_regulatory",   "gene_to_molecular"),
    ("molecular_impact",   "molecular",         "molecular_to_pathway"),
    ("pathway",            "pathway",           "pathway_to_trait"),
    ("trait",              "trait",             "trait_to_breeding"),
    ("breeding_action",    "breeding",          None),
]


def _build_evidence_chain(
    variant_id: str,
    variant_info: dict[str, Any],
    state_evidence: list[dict[str, Any]],
    state_edges: list[dict[str, Any]],
    model_info: dict[str, Any],
    species: str,
    trait: str,
    target_direction: str | None,
) -> tuple[list[EvidenceNode], list[EvidenceEdge], dict[str, Any]]:
    """Build an evidence chain for a single candidate variant.

    Returns (nodes, edges, conflicts_dict).
    """
    nodes: list[EvidenceNode] = []
    edges: list[EvidenceEdge] = []
    conflicts: dict[str, list[str]] = {}

    def _ids(items: list[dict[str, Any]]) -> list[str]:
        """Non-empty evidence IDs referenced by the supplied evidence items."""
        return [
            str(ev.get("evidence_id", "")).strip()
            for ev in items
            if str(ev.get("evidence_id", "")).strip()
        ]


    chrom = str(variant_info.get("chrom", ""))
    pos = int(variant_info.get("pos", 0))
    dpc_effect = variant_info.get("dpc_additive_effect_mean")
    cad_score = variant_info.get("cadd_phred")
    if cad_score is None:
        cad_score = variant_info.get("cadd_score")
    is_core = variant_info.get("is_core", variant_info.get("level1_core", False))

    var_label = f"SNP {variant_id} ({chrom}:{pos})"
    var_attrs = {
        "chrom": chrom,
        "pos": pos,
        "dpc_additive_effect": float(dpc_effect) if dpc_effect is not None else None,
        "cadd_score": float(cad_score) if cad_score is not None else None,
        "is_core": bool(is_core),
        "species": species,
    }
    nodes.append(EvidenceNode(
        node_id=f"var_{short_id(variant_id)}",
        node_type="variant",
        label=var_label,
        attributes=var_attrs,
    ))


    gene_ids: set[str] = set()
    gene_names: list[str] = []
    region_type = "intergenic"
    gene_distance: int | None = None


    

    
    for edge in state_edges:
        if str(edge.get("relation") or "").startswith("gene_"):
            source = str(edge.get("source") or "").strip()
            if source and source not in gene_ids and source != variant_id:
                gene_ids.add(source)
                gene_names.append(source)
    for ev in state_evidence:
        if str(ev.get("subject_type") or "") in {"gene", "protein", "pathway"}:
            subject = str(ev.get("subject_id") or "").strip()
            if subject and subject not in gene_ids:
                gene_ids.add(subject)
                gene_names.append(subject)

    if gene_names:
        gene_label = f"Gene(s): {', '.join(gene_names[:3])}"
    else:
        gene_label = f"Region {chrom}:{pos} (no annotated gene within 500kb)"

    gene_attrs = {
        "gene_ids": list(gene_ids),
        "region_type": region_type,
        "distance_bp": gene_distance,
    }
    gene_node_id = f"gene_{short_id(variant_id)}"
    nodes.append(EvidenceNode(
        node_id=gene_node_id,
        node_type="gene" if region_type == "coding_or_gene_body" else "regulatory_element",
        label=gene_label,
        attributes=gene_attrs,
    ))
    var2gene_ids = _ids(state_evidence)
    if var2gene_ids:
        edges.append(EvidenceEdge(
        edge_id=f"e_var2gene_{short_id(variant_id)}",
        source=nodes[0].node_id,
        target=gene_node_id,
        relation="located_in" if region_type == "coding_or_gene_body" else "near",
        evidence_ids=var2gene_ids,
        confidence=None,
        directness="direct" if region_type == "coding_or_gene_body" else "computational_prediction",
        ))


    has_cadd = cad_score is not None and float(cad_score) > 0
    has_evidence = len(state_evidence) > 0
    mol_label = "Molecular impact"
    mol_level = "unknown"

    if has_cadd:
        mol_level = "model_prediction"
        mol_label = f"PlantVImpact probability ({float(cad_score):.4f})"

    mol_node_id = f"mol_{short_id(variant_id)}"
    nodes.append(EvidenceNode(
        node_id=mol_node_id,
        node_type="protein" if region_type == "coding_or_gene_body" else "regulatory_element",
        label=mol_label,
        attributes={
            "cadd_score": float(cad_score) if has_cadd else None,
            "molecular_evidence_level": mol_level,
            "evidence_count": len(state_evidence),
        },
    ))
    gene2mol_ids = _ids(state_evidence)
    if gene2mol_ids:
        edges.append(EvidenceEdge(
        edge_id=f"e_gene2mol_{short_id(variant_id)}",
        source=gene_node_id,
        target=mol_node_id,
        relation="has_molecular_impact",
        evidence_ids=gene2mol_ids,
        confidence=None,
        directness="computational_prediction",
        ))


    pathway_label = f"Pathway: {trait}-associated" if state_evidence else "Pathway: unknown"
    pathways: set[str] = set()
    for ev in state_evidence:
        statement = str(ev.get("statement") or "").strip()
        if statement:
            pathways.add(statement[:80])

    pathway_node_id = f"path_{short_id(variant_id)}"
    nodes.append(EvidenceNode(
        node_id=pathway_node_id,
        node_type="pathway",
        label=f"Pathway(s): {', '.join(sorted(pathways)[:3])}" if pathways else pathway_label,
        attributes={"pathway_hits": len(pathways), "pathway_sources": sorted(pathways)[:5]},
    ))
    mol2path_ids = _ids(state_evidence)
    if mol2path_ids:
        edges.append(EvidenceEdge(
        edge_id=f"e_mol2path_{short_id(variant_id)}",
        source=mol_node_id,
        target=pathway_node_id,
        relation="involved_in",
        evidence_ids=mol2path_ids,
        confidence=None,
        directness="same_species_indirect" if has_evidence else "computational_prediction",
        ))


    direction_note = ""
    if target_direction:
        direction_note = f" (target: {target_direction})"
    trait_node_id = f"trait_{short_id(variant_id)}"
    nodes.append(EvidenceNode(
        node_id=trait_node_id,
        node_type="trait",
        label=f"Trait: {trait}{direction_note}",
        attributes={"trait": trait, "target_direction": target_direction},
    ))
    path2trait_ids = _ids(state_evidence)
    if path2trait_ids:
        edges.append(EvidenceEdge(
        edge_id=f"e_path2trait_{short_id(variant_id)}",
        source=pathway_node_id,
        target=trait_node_id,
        relation="contributes_to",
        evidence_ids=path2trait_ids,
        confidence=None,
        directness="computational_prediction",
        ))


    breed_node_id = f"breed_{short_id(variant_id)}"
    nodes.append(EvidenceNode(
        node_id=breed_node_id,
        node_type="breeding_action",
        label=f"Breeding candidate: {variant_id}",
        attributes={"candidate_type": "core" if is_core else "proxy"},
    ))
    trait2breed_ids = _ids(state_evidence)
    if trait2breed_ids:
        edges.append(EvidenceEdge(
        edge_id=f"e_trait2breed_{short_id(variant_id)}",
        source=trait_node_id,
        target=breed_node_id,
        relation="recommended_for",
        evidence_ids=trait2breed_ids,
        confidence=None,
        directness="computational_prediction",
        caveats=["Requires experimental validation before breeding deployment"],
        ))

    return nodes, edges, conflicts


class HypothesisReasoningAgent:
    """Build SNP-to-trait evidence chains from Model and Knowledge results.

    Responsibilities:
      - Load candidate variants from Model Agent output
      - Match evidence items from Knowledge Agent per candidate
      - Construct per-candidate evidence chains (variant → gene → molecular → pathway → trait)
      - Detect conflicts between model scores and knowledge evidence
      - Assign evidence levels and control causal language
      - Produce structured CandidateConclusions and global limitations
    """

    def __init__(self, work_dir: Path | None = None):
        self.work_dir = work_dir or (settings.root / "integration_runs")
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def invoke(self, command: MessageEnvelope) -> IntegrationResult:
        """Integrate the A1-owned EvidenceState into per-candidate conclusions.

        A3 deliberately has no artifact reader and no raw-retrieval fallback.
        Candidate identity, deterministic scores, annotation, and admitted
        literature are all projected into EvidenceState by A1 before this
        method is called.
        """
        payload = command.payload
        species = str(payload.get("species", "")).strip()
        trait = str(payload.get("trait", "")).strip()
        target_direction = str(payload.get("target_direction", "")).strip() or None
        evidence_state_bundle = payload.get("evidence_state") or {}
        if not isinstance(evidence_state_bundle, dict):
            return IntegrationResult(
                status="blocked",
                global_limitations=["A3 requires the A1-owned EvidenceState bundle"],
            )
        state_candidates = evidence_state_bundle.get("candidates", {})
        if not isinstance(state_candidates, dict) or not state_candidates:
            return IntegrationResult(
                status="blocked",
                global_limitations=["A3 received no candidate-scoped EvidenceState"],
            )

        run_id = command.run_id or uuid.uuid4().hex[:8]

        candidate_rows: list[dict[str, Any]] = []
        for variant_id, candidate_state in state_candidates.items():
            if not isinstance(candidate_state, dict):
                continue
            context = dict(candidate_state.get("variant_context") or {})
            context.setdefault("species", species)
            context.setdefault("trait", trait)
            context.setdefault("gene_id", candidate_state.get("gene_id", ""))
            context.setdefault("variant_key", candidate_state.get("variant_key", ""))
            candidate_rows.append({"user_snp_id": str(variant_id), **context})
        candidates_df = pd.DataFrame(candidate_rows)

        if candidates_df is None or candidates_df.empty:
            return IntegrationResult(
                status="blocked",
                global_limitations=["No candidate variants available in EvidenceState"],
            )


        evidence_items: list[dict[str, Any]] = []
        graph_context: dict[str, Any] = {}
        seen_ids: set[str] = set()
        for candidate_state in state_candidates.values():
            for item in candidate_state.get("trusted_evidence_items", []) or []:
                eid = str(item.get("evidence_id") or "")
                if eid and eid not in seen_ids:
                    evidence_items.append(item)
                    seen_ids.add(eid)
        model_dict: dict[str, Any] = {}


        

        id_col = next((c for c in ("user_snp_id", "snp_id", "variant_id") if c in candidates_df.columns), None)

        cadd_col = next((c for c in ("pvi_score", "cadd_prob", "cadd_phred", "cadd_score") if c in candidates_df.columns), None)
        dpc_col = next((c for c in ("dpc_additive_effect_mean", "abs_effect") if c in candidates_df.columns), None)
        core_col = next((c for c in ("level1_core", "is_core") if c in candidates_df.columns), None)

        candidates_df["_has_cadd"] = candidates_df[cadd_col].notna() if cadd_col else False
        candidates_df["_has_dpc"] = candidates_df[dpc_col].notna() if dpc_col else False
        candidates_df["_is_core"] = candidates_df[core_col].astype(bool) if core_col else False


        conclusions: list[CandidateConclusion] = []
        all_nodes: list[EvidenceNode] = []
        all_edges: list[EvidenceEdge] = []
        global_conflicts: list[str] = []
        report_facts: list[str] = []
        report_hypotheses: list[str] = []
        for row in candidates_df.itertuples(index=False):
            variant_id = str(getattr(row, id_col, "") if id_col else "").strip()
            if not variant_id:
                continue

            variant_info = {
                "chrom": getattr(row, "chrom", ""),
                "pos": getattr(row, "pos", 0),
                "dpc_additive_effect_mean": getattr(row, dpc_col, None) if dpc_col else None,
                "cadd_phred": getattr(row, cadd_col, None) if cadd_col else None,
                "level1_core": getattr(row, core_col, False) if core_col else False,
                "is_core": bool(getattr(row, core_col, False)) if core_col else False,
            }


            has_cadd = bool(cadd_col and pd.notna(getattr(row, cadd_col, None)))
            has_dpc = bool(dpc_col and pd.notna(getattr(row, dpc_col, None)))
            is_core = bool(getattr(row, core_col, False)) if core_col else False
            candidate_state = state_candidates.get(variant_id, {}) if isinstance(state_candidates, dict) else {}
            trusted_state_items = [
                item for item in (candidate_state.get("trusted_evidence_items", []) or [])
                if isinstance(item, dict) and item.get("evidence_id")
            ]

            

            

            
            state_evidence = trusted_state_items
            has_external_evidence = bool(state_evidence)
            has_gene_context = bool(
                candidate_state.get("gene_id")
                or any(edge.get("relation") in {"gene_trait", "gene_mechanism"}
                       for edge in (candidate_state.get("supported_edges") or []))
            )


            nodes, edges, conflicts = _build_evidence_chain(
                variant_id, variant_info, state_evidence,
                candidate_state.get("supported_edges") or [],
                model_dict, species, trait, target_direction,
            )
            all_nodes.extend(nodes)
            all_edges.extend(edges)


            
            max_supported_level = str(candidate_state.get("max_supported_level") or "L4")
            if max_supported_level not in {"L1", "L2", "L3", "L4"}:
                max_supported_level = "L4"
            level = {"L1": "A", "L2": "B", "L3": "C", "L4": "D"}[max_supported_level]

            
            causal = False


            ranking = {}
            if has_dpc:
                ranking["dpc_effect"] = variant_info.get("dpc_additive_effect_mean")
            ranking.update({k: getattr(row, k, None) for k in ("T", "F", "B", "M", "priority_rank")})
            if has_cadd:
                ranking["cadd_score"] = variant_info.get("cadd_phred")

            mechanism = _state_summary(candidate_state, variant_id, trait)
            supported_claims, mechanistic_hypotheses = _state_claim_slots(
                candidate_state, variant_id, trait,
            )
            state_missing_links = _state_missing_links(candidate_state)

            supporting = sorted({
                str(evidence_id)
                for claim in supported_claims + mechanistic_hypotheses
                for evidence_id in claim.supporting_evidence_ids
                if str(evidence_id).strip()
            })
            conflicting: list[str] = []


            conflicts_detected: list[str] = []
            internal_conflicts: list[str] = []


            if is_core and not has_gene_context:
                conflicts_detected.append("Core SNP with no annotated gene in proximity")
                global_conflicts.append(f"Core SNP {variant_id}: no gene context available")


            if has_dpc and target_direction:
                dpc_eff = variant_info.get("dpc_additive_effect_mean")
                if dpc_eff is not None:
                    dpc_sign = float(dpc_eff)
                    expected_negative = target_direction.lower() in ("decrease", "reduce", "dwarf", "矮化", "降低")
                    expected_positive = target_direction.lower() in ("increase", "elevate", "tall", "增高", "提升")
                    if expected_negative and dpc_sign > 0:
                        conflicts_detected.append(
                            f"DPC effect direction ({dpc_sign:+.4f}) contradicts target direction ({target_direction})"
                        )
                    elif expected_positive and dpc_sign < 0:
                        conflicts_detected.append(
                            f"DPC effect direction ({dpc_sign:+.4f}) contradicts target direction ({target_direction})"
                        )


            if has_external_evidence and not has_cadd:
                global_conflicts.append(
                    f"External evidence for {variant_id} without PlantVImpact functional score"
                )


            if not is_core and has_cadd and has_external_evidence and has_gene_context:
                internal_conflicts.append(
                    f"Proxy SNP {variant_id} has multi-axis evidence (CADD+trusted evidence+gene) "
                    f"but lacks DPC Level-1 core status — may need re-evaluation"
                )

            conflicting = conflicts_detected


            if internal_conflicts:
                global_conflicts.extend(internal_conflicts)


            missing = state_missing_links


            supporting_evs = [
                ev for ev in evidence_items
                if isinstance(ev, dict) and ev.get("evidence_id") in supporting
            ]
            independent_count = len({
                (str(ev.get("source_name", "")), str(ev.get("evidence_type", "")))
                for ev in supporting_evs
            })
            ranking["independent_evidence_count"] = independent_count


            alternatives: list[str] = []
            if conflicting:
                alternatives.append(
                    f"{variant_id} 存在冲突证据，不能排除其他基因或位点解释性状效应"
                )
            if not is_core:
                alternatives.append(
                    f"{variant_id} 为 LD proxy，所属 LD component 中其他功能位点"
                    "可能是真正的因果位点"
                )
            if not has_external_evidence:
                alternatives.append(
                    f"{variant_id} 缺乏已准入的外部证据，需要独立实验验证"
                )

            recommended = ["Validate via gene-editing or complementation experiment"]
            if not has_gene_context:
                recommended.append("Map candidate using higher-resolution LD or long-read sequencing")

            conclusions.append(CandidateConclusion(
                variant_id=variant_id,
                ranking_summary=ranking,
                proposed_mechanism=mechanism,
                evidence_level=level,
                confidence=None,
                supporting_evidence_ids=supporting,
                conflicting_evidence_ids=conflicting,
                missing_evidence=missing,
                causal_claim_allowed=causal,
                recommended_validation=recommended,
                alternative_hypotheses=alternatives,
                claim_type="variant_interpretation",
                inference_status="hypothesis",
                causal_strength="variant_causal" if causal else "none",
                supported_claims=supported_claims,
                mechanistic_hypotheses=mechanistic_hypotheses,
                max_supported_level=max_supported_level,
                max_supported_conclusion=str(
                    candidate_state.get("max_supported_conclusion") or "insufficient_evidence"
                ),
            ))


            if is_core:
                report_facts.append(
                    f"SNP {variant_id} ({variant_info['chrom']}:{variant_info['pos']}) "
                    f"passed DPCformer Level-1 core filtering"
                )
                report_hypotheses.extend(
                    claim.text for claim in mechanistic_hypotheses
                )
                report_hypotheses.extend(alternatives)
            else:
                report_facts.append(
                    f"SNP {variant_id} ({variant_info['chrom']}:{variant_info['pos']}) "
                    f"is a proxy SNP (not Level-1 core)"
                )
                report_hypotheses.extend(alternatives)

        mechanism_source = self._apply_llm_mechanism_drafts(
            conclusions,
            evidence_items,
            species,
            trait,
            runtime_context=payload.get("runtime_context"),
            graph_context=graph_context,
            evidence_state=state_candidates,
            revision_feedback=payload.get("revision_feedback"),
        )

        

        from phenofuncrank_repro.interpretation.evidence_contract import evaluate_claim_gate
        for conclusion in conclusions:
            candidate_state = state_candidates.get(conclusion.variant_id, {}) if isinstance(state_candidates, dict) else {}
            fact_gates: list[dict[str, Any]] = []
            hypothesis_gates: list[dict[str, Any]] = []
            valid_facts: list[AtomicClaim] = []
            promoted_hypotheses: list[AtomicClaim] = list(conclusion.mechanistic_hypotheses)

            def gate_slot(claim: AtomicClaim) -> dict[str, Any]:
                return evaluate_claim_gate({
                    "proposed_mechanism": claim.text,
                    "supporting_evidence_ids": claim.supporting_evidence_ids,
                    "claim_type": claim.relation,
                    "relation": claim.relation,
                    "causal_claim_allowed": False,
                }, candidate_state)

            for claim in conclusion.supported_claims:
                gate = gate_slot(claim)
                claim.gate = gate
                fact_gates.append({"claim_id": claim.claim_id, **gate})
                if gate.get("decision") == "PASS":
                    claim.status = "supported"
                    valid_facts.append(claim)
                else:
                    claim.status = "partial" if gate.get("decision") == "PARTIAL" else "blocked"
                    claim.missing_links = canonical_missing_links(
                        list(claim.missing_links) + list(gate.get("missing_links") or [])
                    )
                    promoted_hypotheses.append(claim)

            for claim in promoted_hypotheses:
                gate = gate_slot(claim)
                claim.gate = gate
                claim.status = "hypothesis_only"
                claim.missing_links = canonical_missing_links(
                    list(claim.missing_links) + list(gate.get("missing_links") or [])
                )
                hypothesis_gates.append({"claim_id": claim.claim_id, **gate})


            
            unique_hypotheses: dict[str, AtomicClaim] = {}
            for claim in promoted_hypotheses:
                unique_hypotheses.setdefault(claim.claim_id, claim)
            conclusion.supported_claims = valid_facts
            conclusion.mechanistic_hypotheses = list(unique_hypotheses.values())
            conclusion.missing_links = canonical_missing_links(
                list(conclusion.missing_links) + _state_missing_links(candidate_state)
            )
            conclusion.missing_evidence = list(dict.fromkeys(
                list(conclusion.missing_evidence) + conclusion.missing_links
            ))

            

            conclusion.proposed_mechanism = " ".join(
                claim.text for claim in conclusion.supported_claims
            ) or "No evidence-supported biological relation is established for this candidate."
            conclusion.claim_gate = {
                "decision": (
                    "PARTIAL" if valid_facts and conclusion.mechanistic_hypotheses
                    else "PASS" if valid_facts else "BLOCK"
                ),
                "support_status": "supported" if valid_facts else "insufficient",
                "max_supported_level": candidate_state.get("max_supported_level", "L4"),
                "supported_fact_gates": fact_gates,
                "hypothesis_gates": hypothesis_gates,
                "missing_links": conclusion.missing_links,
            }
            conclusion.causal_claim_allowed = False
            conclusion.inference_status = (
                "partially_supported" if valid_facts and conclusion.mechanistic_hypotheses
                else "supported" if valid_facts
                else "hypothesis_only" if conclusion.mechanistic_hypotheses
                else "unsupported"
            )
            conclusion.causal_strength = (
                "gene_level" if any(c.relation == "gene_trait" for c in valid_facts)
                else "none"
            )
        for conclusion in conclusions[:20]:
            ctx = graph_context.get(conclusion.variant_id) or {}
            if not ctx:
                continue
            covered = ctx.get("covered_layers") or {}
            missing = [gap.get("missing_link") for gap in (ctx.get("gaps") or [])[:2]]
            report_facts.append(
                f"SNP {conclusion.variant_id} evidence-graph coverage: "
                f"gene_anchored={bool(covered.get('gene_anchored'))}, "
                f"literature_edges={covered.get('literature_edge_count', 0)}, "
                f"regulatory_covered={bool(covered.get('regulatory_covered'))}, "
                f"missing={'; '.join(missing) if missing else 'none'}"
            )


        pathway_enrichment_result = None
        if payload.get("matched_enrichment"):
            from phenofuncrank_repro.interpretation.paper_enrichment import matched_enrichment_from_files
            pathway_enrichment_result = matched_enrichment_from_files(payload["matched_enrichment"])
        try:
            from phenofuncrank_repro.interpretation.pathway_enrichment import run_pathway_enrichment
            uniprot_accessions = [
                str(ev.get("source_record_id", "")).strip()
                for ev in evidence_items
                if isinstance(ev, dict)
                and ev.get("evidence_type") == "uniprot_mapping"
                and ev.get("source_record_id")
            ]
            gene_ids = [
                str(state.get("gene_id"))
                for state in state_candidates.values()
                if isinstance(state, dict) and state.get("gene_id")
            ]
            if id_col and not candidates_df.empty:
                for column in ("gene_id", "candidate_gene"):
                    if column in candidates_df.columns:
                        gene_ids.extend(
                            str(value) for value in candidates_df[column].dropna().tolist()
                        )
            refseq_ids = re.findall(
                r"\b(?:XP|XM)_\d+\.\d+\b",
                " ".join(str(ev.get("statement", "")) for ev in evidence_items if isinstance(ev, dict)),
            )
            if pathway_enrichment_result is None and (uniprot_accessions or gene_ids or refseq_ids):
                pathway_enrichment_result = run_pathway_enrichment(
                    uniprot_accessions,
                    gene_ids=gene_ids,
                    refseq_ids=refseq_ids,
                    species=species,
                )
                if pathway_enrichment_result.get("status") == "success":
                    hit_names = "、".join(
                        h["pathway_name"] for h in pathway_enrichment_result.get("hits", [])[:5]
                    ) or "无命中通路"
                    report_facts.append(
                        f"Pathway enrichment: {hit_names} "
                        f"({pathway_enrichment_result.get('note', '')})"
                    )
        except Exception as exc:
            from phenofuncrank_repro.interpretation.logging_utils import get_logger
            get_logger("rice_ph_agent.hypothesis_reasoning").warning(
                "pathway_enrichment failed: %s", exc,
            )


        global_limitations = [
            "Evidence chain is computational; no experimental validation performed",
            "Causal claims are not warranted without gene-editing/mutant evidence",
            "Proxy SNPs carry propagated DPC effects, not direct allele-swap results",
            "PlantVImpact scores indicate functional-impact potential, not trait-specific impact",
            "Literature evidence is from automated search; may contain false positives",
            f"Analysis covers {len(conclusions)} variant(s) in {species}/{trait}",
        ]
        if global_conflicts:
            global_limitations.append(f"Conflicts detected: {'; '.join(global_conflicts[:5])}")


        now = datetime.now(timezone.utc)
        graph_path = self.work_dir / f"{run_id}_evidence_graph.tsv"

        with open(graph_path, "w") as fh:
            fh.write("type\tnode_id\tnode_type\tlabel\tattributes\n")
            for node in all_nodes:
                fh.write(f"node\t{node.node_id}\t{node.node_type}\t{node.label}\t{node.attributes}\n")
            fh.write("type\tedge_id\tsource\ttarget\trelation\tevidence_ids\tconfidence\tdirectness\tcaveats\n")
            for edge in all_edges:
                fh.write(f"edge\t{edge.edge_id}\t{edge.source}\t{edge.target}\t"
                         f"{edge.relation}\t{','.join(edge.evidence_ids)}\t"
                         f"{edge.confidence}\t{edge.directness}\t{';'.join(edge.caveats)}\n")

        graph_artifact = ArtifactRef(
            artifact_id=f"art_evidence_graph_{run_id}",
            artifact_type="evidence_graph",
            uri=str(graph_path),
            sha256=_sha256_file(graph_path),
            size_bytes=graph_path.stat().st_size,
            media_type="text/tab-separated-values",
            schema_version="1.0",
            created_by="hypothesis_reasoning",
            created_at=now,
        )


        conclusions_path = self.work_dir / f"{run_id}_conclusions.tsv"
        if conclusions:
            conclusion_dicts = [c.model_dump() for c in conclusions]
            pd.DataFrame(conclusion_dicts).to_csv(conclusions_path, sep="\t", index=False)

        conclusions_artifact = ArtifactRef(
            artifact_id=f"art_conclusions_{run_id}",
            artifact_type="candidate_conclusions_table",
            uri=str(conclusions_path),
            sha256=_sha256_file(conclusions_path),
            size_bytes=conclusions_path.stat().st_size if conclusions_path.exists() else 0,
            media_type="text/tab-separated-values",
            schema_version="1.0",
            created_by="hypothesis_reasoning",
            created_at=now,
        )

        status = "success"
        if len(conclusions) == 0:
            status = "partial"
        elif global_conflicts:
            status = "partial"

        return IntegrationResult(
            status=status,
            evidence_graph=graph_artifact,
            conclusions_table=conclusions_artifact,
            candidate_conclusions=conclusions,
            global_limitations=global_limitations,
            report_facts=report_facts,
            report_hypotheses=report_hypotheses,
            mechanism_narrative_source=mechanism_source,
            pathway_enrichment=pathway_enrichment_result,
            evidence_state_version=str(evidence_state_bundle.get("contract_version") or "")
            if isinstance(evidence_state_bundle, dict) else "",
        )


    @staticmethod
    def _apply_llm_mechanism_drafts(
        conclusions: list[CandidateConclusion], evidence_items: list[dict[str, Any]],
        species: str, trait: str, runtime_context: dict[str, Any] | None = None,
        graph_context: dict[str, Any] | None = None,
        evidence_state: dict[str, Any] | None = None,
        revision_feedback: dict[str, Any] | None = None,
    ) -> str:
        """Accept only grounded, non-causal drafts for existing conclusions."""
        evidence_by_id = {
            str(item.get("evidence_id")): {
                "evidence_id": item.get("evidence_id"),
                "subject_id": item.get("subject_id"),
                "evidence_type": item.get("evidence_type"),
                "statement": str(item.get("statement", ""))[:300],
                "claim": item.get("claim", {}),
                "directness": item.get("directness"),
                "proximity_level": item.get("proximity_level"),
                "source_type": item.get("source_type"),
                "claim_support": item.get("claim_support"),
            }
            for item in evidence_items if isinstance(item, dict) and item.get("evidence_id")
        }
        payload_conclusions = [
            {
                "variant_id": conclusion.variant_id,
                "supported_claims": [claim.model_dump(mode="json") for claim in conclusion.supported_claims],
                "mechanistic_hypotheses": [claim.model_dump(mode="json") for claim in conclusion.mechanistic_hypotheses],
                "missing_links": conclusion.missing_links,
            }
            for conclusion in conclusions
        ]

        

        referenced_ids = {
            eid for conclusion in conclusions
            for eid in conclusion.supporting_evidence_ids
        }
        ordered_evidence = [
            evidence_by_id[eid] for eid in referenced_ids if eid in evidence_by_id
        ] + [
            value for key, value in evidence_by_id.items() if key not in referenced_ids
        ]
        provided_evidence = ordered_evidence[:300]
        if len(evidence_by_id) > len(provided_evidence):
            logger.warning(
                "A3 evidence payload capped: %d -> %d items",
                len(evidence_by_id), len(provided_evidence),
            )
        llm_payload: dict[str, Any] = {
            "species": species, "trait": trait,
            "conclusions": payload_conclusions,
            "provided_evidence": provided_evidence,
        }
        if graph_context:

            compact_graph: dict[str, Any] = {}
            total_chars = 0
            for variant_id, ctx in graph_context.items():
                if total_chars >= 8000:
                    break
                compact = {
                    "variant_id": variant_id,
                    "found": ctx.get("found", False),
                    "covered_layers": ctx.get("covered_layers", {}),
                    "gaps": [
                        {
                            "missing_link": gap.get("missing_link"),
                            "priority": gap.get("priority"),
                            "suggested_tools": gap.get("suggested_tools", []),
                        }
                        for gap in (ctx.get("gaps") or [])[:5]
                    ],
                    "context": ctx.get("context", ""),
                }
                compact_graph[variant_id] = compact
                total_chars += len(json.dumps(compact, ensure_ascii=False, default=str))
            if compact_graph:
                llm_payload["graph_context"] = compact_graph
        if evidence_state:
            llm_payload["evidence_state"] = {
                variant_id: {
                    "supported_edges": state.get("supported_edges", []),
                    "predicted_edges": state.get("predicted_edges", []),
                    "missing_edges": state.get("missing_edges", []),
                    "supported_relations": state.get("supported_relations", []),
                    "missing_relations": state.get("missing_relations", []),
                    "forbidden_conclusions": state.get("forbidden_conclusions", []),
                    "max_supported_level": state.get("max_supported_level", "L4"),
                    "max_supported_conclusion": state.get("max_supported_conclusion", "insufficient_evidence"),
                }
                for variant_id, state in list(evidence_state.items())[:50]
            }
        if runtime_context:
            llm_payload["runtime_context"] = runtime_context
        if isinstance(revision_feedback, dict) and revision_feedback:
            llm_payload["revision_feedback"] = {
                "next_action": revision_feedback.get("next_action"),
                "claim_reviews": [
                    {
                        "claim_id": item.get("claim_id"),
                        "semantic_issues": item.get("semantic_issues", []),
                        "allowed_rewrite_level": item.get("allowed_rewrite_level"),
                    }
                    for item in (revision_feedback.get("claim_reviews") or [])
                    if isinstance(item, dict)
                ],
                "evidence_gaps": [
                    {
                        "candidate_id": item.get("candidate_id"),
                        "missing_link": item.get("missing_link"),
                        "priority": item.get("priority"),
                    }
                    for item in (revision_feedback.get("evidence_gaps") or [])
                    if isinstance(item, dict)
                ],
            }
        data = complete_json(
            "你是植物遗传机制整合助手。只能改写已提供的 atomic claim 槽位，"
            "不能创建新的关系、证据 ID、基因、性状、数值或因果结论。"
            "supported_claims 只能改写原有 supported claim，不能把 hypothesis 放入其中；"
            "mechanistic_hypotheses 必须保持 hypothesis_only，并明确缺失链接。"
            "如果存在 revision_feedback，只修改其中列出的 claim_id；"
            "禁止因为反馈而新增关系、证据 ID、基因、性状或证据等级。"
            "EvidenceState 的 supported_edges 是事实边界，predicted/missing edges 不能写成已支持事实。"
            "不要把多个关系合并成一个长句。严格返回 JSON："
            "{\"variants\":[{\"variant_id\":\"...\","
            "\"supported_claims\":[{\"claim_id\":\"...\",\"relation\":\"...\",\"text\":\"...\"}],"
            "\"mechanistic_hypotheses\":[{\"claim_id\":\"...\",\"relation\":\"...\",\"text\":\"...\",\"missing_links\":[\"...\"]}]}]}。",
            llm_payload,
            max_tokens=settings.llm_synthesis_max_tokens,
        )
        if not data:
            return "deterministic"
        by_variant = {conclusion.variant_id: conclusion for conclusion in conclusions}
        changed = False
        banned = ("causes", "proves", "demonstrates", "导致", "证明", "因果", "确定因果", "ir8")
        for draft in data.get("variants", []):
            if not isinstance(draft, dict):
                continue
            conclusion = by_variant.get(str(draft.get("variant_id", "")))
            if not conclusion:
                continue
            slots = {
                str(claim.claim_id): claim
                for claim in conclusion.supported_claims + conclusion.mechanistic_hypotheses
            }
            for key in ("supported_claims", "mechanistic_hypotheses"):
                for proposed in draft.get(key, []) or []:
                    if not isinstance(proposed, dict):
                        continue
                    claim = slots.get(str(proposed.get("claim_id") or ""))
                    text = proposed.get("text")
                    if not claim or not isinstance(text, str) or not text.strip():
                        continue
                    if str(proposed.get("relation") or "") != claim.relation:
                        continue
                    if any(term in text.lower() for term in banned):
                        continue
                    claim.text = text.strip()[:1000]
                    if key == "mechanistic_hypotheses":
                        gaps = proposed.get("missing_links", [])
                        if isinstance(gaps, list):
                            claim.missing_links = canonical_missing_links(
                                list(claim.missing_links)
                                + [str(gap) for gap in gaps if str(gap) in MISSING_LINK_RELATIONS]
                            )
                    changed = True
        return "llm" if changed else "deterministic"


def short_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return "0" * 64
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
