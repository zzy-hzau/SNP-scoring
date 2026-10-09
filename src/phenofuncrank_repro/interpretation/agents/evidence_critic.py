"Evidence Critic Agent (A4) — independent review of A3 hypotheses."

from __future__ import annotations

from pathlib import Path
from typing import Any

from phenofuncrank_repro.interpretation.logging_utils import get_logger, increment
from phenofuncrank_repro.interpretation.evidence_contract import evaluate_claim_gate, is_affirmative_causal_claim
from phenofuncrank_repro.interpretation.evidence_type_registry import CLAIM_TYPE_REQUIREMENTS
from phenofuncrank_repro.interpretation.llm_utils import complete_json
from phenofuncrank_repro.interpretation.schemas.critique import GAP_INTENT_RELATIONS, SEMANTIC_ISSUE_TYPES, ClaimReview, CriticFeedback, CritiqueReport, EvidenceGap, EvidenceGapIntent, SemanticClaimReview
from phenofuncrank_repro.interpretation.schemas.evidence_graph import CandidateConclusion
from phenofuncrank_repro.interpretation.schemas.evidence_state import MISSING_LINK_RELATIONS, canonical_missing_links
from phenofuncrank_repro.interpretation.schemas.knowledge import EvidenceItem
from phenofuncrank_repro.interpretation.schemas.messages import MessageEnvelope
from phenofuncrank_repro.interpretation.settings import settings
from phenofuncrank_repro.interpretation.gap_mapper import GAP_EVIDENCE_POLICY, map_gap_intents

logger = get_logger(__name__)

_BANNED_CAUSAL_WORDS = (
    "causes", "causal", "causative", "demonstrates", "confirms", "proves", "determines",
)
_HIGH_EVIDENCE_LEVELS = {"high", "A"}
_ALLOWED_SEMANTIC_EVIDENCE_TYPES = {
    "variant_specific", "precise_editing", "variant_editing", "natural_allele",
    "allele_specific", "allele_specific_expression", "eqtl", "variant_linked_eqtl",
    "functional_assay", "functional_validation", "gene_trait", "gene_trait_association",
    "mutant_phenotype", "complementation", "rescue", "transgenic", "trait_association",
    "gwas", "qtl", "mechanism", "molecular_mechanism", "pathway", "protein_interaction",
    "protein_function", "expression", "regulatory", "promoter_motif", "tfbs_overlap",
}
_ALLOWED_SEMANTIC_TOOLS = {
    "multi_query_literature_search", "interpro_scan", "pfam_domain_for_protein",
    "fimo_motif_scan", "jaspar_motif_scan", "plantregmap_genome_tfbs_overlap",
    "rnafold_structure_prediction",
}

_SEMANTIC_CRITIC_SYSTEM = """You are the semantic layer of a scientific evidence critic.
Review only the supplied atomic claims against the supplied trusted EvidenceState
and trusted EvidenceItems. Do not retrieve, invent, or assume evidence. Gate 1
and Gate 2 results are authoritative: never turn a BLOCK into ACCEPT and never
add a new claim or evidence ID. Detect semantic mismatches that typed rules can
miss: direction, trait scope, variant/allele identity, ortholog species,
mechanism over-specificity or unverified mediation, tissue/stage scope,
association-to-mechanism, contradiction, and misuse of indirect evidence.

Return exactly one JSON object with this shape:
{"claim_reviews":[{"claim_id":"existing id","support_state":"SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED|CONTRADICTED","semantic_issues":["known issue id"],"decision":"ACCEPT|REVISE|RETRIEVE_MORE_EVIDENCE|HUMAN_REVIEW","allowed_rewrite_level":"fact|mechanistic_hypothesis|remove_claim|none","evidence_ids_checked":["existing evidence id"],"rationale":"short"}],"gap_intents":[{"gap_relation":"canonical relation id","gap_reason":"atomic missing relation","required_evidence_level":"L1|L2|L3|L4","preferred_evidence_types":["type hints only"],"target_variant":"existing variant id","target_gene":"existing gene id","target_trait":"active trait","priority":"high|medium|low"}],"resolution_mode":"REWRITE_ONLY|RETRIEVE_EVIDENCE|HUMAN_REVIEW|NO_ACTION"}
Use an empty list when there is no issue. Gap intents must use one of:
variant_functional_effect, variant_trait, variant_expression,
variant_mechanism, variant_protein_function, gene_trait, gene_mechanism,
contradiction_resolution. Do not output PubMed queries or tool choices.
The word 'may' does not make an unsupported relation evidence-grounded: mark it
as hypothesis-level or revise it. A supported gene-level fact must not be rejected
just because a separate SNP-level link is missing.
When EvidenceState.missing_edges contains an exact canonical relation, use that
relation verbatim for a retrieval gap; do not replace variant_expression with
the broader variant_functional_effect, and do not invent a neighboring relation.
Use RETRIEVE_EVIDENCE only when the absent relation is plausibly obtainable by
new evidence. Use REWRITE_ONLY for identity, direction, trait, species, or scope
mismatches that require changing the claim. If contradictory evidence is marked
unresolved, use HUMAN_REVIEW or contradiction_resolution rather than treating a
new search as an automatic repair. For a retrieval mode, emit at least one
gap_intent; for rewrite-only mode, emit no gap_intent."""


def _claim_type_of(mechanism: str) -> list[str]:
    """Classify a proposed mechanism into claim types (SOP §4.6)."""
    text = (mechanism or "").lower()
    types: list[str] = []
    if any(k in text for k in ("expression", "transcription", "rna", "mrna", "transcript", "表达")):
        types.append("expression")
    if any(k in text for k in ("regulatory", "cis-regulatory", "promoter", "motif", "tfbs", "enhancer", "调控", "启动子")):
        types.append("regulatory")
    if any(k in text for k in ("function", "activity", "enzyme", "protease", "oxidase", "kinase", "功能")):
        types.append("function")
    if any(k in text for k in (
        "trait", "phenotype", "height", "dwarf", "steril", "yield",
        "architecture", "flowering", "drought", "tolerance", "grain",
        "seed", "tiller", "panicle", "spikelet", "size", "resistance",
        "maturity", "root", "leaf", "flower", "性状", "表型",
    )):
        types.append("trait")
    return types


def format_audit_conclusion(
    review: ClaimReview,
    *,
    variant_id: str = "",
) -> dict[str, Any]:
    "Fixed conclusion template (user design / SOP §7.5)."
    covered = set(review.covered_evidence_types)
    required = set(review.required_evidence_types)
    missing = sorted(required - covered)
    has_gene_trait = any(
        t in covered for t in ("mutant_phenotype", "trait_association", "gwas", "qtl")
    )
    has_variant_specific = any(
        t in covered for t in ("variant_specific", "allele_specific", "eqtl")
    )
    if not has_variant_specific and "variant_specific" not in missing:
        missing.insert(0, "variant_specific")
    if review.status == "supported" and has_variant_specific:
        conclusion = "Evidence supports the proposed SNP interpretation."
    elif has_gene_trait:
        conclusion = (
            "Gene-trait association is supported; SNP-level causality "
            "is not established."
        )
    elif review.status == "unsupported":
        conclusion = "Evidence does not support the proposed interpretation."
    else:
        conclusion = "Insufficient evidence for the proposed interpretation."
    return {
        "variant_id": variant_id or review.claim_id,
        "conclusion": conclusion,
        "evidence_supports": sorted(covered),
        "evidence_does_not_establish": (
            ["SNP-level causality"] if not has_variant_specific else []
        ),
        "missing_evidence": missing,
    }


def _suggested_tools_for(missing_link: str) -> list[str]:
    """Map a missing-link description to registered knowledge tools.

    Only tools inside AGENT_WHITELIST["evidence_retrieval"] are returned so the
    Supervisor can forward the gap to A2 without registry violations.
    """
    text = (missing_link or "").lower()
    tools: list[str] = []
    if text in {
        "variant_functional_effect", "variant_trait", "variant_expression",
        "variant_mechanism", "variant_protein_function", "gene_trait",
        "gene_mechanism",
    }:
        tools.append("multi_query_literature_search")
    if any(k in text for k in ("protein", "domain", "蛋白", "结构")):
        tools += ["interpro_scan", "pfam_domain_for_protein"]
    if any(k in text for k in ("motif", "regulat", "promoter", "调控", "启动子")):
        tools += ["fimo_motif_scan", "jaspar_motif_scan", "plantregmap_genome_tfbs_overlap"]
    if any(k in text for k in ("splice", "rna", "剪接", "rna 结构")):
        tools += ["rnafold_structure_prediction"]
    if any(k in text for k in ("literature", "文献", "性状关联")):
        tools += ["multi_query_literature_search"]
    return tools


def _compact_semantic_evidence(item: EvidenceItem) -> dict[str, Any]:
    """Expose only trusted, typed evidence needed by the semantic critic."""
    provenance = item.provenance if isinstance(item.provenance, dict) else {}
    claim = item.claim if isinstance(item.claim, dict) else {}
    return {
        "evidence_id": item.evidence_id,
        "source_type": item.source_type,
        "subject_type": item.subject_type,
        "subject_id": item.subject_id,
        "evidence_type": item.evidence_type,
        "proximity_level": item.proximity_level,
        "directness": item.directness,
        "species": item.species,
        "trait": item.trait,
        "statement": item.statement[:1800],
        "claim": {
            "relations": list(claim.get("relations") or []),
            "text": str(claim.get("text") or "")[:1000],
        },
        "stance": item.stance,
        "claim_support": item.claim_support,
        "provenance": {
            "source_record_id": item.source_record_id,
            "title": provenance.get("title") or provenance.get("paper_title"),
            "quote": str(provenance.get("quote") or "")[:1200],
            "chunk_id": provenance.get("chunk_id"),
            "status": item.provenance_status,
        },
    }


def _semantic_payload(
    command: MessageEnvelope,
    report: CritiqueReport,
    evidence_items: list[EvidenceItem],
    state_by_variant: dict[str, Any],
) -> tuple[dict[str, Any], set[str], set[str], set[str]]:
    """Build the A1-owned semantic review payload and its ID allow-lists."""
    payload = command.payload
    integration = payload.get("integration_result") or {}
    allowed_claim_ids: set[str] = set()
    allowed_evidence_ids = {item.evidence_id for item in evidence_items}
    allowed_variant_ids = set(state_by_variant)
    claims: list[dict[str, Any]] = []
    for raw in (integration.get("candidate_conclusions") or [])[:settings.hybrid_a4_max_claims]:
        try:
            conclusion = raw if isinstance(raw, CandidateConclusion) else CandidateConclusion(**raw)
        except Exception:
            continue
        units = list(conclusion.supported_claims) + list(conclusion.mechanistic_hypotheses)
        for atom in units:
            allowed_claim_ids.add(atom.claim_id)
            claims.append({
                "claim_id": atom.claim_id,
                "variant_id": conclusion.variant_id,
                "relation": atom.relation,
                "text": atom.text,
                "status": atom.status,
                "evidence_level": atom.evidence_level,
                "supporting_evidence_ids": [
                    eid for eid in atom.supporting_evidence_ids if eid in allowed_evidence_ids
                ],
                "missing_links": canonical_missing_links(atom.missing_links),
                "deterministic_gate": next(
                    (review.model_dump() for review in report.claim_reviews if review.claim_id == atom.claim_id),
                    {"gate_decision": "UNKNOWN"},
                ),
            })

    states: list[dict[str, Any]] = []
    for variant_id, state in list(state_by_variant.items())[:settings.hybrid_a4_max_claims]:
        if not isinstance(state, dict):
            continue
        states.append({
            "variant_id": variant_id,
            "gene_id": state.get("gene_id", ""),
            "trait": state.get("trait") or payload.get("trait", ""),
            "supported_edges": state.get("supported_edges", []),
            "predicted_edges": state.get("predicted_edges", []),
            "missing_edges": state.get("missing_edges", []),
            "forbidden_conclusions": state.get("forbidden_conclusions", []),
            "contradictory_evidence_ids": state.get("contradictory_evidence_ids", []),
            "max_supported_level": state.get("max_supported_level", "L4"),
            "max_supported_conclusion": state.get("max_supported_conclusion", "insufficient_evidence"),
        })

    evidence = [
        _compact_semantic_evidence(item)
        for item in evidence_items[:settings.hybrid_a4_max_evidence_items]
    ]
    return {
        "case_context": {
            "species": payload.get("species", ""),
            "trait": payload.get("trait", ""),
            "target_direction": payload.get("target_direction"),
        },
        "evidence_state": states,
        "trusted_evidence_items": evidence,
        "atomic_claims": claims,
        "known_missing_links": sorted({
            str(edge.get("relation"))
            for state in states
            for edge in (state.get("missing_edges") or [])
            if str(edge.get("relation")) in MISSING_LINK_RELATIONS
            }),
            "allowed_gap_intent_relations": sorted(GAP_INTENT_RELATIONS),
        }, allowed_claim_ids, allowed_evidence_ids, allowed_variant_ids


class EvidenceCriticAgent:
    """Deterministic reviewer with an advisory DeepSeek semantic layer.

    Gate 1 and Gate 2 remain pure code and are always run first.  The DeepSeek
    layer is advisory: its output is allow-listed and can only add semantic
    review feedback or request an A1-mediated action.
    """

    def __init__(self, work_dir: Path | None = None):
        self.work_dir = work_dir or (settings.root / "critic_runs")
        self.work_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _semantic_feedback(
        command: MessageEnvelope,
        report: CritiqueReport,
        evidence_items: list[EvidenceItem],
        state_by_variant: dict[str, Any],
    ) -> CriticFeedback:
        """Run and strictly normalize the optional local semantic critic."""
        if not settings.hybrid_a4_enabled:
            return CriticFeedback(
                enabled=False, provider="none", model=settings.hybrid_a4_model,
                status="not_run", next_action="ACCEPT",
            )

        request, claim_ids, evidence_ids, variant_ids = _semantic_payload(
            command, report, evidence_items, state_by_variant,
        )
        raw = complete_json(
            _SEMANTIC_CRITIC_SYSTEM,
            request,
            max_tokens=settings.hybrid_a4_max_tokens,
            model=settings.hybrid_a4_model,
            base_url=settings.hybrid_a4_base_url,
        )
        if not isinstance(raw, dict):
            increment("hybrid_critic_invalid_output")
            return CriticFeedback(
                enabled=True, provider="deepseek_api", model=settings.hybrid_a4_model,
                status="invalid_output", next_action="ACCEPT",
                error="semantic critic returned no JSON object; deterministic decision retained",
            )

        reviews: list[SemanticClaimReview] = []
        for candidate in raw.get("claim_reviews") or []:
            if not isinstance(candidate, dict):
                continue
            claim_id = str(candidate.get("claim_id") or "")
            if claim_id not in claim_ids:
                continue
            state = str(candidate.get("support_state") or "").upper()
            action = str(candidate.get("decision") or "").upper()
            if state not in {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED"}:
                continue
            if action not in {"ACCEPT", "REVISE", "RETRIEVE_MORE_EVIDENCE", "HUMAN_REVIEW"}:
                continue
            issues = [
                str(issue) for issue in (candidate.get("semantic_issues") or [])
                if str(issue) in SEMANTIC_ISSUE_TYPES
            ]
            rewrite = candidate.get("allowed_rewrite_level")
            if rewrite not in {"fact", "mechanistic_hypothesis", "remove_claim", "none", None}:
                rewrite = None
            checked = [
                str(eid) for eid in (candidate.get("evidence_ids_checked") or [])
                if str(eid) in evidence_ids
            ]
            reviews.append(SemanticClaimReview(
                claim_id=claim_id,
                support_state=state,  
                semantic_issues=sorted(set(issues)),
                decision=action,  
                allowed_rewrite_level=rewrite,
                evidence_ids_checked=checked,
                rationale=str(candidate.get("rationale") or "")[:1200],
            ))


        

        

        intents: list[EvidenceGapIntent] = []
        raw_intents = raw.get("gap_intents")
        if isinstance(raw_intents, list):
            for candidate in raw_intents:
                if not isinstance(candidate, dict):
                    continue
                relation = str(candidate.get("gap_relation") or "")
                if relation not in GAP_INTENT_RELATIONS or relation not in GAP_EVIDENCE_POLICY:
                    continue
                candidate_id = str(candidate.get("target_variant") or "")
                if not candidate_id and len(variant_ids) == 1:
                    candidate_id = next(iter(variant_ids))
                if candidate_id not in variant_ids:
                    continue
                state = state_by_variant.get(candidate_id, {})
                policy = GAP_EVIDENCE_POLICY[relation]
                level = str(candidate.get("required_evidence_level") or policy["level"]).upper()
                if level not in {"L1", "L2", "L3", "L4"}:
                    level = policy["level"]
                preferred = [
                    str(value)[:80] for value in (candidate.get("preferred_evidence_types") or [])
                    if isinstance(value, str) and value.strip()
                ][:8]
                priority = str(candidate.get("priority") or "medium").lower()
                if priority not in {"high", "medium", "low"}:
                    priority = "medium"
                try:
                    intents.append(EvidenceGapIntent(
                        gap_relation=relation,
                        gap_reason=str(candidate.get("gap_reason") or "")[:1200],
                        required_evidence_level=level,  
                        preferred_evidence_types=preferred,
                        target_variant=candidate_id,
                        target_gene=str(candidate.get("target_gene") or state.get("gene_id") or "")[:200],
                        target_trait=str(candidate.get("target_trait") or state.get("trait") or command.payload.get("trait") or "")[:300],
                        priority=priority,  
                    ))
                except Exception:
                    continue

        gaps: list[EvidenceGap] = []
        if raw_intents is not None:
            gaps = map_gap_intents(intents, candidate_id=next(iter(variant_ids), ""))
        else:

            

            for candidate in raw.get("evidence_gaps") or []:
                if not isinstance(candidate, dict):
                    continue
                candidate_id = str(candidate.get("candidate_id") or "")
                relation = str(candidate.get("missing_link") or "")
                if candidate_id not in variant_ids or relation not in MISSING_LINK_RELATIONS:
                    continue
                priority = str(candidate.get("priority") or "important")
                if priority not in {"critical", "important", "optional"}:
                    priority = "important"
                evidence_types = sorted({
                    str(value) for value in (candidate.get("recommended_evidence_types") or [])
                    if str(value) in _ALLOWED_SEMANTIC_EVIDENCE_TYPES
                })
                tools = sorted({
                    str(value) for value in (candidate.get("suggested_tools") or [])
                    if str(value) in _ALLOWED_SEMANTIC_TOOLS
                })
                gaps.append(EvidenceGap(
                    gap_id=f"semantic_{candidate_id}_{relation}",
                    candidate_id=candidate_id,
                    missing_link=relation,
                    priority=priority,  
                    recommended_evidence_types=evidence_types,
                    suggested_tools=tools,
                ))

        requested_mode = str(raw.get("resolution_mode") or "").upper()
        old_action = str(raw.get("next_action") or "").upper()
        mode_from_action = {
            "ACCEPT": "NO_ACTION",
            "REVISE": "REWRITE_ONLY",
            "RETRIEVE_MORE_EVIDENCE": "RETRIEVE_EVIDENCE",
            "HUMAN_REVIEW": "HUMAN_REVIEW",
        }
        if requested_mode not in {
            "REWRITE_ONLY", "RETRIEVE_EVIDENCE", "HUMAN_REVIEW", "NO_ACTION",
        }:
            requested_mode = mode_from_action.get(old_action, "")
        if not requested_mode:
            review_actions = {review.decision for review in reviews}
            if "HUMAN_REVIEW" in review_actions:
                requested_mode = "HUMAN_REVIEW"
            elif "RETRIEVE_MORE_EVIDENCE" in review_actions:
                requested_mode = "RETRIEVE_EVIDENCE"
            elif "REVISE" in review_actions:
                requested_mode = "REWRITE_ONLY"
            else:
                requested_mode = "NO_ACTION"


        

        if requested_mode == "RETRIEVE_EVIDENCE" and not gaps:
            requested_mode = "REWRITE_ONLY" if reviews else "NO_ACTION"
        if requested_mode == "REWRITE_ONLY" and not reviews:
            requested_mode = "NO_ACTION"
        next_action = {
            "NO_ACTION": "ACCEPT",
            "REWRITE_ONLY": "REVISE",
            "RETRIEVE_EVIDENCE": "RETRIEVE_MORE_EVIDENCE",
            "HUMAN_REVIEW": "HUMAN_REVIEW",
        }[requested_mode]
        return CriticFeedback(
            enabled=True, provider="deepseek_api", model=settings.hybrid_a4_model,
            status="success", resolution_mode=requested_mode, next_action=next_action,
            claim_reviews=reviews, gap_intents=intents, evidence_gaps=gaps,
        )

    @staticmethod
    def _merge_semantic_decision(
        report: CritiqueReport, feedback: CriticFeedback,
    ) -> None:
        """Apply DeepSeek feedback without weakening deterministic decisions."""
        if not feedback.enabled or feedback.status != "success":
            return
        deterministic = report.decision

        
        existing_gap_ids = {existing.gap_id for existing in report.evidence_gaps}
        for gap in feedback.evidence_gaps:
            if gap.gap_id not in existing_gap_ids:
                report.evidence_gaps.append(gap)
        if deterministic == "HUMAN_REVIEW":
            feedback.resolution_mode = "HUMAN_REVIEW"
            feedback.next_action = "HUMAN_REVIEW"
            return
        if deterministic == "RETRIEVE_MORE_EVIDENCE":
            feedback.resolution_mode = "RETRIEVE_EVIDENCE"
            feedback.next_action = "RETRIEVE_MORE_EVIDENCE"
            return
        if feedback.next_action == "HUMAN_REVIEW":
            report.decision = "HUMAN_REVIEW"
        elif feedback.next_action == "RETRIEVE_MORE_EVIDENCE":
            report.decision = "RETRIEVE_MORE_EVIDENCE"
        elif feedback.next_action == "REVISE":
            report.decision = "REVISE_WITH_EXISTING_EVIDENCE"

    def invoke(self, command: MessageEnvelope) -> CritiqueReport:
        """Review CandidateConclusions against the A1-owned EvidenceState."""
        payload = command.payload
        integration_raw = payload.get("integration_result") or {}
        evidence_state_bundle = payload.get("evidence_state") or {}

        
        knowledge_raw = payload.get("knowledge_result") or {} if not evidence_state_bundle else {}

        evidence_items: list[EvidenceItem] = []
        raw_items: list[Any] = []
        if isinstance(evidence_state_bundle, dict) and evidence_state_bundle.get("candidates"):
            for candidate_state in evidence_state_bundle.get("candidates", {}).values():
                raw_items.extend(candidate_state.get("trusted_evidence_items", []) or [])
        else:
            raw_items = list(knowledge_raw.get("evidence_items", []) or [])
        for item in raw_items:
            if isinstance(item, EvidenceItem):
                evidence_items.append(item)
            elif isinstance(item, dict):
                try:
                    evidence_items.append(EvidenceItem(**item))
                except Exception:
                    continue
        evidence_ids = {item.evidence_id for item in evidence_items}
        evidence_by_id = {item.evidence_id: item for item in evidence_items}
        state_by_variant = (
            evidence_state_bundle.get("candidates", {})
            if isinstance(evidence_state_bundle, dict) else {}
        )
        conclusions: list[CandidateConclusion] = []
        for item in integration_raw.get("candidate_conclusions", []) or []:
            if isinstance(item, CandidateConclusion):
                conclusions.append(item)
            elif isinstance(item, dict):
                try:
                    conclusions.append(CandidateConclusion(**item))
                except Exception:
                    continue

        claim_reviews: list[ClaimReview] = []
        evidence_gaps: list[EvidenceGap] = []
        causal_violations: list[str] = []
        factual_errors: list[str] = []
        ref_errors: list[str] = []
        accepted: list[str] = []

        review_units: list[tuple[CandidateConclusion, CandidateConclusion, str, str, str]] = []
        for original in conclusions:
            atoms = [
                ("supported_fact", atom) for atom in original.supported_claims
            ] + [
                ("hypothesis", atom) for atom in original.mechanistic_hypotheses
            ]
            if not atoms:
                if (
                    original.proposed_mechanism
                    or original.supporting_evidence_ids
                    or original.causal_claim_allowed
                ):
                    legacy = original.model_copy(deep=True)
                    legacy.missing_evidence = list(original.missing_evidence)
                    legacy.missing_links = list(original.missing_links)
                    legacy.claim_gate = dict(original.claim_gate)
                    review_units.append((
                        original, legacy, f"{original.variant_id}:legacy",
                        "legacy_combined", original.claim_type,
                    ))
                else:
                    ref_errors.append(
                        f"{original.variant_id}: missing supported_claims or mechanistic_hypotheses slot"
                    )
                continue
            for kind, atom in atoms:
                review = original.model_copy(deep=True)
                review.proposed_mechanism = atom.text
                review.supporting_evidence_ids = list(atom.supporting_evidence_ids)
                review.missing_evidence = list(atom.missing_links)
                review.missing_links = list(atom.missing_links)
                review.claim_type = atom.relation or original.claim_type
                review.causal_claim_allowed = False
                review.claim_gate = dict(atom.gate)

                

                level_map = {"L1": "A", "L2": "B", "L3": "C", "L4": "D"}
                review.evidence_level = level_map.get(atom.evidence_level, original.evidence_level)
                review_units.append((original, review, atom.claim_id, kind, atom.relation))

        for original_conclusion, conclusion, claim_id, claim_kind, claim_relation in review_units:
            checked = list(conclusion.supporting_evidence_ids)
            problems: list[str] = []
            candidate_state = state_by_variant.get(conclusion.variant_id, {}) if isinstance(state_by_variant, dict) else {}
            gate_result = (
                evaluate_claim_gate(conclusion.model_dump(), candidate_state)
                if candidate_state else {"decision": "PASS", "violations": []}
            )
            contradictory_ids = {
                str(eid) for eid in (candidate_state.get("contradictory_evidence_ids") or [])
            }


            invented = [eid for eid in conclusion.supporting_evidence_ids if eid not in evidence_ids]
            if invented:
                ref_errors.append(f"{claim_id}: unknown evidence IDs {invented}")
                problems.append("references unknown evidence IDs")
            if gate_result.get("decision") == "BLOCK":

                

                if claim_kind == "supported_fact":
                    problems.extend(str(v) for v in gate_result.get("violations") or [])
                if any("causal" in str(v) for v in gate_result.get("violations") or []):
                    causal_violations.append(
                        f"{claim_id}: claim exceeded the independent EvidenceState boundary"
                    )


            

            

            
            cited_contradictions = contradictory_ids.intersection(
                str(eid) for eid in conclusion.supporting_evidence_ids
            )
            affirmative_claim = is_affirmative_causal_claim(conclusion.proposed_mechanism or "")
            if cited_contradictions:
                message = (
                    f"{claim_id}: cites contradictory evidence as support "
                    f"{sorted(cited_contradictions)}"
                )
                factual_errors.append(message)
                problems.append("contradictory evidence cited as supporting evidence")
            elif contradictory_ids and affirmative_claim:
                message = (
                    f"{claim_id}: unresolved contradictory evidence requires review "
                    f"{sorted(contradictory_ids)}"
                )
                factual_errors.append(message)
                problems.append("unresolved contradictory evidence requires review")

            mechanism = (conclusion.proposed_mechanism or "").lower()
            if not candidate_state:
                for word in _BANNED_CAUSAL_WORDS:
                    if f" {word} " in f" {mechanism} ":
                        causal_violations.append(
                            f"{claim_id}: banned causal language '{word}'"
                        )
                        problems.append(f"banned causal language '{word}'")


            if conclusion.causal_claim_allowed and not conclusion.supporting_evidence_ids:
                ref_errors.append(f"{claim_id}: causal claim requested without supporting evidence")
                problems.append("causal claim requested without supporting evidence")


            

            gate_violations = [str(v) for v in gate_result.get("violations") or []]
            entailment = gate_result.get("claim_evidence_entailment") or {}
            if entailment.get("decision") in {"BLOCK", "PARTIAL"}:
                entailment_status = str(entailment.get("status") or "claim-evidence mismatch")
                if claim_kind == "supported_fact":
                    problems.append(f"claim-evidence entailment: {entailment_status}")

                
                for relation in sorted(set(
                    (entailment.get("unsupported_relations") or [])
                    + (entailment.get("partial_relations") or [])
                )) if claim_kind == "supported_fact" else []:
                    relation_types = {
                        "variant_functional_effect": ["variant_specific"],
                        "variant_trait": ["variant_specific", "allele_specific", "variant_trait"],
                        "variant_expression": ["eqtl", "allele_specific_expression", "variant_specific"],
                        "variant_mechanism": ["variant_specific", "functional_assay"],
                        "variant_protein_function": ["variant_specific", "functional_assay"],
                        "gene_trait": ["gene_trait", "mutant_phenotype"],
                        "gene_mechanism": ["mechanism", "pathway"],
                    }
                    priority = "critical" if relation in (entailment.get("unsupported_relations") or []) else "important"
                    evidence_gaps.append(EvidenceGap(
                        gap_id=f"gap_{claim_id}_entailment_{relation}",
                        candidate_id=conclusion.variant_id,
                        missing_link=relation,
                        priority=priority,
                        recommended_evidence_types=relation_types.get(relation, []),
                        suggested_tools=["multi_query_literature_search"],
                    ))
            if any(
                token in v
                for v in gate_violations
                for token in (
                    "requires_trusted_direct_L1",
                    "requires_exact_variant_assay",
                    "requires_supported_gene_trait_edge",
                )
            ):
                missing_link = "variant → gene functional effect (trusted direct L1 evidence)"
                evidence_gaps.append(EvidenceGap(
                    gap_id=f"gap_{claim_id}_variant_l1",
                    candidate_id=conclusion.variant_id,
                    missing_link=missing_link,
                    priority="critical",
                    recommended_evidence_types=["variant_specific"],
                    suggested_tools=["multi_query_literature_search"],
                ))
            elif not candidate_state:

                
                for missing in conclusion.missing_evidence:
                    priority = "critical" if any(
                        k in (missing or "").lower()
                        for k in ("consequence", "effect", "trait", "phenotype", "功能", "性状")
                    ) else "important"
                    evidence_gaps.append(EvidenceGap(
                        gap_id=f"gap_{claim_id}_{len(evidence_gaps)}",
                        candidate_id=conclusion.variant_id,
                        missing_link=missing,
                        priority=priority,
                        recommended_evidence_types=[],
                        suggested_tools=_suggested_tools_for(missing),
                    ))


            
            supporting_items = [
                evidence_by_id[eid] for eid in conclusion.supporting_evidence_ids
                if eid in evidence_by_id
            ]
            if conclusion.causal_claim_allowed and any(
                item.directness == "orthology_inference" for item in supporting_items
            ):
                factual_errors.append(
                    f"{claim_id}: causal claim relies on orthology inference"
                )
                problems.append("causal claim relies on orthology inference")
            if conclusion.evidence_level in _HIGH_EVIDENCE_LEVELS and supporting_items and all(
                item.directness in {"computational_prediction", "orthology_inference"}
                for item in supporting_items
            ):
                factual_errors.append(
                    f"{claim_id}: high evidence level based only on "
                    "computational/orthology evidence"
                )
                problems.append("high evidence level based only on computational/orthology evidence")


            claim_types = _claim_type_of(conclusion.proposed_mechanism or "")
            covered = sorted({item.evidence_type for item in supporting_items})

            

            
            if candidate_state:
                claim_types = []

            

            missing_types = [
                ct for ct in claim_types
                if not any(et in covered for et in CLAIM_TYPE_REQUIREMENTS.get(ct, []))
            ]
            required = sorted({
                et for ct in claim_types
                for et in CLAIM_TYPE_REQUIREMENTS.get(ct, [])
            })
            missing = sorted({
                et for ct in missing_types
                for et in CLAIM_TYPE_REQUIREMENTS.get(ct, [])
            })
            if missing:
                problems.append(
                    f"missing required evidence types: {sorted(set(missing))}"
                )
                evidence_gaps.append(EvidenceGap(
                    gap_id=f"gap_{claim_id}_coverage_{len(evidence_gaps)}",
                    candidate_id=conclusion.variant_id,
                    missing_link=(
                        f"claim type {missing_types} requires evidence types "
                        f"{sorted(set(missing))}"
                    ),
                    priority="critical",
                    recommended_evidence_types=sorted(set(missing)),
                    suggested_tools=["multi_query_literature_search"],
                ))

                conclusion.causal_claim_allowed = False

            if invented:
                status = "unsupported"
            elif problems:
                status = "partially_supported"
            else:
                status = "supported"
                accepted.append(claim_id)

            if claim_kind == "hypothesis" and not invented and not problems:
                status = "partially_supported"

            claim_reviews.append(ClaimReview(
                claim_id=claim_id,
                claim_kind=claim_kind,
                relation=claim_relation,
                evidence_state_level=(candidate_state.get("max_supported_level") if candidate_state else None),
                gate_decision=gate_result.get("decision", "UNKNOWN") if gate_result.get("decision") in {"PASS", "PARTIAL", "BLOCK"} else "UNKNOWN",
                status=status,
                evidence_ids_checked=checked,
                problems=problems,
                required_evidence_types=required,
                covered_evidence_types=covered,
            ))

        if ref_errors:
            decision = "REVISE_WITH_EXISTING_EVIDENCE"
            if any(g.priority == "critical" for g in evidence_gaps):
                decision = "RETRIEVE_MORE_EVIDENCE"
        elif evidence_gaps:
            decision = (
                "RETRIEVE_MORE_EVIDENCE"
                if any(g.priority == "critical" for g in evidence_gaps)
                else "REVISE_WITH_EXISTING_EVIDENCE"
            )
        elif not conclusions:
            decision = "INSUFFICIENT_EVIDENCE"
        elif causal_violations or factual_errors:
            decision = "REVISE_WITH_EXISTING_EVIDENCE"
        else:
            decision = "ACCEPT"

        report = CritiqueReport(
            decision=decision,
            claim_reviews=claim_reviews,
            evidence_gaps=evidence_gaps,
            causal_language_violations=causal_violations,
            factual_consistency_errors=factual_errors,
            evidence_reference_errors=ref_errors,
            accepted_hypothesis_ids=accepted,
        )

        

        if settings.hybrid_a4_enabled:
            try:
                feedback = self._semantic_feedback(
                    command, report, evidence_items, state_by_variant,
                )
            except Exception as exc:
                logger.exception("Hybrid semantic critic failed; retaining deterministic decision")
                feedback = CriticFeedback(
                    enabled=True, provider="deepseek_api", model=settings.hybrid_a4_model,
                    status="error", next_action="ACCEPT",
                    error=f"{type(exc).__name__}: {exc}",
                )
            report.hybrid_feedback = feedback
            self._merge_semantic_decision(report, feedback)

        increment("critic_decision", decision=report.decision)
        increment("critic_evidence_gaps", value=float(len(report.evidence_gaps)))
        if report.decision != "ACCEPT":
            increment("critic_reject", decision=report.decision)
        logger.info(
            "Evidence Critic: decision=%s claims=%d gaps=%d violations=%d",
            report.decision, len(report.claim_reviews), len(report.evidence_gaps), len(report.causal_language_violations),
        )
        return report
