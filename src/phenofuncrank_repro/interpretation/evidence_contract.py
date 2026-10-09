"Evidence admission, EvidenceState construction, and deterministic claim gate."

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from phenofuncrank_repro.interpretation.schemas.evidence_state import EvidenceState, EvidenceStateBundle, EvidenceStateEdge
from phenofuncrank_repro.interpretation.schemas.knowledge import EvidenceItem


_L1_TYPES = {
    "variant_specific", "variant_specific_natural_allele", "allele_specific",
    "natural_allele", "precise_editing", "variant_editing", "validated_eqtl",
    "eqtl", "eqtl_variant", "variant_linked_eqtl", "reporter_assay",
}
_L2_TYPES = {
    "functional_validation", "mutant_phenotype", "complementation", "rescue",
    "transgenic", "gene_trait", "gene_trait_association", "trait_association",
    "gwas", "qtl", "phenotype_association",
}
_L3_TYPES = {
    "molecular_mechanism", "mechanism", "pathway", "expression", "rna_seq",
    "regulatory", "regulatory_annotation", "promoter_motif", "tfbs_overlap",
    "motif_scan", "rna_structure", "protein_interaction", "protein_function",
}
_EXACT_VARIANT_CAUSAL_TYPES = {
    "precise_editing", "variant_editing", "variant_specific_editing",
    "allele_swap", "variant_functional_validation",
}
_COMPUTATIONAL_TYPES = {
    "model_effect", "functional_score", "cadd", "cadd_score", "dpc_effect",
    "prediction", "computational_prediction", "variant_context", "gene_context",
    "ld_proxy", "population_metric",
}
_CAUSAL_RE = re.compile(
    r"\b(caus(?:e|es|al|ative)|caused|determines?|proves?|confirms?|demonstrates?)\b"
    r"|导致|因果|证明|确定",
    re.I,
)

_RELATION_LEVELS = {
    "variant_functional_effect": "L1",
    "variant_trait": "L1",
    "variant_expression": "L1",
    "variant_mechanism": "L1",
    "variant_protein_function": "L1",
    "gene_trait": "L2",
    "gene_mechanism": "L3",
}


def _trait_terms(trait: str) -> list[str]:
    """Return conservative searchable forms for a benchmark/runtime trait."""
    text = re.sub(r"[_\-]+", " ", str(trait or "").strip().lower())
    text = re.sub(r"\s+", " ", text)
    if not text:
        return []
    terms = [text]
    words = [word for word in text.split() if len(word) >= 3]
    if len(words) > 1:
        terms.append(" ".join(words))
    return list(dict.fromkeys(terms))


def _claim_mentions_trait(claim: dict[str, Any], trait: str) -> bool:
    text = " ".join(
        str(claim.get(key) or "")
        for key in ("text", "trait", "trait_match", "phenotype")
    ).lower()
    return bool(text and any(term in text for term in _trait_terms(trait)))


def _claim_relations(raw: EvidenceItem | dict[str, Any], trait: str = "") -> list[str]:
    item = _as_dict(raw)
    claim = item.get("claim") if isinstance(item.get("claim"), dict) else {}
    explicit = claim.get("relations") or item.get("supported_relations") or []
    if isinstance(explicit, str):
        explicit = [explicit]
    relations = [str(value).strip().lower() for value in explicit if str(value).strip()]
    if relations:
        return list(dict.fromkeys(relations))

    supports = str(claim.get("supports") or "").strip().lower()
    scope = str(item.get("claim_scope") or "").strip().lower()
    level = str(item.get("proximity_level") or item.get("veib_evidence_level") or "").upper()
    evidence_type = str(item.get("evidence_type") or "").lower()
    if supports == "gene_trait":
        relations.append("gene_trait")
    elif supports == "mechanism":
        relations.append("variant_mechanism" if scope == "variant" and level == "L1" else "gene_mechanism")
    elif supports == "variant":
        relations.append("variant_functional_effect")

        

        if level == "L1" and scope == "variant" and _claim_mentions_trait(claim, trait):
            relations.append("variant_trait")
        if "express" in evidence_type or "expression" in str(claim.get("text") or "").lower():
            relations.append("variant_expression")
        if any(token in evidence_type for token in ("protein", "missense", "enzyme", "functional")):
            relations.append("variant_protein_function")

    

    if not relations and level == "L2" and str(item.get("subject_type") or "") == "gene":
        if evidence_type in _L2_TYPES or any(token in evidence_type for token in ("mutant", "trait", "phenotype", "complement")):
            relations.append("gene_trait")
    if not relations and level == "L3" and str(item.get("subject_type") or "") in {"gene", "protein", "pathway"}:
        if evidence_type in _L3_TYPES or any(token in evidence_type for token in ("pathway", "mechan", "regulat", "express")):
            relations.append("gene_mechanism")
    return list(dict.fromkeys(relations))


def is_affirmative_causal_claim(text: str) -> bool:
    "Detect affirmative causal wording."
    for clause in re.split(r"[.!?;\n]", str(text or "")):
        low = clause.lower().strip()
        if not _CAUSAL_RE.search(low):
            continue
        if re.search(r"\b(not|no|without|insufficient|unresolved|unknown|cannot|never|doesn't|does not|rather than|remains)\b", low):
            continue
        return True
    return False


def is_variant_level_claim(conclusion: dict[str, Any]) -> bool:
    "Return whether a claim makes a variant-level assertion."
    strength = str(conclusion.get("causal_strength") or "").lower()
    if strength in {"variant_level", "variant_causal"}:
        return True
    claim_type = str(conclusion.get("claim_type") or "").lower()
    if claim_type != "variant_interpretation" and any(
        token in claim_type for token in ("variant", "snp", "allele", "haplotype")
    ):
        return True
    text = str(conclusion.get("proposed_mechanism") or "").lower()
    return bool(re.search(r"\b(?:snp|variant|allele|haplotype|polymorphism)\b|变异|等位基因|单核苷酸", text))


def _source_type(raw: dict[str, Any]) -> str:
    explicit = str(raw.get("source_type") or "").strip().lower()
    if explicit in {"literature", "database", "computational", "population", "prediction", "annotation"}:
        return explicit
    et = str(raw.get("evidence_type") or "").strip().lower()
    source = str(raw.get("source_name") or "").lower()
    if et == "literature" or any(k in source for k in ("pubmed", "europepmc", "openalex", "semantic")):
        return "literature"
    if et in {"population_evidence", "population_metric", "trait_association", "gwas", "qtl"}:
        return "population"
    if et in _COMPUTATIONAL_TYPES or any(k in source for k in ("cadd", "dpc", "model", "fimo", "rnafold")):
        return "prediction" if et in {"prediction", "model_effect", "functional_score"} else "computational"
    if et in {"gene_context", "go", "gene_function", "pathway", "uniprot_mapping", "pfam_domain"}:
        return "database"

    

    

    
    if et in _L1_TYPES or et in _L2_TYPES or et in _L3_TYPES:
        if any(k in source for k in ("pubmed", "europepmc", "paper", "doi", "literature")):
            return "literature"
        return "database"
    return "unknown"


def _proximity_level(raw: dict[str, Any]) -> str:
    explicit = str(raw.get("proximity_level") or raw.get("veib_evidence_level") or "").upper()
    if explicit in {"L1", "L2", "L3", "L4"}:
        return explicit
    et = str(raw.get("evidence_type") or "").lower()
    if et in _L1_TYPES or "variant_specific" in et or "allele" in et:
        return "L1"
    if et in _L2_TYPES or any(k in et for k in ("mutant", "complement", "transgenic", "trait")):
        return "L2"
    if et in _L3_TYPES or any(k in et for k in ("pathway", "mechan", "regulat", "express")):
        return "L3"
    return "L4"


def _is_exact_variant_assay(item: EvidenceItem | dict[str, Any]) -> bool:
    raw = _as_dict(item)
    et = str(raw.get("evidence_type") or "").strip().lower()
    return (
        str(raw.get("proximity_level") or "").upper() == "L1"
        and str(raw.get("source_type") or "") not in {"computational", "prediction", "annotation"}
        and str(raw.get("directness") or "") == "direct"
        and str(raw.get("claim_support") or "") == "direct"
        and et in _EXACT_VARIANT_CAUSAL_TYPES
    )


def _as_dict(item: EvidenceItem | dict[str, Any]) -> dict[str, Any]:
    return item.model_dump(mode="json") if isinstance(item, EvidenceItem) else dict(item)


def admit_evidence_item(
    item: EvidenceItem | dict[str, Any], *, species: str = "", trait: str = "",
) -> tuple[EvidenceItem, str]:
    "Normalize one item and return ``(canonical_item, admission_status)``."
    raw = _as_dict(item)
    payload = dict(raw.get("payload") or {})
    provenance = dict(raw.get("provenance") or payload.get("provenance") or {})

    

    

    

    citation = payload.get("chunk_citation")
    if isinstance(citation, dict):
        article_id = str(citation.get("article_id") or "").strip()
        start = citation.get("start_char", 0)
        end = citation.get("end_char", 0)
        if article_id:
            provenance.setdefault(
                "chunk_id",
                f"{article_id}:{start}:{end}",
            )
        provenance.setdefault("article_id", article_id)
        provenance.setdefault("section", str(citation.get("section") or ""))
        provenance.setdefault("start_char", start)
        provenance.setdefault("end_char", end)
        provenance.setdefault("quote", str(citation.get("quote") or ""))
        provenance.setdefault("status", "valid")
        provenance.setdefault("source_name", str(payload.get("provider") or "literature_engine"))

    

    if not provenance.get("chunk_id") and payload.get("chunk_id"):
        provenance["chunk_id"] = str(payload.get("chunk_id"))
    if not provenance.get("quote") and payload.get("quote"):
        provenance["quote"] = str(payload.get("quote"))
    claim = raw.get("claim") if isinstance(raw.get("claim"), dict) else (
        payload.get("claim") if isinstance(payload.get("claim"), dict) else {}
    )
    raw.setdefault("evidence_id", str(raw.get("pack_item_id") or raw.get("proposal_id") or "evidence_unknown"))
    raw.setdefault("evidence_type", str(raw.get("evidence_class") or "unknown"))
    raw.setdefault("subject_type", "variant" if str(raw.get("claim_scope") or "") == "variant" else "gene")
    raw.setdefault("subject_id", str(raw.get("variant_key") or raw.get("gene") or raw.get("evidence_id") or "unknown"))
    raw.setdefault("directness", str(raw.get("veib_directness") or "computational_prediction"))
    raw["directness"] = {
        "direct_gene_level": "direct", "partial": "same_species_indirect",
        "indirect": "same_species_indirect",
    }.get(str(raw["directness"]), raw["directness"])
    raw.setdefault("supports", str(raw.get("stance") or "support").lower() in {"support", "yes", "supported"})
    raw.setdefault("stance", "contradict" if str(raw.get("stance") or "").lower() in {"contradict", "refute", "refuting"} else "support")
    raw.setdefault("statement", str(claim.get("text") or provenance.get("quote") or raw.get("claim_text") or "").strip())
    raw.setdefault("source_name", str(provenance.get("source_name") or "unknown_source"))
    raw.setdefault("source_record_id", str(provenance.get("source_record_id") or provenance.get("paper_id") or "") or None)
    raw.setdefault("species", species or "unknown")
    raw.setdefault("trait", trait or None)
    raw["payload"] = payload
    raw["provenance"] = provenance

    

    
    raw["claim"] = claim

    src_type = _source_type(raw)
    level = _proximity_level(raw)
    raw["source_type"] = src_type
    raw["proximity_level"] = level

    

    normalized_claim = dict(claim)
    normalized_claim["relations"] = _claim_relations(
        {**raw, "claim": normalized_claim, "proximity_level": level}, trait
    )
    raw["claim"] = normalized_claim

    verification = payload.get("verification") if isinstance(payload.get("verification"), dict) else {}
    verifier_support = str(verification.get("level3_support") or "").lower()
    explicit_support = str(raw.get("claim_support") or "").lower()
    if verifier_support == "no":
        semantic = "unsupported"
    elif verifier_support == "uncertain":
        semantic = "uncertain"
    elif verifier_support == "yes":
        semantic = "direct"
    elif explicit_support in {"direct", "partial", "unsupported", "uncertain"}:
        semantic = explicit_support
    elif (
        isinstance(claim, dict)
        and claim.get("supports")
        and str(raw.get("directness") or "") == "direct"
    ):

        

        
        semantic = "direct"
    elif (
        str(raw.get("evidence_type") or "").lower() in (_L1_TYPES | _L2_TYPES | _L3_TYPES)
        and str(raw.get("directness") or "") == "direct"
    ):

        

        semantic = "direct"
    elif str(raw.get("stance") or "").lower() == "contradict" and raw.get("statement"):

        
        semantic = "direct"
    elif raw.get("supports") is False or not raw.get("statement"):
        semantic = "unsupported"
    else:
        semantic = "partial"
    raw["claim_support"] = semantic

    source_name = str(raw.get("source_name") or "").strip()
    record_id = str(raw.get("source_record_id") or "").strip()
    statement = str(raw.get("statement") or "").strip()
    declared_provenance = str(raw.get("provenance_status") or "").lower()
    basic = bool(source_name and record_id and statement)
    if src_type == "literature":
        quote = str(provenance.get("quote") or payload.get("quote") or "").strip()
        chunk = str(provenance.get("chunk_id") or payload.get("chunk_id") or "").strip()
        source_status = str(provenance.get("status") or "").lower()
        exact = bool(quote and chunk and record_id and source_status in {"valid", "exact_source_checked"})
        prov_status = "valid" if exact else ("uncertain" if basic else "invalid")
    else:
        explicit_status = str(provenance.get("status") or "").lower()
        prov_status = "valid" if basic and (
            explicit_status in {"valid", "exact_source_checked"}
            or raw.get("query_hash") or raw.get("tool_call_id") or raw.get("raw_result")
        ) else ("uncertain" if source_name and statement else "invalid")

    

    

    if declared_provenance == "invalid":
        prov_status = "invalid"
    elif declared_provenance == "uncertain" and prov_status == "valid":
        prov_status = "uncertain"
    raw["provenance_status"] = prov_status

    if prov_status == "invalid" or semantic == "unsupported":
        status = "discarded"
    elif prov_status == "valid" and semantic in {"direct", "partial"}:
        status = "trusted"
    else:
        status = "provisional"
    raw["admission_status"] = status
    raw["evidence_level"] = {"L1": "A", "L2": "B", "L3": "C", "L4": "D"}[level]
    return EvidenceItem.model_validate(raw), status


def _candidate_rows(candidate_ref: Any) -> list[dict[str, Any]]:
    if not isinstance(candidate_ref, dict):
        return []
    uri = str(candidate_ref.get("uri") or "")
    if not uri or not Path(uri).is_file():
        return []
    try:
        df = pd.read_csv(uri, sep="\t")
    except Exception:
        try:
            df = pd.read_csv(uri)
        except Exception:
            return []
    return [dict(row) for row in df.to_dict(orient="records")]


def _clean_candidate_value(value: Any) -> Any:
    """Make a candidate-table cell safe for the EvidenceState JSON contract."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"nan", "none", "null", ""}:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _candidate_projection(row: dict[str, Any], *, species: str, trait: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Project deterministic model/annotation output into A1 EvidenceState.

    This is intentionally a small, typed view of the candidate artifact.  It
    preserves scores and identity needed for interpretation, while excluding
    the artifact URI and all raw tool tables from the A3/A4 handoff.
    """
    identity_fields = (
        "chrom", "pos", "ref", "alt", "variant_key", "canonical_key",
        "variant_type", "region_type", "consequence", "candidate_gene",
        "gene_id", "transcript_id", "protein_change", "gene_distance_bp",
        "cadd_gene", "cadd_consequence", "gene_function_note",
        "gene_go_terms", "gene_interpro", "protein_accession", "is_core",
        "level1_core", "ld_component_id", "core_snp_id", "proxy_snp_id",
        "phase_status", "alignment_status", "reference_assembly",
    )
    score_fields = (
        "pvi_score", "cadd_phred", "cadd_percentile", "cadd_prob",
        "T", "F", "B", "M", "priority_rank", "N", "call_rate", "MAC",
        "dpc_additive_effect_mean", "test_effect_mean", "abs_effect",
        "sample_direction_consistency", "r2_to_best_core", "r2_to_core",
        "He", "AF_cultivar", "phase_signed_r",
    )
    context: dict[str, Any] = {
        "species": species,
        "trait": trait,
    }
    for field in identity_fields + score_fields:
        if field in row:
            value = _clean_candidate_value(row.get(field))
            if value is not None:
                context[field] = value

    toolkit: list[dict[str, Any]] = []
    cadd = {field: context[field] for field in ("pvi_score", "cadd_phred", "cadd_percentile", "cadd_prob") if field in context}
    if cadd:
        toolkit.append({
            "tool": "PlantVImpact",
            "source_type": "computational",
            "metrics": cadd,
            "interpretation": "functional-impact prioritization only; not causal evidence",
        })
    dpc = {field: context[field] for field in (
        "dpc_additive_effect_mean", "test_effect_mean", "abs_effect",
        "sample_direction_consistency",
    ) if field in context}
    if dpc:
        toolkit.append({
            "tool": "DPCformer",
            "source_type": "computational",
            "metrics": dpc,
            "interpretation": "phenotype-direction prediction only; not causal evidence",
        })
    ld = {field: context[field] for field in (
        "ld_component_id", "r2_to_best_core", "r2_to_core", "phase_status",
    ) if field in context}
    if ld:
        toolkit.append({
            "tool": "LD/phase analysis",
            "source_type": "computational",
            "metrics": ld,
            "interpretation": "linkage/proxy context; not variant-specific biological evidence",
        })
    annotation = {field: context[field] for field in (
        "variant_type", "region_type", "consequence", "candidate_gene", "gene_id",
        "transcript_id", "protein_change", "gene_distance_bp", "gene_function_note",
    ) if field in context}
    if annotation:
        toolkit.append({
            "tool": "genomic annotation",
            "source_type": "annotation",
            "metrics": annotation,
            "interpretation": "genomic context only; not causal evidence",
        })
    return context, toolkit


def _edge(edge_id: str, source: str, relation: str, target: str, status: str,
          items: Iterable[EvidenceItem] = (), rationale: str = "") -> EvidenceStateEdge:
    rows = list(items)
    return EvidenceStateEdge(
        edge_id=edge_id, source=source, relation=relation, target=target,
        status=status,  
        evidence_ids=[str(x.evidence_id) for x in rows],
        evidence_levels=sorted({str(x.proximity_level) for x in rows}),
        directness=sorted({str(x.directness) for x in rows}),
        rationale=rationale,
    )


def build_evidence_state(
    items: Iterable[EvidenceItem | dict[str, Any]], *, candidate_ref: Any = None,
    species: str = "", trait: str = "",
) -> dict[str, Any]:
    """Create the A1-owned bundle consumed by A3 and A4."""
    candidates: dict[str, dict[str, Any]] = {}
    for row in _candidate_rows(candidate_ref):
        vid = str(row.get("user_snp_id") or row.get("snp_id") or row.get("variant_id") or "").strip()
        if vid:
            context, toolkit = _candidate_projection(row, species=species, trait=trait)
            candidates[vid] = {
                "gene_id": str(row.get("candidate_gene") or row.get("gene_id") or row.get("gene") or "").strip(),
                "variant_key": str(row.get("variant_key") or row.get("canonical_key") or "").strip(),
                "variant_context": context,
                "toolkit_evidence": toolkit,
            }

    trusted: list[EvidenceItem] = []
    provisional: list[EvidenceItem] = []
    discarded: list[EvidenceItem] = []
    for raw in items:
        try:
            item, status = admit_evidence_item(raw, species=species, trait=trait)
        except Exception:
            continue
        {"trusted": trusted, "provisional": provisional, "discarded": discarded}[status].append(item)

    if not candidates:
        for item in trusted + provisional + discarded:
            if item.subject_type == "variant" and item.subject_id:
                candidates.setdefault(item.subject_id, {
                    "gene_id": "", "variant_key": item.variant_key or "",
                    "variant_context": {"species": species, "trait": trait},
                    "toolkit_evidence": [],
                })
    if not candidates:
        candidates["__unresolved__"] = {"gene_id": "", "variant_key": ""}

    def matches(item: EvidenceItem, vid: str, meta: dict[str, str]) -> bool:
        text = f"{item.evidence_id} {item.statement} {item.variant_key or ''}".lower()
        if vid.lower() in text:
            return True
        key = meta.get("variant_key") or ""
        if key and key.lower() in text:
            return True
        if item.subject_type == "variant" and item.subject_id == vid:
            return True
        gene = meta.get("gene_id") or ""
        return bool(gene and item.subject_id.lower() == gene.lower())

    grouped: dict[str, list[EvidenceItem]] = defaultdict(list)
    for vid, meta in candidates.items():
        for item in trusted:
            if matches(item, vid, meta) or (len(candidates) == 1 and item.subject_type != "variant"):
                grouped[vid].append(item)

    states: dict[str, EvidenceState] = {}
    for vid, meta in candidates.items():
        current = grouped.get(vid, [])
        gene = meta.get("gene_id") or next((x.subject_id for x in current if x.subject_type == "gene"), "")
        provisional_current = [x for x in provisional if matches(x, vid, meta)]
        discarded_current = [x for x in discarded if matches(x, vid, meta)]

        

        
        eligible = [
            x for x in current
            if x.stance == "support" and x.claim_support in {"direct", "partial"}
        ]
        relation_items: dict[str, list[EvidenceItem]] = defaultdict(list)
        for item in eligible:
            for relation in _claim_relations(item, trait):
                relation_items[relation].append(item)

        l1_direct = [x for x in eligible if x.proximity_level == "L1" and x.directness == "direct"
                     and x.claim_support == "direct"]
        l1_exact = [x for x in l1_direct if _is_exact_variant_assay(x)]
        l1_any = [x for x in current if x.proximity_level == "L1"]
        l2 = [x for x in eligible if x.proximity_level == "L2"]
        l3 = [x for x in eligible if x.proximity_level == "L3"]
        contradictory = [x for x in current if x.stance == "contradict"]
        supported: list[EvidenceStateEdge] = []
        predicted: list[EvidenceStateEdge] = []
        missing: list[EvidenceStateEdge] = []

        relation_targets = {
            "variant_functional_effect": (vid, gene or "gene_unresolved"),
            "variant_trait": (vid, trait or "trait_unresolved"),
            "variant_expression": (vid, "expression"),
            "variant_mechanism": (vid, "mechanism"),
            "variant_protein_function": (vid, "protein_function"),
            "gene_trait": (gene or vid, trait or "trait_unresolved"),
            "gene_mechanism": (gene or vid, "mechanism"),
        }
        relation_rationales = {
            "variant_functional_effect": "Trusted semantic annotation supports a variant-level functional relation.",
            "variant_trait": "Trusted direct L1 annotation names the active variant and trait.",
            "variant_expression": "Trusted direct L1 annotation supports a variant-linked expression relation.",
            "variant_mechanism": "Trusted direct L1 annotation supports a variant-linked mechanism relation.",
            "variant_protein_function": "Trusted direct L1 annotation supports a variant-linked protein-function relation.",
            "gene_trait": "Trusted L2 annotation supports the gene-to-trait relation.",
            "gene_mechanism": "Trusted L3 annotation supports the gene-to-mechanism relation.",
        }
        for relation, relation_level in _RELATION_LEVELS.items():
            items_for_relation = relation_items.get(relation, [])

            

            if relation.startswith("variant_"):
                items_for_relation = [
                    item for item in items_for_relation
                    if item.proximity_level == "L1"
                    and item.claim_support == "direct"
                    and item.directness == "direct"
                ]
            direct_items = [
                item for item in items_for_relation
                if item.claim_support == "direct"
            ]
            partial_items = [
                item for item in items_for_relation
                if item.claim_support == "partial"
            ]
            source, target = relation_targets[relation]
            if direct_items:
                supported.append(_edge(
                    f"{source}:{relation}:{target}", source, relation, target,
                    "supported", direct_items, relation_rationales[relation],
                ))
            elif partial_items:

                

                
                predicted.append(_edge(
                    f"{source}:{relation}:{target}:partial", source, relation,
                    target, "predicted", partial_items,
                    "Trusted source/quote exists, but the extracted claim only partially supports this relation.",
                ))

        required_relations = (
            ("variant_functional_effect", vid, gene or "gene_unresolved"),
            ("gene_trait", gene or vid, trait or "trait_unresolved"),
            ("gene_mechanism", gene or vid, "mechanism"),
        )
        for relation, source, target in required_relations:
            if relation in {edge.relation for edge in supported}:
                continue
            relation_has_partial = any(
                item.claim_support == "partial"
                for item in relation_items.get(relation, [])
            )
            if relation == "variant_functional_effect" and l1_any and not relation_has_partial:
                predicted.append(_edge(
                    f"{source}:{relation}:{target}", source, relation, target,
                    "predicted", l1_any,
                    "Variant-level evidence exists but is indirect or not trusted as a direct functional relation.",
                ))
            missing.append(_edge(
                f"{source}:{relation}:{target}", source, relation, target,
                "missing", (),
                {
                    "variant_functional_effect": "No trusted semantic annotation establishes the variant-to-gene functional relation.",
                    "gene_trait": "No trusted semantic annotation establishes the gene-to-trait relation.",
                    "gene_mechanism": "No trusted semantic annotation establishes the gene-to-mechanism relation; pathway language remains provisional.",
                }[relation],
            ))

        supported_relations = [edge.relation for edge in supported]
        predicted_relations = [edge.relation for edge in predicted]
        missing_relations = [edge.relation for edge in missing]
        level_candidates = [
            str(item.proximity_level) for item in eligible
            if str(item.proximity_level) in {"L1", "L2", "L3", "L4"}
        ]
        max_level = min(level_candidates, key=lambda value: {"L1": 1, "L2": 2, "L3": 3, "L4": 4}[value]) if level_candidates else "L4"
        if "variant_trait" in supported_relations:
            maximum = "variant_trait_supported;_established_causality_requires_assay_scope_review"
        elif "variant_functional_effect" in supported_relations:
            maximum = "variant_specific_effect_supported;_variant_trait_causality_unresolved"
        elif "gene_trait" in supported_relations:
            maximum = "gene_trait_supported;_variant_causality_unresolved"
        elif "gene_mechanism" in supported_relations:
            maximum = "mechanistic_plausibility_only"
        elif current:
            maximum = "computational_or_background_evidence_only"
        else:
            maximum = "insufficient_evidence"
        forbidden = [] if l1_exact and "variant_trait" in supported_relations else ["variant_trait_causality"]
        if "gene_trait" not in supported_relations and "variant_trait" not in supported_relations:
            forbidden.append("gene_trait_established")
        states[vid] = EvidenceState(
            variant_id=vid, gene_id=gene, trait=trait,
            variant_context=dict(meta.get("variant_context") or {}),
            toolkit_evidence=list(meta.get("toolkit_evidence") or []),
            trusted_evidence_items=[x.model_dump(mode="json") for x in current],
            provisional_evidence_items=[x.model_dump(mode="json") for x in provisional_current],
            discarded_evidence_items=[x.model_dump(mode="json") for x in discarded_current],
            supported_edges=supported, predicted_edges=predicted, missing_edges=missing,
            supported_relations=sorted(set(supported_relations)),
            predicted_relations=sorted(set(predicted_relations)),
            missing_relations=sorted(set(missing_relations)),
            contradictory_evidence_ids=[x.evidence_id for x in contradictory],
            contradictory_evidence_notes=[x.statement[:240] for x in contradictory],
            forbidden_conclusions=forbidden, max_supported_level=max_level,
            max_supported_conclusion=maximum,
            admission_summary={"trusted": len(current), "provisional": len(provisional_current), "discarded": len(discarded_current)},
        )
    return EvidenceStateBundle(
        candidates=states,
        totals={"trusted": len(trusted), "provisional": len(provisional), "discarded": len(discarded)},
    ).model_dump(mode="json")


def evaluate_claim_gate(conclusion: dict[str, Any], state_raw: dict[str, Any] | None) -> dict[str, Any]:
    "Apply causal-boundary and claim-evidence entailment gates."
    state = state_raw or {}
    trusted = {str(x.get("evidence_id")): x for x in state.get("trusted_evidence_items", []) if x.get("evidence_id")}
    ids = [str(x) for x in (conclusion.get("supporting_evidence_ids") or [])]
    unknown = sorted(set(ids) - set(trusted))
    text = str(conclusion.get("proposed_mechanism") or "")
    causal = bool(conclusion.get("causal_claim_allowed")) or is_affirmative_causal_claim(text)
    variant_causal = causal and is_variant_level_claim(conclusion)
    l1_direct = [x for x in trusted.values() if str(x.get("proximity_level")) == "L1"
                 and str(x.get("source_type")) not in {"computational", "prediction", "annotation"}
                 and str(x.get("directness")) == "direct" and str(x.get("claim_support")) == "direct"]
    l1_exact = [x for x in l1_direct if _is_exact_variant_assay(x)]
    violations: list[str] = []
    if unknown:
        violations.append(f"unknown_or_untrusted_evidence_ids:{','.join(unknown)}")
    if variant_causal and not l1_exact:
        violations.append("variant_causal_claim_requires_exact_variant_assay")
    gene_id = str(state.get("gene_id") or "").strip().lower()
    text_lower = text.lower()
    mechanism_link_claim = bool(re.search(r"\b(?:through|via)\b", text_lower)) or bool(
        gene_id and gene_id in text_lower
    )
    if variant_causal and mechanism_link_claim and not any(
        str(edge.get("relation")) == "gene_trait" and str(edge.get("status")) == "supported"
        for edge in state.get("supported_edges", [])
    ):
        violations.append("variant_mechanism_claim_requires_supported_gene_trait_edge")
    entailment = evaluate_claim_evidence_entailment(conclusion, state)
    if entailment.get("decision") == "BLOCK":
        violations.extend(str(x) for x in entailment.get("violations", []))
    hard_block = bool(violations)
    if hard_block:
        decision = "BLOCK"
        status = "overbound"
    elif entailment.get("decision") == "PARTIAL":
        decision = "PARTIAL"
        status = "partially_grounded"
    else:
        decision = "PASS"
        status = "within_boundary"
    support_status = {
        "PASS": "supported",
        "PARTIAL": "partial",
        "BLOCK": "insufficient",
    }[decision]
    return {
        "decision": decision,
        "status": status,
        "support_status": support_status,
        "violations": violations,
        "missing_links": [
            f"{x.get('source', '')} -{x.get('relation', '')}-> {x.get('target', '')}"
            for x in state.get("missing_edges", [])
        ],
        "max_supported_conclusion": state.get("max_supported_conclusion", "insufficient_evidence"),
        "trusted_evidence_ids": sorted(trusted),
        "claim_evidence_entailment": entailment,
    }


_EPISTEMIC_GAP_RE = re.compile(
    r"\b(?:insufficient|unsupported|unresolved|unknown|unclear|lacks?\s+(?:(?:direct|sufficient)\s+)?evidence|not\s+established|cannot\s+establish)\b"
    r"|证据不足|尚未建立|尚不明确|无法确定|缺乏(?:直接)?证据|未建立|未得到支持",
    re.I,
)
_POSITIVE_ASSERTION_RE = re.compile(
    r"\b(?:may|might|could|influence|affect|through|via|regulate|associated|change|changes|support|supports|involved|participate)\b"
    r"|可能|影响|通过|经由|参与|调控|关联|相关|作用|改变|支持|涉及",
    re.I,
)
_MECHANISM_RE = re.compile(
    r"\b(?:mechanism|pathway|signaling|signalling|expression|protein|interaction|biosynthesis|hormone|through|via)\b"
    r"|机制|通路|信号|表达|蛋白|互作|生物合成|激素|通过|经由",
    re.I,
)


def _claim_is_epistemic_gap(text: str) -> bool:
    """True for a claim whose proposition is explicitly an evidence gap."""
    text = str(text or "").strip()
    if not text or not _EPISTEMIC_GAP_RE.search(text):
        return False

    

    

    
    if _POSITIVE_ASSERTION_RE.search(text):
        return False
    return not is_affirmative_causal_claim(text)


def _claim_atoms(conclusion: dict[str, Any], state: dict[str, Any]) -> list[dict[str, str]]:
    text = str(conclusion.get("proposed_mechanism") or "").strip()
    if _claim_is_epistemic_gap(text):
        return []
    low = text.lower()
    claim_type = str(conclusion.get("claim_type") or "").lower()
    explicit_relation = str(conclusion.get("relation") or "").lower()
    if explicit_relation in _RELATION_LEVELS:
        return [{
            "relation": explicit_relation,
            "description": explicit_relation.replace("_", " "),
        }]
    if claim_type in _RELATION_LEVELS:
        return [{
            "relation": claim_type,
            "description": claim_type.replace("_", " "),
        }]
    mechanism_claim = "mechanism" in claim_type
    atoms: list[dict[str, str]] = []
    variant_assertion = is_variant_level_claim(conclusion)
    if variant_assertion:
        atoms.append({
            "relation": "variant_functional_effect",
            "description": "variant → gene functional effect",
        })

    

    
    gene_id = str(state.get("gene_id") or "").strip().lower()
    gene_level_words = (
        "gene_trait", "gene-level", "gene level", "knockout", "mutant",
        "overexpression", "complementation", "regulates", "regulate",
        "gene affects", "gene controls", "gene causes",
        "基因层面", "基因敲除", "突变体", "过表达", "调控",
    )
    if "gene_trait" in claim_type or (
        gene_id and gene_id in low and any(word in low for word in gene_level_words)
    ) or (not variant_assertion and not mechanism_claim and gene_id and gene_id in low):
        atoms.append({"relation": "gene_trait", "description": "gene → trait"})
    if "mechanism" in claim_type or _MECHANISM_RE.search(text):
        atoms.append({"relation": "gene_mechanism", "description": "gene → molecular mechanism/pathway"})

    

    if re.search(r"\b(?:through|via)\b|通过|经由", text, re.I):
        atoms.append({
            "relation": "mechanism_mediates_gene_trait",
            "description": "mechanism/pathway mediates gene → trait",
        })

    unique: dict[str, dict[str, str]] = {}
    for atom in atoms:
        unique.setdefault(atom["relation"], atom)
    return list(unique.values())


def _item_claim_annotation(item: dict[str, Any]) -> dict[str, Any]:
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    claim = item.get("claim") if isinstance(item.get("claim"), dict) else {}
    if not claim:
        claim = payload.get("claim") if isinstance(payload.get("claim"), dict) else {}
    return claim


def _item_relation_support(item: dict[str, Any], relation: str, state: dict[str, Any]) -> str:
    if str(item.get("stance") or "support") in {"contradict", "neutral", "uncertain"}:
        return "unsupported"
    claim = _item_claim_annotation(item)
    declared = str(claim.get("supports") or "").lower()
    if relation == "variant_functional_effect":
        declared_ok = declared == "variant" or item.get("supports_variant") is True
        typed_ok = str(item.get("proximity_level") or "") == "L1" and (
            str(item.get("claim_scope") or "") == "variant"
            or str(item.get("subject_type") or "") == "variant"
        )
    elif relation in {
        "variant_trait", "variant_expression", "variant_mechanism",
        "variant_protein_function",
    }:
        declared_ok = relation in _claim_relations(item, str(state.get("trait") or ""))
        typed_ok = str(item.get("proximity_level") or "") == "L1" and (
            str(item.get("claim_scope") or "") == "variant"
            or str(item.get("subject_type") or "") == "variant"
        ) and declared_ok
    elif relation == "gene_trait":
        declared_ok = declared == "gene_trait" or item.get("supports_gene_trait") is True
        typed_ok = str(item.get("proximity_level") or "") == "L2" and str(item.get("subject_type") or "") == "gene"
    elif relation == "gene_mechanism":
        declared_ok = declared == "mechanism" or item.get("supports_mechanism") is True
        typed_ok = str(item.get("proximity_level") or "") == "L3" and str(item.get("subject_type") or "") in {"gene", "protein", "pathway"}
    elif relation == "mechanism_mediates_gene_trait":

        
        declared_ok = declared in {"mechanism_gene_trait", "mechanism_to_trait", "mediates_gene_trait"}
        typed_ok = False
    else:
        return "unsupported"
    if not (declared_ok or typed_ok):
        return "unsupported"
    support = str(item.get("claim_support") or "partial").lower()
    if support == "direct":
        return "supported"
    if support == "partial":
        return "partial"
    return "unsupported"


def evaluate_claim_evidence_entailment(
    conclusion: dict[str, Any], state_raw: dict[str, Any] | None,
) -> dict[str, Any]:
    """Check whether cited trusted items entail each typed claim atom.

    The result is intentionally tri-valued.  ``PARTIAL`` is useful for a
    hypothesis that has some supported edges but an unresolved variant link;
    ``BLOCK`` is used for an affirmative claim whose required relation has no
    cited support at all.
    """
    state = state_raw or {}
    trusted = {
        str(x.get("evidence_id")): x
        for x in state.get("trusted_evidence_items", [])
        if x.get("evidence_id")
    }
    cited = [str(x) for x in (conclusion.get("supporting_evidence_ids") or [])]
    atoms = _claim_atoms(conclusion, state)
    if not atoms:
        has_gap = bool(state.get("missing_edges") or state.get("forbidden_conclusions"))
        return {
            "decision": "BLOCK" if has_gap or _claim_is_epistemic_gap(str(conclusion.get("proposed_mechanism") or "")) else "PARTIAL",
            "status": "gap_disclosure" if has_gap else "no_typed_atom_detected",
            "atoms": [],
            "violations": ["gap_disclosure_is_not_evidence_supported_claim"] if has_gap else [],
            "supported_relations": [],
            "partial_relations": [], "unsupported_relations": [],
            "cited_evidence_ids": cited,
        }

    supported_relations: list[str] = []
    partial_relations: list[str] = []
    unsupported_relations: list[str] = []
    atom_results: list[dict[str, Any]] = []
    for atom in atoms:
        relation = atom["relation"]
        statuses = [
            _item_relation_support(trusted[eid], relation, state)
            for eid in cited if eid in trusted
        ]
        edge_supported = any(
            str(edge.get("relation")) == relation
            and str(edge.get("status")) == "supported"
            and bool(set(cited).intersection(str(x) for x in (edge.get("evidence_ids") or [])))
            for edge in state.get("supported_edges", []) or []
        )
        if edge_supported or "supported" in statuses:
            result = "supported"
            supported_relations.append(relation)
        elif "partial" in statuses:
            result = "partial"
            partial_relations.append(relation)
        else:
            result = "unsupported"
            unsupported_relations.append(relation)
        atom_results.append({
            "relation": relation, "description": atom["description"],
            "result": result,
            "cited_evidence_ids": [eid for eid in cited if eid in trusted],
        })

    affirmative = bool(conclusion.get("causal_claim_allowed")) or is_affirmative_causal_claim(
        str(conclusion.get("proposed_mechanism") or "")
    )
    if unsupported_relations:
        decision = "BLOCK" if affirmative or not supported_relations else "PARTIAL"
    elif partial_relations:
        decision = "PARTIAL"
    else:
        decision = "PASS"
    violations = [
        f"claim_evidence_entailment_failed:{relation}"
        for relation in unsupported_relations
    ]
    return {
        "decision": decision,
        "status": "entailed" if decision == "PASS" else "partially_entailed" if decision == "PARTIAL" else "not_entailed",
        "atoms": atom_results,
        "violations": violations,
        "supported_relations": supported_relations,
        "partial_relations": partial_relations,
        "unsupported_relations": unsupported_relations,
        "cited_evidence_ids": cited,
    }
