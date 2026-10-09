"A2 literature-proposal projection and provenance stamping."
from __future__ import annotations
import hashlib
from phenofuncrank_repro.interpretation.schemas.knowledge import EvidenceItem
from phenofuncrank_repro.interpretation.variant_evidence_verifier import apply_verification, verify_proposal, verification_payload

_EVIDENCE_TOOL_PREFIXES = {
    "blastn_context_": "blastn_context",
    "fimo_": "fimo_motif_scan",
    "rnafold_": "rnafold_structure_prediction",
    "motif_": "jaspar_motif_scan",
    "jaspar_local_": "jaspar_local_pwm_scan",
    "tfbs_": "plantregmap_genome_tfbs_overlap",
    "tfden_": "promoter_tfbs_density",
    "ptdb_motif_": "planttfdb_motif_scan",
    "blast_flank_": "quick_blastn",
    "alphafold_": "alphafold_structure_lookup",
    "local_": "local_protein_evidence",
}

def _stamp_evidence_provenance(
    items: list[EvidenceItem],
    *,
    run_id: str,
    variant_keys: dict[str, str],
    species: str,
) -> list[EvidenceItem]:
    """Fill query_hash / tool_call_id / variant_key for every evidence item."""
    for item in items:
        eid = item.evidence_id or ""
        sid = item.subject_id or ""
        if item.subject_type == "variant" and sid in variant_keys:
            item.variant_key = variant_keys[sid]
        if item.query_hash and item.tool_call_id:
            continue
        if item.evidence_type == "gene_context" or eid.startswith("gff_"):
            item.tool_call_id = f"gene_ctx:{run_id}:{sid or eid}"
            item.query_hash = hashlib.sha256(f"gff:{species}:{sid}".encode()).hexdigest()
        elif item.evidence_type == "literature":
            item.tool_call_id = f"lit:{run_id}:{sid or eid}"
            item.query_hash = hashlib.sha256(f"lit:{species}:{sid}".encode()).hexdigest()
        else:
            for prefix, tool in _EVIDENCE_TOOL_PREFIXES.items():
                if eid.startswith(prefix):
                    item.tool_call_id = f"{tool}:{run_id}:{sid or eid}"
                    item.query_hash = hashlib.sha256(
                        f"{tool}:{species}:{sid or eid}".encode()
                    ).hexdigest()
                    break
            if not item.tool_call_id:
                item.tool_call_id = f"ev:{run_id}:{eid or 'unknown'}"
                item.query_hash = hashlib.sha256(
                    f"{item.evidence_type}:{species}:{sid or eid}".encode()
                ).hexdigest()
    return items

def proposals_to_evidence_items(proposals,gap_intent,candidate_ctx,*,snp_id,plan_gene_name,species,reference_assembly,trait):
    evidence_items=[]
    for raw_prop in proposals:
        verification = verify_proposal(raw_prop, gap_intent, candidate_ctx)

        

        prop = apply_verification(raw_prop, verification)
        target_relation = str(gap_intent.missing_link or "").strip()
        supported_relation = str(prop.supported_relation or "").strip()
        relation_exact = bool(
            prop.relation_match == "exact"
            and supported_relation == target_relation
        )
        relation = supported_relation if supported_relation else ""
        variant_scoped = relation.startswith("variant_")
        claim_scope = "variant" if variant_scoped else "gene"
        if relation_exact and prop.support == "direct":
            claim_support = "direct"
        elif prop.relation_match == "partial":
            claim_support = "partial"
        else:
            claim_support = "unsupported"
        source_record_id = str(
            prop.source.get("pmid") or prop.source.get("doi") or ""
        )
        citation = prop.chunk.model_dump()
        chunk_id = (
            f"{citation.get('article_id', '')}:"
            f"{citation.get('start_char', 0)}:"
            f"{citation.get('end_char', 0)}"
        )
        evidence_items.append(EvidenceItem(
            evidence_id=f"liteng_{prop.proposal_id[:16]}",
            subject_type="variant" if variant_scoped else "gene",
            subject_id=snp_id if variant_scoped else plan_gene_name,
            evidence_type=prop.evidence_type,
            source_name="literature_engine",
            source_record_id=source_record_id,
            species=species,
            reference_assembly=reference_assembly,
            statement=prop.claim,
            directness=(
                "direct" if relation_exact and prop.directness == "direct"
                else "orthology_inference" if prop.species_match == "ortholog"
                else "same_species_indirect"
            ),
            evidence_level="B" if prop.confidence >= 0.8 else "C",
            proximity_level=verification.effective_level,
            supports=claim_support != "unsupported",
            trait=trait,
            claim_scope=claim_scope,
            claim_support=claim_support,
            source_type="literature",
            claim={
                "relations": [relation] if relation else [],
                "supports": relation,
                "text": prop.claim,
                "trait_match": trait,
                "target_relation": target_relation,
                "relation_match": prop.relation_match,
                "supported_relation": supported_relation,
                "gene": prop.gene,
                "variant_or_allele": prop.variant_or_allele,
                "variant_mention": prop.variant_mention.model_dump(mode="json"),
                "variant_verification": verification_payload(verification),
            },
            provenance={
                "paper_id": source_record_id,
                "paper_id_type": "PMID" if prop.source.get("pmid") else "DOI",
                "source_name": "literature_engine",
                "source_record_id": source_record_id,
                "chunk_id": chunk_id,
                "article_id": citation.get("article_id", ""),
                "section": citation.get("section", ""),
                "start_char": citation.get("start_char", 0),
                "end_char": citation.get("end_char", 0),
                "quote": citation.get("quote", ""),
                "status": "valid",
                "verification": "single_chunk_quote_checked",
                "title": prop.source.get("title", ""),
            },
            payload={
                "title": prop.source.get("title", ""),
                "doi": prop.source.get("doi", ""),
                "pmid": prop.source.get("pmid", ""),
                "chunk_citation": prop.chunk.model_dump(),
                "confidence": prop.confidence,
                "verification": verification_payload(verification),
                "direction": prop.direction,
                "gap_id": prop.gap_id,
                "evidence_type": prop.evidence_type,
                "target_relation": target_relation,
                "supported_relation": supported_relation,
                "variant_mention": prop.variant_mention.model_dump(mode="json"),
                "relation_match": prop.relation_match,
                "verification_status": prop.verification_status,
                "provider": "literature_engine",
            },
        ))
    return evidence_items
