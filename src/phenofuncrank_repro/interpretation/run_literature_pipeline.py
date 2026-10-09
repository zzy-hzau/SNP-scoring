"Connect the literature retrieval, extraction and reasoning stages."
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def run_pipeline(payload, output_dir: Path, *, mode="replay", run_id="paper_literature"):
    from phenofuncrank_repro.interpretation.retrieve import run_retrieval
    from phenofuncrank_repro.interpretation.extract_evidence import extract_evidence
    from phenofuncrank_repro.interpretation.literature_admission import proposals_to_evidence_items, _stamp_evidence_provenance
    from phenofuncrank_repro.interpretation.evidence_contract import admit_evidence_item, build_evidence_state
    from phenofuncrank_repro.interpretation.schemas.literature import CandidateContext, EvidenceGapIntent, EvidenceProposal
    from phenofuncrank_repro.interpretation.run_roles import run_roles

    candidate = CandidateContext(**payload["candidate_context"])
    responses = payload.get("model_responses", {})
    retrieval = run_retrieval(payload, output_dir / "retrieval", mode=mode)
    evidence_items = []
    extractions = {}
    for row in payload["gap_intents"]:
        gap = EvidenceGapIntent(**row)
        result = extract_evidence({
            "candidate_context": payload["candidate_context"],
            "gap_intent": row,
            "retrieval_hits": retrieval["source_result"]["gap_results"].get(gap.gap_id, []),
            "model_response": responses.get("extraction", {}).get(gap.gap_id),
        }, mode=mode)
        extractions[gap.gap_id] = result
        proposals = [EvidenceProposal(**record["proposal"]) for record in result["records"]]
        evidence_items.extend(proposals_to_evidence_items(
            proposals, gap, candidate, snp_id=candidate.snp_id,
            plan_gene_name=candidate.gene, species=candidate.species,
            reference_assembly=candidate.assembly, trait=candidate.trait,
        ))
    _stamp_evidence_provenance(
        evidence_items, run_id=run_id,
        variant_keys=payload.get("variant_keys", {}), species=candidate.species,
    )
    admitted = [admit_evidence_item(
        item, species=candidate.species, trait=candidate.trait
    )[0] for item in evidence_items]
    state = build_evidence_state(admitted, species=candidate.species, trait=candidate.trait)
    roles = run_roles({
        "species": candidate.species, "trait": candidate.trait,
        "evidence_state": state,
    }, output_dir / "roles", run_id=run_id,
        llm_mode="live" if mode == "live" else "disabled",
        model_responses=responses if mode == "replay" and responses else None)
    result = {
        "mode": mode, "extractions": extractions,
        "evidence_items": [item.model_dump(mode="json") for item in admitted],
        "evidence_state": state, "roles": roles,
    }
    (output_dir / "literature_pipeline_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("replay", "live"), default="replay")
    parser.add_argument("--run-id", default="paper_literature")
    args = parser.parse_args()
    os.environ["SENTRY_DSN"] = ""
    run_pipeline(json.loads(args.input.read_text(encoding="utf-8")), args.output_dir,
                 mode=args.mode, run_id=args.run_id)


if __name__ == "__main__":
    main()
