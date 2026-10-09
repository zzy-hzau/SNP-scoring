"Run the evidence extractor and identity/relation verifier."
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def extract_evidence(payload, *, mode="replay"):
    from phenofuncrank_repro.interpretation import evidence_extractor as extractor
    from phenofuncrank_repro.interpretation import variant_evidence_verifier as verifier
    from phenofuncrank_repro.interpretation.schemas.literature import CandidateContext, EvidenceGapIntent, RetrievalHit

    candidate = CandidateContext(**payload["candidate_context"])
    gap = EvidenceGapIntent(**payload["gap_intent"])
    hits = [RetrievalHit(**row) for row in payload["retrieval_hits"]]
    original_call = extractor.complete_json
    if mode == "replay":
        response = payload["model_response"]
        extractor.complete_json = lambda *args, **kwargs: response
    try:
        proposals, stats = extractor.extract_proposals(hits, gap, candidate)
    finally:
        extractor.complete_json = original_call

    records = []
    for proposal in proposals:
        result = verifier.verify_proposal(proposal, gap, candidate)
        records.append({
            "proposal": proposal.model_dump(mode="json"),
            "verification": verifier.verification_payload(result),
            "after_verification": verifier.apply_verification(
                proposal, result
            ).model_dump(mode="json"),
        })
    return {"mode": mode, "records": records, "stats": stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("replay", "live"), default="replay")
    args = parser.parse_args()
    os.environ["SENTRY_DSN"] = ""
    result = extract_evidence(
        json.loads(args.input.read_text(encoding="utf-8")), mode=args.mode
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
