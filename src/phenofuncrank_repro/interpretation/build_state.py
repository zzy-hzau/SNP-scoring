"Build the evidence-state contract from caller-provided JSON data."
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from phenofuncrank_repro.interpretation.evidence_contract import build_evidence_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON: evidence_items, species, trait")
    parser.add_argument("--candidate-table", type=Path, required=True, help="Candidate TSV consumed by the original ArtifactRef contract")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    state = build_evidence_state(
        payload["evidence_items"], candidate_ref={"uri": str(args.candidate_table)},
        species=payload["species"], trait=payload["trait"],
    )
    result = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "candidate_table_sha256": hashlib.sha256(args.candidate_table.read_bytes()).hexdigest(),
        "scope": "Deterministic evidence-state construction; no new literature retrieval or model generation",
        "evidence_state": state,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
