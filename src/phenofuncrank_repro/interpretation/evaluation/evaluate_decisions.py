"Evaluate decisions."
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from phenofuncrank_repro.interpretation.evaluation.metrics import critic_decision_accuracy, rejection_rate
from phenofuncrank_repro.interpretation.evaluation.stats import mcnemar, paired_bootstrap_ci


def evaluate(records: list[dict], reference_condition: str | None = None) -> dict:
    groups: dict[str, dict[str, dict]] = {}
    for row in records:
        condition, case_id = str(row["condition"]), str(row["case_id"])
        if not row["decision"] or not row["gold"]:
            raise ValueError("A decision and explicit reference label are required")
        if case_id in groups.setdefault(condition, {}):
            raise ValueError(f"Duplicate case {case_id} in condition {condition}")
        groups[condition][case_id] = row
    output = {"conditions": {}, "paired_comparisons": []}
    for condition, keyed in groups.items():
        rows = list(keyed.values())
        errors = [row for row in rows if row.get("error_claim") is True]
        output["conditions"][condition] = {
            "cases": len(rows),
            "exact_decision_accuracy": critic_decision_accuracy(rows),
            "error_claim_count": len(errors),
            "error_claim_rejection_rate": rejection_rate(errors) if errors else None,
        }
    if reference_condition is not None:
        baseline = groups[reference_condition]
        for condition, keyed in groups.items():
            if condition == reference_condition:
                continue
            if set(keyed) != set(baseline):
                raise ValueError("Paired comparisons require identical case sets")
            case_ids = sorted(keyed)
            if any(keyed[k]["gold"] != baseline[k]["gold"] for k in case_ids):
                raise ValueError("Paired comparisons require identical reference labels")
            baseline_hits = [str(baseline[k]["decision"]).upper() == str(baseline[k]["gold"]).upper() for k in case_ids]
            hits = [str(keyed[k]["decision"]).upper() == str(keyed[k]["gold"]).upper() for k in case_ids]
            output["paired_comparisons"].append({
                "condition": condition,
                "baseline": reference_condition,
                "paired_cases": len(case_ids),
                "bootstrap": paired_bootstrap_ci(list(zip(hits, baseline_hits))),
                "mcnemar": mcnemar(hits, baseline_hits),
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True, help="JSONL: case_id, condition, decision, gold; optional explicit error_claim boolean")
    parser.add_argument("--reference-condition", help="Baseline for paired comparisons")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = evaluate(records, args.reference_condition)
    result["input_sha256"] = hashlib.sha256(args.records.read_bytes()).hexdigest()
    result["scope"] = "Frozen decision-label replay; not live agent or semantic validation"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
