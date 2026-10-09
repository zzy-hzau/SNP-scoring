"Run A3 reasoning and A4 criticism on a supplied EvidenceState."
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def run_roles(payload, output_dir: Path, *, run_id: str, llm_mode="disabled", model_responses=None):
    if llm_mode != "live":
        pass
    from phenofuncrank_repro.interpretation.agents import hypothesis_reasoning as a3
    from phenofuncrank_repro.interpretation.agents import evidence_critic as a4
    from phenofuncrank_repro.interpretation.schemas.messages import MessageEnvelope
    output_dir.mkdir(parents=True, exist_ok=True)
    original_calls = a3.complete_json, a4.complete_json
    if llm_mode != "live":
        responses = model_responses or {}
        a3.complete_json = lambda *args, **kwargs: responses.get("a3")
        a4.complete_json = lambda *args, **kwargs: responses.get("a4")
    def message(recipient, content):
        return MessageEnvelope(
            message_id=f"{run_id}_{recipient}", trace_id=run_id, run_id=run_id,
            task_id=run_id, sender="supervisor", recipient=recipient,
            message_type="command", schema_version="1.0", idempotency_key=f"{run_id}_{recipient}",
            status="success", payload=content,
        )
    try:
        integration = a3.HypothesisReasoningAgent(output_dir / "reasoning").invoke(
            message("hypothesis_reasoning", payload))
        critic_payload = {**payload, "integration_result": integration.model_dump(mode="json")}
        critique = a4.EvidenceCriticAgent(output_dir / "critic").invoke(
            message("evidence_critic", critic_payload))
    finally:
        a3.complete_json, a4.complete_json = original_calls
    result = {
        "execution_mode": "live" if llm_mode == "live" else ("supplied_model_response_replay" if model_responses is not None else "source_no_response_fallback"),
        "integration_result": integration.model_dump(mode="json"),
        "critique_report": critique.model_dump(mode="json"),
        "scope": "Current preserved role source; historical Figure 6 metrics are recomputed by the frozen scoring modules.",
    }
    (output_dir / "roles_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON: species, trait and evidence_state; or evidence_items with --candidate-table")
    parser.add_argument("--candidate-table", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="paper_roles")
    parser.add_argument("--llm-mode", choices=("disabled", "live"), default="disabled")
    parser.add_argument("--model-responses", type=Path, help="Parsed JSON with a3 and a4 model responses; does not itself establish historical request provenance")
    args = parser.parse_args()
    if args.llm_mode == "live" and args.model_responses:
        parser.error("Choose live model calls or supplied response replay")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if "evidence_state" not in payload:
        if args.candidate_table is None:
            parser.error("--candidate-table is needed to construct EvidenceState")
        from phenofuncrank_repro.interpretation.evidence_contract import build_evidence_state
        payload["evidence_state"] = build_evidence_state(
            payload["evidence_items"], candidate_ref={"uri": str(args.candidate_table)},
            species=payload["species"], trait=payload["trait"])
    responses = json.loads(args.model_responses.read_text(encoding="utf-8")) if args.model_responses else None
    run_roles(payload, args.output_dir, run_id=args.run_id, llm_mode=args.llm_mode, model_responses=responses)

if __name__ == "__main__":
    main()
