import pytest

from phenofuncrank_repro.interpretation.evaluation.evaluate_decisions import evaluate


def test_missing_error_status_does_not_invent_error_denominator():
    result = evaluate([{"case_id": "c1", "condition": "A", "decision": "REVISE", "gold": "ACCEPT"}])
    assert result["conditions"]["A"]["exact_decision_accuracy"] == 0.0
    assert result["conditions"]["A"]["error_claim_count"] == 0
    assert result["conditions"]["A"]["error_claim_rejection_rate"] is None


@pytest.mark.parametrize("rows, message", [
    ([{"case_id": "c1", "condition": "A", "decision": "ACCEPT", "gold": "ACCEPT"}] * 2, "Duplicate case"),
    ([{"case_id": "c1", "condition": "A", "decision": "ACCEPT", "gold": "ACCEPT"}, {"case_id": "c2", "condition": "B", "decision": "ACCEPT", "gold": "ACCEPT"}], "identical case sets"),
    ([{"case_id": "c1", "condition": "A", "decision": "ACCEPT", "gold": "ACCEPT"}, {"case_id": "c1", "condition": "B", "decision": "ACCEPT", "gold": "REVISE"}], "identical reference labels"),
])
def test_pairing_does_not_silently_drop_or_relabel_cases(rows, message):
    with pytest.raises(ValueError, match=message):
        evaluate(rows, "A")
