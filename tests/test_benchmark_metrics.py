"""SOP §7 / test 11: four-layer benchmark metric calculators."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phenofuncrank_repro.interpretation.evaluation.metrics import (  
    aggregate_expert_scores,
    canonical_hit_at_k,
    canonical_paper_recall_at_k,
    canonical_recall_at_k,
    causal_recall_at_20,
    chain_correctness,
    citation_hallucination_rate,
    critic_decision_accuracy,
    evidence_leap_rate,
    evidence_completeness,
    evidence_precision_and_completeness,
    evidence_recall_at_k,
    gap_routing_accuracy,
    level3_penetration,
    pmid_accuracy,
    rejection_rate,
    validate_case,
)


def _case() -> dict:
    return {
        "case_id": "rice_PH_SD1_001",
        "species": "Oryza sativa",
        "trait": "plant height",
        "variant": {"chromosome": "Chr1", "position": 38384001, "gene": "SD1", "consequence": "missense"},
        "gold_evidence_chain": [
            {"layer": "variant_gene", "claim": "Variant affects SD1 protein function",
             "evidence_type": "functional_validation", "source": "PMID_11939564"},
            {"layer": "gene_trait", "claim": "SD1 regulates plant height",
             "evidence_type": "mutant_phenotype", "source": "PMID_11961544"},
        ],
        "gold_papers": {
            "evidence_levels": {
                "1": [], "2": ["11939564"], "3": ["11961544"], "4": [],
            },
            "irrelevant_neighbors": ["99999999"],
        },
        "required_evidence": [
            {"type": "variant_specific", "required": True},
            {"type": "gene_trait", "required": True},
            {"type": "mechanism", "required": False},
        ],
        "gold_causal_answer": True,
    }


def test_validate_case_ok_and_missing():
    assert validate_case(_case()) == []
    errors = validate_case({})
    assert any("missing fields" in e for e in errors)
    bad = _case()
    bad["required_evidence"] = [{"type": "variant_specific", "required": "yes"}]
    assert any("required_evidence.required must be a bool" in e for e in validate_case(bad))


def test_l1_recall_metrics():
    case = _case()
    records = [
        {"pmid": "11939564", "title": "sd-1 paper", "doi": "10.1/a"},
        {"pmid": "11961544", "title": "green revolution", "doi": "10.1/b"},
        {"pmid": "99999999", "title": "same family", "doi": "10.1/c"},
    ]
    assert canonical_recall_at_k(records, case, 1) is True
    assert canonical_recall_at_k(records[:1], case, 1) is True
    
    assert canonical_recall_at_k([records[1]], case, 1) is False
    assert evidence_recall_at_k([records[1]], case, 1) is True
    assert level3_penetration(records, case, 3) is True
    assert level3_penetration(records[:2], case, 2) is False


def test_canonical_hit_vs_paper_recall():
    case = _case()
    case["gold_papers"] = {
        "evidence_levels": {"1": ["111", "222"], "2": ["333"], "3": [], "4": []},
        "irrelevant_neighbors": [],
    }
    records = [
        {"pmid": "111", "title": "one"},
        {"pmid": "999", "title": "other"},
    ]
    assert canonical_hit_at_k(records, case, 5) is True
    assert canonical_recall_at_k(records, case, 5) is True  
    assert canonical_paper_recall_at_k(records, case, 5) == round(1 / 3, 4)
    assert canonical_paper_recall_at_k([], case, 5) == 0.0


def test_pmid_accuracy_and_hallucination():
    case = _case()
    records = [
        {"pmid": "11939564", "title": "sd-1 paper"},
        {"pmid": "00000000", "title": "SD1 regulates plant height", "level1_exists": False, "level2_match": False},
        {"pmid": "11111111", "title": "SD1 regulates plant height", "level1_exists": True, "level2_match": False},
    ]
    acc = pmid_accuracy(records, case)
    assert acc["pmid_accuracy"] == round(1 / 3, 4)
    assert acc["title_correct_pmid_wrong"] == 2
    assert citation_hallucination_rate(records) == round(2 / 3, 4)


def test_l2_precision_completeness():
    case = _case()
    proposals = [
        {"claim": "Variant affects SD1 protein function", "layer": "variant_gene"},
        {"claim": "unrelated claim", "layer": "mechanism"},
    ]
    out = evidence_precision_and_completeness(proposals, case)
    assert out["evidence_precision"] == 0.5
    assert out["covered_layers"] == 1
    assert out["evidence_completeness"] == 0.5


def test_evidence_leap_rate():
    claims = [
        {"steps": 3, "supported_steps": 3},
        {"steps": 3, "supported_steps": 1},
        {"steps": 2, "supported_steps": 0},
    ]
    
    assert evidence_leap_rate(claims) == 0.5
    assert evidence_leap_rate([]) == 0.0


def test_l3_chain_and_expert():
    claim = {"steps": {"variant_gene": 1, "gene_mechanism": 1, "mechanism_trait": 0}}
    assert chain_correctness(claim) == 2
    scores = [
        {"overinterpretation": 2, "biological_plausibility": 3, "evidence_completeness": 2},
        {"overinterpretation": 0, "biological_plausibility": 5, "evidence_completeness": 4},
    ]
    out = aggregate_expert_scores(scores)
    assert out["mean_overinterpretation"] == 1.0


def test_l4_rejection_and_accuracy():
    decisions = [
        {"decision": "REJECT", "gold": "REJECT"},
        {"decision": "ACCEPT", "gold": "REJECT"},
        {"decision": "REVISE", "gold": "REVISE"},
    ]
    assert rejection_rate(decisions) == round(2 / 3, 4)
    assert critic_decision_accuracy(decisions) == round(2 / 3, 4)


def test_a1_routing_and_causal_recall():
    assert gap_routing_accuracy(["gene_function", "trait_evidence"], ["gene_function", "trait_evidence", "regulatory"]) == round(2 / 3, 4)
    assert causal_recall_at_20(["v1", "v2", "v3"], ["v2", "v9"]) == 0.5


def test_evidence_completeness_required_layers():
    case = _case()
    
    records = [
        {"pmid": "1", "title": "Haplotype editing validates the variant effect on height"},
        {"pmid": "11939564", "title": "sd-1 mutant shows dwarf phenotype"},
    ]
    out = evidence_completeness(records, case)
    assert out["required_layers"] == ["variant_specific", "gene_trait"]
    assert out["covered_layers"] == ["variant_specific", "gene_trait"]
    assert out["completeness"] == 1.0


def test_evidence_completeness_missing_l1():
    case = _case()
    records = [{"pmid": "11939564", "title": "sd-1 mutant shows dwarf phenotype"}]
    out = evidence_completeness(records, case)
    assert out["missing_layers"] == ["variant_specific"]
    assert out["completeness"] == 0.5
    assert out["optional_covered"] == []
