"""Real published cases exercise exclusions, uncertain labels and partial scores."""
import json
from pathlib import Path
from phenofuncrank_repro.interpretation.evaluation.score_frozen_figure6 import score, score_refinement

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    records = json.loads((ROOT/'examples/data/benchmarks/figure6_small_real_scoring_example.json').read_text())
    protocol = json.loads((ROOT/'configs/benchmarks/figure6_protocol_20261003.json').read_text())
    return records, protocol


def test_real_reasoning_admission_retains_excluded_and_unresolved_cases():
    records, protocol = inputs()
    result = score(records, protocol)
    pairs = result['panels']['reasoning']
    assert pairs['eligible_generation_pairs'] == 9
    assert pairs['admitted_generation_pairs'] == 7
    excluded = {(row['case_id'], row['generation']) for row in pairs['inclusion_decisions'] if not row['included']}
    assert excluded == {('R003', 1), ('R003', 2), ('R003', 3), ('R014', 3), ('R020', 1)}
    assert all(row['stored_inclusion_matches'] for row in pairs['inclusion_decisions'])
    assert not any(result['consensus_differences'].values())


def test_critic_cohort_uses_construction_group_and_preserves_unknown_reference():
    records, protocol = inputs()
    result = score(records, protocol)['panels']['critic']
    assert 'C061' in result['challenge_cases']
    raw = next(r for r in records['critic_consensus'] if r['case_id'] == 'C061' and r['condition'] == 'M1')
    assert raw['scores']['reference_state'] == 'NEEDS_HUMAN_REVIEW'
    assert 'C001' not in result['challenge_cases']
    assert set(result['control_cases']) == {'C101', 'C111', 'C119', 'C120'}
    assert result['methods']['M1']['action_appropriate']['denominator'] == 7


def test_partial_real_refinement_rating_is_not_full_grounding_success():
    records, protocol = inputs()
    for key in ['refinement_consensus', 'refinement_judges']:
        records[key] = [r for r in records[key] if r['case_id'] == 'E001']
    result, differences, warnings = score_refinement(records, protocol)
    metric = result['metrics']['evidence_grounding']
    assert result['paired_cases'] == 1
    assert metric['t0_yes']['numerator'] == 0
    assert metric['t1_yes']['numerator'] == 1
    assert metric['paired_t1_better'] == 1
    assert not differences and not warnings


