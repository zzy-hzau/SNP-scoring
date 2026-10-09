"Calculate interpretation metrics from supplied judge labels."
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from phenofuncrank_repro.interpretation.evaluation.figure6_native_primitives import frac, majority, metric, paired_metric, rate

REVIEW = 'NEEDS_HUMAN_REVIEW'
REASON_QUALITY = ('grounding', 'missing_link_detection', 'conclusion_calibration')
REASON_ERRORS = ('evidence_leap', 'unsupported_causal_upgrade', 'evidence_id_error')
REFINE_QUALITY = ('conclusion_correct', 'causal_calibration', 'evidence_grounding', 'missing_link_handling', 'max_defensible_conclusion')


def key(row):
    return row['case_id'], row.get('generation'), row['condition']


def voting(judges):
    grouped = defaultdict(dict)
    for row in judges:
        identity, judge_pass = key(row), row['judge_pass']
        if judge_pass not in (1, 2, 3) or judge_pass in grouped[identity]:
            raise ValueError('Judge passes must identify distinct original passes 1,2,3')
        grouped[identity][judge_pass] = row['scores']
    result, warnings = {}, []
    for identity, passes in grouped.items():
        fields = set().union(*(p.keys() for p in passes.values()))
        result[identity] = {field: majority([passes.get(i, {}).get(field, REVIEW) for i in (1, 2, 3)]) for field in fields}
        if set(passes) != {1, 2, 3}: warnings.append({'identity': list(identity), 'missing_judge_passes': sorted({1, 2, 3} - set(passes))})
    return result, warnings


def compare_consensus(stored, calculated):
    mismatches = []
    for row in stored:
        observed = calculated.get(key(row), {})
        for field, value in row['scores'].items():
            if observed.get(field, REVIEW) != value:
                mismatches.append({'case_id': row['case_id'], 'generation': row.get('generation'), 'condition': row['condition'], 'field': field, 'stored_label': value, 'recomputed_label': observed.get(field, REVIEW), 'source_row': row['_source_row']})
    return mismatches


def score_reasoning(records, protocol):
    calculated, warnings = voting(records['reasoning_judges'])
    grouped = defaultdict(dict)
    for row in records['reasoning_consensus']:
        identity = row['case_id'], row['generation']
        if row['condition'] in grouped[identity]: raise ValueError('Duplicate reasoning condition')
        grouped[identity][row['condition']] = row
    admitted, decisions = [], []
    excluded = set(protocol['reasoning_excluded_cases'])
    fields = REASON_QUALITY + REASON_ERRORS
    for (case_id, generation), pair in sorted(grouped.items()):
        if set(pair) != {'vanilla', 'full'}: raise ValueError('Paired reasoning requires both recorded conditions')
        values = {arm: calculated.get(key(row), {f: REVIEW for f in fields}) for arm, row in pair.items()}
        complete = all(values[arm].get(field, REVIEW) != REVIEW for arm in pair for field in fields)
        included = case_id not in excluded and complete
        reason = 'source-defined excluded task' if case_id in excluded else 'no complete two-of-three consensus' if not complete else None
        decisions.append({'case_id': case_id, 'generation': generation, 'included': included, 'reason': reason, 'stored_inclusion_matches': all(row['included_in_paired_analysis'] == included for row in pair.values())})
        if included:
            admitted.append({'arm_order_private': {'public_full': 'full', 'public_vanilla': 'vanilla'}, 'consensus_blind': {'public_full': values['full'], 'public_vanilla': values['vanilla']}})
    result = {'case_count_after_task_exclusion': len({case for case, _ in grouped if case not in excluded}), 'eligible_generation_pairs': sum(case not in excluded for case, _ in grouped), 'admitted_generation_pairs': len(admitted),
        'metrics': {**{f: metric(admitted, f, error=False) for f in REASON_QUALITY}, **{f: metric(admitted, f, error=True) for f in REASON_ERRORS}},
        'inclusion_decisions': decisions}
    return result, compare_consensus(records['reasoning_consensus'], calculated), warnings


def score_critic(records, protocol):
    calculated, warnings = voting(records['critic_judges'])
    definitions = {row['case_id']: row for row in records['critic_definitions']}
    if len(definitions) != len(records['critic_definitions']): raise ValueError('Duplicate critic definition')
    grouped = defaultdict(dict)
    for row in records['critic_consensus']:
        if row['condition'] in grouped[row['case_id']]: raise ValueError('Duplicate critic condition')
        grouped[row['case_id']][row['condition']] = calculated.get(key(row), {})
    excluded = set(protocol['critic_excluded_cases'])
    challenges, controls = [], []
    for case_id, definition in definitions.items():
        if case_id in excluded: continue
        if definition['analysis_subset'] == 'Constructed challenge': challenges.append(case_id)
        elif definition['analysis_subset'] == 'Supported control': controls.append(case_id)
        else: continue
        if set(grouped[case_id]) != {'M1', 'M2'}: raise ValueError('Critic pairing requires both recorded methods')
    included = challenges + controls
    contradiction = [case for case in challenges if definitions[case]['category'].replace('_', ' ').lower() == 'contradiction unresolved']
    partial = [case for case in challenges if grouped[case]['M1'].get('reference_state') == 'PARTIALLY_SUPPORTED']
    reference_disagreement = [case for case in included if any(grouped[case]['M1'].get(f) != grouped[case]['M2'].get(f) for f in ('reference_state', 'reference_action'))]
    result = {'challenge_cases': challenges, 'control_cases': controls, 'contradiction_cases': contradiction, 'partial_support_cases': partial,
        'reference_disagreement_cases': reference_disagreement, 'methods': {}}
    specifications = {'action_appropriate': (challenges, 'action_appropriate', 'yes'), 'semantic_error_detected': (challenges, 'semantic_error_detected', 'yes'), 'supported_claim_retained': (controls, 'supported_retained', 'yes'),
        'guardrail_nonretention': (controls, 'supported_retained', 'no'), 'contradiction_detected': (contradiction, 'contradiction_detected', 'yes'), 'partial_support_calibrated': (partial, 'partial_calibrated', 'yes')}
    for condition in ('M1', 'M2'):
        result['methods'][condition] = {}
        for name, (subset, field, success) in specifications.items():
            values = [grouped[c][condition].get(field, REVIEW) for c in subset]
            statistic = frac(sum(v == success for v in values), len(subset))
            statistic['native_rate_on_usable_labels'] = rate(values, success)
            statistic['review_labels'] = sum(v == REVIEW for v in values)
            result['methods'][condition][name] = statistic
    result['excluded_case_details'] = [{'case_id': c, 'present_in_public_records': c in grouped, 'excluded_from_summary': True} for c in sorted(excluded)]
    return result, compare_consensus(records['critic_consensus'], calculated), warnings


def score_refinement(records, protocol):
    calculated, warnings = voting(records['refinement_judges'])
    grouped = defaultdict(dict)
    for row in records['refinement_consensus']:
        if row['condition'] in grouped[row['case_id']]: raise ValueError('Duplicate refinement stage')
        grouped[row['case_id']][row['condition']] = calculated.get(key(row), {})
    excluded = set(protocol['refinement_excluded_cases'])
    admitted, decisions = [], []
    fields = REFINE_QUALITY + ('unsupported_causal_upgrade',)
    for case_id, pair in sorted(grouped.items()):
        if set(pair) != {'t0', 't1'}: raise ValueError('Refinement requires initial and refined stages')
        complete = all(pair[stage].get(f, REVIEW) != REVIEW for stage in pair for f in fields)
        include = case_id not in excluded and complete
        decisions.append({'case_id': case_id, 'included': include, 'reason': 'source-defined excluded task' if case_id in excluded else 'no complete two-of-three consensus' if not complete else None})
        if include: admitted.append(pair)
    result = {'paired_cases': len(admitted), 'metrics': {**{f: paired_metric(admitted, f) for f in REFINE_QUALITY}, 'unsupported_causal_upgrade': paired_metric(admitted, 'unsupported_causal_upgrade', error=True)}, 'inclusion_decisions': decisions}
    return result, compare_consensus(records['refinement_consensus'], calculated), warnings


def score(records, protocol):
    output = {'target_id': protocol['target_id'], 'panels': {}, 'consensus_differences': {}, 'missing_judge_warnings': {}}
    for panel, method in [('reasoning', score_reasoning), ('critic', score_critic), ('refinement', score_refinement)]:
        result, differences, warnings = method(records, protocol)
        output['panels'][panel] = result; output['consensus_differences'][panel] = differences; output['missing_judge_warnings'][panel] = warnings
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = score(json.loads(args.records.read_text(encoding='utf-8')), json.loads(args.protocol.read_text(encoding='utf-8')))
    output['records_sha256'] = hashlib.sha256(args.records.read_bytes()).hexdigest()
    output['protocol_sha256'] = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
