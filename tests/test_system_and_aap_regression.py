"""Real fixtures protect fractional metrics, citation scope and sample alignment."""
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import statsmodels.api as sm
from phenofuncrank_repro.interpretation.evaluation.aggregate_system_scores import aggregate, rows_for
from phenofuncrank_repro.validation.aap_categorical_hc3 import analyze

ROOT = Path(__file__).resolve().parents[1]


def test_real_fractional_coverage_and_metric_denominators():
    example = ROOT/'examples/data/benchmarks/system_small_real'
    result = aggregate(example, example/'answers.jsonl')
    coverage = next(r for r in result['primary_score_means'] if r['condition']=='B1' and r['metric']=='Evidence coverage')
    assert coverage['sum_scores'] == 6.5
    assert coverage['n_applicable'] == 7
    assert coverage['mean_score'] == 6.5/7
    negative = next(r for r in result['primary_score_means'] if r['condition']=='B1' and r['metric']=='Negative control correct')
    assert negative['n_applicable'] == 2


def test_uncited_real_claims_stay_unscored_and_constructed_id_control_is_detected(tmp_path):
    example = ROOT/'examples/data/benchmarks/system_small_real'
    answers = {r['task_id']:r for r in [json.loads(line) for line in (example/'answers.jsonl').read_text().splitlines()]}
    copied = tmp_path/'full_30';shutil.copytree(example/'full_30', copied)
    before = rows_for(copied, answers, 'full_scorecard.tsv')
    
    
    assert before.attrs['grounded_claims'] == []
    cards = [json.loads(line) for line in (copied/'answer_cards.jsonl').read_text().splitlines()]
    changed = False
    for card in cards:
        for claim in card['claims']:
            if card['evidence']:
                claim['supporting_evidence_ids'] = [card['evidence'][0]['evidence_id']]
                (copied/'answer_cards.jsonl').write_text('\n'.join(json.dumps(c) for c in cards)+'\n')
                valid_control = rows_for(copied, answers, 'full_scorecard.tsv')
                assert valid_control.attrs['grounded_claims'] == [True]
                claim['supporting_evidence_ids'] = ['NEW_UNLISTED_TEST_ID'];changed = True;break
        if changed:break
    assert changed
    (copied/'answer_cards.jsonl').write_text('\n'.join(json.dumps(c) for c in cards)+'\n')
    after = rows_for(copied, answers, 'full_scorecard.tsv')
    assert after.attrs['grounded_claims'] == [False]
    assert after.attrs['evidence_id_errors'] == [True]


def test_aap_real_sample_id_join_is_independent_of_covariate_row_order():
    samples = pd.read_csv(ROOT/'examples/data/cases/AAP/small_samples.tsv', sep='\t')
    pcs = pd.read_csv(ROOT/'examples/data/cases/AAP/small_PCs.tsv', sep='\t')
    a = analyze(samples, pcs)
    b = analyze(samples, pcs.iloc[::-1])
    assert a['n'] == 72
    assert a['contrasts'] == b['contrasts']
    assert a['contrasts'][0]['df'] == 64


def test_small_real_aap_hc3_against_independent_covariance_implementation():
    samples = pd.read_csv(ROOT/'examples/data/cases/AAP/small_samples.tsv', sep='\t')
    pcs = pd.read_csv(ROOT/'examples/data/cases/AAP/small_PCs.tsv', sep='\t')
    result = analyze(samples, pcs)
    d = samples.merge(pcs, on='sample_id', how='left', validate='one_to_one', sort=False)
    pc = d[[f'PC{i}' for i in range(1,6)]].to_numpy()
    pc = (pc-pc.mean(axis=0))/pc.std(axis=0, ddof=1)
    X = np.column_stack([np.ones(len(d)), (d.genotype=='G/A').astype(float), (d.genotype=='A/A').astype(float), pc])
    fit = sm.OLS(d.hundred_seed_weight_g.to_numpy(), X).fit().get_robustcov_results(cov_type='HC3', use_t=True)
    L = np.array([[0,-1,0,0,0,0,0,0], [0,1,-1,0,0,0,0,0], [0,0,-1,0,0,0,0,0]], dtype=float)
    contrast = fit.t_test(L)
    np.testing.assert_allclose([r['HC3_SE_g'] for r in result['contrasts']], contrast.sd, rtol=1e-8, atol=0.)
    np.testing.assert_allclose([r['P_two_sided_HC3'] for r in result['contrasts']], contrast.pvalue, rtol=1e-7, atol=0.)
