"""Run the chickpea training, prediction and T/F/B/M ranking example."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd


def command(arguments, environment):
    subprocess.run([sys.executable, *arguments], env=environment, check=True)


def effect_targets(data, output):
    targets = pd.read_csv(data/'candidates/allele_effect_targets.tsv', sep='\t')
    maps = sorted((output/'dpcformer').glob('fold_*/Hundred_seed_weight.fold_*.selected_snp_map.tsv'))
    if not maps:
        raise FileNotFoundError(output/'dpcformer')
    counts = pd.concat([pd.read_csv(p, sep='\t').snp_id for p in maps]).value_counts()
    targets['selected_fold_count'] = targets.snp_id.map(counts).fillna(0).astype(int)
    targets['fold_selection_rate_prior'] = targets.selected_fold_count / len(maps)
    path = output/'allele_effect_targets.tsv'
    targets.to_csv(path, sep='\t', index=False)
    return path


def rank_candidates(data, output, budget):
    from phenofuncrank_repro.prioritization.paper_selection import (
        _normalize_block, build_context_from_calibrated_scores, select_greedy_fast,
    )
    meta = json.loads((data/'dpcformer/metadata.json').read_text())
    snps = pd.read_csv(data/'dpcformer/cultivar.snp_map.tsv', sep='\t')
    geno = np.memmap(data/'dpcformer'/meta['genotype_memmap'], mode='r', dtype='uint8', shape=tuple(meta['shape']))
    effects = pd.read_csv(output/'allele_effects/allele_perturbation_all_valid.aggregated.tsv', sep='\t')
    effects = effects[np.isfinite(effects.dpc_additive_effect_mean)].copy()
    effects['abs_effect_real'] = effects.dpc_additive_effect_mean.abs()
    core = effects.sort_values(['abs_effect_real','snp_id'], ascending=[False,True]).head(20)
    core_ids = set(core.snp_id.astype(str))
    z_all = _normalize_block(np.asarray(geno))
    r2 = np.clip(np.square(z_all @ z_all.T), 0, 1)
    core_positions = np.flatnonzero(snps.snp_id.astype(str).isin(core_ids))
    supported = np.max(r2[core_positions], axis=0) >= .60
    indices = np.flatnonzero(supported)
    pool = snps.iloc[indices].copy().reset_index(drop=True)
    pool['panel_id'] = pool.snp_id.astype(str)
    pool['chrom'] = pool.Chr.astype(str)
    pool['pos'] = pool.Pos.astype(int)
    pool['ref'] = pool.Ref.astype(str)
    pool['alt'] = pool.Alt.astype(str)
    pool['role'] = np.where(pool.panel_id.isin(core_ids), 'core', 'proxy')
    score_frames = []
    for mode in ['coding','noncoding']:
        frame = pd.read_parquet(output/f'scores/{mode}.parquet')
        frame['variant_type'] = mode
        score_frames.append(frame[['snp_id','variant_type','strict_v3_prob']])
    scores = pd.concat(score_frames, ignore_index=True)
    scores = pool[['panel_id']].merge(scores, left_on='panel_id', right_on='snp_id', how='left', validate='one_to_one')
    assert scores.strict_v3_prob.notna().all()
    scores['F'] = scores.groupby('variant_type').strict_v3_prob.rank(method='average',pct=True)
    core = core.rename(columns={'snp_id':'pool_id'})[['pool_id','abs_effect_real']]
    position = {int(g):i for i,g in enumerate(indices)}
    rows=[]
    for g in core_positions:
        for h in indices[r2[g,indices] >= .60]:
            if g != h:
                rows.append({'core_idx':position[int(g)],'cand_idx':position[int(h)],'r2':float(r2[g,h])})
    pairs = pd.DataFrame(rows,columns=['core_idx','cand_idx','r2'])
    context = build_context_from_calibrated_scores(pool,core,scores,pairs,indices)
    selected, state = select_greedy_fast(context,z_all[indices],min(budget,len(pool)),return_state=True)
    full = pd.DataFrame({'snp_id':context['snp_id'],'variant_type':scores.variant_type.to_numpy(),'T':context['T'],'F':context['F'],'B':context['B'],'M_after_selection':state['marginal_final'],'M_at_selection':state['selection_m'],'selection_step':state['selection_step']})
    target = output/'ranking'
    target.mkdir(parents=True,exist_ok=True)
    full.to_csv(target/'candidate_scores.tsv',sep='\t',index=False)
    selected.to_csv(target/'selected_candidates.tsv',sep='\t',index=False)
    pairs.to_csv(target/'core_ld_support.tsv',sep='\t',index=False)
    summary={'input_snps':len(snps),'core_snps':len(core),'eligible_candidates':len(pool),'selected_candidates':len(selected),'coding_candidates':int((scores.variant_type=='coding').sum()),'noncoding_candidates':int((scores.variant_type=='noncoding').sum())}
    (target/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/example'))
    parser.add_argument('--output',type=Path,default=Path('results/example'))
    parser.add_argument('--step',choices=['all','train-dpc','predict-dpc','train-function','score-function','rank'],default='all')
    args=parser.parse_args()
    data=args.data.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    environment=os.environ.copy()
    environment.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    steps={};start=time.perf_counter()
    def run_step(name,action):
        if args.step not in ['all',name]:return
        t=time.perf_counter();action();steps[name]=round(time.perf_counter()-t,3)
    run_step('train-dpc',lambda:command(['-m','phenofuncrank_repro','dpc-train-diploid','--inputs-dir',str(data/'dpcformer'),'--outdir',str(out/'dpcformer'),'--trait','Hundred_seed_weight','--pheno-col','Hundred_seed_weight','--selector','minepy_mic','--selected-snp-count','128','--folds','10','--test-ratio','0.1','--epochs','3','--batch-size','32','--learning-rate','0.001','--seed','42','--selection-workers','1','--min-available-mem-gb','0','--cpu'],environment))
    run_step('predict-dpc',lambda:command(['-m','phenofuncrank_repro','allele-effects','--targets',str(effect_targets(data,out)),'--inputs-dir',str(data/'dpcformer'),'--model-dir',str(out/'dpcformer'),'--out-dir',str(out/'allele_effects'),'--trait-prefix','Hundred_seed_weight','--pheno-col','Hundred_seed_weight','--sample-scope','all_valid','--folds','1-10','--total-folds-for-rate','10','--selected-snp-count','128','--device','cpu','--snp-batch-size','16'],environment))
    def train_function():
        for mode in ['coding','noncoding']:
            command(['-m','phenofuncrank_repro.plantvimpact.strict_model','train','--profile',f'configs/examples/chickpea_{mode}.json','--data-root',str(data/'plantvimpact'),'--out-dir',str(out/f'plantvimpact/{mode}'),'--folds','5','--device','cpu','--nthread','4'],environment)
    run_step('train-function',train_function)
    def score_function():
        for mode in ['coding','noncoding']:
            command(['-m','phenofuncrank_repro','pvi-score','--input',str(data/f'candidates/{mode}.parquet'),'--model-root',str(out/f'plantvimpact/{mode}'),'--output',str(out/f'scores/{mode}.parquet'),'--folds','5'],environment)
    run_step('score-function',score_function)
    run_step('rank',lambda:rank_candidates(data,out,10))
    steps['total']=round(time.perf_counter()-start,3)
    (out/f'run_time_{args.step}.json').write_text(json.dumps(steps,indent=2)+'\n')
    print(json.dumps(steps))


if __name__=='__main__':
    main()
