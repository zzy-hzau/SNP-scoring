"Compute external-variant benchmark statistics."
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

specs={
 'maize_coding':['PlantVImpact','ESM1b','PlantCaduceus','LM_fixed_blend'],
 'rice_coding':['PlantVImpact','ESM1b','PlantCaduceus'],
 'rice_noncoding':['PlantVImpact','PlantCaduceus','PlantDeepSEA'],
 'rice_regulatory':['PlantVImpact','PlantCaduceus','PlantDeepSEA'],
 'rice_coding_classic':['PlantVImpact','ESM1b','PlantCaduceus','SIFT','PolyPhen2'],
}

def weights(score,k):
    x=np.asarray(score);cut=np.sort(x)[-k];above=x>cut;ties=x==cut;w=above.astype(float);w[ties]=(k-int(above.sum()))/int(ties.sum())
    assert np.isclose(w.sum(),k)
    return w,dict(cutoff=float(cut),above=int(above.sum()),tied=int(ties.sum()),slots_for_ties=k-int(above.sum()))


def analyze(benchmark_dir, output_dir, *, seed=20260926, n_resamples=10000):
    O=Path(benchmark_dir)
    OUT=Path(output_dir)
    OUT.mkdir(parents=True,exist_ok=True)
    RNG=np.random.default_rng(seed)
    NRESAMPLE=n_resamples
    metrics=[];paired=[];coverage=[];rank_rows=[];common=[]


    
    legacy_nc=pd.read_parquet(O/'rice_noncoding_scored.parquet')
    annotation=legacy_nc.consequence.fillna('')
    eligible=annotation.str.contains('upstream|downstream|UTR|intron|intergenic|regulatory|non_coding',case=False,regex=True)&~legacy_nc.protein_consequence
    regulatory=legacy_nc[legacy_nc.role.eq('matched_background')|eligible].copy()
    regulatory.to_parquet(OUT/'rice_regulatory_scored.parquet',index=False)
    legacy_nc.loc[legacy_nc.role.eq('benchmark_positive'),['snp_id','consequence','protein_consequence']].assign(regulatory_eligible=eligible[legacy_nc.role.eq('benchmark_positive')].to_numpy()).to_csv(OUT/'regulatory_eligibility.tsv',sep='\t',index=False)

    for tag,methods in specs.items():
        p=(OUT if tag=='rice_regulatory' else O)/f'{tag}_scored.parquet'
        if not p.exists():continue
        d=pd.read_parquet(p)
        for m in [c for c in ['PlantVImpact','ESM1b','PlantCaduceus','LM_fixed_blend','PlantDeepSEA','SIFT','PolyPhen2','PICNC'] if c in d]:
            for role,g in d.groupby('role'):
                coverage.append(dict(dataset=tag,method=m,role=role,n_total=len(g),n_scored=int(g[m].notna().sum())))
        present=[m for m in methods if m in d]
        if present!=methods:raise RuntimeError((tag,'missing comparators',set(methods)-set(present)))
        mask=d[methods].apply(pd.to_numeric,errors='coerce').notna().all(axis=1);d=d.loc[mask].reset_index(drop=True)
        ispos=d.role.eq('benchmark_positive').to_numpy();pos=d.loc[ispos];n=len(pos)
        if n==0:continue
        common.append(dict(dataset=tag,methods=','.join(methods),positive_n=n,background_n=int((~ispos).sum()),common_pool_n=len(d)))
        groups=pos.GeneID.astype('string').fillna(pos.snp_id.astype('string')).to_numpy(str);groupnames,gi=np.unique(groups,return_inverse=True);ng=len(groupnames);counts=np.bincount(gi)
        resamples=RNG.integers(0,ng,size=(NRESAMPLE,ng));signs=RNG.choice([-1,1],size=(NRESAMPLE,ng))
        methodweights={}
        for m in methods:
            ranks=rankdata(-d[m].to_numpy(),method='average');percentile=(ranks-.5)/len(d)
            for idx in np.where(ispos)[0]:rank_rows.append(dict(dataset=tag,snp_id=d.at[idx,'snp_id'],method=m,score=float(d.at[idx,m]),rank=float(ranks[idx]),top_percentile=float(100*percentile[idx])))
            for k in [20,50,100]:
                if len(d)<k:continue
                w,tie=weights(d[m],k);positive_w=w[ispos];methodweights[(m,k)]=positive_w
                sums=np.bincount(gi,weights=positive_w,minlength=ng);boot=sums[resamples].sum(axis=1)/counts[resamples].sum(axis=1);lo,hi=np.quantile(boot,[.025,.975])
                metrics.append(dict(dataset=tag,method=m,top_k=k,positive_n=n,background_n=len(d)-n,n_gene_clusters=ng,expected_positive_hits=float(positive_w.sum()),recall=float(positive_w.mean()),recall_ci_low=float(lo),recall_ci_high=float(hi),mean_positive_top_percentile=float(100*percentile[ispos].mean()),**tie))
        for baseline in methods[1:]:
            for k in [20,50,100]:
                if ('PlantVImpact',k) not in methodweights:continue
                diff=methodweights[('PlantVImpact',k)]-methodweights[(baseline,k)];sums=np.bincount(gi,weights=diff,minlength=ng)
                boots=sums[resamples].sum(axis=1)/counts[resamples].sum(axis=1);lo,hi=np.quantile(boots,[.025,.975]);null=(signs*sums).sum(axis=1);obs=sums.sum();pval=(1+np.sum(np.abs(null)>=abs(obs)-1e-12))/(NRESAMPLE+1)
                paired.append(dict(dataset=tag,top_k=k,contrast='PlantVImpact - '+baseline,positive_n=n,gene_clusters=ng,delta_recall=float(diff.mean()),ci_low=float(lo),ci_high=float(hi),p=float(pval),resamples=NRESAMPLE))

    pd.DataFrame(metrics).to_csv(OUT/'fixed_budget_metrics.tsv',sep='\t',index=False)
    t=pd.DataFrame(paired)
    if len(t):
        order=np.argsort(t.p.to_numpy());v=t.p.to_numpy()[order];q=np.minimum.accumulate((v*len(v)/np.arange(1,len(v)+1))[::-1])[::-1];q=np.minimum(q,1);values=np.empty(len(q));values[order]=q;t['q_BH']=values
    t.to_csv(OUT/'paired_comparisons_10000.tsv',sep='\t',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'method_coverage.tsv',sep='\t',index=False)
    pd.DataFrame(common).to_csv(OUT/'common_candidate_pools.tsv',sep='\t',index=False)
    pd.DataFrame(rank_rows).to_csv(OUT/'positive_variant_ranks.tsv',sep='\t',index=False)
    (OUT/'statistics_protocol.json').write_text(json.dumps(dict(seed=seed,n_resamples=NRESAMPLE,uncertainty='gene-cluster bootstrap of benchmark positives conditional on the fixed scored pool',test='two-sided paired gene-cluster sign randomization; (extreme+1)/(10000+1)',multiple_testing='Benjamini-Hochberg across all reported PVI-versus-comparator contrasts and budgets',ties='equal expected allocation of remaining TopK slots among exact score ties; total selected weight exactly K',negative_truth_available=False,auc_not_used_as_functional_specificity=True),indent=2))
    print(pd.DataFrame(metrics).pivot_table(index=['dataset','top_k'],columns='method',values='recall').round(4).to_string())

    return {"metrics":pd.DataFrame(metrics),"paired":t,"coverage":pd.DataFrame(coverage),"common":pd.DataFrame(common),"ranks":pd.DataFrame(rank_rows)}
