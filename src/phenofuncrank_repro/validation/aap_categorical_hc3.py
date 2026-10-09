"Source-categorical genotype, PC5 and HC3 comparisons for the AAP case."

import argparse,json

from pathlib import Path

import numpy as np

import pandas as pd

from scipy import stats

def ols(X, y):
 b=np.linalg.lstsq(X,y,rcond=None)[0]
 resid=y-X@b;C=np.linalg.inv(X.T@X);rank=np.linalg.matrix_rank(X);n=len(y)
 assert rank==X.shape[1]
 V=(resid@resid/(n-rank))*C
 h=np.einsum('ij,jk,ik->i',X,C,X)
 assert h.max()<1
 scaled=X*(resid/(1-h))[:,None]
 HC3=C@(scaled.T@scaled)@C
 return b,V,HC3,n-rank

def holm(ps):
 ps=np.asarray(ps);order=np.argsort(ps);q=np.empty(len(ps))
 q[order]=np.minimum(1,np.maximum.accumulate(ps[order]*(len(ps)-np.arange(len(ps)))))
 return q


def analyze(samples, covariates):
    if not samples.sample_id.is_unique or not covariates.sample_id.is_unique:
        raise ValueError('Sample identifiers must be unique in each supplied table')
    d=samples.merge(covariates,on='sample_id',how='left',validate='one_to_one',sort=False)
    if d.sample_id.tolist()!=samples.sample_id.tolist():
        raise ValueError('Covariate joining changed the original sample order')
    pc=d[[f'PC{i}' for i in range(1,6)]].to_numpy(dtype=float)
    y=d.hundred_seed_weight_g.to_numpy(dtype=float)
    if not np.isfinite(pc).all() or not np.isfinite(y).all():
        raise ValueError('Provide complete phenotype and original PC1-5 observations')
    gwas_check='not_supplied'
    if 'GWAS_fam_HSW' in d:
        if not np.allclose(y,d.GWAS_fam_HSW.to_numpy(float),atol=1e-8,rtol=0):
            raise ValueError('Phenotypes do not match the supplied GWAS phenotype export')
        gwas_check='passed'
    pc=(pc-pc.mean(axis=0))/pc.std(axis=0,ddof=1)
    dose=d.genotype.map({'G/G':0.,'G/A':1.,'A/A':2.}).to_numpy()
    if not np.isfinite(dose).all():
        raise ValueError('Genotypes must use the genomic G/G, G/A or A/A labels')
    Xa=np.column_stack([np.ones(len(y)),pc,dose]);ba,Va,Ha,dfa=ols(Xa,y)
    se=float(np.sqrt(Va[-1,-1]));pa=float(2*stats.t.sf(abs(ba[-1]/se),dfa))
    X=np.column_stack([np.ones(len(y)),(d.genotype=='G/A').astype(float),(d.genotype=='A/A').astype(float),pc])
    b,V,HC3,dfres=ols(X,y)
    specs=[('G/G','G/A',0,1,[0,-1,0,0,0,0,0,0]),
           ('G/A','A/A',1,2,[0,1,-1,0,0,0,0,0]),
           ('G/G','A/A',0,2,[0,0,-1,0,0,0,0,0])]
    rows=[]
    for g1,g2,x1,x2,L in specs:
        L=np.array(L,dtype=float);effect=float(L@b)
        ses=float(np.sqrt(L@V@L));seh=float(np.sqrt(L@HC3@L))
        p=float(2*stats.t.sf(abs(effect/seh),dfres));p_ols=float(2*stats.t.sf(abs(effect/ses),dfres))
        critical=stats.t.ppf(.975,dfres)
        rows.append({'group1':g1,'group2':g2,'x1':x1,'x2':x2,'adjusted_mean_difference_g':effect,
          'HC3_SE_g':seh,'df':int(dfres),'HC3_t':effect/seh,'P_two_sided_HC3':p,
          'OLS_P_sensitivity':p_ols,'95CI_low_g':effect-critical*seh,'95CI_high_g':effect+critical*seh})
    for r,q,qs in zip(rows,holm([z['P_two_sided_HC3'] for z in rows]),holm([z['OLS_P_sensitivity'] for z in rows])):
        r['P_Holm']=float(q);r['OLS_P_Holm_sensitivity']=float(qs);r['significant_at_0_05']=bool(q<.05)
        r['annotation']='ns' if q>=.05 else '*' if q>=.01 else '**' if q>=.001 else '***' if q>=.0001 else '****'
    return {'n':len(d),'group_counts':{str(k):int(v) for k,v in d.genotype.value_counts().items()},
      'categorical_group_means_at_common_PC_mean':{'G/G':float(b[0]),'G/A':float(b[0]+b[1]),'A/A':float(b[0]+b[2])},
      'additive_sensitivity':{'effect_allele':'A','beta_g':float(ba[-1]),'SE_g':se,'P':pa},
      'contrasts':rows,'PCs_refit':False,'PC_center_scale_ddof':1,'GWAS_phenotype_export_check':gwas_check,
      'inference_target':'Conditional group mean differences, not median differences',
      'genotype_distributions_remain_unadjusted':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,required=True,help='TSV: sample_id, genotype, hundred_seed_weight_g')
    parser.add_argument('--covariates',type=Path,required=True,help='TSV: sample_id, PC1..PC5; optional GWAS_fam_HSW')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    samples=pd.read_csv(args.samples,sep='\t',dtype={'sample_id':str})
    covariates=pd.read_csv(args.covariates,sep='\t',dtype={'sample_id':str})
    result=analyze(samples,covariates)
    import hashlib
    result['input_sha256']={'samples':hashlib.sha256(args.samples.read_bytes()).hexdigest(),'covariates':hashlib.sha256(args.covariates.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
