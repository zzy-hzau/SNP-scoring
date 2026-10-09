"""Canonical real-allele Core reference: top 5% of the whole trait locus union."""
from functools import lru_cache
from pathlib import Path
import math
import numpy as np
import pandas as pd
from phenofuncrank_repro.common.array_fields import midrank_percentile

def canonical_core_reference(table,fraction=.05):
    d=table.copy()
    for column in ('schema_ref','schema_alt','from_base','to_base'):
        if column not in d:raise ValueError(f'Whole-trait Core table missing {column}')
        d[column]=d[column].astype(str).str.upper()
    d=d[d.from_base.eq(d.schema_ref)&d.to_base.eq(d.schema_alt)].copy()
    d['abs_effect_real']=pd.to_numeric(d.dpc_additive_effect_mean,errors='coerce').abs()
    d=d[np.isfinite(d.abs_effect_real)]
    if d.empty:raise ValueError('No real reference-to-alternative DPC effects')
    if d.groupby(['chrom','pos']).size().ne(1).any():raise ValueError('Canonical biallelic loci must have exactly one real-direction effect')
    d=d.sort_values(['abs_effect_real','snp_id'],ascending=[False,True],kind='stable').reset_index(drop=True)
    core=d.head(math.ceil(len(d)*fraction-1e-10)).copy()
    core['core_percentile']=midrank_percentile(core.abs_effect_real.to_numpy())
    return core

@lru_cache(maxsize=64)
def _load_core(path,size,mtime,fraction):
    del size,mtime
    columns=['snp_id','chrom','pos','schema_ref','schema_alt','from_base','to_base','dpc_additive_effect_mean']
    return canonical_core_reference(pd.read_csv(path,sep='\t',usecols=columns),fraction)

def core_reference(spec):
    if spec.get("trait_config",{}).get("dpc_effect_mode")=="uploaded_population_perturbation":return None
    path=Path(spec.get('trait_config',{}).get('dpc_master',''))
    if not path.is_file():return None
    stat=path.stat();fraction=1-float(spec.get('trait_config',{}).get('level1_thresholds',{}).get('effect_quantile',.95))
    return _load_core(str(path),stat.st_size,stat.st_mtime_ns,fraction)
