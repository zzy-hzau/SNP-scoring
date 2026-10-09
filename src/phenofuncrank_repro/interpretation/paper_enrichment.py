"""SNP enrichment with matched genomic backgrounds and exact upper-tail tests."""
from __future__ import annotations
import numpy as np
import pandas as pd

def matched_enrichment_from_files(request):
 from pathlib import Path
 from phenofuncrank_repro.common.reference_contract import require_same_assembly
 from phenofuncrank_repro.interpretation.settings import settings
 frames={}
 for key in ('candidates','term_intervals','backgrounds','matches'):
  path=Path(request[key+'_path']).resolve()
  if not path.is_relative_to(settings.workspace.resolve()):raise ValueError('Enrichment input outside workspace')
  frames[key]=pd.read_csv(path,sep='\t')
  if key!='matches':
   if 'assembly' not in frames[key]:raise ValueError(f'{key}: assembly column required')
   for assembly in frames[key].assembly.drop_duplicates():require_same_assembly(assembly,request['assembly'])
 backgrounds={trait:group for trait,group in frames['backgrounds'].groupby('trait')}
 result=matched_snp_enrichment(frames['candidates'],frames['term_intervals'],backgrounds,frames['matches'],window=100000)
 return {'status':'success','analysis_type':'matched_snp_enrichment','window_bp':100000,
  'null_test':'exact_poisson_binomial_upper_tail','multiple_testing':'BH within trait and annotation source',
  'tested_terms':len(result),'hits':result.to_dict(orient='records')}


from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_functions
from phenofuncrank_repro.common import enrichment_kernels as _source_kernels
_bind_functions(globals(), _source_kernels, ['bh_adjust', 'interval_membership', 'matched_snp_enrichment', 'poisson_binomial_tail'])

