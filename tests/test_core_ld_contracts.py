"""Downstream Core/LD method contracts; these are constructed cases, not model tests."""
import numpy as np
import pandas as pd
import pytest
from phenofuncrank_repro.prioritization.core_reference_current import canonical_core_reference
from phenofuncrank_repro.prioritization.genotype_ld_current import annotate_proxy_direction


def test_core_uses_real_allele_fraction_stable_ties_without_pvalue_gate():
    rows=[]
    for i in range(20):
        rows.append({'snp_id':f's{i:02d}','chrom':'1','pos':i+1,'schema_ref':'G','schema_alt':'A','from_base':'G','to_base':'A','dpc_additive_effect_mean':20. if i in [18,19] else float(i),'permutation_q_signflip':1.,'sample_direction_consistency':0.})
        rows.append({**rows[-1],'from_base':'A','to_base':'T','dpc_additive_effect_mean':1000.})
    selected=canonical_core_reference(pd.DataFrame(rows))
    assert selected.snp_id.tolist()==['s18']
    assert selected.core_percentile.tolist()==[0.5]


def test_core_duplicate_real_locus_is_not_counted_as_another_eligible_site():
    row={'snp_id':'s1','chrom':'1','pos':1,'schema_ref':'G','schema_alt':'A','from_base':'G','to_base':'A','dpc_additive_effect_mean':1.}
    with pytest.raises(ValueError,match='exactly one real-direction effect'):
        canonical_core_reference(pd.DataFrame([row,{**row,'snp_id':'s2'}]))


def test_proxy_direction_retains_signed_r_and_reports_conflicting_sources():
    frame=pd.DataFrame({'snp_id':['c1','c2','p1'],'is_core':[True,True,False],'dpc_evaluable':[True,True,False],'dpc_additive_effect_mean':[2.,-3.,np.nan]})
    edges=pd.DataFrame({'core_snp_id':['c1','c2'],'proxy_snp_id':['p1','p1'],'r':[.8,.8],'r2':[.64,.64]})
    result=annotate_proxy_direction(frame,edges).set_index('snp_id')
    assert result.loc['p1','trait_support_direction_status']=='conflicting_sources'
    assert result.loc['p1','trait_support_direction'] is None
    edges.loc[1,'r']=-.8
    resolved=annotate_proxy_direction(frame,edges).set_index('snp_id')
    assert resolved.loc['p1','trait_support_direction']=='increase'
    assert resolved.loc['p1','trait_support_direction_status']=='ld_supported'
