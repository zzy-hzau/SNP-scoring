"""Actual genotype bytes and explicit rule controls for the formal PC-only workflow."""
from pathlib import Path
import numpy as np
import pandas as pd
from phenofuncrank_repro.validation.chickpea_pc5_gwas import export_original_codes
from phenofuncrank_repro.validation.chickpea_gwas_leads import Panel, greedy_clump, expand_perlead, bh, select_leads

F=Path(__file__).resolve().parents[1]/'examples/data/validation/chickpea_pc5_demo'


def test_actual_original_code_bed_roundtrip_with_partial_final_byte(tmp_path):
    codes=np.load(F/'codes.npy')[:,:5];variants=pd.read_csv(F/'variants.tsv',sep='\t');ph=pd.read_csv(F/'phenotypes.tsv',sep='\t').iloc[:5]
    prefix=tmp_path/'actual'
    export_original_codes(codes,variants,ph,prefix)
    encoded=np.frombuffer(prefix.with_suffix('.bed').read_bytes()[3:],dtype=np.uint8).reshape(len(codes),2)
    states=((encoded[:,:,None]>>np.array([0,2,4,6],dtype=np.uint8))&3).reshape(len(codes),8)
    decoded=np.array([1,0,2,3],dtype=np.uint8)[states[:,:5]]
    np.testing.assert_array_equal(decoded,codes)
    assert np.all(states[:,5:]==1)
    bim=pd.read_csv(prefix.with_suffix('.bim'),sep='\t',header=None)
    assert bim[4].tolist()==variants.ref.tolist() and bim[5].tolist()==variants.alt.tolist()


def test_source_window_boundaries_use_constructed_positions_on_real_genotypes():
    real=np.load(F/'codes.npy')[0]
    codes=np.stack([real,real,real])
    variants=pd.DataFrame({'chrom':['Ca5']*3,'pos':[1000000,1000010,1500000]})
    kept=greedy_clump(Panel(codes,variants),variants)
    assert kept.pos.tolist()==[1000000]
    interval=expand_perlead(Panel(codes,variants),kept)
    
    
    assert interval.iloc[0]['start']==1000000 and interval.iloc[0]['end']==1000010


def test_bh_keeps_full_supplied_family_and_monotone_rank_adjustment():
    p=np.array([.2,.001,.01,.04])
    q=bh(p)
    np.testing.assert_allclose(q,[.2,.004,.02,.05333333333333334],rtol=1e-14,atol=0)
    assert np.all(np.diff(q[np.argsort(p)])>=0)


def test_no_significant_candidates_stays_empty_with_usable_interval_schema():
    codes=np.load(F/'codes.npy')[:3];variants=pd.read_csv(F/'variants.tsv',sep='\t').iloc[:3]
    association=pd.DataFrame({'chrom':variants.chrom.to_numpy(),'pos':variants.pos.to_numpy(),'p_wald':[.4,.5,.6]})
    tested,leads,intervals=select_leads(association,codes,variants)
    assert len(tested)==3 and leads.empty and intervals.empty
    assert intervals.columns.tolist()==['seed_row','chrom','start','end','region_id']
