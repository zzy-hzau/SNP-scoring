from phenofuncrank_repro.common.paths import package_dir
"""Coordinate/strand/CDS contracts for the generic candidate module."""
from pathlib import Path
import importlib.util
import sys

scripts=package_dir('resources')
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('coding_candidate_coordinates',scripts/'candidates/make_coding_candidates.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_minus_strand_identity_uses_genomic_forward_alleles():
    rows=list(module.iter_coding_candidates('minus',[('chr1',1,3,'-',0)],{'chr1':'CAT'},'rice'))
    expected={1:'C',2:'A',3:'T'}
    assert all(row[5]==expected[row[4]] for row in rows)
    assert [row[6] for row in rows if row[4]==3]==['A','C','G']
    assert all(row[8]=='ATG' and row[9]=='M' for row in rows)


def test_splice_codon_is_preserved_and_CDS_indices_are_cumulative():
    seq,mapping,strand=module.build_cds_sequence('split',[('chr1',1,2,'+',0),('chr1',6,6,'+',1)],{'chr1':'ATCCCG'})
    assert seq=='ATG'
    assert mapping==[(0,('chr1',1)),(1,('chr1',2)),(2,('chr1',6))]
    assert strand=='+'


def test_forward_allele_order_and_consequence_are_consistent():
    rows=list(module.iter_coding_candidates('plus',[('chr1',1,3,'+',0)],{'chr1':'GCT'},'rice'))
    third=[row for row in rows if row[4]==3]
    assert [row[6] for row in third]==['A','C','G']
    assert all(row[9]=='A' and row[10]=='A' and row[11]=='synonymous' for row in third)
