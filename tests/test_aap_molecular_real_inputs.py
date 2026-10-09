"""Actual AAP sequences protect strand orientation, locus identity and raw FPKM."""
import json
from pathlib import Path
import pytest
from phenofuncrank_repro.validation.aap_molecular import sequence_consequence, fasta_sequence, parse_blast_alignments, expression_rows, verify_expression_mapping

F=Path(__file__).resolve().parents[1]/'examples/data/cases/AAP_molecular'


def test_actual_minus_strand_conversion_changes_one_protein_residue():
    region=json.loads((F/'region_metadata.json').read_text())
    result=sequence_consequence(region,33198529)
    assert result['coding_position_1based']==968
    assert result['ref_codon']=='GCA' and result['alt_codon']=='GTA'
    assert result['reference_protein']==fasta_sequence(F/'ncbi_candidate_protein.fa')
    assert [i+1 for i,(a,b) in enumerate(zip(result['reference_protein'],result['alternative_protein'])) if a!=b]==[323]
    with pytest.raises(ValueError,match='Genomic REF'):
        sequence_consequence(region,33198529,genomic_ref='A',genomic_alt='G')


def test_actual_gapped_alignment_maps_query_position_to_subject_position():
    hit=parse_blast_alignments(F/'aap_swiss_aligned.tsv')[0]
    assert hit['accession']=='Q42400'
    assert hit['query_start']==4
    assert hit['focal_mapping']['subject_position']==341
    assert hit['focal_mapping']['query_residue']=='A' and hit['focal_mapping']['subject_residue']=='A'


def test_expression_mapping_checks_the_actual_transcript_not_only_a_hit_label():
    region=json.loads((F/'region_metadata.json').read_text());cds=sequence_consequence(region,33198529)['CDS']
    transcript=fasta_sequence(F/'Ca_TC15673.1_atlas.fa')
    result=verify_expression_mapping(F/'AAP_focal_CDS_atlas_alignment.tsv',cds,transcript)
    assert result['subject_start_1based']==129 and result['subject_end_1based']==1526
    altered=transcript[:128]+'N'+transcript[129:]
    with pytest.raises(ValueError,match='Actual atlas transcript'):
        verify_expression_mapping(F/'AAP_focal_CDS_atlas_alignment.tsv',cds,altered)


def test_processed_expression_retains_original_values_and_units():
    rows=expression_rows(F/'AAP_gene_only_FPKM.tar')
    assert len(rows)==32
    assert next(r['FPKM'] for r in rows if r['tissue_code']=='Endo')==32.397
    assert next(r['FPKM'] for r in rows if r['tissue_code']=='20DAP')==6.0
    assert {r['gene'] for r in rows}=={'Ca_TC15673'}
