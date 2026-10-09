import json
from pathlib import Path
from phenofuncrank_repro.resources.conservation_upstream.dna import normalize_wig,run,seqfile
from phenofuncrank_repro.resources.conservation_upstream.native.windowed_maf import split_maf
from phenofuncrank_repro.resources.conservation_upstream.protein import restore_orthogroup,sequence_ids


def test_wig_chromosome_repair_preserves_scores(tmp_path):
    source=tmp_path/'raw.wig';source.write_text('fixedStep chrom=leaf.chr1 start=6 step=1\n0.25\n-2.3\n')
    out=tmp_path/'out.wig';normalize_wig(source,out,'chr01')
    assert out.read_text()=='fixedStep chrom=chr01 start=6 step=1\n0.25\n-2.3\n'


def test_native_windowed_maf_keeps_first_species_and_reference_first(tmp_path):
    source=tmp_path/'in.maf';source.write_text('##maf version=1\n\na\ns Sly_SL5.chr1 20001 3 + 90000 ACT\ns Sly_SL3.chr1 5 3 + 80000 ACT\ns Sly_SL3.copy 9 3 + 80000 AAT\ns Stu_SolTub3.chr1 9 3 + 80000 AGT\n\n')
    out=tmp_path/'chunks';split_maf(source,out,20000,'Sly_SL5.chr1')
    text=(out/'chunk_00000001.maf').read_text()
    assert 'Sly_SL5\t20001' in text
    assert text.count('Sly_SL3\t')==1
    assert 'AAT' not in text
    assert 'Stu_SolTub3\t9' in text


def test_sequence_id_restore_preserves_untrimmed_alignment(tmp_path):
    root=tmp_path/'result';(root/'WorkingDirectory').mkdir(parents=True)
    (root/'WorkingDirectory/SequenceIDs.txt').write_text('0_1: rice|P1\n1_2: relative|P2\n')
    mapping=sequence_ids(root);source=tmp_path/'og.fa';source.write_text('>Species0_1\nAC-DX\n>relative_relative|P2\nACEDX\n');out=tmp_path/'restored.fa'
    restore_orthogroup(source,out,mapping)
    assert out.read_text()=='>rice|P1\nAC-DX\n>relative|P2\nACEDX\n'


def test_command_plan_has_actual_phast_settings(tmp_path,capsys):
    config={'_output':tmp_path,'target':'rice','sequences':['chr01'],'phastcons':{'target_coverage':.25,'expected_length':45},'phylop':{'method':'LRT','mode':'CONACC'}}
    run(config,'scores',dry_run=True)
    text=capsys.readouterr().out
    assert '--target-coverage 0.25 --expected-length 45' in text
    assert '--method LRT --mode CONACC' in text
    line=next(line for line in text.splitlines() if line.startswith('phyloP'))
    assert line.index('neutral.mod')<line.index('chr01.maf')


def test_seqfile_retains_exact_tree_and_leaf_names(tmp_path):
    config={'_output':tmp_path,'tree':'(leaf1:1,leaf2:1);','genomes':[{'name':'leaf1'},{'name':'leaf2'}]}
    text=seqfile(config).read_text()
    assert text.startswith('(leaf1:1,leaf2:1);\nleaf1\t')
    assert 'leaf2\t' in text
