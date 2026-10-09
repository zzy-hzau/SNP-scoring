"""Behavior checks for the corrected training and supplied archive input route."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from phenofuncrank_repro.models.dpcformer import train_diploid as diploid
from phenofuncrank_repro.models.dpcformer import train_cotton as cotton
from phenofuncrank_repro.models.dpcformer.inputs_csv import prepare_csv
from phenofuncrank_repro.models.dpcformer.report_folds import summarize_folds


def test_representative_fold_uses_validation_loss():
    table = pd.DataFrame({'fold':[1,2,3], 'best_val_loss':[0.1,0.4,0.3], 'test_pearson':[0.2,0.8,0.5]})
    result = summarize_folds(table)
    assert result['canonical_fold_by_validation_loss'] == 1
    assert result['representative_fold_by_validation_loss'] == 1
    assert result['Selected_Test_PCC'] == pytest.approx(0.2)
    assert result['representative_uses_test_performance'] is False
    changed = table.copy(); changed['test_pearson'] = [0.01, 0.99, 0.3]
    assert summarize_folds(changed)['representative_fold_by_validation_loss'] == 1
    assert result['Mean_Test_PCC'] == pytest.approx(0.5)
    assert result['SD_Test_PCC_ddof0'] == pytest.approx(np.std([0.2,0.8,0.5],ddof=0))


def test_nonfinite_fold_is_disclosed_in_report():
    result = summarize_folds(pd.DataFrame({'fold':[1,2], 'best_val_loss':[1,2], 'test_pearson':[np.nan,0.3]}))
    assert result['folds_supplied'] == 2 and result['finite_test_PCC_folds'] == 1
    assert result['all_test_PCCs_finite'] is False


def test_csv_does_not_silently_truncate_odd_allele_string(tmp_path):
    data = tmp_path/'input.csv'; loci = tmp_path/'input.map'
    pd.DataFrame({'individual_id':['sample'], 'genotype':['ACG'], 'trait':[1]}).to_csv(data,index=False)
    loci.write_text('1 snp1 0 10\n1 snp2 0 20\n')
    with pytest.raises(ValueError, match='expected exactly 4 alleles'):
        prepare_csv(data,loci,tmp_path/'out','trait')


def test_csv_allele_encoding_and_unsupported_multiallelic_site(tmp_path):
    data = tmp_path/'input.csv'; loci = tmp_path/'input.map'
    pd.DataFrame({'individual_id':['s1','s2','s3'], 'genotype':['A A A A','AGCC','GGGG'], 'trait':[1,2,3]}).to_csv(data,index=False)
    loci.write_text('1 snp1 0 10\n1 snp2 0 20\n')
    result = prepare_csv(data,loci,tmp_path/'out','trait')
    codes = np.memmap(tmp_path/'out'/result['genotype_memmap'],mode='r',dtype=np.uint8,shape=(2,3))
    np.testing.assert_array_equal(codes[0],[1,2,3])
    np.testing.assert_array_equal(codes[1],[0,0,0])
    assert result['multi_or_non_biallelic_variants'] == 1


def test_archive_demo_is_a_conversion_example_not_a_training_benchmark(tmp_path):
    root = Path(__file__).resolve().parents[1]/'examples/data/dpcformer'
    result = prepare_csv(root/'sample_genotype_phenotype.csv',root/'sample_map.map',tmp_path/'out','Sucrose')
    assert result['n_samples'] == 50 and result['n_snps'] == 200
    assert result['multi_or_non_biallelic_variants'] == 74


@pytest.mark.parametrize('module',[diploid,cotton])
def test_fixed_test_is_excluded_from_every_inner_training_and_validation_split(module, monkeypatch, tmp_path):
    sample_count = 120
    phenotype = np.linspace(1,4,sample_count)
    snps = pd.DataFrame({'group_id':[1]})
    monkeypatch.setattr(module,'load_inputs',lambda args: ({},None,snps,None,phenotype,np.arange(sample_count),phenotype))
    observed = []
    def fold(args, number, split, *rest, evaluate_test=True):
        assert evaluate_test is False
        train, validation, test = map(np.asarray,split)
        assert not set(train)&set(test) and not set(validation)&set(test)
        assert not set(train)&set(validation)
        observed.append((train,validation,test))
        saved = Path(args.outdir)/'saved_models'/f'fold_{number}.pth'; saved.write_bytes(b'checkpoint')
        selected = Path(args.outdir)/f'selected_{number}.tsv'; selected.write_text('selected\n')
        return {'fold':number,'best_val_loss':float(number),'test_loss':1.,'test_pearson':number/20,
                'test_spearman':number/20,'model_path':str(saved),'selected_map_path':str(selected)}
    monkeypatch.setattr(module,'train_one_fold',fold)
    def test_after_selection(args, result, *rest):
        assert len(observed) == 10
        assert (Path(args.outdir)/'FS.model_selection.json').exists()
        return result
    monkeypatch.setattr(module,'evaluate_retained_fold',test_after_selection)
    args = SimpleNamespace(seed=42,cpu=True,outdir=str(tmp_path/'run'),trait='FS',pheno_col='trait',
         test_ratio=0.1,folds=10,validation_ratio=0.2,selector='minepy_mic',selection_workers=1,
         missing_phenotype_values='',min_available_mem_gb=0,learning_rate=.001,lr_patience=10,
         early_stopping_patience=20,selected_snp_count=1000,first_stage_k=1000)
    module.train_one_trait(args)
    assert len(observed)==10
    assert all(np.array_equal(row[2],observed[0][2]) for row in observed)
    assert set(np.concatenate([row[1] for row in observed])) == set(range(sample_count))-set(observed[0][2])
    import json
    protocol = json.loads((tmp_path/'run/FS.training_protocol.json').read_text())
    assert protocol['validation_fraction_within_train_val']==0.1
    assert (tmp_path/'run/FS.canonical_model_by_validation_loss.pth').read_bytes()==b'checkpoint'


@pytest.mark.parametrize('module',[diploid,cotton])
def test_selection_scaler_and_frozen_best_checkpoint_use_their_correct_scopes(module, monkeypatch, tmp_path):
    """Exercise optimizer/scaler/save/reload order with a tiny model, not a paper refit."""
    torch.set_num_threads(1)
    sample_count = 120
    phenotype = np.linspace(10,70,sample_count)
    train = np.arange(80); validation = np.arange(80,100); test = np.arange(100,120)
    snps = pd.DataFrame({'variant_index':[0,1], 'group_id':[1,1], 'Chr':['A01','D01'],
                         'Pos':[1,2], 'Ref':['A','A'],'Alt':['G','G'],'snp_id':['v1','v2']})
    selected = snps.assign(selected_group=1)
    def select(*,train_sample_cols,y_train_raw,fold_dir,**kwargs):
        np.testing.assert_array_equal(train_sample_cols,train)
        np.testing.assert_array_equal(y_train_raw,phenotype[train])
        path=fold_dir/'selected.tsv';selected.to_csv(path,sep='\t',index=False)
        return selected,path,None
    monkeypatch.setattr(module,'diploid_select_for_fold' if module is diploid else 'hierarchical_select_for_fold',select)
    tensor = torch.linspace(0,1,sample_count).reshape(sample_count,1,1,1)
    monkeypatch.setattr(module,'build_fold_tensor',lambda *args:tensor)
    class TinyModel(torch.nn.Module):
        def __init__(self,*args):
            super().__init__();self.layer=torch.nn.Linear(1,1)
        def forward(self,values):
            return self.layer(values.reshape(len(values),1))
    monkeypatch.setattr(module,'PhenotypePredictor_1v',TinyModel)
    original_evaluate = module.evaluate
    validation_states = []
    def evaluate(model,loader,criterion,device):
        if len(validation_states)<2:
            validation_states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
            return {'loss':float(len(validation_states)), 'pearson':0., 'spearman':0.}
        assert all(torch.equal(v,validation_states[0][k]) for k,v in model.state_dict().items())
        return original_evaluate(model,loader,criterion,device)
    monkeypatch.setattr(module,'evaluate',evaluate)
    (tmp_path/'saved_models').mkdir()
    args = SimpleNamespace(seed=42,trait='FS',min_available_mem_gb=0,preselected_snp_map='',
        selected_snp_count=128,first_stage_k=128,selector='minepy_mic',selection_workers=1,
        batch_size=32,learning_rate=.001,lr_patience=10,min_learning_rate=1e-8,epochs=2,
        min_delta=0.,early_stopping_patience=20)
    result = module.train_one_fold(args,1,(train,validation,test),np.zeros((2,sample_count),dtype=np.uint8),
                                   snps,np.arange(sample_count),phenotype,torch.device('cpu'),tmp_path)
    scaler=np.load(tmp_path/'fold_01/FS.fold_01.phenotype_scaler.npz')
    assert scaler['mean_'][0]==pytest.approx(phenotype[train].mean())
    assert int(scaler['n_samples_seen_'])==len(train)
    assert result['best_epoch']==1
    assert any(not torch.equal(validation_states[0][k],v) for k,v in validation_states[1].items())


def test_frozen_simulation_still_requires_a_frozen_map():
    from phenofuncrank_repro.simulation.frozen_training import train_one_fold
    with pytest.raises(ValueError,match='recorded selected SNP map'):
        train_one_fold(SimpleNamespace(preselected_snp_map=''))
