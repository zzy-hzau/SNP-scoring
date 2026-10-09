"""Shared strict DPCformer functions for the reported frozen-map simulations."""
from types import SimpleNamespace
from phenofuncrank_repro.models.dpcformer.train_diploid import (
    BASE_TO_VEC,
    available_memory_gb,
    build_fold_tensor,
    codes_to_8d,
    enforce_memory_reserve,
    evaluate,
    first_present,
    load_inputs,
    make_loader,
    parse_missing_phenotype_values,
    resolve_path,
    safe_corr,
    set_seed,
)
from phenofuncrank_repro.models.dpcformer.train_diploid import train_one_fold as _train_one_fold

def train_one_fold(args, *positional, **keywords):
    if not getattr(args, "preselected_snp_map", None):
        raise ValueError("Frozen simulation refits require the recorded selected SNP map")
    return _train_one_fold(args, *positional, **keywords)


def simulation_args(inputs_dir, outdir, trait, pheno_col):
    return SimpleNamespace(inputs_dir=inputs_dir, outdir=outdir, trait=trait,
                           pheno_col=pheno_col, missing_phenotype_values='',
                           environment_column='', environment='',
                           preselected_snp_map='')
