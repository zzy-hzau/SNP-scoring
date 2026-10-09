#!/usr/bin/env python3
"simulation fold refit with explicit resources/devices."
from __future__ import annotations
from phenofuncrank_repro.common.paths import project_root, package_dir
import argparse
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split,KFold

ROOT = project_root(__file__)
_CONTEXT={}


def initialize(inputs_dir,outdir,frozen_model_dir,trait,pheno_col,
               profile,devices,threads):
    import os
    os.environ["OMP_NUM_THREADS"]=str(threads)
    os.environ["MKL_NUM_THREADS"]=str(threads)
    torch.set_num_threads(threads)
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    from phenofuncrank_repro.simulation import frozen_training as trainer
    args=trainer.simulation_args(inputs_dir,outdir,trait,pheno_col)
    for key in ["folds","test_ratio","validation_ratio","selected_snp_count",
                "selector","selection_workers","epochs","batch_size",
                "learning_rate","min_learning_rate","lr_patience",
                "early_stopping_patience","min_delta","seed"]:
        setattr(args,key,profile[key])
    args.min_available_mem_gb=profile["execution"]["min_available_mem_gb"]
    metadata,genotype,snp_map,phenotypes,y_all,valid_cols,y_valid=trainer.load_inputs(args)
    all_local=np.arange(len(y_valid),dtype=np.int64)
    train_val,test=train_test_split(all_local,test_size=args.test_ratio,
                                    random_state=args.seed,shuffle=True)
    splitter=KFold(n_splits=args.folds,shuffle=True,random_state=args.seed)
    splits=[(train_val[tr],train_val[va],test)
            for tr,va in splitter.split(train_val)]
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    (out/"saved_models").mkdir(exist_ok=True)
    _CONTEXT.update(trainer=trainer,args=args,genotype=genotype,snp_map=snp_map,
        valid_cols=valid_cols,y_valid=y_valid,splits=splits,out=out,
        frozen=Path(frozen_model_dir),devices=devices)


def run_fold(fold):
    data=_CONTEXT
    args=SimpleNamespace(**vars(data["args"]))
    selected=data["frozen"]/f"fold_{fold:02d}"/f"{args.trait}.fold_{fold:02d}.selected_snp_map.tsv"
    args.preselected_snp_map=str(selected)
    device=torch.device(data["devices"][(fold-1)%len(data["devices"])])
    started=time.monotonic()
    result=data["trainer"].train_one_fold(args,fold,data["splits"][fold-1],
        data["genotype"],data["snp_map"],data["valid_cols"],data["y_valid"],
        device,data["out"])
    for key in ["model_path","selected_map_path","fold_dir"]:
        result[key]=str(Path(result[key]).relative_to(data["out"]))
    return {"fold":fold,"device_type":device.type,
            "elapsed_seconds":time.monotonic()-started,"result":result}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir",required=True,type=Path)
    parser.add_argument("--out-dir",required=True,type=Path)
    parser.add_argument("--frozen-model-dir",required=True,type=Path)
    parser.add_argument("--trait",required=True)
    parser.add_argument("--pheno-col",required=True)
    parser.add_argument("--profile",type=Path,
                        default=ROOT/"configs/simulation_network_refit_20260918.json")
    parser.add_argument("--devices",nargs="+",default=["cpu"])
    parser.add_argument("--workers",default=1,type=int)
    parser.add_argument("--threads",default=4,type=int)
    parser.add_argument("--only-folds",nargs="+",type=int)
    args=parser.parse_args()
    profile=json.loads(args.profile.read_text())
    selected=args.only_folds or list(range(1,profile["folds"]+1))
    if any(x<1 or x>profile["folds"] for x in selected):
        parser.error("Requested fold is outside the profile")
    args.out_dir.mkdir(parents=True,exist_ok=False)
    initialization=(str(args.inputs_dir),str(args.out_dir),
                    str(args.frozen_model_dir),args.trait,args.pheno_col,
                    profile,args.devices,args.threads)
    if args.workers==1:
        initialize(*initialization)
        results=[run_fold(fold) for fold in selected]
    else:

        with mp.get_context("spawn").Pool(args.workers,initializer=initialize,
                                          initargs=initialization) as pool:
            results=pool.map(run_fold,selected,chunksize=1)
    report={"status":"COMPLETED","scope":"Frozen real-run SNP maps, newly fit "
            "network on the supplied local simulated phenotype.",
            "profile":profile,"trained_folds":selected,"results":results}
    (args.out_dir/"simulation_refit_results.json").write_text(json.dumps(report,indent=2)+"\n")
    metrics=pd.DataFrame([row["result"] for row in results])
    metrics.to_csv(args.out_dir/f"{args.trait}.fold_metrics.tsv",sep="\t",index=False)


if __name__=="__main__":
    main()
