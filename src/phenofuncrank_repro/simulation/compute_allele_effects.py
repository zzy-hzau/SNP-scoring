#!/usr/bin/env python3
"""Original group-cached simulation allele effects over each fold's fixed test cohort."""
from phenofuncrank_repro.common.paths import project_root, package_dir
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
import torch

ROOT = project_root(__file__)
from phenofuncrank_repro.models.dpcformer.inputs import read_genotype_memmap
from phenofuncrank_repro.prioritization.allele_swap import _BASE_TO_VEC, _load_fold_selected_map, _load_fold_scaler, _load_fold_splits, _load_fold_model, _build_input_tensor, group_cached_deltas


def compute(inputs_dir,model_dir,trait,out_dir,*,folds=range(1,11),
            device="cpu",snp_chunk=128,selected_count=1000):
    genotype,metadata=read_genotype_memmap(inputs_dir)
    n_groups=int(metadata["n_groups"])
    device=torch.device(device)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    records=[];timings=[]
    for fold in folds:
        started=time.monotonic()
        selected=_load_fold_selected_map(Path(model_dir),trait,fold)
        scaler=_load_fold_scaler(Path(model_dir),trait,fold)
        test=_load_fold_splits(Path(model_dir),trait,fold)
        if selected is None or scaler is None or test is None:
            raise FileNotFoundError("Requested fold lacks selected map/scaler/test cohort")
        model=_load_fold_model(Path(model_dir),trait,fold,n_groups,selected_count,device)
        tensor=_build_input_tensor(genotype,selected,test,selected_count,n_groups).to(device)
        items=[];meta_rows=[]
        for row in selected.itertuples(index=False):
            ref=str(row.Ref).upper();alt=str(row.Alt).upper()
            if ref not in _BASE_TO_VEC or alt not in _BASE_TO_VEC:continue
            items.append((int(row.group_id)-1,int(row.selected_rank)-1,ref,alt))
            meta_rows.append((str(row.snp_id),str(row.chromosome),int(row.Pos)))
        deltas=group_cached_deltas(model,tensor,items,_BASE_TO_VEC,device,
                                   snp_chunk=snp_chunk)
        _,scale=scaler
        for (snp_id,chrom,pos),(_,_,ref,alt),delta in zip(meta_rows,items,deltas):
            records.append({"snp_id":snp_id,"chrom":chrom,"pos":pos,"ref":ref,
                            "alt":alt,"fold":fold,"delta_y":float(delta*scale)})
        del tensor,model
        if device.type!="cpu":torch.cuda.empty_cache()
        timings.append({"fold":fold,"test_samples":len(test),
                        "selected_snp_records":len(items),
                        "elapsed_seconds":time.monotonic()-started})
        print(json.dumps(timings[-1]),flush=True)
    frame=pd.DataFrame(records)
    if frame.empty:raise ValueError("No allele effects calculated")
    aggregate=frame.groupby("snp_id").agg(
        dpc_additive_effect_mean=("delta_y","mean"),
        fold_effect_sd=("delta_y",lambda s:float(np.std(s,ddof=1)) if len(s)>1 else np.nan),
        n_evaluable_folds=("delta_y","size"),chrom=("chrom","first"),pos=("pos","first"),
        schema_ref=("ref","first"),schema_alt=("alt","first")).reset_index()
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/"dpc_fold_level.tsv",sep="\t",index=False)
    aggregate.to_csv(out/"dpc_master_sim.tsv",sep="\t",index=False)
    profile={"scope":"Simulation scaled-mean allele contrast multiplied by "
                    "fold training phenotype scale; saved test cohort only.",
             "folds":list(folds),"snp_chunk":snp_chunk,
             "selected_count":selected_count,"timings":timings,
             "reference_alt_layout":"selected map rank and exact original nucleotide vectors",
             "mean_inverse_transformed_predictions_used":False}
    (out/"allele_effect_protocol.json").write_text(json.dumps(profile,indent=2)+"\n")
    return frame,aggregate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir",required=True,type=Path)
    parser.add_argument("--model-dir",required=True,type=Path)
    parser.add_argument("--trait",required=True)
    parser.add_argument("--out-dir",required=True,type=Path)
    parser.add_argument("--folds",nargs="+",type=int,default=list(range(1,11)))
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--snp-chunk",default=128,type=int)
    parser.add_argument("--selected-count",default=1000,type=int)
    args=parser.parse_args()
    compute(args.inputs_dir,args.model_dir,args.trait,args.out_dir,
            folds=args.folds,device=args.device,snp_chunk=args.snp_chunk,
            selected_count=args.selected_count)


if __name__=="__main__":
    main()
