"20260927 QTL overlap and pooled exact-test profile."
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

traits = {"Rice PH": "Rice plant height", "Cotton FS": "Cotton fibre strength", "Chickpea HSW": "Chickpea hundred-seed weight", "Chickpea PW": "Chickpea plant width"}

def overlaps(qtls, *, rank, gwas, ld):
 out=[]
 for tr,method,n,ch,left,right in ld:
  lead=(rank if method=='PhenoFuncRank' else gwas)[(tr,n)];pos=lead[3]
  for q in qtls:
   if traits[q[1]]!=tr or q[2]!=ch:continue
   if q[3]<=pos<=q[4]:out.append([tr,method,n,'SNP',q[0]])
   if left<=q[4] and right>=q[3]:out.append([tr,method,n,'LD interval',q[0]])
 return sorted(out)

def stats(rows):
 out=[]
 for mode in ['SNP','LD interval']:
  block=[]
  for k in [20,50,100]:
   a={r[4] for r in rows if r[1]=='PhenoFuncRank' and r[2]<=k and r[3]==mode};b={r[4] for r in rows if r[1]=='GWAS' and r[2]<=k and r[3]==mode};x=len(a-b);y=len(b-a);p=float(binomtest(x,x+y,.5,alternative='greater').pvalue) if x+y else 1.
   block.append({'overlap_definition':mode,'budget':k,'PhenoFuncRank_only':x,'GWAS_only':y,'both':len(a&b),'neither':34-len(a|b),'P':p})
  ps=np.array([r['P'] for r in block]);ix=np.argsort(ps);qs=np.empty(3);qs[ix]=np.minimum(1,np.minimum.accumulate((ps[ix]*3/np.arange(1,4))[::-1])[::-1])
  for r,q in zip(block,qs):r['BH_q']=float(q)
  out+=block
 return out


def evaluate(catalogue, candidates):
    if len(catalogue) != 34 or len({row[0] for row in catalogue}) != 34:
        raise ValueError("This frozen profile requires the 34 distinct catalogue regions")
    ranked, gwas, intervals = {}, {}, []
    for candidate in candidates:
        trait = traits[candidate["trait"].replace("_", " ")]
        method = "PhenoFuncRank" if candidate["method"] == "Framework" else candidate["method"]
        if method not in {"PhenoFuncRank", "GWAS"}:
            raise ValueError(f"Unknown method {method}")
        rank = int(candidate["rank"])
        row = [trait, rank, str(candidate["chrom"]), int(candidate["pos"]), candidate["ref"], candidate["alt"]]
        target = ranked if method == "PhenoFuncRank" else gwas
        if (trait, rank) in target:
            raise ValueError("Duplicate within-trait method/rank candidate")
        target[(trait, rank)] = row
        intervals.append([trait, method, rank, str(candidate["chrom"]), int(candidate["ld_start"]), int(candidate["ld_end"])])
    overlap_rows = overlaps(catalogue, rank=ranked, gwas=gwas, ld=intervals)
    cells = []
    for q in catalogue:
        for method in ["PhenoFuncRank", "GWAS"]:
            for mode in ["SNP", "LD interval"]:
                for k in [20, 50, 100]:
                    cells.append({"QTL ID": q[0], "Trait": q[1], "Method": method, "Overlap definition": mode, "Candidate budget": k, "Candidate count": len({r[2] for r in overlap_rows if r[4] == q[0] and r[1] == method and r[3] == mode and r[2] <= k})})
    return {"overlap_rows": overlap_rows, "statistics": stats(overlap_rows), "cells": cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON with catalogue and candidates; explicit frozen 34-region profile")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = evaluate(payload["catalogue"], payload["candidates"])
    result["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    result["computation_profile"] = "34_region_QTL_overlap_statistics_from_20260927_source"
    result["input_profile"] = payload.get("input_profile", "caller_supplied_catalogue")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "qtl_recovery.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(result["statistics"]).to_csv(args.output_dir / "pooled_statistics.tsv", sep="\t", index=False)
    pd.DataFrame(result["cells"]).to_csv(args.output_dir / "qtl_incidence_cells.tsv", sep="\t", index=False)
    pd.DataFrame(result["overlap_rows"], columns=["Trait", "Method", "Candidate rank", "Overlap definition", "QTL ID"]).to_csv(args.output_dir / "candidate_qtl_overlaps.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
