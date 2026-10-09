"downstream union-find LD components; no model inference or training."
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def build_ld_components(expanded_df: pd.DataFrame, pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    if expanded_df.empty:
        return expanded_df.assign(
            ld_component_id=None, is_core=False,
            r2_to_best_core=np.nan, phase_signed_r=np.nan,
        )

    frame = expanded_df.copy()
    frame["chrom"] = frame["chrom"].astype(str)

    if "level1_class" not in frame:
        frame["level1_class"] = "core"
    if "core_snp_id" not in frame:
        frame["core_snp_id"] = frame["user_snp_id"]
    if "r2_to_core" not in frame:
        frame["r2_to_core"] = np.where(frame["level1_class"] == "core", 1.0, np.nan)
    if "dpc_additive_effect_mean" not in frame:
        frame["dpc_additive_effect_mean"] = 0.0


    parent: dict[str, str] = {}

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
            return x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra


    for row in frame.itertuples(index=False):
        snp = str(row.user_snp_id)
        find(snp)
        if row.level1_class != "core":
            core = str(row.core_snp_id if row.core_snp_id is not None else snp)
            r2 = float(row.r2_to_core if row.r2_to_core is not None else 0)
            if r2 >= .6 and core != snp:
                find(core)
                union(core, snp)

    if pairs is not None:
        for edge in pairs.itertuples(index=False):
            if float(edge.r2_to_core) >= .6:
                union(str(edge.core_snp_id),str(edge.proxy_snp_id))


    comp_by_chrom: dict[str, dict[str, str]] = {}
    for row in frame.itertuples(index=False):
        chrom = str(row.chrom)
        snp = str(row.user_snp_id)
        root = find(snp)
        comp_by_chrom.setdefault(chrom, {})[snp] = root


    component_counter = 0
    root_to_cid: dict[str, int] = {}
    component_ids: list[int | None] = []
    for row in frame.itertuples(index=False):
        chrom = str(row.chrom)
        snp = str(row.user_snp_id)
        root = comp_by_chrom.get(chrom, {}).get(snp)
        if root is None:
            component_ids.append(None)
            continue
        if root not in root_to_cid:
            component_counter += 1
            root_to_cid[root] = component_counter
        component_ids.append(root_to_cid[root])

    frame["ld_component_id"] = component_ids

    frame["is_core"] = frame["level1_class"].eq("core")
    frame["r2_to_best_core"] = np.where(frame.is_core, 1., np.nan)

    
    frame["phase_signed_r"] = np.where(frame.is_core, 1., np.nan)

    return frame

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--pairs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    frame=pd.read_csv(args.candidates,sep='\t')
    pairs=pd.read_csv(args.pairs,sep='\t')
    result=build_ld_components(frame,pairs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,sep='\t',index=False)

if __name__=='__main__':
    main()
