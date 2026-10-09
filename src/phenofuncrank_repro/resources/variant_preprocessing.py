"""Frozen repaired-model allele coordinates, dense conservation and gene features."""
import re
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from phenofuncrank_repro.plantvimpact.strict_model import read_redundancy_map


def parse_chromosome(value):
    match=re.search(r"(\d+)",str(value))
    return int(match.group(1)) if match else -1


def apply_dense_rice_conservation(frame, track_dir, cache=None):
    """Original uniform two-track extraction, with 1-based variant coordinates."""
    cache={} if cache is None else cache
    d=frame.copy()
    ids=d.snp_id.astype(str).str.rsplit("_",n=3,expand=True)
    for i,column in enumerate(["Chr","Pos","Ref","Alt"]):
        if column not in d:
            d[column]=ids[i]
        else:
            d[column]=d[column].astype("string").replace({
                "nan":pd.NA,"None":pd.NA,"":pd.NA}).fillna(ids[i])
    d["Pos"]=pd.to_numeric(d.Pos,errors="raise").astype("int64")
    chromosomes=d.Chr.astype(str).str.extract(r"(\d+)")[0].astype(int).to_numpy()
    positions=pd.to_numeric(d.Pos).to_numpy(np.int64)-1
    for track in ["phastCons","phyloP"]:
        values=np.full(len(d),np.nan,dtype=np.float32)
        for c in np.unique(chromosomes):
            key=(int(c),track)
            if key not in cache:
                cache[key]=np.memmap(
                    Path(track_dir)/f"chr{c:02d}.{track}.f32",
                    dtype="float32",mode="r")
            array=cache[key];idx=np.flatnonzero(chromosomes==c)
            if not ((positions[idx]>=0)&(positions[idx]<len(array))).all():
                raise ValueError("Variant lies outside its conservation track")
            values[idx]=array[positions[idx]]
        d["dna_cons__"+track]=values
    return d


def recompute_redundancy(frame, redundancy_tsv, prefix):
    """Unknown GeneID remains missing; a known gene with no hit receives zero."""
    d=frame.copy()
    red=read_redundancy_map(SimpleNamespace(redundancy_tsv=Path(redundancy_tsv)))
    target_prefix=prefix+"_" if prefix else ""
    gene=d.GeneID.astype("string")
    known=gene.notna()&~gene.isin(["","nan","None","\0"])
    d["GeneID"]=gene.mask(~known)
    for source,target in [
        ("has_homolog",target_prefix+"has_homolog"),
        ("functional_redundancy",target_prefix+"functional_redundancy"),
    ]:
        value=gene.map(red[source])
        d[target]=value.where(~known,value.fillna(0)).astype("float32")
    return d
