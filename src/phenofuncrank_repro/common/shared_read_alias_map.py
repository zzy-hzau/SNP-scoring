"""Identical source function; caller module supplies its recorded policy bindings."""
import argparse
import gzip
import math
from pathlib import Path
from collections import Counter, OrderedDict
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
KEYS = ["species", "kind", "snp_id", "Chr", "Pos", "Ref", "Alt"]
AA_STD = set("ACDEFGHIKLMNPQRSTVWY")

def read_alias_map(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"alias map not found: {path}")

    raw = pd.read_csv(path, sep="\t", dtype=str, header=None).fillna("")
    if raw.shape[1] < 2:
        raise RuntimeError(f"alias map must have at least 2 columns: {path}")


    h0 = str(raw.iloc[0, 0]).lower()
    h1 = str(raw.iloc[0, 1]).lower()
    if h0 in {"alias", "source", "id"} and h1 in {"protein_id", "target", "target_protein_id"}:
        raw = raw.iloc[1:].copy()

    aliases = {}
    for _, r in raw.iterrows():
        a = str(r.iloc[0]).strip()
        p = str(r.iloc[1]).strip()
        if not a or not p:
            continue
        pcore = core_id(p)
        if not pcore:
            continue

        for k in {a, core_id(a), p, pcore}:
            if k:
                aliases[k] = pcore

    return aliases
