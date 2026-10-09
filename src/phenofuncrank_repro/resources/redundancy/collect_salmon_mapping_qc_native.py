#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

ROOT = Path(args.root)
rows = []

for meta in sorted((ROOT / "04_quant").glob("*/aux_info/meta_info.json")):
    sample = meta.parent.parent.name

    with open(meta) as f:
        j = json.load(f)

    n_proc = (
        j.get("num_processed")
        or j.get("n_processed")
        or j.get("num_processed_fragments")
        or 0
    )

    n_map = (
        j.get("num_mapped")
        or j.get("n_pseudoaligned")
        or j.get("num_mapped_fragments")
        or 0
    )

    pct = j.get("percent_mapped")

    if pct is None:
        pct = 100.0 * n_map / n_proc if n_proc else np.nan

    rows.append({
        "sample": sample,
        "num_processed": n_proc,
        "num_mapped": n_map,
        "percent_mapped": pct,
    })

df = pd.DataFrame(rows)
df.to_csv(args.out, sep="\t", index=False)

print("[OUT]", args.out)
print("[samples]", df.shape[0])
print(df["percent_mapped"].describe())

print("\n[lowest mapping samples]")
print(df.sort_values("percent_mapped").head(40).to_string(index=False))

print("\n[threshold counts]")
for t in [20, 30, 40, 50, 60, 70, 80]:
    print(f">= {t}%:", int((df["percent_mapped"] >= t).sum()))
