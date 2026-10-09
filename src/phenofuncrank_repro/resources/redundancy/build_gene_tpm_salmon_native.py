#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--tx2gene", required=True)
ap.add_argument("--qc", required=True)
ap.add_argument("--status", required=True)
ap.add_argument("--out-prefix", required=True)
args = ap.parse_args()

ROOT = Path(args.root)
TX2GENE = Path(args.tx2gene)
QC = Path(args.qc)
STATUS = Path(args.status)
OUT_PREFIX = Path(args.out_prefix)

print("[ROOT]", ROOT)
print("[TX2GENE]", TX2GENE)
print("[QC]", QC)
print("[STATUS]", STATUS)
print("[OUT_PREFIX]", OUT_PREFIX)

tx2gene = pd.read_csv(TX2GENE, sep="\t", dtype=str).fillna("")
tx2gene = tx2gene[["tx_id", "gene_id"]].drop_duplicates()
tx2gene = tx2gene.drop_duplicates("tx_id")

print("[tx2gene rows]", tx2gene.shape[0])
print("[unique genes in tx2gene]", tx2gene["gene_id"].nunique())

status = pd.read_csv(STATUS, sep="\t", dtype=str).fillna("")
status = status[status["status"].isin(["OK", "SKIP_COMPLETE"])].copy()
samples = status["sample"].tolist()

print("[requested OK samples]", len(samples))

qc = pd.read_csv(QC, sep="\t", dtype={"sample": str})
qc_map = qc.set_index("sample")["percent_mapped"].to_dict()

def load_gene_tpm(sample):
    q = ROOT / "04_quant" / sample / "quant.sf"
    if not q.exists():
        return None

    df = pd.read_csv(q, sep="\t", usecols=["Name", "TPM"], dtype={"Name": str})
    df = df.merge(tx2gene, left_on="Name", right_on="tx_id", how="inner")

    if df.empty:
        return pd.Series(dtype=float, name=sample)

    s = df.groupby("gene_id")["TPM"].sum()
    s.name = sample
    return s

series = []
missing = []
empty = []

for i, sample in enumerate(samples, 1):
    if i % 20 == 0:
        print(f"[load] {i}/{len(samples)}")

    s = load_gene_tpm(sample)

    if s is None:
        missing.append(sample)
        continue

    if len(s) == 0:
        empty.append(sample)
        continue

    series.append(s)

if not series:
    raise SystemExit("[ERROR] no usable quant.sf loaded")

mat = pd.concat(series, axis=1).fillna(0.0)
mat.index.name = "gene_id"
mat = mat.reset_index()

print("[used samples]", len(series))
print("[missing quant]", len(missing))
print("[empty tx2gene merge]", len(empty))
print("[matrix shape]", mat.shape)

OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)

def write_matrix(sample_subset, suffix):
    cols = ["gene_id"] + sample_subset
    out = Path(str(OUT_PREFIX) + f".{suffix}.tsv")
    mat[cols].to_csv(out, sep="\t", index=False)
    print("[OUT]", out, "samples:", len(sample_subset), "shape:", (mat.shape[0], len(cols)))

all_samples = [c for c in mat.columns if c != "gene_id"]
write_matrix(all_samples, "all")

for cutoff in [40, 50, 60]:
    keep = [s for s in all_samples if qc_map.get(s, -1) >= cutoff]
    write_matrix(keep, f"QC{cutoff}")

print("[DONE]")
