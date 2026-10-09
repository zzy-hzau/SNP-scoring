#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--prefix", required=True)
ap.add_argument("--expr", required=True)
ap.add_argument("--tpm-cutoff", type=float, default=1.0)
args = ap.parse_args()

ROOT = Path(args.root)
PREFIX = args.prefix
EXPR = Path(args.expr)
TPM_CUTOFF = args.tpm_cutoff

COL = ROOT / f"{PREFIX}.collinearity"
DIAMOND = ROOT / f"{PREFIX}.diamond.extended.tsv"
GENE_INFO = ROOT / f"{PREFIX}.gene_info.tsv"

OUT_PAIR = ROOT / f"{PREFIX}.syntenic_pairs.with_expr_redundancy.QC50.cotton_strict.tsv"
OUT_FINAL = ROOT / f"{PREFIX}.paralog_redundancy_features_for_GP.QC50.cotton_strict.tsv"

print("[ROOT]", ROOT)
print("[PREFIX]", PREFIX)
print("[EXPR]", EXPR)
print("[COL]", COL)
print("[DIAMOND]", DIAMOND)
print("[GENE_INFO]", GENE_INFO)


expr = pd.read_csv(EXPR, sep="\t")
if "gene_id" not in expr.columns:
    raise SystemExit("[ERROR] expression matrix must contain gene_id column")

expr["gene_id"] = expr["gene_id"].astype(str)
expr = expr.drop_duplicates("gene_id")

sample_cols = [c for c in expr.columns if c != "gene_id"]
expr_mat = expr.set_index("gene_id")[sample_cols].astype(float)
log_expr = np.log2(expr_mat + 1.0)

print("[expression genes]", expr_mat.shape[0])
print("[expression samples]", expr_mat.shape[1])


gene_info = pd.read_csv(GENE_INFO, sep="\t", dtype=str)
gene_info["gene_id"] = gene_info["gene_id"].astype(str)
all_genes = gene_info["gene_id"].drop_duplicates().tolist()
known_genes = set(all_genes)

print("[gene_info genes]", len(all_genes))


pairs = set()

with open(COL) as f:
    for line in f:
        if not line.strip() or line.startswith("#"):
            continue

        toks = line.strip().split()


        gene_tokens = [t for t in toks if t in known_genes]

        if len(gene_tokens) >= 2:
            a, b = gene_tokens[0], gene_tokens[1]
        elif len(toks) >= 4:
            a, b = toks[2], toks[3]
        else:
            continue

        if a == b:
            continue

        if a not in known_genes or b not in known_genes:
            continue

        x, y = sorted([a, b])
        pairs.add((x, y))

pairs = sorted(pairs)
print("[syntenic pairs parsed]", len(pairs))

if not pairs:
    raise SystemExit("[ERROR] no syntenic pairs parsed")


cols = [
    "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore", "qlen", "slen"
]

dia = pd.read_csv(DIAMOND, sep="\t", names=cols, dtype={"qseqid": str, "sseqid": str})
dia = dia[dia["qseqid"] != dia["sseqid"]].copy()

for c in ["pident", "length", "bitscore", "qlen", "slen"]:
    dia[c] = pd.to_numeric(dia[c], errors="coerce")

dia["qcov"] = dia["length"] / dia["qlen"]
dia["scov"] = dia["length"] / dia["slen"]
dia["mincov"] = dia[["qcov", "scov"]].min(axis=1)
dia["seq_weight"] = (dia["pident"] / 100.0) * dia["mincov"]

dia["gene_a"] = dia[["qseqid", "sseqid"]].min(axis=1)
dia["gene_b"] = dia[["qseqid", "sseqid"]].max(axis=1)

dia_best = (
    dia.sort_values(["gene_a", "gene_b", "seq_weight", "bitscore"],
                    ascending=[True, True, False, False])
       .drop_duplicates(["gene_a", "gene_b"])
       [["gene_a", "gene_b", "pident", "qcov", "scov", "mincov", "seq_weight", "bitscore"]]
)

seq_map = {
    (r.gene_a, r.gene_b): r
    for r in dia_best.itertuples(index=False)
}


rows = []

for i, (a, b) in enumerate(pairs, 1):
    if i % 5000 == 0:
        print(f"[pair] {i}/{len(pairs)}")

    m = seq_map.get((a, b), None)

    if a in expr_mat.index and b in expr_mat.index:
        x_raw = expr_mat.loc[a].values.astype(float)
        y_raw = expr_mat.loc[b].values.astype(float)

        x = log_expr.loc[a].values.astype(float)
        y = log_expr.loc[b].values.astype(float)

        valid = np.isfinite(x) & np.isfinite(y)

        if valid.sum() >= 4 and np.std(x[valid]) > 0 and np.std(y[valid]) > 0:
            r, pval = pearsonr(x[valid], y[valid])
        else:
            r, pval = 0.0, 1.0

        breadth = float(((x_raw > TPM_CUTOFF) | (y_raw > TPM_CUTOFF)).mean())
        both_in_expr = 1
    else:
        r, pval, breadth = 0.0, 1.0, 0.0
        both_in_expr = 0

    functional_redundancy = r * breadth

    rows.append({
        "gene_a": a,
        "gene_b": b,
        "both_in_expr": both_in_expr,
        "expr_corr": r,
        "expr": r,
        "p_value": pval,
        "breadth": breadth,
        "functional_redundancy": functional_redundancy,
        "paralog_redundancy": functional_redundancy,
        "protein_identity": getattr(m, "pident", np.nan) if m is not None else np.nan,
        "qcov": getattr(m, "qcov", np.nan) if m is not None else np.nan,
        "scov": getattr(m, "scov", np.nan) if m is not None else np.nan,
        "mincov": getattr(m, "mincov", np.nan) if m is not None else np.nan,
        "seq_weight": getattr(m, "seq_weight", np.nan) if m is not None else np.nan,
        "bitscore": getattr(m, "bitscore", np.nan) if m is not None else np.nan,
    })

pair_df = pd.DataFrame(rows)
pair_df.to_csv(OUT_PAIR, sep="\t", index=False)

print("[OUT_PAIR]", OUT_PAIR)
print("[pair rows]", pair_df.shape[0])
print("[both_in_expr pairs]", int(pair_df["both_in_expr"].sum()))


g1 = pair_df.rename(columns={"gene_a": "gene_id", "gene_b": "paralog_gene"}).copy()
g2 = pair_df.rename(columns={"gene_b": "gene_id", "gene_a": "paralog_gene"}).copy()

gene_pair = pd.concat([g1, g2], ignore_index=True)
gene_pair["has_homolog"] = 1
gene_pair["has_paralog"] = 1


n_partner = gene_pair.groupby("gene_id")["paralog_gene"].nunique().rename("n_syntenic_partners")
min_red = gene_pair.groupby("gene_id")["functional_redundancy"].min().rename("min_pair_redundancy")
mean_red = gene_pair.groupby("gene_id")["functional_redundancy"].mean().rename("mean_pair_redundancy")


gene_pair = gene_pair.sort_values(
    ["gene_id", "functional_redundancy", "seq_weight", "bitscore"],
    ascending=[True, False, False, False]
)

best = gene_pair.drop_duplicates("gene_id").copy()
best = best.merge(n_partner, on="gene_id", how="left")
best = best.merge(min_red, on="gene_id", how="left")
best = best.merge(mean_red, on="gene_id", how="left")

final = pd.DataFrame({"gene_id": all_genes})
final = final.merge(gene_info, on="gene_id", how="left")
final = final.merge(best, on="gene_id", how="left")

for c in [
    "has_homolog", "has_paralog", "both_in_expr",
    "expr", "expr_corr", "breadth",
    "functional_redundancy", "paralog_redundancy",
    "n_syntenic_partners", "min_pair_redundancy", "mean_pair_redundancy"
]:
    if c in final.columns:
        final[c] = final[c].fillna(0)

for c in ["p_value"]:
    if c in final.columns:
        final[c] = final[c].fillna(1.0)

for c in ["has_homolog", "has_paralog", "both_in_expr", "n_syntenic_partners"]:
    if c in final.columns:
        final[c] = final[c].astype(int)


preferred = [
    "gene_id", "chrom", "start", "end", "strand", "transcript_id", "protein_len",
    "has_homolog", "has_paralog", "paralog_gene", "both_in_expr",
    "expr", "expr_corr", "p_value", "breadth",
    "functional_redundancy", "paralog_redundancy",
    "n_syntenic_partners", "min_pair_redundancy", "mean_pair_redundancy",
    "protein_identity", "qcov", "scov", "mincov", "seq_weight", "bitscore"
]
cols = [c for c in preferred if c in final.columns] + [c for c in final.columns if c not in preferred]
final = final[cols]

final.to_csv(OUT_FINAL, sep="\t", index=False)

print("[OUT_FINAL]", OUT_FINAL)
print("[final rows]", final.shape[0])
print("[has_homolog genes]", int(final["has_homolog"].sum()))
print("[both_in_expr genes]", int(final["both_in_expr"].sum()))

print("\n[paralog_redundancy overall]")
print(final["paralog_redundancy"].describe())

print("\n[paralog_redundancy has_homolog=1]")
print(final.loc[final["has_homolog"] == 1, "paralog_redundancy"].describe())

print("\n[top positive]")
show = ["gene_id", "paralog_gene", "expr_corr", "breadth", "paralog_redundancy", "seq_weight", "n_syntenic_partners"]
print(final.sort_values("paralog_redundancy", ascending=False).head(15)[show].to_string(index=False))

print("\n[top negative]")
print(final.sort_values("paralog_redundancy", ascending=True).head(15)[show].to_string(index=False))
