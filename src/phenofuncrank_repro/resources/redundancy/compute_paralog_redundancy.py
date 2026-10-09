#!/usr/bin/env python3
"Compute paralog redundancy for diploid species (maize / tomato / rice)."
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent

from phenofuncrank_repro.resources.utils.common import log, open_text


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--collinearity-pairs", required=True, type=Path,
                    help="TSV from parse_mcscanx_collinearity.py.")
    ap.add_argument("--tpm-matrix", required=True, type=Path,
                    help="Wide TSV: gene_id + sample TPM columns.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output TSV with per-gene paralog redundancy.")
    ap.add_argument("--tpm-cutoff", type=float, default=1.0,
                    help="TPM threshold for 'expressed' (default 1.0).")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Process only the first N syntenic pairs (toy/test mode).")
    return ap.parse_args(argv)


def safe_pearson(x, y):
    """Signed Pearson correlation of two vectors; returns 0.0 when undefined.

    Undefined cases (fewer than 4 finite pairs, or zero variance in either
    vector) yield 0.0 rather than NaN, per the spec.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return 0.0
    xv = x[mask]
    yv = y[mask]
    if xv.std() == 0.0 or yv.std() == 0.0:
        return 0.0
    r = float(np.corrcoef(xv, yv)[0, 1])
    if not np.isfinite(r):
        return 0.0
    return r


def main(argv=None):
    args = parse_args(argv)

    if not args.collinearity_pairs.exists():
        raise SystemExit(
            f"[ERROR] collinearity-pairs not found: {args.collinearity_pairs}"
        )
    if not args.tpm_matrix.exists():
        raise SystemExit(f"[ERROR] tpm-matrix not found: {args.tpm_matrix}")

    args.output.parent.mkdir(parents=True, exist_ok=True)


    tpm = pd.read_csv(args.tpm_matrix, sep="\t", dtype={"gene_id": str})
    if "gene_id" not in tpm.columns:
        raise SystemExit(
            f"[ERROR] tpm-matrix must have a 'gene_id' column; "
            f"got {list(tpm.columns)}"
        )
    tpm = tpm.drop_duplicates("gene_id")
    sample_cols = [c for c in tpm.columns if c != "gene_id"]
    if not sample_cols:
        raise SystemExit("[ERROR] tpm-matrix has no sample columns")
    tpm_mat = tpm.set_index("gene_id")[sample_cols].astype(float)
    log_tpm = np.log2(tpm_mat + 1.0)
    gene_set = set(tpm_mat.index)
    log(f"TPM matrix: {tpm_mat.shape[0]} genes x {tpm_mat.shape[1]} samples")


    pairs_df = pd.read_csv(args.collinearity_pairs, sep="\t", dtype=str)
    for required in ("gene_a", "gene_b"):
        if required not in pairs_df.columns:
            raise SystemExit(
                f"[ERROR] collinearity-pairs must have '{required}' column; "
                f"got {list(pairs_df.columns)}"
            )
    pairs_df = pairs_df[pairs_df["gene_a"] != pairs_df["gene_b"]].copy()
    if args.max_rows is not None:
        pairs_df = pairs_df.head(args.max_rows)
    log(f"Collinearity pairs loaded: {len(pairs_df)}")


    
    gene_pairs = {}

    n_both = 0
    n_one_missing = 0
    for _, row in pairs_df.iterrows():
        a, b = row["gene_a"], row["gene_b"]
        if a not in gene_set or b not in gene_set:

            

            n_one_missing += 1
            for g, p in ((a, b), (b, a)):
                if g in gene_set:
                    gene_pairs.setdefault(g, []).append((p, 0.0, 0.0, 0.0))
            continue
        n_both += 1

        x_raw = tpm_mat.loc[a].values.astype(float)
        y_raw = tpm_mat.loc[b].values.astype(float)
        x_log = log_tpm.loc[a].values.astype(float)
        y_log = log_tpm.loc[b].values.astype(float)

        expr_corr = safe_pearson(x_log, y_log)
        breadth = float(((x_raw > args.tpm_cutoff) |
                         (y_raw > args.tpm_cutoff)).mean())
        pair_redundancy = expr_corr * breadth

        for g, p in ((a, b), (b, a)):
            gene_pairs.setdefault(g, []).append(
                (p, expr_corr, breadth, pair_redundancy)
            )

    log(f"Pairs with both genes in TPM: {n_both}; one missing: {n_one_missing}")


    rows = []
    n_with_paralog = 0
    n_without = 0
    for gene in tpm_mat.index:
        partners = gene_pairs.get(gene)
        if not partners:
            n_without += 1
            rows.append({
                "gene_id": gene,
                "paralog_redundancy": 0.0,
                "has_paralog": 0,
                "n_paralogs": 0,
                "best_pair_expr_corr": "NA",
                "best_pair_breadth": "NA",
            })
            continue
        n_with_paralog += 1

        best = max(partners, key=lambda t: t[3])
        _, best_corr, best_breadth, best_red = best
        n_paralogs = len(set(p for p, _, _, _ in partners))
        rows.append({
            "gene_id": gene,
            "paralog_redundancy": float(best_red),
            "has_paralog": 1,
            "n_paralogs": n_paralogs,
            "best_pair_expr_corr": float(best_corr),
            "best_pair_breadth": float(best_breadth),
        })

    out_df = pd.DataFrame(rows, columns=[
        "gene_id", "paralog_redundancy", "has_paralog", "n_paralogs",
        "best_pair_expr_corr", "best_pair_breadth",
    ])

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    out_df.to_csv(tmp, sep="\t", index=False)
    tmp.replace(args.output)

    log(f"Wrote {len(out_df)} genes to {args.output}")
    log(f"  with_paralog={n_with_paralog} without_paralog={n_without}")


    vals = out_df.loc[out_df["has_paralog"] == 1, "paralog_redundancy"].astype(float)
    if len(vals) > 0:
        n_neg = int((vals < 0).sum())
        n_zero = int((vals == 0).sum())
        n_pos = int((vals > 0).sum())
        log(f"  paralog_redundancy (has_paralog=1): "
            f"min={vals.min():.4f} max={vals.max():.4f} "
            f"mean={vals.mean():.4f} std={vals.std():.4f}")
        log(f"  negative={n_neg} zero={n_zero} positive={n_pos} "
            f"(negatives preserved, not truncated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
