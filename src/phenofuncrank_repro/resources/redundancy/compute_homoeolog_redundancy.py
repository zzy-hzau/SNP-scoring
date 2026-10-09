#!/usr/bin/env python3
"Compute At-Dt homoeolog redundancy for cotton."
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent

from phenofuncrank_repro.resources.utils.common import log


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--homoeolog-pairs", required=True, type=Path,
                    help="TSV with columns gene_at and gene_dt.")
    ap.add_argument("--tpm-matrix", required=True, type=Path,
                    help="Wide TSV: gene_id + context TPM columns.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output TSV with per-gene homoeolog redundancy.")
    ap.add_argument("--tpm-cutoff", type=float, default=1.0,
                    help="TPM threshold for 'expressed' (default 1.0).")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Process only the first N homoeolog pairs (toy/test mode).")
    return ap.parse_args(argv)


def safe_pearson(x, y):
    """Signed Pearson correlation; returns 0.0 when undefined.

    Undefined cases (fewer than 4 finite pairs, or zero variance) yield 0.0.
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

    if not args.homoeolog_pairs.exists():
        raise SystemExit(
            f"[ERROR] homoeolog-pairs not found: {args.homoeolog_pairs}"
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
        raise SystemExit("[ERROR] tpm-matrix has no context columns")
    tpm_mat = tpm.set_index("gene_id")[sample_cols].astype(float)
    log_tpm = np.log2(tpm_mat + 1.0)
    gene_set = set(tpm_mat.index)
    log(f"TPM matrix: {tpm_mat.shape[0]} genes x {tpm_mat.shape[1]} contexts")


    pairs_df = pd.read_csv(args.homoeolog_pairs, sep="\t", dtype=str)
    for required in ("gene_at", "gene_dt"):
        if required not in pairs_df.columns:
            raise SystemExit(
                f"[ERROR] homoeolog-pairs must have '{required}' column; "
                f"got {list(pairs_df.columns)}"
            )
    pairs_df = pairs_df.dropna(subset=["gene_at", "gene_dt"])
    pairs_df = pairs_df[pairs_df["gene_at"] != pairs_df["gene_dt"]]
    if args.max_rows is not None:
        pairs_df = pairs_df.head(args.max_rows)

    homoeolog = {}
    for _, row in pairs_df.iterrows():
        at, dt = row["gene_at"], row["gene_dt"]

        homoeolog.setdefault(at, dt)
        homoeolog.setdefault(dt, at)
    log(f"Homoeolog pairs loaded: {len(pairs_df)} "
        f"({len(homoeolog)} genes with a partner)")


    rows = []
    n_with = 0
    n_without = 0
    for gene in tpm_mat.index:
        partner = homoeolog.get(gene)
        if partner is None or partner not in gene_set:
            n_without += 1
            rows.append({
                "gene_id": gene,
                "functional_redundancy": 0.0,
                "has_homolog": 0,
                "expr_corr": 0.0,
                "breadth": 0.0,
            })
            continue
        n_with += 1

        x_raw = tpm_mat.loc[gene].values.astype(float)
        y_raw = tpm_mat.loc[partner].values.astype(float)
        x_log = log_tpm.loc[gene].values.astype(float)
        y_log = log_tpm.loc[partner].values.astype(float)

        expr_corr = safe_pearson(x_log, y_log)
        breadth = float(((x_raw > args.tpm_cutoff) |
                         (y_raw > args.tpm_cutoff)).mean())
        functional_redundancy = 1 * expr_corr * breadth  

        rows.append({
            "gene_id": gene,
            "functional_redundancy": float(functional_redundancy),
            "has_homolog": 1,
            "expr_corr": float(expr_corr),
            "breadth": float(breadth),
        })

    out_df = pd.DataFrame(rows, columns=[
        "gene_id", "functional_redundancy", "has_homolog",
        "expr_corr", "breadth",
    ])

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    out_df.to_csv(tmp, sep="\t", index=False)
    tmp.replace(args.output)

    log(f"Wrote {len(out_df)} genes to {args.output}")
    log(f"  with_homolog={n_with} without_homolog={n_without}")


    vals = out_df.loc[out_df["has_homolog"] == 1, "functional_redundancy"].astype(float)
    if len(vals) > 0:
        n_neg = int((vals < 0).sum())
        n_zero = int((vals == 0).sum())
        n_pos = int((vals > 0).sum())
        log(f"  functional_redundancy (has_homolog=1): "
            f"min={vals.min():.4f} max={vals.max():.4f} "
            f"mean={vals.mean():.4f} std={vals.std():.4f}")
        log(f"  negative={n_neg} (subgenome antagonism) zero={n_zero} "
            f"positive={n_pos} (negatives preserved, not truncated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
