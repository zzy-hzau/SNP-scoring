#!/usr/bin/env python3


"Build cotton At/Dt-aware P_mut rate cache from existing 01_calc_pmut_fast_cotton.py functions."

from __future__ import annotations

import argparse
import importlib.util
import pickle
from pathlib import Path

import numpy as np


def plainify(x):
    """Recursively convert defaultdict/Counter-like objects into plain dict/list scalars."""
    from collections import Counter, defaultdict
    if isinstance(x, dict):
        return {plainify(k): plainify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plainify(v) for v in x]
    return x


def load_module(script_path: str):
    script_path = str(script_path)
    spec = importlib.util.spec_from_file_location("cotton_calc_pmut_legacy", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calc-script", required=True, help="Path to 01_calc_pmut_fast_cotton.py")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--sites-4d", required=True)
    ap.add_argument("--n-chr", type=int, default=13)
    ap.add_argument("--out", required=True)
    ap.add_argument("--global-json", default=None)
    args = ap.parse_args()

    mod = load_module(args.calc_script)

    required = ["build_all_fold_rate_tables_fast", "build_rate_table_excluding_chrnums"]
    missing = [x for x in required if not hasattr(mod, x)]
    if missing:
        raise RuntimeError(f"Legacy calc script missing functions: {missing}")

    print("[1/3] Building fold-specific At/Dt P_mut rate tables...", flush=True)
    rate_tables, genome_total, genome_by_chrnum, mut_total, mut_by_chrnum = mod.build_all_fold_rate_tables_fast(
        fasta_path=args.fasta,
        vcf_path=args.vcf,
        sites_4d_path=args.sites_4d,
        n_chr=args.n_chr,
    )

    print("[2/3] Building global At/Dt P_mut rate table...", flush=True)
    global_rate_table = mod.build_rate_table_excluding_chrnums(
        genome_total,
        genome_by_chrnum,
        mut_total,
        mut_by_chrnum,
        exclude_chrs=[],
    )

    payload = plainify({
        "rate_tables": rate_tables,
        "genome_total": genome_total,
        "genome_by_chrnum": genome_by_chrnum,
        "mut_total": mut_total,
        "mut_by_chrnum": mut_by_chrnum,
        "global_rate_table": global_rate_table,
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "wb") as f:
        pickle.dump(payload, f)

    vals = np.array(list(global_rate_table.values()), dtype=float)
    print("[3/3] Saved:", out, flush=True)
    print(f"global_rate_table keys = {len(global_rate_table):,}", flush=True)
    if len(vals):
        print(
            "global P_mut summary: "
            f"min={vals.min():.3e}, median={np.median(vals):.3e}, "
            f"mean={vals.mean():.3e}, max={vals.max():.3e}",
            flush=True,
        )


    if args.global_json:
        import json
        gj = Path(args.global_json)
        gj.parent.mkdir(parents=True, exist_ok=True)
        with open(gj, "w") as f:
            json.dump(global_rate_table, f, indent=2)
        print("Saved global json:", gj, flush=True)


if __name__ == "__main__":
    main()
