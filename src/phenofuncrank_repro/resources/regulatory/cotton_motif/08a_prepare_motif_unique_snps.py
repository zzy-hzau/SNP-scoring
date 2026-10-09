#!/usr/bin/env python3


import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np


CONTEXT_COLS = [
    "snp_id", "Chr", "Pos", "Ref", "Alt",

    "is_promoter", "is_UTR5", "is_UTR3", "is_splice_near",
    "promoter_core_flag", "promoter_proximal_flag", "promoter_distal_flag",
    "region_type", "region_subtype",

    "atac_overlap_any",
    "h3k27ac_overlap_any",
    "active_regulatory_flag",
    "active_enhancer_like_flag",
    "promoter_with_activity",
]


def choose_candidates(df, scope):
    def bcol(c):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0) > 0
        return pd.Series(False, index=df.index)

    is_promoter = bcol("is_promoter")
    is_utr = bcol("is_UTR5") | bcol("is_UTR3")
    is_splice = bcol("is_splice_near")
    promoter_core = bcol("promoter_core_flag")
    promoter_prox = bcol("promoter_proximal_flag")
    active = bcol("active_regulatory_flag")
    enhancer_like = bcol("active_enhancer_like_flag")
    atac = bcol("atac_overlap_any")

    if scope == "activity_or_functional":
        cand = active | is_promoter | is_utr | is_splice
    elif scope == "atac_or_core":
        cand = atac | enhancer_like | promoter_core | promoter_prox | is_utr | is_splice
    elif scope == "all":
        cand = pd.Series(True, index=df.index)
    else:
        raise ValueError(scope)

    return cand.astype("int8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scope", default="activity_or_functional",
                    choices=["activity_or_functional", "atac_or_core", "all"])
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted(in_root.glob("fold*"))
    if not fold_dirs:
        raise RuntimeError(f"No fold dirs found under {in_root}")

    store = {}
    total = 0

    for fd in fold_dirs:
        for sp in ["train", "val", "test"]:
            p = fd / f"{sp}.parquet"
            if not p.exists():
                continue

            schema_cols = list(pd.read_parquet(p, columns=None).columns)
            cols = [c for c in CONTEXT_COLS if c in schema_cols]
            if "snp_id" not in cols:
                raise RuntimeError(f"snp_id missing in {p}")

            df = pd.read_parquet(p, columns=cols)
            total += len(df)
            df = df.drop_duplicates("snp_id")

            before = len(store)
            for r in df.itertuples(index=False):
                d = r._asdict()
                sid = str(d["snp_id"])
                if sid not in store:
                    store[sid] = d

            after = len(store)
            print(f"[collect] {p} rows={len(df):,} unique_total={after:,} new={after-before:,}", flush=True)

    x = pd.DataFrame(list(store.values()))

    required = ["snp_id", "Chr", "Pos", "Ref", "Alt"]
    for c in required:
        if c not in x.columns:
            raise RuntimeError(f"required column missing: {c}")

    x["Chr"] = x["Chr"].astype(str)
    x["Pos"] = pd.to_numeric(x["Pos"], errors="coerce").astype("int64")
    x["Ref"] = x["Ref"].astype(str).str.upper()
    x["Alt"] = x["Alt"].astype(str).str.upper()


    for c in CONTEXT_COLS:
        if c not in x.columns:
            x[c] = 0

    bool_like = [
        "is_promoter", "is_UTR5", "is_UTR3", "is_splice_near",
        "promoter_core_flag", "promoter_proximal_flag", "promoter_distal_flag",
        "atac_overlap_any", "h3k27ac_overlap_any",
        "active_regulatory_flag", "active_enhancer_like_flag",
        "promoter_with_activity",
    ]
    for c in bool_like:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0).astype("float32")

    x["motif_candidate"] = choose_candidates(x, args.scope)

    x.to_parquet(out, index=False)

    manifest = {
        "in_root": str(in_root),
        "out": str(out),
        "total_scanned_rows": int(total),
        "n_unique_snps": int(len(x)),
        "scope": args.scope,
        "n_candidates": int(x["motif_candidate"].sum()),
        "candidate_rate": float(x["motif_candidate"].mean()),
        "chrom_counts": x["Chr"].value_counts().sort_index().to_dict(),
    }
    with open(str(out) + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("[DONE]", out, x.shape, flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
