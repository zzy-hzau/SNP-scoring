#!/usr/bin/env python3
"Post-process the existing rice_coding_SAFE_protein_conservation.parquet to add"
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        required=True,
    )
    ap.add_argument(
        "--out",
        required=True,
    )
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.is_file():
        raise FileNotFoundError(src)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[READ] {src}", flush=True)
    df = pd.read_parquet(src)
    print(f"  rows={len(df):,}  cols={len(df.columns)}", flush=True)


    for required in ("prot_safe_delta_log_odds", "prot_safe_alt_aa_freq"):
        if required not in df.columns:
            raise RuntimeError(f"missing required column: {required}")


    
    delta = df["prot_safe_delta_log_odds"]
    df["prot_safe_abs_delta_log_odds"] = delta.abs()


    

    

    alt_freq = df["prot_safe_alt_aa_freq"]
    flag = np.where(alt_freq.isna(), 0.0, (alt_freq == 0.0).astype(np.float32))
    df["prot_safe_alt_absent_flag"] = flag


    print("\n[NEW COLUMN STATS]")
    print(f"  prot_safe_abs_delta_log_odds:")
    print(df["prot_safe_abs_delta_log_odds"].describe().to_string())
    print(f"\n  prot_safe_alt_absent_flag distribution:")
    print(df["prot_safe_alt_absent_flag"].value_counts(dropna=False).to_string())


    print(f"\n[WRITE] {out}", flush=True)
    df.to_parquet(out, index=False, compression="zstd")
    print(f"  rows={len(df):,}  cols={len(df.columns)}", flush=True)
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
