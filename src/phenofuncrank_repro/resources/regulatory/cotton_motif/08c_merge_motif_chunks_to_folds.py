#!/usr/bin/env python3


import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np


MOTIF_FEATURES = [
    "motif_hit_ref_count",
    "motif_hit_alt_count",
    "motif_loss_count",
    "motif_gain_count",
    "motif_loss_any",
    "motif_gain_any",
    "motif_disruption_any",
    "max_abs_delta_pwm",
    "max_loss_delta_pwm",
    "max_gain_delta_pwm",

    "MYB_motif_disruption",
    "bHLH_motif_disruption",
    "NAC_motif_disruption",
    "HDZIP_motif_disruption",
    "WRKY_motif_disruption",
    "AP2ERF_motif_disruption",
    "bZIP_motif_disruption",
    "TCP_motif_disruption",
    "MADS_motif_disruption",
    "DOF_motif_disruption",
    "ARF_motif_disruption",
    "GATA_motif_disruption",
    "C2H2_ZF_motif_disruption",

    "motif_disruption_in_ATAC",
    "motif_loss_in_ATAC",
    "motif_gain_in_ATAC",
    "motif_disruption_in_H3K27ac",
    "motif_loss_in_H3K27ac",
    "motif_gain_in_H3K27ac",
    "motif_disruption_in_active_enhancer_like",
    "motif_loss_in_active_enhancer_like",
    "motif_disruption_in_active_promoter",
    "motif_loss_in_active_promoter",

    "max_abs_delta_pwm_in_ATAC",
    "max_loss_delta_pwm_in_ATAC",
    "max_gain_delta_pwm_in_ATAC",
    "max_abs_delta_pwm_in_H3K27ac",
    "max_loss_delta_pwm_in_H3K27ac",
    "max_gain_delta_pwm_in_H3K27ac",
    "max_abs_delta_pwm_in_active_enhancer_like",
    "max_loss_delta_pwm_in_active_enhancer_like",
    "max_abs_delta_pwm_in_active_promoter",
    "max_loss_delta_pwm_in_active_promoter",

    "motif_scanned",
    "motif_ref_mismatch",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True)
    ap.add_argument("--chunks-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--feature-table-out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    chunks_dir = Path(args.chunks_dir)
    out_root = Path(args.out_root)
    feature_table_out = Path(args.feature_table_out)

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    feature_table_out.parent.mkdir(parents=True, exist_ok=True)

    chunk_paths = sorted(chunks_dir.glob("motif_*.parquet"))
    if not chunk_paths:
        raise RuntimeError(f"No motif chunks found in {chunks_dir}")

    bad = []
    for p in chunk_paths:
        done = p.with_suffix(".done")
        if not done.exists():
            bad.append(str(p))
    if bad:
        raise RuntimeError("Some chunks missing .done, first 10:\n" + "\n".join(bad[:10]))

    print(f"[merge chunks] n={len(chunk_paths):,}", flush=True)

    dfs = []
    for i, p in enumerate(chunk_paths, 1):
        df = pd.read_parquet(p)
        dfs.append(df)
        if i % 100 == 0:
            print(f"[read chunks] {i:,}/{len(chunk_paths):,}", flush=True)

    motif = pd.concat(dfs, ignore_index=True)
    motif = motif.drop_duplicates("snp_id")

    for c in MOTIF_FEATURES:
        if c not in motif.columns:
            motif[c] = 0.0
        motif[c] = pd.to_numeric(motif[c], errors="coerce").fillna(0.0).astype("float32")

    motif.to_parquet(feature_table_out, index=False)
    print("[feature table]", feature_table_out, motif.shape, flush=True)

    manifest = {
        "chunks_dir": str(chunks_dir),
        "n_chunks": len(chunk_paths),
        "n_unique_motif_rows": int(len(motif)),
        "feature_table": str(feature_table_out),
        "feature_means": {c: float(motif[c].mean()) for c in MOTIF_FEATURES},
    }
    with open(str(feature_table_out) + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


    for fd in sorted(in_root.glob("fold*")):
        out_fd = out_root / fd.name
        out_fd.mkdir(parents=True, exist_ok=True)

        for sp in ["train", "val", "test"]:
            p = fd / f"{sp}.parquet"
            if not p.exists():
                continue

            df = pd.read_parquet(p)
            out = df.merge(motif, on="snp_id", how="left")

            for c in MOTIF_FEATURES:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype("float32")

            op = out_fd / f"{sp}.parquet"
            out.to_parquet(op, index=False)

            print(
                f"[merge split] {fd.name} {sp}: {df.shape} -> {out.shape} "
                f"motif_scanned={out['motif_scanned'].mean():.4f} "
                f"motif_disruption={out['motif_disruption_any'].mean():.4f} "
                f"loss_H3K27ac={out['motif_loss_in_H3K27ac'].mean():.4f}",
                flush=True
            )


        for man in ["manifest.json", "manifest.original.json", "manifest.regulatory_original.json"]:
            src = fd / man
            if src.exists():
                (out_fd / man).write_text(src.read_text())

    print("[DONE]", out_root, flush=True)


if __name__ == "__main__":
    main()
