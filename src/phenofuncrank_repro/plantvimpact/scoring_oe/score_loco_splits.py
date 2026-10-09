#!/usr/bin/env python3
"Score every LOCO split with the local fold-aware scorer."
from __future__ import annotations

import argparse
import json
from pathlib import Path

from phenofuncrank_repro.plantvimpact.scoring_oe.score_loco_table import parse_folds, run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", required=True, type=Path)
    ap.add_argument("--model-root", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--fold-col", default="chr_num")
    ap.add_argument("--probability-col", default="strict_v3_prob")
    ap.add_argument("--margin-col", default="strict_v3_margin")
    ap.add_argument("--feature-contract", type=Path)
    args = ap.parse_args()

    reports = []
    all_folds = parse_folds(args.folds)
    for fold in all_folds:
        for split in [x.strip() for x in args.splits.split(",") if x.strip()]:
            input_path = args.input_root / f"fold{fold:02d}" / f"{split}.parquet"
            if not input_path.is_file():
                raise FileNotFoundError(input_path)
            output_path = args.output_root / f"fold{fold:02d}" / f"{split}.parquet"

            

            
            reports.append(run(
                input_path,
                args.model_root,
                output_path,
                all_folds,
                args.fold_col,
                args.probability_col,
                args.margin_col,
                args.feature_contract,
            ))
            print(f"[SCORED] fold{fold:02d} {split} rows={reports[-1]['rows']:,}", flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_root": str(args.input_root.resolve()),
        "model_root": str(args.model_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "folds": parse_folds(args.folds),
        "splits": [x.strip() for x in args.splits.split(",") if x.strip()],
        "reports": reports,
    }
    (args.output_root / "score_splits_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
