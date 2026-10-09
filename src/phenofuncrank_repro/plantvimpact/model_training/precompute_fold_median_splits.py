#!/usr/bin/env python3
"""Build selected-model split parquets using train-fold median imputation."""
from __future__ import annotations
from phenofuncrank_repro.common.paths import project_root, package_dir

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_ROOT = package_dir('plantvimpact')
from phenofuncrank_repro.plantvimpact.common.config_utils import load_json, parse_folds
from phenofuncrank_repro.plantvimpact.common.feature_policy import assert_predictors_allowed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--folds", required=True)
    args = parser.parse_args(argv)
    descriptor = load_json(args.features)
    features = descriptor.get("features") or (descriptor.get("feature_columns") or {}).get("features")
    if not isinstance(features, list):
        raise ValueError(f"no features in {args.features}")
    exceptions = descriptor.get("allowed_legacy_exceptions") or []
    assert_predictors_allowed(features, allowed_exceptions=exceptions, context=str(args.features))

    audit = []
    for fold in parse_folds(args.folds):
        fold_input = args.input_root / f"fold{fold:02d}"
        train = pd.read_parquet(fold_input / "train.parquet", columns=features)
        medians = {}
        for feature in features:
            values = pd.to_numeric(train[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
            median = values.median(skipna=True)
            fill = 0.0 if pd.isna(median) else float(median)
            medians[feature] = fill
            audit.append({
                "fold": fold, "feature": feature, "train_median": fill,
                "fill_value": fill, "policy": "fold_train_median",
            })
        for split in ("train", "val", "test"):
            source = fold_input / f"{split}.parquet"
            frame = pd.read_parquet(source)
            for feature in features:
                frame[feature] = (
                    pd.to_numeric(frame[feature], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(medians[feature])
                    .astype("float32")
                )
            destination = args.output_root / f"fold{fold:02d}" / f"{split}.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(destination, index=False, compression="zstd")
    table = pd.DataFrame(audit)
    args.output_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_root / "train_fold_median_imputation.tsv", sep="\t", index=False)
    (args.output_root / "imputation_manifest.json").write_text(json.dumps({
        "input_root": str(args.input_root), "output_root": str(args.output_root),
        "features": features, "policy": "fold_train_median_from_train_split_only",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
