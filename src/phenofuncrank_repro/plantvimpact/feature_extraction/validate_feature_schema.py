#!/usr/bin/env python3
"""Verify every LOCO split contains exactly the selected predictor contract."""
from __future__ import annotations
from phenofuncrank_repro.common.paths import project_root, package_dir

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

SCRIPTS_ROOT = package_dir('plantvimpact')
from phenofuncrank_repro.plantvimpact.common.config_utils import load_json, parse_folds
from phenofuncrank_repro.plantvimpact.common.feature_policy import assert_predictors_allowed


def feature_contract(path: Path) -> tuple[list[str], list[str]]:
    data = load_json(path)
    features = data.get("features")
    if not isinstance(features, list):
        features = (data.get("feature_columns") or {}).get("features")
    if not isinstance(features, list):
        raise ValueError(f"no features list in {path}")
    exceptions = data.get("allowed_legacy_exceptions") or []
    assert_predictors_allowed(features, allowed_exceptions=exceptions, context=str(path))
    return [str(v) for v in features], [str(v) for v in exceptions]


def run(split_root: Path, features_path: Path, folds: list[int]) -> dict:
    features, exceptions = feature_contract(features_path)
    rows = []
    missing_files = []
    for fold in folds:
        for split in ("train", "val", "test"):
            path = split_root / f"fold{fold:02d}" / f"{split}.parquet"
            if not path.is_file():
                missing_files.append(str(path))
                continue
            columns = pq.ParquetFile(path).schema.names
            missing = [name for name in features if name not in columns]
            rows.append({
                "fold": fold,
                "split": split,
                "path": str(path),
                "n_columns": len(columns),
                "n_selected_features": len(features),
                "missing_features": missing,
                "ok": not missing,
            })
    ok = bool(rows) and not missing_files and all(row["ok"] for row in rows)
    return {
        "ok": ok,
        "split_root": str(split_root),
        "features_path": str(features_path),
        "features": features,
        "allowed_legacy_exceptions": exceptions,
        "missing_files": missing_files,
        "splits": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run(args.split_root, args.features, parse_folds(args.folds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not report["ok"]:
        print(json.dumps(report, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
