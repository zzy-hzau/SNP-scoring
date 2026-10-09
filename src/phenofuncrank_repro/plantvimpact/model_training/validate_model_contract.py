#!/usr/bin/env python3
"""Verify trained fold artifacts used exactly the checked-in predictor list."""
from __future__ import annotations
from phenofuncrank_repro.common.paths import project_root, package_dir

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_ROOT = package_dir('plantvimpact')
from phenofuncrank_repro.plantvimpact.common.config_utils import load_json, parse_folds
from phenofuncrank_repro.plantvimpact.common.feature_policy import assert_predictors_allowed


def expected_features(path: Path) -> list[str]:
    data = load_json(path)
    values = data.get("features") or (data.get("feature_columns") or {}).get("features")
    if not isinstance(values, list):
        raise ValueError(f"no features in {path}")
    exceptions = data.get("allowed_legacy_exceptions") or []
    assert_predictors_allowed(values, allowed_exceptions=exceptions, context=str(path))
    return [str(v) for v in values]


def audit(
    model_root: Path, features_path: Path, feature_set: str, profile: str, folds: list[int]
) -> dict:
    expected = expected_features(features_path)
    checks = []
    if profile == "cotton_ablation":
        for fold in folds:
            fold_root = model_root / feature_set / "folds" / f"fold{fold:02d}"
            feature_file = fold_root / "features.json"
            model_file = fold_root / "xgb_model.json"
            actual = load_json(feature_file) if feature_file.is_file() else None
            checks.append({
                "fold": fold, "feature_file": str(feature_file), "model_file": str(model_file),
                "actual_features": actual,
                "features_match": actual == expected,
                "model_exists": model_file.is_file(),
            })
    else:
        config_path = model_root / "config.json"
        config = load_json(config_path) if config_path.is_file() else {}
        actual = (config.get("feature_sets") or {}).get(feature_set)
        for fold in folds:
            model_file = model_root / "folds" / f"fold{fold:02d}" / f"xgboost_{feature_set}.json"
            checks.append({
                "fold": fold, "config": str(config_path), "model_file": str(model_file),
                "actual_features": actual,
                "features_match": actual == expected,
                "model_exists": model_file.is_file(),
            })
    ok = bool(checks) and all(v["features_match"] and v["model_exists"] for v in checks)
    return {
        "ok": ok, "model_root": str(model_root), "features_path": str(features_path),
        "feature_set": feature_set, "trainer_profile": profile,
        "expected_features": expected, "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = audit(
        args.model_root, args.features, args.feature_set, args.profile, parse_folds(args.folds)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not report["ok"]:
        print(json.dumps(report, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
