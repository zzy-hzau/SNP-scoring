#!/usr/bin/env python3
"""Dispatch the ten selected LOCO models to the local trainer."""
from __future__ import annotations
from phenofuncrank_repro.common.paths import project_root, package_dir

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPTS_ROOT = package_dir('plantvimpact')
REPO_ROOT = project_root(__file__)
from phenofuncrank_repro.plantvimpact.common.config_utils import configured_models, load_json, load_yaml, model_spec, resolve_mapping
from phenofuncrank_repro.plantvimpact.common.feature_policy import assert_predictors_allowed


def log(message: str) -> None:
    print(f"[train_xgb_loco] {message}", file=sys.stderr, flush=True)


def feature_contract(path: Path) -> tuple[list[str], list[str]]:
    data = load_json(path)
    features = data.get("features") or (data.get("feature_columns") or {}).get("features")
    if not isinstance(features, list):
        raise ValueError(f"no selected feature list in {path}")
    features = [str(v) for v in features]
    exceptions = [str(v) for v in data.get("allowed_legacy_exceptions", [])]
    assert_predictors_allowed(features, allowed_exceptions=exceptions, context=str(path))
    if len(features) != len(set(features)):
        raise ValueError(f"duplicate predictors in {path}")
    return features, exceptions


def base_parameter_args(parameters: dict[str, Any]) -> list[str]:
    return [
        "--max-depth", str(parameters["max_depth"]),
        "--eta", str(parameters["eta"]),
        "--subsample", str(parameters["subsample"]),
        "--colsample-bytree", str(parameters["colsample_bytree"]),
        "--min-child-weight", str(parameters["min_child_weight"]),
        "--max-bin", str(parameters["max_bin"]),
        "--num-boost-round", str(parameters["num_boost_round"]),
        "--early-stopping-rounds", str(
            parameters["early_stopping_rounds"]
            if parameters.get("early_stopping_rounds") is not None
            else parameters["num_boost_round"]
        ),
        "--seed", str(parameters["seed"]),
    ]


def build_command(
    spec: dict[str, Any],
    species: str,
    mode: str,
    features: list[str],
    split_root: str,
    model_out: Path,
    device: str,
    nthread: int,
) -> list[str]:
    parameters = spec["parameters"]
    command = [
        sys.executable,
        str(Path(__file__).with_name("strict_xgb_retrain.py")),
        "train",
        "--species", species,
        "--model-type", mode,
        "--split-root", split_root,
        "--out-dir", str(model_out),
        "--features", str(REPO_ROOT / spec["features_file"]),
        "--feature-set", str(spec["feature_set"]),
        "--folds", str(spec["folds"]),
        "--device", device,
    ]
    common = base_parameter_args(parameters)
    command += [
        *common,
        "--lambda-reg", str(parameters["lambda"]),
        "--alpha-reg", str(parameters["alpha"]),
        "--nthread", str(parameters.get("nthread", nthread)),
    ]
    return command


def select_targets(registry: dict[str, Any], args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.all_species:
        if args.features or args.split_root:
            raise ValueError("--features/--split-root overrides require a single model")
        return configured_models(registry)
    if not args.species or not args.mode:
        raise ValueError("use --all-species or provide both --species and --mode")
    return [(args.species, args.mode)]


def run_plan(plan: list[dict[str, Any]], dry_run: bool) -> int:
    for item in plan:
        for prepare in item["prepare_commands"]:
            log(f"{item['species']}/{item['mode']} prepare: {' '.join(prepare)}")
            if not dry_run:
                result = subprocess.run(prepare)
                if result.returncode:
                    return result.returncode
        command = item["train_command"]
        log(f"{item['species']}/{item['mode']}: {' '.join(command)}")
        if dry_run:
            continue
        if not Path(command[1]).is_file():
            raise FileNotFoundError(f"local trainer not found: {command[1]}")
        result = subprocess.run(command)
        if result.returncode:
            return result.returncode
        validate = item["validate_command"]
        result = subprocess.run(validate)
        if result.returncode:
            return result.returncode
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-registry", default="configs/plantvimpact/model_registry.yaml")
    parser.add_argument("--species", choices=("rice", "maize", "tomato", "chickpea", "cotton"))
    parser.add_argument("--mode", choices=("coding", "noncoding"))
    parser.add_argument("--all-species", action="store_true")
    parser.add_argument("--features", type=Path, help="Single-model contract override.")
    parser.add_argument("--split-root", help="Single-model projected LOCO split override.")
    parser.add_argument(
        "--feature-root",
        type=Path,
        help=(
            "Consume projected splits from ROOT/<species>_<mode>/loco_splits; "
            "valid for one model or --all-species."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nthread", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_paths = load_yaml(args.config)
    paths = resolve_mapping(raw_paths, raw_paths)
    registry = load_yaml(args.model_registry)
    if args.feature_root and args.split_root:
        raise ValueError("--feature-root and --split-root are mutually exclusive")
    plan = []
    for species, mode in select_targets(registry, args):
        spec = resolve_mapping(model_spec(registry, species, mode), paths)
        feature_path = args.features or REPO_ROOT / spec["features_file"]
        features, exceptions = feature_contract(feature_path)
        generated_split_root = (
            args.feature_root / f"{species}_{mode}" / "loco_splits"
            if args.feature_root
            else None
        )
        split_root = str(args.split_root or generated_split_root or spec["split_root"])
        model_out = args.output / species / mode
        prepare_commands = []
        if spec.get("imputation_policy") == "precomputed_fold_train_median":
            prepared_root = model_out / "prepared_loco_splits"
            prepare_commands.append([
                sys.executable,
                str(Path(__file__).with_name("precompute_fold_median_splits.py")),
                "--input-root", split_root,
                "--output-root", str(prepared_root),
                "--features", str(feature_path),
                "--folds", str(spec["folds"]),
            ])
            split_root = str(prepared_root)
        command = build_command(
            spec, species, mode, features, str(split_root), model_out,
            args.device, args.nthread,
        )
        audit_out = model_out / "selected_feature_contract_audit.json"
        validate = [
            sys.executable,
            str(Path(__file__).with_name("validate_model_contract.py")),
            "--model-root", str(model_out),
            "--features", str(feature_path),
            "--feature-set", str(spec["feature_set"]),
            "--profile", str(spec["trainer_profile"]),
            "--folds", str(spec["folds"]),
            "--output", str(audit_out),
        ]
        plan.append({
            "species": species, "mode": mode, "trainer_profile": spec["trainer_profile"],
            "features": features, "allowed_legacy_exceptions": exceptions,
            "split_root": str(split_root), "model_root": str(model_out),
            "imputation_policy": spec.get("imputation_policy"),
            "prepare_commands": prepare_commands,
            "train_command": command, "validate_command": validate,
        })
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training_plan.json").write_text(
        json.dumps({"module": "M3", "models": plan}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return run_plan(plan, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
