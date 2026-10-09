#!/usr/bin/env python3
"""Build every declared sampled/exhaustive selected-score table from features."""
from __future__ import annotations
from phenofuncrank_repro.common.paths import project_root, package_dir

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

SCRIPTS_ROOT = package_dir('plantvimpact')
REPO_ROOT = project_root(__file__)

from phenofuncrank_repro.plantvimpact.common.config_utils import configured_models, dotted_get, load_yaml, model_spec, resolve_mapping
from phenofuncrank_repro.plantvimpact.scoring_oe.compose_selected_score import run as compose_score
from phenofuncrank_repro.plantvimpact.scoring_oe.prepare_oe_columns import run as prepare_oe_columns
from phenofuncrank_repro.plantvimpact.scoring_oe.score_loco_table import parse_folds, run as score_loco


def log(message: str) -> None:
    print(f"[score_selected_tables] {message}", file=sys.stderr, flush=True)


def choose_targets(registry: dict[str, Any], args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.all_species:
        if args.species or args.mode:
            raise ValueError("--all-species cannot be combined with --species/--mode")
        return configured_models(registry)
    if not (args.species and args.mode):
        raise ValueError("use --all-species or provide --species and --mode")
    return [(args.species, args.mode)]


def selected_output(root: Path, species: str, mode: str, kind: str) -> Path:
    return root / f"{species}_{mode}" / f"{kind}_selected_scores.parquet"


def validate_scored_schema(
    path: Path, spec: dict[str, Any], shared_oe_inputs: dict[str, Any]
) -> list[str]:
    columns = set(pq.ParquetFile(path).schema.names)
    required = [str(spec["selected_score"]["column"])]
    required.extend(str(raw["column"]) for raw in spec.get("raw_scores", []))
    oe_inputs = {**shared_oe_inputs, **(spec.get("oe_inputs") or {})}
    required.extend(
        column.strip()
        for column in str(oe_inputs.get("strata_columns", "")).split(",")
        if column.strip()
    )
    required.extend(
        str(oe_inputs[key])
        for key in ("observed_column", "id_column", "region_column")
        if oe_inputs.get(key)
    )
    required = list(dict.fromkeys(required))
    missing = [column for column in required if column not in columns]
    if missing:
        raise RuntimeError(f"{path} lacks O/E score columns after scoring: {missing}")
    return required


def run_target(
    paths: dict[str, Any],
    registry: dict[str, Any],
    species: str,
    mode: str,
    output_root: Path,
    model_root_base: Path | None,
    dry_run: bool,
) -> dict[str, Any]:
    spec = model_spec(registry, species, mode)
    inputs = dotted_get(paths, f"model_inputs.{species}.{mode}") or {}
    model_root = (
        model_root_base / species / mode
        if model_root_base is not None
        else Path(str(spec["selected_model_root"]))
    )
    if model_root_base is not None and spec.get("trainer_profile") == "cotton_ablation":
        model_root = model_root / str(spec["feature_set"])
    feature_contract = REPO_ROOT / str(spec["features_file"])
    tables = [("sample", inputs.get("sample_feature_table"))]
    if inputs.get("exhaustive_feature_table"):
        tables.append(("exhaustive", inputs["exhaustive_feature_table"]))
    if (spec.get("oe") or {}).get("exact_required") and len(tables) == 1:
        raise KeyError(f"{species}/{mode} requires exhaustive_feature_table")
    if not tables[0][1]:
        raise KeyError(f"missing model_inputs.{species}.{mode}.sample_feature_table")

    target: dict[str, Any] = {
        "species": species,
        "mode": mode,
        "model_root": str(model_root),
        "feature_contract": str(feature_contract),
        "folds": str(spec["folds"]),
        "selected_score": spec["selected_score"],
        "tables": [],
    }
    for kind, configured_input in tables:
        input_path = Path(str(configured_input))
        base_path = output_root / f"{species}_{mode}" / f"{kind}_base_scores.parquet"
        operational_path = (
            output_root / f"{species}_{mode}" / f"{kind}_oe_operational.parquet"
        )
        output_path = selected_output(output_root, species, mode, kind)
        item = {
            "kind": kind,
            "input": str(input_path),
            "base_output": str(base_path),
            "operational_output": str(operational_path),
            "selected_output": str(output_path),
        }
        target["tables"].append(item)
        if dry_run:
            continue
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        if not model_root.is_dir():
            raise FileNotFoundError(model_root)
        log(f"{species}/{mode} {kind}: fold-aware base scoring")
        item["base_manifest"] = score_loco(
            input_path=input_path,
            model_root=model_root,
            output_path=base_path,
            folds=parse_folds(str(spec["folds"])),
            fold_col=str(inputs.get("fold_column", "chr_num")),
            probability_col="strict_v3_prob",
            margin_col="strict_v3_margin",
            feature_contract=feature_contract,
        )
        log(f"{species}/{mode} {kind}: registered O/E operational columns")
        item["operational_manifest"] = prepare_oe_columns(
            base_path,
            operational_path,
            spec,
            inputs,
        )
        log(f"{species}/{mode} {kind}: registered selected-score composition")
        item["composition_manifest"] = compose_score(
            operational_path,
            output_path,
            registry,
            species,
            mode,
            "strict_v3_margin",
            "strict_v3_prob",
            1_000_000,
        )
        output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
            json.dumps(item["composition_manifest"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        item["oe_score_columns"] = validate_scored_schema(
            output_path, spec, paths.get("oe_inputs") or {}
        )
    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-registry", default="configs/plantvimpact/model_registry.yaml")
    parser.add_argument("--species")
    parser.add_argument("--mode", choices=("coding", "noncoding"))
    parser.add_argument("--all-species", action="store_true")
    parser.add_argument(
        "--model-root-base",
        type=Path,
        help="Use ROOT/<species>/<mode>; otherwise use each registered selected_model_root.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_paths = load_yaml(args.config)
    paths = resolve_mapping(raw_paths, raw_paths)
    raw_registry = load_yaml(args.model_registry)
    registry = resolve_mapping(raw_registry, paths)
    targets = choose_targets(registry, args)
    args.output.mkdir(parents=True, exist_ok=True)
    plan: dict[str, Any] = {
        "schema_version": 1,
        "config": str(Path(args.config).resolve()),
        "model_registry": str(Path(args.model_registry).resolve()),
        "model_root_base": str(args.model_root_base.resolve()) if args.model_root_base else None,
        "output_root": str(args.output.resolve()),
        "dry_run": bool(args.dry_run),
        "targets": [],
    }
    for species, mode in targets:
        plan["targets"].append(
            run_target(
                paths,
                registry,
                species,
                mode,
                args.output.resolve(),
                args.model_root_base.resolve() if args.model_root_base else None,
                args.dry_run,
            )
        )
    plan_path = args.output / "scoring_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    gate = {
        "status": "planned" if args.dry_run else "pass",
        "targets": len(plan["targets"]),
        "tables": sum(len(target["tables"]) for target in plan["targets"]),
        "plan": str(plan_path.resolve()),
    }
    gate_path = args.output / "scoring_gate_report.json"
    gate_path.write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
