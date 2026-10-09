#!/usr/bin/env python3
"Run the local weighted sampled O/E and cotton exact validation workflows."
from __future__ import annotations
from phenofuncrank_repro.common.paths import project_root, package_dir

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds

SCRIPTS_ROOT = package_dir('plantvimpact')
REPO_ROOT = project_root(__file__)

from phenofuncrank_repro.plantvimpact.common.config_utils import configured_models, dotted_get, load_yaml, model_spec, resolve_mapping
from phenofuncrank_repro.plantvimpact.scoring_oe.prepare_oe_universe import run as prepare_oe_universe


def log(message: str) -> None:
    print(f"[run_oe_validation] {message}", file=sys.stderr, flush=True)


def score_specs(spec: dict[str, Any]) -> list[dict[str, str]]:
    selected = spec.get("selected_score") or {}
    scores = [{
        "name": "selected",
        "column": str(selected["column"]),
        "direction": str(selected.get("direction", "high")),
    }]
    scores.extend(
        {
            "name": str(raw["name"]),
            "column": str(raw["column"]),
            "direction": str(raw.get("direction", "high")),
        }
        for raw in spec.get("raw_scores", [])
    )
    names = [item["name"] for item in scores]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate O/E score names: {names}")
    for item in scores:
        if item["direction"] not in {"high", "low"}:
            raise ValueError(f"invalid score direction: {item}")
    return scores


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def require_file(path: str, label: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def require_table(path: str, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def require_columns(path: str, columns: list[str], label: str) -> None:
    available = set(ds.dataset(path, format="parquet").schema.names)
    missing = [column for column in dict.fromkeys(columns) if column not in available]
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")


def build_sampled_command(
    weighted_script: str,
    universe_table: str,
    sample_score_table: str,
    out_dir: Path,
    score: dict[str, str],
    oe_inputs: dict[str, Any],
    oe_cfg: dict[str, Any],
    batch_size: int,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        weighted_script,
        "--universe-table",
        universe_table,
        "--sample-score-table",
        sample_score_table,
        "--out-dir",
        str(out_dir),
        "--score-spec",
        f"{score['name']}:{score['column']}:{score['direction']}",
        "--observed-col",
        str(oe_inputs["observed_column"]),
        "--id-col",
        str(oe_inputs["id_column"]),
        "--region-col",
        str(oe_inputs.get("region_column") or ""),
        "--strata-cols",
        str(oe_inputs["strata_columns"]),
        "--top-fracs",
        str(oe_cfg["top_fractions"]),
        "--batch-size",
        str(batch_size),
    ]
    if oe_inputs.get("qc_columns"):
        command += ["--qc-cols", str(oe_inputs["qc_columns"])]
    if oe_inputs.get("sample_weight_column"):
        command += ["--sample-weight-col", str(oe_inputs["sample_weight_column"])]
    if oe_cfg.get("allow_missing_observed"):
        command.append("--allow-missing-observed")
    if not oe_inputs.get("region_column"):
        command.append("--no-within-region")
    return command


def build_exact_command(
    validator: str,
    exhaustive_score_table: str,
    out_dir: Path,
    sampled_root: Path,
    scores: list[dict[str, str]],
    oe_inputs: dict[str, Any],
    oe_cfg: dict[str, Any],
    batch_size: int,
    sample_sizes: str,
    seeds: str,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        validator,
        "--full-score-table",
        exhaustive_score_table,
        "--out-dir",
        str(out_dir),
        "--sampled-root",
        str(sampled_root),
        "--top-fracs",
        str(oe_cfg["top_fractions"]),
        "--observed-col",
        str(oe_inputs["observed_column"]),
        "--id-col",
        str(oe_inputs["id_column"]),
        "--strata-cols",
        str(oe_inputs["strata_columns"]),
        "--region-col",
        str(oe_inputs.get("region_column") or ""),
        "--batch-size",
        str(batch_size),
    ]
    for score in scores:
        command += [
            "--score-spec",
            f"{score['name']}:{score['column']}:{score['direction']}",
        ]
    if not oe_inputs.get("region_column"):
        command.append("--no-within-region")
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local sampled O/E and required cotton exhaustive checks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Resolved paths YAML")
    parser.add_argument("--model-registry", default="configs/plantvimpact/model_registry.yaml")
    parser.add_argument("--species")
    parser.add_argument("--mode", choices=("coding", "noncoding"))
    parser.add_argument("--all-species", action="store_true")
    parser.add_argument("--output", required=True, help="O/E output directory")
    parser.add_argument(
        "--score-root",
        type=Path,
        help=(
            "Use generated ROOT/<species>_<mode>/sample_selected_scores.parquet "
            "and exhaustive_selected_scores.parquet instead of configured legacy score tables."
        ),
    )
    parser.add_argument(
        "--input",
        dest="sample_score_table",
        help="Single-target scored-table override (normally use --score-root).",
    )
    parser.add_argument("--universe-table", help="Single-target universe override")
    parser.add_argument("--exhaustive-score-table", help="Single cotton-target override")
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--sample-sizes", default="1000000,2000000,5000000")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-gates",
        action="store_true",
        help="Do not evaluate promotion/concordance gates after execution",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.all_species and (args.species or args.mode):
        raise SystemExit("--all-species cannot be combined with --species/--mode")
    if not args.all_species and not (args.species and args.mode):
        raise SystemExit("use --all-species or provide both --species and --mode")
    if args.all_species and any(
        (args.sample_score_table, args.universe_table, args.exhaustive_score_table)
    ):
        raise SystemExit("input overrides are only valid for one species/mode")

    raw_paths = load_yaml(args.config)
    paths = resolve_mapping(raw_paths, raw_paths)
    raw_registry = load_yaml(args.model_registry)
    registry = resolve_mapping(raw_registry, paths)
    oe_cfg = dotted_get(registry, "xgboost.oe_validation")
    shared_oe_inputs = paths.get("oe_inputs") or {}
    required_shared_keys = {"observed_column", "id_column"}
    missing_oe = sorted(required_shared_keys - set(shared_oe_inputs))
    if missing_oe:
        raise KeyError(f"missing oe_inputs settings: {missing_oe}")

    weighted_script = str(Path(__file__).with_name("weighted_stratified_oe.py"))
    validator = str(Path(__file__).with_name("exact_oe_validation.py"))
    if not args.dry_run:
        require_file(weighted_script, "local weighted O/E script")
        require_file(validator, "local exact O/E validator")

    targets = configured_models(registry) if args.all_species else [(args.species, args.mode)]
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan: dict[str, Any] = {
        "schema_version": 1,
        "config": str(Path(args.config).resolve()),
        "model_registry": str(Path(args.model_registry).resolve()),
        "output_dir": str(output_dir),
        "dry_run": bool(args.dry_run),
        "targets": [],
    }
    commands: list[list[str]] = []

    for species, mode in targets:
        spec = model_spec(registry, species, mode)
        oe_inputs = {**shared_oe_inputs, **(spec.get("oe_inputs") or {})}
        if not oe_inputs.get("strata_columns"):
            raise KeyError(f"missing registered O/E strata for {species}/{mode}")
        target_oe_cfg = dict(oe_cfg)
        if (spec.get("oe") or {}).get("top_fractions"):
            target_oe_cfg["top_fractions"] = spec["oe"]["top_fractions"]
        inputs = dotted_get(paths, f"model_inputs.{species}.{mode}") or {}
        universe_source = str((spec.get("oe") or {}).get("universe_source", "configured"))
        universe_table = str(args.universe_table or inputs.get("universe_table", ""))
        if universe_source == "configured" and not universe_table:
            raise KeyError(f"missing universe_table for {species}/{mode}")
        if universe_source not in {"configured", "sample_score_table"}:
            raise ValueError(f"unsupported universe_source for {species}/{mode}: {universe_source}")
        target_dir = output_dir / f"{species}_{mode}"
        universe_preparation: dict[str, Any] | None = None
        if str(oe_inputs.get("bin_policy")) == "maize_lookup_edges":
            raw_universe_table = universe_table
            prepared_universe = target_dir / "prepared_universe.parquet"
            universe_table = str(prepared_universe)
            if args.dry_run:
                universe_preparation = {
                    "status": "planned",
                    "input": raw_universe_table,
                    "output": universe_table,
                    "metadata": str(inputs.get("oe_bin_metadata", "")),
                    "method": "streamed Context_Alt lookup with saved edges",
                }
            else:
                require_table(raw_universe_table, f"{species}/{mode} raw universe table")
                universe_preparation = prepare_oe_universe(
                    Path(raw_universe_table),
                    prepared_universe,
                    spec,
                    inputs,
                    args.batch_size,
                )
        generated_sample = (
            args.score_root / f"{species}_{mode}" / "sample_selected_scores.parquet"
            if args.score_root else None
        )
        sample_score_table = str(
            args.sample_score_table
            or generated_sample
            or inputs.get("sample_score_table", "")
        )
        if universe_source == "sample_score_table":
            if args.universe_table:
                raise ValueError(
                    f"{species}/{mode} is locked to universe_source=sample_score_table; "
                    "--universe-table would change the registered experiment"
                )
            universe_table = sample_score_table
            universe_preparation = {
                "status": "registered",
                "input": sample_score_table,
                "output": sample_score_table,
                "method": "reuse selected score table as universe",
            }
        if not universe_table or not sample_score_table:
            raise KeyError(f"missing O/E input tables for {species}/{mode}")
        scores = score_specs(spec)
        if not args.dry_run:
            require_table(universe_table, f"{species}/{mode} universe table")
            require_table(sample_score_table, f"{species}/{mode} sample score table")
            operational = [
                str(oe_inputs["observed_column"]),
                str(oe_inputs["id_column"]),
                *[
                    value.strip()
                    for value in str(oe_inputs["strata_columns"]).split(",")
                    if value.strip()
                ],
            ]
            if oe_inputs.get("region_column"):
                operational.append(str(oe_inputs["region_column"]))
            require_columns(
                universe_table, operational, f"{species}/{mode} universe table"
            )
            require_columns(
                sample_score_table,
                operational + [score["column"] for score in scores],
                f"{species}/{mode} sample score table",
            )
        sampled_outputs: dict[str, str] = {}
        sampled_required = bool((spec.get("oe") or {}).get("sampled_required", True))
        if sampled_required:
            for score in scores:
                score_dir = target_dir / "sampled" / score["name"]
                sampled_outputs[score["name"]] = str(score_dir)
                commands.append(
                    build_sampled_command(
                        weighted_script,
                        universe_table,
                        sample_score_table,
                        score_dir,
                        score,
                        oe_inputs,
                        target_oe_cfg,
                        args.batch_size,
                    )
                )

        exact_output: str | None = None
        exact_required = bool((spec.get("oe") or {}).get("exact_required", False))
        generated_exhaustive = (
            args.score_root / f"{species}_{mode}" / "exhaustive_selected_scores.parquet"
            if args.score_root else None
        )
        exhaustive_score_table = str(
            args.exhaustive_score_table
            or generated_exhaustive
            or inputs.get("exhaustive_score_table", "")
        )
        if exact_required:
            if not exhaustive_score_table:
                raise KeyError(f"missing exhaustive_score_table for {species}/{mode}")
            if not args.dry_run:
                require_table(
                    exhaustive_score_table,
                    f"{species}/{mode} exhaustive score table",
                )
                require_columns(
                    exhaustive_score_table,
                    operational + [score["column"] for score in scores],
                    f"{species}/{mode} exhaustive score table",
                )
            exact_dir = target_dir / "exact_vs_sampled"
            exact_output = str(exact_dir)
            commands.append(
                build_exact_command(
                    validator,
                    exhaustive_score_table,
                    exact_dir,
                    target_dir / "sampled",
                    scores,
                    oe_inputs,
                    target_oe_cfg,
                    args.batch_size,
                    args.sample_sizes,
                    args.seeds,
                )
            )

        plan["targets"].append({
            "species": species,
            "mode": mode,
            "scores": scores,
            "universe_table": universe_table,
            "universe_source": universe_source,
            "universe_preparation": universe_preparation,
            "sample_score_table": sample_score_table,
            "exhaustive_score_table": exhaustive_score_table or None,
            "sampled_outputs": sampled_outputs,
            "exact_output": exact_output,
            "top_fractions": str(target_oe_cfg["top_fractions"]),
        })

    plan["commands"] = commands
    plan_path = output_dir / "oe_validation_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    log(f"wrote {plan_path}")

    for index, command in enumerate(commands, start=1):
        log(f"[{index}/{len(commands)}] {command_text(command)}")
        if not args.dry_run:
            subprocess.run(command, check=True)

    if not args.dry_run and not args.skip_gates:
        gate_report = output_dir / "oe_gate_report.json"
        gate_command = [
            sys.executable,
            str(Path(__file__).with_name("evaluate_oe_gates.py")),
            "--model-registry",
            args.model_registry,
            "--plan",
            str(plan_path),
            "--output",
            str(gate_report),
        ]
        log(command_text(gate_command))
        subprocess.run(gate_command, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
