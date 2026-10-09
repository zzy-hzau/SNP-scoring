#!/usr/bin/env python3
"Prepare feature plans and LOCO feature splits."
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

from phenofuncrank_repro.plantvimpact.common.config_utils import configured_models, dotted_get, folds_cli, load_json, load_yaml, model_spec, resolve_mapping, resolve_template
from phenofuncrank_repro.plantvimpact.common.feature_policy import assert_predictors_allowed


CODING_LM_SCRIPT = Path(__file__).with_name("score_coding_lm.py")
NONCODING_LM_SCRIPT = Path(__file__).with_name("score_noncoding_lm.py")
ATTACH_SCRIPT = Path(__file__).with_name("attach_species_features.py")
SUPPORTED_SPECIES = ("rice", "maize", "tomato", "chickpea", "cotton")


def log(message: str) -> None:
    print(f"[extract_features] {message}", file=sys.stderr, flush=True)


def load_feature_contract(
    feature_path: Path,
    paths: dict[str, Any],
    species: str,
    mode: str,
) -> dict[str, Any]:
    data = load_json(feature_path)
    if not isinstance(data, dict) or not isinstance(data.get("features"), list):
        raise ValueError(f"feature contract must contain a features list: {feature_path}")
    features = [str(v) for v in data["features"]]
    exceptions = [str(v) for v in data.get("allowed_legacy_exceptions", [])]
    assert_predictors_allowed(
        features,
        allowed_exceptions=exceptions,
        context=f"{species}/{mode} {feature_path}",
    )
    if len(features) != len(set(features)):
        raise ValueError(f"duplicate predictors in {feature_path}")
    data = dict(data)
    data["features"] = features
    data["allowed_legacy_exceptions"] = exceptions
    if data.get("source_artifact"):
        data["source_artifact"] = resolve_template(data["source_artifact"], paths)
    return data


def _script(root: str, relative: str) -> str:
    return str(Path(root) / relative)


def _step(name: str, command: list[str], produces: list[str]) -> dict[str, Any]:
    return {"step": name, "command": command, "produces": produces}


def build_plan(
    paths: dict[str, Any],
    registry: dict[str, Any],
    species: str,
    mode: str,
    target_path: str,
    model_out: Path,
    feature_contract: dict[str, Any],
    config_path: str,
    reuse_lm: bool,
    reuse_safe: bool,
    reuse_extra: bool,
    label_root_override: str | None = None,
) -> list[dict[str, Any]]:
    spec = resolve_mapping(model_spec(registry, species, mode), paths)
    sources_raw = dotted_get(paths, f"feature_sources.{species}.{mode}")
    if not isinstance(sources_raw, dict):
        raise KeyError(f"missing feature_sources.{species}.{mode} in paths config")
    sources = resolve_mapping(sources_raw, paths)
    features = feature_contract["features"]
    model_out.mkdir(parents=True, exist_ok=True)
    lm_out = model_out / "lm_scores"
    safe_out = model_out / f"{species}_prot_safe.parquet"
    attached_raw_out = model_out / "attached_raw"
    attached_out = model_out / "loco_splits"
    plan: list[dict[str, Any]] = []

    lm_source = str(sources.get("lm_score") or "")
    if not reuse_lm:
        if mode == "coding":
            command = [
                sys.executable,
                str(CODING_LM_SCRIPT),
                "--targets", target_path,
                "--genome", resolve_template(dotted_get(paths, f"reference_data.genome_fasta.{species}"), paths),
                "--gff", resolve_template(dotted_get(paths, f"reference_data.gff.{species}"), paths),
                "--outdir", str(lm_out),
                "--dna_model", resolve_template(dotted_get(paths, "lm_models.plantcaduceus"), paths),
                "--esm_model", resolve_template(dotted_get(paths, "lm_models.esm"), paths),
            ]
            produces = ["S_dna", "S_esm"]
        else:
            command = [
                sys.executable,
                str(NONCODING_LM_SCRIPT),
                "--targets", target_path,
                "--genome", resolve_template(dotted_get(paths, f"reference_data.genome_fasta.{species}"), paths),
                "--outdir", str(lm_out),
                "--dna_model", resolve_template(dotted_get(paths, "lm_models.plantcaduceus"), paths),
            ]
            produces = ["S_dna"]
        plan.append(_step(f"{species}/{mode} masked LM", command, produces))
        lm_source = str(lm_out)

    safe_source = str(sources.get("protein_cons") or "")
    needs_protein = any(
        value.startswith("prot_safe_") or value.endswith("_all_ext")
        for value in features
    )

    

    core_safe = {
        "prot_safe_ref_aa_freq",
        "prot_safe_alt_aa_freq",
        "prot_safe_entropy",
        "prot_safe_delta_log_odds",
        "prot_safe_conservation_penalty",
        "prot_safe_abs_delta_log_odds",
        "prot_safe_alt_absent_flag",
    }
    needed_core_safe = {value for value in features if value.startswith("prot_safe_")}
    can_build_core_safe = bool(needed_core_safe) and needed_core_safe.issubset(core_safe)
    if needs_protein and can_build_core_safe and not reuse_safe:
        command = [
            sys.executable,
            str(Path(__file__).with_name("build_safe_features.py")),
            "--config", config_path,
            "--candidates", target_path,
            "--species", species,
            "--output", str(safe_out),
            "--pseudocount", "0.001",
            "--aa-pos-offset", "1",
        ]
        plan.append(_step(f"{species}/{mode} protein SAFE", command, [v for v in features if v.startswith("prot_safe_")]))
        safe_source = str(safe_out)

    extra_sources = [str(value) for value in (sources.get("extra_features") or []) if value]
    if species == "rice" and mode == "noncoding" and not reuse_extra:
        checkpoint = dotted_get(paths, "lm_models.predmoter")
        if not checkpoint:
            raise KeyError("lm_models.predmoter is required for rice/noncoding")
        predmoter_out = model_out / "predmoter"
        plan.append(_step(
            "rice/noncoding local PredMotER scoring",
            [
                sys.executable,
                str(Path(__file__).with_name("score_rice_predmoter.py")),
                "--targets", target_path,
                "--genome", resolve_template(dotted_get(paths, "reference_data.genome_fasta.rice"), paths),
                "--checkpoint", resolve_template(checkpoint, paths),
                "--outdir", str(predmoter_out),
            ],
            [str(predmoter_out)],
        ))
        extra_sources = [str(predmoter_out)]

    attach_command = [
        sys.executable,
        str(ATTACH_SCRIPT),
        "--species", species,
        "--kind", mode,
        "--label-root", str(label_root_override or sources["label_root"]),
        "--out-root", str(attached_raw_out),
        "--folds", folds_cli(spec["folds"]),
        "--require-nonnegative-sdna",
    ]
    optional = {
        "--lm-score": lm_source,
        "--protein-cons": safe_source,
        "--paralog": sources.get("paralog"),
        "--gene-map": sources.get("gene_map"),
        "--gene-map-root": sources.get("gene_map_root"),
        "--pcsd26": sources.get("pcsd26"),
    }
    for flag, value in optional.items():
        if value:
            attach_command.extend([flag, str(value)])
    if bool(sources.get("fill_unmatched_redundancy", False)):
        attach_command.append("--fill-unmatched-redundancy")
    dna_sources = sources.get("dna_cons") or []
    if isinstance(dna_sources, str):
        dna_sources = [dna_sources]
    for value in dna_sources:
        if value:
            attach_command.extend(["--dna-cons", str(value)])
    for value in extra_sources:
        attach_command.extend(["--extra-feature", str(value)])
    plan.append(_step(f"{species}/{mode} source attachment", attach_command, [str(attached_raw_out)]))

    project_command = [
        sys.executable,
        str(Path(__file__).with_name("project_selected_splits.py")),
        "--input-root", str(attached_raw_out),
        "--output-root", str(attached_out),
        "--features", str(model_out / "features.json"),
        "--folds", folds_cli(spec["folds"]),
        "--manifest", str(model_out / "feature_projection_manifest.json"),
    ]
    plan.append(_step(
        f"{species}/{mode} exact selected-feature projection",
        project_command,
        [str(attached_out), str(model_out / "feature_projection_manifest.json")],
    ))

    validate_command = [
        sys.executable,
        str(Path(__file__).with_name("validate_feature_schema.py")),
        "--split-root", str(attached_out),
        "--features", str(model_out / "features.json"),
        "--folds", folds_cli(spec["folds"]),
        "--output", str(model_out / "feature_schema_audit.json"),
    ]
    plan.append(_step(f"{species}/{mode} selected-schema audit", validate_command, [str(model_out / "feature_schema_audit.json")]))
    return plan


def contract_output(
    contract: dict[str, Any],
    species: str,
    mode: str,
    plan: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        **contract,
        "module": "M1",
        "species": species,
        "mode": mode,
        "id_columns": ["species", "kind", "snp_id", "Chr", "Pos", "Ref", "Alt"],
        "feature_columns": {"features": contract["features"]},
        "safe_features": {
            "pseudocount": 0.001,
            "frequency_pseudocount": False,
            "entropy_log_base": 2,
            "delta_log_base": "natural",
            "penalty_is_inverse_log_ratio": True,
        },
        "command_plan": plan,
    }


def execute_plan(plan: list[dict[str, Any]], dry_run: bool) -> int:
    for entry in plan:
        command = [str(v) for v in entry["command"]]
        log(f"{entry['step']}: {' '.join(command)}")
        if dry_run:
            continue
        proc = subprocess.run(command)
        if proc.returncode:
            return proc.returncode
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-registry", default="configs/plantvimpact/model_registry.yaml")
    parser.add_argument("--species", choices=SUPPORTED_SPECIES)
    parser.add_argument("--mode", choices=("coding", "noncoding"))
    parser.add_argument("--all-species", action="store_true", help="Run all ten registered species/region models.")
    parser.add_argument("--output", required=True, help="Output root for one or all model feature plans/splits.")
    parser.add_argument("--targets", default=None, help="Override candidate table for a single model.")
    parser.add_argument("--features", default=None, help="Override selected feature contract for a single model.")
    parser.add_argument(
        "--label-root-base",
        default=None,
        help=(
            "Consume labels from BASE/<species>/<mode> instead of the "
            "feature_sources label roots; used by the from-scratch workflow."
        ),
    )
    parser.add_argument("--reuse-lm", action="store_true", help="Use feature_sources.*.lm_score instead of recomputing LM scores.")
    parser.add_argument("--reuse-safe", action="store_true", help="Use feature_sources.*.protein_cons instead of rebuilding core SAFE features.")
    parser.add_argument("--reuse-extra", action="store_true", help="Use configured extra-feature tables instead of rebuilding local rice PredMotER features.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = load_yaml(args.config)
    registry = load_yaml(args.model_registry)
    targets = configured_models(registry) if args.all_species else [(args.species, args.mode)]
    if not args.all_species and (not args.species or not args.mode):
        raise SystemExit("provide --all-species or both --species and --mode")
    if args.all_species and (args.targets or args.features):
        raise SystemExit("--targets/--features overrides are only valid for one model")

    output_root = Path(args.output)
    rc = 0
    for species, mode in targets:
        spec = model_spec(registry, species, mode)
        feature_path = Path(args.features or spec["features_file"])
        if not feature_path.is_absolute():
            feature_path = REPO_ROOT / feature_path
        contract = load_feature_contract(feature_path, paths, species, mode)
        model_out = output_root / f"{species}_{mode}" if args.all_species else output_root

        

        
        target_key = "coding_pre_lm" if mode == "coding" else "noncoding_pre_lm"
        configured_target = dotted_get(paths, f"label_inputs.{species}.{target_key}")
        if not args.targets and not configured_target:
            raise KeyError(f"missing label_inputs.{species}.{target_key}")
        target_path = args.targets or resolve_template(configured_target, paths)
        label_root_override = (
            str(Path(args.label_root_base) / species / mode)
            if args.label_root_base
            else None
        )
        plan = build_plan(
            paths, registry, species, mode, str(target_path), model_out,
            contract, args.config, args.reuse_lm, args.reuse_safe, args.reuse_extra,
            label_root_override,
        )
        descriptor = contract_output(contract, species, mode, plan)
        features_out = model_out / "features.json"
        features_out.parent.mkdir(parents=True, exist_ok=True)
        features_out.write_text(json.dumps(descriptor, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.dry_run:
            print(f"DRY-RUN plan for {species}/{mode}")
            print(json.dumps(descriptor, indent=2, ensure_ascii=False))
        rc = rc or execute_plan(plan, args.dry_run)
        if rc:
            break
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
