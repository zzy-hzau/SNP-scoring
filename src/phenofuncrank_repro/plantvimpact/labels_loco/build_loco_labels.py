#!/usr/bin/env python3
"Build LOCO-Ne proxy labels for all selected models."
from __future__ import annotations
from phenofuncrank_repro.common.paths import project_root, package_dir

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPTS_ROOT = package_dir('plantvimpact')
from phenofuncrank_repro.plantvimpact.common.config_utils import dotted_get, load_yaml, parse_folds, resolve_mapping


LABEL_DIR = Path(__file__).resolve().parent
COMMON_PMUT = LABEL_DIR / "pmut.py"
COMMON_CODING = LABEL_DIR.parent / "native_upstream/coding_labels.py"
COMMON_NONCODING = LABEL_DIR / "strict_noncoding_labels.py"


COTTON_PMUT_WRAPPER = LABEL_DIR / "cotton_pmut_cli.py"
COTTON_PMUT_CORE = LABEL_DIR / "cotton_calc_pmut.py"
COTTON_CODING = COMMON_CODING
COTTON_NONCODING = LABEL_DIR / "cotton_noncoding_labels.py"


def log(message: str) -> None:
    print(f"[build_loco_labels] {message}", file=sys.stderr, flush=True)


def targets(species_cfg: dict[str, Any], args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    if args.all_species:
        return [(name, ["coding", "noncoding"]) for name in species_cfg["species"]]
    if not args.species or not args.mode:
        raise ValueError("use --all-species or provide both --species and --mode")
    if args.species not in species_cfg["species"]:
        raise KeyError(f"unknown species: {args.species}")
    return [(args.species, [args.mode])]


def common_label_options(n_chr: int, folds: str, max_callable_n: int) -> list[str]:
    return [
        "--n-chr", str(n_chr),
        "--folds", folds,
        "--max-callable-n", str(max_callable_n),
        "--fallback-mu", "4.56e-9",
        "--pmut-pseudo-count", "0.5",
        "--min-callable-n", "5",
        "--opportunity-min-quantile", "0.20",
        "--opportunity-bins", "10",
        "--callable-n-bins", "10",
        "--label1-per-label0", "1.0",
        "--max-label0-per-group", "20000",
        "--min-group-label0", "3",
        "--min-group-label1", "3",
        "--seed", "17",
        "--ne-calibration-iter", "80",
        "--ne-calibration-max", "1e12",
    ]


def build_species_plan(
    paths: dict[str, Any],
    species_cfg: dict[str, Any],
    species: str,
    modes: list[str],
    output_root: Path,
    reuse_rate_cache: bool,
    rate_cache_override: str | None,
) -> list[dict[str, Any]]:
    scientific = species_cfg["species"][species]
    inputs = dotted_get(paths, f"label_inputs.{species}")
    if not isinstance(inputs, dict):
        raise KeyError(f"paths.label_inputs.{species} is missing")
    folds = str(scientific["loco_folds"])
    n_chr = max(parse_folds(folds))
    species_out = output_root / species
    cache = Path(
        rate_cache_override
        or inputs.get("rate_cache")
        or species_out / "rate_tables_cache.pkl"
    )
    global_json = species_out / "global_rate_table.json"
    fasta = str(dotted_get(paths, f"reference_data.genome_fasta.{species}"))
    vcf = str(inputs["wild_vcf"])
    sites_4d = str(inputs["sites_4d"])
    max_callable_n = int(inputs["max_callable_n"])
    plan: list[dict[str, Any]] = []

    if not reuse_rate_cache:
        if species == "cotton":
            command = [
                sys.executable, str(COTTON_PMUT_WRAPPER),
                "--calc-script", str(COTTON_PMUT_CORE),
                "--fasta", fasta, "--vcf", vcf, "--sites-4d", sites_4d,
                "--n-chr", str(n_chr), "--out", str(cache),
                "--global-json", str(global_json),
            ]
        else:
            command = [
                sys.executable, str(COMMON_PMUT),
                "--fasta", fasta, "--vcf", vcf, "--sites-4d", sites_4d,
                "--n-chr", str(n_chr), "--group-label", str(inputs["group_label"]),
                "--out", str(cache), "--global-json", str(global_json),
            ]
        plan.append({"step": f"{species}/P_mut", "command": command, "produces": [str(cache)]})

    for mode in modes:
        label_out = species_out / mode
        shared = common_label_options(n_chr, folds, max_callable_n)
        if mode == "coding":
            script = COTTON_CODING if species == "cotton" else COMMON_CODING
            command = [
                sys.executable, str(script),
                "--input", str(inputs["coding_pre_lm"]),
                "--out-dir", str(label_out),
                "--rate-cache", str(cache),
                *shared,
                "--mu-base", "4.56e-9",
                "--batch-size", "300000",
                "--sample-rows-for-bins", "50000000",
                "--edge-sample-fraction", "1.0",
            ]
        else:
            script = COTTON_NONCODING if species == "cotton" else COMMON_NONCODING
            noncoding_callrate = inputs.get("noncoding_callrate", inputs.get("callrate"))
            if not noncoding_callrate:
                raise KeyError(f"paths.label_inputs.{species}.noncoding_callrate or callrate is missing")
            command = [
                sys.executable, str(script),
                "--input", str(inputs["noncoding_pre_lm"]),
                "--callrate", str(noncoding_callrate),
                "--out-dir", str(label_out),
                "--fasta", fasta,
                "--wild-vcf", vcf,
                "--sites-4d", sites_4d,
                "--rate-cache", str(cache),
                "--subgenome-mode", "auto" if species == "cotton" else "single",
                "--ne-mode", "train_loco",
                *shared,
                "--batch-size", "100000" if species == "cotton" else "1000000",
                "--sample-rows-for-bins", "2000000" if species == "cotton" else "50000000",
                "--edge-sample-fraction", "0.03",
                "--progress-every", "10",
                "--trust-input-callrate",
            ]
        plan.append({
            "step": f"{species}/{mode} strict LOCO-Ne labels",
            "command": command,
            "produces": [str(label_out / f"fold{fold:02d}") for fold in parse_folds(folds)],
        })
    return plan


def execute(plan: list[dict[str, Any]], dry_run: bool) -> int:
    for item in plan:
        command = [str(v) for v in item["command"]]
        log(f"{item['step']}: {' '.join(command)}")
        if not dry_run:
            missing_scripts = [v for i, v in enumerate(command) if i == 1 and not Path(v).is_file()]
            if missing_scripts:
                raise FileNotFoundError(f"local label script not found: {missing_scripts[0]}")
            result = subprocess.run(command)
            if result.returncode:
                return result.returncode
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--species-config", default="configs/plantvimpact/species.yaml")
    parser.add_argument("--species")
    parser.add_argument("--mode", choices=("coding", "noncoding"))
    parser.add_argument("--all-species", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rate-cache", help="Override cache path for a single species.")
    parser.add_argument("--reuse-rate-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_paths = load_yaml(args.config)
    paths = resolve_mapping(raw_paths, raw_paths)
    species_cfg = load_yaml(args.species_config)
    selected = targets(species_cfg, args)
    if args.rate_cache and len(selected) != 1:
        raise ValueError("--rate-cache is only valid for one species")
    plan: list[dict[str, Any]] = []
    for species, modes in selected:
        plan.extend(build_species_plan(
            paths, species_cfg, species, modes, args.output,
            args.reuse_rate_cache, args.rate_cache,
        ))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "module": "M2",
        "logic": "strict_loco_train_only_ne_local_source",
        "negative_sampling": {"hash": "CRC32", "seed": 17},
        "ne_calibration": {"iterations": 80, "maximum": 1e12},
        "plan": plan,
    }
    (args.output / "label_build_plan.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return execute(plan, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
