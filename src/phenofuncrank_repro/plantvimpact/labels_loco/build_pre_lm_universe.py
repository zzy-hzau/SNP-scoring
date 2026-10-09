#!/usr/bin/env python3
"Build coding and noncoding candidate SNV tables."
from __future__ import annotations
from phenofuncrank_repro.common.paths import project_root, package_dir

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPTS_ROOT = package_dir('plantvimpact')
REPO_ROOT = project_root(__file__)

from phenofuncrank_repro.plantvimpact.common.config_utils import dotted_get, load_yaml, resolve_mapping


SPECIES = ("rice", "maize", "tomato", "chickpea", "cotton")
BUILDERS = {
    "rice": "diploid_universe.py",
    "maize": "diploid_universe.py",
    "tomato": "diploid_universe.py",
    "chickpea": "diploid_universe.py",
    "cotton": "cotton_universe.py",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Resolved paths YAML")
    parser.add_argument("--species", choices=SPECIES)
    parser.add_argument("--all-species", action="store_true")
    parser.add_argument("--mode", choices=("coding", "noncoding", "both"), default="both")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chrom-regex")
    parser.add_argument("--max-rows-per-mode", type=int)
    parser.add_argument(
        "--require-callrate-hit",
        action="store_true",
        help="Keep only candidates found in the supplied callability table (useful for small callable smoke tests).",
    )
    parser.add_argument(
        "--intron-selected",
        type=Path,
        help="Optional parquet whitelist of Chr/Pos sites used for intron candidates.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def targets(args: argparse.Namespace) -> list[str]:
    if args.all_species:
        if args.species:
            raise ValueError("--all-species cannot be combined with --species")
        return list(SPECIES)
    if not args.species:
        raise ValueError("provide --species or --all-species")
    return [args.species]


def command_for(
    paths: dict[str, Any],
    species_cfg: dict[str, Any],
    species: str,
    mode: str,
    args: argparse.Namespace,
) -> list[str]:
    cfg = species_cfg["species"][species]
    inputs = dotted_get(paths, f"label_inputs.{species}")
    if not isinstance(inputs, dict):
        raise KeyError(f"missing label_inputs.{species}")
    genome = resolve_mapping(dotted_get(paths, f"reference_data.genome_fasta.{species}"), paths)
    gff = resolve_mapping(dotted_get(paths, f"reference_data.gff.{species}"), paths)
    wild_vcf = resolve_mapping(inputs["wild_vcf"], paths)
    callrate_key = f"{mode}_callrate"
    callrate_value = inputs.get(callrate_key, inputs.get("callrate"))
    if not callrate_value:
        raise KeyError(f"missing label_inputs.{species}.{callrate_key} or callrate")
    callrate = resolve_mapping(callrate_value, paths)
    out_dir = args.output / species
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "prelm" / BUILDERS[species]),
        "--genome", str(genome),
        "--gff", str(gff),
        "--wild-vcf", str(wild_vcf),
        "--callrate", str(callrate),
        "--out-dir", str(out_dir),
        "--mode", mode,

        

        "--n-chr", str(cfg.get("loco_groups", cfg["n_chromosomes"])),
        "--subgenome-mode", "auto" if species == "cotton" else "single",
        "--max-callable-n", str(inputs["max_callable_n"]),
    ]
    if args.chrom_regex:
        command += ["--chrom-regex", args.chrom_regex]
    if args.max_rows_per_mode is not None:
        command += ["--max-rows-per-mode", str(args.max_rows_per_mode)]
    if args.require_callrate_hit:
        command.append("--require-callrate-hit")
    if args.intron_selected:
        command += ["--intron-selected", str(args.intron_selected)]
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_mapping(load_yaml(args.config), load_yaml(args.config))
    species_cfg = load_yaml(REPO_ROOT / "configs/plantvimpact/species.yaml")
    command_specs = []
    for species in targets(args):
        modes = ("coding", "noncoding") if args.mode == "both" else (args.mode,)
        for mode in modes:
            command_specs.append((species, mode, command_for(paths, species_cfg, species, mode, args)))
    commands = [command for _, _, command in command_specs]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "prelm_build_plan.json").write_text(
        json.dumps({"commands": commands}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for species, mode, command in command_specs:
        print("[preLM]", " ".join(str(v) for v in command), file=sys.stderr, flush=True)
        if not args.dry_run:
            result = subprocess.run(command)
            if result.returncode:
                return result.returncode
            manifest = args.output / species / "preLM_universe_manifest.json"
            if args.mode == "both" and manifest.is_file():
                shutil.copyfile(
                    manifest,
                    args.output / species / f"preLM_universe_manifest.{mode}.json",
                )
    if not args.dry_run and args.mode == "both":
        for species in targets(args):
            species_dir = args.output / species
            combined = {"mode": "both", "species": species, "outputs": {}}
            found = False
            for mode in ("coding", "noncoding"):
                path = species_dir / f"preLM_universe_manifest.{mode}.json"
                if not path.is_file():
                    continue
                found = True
                item = json.loads(path.read_text(encoding="utf-8"))
                combined["genome"] = item.get("genome", combined.get("genome"))
                combined["gff"] = item.get("gff", combined.get("gff"))
                combined["wild_vcf"] = item.get("wild_vcf", combined.get("wild_vcf"))
                combined["callrate_by_mode"] = combined.get("callrate_by_mode", {})
                combined["callrate_by_mode"][mode] = item.get("callrate")
                combined["outputs"].update(item.get("outputs", {}))
            if found:
                (species_dir / "preLM_universe_manifest.json").write_text(
                    json.dumps(combined, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
