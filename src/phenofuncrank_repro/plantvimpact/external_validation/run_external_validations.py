#!/usr/bin/env python3
"Run maize 3DM and rice GWAS Atlas validation."
from __future__ import annotations
from phenofuncrank_repro.common.paths import project_root, package_dir

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

SCRIPTS_ROOT = package_dir('plantvimpact')

from phenofuncrank_repro.plantvimpact.common.config_utils import dotted_get, load_yaml, resolve_mapping


def log(message: str) -> None:
    print(f"[run_external_validations] {message}", file=sys.stderr, flush=True)


def public_path(path: Path) -> str:
    """Serialize a path without exposing a user's local directory prefix."""
    path = Path(path)
    return f"<local>/{path.name}" if path.is_absolute() else str(path)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix in {".parquet", ".pq"} or path.is_dir():
        return pd.read_parquet(path)
    separator = "\t" if path.name.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")) else ","
    return pd.read_csv(path, sep=separator)


def require_path(path: Path, label: str, *, directory: bool = False) -> None:
    valid = path.is_dir() if directory else path.exists()
    if not valid:
        kind = "directory" if directory else "path"
        raise FileNotFoundError(f"{label} {kind} does not exist: {path}")


def load_canonical(script: Path) -> ModuleType:
    module_spec = importlib.util.spec_from_file_location(
        "plant_cadd_original_strict_external_validations", script
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load validation implementation: {script}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def get_paths(paths: dict[str, Any], model_root_base: Path | None = None) -> dict[str, Path]:
    maize = dotted_get(paths, "external_validation.maize_3dm") or {}
    rice = dotted_get(paths, "external_validation.rice_gwas_atlas") or {}
    canonical_value = str(dotted_get(paths, "codebase.strict_external_validation_script", ""))
    canonical = Path(canonical_value).expanduser()
    if not canonical.is_absolute():
        canonical = (Path(__file__).resolve().parents[2] / canonical).resolve()
    return {
        "canonical_script": canonical,
        "maize_positives": Path(str(maize.get("positives", ""))),
        "maize_redundancy": Path(str(maize.get("redundancy", ""))),
        "maize_legacy_scores": Path(str(maize.get("legacy_scores", ""))),
        "maize_model": (
            model_root_base / "maize" / "coding"
            if model_root_base else Path(str(dotted_get(
                paths, "model_inputs.maize.coding.model_root", ""
            )))
        ),
        "rice_usable_211": Path(str(rice.get("usable_211", ""))),
        "rice_primary_157": Path(str(rice.get("primary_157", ""))),
        "rice_oof_keys": Path(str(rice.get("model_oof_keys", ""))),
        "rice_selected_91": Path(str(rice.get("selected_91", ""))),
        "rice_completed": Path(str(rice.get("completed_scores", ""))),
        "rice_redundancy": Path(str(rice.get("redundancy", ""))),
        "rice_dna_coding": Path(str(rice.get("coding_dna_conservation", ""))),
        "rice_prot_coding": Path(str(rice.get("coding_protein_conservation", ""))),
        "rice_coding_model": (
            model_root_base / "rice" / "coding"
            if model_root_base else Path(str(dotted_get(
                paths, "model_inputs.rice.coding.model_root", ""
            )))
        ),
        "rice_noncoding_model": (
            model_root_base / "rice" / "noncoding"
            if model_root_base else Path(str(dotted_get(
                paths, "model_inputs.rice.noncoding.model_root", ""
            )))
        ),
    }


def inject_canonical_paths(module: ModuleType, configured: dict[str, Path]) -> None:
    assignments = {
        "MAIZE_3MD_PRED": "maize_positives",
        "MAIZE_RED": "maize_redundancy",
        "MAIZE_MODEL": "maize_model",
        "MAIZE_BLEND91_3MD": "maize_legacy_scores",
        "RICE_91": "rice_selected_91",
        "RICE_COMPLETED": "rice_completed",
        "RICE_RED": "rice_redundancy",
        "RICE_DNA_CODING": "rice_dna_coding",
        "RICE_PROT_CODING": "rice_prot_coding",
        "RICE_CODING_MODEL": "rice_coding_model",
        "RICE_NONCODING_MODEL": "rice_noncoding_model",
    }
    for global_name, config_name in assignments.items():
        setattr(module, global_name, configured[config_name])


def audit_rice_selection(configured: dict[str, Path], output_dir: Path) -> dict[str, Any]:
    usable = read_table(configured["rice_usable_211"])
    primary = read_table(configured["rice_primary_157"])
    overlap = read_table(configured["rice_oof_keys"])
    selected = read_table(configured["rice_selected_91"])
    for name, frame in {
        "usable_211": usable,
        "primary_157": primary,
        "model_oof_keys": overlap,
        "selected_91": selected,
    }.items():
        if "variant_key" not in frame.columns:
            raise KeyError(f"{name} has no variant_key column")

    usable_keys = usable["variant_key"].astype(str)
    primary_keys = primary["variant_key"].astype(str)
    overlap_keys = set(overlap["variant_key"].astype(str))
    selected_keys = selected["variant_key"].astype(str)
    primary_overlap = primary_keys.isin(overlap_keys)
    expected_selected = primary_keys.loc[~primary_overlap]

    checks = {
        "usable_rows_211": len(usable) == 211,
        "primary_rows_157": len(primary) == 157,
        "primary_is_subset_of_usable": set(primary_keys).issubset(set(usable_keys)),
        "primary_exact_oof_overlap_rows_66": int(primary_overlap.sum()) == 66,
        "selected_rows_91": len(selected) == 91,
        "selected_exact_oof_overlap_rows_0": int(selected_keys.isin(overlap_keys).sum()) == 0,
        "selected_multiset_equals_primary_minus_oof": (
            Counter(selected_keys.tolist()) == Counter(expected_selected.tolist())
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"rice external-set selection audit failed: {failed}")

    categories: list[dict[str, Any]] = []
    for column in ("model_space", "source", "gold_tier", "broad_consequence_class"):
        if column not in selected.columns:
            continue
        counts = selected[column].fillna("__NA__").astype(str).value_counts(dropna=False)
        categories.extend(
            {"category": column, "value": value, "rows": int(count)}
            for value, count in counts.items()
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(categories).to_csv(
        output_dir / "rice_gwas_primary91_category_counts.tsv", sep="\t", index=False
    )
    audit = {
        "validation_set_semantics": (
            "external functional-effect positives; not a claim that every allele is "
            "deleterious or causal truth"
        ),
        "selection_chain": {
            "usable_rows": int(len(usable)),
            "primary_clear_effect_rows": int(len(primary)),
            "removed_exact_oof_overlap_rows": int(primary_overlap.sum()),
            "external_rows": int(len(selected)),
        },
        "unique_variant_keys": {
            "usable": int(usable_keys.nunique()),
            "primary": int(primary_keys.nunique()),
            "removed_overlap": int(primary_keys.loc[primary_overlap].nunique()),
            "selected": int(selected_keys.nunique()),
        },
        "checks": checks,
    }
    (output_dir / "rice_gwas_primary91_selection_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    )
    return audit


def validate_manifest(
    path: Path,
    expected_n: int,
    raw_key: str,
    expected_selected_hits: int,
    expected_raw_hits: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"canonical validation did not write {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest["n_rows"]) != expected_n:
        raise RuntimeError(
            f"unexpected validation row count {manifest['n_rows']} != {expected_n}"
        )
    selected_hits = int(manifest["strict_v3_hits_at_0_5"])
    raw_hits = int(manifest[raw_key])
    if selected_hits != expected_selected_hits or raw_hits != expected_raw_hits:
        raise RuntimeError(
            "external recall changed from the selected reference: "
            f"selected {selected_hits} != {expected_selected_hits} or "
            f"raw {raw_hits} != {expected_raw_hits}"
        )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run original-project strict external validations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("command", choices=("maize-3md", "rice-atlas", "all"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-registry", default="configs/plantvimpact/model_registry.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model-root-base",
        type=Path,
        help="Use ROOT/<species>/<mode> models produced by the current workflow.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_paths = load_yaml(args.config)
    paths = resolve_mapping(raw_paths, raw_paths)
    registry = load_yaml(args.model_registry)
    external_spec = registry.get("external_validation") or {}
    maize_expected = external_spec.get("maize_3dm") or {}
    rice_expected = external_spec.get("rice_gwas_atlas") or {}
    configured = get_paths(
        paths, args.model_root_base.resolve() if args.model_root_base else None
    )
    output_dir = Path(args.output).resolve()
    requested = (
        ["maize-3md", "rice-atlas"] if args.command == "all" else [args.command]
    )
    plan = {
        "schema_version": 1,
        "canonical_script": public_path(configured["canonical_script"]),
        "requested": requested,
        "configured_paths": {key: public_path(value) for key, value in configured.items()},
        "output_dir": public_path(output_dir),
        "dry_run": bool(args.dry_run),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "external_validation_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    log(f"wrote {plan_path}")
    if args.dry_run:
        return 0

    require_path(configured["canonical_script"], "canonical script")
    if "maize-3md" in requested:
        for key in ("maize_positives", "maize_redundancy", "maize_legacy_scores"):
            require_path(configured[key], key)
        require_path(configured["maize_model"], "selected maize model", directory=True)
    if "rice-atlas" in requested:
        for key in (
            "rice_usable_211",
            "rice_primary_157",
            "rice_oof_keys",
            "rice_selected_91",
            "rice_completed",
            "rice_redundancy",
            "rice_dna_coding",
            "rice_prot_coding",
        ):
            require_path(configured[key], key)
        require_path(configured["rice_coding_model"], "selected rice coding model", directory=True)
        require_path(
            configured["rice_noncoding_model"], "selected rice noncoding model", directory=True
        )

    canonical = load_canonical(configured["canonical_script"])
    inject_canonical_paths(canonical, configured)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "validation_set_semantics": "external functional-effect positives",
        "results": {},
        "passed": True,
    }
    if "maize-3md" in requested:
        canonical.run_maize_3md(output_dir / "maize_3md")
        maize_manifest = validate_manifest(
            output_dir / "maize_3md" / "maize_3md_positive45_manifest.json",
            45,
            "raw_any_hits",
            int(maize_expected["expected_selected_hits_at_0_5"]),
            int(maize_expected["expected_raw_any_hits"]),
        )
        if Path(maize_manifest["model_root"]).resolve() != configured["maize_model"].resolve():
            raise RuntimeError("maize validation did not use the configured selected model")
        summary["results"]["maize_3md"] = maize_manifest

    if "rice-atlas" in requested:
        selection_audit = audit_rice_selection(configured, output_dir / "rice_gwas91")
        canonical.run_rice_91(output_dir / "rice_gwas91")
        rice_manifest = validate_manifest(
            output_dir / "rice_gwas91" / "rice_gwas_primary91_manifest.json",
            91,
            "raw_paper_threshold_hits",
            int(rice_expected["expected_selected_hits_at_0_5"]),
            int(rice_expected["expected_raw_hits"]),
        )
        if int(rice_manifest["n_unique_variant_keys"]) != 90:
            raise RuntimeError("rice primary 91 set must contain 90 unique variant keys")
        if Path(rice_manifest["coding_model_root"]).resolve() != configured[
            "rice_coding_model"
        ].resolve():
            raise RuntimeError("rice validation did not use the configured coding model")
        if Path(rice_manifest["noncoding_model_root"]).resolve() != configured[
            "rice_noncoding_model"
        ].resolve():
            raise RuntimeError("rice validation did not use the configured noncoding model")
        summary["results"]["rice_gwas91"] = {
            "selection_audit": selection_audit,
            "validation": rice_manifest,
        }

    summary_path = output_dir / "external_validation_gate_report.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    log(f"PASS; wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
