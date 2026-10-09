#!/usr/bin/env python3
"""Fail closed on selected-score O/E promotion and sampled/exact gates."""
from __future__ import annotations
from phenofuncrank_repro.common.paths import project_root, package_dir

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPTS_ROOT = package_dir('plantvimpact')

from phenofuncrank_repro.plantvimpact.common.config_utils import dotted_get, load_yaml, model_spec


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required O/E result is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def finite_float(row: dict[str, str], column: str, path: Path) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {column!r} in {path}: {row}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite {column!r} in {path}: {value}")
    return value


def rows_by_fraction(path: Path, expected: list[float]) -> dict[float, float]:
    rows = read_tsv(path)
    result: dict[float, float] = {}
    for row in rows:
        fraction = finite_float(row, "top_fraction", path)
        result[fraction] = finite_float(row, "strat_OE", path)
    missing = [value for value in expected if value not in result]
    if missing:
        raise ValueError(f"missing top fractions {missing} in {path}")
    return result


def parse_fraction_csv(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-registry", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    registry = load_yaml(args.model_registry)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    global_oe = dotted_get(registry, "xgboost.oe_validation") or {}
    default_gate_fractions = [float(v) for v in global_oe.get(
        "promotion_gate_top_fractions", [0.01, 0.05, 0.10]
    )]
    default_tolerance = float(global_oe.get("concordance_max_abs_error", 0.06))

    report: dict[str, Any] = {
        "schema_version": 1,
        "lower_oe_is_better": True,
        "passed": True,
        "targets": [],
    }
    failures: list[str] = []

    for target in plan["targets"]:
        species = str(target["species"])
        mode = str(target["mode"])
        spec = model_spec(registry, species, mode)
        oe_spec = spec.get("oe") or {}
        target_fractions = parse_fraction_csv(
            str(target.get("top_fractions") or global_oe["top_fractions"])
        )
        target_report: dict[str, Any] = {
            "species": species,
            "mode": mode,
            "sampled_complete": False,
            "promotion_gate": None,
            "concordance_gate": None,
            "passed": True,
        }
        try:
            sampled_values: dict[str, dict[float, float]] = {}
            for score in target["scores"]:
                name = str(score["name"])
                if name not in target["sampled_outputs"]:
                    continue
                summary = (
                    Path(target["sampled_outputs"][name]) / "weighted_oe_summary.tsv"
                )
                sampled_values[name] = rows_by_fraction(summary, target_fractions)
            target_report["sampled_complete"] = bool(sampled_values)
            if oe_spec.get("sampled_required", True):
                expected_score_names = {str(item["name"]) for item in target["scores"]}
                if set(sampled_values) != expected_score_names:
                    raise ValueError(
                        "sampled summaries are incomplete: expected "
                        f"{sorted(expected_score_names)}, got {sorted(sampled_values)}"
                    )

            comparator = oe_spec.get("must_beat")
            if comparator:
                gate_fractions = [float(v) for v in oe_spec.get(
                    "gate_top_fractions", default_gate_fractions
                )]
                if "selected" not in sampled_values or comparator not in sampled_values:
                    raise ValueError(
                        f"promotion gate lacks selected/{comparator} sampled results"
                    )
                comparisons = []
                gate_passed = True
                for fraction in gate_fractions:
                    selected_oe = sampled_values["selected"][fraction]
                    raw_oe = sampled_values[str(comparator)][fraction]
                    passed = selected_oe < raw_oe
                    gate_passed = gate_passed and passed
                    comparisons.append({
                        "top_fraction": fraction,
                        "selected_strat_OE": selected_oe,
                        "comparator": str(comparator),
                        "comparator_strat_OE": raw_oe,
                        "passed": passed,
                    })
                target_report["promotion_gate"] = {
                    "rule": "selected_strat_OE < comparator_strat_OE",
                    "comparisons": comparisons,
                    "passed": gate_passed,
                }
                if not gate_passed:
                    raise ValueError("selected score did not beat its raw comparator")

            if oe_spec.get("require_concordance"):
                exact_dir = Path(str(target["exact_output"]))
                exact_rows = read_tsv(exact_dir / "exact_exhaustive_oe_summary.tsv")
                if not exact_rows:
                    raise ValueError("empty exhaustive O/E summary")
                comparison_path = (
                    exact_dir / "exact_vs_sampled_global_comparison_summary.tsv"
                )
                comparison_rows = read_tsv(comparison_path)
                if not comparison_rows:
                    raise ValueError("empty exact-vs-sampled comparison summary")
                tolerance = float(
                    oe_spec.get("concordance_max_abs_error", default_tolerance)
                )
                errors = [
                    finite_float(row, "max_abs_error", comparison_path)
                    for row in comparison_rows
                ]
                expected_pairs = {
                    (str(score["name"]), fraction)
                    for score in target["scores"]
                    for fraction in target_fractions
                }
                exact_pairs = {
                    (str(row.get("method", "")), finite_float(row, "top_fraction", exact_dir / "exact_exhaustive_oe_summary.tsv"))
                    for row in exact_rows
                }
                comparison_pairs = {
                    (str(row.get("method", "")), finite_float(row, "top_fraction", comparison_path))
                    for row in comparison_rows
                }
                if not expected_pairs.issubset(exact_pairs):
                    raise ValueError("exhaustive O/E summary lacks configured score/fraction pairs")
                if not expected_pairs.issubset(comparison_pairs):
                    raise ValueError("exact-vs-sampled summary lacks configured score/fraction pairs")
                max_error = max(errors)
                concordance_passed = max_error <= tolerance
                target_report["concordance_gate"] = {
                    "rule": "max(max_abs_error) <= tolerance",
                    "max_abs_error": max_error,
                    "tolerance": tolerance,
                    "passed": concordance_passed,
                }
                if not concordance_passed:
                    raise ValueError(
                        f"sampled/exact max error {max_error} exceeds {tolerance}"
                    )
        except Exception as exc:  
            target_report["passed"] = False
            target_report["error"] = str(exc)
            failures.append(f"{species}/{mode}: {exc}")
        report["targets"].append(target_report)

    report["passed"] = not failures
    report["failures"] = failures
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"[evaluate_oe_gates] wrote {output}", file=sys.stderr)
    if failures:
        for failure in failures:
            print(f"[evaluate_oe_gates] FAIL {failure}", file=sys.stderr)
        return 1
    print("[evaluate_oe_gates] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
