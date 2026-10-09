#!/usr/bin/env python3
"""Project attached LOCO parquets to exact predictors plus operational data."""
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


OPERATIONAL_COLUMNS = [
    "species", "kind", "snp_id", "variant_id",
    "Chr", "chrom", "chromosome", "Pos", "position", "Ref", "Alt",
    "GeneID", "gene_id", "target_gene_id", "Mutation", "aa_pos", "ref_aa", "alt_aa",
    "region_type", "region_subtype", "Sg_code", "Ctx_code", "Alt_code",
    "label", "new_label", "Y_Label", "old_label", "observed_in_vcf",
    "AC", "AN", "AF", "AN_obs", "callable_n", "callrate", "callrate_hit",
    "P_mut", "P_mut_fold", "p_base_eff", "p_base_eff_fold",
    "p_base_eff_bin", "callable_n_bin", "opp_bin", "call_bin", "match_group",
    "sample_weight", "inclusion_probability", "fold", "fold_id", "split",
]


def contract(path: Path) -> tuple[list[str], list[str]]:
    data = load_json(path)
    features = data.get("features") or (data.get("feature_columns") or {}).get("features")
    if not isinstance(features, list):
        raise ValueError(f"no selected features in {path}")
    exceptions = [str(v) for v in data.get("allowed_legacy_exceptions", [])]
    features = [str(v) for v in features]
    assert_predictors_allowed(features, allowed_exceptions=exceptions, context=str(path))
    return features, exceptions


def project(input_root: Path, output_root: Path, features_path: Path, folds: list[int]) -> dict:
    features, exceptions = contract(features_path)
    records = []
    for fold in folds:
        for split in ("train", "val", "test"):
            source = input_root / f"fold{fold:02d}" / f"{split}.parquet"
            if not source.is_file():
                raise FileNotFoundError(source)
            available = pq.ParquetFile(source).schema.names
            missing = [name for name in features if name not in available]
            if missing:
                raise RuntimeError(f"{source} misses selected predictors: {missing}")
            keep = []
            for name in OPERATIONAL_COLUMNS + features:
                if name in available and name not in keep:
                    keep.append(name)
            table = pq.read_table(source, columns=keep)
            destination = output_root / f"fold{fold:02d}" / f"{split}.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination, compression="zstd")
            records.append({
                "fold": fold, "split": split, "input": str(source),
                "output": str(destination), "n_rows": table.num_rows,
                "selected_predictors": features,
                "operational_columns": [name for name in keep if name not in features],
                "dropped_attached_columns": [name for name in available if name not in keep],
            })
    return {
        "ok": True,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "features_path": str(features_path),
        "selected_predictors": features,
        "allowed_legacy_exceptions": exceptions,
        "splits": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    report = project(args.input_root, args.output_root, args.features, parse_folds(args.folds))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
