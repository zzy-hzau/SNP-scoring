#!/usr/bin/env python3
"Score a feature-rich O/E table with the selected fold-specific XGBoost models."
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb


def log(message: str) -> None:
    print(f"[score_loco_table] {message}", file=sys.stderr, flush=True)


def parse_folds(text: str) -> list[int]:
    folds: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            folds.extend(range(int(start), int(end) + 1))
        else:
            folds.append(int(part))
    return sorted(set(folds))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_path(value: str | None, model_root: Path) -> list[Path]:
    if not value:
        return []
    path = Path(value)
    if path.is_absolute():
        return [path]
    return [path, model_root / path, model_root.parent / path]


def _read_median_table(path: Path, default_fold: int | None = None) -> dict[int, dict[str, float]]:
    table = pd.read_csv(path, sep="\t")
    if "feature" not in table.columns:
        raise RuntimeError(f"median table has no feature column: {path}")
    value_col = next(
        (name for name in ("train_median", "fill_value") if name in table.columns),
        None,
    )
    if value_col is None:
        raise RuntimeError(f"median table has no train_median/fill_value column: {path}")
    if "fold" not in table.columns:
        if default_fold is None:
            raise RuntimeError(f"median table has no fold column: {path}")
        table["fold"] = int(default_fold)
    result: dict[int, dict[str, float]] = {}
    for fold, group in table.groupby("fold", sort=False):
        result[int(fold)] = {
            str(row.feature): float(getattr(row, value_col))
            for row in group.itertuples(index=False)
            if pd.notna(getattr(row, value_col))
        }
    return result


def load_fold_medians(model_root: Path, cfg: dict[str, Any], folds: Sequence[int]) -> tuple[dict[int, dict[str, float]], list[str]]:
    candidates: list[Path] = []
    candidates.extend(_candidate_path(cfg.get("fold_train_median_imputation"), model_root))
    for split_root in _candidate_path(cfg.get("split_root"), model_root):
        candidates.append(split_root / "train_fold_median_imputation.tsv")
    candidates.append(model_root / "train_fold_median_imputation.tsv")

    medians: dict[int, dict[str, float]] = {}
    sources: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        loaded = _read_median_table(path)
        for fold, values in loaded.items():
            medians.setdefault(fold, {}).update(values)
        sources.append(str(path.resolve()))
        break

    for fold in folds:
        path = model_root / "folds" / f"fold{fold:02d}" / "train_fold_median_imputation.tsv"
        if not path.is_file():
            continue
        loaded = _read_median_table(path, default_fold=int(fold))
        medians.setdefault(int(fold), {}).update(loaded[int(fold)])
        sources.append(str(path.resolve()))

    if sources:
        log(f"loaded imputation values for {len(medians)} folds")
    else:
        log("no fold median artifact found; retaining XGBoost native missing values")
    return medians, list(dict.fromkeys(sources))


def read_fold_summary(model_root: Path, fold: int) -> dict[str, Any]:
    path = model_root / "folds" / f"fold{fold:02d}" / "fold_summary.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def read_iteration_end(model_root: Path, fold: int, rounds: int) -> int | None:
    """Resolve the prediction end while supporting both historical conventions."""
    summary = read_fold_summary(model_root, fold)
    for key in ("xgb_iteration_end", "iteration_end"):
        if summary.get(key) is not None:
            return min(int(summary[key]), int(rounds))
    if summary.get("xgb_best_iteration_zero_based") is not None:
        return min(int(summary["xgb_best_iteration_zero_based"]) + 1, int(rounds))
    value = summary.get("xgb_best_iteration")
    if value is None:
        return None
    value = int(value)
    if value >= int(rounds):
        return int(rounds)
    return min(value + 1, int(rounds))


def resolve_model_path(model_root: Path, fold: int, feature_set: str) -> Path:
    fold_dir = model_root / "folds" / f"fold{fold:02d}"
    candidates = [
        fold_dir / f"xgboost_{feature_set}.json",
        fold_dir / "xgb_model.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "no registered XGBoost fold artifact found; tried "
        + ", ".join(str(path) for path in candidates)
    )


def _formula_raw_score(frame: pd.DataFrame, formula: str) -> pd.Series:
    expression = formula.replace(" ", "")
    if expression.lower() in {"s_dna", "mean(s_dna)"}:
        return pd.to_numeric(frame["S_dna"], errors="coerce")
    if expression.lower() == "s_esm":
        return pd.to_numeric(frame["S_esm"], errors="coerce")
    total = pd.Series(0.0, index=frame.index, dtype="float64")
    matched = False
    pattern = r"([+-]?)(?:(\d+(?:\.\d+)?)\*)?(S_dna|S_esm)"
    for sign, coefficient, column in re.findall(pattern, expression):
        if column not in frame.columns:
            raise RuntimeError(f"raw base formula needs missing column {column}: {formula}")
        weight = float(coefficient) if coefficient else 1.0
        if sign == "-":
            weight = -weight
        total += weight * pd.to_numeric(frame[column], errors="coerce")
        matched = True
    if not matched:
        raise RuntimeError(f"unsupported raw base formula: {formula}")
    return total


def raw_score_series(frame: pd.DataFrame, spec: dict[str, Any], cfg: dict[str, Any]) -> pd.Series:
    if spec.get("raw_score_formula"):
        return _formula_raw_score(frame, str(spec["raw_score_formula"]))
    source = str(spec.get("raw_score", cfg.get("raw_base_source", "s_dna"))).lower()
    if source in {"s_dna", "dna", "raw_dna"}:
        return pd.to_numeric(frame["S_dna"], errors="coerce")
    if source in {"s_esm", "esm", "raw_esm"}:
        return pd.to_numeric(frame["S_esm"], errors="coerce")
    if source in {"blend", "raw_blend", "raw_mean", "mean"}:
        weight = float(spec.get("dna_weight", cfg.get("raw_base_dna_weight", 0.5)))
        return (
            weight * pd.to_numeric(frame["S_dna"], errors="coerce")
            + (1.0 - weight) * pd.to_numeric(frame["S_esm"], errors="coerce")
        )
    raise RuntimeError(f"unsupported raw base score source: {source}")


def load_raw_margin_spec(model_root: Path, cfg: dict[str, Any], fold: int) -> dict[str, Any] | None:
    mode = cfg.get("raw_base_mode")
    enabled = bool(cfg.get("raw_base_margin_enabled", mode not in {None, "", "none"}))
    if not enabled or mode in {None, "", "none"}:
        return None
    fold_key = f"fold{fold:02d}"
    fold_spec = (cfg.get("fold_specs") or {}).get(fold_key) or {}
    if fold_spec.get("raw_base_margin"):
        return dict(fold_spec["raw_base_margin"])
    path = model_root / "folds" / fold_key / "raw_margin_spec.json"
    if path.is_file():
        return json.loads(path.read_text())
    summary = read_fold_summary(model_root, fold)
    for key in ("raw_base_margin", "raw_margin_spec"):
        if summary.get(key):
            return dict(summary[key])
    if mode != "fixed_z":
        raise RuntimeError(f"raw_base_mode={mode!r} but {fold_key} has no saved raw margin spec")
    required = ("fixed_raw_coef", "fixed_raw_mean", "fixed_raw_std")
    missing = [key for key in required if cfg.get(key) is None]
    if missing:
        raise RuntimeError(f"raw_base_mode=fixed_z but config lacks {missing}")
    source = cfg.get("raw_base_source", "s_dna")
    dna_weight = cfg.get("raw_base_dna_weight")
    formula = cfg.get("raw_score_formula")
    if source == "blend" and formula is None and dna_weight is not None:
        formula = f"{float(dna_weight):g}*S_dna + {1.0 - float(dna_weight):g}*S_esm"
    elif source in {None, "s_dna", "S_dna"}:
        formula = "S_dna"
    return {
        "raw_score": source or "s_dna",
        "raw_score_formula": formula,
        "dna_weight": dna_weight,
        "mode": "fixed_global_z",
        "mean": float(cfg["fixed_raw_mean"]),
        "std": float(cfg["fixed_raw_std"]),
        "coef": float(cfg["fixed_raw_coef"]),
        "intercept": 0.0,
    }


def raw_margin_from_spec(frame: pd.DataFrame, spec: dict[str, Any], cfg: dict[str, Any]) -> np.ndarray:
    raw = raw_score_series(frame, spec, cfg)
    std = float(spec["std"])
    if not np.isfinite(std) or std <= 0:
        raise RuntimeError(f"invalid raw base std: {std}")
    values = (
        float(spec.get("intercept", 0.0))
        + float(spec["coef"]) * ((raw - float(spec["mean"])) / std)
    ).to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError(f"raw base margin contains {int((~np.isfinite(values)).sum()):,} non-finite values")
    return values


def coerce_features(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise RuntimeError(f"input table missing model features: {missing}")
    return (
        frame.loc[:, list(features)]
        .replace([np.inf, -np.inf], np.nan)
        .apply(pd.to_numeric, errors="coerce")
        .astype("float32")
    )


def add_canonical_derived_features(
    frame: pd.DataFrame, features: Sequence[str]
) -> pd.DataFrame:
    result = frame.copy()
    if "S_dna" in features and "S_dna" not in result and "raw_S_dna" in result:
        result["S_dna"] = pd.to_numeric(result["raw_S_dna"], errors="coerce")
    if (
        "prot_safe_abs_delta_log_odds" in features
        and "prot_safe_abs_delta_log_odds" not in result
        and "prot_safe_delta_log_odds" in result
    ):
        result["prot_safe_abs_delta_log_odds"] = pd.to_numeric(
            result["prot_safe_delta_log_odds"], errors="coerce"
        ).abs()
    if (
        "prot_safe_alt_absent_flag" in features
        and "prot_safe_alt_absent_flag" not in result
        and "prot_safe_alt_aa_freq" in result
    ):
        result["prot_safe_alt_absent_flag"] = (
            pd.to_numeric(result["prot_safe_alt_aa_freq"], errors="coerce").fillna(0) <= 0
        ).astype("float32")
    return result


def impute_for_fold(features: pd.DataFrame, medians: dict[str, float]) -> tuple[pd.DataFrame, int]:
    if not medians:
        return features, 0
    result = features.copy()
    filled = 0
    for column in result.columns:
        if column not in medians:
            continue
        count = int(result[column].isna().sum())
        if count:
            result[column] = result[column].fillna(float(medians[column]))
            filled += count
    return result, filled


def score_frame(
    frame: pd.DataFrame,
    model_root: Path,
    cfg: dict[str, Any],
    feature_set: str,
    features: Sequence[str],
    folds: Sequence[int],
    fold_col: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fold_medians, median_sources = load_fold_medians(model_root, cfg, folds)
    probability = np.full(len(frame), np.nan, dtype=np.float32)
    margin = np.full(len(frame), np.nan, dtype=np.float32)
    fold_values = pd.to_numeric(frame[fold_col], errors="coerce").astype("Int64")
    summary: dict[str, Any] = {
        "feature_set": feature_set,
        "features": list(features),
        "median_sources": median_sources,
        "folds": {},
    }
    for fold in folds:
        indices = np.flatnonzero((fold_values == fold).to_numpy(dtype=bool, na_value=False))
        if len(indices) == 0:
            continue
        model_path = resolve_model_path(model_root, int(fold), feature_set)
        matrix = coerce_features(frame.iloc[indices], features)
        matrix, filled = impute_for_fold(matrix, fold_medians.get(int(fold), {}))

        if fold_medians.get(int(fold)) and matrix.isna().any().any():
            counts = matrix.isna().sum()
            raise RuntimeError(
                f"{feature_set} fold{fold:02d} still has NaN after imputation: "
                f"{counts[counts > 0].to_dict()}"
            )
        raw_spec = load_raw_margin_spec(model_root, cfg, int(fold))
        base_margin = None
        if raw_spec is not None:
            raw_frame = frame.iloc[indices].copy()
            for column in matrix.columns:
                raw_frame[column] = matrix[column].to_numpy()
            base_margin = raw_margin_from_spec(raw_frame, raw_spec, cfg)
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        rounds = int(booster.num_boosted_rounds())
        iteration_end = read_iteration_end(model_root, int(fold), rounds)
        if iteration_end is None and booster.attr("best_iteration") is not None:
            iteration_end = min(int(booster.attr("best_iteration")) + 1, rounds)
        predict_args = {"iteration_range": (0, iteration_end)} if iteration_end else {}
        dmatrix = xgb.DMatrix(
            matrix,
            feature_names=list(features),
            missing=np.nan,
            base_margin=base_margin,
        )
        probability[indices] = booster.predict(dmatrix, **predict_args).astype(np.float32)
        margin[indices] = booster.predict(
            dmatrix, output_margin=True, **predict_args
        ).astype(np.float32)
        summary["folds"][str(fold)] = {
            "rows": int(len(indices)),
            "filled_values": int(filled),
            "model": str(model_path.resolve()),
            "model_sha256": sha256_file(model_path),
            "num_boosted_rounds": rounds,
            "iteration_end_used": int(iteration_end) if iteration_end else None,
            "raw_base_margin": raw_spec,
        }
        log(f"fold{fold:02d} rows={len(indices):,} filled={filled:,} end={iteration_end or rounds}")
    unscored = int(np.isnan(probability).sum())
    if unscored:
        seen = sorted(set(fold_values.dropna().astype(int).tolist()))
        raise RuntimeError(f"left {unscored:,} rows unscored; input folds={seen}, requested={list(folds)}")
    return probability, margin, summary


def derive_fold_column(frame: pd.DataFrame, requested: str) -> str:
    if requested in frame.columns:
        return requested
    for candidate in ("chr_num", "fold"):
        if candidate in frame.columns:
            return candidate
    for candidate in ("Chr", "chrom", "Chromosome"):
        if candidate in frame.columns:
            values = frame[candidate].astype(str).str.extract(r"(\d+)", expand=False)
            if values.isna().any():
                raise RuntimeError(f"cannot derive every LOCO fold from {candidate}")
            frame["chr_num"] = values.astype("int16")
            return "chr_num"
    raise RuntimeError(f"missing requested fold column {requested} and no chromosome fallback")


def run(
    input_path: Path,
    model_root: Path,
    output_path: Path,
    folds: Sequence[int],
    fold_col: str,
    probability_col: str,
    margin_col: str,
    feature_contract: Path | None = None,
) -> dict[str, Any]:
    config_path = model_root / "config.json"
    contract = json.loads(feature_contract.read_text()) if feature_contract else None
    if config_path.is_file():
        cfg = json.loads(config_path.read_text())
        feature_set = str(cfg["main_feature_set"])
        features = [str(value) for value in cfg["feature_sets"][feature_set]]
    elif contract is not None:
        feature_set = str(contract["main_feature_set"])
        features = [str(value) for value in contract["features"]]
        cfg = {
            "main_feature_set": feature_set,
            "feature_sets": {feature_set: features},
            "raw_base_mode": "none",
        }
        for fold in folds:
            saved_features_path = (
                model_root / "folds" / f"fold{int(fold):02d}" / "features.json"
            )
            if not saved_features_path.is_file():
                raise FileNotFoundError(saved_features_path)
            saved_features = [str(value) for value in json.loads(saved_features_path.read_text())]
            if saved_features != features:
                raise RuntimeError(
                    f"fold{int(fold):02d} saved feature order differs from selected contract"
                )
    else:
        raise FileNotFoundError(
            f"{config_path} is absent and no --feature-contract was provided"
        )
    if contract is not None:
        expected = [str(value) for value in contract["features"]]
        if features != expected:
            raise RuntimeError(
                "saved model feature order differs from selected contract: "
                f"model={features}, contract={expected}"
            )
    frame = pd.read_parquet(input_path)
    if frame.empty:
        raise RuntimeError("cannot score an empty table")
    frame = add_canonical_derived_features(frame, features)
    resolved_fold_col = derive_fold_column(frame, fold_col)
    probability, margin, scoring = score_frame(
        frame, model_root, cfg, feature_set, features, folds, resolved_fold_col
    )
    frame[probability_col] = probability
    frame[margin_col] = margin
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), output_path, compression="zstd"
    )
    manifest = {
        "schema_version": 1,
        "input": str(input_path.resolve()),
        "model_root": str(model_root.resolve()),
        "model_config_sha256": sha256_file(config_path) if config_path.is_file() else None,
        "feature_contract_sha256": (
            sha256_file(feature_contract) if feature_contract is not None else None
        ),
        "output": str(output_path.resolve()),
        "rows": int(len(frame)),
        "fold_col": resolved_fold_col,
        "folds": list(folds),
        "probability_column": probability_col,
        "margin_column": margin_col,
        "ranking_column": None,
        "ranking_transform": None,
        "scoring": scoring,
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--fold-col", default="chr_num")
    parser.add_argument("--probability-col", default="strict_v3_prob")
    parser.add_argument("--margin-col", default="strict_v3_margin")
    parser.add_argument("--feature-contract", type=Path)
    args = parser.parse_args(argv)
    report = run(
        args.input,
        args.model_root,
        args.output,
        parse_folds(args.folds),
        args.fold_col,
        args.probability_col,
        args.margin_col,
        args.feature_contract,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
