#!/usr/bin/env python3
"Independent fold-aware XGBoost trainer for the published contracts."
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


MAIZE_CODING_MONOTONE = {
    "S_esm": 1,
    "S_dna": 1,
    "dna_cons__phastCons": 1,
    "dna_cons__phyloP": 1,
    "prot_safe_ref_aa_freq": 1,
    "prot_safe_alt_aa_freq": -1,
    "prot_safe_entropy": -1,
    "prot_safe_delta_log_odds": -1,
    "prot_safe_conservation_penalty": 1,
    "prot_safe_abs_delta_log_odds": 1,
    "prot_safe_alt_absent_flag": 1,
}
MAIZE_NONCODING_MONOTONE = {
    "S_dna": 1,
    "dna_cons__phastCons": 1,
    "dna_cons__phyloP": 1,
}


def parse_folds(text: str) -> list[int]:
    result: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def load_features(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("features") or (payload.get("feature_columns") or {}).get("features")
    if not isinstance(values, list) or not values:
        raise ValueError(f"feature contract has no features: {path}")
    return [str(value) for value in values]


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    result: dict[str, Any] = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "pos_rate": float(y.mean()) if len(y) else math.nan,
        "logloss": float(log_loss(y, p, labels=[0, 1])) if len(y) else math.nan,
        "brier": float(brier_score_loss(y, p)) if len(y) else math.nan,
    }
    if len(y) and len(np.unique(y)) > 1:
        result["auc"] = float(roc_auc_score(y, p))
        result["ap"] = float(average_precision_score(y, p))
    else:
        result["auc"] = math.nan
        result["ap"] = math.nan
    return result


def _feature_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise RuntimeError(f"split is missing selected features: {missing}")
    return (
        frame.loc[:, features]
        .replace([np.inf, -np.inf], np.nan)
        .apply(pd.to_numeric, errors="coerce")
        .astype("float32")
    )


def _raw_base_margin(frame: pd.DataFrame, species: str, model_type: str) -> np.ndarray | None:
    """Return the registered raw margin used as the XGBoost base margin."""
    if species == "maize" and model_type == "coding":
        raw = 0.91 * pd.to_numeric(frame["S_dna"], errors="coerce") + 0.09 * pd.to_numeric(frame["S_esm"], errors="coerce")
        return (0.37 * ((raw - 2.46) / 2.4)).fillna(0).to_numpy(dtype="float32")
    if species == "maize" and model_type == "noncoding":
        raw = pd.to_numeric(frame["S_dna"], errors="coerce")
        return (0.057716 * ((raw - 1.217034) / 1.14099)).fillna(0).to_numpy(dtype="float32")
    return None


def _impute(train: pd.DataFrame, others: list[pd.DataFrame], features: list[str]) -> tuple[pd.DataFrame, list[pd.DataFrame], pd.DataFrame]:
    train = train.copy()
    others = [frame.copy() for frame in others]
    rows = []
    zero_policy = {
        feature for feature in features
        if feature.endswith("_has_homolog")
        or feature.endswith("_functional_redundancy")
        or feature in {"has_homolog", "functional_redundancy"}
    }
    for feature in features:
        before = int(train[feature].isna().sum())
        if feature in zero_policy:
            fill = 0.0
            policy = "zero_no_homolog"
        else:
            median = train[feature].median(skipna=True)
            fill = float(median) if pd.notna(median) and math.isfinite(float(median)) else 0.0
            policy = "fold_train_median"
        train[feature] = train[feature].fillna(fill).astype("float32")
        for frame in others:
            frame[feature] = frame[feature].fillna(fill).astype("float32")
        rows.append({"feature": feature, "train_median": fill, "fill_value": fill, "policy": policy, "train_missing_before": before})
    return train, others, pd.DataFrame(rows)


def _read_split(path: Path, features: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    columns = pq.ParquetFile(path).schema.names
    wanted = list(dict.fromkeys(["new_label", "snp_id", "SNP_ID", "Chr", "Pos", "Ref", "Alt", "fold", "chr_num"] + features))
    wanted = [column for column in wanted if column in columns]
    frame = pd.read_parquet(path, columns=wanted, engine="pyarrow")
    if "new_label" not in frame.columns:
        raise RuntimeError(f"split lacks new_label: {path}")
    return frame


def train_fold(args: argparse.Namespace, features: list[str], fold: int) -> dict[str, Any]:
    fold_root = args.split_root / f"fold{fold:02d}"
    train = _read_split(fold_root / "train.parquet", features)
    valid = _read_split(fold_root / "val.parquet", features)
    test = _read_split(fold_root / "test.parquet", features)
    x_train, [x_valid, x_test], imputation = _impute(
        _feature_frame(train, features),
        [_feature_frame(valid, features), _feature_frame(test, features)],
        features,
    )
    y_train = pd.to_numeric(train["new_label"], errors="coerce").fillna(0).astype("int8").to_numpy()
    y_valid = pd.to_numeric(valid["new_label"], errors="coerce").fillna(0).astype("int8").to_numpy()
    y_test = pd.to_numeric(test["new_label"], errors="coerce").fillna(0).astype("int8").to_numpy()
    train_base = _raw_base_margin(train, args.species, args.model_type)
    valid_base = _raw_base_margin(valid, args.species, args.model_type)
    test_base = _raw_base_margin(test, args.species, args.model_type)
    def make_matrix(values: pd.DataFrame, labels: np.ndarray, base: np.ndarray | None, ref: xgb.QuantileDMatrix | None = None):
        try:
            kwargs = {"label": labels, "feature_names": features, "missing": np.nan, "max_bin": args.max_bin}
            if base is not None:
                kwargs["base_margin"] = base
            if ref is not None:
                kwargs["ref"] = ref
            return xgb.QuantileDMatrix(values, **kwargs)
        except Exception as exc:
            if ref is not None:

                
                ref = None
            kwargs = {"label": labels, "feature_names": features, "missing": np.nan}
            if base is not None:
                kwargs["base_margin"] = base
            return xgb.DMatrix(values, **kwargs)

    d_train = make_matrix(x_train, y_train, train_base)
    d_valid = make_matrix(x_valid, y_valid, valid_base, ref=d_train)
    d_test = make_matrix(x_test, y_test, test_base, ref=d_train)
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc", "aucpr"],
        "tree_method": "hist",
        "device": args.device,
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "lambda": args.lambda_reg,
        "alpha": args.alpha_reg,
        "max_bin": args.max_bin,
        "nthread": args.nthread,
        "seed": args.seed,
    }
    if args.species == "maize":
        monotone = (
            MAIZE_CODING_MONOTONE
            if args.model_type == "coding"
            else MAIZE_NONCODING_MONOTONE
        )
        params["monotone_constraints"] = "(" + ",".join(
            str(int(monotone.get(feature, 0))) for feature in features
        ) + ")"
    out_fold = args.out_dir / "folds" / f"fold{fold:02d}"
    out_fold.mkdir(parents=True, exist_ok=True)
    booster = xgb.train(
        params,
        d_train,
        num_boost_round=args.num_boost_round,
        evals=[(d_train, "train"), (d_valid, "valid")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=args.verbose_eval,
    )
    model_path = out_fold / f"xgboost_{args.feature_set}.json"
    booster.save_model(str(model_path))
    (out_fold / "features.json").write_text(json.dumps(features, indent=2) + "\n", encoding="utf-8")
    imputation.to_csv(out_fold / "train_fold_median_imputation.tsv", sep="\t", index=False)
    end = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1)) + 1
    valid_p = booster.predict(d_valid, iteration_range=(0, end))
    test_p = booster.predict(d_test, iteration_range=(0, end))
    prediction_columns = [column for column in ["snp_id", "SNP_ID", "Chr", "Pos", "Ref", "Alt"] if column in test]
    predictions = test[prediction_columns].copy()
    predictions["new_label"] = y_test
    predictions["strict_v3_prob"] = test_p.astype("float32")
    predictions["strict_v3_margin"] = booster.predict(d_test, output_margin=True, iteration_range=(0, end)).astype("float32")
    predictions["fold"] = fold
    predictions.to_parquet(out_fold / "test_predictions.parquet", index=False)
    summary = {
        "fold": fold,
        "n_train": len(train), "n_val": len(valid), "n_test": len(test),
        "feature_set": args.feature_set, "n_features": len(features),
        "best_iteration": end - 1,
        "iteration_end": end,
        "val_metrics": _metrics(y_valid, valid_p),
        "test_metrics": _metrics(y_test, test_p),
        "model_path": str(model_path),
    }
    (out_fold / "fold_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_or_audit", choices=["train"])
    parser.add_argument("--species", required=True)
    parser.add_argument("--model-type", required=True, choices=["coding", "noncoding"])
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--eta", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--min-child-weight", type=float, default=20.0)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--alpha-reg", type=float, default=0.0)
    parser.add_argument("--max-bin", type=int, default=64)
    parser.add_argument("--nthread", type=int, default=8)
    parser.add_argument("--num-boost-round", type=int, default=3000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose-eval", type=int, default=100)
    args = parser.parse_args(argv)
    features = load_features(args.features)
    summaries = [train_fold(args, features, fold) for fold in parse_folds(args.folds)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_json(args.out_dir / "fold_summaries.json", orient="records", indent=2)
    cfg = {
        "schema_version": 1,
        "species": args.species,
        "model_type": args.model_type,
        "split_root": str(args.split_root),
        "main_feature_set": args.feature_set,
        "feature_sets": {args.feature_set: features},
        "imputation_policy": "fold_train_median_runtime",
        "trainer": "scripts/model_training/strict_xgb_retrain.py",
        "params": {key: getattr(args, key) for key in ("max_depth", "eta", "subsample", "colsample_bytree", "min_child_weight", "lambda_reg", "alpha_reg", "max_bin", "nthread", "num_boost_round", "early_stopping_rounds", "seed", "device")},
    }
    if args.species == "maize":
        monotone = MAIZE_CODING_MONOTONE if args.model_type == "coding" else MAIZE_NONCODING_MONOTONE
        cfg["monotone_constraints"] = {feature: int(monotone.get(feature, 0)) for feature in features}
    if args.species == "maize" and args.model_type == "coding":
        cfg.update({
            "raw_base_mode": "fixed_z",
            "raw_base_source": "blend",
            "raw_base_dna_weight": 0.91,
            "raw_score_formula": "0.91*S_dna + 0.09*S_esm",
            "fixed_raw_coef": 0.37,
            "fixed_raw_mean": 2.46,
            "fixed_raw_std": 2.4,
        })
    elif args.species == "maize" and args.model_type == "noncoding":
        cfg.update({
            "raw_base_mode": "fixed_z",
            "raw_base_source": "s_dna",
            "raw_score_formula": "S_dna",
            "fixed_raw_coef": 0.057716,
            "fixed_raw_mean": 1.217034,
            "fixed_raw_std": 1.14099,
        })
    (args.out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "training_complete.flag").write_text("ok\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
