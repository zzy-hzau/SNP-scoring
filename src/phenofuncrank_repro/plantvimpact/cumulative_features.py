#!/usr/bin/env python3

"Cumulative forward ablation trainer (5 species x 2 regions)."
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss


FORBIDDEN_EXACT = {
    "new_label", "old_label", "observed_in_vcf",
    "AN_obs", "fold", "split",
    "snp_id", "Chr", "chrom", "chr", "Pos", "pos", "Ref", "Alt", "ref", "alt",
    "GeneID", "gene_id", "target_gene_id",
    "region_type", "region_subtype", "pcsd26_state",
    "match_group",
}
FORBIDDEN_PREFIXES = ("P_mut", "p_base", "callable", "Y_Label", "Sg_code", "Ctx_code", "Alt_code")


def safe_auc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    m = np.isfinite(s)
    y = y[m]; s = s[m]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_ap(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    m = np.isfinite(s)
    y = y[m]; s = s[m]
    if len(y) == 0 or int(y.sum()) == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def clf_metrics(y, prob, raw=None):
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    prob_clip = np.clip(prob, 1e-7, 1 - 1e-7)
    out = {
        "auc": safe_auc(y, prob),
        "ap": safe_ap(y, prob),
        "logloss": float(log_loss(y, prob_clip, labels=[0, 1])),
        "brier": float(brier_score_loss(y, prob_clip)),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "pos_rate": float(y.mean()) if len(y) else float("nan"),
    }
    if raw is not None:
        out["raw_auc"] = safe_auc(y, raw)
        out["raw_ap"] = safe_ap(y, raw)
    return out


def phred_from_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.argsort(-scores, kind="mergesort")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(1, n + 1)
    return (-10.0 * np.log10(rank / float(n))).astype(np.float32)


def parse_folds(s):
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        if "-" in x:
            a, b = x.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(x))
    return sorted(set(out))


def read_split(path, feature_union):
    keep = ["new_label", "snp_id", "Chr", "Pos", "Ref", "Alt"] + list(feature_union)
    pf = pq.ParquetFile(path)
    cols = pf.schema_arrow.names
    keep = [c for c in keep if c in cols]
    df = pd.read_parquet(path, columns=keep)
    if "new_label" not in df.columns:
        raise RuntimeError(f"Missing new_label in {path}")
    df["new_label"] = pd.to_numeric(df["new_label"], errors="coerce").fillna(0).astype(np.int32)
    for c in feature_union:
        if c not in df.columns:
            raise RuntimeError(f"Missing feature {c} in {path}")
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    return df


def make_dmatrix(df, feats, ref=None, max_bin=64):
    X = df[feats]
    y = df["new_label"].astype(np.int32).values
    try:
        if ref is None:
            return xgb.QuantileDMatrix(X, label=y, missing=np.nan, max_bin=max_bin)
        return xgb.QuantileDMatrix(X, label=y, missing=np.nan, ref=ref, max_bin=max_bin)
    except Exception as e:
        print("[WARN] QuantileDMatrix failed, fallback to DMatrix:", repr(e))
        return xgb.DMatrix(X, label=y, missing=np.nan)


def train_one(train_df, val_df, test_df, feats, params, max_bin, num_boost_round, early_stopping_rounds, model_path):
    dtrain = make_dmatrix(train_df, feats, max_bin=max_bin)
    dval = make_dmatrix(val_df, feats, ref=dtrain, max_bin=max_bin)
    dtest = make_dmatrix(test_df, feats, ref=dtrain, max_bin=max_bin)

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=200,
    )
    booster.save_model(str(model_path))
    best_iter = booster.best_iteration if booster.best_iteration is not None else num_boost_round - 1
    val_prob = booster.predict(dval, iteration_range=(0, best_iter + 1))
    test_prob = booster.predict(dtest, iteration_range=(0, best_iter + 1))
    val_margin = booster.predict(dval, output_margin=True, iteration_range=(0, best_iter + 1))
    test_margin = booster.predict(dtest, output_margin=True, iteration_range=(0, best_iter + 1))

    val_m = clf_metrics(val_df["new_label"], val_prob, val_margin)
    test_m = clf_metrics(test_df["new_label"], test_prob, test_margin)
    metrics = {
        "best_iteration": int(best_iter + 1),
        "val_auc": val_m["auc"],
        "val_ap": val_m["ap"],
        "val_logloss": val_m["logloss"],
        "val_brier": val_m["brier"],
        "test_auc": test_m["auc"],
        "test_ap": test_m["ap"],
        "test_logloss": test_m["logloss"],
        "test_brier": test_m["brier"],
        "test_n": test_m["n"],
        "test_n_pos": test_m["n_pos"],
        "test_pos_rate": test_m["pos_rate"],
    }
    return booster, metrics, test_prob, test_margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", default=None, help="Override folds (e.g., '1-12')")
    ap.add_argument("--num-boost-round", type=int, default=3000)
    ap.add_argument("--early-stopping-rounds", type=int, default=100)
    ap.add_argument("--levels", default=None, help="Comma-separated subset of level names to run (default: all)")
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text())

    split_root = Path(cfg["split_root"])
    out_dir = Path(cfg["out_dir"])
    model_type = cfg["model_type"]
    feature_levels = cfg["feature_levels"]  
    level_order = [name for name, _ in feature_levels]
    level_features = {name: list(feats) for name, feats in feature_levels}
    feature_levels_dict = {name: list(feats) for name, feats in feature_levels}
    base_params = cfg["params"].copy()
    num_boost_round = args.num_boost_round
    early_stopping_rounds = args.early_stopping_rounds

    if args.levels:
        wanted = [s.strip() for s in args.levels.split(",") if s.strip()]
        level_order = [n for n in level_order if n in wanted]
    if not level_order:
        print("[ERROR] No levels to run.")
        sys.exit(1)

    folds = parse_folds(args.folds) if args.folds else list(cfg["folds"])
    max_bin = base_params.get("max_bin", 64)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "folds").mkdir(parents=True, exist_ok=True)


    feature_union = sorted(set().union(*[level_features[n] for n in level_order]))


    first_train = split_root / f"fold{folds[0]:02d}" / "train.parquet"
    if not first_train.exists():
        raise RuntimeError(f"Missing first split: {first_train}")
    cols = pq.ParquetFile(first_train).schema_arrow.names
    missing = [f for f in feature_union if f not in cols]
    if missing:
        raise RuntimeError(f"Missing features in first split: {missing[:10]}{'...' if len(missing) > 10 else ''}")


    (out_dir / "training_config.json").write_text(json.dumps({
        "config_path": str(config_path),
        "split_root": str(split_root),
        "out_dir": str(out_dir),
        "model_type": model_type,
        "level_order": level_order,
        "level_features": level_features,
        "params": base_params,
        "folds": folds,
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
    }, indent=2))

    metric_rows = []
    fold_rows = []
    importance_rows = []
    pred_paths = []

    for fold in folds:
        fold_in = split_root / f"fold{fold:02d}"
        train_p = fold_in / "train.parquet"
        val_p = fold_in / "val.parquet"
        test_p = fold_in / "test.parquet"
        if not (train_p.exists() and val_p.exists() and test_p.exists()):
            print(f"[SKIP] fold{fold:02d}: missing split files")
            continue
        print("=" * 80)
        print(f"[fold{fold:02d}] reading splits")
        print("=" * 80)
        train_df = read_split(train_p, feature_union)
        val_df = read_split(val_p, feature_union)
        test_df = read_split(test_p, feature_union)
        if (train_df["new_label"].nunique() < 2 or
            val_df["new_label"].nunique() < 2 or
            test_df["new_label"].nunique() < 2):
            print(f"[SKIP] fold{fold:02d}: single class")
            continue

        fold_out = out_dir / "folds" / f"fold{fold:02d}"
        fold_out.mkdir(parents=True, exist_ok=True)

        fold_pred = test_df[["snp_id", "Chr", "Pos", "new_label"]].copy() if "snp_id" in test_df.columns else test_df[["Chr", "Pos", "new_label"]].copy()
        fold_pred["fold"] = int(fold)


        params = base_params.copy()
        params["seed"] = int(base_params.get("seed", 42)) + fold
        params["device"] = base_params.get("device", "cuda:0")

        for level_name in level_order:
            feats = level_features[level_name]
            n_feats = len(feats)
            print(f"[fold{fold:02d} | {level_name} | n={n_feats}] training")
            model_path = fold_out / f"xgboost_{level_name}.json"
            booster, m, test_prob, test_margin = train_one(
                train_df, val_df, test_df, feats, params, max_bin, num_boost_round, early_stopping_rounds, model_path,
            )
            m_row = {
                "fold": fold,
                "feature_set_name": level_name,
                "n_features": n_feats,
                "model_path": str(model_path),
                "method": "xgboost",
                **m,
            }
            metric_rows.append(m_row)
            print(json.dumps({k: m_row[k] for k in ("fold", "feature_set_name", "n_features", "test_auc", "test_ap", "best_iteration")}, indent=2))


            score_gain = booster.get_score(importance_type="gain")
            score_weight = booster.get_score(importance_type="weight")
            for feat in feats:
                importance_rows.append({
                    "fold": fold,
                    "level_name": level_name,
                    "feature": feat,
                    "importance_gain": float(score_gain.get(feat, 0.0)),
                    "importance_weight": float(score_weight.get(feat, 0.0)),
                })

            fold_pred[f"prob_{level_name}"] = test_prob.astype("float32")
            fold_pred[f"margin_{level_name}"] = test_margin.astype("float32")
            fold_pred[f"PHRED_{level_name}"] = phred_from_scores(test_margin)

        pred_path = fold_out / "test_predictions.parquet"
        fold_pred.to_parquet(pred_path, index=False)
        pred_paths.append(pred_path)
        fold_rows.append({
            "fold": fold,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "test_pos_rate": float(test_df["new_label"].mean()),
            "n_levels": len(level_order),
            "prediction_path": str(pred_path),
        })

    pd.DataFrame(metric_rows).to_csv(out_dir / "ablation_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / "outer_loco_summary.tsv", sep="\t", index=False)
    pd.DataFrame(importance_rows).to_csv(out_dir / "feature_importance.tsv", sep="\t", index=False)

    if pred_paths:
        oof = pd.concat([pd.read_parquet(p) for p in pred_paths], ignore_index=True)
        oof.to_parquet(out_dir / "all_oof_predictions.parquet", index=False)

    if metric_rows:
        s = pd.DataFrame(metric_rows)
        summary = (
            s.groupby("feature_set_name", sort=False)
            .agg(
                n_folds=("fold", "nunique"),
                n_features=("n_features", "first"),
                mean_test_auc=("test_auc", "mean"),
                std_test_auc=("test_auc", "std"),
                mean_test_ap=("test_ap", "mean"),
                std_test_ap=("test_ap", "std"),
            )
            .reset_index()
        )

        summary["__order"] = summary["feature_set_name"].map({n: i for i, n in enumerate(level_order)})
        summary = summary.sort_values("__order").drop(columns="__order")
        summary.to_csv(out_dir / "ablation_mean_summary.tsv", sep="\t", index=False)
        (out_dir / "mean_summary.json").write_text(json.dumps({
            "levels": level_order,
            "mean_auc_by_level": {row["feature_set_name"]: float(row["mean_test_auc"]) for _, row in summary.iterrows()},
        }, indent=2))
        print("[MEAN SUMMARY]")
        print(summary.to_string(index=False))
    print("[DONE]", out_dir)


if __name__ == "__main__":
    main()
