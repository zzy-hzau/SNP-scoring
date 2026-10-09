"PlantVImpact preprocessing and model training."
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

PROT_SAFE_CORE = [
    "prot_safe_ref_aa_freq",
    "prot_safe_alt_aa_freq",
    "prot_safe_entropy",
    "prot_safe_delta_log_odds",
    "prot_safe_conservation_penalty",
    "prot_safe_abs_delta_log_odds",
    "prot_safe_alt_absent_flag",
]


DNA_CONS = ["dna_cons__phastCons", "dna_cons__phyloP"]


CONTEXT = [
    "dist_to_TSS",
    "dist_to_gene",
    "promoter_core_flag",
    "promoter_proximal_flag",
    "promoter_distal_flag",
    "canonical_splice_core_flag",
    "donor_offset",
    "acceptor_offset",
    "region_type_UTR3",
    "region_type_UTR5",
    "region_type_intron",
    "region_type_promoter",
    "region_type_splice_near",
    "region_subtype_UTR3",
    "region_subtype_UTR5",
    "region_subtype_intron",
    "region_subtype_promoter_core",
    "region_subtype_promoter_distal",
    "region_subtype_promoter_proximal",
    "region_subtype_splice_core",
    "region_subtype_splice_near_other",
]


COTTON_NONCODING_REG = [
    "homoeolog_expr_bias",
    "atac_overlap_any",
    "atac_overlap_count",
    "atac_sample_count",
    "atac_tissue_group_count",
    "atac_overlap_fiber",
    "log_dist_to_nearest_atac_peak",
    "log_dist_to_nearest_atac_summit",
    "atac_max_signalValue_log1p",
    "h3k27ac_overlap_any",
    "h3k27ac_overlap_count",
    "h3k27ac_sample_count",
    "h3k27ac_tissue_group_count",
    "h3k27ac_overlap_fiber",
    "log_dist_to_nearest_h3k27ac_peak",
    "h3k27ac_max_signalValue_log1p",
    "active_regulatory_flag",
    "active_enhancer_like_flag",
    "fiber_active_enhancer_like_flag",
    "promoter_with_activity",
    "motif_loss_count",
    "motif_gain_count",
    "motif_disruption_any",
    "max_abs_delta_pwm",
    "max_loss_delta_pwm",
    "max_gain_delta_pwm",
    "motif_disruption_in_ATAC",
    "motif_disruption_in_H3K27ac",
    "motif_disruption_in_active_enhancer_like",
    "motif_disruption_in_active_promoter",
]


@dataclass(frozen=True)
class SpeciesSpec:
    species: str
    model_type: str
    split_root: Path
    out_dir: Path
    folds: str
    feature_set: str
    dna_cons: bool
    redundancy_prefix: str | None
    redundancy_tsv: Path | None = None
    gene_map_root: Path | None = None
    gene_map_parquet: Path | None = None


def parse_folds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def natural_pcsd(cols: Iterable[str]) -> list[str]:
    def key(c: str) -> int:
        m = re.search(r"__E(\d+)$", c)
        return int(m.group(1)) if m else 999

    return sorted([c for c in cols if c.startswith("pcsd26_state__")], key=key)


def required_features(spec: SpeciesSpec, available_cols: Iterable[str]) -> list[str]:
    cols = set(available_cols)
    feats: list[str] = []
    if spec.model_type == "coding":
        feats.extend(["S_esm", "S_dna"])
        if spec.dna_cons:
            feats.extend(DNA_CONS)
        if spec.redundancy_prefix is not None:
            if spec.redundancy_prefix:
                feats.extend([f"{spec.redundancy_prefix}_has_homolog", f"{spec.redundancy_prefix}_functional_redundancy"])
            else:
                feats.extend(["has_homolog", "functional_redundancy"])
        feats.extend(PROT_SAFE_CORE)
    else:
        feats.append("S_dna")
        if spec.dna_cons:
            feats.extend(DNA_CONS)
        if spec.redundancy_prefix is not None:
            if spec.redundancy_prefix:
                feats.extend([f"{spec.redundancy_prefix}_has_homolog", f"{spec.redundancy_prefix}_functional_redundancy"])
            else:
                feats.extend(["has_homolog", "functional_redundancy"])
        if spec.species == "maize":
            feats.extend(natural_pcsd(cols))
        feats.extend([c for c in CONTEXT if c in cols])
        if spec.species == "rice":
            feats.extend(sorted([c for c in cols if c.startswith("predmoter_")]))
        if spec.species == "cotton":
            feats.extend([c for c in COTTON_NONCODING_REG if c in cols])
    return list(dict.fromkeys(feats))


def read_schema_cols(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema.names


def read_redundancy_map(spec: SpeciesSpec) -> pd.DataFrame | None:
    if spec.redundancy_tsv is None:
        return None
    cols = ["gene_id", "has_homolog", "functional_redundancy"]
    df = pd.read_csv(spec.redundancy_tsv, sep="\t", usecols=lambda c: c in cols)
    df["gene_id"] = df["gene_id"].astype(str)
    df["has_homolog"] = pd.to_numeric(df["has_homolog"], errors="coerce").fillna(0).astype("float32")
    df["functional_redundancy"] = pd.to_numeric(df["functional_redundancy"], errors="coerce").fillna(0).astype("float32")
    df = df.drop_duplicates("gene_id", keep="first").set_index("gene_id")
    return df


def load_gene_map_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["snp_id", "GeneID"])
    df = df.dropna(subset=["snp_id"]).drop_duplicates("snp_id", keep="first")
    df["snp_id"] = df["snp_id"].astype(str)
    df["GeneID"] = df["GeneID"].astype(str)
    return df.set_index("snp_id")


def attach_gene_id(df: pd.DataFrame, spec: SpeciesSpec, fold: int, split: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "GeneID" in df.columns or spec.redundancy_tsv is None:
        return df
    if spec.gene_map_root is not None:
        p = spec.gene_map_root / f"fold{fold:02d}" / f"{split}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing gene map split: {p}")
        gm = pd.read_parquet(p, columns=["snp_id", "GeneID"])
        gm = gm.drop_duplicates("snp_id", keep="first")
        df = df.merge(gm, on="snp_id", how="left", validate="m:1")
        return df
    if spec.gene_map_parquet is not None:
        key = str(spec.gene_map_parquet)
        if key not in cache:
            cache[key] = load_gene_map_parquet(spec.gene_map_parquet)
        gene = df["snp_id"].astype(str).map(cache[key]["GeneID"])
        df = df.copy()
        df["GeneID"] = gene.to_numpy()
        return df
    return df


def attach_redundancy(df: pd.DataFrame, spec: SpeciesSpec, red: pd.DataFrame | None) -> pd.DataFrame:
    if red is None or spec.redundancy_prefix is None:
        return df
    if spec.redundancy_prefix:
        target_cols = [f"{spec.redundancy_prefix}_has_homolog", f"{spec.redundancy_prefix}_functional_redundancy"]
    else:
        target_cols = ["has_homolog", "functional_redundancy"]
    if all(c in df.columns for c in target_cols) and "GeneID" not in df.columns:
        df = df.copy()
        for c in target_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")
        return df
    if "GeneID" not in df.columns:
        raise RuntimeError(f"{spec.species} {spec.model_type}: cannot attach redundancy without GeneID")
    gene = df["GeneID"].astype(str)
    has = gene.map(red["has_homolog"]).fillna(0).astype("float32")
    val = gene.map(red["functional_redundancy"]).fillna(0).astype("float32")
    df = df.copy()
    df[target_cols[0]] = has.to_numpy()
    df[target_cols[1]] = val.to_numpy()
    return df


def read_split(path: Path, spec: SpeciesSpec, fold: int, split: str, base_cols: list[str], red: pd.DataFrame | None, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = read_schema_cols(path)
    read_cols = [c for c in ["new_label", "snp_id", "Chr", "Pos", "Ref", "Alt", "GeneID"] + base_cols if c in cols]
    df = pd.read_parquet(path, columns=list(dict.fromkeys(read_cols)))
    df = attach_gene_id(df, spec, fold, split, cache)
    df = attach_redundancy(df, spec, red)
    return df


def make_feature_frame(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required feature columns: {missing}")
    x = df[feats].copy()
    for c in feats:
        x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    return x


def impute_features(train_x: pd.DataFrame, others: list[pd.DataFrame], feats: list[str], red_cols: set[str]) -> tuple[pd.DataFrame, list[pd.DataFrame], pd.DataFrame]:
    rows = []
    train_out = train_x.copy()
    other_out = [x.copy() for x in others]
    for c in feats:
        before_train = int(train_out[c].isna().sum())
        if c in red_cols:
            fill = 0.0
            policy = "zero_no_homolog"
        else:
            med = float(train_out[c].median(skipna=True))
            fill = med if math.isfinite(med) else 0.0
            policy = "fold_train_median"
        train_out[c] = train_out[c].fillna(fill).astype("float32")
        for x in other_out:
            x[c] = x[c].fillna(fill).astype("float32")
        rows.append({"feature": c, "fill_value": fill, "policy": policy, "train_missing_before": before_train})
    return train_out, other_out, pd.DataFrame(rows)


def safe_metric(fn, y, s) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    m = np.isfinite(s)
    y = y[m]
    s = s[m]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(fn(y, s))


def metrics(y, p) -> dict[str, float | int]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "auc": safe_metric(roc_auc_score, y, p),
        "ap": safe_metric(average_precision_score, y, p),
        "logloss": float(log_loss(y, pc, labels=[0, 1])),
        "brier": float(brier_score_loss(y, pc)),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "pos_rate": float(y.mean()) if len(y) else float("nan"),
    }


def train_one_fold(spec: SpeciesSpec, fold: int, args, red: pd.DataFrame | None, cache: dict[str, pd.DataFrame]) -> dict:
    fold_dir = spec.split_root / f"fold{fold:02d}"
    train_p = fold_dir / "train.parquet"
    val_p = fold_dir / "val.parquet"
    test_p = fold_dir / "test.parquet"
    if not train_p.exists() or not val_p.exists() or not test_p.exists():
        raise FileNotFoundError(f"Missing split files for fold{fold:02d} under {spec.split_root}")

    schema_cols = read_schema_cols(train_p)
    prelim = required_features(spec, schema_cols)
    base_read = sorted(set(prelim + ["GeneID"]))
    train_df = read_split(train_p, spec, fold, "train", base_read, red, cache)
    schema_aug_cols = set(schema_cols) | set(train_df.columns)
    feats = required_features(spec, schema_aug_cols)
    val_df = read_split(val_p, spec, fold, "val", feats, red, cache)
    test_df = read_split(test_p, spec, fold, "test", feats, red, cache)

    red_cols = {c for c in feats if c.endswith("_has_homolog") or c.endswith("_functional_redundancy") or c in {"has_homolog", "functional_redundancy"}}
    train_x = make_feature_frame(train_df, feats)
    val_x = make_feature_frame(val_df, feats)
    test_x = make_feature_frame(test_df, feats)
    train_x, (val_x, test_x), imp = impute_features(train_x, [val_x, test_x], feats, red_cols)

    y_train = pd.to_numeric(train_df["new_label"], errors="coerce").fillna(0).astype("int8").to_numpy()
    y_val = pd.to_numeric(val_df["new_label"], errors="coerce").fillna(0).astype("int8").to_numpy()
    y_test = pd.to_numeric(test_df["new_label"], errors="coerce").fillna(0).astype("int8").to_numpy()

    dtrain = xgb.QuantileDMatrix(train_x, label=y_train, max_bin=args.max_bin)
    dval = xgb.QuantileDMatrix(val_x, label=y_val, ref=dtrain, max_bin=args.max_bin)
    dtest = xgb.QuantileDMatrix(test_x, label=y_test, ref=dtrain, max_bin=args.max_bin)

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
        "seed": args.seed + fold,
    }

    out_fold = spec.out_dir / "folds" / f"fold{fold:02d}"
    out_fold.mkdir(parents=True, exist_ok=True)
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=args.verbose_eval,
    )
    model_path = out_fold / f"xgboost_{spec.feature_set}.json"
    booster.save_model(str(model_path))
    (out_fold / "features.json").write_text(json.dumps(feats, indent=2) + "\n")
    imp.to_csv(out_fold / "train_fold_median_imputation.tsv", sep="\t", index=False)

    val_prob = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    test_prob = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
    val_m = metrics(y_val, val_prob)
    test_m = metrics(y_test, test_prob)

    pred_cols = [c for c in ["snp_id", "Chr", "Pos", "Ref", "Alt"] if c in test_df.columns]
    pred = test_df[pred_cols].copy()
    pred["new_label"] = y_test
    pred[f"xgb_prob_{spec.feature_set}"] = test_prob.astype("float32")
    pred["fold"] = int(fold)
    pred.to_parquet(out_fold / "test_predictions.parquet", index=False)

    fold_summary = {
        "fold": int(fold),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "test_pos_rate": float(y_test.mean()) if len(y_test) else float("nan"),
        "main_feature_set": spec.feature_set,
        "main_n_features": int(len(feats)),
        "xgb_val_auc": val_m["auc"],
        "xgb_val_ap": val_m["ap"],
        "xgb_test_auc": test_m["auc"],
        "xgb_test_ap": test_m["ap"],
        "xgb_best_iteration": int(booster.best_iteration),
        "prediction_path": str((out_fold / "test_predictions.parquet").relative_to(spec.out_dir)),
        "model_path": str(model_path.relative_to(spec.out_dir)),
    }
    (out_fold / "fold_summary.json").write_text(json.dumps(fold_summary, indent=2) + "\n")
    print(json.dumps(fold_summary, indent=2), flush=True)
    return fold_summary


def train_spec(spec: SpeciesSpec, args) -> None:
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    (spec.out_dir / "folds").mkdir(exist_ok=True)
    folds = parse_folds(args.folds or spec.folds)
    red = read_redundancy_map(spec)
    cache: dict[str, pd.DataFrame] = {}
    rows = []
    for fold in folds:
        print(f"\n========== {spec.species} {spec.model_type} fold{fold:02d} ==========", flush=True)
        rows.append(train_one_fold(spec, fold, args, red, cache))
    summary = pd.DataFrame(rows)
    summary.to_csv(spec.out_dir / "outer_loco_summary.tsv", sep="\t", index=False)
    mean_summary = {
        "species": spec.species,
        "model_type": spec.model_type,
        "n_folds": int(len(summary)),
        "main_feature_set": spec.feature_set,
        "mean_test_auc": float(summary["xgb_test_auc"].mean()),
        "mean_test_ap": float(summary["xgb_test_ap"].mean()),
        "mean_best_iteration": float(summary["xgb_best_iteration"].mean()),
    }
    (spec.out_dir / "mean_summary.json").write_text(json.dumps(mean_summary, indent=2) + "\n")
    first_features = json.loads((spec.out_dir / "folds" / f"fold{folds[0]:02d}" / "features.json").read_text())
    cfg = {
        "species": spec.species,
        "model_type": spec.model_type,
        "split_root": "resources/splits",
        "out_dir": ".",
        "main_feature_set": spec.feature_set,
        "feature_sets": {spec.feature_set: first_features},
        "imputation_policy": "fold_train_median_runtime",
        "redundancy_source": "resources/redundancy.tsv" if spec.redundancy_tsv else None,
        "gene_map_root": "resources/gene_map_splits" if spec.gene_map_root else None,
        "gene_map_parquet": "resources/gene_map.parquet" if spec.gene_map_parquet else None,
        "cotton_dna_conservation_used": False if spec.species == "cotton" else None,
        "params": {
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
        },
    }
    (spec.out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (spec.out_dir / "training_complete.flag").write_text("ok\n")


def audit_spec(spec: SpeciesSpec) -> None:
    cfg_p = spec.out_dir / "config.json"
    if not cfg_p.exists():
        raise FileNotFoundError(cfg_p)
    cfg = json.loads(cfg_p.read_text())
    feature_set = cfg["main_feature_set"]
    expected = cfg["feature_sets"][feature_set]
    rows = []
    for fold_dir in sorted((spec.out_dir / "folds").glob("fold*")):
        feat_p = fold_dir / "features.json"
        model_p = fold_dir / f"xgboost_{feature_set}.json"
        imp_p = fold_dir / "train_fold_median_imputation.tsv"
        feats = json.loads(feat_p.read_text())
        booster = xgb.Booster()
        booster.load_model(str(model_p))
        bfeats = booster.feature_names or []
        imp = pd.read_csv(imp_p, sep="\t")
        rows.append(
            {
                "fold": fold_dir.name,
                "features_match_config": feats == expected,
                "booster_match_features": bfeats == feats,
                "n_features": len(feats),
                "unexpected_dna_cons_cotton": bool(spec.species == "cotton" and any(c.startswith("dna_cons__") for c in feats)),
                "redundancy_zero_policy_rows": int((imp["policy"] == "zero_no_homolog").sum()),
                "median_policy_rows": int((imp["policy"] == "fold_train_median").sum()),
                "missing_impute_features": ",".join(imp.loc[imp["train_missing_before"] > 0, "feature"].astype(str).tolist()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(spec.out_dir / "post_train_feature_audit.tsv", sep="\t", index=False)
    fail = out[
        ~out["features_match_config"]
        | ~out["booster_match_features"]
        | out["unexpected_dna_cons_cotton"]
    ]
    if len(fail):
        raise RuntimeError(f"Feature audit failed for {spec.species} {spec.model_type}:\n{fail.to_string(index=False)}")
    print(out.to_string(index=False))


def species_spec(profile, data_root, output_root):
    """Resolve only caller-supplied relative resource references."""
    data_root = Path(data_root)
    resources = profile["resources"]
    def asset(key):
        value = resources.get(key)
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute():
            raise ValueError("Profiles use paths relative to --data-root")
        return data_root / path
    return SpeciesSpec(
        profile["species"], profile["model_type"], asset("splits"),
        Path(output_root), profile["folds"], profile["main_feature_set"],
        profile["dna_conservation"], profile["redundancy_prefix"],
        asset("redundancy"), asset("gene_map_splits"), asset("gene_map"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "audit"])
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nthread", type=int, default=4)
    parser.add_argument("--folds")
    parser.add_argument("--verbose-eval", type=int, default=100)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    for name, value in profile["training"].items():
        setattr(args, name, value)
    spec = species_spec(profile, args.data_root, args.out_dir)
    if args.command == "train":
        train_spec(spec, args)
    else:
        audit_spec(spec)


if __name__ == "__main__":
    main()
