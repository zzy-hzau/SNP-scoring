#!/usr/bin/env python3
from __future__ import annotations
from phenofuncrank_repro.common.paths import project_root, package_dir

import argparse
import json
import math
import re
from pathlib import Path
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import xgboost as xgb


ROOT = project_root(__file__)


MAIZE_3MD_PRED = ROOT / "inputs/external_validation/maize_3dm_positive45.parquet"
MAIZE_RED = ROOT / "inputs/external_validation/maize_paralog_redundancy.tsv"
MAIZE_MODEL = Path(os.environ.get("MAIZE_MODEL", str(ROOT / "inputs/models/maize/coding")))
MAIZE_BLEND91_3MD = ROOT / "inputs/external_validation/maize_3dm_legacy_blend91.parquet"

RICE_91 = ROOT / "inputs/external_validation/rice_primary_no_oof_overlap_91.tsv"
RICE_COMPLETED = ROOT / "inputs/external_validation/rice_gwas_atlas_completed_scores.parquet"
RICE_RED = ROOT / "inputs/external_validation/rice_paralog_redundancy.tsv"
RICE_DNA_CODING = ROOT / "inputs/external_validation/rice_coding_dna_conservation.parquet"
RICE_PROT_CODING = ROOT / "inputs/external_validation/rice_coding_protein_conservation.parquet"
RICE_CODING_MODEL = ROOT / "inputs/models/rice/coding"
RICE_NONCODING_MODEL = ROOT / "inputs/models/rice/noncoding"


def chr_to_fold(value) -> int:
    text = str(value)
    match = re.search(r"(\d+)", text)
    if not match:
        raise ValueError(f"Cannot parse chromosome/fold from {value!r}")
    return int(match.group(1))


def public_path(path: Path) -> str:
    """Serialize a path without exposing a user's local directory prefix."""
    path = Path(path)
    return f"<local>/{path.name}" if path.is_absolute() else str(path)


def norm_chr(value) -> str:
    return f"chr{chr_to_fold(value):02d}"


def variant_key_from_cols(df: pd.DataFrame, chr_col: str = "Chr") -> pd.Series:
    return (
        df[chr_col].map(norm_chr)
        + ":"
        + df["Pos"].astype("int64").astype(str)
        + ":"
        + df["Ref"].astype(str)
        + ":"
        + df["Alt"].astype(str)
    )


def snp_id_from_cols(df: pd.DataFrame, chr_col: str = "Chr") -> pd.Series:
    return (
        df[chr_col].map(norm_chr)
        + "_"
        + df["Pos"].astype("int64").astype(str)
        + "_"
        + df["Ref"].astype(str)
        + "_"
        + df["Alt"].astype(str)
    )


def read_model_config(model_root: Path) -> tuple[str, list[str]]:
    cfg = json.loads((model_root / "config.json").read_text())
    feature_set = cfg["main_feature_set"]
    return feature_set, list(cfg["feature_sets"][feature_set])


def raw_base_margin_spec(cfg: dict, fold: int) -> dict | None:
    mode = cfg.get("raw_base_mode")
    enabled = bool(cfg.get("raw_base_margin_enabled", mode not in {None, "", "none"}))
    if not enabled:
        return None
    spec = cfg.get("fold_specs", {}).get(f"fold{fold:02d}", {}).get("raw_base_margin")
    if spec:
        return dict(spec)
    required = ("fixed_raw_coef", "fixed_raw_mean", "fixed_raw_std")
    if any(cfg.get(key) is None for key in required):
        raise RuntimeError(f"Raw-base model config lacks a base-margin specification for fold {fold}")
    return {
        "raw_score": cfg.get("raw_base_source", "blend"),
        "dna_weight": float(cfg.get("raw_base_dna_weight", 0.5)),
        "mode": "fixed_global_z",
        "mean": float(cfg["fixed_raw_mean"]),
        "std": float(cfg["fixed_raw_std"]),
        "coef": float(cfg["fixed_raw_coef"]),
        "intercept": 0.0,
    }


def compute_raw_base_margin(frame: pd.DataFrame, spec: dict) -> np.ndarray:
    """Compute the configured raw-base margin for externally scored rows."""
    missing = [c for c in ("S_dna", "S_esm") if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Cannot score raw-base model: missing raw-score columns {missing}")
    source = str(spec.get("raw_score", "blend")).lower()
    s_dna = pd.to_numeric(frame.get("S_dna"), errors="coerce").to_numpy(dtype="float64")
    s_esm = pd.to_numeric(frame.get("S_esm"), errors="coerce").to_numpy(dtype="float64")
    if source == "blend":
        raw = float(spec["dna_weight"]) * s_dna + (1.0 - float(spec["dna_weight"])) * s_esm
    elif source in {"dna", "s_dna"}:
        raw = s_dna
    elif source in {"esm", "s_esm"}:
        raw = s_esm
    else:
        raise RuntimeError(f"Unsupported raw-base source: {source!r}")
    std = float(spec["std"])
    if not np.isfinite(std) or std <= 0:
        raise RuntimeError(f"Invalid raw-base standard deviation: {std}")
    margin = float(spec["coef"]) * ((raw - float(spec["mean"])) / std) + float(spec.get("intercept", 0.0))
    if not np.isfinite(margin).all():
        raise RuntimeError("Cannot score raw-base model: S_dna/S_esm produced a non-finite base_margin")
    return margin.astype("float32")


def read_redundancy(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", usecols=["gene_id", "has_homolog", "functional_redundancy"])
    df["gene_id"] = df["gene_id"].astype(str)
    out = pd.DataFrame({"gene_id": df["gene_id"]})
    out[f"{prefix}_has_homolog"] = pd.to_numeric(df["has_homolog"], errors="coerce").fillna(0).astype("float32")
    out[f"{prefix}_functional_redundancy"] = pd.to_numeric(df["functional_redundancy"], errors="coerce").fillna(0).astype("float32")
    return out.drop_duplicates("gene_id", keep="first").set_index("gene_id")


def coalesce(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    found = [c for c in columns if c in frame.columns]
    if not found:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for c in found:
        out = out.combine_first(pd.to_numeric(frame[c], errors="coerce"))
    return out


def source_feature(frame: pd.DataFrame, feature: str, source_col: str = "source") -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    supp = f"supplemental_noncoding__{feature}"
    fallback = f"fallback_dna_context__{feature}"
    if supp in frame.columns:
        mask = frame[source_col].astype(str).str.contains("supplemental_noncoding", na=False)
        out.loc[mask] = pd.to_numeric(frame.loc[mask, supp], errors="coerce")
    if fallback in frame.columns:
        mask = frame[source_col].astype(str).str.contains("fallback_dna_context", na=False)
        out.loc[mask] = pd.to_numeric(frame.loc[mask, fallback], errors="coerce")
    if feature in frame.columns:
        out = out.combine_first(pd.to_numeric(frame[feature], errors="coerce"))
    if supp in frame.columns:
        out = out.combine_first(pd.to_numeric(frame[supp], errors="coerce"))
    if fallback in frame.columns:
        out = out.combine_first(pd.to_numeric(frame[fallback], errors="coerce"))
    return out


def load_parquet_hits(path: Path, keys: set[str], columns: list[str]) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    cols = [c for c in columns if c in dataset.schema.names]
    need_cols = list(dict.fromkeys([c for c in ["snp_id", "Chr", "Pos", "Ref", "Alt"] + cols if c in dataset.schema.names]))
    table = dataset.to_table(columns=need_cols)
    df = table.to_pandas()
    df["variant_key"] = variant_key_from_cols(df)
    hit = df[df["variant_key"].isin(keys)].copy()
    return hit.drop_duplicates("variant_key", keep="first")


def feature_missingness(frame: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
    rows = []
    for c in features:
        if c not in frame.columns:
            rows.append({"scope": label, "feature": c, "missing_before_fold_impute": len(frame), "present": False})
        else:
            rows.append(
                {
                    "scope": label,
                    "feature": c,
                    "missing_before_fold_impute": int(pd.to_numeric(frame[c], errors="coerce").isna().sum()),
                    "present": True,
                }
            )
    return pd.DataFrame(rows)


def score_by_loco_fold(frame: pd.DataFrame, model_root: Path, score_col: str) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    feature_set, features = read_model_config(model_root)
    cfg = json.loads((model_root / "config.json").read_text())
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{model_root}: missing feature columns before scoring: {missing}")

    scores = pd.Series(np.nan, index=frame.index, dtype="float64")
    fill_rows: list[dict] = []
    used_rows: list[pd.DataFrame] = []
    for fold, idx in frame.groupby("fold", sort=True).groups.items():
        fold = int(fold)
        fold_dir = model_root / "folds" / f"fold{fold:02d}"
        model_path = fold_dir / f"xgboost_{feature_set}.json"
        imp_path = fold_dir / "train_fold_median_imputation.tsv"
        feature_path = fold_dir / "features.json"
        fold_features = json.loads(feature_path.read_text())
        if fold_features != features:
            raise RuntimeError(f"Feature mismatch in {fold_dir}")

        x = frame.loc[idx, features].copy()
        for c in features:
            x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
        before = x.isna().sum().astype(int).to_dict()

        imp = pd.read_csv(imp_path, sep="\t").set_index("feature")
        for c in features:
            fill = float(imp.loc[c, "fill_value"])
            policy = str(imp.loc[c, "policy"])
            x[c] = x[c].fillna(fill).astype("float32")
            fill_rows.append(
                {
                    "model_root": str(model_root),
                    "fold": fold,
                    "feature": c,
                    "policy": policy,
                    "fill_value": fill,
                    "n_rows": int(len(idx)),
                    "missing_before": int(before.get(c, 0)),
                    "missing_after": int(x[c].isna().sum()),
                }
            )

        booster = xgb.Booster()
        booster.load_model(str(model_path))
        dmat = xgb.DMatrix(x, feature_names=features)
        raw_spec = raw_base_margin_spec(cfg, fold)
        if raw_spec is not None:
            base_margin = compute_raw_base_margin(frame.loc[idx], raw_spec)

            
            dmat.set_base_margin(base_margin)
            fill_rows.append(
                {
                    "model_root": str(model_root),
                    "fold": fold,
                    "feature": "__raw_base_margin__",
                    "policy": "passed_to_xgboost_dmatrix",
                    "fill_value": float("nan"),
                    "n_rows": int(len(idx)),
                    "missing_before": 0,
                    "missing_after": 0,
                }
            )
        best_iter = booster.attributes().get("best_iteration")
        if best_iter is not None and best_iter != "":
            pred = booster.predict(dmat, iteration_range=(0, int(best_iter) + 1))
        else:
            pred = booster.predict(dmat)
        scores.loc[idx] = pred

        used = x.copy()
        used.insert(0, "_row_index", idx)
        used.insert(1, "fold", fold)
        if raw_spec is not None:
            used.insert(2, "__raw_base_margin__", base_margin)
        used_rows.append(used)

    if scores.isna().any():
        raise RuntimeError(f"{score_col}: some rows were not scored")
    used_features = pd.concat(used_rows, ignore_index=True) if used_rows else pd.DataFrame()
    return scores.rename(score_col), pd.DataFrame(fill_rows), used_features


def summarize_recall(frame: pd.DataFrame, score_col: str, hit_col: str, groups: list[str] | None = None) -> pd.DataFrame:
    rows = []

    def one(scope: str, sub: pd.DataFrame) -> None:
        n = int(len(sub))
        hit = int(sub[hit_col].sum())
        rows.append({"scope": scope, "n": n, "hits": hit, "recall": float(hit / n) if n else float("nan")})

    one("all_rows", frame)
    if groups:
        for g in groups:
            for val, sub in frame.groupby(g, dropna=False, sort=True):
                one(f"{g}={val}", sub)
    out = pd.DataFrame(rows)
    out["score_col"] = score_col
    out["hit_col"] = hit_col
    return out


def build_external_score_sanity(
    frame: pd.DataFrame, score_col: str, model_root: Path, used_features: pd.DataFrame
) -> dict:
    """Record score-scale and ranking checks for every external validation run."""
    cfg = json.loads((model_root / "config.json").read_text())
    rawbase_enabled = raw_base_margin_spec(cfg, int(frame["fold"].iloc[0])) is not None
    applied = "__raw_base_margin__" in used_features.columns
    if rawbase_enabled and not applied:
        raise RuntimeError("SANITY CHECK FAILED: raw-base model was scored without base_margin")

    score = pd.to_numeric(frame[score_col], errors="coerce")
    if not np.isfinite(score).all() or not score.between(0.0, 1.0).all():
        raise RuntimeError("SANITY CHECK FAILED: final probabilities are non-finite or outside [0, 1]")
    audit: dict[str, object] = {
        "model_root": public_path(model_root),
        "score_col": score_col,
        "raw_base_margin_enabled": rawbase_enabled,
        "base_margin_passed_to_dmatrix": applied,
        "external_n": int(len(frame)),
        "external_prob_min": float(score.min()),
        "external_prob_median": float(score.median()),
        "external_prob_max": float(score.max()),
    }
    if applied:
        base = pd.to_numeric(used_features["__raw_base_margin__"], errors="coerce")
        audit.update(
            {
                "base_margin_min": float(base.min()),
                "base_margin_median": float(base.median()),
                "base_margin_max": float(base.max()),
                "base_margin_nonzero_fraction": float((base != 0).mean()),
            }
        )


    
    oof_path = model_root / "all_oof_predictions.parquet"
    if oof_path.exists():
        oof = pd.read_parquet(oof_path, columns=["xgb_prob"])["xgb_prob"]
        audit.update(
            {
                "oof_prob_min": float(oof.min()),
                "oof_prob_median": float(oof.median()),
                "oof_prob_max": float(oof.max()),
                "external_minus_oof_median": float(score.median() - oof.median()),
            }
        )


    

    if MAIZE_BLEND91_3MD.exists() and "variant_key" in frame.columns:
        legacy = pd.read_parquet(MAIZE_BLEND91_3MD, columns=["variant_key", "label", "blend91_prob"])
        legacy = legacy[legacy["label"].astype(int).eq(1)].drop_duplicates("variant_key")
        paired = frame[["variant_key", score_col]].merge(legacy[["variant_key", "blend91_prob"]], on="variant_key", how="inner")
        if len(paired) >= 2:
            audit["legacy_blend91_paired_n"] = int(len(paired))
            audit["legacy_blend91_spearman"] = float(paired[[score_col, "blend91_prob"]].corr(method="spearman").iloc[0, 1])
            for k in (5, 10, 20):
                left = set(paired.nlargest(k, score_col)["variant_key"])
                right = set(paired.nlargest(k, "blend91_prob")["variant_key"])
                audit[f"legacy_blend91_top{k}_overlap"] = int(len(left & right))
    return audit


def run_maize_3md(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_set, features = read_model_config(MAIZE_MODEL)
    df = pd.read_parquet(MAIZE_3MD_PRED)
    df = df[df["label"].astype(int) == 1].copy()
    df["variant_key"] = df["variant_key"].astype(str)
    df["fold"] = df["Chr"].map(chr_to_fold).astype(int)

    if "prot_safe_abs_delta_log_odds" not in df.columns:
        df["prot_safe_abs_delta_log_odds"] = pd.to_numeric(df["prot_safe_delta_log_odds"], errors="coerce").abs()
    if "prot_safe_alt_absent_flag" not in df.columns:
        df["prot_safe_alt_absent_flag"] = (pd.to_numeric(df["prot_safe_alt_aa_freq"], errors="coerce").fillna(0) <= 0).astype("float32")

    red = read_redundancy(MAIZE_RED, "maize")
    gene = df.get("gene_norm", df.get("gene_id")).astype(str)
    df["maize_has_homolog"] = gene.map(red["maize_has_homolog"]).fillna(0).astype("float32").to_numpy()
    df["maize_functional_redundancy"] = gene.map(red["maize_functional_redundancy"]).fillna(0).astype("float32").to_numpy()

    miss = feature_missingness(df, features, "maize_3md_positive_45")
    score, fill_audit, used = score_by_loco_fold(df, MAIZE_MODEL, "strict_v3_maize_coding_prob")
    df[score.name] = score
    df["strict_v3_hit_0_5"] = df[score.name] >= 0.5
    df["raw_esm_hit_paper_threshold"] = pd.to_numeric(df["S_esm"], errors="coerce") >= 7.5
    df["raw_dna_hit_paper_threshold"] = pd.to_numeric(df["S_dna"], errors="coerce") >= 6.0
    df["raw_any_hit_paper_threshold"] = df["raw_esm_hit_paper_threshold"] | df["raw_dna_hit_paper_threshold"]

    recall = summarize_recall(df, score.name, "strict_v3_hit_0_5", ["fold"])
    raw_rows = []
    for col in ["raw_esm_hit_paper_threshold", "raw_dna_hit_paper_threshold", "raw_any_hit_paper_threshold"]:
        n = int(len(df))
        h = int(df[col].sum())
        raw_rows.append({"method": col, "n": n, "hits": h, "recall": float(h / n)})
    raw_recall = pd.DataFrame(raw_rows)

    out_cols = [
        "variant_key",
        "snp_id",
        "Chr",
        "Pos",
        "Ref",
        "Alt",
        "gene_norm",
        "ref_aa",
        "alt_aa",
        "aa_pos",
        score.name,
        "strict_v3_hit_0_5",
        "S_esm",
        "raw_esm_hit_paper_threshold",
        "S_dna",
        "raw_dna_hit_paper_threshold",
        "raw_any_hit_paper_threshold",
    ] + features
    out_cols = list(dict.fromkeys([c for c in out_cols if c in df.columns]))
    df[out_cols].to_csv(out_dir / "maize_3md_positive45_strict_v3_scores.tsv", sep="\t", index=False)
    recall.to_csv(out_dir / "maize_3md_positive45_strict_v3_recall.tsv", sep="\t", index=False)
    raw_recall.to_csv(out_dir / "maize_3md_positive45_raw_lm_recall.tsv", sep="\t", index=False)
    miss.to_csv(out_dir / "maize_3md_positive45_feature_missingness_before_impute.tsv", sep="\t", index=False)
    fill_audit.to_csv(out_dir / "maize_3md_positive45_fold_fill_audit.tsv", sep="\t", index=False)
    used.to_parquet(out_dir / "maize_3md_positive45_model_features_after_impute.parquet", index=False)
    sanity = build_external_score_sanity(df, score.name, MAIZE_MODEL, used)
    (out_dir / "maize_3md_positive45_score_sanity.json").write_text(json.dumps(sanity, indent=2) + "\n")
    manifest = {
        "validation": "maize_3md_positive45",
        "model_root": public_path(MAIZE_MODEL),
        "feature_set": feature_set,
        "n_rows": int(len(df)),
        "strict_v3_hits_at_0_5": int(df["strict_v3_hit_0_5"].sum()),
        "strict_v3_recall_at_0_5": float(df["strict_v3_hit_0_5"].mean()),
        "raw_thresholds": {"S_esm": 7.5, "S_dna": 6.0},
        "raw_any_hits": int(df["raw_any_hit_paper_threshold"].sum()),
        "raw_any_recall": float(df["raw_any_hit_paper_threshold"].mean()),
        "score_sanity_path": public_path(out_dir / "maize_3md_positive45_score_sanity.json"),
    }
    (out_dir / "maize_3md_positive45_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def make_rice_completed_with_key() -> pd.DataFrame:
    comp = pd.read_parquet(RICE_COMPLETED)
    comp["variant_key"] = variant_key_from_cols(comp)
    comp = comp.sort_values(["variant_key", "atlas_row_id", "atlas_alt_index"], kind="mergesort")
    return comp.drop_duplicates("variant_key", keep="first")


def attach_rice_original_coding_features(frame: pd.DataFrame) -> pd.DataFrame:
    coding_keys = set(frame.loc[frame["model_space"].eq("coding"), "variant_key"].astype(str))
    if not coding_keys:
        return frame

    dna = load_parquet_hits(
        RICE_DNA_CODING,
        coding_keys,
        ["dna_cons__phastCons", "dna_cons__phyloP", "dna_cons_missing"],
    )
    prot_cols = [
        "prot_safe_ref_aa_freq",
        "prot_safe_alt_aa_freq",
        "prot_safe_entropy",
        "prot_safe_delta_log_odds",
        "prot_safe_conservation_penalty",
        "prot_safe_abs_delta_log_odds",
        "prot_safe_alt_absent_flag",
        "prot_safe_missing",
        "prot_safe_protein_id",
    ]
    prot = load_parquet_hits(RICE_PROT_CODING, coding_keys, prot_cols)

    out = frame.merge(
        dna[["variant_key", "dna_cons__phastCons", "dna_cons__phyloP", "dna_cons_missing"]].rename(
            columns={
                "dna_cons__phastCons": "orig_coding_dna_cons__phastCons",
                "dna_cons__phyloP": "orig_coding_dna_cons__phyloP",
                "dna_cons_missing": "orig_coding_dna_cons_missing",
            }
        ),
        on="variant_key",
        how="left",
        validate="m:1",
    )
    keep = ["variant_key"] + [c for c in prot_cols if c in prot.columns]
    out = out.merge(
        prot[keep].rename(columns={c: f"orig_coding_{c}" for c in keep if c != "variant_key"}),
        on="variant_key",
        how="left",
        validate="m:1",
    )
    return out


def run_rice_91(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    coding_feature_set, coding_features = read_model_config(RICE_CODING_MODEL)
    noncoding_feature_set, noncoding_features = read_model_config(RICE_NONCODING_MODEL)

    sel = pd.read_csv(RICE_91, sep="\t")
    sel.insert(0, "validation_row_id", np.arange(len(sel), dtype=int))
    comp = make_rice_completed_with_key()
    df = sel.merge(comp, on="variant_key", how="left", suffixes=("", "_completed"), validate="m:1")
    if df["Chr"].isna().any():
        missing = df.loc[df["Chr"].isna(), "variant_key"].tolist()
        raise RuntimeError(f"Missing completed score rows for rice 91 validation keys: {missing}")
    df["fold"] = df["Chr"].map(chr_to_fold).astype(int)
    df["snp_id_strict"] = snp_id_from_cols(df)
    df = attach_rice_original_coding_features(df)

    red = read_redundancy(RICE_RED, "rice")
    gene = (
        df["gene"].astype(str)
        .where(df["gene"].notna(), df.get("paralog_gene_id", pd.Series(index=df.index, dtype=object)))
        .astype(str)
    )
    df["rice_has_homolog"] = gene.map(red["rice_has_homolog"]).fillna(0).astype("float32").to_numpy()
    df["rice_functional_redundancy"] = gene.map(red["rice_functional_redundancy"]).fillna(0).astype("float32").to_numpy()

    df["strict_feature_S_dna"] = coalesce(
        df,
        [
            "S_dna",
            "S_dna_completed",
            "S_dna_extra_coding_lm",
            "S_dna_coding_lm",
            "S_dna_noncoding_lm",
            "S_dna_extra_noncoding_lm",
            "supplemental_noncoding__S_dna",
            "fallback_dna_context__S_dna",
        ],
    )
    df["strict_feature_S_esm"] = coalesce(
        df,
        ["S_esm", "S_esm_completed", "S_esm_extra_coding_lm", "S_esm_coding_lm", "S_esm_sampled_coding_lm"],
    )

    for feature in noncoding_features:
        if feature in {"S_dna", "rice_has_homolog", "rice_functional_redundancy"}:
            continue
        df[f"strict_feature_noncoding__{feature}"] = source_feature(df, feature)

    coding_mask = df["model_space"].eq("coding")
    noncoding_mask = df["model_space"].eq("noncoding")

    score_frame = df[
        [
            "validation_row_id",
            "variant_key",
            "model_space",
            "source",
            "fold",
            "strict_feature_S_dna",
            "strict_feature_S_esm",
            "rice_has_homolog",
            "rice_functional_redundancy",
        ]
    ].copy()
    score_frame["S_dna"] = score_frame["strict_feature_S_dna"]
    score_frame["S_esm"] = score_frame["strict_feature_S_esm"]


    
    score_frame["dna_cons__phastCons"] = np.nan
    score_frame["dna_cons__phyloP"] = np.nan
    score_frame.loc[coding_mask, "dna_cons__phastCons"] = pd.to_numeric(
        df.loc[coding_mask, "orig_coding_dna_cons__phastCons"], errors="coerce"
    )
    score_frame.loc[coding_mask, "dna_cons__phyloP"] = pd.to_numeric(
        df.loc[coding_mask, "orig_coding_dna_cons__phyloP"], errors="coerce"
    )
    for f in [
        "prot_safe_ref_aa_freq",
        "prot_safe_alt_aa_freq",
        "prot_safe_entropy",
        "prot_safe_delta_log_odds",
        "prot_safe_conservation_penalty",
        "prot_safe_abs_delta_log_odds",
        "prot_safe_alt_absent_flag",
    ]:
        score_frame[f] = np.nan
        source_col = f"orig_coding_{f}"
        if source_col in df.columns:
            score_frame.loc[coding_mask, f] = pd.to_numeric(df.loc[coding_mask, source_col], errors="coerce")


    
    score_frame.loc[noncoding_mask, "dna_cons__phastCons"] = df.loc[
        noncoding_mask, "strict_feature_noncoding__dna_cons__phastCons"
    ]
    score_frame.loc[noncoding_mask, "dna_cons__phyloP"] = df.loc[
        noncoding_mask, "strict_feature_noncoding__dna_cons__phyloP"
    ]
    for f in noncoding_features:
        if f in {"S_dna", "dna_cons__phastCons", "dna_cons__phyloP", "rice_has_homolog", "rice_functional_redundancy"}:
            continue
        score_frame[f] = np.nan
        score_frame.loc[noncoding_mask, f] = df.loc[noncoding_mask, f"strict_feature_noncoding__{f}"]

    coding_missing = feature_missingness(score_frame.loc[coding_mask], coding_features, "rice_91_coding")
    noncoding_missing = feature_missingness(score_frame.loc[noncoding_mask], noncoding_features, "rice_91_noncoding")

    coding_scores, coding_fill, coding_used = score_by_loco_fold(
        score_frame.loc[coding_mask].copy(), RICE_CODING_MODEL, "strict_v3_rice_coding_prob"
    )
    noncoding_scores, noncoding_fill, noncoding_used = score_by_loco_fold(
        score_frame.loc[noncoding_mask].copy(), RICE_NONCODING_MODEL, "strict_v3_rice_noncoding_prob"
    )
    df["strict_v3_prob"] = np.nan
    df.loc[coding_scores.index, "strict_v3_prob"] = coding_scores
    df.loc[noncoding_scores.index, "strict_v3_prob"] = noncoding_scores
    df["strict_v3_hit_0_5"] = df["strict_v3_prob"] >= 0.5
    df["raw_any_hit_paper_threshold"] = df["raw_lm_hit"].astype(bool)
    df["raw_any_hit_recomputed_esm7_5_or_dna6"] = (
        pd.to_numeric(df["S_esm"], errors="coerce").ge(7.5)
        | pd.to_numeric(df["S_dna"], errors="coerce").ge(6.0)
    )

    recall = summarize_recall(df, "strict_v3_prob", "strict_v3_hit_0_5", ["model_space", "source", "gold_tier"])
    raw_rows = []
    for col in ["raw_any_hit_paper_threshold", "raw_any_hit_recomputed_esm7_5_or_dna6"]:
        n = int(len(df))
        h = int(df[col].sum())
        raw_rows.append({"method": col, "n": n, "hits": h, "recall": float(h / n)})
    raw_recall = pd.DataFrame(raw_rows)

    out_cols = [
        "validation_row_id",
        "variant_key",
        "model_space",
        "source",
        "gene",
        "trait",
        "gold_tier",
        "broad_consequence_class",
        "consequence",
        "Chr",
        "Pos",
        "Ref",
        "Alt",
        "strict_v3_prob",
        "strict_v3_hit_0_5",
        "xgb_prob_completed",
        "xgb_hit_0_5",
        "S_dna",
        "S_esm",
        "raw_any_hit_paper_threshold",
        "raw_any_hit_recomputed_esm7_5_or_dna6",
    ]
    for f in sorted(set(coding_features + noncoding_features)):
        if f in score_frame.columns:
            df[f"strict_input__{f}"] = score_frame[f].to_numpy()
            out_cols.append(f"strict_input__{f}")
    df[out_cols].to_csv(out_dir / "rice_gwas_primary91_strict_v3_scores.tsv", sep="\t", index=False)
    recall.to_csv(out_dir / "rice_gwas_primary91_strict_v3_recall.tsv", sep="\t", index=False)
    raw_recall.to_csv(out_dir / "rice_gwas_primary91_raw_lm_recall.tsv", sep="\t", index=False)
    pd.concat([coding_missing, noncoding_missing], ignore_index=True).to_csv(
        out_dir / "rice_gwas_primary91_feature_missingness_before_impute.tsv", sep="\t", index=False
    )
    pd.concat([coding_fill, noncoding_fill], ignore_index=True).to_csv(
        out_dir / "rice_gwas_primary91_fold_fill_audit.tsv", sep="\t", index=False
    )
    coding_used.to_parquet(out_dir / "rice_gwas_primary91_coding_features_after_impute.parquet", index=False)
    noncoding_used.to_parquet(out_dir / "rice_gwas_primary91_noncoding_features_after_impute.parquet", index=False)
    coding_sanity_frame = score_frame.loc[coding_mask].copy()
    coding_sanity_frame[coding_scores.name] = coding_scores
    noncoding_sanity_frame = score_frame.loc[noncoding_mask].copy()
    noncoding_sanity_frame[noncoding_scores.name] = noncoding_scores
    score_sanity = {
        "coding": build_external_score_sanity(coding_sanity_frame, coding_scores.name, RICE_CODING_MODEL, coding_used),
        "noncoding": build_external_score_sanity(noncoding_sanity_frame, noncoding_scores.name, RICE_NONCODING_MODEL, noncoding_used),
    }
    (out_dir / "rice_gwas_primary91_score_sanity.json").write_text(json.dumps(score_sanity, indent=2) + "\n")
    manifest = {
        "validation": "rice_gwas_primary91_no_oof_overlap",
        "coding_model_root": public_path(RICE_CODING_MODEL),
        "noncoding_model_root": public_path(RICE_NONCODING_MODEL),
        "coding_feature_set": coding_feature_set,
        "noncoding_feature_set": noncoding_feature_set,
        "n_rows": int(len(df)),
        "n_unique_variant_keys": int(df["variant_key"].nunique()),
        "strict_v3_hits_at_0_5": int(df["strict_v3_hit_0_5"].sum()),
        "strict_v3_recall_at_0_5": float(df["strict_v3_hit_0_5"].mean()),
        "raw_paper_threshold_hits": int(df["raw_any_hit_paper_threshold"].sum()),
        "raw_paper_threshold_recall": float(df["raw_any_hit_paper_threshold"].mean()),
        "score_sanity_path": public_path(out_dir / "rice_gwas_primary91_score_sanity.json"),
    }
    (out_dir / "rice_gwas_primary91_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict v3 external validations for maize 3DM and rice GWAS Atlas sets.")
    ap.add_argument("command", choices=["maize-3md", "rice-gwas91", "all"])
    ap.add_argument("--out-dir", default=str(ROOT / "validation_results/strict_v3_20260709"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.command in {"maize-3md", "all"}:
        run_maize_3md(out_dir / "maize_3md")
    if args.command in {"rice-gwas91", "all"}:
        run_rice_91(out_dir / "rice_gwas91")


if __name__ == "__main__":
    main()
