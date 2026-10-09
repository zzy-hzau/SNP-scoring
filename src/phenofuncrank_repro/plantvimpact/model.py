"scoring of LOCO PlantVImpact models."
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb


def raw_base_margin(frame: pd.DataFrame, specification: dict | None):
    if not specification:
        return None
    dna = frame["S_dna"].to_numpy()
    weight = float(specification.get("dna_weight", 0.91))
    if specification.get("feature") == "S_dna" or "S_esm" not in frame:
        value = dna
    else:
        value = weight * dna + (1.0 - weight) * frame["S_esm"].to_numpy()
    return np.asarray(
        float(specification.get("coef", 1.0))
        * ((value - float(specification.get("mean", 0.0)))
           / float(specification.get("std", 1.0))),
        dtype=np.float32,
    )


class FrozenFoldPredictor:
    """Load the exact fold weights, training fill values and stopping iteration."""

    def __init__(self, model_root: str | Path, fold: int, threads: int = 4):
        root = Path(model_root)
        self.protocol = json.loads((root / "protocol.json").read_text())
        self.features = self.protocol["features"]
        folder = root / f"fold{fold:02d}"
        imputation = pd.read_csv(folder / "imputation.tsv", sep="\t")
        self.fill_values = dict(zip(imputation.feature, imputation.fill_value))
        completion = json.loads((folder / "COMPLETE.json").read_text())
        self.iteration_end = int(completion["iteration_end"])
        self.threads = threads
        self.booster = xgb.Booster()
        self.booster.load_model(folder / "model.json")
        self.booster.set_param({"device": "cpu", "nthread": threads})

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        absent = [name for name in self.features if name not in frame]
        if absent:
            raise ValueError("Required model-ready feature columns are absent: " + ", ".join(absent))
        values = (
            frame[self.features]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype("float32")
            .fillna(self.fill_values)
        )
        matrix = xgb.DMatrix(
            values,
            base_margin=raw_base_margin(values, self.protocol.get("raw_base_margin")),
            nthread=self.threads,
        )
        return self.booster.predict(matrix, iteration_range=(0, self.iteration_end))


class ExistingFoldPredictor:

    def __init__(self, model_root: str | Path, fold: int, threads: int = 4):
        root = Path(model_root)
        config = json.loads((root / "config.json").read_text())
        self.feature_set = config["main_feature_set"]
        self.features = config["feature_sets"][self.feature_set]
        folder = root / "folds" / f"fold{fold:02d}"
        recorded_features = json.loads((folder / "features.json").read_text())
        if recorded_features != self.features:
            raise ValueError("Fold features differ from model configuration")
        imputation = pd.read_csv(
            folder / "train_fold_median_imputation.tsv", sep="\t"
        )
        self.fill_values = dict(zip(imputation.feature, imputation.fill_value))
        self.threads = threads
        self.booster = xgb.Booster()
        self.booster.load_model(folder / f"xgboost_{self.feature_set}.json")
        self.booster.set_param({"device": "cpu", "nthread": threads})
        summary = json.loads((folder / "fold_summary.json").read_text())
        self.iteration_end = int(
            summary.get("xgb_best_iteration",
                        self.booster.num_boosted_rounds() - 1)
        ) + 1

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        absent = [name for name in self.features if name not in frame]
        if absent:
            raise ValueError("Required model-ready feature columns are absent: "
                             + ", ".join(absent))
        values = (
            frame[self.features]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype("float32")
            .fillna(self.fill_values)
        )
        return self.booster.predict(
            xgb.DMatrix(values, nthread=self.threads),
            iteration_range=(0, self.iteration_end),
        )
