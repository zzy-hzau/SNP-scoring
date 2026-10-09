#!/usr/bin/env python3
"Compose selected score."
from __future__ import annotations
from phenofuncrank_repro.common.paths import project_root, package_dir

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

SCRIPTS_ROOT = package_dir('plantvimpact')
from phenofuncrank_repro.plantvimpact.common.config_utils import model_spec


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_value = np.exp(values[~positive])
    out[~positive] = exp_value / (1.0 + exp_value)
    return out


def selected_values(
    frame: pd.DataFrame, species: str, mode: str, spec: dict[str, Any],
    margin_col: str, probability_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    if species == "maize" and mode == "coding":
        raw_cfg = spec["raw_base"]
        selected = spec["selected_score"]
        raw91 = (
            float(raw_cfg["dna_weight"]) * pd.to_numeric(frame["S_dna"], errors="coerce").to_numpy(float)
            + (1.0 - float(raw_cfg["dna_weight"]))
            * pd.to_numeric(frame["S_esm"], errors="coerce").to_numpy(float)
        )
        training_base = float(raw_cfg["coefficient"]) * (
            (raw91 - float(raw_cfg["training_mean"])) / float(raw_cfg["training_std"])
        )
        residual = pd.to_numeric(frame[margin_col], errors="coerce").to_numpy(float) - training_base
        background_base = float(raw_cfg["coefficient"]) * (
            (raw91 - float(selected["background_mean"])) / float(selected["background_std"])
        )
        margin = background_base + float(selected["residual_shrinkage"]) * residual
        return margin, sigmoid(margin)
    if species == "maize" and mode == "noncoding":
        raw_cfg = spec["raw_base"]
        raw = pd.to_numeric(frame["S_dna"], errors="coerce").to_numpy(float)
        raw_margin = float(raw_cfg["coefficient"]) * (
            (raw - float(raw_cfg["training_mean"])) / float(raw_cfg["training_std"])
        )
        full_margin = pd.to_numeric(frame[margin_col], errors="coerce").to_numpy(float)
        margin = raw_margin + float(spec["selected_score"]["alpha"]) * (full_margin - raw_margin)
        return margin, sigmoid(margin)
    margin = pd.to_numeric(frame[margin_col], errors="coerce").to_numpy(float)
    probability = pd.to_numeric(frame[probability_col], errors="coerce").to_numpy(float)
    return margin, probability


def required_columns(species: str, mode: str, margin_col: str, probability_col: str) -> list[str]:
    columns = [margin_col] if species == "maize" else [margin_col, probability_col]
    if species == "maize":
        columns.append("S_dna")
        if mode == "coding":
            columns.append("S_esm")
    return list(dict.fromkeys(columns))


def run(
    input_path: Path, output_path: Path, registry: dict[str, Any], species: str,
    mode: str, margin_col: str, probability_col: str, batch_size: int,
) -> dict[str, Any]:
    spec = model_spec(registry, species, mode)
    pf = pq.ParquetFile(input_path)
    needed = required_columns(species, mode, margin_col, probability_col)
    missing = [name for name in needed if name not in pf.schema.names]
    if missing:
        raise ValueError(f"input lacks columns needed for selected score: {missing}")

    selected = spec["selected_score"]
    selected_column = str(selected["column"])
    value_type = str(selected.get("value_type", "probability"))
    if value_type not in {"margin", "probability"}:
        raise ValueError(f"unsupported selected score value_type: {value_type}")
    selected_margin = selected_column if value_type == "margin" else selected.get("margin_column")
    selected_probability = str(
        selected.get("probability_column")
        or (selected_column if value_type == "probability" else "")
    )
    if not selected_probability:
        raise ValueError(f"{species}/{mode} margin selection lacks probability_column")
    writer = None
    offset = 0
    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            frame = batch.to_pandas()
            margin, probability = selected_values(frame, species, mode, spec, margin_col, probability_col)
            n_rows = len(frame)
            if not np.isfinite(margin).all() or not np.isfinite(probability).all():
                raise ValueError("selected score contains non-finite values")
            if ((probability < 0.0) | (probability > 1.0)).any():
                raise ValueError("selected probability is outside [0, 1]")
            if species == "maize" and mode == "coding":
                dna_weight = float(spec["raw_base"]["dna_weight"])
                frame["raw91"] = (
                    dna_weight * pd.to_numeric(frame["S_dna"], errors="coerce")
                    + (1.0 - dna_weight) * pd.to_numeric(frame["S_esm"], errors="coerce")
                ).astype(np.float32)
            if selected_margin:
                frame[str(selected_margin)] = margin.astype(np.float32)
            frame[selected_probability] = probability.astype(np.float32)
            offset += n_rows
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if offset == 0:
        raise ValueError("cannot score an empty table")
    return {
        "species": species, "mode": mode, "input": str(input_path), "output": str(output_path),
        "rows": int(offset), "source_margin_column": margin_col,
        "source_probability_column": probability_col,
        "selected_score_column": selected_column,
        "selected_score_value_type": value_type,
        "selected_probability_column": selected_probability,
        "selected_margin_column": selected_margin,
        "selected_formula": spec["selected_score"].get("formula", "strict_probability"),
        "probability_transform": "sigmoid" if species == "maize" else "identity",
        "ranking": f"{selected_column}_descending",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-registry", default="configs/plantvimpact/model_registry.yaml")
    parser.add_argument("--species", required=True)
    parser.add_argument("--mode", required=True, choices=("coding", "noncoding"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--margin-col", default="strict_v3_margin")
    parser.add_argument("--probability-col", default="strict_v3_prob")
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    with open(args.model_registry, encoding="utf-8") as handle:
        registry = yaml.safe_load(handle) or {}
    report = run(
        args.input, args.output, registry, args.species, args.mode,
        args.margin_col, args.probability_col, args.batch_size,
    )
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
