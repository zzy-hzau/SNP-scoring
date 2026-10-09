#!/usr/bin/env python3
"""Attach the registered non-model columns required by weighted O/E."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def rank_first_qcut(values: pd.Series, bins: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return (
        pd.qcut(
            numeric.rank(method="first"),
            q=bins,
            labels=False,
            duplicates="drop",
        )
        .astype("Int64")
        .fillna(0)
        .astype("int16")
    )


def add_shared_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "chr_num" not in result:
        chromosome = next(
            (result[name] for name in ("Chr", "chrom", "Chromosome") if name in result),
            None,
        )
        if chromosome is None:
            raise RuntimeError("O/E input lacks chr_num and chromosome fallback")
        extracted = chromosome.astype(str).str.extract(r"(\d+)", expand=False)
        if extracted.isna().any():
            raise RuntimeError("cannot derive every chr_num value")
        result["chr_num"] = extracted.astype("int16")
    if "subgenome_code" not in result:
        result["subgenome_code"] = 0
    if "Sg_code" not in result:
        result["Sg_code"] = result["subgenome_code"]
    if "Y_Label" not in result and "observed_in_vcf" in result:
        result["Y_Label"] = result["observed_in_vcf"]
    if "observed_in_vcf" not in result and "Y_Label" in result:
        result["observed_in_vcf"] = result["Y_Label"]
    return result


def attach_rank_bins(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = frame.copy()
    generated: dict[str, Any] = {}
    if "callable_bin" not in result:
        source = next(
            (name for name in ("callable_n", "callable_rate", "call_rate") if name in result),
            None,
        )
        if source is None:
            result["callable_bin"] = np.int16(0)
            generated["callable_bin"] = {"source": None, "method": "constant", "value": 0}
        else:
            result["callable_bin"] = rank_first_qcut(result[source], 20)
            generated["callable_bin"] = {
                "source": source,
                "method": "rank_first_qcut",
                "q": 20,
            }
    if "opportunity_bin" not in result:
        source = next((name for name in ("P_mut", "P_mut_global") if name in result), None)
        if source is None:
            result["opportunity_bin"] = np.int16(0)
            generated["opportunity_bin"] = {"source": None, "method": "constant", "value": 0}
        else:
            result["opportunity_bin"] = rank_first_qcut(result[source], 10)
            generated["opportunity_bin"] = {"source": source, "method": "rank_first_qcut", "q": 10}
    return result, generated


def attach_maize_lookup_bins(
    frame: pd.DataFrame, metadata_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text())
    result = frame.drop(
        columns=[name for name in ("P_mut", "opportunity_bin", "callable_bin") if name in frame]
    ).copy()
    required = ["Context", "Alt", "callable_n"]
    missing = [name for name in required if name not in result]
    if missing:
        raise RuntimeError(f"maize noncoding O/E binning lacks columns: {missing}")
    key = result["Context"].astype(str) + "_" + result["Alt"].astype(str)
    result["P_mut"] = pd.to_numeric(key.map(metadata["pmut_lookup"]), errors="coerce")
    opportunity = pd.to_numeric(
        key.map(metadata["opportunity_bin_lookup"]), errors="coerce"
    )
    missing_opp = opportunity.isna()
    if missing_opp.any():
        edges = np.asarray(metadata["opportunity_bin_edges"], dtype=np.float64)
        values = result.loc[missing_opp, "P_mut"].to_numpy(np.float64)
        computed = (np.searchsorted(edges, values, side="right") - 1).clip(
            -1, len(edges) - 2
        )
        opportunity.loc[missing_opp] = computed.astype(np.int16)
    result["opportunity_bin"] = opportunity.fillna(-1).astype("int16")
    call_edges = np.asarray(metadata["callable_bin_edges"], dtype=np.float64)
    callable_values = pd.to_numeric(result["callable_n"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(callable_values).all():
        raise RuntimeError("callable_n contains non-finite values during maize binning")
    result["callable_bin"] = (
        np.searchsorted(call_edges, callable_values, side="right") - 1
    ).clip(0, len(call_edges) - 2).astype(np.int16)
    return result, {
        "metadata": str(metadata_path.resolve()),
        "method": "Context_Alt_lookup_with_saved_edges",
        "opportunity_bins": sorted(int(v) for v in result["opportunity_bin"].unique()),
        "callable_bins": sorted(int(v) for v in result["callable_bin"].unique()),
    }


def prepare_frame(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    model_inputs: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the registered O/E operational-column contract to one table."""
    result = add_shared_columns(frame)
    oe_inputs = spec.get("oe_inputs") or {}
    policy = str(oe_inputs.get("bin_policy", "require_existing"))
    if policy == "rank_if_missing":
        result, bin_audit = attach_rank_bins(result)
    elif policy == "maize_lookup_edges":
        metadata = model_inputs.get("oe_bin_metadata")
        if not metadata:
            raise KeyError("maize_lookup_edges requires model_inputs.*.*.oe_bin_metadata")
        result, bin_audit = attach_maize_lookup_bins(result, Path(str(metadata)))
    elif policy == "require_existing":
        bin_audit = {"method": "preserve_existing"}
    else:
        raise ValueError(f"unsupported O/E bin policy: {policy}")

    required = [
        value.strip()
        for value in str(oe_inputs["strata_columns"]).split(",")
        if value.strip()
    ]
    if oe_inputs.get("region_column"):
        required.append(str(oe_inputs["region_column"]))
    missing = [name for name in required if name not in result]
    if missing:
        raise RuntimeError(f"O/E operational preparation lacks columns: {missing}")
    return result, {
        "bin_policy": policy,
        "bin_audit": bin_audit,
        "required_operational_columns": required,
    }


def run(
    input_path: Path,
    output_path: Path,
    spec: dict[str, Any],
    model_inputs: dict[str, Any],
) -> dict[str, Any]:
    frame, audit = prepare_frame(pd.read_parquet(input_path), spec, model_inputs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), output_path, compression="zstd"
    )
    report = {
        "schema_version": 1,
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "rows": int(len(frame)),
        **audit,
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-registry", default="configs/plantvimpact/model_registry.yaml")
    parser.add_argument("--paths", required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--mode", required=True, choices=("coding", "noncoding"))
    args = parser.parse_args()
    registry = yaml.safe_load(Path(args.model_registry).read_text())
    paths = yaml.safe_load(Path(args.paths).read_text())
    report = run(
        args.input,
        args.output,
        registry["models"][args.species][args.mode],
        paths["model_inputs"][args.species][args.mode],
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
