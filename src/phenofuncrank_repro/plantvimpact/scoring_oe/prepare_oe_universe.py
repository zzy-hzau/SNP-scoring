#!/usr/bin/env python3
"""Stream a registered full universe into the exact O/E stratum contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

try:
    from phenofuncrank_repro.plantvimpact.scoring_oe.prepare_oe_columns import prepare_frame
except ImportError:  
    from phenofuncrank_repro.plantvimpact.scoring_oe.prepare_oe_columns import prepare_frame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    input_path: Path,
    output_path: Path,
    spec: dict[str, Any],
    model_inputs: dict[str, Any],
    batch_size: int = 1_000_000,
) -> dict[str, Any]:
    """Prepare a potentially multi-file universe without loading it all at once."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    policy = str((spec.get("oe_inputs") or {}).get("bin_policy", "require_existing"))
    if policy != "maize_lookup_edges":
        raise ValueError(
            "streamed universe preparation is only registered for maize_lookup_edges; "
            "all other universes must already contain their production strata"
        )
    metadata_path = Path(str(model_inputs.get("oe_bin_metadata", "")))
    if not metadata_path.is_file():
        raise FileNotFoundError(f"O/E bin metadata does not exist: {metadata_path}")

    source = ds.dataset(str(input_path), format="parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    batches = 0
    last_audit: dict[str, Any] | None = None
    try:
        for batch in source.scanner(batch_size=batch_size, use_threads=True).to_batches():
            prepared, last_audit = prepare_frame(batch.to_pandas(), spec, model_inputs)
            table = pa.Table.from_pandas(prepared, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary_path, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(prepared)
            batches += 1
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    if writer is not None:
        writer.close()
    if writer is None:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"full universe has no rows: {input_path}")
    temporary_path.replace(output_path)
    report = {
        "schema_version": 1,
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "rows": int(rows),
        "batches": int(batches),
        "batch_size": int(batch_size),
        "metadata": str(metadata_path.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "operational_audit": last_audit,
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
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    args = parser.parse_args()
    registry = yaml.safe_load(Path(args.model_registry).read_text())
    paths = yaml.safe_load(Path(args.paths).read_text())
    report = run(
        args.input,
        args.output,
        registry["models"][args.species][args.mode],
        paths["model_inputs"][args.species][args.mode],
        args.batch_size,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
