"""Run original M0-M3 matching and technical AUC/SMD on prepared fold inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from phenofuncrank_repro.plantvimpact.label_matching_methods import METHODS, normalize_technical, select_matched_backgrounds, balance_records, technical_diagnostic


def run_matching(config, output_dir: Path):
    fold, seed = int(config["fold"]), int(config.get("seed", 42))
    frames = {method: {} for method in METHODS}
    selection = {}
    balance = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        inputs = config["splits"][split]
        def prepare(path):
            return normalize_technical(
                pd.read_parquet(path), float(config["max_callable_n"]),
                config["p_base_eff_edges"], config["callable_n_edges"],
            )
        selected, info = select_matched_backgrounds(
            prepare(inputs["pool"]), prepare(inputs["model"]),
            seed=seed, fold=fold, split=split,
        )
        selection[split] = info
        for method, frame in selected.items():
            frames[method][split] = frame
            balance.extend(balance_records(frame, method, fold, split))
            destination = output_dir / method
            destination.mkdir(exist_ok=True)
            saved = frame.copy()
            for column in ["snp_id", "Chr", "Pos", "Ref", "Alt", "region_type", "region_subtype"]:
                if column in saved:
                    saved[column] = saved[column].astype("string")
            saved.to_parquet(destination / (split + ".parquet"), index=False)
    metrics = [technical_diagnostic(
        values["train"], values["test"], method=method, fold=fold
    ) for method, values in frames.items()]
    pd.DataFrame(metrics).to_csv(output_dir / "technical_auc.tsv", sep="\t", index=False)
    pd.DataFrame(balance).to_csv(output_dir / "balance_smd.tsv", sep="\t", index=False)
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return {"metrics": metrics, "balance": balance, "selection": selection}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_matching(json.loads(args.config.read_text()), args.output_dir)


if __name__ == "__main__":
    main()
