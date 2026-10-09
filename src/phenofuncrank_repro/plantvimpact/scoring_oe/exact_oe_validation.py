#!/usr/bin/env python3
"Compute exhaustive O/E and compare it with local sampled O/E outputs."
from __future__ import annotations
from phenofuncrank_repro.common.paths import project_root, package_dir

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_ROOT = package_dir('plantvimpact')

from phenofuncrank_repro.plantvimpact.scoring_oe.weighted_stratified_oe import attach_sample_weights, build_strata_counts, parse_csv_list, parse_score_specs, parse_top_fracs, read_table, run_weighted_oe, write_tsv


def _read_sample_summary(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"sampled O/E summary is missing: {path}")
    frame = pd.read_csv(path, sep="\t")
    required = {"method", "top_fraction", "strat_OE"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"sampled summary {path} lacks columns: {sorted(missing)}")
    return frame[["method", "top_fraction", "strat_OE"]].copy()


def _compare_sampled(exact: pd.DataFrame, sampled_root: Path, out: Path) -> None:
    rows: list[dict[str, float | str]] = []
    for method in exact["method"].astype(str).unique():
        sampled = _read_sample_summary(
            sampled_root / method / "weighted_oe_summary.tsv"
        )
        exact_method = exact[exact["method"].astype(str) == method]
        for exact_row in exact_method.itertuples(index=False):
            matches = sampled[
                (sampled["method"].astype(str) == method)
                & (sampled["top_fraction"].astype(float) == float(exact_row.top_fraction))
            ]
            if matches.empty:
                raise ValueError(
                    f"sampled summary lacks {method}/{exact_row.top_fraction}: "
                    f"{sampled_root / method}"
                )
            sampled_value = float(matches.iloc[0]["strat_OE"])
            exact_value = float(exact_row.strat_OE)
            error = abs(exact_value - sampled_value)
            rows.append({
                "method": method,
                "top_fraction": float(exact_row.top_fraction),
                "exact_strat_OE": exact_value,
                "sampled_strat_OE": sampled_value,
                "abs_error": error,
                "max_abs_error": error,
            })
    write_tsv(pd.DataFrame(rows), out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-score-table", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sampled-root", required=True)
    parser.add_argument("--score-spec", action="append", required=True)
    parser.add_argument("--top-fracs", required=True)
    parser.add_argument("--observed-col", default="Y_Label")
    parser.add_argument("--id-col", default="snp_id")
    parser.add_argument("--strata-cols", required=True)
    parser.add_argument("--region-col", default="region_type")
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--no-within-region", action="store_true")
    args = parser.parse_args(argv)

    score_specs = parse_score_specs(args.score_spec)
    top_fracs = parse_top_fracs(args.top_fracs)
    strata_cols = parse_csv_list(args.strata_cols)
    region_col = args.region_col or None
    columns = list(strata_cols) + [args.observed_col, args.id_col]
    if region_col:
        columns.append(region_col)
    columns.extend(score_col for _, score_col, _ in score_specs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[exact O/E] building full-universe strata", file=sys.stderr)
    strata_counts = build_strata_counts(
        args.full_score_table,
        observed_col=args.observed_col,
        strata_cols=strata_cols,
        region_col=region_col,
        batch_size=args.batch_size,
    )
    write_tsv(strata_counts, out_dir / "exact_strata_counts.tsv.gz")
    full = read_table(args.full_score_table, columns)
    full, sampling_plan = attach_sample_weights(
        full,
        strata_counts,
        observed_col=args.observed_col,
        strata_cols=strata_cols,
        weight_col=None,
        require_all_observed=True,
    )
    write_tsv(sampling_plan, out_dir / "exact_sampling_plan.tsv.gz")
    summary, region_global, region_within = run_weighted_oe(
        full,
        strata_counts,
        score_specs=score_specs,
        top_fracs=top_fracs,
        region_col=region_col,
        within_region=not args.no_within_region,
    )
    write_tsv(summary, out_dir / "exact_exhaustive_oe_summary.tsv")
    if len(region_global):
        write_tsv(region_global, out_dir / "exact_region_oe_global_threshold.tsv")
    if len(region_within):
        write_tsv(region_within, out_dir / "exact_region_oe_within_region_threshold.tsv")
    _compare_sampled(
        summary,
        Path(args.sampled_root),
        out_dir / "exact_vs_sampled_global_comparison_summary.tsv",
    )
    manifest = {
        "script": "exact_oe_validation.py",
        "full_score_table": args.full_score_table,
        "sampled_root": args.sampled_root,
        "score_specs": [
            {"method": method, "score_col": col, "direction": direction}
            for method, col, direction in score_specs
        ],
        "top_fracs": top_fracs,
        "weights": "one for every exhaustive row",
        "outputs": [
            "exact_exhaustive_oe_summary.tsv",
            "exact_vs_sampled_global_comparison_summary.tsv",
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[exact O/E] wrote {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
