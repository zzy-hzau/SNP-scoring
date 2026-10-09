#!/usr/bin/env python3

"Calculate a weighted, stratified observed/expected ratio."

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

NA_TOKEN = "__NA__"


def _is_parquet_path(path: str) -> bool:
    p = str(path)
    if os.path.isdir(p):
        return True
    return p.endswith(".parquet") or p.endswith(".pq")


def _parquet_sources(path: str):
    "Return a parquet file or an parquet-file list."
    if not os.path.isdir(path):
        return path
    files = sorted(
        str(p) for p in Path(path).rglob("*")
        if p.is_file() and p.suffix.lower() in {".parquet", ".pq"}
    )
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return files


def parse_csv_list(x: Optional[str]) -> List[str]:
    if x is None:
        return []
    return [v.strip() for v in str(x).split(",") if v.strip()]


def parse_top_fracs(x: str) -> List[float]:
    out = []
    for v in parse_csv_list(x):
        f = float(v)
        if not (0 < f < 1):
            raise ValueError(f"top fraction must be in (0,1), got {f}")
        out.append(f)
    return out


def parse_score_specs(specs: Sequence[str]) -> List[Tuple[str, str, str]]:
    out = []
    for s in specs:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError(f"score spec must be name:column:high|low, got {s}")
        name, col, direction = parts
        direction = direction.lower()
        if direction not in {"high", "low"}:
            raise ValueError(f"direction must be high or low, got {direction}")
        out.append((name, col, direction))
    if not out:
        raise ValueError("At least one --score-spec is required")
    return out


def table_columns(path: str) -> List[str]:
    if _is_parquet_path(path):
        dataset = ds.dataset(path, format="parquet")
        return list(dataset.schema.names)

    sep = "\t" if str(path).endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")) else ","
    return list(pd.read_csv(path, sep=sep, nrows=0).columns)


def iter_table_batches(path: str, columns: Sequence[str], batch_size: int = 1_000_000) -> Iterator[pd.DataFrame]:
    columns = list(dict.fromkeys([c for c in columns if c is not None]))
    if _is_parquet_path(path):
        dataset = ds.dataset(_parquet_sources(path), format="parquet")
        scanner = dataset.scanner(columns=columns, batch_size=batch_size)
        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            yield batch.to_pandas(self_destruct=True)
    else:
        sep = "\t" if str(path).endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")) else ","
        for chunk in pd.read_csv(path, sep=sep, usecols=columns, chunksize=batch_size):
            yield chunk


def read_table(path: str, columns: Sequence[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys([c for c in columns if c is not None]))
    if _is_parquet_path(path):
        return ds.dataset(_parquet_sources(path), format="parquet").to_table(columns=columns).to_pandas()
    sep = "\t" if str(path).endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")) else ","
    return pd.read_csv(path, sep=sep, usecols=columns)


def write_tsv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".gz"):
        df.to_csv(path, sep="\t", index=False, compression="gzip")
    else:
        df.to_csv(path, sep="\t", index=False)


def read_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def observed_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float) > 0
    x = s.astype(str).str.lower().fillna("0")
    return x.isin(["1", "true", "t", "yes", "y", "observed"])


def normalize_strata_values(df: pd.DataFrame, strata_cols: Sequence[str]) -> pd.DataFrame:
    for c in strata_cols:
        if c not in df.columns:
            raise KeyError(f"Missing stratum column: {c}")

        df[c] = df[c].astype("string").fillna(NA_TOKEN).astype(str)
    return df


def make_stratum_id_df(df: pd.DataFrame, strata_cols: Sequence[str]) -> pd.Series:
    if not strata_cols:
        return pd.Series(["ALL"] * len(df), index=df.index)
    return df[list(strata_cols)].astype(str).agg("||".join, axis=1)


def build_strata_counts(
    universe_path: str,
    observed_col: str,
    strata_cols: Sequence[str],
    region_col: Optional[str] = None,
    batch_size: int = 1_000_000,
) -> pd.DataFrame:
    cols = list(strata_cols) + [observed_col]
    if region_col and region_col not in cols:
        cols.append(region_col)

    acc_total: Dict[str, int] = defaultdict(int)
    acc_obs: Dict[str, int] = defaultdict(int)
    meta: Dict[str, Tuple[str, ...]] = {}
    t0 = time.time()
    n_seen = 0

    for i, df in enumerate(iter_table_batches(universe_path, cols, batch_size=batch_size), 1):
        df = normalize_strata_values(df, strata_cols)
        sid = make_stratum_id_df(df, strata_cols)
        obs = observed_bool(df[observed_col]).to_numpy(np.int8)
        tmp = pd.DataFrame({"stratum_id": sid, "obs": obs})
        g = tmp.groupby("stratum_id", sort=False)["obs"].agg(["count", "sum"]).reset_index()
        for row in g.itertuples(index=False):
            st = str(row.stratum_id)
            acc_total[st] += int(row.count)
            acc_obs[st] += int(row.sum)

        firsts = df.assign(stratum_id=sid).drop_duplicates("stratum_id")[["stratum_id"] + list(strata_cols)]
        for row in firsts.itertuples(index=False):
            st = str(row[0])
            if st not in meta:
                meta[st] = tuple(str(x) for x in row[1:])
        n_seen += len(df)
        if i % 20 == 0:
            print(f"[strata] batches={i} rows={n_seen:,} strata={len(acc_total):,} elapsed={time.time()-t0:.1f}s", file=sys.stderr)

    rows = []
    for st, n in acc_total.items():
        o = acc_obs.get(st, 0)
        vals = meta.get(st, tuple([NA_TOKEN] * len(strata_cols)))
        rec = {"stratum_id": st, "N_h": int(n), "O_h": int(o), "U_h": int(n - o)}
        for c, v in zip(strata_cols, vals):
            rec[c] = v
        rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise RuntimeError("No strata were constructed from universe table")
    out["observed_rate_h"] = np.where(out["N_h"] > 0, out["O_h"] / out["N_h"], np.nan)
    return out


def load_strata_counts_cache(path: str | Path, strata_cols: Sequence[str]) -> pd.DataFrame:
    df = read_tsv(path)
    required = {"N_h", "O_h", "U_h"} | set(strata_cols)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Strata-count cache is missing columns {sorted(missing)}: {path}")

    df = df.copy()
    df = normalize_strata_values(df, strata_cols)
    if "stratum_id" not in df.columns:
        df["stratum_id"] = make_stratum_id_df(df, strata_cols)
    else:
        df["stratum_id"] = df["stratum_id"].astype("string").fillna(NA_TOKEN).astype(str)

    for c in ["N_h", "O_h", "U_h"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(np.int64)
    if "observed_rate_h" not in df.columns:
        df["observed_rate_h"] = np.where(df["N_h"] > 0, df["O_h"] / df["N_h"], np.nan)
    else:
        df["observed_rate_h"] = pd.to_numeric(df["observed_rate_h"], errors="coerce").astype(float)

    dup = df["stratum_id"].duplicated()
    if dup.any():
        examples = df.loc[dup, "stratum_id"].head().tolist()
        raise ValueError(f"Strata-count cache has duplicated stratum_id values, examples={examples}: {path}")

    expected_u = df["N_h"] - df["O_h"]
    if not np.array_equal(expected_u.to_numpy(), df["U_h"].to_numpy()):
        raise ValueError(f"Strata-count cache has U_h != N_h - O_h: {path}")
    if (df[["N_h", "O_h", "U_h"]] < 0).any().any():
        raise ValueError(f"Strata-count cache contains negative counts: {path}")
    if (df["O_h"] > df["N_h"]).any():
        raise ValueError(f"Strata-count cache contains O_h > N_h: {path}")
    return df


def attach_sample_weights(
    sample_df: pd.DataFrame,
    strata_counts: pd.DataFrame,
    observed_col: str,
    strata_cols: Sequence[str],
    weight_col: Optional[str],
    require_all_observed: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sample_df = normalize_strata_values(sample_df, strata_cols)
    sample_df["stratum_id"] = make_stratum_id_df(sample_df, strata_cols)
    sample_df["__observed__"] = observed_bool(sample_df[observed_col]).astype(np.int8)

    sc = strata_counts[["stratum_id", "N_h", "O_h", "U_h", "observed_rate_h"]].copy()

    if weight_col and weight_col in sample_df.columns:
        sample_df["__weight__"] = pd.to_numeric(sample_df[weight_col], errors="coerce").astype(float)
        if sample_df["__weight__"].isna().any() or (sample_df["__weight__"] <= 0).any():
            raise ValueError(f"Invalid non-positive or NA weights in {weight_col}")
        m = sample_df[sample_df["__observed__"] == 0].groupby("stratum_id", sort=False).size().rename("m_h").reset_index()
        plan = sc.merge(m, on="stratum_id", how="left")
        plan["m_h"] = plan["m_h"].fillna(0).astype(int)
        plan["sampling_prob_h"] = np.where(plan["U_h"] > 0, plan["m_h"] / plan["U_h"], np.nan)
        plan["sampling_weight_h"] = np.where(plan["sampling_prob_h"] > 0, 1.0 / plan["sampling_prob_h"], np.nan)
    else:
        m = sample_df[sample_df["__observed__"] == 0].groupby("stratum_id", sort=False).size().rename("m_h").reset_index()
        plan = sc.merge(m, on="stratum_id", how="left")
        plan["m_h"] = plan["m_h"].fillna(0).astype(int)
        plan["sampling_prob_h"] = np.where(plan["U_h"] > 0, plan["m_h"] / plan["U_h"], np.nan)
        plan["sampling_weight_h"] = np.where(plan["sampling_prob_h"] > 0, 1.0 / plan["sampling_prob_h"], np.nan)
        sample_df = sample_df.merge(plan[["stratum_id", "sampling_weight_h"]], on="stratum_id", how="left", sort=False)
        sample_df["__weight__"] = np.where(
            sample_df["__observed__"].to_numpy() == 1,
            1.0,
            sample_df["sampling_weight_h"].to_numpy(dtype=float),
        )
        if sample_df.loc[sample_df["__observed__"] == 0, "__weight__"].isna().any():
            bad = sample_df.loc[(sample_df["__observed__"] == 0) & sample_df["__weight__"].isna(), "stratum_id"].head().tolist()
            raise RuntimeError(f"Some sampled unobserved variants have no valid stratum sampling weight: {bad}")

    universe_obs = int(sc["O_h"].sum())
    sample_obs = int(sample_df["__observed__"].sum())
    if require_all_observed and sample_obs != universe_obs:
        raise RuntimeError(
            f"Observed variants are not fully included: universe O={universe_obs:,}, sample O={sample_obs:,}. "
            "For rigorous O/E, include all observed variants or rerun with --allow-missing-observed."
        )

    sample_df = sample_df.merge(sc, on="stratum_id", how="left", sort=False)
    return sample_df, plan


def weighted_top_threshold(score: np.ndarray, weight: np.ndarray, frac: float, direction: str) -> Tuple[float, float, float]:
    return weighted_top_thresholds(score, weight, [frac], direction)[frac]


def weighted_top_thresholds(
    score: np.ndarray,
    weight: np.ndarray,
    fracs: Sequence[float],
    direction: str,
) -> Dict[float, Tuple[float, float, float]]:
    ok = np.isfinite(score) & np.isfinite(weight) & (weight > 0)
    score_ok = score[ok]
    weight_ok = weight[ok]
    if score_ok.size == 0:
        return {float(frac): (np.nan, 0.0, 0.0) for frac in fracs}

    order = np.argsort(score_ok)
    if direction == "high":
        order = order[::-1]
    s = score_ok[order]
    w = weight_ok[order]
    total_w = float(w.sum())
    csum = np.cumsum(w)

    out: Dict[float, Tuple[float, float, float]] = {}
    for frac in fracs:
        frac = float(frac)
        target = total_w * frac
        idx = int(np.searchsorted(csum, target, side="left"))
        idx = min(max(idx, 0), len(s) - 1)
        thr = float(s[idx])
        if direction == "high":
            top_w = float(weight_ok[score_ok >= thr].sum())
        else:
            top_w = float(weight_ok[score_ok <= thr].sum())
        out[frac] = (thr, target, top_w)
    return out


def top_mask(df: pd.DataFrame, score_col: str, threshold: float, direction: str) -> np.ndarray:
    x = pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=float)
    if direction == "high":
        return np.isfinite(x) & (x >= threshold)
    return np.isfinite(x) & (x <= threshold)


def compute_oe_for_threshold(
    sample_df: pd.DataFrame,
    strata_counts: pd.DataFrame,
    score_col: str,
    threshold: float,
    direction: str,
    region_col: Optional[str] = None,
) -> Tuple[dict, pd.DataFrame]:
    mask = top_mask(sample_df, score_col, threshold, direction)
    top_cols = ["stratum_id", "__observed__", "__weight__", "observed_rate_h"]
    if region_col and region_col in sample_df.columns:
        top_cols.append(region_col)
    top = sample_df.loc[mask, top_cols].copy()

    total_N = float(strata_counts["N_h"].sum())
    total_O = float(strata_counts["O_h"].sum())
    global_rate = total_O / total_N if total_N > 0 else np.nan

    top_weight = float(top["__weight__"].sum())
    top_observed = float(top["__observed__"].sum())
    top_rate = top_observed / top_weight if top_weight > 0 else np.nan
    crude_oe = top_rate / global_rate if global_rate and global_rate > 0 else np.nan

    expected = float((top["__weight__"] * top["observed_rate_h"]).sum())
    strat_oe = top_observed / expected if expected > 0 else np.nan

    summary = {
        "threshold": threshold,
        "top_weight": top_weight,
        "top_observed": top_observed,
        "top_observed_rate": top_rate,
        "global_N": total_N,
        "global_O": total_O,
        "global_observed_rate": global_rate,
        "crude_OE": crude_oe,
        "strat_expected_obs": expected,
        "strat_OE": strat_oe,
    }

    region_df = pd.DataFrame()
    if region_col and region_col in sample_df.columns:
        regs = []

        sc = strata_counts.copy()
        if region_col not in sc.columns:

            pass
        for region, subtop in top.groupby(region_col, sort=False):
            region_top_weight = float(subtop["__weight__"].sum())
            region_top_obs = float(subtop["__observed__"].sum())
            region_expected = float((subtop["__weight__"] * subtop["observed_rate_h"]).sum())

            if region_col in strata_counts.columns:
                rsc = strata_counts[strata_counts[region_col].astype(str) == str(region)]
                region_N = float(rsc["N_h"].sum())
                region_O = float(rsc["O_h"].sum())
                region_rate = region_O / region_N if region_N > 0 else np.nan
            else:
                region_N = np.nan
                region_O = np.nan
                region_rate = np.nan
            regs.append({
                region_col: region,
                "region_N": region_N,
                "region_O": region_O,
                "region_observed_rate": region_rate,
                "region_top_weight": region_top_weight,
                "region_top_observed": region_top_obs,
                "region_top_observed_rate": region_top_obs / region_top_weight if region_top_weight > 0 else np.nan,
                "region_expected_obs": region_expected,
                "region_strat_OE": region_top_obs / region_expected if region_expected > 0 else np.nan,
            })
        region_df = pd.DataFrame(regs)

    return summary, region_df


def run_weighted_oe(
    sample_df: pd.DataFrame,
    strata_counts: pd.DataFrame,
    score_specs: Sequence[Tuple[str, str, str]],
    top_fracs: Sequence[float],
    region_col: Optional[str],
    within_region: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    region_global_rows = []
    region_within_rows = []

    w = sample_df["__weight__"].to_numpy(dtype=float)

    for method, score_col, direction in score_specs:
        if score_col not in sample_df.columns:
            raise KeyError(f"Score column not found in sample: {score_col}")
        score = pd.to_numeric(sample_df[score_col], errors="coerce").to_numpy(dtype=float)
        thresholds = weighted_top_thresholds(score, w, top_fracs, direction)
        for frac in top_fracs:
            thr, target_w, top_w_by_thr = thresholds[float(frac)]
            summary, reg = compute_oe_for_threshold(sample_df, strata_counts, score_col, thr, direction, region_col=region_col)
            summary.update({
                "method": method,
                "score_col": score_col,
                "direction": direction,
                "top_fraction": frac,
                "weighted_target_top": target_w,
                "weighted_top_by_threshold": top_w_by_thr,
            })
            summaries.append(summary)
            if len(reg):
                reg.insert(0, "method", method)
                reg.insert(1, "top_fraction", frac)
                reg.insert(2, "threshold", thr)
                region_global_rows.append(reg)

            if within_region and region_col and region_col in sample_df.columns:
                for region, idx in sample_df.groupby(region_col, sort=False).groups.items():
                    sub = sample_df.loc[idx]
                    sw = sub["__weight__"].to_numpy(dtype=float)
                    ss = pd.to_numeric(sub[score_col], errors="coerce").to_numpy(dtype=float)
                    rthr, rtarget, rtopw = weighted_top_threshold(ss, sw, frac, direction)
                    m = top_mask(sub, score_col, rthr, direction)
                    subtop = sub.loc[m]
                    expected = float((subtop["__weight__"] * subtop["observed_rate_h"]).sum())
                    observed = float(subtop["__observed__"].sum())
                    top_weight = float(subtop["__weight__"].sum())
                    if region_col in strata_counts.columns:
                        rsc = strata_counts[strata_counts[region_col].astype(str) == str(region)]
                        region_N = float(rsc["N_h"].sum())
                        region_O = float(rsc["O_h"].sum())
                    else:
                        region_N = np.nan
                        region_O = np.nan
                    region_within_rows.append({
                        "method": method,
                        "score_col": score_col,
                        "direction": direction,
                        region_col: region,
                        "top_fraction": frac,
                        "within_region_threshold": rthr,
                        "within_region_target_top_weight": rtarget,
                        "within_region_top_weight_by_threshold": rtopw,
                        "region_N": region_N,
                        "region_O": region_O,
                        "top_weight": top_weight,
                        "top_observed": observed,
                        "expected_obs": expected,
                        "strat_OE": observed / expected if expected > 0 else np.nan,
                    })

    summary_df = pd.DataFrame(summaries)
    region_global_df = pd.concat(region_global_rows, ignore_index=True) if region_global_rows else pd.DataFrame()
    region_within_df = pd.DataFrame(region_within_rows)
    return summary_df, region_global_df, region_within_df


def full_distribution_counts(path: str, col: str, batch_size: int = 1_000_000) -> pd.DataFrame:
    acc = defaultdict(int)
    for df in iter_table_batches(path, [col], batch_size=batch_size):
        x = df[col].astype("string").fillna(NA_TOKEN).astype(str)
        vc = x.value_counts(dropna=False)
        for k, v in vc.items():
            acc[str(k)] += int(v)
    return pd.DataFrame({"level": list(acc.keys()), "full_n": list(acc.values())})


def weighted_sample_distribution(sample_df: pd.DataFrame, col: str) -> pd.DataFrame:
    x = sample_df[col].astype("string").fillna(NA_TOKEN).astype(str)
    tmp = pd.DataFrame({"level": x, "w": sample_df["__weight__"].to_numpy(dtype=float)})
    return tmp.groupby("level", sort=False)["w"].sum().rename("sample_weighted_n").reset_index()


def load_distribution_full_counts_cache(path: str | Path, qc_cols: Sequence[str]) -> Dict[str, pd.DataFrame]:
    df = read_tsv(path)
    required = {"qc_col", "level", "full_n"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Distribution-QC cache is missing columns {sorted(missing)}: {path}")

    df = df[["qc_col", "level", "full_n"]].copy()
    df["qc_col"] = df["qc_col"].astype("string").fillna(NA_TOKEN).astype(str)
    df["level"] = df["level"].astype("string").fillna(NA_TOKEN).astype(str)
    df["full_n"] = pd.to_numeric(df["full_n"], errors="raise").astype(np.int64)
    if (df["full_n"] < 0).any():
        raise ValueError(f"Distribution-QC cache contains negative full_n values: {path}")

    out: Dict[str, pd.DataFrame] = {}
    for col in qc_cols:
        sub = df[df["qc_col"] == str(col)].copy()
        if sub.empty:
            raise ValueError(f"Distribution-QC cache does not contain qc_col={col!r}: {path}")
        if sub["level"].duplicated().any():
            raise ValueError(f"Distribution-QC cache has duplicate levels for qc_col={col!r}: {path}")
        out[str(col)] = sub.drop(columns=["qc_col"]).reset_index(drop=True)
    return out


def run_distribution_qc(
    universe_path: str,
    sample_df: pd.DataFrame,
    qc_cols: Sequence[str],
    batch_size: int,
    full_counts_cache: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    rows = []
    for col in qc_cols:
        if col not in sample_df.columns:
            continue
        if full_counts_cache is not None:
            if col not in full_counts_cache:
                raise ValueError(f"Distribution full-count cache missing qc_col={col!r}")
            full = full_counts_cache[col].copy()
        else:
            full = full_distribution_counts(universe_path, col, batch_size=batch_size)
        samp = weighted_sample_distribution(sample_df, col)
        m = full.merge(samp, on="level", how="outer").fillna(0)
        full_total = m["full_n"].sum()
        samp_total = m["sample_weighted_n"].sum()
        m["full_prop"] = m["full_n"] / full_total if full_total > 0 else np.nan
        m["sample_weighted_prop"] = m["sample_weighted_n"] / samp_total if samp_total > 0 else np.nan
        m["abs_prop_diff"] = (m["sample_weighted_prop"] - m["full_prop"]).abs()
        m.insert(0, "qc_col", col)
        rows.append(m)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description="Weighted stratified estimate of genome-wide O/E depletion")
    ap.add_argument("--universe-table", required=True, help="Full cheap genome-wide universe table: all possible/evaluable variants.")
    ap.add_argument("--sample-score-table", required=True, help="Scored phase-2 sample: all observed + sampled unobserved.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--score-spec", action="append", required=True, help="name:score_column:high|low. Repeatable.")
    ap.add_argument("--top-fracs", default="0.0001,0.001,0.01,0.05,0.10")
    ap.add_argument(
        "--strata-counts-cache",
        default=None,
        help="Optional cached strata_counts.tsv(.gz). If set, skip the full-universe stratum-count scan.",
    )
    ap.add_argument(
        "--distribution-qc-cache",
        default=None,
        help=(
            "Optional cached weighted_distribution_qc.tsv(.gz). Only qc_col/level/full_n are reused; "
            "sample weighted distribution is recomputed for the current score table."
        ),
    )
    ap.add_argument("--observed-col", default="Y_Label")
    ap.add_argument("--id-col", default="snp_id")
    ap.add_argument("--strata-cols", required=True, help="Comma-separated strata columns, e.g. region_type,chr_num,P_mut_bin,callable_bin,cheap_risk_bin")
    ap.add_argument("--region-col", default="region_type")
    ap.add_argument("--sample-weight-col", default=None, help="Optional precomputed weight column in sample table.")
    ap.add_argument("--qc-cols", default=None, help="Distribution QC columns. Default=strata columns plus region.")
    ap.add_argument("--batch-size", type=int, default=1_000_000)
    ap.add_argument("--allow-missing-observed", action="store_true")
    ap.add_argument("--no-within-region", action="store_true")
    ap.add_argument(
        "--skip-distribution-qc",
        action="store_true",
        help="Skip distribution QC. Intended for parameter sweeps; final validation should keep QC enabled.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings_seconds = {}
    run_t0 = time.time()

    score_specs = parse_score_specs(args.score_spec)
    top_fracs = parse_top_fracs(args.top_fracs)
    strata_cols = parse_csv_list(args.strata_cols)
    region_col = args.region_col if args.region_col else None
    qc_cols = parse_csv_list(args.qc_cols) if args.qc_cols else list(dict.fromkeys(strata_cols + ([region_col] if region_col else [])))

    step_t0 = time.time()
    if args.strata_counts_cache:
        print(f"[1/6] Loading cached full-universe stratum counts: {args.strata_counts_cache}", file=sys.stderr)
        strata_counts = load_strata_counts_cache(args.strata_counts_cache, strata_cols=strata_cols)
    else:
        print("[1/6] Building full-universe stratum counts...", file=sys.stderr)
        strata_counts = build_strata_counts(
            args.universe_table,
            observed_col=args.observed_col,
            strata_cols=strata_cols,
            region_col=region_col,
            batch_size=args.batch_size,
        )
    write_tsv(strata_counts, out_dir / "strata_counts.tsv.gz")
    timings_seconds["strata_counts"] = time.time() - step_t0

    step_t0 = time.time()
    print("[2/6] Loading scored sample...", file=sys.stderr)
    sample_cols = list(strata_cols) + [args.observed_col]
    if region_col and region_col not in sample_cols:
        sample_cols.append(region_col)
    if args.id_col:
        sample_cols.append(args.id_col)
    if args.sample_weight_col:
        sample_cols.append(args.sample_weight_col)
    for _, col, _ in score_specs:
        sample_cols.append(col)
    sample_df = read_table(args.sample_score_table, sample_cols)
    timings_seconds["load_scored_sample"] = time.time() - step_t0

    step_t0 = time.time()
    print("[3/6] Attaching inverse-probability weights...", file=sys.stderr)
    sample_df, sampling_plan = attach_sample_weights(
        sample_df,
        strata_counts=strata_counts,
        observed_col=args.observed_col,
        strata_cols=strata_cols,
        weight_col=args.sample_weight_col,
        require_all_observed=(not args.allow_missing_observed),
    )

    comp_cols = [c for c in strata_cols if c in strata_counts.columns]
    sampling_plan = sampling_plan.merge(strata_counts[["stratum_id"] + comp_cols], on="stratum_id", how="left")
    write_tsv(sampling_plan, out_dir / "sampling_plan.tsv.gz")
    timings_seconds["attach_weights"] = time.time() - step_t0

    step_t0 = time.time()
    print("[4/6] Computing weighted O/E...", file=sys.stderr)
    summary, region_global, region_within = run_weighted_oe(
        sample_df,
        strata_counts,
        score_specs=score_specs,
        top_fracs=top_fracs,
        region_col=region_col,
        within_region=(not args.no_within_region),
    )
    write_tsv(summary, out_dir / "weighted_oe_summary.tsv")
    if len(region_global):
        write_tsv(region_global, out_dir / "region_oe_global_threshold.tsv")
    if len(region_within):
        write_tsv(region_within, out_dir / "region_oe_within_region_threshold.tsv")
    timings_seconds["weighted_oe"] = time.time() - step_t0

    step_t0 = time.time()
    if args.skip_distribution_qc:
        print("[5/6] Skipping weighted distribution QC...", file=sys.stderr)
    else:
        print("[5/6] Running weighted distribution QC...", file=sys.stderr)
        distribution_full_counts_cache = None
        if args.distribution_qc_cache:
            print(f"      Loading cached full distribution counts: {args.distribution_qc_cache}", file=sys.stderr)
            distribution_full_counts_cache = load_distribution_full_counts_cache(args.distribution_qc_cache, qc_cols=qc_cols)
        dist_qc = run_distribution_qc(
            args.universe_table,
            sample_df,
            qc_cols=qc_cols,
            batch_size=args.batch_size,
            full_counts_cache=distribution_full_counts_cache,
        )
        if len(dist_qc):
            write_tsv(dist_qc, out_dir / "weighted_distribution_qc.tsv.gz")
    timings_seconds["distribution_qc"] = time.time() - step_t0

    step_t0 = time.time()
    print("[6/6] Writing manifest...", file=sys.stderr)
    timings_seconds["total_before_manifest"] = time.time() - run_t0
    manifest = {
        "script": "weighted_stratified_genomewide_oe.py",
        "universe_table": args.universe_table,
        "sample_score_table": args.sample_score_table,
        "score_specs": [{"method": m, "score_col": c, "direction": d} for m, c, d in score_specs],
        "top_fracs": top_fracs,
        "strata_counts_cache": args.strata_counts_cache,
        "distribution_qc_cache": args.distribution_qc_cache,
        "observed_col": args.observed_col,
        "strata_cols": strata_cols,
        "region_col": region_col,
        "sample_weight_col": args.sample_weight_col,
        "allow_missing_observed": bool(args.allow_missing_observed),
        "skip_distribution_qc": bool(args.skip_distribution_qc),
        "timings_seconds": timings_seconds,
        "outputs": [
            "strata_counts.tsv.gz",
            "sampling_plan.tsv.gz",
            "weighted_oe_summary.tsv",
            "region_oe_global_threshold.tsv",
            "region_oe_within_region_threshold.tsv",
            "weighted_distribution_qc.tsv.gz",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
