#!/usr/bin/env python3


import argparse
import json
import math
import os
import re
import shutil
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ATAC_FEATURES = [
    "atac_overlap_any",
    "atac_overlap_count",
    "atac_sample_count",
    "atac_tissue_group_count",
    "atac_overlap_fiber",
    "atac_overlap_reproductive",
    "atac_overlap_vegetative",
    "log_dist_to_nearest_atac_peak",
    "log_dist_to_nearest_atac_summit",
    "atac_max_signalValue_log1p",
    "atac_max_qValue_log1p",
]

H3K27AC_FEATURES = [
    "h3k27ac_overlap_any",
    "h3k27ac_overlap_count",
    "h3k27ac_sample_count",
    "h3k27ac_tissue_group_count",
    "h3k27ac_overlap_fiber",
    "h3k27ac_overlap_reproductive",
    "h3k27ac_overlap_vegetative",
    "log_dist_to_nearest_h3k27ac_peak",
    "h3k27ac_max_signalValue_log1p",
    "h3k27ac_max_qValue_log1p",
]

COMBINED_FEATURES = [
    "active_regulatory_flag",
    "active_regulatory_count",
    "active_regulatory_tissue_group_count",
    "active_regulatory_fiber",
    "active_regulatory_reproductive",
    "active_regulatory_vegetative",
    "active_enhancer_like_flag",
    "fiber_active_enhancer_like_flag",
    "promoter_with_atac",
    "promoter_with_h3k27ac",
    "promoter_with_activity",
]

ALL_REG_FEATURES = ATAC_FEATURES + H3K27AC_FEATURES + COMBINED_FEATURES


def run(cmd):
    print("[CMD]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def sort_bed_inplace(bed_path):
    """
    Force BED sorting by chrom, start, end.
    This avoids bedtools closest/intersect failures caused by out-of-order records.
    """
    bed_path = Path(bed_path)
    if not bed_path.exists():
        return bed_path

    tmp = bed_path.with_suffix(bed_path.suffix + ".sorted.tmp")

    cmd = (
        "LC_ALL=C sort "
        "-k1,1 -k2,2n -k3,3n "
        f"{shlex.quote(str(bed_path))} > {shlex.quote(str(tmp))} "
        f"&& mv {shlex.quote(str(tmp))} {shlex.quote(str(bed_path))}"
    )
    subprocess.run(["bash", "-lc", cmd], check=True)
    return bed_path


def read_first_non_comment(path):
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            return line
    return None


def is_int_like(x):
    try:
        int(float(str(x)))
        return True
    except Exception:
        return False


def clean_token(x):
    x = str(x)
    x = re.sub(r"\s+", "_", x)
    return x


def infer_sample_id_from_peak_name(x):
    """
    Parse sample_id from peak names like:
      SRR26894180_hypocotyl_ATAC_peak_2803
      SRRxxxx_tissue_H3K27ac_peak_123
    """
    x = str(x)
    m = re.search(r"(SRR\d+|ERR\d+|DRR\d+)", x)
    if m:
        return m.group(1)
    if "_peak_" in x:
        return x.split("_peak_")[0]
    return x


def infer_tissue_group(text):
    x = str(text).lower()
    if x in {"fiber", "reproductive", "vegetative", "unknown"}:
        return x


    if any(k in x for k in [
        "fiber", "fibre", "fib", "dpa", "lint"
    ]):
        return "fiber"


    if any(k in x for k in [
        "ovule", "flower", "floral", "anther", "pollen", "seed",
        "embryo", "boll", "petal", "stamen", "pistil", "stigma",
        "reproductive"
    ]):
        return "reproductive"


    if any(k in x for k in [
        "leaf", "root", "stem", "shoot", "cotyledon", "seedling",
        "hypocotyl", "petiole", "radicle", "vegetative"
    ]):
        return "vegetative"

    return "unknown"


def normalize_peak_bed(
    bed_path,
    assay,
    out_prefix,
    has_summit=False,
    sample_col=None,
    tissue_col=None,
    signal_col=None,
    qvalue_col=None,
    summit_col=None,
    summit_coordinate_mode="offset",
):
    """
    Normalize peak BED to:
      chrom start end peak_id sample_id tissue_group signalValue qValue

    Supports narrowPeak/broadPeak-like files:
      chrom start end name score strand signalValue pValue qValue peak
    """
    bed_path = Path(bed_path)
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    first = read_first_non_comment(bed_path)
    if first is None:
        raise RuntimeError(f"Empty BED: {bed_path}")

    fields = first.split("\t")
    has_header = not (len(fields) >= 3 and is_int_like(fields[1]) and is_int_like(fields[2]))

    if has_header:
        raw = pd.read_csv(bed_path, sep="\t", comment="#")
        lower = {c.lower(): c for c in raw.columns}

        def find_col(cands):
            for c in cands:
                if c.lower() in lower:
                    return lower[c.lower()]
            return None

        chrom_c = find_col(["chrom", "chr", "chromosome"])
        start_c = find_col(["start", "chromStart", "chrom_start"])
        end_c = find_col(["end", "chromEnd", "chrom_end"])
        name_c = find_col(["name", "peak_id", "id"])
        sample_c = find_col(["sample", "sample_id", "sampleid"])
        tissue_c = find_col(["tissue_group", "group", "tissue", "organ"])
        signal_c = find_col(["signalValue", "signal", "signal_value"])
        qvalue_c = find_col(["qValue", "qvalue", "q_value"])
        summit_c = find_col(["peak", "summit", "summit_offset"])

        if chrom_c is None or start_c is None or end_c is None:
            raise RuntimeError(f"Cannot infer chrom/start/end from header: {bed_path}")

        df = pd.DataFrame({
            "chrom": raw[chrom_c].astype(str),
            "start": pd.to_numeric(raw[start_c], errors="coerce").astype("Int64"),
            "end": pd.to_numeric(raw[end_c], errors="coerce").astype("Int64"),
        })

        if name_c:
            df["name"] = raw[name_c].astype(str)
        else:
            df["name"] = [f"{assay}_peak_{i}" for i in range(len(df))]

        if sample_c:
            df["sample_id"] = raw[sample_c].astype(str).map(infer_sample_id_from_peak_name)
        else:
            df["sample_id"] = df["name"].astype(str).map(infer_sample_id_from_peak_name)

        if tissue_c:
            df["tissue_group"] = raw[tissue_c].astype(str).map(infer_tissue_group)
        else:
            df["tissue_group"] = df["name"].astype(str).map(infer_tissue_group)

        if signal_c:
            df["signalValue"] = pd.to_numeric(raw[signal_c], errors="coerce")
        else:
            df["signalValue"] = 0.0

        if qvalue_c:
            df["qValue"] = pd.to_numeric(raw[qvalue_c], errors="coerce")
        else:
            df["qValue"] = 0.0

        if has_summit and summit_c:
            df["summit_offset"] = pd.to_numeric(raw[summit_c], errors="coerce")
        else:
            df["summit_offset"] = np.nan

    else:
        raw = pd.read_csv(bed_path, sep="\t", comment="#", header=None)
        ncol = raw.shape[1]

        def col_or_default(idx, default):
            if idx is not None and idx >= 0 and idx < ncol:
                return raw.iloc[:, idx]
            return default

        chrom_idx, start_idx, end_idx = 0, 1, 2


        name_idx = 3 if ncol > 3 else None
        signal_idx = 6 if ncol > 6 else None
        qvalue_idx = 8 if ncol > 8 else None
        summit_idx = 9 if ncol > 9 else None


        if sample_col is not None:
            sample_idx = sample_col - 1
        else:
            sample_idx = name_idx

        if tissue_col is not None:
            tissue_idx = tissue_col - 1
        else:
            tissue_idx = None

        if signal_col is not None:
            signal_idx = signal_col - 1

        if qvalue_col is not None:
            qvalue_idx = qvalue_col - 1

        if summit_col is not None:
            summit_idx = summit_col - 1

        df = pd.DataFrame({
            "chrom": raw.iloc[:, chrom_idx].astype(str),
            "start": pd.to_numeric(raw.iloc[:, start_idx], errors="coerce").astype("Int64"),
            "end": pd.to_numeric(raw.iloc[:, end_idx], errors="coerce").astype("Int64"),
        })

        if name_idx is not None:
            df["name"] = raw.iloc[:, name_idx].astype(str)
        else:
            df["name"] = [f"{assay}_peak_{i}" for i in range(len(df))]

        if sample_idx is not None and sample_idx >= 0 and sample_idx < ncol:
            df["sample_id"] = raw.iloc[:, sample_idx].astype(str).map(infer_sample_id_from_peak_name)
        else:
            df["sample_id"] = df["name"].astype(str).map(infer_sample_id_from_peak_name)

        if tissue_idx is not None and tissue_idx >= 0 and tissue_idx < ncol:
            df["tissue_group"] = raw.iloc[:, tissue_idx].astype(str).map(infer_tissue_group)
        else:

            text = raw.astype(str).agg("|".join, axis=1)
            df["tissue_group"] = text.map(infer_tissue_group)

        if signal_idx is not None and signal_idx >= 0 and signal_idx < ncol:
            df["signalValue"] = pd.to_numeric(raw.iloc[:, signal_idx], errors="coerce")
        else:
            df["signalValue"] = 0.0

        if qvalue_idx is not None and qvalue_idx >= 0 and qvalue_idx < ncol:
            df["qValue"] = pd.to_numeric(raw.iloc[:, qvalue_idx], errors="coerce")
        else:
            df["qValue"] = 0.0

        if has_summit and summit_idx is not None and summit_idx >= 0 and summit_idx < ncol:
            df["summit_offset"] = pd.to_numeric(raw.iloc[:, summit_idx], errors="coerce")
        else:
            df["summit_offset"] = np.nan

    df = df.dropna(subset=["chrom", "start", "end"])
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df = df[df["end"] > df["start"]].copy()

    df["peak_id"] = [f"{assay}_peak_{i}" for i in range(len(df))]
    df["sample_id"] = df["sample_id"].map(clean_token)
    df["tissue_group"] = df["tissue_group"].map(clean_token)
    df["signalValue"] = pd.to_numeric(df["signalValue"], errors="coerce").fillna(0.0)
    df["qValue"] = pd.to_numeric(df["qValue"], errors="coerce").fillna(0.0)

    df = df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)

    norm_bed = out_prefix.with_suffix(".norm.bed")
    peak3_bed = out_prefix.with_suffix(".peak3.bed")
    summit_bed = out_prefix.with_suffix(".summit.bed")

    df[[
        "chrom", "start", "end",
        "peak_id", "sample_id", "tissue_group", "signalValue", "qValue"
    ]].to_csv(norm_bed, sep="\t", header=False, index=False)
    sort_bed_inplace(norm_bed)

    df[["chrom", "start", "end"]].to_csv(peak3_bed, sep="\t", header=False, index=False)
    sort_bed_inplace(peak3_bed)

    summit_path = None
    if has_summit:
        summit_offset = pd.to_numeric(df["summit_offset"], errors="coerce")
        summit_pos = np.where(
            summit_offset.notna() & (summit_offset >= 0),
            (summit_offset.fillna(0).astype(int).values if summit_coordinate_mode == "absolute"
             else df["start"].values + summit_offset.fillna(0).astype(int).values),
            ((df["start"].values + df["end"].values) // 2)
        )
        summit = pd.DataFrame({
            "chrom": df["chrom"].values,
            "start": summit_pos.astype(int),
            "end": summit_pos.astype(int) + 1,
        })
        summit = summit[summit["end"] > summit["start"]]
        summit.to_csv(summit_bed, sep="\t", header=False, index=False)
        sort_bed_inplace(summit_bed)
        summit_path = summit_bed

    summary = {
        "assay": assay,
        "input": str(bed_path),
        "n_peaks": int(len(df)),
        "chrom_count": int(df["chrom"].nunique()),
        "sample_id_count": int(df["sample_id"].nunique()),
        "summit_coordinate_mode": summit_coordinate_mode if has_summit else None,
        "tissue_group_counts": df["tissue_group"].value_counts(dropna=False).to_dict(),
        "norm_bed": str(norm_bed),
        "peak3_bed": str(peak3_bed),
        "summit_bed": str(summit_path) if summit_path else None,
    }

    return norm_bed, peak3_bed, summit_path, summary


def collect_unique_snps(in_root, collect_fold="fold01"):
    """
    Collect unique SNPs for regulatory annotation.

    IMPORTANT:
    For LOCO strict datasets with fold-specific negative sampling,
    collect_fold='all' is required. If only fold01 is collected, SNPs newly
    sampled in fold02-fold13 will receive default regulatory features and
    create severe missingness leakage.
    """
    in_root = Path(in_root)
    cols = ["snp_id", "Chr", "Pos", "Ref", "Alt"]

    if str(collect_fold).lower() in {"all", "*", "all_folds"}:
        fold_dirs = sorted(in_root.glob("fold*"))
        if not fold_dirs:
            raise RuntimeError(f"No fold directories found under {in_root}")
        print(f"[collect] collecting ALL folds: {len(fold_dirs)} fold dirs", flush=True)
    else:
        fold_dir = in_root / collect_fold
        if not fold_dir.exists():
            raise RuntimeError(f"collect fold not found: {fold_dir}")
        fold_dirs = [fold_dir]

    store = {}
    total_rows = 0

    for fd in fold_dirs:
        for sp in ["train", "val", "test"]:
            pp = fd / f"{sp}.parquet"
            if not pp.exists():
                continue

            df = pd.read_parquet(pp, columns=cols)
            total_rows += len(df)
            df = df.drop_duplicates("snp_id")

            before = len(store)
            for r in df.itertuples(index=False):
                sid = str(r.snp_id)
                if sid not in store:
                    store[sid] = (sid, str(r.Chr), int(r.Pos), str(r.Ref), str(r.Alt))

            after = len(store)
            print(
                f"[collect] {pp} rows={len(df):,} unique_total={after:,} new={after-before:,}",
                flush=True
            )

    x = pd.DataFrame.from_records(
        list(store.values()),
        columns=["snp_id", "Chr", "Pos", "Ref", "Alt"]
    )
    x["Chr"] = x["Chr"].astype(str)
    x["Pos"] = pd.to_numeric(x["Pos"], errors="coerce").astype(int)

    print(
        f"[collect] total scanned rows={total_rows:,}; unique SNPs from {collect_fold}: {len(x):,}",
        flush=True
    )

    return x


def collect_annotation_df(in_root, collect_fold="fold01"):
    in_root = Path(in_root)

    wanted = [
        "snp_id",
        "region_type",
        "promoter_core_flag",
        "is_promoter",
        "dist_to_TSS",
        "log_dist_to_TSS",
    ]

    if str(collect_fold).lower() in {"all", "*", "all_folds"}:
        fold_dirs = sorted(in_root.glob("fold*"))
        if not fold_dirs:
            raise RuntimeError(f"No fold directories found under {in_root}")
        print(f"[anno] collecting annotations from ALL folds: {len(fold_dirs)} fold dirs", flush=True)
    else:
        fold_dir = in_root / collect_fold
        if not fold_dir.exists():
            raise RuntimeError(f"annotation collect fold not found: {fold_dir}")
        fold_dirs = [fold_dir]

    store = {}
    total_rows = 0

    for fd in fold_dirs:
        for sp in ["train", "val", "test"]:
            pp = fd / f"{sp}.parquet"
            if not pp.exists():
                continue

            schema_cols = list(pq.read_schema(pp).names)
            cols = [c for c in wanted if c in schema_cols]
            if "snp_id" not in cols:
                raise RuntimeError(f"snp_id missing in {pp}")

            df = pd.read_parquet(pp, columns=cols)
            total_rows += len(df)
            df = df.drop_duplicates("snp_id")

            before = len(store)
            for r in df.itertuples(index=False):
                d = r._asdict()
                sid = str(d["snp_id"])
                if sid not in store:
                    store[sid] = d

            after = len(store)
            print(
                f"[anno] {pp} rows={len(df):,} anno_total={after:,} new={after-before:,}",
                flush=True
            )

    anno = pd.DataFrame(list(store.values()))
    print(
        f"[anno] total scanned rows={total_rows:,}; unique annotation SNPs={len(anno):,}",
        flush=True
    )

    return anno


def write_snp_bed(snps, out_bed):
    bed = pd.DataFrame({
        "chrom": snps["Chr"].astype(str),
        "start": snps["Pos"].astype(int) - 1,
        "end": snps["Pos"].astype(int),
        "snp_id": snps["snp_id"].astype(str),
    })
    bed = bed.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    bed.to_csv(out_bed, sep="\t", header=False, index=False)
    sort_bed_inplace(out_bed)
    return out_bed


def aggregate_overlap(intersect_path, assay):
    cols = [
        "a_chrom", "a_start", "a_end", "snp_id",
        "b_chrom", "b_start", "b_end",
        "peak_id", "sample_id", "tissue_group", "signalValue", "qValue"
    ]

    if not Path(intersect_path).exists() or Path(intersect_path).stat().st_size == 0:
        return pd.DataFrame(columns=["snp_id"])

    x = pd.read_csv(intersect_path, sep="\t", header=None, names=cols)
    x["signalValue"] = pd.to_numeric(x["signalValue"], errors="coerce").fillna(0.0)
    x["qValue"] = pd.to_numeric(x["qValue"], errors="coerce").fillna(0.0)
    x["tissue_group"] = x["tissue_group"].astype(str)

    def tissue_nunique(s):
        vals = set(v for v in s.astype(str) if v not in {"unknown", "nan", ""})
        return len(vals)

    g = x.groupby("snp_id", sort=False)

    out = pd.DataFrame({
        "snp_id": g.size().index,
        f"{assay}_overlap_any": 1,
        f"{assay}_overlap_count": g.size().values,
        f"{assay}_sample_count": g["sample_id"].nunique().values,
        f"{assay}_tissue_group_count": g["tissue_group"].agg(tissue_nunique).values,
        f"{assay}_overlap_fiber": g["tissue_group"].agg(lambda s: int((s == "fiber").any())).values,
        f"{assay}_overlap_reproductive": g["tissue_group"].agg(lambda s: int((s == "reproductive").any())).values,
        f"{assay}_overlap_vegetative": g["tissue_group"].agg(lambda s: int((s == "vegetative").any())).values,
        f"{assay}_max_signalValue_log1p": np.log1p(g["signalValue"].max().values),
        f"{assay}_max_qValue_log1p": np.log1p(g["qValue"].max().values),
    })

    return out


def read_closest_distance(closest_path, feature_name):

    cols = [
        "a_chrom", "a_start", "a_end", "snp_id",
        "b_chrom", "b_start", "b_end",
        "distance"
    ]

    if not Path(closest_path).exists() or Path(closest_path).stat().st_size == 0:
        return pd.DataFrame(columns=["snp_id", feature_name])

    x = pd.read_csv(closest_path, sep="\t", header=None, names=cols)
    x["distance"] = pd.to_numeric(x["distance"], errors="coerce")
    x.loc[x["distance"] < 0, "distance"] = np.nan
    d = x.groupby("snp_id", sort=False)["distance"].min().reset_index()
    d[feature_name] = np.log1p(d["distance"].fillna(1e9))
    return d[["snp_id", feature_name]]


def build_assay_features(snps, snp_bed, norm_bed, peak3_bed, assay, tmpdir, summit_bed=None):
    intersect_out = Path(tmpdir) / f"{assay}.intersect.tsv"
    closest_peak_out = Path(tmpdir) / f"{assay}.closest_peak.tsv"


    with open(intersect_out, "w") as fout:
        subprocess.run([
            "bedtools", "intersect",
            "-a", str(snp_bed),
            "-b", str(norm_bed),
            "-wa", "-wb"
        ], check=True, stdout=fout)

    with open(closest_peak_out, "w") as fout:
        subprocess.run([
            "bedtools", "closest",
            "-a", str(snp_bed),
            "-b", str(peak3_bed),
            "-d"
        ], check=True, stdout=fout)

    feat = pd.DataFrame({"snp_id": snps["snp_id"].astype(str).values})

    ov = aggregate_overlap(intersect_out, assay)
    feat = feat.merge(ov, on="snp_id", how="left")

    dist_peak_name = f"log_dist_to_nearest_{assay}_peak"
    dp = read_closest_distance(closest_peak_out, dist_peak_name)
    feat = feat.merge(dp, on="snp_id", how="left")

    if summit_bed is not None:
        closest_summit_out = Path(tmpdir) / f"{assay}.closest_summit.tsv"
        with open(closest_summit_out, "w") as fout:
            subprocess.run([
                "bedtools", "closest",
                "-a", str(snp_bed),
                "-b", str(summit_bed),
                "-d"
            ], check=True, stdout=fout)

        dist_summit_name = f"log_dist_to_nearest_{assay}_summit"
        ds = read_closest_distance(closest_summit_out, dist_summit_name)
        feat = feat.merge(ds, on="snp_id", how="left")

    return feat


def fill_defaults(df, feature_cols):
    df = df.copy()

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    for c in feature_cols:
        if c.startswith("log_dist_to_nearest_"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(np.log1p(1e9)).astype("float32")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype("float32")

    return df


def add_combined_features(df, enhancer_dist_threshold=1000):
    df = df.copy()


    df["active_regulatory_flag"] = (
        (df["atac_overlap_any"] > 0) |
        (df["h3k27ac_overlap_any"] > 0)
    ).astype("float32")

    df["active_regulatory_count"] = (
        df["atac_sample_count"] + df["h3k27ac_sample_count"]
    ).astype("float32")

    tissue_flags = []
    for group in ["fiber", "reproductive", "vegetative"]:
        c = f"active_regulatory_{group}"
        df[c] = (
            (df[f"atac_overlap_{group}"] > 0) |
            (df[f"h3k27ac_overlap_{group}"] > 0)
        ).astype("float32")
        tissue_flags.append(c)

    df["active_regulatory_tissue_group_count"] = df[tissue_flags].sum(axis=1).astype("float32")


    if "promoter_core_flag" in df.columns:
        promoter_core = pd.to_numeric(df["promoter_core_flag"], errors="coerce").fillna(0) > 0
    else:
        promoter_core = pd.Series(False, index=df.index)

    if "is_promoter" in df.columns:
        promoter_flag = pd.to_numeric(df["is_promoter"], errors="coerce").fillna(0) > 0
    elif "region_type" in df.columns:
        promoter_flag = df["region_type"].astype(str).eq("promoter")
    else:
        promoter_flag = pd.Series(False, index=df.index)

    if "dist_to_TSS" in df.columns:
        dist_to_tss = pd.to_numeric(df["dist_to_TSS"], errors="coerce").abs().fillna(0)
    elif "log_dist_to_TSS" in df.columns:
        dist_to_tss = np.expm1(pd.to_numeric(df["log_dist_to_TSS"], errors="coerce").fillna(0)).abs()
    else:
        dist_to_tss = pd.Series(0, index=df.index)

    distal_enough = dist_to_tss >= enhancer_dist_threshold

    df["active_enhancer_like_flag"] = (
        (df["atac_overlap_any"] > 0) &
        (df["h3k27ac_overlap_any"] > 0) &
        (~promoter_core) &
        distal_enough
    ).astype("float32")

    df["fiber_active_enhancer_like_flag"] = (
        (df["atac_overlap_fiber"] > 0) &
        (df["h3k27ac_overlap_fiber"] > 0) &
        (~promoter_core) &
        distal_enough
    ).astype("float32")

    df["promoter_with_atac"] = (
        promoter_flag & (df["atac_overlap_any"] > 0)
    ).astype("float32")

    df["promoter_with_h3k27ac"] = (
        promoter_flag & (df["h3k27ac_overlap_any"] > 0)
    ).astype("float32")

    df["promoter_with_activity"] = (
        promoter_flag & (df["active_regulatory_flag"] > 0)
    ).astype("float32")

    return df


def build_regulatory_feature_table(args):
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    snps = collect_unique_snps(args.in_root, args.collect_fold)
    snp_bed = work / "unique_snps.bed"
    write_snp_bed(snps, snp_bed)

    atac_norm, atac_peak3, atac_summit, atac_summary = normalize_peak_bed(
        args.atac_bed,
        assay="atac",
        out_prefix=work / "atac",
        has_summit=True,
        sample_col=args.atac_sample_col,
        tissue_col=args.atac_tissue_col,
        signal_col=args.atac_signal_col,
        qvalue_col=args.atac_qvalue_col,
        summit_col=args.atac_summit_col,
        summit_coordinate_mode=args.atac_summit_coordinate_mode,
    )

    h3_norm, h3_peak3, h3_summit, h3_summary = normalize_peak_bed(
        args.h3k27ac_bed,
        assay="h3k27ac",
        out_prefix=work / "h3k27ac",
        has_summit=False,
        sample_col=args.h3k27ac_sample_col,
        tissue_col=args.h3k27ac_tissue_col,
        signal_col=args.h3k27ac_signal_col,
        qvalue_col=args.h3k27ac_qvalue_col,
        summit_col=None,
    )

    print("[ATAC summary]", json.dumps(atac_summary, indent=2), flush=True)
    print("[H3K27ac summary]", json.dumps(h3_summary, indent=2), flush=True)

    tmpdir = work / "bedtools_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)

    atac_feat = build_assay_features(
        snps=snps,
        snp_bed=snp_bed,
        norm_bed=atac_norm,
        peak3_bed=atac_peak3,
        assay="atac",
        tmpdir=tmpdir,
        summit_bed=atac_summit,
    )
    atac_feat = fill_defaults(atac_feat, ATAC_FEATURES)

    h3_feat = build_assay_features(
        snps=snps,
        snp_bed=snp_bed,
        norm_bed=h3_norm,
        peak3_bed=h3_peak3,
        assay="h3k27ac",
        tmpdir=tmpdir,
        summit_bed=None,
    )
    h3_feat = fill_defaults(h3_feat, H3K27AC_FEATURES)

    reg = snps[["snp_id", "Chr", "Pos", "Ref", "Alt"]].copy()
    reg = reg.merge(atac_feat[["snp_id"] + ATAC_FEATURES], on="snp_id", how="left")
    reg = reg.merge(h3_feat[["snp_id"] + H3K27AC_FEATURES], on="snp_id", how="left")
    reg = fill_defaults(reg, ATAC_FEATURES + H3K27AC_FEATURES)


    

    anno = collect_annotation_df(args.in_root, args.collect_fold)
    reg = reg.merge(anno, on="snp_id", how="left")
    reg = add_combined_features(reg, args.enhancer_dist_threshold)
    reg = fill_defaults(reg, COMBINED_FEATURES)

    keep_cols = ["snp_id"] + ALL_REG_FEATURES
    reg = reg[keep_cols].drop_duplicates("snp_id")

    reg_path = Path(args.out_root) / "regulatory_features.atac_h3k27ac.parquet"
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    reg.to_parquet(reg_path, index=False)

    manifest = {
        "n_unique_snps": int(len(reg)),
        "regulatory_feature_path": str(reg_path),
        "atac_summary": atac_summary,
        "h3k27ac_summary": h3_summary,
        "enhancer_dist_threshold": args.enhancer_dist_threshold,
        "feature_means": {c: float(reg[c].mean()) for c in ALL_REG_FEATURES if c in reg.columns},
    }
    with open(str(reg_path) + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("[DONE feature table]", reg_path, reg.shape, flush=True)
    print(json.dumps(manifest, indent=2), flush=True)

    return reg


def merge_into_splits(args, reg):
    in_root = Path(args.in_root)
    out_root = Path(args.out_root)


    out_root.mkdir(parents=True, exist_ok=True)

    for fold_dir in sorted(in_root.glob("fold*")):
        out_fd = out_root / fold_dir.name
        out_fd.mkdir(parents=True, exist_ok=True)

        for sp in ["train", "val", "test"]:
            p = fold_dir / f"{sp}.parquet"
            if not p.exists():
                continue

            df = pd.read_parquet(p)
            n0 = len(df)
            existing_features = [c for c in ALL_REG_FEATURES if c in df.columns]
            df = df.drop(columns=existing_features)
            out = df.merge(reg, on="snp_id", how="left")
            out = fill_defaults(out, ALL_REG_FEATURES)

            out_p = out_fd / f"{sp}.parquet"
            out.to_parquet(out_p, index=False)

            print(
                f"[merge] {fold_dir.name} {sp}: {df.shape} -> {out.shape} "
                f"active_regulatory_rate={out['active_regulatory_flag'].mean():.4f} "
                f"atac_rate={out['atac_overlap_any'].mean():.4f} "
                f"h3k27ac_rate={out['h3k27ac_overlap_any'].mean():.4f}",
                flush=True
            )

        man = fold_dir / "manifest.json"
        if man.exists():
            (out_fd / "manifest.original.json").write_text(man.read_text())

    print("[DONE merge]", out_root, flush=True)


def prepare_output_root(args):
    out_root = Path(args.out_root).resolve()

    for source in [args.in_root, args.atac_bed, args.h3k27ac_bed]:
        input_path = Path(source).resolve()
        if input_path == out_root or out_root in input_path.parents:
            raise ValueError("Output cleanup would contain an input path; choose a separate output directory.")
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--atac-bed", required=True)
    ap.add_argument("--h3k27ac-bed", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--collect-fold", default="fold01")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--enhancer-dist-threshold", type=int, default=1000)


    ap.add_argument("--atac-sample-col", type=int, default=None)
    ap.add_argument("--atac-tissue-col", type=int, default=None)
    ap.add_argument("--atac-signal-col", type=int, default=None)
    ap.add_argument("--atac-qvalue-col", type=int, default=None)
    ap.add_argument("--atac-summit-col", type=int, default=None)
    ap.add_argument("--atac-summit-coordinate-mode", choices=["offset", "absolute"], default="offset",
                    help="Raw narrowPeak uses offset; project feature_peaks col10 stores absolute BED coordinate.")

    ap.add_argument("--h3k27ac-sample-col", type=int, default=None)
    ap.add_argument("--h3k27ac-tissue-col", type=int, default=None)
    ap.add_argument("--h3k27ac-signal-col", type=int, default=None)
    ap.add_argument("--h3k27ac-qvalue-col", type=int, default=None)

    args = ap.parse_args()

    prepare_output_root(args)
    reg = build_regulatory_feature_table(args)
    merge_into_splits(args, reg)


if __name__ == "__main__":
    main()
