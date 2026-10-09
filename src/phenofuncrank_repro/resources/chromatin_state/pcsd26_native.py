#!/usr/bin/env python3
"Task 2: Genome-wide PCSD26 state sampling for maize noncoding sampled SNPs."

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SAMPLED = PCSD_BED = OUT = None
TMP_DIR = None

ALL_STATES = [f"E{i}" for i in range(1, 27)]


def run(command, output_path):
    print("[CMD]", command, flush=True)
    with Path(output_path).open("w") as handle:
        subprocess.run(command, check=True, stdout=handle)


def main():
    t0 = time.time()
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)

    print("[1/6] Reading sampled SNPs and writing SNP BED...", flush=True)
    snps = pd.read_parquet(SAMPLED, columns=["snp_id", "Chr", "Pos"])
    snps["Chr"] = snps["Chr"].astype(str)
    snps["Pos"] = snps["Pos"].astype("int64")

    snp_bed = TMP_DIR / "snps.bed"
    bed_df = snps[["Chr", "Pos"]].copy()
    bed_df["start"] = bed_df["Pos"] - 1  
    bed_df["end"] = bed_df["Pos"]
    bed_df = bed_df[["Chr", "start", "end"]].copy()
    bed_df.insert(3, "snp_id", snps["snp_id"].astype(str).values)
    bed_df = bed_df.sort_values(["Chr", "start", "end"])
    bed_df.to_csv(snp_bed, sep="\t", header=False, index=False)
    print(f"  wrote {len(bed_df):,} SNPs to {snp_bed}", flush=True)

    print("[2/6] Cleaning PCSD BED...", flush=True)
    clean_bed = TMP_DIR / "pcsd26.clean.bed"
    n_pcsd = 0
    states_seen = set()
    with open(PCSD_BED) as fin, open(clean_bed, "w") as fout:
        for line in fin:
            if not line.strip() or line.startswith("#"):
                continue
            arr = line.rstrip("\n").split("\t")
            if len(arr) < 4:
                arr = line.strip().split()
            if len(arr) < 4:
                continue
            chrom = arr[0]
            try:
                start = int(float(arr[1]))
                end = int(float(arr[2]))
            except ValueError:
                continue
            if end <= start:
                continue
            state = arr[3]
            if state == "":
                state = "unknown"
            fout.write(f"{chrom}\t{start}\t{end}\t{state}\n")
            states_seen.add(state)
            n_pcsd += 1
    print(f"  clean pcsd records={n_pcsd:,} states={sorted(states_seen)}", flush=True)


    sorted_bed = TMP_DIR / "pcsd26.sorted.bed"
    run(["bedtools", "sort", "-i", str(clean_bed)], sorted_bed)

    print("[3/6] Running bedtools intersect -loj...", flush=True)
    intersect_tsv = TMP_DIR / "intersect.tsv"
    run(["bedtools", "intersect", "-loj", "-a", str(snp_bed), "-b", str(sorted_bed)], intersect_tsv)

    print("[4/6] Building feature table from intersect...", flush=True)
    names = ["a_chr", "a_start", "a_end", "snp_id", "b_chr", "b_start", "b_end", "state"]
    parts = []
    total = 0
    for i, df in enumerate(pd.read_csv(intersect_tsv, sep="\t", names=names, chunksize=2_000_000, low_memory=False)):
        total += len(df)
        df = df[["snp_id", "state"]].copy()
        df["snp_id"] = df["snp_id"].astype(str)
        df.loc[df["state"].isin([".", "-1", -1]), "state"] = pd.NA
        df = df.drop_duplicates("snp_id", keep="first")
        parts.append(df)
        if (i + 1) % 5 == 0:
            print(f"  [chunk {i+1}] total_raw={total:,}", flush=True)

    feat = pd.concat(parts, ignore_index=True)
    feat = feat.drop_duplicates("snp_id", keep="first").reset_index(drop=True)
    print(f"  matched unique snps={len(feat):,}", flush=True)

    print("[5/6] One-hot encoding E1-E26...", flush=True)
    feat["pcsd26_state"] = feat["state"].astype("string")
    feat = feat.drop(columns=["state"])
    feat["pcsd26_missing"] = feat["pcsd26_state"].isna().astype("int8")

    for st in ALL_STATES:
        col = f"pcsd26_state__{st}"
        feat[col] = (feat["pcsd26_state"] == st).fillna(False).astype("int8")


    out = snps[["snp_id"]].merge(feat, on="snp_id", how="left")
    out["pcsd26_missing"] = out["pcsd26_missing"].fillna(1).astype("int8")
    for st in ALL_STATES:
        col = f"pcsd26_state__{st}"
        out[col] = out[col].fillna(0).astype("int8")

    n_total = len(out)
    n_matched = out["pcsd26_missing"].eq(0).sum()
    print(f"  total snps={n_total:,}", flush=True)
    print(f"  pcsd26 coverage={n_matched/n_total:.6f} ({n_matched:,}/{n_total:,})", flush=True)

    print("[6/6] Writing output...", flush=True)
    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), str(out_path), compression="snappy")
    print(f"  wrote {out_path} ({len(out):,} rows x {len(out.columns)} cols)", flush=True)


    shutil.rmtree(TMP_DIR)
    print(f"[DONE] elapsed={time.time()-t0:.1f}s", flush=True)


def cli():
    import argparse
    import tempfile
    global SAMPLED, PCSD_BED, OUT, TMP_DIR
    parser = argparse.ArgumentParser(description="Native PCSD E1-E26 mapping; preserved first-overlap and missing-state policy.")
    parser.add_argument("--sampled", required=True)
    parser.add_argument("--pcsd-bed", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir")
    args = parser.parse_args()
    SAMPLED, PCSD_BED, OUT = args.sampled, args.pcsd_bed, args.output
    if args.work_dir:
        Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pcsd26-", dir=args.work_dir) as temporary:
        TMP_DIR = Path(temporary)
        main()

if __name__ == "__main__":
    cli()
