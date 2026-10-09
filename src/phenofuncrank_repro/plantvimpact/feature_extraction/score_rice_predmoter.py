#!/usr/bin/env python3


from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import zlib
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

BASES = {"A", "C", "G", "T"}
SEQ_LEN_DEFAULT = 21384

PRED_FEATURES = [
    "predmoter_atac_w151_ref_mean",
    "predmoter_atac_w151_ref_max",
    "predmoter_atac_w151_delta_signed_mean",
    "predmoter_atac_w151_delta_abs_mean",
    "predmoter_atac_w151_delta_abs_max",
    "predmoter_atac_w151_delta_log_abs_mean",
    "predmoter_h3k4me3_w501_ref_mean",
    "predmoter_h3k4me3_w501_ref_max",
    "predmoter_h3k4me3_w501_delta_signed_mean",
    "predmoter_h3k4me3_w501_delta_abs_mean",
    "predmoter_h3k4me3_w501_delta_abs_max",
    "predmoter_h3k4me3_w501_delta_log_abs_mean",
]


def open_text(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def load_fasta(path: str) -> Dict[str, str]:
    genome: Dict[str, list[str]] = {}
    name = None
    with open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
                if name in genome:
                    raise RuntimeError(f"Duplicate FASTA header: {name}")
                genome[name] = []
            else:
                if name is None:
                    raise RuntimeError("FASTA sequence before first header")
                genome[name].append(line.upper())
    out = {k: "".join(v) for k, v in genome.items()}
    print(f"[FASTA] contigs={len(out)} total_bp={sum(map(len, out.values())):,}")
    return out


def chromosome_aliases(genome: Dict[str, str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for k in genome:
        vals = {k, k.lower()}
        kl = k.lower()
        if kl.startswith("chr"):
            vals.add(k[3:])
            vals.add(k[3:].lower())
        else:
            vals.add("chr" + k)
            vals.add(("chr" + k).lower())

        tail = re_strip_chr(k)
        if tail.isdigit():
            vals.add(str(int(tail)))
            vals.add(f"chr{int(tail)}")
            vals.add(f"Chr{int(tail)}")
        for v in vals:
            aliases[str(v)] = k
    return aliases


def re_strip_chr(x: str) -> str:
    s = str(x)
    return s[3:] if s.lower().startswith("chr") else s


def stable_mod(s: str, n: int) -> int:
    return zlib.crc32(str(s).encode("utf-8")) % n


def read_targets(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        cols = pq.ParquetFile(path).schema.names
        need = [c for c in ["snp_id", "Chr", "Pos", "Ref", "Alt"] if c in cols]
        df = pd.read_parquet(path, columns=need)
    else:
        df = pd.read_csv(path, sep="\t", compression="infer", usecols=lambda c: c in {"snp_id", "Chr", "Pos", "Ref", "Alt"})
    required = ["snp_id", "Chr", "Pos", "Ref", "Alt"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Targets missing columns: {missing}")
    df = df[required].copy()
    df["snp_id"] = df["snp_id"].astype(str)
    df["Chr"] = df["Chr"].astype(str)
    df["Pos"] = pd.to_numeric(df["Pos"], errors="coerce").astype("Int64")
    df["Ref"] = df["Ref"].astype(str).str.upper().str[:1]
    df["Alt"] = df["Alt"].astype(str).str.upper().str[:1]
    df = df.dropna(subset=["Pos"]).copy()
    df["Pos"] = df["Pos"].astype(np.int64)
    df = df[df["Ref"].isin(BASES) & df["Alt"].isin(BASES) & (df["Ref"] != df["Alt"])].copy()

    dup_conflict = df.groupby("snp_id")[["Chr", "Pos", "Ref", "Alt"]].nunique(dropna=False).max(axis=1)
    if (dup_conflict > 1).any():
        raise RuntimeError(f"Conflicting coordinates/alleles for duplicated snp_id: {(dup_conflict > 1).sum()}")
    df = df.drop_duplicates("snp_id", keep="first").reset_index(drop=True)
    return df


def make_lookup() -> np.ndarray:

    lut = np.zeros((256, 4), dtype=np.float32)
    lut[ord("C"), 0] = 1.0
    lut[ord("A"), 1] = 1.0
    lut[ord("T"), 2] = 1.0
    lut[ord("G"), 3] = 1.0
    return lut


LUT = make_lookup()


def encode_sequences(seqs: list[str]) -> torch.Tensor:
    arr = np.empty((len(seqs), len(seqs[0]), 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        b = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
        arr[i] = LUT[b]
    return torch.from_numpy(arr)


def extract_window(seq: str, pos1: int, ref: str, alt: str, seq_len: int) -> tuple[str, str, int, int, str]:
    center = seq_len // 2
    pos0 = pos1 - 1
    if pos0 < 0 or pos0 >= len(seq):
        return "", "", 0, 0, "position_out_of_range"
    if seq[pos0].upper() != ref:
        return "", "", 0, 0, "ref_mismatch"

    start0 = pos0 - center
    end0 = start0 + seq_len
    src_start = max(0, start0)
    src_end = min(len(seq), end0)
    left_pad = max(0, -start0)
    right_pad = max(0, end0 - len(seq))
    w = "N" * left_pad + seq[src_start:src_end].upper() + "N" * right_pad
    if len(w) != seq_len:
        raise AssertionError((len(w), seq_len))
    if w[center] != ref:
        raise AssertionError(f"center ref mismatch after extraction: {w[center]} != {ref}")
    wa = w[:center] + alt + w[center + 1:]

    valid_left = left_pad
    valid_right = seq_len - right_pad
    return w, wa, valid_left, valid_right, "ok"


def aggregate_track(ref: np.ndarray, alt: np.ndarray, center: int, half_window: int,
                    valid_left: int, valid_right: int, prefix: str) -> dict[str, float]:
    lo = max(center - half_window, valid_left)
    hi = min(center + half_window + 1, valid_right)
    if hi <= lo:
        return {
            f"{prefix}_ref_mean": np.nan,
            f"{prefix}_ref_max": np.nan,
            f"{prefix}_delta_signed_mean": np.nan,
            f"{prefix}_delta_abs_mean": np.nan,
            f"{prefix}_delta_abs_max": np.nan,
            f"{prefix}_delta_log_abs_mean": np.nan,
        }
    r = ref[lo:hi].astype(np.float64)
    a = alt[lo:hi].astype(np.float64)
    d = a - r
    ld = np.log1p(np.maximum(a, 0.0)) - np.log1p(np.maximum(r, 0.0))
    return {
        f"{prefix}_ref_mean": float(np.mean(r)),
        f"{prefix}_ref_max": float(np.max(r)),
        f"{prefix}_delta_signed_mean": float(np.mean(d)),
        f"{prefix}_delta_abs_mean": float(np.mean(np.abs(d))),
        f"{prefix}_delta_abs_max": float(np.max(np.abs(d))),
        f"{prefix}_delta_log_abs_mean": float(np.mean(np.abs(ld))),
    }


def score_valid_batch(model: LitHybridNet, rows: list[dict], device: torch.device,
                      seq_len: int, dataset_index: dict[str, int]) -> list[dict]:
    refs = [r["ref_window"] for r in rows]
    alts = [r["alt_window"] for r in rows]
    x = encode_sequences(refs + alts).to(device, non_blocking=True)

    x_rc = torch.flip(x, dims=[1])[:, :, [3, 2, 1, 0]]
    xx = torch.cat([x, x_rc], dim=0)

    try:
        with torch.inference_mode():
            pred_log = model(xx)
            pred = torch.exp(pred_log)
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            "CUDA OOM in Predmoter scoring. Reduce --batch-size; each SNP generates ref/alt on both strands (4 sequences)."
        ) from e

    n2 = x.shape[0]  
    pf = pred[:n2]
    prc = torch.flip(pred[n2:], dims=[1])
    p = ((pf + prc) * 0.5).detach().cpu().numpy().astype(np.float32)
    b = len(rows)
    pref = p[:b]
    palt = p[b:]
    center = seq_len // 2

    out = []
    for i, r in enumerate(rows):
        rec = {k: r[k] for k in ["snp_id", "Chr", "Pos", "Ref", "Alt"]}
        rec["predmoter_status"] = "ok"
        rec["predmoter_model_seq_len"] = int(seq_len)
        rec["predmoter_atac_window_bp"] = 151
        rec["predmoter_h3k4me3_window_bp"] = 501

        atac_i = dataset_index["atacseq"]
        h3_i = dataset_index["h3k4me3"]
        rec.update(aggregate_track(
            pref[i, :, atac_i], palt[i, :, atac_i], center, 75,
            r["valid_left"], r["valid_right"], "predmoter_atac_w151"
        ))
        rec.update(aggregate_track(
            pref[i, :, h3_i], palt[i, :, h3_i], center, 250,
            r["valid_left"], r["valid_right"], "predmoter_h3k4me3_w501"
        ))
        out.append(rec)

    del x, x_rc, xx, pred_log, pred, pf, prc, p
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Score Rice noncoding candidate SNPs with frozen Predmoter Combined_02")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16, help="SNPs per batch; actual model batch is 4x due to ref/alt and two strands")
    ap.add_argument("--chunk-size", type=int, default=20000)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--max-targets", type=int, default=None, help="Pilot/debug only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    try:
        from predmoter.prediction.HybridModel import LitHybridNet
    except ImportError as exc:
        raise RuntimeError(
            "PredMotER is an external model package. Install the pinned "
            "PredMotER checkout configured in configs/plantvimpact/paths.yaml, "
            "then rerun this local scorer."
        ) from exc

    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")
    target_path = Path(args.targets)
    checkpoint = Path(args.checkpoint)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for p in [target_path, checkpoint, Path(args.genome)]:
        if not p.exists():
            raise FileNotFoundError(p)

    if args.overwrite:
        for p in outdir.glob("predmoter_scored_chunk_*.parquet"):
            p.unlink()
        for p in [outdir / "manifest.json"]:
            if p.exists():
                p.unlink()

    targets = read_targets(target_path)
    targets = targets[targets["snp_id"].map(lambda x: stable_mod(x, args.num_shards) == args.shard_id)].reset_index(drop=True)
    if args.max_targets is not None:
        targets = targets.iloc[:args.max_targets].copy()
    print(f"[TARGETS] shard={args.shard_id}/{args.num_shards} n={len(targets):,}")

    genome = load_fasta(args.genome)
    aliases = chromosome_aliases(genome)

    device = torch.device(args.device)
    model = LitHybridNet.load_from_checkpoint(str(checkpoint), map_location="cpu", weights_only=False)
    model = model.to(device)
    model.eval()
    seq_len = int(getattr(model, "seq_len", SEQ_LEN_DEFAULT))
    datasets = [str(x.decode() if isinstance(x, bytes) else x).lower() for x in model.datasets]
    dataset_index = {name: i for i, name in enumerate(datasets)}
    print(f"[MODEL] seq_len={seq_len} datasets={datasets}")
    missing_dsets = [x for x in ["atacseq", "h3k4me3"] if x not in dataset_index]
    if missing_dsets:
        raise RuntimeError(f"Checkpoint does not contain required outputs: {missing_dsets}; datasets={datasets}")

    chunk_paths = []
    status_counts: dict[str, int] = {}
    total = 0

    for chunk_id, start in enumerate(tqdm(range(0, len(targets), args.chunk_size), desc="Predmoter chunks")):
        out_path = outdir / f"predmoter_scored_chunk_{chunk_id:06d}.parquet"
        if out_path.exists() and not args.overwrite:
            chunk_paths.append(str(out_path))
            total += pq.ParquetFile(out_path).metadata.num_rows
            continue

        sub = targets.iloc[start:start + args.chunk_size]
        records = []
        valid = []
        for row in sub.itertuples(index=False):
            chrom_input = str(row.Chr)
            ck = aliases.get(chrom_input, aliases.get(chrom_input.lower()))
            base = {"snp_id": str(row.snp_id), "Chr": chrom_input, "Pos": int(row.Pos), "Ref": str(row.Ref), "Alt": str(row.Alt)}
            if ck is None:
                rec = dict(base)
                rec["predmoter_status"] = "missing_chrom"
                records.append(rec)
                status_counts["missing_chrom"] = status_counts.get("missing_chrom", 0) + 1
                continue
            rw, aw, vl, vr, status = extract_window(genome[ck], int(row.Pos), str(row.Ref), str(row.Alt), seq_len)
            if status != "ok":
                rec = dict(base)
                rec["predmoter_status"] = status
                records.append(rec)
                status_counts[status] = status_counts.get(status, 0) + 1
                continue
            item = dict(base)
            item.update({"ref_window": rw, "alt_window": aw, "valid_left": vl, "valid_right": vr})
            valid.append(item)

        for bs in range(0, len(valid), args.batch_size):
            scored = score_valid_batch(model, valid[bs:bs + args.batch_size], device, seq_len, dataset_index)
            records.extend(scored)
            status_counts["ok"] = status_counts.get("ok", 0) + len(scored)

        out_df = pd.DataFrame(records)
        for c in PRED_FEATURES:
            if c not in out_df.columns:
                out_df[c] = np.nan
        for c in ["predmoter_model_seq_len", "predmoter_atac_window_bp", "predmoter_h3k4me3_window_bp"]:
            if c not in out_df.columns:
                out_df[c] = np.nan

        order = {sid: i for i, sid in enumerate(sub["snp_id"].astype(str).tolist())}
        out_df["_order"] = out_df["snp_id"].map(order)
        out_df = out_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
        pq.write_table(pa.Table.from_pandas(out_df, preserve_index=False), out_path, compression="zstd")
        chunk_paths.append(str(out_path))
        total += len(out_df)
        print(f"[WRITE] chunk={chunk_id:06d} rows={len(out_df):,} ok={(out_df['predmoter_status']=='ok').sum():,} -> {out_path}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    status_counts = {}
    total_from_files = 0
    for cp in chunk_paths:
        st = pd.read_parquet(cp, columns=["predmoter_status"])["predmoter_status"].astype(str).value_counts()
        total_from_files += int(st.sum())
        for k, v in st.items():
            status_counts[k] = status_counts.get(k, 0) + int(v)
    total = total_from_files

    manifest = {
        "targets": str(target_path),
        "genome": str(args.genome),
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "seq_len": seq_len,
        "datasets": datasets,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "batch_size_snps": args.batch_size,
        "actual_sequence_batch_multiplier": 4,
        "chunk_size": args.chunk_size,
        "target_rows_in_shard": int(len(targets)),
        "written_rows": int(total),
        "status_counts": status_counts,
        "chunks": chunk_paths,
        "feature_definition": {
            "ATAC": "strand-averaged raw float coverage; local 151-bp window around SNP",
            "H3K4me3": "strand-averaged raw float coverage; local 501-bp window around SNP",
            "delta": "Alt minus Ref; raw predictions are used before official integer rounding",
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    if total != len(targets):
        raise SystemExit(f"[FAIL] written rows {total} != targets {len(targets)}")
    print("[DONE]", json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
