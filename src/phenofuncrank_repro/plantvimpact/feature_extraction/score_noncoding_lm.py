#!/usr/bin/env python3


from __future__ import annotations

import argparse
import gc
import gzip
import json
import logging
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("score_noncoding_lm_targets_only")

BASES = ("A", "C", "G", "T")


def stable_mod(x: str, n: int) -> int:
    return zlib.crc32(str(x).encode("utf-8")) % int(n)


def load_genome_fasta(fasta_file: str) -> Dict[str, str]:
    logger.info(f"Loading genome FASTA: {fasta_file}")
    if str(fasta_file).endswith(".gz"):
        with gzip.open(fasta_file, "rt") as handle:
            genome = {record.id: str(record.seq).upper() for record in SeqIO.parse(handle, "fasta")}
    else:
        with open(fasta_file, "rt") as handle:
            genome = {record.id: str(record.seq).upper() for record in SeqIO.parse(handle, "fasta")}
    logger.info(f"Genome loaded: contigs={len(genome)}")
    return genome


def resolve_chrom_name(chrom: str, genome: Dict[str, str]) -> Optional[str]:
    chrom = str(chrom)
    if chrom in genome:
        return chrom
    cands = [chrom]
    if chrom.startswith("chr"):
        cands += [chrom[3:], chrom.replace("chr", "Chr", 1)]
    elif chrom.startswith("Chr"):
        cands += [chrom[3:], "chr" + chrom[3:]]
    else:
        cands += ["chr" + chrom, "Chr" + chrom]
    for c in cands:
        if c in genome:
            return c
    return None


def pick_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def load_targets(path: str, num_shards: int, shard_id: int) -> pd.DataFrame:
    logger.info(f"Loading noncoding DNA LM targets: {path}")
    df = pd.read_csv(path, sep="\t", compression="infer")
    if df.empty:
        raise RuntimeError("Target file is empty.")

    c_snp = pick_col(df, ["snp_id", "SNP_ID", "variant_id"])
    c_chr = pick_col(df, ["Chr", "chrom", "chr"])
    c_pos = pick_col(df, ["Pos", "pos", "position"])
    c_ref = pick_col(df, ["Ref", "ref"])
    c_alt = pick_col(df, ["Alt", "alt"])

    missing = [name for name, c in [("Chr", c_chr), ("Pos", c_pos), ("Ref", c_ref), ("Alt", c_alt)] if c is None]
    if missing:
        raise RuntimeError(f"Target file missing required columns: {missing}. Columns={list(df.columns)}")

    out = pd.DataFrame()
    if c_snp is not None:
        out["snp_id"] = df[c_snp].astype(str)
    else:
        out["snp_id"] = (
            df[c_chr].astype(str) + "_" +
            df[c_pos].astype(str) + "_" +
            df[c_ref].astype(str).str.upper().str[:1] + "_" +
            df[c_alt].astype(str).str.upper().str[:1]
        )

    out["Chr"] = df[c_chr].astype(str)
    out["Pos"] = pd.to_numeric(df[c_pos], errors="coerce").astype("Int64")
    out["Ref"] = df[c_ref].astype(str).str.upper().str[:1]
    out["Alt"] = df[c_alt].astype(str).str.upper().str[:1]

    for c in ["GeneID", "region_type", "region_subtype"]:
        cc = pick_col(df, [c])
        if cc is not None:
            out[c] = df[cc]

    out = out.dropna(subset=["Pos"]).copy()
    out["Pos"] = out["Pos"].astype(np.int64)
    out = out[out["Ref"].isin(BASES) & out["Alt"].isin(BASES) & (out["Ref"] != out["Alt"])].copy()
    out = out.drop_duplicates("snp_id").reset_index(drop=True)

    if num_shards > 1:
        keep = out["snp_id"].map(lambda x: stable_mod(x, num_shards) == shard_id)
        out = out.loc[keep].reset_index(drop=True)

    logger.info(f"Targets after shard filter: rows={len(out):,}, shard={shard_id}/{num_shards}")
    return out


class DNATargetScorer:
    def __init__(self, model_path: str, genome: Dict[str, str], max_length: int = 512, device: str = "cuda"):
        self.genome = genome
        self.max_length = int(max_length)
        self.device = device
        logger.info(f"Loading DNA model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        self.model = self.model.to(device)
        self.model.eval()
        self.mask_id = int(self.tokenizer.mask_token_id)
        self.base2id = {b: int(self.tokenizer(b, add_special_tokens=False)["input_ids"][0]) for b in BASES}
        logger.info("DNA model loaded.")

    def _build_window(self, chrom_key: str, pos_1based: int) -> Tuple[str, int]:
        seq = self.genome[chrom_key]
        L = self.max_length
        start = max(1, int(pos_1based) - L // 2)
        end = start + L - 1
        if end > len(seq):
            end = len(seq)
            start = max(1, end - L + 1)
        window = seq[start - 1:end].upper()
        if len(window) < L:
            window += "N" * (L - len(window))
        return window, int(pos_1based - start)

    @torch.no_grad()
    def score(self, tasks: List[dict], batch_size: int) -> np.ndarray:
        scores = np.full(len(tasks), np.nan, dtype=np.float32)

        mapping = {}
        for i, t in enumerate(tasks):
            key = (t["chrom_key"], int(t["Pos"]), t["Ref"])
            mapping.setdefault(key, []).append((i, t["Alt"]))

        queries = list(mapping.keys())

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            windows, idxs = [], []

            for chrom_key, pos, _ref in batch:
                w, idx = self._build_window(chrom_key, pos)
                windows.append(w)
                idxs.append(idx)

            enc = self.tokenizer(
                windows,
                add_special_tokens=False,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].to(self.device)

            for b, idx in enumerate(idxs):
                input_ids[b, idx] = self.mask_id

            use_amp = torch.cuda.is_available() and str(self.device).startswith("cuda")
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = self.model(input_ids=input_ids)
                logits = out.logits

            for b, q in enumerate(batch):
                _chrom_key, _pos, ref = q
                logp = torch.log_softmax(logits[b, idxs[b]], dim=-1)
                ref_id = self.base2id.get(ref)

                for task_idx, alt in mapping[q]:
                    alt_id = self.base2id.get(alt)
                    if ref_id is not None and alt_id is not None:
                        scores[task_idx] = float(abs((logp[alt_id] - logp[ref_id]).item()))

        return scores

    def clear_cache(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def prepare_tasks(df: pd.DataFrame, genome: Dict[str, str]) -> tuple[list, pd.DataFrame, dict]:
    tasks = []
    rows = []
    stats = {"ok": 0, "missing_chrom": 0, "ref_mismatch": 0}

    for row in df.itertuples(index=False):
        chrom = str(row.Chr)
        pos = int(row.Pos)
        ref = str(row.Ref).upper()
        alt = str(row.Alt).upper()

        ck = resolve_chrom_name(chrom, genome)
        if ck is None:
            stats["missing_chrom"] += 1
            continue

        if pos < 1 or pos > len(genome[ck]):
            stats["missing_chrom"] += 1
            continue

        genome_ref = genome[ck][pos - 1].upper()
        if genome_ref != ref:
            stats["ref_mismatch"] += 1
            continue

        t = {
            "snp_id": str(row.snp_id),
            "Chr": chrom,
            "Pos": pos,
            "Ref": ref,
            "Alt": alt,
            "chrom_key": ck,
        }

        for c in ["GeneID", "region_type", "region_subtype"]:
            if hasattr(row, c):
                t[c] = getattr(row, c)

        tasks.append(t)
        rows.append({k: v for k, v in t.items() if k != "chrom_key"})
        stats["ok"] += 1

    return tasks, pd.DataFrame(rows), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dna_model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dna_max_length", type=int, default=512)
    ap.add_argument("--chunk_size", type=int, default=100000)
    ap.add_argument("--dna_batch_size", type=int, default=128)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for p in outdir.glob("noncoding_lm_targets_scored_chunk_*.parquet"):
            p.unlink()
        mf = outdir / "manifest.json"
        if mf.exists():
            mf.unlink()

    genome = load_genome_fasta(args.genome)
    targets = load_targets(args.targets, args.num_shards, args.shard_id)
    if targets.empty:
        logger.warning("No targets in this shard.")
        return

    dna = DNATargetScorer(args.dna_model, genome, args.dna_max_length, args.device)

    chunk_paths = []
    total_ok = 0
    total_stats = {}

    for chunk_idx, start in enumerate(tqdm(range(0, len(targets), args.chunk_size), desc="Target chunks")):
        sub = targets.iloc[start:start + args.chunk_size].copy()
        tasks, out_df, stats = prepare_tasks(sub, genome)
        total_stats[chunk_idx] = stats

        if not tasks:
            logger.warning(f"chunk={chunk_idx} no valid noncoding targets; stats={stats}")
            continue

        s_dna = dna.score(tasks, args.dna_batch_size)
        out_df["S_dna"] = s_dna

        out_path = outdir / f"noncoding_lm_targets_scored_chunk_{chunk_idx:06d}.parquet"
        pq.write_table(pa.Table.from_pandas(out_df, preserve_index=False), str(out_path), compression="snappy")

        chunk_paths.append(str(out_path))
        total_ok += len(out_df)

        logger.info(f"chunk={chunk_idx:06d} written={len(out_df):,} stats={stats} -> {out_path.name}")
        dna.clear_cache()

    manifest = {
        "targets": args.targets,
        "outdir": str(outdir),
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "total_targets_in_shard": int(len(targets)),
        "total_scored": int(total_ok),
        "chunks": chunk_paths,
        "chunk_stats": total_stats,
        "score_definition": "S_dna = abs(logP(ALT|masked DNA context) - logP(REF|masked DNA context))",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info(f"DONE shard={args.shard_id}; scored={total_ok:,}/{len(targets):,}")


if __name__ == "__main__":
    main()
