#!/usr/bin/env python3


from __future__ import annotations

import argparse
import gc
import gzip
import json
import logging
import os
import re
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("score_coding_lm_targets_only")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A", "GCA": "GCG".replace("GCG", "A"), "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

BASES = ("A", "C", "G", "T")


def open_text(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def stable_mod(x: str, n: int) -> int:
    return zlib.crc32(str(x).encode("utf-8")) % int(n)


def norm_gene_id(x: str) -> str:
    x = str(x).strip().strip('"')
    x = x.split(",", 1)[0]
    for p in ("gene:", "transcript:", "mRNA:", "CDS:", "cds:"):
        if x.startswith(p):
            x = x[len(p):]
    x = x.split(".", 1)[0]
    return x


def comp_base(b: str) -> str:
    return str(Seq(str(b).upper()).reverse_complement())


def translate_cds(cds: str) -> str:
    cds = cds.upper()
    n = (len(cds) // 3) * 3
    aa = []
    for i in range(0, n, 3):
        a = CODON_TABLE.get(cds[i:i+3], "X")
        if a == "*":
            break
        aa.append(a)
    return "".join(aa)


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


class GFFParser:
    def __init__(self, gff_file: str):
        self.gff_file = gff_file
        self.gene_cds: Dict[str, List[dict]] = {}
        self._parse_gff()

    def _extract_gene_id(self, attrs: str) -> Optional[str]:
        for attr in attrs.split(";"):
            attr = attr.strip()
            if attr.startswith("Parent="):
                return norm_gene_id(attr.split("=", 1)[1])
            if attr.startswith("gene_id="):
                return norm_gene_id(attr.split("=", 1)[1])
            if attr.startswith("ID="):
                return norm_gene_id(attr.split("=", 1)[1].replace("CDS:", "").replace("cds:", ""))
        return None

    def _parse_gff(self):
        logger.info(f"Parsing GFF: {self.gff_file}")
        cds_count = 0
        with open_text(self.gff_file) as f:
            for line in f:
                if not line or line.startswith("#"):
                    continue
                p = line.rstrip("\n").split("\t")
                if len(p) < 9 or p[2] != "CDS":
                    continue
                gid = self._extract_gene_id(p[8])
                if not gid:
                    continue
                rec = {
                    "chr": p[0],
                    "start": int(p[3]),
                    "end": int(p[4]),
                    "strand": p[6],
                    "gene_id": gid,
                }
                self.gene_cds.setdefault(gid, []).append(rec)
                cds_count += 1

        for gid, arr in self.gene_cds.items():
            strand = arr[0]["strand"]
            arr.sort(key=lambda x: x["start"], reverse=(strand == "-"))

        logger.info(f"GFF parsed: CDS={cds_count:,}, genes/transcripts={len(self.gene_cds):,}")


class GeneCDSMapper:
    def __init__(self, gene_cds: Dict[str, List[dict]], genome: Dict[str, str]):
        self.gene_cds = gene_cds
        self.genome = genome
        self._cache: Dict[str, dict] = {}

    def build_gene_map(self, gene_id: str) -> Optional[dict]:
        gid = norm_gene_id(gene_id)
        if gid in self._cache:
            return self._cache[gid]
        if gid not in self.gene_cds:
            return None

        cds_list = self.gene_cds[gid]
        chrom = cds_list[0]["chr"]
        strand = cds_list[0]["strand"]
        chrom_key = resolve_chrom_name(chrom, self.genome)
        if chrom_key is None:
            return None

        cds_seq_parts = []
        cds_to_genome = []
        for cds in cds_list:
            start, end = int(cds["start"]), int(cds["end"])
            frag = self.genome[chrom_key][start - 1:end]
            if strand == "-":
                frag = str(Seq(frag).reverse_complement())
                pos_list = list(range(end, start - 1, -1))
            else:
                pos_list = list(range(start, end + 1))
            cds_seq_parts.append(frag)
            cds_to_genome.extend(pos_list)

        cds_sequence = "".join(cds_seq_parts).upper()
        protein_seq = translate_cds(cds_sequence)
        gm = {
            "gene_id": gid,
            "chr": chrom,
            "strand": strand,
            "chrom_key": chrom_key,
            "cds_sequence": cds_sequence,
            "cds_to_genome": cds_to_genome,
            "protein_seq": protein_seq,
            "pos_to_cds": {int(gp): i for i, gp in enumerate(cds_to_genome)},
        }
        self._cache[gid] = gm
        return gm

    def _ensure_interval_bins(self):
        if hasattr(self, "_interval_bins"):
            return
        binsize = 10000
        self._interval_binsize = binsize
        self._interval_bins = {}
        for gid, cds_list in self.gene_cds.items():
            for cds in cds_list:
                chrom = str(cds["chr"])
                start = int(cds["start"])
                end = int(cds["end"])
                if start > end:
                    start, end = end, start
                b0 = start // binsize
                b1 = end // binsize
                for key_chrom in {chrom, chrom.replace("Chr", ""), "Chr" + chrom if not chrom.startswith("Chr") else chrom}:
                    d = self._interval_bins.setdefault(str(key_chrom), {})
                    for b in range(b0, b1 + 1):
                        d.setdefault(b, []).append((start, end, gid))

    def find_gene_maps_by_position(self, chrom: str, pos: int) -> list:
        """Find transcript/gene maps whose CDS interval contains chrom:pos."""
        self._ensure_interval_bins()
        binsize = self._interval_binsize
        chrom = str(chrom)
        keys = [chrom]
        if chrom.startswith("Chr"):
            keys.append(chrom[3:])
        else:
            keys.append("Chr" + chrom)

        seen = set()
        out = []
        b = int(pos) // binsize

        for ck in keys:
            for start, end, gid in self._interval_bins.get(str(ck), {}).get(b, []):
                if gid in seen:
                    continue
                if int(start) <= int(pos) <= int(end):
                    gm = self.build_gene_map(gid)
                    if gm is not None:
                        seen.add(gid)
                        out.append(gm)
        return out


def pick_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def load_targets(path: str, num_shards: int, shard_id: int) -> pd.DataFrame:
    logger.info(f"Loading coding LM targets: {path}")
    df = pd.read_csv(path, sep="\t", compression="infer")
    if df.empty:
        raise RuntimeError("Target file is empty.")

    c_snp = pick_col(df, ["snp_id", "SNP_ID", "variant_id"])
    c_chr = pick_col(df, ["Chr", "chrom", "chr"])
    c_pos = pick_col(df, ["Pos", "pos", "position"])
    c_ref = pick_col(df, ["Ref", "ref"])
    c_alt = pick_col(df, ["Alt", "alt"])
    c_gid = pick_col(df, ["gene_id", "GeneID", "gene", "Gene"])

    missing = [name for name, c in [("Chr", c_chr), ("Pos", c_pos), ("Ref", c_ref), ("Alt", c_alt), ("gene_id/GeneID", c_gid)] if c is None]
    if missing:
        raise RuntimeError(f"Target file missing required columns for targeted coding scoring: {missing}. Columns={list(df.columns)}")

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

    out["chrom"] = df[c_chr].astype(str)
    out["pos"] = pd.to_numeric(df[c_pos], errors="coerce").astype("Int64")
    out["ref"] = df[c_ref].astype(str).str.upper().str[:1]
    out["alt"] = df[c_alt].astype(str).str.upper().str[:1]
    out["gene_id"] = df[c_gid].astype(str).map(norm_gene_id)


    for col in ["ref_aa", "alt_aa", "aa_pos", "aa_mut"]:
        c = pick_col(df, [col])
        if c is not None:
            out[col] = df[c]

    out = out.dropna(subset=["pos"]).copy()
    out["pos"] = out["pos"].astype(np.int64)
    out = out[out["ref"].isin(BASES) & out["alt"].isin(BASES) & (out["ref"] != out["alt"])].copy()
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
            key = (t["chrom_key"], int(t["pos"]), t["ref"])
            mapping.setdefault(key, []).append((i, t["alt"]))

        queries = list(mapping.keys())
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
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


class ESMTargetScorer:
    def __init__(self, model_name: str, protein_by_gene: Dict[str, str], device: str = "cuda", window_aa: int = 1020):
        self.protein_by_gene = protein_by_gene
        self.device = device
        self.window_aa = int(min(window_aa, 1020))
        logger.info(f"Loading ESM model: {model_name}")
        import esm

        orig_torch_load = torch.load

        def torch_load_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return orig_torch_load(*args, **kwargs)

        try:
            torch.load = torch_load_compat
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        finally:
            torch.load = orig_torch_load

        self.model = self.model.to(device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        logger.info("ESM model loaded.")

    @torch.no_grad()
    def score(self, tasks: List[dict], batch_size: int) -> np.ndarray:
        scores = np.full(len(tasks), np.nan, dtype=np.float32)
        mapping = {}
        for i, t in enumerate(tasks):
            key = (t["gene_id"], int(t["aa_pos"]), t["ref_aa"])
            mapping.setdefault(key, []).append((i, t["alt_aa"]))

        queries = list(mapping.keys())
        queries.sort(key=lambda q: min(len(self.protein_by_gene[q[0]]), self.window_aa))

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
            batch_data, pos_in_windows = [], []
            for gid, aa_pos, _ref_aa in batch:
                prot = self.protein_by_gene[gid]
                start = max(0, aa_pos - self.window_aa // 2)
                end = min(len(prot), start + self.window_aa)
                start = max(0, end - self.window_aa)
                win = prot[start:end]
                batch_data.append(win)
                pos_in_windows.append(aa_pos - start)

            data = [(str(j), seq) for j, seq in enumerate(batch_data)]
            _, _, toks = self.batch_converter(data)
            toks = toks.to(self.device)
            for j, pos in enumerate(pos_in_windows):
                toks[j, pos + 1] = self.alphabet.mask_idx

            use_amp = torch.cuda.is_available() and str(self.device).startswith("cuda")
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = self.model(toks)
            logits = out["logits"] if isinstance(out, dict) else out[0]

            for j, q in enumerate(batch):
                gid, aa_pos, ref_aa = q
                logp = torch.log_softmax(logits[j, pos_in_windows[j] + 1], dim=-1)
                ref_idx = self.alphabet.get_idx(ref_aa)
                for task_idx, alt_aa in mapping[q]:
                    alt_idx = self.alphabet.get_idx(alt_aa)
                    scores[task_idx] = float(abs((logp[alt_idx] - logp[ref_idx]).item()))
        return scores

    def clear_cache(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def prepare_tasks(df: pd.DataFrame, mapper: GeneCDSMapper, genome: Dict[str, str]) -> Tuple[List[dict], pd.DataFrame, Dict[str, str], Counter]:
    from collections import Counter
    tasks = []
    out_rows = []
    protein_by_gene = {}
    stats = Counter()

    def get_optional(row, name):
        return getattr(row, name) if hasattr(row, name) else None

    def clean_aa(x):
        if x is None:
            return None
        x = str(x)
        if x in ("", "nan", "None", "<NA>"):
            return None
        return x.upper()[:1]

    def clean_int(x):
        if x is None:
            return None
        try:
            if pd.isna(x):
                return None
            return int(x)
        except Exception:
            return None

    def try_make_task(gm, snp_id, chrom, pos, ref, alt, target_aa_pos, target_ref_aa, target_alt_aa):
        if pos not in gm["pos_to_cds"]:
            return None, "pos_not_in_cds"

        cds_pos = int(gm["pos_to_cds"][pos])
        cds_seq = gm["cds_sequence"]
        codon_idx = cds_pos // 3
        pos_in_codon = cds_pos % 3

        if codon_idx >= len(gm["protein_seq"]):
            return None, "aa_pos_outside_protein"

        ref_coding = cds_seq[cds_pos].upper()
        expected_ref = ref_coding if gm["strand"] == "+" else comp_base(ref_coding)
        if expected_ref != ref:
            return None, "ref_mismatch"

        alt_coding = alt if gm["strand"] == "+" else comp_base(alt)
        codon = cds_seq[codon_idx * 3: codon_idx * 3 + 3].upper()
        if len(codon) != 3 or any(b not in BASES for b in codon):
            return None, "bad_codon"

        mut = list(codon)
        mut[pos_in_codon] = alt_coding
        mut_codon = "".join(mut)

        ref_aa = CODON_TABLE.get(codon)
        alt_aa = CODON_TABLE.get(mut_codon)

        if not ref_aa or not alt_aa or ref_aa == "*" or alt_aa == "*":
            return None, "stop_or_unknown"
        if ref_aa == alt_aa:
            return None, "synonymous"


        if target_aa_pos is not None and int(target_aa_pos) != int(codon_idx):
            return None, "aa_pos_mismatch"
        if target_ref_aa is not None and target_ref_aa != ref_aa:
            return None, "ref_aa_mismatch"
        if target_alt_aa is not None and target_alt_aa != alt_aa:
            return None, "alt_aa_mismatch"

        chrom_key = resolve_chrom_name(chrom, genome)
        if chrom_key is None:
            return None, "missing_chrom"

        gid = gm["gene_id"]
        protein_by_gene[gid] = gm["protein_seq"]

        t = {
            "snp_id": snp_id,
            "chrom": chrom,
            "pos": int(pos),
            "ref": ref,
            "alt": alt,
            "gene_id": gid,
            "target_gene_id": None,
            "strand": gm["strand"],
            "chrom_key": chrom_key,
            "aa_pos": int(codon_idx),
            "ref_aa": ref_aa,
            "alt_aa": alt_aa,
            "aa_mut": f"{ref_aa}{codon_idx + 1}{alt_aa}",
        }
        return t, "ok"

    for row in df.itertuples(index=False):
        snp_id = str(row.snp_id)
        target_gid = norm_gene_id(row.gene_id)
        chrom = str(row.chrom)
        pos = int(row.pos)
        ref = str(row.ref).upper()
        alt = str(row.alt).upper()

        target_aa_pos = clean_int(get_optional(row, "aa_pos"))
        target_ref_aa = clean_aa(get_optional(row, "ref_aa"))
        target_alt_aa = clean_aa(get_optional(row, "alt_aa"))

        candidates = []
        gm0 = mapper.build_gene_map(target_gid)
        if gm0 is not None:
            candidates.append(gm0)


        for gm in mapper.find_gene_maps_by_position(chrom, pos):
            if gm["gene_id"] not in {x["gene_id"] for x in candidates}:
                candidates.append(gm)

        if not candidates:
            stats["no_cds_interval_or_gene_map"] += 1
            continue

        fail = Counter()
        chosen = None
        for gm in candidates:
            t, reason = try_make_task(
                gm, snp_id, chrom, pos, ref, alt,
                target_aa_pos, target_ref_aa, target_alt_aa
            )
            if t is not None:
                t["target_gene_id"] = target_gid
                chosen = t
                break
            fail[reason] += 1

        if chosen is None:
            if fail:
                stats[fail.most_common(1)[0][0]] += 1
            else:
                stats["unresolved"] += 1
            continue

        tasks.append(chosen)
        out_rows.append({k: v for k, v in chosen.items() if k != "chrom_key"})
        stats["ok"] += 1

    return tasks, pd.DataFrame(out_rows), protein_by_gene, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--gff", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dna_model", required=True)
    ap.add_argument("--esm_model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dna_max_length", type=int, default=512)
    ap.add_argument("--protein_window_aa", type=int, default=1020)
    ap.add_argument("--chunk_size", type=int, default=50000)
    ap.add_argument("--dna_batch_size", type=int, default=64)
    ap.add_argument("--esm_batch_size", type=int, default=16)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for p in outdir.glob("coding_lm_targets_scored_chunk_*.parquet"):
            p.unlink()
        m = outdir / "manifest.json"
        if m.exists():
            m.unlink()

    genome = load_genome_fasta(args.genome)
    gff = GFFParser(args.gff)
    mapper = GeneCDSMapper(gff.gene_cds, genome)

    targets = load_targets(args.targets, args.num_shards, args.shard_id)
    if targets.empty:
        logger.warning("No targets in this shard.")
        return

    dna = DNATargetScorer(args.dna_model, genome, args.dna_max_length, args.device)


    esm = ESMTargetScorer(args.esm_model, {}, args.device, args.protein_window_aa)

    chunk_paths = []
    total_ok = 0
    total_stats = {}

    for chunk_idx, start in enumerate(tqdm(range(0, len(targets), args.chunk_size), desc="Target chunks")):
        sub = targets.iloc[start:start + args.chunk_size].copy()
        tasks, out_df, protein_by_gene, stats = prepare_tasks(sub, mapper, genome)
        total_stats[chunk_idx] = dict(stats)

        if not tasks:
            logger.warning(f"chunk={chunk_idx} no valid coding missense targets; stats={dict(stats)}")
            continue

        esm.protein_by_gene = protein_by_gene

        s_dna = dna.score(tasks, args.dna_batch_size)
        s_esm = esm.score(tasks, args.esm_batch_size)

        out_df["S_dna"] = s_dna
        out_df["S_esm"] = s_esm

        out_path = outdir / f"coding_lm_targets_scored_chunk_{chunk_idx:06d}.parquet"
        pq.write_table(pa.Table.from_pandas(out_df, preserve_index=False), str(out_path), compression="snappy")
        chunk_paths.append(str(out_path))
        total_ok += len(out_df)

        logger.info(f"chunk={chunk_idx:06d} written={len(out_df):,} stats={dict(stats)} -> {out_path.name}")

        dna.clear_cache()

    esm.clear_cache()

    manifest = {
        "targets": args.targets,
        "outdir": str(outdir),
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "total_targets_in_shard": int(len(targets)),
        "total_scored": int(total_ok),
        "chunks": chunk_paths,
        "chunk_stats": total_stats,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info(f"DONE shard={args.shard_id}; scored={total_ok:,}/{len(targets):,}")


if __name__ == "__main__":
    main()
