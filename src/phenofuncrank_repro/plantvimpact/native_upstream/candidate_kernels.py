"Common source computations; profile dispatch lives in leaf modules."
from __future__ import annotations
import argparse
import gzip
import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("build_preLM_universe_from_raw")
BASES = np.array(["A", "C", "G", "T"], dtype=object)
BASE2INT = {"A": 0, "C": 1, "G": 2, "T": 3}
RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "C", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
CODON_TABLE["CGC"] = "R"

def _exact_chroms_from_regex(chrom_regex: Optional[str]) -> Optional[List[str]]:
    """Return exact chromosome names only for simple regex such as ^1$ or ^(1|2|3)$."""
    if not chrom_regex:
        return None
    x = str(chrom_regex)
    if not (x.startswith("^") and x.endswith("$")):
        return None
    inner = x[1:-1]
    if re.fullmatch(r"[A-Za-z0-9_.-]+", inner):
        return [inner]
    if inner.startswith("(") and inner.endswith(")") and "|" in inner:
        xs = inner[1:-1].split("|")
        if xs and all(re.fullmatch(r"[A-Za-z0-9_.-]+", z) for z in xs):
            return xs
    return None

def _runs_from_bool_mask(mask: np.ndarray, start_pos_1based: int) -> List[Tuple[int, int]]:
    """Convert a boolean slice for positions [start_pos_1based, ...] into 1-based closed intervals."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(idx[1:] != idx[:-1] + 1) + 1
    run_starts = np.r_[0, breaks]
    run_ends = np.r_[breaks - 1, idx.size - 1]
    out = []
    for rs, re in zip(run_starts, run_ends):
        a = int(start_pos_1based + idx[rs])
        b = int(start_pos_1based + idx[re])
        out.append((a, b))
    return out

def build_cds_map(cds_blocks: List[dict], genome: Dict[str, str]) -> Optional[dict]:
    if not cds_blocks:
        return None
    chrom = cds_blocks[0]["Chr"]
    if chrom not in genome:
        return None
    strand = cds_blocks[0]["strand"]

    blocks = sorted(cds_blocks, key=lambda x: x["start"], reverse=(strand == "-"))
    seq_parts = []
    cds_to_genome = []
    chrom_seq = genome[chrom]
    for b in blocks:
        frag = chrom_seq[b["start"] - 1:b["end"]].upper()
        if strand == "-":
            frag = frag.translate(RC)[::-1]
            positions = list(range(b["end"], b["start"] - 1, -1))
        else:
            positions = list(range(b["start"], b["end"] + 1))
        seq_parts.append(frag)
        cds_to_genome.extend(positions)
    return {"Chr": chrom, "strand": strand, "cds_seq": "".join(seq_parts), "cds_to_genome": cds_to_genome}

def callrate_lookup(call_idx: Dict[str, Tuple[np.ndarray, dict]], chrom: str, pos_arr: np.ndarray) -> dict:
    n = len(pos_arr)
    res = {
        "AN_obs": np.zeros(n, dtype=np.float64),
        "callable_n": np.zeros(n, dtype=np.float64),
        "callable_rate": np.zeros(n, dtype=np.float64),
        "call_rate": np.zeros(n, dtype=np.float64),
        "callrate_hit": np.zeros(n, dtype=np.int8),
    }
    if not call_idx or chrom not in call_idx or n == 0:
        return res
    sorted_pos, data = call_idx[chrom]
    loc = np.searchsorted(sorted_pos, pos_arr)
    hit = (loc < len(sorted_pos)) & (sorted_pos[np.minimum(loc, len(sorted_pos)-1)] == pos_arr)
    hit_idx = np.flatnonzero(hit)
    src_idx = loc[hit_idx]
    for col in ["AN_obs", "callable_n", "callable_rate", "call_rate"]:
        res[col][hit_idx] = data[col][src_idx]
    res["callrate_hit"][hit_idx] = 1
    return res

def clean_gene_id(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    x = str(x).split(",")[0]
    for pref in ["gene:", "transcript:", "CDS:", "cds:", "rna:", "mRNA:"]:
        if x.startswith(pref):
            x = x[len(pref):]
    return x.split(".")[0]

def collect_noncoding_intervals(gff_obj: dict, genome: Dict[str, str], args) -> Dict[str, List[dict]]:
    logger.info("[mask-opt] Building prioritized noncoding intervals with boolean masks ...")
    t0 = time.time()

    genes = gff_obj["genes"]
    cds_by_gene = gff_obj["cds_by_gene"]


    cds_mask_by_chr = {}
    assigned_mask_by_chr = {}
    for chrom, seq in genome.items():
        L = len(seq)
        cds_mask_by_chr[chrom] = np.zeros(L + 1, dtype=bool)
        assigned_mask_by_chr[chrom] = np.zeros(L + 1, dtype=bool)


    n_cds_blocks = 0
    for gid, cds_blocks in cds_by_gene.items():
        for b in cds_blocks:
            chrom = b["Chr"]
            if chrom not in genome:
                continue
            L = len(genome[chrom])
            a = max(1, int(b["start"]))
            e = min(L, int(b["end"]))
            if a <= e:
                cds_mask_by_chr[chrom][a:e+1] = True
                n_cds_blocks += 1

    logger.info(f"[mask-opt] CDS blockers marked: blocks={n_cds_blocks:,}")

    raw_by_priority = {
        "splice_near": [],
        "UTR5": [],
        "UTR3": [],
        "promoter": [],
        "intron": [],
    }

    def gene_meta(gid, chrom, strand):
        gene = genes.get(gid)
        if gene is None:
            return None, None, None
        tss = gene["start"] if strand == "+" else gene["end"]
        return tss, gene["start"], gene["end"]

    def add_raw(gid: str, chrom: str, strand: str, intervals: List[Tuple[int, int]], region_type: str):
        if chrom not in genome:
            return
        L = len(genome[chrom])
        tss, gstart, gend = gene_meta(gid, chrom, strand)
        if tss is None:
            return
        for a, b in intervals:
            a = max(1, int(a))
            b = min(L, int(b))
            if a <= b:
                raw_by_priority[region_type].append({
                    "start": a,
                    "end": b,
                    "GeneID": gid,
                    "gene_id": gid,
                    "strand": strand,
                    "region_type": region_type,
                    "tss": tss,
                    "gene_start": gstart,
                    "gene_end": gend,
                    "Chr": chrom,
                })


    for gid, cds_blocks in cds_by_gene.items():
        if gid not in genes or not cds_blocks:
            continue
        chrom = cds_blocks[0]["Chr"]
        strand = cds_blocks[0]["strand"]
        if chrom not in genome:
            continue
        blocks = sorted(intervals_from_blocks(cds_blocks))
        w = int(args.splice_window)
        splice_ints = []
        for (a1, b1), (a2, b2) in zip(blocks, blocks[1:]):
            if a2 <= b1 + 1:
                continue
            intron_start, intron_end = b1 + 1, a2 - 1
            splice_ints.extend([
                (intron_start, min(intron_end, intron_start + w - 1)),
                (max(intron_start, intron_end - w + 1), intron_end),
            ])
        add_raw(gid, chrom, strand, splice_ints, "splice_near")


    for gid, blocks in gff_obj["utr5_by_gene"].items():
        if blocks:
            add_raw(gid, blocks[0]["Chr"], blocks[0]["strand"], intervals_from_blocks(blocks), "UTR5")

    for gid, blocks in gff_obj["utr3_by_gene"].items():
        if blocks:
            add_raw(gid, blocks[0]["Chr"], blocks[0]["strand"], intervals_from_blocks(blocks), "UTR3")


    for gid, gene in genes.items():
        chrom = gene["Chr"]
        if chrom not in genome:
            continue
        strand = gene["strand"]
        if strand == "-":
            tss = gene["end"]
            ints = [(tss - args.promoter_downstream, tss + args.promoter_upstream)]
        else:
            tss = gene["start"]
            ints = [(tss - args.promoter_upstream, tss + args.promoter_downstream)]
        add_raw(gid, chrom, strand, ints, "promoter")


    for gid, cds_blocks in cds_by_gene.items():
        if gid not in genes or not cds_blocks:
            continue
        chrom = cds_blocks[0]["Chr"]
        strand = cds_blocks[0]["strand"]
        if chrom not in genome:
            continue
        blocks = sorted(intervals_from_blocks(cds_blocks))
        introns = []
        for (a1, b1), (a2, b2) in zip(blocks, blocks[1:]):
            if a2 > b1 + 1:
                introns.append((b1 + 1, a2 - 1))
        add_raw(gid, chrom, strand, introns, "intron")

    logger.info("[mask-opt] Raw interval records: " + json.dumps({k: len(v) for k, v in raw_by_priority.items()}, ensure_ascii=False))

    by_chr = defaultdict(list)

    for region_type in ["splice_near", "UTR5", "UTR3", "promoter", "intron"]:
        recs = raw_by_priority[region_type]
        kept_segments = 0
        kept_bp = 0

        for rec in recs:
            chrom = rec["Chr"]
            a, b = int(rec["start"]), int(rec["end"])
            if chrom not in genome or b < a:
                continue

            cds_mask = cds_mask_by_chr[chrom]
            assigned = assigned_mask_by_chr[chrom]


            clean = ~(cds_mask[a:b+1] | assigned[a:b+1])
            if not np.any(clean):
                continue

            for ca, cb in _runs_from_bool_mask(clean, a):
                out = {
                    "start": ca,
                    "end": cb,
                    "GeneID": rec["GeneID"],
                    "gene_id": rec["gene_id"],
                    "strand": rec["strand"],
                    "region_type": rec["region_type"],
                    "tss": rec["tss"],
                    "gene_start": rec["gene_start"],
                    "gene_end": rec["gene_end"],
                }
                by_chr[chrom].append(out)
                assigned[ca:cb+1] = True
                kept_segments += 1
                kept_bp += (cb - ca + 1)

        logger.info(f"[mask-opt] priority={region_type} raw_records={len(recs):,} kept_segments={kept_segments:,} kept_bp={kept_bp:,}")

    for chrom in by_chr:
        by_chr[chrom].sort(key=lambda r: (r["start"], r["end"], str(r["GeneID"]), r["region_type"]))

    logger.info("[mask-opt] Noncoding intervals by region_type: " + json.dumps(
        pd.Series([r["region_type"] for rows in by_chr.values() for r in rows]).value_counts().to_dict(),
        ensure_ascii=False
    ))
    logger.info(f"[mask-opt] Interval build done in {time.time() - t0:.1f}s; total_intervals={sum(len(v) for v in by_chr.values()):,}")

    return dict(by_chr)

def ctx_code(seq: str, pos1: int) -> Tuple[str, int]:
    i = pos1 - 1
    if i <= 0 or i >= len(seq) - 1:
        return "NNN", -1
    tri = seq[i - 1:i + 2].upper()
    if len(tri) != 3 or any(b not in BASE2INT for b in tri):
        return tri, -1
    return tri, BASE2INT[tri[0]] * 16 + BASE2INT[tri[1]] * 4 + BASE2INT[tri[2]]

def first_num(x: Optional[str], default=0.0) -> float:
    if x is None:
        return default
    try:
        return float(str(x).split(",")[0])
    except Exception:
        return default

def gene_id_from_feature(ftype: str, attrs: Dict[str, str]) -> Optional[str]:

    for key in ["gene_id", "gene", "locus_tag", "Name"]:
        if key in attrs:
            return clean_gene_id(attrs[key])
    if ftype.lower() in {"gene", "pseudogene"} and "ID" in attrs:
        return clean_gene_id(attrs["ID"])
    for key in ["Parent", "ID"]:
        if key in attrs:
            return clean_gene_id(attrs[key])
    return None

def infer_chr_num(chrom: str, n_chr: int) -> int:
    m = re.search(r"(\d+)$", str(chrom)) or re.search(r"(\d+)", str(chrom))
    if not m:
        return -1
    v = int(m.group(1))
    return ((v - 1) % n_chr) + 1 if v > 0 else -1

def infer_region_subtype(region_type: str, dist_tss: int, donor_offset: int, acceptor_offset: int) -> str:
    if region_type == "promoter":
        d = abs(int(dist_tss)) if pd.notna(dist_tss) else 10**9
        if d <= 500:
            return "promoter_core"
        if d <= 1000:
            return "promoter_proximal"
        return "promoter_distal"
    if region_type == "splice_near":
        return "splice_core" if min(abs(donor_offset), abs(acceptor_offset)) <= 2 else "splice_flank"
    return str(region_type)

def infer_sg(chrom: str, mode: str = "auto") -> Tuple[str, int]:
    if mode == "single":
        return "Single", 0
    m = re.search(r"([AD])\d+", str(chrom).upper())
    if m:
        sg = "At" if m.group(1) == "A" else "Dt"
        return sg, 0 if sg == "At" else 1

    return "Single", 0

def intervals_from_blocks(blocks: List[dict]) -> List[Tuple[int, int]]:
    return [(int(b["start"]), int(b["end"])) for b in blocks]

def iter_coding_missense(gff_obj: dict, genome: Dict[str, str], vcf_idx: dict, call_idx: dict, args) -> Iterator[dict]:
    genes_cds = gff_obj["cds_by_gene"]
    total_genes = len(genes_cds)
    for gi, (gid, cds_blocks) in enumerate(sorted(genes_cds.items()), 1):
        gm = build_cds_map(cds_blocks, genome)
        if gm is None:
            continue
        chrom = gm["Chr"]
        seq = genome.get(chrom)
        if seq is None:
            continue
        strand = gm["strand"]
        cds_seq = gm["cds_seq"]
        cds_to_genome = gm["cds_to_genome"]
        L3 = (len(cds_seq) // 3) * 3
        if L3 < 3:
            continue
        protein = []
        stop_seen = False
        for codon_start in range(0, L3, 3):
            codon = cds_seq[codon_start:codon_start+3].upper()
            aa = CODON_TABLE.get(codon)
            if aa is None or aa == "*":
                stop_seen = True
                break
            protein.append(aa)
        L3 = len(protein) * 3
        if L3 < 3:
            continue
        vcf_chrom = vcf_idx.get(chrom, {})
        sg_str, sg_code = infer_sg(chrom, args.subgenome_mode)
        chr_num = infer_chr_num(chrom, args.n_chr)


        
        for cds_pos in range(L3):
            ref_base = cds_seq[cds_pos].upper()
            if ref_base not in BASE2INT:
                continue
            codon_idx = cds_pos // 3
            pos_in_codon = cds_pos % 3
            ref_codon = cds_seq[codon_idx*3:codon_idx*3+3].upper()
            ref_aa = protein[codon_idx]
            genome_pos = int(cds_to_genome[cds_pos])
            genome_ref = seq[genome_pos - 1].upper()
            if strand == "+":
                if genome_ref != ref_base:
                    continue
                out_ref = ref_base
            else:
                out_ref = ref_base.translate(RC)
                if genome_ref != out_ref:
                    continue
            tri, ccode = ctx_code(seq, genome_pos)
            if ccode < 0:
                continue
            cr = callrate_lookup(call_idx, chrom, np.array([genome_pos], dtype=np.int64))
            for alt_base in ["A", "C", "G", "T"]:
                if alt_base == ref_base:
                    continue
                mc = list(ref_codon)
                mc[pos_in_codon] = alt_base
                alt_aa = CODON_TABLE.get("".join(mc))
                if alt_aa is None or alt_aa == "*" or alt_aa == ref_aa:
                    continue
                out_alt = alt_base if strand == "+" else alt_base.translate(RC)
                if out_alt not in BASE2INT or out_alt == out_ref:
                    continue
                obs, ac, an, af = vcf_annot(vcf_chrom, genome_pos, out_ref, out_alt)
                yield {
                    "SNP_ID": f"{chrom}_{genome_pos}_{out_ref}_{out_alt}",
                    "snp_id": f"{chrom}_{genome_pos}_{out_ref}_{out_alt}",
                    "Chr": chrom, "Pos": genome_pos, "Ref": out_ref, "Alt": out_alt,
                    "GeneID": gid, "gene_id": gid, "Mutation": f"{ref_aa}{codon_idx+1}{alt_aa}",
                    "aa_pos": int(codon_idx), "ref_aa": ref_aa, "alt_aa": alt_aa,
                    "chr_num": chr_num, "subgenome": sg_str, "subgenome_code": sg_code,
                    "Sg_code": sg_code, "Context": tri, "Ctx_code": ccode, "Alt_code": BASE2INT[out_alt],
                    "observed_in_vcf": obs, "Y_Label": obs, "AC": ac, "AN": an, "AF": af,
                    "AN_obs": float(cr["AN_obs"][0]), "callable_n": float(cr["callable_n"][0]),
                    "call_rate": float(cr["call_rate"][0]), "callable_rate": float(cr["callable_rate"][0]),
                    "callrate_hit": int(cr["callrate_hit"][0]),
                }
        if gi % 5000 == 0:
            logger.info(f"  coding genes processed: {gi:,}/{total_genes:,}")

def load_callrate(path: Optional[str], max_callable_n: Optional[int] = None, chrom_regex: Optional[str] = None, parquet_batch_size: int = 2_000_000) -> Dict[str, Tuple[np.ndarray, dict]]:
    """Load callrate as chrom -> (sorted_pos, dict arrays).

    Optimized for chromosome-sharded runs:
      - read only needed columns
      - stream parquet batches
      - keep only chromosomes matching chrom_regex
      - store compact int32/float32 arrays where safe
    """
    if path is None:
        return {}

    logger.info(f"Loading callrate table: {path}")
    if chrom_regex:
        logger.info(f"Callrate chromosome filter: {chrom_regex}")
    pat = re.compile(chrom_regex) if chrom_regex else None
    p = str(path)

    buckets = defaultdict(lambda: defaultdict(list))
    total_seen = 0
    total_kept = 0

    def normalize_and_store(df: pd.DataFrame):
        nonlocal total_seen, total_kept
        total_seen += len(df)
        if len(df) == 0:
            return

        if "Chr" not in df.columns:
            if "chrom" in df.columns:
                df = df.rename(columns={"chrom": "Chr"})
            elif "#CHROM" in df.columns:
                df = df.rename(columns={"#CHROM": "Chr"})
        if "Pos" not in df.columns:
            if "pos" in df.columns:
                df = df.rename(columns={"pos": "Pos"})
            elif "position" in df.columns:
                df = df.rename(columns={"position": "Pos"})

        if "Chr" not in df.columns or "Pos" not in df.columns:
            raise ValueError("callrate table must contain Chr/Pos columns")

        df["Chr"] = df["Chr"].astype(str)
        if pat is not None:
            df = df[df["Chr"].map(lambda x: bool(pat.search(x)))].copy()
            if len(df) == 0:
                return

        df["Pos"] = pd.to_numeric(df["Pos"], errors="coerce").fillna(-1).astype(np.int64)

        for col in ["AN_obs", "callable_n", "callable_rate", "call_rate"]:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df["callable_rate"].isna().all():
            if not df["call_rate"].isna().all():
                df["callable_rate"] = df["call_rate"]
            elif not df["callable_n"].isna().all() and max_callable_n:
                df["callable_rate"] = df["callable_n"] / float(max_callable_n)

        if df["call_rate"].isna().all():
            df["call_rate"] = df["callable_rate"]

        if df["AN_obs"].isna().all() and not df["callable_n"].isna().all():
            df["AN_obs"] = df["callable_n"] * 2

        if df["callable_n"].isna().all() and not df["AN_obs"].isna().all() and max_callable_n:
            df["callable_n"] = np.minimum(max_callable_n, np.rint(df["AN_obs"] / 2.0))

        df = df[df["Pos"] > 0].copy()
        if len(df) == 0:
            return

        total_kept += len(df)

        for chrom, sub in df.groupby("Chr", sort=False):
            buckets[chrom]["Pos"].append(sub["Pos"].to_numpy(np.int64, copy=False))
            buckets[chrom]["AN_obs"].append(sub["AN_obs"].fillna(0).to_numpy(np.float32, copy=False))
            buckets[chrom]["callable_n"].append(sub["callable_n"].fillna(0).to_numpy(np.float32, copy=False))
            buckets[chrom]["callable_rate"].append(sub["callable_rate"].fillna(0).to_numpy(np.float32, copy=False))
            buckets[chrom]["call_rate"].append(sub["call_rate"].fillna(0).to_numpy(np.float32, copy=False))

    if p.endswith(".parquet") or p.endswith(".pq"):
        pf = pq.ParquetFile(p)
        names = pf.schema_arrow.names

        chr_col = "Chr" if "Chr" in names else "chrom" if "chrom" in names else "#CHROM" if "#CHROM" in names else None
        pos_col = "Pos" if "Pos" in names else "pos" if "pos" in names else "position" if "position" in names else None
        if chr_col is None or pos_col is None:
            raise ValueError("callrate parquet must contain Chr/Pos columns")

        read_cols = [chr_col, pos_col] + [c for c in ["AN_obs", "callable_n", "callable_rate", "call_rate"] if c in names]
        read_cols = list(dict.fromkeys(read_cols))

        for rb in pf.iter_batches(batch_size=parquet_batch_size, columns=read_cols):
            normalize_and_store(rb.to_pandas())
            if total_seen % 20_000_000 < parquet_batch_size:
                logger.info(f"  callrate streamed rows seen={total_seen:,}, kept={total_kept:,}")
    else:
        sep = "	" if p.endswith((".tsv", ".tsv.gz", ".bed", ".bed.gz")) else ","
        for df in pd.read_csv(p, sep=sep, chunksize=parquet_batch_size):
            normalize_and_store(df)
            logger.info(f"  callrate streamed rows seen={total_seen:,}, kept={total_kept:,}")

    out = {}
    for chrom, d in buckets.items():
        pos = np.concatenate(d["Pos"])
        order = np.argsort(pos, kind="mergesort")
        pos = pos[order]

        data = {}
        for col in ["AN_obs", "callable_n", "callable_rate", "call_rate"]:
            arr = np.concatenate(d[col])[order]
            data[col] = arr

        if len(pos) > 1:
            keep = np.ones(len(pos), dtype=bool)
            keep[1:] = pos[1:] != pos[:-1]
            pos = pos[keep]
            for col in data:
                data[col] = data[col][keep]

        out[chrom] = (pos, data)

    logger.info(
        f"Callrate loaded: chroms={len(out)}, positions={sum(len(v[0]) for v in out.values()):,}; "
        f"streamed_seen={total_seen:,}, kept={total_kept:,}"
    )
    return out

def load_fasta(path: str, chrom_regex: Optional[str] = None) -> Dict[str, str]:
    logger.info(f"Loading FASTA: {path}")
    seqs: Dict[str, List[str]] = {}
    chrom = None
    pat = re.compile(chrom_regex) if chrom_regex else None
    keep = True
    with open_text(path) as f:
        for line in f:
            if line.startswith(">"):
                name = line[1:].strip().split()[0]
                chrom = name
                keep = True if pat is None else bool(pat.search(name))
                if keep:
                    seqs[chrom] = []
            else:
                if chrom is not None and keep:
                    seqs[chrom].append(line.strip().upper())
    out = {k: "".join(v).upper() for k, v in seqs.items()}
    logger.info(f"Loaded FASTA contigs: {len(out)}; total bp={sum(map(len, out.values())):,}")
    return out

def load_vcf_snv_index(vcf_path: str, chrom_regex: Optional[str] = None) -> Dict[str, Dict[Tuple[int, str, str], Tuple[int, int, float]]]:
    """Load SNV VCF entries as chrom -> (pos, ref, alt) -> (AC, AN, AF).

    Optimized path:
      if chrom_regex is exact and pysam + .tbi are available, query only that chromosome.
    Fallback:
      stream the whole VCF and keep records matching chrom_regex.
    """
    logger.info(f"Loading observed SNVs from VCF: {vcf_path}")
    pat = re.compile(chrom_regex) if chrom_regex else None
    idx: Dict[str, Dict[Tuple[int, str, str], Tuple[int, int, float]]] = defaultdict(dict)

    def consume_record(parts):
        if len(parts) < 8:
            return 0
        chrom, pos, vid, ref, alts, qual, flt, info = parts[:8]
        if pat is not None and not pat.search(chrom):
            return 0
        ref = ref.upper()
        if len(ref) != 1 or ref not in BASE2INT:
            return 0
        pos_i = int(pos)
        info_d = parse_info(info)
        ac_vals = str(info_d.get("AC", "")).split(",") if "AC" in info_d else []
        af_vals = str(info_d.get("AF", "")).split(",") if "AF" in info_d else []
        an = int(first_num(info_d.get("AN"), 0))
        added = 0
        for j, alt in enumerate(alts.split(",")):
            alt = alt.upper()
            if len(alt) != 1 or alt not in BASE2INT or alt == ref:
                continue
            ac = int(first_num(ac_vals[j] if j < len(ac_vals) and ac_vals[j] != "" else None, 1))
            af = float(first_num(af_vals[j] if j < len(af_vals) and af_vals[j] != "" else None, (ac / an if an > 0 else np.nan)))
            idx[chrom][(pos_i, ref, alt)] = (ac, an, af)
            added += 1
        return added

    exact_chroms = _exact_chroms_from_regex(chrom_regex)

    if exact_chroms:
        try:
            import pysam
            tbx = pysam.TabixFile(vcf_path)
            n_records = n_snv = 0
            for chrom in exact_chroms:
                try:
                    iterator = tbx.fetch(chrom)
                except Exception as e:
                    logger.warning(f"pysam fetch failed for chrom={chrom}: {e}; falling back to full scan.")
                    raise
                for line in iterator:
                    n_records += 1
                    n_snv += consume_record(line.rstrip("\n").split("\t"))
                    if n_records % 2_000_000 == 0:
                        logger.info(f"  tabix VCF records={n_records:,}, SNV alleles={n_snv:,}")
            logger.info(f"Observed VCF SNV alleles loaded by tabix: chroms={len(idx)}, alleles={n_snv:,}")
            return dict(idx)
        except Exception as e:
            logger.warning(f"pysam/tabix path unavailable; fallback to streaming full VCF. reason={e}")

    n_records = n_snv = 0
    with open_text(vcf_path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            n_records += 1
            n_snv += consume_record(line.rstrip("\n").split("\t"))
            if n_records % 2_000_000 == 0:
                logger.info(f"  parsed VCF records={n_records:,}, SNV alleles={n_snv:,}")

    logger.info(f"Observed VCF SNV alleles loaded: chroms={len(idx)}, alleles={n_snv:,}")
    return dict(idx)

def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    ints = sorted((max(1, int(a)), int(b)) for a, b in intervals if b >= a)
    out = [ints[0]]
    for a, b in ints[1:]:
        la, lb = out[-1]
        if a <= lb + 1:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out

def open_text(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")

def parse_attrs(attr: str) -> Dict[str, str]:
    d = {}

    for x in attr.strip().split(";"):
        x = x.strip()
        if not x:
            continue
        if "=" in x:
            k, v = x.split("=", 1)
            d[k.strip()] = v.strip().strip('"')
        else:
            parts = x.split(None, 1)
            if len(parts) == 2:
                d[parts[0].strip()] = parts[1].strip().strip('"')
    return d

def parse_gff(path: str, fasta_chroms: set) -> dict:
    """
    Parse GFF/GTF into gene-level CDS/UTR/exon blocks.

    This patch preserves the original label/feature semantics.
    It only improves transcript -> gene mapping across species.

    Supported:
      - standard: mRNA/transcript ID=tx; Parent=gene
      - direct: CDS/UTR/exon Parent=gene
      - rice IRGSP: mRNA ID=OsXXt...; Locus_id=OsXXg...
      - rice fallback: gene attribute "Transcript variants=OsXXt..."
    """
    logger.info(f"Parsing GFF/GTF: {path}")
    genes = {}
    cds_by_gene = defaultdict(list)
    utr5_by_gene = defaultdict(list)
    utr3_by_gene = defaultdict(list)
    exon_by_gene = defaultdict(list)
    feature_counts = defaultdict(int)

    def _split_ids(x):
        if not x:
            return []
        out = []
        for y in re.split(r"[, ]+", str(x).strip()):
            y = y.strip()
            if y:
                out.append(clean_gene_id(y))
        return [x for x in out if x]


    tx_to_gene = {}
    with open_text(path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue

            chrom, source, ftype, start, end, score, strand, phase, attr = parts
            attrs = parse_attrs(attr)
            ft = ftype.lower()

            if ft in {"gene", "pseudogene"}:
                gid = None
                for key in ["gene_id", "gene", "locus_tag", "ID", "Name"]:
                    if key in attrs:
                        gid = clean_gene_id(attrs[key])
                        break

                if gid:

                    
                    for key in ["Transcript variants", "transcript_variants", "transcript_id"]:
                        if key in attrs:
                            for tx in _split_ids(attrs[key]):
                                tx_to_gene[tx] = gid

            elif ft in {"mrna", "transcript", "rna"}:
                tx = None
                for key in ["ID", "transcript_id", "Name"]:
                    if key in attrs:
                        tx = clean_gene_id(attrs[key])
                        break

                parent = None
                for key in ["Parent", "Locus_id", "locus_id", "gene_id", "gene"]:
                    if key in attrs:
                        parent = clean_gene_id(attrs[key])
                        break

                if tx and parent:
                    tx_to_gene[tx] = parent

    def _feature_gene_id(ftype: str, attrs: dict):
        ft = ftype.lower()

        if ft in {"gene", "pseudogene"}:
            for key in ["gene_id", "gene", "locus_tag", "ID", "Name"]:
                if key in attrs:
                    return clean_gene_id(attrs[key])
            return None

        if ft in {
            "cds", "exon",
            "five_prime_utr", "5utr", "5'utr", "utr5",
            "three_prime_utr", "3utr", "3'utr", "utr3",
        }:
            raw = None
            for key in ["Parent", "gene_id", "gene", "locus_tag", "ID", "Name"]:
                if key in attrs:
                    raw = clean_gene_id(attrs[key])
                    break
            if raw is None:
                return None
            return tx_to_gene.get(raw, raw)

        return gene_id_from_feature(ftype, attrs)


    with open_text(path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue

            chrom, source, ftype, start, end, score, strand, phase, attr = parts

            try:
                start_i, end_i = int(start), int(end)
            except Exception:
                continue

            attrs = parse_attrs(attr)
            gid = _feature_gene_id(ftype, attrs)
            if gid is None:
                continue

            feature_counts[ftype] += 1
            rec = {
                "Chr": chrom,
                "start": start_i,
                "end": end_i,
                "strand": strand,
                "phase": phase,
                "gene_id": gid,
            }
            ft = ftype.lower()

            if ft == "gene":
                old = genes.get(gid)
                if old is None:
                    genes[gid] = rec.copy()
                else:
                    old["start"] = min(old["start"], start_i)
                    old["end"] = max(old["end"], end_i)

            elif ft == "cds":
                cds_by_gene[gid].append(rec)

            elif ft in {"five_prime_utr", "5utr", "5'utr", "utr5"}:
                utr5_by_gene[gid].append(rec)

            elif ft in {"three_prime_utr", "3utr", "3'utr", "utr3"}:
                utr3_by_gene[gid].append(rec)

            elif ft == "exon":
                exon_by_gene[gid].append(rec)


    all_gids = (
        set(genes)
        | set(cds_by_gene)
        | set(exon_by_gene)
        | set(utr5_by_gene)
        | set(utr3_by_gene)
    )

    for gid in all_gids:
        blocks = (
            cds_by_gene.get(gid, [])
            + exon_by_gene.get(gid, [])
            + utr5_by_gene.get(gid, [])
            + utr3_by_gene.get(gid, [])
        )
        if not blocks:
            continue

        chrom = blocks[0]["Chr"]
        strand = blocks[0]["strand"]
        mn = min(b["start"] for b in blocks)
        mx = max(b["end"] for b in blocks)

        if gid not in genes:
            genes[gid] = {
                "Chr": chrom,
                "start": mn,
                "end": mx,
                "strand": strand,
                "phase": ".",
                "gene_id": gid,
            }
        else:
            genes[gid]["start"] = min(genes[gid]["start"], mn)
            genes[gid]["end"] = max(genes[gid]["end"], mx)

    for gid, blocks in cds_by_gene.items():
        strand = blocks[0]["strand"] if blocks else "+"
        blocks.sort(key=lambda x: x["start"], reverse=(strand == "-"))

    logger.info(
        f"GFF parsed: genes={len(genes):,}, genes_with_CDS={len(cds_by_gene):,}, "
        f"UTR5_genes={len(utr5_by_gene):,}, UTR3_genes={len(utr3_by_gene):,}, "
        f"tx_to_gene={len(tx_to_gene):,}"
    )

    return {
        "genes": genes,
        "cds_by_gene": dict(cds_by_gene),
        "utr5_by_gene": dict(utr5_by_gene),
        "utr3_by_gene": dict(utr3_by_gene),
        "exon_by_gene": dict(exon_by_gene),
        "feature_counts": dict(feature_counts),
    }

def parse_info(info: str) -> Dict[str, str]:
    d = {}
    if not info or info == ".":
        return d
    for item in info.split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            d[k] = v
        else:
            d[item] = "1"
    return d

def subtract_intervals(intervals: List[Tuple[int, int]], blockers: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    intervals = merge_intervals(intervals)
    blockers = merge_intervals(blockers)
    out = []
    j = 0
    for a, b in intervals:
        cur = a
        while j < len(blockers) and blockers[j][1] < a:
            j += 1
        k = j
        while k < len(blockers) and blockers[k][0] <= b:
            ba, bb = blockers[k]
            if ba > cur:
                out.append((cur, min(b, ba - 1)))
            cur = max(cur, bb + 1)
            if cur > b:
                break
            k += 1
        if cur <= b:
            out.append((cur, b))
    return out

def vcf_annot(vcf_chrom: dict, pos: int, ref: str, alt: str) -> Tuple[int, int, int, float]:
    if vcf_chrom:
        val = vcf_chrom.get((int(pos), ref, alt))
        if val is not None:
            ac, an, af = val
            return 1, int(ac), int(an), float(af)
    return 0, 0, 0, 0.0

def write_iterator_to_parquet(iterator: Iterator[dict], path: Path, chunk_rows: int, max_rows: Optional[int] = None) -> int:
    w = ParquetChunkWriter(path, chunk_rows=chunk_rows)
    n = 0
    try:
        for rec in iterator:
            w.add(rec)
            n += 1
            if max_rows is not None and n >= max_rows:
                logger.warning(f"Reached --max-rows-per-mode={max_rows}; stopping early for this mode.")
                break
    finally:
        w.close()
    return n


class ParquetChunkWriter:
    def __init__(self, path: Path, chunk_rows: int = 500_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.chunk_rows = int(chunk_rows)
        self.buf: List[dict] = []
        self.writer: Optional[pq.ParquetWriter] = None
        self.rows = 0

    def add(self, rec: dict):
        self.buf.append(rec)
        if len(self.buf) >= self.chunk_rows:
            self.flush()

    def add_many(self, recs: List[dict]):
        self.buf.extend(recs)
        if len(self.buf) >= self.chunk_rows:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        df = pd.DataFrame(self.buf)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="snappy")
        self.writer.write_table(table)
        self.rows += len(df)
        logger.info(f"  wrote {self.rows:,} rows -> {self.path.name}")
        self.buf.clear()

    def close(self):
        self.flush()
        if self.writer is not None:
            self.writer.close()
