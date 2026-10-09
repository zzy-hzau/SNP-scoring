#!/usr/bin/env python3

"make_candidate_positions_from_ref_gff_v2.py"
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import tempfile
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

BASES = set("ACGT")
CODON_TABLE = {
    "TTT":"F","TTC":"F","TTA":"L","TTG":"L","CTT":"L","CTC":"L","CTA":"L","CTG":"L",
    "ATT":"I","ATC":"I","ATA":"I","ATG":"M","GTT":"V","GTC":"V","GTA":"V","GTG":"V",
    "TCT":"S","TCC":"S","TCA":"S","TCG":"S","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "ACT":"T","ACC":"T","ACA":"T","ACG":"T","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*","CAT":"H","CAC":"H","CAA":"Q","CAG":"Q",
    "AAT":"N","AAC":"N","AAA":"K","AAG":"K","GAT":"D","GAC":"D","GAA":"E","GAG":"E",
    "TGT":"C","TGC":"C","TGA":"*","TGG":"W","CGT":"R","CGC":"R","CGA":"R","CGG":"R",
    "AGT":"S","AGC":"S","AGA":"R","AGG":"R","GGT":"G","GGC":"G","GGA":"G","GGG":"G",
}
RCMAP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def open_text(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r")

def fasta_iter(path: str):
    name = None
    chunks = []
    with open_text(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks).upper()
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if name is not None:
            yield name, "".join(chunks).upper()

def load_fasta(path: str, min_seq_len: int) -> Dict[str, str]:
    genome = {}
    skipped = 0
    for name, seq in fasta_iter(path):
        if len(seq) >= min_seq_len:
            genome[name] = seq
        else:
            skipped += 1
    log(f"loaded FASTA main sequences={len(genome)} skipped_short={skipped}")
    return genome

def parse_attrs(attr: str) -> Dict[str, str]:
    out = {}
    for item in attr.strip().split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
        elif " " in item:
            k, v = item.split(" ", 1)
            out[k.strip()] = v.strip().strip('"')
    return out

def clean_id(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    x = str(x).split(",")[0].strip().strip('"')
    for p in ("transcript:", "mRNA:", "rna:", "CDS:", "cds:", "gene:"):
        if x.startswith(p):
            x = x[len(p):]
    return x

def norm_chr_key(x: str) -> str:
    return re.sub(r"^(chr|Chr|CHR)", "", str(x).strip())

def resolve_chrom(chrom: str, genome_names: set[str]) -> Optional[str]:
    if chrom in genome_names:
        return chrom
    candidates = [chrom]
    c = str(chrom)
    if c.startswith("chr"):
        candidates += [c[3:], "Chr" + c[3:]]
    elif c.startswith("Chr"):
        candidates += [c[3:], "chr" + c[3:]]
    else:
        candidates += ["chr" + c, "Chr" + c]
    nk = norm_chr_key(c)
    for g in genome_names:
        if norm_chr_key(g) == nk:
            candidates.append(g)
    for x in candidates:
        if x in genome_names:
            return x
    return None

def revcomp(seq: str) -> str:
    return seq.translate(RCMAP)[::-1].upper()

def atomic_write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        json.dump(obj, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)

def atomic_write_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, path)

def parse_gff(gff: str, genome_names: set[str]):
    feature_counts = Counter()
    genes_by_chrom = defaultdict(list)
    cds_by_tx = defaultdict(list)
    exon_by_tx = defaultdict(list)
    seqid_unmatched = Counter()
    exon_types = {"exon", "five_prime_UTR", "three_prime_UTR", "UTR"}
    with open_text(gff) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom0, source, ftype, start, end, score, strand, phase, attr = parts
            try:
                start_i, end_i = int(start), int(end)
            except Exception:
                continue
            chrom = resolve_chrom(chrom0, genome_names)
            if chrom is None:
                seqid_unmatched[chrom0] += 1
                continue
            attrs = parse_attrs(attr)
            feature_counts[ftype] += 1
            if ftype == "gene":
                gid = clean_id(attrs.get("ID") or attrs.get("gene_id") or attrs.get("Name")) or f"{chrom}:{start_i}-{end_i}"
                genes_by_chrom[chrom].append((start_i, end_i, strand, gid))
            if ftype == "CDS":
                parent = clean_id(attrs.get("Parent") or attrs.get("transcript_id") or attrs.get("gene_id") or attrs.get("ID"))
                if parent:
                    cds_by_tx[parent].append((chrom, start_i, end_i, strand))
            if ftype in exon_types:
                parent = clean_id(attrs.get("Parent") or attrs.get("transcript_id") or attrs.get("gene_id") or attrs.get("ID"))
                if parent:
                    exon_by_tx[parent].append((chrom, start_i, end_i, strand, ftype))
    for tx, xs in list(cds_by_tx.items()):
        if not xs:
            continue
        strand = xs[0][3]
        cds_by_tx[tx] = sorted(xs, key=lambda x: x[1], reverse=(strand == "-"))
    return genes_by_chrom, cds_by_tx, exon_by_tx, dict(feature_counts), dict(seqid_unmatched)

def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted((int(a), int(b)) for a, b in intervals if int(a) <= int(b))
    out = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce + 1:
            ce = max(ce, e)
        else:
            out.append((cs, ce))
            cs, ce = s, e
    out.append((cs, ce))
    return out

def subtract_intervals(base: List[Tuple[int, int]], mask: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not base:
        return []
    mask = merge_intervals(mask)
    out = []
    j = 0
    for s, e in merge_intervals(base):
        cur = s
        while j < len(mask) and mask[j][1] < s:
            j += 1
        k = j
        while k < len(mask) and mask[k][0] <= e:
            ms, me = mask[k]
            if ms > cur:
                out.append((cur, min(e, ms - 1)))
            cur = max(cur, me + 1)
            if cur > e:
                break
            k += 1
        if cur <= e:
            out.append((cur, e))
    return out

def has_missense_position(codon: str, offset: int) -> bool:
    ref_aa = CODON_TABLE.get(codon)
    if ref_aa is None or ref_aa == "*":
        return False
    ref = codon[offset]
    for alt in "ACGT":
        if alt == ref:
            continue
        m = list(codon)
        m[offset] = alt
        aa = CODON_TABLE.get("".join(m))
        if aa is not None and aa != "*" and aa != ref_aa:
            return True
    return False

def build_coding_positions(genome: Dict[str, str], cds_by_tx):
    pos_sets = defaultdict(set)
    stats = Counter()
    for tx in sorted(cds_by_tx):
        cds_list = cds_by_tx[tx]
        if not cds_list:
            continue
        chrom = cds_list[0][0]
        strand = cds_list[0][3]
        if chrom not in genome:
            stats["chrom_missing"] += 1
            continue
        if any(x[0] != chrom or x[3] != strand for x in cds_list):
            stats["mixed_chrom_or_strand"] += 1
            continue
        chrom_seq = genome[chrom]
        seq_parts = []
        pos_parts = []
        ok = True
        for _, start, end, _ in cds_list:
            if start < 1 or end > len(chrom_seq) or start > end:
                ok = False
                break
            frag = chrom_seq[start-1:end]
            if strand == "-":
                frag = revcomp(frag)
                pos = list(range(end, start-1, -1))
            else:
                pos = list(range(start, end+1))
            seq_parts.append(frag)
            pos_parts.extend(pos)
        if not ok:
            stats["out_of_bounds"] += 1
            continue
        seq = "".join(seq_parts).upper()
        if len(seq) < 3:
            stats["too_short"] += 1
            continue
        if len(seq) % 3 != 0:
            stats["mod3_bad"] += 1
            continue
        if any(b not in BASES for b in seq):
            stats["bad_base"] += 1
            continue
        L = len(seq)
        internal_stop = False
        for i in range(0, L - 3, 3):
            if CODON_TABLE.get(seq[i:i+3]) == "*":
                internal_stop = True
                break
        if internal_stop:
            stats["internal_stop"] += 1
            continue
        stats["valid_tx"] += 1
        for i in range(0, L, 3):
            codon = seq[i:i+3]
            if CODON_TABLE.get(codon) in (None, "*"):
                continue
            for off in (0, 1, 2):
                if has_missense_position(codon, off):
                    pos_sets[chrom].add(int(pos_parts[i+off]))
    return {c: np.fromiter(sorted(v), dtype=np.int64) for c, v in sorted(pos_sets.items())}, dict(stats)

def write_position_cache(pos_by_chr: Dict[str, np.ndarray], prefix: str, outdir: Path):
    meta = {
        "chr_order": list(pos_by_chr.keys()),
        "site_counts": {c: int(len(a)) for c, a in pos_by_chr.items()},
        "n_sites_total": int(sum(len(a) for a in pos_by_chr.values())),
        "cache_type": "position_only",
        "format_version": 2,
    }
    atomic_write_npz(outdir / f"{prefix}.pos_by_chr.npz", **{c: np.asarray(a, dtype=np.int64) for c, a in pos_by_chr.items()})
    atomic_write_json(meta, outdir / f"{prefix}.meta.json")
    return meta

def build_regulatory_intervals(genome: Dict[str, str], genes_by_chrom, cds_by_tx, promoter_upstream: int):
    regulatory = defaultdict(list)
    cds_intervals = defaultdict(list)
    for tx, cds_list in cds_by_tx.items():
        for chrom, start, end, strand in cds_list:
            if chrom in genome:
                cds_intervals[chrom].append((start, end))
    for chrom, genes in genes_by_chrom.items():
        if chrom not in genome:
            continue
        chrom_len = len(genome[chrom])
        for start, end, strand, gid in genes:
            regulatory[chrom].append((max(1, start), min(chrom_len, end)))
            if strand == "-":
                ps, pe = end + 1, min(chrom_len, end + promoter_upstream)
            else:
                ps, pe = max(1, start - promoter_upstream), start - 1
            if ps <= pe:
                regulatory[chrom].append((ps, pe))
    regulatory = {c: merge_intervals(v) for c, v in regulatory.items()}
    cds_intervals = {c: merge_intervals(v) for c, v in cds_intervals.items()}
    return {c: subtract_intervals(v, cds_intervals.get(c, [])) for c, v in regulatory.items()}, cds_intervals

def positions_from_intervals_acgt(seq: str, intervals: List[Tuple[int, int]], chunk_bases: int = 5_000_000) -> np.ndarray:
    parts = []
    seq_len = len(seq)
    for s, e in intervals:
        s, e = max(1, int(s)), min(seq_len, int(e))
        if s > e:
            continue
        cur = s
        while cur <= e:
            end = min(e, cur + chunk_bases - 1)
            sub = np.frombuffer(seq[cur-1:end].encode("ascii"), dtype="S1")
            keep = np.isin(sub, [b"A", b"C", b"G", b"T"])
            if np.any(keep):
                pos = np.arange(cur, end+1, dtype=np.int64)
                parts.append(pos[keep])
            cur = end + 1
    if not parts:
        return np.array([], dtype=np.int64)
    arr = np.concatenate(parts).astype(np.int64, copy=False)
    return np.unique(arr) if arr.size > 1 else arr

def build_noncoding_positions(genome: Dict[str, str], genes_by_chrom, cds_by_tx, promoter_upstream: int):
    intervals, cds_intervals = build_regulatory_intervals(genome, genes_by_chrom, cds_by_tx, promoter_upstream)
    pos_by_chr = {}
    interval_counts = {}
    for chrom in sorted(intervals):
        log(f"noncoding positions: {chrom} intervals={len(intervals[chrom])}")
        arr = positions_from_intervals_acgt(genome[chrom], intervals[chrom])
        pos_by_chr[chrom] = arr
        interval_counts[chrom] = len(intervals[chrom])
        log(f"noncoding positions: {chrom} sites={arr.size:,}")
    return pos_by_chr, interval_counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--gff", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-seq-len", type=int, default=1_000_000)
    ap.add_argument("--include-seqids", default=None,
                    help="Comma-separated FASTA/GFF seqids to keep, e.g. chr01,chr02 or 1,2,3. Overrides min-seq-len filtering after FASTA load.")
    ap.add_argument("--noncoding-mode", choices=["regulatory"], default="regulatory")
    ap.add_argument("--promoter-upstream", type=int, default=2000)
    ap.add_argument("--subgenome-mode", choices=["none"], default="none")
    args = ap.parse_args()

    t0 = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ncdir = outdir / "noncoding_cache1"
    ncdir.mkdir(parents=True, exist_ok=True)

    log(f"species={args.species}")
    genome = load_fasta(args.fasta, args.min_seq_len)

    if args.include_seqids:
        keep = [x.strip() for x in args.include_seqids.split(",") if x.strip()]
        missing = [x for x in keep if x not in genome]
        if missing:
            raise RuntimeError(f"--include-seqids contains IDs not loaded from FASTA after min-length filtering: {missing}")
        genome = {x: genome[x] for x in keep}

    if not genome:
        raise RuntimeError("No FASTA sequence passed --min-seq-len / --include-seqids.")
    genes_by_chrom, cds_by_tx, exon_by_tx, feature_counts, seqid_unmatched = parse_gff(args.gff, set(genome))
    log(f"GFF feature_counts={feature_counts}")
    log(f"genes={sum(len(v) for v in genes_by_chrom.values()):,} cds_groups={len(cds_by_tx):,} unmatched_seqids={len(seqid_unmatched)}")

    log("building coding missense-capable positions ...")
    coding_pos, coding_stats = build_coding_positions(genome, cds_by_tx)
    coding_meta = write_position_cache(coding_pos, "candidate_index", outdir)
    rows = [pd.DataFrame({"Chr": np.repeat(chrom, len(arr)), "Pos": arr.astype(np.int64)}) for chrom, arr in coding_pos.items()]
    cand = pd.concat(rows, ignore_index=True).sort_values(["Chr", "Pos"], kind="mergesort") if rows else pd.DataFrame(columns=["Chr", "Pos"])
    cand.to_csv(outdir / "candidate_sites.tsv", sep="\t", index=False)
    log(f"coding candidate_sites.tsv rows={len(cand):,}")

    log("building noncoding regulatory positions ...")
    nc_pos, nc_interval_counts = build_noncoding_positions(genome, genes_by_chrom, cds_by_tx, args.promoter_upstream)
    nc_meta = write_position_cache(nc_pos, "noncoding_index", ncdir)
    nc_meta["definition"] = f"gene body + {args.promoter_upstream}bp promoter, excluding CDS, A/C/G/T only"
    atomic_write_json(nc_meta, ncdir / "noncoding_index.meta.json")

    summary = {
        "species": args.species,
        "fasta": args.fasta,
        "gff": args.gff,
        "min_seq_len": args.min_seq_len,
        "noncoding_mode": args.noncoding_mode,
        "promoter_upstream": args.promoter_upstream,
        "feature_counts": feature_counts,
        "seqid_unmatched_top": sorted(seqid_unmatched.items(), key=lambda x: -x[1])[:30],
        "coding": {"stats": coding_stats, "n_sites_total": coding_meta["n_sites_total"], "site_counts": coding_meta["site_counts"]},
        "noncoding": {"n_sites_total": nc_meta["n_sites_total"], "site_counts": nc_meta["site_counts"], "interval_counts": nc_interval_counts, "definition": nc_meta["definition"]},
        "elapsed_sec": time.time() - t0,
    }
    atomic_write_json(summary, outdir / "build_summary.json")
    log(json.dumps(summary, indent=2, ensure_ascii=False)[:5000])
    log("done")

if __name__ == "__main__":
    main()
