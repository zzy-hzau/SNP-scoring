#!/usr/bin/env python3

"preflight_ref_gff_qc.py"
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Optional

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

def load_fasta(path: str, min_len: int):
    genome = {}
    lengths = {}
    for name, seq in fasta_iter(path):
        lengths[name] = len(seq)
        if len(seq) >= min_len:
            genome[name] = seq
    return genome, lengths

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
    s = str(x).strip()
    return re.sub(r"^(chr|Chr|CHR)", "", s)

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

def parse_gff(gff: str):
    feature_counts = Counter()
    seqid_counts = Counter()
    genes = []
    tx_seen = set()
    cds_by_tx = defaultdict(list)
    exon_by_tx = defaultdict(list)
    bad_rows = 0
    tx_types = {"mRNA", "transcript", "lnc_RNA", "ncRNA", "rRNA", "tRNA", "miRNA", "primary_transcript"}
    exon_types = {"exon", "five_prime_UTR", "three_prime_UTR", "UTR"}
    with open_text(gff) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                bad_rows += 1
                continue
            chrom, source, ftype, start, end, score, strand, phase, attr = parts
            try:
                start_i, end_i = int(start), int(end)
            except Exception:
                bad_rows += 1
                continue
            attrs = parse_attrs(attr)
            feature_counts[ftype] += 1
            seqid_counts[chrom] += 1
            if ftype == "gene":
                gid = clean_id(attrs.get("ID") or attrs.get("gene_id") or attrs.get("Name"))
                genes.append((chrom, start_i, end_i, strand, gid or "NA"))
            if ftype in tx_types:
                tid = clean_id(attrs.get("ID") or attrs.get("transcript_id") or attrs.get("Name"))
                if tid:
                    tx_seen.add(tid)
            if ftype == "CDS":
                parent = clean_id(attrs.get("Parent") or attrs.get("transcript_id") or attrs.get("gene_id") or attrs.get("ID"))
                if parent:
                    try:
                        phase_i = int(phase) if phase not in (".", "") else None
                    except Exception:
                        phase_i = None
                    cds_by_tx[parent].append((chrom, start_i, end_i, strand, phase_i))
            if ftype in exon_types:
                parent = clean_id(attrs.get("Parent") or attrs.get("transcript_id") or attrs.get("gene_id") or attrs.get("ID"))
                if parent:
                    exon_by_tx[parent].append((chrom, start_i, end_i, strand, ftype))
    return {
        "feature_counts": dict(feature_counts),
        "seqid_counts": dict(seqid_counts),
        "genes": genes,
        "tx_seen": tx_seen,
        "cds_by_tx": cds_by_tx,
        "exon_by_tx": exon_by_tx,
        "bad_rows": bad_rows,
    }

def cds_sequence_for_tx(tx: str, cds_list, genome: Dict[str, str]):
    if not cds_list:
        return None, None, None, "no_cds"
    chrom0 = cds_list[0][0]
    chrom = resolve_chrom(chrom0, set(genome.keys()))
    if chrom is None:
        return None, None, None, "chrom_unmatched"
    strand = cds_list[0][3]
    if strand not in ("+", "-"):
        return None, None, None, "bad_strand"
    for c, s, e, st, ph in cds_list:
        rc = resolve_chrom(c, set(genome.keys()))
        if rc != chrom or st != strand:
            return None, None, None, "mixed_chrom_or_strand"
    cds_sorted = sorted(cds_list, key=lambda x: x[1], reverse=(strand == "-"))
    seq_parts = []
    pos_parts = []
    chrom_seq = genome[chrom]
    for c, start, end, st, phase in cds_sorted:
        if start < 1 or end > len(chrom_seq) or start > end:
            return None, None, None, "cds_out_of_bounds"
        frag = chrom_seq[start-1:end]
        if strand == "-":
            frag = revcomp(frag)
            pos = list(range(end, start-1, -1))
        else:
            pos = list(range(start, end+1))
        seq_parts.append(frag)
        pos_parts.extend(pos)
    return chrom, "".join(seq_parts).upper(), pos_parts, "ok"

def translate(seq: str):
    aas = []
    L = (len(seq) // 3) * 3
    for i in range(0, L, 3):
        aas.append(CODON_TABLE.get(seq[i:i+3], "X"))
    return "".join(aas)

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--gff", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-seq-len", type=int, default=1_000_000)
    ap.add_argument("--max-transcripts-for-deep-qc", type=int, default=0)
    ap.add_argument("--max-internal-stop-rate", type=float, default=0.10)
    ap.add_argument("--min-valid-cds-tx", type=int, default=1000)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] species={args.species}")
    print("[INFO] loading FASTA ...", flush=True)
    genome, all_lengths = load_fasta(args.fasta, args.min_seq_len)
    genome_names = set(genome.keys())

    print("[INFO] parsing GFF ...", flush=True)
    gff_obj = parse_gff(args.gff)
    cds_by_tx = gff_obj["cds_by_tx"]

    gff_seqids = set(gff_obj["seqid_counts"])
    matched, unmatched = {}, []
    for s in sorted(gff_seqids):
        r = resolve_chrom(s, genome_names)
        if r is None:
            unmatched.append(s)
        else:
            matched[s] = r

    print("[INFO] deep CDS QC ...", flush=True)
    tx_ids = sorted(cds_by_tx)
    if args.max_transcripts_for_deep_qc and args.max_transcripts_for_deep_qc > 0:
        tx_ids = tx_ids[:args.max_transcripts_for_deep_qc]

    qc_rows = []
    n_valid = n_mod3_bad = n_internal_stop = n_chrom_bad = n_oob = n_mixed = n_bad_base = 0
    coding_candidate_positions = set()
    for tx in tx_ids:
        chrom, seq, pos, status = cds_sequence_for_tx(tx, cds_by_tx[tx], genome)
        if status != "ok":
            if status == "chrom_unmatched":
                n_chrom_bad += 1
            elif status == "cds_out_of_bounds":
                n_oob += 1
            elif status == "mixed_chrom_or_strand":
                n_mixed += 1
            qc_rows.append((tx, status, None, None, None, None, None))
            continue
        bad_base = any(b not in BASES for b in seq)
        if bad_base:
            n_bad_base += 1
        mod3 = len(seq) % 3
        aa = translate(seq)
        internal_stop = ("*" in aa[:-1]) if len(aa) > 1 else False
        if mod3 != 0:
            n_mod3_bad += 1
        if internal_stop:
            n_internal_stop += 1
        valid = (mod3 == 0 and not internal_stop and not bad_base)
        if valid:
            n_valid += 1
            L = (len(seq) // 3) * 3
            for i in range(0, L, 3):
                codon = seq[i:i+3]
                if len(codon) != 3 or any(b not in BASES for b in codon):
                    continue
                for off in range(3):
                    if has_missense_position(codon, off):
                        coding_candidate_positions.add((chrom, int(pos[i+off])))
        qc_rows.append((tx, "ok", len(seq), mod3, int(internal_stop), int(valid), chrom))

    total_deep = len(tx_ids)
    internal_stop_rate = n_internal_stop / max(total_deep, 1)
    mod3_bad_rate = n_mod3_bad / max(total_deep, 1)
    status = "PASS"
    reasons = []
    if len(genome) == 0:
        status = "FAIL"; reasons.append("No FASTA sequence passed min length.")
    if len(cds_by_tx) == 0:
        status = "FAIL"; reasons.append("No CDS grouped by transcript/gene was found in GFF.")
    if len(unmatched) > 0 and len(matched) == 0:
        status = "FAIL"; reasons.append("No GFF seqid can be matched to FASTA sequence id.")
    if n_valid < args.min_valid_cds_tx:
        status = "FAIL"; reasons.append(f"Valid CDS transcript count too low: {n_valid} < {args.min_valid_cds_tx}.")
    if internal_stop_rate > args.max_internal_stop_rate:
        if status == "PASS":
            status = "WARN"
        reasons.append(f"Internal stop rate high: {internal_stop_rate:.4f} > {args.max_internal_stop_rate}. Check GFF/reference match.")

    summary = {
        "species": args.species,
        "status": status,
        "reasons": reasons,
        "fasta": {
            "path": args.fasta,
            "n_sequences_total": len(all_lengths),
            "n_sequences_min_len": len(genome),
            "total_bp_all": int(sum(all_lengths.values())),
            "total_bp_min_len": int(sum(len(x) for x in genome.values())),
            "top_sequences": sorted(all_lengths.items(), key=lambda x: x[1], reverse=True)[:30],
        },
        "gff": {
            "path": args.gff,
            "bad_rows": gff_obj["bad_rows"],
            "feature_counts": gff_obj["feature_counts"],
            "n_seqids": len(gff_seqids),
            "seqid_top_counts": sorted(gff_obj["seqid_counts"].items(), key=lambda x: x[1], reverse=True)[:50],
            "n_genes": len(gff_obj["genes"]),
            "n_transcript_features": len(gff_obj["tx_seen"]),
            "n_cds_groups": len(cds_by_tx),
            "n_exon_groups": len(gff_obj["exon_by_tx"]),
        },
        "seqid_match": {
            "n_gff_seqids_matched": len(matched),
            "n_gff_seqids_unmatched": len(unmatched),
            "unmatched_examples": unmatched[:50],
        },
        "cds_deep_qc": {
            "n_tx_checked": total_deep,
            "n_valid_cds_tx": n_valid,
            "n_mod3_bad": n_mod3_bad,
            "mod3_bad_rate": mod3_bad_rate,
            "n_internal_stop": n_internal_stop,
            "internal_stop_rate": internal_stop_rate,
            "n_chrom_unmatched_tx": n_chrom_bad,
            "n_cds_out_of_bounds_tx": n_oob,
            "n_mixed_chrom_or_strand_tx": n_mixed,
            "n_bad_base_tx": n_bad_base,
            "rough_coding_candidate_positions_valid_tx": int(len(coding_candidate_positions)),
        },
    }
    with open(outdir / f"{args.species}.ref_gff_qc.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(outdir / f"{args.species}.feature_counts.tsv", "w", encoding="utf-8") as f:
        f.write("feature\tcount\n")
        for k, v in sorted(gff_obj["feature_counts"].items(), key=lambda x: (-x[1], x[0])):
            f.write(f"{k}\t{v}\n")
    with open(outdir / f"{args.species}.cds_tx_qc.tsv", "w", encoding="utf-8") as f:
        f.write("tx_id\tstatus\tcds_len\tmod3\tinternal_stop\tvalid\tchrom\n")
        for row in qc_rows:
            f.write("\t".join("" if x is None else str(x) for x in row) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if status == "FAIL":
        sys.exit(2)
    print(f"[INFO] QC status={status}. Outputs: {outdir}", flush=True)

if __name__ == "__main__":
    main()
