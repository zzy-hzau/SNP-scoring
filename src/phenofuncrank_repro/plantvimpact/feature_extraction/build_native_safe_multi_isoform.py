#!/usr/bin/env python3
"Multi-isoform variant of 25d_build_SAFE_protein_conservation_from_msa_root_strict.py."
import argparse
import gzip
import math
from pathlib import Path
from collections import Counter, OrderedDict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

KEYS = ["species", "kind", "snp_id", "Chr", "Pos", "Ref", "Alt"]
AA_STD = set("ACDEFGHIKLMNPQRSTVWY")


def open_text(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def core_id(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    x = str(x).strip()
    if not x:
        return ""
    if x.startswith(">"):
        x = x[1:]
    x = x.split()[0]
    if "|" in x:
        x = x.split("|")[-1]
    return x


def to_int_pos(x):
    try:
        if pd.isna(x):
            return -1
        return int(float(x))
    except Exception:
        return -1


def norm_aa(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    x = str(x).strip().upper()
    return x if len(x) == 1 else ""


def read_alias_map(path):
    """Return dict mapping any alias/core id to a list of unique protein ids."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"alias map not found: {path}")

    raw = pd.read_csv(path, sep="\t", dtype=str, header=None).fillna("")
    if raw.shape[1] < 2:
        raise RuntimeError(f"alias map must have at least 2 columns: {path}")


    h0 = str(raw.iloc[0, 0]).lower()
    h1 = str(raw.iloc[0, 1]).lower()
    if h0 in {"alias", "source", "id"} and h1 in {"protein_id", "target", "target_protein_id"}:
        raw = raw.iloc[1:].copy()

    aliases = {}

    def add(key, pcore):
        if not key or not pcore:
            return
        aliases.setdefault(key, [])
        if pcore not in aliases[key]:
            aliases[key].append(pcore)

    for _, r in raw.iterrows():
        a = str(r.iloc[0]).strip()
        p = str(r.iloc[1]).strip()
        if not a or not p:
            continue
        pcore = core_id(p)
        if not pcore:
            continue

        add(a, pcore)
        add(core_id(a), pcore)
        add(p, pcore)
        add(pcore, pcore)

    return aliases


def resolve_candidate(row, aliases):
    """Return ordered list of candidate protein ids for the row's gene/transcript."""
    for c in ["protein_id", "transcript_id", "gene_id", "GeneID"]:
        if c not in row:
            continue
        v = row[c]
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass
        v = str(v).strip()
        if not v:
            continue

        out = []
        for k in (v, core_id(v)):
            if k in aliases:
                for p in aliases[k]:
                    if p not in out:
                        out.append(p)
        if out:
            return out
    return []


def iter_fasta_records(path):
    h = None
    buf = []
    with open_text(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if h is not None:
                    yield h, "".join(buf)
                h = line[1:].strip()
                buf = []
            else:
                buf.append(line.strip())
        if h is not None:
            yield h, "".join(buf)


def fasta_files(root):
    suffixes = (".fa", ".faa", ".fasta", ".aln", ".fa.gz", ".faa.gz", ".fasta.gz", ".aln.gz")
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and str(p).endswith(suffixes))


class MsaCache:
    def __init__(self, max_files=4096):
        self.max_files = max_files
        self.cache = OrderedDict()

    def get(self, path):
        path = str(path)
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]

        records = []
        for h, seq in iter_fasta_records(path):
            records.append((h, seq))

        self.cache[path] = records
        if len(self.cache) > self.max_files:
            self.cache.popitem(last=False)
        return records


def build_pos_map(aligned_seq, wanted_positions):
    wanted_positions = {int(x) for x in wanted_positions if int(x) > 0}
    out = {}
    if not wanted_positions:
        return out

    ungap = 0
    for col0, ch in enumerate(aligned_seq):
        if ch != "-":
            ungap += 1
            if ungap in wanted_positions:
                out[ungap] = (col0, ch.upper())
    return out


def build_ungap_index(aligned_seq):
    """Return 1-based list mapping ungapped position -> (col0, aa). Index 0 is unused."""
    index = [(None, "")]
    for col0, ch in enumerate(aligned_seq):
        if ch != "-":
            index.append((col0, ch.upper()))
    return index


def missing_record(pid, aa_pos_int, ref, alt, reason, msa_available=0, target_aa=None, aa_pos_used=None, msa_file=None, header=None, msa_col=None):
    return {
        "_resolved_protein_ids": pid,
        "_aa_pos_int": aa_pos_int,
        "_ref_aa_norm": ref,
        "_alt_aa_norm": alt,
        "prot_safe_protein_id": pid if pid else None,
        "prot_safe_msa_file": str(Path(msa_file).name) if msa_file else None,
        "prot_safe_msa_header": header,
        "prot_safe_aa_pos_used": aa_pos_used,
        "prot_safe_msa_col": msa_col,
        "prot_safe_target_aa": target_aa,
        "prot_safe_ref_match": 0,
        "prot_safe_alignment_ok": 0,
        "prot_safe_msa_available": msa_available,
        "prot_safe_missing": 1,
        "prot_safe_reason": reason,
        "prot_safe_msa_depth": None,
        "prot_safe_gap_frac": None,
        "prot_safe_ref_aa_freq": None,
        "prot_safe_alt_aa_freq": None,
        "prot_safe_ref_aa_identity": None,
        "prot_safe_entropy": None,
        "prot_safe_delta_log_odds": None,
        "prot_safe_conservation_penalty": None,
        "prot_safe_abs_delta_log_odds": None,
        "prot_safe_alt_absent_flag": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--msa-root", required=True)
    ap.add_argument("--alias-map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--aa-pos-offset", type=int, default=1)
    ap.add_argument("--pseudocount", type=float, default=1e-3)
    args = ap.parse_args()

    cand_path = Path(args.candidates)
    msa_root = Path(args.msa_root)
    out_path = Path(args.out)

    if not cand_path.is_file():
        raise FileNotFoundError(cand_path)
    if not msa_root.is_dir():
        raise FileNotFoundError(msa_root)

    df = pd.read_parquet(cand_path)
    print("[CANDIDATES]", cand_path, "rows", len(df), flush=True)

    for forbidden in ["new_label", "fold", "split"]:
        if forbidden in df.columns:
            raise RuntimeError(f"candidate contains label column: {forbidden}")

    missing_keys = [k for k in KEYS if k not in df.columns]
    if missing_keys:
        raise RuntimeError(f"candidate missing key columns: {missing_keys}")

    for c in ["aa_pos", "ref_aa", "alt_aa"]:
        if c not in df.columns:
            raise RuntimeError(f"candidate missing required column: {c}")

    aliases = read_alias_map(args.alias_map)
    print("[ALIASES]", args.alias_map, "n keys=", len(aliases), flush=True)

    df["_aa_pos_int"] = df["aa_pos"].map(to_int_pos).astype("int64")
    df["_ref_aa_norm"] = df["ref_aa"].map(norm_aa)
    df["_alt_aa_norm"] = df["alt_aa"].map(norm_aa)


    

    gene_ids = df["gene_id"].astype(str)
    resolved_lists = gene_ids.map(lambda x: aliases.get(x, []))
    df["_resolved_protein_ids"] = resolved_lists.map(lambda x: ",".join(x))

    n_resolved = (df["_resolved_protein_ids"] != "").sum()
    print("[resolved rows]", int(n_resolved), flush=True)

    wanted = sorted({pid for ids in resolved_lists for pid in ids})
    print("[wanted proteins]", len(wanted), flush=True)

    found = scan_msa_root(msa_root, wanted)
    cache = MsaCache(max_files=4096)

    combo_cols = ["_resolved_protein_ids", "_aa_pos_int", "_ref_aa_norm", "_alt_aa_norm"]
    combos = df[combo_cols].drop_duplicates().reset_index(drop=True)
    print("[combos]", len(combos), flush=True)

    records = []
    reason_counts = Counter()

    for _, r in combos.iterrows():
        ids_str = "" if pd.isna(r["_resolved_protein_ids"]) else str(r["_resolved_protein_ids"])
        pid_list = [x for x in ids_str.split(",") if x]

        aa_raw = int(r["_aa_pos_int"])
        aa_used = aa_raw + args.aa_pos_offset
        ref = r["_ref_aa_norm"]
        alt = r["_alt_aa_norm"]

        if not pid_list:
            rec = missing_record("", aa_raw, ref, alt, "no_resolved_protein_id")
            records.append(rec)
            reason_counts["no_resolved_protein_id"] += 1
            continue

        candidate_infos = [(pid, found[pid]) for pid in pid_list if pid in found]
        if not candidate_infos:
            rec = missing_record(ids_str, aa_raw, ref, alt, "target_sequence_not_found_all_isoforms")
            records.append(rec)
            reason_counts["target_sequence_not_found_all_isoforms"] += 1
            continue

        if aa_raw < 0 or not ref:
            rec = missing_record(ids_str, aa_raw, ref, alt, "missing_aa_pos_or_ref_aa", 1, None, aa_used)
            records.append(rec)
            reason_counts["missing_aa_pos_or_ref_aa"] += 1
            continue

        success = False
        attempt_reasons = []
        for pid, info in candidate_infos:
            pos_index = info.get("_pos_index")
            if pos_index is None:
                pos_index = build_ungap_index(info["seq"])
                info["_pos_index"] = pos_index

            if aa_used >= len(pos_index):
                attempt_reasons.append("aa_pos_unmapped")
                continue

            col0, target_aa = pos_index[aa_used]
            if target_aa == "":
                attempt_reasons.append("aa_pos_unmapped")
                continue
            msa_col_1based = col0 + 1

            if target_aa != ref:
                attempt_reasons.append("ref_mismatch")
                continue

            msa_records = cache.get(info["msa_file"])
            feats = column_features(msa_records, col0, ref, alt, args.pseudocount)
            if feats is None:
                attempt_reasons.append("empty_msa_column")
                continue

            rec = {
                "_resolved_protein_ids": ids_str,
                "_aa_pos_int": aa_raw,
                "_ref_aa_norm": ref,
                "_alt_aa_norm": alt,
                "prot_safe_protein_id": pid,
                "prot_safe_msa_file": str(Path(info["msa_file"]).name),
                "prot_safe_msa_header": info["header"],
                "prot_safe_aa_pos_used": aa_used,
                "prot_safe_msa_col": msa_col_1based,
                "prot_safe_target_aa": target_aa,
                "prot_safe_ref_match": 1,
                "prot_safe_alignment_ok": 1,
                "prot_safe_msa_available": 1,
                "prot_safe_missing": 0,
                "prot_safe_reason": "ok",
            }
            rec.update(feats)
            records.append(rec)
            reason_counts["ok"] += 1
            success = True
            break

        if not success:
            if "ref_mismatch" in attempt_reasons:
                reason = "ref_mismatch_all_isoforms"
            elif "aa_pos_unmapped" in attempt_reasons:
                reason = "aa_pos_unmapped_all_isoforms"
            elif "empty_msa_column" in attempt_reasons:
                reason = "empty_msa_column_all_isoforms"
            else:
                reason = "unknown"
            rec = missing_record(ids_str, aa_raw, ref, alt, reason, 1, None, aa_used)
            records.append(rec)
            reason_counts[reason] += 1

        if len(records) % 100000 == 0:
            print("[feature combos processed]", len(records), flush=True)

    feat = pd.DataFrame.from_records(records)
    merged = df.merge(feat, on=combo_cols, how="left", sort=False)

    if len(merged) != len(df):
        raise RuntimeError(f"feature merge changed row count: input={len(df)} merged={len(merged)}")

    feature_cols = [
        "prot_safe_protein_id",
        "prot_safe_msa_file",
        "prot_safe_msa_header",
        "prot_safe_aa_pos_used",
        "prot_safe_msa_col",
        "prot_safe_target_aa",
        "prot_safe_ref_match",
        "prot_safe_alignment_ok",
        "prot_safe_msa_available",
        "prot_safe_missing",
        "prot_safe_reason",
        "prot_safe_msa_depth",
        "prot_safe_gap_frac",
        "prot_safe_ref_aa_freq",
        "prot_safe_alt_aa_freq",
        "prot_safe_ref_aa_identity",
        "prot_safe_entropy",
        "prot_safe_delta_log_odds",
        "prot_safe_conservation_penalty",
    ]

    out_df = merged[KEYS + feature_cols].copy()

    for forbidden in ["new_label", "fold", "split"]:
        if forbidden in out_df.columns:
            raise RuntimeError(f"output contains label column: {forbidden}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(out_df, preserve_index=False), out_path, compression="snappy")

    print("[WRITE]", out_path, flush=True)
    print("rows", len(out_df), flush=True)
    print("prot_safe_missing", float(out_df["prot_safe_missing"].mean()), flush=True)
    print("prot_safe_alignment_ok", float(out_df["prot_safe_alignment_ok"].mean()), flush=True)
    print("prot_safe_msa_available", float(out_df["prot_safe_msa_available"].mean()), flush=True)
    print("[reason counts]", flush=True)
    for k, v in reason_counts.most_common():
        print(k, v, flush=True)


from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_identical
from phenofuncrank_repro.common import shared_scan_msa_root as _profile_kernel_1
_bind_identical(globals(), _profile_kernel_1, ['scan_msa_root'])
from phenofuncrank_repro.common import shared_column_features as _profile_kernel_2
_bind_identical(globals(), _profile_kernel_2, ['column_features'])

if __name__ == "__main__":
    main()
