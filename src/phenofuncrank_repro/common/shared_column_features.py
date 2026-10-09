"""Identical source function; caller module supplies its recorded policy bindings."""
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

def column_features(records, col0, ref_aa, alt_aa, pseudocount):
    vals = []
    nseq = len(records)

    for _, seq in records:
        if col0 >= len(seq):
            ch = "-"
        else:
            ch = seq[col0].upper()
        if ch in AA_STD:
            vals.append(ch)

    depth = len(vals)
    gap_frac = 1.0 - (depth / nseq) if nseq else None

    if depth == 0:
        return None

    counts = Counter(vals)
    ref_count = counts.get(ref_aa, 0)
    alt_count = counts.get(alt_aa, 0)

    ref_freq = ref_count / depth
    alt_freq = alt_count / depth

    entropy = 0.0
    for c in counts.values():
        p = c / depth
        entropy -= p * math.log(p, 2)

    delta_log_odds = math.log((alt_freq + pseudocount) / (ref_freq + pseudocount))
    conservation_penalty = math.log((ref_freq + pseudocount) / (alt_freq + pseudocount))


    

    

    abs_delta_log_odds = abs(delta_log_odds)
    alt_absent_flag = 1 if alt_count == 0 else 0

    return {
        "prot_safe_msa_depth": depth,
        "prot_safe_gap_frac": gap_frac,
        "prot_safe_ref_aa_freq": ref_freq,
        "prot_safe_alt_aa_freq": alt_freq,
        "prot_safe_ref_aa_identity": ref_freq,
        "prot_safe_entropy": entropy,
        "prot_safe_delta_log_odds": delta_log_odds,
        "prot_safe_conservation_penalty": conservation_penalty,
        "prot_safe_abs_delta_log_odds": abs_delta_log_odds,
        "prot_safe_alt_absent_flag": alt_absent_flag,
    }
