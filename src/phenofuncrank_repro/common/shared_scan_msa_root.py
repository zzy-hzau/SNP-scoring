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

def scan_msa_root(msa_root, wanted):
    wanted = set(wanted)
    found = {}
    files = fasta_files(msa_root)
    print("[MSA files]", len(files), flush=True)
    print("[wanted proteins]", len(wanted), flush=True)

    for i, p in enumerate(files, 1):
        if len(found) >= len(wanted):
            break
        try:
            for h, seq in iter_fasta_records(p):
                cid = core_id(h)
                if cid in wanted and cid not in found:
                    found[cid] = {
                        "msa_file": str(p),
                        "header": h,
                        "seq": seq,
                    }
        except Exception as e:
            print("[WARN] failed reading", p, e, flush=True)

        if i % 5000 == 0:
            print(f"[scan MSA] {i}/{len(files)} found={len(found)}/{len(wanted)}", flush=True)

    print("[found proteins]", len(found), flush=True)
    return found
