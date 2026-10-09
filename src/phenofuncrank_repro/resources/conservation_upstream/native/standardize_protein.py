#!/usr/bin/env python3
import argparse
import gzip
import re
from pathlib import Path

def open_text(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")

def clean_protein_id(header):
    token = header[1:].strip().split()[0]
    if "|" in token:
        token = token.split("|")[-1]
    token = re.sub(r"[^A-Za-z0-9_.:-]", "_", token)
    if not token:
        raise ValueError(f"empty protein id from header: {header!r}")
    return token

def wrap(seq, width=60):
    for i in range(0, len(seq), width):
        yield seq[i:i+width]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strip-stop", action="store_true", default=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    n = 0
    cur_header = None
    cur_seq = []

    def emit(fo, header, seq_parts):
        nonlocal n
        if header is None:
            return
        pid = clean_protein_id(header)
        std = f"{args.prefix}|{pid}"
        if std in seen:
            raise RuntimeError(f"duplicate standardized header: {std}")
        seen.add(std)

        seq = "".join(seq_parts).replace(" ", "").replace("\t", "").upper()
        if args.strip_stop:
            seq = seq.replace("*", "")
        if not seq:
            raise RuntimeError(f"empty sequence for {std}")

        fo.write(f">{std}\n")
        for line in wrap(seq):
            fo.write(line + "\n")
        n += 1

    with open_text(args.input) as fi, open(out, "w", newline="\n") as fo:
        for line in fi:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith(">"):
                emit(fo, cur_header, cur_seq)
                cur_header = line
                cur_seq = []
            else:
                cur_seq.append(line.strip())
        emit(fo, cur_header, cur_seq)

    print(f"input={args.input}")
    print(f"prefix={args.prefix}")
    print(f"out={args.out}")
    print(f"seq_count={n}")
    print(f"unique_headers={len(seen)}")

if __name__ == "__main__":
    main()
