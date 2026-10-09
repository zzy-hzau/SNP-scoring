#!/usr/bin/env python3
import argparse, gzip, re
from pathlib import Path

def opn(p, mode="rt"):
    return gzip.open(p, mode) if str(p).endswith(".gz") else open(p, mode)

ap = argparse.ArgumentParser()
ap.add_argument("--result", required=True)
ap.add_argument("--outdir", required=True)
args = ap.parse_args()

result = Path(args.result)
seqids = result / "WorkingDirectory" / "SequenceIDs.txt"
msa_dir = result / "MultipleSequenceAlignments"
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

mp = {}

with open(seqids, encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.rstrip("\n")
        if ": " not in line:
            continue

        k, v = line.split(": ", 1)
        v = v.split()[0]          
        mp[k] = v
        mp["Species" + k] = v


        if "|" in v:
            prefix, rest = v.split("|", 1)
            mp[prefix + "_" + prefix + "|" + rest] = v
            mp[prefix + "|" + rest] = v

files = sorted(msa_dir.glob("OG*.fa*"))
headers = converted = missing = unchanged = 0

for fa in files:
    name = fa.name[:-3] if fa.name.endswith(".gz") else fa.name
    out = outdir / name

    with opn(fa) as fi, open(out, "w") as fo:
        for line in fi:
            if line.startswith(">"):
                headers += 1
                old = line[1:].strip().split()[0]
                new = mp.get(old)

                if new is None:

                    m = re.match(r"^Species(\d+_\d+)$", old)
                    if m:
                        new = mp.get(m.group(1))

                if new is None:
                    new = old
                    missing += 1
                else:
                    if new == old:
                        unchanged += 1
                    else:
                        converted += 1

                fo.write(f">{new}\n")
            else:
                fo.write(line)

print(f"msa_files\t{len(files)}")
print(f"headers\t{headers}")
print(f"converted\t{converted}")
print(f"unchanged\t{unchanged}")
print(f"missing_mapping\t{missing}")
