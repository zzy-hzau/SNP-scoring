from pathlib import Path
import gzip,re

def op(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")

def iter_fasta(path):
    name = None
    seq = []
    with op(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line.strip())
        if name is not None:
            yield name, "".join(seq)

def wrap_write(w, name, seq):
    w.write(f">{name}\n")
    seq = seq.upper()
    for i in range(0, len(seq), 80):
        w.write(seq[i:i+80] + "\n")

def prepare(reference, queries, clean_dir, qc_dir):
    REF=Path(reference);CLEAN=Path(clean_dir);QC=Path(qc_dir)
    CLEAN.mkdir(parents=True,exist_ok=True);QC.mkdir(parents=True,exist_ok=True)
    target_keep = re.compile(r"^(A|D)(0[1-9]|1[0-3])$")

    target_out = CLEAN / "tm1.primary.fa"
    target_map = QC / "tm1.primary.name_map.tsv"

    n = 0
    total_bp = 0
    with open(target_out, "w") as w, open(target_map, "w") as m:
        m.write("new_name\told_name\tlength\n")
        for name, seq in iter_fasta(REF):
            short = name.split()[0]
            if target_keep.match(short):
                wrap_write(w, short, seq)
                m.write(f"{short}\t{name}\t{len(seq)}\n")
                n += 1
                total_bp += len(seq)

    print(f"tm1: wrote {n} primary chromosomes, bp={total_bp}")
    if n != 26:
        raise SystemExit(f"Expected 26 TM-1 chromosomes A01-D13, got {n}. Check {REF}")

    for sp, path, keep_n in queries:
        if not path.exists():
            raise SystemExit(f"Missing query FASTA: {path}")

        lengths = []
        for name, seq in iter_fasta(path):
            lengths.append((len(seq), name))
        lengths.sort(reverse=True)

        keep = {name for _, name in lengths[:keep_n]}
        rank = {name: i + 1 for i, (_, name) in enumerate(lengths[:keep_n])}

        out = CLEAN / f"{sp}.primary.fa"
        mp = QC / f"{sp}.primary.name_map.tsv"

        n = 0
        total_bp = 0
        with open(out, "w") as w, open(mp, "w") as m:
            m.write("new_name\told_name\tlength\n")
            for name, seq in iter_fasta(path):
                if name not in keep:
                    continue
                new = f"{sp}_chr{rank[name]:02d}"
                wrap_write(w, new, seq)
                m.write(f"{new}\t{name}\t{len(seq)}\n")
                n += 1
                total_bp += len(seq)

        print(f"{sp}: wrote {n} primary sequences, bp={total_bp}, out={out}")

    print("DONE")
