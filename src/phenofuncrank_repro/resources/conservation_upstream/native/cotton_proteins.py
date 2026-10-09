from pathlib import Path
import gzip,re,collections
allowed=set("ACDEFGHIKLMNPQRSTVWYBXZ")

def op(p):
    return gzip.open(p, "rt") if str(p).endswith(".gz") else open(p)

def records(p):
    h, s = None, []
    with op(p) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if h:
                    yield h, "".join(s)
                h, s = line[1:], []
            else:
                s.append(line)
        if h:
            yield h, "".join(s)

def clean_id(x):
    x = x.strip().split()[0]
    return re.sub(r"[^A-Za-z0-9_.:-]", "_", x)

def gene_id(header):
    m = re.search(r"\bgene=([^\s;]+)", header)
    if m:
        return m.group(1)
    first = header.split()[0]
    m = re.match(r"(.+)\.\d+$", first)
    return m.group(1) if m else first

def prepare(sources, raw_dir, clean_dir, qc_dir):
    RAW=Path(raw_dir);CLEAN=Path(clean_dir);QC=Path(qc_dir)
    for directory in [RAW,CLEAN,QC]:directory.mkdir(parents=True,exist_ok=True)
    with open(QC/"protein_id_map.tsv","w") as mapout, open(QC/"protein_cleaning_report.tsv","w") as rep:
        mapout.write("species\tprotein_id\tgene_id\tsource_id\tlength\theader\n")
        rep.write("species\traw_records\tkept_records\tconverted_to_X\n")
        for sp, path in sources:
            if not path.exists():
                raise SystemExit(f"Missing {path}")

            best = {}
            meta = {}
            raw_n = 0
            converted_total = 0

            for h, seq in records(path):
                raw_n += 1
                sid = clean_id(h)
                gid = clean_id(gene_id(h))
                seq = seq.upper().replace("*", "")
                cleaned = []
                converted = 0
                for aa in seq:
                    if aa in allowed:
                        cleaned.append(aa)
                    elif aa.strip():
                        cleaned.append("X")
                        converted += 1
                seq = "".join(cleaned)
                if not seq:
                    continue
                if gid not in best or len(seq) > len(best[gid]):
                    best[gid] = seq
                    meta[gid] = (sid, h)
                    converted_total += converted

            for outdir in (RAW, CLEAN):
                with open(outdir/f"{sp}.fa","w") as out:
                    for gid in sorted(best):
                        sid, h = meta[gid]
                        out.write(f">{sid}\n")
                        seq = best[gid]
                        for i in range(0, len(seq), 60):
                            out.write(seq[i:i+60] + "\n")

            for gid in sorted(best):
                sid, h = meta[gid]
                mapout.write(f"{sp}\t{sid}\t{gid}\t{sid}\t{len(best[gid])}\t{h}\n")

            rep.write(f"{sp}\t{raw_n}\t{len(best)}\t{converted_total}\n")
            print(sp, raw_n, "->", len(best))
