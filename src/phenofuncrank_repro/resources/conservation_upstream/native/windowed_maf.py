"""Original sanitized, first-hit-per-species windowed MAF algorithm."""
import sys,os

def split_maf(maf, outdir, window, ref_src):
    maf=str(maf);outdir=str(outdir);window=int(window)

    os.makedirs(outdir, exist_ok=True)

    def genome(src):
        for g in ("Sly_SL5", "Sly_SL3", "Stu_SolTub3", "Can_ASM51225v2"):
            if src == g or src.startswith(g + "."):
                return g
        return src.split(".", 1)[0]

    handles = {}
    counts = {}

    def get_handle(idx):
        if idx not in handles:
            path = os.path.join(outdir, f"chunk_{idx:08d}.maf")
            fh = open(path, "w")
            fh.write("##maf version=1\n\n")
            handles[idx] = fh
            counts[idx] = 0
        return handles[idx]

    def flush_block(lines):
        srows = []
        for line in lines:
            p = line.split()
            if len(p) >= 7 and p[0] == "s":
                srows.append(p)

        ref = None
        for p in srows:
            if p[1] == ref_src:
                ref = p
                break
        if ref is None:
            return

        chosen = {}
        chosen[genome(ref[1])] = ref

        for p in srows:
            g = genome(p[1])
            if g not in chosen:
                chosen[g] = p

        if len(chosen) < 2:
            return

        idx = int(ref[2]) // window
        fh = get_handle(idx)
        fh.write("a\n")
        ref_g = genome(ref[1])
        ordered = [chosen[ref_g]] + [chosen[g] for g in sorted(chosen) if g != ref_g]
        for p in ordered:
            g = genome(p[1])
            fh.write("s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (g, p[2], p[3], p[4], p[5], p[6]))
        fh.write("\n")
        counts[idx] += 1

    block = []
    with open(maf) as fh:
        for line in fh:
            p = line.split()
            if not p:
                if block:
                    flush_block(block)
                    block = []
                continue
            if p[0].startswith("#"):
                continue
            if p[0] == "a":
                if block:
                    flush_block(block)
                block = [line]
            else:
                block.append(line)

    if block:
        flush_block(block)

    for fh in handles.values():
        fh.close()

    with open(os.path.join(outdir, "chunks.list"), "w") as out:
        for idx in sorted(counts):
            if counts[idx] > 0:
                out.write(os.path.join(outdir, f"chunk_{idx:08d}.maf") + "\n")

    print("chunks", len(counts))
    print("blocks", sum(counts.values()))
