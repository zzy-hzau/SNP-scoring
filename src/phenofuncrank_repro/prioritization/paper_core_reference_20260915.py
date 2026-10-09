"-paper real-allele Core selector and explicit panel mapping."
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd

def norm_chr(value: object) -> str:
    s = str(value).strip().lower().replace("chr", "")
    if s.isdigit():
        s = str(int(s))
    return s

def load_real_core(master_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "snp_id", "chrom", "pos", "schema_ref", "schema_alt",
        "from_base", "to_base", "dpc_additive_effect_mean",
        "n_evaluable_folds",
    ]
    d = pd.read_csv(master_path, sep="\t", usecols=cols, low_memory=False)
    for c in ["schema_ref", "schema_alt", "from_base", "to_base"]:
        d[c] = d[c].astype(str).str.upper()
    d["pos"] = pd.to_numeric(d["pos"], errors="raise").astype(np.int64)
    d["effect"] = pd.to_numeric(d["dpc_additive_effect_mean"], errors="coerce")
    d["abs_effect_real"] = d["effect"].abs()
    real = d[(d["from_base"] == d["schema_ref"]) & (d["to_base"] == d["schema_alt"])].copy()
    if real.empty:
        raise RuntimeError(f"No canonical real-direction rows in {master_path}")
    real = real[np.isfinite(real["abs_effect_real"])].copy()

    counts = real.groupby(["chrom", "pos"], sort=False).size()
    if (counts != 1).any():
        bad = int((counts != 1).sum())
        raise RuntimeError(f"{master_path}: {bad} loci do not have exactly one real row")
    loci = real.sort_values(
        ["abs_effect_real", "snp_id"], ascending=[False, True], kind="mergesort"
    ).drop_duplicates(["chrom", "pos"], keep="first").reset_index(drop=True)
    n_core = int(math.ceil(len(loci) * 0.05))
    core = loci.head(n_core).copy()
    core["core_rank"] = np.arange(1, len(core) + 1, dtype=np.int64)
    core["locus_key"] = [f"{norm_chr(c)}:{int(p)}" for c, p in zip(core["chrom"], core["pos"])]
    return loci, core

def map_core_to_panel(tag: str, core: pd.DataFrame, network_dir: Path) -> pd.DataFrame:
    anchors = pd.read_csv(network_dir / "anchors.tsv", sep="\t", low_memory=False)
    anchors["locus_key"] = [
        f"{norm_chr(c)}:{int(p)}" for c, p in zip(anchors["chrom"], anchors["pos"])
    ]
    keep = anchors[["locus_key", "pool_id"]].drop_duplicates("locus_key")
    out = core.merge(keep, on="locus_key", how="left", validate="one_to_one")
    if out["pool_id"].isna().any():
        missing = int(out["pool_id"].isna().sum())
        raise RuntimeError(f"{tag}: {missing} revised Core loci missing from the panel map")
    out["pool_id"] = out["pool_id"].astype(str)
    return out

def read_bim(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        str(path), sep="\t", header=None,
        names=["chrom", "id", "cm", "pos", "a1", "a2"],
        dtype={"chrom": str, "id": str, "pos": np.int64, "a1": str, "a2": str},
    )

def panel_meta_for_ids(kind: str, wanted: set[str], *, bfile: Path | None = None,
                       pfile: Path | None = None) -> pd.DataFrame:
    rows: list[tuple[str, str, int, str, str]] = []
    if kind in {"cotton", "chickpea"}:
        if bfile is None:
            raise ValueError("bfile required")
        bfiles = list(bfile.values()) if isinstance(bfile, dict) else [bfile]
        for bf in bfiles:
            bim = read_bim(Path(bf).with_suffix(".bim"))
            sub = bim[bim["id"].isin(wanted)]
            rows.extend(
                (str(r.id), str(r.chrom), int(r.pos), str(r.a1), str(r.a2))
                for r in sub.itertuples(index=False)
            )
    elif kind == "rice":
        if pfile is None:
            raise ValueError("pfile required")
        pvar = pfile.with_suffix(".pvar")
        with pvar.open() as fh:
            header = None
            for line in fh:
                if line.startswith("#CHROM"):
                    header = line.rstrip("\n").split()
                    break
            if header is None:
                raise RuntimeError(f"No #CHROM header in {pvar}")
            idx = {name.lstrip("#"): i for i, name in enumerate(header)}
            for line in fh:
                if not line.strip() or line.startswith("#"):
                    continue
                f = line.rstrip("\n").split()
                sid = f[idx["ID"]]
                if sid in wanted:
                    rows.append((sid, f[idx["CHROM"]], int(f[idx["POS"]]),
                                 f[idx["REF"]], f[idx["ALT"]]))
    out = pd.DataFrame(rows, columns=["panel_id", "chrom", "pos", "ref", "alt"])
    if len(out) != len(wanted):
        missing = len(wanted) - len(out)
        raise RuntimeError(f"Panel metadata missing {missing}/{len(wanted)} requested IDs")
    return out


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--panel-map-directory',type=Path)
    parser.add_argument('--trait-label',default='supplied_trait')
    args=parser.parse_args()
    loci,core=load_real_core(args.master)
    if args.panel_map_directory is not None:
        core=map_core_to_panel(args.trait_label,core,args.panel_map_directory)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    loci.to_csv(args.output_dir/'real_direction_locus_universe.tsv',sep='\t',index=False)
    core.to_csv(args.output_dir/'core_real_direction_top5.tsv',sep='\t',index=False)

if __name__=='__main__':
    main()
