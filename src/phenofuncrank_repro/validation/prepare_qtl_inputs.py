"""Prepare the reported per-lead QTL inputs from ranked variants and full GWAS tests."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from phenofuncrank_repro.validation.chickpea_gwas_leads import Panel, bh, greedy_clump, expand_perlead


def prepare(config, directory):
    import pysam
    base = Path(config.pop("_base", "."))
    def path(value):
        value = Path(value)
        return value if value.is_absolute() else base / value
    catalogue = json.loads(path(config["catalogue"]).read_text())
    if isinstance(catalogue, dict):
        catalogue = catalogue["QTL_rows"] if "QTL_rows" in catalogue else catalogue["records"]
    if catalogue and isinstance(catalogue[0], dict):
        catalogue = [[r["QTL ID"], r["Trait"], str(r["Chromosome"]), int(r["Start (bp)"]), int(r["End (bp)"])] for r in catalogue]
    names = {"Rice plant height": "Rice PH", "Cotton fibre strength": "Cotton FS", "Chickpea hundred-seed weight": "Chickpea HSW", "Chickpea plant width": "Chickpea PW"}
    catalogue = [[row[0], names.get(row[1], row[1]), *row[2:]] for row in catalogue]
    output, counts = [], []
    for item in config["traits"]:
        variants = pd.read_csv(path(item["variants"]), sep="\t", dtype={"chrom": str})
        codes = np.load(path(item["codes"]), mmap_mode="r")
        if codes.shape[0] != len(variants):
            raise ValueError("Genotype rows and variant map differ")
        panel = Panel(codes, variants)
        a = pd.read_csv(path(item["association"]), sep="\t")
        columns = item.get("association_columns", {"chr": "chrom", "ps": "pos", "allele0": "ref", "allele1": "alt"})
        a = a.rename(columns=columns)
        p = a.p_wald.to_numpy(float)
        a = a.loc[np.isfinite(p) & (p > 0)].copy()
        a["BH_q"] = bh(a.p_wald)

        aliases = item.get("chromosome_aliases", {})
        a["chrom"] = a.chrom.astype(str).map(lambda x: aliases.get(x, x))
        a["pos"] = a.pos.astype(np.int64)
        significant = a[a.BH_q <= .05].sort_values("p_wald", kind="mergesort").drop_duplicates(["chrom", "pos"]).reset_index(drop=True)
        leads = greedy_clump(panel, significant).head(100)
        framework = pd.read_parquet(path(item["ranked"])) if str(item["ranked"]).endswith(".parquet") else pd.read_csv(path(item["ranked"]), sep="\t")
        framework = framework.rename(columns=item.get("ranked_columns", {"priority_rank": "rank"}))
        framework["chrom"] = framework.chrom.astype(str).map(lambda x: aliases.get(x, x))
        framework = framework.sort_values("rank", kind="mergesort").head(100)
        reference = pysam.FastaFile(str(path(item["reference"])))
        try:
            for method, frame in [("Framework", framework), ("GWAS", leads)]:
                frame = frame.reset_index(drop=True)
                intervals = expand_perlead(panel, frame[["chrom", "pos"]]) if len(frame) else pd.DataFrame(columns=["seed_row", "chrom", "start", "end", "region_id"])
                if len(intervals) != len(frame):
                    raise ValueError("A selected lead is missing from the supplied genotype panel")
                for rank, (row, interval) in enumerate(zip(frame.to_dict("records"), intervals.to_dict("records")), 1):
                    chrom, pos = str(row["chrom"]), int(row["pos"])
                    rc = item.get("reference_chromosome_aliases", {}).get(chrom, chrom)
                    ref = reference.fetch(rc, pos-1, pos).upper()
                    alleles = {str(row["ref"]).upper(), str(row["alt"]).upper()}
                    if len(alleles) != 2 or ref not in alleles or not alleles <= set("ACGT"):
                        raise ValueError(f"Observed alleles disagree with the reference at {chrom}:{pos}")
                    output.append(dict(trait=item["trait"], method=method, rank=rank, chrom=chrom, pos=pos, ref=ref, alt=next(iter(alleles-{ref})), ld_start=int(interval["start"]), ld_end=int(interval["end"]), p=float(row["p_wald"]) if method=="GWAS" else None, q=float(row["BH_q"]) if method=="GWAS" else None))
        finally:
            reference.close()
        counts.append(dict(trait=item["trait"], tested_family=len(a), significant=len(significant), retained_gwas=len(leads), framework=len(framework)))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(catalogue=catalogue, candidates=output, input_profile="reported_complete_GWAS_BH_and_unchained_perlead_LD")
    (directory / "qtl_inputs.json").write_text(json.dumps(payload, indent=2)+"\n")
    pd.DataFrame(output).to_csv(directory / "candidates_with_intervals.tsv", sep="\t", index=False)
    (directory / "input_manifest.json").write_text(json.dumps(dict(traits=counts, window_bp=500000, r2_strictly_above=.60, intervals_chained=False, coordinates="1-based inclusive"), indent=2)+"\n")
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    config["_base"] = args.config.resolve().parent
    prepare(config, args.output_dir)


if __name__ == "__main__":
    main()
