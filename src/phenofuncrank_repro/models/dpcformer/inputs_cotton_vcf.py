#!/usr/bin/env python3
import argparse
import json
import subprocess
from shlex import quote
from pathlib import Path

import numpy as np
import pandas as pd


def run(cmd):
    return subprocess.check_output(cmd, text=True).strip()


def chrom_to_numeric(chrom):
    c = str(chrom)
    if c.startswith("A"):
        return int(c[1:])
    if c.startswith("D"):
        return int(c[1:]) + 13
    if c.isdigit():
        return int(c)
    raise ValueError(f"Unsupported chromosome: {chrom}")


def chrom_to_group(chrom):
    c = str(chrom)
    if c.startswith("A") or c.startswith("D"):
        return int(c[1:])
    n = int(c)
    return n if n <= 13 else n - 13


def get_variant_count(vcf):
    try:
        return int(run(["bcftools", "index", "-n", vcf]))
    except Exception:
        txt = run(["bcftools", "index", "-s", vcf])
        return sum(int(x.split("\t")[2]) for x in txt.splitlines() if x.strip())


def gt_to_code(gt):
    if gt in {".", "./.", ".|.", "./././.", ".|.|.|."}:
        return 0
    sep = "/" if "/" in gt else "|"
    alleles = gt.split(sep)
    if any(a == "." for a in alleles):
        return 0
    if len(alleles) != 2:
        return 0
    if any(a not in {"0", "1"} for a in alleles):
        return 0
    alt_count = (alleles[0] == "1") + (alleles[1] == "1")
    return int(alt_count) + 1  


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--pheno", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/6] read phenotype")
    pheno = pd.read_csv(args.pheno, sep="\t")
    if "LINE" not in pheno.columns:
        raise ValueError("Phenotype file must contain LINE column")
    pheno["LINE"] = pheno["LINE"].astype(str)
    pheno = pheno.set_index("LINE", drop=False)

    print("[2/6] read VCF sample list")
    samples = run(["bcftools", "query", "-l", args.vcf]).splitlines()
    matched_samples = [s for s in samples if s in pheno.index]
    if not matched_samples:
        raise RuntimeError("No matched samples between VCF and phenotype LINE")

    sample_file = outdir / "matched_samples.vcf_order.txt"
    sample_file.write_text("\n".join(matched_samples) + "\n")

    pheno.loc[matched_samples].to_csv(outdir / "matched_phenotypes.tsv", sep="\t", index=False)

    print(f"[INFO] VCF samples={len(samples)} matched_samples={len(matched_samples)}")

    print("[3/6] count variants")
    n_snps = get_variant_count(args.vcf)
    n_samples = len(matched_samples)
    print(f"[INFO] n_snps={n_snps:,}; n_samples={n_samples:,}")

    geno_path = outdir / "cultivar_genotype_codes.variant_by_sample.uint8.memmap"
    geno = np.memmap(geno_path, mode="w+", dtype=np.uint8, shape=(n_snps, n_samples))

    print("[4/6] stream VCF GT and build memmap")
    cmd = (
        f"bcftools view --threads {args.threads} -S {quote(str(sample_file))} -Ou {quote(str(args.vcf))} | "
        f"bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT[\\t%GT]\\n'"
    )

    map_rows = []
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    i = 0
    for line in p.stdout:
        parts = line.rstrip("\n").split("\t")
        chrom, pos, ref, alt = parts[:4]
        gts = parts[4:]

        if len(gts) != n_samples:
            raise RuntimeError(f"Sample count mismatch at {chrom}:{pos}: {len(gts)} vs {n_samples}")

        codes = np.fromiter((gt_to_code(x) for x in gts), dtype=np.uint8, count=n_samples)
        geno[i, :] = codes

        snp_id = f"{chrom}_{pos}_{ref}_{alt}"
        numeric_chrom = chrom_to_numeric(chrom)
        group_id = chrom_to_group(chrom)
        map_rows.append((i, chrom, numeric_chrom, group_id, int(pos), ref, alt, snp_id))

        i += 1
        if i % 100000 == 0:
            print(f"[INFO] parsed {i:,}/{n_snps:,}")

    stderr = p.stderr.read()
    ret = p.wait()
    if ret != 0:
        raise RuntimeError(f"bcftools pipeline failed:\n{stderr}")

    if i != n_snps:
        raise RuntimeError(f"Parsed variant count {i} != index count {n_snps}")

    geno.flush()

    print("[5/6] write SNP map")
    snp_map = pd.DataFrame(
        map_rows,
        columns=["variant_index", "Chr", "chromosome", "group_id", "Pos", "Ref", "Alt", "snp_id"]
    )
    snp_map["map"] = 0
    snp_map.to_csv(outdir / "cultivar.snp_map.tsv.gz", sep="\t", index=False, compression="gzip")


    classic = snp_map[["chromosome", "snp_id", "map", "Pos"]].copy()
    classic.to_csv(outdir / "cotton.map", sep="\t", header=False, index=False)

    print("[6/6] write metadata")
    meta = {
        "vcf": "resources/input.vcf.gz",
        "pheno": "resources/phenotypes",
        "n_snps": int(n_snps),
        "n_samples": int(n_samples),
        "genotype_memmap": geno_path.name,
        "snp_map": "cultivar.snp_map.tsv.gz",
        "samples": sample_file.name,
        "phenotypes": "matched_phenotypes.tsv",
        "dtype": "uint8",
        "shape": [int(n_snps), int(n_samples)]
    }
    with open(outdir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("[DONE]", outdir)


if __name__ == "__main__":
    main()
