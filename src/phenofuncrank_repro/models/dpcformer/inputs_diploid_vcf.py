#!/usr/bin/env python3
import argparse
import gzip
import json
import re
import subprocess
from shlex import quote
from pathlib import Path

import numpy as np
import pandas as pd


def run(cmd):
    return subprocess.check_output(cmd, text=True).strip()


def natural_chrom_key(chrom):
    text = str(chrom)
    parts = re.split(r"(\d+)", text)
    out = []
    for part in parts:
        if part == "":
            continue
        out.append(int(part) if part.isdigit() else part.lower())
    return out


def get_variant_count(vcf):
    try:
        return int(run(["bcftools", "index", "-n", vcf]))
    except Exception:
        txt = run(["bcftools", "index", "-s", vcf])
        return sum(int(x.split("\t")[2]) for x in txt.splitlines() if x.strip())


def vcf_has_gt_format(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("##FORMAT=<ID=GT,"):
                return True
            if line.startswith("#CHROM"):
                return False
    return False


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
    ap.add_argument("--sample-id-col", default="LINE")
    ap.add_argument(
        "--pheno-sep",
        default="auto",
        help="Phenotype separator: auto, comma, tab, or whitespace.",
    )
    ap.add_argument("--species", default="diploid")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/6] read phenotype")
    sep_arg = args.pheno_sep.lower()
    if sep_arg == "auto":
        sep = "," if str(args.pheno).endswith(".csv") else "\t"
        read_kwargs = {"sep": sep}
    elif sep_arg == "comma":
        read_kwargs = {"sep": ","}
    elif sep_arg == "tab":
        read_kwargs = {"sep": "\t"}
    elif sep_arg == "whitespace":
        read_kwargs = {"sep": r"\s+", "engine": "python"}
    else:
        read_kwargs = {"sep": args.pheno_sep}
    pheno = pd.read_csv(args.pheno, **read_kwargs)
    if args.sample_id_col not in pheno.columns:
        raise ValueError(
            f"Phenotype file must contain sample ID column {args.sample_id_col!r}; "
            f"available={pheno.columns.tolist()}"
        )
    pheno[args.sample_id_col] = pheno[args.sample_id_col].astype(str)
    pheno = pheno.set_index(args.sample_id_col, drop=False)

    print("[2/6] read VCF sample list")
    samples = run(["bcftools", "query", "-l", args.vcf]).splitlines()
    matched_samples = [s for s in samples if s in pheno.index]
    if not matched_samples:
        raise RuntimeError(
            f"No matched samples between VCF and phenotype {args.sample_id_col}"
        )

    sample_file = outdir / "matched_samples.vcf_order.txt"
    sample_file.write_text("\n".join(matched_samples) + "\n")

    matched_pheno = pheno.loc[matched_samples].copy()
    if "LINE" not in matched_pheno.columns:
        matched_pheno.insert(0, "LINE", matched_pheno[args.sample_id_col].astype(str).values)
    matched_pheno.to_csv(outdir / "matched_phenotypes.tsv", sep="\t", index=False)

    print(f"[INFO] VCF samples={len(samples)} matched_samples={len(matched_samples)}")

    print("[3/6] count variants")
    n_snps = get_variant_count(args.vcf)
    n_samples = len(matched_samples)
    print(f"[INFO] n_snps={n_snps:,}; n_samples={n_samples:,}")

    geno_path = outdir / "cultivar_genotype_codes.variant_by_sample.uint8.memmap"
    geno = np.memmap(geno_path, mode="w+", dtype=np.uint8, shape=(n_snps, n_samples))

    print("[4/6] stream VCF GT and build memmap")
    if vcf_has_gt_format(args.vcf):
        source_cmd = f"bcftools view --threads {args.threads} -S {quote(str(sample_file))} -Ou {quote(str(args.vcf))}"
    else:
        gt_header = outdir / "added_GT_format_header.txt"
        gt_header.write_text('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\\n')
        source_cmd = (
            f"bcftools annotate -h {quote(str(gt_header))} -Ou {quote(str(args.vcf))} | "
            f"bcftools view --threads {args.threads} -S {quote(str(sample_file))} -Ou"
        )
    cmd = source_cmd + " | bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT[\\t%GT]\\n'"

    map_rows = []
    chrom_to_group = {}
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

        if chrom not in chrom_to_group:
            chrom_to_group[chrom] = len(chrom_to_group) + 1
        snp_id = f"{chrom}_{pos}_{ref}_{alt}"
        group_id = chrom_to_group[chrom]
        numeric_chrom = group_id
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
    classic.to_csv(outdir / "diploid.map", sep="\t", header=False, index=False)

    print("[6/6] write metadata")
    meta = {
        "vcf": "resources/input.vcf.gz",
        "pheno": "resources/phenotypes",
        "species": args.species,
        "sample_id_col": args.sample_id_col,
        "n_snps": int(n_snps),
        "n_samples": int(n_samples),
        "n_groups": int(len(chrom_to_group)),
        "chromosomes": [
            chrom for chrom, _ in sorted(chrom_to_group.items(), key=lambda kv: kv[1])
        ],
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
