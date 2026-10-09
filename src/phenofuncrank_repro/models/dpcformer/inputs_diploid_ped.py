#!/usr/bin/env python3
"Build diploid DPCformer inputs from PLINK PED + MAP files."

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ALLELE_TO_BIT = {
    ord("A"): 1,
    ord("C"): 2,
    ord("G"): 4,
    ord("T"): 8,
}
BIT_TO_BASE = {
    1: "A",
    2: "C",
    4: "G",
    8: "T",
}


def natural_chrom_key(chrom: object):
    text = str(chrom)
    parts = re.split(r"(\d+)", text)
    key = []
    for part in parts:
        if part == "":
            continue
        key.append(int(part) if part.isdigit() else part.lower())
    return key


def read_pheno(path: Path, sep_arg: str, sample_id_col: str, aggregate: str) -> pd.DataFrame:
    sep_arg = sep_arg.lower()
    if sep_arg == "auto":
        kwargs = {"sep": "," if str(path).endswith(".csv") else "\t"}
    elif sep_arg == "comma":
        kwargs = {"sep": ","}
    elif sep_arg == "tab":
        kwargs = {"sep": "\t"}
    elif sep_arg == "whitespace":
        kwargs = {"sep": r"\s+", "engine": "python"}
    else:
        kwargs = {"sep": sep_arg}
    pheno = pd.read_csv(path, **kwargs)
    if sample_id_col not in pheno.columns:
        raise RuntimeError(f"{path} missing sample ID column {sample_id_col!r}")
    pheno[sample_id_col] = pheno[sample_id_col].astype(str)
    if aggregate == "mean":
        numeric = [
            c for c in pheno.columns
            if c != sample_id_col and pd.to_numeric(pheno[c], errors="coerce").notna().any()
        ]
        nonnumeric = [c for c in pheno.columns if c not in numeric and c != sample_id_col]
        pieces = []
        if numeric:
            num = pheno[[sample_id_col] + numeric].copy()
            for col in numeric:
                num[col] = pd.to_numeric(num[col], errors="coerce")
            pieces.append(num.groupby(sample_id_col, as_index=False).mean())
        if nonnumeric:
            pieces.append(pheno[[sample_id_col] + nonnumeric].groupby(sample_id_col, as_index=False).first())
        if len(pieces) == 1:
            pheno = pieces[0]
        else:
            pheno = pieces[0]
            for extra in pieces[1:]:
                pheno = pheno.merge(extra, on=sample_id_col, how="outer")
    elif aggregate != "none":
        raise RuntimeError(f"Unsupported aggregate mode: {aggregate}")
    return pheno.set_index(sample_id_col, drop=False)


def read_map(path: Path, chrom_label_mode: str) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if raw.shape[1] == 4:
        raw.columns = ["Chr", "snp_id", "map", "Pos"]
        m = raw
    elif raw.shape[1] == 6:
        raw.columns = ["Chr", "snp_id", "map", "Pos", "bim_a1", "bim_a2"]
        m = raw
    else:
        raise RuntimeError(f"{path} must have 4 MAP columns or 6 BIM columns; observed {raw.shape[1]}")
    if m.empty:
        raise RuntimeError(f"Empty MAP: {path}")
    m["Chr"] = m["Chr"].astype(str)
    m["snp_id"] = m["snp_id"].astype(str)
    if chrom_label_mode == "snpid_prefix_underscore":
        inferred = m["snp_id"].str.extract(r"^([^_]+)_", expand=False)
        if inferred.isna().any():
            raise RuntimeError("Could not infer chromosome from every snp_id with prefix_underscore mode")
        m["Chr"] = inferred.astype(str)
    elif chrom_label_mode == "snpid_chr_dot":
        inferred = m["snp_id"].str.extract(r"^(chr[^.]+)\\.", expand=False)
        if inferred.isna().any():
            raise RuntimeError("Could not infer chromosome from every snp_id with chr_dot mode")
        m["Chr"] = inferred.astype(str)
    elif chrom_label_mode != "map":
        raise RuntimeError(f"Unsupported chrom-label-mode: {chrom_label_mode}")
    m["Pos"] = pd.to_numeric(m["Pos"], errors="raise").astype(np.int64)
    chrom_order = {
        chrom: i + 1
        for i, chrom in enumerate(sorted(m["Chr"].drop_duplicates(), key=natural_chrom_key))
    }
    m["chromosome"] = m["Chr"].map(chrom_order).astype(int)
    m["group_id"] = m["chromosome"]
    m["variant_index"] = np.arange(len(m), dtype=np.int64)
    return m


def ped_variant_count(path: Path) -> int:
    with path.open() as fh:
        first = fh.readline().split()
    if len(first) < 6 or (len(first) - 6) % 2:
        raise RuntimeError(f"Invalid PED first line field count: {len(first)}")
    return (len(first) - 6) // 2


def ped_ids(path: Path) -> list[str]:
    ids = []
    with path.open() as fh:
        for line in fh:
            parts = line.split(maxsplit=2)
            if len(parts) >= 2:
                ids.append(parts[1])
    return ids


def genotype_bits_from_string(genotype_string: str, n_variants: int, sample_id: str) -> np.ndarray:
    raw = np.frombuffer(genotype_string.encode("ascii"), dtype=np.uint8)
    raw = raw[raw > 32]
    if raw.size != n_variants * 2:
        raise RuntimeError(
            f"PED genotype token count mismatch for {sample_id}: {raw.size}; "
            f"expected {n_variants * 2}"
        )
    bits = np.zeros(raw.size, dtype=np.uint8)
    for allele, bit in ALLELE_TO_BIT.items():
        bits[raw == allele] = bit
    return bits.reshape(n_variants, 2)


def allele_bit_count(mask: np.ndarray) -> np.ndarray:
    return (
        ((mask & 1) > 0).astype(np.uint8)
        + ((mask & 2) > 0).astype(np.uint8)
        + ((mask & 4) > 0).astype(np.uint8)
        + ((mask & 8) > 0).astype(np.uint8)
    )


def first_two_bits(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.zeros(mask.shape, dtype=np.uint8)
    alt = np.zeros(mask.shape, dtype=np.uint8)
    for bit in [1, 2, 4, 8]:
        has = (mask & bit) > 0
        take_ref = has & (ref == 0)
        ref[take_ref] = bit
        take_alt = has & (ref != bit) & (alt == 0)
        alt[take_alt] = bit
    return ref, alt


def encode_codes(ref_bit: np.ndarray, alt_bit: np.ndarray, usable: np.ndarray, bits: np.ndarray) -> np.ndarray:
    a = bits[:, 0]
    b = bits[:, 1]
    codes = np.zeros(bits.shape[0], dtype=np.uint8)
    hom_ref = usable & (a == ref_bit) & (b == ref_bit)
    het = usable & (((a == ref_bit) & (b == alt_bit)) | ((a == alt_bit) & (b == ref_bit)))
    hom_alt = usable & (a == alt_bit) & (b == alt_bit)
    codes[hom_ref] = 1
    codes[het] = 2
    codes[hom_alt] = 3
    return codes


def parse_matched_ped_line(line: str, matched: set[str]) -> tuple[str | None, str | None]:
    head = line.rstrip("\n").split(maxsplit=6)
    if len(head) < 7:
        return None, None
    iid = head[1]
    if iid not in matched:
        return None, None
    return iid, head[6]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ped", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--pheno", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample-id-col", required=True)
    ap.add_argument("--pheno-sep", default="auto")
    ap.add_argument("--aggregate-duplicates", choices=["mean", "none"], default="mean")
    ap.add_argument(
        "--chrom-label-mode",
        choices=["map", "snpid_prefix_underscore", "snpid_chr_dot"],
        default="map",
        help=(
            "How to name chromosomes in the output SNP map. Use snpid_prefix_underscore "
            "for MAP rows such as Ca1_79 when the reference VCF uses Ca1."
        ),
    )
    ap.add_argument("--species", default="diploid")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ped = Path(args.ped)
    map_path = Path(args.map)

    print("[1/7] read phenotype")
    pheno = read_pheno(Path(args.pheno), args.pheno_sep, args.sample_id_col, args.aggregate_duplicates)

    print("[2/7] read MAP and PED dimensions")
    m = read_map(map_path, args.chrom_label_mode)
    n_variants = ped_variant_count(ped)
    if len(m) != n_variants:
        raise RuntimeError(f"MAP rows={len(m)} but PED variants={n_variants}")

    print("[3/7] match samples")
    ids = ped_ids(ped)
    matched = [iid for iid in ids if iid in pheno.index]
    if not matched:
        raise RuntimeError("No matched PED samples in phenotype table")
    matched_set = set(matched)
    (outdir / "matched_samples.vcf_order.txt").write_text("\n".join(matched) + "\n")
    matched_pheno = pheno.loc[matched].copy()
    if "LINE" not in matched_pheno.columns:
        matched_pheno.insert(0, "LINE", matched_pheno[args.sample_id_col].astype(str).values)
    matched_pheno.to_csv(outdir / "matched_phenotypes.tsv", sep="\t", index=False)
    print(f"[INFO] PED samples={len(ids):,}; matched={len(matched):,}")

    print("[4/7] infer biallelic bases from matched PED rows")
    allele_mask = np.zeros(n_variants, dtype=np.uint8)
    with ped.open() as fh:
        for row_i, line in enumerate(fh, start=1):
            iid, genotype_string = parse_matched_ped_line(line, matched_set)
            if iid is None:
                continue
            bits = genotype_bits_from_string(genotype_string, n_variants, iid)
            allele_mask |= bits[:, 0] | bits[:, 1]
            if row_i % 1000 == 0:
                print(f"[INFO] allele pass scanned PED rows={row_i:,}")

    bit_counts = allele_bit_count(allele_mask)
    ref_bit, alt_bit = first_two_bits(allele_mask)
    usable = bit_counts == 2
    multi = bit_counts > 2
    print(f"[INFO] biallelic usable variants={int(usable.sum()):,}/{n_variants:,}")

    print("[5/7] write genotype memmap")
    geno_path = outdir / "cultivar_genotype_codes.variant_by_sample.uint8.memmap"
    geno = np.memmap(geno_path, mode="w+", dtype=np.uint8, shape=(n_variants, len(matched)))
    matched_index = {iid: i for i, iid in enumerate(matched)}
    with ped.open() as fh:
        for row_i, line in enumerate(fh, start=1):
            iid, genotype_string = parse_matched_ped_line(line, matched_set)
            if iid is None:
                continue
            bits = genotype_bits_from_string(genotype_string, n_variants, iid)
            geno[:, matched_index[iid]] = encode_codes(ref_bit, alt_bit, usable, bits)
            if row_i % 1000 == 0:
                print(f"[INFO] genotype pass scanned PED rows={row_i:,}")
    geno.flush()

    print("[6/7] write SNP map")
    m["Ref"] = [BIT_TO_BASE.get(int(x), "N") for x in ref_bit]
    m["Alt"] = [BIT_TO_BASE.get(int(x), "N") for x in alt_bit]
    m.loc[multi, "Alt"] = "MULTI"
    m["snp_id"] = m["Chr"].astype(str) + "_" + m["Pos"].astype(str) + "_" + m["Ref"] + "_" + m["Alt"]
    m.to_csv(outdir / "cultivar.snp_map.tsv.gz", sep="\t", index=False, compression="gzip")
    m[["chromosome", "snp_id", "map", "Pos"]].to_csv(
        outdir / "diploid.map", sep="\t", header=False, index=False
    )

    print("[7/7] write metadata")
    meta = {
        "ped": "resources/input.ped",
        "map": "resources/input.map",
        "pheno": "resources/phenotypes",
        "species": args.species,
        "sample_id_col": args.sample_id_col,
        "n_snps": int(n_variants),
        "n_samples": int(len(matched)),
        "n_groups": int(m["group_id"].max()),
        "chromosomes": m.sort_values("group_id")["Chr"].drop_duplicates().astype(str).tolist(),
        "genotype_memmap": geno_path.name,
        "snp_map": "cultivar.snp_map.tsv.gz",
        "samples": "matched_samples.vcf_order.txt",
        "phenotypes": "matched_phenotypes.tsv",
        "dtype": "uint8",
        "shape": [int(n_variants), int(len(matched))],
        "ped_allele_orientation": "Ref/Alt inferred from first two observed A/C/G/T alleles in matched PED samples",
        "chrom_label_mode": args.chrom_label_mode,
        "multi_or_non_biallelic_variants": int((~usable).sum()),
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("[DONE]", outdir)


if __name__ == "__main__":
    main()
