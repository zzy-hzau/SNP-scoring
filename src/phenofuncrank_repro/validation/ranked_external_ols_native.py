#!/usr/bin/env python3
"Compute OLS associations for ranked SNPs in the RFGB panel."
from __future__ import annotations
import sys
from pathlib import Path
import cyvcf2
import numpy as np
import pandas as pd
from scipy.stats import t as tdist


def ols(y, x):
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    n, k = x.shape
    s2 = float(resid @ resid) / (n - k)
    se = np.sqrt(np.diag(xtx_inv) * s2)
    tval = beta[1] / se[1]
    p = 2.0 * tdist.sf(abs(tval), n - k)
    return {"beta": float(beta[1]), "se": float(se[1]), "t": float(tval), "p": float(p)}


def partial_r2(y, g, pcv):
    n = len(y)
    Xf = np.column_stack([np.ones(n), g, pcv])
    Xr = np.column_stack([np.ones(n), pcv])
    bf = np.linalg.lstsq(Xf, y, rcond=None)[0]
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    rss_f = float(np.sum((y - Xf @ bf) ** 2))
    rss_r = float(np.sum((y - Xr @ br) ** 2))
    return (rss_r - rss_f) / rss_r if rss_r > 0 else np.nan


def get_variant(vcf, chrom, pos, ref, alt):
    target = [f"chr{int(chrom):02d}"]
    for c in target:
        try:
            for var in vcf(f"{c}:{pos}-{pos}"):
                if int(var.POS) == pos and var.REF == ref and alt in list(var.ALT):
                    return var, c, False
        except Exception:
            continue

    for c in target:
        try:
            for var in vcf(f"{c}:{pos}-{pos}"):
                if int(var.POS) == pos and var.REF == alt and ref in list(var.ALT):
                    return var, c, True
        except Exception:
            continue
    return None, None, None


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Association of a frozen ranked Rice PH candidate table in an external panel.")
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--vcf", type=Path, required=True)
    ap.add_argument("--pheno", type=Path, required=True)
    ap.add_argument("--pc", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()
    VCF, PHENO, PC = args.vcf, args.pheno, args.pc
    top = pd.read_parquet(args.candidates)
    top = top.sort_values("priority_rank").head(args.top_n).reset_index(drop=True)
    ph = pd.read_csv(PHENO, sep="\t")
    ph["vcf_id"] = ph["vcf_id"].astype(str)
    ph["plant_height_cm"] = pd.to_numeric(ph["plant_height_cm"], errors="coerce")
    ph = ph.dropna(subset=["plant_height_cm"]).drop_duplicates("vcf_id")
    pc = pd.read_csv(PC, sep=r"\s+", comment="#", header=None)
    pc = pc.iloc[:, :7].copy()
    pc.columns = ["FID", "IID", "PC1", "PC2", "PC3", "PC4", "PC5"]
    pc["IID"] = pc["IID"].astype(str)
    panel = ph.merge(pc, left_on="vcf_id", right_on="IID", how="inner", validate="one_to_one")
    if len(panel) != len(ph):
        raise ValueError(f"phenotype/PCA mismatch: {len(ph)-len(panel)} samples")
    panel = panel.reset_index(drop=True)
    vcf = cyvcf2.VCF(str(VCF))
    sample_index = {str(s): i for i, s in enumerate(vcf.samples)}
    idx = np.array([sample_index.get(s, -1) for s in panel["vcf_id"]], dtype=int)
    if np.any(idx < 0):
        raise ValueError(f"missing samples in VCF: {int((idx<0).sum())}")
    y_all = panel["plant_height_cm"].to_numpy(float)
    pcs_all = panel[["PC1", "PC2", "PC3", "PC4", "PC5"]].to_numpy(float)
    rows = []
    for r in top.itertuples(index=False):
        chrom, rest = str(r.snp_id).split(":", 1)
        pos_s, ref, alt = rest.split("_", 2)
        var, used_chrom, flipped = get_variant(vcf, chrom, int(pos_s), ref, alt)
        base = {
            "tag": "final",
            "priority_rank": int(r.priority_rank),
            "snp_id": str(r.snp_id),
            "chrom": int(chrom),
            "pos": int(pos_s),
            "ref": ref,
            "alt": alt,
            "trait_evidence_type": str(r.trait_evidence_type),
            "B": float(r.B),
            "allele_flipped": bool(flipped) if flipped is not None else None,
        }
        if var is None:
            base.update({"n": 0, "call_rate": 0.0, "maf": np.nan, "status": "variant_not_found",
                         "beta_raw": np.nan, "p_raw": np.nan, "beta_pc": np.nan, "se_pc": np.nan,
                         "p_pc": np.nan, "partial_r2": np.nan, "p_pc_bonf": np.nan})
            rows.append(base)
            continue
        if len(var.ALT) != 1:
            raise ValueError(f"{r.snp_id}: normalized biallelic input is required.")
        gts = np.asarray(var.genotypes, dtype=np.int16)[idx, :2]
        dosage = gts.sum(axis=1).astype(float)
        dosage[(gts < 0).any(axis=1)] = np.nan
        if flipped:
            dosage = 2.0 - dosage
        valid = np.isfinite(dosage) & np.isfinite(y_all) & np.isfinite(pcs_all).all(axis=1)
        n = int(valid.sum())
        g = dosage[valid]
        y = y_all[valid]
        pcv = pcs_all[valid]
        maf = float(min(np.nanmean(g) / 2.0, 1.0 - np.nanmean(g) / 2.0)) if n else np.nan
        base.update({"n": n, "call_rate": n / len(panel), "maf": maf})
        if n < 10 or np.nanstd(g) < 1e-12:
            base.update({"beta_raw": np.nan, "p_raw": np.nan, "beta_pc": np.nan, "se_pc": np.nan,
                         "p_pc": np.nan, "partial_r2": np.nan, "p_pc_bonf": np.nan,
                         "status": "monomorphic_or_too_few_calls"})
        else:
            raw = ols(y, np.column_stack([np.ones(n), g]))
            adj = ols(y, np.column_stack([np.ones(n), g, pcv]))
            pr2 = partial_r2(y, g, pcv)
            base.update({"beta_raw": raw["beta"], "p_raw": raw["p"], "beta_pc": adj["beta"],
                         "se_pc": adj["se"], "p_pc": adj["p"], "partial_r2": pr2,
                         "status": "tested"})
        rows.append(base)
    out = pd.DataFrame(rows).sort_values("priority_rank")
    ntest = int((out.status == "tested").sum())
    out["p_pc_bonf"] = out["p_pc"] * ntest
    out["p_pc_bonf"] = out["p_pc_bonf"].clip(upper=1.0)
    out["sig_raw_p05"] = (out.status == "tested") & (out.p_raw < 0.05)
    out["sig_pc_p05"] = (out.status == "tested") & (out.p_pc < 0.05)
    out["sig_pc_bonf"] = (out.status == "tested") & (out.p_pc_bonf < 0.05)
    dst = args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, sep="\t", index=False)
    print(out[["priority_rank", "snp_id", "status", "n", "call_rate", "maf", "beta_raw", "p_raw", "beta_pc", "p_pc", "partial_r2", "p_pc_bonf"]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nexternal n={len(panel)}; tested={ntest}")
    print(f"raw P<0.05: {int(out.sig_raw_p05.sum())}; PC P<0.05: {int(out.sig_pc_p05.sum())}; Bonferroni: {int(out.sig_pc_bonf.sum())}")
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
