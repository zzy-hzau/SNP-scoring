#!/usr/bin/env python3
"Compute OLS associations for discovery-selected Core SNPs in the RFGB panel."

from __future__ import annotations

import argparse
from pathlib import Path

import cyvcf2
import numpy as np
import pandas as pd
from scipy.stats import t as tdist


def ols(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """Return coefficient, SE and two-sided Wald P for dosage (column 1)."""
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    n, k = x.shape
    s2 = float(resid @ resid) / (n - k)
    se = np.sqrt(np.diag(xtx_inv) * s2)
    tval = beta[1] / se[1]
    p = 2.0 * tdist.sf(abs(tval), n - k)
    return {"beta": float(beta[1]), "se": float(se[1]), "t": float(tval), "p": float(p)}


def load_pcs(path: Path) -> pd.DataFrame:
    pc = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    if pc.shape[1] < 7:
        raise ValueError(f"Expected FID/IID + PC1..PC5 in {path}, got {pc.shape[1]} columns")
    pc = pc.iloc[:, :7].copy()
    pc.columns = ["FID", "IID", "PC1", "PC2", "PC3", "PC4", "PC5"]
    pc["IID"] = pc["IID"].astype(str)
    if pc["IID"].duplicated().any():
        raise ValueError("Duplicate IID in external PCA file")
    return pc


def get_variant(vcf: cyvcf2.VCF, chrom: int, pos: int, ref: str, alt: str):
    target_chroms = [f"chr{chrom:02d}", str(chrom), f"Chr{chrom}"]
    for c in target_chroms:
        try:
            variants = vcf(f"{c}:{pos}-{pos}")
        except Exception:
            continue
        for var in variants:
            if int(var.POS) == pos and var.REF == ref and alt in list(var.ALT):
                return var, c
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", type=Path, required=True)
    ap.add_argument("--vcf", type=Path, required=True)
    ap.add_argument("--pheno", type=Path, required=True)
    ap.add_argument("--pc", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--role", choices=["core_max_abs_DPC", "representative"], default="core_max_abs_DPC")
    args = ap.parse_args()

    core = pd.read_csv(args.core, sep="\t")
    core = core.loc[core["role"].eq(args.role)].copy()
    core = core.sort_values(["rank", "ldcc"]).drop_duplicates("ldcc", keep="first")
    core["tested_snp_id"] = core["tested_snp_id"].astype(str)

    ph = pd.read_csv(args.pheno, sep="\t")
    ph["vcf_id"] = ph["vcf_id"].astype(str)
    ph["plant_height_cm"] = pd.to_numeric(ph["plant_height_cm"], errors="coerce")
    ph = ph.dropna(subset=["plant_height_cm"]).drop_duplicates("vcf_id")
    pc = load_pcs(args.pc)
    panel = ph.merge(pc, left_on="vcf_id", right_on="IID", how="inner", validate="one_to_one")
    if len(panel) != len(ph):
        missing = sorted(set(ph["vcf_id"]) - set(panel["vcf_id"]))
        raise ValueError(f"Phenotype/PCA mismatch: {len(missing)} samples missing PCs")
    panel = panel.reset_index(drop=True)

    vcf = cyvcf2.VCF(str(args.vcf))
    sample_index = {str(s): i for i, s in enumerate(vcf.samples)}
    idx = np.array([sample_index.get(s, -1) for s in panel["vcf_id"]], dtype=int)
    if np.any(idx < 0):
        missing = panel.loc[idx < 0, "vcf_id"].tolist()
        raise ValueError(f"External samples missing from VCF: {missing[:5]} (n={len(missing)})")

    y_all = panel["plant_height_cm"].to_numpy(float)
    pcs_all = panel[["PC1", "PC2", "PC3", "PC4", "PC5"]].to_numpy(float)
    rows = []
    for row in core.itertuples(index=False):
        sid = str(row.tested_snp_id)
        parts = sid.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Cannot parse SNP ID {sid}")
        chrom, rest = parts
        pos_s, ref, alt = rest.split("_", 2)
        var, used_chrom = get_variant(vcf, int(chrom), int(pos_s), ref, alt)
        base = {
            "rank": int(row.rank), "ldcc": row.ldcc, "representative_snp_id": row.representative_snp_id,
            "core_snp_id": sid, "chrom": int(chrom), "pos": int(pos_s), "ref": ref, "alt": alt,
            "vcf_chrom": used_chrom or "", "n_external": len(panel),
        }
        if var is None:
            base.update({"n_called": 0, "call_rate": 0.0, "maf": np.nan,
                         "beta_raw": np.nan, "p_raw": np.nan, "beta_pc": np.nan,
                         "se_pc": np.nan, "p_pc": np.nan, "partial_r2": np.nan,
                         "status": "variant_not_found"})
            rows.append(base)
            continue
        if len(var.ALT) != 1:
            raise ValueError(f"{sid}: native additive REF/ALT association requires normalized biallelic variants.")
        gts = np.asarray(var.genotypes, dtype=np.int16)[idx, :2]
        dosage = gts.sum(axis=1).astype(float)
        dosage[(gts < 0).any(axis=1)] = np.nan
        valid = np.isfinite(dosage) & np.isfinite(y_all) & np.isfinite(pcs_all).all(axis=1)
        n = int(valid.sum())
        g = dosage[valid]
        y = y_all[valid]
        pcv = pcs_all[valid]
        maf = float(min(np.nanmean(g) / 2.0, 1.0 - np.nanmean(g) / 2.0)) if n else np.nan
        base.update({"n_called": n, "call_rate": n / len(panel), "maf": maf})
        if n < 10 or np.nanstd(g) < 1e-12:
            base.update({"beta_raw": np.nan, "p_raw": np.nan, "beta_pc": np.nan,
                         "se_pc": np.nan, "p_pc": np.nan, "partial_r2": np.nan,
                         "status": "monomorphic_or_too_few_calls"})
        else:
            raw = ols(y, np.column_stack([np.ones(n), g]))
            adj = ols(y, np.column_stack([np.ones(n), g, pcv]))
            red = ols(y, np.column_stack([np.ones(n), pcv]))

            rss_full = float((y - np.column_stack([np.ones(n), g, pcv]) @ np.linalg.lstsq(
                np.column_stack([np.ones(n), g, pcv]), y, rcond=None)[0]) @
                (y - np.column_stack([np.ones(n), g, pcv]) @ np.linalg.lstsq(
                np.column_stack([np.ones(n), g, pcv]), y, rcond=None)[0]))
            rss_red = float((y - np.column_stack([np.ones(n), pcv]) @ np.linalg.lstsq(
                np.column_stack([np.ones(n), pcv]), y, rcond=None)[0]) @
                (y - np.column_stack([np.ones(n), pcv]) @ np.linalg.lstsq(
                np.column_stack([np.ones(n), pcv]), y, rcond=None)[0]))
            partial = (rss_red - rss_full) / rss_red if rss_red > 0 else np.nan
            base.update({"beta_raw": raw["beta"], "p_raw": raw["p"], "beta_pc": adj["beta"],
                         "se_pc": adj["se"], "p_pc": adj["p"], "partial_r2": partial,
                         "status": "tested"})
        rows.append(base)

    out = pd.DataFrame(rows).sort_values("rank")
    tested = out["status"].eq("tested")
    out["sig_raw_p05"] = tested & (out["p_raw"] < 0.05)
    out["sig_pc_p05"] = tested & (out["p_pc"] < 0.05)
    out["sig_pc_bonf"] = tested & (out["p_pc"] < 0.05 / max(1, int(tested.sum())))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)
    print(out.to_string(index=False))
    print(f"\nExternal samples: {len(panel)}")
    print(f"Unique LDCC core-max SNPs: {len(out)}; tested: {int(tested.sum())}")
    print(f"Raw P<0.05: {int(out.sig_raw_p05.sum())}; PC-adjusted P<0.05: {int(out.sig_pc_p05.sum())}; Bonferroni: {int(out.sig_pc_bonf.sum())}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
