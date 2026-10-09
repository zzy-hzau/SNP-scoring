"PC5 dosage association; exact chromosome/allele-aware mapping."
import argparse,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from phenofuncrank_repro.prioritization.allele_genotypes import norm_chrom
N_PC = 5
LOOKUP = np.array([np.nan,0.0,1.0,2.0])

def canonical_map(snp_map, chromosome_style):
    out = snp_map.copy()
    aliases = {x: norm_chrom(x, {"chromosome_style": chromosome_style}) for x in out["Chr"].astype(str).unique()}
    out["_canonical_chr"] = out["Chr"].astype(str).map(aliases)
    return out


def exact_variant_row(snp_map, snp_id, chromosome_style):
    chrom, rest = str(snp_id).split(":", 1)
    position, ref, alt = rest.split("_", 2)
    chromosome = norm_chrom(chrom, {"chromosome_style": chromosome_style})
    same_position = snp_map["_canonical_chr"].eq(chromosome) & pd.to_numeric(snp_map["Pos"]).eq(int(position))
    compatible = ((snp_map["Ref"].astype(str).eq(ref) & snp_map["Alt"].astype(str).eq(alt))
                  | (snp_map["Ref"].astype(str).eq(alt) & snp_map["Alt"].astype(str).eq(ref)))
    hits = snp_map.loc[same_position & compatible]
    if hits.empty:
        return None
    index_column = "variant_index" if "variant_index" in hits.columns else "map"
    if hits[index_column].nunique() != 1:
        raise ValueError(f"{snp_id}: ambiguous exact genotype mapping")
    return hits.iloc[0]


def to_dosage(codes):
    return np.take(LOOKUP, codes.astype(np.int64), axis=0)

def ols(y, X):
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    n, k = X.shape
    s2 = float(resid @ resid) / (n - k)
    se = np.sqrt(np.diag(XtX_inv) * s2)
    tv = beta / se
    p = 2.0 * tdist.sf(np.abs(tv), n - k)
    r2 = 1.0 - float(resid @ resid) / float((y - y.mean()) @ (y - y.mean()))
    return beta, se, tv, p, r2

def run_trait(tag: str, sp_dir: str, geno_path: Path, snp_map_path: Path,
              pheno_path: Path, samples_path: Path, pheno_col: str,
              pc_cache: Path, snp_ids: list[str]) -> pd.DataFrame:
    print("=" * 78)
    print(f"{tag.upper()}: {pheno_col} ~ SNP + PC1-{N_PC}")
    print("=" * 78)

    snp_map = pd.read_csv(snp_map_path, sep="\t", low_memory=False)
    snp_map = canonical_map(snp_map, sp_dir)
    n_snps = len(snp_map)
    n_samp = os.path.getsize(geno_path) // n_snps
    geno = np.memmap(geno_path, dtype="uint8", mode="r", shape=(n_snps, n_samp))
    print(f"Genotype: {geno.shape}")

    pheno = pd.read_csv(pheno_path, sep="\t")
    sids = [l.strip() for l in open(samples_path) if l.strip()]
    assert len(sids) == n_samp, f"sample list {len(sids)} != {n_samp}"
    pid = dict(zip(pheno["LINE"], pd.to_numeric(pheno[pheno_col], errors="coerce")))
    y_all = np.array([pid.get(s, np.nan) for s in sids], dtype=float)
    print(f"Phenotype available: {int(np.sum(~np.isnan(y_all)))}/{n_samp}")

    if pc_cache.exists():
        PCs = np.load(pc_cache)
        assert PCs.shape == (n_samp, N_PC), f"PC shape {PCs.shape} != {(n_samp, N_PC)}"
        print(f"Loaded cached PCs: {PCs.shape}")
    else:
        raise FileNotFoundError(f"PC cache missing: {pc_cache}")


    rows = []
    for sid in snp_ids:
        r = exact_variant_row(snp_map, sid, sp_dir)
        if r is None:
            rows.append({"snp_id": sid, "n": 0, "note": "NOT_IN_GENOTYPE"})
            continue
        idx = int(r["variant_index"]) if "variant_index" in snp_map.columns else int(r["map"])
        ref, alt = str(r["Ref"]), str(r["Alt"])
        want_ref = sid.split("_")[1]
        want_alt = sid.split("_")[2] if len(sid.split("_")) > 2 else sid.split("_")[-1]
        flip = (ref == want_alt and alt == want_ref)
        d = to_dosage(geno[idx, :])
        if flip:
            d = 2.0 - d
        v = ~np.isnan(d) & ~np.isnan(y_all)
        n = int(v.sum())
        y = y_all[v]
        gv = d[v]
        if n < 10:
            print(f"  {sid}: too few samples ({n})")
            rows.append({"snp_id": sid, "n": n, "note": "TOO_FEW_SAMPLES"})
            continue
        maf = min(gv.mean() / 2, 1 - gv.mean() / 2)

        X1 = np.column_stack([np.ones(n), gv])
        b1, _, _, p1, _ = ols(y, X1)
        X2 = np.column_stack([np.ones(n), gv, PCs[v, :N_PC]])
        b2, se2, _, p2, r2_adj = ols(y, X2)
        Xr = np.column_stack([np.ones(n), PCs[v, :N_PC]])
        _, _, _, _, r2_red = ols(y, Xr)
        part_r2 = (r2_adj - r2_red) / (1.0 - r2_red) if r2_red < 1.0 else np.nan

        p1s, p2s = p1[1], p2[1]
        sig = "***" if p2s < 0.001 else "**" if p2s < 0.01 else "*" if p2s < 0.05 else "ns"
        print(f"{sid:<24s} maf={maf:.3f} n={n:>5d} naive β={b1[1]:>10.4f} p={p1s:>10.2e}"
              f" adj β={b2[1]:>10.4f} se={se2[1]:>8.4f} p={p2s:>10.2e} partR2={part_r2:>8.3f} {sig}")
        rows.append({"snp_id": sid, "maf": maf, "n": n, "flip": flip,
                     "beta_naive": b1[1], "p_naive": p1s,
                     "beta_adj": b2[1], "se_adj": se2[1], "p_adj": p2s,
                     "partial_r2": part_r2, "note": ""})

    res = pd.DataFrame(rows)
    for field in ["p_adj", "maf", "flip", "beta_naive", "p_naive", "beta_adj", "se_adj", "partial_r2"]:
        if field not in res.columns:
            res[field] = np.nan
    res["sig_005"] = res["p_adj"].astype(float) < 0.05
    res["sig_001"] = res["p_adj"].astype(float) < 0.001
    res["bonf"] = res["p_adj"].astype(float) < (0.05 / max(1, len(res)))
    return res

def cli():
    ap = argparse.ArgumentParser(description="PC1-PC5 association for supplied frozen candidate genotypes.")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--chromosome-style", choices=["cotton_ad","chickpea_ca"], required=True)
    ap.add_argument("--genotypes", type=Path, required=True)
    ap.add_argument("--snp-map", type=Path, required=True)
    ap.add_argument("--phenotypes", type=Path, required=True)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--phenotype-column", required=True)
    ap.add_argument("--pcs", type=Path, required=True)
    ap.add_argument("--candidate-table", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    table = pd.read_parquet(args.candidate_table).sort_values("priority_rank").head(args.top_n)
    result = run_trait(args.tag,args.chromosome_style,args.genotypes,args.snp_map,
                       args.phenotypes,args.samples,args.phenotype_column,args.pcs,
                       table["snp_id"].astype(str).tolist())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,sep="\t",index=False)

if __name__ == "__main__":
    cli()
