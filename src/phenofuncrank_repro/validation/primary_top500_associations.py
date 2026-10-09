"Primary Figure4b/c statistics; mathematical functions."

import argparse,json

from pathlib import Path

import numpy as np

import pandas as pd

from scipy.stats import t as tdist

N_TOP = 500

BONF = 0.05 / N_TOP

def bh_q(pv: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q values, NaN-safe."""
    pv = np.asarray(pv, dtype=float)
    q = np.full(len(pv), np.nan)
    ok = np.isfinite(pv)
    if ok.any():
        order = np.argsort(pv[ok])
        ranked = pv[ok][order] * ok.sum() / (np.arange(1, ok.sum() + 1))
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        qq = np.empty(ok.sum())
        qq[order] = np.clip(ranked, 0, 1)
        q[ok] = qq
    return q

def allelic_stats_matrix(D: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """D: (m, n) dosage with NaN; y: (n,).  Vectorised over the m SNPs."""
    m, n = D.shape
    okd = np.isfinite(D)
    oky = np.isfinite(y)
    yv = np.where(oky, y, 0.0)
    cnt = np.zeros((m, 3))
    s = np.zeros((m, 3))
    ss = np.zeros((m, 3))
    for g in (0, 1, 2):
        mask = (D == g) & oky[None, :]
        cnt[:, g] = mask.sum(axis=1)
        s[:, g] = (mask * yv[None, :]).sum(axis=1)
        ss[:, g] = (mask * (yv[None, :] ** 2)).sum(axis=1)

    use_ref = cnt[:, 0] >= 2
    iA = np.where(use_ref, 0, 1)
    rows = np.arange(m)
    nA, sA, ssA = cnt[rows, iA], s[rows, iA], ss[rows, iA]
    nB, sB, ssB = cnt[:, 2], s[:, 2], ss[:, 2]

    good = (nA >= 2) & (nB >= 2)
    out = pd.DataFrame({
        "group1": np.where(use_ref, "REF/REF", "HET"),
        "group2": "ALT/ALT",
        "group1_n": nA.astype(int), "group2_n": nB.astype(int),
        "n_evaluable": (cnt.sum(axis=1)).astype(int),
    })
    out["group1_mean"] = np.where(good, sA / np.where(nA > 0, nA, 1), np.nan)
    out["group2_mean"] = np.where(good, sB / np.where(nB > 0, nB, 1), np.nan)
    varA = np.where(good, (ssA - sA**2 / np.where(nA > 0, nA, 1)) / np.maximum(nA - 1, 1), np.nan)
    varB = np.where(good, (ssB - sB**2 / np.where(nB > 0, nB, 1)) / np.maximum(nB - 1, 1), np.nan)
    out["group1_sd"] = np.sqrt(np.maximum(varA, 0))
    out["group2_sd"] = np.sqrt(np.maximum(varB, 0))
    out["delta"] = out["group2_mean"] - out["group1_mean"]

    se2 = varA / np.where(nA > 0, nA, 1) + varB / np.where(nB > 0, nB, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tval = out["delta"].to_numpy() / np.sqrt(se2)
        dfnum = se2**2
        dfden = (varA / np.where(nA > 0, nA, 1)) ** 2 / np.maximum(nA - 1, 1) \
            + (varB / np.where(nB > 0, nB, 1)) ** 2 / np.maximum(nB - 1, 1)
        df = np.where(dfden > 0, dfnum / np.where(dfden > 0, dfden, 1), np.nan)
        p = 2 * tdist.sf(np.abs(tval), df)
    out["t_welch"] = tval
    out["df_welch"] = df
    out["p_welch"] = np.where(good, p, np.nan)

    tot = nA + nB
    both_mean = (sA + sB) / np.where(tot > 0, tot, 1)
    ss_t = (ssA + ssB) - (sA + sB) ** 2 / np.where(tot > 0, tot, 1)
    ss_b = nA * (out["group1_mean"] - both_mean) ** 2 + nB * (out["group2_mean"] - both_mean) ** 2
    out["r2_group"] = np.where((ss_t > 0) & good, ss_b / np.where(ss_t > 0, ss_t, 1), np.nan)
    return out

def ols_beta_p(y: np.ndarray, X: np.ndarray) -> tuple[float, float, float]:
    """Return (beta_g, p_g, rss)."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = max(1, n - k)
    rss = float(resid @ resid)
    sigma2 = rss / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    se1 = se[1] if se[1] > 0 else np.inf
    tval = beta[1] / se1
    return float(beta[1]), float(2 * tdist.sf(abs(tval), dof)), rss

def pc_results(D: np.ndarray, y: np.ndarray, pcs: np.ndarray) -> pd.DataFrame:
    m = D.shape[0]
    rows = []
    pc5 = pcs[:, :5]
    for i in range(m):
        d = D[i]
        valid = np.isfinite(d) & np.isfinite(y) & np.all(np.isfinite(pc5), axis=1)
        n = int(valid.sum())
        rec = {"n": n}
        if n <= 8 or np.nanstd(d[valid]) == 0:
            rec.update({"beta_naive": np.nan, "p_naive": np.nan,
                        "beta_adj": np.nan, "p_adj": np.nan, "partial_r2": np.nan})
            rows.append(rec)
            continue
        yv, gv, pv = y[valid], d[valid], pc5[valid]
        X0 = np.column_stack([np.ones(n), gv])
        Xr = np.column_stack([np.ones(n), pv])
        X1 = np.column_stack([np.ones(n), gv, pv])
        b0, p0, _ = ols_beta_p(yv, X0)
        b1, p1, rss1 = ols_beta_p(yv, X1)
        _, _, rss_r = ols_beta_p(yv, Xr)
        rec.update({"beta_naive": b0, "p_naive": p0, "beta_adj": b1, "p_adj": p1,
                    "partial_r2": (rss_r - rss1) / rss_r if rss_r > 0 else np.nan})
        rows.append(rec)
    return pd.DataFrame(rows)

def run_bundle(bundle, variants, output):
    with np.load(bundle, allow_pickle=False) as b:
        D, y, pcs = b['dosage'], b['phenotype'], b['pcs']
    catalog = pd.read_csv(variants, sep='\t')
    if D.ndim != 2 or y.shape != (D.shape[1],) or pcs.shape != (D.shape[1], 5):
        raise ValueError('Require dosage(m,n), phenotype(n), PCs(n,5) in original sample order')
    if len(catalog) != len(D) or catalog['rank'].duplicated().any():
        raise ValueError('Variants must identify every dosage row by unique original rank')
    finite = D[np.isfinite(D)]
    if not np.isin(finite, [0., 1., 2.]).all():
        raise ValueError('Dosage must be ALT-oriented 0/1/2 or NaN')
    group = allelic_stats_matrix(D,y)
    association = pc_results(D,y,pcs)
    for frame, pfield in [(group,'p_welch'),(association,'p_adj')]:
        for column in catalog.columns: frame[column] = catalog[column].to_numpy()
        frame['p_bonf_500'] = (frame[pfield]*N_TOP).clip(upper=1.)
        frame['sig_bonf_500'] = frame[pfield] < BONF

        if len(frame) == N_TOP:
            frame['q_bh'] = bh_q(frame[pfield].to_numpy(float))
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    group.to_csv(output/'genotype_group_statistics.tsv',sep='\t',index=False)
    association.to_csv(output/'dosage_PC5_associations.tsv',sep='\t',index=False)
    (output/'analysis_scope.json').write_text(json.dumps({'rows':len(D),'samples':D.shape[1],'Bonferroni_family_size':N_TOP,'BH_computed':len(D)==N_TOP,'PCs_refit':False,'source_mathematical_functions_preserved':True},indent=2)+'\n')
    return group,association


def main():
    parser=argparse.ArgumentParser(description='Primary Top500 Figure4 statistics from supplied aligned dosage, phenotype and fixed PCs.')
    parser.add_argument('--bundle',type=Path,required=True,help='NPZ: dosage, phenotype, pcs arrays; allow_pickle=False')
    parser.add_argument('--variants',type=Path,required=True,help='TSV: unique original rank plus variant identity and orientation')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run_bundle(args.bundle,args.variants,args.output)


if __name__ == '__main__':
    main()

