"additive-QTN/polygenic simulation, with caller-supplied real panels."
import numpy as np
import pandas as pd

def simulate_rice(geno, qc, scored_ids, *, seed, n_qtn=5,
                   h2=0.5, polygenic_fraction=0.2, qtn_universe="scored"):
    N_VAR,N_SAM=geno.shape
    SEED,N_QTN,H2,H2_POLY_FRAC=seed,n_qtn,h2,polygenic_fraction
    qc=qc.copy()
    qc["snp_id"]=qc.snp_id.astype(str)
    qc["is_scored"]=qc.snp_id.isin(set(map(str,scored_ids)))
    qc["pass_gate"]=(qc.call_rate>=0.90)&(qc.mac>=10)
    universe=qc[qc.pass_gate if qtn_universe=="all" else (qc.is_scored&qc.pass_gate)].copy()
    rng = np.random.default_rng(SEED)
    pick = [rng.choice(g.index.to_numpy(), size=N_QTN, replace=False)
            for _, g in universe.groupby("chrom")]
    flat = np.concatenate(pick)
    qtn = universe.loc[np.sort(rng.choice(flat, size=N_QTN, replace=False))].copy()
    w = np.abs(rng.normal(size=len(qtn)))
    pve = H2 * w ** 2 / (w ** 2).sum()
    beta = np.sqrt(pve) * rng.choice([-1.0, 1.0], size=len(qtn))
    qtn["weight"] = w; qtn["pve"] = pve; qtn["beta"] = beta

    def dosage(vi):
        a = np.asarray(geno[vi])
        return np.where(a == 3, 2.0, np.where(a == 2, 1.0, 0.0))

    y = beta @ dosage(qtn.variant_index.to_numpy(np.int64))
    h2_poly = H2 * H2_POLY_FRAC
    h2_qtn = H2 - h2_poly
    print(f"  h2_QTN={h2_qtn:.3f}  h2_poly={h2_poly:.3f}  h2_total={H2:.3f}", flush=True)
    y = y * np.sqrt(h2_qtn / max(float(np.var(y)), 1e-12))
    gm = np.setdiff1d(rng.choice(N_VAR, size=100_000, replace=False),
                      qtn.variant_index.to_numpy(np.int64))
    Z = dosage(np.sort(gm)).astype(np.float32)
    Z -= Z.mean(axis=1, keepdims=True)
    sd = Z.std(axis=1, keepdims=True); sd[sd == 0] = 1.0
    Z /= sd
    K = (Z.T @ Z) / Z.shape[0]                       
    K /= np.mean(np.diag(K))
    del Z, sd

    
    ev, EV = np.linalg.eigh(K)
    ev = np.clip(ev, 1e-8, None)
    L = EV * np.sqrt(ev)
    xi = L @ rng.normal(size=N_SAM)
    if h2_poly > 0:
        xi *= np.sqrt(h2_poly / max(float(np.var(xi)), 1e-12))
    else:
        xi = np.zeros(N_SAM, dtype=np.float64)
    y = y + xi + rng.normal(size=N_SAM) * np.sqrt(max(1.0 - H2, 0.0))
    y = (y - y.mean()) / y.std()

    return qtn,y

def simulate_species(geno, qc, scored_ids, *, seed, n_qtn=5,
                   h2=0.5, polygenic_fraction=0.2, qtn_universe="scored"):
    N_VAR,N_SAM=geno.shape
    SEED,N_QTN,H2,H2_POLY_FRAC=seed,n_qtn,h2,polygenic_fraction
    qc=qc.copy()
    qc["snp_id"]=qc.snp_id.astype(str)
    qc["is_scored"]=qc.snp_id.isin(set(map(str,scored_ids)))
    universe=qc[qc.pass_gate if qtn_universe=="all" else (qc.is_scored&qc.pass_gate)].copy()
    rng = np.random.default_rng(SEED)
    pick = [g.index.to_numpy() for _, g in universe.groupby("chrom")]
    flat = np.concatenate(pick)
    n_qtn = min(N_QTN, len(flat))
    qtn = universe.loc[np.sort(rng.choice(flat, size=n_qtn, replace=False))].copy()
    w = np.abs(rng.normal(size=len(qtn)))
    pve = H2 * w ** 2 / (w ** 2).sum()
    beta = np.sqrt(pve) * rng.choice([-1.0, 1.0], size=len(qtn))
    qtn["weight"] = w; qtn["pve"] = pve; qtn["beta"] = beta

    def dosage(vi):
        a = np.asarray(geno[vi])
        return np.where(a == 3, 2.0, np.where(a == 2, 1.0, 0.0))

    y = beta @ dosage(qtn.variant_index.to_numpy(np.int64))
    h2_poly = H2 * H2_POLY_FRAC
    h2_qtn = H2 - h2_poly
    print(f"  h2_QTN={h2_qtn:.3f}  h2_poly={h2_poly:.3f}  h2_total={H2:.3f}", flush=True)
    y = y * np.sqrt(h2_qtn / max(float(np.var(y)), 1e-12))
    gm = np.setdiff1d(rng.choice(N_VAR, size=min(100_000, N_VAR), replace=False),
                      qtn.variant_index.to_numpy(np.int64))
    Z = dosage(np.sort(gm)).astype(np.float32)
    Z -= Z.mean(axis=1, keepdims=True)
    sd = Z.std(axis=1, keepdims=True); sd[sd == 0] = 1.0
    Z /= sd
    K = (Z.T @ Z) / Z.shape[0]
    K /= np.mean(np.diag(K))
    del Z, sd
    ev, EV = np.linalg.eigh(K)
    ev = np.clip(ev, 1e-8, None)
    L = EV * np.sqrt(ev)
    xi = L @ rng.normal(size=N_SAM)
    if h2_poly > 0:
        xi *= np.sqrt(h2_poly / max(float(np.var(xi)), 1e-12))
    else:
        xi = np.zeros(N_SAM, dtype=np.float64)
    y = y + xi + rng.normal(size=N_SAM) * np.sqrt(max(1.0 - H2, 0.0))
    y = (y - y.mean()) / y.std()

    return qtn,y

