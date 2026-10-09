"four-background selection and technical AUC/SMD methods."
from __future__ import annotations
import math,re,zlib
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.linear_model import LogisticRegression

METHODS = ["M0_Random", "M1_Context", "M2_Opportunity", "M3_Full"]

METHOD_SALTS = {m: i + 101 for i, m in enumerate(METHODS)}

TECHNICAL_NUMERIC_FEATURES = ["P_mut_fold", "p_base_eff_fold", "callability"]

def stable_hash(text: str, seed: int) -> int:
    return zlib.crc32(f"{seed}|{text}".encode("utf-8")) & 0xFFFFFFFF

def eligible_mask(df: pd.DataFrame, min_callable_n: int) -> pd.Series:
    return (
        (pd.to_numeric(df["Y_Label"], errors="coerce").fillna(0) == 1) | (pd.to_numeric(df["Y_Label"], errors="coerce").fillna(0) == 0)
    ) & (pd.to_numeric(df["callrate_hit"], errors="coerce").fillna(0) == 1) & (pd.to_numeric(df["callable_n"], errors="coerce").fillna(0) >= min_callable_n)

def add_hash_order(df: pd.DataFrame, key_col: str, seed: int) -> pd.DataFrame:
    out = df.copy()
    out["_hash"] = [stable_hash(str(k), seed) for k in out[key_col].astype(str)]
    return out.sort_values([key_col, "_hash", "snp_id"], kind="mergesort")

def select_negatives(neg: pd.DataFrame, obs: pd.DataFrame, method: str, seed: int) -> pd.DataFrame:
    n = len(obs)
    if n == 0:
        return neg.iloc[0:0].copy()
    if method == "M0_Random":
        tmp = neg.copy()
        tmp["_hash"] = [stable_hash(str(k), seed + METHOD_SALTS[method]) for k in tmp["snp_id"]]
        return tmp.sort_values(["_hash", "snp_id"], kind="mergesort").head(n).drop(columns="_hash")
    group_col = {"M1_Context": "context_ref_alt_group", "M2_Opportunity": "m2_group", "M3_Full": "m3_exact_group"}[method]
    targets = obs.groupby(group_col, sort=False).size()
    tmp = add_hash_order(neg, group_col, seed + METHOD_SALTS[method])
    tmp["_target"] = tmp[group_col].map(targets).fillna(0).astype(int)
    tmp["_rank"] = tmp.groupby(group_col, sort=False).cumcount()
    tmp = tmp[tmp["_rank"] < tmp["_target"]].drop(columns=["_hash", "_target", "_rank"])
    if len(tmp) != n:
        raise RuntimeError(f"{method}: expected {n} negatives but selected {len(tmp)}")
    return tmp

def smd_numeric(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").to_numpy(float)
    y = pd.to_numeric(b, errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    vx = np.var(x, ddof=1) if len(x) > 1 else 0.0
    vy = np.var(y, ddof=1) if len(y) > 1 else 0.0
    den = math.sqrt(max((vx + vy) / 2.0, 0.0))
    return float((np.mean(x) - np.mean(y)) / den) if den > 0 else 0.0

def smd_categorical(a: pd.Series, b: pd.Series) -> float:
    x = a.astype(str)
    y = b.astype(str)
    levels = sorted(set(x.unique()) | set(y.unique()))
    vals = []
    for level in levels:
        p0 = float((x == level).mean())
        p1 = float((y == level).mean())
        p = (p0 + p1) / 2.0
        den = math.sqrt(p * (1.0 - p))
        vals.append(abs(p0 - p1) / den if den > 0 else 0.0)
    return float(max(vals)) if vals else np.nan

def select_common_observed_from_source(pool: pd.DataFrame, observed_source: pd.DataFrame, seed: int, feasibility_group: str = "m3_group") -> tuple[pd.DataFrame, dict]:
    obs = observed_source[observed_source["Y_Label"] == 1].copy()
    neg = pool[pool["Y_Label"] == 0].copy()
    cap = neg.groupby(feasibility_group, sort=False).size()
    obs = add_hash_order(obs, feasibility_group, seed)
    obs["_rank"] = obs.groupby(feasibility_group, sort=False).cumcount()
    obs["_cap"] = obs[feasibility_group].map(cap).fillna(0).astype(int)
    obs = obs[obs["_rank"] < obs["_cap"]].drop(columns=["_hash", "_rank", "_cap"])
    info = {
        "observed_available": int((observed_source["Y_Label"] == 1).sum()),
        "unobserved_available": int((pool["Y_Label"] == 0).sum()),
        "observed_common_m3_feasible": int(len(obs)),
        "observed_attrition_from_m3_feasibility": int((observed_source["Y_Label"] == 1).sum() - len(obs)),
        "m3_groups_with_observed": int(obs[feasibility_group].nunique()),
    }
    return obs, info

def metric_row(method: str, fold: int, split: str, y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "method": method,
        "fold": fold,
        "split": split,
        "n": len(y),
        "positive_rate": float(np.mean(y)),
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
    }

def technical_matrix(df: pd.DataFrame, train_stats: dict | None = None, fit: bool = False) -> tuple[np.ndarray, dict]:
    "Build the matched-variable-only technical calculation matrix."
    numeric_cols = TECHNICAL_NUMERIC_FEATURES
    parts = []
    stats = {} if fit or train_stats is None else train_stats
    for c in numeric_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        if fit or c not in stats:
            med = float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0
            mean = float(np.nanmean(np.where(np.isfinite(x), x, med)))
            std = float(np.nanstd(np.where(np.isfinite(x), x, med)))
            stats[c] = {"median": med, "mean": mean, "std": std if std > 1e-12 else 1.0}
        s = stats[c]
        x = np.where(np.isfinite(x), x, s["median"])
        parts.append(((x - s["mean"]) / s["std"]).reshape(-1, 1))
    for col, n in [("Ctx_code", 64), ("Alt_code", 4), ("Sg_code", 4)]:
        x = pd.to_numeric(df[col], errors="coerce").fillna(-1).to_numpy(int)
        one = np.zeros((len(df), n), dtype=np.float32)
        good = (x >= 0) & (x < n)
        one[np.arange(len(df))[good], x[good]] = 1.0
        parts.append(one)
    ref = df["Ref"].astype(str).to_numpy()
    alt = df["Alt"].astype(str).to_numpy()
    sub = np.array(["AC", "AG", "AT", "CA", "CG", "CT", "GA", "GC", "GT", "TA", "TC", "TG"])
    one = np.zeros((len(df), len(sub)), dtype=np.float32)
    index = {x: i for i, x in enumerate(sub)}
    for i, key in enumerate(ref + ">" + alt):
        if key.replace(">", "") in index:
            one[i, index[key.replace(">", "")]] = 1.0
    parts.append(one)
    return np.hstack(parts).astype(np.float32), stats

def parse_chr_num(value) -> int:
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else -1

def fast_hash_order(df: pd.DataFrame, key_col: str, seed: int) -> pd.DataFrame:
    out = df.copy()
    h = pd.util.hash_pandas_object(out[key_col].astype(str), index=False).to_numpy(dtype=np.uint64)
    out["_hash"] = h ^ np.uint64(seed * 2654435761)
    return out.sort_values([key_col, "_hash", "snp_id"], kind="mergesort")

def fast_select_common_observed(pool: pd.DataFrame, observed_source: pd.DataFrame, seed: int, feasibility_group: str = "m3_group") -> tuple[pd.DataFrame, dict]:
    obs = observed_source[observed_source["Y_Label"] == 1].copy()
    neg = pool[pool["Y_Label"] == 0].copy()
    cap = neg.groupby(feasibility_group, sort=False).size()
    obs = fast_hash_order(obs, feasibility_group, seed)
    obs["_rank"] = obs.groupby(feasibility_group, sort=False).cumcount()
    obs["_cap"] = obs[feasibility_group].map(cap).fillna(0).astype(int)
    obs = obs[obs["_rank"] < obs["_cap"]].drop(columns=["_hash", "_rank", "_cap"])
    info = {
        "observed_available": int((observed_source["Y_Label"] == 1).sum()),
        "unobserved_available": int((pool["Y_Label"] == 0).sum()),
        "observed_common_m3_feasible": int(len(obs)),
        "observed_attrition_from_m3_feasibility": int((observed_source["Y_Label"] == 1).sum() - len(obs)),
        "m3_groups_with_observed": int(obs[feasibility_group].nunique()),
    }
    return obs, info

def fast_select_all_negatives(neg: pd.DataFrame, obs: pd.DataFrame, seed: int, m3_group: str = "m3_group") -> dict[str, pd.DataFrame]:
    n = len(obs)
    if n == 0:
        empty = neg.iloc[0:0].copy()
        return {m: empty.copy() for m in METHODS}

    rng0 = np.random.default_rng(seed + 101)
    take0 = rng0.choice(len(neg), size=min(n, len(neg)), replace=False)
    out = {"M0_Random": neg.iloc[take0].copy()}

    def quota_sample(group_col: str, method_seed: int) -> pd.DataFrame:
        targets = obs[group_col].value_counts(sort=False)
        codes, levels = pd.factorize(neg[group_col], sort=False)
        target_by_code = targets.reindex(levels, fill_value=0).to_numpy(dtype=np.int64)
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
        ends = np.r_[starts[1:], len(order)]
        picked = []
        for code, start, end in zip(sorted_codes[starts], starts, ends):
            k = int(target_by_code[int(code)])
            available = int(end - start)
            if k <= 0:
                continue
            if k > available:
                raise RuntimeError(f"{group_col}: insufficient negatives for group code {code}: need={k} available={available}")
            rng = np.random.default_rng(method_seed + int(code) * 104729)
            local = rng.choice(available, size=k, replace=False)
            picked.append(order[start + local])
        if not picked:
            return neg.iloc[0:0].copy()
        idx = np.concatenate(picked)
        return neg.iloc[idx].copy()

    out["M1_Context"] = quota_sample("context_ref_alt_group", seed + 102)
    out["M2_Opportunity"] = quota_sample("m2_group", seed + 103)
    out["M3_Full"] = quota_sample(m3_group, seed + 104)
    for method, frame in out.items():
        if len(frame) != n:
            raise RuntimeError(f"{method}: expected {n} negatives but selected {len(frame)}")
    return out

def normalize_technical(df: pd.DataFrame, max_callable_n: float, p_edges=None, cn_edges=None) -> pd.DataFrame:
    out = df.copy()
    if "snp_id" not in out and "SNP_ID" in out:
        out["snp_id"] = out["SNP_ID"]
    out["snp_id"] = out["snp_id"].astype(str)
    for c in ["Chr", "Ref", "Alt"]:
        if c in out:
            out[c] = out[c].astype(str)
    if "chr_num" not in out:
        out["chr_num"] = out["Chr"].map(parse_chr_num)
    out["chr_num"] = pd.to_numeric(out["chr_num"], errors="coerce").fillna(-1).astype(np.int16)
    for c in ["Ctx_code", "Alt_code", "Sg_code", "Y_Label", "callrate_hit"]:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(np.int16)
    if "callable_n" not in out:
        if "callable_rate" in out:
            out["callable_n"] = pd.to_numeric(out["callable_rate"], errors="coerce").fillna(0) * float(max_callable_n)
        elif "AN_obs" in out:
            out["callable_n"] = pd.to_numeric(out["AN_obs"], errors="coerce").fillna(0) / 2.0
        else:
            out["callable_n"] = 0.0
    out["callable_n"] = pd.to_numeric(out["callable_n"], errors="coerce").fillna(0).astype("float32")
    if "callrate_hit" not in out:
        out["callrate_hit"] = (out["callable_n"] > 0).astype(np.int8)
    if "AN_obs" not in out:
        out["AN_obs"] = out["callable_n"] * 2.0
    if "call_rate" not in out:
        out["call_rate"] = out["callable_n"] / float(max_callable_n)
    if "callable_rate" not in out:
        out["callable_rate"] = out["callable_n"] / float(max_callable_n)
    out["callability"] = np.clip(out["callable_n"].to_numpy(float) / float(max_callable_n), 0.0, 1.0)
    if "Context" not in out:
        out["Context"] = out["Ctx_code"].astype(str)
    else:
        out["Context"] = out["Ctx_code"].astype(str)
    if "Ref" not in out:
        out["Ref"] = "N"
    if "Alt" not in out:
        out["Alt"] = "N"
    out["substitution_class"] = out["Ref"].astype(str) + ">" + out["Alt"].astype(str)
    if p_edges is not None:
        out["p_base_eff_bin"] = np.searchsorted(np.asarray(p_edges, float), out["p_base_eff_fold"].to_numpy(float), side="right") - 1
    if cn_edges is not None:
        out["callable_n_bin"] = np.searchsorted(np.asarray(cn_edges, float), out["callable_n"].to_numpy(float), side="right") - 1

    

    sub_code = (out["Ref"].astype(str) + ">" + out["Alt"].astype(str)).map({
        "A>C": 0, "A>G": 1, "A>T": 2, "C>A": 3, "C>G": 4, "C>T": 5,
        "G>A": 6, "G>C": 7, "G>T": 8, "T>A": 9, "T>C": 10, "T>G": 11,
    }).fillna(15).astype(np.int64)
    base = ((out["Sg_code"].astype(np.int64) * 64 + out["Ctx_code"].astype(np.int64)) * 16 + sub_code)
    pbin_src = out["p_base_eff_bin"] if "p_base_eff_bin" in out else pd.Series(0, index=out.index)
    cbin_src = out["callable_n_bin"] if "callable_n_bin" in out else pd.Series(0, index=out.index)
    pbin = pd.to_numeric(pbin_src, errors="coerce").fillna(-1).astype(np.int64) + 1
    cbin = pd.to_numeric(cbin_src, errors="coerce").fillna(-1).astype(np.int64) + 1
    out["context_ref_alt_group"] = base
    out["m2_group"] = base * 16 + pbin
    out["m3_group"] = out["m2_group"] * 32 + cbin
    out["m3_exact_group"] = out["m3_group"] * 512 + out["callable_n"].round().astype(np.int64)
    return out

def balance_records(selected: pd.DataFrame, method: str, fold: int, split: str) -> list[dict]:
    rows = []
    a = selected[selected["label"] == 0]
    b = selected[selected["label"] == 1]
    for name, col, kind in [
        ("P_mut", "P_mut_fold", "numeric"),
        ("p_base_eff", "p_base_eff_fold", "numeric"),
        ("callable_n", "callable_n", "numeric"),
        ("callability", "callability", "numeric"),
        ("Context", "Context", "categorical"),
        ("substitution_class", "substitution_class", "categorical"),
    ]:
        value = smd_numeric(a[col], b[col]) if kind == "numeric" else smd_categorical(a[col], b[col])
        rows.append({"method": method, "fold": fold, "split": split, "covariate": name, "smd": value, "abs_smd": abs(value) if np.isfinite(value) else np.nan})
    return rows

def select_matched_backgrounds(pool,model,*,seed=42,fold=1,split="train"):
    selected_by_method={method:{} for method in METHODS}
    eligible = pool[eligible_mask(pool, 1)].copy()
    observed_source = model[eligible_mask(model, 1)].copy()
    obs, info = fast_select_common_observed(pool, observed_source, seed + fold * 1000 + len(split), feasibility_group="m3_exact_group")
    neg_pool = eligible[eligible["Y_Label"] == 0].copy()
    selected_negatives = fast_select_all_negatives(neg_pool, obs, seed + fold * 1000 + len(split), m3_group="m3_exact_group")
    for method in METHODS:
        neg = selected_negatives[method]
        pos = obs.copy()
        pos["label"] = 0
        pos["selection_role"] = "observed"
        neg["label"] = 1
        neg["selection_role"] = "unobserved_proxy_depleted"
        selected = pd.concat([pos, neg], ignore_index=True, sort=False)
        selected["method"] = method
        selected["fold"] = fold
        selected["split"] = split
        selected_by_method[method][split] = selected
    return {method:frames[split] for method,frames in selected_by_method.items()},info


def technical_diagnostic(train,test,*,method,fold):
    frames=[train,None,test]
    metric_rows=[]
    Xtrain, stats = technical_matrix(frames[0], fit=True)
    Xtest, _ = technical_matrix(frames[2], train_stats=stats)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(Xtrain, frames[0].label.to_numpy(np.int8))
    pred = model.predict_proba(Xtest)[:, 1]
    metric_rows.append(metric_row(method, fold, "test", frames[2].label.to_numpy(np.int8), pred))
    return metric_rows[0]
