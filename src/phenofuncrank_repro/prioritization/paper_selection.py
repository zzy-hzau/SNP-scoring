"Calculate T, F, B and sequential M rankings."
from __future__ import annotations
import numpy as np
import pandas as pd

EPS=1e-12
SUPPORT_R2=0.60
DEFAULT_R0=0.80
LUT=np.array([np.nan,0.0,1.0,2.0],dtype=np.float32)

def midrank_percentile(values: np.ndarray) -> np.ndarray:
    """Midrank empirical percentile in (0, 1), larger values rank higher."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Direct DPC values must be a non-empty finite vector")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    left = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    group = np.cumsum(left) - 1
    counts = np.bincount(group)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    avg_rank = starts + (counts + 1.0) / 2.0
    out = np.empty(len(values), dtype=float)
    out[order] = (avg_rank[group] - 0.5) / len(values)
    return np.clip(out, EPS, 1.0 - EPS)


def _normalize_block(codes: np.ndarray) -> np.ndarray:
    x = np.take(LUT, codes.astype(np.int64))
    valid = np.isfinite(x)
    means = np.nanmean(x, axis=1)
    z = np.where(valid, x - means[:, None], 0.0).astype(np.float32, copy=False)
    norms = np.sqrt(np.einsum("ij,ij->i", z, z))
    z /= np.where(norms[:, None] > 0, norms[:, None], 1.0)
    return z


def clump_sources(base: pd.DataFrame, pairs: pd.DataFrame, r0: float) -> tuple[np.ndarray, dict[int, list[int]], np.ndarray]:
    core_mask = base.trait_evidence_type.astype(str).str.lower().eq("direct")
    core_pos = np.flatnonzero(core_mask.to_numpy())
    core_set = set(core_pos.tolist())
    n = len(base)
    core_T = pd.to_numeric(base.loc[core_pos, "T"], errors="coerce").to_numpy(float)
    edge: dict[int, list[int]] = {int(c): [] for c in core_pos}
    sub = pairs[(pairs.core_idx.isin(core_set)) & (pairs.cand_idx.isin(core_set)) & (pairs.r2 >= r0)]
    for c, k in zip(sub.core_idx.to_numpy(np.int64), sub.cand_idx.to_numpy(np.int64)):
        edge[int(c)].append(int(k))
        edge[int(k)].append(int(c))
    order = core_pos[np.argsort(-core_T, kind="mergesort")]
    source_of_core = np.full(n, -1, dtype=np.int64)
    sources: dict[int, list[int]] = {}
    n_source = 0
    for c in order.tolist():
        if source_of_core[c] != -1:
            continue
        stack = [c]
        source_of_core[c] = n_source
        members = []
        while stack:
            u = stack.pop()
            members.append(u)
            for v in edge.get(u, ()):
                if source_of_core[v] == -1:
                    source_of_core[v] = n_source
                    stack.append(v)
        sources[n_source] = members
        n_source += 1
    anchors = np.full(n_source, -1, dtype=np.int64)
    pos_of_core = {int(c): i for i, c in enumerate(core_pos)}
    for s, members in sources.items():
        anchors[s] = members[int(np.argmax([core_T[pos_of_core[m]] for m in members]))]
    return source_of_core, sources, anchors


def build_support(base: pd.DataFrame, pairs: pd.DataFrame, source_of_core: np.ndarray, anchors: np.ndarray) -> dict:
    n = len(base)
    core_mask = base.trait_evidence_type.astype(str).str.lower().eq("direct")
    core_pos = np.flatnonzero(core_mask.to_numpy())
    core_T = np.zeros(n, dtype=np.float64)
    core_T[core_pos] = pd.to_numeric(base.loc[core_pos, "T"], errors="coerce").to_numpy(float)

    pair_core = pairs.core_idx.to_numpy(np.int64)
    pair_cand = pairs.cand_idx.to_numpy(np.int64)
    pair_r2 = pairs.r2.to_numpy(np.float64)
    pair_source = source_of_core[pair_core]
    pair_a = core_T[pair_core] * pair_r2

    cand = np.concatenate([pair_cand, core_pos])
    src = np.concatenate([pair_source, source_of_core[core_pos]])
    a = np.concatenate([pair_a, core_T[core_pos]])
    valid = src >= 0
    cand, src, a = cand[valid], src[valid], a[valid]


    order = np.lexsort((a, cand, src))
    cs = cand[order]
    ss = src[order]
    aa = a[order]
    is_start = np.ones(len(cs), dtype=bool)
    if len(cs) > 1:
        is_start[1:] = (ss[1:] != ss[:-1]) | (cs[1:] != cs[:-1])
    starts = np.flatnonzero(is_start)
    ends = np.empty(len(starts), dtype=np.int64)
    ends[:-1] = starts[1:]
    if len(ends):
        ends[-1] = len(cs)
    max_a = aa[ends - 1]
    max_cand = cs[ends - 1]
    max_src = ss[ends - 1]

    cand_sums = np.bincount(max_cand, weights=max_a, minlength=n).astype(np.float64)
    T_new = np.full(n, EPS, dtype=np.float64)
    np.maximum.at(T_new, max_cand, max_a)
    F = pd.to_numeric(base["F"], errors="coerce").to_numpy(float)
    B_new = np.sqrt(np.clip(T_new, EPS, 1.0 - EPS) * np.clip(F, EPS, 1.0 - EPS))

    o2 = np.argsort(max_cand, kind="stable")
    cand_ids = max_cand[o2]
    cand_src = max_src[o2]
    cand_p = (max_a[o2] / np.maximum(cand_sums[cand_ids], 1e-300)).astype(np.float32)
    cand_ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(cand_ids, minlength=n), out=cand_ptr[1:])

    n_source = int(source_of_core.max()) + 1
    o3 = np.argsort(cand_src, kind="stable")
    src_sorted = cand_src[o3]
    cand_sorted = cand_ids[o3]
    p_sorted = cand_p[o3]
    src_ptr = np.zeros(n_source + 1, dtype=np.int64)
    np.cumsum(np.bincount(src_sorted, minlength=n_source), out=src_ptr[1:])
    sources = {s: cand_sorted[src_ptr[s]:src_ptr[s + 1]] for s in range(n_source)}
    source_p = {s: p_sorted[src_ptr[s]:src_ptr[s + 1]] for s in range(n_source)}

    supported = T_new > EPS
    typ = base.trait_evidence_type.astype(str).str.lower().replace({"core": "direct", "proxy": "ld-supported", "unsupported-proxy": "unsupported"})
    out_type = np.where(core_mask, "direct", np.where(supported, "ld-supported", "unsupported"))

    best_source = np.full(n, -1, dtype=np.int64)
    best_r2 = np.zeros(n, dtype=np.float64)
    n_support = np.zeros(n, dtype=np.int64)
    top_share = np.zeros(n, dtype=np.float32)
    o4 = np.lexsort((-max_a, max_cand))
    bc = max_cand[o4]
    bs = max_src[o4]
    first = np.ones(len(bc), dtype=bool)
    if len(bc) > 1:
        first[1:] = bc[1:] != bc[:-1]
    best_source[bc[first]] = bs[first]
    n_support = np.bincount(max_cand, minlength=n).astype(np.int64)
    top_share[bc[first]] = (max_a[o4[first]] / np.maximum(cand_sums[bc[first]], 1e-300)).astype(np.float32)
    np.maximum.at(best_r2, pair_cand, pair_r2)

    snp_id_str = base.snp_id.astype(str).to_numpy()
    return {
        "n": n,
        "snp_id": snp_id_str,
        "chr": base["chr"].astype(str).to_numpy(),
        "pos": base["pos"].to_numpy(np.int64),
        "ref": base["ref"].astype(str).to_numpy(),
        "alt": base["alt"].astype(str).to_numpy(),
        "ldcc_id": base.ldcc_id.astype(str).to_numpy(),
        "geno_idx": base["geno_idx"].to_numpy(np.int64),
        "trait_evidence_type": out_type,
        "T": T_new,
        "F": F,
        "B": B_new,
        "abs_dpc_beta": pd.to_numeric(base["abs_dpc_beta"], errors="coerce").to_numpy(float),
        "core_pos": core_pos,
        "source_of_core": source_of_core,
        "sources": sources,
        "source_p": source_p,
        "cand_ptr": cand_ptr,
        "cand_src": cand_src,
        "cand_p": cand_p,
        "best_source": best_source,
        "source_anchor": np.asarray([str(snp_id_str[a]) if a >= 0 else "" for a in anchors]),
        "best_core_r2": best_r2,
        "n_support_sources": n_support,
        "top_provenance_share": top_share,
        "pairs_core": pairs.core_idx.to_numpy(np.int64),
        "pairs_cand": pairs.cand_idx.to_numpy(np.int64),
        "pairs_r2": pairs.r2.to_numpy(np.float32),
    }


def assign_global_ranks(ctx: dict) -> None:
    n = ctx["n"]
    order = np.lexsort((
        ctx["snp_id"], ctx["alt"], ctx["ref"], ctx["pos"],
        np.char.replace(np.char.replace(ctx["chr"].astype(str), "chr", ""), "Chr", ""),
        -ctx["F"], -np.nan_to_num(ctx["abs_dpc_beta"], nan=0.0), -ctx["T"],
        -np.minimum(ctx["T"], ctx["F"]), -ctx["B"],
    ))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(1, n + 1)
    ctx["global_rank"] = rank


def build_neighbor_csr(ctx: dict) -> None:
    a = np.concatenate([ctx["pairs_core"], ctx["pairs_cand"]])
    b = np.concatenate([ctx["pairs_cand"], ctx["pairs_core"]])
    r = np.concatenate([ctx["pairs_r2"], ctx["pairs_r2"]])
    order = np.argsort(a, kind="stable")
    a, b, r = a[order], b[order], r[order]
    ctx["neighbor_b"] = b.astype(np.int64)
    ctx["neighbor_r2"] = r.astype(np.float32)
    ctx["neighbor_start"] = np.searchsorted(a, np.arange(ctx["n"] + 1))


def select_greedy_fast(
    ctx: dict,
    z: np.ndarray,
    budget: int,
    *,
    return_state: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, np.ndarray]]:
    "Greedy Framework-M selection."
    n = ctx["n"]
    B = ctx["B"].astype(np.float64, copy=False)
    max_r2 = np.zeros(n, dtype=np.float64)
    load = np.zeros(n, dtype=np.float64)
    marginal = B.copy()
    selected_mask = np.zeros(n, dtype=bool)
    selection_m = np.full(n, np.nan, dtype=np.float64)
    selection_step = np.full(n, -1, dtype=np.int64)
    cur = np.zeros(n, dtype=np.float64)
    stamp = np.zeros(n, dtype=np.int32)

    ld_groups: dict[str, np.ndarray] = {}
    for i, g in enumerate(ctx["ldcc_id"]):
        ld_groups.setdefault(str(g), []).append(i)
    ld_groups = {g: np.asarray(v, dtype=np.int64) for g, v in ld_groups.items()}
    ld_z_cache: dict[str, np.ndarray] = {}

    rows = []
    for step in range(1, budget + 1):
        marginal[selected_mask] = -np.inf
        i = int(np.argmax(marginal))
        if not np.isfinite(marginal[i]) or len(rows) >= budget:
            break
        selected_mask[i] = True
        ldcc = str(ctx["ldcc_id"][i])
        row = {
            "priority_rank": step,
            "global_rank": int(ctx["global_rank"][i]),
            "snp_id": str(ctx["snp_id"][i]),
            "chrom": str(ctx["chr"][i]),
            "pos": int(ctx["pos"][i]),
            "ref": str(ctx["ref"][i]),
            "alt": str(ctx["alt"][i]),
            "ldcc_id": ldcc,
            "trait_evidence_type": str(ctx["trait_evidence_type"][i]),
            "source_id": int(ctx["best_source"][i]) if ctx["best_source"][i] >= 0 else -1,
            "source_anchor": str(ctx["source_anchor"][ctx["best_source"][i]]) if ctx["best_source"][i] >= 0 else "",
            "B": float(B[i]),
            "T": float(ctx["T"][i]),
            "F": float(ctx["F"][i]),
            "max_r2_to_selected": float(max_r2[i]),
            "genotype_novelty": float(np.clip(1.0 - max_r2[i], 0.0, 1.0)),
            "provenance_load_before": float(load[i]),
            "provenance_novelty": float(1.0 / np.sqrt(1.0 + load[i])),
            "marginal_priority": float(marginal[i]),
            "n_support_sources": int(ctx["n_support_sources"][i]),
            "top_provenance_share": float(ctx["top_provenance_share"][i]),
            "best_core_r2": float(ctx["best_core_r2"][i]),
        }
        rows.append(row)
        selection_m[i] = marginal[i]
        selection_step[i] = step


        ii = ld_groups[ldcc]
        if ldcc not in ld_z_cache:
            ld_z_cache[ldcc] = np.ascontiguousarray(z[ii])
        r2 = np.square(ld_z_cache[ldcc] @ z[i])
        take = ~selected_mask[ii] & np.isfinite(r2)
        max_r2[ii[take]] = np.maximum(max_r2[ii[take]], r2[take])

        s0, s1 = int(ctx["neighbor_start"][i]), int(ctx["neighbor_start"][i + 1])
        if s1 > s0:
            nb = ctx["neighbor_b"][s0:s1]
            nr = ctx["neighbor_r2"][s0:s1]
            keep = ~selected_mask[nb]
            nb, nr = nb[keep], nr[keep]
            if len(nb):
                max_r2[nb] = np.maximum(max_r2[nb], nr.astype(np.float64))


        touched: list[np.ndarray] = []
        p0, p1 = int(ctx["cand_ptr"][i]), int(ctx["cand_ptr"][i + 1])
        for s, pj in zip(ctx["cand_src"][p0:p1], ctx["cand_p"][p0:p1]):
            idx_s = ctx["sources"][int(s)]
            p_s = ctx["source_p"][int(s)]
            if len(idx_s) == 0:
                continue
            newmask = stamp[idx_s] != step
            if newmask.any():
                new_idx = idx_s[newmask]
                stamp[new_idx] = step
                touched.append(new_idx)
            cur[idx_s] += np.minimum(p_s, np.float32(pj))
        affected = ii
        if touched:
            touched_all = np.unique(np.concatenate(touched))
            sel_t = touched_all[~selected_mask[touched_all]]
            if len(sel_t):
                m = cur[sel_t]
                jval = m / (2.0 - m)
                load[sel_t] += jval
            cur[touched_all] = 0.0
            affected = np.unique(np.concatenate([affected, touched_all]))
        if len(affected):
            marginal[affected] = B[affected] * np.clip(1.0 - max_r2[affected], 0.0, 1.0) / np.sqrt(1.0 + load[affected])
    selected = pd.DataFrame(rows)
    if return_state:
        return selected, {
            "marginal_final": marginal.copy(),
            "max_r2_final": max_r2.copy(),
            "load_final": load.copy(),
            "selected_mask": selected_mask.copy(),
            "selection_m": selection_m,
            "selection_step": selection_step,
        }
    return selected


def selected_pairwise_stats(ctx: dict, selected: pd.DataFrame, z: np.ndarray) -> dict:
    id_to_ctx = {str(ctx["snp_id"][i]): int(i) for i in range(ctx["n"])}
    sel_idx = selected.snp_id.astype(str).map(id_to_ctx).to_numpy(np.int64)
    sel_set = set(int(x) for x in sel_idx)
    r2_vals: list[float] = []
    j_vals: list[float] = []
    by_ldcc: dict[str, list[int]] = {}
    for idx in sel_idx:
        by_ldcc.setdefault(str(ctx["ldcc_id"][idx]), []).append(int(idx))
    for g, members in by_ldcc.items():
        arr = np.asarray(members, dtype=np.int64)
        if len(arr) >= 2:
            r2 = np.square(z[arr] @ z[arr].T)
            iu = np.triu_indices(len(arr), 1)
            r2_vals.extend(r2[iu].tolist())
    for idx in sel_idx:
        s0, s1 = int(ctx["neighbor_start"][idx]), int(ctx["neighbor_start"][idx + 1])
        for k in range(s0, s1):
            nb = int(ctx["neighbor_b"][k])
            if nb in sel_set and nb > idx and str(ctx["ldcc_id"][nb]) != str(ctx["ldcc_id"][idx]):
                r2_vals.append(float(ctx["neighbor_r2"][k]))
    pos_of_sel = {int(idx): k for k, idx in enumerate(sel_idx)}
    pair_M: dict[tuple[int, int], float] = {}
    n_source = int(ctx["best_source"].max()) + 1
    for s in range(n_source):
        idx_s = ctx["sources"].get(s, np.zeros(0, dtype=np.int64))
        p_s = ctx["source_p"].get(s, np.zeros(0, dtype=np.float32))
        local = [(pos_of_sel[int(v)], float(p_s[k])) for k, v in enumerate(idx_s) if int(v) in pos_of_sel]
        for a_i in range(len(local)):
            for a_j in range(a_i + 1, len(local)):
                key = tuple(sorted((local[a_i][0], local[a_j][0])))
                pair_M[key] = pair_M.get(key, 0.0) + min(local[a_i][1], local[a_j][1])
    for (_a, _b), m in pair_M.items():
        j_vals.append(m / (2.0 - m))
    r2_arr = np.asarray(r2_vals, dtype=float)
    j_arr = np.asarray(j_vals, dtype=float)

    def q(v, pp):
        return float(np.percentile(v, pp)) if len(v) else np.nan
    return {
        "selected_pairwise_r2_median": q(r2_arr, 50),
        "selected_pairwise_r2_p95": q(r2_arr, 95),
        "selected_pairwise_r2_p99": q(r2_arr, 99),
        "selected_pairwise_J_median": q(j_arr, 50),
        "selected_pairwise_J_p95": q(j_arr, 95),
        "n_selected_pairs": int(len(r2_arr)),
        "n_selected_J_pairs": int(len(j_arr)),
    }


class HybridZ:
    """Lazy normalized genotype rows, reusing old z rows where available."""

    def __init__(self, old_z: np.ndarray | None, old_row: np.ndarray, geno_path: str, shape: tuple[int, int], geno_idx: np.ndarray):
        self.old_z = old_z
        self.old_row = np.asarray(old_row, dtype=np.int64)
        self.geno = np.memmap(geno_path, dtype="uint8", mode="r", shape=shape)
        self.geno_idx = np.asarray(geno_idx, dtype=np.int64)
        self.n_samples = int(shape[1])

    @staticmethod
    def _normalize(codes: np.ndarray) -> np.ndarray:
        lut = np.array([np.nan, 0.0, 1.0, 2.0], dtype=np.float32)
        x = np.take(lut, codes.astype(np.int64))
        valid = np.isfinite(x)
        means = np.nanmean(x, axis=1)
        z = np.where(valid, x - means[:, None], 0.0).astype(np.float32, copy=False)
        norms = np.sqrt(np.einsum("ij,ij->i", z, z))
        z /= np.where(norms[:, None] > 0, norms[:, None], 1.0)
        return z

    def __getitem__(self, key):
        scalar = np.isscalar(key)
        pos = np.asarray([key], dtype=np.int64) if scalar else np.asarray(key, dtype=np.int64).reshape(-1)
        out = np.empty((len(pos), self.n_samples), dtype=np.float32)
        old = self.old_row[pos] if len(pos) else np.zeros(0, dtype=np.int64)
        use_old = (old >= 0) & (self.old_z is not None)
        if use_old.any():
            out[use_old] = np.asarray(self.old_z[old[use_old]], dtype=np.float32)
        if (~use_old).any():
            raw_idx = self.geno_idx[pos[~use_old]]
            codes = np.asarray(self.geno[raw_idx], dtype=np.uint8)
            out[~use_old] = self._normalize(codes)
        return out[0] if scalar else out


def component_labels(pool, pairs):
    parent=np.arange(len(pool),dtype=np.int64)
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=int(parent[x])
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb:
            parent[rb]=ra
    for a,b in zip(pairs.core_idx.to_numpy(np.int64),
                   pairs.cand_idx.to_numpy(np.int64),strict=False):
        union(int(a),int(b))
    roots={}
    labels=np.full(len(pool),"",dtype=object)
    for i in range(len(pool)):
        roots.setdefault(find(i),len(roots)+1)
        labels[i]=f"LDCC_REVISED_{roots[find(i)]:04d}"
    return labels


def build_context_from_calibrated_scores(pool, core, scores, pairs,
                                         genotype_indices, ldcc=None):
    pool=pool.copy().reset_index(drop=True)
    pool["panel_id"]=pool["panel_id"].astype(str)
    scores=scores.set_index("panel_id").loc[pool.panel_id].reset_index()
    core_values=midrank_percentile(
        pd.to_numeric(core.abs_effect_real,errors="raise").to_numpy(float))
    core_t=dict(zip(core.pool_id.astype(str),core_values,strict=False))
    core_beta=dict(zip(core.pool_id.astype(str),
                       pd.to_numeric(core.abs_effect_real,errors="raise"),
                       strict=False))
    pool["ldcc_id"]=component_labels(pool,pairs) if ldcc is None else ldcc
    pool["trait_evidence_type"]=np.where(pool.role.eq("core"),"direct","ld-supported")
    pool["snp_id"]=pool["panel_id"].astype(str)
    pool["chr"]=pool["chrom"].astype(str)
    pool["T"]=[core_t.get(str(x),EPS) if role=="core" else EPS
               for x,role in zip(pool.panel_id,pool.role,strict=False)]
    pool["F"]=pd.to_numeric(scores["F"],errors="raise").to_numpy(float)
    pool["abs_dpc_beta"]=[float(core_beta[str(x)]) if role=="core" else np.nan
                          for x,role in zip(pool.panel_id,pool.role,strict=False)]
    pool["geno_idx"]=np.asarray(genotype_indices,dtype=np.int64)
    source_of_core,_,anchors=clump_sources(pool,pairs,DEFAULT_R0)
    context=build_support(pool,pairs,source_of_core,anchors)
    assign_global_ranks(context)
    build_neighbor_csr(context)
    return context
