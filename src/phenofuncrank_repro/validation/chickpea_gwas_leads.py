"""Actual stableP/BH and sourceFloat32 greedy GWAS lead selection."""

from __future__ import annotations

import argparse,json

from pathlib import Path

import numpy as np

import pandas as pd

R2_MIN = 0.60

WIN = 500_000

LUT = np.asarray([np.nan,0.,1.,2.],dtype=np.float32)

class Panel:
    def __init__(self, codes, variants, z_chunk=5000):
        self.geno=np.asarray(codes,dtype=np.uint8);self.n_ind=self.geno.shape[1]
        self.sm_c=variants['chrom'].astype(str).to_numpy();self.sm_p=variants['pos'].to_numpy(np.int64);self.sm_vi=np.arange(len(variants))
        self.n_total=len(variants);self.z_chunk=z_chunk;self._cache={};self._poscache={};self._sparse={}
    def chrom(self, chrom: str):
            if chrom in self._cache:
                return self._cache[chrom]
            rows = np.flatnonzero(self.sm_c == chrom)
            pos = self.sm_p[rows]
            vi = self.sm_vi[rows]
            o = np.argsort(pos, kind="mergesort")
            pos, vi = pos[o], vi[o]
            Z = np.empty((len(vi), self.n_ind), dtype=np.float32)
            for st in range(0, len(vi), self.z_chunk):
                en = min(st + self.z_chunk, len(vi))
                Z[st:en] = normalize(np.asarray(self.geno[vi[st:en]]))
            self._cache[chrom] = (pos, vi, Z)
            return self._cache[chrom]
    def chrom_Z_sparse(self, chrom: str, positions: np.ndarray):
            key = (chrom, len(positions), int(positions.min()) if len(positions) else -1,
                   int(positions.max()) if len(positions) else -1)
            if key in self._sparse:
                return self._sparse[key]
            pos, vi = self.chrom_pos(chrom)
            if len(positions) == 0:
                out = (pos, vi, np.zeros(0, dtype=np.int64), np.zeros((0, self.n_ind), np.float32))
                self._sparse[key] = out
                return out
            mark = np.zeros(len(pos) + 1, dtype=np.int32)
            sl = np.searchsorted(pos, positions - WIN, side="left")
            sr = np.searchsorted(pos, positions + WIN, side="right")
            np.add.at(mark, sl, 1)
            np.add.at(mark, sr, -1)
            keep = np.flatnonzero(np.cumsum(mark[:-1]) > 0)
            Z = self.z_rows(vi[keep])
            out = (pos, vi, keep, Z)
            self._sparse[key] = out
            return out
    def chrom_pos(self, chrom: str):
            """Sorted (pos, variant_index) for a chromosome — no genotype work."""
            if chrom not in self._poscache:
                rows = np.flatnonzero(self.sm_c == chrom)
                pos = self.sm_p[rows]; vi = self.sm_vi[rows]
                o = np.argsort(pos, kind="mergesort")
                self._poscache[chrom] = (pos[o], vi[o])
            return self._poscache[chrom]
    def z_rows(self, vi: np.ndarray) -> np.ndarray:
            """Normalised genotypes for just these variant indices."""
            Z = np.empty((len(vi), self.n_ind), dtype=np.float32)
            for st in range(0, len(vi), self.z_chunk):
                en = min(st + self.z_chunk, len(vi))
                Z[st:en] = normalize(np.asarray(self.geno[vi[st:en]]))
            return Z
    def release(self, chrom: str):
            self._cache.pop(chrom, None)


def normalize(codes: np.ndarray) -> np.ndarray:
    v = np.take(LUT, codes.astype(np.int64))
    centered = np.where(np.isfinite(v), v - np.nanmean(v, axis=1)[:, None], 0.0) \
        .astype(np.float32)
    n = np.sqrt(np.einsum("ij,ij->i", centered, centered))
    return centered / np.where(n[:, None] > 0, n[:, None], 1.0)

def greedy_clump(pc: Panel, cand: pd.DataFrame, block: int = 2000):
    """PLINK-style greedy clumping in the row order of `cand`.

    Builds the candidate x candidate LD adjacency with batched GEMM (only pairs
    within the 500-kb window are kept), then walks the list in order.
    """
    cand = cand.reset_index(drop=True)
    n = len(cand)
    adj: list[np.ndarray | None] = [None] * n
    for chrom, cg in cand.groupby("chrom", sort=False):
        pos, vi = pc.chrom_pos(chrom)
        rows = cg.index.to_numpy()
        c_pos = cg["pos"].to_numpy(np.int64)
        ks = np.searchsorted(pos, c_pos)
        clip = np.clip(ks, 0, len(pos) - 1)
        ok = (ks < len(pos)) & (pos[clip] == c_pos)
        rows, c_pos, ks = rows[ok], c_pos[ok], ks[ok]
        order = np.argsort(c_pos, kind="mergesort")
        rows, cpos_s = rows[order], c_pos[order]
        Zc = pc.z_rows(vi[ks[order]])          
        m = len(rows)
        lo_idx = np.searchsorted(cpos_s, cpos_s - WIN, side="left")
        hi_idx = np.searchsorted(cpos_s, cpos_s + WIN, side="right")
        for st in range(0, m, block):
            en = min(st + block, m)
            R = Zc[st:en] @ Zc.T                    
            M = np.square(R) > R2_MIN
            for b in range(en - st):
                i = st + b
                row = M[b]
                row[: lo_idx[i]] = False
                row[hi_idx[i]:] = False
                row[i] = False
                if row.any():
                    adj[rows[i]] = rows[np.flatnonzero(row)]
                else:
                    adj[rows[i]] = np.zeros(0, dtype=np.int64)
            del R, M
    kept, blocked = [], np.zeros(n, dtype=bool)
    for i in range(n):
        if blocked[i]:
            continue
        kept.append(i)
        a = adj[i]
        if a is not None and len(a):
            blocked[a] = True
        blocked[i] = False
    return cand.iloc[kept].reset_index(drop=True)

def expand_perlead(pc: Panel, seeds: pd.DataFrame, hint: np.ndarray | None = None):
    """One block per lead: each lead's own LD neighbourhood, no chaining.

    Returns a DataFrame with one row per seed (region_id == seed row order) whose
    span is (min seed/proxy position, max seed/proxy position).  Leads whose
    neighbourhoods overlap therefore produce overlapping blocks, which is the
    point: B leads give exactly B blocks, so the budget semantics match the
    Framework side without any connected-component chaining.
    """
    seeds = seeds.reset_index(drop=True)
    rows = []
    for chrom, sg in seeds.groupby("chrom", sort=False):
        s_pos = sg["pos"].to_numpy(np.int64)
        pool = np.unique(hint) if hint is not None else np.unique(s_pos)
        pos, vi, keep, Z = pc.chrom_Z_sparse(chrom, pool)
        ks = np.searchsorted(pos, s_pos)
        clip = np.clip(ks, 0, len(pos) - 1)
        valid = (ks < len(pos)) & (pos[clip] == s_pos)
        ok_ks = ks[valid]
        Zs = Z[np.searchsorted(keep, ok_ks)]
        for j, i in enumerate(sg.index.to_numpy()):
            if not valid[j]:
                continue
            sp = int(s_pos[j])
            a = int(np.searchsorted(pos, sp - WIN, side="left"))
            b = int(np.searchsorted(pos, sp + WIN, side="left"))
            q0 = max(int(np.searchsorted(keep, a, side="left")), 0)
            q1 = int(np.searchsorted(keep, b, side="left"))
            r2 = np.square(Z[q0:q1] @ Zs[j])
            sel = keep[q0:q1][r2 > R2_MIN]
            pp = pos[sel] if len(sel) else np.array([sp], dtype=np.int64)
            rows.append({"seed_row": int(i), "chrom": chrom,
                         "start": int(min(pp.min(), sp)), "end": int(max(pp.max(), sp))})
        pc.release(chrom)
        pc._sparse.clear()
    out = pd.DataFrame(rows).sort_values("seed_row").reset_index(drop=True)
    out["region_id"] = np.arange(len(out))
    return out

def bh(p):
 p=np.array(p,float);o=np.argsort(p,kind='mergesort');q=np.minimum(1,np.minimum.accumulate((p[o]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]);ret=np.empty(len(p));ret[o]=q;return ret

def select_leads(association,codes,variants):
    association=association.copy();p=association.p_wald.to_numpy(float)
    if not (np.isfinite(p)&(p>0)&(p<=1)).all():raise ValueError('The native formal family requires all tested positive finite P values')
    association['BH_q']=bh(p)
    if 'chrom' not in association:association['chrom']='Ca'+association.chr.astype(str).str.replace(r'^[Cc]a','',regex=True).str.lstrip('0')
    if 'pos' not in association:association=association.rename(columns={'ps':'pos'})
    significant=association[association.BH_q<=.05].sort_values('p_wald',kind='mergesort').drop_duplicates(['chrom','pos']).reset_index(drop=True)
    panel=Panel(codes,variants)
    leads=greedy_clump(panel,significant[['chrom','pos']]).merge(significant,on=['chrom','pos'],how='left',validate='one_to_one',sort=False)
    leads.insert(0,'lead_rank',np.arange(1,len(leads)+1))
    intervals=expand_perlead(panel,leads.head(100)[['chrom','pos']]) if len(leads) else pd.DataFrame(columns=['seed_row','chrom','start','end','region_id'])
    return association,leads,intervals


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--association',type=Path,required=True)
    parser.add_argument('--codes',type=Path,required=True)
    parser.add_argument('--variants',type=Path,required=True,help='TSV chrom,pos;source canonical0..3 genotypes')
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args();a,l,i=select_leads(pd.read_csv(args.association,sep='\t'),np.load(args.codes),pd.read_csv(args.variants,sep='\t'))
    args.output_directory.mkdir(parents=True,exist_ok=True)
    a.to_csv(args.output_directory/'tested_BH.tsv',sep='\t',index=False);l.to_csv(args.output_directory/'independent_leads.tsv',sep='\t',index=False);i.to_csv(args.output_directory/'Top100_perlead_intervals.tsv',sep='\t',index=False)
    (args.output_directory/'family_scope.json').write_text(json.dumps({'tested_family':len(a),'significant_threshold':.05,'clump_r2_strictly_above':.60,'window_bp':500000,'stable_P_order':True,'one_interval_per_lead_no_chaining':True},indent=2)+'\n')


if __name__=='__main__':main()

