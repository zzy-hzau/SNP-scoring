"""Allele-aware candidate genotypes, QC and pairwise-complete LD."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from phenofuncrank_repro.prioritization.allele_genotypes import norm_chrom

def decode_genotype_codes(codes, encoding="missing0_ref1_het2_alt3"):
    x = np.asarray(codes, dtype=np.float32)
    if encoding == "missing0_ref1_het2_alt3":
        x = x - 1
    elif encoding != "dosage012":
        raise ValueError(f"Unsupported genotype encoding: {encoding}")
    return np.where((x >= 0) & (x <= 2), x, np.nan)

class CandidateLD:
    def __init__(self, dosages):
        self.x = np.asarray(dosages, dtype=np.float32)
        if self.x.ndim != 2 or not self.x.shape[1]: raise ValueError("No candidate genotype panel")
        self._z = None

    def pair_stats(self, i, indices=None):
        idx = np.arange(len(self.x)) if indices is None else np.asarray(indices, int)
        out = []
        source = self.x[i]
        for start in range(0, len(idx), 2048):
            ids = idx[start:start+2048]; y = self.x[ids]
            valid = np.isfinite(y) & np.isfinite(source)[None, :]
            n = valid.sum(axis=1)
            xx = np.where(valid, source, 0).astype(np.float64)
            yy = np.where(valid, y, 0).astype(np.float64)
            sx = xx.sum(axis=1); sy = yy.sum(axis=1)
            cov = (xx*yy).sum(axis=1) - sx*sy/np.maximum(n, 1)
            vx = (xx*xx).sum(axis=1) - sx*sx/np.maximum(n, 1)
            vy = (yy*yy).sum(axis=1) - sy*sy/np.maximum(n, 1)
            r = np.divide(cov, np.sqrt(np.maximum(vx*vy, 0)), out=np.full(len(ids), np.nan), where=(vx>0)&(vy>0)&(n>=3))
            out.append(pd.DataFrame({"candidate_idx": ids, "r": np.clip(r,-1,1), "r2": np.clip(r*r,0,1), "n_pair": n, "n_pair_fraction": n/self.x.shape[1], "MAC_source_pair": np.minimum(sx, 2*n-sx), "MAC_target_pair": np.minimum(sy,2*n-sy)}))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    def __call__(self, i):

        
        if self._z is None:
            valid=np.isfinite(self.x);n=valid.sum(axis=1)
            means=np.divide(np.nansum(self.x,axis=1),n,out=np.zeros(len(n),np.float32),where=n>0)
            z=np.where(valid,self.x-means[:,None],0).astype(np.float32)
            norm=np.sqrt(np.einsum('ij,ij->i',z,z))
            self._z=z/np.where(norm[:,None]>0,norm[:,None],1)
        return np.clip(np.square(self._z@self._z[i]),0,1)

    def annotate(self, frame):
        f = frame.copy().reset_index(drop=True)
        n = np.isfinite(self.x).sum(axis=1); ac = np.nansum(self.x,axis=1); an = 2*n
        f["call_rate"] = n/self.x.shape[1]; f["MAC"] = np.minimum(ac, an-ac); f["N"] = self.x.shape[1]
        f["AF_cultivar"] = np.divide(ac, an, out=np.full(len(n), np.nan), where=an>0)
        f["He"] = 2*f.AF_cultivar*(1-f.AF_cultivar)
        return f

    def support_pairs(self, frame, spec):
        from phenofuncrank_repro.common.array_fields import _boolean
        core = _boolean(frame, ("is_core", "level1_core", "direct_core"))
        if "level1_class" in frame: core = frame.level1_class.eq("core").to_numpy()
        ids = frame.user_snp_id.astype(str).to_numpy() if "user_snp_id" in frame else frame.snp_id.astype(str).to_numpy()
        chrom = frame.chrom.map(lambda c: norm_chrom(c, spec)).to_numpy(); pos = pd.to_numeric(frame.pos).to_numpy()
        records = []
        for i in np.flatnonzero(core):
            mask = chrom == chrom[i]
            if spec.get("ld_window_bp") is not None: mask &= np.abs(pos-pos[i]) <= int(spec["ld_window_bp"])
            p = self.pair_stats(int(i), np.flatnonzero(mask))
            p = p[(p.r2 >= float(spec.get("ld_r2_threshold", .6))) & (p.n_pair_fraction >= .9) & (p.MAC_source_pair >= 10) & (p.MAC_target_pair >= 10)].copy()
            p["core_snp_id"] = ids[i]; p["proxy_snp_id"] = ids[p.candidate_idx.to_numpy(int)]
            records.append(p.drop(columns="candidate_idx"))
        return pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["core_snp_id","proxy_snp_id","r2","r","n_pair_fraction","MAC_source_pair","MAC_target_pair"])

def candidate_ld(frame, spec, vcf_path=None):
    f = frame.reset_index(drop=True)
    if vcf_path and not Path(str(vcf_path)).is_file():
        raise ValueError("Uploaded genotype VCF is unavailable")
    if vcf_path and Path(str(vcf_path)).is_file():
        import pysam
        with pysam.VariantFile(str(vcf_path)) as vcf:
            samples = list(vcf.header.samples)
            x = np.full((len(f),len(samples)),np.nan,dtype=np.float32)
            wanted = {}
            for i,row in enumerate(f.itertuples(index=False)):
                wanted.setdefault((norm_chrom(row.chrom,spec),int(row.pos)),[]).append((i,str(row.ref).upper(),str(row.alt).upper()))

            contigs = {norm_chrom(c,spec): c for c in vcf.header.contigs}
            intervals = []
            for chrom,group in f.groupby("chrom"):
                c = contigs.get(norm_chrom(chrom,spec))
                if not c: continue
                positions = sorted(set(pd.to_numeric(group.pos).astype(int)))
                start = end = positions[0]
                for p in positions[1:]:
                    if p-end <= 10000: end = p
                    else: intervals.append((c,start-1,end)); start=end=p
                intervals.append((c,start-1,end))
            try:
                for c,start,end in intervals:
                    for record in vcf.fetch(c,start,end): _store_record(record,samples,wanted,x,spec)
            except (ValueError,OSError):
                vcf.reset()
                for record in vcf: _store_record(record,samples,wanted,x,spec)
        return CandidateLD(x)
    meta = json.loads((Path(spec["inputs_dir"])/"metadata.json").read_text())
    if "original_snp_index" not in f: raise ValueError("Candidate genotype indices unavailable")
    indices = pd.to_numeric(f.original_snp_index,errors="coerce").to_numpy(float)
    genotype_file = Path(meta["genotype_memmap"])
    if not genotype_file.is_absolute(): genotype_file = Path(spec["inputs_dir"]) / genotype_file
    g = np.memmap(genotype_file,dtype=meta.get("dtype","uint8"),mode="r",shape=tuple(meta["shape"]))
    valid = np.isfinite(indices) & (indices>=0) & (indices<g.shape[0])
    x = np.full((len(f),g.shape[1]),np.nan,dtype=np.float32)
    x[valid] = decode_genotype_codes(g[indices[valid].astype(int)],meta.get("genotype_encoding","missing0_ref1_het2_alt3"))
    if 'dpc_ref' in f and 'dpc_alt' in f:
        source_ref=f.dpc_ref.astype(str).str.upper();source_alt=f.dpc_alt.astype(str).str.upper()
        target_ref=f.ref.astype(str).str.upper();target_alt=f.alt.astype(str).str.upper()
        flip=source_ref.eq(target_alt)&source_alt.eq(target_ref)
        exact=source_ref.eq(target_ref)&source_alt.eq(target_alt)
        x[flip.to_numpy()]=2-x[flip.to_numpy()]
        known=f.dpc_ref.notna()&f.dpc_alt.notna()
        x[(known&~exact&~flip).to_numpy()]=np.nan
    return CandidateLD(x)


def expand_uploaded_ld(core, panel, spec, vcf_path, *, return_audit=False):
    "Expand against submitted GTs in chromosome batches."
    core = core.copy().drop_duplicates("user_snp_id").reset_index(drop=True)
    panel = panel.copy().drop_duplicates("user_snp_id").reset_index(drop=True)
    core_ld = candidate_ld(core, spec, vcf_path)
    core = core_ld.annotate(core)
    core["level1_class"] = "core"
    core["is_core"] = True
    core["level1_core"] = True
    core["r2_to_core"] = 1.0
    core["phase_status"] = "core"
    core_ids = set(core.user_snp_id.astype(str))
    retained_ids = set()
    support = []
    audit = []
    detailed_audit = return_audit and len(core) * len(panel) <= 100000
    threshold = float(spec.get("ld_r2_threshold", .6))
    for chromosome, group in panel.groupby("chrom", sort=False):
        canonical = norm_chrom(chromosome, spec)
        anchors = [i for i, c in enumerate(core.chrom) if norm_chrom(c, spec) == canonical]
        for start in range(0, len(group), 20000):
            block = group.iloc[start:start + 20000].reset_index(drop=True)
            block_ld = candidate_ld(block, spec, vcf_path)
            combined = CandidateLD(np.vstack([core_ld.x[anchors], block_ld.x]))
            for offset, index in enumerate(anchors):
                anchor = core.iloc[index]
                pairs = combined.pair_stats(offset, np.arange(len(anchors), len(anchors) + len(block)))
                local_ids = pairs.candidate_idx.to_numpy(int) - len(anchors)
                pairs["panel_idx"] = local_ids
                pairs["core_snp_id"] = str(anchor.user_snp_id)
                pairs["proxy_snp_id"] = block.user_snp_id.astype(str).to_numpy()[local_ids]
                good = (pairs.r2 >= threshold) & (pairs.n_pair_fraction >= .9) & (pairs.MAC_source_pair >= 10) & (pairs.MAC_target_pair >= 10)
                good &= pairs.proxy_snp_id.ne(str(anchor.user_snp_id))
                if spec.get("ld_window_bp") is not None:
                    good &= np.abs(block.pos.to_numpy()[local_ids] - int(anchor.pos)) <= int(spec["ld_window_bp"])
                kept = pairs[good].copy()
                retained_ids.update(kept.proxy_snp_id)
                kept["r2_to_core"] = kept.r2
                kept["phase_status"] = "retained_uploaded_population"
                support.append(kept.drop(columns=["candidate_idx", "panel_idx"]))
                if return_audit:
                    observed = pairs if detailed_audit else kept
                    accepted_ids = set(kept.proxy_snp_id)
                    for row in observed.itertuples(index=False):
                        if row.proxy_snp_id == str(anchor.user_snp_id):
                            continue
                        accepted = row.proxy_snp_id in accepted_ids
                        audit.append({
                            "core_snp_id": row.core_snp_id, "proxy_snp_id": row.proxy_snp_id,
                            "r": row.r, "r2_to_core": row.r2, "n_pair": row.n_pair,
                            "n_pair_fraction": row.n_pair_fraction,
                            "MAC_source_pair": row.MAC_source_pair, "MAC_target_pair": row.MAC_target_pair,
                            "within_submitted_panel": True, "decision": "retained" if accepted else "excluded",
                            "reason": "uploaded_population_support" if accepted else "uploaded_population_ld_or_qc",
                            "genotype_population": "uploaded_population",
                        })
    columns = ["core_snp_id", "proxy_snp_id", "r", "r2", "r2_to_core", "n_pair", "n_pair_fraction", "MAC_source_pair", "MAC_target_pair", "phase_status"]
    pairs = pd.concat(support, ignore_index=True) if support else pd.DataFrame(columns=columns)
    proxies = panel[panel.user_snp_id.astype(str).isin(retained_ids - core_ids)].copy()
    proxies["level1_class"] = "ld_proxy"
    proxies["is_core"] = False
    proxies["level1_core"] = False
    proxies["phase_status"] = "retained_uploaded_population"
    if not proxies.empty:
        proxies["r2_to_core"] = proxies.user_snp_id.map(pairs.groupby("proxy_snp_id").r2.max())
    expanded = pd.concat([core, proxies], ignore_index=True).drop_duplicates("user_snp_id")
    result = (expanded, pairs)
    audit_columns = ["core_snp_id", "proxy_snp_id", "r", "r2_to_core", "n_pair", "n_pair_fraction", "MAC_source_pair", "MAC_target_pair", "within_submitted_panel", "decision", "reason", "genotype_population"]
    return (*result, pd.DataFrame(audit, columns=audit_columns)) if return_audit else result

def _store_record(record,samples,wanted,x,spec):
    if len(record.alts or ()) != 1: return
    for i,ref,alt in wanted.get((norm_chrom(record.chrom,spec),record.pos),[]):
        alleles = (record.ref.upper(),record.alts[0].upper())
        if alleles not in {(ref,alt),(alt,ref)}: continue
        swapped = alleles == (alt,ref)
        for j,sample in enumerate(samples):
            gt = record.samples[sample].get("GT")
            if not gt or len(gt)!=2 or any(a is None or a not in (0,1) for a in gt): continue
            dose = sum(gt); x[i,j] = 2-dose if swapped else dose

def annotate_proxy_direction(frame,pairs):
    "Record signed LD-supported direction."
    from phenofuncrank_repro.common.array_fields import _boolean
    f=frame.copy();id_col='user_snp_id' if 'user_snp_id' in f else 'snp_id'
    effects=pd.to_numeric(f.get('dpc_additive_effect_mean',pd.Series(np.nan,index=f.index)),errors='coerce')
    valid=_boolean(f,('is_core','level1_core')) & _boolean(f,('dpc_valid','dpc_evaluable'))
    lookup=dict(zip(f.loc[valid,id_col].astype(str),effects[valid]))
    directions={}
    for r in pairs.itertuples(index=False):
        beta=lookup.get(str(r.core_snp_id))
        if beta is None or not np.isfinite(beta) or beta==0 or not np.isfinite(r.r):continue
        directions.setdefault(str(r.proxy_snp_id),set()).add(int(np.sign(beta*r.r)))
    f['trait_support_direction']=[('increase' if next(iter(directions.get(str(s),{0})))>0 else 'decrease') if len(directions.get(str(s),set()))==1 else None for s in f[id_col]]
    f['trait_support_direction_status']=['ld_supported' if len(directions.get(str(s),set()))==1 else 'conflicting_sources' if len(directions.get(str(s),set()))>1 else 'unavailable' for s in f[id_col]]
    for i in np.flatnonzero(valid):
        if np.isfinite(effects.iloc[i]) and effects.iloc[i]!=0:
            f.loc[f.index[i],'trait_support_direction']='increase' if effects.iloc[i]>0 else 'decrease'
            f.loc[f.index[i],'trait_support_direction_status']='direct_model_effect'
    return f
