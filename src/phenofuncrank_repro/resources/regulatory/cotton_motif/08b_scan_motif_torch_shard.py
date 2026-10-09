#!/usr/bin/env python3


import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pysam


BASES = "ACGT"
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}
COMP_IDX = np.array([3, 2, 1, 0], dtype=np.int64)


MOTIF_BASIC_FEATURES = [
    "motif_hit_ref_count",
    "motif_hit_alt_count",
    "motif_loss_count",
    "motif_gain_count",
    "motif_loss_any",
    "motif_gain_any",
    "motif_disruption_any",
    "max_abs_delta_pwm",
    "max_loss_delta_pwm",
    "max_gain_delta_pwm",
]

MOTIF_FAMILY_FEATURES = [
    "MYB_motif_disruption",
    "bHLH_motif_disruption",
    "NAC_motif_disruption",
    "HDZIP_motif_disruption",
    "WRKY_motif_disruption",
    "AP2ERF_motif_disruption",
    "bZIP_motif_disruption",
    "TCP_motif_disruption",
    "MADS_motif_disruption",
    "DOF_motif_disruption",
    "ARF_motif_disruption",
    "GATA_motif_disruption",
    "C2H2_ZF_motif_disruption",
]

MOTIF_ACTIVITY_FEATURES = [
    "motif_disruption_in_ATAC",
    "motif_loss_in_ATAC",
    "motif_gain_in_ATAC",

    "motif_disruption_in_H3K27ac",
    "motif_loss_in_H3K27ac",
    "motif_gain_in_H3K27ac",

    "motif_disruption_in_active_enhancer_like",
    "motif_loss_in_active_enhancer_like",

    "motif_disruption_in_active_promoter",
    "motif_loss_in_active_promoter",

    "max_abs_delta_pwm_in_ATAC",
    "max_loss_delta_pwm_in_ATAC",
    "max_gain_delta_pwm_in_ATAC",

    "max_abs_delta_pwm_in_H3K27ac",
    "max_loss_delta_pwm_in_H3K27ac",
    "max_gain_delta_pwm_in_H3K27ac",

    "max_abs_delta_pwm_in_active_enhancer_like",
    "max_loss_delta_pwm_in_active_enhancer_like",

    "max_abs_delta_pwm_in_active_promoter",
    "max_loss_delta_pwm_in_active_promoter",
]

MOTIF_QC_FEATURES = [
    "motif_scanned",
    "motif_ref_mismatch",
]

ALL_FEATURES = MOTIF_BASIC_FEATURES + MOTIF_FAMILY_FEATURES + MOTIF_ACTIVITY_FEATURES + MOTIF_QC_FEATURES


def load_family_map(path):
    if path is None:
        return {}

    df = pd.read_csv(path, sep="\t")
    fmap = {}
    if "motif_id" in df.columns and "family" in df.columns:
        for _, r in df.iterrows():
            fmap[str(r["motif_id"])] = str(r["family"])
    if "motif_name" in df.columns and "family" in df.columns:
        for _, r in df.iterrows():
            fmap[str(r["motif_name"])] = str(r["family"])
    return fmap


def parse_meme_motifs(meme_path, family_map, families):
    families = set(families)
    lines = Path(meme_path).read_text().splitlines()

    motifs = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("MOTIF "):
            i += 1
            continue

        parts = line.split()
        motif_id = parts[1]
        motif_name = parts[2] if len(parts) > 2 else motif_id
        family = family_map.get(motif_id) or family_map.get(motif_name) or "OTHER"

        i += 1
        while i < len(lines) and "letter-probability matrix" not in lines[i]:
            i += 1
        if i >= len(lines):
            break

        header = lines[i]
        m = re.search(r"w=\s*(\d+)", header)
        if not m:
            i += 1
            continue
        w = int(m.group(1))

        i += 1
        mat = []
        for _ in range(w):
            if i >= len(lines):
                break
            vals = lines[i].strip().split()
            if len(vals) < 4:
                break
            mat.append([float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])])
            i += 1

        if len(mat) != w:
            continue

        if family not in families:
            continue

        mat = np.array(mat, dtype=np.float32)
        mat = mat + 1e-4
        mat = mat / mat.sum(axis=1, keepdims=True)
        log_odds = np.log2(mat / 0.25).astype(np.float32)
        rc_log_odds = log_odds[::-1, :][:, COMP_IDX].copy()

        min_score = np.min(log_odds, axis=1).sum().astype(np.float32)
        max_score = np.max(log_odds, axis=1).sum().astype(np.float32)
        denom = np.float32(max(max_score - min_score, 1e-6))

        motifs.append({
            "motif_id": motif_id,
            "motif_name": motif_name,
            "family": family,
            "length": w,
            "log_odds": log_odds,
            "rc_log_odds": rc_log_odds,
            "min_score": min_score,
            "max_score": max_score,
            "denom": denom,
        })

    if not motifs:
        raise RuntimeError("No motifs kept. Check family map / families.")

    fam_counts = pd.Series([m["family"] for m in motifs]).value_counts().to_dict()
    len_counts = pd.Series([m["length"] for m in motifs]).value_counts().sort_index().to_dict()
    print("[motif] n=", len(motifs), flush=True)
    print("[motif] family counts=", fam_counts, flush=True)
    print("[motif] length counts=", len_counts, flush=True)

    return motifs


def make_motif_groups(motifs, device):
    groups = {}
    for m in motifs:
        L = m["length"]
        groups.setdefault(L, []).append(m)

    out = {}
    for L, ms in groups.items():
        log = np.stack([m["log_odds"] for m in ms], axis=0)
        rc = np.stack([m["rc_log_odds"] for m in ms], axis=0)
        min_scores = np.array([m["min_score"] for m in ms], dtype=np.float32)
        denoms = np.array([m["denom"] for m in ms], dtype=np.float32)
        families = [m["family"] for m in ms]

        out[L] = {
            "length": L,
            "n": len(ms),
            "log": torch.tensor(log, dtype=torch.float32, device=device),
            "rc": torch.tensor(rc, dtype=torch.float32, device=device),
            "min_scores": torch.tensor(min_scores, dtype=torch.float32, device=device),
            "denoms": torch.tensor(denoms, dtype=torch.float32, device=device),
            "families": families,
        }

    return out


def encode_chrom_seq(seq):
    arr = np.full(len(seq), -1, dtype=np.int16)
    b = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    arr[b == ord("A")] = 0
    arr[b == ord("C")] = 1
    arr[b == ord("G")] = 2
    arr[b == ord("T")] = 3
    return arr


def refs_to_idx(refs):
    return np.array([BASE_TO_IDX.get(str(x).upper(), -9) for x in refs], dtype=np.int16)


def alts_to_idx(alts):
    return np.array([BASE_TO_IDX.get(str(x).upper(), -9) for x in alts], dtype=np.int16)


def compute_best_norm_scores(codes_np, motif_groups, max_len, device):
    """
    codes_np: N x (2*max_len+1), int in [-1,0,1,2,3]
    returns:
      ref_scores_by_group, but directly aggregated outside to save memory.
    """
    codes = torch.tensor(codes_np.astype(np.int64), device=device)
    N = codes.shape[0]

    group_results = {}

    for L, g in motif_groups.items():
        M = g["n"]
        best = torch.full((N, M), -1e9, dtype=torch.float32, device=device)


        start_min = max_len - L + 1
        start_max = max_len

        for st in range(start_min, start_max + 1):
            sub = codes[:, st:st+L]
            valid = (sub >= 0).all(dim=1)

            sub2 = sub.clamp(min=0, max=3)
            oh = F.one_hot(sub2, num_classes=4).to(torch.float32)

            fw = torch.einsum("nlb,mlb->nm", oh, g["log"])
            rc = torch.einsum("nlb,mlb->nm", oh, g["rc"])
            sc = torch.maximum(fw, rc)
            sc[~valid, :] = -1e9
            best = torch.maximum(best, sc)

        norm = (best - g["min_scores"]) / g["denoms"]
        norm = torch.clamp(norm, 0.0, 1.0)
        group_results[L] = norm

    return group_results


def init_chunk_features(n):
    d = {}
    for c in ALL_FEATURES:
        d[c] = np.zeros(n, dtype=np.float32)
    return d


def add_activity_interactions(feat, ctx):
    atac = pd.to_numeric(ctx.get("atac_overlap_any", 0), errors="coerce").fillna(0).values.astype(np.float32)
    h3 = pd.to_numeric(ctx.get("h3k27ac_overlap_any", 0), errors="coerce").fillna(0).values.astype(np.float32)
    enh = pd.to_numeric(ctx.get("active_enhancer_like_flag", 0), errors="coerce").fillna(0).values.astype(np.float32)
    prom = pd.to_numeric(ctx.get("promoter_with_activity", 0), errors="coerce").fillna(0).values.astype(np.float32)

    md = feat["motif_disruption_any"]
    ml = feat["motif_loss_any"]
    mg = feat["motif_gain_any"]

    feat["motif_disruption_in_ATAC"] = ((md > 0) & (atac > 0)).astype(np.float32)
    feat["motif_loss_in_ATAC"] = ((ml > 0) & (atac > 0)).astype(np.float32)
    feat["motif_gain_in_ATAC"] = ((mg > 0) & (atac > 0)).astype(np.float32)

    feat["motif_disruption_in_H3K27ac"] = ((md > 0) & (h3 > 0)).astype(np.float32)
    feat["motif_loss_in_H3K27ac"] = ((ml > 0) & (h3 > 0)).astype(np.float32)
    feat["motif_gain_in_H3K27ac"] = ((mg > 0) & (h3 > 0)).astype(np.float32)

    feat["motif_disruption_in_active_enhancer_like"] = ((md > 0) & (enh > 0)).astype(np.float32)
    feat["motif_loss_in_active_enhancer_like"] = ((ml > 0) & (enh > 0)).astype(np.float32)

    feat["motif_disruption_in_active_promoter"] = ((md > 0) & (prom > 0)).astype(np.float32)
    feat["motif_loss_in_active_promoter"] = ((ml > 0) & (prom > 0)).astype(np.float32)

    feat["max_abs_delta_pwm_in_ATAC"] = feat["max_abs_delta_pwm"] * atac
    feat["max_loss_delta_pwm_in_ATAC"] = feat["max_loss_delta_pwm"] * atac
    feat["max_gain_delta_pwm_in_ATAC"] = feat["max_gain_delta_pwm"] * atac

    feat["max_abs_delta_pwm_in_H3K27ac"] = feat["max_abs_delta_pwm"] * h3
    feat["max_loss_delta_pwm_in_H3K27ac"] = feat["max_loss_delta_pwm"] * h3
    feat["max_gain_delta_pwm_in_H3K27ac"] = feat["max_gain_delta_pwm"] * h3

    feat["max_abs_delta_pwm_in_active_enhancer_like"] = feat["max_abs_delta_pwm"] * enh
    feat["max_loss_delta_pwm_in_active_enhancer_like"] = feat["max_loss_delta_pwm"] * enh

    feat["max_abs_delta_pwm_in_active_promoter"] = feat["max_abs_delta_pwm"] * prom
    feat["max_loss_delta_pwm_in_active_promoter"] = feat["max_loss_delta_pwm"] * prom


def process_chunk(df, chrom_codes, motif_groups, max_len, hit_thr, delta_thr, device):
    n = len(df)
    feat = init_chunk_features(n)

    candidate = pd.to_numeric(df["motif_candidate"], errors="coerce").fillna(0).values.astype(bool)
    ref_idx = refs_to_idx(df["Ref"].values)
    alt_idx = alts_to_idx(df["Alt"].values)
    pos0 = pd.to_numeric(df["Pos"], errors="coerce").values.astype(np.int64) - 1

    valid_base = candidate & (ref_idx >= 0) & (alt_idx >= 0)
    valid_base &= (pos0 - max_len >= 0) & (pos0 + max_len < len(chrom_codes))

    if valid_base.sum() == 0:
        add_activity_interactions(feat, df)
        out = pd.DataFrame({"snp_id": df["snp_id"].astype(str).values})
        for c in ALL_FEATURES:
            out[c] = feat[c]
        return out

    idx_all = np.where(valid_base)[0]
    offsets = np.arange(-max_len, max_len + 1, dtype=np.int64)
    mat_idx = pos0[idx_all, None] + offsets[None, :]
    seq_codes = chrom_codes[mat_idx]

    center = max_len
    ref_match = seq_codes[:, center] == ref_idx[idx_all]
    mismatch_idx = idx_all[~ref_match]
    if len(mismatch_idx) > 0:
        feat["motif_ref_mismatch"][mismatch_idx] = 1.0

    scan_local = np.where(ref_match)[0]
    scan_idx = idx_all[scan_local]

    if len(scan_idx) == 0:
        add_activity_interactions(feat, df)
        out = pd.DataFrame({"snp_id": df["snp_id"].astype(str).values})
        for c in ALL_FEATURES:
            out[c] = feat[c]
        return out

    ref_codes = seq_codes[scan_local].copy()
    alt_codes = ref_codes.copy()
    alt_codes[:, center] = alt_idx[scan_idx]

    feat["motif_scanned"][scan_idx] = 1.0

    ref_scores = compute_best_norm_scores(ref_codes, motif_groups, max_len, device)
    alt_scores = compute_best_norm_scores(alt_codes, motif_groups, max_len, device)


    N = len(scan_idx)
    hit_ref_count = np.zeros(N, dtype=np.float32)
    hit_alt_count = np.zeros(N, dtype=np.float32)
    loss_count = np.zeros(N, dtype=np.float32)
    gain_count = np.zeros(N, dtype=np.float32)
    max_abs = np.zeros(N, dtype=np.float32)
    max_loss = np.zeros(N, dtype=np.float32)
    max_gain = np.zeros(N, dtype=np.float32)

    fam_hits = {c: np.zeros(N, dtype=np.float32) for c in MOTIF_FAMILY_FEATURES}
    fam_to_feature = {
        "MYB": "MYB_motif_disruption",
        "bHLH": "bHLH_motif_disruption",
        "NAC": "NAC_motif_disruption",
        "HDZIP": "HDZIP_motif_disruption",
        "WRKY": "WRKY_motif_disruption",
        "AP2ERF": "AP2ERF_motif_disruption",
        "bZIP": "bZIP_motif_disruption",
        "TCP": "TCP_motif_disruption",
        "MADS": "MADS_motif_disruption",
        "DOF": "DOF_motif_disruption",
        "ARF": "ARF_motif_disruption",
        "GATA": "GATA_motif_disruption",
        "C2H2_ZF": "C2H2_ZF_motif_disruption",
    }

    for L, g in motif_groups.items():
        ref = ref_scores[L]
        alt = alt_scores[L]
        delta = alt - ref

        ref_hit = ref >= hit_thr
        alt_hit = alt >= hit_thr
        loss = ref_hit & ((~alt_hit) | (delta <= -delta_thr))
        gain = alt_hit & ((~ref_hit) | (delta >= delta_thr))
        disrupt = (loss | gain | (torch.abs(delta) >= delta_thr)) & (ref_hit | alt_hit)

        hit_ref_count += ref_hit.sum(dim=1).detach().cpu().numpy().astype(np.float32)
        hit_alt_count += alt_hit.sum(dim=1).detach().cpu().numpy().astype(np.float32)
        loss_count += loss.sum(dim=1).detach().cpu().numpy().astype(np.float32)
        gain_count += gain.sum(dim=1).detach().cpu().numpy().astype(np.float32)

        max_abs = np.maximum(max_abs, torch.abs(delta).max(dim=1).values.detach().cpu().numpy().astype(np.float32))
        max_loss = np.maximum(max_loss, torch.clamp(-delta, min=0).max(dim=1).values.detach().cpu().numpy().astype(np.float32))
        max_gain = np.maximum(max_gain, torch.clamp(delta, min=0).max(dim=1).values.detach().cpu().numpy().astype(np.float32))

        families = np.array(g["families"], dtype=object)
        dis_np = disrupt.detach().cpu().numpy()

        for fam, col in fam_to_feature.items():
            m = families == fam
            if m.any():
                fam_hits[col] = np.maximum(fam_hits[col], dis_np[:, m].any(axis=1).astype(np.float32))

        del ref, alt, delta, ref_hit, alt_hit, loss, gain, disrupt
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    feat["motif_hit_ref_count"][scan_idx] = hit_ref_count
    feat["motif_hit_alt_count"][scan_idx] = hit_alt_count
    feat["motif_loss_count"][scan_idx] = loss_count
    feat["motif_gain_count"][scan_idx] = gain_count
    feat["motif_loss_any"][scan_idx] = (loss_count > 0).astype(np.float32)
    feat["motif_gain_any"][scan_idx] = (gain_count > 0).astype(np.float32)
    feat["motif_disruption_any"][scan_idx] = ((loss_count > 0) | (gain_count > 0)).astype(np.float32)
    feat["max_abs_delta_pwm"][scan_idx] = max_abs
    feat["max_loss_delta_pwm"][scan_idx] = max_loss
    feat["max_gain_delta_pwm"][scan_idx] = max_gain

    for col, arr in fam_hits.items():
        feat[col][scan_idx] = arr

    add_activity_interactions(feat, df)

    out = pd.DataFrame({"snp_id": df["snp_id"].astype(str).values})
    for c in ALL_FEATURES:
        out[c] = feat[c].astype(np.float32)

    return out


def process_indexed_chunk(df, fasta, chrom, motif_groups, max_len, hit_thr, delta_thr, device):
    """Reuse native scoring with only indexed windows; no full chromosome encoding."""
    width = 2 * max_len + 1
    windows = np.full((len(df), width), -1, dtype=np.int16)
    local = df.copy()
    chromosome_length = fasta.get_reference_length(chrom)
    position0 = pd.to_numeric(df["Pos"], errors="coerce").to_numpy(dtype=np.int64) - 1
    within = (position0 - max_len >= 0) & (position0 + max_len < chromosome_length)
    candidate = pd.to_numeric(df["motif_candidate"], errors="coerce").fillna(0).to_numpy(dtype=bool)
    for index in np.flatnonzero(within & candidate):
        sequence = fasta.fetch(chrom, int(position0[index] - max_len),
                               int(position0[index] + max_len + 1)).upper()
        windows[index] = encode_chrom_seq(sequence)

    
    local["Pos"] = np.arange(len(df), dtype=np.int64) * width + max_len + 1
    local["motif_candidate"] = candidate & within
    return process_chunk(local, windows.reshape(-1), motif_groups, max_len,
                         hit_thr, delta_thr, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--motif-file", required=True)
    ap.add_argument("--tf-family-map", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--chroms", required=True, help="comma-separated chromosome list")
    ap.add_argument("--families", required=True, help="comma-separated families")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk-size", type=int, default=20000)
    ap.add_argument("--reference-mode", choices=["chromosome", "indexed"], default="chromosome",
                    help="Preserve chromosome mode or fetch only candidate windows with indexed FASTA.")
    ap.add_argument("--hit-threshold", type=float, default=0.80)
    ap.add_argument("--delta-threshold", type=float, default=0.10)
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available; using CPU", flush=True)
        device = "cpu"

    chroms = [x.strip() for x in args.chroms.split(",") if x.strip()]
    families = [x.strip() for x in args.families.split(",") if x.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    family_map = load_family_map(args.tf_family_map)
    motifs = parse_meme_motifs(args.motif_file, family_map, families)
    max_len = max(m["length"] for m in motifs)
    motif_groups = make_motif_groups(motifs, device)

    ctx = pd.read_parquet(args.context)
    ctx = ctx[ctx["Chr"].astype(str).isin(chroms)].copy()
    ctx = ctx.sort_values(["Chr", "Pos", "snp_id"]).reset_index(drop=True)

    print("[scan] context rows:", len(ctx), flush=True)
    print("[scan] candidate rows:", int(pd.to_numeric(ctx["motif_candidate"], errors="coerce").fillna(0).sum()), flush=True)
    print("[scan] chroms:", chroms, flush=True)
    print("[scan] device:", device, flush=True)
    print("[scan] max motif length:", max_len, flush=True)

    fasta = pysam.FastaFile(args.fasta)

    shard_manifest = {
        "context": args.context,
        "fasta": args.fasta,
        "motif_file": args.motif_file,
        "tf_family_map": args.tf_family_map,
        "chroms": chroms,
        "families": families,
        "device": device,
        "chunk_size": args.chunk_size,
        "reference_mode": args.reference_mode,
        "hit_threshold": args.hit_threshold,
        "delta_threshold": args.delta_threshold,
        "n_rows": int(len(ctx)),
        "n_candidates": int(pd.to_numeric(ctx["motif_candidate"], errors="coerce").fillna(0).sum()),
        "n_motifs": len(motifs),
        "family_motif_counts": {family: sum(m["family"] == family for m in motifs) for family in families},
        "families_without_motifs": [family for family in families if not any(m["family"] == family for m in motifs)],
    }
    with open(out_dir / ("manifest_" + "_".join(chroms) + ".json"), "w") as f:
        json.dump(shard_manifest, f, indent=2)

    for chrom, sub in ctx.groupby("Chr", sort=False):
        print(f"[chrom] {chrom} n={len(sub):,}", flush=True)

        try:
            fasta.get_reference_length(chrom)
            seq = fasta.fetch(chrom).upper() if args.reference_mode == "chromosome" else ""
        except Exception as e:
            print(f"[WARN] cannot fetch {chrom}: {e}; writing zero features", flush=True)
            seq = None

        chrom_codes = encode_chrom_seq(seq) if seq is not None and args.reference_mode == "chromosome" else None

        sub = sub.reset_index(drop=True)
        n = len(sub)
        for start in range(0, n, args.chunk_size):
            end = min(start + args.chunk_size, n)
            chunk_id = f"{chrom}_{start:012d}_{end:012d}"
            out_p = out_dir / f"motif_{chunk_id}.parquet"
            done_p = out_dir / f"motif_{chunk_id}.done"

            if done_p.exists() and out_p.exists() and not args.overwrite:
                print("[skip]", out_p, flush=True)
                continue

            df = sub.iloc[start:end].copy()

            if args.reference_mode == "indexed" and seq is not None:
                feat = process_indexed_chunk(df, fasta, chrom, motif_groups, max_len,
                                             args.hit_threshold, args.delta_threshold, device)
            elif chrom_codes is None:
                feat = pd.DataFrame({"snp_id": df["snp_id"].astype(str).values})
                for c in ALL_FEATURES:
                    feat[c] = np.zeros(len(df), dtype=np.float32)
            else:
                feat = process_chunk(
                    df=df,
                    chrom_codes=chrom_codes,
                    motif_groups=motif_groups,
                    max_len=max_len,
                    hit_thr=args.hit_threshold,
                    delta_thr=args.delta_threshold,
                    device=device,
                )

            feat.to_parquet(out_p, index=False)
            done_p.write_text("DONE\n")
            print(f"[done] {out_p} rows={len(feat):,}", flush=True)

    print("[DONE shard]", chroms, flush=True)


if __name__ == "__main__":
    main()
