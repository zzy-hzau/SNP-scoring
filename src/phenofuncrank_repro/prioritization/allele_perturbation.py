#!/usr/bin/env python3
"Optimized allele perturbation for DPCformer vs GWAS beta validation."
from __future__ import annotations
from phenofuncrank_repro.common.paths import project_root, package_dir

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PUBLIC_ROOT = project_root(__file__)
from phenofuncrank_repro.models.dpcformer.network import PhenotypePredictor_1v
from phenofuncrank_repro.models.dpcformer.inputs import read_genotype_memmap

BASE_TO_VEC = {
    "A": np.array([1, 0, 0, 0], dtype=np.float32),
    "C": np.array([0, 1, 0, 0], dtype=np.float32),
    "G": np.array([0, 0, 1, 0], dtype=np.float32),
    "T": np.array([0, 0, 0, 1], dtype=np.float32),
}


def parse_folds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def trimmed_mean(values: np.ndarray, proportion: float = 0.1) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    if len(values) < 3 or proportion <= 0:
        return float(np.mean(values))
    vals = np.sort(values)
    k = int(np.floor(len(vals) * proportion))
    if k == 0 or 2 * k >= len(vals):
        return float(np.mean(vals))
    return float(np.mean(vals[k:-k]))


def safe_sign_match(a: float, b: float) -> float:
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(a) or not np.isfinite(b) or a == 0 or b == 0:
        return np.nan
    return float(np.sign(a) == np.sign(b))


def normalize_base(x) -> str | float:
    if pd.isna(x):
        return np.nan
    s = str(x).upper()
    return s if s in BASE_TO_VEC else np.nan


def codes_to_onehot(codes_snp_by_sample: np.ndarray, snp_meta: pd.DataFrame, selected_snp_count: int) -> np.ndarray:
    n_selected, n_samples = codes_snp_by_sample.shape
    out = np.zeros((n_samples, selected_snp_count, 8), dtype=np.float32)
    refs = snp_meta["Ref"].astype(str).str.upper().values
    alts = snp_meta["Alt"].astype(str).str.upper().values
    for j in range(n_selected):
        ref = refs[j]
        alt = alts[j]
        if ref not in BASE_TO_VEC or alt not in BASE_TO_VEC:
            continue
        rv = BASE_TO_VEC[ref]
        av = BASE_TO_VEC[alt]
        c = codes_snp_by_sample[j, :]
        m = c == 1
        out[m, j, 0:4] = rv
        out[m, j, 4:8] = rv
        m = c == 2
        out[m, j, 0:4] = rv
        out[m, j, 4:8] = av
        m = c == 3
        out[m, j, 0:4] = av
        out[m, j, 4:8] = av
    return out


def build_input_tensor(
    geno: np.memmap,
    selected_map: pd.DataFrame,
    sample_indices: np.ndarray,
    selected_snp_count: int,
    n_groups: int,
) -> torch.Tensor:
    grouped = []
    for group_id in range(1, n_groups + 1):
        gm = (
            selected_map[selected_map["group_id"].eq(group_id)]
            .sort_values("selected_rank")
            .reset_index(drop=True)
        )
        if len(gm) == 0:
            raise RuntimeError(f"No selected SNPs for group {group_id}")
        if len(gm) > selected_snp_count:
            gm = gm.iloc[:selected_snp_count].copy()
        variant_idx = gm["original_snp_index"].to_numpy(dtype=np.int64)
        codes = np.asarray(geno[variant_idx, :][:, sample_indices], dtype=np.uint8)
        grouped.append(codes_to_onehot(codes, gm, selected_snp_count))
    return torch.tensor(np.stack(grouped, axis=1), dtype=torch.float32)


@torch.inference_mode()
def predict_batches(model, x: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size].to(device, non_blocking=True)
        y = model(xb).detach().cpu().numpy().reshape(-1)
        preds.append(y)
    return np.concatenate(preds)


def load_model(model_dir: Path, fold: int, n_groups: int, selected_snp_count: int, device: torch.device, trait_prefix: str = "Plant_height"):
    model_path = model_dir / "saved_models" / f"{trait_prefix}.best_model_fold_{fold}.pth"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model = PhenotypePredictor_1v(n_groups, [selected_snp_count] * n_groups, 8).to(device)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_fold_scaler(model_dir: Path, fold: int, trait_prefix: str = "Plant_height") -> tuple[float, float]:
    path = model_dir / f"fold_{fold:02d}" / f"{trait_prefix}.fold_{fold:02d}.phenotype_scaler.npz"
    data = np.load(path)
    mean = float(np.asarray(data["mean_"]).reshape(-1)[0])
    scale = float(np.asarray(data["scale_"]).reshape(-1)[0])
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return mean, scale


def infer_pheno_col(model_dir: Path, explicit: str | None, trait_prefix: str = "Plant_height") -> str:
    if explicit:
        return explicit
    summary = model_dir / f"{trait_prefix}.cross_validation_summary.txt"
    if summary.exists():
        for line in summary.read_text().splitlines():
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[0] == "pheno_col":
                return parts[1]
    return trait_prefix


def valid_sample_columns(inputs_dir: Path, meta: dict, pheno_col: str) -> np.ndarray:
    phenotype_path = inputs_dir / "matched_phenotypes.tsv"
    if not phenotype_path.exists():
        relative = meta.get("phenotypes")
        if not relative or Path(relative).is_absolute():
            raise FileNotFoundError("Supply the local phenotype resource within the bundle")
        phenotype_path = inputs_dir / relative
    pheno = pd.read_csv(phenotype_path, sep="\t")
    if pheno_col not in pheno.columns:
        raise KeyError(f"Phenotype column {pheno_col!r} not found in {phenotype_path}")
    y_all = pd.to_numeric(pheno[pheno_col], errors="coerce").to_numpy(dtype=float)
    return np.flatnonzero(np.isfinite(y_all) & (y_all != -9.0)).astype(np.int64)


def fold_sample_columns(model_dir: Path, fold: int, sample_scope: str, valid_cols: np.ndarray, trait_prefix: str = "Plant_height") -> np.ndarray:
    splits_path = model_dir / f"fold_{fold:02d}" / f"{trait_prefix}.fold_{fold:02d}.splits.npz"
    splits = np.load(splits_path, allow_pickle=True)
    if sample_scope == "test":
        return splits["test_memmap_cols"].astype(np.int64)
    if sample_scope == "all_valid":
        return valid_cols.astype(np.int64)
    raise ValueError(sample_scope)


def summarize_per_sample(add: np.ndarray, homo: np.ndarray, het: np.ndarray, dom: np.ndarray) -> dict:
    mean_add = float(np.mean(add))
    nz = add[np.isfinite(add) & (add != 0)]
    if mean_add == 0 or len(nz) == 0:
        direction_consistency = np.nan
    else:
        direction_consistency = float(np.mean(np.sign(nz) == np.sign(mean_add)))
    return {
        "dpc_homo_effect_mean": float(np.mean(homo)),
        "dpc_homo_effect_median": float(np.median(homo)),
        "dpc_additive_effect_mean": mean_add,
        "dpc_additive_effect_median": float(np.median(add)),
        "dpc_additive_effect_trimmed_mean": trimmed_mean(add, 0.1),
        "sample_delta_sd": float(np.std(add, ddof=1)) if len(add) > 1 else 0.0,
        "sample_direction_consistency": direction_consistency,
        "dpc_het_effect_mean": float(np.mean(het)),
        "dpc_dominance_deviation_mean": float(np.mean(dom)),
    }


@torch.inference_mode()
def run_fold_optimized(
    args,
    fold: int,
    targets: pd.DataFrame,
    geno: np.memmap,
    n_groups: int,
    selected_snp_count: int,
    sample_indices: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    """
    Optimized fold processing with CNN feature caching.

    Instead of running the full model (12 CNNs + Transformer + MLP) for each
    (SNP, state, sample) triple, we:
    1. Pre-compute base CNN outputs for all 12 chromosomes (cache)
    2. For each SNP, only re-compute the ONE affected chromosome's CNN
    3. Combine modified + cached CNN outputs → Transformer → MLP
    This reduces CNN computation by ~12x.
    """
    fold_dir = args.model_dir / f"fold_{fold:02d}"
    selected_map_path = fold_dir / f"{args.trait_prefix}.fold_{fold:02d}.selected_snp_map.tsv"
    if not selected_map_path.exists():
        raise FileNotFoundError(selected_map_path)

    selected = pd.read_csv(selected_map_path, sep="\t")
    selected["chrom"] = selected["chromosome"].astype(int)
    selected["pos"] = selected["Pos"].astype(int)
    selected["ref_selected_fold"] = selected["Ref"].map(normalize_base)
    selected["alt_selected_fold"] = selected["Alt"].map(normalize_base)
    selected["dpc_selected_snp_id"] = selected["snp_id"]

    merge_cols = [
        "chrom", "pos", "ref_selected_fold", "alt_selected_fold",
        "dpc_selected_snp_id", "group_id", "selected_rank", "original_snp_index",
    ]
    fold_targets = targets.merge(
        selected[merge_cols],
        left_on=["chrom", "pos", "ref", "alt"],
        right_on=["chrom", "pos", "ref_selected_fold", "alt_selected_fold"],
        how="inner",
    )
    if args.limit_snps:
        fold_targets = fold_targets.head(args.limit_snps).copy()
    fold_targets = fold_targets.reset_index(drop=True)
    if len(fold_targets) == 0:
        return pd.DataFrame()

    mean, scale = load_fold_scaler(args.model_dir, fold, args.trait_prefix)
    model = load_model(args.model_dir, fold, n_groups, selected_snp_count, device, args.trait_prefix)
    n_samples = len(sample_indices)


    x_base = build_input_tensor(geno, selected, sample_indices, selected_snp_count, n_groups)
    x_base = x_base.to(device)  


    base_cnn_outputs = []
    for g in range(n_groups):
        chrom_in = x_base[:, g, :, :].permute(0, 2, 1)  
        cnn_out = model.chromosome_cnns[g](chrom_in)     
        base_cnn_outputs.append(cnn_out)


    

    cnn_channels = int(base_cnn_outputs[0].shape[1])
    cnn_length = int(base_cnn_outputs[0].shape[2])


    
    base_cat = torch.cat(base_cnn_outputs, dim=2)  
    base_trans_in = base_cat.permute(0, 2, 1).contiguous()  


    snp_batch_size = args.snp_batch_size
    records = []
    per_sample_records = []
    n_total_snps = len(fold_targets)
    t_fold_start = time.time()

    for batch_start in range(0, n_total_snps, snp_batch_size):
        batch = fold_targets.iloc[batch_start:batch_start + snp_batch_size]
        n_batch = len(batch)


        by_chrom: dict[int, list] = {}
        for i, (_, row) in enumerate(batch.iterrows()):
            g = int(row["group_id"]) - 1
            r = int(row["selected_rank"]) - 1
            ref = str(row["ref"]).upper()
            alt = str(row["alt"]).upper()
            if ref not in BASE_TO_VEC or alt not in BASE_TO_VEC:
                continue
            by_chrom.setdefault(g, []).append((i, r, ref, alt))


        
        mod_cnn_by_batch_idx: dict[int, tuple[int, torch.Tensor]] = {}

        for g, snps_on_g in by_chrom.items():
            n_snps_g = len(snps_on_g)
            base_chrom = x_base[:, g, :, :].permute(0, 2, 1)  


            

            max_cnn_snps = args.max_cnn_snps
            mod_cnn_parts = []

            for cnn_start in range(0, n_snps_g, max_cnn_snps):
                snps_sub = snps_on_g[cnn_start:cnn_start + max_cnn_snps]
                n_sub_g = len(snps_sub)

                mod_chrom = base_chrom.unsqueeze(0).repeat(n_sub_g * 3, 1, 1, 1).clone()

                for j, (batch_idx, r, ref, alt) in enumerate(snps_sub):
                    ref_vec = torch.tensor(BASE_TO_VEC[ref], device=device, dtype=torch.float32)
                    alt_vec = torch.tensor(BASE_TO_VEC[alt], device=device, dtype=torch.float32)
                    mod_chrom[j * 3, :, 0:4, r] = ref_vec
                    mod_chrom[j * 3, :, 4:8, r] = ref_vec
                    mod_chrom[j * 3 + 1, :, 0:4, r] = ref_vec
                    mod_chrom[j * 3 + 1, :, 4:8, r] = alt_vec
                    mod_chrom[j * 3 + 2, :, 0:4, r] = alt_vec
                    mod_chrom[j * 3 + 2, :, 4:8, r] = alt_vec

                mod_chrom_flat = mod_chrom.reshape(-1, 8, selected_snp_count)
                del mod_chrom

                cnn_outs = []
                for sub_start in range(0, mod_chrom_flat.shape[0], args.cnn_sub_batch):
                    sub = mod_chrom_flat[sub_start:sub_start + args.cnn_sub_batch]
                    cnn_outs.append(model.chromosome_cnns[g](sub))
                mod_cnn_sub = torch.cat(cnn_outs, dim=0)
                del mod_chrom_flat, cnn_outs

                mod_cnn_sub = mod_cnn_sub.reshape(n_sub_g * 3, n_samples, cnn_channels, cnn_length)
                mod_cnn_parts.append(mod_cnn_sub)

            mod_cnn = torch.cat(mod_cnn_parts, dim=0)  
            del mod_cnn_parts

            for j, (batch_idx, r, ref, alt) in enumerate(snps_on_g):
                mod_cnn_by_batch_idx[batch_idx] = (g, mod_cnn[j * 3:j * 3 + 3])


        
        pairs = []
        for batch_idx in range(n_batch):
            if batch_idx not in mod_cnn_by_batch_idx:
                continue
            g, mod_cnn_states = mod_cnn_by_batch_idx[batch_idx]
            for state in range(3):
                pairs.append((batch_idx, state, g, mod_cnn_states[state]))


        

        

        

        

        
        max_transformer_sequences = args.max_transformer_sequences
        trans_sub = min(
            args.transformer_sub_batch,
            max(1, max_transformer_sequences // max(1, n_samples)),
        )
        all_preds: dict[tuple[int, int], np.ndarray] = {}

        for sub_start in range(0, len(pairs), trans_sub):
            sub_pairs = pairs[sub_start:sub_start + trans_sub]
            n_sub = len(sub_pairs)


            trans_in = base_trans_in.unsqueeze(0).expand(n_sub, -1, -1, -1).clone()


            for k, (batch_idx, state, g, mod_state) in enumerate(sub_pairs):

                mod_trans = mod_state.permute(0, 2, 1)
                trans_in[k, :, g * cnn_length:(g + 1) * cnn_length, :] = mod_trans


            trans_in = trans_in.reshape(-1, n_groups * cnn_length, cnn_channels)


            trans_out = model.transformer_decoder(trans_in)
            flat = torch.flatten(trans_out, 1)
            pred = model.mlp(flat).squeeze(-1)  


            pred_raw = (pred * scale + mean).cpu().numpy()


            for k, (batch_idx, state, g, mod_state) in enumerate(sub_pairs):
                all_preds[(batch_idx, state)] = pred_raw[k * n_samples:(k + 1) * n_samples]

            del trans_in, trans_out, flat, pred
            if device.type == "cuda":
                torch.cuda.empty_cache()


        for i, (_, row) in enumerate(batch.iterrows()):
            if (i, 0) not in all_preds:
                continue

            pred_ref = all_preds[(i, 0)]
            pred_het = all_preds[(i, 1)]
            pred_alt = all_preds[(i, 2)]

            homo = pred_alt - pred_ref
            add = homo / 2.0
            het = pred_het - pred_ref
            dom = pred_het - (pred_ref + pred_alt) / 2.0
            if args.save_per_sample:
                per_sample_records.append({
                    "snp_id": row["snp_id"],
                    "chrom": int(row["chrom"]),
                    "pos": int(row["pos"]),
                    "ref": row["ref"],
                    "alt": row["alt"],
                    "pred_ref": pred_ref.copy(),
                    "pred_het": pred_het.copy(),
                    "pred_alt": pred_alt.copy(),
                    "additive_effect": add.copy(),
                })
            stats = summarize_per_sample(add, homo, het, dom)

            rec = {
                "sample_scope": args.sample_scope,
                "fold": fold,
                "n_samples": n_samples,
                "snp_id": row["snp_id"],
                "chrom": int(row["chrom"]),
                "pos": int(row["pos"]),
                "ref": row["ref"],
                "alt": row["alt"],
                "dpc_selected_snp_id": row.get("dpc_selected_snp_id"),
                "locus_id_final": row.get("locus_id_final"),
                "block_id": row.get("block_id"),
                "candidate_tier": row.get("candidate_tier"),
                "group_id": int(row["group_id"]),
                "selected_rank": int(row["selected_rank"]),
                "beta_source": row.get("beta_source"),
                "beta_alt_marginal": row.get("beta_alt_marginal"),
                "beta_alt_conditional": row.get("beta_alt_conditional"),
                "p_marginal": row.get("p_marginal"),
                "p_conditional": row.get("p_conditional"),
                "pip": row.get("pip"),
                "best_PHRED": row.get("best_PHRED"),
                "mean_pred_ref": float(np.mean(pred_ref)),
                "mean_pred_het": float(np.mean(pred_het)),
                "mean_pred_alt": float(np.mean(pred_alt)),
            }
            for col in row.index:
                if str(col).startswith("set_") or str(col).startswith("dpc_input_"):
                    rec[col] = row[col]
            rec.update(stats)
            records.append(rec)

        elapsed = time.time() - t_fold_start
        done = min(batch_start + snp_batch_size, n_total_snps)
        rate = done / max(elapsed, 0.001)
        eta = (n_total_snps - done) / max(rate, 0.001)
        print(
            f"  [FOLD {fold}] {done}/{n_total_snps} SNPs | "
            f"{rate:.0f} SNP/s | ETA {eta:.0f}s | "
            f"GPU mem {torch.cuda.memory_allocated()/1e9:.1f} GB"
            if device.type == "cuda"
            else f"  [FOLD {fold}] {done}/{n_total_snps} SNPs | {rate:.0f} SNP/s | ETA {eta:.0f}s",
            flush=True,
        )


    del x_base, base_cnn_outputs, base_cat, base_trans_in, model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.save_per_sample and per_sample_records:
        ps_path = Path(args.per_sample_out)
        ps_name = ps_path.name
        if "{fold" in ps_name:
            ps_name = ps_name.format(fold=fold)
        else:
            ps_name = f"{ps_path.stem}.fold{fold:02d}{ps_path.suffix}"
        ps_path = ps_path.with_name(ps_name)
        if not ps_path.is_absolute():
            ps_path = args.out_dir / ps_path
        ps_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            ps_path,
            snp_ids=np.array([str(r["snp_id"]) for r in per_sample_records]),
            chroms=np.array([r["chrom"] for r in per_sample_records], dtype=np.int64),
            positions=np.array([r["pos"] for r in per_sample_records], dtype=np.int64),
            refs=np.array([str(r["ref"]) for r in per_sample_records]),
            alts=np.array([str(r["alt"]) for r in per_sample_records]),
            sample_indices=sample_indices.astype(np.int64),
            effects=np.stack([r["additive_effect"] for r in per_sample_records]),
            pred_ref=np.stack([r["pred_ref"] for r in per_sample_records]),
            pred_het=np.stack([r["pred_het"] for r in per_sample_records]),
            pred_alt=np.stack([r["pred_alt"] for r in per_sample_records]),
        )
        print(f"[FOLD {fold}] saved per-sample effects to {ps_path} "
              f"({len(per_sample_records)} SNPs x {len(sample_indices)} samples)", flush=True)

    return pd.DataFrame(records)


def aggregate_fold_results(
    per_fold: pd.DataFrame,
    targets: pd.DataFrame,
    total_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    if len(per_fold):
        for snp_id, sub in per_fold.groupby("snp_id", sort=False):
            first = sub.iloc[0]
            fold_mean = pd.to_numeric(sub["dpc_additive_effect_mean"], errors="coerce").to_numpy(dtype=float)
            agg_mean = float(np.nanmean(fold_mean))
            valid_fold = np.isfinite(fold_mean)
            fold_nonzero = fold_mean[valid_fold & (fold_mean != 0)]
            if agg_mean == 0 or len(fold_nonzero) == 0:
                fold_dir_cons = np.nan
            else:
                fold_dir_cons = float(np.mean(np.sign(fold_nonzero) == np.sign(agg_mean)))
            row = {
                "sample_scope": first["sample_scope"],
                "snp_id": snp_id,
                "chrom": int(first["chrom"]),
                "pos": int(first["pos"]),
                "ref": first["ref"],
                "alt": first["alt"],
                "dpc_selected_snp_ids": ",".join(sorted(set(map(str, sub["dpc_selected_snp_id"].dropna())))),
                "locus_id_final": first.get("locus_id_final"),
                "block_id": first.get("block_id"),
                "candidate_tier": first.get("candidate_tier"),
                "beta_source": first.get("beta_source"),
                "beta_alt_marginal": first.get("beta_alt_marginal"),
                "beta_alt_conditional": first.get("beta_alt_conditional"),
                "p_marginal": first.get("p_marginal"),
                "p_conditional": first.get("p_conditional"),
                "pip": first.get("pip"),
                "best_PHRED": first.get("best_PHRED"),
                "n_evaluable_folds": int(sub["fold"].nunique()),
                "fold_selection_rate": float(sub["fold"].nunique() / total_folds),
                "evaluable_folds": ",".join(map(str, sorted(sub["fold"].astype(int).unique()))),
                "fold_effect_sd": float(np.nanstd(fold_mean, ddof=1)) if np.sum(valid_fold) > 1 else 0.0,
                "fold_direction_consistency": fold_dir_cons,
                "dpc_additive_effect_mean": agg_mean,
                "dpc_additive_effect_median": float(np.nanmedian(pd.to_numeric(sub["dpc_additive_effect_median"], errors="coerce"))),
                "dpc_additive_effect_trimmed_mean": float(np.nanmean(pd.to_numeric(sub["dpc_additive_effect_trimmed_mean"], errors="coerce"))),
                "dpc_homo_effect_mean": float(np.nanmean(pd.to_numeric(sub["dpc_homo_effect_mean"], errors="coerce"))),
                "dpc_het_effect_mean": float(np.nanmean(pd.to_numeric(sub["dpc_het_effect_mean"], errors="coerce"))),
                "dpc_dominance_deviation_mean": float(np.nanmean(pd.to_numeric(sub["dpc_dominance_deviation_mean"], errors="coerce"))),
                "sample_delta_sd_mean_across_folds": float(np.nanmean(pd.to_numeric(sub["sample_delta_sd"], errors="coerce"))),
                "sample_direction_consistency_mean_across_folds": float(np.nanmean(pd.to_numeric(sub["sample_direction_consistency"], errors="coerce"))),
            }
            for col in sub.columns:
                if str(col).startswith("set_") or str(col).startswith("dpc_input_"):
                    row[col] = bool(sub[col].fillna(False).astype(bool).any())
            row["sign_match_marginal_mean"] = safe_sign_match(row["dpc_additive_effect_mean"], row["beta_alt_marginal"])
            row["sign_match_marginal_median"] = safe_sign_match(row["dpc_additive_effect_median"], row["beta_alt_marginal"])
            row["sign_match_conditional_mean"] = safe_sign_match(row["dpc_additive_effect_mean"], row["beta_alt_conditional"])
            row["sign_match_conditional_median"] = safe_sign_match(row["dpc_additive_effect_median"], row["beta_alt_conditional"])
            rows.append(row)
    agg = pd.DataFrame(rows)

    target_base = targets[["snp_id", "chrom", "pos", "ref", "alt", "selected_fold_count", "fold_selection_rate_prior"]].copy()
    if len(agg):
        not_sel = target_base.merge(agg[["snp_id", "n_evaluable_folds"]], on="snp_id", how="left")
    else:
        not_sel = target_base.copy()
        not_sel["n_evaluable_folds"] = np.nan
    not_sel["n_evaluable_folds"] = not_sel["n_evaluable_folds"].fillna(0).astype(int)
    not_sel["not_evaluable_reason"] = np.where(
        not_sel["n_evaluable_folds"].eq(0),
        "not_selected_in_requested_folds_or_allele_mismatch",
        "evaluable",
    )
    for col in targets.columns:
        if str(col).startswith("set_") or str(col).startswith("dpc_input_") or col in ["locus_id_final", "block_id", "candidate_tier"]:
            if col not in not_sel.columns:
                not_sel = not_sel.merge(targets[["snp_id", col]], on="snp_id", how="left")

    set_cols = [c for c in targets.columns if c.startswith("set_")]
    eval_rows = []
    for col in set_cols:
        mask = targets[col].fillna(False).astype(bool)
        n_input = int(mask.sum())
        if n_input == 0:
            continue
        input_ids = set(targets.loc[mask, "snp_id"])
        n_eval = int(len(set(agg["snp_id"]) & input_ids)) if len(agg) else 0
        eval_rows.append({
            "snp_set": col.replace("set_", ""),
            "n_input_snps": n_input,
            "n_evaluable_snps": n_eval,
            "evaluable_rate": n_eval / n_input if n_input else np.nan,
        })
    evaluability = pd.DataFrame(eval_rows)
    return agg, not_sel, evaluability


def parse_args():
    ap = argparse.ArgumentParser(description="Optimized DPCformer allele perturbation")
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--inputs-dir", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sample-scope", choices=["test", "all_valid"], default="all_valid")
    ap.add_argument("--folds", default="1-10")
    ap.add_argument(
        "--total-folds-for-rate", type=int, default=10,
        help="Denominator for fold_selection_rate; keep 10 for formal rice DPCformer validation.",
    )
    ap.add_argument("--selected-snp-count", type=int, default=1000)
    ap.add_argument("--pheno-col", default=None)
    ap.add_argument("--trait-prefix", default=None,
                    help="Prefix used by model fold files; defaults to --pheno-col or Plant_height.")
    ap.add_argument("--device", default="auto")

    ap.add_argument("--snp-batch-size", type=int, default=500,
                    help="Number of SNPs per batch (default 500, was 8 in original)")
    ap.add_argument("--cnn-sub-batch", type=int, default=20000,
                    help="Max samples per CNN forward pass (controls peak memory)")
    ap.add_argument("--max-cnn-snps", type=int, default=30,
                    help="Max SNPs per chromosome per CNN sub-batch (controls mod_chrom tensor size)")
    ap.add_argument("--transformer-sub-batch", type=int, default=100,
                    help="Number of (snp,state) pairs per transformer forward pass")
    ap.add_argument("--max-transformer-sequences", type=int, default=60000,
                    help="Maximum perturbation-pair x sample sequences per transformer pass")
    ap.add_argument("--pred-batch-size", type=int, default=512)
    ap.add_argument("--limit-snps", type=int, default=0)
    ap.add_argument("--gpu-id", type=int, default=0)

    ap.add_argument("--save-per-sample", action="store_true",
                    help="Save per-sample effects (pred_ref/het/alt, additive) to npz file")
    ap.add_argument("--per-sample-out", type=str, default="per_sample_effects.fold{fold:02d}.npz",
                    help="Output path for per-sample effects npz (supports {fold} template)")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(32, os.cpu_count() or 16))
    return args, device


def main():
    args, device = parse_args()
    print(f"[INFO] device={device}", flush=True)
    print(f"[INFO] sample_scope={args.sample_scope}", flush=True)
    print(f"[INFO] snp_batch_size={args.snp_batch_size} cnn_sub_batch={args.cnn_sub_batch} "
          f"transformer_sub_batch={args.transformer_sub_batch}", flush=True)

    meta = json.loads((args.inputs_dir / "metadata.json").read_text())
    n_snps, n_samples_meta = map(int, meta["shape"])
    n_groups = int(meta["n_groups"])
    geno, _ = read_genotype_memmap(args.inputs_dir)

    args.trait_prefix = args.trait_prefix or args.pheno_col or "Plant_height"
    pheno_col = infer_pheno_col(args.model_dir, args.pheno_col, args.trait_prefix)
    valid_cols = valid_sample_columns(args.inputs_dir, meta, pheno_col)
    folds = parse_folds(args.folds)

    targets = pd.read_csv(args.targets, sep="\t", low_memory=False)
    for col in ["ref", "alt"]:
        targets[col] = targets[col].map(normalize_base)
    targets = targets[targets["ref"].notna() & targets["alt"].notna()].copy()
    targets = targets.drop_duplicates("snp_id").reset_index(drop=True)
    print(f"[INFO] targets={len(targets)} folds={folds} n_groups={n_groups}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_fold_parts = []
    t_total_start = time.time()

    for fold in folds:
        sample_indices = fold_sample_columns(args.model_dir, fold, args.sample_scope, valid_cols, args.trait_prefix)
        print(f"\n===== FOLD {fold}/{len(folds)} | sample_scope={args.sample_scope} | "
              f"samples={len(sample_indices)} =====", flush=True)

        fold_df = run_fold_optimized(
            args=args,
            fold=fold,
            targets=targets,
            geno=geno,
            n_groups=n_groups,
            selected_snp_count=args.selected_snp_count,
            sample_indices=sample_indices,
            device=device,
        )
        if len(fold_df):
            fold_path = args.out_dir / f"allele_perturbation_{args.sample_scope}.fold{fold:02d}.tsv.gz"
            fold_df.to_csv(fold_path, sep="\t", index=False, compression="gzip")
            per_fold_parts.append(fold_df)
            print(f"[FOLD {fold}] wrote {fold_path} ({len(fold_df)} rows)", flush=True)
        else:
            print(f"[FOLD {fold}] no evaluable targets", flush=True)

    total_elapsed = time.time() - t_total_start
    print(f"\n[INFO] All folds done in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)", flush=True)

    per_fold = pd.concat(per_fold_parts, ignore_index=True) if per_fold_parts else pd.DataFrame()
    per_fold_path = args.out_dir / f"allele_perturbation_{args.sample_scope}.per_fold.tsv.gz"
    per_fold.to_csv(per_fold_path, sep="\t", index=False, compression="gzip")

    agg, not_selected, evaluability = aggregate_fold_results(
        per_fold, targets, total_folds=args.total_folds_for_rate,
    )
    agg_path = args.out_dir / f"allele_perturbation_{args.sample_scope}.aggregated.tsv"
    not_path = args.out_dir / f"allele_perturbation_{args.sample_scope}.not_selected_report.tsv"
    eval_path = args.out_dir / f"allele_perturbation_{args.sample_scope}.evaluability_by_set.tsv"
    agg.to_csv(agg_path, sep="\t", index=False)
    not_selected.to_csv(not_path, sep="\t", index=False)
    evaluability.to_csv(eval_path, sep="\t", index=False)

    print(f"[DONE] per-fold: {per_fold_path}", flush=True)
    print(f"[DONE] aggregated: {agg_path} ({len(agg)} rows)", flush=True)
    print(f"[DONE] not-selected report: {not_path}", flush=True)
    print(f"[DONE] evaluability: {eval_path}", flush=True)


if __name__ == "__main__":
    main()
