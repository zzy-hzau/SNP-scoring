#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import gc
import json
import math
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from phenofuncrank_repro.models.dpcformer.training_utils import set_seed, safe_corr, available_memory_gb, enforce_memory_reserve, first_present, resolve_path, select_top, make_loader, evaluate
from phenofuncrank_repro.models.dpcformer.mic import score_discrete_features_batched

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from phenofuncrank_repro.models.dpcformer.network import PhenotypePredictor_1v


BASE_TO_VEC = {
    "A": np.array([1, 0, 0, 0], dtype=np.float32),
    "C": np.array([0, 1, 0, 0], dtype=np.float32),
    "G": np.array([0, 0, 1, 0], dtype=np.float32),
    "T": np.array([0, 0, 0, 1], dtype=np.float32),
}


def parse_missing_phenotype_values(raw):
    if not raw:
        return []
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def load_inputs(args):
    inputs_dir = Path(args.inputs_dir).resolve()
    meta_path = inputs_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text())

    memmap_path = resolve_path(
        inputs_dir,
        first_present(meta, ["genotype_memmap", "memmap", "genotype_path"]),
        ["cultivar_genotype_codes.variant_by_sample.uint8.memmap"],
    )
    snp_map_path = resolve_path(
        inputs_dir,
        first_present(meta, ["snp_map", "snp_map_path"]),
        ["cultivar.snp_map.tsv.gz", "cultivar.snp_map.tsv"],
    )
    phenotype_path = resolve_path(
        inputs_dir,
        first_present(meta, ["phenotype_table", "phenotypes", "phenotype_path"]),
        ["matched_phenotypes.tsv"],
    )

    snp_map = pd.read_csv(snp_map_path, sep="\t", compression="infer")
    required = {"variant_index", "Chr", "group_id", "Pos", "Ref", "Alt", "snp_id"}
    missing = required - set(snp_map.columns)
    if missing:
        raise RuntimeError(f"SNP map missing columns: {sorted(missing)}")

    snp_map = snp_map.copy()
    snp_map["Chr"] = snp_map["Chr"].astype(str).str.strip()
    snp_map["Pos"] = pd.to_numeric(snp_map["Pos"], errors="raise").astype(np.int64)
    snp_map["variant_index"] = pd.to_numeric(
        snp_map["variant_index"], errors="raise"
    ).astype(np.int64)
    snp_map["group_id"] = pd.to_numeric(
        snp_map["group_id"], errors="raise"
    ).astype(np.int64)
    snp_map = snp_map.sort_values("variant_index").reset_index(drop=True)

    n_snps = int(first_present(meta, ["n_snps", "n_variants", "num_snps"]) or len(snp_map))
    if n_snps != len(snp_map):
        raise RuntimeError(f"metadata n_snps={n_snps} but SNP map rows={len(snp_map)}")
    if not np.array_equal(snp_map["variant_index"].to_numpy(), np.arange(n_snps)):
        raise RuntimeError("variant_index must be exactly 0..n_snps-1")

    pheno = pd.read_csv(phenotype_path, sep="\t")
    if args.pheno_col not in pheno.columns:
        raise RuntimeError(
            f"Phenotype column {args.pheno_col!r} not found; available={pheno.columns.tolist()}"
        )
    y_all = pd.to_numeric(pheno[args.pheno_col], errors="coerce").to_numpy(dtype=float)
    if getattr(args, "environment_column", None):
        if not getattr(args, "environment", None):
            raise ValueError("--environment must be specified with --environment-column")
        if args.environment_column not in pheno.columns:
            raise RuntimeError(f"Environment column {args.environment_column!r} not found")
        env_mask = pheno[args.environment_column].astype(str).eq(str(args.environment)).to_numpy()
        y_all = y_all.copy()
        y_all[~env_mask] = np.nan
        print(f"[INFO] Environment filter {args.environment_column}={args.environment}: keeping {int(env_mask.sum())} samples out of {len(pheno)}")
    missing_values = parse_missing_phenotype_values(args.missing_phenotype_values)
    if missing_values:
        missing_mask = np.zeros(y_all.shape, dtype=bool)
        for value in missing_values:
            missing_mask |= y_all == value
        if missing_mask.any():
            print(
                f"[INFO] treating phenotype values {missing_values} as missing: "
                f"n={int(missing_mask.sum())}"
            )
            y_all = y_all.copy()
            y_all[missing_mask] = np.nan
    n_samples = int(first_present(meta, ["n_samples", "num_samples"]) or len(pheno))
    if len(pheno) != n_samples:
        raise RuntimeError(f"metadata n_samples={n_samples} but phenotype rows={len(pheno)}")

    expected_bytes = n_snps * n_samples * np.dtype(np.uint8).itemsize
    actual_bytes = memmap_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"memmap byte-size mismatch: expected={expected_bytes}, actual={actual_bytes}"
        )
    geno = np.memmap(memmap_path, mode="r", dtype=np.uint8, shape=(n_snps, n_samples))
    observed_codes = np.unique(np.asarray(geno[: min(n_snps, 10_000), :]))
    invalid_codes = sorted(set(observed_codes.tolist()) - {0, 1, 2, 3})
    if invalid_codes:
        raise RuntimeError(
            f"Unexpected genotype codes in {memmap_path}: {invalid_codes}; "
            "expected 0=missing, 1=REF/REF, 2=REF/ALT, 3=ALT/ALT"
        )

    valid_sample_cols = np.flatnonzero(np.isfinite(y_all)).astype(np.int64)
    y_valid = y_all[valid_sample_cols]
    if len(valid_sample_cols) < 100:
        raise RuntimeError(f"Too few valid phenotype samples: {len(valid_sample_cols)}")

    return meta, geno, snp_map, pheno, y_all, valid_sample_cols, y_valid


def score_discrete_features(geno, variant_indices, sample_cols, y, seed, selector):
    return score_discrete_features_batched(geno, variant_indices, sample_cols, y, seed, selector)


def select_diploid_group_from_chr_df(
    geno,
    chr_df,
    train_sample_cols,
    y_train_raw,
    final_k,
    seed,
    selector,
    trait,
    fold,
    group_id,
):
    if chr_df.empty:
        raise RuntimeError(f"No SNPs found for group_id={group_id}")
    label = ",".join(chr_df["Chr"].astype(str).drop_duplicates().head(3).tolist())
    scores = score_discrete_features(
        geno,
        chr_df["variant_index"].to_numpy(),
        train_sample_cols,
        y_train_raw,
        seed + group_id * 100,
        selector,
    )
    selected = select_top(chr_df, scores, final_k, "mic_score")
    if len(selected) != final_k:
        raise RuntimeError(
            f"group {group_id} ({label}): selected={len(selected)} != final_k={final_k}"
        )
    selected["selection_stage"] = "diploid_chromosome"
    selected["fold"] = fold
    selected["trait"] = trait
    selected["selected_group"] = int(group_id)
    selected = selected.sort_values(
        ["Chr", "Pos", "Ref", "Alt", "snp_id"], kind="mergesort"
    ).reset_index(drop=True)
    selected["selected_rank_physical"] = np.arange(1, len(selected) + 1)
    selected["selected_rank"] = selected["selected_rank_physical"]
    selected["original_snp_index"] = selected["variant_index"].astype(np.int64)
    message = (
        f"[SELECT] trait={trait} fold={fold:02d} group={group_id} "
        f"chr={label} available={len(chr_df)} selected={len(selected)}"
    )
    return group_id, selected, message


def select_diploid_group_from_memmap(
    memmap_path,
    geno_shape,
    chr_df,
    train_sample_cols,
    y_train_raw,
    final_k,
    seed,
    selector,
    trait,
    fold,
    group_id,
):
    geno = np.memmap(memmap_path, mode="r", dtype=np.uint8, shape=tuple(geno_shape))
    return select_diploid_group_from_chr_df(
        geno=geno,
        chr_df=chr_df,
        train_sample_cols=train_sample_cols,
        y_train_raw=y_train_raw,
        final_k=final_k,
        seed=seed,
        selector=selector,
        trait=trait,
        fold=fold,
        group_id=group_id,
    )


def diploid_select_for_fold(
    geno,
    snp_map,
    train_sample_cols,
    y_train_raw,
    final_k,
    seed,
    trait,
    fold,
    fold_dir,
    args_selector,
    selection_workers,
):
    group_outputs = []
    group_ids = sorted(pd.to_numeric(snp_map["group_id"], errors="raise").astype(int).unique())
    selection_workers = max(1, int(selection_workers))

    by_group = {}
    if selection_workers == 1:
        for group_id in group_ids:
            chr_df = snp_map[snp_map["group_id"].eq(group_id)].copy()
            key_group, selected, message = select_diploid_group_from_chr_df(
                geno=geno,
                chr_df=chr_df,
                train_sample_cols=train_sample_cols,
                y_train_raw=y_train_raw,
                final_k=final_k,
                seed=seed,
                selector=args_selector,
                trait=trait,
                fold=fold,
                group_id=group_id,
            )
            by_group[key_group] = selected
            print(message)
    else:
        memmap_path = getattr(geno, "filename", None)
        use_process_workers = bool(memmap_path)
        executor_cls = ProcessPoolExecutor if use_process_workers else ThreadPoolExecutor
        backend = "process" if use_process_workers else "thread"
        print(
            f"[SELECT] trait={trait} fold={fold:02d} "
            f"parallel_group_workers={selection_workers} backend={backend}"
        )
        with executor_cls(max_workers=selection_workers) as executor:
            futures = []
            tasks = []
            for group_id in group_ids:
                chr_df = snp_map[snp_map["group_id"].eq(group_id)].copy()
                tasks.append((len(chr_df), group_id, chr_df))
            tasks.sort(key=lambda x: (-x[0], x[1]))
            for _, group_id, chr_df in tasks:
                if use_process_workers:
                    futures.append(
                        executor.submit(
                            select_diploid_group_from_memmap,
                            memmap_path,
                            geno.shape,
                            chr_df,
                            train_sample_cols,
                            y_train_raw,
                            final_k,
                            seed,
                            args_selector,
                            trait,
                            fold,
                            group_id,
                        )
                    )
                else:
                    futures.append(
                        executor.submit(
                            select_diploid_group_from_chr_df,
                            geno=geno,
                            chr_df=chr_df,
                            train_sample_cols=train_sample_cols,
                            y_train_raw=y_train_raw,
                            final_k=final_k,
                            seed=seed,
                            selector=args_selector,
                            trait=trait,
                            fold=fold,
                            group_id=group_id,
                        )
                    )
            for future in as_completed(futures):
                key_group, selected, message = future.result()
                by_group[key_group] = selected
                print(message)

    for group_id in group_ids:
        group_outputs.append(by_group[group_id])
    selected_map = pd.concat(group_outputs, ignore_index=True, sort=False)
    stage1_path = fold_dir / f"{trait}.fold_{fold:02d}.stage1_candidates.tsv.gz"
    selected_path = fold_dir / f"{trait}.fold_{fold:02d}.selected_snp_map.tsv"
    selected_map.to_csv(stage1_path, sep="\t", index=False, compression="gzip")
    selected_map.to_csv(selected_path, sep="\t", index=False)
    return selected_map, selected_path, stage1_path


def codes_to_8d(selected_codes, selected_map):
    n_selected, n_samples = selected_codes.shape
    out = np.zeros((n_samples, n_selected, 8), dtype=np.float32)

    ref_list = [BASE_TO_VEC.get(str(row.Ref).upper(), np.zeros(4, dtype=np.float32)) for row in selected_map.itertuples(index=False)]
    alt_list = [BASE_TO_VEC.get(str(row.Alt).upper(), np.zeros(4, dtype=np.float32)) for row in selected_map.itertuples(index=False)]

    ref_vecs = np.stack(ref_list, axis=0)[np.newaxis, :, :] 
    alt_vecs = np.stack(alt_list, axis=0)[np.newaxis, :, :] 

    codes = selected_codes.T 
    m_ref = (codes == 1)[:, :, np.newaxis]
    m_het = (codes == 2)[:, :, np.newaxis]
    m_alt = (codes == 3)[:, :, np.newaxis]

    out[:, :, 0:4] = np.where(m_ref | m_het, ref_vecs, 0.0) + np.where(m_alt, alt_vecs, 0.0)
    out[:, :, 4:8] = np.where(m_ref, ref_vecs, 0.0) + np.where(m_het | m_alt, alt_vecs, 0.0)
    return out


def build_fold_tensor(geno, selected_map, valid_sample_cols, final_k, n_groups):
    groups = []
    for group_id in range(1, n_groups + 1):
        sub = selected_map[selected_map["selected_group"].eq(group_id)].copy()
        sub = sub.sort_values("selected_rank_physical")
        if len(sub) != final_k:
            raise RuntimeError(f"group {group_id}: expected {final_k} SNPs, found {len(sub)}")
        codes = np.asarray(
            geno[sub["variant_index"].to_numpy(dtype=np.int64), :][:, valid_sample_cols],
            dtype=np.uint8,
        )
        groups.append(codes_to_8d(codes, sub))
    tensor = np.stack(groups, axis=1)
    return torch.from_numpy(tensor)


def train_one_fold(args, fold, split, geno, snp_map, valid_sample_cols, y_valid, device, outdir, *, evaluate_test=True):
    fold_seed = args.seed + fold * 1009
    set_seed(fold_seed)
    enforce_memory_reserve(args.min_available_mem_gb, f"{args.trait} fold {fold}")
    fold_dir = outdir / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=False)

    train_local = np.asarray(split[0], dtype=np.int64)
    val_local = np.asarray(split[1], dtype=np.int64)
    test_local = np.asarray(split[2], dtype=np.int64)
    if set(train_local) & set(val_local) or set(train_local) & set(test_local) or set(val_local) & set(test_local):
        raise RuntimeError("Split overlap detected")

    scaler = StandardScaler()
    scaler.fit(y_valid[train_local].reshape(-1, 1))
    y_scaled = scaler.transform(y_valid.reshape(-1, 1)).astype(np.float32).reshape(-1)
    np.savez_compressed(
        fold_dir / f"{args.trait}.fold_{fold:02d}.phenotype_scaler.npz",
        mean_=scaler.mean_,
        scale_=scaler.scale_,
        var_=scaler.var_,
        n_samples_seen_=scaler.n_samples_seen_,
    )
    np.savez_compressed(
        fold_dir / f"{args.trait}.fold_{fold:02d}.splits.npz",
        train_local=train_local,
        val_local=val_local,
        test_local=test_local,
        train_memmap_cols=valid_sample_cols[train_local],
        val_memmap_cols=valid_sample_cols[val_local],
        test_memmap_cols=valid_sample_cols[test_local],
    )

    n_groups = int(pd.to_numeric(snp_map["group_id"], errors="raise").max())
    if getattr(args, "preselected_snp_map", None):
        selected_map = pd.read_csv(args.preselected_snp_map, sep="\t")
        selected_map["selected_group"] = selected_map["group_id"]
        selected_path = fold_dir / f"{args.trait}.fold_{fold:02d}.selected_snp_map.tsv"
        selected_map.to_csv(selected_path, sep="\t", index=False)
        stage1_path = None
    else:
        selected_map, selected_path, stage1_path = diploid_select_for_fold(
            geno=geno,
            snp_map=snp_map,
            train_sample_cols=valid_sample_cols[train_local],
            y_train_raw=y_valid[train_local],
            final_k=args.selected_snp_count,
            seed=fold_seed,
            trait=args.trait,
            fold=fold,
            fold_dir=fold_dir,
            args_selector=args.selector,
            selection_workers=args.selection_workers,
        )

    x = build_fold_tensor(
        geno, selected_map, valid_sample_cols, args.selected_snp_count, n_groups
    )
    y = torch.from_numpy(y_scaled.reshape(-1, 1))
    train_loader = make_loader(x, y, train_local, args.batch_size, True, fold_seed)
    val_loader = make_loader(x, y, val_local, args.batch_size, False, fold_seed)
    test_loader = make_loader(x, y, test_local, args.batch_size, False, fold_seed)

    model = PhenotypePredictor_1v(n_groups, [args.selected_snp_count] * n_groups, 8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=args.lr_patience,
        threshold=0.0,
        threshold_mode="rel",
        cooldown=0,
        min_lr=args.min_learning_rate,
    )

    model_path = outdir / "saved_models" / f"{args.trait}.best_model_fold_{fold}.pth"
    meta_path = fold_dir / f"{args.trait}.fold_{fold:02d}.model_metadata.json"
    log_path = fold_dir / f"{args.trait}.fold_{fold:02d}.training_log.tsv"
    best_val_loss = math.inf
    best_epoch = 0
    no_improve = 0

    with log_path.open("w") as log:
        log.write("epoch\ttrain_loss\tval_loss\tval_pearson\tval_spearman\tlearning_rate\n")
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            n_train = 0
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * xb.size(0)
                n_train += xb.size(0)
            train_loss = total_loss / max(n_train, 1)
            val = evaluate(model, val_loader, criterion, device)
            scheduler.step(val["loss"])
            lr = float(optimizer.param_groups[0]["lr"])
            print(
                f"epoch={epoch} train={train_loss:.4f} val={val['loss']:.4f} "
                f"pearson={val['pearson']:.4f} lr={lr:.3e}"
            )
            log.write(
                f"{epoch}\t{train_loss:.8g}\t{val['loss']:.8g}\t{val['pearson']:.8g}\t"
                f"{val['spearman']:.8g}\t{lr:.8g}\n"
            )
            log.flush()

            if val["loss"] < best_val_loss - args.min_delta:
                best_val_loss = float(val["loss"])
                best_epoch = epoch
                no_improve = 0
                torch.save(model.state_dict(), model_path)
                meta_path.write_text(
                    json.dumps(
                        {
                            "trait": args.trait,
                            "fold": fold,
                            "best_epoch": best_epoch,
                            "best_val_loss": best_val_loss,
                            "selection_scope": "fold_training_only",
                            "selector": args.selector,
                            "n_groups": n_groups,
                            "final_k": args.selected_snp_count,
                            "selected_map": str(selected_path.relative_to(outdir)),
                            "stage1_candidates": str(stage1_path.relative_to(outdir)) if stage1_path else None,
                        },
                        indent=2,
                    )
                )
            else:
                no_improve += 1
                print(f"EarlyStopping counter: {no_improve} out of {args.early_stopping_patience}")
                if no_improve >= args.early_stopping_patience:
                    print(f"[EARLY STOP] fold={fold} epoch={epoch}")
                    break

    if evaluate_test:
        model.load_state_dict(torch.load(model_path, map_location=device))
        test = evaluate(model, test_loader, criterion, device)
        pred_raw = scaler.inverse_transform(test["preds"].reshape(-1, 1)).reshape(-1)
        obs_raw = y_valid[test_local]
        raw_pearson = safe_corr(pearsonr, pred_raw, obs_raw)
        raw_spearman = safe_corr(spearmanr, pred_raw, obs_raw)
        pd.DataFrame(
            {
                "valid_local_index": test_local,
                "memmap_sample_col": valid_sample_cols[test_local],
                "observed_raw": obs_raw,
                "predicted_raw": pred_raw,
                "observed_scaled": test["obs"],
                "predicted_scaled": test["preds"],
            }
        ).to_csv(fold_dir / f"{args.trait}.fold_{fold:02d}.test_predictions.tsv", sep="\t", index=False)

        result = {
            "fold": fold,
            "n_train": len(train_local),
            "n_val": len(val_local),
            "n_test": len(test_local),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_loss": float(test["loss"]),
            "test_pearson": raw_pearson,
            "test_spearman": raw_spearman,
            "model_path": str(model_path),
            "selected_map_path": str(selected_path),
            "fold_dir": str(fold_dir),
        }
        print(
            f"[TEST] fold={fold} loss={result['test_loss']:.4f} "
            f"pearson={result['test_pearson']:.4f} spearman={result['test_spearman']:.4f}"
        )

    else:
        result = {
            "fold": fold,
            "n_train": len(train_local),
            "n_val": len(val_local),
            "n_test": len(test_local),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "model_path": str(model_path),
            "selected_map_path": str(selected_path),
            "fold_dir": str(fold_dir),
        }

    del x, y, train_loader, val_loader, test_loader, model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def evaluate_retained_fold(args, result, geno, snp_map, valid_sample_cols, y_valid, device):
    fold = int(result["fold"])
    fold_dir = Path(result["fold_dir"])
    selected_map = pd.read_csv(result["selected_map_path"], sep="\t")
    saved_scaler = np.load(fold_dir / f"{args.trait}.fold_{fold:02d}.phenotype_scaler.npz")
    scaler = StandardScaler()
    scaler.mean_ = saved_scaler["mean_"]
    scaler.scale_ = saved_scaler["scale_"]
    scaler.var_ = saved_scaler["var_"]
    scaler.n_samples_seen_ = saved_scaler["n_samples_seen_"]
    scaler.n_features_in_ = 1
    test_local = np.load(fold_dir / f"{args.trait}.fold_{fold:02d}.splits.npz")["test_local"]
    n_groups = int(pd.to_numeric(snp_map["group_id"], errors="raise").max())
    x = build_fold_tensor(geno, selected_map, valid_sample_cols, args.selected_snp_count, n_groups)
    y_scaled = scaler.transform(y_valid.reshape(-1, 1)).astype(np.float32)
    y = torch.from_numpy(y_scaled)
    loader = make_loader(x, y, test_local, args.batch_size, False, args.seed + fold * 1009)
    model = PhenotypePredictor_1v(n_groups, [args.selected_snp_count] * n_groups, 8).to(device)
    model.load_state_dict(torch.load(result["model_path"], map_location=device))
    test = evaluate(model, loader, nn.MSELoss(), device)
    pred_raw = scaler.inverse_transform(test["preds"].reshape(-1, 1)).reshape(-1)
    obs_raw = y_valid[test_local]
    pd.DataFrame({
        "valid_local_index": test_local,
        "memmap_sample_col": valid_sample_cols[test_local],
        "observed_raw": obs_raw,
        "predicted_raw": pred_raw,
        "observed_scaled": test["obs"],
        "predicted_scaled": test["preds"],
    }).to_csv(fold_dir / f"{args.trait}.fold_{fold:02d}.test_predictions.tsv", sep="\t", index=False)
    result = dict(result)
    result.update(
        test_loss=float(test["loss"]),
        test_pearson=safe_corr(pearsonr, pred_raw, obs_raw),
        test_spearman=safe_corr(spearmanr, pred_raw, obs_raw),
    )
    print(f"[TEST] fold={fold} loss={result['test_loss']:.4f} pearson={result['test_pearson']:.4f} spearman={result['test_spearman']:.4f}")
    del x, y, loader, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def train_one_trait(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    outdir = Path(args.outdir).resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "saved_models").mkdir()

    meta, geno, snp_map, pheno, y_all, valid_sample_cols, y_valid = load_inputs(args)
    n_valid = len(y_valid)
    all_local = np.arange(n_valid, dtype=np.int64)
    train_val_local, test_local = train_test_split(
        all_local,
        test_size=args.test_ratio,
        random_state=args.seed,
        shuffle=True,
    )
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = []
    for train_rel, val_rel in splitter.split(train_val_local):
        splits.append((train_val_local[train_rel], train_val_local[val_rel], test_local))

    np.savez_compressed(
        outdir / f"{args.trait}.fixed_test_and_cv_protocol.npz",
        valid_memmap_cols=valid_sample_cols,
        train_val_local=train_val_local,
        fixed_test_local=test_local,
        fixed_test_memmap_cols=valid_sample_cols[test_local],
    )
    protocol = {
        "paper_interpretation": "fixed 90/10 train-validation/test; 10-fold CV inside train-validation",
        "test_ratio": args.test_ratio,
        "validation_fraction_within_train_val": 1.0 / args.folds,
        "effective_validation_split": "KFold inside fixed train-validation subset",
        "folds": args.folds,
        "feature_selection_scope": "fold training only",
        "feature_selection": "diploid per-chromosome MIC selection",
        "selector_backend": args.selector,
        "selection_workers": args.selection_workers,
        "missing_phenotype_values": parse_missing_phenotype_values(args.missing_phenotype_values),
        "min_available_mem_gb": args.min_available_mem_gb,
        "physical_sort": True,
        "genotype_code_mapping": {
            "0": "missing_or_unusable",
            "1": "REF/REF",
            "2": "REF/ALT",
            "3": "ALT/ALT",
        },
        "missing_genotype_tensor": "all_zero_8d_vector",
        "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.1, "patience": args.lr_patience},
        "optimizer": {"name": "Adam", "learning_rate": args.learning_rate},
        "early_stopping_patience": args.early_stopping_patience,
        "model_selection": "validation loss only",
        "test_used_for_model_selection": False,
    }
    (outdir / f"{args.trait}.training_protocol.json").write_text(json.dumps(protocol, indent=2))

    print(f"[INFO] device={device}")
    print(f"[INFO] trait={args.trait}; valid samples={n_valid}")
    print(f"[INFO] n_groups={int(pd.to_numeric(snp_map['group_id'], errors='raise').max())}")
    print(f"[INFO] train_val={len(train_val_local)} fixed_test={len(test_local)}")

    results = []
    for fold, split in enumerate(splits, start=1):
        print(f"\n===== TRAIT {args.trait} FOLD {fold}/{args.folds} =====")
        results.append(
            train_one_fold(
                args, fold, split, geno, snp_map, valid_sample_cols, y_valid, device, outdir,
                evaluate_test=False,
            )
        )

    metrics = pd.DataFrame(results)
    canonical_row = metrics.sort_values(["best_val_loss", "fold"]).iloc[0]
    canonical_fold = int(canonical_row["fold"])
    shutil.copy2(
        canonical_row["model_path"],
        outdir / f"{args.trait}.canonical_model_by_validation_loss.pth",
    )
    shutil.copy2(
        canonical_row["selected_map_path"],
        outdir / f"{args.trait}.canonical_selected_snp_map_by_validation_loss.tsv",
    )

    (outdir / f"{args.trait}.model_selection.json").write_text(json.dumps({
        "criterion": "minimum validation loss",
        "representative_fold": canonical_fold,
        "validation_loss": float(canonical_row["best_val_loss"]),
    }, indent=2))
    results = [evaluate_retained_fold(args, row, geno, snp_map, valid_sample_cols, y_valid, device) for row in results]
    metrics = pd.DataFrame(results)
    metrics.to_csv(outdir / f"{args.trait}.fold_metrics.tsv", sep="\t", index=False)

    summary_path = outdir / f"{args.trait}.cross_validation_summary.txt"
    with summary_path.open("w") as f:
        f.write(f"trait\t{args.trait}\n")
        f.write(f"pheno_col\t{args.pheno_col}\n")
        f.write(f"cv_protocol\tfixed_test_ratio_{args.test_ratio:g}_plus_{args.folds}fold_train_validation\n")
        f.write("feature_selection_scope\tfold_training_only\n")
        f.write("diploid_feature_selection\tper_chromosome_mic_selection\n")
        f.write(f"selector_backend\t{args.selector}\n")
        f.write(f"selection_workers\t{args.selection_workers}\n")
        f.write(f"missing_phenotype_values\t{args.missing_phenotype_values or 'none'}\n")
        f.write(f"min_available_mem_gb\t{args.min_available_mem_gb:.1f}\n")
        f.write(f"selected_snp_count\t{args.selected_snp_count}\n")
        f.write(f"folds\t{args.folds}\n")
        f.write(f"fixed_test_n\t{len(test_local)}\n")
        for col in ["test_loss", "test_pearson", "test_spearman"]:
            f.write(f"{col}_mean\t{metrics[col].mean():.6f}\n")
            f.write(f"{col}_sd\t{metrics[col].std(ddof=0):.6f}\n")
        f.write(f"canonical_fold_by_validation_loss\t{canonical_fold}\n")
        f.write(f"canonical_best_val_loss\t{float(canonical_row['best_val_loss']):.8g}\n")
        f.write("test_based_fold_selection\tfalse\n")

    print("\n===== SUMMARY =====")
    print(metrics[["fold", "best_val_loss", "test_pearson", "test_spearman"]].to_string(index=False))
    print(f"[CANONICAL BY VALIDATION] fold={canonical_fold}")
    print(f"[DONE] {summary_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--pheno-col", required=True)
    ap.add_argument(
        "--missing-phenotype-values",
        default="",
        help="Comma-separated numeric phenotype sentinels to treat as missing, for example -9.",
    )
    ap.add_argument("--selected-snp-count", type=int, default=1000)
    ap.add_argument(
        "--selector",
        choices=["minepy_mic", "public_mi"],
        default="minepy_mic",
        help="minepy_mic follows the paper's explicit MIC definition; public_mi matches the released code.",
    )
    ap.add_argument(
        "--first-stage-k",
        type=int,
        default=1000,
        help="Deprecated compatibility flag; diploid selection uses --selected-snp-count per chromosome.",
    )
    ap.add_argument(
        "--selection-workers",
        type=int,
        default=2,
        help=(
            "Parallel chromosome workers for stage-1 MIC selection. "
            "This does not change the MIC method, fold split, selected SNP count, "
            "or downstream model; it only runs independent chromosome scans concurrently."
        ),
    )
    ap.add_argument(
        "--min-available-mem-gb",
        type=float,
        default=80.0,
        help="Abort before major training work if MemAvailable is below this GiB reserve. Use 0 to disable.",
    )
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--test-ratio", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--min-learning-rate", type=float, default=1e-8)
    ap.add_argument("--lr-patience", type=int, default=10)
    ap.add_argument("--early-stopping-patience", type=int, default=20)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--environment-column", default="")
    ap.add_argument("--environment", default="")

    
    ap.set_defaults(preselected_snp_map="")
    return ap.parse_args()


if __name__ == "__main__":
    train_one_trait(parse_args())
