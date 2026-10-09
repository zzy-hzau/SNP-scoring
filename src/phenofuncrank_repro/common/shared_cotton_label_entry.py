"""Identical source function; caller module supplies its recorded policy bindings."""
from __future__ import annotations
import argparse
import gzip
import json
import logging
import math
import os
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
try:
    import joblib
except Exception:  
    joblib = None
logger = logging.getLogger("noncoding_proxy_labels_preLM_strict")
BASE2INT = {"A": 0, "C": 1, "G": 2, "T": 3}
REQUIRED_BASE_COLS = [
    "snp_id", "Chr", "Pos", "Ref", "Alt",
    "region_type", "Y_Label", "chr_num", "Sg_code", "Ctx_code", "Alt_code",
]
OPTIONAL_KEEP_COLS = [
    "GeneID", "dist_to_TSS", "dist_to_gene",
    "promoter_core_flag", "promoter_proximal_flag", "promoter_distal_flag",
    "donor_offset", "acceptor_offset", "canonical_splice_core_flag",
    "homoeolog_expr_bias", "has_homolog",
]
OUTPUT_COLS = [
    "snp_id", "Chr", "Pos", "Ref", "Alt", "GeneID",
    "region_type", "region_subtype", "dist_to_TSS", "dist_to_gene",
    "promoter_core_flag", "promoter_proximal_flag", "promoter_distal_flag",
    "donor_offset", "acceptor_offset", "canonical_splice_core_flag",
    "homoeolog_expr_bias", "has_homolog",
    "chr_num", "Sg_code", "Alt_code", "Ctx_code",
    "Y_Label", "AN_obs", "callable_n", "callrate_hit",
    "P_mut_fold", "p_base_eff_fold", "p_base_eff_bin", "callable_n_bin",
    "match_group", "new_label", "fold", "split",
]
SPLIT_PARQUET_NAMES = ("train.parquet", "val.parquet", "test.parquet")
from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_profile

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", required=True, help="Pre-LM noncoding universe parquet with cheap annotations")
    ap.add_argument("--callrate", required=True, help="Parquet with Chr, Pos, callable_n, optional AN_obs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--wild-vcf", required=True)
    ap.add_argument("--sites-4d", required=True)
    ap.add_argument("--rate-cache", default=None)
    ap.add_argument("--force-internal-pmut", action="store_true")
    ap.add_argument("--n-chr", type=int, default=13)
    ap.add_argument("--folds", default="1-13")
    ap.add_argument("--subgenome-mode", default="auto", choices=["auto", "single"])
    ap.add_argument("--ne", type=float, default=None, help="Optional. In --ne-mode fixed, this is the fixed Ne. In train_loco mode, it is only used as the initial upper-bound seed if provided.")
    ap.add_argument("--ne-mode", default="train_loco", choices=["train_loco", "fixed"], help="train_loco calibrates one Ne per fold using train chromosomes only; fixed keeps old single-Ne behavior.")
    ap.add_argument("--ne-calibration-iter", type=int, default=80, help="Bisection iterations for train_loco Ne calibration.")
    ap.add_argument("--ne-calibration-max", type=float, default=1e12, help="Maximum upper bound for train_loco Ne calibration.")
    ap.add_argument("--max-callable-n", type=int, required=True, help="Usually number of wild samples, e.g. 161")
    ap.add_argument("--fallback-mu", type=float, default=4.56e-9)
    ap.add_argument("--pmut-pseudo-count", type=float, default=0.5)
    ap.add_argument("--min-callable-n", type=int, default=5)
    ap.add_argument("--opportunity-min-quantile", type=float, default=0.20)
    ap.add_argument("--opportunity-bins", type=int, default=10)
    ap.add_argument("--callable-n-bins", type=int, default=10)
    ap.add_argument("--label1-per-label0", type=float, default=1.0)
    ap.add_argument("--max-label0-per-group", type=int, default=20000)
    ap.add_argument("--min-group-label0", type=int, default=3)
    ap.add_argument("--min-group-label1", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=300000)
    ap.add_argument("--callrate-batch-size", type=int, default=2_000_000)
    ap.add_argument("--sample-rows-for-bins", type=int, default=2_000_000)
    ap.add_argument("--edge-sample-fraction", type=float, default=0.03)
    ap.add_argument("--trust-input-callrate", action="store_true", help="Use AN_obs/callable_n/callrate_hit already present in --input. Exact only if input was built from the same callrate table.")
    ap.add_argument("--progress-every", type=int, default=20, help="Log progress every N scanner batches.")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = open_parquet_dataset_only(args.input)
    ensure_columns(dataset, REQUIRED_BASE_COLS)

    call_idx = None if args.trust_input_callrate else load_callrate_index(args.callrate, args.callrate_batch_size)
    rate_cd, pmut_source = get_rate_tables(args)
    rate_tables = rate_cd["rate_tables"]
    genome_total = rate_cd["genome_total"]
    genome_by_chrnum = rate_cd["genome_by_chrnum"]
    mut_total = rate_cd["mut_total"]
    mut_by_chrnum = rate_cd["mut_by_chrnum"]
    _, build_excl_external = None, None  

    manifest_all = dict(
        input=args.input,
        out_dir=str(out_dir),
        ne=args.ne,
        ne_mode=args.ne_mode,
        ne_note='train_loco calibrates one fold-specific Ne from train chromosomes only; fixed keeps a single Ne.',
        max_callable_n=args.max_callable_n,
        pmut_source=pmut_source,
        p_base_eff_formula="1 - exp(-Ne * P_mut_fold * callable_n / max_callable_n)",
        label_semantics={
            "0": "Y_Label=1 wild observed noncoding SNP; tolerated proxy",
            "1": "Y_Label=0 high-opportunity callable wild-unobserved SNP; depletion/proxy-deleterious",
        },
        matching_group="region_type|region_subtype|Sg_code|Ctx_code|Alt_code|p_base_eff_bin|callable_n_bin",
        model_feature_note="Do not use Y_Label/P_mut_fold/p_base_eff_fold/callability/bins/match_group/fold/split as model features.",
        folds=[],
    )

    for fold in parse_folds(args.folds):
        test_chr = int(fold)
        val_chr = val_chr_for_test(test_chr, args.n_chr)
        train_chrs = [c for c in range(1, args.n_chr + 1) if c not in (test_chr, val_chr)]
        fold_dir = out_dir / f"fold{fold:02d}"
        fold_dir.mkdir(exist_ok=True)
        stale_manifest = fold_dir / "manifest.json"
        if stale_manifest.exists():
            logger.warning(f"[WRITEFIX] removing existing fold manifest before rewrite: {stale_manifest}")
            stale_manifest.unlink()
        logger.info("=" * 80)
        logger.info(f"FOLD {fold:02d}: train={train_chrs}, val={val_chr}, test={test_chr}")
        logger.info("=" * 80)

        train_rate = build_train_rate_from_cache_loco(rate_cd, exclude_chrs=[test_chr, val_chr], args=args)
        test_rate = rate_tables[test_chr] if test_chr in rate_tables else build_rate_table_excluding(
            genome_total, genome_by_chrnum, mut_total, mut_by_chrnum,
            exclude_chrs=[test_chr], min_mu=args.fallback_mu, pseudo_count=args.pmut_pseudo_count)
        train_arr = build_rate_array(train_rate, args.fallback_mu)
        test_arr = build_rate_array(test_rate, args.fallback_mu)

        if args.ne_mode == "train_loco":
            fold_ne, ne_calibration = calibrate_ne_for_train_chrs(dataset, train_chrs, args, call_idx, train_arr)
        else:
            if args.ne is None:
                raise RuntimeError("--ne-mode fixed requires --ne")
            fold_ne = float(args.ne)
            ne_calibration = {
                "ne": fold_ne,
                "mode": "fixed",
                "note": "Fixed single Ne; not train-LOCO calibrated.",
            }

        p_edges, cn_edges, p_thr = collect_train_edges(dataset, train_chrs, args, call_idx, train_arr, fold_ne)
        split_defs = [
            ("train", train_chrs, train_arr, fold_dir / "train.parquet"),
            ("val", [val_chr], train_arr, fold_dir / "val.parquet"),
            ("test", [test_chr], test_arr, fold_dir / "test.parquet"),
        ]

        count_result = count_groups_for_all_splits(
            dataset, split_defs, args, call_idx, fold_ne, p_edges, cn_edges, p_thr
        )

        probs_by_split = {}
        split_manifest = {}
        for split, chrs, arr, out_path in split_defs:
            c0, c1, cand0, cand1, scanned = count_result[split]
            probs0, probs1, plan = decide_sampling_probs(c0, c1, args)

            plan_path = fold_dir / f"{split}.sampling_plan.tsv"
            import os
            os.makedirs(os.path.dirname(str(plan_path)), exist_ok=True)
            plan.to_csv(plan_path, sep="\t", index=False)

            probs_by_split[split] = (probs0, probs1)
            split_manifest[split] = dict(
                scanned_rows=int(scanned),
                candidate0=int(cand0),
                candidate1=int(cand1),
                groups0=len(c0),
                groups1=len(c1),
                groups_overlap=len(set(c0) & set(c1)),
                kept_groups=int(len(plan)),
                planned0=int(plan["target0"].sum()) if len(plan) else 0,
                planned1=int(plan["target1"].sum()) if len(plan) else 0,
                sampling_plan=str(plan_path),
            )
            logger.info(f"[fold {fold:02d} {split}] {split_manifest[split]}")

        write_stats = {}
        for split, chrs, arr, out_path in split_defs:
            out_path = Path(out_path)
            if out_path.exists():
                logger.warning(f"[WRITEFIX] removing existing output before rewrite: {out_path}")
                out_path.unlink()

            probs0, probs1 = probs_by_split[split]

            write_stats[split] = write_selected_split(
                dataset=dataset,
                out_path=out_path,
                chr_keep=chrs,
                split=split,
                fold=fold,
                args=args,
                call_idx=call_idx,
                rate_arr=arr,
                ne=fold_ne,
                p_edges=p_edges,
                cn_edges=cn_edges,
                p_thr=p_thr,
                probs0=probs0,
                probs1=probs1,
            )

        for split in ["train", "val", "test"]:
            logger.info(f"[fold {fold:02d} {split} write] {write_stats[split]}")

        fold_manifest = dict(fold=int(fold), train_chrs=train_chrs, val_chr=int(val_chr), test_chr=int(test_chr),
                             ne=float(fold_ne), ne_mode=args.ne_mode, ne_calibration=ne_calibration,
                             p_base_eff_threshold=float(p_thr), p_base_eff_edges=p_edges.tolist(), callable_n_edges=cn_edges.tolist(),
                             split_count_manifest=split_manifest, write_stats=write_stats)
        atomic_write_json(fold_dir / "manifest.json", fold_manifest)
        manifest_all["folds"] = collect_completed_fold_manifests(out_dir)
        atomic_write_json(out_dir / "manifest_all.partial.json", manifest_all)

    manifest_all["folds"] = collect_completed_fold_manifests(out_dir)
    target_path, target_n = export_lm_targets(out_dir)
    manifest_all["lm_targets"] = target_path
    manifest_all["lm_target_n"] = target_n
    atomic_write_json(out_dir / "manifest_all.json", manifest_all)
    logger.info("DONE")
