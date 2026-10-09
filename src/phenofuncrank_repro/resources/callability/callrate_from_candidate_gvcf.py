#!/usr/bin/env python3


from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import re
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

try:
    import pysam
except Exception as e:
    raise RuntimeError("需要安装 pysam") from e


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def norm_key(x: str) -> str:
    s = str(x).strip()
    s = re.sub(r"^(chr|Chr|CHR)", "", s)
    if s.isdigit():
        s = str(int(s))
    return s.lower()


def gt_allele_count(gt_tuple):
    if gt_tuple is None:
        return 0
    vals = [x for x in gt_tuple if x is not None and x >= 0]
    return len(vals)


def is_reference_block_record(rec, block_mode="reference-blocks") -> bool:
    if block_mode == "legacy-records":

        if "END" not in rec.info:
            return False
        alts = rec.alts or ()
        return bool(alts) and not any(
            not (str(a).startswith("<") and str(a).endswith(">")) for a in alts
        )

    
    return bool(rec.alts) and all(a in {"<NON_REF>", "<*>"} for a in rec.alts)


def read_gvcf_list(path: str) -> List[str]:
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out


def atomic_write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        json.dump(obj, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def atomic_write_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, path)


def load_position_cache(cache_dir: Path, prefix: str):
    meta_path = cache_dir / f"{prefix}.meta.json"
    npz_path = cache_dir / f"{prefix}.pos_by_chr.npz"

    if not meta_path.exists():
        raise FileNotFoundError(f"cache meta not found: {meta_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"cache pos npz not found: {npz_path}")

    meta = json.load(open(meta_path))
    chr_order = list(meta["chr_order"])

    pos_by_chr = {}
    with np.load(npz_path, allow_pickle=False) as npz:
        for chrom in chr_order:
            if chrom not in npz:
                raise RuntimeError(f"cache npz missing chrom array: {chrom}")
            arr = np.asarray(npz[chrom], dtype=np.int64)
            if arr.size > 1 and np.any(arr[1:] < arr[:-1]):
                arr = np.sort(arr)
            pos_by_chr[chrom] = arr

    return meta, pos_by_chr


def init_count_arrays(pos_by_chr: Dict[str, np.ndarray]):
    an_obs_by_chr = {}
    callable_n_by_chr = {}
    for chrom, pos_arr in pos_by_chr.items():
        n = int(pos_arr.size)
        an_obs_by_chr[chrom] = np.zeros(n, dtype=np.int32)
        callable_n_by_chr[chrom] = np.zeros(n, dtype=np.int32)
    return an_obs_by_chr, callable_n_by_chr


def resolve_header_contigs(vf, wanted_chroms: List[str]):
    raw_names = list(vf.header.contigs)
    exact = {x: x for x in raw_names}
    norm_map = {}
    for r in raw_names:
        k = norm_key(r)
        if k not in norm_map:
            norm_map[k] = r

    out = {}
    for c in wanted_chroms:
        if c in exact:
            out[c] = c
        else:
            out[c] = norm_map.get(norm_key(c))
    return out


def scan_one_gvcf(
    gvcf_path: str,
    pos_by_chr: Dict[str, np.ndarray],
    an_obs_by_chr: Dict[str, np.ndarray],
    callable_n_by_chr: Dict[str, np.ndarray],
    block_mode: str = "reference-blocks",
):
    vf = pysam.VariantFile(gvcf_path)

    samples = list(vf.header.samples)
    if len(samples) != 1:
        raise ValueError(f"{gvcf_path} 不是单样本 gVCF，样本数={len(samples)}")
    sample = samples[0]

    contig_map = resolve_header_contigs(vf, list(pos_by_chr.keys()))

    missing = [c for c, raw in contig_map.items() if raw is None]
    if missing:
        log(f"[WARN] {Path(gvcf_path).name}: missing contigs in gVCF header: {missing[:20]} total={len(missing)}")

    for chrom in tqdm(list(pos_by_chr.keys()), desc=f"{Path(gvcf_path).name}", unit="chr", leave=False):
        raw_chr = contig_map.get(chrom)
        if raw_chr is None:
            continue

        cand_positions = pos_by_chr[chrom]
        if cand_positions.size == 0:
            continue

        region_start = int(cand_positions[0])
        region_end = int(cand_positions[-1])

        an_obs_arr = an_obs_by_chr[chrom]
        callable_n_arr = callable_n_by_chr[chrom]

        explicit_hit_mask = np.zeros(cand_positions.shape[0], dtype=bool)


        try:
            rec_iter = vf.fetch(raw_chr)
        except ValueError:
            del explicit_hit_mask
            continue

        for rec in rec_iter:
            pos = int(rec.pos)
            gt = rec.samples[sample].get("GT", None)
            an_add = gt_allele_count(gt)
            called = an_add > 0

            if is_reference_block_record(rec, block_mode=block_mode):
                if not called:
                    continue

                end_pos = int(rec.stop)
                if end_pos < region_start or pos > region_end:
                    continue

                left_pos = max(pos, region_start)
                right_pos = min(end_pos, region_end)

                lo = int(np.searchsorted(cand_positions, left_pos, side="left"))
                hi = int(np.searchsorted(cand_positions, right_pos, side="right"))

                if lo < hi:
                    submask = ~explicit_hit_mask[lo:hi]
                    if np.any(submask):
                        idxs = np.flatnonzero(submask) + lo
                        callable_n_arr[idxs] += 1
                        an_obs_arr[idxs] += an_add
                continue

            if pos < region_start or pos > region_end:
                continue

            idx = int(np.searchsorted(cand_positions, pos, side="left"))
            if idx >= cand_positions.shape[0] or int(cand_positions[idx]) != pos:
                continue

            explicit_hit_mask[idx] = True

            if called:
                callable_n_arr[idx] += 1
                an_obs_arr[idx] += an_add

        del explicit_hit_mask
        gc.collect()

    vf.close()


def save_counts_npz(
    output_npz: Path,
    manifest_json: Path,
    pos_by_chr: Dict[str, np.ndarray],
    an_obs_by_chr: Dict[str, np.ndarray],
    callable_n_by_chr: Dict[str, np.ndarray],
    gvcf_files: List[str],
    cache_dir: str,
    cache_prefix: str,
    block_mode: str = "reference-blocks",
):
    payload = {}
    for chrom in pos_by_chr:
        payload[f"{chrom}__an_obs"] = an_obs_by_chr[chrom].astype(np.int32, copy=False)
        payload[f"{chrom}__callable_n"] = callable_n_by_chr[chrom].astype(np.int32, copy=False)

    atomic_write_npz(output_npz, **payload)

    manifest = {
        "output_npz": str(output_npz),
        "gvcf_files": gvcf_files,
        "n_gvcfs": len(gvcf_files),
        "counting_mode": block_mode,
        "counting_definition": "nonmissing GT allele/sample counts",
        "DP_BQ_GQ_filters_applied": False,
        "cache_dir": str(cache_dir),
        "cache_prefix": cache_prefix,
        "chr_order": list(pos_by_chr.keys()),
        "site_counts": {c: int(len(a)) for c, a in pos_by_chr.items()},
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(manifest, manifest_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--cache-prefix", required=True, choices=["candidate_index", "noncoding_index"])
    ap.add_argument("--gvcf-list", required=True)
    ap.add_argument("--output-npz", required=True)
    ap.add_argument("--manifest-json", default=None)
    ap.add_argument("--block-mode", choices=("reference-blocks", "legacy-records"), default="reference-blocks", help="Expand reference blocks, or replay original record-only historical counting")
    args = ap.parse_args()

    t0 = time.time()

    cache_dir = Path(args.cache_dir)
    output_npz = Path(args.output_npz)
    manifest_json = Path(args.manifest_json) if args.manifest_json else output_npz.with_suffix(".manifest.json")

    meta, pos_by_chr = load_position_cache(cache_dir, args.cache_prefix)
    gvcf_files = read_gvcf_list(args.gvcf_list)

    if not gvcf_files:
        raise RuntimeError(f"empty gVCF list: {args.gvcf_list}")

    log(f"cache_dir={cache_dir}")
    log(f"cache_prefix={args.cache_prefix}")
    log(f"n_sites={sum(len(a) for a in pos_by_chr.values()):,}")
    log(f"n_gvcfs={len(gvcf_files)}")
    log(f"output_npz={output_npz}")

    an_obs_by_chr, callable_n_by_chr = init_count_arrays(pos_by_chr)

    stop = {"flag": False}

    def _handle_signal(signum=None, frame=None):
        log(f"[WARN] received signal={signum}; saving partial counts")
        save_counts_npz(
            output_npz=output_npz,
            manifest_json=manifest_json,
            pos_by_chr=pos_by_chr,
            an_obs_by_chr=an_obs_by_chr,
            callable_n_by_chr=callable_n_by_chr,
            gvcf_files=gvcf_files,
            cache_dir=str(cache_dir),
            cache_prefix=args.cache_prefix,
            block_mode=args.block_mode,
        )
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    for i, fp in enumerate(gvcf_files, 1):
        log(f"scanning {i}/{len(gvcf_files)}: {fp}")
        scan_one_gvcf(
            gvcf_path=fp,
            pos_by_chr=pos_by_chr,
            an_obs_by_chr=an_obs_by_chr,
            callable_n_by_chr=callable_n_by_chr,
            block_mode=args.block_mode,
        )
        if stop["flag"]:
            return

    save_counts_npz(
        output_npz=output_npz,
        manifest_json=manifest_json,
        pos_by_chr=pos_by_chr,
        an_obs_by_chr=an_obs_by_chr,
        callable_n_by_chr=callable_n_by_chr,
        gvcf_files=gvcf_files,
        cache_dir=str(cache_dir),
        cache_prefix=args.cache_prefix,
        block_mode=args.block_mode,
    )

    log(f"done elapsed_sec={time.time() - t0:.2f}")


if __name__ == "__main__":
    main()
