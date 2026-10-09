#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    raise RuntimeError("需要 pyarrow 来流式写 parquet") from e


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_position_cache(cache_dir: Path, prefix: str):
    meta_path = cache_dir / f"{prefix}.meta.json"
    npz_path = cache_dir / f"{prefix}.pos_by_chr.npz"

    meta = json.load(open(meta_path))
    chr_order = list(meta["chr_order"])

    pos_by_chr = {}
    with np.load(npz_path, allow_pickle=False) as npz:
        for chrom in chr_order:
            pos_by_chr[chrom] = np.asarray(npz[chrom], dtype=np.int64)
    return meta, pos_by_chr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--cache-prefix", required=True, choices=["candidate_index", "noncoding_index"])
    ap.add_argument("--parts-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--chunk-rows", type=int, default=1_000_000)
    ap.add_argument("--subgenome-code", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    cache_dir = Path(args.cache_dir)
    parts_dir = Path(args.parts_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    parts = sorted(parts_dir.glob("*.counts.npz"))
    if not parts:
        raise RuntimeError(f"no *.counts.npz found in {parts_dir}")

    meta, pos_by_chr = load_position_cache(cache_dir, args.cache_prefix)
    chr_order = list(pos_by_chr.keys())

    log(f"cache_dir={cache_dir}")
    log(f"cache_prefix={args.cache_prefix}")
    log(f"parts={len(parts)}")
    log(f"output={output}")
    log(f"n_sites={sum(len(a) for a in pos_by_chr.values()):,}")

    sums_an = {c: np.zeros(len(pos_by_chr[c]), dtype=np.int64) for c in chr_order}
    sums_call = {c: np.zeros(len(pos_by_chr[c]), dtype=np.int64) for c in chr_order}

    for i, fp in enumerate(parts, 1):
        log(f"merge part {i}/{len(parts)}: {fp.name}")
        with np.load(fp, allow_pickle=False) as npz:
            for chrom in chr_order:
                k1 = f"{chrom}__an_obs"
                k2 = f"{chrom}__callable_n"
                if k1 not in npz or k2 not in npz:
                    raise RuntimeError(f"{fp} missing keys for chrom {chrom}")
                a1 = np.asarray(npz[k1], dtype=np.int64)
                a2 = np.asarray(npz[k2], dtype=np.int64)
                if a1.shape[0] != sums_an[chrom].shape[0] or a2.shape[0] != sums_call[chrom].shape[0]:
                    raise RuntimeError(f"{fp} shape mismatch at {chrom}")
                sums_an[chrom] += a1
                sums_call[chrom] += a2

    schema = pa.schema([
        pa.field("Chr", pa.string()),
        pa.field("Pos", pa.int64()),
        pa.field("subgenome_code", pa.int8()),
        pa.field("AN_obs", pa.int64()),
        pa.field("callable_n", pa.int64()),
    ])

    writer = pq.ParquetWriter(str(output), schema=schema, compression="snappy")

    try:
        for chrom in chr_order:
            pos = pos_by_chr[chrom]
            an = sums_an[chrom]
            call = sums_call[chrom]
            n = len(pos)
            log(f"write chrom={chrom} rows={n:,}")

            for s in range(0, n, args.chunk_rows):
                e = min(n, s + args.chunk_rows)
                m = e - s

                table = pa.Table.from_arrays(
                    [
                        pa.array([chrom] * m, type=pa.string()),
                        pa.array(pos[s:e], type=pa.int64()),
                        pa.array(np.full(m, args.subgenome_code, dtype=np.int8), type=pa.int8()),
                        pa.array(an[s:e], type=pa.int64()),
                        pa.array(call[s:e], type=pa.int64()),
                    ],
                    schema=schema,
                )
                writer.write_table(table)
    finally:
        writer.close()

    summary = {
        "output": str(output),
        "cache_dir": str(cache_dir),
        "cache_prefix": args.cache_prefix,
        "parts_dir": str(parts_dir),
        "n_parts": len(parts),
        "n_sites_total": int(sum(len(a) for a in pos_by_chr.values())),
        "chr_order": chr_order,
        "elapsed_sec": time.time() - t0,
    }
    with open(output.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log(f"done elapsed_sec={time.time() - t0:.2f}")


if __name__ == "__main__":
    main()
