"""Identical source function; caller module supplies its recorded policy bindings."""
from phenofuncrank_repro.plantvimpact.native_upstream.cotton_candidate_universe import *
from phenofuncrank_repro.plantvimpact.native_upstream.cotton_candidate_universe import _exact_chroms_from_regex

def load_callrate(path: Optional[str], max_callable_n: Optional[int] = None, chrom_regex: Optional[str] = None, parquet_batch_size: int = 2_000_000) -> Dict[str, Tuple[np.ndarray, dict]]:
    """Load callrate as chrom -> (sorted_pos, dict arrays).

    Optimized for chromosome-sharded runs:
      - read only needed columns
      - stream parquet batches
      - keep only chromosomes matching chrom_regex
      - store compact int32/float32 arrays where safe
    """
    if path is None:
        return {}

    logger.info(f"Loading callrate table: {path}")
    if chrom_regex:
        logger.info(f"Callrate chromosome filter: {chrom_regex}")
    pat = re.compile(chrom_regex) if chrom_regex else None
    p = str(path)

    buckets = defaultdict(lambda: defaultdict(list))
    total_seen = 0
    total_kept = 0

    def normalize_and_store(df: pd.DataFrame):
        nonlocal total_seen, total_kept
        total_seen += len(df)
        if len(df) == 0:
            return

        if "Chr" not in df.columns:
            if "chrom" in df.columns:
                df = df.rename(columns={"chrom": "Chr"})
            elif "#CHROM" in df.columns:
                df = df.rename(columns={"#CHROM": "Chr"})
        if "Pos" not in df.columns:
            if "pos" in df.columns:
                df = df.rename(columns={"pos": "Pos"})
            elif "position" in df.columns:
                df = df.rename(columns={"position": "Pos"})

        if "Chr" not in df.columns or "Pos" not in df.columns:
            raise ValueError("callrate table must contain Chr/Pos columns")

        df["Chr"] = df["Chr"].astype(str)
        if pat is not None:
            df = df[df["Chr"].map(lambda x: bool(pat.search(x)))].copy()
            if len(df) == 0:
                return

        df["Pos"] = pd.to_numeric(df["Pos"], errors="coerce").fillna(-1).astype(np.int64)

        for col in ["AN_obs", "callable_n", "callable_rate", "call_rate"]:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df["callable_rate"].isna().all():
            if not df["call_rate"].isna().all():
                df["callable_rate"] = df["call_rate"]
            elif not df["callable_n"].isna().all() and max_callable_n:
                df["callable_rate"] = df["callable_n"] / float(max_callable_n)

        if df["call_rate"].isna().all():
            df["call_rate"] = df["callable_rate"]

        if df["AN_obs"].isna().all() and not df["callable_n"].isna().all():
            df["AN_obs"] = df["callable_n"] * 2

        if df["callable_n"].isna().all() and not df["AN_obs"].isna().all() and max_callable_n:
            df["callable_n"] = np.minimum(max_callable_n, np.rint(df["AN_obs"] / 2.0))

        df = df[df["Pos"] > 0].copy()
        if len(df) == 0:
            return

        total_kept += len(df)

        for chrom, sub in df.groupby("Chr", sort=False):
            buckets[chrom]["Pos"].append(sub["Pos"].to_numpy(np.int64, copy=False))
            buckets[chrom]["AN_obs"].append(sub["AN_obs"].fillna(0).to_numpy(np.float32, copy=False))
            buckets[chrom]["callable_n"].append(sub["callable_n"].fillna(0).to_numpy(np.float32, copy=False))
            buckets[chrom]["callable_rate"].append(sub["callable_rate"].fillna(0).to_numpy(np.float32, copy=False))
            buckets[chrom]["call_rate"].append(sub["call_rate"].fillna(0).to_numpy(np.float32, copy=False))

    if p.endswith(".parquet") or p.endswith(".pq"):
        pf = pq.ParquetFile(p)
        names = pf.schema_arrow.names

        chr_col = "Chr" if "Chr" in names else "chrom" if "chrom" in names else "#CHROM" if "#CHROM" in names else None
        pos_col = "Pos" if "Pos" in names else "pos" if "pos" in names else "position" if "position" in names else None
        if chr_col is None or pos_col is None:
            raise ValueError("callrate parquet must contain Chr/Pos columns")

        read_cols = [chr_col, pos_col] + [c for c in ["AN_obs", "callable_n", "callable_rate", "call_rate"] if c in names]
        read_cols = list(dict.fromkeys(read_cols))

        exact_chroms = _exact_chroms_from_regex(chrom_regex)
        if exact_chroms:
            source = ds.dataset(p, format="parquet")
            batches = source.to_batches(
                columns=read_cols,
                filter=ds.field(chr_col).isin(exact_chroms),
                batch_size=parquet_batch_size,
                use_threads=True,
            )
        else:
            batches = pf.iter_batches(batch_size=parquet_batch_size, columns=read_cols)
        for rb in batches:
            normalize_and_store(rb.to_pandas())
            if total_seen % 20_000_000 < parquet_batch_size:
                logger.info(f"  callrate streamed rows seen={total_seen:,}, kept={total_kept:,}")
    else:
        sep = "	" if p.endswith((".tsv", ".tsv.gz", ".bed", ".bed.gz")) else ","
        for df in pd.read_csv(p, sep=sep, chunksize=parquet_batch_size):
            normalize_and_store(df)
            logger.info(f"  callrate streamed rows seen={total_seen:,}, kept={total_kept:,}")

    out = {}
    for chrom, d in buckets.items():
        pos = np.concatenate(d["Pos"])
        order = np.argsort(pos, kind="mergesort")
        pos = pos[order]

        data = {}
        for col in ["AN_obs", "callable_n", "callable_rate", "call_rate"]:
            arr = np.concatenate(d[col])[order]
            data[col] = arr

        if len(pos) > 1:
            keep = np.ones(len(pos), dtype=bool)
            keep[1:] = pos[1:] != pos[:-1]
            pos = pos[keep]
            for col in data:
                data[col] = data[col][keep]

        out[chrom] = (pos, data)

    logger.info(
        f"Callrate loaded: chroms={len(out)}, positions={sum(len(v[0]) for v in out.values()):,}; "
        f"streamed_seen={total_seen:,}, kept={total_kept:,}"
    )
    return out
