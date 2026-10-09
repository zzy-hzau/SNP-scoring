"Previous I/O interface; allele/annotation functions are shared."
from phenofuncrank_repro.plantvimpact.native_upstream.cotton_candidate_universe import *
from phenofuncrank_repro.plantvimpact.native_upstream.cotton_candidate_universe import _exact_chroms_from_regex

def _exact_chroms_from_regex(chrom_regex: Optional[str]) -> Optional[List[str]]:
    """Return exact chromosome names only for simple regex such as ^1$ or ^(1|2|3)$."""
    if not chrom_regex:
        return None
    x = str(chrom_regex)
    if not (x.startswith("^") and x.endswith("$")):
        return None
    inner = x[1:-1]
    if re.fullmatch(r"[A-Za-z0-9_.-]+", inner):
        return [inner]
    if inner.startswith("(") and inner.endswith(")") and "|" in inner:
        xs = inner[1:-1].split("|")
        if xs and all(re.fullmatch(r"[A-Za-z0-9_.-]+", z) for z in xs):
            return xs
    if inner.startswith("(") and inner.endswith(")"):
        item = inner[1:-1]
        if re.fullmatch(r"[A-Za-z0-9_.-]+", item):
            return [item]
    return None


def write_iterator_to_parquet(iterator: Iterator[dict], path: Path, chunk_rows: int, max_rows: Optional[int] = None, require_callrate_hit: bool = False) -> int:
    w = ParquetChunkWriter(path, chunk_rows=chunk_rows)
    n = 0
    try:
        for rec in iterator:
            if require_callrate_hit and int(rec.get("callrate_hit", 0)) != 1:
                continue
            w.add(rec)
            n += 1
            if max_rows is not None and n >= max_rows:
                logger.warning(f"Reached --max-rows-per-mode={max_rows}; stopping early for this mode.")
                break
    finally:
        w.close()
    return n

def main():
    ap = argparse.ArgumentParser(description="Build pre-LM coding/noncoding SNP universe from raw FASTA/GFF/VCF/callrate")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--gff", required=True)
    ap.add_argument("--wild-vcf", required=True)
    ap.add_argument("--callrate", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", default="both", choices=["coding", "noncoding", "both"])
    ap.add_argument("--n-chr", type=int, default=13)
    ap.add_argument("--subgenome-mode", default="auto", choices=["auto", "single"])
    ap.add_argument("--chrom-regex", default=None, help="Optional regex to keep only main chromosomes in FASTA/VCF")
    ap.add_argument("--max-callable-n", type=int, default=None, help="Used to derive callable_rate from callable_n if needed")
    ap.add_argument("--promoter-upstream", type=int, default=2000)
    ap.add_argument("--promoter-downstream", type=int, default=200)
    ap.add_argument("--splice-window", type=int, default=8)
    ap.add_argument("--chunk-rows", type=int, default=500000)
    ap.add_argument("--position-chunk-size", type=int, default=200000)
    ap.add_argument("--max-rows-per-mode", type=int, default=None, help="Debug only; do not use for final runs")
    ap.add_argument("--require-callrate-hit", action="store_true", help="Keep only candidates present in the callability table")
    ap.add_argument("--intron-selected", default=None, help="Optional parquet from annotation-first intron top1%% selection; if provided, noncoding intron positions are restricted to this whitelist.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genome = load_fasta(args.genome, args.chrom_regex)
    gff_obj = parse_gff(args.gff, set(genome))
    vcf_idx = load_vcf_snv_index(args.wild_vcf, args.chrom_regex)
    call_idx = load_callrate(args.callrate, args.max_callable_n, args.chrom_regex)

    manifest = {
        "genome": args.genome,
        "gff": args.gff,
        "wild_vcf": args.wild_vcf,
        "callrate": args.callrate,
        "mode": args.mode,
        "n_chr": args.n_chr,
        "subgenome_mode": args.subgenome_mode,
        "promoter_upstream": args.promoter_upstream,
        "promoter_downstream": args.promoter_downstream,
        "splice_window": args.splice_window,
        "fasta_chroms": list(genome.keys()),
        "gff_feature_counts": gff_obj["feature_counts"],
        "outputs": {},
    }

    if args.mode in {"coding", "both"}:
        path = out_dir / "coding_preLM_universe.parquet"
        logger.info("Building CODING missense pre-LM universe...")
        n = write_iterator_to_parquet(iter_coding_missense(gff_obj, genome, vcf_idx, call_idx, args), path, args.chunk_rows, args.max_rows_per_mode, args.require_callrate_hit)
        manifest["outputs"]["coding_preLM_universe"] = str(path)
        manifest["outputs"]["coding_rows"] = n
        logger.info(f"CODING pre-LM universe rows: {n:,}")

    if args.mode in {"noncoding", "both"}:
        path = out_dir / "noncoding_preLM_universe.parquet"
        logger.info("Building NONCODING pre-LM universe...")
        n = write_iterator_to_parquet(iter_noncoding(gff_obj, genome, vcf_idx, call_idx, args), path, args.chunk_rows, args.max_rows_per_mode, args.require_callrate_hit)
        manifest["outputs"]["noncoding_preLM_universe"] = str(path)
        manifest["outputs"]["noncoding_rows"] = n
        logger.info(f"NONCODING pre-LM universe rows: {n:,}")

    (out_dir / "preLM_universe_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Done. Manifest: {out_dir / 'preLM_universe_manifest.json'}")

from phenofuncrank_repro.common.source_bindings import bind_functions as _bind_identical
from phenofuncrank_repro.common import shared_load_callrate as _profile_kernel_15
_bind_identical(globals(), _profile_kernel_15, ['load_callrate'])
