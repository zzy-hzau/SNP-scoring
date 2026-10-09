#!/usr/bin/env python3
"Out-of-core feature attachment for LOCO split parquets using DuckDB."

from pathlib import Path
import argparse
import json
import re
import sys

import duckdb
import pandas as pd


def qid(x: str) -> str:
    return '"' + x.replace('"', '""') + '"'


def qstr(x: str) -> str:
    return "'" + str(x).replace("'", "''") + "'"


def parse_folds(s: str):
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    return sorted(set(out))


def path_expr(path: str, fold: int = None, split: str = None, expect_tsv=False):
    """
    Resolve path to DuckDB read expression.

    If a directory contains foldXX/split.parquet, use that exact split file.
    Else if directory contains parquet files, use recursive glob.
    Else if file, use file.
    """
    if not path:
        return None, None

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.is_dir():
        if fold is not None and split is not None:
            candidate = p / f"fold{fold:02d}" / f"{split}.parquet"
            if candidate.exists():
                return f"read_parquet({qstr(str(candidate))})", str(candidate)

        parqs = sorted(p.rglob("*.parquet"))
        if parqs:
            globp = str(p / "**" / "*.parquet")
            return f"read_parquet({qstr(globp)})", globp

        tsvs = sorted(list(p.rglob("*.tsv")) + list(p.rglob("*.tsv.gz")))
        if tsvs:
            return f"read_csv_auto({qstr(str(tsvs[0]))}, delim='\\t', header=true)", str(tsvs[0])

        raise FileNotFoundError(f"No parquet/tsv files under {p}")

    if p.is_file():
        if str(p).endswith(".parquet"):
            return f"read_parquet({qstr(str(p))})", str(p)
        if str(p).endswith(".tsv") or str(p).endswith(".tsv.gz"):
            return f"read_csv_auto({qstr(str(p))}, delim='\\t', header=true)", str(p)

    raise ValueError(f"Unsupported feature path: {p}")


def describe_cols(con, expr: str):
    rows = con.execute(f"DESCRIBE SELECT * FROM {expr} LIMIT 0").fetchall()
    return [r[0] for r in rows]


def _norm_chrom_sql(expr: str) -> str:
    """Normalize chr01/Chr1/1 without damaging A01/D01 contigs."""
    raw = f"regexp_replace(lower(trim(CAST({expr} AS VARCHAR))), '^chr', '')"
    stripped = f"regexp_replace({raw}, '^0+', '')"
    return f"CASE WHEN {stripped} = '' THEN '0' ELSE {stripped} END"


def _canonical_gene_sql(expr: str, species: str) -> str:
    """Normalize transcript/model suffixes to the gene-level key."""
    value = f"CAST({expr} AS VARCHAR)"
    if species == "maize":

        
        value = f"regexp_replace({value}, '_[A-Z]+[0-9]+$', '')"
    return value


def make_feature_view(con, view_name, expr, wanted_cols, aliases=None,
                      join_mode="auto"):
    cols = describe_cols(con, expr)
    coordinate_cols = (
        any(c in cols for c in ("Chr", "chr", "chrom", "chromosome"))
        and any(c in cols for c in ("Pos", "pos", "position", "start"))
    )
    use_coordinates = join_mode == "coordinates" or (
        join_mode == "auto" and "snp_id" not in cols
    )
    if not use_coordinates and "snp_id" in cols:
        key_mode = "snp_id"
        key_expr = f"CAST({qid('snp_id')} AS VARCHAR)"
        key_alias = "__join_snp"
    elif coordinate_cols:
        chr_col = next((c for c in ("Chr", "chr", "chrom", "chromosome") if c in cols), None)
        pos_col = next((c for c in ("Pos", "pos", "position", "start") if c in cols), None)
        key_mode = "coordinates"
        key_expr = (
            f"{_norm_chrom_sql(qid(chr_col))} "
            f"AS __join_chr, TRY_CAST({qid(pos_col)} AS BIGINT) AS __join_pos"
        )
        key_alias = "__join_chr"
    else:
        raise RuntimeError(
            f"{view_name} needs snp_id or Chr/Pos join keys; columns={cols[:30]}"
        )

    if aliases is None:
        selected = {c: c for c in wanted_cols if c in cols}
    else:
        selected = {}
        for output_col, candidates in aliases.items():
            for candidate in candidates:
                if candidate in cols:
                    selected[output_col] = candidate
                    break
    if not selected:
        raise RuntimeError(
            f"{view_name}: no wanted feature columns found. "
            f"wanted={wanted_cols}; cols={cols[:50]}"
        )

    select_cols = []
    if key_mode == "snp_id":
        select_cols.append(f"{key_expr} AS __join_snp")
    else:
        select_cols.extend(key_expr.split(", "))
    for output_col, source_col in selected.items():
        select_cols.append(
            f"any_value({qid(source_col)}) AS {qid(output_col)}"
        )

    sql = f"""
    CREATE OR REPLACE VIEW {qid(view_name)} AS
    SELECT
      {", ".join(select_cols)}
    FROM {expr}
    GROUP BY {qid(key_alias)}{", " + qid("__join_pos") if key_mode == "coordinates" else ""}
    """
    con.execute(sql)
    return list(selected), key_mode


def join_condition(view_name: str, key_mode: str) -> str:
    if key_mode == "snp_id":
        return f"CAST(l.snp_id AS VARCHAR)={qid(view_name)}.__join_snp"
    return (
        f"{_norm_chrom_sql('l.Chr')}="
        f"{qid(view_name)}.__join_chr AND "
        f"TRY_CAST(l.Pos AS BIGINT)={qid(view_name)}.__join_pos"
    )


def make_paralog_view(con, view_name, expr, species):
    cols = describe_cols(con, expr)
    gene_col = next((c for c in ("gene_id", "GeneID", "target_gene_id") if c in cols), None)
    if gene_col is None:
        raise RuntimeError(f"paralog table missing gene_id/GeneID; columns={cols[:30]}")

    aliases = {
        "has_homolog": ["has_homolog", "has_paralog"],
        "functional_redundancy": ["functional_redundancy", "paralog_redundancy"],
        "both_in_expr": ["both_in_expr"],
        "expr": ["expr"],
        "expr_corr": ["expr_corr", "best_pair_expr_corr"],
        "breadth": ["breadth", "best_pair_breadth"],
        "n_syntenic_partners": ["n_syntenic_partners", "n_paralogs"],
        "min_pair_redundancy": ["min_pair_redundancy"],
        "mean_pair_redundancy": ["mean_pair_redundancy"],
        "protein_identity": ["protein_identity"],
        "qcov": ["qcov"],
        "scov": ["scov"],
        "mincov": ["mincov"],
        "seq_weight": ["seq_weight"],
        "bitscore": ["bitscore", "score"],
    }
    keep = [name for name, candidates in aliases.items() if any(c in cols for c in candidates)]

    if not keep:
        raise RuntimeError(f"No paralog/redundancy feature columns found in {expr}")

    canonical_gene = _canonical_gene_sql(qid(gene_col), species)
    select_cols = [f"{canonical_gene} AS {qid('gene_id')}" ]
    output_names = []
    for c in keep:
        source = next(candidate for candidate in aliases[c] if candidate in cols)
        out = c if species == "cotton" else f"{species}_{c}"
        select_cols.append(f"any_value({qid(source)}) AS {qid(out)}")
        output_names.append(out)

    sql = f"""
    CREATE OR REPLACE VIEW {qid(view_name)} AS
    SELECT
      {", ".join(select_cols)}
    FROM {expr}
    GROUP BY {canonical_gene}
    """
    con.execute(sql)

    return output_names


def make_gene_map_view(con, view_name, expr, species):
    """Build a de-duplicated SNP-to-gene view for noncoding splits."""
    cols = describe_cols(con, expr)
    if "snp_id" not in cols:
        raise RuntimeError(f"gene map must contain snp_id; columns={cols[:30]}")
    gene_col = next((c for c in ("GeneID", "gene_id", "target_gene_id") if c in cols), None)
    if gene_col is None:
        raise RuntimeError(f"gene map must contain GeneID/gene_id; columns={cols[:30]}")
    gene_expr = _canonical_gene_sql(qid(gene_col), species)
    con.execute(f"""
    CREATE OR REPLACE VIEW {qid(view_name)} AS
    SELECT
      CAST({qid('snp_id')} AS VARCHAR) AS __join_snp,
      any_value({gene_expr}) AS __gene_id
    FROM {expr}
    WHERE {qid('snp_id')} IS NOT NULL AND {qid(gene_col)} IS NOT NULL
    GROUP BY CAST({qid('snp_id')} AS VARCHAR)
    """)
    return gene_col


def infer_gene_col(label_cols, forced="auto"):
    if forced and forced != "auto":
        if forced not in label_cols:
            raise RuntimeError(f"Forced gene column {forced} not in label columns")
        return forced
    for c in ["GeneID", "gene_id", "target_gene_id"]:
        if c in label_cols:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--kind", choices=["coding", "noncoding"], required=True)
    ap.add_argument("--label-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--splits", default="train,val,test")

    ap.add_argument("--lm-score", default=None)
    ap.add_argument("--dna-cons", action="append", default=None)
    ap.add_argument("--protein-cons", default=None)
    ap.add_argument("--paralog", default=None)
    ap.add_argument(
        "--gene-map", default=None,
        help="Static SNP-to-gene parquet/TSV used when label splits lack GeneID.",
    )
    ap.add_argument(
        "--gene-map-root", default=None,
        help="Fold-aware SNP-to-gene root containing foldXX/split.parquet files.",
    )
    ap.add_argument("--pcsd26", default=None)
    ap.add_argument("--extra-feature", action="append", default=[])

    ap.add_argument("--gene-col", default="auto")
    ap.add_argument("--require-nonnegative-sdna", action="store_true")
    ap.add_argument(
        "--min-conservation-coverage", type=float, default=0.50,
        help=(
            "Fail if an attached conservation score is less complete than this "
            "fraction. This catches broken joins; default=0.50."
        ),
    )
    ap.add_argument(
        "--min-redundancy-coverage", type=float, default=0.50,
        help=(
            "Fail if an attached redundancy score is less complete than this "
            "fraction; default=0.50."
        ),
    )
    ap.add_argument(
        "--fill-unmatched-redundancy", action="store_true",
        help=(
            "Explicitly encode SNPs whose GeneID is absent from the supplied "
            "redundancy table as zero for redundancy predictors. The attached "
            "<species>_paralog_feature_missing flag and source_coverage audit "
            "remain, so this is not silent imputation."
        ),
    )
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--memory-limit", default="160GB")
    args = ap.parse_args()

    label_root = Path(args.label_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    folds = parse_folds(args.folds)
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(args.threads)}")
    con.execute(f"PRAGMA memory_limit={qstr(args.memory_limit)}")

    manifest = []

    for fold in folds:
        for split in splits:
            label_p = label_root / f"fold{fold:02d}" / f"{split}.parquet"
            out_p = out_root / f"fold{fold:02d}" / f"{split}.parquet"
            out_p.parent.mkdir(parents=True, exist_ok=True)

            if not label_p.exists():
                print(f"[SKIP missing label] {label_p}")
                continue

            label_expr = f"read_parquet({qstr(str(label_p))})"
            label_cols = describe_cols(con, label_expr)
            if "snp_id" not in label_cols:
                raise RuntimeError(f"label missing snp_id: {label_p}")

            con.execute(f"CREATE OR REPLACE VIEW l AS SELECT * FROM {label_expr}")

            base_count = con.execute("SELECT count(*) FROM l").fetchone()[0]

            joins = []
            select_extra = []
            added = []
            source_paths = {}
            conservation_columns = []
            redundancy_columns = []

            if args.gene_map and args.gene_map_root:
                raise RuntimeError("use only one of --gene-map and --gene-map-root")


            

            gene_map_expr = None
            if args.gene_map or args.gene_map_root:
                map_path = args.gene_map_root or args.gene_map
                gene_map_expr, map_src = path_expr(map_path, fold, split)
                source_paths["gene_map"] = map_src
                make_gene_map_view(con, "gene_map", gene_map_expr, args.species)
                joins.append(
                    f"LEFT JOIN gene_map ON CAST(l.snp_id AS VARCHAR)=gene_map.__join_snp"
                )


            if args.lm_score:
                expr, src_used = path_expr(args.lm_score, fold, split)
                source_paths["lm_score"] = src_used
                lm_keep, lm_key = make_feature_view(con, "lm", expr, ["S_dna", "S_esm"])
                joins.append(f"LEFT JOIN lm ON {join_condition('lm', lm_key)}")
                for c in lm_keep:
                    if c != "snp_id":
                        select_extra.append(f"lm.{qid(c)} AS {qid(c)}")
                        added.append(c)


            if args.dna_cons:
                dna_aliases = {
                    "dna_cons__phastCons": [
                        "dna_cons__phastCons", "phastCons_score", "phastCons", "phastcons"
                    ],
                    "dna_cons__phyloP": [
                        "dna_cons__phyloP", "phyloP_score", "phyloP", "phylop"
                    ],
                    "dna_cons_missing": ["dna_cons_missing", "phastCons_missing", "phyloP_missing"],
                }
                for i, dna_path in enumerate(args.dna_cons):
                    expr, src_used = path_expr(dna_path, fold, split)
                    view = f"dna{i}"
                    source_paths[f"dna_cons_{i}"] = src_used
                    dna_keep, dna_key = make_feature_view(
                        con, view, expr, list(dna_aliases), dna_aliases,
                        join_mode="coordinates",
                    )
                    joins.append(
                        f"LEFT JOIN {qid(view)} ON {join_condition(view, dna_key)}"
                    )
                    for c in dna_keep:
                        select_extra.append(f"{qid(view)}.{qid(c)} AS {qid(c)}")
                        added.append(c)
                        if c in {"dna_cons__phastCons", "dna_cons__phyloP"}:
                            conservation_columns.append(c)


            if args.protein_cons:
                expr, src_used = path_expr(args.protein_cons, fold, split)
                source_paths["protein_cons"] = src_used
                cols = describe_cols(con, expr)
                wanted = [c for c in cols if c.startswith("prot_safe_")]
                prot_keep, prot_key = make_feature_view(con, "prot", expr, wanted)
                joins.append(f"LEFT JOIN prot ON {join_condition('prot', prot_key)}")
                for c in prot_keep:
                    if c != "snp_id":
                        select_extra.append(f"prot.{qid(c)} AS {qid(c)}")
                        added.append(c)


            if args.pcsd26:
                expr, src_used = path_expr(args.pcsd26, fold, split)
                source_paths["pcsd26"] = src_used
                cols = describe_cols(con, expr)
                pcsd_aliases = {
                    f"pcsd26_state__E{i}": [f"pcsd26_state__E{i}", f"pcsd_E{i}", f"E{i}"]
                    for i in range(1, 27)
                }
                pcsd_aliases["pcsd26_state_missing"] = [
                    "pcsd26_state_missing", "pcsd26_state_missing_flag"
                ]
                wanted = list(pcsd_aliases)
                if any(candidate in cols for values in pcsd_aliases.values() for candidate in values):
                    pcsd_keep, pcsd_key = make_feature_view(
                        con, "pcsd", expr, wanted, pcsd_aliases
                    )
                    joins.append(f"LEFT JOIN pcsd ON {join_condition('pcsd', pcsd_key)}")
                    for c in pcsd_keep:
                        if c != "snp_id":
                            select_extra.append(f"pcsd.{qid(c)} AS {qid(c)}")
                            added.append(c)
                else:
                    print(f"[WARN] pcsc26 path has no pcsd26_* columns: {src_used}")


            for i, ex_path in enumerate(args.extra_feature or []):
                expr, src_used = path_expr(ex_path, fold, split)
                view = f"extra{i}"
                cols = describe_cols(con, expr)
                wanted = [c for c in cols if c not in {"snp_id", "Chr", "chr", "chrom", "chromosome", "Pos", "pos", "position", "start"}]
                keep, extra_key = make_feature_view(con, view, expr, wanted)
                joins.append(f"LEFT JOIN {qid(view)} ON {join_condition(view, extra_key)}")
                for c in keep:
                    if c != "snp_id":
                        select_extra.append(f"{qid(view)}.{qid(c)} AS {qid(c)}")
                        added.append(c)
                source_paths[f"extra_{i}"] = src_used


            if args.paralog:
                gene_col = infer_gene_col(label_cols, args.gene_col)
                if gene_col is None and gene_map_expr is None:
                    raise RuntimeError(f"Could not infer gene col for paralog join in {label_p}")
                expr, src_used = path_expr(args.paralog, fold, split)
                source_paths["paralog"] = src_used
                paralog_cols = make_paralog_view(con, "paralog", expr, args.species)
                if gene_col is not None:
                    label_gene = _canonical_gene_sql(f"l.{qid(gene_col)}", args.species)
                    gene_join = (
                        f"COALESCE(NULLIF({label_gene}, ''), "
                        f"gene_map.__gene_id)"
                    ) if gene_map_expr is not None else label_gene
                else:
                    gene_join = "gene_map.__gene_id"
                joins.append(f"LEFT JOIN paralog ON {gene_join}=CAST(paralog.gene_id AS VARCHAR)")
                for c in paralog_cols:
                    if args.fill_unmatched_redundancy and (
                        c.endswith("_has_homolog")
                        or c.endswith("_functional_redundancy")
                        or c in {"has_homolog", "functional_redundancy"}
                    ):
                        select_extra.append(
                            f"COALESCE(paralog.{qid(c)}, 0.0) AS {qid(c)}"
                        )
                    else:
                        select_extra.append(f"paralog.{qid(c)} AS {qid(c)}")
                    added.append(c)
                    if c.endswith("_has_homolog") or c.endswith("_functional_redundancy") or c in {
                        "has_homolog", "functional_redundancy"
                    }:
                        redundancy_columns.append(c)
                miss = f"{args.species}_paralog_feature_missing"
                select_extra.append(f"CASE WHEN paralog.gene_id IS NULL THEN 1 ELSE 0 END AS {qid(miss)}")
                added.append(miss)

            if gene_map_expr is not None and infer_gene_col(label_cols, args.gene_col) is None:
                select_extra.append("gene_map.__gene_id AS GeneID")
                added.append("GeneID")

            select_part = "l.*"
            if select_extra:
                select_part += ",\n      " + ",\n      ".join(select_extra)

            sql = f"""
            COPY (
              SELECT
                {select_part}
              FROM l
              {' '.join(joins)}
            )
            TO {qstr(str(out_p))}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
            print(f"[ATTACH] {args.species} {args.kind} fold{fold:02d} {split}")
            con.execute(sql)

            out_count = con.execute(f"SELECT count(*) FROM read_parquet({qstr(str(out_p))})").fetchone()[0]
            if out_count != base_count:
                raise RuntimeError(f"Row count changed: {label_p} {base_count} -> {out_count}")

            neg_s = None
            if args.require_nonnegative_sdna:
                out_cols = describe_cols(con, f"read_parquet({qstr(str(out_p))})")
                if "S_dna" in out_cols:
                    neg_s = con.execute(
                        f"SELECT count(*) FROM read_parquet({qstr(str(out_p))}) WHERE TRY_CAST(S_dna AS DOUBLE) < 0"
                    ).fetchone()[0]
                    if neg_s:
                        raise RuntimeError(f"Negative S_dna detected in {out_p}: {neg_s}")


            

            coverage = {}
            coverage_columns = list(dict.fromkeys(conservation_columns + redundancy_columns))
            for col in coverage_columns:
                nonnull = con.execute(
                    f"SELECT count({qid(col)}) FROM read_parquet({qstr(str(out_p))})"
                ).fetchone()[0]
                rate = float(nonnull / out_count) if out_count else 0.0
                coverage[col] = {
                    "rows": int(out_count), "nonnull": int(nonnull),
                    "missing": int(out_count - nonnull), "coverage": rate,
                }
                threshold = (
                    args.min_conservation_coverage
                    if col in conservation_columns
                    else args.min_redundancy_coverage
                )
                if col in redundancy_columns and args.fill_unmatched_redundancy:
                    missing_flag = f"{args.species}_paralog_feature_missing"
                    source_nonnull = con.execute(
                        f"SELECT count(*) FROM read_parquet({qstr(str(out_p))}) "
                        f"WHERE TRY_CAST({qid(missing_flag)} AS BIGINT)=0 "
                        f"AND {qid(col)} IS NOT NULL"
                    ).fetchone()[0]
                    source_rate = float(source_nonnull / out_count) if out_count else 0.0
                    coverage[col]["source_nonnull"] = int(source_nonnull)
                    coverage[col]["source_coverage"] = source_rate
                    if source_rate < threshold:
                        raise RuntimeError(
                            f"{out_p}: {col} source_coverage={source_rate:.4f} is below "
                            f"the configured threshold={threshold:.4f}; zeros were not "
                            "accepted as evidence of a complete source table"
                        )
                if not args.fill_unmatched_redundancy and rate < threshold:
                    raise RuntimeError(
                        f"{out_p}: {col} coverage={rate:.4f} is below "
                        f"the configured attachment threshold={threshold:.4f}; "
                        "check source coverage, gene-map joins, and species-specific ID normalization"
                    )

            row = {
                "species": args.species,
                "kind": args.kind,
                "fold": fold,
                "split": split,
                "input": str(label_p),
                "output": str(out_p),
                "rows": int(out_count),
                "added_cols": ",".join(added),
                "negative_Sdna": int(neg_s) if neg_s is not None else "",
                "coverage": json.dumps(coverage, ensure_ascii=False, sort_keys=True),
                "sources": json.dumps(source_paths, ensure_ascii=False),
            }
            manifest.append(row)

    mf = pd.DataFrame(manifest)
    mf.to_csv(out_root / "feature_attach_manifest.tsv", sep="\t", index=False)

    cfg = {
        "species": args.species,
        "kind": args.kind,
        "label_root": args.label_root,
        "out_root": args.out_root,
        "folds": args.folds,
        "splits": splits,
        "lm_score": args.lm_score,
        "dna_cons": args.dna_cons,
        "protein_cons": args.protein_cons,
        "paralog": args.paralog,
        "gene_map": args.gene_map,
        "gene_map_root": args.gene_map_root,
        "pcsd26": args.pcsd26,
        "extra_feature": args.extra_feature,
        "fill_unmatched_redundancy": args.fill_unmatched_redundancy,
        "note": "DuckDB out-of-core feature attachment. Does not modify label construction variables.",
    }
    (out_root / "feature_attach_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    print("[DONE]", out_root)
    print("[WRITE]", out_root / "feature_attach_manifest.tsv")


if __name__ == "__main__":
    main()
