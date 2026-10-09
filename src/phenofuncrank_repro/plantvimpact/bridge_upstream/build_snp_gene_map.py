#!/usr/bin/env python3
"Build a compact SNP-to-gene map from real pre-LM and split tables."
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def sql_string(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def parquet_list(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix != ".parquet":
            raise ValueError(f"expected Parquet file: {path}")
        return [path]
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files under {path}")
    return files


def list_expr(paths: list[Path]) -> str:
    return "read_parquet([" + ",".join(sql_string(str(p)) for p in paths) + "], union_by_name=true)"


def csv_expr(path: Path) -> str:
    return f"read_csv_auto({sql_string(str(path))}, delim='\\t', header=true, union_by_name=true)"


def ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-root", required=True, type=Path,
                    help="LOCO split root containing foldXX/{train,val,test}.parquet")
    ap.add_argument("--source-root", required=True, type=Path,
                    help="raw-derived pre-LM Parquet universe or directory")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--source-snp-col", default="snp_id")
    ap.add_argument("--source-gene-col", default="GeneID")
    ap.add_argument("--target-splits", default="train,val,test")
    args = ap.parse_args()

    target_files = []
    for fold in sorted(args.target_root.glob("fold*")):
        for split in [x.strip() for x in args.target_splits.split(",") if x.strip()]:
            p = fold / f"{split}.parquet"
            if p.exists():
                target_files.append(p)
    if not target_files:
        raise FileNotFoundError(f"no target split Parquets under {args.target_root}")

    if args.source_root.is_file() and args.source_root.suffix in {".tsv", ".gz"}:
        source_files = [args.source_root]
        source_expr = csv_expr(args.source_root)
    else:
        source_files = parquet_list(args.source_root)
        source_expr = list_expr(source_files)

    con = duckdb.connect()
    target_expr = list_expr(target_files)
    target_count = con.execute(
        f"SELECT count(DISTINCT CAST(snp_id AS VARCHAR)) FROM {target_expr} "
        "WHERE snp_id IS NOT NULL"
    ).fetchone()[0]
    source_cols = [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {source_expr} LIMIT 0").fetchall()]
    if args.source_snp_col not in source_cols or args.source_gene_col not in source_cols:
        raise RuntimeError(
            f"source lacks requested columns {args.source_snp_col}/{args.source_gene_col}; "
            f"available={source_cols[:40]}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (
            SELECT
                CAST(t.snp_id AS VARCHAR) AS snp_id,
                any_value(CAST(s.{ident(args.source_gene_col)} AS VARCHAR)) AS GeneID
            FROM {target_expr} t
            INNER JOIN {source_expr} s
              ON CAST(t.snp_id AS VARCHAR)=CAST(s.{ident(args.source_snp_col)} AS VARCHAR)
            WHERE t.snp_id IS NOT NULL AND s.{ident(args.source_gene_col)} IS NOT NULL
            GROUP BY CAST(t.snp_id AS VARCHAR)
        ) TO {sql_string(str(args.output))}
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    mapped = con.execute(
        f"SELECT count(*) FROM read_parquet({sql_string(str(args.output))})"
    ).fetchone()[0]
    audit = {
        "target_root": str(args.target_root),
        "source_root": str(args.source_root),
        "target_files": [str(p) for p in target_files],
        "source_files": [str(p) for p in source_files],
        "target_distinct_snps": int(target_count),
        "mapped_distinct_snps": int(mapped),
        "mapping_coverage": float(mapped / target_count) if target_count else 0.0,
        "unmapped_distinct_snps": int(target_count - mapped),
        "source_snp_col": args.source_snp_col,
        "source_gene_col": args.source_gene_col,
    }
    audit_path = args.output.with_suffix(args.output.suffix + ".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
