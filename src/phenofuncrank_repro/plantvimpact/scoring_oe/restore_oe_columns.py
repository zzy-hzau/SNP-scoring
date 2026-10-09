#!/usr/bin/env python3
"Join O/E fields from label tables to scored splits."
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def q(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def cols(con, expr: str) -> list[str]:
    return [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {expr} LIMIT 0").fetchall()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored-root", required=True, type=Path)
    ap.add_argument("--label-root", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--min-coverage", type=float, default=0.99)
    ap.add_argument(
        "--columns",
        default="Y_Label,observed_in_vcf,P_mut_fold,p_base_eff_fold,p_base_eff_bin,opportunity_bin,callable_n,callable_n_bin,callable_bin,chr_num,subgenome_code,Sg_code,Ctx_code,Alt_code,region_type,Context",
    )
    args = ap.parse_args()
    folds = []
    for part in args.folds.split(","):
        if "-" in part:
            a, b = part.split("-", 1); folds.extend(range(int(a), int(b) + 1))
        elif part.strip():
            folds.append(int(part))
    wanted = [x.strip() for x in args.columns.split(",") if x.strip()]
    required_columns = {
        "Y_Label", "opportunity_bin", "callable_bin", "chr_num",
        "Sg_code", "Ctx_code", "Alt_code",
    }
    con = duckdb.connect()
    reports = []
    for fold in sorted(set(folds)):
        for split in [x.strip() for x in args.splits.split(",") if x.strip()]:
            scored = args.scored_root / f"fold{fold:02d}" / f"{split}.parquet"
            labels = args.label_root / f"fold{fold:02d}" / f"{split}.parquet"
            out = args.output_root / f"fold{fold:02d}" / f"{split}.parquet"
            if not scored.is_file() or not labels.is_file():
                raise FileNotFoundError(f"missing scored/label split: {scored} / {labels}")
            se = f"read_parquet({q(str(scored))})"
            le = f"read_parquet({q(str(labels))})"
            scored_cols = cols(con, se)
            label_cols = cols(con, le)
            source_aliases = {
                "opportunity_bin": "p_base_eff_bin",
                "callable_bin": "callable_n_bin",
                "subgenome_code": "Sg_code",
            }
            add = [
                c for c in wanted
                if c not in scored_cols
                and source_aliases.get(c, c) in label_cols
            ]
            missing_source = [
                c for c in wanted
                if c not in scored_cols
                and source_aliases.get(c, c) not in label_cols
                and c in required_columns
            ]
            if missing_source:
                raise RuntimeError(f"{labels} lacks requested O/E columns: {missing_source}")
            selects = [f"s.{ident(c)}" for c in scored_cols]
            for c in add:
                source = {
                    "opportunity_bin": "p_base_eff_bin",
                    "callable_bin": "callable_n_bin",
                }.get(c, c)
                if source == "subgenome_code" and "Sg_code" in label_cols:
                    source = "Sg_code"
                if source not in label_cols:
                    if c not in required_columns:
                        continue
                    raise RuntimeError(f"{labels} lacks source column {source} for {c}")
                selects.append(f"m.{ident(source)} AS {ident(c)}")
            out.parent.mkdir(parents=True, exist_ok=True)
            con.execute(f"""
                COPY (
                    SELECT {', '.join(selects)},
                           CASE WHEN m.snp_id IS NULL THEN 1 ELSE 0 END AS oe_metadata_missing
                    FROM {se} s
                    LEFT JOIN {le} m ON CAST(s.snp_id AS VARCHAR)=CAST(m.snp_id AS VARCHAR)
                ) TO {q(str(out))} (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
            n, matched = con.execute(
                f"SELECT count(*), count(CASE WHEN oe_metadata_missing=0 THEN 1 END) FROM read_parquet({q(str(out))})"
            ).fetchone()
            coverage = float(matched / n) if n else 0.0
            if coverage < args.min_coverage:
                raise RuntimeError(f"{out}: O/E metadata coverage={coverage:.6f} < {args.min_coverage}")
            reports.append({"fold": fold, "split": split, "rows": int(n), "matched": int(matched), "coverage": coverage, "added": add})
            print(f"[OE-COLUMNS] fold{fold:02d} {split} coverage={coverage:.6f}", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {"scored_root": str(args.scored_root), "label_root": str(args.label_root), "output_root": str(args.output_root), "reports": reports}
    (args.output_root / "restore_oe_columns_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
