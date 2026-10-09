#!/usr/bin/env python3
"Map missense variant positions to MSA column indices."
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

from phenofuncrank_repro.resources.utils.common import log, open_text, fasta_iter, load_fasta


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--variants", required=True, type=Path,
                    help="TSV with gene_id, protein_pos, ref_aa, alt_aa.")
    ap.add_argument("--msa-dir", required=True, type=Path,
                    help="Directory with per-gene MSA FASTA files.")
    ap.add_argument("--alias-map", type=Path, default=None,
                    help="TSV mapping variant gene_id -> msa_protein_id.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output TSV with MSA column mapping.")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Process only the first N variants (toy/test mode).")
    return ap.parse_args(argv)


def load_alias_map(path):
    """Read a TSV with columns gene_id, msa_protein_id -> dict."""
    alias = {}
    with open_text(path) as fh:
        header = fh.readline().rstrip("\n")
        cols = header.split("\t")
        try:
            g_idx = cols.index("gene_id")
            p_idx = cols.index("msa_protein_id")
        except ValueError:
            raise SystemExit(
                f"[ERROR] alias-map must have 'gene_id' and 'msa_protein_id' "
                f"columns; got {cols}"
            )
        for line in fh:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            alias[parts[g_idx]] = parts[p_idx]
    return alias


def find_msa_file(msa_dir, key):
    """Locate an MSA FASTA file whose stem equals ``key``.

    Looks for ``<key>.fa``, ``<key>.fasta``, ``<key>.fa.gz`` and a case-insensitive
    match, returning the first hit or ``None``.
    """
    candidates = []
    for ext in (".fa", ".fasta", ".fa.gz", ".fasta.gz", ".aln", ".msa"):
        candidates.append(msa_dir / f"{key}{ext}")
    for cand in candidates:
        if cand.exists():
            return cand

    key_lower = key.lower()
    for cand in msa_dir.iterdir():
        if cand.is_file() and cand.stem.lower().split(".")[0] == key_lower:
            return cand
    return None


def protein_pos_to_msa_col(msa_seq, aa_pos):
    """Convert a 1-based protein residue position to a 0-based MSA column.

    Gaps (``-`` and ``.``) are skipped.  Returns ``(col, residue)`` or
    ``(None, None)`` when the position is out of range.
    """
    seen = 0
    for i, aa in enumerate(msa_seq):
        if aa in ("-", "."):
            continue
        seen += 1
        if seen == aa_pos:
            return i, aa
    return None, None


def main(argv=None):
    args = parse_args(argv)

    if not args.variants.exists():
        raise SystemExit(f"[ERROR] variants file not found: {args.variants}")
    if not args.msa_dir.is_dir():
        raise SystemExit(f"[ERROR] msa-dir not found: {args.msa_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    alias = {}
    if args.alias_map is not None:
        log(f"Loading alias map from {args.alias_map}")
        alias = load_alias_map(args.alias_map)
        log(f"Loaded {len(alias)} alias entries")


    msa_cache = {}

    def get_msa(protein_id):
        if protein_id in msa_cache:
            return msa_cache[protein_id]
        msa_file = find_msa_file(args.msa_dir, protein_id)
        if msa_file is None:
            msa_cache[protein_id] = (None, None)
            return None, None
        seqs = load_fasta(msa_file)
        msa_cache[protein_id] = (msa_file, seqs)
        return msa_file, seqs

    rows = []
    n_total = 0
    n_mapped = 0
    n_no_file = 0
    n_no_protein = 0
    n_out_of_range = 0

    with open_text(args.variants) as fh:
        header = fh.readline().rstrip("\n")
        cols = header.split("\t")
        col_idx = {c: i for i, c in enumerate(cols)}
        for required in ("gene_id", "protein_pos", "ref_aa", "alt_aa"):
            if required not in col_idx:
                raise SystemExit(
                    f"[ERROR] variants file must have a '{required}' column; "
                    f"got {cols}"
                )
        has_vid = "variant_id" in col_idx

        for line in fh:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            gene_id = parts[col_idx["gene_id"]]
            try:
                protein_pos = int(parts[col_idx["protein_pos"]])
            except ValueError:
                continue
            ref_aa = parts[col_idx["ref_aa"]].strip().upper()
            alt_aa = parts[col_idx["alt_aa"]].strip().upper()
            variant_id = (parts[col_idx["variant_id"]] if has_vid
                          else f"{gene_id}:{protein_pos}:{ref_aa}:{alt_aa}")

            n_total += 1

            msa_protein_id = alias.get(gene_id, gene_id)
            msa_file, seqs = get_msa(msa_protein_id)

            if msa_file is None:
                n_no_file += 1
                rows.append((variant_id, gene_id, msa_protein_id, "NA",
                             "NA", "NA"))
                continue

            if seqs is None or msa_protein_id not in seqs:
                n_no_protein += 1
                rows.append((variant_id, gene_id, msa_protein_id,
                             str(msa_file), "NA", "NA"))
                continue

            target_seq = seqs[msa_protein_id].upper()
            col, residue = protein_pos_to_msa_col(target_seq, protein_pos)

            if col is None:
                n_out_of_range += 1
                rows.append((variant_id, gene_id, msa_protein_id,
                             str(msa_file), "NA", "NA"))
                continue

            ref_match = 1 if residue == ref_aa else 0
            n_mapped += 1
            rows.append((variant_id, gene_id, msa_protein_id, str(msa_file),
                         col, ref_match))

            if args.max_rows is not None and n_total >= args.max_rows:
                break

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(tmp, "w") as out:
        out.write("variant_id\tgene_id\tmsa_protein_id\tmsa_file\t"
                  "msa_column\tref_aa_match\n")
        for r in rows:
            out.write("\t".join(str(x) for x in r) + "\n")
    tmp.replace(args.output)

    log(f"Processed {n_total} variants; wrote {args.output}")
    log(f"  mapped={n_mapped} no_file={n_no_file} "
        f"no_protein={n_no_protein} out_of_range={n_out_of_range}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
