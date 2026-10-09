#!/usr/bin/env python3
"Generate coding candidate positions from a reference FASTA + GFF3."

import argparse
import os
import random
import sys
from collections import defaultdict


from phenofuncrank_repro.resources.utils.common import *


def parse_args(argv=None):
    """Build the argparse argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate coding candidate positions from a reference FASTA "
            "and GFF3 annotation, classifying each SNV as synonymous, "
            "missense, or nonsense."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fasta", required=True, help="Path to the reference FASTA file."
    )
    parser.add_argument(
        "--gff", required=True, help="Path to the GFF3 annotation file."
    )
    parser.add_argument(
        "--species", required=True, help="Species identifier for the output."
    )
    parser.add_argument(
        "--output", required=True, help="Path to the output TSV file."
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Cap on the number of output rows (toy/test mode).",
    )
    parser.add_argument(
        "--chromosomes",
        default=None,
        help="Comma-separated list of chromosomes to restrict to.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML configuration file with 'species' overrides.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed used when downsampling with --max-positions.",
    )
    return parser.parse_args(argv)


def load_yaml_config(path):
    """Load a YAML configuration file if PyYAML is available."""
    if not path:
        return {}
    try:
        import yaml  
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to use --config but is not installed."
        ) from exc
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def collect_cds_by_parent(gff_path, allowed_chroms):
    """Group CDS exons by their Parent attribute (gene/transcript)."""
    grouped = defaultdict(list)
    with open_text(gff_path, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "CDS":
                continue
            seqid = fields[0]
            if allowed_chroms is not None and seqid not in allowed_chroms:
                continue
            try:
                start = int(fields[3])
                end = int(fields[4])
            except ValueError:
                continue
            strand = fields[6]
            phase_str = fields[7]
            attrs = parse_gff_attrs(fields[8])
            parent = attrs.get("Parent", attrs.get("ID", ""))
            if isinstance(parent, list):
                parent = parent[0] if parent else ""
            try:
                phase = int(phase_str) if phase_str in ("0", "1", "2") else 0
            except ValueError:
                phase = 0
            grouped[parent].append((seqid, start, end, strand, phase))
    return grouped


def build_cds_sequence(parent, exons, fasta):
    """Concatenate exons for a parent CDS, returning the spliced sequence.

    Returns ``(sequence, position_map, strand)`` where ``position_map`` is a
    list of (cds_index, (seqid, genome_pos)) entries linking each CDS index
    back to its genomic coordinate. ``genome_pos`` is 1-based.
    """
    by_strand = defaultdict(list)
    for exon in exons:
        by_strand[exon[3]].append(exon)
    if len(by_strand) > 1:

        strand = max(by_strand, key=lambda s: len(by_strand[s]))
    else:
        strand = exons[0][3]


    sorted_exons = sorted(by_strand[strand], key=lambda e: e[1])
    if strand == "-":
        sorted_exons = sorted(sorted_exons, key=lambda e: e[1], reverse=True)

    sequence = []
    position_map = []
    cds_offset = 0
    for seqid, start, end, _strand, _phase in sorted_exons:
        chrom = fasta.get(seqid, "")
        if not chrom:
            continue
        sub = chrom[start - 1:end].upper()
        if strand == "-":
            sub = reverse_complement(sub)
            for i, _base in enumerate(sub):
                position_map.append((cds_offset + i, (seqid, end - i)))
        else:
            for i, _base in enumerate(sub):
                position_map.append((cds_offset + i, (seqid, start + i)))
        sequence.append(sub)
        cds_offset += len(sub)
    return "".join(sequence), position_map, strand


def classify_alt(ref_codon, codon_idx, alt_base):
    """Return (alt_codon, ref_aa, alt_aa, variant_type) for a substitution."""
    alt_codon = ref_codon[:codon_idx] + alt_base + ref_codon[codon_idx + 1:]
    ref_aa = CODON_TABLE.get(ref_codon, "X")
    alt_aa = CODON_TABLE.get(alt_codon, "X")
    if alt_aa == "*":
        vtype = "nonsense" if ref_aa != "*" else "synonymous"
    elif alt_aa == ref_aa:
        vtype = "synonymous"
    else:
        vtype = "missense"
    return alt_codon, ref_aa, alt_aa, vtype


def iter_coding_candidates(parent, exons, fasta, species):
    """Yield candidate rows for a single parent CDS record."""
    seq, pos_map, strand = build_cds_sequence(parent, exons, fasta)
    if not seq or len(seq) < 3:
        return

    n_codons = len(seq) // 3
    for codon_i in range(n_codons):
        codon_start = codon_i * 3
        ref_codon = seq[codon_start:codon_start + 3].upper()
        if len(ref_codon) < 3:
            continue
        if any(b not in BASES for b in ref_codon):
            continue
        for codon_pos in range(3):
            cds_index = codon_start + codon_pos
            if cds_index >= len(pos_map):
                continue
            _idx, (seqid, genome_pos) = pos_map[cds_index]
            coding_ref_base = seq[cds_index].upper()
            ref_base = reverse_complement(coding_ref_base) if strand == "-" else coding_ref_base

            
            for alt_base in sorted(BASES):
                if alt_base == ref_base:
                    continue
                coding_alt_base = reverse_complement(alt_base) if strand == "-" else alt_base
                alt_codon, ref_aa, alt_aa, vtype = classify_alt(
                    ref_codon, codon_pos, coding_alt_base
                )
                snp_id = f"{seqid}:{genome_pos}:{ref_base}:{alt_base}"
                yield (
                    species,
                    "coding",
                    snp_id,
                    seqid,
                    genome_pos,
                    ref_base,
                    alt_base,
                    codon_pos + 1,
                    ref_codon,
                    ref_aa,
                    alt_aa,
                    vtype,
                )


def main(argv=None):
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    if args.species is None:
        args.species = config.get("species")
    if args.random_seed is None and "random_seed" in config:
        args.random_seed = config["random_seed"]

    if args.random_seed is not None:
        random.seed(args.random_seed)

    allowed_chroms = None
    if args.chromosomes:
        allowed_chroms = set(c.strip() for c in args.chromosomes.split(",") if c.strip())
    elif config.get("chromosomes"):
        allowed_chroms = set(config["chromosomes"])

    species_tag = f"[{args.species}] " if args.species else ""
    log(f"{species_tag}Building coding candidates from {args.gff}")

    try:
        fasta = load_fasta(args.fasta)
    except Exception as exc:  
        log(f"ERROR loading FASTA: {exc}")
        return 1

    if allowed_chroms is not None:
        allowed_chroms = {
            c for c in allowed_chroms if c in fasta or norm_chrom(c) in fasta
        }

        if allowed_chroms:
            normalized = {norm_chrom(c): c for c in fasta}
            allowed_chroms = {
                normalized.get(norm_chrom(c), c) for c in allowed_chroms
            }

    grouped = collect_cds_by_parent(args.gff, allowed_chroms)

    try:
        out_handle = open_text(args.output, "wt")
    except OSError as exc:
        log(f"ERROR: cannot open output {args.output}: {exc}")
        return 1

    header = [
        "species", "kind", "snp_id", "Chr", "Pos", "Ref", "Alt",
        "codon_pos", "ref_codon", "ref_aa", "alt_aa", "variant_type",
    ]

    try:
        out_handle.write("\t".join(header) + "\n")
        n_written = 0
        for parent, exons in grouped.items():
            parent_clean = clean_gene_id(parent) if parent else parent
            for row in iter_coding_candidates(parent_clean, exons, fasta, args.species):
                out_handle.write("\t".join(str(x) for x in row) + "\n")
                n_written += 1
                if args.max_positions and n_written >= args.max_positions:
                    log(
                        f"{species_tag}Reached --max-positions="
                        f"{args.max_positions}; stopping early."
                    )
                    return 0
    finally:
        out_handle.close()

    log(
        f"{species_tag}Done. coding_candidates_written={n_written} "
        f"parents_seen={len(grouped)} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
