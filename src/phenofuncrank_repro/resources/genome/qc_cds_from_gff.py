#!/usr/bin/env python3
"Nine-point CDS quality check from a reference FASTA + GFF3."

import argparse
import json
import os
import sys
from collections import Counter, defaultdict


from phenofuncrank_repro.resources.utils.common import *


def parse_args(argv=None):
    """Build the argparse argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a 9-point CDS quality check against a reference FASTA + "
            "GFF3 file and write a JSON report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Path to the reference FASTA file.",
    )
    parser.add_argument(
        "--gff",
        required=True,
        help="Path to the GFF3 annotation file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output JSON QC report.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML configuration file.",
    )
    parser.add_argument(
        "--species",
        default=None,
        help="Optional species identifier (used for logging and report).",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Optional cap on the number of CDS records inspected (toy mode).",
    )
    parser.add_argument(
        "--internal-stop-threshold",
        type=float,
        default=0.005,
        help="Maximum tolerated fraction of internal stop codons.",
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


def iter_gff_cds(handle):
    """Yield (seqid, start, end, strand, phase, attrs_str) tuples for CDS rows."""
    for line in handle:
        if not line or line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 9:
            continue
        if fields[2] != "CDS":
            continue
        seqid = fields[0]
        try:
            start = int(fields[3])
            end = int(fields[4])
        except ValueError:
            continue
        strand = fields[6]
        phase_str = fields[7]
        attrs = fields[8]
        yield seqid, start, end, strand, phase_str, attrs


def translate(seq):
    """Translate a nucleotide sequence to amino acids (first frame, no trim)."""
    aa = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i + 3].upper()
        aa.append(CODON_TABLE.get(codon, "X"))
    return "".join(aa)


def main(argv=None):
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    if args.species is None:
        args.species = config.get("species")
    if "internal_stop_threshold" in config and \
            args.internal_stop_threshold == 0.005:
        args.internal_stop_threshold = float(
            config["internal_stop_threshold"]
        )

    species_tag = f"[{args.species}] " if args.species else ""
    log(f"{species_tag}Running 9-point CDS QC on {args.gff}")

    report = {
        "species": args.species,
        "fasta": args.fasta,
        "gff": args.gff,
        "checks": {},
    }


    try:
        fasta = load_fasta(args.fasta)
        report["checks"]["fasta_readable"] = {
            "pass": True,
            "n_seqids": len(fasta),
        }
    except Exception as exc:  
        report["checks"]["fasta_readable"] = {"pass": False, "error": str(exc)}
        log(f"{species_tag}ERROR reading FASTA: {exc}")
        try:
            with open(args.output, "w") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
        finally:
            return 1


    cds_records = []
    bad_field_rows = 0
    try:
        with open_text(args.gff, "rt") as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 9:
                    bad_field_rows += 1
                    continue
                if fields[2] != "CDS":
                    continue
                seqid = fields[0]
                try:
                    start = int(fields[3])
                    end = int(fields[4])
                except ValueError:
                    bad_field_rows += 1
                    continue
                strand = fields[6]
                phase_str = fields[7]
                attrs = fields[8]
                parent = parse_gff_attrs(attrs).get("Parent", "")
                cds_records.append(
                    (seqid, start, end, strand, phase_str, parent, attrs)
                )
                if args.max_positions and len(cds_records) >= args.max_positions:
                    break
    except OSError as exc:
        report["checks"]["gff_readable"] = {"pass": False, "error": str(exc)}
        log(f"{species_tag}ERROR reading GFF: {exc}")
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        return 1


    report["checks"]["gff_field_count"] = {
        "pass": bad_field_rows == 0,
        "short_rows": bad_field_rows,
    }


    gff_seqids = {rec[0] for rec in cds_records}
    missing_in_fasta = sorted(gff_seqids - set(fasta))
    report["checks"]["seqid_match"] = {
        "pass": len(missing_in_fasta) == 0,
        "gff_seqids": len(gff_seqids),
        "missing_in_fasta": missing_in_fasta[:50],
    }


    bad_phase = [
        rec for rec in cds_records if rec[4] not in ("0", "1", "2", ".")
    ]
    report["checks"]["cds_phase"] = {
        "pass": len(bad_phase) == 0,
        "bad_phase_count": len(bad_phase),
    }


    not_div3 = [
        rec for rec in cds_records
        if (rec[2] - rec[1] + 1) % 3 != 0
    ]
    report["checks"]["cds_length_div3"] = {
        "pass": len(not_div3) == 0,
        "not_div3_count": len(not_div3),
    }


    n_codons = 0
    n_internal_stops = 0
    for seqid, start, end, strand, _phase, _parent, _attrs in cds_records:
        chrom = fasta.get(seqid, "")
        if not chrom:
            continue
        sub = chrom[start - 1:end].upper()
        if strand == "-":
            sub = reverse_complement(sub)

        for i in range(0, len(sub) - 3, 3):
            codon = sub[i:i + 3]
            if len(codon) < 3:
                break
            n_codons += 1
            if CODON_TABLE.get(codon, "X") == "*":
                n_internal_stops += 1
    stop_rate = (n_internal_stops / n_codons) if n_codons else 0.0
    report["checks"]["internal_stop_rate"] = {
        "pass": stop_rate < args.internal_stop_threshold and n_codons > 0,
        "internal_stops": n_internal_stops,
        "total_internal_codons": n_codons,
        "rate": stop_rate,
        "threshold": args.internal_stop_threshold,
    }


    out_of_bounds = []
    for seqid, start, end, _strand, _phase, _parent, _attrs in cds_records:
        chrom_len = len(fasta.get(seqid, ""))
        if chrom_len == 0:
            continue
        if start < 1 or end > chrom_len:
            out_of_bounds.append((seqid, start, end, chrom_len))
    report["checks"]["cds_within_bounds"] = {
        "pass": len(out_of_bounds) == 0,
        "out_of_bounds_count": len(out_of_bounds),
        "examples": out_of_bounds[:10],
    }


    parent_chroms = defaultdict(set)
    for seqid, _start, _end, _strand, _phase, parent, _attrs in cds_records:
        if parent:
            parent_chroms[parent].add(seqid)
    multi_chrom_parents = {
        p: sorted(c) for p, c in parent_chroms.items() if len(c) > 1
    }
    report["checks"]["cds_single_chromosome"] = {
        "pass": len(multi_chrom_parents) == 0,
        "multi_chromosome_genes": len(multi_chrom_parents),
        "examples": dict(list(multi_chrom_parents.items())[:10]),
    }


    comp = Counter()
    for seqid, start, end, strand, _phase, _parent, _attrs in cds_records:
        chrom = fasta.get(seqid, "")
        if not chrom:
            continue
        sub = chrom[start - 1:end].upper()
        if strand == "-":
            sub = reverse_complement(sub)
        comp.update(sub)
    report["checks"]["nucleotide_composition"] = {
        "pass": True,
        "counts": dict(comp),
    }


    report["overall_pass"] = all(
        c.get("pass", False) for c in report["checks"].values()
    )
    report["n_cds_records"] = len(cds_records)

    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    status = "PASS" if report["overall_pass"] else "FAIL"
    log(
        f"{species_tag}QC complete: {status} "
        f"(n_cds={len(cds_records)}); report -> {args.output}"
    )
    return 0 if report["overall_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
