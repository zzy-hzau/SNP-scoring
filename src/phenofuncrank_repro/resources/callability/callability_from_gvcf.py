#!/usr/bin/env python3
"Compute callable positions and observed allele numbers from gVCF files."

import argparse
import os
import sys
from collections import defaultdict


from phenofuncrank_repro.resources.utils.common import *


def parse_args(argv=None):
    """Build the argparse argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-position callable_n and AN_obs from a list of "
            "gVCF files using pysam."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gvcf-list",
        required=True,
        help="Text file with one gVCF path per line.",
    )
    parser.add_argument(
        "--output", required=True, help="Output TSV path."
    )
    parser.add_argument(
        "--chromosomes",
        default=None,
        help="Comma-separated list of chromosomes to restrict to.",
    )
    parser.add_argument(
        "--region", default=None,
        help="One 1-based inclusive chr:start-end interval; clips blocks before expansion.",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Cap on the number of output positions (toy/test mode).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML configuration file.",
    )
    parser.add_argument(
        "--species",
        default=None,
        help="Optional species identifier (used for logging only).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed used when downsampling positions with --max-positions.",
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


def import_pysam():
    """Import pysam with a helpful error message if unavailable."""
    try:
        import pysam  
        return pysam
    except ImportError as exc:
        raise RuntimeError(
            "pysam is required for gVCF parsing. Install via "
            "`pip install pysam`."
        ) from exc


def parse_region(value):
    if not value:
        return None
    chrom, coordinates = value.rsplit(":", 1)
    start, end = map(int, coordinates.replace(",", "").split("-", 1))
    if start < 1 or end < start:
        raise ValueError("Region must be a 1-based inclusive chr:start-end interval.")
    return chrom, start, end


def iter_gvcf_callable(vcf_path, allowed_chroms, region=None):
    "Yield (chrom, 1-based pos, called allele count) for one sample gVCF."
    pysam = import_pysam()
    with pysam.VariantFile(vcf_path) as vcf:
        if len(vcf.header.samples) != 1:
            raise ValueError("Expected one sample per gVCF file.")
        sample_name = next(iter(vcf.header.samples))
        if region is None:
            records = iter(vcf)
        else:
            chrom, start, end = region
            try:
                records = vcf.fetch(chrom, start - 1, end)
            except (ValueError, OSError):
                records = iter(vcf)
        for rec in records:
            chrom = rec.chrom
            if allowed_chroms is not None and chrom not in allowed_chroms:
                continue
            if region is not None and chrom != region[0]:
                continue
            gt = rec.samples[sample_name].get("GT")
            an_obs = sum(allele is not None for allele in (gt or ()))
            if an_obs == 0:
                continue
            symbolic_only = not rec.alts or all(
                alt in {"<NON_REF>", "<*>"} for alt in rec.alts
            )
            end_pos = rec.stop if symbolic_only else rec.pos
            start_pos = rec.pos
            if region is not None:
                start_pos = max(start_pos, region[1])
                end_pos = min(end_pos, region[2])
            for pos in range(start_pos, end_pos + 1):
                yield chrom, pos, an_obs


def main(argv=None):
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    if args.species is None:
        args.species = config.get("species")
    if args.random_seed is None and "random_seed" in config:
        args.random_seed = config["random_seed"]

    allowed_chroms = None
    if args.chromosomes:
        allowed_chroms = set(c.strip() for c in args.chromosomes.split(",") if c.strip())
    elif config.get("chromosomes"):
        allowed_chroms = set(config["chromosomes"])

    species_tag = f"[{args.species}] " if args.species else ""

    region = parse_region(args.region)


    try:
        with open_text(args.gvcf_list, "rt") as handle:
            gvcf_paths = [line.strip() for line in handle if line.strip()]
    except OSError as exc:
        log(f"ERROR: cannot read gVCF list {args.gvcf_list}: {exc}")
        return 1

    log(
        f"{species_tag}Processing {len(gvcf_paths)} gVCF files; "
        f"output -> {args.output}"
    )


    callable_n = defaultdict(int)
    an_obs_total = defaultdict(int)

    sample_ids = set()
    for path in gvcf_paths:
        seen = {}
        try:
            with import_pysam().VariantFile(path) as header_reader:
                names = list(header_reader.header.samples)
            if len(names) != 1 or names[0] in sample_ids:
                raise ValueError("Each gVCF must contain one distinct sample; combine same-sample fragments first.")
            sample_ids.add(names[0])
            for chrom, pos, an_obs in iter_gvcf_callable(path, allowed_chroms, region):
                key = (chrom, pos)
                if key in seen:
                    if seen[key] != an_obs:
                        raise ValueError(f"Conflicting called allele counts at {chrom}:{pos}")
                    continue
                seen[key] = an_obs
                an_obs_total[key] += an_obs
                callable_n[key] += 1
        except Exception as exc:
            log(f"ERROR: cannot include gVCF {path}: {exc}")
            return 1


    items = sorted(callable_n.items(), key=lambda kv: (kv[0][0], kv[0][1]))

    if args.max_positions is not None and len(items) > args.max_positions:
        import random as _random
        if args.random_seed is not None:
            _random.seed(args.random_seed)
        items = _random.sample(items, args.max_positions)
        items.sort(key=lambda kv: (kv[0][0], kv[0][1]))

    try:
        out_handle = open_text(args.output, "wt")
    except OSError as exc:
        log(f"ERROR: cannot open output {args.output}: {exc}")
        return 1

    try:
        out_handle.write("Chr\tPos\tcallable_n\tAN_obs\n")
        for (chrom, pos), n in items:
            out_handle.write(
                f"{chrom}\t{pos}\t{n}\t{an_obs_total[(chrom, pos)]}\n"
            )
    finally:
        out_handle.close()

    log(
        f"{species_tag}Done. callable_positions={len(items)} "
        f"samples={len(gvcf_paths)} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
