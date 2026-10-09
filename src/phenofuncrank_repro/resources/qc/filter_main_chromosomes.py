#!/usr/bin/env python3
"Filter a VCF to keep only main chromosomes."

import argparse
import os
import sys


from phenofuncrank_repro.resources.utils.common import *


def parse_args(argv=None):
    """Build the argparse argument parser for this script."""
    parser = argparse.ArgumentParser(
        description=(
            "Filter a VCF to keep only main chromosomes, preserving the "
            "header."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vcf", required=True, help="Path to the input VCF file."
    )
    parser.add_argument(
        "--output", required=True, help="Path to the filtered VCF output."
    )
    parser.add_argument(
        "--chromosomes",
        default=None,
        help="Comma-separated list of main chromosomes, or path to a YAML "
        "file with a 'chromosomes' list.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML configuration file with a 'chromosomes' list.",
    )
    parser.add_argument(
        "--species",
        default=None,
        help="Optional species identifier (used for logging only).",
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


def resolve_chromosome_set(spec, config):
    """Resolve the --chromosomes argument into a set of allowed seqids.

    The spec may be either a comma-separated list of seqids or a path to a
    YAML file containing a ``chromosomes`` list.
    """
    if spec is None and config.get("chromosomes"):
        return set(str(c) for c in config["chromosomes"])
    if spec is None:
        return None

    if os.path.isfile(spec):
        cfg = load_yaml_config(spec)
        if "chromosomes" in cfg:
            return set(str(c) for c in cfg["chromosomes"])

    return set(c.strip() for c in spec.split(",") if c.strip())


def main(argv=None):
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    if args.species is None:
        args.species = config.get("species")

    allowed_chroms = resolve_chromosome_set(args.chromosomes, config)
    if allowed_chroms is None:
        log("ERROR: --chromosomes or a 'chromosomes' config key is required")
        return 1

    species_tag = f"[{args.species}] " if args.species else ""
    log(
        f"{species_tag}Filtering {args.vcf} to {len(allowed_chroms)} "
        f"main chromosomes -> {args.output}"
    )

    try:
        in_handle = open_text(args.vcf, "rt")
    except OSError as exc:
        log(f"ERROR: cannot open VCF {args.vcf}: {exc}")
        return 1

    try:
        out_handle = open_text(args.output, "wt")
    except OSError as exc:
        log(f"ERROR: cannot open output {args.output}: {exc}")
        in_handle.close()
        return 1

    try:
        n_total = 0
        n_kept = 0
        n_filtered = 0
        for line in in_handle:
            if not line:
                continue
            if line.startswith("#"):
                out_handle.write(line)
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            chrom = fields[0]
            n_total += 1
            if chrom in allowed_chroms or norm_chrom(chrom) in {
                norm_chrom(c) for c in allowed_chroms
            }:
                out_handle.write(line if line.endswith("\n") else line + "\n")
                n_kept += 1
            else:
                n_filtered += 1
    finally:
        in_handle.close()
        out_handle.close()

    log(
        f"{species_tag}Done. total={n_total} kept={n_kept} filtered={n_filtered} "
        f"-> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
