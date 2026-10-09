#!/usr/bin/env python3
"M1 — build_safe_features.py"
from __future__ import annotations
from phenofuncrank_repro.common.paths import project_root, package_dir

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPTS_ROOT = package_dir('plantvimpact')
from phenofuncrank_repro.plantvimpact.common.config_utils import resolve_template


SAFE_FEATURES: List[str] = [
    "prot_safe_ref_aa_freq",
    "prot_safe_alt_aa_freq",
    "prot_safe_entropy",
    "prot_safe_delta_log_odds",
    "prot_safe_conservation_penalty",
    "prot_safe_abs_delta_log_odds",
    "prot_safe_alt_absent_flag",
]

DEFAULT_PSEUDOCOUNT = 1e-3


def load_yaml_config(path: str) -> Dict[str, Any]:
    try:
        import yaml  
    except ImportError as exc:  
        raise RuntimeError(
            "PyYAML is required to read --config but is not installed."
        ) from exc
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def resolve_local_script() -> str:
    """Resolve the SAFE implementation shipped in this repository."""
    local = Path(__file__).with_name("build_prot_safe_features_from_msa.py")
    return str(local) if local.is_file() else ""


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local build_prot_safe_features_from_msa.py "
            "that computes the 7 core protein-SAFE features (pseudocount=1e-3 "
            "for natural-log delta_log_odds only, Shannon entropy base 2). "
            "All paths from "
            "--config; no hardcoded paths."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="",
        help="Path to paths.yaml, used to resolve codebase.msa_root when needed.",
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Candidate variants TSV (Chr/Pos/Ref/Alt/codon_pos/ref_aa/alt_aa).",
    )
    parser.add_argument(
        "--msa-root", default=None,
        help="Directory of MSA FASTA files. If omitted, read from "
             "codebase.msa_root in config.",
    )
    parser.add_argument(
        "--alias-map", default=None,
        help="Optional transcript_id -> MSA header mapping TSV.",
    )
    parser.add_argument(
        "--species", required=True,
        help="Species name (written into the 'species' column of the output).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output parquet path.",
    )
    parser.add_argument(
        "--pseudocount", type=float, default=DEFAULT_PSEUDOCOUNT,
        help="Pseudocount epsilon for delta_log_odds ONLY (frequencies and "
             "entropy use raw n/depth with no pseudocount).",
    )
    parser.add_argument(
        "--aa-pos-offset", type=int, default=1,
        help="Optional 1-based amino-acid position offset (passed through).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the local command and exit without executing.",
    )
    return parser.parse_args(argv)


def build_command(args: argparse.Namespace, config: Dict[str, Any]) -> List[str]:
    local_script = resolve_local_script()
    if not local_script:
        raise RuntimeError(
            "The repository-local build_prot_safe_features_from_msa.py is missing."
        )

    msa_root = args.msa_root
    if not msa_root:
        msa_root = (config.get("codebase", {}) or {}).get("msa_root", "")
    if msa_root:
        msa_root = str(resolve_template(msa_root, config))
    if not msa_root:
        raise RuntimeError(
            "No MSA root: provide --msa-root or set codebase.msa_root in config."
        )

    cmd: List[str] = [
        sys.executable, local_script,
        "--candidates", args.candidates,
        "--msa-root", msa_root,
        "--species", args.species,
        "--output", args.output,
        "--pseudocount", repr(args.pseudocount),
        "--aa-pos-offset", str(args.aa_pos_offset),
    ]
    if args.alias_map:
        cmd += ["--alias-map", args.alias_map]
    return cmd


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    config: Dict[str, Any] = {}
    if args.config:
        if not os.path.isfile(args.config):
            print(f"ERROR: config file not found: {args.config}",
                  file=sys.stderr)
            return 2
        try:
            config = load_yaml_config(args.config)
        except Exception as exc:
            print(f"ERROR: failed to load config: {exc}", file=sys.stderr)
            return 2

    try:
        cmd = build_command(args, config)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


    print("[build_safe_features] 7 core features:", file=sys.stderr)
    print("  1. prot_safe_ref_aa_freq  = n_ref / depth   (no pseudocount)",
          file=sys.stderr)
    print("  2. prot_safe_alt_aa_freq  = n_alt / depth   (no pseudocount)",
          file=sys.stderr)
    print("  3. prot_safe_entropy      = -sum p*log2(p)  (Shannon, base 2)",
          file=sys.stderr)
    print(f"  4. prot_safe_delta_log_odds = ln((p_alt+eps)/(p_ref+eps)), "
          f"eps={args.pseudocount}", file=sys.stderr)
    print("  5. prot_safe_conservation_penalty = "
          "ln((p_ref+eps)/(p_alt+eps)) = -(delta_log_odds)",
          file=sys.stderr)
    print("  6. prot_safe_abs_delta_log_odds   = |delta_log_odds|",
          file=sys.stderr)
    print("  7. prot_safe_alt_absent_flag = 1 if p_alt==0 else 0",
          file=sys.stderr)

    if args.dry_run:
        print("[build_safe_features] DRY-RUN command:")
        print("  " + " ".join(cmd))
        return 0

    print("[build_safe_features] RUN: " + " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"[build_safe_features] local SAFE implementation failed (rc="
              f"{proc.returncode})", file=sys.stderr)
        return proc.returncode
    print(f"[build_safe_features] -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
