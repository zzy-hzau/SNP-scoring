#!/usr/bin/env python3
"M0 — verify_upstream.py"
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional


EXPECTED_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "candidate_universe_coding": {
        "required_cols": [
            "species", "kind", "snp_id", "Chr", "Pos", "Ref", "Alt",
            "codon_pos", "ref_codon", "ref_aa", "alt_aa", "variant_type",
        ],
        "min_cols": 0,
        "produced_by": "scripts/candidates/make_coding_candidates.py",
    },
    "candidate_universe_noncoding": {
        "required_cols": [
            "species", "kind", "snp_id", "Chr", "Pos", "Ref", "Alt",
        ],
        "min_cols": 0,
        "produced_by": "scripts/candidates/make_noncoding_candidates.py",
    },
    "callable_n": {
        "required_cols": ["Chr", "Pos", "callable_n"],
        "min_cols": 0,
        "produced_by": "scripts/callability/callability_from_gvcf.py",
    },
    "an_obs": {
        "required_cols": ["Chr", "Pos", "AN_obs"],
        "min_cols": 0,
        "produced_by": "scripts/callability/callability_from_gvcf.py",
    },
    "phastcons": {
        "required_cols": ["Chr", "Pos", "phastCons_score"],
        "min_cols": 0,
        "alternatives": [{"required_cols": [], "min_cols": 4}],
        "produced_by": "scripts/conservation/map_bedgraph_to_snps.py",
    },
    "phylop": {
        "required_cols": ["Chr", "Pos", "phyloP_score"],
        "min_cols": 0,
        "alternatives": [{"required_cols": [], "min_cols": 4}],
        "produced_by": "scripts/conservation/map_bedgraph_to_snps.py",
    },
    "protein_safe": {
        "required_cols": [
            "variant_id", "prot_safe_msa_depth",
            "prot_safe_ref_aa_freq", "prot_safe_alt_aa_freq",
            "prot_safe_entropy", "prot_safe_delta_log_odds",
            "prot_safe_conservation_penalty",
        ],
        "min_cols": 0,
        "produced_by": "scripts/feature_extraction/build_prot_safe_features_from_msa.py",
    },
    "paralog_redundancy": {
        "required_cols": [
            "gene_id", "paralog_redundancy", "has_paralog",
        ],
        "min_cols": 0,
        "produced_by": "scripts/redundancy/compute_paralog_redundancy.py",
    },
    "functional_redundancy": {
        "required_cols": ["gene_id", "functional_redundancy", "has_homolog"],
        "min_cols": 0,
        "produced_by": "scripts/redundancy/compute_homoeolog_redundancy.py",
    },
    "cotton_atac": {
        "required_cols": [],
        "min_cols": 3,
        "produced_by": "scripts/regulatory/build_cotton_regulatory_features.py",
    },
    "cotton_h3k27ac": {
        "required_cols": [],
        "min_cols": 3,
        "produced_by": "scripts/regulatory/build_cotton_regulatory_features.py",
    },
    "maize_pcsd": {
        "required_cols": ["Chr", "Pos"],
        "min_cols": 0,
        "alternatives": [{"required_cols": [], "min_cols": 4}],
        "produced_by": "scripts/chromatin_state/build_pcsd_state_features.py",
    },
    "regulatory_features": {
        "required_cols": ["Chr", "Pos"],
        "min_cols": 0,
        "produced_by": "scripts/regulatory/build_cotton_regulatory_features.py",
    },
}

COMMON_SPECIES_KEYS = [
    "candidate_universe_coding", "candidate_universe_noncoding",
    "callable_n", "an_obs",
]
SPECIES_REQUIRED_KEYS = {
    "rice": COMMON_SPECIES_KEYS + ["phastcons", "phylop", "protein_safe", "paralog_redundancy"],
    "maize": COMMON_SPECIES_KEYS + ["phastcons", "phylop", "protein_safe", "paralog_redundancy", "maize_pcsd"],
    "tomato": COMMON_SPECIES_KEYS + ["phastcons", "phylop", "protein_safe", "paralog_redundancy"],
    "chickpea": COMMON_SPECIES_KEYS + ["phastcons", "phylop", "protein_safe"],
    "cotton": COMMON_SPECIES_KEYS + ["protein_safe", "functional_redundancy", "regulatory_features"],
}
LEGACY_EXPECTED_KEYS = [
    "candidate_universe_coding", "candidate_universe_noncoding",
    "callable_n", "an_obs", "phastcons", "phylop", "protein_safe",
    "paralog_redundancy", "cotton_atac", "cotton_h3k27ac", "maize_pcsd",
]

QC_SCRIPT_MAP: Dict[str, str] = {
    "ref_alt_qc": "scripts/qc/check_ref_alt_consistency.py",
}


_INTERP_RE = re.compile(r"\{([a-zA-Z_][\w\.]*)\}")


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file, requiring PyYAML."""
    try:
        import yaml  
    except ImportError as exc:  
        raise RuntimeError(
            "PyYAML is required to read --config but is not installed."
        ) from exc
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(raw: str, config: Dict[str, Any]) -> str:
    """Resolve ``{section.key}`` placeholders against the config.

    Supports the interpolation syntax used in paths.template.yaml, e.g.
    ``{upstream_repo.root}/outputs/...`` and ``{output.root}/...``.
    Unknown placeholders are left untouched (and reported by the caller).
    """
    def _sub(match: re.Match) -> str:
        key = match.group(1)
        parts = key.split(".")
        node: Any = config
        for p in parts:
            if isinstance(node, dict) and p in node:
                node = node[p]
            else:
                return match.group(0)  
        return str(node)

    return _INTERP_RE.sub(_sub, raw)


def _read_header_fields(path: str) -> Optional[List[str]]:
    """Return the tab-split fields of the first non-comment line, or None."""
    try:
        with open(path, "r") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith("#"):
                    continue
                if line.startswith("track") or line.startswith("browser"):
                    continue
                return line.split("\t")
    except OSError:
        return None
    return None


def _looks_like_header(fields: List[str], required: List[str]) -> bool:
    """Heuristic: a line is a header if it contains at least one required col
    name as an exact string match."""
    field_set = set(fields)
    return any(col in field_set for col in required)


def _validate_schema_variant(fields: List[str], spec: Dict[str, Any]) -> tuple[bool, str | None]:
    required = spec.get("required_cols", []) or []
    min_cols = spec.get("min_cols", 0) or 0
    field_set = set(fields)
    if required:
        missing = [c for c in required if c not in field_set]
        if missing:
            return False, f"header missing required columns: {missing}"
        return True, None
    if min_cols and len(fields) < min_cols:
        return False, f"expected >= {min_cols} columns, found {len(fields)}"
    return True, None


def validate_schema(path: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a single upstream output file against its schema spec.

    Returns a result dict with keys:
        ok (bool), reason (str|None), header (list|None), n_rows_checked (int)
    """
    result: Dict[str, Any] = {
        "ok": False, "reason": None, "header": None, "n_rows_checked": 0,
    }
    if not os.path.isfile(path):
        result["reason"] = "file does not exist"
        return result

    fields = _read_header_fields(path)
    if fields is None:
        result["reason"] = "file is empty or unreadable"
        return result

    result["header"] = fields
    result["n_rows_checked"] = 1
    variants = [spec, *(spec.get("alternatives") or [])]
    errors: list[str] = []
    for variant in variants:
        ok, reason = _validate_schema_variant(fields, variant)
        if ok:
            result["ok"] = True
            result["reason"] = None
            return result
        if reason:
            errors.append(reason)
    result["reason"] = " or ".join(errors)
    return result


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify upstream plant-cadd-supplementary-methods outputs: "
            "check that every required file exists and matches its schema. "
            "All paths are read from --config (paths.yaml); no hardcoded paths."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to paths.yaml (copy of configs/plantvimpact/paths.template.yaml).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write the JSON verification report. "
             "If omitted, the report is printed to stdout.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero if any required file is missing or fails schema.",
    )
    parser.add_argument(
        "--upstream-root", default=None,
        help="Override the upstream_repo.root value from config "
             "(mostly for testing).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not os.path.isfile(args.config):
        print(f"ERROR: config file not found: {args.config}", file=sys.stderr)
        return 2

    try:
        config = load_yaml_config(args.config)
    except Exception as exc:
        print(f"ERROR: failed to load config {args.config}: {exc}",
              file=sys.stderr)
        return 2

    upstream = config.get("upstream_repo") or {}
    if not isinstance(upstream, dict):
        print("ERROR: 'upstream_repo' section is not a mapping in config",
              file=sys.stderr)
        return 2

    if args.upstream_root is not None:
        upstream = dict(upstream)
        upstream["root"] = args.upstream_root
        config = dict(config)
        config["upstream_repo"] = upstream

    root = upstream.get("root")
    if not root:
        print("ERROR: 'upstream_repo.root' not set in config", file=sys.stderr)
        return 2

    files: Dict[str, Any] = {}
    n_ok = 0
    n_fail = 0

    def verify_one(file_key: str, raw: Any, spec: Dict[str, Any], label: str) -> None:
        nonlocal n_ok, n_fail
        entry: Dict[str, Any] = {
            "config_key": label,
            "produced_by": spec.get("produced_by"),
        }
        if not raw:
            entry["path"] = None
            entry["exists"] = False
            entry["schema_ok"] = False
            entry["reason"] = "key not present in upstream_repo section"
            n_fail += 1
            files[label] = entry
            return

        resolved = resolve_path(str(raw), config)
        entry["path"] = resolved
        entry["exists"] = os.path.isfile(resolved)
        if not entry["exists"]:
            entry["schema_ok"] = False
            entry["reason"] = "file does not exist"
            n_fail += 1
            files[label] = entry
            return

        v = validate_schema(resolved, spec)
        entry["schema_ok"] = v["ok"]
        entry["reason"] = v["reason"]
        entry["header"] = v["header"]
        entry["n_rows_checked"] = v["n_rows_checked"]
        if v["ok"]:
            n_ok += 1
        else:
            n_fail += 1
        files[label] = entry

    species_mapping = upstream.get("species")
    if isinstance(species_mapping, dict):

        

        
        for species in sorted(SPECIES_REQUIRED_KEYS):
            species_paths = species_mapping.get(species)
            if not isinstance(species_paths, dict):
                for key in SPECIES_REQUIRED_KEYS[species]:
                    verify_one(key, None, EXPECTED_SCHEMAS[key], f"{species}.{key}")
                continue
            for key in SPECIES_REQUIRED_KEYS[species]:
                verify_one(
                    key,
                    species_paths.get(key),
                    EXPECTED_SCHEMAS[key],
                    f"{species}.{key}",
                )
    else:

        for key in LEGACY_EXPECTED_KEYS:
            spec = EXPECTED_SCHEMAS[key]
            verify_one(key, upstream.get(key), spec, key)

    report: Dict[str, Any] = {
        "module": "M0",
        "script": os.path.basename(__file__),
        "config": args.config,
        "upstream_root": root,
        "n_required": n_ok + n_fail,
        "n_ok": n_ok,
        "n_failed": n_fail,
        "files": files,
        "qc_scripts_documentation": QC_SCRIPT_MAP,
    }
    report["ok"] = (n_fail == 0)

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as handle:
            handle.write(payload + "\n")
        print(f"[verify_upstream] report -> {args.output}")
    else:
        print(payload)

    print(f"[verify_upstream] {n_ok}/{n_ok + n_fail} files OK, "
          f"{n_fail} failed", file=sys.stderr)

    if args.strict and n_fail > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
