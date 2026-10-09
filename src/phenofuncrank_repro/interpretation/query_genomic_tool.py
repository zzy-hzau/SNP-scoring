"Call a genomic annotation tool with caller-supplied arguments."
from __future__ import annotations
import argparse
import json
from pathlib import Path

TOOLS = (
    "blast_search", "quick_blastn", "interpro_scan", "plantregmap_search",
    "jaspar_motif_scan", "pfam_domain_for_protein", "resolve_to_uniprot",
    "fetch_protein_sequence",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", choices=TOOLS, required=True)
    parser.add_argument("--input", type=Path, required=True, help="JSON keyword arguments for the selected source function")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from phenofuncrank_repro.interpretation import live_apis
    result = getattr(live_apis, args.tool)(**json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"tool": args.tool, "result": result}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
