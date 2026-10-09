"Run the A2 nearby-gene and reference-window methods."
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nearest-window-bp", type=int, default=500000)
    parser.add_argument("--upstream-window-bp", type=int, default=2000)
    parser.add_argument("--flank-bp", type=int, default=500)
    args = parser.parse_args()
    os.environ["RICE_PH_AGENT_RESOURCE_FILE"] = str(args.registry.resolve())
    os.environ["SENTRY_DSN"] = ""
    from phenofuncrank_repro.interpretation.resources import registry
    from phenofuncrank_repro.interpretation.gene_context import _load_gff_genes, _find_nearest_gene, GENE_ANNOTATION_MAX_DIST_BP
    from phenofuncrank_repro.interpretation.promoter_extractor import extract_promoter, extract_flanking

    resource = registry()[args.species]
    genes = _load_gff_genes(resource["gff_path"])
    rows = []
    for variant in json.loads(args.variants.read_text(encoding="utf-8")):
        chrom, pos = str(variant["chrom"]), int(variant["pos"])
        nearest = _find_nearest_gene(chrom, pos, genes, window_bp=args.nearest_window_bp)
        rows.append({
            "variant": variant,
            "nearest_gene": nearest,
            "gene_annotation_eligible": bool(nearest and nearest["distance"] <= GENE_ANNOTATION_MAX_DIST_BP),
            "variant_upstream_reference": extract_promoter(
                args.species, chrom, pos, window_bp=args.upstream_window_bp
            ),
            "flanking_reference": extract_flanking(
                args.species, chrom, pos,
                upstream_bp=args.flank_bp, downstream_bp=args.flank_bp,
            ),
        })
    result = {
        "species": args.species,
        "parameters": {
            "nearest_window_bp": args.nearest_window_bp,
            "upstream_window_bp": args.upstream_window_bp,
            "flank_bp": args.flank_bp,
            "gene_annotation_max_distance_bp": GENE_ANNOTATION_MAX_DIST_BP,
        },
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
