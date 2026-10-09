"""Export complete tested-term and matched-SNP inputs from supplied source archives."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ["publication_root", "spec_path", "term_catalog_path", "term_metadata_path"]:
        value = Path(config[key])
        if not value.is_absolute():
            value = args.config.resolve().parent / value
        config[key] = value
    from phenofuncrank_repro.validation.enrichment_inputs import public_inputs
    c, catalogue, background, matches, terms = public_inputs(
        config["publication_root"], config["trait"],
        json.loads(config["spec_path"].read_text()), config["term_catalog_path"],
        json.loads(config["term_metadata_path"].read_text()),
        frozen_endpoints=config.get("historical_coordinates", False),
    )
    expected = set(map(tuple, terms[["trait", "database", "term_id"]].drop_duplicates().to_numpy()))
    actual = set(map(tuple, catalogue[["trait", "database", "term_id"]].drop_duplicates().to_numpy()))
    if expected != actual:
        raise ValueError("The complete tested term family is not represented by the supplied gene catalogue")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("candidates", c), ("term_intervals", catalogue), ("backgrounds", background)]:
        frame = frame.copy()
        frame["assembly"] = config["assembly"]
        if name == "backgrounds":
            frame["trait"] = config["trait"]
        frame.to_csv(args.output_dir / (name + ".tsv"), sep="\t", index=False)
    matches.to_csv(args.output_dir / "matches.tsv", sep="\t", index=False)
    terms[["trait", "database", "term_id", "term"]].to_csv(args.output_dir / "tested_terms.tsv", sep="\t", index=False)
    request = {
        "assembly": config["assembly"],
        **{name + "_path": name + ".tsv" for name in ["candidates", "term_intervals", "backgrounds", "matches"]},
    }
    (args.output_dir / "enrichment_request.json").write_text(json.dumps(request, indent=2) + "\n")
    manifest = {"trait": config["trait"], "tested_terms": len(expected), "catalogue_rows": len(catalogue), "background_rows": len(background), "candidate_rows": len(c), "foreground_filter_applied": False, "historical_coordinates": config.get("historical_coordinates", False)}
    (args.output_dir / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
