"""Environment-specific phenotype resources in a fixed genotype sample order."""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd


def prepare_environment_inputs(base_inputs, raw_phenotypes, output_dir, *,
                               sample_id_column="Lineid", environment_column="loc",
                               environment, trait_columns=("DTT", "PH", "EW")):
    base, out = Path(base_inputs), Path(output_dir)
    if (out / "metadata.json").exists() or (out / "matched_phenotypes.tsv").exists():
        raise FileExistsError("Environment output already exists; use a new output directory")
    original = pd.read_csv(base / "matched_phenotypes.tsv", sep="\t")
    raw = pd.read_csv(raw_phenotypes, sep="\t")
    chosen = raw.loc[raw[environment_column].astype(str).eq(str(environment))].copy()
    chosen[sample_id_column] = chosen[sample_id_column].astype(str)
    for column in trait_columns:
        chosen[column] = pd.to_numeric(chosen[column], errors="coerce")
    values = chosen.groupby(sample_id_column, sort=False)[list(trait_columns)].mean()
    result = original.copy()
    for column in trait_columns:
        result[column] = result[sample_id_column].astype(str).map(values[column]).to_numpy()
    result[environment_column] = str(environment)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "matched_phenotypes.tsv", sep="\t", index=False)
    source = json.loads((base / "metadata.json").read_text())
    allowed = ("species", "sample_id_col", "n_snps", "n_samples", "n_groups",
               "chromosomes", "dtype", "shape", "ped_allele_orientation",
               "chrom_label_mode", "multi_or_non_biallelic_variants")
    metadata = {k:source[k] for k in allowed if k in source}
    metadata.update(environment=str(environment), environment_column=environment_column,
                    phenotype_aggregation="mean within environment by sample identity",
                    phenotype_sample_order="unchanged base genotype order",
                    phenotypes="matched_phenotypes.tsv")
    for role, filename in (
            ("genotype_memmap","cultivar_genotype_codes.variant_by_sample.uint8.memmap"),
            ("snp_map","cultivar.snp_map.tsv.gz"),
            ("samples","matched_samples.vcf_order.txt")):
        metadata[role] = os.path.relpath(base / filename, out)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return result
