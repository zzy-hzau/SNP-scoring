"Command interface for PhenoFuncRank."
from __future__ import annotations
import argparse
import runpy
import sys

COMMANDS = {
    "dpc-inputs-csv": "phenofuncrank_repro.models.dpcformer.inputs_csv",
    "dpc-train-diploid": "phenofuncrank_repro.models.dpcformer.train_diploid",
    "dpc-train-cotton": "phenofuncrank_repro.models.dpcformer.train_cotton",
    "dpc-report": "phenofuncrank_repro.models.dpcformer.report_folds",
    "dna-conservation": "phenofuncrank_repro.resources.conservation_upstream.dna",
    "protein-msa": "phenofuncrank_repro.resources.conservation_upstream.protein",
    "upstream": "phenofuncrank_repro.plantvimpact.run_native_upstream",
    "candidate-select": "phenofuncrank_repro.plantvimpact.select_candidate_alleles",
    "enrichment-inputs": "phenofuncrank_repro.validation.prepare_enrichment_inputs",
    "enrichment": "phenofuncrank_repro.validation.matched_enrichment",
    "qtl-inputs": "phenofuncrank_repro.validation.prepare_qtl_inputs",
    "qtl": "phenofuncrank_repro.validation.qtl_recovery_20260927",
    "gwas-leads": "phenofuncrank_repro.validation.chickpea_gwas_leads",
    "interpretation": "phenofuncrank_repro.interpretation.run_literature_pipeline",
    "evidence-roles": "phenofuncrank_repro.interpretation.run_roles",
    "callrate": "phenofuncrank_repro.resources.callability.callrate_from_candidate_gvcf",
    "candidate-positions": "phenofuncrank_repro.resources.candidates.build_native_position_universe",
    "core-ld": "phenofuncrank_repro.prioritization.downstream_cli",
    "allele-effects": "phenofuncrank_repro.prioritization.allele_perturbation",
    "primary-validation": "phenofuncrank_repro.validation.primary_top500_associations",
    "pc5-gwas": "phenofuncrank_repro.validation.chickpea_pc5_gwas",
    "aap-molecular": "phenofuncrank_repro.validation.aap_molecular",
    "aap-association": "phenofuncrank_repro.validation.aap_categorical_hc3",
    "pvi-train": "phenofuncrank_repro.plantvimpact.model_training.train_xgb_loco",
    "pvi-score": "phenofuncrank_repro.plantvimpact.scoring_oe.score_loco_table",
    "coding-lm": "phenofuncrank_repro.plantvimpact.feature_extraction.score_coding_lm",
    "noncoding-lm": "phenofuncrank_repro.plantvimpact.feature_extraction.score_noncoding_lm",
}


def main():
    if len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        module = COMMANDS[sys.argv[1]]
        sys.argv = [module, *sys.argv[2:]]
        runpy.run_module(module, run_name="__main__")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.parse_args()


if __name__ == "__main__":
    main()
