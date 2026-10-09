# Commands

Run from the repository root after installation.

```sh
phenofuncrank --help
phenofuncrank COMMAND --help
```

| Command | Purpose |
| --- | --- |
| `dpc-inputs-csv` | Prepare genotype and phenotype inputs from CSV and MAP files |
| `dpc-train-diploid` | Train DPCformer for diploid crop inputs |
| `dpc-train-cotton` | Train DPCformer for cotton inputs |
| `dpc-report` | Summarize fold metrics and the representative fold |
| `dna-conservation` | Prepare DNA conservation with Cactus and PHAST |
| `protein-msa` | Prepare protein alignments with OrthoFinder and FAMSA/MAFFT |
| `upstream` | Prepare candidate alleles, mutation-rate inputs and fold labels |
| `candidate-select` | Select candidate alleles and annotate transcript consequences |
| `enrichment-inputs` | Prepare SNP and annotation inputs for enrichment |
| `enrichment` | Calculate matched-background enrichment |
| `qtl-inputs` | Prepare QTL and genotype-interval inputs |
| `qtl` | Calculate reference-QTL recovery |
| `gwas-leads` | Select independent GWAS lead variants |
| `interpretation` | Run literature retrieval and evidence extraction |
| `evidence-roles` | Run hypothesis reasoning and critic evaluation |
| `callrate` | Calculate candidate callability from gVCF inputs |
| `candidate-positions` | Prepare candidate positions from reference annotations |
| `core-ld` | Prepare Core SNPs and LD support |
| `allele-effects` | Calculate allele-substitution effects |
| `primary-validation` | Calculate genotype-phenotype associations |
| `pc5-gwas` | Run PC-adjusted chickpea GWAS |
| `aap-molecular` | Process AAP protein and expression evidence |
| `aap-association` | Compare AAP genotype groups with PC adjustment |
| `pvi-train` | Train PlantVImpact LOCO models |
| `pvi-score` | Score candidate variants with PlantVImpact |
| `coding-lm` | Extract coding DNA/protein sequence-model features |
| `noncoding-lm` | Extract noncoding DNA sequence-model features |

Configuration files are in `configs/`; dependencies are in `environments/`
and `pyproject.toml`. Additional ranking and simulation APIs are in
`src/phenofuncrank_repro/prioritization/` and `simulation/`.
