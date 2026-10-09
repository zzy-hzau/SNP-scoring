# PhenoFuncRank

PhenoFuncRank provides computational methods for plant variant prioritization
and evidence-based interpretation. This repository accompanies the main text
and supplementary methods and includes PlantVImpact functional scoring,
allele-effect and LD analysis, candidate ranking, statistical validation, and
the research Agent components used for interpretation. Strict DPCformer
training, evaluation, and genotype-input preparation are included in this package.

Crop-specific input and parameter profiles cover rice, maize, tomato, chickpea,
and cotton. Select the profile matching your crop, reference assembly, and
analysis; inputs and parameters are not interchangeable across profiles.

- [Quick start](#quick-start): run a bundled CPU example.
- [Workflow overview](#workflow-overview): understand inputs and outputs.
- [Methods and commands](#methods-and-commands): find the relevant entry point.
- [Data and models](#data-and-models): prepare full-analysis resources.
- [Repository layout](#repository-layout), [reproducibility](#reproducibility-notes),
  and [citation and license](#citation-and-license).

## Workflow overview

The methods can be run separately with the required intermediate inputs. A
full analysis follows these connected stages:

```text
Multi-species genomes -> alignment -> evolutionary model -> phastCons / phyloP
Multi-species proteins -> orthogroups -> protein MSA -> protein constraint features
                                  |
Reference + annotation + candidate alleles + population/resource features
                                  |
             PlantVImpact labels -> fold training -> functional scores
                                  |
Trained trait models + genotypes -> allele effects -> Core SNP / LD support
                                  |
                 T / F / B / M candidate prioritization
                                  |
       Association / enrichment / GWAS-QTL validation and Agent interpretation
```

The Agent source includes supervisor planning, GraphRAG, retrieval and annotation
tools, evidence extraction and verification, evidence-state construction,
hypothesis reasoning, and an independent critic. The executable literature
route connects retrieval through reasoning and criticism; supervisor/graph
planning also has a separate entry. See the
[interpretation methods](docs/methods/interpretation.md) for their contracts.

## Quick start

Use Python 3.10 or newer. Extract the source archive (or clone the repository)
and enter the directory containing `pyproject.toml`. The following commands use
a POSIX shell; [the quick-start guide](docs/quickstart.md) also covers PowerShell.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install .
phenofuncrank --help
```

### Run a complete CPU example

The bundled AAP example compares hundred-seed weight across three genotype
groups, adjusting for the supplied PC1-PC5. It needs no model weights, reference
download, API credential, or external command-line tool.

```sh
phenofuncrank aap-association \
  --samples examples/data/cases/AAP/small_samples.tsv \
  --covariates examples/data/cases/AAP/small_PCs.tsv \
  --output results/quickstart/aap_association.json

python -m json.tool results/quickstart/aap_association.json
```

Inputs are two tab-separated tables:

| File | Required columns |
| --- | --- |
| `small_samples.tsv` | `sample_id`, `genotype`, `hundred_seed_weight_g` |
| `small_PCs.tsv` | `sample_id`, `PC1`, `PC2`, `PC3`, `PC4`, `PC5` |

The result contains adjusted group means, three pairwise contrasts with HC3
standard errors and Holm-adjusted P-values, an additive sensitivity analysis,
and input SHA-256 hashes. For the bundled files, these summary fields are:

```json
{
  "n": 72,
  "group_counts": {"G/G": 24, "G/A": 24, "A/A": 24},
  "PCs_refit": false,
  "PC_center_scale_ddof": 1
}
```

This is a compact method example, not a rerun of the full AAP population or the
paper's complete experiments. Detailed inputs and numerical definitions are in
the [validation methods](docs/methods/validation.md).

`python -m phenofuncrank_repro` is equivalent to `phenofuncrank`. Use
`phenofuncrank COMMAND --help` to inspect a method's actual arguments.

## Methods and commands

| Stage | Command or source entry | Guide |
| --- | --- | --- |
| Raw DNA and protein conservation resources | `dna-conservation`, `protein-msa` | [Resources](docs/methods/resources.md) |
| Reference positions, alleles, annotation, and population counts | `candidate-positions`, `candidate-select`, `callrate`, `upstream` | [Resources](docs/methods/resources.md), [PlantVImpact](docs/methods/plantvimpact.md) |
| PlantVImpact features, fold labels, training, and functional scores | `coding-lm`, `noncoding-lm`, `upstream`, `pvi-train`, `pvi-score` | [PlantVImpact](docs/methods/plantvimpact.md) |
| Strict DPCformer input preparation, training, and fold reporting | `dpc-inputs-csv`, `dpc-train-diploid`, `dpc-train-cotton`, `dpc-report` | [DPCformer](docs/methods/dpcformer.md) |
| Allele effects, Core SNP, LD, and T/F/B/M ranking | `allele-effects`, `core-ld`; ranking APIs in `prioritization/` | [Prioritization](docs/methods/prioritization.md) |
| Phenotype comparisons and AAP case | `primary-validation`, `aap-association`, `aap-molecular` | [Validation](docs/methods/validation.md) |
| Matched-background enrichment and GWAS/QTL comparison | `enrichment-inputs`, `enrichment`, `pc5-gwas`, `gwas-leads`, `qtl-inputs`, `qtl` | [Validation](docs/methods/validation.md) |
| Reported simulation and frozen-input refits | Module entries in `simulation/` | [Simulation](docs/methods/simulation.md) |
| Agent planning, evidence, reasoning, and criticism | `interpretation`, `evidence-roles`; planning/tool entries in `interpretation/` | [Interpretation](docs/methods/interpretation.md) |
| Interpretation-system evaluation | Module entries in `interpretation/evaluation/` | [Evaluation](docs/methods/evaluation.md) |

The [command index](docs/commands.md) lists dependencies. Study-specific ranking,
simulation, planning, and evaluation APIs are documented in their method guides;
they are not all subcommands of the top-level CLI.

## Data and models

| Resource | Included in this source package? | Preparation |
| --- | --- | --- |
| Small examples and input schemas | Yes | See [examples](examples/README.md). |
| Complete validation term catalogue, matching ranges, and QTL catalogue | Yes | See `data/validation/` and the [validation guide](docs/methods/validation.md). |
| Full genomes, GFF annotations, multi-species proteomes, and alignments | No | Follow the assemblies, provider URLs, and local paths in `configs/`; see [resource inputs](docs/resources.md). |
| Population genotypes, phenotypes, FASTQ/BAM/gVCF, and full background archives | No | Supply the matching original-format inputs and the study-specific profiles. |
| PlantCaduceus, ESM-1b, PlantVImpact, and DPCformer model checkpoints | No | Obtain or train the applicable weights; model identifiers and input conventions are in the method guides and `configs/models/`. |
| Live Agent API credentials and runtime stores | No | Configure your own environment and resource paths as described in the [interpretation guide](docs/methods/interpretation.md). |

Run examples from the repository root. When running an installed package from
another directory, set `PHENO_PROJECT_ROOT` to the extracted source directory
and pass explicit configuration/input paths. Profiles using
`${PHENO_REFERENCE_ROOT}` require that variable to point to your matching local
references. The [resource guide](docs/resources.md) describes these conventions.

### Choose the matching environment

The base install supports the CPU statistical example above. Additional methods
use separate extras or external tools:

| Task | Python extra or environment | External tools/resources |
| --- | --- | --- |
| Candidate/reference upstream methods | `upstream` | Matching references and annotations; method-specific tools |
| DNA/protein LM features | `sequence-features` | Separately supplied LM weights and a suitable model runtime |
| Agent interpretation | `interpretation`; `interpretation-dense` for dense retrieval | Embedding resources; credentials for live model/provider calls |
| Strict DPCformer training | `dpcformer-training` | Fold-local MIC via minepy; see the [training guide](docs/methods/dpcformer.md) |
| Current trait-model allele effects | `dpcformer` | Matching trained checkpoints and genotype mapping |
| Historical rice/cotton simulations | Matching files in `environments/` | Frozen inputs and the corresponding numerical runtime |
| Genome alignment and protein MSA | Upstream profiles | Cactus/HAL/PHAST, gffread, OrthoFinder, FAMSA/MAFFT |
| Raw reads and chickpea GWAS | Workflow/method-specific environment | fastp, bwa-mem2, samtools, GATK; PLINK and GEMMA for GWAS |

For example, install `python -m pip install '.[upstream]'` for the relevant
resource methods. Use separate environments for incompatible historical
Torch/NumPy versions; do not install all optional extras together. Whole-genome
and whole-proteome workflows have profile-specific compute budgets and require
their external scientific tools; the small CPU example does not establish their
hardware requirements. See [environments](environments/) and
[the dependency index](docs/commands.md).

## Repository layout

```text
src/phenofuncrank_repro/
  resources/        Reference, candidate, conservation, and protein resources
  plantvimpact/     Features, labels, fold training, and functional scoring
  prioritization/   Allele effects, Core SNP, LD, and candidate ranking
  validation/       Association, enrichment, GWAS/QTL, and AAP methods
  simulation/       Reported simulations and shared frozen-input refits
  interpretation/   Agent roles, tools, evidence, prompts, schemas, and evaluation
  models/           Strict DPCformer training, input preparation, reporting, and shared network
  common/, vendor/  Shared calculations and required dependency implementation
configs/            Crop/model/method profiles and explicit input paths
environments/       Method-specific dependency files
examples/           Compact method inputs and schemas
data/validation/    Complete required validation catalogues/background definitions
tests/              Mathematical, coordinate, and input-contract regression tests
workflows/          Paired-read alignment and variant-calling recipe
docs/               Quick start, commands, resources, and method specifications
third_party/        Third-party licenses and provenance
```

## Reproducibility notes

- Reference assembly, coordinate conventions, allele identity, fold membership,
  and model/resource versions are part of each method's input contract. Do not
  combine raw-conservation profiles with model-input profiles from another assembly.
- Chickpea GWAS uses the additive GEMMA linear model with intercept and PC1-PC5,
  without kinship or LOCO. PlantVImpact's functional-model LOCO is a separate method.
- Enrichment uses the complete 5,193-term family, including zero-hit terms,
  with the corresponding matched SNP backgrounds and coordinate profile.
- Strict DPCformer training and input preparation are included. The supplied
  archive's preprocessing order, feature-selector naming, and checkpoint-state
  handling are replaced by the native strict route. Its model architecture is
  preserved. Structure ablations and comparator training are not present in that
  archive and are not asserted to be part of this release. See [DPCformer](docs/methods/dpcformer.md).
- The available rice count producer is GT-only. It does not demonstrate
  application of the manuscript's DP/BQ/GQ thresholds; the distinction and
  original/corrected count modes are documented in the resource methods.
- Agent replay/no-response modes and live model calls are distinct. Executable
  examples and regression checks do not reproduce all historical LLM answers
  or certify a new full-system benchmark.

The method guides retain version-specific definitions and known scientific
limits. This source release covers reported computational methods; figure
layout, private deployment material, and private preparation records are outside
its scope.

## Citation and license

If you use these methods, cite the associated PhenoFuncRank manuscript and the
upstream tools/models used in your analysis. Software citation metadata is in
[CITATION.cff](CITATION.cff); method-specific source/model references are linked
in [the method guides](docs/methods.md).

The repository code is distributed under the [MIT license](LICENSE). Retained
third-party notices are in [third_party/](third_party/). External datasets and
model weights remain subject to their providers' licenses.
