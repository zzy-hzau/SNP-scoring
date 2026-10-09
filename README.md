# PhenoFuncRank

Trait and functional scoring for sequential prioritization of crop variants.

## Overview

PhenoFuncRank combines genotype-based allele effects from DPCformer with
functional scores from PlantVImpact. Core SNPs supply trait support, LD screening
retains linked candidates, and sequential selection updates priorities as
candidates enter the selected set. The outputs include T, F, B, M and selected
candidate ranks.

The repository contains input preparation, model training and inference,
conservation and annotation feature processing, prioritization, statistical
validation and evidence-guided interpretation.

## Installation

### Tested environment

| Component | Version |
| --- | --- |
| Operating system | Ubuntu 24.04.3 LTS, x86-64 |
| Hardware | x86-64 CPU |
| Python | 3.10.20 |
| PyTorch | 2.6.0+cpu |
| NumPy | 1.26.4 |
| pandas | 2.3.3 |
| SciPy | 1.15.3 |
| XGBoost | 3.2.0 |
| scikit-learn | 1.7.2 |
| minepy | 1.2.6 |
| PyArrow | 25.0.1 |

The example runs on a CPU. Building minepy requires a C compiler and Python
development headers. Use Python 3.10 in an activated virtual or conda environment.

```sh
git clone https://github.com/zzy-hzau/SNP-scoring.git
cd SNP-scoring
python -m pip install setuptools==70.3.0 wheel numpy==1.26.4
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-build-isolation minepy==1.2.6
python -m pip install -c environments/example-requirements.txt '.[dpcformer-training]'
phenofuncrank --help
```

Method-specific dependencies are listed in `environments/` and the optional
dependency groups in `pyproject.toml`.

Installation in a fresh environment took approximately
70 seconds with package downloads cached.

## Example data

The [example archive](data/Example_data.tar.gz) contains a chickpea subset with
160 accessions and 384 SNPs on Ca5: 64 coding and 320 noncoding candidates.
The trait is hundred-seed weight. Prepared DNA/protein sequence, conservation
and annotation features accompany the genotype and phenotype inputs.

Each functional model has 2,048 training rows, 512 validation rows and 512 test
rows. Training uses Ca1–Ca4 and Ca7–Ca8; validation uses Ca6; testing uses Ca5.

Extract the archive from the repository root:

```sh
tar -xzf data/Example_data.tar.gz -C data
```

```text
data/example/
  dataset.json
  dpcformer/
    metadata.json
    cultivar.snp_map.tsv
    cultivar_genotype_codes.variant_by_sample.uint8.memmap
    matched_phenotypes.tsv
  plantvimpact/
    chickpea_coding/splits/fold05/{train,val,test}.parquet
    chickpea_noncoding/splits/fold05/{train,val,test}.parquet
  candidates/
    candidate_map.tsv
    allele_effect_targets.tsv
    coding.parquet
    noncoding.parquet
  expected_outputs/
```

### Input fields

| Input | Required contents |
| --- | --- |
| Genotype matrix | Variant-by-sample uint8; 0 = missing, 1 = REF/REF, 2 = REF/ALT, 3 = ALT/ALT |
| SNP map | `variant_index`, `Chr`, `chromosome`, `group_id`, `Pos`, `Ref`, `Alt`, `snp_id` |
| Phenotypes | One row per genotype sample, in matching order; `Hundred_seed_weight` |
| Functional training tables | Variant identifiers, `new_label` and profile-specific feature columns |
| Candidate feature tables | Variant identifiers, `chr_num` and profile-specific feature columns |

`Pos` is a 1-based genomic coordinate. The functional label is 0 for the matched
wild-observed class and 1 for the matched wild-unobserved class. Exact feature
lists are in `configs/examples/chickpea_coding.json` and
`configs/examples/chickpea_noncoding.json`.

## Run the complete example

```sh
python examples/run_example.py --data data/example --output results/example
```

The example performs these steps:

1. Train DPCformer with fold-local MIC selection and phenotype scaling.
2. Load the saved fold models and calculate allele-substitution effects.
3. Train coding and noncoding PlantVImpact models.
4. Predict candidate functional scores.
5. Select Core SNPs, screen LD support, calculate T/F/B and select candidates by M.

The CPU example completes in approximately 35 seconds after installation,
with peak process memory of approximately 860 MiB.

### Example configuration

| Setting | Value |
| --- | --- |
| DPCformer fixed test fraction | 0.10 |
| Inner training/validation folds | 10 |
| SNPs selected per chromosome group | 128 |
| Selection statistic | MIC, alpha = 0.6, c = 15 |
| DPCformer epochs / batch size | 3 / 32 |
| DPCformer learning rate / seed | 0.001 / 42 |
| Functional model fold | 5 |
| Functional boosting limit / early stopping | 3,000 / 100 |
| Core SNP count | 20 |
| LD support threshold | dosage r² ≥ 0.60 |
| Core-source grouping threshold | dosage r² ≥ 0.80 |
| Selected candidate budget | 10 |

### Run individual steps

Use the same output directory for consecutive steps:

```sh
python examples/run_example.py --step train-dpc --output results/example
python examples/run_example.py --step predict-dpc --output results/example
python examples/run_example.py --step train-function --output results/example
python examples/run_example.py --step score-function --output results/example
python examples/run_example.py --step rank --output results/example
```

Use a new output directory when starting another training run.

## Outputs

| Location under `results/example/` | Contents |
| --- | --- |
| `dpcformer/` | Fold checkpoints, selected SNP maps, phenotype scalers, validation metrics and test predictions |
| `allele_effects/` | Per-fold and aggregated allele-substitution effects |
| `plantvimpact/coding/`, `plantvimpact/noncoding/` | Trained XGBoost models, feature order, imputation values and metrics |
| `scores/coding.parquet`, `scores/noncoding.parquet` | Candidate functional probabilities and raw margins |
| `ranking/candidate_scores.tsv` | T, F, B, M at selection and M after selection |
| `ranking/selected_candidates.tsv` | The 10 selected variants in selection order |
| `run_time_all.json` | Time spent in each step |

Reference tables are in `data/example/expected_outputs/`. T is calibrated trait
support, F is the functional percentile within the coding/noncoding candidate
stratum, and B = √(T × F). M records the priority after accounting for genotype
correlation and Core-source support already represented by selected candidates.

## Training with your own data

### DPCformer

Prepare the genotype matrix, SNP map, matched phenotype table and metadata using
the input formats above. CSV/MAP input preparation is available through
`phenofuncrank dpc-inputs-csv`.

```sh
phenofuncrank dpc-train-diploid --inputs-dir INPUTS --outdir RESULTS --trait Hundred_seed_weight --pheno-col Hundred_seed_weight --selector minepy_mic
phenofuncrank dpc-report --metrics RESULTS/Hundred_seed_weight.fold_metrics.tsv --output RESULTS/fold_report.json
```

Use `dpc-train-cotton` for the cotton A/D grouping route. The species/trait
settings are in `configs/models/dpcformer_strict_training_profiles.json`.

### PlantVImpact training and inference

Place prepared `train.parquet`, `val.parquet` and `test.parquet` files in the
profile's `foldNN` directories. Set their relative locations under `resources`
in the selected model profile.

```sh
python -m phenofuncrank_repro.plantvimpact.strict_model train --profile configs/models/chickpea_coding_verified_existing.json --data-root INPUTS --out-dir MODELS/coding --device cpu --nthread 4
phenofuncrank pvi-score --input CANDIDATES.parquet --model-root MODELS/coding --output SCORES.parquet --folds 1-8
```

For chickpea noncoding, select
`configs/models/chickpea_noncoding_verified_existing.json` and its corresponding
input features. Other species profiles and model-training entries are in
`configs/models/` and the [command index](docs/commands.md).

### Feature preparation and ranking

Sequence-model inputs use
[PlantCaduceus_l32](https://huggingface.co/kuleshov-group/PlantCaduceus_l32/tree/e624c13c3d35415348b854c87a218893b23564f7)
and [ESM-1b](https://github.com/facebookresearch/esm#pre-trained-models).
Install the `sequence-features` extra for sequence scoring. The `coding-lm`
and `noncoding-lm` commands extract these features.
`dna-conservation` and `protein-msa` prepare conservation resources;
`upstream` prepares candidate and label inputs. Species configurations are in
`configs/`. Run `phenofuncrank COMMAND --help` for the input arguments.

The example runner shows how to connect allele effects, functional scores and
genotype LD to the ranking functions in
`src/phenofuncrank_repro/prioritization/paper_selection.py`.

## Web server

The [PhenoFuncRank web server](https://cotton-bigdata.hzau.edu.cn/phenofuncrank/)
provides the online interface.

## Methods and tests

Detailed definitions are in [docs/methods.md](docs/methods.md). Commands and input arguments are indexed in [docs/commands.md](docs/commands.md).

~~~sh
python -m pip install '.[test]'
python -m pytest tests -q
~~~

## Citation and license

Chickpea data source: Varshney et al., *Nature* 599, 622–627 (2021),
[doi:10.1038/s41586-021-04066-1](https://doi.org/10.1038/s41586-021-04066-1).
The example is a subset of the genotype/phenotype and prepared functional-feature
inputs used in PhenoFuncRank.

Code license: [MIT](LICENSE). Third-party notices are in `third_party/`.
Questions and bug reports can be submitted through the repository's GitHub issues.
