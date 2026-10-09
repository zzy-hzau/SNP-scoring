# Method definitions

## Genotypes and trait prediction

The model input is a variant-by-sample uint8 matrix: 0 denotes missing calls,
1 reference homozygotes, 2 heterozygotes and 3 alternative homozygotes. SNP maps
link matrix rows to assembly, chromosome, position and alleles. Each training
fold retains its marker map and phenotype scaler.

DPCformer fixes 10% of eligible samples as the test partition and applies
ten-fold cross-validation to the remainder. MIC selection and phenotype scaling
use fold-training samples. MIC uses alpha 0.6, c 15 and mic_approx. Selected
markers are ordered by genomic position. Cotton selects within A/D chromosome
pairs. Adam uses learning rate 0.001, batch size 64 and at most 300 epochs.
Validation-loss scheduling uses factor 0.1 and patience 10; early stopping uses
patience 20. Fold initialization uses seed + 1009 × fold index.

Validation loss selects each checkpoint and the representative fold. Fixed-test
evaluation follows selection. Allele perturbation averages over retained folds
whose selected-marker maps contain the focal SNP.

## Functional features and labels

Candidate alleles are defined from assembly-matched FASTA and GFF annotations.
Noncoding regions comprise gene bodies outside CDS and strand-specific 2-kb
promoters. Mutation-rate estimates use fourfold-degenerate sites, context,
alternative allele and chromosome; cotton also separates subgenomes.

M0 samples callable unobserved alternatives, M1 matches context and alleles,
M2 adds mutation opportunity and M3 adds callability. Feasible M3 strata sample
observed and unobserved alleles at 1:1. Class 0 denotes wild-observed alleles;
class 1 denotes matched wild-unobserved alleles.

DNA conservation uses Cactus/HAL/MAF, phyloFit REV, phastCons target coverage 0.25
and expected length 45, and phyloP LRT/CONACC. Protein constraint features use
target-containing orthogroups and untrimmed MSAs. Column frequencies include
the focal sequence and standard amino acids; entropy uses log2 and substitution
log odds use natural logarithms with epsilon 0.001.

Sequence-model features use PlantCaduceus_l32 and ESM-1b. The feature is the
absolute masked contrast abs(log P(ALT) - log P(REF)). Coding contexts use
strand-aware CDS reconstruction. Resource and crop profiles specify the
assembly and feature columns.

## PlantVImpact training and inference

LOCO holds out one chromosome for test and a separate chromosome for validation.
Cotton holds out the corresponding A/D pair together. Fold counts are 12 rice,
10 maize, 13 cotton, 12 tomato and 8 chickpea.

Rice, cotton, tomato and chickpea use depth-4 binary-logistic histogram XGBoost,
learning rate 0.03, minimum child weight 20, row/column subsampling 0.9,
L2 penalty 1, L1 penalty 0 and max_bin 64. Validation AUCPR early stopping has
patience 100 and a 3000-round cap.

Maize coding uses depth 1, learning rate 0.001, minimum child weight and L2
penalty 5000, a 500-round cap, specified monotonic constraints and the
DNA/protein initial margin. Maize noncoding uses depth 3, learning rate 0.02,
minimum child weight 50, L2 penalty 100, specified monotonic constraints and
the DNA initial margin.

Rice uses training medians with zero for entirely missing training columns.
Cotton and tomato use training medians and zero for the redundancy columns
under zero_no_homolog. Chickpea uses training medians for its retained features.
Maize passes nonfinite values to XGBoost as NaN. Prediction uses the selected
tree prefix. Model profiles contain the feature order, constraints and margins.

## Core SNP, LD and sequential ranking

The allele effect is half the alternative-homozygote minus reference-homozygote
prediction difference in phenotype units, averaged over evaluable folds.
Initial Core SNPs are the upper 5% by absolute real-allele effect, rounded upward.
Core and proxy eligibility uses call rate at least 0.90 and MAC at least 10.
Proxy support uses same-chromosome dosage r² at least 0.60 across the chromosome.

Core absolute effects use empirical midrank percentiles. Core-source groups
are connected components at r² at least 0.80. Group support is the maximum
product of Core percentile and candidate-Core r². T is maximum group support;
F is the functional percentile within the relevant candidate stratum.

B = sqrt(T × F). Source overlap uses weighted Jaccard of normalized support
vectors. Genotype redundancy is the maximum evaluated r² with selected SNPs;
source load sums overlaps with the selected set. The selection score is
M = B × (1 - redundancy) / sqrt(1 + source load). Both loads start at zero.

Selection continues to 500 candidates. Top20, Top50 and Top100 are prefixes
of that sequence. Ties follow B, min(T,F), T, absolute effect and F descending,
then normalized chromosome, position, REF, ALT and SNP identifier.

## Validation

Top500 association tests use Welch comparisons and additive-dosage models,
including PC1–5 adjustment. Bonferroni uses the fixed family 500.

Matched GO/KEGG enrichment matches chromosome, nearest-gene-distance bins
(2, 10, 50, 200 kb) and MAF within 0.01. Term-member intervals extend 100 kb
with inclusive endpoints. The exact Poisson-binomial upper tail uses per-query
matched-background hit probabilities. The catalogue retains all 5193 terms;
BH adjustment applies within trait and annotation source.

Chickpea GWAS uses MAF at least 0.05, missing-genotype fraction at most 0.05,
five PCs from the QC marker panel and fixed-effect GEMMA. Leads are clumped
within 500 kb at dosage r² greater than 0.60. Lead-specific LD intervals use
that window for QTL overlap. Paired recovery uses one-sided exact McNemar/binomial
tests and BH across budgets within overlap definition.

Simulations use 50 replicates per trait and five QTNs. Phenotype variance
components are QTN 0.4, polygenic 0.1 and noise 0.5, followed by standardization.
Refits use the real-data selected-marker maps. Capture uses same-chromosome
r² at least 0.60. Paired Wilcoxon tests use a normal approximation and tie
correction; BH covers 12 trait/budget/contrast tests.

AAP uses PC-adjusted genotype comparisons with HC3 covariance and Holm
adjustment, plus the reported sequence, domain and expression analyses.

## Evidence-guided interpretation

The supervisor plans candidate-scoped evidence tasks. Retrieval and annotation
tools populate typed evidence records. Variant identity, quoted support and
source relationships determine admission to the evidence state. The reasoning
Agent builds evidence-linked conclusions; the critic reviews them and identifies
evidence gaps. GraphRAG supplies candidate-centred graph context and retrieval
priorities. Evaluation modules calculate the reported decision and quality
metrics from supplied labels and protocol definitions.
