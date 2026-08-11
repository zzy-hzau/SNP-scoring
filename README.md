# DPCformer: CNN–Transformer Deep Learning for Cross-species Genomic Prediction

DPCformer is a unified deep learning framework for genomic prediction (GP) that combines **residual convolutional neural networks (CNNs)** with **Transformer encoders** to predict phenotypic traits from genome-wide SNP data. A single training script automatically handles all ploidy levels — diploid, tetraploid, and hexaploid — by grouping chromosomes into subgenome units without species-specific configuration.

## Features

- **Hybrid CNN–Transformer**: Per-chromosome residual Conv1D blocks → Transformer self-attention → MLP prediction
- **Universal ploidy support**: diploid (rice, tomato, chickpea), tetraploid (cotton), hexaploid (wheat, maize)
- **Auto chromosome grouping**: chromosomes with letter suffixes (chr1A/B/D) or numeric cotton-style (1–26) are automatically grouped
- **MIC feature selection**: top-k SNPs selected per group via mutual information
- **10-fold cross-validation**: early stopping, LR scheduling, Pearson & Spearman correlation, MSE
- **Dual genotype format**: space-separated (`C C T T`) or concatenated (`CCTT`) alleles, auto-detected

## Model Architecture

```
Input (Samples × Groups × SNPs × 8)
        │
        ▼
┌──────────────────────────────┐
│ Per-Group CNN (shared)        │
│  ResConv1 (8→32, k=5)        │
│  ResConv2 (32→64, k=5)       │
│  ResConv3 (64→16, k=5)       │
│  [MaxPool + BatchNorm + DO]  │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│ Transformer Encoder           │
│  Positional encoding           │
│  1 layer, 4 heads, d_model=16 │
│  Cross-chromosome attention   │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│ MLP Head                      │
│  Flatten → 512 → ReLU → DO    │
│  128 → ReLU → DO → 1          │
└──────────────────────────────┘
```

## Project Structure

```
DPCformer/
├── README.md
├── LICENSE
└── dpc/
    ├── environment.yml
    ├── train.py                      # Unified training script (all species)
    ├── data_process/
    │   ├── __init__.py
    │   └── encode.py                 # Genotype encoding & MIC selector
    ├── model/
    │   ├── __init__.py
    │   ├── model_1v_depth.py         # DPCformer model architecture
    │   └── early_stop.py
    └── sample_data/                  # ⚠️ SMALL SUBSETS FOR TESTING ONLY
        ├── sample_genotype_phenotype.csv   # 50 maize individuals, 200 SNPs
        └── sample_map.map                  # 200 SNP loci (chr1A–chr7D)
```

> ⚠️ **Important**: Sample data contains **only 50 individuals and 200 SNPs** — insufficient for meaningful prediction. Use your own full datasets for real experiments.

## Installation

### Requirements
- Python 3.9+
- CUDA-capable GPU (recommended)
- Conda

```bash
git clone https://github.com/yourusername/DPCformer.git
cd DPCformer/dpc
conda env create -f environment.yml
conda activate dpcformer
```

### Core Dependencies

| Package | Version |
|---------|---------|
| Python | 3.9 |
| PyTorch | 2.3.1 |
| NumPy | ≥1.21 |
| Pandas | ≥1.3 |
| scikit-learn | ≥1.0 |
| SciPy | ≥1.7 |
| Matplotlib | ≥3.5 |

## Data Format

### Genotype-Phenotype CSV

Comma or tab-separated file with columns:

| Column | Description |
|--------|-------------|
| `individual_id` | Sample identifier |
| `genotype` | SNP alleles. Accepted: space-separated (`C C T T G G ...`) or concatenated (`CCTTGG...`). Two alleles per SNP |
| `phenotype` | Numeric trait value |

Example (comma-separated, concatenated):
```
individual_id,genotype,Sucrose
Sample_1,GGCCTTCCCCGGTTAAGG...,12.5
Sample_2,AAAACCTTCCAAGGGGAA...,11.3
```

### Map File (PLINK `.map`)

Tab-separated, no header, 4 columns:

| Column | Description |
|--------|-------------|
| `chromosome` | Integer or label with letter suffix (e.g. `chr1A`, `chr2D`, `1`, `14`) |
| `snp_id` | SNP identifier |
| `genetic_dist` | Usually 0 |
| `position` | Base-pair position |

## Quick Start

### Test with sample data
```bash
cd dpc
python train.py
```
Runs on tiny maize subset — pipeline verification only.

### Train on your data
Edit the `CONFIG` block at the top of `train.py`:
```python
OUTPUT_FILE_PATH = "path/to/your/genotype_phenotype.csv"
MAP_FILE_PATH = "path/to/your/map.map"
PHENOTYPE_COLUMN = "Sucrose"   # your trait column name
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SELECTED_SNP_COUNT` | 200 | SNPs per group via MIC |
| `NUM_EPOCHS` | 300 | Max epochs |
| `BATCH_SIZE` | 64 | Batch size |
| `LEARNING_RATE` | 0.001 | Adam LR |
| `PATIENCE` | 20 | Early stopping |
| `K_FOLDS` | 10 | CV folds |
| `TEST_RATIO` | 0.1 | Hold-out ratio |

## Output

```
output/
├── cross_validation_results.txt     # CV summary
├── saved_models/
│   └── best_model_fold_1.pth        # model checkpoints
└── loss_plots/
    └── fold_1_loss.png              # loss curves
```

## Supported Species

| Species | Ploidy | Example Traits |
|---------|--------|---------------|
| Rice | 2n=24 | Grain weight, plant height, heading date |
| Tomato | 2n=24 | Fruit weight, soluble solids |
| Chickpea | 2n=16 | 100-seed weight, plant height |
| Cotton | 4n=52 | Fiber length, strength, micronaire |
| Wheat | 6n=42 | Kernel length, width, flour yield |
| Maize | 2n=20 | Sucrose, catechin content |

## Citation

```bibtex
@article{dpcformer2025,
  title={DPCformer: A CNN-Transformer Deep Learning Framework for Genomic Prediction},
  author={...},
  journal={...},
  year={2025},
}
```

## License

MIT License — see [LICENSE](LICENSE).
