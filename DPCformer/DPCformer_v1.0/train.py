"""
DPCformer: Deep Phenotype prediction with CNN-Transformer for Genomic Selection.

Single unified training script supporting all ploidy levels:
  - Diploid (rice, tomato, chickpea): each chromosome processed independently
  - Tetraploid (cotton): chromosomes auto-paired into subgenome groups
  - Hexaploid (wheat, maize): chromosomes auto-grouped by subgenome (A/B/D)

Usage:
    python train.py
    (Edit the CONFIG section below to set your data paths)
"""

import os
import re
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from data_process.encode import (
    base_one_hot_encoding_8v_dif, genotype_to_dataframe,
    MICSelector, detect_separator,
)
from model.model_1v_depth import PhenotypePredictor_1v
from model.early_stop import EarlyStopping


# ╔══════════════════════════════════════════════════════════════╗
# ║                    CONFIGURATION                             ║
# ╚══════════════════════════════════════════════════════════════╝
OUTPUT_FILE_PATH = "sample_data/sample_genotype_phenotype.csv"
MAP_FILE_PATH = "sample_data/sample_map.map"

# Column names for phenotype and genotype in the CSV
PHENOTYPE_COLUMN = "Sucrose"       # change to your phenotype column name
GENOTYPE_COLUMN = "genotype"
INDIVIDUAL_COLUMN = "individual_id"

# Training hyperparameters
SELECTED_SNP_COUNT = 200      # SNPs per chromosome group (MIC selection)
NUM_EPOCHS = 300
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 20                 # early stopping
K_FOLDS = 10
TEST_RATIO = 0.1
INPUT_DIM = 8                 # one-hot encoding dimension (8 for diploid)
RANDOM_SEED = 42
OUTPUT_DIR = "output"
# ╔══════════════════════════════════════════════════════════════╗


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def parse_chromosome_key(chrom_name):
    """Extract (base_number, subgenome_letter) from chromosome name.
    
    Examples: 1 -> (1, ''), 'chr1A' -> (1, 'A'), 14 -> (14, '')
    For numeric-only cotton: 1-13 = At, 14-26 = Dt
    """
    s = str(chrom_name).strip()
    # Handle string labels like 'chr1A', 'chr2B'
    m = re.match(r'chr(\d+)([A-Za-z])?', s)
    if m:
        num = int(m.group(1))
        letter = m.group(2) or ''
        return num, letter
    # Handle plain integers
    try:
        num = int(s)
        # Cotton-style: 1-13 are At, 14-26 are Dt
        if 14 <= num <= 26:
            return num - 13, 'D'
        else:
            return num, ''
    except ValueError:
        return 0, s


def auto_group_chromosomes(chromosome_snp_counts):
    """Automatically group chromosomes by subgenome.
    
    For species with letter suffixes (wheat chr1A/B/D, maize chr1A/B/D):
      groups by base number -> {1: ['A','B','D'], 2: ['A','B','D'], ...}
    
    For plain numbers (rice 1-12, cotton 1-26):
      groups each chromosome independently except cotton (>13 implies Dt)
    """
    # Parse all chromosome names
    parsed = [(c, parse_chromosome_key(c)) for c in chromosome_snp_counts.keys()]
    
    # Group by base number
    groups = {}
    for chr_name, (base_num, letter) in parsed:
        if base_num not in groups:
            groups[base_num] = []
        groups[base_num].append((chr_name, letter, chromosome_snp_counts[chr_name]))
    
    # Sort groups by base number
    sorted_groups = []
    for base_num in sorted(groups.keys()):
        group_chrs = sorted(groups[base_num], key=lambda x: x[1])  # sort by letter
        chr_names = [g[0] for g in group_chrs]
        total_snps = sum(g[2] for g in group_chrs)
        sorted_groups.append((base_num, chr_names, total_snps))
    
    return sorted_groups


def preprocess_data(csv_path, map_path):
    """Load and preprocess genotype-phenotype data."""
    sep = detect_separator(csv_path)
    print(f"Detected separator: {repr(sep)}")
    
    # Read CSV
    merged_df_chunks = pd.read_csv(csv_path, sep=sep, chunksize=100)
    one_hot_genotype_list = []
    phenotypes_list = []
    
    print("Loading genotype data...")
    for i, chunk in enumerate(merged_df_chunks):
        print(f"  Processing chunk {i + 1}...", end='\r')
        chunk = chunk.dropna(subset=[GENOTYPE_COLUMN, PHENOTYPE_COLUMN])
        if len(chunk) == 0:
            continue
        genotype_df = genotype_to_dataframe(chunk[GENOTYPE_COLUMN])
        one_hot_genotype = base_one_hot_encoding_8v_dif(genotype_df).values
        one_hot_genotype_list.append(one_hot_genotype)
        phenotypes_list.append(chunk[PHENOTYPE_COLUMN].values)
    
    print("\nDone loading.")
    one_hot_genotype_array = np.concatenate(one_hot_genotype_list, axis=0)
    phenotypes_array = np.concatenate(phenotypes_list, axis=0).astype(np.float64)
    
    # Remove NaN phenotypes
    nan_mask = np.isnan(phenotypes_array)
    if np.any(nan_mask):
        print(f"Removing {np.sum(nan_mask)} NaN phenotype values")
        phenotypes_array = phenotypes_array[~nan_mask]
        one_hot_genotype_array = one_hot_genotype_array[~nan_mask]
    
    print(f"Data: {one_hot_genotype_array.shape[0]} samples, "
          f"{one_hot_genotype_array.shape[1]} one-hot features")
    
    # Standardize phenotypes
    scaler = StandardScaler()
    phenotypes_array = scaler.fit_transform(phenotypes_array.reshape(-1, 1)).flatten()
    
    # Read map file
    map_df = pd.read_csv(map_path, sep='\t', header=None,
                         names=['chromosome', 'snp_id', 'map', 'position'])
    snp_counts = map_df.groupby('chromosome').size().to_dict()
    
    print(f"Chromosomes: {len(snp_counts)}, SNPs: {sum(snp_counts.values())}")
    
    return one_hot_genotype_array, phenotypes_array, snp_counts, scaler


def prepare_grouped_tensors(genotype_array, phenotypes_array, snp_counts, k):
    """Group chromosomes, apply MIC selection, and build tensor."""
    groups = auto_group_chromosomes(snp_counts)
    print(f"\nChromosome groups ({len(groups)} groups):")
    for base_num, chr_names, total_snps in groups:
        print(f"  Group {base_num}: {chr_names} ({total_snps} SNPs)")
    
    # Build cumulative index map
    sorted_keys = sorted(snp_counts.keys())
    snp_indices = np.cumsum([0] + [snp_counts[c] for c in sorted_keys])
    snp_positions = {c: (snp_indices[i], snp_indices[i + 1]) 
                     for i, c in enumerate(sorted_keys)}
    
    # Encoding helpers
    snp_encoding_map = {
        "AA": 0, "AT": 1, "TA": 1, "AC": 2, "CA": 2,
        "AG": 3, "GA": 3, "TT": 4, "TC": 5, "CT": 5,
        "TG": 6, "GT": 6, "CC": 7, "CG": 8, "GC": 8, "GG": 9,
        "00": -1, "A0": -1, "0A": -1, "T0": -1, "0T": -1,
        "C0": -1, "0C": -1, "G0": -1, "0G": -1
    }
    base_mapping = {0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: '0', 5: '0', 6: '0', 7: '0'}
    
    grouped_data_list = []
    
    for base_num, chr_names, total_snps in groups:
        # Collect data for chromosomes in this group
        group_8d_parts = []
        for chr_name in chr_names:
            if chr_name not in snp_counts:
                continue
            count = snp_counts[chr_name]
            start_idx, end_idx = snp_positions[chr_name]
            data_8d = genotype_array[:, start_idx * 8: end_idx * 8].reshape(-1, count, 8)
            group_8d_parts.append(data_8d)
        
        if not group_8d_parts:
            continue
        
        concat_data = np.concatenate(group_8d_parts, axis=1)
        n_samples, n_snps, _ = concat_data.shape
        
        # Convert 8-dim to 1-dim SNP features for MIC
        snp_1d = np.zeros((n_samples, n_snps), dtype=int)
        for i in range(n_samples):
            for j in range(n_snps):
                indices = np.where(concat_data[i, j, :] == 1)[0]
                if len(indices) == 2:
                    b1 = base_mapping.get(indices[0], '0')
                    b2 = base_mapping.get(indices[1], '0')
                    base_pair = ''.join(sorted([b1, b2]))
                elif len(indices) == 1:
                    b = base_mapping.get(indices[0], '0')
                    base_pair = b + '0'
                else:
                    base_pair = '00'
                snp_1d[i, j] = snp_encoding_map.get(base_pair, 16)
        
        # MIC selection
        try:
            mic_selector = MICSelector(k=min(k, n_snps))
            mic_selector.fit(snp_1d, phenotypes_array)
            selected_indices = mic_selector.top_k_indices_
        except Exception:
            selected_indices = np.random.choice(n_snps, min(k, n_snps), replace=False)
        
        selected_data = concat_data[:, selected_indices, :]
        print(f"  Group {base_num}: selected {len(selected_indices)}/{n_snps} SNPs")
        
        # Pad to k
        pad_size = k - selected_data.shape[1]
        if pad_size > 0:
            pad = np.zeros((n_samples, pad_size, 8))
            selected_data = np.concatenate((selected_data, pad), axis=1)
        
        grouped_data_list.append(selected_data)
    
    # Stack: (samples, groups, k, 8)
    grouped_tensor = np.stack(grouped_data_list, axis=1)
    print(f"Final tensor shape: {grouped_tensor.shape}")
    
    return torch.tensor(grouped_tensor, dtype=torch.float32)


def train_and_evaluate(model, train_loader, val_loader, test_loader, 
                       fold, output_dir):
    """Train model for one CV fold."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10)
    early_stopping = EarlyStopping(patience=PATIENCE)
    
    log_file = os.path.join(output_dir, f"training_log_fold_{fold + 1}.txt")
    with open(log_file, "w") as f:
        f.write("Epoch,Train Loss,Val Loss,Val Pearson,Val Spearman\n")
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_val_pearson = -1.0
    best_model_state = None
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(targets.cpu().numpy().flatten())
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        val_pearson, _ = pearsonr(val_preds, val_targets)
        val_spearman, _ = spearmanr(val_preds, val_targets)
        
        scheduler.step(val_loss)
        early_stopping(val_loss)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{NUM_EPOCHS} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"r={val_pearson:.4f} ρ={val_spearman:.4f}")
        
        with open(log_file, "a") as f:
            f.write(f"{epoch + 1},{train_loss:.4f},{val_loss:.4f},"
                    f"{val_pearson:.4f},{val_spearman:.4f}\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_pearson = val_pearson
            best_model_state = model.state_dict()
        
        if early_stopping.early_stop:
            print(f"  Early stopping at epoch {epoch + 1}")
            break
    
    # Save best model
    model_path = os.path.join(output_dir, "saved_models",
                              f"best_model_fold_{fold + 1}.pth")
    torch.save(best_model_state, model_path)
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch'), plt.ylabel('Loss')
    plt.title(f'Fold {fold + 1} Loss')
    plt.legend()
    loss_dir = os.path.join(output_dir, "loss_plots")
    os.makedirs(loss_dir, exist_ok=True)
    plt.savefig(os.path.join(loss_dir, f"fold_{fold + 1}_loss.png"))
    plt.close()
    
    # Test evaluation
    model.load_state_dict(best_model_state)
    model.eval()
    test_loss = 0.0
    test_preds, test_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            test_preds.extend(outputs.cpu().numpy().flatten())
            test_targets.extend(targets.cpu().numpy().flatten())
    test_loss /= len(test_loader.dataset)
    test_pearson, _ = pearsonr(test_preds, test_targets)
    test_spearman, _ = spearmanr(test_preds, test_targets)
    
    print(f"  Fold {fold + 1} Test: Loss={test_loss:.4f}, "
          f"r={test_pearson:.4f}, ρ={test_spearman:.4f}")
    
    return {
        'fold': fold + 1,
        'val_loss': best_val_loss,
        'val_pearson': best_val_pearson,
        'test_loss': test_loss,
        'test_pearson': test_pearson,
        'test_spearman': test_spearman,
    }


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "saved_models"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "loss_plots"), exist_ok=True)
    
    # 1. Load and preprocess data
    print("=" * 60)
    print("DPCformer Training Pipeline")
    print("=" * 60)
    
    genotype_array, phenotypes_array, snp_counts, scaler = \
        preprocess_data(OUTPUT_FILE_PATH, MAP_FILE_PATH)
    
    # 2. Prepare grouped tensors
    grouped_tensors = prepare_grouped_tensors(
        genotype_array, phenotypes_array, snp_counts, SELECTED_SNP_COUNT)
    
    phenotypes_tensor = torch.tensor(phenotypes_array, dtype=torch.float32).reshape(-1, 1)
    
    # 3. Create dataset and splits
    dataset = TensorDataset(grouped_tensors, phenotypes_tensor)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(TEST_RATIO * dataset_size))
    train_val_indices, test_indices = indices[split:], indices[:split]
    
    test_dataset = Subset(dataset, test_indices)
    train_val_dataset = Subset(dataset, train_val_indices)
    
    num_groups = grouped_tensors.shape[1]
    group_snps_list = [SELECTED_SNP_COUNT] * num_groups
    
    print(f"\nSamples: {dataset_size} (train_val={len(train_val_indices)}, "
          f"test={len(test_indices)})")
    print(f"Groups: {num_groups}, SNPs per group: {SELECTED_SNP_COUNT}")
    
    # 4. Cross-validation
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_results = []
    best_test_pearson = -1.0
    best_fold_info = None
    
    print(f"\nStarting {K_FOLDS}-fold cross-validation...")
    print("-" * 60)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
        print(f"\n{'=' * 20} FOLD {fold + 1}/{K_FOLDS} {'=' * 20}")
        
        train_subset = Subset(train_val_dataset, train_idx)
        val_subset = Subset(train_val_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = PhenotypePredictor_1v(num_groups, group_snps_list, INPUT_DIM).to(device)
        
        result = train_and_evaluate(model, train_loader, val_loader, test_loader,
                                    fold, OUTPUT_DIR)
        cv_results.append(result)
        
        if result['test_pearson'] > best_test_pearson:
            best_test_pearson = result['test_pearson']
            best_fold_info = result
    
    # 5. Summary
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    
    metrics = {
        'Test Loss': [r['test_loss'] for r in cv_results],
        'Test Pearson r': [r['test_pearson'] for r in cv_results],
        'Test Spearman ρ': [r['test_spearman'] for r in cv_results],
    }
    
    for name, values in metrics.items():
        print(f"  {name}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    print(f"\n  Best fold: {best_fold_info['fold']} "
          f"(Pearson r={best_fold_info['test_pearson']:.4f}, "
          f"Spearman ρ={best_fold_info['test_spearman']:.4f})")
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "cross_validation_results.txt")
    with open(summary_path, "w") as f:
        f.write("DPCformer Cross-Validation Results\n")
        f.write("=" * 40 + "\n")
        for name, values in metrics.items():
            f.write(f"{name}: {np.mean(values):.4f} ± {np.std(values):.4f}\n")
        if best_fold_info:
            f.write(f"\nBest fold: {best_fold_info['fold']} "
                    f"(Pearson r={best_fold_info['test_pearson']:.4f})\n")
    
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("Done.")
