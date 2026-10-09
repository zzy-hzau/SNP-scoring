"""Report the validation-selected fold and fixed-test metrics."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def summarize_folds(metrics: pd.DataFrame) -> dict:
    required = {'fold', 'best_val_loss', 'test_pearson'}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f'Missing fold metrics: {sorted(missing)}')
    if metrics.empty or not metrics['fold'].is_unique:
        raise ValueError('Supply nonempty metrics with one row per fold')
    data = metrics.copy()
    data['fold'] = pd.to_numeric(data['fold'], errors='raise').astype(int)
    for column in ['best_val_loss', 'test_pearson']:
        data[column] = pd.to_numeric(data[column], errors='raise')
    finite_test = data[np.isfinite(data.test_pearson)]
    finite_validation = data[np.isfinite(data.best_val_loss)]
    if finite_test.empty or finite_validation.empty:
        raise ValueError('No finite test PCC or validation loss is available')
    canonical = finite_validation.sort_values(['best_val_loss', 'fold']).iloc[0]
    representative = canonical
    return {
        'folds_supplied': len(data),
        'finite_test_PCC_folds': len(finite_test),
        'all_test_PCCs_finite': len(finite_test) == len(data),
        'Mean_Test_PCC': float(finite_test.test_pearson.mean()),
        'SD_Test_PCC_ddof0': float(finite_test.test_pearson.std(ddof=0)),
        'Selected_Test_PCC': float(representative.test_pearson) if np.isfinite(representative.test_pearson) else None,
        'representative_fold_by_validation_loss': int(representative.fold),
        'canonical_fold_by_validation_loss': int(canonical.fold),
        'checkpoint_selection_within_fold': 'validation loss',
        'representative_selection_scope': 'minimum validation loss',
        'representative_uses_test_performance': False,
        'required_training_protocol': 'Test is excluded from selection, scaler fitting, optimization, and checkpoint selection.',
        'fixed_test_identity_checked_from_this_metrics_table': False,
        'artifact_binding': 'Use the representative fold checkpoint, SNP map, phenotype scaler, and split together. Allele effects retain the separately documented all-fold policy.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize_folds(pd.read_csv(args.metrics, sep='\t'))
    result['metrics_sha256'] = hashlib.sha256(args.metrics.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
