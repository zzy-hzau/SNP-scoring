from __future__ import annotations

import numpy as np

MIC_BATCH_SIZE = 5_000


def score_discrete_features_batched(geno, variant_indices, sample_cols, y, seed, selector, *, batch_size: int = MIC_BATCH_SIZE):
    """Score SNPs in deterministic batches without changing the active selector.

    ``minepy.cstats`` remains the sole implementation of paper-exact MIC.  The
    batching only bounds working memory and removes repeated per-SNP calls.
    """
    variant_indices = np.asarray(variant_indices, dtype=np.int64)
    values = []
    for start in range(0, len(variant_indices), int(batch_size)):
        ids = variant_indices[start:start + int(batch_size)]
        codes = np.asarray(geno[ids, :][:, sample_cols], dtype=np.uint8)
        if selector == "minepy_mic":
            try:
                from minepy import cstats
            except ImportError as exc:  
                raise RuntimeError("selector=minepy_mic requires minepy") from exc
            mic, _ = cstats(codes.astype(np.float64, copy=False), np.asarray(y, dtype=np.float64).reshape(1, -1), alpha=0.6, c=15, est="mic_approx")
            values.append(np.asarray(mic[:, 0], dtype=float))
        elif selector == "public_mi":
            from sklearn.feature_selection import mutual_info_regression
            values.append(mutual_info_regression(codes.T, np.asarray(y, dtype=float), discrete_features=True, n_neighbors=7, random_state=seed))
        else:
            raise ValueError(f"Unknown selector: {selector}")
    return np.concatenate(values) if values else np.empty(0, dtype=float)
