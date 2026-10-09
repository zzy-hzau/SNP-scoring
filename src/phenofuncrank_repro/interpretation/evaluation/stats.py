"Paired statistics for benchmark comparisons (RPA ablation program)."

from __future__ import annotations

import math
import random
from typing import Sequence


def paired_bootstrap_ci(
    pairs: Sequence[tuple[float, float]],
    *,
    n_boot: int = 10000,
    seed: int = 20260809,
    alpha: float = 0.05,
) -> dict:
    """95% percentile bootstrap CI for mean(engine - baseline) per case."""
    deltas = [float(e) - float(b) for e, b in pairs]
    n = len(deltas)
    if n == 0:
        return {"n": 0, "mean_delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0,
                "bootstrap": n_boot, "seed": seed}
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int(n_boot * alpha / 2)
    hi_idx = int(n_boot * (1 - alpha / 2)) - 1
    return {
        "n": n,
        "mean_delta": round(sum(deltas) / n, 4),
        "ci95_low": round(means[lo_idx], 4),
        "ci95_high": round(means[hi_idx], 4),
        "bootstrap": n_boot,
        "seed": seed,
    }


def mcnemar(
    engine_hits: Sequence[bool],
    baseline_hits: Sequence[bool],
) -> dict:
    """Exact McNemar test for paired binary outcomes (two-sided)."""
    b01 = sum(1 for e, b in zip(engine_hits, baseline_hits) if e and not b)
    b10 = sum(1 for e, b in zip(engine_hits, baseline_hits) if b and not e)
    both = sum(1 for e, b in zip(engine_hits, baseline_hits) if e and b)
    neither = sum(1 for e, b in zip(engine_hits, baseline_hits) if not e and not b)
    n_discordant = b01 + b10
    if n_discordant == 0:
        p = 1.0
    else:
        try:
            from scipy.stats import binomtest
            p = float(binomtest(b01, n_discordant, 0.5, alternative="two-sided").pvalue)
        except Exception:
            k = min(b01, b10)
            p = 2.0 * sum(
                math.comb(n_discordant, i) * 0.5 ** n_discordant
                for i in range(k + 1)
            )
            p = min(1.0, p)
    return {
        "engine_only": b01,
        "baseline_only": b10,
        "both": both,
        "neither": neither,
        "n_discordant": n_discordant,
        "p_value": round(p, 6),
    }
