"""Statistics for the paper's paired frozen simulation experiments."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon

METHODS = ("T", "F", "B", "T_greedy", "M")
BUDGETS = (20, 50, 100)
METRICS = (
    "exact_qtn_hits", "samechrom_ld_qtn_covered", "distinct_ld_components",
    "max_samechrom_pair_r2", "mean_samechrom_pair_r2",
    "samechrom_pairs_r2ge06", "mean_pair_source_jaccard", "pairs_sharing_source",
)
CONTRASTS = (("B", "T"), ("M", "B"), ("T_greedy", "T"), ("M", "T_greedy"), ("F", "T"), ("M", "T"))


def mean_ci(values):
    x = np.asarray(values, float)
    n = len(x)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    delta = float(t.ppf(0.975, n - 1) * se) if n > 1 else 0.0
    return mean, mean - delta, mean + delta


def paired_p(diff):
    diff = np.asarray(diff, float)
    return 1.0 if np.all(np.abs(diff) < 1e-12) else float(
        wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)


def bh(values):
    p = np.asarray(values, float)
    order = np.argsort(p)
    q = np.empty(len(p))
    q[order] = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def summarize_simulation(detail: pd.DataFrame, n_replicates: int = 50):
    keys = ["trait", "rep", "method", "K"]
    if not detail.groupby(keys).size().eq(1).all():
        raise ValueError("Simulation records must be unique by trait/replicate/method/budget")
    rows = []
    for (trait, method, budget), group in detail.groupby(["trait", "method", "K"], sort=False):
        if len(group) != n_replicates:
            raise ValueError("A simulation cell is missing paired replicates")
        row = dict(trait=trait, method=method, K=int(budget), n_replicates=len(group))
        for metric in METRICS:
            mean, low, high = mean_ci(group[metric])
            row.update({metric + "_mean": mean, metric + "_ci95_low": low, metric + "_ci95_high": high})
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["trait", "K", "method"])
    rows = []
    for trait in ("Rice_PH", "Cotton_FS"):
        for budget in BUDGETS:
            block = detail[(detail.trait == trait) & (detail.K == budget)]
            for metric in METRICS:
                pivot = block.pivot(index="rep", columns="method", values=metric).sort_index()
                if list(pivot.index) != list(range(n_replicates)):
                    raise ValueError("Simulation pairing requires every original replicate identifier")
                for first, second in CONTRASTS:
                    diff = pivot[first].to_numpy(float) - pivot[second].to_numpy(float)
                    mean, low, high = mean_ci(diff)
                    rows.append(dict(
                        trait=trait, K=budget, metric=metric, contrast=first + "-" + second,
                        mean_paired_difference=mean, ci95_low=low, ci95_high=high,
                        p_wilcoxon_two_sided=paired_p(diff), wins=int((diff > 1e-12).sum()),
                        ties=int((np.abs(diff) <= 1e-12).sum()), losses=int((diff < -1e-12).sum())))
    comparisons = pd.DataFrame(rows)
    for name in ("q_bh_12_primary", "q_bh_12_greedy", "q_bh_6_M_vs_T"):
        comparisons[name] = np.nan
    for metric in ("exact_qtn_hits", "samechrom_ld_qtn_covered"):
        for column, contrasts, expected_count in (
            ("q_bh_12_primary", ("B-T", "M-B"), 12),
            ("q_bh_12_greedy", ("T_greedy-T", "M-T_greedy"), 12),
            ("q_bh_6_M_vs_T", ("M-T",), 6),
        ):
            chosen = (comparisons.metric == metric) & comparisons.contrast.isin(contrasts)
            if chosen.sum() != expected_count:
                raise ValueError("Multiple-testing family does not match the frozen paper experiment")
            comparisons.loc[chosen, column] = bh(comparisons.loc[chosen, "p_wilcoxon_two_sided"])
    return summary, comparisons
