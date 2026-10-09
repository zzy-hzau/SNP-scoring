"""Paired statistics (bootstrap CI + McNemar) for benchmark comparisons."""

from __future__ import annotations

from phenofuncrank_repro.interpretation.evaluation.stats import (
    mcnemar,
    paired_bootstrap_ci,
)


def test_paired_bootstrap_ci_constant_delta():
    pairs = [(i + 2.0, float(i)) for i in range(20)]
    out = paired_bootstrap_ci(pairs, n_boot=2000, seed=1)
    assert out["mean_delta"] == 2.0
    assert out["ci95_low"] <= 2.0 <= out["ci95_high"]
    assert out["n"] == 20


def test_mcnemar_extreme_discordance_is_significant():
    engine = [True] * 15 + [False] * 5
    baseline = [False] * 15 + [True] * 5
    out = mcnemar(engine, baseline)
    assert out["engine_only"] == 15
    assert out["baseline_only"] == 5
    assert out["n_discordant"] == 20
    assert out["p_value"] < 0.05


def test_mcnemar_tied_returns_p_one():
    engine = [True, False, True, False]
    baseline = [True, False, True, False]
    out = mcnemar(engine, baseline)
    assert out["n_discordant"] == 0
    assert out["p_value"] == 1.0
