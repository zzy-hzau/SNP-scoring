import numpy as np
import pandas as pd
import pytest

from phenofuncrank_repro.validation.matched_enrichment import bh_adjust, interval_membership, poisson_binomial_tail
from phenofuncrank_repro.common.reference_contract import require_same_assembly


def test_exact_upper_tail_retains_rare_joint_event():
    assert poisson_binomial_tail([1e-100, 1e-100], 2) == pytest.approx(1e-200, rel=1e-15, abs=0)
    assert poisson_binomial_tail([0.2, 0.3], 0) == 1.0
    assert poisson_binomial_tail([0.2, 0.3], 3) == 0.0


def test_gene_window_endpoints_and_overlapping_genes_count_each_snp_once():
    intervals = pd.DataFrame({"chrom": ["1", "1"], "start": [100, 120], "end": [150, 180]})
    observed = interval_membership(["1", "1", "1", "1", "2"], [90, 130, 190, 191, 130], intervals, window=10)
    assert np.array_equal(observed, [True, True, True, False, False])


def test_BH_preserves_tied_pvalues_in_complete_family():
    assert np.allclose(bh_adjust([0.01, 0.01, 0.4]), [0.015, 0.015, 0.4], rtol=0, atol=1e-16)


def test_distinct_reference_assemblies_are_not_silently_joined():
    with pytest.raises(ValueError, match="Coordinate mapping required"):
        require_same_assembly("SL4.0", "SL5.0")
