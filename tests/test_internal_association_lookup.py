import pandas as pd
import pytest
from phenofuncrank_repro.validation.internal_pc5_ols_native import canonical_map, exact_variant_row


def test_coordinate_collision_cannot_cross_chromosome_or_alleles():
    table = pd.DataFrame({
        "Chr":["A01","D01","A01"], "Pos":[100,100,100],
        "Ref":["A","A","C"], "Alt":["G","G","T"], "variant_index":[0,1,2],
    })
    mapping = canonical_map(table,"cotton_ad")
    assert exact_variant_row(mapping,"D01:100_A_G","cotton_ad").variant_index == 1
    assert exact_variant_row(mapping,"A01:100_T_C","cotton_ad").variant_index == 2
    assert exact_variant_row(mapping,"A01:100_A_T","cotton_ad") is None


def test_ambiguous_compatible_rows_are_not_silently_first_picked():
    table = pd.DataFrame({
        "Chr":["Ca2","Ca2"], "Pos":[100,100],
        "Ref":["A","A"], "Alt":["G","G"], "variant_index":[0,1],
    })
    with pytest.raises(ValueError):
        exact_variant_row(canonical_map(table,"chickpea_ca"),"Ca2:100_A_G","chickpea_ca")
