from phenofuncrank_repro.common.paths import package_dir
from pathlib import Path
import importlib.util
from types import SimpleNamespace

import pytest
pytest.importorskip("pyarrow")


def module():
    source = package_dir("resources") / "regulatory/build_cotton_atac_h3k27ac_features.py"
    spec = importlib.util.spec_from_file_location("regulatory_contracts", source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_recorded_peak_offset_and_project_absolute_summit_agree(tmp_path):
    m = module()
    
    base = ["A01", "2198976", "2199311",
            "SRR26894181_fiber_10DPA_ATAC_peak_36", "25", ".",
            "3.26874", "5.03312", "2.59461"]
    raw = tmp_path / "actual.narrowPeak"
    derived = tmp_path / "actual.feature_peaks.bed"
    raw.write_text("\t".join(base + ["150"]) + "\n")
    derived.write_text("\t".join(base + ["2199126", "SRR26894181", "fiber_10DPA", "rep1", "fiber"]) + "\n")
    a = m.normalize_peak_bed(raw, "atac", tmp_path / "raw", has_summit=True)
    b = m.normalize_peak_bed(derived, "atac", tmp_path / "derived", has_summit=True,
                             sample_col=11, tissue_col=14,
                             summit_coordinate_mode="absolute")
    assert a[2].read_text() == b[2].read_text() == "A01\t2199126\t2199127\n"


def test_canonical_tissue_metadata_is_preserved():
    m = module()
    assert [m.infer_tissue_group(x) for x in ["fiber", "reproductive", "vegetative"]] == [
        "fiber", "reproductive", "vegetative"
    ]
    assert m.infer_tissue_group("stigma") == "reproductive"
    assert m.infer_tissue_group("radicle") == "vegetative"


def test_output_cleanup_cannot_remove_input(tmp_path):
    m = module()
    root = tmp_path / "inputs"
    root.mkdir()
    sentinel = root / "retained.txt"
    sentinel.write_text("input")
    args = SimpleNamespace(in_root=str(root), out_root=str(root), overwrite=True,
                           atac_bed=str(tmp_path / "ATAC.bed"),
                           h3k27ac_bed=str(tmp_path / "H3.bed"))
    with pytest.raises(ValueError):
        m.prepare_output_root(args)
    assert sentinel.read_text() == "input"
