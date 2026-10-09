from phenofuncrank_repro.common.paths import package_dir
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("pysam")


def test_indexed_windows_preserve_boundary_mismatch_and_candidate_policy():
    path = package_dir("resources") / "regulatory/cotton_motif/08b_scan_motif_torch_shard.py"
    spec = importlib.util.spec_from_file_location("motif_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sequence = "ACGTNACGTACGT"
    matrix = np.array([[2,-1,-1,-1],[-1,2,-1,-1]], dtype=np.float32)
    motif = {"length":2, "log_odds":matrix,
             "rc_log_odds":matrix[::-1,:][:,[3,2,1,0]].copy(),
             "min_score":np.float32(-2), "denom":np.float32(6), "family":"MYB"}
    groups = module.make_motif_groups([motif], "cpu")
    frame = pd.DataFrame({
        "snp_id":["boundary","valid","mismatch","not_candidate"],
        "Pos":[1,7,9,11], "Ref":["A","C","A","C"], "Alt":["T","G","C","G"],
        "motif_candidate":[1,1,1,0],
        "atac_overlap_any":[1,1,1,1], "h3k27ac_overlap_any":[1,1,1,1],
        "active_enhancer_like_flag":[0,0,0,0], "promoter_with_activity":[1,1,1,1],
    })

    class IndexedReference:
        def get_reference_length(self, chrom):
            return len(sequence)
        def fetch(self, chrom, start, end):
            return sequence[start:end]

    original = module.process_chunk(frame, module.encode_chrom_seq(sequence),
                                     groups,2,0.8,0.1,"cpu")
    indexed = module.process_indexed_chunk(frame, IndexedReference(),"chr",
                                           groups,2,0.8,0.1,"cpu")
    pd.testing.assert_frame_equal(indexed, original, check_exact=True)
    assert indexed.loc[0,"motif_scanned"] == 0
    assert indexed.loc[1,"motif_scanned"] == 1
    assert indexed.loc[2,"motif_ref_mismatch"] == 1
    assert indexed.loc[3,"motif_scanned"] == 0
