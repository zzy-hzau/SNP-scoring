import csv
from pathlib import Path

from conftest import run_script, DATA_DIR


def test_cotton_regulatory(tmp_path):
    variants = DATA_DIR / "toy_peaks" / "cotton_candidates.tsv"
    atac_peaks = DATA_DIR / "toy_peaks" / "cotton_toy_atac.narrowPeak"
    h3k27ac_peaks = DATA_DIR / "toy_peaks" / "cotton_toy_h3k27ac.broadPeak"
    output_tsv = tmp_path / "regulatory_features.tsv"

    result = run_script(
        "regulatory/build_cotton_regulatory_features.py",
        [
            "--variants", str(variants),
            "--atac-peaks", str(atac_peaks),
            "--h3k27ac-peaks", str(h3k27ac_peaks),
            "--output", str(output_tsv),
        ],
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert output_tsv.exists()

    with open(output_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames
        expected_cols = [
            "Chr", "Pos",
            "atac_in_peak", "atac_dist_to_peak",
            "h3k27ac_in_peak", "h3k27ac_dist_to_peak",
        ]
        for col in expected_cols:
            assert col in header, f"Missing column: {col}"

        rows = list(reader)
        assert len(rows) == 6

        atac_in = [int(r["atac_in_peak"]) for r in rows]
        assert 1 in atac_in
        assert 0 in atac_in

        h3k_in = [int(r["h3k27ac_in_peak"]) for r in rows]
        assert 1 in h3k_in
        assert 0 in h3k_in
