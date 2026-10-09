import csv
from pathlib import Path

from conftest import run_script, DATA_DIR


def test_pcsd_features(tmp_path):
    variants = DATA_DIR / "toy_pcsd" / "maize_candidates.tsv"
    pcsd_bed = DATA_DIR / "toy_pcsd" / "maize_toy_pcsd.bed"
    output_tsv = tmp_path / "pcsd_features.tsv"

    result = run_script(
        "chromatin_state/build_pcsd_state_features.py",
        [
            "--variants", str(variants),
            "--pcsd-bed", str(pcsd_bed),
            "--output", str(output_tsv),
        ],
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert output_tsv.exists()

    with open(output_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames

        pcsd_cols = [f"pcsd_E{i}" for i in range(1, 27)]
        for col in pcsd_cols:
            assert col in header, f"Missing column: {col}"
        assert "pcsd26_state_missing" in header

        rows = list(reader)
        assert len(rows) == 6

        for row in rows:
            onehot_sum = sum(int(row[f"pcsd_E{i}"]) for i in range(1, 27))
            assert onehot_sum in (0, 1), (
                f"Expected 0 or 1 one-hot states, got {onehot_sum}"
            )
