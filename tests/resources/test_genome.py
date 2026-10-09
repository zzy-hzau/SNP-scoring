import json
import os
from pathlib import Path

from conftest import run_script, DATA_DIR


def test_standardize_fasta(tmp_path):
    input_fasta = DATA_DIR / "toy_genome" / "rice_toy.fa"
    output_fasta = tmp_path / "standardized.fa"

    result = run_script(
        "genome/standardize_fasta.py",
        [
            "--input", str(input_fasta),
            "--output", str(output_fasta),
            "--min-length", "50",
        ],
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert output_fasta.exists()

    with open(output_fasta) as f:
        content = f.read()
    assert ">chr01" in content
    assert ">chr02" in content
    assert len(content) > 0


def test_qc_cds(tmp_path):
    fasta = DATA_DIR / "toy_genome" / "rice_toy.fa"
    gff = DATA_DIR / "toy_gff" / "rice_toy.gff3"
    output_json = tmp_path / "qc_report.json"

    result = run_script(
        "genome/qc_cds_from_gff.py",
        [
            "--fasta", str(fasta),
            "--gff", str(gff),
            "--output", str(output_json),
            "--species", "rice",
        ],
    )

    assert result.returncode in (0, 2), f"Script failed: {result.stderr}"
    assert output_json.exists()

    with open(output_json) as f:
        report = json.load(f)

    assert "checks" in report
    assert "fasta_readable" in report["checks"]
    assert report["checks"]["fasta_readable"]["pass"] is True
    assert "overall_pass" in report
    assert "n_cds_records" in report
    assert report["n_cds_records"] >= 3
