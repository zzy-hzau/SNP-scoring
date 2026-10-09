import csv
from pathlib import Path

from conftest import run_script, DATA_DIR


def test_ref_alt_check(tmp_path):
    vcf = DATA_DIR / "toy_vcf" / "rice_toy.vcf"
    fasta = DATA_DIR / "toy_genome" / "rice_toy.fa"
    output_tsv = tmp_path / "ref_check.tsv"

    result = run_script(
        "qc/check_ref_alt_against_fasta.py",
        [
            "--vcf", str(vcf),
            "--fasta", str(fasta),
            "--output", str(output_tsv),
            "--chromosomes", "chr01,chr02",
        ],
    )

    assert result.returncode in (0, 2), f"Script failed: {result.stderr}"
    assert output_tsv.exists()

    with open(output_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames
        assert "match" in header
        assert "vcf_ref" in header
        assert "fasta_ref" in header

        rows = list(reader)
        assert len(rows) >= 6
        for row in rows:
            assert row["match"] in ("True", "False")
            assert row["vcf_ref"] == row["fasta_ref"]


def test_filter_chromosomes(tmp_path):
    vcf = DATA_DIR / "toy_vcf" / "rice_toy.vcf"
    output_vcf = tmp_path / "filtered.vcf"

    result = run_script(
        "qc/filter_main_chromosomes.py",
        [
            "--vcf", str(vcf),
            "--output", str(output_vcf),
            "--chromosomes", "chr01",
        ],
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert output_vcf.exists()

    with open(output_vcf) as f:
        lines = [l for l in f if l.strip()]

    header_lines = [l for l in lines if l.startswith("#")]
    data_lines = [l for l in lines if not l.startswith("#")]

    assert len(header_lines) >= 2
    assert len(data_lines) >= 3
    for line in data_lines:
        fields = line.strip().split("\t")
        assert fields[0] == "chr01"
