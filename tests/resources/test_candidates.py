import csv
from pathlib import Path

from conftest import run_script, DATA_DIR


def test_coding_candidates(tmp_path):
    fasta = DATA_DIR / "toy_genome" / "rice_toy.fa"
    gff = DATA_DIR / "toy_gff" / "rice_toy.gff3"
    output_tsv = tmp_path / "coding_candidates.tsv"

    result = run_script(
        "candidates/make_coding_candidates.py",
        [
            "--fasta", str(fasta),
            "--gff", str(gff),
            "--species", "rice",
            "--output", str(output_tsv),
            "--max-positions", "100",
        ],
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert output_tsv.exists()

    with open(output_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames
        expected_cols = [
            "species", "kind", "snp_id", "Chr", "Pos", "Ref", "Alt",
            "codon_pos", "ref_codon", "ref_aa", "alt_aa", "variant_type",
        ]
        for col in expected_cols:
            assert col in header, f"Missing column: {col}"

        rows = list(reader)
        assert len(rows) > 0
        assert all(row["kind"] == "coding" for row in rows)
        assert all(row["variant_type"] in ("synonymous", "missense", "nonsense")
                   for row in rows)


def test_noncoding_candidates(tmp_path):
    fasta = DATA_DIR / "toy_genome" / "rice_toy.fa"
    gff = DATA_DIR / "toy_gff" / "rice_toy.gff3"
    output_tsv = tmp_path / "noncoding_candidates.tsv"

    result = run_script(
        "candidates/make_noncoding_candidates.py",
        [
            "--fasta", str(fasta),
            "--gff", str(gff),
            "--species", "rice",
            "--output", str(output_tsv),
            "--max-positions", "100",
            "--promoter-window", "20",
        ],
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert output_tsv.exists()

    with open(output_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames
        assert "region_type" in header
        assert "Chr" in header
        assert "Pos" in header

        rows = list(reader)
        assert len(rows) > 0
        assert all(row["kind"] == "noncoding" for row in rows)
        region_types = set(row["region_type"] for row in rows)
        assert len(region_types) >= 1
