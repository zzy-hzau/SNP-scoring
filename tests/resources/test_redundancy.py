import csv
from pathlib import Path

from conftest import run_script, DATA_DIR


def _parse_collinearity(tmp_path):
    collinearity = DATA_DIR / "toy_mcscanx" / "rice_toy.collinearity"
    pairs_tsv = tmp_path / "collinearity_pairs.tsv"

    result = run_script(
        "redundancy/parse_mcscanx_collinearity.py",
        [
            "--collinearity", str(collinearity),
            "--output", str(pairs_tsv),
        ],
    )
    assert result.returncode == 0, f"parse_mcscanx failed: {result.stderr}"
    assert pairs_tsv.exists()
    return pairs_tsv


def test_paralog_redundancy(tmp_path):
    pairs_tsv = _parse_collinearity(tmp_path)
    tpm_matrix = DATA_DIR / "toy_tpm" / "rice_toy_tpm.tsv"
    output_tsv = tmp_path / "paralog_redundancy.tsv"

    result = run_script(
        "redundancy/compute_paralog_redundancy.py",
        [
            "--collinearity-pairs", str(pairs_tsv),
            "--tpm-matrix", str(tpm_matrix),
            "--output", str(output_tsv),
        ],
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert output_tsv.exists()

    with open(output_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames
        expected_cols = [
            "gene_id", "paralog_redundancy", "has_paralog",
            "n_paralogs", "best_pair_expr_corr", "best_pair_breadth",
        ]
        for col in expected_cols:
            assert col in header, f"Missing column: {col}"

        rows = list(reader)
        assert len(rows) == 6

        gene_rows = {row["gene_id"]: row for row in rows}

        assert "Os01g0100100" in gene_rows
        assert "Os02g0100100" in gene_rows
        assert "Os01g0100200" in gene_rows
        assert "Os02g0100200" in gene_rows
        assert "Os02g0100300" in gene_rows

        row1 = gene_rows["Os01g0100100"]
        assert int(row1["has_paralog"]) == 1
        score1 = float(row1["paralog_redundancy"])
        assert -1.0 <= score1 <= 1.0

        row_neg = gene_rows["Os01g0100200"]
        assert int(row_neg["has_paralog"]) == 1
        score_neg = float(row_neg["paralog_redundancy"])
        assert score_neg < 0.0, f"Expected negative score, got {score_neg}"

        row_nopar = gene_rows["Os02g0100300"]
        assert int(row_nopar["has_paralog"]) == 0
        assert float(row_nopar["paralog_redundancy"]) == 0.0


def test_homoeolog_redundancy(tmp_path):
    """Check signed At-Dt correlation and zero scores for missing partners."""
    pairs = DATA_DIR / "toy_homoeolog" / "cotton_toy_homoeolog_pairs.tsv"
    tpm = DATA_DIR / "toy_homoeolog" / "cotton_toy_tpm.tsv"
    output_tsv = tmp_path / "homoeolog_redundancy.tsv"

    result = run_script(
        "redundancy/compute_homoeolog_redundancy.py",
        [
            "--homoeolog-pairs", str(pairs),
            "--tpm-matrix", str(tpm),
            "--output", str(output_tsv),
        ],
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert output_tsv.exists()

    with open(output_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames
        expected_cols = [
            "gene_id", "functional_redundancy", "has_homolog",
            "expr_corr", "breadth",
        ]
        for col in expected_cols:
            assert col in header, f"Missing column: {col}"

        rows = list(reader)
        assert len(rows) == 16

        gene_rows = {row["gene_id"]: row for row in rows}

        
        at1 = gene_rows["Gh_A01G0001"]
        assert int(at1["has_homolog"]) == 1
        assert float(at1["functional_redundancy"]) > 0.9

        
        
        at2 = gene_rows["Gh_A01G0002"]
        assert int(at2["has_homolog"]) == 1
        fr = float(at2["functional_redundancy"])
        assert fr < 0.0, f"Expected negative score for anti-correlated pair, got {fr}"

        
        at5 = gene_rows["Gh_A01G0005"]
        assert int(at5["has_homolog"]) == 1
        assert float(at5["functional_redundancy"]) == 0.0

        
        no_homolog = [g for g, r in gene_rows.items() if r["has_homolog"] == "0"]
        assert len(no_homolog) == 2
        for g in no_homolog:
            assert float(gene_rows[g]["functional_redundancy"]) == 0.0
