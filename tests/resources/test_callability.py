from phenofuncrank_repro.common.paths import package_dir
import csv
from pathlib import Path

import pytest

from conftest import run_script, DATA_DIR

pysam = pytest.importorskip("pysam", reason="pysam not installed")


def test_callability_gvcf(tmp_path):
    gvcf = DATA_DIR / "toy_gvcf" / "rice_toy_sample1.g.vcf"
    gvcf_list = tmp_path / "gvcf_list.txt"
    with open(gvcf_list, "w") as f:
        f.write(str(gvcf) + "\n")

    output_tsv = tmp_path / "callability.tsv"

    result = run_script(
        "callability/callability_from_gvcf.py",
        [
            "--gvcf-list", str(gvcf_list),
            "--output", str(output_tsv),
            "--max-positions", "50",
        ],
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert output_tsv.exists()

    with open(output_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames
        assert "Chr" in header
        assert "Pos" in header
        assert "callable_n" in header
        assert "AN_obs" in header

        rows = list(reader)
        assert len(rows) > 0
        for row in rows:
            assert int(row["callable_n"]) >= 0
            assert int(row["AN_obs"]) >= 0


def test_callability_merge_and_qc(tmp_path):
    """End-to-end: per-sample callability -> merge -> denominator QC."""
    gvcf1 = DATA_DIR / "toy_gvcf" / "rice_toy_sample1.g.vcf"
    gvcf2 = DATA_DIR / "toy_gvcf" / "rice_toy_sample2.g.vcf"
    list_file = tmp_path / "gvcf_list.txt"
    with open(list_file, "w") as f:
        f.write(str(gvcf1) + "\n" + str(gvcf2) + "\n")

    per_sample = tmp_path / "callability.tsv"
    result = run_script(
        "callability/callability_from_gvcf.py",
        [
            "--gvcf-list", str(list_file),
            "--output", str(per_sample),
            "--chromosomes", "chr01",
            "--max-positions", "200",
        ],
    )
    assert result.returncode == 0, f"callability_from_gvcf failed: {result.stderr}"

    merged = tmp_path / "callability_merged.tsv"
    result = run_script(
        "callability/merge_callability_counts.py",
        ["--inputs", f"{per_sample},{per_sample}", "--output", str(merged)],
    )
    assert result.returncode == 0, f"merge_callability_counts failed: {result.stderr}"
    assert merged.exists()

    qc_report = tmp_path / "callability_qc.tsv"
    result = run_script(
        "callability/qc_callability_denominator.py",
        [
            "--input", str(merged),
            "--ploidy", "2",
            "--output", str(qc_report),
        ],
    )
    
    
    assert result.returncode == 0, f"qc_callability_denominator failed: {result.stderr}"
    with open(qc_report) as f:
        content = f.read()
        assert "qc_pass\tTrue" in content


def load_callability_reader():
    import importlib.util
    path = package_dir("resources") / "callability/callability_from_gvcf.py"
    spec = importlib.util.spec_from_file_location("callability_reader_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_small_gvcf(path, records, sample="one"):
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr01,length=100>\n"
        '##INFO=<ID=END,Number=1,Type=Integer,Description="End">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + sample + "\n"
        + "\n".join(records) + "\n"
    )


def test_reference_blocks_cover_all_declared_bases():
    reader = load_callability_reader()
    file = DATA_DIR / "toy_gvcf/rice_toy_sample1.g.vcf"
    rows = list(reader.iter_gvcf_callable(str(file), None))
    assert rows == [("chr01", pos, 2) for pos in range(1, 101)]
    clipped = list(reader.iter_gvcf_callable(str(file), None, ("chr01", 10, 23)))
    assert clipped == [("chr01", pos, 2) for pos in range(10, 24)]


def test_missing_and_partial_reference_genotypes(tmp_path):
    reader = load_callability_reader()
    file = tmp_path / "missing_partial.g.vcf"
    write_small_gvcf(file, [
        "chr01\t1\t.\tA\t<NON_REF>\t.\tPASS\tEND=4\tGT\t0/0",
        "chr01\t5\t.\tA\t<NON_REF>\t.\tPASS\tEND=10\tGT\t./.",
        "chr01\t11\t.\tA\t<NON_REF>\t.\tPASS\tEND=14\tGT\t0/.",
        "chr01\t15\t.\tATG\tA\t.\tPASS\t.\tGT\t0/1",
    ])
    rows = list(reader.iter_gvcf_callable(str(file), None))
    assert rows == (
        [("chr01", pos, 2) for pos in range(1, 5)]
        + [("chr01", pos, 1) for pos in range(11, 15)]
        + [("chr01", 15, 2)]
    )


def test_sample_position_is_counted_once(tmp_path):
    reader = load_callability_reader()
    file = tmp_path / "duplicate.g.vcf"
    record = "chr01\t1\t.\tA\t<NON_REF>\t.\tPASS\tEND=3\tGT\t0/0"
    write_small_gvcf(file, [record, record])
    inputs = tmp_path / "list.txt"
    inputs.write_text(str(file) + "\n")
    output = tmp_path / "calls.tsv"
    assert reader.main(["--gvcf-list", str(inputs), "--output", str(output)]) == 0
    with output.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [(int(r["Pos"]), int(r["callable_n"]), int(r["AN_obs"])) for r in rows] == [
        (1, 1, 2), (2, 1, 2), (3, 1, 2)
    ]


def test_duplicate_sample_does_not_inflate_panel_denominator(tmp_path):
    reader = load_callability_reader()
    file = tmp_path / "one.g.vcf"
    write_small_gvcf(file, ["chr01\t1\t.\tA\t<NON_REF>\t.\tPASS\tEND=3\tGT\t0/0"])
    inputs = tmp_path / "list.txt"
    inputs.write_text(str(file) + "\n" + str(file) + "\n")
    output = tmp_path / "calls.tsv"
    assert reader.main(["--gvcf-list", str(inputs), "--output", str(output)]) == 1
    assert not output.exists()
