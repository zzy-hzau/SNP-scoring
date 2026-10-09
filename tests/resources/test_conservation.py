import csv
from pathlib import Path

from conftest import run_script, DATA_DIR


def _make_snps_file(tmp_path):
    snps_file = tmp_path / "snps.tsv"
    with open(snps_file, "w") as f:
        f.write("Chr\tPos\tRef\tAlt\n")
        f.write("chr01\t5\tC\tA\n")
        f.write("chr01\t25\tT\tG\n")
        f.write("chr01\t50\tG\tC\n")
        f.write("chr01\t70\tC\tA\n")
        f.write("chr02\t15\tG\tC\n")
        f.write("chr02\t50\tG\tA\n")
    return snps_file


def test_map_bedgraph(tmp_path):
    snps_file = _make_snps_file(tmp_path)
    bedgraph = DATA_DIR / "toy_bedgraph" / "rice_toy_phastcons.bedgraph"
    output_tsv = tmp_path / "phastcons_scores.tsv"

    result = run_script(
        "conservation/map_bedgraph_to_snps.py",
        [
            "--snps", str(snps_file),
            "--bedgraph", str(bedgraph),
            "--output", str(output_tsv),
            "--track-name", "phastCons",
        ],
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert output_tsv.exists()

    with open(output_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames
        assert "phastCons_score" in header
        assert "phastCons_missing" in header

        rows = list(reader)
        assert len(rows) >= 4
        for row in rows:
            assert row["phastCons_missing"] in ("0", "1")


def test_real_chr09_to_chr10_boundary(tmp_path):
    """Check numeric chromosome order at the chr09/chr10 boundary."""
    snps=tmp_path/"snps.tsv"
    snps.write_text("Chr\tPos\nchr09\t30001\nchr09\t30002\nchr09\t30003\n"
                    "chr10\t1\nchr10\t2\nchr10\t3\nchr10\t4\n")
    track=tmp_path/"real_track_excerpt.bedgraph"
    track.write_text("chr09\t30000\t30001\t0.007\n"
                     "chr09\t30001\t30002\t0.015\n"
                     "chr09\t30002\t30003\t0.022\n"
                     "chr10\t0\t1\t0.25\nchr10\t1\t2\t0.25\n"
                     "chr10\t2\t3\t0.25\n")
    output=tmp_path/"mapped.tsv"
    result=run_script("conservation/map_bedgraph_to_snps.py",[
        "--snps",str(snps),"--bedgraph",str(track),"--output",str(output),
        "--track-name","phastCons"])
    assert result.returncode==0,result.stderr
    with output.open() as stream:rows=list(csv.DictReader(stream,delimiter="\t"))
    assert [row["phastCons_score"] for row in rows[:6]]==[
        "0.007","0.015","0.022","0.25","0.25","0.25"]
    assert all(row["phastCons_missing"]=="0" for row in rows[:6])
    assert rows[6]["phastCons_score"]=="NA"
    assert rows[6]["phastCons_missing"]=="1"


def test_subgenome_chromosomes_remain_distinct(tmp_path):
    snps=tmp_path/"snps.tsv"
    snps.write_text("Chr\tPos\nA09\t2\nA10\t2\nD01\t2\n")
    track=tmp_path/"track.bedgraph"
    track.write_text("A09\t0\t3\t0.1\nA10\t0\t3\t0.2\nD01\t0\t3\t0.3\n")
    output=tmp_path/"scores.tsv"
    result=run_script("conservation/map_bedgraph_to_snps.py",[
        "--snps",str(snps),"--bedgraph",str(track),"--output",str(output),
        "--track-name","phyloP"])
    assert result.returncode==0,result.stderr
    with output.open() as stream:rows=list(csv.DictReader(stream,delimiter="\t"))
    assert [row["Chr"] for row in rows]==["A09","A10","D01"]
    assert [row["phyloP_score"] for row in rows]==["0.1","0.2","0.3"]
