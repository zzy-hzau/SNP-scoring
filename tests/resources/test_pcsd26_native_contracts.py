from phenofuncrank_repro.common.paths import package_dir
from pathlib import Path
import subprocess
import sys

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def test_first_overlap_missing_and_all_state_columns(tmp_path):
    pytest.importorskip("numpy")
    import shutil
    if shutil.which("bedtools") is None:
        pytest.skip("bedtools is required")
    script = package_dir("resources") / "chromatin_state/pcsd26_native.py"
    inputs = tmp_path / "sampled input.parquet"
    bed = tmp_path / "state reference.bed"
    output = tmp_path / "output table.parquet"
    rows = []
    records = []
    for index in range(1, 27):
        start = index * 100
        records.append(f"1\t{start}\t{start+10}\tE{index}\n")
        rows.append({"snp_id": f"state{index}", "Chr": "1", "Pos": start+(8 if index == 2 else 1)})
    
    records.append("1\t195\t205\tE1\n")
    rows += [{"snp_id": "multiple_hit", "Chr": "1", "Pos": 201},
             {"snp_id": "missing", "Chr": "__none__", "Pos": 1}]
    pd.DataFrame(rows).to_parquet(inputs, index=False)
    bed.write_text("".join(records))
    result = subprocess.run([sys.executable, str(script), "--sampled", str(inputs),
                             "--pcsd-bed", str(bed), "--output", str(output),
                             "--work-dir", str(tmp_path / "temporary children")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = pd.read_parquet(output).set_index("snp_id")
    assert data.loc["multiple_hit", "pcsd26_state"] == "E1"
    for index in range(1, 27):
        expected = index
        row = data.loc[f"state{index}"]
        assert row.pcsd26_missing == 0
        assert row[f"pcsd26_state__E{expected}"] == 1
        assert row[[f"pcsd26_state__E{x}" for x in range(1, 27)]].sum() == 1
    assert data.loc["missing", "pcsd26_missing"] == 1
    assert data.loc["missing", [f"pcsd26_state__E{x}" for x in range(1, 27)]].sum() == 0
    assert inputs.exists() and bed.exists()
