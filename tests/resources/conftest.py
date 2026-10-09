import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
from phenofuncrank_repro.common.paths import package_dir
SCRIPTS_DIR = package_dir("resources")
DATA_DIR = Path(__file__).resolve().parent / "data"


def run_script(script_name, args=None, cwd=None):
    if args is None:
        args = []
    script_path = SCRIPTS_DIR / script_name
    cmd = [sys.executable, str(script_path)] + args
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        env=env,
    )
    return result
