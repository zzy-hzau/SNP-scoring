"""Source-extracted dependency used by the reported graph planning route."""
from __future__ import annotations
import sqlite3
from pathlib import Path

def staged_user_run_ids(runtime_root: Path) -> set[str]:
    """Run IDs that already have a provisional source row (user runs)."""
    db_path = runtime_root / "jobs.sqlite3"
    if not db_path.is_file():
        return set()
    try:
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT run_id FROM provisional_sources").fetchall()
        con.close()
    except Exception:
        return set()
    return {str(row["run_id"]) for row in rows}
