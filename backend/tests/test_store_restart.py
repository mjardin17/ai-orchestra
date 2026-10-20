import os, sqlite3, subprocess, sys
from pathlib import Path


def test_startup_succeeds_on_a_database_already_migrated_to_the_skills_column(tmp_path):
    """Regression: the seed insert once used positional VALUES, which fails after the skills column exists."""
    con = sqlite3.connect(tmp_path / "orchestra.db")
    con.execute("create table workspaces(id text primary key, name text, notes text default '', skills text default '[]')")
    con.commit()
    con.close()
    backend = Path(__file__).resolve().parents[1]
    res = subprocess.run([sys.executable, "-c", "import app.store"], cwd=tmp_path, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(backend)})
    assert res.returncode == 0, res.stderr[-300:]
