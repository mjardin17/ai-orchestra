import os, sqlite3, subprocess, sys, tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def _start(cwd):
    env = dict(os.environ, PYTHONPATH=str(BACKEND))
    return subprocess.run([sys.executable, "-c", "import app.store"], cwd=cwd, env=env, capture_output=True, text=True)


def test_store_starts_again_on_an_existing_database():
    """The app must survive a restart: the second import opens the db created (and migrated) by the first."""
    with tempfile.TemporaryDirectory() as d:
        first, second = _start(d), _start(d)
        assert first.returncode == 0, first.stderr
        assert second.returncode == 0, second.stderr


def test_startup_succeeds_on_a_database_already_migrated_to_the_skills_column(tmp_path):
    """Regression: the seed insert once used positional VALUES, which fails after the skills column exists."""
    con = sqlite3.connect(tmp_path / "orchestra.db")
    con.execute("create table workspaces(id text primary key, name text, notes text default '', skills text default '[]')")
    con.commit()
    con.close()
    res = subprocess.run([sys.executable, "-c", "import app.store"], cwd=tmp_path, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(BACKEND)})
    assert res.returncode == 0, res.stderr[-300:]
