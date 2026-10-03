import os, subprocess, sys, tempfile
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
