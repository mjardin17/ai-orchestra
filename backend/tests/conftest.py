import os, sys, tempfile
from pathlib import Path

# store.py opens "orchestra.db" in the CWD at import time, so isolate every test run in a temp dir.
_tmp = tempfile.mkdtemp(prefix="orchestra-test-")
os.chdir(_tmp)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
