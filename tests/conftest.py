import json
import os
import secrets
import shutil
import string
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = ROOT / "tests" / "_tmp"
PYTHON_TEMP_ROOT = TEST_TMP_ROOT / "python-temp"
TMP_PATH_ROOT = TEST_TMP_ROOT / "tmp-path"
PYTHON_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
TMP_PATH_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(PYTHON_TEMP_ROOT)
os.environ["TMP"] = str(PYTHON_TEMP_ROOT)
os.environ["TMPDIR"] = str(PYTHON_TEMP_ROOT)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_path():
    suffix = "".join(secrets.choice(string.ascii_lowercase) for _ in range(12))
    path = TMP_PATH_ROOT / f"case-{suffix}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def example_profile_copy(tmp_path):
    source = ROOT / "configs" / "example"
    target = tmp_path / "example"
    shutil.copytree(source, target)
    return target / "example.yaml"


@pytest.fixture
def run_cli(monkeypatch):
    def _run(func, argv):
        stream = StringIO()
        monkeypatch.setattr(sys, "argv", argv)
        with redirect_stdout(stream):
            func()
        return json.loads(stream.getvalue())

    return _run
