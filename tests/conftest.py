"""
Shared pytest setup. Makes the repository importable, runs every test from the repository root
(environment configs reference feature configs by repo-relative paths), and registers the
``slow`` marker for long statistical tests.

Run all tests from the repository root with ``pytest tests``. Skip the slow ones with
``pytest tests -m "not slow"``.
"""

import sys
from pathlib import Path

import pytest

# Make ``import src...`` work without setting PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config):
    """Registers the custom markers used by the test suite."""
    config.addinivalue_line(
        "markers", 'slow: long statistical test (deselect with -m "not slow")'
    )


@pytest.fixture(autouse=True)
def run_from_repo_root(monkeypatch):
    """Runs every test with the repository root as working directory."""
    monkeypatch.chdir(REPO_ROOT)
