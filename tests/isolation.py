"""Redirect and clean up runtime outputs produced during tests."""

from __future__ import annotations

import atexit
import re
import shutil
import unittest
from pathlib import Path

from src import results_layout

REPO_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
TEST_OUTPUTS_ROOT = Path(__file__).resolve().parent / ".outputs"
_INSTALLED = False


def _safe_test_name(test_id: str) -> str:
    return re.sub(r"[^\w.-]+", "_", test_id)


def install_test_output_isolation() -> None:
    """Route each unittest to a private output directory and delete it afterward."""

    global _INSTALLED
    if _INSTALLED:
        return

    original_run = unittest.TestCase.run

    def isolated_run(self, result=None):
        test_dir = TEST_OUTPUTS_ROOT / _safe_test_name(self.id())
        previous_root = results_layout.RESULTS_ROOT
        results_layout.RESULTS_ROOT = test_dir
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            return original_run(self, result)
        finally:
            results_layout.RESULTS_ROOT = previous_root
            shutil.rmtree(test_dir, ignore_errors=True)

    unittest.TestCase.run = isolated_run  # type: ignore[method-assign]
    _INSTALLED = True


def cleanup_test_outputs() -> None:
    """Remove any leftover test outputs and restore the project results root."""

    results_layout.RESULTS_ROOT = REPO_RESULTS_ROOT
    shutil.rmtree(TEST_OUTPUTS_ROOT, ignore_errors=True)


atexit.register(cleanup_test_outputs)
