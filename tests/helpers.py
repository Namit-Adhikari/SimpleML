"""Shared helpers for test fixtures."""

from __future__ import annotations

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE_DATA_DIR = FIXTURES_DIR / "data"
FIXTURE_SCRIPTS_DIR = FIXTURES_DIR / "scripts"
FIXTURE_CSV = FIXTURE_DATA_DIR / "regression.csv"
FIXTURE_PIPELINE_SCRIPT = FIXTURE_SCRIPTS_DIR / "pipeline.dsl"


def get_fixture_csv_info() -> tuple[Path, str, str]:
    """Return the fixture CSV path, filename, and last column name."""

    with FIXTURE_CSV.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
    columns = [column.strip() for column in header.split(",") if column.strip()]
    return FIXTURE_CSV, FIXTURE_CSV.name, columns[-1]
