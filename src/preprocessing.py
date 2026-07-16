"""Preprocessing operations applied to CSV datasets at preprocess time.

The DSL auto-applies the supported preprocessing operations when a ``preprocess``
statement runs, writing the transformed copy to a transient cache directory so the
original ``data/*.csv`` is never modified. Downstream stages transparently consume
the processed copy because the evaluator points ``state.datasets`` at it.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .grammar import SIMPLE_IMPUTATION_BEHAVIOR, SUPPORTED_PREPROCESSING_OPERATIONS

CACHE_ROOT = Path(__file__).resolve().parent.parent / ".cache" / "preprocessed"


def _is_numeric(value: str) -> bool:
    """Return True when a trimmed cell value parses as a float."""
    try:
        float(value.strip())
        return True
    except (ValueError, TypeError):
        return False


def _read_rows(dataset_path: str) -> Tuple[List[str], List[List[str]]]:
    """Read a CSV into a header list and a list of row lists (as strings)."""
    with Path(dataset_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        all_rows = [row for row in reader]
    if not all_rows:
        return [], []
    header = [col.strip() for col in all_rows[0]]
    data_rows = [
        [cell.strip() for cell in row]
        for row in all_rows[1:]
        if any(cell.strip() for cell in row)
    ]
    return header, data_rows


def _write_rows(processed_path: Path, header: List[str], data_rows: List[List[str]]) -> None:
    """Write header and rows back to disk."""
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with processed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(data_rows)


def _apply_simple_imputation(
    header: List[str], data_rows: List[List[str]], target_idx: int
) -> None:
    """Fill missing numeric cells with median, missing categorical cells with mode."""
    numeric_behavior = SIMPLE_IMPUTATION_BEHAVIOR.get("numerical", "median")
    categorical_behavior = SIMPLE_IMPUTATION_BEHAVIOR.get("categorical", "mode")

    for col_idx, col_name in enumerate(header):
        if col_idx == target_idx:
            continue
        values = [row[col_idx] for row in data_rows]
        numeric_values = [float(v) for v in values if v != "" and _is_numeric(v)]

        if numeric_values:
            fill = (
                statistics.median(numeric_values)
                if numeric_behavior == "median"
                else sum(numeric_values) / len(numeric_values)
            )
            for row in data_rows:
                if row[col_idx] == "":
                    row[col_idx] = str(fill)
        else:
            non_empty = [v for v in values if v != ""]
            fill = statistics.mode(non_empty) if non_empty else ""
            for row in data_rows:
                if row[col_idx] == "":
                    row[col_idx] = fill if categorical_behavior == "mode" else fill


def _apply_z_score_normalization(
    header: List[str], data_rows: List[List[str]], target_idx: int
) -> None:
    """Scale numeric, non-target columns to z-scores (x - mean) / std (std=0 -> 0)."""
    for col_idx, col_name in enumerate(header):
        if col_idx == target_idx:
            continue
        values = [
            float(row[col_idx])
            for row in data_rows
            if row[col_idx] != "" and _is_numeric(row[col_idx])
        ]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)
        if std == 0:
            for row in data_rows:
                if row[col_idx] != "" and _is_numeric(row[col_idx]):
                    row[col_idx] = "0.0"
        else:
            for row in data_rows:
                if row[col_idx] != "" and _is_numeric(row[col_idx]):
                    z = (float(row[col_idx]) - mean) / std
                    row[col_idx] = str(z)


def _apply_label_encoding(
    header: List[str], data_rows: List[List[str]], target_idx: int
) -> None:
    """Map each categorical (non-numeric, non-target) column to integer ids."""
    for col_idx, col_name in enumerate(header):
        if col_idx == target_idx:
            continue
        if any(row[col_idx] != "" and not _is_numeric(row[col_idx]) for row in data_rows):
            label_to_id: Dict[str, int] = {}
            next_id = 0
            for row in data_rows:
                raw = row[col_idx]
                if raw == "":
                    continue
                if raw not in label_to_id:
                    label_to_id[raw] = next_id
                    next_id += 1
                row[col_idx] = str(label_to_id[raw])


def _apply_duplicate_removal(
    header: List[str], data_rows: List[List[str]]
) -> List[List[str]]:
    """Deduplicate rows by all columns (including the target)."""
    seen = set()
    unique_rows: List[List[str]] = []
    for row in data_rows:
        key = tuple(row)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def apply_preprocessing(
    dataset_path: str,
    target_name: str,
    *,
    cache_dir: Optional[Path] = None,
    operations: Tuple[str, ...] = SUPPORTED_PREPROCESSING_OPERATIONS,
) -> Tuple[str, Dict[str, Any]]:
    """Apply supported preprocessing operations and write a cache copy.

    Returns the processed file path and an info dict describing the operations
    that were applied. The original CSV is never modified.
    """

    cache_root = cache_dir if cache_dir is not None else CACHE_ROOT
    source = Path(dataset_path)
    stem = source.stem
    processed_path = cache_root / f"{stem}_preprocessed.csv"

    header, data_rows = _read_rows(dataset_path)
    try:
        target_idx = header.index(target_name)
    except ValueError:
        target_idx = -1

    applied: List[str] = []
    for op in operations:
        if op == "simple_imputation":
            _apply_simple_imputation(header, data_rows, target_idx)
            applied.append(op)
        elif op == "z_score_normalization":
            _apply_z_score_normalization(header, data_rows, target_idx)
            applied.append(op)
        elif op == "label_encoding":
            _apply_label_encoding(header, data_rows, target_idx)
            applied.append(op)
        elif op == "duplicate_removal":
            data_rows = _apply_duplicate_removal(header, data_rows)
            applied.append(op)

    _write_rows(processed_path, header, data_rows)

    info: Dict[str, Any] = {
        "processed_path": str(processed_path),
        "operations": applied,
        "applied": applied,
        "original_path": str(dataset_path),
        "target_name": target_name,
        "row_count": len(data_rows),
        "column_count": len(header),
    }
    return str(processed_path), info


def preprocess_dataset(
    dataset_path: str,
    target_name: str,
    *,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Public wrapper that applies preprocessing and returns the info dict."""

    processed_path, info = apply_preprocessing(
        dataset_path,
        target_name,
        cache_dir=cache_dir,
    )
    return info


def clean_preprocessing_cache(cache_dir: Optional[Path] = None) -> None:
    """Remove generated preprocessed cache files to avoid silent resource growth.

    Deletes the contents of the transient cache directory (``.cache/preprocessed``
    by default). The original datasets under ``data/`` are never touched.
    """

    cache_root = cache_dir if cache_dir is not None else CACHE_ROOT
    if not cache_root.exists():
        return
    for entry in cache_root.iterdir():
        try:
            if entry.is_file() or entry.is_symlink():
                entry.unlink()
            elif entry.is_dir():
                import shutil

                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            continue
