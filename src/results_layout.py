"""Structured on-disk layout for pipeline results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"


def ensure_results_root_with_gitkeep(results_root: Path = RESULTS_ROOT) -> None:
    """Ensure the results root directory exists and has a .gitkeep file."""
    # Create the results root directory if it doesn't exist
    results_root.mkdir(parents=True, exist_ok=True)
    # Create .gitkeep file if it doesn't exist
    gitkeep_path = results_root / ".gitkeep"
    if not gitkeep_path.exists():
        gitkeep_path.touch()


def build_project_name(dataset_path: str, task_type: str) -> str:
    """Derive a project slug from the dataset filename and inferred task type."""

    stem = Path(dataset_path).stem
    # Drop the preprocessing cache suffix so the project slug stays stable when
    # the evaluator points the dataset at a derived cache file.
    if stem.endswith("_preprocessed"):
        stem = stem[: -len("_preprocessed")]
    normalized_task = task_type.strip().lower().replace(" ", "_")
    return f"{stem}_{normalized_task}"


def infer_task_type(dataset_path: str, target_name: str) -> str:
    """Infer classification vs regression from the preprocess target column."""

    values = _read_column_values(dataset_path, target_name)
    if not values:
        return "regression"

    numeric_values: List[float] = []
    for value in values:
        try:
            numeric_values.append(float(value))
        except ValueError:
            return "classification"  # non-numeric → classification

    unique_count = len(set(numeric_values))
    if unique_count <= 2:
        return "classification"  # binary classification
    return "regression"  # numeric with >2 unique values → regression


def resolve_task_type(
    dataset_path: str,
    dataset_name: str,
    model_type: str,
    preprocesses: Dict[str, Dict[str, Any]],
) -> str:
    """Resolve task type: prioritize explicit model type, then preprocess inference."""

    # 1. First, check explicit model type (highest priority)
    if model_type == "logistic_regression":
        return "classification"
    if model_type == "linear_regression":
        return "regression"


    # 2. Then check preprocess metadata
    preprocess = preprocesses.get(dataset_name)
    if preprocess and preprocess.get("task_type"):
        return str(preprocess["task_type"])

    # 3. Then infer from target column
    if preprocess and preprocess.get("target_name"):
        return infer_task_type(dataset_path, str(preprocess["target_name"]))

    # 4. Default to regression
    return "regression"


def create_run_layout(
    project_name: str,
    metadata: Dict[str, Any],
    *,
    results_root: Optional[Path] = None,
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Create a timestamped run directory with train, evaluate, and predict folders."""

    root = results_root if results_root is not None else RESULTS_ROOT
    # Ensure results root directory exists and has a .gitkeep file
    ensure_results_root_with_gitkeep(root)
    run_moment = timestamp if timestamp is not None else datetime.now().astimezone()
    run_id = run_moment.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = root / project_name / run_id

    train_dir = run_dir / "train"
    evaluate_dir = run_dir / "evaluate"
    predict_dir = run_dir / "predict"
    for directory in (train_dir, evaluate_dir, predict_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metadata_path = run_dir / "metadata.json"
    payload = {
        "project_name": project_name,
        "run_id": run_id,
        "created_at": run_moment.isoformat(timespec="seconds"),
        **metadata,
    }
    write_json(metadata_path, payload)

    return {
        "project_name": project_name,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "train_dir": str(train_dir),
        "evaluate_dir": str(evaluate_dir),
        "predict_dir": str(predict_dir),
        "metadata_path": str(metadata_path),
        "predict_counter": 0,
    }


def next_predict_dir(run_context: Dict[str, Any]) -> Path:
    """Allocate the next numbered predict output directory for a run."""

    run_context["predict_counter"] = int(run_context.get("predict_counter", 0)) + 1
    predict_root = Path(run_context["predict_dir"])
    predict_output = predict_root / f"{run_context['predict_counter']:03d}"
    predict_output.mkdir(parents=True, exist_ok=True)
    return predict_output


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON payload to disk, creating parent directories when needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_json_if_changed(path: Path, payload: Dict[str, Any]) -> bool:
    """Write JSON only when content differs from the existing file."""

    content = json.dumps(payload, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    write_json(path, payload)
    return True


def update_metadata(metadata_path: str, updates: Dict[str, Any]) -> None:
    """Merge updates into an existing metadata.json file."""

    path = Path(metadata_path)
    payload: Dict[str, Any] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    write_json(path, payload)


def _read_column_values(dataset_path: str, column_name: str) -> List[str]:
    path = Path(dataset_path)
    if not path.exists():
        return []

    import csv
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            all_rows = list(reader)
    except Exception:
        return []

    if not all_rows:
        return []

    headers = [header.strip() for header in all_rows[0]]
    try:
        column_index = headers.index(column_name)
    except ValueError:
        return []

    values: List[str] = []
    for row in all_rows[1:]:
        if len(row) <= column_index:
            continue
        values.append(row[column_index].strip())
    return values
