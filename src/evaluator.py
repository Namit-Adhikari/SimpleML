"""Runtime evaluator for the SimpleML DSL AST."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# Plain text output (no colors)
BOLD = ""
RESET = ""

from .ast import (
    EvaluateStatement,
    LoadStatement,
    Pipeline,
    PredictStatement,
    PreprocessStatement,
    StatementNode,
    SummaryStatement,
    TrainStatement,
)
from .model_runtime import evaluate_model, predict_model, train_linear_regression, train_logistic_regression, train_random_forest
from .preprocessing import CACHE_ROOT, preprocess_dataset
from .results_layout import (
    build_project_name,
    create_run_layout,
    infer_task_type,
    next_predict_dir,
    resolve_task_type,
    update_metadata,
    write_json,
)


@dataclass
class RuntimeState:
    """Mutable execution state for AST evaluation."""

    datasets: Dict[str, str] = field(default_factory=dict)
    preprocesses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    runs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summaries: List[str] = field(default_factory=list)
    last_model_name: Optional[str] = None
    # Track pipeline stages per model
    pipeline_stages: Dict[str, Dict[str, bool]] = field(default_factory=dict)


class EvaluationError(ValueError):
    """Raised when an AST node cannot be evaluated."""


def evaluate_ast(node: StatementNode, state: Optional[RuntimeState] = None) -> RuntimeState:
    """Evaluate a single statement or a pipeline of statements, using an optional existing state."""

    if state is None:
        state = RuntimeState()

    if isinstance(node, Pipeline):
        for statement in node.statements:
            _evaluate_statement(state, statement)
        return state

    _evaluate_statement(state, node)
    return state


def _evaluate_statement(state: RuntimeState, node: StatementNode) -> None:
    if isinstance(node, LoadStatement):
        _handle_load(state, node)
        return
    if isinstance(node, PreprocessStatement):
        _handle_preprocess(state, node)
        return
    if isinstance(node, TrainStatement):
        _handle_train(state, node)
        return
    if isinstance(node, PredictStatement):
        _handle_predict(state, node)
        return
    if isinstance(node, EvaluateStatement):
        _handle_evaluate(state, node)
        return
    if isinstance(node, SummaryStatement):
        _handle_summary(state, node)
        return

    raise EvaluationError(f"Unsupported AST node: {type(node)!r}")


def _handle_load(state: RuntimeState, statement: LoadStatement) -> None:
    candidate = Path(statement.source_path)
    if candidate.is_absolute():
        dataset_path = candidate
    else:
        dataset_path = DATA_ROOT / candidate
        if not dataset_path.exists():
            dataset_path = candidate

    if not dataset_path.exists():
        raise EvaluationError(f"Dataset not found: {dataset_path}")
    state.datasets[statement.dataset_name] = str(dataset_path)


def _handle_preprocess(state: RuntimeState, statement: PreprocessStatement) -> None:
    dataset_path = state.datasets.get(statement.dataset_name)
    if not dataset_path:
        raise EvaluationError(f"Dataset not found: {statement.dataset_name}")

    task_type = infer_task_type(dataset_path, statement.target_name)

    info = preprocess_dataset(
        dataset_path,
        statement.target_name,
        cache_dir=CACHE_ROOT,
    )
    processed_path = info["processed_path"]
    # Keep the processed copy for downstream train/evaluate/predict, but record
    # the original path so metadata and summaries point at the real dataset.
    state.datasets[statement.dataset_name] = processed_path

    state.preprocesses[statement.dataset_name] = {
        "dataset_name": statement.dataset_name,
        "target_name": statement.target_name,
        "task_type": task_type,
        "original_path": dataset_path,
        "processed_path": processed_path,
        "operations": info["operations"],
        "applied": info["applied"],
    }


def _handle_train(state: RuntimeState, statement: TrainStatement) -> None:
    dataset_path = state.datasets.get(statement.dataset_name)
    if not dataset_path:
        raise EvaluationError(f"Dataset not found: {statement.dataset_name}")

    task_type = resolve_task_type(
        dataset_path,
        statement.dataset_name,
        statement.model_type,
        state.preprocesses,
    )
    project_name = build_project_name(dataset_path, task_type)
    preprocess = state.preprocesses.get(statement.dataset_name, {})
    target_name = preprocess.get("target_name")
    # Metadata should reference the original dataset, not the transient cache copy.
    displayed_path = preprocess.get("original_path", dataset_path)
    run_context = create_run_layout(
        project_name,
        {
            "task_type": task_type,
            "dataset_name": statement.dataset_name,
            "dataset_path": displayed_path,
            "target_name": target_name,
            "model_name": statement.model_name,
            "model_type": statement.model_type,
        },
    )
    state.runs[statement.model_name] = run_context

    # Call appropriate training function
    if statement.model_type == "linear_regression":
        artifact_info = train_linear_regression(dataset_path, run_context["train_dir"], target_name=target_name)
    elif statement.model_type == "logistic_regression":
        artifact_info = train_logistic_regression(dataset_path, run_context["train_dir"], target_name=target_name)
    elif statement.model_type == "random_forest":
        artifact_info = train_random_forest(dataset_path, run_context["train_dir"], target_name=target_name, task_type=task_type)
    else:
        raise EvaluationError(f"Model type {statement.model_type} not supported")
    
    state.models[statement.model_name] = {
        "model_type": statement.model_type,
        "dataset_name": statement.dataset_name,
        "task_type": task_type,
        "project_name": project_name,
        **artifact_info,
        **run_context,
    }
    state.last_model_name = statement.model_name
    # Track pipeline stages
    state.pipeline_stages.setdefault(statement.model_name, {})["train"] = True


def _handle_predict(state: RuntimeState, statement: PredictStatement) -> None:
    model = state.models.get(statement.model_name)
    if not model:
        raise EvaluationError(f"Model not found: {statement.model_name}")

    run_context = state.runs.get(statement.model_name) or model
    dataset_name = model.get("dataset_name")
    dataset_path = state.datasets.get(dataset_name) if dataset_name else None
    model_path = model.get("model_path")
    
    # Load model metadata
    metadata = {}
    if model.get("metadata_path"):
        metadata_path = Path(model["metadata_path"])
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    predict_output = next_predict_dir(run_context)
    
    # Check which mode we are in
    if isinstance(statement.input_data, (dict, list)):
        # New mode: using input data
        prediction_info: Dict[str, Any] = {"predict_dir": str(predict_output)}
        if model_path:
            prediction_info.update(
                predict_model(model_path, statement.input_data, str(predict_output), metadata)
            )
    else:
        # Old mode: using dataset
        prediction_info: Dict[str, Any] = {"predict_dir": str(predict_output)}
        if dataset_path and model_path:
            prediction_info.update(
                predict_model(model_path, dataset_path, str(predict_output), metadata)
            )

    # Read predictions from CSV to store in state
    prediction_values = []
    if "predictions_path" in prediction_info:
        csv_path = Path(prediction_info["predictions_path"])
        if csv_path.exists():
            import csv
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    prediction_values.append(row)

    state.models.setdefault(statement.model_name, {}).update(
        {
            **prediction_info,
            "predictions": prediction_values,
            "predict_input_type": "dict" if isinstance(statement.input_data, (dict, list)) else "dataset",
        }
    )
    state.runs[statement.model_name] = run_context
    state.last_model_name = statement.model_name
    # Track pipeline stages
    state.pipeline_stages.setdefault(statement.model_name, {})["predict"] = True


def _handle_evaluate(state: RuntimeState, statement: EvaluateStatement) -> None:
    model = state.models.get(statement.model_name)
    if not model:
        if not state.datasets:
            raise EvaluationError("No datasets available for evaluation")
        dataset_name = next(iter(state.datasets))
        dataset_path = state.datasets[dataset_name]
        train_statement = TrainStatement(
            model_name=statement.model_name,
            model_type="linear_regression",
            dataset_name=dataset_name,
        )
        _handle_train(state, train_statement)
        model = state.models[statement.model_name]

    dataset_name = model.get("dataset_name")
    dataset_path = state.datasets.get(dataset_name)
    if not dataset_path:
        raise EvaluationError(f"Dataset not found: {dataset_name}")

    # Load model metadata
    metadata = {}
    if model.get("metadata_path"):
        metadata_path = Path(model["metadata_path"])
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    evaluate_dir = model.get("evaluate_dir")
    metrics_data = evaluate_model(
        model["model_path"],
        dataset_path,
        metadata,
        evaluate_dir=evaluate_dir,
    )
    
    state.metrics[statement.metrics_name] = {
        "model_name": statement.model_name,
        "metrics_name": statement.metrics_name,
        **metrics_data,
    }

    metadata_path = model.get("metadata_path")
    if metadata_path:
        from datetime import datetime, timezone

        update_metadata(
            metadata_path,
            {
                "last_evaluated_at": datetime.now(timezone.utc)
                .astimezone()
                .isoformat(timespec="seconds"),
            },
        )
    # Track pipeline stages
    state.pipeline_stages.setdefault(statement.model_name, {})["evaluate"] = True


def _print_summary_table(data: Dict[str, Any], title: str) -> None:
    """Helper function to print a compact, left-aligned two-column table."""
    if not data:
        return
    max_key_len = max(len(str(key)) for key in data.keys())
    max_val_len = max(len(str(val)) for val in data.values())

    print(f"\n{BOLD}{title}{RESET}")
    print(f"{BOLD}{'-' * (max_key_len + max_val_len + 5)}{RESET}")
    for key, value in data.items():
        val_str = str(value)
        print(
            f"{str(key).ljust(max_key_len)} | {val_str}"
        )
    print(f"{BOLD}{'-' * (max_key_len + max_val_len + 5)}{RESET}\n")


def _handle_summary(state: RuntimeState, statement: SummaryStatement) -> None:
    target_name = statement.target_name
    model_info = state.models.get(target_name)
    run_info = state.runs.get(target_name) if not model_info else model_info

    if not model_info and not run_info:
        print(f"\n! No information found for target '{target_name}'\n")
        state.summaries.append(f"summary:{target_name} - no info")
        return

    # Get pipeline stages
    pipeline_stages = state.pipeline_stages.get(target_name, {})
    train_completed = pipeline_stages.get("train", False)
    evaluate_completed = pipeline_stages.get("evaluate", False)
    predict_completed = pipeline_stages.get("predict", False)

    # Gather all information
    summary_data: Dict[str, Any] = {}

    # Load metadata.json if available
    metadata: Dict[str, Any] = {}
    if run_info and run_info.get("metadata_path"):
        metadata_path = Path(run_info["metadata_path"])
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            summary_data["Project Name"] = metadata.get("project_name", "N/A")
            summary_data["Run ID"] = metadata.get("run_id", "N/A")
            summary_data["Created At"] = metadata.get("created_at", "N/A")
            summary_data["Dataset Name"] = metadata.get("dataset_name", "N/A")
            summary_data["Dataset Path"] = metadata.get("dataset_path", "N/A")
            summary_data["Target Name"] = metadata.get("target_name", "N/A")
            summary_data["Task Type"] = metadata.get("task_type", "N/A")
            summary_data["Model Type"] = metadata.get("model_type", "N/A")
            summary_data["Train/Test Split"] = "80:20"

    # Load evaluation metrics (now direct keys!)
    metrics: Dict[str, Any] = {}
    if run_info and run_info.get("evaluate_dir"):
        metrics_path = Path(run_info["evaluate_dir"]) / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Print main summary (train info as base)
    if train_completed:
        _print_summary_table(summary_data, f"Summary: {target_name} (Train)")

    # Print metrics if evaluate has been run
    if evaluate_completed and metrics:
        _print_summary_table(metrics, f"Evaluation Metrics: {target_name}")

    # Print predict info if predict has been run
    if predict_completed and model_info:
        predict_data: Dict[str, Any] = {}
        predict_data["Input Type"] = model_info.get("predict_input_type", "N/A")
        predict_data["Predictions Path"] = model_info.get("predictions_path", "N/A")
        # Show first few predictions if available
        predictions = model_info.get("predictions", [])
        if predictions:
            predict_data["Prediction Count"] = len(predictions)
            # Limit to first 5 predictions
            for i, pred in enumerate(predictions[:5]):
                predict_data[f"Prediction {i+1}"] = str(list(pred.values())[0] if pred else "N/A")
            if len(predictions) > 5:
                predict_data[f"... and {len(predictions)-5} more"] = ""
        _print_summary_table(predict_data, f"Predictions: {target_name}")

    state.summaries.append(f"summary:{target_name}")
