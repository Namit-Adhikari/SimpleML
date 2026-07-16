"""Minimal model runtime for CSV-based tabular workflows."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

random.seed(42)  # Fixed random seed for reproducibility!


class ModelRuntimeError(ValueError):
    """Raised when a model workflow cannot be executed."""



def _split_train_test(features: List[List[float]], targets: List[Any], test_size: float = 0.2) -> Tuple[List[List[float]], List[Any], List[List[float]], List[Any]]:
    """
    Split features and targets into 80:20 train/test (default).
    Uses fixed random seed 42 for reproducibility.
    """
    # Shuffle indices!
    indices = list(range(len(features)))
    random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Split features and targets!
    X_train = [features[i] for i in train_indices]
    y_train = [targets[i] for i in train_indices]
    X_test = [features[i] for i in test_indices]
    y_test = [targets[i] for i in test_indices]
    
    return X_train, y_train, X_test, y_test

def _load_csv_dataset_with_columns(
    dataset_path: str, 
    feature_columns: Optional[List[str]] = None, 
    target_column: Optional[str] = None
) -> Tuple[List[List[float]], List[Any], List[str], List[str], Dict[float, Any]]:
    """
    Load CSV, return:
      features: list of feature vectors
      targets: list of target values
      header: column names
      feature_names: list of selected feature column names
      class_labels: mapping from numeric id to original class label (for classification tasks)
    """
    path = Path(dataset_path)
    if not path.exists():
        raise ModelRuntimeError(f"Dataset not found: {dataset_path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        all_rows = list(reader)

    if not all_rows:
        return [], [], [], [], {}

    header = [col.strip() for col in all_rows[0]]
    data_rows = all_rows[1:] if len(all_rows) > 1 else []

    # If no feature columns given, pick a preferred numeric column
    if not feature_columns:
        preferred_features = ["Fare", "Pclass", "Age"]  # For titanic specifically
        feature_columns = []
        for col_name in preferred_features:
            if col_name in header:
                if target_column and col_name == target_column:
                    continue
                feature_columns.append(col_name)
                break
        if not feature_columns:
            for col_name in header:
                if target_column and col_name == target_column:
                    continue
                feature_columns.append(col_name)
                break

    target_idx = header.index(target_column) if target_column else 1
    feature_indices = [header.index(col) for col in feature_columns]

    # For classification tasks, map string class labels to numeric ids
    class_labels: Dict[float, Any] = {}
    label_to_id: Dict[Any, float] = {}
    next_id = 0.0

    features: List[List[float]] = []
    targets: List[Any] = []
    for row in data_rows:
        try:
            feat_vals = []
            for idx in feature_indices:
                if idx >= len(row) or not row[idx].strip():
                    feat_vals.append(0.0)
                else:
                    feat_vals.append(float(row[idx].strip()))

            target_val = None
            if target_column:
                if target_idx < len(row):
                    raw_target = row[target_idx].strip()
                    if raw_target:
                        # Try to parse as float first, if fails, treat as categorical string
                        try:
                            target_val = float(raw_target)
                        except ValueError:
                            # It's a categorical string - map to numeric id
                            if raw_target not in label_to_id:
                                label_to_id[raw_target] = next_id
                                class_labels[next_id] = raw_target
                                next_id += 1.0
                            target_val = label_to_id[raw_target]

            if target_val is not None or not target_column:
                features.append(feat_vals)
                targets.append(target_val if target_val is not None else 0.0)
        except ValueError:
            continue

    return features, targets, header, feature_columns, class_labels


def train_linear_regression(dataset_path: str, artifact_dir: str, target_name: Optional[str] = None) -> Dict[str, Any]:
    """Train linear regression model (univariate)."""
    features, targets, _, feature_names, _ = _load_csv_dataset_with_columns(dataset_path, target_column=target_name)
    if not features or not targets:
        raise ModelRuntimeError("Dataset is empty")

    # Split into 80:20 train/test!
    X_train, y_train, X_test, y_test = _split_train_test(features, targets, test_size=0.2)

    xs = [f[0] for f in X_train]
    ys = y_train
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    slope = numerator / denominator if denominator != 0 else 0.0
    intercept = mean_y - slope * mean_x

    # Save model to model.pkl (include train and test data!)
    model_path = Path(artifact_dir) / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    model_data = {
        "model_type": "linear_regression",
        "slope": slope,
        "intercept": intercept,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    return {
        "model_type": "linear_regression",
        "model_path": str(model_path),
        "slope": slope,
        "intercept": intercept,
        "target_name": target_name,
        "task_type": "regression",
        "dataset_path": dataset_path,
    }


def train_logistic_regression(dataset_path: str, artifact_dir: str, target_name: Optional[str] = None) -> Dict[str, Any]:
    """Train a simple logistic regression model (univariate, threshold-based)."""
    features, targets, _, feature_names, class_labels = _load_csv_dataset_with_columns(dataset_path, target_column=target_name)
    if not features or not targets:
        raise ModelRuntimeError("Dataset is empty")

    # Split into 80:20 train/test!
    X_train, y_train, X_test, y_test = _split_train_test(features, targets, test_size=0.2)

    xs = [f[0] for f in X_train]
    # Simple threshold: predict 1 if x > mean_x, 0 otherwise
    mean_x = sum(xs) / len(xs)

    # Save model to model.pkl (include train and test data!)
    model_path = Path(artifact_dir) / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    model_data = {
        "model_type": "logistic_regression",
        "threshold": mean_x,
        "class_labels": class_labels,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    return {
        "model_type": "logistic_regression",
        "model_path": str(model_path),
        "threshold": mean_x,
        "target_name": target_name,
        "task_type": "classification",
        "dataset_path": dataset_path,
    }


def train_random_forest(
    dataset_path: str,
    artifact_dir: str,
    target_name: Optional[str] = None,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Train a simple random forest.

    The implementation is intentionally minimal: it acts as a baseline that
    predicts a single constant value for every input. For classification the
    constant is the majority class observed in the training set; for regression
    it is the mean of the training targets. This keeps the model portable and
    dependency-free while exercising the full training/evaluation pipeline.
    """
    features, targets, _, feature_names, class_labels = _load_csv_dataset_with_columns(dataset_path, target_column=target_name)
    if not targets:
        raise ModelRuntimeError("Dataset is empty")

    # Infer the task type from the target column when not supplied explicitly.
    if task_type is None:
        if target_name:
            from .results_layout import infer_task_type
            task_type = infer_task_type(dataset_path, target_name)
        else:
            task_type = "regression"

    # Split into 80:20 train/test with the fixed seed for reproducibility.
    X_train, y_train, X_test, y_test = _split_train_test(features, targets, test_size=0.2)

    # Constant baseline prediction learned from the training set.
    if task_type == "classification":
        # Majority class prediction.
        counts: Dict[float, int] = {}
        for t in y_train:
            counts[t] = counts.get(t, 0) + 1
        baseline = max(counts.items(), key=lambda x: x[1])[0]
        baseline_key = "majority_class"
    else:
        # Mean target prediction for regression.
        baseline = sum(y_train) / len(y_train)
        baseline_key = "mean_value"

    # Save model to model.pkl (include train and test data for evaluation).
    model_path = Path(artifact_dir) / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    model_data = {
        "model_type": "random_forest",
        "task_type": task_type,
        baseline_key: baseline,
        "class_labels": class_labels,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    return {
        "model_type": "random_forest",
        "model_path": str(model_path),
        baseline_key: baseline,
        "target_name": target_name,
        "task_type": task_type,
        "dataset_path": dataset_path,
    }


def evaluate_model(
    model_path: str,
    dataset_path: str,
    model_metadata: Dict[str, Any],
    *,
    evaluate_dir: str | None = None,
) -> Dict[str, Any]:
    """Compute simple evaluation metrics for a trained model using test split, and generate plots."""
    import pickle
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    model_type = model_metadata.get("model_type")
    task_type = model_metadata.get("task_type")
    if not task_type:
        task_type = model_data.get("task_type")
    if not task_type:
        if model_type == "linear_regression":
            task_type = "regression"
        else:
            task_type = "classification"
    
    # Get test data from model_data
    X_test = model_data.get("X_test")
    y_test = model_data.get("y_test")
    if not X_test or not y_test:
        raise ModelRuntimeError("Test data not found in model file!")

    test_preds: List[float] = []
    metrics_data: Dict[str, Any] = {}

    if model_type == "linear_regression":
        # Compute test predictions
        for feat in X_test:
            test_preds.append(model_data["slope"] * feat[0] + model_data["intercept"])
    elif model_type == "logistic_regression":
        threshold = model_data["threshold"]
        test_preds = [1.0 if f[0] > threshold else 0.0 for f in X_test]
    elif model_type == "random_forest":
        # Use the constant baseline trained for the model's task type: the
        # majority class for classification, the mean value for regression.
        if task_type == "regression":
            baseline = model_data["mean_value"]
        else:
            baseline = model_data["majority_class"]
        test_preds = [baseline for _ in y_test]
    else:
        raise ModelRuntimeError(f"Unknown model type: {model_type}")

    # Compute metrics based on task_type
    if task_type == "regression":
        test_residuals = [pred - actual for pred, actual in zip(test_preds, y_test)]
        test_mse = sum(r * r for r in test_residuals) / len(test_residuals)
        # Calculate R-squared (R²)
        y_mean = sum(y_test) / len(y_test)
        total_ss = sum((y - y_mean) ** 2 for y in y_test)
        residual_ss = sum(r ** 2 for r in test_residuals)
        r2 = 1 - (residual_ss / total_ss) if total_ss != 0 else 0.0
        metrics_data = {
            "r2": round(r2, 4),
            "mse": round(test_mse, 4),
            "rmse": round(math.sqrt(test_mse), 4),
        }
    elif task_type == "classification":
        test_correct = sum(1 for p, a in zip(test_preds, y_test) if p == a)
        test_acc = test_correct / len(y_test)
        metrics_data = {
            "accuracy": round(test_acc, 4),
        }
    else:
        raise ModelRuntimeError(f"Unknown task type: {task_type}")

    if evaluate_dir is not None:
        from .results_layout import write_json

        evaluate_path = Path(evaluate_dir)
        # Always overwrite metrics.json
        write_json(evaluate_path / "metrics.json", metrics_data)
        
        # Generate plots
        if task_type == "classification":
            # Confusion Matrix
            # Find unique classes
            y_train = model_data.get("y_train", [])
            classes = sorted(list(set(y_train + y_test)))
            n_classes = len(classes)
            # Build confusion matrix
            cm = [[0]*n_classes for _ in range(n_classes)]
            class_to_idx = {c: i for i, c in enumerate(classes)}
            for a, p in zip(y_test, test_preds):
                if a in class_to_idx and p in class_to_idx:
                    cm[class_to_idx[a]][class_to_idx[p]] += 1
            # Plot confusion matrix
            fig, ax = plt.subplots()
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title("Confusion Matrix")
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))
            # Use class labels if available
            class_labels = model_data.get("class_labels", {})
            tick_labels = []
            for c in classes:
                if c in class_labels:
                    tick_labels.append(str(class_labels[c]))
                else:
                    tick_labels.append(str(c))
            ax.set_xticklabels(tick_labels, rotation=45)
            ax.set_yticklabels(tick_labels)
            # Add text labels to cells
            for i in range(n_classes):
                for j in range(n_classes):
                    ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            fig.tight_layout()
            fig.savefig(evaluate_path / "confusion_matrix.png")
            plt.close(fig)

    return metrics_data


def predict_model(
    model_path: str, 
    second_arg: str | dict | list, 
    third_arg: str,
    model_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate predictions, supports both old and new styles!
    
    Old style: predict_model(model_path, dataset_path, output_dir, model_metadata)
    New style: predict_model(model_path, input_data, output_dir, model_metadata)
    """
    import pickle
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    # Get class labels from model data (not metadata!)
    class_labels = model_data.get("class_labels", {})
    
    model_type = model_metadata.get("model_type")
    feature_names = ["Fare", "Pclass", "Age"]  # Default features (matches _load_csv_dataset_with_columns)
    target_name = model_metadata.get("target_name")
    task_type = model_metadata.get("task_type", "regression")
    
    if isinstance(second_arg, (dict, list)):
        # New style: input data
        input_data = second_arg
        output_dir = third_arg
        
        # Normalize input_data to a list of dicts
        if isinstance(input_data, dict):
            input_data = [input_data]
            
        # Process each input dict to make predictions
        predictions = []
        for data_point in input_data:
            # Extract features in the correct order (use first available feature)
            features = []
            for fn in feature_names:
                val = data_point.get(fn, 0.0)
                try:
                    features.append(float(val))
                    break  # Take first available feature (univariate)
                except (ValueError, TypeError):
                    continue
            # If no features found, use 0.0
            if not features:
                features = [0.0]
                    
            # Make prediction based on model type
            if model_type == "linear_regression":
                pred = model_data["slope"] * features[0] + model_data["intercept"]
            elif model_type == "logistic_regression":
                pred = 1.0 if features[0] > model_data["threshold"] else 0.0
            elif model_type == "random_forest":
                # Constant baseline: majority class for classification,
                # mean value for regression.
                if task_type == "regression":
                    pred = model_data["mean_value"]
                else:
                    pred = model_data["majority_class"]
            else:
                raise ModelRuntimeError(f"Unknown model type: {model_type}")
                
            # For classification tasks, map numeric id back to original label
            if task_type == "classification":
                # Try exact match first
                if pred in class_labels:
                    pred = class_labels[pred]
                else:
                    # Try to find matching label by key type
                    if isinstance(pred, float):
                        # Search for a float key that equals the prediction
                        for key, label in class_labels.items():
                            if isinstance(key, float) and key == pred:
                                pred = label
                                break
                    elif isinstance(pred, int):
                        # Search for an integer or float key that equals the prediction
                        for key, label in class_labels.items():
                            if (isinstance(key, int) or isinstance(key, float)) and key == pred:
                                pred = label
                                break
                    
            predictions.append(pred)
            
        # Write predictions to CSV
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_path / "predictions.csv"
        
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if target_name:
                writer.writerow([target_name])
            else:
                writer.writerow(["prediction"])
            for p in predictions:
                writer.writerow([p])
                
        return {
            "predictions_path": str(csv_path),
        }
    else:
        # Old style: dataset_path - also only write CSV
        dataset_path = second_arg
        output_dir = third_arg
        
        # We'll still just generate simple CSV predictions for old style too
        features, _, _, _, _ = _load_csv_dataset_with_columns(dataset_path, target_column=target_name)
        
        # Process each feature to make predictions
        predictions = []
        for feat in features:
            # Make prediction based on model type
            if model_type == "linear_regression":
                pred = model_data["slope"] * feat[0] + model_data["intercept"]
            elif model_type == "logistic_regression":
                pred = 1.0 if feat[0] > model_data["threshold"] else 0.0
            elif model_type == "random_forest":
                # Constant baseline: majority class for classification,
                # mean value for regression.
                if task_type == "regression":
                    pred = model_data["mean_value"]
                else:
                    pred = model_data["majority_class"]
            else:
                raise ModelRuntimeError(f"Unknown model type: {model_type}")
                
            # For classification tasks, map numeric id back to original label
            if task_type == "classification":
                # Try exact match first
                if pred in class_labels:
                    pred = class_labels[pred]
                else:
                    # Try to find matching label by key type
                    if isinstance(pred, float):
                        for key, label in class_labels.items():
                            if isinstance(key, float) and key == pred:
                                pred = label
                                break
                    elif isinstance(pred, int):
                        for key, label in class_labels.items():
                            if (isinstance(key, int) or isinstance(key, float)) and key == pred:
                                pred = label
                                break
                    
            predictions.append(pred)
            
        # Write predictions to CSV
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_path / "predictions.csv"
        
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if target_name:
                writer.writerow([target_name])
            else:
                writer.writerow(["prediction"])
            for p in predictions:
                writer.writerow([p])
                
        return {
            "predictions_path": str(csv_path),
        }


def _load_csv_dataset(dataset_path: str) -> List[Tuple[float, float]]:
    """Backwards-compatible 2-column CSV loader."""
    path = Path(dataset_path)
    if not path.exists():
        raise ModelRuntimeError(f"Dataset not found: {dataset_path}")

    rows: List[Tuple[float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        lines = [line.strip() for line in handle.read().splitlines() if line.strip()]

    if not lines:
        return rows

    data_rows = lines[1:] if len(lines) > 1 else []
    for row in data_rows:
        columns = [value.strip() for value in row.split(",")]
        if len(columns) < 2:
            continue
        try:
            x_value = float(columns[0])
            y_value = float(columns[1])
        except ValueError:
            continue
        rows.append((x_value, y_value))

    return rows


def _fit_line(dataset: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Backwards-compatible line fitting."""
    xs = [x for x, _ in dataset]
    ys = [y for _, y in dataset]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in dataset)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        slope = 0.0
    else:
        slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept
