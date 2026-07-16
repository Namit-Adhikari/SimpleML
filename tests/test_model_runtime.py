import csv
import tempfile
import unittest
from pathlib import Path

from src.evaluator import evaluate_ast
from src.parser import parse_statement
from src.ast import build_ast_from_parse_tree


class ModelRuntimeTests(unittest.TestCase):
    def test_train_statement_creates_model_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.csv"
            with dataset_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["feature", "target"])
                writer.writerow([1.0, 2.0])
                writer.writerow([2.0, 4.0])

            ast_node = build_ast_from_parse_tree(
                [
                    parse_statement(f'load dataset from "{dataset_path}"'),
                    parse_statement("train model using linear_regression on dataset"),
                ]
            )
            state = evaluate_ast(ast_node)

            model_path = Path(state.models["model"]["model_path"])
            self.assertIn("model", state.models)
            self.assertTrue(model_path.exists())
            self.assertEqual(model_path.name, "model.pkl")
            self.assertEqual(model_path.parent.name, "train")

    def test_evaluate_statement_records_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.csv"
            with dataset_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["feature", "target"])
                writer.writerow([0.0, 0.0])
                writer.writerow([1.0, 1.0])

            ast_node = build_ast_from_parse_tree(
                [
                    parse_statement(f'load dataset from "{dataset_path}"'),
                    parse_statement("train model using linear_regression on dataset"),
                    parse_statement("evaluate model using metrics"),
                ]
            )
            state = evaluate_ast(ast_node)

            evaluate_dir = Path(state.models["model"]["evaluate_dir"])
            self.assertIn("metrics", state.metrics)
            self.assertIn("r2", state.metrics["metrics"])
            self.assertTrue((evaluate_dir / "metrics.json").exists())
            # We no longer generate evaluation.json

    def test_random_forest_regression_evaluates_with_r2(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.csv"
            with dataset_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["feature", "target"])
                writer.writerow([1.0, 10.0])
                writer.writerow([2.0, 20.0])
                writer.writerow([3.0, 30.0])

            ast_node = build_ast_from_parse_tree(
                [
                    parse_statement(f'load dataset from "{dataset_path}"'),
                    parse_statement('preprocess dataset using "target"'),
                    parse_statement("train model using random_forest on dataset"),
                    parse_statement("evaluate model using metrics"),
                ]
            )
            state = evaluate_ast(ast_node)

            self.assertIn("metrics", state.metrics)
            self.assertIn("r2", state.metrics["metrics"])
            self.assertIn("mse", state.metrics["metrics"])
            self.assertIn("rmse", state.metrics["metrics"])
            self.assertNotIn("accuracy", state.metrics["metrics"])

    def test_random_forest_classification_evaluates_with_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.csv"
            with dataset_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["feature", "target"])
                writer.writerow([1.0, 0.0])
                writer.writerow([2.0, 1.0])

            ast_node = build_ast_from_parse_tree(
                [
                    parse_statement(f'load dataset from "{dataset_path}"'),
                    parse_statement('preprocess dataset using "target"'),
                    parse_statement("train model using random_forest on dataset"),
                    parse_statement("evaluate model using metrics"),
                ]
            )
            state = evaluate_ast(ast_node)

            self.assertIn("metrics", state.metrics)
            self.assertIn("accuracy", state.metrics["metrics"])
            self.assertNotIn("r2", state.metrics["metrics"])
            self.assertNotIn("mse", state.metrics["metrics"])
            self.assertNotIn("rmse", state.metrics["metrics"])


if __name__ == "__main__":
    unittest.main()
