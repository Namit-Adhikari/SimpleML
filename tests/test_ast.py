import unittest
from typing import cast

from src.ast import (
    EvaluateStatement,
    LoadStatement,
    Pipeline,
    PredictStatement,
    PreprocessStatement,
    SummaryStatement,
    TrainStatement,
    build_ast_from_parse_tree,
)
from src.parser import parse_statement


class AstConversionTests(unittest.TestCase):
    def test_ast_from_load_statement(self) -> None:
        tree = parse_statement('load dataset from "dataset.csv"')
        ast_node = build_ast_from_parse_tree(tree)
        self.assertIsInstance(ast_node, LoadStatement)
        ast_node = cast(LoadStatement, ast_node)
        self.assertEqual(ast_node.dataset_name, "dataset")
        self.assertEqual(ast_node.source_path, "dataset.csv")

    def test_ast_from_preprocess_statement(self) -> None:
        tree = parse_statement("preprocess dataset using target")
        ast_node = build_ast_from_parse_tree(tree)
        self.assertIsInstance(ast_node, PreprocessStatement)
        ast_node = cast(PreprocessStatement, ast_node)
        self.assertEqual(ast_node.dataset_name, "dataset")
        self.assertEqual(ast_node.target_name, "target")

    def test_ast_from_train_statement(self) -> None:
        tree = parse_statement("train model using linear_regression on dataset")
        ast_node = build_ast_from_parse_tree(tree)
        self.assertIsInstance(ast_node, TrainStatement)
        ast_node = cast(TrainStatement, ast_node)
        self.assertEqual(ast_node.model_name, "model")
        self.assertEqual(ast_node.model_type, "linear_regression")
        self.assertEqual(ast_node.dataset_name, "dataset")

    def test_ast_from_predict_statement(self) -> None:
        tree = parse_statement("predict model using checkpoint")
        ast_node = build_ast_from_parse_tree(tree)
        self.assertIsInstance(ast_node, PredictStatement)
        ast_node = cast(PredictStatement, ast_node)
        self.assertEqual(ast_node.model_name, "model")
        self.assertEqual(ast_node.input_data, "checkpoint")

    def test_ast_from_evaluate_statement(self) -> None:
        tree = parse_statement("evaluate model using metrics")
        ast_node = build_ast_from_parse_tree(tree)
        self.assertIsInstance(ast_node, EvaluateStatement)
        ast_node = cast(EvaluateStatement, ast_node)
        self.assertEqual(ast_node.model_name, "model")
        self.assertEqual(ast_node.metrics_name, "metrics")

    def test_ast_from_summary_statement(self) -> None:
        tree = parse_statement("summary model")
        ast_node = build_ast_from_parse_tree(tree)
        self.assertIsInstance(ast_node, SummaryStatement)
        ast_node = cast(SummaryStatement, ast_node)
        self.assertEqual(ast_node.target_name, "model")

    def test_ast_from_multiple_statements(self) -> None:
        trees = [
            parse_statement('load dataset from "dataset.csv"'),
            parse_statement("summary model"),
        ]
        ast_node = build_ast_from_parse_tree(trees)
        self.assertIsInstance(ast_node, Pipeline)
        self.assertEqual(len(ast_node.statements), 2)
        self.assertIsInstance(ast_node.statements[0], LoadStatement)
        self.assertIsInstance(ast_node.statements[1], SummaryStatement)

    def test_ast_rejects_malformed_parse_tree(self) -> None:
        with self.assertRaises(ValueError):
            build_ast_from_parse_tree(("statement", ("load", "dataset")))


if __name__ == "__main__":
    unittest.main()
