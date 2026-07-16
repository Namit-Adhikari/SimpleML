import unittest
from pathlib import Path

from src.ast import build_ast_from_parse_tree
from src.evaluator import RuntimeState, evaluate_ast
from src.parser import parse_statement
from tests.helpers import FIXTURE_CSV, get_fixture_csv_info


class EvaluatorTests(unittest.TestCase):
    def test_evaluate_load_statement_registers_dataset(self) -> None:
        csv_path, csv_name, _ = get_fixture_csv_info()
        ast_node = build_ast_from_parse_tree(
            parse_statement(f'load dataset from "{csv_path.as_posix()}"')
        )
        state = evaluate_ast(ast_node)

        self.assertIsInstance(state, RuntimeState)
        self.assertEqual(Path(state.datasets["dataset"]).name, csv_name)

    def test_evaluate_pipeline_executes_statements_in_order(self) -> None:
        csv_path, csv_name, _ = get_fixture_csv_info()
        ast_node = build_ast_from_parse_tree(
            [
                parse_statement(f'load dataset from "{csv_path.as_posix()}"'),
                parse_statement("train model using linear_regression on dataset"),
                parse_statement("summary model"),
            ]
        )
        state = evaluate_ast(ast_node)

        self.assertEqual(Path(state.datasets["dataset"]).name, csv_name)
        self.assertEqual(state.models["model"]["model_type"], "linear_regression")
        self.assertEqual(state.last_model_name, "model")
        self.assertTrue(state.summaries)
        self.assertIn("model", state.summaries[-1])

    def test_evaluate_preprocess_and_evaluate_store_context(self) -> None:
        csv_path, _, target_name = get_fixture_csv_info()
        ast_node = build_ast_from_parse_tree(
            [
                parse_statement(f'load dataset from "{csv_path.as_posix()}"'),
                parse_statement(f"preprocess dataset using {target_name}"),
                parse_statement("evaluate model using metrics"),
            ]
        )
        state = evaluate_ast(ast_node)

        self.assertEqual(state.preprocesses["dataset"]["target_name"], target_name)
        self.assertEqual(state.preprocesses["dataset"]["task_type"], "regression")
        self.assertEqual(state.metrics["metrics"]["model_name"], "model")
        self.assertEqual(state.models["model"]["project_name"], "regression_regression")

    def test_load_resolves_relative_csv_from_data_directory(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        data_dir = repo_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        csv_name = FIXTURE_CSV.name
        target_csv = data_dir / csv_name
        target_csv.write_text(FIXTURE_CSV.read_text(encoding="utf-8"), encoding="utf-8")

        try:
            ast_node = build_ast_from_parse_tree(
                parse_statement(f'load dataset from "{csv_name}"')
            )
            state = evaluate_ast(ast_node)
            self.assertEqual(Path(state.datasets["dataset"]).name, csv_name)
        finally:
            target_csv.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
