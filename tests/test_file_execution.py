import shutil
import tempfile
import unittest
from pathlib import Path

from src.evaluator import RuntimeState, evaluate_ast
from src.parser import ParseError
from src.runtime import execute_file
from tests.helpers import (
    FIXTURE_CSV,
    FIXTURE_PIPELINE_SCRIPT,
    get_fixture_csv_info,
)


class FileExecutionTests(unittest.TestCase):
    def test_execute_file_parses_and_executes_statements(self) -> None:
        csv_path, csv_name, _ = get_fixture_csv_info()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            local_csv = temp_path / csv_name
            shutil.copy2(csv_path, local_csv)
            script_path = temp_path / "pipeline.dsl"
            script_path.write_text(
                f'load dataset from "{local_csv.as_posix()}"\n'
                "train model using linear_regression on dataset\n"
                "summary model\n",
                encoding="utf-8",
            )

            state = execute_file(script_path)

            self.assertIsInstance(state, RuntimeState)
            self.assertEqual(Path(state.datasets["dataset"]).name, csv_name)
            self.assertEqual(state.models["model"]["model_type"], "linear_regression")
            self.assertTrue(state.summaries)

    def test_execute_file_resolves_bare_name_from_scripts_directory(self) -> None:
        csv_path, csv_name, _ = get_fixture_csv_info()
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "sample_user_script.dsl"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(
            f'load dataset from "{csv_path}"\nsummary model\n', encoding="utf-8"
        )

        try:
            state = execute_file("sample_user_script.dsl")
            self.assertTrue(state.summaries)
        finally:
            script_path.unlink(missing_ok=True)

    def test_execute_file_rejects_invalid_statement(self) -> None:
        csv_path, csv_name, _ = get_fixture_csv_info()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            shutil.copy2(csv_path, temp_path / csv_name)
            script_path = temp_path / "invalid.dsl"
            script_path.write_text(
                f'load dataset from "{csv_name}"\ninvalid syntax\n', encoding="utf-8"
            )

            with self.assertRaises(ParseError):
                execute_file(script_path)

    def test_execute_file_with_empty_script_raises_parse_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "empty.dsl"
            script_path.write_text("\n\n", encoding="utf-8")

            with self.assertRaises(ParseError):
                execute_file(script_path)

    def test_execute_file_skips_comment_lines(self) -> None:
        csv_path, csv_name, _ = get_fixture_csv_info()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            local_csv = temp_path / csv_name
            shutil.copy2(csv_path, local_csv)
            script_path = temp_path / "commented.dsl"
            script_path.write_text(
                "# load a dataset\n"
                f'load dataset from "{local_csv.as_posix()}"\n'
                "# train a model\n"
                "summary model\n",
                encoding="utf-8",
            )

            state = execute_file(script_path)

            self.assertTrue(state.summaries)

    def test_execute_fixture_pipeline_script(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            local_csv = temp_path / FIXTURE_CSV.name
            shutil.copy2(FIXTURE_CSV, local_csv)
            script_path = temp_path / "pipeline.dsl"
            script_path.write_text(
                FIXTURE_PIPELINE_SCRIPT.read_text(encoding="utf-8").replace(
                    '"regression.csv"', f'"{local_csv.as_posix()}"'
                ),
                encoding="utf-8",
            )

            state = execute_file(temp_path / "pipeline.dsl")

            self.assertEqual(
                state.preprocesses["dataset"]["target_name"], "target"
            )
            self.assertIn("model", state.models)
            self.assertIn("metrics", state.metrics)
            self.assertTrue(state.summaries)

    def test_execute_fixture_pipeline_with_absolute_csv_path(self) -> None:
        csv_path, _, target_name = get_fixture_csv_info()
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "pipeline.dsl"
            script_path.write_text(
                f'load dataset from "{csv_path.as_posix()}"\n'
                f"preprocess dataset using {target_name}\n"
                "train model using linear_regression on dataset\n"
                "evaluate model using metrics\n"
                "summary model\n",
                encoding="utf-8",
            )

            state = execute_file(script_path)

            self.assertEqual(state.preprocesses["dataset"]["target_name"], target_name)
            self.assertIn("model", state.models)
            self.assertIn("metrics", state.metrics)
            self.assertTrue(state.summaries)


if __name__ == "__main__":
    unittest.main()
