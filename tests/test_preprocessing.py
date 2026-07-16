import csv
import tempfile
import unittest
from pathlib import Path

from src.ast import build_ast_from_parse_tree
from src.evaluator import RuntimeState, evaluate_ast
from src.parser import parse_statement
from src.preprocessing import apply_preprocessing, preprocess_dataset


def _read_csv(path: str) -> tuple[list[str], list[list[str]]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.reader(handle))
    return all_rows[0], all_rows[1:]


def _make_dataset(temp_dir: Path, name: str, header, rows) -> Path:
    path = temp_dir / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


class PreprocessingTests(unittest.TestCase):
    def test_applies_imputation_normalization_and_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = _make_dataset(
                temp_path,
                "mixed.csv",
                ["num", "cat", "target"],
                [
                    ["1.0", "a", "0"],
                    ["3.0", "b", "1"],
                    ["", "a", "0"],
                    ["5.0", "", "1"],
                ],
            )
            original = open(dataset_path, "r", encoding="utf-8").read()

            processed_path, info = apply_preprocessing(str(dataset_path), "target")

            # Original CSV is untouched.
            self.assertEqual(open(dataset_path, "r", encoding="utf-8").read(), original)

            header, rows = _read_csv(processed_path)
            self.assertEqual(header, ["num", "cat", "target"])

            # No missing values remain.
            for row in rows:
                self.assertNotIn("", row)

            self.assertIn("simple_imputation", info["operations"])
            self.assertIn("z_score_normalization", info["operations"])
            self.assertIn("label_encoding", info["operations"])

    def test_zscore_normalizes_numeric_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = _make_dataset(
                temp_path,
                "num.csv",
                ["num", "target"],
                [["1.0", "0"], ["2.0", "0"], ["3.0", "0"]],
            )
            processed_path, _ = apply_preprocessing(str(dataset_path), "target")
            _, rows = _read_csv(processed_path)
            values = [float(row[0]) for row in rows]
            mean = sum(values) / len(values)
            self.assertAlmostEqual(mean, 0.0, places=6)
            self.assertAlmostEqual(max(values), -min(values), places=6)

    def test_target_column_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = _make_dataset(
                temp_path,
                "t.csv",
                ["num", "target"],
                [["1.0", "7"], ["2.0", "8"], ["3.0", "9"]],
            )
            processed_path, _ = apply_preprocessing(str(dataset_path), "target")
            _, rows = _read_csv(processed_path)
            self.assertEqual([row[1] for row in rows], ["7", "8", "9"])

    def test_preprocess_command_updates_dataset_to_cache_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = _make_dataset(
                temp_path,
                "reg.csv",
                ["feature", "target"],
                [["1.0", "2.0"], ["2.0", "4.0"], ["", "6.0"]],
            )
            original = open(dataset_path, "r", encoding="utf-8").read()

            ast_node = build_ast_from_parse_tree(
                [
                    parse_statement(f'load dataset from "{dataset_path}"'),
                    parse_statement('preprocess dataset using "target"'),
                ]
            )
            state = evaluate_ast(ast_node)

            self.assertEqual(open(dataset_path, "r", encoding="utf-8").read(), original)
            self.assertNotEqual(state.datasets["dataset"], str(dataset_path))
            self.assertTrue(
                str(Path(state.datasets["dataset"])).endswith("_preprocessed.csv")
            )
            self.assertIn("processed_path", state.preprocesses["dataset"])
            self.assertEqual(state.preprocesses["dataset"]["target_name"], "target")

    def test_preprocess_dataset_helper_returns_info(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = _make_dataset(
                temp_path,
                "h.csv",
                ["num", "target"],
                [["1.0", "0"], ["2.0", "1"]],
            )
            info = preprocess_dataset(str(dataset_path), "target")
            self.assertIn("processed_path", info)
            self.assertTrue(Path(info["processed_path"]).exists())


class CacheCleanupTests(unittest.TestCase):
    def test_clean_preprocessing_cache_removes_files(self) -> None:
        from src.preprocessing import CACHE_ROOT, clean_preprocessing_cache

        cache_dir = CACHE_ROOT
        cache_dir.mkdir(parents=True, exist_ok=True)
        leftover = cache_dir / "leftover_preprocessed.csv"
        leftover.write_text("a,b\n1,2\n", encoding="utf-8")

        try:
            clean_preprocessing_cache()
            self.assertFalse(leftover.exists())
        finally:
            leftover.unlink(missing_ok=True)

    def test_clean_preprocessing_cache_is_safe_when_missing(self) -> None:
        from src.preprocessing import clean_preprocessing_cache

        clean_preprocessing_cache(Path(tempfile.mkdtemp()) / "does_not_exist")


class PreprocessPathRepointTests(unittest.TestCase):
    def test_metadata_points_to_original_dataset_not_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = _make_dataset(
                temp_path,
                "orig.csv",
                ["feature", "target"],
                [["1.0", "2.0"], ["2.0", "4.0"], ["", "6.0"]],
            )
            ast_node = build_ast_from_parse_tree(
                [
                    parse_statement(f'load dataset from "{dataset_path}"'),
                    parse_statement('preprocess dataset using "target"'),
                    parse_statement("train model using linear_regression on dataset"),
                ]
            )
            state = evaluate_ast(ast_node)

            # Downstream training still used the processed copy internally.
            self.assertIn("_preprocessed.csv", state.datasets["dataset"])
            # Metadata records the original dataset path.
            metadata_path = Path(state.runs["model"]["metadata_path"])
            import json

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["dataset_path"], str(dataset_path))
            self.assertNotIn("_preprocessed.csv", metadata["dataset_path"])


if __name__ == "__main__":
    unittest.main()
