import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src import results_layout
from src.results_layout import (
    build_project_name,
    create_run_layout,
    infer_task_type,
    next_predict_dir,
    write_json_if_changed,
)
from tests.helpers import FIXTURE_CSV


class ResultsLayoutTests(unittest.TestCase):
    def test_build_project_name_uses_dataset_stem_and_task_type(self) -> None:
        project_name = build_project_name("data/iris.csv", "classification")
        self.assertEqual(project_name, "iris_classification")

    def test_infer_task_type_detects_regression_for_continuous_target(self) -> None:
        task_type = infer_task_type(str(FIXTURE_CSV), "target")
        self.assertEqual(task_type, "regression")

    def test_create_run_layout_builds_expected_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_context = create_run_layout(
                "regression_regression",
                {
                    "task_type": "regression",
                    "dataset_name": "dataset",
                    "model_name": "model",
                    "model_type": "linear_regression",
                },
                results_root=temp_path,
                timestamp=datetime(2026, 7, 13, 18, 45, 12),
            )

            run_dir = Path(run_context["run_dir"])
            self.assertTrue((run_dir / "metadata.json").exists())
            self.assertTrue((run_dir / "train").is_dir())
            self.assertTrue((run_dir / "evaluate").is_dir())
            self.assertTrue((run_dir / "predict").is_dir())
            self.assertEqual(run_dir.parent.name, "regression_regression")
            self.assertEqual(run_dir.name, "2026-07-13_18-45-12")
            self.assertEqual(run_dir.parent.parent, temp_path)

    def test_next_predict_dir_allocates_incrementing_folders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_context = create_run_layout(
                "regression_regression",
                {"task_type": "regression"},
                results_root=temp_path,
                timestamp=datetime(2026, 7, 13, 18, 45, 12),
            )

            first = next_predict_dir(run_context)
            second = next_predict_dir(run_context)

            self.assertEqual(first.name, "001")
            self.assertEqual(second.name, "002")
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())

    def test_write_json_if_changed_skips_unchanged_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            metrics_path = temp_path / "metrics.json"
            payload = {"score": 0.95}

            self.assertTrue(write_json_if_changed(metrics_path, payload))
            self.assertFalse(write_json_if_changed(metrics_path, payload))

            updated_payload = {"score": 0.96}
            self.assertTrue(write_json_if_changed(metrics_path, updated_payload))
            saved = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["score"], 0.96)


if __name__ == "__main__":
    unittest.main()
