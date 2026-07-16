import unittest

from src.grammar import (
    DSLGrammarSpec,
    SIMPLE_IMPUTATION_BEHAVIOR,
    SUPPORTED_COMMANDS,
    SUPPORTED_EVALUATION_METRICS,
    SUPPORTED_MODEL_TYPES,
    SUPPORTED_PREPROCESSING_OPERATIONS,
    SUPPORTED_TASK_TYPES,
    build_ebnf_spec,
)


class GrammarDefinitionTests(unittest.TestCase):
    def test_supported_commands_are_minimal_and_explicit(self) -> None:
        self.assertEqual(
            SUPPORTED_COMMANDS,
            ("load", "preprocess", "train", "predict", "evaluate", "summary"),
        )

    def test_supported_models_include_random_forest(self) -> None:
        self.assertEqual(
            SUPPORTED_MODEL_TYPES,
            ("linear_regression", "logistic_regression", "random_forest"),
        )

    def test_supported_tasks_are_classification_and_regression(self) -> None:
        self.assertEqual(SUPPORTED_TASK_TYPES, ("classification", "regression"))

    def test_supported_preprocessing_operations_are_explicit(self) -> None:
        self.assertEqual(
            SUPPORTED_PREPROCESSING_OPERATIONS,
            (
                "z_score_normalization",
                "one_hot_encoding",
                "label_encoding",
                "duplicate_removal",
                "simple_imputation",
            ),
        )

    def test_simple_imputation_uses_median_for_numerical_and_mode_for_categorical(
        self,
    ) -> None:
        self.assertEqual(SIMPLE_IMPUTATION_BEHAVIOR["numerical"], "median")
        self.assertEqual(SIMPLE_IMPUTATION_BEHAVIOR["categorical"], "mode")

    def test_supported_evaluation_metrics_are_basic(self) -> None:
        self.assertEqual(
            SUPPORTED_EVALUATION_METRICS,
            ("accuracy", "mae", "rmse", "r2"),
        )

    def test_grammar_spec_contains_required_statements(self) -> None:
        spec = DSLGrammarSpec()
        self.assertIn("statement", spec.rules)
        self.assertIn("load_statement", spec.rules)
        self.assertIn("preprocess_statement", spec.rules)
        self.assertIn("train_statement", spec.rules)
        self.assertIn("predict_statement", spec.rules)
        self.assertIn("evaluate_statement", spec.rules)
        self.assertIn("summary_statement", spec.rules)

    def test_ebnf_rendering_contains_command_names(self) -> None:
        ebnf = build_ebnf_spec()
        self.assertIn("load", ebnf)
        self.assertIn("preprocess", ebnf)
        self.assertIn("summary", ebnf)


if __name__ == "__main__":
    unittest.main()
