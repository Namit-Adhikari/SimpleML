import unittest

from src.parser import ParseError, parse_statement


class ParserTests(unittest.TestCase):
    def test_parse_load_statement(self) -> None:
        statement = parse_statement('load dataset from "dataset.csv"')
        self.assertEqual(statement, ("statement", ("load", "dataset", "dataset.csv")))

    def test_parse_preprocess_statement(self) -> None:
        statement = parse_statement("preprocess dataset using target")
        self.assertEqual(statement, ("statement", ("preprocess", "dataset", "target")))

    def test_parse_train_statement(self) -> None:
        statement = parse_statement("train model using linear_regression on dataset")
        self.assertEqual(
            statement, ("statement", ("train", "model", "linear_regression", "dataset"))
        )

    def test_parse_predict_statement(self) -> None:
        statement = parse_statement("predict model using checkpoint")
        self.assertEqual(statement, ("statement", ("predict", "model", "checkpoint")))

    def test_parse_evaluate_statement(self) -> None:
        statement = parse_statement("evaluate model using metrics")
        self.assertEqual(statement, ("statement", ("evaluate", "model", "metrics")))

    def test_parse_summary_statement(self) -> None:
        statement = parse_statement("summary model")
        self.assertEqual(statement, ("statement", ("summary", "model")))

    def test_parse_rejects_invalid_statement(self) -> None:
        with self.assertRaises(ParseError):
            parse_statement("invalid syntax here")

    def test_parse_multiple_statements_from_script_text(self) -> None:
        source = 'load dataset from "dataset.csv"\nsummary model\n'
        parsed_lines = [parse_statement(line) for line in source.splitlines() if line.strip()]
        self.assertEqual(len(parsed_lines), 2)
        self.assertEqual(parsed_lines[0], ("statement", ("load", "dataset", "dataset.csv")))
        self.assertEqual(parsed_lines[1], ("statement", ("summary", "model")))


if __name__ == "__main__":
    unittest.main()
