import io
import unittest
from contextlib import redirect_stdout

from src.repl import REPL, run_repl
from tests.helpers import get_fixture_csv_info


class ReplTests(unittest.TestCase):
    def test_repl_quits_on_exit_command(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            run_repl(iter(["exit"]))
        self.assertIn("SimpleML REPL", output.getvalue())

    def test_repl_executes_single_statement(self) -> None:
        csv_path, _, _ = get_fixture_csv_info()
        output = io.StringIO()
        with redirect_stdout(output):
            run_repl(iter([f'load dataset from "{csv_path.as_posix()}"', "exit"]))
        self.assertIn("Loaded dataset", output.getvalue())


if __name__ == "__main__":
    unittest.main()
