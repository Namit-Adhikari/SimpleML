# AGENTS.md

## Project overview

This repository implements SimpleML, a minimal Domain Specific Language (DSL) for end-to-end tabular machine-learning pipelines. The DSL is implemented with [PLY](https://github.com/dabeaz/ply) for lexing and parsing and supports:

- file-based pipeline execution (`simpleml run <script>`)
- an interactive REPL (`simpleml repl`)
- a Python-native runtime API (`src.runtime.execute_file`, `src.repl.run_repl`)

Scope is intentionally narrow: tabular CSV data only, with a small set of supervised models and no deep learning or non-tabular modalities.

## Repository layout

```
src/                    DSL implementation
  grammar.py            Supported command/model/task/metric/preprocessing vocabularies
  parser.py             PLY lexer and parser
  ast.py                AST node definitions and parse-tree converter
  evaluator.py          Statement evaluator and RuntimeState
  runtime.py            File execution helpers
  model_runtime.py      Training, evaluation, and prediction implementations
  results_layout.py     On-disk artifact directory layout
  repl.py               Interactive REPL
  cli.py                CLI parser and subcommands implementation
main.py                 CLI entry point wrapper
tests/                  Unit and integration tests
data/                   User-provided CSV datasets
scripts/                User-provided `.dsl` pipeline scripts
results/                Generated run artifacts
pyproject.toml          Project metadata and dependencies
uv.lock                 uv lockfile
README.md               User-facing documentation
AGENTS.md               This file
```

`pyproject.toml` maps the package root to `src/` (`package_dir = {"": "src"}`). Tests import directly from `src.<module>`. `main.py` falls back to `src.*` imports when the package is not installed.

The project is licensed under the GNU Affero General Public License v3. See `LICENSE`.

## Dependencies and environment

- Use `uv` as the primary package manager. Use `pip install -e .` only as a fallback when `uv` is unavailable.
- `pyproject.toml` declares `requires-python = ">=3.10"` and dependencies `ply>=3.11` and `matplotlib>=3.8`.
- `.python-version` is pinned to `3.14`. The test suite is currently verified on CPython 3.10; if the pinned 3.14 interpreter causes compiled dependency issues, run commands with `--python 3.10`.

Common commands:

```bash
uv sync
uv run --python 3.10 python main.py run scripts/titanic.dsl
uv run --python 3.10 python main.py repl
uv run --python 3.10 python -m unittest discover -s tests -v
```

## DSL command reference

Each line in a `.dsl` script is one statement. Blank lines and lines starting with `#` are ignored.

| Command | Syntax | Behavior |
|---|---|---|
| load | `load <dataset> from "<path>"` | Load a CSV. Relative paths resolve against `data/`; absolute paths are used as-is. The path may also be an unquoted identifier. |
| preprocess | `preprocess <dataset> using <target>` | Record the target column, infer classification or regression from it, and automatically apply the supported table transformations to a cache copy (original CSV untouched). The target may be a quoted string or an identifier. |
| train | `train <model> using <model_type> on <dataset>` | Train a model. `<model_type>` must be one of `linear_regression`, `logistic_regression`, or `random_forest`. |
| evaluate | `evaluate <model> using <metrics_name>` | Evaluate the trained model and store metrics under the run's `evaluate/` directory. If the model has not been trained, it is auto-trained with `linear_regression` on the first loaded dataset. |
| predict | `predict <model> using <input>` | Generate predictions. `<input>` may be an inline dict (`{...}`), an inline list of dicts (`[...]`), a dataset identifier, or a CSV path. |
| summary | `summary <model_name>` | Print train, evaluation, and prediction summary tables for the model. |

### Supported model types

| Type | Task |
|---|---|
| `linear_regression` | regression |
| `logistic_regression` | classification |
| `random_forest` | classification (majority-class baseline) or regression (mean-value baseline) |

### Task inference

`preprocess` infers the task type from the target column:

- If any target value is non-numeric, the task is `classification`.
- If all values are numeric and there are at most 2 unique values, the task is `classification`.
- Otherwise, the task is `regression`.

`train` overrides this inference when the chosen model type implies a task (`linear_regression` = regression, `logistic_regression` and `random_forest` = classification). `random_forest` also supports regression, in which case it falls back to a mean-value baseline.

### Evaluation metrics

The runtime emits the following metrics:

- Classification: `accuracy`
- Regression: `r2`, `mse`, `rmse`

`src/grammar.py` declares a broader `SUPPORTED_EVALUATION_METRICS` vocabulary (`accuracy`, `mae`, `rmse`, `r2`) used by tests, but the runtime currently only emits the values above.

### Preprocessing vocabulary

`src/grammar.py` defines `SUPPORTED_PREPROCESSING_OPERATIONS` (`z_score_normalization`, `one_hot_encoding`, `label_encoding`, `duplicate_removal`, `simple_imputation`) and `SIMPLE_IMPUTATION_BEHAVIOR`. These constants are used by grammar tests. The `preprocess` command performs target registration and task inference **and** automatically applies the supported table transformations. `src/preprocessing.py` implements `apply_preprocessing()` / `preprocess_dataset()`, which runs the following (target column is always preserved):

- `simple_imputation` — fill missing numeric cells with the median and missing categorical cells with the mode.
- `z_score_normalization` — standardize numeric, non-target columns to z-scores (`std=0` yields `0.0`).
- `label_encoding` — map categorical (non-numeric), non-target columns to integer ids.
- `duplicate_removal` — drop rows duplicated across all columns.

The original CSV is never modified. Transformations are written to a deterministic cache copy under `.cache/preprocessed/` (excluded from version control via `.gitignore`), and the runtime repoints `state.datasets[dataset_name]` at the processed copy so downstream `train`, `evaluate`, and `predict` stages use it transparently. `one_hot_encoding` is declared in the grammar vocabulary but is not yet auto-applied (it changes column count, which the current univariate baseline models cannot consume); prefer `label_encoding` for categorical features.

## Implementation architecture

Maintain this separation when making changes:

- `src/grammar.py` — vocabularies and the canonical EBNF string from `build_ebnf_spec()`.
- `src/parser.py` — PLY tokens and grammar rules. `parse_statement()` returns `("statement", (...))` tuples and raises `ParseError` on invalid input.
- `src/ast.py` — frozen dataclasses for each statement plus `Pipeline`. `build_ast_from_parse_tree()` converts parser tuples to AST nodes.
- `src/evaluator.py` — `RuntimeState` and `evaluate_ast()` execute AST nodes.
- `src/runtime.py` — `execute_file()` loads `.dsl` scripts, resolving bare names from `scripts/`.
- `src/model_runtime.py` — training, evaluation, and prediction for each model type.
- `src/results_layout.py` — `create_run_layout()`, `build_project_name()`, `infer_task_type()`, `next_predict_dir()`, and JSON helpers.
- `src/repl.py` — REPL that executes one DSL statement per line, with `help`, `clear`, and `exit`/`quit` commands.
- `src/cli.py` — CLI implementation (`argparse` parser for `run` and `repl`).
- `main.py` — CLI entry point wrapper.

## Model runtime and artifacts

Training uses the first available numeric feature in the order `"Fare"`, `"Pclass"`, `"Age"`, then the first non-target column. Data is split 80:20 with a fixed random seed `42` for reproducibility. Trained models and their train/test data are serialized to `train/model.pkl` using `pickle`.

A run is stored under:

```
results/<project_slug>/<run_id>/
  metadata.json
  train/
    model.pkl
  evaluate/
    metrics.json
    confusion_matrix.png   # classification only
  predict/
    001/
      predictions.csv
```

`<project_slug>` is `<dataset_stem>_<task_type>`, e.g. `titanic_classification`. Generated `results/` entries are excluded from version control via `.gitignore`; `results_layout.py` creates a `results/.gitkeep` file when writing output.

## Development workflow

1. Read the relevant files before editing.
2. Make the smallest change that satisfies the requirement.
3. Add or update tests in `tests/` covering parser, AST, evaluator, model runtime, file execution, REPL, and results layout as appropriate.
4. Run the unit-test suite and the sample pipeline before declaring success.
5. Update `README.md` and `AGENTS.md` if the change affects user-facing behavior or repository conventions.
6. Do not commit generated artifacts, `results/`, or temporary files.

## Testing

Run the full test suite with:

```bash
uv run --python 3.10 python -m unittest discover -s tests -v
```

Verify end-to-end behavior with:

```bash
uv run --python 3.10 python main.py run scripts/titanic.dsl
```

Start the REPL with:

```bash
uv run --python 3.10 python main.py repl
```

No lint tools or pre-commit hooks are configured; rely on tests and consistent code style.

## Constraints

- Keep the DSL focused on tabular CSV data and supervised learning.
- Do not expand into image, text, audio, or deep-learning systems without explicit approval.
- Do not silently change the supported Python version; update `pyproject.toml`, classifiers, and `.python-version` together if needed.
- Prefer explicit data structures, type hints, and small, composable functions.
- Favor deterministic behavior and clear error messages.
- Keep the command vocabulary compact and stable. Changes to grammar, AST, or runtime interfaces require tests and documentation updates.
- If a requested change is ambiguous or would alter the overall DSL architecture, ask for clarification before implementing.
