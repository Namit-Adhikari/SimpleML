# SimpleML

SimpleML is a minimal Domain Specific Language (DSL) for building end-to-end tabular machine learning pipelines. The DSL uses a compact, English-style command vocabulary and is implemented with [PLY](https://github.com/dabeaz/ply) for lexing and parsing.

## Requirements

- Python 3.10 or newer

## Installation

### Option 1: With uv (recommended)

If you don't have uv installed, install it first from [uv's official website](https://docs.astral.sh/uv/).

Clone the repository and install dependencies:

```bash
uv sync
```

### Option 2: With pip

Clone the repository and install in development mode:

```bash
pip install -e .
```

Or install directly from the repository:
```bash
pip install git+https://github.com/your-username/simpleml.git
```

## Quick start

### Run a pipeline script

With uv:
```bash
uv run simpleml run titanic.dsl
```

With pip (or if installed):
```bash
simpleml run titanic.dsl
```

Or directly:
```bash
python main.py run titanic.dsl
```

### Start the interactive REPL

With uv:
```bash
uv run simpleml repl
```

With pip:
```bash
simpleml repl
```

Or directly:
```bash
python main.py repl
```

### Use the Python API

```python
from simpleml.runtime import execute_file
from simpleml.repl import run_repl

state = execute_file("titanic.dsl")
run_repl()
```

## DSL command vocabulary

Each line in a `.dsl` script is one statement. Blank lines and lines starting with `#` are ignored.

| Command | Syntax | Description |
|---------|--------|-------------|
| `load` | `load <dataset> from "<path>"` | Load a CSV file into a named dataset |
| `preprocess` | `preprocess <dataset> using <target>` | Register preprocessing with an inferred target column |
| `train` | `train <model> using <model_type> on <dataset>` | Train a model on a loaded dataset |
| `predict` | `predict <model> using <input_data>` | Generate predictions from a trained model using a single input dict, list of dicts, or a CSV file path |
| `evaluate` | `evaluate <model> using <metrics_name>` | Compute evaluation metrics for a trained model |
| `summary` | `summary <model_name>` | Show a detailed summary of the model and pipeline state |

### Supported model types

| Model Type | Task Type |
|------------|-----------|
| `linear_regression` | Regression |
| `logistic_regression` | Classification |
| `random_forest` | Classification / Regression |

### Supported evaluation metrics

- Classification: `accuracy`
- Regression: `r2`, `mse`, `rmse`

## Pipeline scripts and data

CSV files belong in `data/`. DSL scripts belong in `scripts/`. Both directories may be empty until you add your own files.

| Script | Dataset | Task | Model |
|--------|---------|------|-------|
| `titanic.dsl` | `titanic.csv` | Classification (`Survived`) | `logistic_regression` |

Relative CSV paths in DSL scripts are resolved against the `data/` directory.

Run a script:
```bash
uv run simpleml run titanic.dsl
```

## REPL usage

The REPL executes one DSL statement per line and preserves state across commands:

```
SimpleML REPL
Type 'help' for available commands, 'exit' to quit.
SimpleML> load titanic from "titanic.csv"
Executed: load titanic from "titanic.csv"
Loaded dataset
SimpleML> preprocess titanic using "Survived"
Executed: preprocess titanic using "Survived"
SimpleML> train titanic_model using logistic_regression on titanic
Executed: train titanic_model using logistic_regression on titanic
SimpleML> summary titanic_model
... (summary output) ...
SimpleML> exit
```

Special REPL commands:
- `help` - Show help message
- `clear` - Clear the console screen
- `exit` / `quit` - Exit the REPL

## Artifact storage

Training, evaluation, and prediction persist results under a structured layout:

```
results/
└── <project_slug>/
    └── <run_id>/
        ├── metadata.json       # Run metadata (dataset, task, model info, etc.)
        ├── train/
        │   └── model.pkl       # Trained model checkpoint (pickle)
        ├── evaluate/
        │   ├── metrics.json    # Evaluation metrics
        │   └── confusion_matrix.png (classification only)
        └── predict/
            └── 001/
                └── predictions.csv
```

The project slug is inferred from the training CSV filename and task type (e.g., "titanic_classification"). Generated results are excluded from version control via `.gitignore`.

## Repository layout

```
simpleml/        Main package (parser, AST, evaluator, runtime, and REPL)
tests/           Unit and integration tests (fixtures under tests/fixtures/)
data/            User CSV datasets
scripts/         User DSL pipeline scripts
results/         Pipeline outputs (train, evaluate, predict)
main.py          CLI entry point
pyproject.toml   Project configuration and metadata
AGENTS.md        Contributor and agent instructions
```

## Development

### Run tests

With uv:
```bash
uv run python -m unittest discover -s tests -v
```

With pip:
```bash
python -m unittest discover -s tests -v
```

Run a single test module:
```bash
python -m unittest tests.test_parser -v
```

### Grammar reference

The formal EBNF specification is defined in `simpleml.grammar`. Inspect it programmatically:

```python
from simpleml.grammar import build_ebnf_spec
print(build_ebnf_spec())
```

## License

See [LICENSE](LICENSE).

