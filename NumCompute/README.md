# NumCompute

NumCompute is a small toolkit built only on Python and NumPy. It loads CSV data, preprocesses it,
sorts, ranks and searches it, computes statistics and evaluation metrics, and estimates gradients
by finite differences. Work is done with whole-array NumPy operations, inputs are checked, and the
numerical routines are written to stay accurate on very large and very small values.

It was written for COMP 5004, Assignment 2.1, Programming Task 1, by Edward Coombs, Jaidev
Parthiban and Sai Siriza Gaddam.

Assignment 2.2 extends the toolkit with `numcompute_stream`, streaming machine learning (a
decision tree retrained from a buffer, running statistics and test-then-train evaluation) built on
the modules below. See [numcompute_stream/README.md](./numcompute_stream/README.md).

## Project setup

### 1. Install Conda

If you don't already have Conda installed, grab **Miniconda**:

- [Miniconda downloads](https://docs.conda.io/en/latest/miniconda.html)

Verify the install:

```bash
conda --version
```

### 2. Create and activate the virtual environment

From the project root:

```bash
conda create -n numComp_env python=3.12
conda activate numComp_env
```

### 3. Install the package (editable mode)

With the environment active, install `numcompute` and its dependencies (NumPy, pytest, pytest-cov and flake8 are listed in `pyproject.toml`):

```bash
pip install -e .
```

### 4. VSCode setup

1. Install the **Python**, **Pylance**, and **Jupyter** extentions.
2. Open the command palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) → **Python: Select Interpreter** → choose the `numComp_env` conda environment.
3. If IntelliSense/linting doesn't pick up the change right away, run **Developer: Reload Window** from the command palette.
4. Confirm the correct interpreter is active by opening a new integrated terminal (it should auto-activate `numComp_env`) and running:

   ```bash
   which python   # macOS/Linux
   where python   # Windows
   ```

   This should point into your conda env's directory, not the system Python.

### 5. Deactivating / removing the environment

```bash
conda deactivate
conda env remove -n numComp_env   # if you need to start fresh
```

## Quick example

```python
import numpy as np
from numcompute import metrics, preprocessing, rank, stats
from numcompute.pipeline import Pipeline

X = np.array([[1.0, 10.0], [3.0, np.nan], [5.0, 30.0]])

pipe = Pipeline([
    ("impute", preprocessing.SimpleImputer(fill_value=20)),
    ("scale", preprocessing.StandardScaler()),
])
X_scaled = pipe.fit_transform(X)
print(stats.mean(X_scaled, axis=0))                 # [0. 0.]

print(metrics.f1([1, 0, 1, 1], [1, 0, 0, 1]))       # 0.8
print(rank.rank([10, 20, 20, 5]))                   # [2.  3.5 3.5 1. ]
```

`demo/quickstart.ipynb` goes further: it loads a CSV with missing values, runs a pipeline, and
shows sorting, ranking, statistics, metrics, gradients and a loop-versus-NumPy benchmark, with
assertions that check each result.

## API overview

| Module | Main functions and classes |
|---|---|
| `io` | `load_csv`: read a CSV with a chosen delimiter, and skip or fill missing values. `dtype=object` keeps text columns. |
| `preprocessing` | `StandardScaler`, `MinMaxScaler`, `OneHotEncoder` and `SimpleImputer`, each with `fit`, `transform` and `fit_transform` |
| `pipeline` | `Pipeline` chains transformers and an optional final model. `Transformer` and `Estimator` are the base classes. |
| `sort_search` | `stable_sort`, `multikey_sort`, `topk` (uses `np.argpartition`), `quickselect` and `binary_search` |
| `rank` | `rank` with `average`, `dense` or `ordinal` ties, and `percentile` with `linear`, `lower`, `higher` or `midpoint` |
| `stats` | `mean`, `std`, `min`, `max`, `median`, `quantile` and `histogram` with `axis` and `skipna`, and `RunningStats` for streaming data |
| `metrics` | `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`, `mse`, `roc_curve` and `auc` |
| `optim` | `grad` and `jacobian` by central or forward finite differences |
| `utils` | `pairwise_distances`, `softmax`, `log_softmax`, `logsumexp`, `sigmoid`, `relu`, `leaky_relu`, `tanh`, `row_topk`, `batch_indices` and `batches` |
| `benchmarking` | `time_it`, `compare`, `report`, `compare_loop_vs_vectorised` and `environment_info` |

Transformers share one interface. `fit(X, y=None)` returns the fitted object, `transform(X)`
returns a new float64 array, and `fit_transform(X)` does both. `fit` accepts `y` and ignores it,
because `Pipeline` passes `y` to every step. Calling `transform` before `fit` raises a
`RuntimeError`.

Public functions check their inputs and raise a `ValueError` or `TypeError` with a message for
the wrong shapes, types and values they test for. Each public function's docstring lists its
parameters, return value, exceptions, and time and space complexity.

## Design notes

- **Vectorisation.** Loops are kept only over columns, features, coordinates or rounds, never over
  rows. `OneHotEncoder` loops over columns because each has its own number of categories.
  `optim` loops over coordinates because the function takes one point at a time. Euclidean
  distance loops over features, so no (n, m, d) array is built.
- **Numerical stability.** `softmax`, `log_softmax` and `logsumexp` subtract the maximum first.
  Quantiles and percentiles interpolate in a form that does not overflow for -1e308 and 1e308.
  Euclidean distance uses `np.hypot`, because the shortcut `|x|² + |y|² - 2x·y` gives 0 for
  the points 1e9 and 1e9 + 1. `StandardScaler` and `MinMaxScaler` handle values near 1e308.
  `RunningStats` uses Welford's update.
- **Repeatable results.** Sorting is stable, so tied values keep their original order, and random
  steps such as `quickselect` and `batches` take a seed.

## Running tests

Tests are written using **pytest** and live under the `tests/` directory.

### 1. Run the full test suite

From the project root:

```bash
pytest
```

### 2. Run tests for a single module

```bash
pytest tests/test_io.py
```

### 3. Run a single test function

```bash
pytest tests/test_io.py::test_tab_delimiter
```

### 4. Verbose output

Shows each test name and pass/fail status individually:

```bash
pytest -v
```

### 5. Stop on first failure

Useful when debugging — stops immediately instead of running the whole suite:

```bash
pytest -x
```

### 6. Show print statements / output

By default pytest captures stdout; use `-s` to see it:

```bash
pytest -s
```

### 7. Check test coverage

```bash
pytest --cov=numcompute tests/
```

For a more detailed line-by-line report:

```bash
pytest --cov=numcompute --cov-report=term-missing tests/
```

### 8. Running tests in VSCode

1. Open the command palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) → **Python: Configure Tests** → select **pytest** → select the `tests/` directory.
2. Tests will then appear in the **Testing** sidebar (flask icon), where you can run/debug individual tests or the whole suite with one click.

### Test files

`tests/` has a test file for each module, plus:

- `test_stream_*.py`: the five test files of `numcompute_stream` (Assignment 2.2)
- `test_integration.py`: io, pipeline, metrics, stats and rank used together
- `test_review_fixes.py`: one regression test for each bug found in code review
- `test_additional_utils_benchmarking.py`: extra edge cases for batching, activations and
  benchmarking

The tests cover empty arrays, all-equal values and ties, extreme `k`, NaN, non-contiguous slices
and extreme magnitudes. GitHub Actions runs the test suite on every pull request.

## Benchmarks and demo

| Task | Command |
|---|---|
| Run the benchmarks | `python benchmark/run_all.py` (writes `benchmark/results.md`) |
| Quick benchmark run | `python benchmark/run_all.py --quick` (does not write a file) |
| Check code style | `flake8` |
| Open the demo | open `demo/quickstart.ipynb` in VS Code or Jupyter and run all cells |
| Streaming benchmark (Assignment 2.2) | `python benchmark/bench_stream.py` (also part of `run_all.py`; output in `benchmark/results_stream.md`) |
| Streaming demo (Assignment 2.2) | open `demo/stream_demo.ipynb` (data from `python data/make_stream_data.py`) |

On Windows, use `py` in place of `python` if `python` is not found. Run every command from the `NumCompute` folder (the one that contains `numcompute/`, `tests/`, `benchmark/` and `demo/`).

## Performance

`benchmark/run_all.py` compares each routine with a loop version and checks both give the same
answer before timing. Each time is the median of 5 runs after a warm-up run (the slow AUC loop is
timed once). These results are from Windows 11 on an Intel CPU, with Python 3.14.7 and NumPy
2.5.2. Speed-ups depend on the machine, the Python and NumPy versions and the input sizes, so
re-run the script to get your own numbers.

| Routine | Input | Loop | NumCompute | Speed-up |
|---|---|---|---|---|
| `roc_curve` + `auc` | n = 3,000 | 48.91 ms | 0.18 ms | 271.7x |
| `RunningStats` (100 chunks) | n = 500,000 | 35.07 ms | 1.87 ms | 18.8x |
| `pairwise_distances` | 300 x 8 | 89.70 ms | 5.45 ms | 16.5x |
| `median` (axis=0) | 2,000 x 200 | 30.28 ms | 3.85 ms | 7.9x |
| `percentile` (axis=0) | 50,000 x 20 | 114.94 ms | 15.03 ms | 7.6x |
| `rank` (average ties) | n = 200,000 | 107.90 ms | 18.53 ms | 5.8x |
| `topk` (k = 10) vs `heapq` | n = 2,000,000 | 50.81 ms | 10.46 ms | 4.9x |
| `softmax` (row loop) | 50,000 x 50 | 126.35 ms | 26.64 ms | 4.7x |
| `f1` (macro) | n = 200,000 | 40.12 ms | 8.65 ms | 4.6x |
| `multikey_sort` vs `sorted` | n = 200,000 | 90.72 ms | 22.10 ms | 4.1x |
| `histogram` | n = 500,000 | 52.55 ms | 13.76 ms | 3.8x |
| `binary_search` vs `bisect` | 200,000 queries | 62.86 ms | 18.75 ms | 3.4x |
| `grad` (forward) | dim = 200 | 1.50 ms | 0.56 ms | 2.7x |
| `confusion_matrix` | n = 200,000 | 10.05 ms | 8.42 ms | 1.2x |

The `topk`, `multikey_sort` and `binary_search` baselines use Python's `heapq`, `sorted` and
`bisect`, which run in C. The `softmax` baseline calls NumPy once per row. `quickselect` is
written for teaching and is slower than `np.sort`. Full results, including the stability and
finite-difference checks, are in `benchmark/results.md`.

## Contributors

| Member | Contribution |
|---|---|
| Edward Coombs | Repository and package set-up, `io`, `preprocessing` and `pipeline` with their tests, CI and test configuration, the setup and testing sections of this README, the first version of the demo notebook, and the demo video |
| Jaidev Parthiban | `metrics`, `optim`, `stats`, `rank`, `sort_search`, `utils` and `benchmarking` with their tests, the benchmark scripts, the code review with its regression and integration tests, and the examples, API overview and performance sections of this README |
| Sai Siriza Gaddam | `tests/test_additional_utils_benchmarking.py` and the finished `demo/quickstart.ipynb` |
