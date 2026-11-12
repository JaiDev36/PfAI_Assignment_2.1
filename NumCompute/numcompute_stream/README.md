# numcompute_stream

Streaming machine learning on top of NumCompute. It trains a decision tree one chunk of data at a
time, with NumPy for all computation and matplotlib only for plots. It extends the `numcompute`
package from Assignment 2.1 and was written for COMP 5004, Assignment 2.2, Programming Task 2.

```python
from numcompute_stream import (DecisionTree, StreamingImputer, StreamingPipeline, iter_chunks,
                               load_labelled_csv, prequential_run)

X, y, feature_names = load_labelled_csv("data/stream_data.csv", label_column="status")

pipe = StreamingPipeline([("imputer", StreamingImputer()),
                          ("tree", DecisionTree(max_depth=6, buffer_size=500))])
metrics = prequential_run(pipe, iter_chunks(X, y, chunk_size=200))   # predict, then train
print(metrics.accuracy, pipe.model.n_retrains_)
```

Or with the trainer from the brief (`stream.py`):

```python
from numcompute_stream import DecisionTreeClassifier, StreamTrainer

trainer = StreamTrainer(DecisionTreeClassifier(max_depth=5, buffer_size=500))
for X_chunk, y_chunk in iter_chunks(X, y, chunk_size=200):
    trainer.score_chunk(X_chunk, y_chunk)    # test on the new chunk first (None until a tree exists)
    trainer.fit_chunk(X_chunk, y_chunk)      # then train; logs time, memory and accuracy
print(trainer.cumulative_accuracy, trainer.log_[-1])
```

## How it builds on numcompute

Nothing from `numcompute` is copied. The new classes subclass or call it:

| numcompute_stream | Built on |
|---|---|
| `StreamingStandardScaler`, `StreamingMinMaxScaler` | subclasses of `numcompute.preprocessing.StandardScaler` and `MinMaxScaler`; only learning is new, so `transform` (and its overflow protection) is inherited |
| `StreamingImputer` | a `numcompute.pipeline.Transformer`; reuses `SimpleImputer` to validate `fill_value` |
| `ColumnStats` | one `numcompute.stats.RunningStats` per feature |
| `DecisionTree` | an `numcompute.pipeline.Estimator`; the split search sorts with `numcompute.sort_search.stable_sort` |
| `StreamingPipeline` | a subclass of `numcompute.pipeline.Pipeline`; `transform` and `predict` are inherited, `fit` is overridden so `y` only goes to steps that take it |
| `StreamingMetrics` | `numcompute.metrics.confusion_matrix` and `accuracy` |
| `load_labelled_csv`, `iter_chunks` | `numcompute.io.load_csv` and `numcompute.utils.batches` |
| benchmarks | `numcompute.benchmarking` (`time_it`, `compare_loop_vs_vectorised`, `environment_info`) |

The error messages match numcompute's ("Input data must have 2 dimensions. Got shape: ...",
"Incompatible dimensionality ...", "fit must be called before transform."), so the same contract
tests run against the streaming classes (`tests/test_stream_preprocessing.py`).

## Modules

| Module | Contents |
|---|---|
| `io` | `load_labelled_csv` (features, labels and feature names from a CSV) and `iter_chunks` (consecutive chunks, never shuffled) |
| `stats` | `ColumnStats` (per-feature count, mean, variance, std, min, max), `StreamingQuantiles` (quantiles over a sliding window of rows) and `StreamingHistogram` (fixed bins, optional sliding window of chunks); all have `update_stats(X_chunk)` |
| `preprocessing` | `StreamingStandardScaler`, `StreamingMinMaxScaler`, `StreamingImputer`, `StreamingOneHotEncoder` (categories grow as new ones arrive) and `LabelEncoder`; the brief's names `StandardScaler`, `Imputer` and `OneHotEncoder` are aliases |
| `metrics` | `StreamingMetrics`: running confusion matrix, `update` / `reset` / `result`, accuracy, macro precision, recall and F1, optional rolling `window`; `StreamingAUC`: binary ROC AUC accumulated from score histograms; `per_class_scores` |
| `stream` | `StreamTrainer`: `fit_chunk(X, y)`, `score_chunk(X, y)` and `run(chunks)`; logs per-chunk time, memory footprint, retrains and cumulative accuracy |
| `tree` | `DecisionTree` (alias `DecisionTreeClassifier`): from-scratch classification tree with buffered retraining, `max_depth`, `min_samples_split`, `max_features`, `buffer_size`; `best_split_vectorised` and `best_split_loop` |
| `pipeline` | `StreamingPipeline` (alias `Pipeline`) and `prequential_run` (test-then-train evaluation with logging) |
| `visualise` | matplotlib plots: `plot_metric_history`, `plot_model_comparison`, `plot_confusion_matrix`, `plot_retrain_log`, `plot_benchmark`, `plot_buffer_tradeoff`, and the three plots named in the brief: `plot_metric_over_time(metric_values, title, ylabel)`, `compare_models(metric1, metric2, labels)` and `plot_predictions_vs_ground_truth(y_true, y_pred)`; each can save to a file (`save_path`) or display inline (`show=True`) |

## One API for everything that learns

| Method | Meaning |
|---|---|
| `partial_fit(X, y=None)` | learn from one more chunk and keep what was learned before; returns `self` |
| `update(...)` | synonym of `partial_fit` |
| `fit(X, y=None)` | forget everything, then learn from `X` alone (on the same code path as `partial_fit`) |
| `reset()` | forget everything; returns `self` |
| `transform(X)` / `predict(X)` | use what has been learned so far |

Any object with these methods can be a pipeline step; it does not have to come from this package
or inherit from anything. `StreamingPipeline` gives `y` only to a `fit` or `partial_fit` whose
signature has room for it, so a third-party transformer written as `partial_fit(X)` /
`transform(X)` (or `fit(X)` / `transform(X)`) works without an adapter. In batch `fit`, a step
without `fit` is reset and given `partial_fit`. This answers a limitation found in
`numcompute.pipeline.Pipeline` during the Assignment 2.1 review (it passes `y` to every `fit`);
the group's class is left unchanged.

Conventions: `X` is always 2-D with shape `(n_samples, n_features)` (a 1-D array is an error, as in
numcompute); `y` is 1-D; `axis=0` means down the rows, one result per feature; NaN in `X` means
missing; infinity is rejected; an empty chunk is a no-op.

## How the tree works

**Growing a tree.** Try every question `x[:, f] <= t` (thresholds halfway between neighbouring
distinct values), keep the one with the largest information gain
`I(parent) - n_left/n * I(left) - n_right/n * I(right)` (Gini or entropy), split, and repeat until
`max_depth`, a pure node, too few rows, or no gain.

**Buffered retraining.** `partial_fit` appends the chunk to a buffer. When the buffer holds at
least `buffer_size` rows, a new tree is trained on the buffered rows and the buffer is cleared;
until then `predict` uses the latest tree. Each tree therefore sees only the rows that arrived since
the previous retrain (a window), which is what lets it follow concept drift. `flush()` trains on
the rows left in the buffer at the end of a stream.

**Vectorised split search.** For one feature, sort the rows. The class counts left of every cut are
the cumulative sum of the one-hot labels, and the right counts are `total - left`. Every threshold
of every feature is then scored at once, in O(n log n) per feature instead of the O(n²) of masking
the rows once per threshold (`best_split_loop` is that version, kept for the benchmark and as an
independent check; tests require both to return the same split). Features are scored in blocks so
the temporary array stays under about 64 MB. Prediction is vectorised too: the rows are routed down
the tree in groups, one comparison per node.

**Numerical stability and edge cases.**

- Ties between equally good splits go to the lowest feature, then the lowest threshold; ties between
  classes in a leaf go to the class seen first.
- A constant feature offers no threshold and is skipped; a constant column scales to a scale of 1,
  never a division by zero; an empty node has impurity 0, not NaN; `0 * log2(0)` is 0.
- NaN is replaced by the per-feature mean of the training buffer (kept and reused at prediction); a
  column that is all NaN is filled with 0. Running statistics skip NaN per feature.
- Two values one float apart still get a threshold that separates them.
- Running variance uses Welford's update, so values near 1e9 with a spread of 1 keep their precision.
- A class that first appears in a later chunk is registered immediately and gets probability 0 until
  a retrain has seen it.
- Predicting before the first tree exists raises `NotFittedError` (a `RuntimeError`) that says how
  full the buffer is.

## Tests

```bash
pytest tests/test_stream_*.py                       # the streaming tests (276)
pytest --cov=numcompute_stream --cov-report=term-missing
```

`tests/` holds five files for this package: `test_stream_preprocessing.py`,
`test_stream_tree.py`, `test_stream_buffering.py`,
`test_stream_metrics_io_pipeline.py` and `test_stream_spec_additions.py` (the names used in the
brief, `max_features`, `StreamTrainer`, windowed quantiles and histograms, rolling and AUC metrics,
the incremental one-hot encoder and the three plots). They cover the contract shared with numcompute, streaming
equal to batch for any chunk size, loop equal to vectorised, when exactly the buffer triggers a
retrain, concept drift, NaN, ties, constant columns, a new class mid-stream, bad input, third-party
pipeline steps that take no `y`, and every plot. The whole suite (numcompute's tests included) runs with warnings treated as errors.

## Benchmarks and demo

```bash
python data/make_stream_data.py          # writes data/stream_data.csv (6000 rows, drift at row 3000)
python benchmark/bench_stream.py         # add --quick for a few seconds; a full run saves figures
```

`benchmark/results_stream.md` has the output of a full run, and `benchmark/stream_figures/` the
figures. On the machine used (Python 3.12, NumPy 2.5.4, Windows 11):

| Operation | Input | Python loop | Vectorised | Speed-up |
|---|---|---|---|---|
| best-split search | n=1,600, 5 features | 211 ms | 1.3 ms | 162x |
| tree training, max_depth=5 | n=800, 5 features | 398 ms | 5.5 ms | 72x |
| tree prediction | n=100,000 | 188 ms | 12.1 ms | 15.5x |
| `ColumnStats` | n=200,000, 5 features | 87 ms | 12.8 ms | 6.8x |

`buffer_size` is a trade between how fresh the tree is and how much data it learns from. On the
drifting stream (chunks of 100 rows), a buffer of 100 gave 0.717 accuracy in the 1000 rows after the
drift, a buffer of 1000 gave 0.603, and the mean retrain time went from 1.1 ms to 8.3 ms. A tree
with a larger buffer also scores fewer rows, because nothing is predicted until the first tree
exists, so overall accuracy is not compared across buffer sizes without that in mind.

The demo, `demo/stream_demo.ipynb`, loads the CSV, chunks it, runs `partial_fit` with logging, and
plots accuracy over time, the retrain log, the confusion matrix and a static-against-buffered
comparison. It also runs the same stream through `StreamTrainer` with the three plots named in the
brief (`plot_metric_over_time`, `compare_models`, `plot_predictions_vs_ground_truth`) and shows
`StreamingOneHotEncoder` adding a column when a new category arrives.

## Limitations

- The CSV is read whole by `numcompute.io.load_csv`, and the stream is cut from arrays in memory.
  The model sees one chunk at a time, but memory is not bounded by the chunk size.
- `StreamingOneHotEncoder` grows its output width when a new category arrives, and sorted
  categories can shift later columns. A tree fed by it needs the categories known up front.
- The tree retrains from scratch on each window and forgets older windows. It does not grow
  incrementally (as a Hoeffding tree does) and handles only classification with numeric features.
- `ColumnStats` has no median or quantiles (they need every value seen); `StreamingQuantiles`
  provides them exactly over a sliding window, or over every row at the cost of unbounded memory.
- `StreamingAUC` reads the AUC from score histograms, so it is exact only up to the bin width.
- Scalers learn as chunks arrive, so the same row transforms differently before and after a
  `partial_fit`. A tree does not need scaling, so the usual pipeline is imputer then tree.
