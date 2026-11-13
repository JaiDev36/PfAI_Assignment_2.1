"""numcompute_stream: StreamingMetrics, CSV loading and chunking, the pipeline and the plots."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numcompute import metrics as nc_metrics
from numcompute.pipeline import Pipeline
from numcompute_stream import (DecisionTree, NotFittedError, StreamingImputer, StreamingMetrics,
                               StreamingMinMaxScaler, StreamingPipeline, StreamingStandardScaler,
                               iter_chunks, load_labelled_csv, prequential_run)
from numcompute_stream import visualise as vis
from numcompute_stream.metrics import per_class_scores


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def test_streaming_metrics_match_numcompute_over_the_whole_stream(rng):
    y_true = rng.integers(0, 4, 600)
    y_pred = np.where(rng.random(600) < 0.7, y_true, rng.integers(0, 4, 600))
    stream = StreamingMetrics()
    for start in range(0, 600, 100):
        stream.update(y_true[start:start + 100], y_pred[start:start + 100])
    assert stream.accuracy == pytest.approx(nc_metrics.accuracy(y_true, y_pred))
    assert stream.macro_f1 == pytest.approx(nc_metrics.f1(y_true, y_pred, average="macro"))
    labels = stream.labels_
    np.testing.assert_array_equal(stream.confusion_,
                                  nc_metrics.confusion_matrix(y_true, y_pred, labels=labels))


def test_streaming_metrics_grow_when_a_new_class_appears():
    stream = StreamingMetrics()
    stream.update([0, 1, 1], [0, 1, 0])
    stream.update([2, 2], [2, 1], retrain_time=0.1)        # class 2 appears
    assert stream.confusion_.shape == (3, 3)
    assert stream.confusion_[:2, :2].tolist() == [[1, 0], [1, 1]]   # old counts did not move
    assert stream.accuracy == pytest.approx(3 / 5)
    assert stream.history["n_seen"] == [3, 5]
    assert stream.history["chunk_accuracy"] == pytest.approx([2 / 3, 1 / 2])
    assert stream.history["retrain_time"] == [0.1]
    assert stream.last_entry_["n_seen"] == 5


def test_streaming_metrics_string_labels_series_and_reset():
    stream = StreamingMetrics().update(["a", "b"], ["a", "a"])
    assert list(stream.labels_) == ["a", "b"]
    np.testing.assert_allclose(stream.series("error"), [0.5])
    stream.reset()
    assert stream.history["chunk"] == []
    with pytest.raises(NotFittedError):
        _ = stream.accuracy
    with pytest.raises(NotFittedError):
        _ = stream.macro_f1


@pytest.mark.parametrize("y_true, y_pred, message", [
    ([0, 1], [0], "same length"),
    ([], [0], "same length"),                              # empty against non-empty is an error
    ([[0, 1]], [[0, 1]], "1-dimensional"),
    ([0.0, np.nan], [0.0, 1.0], "NaN"),
])
def test_streaming_metrics_reject_bad_input(y_true, y_pred, message):
    stream = StreamingMetrics()
    with pytest.raises(ValueError, match=message):
        stream.update(y_true, y_pred)
    assert stream.history["chunk"] == []                   # a failed update recorded nothing


def test_streaming_metrics_ignore_an_empty_update():
    stream = StreamingMetrics(window=2).update([0, 1, 1], [0, 1, 0])
    before = (stream.last_entry_, len(stream.history["chunk"]), stream.confusion_.copy())
    assert stream.update([], []) is stream
    assert stream.last_entry_ is before[0]
    assert len(stream.history["chunk"]) == before[1]
    np.testing.assert_array_equal(stream.confusion_, before[2])


def test_rolling_window_keeps_its_own_copy_of_each_chunk():
    y_true, y_pred = np.array([0, 1, 1]), np.array([0, 1, 0])
    stream = StreamingMetrics(window=3).update(y_true, y_pred)
    expected = stream.result()["rolling_accuracy"]
    y_true[:] = 0                                         # the caller reuses its arrays
    y_pred[:] = 1
    assert stream.result()["rolling_accuracy"] == expected


def test_prequential_skips_an_empty_chunk(blobs):
    X, y = blobs
    chunks = list(iter_chunks(X, y, 50))
    with_empty = chunks[:2] + [(X[:0], y[:0])] + chunks[2:]
    a = prequential_run(DecisionTree(buffer_size=60), iter(chunks), log_every=0)
    b = prequential_run(DecisionTree(buffer_size=60), iter(with_empty), log_every=0)
    np.testing.assert_array_equal(a.series("chunk_accuracy"), b.series("chunk_accuracy"))
    assert list(b.series("rows_read")) == list(a.series("rows_read"))


def test_per_class_scores_handle_zero_division():
    precision, recall, f1 = per_class_scores([[5, 0], [3, 0]])      # class 1 is never predicted
    assert precision[1] == 0 and recall[1] == 0 and f1[1] == 0
    assert np.isfinite(f1).all()
    with pytest.raises(ValueError, match="square"):
        per_class_scores(np.zeros((2, 3)))


# ---------------------------------------------------------------------
# CSV loading and chunking
# ---------------------------------------------------------------------

def test_load_labelled_csv_reads_missing_cells_as_nan(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("a,b,label\n1.5,,0\n2,3,1\n4,5,0\n")
    X, y, names = load_labelled_csv(path)
    assert names == ["a", "b"] and X.shape == (3, 2) and list(y) == [0, 1, 0]
    assert y.dtype == np.int64 and np.isnan(X[0, 1]) and X[0, 0] == 1.5


def test_load_labelled_csv_label_by_name_and_by_index(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("label,a,b\n0,1,2\n1,3,4\n")
    X_name, y_name, names = load_labelled_csv(path, label_column="label")
    X_idx, y_idx, _ = load_labelled_csv(path, label_column=0)
    np.testing.assert_array_equal(X_name, X_idx)
    np.testing.assert_array_equal(y_name, y_idx)
    assert names == ["a", "b"]


def test_load_labelled_csv_text_labels(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("x,label\n1,cat\n2,dog\n3,cat\n")
    X, y, _ = load_labelled_csv(path)
    assert list(y) == ["cat", "dog", "cat"] and X.shape == (3, 1)


def test_load_labelled_csv_without_header(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("1,2,0\n3,4,1\n")
    X, y, names = load_labelled_csv(path, header=False)
    assert names == ["x0", "x1"] and X.shape == (2, 2) and list(y) == [0, 1]


def test_load_labelled_csv_missing_label_raise_or_drop(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("x,label\n1,0\n2,\n3,1\n")
    with pytest.raises(ValueError, match=r"1 row\(s\) have no label.*data row 1"):
        load_labelled_csv(path)
    X, y, _ = load_labelled_csv(path, on_missing_label="drop")
    assert X.shape == (2, 1) and list(y) == [0, 1]


@pytest.mark.parametrize("content, kwargs, error, message", [
    ("", {}, ValueError, "is empty"),
    ("x,label\n", {}, ValueError, "no data rows"),
    ("x,label\n1,0\n", {"label_column": "nope"}, ValueError, "not found"),
    ("x,label\n1,0\n", {"label_column": 5}, ValueError, "out of range"),
    ("x,label\n1,0\n", {"label_column": 1.5}, TypeError, "column name or an integer"),
    ("x,label\n1,0\n", {"on_missing_label": "ignore"}, ValueError, "on_missing_label"),
    ("x\n1\n", {}, ValueError, "at least 2 columns"),
    ("a,b,label\ntext,1,0\n", {}, ValueError, "numeric"),
])
def test_load_labelled_csv_errors(tmp_path, content, kwargs, error, message):
    path = tmp_path / "d.csv"
    path.write_text(content)
    with pytest.raises(error, match=message):
        load_labelled_csv(path, **kwargs)


def test_load_labelled_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_labelled_csv(tmp_path / "nope.csv")


def test_iter_chunks_cover_the_data_in_order(rng):
    X = rng.normal(size=(23, 3))
    y = rng.integers(0, 2, 23)
    chunks = list(iter_chunks(X, y, 5))
    assert [len(c[1]) for c in chunks] == [5, 5, 5, 5, 3]
    np.testing.assert_array_equal(np.vstack([c[0] for c in chunks]), X)
    np.testing.assert_array_equal(np.concatenate([c[1] for c in chunks]), y)


def test_iter_chunks_validates_when_called_not_when_iterated():
    with pytest.raises(ValueError, match="different numbers of samples"):
        iter_chunks(np.zeros((4, 2)), np.zeros(3), 2)
    with pytest.raises(ValueError, match="Input data must have 2 dimensions"):
        iter_chunks(np.zeros(4), np.zeros(4), 2)
    with pytest.raises(ValueError, match="chunk_size"):
        iter_chunks(np.zeros((4, 2)), np.zeros(4), 0)
    with pytest.raises(TypeError, match="chunk_size"):
        iter_chunks(np.zeros((4, 2)), np.zeros(4), 2.5)


def test_csv_to_stream_integration(tmp_path, rng):
    """The whole path of the demo: CSV file, chunks, test-then-train, scores."""
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    path = tmp_path / "stream.csv"
    rows = ["f0,f1,f2,label"] + [",".join(map(repr, [*x.tolist(), int(t)])) for x, t in zip(X, y)]
    path.write_text("\n".join(rows) + "\n")
    X2, y2, names = load_labelled_csv(path)
    np.testing.assert_allclose(X2, X)
    scored = prequential_run(DecisionTree(max_depth=4, buffer_size=100),
                             iter_chunks(X2, y2, 50), log_every=0)
    assert names == ["f0", "f1", "f2"] and scored.accuracy > 0.8


# ---------------------------------------------------------------------
# Pipeline and prequential evaluation
# ---------------------------------------------------------------------

def make_pipeline(**tree_args):
    tree_args.setdefault("max_depth", 4)
    tree_args.setdefault("buffer_size", 100)
    return StreamingPipeline([("imputer", StreamingImputer()),
                              ("scaler", StreamingStandardScaler()),
                              ("tree", DecisionTree(**tree_args))])


def test_pipeline_imputes_scales_then_trains(blobs):
    X, y = blobs
    X = X.copy()
    X[::5, 1] = np.nan
    pipe = make_pipeline()
    for X_chunk, y_chunk in iter_chunks(X, y, 50):
        pipe.partial_fit(X_chunk, y_chunk)
    assert pipe.model.n_retrains_ == 3
    assert np.mean(pipe.predict(X) == y) > 0.9
    np.testing.assert_allclose(pipe.predict_proba(X).sum(axis=1), 1.0)


def test_pipeline_is_a_numcompute_pipeline_so_batch_fit_still_works(blobs):
    X, y = blobs
    pipe = StreamingPipeline([("scaler", StreamingMinMaxScaler()),
                              ("tree", DecisionTree(max_depth=4))])
    assert isinstance(pipe, Pipeline)
    assert pipe.fit(X, y).predict(X).shape == (300,)
    assert np.mean(pipe.predict(X) == y) > 0.97


def test_pipeline_flush_and_named_steps(blobs):
    X, y = blobs
    pipe = make_pipeline(buffer_size=1000)
    pipe.update(X[:120], y[:120])
    assert not pipe.model.is_fitted
    assert pipe.flush().model.is_fitted
    assert list(pipe.named_steps) == ["imputer", "scaler", "tree"]


def test_pipeline_rejects_bad_steps():
    class NoPartialFit:
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    with pytest.raises(TypeError, match="must implement partial_fit"):
        StreamingPipeline([("plain", NoPartialFit()), ("tree", DecisionTree())])
    with pytest.raises(TypeError):
        StreamingPipeline([("x", DecisionTree()), ("tree", DecisionTree())])   # no transform
    with pytest.raises(ValueError, match="unique"):
        StreamingPipeline([("a", StreamingImputer()), ("a", DecisionTree())])


def test_pipeline_predict_proba_needs_a_model_with_probabilities():
    class Majority:
        def partial_fit(self, X, y):
            return self

        def fit(self, X, y=None):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    with pytest.raises(TypeError, match="predict_proba"):
        StreamingPipeline([("m", Majority())]).predict_proba(np.zeros((2, 2)))


def test_prequential_scores_only_after_the_first_tree(blobs, caplog):
    X, y = blobs
    tree = DecisionTree(max_depth=4, buffer_size=100)
    with caplog.at_level(logging.INFO, logger="numcompute_stream"):
        scored = prequential_run(tree, iter_chunks(X, y, 50))
    assert scored.history["stream_chunk"] == [3, 4, 5, 6]     # chunks 1 and 2 only fill the buffer
    assert scored.history["rows_read"] == [150, 200, 250, 300]
    assert len(scored.history["retrains"]) == 4 and scored.accuracy > 0.9
    assert "buffering" in caplog.text and "first tree trained" in caplog.text
    assert "cum_acc" in caplog.text


def test_prequential_tests_each_chunk_before_training_on_it(rng):
    """A chunk the model has never seen is scored with the tree trained BEFORE it arrived."""
    X = rng.normal(size=(400, 1))
    y_old = (X[:, 0] > 0).astype(int)
    labels = np.concatenate([y_old[:200], 1 - y_old[200:]])          # the rule flips at row 200
    tree = DecisionTree(max_depth=2, buffer_size=200)
    scored = prequential_run(tree, iter_chunks(X, labels, 200), log_every=0)
    assert scored.history["chunk_accuracy"][0] < 0.1       # old tree scored on flipped labels


def test_prequential_with_a_scaler_pipeline_survives_the_first_chunks(blobs):
    """numcompute's scalers raise a plain RuntimeError before fit; that must not crash a run."""
    X, y = blobs
    pipe = make_pipeline()
    scored = prequential_run(pipe, iter_chunks(X, y, 50), log_every=0)
    assert len(scored.history["chunk"]) == 4 and scored.accuracy > 0.9


def test_prequential_reraises_a_runtime_error_once_a_tree_exists(blobs):
    X, y = blobs

    class Breaks:
        calls = 0

        def partial_fit(self, X, y=None):
            return self

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            Breaks.calls += 1
            if Breaks.calls > 3:
                raise RuntimeError("something broke")
            return X

    pipe = StreamingPipeline([("breaks", Breaks()), ("tree", DecisionTree(buffer_size=50))])
    with pytest.raises(RuntimeError, match="something broke"):
        prequential_run(pipe, iter_chunks(X, y, 50), log_every=0)


def test_prequential_keeps_adding_to_a_given_metrics_object(blobs):
    X, y = blobs
    shared = StreamingMetrics()
    prequential_run(DecisionTree(buffer_size=60), iter_chunks(X[:150], y[:150], 30),
                    metrics=shared, log_every=0)
    count = len(shared.history["chunk"])
    prequential_run(DecisionTree(buffer_size=60), iter_chunks(X[150:], y[150:], 30),
                    metrics=shared, log_every=0)
    assert len(shared.history["chunk"]) > count


# ---------------------------------------------------------------------
# Third-party steps (2.1 feedback: a fit(X)/transform(X) object should need no adapter)
# ---------------------------------------------------------------------


class ExternalCentre:
    """A third-party streaming transformer: no base class, no fit, no y anywhere."""

    def __init__(self):
        self.total, self.count = 0.0, 0

    def partial_fit(self, X):
        self.total = self.total + np.nansum(X, axis=0)
        self.count += X.shape[0]
        return self

    def transform(self, X):
        return np.nan_to_num(X - self.total / self.count)


class ExternalWithReset(ExternalCentre):
    """Adds reset(), so a batch fit can start it from nothing."""

    def reset(self):
        self.__init__()
        return self


class ExternalFitAndStream:
    """Independent fit(X) (no y) and partial_fit(X): nothing is inherited, calls are recorded."""

    def __init__(self):
        self.calls, self.total, self.count = [], 0.0, 0

    def fit(self, X):
        self.calls.append("fit")
        self.total, self.count = np.nansum(X, axis=0), X.shape[0]      # replaces earlier state
        return self

    def partial_fit(self, X):
        self.calls.append("partial_fit")
        self.total = self.total + np.nansum(X, axis=0)
        self.count += X.shape[0]
        return self

    def transform(self, X):
        return np.nan_to_num(X - self.total / self.count)


class ExternalFitOnly:
    """The shape named in the 2.1 feedback: fit(X) and transform(X), and nothing to stream with."""

    def fit(self, X):
        self.mean_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        return X - self.mean_


def test_pipeline_streams_through_a_third_party_transformer_without_y(blobs):
    X, y = blobs
    pipe = StreamingPipeline([("centre", ExternalCentre()),
                              ("tree", DecisionTree(max_depth=4, buffer_size=100))])
    by_hand_centre = ExternalCentre()
    by_hand_tree = DecisionTree(max_depth=4, buffer_size=100)
    for X_chunk, y_chunk in iter_chunks(X, y, 50):
        pipe.partial_fit(X_chunk, y_chunk)
        by_hand_tree.partial_fit(by_hand_centre.partial_fit(X_chunk).transform(X_chunk), y_chunk)
    expected = by_hand_tree.predict(by_hand_centre.transform(X))
    np.testing.assert_array_equal(pipe.predict(X), expected)
    assert np.mean(pipe.predict(X) == y) > 0.9


def test_pipeline_batch_fit_uses_a_steps_own_fit_and_forgets_earlier_data(blobs):
    X, y = blobs
    step = ExternalFitAndStream()
    pipe = StreamingPipeline([("centre", step), ("tree", DecisionTree(max_depth=4))])
    pipe.partial_fit(X, y)
    assert pipe.fit(X[:100], y[:100]) is pipe
    assert step.calls == ["partial_fit", "fit"]            # fit, not partial_fit, for the batch
    assert step.count == 100                               # and the earlier chunk is forgotten
    assert np.mean(pipe.predict(X[:100]) == y[:100]) > 0.95


def test_pipeline_batch_fit_resets_then_streams_a_step_with_no_fit(blobs):
    X, y = blobs
    step = ExternalWithReset()
    pipe = StreamingPipeline([("centre", step), ("tree", DecisionTree(max_depth=4))])
    pipe.partial_fit(X, y)
    pipe.fit(X[:100], y[:100])
    assert step.count == 100


def test_pipeline_batch_fit_rejects_a_step_with_neither_fit_nor_reset(blobs):
    X, y = blobs
    step = ExternalCentre()
    pipe = StreamingPipeline([("centre", step), ("tree", DecisionTree())])
    pipe.partial_fit(X, y)
    with pytest.raises(TypeError, match="neither fit nor reset"):
        pipe.fit(X, y)
    assert step.count == X.shape[0]                        # nothing was silently added to


def test_pipeline_rejects_a_fit_only_transformer_and_says_why():
    with pytest.raises(TypeError, match="partial_fit.*one chunk at a time"):
        StreamingPipeline([("fit_only", ExternalFitOnly()), ("tree", DecisionTree())])


def test_pipeline_fit_transform_with_third_party_transformers(rng):
    X = rng.normal(5.0, 1.0, (40, 3))
    pipe = StreamingPipeline([("a", ExternalFitAndStream()), ("b", ExternalWithReset())])
    out = pipe.fit_transform(X, rng.integers(0, 2, 40))
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-12)
    with pytest.raises(TypeError, match="must implement transform"):
        StreamingPipeline([("a", ExternalWithReset()),
                           ("tree", DecisionTree())]).fit_transform(X, np.zeros(40))


def test_pipeline_reset_clears_every_step(blobs):
    X, y = blobs
    pipe = StreamingPipeline([("imp", StreamingImputer()), ("tree", DecisionTree(buffer_size=60))])
    for chunk, labels in iter_chunks(X, y, 50):
        pipe.partial_fit(chunk, labels)
    assert pipe.model.is_fitted
    assert pipe.reset() is pipe
    assert not pipe.model.is_fitted and pipe.model.n_buffered_ == 0
    first = StreamingPipeline([("imp", StreamingImputer()), ("tree", DecisionTree(buffer_size=60))])
    for chunk, labels in iter_chunks(X, y, 50):
        pipe.partial_fit(chunk, labels)
        first.partial_fit(chunk, labels)
    np.testing.assert_array_equal(pipe.predict(X), first.predict(X))     # as good as new


def test_pipeline_reset_is_all_or_nothing(blobs):
    X, y = blobs
    kept = ExternalWithReset()
    pipe = StreamingPipeline([("a", kept), ("b", ExternalCentre()), ("tree", DecisionTree())])
    pipe.partial_fit(X, y)
    with pytest.raises(TypeError, match="'b' has no reset"):
        pipe.reset()
    assert kept.count == X.shape[0]                        # the first step was not reset either


@pytest.mark.parametrize("signature", ["positional", "args", "keyword_only", "kwargs"])
def test_steps_that_can_take_y_still_receive_it(blobs, signature):
    X, y = blobs
    seen = []

    class Recorder:
        def transform(self, X):
            return X

    def positional(self, X, y=None):
        seen.append(y)
        return self

    def args(self, X, *extra):
        seen.append(extra[0])
        return self

    def keyword_only(self, X, *, y=None):
        seen.append(y)
        return self

    def kwargs(self, X, **extra):
        seen.append(extra["y"])
        return self

    Recorder.partial_fit = {"positional": positional, "args": args,
                            "keyword_only": keyword_only, "kwargs": kwargs}[signature]
    pipe = StreamingPipeline([("rec", Recorder()), ("tree", DecisionTree(buffer_size=50))])
    pipe.partial_fit(X[:50], y[:50])
    np.testing.assert_array_equal(seen[0], y[:50])


def test_a_third_party_step_without_transform_is_rejected_by_name():
    class NoTransform:
        def partial_fit(self, X):
            return self

    with pytest.raises(TypeError, match="'odd' must implement transform"):
        StreamingPipeline([("odd", NoTransform()), ("tree", DecisionTree())])


def test_a_third_party_step_with_an_unreadable_signature_gets_y(blobs):
    X, y = blobs
    seen = []

    class Opaque:
        __signature__ = "unreadable"          # inspect.signature raises ValueError on this

        def __call__(self, *args):
            seen.append(len(args))

    class Step:
        def __init__(self):
            self.partial_fit = Opaque()

        def transform(self, X):
            return X

    pipe = StreamingPipeline([("s", Step()), ("tree", DecisionTree(buffer_size=50))])
    pipe.partial_fit(X[:50], y[:50])
    assert seen == [2]                        # same as numcompute's Pipeline: X and y


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

@pytest.fixture
def scored_run(blobs):
    X, y = blobs
    tree = DecisionTree(max_depth=3, buffer_size=60)
    return tree, prequential_run(tree, iter_chunks(X, y, 30), log_every=0)


def test_every_plot_returns_a_figure_and_can_save(tmp_path, scored_run):
    tree, scored = scored_run
    for name, make in {
        "history": lambda p: vis.plot_metric_history(scored, save_path=p),
        "confusion": lambda p: vis.plot_confusion_matrix(scored.confusion_, scored.labels_,
                                                         normalise=True, save_path=p),
        "compare": lambda p: vis.plot_model_comparison({"a": scored, "b": scored.history},
                                                       save_path=p),
        "bench": lambda p: vis.plot_benchmark([10, 20], {"loop": [1.0, 2.0], "vec": [0.1, 0.2]},
                                              logx=True, save_path=p),
        "buffer": lambda p: vis.plot_buffer_tradeoff([50, 100], [0.9, 0.92], [0.01, 0.02],
                                                     n_retrains=[6, 3], save_path=p),
        "retrain": lambda p: vis.plot_retrain_log(tree.retrain_log_, save_path=p),
    }.items():
        path = tmp_path / (name + ".png")
        fig, _ = make(path)
        assert path.stat().st_size > 0 and fig is not None
    plt.close("all")


def test_plots_draw_into_a_given_axes(scored_run):
    _, scored = scored_run
    fig, axes = plt.subplots(1, 2)
    _, ax = vis.plot_metric_history(scored.history, ax=axes[0])
    assert ax is axes[0] and len(ax.lines) >= 2
    plt.close(fig)


@pytest.mark.parametrize("call, error, message", [
    (lambda: vis.plot_metric_history({"n_seen": [1]}, keys=["nope"]), KeyError, "Available"),
    (lambda: vis.plot_metric_history({"n_seen": [], "chunk_accuracy": [],
                                      "cumulative_accuracy": []}), ValueError, "empty"),
    (lambda: vis.plot_confusion_matrix(np.zeros((2, 3))), ValueError, "square"),
    (lambda: vis.plot_confusion_matrix(np.zeros((2, 2)), labels=["a"]), ValueError, "labels needs"),
    (lambda: vis.plot_retrain_log([]), ValueError, "not been retrained"),
    (lambda: vis.plot_model_comparison({}), ValueError, "empty"),
    (lambda: vis.plot_model_comparison({"a": {"n_seen": [1]}}), KeyError, "has no"),
    (lambda: vis.plot_benchmark([1, 2], {"a": [1.0]}), ValueError, "timings for 2 sizes"),
    (lambda: vis.plot_benchmark([1], {}), ValueError, "empty"),
    (lambda: vis.plot_buffer_tradeoff([1, 2], [0.5], [0.1, 0.2]), ValueError, "same length"),
])
def test_plot_errors_are_informative(call, error, message):
    with pytest.raises(error, match=message):
        call()
    plt.close("all")


# ---------------------------------------------------------------------
# Second review: labels go only to a parameter named y; an empty step list is an error
# ---------------------------------------------------------------------

def test_a_second_parameter_not_named_y_never_receives_the_labels(blobs):
    X, y = blobs

    class CopyFlag:
        def partial_fit(self, X, copy=True):
            self.copy = copy
            return self

        def transform(self, X):
            return X

    step = CopyFlag()
    StreamingPipeline([("c", step), ("tree", DecisionTree(buffer_size=50))]).partial_fit(X[:50],
                                                                                         y[:50])
    assert step.copy is True


def test_a_positional_only_y_is_passed_by_position(blobs):
    X, y = blobs
    seen = []

    class PositionalOnly:
        def partial_fit(self, X, y, /):
            seen.append(y)
            return self

        def transform(self, X):
            return X

    StreamingPipeline([("p", PositionalOnly()),
                       ("tree", DecisionTree(buffer_size=50))]).partial_fit(X[:50], y[:50])
    np.testing.assert_array_equal(seen[0], y[:50])


def test_an_empty_pipeline_is_rejected_with_a_clear_message():
    with pytest.raises(ValueError, match="at least one step"):
        StreamingPipeline([])
