"""
Tests for the parts of Assignment2.2-specs.md added after the first version: the names used in the
brief, max_features, StreamTrainer, windowed quantiles and histograms, update_stats, rolling and
AUC metrics, the incremental one-hot encoder and the three plots named in the brief.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest

import numcompute_stream as ns
from numcompute.metrics import auc as nc_auc
from numcompute.metrics import roc_curve as nc_roc_curve
from numcompute.stats import quantile as nc_quantile
from numcompute_stream import (ColumnStats, DecisionTree, DecisionTreeClassifier, Imputer,
                               OneHotEncoder, Pipeline, StandardScaler, StreamingAUC,
                               StreamingHistogram, StreamingMetrics, StreamingOneHotEncoder,
                               StreamingQuantiles, StreamTrainer)
from numcompute_stream import (StreamingImputer, StreamingMinMaxScaler, StreamingPipeline,
                               StreamingStandardScaler)
from numcompute_stream.stream import StreamTrainer as StreamTrainerFromModule
from numcompute_stream.visualise import (compare_models, plot_metric_over_time,
                                         plot_predictions_vs_ground_truth)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------- names used in the brief
def test_brief_names_are_the_streaming_classes():
    assert DecisionTreeClassifier is DecisionTree
    assert Pipeline is ns.StreamingPipeline
    assert StandardScaler is ns.StreamingStandardScaler
    assert Imputer is ns.StreamingImputer
    assert OneHotEncoder is StreamingOneHotEncoder
    assert StreamTrainer is StreamTrainerFromModule


def test_example_from_the_brief_runs(blobs):
    X, y = blobs
    pipe = Pipeline([("scale", StandardScaler()),
                     ("model", DecisionTreeClassifier(max_depth=5, buffer_size=100))])
    for i in range(0, 300, 50):
        pipe.partial_fit(X[i:i + 50], y[i:i + 50])
    assert pipe.model.n_retrains_ == 3
    assert np.mean(pipe.predict(X) == y) > 0.9


# ---------------------------------------------------------------- max_features
@pytest.mark.parametrize("bad", [0, -1, 0.0, 1.5, "all", True, [1], 2.5e-1 * -1])
def test_max_features_rejects_bad_values(bad):
    with pytest.raises(ValueError, match="max_features"):
        DecisionTree(max_features=bad)


@pytest.mark.parametrize("value, expected", [(None, 9), ("sqrt", 3), ("log2", 3), (4, 4),
                                             (100, 9), (0.5, 4), (0.01, 1), (1.0, 9)])
def test_max_features_number_of_candidates(value, expected):
    assert DecisionTree(max_features=value)._n_split_features(9) == expected


def test_max_features_is_repeatable_with_random_state(rng):
    X = rng.normal(size=(300, 6))
    y = (X[:, 0] + X[:, 3] > 0).astype(int)
    a = DecisionTree(max_depth=4, max_features=2, random_state=7).fit(X, y)
    b = DecisionTree(max_depth=4, max_features=2, random_state=7).fit(X, y)
    assert a.export_text() == b.export_text()
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_max_features_changes_the_tree_but_still_learns(rng):
    X = rng.normal(size=(400, 6))
    y = (X[:, 0] > 0).astype(int)
    full = DecisionTree(max_depth=10).fit(X, y)
    sub = DecisionTree(max_depth=10, max_features=3, random_state=0).fit(X, y)
    assert sub.score(X, y) > 0.9
    assert full.score(X, y) >= sub.score(X, y)
    assert sub.n_features_in_ == 6


def test_max_features_uses_only_the_drawn_features_and_maps_the_index_back(rng):
    # only feature 4 is informative: with one feature per split the root must still name a real
    # column index (0..5), never an index into the subset
    X = rng.normal(size=(200, 6))
    y = (X[:, 4] > 0).astype(int)
    for seed in range(5):
        tree = DecisionTree(max_depth=1, max_features=1, random_state=seed).fit(X, y)
        root = tree.root_
        assert root.is_leaf or 0 <= root.feature < 6


def test_max_features_all_features_equals_default(rng):
    X = rng.normal(size=(150, 4))
    y = (X[:, 1] > 0.2).astype(int)
    a = DecisionTree(max_depth=3).fit(X, y)
    b = DecisionTree(max_depth=3, max_features=4, random_state=3).fit(X, y)
    assert a.export_text() == b.export_text()


def test_random_state_must_be_a_non_negative_int():
    with pytest.raises(ValueError, match="random_state"):
        DecisionTree(random_state=-1)


def test_memory_bytes_grows_with_buffer_and_tree():
    tree = DecisionTree(max_depth=2, buffer_size=100)
    assert tree.memory_bytes() == 0
    tree.partial_fit(np.zeros((10, 3)), np.zeros(10, dtype=int))
    buffered = tree.memory_bytes()
    assert buffered >= 10 * 3 * 8
    tree.partial_fit(np.random.default_rng(0).normal(size=(90, 3)), np.arange(90) % 2)
    assert tree.n_buffered_ == 0 and tree.memory_bytes() > 0


# ---------------------------------------------------------------- StreamTrainer
def _chunks(X, y, size):
    for i in range(0, len(y), size):
        yield X[i:i + size], y[i:i + size]


def test_stream_trainer_fit_and_score_chunk(blobs):
    X, y = blobs
    trainer = StreamTrainer(DecisionTree(max_depth=4, buffer_size=100), log_every=0)
    assert trainer.score_chunk(X[:50], y[:50]) is None          # no tree yet: nothing recorded
    trainer.fit_chunk(X[:100], y[:100])
    acc = trainer.score_chunk(X[100:150], y[100:150])
    assert 0.0 <= acc <= 1.0
    assert trainer.cumulative_accuracy == pytest.approx(acc)
    assert trainer.n_chunks_ == 1 and len(trainer.log_) == 1


def test_stream_trainer_logs_memory_time_and_accuracy(blobs):
    X, y = blobs
    trainer = StreamTrainer(DecisionTree(max_depth=4, buffer_size=60), log_every=0)
    trainer.run(_chunks(X, y, 30))
    assert len(trainer.log_) == 10
    first, last = trainer.log_[0], trainer.log_[-1]
    assert np.isnan(first["cumulative_accuracy"])
    assert last["cumulative_accuracy"] == pytest.approx(trainer.cumulative_accuracy)
    assert all(e["memory_bytes"] > 0 and e["seconds"] >= 0 for e in trainer.log_)
    assert last["retrains"] == 5 and last["n_rows"] == 30


def test_stream_trainer_works_on_a_pipeline(blobs):
    X, y = blobs
    pipe = ns.StreamingPipeline([("scale", StandardScaler()),
                                 ("tree", DecisionTree(max_depth=4, buffer_size=60))])
    trainer = StreamTrainer(pipe, log_every=0).run(_chunks(X, y, 30))
    assert trainer.cumulative_accuracy > 0.8
    assert trainer.log_[-1]["memory_bytes"] > 0


def test_stream_trainer_matches_prequential_run(blobs):
    X, y = blobs
    trainer = StreamTrainer(DecisionTree(max_depth=4, buffer_size=60), log_every=0)
    trainer.run(_chunks(X, y, 30))
    reference = ns.prequential_run(DecisionTree(max_depth=4, buffer_size=60),
                                   _chunks(X, y, 30), log_every=0)
    assert trainer.cumulative_accuracy == pytest.approx(reference.accuracy)


def test_stream_trainer_writes_one_info_line_per_chunk(blobs, caplog):
    X, y = blobs
    trainer = StreamTrainer(DecisionTree(max_depth=2, buffer_size=50), log_every=1)
    with caplog.at_level(logging.INFO, logger="numcompute_stream"):
        trainer.run(_chunks(X[:100], y[:100], 25))
    assert sum("memory=" in r.message for r in caplog.records) == 4


def test_stream_trainer_reset_clears_log_and_metrics(blobs):
    X, y = blobs
    trainer = StreamTrainer(DecisionTree(buffer_size=50), log_every=0).run(_chunks(X, y, 50))
    trainer.reset()
    assert trainer.log_ == [] and trainer.n_chunks_ == 0
    with pytest.raises(ns.NotFittedError):
        trainer.cumulative_accuracy


@pytest.mark.parametrize("bad", [-1, 1.5, True, "1"])
def test_stream_trainer_rejects_bad_log_every(bad):
    with pytest.raises(ValueError, match="log_every"):
        StreamTrainer(DecisionTree(), log_every=bad)


def test_stream_trainer_rejects_object_without_partial_fit():
    with pytest.raises(TypeError, match="partial_fit"):
        StreamTrainer(object())


def test_stream_trainer_rejects_mismatched_chunk(blobs):
    X, y = blobs
    trainer = StreamTrainer(DecisionTree(), log_every=0)
    with pytest.raises(ValueError):
        trainer.fit_chunk(X[:10], y[:9])
    with pytest.raises(ValueError):
        trainer.score_chunk(X[:10], y[:9])


# ---------------------------------------------------------------- update_stats and windows
def test_update_stats_is_update_for_column_stats(rng):
    X = rng.normal(size=(50, 3))
    a = ColumnStats().update_stats(X[:20]).update_stats(X[20:])
    np.testing.assert_allclose(a.mean, X.mean(axis=0))
    np.testing.assert_allclose(a.var, X.var(axis=0))


def test_streaming_quantiles_match_batch_quantiles(rng):
    X = rng.normal(size=(200, 3))
    sq = StreamingQuantiles()
    for i in range(0, 200, 37):
        sq.update_stats(X[i:i + 37])
    np.testing.assert_allclose(sq.quantile([0.1, 0.5, 0.9]),
                               nc_quantile(X, [0.1, 0.5, 0.9], axis=0))
    np.testing.assert_allclose(sq.median(), np.median(X, axis=0))
    assert sq.n_rows_ == sq.n_seen_ == 200


def test_streaming_quantiles_window_keeps_latest_rows(rng):
    X = rng.normal(size=(100, 2))
    sq = StreamingQuantiles(window=30)
    for i in range(0, 100, 25):
        sq.update(X[i:i + 25])
    assert sq.n_rows_ == 30 and sq.n_seen_ == 100
    np.testing.assert_allclose(sq.median(), np.median(X[-30:], axis=0))


def test_streaming_quantiles_window_trims_inside_one_big_chunk():
    sq = StreamingQuantiles(window=3).update(np.arange(10.0).reshape(-1, 1))
    np.testing.assert_allclose(sq.median(), [8.0])


def test_streaming_quantiles_ignores_nan_and_empty_chunks():
    sq = StreamingQuantiles().update([[1.0], [np.nan], [3.0]]).update(np.empty((0, 1)))
    np.testing.assert_allclose(sq.median(), [2.0])


def test_streaming_quantiles_errors():
    with pytest.raises(ns.NotFittedError):
        StreamingQuantiles().median()
    with pytest.raises(ValueError, match="window"):
        StreamingQuantiles(window=0)
    sq = StreamingQuantiles().update([[1.0, 2.0]])
    with pytest.raises(ValueError):
        sq.update([[1.0]])
    with pytest.raises(ValueError):
        sq.quantile(1.5)


def test_streaming_histogram_matches_batch_counts(rng):
    X = rng.uniform(0, 1, size=(300, 2))
    h = StreamingHistogram(bins=5, value_range=(0, 1))
    for i in range(0, 300, 70):
        h.update_stats(X[i:i + 70])
    expected = np.stack([np.histogram(X[:, j], bins=5, range=(0, 1))[0] for j in range(2)])
    np.testing.assert_array_equal(h.counts_, expected)
    assert h.edges_.shape == (2, 6) and h.n_outside_ == 0


def test_streaming_histogram_range_from_first_chunk_and_outside_count():
    h = StreamingHistogram(bins=2).update([[0.0], [10.0]])
    np.testing.assert_allclose(h.edges_, [[0.0, 5.0, 10.0]])
    h.update([[5.0], [99.0], [-4.0], [np.nan]])
    np.testing.assert_array_equal(h.counts_, [[1, 2]])
    assert h.n_outside_ == 2


def test_streaming_histogram_window_forgets_old_chunks():
    h = StreamingHistogram(bins=2, value_range=(0, 2), window=2)
    h.update([[0.5]]).update([[1.5]]).update([[1.6], [1.7], [9.0]])
    np.testing.assert_array_equal(h.counts_, [[0, 3]])      # chunk 1 (0.5) was dropped
    assert h.n_outside_ == 1


def test_streaming_histogram_constant_first_chunk_gets_a_range():
    h = StreamingHistogram(bins=4).update([[3.0], [3.0]])
    assert h.edges_[0, 0] == 2.5 and h.edges_[0, -1] == 3.5
    assert h.counts_.sum() == 2


def test_streaming_histogram_errors():
    for kwargs in ({"bins": 0}, {"window": 0}, {"value_range": (1, 1)}, {"value_range": "ab"},
                   {"value_range": (0, np.inf)}):
        with pytest.raises(ValueError):
            StreamingHistogram(**kwargs)
    with pytest.raises(ValueError, match="no observed value"):
        StreamingHistogram().update([[np.nan]])
    h = StreamingHistogram(value_range=(0, 1)).update([[0.5, 0.5]])
    with pytest.raises(ValueError):
        h.update([[0.5]])


# ---------------------------------------------------------------- metrics
def test_streaming_metrics_precision_recall_f1_match_batch(rng):
    y = rng.integers(0, 3, size=200)
    pred = np.where(rng.random(200) < 0.7, y, rng.integers(0, 3, size=200))
    m = StreamingMetrics()
    for i in range(0, 200, 40):
        m.update(y[i:i + 40], pred[i:i + 40])
    batch = StreamingMetrics().update(y, pred)
    for name in ("accuracy", "precision", "recall", "f1"):
        assert m.result()[name] == pytest.approx(batch.result()[name])
    assert m.f1 == pytest.approx(m.macro_f1)
    np.testing.assert_array_equal(m.result()["confusion"], batch.confusion_)
    assert m.precision == pytest.approx(m.result()["precision"])
    assert m.recall == pytest.approx(m.result()["recall"])


def test_streaming_metrics_rolling_window():
    m = StreamingMetrics(window=2)
    m.update([0, 1], [0, 1])          # chunk accuracy 1.0
    m.update([0, 1], [1, 0])          # 0.0
    m.update([0, 1], [0, 1])          # 1.0 -> window holds chunks 2 and 3
    assert m.history["rolling_accuracy"] == [1.0, 0.5, 0.5]
    res = m.result()
    assert res["accuracy"] == pytest.approx(4 / 6)
    assert res["rolling_accuracy"] == pytest.approx(0.5)
    assert {"rolling_precision", "rolling_recall", "rolling_f1"} <= set(res)


def test_streaming_metrics_without_window_has_no_rolling_keys():
    res = StreamingMetrics().update([0, 1], [0, 1]).result()
    assert not any(k.startswith("rolling_") for k in res)


def test_streaming_metrics_window_survives_new_class():
    m = StreamingMetrics(window=2).update([0, 0], [0, 0]).update([1, 2], [1, 0])
    assert m.result()["rolling_accuracy"] == pytest.approx(0.75)


def test_streaming_metrics_reset_and_result_before_update():
    m = StreamingMetrics(window=3).update([0, 1], [0, 1])
    m.reset()
    assert m.history["chunk"] == [] and m.window == 3
    for read in (m.result, lambda: m.precision, lambda: m.recall, lambda: m.f1):
        with pytest.raises(ns.NotFittedError):
            read()


def test_streaming_metrics_rejects_bad_window():
    with pytest.raises(ValueError, match="window"):
        StreamingMetrics(window=0)


def test_streaming_auc_matches_numcompute_auc(rng):
    y = rng.integers(0, 2, size=400)
    score = np.clip(0.35 * y + rng.normal(0.4, 0.2, size=400), 0, 1)
    a = StreamingAUC(n_bins=5000)
    for i in range(0, 400, 80):
        a.update(y[i:i + 80], score[i:i + 80])
    fpr, tpr, _ = nc_roc_curve(y, score)
    assert a.result() == pytest.approx(nc_auc(fpr, tpr), abs=2e-3)
    assert a.n_pos_ + a.n_neg_ == 400


def test_streaming_auc_known_values():
    assert StreamingAUC().update([0, 1], [0.1, 0.9]).result() == 1.0
    assert StreamingAUC().update([0, 1], [0.9, 0.1]).result() == 0.0
    assert StreamingAUC().update([0, 1], [0.5, 0.5]).result() == 0.5
    assert StreamingAUC().update([0, 1], [0.2, 0.9]).update([0, 1], [0.4, 0.3]).result() == 0.75


def test_streaming_auc_accumulates_not_averages():
    # chunk 1 is perfect, chunk 2 is reversed: the combined AUC is 0.5, not (1 + 0) / 2 by chunk
    a = StreamingAUC().update([0, 1], [0.1, 0.9]).update([0, 1], [0.9, 0.1])
    assert a.result() == pytest.approx(0.5)


def test_streaming_auc_window_and_reset():
    a = StreamingAUC(window=1).update([0, 1], [0.9, 0.1]).update([0, 1], [0.1, 0.9])
    assert a.result() == 1.0
    assert a.reset().n_pos_ == 0


def test_streaming_auc_errors():
    a = StreamingAUC()
    with pytest.raises(ns.NotFittedError, match="positive and one negative"):
        a.result()
    with pytest.raises(ns.NotFittedError):
        a.update([1, 1], [0.2, 0.3]).result()
    for y, s in (([0, 1], [0.5]), ([[0, 1]], [[0.5, 0.5]]), ([0, 1], [0.5, np.nan]),
                 ([0, 1], [0.5, 1.5])):
        with pytest.raises(ValueError):
            StreamingAUC().update(y, s)
    for kwargs in ({"n_bins": 1}, {"window": 0}, {"score_range": (1, 0)}, {"score_range": 5}):
        with pytest.raises(ValueError):
            StreamingAUC(**kwargs)
    assert StreamingAUC().update([], []).n_pos_ == 0


def test_streaming_auc_with_tree_probabilities(blobs):
    X, y = blobs
    y = (y == 1).astype(int)
    tree = DecisionTree(max_depth=4, buffer_size=60)
    auc_stream = StreamingAUC()
    for Xc, yc in _chunks(X, y, 30):
        if tree.is_fitted:
            auc_stream.update(yc, tree.predict_proba(Xc)[:, 1])
        tree.partial_fit(Xc, yc)
    assert auc_stream.result() > 0.9


# ---------------------------------------------------------------- one-hot encoder
def test_one_hot_encoder_grows_with_new_categories():
    enc = StreamingOneHotEncoder().partial_fit([["a"], ["b"]])
    assert enc.transform([["a"]]).shape == (1, 2)
    enc.partial_fit([["c"], ["a"]])
    assert enc.transform([["a"]]).shape == (1, 3)
    assert enc.categories_[0].tolist() == ["a", "b", "c"]


def test_one_hot_encoder_streaming_equals_batch_for_any_chunk_size(rng):
    X = rng.integers(0, 4, size=(60, 3))
    batch = StreamingOneHotEncoder().fit(X)
    for size in (1, 7, 60):
        enc = StreamingOneHotEncoder()
        for i in range(0, 60, size):
            enc.partial_fit(X[i:i + size])
        np.testing.assert_array_equal(enc.transform(X), batch.transform(X))


def test_one_hot_encoder_numeric_nan_and_empty_chunk():
    enc = StreamingOneHotEncoder().partial_fit([[1.0], [np.nan]]).partial_fit(np.empty((0, 1)))
    enc.partial_fit([[0.0]])
    assert enc.transform([[np.nan]]).tolist() == [[0.0, 0.0, 1.0]]


def test_one_hot_encoder_fit_forgets_and_unseen_category_is_an_error():
    enc = StreamingOneHotEncoder().partial_fit([["a"]])
    enc.fit([["x"], ["y"]])
    assert enc.categories_[0].tolist() == ["x", "y"]
    with pytest.raises(ValueError):
        enc.transform([["a"]])


def test_one_hot_encoder_errors():
    enc = StreamingOneHotEncoder()
    with pytest.raises(RuntimeError):
        enc.transform([["a"]])
    with pytest.raises(ValueError, match="2 dimensions"):
        enc.partial_fit(["a", "b"])
    enc.partial_fit([["a", "b"]])
    with pytest.raises(ValueError, match="features"):
        enc.partial_fit([["a"]])
    with pytest.raises(ValueError):
        StreamingOneHotEncoder().fit(np.empty((0, 2)))
    assert enc.reset() is enc and not hasattr(enc, "categories_")


# ---------------------------------------------------------------- plots named in the brief
def test_plot_metric_over_time_draws_and_saves(tmp_path):
    out = tmp_path / "acc.png"
    fig, ax = plot_metric_over_time([0.5, 0.6, 0.8], title="Accuracy", ylabel="acc",
                                    save_path=str(out))
    assert out.stat().st_size > 0
    assert ax.get_title() == "Accuracy" and ax.get_ylabel() == "acc"
    assert len(ax.lines[0].get_xdata()) == 3


def test_plot_metric_over_time_accepts_metrics_series():
    m = StreamingMetrics().update([0, 1], [0, 1]).update([0, 1], [0, 0])
    _, ax = plot_metric_over_time(m.series("chunk_accuracy"))
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [1.0, 0.5])


def test_plot_metric_over_time_draws_into_a_given_axis():
    fig, ax = plt.subplots()
    fig2, ax2 = plot_metric_over_time([1, 2], ax=ax)
    assert fig2 is fig and ax2 is ax


@pytest.mark.parametrize("bad", [[], [[1, 2]]])
def test_plot_metric_over_time_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        plot_metric_over_time(bad)


def test_plot_metric_over_time_rejects_a_dict():
    with pytest.raises(TypeError, match="dict"):
        plot_metric_over_time({"a": [1, 2]})


def test_compare_models_draws_two_labelled_lines(tmp_path):
    out = tmp_path / "cmp.png"
    _, ax = compare_models([0.5, 0.7], [0.4, 0.6, 0.9], labels=("buffer 100", "buffer 500"),
                           save_path=str(out))
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["buffer 100", "buffer 500"]
    assert len(ax.lines[1].get_xdata()) == 3 and out.stat().st_size > 0


def test_compare_models_errors():
    with pytest.raises(ValueError, match="2 names"):
        compare_models([1], [2], labels=("only one",))
    with pytest.raises(ValueError, match="empty"):
        compare_models([], [1])
    with pytest.raises(TypeError):
        compare_models([1], {"x": 1})


def test_plot_predictions_vs_ground_truth_marks_errors(tmp_path):
    out = tmp_path / "pred.png"
    _, ax = plot_predictions_vs_ground_truth([0, 1, 2, 1], [0, 1, 0, 1], save_path=str(out))
    assert "accuracy 0.750" in ax.get_title()
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["true", "predicted", "wrong"]
    assert out.stat().st_size > 0


def test_plot_predictions_vs_ground_truth_perfect_has_no_wrong_marker():
    _, ax = plot_predictions_vs_ground_truth(["a", "b"], ["a", "b"])
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["true", "predicted"]


def test_plot_predictions_vs_ground_truth_errors():
    for y_true, y_pred in (([0, 1], [0]), ([], []), ([[0, 1]], [[0, 1]])):
        with pytest.raises(ValueError):
            plot_predictions_vs_ground_truth(y_true, y_pred)


def test_show_option_calls_plt_show(monkeypatch):
    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
    plot_metric_over_time([1, 2], show=True)
    compare_models([1, 2], [2, 3], show=True)
    plot_predictions_vs_ground_truth([0, 1], [0, 1], show=True)
    plot_metric_over_time([1, 2])
    assert len(calls) == 3


# ---------------------------------------------------------------------
# Review fixes: copies, and an empty chunk changes nothing in any component
# ---------------------------------------------------------------------

def test_quantiles_hold_their_own_copy_of_a_chunk():
    chunk = np.array([[1.0], [2.0], [3.0]])
    sq = StreamingQuantiles().update(chunk)
    chunk[:] = 100.0                                  # the caller reuses its array
    assert sq.median()[0] == 2.0


def test_trimming_the_window_releases_the_trimmed_rows_memory():
    sq = StreamingQuantiles(window=10)
    sq.update(np.arange(1000.0).reshape(-1, 1))
    assert sq.n_rows_ == 10
    # a view would keep all 8,000 bytes of the original chunk alive through its base array
    assert all(c.base is None for c in sq._chunks)
    assert sum(c.nbytes for c in sq._chunks) == 10 * 8
    assert sq.median()[0] == 994.5


_X_A = np.arange(12.0).reshape(4, 3)
_X_B = np.arange(12.0, 24.0).reshape(4, 3) ** 1.1
_Y_A, _Y_B = np.array([0, 1, 0, 1]), np.array([1, 1, 0, 0])
_EMPTY_X, _EMPTY_Y = np.empty((0, 3)), np.empty(0, dtype=int)

_EMPTY_CASES = {
    "ColumnStats": (ColumnStats, lambda o, X, y: o.update(X), lambda o: o.summary()),
    "StreamingQuantiles": (lambda: StreamingQuantiles(window=6), lambda o, X, y: o.update(X),
                           lambda o: o.quantile([0.1, 0.5, 0.9])),
    "StreamingHistogram": (lambda: StreamingHistogram(bins=4, value_range=(0, 40)),
                           lambda o, X, y: o.update(X), lambda o: (o.counts_, o.n_outside_)),
    "StreamingMetrics": (lambda: StreamingMetrics(window=2),
                         lambda o, X, y: o.update(y, y[::-1]),
                         lambda o: (o.result(), o.history["chunk"])),
    "StreamingAUC": (StreamingAUC, lambda o, X, y: o.update(y, np.where(y == 1, 0.7, 0.3)),
                     lambda o: o.result()),
    "DecisionTree": (lambda: DecisionTree(buffer_size=8, max_depth=3),
                     lambda o, X, y: o.partial_fit(X, y),
                     lambda o: (o.n_buffered_, o.n_retrains_, o.predict(_X_B))),
    "StreamingStandardScaler": (StreamingStandardScaler, lambda o, X, y: o.partial_fit(X),
                                lambda o: o.transform(_X_B)),
    "StreamingMinMaxScaler": (StreamingMinMaxScaler, lambda o, X, y: o.partial_fit(X),
                              lambda o: o.transform(_X_B)),
    "StreamingImputer": (StreamingImputer, lambda o, X, y: o.partial_fit(X),
                         lambda o: o.transform(_X_B)),
    "StreamingPipeline": (lambda: StreamingPipeline([("imp", StreamingImputer()),
                                                     ("tree", DecisionTree(buffer_size=8))]),
                          lambda o, X, y: o.partial_fit(X, y), lambda o: o.predict(_X_B)),
}


@pytest.mark.parametrize("name", sorted(_EMPTY_CASES))
def test_an_empty_chunk_changes_nothing(name):
    make, feed, observe = _EMPTY_CASES[name]
    with_empty, without = make(), make()
    feed(with_empty, _X_A, _Y_A)
    feed(with_empty, _EMPTY_X, _EMPTY_Y)
    feed(with_empty, _X_B, _Y_B)
    feed(without, _X_A, _Y_A)
    feed(without, _X_B, _Y_B)
    a, b = observe(with_empty), observe(without)
    np.testing.assert_equal(a, b)


def test_an_empty_first_chunk_leaves_the_pipeline_untrained():
    pipe = StreamingPipeline([("imp", StreamingImputer()), ("tree", DecisionTree(buffer_size=8))])
    pipe.partial_fit(_EMPTY_X, _EMPTY_Y)
    assert not pipe.model.is_fitted and pipe.model.n_buffered_ == 0


# ---------------------------------------------------------------------
# Second review: StreamTrainer with empty chunks, fixed one-hot categories, AUC labels and
# columns, histogram and AUC ranges at the float limit
# ---------------------------------------------------------------------

def test_stream_trainer_scores_an_empty_chunk_as_none_before_any_score(blobs):
    X, y = blobs
    trainer = StreamTrainer(DecisionTree(buffer_size=50), log_every=0).fit_chunk(X[:50], y[:50])
    assert trainer.score_chunk(X[:0], y[:0]) is None
    assert trainer.metrics.history["chunk"] == []


def test_stream_trainer_does_not_repeat_the_last_accuracy_for_an_empty_chunk(blobs):
    X, y = blobs
    trainer = StreamTrainer(DecisionTree(buffer_size=50), log_every=0).fit_chunk(X[:50], y[:50])
    trainer.score_chunk(X[50:100], y[50:100])
    assert trainer.score_chunk(X[:0], y[:0]) is None
    assert len(trainer.metrics.history["chunk"]) == 1


def test_stream_trainer_run_skips_empty_chunks(blobs):
    X, y = blobs
    chunks = [(X[:50], y[:50]), (X[:0], y[:0]), (X[50:100], y[50:100])]
    with_empty = StreamTrainer(DecisionTree(buffer_size=50), log_every=0).run(chunks)
    without = StreamTrainer(DecisionTree(buffer_size=50), log_every=0).run(chunks[::2])
    assert len(with_empty.log_) == len(without.log_) == 2
    assert with_empty.cumulative_accuracy == without.cumulative_accuracy


def test_one_hot_with_fixed_categories_keeps_its_width_and_rejects_others():
    enc = StreamingOneHotEncoder(categories=[[0, 1, 2]])
    assert enc.partial_fit([[1.0], [2.0]]).transform([[0.0]]).tolist() == [[1.0, 0.0, 0.0]]
    with pytest.raises(ValueError, match="does not know"):
        enc.partial_fit([[5.0]])
    assert len(enc.categories_[0]) == 3
    with pytest.raises(ValueError, match="does not know"):
        StreamingOneHotEncoder(categories=[["a", "b"]]).fit([["c"]])


def test_one_hot_handle_unknown_error_rejects_a_new_category_before_changing_state():
    enc = StreamingOneHotEncoder(handle_unknown="error").partial_fit([[1.0], [2.0]])
    with pytest.raises(ValueError, match="does not know"):
        enc.partial_fit([[0.0], [1.0]])
    assert enc.categories_[0].tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="handle_unknown"):
        StreamingOneHotEncoder(handle_unknown="ignore")


def test_one_hot_with_fixed_categories_works_in_front_of_a_tree():
    X = np.array([[1.0], [2.0], [1.0], [2.0]])
    pipe = StreamingPipeline([("onehot", StreamingOneHotEncoder(categories=[[0.0, 1.0, 2.0]])),
                              ("tree", DecisionTree(buffer_size=4))])
    pipe.partial_fit(X, [0, 1, 0, 1])
    pipe.partial_fit([[0.0], [1.0]], [0, 0])           # a category not seen before: no new column
    assert pipe.predict(X).tolist() == [0, 1, 0, 1]


def test_expanding_one_hot_in_front_of_a_tree_fails_at_the_tree():
    """The documented limit: without fixed categories, a new category widens the input."""
    pipe = StreamingPipeline([("onehot", StreamingOneHotEncoder()),
                              ("tree", DecisionTree(buffer_size=4))])
    pipe.partial_fit([[1.0], [2.0], [1.0], [2.0]], [0, 1, 0, 1])
    with pytest.raises(ValueError, match="feature"):
        pipe.partial_fit([[0.0], [1.0]], [0, 0])


@pytest.mark.parametrize("labels", [[1, np.nan], np.array([1, None], dtype=object)])
def test_streaming_auc_rejects_missing_labels(labels):
    with pytest.raises(ValueError, match="missing labels"):
        StreamingAUC().update(labels, [0.9, 0.1])


def test_streaming_auc_column_must_follow_classes_order():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1, 1, 0, 0])                            # label 1 is seen first
    tree = DecisionTree(buffer_size=4).partial_fit(X, y)
    assert tree.classes_.tolist() == [1, 0]
    proba = tree.predict_proba(X)
    right = proba[:, list(tree.classes_).index(1)]
    assert StreamingAUC(pos_label=1).update(y, right).result() == 1.0
    assert StreamingAUC(pos_label=1).update(y, proba[:, 1]).result() == 0.0   # the trap


def test_streaming_auc_with_a_score_range_near_the_float_limit():
    auc = StreamingAUC(score_range=(-1e308, 1e308)).update([0, 1], [-1e307, 1e307])
    assert auc.result() == 1.0


def test_histogram_of_a_constant_huge_value_gets_a_real_range():
    hist = StreamingHistogram(bins=4).update(np.full((5, 1), 1e20))
    lo, hi = hist.edges_[0, 0], hist.edges_[0, -1]
    assert lo < 1e20 < hi and hist.counts_.sum() == 5 and hist.n_outside_ == 0


def test_histogram_with_a_range_near_the_float_limit():
    hist = StreamingHistogram(bins=4, value_range=(-1e308, 1e308))
    hist.update(np.array([[-1e308], [-1.0], [1e307], [1e308]]))
    np.testing.assert_array_equal(hist.counts_[0], [1, 1, 1, 1])
    np.testing.assert_array_equal(hist.edges_[0], [-1e308, -5e307, 0.0, 5e307, 1e308])


def test_histogram_ordinary_range_is_unchanged_by_the_scaling(rng):
    x = rng.normal(size=(500, 1))
    ours = StreamingHistogram(bins=7, value_range=(-3.0, 3.0)).update(x)
    from numcompute.stats import histogram
    counts, edges = histogram(x[(x >= -3) & (x <= 3)], bins=7, value_range=(-3.0, 3.0))
    np.testing.assert_array_equal(ours.counts_[0], counts)
    np.testing.assert_array_equal(ours.edges_[0], edges)
