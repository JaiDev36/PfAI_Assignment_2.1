"""numcompute_stream.tree: buffered retraining, i.e. when the tree retrains and on what."""

import numpy as np
import pytest

from numcompute_stream import DecisionTree, NotFittedError, iter_chunks


def test_no_retrain_until_the_buffer_is_full(blobs):
    X, y = blobs
    tree = DecisionTree(buffer_size=500)
    tree.partial_fit(X[:200], y[:200])
    tree.partial_fit(X[200:300], y[200:300])
    assert tree.n_retrains_ == 0 and tree.n_buffered_ == 300
    with pytest.raises(NotFittedError):
        tree.predict(X[:5])


def test_retrains_exactly_when_the_buffer_reaches_its_size(rng):
    X = rng.normal(size=(600, 2))
    y = (X[:, 0] > 0).astype(int)
    tree = DecisionTree(buffer_size=500)
    for i, (X_chunk, y_chunk) in enumerate(iter_chunks(X, y, 200), start=1):
        tree.partial_fit(X_chunk, y_chunk)
        assert tree.n_retrains_ == (1 if i == 3 else 0)    # 200, 400, then 600 >= 500
    assert tree.n_buffered_ == 0                           # the buffer is cleared after a retrain
    assert tree.retrain_log_[0]["n_samples"] == 600        # trained on everything buffered


def test_a_buffer_of_exactly_buffer_size_retrains(rng):
    X = rng.normal(size=(100, 2))
    tree = DecisionTree(buffer_size=100).partial_fit(X, (X[:, 0] > 0).astype(int))
    assert tree.n_retrains_ == 1 and tree.n_buffered_ == 0


def test_one_chunk_larger_than_the_buffer_triggers_a_single_retrain(blobs):
    X, y = blobs
    tree = DecisionTree(buffer_size=50).partial_fit(X, y)
    assert tree.n_retrains_ == 1 and tree.retrain_log_[0]["n_samples"] == 300


def test_buffer_size_one_retrains_on_every_chunk(blobs):
    X, y = blobs
    tree = DecisionTree(buffer_size=1)
    for X_chunk, y_chunk in iter_chunks(X, y, 30):
        tree.partial_fit(X_chunk, y_chunk)
    assert tree.n_retrains_ == 10


def test_retrain_log_has_one_entry_per_retrain(blobs):
    X, y = blobs
    tree = DecisionTree(buffer_size=100)
    for X_chunk, y_chunk in iter_chunks(X, y, 50):
        tree.partial_fit(X_chunk, y_chunk)
    assert [e["retrain"] for e in tree.retrain_log_] == [1, 2, 3]
    assert all(set(e) == {"retrain", "n_samples", "seconds", "depth", "n_leaves",
                          "train_accuracy"} for e in tree.retrain_log_)


def test_each_tree_is_trained_on_its_own_window_only(rng):
    """After a retrain the buffer is cleared: the second tree never sees the first window."""
    X = rng.normal(size=(400, 1))
    y_old = (X[:, 0] > 0).astype(int)
    tree = DecisionTree(max_depth=2, buffer_size=200)
    tree.partial_fit(X[:200], y_old[:200])
    tree.partial_fit(X[200:], 1 - y_old[200:])             # the rule flips
    assert tree.n_retrains_ == 2
    assert tree.score(X, 1 - y_old) > 0.95                 # fully on the new rule


def test_predict_uses_the_latest_tree_between_retrains(rng):
    X = rng.normal(size=(200, 1))
    y = (X[:, 0] > 0).astype(int)
    tree = DecisionTree(buffer_size=200).partial_fit(X, y)
    before = tree.predict([[1.0], [-1.0]]).copy()
    tree.partial_fit(X[:50], 1 - y[:50])                   # buffered only: 50 < 200
    np.testing.assert_array_equal(tree.predict([[1.0], [-1.0]]), before)


def test_flush_trains_on_the_leftover_buffer(blobs):
    X, y = blobs
    tree = DecisionTree(buffer_size=1000).partial_fit(X[:120], y[:120])
    assert not tree.is_fitted
    tree.flush()
    assert tree.is_fitted and tree.n_buffered_ == 0 and tree.n_retrains_ == 1
    tree.flush()                                           # nothing left: no extra retrain
    assert tree.n_retrains_ == 1


def test_empty_chunk_is_a_no_op(blobs):
    X, y = blobs
    tree = DecisionTree(buffer_size=10)
    tree.partial_fit(np.empty((0, 2)), np.array([]))
    assert tree.n_buffered_ == 0 and tree.n_features_in_ is None
    tree.partial_fit(X[:20], y[:20])
    assert tree.n_retrains_ == 1


def test_update_is_a_synonym_of_partial_fit(blobs):
    X, y = blobs
    assert DecisionTree(buffer_size=100).update(X, y).n_retrains_ == 1


def test_single_class_chunk_gives_constant_predictions(rng):
    X = rng.normal(size=(50, 3))
    tree = DecisionTree(buffer_size=50).partial_fit(X, np.zeros(50, dtype=int))
    assert tree.get_n_leaves() == 1 and np.all(tree.predict(X) == 0)


def test_a_new_class_in_a_later_chunk_extends_classes(rng):
    X = rng.normal(size=(300, 2))
    tree = DecisionTree(buffer_size=100)
    tree.partial_fit(X[:100], np.where(X[:100, 0] > 0, 0, 1))
    tree.partial_fit(X[100:150], np.full(50, 2))           # class 2 appears but is only buffered
    assert list(tree.classes_) == [0, 1, 2]
    proba = tree.predict_proba(X[:5])
    assert proba.shape == (5, 3) and np.all(proba[:, 2] == 0)
    tree.partial_fit(X[150:200], np.full(50, 2))           # now a retrain includes class 2
    assert 2 in tree.predict(X[100:200])


def test_changing_the_number_of_features_mid_stream_raises(blobs):
    X, y = blobs
    tree = DecisionTree().partial_fit(X[:10], y[:10])
    with pytest.raises(ValueError, match="Incompatible dimensionality"):
        tree.partial_fit(np.zeros((5, 3)), np.zeros(5))


def test_a_constant_chunk_gives_a_single_leaf_and_the_next_chunk_recovers(rng):
    tree = DecisionTree(buffer_size=40)
    tree.partial_fit(np.full((40, 2), 3.0), np.array([0, 1] * 20))   # nothing to split on
    assert tree.get_n_leaves() == 1
    X = rng.normal(size=(40, 2))
    tree.partial_fit(X, (X[:, 0] > 0).astype(int))
    assert tree.get_n_leaves() > 1


def test_nan_inside_a_chunk(rng):
    X = rng.normal(size=(100, 2))
    y = (X[:, 1] > 0).astype(int)
    X[:30, 0] = np.nan
    assert DecisionTree(buffer_size=100).partial_fit(X, y).score(X, y) > 0.95


def test_one_buffer_equals_batch_training(blobs):
    X, y = blobs
    stream = DecisionTree(max_depth=4, buffer_size=300)
    for X_chunk, y_chunk in iter_chunks(X, y, 60):
        stream.partial_fit(X_chunk, y_chunk)
    batch = DecisionTree(max_depth=4).fit(X, y)
    assert stream.export_text() == batch.export_text()


def test_reset_forgets_everything(blobs):
    X, y = blobs
    tree = DecisionTree(buffer_size=100).partial_fit(X, y).reset()
    assert not tree.is_fitted and tree.n_retrains_ == 0 and len(tree.classes_) == 0
    assert tree.n_buffered_ == 0 and tree.retrain_log_ == []
