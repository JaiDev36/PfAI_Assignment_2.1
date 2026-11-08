"""numcompute_stream.tree: impurity, split search, batch training, prediction and bad input."""

import numpy as np
import pytest

from numcompute.pipeline import Estimator, Pipeline
from numcompute_stream import DecisionTree, NotFittedError
from numcompute_stream.tree import (best_split_loop, best_split_vectorised, entropy_from_counts,
                                    gini_from_counts)


# ---------------------------------------------------------------------
# Impurity
# ---------------------------------------------------------------------

def test_gini_known_values():
    assert gini_from_counts([6, 2]) == pytest.approx(0.375)        # 1 - (.75^2 + .25^2)
    assert gini_from_counts([5, 5]) == pytest.approx(0.5)
    assert gini_from_counts([7, 0]) == pytest.approx(0.0)


def test_entropy_known_values_and_zero_log():
    assert entropy_from_counts([5, 5]) == pytest.approx(1.0)
    assert entropy_from_counts([4, 0]) == pytest.approx(0.0)       # 0 * log2(0) is taken as 0
    assert entropy_from_counts([1, 1, 1, 1]) == pytest.approx(2.0)


def test_impurity_of_an_empty_node_is_zero_not_nan():
    assert gini_from_counts([0, 0]) == 0.0
    assert entropy_from_counts([0, 0]) == 0.0


def test_impurity_works_on_many_nodes_at_once():
    counts = np.array([[6, 2], [5, 5], [3, 0]])
    np.testing.assert_allclose(gini_from_counts(counts), [0.375, 0.5, 0.0])


# ---------------------------------------------------------------------
# Split search
# ---------------------------------------------------------------------

def test_split_finds_the_obvious_threshold():
    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    feature, threshold, gain = best_split_vectorised(X, y, 2)
    assert feature == 0 and threshold == pytest.approx(6.5) and gain == pytest.approx(0.5)


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_vectorised_split_matches_the_loop(criterion, seed):
    rng = np.random.default_rng(seed)
    X = np.round(rng.normal(size=(80, 4)), 1)          # rounding creates tied values
    y = rng.integers(0, 3, size=80)
    a = best_split_vectorised(X, y, 3, criterion, min_samples_leaf=3)
    b = best_split_loop(X, y, 3, criterion, min_samples_leaf=3)
    assert a[0] == b[0]
    assert a[1] == pytest.approx(b[1])
    assert a[2] == pytest.approx(b[2])


def test_split_tie_goes_to_the_lowest_feature():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])   # two identical columns
    y = np.array([0, 0, 1, 1])
    assert best_split_vectorised(X, y, 2)[0] == 0
    assert best_split_loop(X, y, 2)[0] == 0


def test_no_split_on_constant_features():
    X = np.ones((10, 3))
    y = np.array([0, 1] * 5)
    assert best_split_vectorised(X, y, 2) is None
    assert best_split_loop(X, y, 2) is None


def test_no_split_for_a_single_row():
    assert best_split_vectorised(np.array([[1.0]]), np.array([0]), 1) is None


def test_min_samples_leaf_is_respected_by_the_search():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = np.array([1] + [0] * 9)                        # the best split alone isolates one row
    _, threshold, _ = best_split_vectorised(X, y, 2, min_samples_leaf=3)
    assert 2.0 <= threshold <= 7.0


def test_threshold_between_adjacent_floats_still_separates_the_rows():
    low = 1.0
    high = np.nextafter(low, 2.0)                      # their midpoint rounds to one of them
    X = np.array([[low], [low], [high], [high]])
    y = np.array([0, 0, 1, 1])
    _, threshold, _ = best_split_vectorised(X, y, 2)
    assert np.all((X[:, 0] <= threshold) == (y == 0))


def test_block_of_features_gives_the_same_answer(monkeypatch, rng):
    """Force one feature per block: the memory-capping path must not change the result."""
    X = rng.normal(size=(60, 5))
    y = rng.integers(0, 3, size=60)
    whole = best_split_vectorised(X, y, 3)
    monkeypatch.setattr("numcompute_stream.tree._MAX_BLOCK_ELEMENTS", 1)
    assert best_split_vectorised(X, y, 3) == whole


# ---------------------------------------------------------------------
# Batch training and prediction
# ---------------------------------------------------------------------

def test_fit_separable_blobs(blobs):
    X, y = blobs
    assert DecisionTree(max_depth=4).fit(X, y).score(X, y) > 0.97


def test_unlimited_depth_memorises_the_training_data(xor_data):
    X, y = xor_data
    assert DecisionTree(max_depth=None).fit(X, y).score(X, y) == 1.0


def test_max_depth_is_respected(xor_data):
    X, y = xor_data
    for depth in (0, 1, 2, 3):
        assert DecisionTree(max_depth=depth).fit(X, y).get_depth() <= depth


def test_depth_zero_predicts_the_majority_class():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.array([1, 1, 1, 1, 0, 0])
    assert np.all(DecisionTree(max_depth=0).fit(X, y).predict(X) == 1)


def test_min_samples_leaf_is_respected_by_the_tree(blobs):
    X, y = blobs
    tree = DecisionTree(max_depth=None, min_samples_leaf=10).fit(X, y)

    def leaf_sizes(node):
        return [node.n_samples] if node.is_leaf else leaf_sizes(node.left) + leaf_sizes(node.right)

    assert min(leaf_sizes(tree.root_)) >= 10


def test_min_impurity_decrease_prunes_weak_splits(rng):
    X = rng.normal(size=(200, 2))
    y = (rng.random(200) < 0.5).astype(int)            # pure noise
    assert DecisionTree(max_depth=None, min_impurity_decrease=0.2).fit(X, y).get_n_leaves() == 1


def test_predict_proba_rows_sum_to_one(blobs):
    X, y = blobs
    proba = DecisionTree(max_depth=2).fit(X, y).predict_proba(X)
    assert proba.shape == (300, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_string_labels_round_trip():
    X = np.array([[0.0], [0.1], [5.0], [5.1]])
    y = np.array(["cat", "cat", "dog", "dog"])
    assert list(DecisionTree().fit(X, y).predict([[0.05], [5.05]])) == ["cat", "dog"]


def test_entropy_and_gini_both_solve_the_blobs(blobs):
    X, y = blobs
    for criterion in ("gini", "entropy"):
        assert DecisionTree(max_depth=4, criterion=criterion).fit(X, y).score(X, y) > 0.97


def test_vectorised_and_loop_trees_are_identical(blobs):
    X, y = blobs
    a = DecisionTree(max_depth=5, splitter="vectorised").fit(X, y)
    b = DecisionTree(max_depth=5, splitter="loop").fit(X, y)
    assert a.export_text() == b.export_text()
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_grouped_prediction_equals_row_by_row_prediction(blobs):
    X, y = blobs
    tree = DecisionTree(max_depth=6).fit(X, y)
    np.testing.assert_array_equal(tree.predict(X), tree.predict_loop(X))


def test_feature_importances_sum_to_one_and_find_the_signal(rng):
    X = rng.normal(size=(300, 3))
    y = (X[:, 1] > 0).astype(int)                      # only feature 1 matters
    importances = DecisionTree(max_depth=3).fit(X, y).feature_importances_
    assert importances.sum() == pytest.approx(1.0)
    assert np.argmax(importances) == 1


def test_export_text_uses_feature_names(blobs):
    X, y = blobs
    text = DecisionTree(max_depth=2).fit(X, y).export_text(["height", "width"])
    assert ("height" in text or "width" in text) and "class:" in text
    with pytest.raises(ValueError, match="feature_names needs 2 names"):
        DecisionTree(max_depth=2).fit(X, y).export_text(["only_one"])


def test_fit_again_forgets_the_old_model(blobs):
    X, y = blobs
    tree = DecisionTree(max_depth=2).fit(X, y)
    tree.fit(X[:60], np.zeros(60, dtype=int))
    assert tree.get_n_leaves() == 1 and tree.n_retrains_ == 1


def test_is_a_numcompute_estimator_and_works_in_its_pipeline(blobs):
    X, y = blobs
    assert isinstance(DecisionTree(), Estimator)
    pipe = Pipeline([("tree", DecisionTree(max_depth=4))]).fit(X, y)
    assert np.mean(pipe.predict(X) == y) > 0.97


# ---------------------------------------------------------------------
# NaN and invalid input
# ---------------------------------------------------------------------

def test_nan_in_training_and_prediction(blobs):
    X, y = blobs
    X = X.copy()
    X[::7, 0] = np.nan
    tree = DecisionTree(max_depth=4).fit(X, y)
    assert not np.isnan(tree.fill_values_).any()
    assert tree.score(X, y) > 0.85
    assert tree.predict([[np.nan, np.nan]]).shape == (1,)


def test_a_feature_that_is_all_nan_is_filled_with_zero():
    X = np.column_stack([np.full(6, np.nan), np.arange(6.0)])
    y = np.array([0, 0, 0, 1, 1, 1])
    tree = DecisionTree().fit(X, y)
    assert tree.fill_values_[0] == 0.0 and tree.score(X, y) == 1.0


def test_predict_before_training_explains_the_buffer():
    with pytest.raises(NotFittedError, match="fit must be called before predict.*0/500"):
        DecisionTree().predict([[1.0, 2.0]])
    with pytest.raises(RuntimeError):                  # NotFittedError is a RuntimeError
        DecisionTree().predict_proba([[1.0, 2.0]])
    with pytest.raises(NotFittedError):
        DecisionTree().export_text()


def test_wrong_number_of_features_raises(blobs):
    X, y = blobs
    with pytest.raises(ValueError, match="Incompatible dimensionality"):
        DecisionTree().fit(X, y).predict(np.zeros((3, 5)))


def test_1d_input_raises(blobs):
    with pytest.raises(ValueError, match="Input data must have 2 dimensions. Got shape:"):
        DecisionTree().fit(np.arange(5.0), np.zeros(5))
    X, y = blobs
    with pytest.raises(ValueError, match="Input data must have 2 dimensions. Got shape:"):
        DecisionTree().fit(X, y).predict(np.zeros(2))


@pytest.mark.parametrize("X, y, message", [
    (np.zeros((5, 2)), np.zeros(4), "different numbers of samples"),
    ([[1.0], [np.inf]], [0, 1], "infinite"),
    ([[1.0], [2.0]], [0.0, np.nan], "NaN labels"),
    (np.empty((0, 2)), np.empty(0), "empty array"),
    ([["a"], ["b"]], [0, 1], "numeric"),
])
def test_fit_rejects_bad_data(X, y, message):
    with pytest.raises(ValueError, match=message):
        DecisionTree().fit(X, y)


def test_fit_without_labels_and_none_input():
    with pytest.raises(ValueError, match="y is required"):
        DecisionTree().fit(np.zeros((3, 2)))
    with pytest.raises(TypeError):
        DecisionTree().fit(None, [0])


@pytest.mark.parametrize("kwargs", [
    {"max_depth": -1}, {"buffer_size": 0}, {"min_samples_split": 1},
    {"min_samples_leaf": 0}, {"criterion": "mse"}, {"splitter": "fast"},
    {"buffer_size": 2.5}, {"max_depth": True}, {"min_impurity_decrease": -0.1},
    {"min_impurity_decrease": float("nan")},
])
def test_invalid_hyperparameters_raise(kwargs):
    with pytest.raises((ValueError, TypeError)):
        DecisionTree(**kwargs)
