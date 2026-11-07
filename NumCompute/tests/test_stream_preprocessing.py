"""numcompute_stream: ColumnStats and the streaming transformers.

The first block is the same contract that tests/test_preprocessing.py applies to numcompute's own
transformers (same inputs, same error messages), run against the streaming classes. The rest tests
what is new: learning chunk by chunk gives the same answer as learning from all the data at once.
"""

import numpy as np
import pytest

from numcompute import preprocessing
from numcompute_stream import (ColumnStats, LabelEncoder, NotFittedError, StreamingImputer,
                               StreamingMinMaxScaler, StreamingStandardScaler, iter_chunks)

STREAM_TRANSFORMERS = [StreamingStandardScaler, StreamingMinMaxScaler, StreamingImputer]


# ---------------------------------------------------------------------
# Contract tests (must hold for every streaming transformer)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("cls", STREAM_TRANSFORMERS, ids=lambda c: c.__name__)
def test_fit_transform_equivalence(cls):
    X = np.array([[1, 10], [3, 30]])
    a = cls().fit_transform(X)
    b = cls().fit(X).transform(X)
    np.testing.assert_array_almost_equal(a, b, decimal=6)


@pytest.mark.parametrize("cls", STREAM_TRANSFORMERS, ids=lambda c: c.__name__)
def test_1d_array_raises(cls):
    with pytest.raises(ValueError, match="Input data must have 2 dimensions. Got shape:"):
        cls().fit(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="Input data must have 2 dimensions. Got shape:"):
        cls().partial_fit(np.array([1, 2, 3]))


@pytest.mark.parametrize("cls", STREAM_TRANSFORMERS, ids=lambda c: c.__name__)
def test_column_count_mismatch_in_transform_and_partial_fit(cls):
    obj = cls().fit(np.array([[1, 10], [3, 30]]))
    with pytest.raises(ValueError, match="Incompatible dimensionality"):
        obj.transform(np.array([[1, 20, 5], [1, 10, 6]]))
    with pytest.raises(ValueError, match="Incompatible dimensionality"):
        obj.partial_fit(np.array([[1, 20, 5]]))


@pytest.mark.parametrize("cls", STREAM_TRANSFORMERS, ids=lambda c: c.__name__)
def test_3d_input_to_transform_raises(cls):
    obj = cls().fit(np.array([[1, 10], [3, 30]]))
    with pytest.raises(ValueError, match="Input data must have 2 dimensions. Got shape:"):
        obj.transform(np.array([[[1, 20], [1, 10]]]))


@pytest.mark.parametrize("cls", STREAM_TRANSFORMERS, ids=lambda c: c.__name__)
def test_fit_on_empty_array_raises(cls):
    with pytest.raises(ValueError, match="Cannot fit .* on an empty array."):
        cls().fit(np.empty((0, 2)))


@pytest.mark.parametrize("cls", STREAM_TRANSFORMERS, ids=lambda c: c.__name__)
def test_transform_without_fit_raises_runtime_error(cls):
    with pytest.raises(RuntimeError, match="fit must be called before transform."):
        cls().transform(np.array([[1.0, 2.0]]))


@pytest.mark.parametrize("cls", STREAM_TRANSFORMERS, ids=lambda c: c.__name__)
def test_empty_chunk_changes_nothing(cls):
    obj = cls().partial_fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    before = obj.transform(np.array([[2.0, 3.0]]))
    obj.partial_fit(np.empty((0, 2)))
    np.testing.assert_array_equal(obj.transform(np.array([[2.0, 3.0]])), before)


@pytest.mark.parametrize("cls", STREAM_TRANSFORMERS, ids=lambda c: c.__name__)
def test_return_type_is_float_array(cls):
    out = cls().fit(np.array([[1, 10], [3, 30]])).transform(np.array([[2, 20]]))
    assert isinstance(out, np.ndarray) and out.dtype == np.float64 and out.shape == (1, 2)


@pytest.mark.parametrize("cls", STREAM_TRANSFORMERS, ids=lambda c: c.__name__)
def test_transform_and_partial_fit_do_not_mutate_input(cls):
    X = np.array([[1.0, np.nan], [3.0, 30.0]])
    copy = X.copy()
    obj = cls().partial_fit(X)
    obj.transform(X)
    np.testing.assert_array_equal(X, copy)


@pytest.mark.parametrize("cls", STREAM_TRANSFORMERS, ids=lambda c: c.__name__)
def test_y_is_ignored(cls):
    X = np.array([[1.0, 10.0], [3.0, 30.0]])
    np.testing.assert_array_equal(cls().fit(X, y=[0, 1]).transform(X), cls().fit(X).transform(X))
    np.testing.assert_array_equal(cls().partial_fit(X, [0, 1]).transform(X),
                                  cls().fit(X).transform(X))


@pytest.mark.parametrize("cls", STREAM_TRANSFORMERS, ids=lambda c: c.__name__)
def test_infinity_is_rejected(cls):
    with pytest.raises(ValueError, match="infinite"):
        cls().partial_fit(np.array([[1.0, np.inf]]))


@pytest.mark.parametrize("cls", STREAM_TRANSFORMERS, ids=lambda c: c.__name__)
def test_reset_forgets_everything(cls):
    obj = cls().partial_fit(np.array([[1.0, 2.0]])).reset()
    with pytest.raises(RuntimeError, match="fit must be called before transform."):
        obj.transform(np.array([[1.0, 2.0]]))
    obj.partial_fit(np.array([[1.0, 2.0, 3.0]]))          # a different width is fine after reset


@pytest.mark.parametrize("cls", STREAM_TRANSFORMERS, ids=lambda c: c.__name__)
def test_update_is_a_synonym_of_partial_fit(cls):
    X = np.array([[1.0, 5.0], [2.0, 7.0], [4.0, 9.0]])
    np.testing.assert_array_equal(cls().update(X).transform(X), cls().partial_fit(X).transform(X))


# ---------------------------------------------------------------------
# Scalers: streaming == batch
# ---------------------------------------------------------------------

@pytest.mark.parametrize("chunk", [1, 7, 33, 500])
def test_standard_scaler_matches_numcompute_for_any_chunk_size(rng, chunk):
    X = rng.normal(loc=5, scale=3, size=(200, 3))
    streamed = StreamingStandardScaler()
    for X_chunk, _ in iter_chunks(X, np.zeros(200), chunk):
        streamed.partial_fit(X_chunk)
    batch = preprocessing.StandardScaler().fit(X)
    np.testing.assert_allclose(streamed.mean_, batch.mean_)
    np.testing.assert_allclose(streamed.scale_, batch.scale_)
    np.testing.assert_allclose(streamed.transform(X), batch.transform(X))


def test_standard_scaler_output_is_standardised_and_invertible(rng):
    X = rng.normal(loc=5, scale=3, size=(200, 3))
    scaler = StreamingStandardScaler()
    for X_chunk, _ in iter_chunks(X, np.zeros(200), 40):
        scaler.partial_fit(X_chunk)
    Z = scaler.transform(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(Z.std(axis=0), 1, atol=1e-10)


def test_standard_scaler_constant_column_is_centred_not_divided_by_zero():
    scaler = StreamingStandardScaler().fit(np.column_stack([np.full(5, 4.0), np.arange(5.0)]))
    Z = scaler.transform([[4.0, 2.0], [9.0, 2.0]])
    assert np.isfinite(Z).all()
    np.testing.assert_allclose(Z[:, 0], [0.0, 5.0])


def test_standard_scaler_large_offset_is_numerically_stable(rng):
    X = 1e9 + rng.normal(size=(2000, 1))               # E[x^2] - E[x]^2 would lose every digit
    scaler = StreamingStandardScaler()
    for X_chunk, _ in iter_chunks(X, np.zeros(2000), 100):
        scaler.partial_fit(X_chunk)
    assert scaler.scale_[0] == pytest.approx(X.std(), rel=1e-6)


def test_standard_scaler_nan_is_ignored_when_learning_and_kept_after():
    scaler = StreamingStandardScaler().partial_fit([[1.0, np.nan], [3.0, 4.0], [np.nan, 8.0]])
    np.testing.assert_allclose(scaler.mean_, [2.0, 6.0])
    out = scaler.transform([[np.nan, 6.0]])
    assert np.isnan(out[0, 0]) and out[0, 1] == 0.0


def test_partial_fit_tolerates_a_column_that_is_all_nan_until_later():
    scaler = StreamingStandardScaler().partial_fit([[np.nan, 1.0], [np.nan, 3.0]])
    assert scaler.mean_[0] == 0.0 and scaler.scale_[0] == 1.0     # not observed yet
    scaler.partial_fit([[10.0, 2.0], [20.0, 2.0]])
    assert scaler.mean_[0] == pytest.approx(15.0)


def test_fit_rejects_a_column_that_is_all_nan():
    for cls in (StreamingStandardScaler, StreamingMinMaxScaler):
        with pytest.raises(ValueError, match="all NaN"):
            cls().fit([[np.nan, 1.0], [np.nan, 2.0]])


@pytest.mark.parametrize("chunk", [1, 9, 100])
def test_minmax_scaler_matches_numcompute(rng, chunk):
    X = rng.uniform(-5, 20, size=(100, 3))
    streamed = StreamingMinMaxScaler(feature_range=(-1, 1))
    for X_chunk, _ in iter_chunks(X, np.zeros(100), chunk):
        streamed.partial_fit(X_chunk)
    batch = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(X)
    np.testing.assert_allclose(streamed.transform(X), batch.transform(X))


def test_minmax_range_widens_as_new_extremes_arrive():
    scaler = StreamingMinMaxScaler().partial_fit([[0.0], [10.0]])
    np.testing.assert_allclose(scaler.transform([[20.0]]), [[2.0]])
    scaler.partial_fit([[20.0]])
    np.testing.assert_allclose(scaler.transform([[20.0]]), [[1.0]])


def test_minmax_clip_keeps_unseen_values_inside_the_range():
    scaler = StreamingMinMaxScaler(clip=True).fit([[0.0], [10.0]])
    np.testing.assert_allclose(scaler.transform([[20.0], [-5.0], [5.0]]), [[1.0], [0.0], [0.5]])
    assert np.isnan(scaler.transform([[np.nan]])[0, 0])


# ---------------------------------------------------------------------
# Imputer
# ---------------------------------------------------------------------

def test_imputer_fills_with_the_running_mean():
    imputer = StreamingImputer()
    imputer.partial_fit([[1.0, 10.0], [3.0, np.nan]])
    imputer.partial_fit([[5.0, 30.0]])
    np.testing.assert_allclose(imputer.transform([[np.nan, np.nan]]), [[3.0, 20.0]])


def test_imputer_constant_strategy_and_never_observed_feature():
    constant = StreamingImputer(strategy="constant", fill_value=-1).fit([[np.nan, 2.0]])
    np.testing.assert_allclose(constant.transform([[np.nan, np.nan]]), [[-1.0, -1.0]])
    mean = StreamingImputer(fill_value=7).fit([[np.nan, 2.0]])      # feature 0 never observed
    np.testing.assert_allclose(mean.transform([[np.nan, np.nan]]), [[7.0, 2.0]])


def test_imputer_leaves_observed_values_alone():
    out = StreamingImputer().fit([[1.0], [3.0]]).transform([[np.nan], [9.0]])
    np.testing.assert_allclose(out, [[2.0], [9.0]])


@pytest.mark.parametrize("kwargs, error", [({"strategy": "median"}, ValueError),
                                           ({"fill_value": np.inf}, (ValueError, TypeError)),
                                           ({"fill_value": "x"}, (ValueError, TypeError))])
def test_imputer_rejects_bad_arguments(kwargs, error):
    with pytest.raises(error):
        StreamingImputer(**kwargs)


# ---------------------------------------------------------------------
# ColumnStats
# ---------------------------------------------------------------------

@pytest.mark.parametrize("chunk", [1, 7, 50, 1000])
def test_column_stats_match_numpy_for_any_chunk_size(rng, chunk):
    X = rng.normal(loc=3, scale=2, size=(500, 4))
    stats = ColumnStats()
    for X_chunk, _ in iter_chunks(X, np.zeros(500), chunk):
        stats.update(X_chunk)
    np.testing.assert_allclose(stats.mean, X.mean(axis=0))
    np.testing.assert_allclose(stats.var, X.var(axis=0))
    np.testing.assert_allclose(stats.min, X.min(axis=0))
    np.testing.assert_allclose(stats.max, X.max(axis=0))
    assert list(stats.count_) == [500] * 4


def test_column_stats_ddof_and_summary(rng):
    X = rng.normal(size=(100, 2))
    stats = ColumnStats(ddof=1).update(X[:30]).update(X[30:])
    np.testing.assert_allclose(stats.var, X.var(axis=0, ddof=1))
    assert set(stats.summary()) == {"count", "mean", "var", "std", "min", "max"}


def test_column_stats_skip_nan_per_feature_and_report_nan_for_unseen():
    stats = ColumnStats().update([[1.0, np.nan]]).update([[3.0, np.nan], [np.nan, np.nan]])
    np.testing.assert_allclose(stats.count_, [2, 0])
    assert stats.mean[0] == 2.0 and np.isnan(stats.mean[1]) and np.isnan(stats.var[1])


def test_column_stats_zero_variance():
    stats = ColumnStats().update(np.full((10, 2), 5.0)).update(np.full((3, 2), 5.0))
    np.testing.assert_allclose(stats.var, [0.0, 0.0])


def test_column_stats_errors():
    with pytest.raises(NotFittedError):
        _ = ColumnStats().mean
    stats = ColumnStats().update(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="fitted with 3"):
        stats.update(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="ddof"):
        ColumnStats(ddof=-1)


# ---------------------------------------------------------------------
# LabelEncoder
# ---------------------------------------------------------------------

def test_label_encoder_codes_are_stable_across_chunks():
    encoder = LabelEncoder().partial_fit(["b", "a", "b"])
    assert list(encoder.transform(["a", "b"])) == [1, 0]           # order of first appearance
    encoder.partial_fit(["c", "a"])
    assert list(encoder.transform(["a", "b", "c"])) == [1, 0, 2]   # old codes did not move
    assert list(encoder.inverse_transform([2, 0])) == ["c", "b"]


def test_label_encoder_errors():
    with pytest.raises(ValueError, match="Unseen label"):
        LabelEncoder().fit([0, 1]).transform([2])
    with pytest.raises(ValueError, match="Codes must lie"):
        LabelEncoder().fit([0, 1]).inverse_transform([5])
    with pytest.raises(ValueError, match="1-dimensional"):
        LabelEncoder().partial_fit([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="NaN"):
        LabelEncoder().partial_fit([0.0, np.nan])
