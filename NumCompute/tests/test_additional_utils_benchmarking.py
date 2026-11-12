"""Additional edge-case tests for batching, activations and benchmarking.

These tests extend the existing suite without changing the toolkit itself.
Run from the project root with:
    python -m pytest tests/test_additional_utils_benchmarking.py -v
"""

import sys

import numpy as np
import pytest

from numcompute import benchmarking, utils


# ---------------------------------------------------------------------
# utils.batches
# ---------------------------------------------------------------------

class TestBatchesAdditional:

    def test_uneven_last_batch_keeps_every_row_in_order(self):
        ids = np.arange(11)
        X = np.column_stack((ids, ids * 2))
        y = ids * 10

        result = list(utils.batches(X, y, batch_size=4))

        assert [len(x_batch) for x_batch, _ in result] == [4, 4, 3]
        np.testing.assert_array_equal(np.concatenate([x for x, _ in result]), X)
        np.testing.assert_array_equal(np.concatenate([y for _, y in result]), y)

    def test_drop_last_removes_only_the_incomplete_batch(self):
        X = np.arange(11)
        result = list(utils.batches(X, batch_size=4, drop_last=True))

        assert [len(batch[0]) for batch in result] == [4, 4]
        np.testing.assert_array_equal(np.concatenate([batch[0] for batch in result]),
                                      np.arange(8))

    def test_drop_last_keeps_full_batches_when_divisible(self):
        result = list(utils.batches(np.arange(12), batch_size=4, drop_last=True))

        assert len(result) == 3
        np.testing.assert_array_equal(np.concatenate([batch[0] for batch in result]),
                                      np.arange(12))

    def test_shuffle_with_same_seed_repeats_order_and_keeps_rows_aligned(self):
        ids = np.arange(30)
        X = np.column_stack((ids, ids ** 2))
        y = ids + 100

        first = list(utils.batches(X, y, batch_size=7, shuffle=True, seed=19))
        second = list(utils.batches(X, y, batch_size=7, shuffle=True, seed=19))

        first_X = np.concatenate([x for x, _ in first])
        first_y = np.concatenate([targets for _, targets in first])
        second_X = np.concatenate([x for x, _ in second])
        second_y = np.concatenate([targets for _, targets in second])

        np.testing.assert_array_equal(first_X, second_X)
        np.testing.assert_array_equal(first_y, second_y)
        np.testing.assert_array_equal(first_X[:, 1], first_X[:, 0] ** 2)
        np.testing.assert_array_equal(first_y, first_X[:, 0] + 100)
        np.testing.assert_array_equal(np.sort(first_X[:, 0]), ids)

    def test_different_seeds_change_shuffle_order(self):
        first = np.concatenate([b[0] for b in
                                utils.batches(np.arange(30), batch_size=8,
                                              shuffle=True, seed=1)])
        second = np.concatenate([b[0] for b in
                                 utils.batches(np.arange(30), batch_size=8,
                                               shuffle=True, seed=2)])

        assert not np.array_equal(first, second)
        np.testing.assert_array_equal(np.sort(first), np.arange(30))
        np.testing.assert_array_equal(np.sort(second), np.arange(30))

    def test_lists_and_three_input_arrays_remain_aligned(self):
        X = [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]
        y = [1, 2, 3, 4, 5]
        names = ["a", "b", "c", "d", "e"]

        result = list(utils.batches(X, y, names, batch_size=2))

        assert [len(batch[0]) for batch in result] == [2, 2, 1]
        for x_batch, y_batch, name_batch in result:
            assert isinstance(x_batch, np.ndarray)
            np.testing.assert_array_equal(x_batch[:, 0], y_batch)
            np.testing.assert_array_equal(name_batch, np.array(names)[y_batch - 1])

    def test_batch_larger_than_dataset_with_and_without_drop_last(self):
        kept = list(utils.batches([5, 6, 7], batch_size=10))
        dropped = list(utils.batches([5, 6, 7], batch_size=10, drop_last=True))

        assert len(kept) == 1
        np.testing.assert_array_equal(kept[0][0], [5, 6, 7])
        assert dropped == []

    def test_mismatched_lengths_raise_even_with_three_inputs(self):
        with pytest.raises(ValueError, match="same length"):
            list(utils.batches(np.zeros((5, 2)), np.zeros(5), np.zeros(4),
                               batch_size=2))

    def test_different_row_counts_raise_even_if_total_sizes_match(self):
        # Both arrays contain 12 elements, but their first dimensions differ.
        with pytest.raises(ValueError, match="same length"):
            list(utils.batches(np.zeros((6, 2)), np.zeros((4, 3)), batch_size=2))

    def test_scalar_input_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1-dimensional"):
            list(utils.batches(np.array(5), batch_size=2))


# ---------------------------------------------------------------------
# utils.tanh and utils.leaky_relu
# ---------------------------------------------------------------------

class TestActivationsAdditional:

    def test_tanh_zero_and_list_input(self):
        assert utils.tanh(0.0) == 0.0
        output = utils.tanh([0.0, 1.0, -1.0])
        assert isinstance(output, np.ndarray)
        np.testing.assert_allclose(output, np.tanh([0.0, 1.0, -1.0]))

    def test_tanh_large_positive_and_negative_values(self):
        with np.errstate(over="raise", invalid="raise"):
            output = utils.tanh([-1000.0, 1000.0])
        np.testing.assert_array_equal(output, [-1.0, 1.0])

    def test_tanh_is_odd_and_stays_between_minus_one_and_one(self):
        x = np.array([0.001, 0.5, 1.5, 10.0])
        np.testing.assert_allclose(utils.tanh(-x), -utils.tanh(x))
        assert np.all(np.abs(utils.tanh(x)) <= 1.0)

    def test_tanh_small_values_and_2d_list_shape(self):
        small = 1e-8
        assert utils.tanh(small) == pytest.approx(small, rel=1e-12)
        output = utils.tanh([[0, 1], [-1, 0]])
        assert output.shape == (2, 2)
        np.testing.assert_allclose(output, np.tanh([[0, 1], [-1, 0]]))

    def test_leaky_relu_zero_and_values_near_zero(self):
        x = [-1e-10, 0.0, 1e-10]
        np.testing.assert_allclose(utils.leaky_relu(x, alpha=0.2),
                                   [-2e-11, 0.0, 1e-10], rtol=1e-14, atol=0)

    def test_leaky_relu_handles_large_positive_and_negative_values(self):
        x = [-1e200, 1e200]
        np.testing.assert_allclose(utils.leaky_relu(x, alpha=0.1),
                                   [-1e199, 1e200])

    @pytest.mark.parametrize("alpha, expected", [
        (0.0, [0.0, 0.0, 3.0]),
        (1.0, [-4.0, 0.0, 3.0]),
        (0.25, [-1.0, 0.0, 3.0]),
    ])
    def test_leaky_relu_alpha_boundaries(self, alpha, expected):
        np.testing.assert_allclose(utils.leaky_relu([-4, 0, 3], alpha=alpha),
                                   expected)

    def test_leaky_relu_list_returns_float_array_with_same_shape(self):
        output = utils.leaky_relu([[-2, 0], [4, -6]], alpha=0.5)
        assert isinstance(output, np.ndarray)
        assert output.shape == (2, 2)
        assert output.dtype == np.float64
        np.testing.assert_array_equal(output, [[-1.0, 0.0], [4.0, -3.0]])

    def test_leaky_relu_does_not_modify_input_and_preserves_nan(self):
        x = np.array([-2.0, np.nan, 3.0])
        original = x.copy()
        result = utils.leaky_relu(x, alpha=0.5)
        np.testing.assert_array_equal(x, original)
        np.testing.assert_allclose(result, [-1.0, np.nan, 3.0], equal_nan=True)


# ---------------------------------------------------------------------
# benchmarking.environment_info
# ---------------------------------------------------------------------

class TestEnvironmentInfoAdditional:

    def test_environment_has_four_nonempty_string_values(self):
        info = benchmarking.environment_info()
        assert set(info.keys()) == {"python", "numpy", "platform", "processor"}
        assert all(isinstance(value, str) and value.strip() for value in info.values())

    def test_environment_versions_match_current_runtime(self):
        info = benchmarking.environment_info()
        assert info["python"] == sys.version.split()[0]
        assert info["numpy"] == np.__version__
        assert info["platform"] == benchmarking.platform.platform()

    def test_processor_uses_machine_as_fallback(self, monkeypatch):
        monkeypatch.setattr(benchmarking.platform, "processor", lambda: "")
        monkeypatch.setattr(benchmarking.platform, "machine", lambda: "test-machine")
        assert benchmarking.environment_info()["processor"] == "test-machine"


# ---------------------------------------------------------------------
# benchmarking.compare_loop_vs_vectorised
# ---------------------------------------------------------------------

class TestCompareLoopVsVectorisedAdditional:

    def test_speedup_is_ratio_of_recorded_times(self):
        values = np.arange(1000.0)
        result = benchmarking.compare_loop_vs_vectorised(
            lambda x: sum(v * v for v in x),
            lambda x: np.sum(x * x),
            values,
            repeats=2,
            warmup=1,
        )

        assert set(result) == {"loop", "vectorised", "speedup"}
        assert result["loop"] > 0
        assert result["vectorised"] > 0
        assert result["speedup"] == pytest.approx(
            result["loop"] / result["vectorised"])

    def test_checks_results_before_timing(self):
        calls = {"loop": 0, "vec": 0}

        def loop_func():
            calls["loop"] += 1
            return 1.0

        def vec_func():
            calls["vec"] += 1
            return 2.0

        with pytest.raises(ValueError, match="different results"):
            benchmarking.compare_loop_vs_vectorised(loop_func, vec_func,
                                                    repeats=4, warmup=2)
        assert calls == {"loop": 1, "vec": 1}

    def test_result_mismatch_raises_with_matching_shapes(self):
        with pytest.raises(ValueError, match="different results"):
            benchmarking.compare_loop_vs_vectorised(
                lambda: np.array([1.0, 2.0]),
                lambda: np.array([1.0, 2.5]))

    def test_shape_mismatch_raises_even_when_broadcastable(self):
        with pytest.raises(ValueError, match="different results"):
            benchmarking.compare_loop_vs_vectorised(
                lambda: np.array([1.0, 2.0]),
                lambda: np.array([[1.0, 2.0]]))

    def test_nan_in_same_position_is_allowed(self):
        result = benchmarking.compare_loop_vs_vectorised(
            lambda: np.array([1.0, np.nan]),
            lambda: np.array([1.0, np.nan]),
            repeats=1, warmup=0)
        assert result["speedup"] > 0

    def test_nan_in_different_positions_raises(self):
        with pytest.raises(ValueError, match="different results"):
            benchmarking.compare_loop_vs_vectorised(
                lambda: np.array([np.nan, 1.0]),
                lambda: np.array([1.0, np.nan]))

    def test_comparison_respects_tolerance(self):
        def loop_func():
            return np.array([1.0, 2.0])

        def vec_func():
            return np.array([1.0 + 1e-10, 2.0])

        # Small floating-point rounding differences should be acceptable.
        result = benchmarking.compare_loop_vs_vectorised(
            loop_func, vec_func, repeats=1, warmup=0)
        assert result["speedup"] > 0

        # An exact comparison should catch the same difference.
        with pytest.raises(ValueError, match="different results"):
            benchmarking.compare_loop_vs_vectorised(
                loop_func, vec_func, rtol=0.0, atol=0.0)

    def test_correct_functions_are_called_for_verification_and_timing(self):
        calls = {"loop": 0, "vec": 0}

        def loop_func():
            calls["loop"] += 1
            return [1, 2, 3]

        def vec_func():
            calls["vec"] += 1
            return np.array([1, 2, 3])

        benchmarking.compare_loop_vs_vectorised(loop_func, vec_func,
                                                repeats=3, warmup=2)
        # One correctness check, two warm-ups and three timed runs for each.
        assert calls == {"loop": 6, "vec": 6}
