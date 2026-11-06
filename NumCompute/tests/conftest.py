"""Shared fixtures. Matplotlib uses the non-interactive Agg backend so plots never open a window."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def blobs(rng):
    """Three well-separated classes in 2-D: an easy problem a tree should solve."""
    centres = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    y = np.repeat([0, 1, 2], 100)
    X = centres[y] + rng.normal(scale=0.6, size=(300, 2))
    order = rng.permutation(300)
    return X[order], y[order]


@pytest.fixture
def xor_data(rng):
    """XOR: no single split helps much, but depth 2 solves it exactly."""
    X = rng.uniform(-1, 1, size=(400, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X, y
