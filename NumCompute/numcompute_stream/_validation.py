"""
_validation.py

Input checks shared by every numcompute_stream module.

The wording of the errors follows numcompute's own (for example "Input data must have 2 dimensions.
Got shape: ..." and "Incompatible dimensionality."), so the contract tests written for
numcompute's transformers also pass for the streaming ones.

Conventions used throughout the package:
- X is a 2-D float array of shape (n_samples, n_features). A 1-D input is an error, as in
  numcompute.
- y is a 1-D array of shape (n_samples,).
- axis=0 always means "down the rows", so one result per feature.
"""

import numpy as np


class NotFittedError(RuntimeError):
    """
    Raised when a model or transformer is used before it has seen enough data.

    Note:
    ----
    - It is a RuntimeError, because numcompute raises RuntimeError for transform before fit.
    """


def check_array(X, name="Input data", allow_nan=True, min_samples=0):
    """
    Convert X to a 2-D float64 array and check it.

    Parameters:
    ----------
    - X: array-like of shape (n_samples, n_features)
    - name: str, used in the messages for the checks other than the dimension check
    - allow_nan: bool, if False a NaN raises ValueError (default True)
    - min_samples: int, minimum number of rows (default 0, so an empty chunk is allowed)

    Returns:
    -------
    - X: np.ndarray of dtype float64 and shape (n_samples, n_features)

    Raises:
    ------
    - TypeError: if X is None
    - ValueError: if X is not numeric, is not 2-dimensional, has too few rows or contains
    infinity (or NaN when allow_nan is False)

    Complexity:
    ----------
    - Time complexity: O(n*m) for n rows and m features
    - Space complexity: O(1) if X is already a float64 array, otherwise O(n*m)
    """
    if X is None:
        raise TypeError("{} must be array-like, got None.".format(name))
    try:
        arr = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("{} must be numeric.".format(name)) from None
    if arr.ndim != 2:
        raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(arr.shape))
    if arr.shape[0] < min_samples:
        raise ValueError("{} needs at least {} sample(s). Got shape: {}"
                         .format(name, min_samples, arr.shape))
    if np.isinf(arr).any():
        raise ValueError("{} contains infinite values.".format(name))
    if not allow_nan and np.isnan(arr).any():
        raise ValueError("{} contains NaN; impute missing values first.".format(name))
    return arr


def check_labels(y, name="y"):
    """
    Return y as a 1-D array and reject missing labels.

    Parameters:
    ----------
    - y: array-like of shape (n_samples,) holding integers, floats or strings. A column vector
    of shape (n_samples, 1) is flattened.
    - name: str, used in the messages

    Returns:
    -------
    - y: np.ndarray of shape (n_samples,)

    Raises:
    ------
    - TypeError: if y is None
    - ValueError: if y is not 1-dimensional or is a float array containing NaN

    Complexity:
    ----------
    - Time complexity: O(n)
    - Space complexity: O(1) if y is already an array
    """
    if y is None:
        raise TypeError("{} must be array-like, got None.".format(name))
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError("{} must be 1-dimensional. Got shape: {}".format(name, arr.shape))
    if arr.dtype.kind == "f" and np.isnan(arr).any():
        raise ValueError("{} contains NaN labels; drop those rows before training.".format(name))
    return arr


def check_X_y(X, y, allow_nan=True, min_samples=0):
    """
    Validate a feature matrix and its labels together.

    Parameters:
    ----------
    - X: array-like of shape (n_samples, n_features)
    - y: array-like of shape (n_samples,)
    - allow_nan: bool, see check_array
    - min_samples: int, see check_array

    Returns:
    -------
    - X: np.ndarray of shape (n_samples, n_features), float64
    - y: np.ndarray of shape (n_samples,)

    Raises:
    ------
    - ValueError: if X or y fail their own checks, or have different numbers of rows

    Complexity:
    ----------
    - Time complexity: O(n*m)
    - Space complexity: O(1) if the inputs are already arrays
    """
    X = check_array(X, allow_nan=allow_nan, min_samples=min_samples)
    y = check_labels(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have different numbers of samples: X has {}, y has {}."
                         .format(X.shape[0], y.shape[0]))
    return X, y


def check_n_features(X, expected):
    """
    Raise ValueError if X does not have the expected number of columns.

    Parameters:
    ----------
    - X: np.ndarray of shape (n_samples, n_features)
    - expected: int or None, the number of features the object was fitted with (None skips)

    Raises:
    ------
    - ValueError: "Incompatible dimensionality" with both feature counts

    Complexity:
    ----------
    - Time and space complexity: O(1)
    """
    if expected is not None and X.shape[1] != expected:
        raise ValueError("Incompatible dimensionality. X has {} feature(s), but this object was "
                         "fitted with {}.".format(X.shape[1], expected))


def check_positive_int(value, name, minimum=1):
    """
    Check that value is an integer of at least minimum (a bool is rejected).

    Parameters:
    ----------
    - value: the value to check
    - name: str, used in the messages
    - minimum: int, smallest allowed value (default 1)

    Returns:
    -------
    - int: value as a Python int

    Raises:
    ------
    - TypeError: if value is not an integer
    - ValueError: if value is smaller than minimum

    Complexity:
    ----------
    - Time and space complexity: O(1)
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("{} must be an integer. Got: {}".format(name, type(value).__name__))
    if value < minimum:
        raise ValueError("{} must be >= {}. Got: {}".format(name, minimum, value))
    return int(value)
