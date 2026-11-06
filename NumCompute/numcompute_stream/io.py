"""
io.py

Loading a labelled CSV and cutting it into chunks.

The parsing is numcompute.io.load_csv (blank and "nan" cells become NaN, errors name the row and
column). This module adds what a classifier needs on top of it: it splits off the label column,
reads the feature names from the header, and turns the arrays into a stream of chunks with
numcompute.utils.batches.

Not provided: reading a file lazily, one chunk at a time. numcompute.io.load_csv reads a whole file,
so the stream here is simulated from arrays in memory. That is faithful for the algorithm (the model
only ever sees one chunk at a time) but it does not bound memory.
"""

from os import path

import numpy as np

from numcompute.io import load_csv as _load_csv
from numcompute.utils import batches

from ._validation import check_labels, check_positive_int


def read_header(file_path, delimiter=","):
    """
    Read the column names from the first line of a CSV file.

    Parameters:
    ----------
    - file_path: path to the CSV file
    - delimiter: str, the delimiter used in the file (default ',')

    Returns:
    -------
    - names: list of str

    Raises:
    ------
    - FileNotFoundError: if the file does not exist
    - ValueError: if the file is empty
    """
    with open(file_path, "r", newline="") as handle:
        first = handle.readline()
    if not first.strip():
        raise ValueError("File {!r} is empty.".format(str(file_path)))
    return [cell.strip() for cell in first.strip("\r\n").split(delimiter)]


def _resolve_column(names, column, n_columns):
    """Turn a column name or a (possibly negative) index into a 0-based index."""
    if isinstance(column, str):
        if column not in names:
            raise ValueError("Label column {!r} not found. Columns are: {}".format(column, names))
        return names.index(column)
    if isinstance(column, bool) or not isinstance(column, (int, np.integer)):
        raise TypeError("label_column must be a column name or an integer. Got: {}"
                        .format(type(column).__name__))
    index = column + n_columns if column < 0 else column
    if not 0 <= index < n_columns:
        raise ValueError("label_column {} is out of range for {} columns."
                         .format(column, n_columns))
    return int(index)


def _tidy_labels(labels):
    """Whole-number labels become int64, other numbers stay float, text stays text."""
    try:
        numbers = np.array([float(v) for v in labels])
    except (TypeError, ValueError):
        return np.array([str(v) for v in labels])
    if numbers.size and np.all(numbers == np.round(numbers)):
        return numbers.astype(np.int64)
    return numbers


def load_labelled_csv(file_path, label_column=-1, delimiter=",", header=True,
                      on_missing_label="raise"):
    """
    Load a CSV file into features, labels and feature names.

    Parameters:
    ----------
    - file_path: path to the CSV file
    - label_column: str (a header name) or int (an index, negative counts from the end); the
    default is the last column
    - delimiter: str (default ',')
    - header: bool, True if the first line holds the column names (default True)
    - on_missing_label: 'raise' (default) or 'drop', what to do with a row whose label is empty

    Returns:
    -------
    - X: np.ndarray of shape (n_samples, n_features), float64, NaN where a cell was empty
    - y: np.ndarray of shape (n_samples,), int64 if every label is a whole number, else float
    or str
    - feature_names: list of str, the header without the label column (x0, x1, ... without a
    header)

    Raises:
    ------
    - FileNotFoundError: if the file does not exist
    - ValueError: if the file has no data, label_column is invalid, a feature cell is not
    numeric, on_missing_label is invalid, or a label is missing while on_missing_label='raise'

    Complexity:
    ----------
    - Time complexity: O(n*m) for n rows and m columns
    - Space complexity: O(n*m)

    Note:
    ----
    - The file is parsed by numcompute.io.load_csv. Numeric labels are read as numbers; if the
    label column holds text the file is read again as text and only the feature columns are
    converted to numbers.
    """
    if on_missing_label not in ("raise", "drop"):
        raise ValueError("on_missing_label must be 'raise' or 'drop'. Got: {!r}"
                         .format(on_missing_label))
    if not path.exists(file_path):
        raise FileNotFoundError("No such file: {!r}".format(str(file_path)))
    names = read_header(file_path, delimiter) if header else None
    skip = 1 if header else 0

    try:
        table = _load_csv(file_path, delimiter=delimiter, skip_header=skip, dtype=float)
        text_labels = False
    except TypeError:                               # a text column: read everything as objects
        table = _load_csv(file_path, delimiter=delimiter, skip_header=skip, dtype=object)
        text_labels = True
    if table.size == 0:
        raise ValueError("File {!r} contains no data rows.".format(str(file_path)))
    if table.shape[1] < 2:
        raise ValueError("A labelled CSV needs at least 2 columns. Got {}.".format(table.shape[1]))
    if names is None:
        names = ["x{}".format(j) for j in range(table.shape[1])]
    if len(names) != table.shape[1]:
        raise ValueError("The header has {} names but the rows have {} columns."
                         .format(len(names), table.shape[1]))

    col = _resolve_column(names, label_column, table.shape[1])
    feature_cols = [j for j in range(table.shape[1]) if j != col]
    try:
        X = np.asarray(table[:, feature_cols], dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("Feature columns must be numeric; encode text features first.") \
            from None
    raw = table[:, col]
    if text_labels:
        missing = np.array([isinstance(v, float) and np.isnan(v) for v in raw])
    else:
        missing = np.isnan(raw.astype(np.float64))
    if missing.any():
        if on_missing_label == "raise":
            raise ValueError("{} row(s) have no label (first at data row {}). Use "
                             "on_missing_label='drop' to skip them."
                             .format(int(missing.sum()), int(np.flatnonzero(missing)[0])))
        X, raw = X[~missing], raw[~missing]
    y = _tidy_labels(raw.tolist())
    return X, y, [names[j] for j in feature_cols]


def iter_chunks(X, y, chunk_size):
    """
    Cut arrays into consecutive chunks, in their original order.

    Parameters:
    ----------
    - X: array-like of shape (n_samples, n_features)
    - y: array-like of shape (n_samples,)
    - chunk_size: int >= 1, rows per chunk (the last chunk may be shorter)

    Returns:
    -------
    - an iterator of (X_chunk, y_chunk) tuples of np.ndarray

    Raises:
    ------
    - ValueError: if X is not 2-dimensional, or X and y have different lengths
    - TypeError: if chunk_size is not an integer

    Complexity:
    ----------
    - Time complexity: O(n*m) in total
    - Space complexity: O(chunk_size * m) per chunk

    Note:
    ----
    - The order is never shuffled: a stream arrives in time order, and a shuffle would hide the
    concept drift that streaming models are meant to cope with.
    - The arguments are checked when iter_chunks is called, not when the first chunk is read.
    """
    chunk_size = check_positive_int(chunk_size, "chunk_size")
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))
    y = check_labels(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have different numbers of samples: X has {}, y has {}."
                         .format(X.shape[0], y.shape[0]))
    return batches(X, y, batch_size=chunk_size)
