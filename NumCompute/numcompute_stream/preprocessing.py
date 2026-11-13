"""
preprocessing.py

Streaming versions of numcompute's preprocessing, learned one chunk at a time:
- StreamingStandardScaler  (subclass of numcompute.preprocessing.StandardScaler)
- StreamingMinMaxScaler    (subclass of numcompute.preprocessing.MinMaxScaler)
- StreamingImputer         (running-mean or constant imputation, a numcompute Transformer)
- LabelEncoder             (class labels to integers, growing as new classes appear)

API, on top of numcompute's fit(X, y=None), transform(X) and fit_transform(X, y=None):
- partial_fit(X, y=None) -> self   learns from one more chunk and keeps what was learned before
- reset() -> self                  forgets everything

fit is reset followed by partial_fit on the whole input, so batch and streaming learning share one
code path. The scalers inherit transform from numcompute, so they keep its checks and its error
messages.

StreamingOneHotEncoder expands its categories as new ones arrive, so by default the number of
output columns can grow during a stream. A model such as DecisionTree needs a fixed width, so in
front of one, give the encoder its categories (categories=[...]) or use handle_unknown="error":
an unknown category is then rejected at the encoder before anything changes.
"""

import numpy as np

from numcompute.pipeline import Transformer
from numcompute.preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler

from ._validation import NotFittedError, check_array, check_labels, check_n_features
from .stats import ColumnStats


def _reject_empty_or_unobserved(owner, X, stats, check_unobserved):
    """Shared checks of the fit methods: an empty array, and (optionally) an all-NaN column."""
    if X.size == 0:
        raise ValueError("Cannot fit {} on an empty array.".format(type(owner).__name__))
    if check_unobserved and np.any(stats.count_ == 0):
        raise ValueError("Cannot fit {} on a column with all NaN values."
                         .format(type(owner).__name__))


class StreamingStandardScaler(StandardScaler):
    """
    Standardise features to zero mean and unit variance, learned incrementally.

    Attributes:
    ----------
    - mean_: np.ndarray of shape (n_features,), the running mean (0 for a feature never observed)
    - scale_: np.ndarray of shape (n_features,), the running standard deviation, or 1 for a
    constant feature and for a feature never observed

    Complexity:
    ----------
    - Time complexity: partial_fit is O(n*m) for a chunk of n rows and m features, so a stream of
    K chunks costs O(total rows), not O(K * total rows) as refitting on all the data would
    - Space complexity: O(m) state

    Note:
    ----
    - Same rules as StandardScaler: NaN is ignored when learning and stays NaN after transform;
    a constant column (running min == max) gets a scale of 1.
    - fit behaves like StandardScaler.fit and raises on an empty array or an all-NaN column.
    partial_fit is more forgiving: an empty chunk changes nothing, and a chunk where one column is
    all NaN is fine, because that column may be observed in a later chunk.
    - The mean and standard deviation come from ColumnStats (Welford's parallel update, scaled
    for values above 2**400), so learning does not overflow even for values near 1e308. The
    inherited transform computes x - mean, which can still overflow when a value is more than
    1.8e308 from the mean, as in numcompute's StandardScaler.
    - Parameters change as chunks arrive, so the same row transforms differently before and after
    a partial_fit. Scale a chunk with the parameters learned so far, which is what a deployed
    streaming pipeline can do.
    """

    def __init__(self):
        super().__init__()
        self._stats = ColumnStats()

    def reset(self):
        """
        Forget everything learned so far.

        Returns:
        -------
        - self
        """
        self._stats.reset()
        self.__dict__.pop("mean_", None)
        self.__dict__.pop("scale_", None)
        return self

    def partial_fit(self, X, y=None):
        """
        Update the running mean and standard deviation with one chunk.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: ignored, accepted for Pipeline compatibility

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if X is not 2-dimensional, contains infinity or has a different number of
        features from earlier chunks

        Complexity:
        ----------
        - Time complexity: O(n*m)
        - Space complexity: O(m)
        """
        self._stats.update(X)
        if self._stats.n_features_ is None:
            return self
        seen = self._stats.count_ > 0
        std = self._stats.std
        varies = seen & (self._stats.max != self._stats.min) & (std > 0)
        self.mean_ = np.where(seen, self._stats.mean, 0.0)
        self.scale_ = np.where(varies, std, 1.0)
        return self

    update = partial_fit

    def fit(self, X, y=None):
        """
        Forget what was learned and learn from X alone.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: ignored

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if X is empty, not 2-dimensional, or has a column with all NaN values
        """
        X = check_array(X)
        _reject_empty_or_unobserved(self, X, None, check_unobserved=False)
        self.reset().partial_fit(X)
        _reject_empty_or_unobserved(self, X, self._stats, check_unobserved=True)
        return self


class StreamingMinMaxScaler(MinMaxScaler):
    """
    Rescale features to feature_range using the running minimum and maximum.

    Parameters:
    ----------
    - feature_range: tuple (low, high) with low < high, default (0, 1)
    - clip: bool, if True the output is clipped to feature_range (default False)

    Attributes:
    ----------
    - min_, max_: np.ndarray of shape (n_features,), the running extremes (0 for a feature never
    observed)
    - scale_: np.ndarray of shape (n_features,), the scaling factor, worked out by
    MinMaxScaler.fit

    Complexity:
    ----------
    - Time complexity: partial_fit is O(n*m) for a chunk of n rows and m features
    - Space complexity: O(m) state

    Note:
    ----
    - A new chunk can hold values outside the range seen so far. Without clip they map outside
    feature_range, exactly as MinMaxScaler does for unseen data; with clip=True they are clipped.
    - The scale is not computed again here: partial_fit passes the two extreme rows (running
    minimum and maximum) to MinMaxScaler.fit, which only looks at the extremes, so the
    half-range arithmetic that avoids overflow is shared with the batch scaler.
    - fit raises on an empty array or an all-NaN column, like MinMaxScaler.fit; partial_fit
    tolerates both (see StreamingStandardScaler).
    """

    def __init__(self, feature_range=(0, 1), clip=False):
        super().__init__(feature_range)
        self.clip = bool(clip)
        self._stats = ColumnStats()

    def reset(self):
        """
        Forget everything learned so far.

        Returns:
        -------
        - self
        """
        self._stats.reset()
        for name in ("min_", "max_", "scale_"):
            self.__dict__.pop(name, None)
        return self

    def partial_fit(self, X, y=None):
        """
        Update the running minimum and maximum with one chunk.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: ignored, accepted for Pipeline compatibility

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if X is not 2-dimensional, contains infinity or has a different number of
        features from earlier chunks

        Complexity:
        ----------
        - Time complexity: O(n*m)
        - Space complexity: O(m)
        """
        self._stats.update(X)
        if self._stats.n_features_ is None:
            return self
        seen = self._stats.count_ > 0
        extremes = np.vstack([np.where(seen, self._stats.min, 0.0),
                              np.where(seen, self._stats.max, 0.0)])
        MinMaxScaler.fit(self, extremes)
        return self

    update = partial_fit

    def fit(self, X, y=None):
        """
        Forget what was learned and learn from X alone.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: ignored

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if X is empty, not 2-dimensional, or has a column with all NaN values
        """
        X = check_array(X)
        _reject_empty_or_unobserved(self, X, None, check_unobserved=False)
        self.reset().partial_fit(X)
        _reject_empty_or_unobserved(self, X, self._stats, check_unobserved=True)
        return self

    def transform(self, X):
        """
        Scale X with the parameters learned so far.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - X_out: np.ndarray of shape (n_samples, n_features), float64

        Raises:
        ------
        - RuntimeError: if no data has been seen yet
        - ValueError: if X is not 2-dimensional or has a different number of features
        """
        out = super().transform(X)
        if self.clip:
            out = np.clip(out, *self.feature_range)  # NaN stays NaN
        return out


class StreamingImputer(Transformer):
    """
    Replace NaN with the running per-feature mean, or with a constant.

    Parameters:
    ----------
    - strategy: 'mean' (the running mean of each feature) or 'constant' (default 'mean')
    - fill_value: finite number used by the 'constant' strategy, and by the 'mean' strategy for a
    feature that has not had an observed value yet (default 0)

    Attributes:
    ----------
    - columns_: int, the number of features
    - fill_values_: np.ndarray of shape (n_features,), the value that replaces NaN in each feature

    Complexity:
    ----------
    - Time complexity: partial_fit and transform are O(n*m) for n rows and m features
    - Space complexity: O(m) state

    Note:
    ----
    - numcompute.preprocessing.SimpleImputer only fills a constant. This class adds the running
    mean, and reuses SimpleImputer to validate fill_value.
    - fit raises on an empty array; partial_fit ignores an empty chunk.
    """

    def __init__(self, strategy="mean", fill_value=0):
        if strategy not in ("mean", "constant"):
            raise ValueError("strategy must be 'mean' or 'constant'. Got: {!r}".format(strategy))
        self.fill_value = SimpleImputer(fill_value).fill_value  # validates the value
        self.strategy = strategy
        self._stats = ColumnStats()

    def reset(self):
        """
        Forget everything learned so far.

        Returns:
        -------
        - self
        """
        self._stats.reset()
        self.__dict__.pop("columns_", None)
        return self

    def partial_fit(self, X, y=None):
        """
        Update the running means with one chunk.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: ignored, accepted for Pipeline compatibility

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if X is not 2-dimensional, contains infinity or has a different number of
        features from earlier chunks
        """
        self._stats.update(X)
        if self._stats.n_features_ is not None:
            self.columns_ = self._stats.n_features_
        return self

    update = partial_fit

    def fit(self, X, y=None):
        """
        Forget what was learned and learn from X alone.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: ignored

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if X is empty or not 2-dimensional
        """
        X = check_array(X)
        _reject_empty_or_unobserved(self, X, None, check_unobserved=False)
        return self.reset().partial_fit(X)

    @property
    def fill_values_(self):
        """np.ndarray of shape (n_features,), the replacement for NaN in each feature."""
        if not hasattr(self, "columns_"):
            raise NotFittedError("fit must be called before transform. Call partial_fit() with a "
                                 "first chunk.")
        if self.strategy == "constant":
            return np.full(self.columns_, float(self.fill_value))
        means = self._stats.mean
        return np.where(np.isnan(means), float(self.fill_value), means)

    def transform(self, X):
        """
        Return a copy of X with every NaN replaced.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - X_out: np.ndarray of shape (n_samples, n_features), float64

        Raises:
        ------
        - RuntimeError: if no data has been seen yet
        - ValueError: if X is not 2-dimensional or has a different number of features
        """
        fill = self.fill_values_
        X = check_array(X)
        check_n_features(X, self.columns_)
        return np.where(np.isnan(X), fill, X)


class LabelEncoder:
    """
    Map class labels to the integers 0 .. K-1, growing as new classes appear in the stream.

    Codes are given in order of first appearance and never change afterwards, so a model trained on
    earlier chunks stays valid when a later chunk brings a new class.

    Attributes:
    ----------
    - classes_: np.ndarray of shape (n_classes,), the labels in code order

    Complexity:
    ----------
    - Time complexity: O(n log n) per chunk of n labels for partial_fit, O(n) for transform
    - Space complexity: O(K) for K classes
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Forget every label.

        Returns:
        -------
        - self
        """
        self.classes_ = np.array([])
        self._index = {}
        return self

    @staticmethod
    def _python(label):
        return label.item() if hasattr(label, "item") else label

    def partial_fit(self, y):
        """
        Register the labels in y that have not been seen before.

        Parameters:
        ----------
        - y: array-like of shape (n_samples,)

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if y is not 1-dimensional or contains NaN
        """
        y = check_labels(y)
        first_seen = np.sort(np.unique(y, return_index=True)[1])
        for label in y[first_seen]:
            key = self._python(label)
            if key not in self._index:
                self._index[key] = len(self._index)
        self.classes_ = np.array(list(self._index))
        return self

    update = partial_fit

    def fit(self, y):
        """
        Forget earlier labels, then register the labels in y.

        Parameters:
        ----------
        - y: array-like of shape (n_samples,)

        Returns:
        -------
        - self
        """
        return self.reset().partial_fit(y)

    def transform(self, y):
        """
        Convert labels to integer codes.

        Parameters:
        ----------
        - y: array-like of shape (n_samples,)

        Returns:
        -------
        - codes: np.ndarray of int64 and shape (n_samples,)

        Raises:
        ------
        - ValueError: if y contains a label that was never passed to partial_fit
        """
        y = check_labels(y)
        keys = [self._python(v) for v in y]
        unknown = sorted({str(k) for k in keys if k not in self._index})
        if unknown:
            raise ValueError("Unseen label(s): {}. Call partial_fit() first."
                             .format(", ".join(unknown)))
        return np.array([self._index[k] for k in keys], dtype=np.int64)

    def fit_transform(self, y):
        """
        Fit to y, then convert it.

        Parameters:
        ----------
        - y: array-like of shape (n_samples,)

        Returns:
        -------
        - codes: np.ndarray of int64 and shape (n_samples,)
        """
        return self.fit(y).transform(y)

    def inverse_transform(self, codes):
        """
        Convert integer codes back to the original labels.

        Parameters:
        ----------
        - codes: array-like of int, values in [0, n_classes)

        Returns:
        -------
        - labels: np.ndarray of the same shape as codes

        Raises:
        ------
        - ValueError: if a code is outside [0, n_classes)
        """
        codes = np.asarray(codes, dtype=np.int64)
        if codes.size and (codes.min() < 0 or codes.max() >= len(self.classes_)):
            raise ValueError("Codes must lie in [0, {}].".format(len(self.classes_) - 1))
        return self.classes_[codes]


class StreamingOneHotEncoder(OneHotEncoder):
    """
    One-hot encoder whose categories can grow as new ones appear in later chunks, or stay fixed.

    Parameters:
    ----------
    - categories: None, or a list with one list of allowed values per feature. When given, the
    output columns are fixed from the start and a value outside the list is an error.
    - handle_unknown: "expand" (default) or "error". With "expand" a new category adds a column.
    With "error" the categories of the first non-empty chunk are kept, and a later chunk with a
    new category raises ValueError before anything changes. Ignored when categories is given
    (the list is then always fixed).

    Attributes:
    ----------
    - categories_: list of arrays, the sorted categories of each feature (NaN last)

    Raises:
    ------
    - ValueError: if handle_unknown is not "expand" or "error", if X is not 2-dimensional or has
    a different number of features from earlier chunks, if a fixed encoder meets a category it
    does not know, or (in transform) if X holds a category no chunk has shown yet
    - TypeError: if a column mixes strings and numbers

    Complexity:
    ----------
    - Time complexity: partial_fit is O(n log n) per column for a chunk of n rows
    - Space complexity: O(k) per column for k categories

    Note:
    ----
    - A subclass of numcompute.preprocessing.OneHotEncoder, so transform and its checks are
    inherited. partial_fit merges the chunk's categories into the known ones.
    - The categories stay sorted, so with "expand" a new category can land between old ones and
    move the later columns of the output, and the width of the output grows. A model after the
    encoder (such as DecisionTree) needs a fixed number of features and rejects the wider
    output. In a pipeline with a tree, give categories or use handle_unknown="error", so the
    stream fails at the encoder, with nothing changed, instead of at the tree.
    - A chunk is registered by partial_fit before it is transformed, so the usual order
    (partial_fit, then transform) never meets an unseen category. fit forgets everything first.
    - An empty chunk changes nothing.

    Examples:
    --------
    >>> enc = StreamingOneHotEncoder().partial_fit([["a"], ["b"]]).partial_fit([["c"]])
    >>> enc.transform([["c"], ["a"]]).tolist()
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    """

    def __init__(self, categories=None, handle_unknown="expand"):
        super().__init__()
        if handle_unknown not in ("expand", "error"):
            raise ValueError('handle_unknown must be "expand" or "error". Got: {!r}'
                             .format(handle_unknown))
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.reset()

    @property
    def _fixed(self):
        return self.categories is not None or self.handle_unknown == "error"

    def reset(self):
        """
        Forget every category learned from data (a categories list given to the constructor stays).

        Returns:
        -------
        - self
        """
        if hasattr(self, "categories_"):
            del self.categories_
        if self.categories is not None:
            self.categories_ = [self._unique(self._as_array([[v] for v in values])[:, 0])
                                for values in self.categories]
        return self

    def _merged(self, X):
        """The categories after adding X, or ValueError if a fixed encoder meets a new one."""
        if X.shape[1] != len(self.categories_):
            raise ValueError("Incompatible dimensionality. Expected {} features, got {}."
                             .format(len(self.categories_), X.shape[1]))
        merged = [self._unique(np.concatenate([known, X[:, i]]))
                  for i, known in enumerate(self.categories_)]
        if self._fixed:
            for i, (known, new) in enumerate(zip(self.categories_, merged)):
                if len(new) != len(known):
                    raise ValueError("Feature {} has a category the encoder does not know, and "
                                     "its categories are fixed (categories given or "
                                     "handle_unknown='error').".format(i))
        return merged

    def partial_fit(self, X, y=None):
        """
        Add the categories of one chunk.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: ignored, present so the signature matches the other transformers

        Returns:
        -------
        - self
        """
        X = self._as_array(X)
        if X.ndim != 2:
            raise ValueError("Input data must have 2 dimensions. Got shape: {}".format(X.shape))
        if X.shape[0] == 0:
            return self
        if not hasattr(self, "categories_"):
            self.categories_ = [self._unique(X[:, i]) for i in range(X.shape[1])]
            return self
        self.categories_ = self._merged(X)      # checked in full before anything is stored
        return self

    update = partial_fit

    def fit(self, X, y=None):
        """
        Forget everything, then learn the categories of X.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features), not empty
        - y: ignored

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if X is empty or not 2-dimensional, or holds a category outside a given
        categories list
        """
        self.reset()
        if self.categories is None:
            return super().fit(X, y)
        fixed = self.categories_
        super().fit(X, y)                       # the inherited checks of X
        self.categories_ = fixed
        self._merged(self._as_array(X))         # raises if X is outside the list
        return self


# The names the assignment brief uses.
StandardScaler = StreamingStandardScaler
Imputer = StreamingImputer
OneHotEncoder = StreamingOneHotEncoder
