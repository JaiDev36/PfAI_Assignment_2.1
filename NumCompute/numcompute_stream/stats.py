"""
stats.py

Per-feature streaming statistics.

numcompute.stats.RunningStats keeps one stream of numbers (count, mean, variance, min, max) and
flattens whatever it is given. A data matrix needs one such stream per feature, which is what
ColumnStats provides: it keeps one RunningStats per column, so Welford's parallel update, the NaN
skipping and the infinity check all stay in one place.
"""

from collections import deque

import numpy as np

from numcompute.stats import RunningStats, histogram, quantile

from ._validation import NotFittedError, check_array, check_n_features, check_positive_int

# Also in this module: StreamingQuantiles (quantiles over a sliding window of rows) and
# StreamingHistogram (per-feature histograms, optionally over a sliding window of chunks). All
# three classes have update_stats(X_chunk), a synonym of update.


class ColumnStats:
    """
    Per-feature count, mean, variance, standard deviation, min and max over a stream of chunks.

    Parameters:
    ----------
    - ddof: int >= 0, delta degrees of freedom for var and std (default 0, the population
    variance, as in NumPy; use 1 for the sample variance)

    Attributes:
    ----------
    - n_features_: int or None, the number of features, set by the first non-empty update
    - n_chunks_: int, the number of non-empty chunks folded in so far
    - count_: np.ndarray of shape (n_features,), the number of non-NaN values seen per feature

    Complexity:
    ----------
    - Time complexity: O(n*m) per chunk of n rows and m features (a Python loop over the m
    columns, never over rows)
    - Space complexity: O(m) state, independent of how much data has been seen

    Note:
    ----
    - NaN values are skipped per feature, so each feature has its own count.
    - A feature with no observed value yet reports NaN for every statistic.
    - The median is not available: it needs every value seen so far, so it cannot be streamed.

    Examples:
    --------
    >>> cs = ColumnStats().update([[1.0], [2.0]]).update([[3.0], [4.0]])
    >>> cs.mean
    array([2.5])
    """

    def __init__(self, ddof=0):
        self.ddof = check_positive_int(ddof, "ddof", minimum=0)
        self.reset()

    def reset(self):
        """
        Forget everything seen so far.

        Returns:
        -------
        - self
        """
        self.n_features_ = None
        self.n_chunks_ = 0
        self._cols = []
        return self

    def update(self, X):
        """
        Fold one chunk into the running statistics.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features), NaN values are skipped per feature

        Returns:
        -------
        - self, so calls can be chained

        Raises:
        ------
        - ValueError: if X is not 2-dimensional, contains infinity or has a different number of
        features from the first chunk

        Complexity:
        ----------
        - Time complexity: O(n*m)
        - Space complexity: O(n) temporarily per column

        Note:
        ----
        - An empty chunk (no rows) changes nothing, not even n_features_.
        """
        X = check_array(X)
        if X.shape[0] == 0:
            return self
        if self.n_features_ is None:
            self.n_features_ = X.shape[1]
            self._cols = [RunningStats() for _ in range(self.n_features_)]
        check_n_features(X, self.n_features_)
        for j, col in enumerate(self._cols):
            col.update(X[:, j])
        self.n_chunks_ += 1
        return self

    partial_fit = update
    update_stats = update

    def _check(self):
        if self.n_features_ is None:
            raise NotFittedError("ColumnStats has not seen any data; call update() first.")

    @property
    def count_(self):
        """np.ndarray of int, non-NaN values seen per feature."""
        self._check()
        return np.array([c.count for c in self._cols], dtype=np.int64)

    @property
    def mean(self):
        """np.ndarray of shape (n_features,), running mean (NaN where nothing was observed)."""
        self._check()
        return np.array([c.mean for c in self._cols])

    @property
    def var(self):
        """np.ndarray of shape (n_features,), variance with ddof (NaN where count <= ddof)."""
        self._check()
        return np.array([c.variance(self.ddof) for c in self._cols])

    @property
    def std(self):
        """np.ndarray of shape (n_features,), standard deviation with ddof."""
        self._check()
        return np.array([c.std(self.ddof) for c in self._cols])

    @property
    def min(self):
        """np.ndarray of shape (n_features,), smallest value seen (NaN if none)."""
        self._check()
        return np.array([c.min for c in self._cols])

    @property
    def max(self):
        """np.ndarray of shape (n_features,), largest value seen (NaN if none)."""
        self._check()
        return np.array([c.max for c in self._cols])

    def summary(self):
        """
        Return every statistic in one dictionary (handy for logging).

        Returns:
        -------
        - dict with keys 'count', 'mean', 'var', 'std', 'min' and 'max', each an array of shape
        (n_features,)
        """
        return {"count": self.count_, "mean": self.mean, "var": self.var, "std": self.std,
                "min": self.min, "max": self.max}


class StreamingQuantiles:
    """
    Per-feature quantiles over a stream, computed on a sliding window of the latest rows.

    Parameters:
    ----------
    - window: int >= 1 or None, how many of the most recent rows to keep (default None keeps
    every row, so the answer is exact but the memory grows with the stream)

    Attributes:
    ----------
    - n_features_: int or None, the number of features
    - n_rows_: int, the number of rows currently held (at most window)
    - n_seen_: int, the number of rows ever given to update

    Raises:
    ------
    - ValueError: if window is not None or an int >= 1

    Complexity:
    ----------
    - Time complexity: update is O(chunk); quantile sorts the held rows, O(w log w) per feature
    for w held rows
    - Space complexity: O(w * m) for w rows and m features

    Note:
    ----
    - A quantile cannot be updated from a few running numbers the way a mean can, so this class
    keeps the rows and calls numcompute.stats.quantile. With a window the result describes the
    latest `window` rows, which also lets it follow a drifting stream.
    - NaN is ignored per feature, as in numcompute.stats.quantile.

    Examples:
    --------
    >>> sq = StreamingQuantiles(window=4).update([[1.0], [2.0]]).update([[3.0], [4.0], [5.0]])
    >>> sq.median()
    array([3.5])
    """

    def __init__(self, window=None):
        if window is not None:
            check_positive_int(window, "window", minimum=1)
        self.window = window
        self.reset()

    def reset(self):
        """
        Forget every row.

        Returns:
        -------
        - self
        """
        self.n_features_ = None
        self.n_rows_ = 0
        self.n_seen_ = 0
        self._chunks = deque()
        return self

    def update(self, X):
        """
        Add one chunk and drop the oldest rows beyond the window.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if X is not 2-dimensional, contains infinity or has a different number of
        features from the first chunk
        """
        X = check_array(X)
        if X.shape[0] == 0:
            return self
        if self.n_features_ is None:
            self.n_features_ = X.shape[1]
        check_n_features(X, self.n_features_)
        self._chunks.append(X)
        self.n_rows_ += X.shape[0]
        self.n_seen_ += X.shape[0]
        if self.window is not None:
            while self.n_rows_ > self.window:
                excess = self.n_rows_ - self.window
                oldest = self._chunks[0]
                if oldest.shape[0] <= excess:
                    self._chunks.popleft()
                    self.n_rows_ -= oldest.shape[0]
                else:
                    self._chunks[0] = oldest[excess:]
                    self.n_rows_ -= excess
        return self

    partial_fit = update
    update_stats = update

    def quantile(self, q):
        """
        Quantile(s) of the rows currently held.

        Parameters:
        ----------
        - q: float or 1D array-like of floats in [0, 1]

        Returns:
        -------
        - np.ndarray of shape (n_features,) for a scalar q, or (len(q), n_features) for a 1D q

        Raises:
        ------
        - NotFittedError: if no rows have been added
        - ValueError: if q is outside [0, 1] (from numcompute.stats.quantile)
        """
        if self.n_features_ is None:
            raise NotFittedError("StreamingQuantiles has not seen any data; call update() first.")
        data = np.vstack(list(self._chunks))
        return quantile(data, q, axis=0)

    def median(self):
        """np.ndarray of shape (n_features,), the median of the rows currently held."""
        return self.quantile(0.5)


class StreamingHistogram:
    """
    Per-feature histograms over a stream, with fixed bins and an optional sliding window.

    Parameters:
    ----------
    - bins: int >= 1, the number of equal-width bins (default 10)
    - value_range: (low, high) or None. None takes each feature's range from the first
    non-empty chunk. Later values outside the range are not counted in a bin; they add to
    n_outside_.
    - window: int >= 1 or None, count only the latest `window` chunks (default None counts every
    chunk since the start)

    Attributes:
    ----------
    - n_features_: int or None
    - edges_: np.ndarray of shape (n_features, bins + 1), the bin edges (fixed once set)
    - counts_: np.ndarray of shape (n_features, bins), the counts over the chunks being counted
    - n_outside_: int, the number of values (over all features) that fell outside the range

    Raises:
    ------
    - ValueError: if bins or window is not a positive int, or value_range is not (low, high)
    with low < high

    Complexity:
    ----------
    - Time complexity: O(n*m + m*bins) per chunk of n rows and m features
    - Space complexity: O(m * bins), or O(window * m * bins) with a window

    Note:
    ----
    - Keeping the bin edges fixed is what makes the counts of different chunks addable. Each
    chunk is counted with numcompute.stats.histogram.
    - NaN is ignored.
    - With a window, the oldest chunk's counts are subtracted when a new chunk pushes it out.
    """

    def __init__(self, bins=10, value_range=None, window=None):
        check_positive_int(bins, "bins", minimum=1)
        if window is not None:
            check_positive_int(window, "window", minimum=1)
        if value_range is not None:
            try:
                low, high = (float(v) for v in value_range)
            except (TypeError, ValueError):
                raise ValueError("value_range must be (low, high). Got: {!r}".format(value_range))
            if not (np.isfinite(low) and np.isfinite(high) and low < high):
                raise ValueError("value_range needs finite low < high. Got: {!r}"
                                 .format(value_range))
            value_range = (low, high)
        self.bins = bins
        self.value_range = value_range
        self.window = window
        self.reset()

    def reset(self):
        """
        Forget every chunk (the bin edges are forgotten too).

        Returns:
        -------
        - self
        """
        self.n_features_ = None
        self.edges_ = None
        self.counts_ = None
        self.n_outside_ = 0
        self._per_chunk = deque()
        return self

    def update(self, X):
        """
        Count one chunk.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if X is not 2-dimensional, contains infinity or has a different number of
        features from the first chunk, or a feature has no value to set the range from
        """
        X = check_array(X)
        if X.shape[0] == 0:
            return self
        if self.n_features_ is None:
            self._set_edges(X)
        check_n_features(X, self.n_features_)
        counts = np.zeros((self.n_features_, self.bins), dtype=np.int64)
        outside = 0
        for j in range(self.n_features_):
            col = X[:, j]
            col = col[~np.isnan(col)]
            lo, hi = self.edges_[j, 0], self.edges_[j, -1]
            inside = col[(col >= lo) & (col <= hi)]
            outside += col.size - inside.size
            if inside.size:
                counts[j] = histogram(inside, bins=self.bins, value_range=(lo, hi))[0]
        self.counts_ = counts if self.counts_ is None else self.counts_ + counts
        self.n_outside_ += outside
        if self.window is not None:
            self._per_chunk.append((counts, outside))
            while len(self._per_chunk) > self.window:
                old_counts, old_outside = self._per_chunk.popleft()
                self.counts_ = self.counts_ - old_counts
                self.n_outside_ -= old_outside
        return self

    partial_fit = update
    update_stats = update

    def _set_edges(self, X):
        self.n_features_ = X.shape[1]
        edges = np.empty((self.n_features_, self.bins + 1))
        for j in range(self.n_features_):
            if self.value_range is not None:
                lo, hi = self.value_range
            else:
                col = X[:, j][~np.isnan(X[:, j])]
                if col.size == 0:
                    self.n_features_ = None
                    raise ValueError("Feature {} has no observed value to set the histogram "
                                     "range from; pass value_range.".format(j))
                lo, hi = float(col.min()), float(col.max())
                if lo == hi:
                    lo, hi = lo - 0.5, hi + 0.5
            edges[j] = np.linspace(lo, hi, self.bins + 1)
        self.edges_ = edges
