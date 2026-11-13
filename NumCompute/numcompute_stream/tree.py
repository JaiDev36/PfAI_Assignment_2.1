"""
tree.py

A classification tree built from scratch (NumPy only), retrained from a buffer as data streams in.

How the tree is grown (the four steps of the Week 10 notes):
1. Try every question "x[:, f] <= t" for every feature f and every threshold t halfway between two
   neighbouring distinct values.
2. Keep the question with the largest information gain
       gain = I(parent) - (n_left / n) * I(left) - (n_right / n) * I(right)
   where I is the Gini impurity 1 - sum(p_k^2) or the entropy -sum(p_k * log2(p_k)).
3. Split the rows and repeat on each side.
4. Stop at max_depth, at a pure node, at a node too small to split, or when no question lowers the
   impurity.

How streaming works (buffered retraining):
partial_fit appends a chunk to an internal buffer. When the buffer holds at least buffer_size
samples, a new tree is trained on the buffered samples and the buffer is cleared. Between retrains
predict uses the most recently trained tree. A larger buffer_size gives each tree more data but
fewer, slower and more out-of-date retrains (benchmark/bench_stream.py measures the trade).

How the split search is vectorised:
For one feature, sort the rows by that feature. The class counts on the left of every possible cut
are then the cumulative sum of the one-hot labels, and the right counts are total - left. So every
threshold of every feature is scored at once with a sort, take_along_axis and cumsum: O(n log n)
per feature, instead of O(n^2) for testing each threshold with a fresh mask (best_split_loop).

Determinism and edge cases:
- Ties between splits go to the lowest feature index, then the lowest threshold, in both searches.
- Ties between classes in a leaf go to the class seen first in the stream.
- NaN in X is replaced by the per-feature mean of the training buffer, which is stored and used
  again at prediction time. A feature that is entirely NaN is replaced by 0.
- A constant feature offers no threshold and is skipped.
- A class that first appears in a later chunk is registered at once; predict_proba gives it
  probability 0 until a retrain has seen it.
"""

import time

import numpy as np

from numcompute.pipeline import Estimator
from numcompute.sort_search import stable_sort

from ._validation import (NotFittedError, check_array, check_n_features, check_positive_int,
                          check_X_y)
from .preprocessing import LabelEncoder

# Largest (n_samples, features, n_classes) array (in float64 elements) the vectorised search builds
# at once; features are scored in blocks of at most this many elements. One such array is 12 MB,
# and the search holds about six of them at the same time (the one-hot rows in sorted order, their
# cumulative sums, the right-hand counts and the impurity temporaries). The measured peak was
# about 77 MB for 2,000 rows, 1,500 features and 3 classes. It is not a hard limit: when one
# feature alone needs more (n_samples * n_classes > 1.5 million), a block is that one feature.
_MAX_BLOCK_ELEMENTS = 1_500_000

# A split must beat the best gain so far by more than this, so rounding noise cannot change which
# of two equally good splits wins.
_GAIN_TOLERANCE = 1e-12


# ---------------------------------------------------------------------
# Impurity
# ---------------------------------------------------------------------

def gini_from_counts(counts):
    """
    Gini impurity 1 - sum(p^2) along the last axis of a class-count array.

    Parameters:
    ----------
    - counts: array-like of shape (..., n_classes), non-negative class counts

    Returns:
    -------
    - impurity: np.ndarray of shape (...), or a float for a 1-D input

    Complexity:
    ----------
    - Time and space complexity: O(size of counts)

    Note:
    ----
    - An empty node (all counts 0) has impurity 0, not NaN.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, n, out=np.zeros_like(counts), where=n > 0)
    return np.where(n[..., 0] > 0, 1.0 - (p ** 2).sum(axis=-1), 0.0)


def entropy_from_counts(counts):
    """
    Entropy -sum(p * log2(p)) in bits along the last axis of a class-count array.

    Parameters:
    ----------
    - counts: array-like of shape (..., n_classes), non-negative class counts

    Returns:
    -------
    - entropy: np.ndarray of shape (...), or a float for a 1-D input

    Complexity:
    ----------
    - Time and space complexity: O(size of counts)

    Note:
    ----
    - 0 * log2(0) is taken as 0, and an empty node has entropy 0.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, n, out=np.zeros_like(counts), where=n > 0)
    log_p = np.log2(p, out=np.zeros_like(p), where=p > 0)
    return -(p * log_p).sum(axis=-1)


CRITERIA = {"gini": gini_from_counts, "entropy": entropy_from_counts}


# ---------------------------------------------------------------------
# Split search
# ---------------------------------------------------------------------

def _midpoint(lo, hi):
    """Threshold between two distinct neighbours lo < hi, always satisfying lo <= t < hi."""
    t = lo / 2.0 + hi / 2.0               # halving first: hi - lo overflows for lo=-1e308, hi=1e308
    return t if lo <= t < hi else lo      # adjacent floats: the midpoint can round up to hi


def _observed_means(X, observed):
    """
    Mean of the non-NaN values of each column (0 where a column is all NaN), without overflow.

    Parameters:
    ----------
    - X: np.ndarray of shape (n_samples, n_features), finite or NaN
    - observed: np.ndarray of bool, the same shape, True where X is not NaN

    Returns:
    -------
    - np.ndarray of shape (n_features,)

    Note:
    ----
    - The plain sum / count is used first, so ordinary data gives the same bits as before. A
    column whose sum overflows (values near 1e308) is redone as the sum of value / count, which
    never exceeds the largest value.
    """
    counts = observed.sum(axis=0)
    values = np.where(observed, X, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        means = np.divide(values.sum(axis=0), counts, out=np.zeros(X.shape[1]), where=counts > 0)
    bad = ~np.isfinite(means)
    if bad.any():
        means[bad] = (values[:, bad] / counts[bad]).sum(axis=0)
    return means


def best_split_vectorised(X, y_codes, n_classes, criterion="gini", min_samples_leaf=1):
    """
    Find the best split "x[:, f] <= t" by scoring every candidate at once.

    Parameters:
    ----------
    - X: np.ndarray of shape (n_samples, n_features), without NaN
    - y_codes: np.ndarray of int, shape (n_samples,), values in [0, n_classes)
    - n_classes: int
    - criterion: 'gini' or 'entropy'
    - min_samples_leaf: int, each side of the split must keep at least this many rows

    Returns:
    -------
    - (feature, threshold, gain): tuple, or None when no split lowers the impurity

    Complexity:
    ----------
    - Time complexity: O(d * n log n + d * n * k) for n samples, d features and k classes
    - Space complexity: O(n * k) per feature, in blocks of at most _MAX_BLOCK_ELEMENTS values
    (12 MB per array; about 77 MB peak measured for 2,000 rows, 1,500 features, 3 classes)

    Note:
    ----
    - Ties are broken by the lowest feature, then the lowest threshold, because argmax returns
    the first maximum of a feature-major layout.
    - The cumulative sums are exact (they count integers), so the scores match the loop search
    to the last bit.
    """
    impurity = CRITERIA[criterion]
    n, d = X.shape
    if n < 2:
        return None
    onehot = np.eye(n_classes)[y_codes]                        # (n, k)
    total = onehot.sum(axis=0)                                 # (k,)
    parent = impurity(total)
    n_left = np.arange(1, n, dtype=np.float64)[:, None]        # (n-1, 1): rows left of cut i
    n_right = n - n_left

    best = None
    best_gain = 0.0
    block = max(1, _MAX_BLOCK_ELEMENTS // max(1, n * n_classes))
    for start in range(0, d, block):                           # a single pass for ordinary data
        block_X = X[:, start:start + block]
        xs, order = stable_sort(block_X, axis=0, return_indices=True)    # both (n, b)
        left = np.cumsum(onehot[order], axis=0)[:-1]           # (n-1, b, k): counts left of cut
        right = total - left
        child = (n_left * impurity(left) + n_right * impurity(right)) / n
        gain = parent - child                                  # (n-1, b)

        valid = xs[1:] > xs[:-1]                               # cut only between distinct values
        valid &= (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)
        gain = np.where(valid, gain, -np.inf)

        flat = int(np.argmax(gain.T))                          # (b, n-1): lowest feature first
        f_local, i = divmod(flat, n - 1)
        g = gain[i, f_local]
        if g > best_gain + _GAIN_TOLERANCE:
            best = (start + f_local, float(_midpoint(xs[i, f_local], xs[i + 1, f_local])),
                    float(g))
            best_gain = g
    return best


def best_split_loop(X, y_codes, n_classes, criterion="gini", min_samples_leaf=1):
    """
    The same search as best_split_vectorised, written with Python loops.

    Parameters:
    ----------
    - X, y_codes, n_classes, criterion, min_samples_leaf: see best_split_vectorised

    Returns:
    -------
    - (feature, threshold, gain): tuple, or None when no split lowers the impurity

    Complexity:
    ----------
    - Time complexity: O(d * n^2), because every threshold gets a fresh mask over all rows
    - Space complexity: O(n)

    Note:
    ----
    - Kept as the loop baseline of the benchmark and as an independent check in the tests.
    """
    impurity = CRITERIA[criterion]
    n, d = X.shape
    if n < 2:
        return None
    parent = impurity(np.bincount(y_codes, minlength=n_classes))
    best, best_gain = None, 0.0
    for f in range(d):
        values = np.unique(X[:, f])                            # sorted distinct values
        for k in range(len(values) - 1):
            t = _midpoint(values[k], values[k + 1])
            mask = X[:, f] <= t
            n_l = int(mask.sum())
            n_r = n - n_l
            if n_l < min_samples_leaf or n_r < min_samples_leaf:
                continue
            left = np.bincount(y_codes[mask], minlength=n_classes)
            right = np.bincount(y_codes[~mask], minlength=n_classes)
            g = parent - (n_l * impurity(left) + n_r * impurity(right)) / n
            if g > best_gain + _GAIN_TOLERANCE:
                best, best_gain = (f, float(t), float(g)), g
    return best


SPLITTERS = {"vectorised": best_split_vectorised, "loop": best_split_loop}


# ---------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------

class Node:
    """
    One node of the tree: a question with two children, or a leaf holding class counts.

    Attributes:
    ----------
    - counts: np.ndarray of shape (n_classes,), training samples of each class that reached it
    - impurity: float
    - depth: int
    - feature, threshold: the question "x[feature] <= threshold", None on a leaf
    - left, right: Node or None, rows answering yes go left and the others go right
    """

    __slots__ = ("feature", "threshold", "left", "right", "counts", "impurity", "depth")

    def __init__(self, counts, impurity, depth, feature=None, threshold=None,
                 left=None, right=None):
        self.counts = counts
        self.impurity = impurity
        self.depth = depth
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        """bool, True when the node has no children."""
        return self.left is None

    @property
    def n_samples(self):
        """int, number of training samples that reached the node."""
        return int(self.counts.sum())


class DecisionTree(Estimator):
    """
    Classification tree with buffered retraining for streaming data.

    Parameters:
    ----------
    - max_depth: int >= 0 or None, maximum depth; None grows until leaves are pure or too small
    (default 5)
    - min_samples_split: int >= 2, a node with fewer rows becomes a leaf (default 2)
    - min_samples_leaf: int >= 1, every leaf keeps at least this many training rows (default 1)
    - criterion: 'gini' or 'entropy' (default 'gini')
    - buffer_size: int >= 1, partial_fit retrains once the buffer holds this many samples
    (default 500)
    - splitter: 'vectorised' or 'loop', the split search to use; 'loop' exists for benchmarking
    (default 'vectorised')
    - min_impurity_decrease: float >= 0, a split must lower the impurity by more than this
    (default 0)
    - max_features: None, 'sqrt', 'log2', an int >= 1 or a float in (0, 1], how many features are
    tried at each split; None tries all of them (default None). The features are drawn at random
    for every node, so use random_state for a repeatable tree.
    - random_state: int or None, seed of the feature draw (default None); only used when
    max_features is not None. fit and reset restart the sequence from the seed.

    Attributes:
    ----------
    - classes_: np.ndarray, every class label seen so far, in order of first appearance
    - n_features_in_: int or None, the number of features
    - root_: Node or None, the most recently trained tree (None before the first retrain)
    - n_retrains_: int
    - n_buffered_: int, samples waiting in the buffer
    - retrain_log_: list of dict, one per retrain with keys 'retrain', 'n_samples', 'seconds',
    'depth', 'n_leaves' and 'train_accuracy'
    - feature_importances_: np.ndarray of shape (n_features,), the share of the total weighted
    gain that each feature produced in the current tree (all zeros for a single leaf)
    - fill_values_: np.ndarray of shape (n_features,), the replacement for NaN

    Complexity:
    ----------
    - Time complexity: a retrain on n samples with d features and depth h costs about
    O(h * d * n log n) with the vectorised search; predict is O(n * h) with the rows routed in
    groups; partial_fit between retrains is O(chunk)
    - Space complexity: O(buffer_size * d) for the buffer, O(nodes) for the tree

    Note:
    ----
    - Subclass of numcompute.pipeline.Estimator, so it can be the last step of a numcompute
    Pipeline: fit trains one tree on everything, predict uses the current tree.
    - update is a synonym of partial_fit.
    - Call flush() at the end of a stream to train on the rows still in the buffer.

    Examples:
    --------
    >>> tree = DecisionTree(max_depth=3, buffer_size=4)
    >>> _ = tree.partial_fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
    >>> tree.predict([[0.4], [2.6]]).tolist()
    [0, 1]
    """

    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1,
                 criterion="gini", buffer_size=500, splitter="vectorised",
                 min_impurity_decrease=0.0, max_features=None, random_state=None):
        super().__init__()
        self._check_max_features(max_features)
        if random_state is not None:
            check_positive_int(random_state, "random_state", minimum=0)
        if max_depth is not None:
            check_positive_int(max_depth, "max_depth", minimum=0)
        check_positive_int(min_samples_split, "min_samples_split", minimum=2)
        check_positive_int(min_samples_leaf, "min_samples_leaf", minimum=1)
        check_positive_int(buffer_size, "buffer_size", minimum=1)
        if criterion not in CRITERIA:
            raise ValueError("criterion must be one of {}. Got: {!r}"
                             .format(sorted(CRITERIA), criterion))
        if splitter not in SPLITTERS:
            raise ValueError("splitter must be one of {}. Got: {!r}"
                             .format(sorted(SPLITTERS), splitter))
        if isinstance(min_impurity_decrease, bool) or not np.isfinite(min_impurity_decrease) \
                or min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease must be a finite number >= 0. Got: {!r}"
                             .format(min_impurity_decrease))
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.buffer_size = buffer_size
        self.splitter = splitter
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.max_features = max_features
        self.random_state = random_state
        self.reset()

    @staticmethod
    def _check_max_features(max_features):
        """Raise ValueError unless max_features is None, 'sqrt', 'log2', an int or a float."""
        if max_features is None or max_features in ("sqrt", "log2"):
            return
        if isinstance(max_features, bool):
            raise ValueError("max_features must not be a bool. Got: {!r}".format(max_features))
        if isinstance(max_features, (int, np.integer)):
            if max_features >= 1:
                return
        elif isinstance(max_features, (float, np.floating)):
            if 0.0 < max_features <= 1.0:
                return
        raise ValueError("max_features must be None, 'sqrt', 'log2', an int >= 1 or a float in "
                         "(0, 1]. Got: {!r}".format(max_features))

    def _n_split_features(self, n_features):
        """How many features to try at one split (between 1 and n_features)."""
        mf = self.max_features
        if mf is None:
            return n_features
        if mf == "sqrt":
            k = int(np.sqrt(n_features))
        elif mf == "log2":
            k = int(np.log2(n_features))
        elif isinstance(mf, (int, np.integer)):
            k = int(mf)
        else:
            k = int(mf * n_features)
        return min(n_features, max(1, k))

    # ------------------------------------------------------------------ state
    def reset(self):
        """
        Forget the tree, the buffer, the classes and the log.

        Returns:
        -------
        - self
        """
        self._encoder = LabelEncoder()
        self._rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = None
        self.root_ = None
        self.fill_values_ = None
        self.feature_importances_ = None
        self.n_retrains_ = 0
        self.retrain_log_ = []
        self.n_buffered_ = 0
        self._buffer_X = []
        self._buffer_y = []
        return self

    @property
    def classes_(self):
        """np.ndarray, every class label seen so far, in order of first appearance."""
        return self._encoder.classes_

    @property
    def is_fitted(self):
        """bool, True once a tree has been trained."""
        return self.root_ is not None

    # ------------------------------------------------------------------ training
    def partial_fit(self, X, y):
        """
        Add a chunk to the buffer, and retrain if the buffer has reached buffer_size.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features), NaN allowed
        - y: array-like of shape (n_samples,), class labels

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if X is not 2-dimensional, contains infinity, has a different number of
        features from earlier chunks, or y does not match X or contains NaN
        - TypeError: if X or y is None

        Complexity:
        ----------
        - Time complexity: O(chunk) while the buffer fills; one retrain when it is full
        - Space complexity: O(buffer_size * d)

        Note:
        ----
        - An empty chunk changes nothing.
        - The buffer holds its own copy of the chunk, so changing the caller's arrays afterwards
        (a reused read buffer, say) cannot change what the next tree is trained on.
        - The buffer is cleared after every retrain, so each tree is trained on the rows that
        arrived since the previous one (a window), not on the whole stream.
        """
        X, y = check_X_y(X, y)
        if X.shape[0] == 0:
            return self
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        check_n_features(X, self.n_features_in_)
        self._encoder.partial_fit(y)
        self._buffer_X.append(X.copy())
        self._buffer_y.append(y.copy())
        self.n_buffered_ += X.shape[0]
        if self.n_buffered_ >= self.buffer_size:
            self._retrain_from_buffer()
        return self

    update = partial_fit

    def flush(self):
        """
        Train on whatever is left in the buffer (call it at the end of a stream).

        Returns:
        -------
        - self

        Note:
        ----
        - Does nothing when the buffer is empty.
        """
        if self.n_buffered_ > 0:
            self._retrain_from_buffer()
        return self

    def fit(self, X, y=None):
        """
        Forget everything, then train one tree on all of X (batch training).

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: array-like of shape (n_samples,), class labels

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if X is empty, or y is missing or does not match X
        """
        if y is None:
            raise ValueError("y is required to fit a DecisionTree.")
        self.reset()
        X, y = check_X_y(X, y)
        if X.shape[0] == 0:
            raise ValueError("Cannot fit DecisionTree on an empty array.")
        self.n_features_in_ = X.shape[1]
        self._encoder.partial_fit(y)
        self._train(X, y)
        return self

    def _retrain_from_buffer(self):
        X = np.vstack(self._buffer_X)
        y = np.concatenate(self._buffer_y)
        self._buffer_X, self._buffer_y = [], []
        self.n_buffered_ = 0
        self._train(X, y)

    def _train(self, X, y):
        start = time.perf_counter()
        observed = ~np.isnan(X)
        # mean of the observed values of each feature; 0 where the whole column is NaN
        self.fill_values_ = _observed_means(X, observed)
        X = np.where(observed, X, self.fill_values_)

        codes = self._encoder.transform(y)
        self._n_classes_trained = len(self.classes_)
        self._importance = np.zeros(X.shape[1])
        self.root_ = self._build(X, codes, depth=0)

        total = self._importance.sum()
        self.feature_importances_ = self._importance / total if total > 0 else self._importance
        seconds = time.perf_counter() - start

        self.n_retrains_ += 1
        self.retrain_log_.append({
            "retrain": self.n_retrains_,
            "n_samples": int(X.shape[0]),
            "seconds": seconds,
            "depth": self.get_depth(),
            "n_leaves": self.get_n_leaves(),
            "train_accuracy": float(np.mean(self.predict(X) == y)),
        })

    def _build(self, X, codes, depth):
        """Grow the tree on X recursively and return the root of this subtree."""
        k = self._n_classes_trained
        counts = np.bincount(codes, minlength=k).astype(np.float64)
        impurity = float(CRITERIA[self.criterion](counts))
        node = Node(counts, impurity, depth)
        n = X.shape[0]
        if ((self.max_depth is not None and depth >= self.max_depth)
                or n < self.min_samples_split
                or n < 2 * self.min_samples_leaf
                or impurity <= _GAIN_TOLERANCE):                  # pure node
            return node
        n_try = self._n_split_features(X.shape[1])
        if n_try < X.shape[1]:
            # a random subset of the features, in index order so ties still go to the lowest
            subset = np.sort(self._rng.choice(X.shape[1], size=n_try, replace=False))
            split = SPLITTERS[self.splitter](X[:, subset], codes, k, self.criterion,
                                             self.min_samples_leaf)
            if split is not None:
                split = (int(subset[split[0]]), split[1], split[2])
        else:
            split = SPLITTERS[self.splitter](X, codes, k, self.criterion, self.min_samples_leaf)
        if split is None or split[2] <= self.min_impurity_decrease:
            return node                                           # no question helps
        feature, threshold, gain = split
        self._importance[feature] += gain * n
        go_left = X[:, feature] <= threshold
        node.feature, node.threshold = feature, threshold
        node.left = self._build(X[go_left], codes[go_left], depth + 1)
        node.right = self._build(X[~go_left], codes[~go_left], depth + 1)
        return node

    # ------------------------------------------------------------------ prediction
    def _prepare(self, X):
        if self.root_ is None:
            raise NotFittedError(
                "fit must be called before predict. No tree has been trained yet: the buffer "
                "holds {}/{} samples. Send more data with partial_fit() or call flush()."
                .format(self.n_buffered_, self.buffer_size))
        X = check_array(X)
        check_n_features(X, self.n_features_in_)
        return np.where(np.isnan(X), self.fill_values_, X)

    def predict_proba(self, X):
        """
        Class probabilities from the current tree.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - proba: np.ndarray of shape (n_samples, n_classes), columns in classes_ order, each row
        summing to 1 (a class the tree has not seen yet gets probability 0)

        Raises:
        ------
        - RuntimeError: (NotFittedError) if no tree has been trained yet
        - ValueError: if X is not 2-dimensional or has a different number of features

        Complexity:
        ----------
        - Time complexity: O(n * h) for n rows and tree depth h
        - Space complexity: O(n * k)
        """
        X = self._prepare(X)
        out = np.zeros((X.shape[0], len(self.classes_)))
        self._route(self.root_, X, np.arange(X.shape[0]), out)
        return out

    def _route(self, node, X, rows, out):
        """Send the group of rows down from node: one comparison per node, not per row."""
        if rows.size == 0:
            return
        if node.is_leaf:
            out[rows, :node.counts.size] = node.counts / node.counts.sum()
            return
        go_left = X[rows, node.feature] <= node.threshold
        self._route(node.left, X, rows[go_left], out)
        self._route(node.right, X, rows[~go_left], out)

    def predict(self, X):
        """
        Predicted class label of each row.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - y_pred: np.ndarray of shape (n_samples,), values from classes_

        Raises:
        ------
        - RuntimeError: (NotFittedError) if no tree has been trained yet
        - ValueError: if X is not 2-dimensional or has a different number of features

        Note:
        ----
        - A tie between classes in a leaf goes to the class seen first in the stream.
        """
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_loop(self, X):
        """
        Row-by-row prediction (the Week 10 version), kept as the loop baseline of the benchmark.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - y_pred: np.ndarray of shape (n_samples,), the same values as predict
        """
        X = self._prepare(X)
        out = np.empty(X.shape[0], dtype=np.int64)
        for i, x in enumerate(X):
            node = self.root_
            while not node.is_leaf:
                node = node.left if x[node.feature] <= node.threshold else node.right
            out[i] = np.argmax(node.counts)
        return self.classes_[out]

    def score(self, X, y):
        """
        Accuracy of predict(X) against y.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: array-like of shape (n_samples,)

        Returns:
        -------
        - float in [0, 1]
        """
        X, y = check_X_y(X, y, min_samples=1)
        return float(np.mean(self.predict(X) == y))

    # ------------------------------------------------------------------ inspection
    def get_depth(self):
        """
        Depth of the current tree.

        Returns:
        -------
        - int, 0 for a single leaf or before the first retrain
        """
        def depth(node):
            return 0 if node.is_leaf else 1 + max(depth(node.left), depth(node.right))
        return 0 if self.root_ is None else depth(self.root_)

    def get_n_leaves(self):
        """
        Number of leaves in the current tree.

        Returns:
        -------
        - int, 0 before the first retrain
        """
        def leaves(node):
            return 1 if node.is_leaf else leaves(node.left) + leaves(node.right)
        return 0 if self.root_ is None else leaves(self.root_)

    def export_text(self, feature_names=None, decimals=3):
        """
        Readable drawing of the current tree, one node per line.

        Parameters:
        ----------
        - feature_names: list of str of length n_features, default x0, x1, ...
        - decimals: int, digits kept in the thresholds (default 3)

        Returns:
        -------
        - text: str

        Raises:
        ------
        - RuntimeError: (NotFittedError) if no tree has been trained yet
        - ValueError: if feature_names has the wrong length
        """
        if self.root_ is None:
            raise NotFittedError("fit must be called before export_text. No tree trained yet.")
        names = feature_names or ["x{}".format(i) for i in range(self.n_features_in_)]
        if len(names) != self.n_features_in_:
            raise ValueError("feature_names needs {} names. Got: {}"
                             .format(self.n_features_in_, len(names)))
        lines = []

        def walk(node, indent):
            pad = "|   " * indent
            if node.is_leaf:
                label = self.classes_[int(np.argmax(node.counts))]
                lines.append("{}|--- class: {}  (n={})".format(pad, label, node.n_samples))
                return
            t = round(node.threshold, decimals)
            lines.append("{}|--- {} <= {}".format(pad, names[node.feature], t))
            walk(node.left, indent + 1)
            lines.append("{}|--- {} >  {}".format(pad, names[node.feature], t))
            walk(node.right, indent + 1)

        walk(self.root_, 0)
        return "\n".join(lines)

    def memory_bytes(self):
        """
        Approximate memory held by the model, in bytes.

        Returns:
        -------
        - int, the size of the buffered chunks plus the class counts stored in the tree nodes

        Note:
        ----
        - It counts the NumPy arrays only (the buffer and the nodes' counts), not Python object
        overhead, so it is a lower bound that grows and shrinks with the buffer.
        """
        total = sum(a.nbytes for a in self._buffer_X) + sum(a.nbytes for a in self._buffer_y)
        stack = [self.root_] if self.root_ is not None else []
        while stack:
            node = stack.pop()
            total += node.counts.nbytes
            if not node.is_leaf:
                stack.extend((node.left, node.right))
        return int(total)

    def __repr__(self):
        return ("DecisionTree(max_depth={}, criterion={!r}, buffer_size={}, max_features={!r}, "
                "retrains={})".format(self.max_depth, self.criterion, self.buffer_size,
                                      self.max_features, self.n_retrains_))


# The name the assignment brief uses.
DecisionTreeClassifier = DecisionTree
