"""
metrics.py

Classification metrics accumulated over a stream.

numcompute.metrics scores one batch of predictions. StreamingMetrics scores a stream: it keeps a
running confusion matrix, so the accuracy and macro-F1 over everything seen so far are exact
without storing old predictions, and it logs one history entry per chunk for plotting.

Typical use is prequential ("test, then train") evaluation: predict a chunk the model has not
seen, call update(), then train on the chunk.
"""

from collections import deque

import numpy as np

from numcompute.metrics import accuracy, confusion_matrix

from ._validation import NotFittedError, check_positive_int


def per_class_scores(cm):
    """
    Per-class precision, recall and F1 from a confusion matrix.

    Parameters:
    ----------
    - cm: array-like of shape (n_classes, n_classes), rows are true labels, columns predictions

    Returns:
    -------
    - precision, recall, f1: np.ndarray of shape (n_classes,)

    Raises:
    ------
    - ValueError: if cm is not square and 2-dimensional

    Complexity:
    ----------
    - Time and space complexity: O(k^2) for k classes

    Note:
    ----
    - Same convention as numcompute.metrics: a class that is never predicted has precision 0, a
    class that never occurs has recall 0, and F1 is 0 when precision + recall is 0.
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("cm must be a square 2-dimensional array. Got shape: {}".format(cm.shape))
    tp = np.diag(cm)
    predicted = cm.sum(axis=0)
    actual = cm.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


class StreamingMetrics:
    """
    Accumulate a confusion matrix over chunks and log a history for plotting.

    Attributes:
    ----------
    - last_entry_: dict or None, the history entry of the most recent update
    - labels_: np.ndarray, every label seen so far in order of first appearance; it grows when a
    new class shows up
    - confusion_: np.ndarray of int, shape (n_labels, n_labels), the cumulative confusion matrix
    in labels_ order (rows are true labels)
    - history: dict of lists, one entry per update with keys 'chunk', 'n_seen', 'chunk_accuracy',
    'cumulative_accuracy', 'error' (1 - cumulative accuracy) and 'macro_f1', plus any keys passed
    through update's keyword arguments

    Complexity:
    ----------
    - Time complexity: update is O(n log k + k^2) for a chunk of n samples and k classes
    - Space complexity: O(k^2) for the matrix plus O(chunks) for the history

    Note:
    ----
    - Accuracy uses numcompute.metrics.accuracy and the counts use numcompute.metrics
    .confusion_matrix, so the two libraries cannot disagree.
    - An existing count never moves when a class is added: the matrix is padded with zeros.
    """

    def __init__(self, window=None):
        if window is not None:
            check_positive_int(window, "window", minimum=1)
        self.window = window
        self.reset()

    def reset(self):
        """
        Forget every update.

        Returns:
        -------
        - self
        """
        self._recent = deque(maxlen=self.window)
        self.last_entry_ = None
        self.labels_ = np.array([])
        self.confusion_ = np.zeros((0, 0), dtype=np.int64)
        self.history = {"chunk": [], "n_seen": [], "chunk_accuracy": [],
                        "cumulative_accuracy": [], "error": [], "macro_f1": []}
        return self

    def _grow(self, labels):
        """Append labels not seen before, padding the confusion matrix with zeros."""
        known = set(self.labels_.tolist())
        fresh = [v for v in labels.tolist() if v not in known]
        if not fresh:
            return
        k_old = len(self.labels_)
        self.labels_ = np.array(self.labels_.tolist() + fresh)
        grown = np.zeros((len(self.labels_), len(self.labels_)), dtype=np.int64)
        grown[:k_old, :k_old] = self.confusion_
        self.confusion_ = grown

    def update(self, y_true, y_pred, **extra):
        """
        Add one chunk of predictions and record a history entry.

        Parameters:
        ----------
        - y_true: array-like of shape (n_samples,), the true labels
        - y_pred: array-like of shape (n_samples,), the predicted labels
        - extra: numbers to log with this chunk, for example retrain_time=0.02

        Returns:
        -------
        - self, like every other update; the entry just recorded is in last_entry_

        Raises:
        ------
        - ValueError: if the inputs are not 1-dimensional, differ in length or contain NaN (the
        checks of numcompute.metrics)

        Complexity:
        ----------
        - Time complexity: O(n log k + k^2)
        - Space complexity: O(n + k^2)

        Note:
        ----
        - Two empty inputs (a chunk with no rows) change nothing: no history entry is added and
        last_entry_ stays as it was. An empty array against a non-empty one is still an error.
        - With a window the chunk is stored as a copy, so changing the caller's arrays afterwards
        does not change the rolling scores.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.ndim == 1 and y_pred.ndim == 1 and y_true.size == 0 and y_pred.size == 0:
            return self
        chunk_accuracy = accuracy(y_true, y_pred)        # validates both inputs first
        self._grow(np.unique(np.concatenate([y_true, y_pred])))
        self.confusion_ += confusion_matrix(y_true, y_pred, labels=self.labels_)

        total = int(self.confusion_.sum())
        cumulative = float(np.trace(self.confusion_) / total)
        entry = {
            "chunk": len(self.history["chunk"]) + 1,
            "n_seen": total,
            "chunk_accuracy": chunk_accuracy,
            "cumulative_accuracy": cumulative,
            "error": 1.0 - cumulative,
            "macro_f1": float(per_class_scores(self.confusion_)[2].mean()),
        }
        if self.window is not None:
            self._recent.append((y_true.copy(), y_pred.copy()))
            entry["rolling_accuracy"] = self._window_scores()["accuracy"]
        entry.update(extra)
        for key, value in entry.items():
            self.history.setdefault(key, []).append(value)
        self.last_entry_ = entry
        return self

    partial_fit = update

    @property
    def accuracy(self):
        """float, accuracy over every update so far."""
        total = self.confusion_.sum()
        if total == 0:
            raise NotFittedError("No predictions recorded yet; call update() first.")
        return float(np.trace(self.confusion_) / total)

    @property
    def macro_f1(self):
        """float, unweighted mean of the per-class F1 over every update so far."""
        if self.confusion_.sum() == 0:
            raise NotFittedError("No predictions recorded yet; call update() first.")
        return float(per_class_scores(self.confusion_)[2].mean())

    def _scores(self, cm):
        """Accuracy and macro precision, recall and F1 from a confusion matrix."""
        precision, recall, f1 = per_class_scores(cm)
        return {"accuracy": float(np.trace(cm) / cm.sum()), "precision": float(precision.mean()),
                "recall": float(recall.mean()), "f1": float(f1.mean())}

    def _window_scores(self):
        """The same scores over the latest `window` chunks only."""
        y_true = np.concatenate([t for t, _ in self._recent])
        y_pred = np.concatenate([p for _, p in self._recent])
        cm = confusion_matrix(y_true, y_pred, labels=self.labels_)
        return self._scores(cm.astype(np.float64))

    @property
    def precision(self):
        """float, macro precision (mean over classes) over every update so far."""
        return self.result()["precision"]

    @property
    def recall(self):
        """float, macro recall (mean over classes) over every update so far."""
        return self.result()["recall"]

    @property
    def f1(self):
        """float, macro F1 over every update so far (same as macro_f1)."""
        return self.result()["f1"]

    def result(self):
        """
        Every metric over the whole stream so far (and over the window, if there is one).

        Returns:
        -------
        - dict with keys 'n_seen', 'accuracy', 'precision', 'recall', 'f1' (precision, recall and
        F1 are macro averages over the classes) and 'confusion' (a copy of confusion_). With a
        window it also has 'rolling_accuracy', 'rolling_precision', 'rolling_recall' and
        'rolling_f1' over the latest `window` chunks.

        Raises:
        ------
        - NotFittedError: if nothing has been recorded yet
        """
        if self.confusion_.sum() == 0:
            raise NotFittedError("No predictions recorded yet; call update() first.")
        out = {"n_seen": int(self.confusion_.sum())}
        out.update(self._scores(self.confusion_.astype(np.float64)))
        out["confusion"] = self.confusion_.copy()
        if self.window is not None:
            for key, value in self._window_scores().items():
                out["rolling_" + key] = value
        return out

    def series(self, key="chunk_accuracy"):
        """
        Return one history column as a float array.

        Parameters:
        ----------
        - key: str, a key of history (default 'chunk_accuracy')

        Returns:
        -------
        - np.ndarray of shape (n_updates,)

        Raises:
        ------
        - KeyError: if key was never logged
        """
        return np.asarray(self.history[key], dtype=np.float64)


class StreamingAUC:
    """
    Binary ROC AUC accumulated over a stream, from score histograms.

    Parameters:
    ----------
    - n_bins: int >= 2, the number of equal-width score bins (default 1000)
    - score_range: (low, high), the range of the scores (default (0, 1), a probability)
    - pos_label: the label counted as positive (default 1)
    - window: int >= 1 or None, use only the latest `window` chunks (default None, all chunks)

    Attributes:
    ----------
    - n_pos_, n_neg_: int, the positive and negative samples currently counted

    Raises:
    ------
    - ValueError: if the parameters are invalid

    Complexity:
    ----------
    - Time complexity: update is O(n + bins) for n samples; result is O(bins)
    - Space complexity: O(bins), or O(window * bins) with a window

    Note:
    ----
    - AUC is the probability that a random positive scores above a random negative (ties count
    one half). A stream cannot keep every score, so each class keeps a histogram of its scores
    and the AUC is read from the two histograms. Scores that share a bin count as tied, so the
    value is an approximation: a positive and a negative that fall in the same bin count as
    half a win however their scores compare. It equals numcompute.metrics.auc when no bin holds
    both a positive and a negative score, and it approaches it as n_bins grows.
    - update(y_true, y_score) takes the chunk's labels and the model's score for the positive
    class (for example DecisionTree.predict_proba(X)[:, 1]).

    Examples:
    --------
    >>> auc = StreamingAUC().update([0, 1], [0.2, 0.9]).update([0, 1], [0.4, 0.3])
    >>> round(auc.result(), 2)
    0.75
    """

    def __init__(self, n_bins=1000, score_range=(0.0, 1.0), pos_label=1, window=None):
        check_positive_int(n_bins, "n_bins", minimum=2)
        if window is not None:
            check_positive_int(window, "window", minimum=1)
        try:
            low, high = (float(v) for v in score_range)
        except (TypeError, ValueError):
            raise ValueError("score_range must be (low, high). Got: {!r}".format(score_range))
        if not (np.isfinite(low) and np.isfinite(high) and low < high):
            raise ValueError("score_range needs finite low < high. Got: {!r}".format(score_range))
        self.n_bins = n_bins
        self.score_range = (low, high)
        self.pos_label = pos_label
        self.window = window
        self.reset()

    def reset(self):
        """
        Forget every update.

        Returns:
        -------
        - self
        """
        self._pos = np.zeros(self.n_bins, dtype=np.int64)
        self._neg = np.zeros(self.n_bins, dtype=np.int64)
        self._chunks = deque()
        return self

    @property
    def n_pos_(self):
        return int(self._pos.sum())

    @property
    def n_neg_(self):
        return int(self._neg.sum())

    def update(self, y_true, y_score):
        """
        Add one chunk of labels and scores.

        Parameters:
        ----------
        - y_true: array-like of shape (n_samples,), the true labels
        - y_score: array-like of shape (n_samples,), the score of the positive class, inside
        score_range

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if the inputs are not 1-dimensional of the same length, or a score is NaN
        or outside score_range
        """
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score, dtype=np.float64)
        if y_true.ndim != 1 or y_score.ndim != 1:
            raise ValueError("y_true and y_score must be 1-dimensional. Got shapes: {} and {}"
                             .format(y_true.shape, y_score.shape))
        if y_true.shape != y_score.shape:
            raise ValueError("y_true and y_score must have the same length. Got: {} and {}"
                             .format(y_true.size, y_score.size))
        if y_true.size == 0:
            return self
        low, high = self.score_range
        if np.isnan(y_score).any() or (y_score < low).any() or (y_score > high).any():
            raise ValueError("y_score must be numbers inside score_range {}. Found NaN or "
                             "values outside it.".format(self.score_range))
        bins = np.minimum(((y_score - low) / (high - low) * self.n_bins).astype(np.intp),
                          self.n_bins - 1)
        positive = y_true == self.pos_label
        pos = np.bincount(bins[positive], minlength=self.n_bins)
        neg = np.bincount(bins[~positive], minlength=self.n_bins)
        self._pos += pos
        self._neg += neg
        if self.window is not None:
            self._chunks.append((pos, neg))
            while len(self._chunks) > self.window:
                old_pos, old_neg = self._chunks.popleft()
                self._pos -= old_pos
                self._neg -= old_neg
        return self

    partial_fit = update

    def result(self):
        """
        ROC AUC of every sample counted so far.

        Returns:
        -------
        - float in [0, 1]

        Raises:
        ------
        - NotFittedError: if there is no positive or no negative sample yet (AUC is undefined)
        """
        n_pos, n_neg = self.n_pos_, self.n_neg_
        if n_pos == 0 or n_neg == 0:
            raise NotFittedError("AUC needs at least one positive and one negative sample. "
                                 "Counted: {} positive, {} negative.".format(n_pos, n_neg))
        below = np.cumsum(self._neg) - self._neg          # negatives in strictly lower bins
        wins = (self._pos * (below + 0.5 * self._neg)).sum()
        return float(wins / (n_pos * n_neg))
