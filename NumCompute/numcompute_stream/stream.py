"""
stream.py

StreamTrainer: one object that manages a model (or pipeline), the metrics and a per-chunk log.

fit_chunk trains on a chunk and logs how long it took and how much memory the model holds.
score_chunk predicts a chunk and adds the result to the running metrics. Calling score_chunk and
then fit_chunk on every chunk is test-then-train ("prequential") evaluation, which is what run()
does. pipeline.prequential_run is the same loop as one function.
"""

import logging
import time

import numpy as np

from ._validation import NotFittedError, check_X_y
from .metrics import StreamingMetrics
from .pipeline import StreamingPipeline

logger = logging.getLogger("numcompute_stream")


class StreamTrainer:
    """
    Train and score a streaming model one chunk at a time, and keep a log.

    Parameters:
    ----------
    - model: DecisionTree or StreamingPipeline (anything with partial_fit and predict)
    - metrics: StreamingMetrics, optional; a new one is made if not given
    - log_every: int >= 0, write one INFO line every log_every chunks (0 turns logging off)

    Attributes:
    ----------
    - model: the model being trained
    - metrics: the StreamingMetrics that score_chunk adds to
    - log_: list of dict, one per fit_chunk call with keys 'chunk', 'n_rows', 'seconds',
    'memory_bytes', 'retrains' and 'cumulative_accuracy' (NaN until a chunk has been scored)
    - n_chunks_: int, the number of chunks given to fit_chunk
    - cumulative_accuracy: float, accuracy over every scored chunk so far

    Raises:
    ------
    - TypeError: if model has no partial_fit or no predict
    - ValueError: if log_every is not an int >= 0

    Complexity:
    ----------
    - Time complexity: one partial_fit (fit_chunk) or one predict (score_chunk) per chunk
    - Space complexity: the model plus O(chunks) for the log

    Note:
    ----
    - The tree is the last step of a pipeline, so the memory figure is the tree's buffer and
    nodes (DecisionTree.memory_bytes), and 0 for a model that does not report it.

    Examples:
    --------
    >>> from numcompute_stream import DecisionTree
    >>> trainer = StreamTrainer(DecisionTree(max_depth=2, buffer_size=4), log_every=0)
    >>> _ = trainer.fit_chunk([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
    >>> trainer.score_chunk([[0.2], [2.8]], [0, 1])
    1.0
    """

    def __init__(self, model, metrics=None, log_every=1):
        for method in ("partial_fit", "predict"):
            if not hasattr(model, method):
                raise TypeError("model must implement {}. Got: {!r}".format(method, model))
        if isinstance(log_every, bool) or not isinstance(log_every, (int, np.integer)) \
                or log_every < 0:
            raise ValueError("log_every must be an int >= 0. Got: {!r}".format(log_every))
        self.model = model
        self.metrics = metrics if metrics is not None else StreamingMetrics()
        self.log_every = int(log_every)
        self.log_ = []
        self.n_chunks_ = 0

    @property
    def _tree(self):
        return self.model.model if isinstance(self.model, StreamingPipeline) else self.model

    @property
    def cumulative_accuracy(self):
        """float, accuracy over every scored chunk (raises NotFittedError before the first)."""
        return self.metrics.accuracy

    def fit_chunk(self, X, y):
        """
        Train the model on one chunk and log it.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: array-like of shape (n_samples,)

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if X and y do not match, or the model rejects the chunk

        Note:
        ----
        - An empty chunk changes nothing: the model is not called and no log entry is added.
        """
        X, y = check_X_y(X, y)
        if X.shape[0] == 0:
            return self
        start = time.perf_counter()
        self.model.partial_fit(X, y)
        seconds = time.perf_counter() - start
        self.n_chunks_ += 1
        tree = self._tree
        memory = tree.memory_bytes() if hasattr(tree, "memory_bytes") else 0
        accuracy = self.metrics.accuracy if self.metrics.confusion_.sum() else float("nan")
        entry = {"chunk": self.n_chunks_, "n_rows": int(X.shape[0]), "seconds": seconds,
                 "memory_bytes": memory, "retrains": getattr(tree, "n_retrains_", 0),
                 "cumulative_accuracy": accuracy}
        self.log_.append(entry)
        if self.log_every and self.n_chunks_ % self.log_every == 0:
            logger.info("chunk %3d | rows=%5d | fit=%.4fs | memory=%d B | retrains=%d | "
                        "cum_acc=%.3f", entry["chunk"], entry["n_rows"], seconds, memory,
                        entry["retrains"], accuracy)
        return self

    def score_chunk(self, X, y):
        """
        Predict one chunk with the current model and add the result to the metrics.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: array-like of shape (n_samples,), the true labels

        Returns:
        -------
        - float, the accuracy on this chunk, or None if the chunk is empty or the model cannot
        predict yet (no tree has been trained); nothing is recorded in either case

        Raises:
        ------
        - ValueError: if X and y do not match
        """
        X, y = check_X_y(X, y)
        if X.shape[0] == 0:                    # no rows: no accuracy, and the metrics are untouched
            return None
        try:
            y_pred = self.model.predict(X)
        except NotFittedError:
            return None
        except RuntimeError:
            # a pipeline whose scaler has seen no data raises a plain RuntimeError
            if getattr(self._tree, "is_fitted", False):
                raise
            return None
        self.metrics.update(y, y_pred, retrains=getattr(self._tree, "n_retrains_", 0))
        return self.metrics.last_entry_["chunk_accuracy"]

    def run(self, chunks):
        """
        Test-then-train over an iterable of (X_chunk, y_chunk) pairs.

        Parameters:
        ----------
        - chunks: iterable of (X_chunk, y_chunk), for example from io.iter_chunks

        Returns:
        -------
        - self

        Note:
        ----
        - An empty chunk is skipped: it is neither scored nor logged.
        """
        for X_chunk, y_chunk in chunks:
            self.score_chunk(X_chunk, y_chunk)
            self.fit_chunk(X_chunk, y_chunk)
        return self

    def reset(self):
        """
        Forget the log and the metrics (the model is left as it is).

        Returns:
        -------
        - self
        """
        self.metrics.reset()
        self.log_ = []
        self.n_chunks_ = 0
        return self
