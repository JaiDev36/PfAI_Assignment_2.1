"""
pipeline.py

Chain streaming transformers and a model, and run test-then-train ("prequential") evaluation.

StreamingPipeline is a numcompute.pipeline.Pipeline with one extra method, partial_fit. For each
chunk, every transformer first learns from the chunk (partial_fit) and then transforms it, and the
result goes to the next step. The last step is the model.

Design note: transformer parameters keep moving as chunks arrive, so rows waiting in the tree's
buffer may have been transformed with slightly different parameters from later rows. A running-mean
imputer is barely affected. A tree needs no scaling at all (a split x <= t only depends on the order
of the values), so the usual pipeline is an imputer followed by the tree.
"""

import logging
import time

from numcompute.pipeline import Pipeline

from ._validation import NotFittedError, check_X_y
from .metrics import StreamingMetrics

logger = logging.getLogger("numcompute_stream")


class StreamingPipeline(Pipeline):
    """
    A numcompute Pipeline that can also learn one chunk at a time.

    Parameters:
    ----------
    - steps: list of (name, object) tuples. Every transformer needs partial_fit(X) and
    transform(X); the final step needs partial_fit(X, y) and predict(X). The rules of
    numcompute.pipeline.Pipeline also apply (they are checked by its constructor).

    Attributes:
    ----------
    - steps: the list of (name, object) tuples
    - named_steps: dict of name to object
    - model: the final step

    Raises:
    ------
    - TypeError: if a transformer has no partial_fit, or the final step has no partial_fit
    - ValueError: if the step names are not unique

    Complexity:
    ----------
    - Time complexity: the sum over the steps
    - Space complexity: the sum over the steps

    Note:
    ----
    - fit, transform and predict come from Pipeline unchanged, so batch use still works.
    - update is a synonym of partial_fit.
    """

    def __init__(self, steps):
        names = [name for name, _ in steps]
        if len(set(names)) != len(names):
            raise ValueError("Step names must be unique. Got: {}".format(names))
        super().__init__(steps)
        for name, step in steps:
            if not hasattr(step, "partial_fit"):
                raise TypeError("Pipeline step {!r} must implement partial_fit".format(name))

    @property
    def named_steps(self):
        """dict, step name to step object."""
        return dict(self.steps)

    @property
    def model(self):
        """The final step of the pipeline."""
        return self.steps[-1][1]

    def partial_fit(self, X, y):
        """
        Let every transformer learn from one chunk, then train the model on the transformed chunk.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: array-like of shape (n_samples,)

        Returns:
        -------
        - self

        Raises:
        ------
        - ValueError: if X and y do not match, or a step rejects the chunk
        """
        X, y = check_X_y(X, y)
        for _, step in self.steps[:-1]:
            step.partial_fit(X, y)
            X = step.transform(X)
        self.model.partial_fit(X, y)
        return self

    update = partial_fit

    def flush(self):
        """
        Ask the model to train on the rows left in its buffer, if it has a buffer.

        Returns:
        -------
        - self
        """
        if hasattr(self.model, "flush"):
            self.model.flush()
        return self

    def predict_proba(self, X):
        """
        Transform X with the parameters learned so far, then return the model's probabilities.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)

        Returns:
        -------
        - proba: np.ndarray of shape (n_samples, n_classes)

        Raises:
        ------
        - TypeError: if the model has no predict_proba
        """
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("Final pipeline step {!r} must implement predict_proba"
                            .format(self.steps[-1][0]))
        for _, step in self.steps[:-1]:
            X = step.transform(X)
        return self.model.predict_proba(X)


def prequential_run(model, chunks, metrics=None, log_every=1):
    """
    Run a stream with test-then-train evaluation.

    For every chunk: if the model can already predict, it predicts the chunk first and the score is
    recorded (the model has never seen these rows, so the score is out-of-sample); then the model
    trains on the chunk. Chunks that arrive before the first tree is trained are trained on but not
    scored.

    Parameters:
    ----------
    - model: DecisionTree or StreamingPipeline
    - chunks: iterable of (X_chunk, y_chunk) pairs, for example from io.iter_chunks
    - metrics: StreamingMetrics, optional; pass one to keep adding to an existing history
    - log_every: int >= 0, log one INFO line every log_every chunks (0 turns logging off)

    Returns:
    -------
    - metrics: StreamingMetrics. For every scored chunk its history also holds 'retrains' (the
    cumulative number), 'partial_fit_seconds', 'stream_chunk' (chunk number in the stream) and
    'rows_read' (position in the stream after the chunk).

    Raises:
    ------
    - ValueError: if a chunk is malformed (see DecisionTree.partial_fit)

    Complexity:
    ----------
    - Time complexity: one predict and one partial_fit per chunk
    - Space complexity: O(buffer_size * d) plus the history

    Note:
    ----
    - Before the first tree exists the model cannot predict. A DecisionTree raises NotFittedError;
    a pipeline whose scaler has not seen data raises numcompute's plain RuntimeError. Both mean
    "not ready yet" and are caught, but only while no tree has been trained: once a tree exists,
    a RuntimeError is a real error and is raised.
    """
    metrics = metrics if metrics is not None else StreamingMetrics()
    tree = model.model if isinstance(model, StreamingPipeline) else model
    rows_read = 0
    for i, (X_chunk, y_chunk) in enumerate(chunks, start=1):
        rows_read += len(y_chunk)
        y_pred = None
        try:
            y_pred = model.predict(X_chunk)
        except NotFittedError:
            pass
        except RuntimeError:
            if tree.is_fitted:
                raise
        start = time.perf_counter()
        model.partial_fit(X_chunk, y_chunk)
        seconds = time.perf_counter() - start
        if y_pred is not None:
            entry = metrics.update(y_chunk, y_pred, retrains=tree.n_retrains_,
                                   partial_fit_seconds=seconds, stream_chunk=i,
                                   rows_read=rows_read).last_entry_
            if log_every and i % log_every == 0:
                logger.info("chunk %3d | rows_read=%6d | chunk_acc=%.3f | cum_acc=%.3f "
                            "| retrains=%d", i, rows_read, entry["chunk_accuracy"],
                            entry["cumulative_accuracy"], tree.n_retrains_)
        elif log_every:
            if tree.n_retrains_:
                logger.info("chunk %3d | first tree trained on %d buffered rows (not scored)",
                            i, tree.retrain_log_[0]["n_samples"])
            else:
                logger.info("chunk %3d | buffering (%d/%d), no tree yet",
                            i, tree.n_buffered_, tree.buffer_size)
    return metrics


# The name the assignment brief uses.
Pipeline = StreamingPipeline
