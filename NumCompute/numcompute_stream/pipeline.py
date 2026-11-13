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

Interoperability: a step does not have to come from this package or inherit from anything. y is
passed only to a fit or partial_fit that can take it, so a third-party transformer written as
partial_fit(X) / transform(X) works without an adapter. Every step must have partial_fit, because
that is how a stream is learned; an object with only fit(X) / transform(X) cannot learn one chunk
at a time and is rejected with a TypeError that says so.
"""

import inspect
import logging
import time

from numcompute.pipeline import Pipeline

from ._validation import NotFittedError, check_X_y
from .metrics import StreamingMetrics

logger = logging.getLogger("numcompute_stream")


def _accepts_y(method):
    """
    Tell whether a fit or partial_fit method can be given the labels.

    Parameters:
    ----------
    - method: a bound method, for example step.partial_fit

    Returns:
    -------
    - "positional" if it can be called as method(X, y), "keyword" if only as method(X, y=y),
    or None if it takes X alone

    Note:
    ----
    - A method whose signature cannot be read (some built-ins) is assumed to take y, which is
    what numcompute.pipeline.Pipeline always assumed.
    """
    try:
        params = list(inspect.signature(method).parameters.values())
    except (TypeError, ValueError):
        return "positional"
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if len(positional) >= 2 or any(p.kind == p.VAR_POSITIONAL for p in params):
        return "positional"
    if any(p.kind == p.VAR_KEYWORD or (p.kind == p.KEYWORD_ONLY and p.name == "y")
           for p in params):
        return "keyword"
    return None


def _call_with_optional_y(method, X, y):
    """
    Call method(X, y) if the method takes labels, otherwise method(X).

    Parameters:
    ----------
    - method: a bound fit or partial_fit
    - X: the data
    - y: the labels (may be None)

    Returns:
    -------
    - whatever the method returns
    """
    how = _accepts_y(method)
    if how == "positional":
        return method(X, y)
    if how == "keyword":
        return method(X, y=y)
    return method(X)


def _batch_fit(step, X, y, name="step"):
    """
    Train one step on all of X: fit if it has one, otherwise reset and partial_fit.

    Parameters:
    ----------
    - step: a pipeline step
    - X, y: the data and labels
    - name: str, the step's name, used in the error message

    Returns:
    -------
    - None

    Raises:
    ------
    - TypeError: if the step has neither fit nor reset. A batch fit must start from nothing, and
    calling partial_fit on a step that may already hold state would silently add to it.
    """
    if hasattr(step, "fit"):
        _call_with_optional_y(step.fit, X, y)
    elif hasattr(step, "reset"):
        step.reset()
        _call_with_optional_y(step.partial_fit, X, y)
    else:
        raise TypeError("Pipeline step {!r} has neither fit nor reset, so fit cannot start it "
                        "from scratch. Add one of them, or use partial_fit.".format(name))


class StreamingPipeline(Pipeline):
    """
    A numcompute Pipeline that can also learn one chunk at a time.

    Parameters:
    ----------
    - steps: list of (name, object) tuples. Every transformer needs partial_fit and transform;
    the final step needs partial_fit(X, y) and predict(X). fit is optional: fit(X, y) falls back
    to reset() and partial_fit for a step that has no fit, and raises TypeError for a step with
    neither.

    Attributes:
    ----------
    - steps: the list of (name, object) tuples
    - named_steps: dict of name to object
    - model: the final step

    Raises:
    ------
    - TypeError: if a step has no partial_fit (an object with only fit cannot learn a stream),
    or a transformer has no transform
    - ValueError: if the step names are not unique

    Complexity:
    ----------
    - Time complexity: the sum over the steps
    - Space complexity: the sum over the steps

    Note:
    ----
    - transform and predict come from Pipeline unchanged. fit and fit_transform are overridden
    only so that y goes to the steps that take it.
    - y reaches a step's fit or partial_fit only if its signature has room for it (a second
    positional parameter, *args, a keyword-only y or **kwargs); otherwise the step gets X alone.
    - An empty chunk changes nothing: no step sees it.
    - update is a synonym of partial_fit.
    """

    def __init__(self, steps):
        names = [name for name, _ in steps]
        if len(set(names)) != len(names):
            raise ValueError("Step names must be unique. Got: {}".format(names))
        for name, step in steps:
            if not hasattr(step, "partial_fit"):
                raise TypeError("Pipeline step {!r} must implement partial_fit: a streaming "
                                "pipeline learns one chunk at a time, which fit alone cannot do"
                                .format(name))
        for name, step in steps[:-1]:
            if not hasattr(step, "transform"):
                raise TypeError("Pipeline step {!r} must implement transform".format(name))
        # Pipeline.__init__ is not called: it would also demand fit on every transformer, which
        # a streaming-only transformer need not have. It only stores and checks the steps.
        self.steps = steps

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
        if X.shape[0] == 0:
            return self
        for _, step in self.steps[:-1]:
            _call_with_optional_y(step.partial_fit, X, y)
            X = step.transform(X)
        self.model.partial_fit(X, y)
        return self

    update = partial_fit

    def fit(self, X, y=None):
        """
        Batch training: every transformer learns from all of X and transforms it, then the model
        is fitted on the result.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: array-like of shape (n_samples,), given to the steps that take it (default None)

        Returns:
        -------
        - self
        """
        if not self.steps:
            return self
        for name, step in self.steps[:-1]:
            _batch_fit(step, X, y, name)
            X = step.transform(X)
        _batch_fit(self.model, X, y, self.steps[-1][0])
        return self

    def fit_transform(self, X, y=None):
        """
        Fit every step on X in turn and return the output of the last one's transform.

        Parameters:
        ----------
        - X: array-like of shape (n_samples, n_features)
        - y: array-like of shape (n_samples,), given to the steps that take it (default None)

        Returns:
        -------
        - X_out: output of the final step's transform

        Raises:
        ------
        - TypeError: if the final step has no transform
        """
        for name, step in self.steps:
            if not hasattr(step, "transform"):
                raise TypeError("Pipeline step {!r} must implement transform".format(name))
            _batch_fit(step, X, y, name)
            X = step.transform(X)
        return X

    def reset(self):
        """
        Forget everything every step has learned, so the pipeline can start a new stream.

        Returns:
        -------
        - self

        Raises:
        ------
        - TypeError: if a step has no reset(). Nothing is reset in that case, so the pipeline is
        never left half-cleared.
        """
        for name, step in self.steps:
            if not hasattr(step, "reset"):
                raise TypeError("Pipeline step {!r} has no reset()".format(name))
        for _, step in self.steps:
            step.reset()
        return self

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
    - A chunk with no rows is skipped (it still counts in the chunk numbers).
    - Before the first tree exists the model cannot predict. A DecisionTree raises NotFittedError;
    a pipeline whose scaler has not seen data raises numcompute's plain RuntimeError. Both mean
    "not ready yet" and are caught, but only while no tree has been trained: once a tree exists,
    a RuntimeError is a real error and is raised.
    """
    metrics = metrics if metrics is not None else StreamingMetrics()
    tree = model.model if isinstance(model, StreamingPipeline) else model
    rows_read = 0
    for i, (X_chunk, y_chunk) in enumerate(chunks, start=1):
        if len(y_chunk) == 0:                  # nothing to score or learn from
            continue
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
