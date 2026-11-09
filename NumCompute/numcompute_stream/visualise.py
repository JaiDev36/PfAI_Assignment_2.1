"""
visualise.py

Built-in plots for streaming runs, using matplotlib only.

Every function:
- accepts an optional ax to draw into (so plots can share one figure) where it draws one panel,
- accepts an optional save_path and writes a PNG there,
- returns (fig, ax) so the caller can keep styling the figure.
"""

import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#2b55d4", "#a8570b", "#1a7a5e", "#cf4320", "#6b4fbb", "#55627a"]


def _axes(ax, figsize=(7, 4)):
    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.figure, ax


def _finish(fig, ax, save_path):
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_metric_history(history, keys=("chunk_accuracy", "cumulative_accuracy"), x_key="n_seen",
                        ax=None, title="Metrics over the stream", retrain_marks=True,
                        save_path=None):
    """
    Line plot of one or more logged metrics against the number of samples scored.

    Parameters:
    ----------
    - history: dict of lists, or a StreamingMetrics (its history is used)
    - keys: sequence of str, the metrics to draw
    - x_key: str, the history column for the x axis (default 'n_seen')
    - ax: matplotlib Axes to draw into (default a new figure)
    - title: str
    - retrain_marks: bool, if the history has a 'retrains' column, draw a faint vertical line
    wherever the count went up
    - save_path: str, write a PNG here if given

    Returns:
    -------
    - (fig, ax)

    Raises:
    ------
    - KeyError: if keys or x_key are not in the history
    - ValueError: if the history is empty
    """
    history = getattr(history, "history", history)
    missing = [k for k in list(keys) + [x_key] if k not in history]
    if missing:
        raise KeyError("history has no {}. Available: {}".format(missing, sorted(history)))
    if len(history[x_key]) == 0:
        raise ValueError("history is empty; score at least one chunk first.")
    fig, ax = _axes(ax)
    x = np.asarray(history[x_key])
    for i, key in enumerate(keys):
        ax.plot(x, history[key], marker="o", ms=3, lw=1.8, color=PALETTE[i % len(PALETTE)],
                label=key.replace("_", " "))
    if retrain_marks and "retrains" in history:
        retrains = np.asarray(history["retrains"])
        for xv in x[1:][np.diff(retrains) > 0]:
            ax.axvline(xv, color="#8b97ab", lw=0.8, ls=":", zorder=0)
    ax.set_xlabel(x_key.replace("_", " "))
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    return _finish(fig, ax, save_path)


def plot_model_comparison(histories, key="cumulative_accuracy", x_key="n_seen", ax=None,
                          title=None, save_path=None):
    """
    Compare one metric across several runs, for example different buffer_size values.

    Parameters:
    ----------
    - histories: dict of {label: history}, each a dict of lists or a StreamingMetrics
    - key, x_key: str, the history columns to draw
    - ax, title, save_path: see plot_metric_history

    Returns:
    -------
    - (fig, ax)

    Raises:
    ------
    - ValueError: if histories is empty
    - KeyError: if a history lacks key or x_key
    """
    if not histories:
        raise ValueError("histories is empty.")
    fig, ax = _axes(ax)
    for i, (label, hist) in enumerate(histories.items()):
        hist = getattr(hist, "history", hist)
        for column in (key, x_key):
            if column not in hist:
                raise KeyError("history {!r} has no {!r}. Available: {}"
                               .format(label, column, sorted(hist)))
        ax.plot(hist[x_key], hist[key], lw=1.8, color=PALETTE[i % len(PALETTE)], label=str(label))
    ax.set_xlabel(x_key.replace("_", " "))
    ax.set_ylabel(key.replace("_", " "))
    ax.set_title(title or "{} by run".format(key.replace("_", " ")))
    ax.legend(fontsize=9)
    return _finish(fig, ax, save_path)


def plot_confusion_matrix(cm, labels=None, normalise=False, ax=None,
                          title="Confusion matrix", save_path=None):
    """
    Heat map of a confusion matrix with the count (or the row share) in each cell.

    Parameters:
    ----------
    - cm: array-like of shape (n_classes, n_classes), rows are true labels
    - labels: sequence of the class names, default 0, 1, ...
    - normalise: bool, show each row as a share of its total (default False)
    - ax, title, save_path: see plot_metric_history

    Returns:
    -------
    - (fig, ax)

    Raises:
    ------
    - ValueError: if cm is not square and 2-dimensional, or labels has the wrong length
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("cm must be a square 2-dimensional array. Got shape: {}".format(cm.shape))
    labels = list(range(cm.shape[0])) if labels is None else list(labels)
    if len(labels) != cm.shape[0]:
        raise ValueError("labels needs {} entries. Got: {}".format(cm.shape[0], len(labels)))
    if normalise:
        rows = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    fig, ax = _axes(ax, figsize=(4.8, 4.2))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(labels)), labels=[str(v) for v in labels])
    ax.set_yticks(range(len(labels)), labels=[str(v) for v in labels])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    cutoff = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = "{:.2f}".format(cm[i, j]) if normalise else "{}".format(int(cm[i, j]))
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cutoff else "#141c28")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_benchmark(sizes, timings, ax=None, logy=True, logx=False, xlabel="number of samples",
                   title="Loop vs vectorised", save_path=None):
    """
    Runtime curves for several implementations.

    Parameters:
    ----------
    - sizes: sequence of numbers, the x values
    - timings: dict of {label: sequence of seconds}, each as long as sizes
    - ax, title, save_path: see plot_metric_history
    - logy, logx: bool, logarithmic axes (defaults True and False)
    - xlabel: str

    Returns:
    -------
    - (fig, ax)

    Raises:
    ------
    - ValueError: if a timing series and sizes differ in length, or timings is empty
    """
    if not timings:
        raise ValueError("timings is empty.")
    fig, ax = _axes(ax)
    for i, (label, seconds) in enumerate(timings.items()):
        if len(seconds) != len(sizes):
            raise ValueError("{!r} has {} timings for {} sizes."
                             .format(label, len(seconds), len(sizes)))
        ax.plot(sizes, seconds, marker="o", lw=1.8, color=PALETTE[i % len(PALETTE)], label=label)
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("seconds")
    ax.set_title(title)
    ax.legend(fontsize=9)
    return _finish(fig, ax, save_path)


def plot_buffer_tradeoff(buffer_sizes, accuracy, retrain_seconds, n_retrains=None,
                         title="Effect of buffer_size", save_path=None):
    """
    Accuracy (left axis) and mean retrain time (right axis) against buffer_size.

    Parameters:
    ----------
    - buffer_sizes: sequence of int, the x values
    - accuracy: sequence of float, one per buffer size
    - retrain_seconds: sequence of float, mean seconds per retrain, one per buffer size
    - n_retrains: sequence of int, optional, written above each accuracy point
    - title, save_path: see plot_metric_history

    Returns:
    -------
    - (fig, (ax, ax_right))

    Raises:
    ------
    - ValueError: if the sequences differ in length
    """
    lengths = {len(buffer_sizes), len(accuracy), len(retrain_seconds)}
    if n_retrains is not None:
        lengths.add(len(n_retrains))
    if len(lengths) != 1:
        raise ValueError("buffer_sizes, accuracy, retrain_seconds and n_retrains must have the "
                         "same length.")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(buffer_sizes, accuracy, marker="o", lw=2, color=PALETTE[0])
    ax.set_xlabel("buffer_size")
    ax.set_ylabel("prequential accuracy", color=PALETTE[0])
    ax_right = ax.twinx()
    ax_right.plot(buffer_sizes, np.asarray(retrain_seconds) * 1000, marker="s", lw=2,
                  color=PALETTE[1])
    ax_right.set_ylabel("mean retrain time (ms)", color=PALETTE[1])
    if n_retrains is not None:
        for b, a, r in zip(buffer_sizes, accuracy, n_retrains):
            ax.annotate("{} retrains".format(r), (b, a), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color="#55627a")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, (ax, ax_right)


def plot_retrain_log(retrain_log, ax=None, title="Tree after each retrain", save_path=None):
    """
    Depth, leaf count and training accuracy of each retrained tree.

    Parameters:
    ----------
    - retrain_log: list of dict, DecisionTree.retrain_log_
    - ax, title, save_path: see plot_metric_history

    Returns:
    -------
    - (fig, ax)

    Raises:
    ------
    - ValueError: if the log is empty
    """
    if not retrain_log:
        raise ValueError("retrain_log is empty; the tree has not been retrained yet.")
    number = [e["retrain"] for e in retrain_log]
    fig, ax = _axes(ax)
    ax.bar(number, [e["n_leaves"] for e in retrain_log], color=PALETTE[0], alpha=0.35,
           label="leaves")
    ax.plot(number, [e["depth"] for e in retrain_log], marker="o", color=PALETTE[0],
            label="depth")
    ax.set_xlabel("retrain number")
    ax.set_ylabel("depth / leaves")
    ax_right = ax.twinx()
    ax_right.plot(number, [e["train_accuracy"] for e in retrain_log], marker="s",
                  color=PALETTE[2], label="train accuracy")
    ax_right.set_ylabel("train accuracy", color=PALETTE[2])
    ax_right.set_ylim(0, 1.05)
    left, right = ax.get_legend_handles_labels(), ax_right.get_legend_handles_labels()
    ax.legend(left[0] + right[0], left[1] + right[1], fontsize=9, loc="lower right")
    ax.set_title(title)
    return _finish(fig, ax, save_path)


# ---------------------------------------------------------------------
# Plots named in the assignment brief
# ---------------------------------------------------------------------
def _as_series(values, name):
    arr = np.asarray(getattr(values, "history", values), dtype=np.float64) \
        if not isinstance(values, dict) else None
    if arr is None:
        raise TypeError("{} must be a sequence of numbers, not a dict. Pass one column, e.g. "
                        "metrics.series('chunk_accuracy').".format(name))
    if arr.ndim != 1:
        raise ValueError("{} must be 1-dimensional. Got shape: {}".format(name, arr.shape))
    if arr.size == 0:
        raise ValueError("{} is empty; score at least one chunk first.".format(name))
    return arr


def _finish_shown(fig, ax, save_path, show):
    fig, ax = _finish(fig, ax, save_path)
    if show:
        plt.show()
    return fig, ax


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Accuracy", ax=None,
                          xlabel="chunk", save_path=None, show=False):
    """
    Plot one metric (for example accuracy) across the chunks of a stream.

    Parameters:
    ----------
    - metric_values: sequence of numbers, one per chunk (for example metrics.series('error'))
    - title: str
    - ylabel: str
    - ax: matplotlib Axes to draw into (default a new figure)
    - xlabel: str (default 'chunk')
    - save_path: str, write a PNG here if given
    - show: bool, call plt.show() for inline display (default False)

    Returns:
    -------
    - (fig, ax)

    Raises:
    ------
    - ValueError: if metric_values is empty or not 1-dimensional
    - TypeError: if metric_values is a dict

    Complexity:
    ----------
    - Time and space complexity: O(chunks)
    """
    values = _as_series(metric_values, "metric_values")
    fig, ax = _axes(ax)
    ax.plot(np.arange(1, values.size + 1), values, marker="o", ms=3, lw=1.8, color=PALETTE[0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return _finish_shown(fig, ax, save_path, show)


def compare_models(metric1, metric2, labels=("model 1", "model 2"), ax=None,
                   title="Model comparison", ylabel="Accuracy", xlabel="chunk",
                   save_path=None, show=False):
    """
    Compare two models on a streaming metric, for example trees with different buffer_size.

    Parameters:
    ----------
    - metric1, metric2: sequences of numbers, one per chunk (they may differ in length)
    - labels: pair of str, the legend names (default 'model 1', 'model 2')
    - ax, title, ylabel, xlabel, save_path, show: see plot_metric_over_time

    Returns:
    -------
    - (fig, ax)

    Raises:
    ------
    - ValueError: if a series is empty or not 1-dimensional, or labels does not hold two names
    - TypeError: if a metric is a dict
    """
    labels = list(labels)
    if len(labels) != 2:
        raise ValueError("labels needs exactly 2 names. Got: {}".format(len(labels)))
    series = [_as_series(metric1, "metric1"), _as_series(metric2, "metric2")]
    fig, ax = _axes(ax)
    for i, values in enumerate(series):
        ax.plot(np.arange(1, values.size + 1), values, lw=1.8, color=PALETTE[i],
                label=str(labels[i]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    return _finish_shown(fig, ax, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, ax=None,
                                     title="Predictions vs ground truth", save_path=None,
                                     show=False):
    """
    Show the predicted and the true class of every sample of a chunk (normally the latest one).

    Parameters:
    ----------
    - y_true: array-like of shape (n_samples,), the true labels
    - y_pred: array-like of shape (n_samples,), the predicted labels
    - ax, title, save_path, show: see plot_metric_over_time

    Returns:
    -------
    - (fig, ax)

    Raises:
    ------
    - ValueError: if the inputs are not 1-dimensional, differ in length or are empty

    Note:
    ----
    - Each sample is a position on the x axis. The true class is a hollow circle and the
    prediction a filled dot; a red cross marks every wrong prediction. The title reports the
    chunk's accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-dimensional. Got shapes: {} and {}"
                         .format(y_true.shape, y_pred.shape))
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length. Got: {} and {}"
                         .format(y_true.size, y_pred.size))
    if y_true.size == 0:
        raise ValueError("y_true and y_pred are empty.")
    classes = np.unique(np.concatenate([y_true, y_pred]))
    true_pos = np.searchsorted(classes, y_true)
    pred_pos = np.searchsorted(classes, y_pred)
    x = np.arange(y_true.size)
    wrong = true_pos != pred_pos
    fig, ax = _axes(ax, figsize=(8, 3.6))
    ax.scatter(x, true_pos, s=60, facecolors="none", edgecolors=PALETTE[0], label="true")
    ax.scatter(x, pred_pos, s=18, color=PALETTE[1], label="predicted")
    if wrong.any():
        ax.scatter(x[wrong], pred_pos[wrong], marker="x", s=60, color=PALETTE[3],
                   label="wrong")
    ax.set_yticks(range(classes.size), labels=[str(c) for c in classes])
    ax.set_xlabel("sample in chunk")
    ax.set_ylabel("class")
    ax.set_title("{} (accuracy {:.3f})".format(title, float(np.mean(~wrong))))
    ax.legend(loc="best", fontsize=9)
    return _finish_shown(fig, ax, save_path, show)
