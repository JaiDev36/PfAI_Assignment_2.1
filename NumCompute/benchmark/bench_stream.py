"""
bench_stream.py

Reproducible benchmarks for numcompute_stream (Assignment 2.2).

Two questions, answered with numcompute.benchmarking:

1) Speed: a plain-Python loop against the vectorised version, for the split search, tree training,
   tree prediction and per-feature running statistics. Both versions run once first and must agree
   (compare_loop_vs_vectorised raises otherwise), then the median time of several runs is reported.
2) The effect of buffer_size: the tree is run over data/stream_data.csv (6000 rows, concept drift
   at row 3000) in chunks of 100 rows with test-then-train evaluation, for several buffer sizes.

Run from the NumCompute folder:

    python benchmark/bench_stream.py
    python benchmark/bench_stream.py --quick        (smaller sizes, a few seconds)

A full run also saves the figures to benchmark/stream_figures/.

Reproducibility:
- every random input comes from a fixed seed (SEED below) and the stream file is generated from a
  fixed seed too
- each timing is the median of REPEATS runs after one warm-up run
- Python, NumPy and platform versions are printed with the results
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# make `import numcompute`, `numcompute_stream` and the data script work from NumCompute/
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))

from numcompute import benchmarking  # noqa: E402
from numcompute_stream import (DecisionTree, StreamingImputer, StreamingPipeline,  # noqa: E402
                               iter_chunks, load_labelled_csv, prequential_run)
from numcompute_stream.stats import ColumnStats  # noqa: E402
from numcompute_stream.tree import best_split_loop, best_split_vectorised  # noqa: E402
from make_stream_data import DRIFT_AT, ensure_stream_csv  # noqa: E402

SEED = 5004
REPEATS = 5
CHUNK = 100
BUFFER_SIZES = [100, 250, 500, 1000, 2000, 3000]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def fmt_time(seconds):
    """Format seconds as s, ms or µs for the tables."""
    if seconds >= 1:
        return "{:.2f} s".format(seconds)
    if seconds >= 1e-3:
        return "{:.2f} ms".format(seconds * 1e3)
    return "{:.1f} µs".format(seconds * 1e6)


def make_data(rng, n, d=5):
    """Three classes that depend on the first two features, plus noise."""
    X = rng.normal(size=(n, d))
    score = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.3, n)
    return X, np.digitize(score, [-0.5, 0.5]).astype(np.int64)


def loop_column_stats(X):
    """One value at a time, textbook Welford, for every column. Baseline for ColumnStats."""
    out = []
    for j in range(X.shape[1]):
        count, mean, m2 = 0, 0.0, 0.0
        for v in X[:, j].tolist():
            count += 1
            delta = v - mean
            mean += delta / count
            m2 += delta * (v - mean)
        out.append((mean, m2 / count))
    return np.array(out)


def vectorised_column_stats(X, chunk=1000):
    """ColumnStats fed in chunks, as a stream would."""
    stats = ColumnStats()
    for start in range(0, len(X), chunk):
        stats.update(X[start:start + chunk])
    return np.column_stack([stats.mean, stats.var])


# ---------------------------------------------------------------------
# 1) loop vs vectorised
# ---------------------------------------------------------------------

def bench_speed(quick):
    """Return a list of rows (what, input, loop seconds, vectorised seconds, speed-up)."""
    rng = np.random.default_rng(SEED)
    repeats = 2 if quick else REPEATS
    rows = []

    def add(what, size, result):
        rows.append((what, size, result["loop"], result["vectorised"], result["speedup"]))

    for n in ([200, 400] if quick else [200, 400, 800, 1600]):
        X, y = make_data(rng, n)
        add("tree: best-split search", "n={:,}, 5 features".format(n),
            benchmarking.compare_loop_vs_vectorised(
                lambda: best_split_loop(X, y, 3), lambda: best_split_vectorised(X, y, 3),
                repeats=repeats))

    n = 400 if quick else 800
    X, y = make_data(rng, n)
    add("tree: training (max_depth=5)", "n={:,}, 5 features".format(n),
        benchmarking.compare_loop_vs_vectorised(
            lambda: DecisionTree(max_depth=5, splitter="loop").fit(X, y).predict(X),
            lambda: DecisionTree(max_depth=5).fit(X, y).predict(X),
            repeats=repeats))

    X_train, y_train = make_data(rng, 2000)
    tree = DecisionTree(max_depth=8).fit(X_train, y_train)
    n = 20_000 if quick else 100_000
    X, _ = make_data(rng, n)
    add("tree: predict (depth-8 tree)", "n={:,}".format(n),
        benchmarking.compare_loop_vs_vectorised(
            lambda: tree.predict_loop(X), lambda: tree.predict(X), repeats=repeats))

    n = 20_000 if quick else 200_000
    X = rng.normal(size=(n, 5))
    add("ColumnStats (chunks of 1000)", "n={:,}, 5 features".format(n),
        benchmarking.compare_loop_vs_vectorised(
            lambda: loop_column_stats(X), lambda: vectorised_column_stats(X),
            repeats=repeats, rtol=1e-7))
    return rows


# ---------------------------------------------------------------------
# 2) buffer_size
# ---------------------------------------------------------------------

def bench_buffer(X, y, sizes):
    """
    For each buffer size: overall and post-drift accuracy, retrain count and retrain cost.

    Returns a list of dicts and a dict of {buffer_size: metrics history}.
    """
    results, histories = [], {}
    for size in sizes:
        pipe = StreamingPipeline([("imputer", StreamingImputer()),
                                  ("tree", DecisionTree(max_depth=6, min_samples_leaf=5,
                                                        buffer_size=size))])
        start = time.perf_counter()
        scored = prequential_run(pipe, iter_chunks(X, y, CHUNK), log_every=0)
        wall = time.perf_counter() - start
        history = scored.history
        first_row = np.array(history["stream_chunk"]) * CHUNK - CHUNK
        after = (first_row >= DRIFT_AT) & (first_row < DRIFT_AT + 1000)
        log = pipe.model.retrain_log_
        results.append({
            "buffer_size": size,
            "accuracy": scored.accuracy,
            "post_drift": float(np.mean(np.array(history["chunk_accuracy"])[after])),
            "retrains": len(log),
            "mean_retrain": float(np.mean([e["seconds"] for e in log])),
            "wall": wall,
            "first_scored_row": int(first_row[0]),
        })
        histories[size] = history
    return results, histories


def save_figures(results, histories, folder):
    """Write the buffer-size figures (a full run only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from numcompute_stream import visualise

    os.makedirs(folder, exist_ok=True)
    fig, _ = visualise.plot_buffer_tradeoff(
        [r["buffer_size"] for r in results], [r["accuracy"] for r in results],
        [r["mean_retrain"] for r in results], n_retrains=[r["retrains"] for r in results],
        title="buffer_size: accuracy against retrain cost")
    fig.savefig(os.path.join(folder, "buffer_tradeoff.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    smooth = {}
    for size in (100, 500, 2000):
        acc = np.array(histories[size]["chunk_accuracy"])
        rows_read = np.array(histories[size]["rows_read"])
        rolling = np.convolve(acc, np.ones(5) / 5, "valid")        # 5-chunk moving average
        smooth["buffer_size={}".format(size)] = {"rows_read": rows_read[4:],
                                                 "rolling_accuracy": rolling}
    visualise.plot_model_comparison(smooth, key="rolling_accuracy", x_key="rows_read", ax=ax,
                                    title="Rolling accuracy (5 chunks) through the stream")
    ax.axvline(DRIFT_AT, color="#cf4320", ls="--", lw=1.2)
    ax.text(DRIFT_AT, ax.get_ylim()[0] + 0.02, " drift", color="#cf4320")
    fig.savefig(os.path.join(folder, "buffer_accuracy_over_time.png"), dpi=130)
    plt.close("all")


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--quick", action="store_true", help="smaller sizes, nothing is saved")
    args = parser.parse_args()

    env = benchmarking.environment_info()
    print("## NumCompute-Stream benchmarks\n")
    for key in ("python", "numpy", "platform", "processor"):
        print("- {}: {}".format(key, env[key]))
    print("- seed: {}, median of {} runs\n".format(SEED, 2 if args.quick else REPEATS))

    print("### 1) Speed: plain-Python loop vs vectorised\n")
    print("| Operation | Input | Python loop | Vectorised | Speed-up |")
    print("|---|---|---|---|---|")
    for what, size, loop, vec, speedup in bench_speed(args.quick):
        print("| {} | {} | {} | {} | {:,.1f}x |".format(what, size, fmt_time(loop),
                                                        fmt_time(vec), speedup))
    print("\nBoth versions were checked to give the same answer before timing.\n")

    print("### 2) The effect of buffer_size\n")
    path = ensure_stream_csv()
    X, y, _ = load_labelled_csv(path)
    sizes = [100, 500, 2000] if args.quick else BUFFER_SIZES
    results, histories = bench_buffer(X, y, sizes)
    print("Stream: data/stream_data.csv, {:,} rows in chunks of {}, concept drift at row {}. "
          "Test-then-train, tree max_depth=6.\n".format(len(y), CHUNK, DRIFT_AT))
    print("| buffer_size | Accuracy | Accuracy in the 1000 rows after the drift | Retrains "
          "| Mean retrain | Total time |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print("| {} | {:.3f} | {:.3f} | {} | {} | {} |".format(
            r["buffer_size"], r["accuracy"], r["post_drift"], r["retrains"],
            fmt_time(r["mean_retrain"]), fmt_time(r["wall"])))
    print("\nChunks that arrive before the first tree exists are used for training but are not "
          "scored, so a larger buffer_size scores fewer rows (the first scored row is "
          "{}).".format(", ".join("{} for {}".format(r["first_scored_row"], r["buffer_size"])
                                  for r in results)))

    if not args.quick:
        folder = os.path.join(HERE, "stream_figures")
        save_figures(results, histories, folder)
        print("\nFigures saved to benchmark/stream_figures/.")


if __name__ == "__main__":
    main()
