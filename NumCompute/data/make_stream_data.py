"""
make_stream_data.py

Generate data/stream_data.csv: a synthetic 3-class stream with missing values and concept drift.

Hypothetical scenario: a monitoring system reads five sensor values per reading and labels each
reading as normal, warning or fault. Halfway through the stream the equipment is recalibrated, so
the boundaries between the classes move (concept drift). A model trained only on old data gets
worse after the drift; a model that keeps retraining recovers.

Properties:
- 6000 rows, 5 numeric features, 3 text classes
- about 3% of the feature values are missing (empty cells)
- drift at row 3000
- fixed seed, so the file is identical on every run

Run from the NumCompute folder:

    python data/make_stream_data.py
"""

import os

import numpy as np

N_ROWS = 6000
DRIFT_AT = 3000
SEED = 2026
FEATURES = ["temperature", "vibration", "pressure", "load", "noise"]
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "stream_data.csv")


def make_stream(n=N_ROWS, drift_at=DRIFT_AT, seed=SEED, missing_rate=0.03):
    """
    Build the stream in memory.

    Parameters:
    ----------
    - n: int, number of rows
    - drift_at: int, row at which the class boundaries move
    - seed: int, seed of the random generator
    - missing_rate: float in [0, 1), share of feature values replaced by NaN

    Returns:
    -------
    - X: np.ndarray of shape (n, 5), float64, with NaN for missing values
    - y: np.ndarray of shape (n,), str, one of 'normal', 'warning', 'fault'
    """
    rng = np.random.default_rng(seed)
    temperature = rng.normal(60, 10, n)
    vibration = rng.gamma(2.0, 1.5, n)
    pressure = rng.normal(100, 15, n)
    load = rng.uniform(0, 1, n)
    noise = rng.normal(0, 1, n)                      # carries no signal

    after = np.arange(n) >= drift_at
    # risk score; after recalibration temperature matters less and vibration more
    risk = np.where(after,
                    0.03 * (temperature - 60) + 0.55 * (vibration - 3) + 1.2 * (load - 0.5),
                    0.08 * (temperature - 60) + 0.25 * (vibration - 3) + 1.2 * (load - 0.5))
    risk += 0.01 * (pressure - 100) + rng.normal(0, 0.35, n)
    y = np.where(risk > 0.9, "fault", np.where(risk > 0.1, "warning", "normal"))

    X = np.column_stack([temperature, vibration, pressure, load, noise]).round(3)
    X[rng.random(X.shape) < missing_rate] = np.nan
    return X, y


def write_csv(path, X, y):
    """Write X and y as a CSV with a header; NaN becomes an empty cell."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(FEATURES + ["status"]) + "\n")
        for row, label in zip(X.tolist(), y.tolist()):
            cells = ["" if v != v else repr(v) for v in row]
            f.write(",".join(cells + [label]) + "\n")


def ensure_stream_csv(path=DEFAULT_PATH):
    """Create the CSV file if it does not exist yet and return its path."""
    if not os.path.exists(path):
        write_csv(path, *make_stream())
    return path


if __name__ == "__main__":
    X_all, y_all = make_stream()
    write_csv(DEFAULT_PATH, X_all, y_all)
    classes, counts = np.unique(y_all, return_counts=True)
    print("wrote {} with {} rows; classes: {}".format(
        DEFAULT_PATH, len(y_all), dict(zip(classes.tolist(), counts.tolist()))))
