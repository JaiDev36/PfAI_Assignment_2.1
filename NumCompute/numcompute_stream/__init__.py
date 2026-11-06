"""
numcompute_stream

Streaming learning on top of numcompute (Assignment 2.2, NumCompute-Stream).

Modules so far:
- io: load a labelled CSV (via numcompute.io) and cut it into chunks
- stats: ColumnStats (running mean, variance, min, max), StreamingQuantiles, StreamingHistogram
"""

from ._validation import NotFittedError
from .io import iter_chunks, load_labelled_csv
from .stats import ColumnStats, StreamingHistogram, StreamingQuantiles

__version__ = "0.1.0"

__all__ = [
    "ColumnStats", "NotFittedError", "StreamingHistogram", "StreamingQuantiles",
    "iter_chunks", "load_labelled_csv",
]
