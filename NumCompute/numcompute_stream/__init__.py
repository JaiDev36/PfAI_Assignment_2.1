"""
numcompute_stream

Streaming learning on top of numcompute (Assignment 2.2, NumCompute-Stream).

Modules so far:
- io: load a labelled CSV (via numcompute.io) and cut it into chunks
- stats: ColumnStats (running mean, variance, min, max), StreamingQuantiles, StreamingHistogram
- preprocessing: StreamingStandardScaler, StreamingMinMaxScaler, StreamingImputer,
  StreamingOneHotEncoder, LabelEncoder
"""

from ._validation import NotFittedError
from .io import iter_chunks, load_labelled_csv
from .preprocessing import (Imputer, LabelEncoder, OneHotEncoder, StandardScaler,
                            StreamingImputer, StreamingMinMaxScaler, StreamingOneHotEncoder,
                            StreamingStandardScaler)
from .stats import ColumnStats, StreamingHistogram, StreamingQuantiles

__version__ = "0.2.0"

__all__ = [
    "ColumnStats", "Imputer", "LabelEncoder", "NotFittedError", "OneHotEncoder",
    "StandardScaler", "StreamingHistogram", "StreamingImputer", "StreamingMinMaxScaler",
    "StreamingOneHotEncoder", "StreamingQuantiles", "StreamingStandardScaler", "iter_chunks",
    "load_labelled_csv",
]
