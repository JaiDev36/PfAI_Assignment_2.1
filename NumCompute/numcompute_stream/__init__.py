"""
numcompute_stream

Streaming learning on top of numcompute (Assignment 2.2, NumCompute-Stream).

Modules so far:
- io: load a labelled CSV (via numcompute.io) and cut it into chunks
- stats: ColumnStats (running mean, variance, min, max), StreamingQuantiles, StreamingHistogram
- preprocessing: StreamingStandardScaler, StreamingMinMaxScaler, StreamingImputer,
  StreamingOneHotEncoder, LabelEncoder
- metrics: StreamingMetrics (running confusion matrix, rolling window), StreamingAUC
- tree: DecisionTree (also DecisionTreeClassifier), a from-scratch tree with buffered retraining
- pipeline: StreamingPipeline and prequential_run (test-then-train evaluation)
- visualise: matplotlib plots of metric histories, confusion matrices and benchmarks
"""

from ._validation import NotFittedError
from .io import iter_chunks, load_labelled_csv
from .metrics import StreamingAUC, StreamingMetrics
from .pipeline import Pipeline, StreamingPipeline, prequential_run
from .preprocessing import (Imputer, LabelEncoder, OneHotEncoder, StandardScaler,
                            StreamingImputer, StreamingMinMaxScaler, StreamingOneHotEncoder,
                            StreamingStandardScaler)
from .stats import ColumnStats, StreamingHistogram, StreamingQuantiles
from .tree import DecisionTree, DecisionTreeClassifier

__version__ = "0.4.0"

__all__ = [
    "ColumnStats", "DecisionTree", "DecisionTreeClassifier", "Imputer", "LabelEncoder",
    "NotFittedError", "OneHotEncoder", "Pipeline", "StandardScaler", "StreamingAUC",
    "StreamingHistogram", "StreamingImputer", "StreamingMetrics", "StreamingMinMaxScaler",
    "StreamingOneHotEncoder", "StreamingPipeline", "StreamingQuantiles",
    "StreamingStandardScaler", "iter_chunks", "load_labelled_csv", "prequential_run",
]
