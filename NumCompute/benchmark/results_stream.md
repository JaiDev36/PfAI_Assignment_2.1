## NumCompute-Stream benchmarks

- python: 3.12.16
- numpy: 2.5.4
- platform: Windows-11-10.0.26200-SP0
- processor: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- seed: 5004, median of 5 runs

### 1) Speed: plain-Python loop vs vectorised

| Operation | Input | Python loop | Vectorised | Speed-up |
|---|---|---|---|---|
| tree: best-split search | n=200, 5 features | 20.07 ms | 168.8 µs | 118.9x |
| tree: best-split search | n=400, 5 features | 41.48 ms | 281.8 µs | 147.2x |
| tree: best-split search | n=800, 5 features | 92.58 ms | 507.7 µs | 182.4x |
| tree: best-split search | n=1,600, 5 features | 211.23 ms | 1.30 ms | 162.3x |
| tree: training (max_depth=5) | n=800, 5 features | 397.92 ms | 5.53 ms | 72.0x |
| tree: predict (depth-8 tree) | n=100,000 | 187.91 ms | 12.10 ms | 15.5x |
| ColumnStats (chunks of 1000) | n=200,000, 5 features | 86.91 ms | 12.81 ms | 6.8x |

Both versions were checked to give the same answer before timing.

### 2) The effect of buffer_size

Stream: data/stream_data.csv, 6,000 rows in chunks of 100, concept drift at row 3000. Test-then-train, tree max_depth=6.

| buffer_size | Accuracy | Accuracy in the 1000 rows after the drift | Retrains | Mean retrain | Total time |
|---|---|---|---|---|---|
| 100 | 0.694 | 0.717 | 60 | 1.10 ms | 88.52 ms |
| 250 | 0.732 | 0.703 | 20 | 3.07 ms | 84.90 ms |
| 500 | 0.729 | 0.678 | 12 | 4.52 ms | 76.48 ms |
| 1000 | 0.729 | 0.603 | 6 | 8.26 ms | 74.54 ms |
| 2000 | 0.713 | 0.619 | 3 | 14.62 ms | 64.59 ms |
| 3000 | 0.626 | 0.622 | 2 | 20.14 ms | 57.80 ms |

Chunks that arrive before the first tree exists are used for training but are not scored, so a larger buffer_size scores fewer rows (the first scored row is 100 for 100, 300 for 250, 500 for 500, 1000 for 1000, 2000 for 2000, 3000 for 3000).

Figures saved to benchmark/stream_figures/.
