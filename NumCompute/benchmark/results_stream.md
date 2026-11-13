## NumCompute-Stream benchmarks

- python: 3.12.16
- numpy: 2.5.4
- platform: Windows-11-10.0.26200-SP0
- processor: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- seed: 5004, median of 5 runs

### 1) Speed: plain-Python loop vs vectorised

| Operation | Input | Python loop | Vectorised | Speed-up |
|---|---|---|---|---|
| tree: best-split search | n=200, 5 features | 20.75 ms | 172.3 µs | 120.5x |
| tree: best-split search | n=400, 5 features | 42.22 ms | 276.5 µs | 152.7x |
| tree: best-split search | n=800, 5 features | 95.54 ms | 538.5 µs | 177.4x |
| tree: best-split search | n=1,600, 5 features | 212.85 ms | 1.37 ms | 155.2x |
| tree: fit only (max_depth=5) | n=800, 5 features | 391.11 ms | 5.20 ms | 75.2x |
| tree: predict (depth-8 tree) | n=100,000 | 185.85 ms | 12.70 ms | 14.6x |
| ColumnStats (chunks of 1000) | n=200,000, 5 features | 88.06 ms | 16.89 ms | 5.2x |

Both versions were checked to give the same answer before timing.

### 2) The effect of buffer_size

Stream: data/stream_data.csv, 6,000 rows in chunks of 100, concept drift at row 3000. Test-then-train, tree max_depth=6, min_samples_leaf=5.

| buffer_size | Accuracy, all scored rows | Rows 3000-3999 (just after the drift) | Rows 3000-5999 (all after the drift) | Retrains | Mean retrain | Total time |
|---|---|---|---|---|---|---|
| 100 | 0.694 | 0.717 | 0.727 | 60 | 1.11 ms | 92.75 ms |
| 250 | 0.732 | 0.703 | 0.753 | 20 | 2.87 ms | 81.71 ms |
| 500 | 0.729 | 0.678 | 0.753 | 12 | 4.53 ms | 80.06 ms |
| 1000 | 0.729 | 0.603 | 0.732 | 6 | 7.56 ms | 69.00 ms |
| 2000 | 0.713 | 0.619 | 0.700 | 3 | 13.76 ms | 62.82 ms |
| 3000 | 0.626 | 0.622 | 0.626 | 2 | 19.58 ms | 59.10 ms |

How the times were taken: each buffer size is one run over the stream (not a median). Mean retrain is the average of the seconds the tree logs for each retrain, and Total time is the wall-clock time of the whole test-then-train run, including prediction and scoring. Each accuracy is the share of rows predicted correctly in the stated range (every chunk has 100 rows, so a range is a whole number of chunks).

Chunks that arrive before the first tree exists are used for training but are not scored, so a larger buffer_size scores fewer rows (the first scored row is 100 for 100, 300 for 250, 500 for 500, 1000 for 1000, 2000 for 2000, 3000 for 3000).

Figures saved to benchmark/stream_figures/.
