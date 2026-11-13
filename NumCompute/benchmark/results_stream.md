## NumCompute-Stream benchmarks

- python: 3.12.16
- numpy: 2.5.4
- platform: Windows-11-10.0.26200-SP0
- processor: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- seed: 5004, median of 5 runs

### 1) Speed: plain-Python loop vs vectorised

| Operation | Input | Python loop | Vectorised | Speed-up |
|---|---|---|---|---|
| tree: best-split search | n=200, 5 features | 20.17 ms | 187.2 µs | 107.7x |
| tree: best-split search | n=400, 5 features | 42.60 ms | 327.1 µs | 130.2x |
| tree: best-split search | n=800, 5 features | 93.95 ms | 530.3 µs | 177.2x |
| tree: best-split search | n=1,600, 5 features | 214.45 ms | 1.42 ms | 151.0x |
| tree: training (max_depth=5) | n=800, 5 features | 401.87 ms | 5.27 ms | 76.2x |
| tree: predict (depth-8 tree) | n=100,000 | 191.40 ms | 12.04 ms | 15.9x |
| ColumnStats (chunks of 1000) | n=200,000, 5 features | 89.39 ms | 17.73 ms | 5.0x |

Both versions were checked to give the same answer before timing.

### 2) The effect of buffer_size

Stream: data/stream_data.csv, 6,000 rows in chunks of 100, concept drift at row 3000. Test-then-train, tree max_depth=6, min_samples_leaf=5.

| buffer_size | Accuracy, all scored rows | Rows 3000-3999 (just after the drift) | Rows 3000-5999 (all after the drift) | Retrains | Mean retrain | Total time |
|---|---|---|---|---|---|---|
| 100 | 0.694 | 0.717 | 0.727 | 60 | 1.10 ms | 91.14 ms |
| 250 | 0.732 | 0.703 | 0.753 | 20 | 3.08 ms | 87.39 ms |
| 500 | 0.729 | 0.678 | 0.753 | 12 | 4.70 ms | 83.07 ms |
| 1000 | 0.729 | 0.603 | 0.732 | 6 | 8.36 ms | 77.12 ms |
| 2000 | 0.713 | 0.619 | 0.700 | 3 | 14.29 ms | 66.20 ms |
| 3000 | 0.626 | 0.622 | 0.626 | 2 | 19.14 ms | 57.32 ms |

How the times were taken: each buffer size is one run over the stream (not a median). Mean retrain is the average of the seconds the tree logs for each retrain, and Total time is the wall-clock time of the whole test-then-train run, including prediction and scoring. Each accuracy is the share of rows predicted correctly in the stated range (every chunk has 100 rows, so a range is a whole number of chunks).

Chunks that arrive before the first tree exists are used for training but are not scored, so a larger buffer_size scores fewer rows (the first scored row is 100 for 100, 300 for 250, 500 for 500, 1000 for 1000, 2000 for 2000, 3000 for 3000).

Figures saved to benchmark/stream_figures/.
