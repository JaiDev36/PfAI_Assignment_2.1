"""
run_all.py

Runs every benchmark script in this folder one after the other.

    python benchmark/run_all.py            (full sizes, saves results.md)
    python benchmark/run_all.py --quick    (smaller sizes, a few seconds, nothing is saved)

Covers metrics, optim, stats, rank, sort_search, utils, benchmarking and numcompute_stream.
The output of a full run is written to benchmark/results.md.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["bench_metrics_optim_stats.py", "bench_rank_sort_utils.py", "bench_stream.py"]
HEADER = (
    "# NumCompute benchmark results\n\n"
    "Produced by `python benchmark/run_all.py`. Both the timings and the speed-up ratios depend\n"
    "on the machine, the Python and NumPy versions, and the input sizes used here.\n\n"
)


def main():
    extra = sys.argv[1:]
    quick = "--quick" in extra
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    parts = [HEADER]
    for name in SCRIPTS:
        print("Running {} ...".format(name), flush=True)
        run = subprocess.run([sys.executable, os.path.join(HERE, name)] + extra,
                             capture_output=True, text=True, encoding="utf-8", env=env)
        print(run.stdout)
        if run.returncode != 0:
            print(run.stderr, file=sys.stderr)
            sys.exit(run.returncode)
        parts.append(run.stdout)

    if quick:
        print("(--quick run: results.md was not changed)")
        return
    path = os.path.join(HERE, "results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("Saved {}".format(path))


if __name__ == "__main__":
    main()
