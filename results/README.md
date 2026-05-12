# results/

Datasets land in this directory, organized by stage tag.

```
results/
├── baseline-<utc-timestamp>/    # Stage 0 baseline runs
│   ├── env.txt                  # host / kernel / git SHA / model SHA-256 / NUMA topology / THP
│   ├── baseline/
│   │   ├── bench.jsonl          # llama-bench JSONL output (per-iteration samples_ns)
│   │   ├── perf.log             # perf stat output
│   │   ├── cmd.txt              # exact command used
│   │   └── stderr.log
│   ├── distribute/...
│   ├── isolate/...
│   ├── interleave/...
│   └── bind0/...
└── mbind-poc-<utc-timestamp>/   # Stage 1 mbind PoC runs (pinned vs unpinned)
    └── ...
```

## Reading the data

`bench.jsonl` is one JSON object per llama-bench test (`pp512`, `tg128`).
Each has a `samples_ns` array — that's per-iteration wall-time in
nanoseconds, suitable for computing tail percentiles. Reps default 1000.

`perf.log` is plain `perf stat` text output. Of these events the
NUMA-relevant ones are `node-load-misses` and `node-loads` (lower ratio =
more local memory access).

`env.txt` is essential for interpretation — model SHA-256 catches silent
file corruption, kernel version + THP state determine whether transparent
hugepages are confounding the result, NUMA topology shows actual node layout.

## Re-running on different hardware

The scripts under `markkobo/llama.cpp:main/scripts/` are hardware-agnostic;
they detect NUMA topology at runtime and skip `numactl` variants if it's not
installed. Drop their output into a new subdirectory here and the comparison
is plug-and-play.
