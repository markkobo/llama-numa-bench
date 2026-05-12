# llama-numa-bench

Reproducible NUMA benchmark harness for [llama.cpp](https://github.com/ggml-org/llama.cpp).

Companion repo to ongoing NUMA-related work in
[markkobo/llama.cpp](https://github.com/markkobo/llama.cpp). Datasets here are
the "before / after" evidence behind upstream PR proposals.

## What this is

- Stage-tagged baseline datasets — per-iteration latency (p50/p95/p99/p99.9)
  plus `perf stat` event counts (`node-load-misses`, `cpu-migrations`,
  `dTLB-load-misses`, etc.) measured on real NUMA hardware.
- The scripts and exact commands that produced those numbers.
- Methodology notes — what each variant means, what's measured, how to redo
  it on different hardware.

## Why it exists

NUMA performance discussion in the llama.cpp community (issue
[#1437](https://github.com/ggml-org/llama.cpp/issues/1437) and friends) tends
to be model- and hardware-specific anecdote. This repo collects rigorous,
reproducible numbers with explicit protocol, so claims about NUMA-related
changes can be argued from data rather than intuition.

## Status

**v0.1 — skeleton.** Stage 0 baseline datasets land here once measured on
AWS c7a.32xlarge. See [`results/`](results/) for protocol notes and the
data as it accumulates.

## Stage 0 protocol (summary)

- **Hardware:** AWS c7a.32xlarge — dual-socket AMD EPYC Zen 4, 192 vCPU,
  two NUMA nodes.
- **Model:** Mixtral 8x7B Q4_K_M
  ([TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF)),
  ~24.6 GiB.
- **Workload:** `llama-bench -p 512 -n 128 -r 1000` (one thousand reps of
  prompt-processing-512 and token-generation-128).
- **Variants (5):**
  1. `baseline` — no NUMA flag, no `numactl`.
  2. `distribute` — `llama-bench --numa distribute`.
  3. `isolate` — `llama-bench --numa isolate`.
  4. `interleave` — `numactl --interleave=all llama-bench --numa numactl`.
  5. `bind0` — `numactl --cpubind=0 --membind=0 llama-bench --numa numactl`.
- **Co-measurement:** `perf stat -e
  task-clock,context-switches,cpu-migrations,cycles,instructions,branches,branch-misses,node-load-misses,node-loads,dTLB-load-misses,LLC-load-misses,LLC-loads`.
- **Output per variant:** JSONL bench results, `perf.log`, captured stderr,
  exact command line. Plus a per-run `env.txt` with kernel, git SHA, model
  SHA-256, NUMA topology, THP state.

## Reproducing

Scripts currently live in the working fork at
`markkobo/llama.cpp:main/scripts/` and will migrate here once Stage 0 ships.

In brief:

```bash
# in your llama.cpp checkout
./main/scripts/build.sh
./main/scripts/bench.sh --model /path/to/mixtral.Q4_K_M.gguf --smoke    # validate harness
./main/scripts/bench.sh --model /path/to/mixtral.Q4_K_M.gguf            # the real run
./main/scripts/analyze.py main/results/baseline-<timestamp>/
```

Each variant's directory ends up here as `results/<stage-tag>/<variant>/`.

## License

MIT — see [`LICENSE`](LICENSE).
