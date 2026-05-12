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
  exact command line, plus four external snapshots:
  - `threads.json` — per-worker NUMA affinity + last CPU + context-switch counts
  - `numa_maps.json` — per-VMA page placement by node, by buffer category
    (`model_mmap`, `anon`, `code_libs`, `special`)
  - `kcounters_start.json` + `kcounters_end.json` — `/proc/vmstat` deltas
    showing kernel auto-balance activity during the variant
  - `kcounters_mid.json` — mid-run snapshot including per-process context
    switches

  Plus a per-run `env.txt` with kernel, git SHA, model SHA-256, NUMA
  topology, THP state.

  All snapshots are produced by external Python/bash reading `/proc` and
  `/sys` — **zero changes to llama.cpp itself**. The Stage 1.5 upstream PR
  will reimplement these as a built-in `--numa-diagnostics` flag; until
  then, external scripts give the same data without contaminating the
  baseline.

## Reproducing on AWS c7a.32xlarge

One bootstrap script, then four commands. Total wall time ~16–22 h
(~5–15 min model download + ~10 min build + ~15–20 hr bench).
Cost: ~$108–120 on-demand.

```bash
# on a fresh c7a.32xlarge (Ubuntu 24.04), pull and read the bootstrap script first
curl -sSL https://raw.githubusercontent.com/markkobo/llama-numa-bench/main/scripts/c7a_bootstrap.sh -o /tmp/boot.sh
less /tmp/boot.sh                # always read before piping shell scripts
bash /tmp/boot.sh                # installs deps, clones both repos, wires scripts

# inside tmux (so SSH disconnect doesn't kill the 20-hr bench)
tmux new -s bench

# follow the printed next-steps. Summary:
curl -L -C - -o ~/models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf \
    https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf
cd ~/llama.cpp
./main/scripts/build.sh
./main/scripts/test.sh
./main/scripts/bench.sh --model ~/models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf --smoke
./main/scripts/bench.sh --model ~/models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf
./main/scripts/analyze.py main/results/baseline-<timestamp>/
```

Each variant's directory then gets committed here under
`results/<stage-tag>/<variant>/`.

## Running locally (single-NUMA dev box)

For harness validation without paying the c7a hourly:

```bash
./scripts/build.sh
./scripts/bench.sh --model /path/to/mixtral.Q4_K_M.gguf --smoke
```

Single-NUMA boxes silently skip the two `numactl`-prefixed variants
(`interleave`, `bind0`); the other three still run.

## License

MIT — see [`LICENSE`](LICENSE).
