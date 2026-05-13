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

**v0.2 — first partial results.** Stage 0 baseline data from rding-bench
(AMD EPYC 9R14, 128 vCPU, dual-NUMA, 256 GiB) under
[`results/baseline-20260512-054249/`](results/baseline-20260512-054249/).
Two of five NUMA variants complete; see commit history and the
[per-results README](results/README.md) for what shipped when.

## Stage 0 protocol (summary)

- **Hardware:** dual-socket AMD EPYC Zen 4 (AMD EPYC 9R14 verified on
  rding-bench; equivalent c7a.32xlarge spec also works). 128+ vCPU,
  two NUMA nodes, ≥128 GiB RAM per node.
- **Model:** Qwen3-30B-A3B Q4_K_M
  ([unsloth/Qwen3-30B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF)),
  ~17.3 GiB. 30B total params, 3B active, 128 experts top-8 — a current
  sparse-MoE design well-suited to per-expert NUMA-skew analysis. SHA-256:
  `9f1a24700a339b09c06009b729b5c809e0b64c213b8af5b711b3dbdfd0c5ba48`.
  (The roadmap originally named Mixtral 8x7B Q4_K_M; pivoted on 2026-05-12
  because available Mixtral GGUFs use the pre-stacked-experts tensor
  layout that current llama.cpp master no longer accepts. Qwen3-30B-A3B
  is both currently supported and more representative of where sparse MoE
  is heading.)
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
curl -L -C - -o ~/models/Qwen3-30B-A3B-Q4_K_M.gguf \
    https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf
cd ~/llama.cpp
./main/scripts/build.sh
./main/scripts/test.sh
./main/scripts/bench.sh --model ~/models/Qwen3-30B-A3B-Q4_K_M.gguf --smoke
./main/scripts/bench.sh --model ~/models/Qwen3-30B-A3B-Q4_K_M.gguf
./main/scripts/analyze.py main/results/baseline-<timestamp>/
```

Each variant's directory then gets committed here under
`results/<stage-tag>/<variant>/`.

## Running locally (single-NUMA dev box)

For harness validation without paying the c7a hourly:

```bash
./scripts/build.sh
./scripts/bench.sh --model /path/to/Qwen3-30B-A3B-Q4_K_M.gguf --smoke
```

Single-NUMA boxes silently skip the two `numactl`-prefixed variants
(`interleave`, `bind0`); the other three still run.

## License

MIT — see [`LICENSE`](LICENSE).
