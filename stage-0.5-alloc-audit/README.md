# Stage 0.5 — Allocation-path audit

**Purpose**: before pinning anything, map where Qwen3-30B-A3B's bytes actually come from at runtime. Without this, the Stage 1 mbind PoC may pin memory that doesn't dominate the working set.

**Method**: two configurations (default mmap vs `--no-mmap`), captured via `/proc/<pid>/smaps` + `/proc/<pid>/numa_maps`. Cross-referenced with a source-code trace of llama.cpp's allocation paths so each VMA can be tied back to a specific code path.

**No llama.cpp code changes.** Pure observation + source analysis.

**Status**: source-code trace complete (Part A below). Empirical numbers populated by running `measure.sh` on the bench box (Part B). The Part A predictions and Part B measurements should agree within a few percent; disagreements indicate either a code path missed in the trace or a methodology error in the measurement.

---

## Part A — Source-code trace (build SHA `8e1f9d083`)

For each memory class, traces from the user-visible CLI flag → the actual allocator function. Cited as `file:line`.

### Hook under test

The PoC's single touch point in Stage 1 is:

```cpp
// ggml/src/ggml-backend.cpp:2305-2327
static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(
    ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);   // <-- this is where mbind would attach
    ...
}
```

"Visible to the Stage 1 hook" below means: bytes pass through `ggml_aligned_malloc()` at `ggml-backend.cpp:2306` (in the CPU backend's buffer allocator) and would therefore be subject to the mbind policy a Stage 1 hook installs.

### Per-class trace

1. **Model weights, default `mmap`** — *NOT visible to the hook.*
   - `llama-model-loader.cpp:1327` — `init_mappings()` calls `llama_mmap` ctor (POSIX `mmap()` of the GGUF file).
   - `llama-model.cpp:1457-1465` — buffer is created via `ggml_backend_dev_buffer_from_host_ptr(dev, addr+first, last-first, max_size)` which **wraps the existing mmap'd VA range** without copying.
   - `llama-model-loader.cpp:1545` — tensors `ggml_backend_tensor_alloc(buf_mmap, cur, data)` where `data = mapping->addr() + weight->offs`. Tensor data pointers point directly into the file-backed VMA.
   - No `ggml_aligned_malloc()` call in this path.

2. **Model weights, `--no-mmap`** — *visible to the hook (100%).*
   - `llama.cpp:412` — CLI flag sets `params.use_mmap = false`.
   - `llama-model.cpp:1466-1488` — falls through to `ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)`.
   - `ggml-alloc.c:1188` (via `1238`) — calls `ggml_backend_buft_alloc_buffer()` for each tensor range.
   - For CPU backend, this dispatches to `ggml_backend_cpu_buffer_type_alloc_buffer()` → `ggml_aligned_malloc()`. ✅
   - `llama-model-loader.cpp:1560-1562` — `file->read_raw(cur->data, n_size)` then memcpys file contents into the freshly allocated anonymous buffer.

3. **KV cache** — *visible to the hook.*
   - `llama-kv-cache.cpp:191` — `buft = ggml_backend_cpu_buffer_type()`.
   - `llama-kv-cache.cpp:255-262` — allocates via `ggml_backend_alloc_ctx_tensors_from_buft()` → hook. ✅
   - Same code path for `llama-kv-cache-iswa.cpp` (interleaved sliding window variant; Qwen3 doesn't use it but listed for completeness).

4. **Compute scratch / graph buffers** — *visible to the hook.*
   - `llama-context.cpp:412` — `ggml_backend_sched_new(...)` initializes the scheduler with the CPU buffer type.
   - `llama-context.cpp:1214` — `ggml_backend_sched_alloc_graph()` per `decode()` call drives buffer allocation through `ggml_backend_buft_alloc_buffer()` → hook. ✅
   - These are reused across decode steps via the scheduler's gallocr internals.

5. **Activations / intermediate tensors** — *visible to the hook.*
   - No separate allocator. Activations live inside the compute-scratch buffer (4 above). Same code path, same hook visibility. ✅

6. **Tensor metadata (the `ggml_tensor` struct bodies themselves, not their data)** — *NOT visible to the hook.*
   - `ggml.c:1586` — `ctx = GGML_MALLOC(sizeof(struct ggml_context))` (libc `malloc`, not the hook).
   - `ggml.c:1597` — `ctx->mem_buffer = ggml_aligned_malloc(mem_size)` — but this is a *separate* `ggml_aligned_malloc` call inside `ggml.c`, NOT routed through the backend buffer-type hook. It's part of the GGML core's context allocator, not the CPU backend's.
   - Tensor structs (~100 bytes each × N tensors) live in this context pool. Small in absolute terms (sub-GB for any reasonable model) but worth noting it's outside Stage 1's control.

7. **MoE expert weights (Qwen3-30B-A3B has 128 routed experts, top-8)** — *follows model-weights rules.*
   - Expert weight tensors are loaded through the same `create_tensor()` flow at `llama-model-loader.cpp:1045-1210`. They get the same buffer-type as other weights for the layer.
   - Under default mmap: not visible to the hook (same wrapper path as regular weights).
   - Under `--no-mmap`: visible to the hook (same `ggml_aligned_malloc()` path).
   - **Router / gating tensor** (the `expert_ids` indices produced per token): created as a *compute graph* tensor, lives in compute scratch, visible to the hook (4 above).

### Source-trace summary table

| Memory class | Default mmap | `--no-mmap` | Visible to Stage 1 hook? | Evidence |
|---|---|---|---|---|
| Model weights (regular) | mmap'd file, wrapped via `buffer_from_host_ptr` | anonymous, `ggml_aligned_malloc` | NO (mmap) / YES (--no-mmap) | `llama-model.cpp:1459` vs `:1474` |
| Model weights (MoE expert weights) | same as regular | same as regular | same as regular | `llama-model-loader.cpp:1045-1210` |
| KV cache | `ggml_aligned_malloc` via CPU buft | same | **YES** | `llama-kv-cache.cpp:262`, `ggml-alloc.c:1238` |
| Compute scratch / graph | `ggml_aligned_malloc` via CPU buft | same | **YES** | `llama-context.cpp:1214` |
| Activations | shares compute scratch | shares compute scratch | **YES** | (same as compute scratch) |
| Tensor metadata (struct bodies) | GGML core `ggml_aligned_malloc`, separate pool | same | **NO** | `ggml.c:1586`, `:1597` |
| Expert router / gating tensor | compute scratch | same | **YES** | `llama-context.cpp:1214` |

### Expected rough footprint (Qwen3-30B-A3B Q4_K_M, ~17.3 GiB on disk)

| Memory class | Rough share of resident set | Notes |
|---|---:|---|
| Model weights | 80–90% (~14–16 GiB) | dominates; classification flips between mmap and `--no-mmap` |
| KV cache | 1–8% | grows with context length; pp512 + tg128 is small |
| Compute scratch / activations | 1–5% | per-token graph allocation |
| Tensor metadata | <1% | thousands of `ggml_tensor` structs, ~100 B each |
| Code / libs / heap / stack | <1% | irrelevant for NUMA placement |

**The headline finding**: in default mmap, the Stage 1 hook would control **~10–15% of resident memory** at most (KV + scratch only). Model weights — the 80%+ majority — sit in mmap'd VMAs that bypass the hook entirely. Under `--no-mmap` the hook controls **~95%+**.

### Implications for Stage 1 PoC scope

- The Stage 1 PoC writeup must be precise: under default mmap, "I pinned KV cache + compute scratch + activations" — not "I pinned the model."
- The cleanest narrative runs Stage 1 with `--no-mmap`, where the hook genuinely covers the bulk of memory. The cost: model load is slower (file read → memcpy → anon buffer) and total RSS goes up modestly (no page-cache sharing if multiple processes load the same file).
- For a hypothetical Stage 2 that wants to pin mmap'd model weights too: hook would need to land at `ggml/src/ggml-backend.cpp:618-620` where `ggml_backend_buffer_from_ptr` wraps the host pointer. That's a separate touch point and a separate PoC; intentionally out of Stage 1 scope.

---

## Part B — Empirical measurement protocol

To validate the source trace, the bench box runs `measure.sh` (sibling file) which:

1. Loads Qwen3-30B-A3B Q4_K_M via `llama-cli` with a short prompt so all the alloc paths fire.
2. SIGSTOPs the process after warmup (RSS stable).
3. Captures `/proc/<pid>/{maps,smaps,numa_maps,status}`.
4. Resumes and exits cleanly.
5. Repeats with `--no-mmap`.

After both runs:
- `mem_breakdown.py --both` (in `scripts/`) does the headline VMA categorization (`model_mmap` / `anon` / `code_libs` / `special`).
- `snapshot_numa_maps.py --pid <pid>` would have been captured live, but for Stage 0.5 we use offline copies of the snapshots so the bench box can be torn down between sub-runs.

### Tables to populate (run on bench box, paste numbers here)

**Table B-1: VMA-categorized RSS, default `mmap`**

| Category | VMA count | Total RSS | % of RSS | Notes |
|---|---:|---:|---:|---|
| model_mmap (file-backed, `.gguf`) | TBD | TBD GiB | TBD % | expected: dominant share |
| anon (KV + scratch + heap) | TBD | TBD GiB | TBD % | expected: 1–8 % |
| code_libs (binary + .so) | TBD | TBD MiB | TBD % | expected: <1 % |
| special (`[heap]`, `[stack]`, etc.) | TBD | TBD MiB | TBD % | <1 % |
| **TOTAL** | | TBD GiB | 100 % | |

**Table B-2: VMA-categorized RSS, `--no-mmap`**

| Category | VMA count | Total RSS | % of RSS | Notes |
|---|---:|---:|---:|---|
| model_mmap | TBD | TBD GiB | TBD % | expected: ~0 (no GGUF mmap) |
| anon | TBD | TBD GiB | TBD % | expected: ~95 % (weights moved here) |
| code_libs | TBD | TBD MiB | TBD % | <1 % |
| special | TBD | TBD MiB | TBD % | <1 % |
| **TOTAL** | | TBD GiB | 100 % | |

**Table B-3: Cross-table delta**

| Quantity | `mmap` | `--no-mmap` | Δ |
|---|---:|---:|---:|
| Total RSS | TBD GiB | TBD GiB | TBD GiB |
| model_mmap bytes | TBD GiB | TBD GiB | TBD GiB |
| anon bytes | TBD GiB | TBD GiB | TBD GiB |
| % subject to Stage 1 hook | TBD % | TBD % | TBD % |

The headline number for the writeup is the last row.

### Per-VMA NUMA placement (validates Stage 0's `--numa distribute` baseline)

Take the top 5 anon VMAs by size from each configuration; cross-reference with `numa_maps.json` for N0/N1 page counts. Goal: show how `--numa distribute` currently splits the resident-anon memory across nodes. This is the "before" picture that Stage 1's mbind PoC will improve on.

| Config | VMA range | Size MiB | N0 pages | N1 pages | % local to N0 |
|---|---|---:|---:|---:|---:|
| mmap | TBD | TBD | TBD | TBD | TBD |
| ... | | | | | |

---

## Done when

- [ ] Tables B-1, B-2, B-3 filled from real measurements on `rding-bench` (or equivalent dual-NUMA host).
- [ ] Per-VMA NUMA table populated for top-5 anon VMAs in both configurations.
- [ ] If empirical % matches source-trace prediction within ±5 percentage points: source trace validated; cite this writeup from Stage 1 PoC design.
- [ ] If empirical disagrees by >5 pp: investigate which allocation path was missed in the trace (likely candidates: tensor metadata sub-pool, libc thread arenas, jemalloc/tcmalloc if linked) and document the gap before proceeding to Stage 1.

The writeup itself is the Stage 0.5 deliverable. It feeds directly into Stage 1's PR description ("here's the audit that justifies running Stage 1 under `--no-mmap`").
