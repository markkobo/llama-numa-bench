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

## Part A2 — Empirical surprises that contradict Part A

**Measured 2026-06-04 on `rding-bench` (EPYC 9R14, 128 vCPU, dual-NUMA, 256 GiB)** using `scripts/mem_breakdown.py --both` on Qwen3-30B-A3B Q4_K_M, llama.cpp @ `8e1f9d083`, llama-cli with `-p 32 -n 8 -no-cnv`. Raw output archived under `raw/`.

### Surprise 1: a 17 GiB anonymous VMA exists under default mmap

Part A predicted that default mmap would keep almost all the resident model bytes in the file-backed mapping, with only KV + scratch (~1–8% of RSS) showing as anonymous. **Empirically wrong**:

| Configuration | model_mmap RSS | anon RSS | Total RSS |
|---|---:|---:|---:|
| default `mmap` | 17,583 MiB | **17,282 MiB** | 34,931 MiB |
| `--no-mmap` | 0 MiB | 21,550 MiB | 21,621 MiB |

Under default mmap there is a single **17.3 GiB anonymous VMA** at `0x72304183a000-0x723491e00000` that Part A's source trace did not predict. Its size is approximately the model weights' size, but the file-backed mmap *also* shows full resident pages. So the process is paying for ~17 GiB of unique anon memory on top of the shared file mapping.

Candidate explanations (not yet validated against source):

- **The mmap-wrap fast path may be disabled in this configuration**: `llama-model.cpp:1457-1465` requires `ml.use_mmap && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft` *all* true to skip the anon copy. If any condition is false the fall-through allocates anon and *also* memcpys from the mmap. The mmap would still be charged to RSS via the file mapping, and the anon would hold its own copy.
- **A pre-allocated workspace** sized to the model (compute scratch worst-case, expert dequant cache, or similar) that I missed in the source trace.

**Implication for Stage 1**: the Part A claim "default mmap leaves only 10–15% of RSS visible to the hook" is wrong. Empirically the hook would control ~50% under default mmap (the 17.3 GiB anon block + smaller anon VMAs). The Part A claim for `--no-mmap` (~95%+) holds at 99.7%.

**Open question — Stage 0.6 candidate**: trace where that 17.3 GiB anon comes from. Either a missed allocator path or a runtime-mode condition (e.g., the host-ptr fast path is gated off here) is responsible. Until that's resolved, Stage 1 should still prefer `--no-mmap` for the cleanest narrative, but the default-mmap result is also a legitimate Stage 1 target with the corrected "hook covers ~50%" framing.

### Surprise 2: total RSS under `mmap` is *larger* than under `--no-mmap`

| Configuration | Total RSS |
|---|---:|
| default `mmap` | 34.9 GiB |
| `--no-mmap` | 21.6 GiB |

Naïvely, mmap should be smaller (page cache shared with the OS). Here mmap is 13.3 GiB **bigger** because the process has both the file-backed RSS *and* the unexplained anon copy. This is consistent with hypothesis 1 above (mmap-wrap fallback to anon copy while still keeping the file mmap). It's also independently a useful finding: `--no-mmap` is *the* lower-RSS option for this build, contrary to common llama.cpp guidance.

### NUMA placement of the resident anon (per-VMA, top-1)

| Configuration | Top anon VMA size | N0 pages | N1 pages | % local to N0 |
|---|---:|---:|---:|---:|
| `mmap` | 17,278 MiB | ~3.6 M | ~0.8 M | 81% |
| `--no-mmap` | 21,542 MiB | ~0.5 M | ~5.0 M | 10% |

Both runs happened to land first-touch placement on different nodes (the `mmap` run started on a node-0 CPU, the `--no-mmap` run on a node-1 CPU). Either way the *split is heavily lopsided* — the kernel's default first-touch isn't producing a balanced placement under load. This is exactly the "before" picture that Stage 1's mbind PoC will improve on.

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

**Table B-1: VMA-categorized RSS, default `mmap`** (measured 2026-06-04)

| Category | VMA count | Total RSS | % of RSS | Notes |
|---|---:|---:|---:|---|
| model_mmap (file-backed, `.gguf`) | 2 | 17,583 MiB | 50.3 % | the GGUF file-backed mapping |
| anon (KV + scratch + ?17 GiB unexplained) | 270 | 17,282 MiB | 49.5 % | dominated by ONE 17.3 GiB VMA, see Part A2 surprise #1 |
| code_libs (binary + .so) | 76 | 13 MiB | 0.0 % | negligible |
| special (`[heap]`, `[stack]`, etc.) | 5 | 53 MiB | 0.2 % | negligible |
| **TOTAL** | 353 | **34,931 MiB** | 100 % | larger than `--no-mmap` (see surprise #2) |

**Table B-2: VMA-categorized RSS, `--no-mmap`** (measured 2026-06-04)

| Category | VMA count | Total RSS | % of RSS | Notes |
|---|---:|---:|---:|---|
| model_mmap | 0 | 0 MiB | 0 % | as expected, no GGUF mmap |
| anon | 528 | 21,550 MiB | 99.7 % | weights + KV + scratch all anon-backed |
| code_libs | 76 | 14 MiB | 0.1 % | |
| special | 5 | 57 MiB | 0.3 % | |
| **TOTAL** | 609 | **21,621 MiB** | 100 % | |

**Table B-3: Cross-table delta**

| Quantity | `mmap` | `--no-mmap` | Δ |
|---|---:|---:|---:|
| Total RSS | 34.9 GiB | 21.6 GiB | **-13.3 GiB** (no-mmap is *smaller*) |
| model_mmap bytes | 17.6 GiB | 0 | -17.6 GiB |
| anon bytes | 17.3 GiB | 21.6 GiB | +4.3 GiB |
| % subject to Stage 1 hook | **49.5 %** | **99.7 %** | +50.2 pp |

The headline number for the writeup is the last row.

The Part A prediction ("10–15 % under default mmap") was wrong; the measured ~50 % means a Stage 1 PoC under default mmap is also legitimate, not just a `--no-mmap`-only story. But the source-trace gap behind that 17 GiB anon block is unresolved (see Part A2 surprise #1).

### Per-VMA NUMA placement (validates Stage 0's `--numa distribute` baseline)

Take the top 5 anon VMAs by size from each configuration; cross-reference with `numa_maps.json` for N0/N1 page counts. Goal: show how `--numa distribute` currently splits the resident-anon memory across nodes. This is the "before" picture that Stage 1's mbind PoC will improve on.

**Aggregate per-category NUMA pages** (measured 2026-06-04):

| Category | `mmap` N0 / N1 (4K pages) | `--no-mmap` N0 / N1 |
|---|---|---|
| model_mmap | 4,501,338 / 0 (100% N0) | 0 / 0 |
| anon | 3,589,959 / 834,278 (81% N0) | 496,107 / 5,020,829 (10% N0) |
| code_libs | 2,890 / 492 | 2,972 / 672 |
| special | 13,561 / 0 | 0 / 14,551 |

**Top anonymous VMA per run** (dominates the breakdown):

| Config | VMA range | Size MiB | Top-VMA placement implication |
|---|---|---:|---|
| `mmap` | `0x72304183a000-0x723491e00000` | 17,278 | one big anon block; placement followed first-touch on N0 |
| `--no-mmap` | `0x711839214000-0x711d98202000` | 21,542 | one big anon block; first-touch on N1 |

Both runs are heavily lopsided (the single dominant anon VMA is on one node, not balanced). First-touch alone is producing 80/20 or 10/90 splits depending on which CPU the launcher happened to land on. This is the empirical "before" picture for Stage 1's mbind PoC: even `--numa distribute` doesn't help when one giant VMA carries most of the resident set.

---

## Done when

- [x] Tables B-1, B-2, B-3 filled from real measurements on `rding-bench` (2026-06-04, raw output `raw/mem_breakdown-20260604-2122.out`).
- [x] Per-VMA NUMA table populated.
- [x] Empirical numbers compared to source-trace predictions.
- [ ] **Open: empirical disagrees with source-trace by >5 pp under default mmap** (Part A predicted 10–15 % hook coverage; measured 49.5 %). Source-trace gap is the unexplained 17 GiB anon VMA. Either a missed allocator path (likely candidate: the host-ptr fast path is being rejected and the alloc-ctx-tensors-from-buft fallback is running) or a runtime workspace I missed in the source. **Stage 0.6 candidate**: run `llama-cli` with `LLAMA_LOG_LEVEL=DEBUG` or instrument to log every backend buffer allocation, identify which call site produces the 17 GiB anon, update Part A. Until that's resolved, Stage 1 should still prefer `--no-mmap` for the cleanest narrative, but the default-mmap result is also legitimate at the corrected ~50 % framing.

The writeup itself is the Stage 0.5 deliverable. It feeds directly into Stage 1's PR description ("here's the audit that justifies running Stage 1 under `--no-mmap`" — or under default mmap with the corrected hook-coverage claim, depending on which configuration we pick after the Stage 0.6 trace-gap resolution).
