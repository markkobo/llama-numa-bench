#!/usr/bin/env bash
# Stage 0.5 measurement protocol — runs on the bench box (dual-NUMA Linux,
# llama.cpp built, Qwen3-30B-A3B GGUF in ~/models/).
#
# For each of the two configurations (default mmap, --no-mmap):
#   1. Launch llama-cli with a short prompt so all alloc paths fire
#   2. Wait for RSS to stabilize (model loaded + first tokens decoded)
#   3. SIGSTOP the process
#   4. Capture /proc/<pid>/{maps,smaps,numa_maps,status,statm}
#   5. Resume and let it exit cleanly
#
# After both runs, point mem_breakdown.py at the binary/model to populate
# the audit tables in README.md.

set -euo pipefail

# Locate the bench repo from this script's location, regardless of where it's run from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$SCRIPT_DIR/snapshots-$(date -u +%Y%m%d-%H%M%S)"

LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin/llama-cli}"
MODEL="${MODEL:-$HOME/models/Qwen3-30B-A3B-Q4_K_M.gguf}"
PROMPT="${PROMPT:-The quick brown fox jumps over the lazy dog.}"
N_TOKENS="${N_TOKENS:-32}"
SETTLE_SECS="${SETTLE_SECS:-60}"   # after launch, wait for RSS to be stable
SETTLE_MIN_GIB="${SETTLE_MIN_GIB:-10}"  # require ≥ this RSS before snapshotting

if [[ ! -x "$LLAMA_BIN" ]]; then
    echo "ERR: llama-cli not at $LLAMA_BIN (override with LLAMA_BIN=)" >&2
    exit 2
fi
if [[ ! -f "$MODEL" ]]; then
    echo "ERR: model not at $MODEL (override with MODEL=)" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"
echo "=== output dir: $OUT_DIR ==="
echo

# --- env snapshot ---
{
    echo "timestamp_utc: $(date -u --iso-8601=seconds)"
    echo "host:          $(hostname)"
    echo "kernel:        $(uname -srm)"
    echo "git_sha:       $(cd "${HOME}/llama.cpp" 2>/dev/null && git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "model:         $MODEL"
    echo "model_sha:     $(sha256sum "$MODEL" 2>/dev/null | awk '{print $1}' || echo unknown)"
    echo "model_size:    $(stat -c %s "$MODEL") bytes"
    echo "n_tokens:      $N_TOKENS"
    echo "prompt:        ${PROMPT}"
    echo
    echo "--- numactl --hardware ---"
    numactl --hardware 2>&1 | head -20 || echo "(numactl unavailable)"
    echo
    echo "--- THP / numa_balancing ---"
    echo "thp_enabled:      $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo unavailable)"
    echo "numa_balancing:   $(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo unavailable)"
} > "$OUT_DIR/env.txt"

run_one() {
    local label="$1"; shift
    local extra_args=("$@")
    local subdir="$OUT_DIR/$label"
    mkdir -p "$subdir"

    echo "=== variant: $label (extra args: ${extra_args[*]:-(none)}) ==="
    # Launch in background; redirect outputs but keep them for archive
    "$LLAMA_BIN" -m "$MODEL" -p "$PROMPT" -n "$N_TOKENS" -no-cnv \
        "${extra_args[@]}" \
        > "$subdir/stdout.log" 2> "$subdir/stderr.log" &
    local pid=$!
    echo "  pid: $pid"

    # Wait for RSS to be stable AND above the minimum threshold
    echo "  waiting up to ${SETTLE_SECS}s for RSS to stabilize (≥${SETTLE_MIN_GIB} GiB)..."
    local last_rss=0 same=0 elapsed=0
    while [[ $elapsed -lt $SETTLE_SECS ]]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "  ERR: process exited before RSS stabilized (see $subdir/stderr.log)" >&2
            return 1
        fi
        local rss_kb
        rss_kb=$(awk '/^VmRSS:/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
        local rss_gib=$(( rss_kb / 1024 / 1024 ))
        if [[ "$rss_kb" -eq "$last_rss" && "$rss_gib" -ge "$SETTLE_MIN_GIB" ]]; then
            same=$((same + 1))
            if [[ $same -ge 3 ]]; then
                echo "  stable at $((rss_kb / 1024)) MiB RSS"
                break
            fi
        else
            same=0
        fi
        last_rss=$rss_kb
        sleep 2
        elapsed=$((elapsed + 2))
    done

    # Freeze and snapshot
    echo "  SIGSTOP and snapshot"
    kill -STOP "$pid"
    sleep 1

    for f in maps smaps numa_maps status statm cmdline; do
        cp "/proc/$pid/$f" "$subdir/proc.$f" 2>/dev/null \
            || echo "  (couldn't copy /proc/$pid/$f)" >&2
    done
    # smaps_rollup is the headline summary
    cp "/proc/$pid/smaps_rollup" "$subdir/proc.smaps_rollup" 2>/dev/null || true

    # Per-thread snapshots for the snapshot_thread_domains tool if needed later
    mkdir -p "$subdir/tasks"
    for tid in "/proc/$pid/task"/*; do
        local tname=$(basename "$tid")
        cp "$tid/status" "$subdir/tasks/$tname.status" 2>/dev/null || true
        cp "$tid/stat"   "$subdir/tasks/$tname.stat"   2>/dev/null || true
    done

    # Resume and let it finish
    echo "  SIGCONT and let exit"
    kill -CONT "$pid"
    wait "$pid" 2>/dev/null || true
    echo "  done"
    echo
}

# --- run both variants ---
run_one "mmap"     # default
run_one "no-mmap" --no-mmap

# --- summarize on the spot ---
echo "=== quick rollup ==="
for v in mmap no-mmap; do
    if [[ -f "$OUT_DIR/$v/proc.smaps_rollup" ]]; then
        echo "--- $v ---"
        head -10 "$OUT_DIR/$v/proc.smaps_rollup"
        echo
    fi
done

echo "=== next: analyze ==="
echo "1. Run mem_breakdown.py against EACH captured run:"
echo "     ./scripts/mem_breakdown.py --bin $LLAMA_BIN --model $MODEL --no-mmap"
echo "     ./scripts/mem_breakdown.py --bin $LLAMA_BIN --model $MODEL"
echo "   (these spawn fresh processes for self-categorization; the offline"
echo "   snapshots above are kept for archival comparison and per-VMA detail.)"
echo
echo "2. Compare numbers against Part A predictions in"
echo "     stage-0.5-alloc-audit/README.md"
echo
echo "3. Populate Tables B-1, B-2, B-3 with measurements + commit + push."
echo
echo "Snapshots committed at: $OUT_DIR"
