#!/usr/bin/env bash
# bench.sh — Stage 0 baseline matrix.
# 5 variants × llama-bench (JSONL output) × perf stat side-recording.
# Resumable: variants whose output already exists are skipped.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
LLAMA_BENCH="${BUILD_DIR}/bin/llama-bench"

MODEL=""
REPS=1000
PP=512
TG=128
THREADS=""
RESULTS_DIR=""
SMOKE=0
SKIP_PERF=0
SKIP_INSTRUMENT=0
WARMUP_SEC=60
VARIANTS_FILTER=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") --model PATH [options]

Runs the Stage 0 baseline matrix:
  baseline    no NUMA flag, no numactl
  distribute  llama-bench --numa distribute
  isolate     llama-bench --numa isolate
  interleave  numactl --interleave=all  + llama-bench --numa numactl
  bind0       numactl --cpubind=0 --membind=0 + llama-bench --numa numactl

Each variant's output (JSONL + perf log + meta) lands in
  results/baseline-<utc-timestamp>/<variant>/

Variants whose bench.jsonl already exists are skipped (resumable).

Required:
  --model PATH         Path to a .gguf model.

Options:
  --reps N             llama-bench -r (default: ${REPS}). Each variant runs pp${PP} + tg${TG} N times.
  --pp N               Prompt length for prompt-processing test (default: ${PP}).
  --tg N               Generated tokens for token-generation test (default: ${TG}).
  --threads N          llama-bench -t. Defaults to llama-bench's auto.
  --variants v1,v2     Only run named variants (comma-separated). Default: all 5.
  --smoke              Quick mode: reps=5, no perf, no instrument. For harness validation only.
  --no-perf            Skip perf stat side-recording (useful if perf perms unavailable).
  --no-instrument      Skip mid-variant snapshots (threads/numa_maps/kernel counters).
  --warmup-sec N       Seconds after launching llama-bench before snapshots (default: ${WARMUP_SEC}).
  --results-dir DIR    Override output dir (default: results/baseline-<timestamp>).
  -h, --help           Show this help.

Example:
  $(basename "$0") --model /data/mixtral.Q4_K_M.gguf --reps 1000
  $(basename "$0") --model /data/mixtral.Q4_K_M.gguf --smoke
  $(basename "$0") --model /data/mixtral.Q4_K_M.gguf --variants baseline,distribute
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --reps) REPS="$2"; shift 2 ;;
        --pp) PP="$2"; shift 2 ;;
        --tg) TG="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --variants) VARIANTS_FILTER="$2"; shift 2 ;;
        --smoke) SMOKE=1; shift ;;
        --no-perf) SKIP_PERF=1; shift ;;
        --no-instrument) SKIP_INSTRUMENT=1; shift ;;
        --warmup-sec) WARMUP_SEC="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

# Validate
if [[ -z "${MODEL}" ]]; then
    echo "missing --model" >&2; usage; exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
    echo "model not found: ${MODEL}" >&2; exit 2
fi
if [[ ! -x "${LLAMA_BENCH}" ]]; then
    echo "llama-bench not built at ${LLAMA_BENCH}. Run ./build.sh first." >&2
    exit 2
fi

# Smoke mode overrides
if [[ ${SMOKE} -eq 1 ]]; then
    echo "==> SMOKE mode: reps=5, no perf, no instrument"
    REPS=5
    SKIP_PERF=1
    SKIP_INSTRUMENT=1
fi

# Resolve instrumentation availability
HAVE_INSTRUMENT=0
if [[ ${SKIP_INSTRUMENT} -eq 0 ]]; then
    if [[ -x "${SCRIPT_DIR}/snapshot_thread_domains.py" && \
          -x "${SCRIPT_DIR}/snapshot_numa_maps.py" && \
          -x "${SCRIPT_DIR}/snapshot_kernel_counters.sh" ]]; then
        HAVE_INSTRUMENT=1
    else
        echo "NOTE: snapshot scripts not all executable; mid-variant snapshots disabled." >&2
        echo "      checked: ${SCRIPT_DIR}/snapshot_*" >&2
    fi
fi

# Resolve perf availability
HAVE_PERF=0
if [[ ${SKIP_PERF} -eq 0 ]] && command -v perf >/dev/null 2>&1; then
    if perf stat -e task-clock -- true >/dev/null 2>&1; then
        HAVE_PERF=1
    else
        echo "WARN: perf installed but stat failed (perf_event_paranoid?). Skipping perf side-recording." >&2
        echo "      to allow: sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'" >&2
    fi
fi

# Resolve numactl availability
HAVE_NUMACTL=0
if command -v numactl >/dev/null 2>&1; then
    HAVE_NUMACTL=1
fi

# Results dir
if [[ -z "${RESULTS_DIR}" ]]; then
    TS="$(date -u +%Y%m%d-%H%M%S)"
    RESULTS_DIR="${REPO_ROOT}/main/results/baseline-${TS}"
fi
mkdir -p "${RESULTS_DIR}"

# Variants table: name | numactl_prefix | --numa flag
# (use $'\t' or just rely on awk-style; here we use parallel arrays)
VARIANT_NAMES=(baseline    distribute       isolate          interleave                              bind0)
VARIANT_PFX=(  ""          ""               ""               "numactl --interleave=all"              "numactl --cpubind=0 --membind=0")
VARIANT_NUMA=( ""          "distribute"     "isolate"        "numactl"                               "numactl")

# Filter
SELECTED=()
if [[ -n "${VARIANTS_FILTER}" ]]; then
    IFS=',' read -r -a FILTER_ARR <<< "${VARIANTS_FILTER}"
    for v in "${FILTER_ARR[@]}"; do
        FOUND=0
        for i in "${!VARIANT_NAMES[@]}"; do
            if [[ "${VARIANT_NAMES[$i]}" == "$v" ]]; then
                SELECTED+=("$i"); FOUND=1; break
            fi
        done
        if [[ $FOUND -eq 0 ]]; then
            echo "unknown variant: $v (valid: ${VARIANT_NAMES[*]})" >&2; exit 2
        fi
    done
else
    SELECTED=("${!VARIANT_NAMES[@]}")
fi

PERF_EVENTS="task-clock,context-switches,cpu-migrations,cycles,instructions,branches,branch-misses,node-load-misses,node-loads,dTLB-load-misses,LLC-load-misses,LLC-loads"

# Capture environment up front for reproducibility
META_FILE="${RESULTS_DIR}/env.txt"
{
    echo "timestamp: $(date -u --iso-8601=seconds)"
    echo "host:      $(hostname)"
    echo "kernel:    $(uname -srm)"
    echo "git:       $(cd "${REPO_ROOT}" && git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "model:     ${MODEL}"
    echo "model_sha: $(sha256sum "${MODEL}" 2>/dev/null | awk '{print $1}' || echo unknown)"
    echo "pp:        ${PP}"
    echo "tg:        ${TG}"
    echo "reps:      ${REPS}"
    echo "threads:   ${THREADS:-auto}"
    echo "have_perf: ${HAVE_PERF}"
    echo "have_numactl: ${HAVE_NUMACTL}"
    echo "perf_events: ${PERF_EVENTS}"
    echo
    echo "--- numactl --hardware ---"
    if [[ ${HAVE_NUMACTL} -eq 1 ]]; then
        numactl --hardware 2>&1 || true
    else
        echo "(numactl not installed)"
    fi
    echo
    echo "--- /proc/cpuinfo (head) ---"
    head -25 /proc/cpuinfo
    echo
    echo "--- THP state ---"
    cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo "(unavailable)"
    cat /sys/kernel/mm/transparent_hugepage/defrag  2>/dev/null || echo "(unavailable)"
} > "${META_FILE}"

echo "==> results in: ${RESULTS_DIR}"
SELECTED_NAMES=""
for idx in "${SELECTED[@]}"; do
    SELECTED_NAMES+="${VARIANT_NAMES[$idx]} "
done
echo "==> running variants: ${SELECTED_NAMES}"
echo "==> reps=${REPS}, pp=${PP}, tg=${TG}, perf=${HAVE_PERF}, numactl=${HAVE_NUMACTL}, instrument=${HAVE_INSTRUMENT}"
echo

# Per-variant runner
run_variant() {
    local idx="$1"
    local name="${VARIANT_NAMES[$idx]}"
    local pfx="${VARIANT_PFX[$idx]}"
    local numa="${VARIANT_NUMA[$idx]}"

    local outdir="${RESULTS_DIR}/${name}"
    mkdir -p "${outdir}"
    local bench_out="${outdir}/bench.jsonl"
    local perf_out="${outdir}/perf.log"
    local cmd_out="${outdir}/cmd.txt"
    local stderr_out="${outdir}/stderr.log"

    if [[ -s "${bench_out}" ]]; then
        echo "  [${name}] already done (${bench_out}), skipping. Delete to rerun."
        return 0
    fi

    # Skip variants needing numactl if unavailable
    if [[ -n "${pfx}" && ${HAVE_NUMACTL} -eq 0 ]]; then
        echo "  [${name}] skipped: requires numactl (not installed)"
        return 0
    fi

    # Build the command
    local bench_cmd=()
    bench_cmd+=("${LLAMA_BENCH}" -m "${MODEL}" -p "${PP}" -n "${TG}" -r "${REPS}" --output jsonl)
    if [[ -n "${numa}" ]]; then
        bench_cmd+=(--numa "${numa}")
    fi
    if [[ -n "${THREADS}" ]]; then
        bench_cmd+=(-t "${THREADS}")
    fi

    local full_cmd
    if [[ -n "${pfx}" ]]; then
        # Tokenize prefix safely
        # shellcheck disable=SC2206
        local pfx_arr=(${pfx})
        full_cmd=("${pfx_arr[@]}" "${bench_cmd[@]}")
    else
        full_cmd=("${bench_cmd[@]}")
    fi

    if [[ ${HAVE_PERF} -eq 1 ]]; then
        full_cmd=(perf stat -e "${PERF_EVENTS}" -o "${perf_out}" -- "${full_cmd[@]}")
    fi

    # Record the exact command
    {
        echo "# variant: ${name}"
        echo "# numactl prefix: '${pfx}'"
        echo "# --numa: '${numa}'"
        echo "# generated: $(date -u --iso-8601=seconds)"
        echo
        printf '%q ' "${full_cmd[@]}"
        echo
    } > "${cmd_out}"

    echo "  [${name}] starting (logs: ${outdir}/)"
    local t0
    t0=$(date +%s)

    # Snapshot kernel counters BEFORE this variant
    if [[ ${HAVE_INSTRUMENT} -eq 1 ]]; then
        "${SCRIPT_DIR}/snapshot_kernel_counters.sh" --out "${outdir}/kcounters_start.json" || \
            echo "  [${name}] (kcounters_start failed; continuing)" >&2
    fi

    # Launch the bench (perf-wrapped, maybe numactl-prefixed) in background
    "${full_cmd[@]}" > "${bench_out}" 2> "${stderr_out}" &
    local bg_pid=$!

    # Mid-variant snapshots: wait for llama-bench warmup, then capture state
    if [[ ${HAVE_INSTRUMENT} -eq 1 ]]; then
        (
            # Sleep, then find the llama-bench grandchild via pgrep
            sleep "${WARMUP_SEC}"
            # Verify the bench is still running
            if ! kill -0 "${bg_pid}" 2>/dev/null; then
                echo "  [${name}] (bench finished before warmup snapshot)" >&2
                exit 0
            fi
            local llama_pid
            llama_pid=$(pgrep -nf "${LLAMA_BENCH}" 2>/dev/null || true)
            if [[ -z "${llama_pid}" ]]; then
                echo "  [${name}] (could not pgrep llama-bench for snapshot)" >&2
                exit 0
            fi
            "${SCRIPT_DIR}/snapshot_thread_domains.py" --pid "${llama_pid}" \
                --out "${outdir}/threads.json" 2> "${outdir}/threads.err" || \
                echo "  [${name}] (threads snapshot failed)" >&2
            "${SCRIPT_DIR}/snapshot_numa_maps.py" --pid "${llama_pid}" \
                --model "${MODEL}" --out "${outdir}/numa_maps.json" \
                2> "${outdir}/numa_maps.err" || \
                echo "  [${name}] (numa_maps snapshot failed)" >&2
            "${SCRIPT_DIR}/snapshot_kernel_counters.sh" --pid "${llama_pid}" \
                --out "${outdir}/kcounters_mid.json" || \
                echo "  [${name}] (kcounters_mid failed)" >&2
        ) &
        local snap_pid=$!
    fi

    # Wait for the bench
    local rc=0
    wait "${bg_pid}" || rc=$?
    # Wait for the snapshot subshell too (it may have backgrounded; harmless)
    if [[ ${HAVE_INSTRUMENT} -eq 1 ]] && [[ -n "${snap_pid:-}" ]]; then
        wait "${snap_pid}" 2>/dev/null || true
    fi

    # Snapshot kernel counters AFTER this variant
    if [[ ${HAVE_INSTRUMENT} -eq 1 ]]; then
        "${SCRIPT_DIR}/snapshot_kernel_counters.sh" --out "${outdir}/kcounters_end.json" || \
            echo "  [${name}] (kcounters_end failed; continuing)" >&2
    fi

    local dt=$(( $(date +%s) - t0 ))
    if [[ ${rc} -eq 0 ]]; then
        echo "  [${name}] done in ${dt}s"
    else
        echo "  [${name}] FAILED rc=${rc} after ${dt}s (see ${stderr_out})" >&2
        return ${rc}
    fi
}

OVERALL_FAIL=0
for idx in "${SELECTED[@]}"; do
    run_variant "$idx" || OVERALL_FAIL=$((OVERALL_FAIL + 1))
done

echo
echo "==> baseline matrix complete (${OVERALL_FAIL} failures)"
echo "==> next: ./analyze.py ${RESULTS_DIR}"
exit ${OVERALL_FAIL}
