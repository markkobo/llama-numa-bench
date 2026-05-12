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
VARIANTS_FILTER=""

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
  --smoke              Quick mode: reps=5, no perf. For harness validation only.
  --no-perf            Skip perf stat side-recording (useful if perf perms unavailable).
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
    echo "==> SMOKE mode: reps=5, no perf"
    REPS=5
    SKIP_PERF=1
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
echo "==> running variants: ${SELECTED[@]/#/${VARIANT_NAMES[}}"
echo "==> reps=${REPS}, pp=${PP}, tg=${TG}, perf=${HAVE_PERF}, numactl=${HAVE_NUMACTL}"
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
    if "${full_cmd[@]}" > "${bench_out}" 2> "${stderr_out}"; then
        local dt=$(( $(date +%s) - t0 ))
        echo "  [${name}] done in ${dt}s"
    else
        local rc=$?
        local dt=$(( $(date +%s) - t0 ))
        echo "  [${name}] FAILED rc=${rc} after ${dt}s (see ${stderr_out})" >&2
        # Don't bail — let other variants try
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
