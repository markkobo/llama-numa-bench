#!/usr/bin/env bash
# c7a_bootstrap.sh — first-run setup on a fresh AWS c7a.32xlarge (Ubuntu 24.04).
#
# What it does:
#   1. apt-install build deps + perf + numactl + tmux + git
#   2. enable perf for non-root (sets perf_event_paranoid=0)
#   3. clone markkobo/llama.cpp into ~/llama.cpp
#   4. clone markkobo/llama-numa-bench into ~/llama-numa-bench
#   5. symlink scripts into ~/llama.cpp/main/scripts/ so they run as designed
#   6. create ~/models/ for the GGUF file
#   7. print exact next steps
#
# Run on a fresh c7a:
#   curl -sSL https://raw.githubusercontent.com/markkobo/llama-numa-bench/main/scripts/c7a_bootstrap.sh -o /tmp/boot.sh
#   less /tmp/boot.sh         # READ IT before running, even though it's "yours"
#   bash /tmp/boot.sh
#
# (Don't curl|bash without reading — habit of a lifetime, even for your own scripts.)

set -euo pipefail

LLAMA_REPO="${LLAMA_REPO:-https://github.com/markkobo/llama.cpp.git}"
BENCH_REPO="${BENCH_REPO:-https://github.com/markkobo/llama-numa-bench.git}"
LLAMA_DIR="${HOME}/llama.cpp"
BENCH_DIR="${HOME}/llama-numa-bench"
MODELS_DIR="${HOME}/models"
SKIP_APT=0
SKIP_PERF=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [--skip-apt] [--skip-perf]

Bootstraps a fresh c7a.32xlarge (or any Ubuntu 24.04 host) for Stage 0 work.

Options:
  --skip-apt    Skip the system package install (use if already installed).
  --skip-perf   Don't touch /proc/sys/kernel/perf_event_paranoid.
  -h, --help    Show this help.

Idempotent: re-running on a partially-bootstrapped host is safe.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-apt)  SKIP_APT=1; shift ;;
        --skip-perf) SKIP_PERF=1; shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

echo "==> c7a bootstrap starting at $(date -u --iso-8601=seconds)"
echo "    LLAMA_REPO=${LLAMA_REPO}"
echo "    BENCH_REPO=${BENCH_REPO}"
echo "    target dirs: ${LLAMA_DIR}  ${BENCH_DIR}  ${MODELS_DIR}"
echo

# ---- system deps ----
if [[ ${SKIP_APT} -eq 0 ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
        echo "WARN: sudo not found; assuming root or apt unavailable" >&2
    fi
    echo "==> apt-installing packages (sudo will prompt if needed)"
    sudo apt update -q
    sudo apt install -y --no-install-recommends \
        cmake ninja-build build-essential \
        git curl ca-certificates \
        numactl \
        linux-tools-generic "linux-tools-$(uname -r)" \
        tmux \
        python3 python3-pip
    echo "==> apt done"
else
    echo "==> skipping apt (--skip-apt)"
fi
echo

# ---- perf permissions ----
if [[ ${SKIP_PERF} -eq 0 ]]; then
    CURRENT=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "?")
    echo "==> perf_event_paranoid currently: ${CURRENT}"
    if [[ "${CURRENT}" != "0" && "${CURRENT}" != "-1" ]]; then
        echo "    setting to 0 (allows perf stat for unprivileged users)"
        sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'
        # persist across reboots (boot isn't expected, but be tidy)
        sudo sh -c 'echo "kernel.perf_event_paranoid = 0" > /etc/sysctl.d/99-perf.conf'
    else
        echo "    already permissive, leaving as is"
    fi
else
    echo "==> skipping perf perms (--skip-perf)"
fi
echo

# ---- clone repos ----
clone_or_pull() {
    local url="$1" dir="$2"
    if [[ -d "${dir}/.git" ]]; then
        echo "==> ${dir} exists, fetching"
        git -C "${dir}" fetch --all --prune
        local branch
        branch=$(git -C "${dir}" symbolic-ref --short HEAD 2>/dev/null || echo main)
        git -C "${dir}" pull --ff-only origin "${branch}" || \
            echo "    (could not fast-forward; you may have local commits — investigate)"
    else
        echo "==> cloning ${url} -> ${dir}"
        git clone "${url}" "${dir}"
    fi
}

clone_or_pull "${LLAMA_REPO}" "${LLAMA_DIR}"
clone_or_pull "${BENCH_REPO}" "${BENCH_DIR}"
echo

# ---- wire scripts into the llama.cpp checkout ----
SCRIPTS_TARGET="${LLAMA_DIR}/main/scripts"
mkdir -p "${LLAMA_DIR}/main"

if [[ -L "${SCRIPTS_TARGET}" ]]; then
    echo "==> ${SCRIPTS_TARGET} already a symlink -> $(readlink "${SCRIPTS_TARGET}")"
elif [[ -d "${SCRIPTS_TARGET}" && -n "$(ls -A "${SCRIPTS_TARGET}" 2>/dev/null)" ]]; then
    echo "==> ${SCRIPTS_TARGET} is a non-empty directory; leaving alone"
    echo "    (existing scripts override bench-repo versions)"
else
    rm -rf "${SCRIPTS_TARGET}"
    ln -s "${BENCH_DIR}/scripts" "${SCRIPTS_TARGET}"
    echo "==> symlinked ${SCRIPTS_TARGET} -> ${BENCH_DIR}/scripts"
fi
echo

# ---- models dir ----
mkdir -p "${MODELS_DIR}"
echo "==> models dir: ${MODELS_DIR}"
echo

# ---- environment snapshot ----
echo "==> environment summary"
echo "    host:    $(hostname)"
echo "    kernel:  $(uname -srm)"
echo "    cpu:     $(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //')"
echo "    cores:   $(nproc)"
echo "    mem:     $(awk '/MemTotal/ {printf "%.1f GiB\n", $2/1024/1024}' /proc/meminfo)"
echo "    disk:    $(df -h "${HOME}" | awk 'NR==2 {print $4 " free of " $2}')"
if command -v numactl >/dev/null 2>&1; then
    echo "    numa:"
    numactl --hardware | sed 's/^/             /'
fi
echo

# ---- next steps ----
cat <<EOF
================================================================================
bootstrap complete.

next steps (run inside 'tmux new -s bench' so SSH disconnect does not kill the
20-hour bench):

  # 1. download the model (about 5-10 min on AWS bandwidth, ~17.3 GiB)
  curl -L --fail --retry 5 -C - \\
      -o "${MODELS_DIR}/Qwen3-30B-A3B-Q4_K_M.gguf" \\
      "https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf"
  # expected sha256: 9f1a24700a339b09c06009b729b5c809e0b64c213b8af5b711b3dbdfd0c5ba48

  # 2. build llama.cpp (~5-10 min on c7a's 192 cores)
  cd "${LLAMA_DIR}"
  ./main/scripts/build.sh

  # 3. test (required by contribution policy when ggml is touched)
  ./main/scripts/test.sh

  # 4. smoke run (~5 min, validates harness, no perf)
  ./main/scripts/bench.sh --model ~/models/Qwen3-30B-A3B-Q4_K_M.gguf --smoke

  # 5. real Stage 0 (~15-20 hr — keep tmux attached or detach with C-b d)
  ./main/scripts/bench.sh --model ~/models/Qwen3-30B-A3B-Q4_K_M.gguf

  # 6. analyze
  ./main/scripts/analyze.py main/results/baseline-<timestamp>/

  # 7. ship results to the bench repo
  RESULTS_DIR=\$(ls -td main/results/baseline-* | head -1)
  cp -r "\$RESULTS_DIR" "${BENCH_DIR}/results/"
  cd "${BENCH_DIR}"
  git add results/
  git status   # review
  git commit -m "Stage 0 baseline: c7a.32xlarge \$(date -u +%Y-%m-%d)"
  git push

cost reminder: c7a.32xlarge is ~\$5.40/hr on-demand. ~20-hour bench => ~\$108.
================================================================================
EOF
