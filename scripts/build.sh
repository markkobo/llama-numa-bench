#!/usr/bin/env bash
# build.sh — configure + build llama.cpp for Stage 0 baseline work.
# Defaults: Release, native arch, LTO, CPU-only. Use --cuda for GPU, --debug for debug builds.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
BUILD_TYPE="Release"
ENABLE_CUDA=0
CLEAN=0
JOBS="$(nproc)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [--cuda] [--debug] [--clean] [--jobs N]

Builds llama.cpp into ${BUILD_DIR}.

Options:
  --cuda      Enable CUDA backend (requires nvcc on PATH or CMAKE_CUDA_COMPILER).
  --debug     Build type Debug (default: Release with LTO + native arch).
  --clean     Wipe build directory before configuring.
  --jobs N    Parallel build jobs (default: ${JOBS}, from nproc).
  -h, --help  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)   ENABLE_CUDA=1; shift ;;
        --debug)  BUILD_TYPE="Debug"; shift ;;
        --clean)  CLEAN=1; shift ;;
        --jobs)   JOBS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

# Pre-flight checks
need() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "missing tool: $1 (install with: $2)" >&2
        return 1
    fi
}
need cmake "apt install cmake" || exit 1
need ninja "apt install ninja-build" || true   # nice to have, falls back to make
need git   "apt install git" || exit 1

if [[ ${ENABLE_CUDA} -eq 1 ]] && ! command -v nvcc >/dev/null 2>&1; then
    echo "WARN: --cuda set but nvcc not on PATH. CMake may still find it via CMAKE_CUDA_COMPILER." >&2
fi

# Optional but useful
if ! command -v perf >/dev/null 2>&1; then
    echo "NOTE: 'perf' not installed. bench.sh will run without it. Install with:" >&2
    echo "      sudo apt install linux-tools-generic linux-tools-\$(uname -r)" >&2
fi
if ! command -v numactl >/dev/null 2>&1; then
    echo "NOTE: 'numactl' not installed. bench.sh's NUMA variants need it:" >&2
    echo "      sudo apt install numactl" >&2
fi

# Clean if requested
if [[ ${CLEAN} -eq 1 && -d "${BUILD_DIR}" ]]; then
    echo "==> wiping ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

# Configure
CMAKE_ARGS=(
    -S "${REPO_ROOT}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DLLAMA_BUILD_TESTS=ON
    -DLLAMA_BUILD_EXAMPLES=ON
    -DLLAMA_BUILD_TOOLS=ON
    -DGGML_NATIVE=ON
)
if [[ "${BUILD_TYPE}" == "Release" ]]; then
    CMAKE_ARGS+=(-DGGML_LTO=ON)
fi
if [[ ${ENABLE_CUDA} -eq 1 ]]; then
    CMAKE_ARGS+=(-DGGML_CUDA=ON)
fi
if command -v ninja >/dev/null 2>&1; then
    CMAKE_ARGS+=(-G Ninja)
fi

echo "==> cmake configure: ${BUILD_TYPE}, cuda=${ENABLE_CUDA}"
cmake "${CMAKE_ARGS[@]}"

# Build
echo "==> cmake build (-j ${JOBS})"
cmake --build "${BUILD_DIR}" -j "${JOBS}"

# Quick sanity
echo "==> built binaries:"
for b in llama-bench llama-cli llama-server test-backend-ops; do
    if [[ -x "${BUILD_DIR}/bin/${b}" ]]; then
        echo "    ${BUILD_DIR}/bin/${b}"
    else
        echo "    MISSING: ${b}" >&2
    fi
done
echo "==> done."
