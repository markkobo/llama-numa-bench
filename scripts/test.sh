#!/usr/bin/env bash
# test.sh — run test-backend-ops (contribution-policy requirement when ggml is touched).
# By default runs the CPU backend; pass --cuda to also run CUDA if built.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
TBO="${BUILD_DIR}/bin/test-backend-ops"

RUN_CUDA=0
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [--cuda] [-- <extra args passed to test-backend-ops>]

Runs ${TBO} on the CPU backend. Use --cuda to also run on the CUDA backend.
Anything after -- is passed through (e.g. -o MUL_MAT to test a single op).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda) RUN_CUDA=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ ! -x "${TBO}" ]]; then
    echo "test-backend-ops not built. Run ./build.sh first." >&2
    exit 1
fi

run_backend() {
    local backend="$1"
    local label="$2"
    echo
    echo "================================================================"
    echo "  test-backend-ops [${label}]"
    echo "================================================================"
    if "${TBO}" -b "${backend}" "${EXTRA_ARGS[@]}"; then
        echo "==> ${label}: PASS"
        return 0
    else
        echo "==> ${label}: FAIL" >&2
        return 1
    fi
}

FAILED=0
run_backend "CPU" "CPU backend" || FAILED=$((FAILED + 1))

if [[ ${RUN_CUDA} -eq 1 ]]; then
    if "${TBO}" --list-backends 2>&1 | grep -qi cuda; then
        run_backend "CUDA0" "CUDA backend" || FAILED=$((FAILED + 1))
    else
        echo "==> CUDA backend not available in this build. Re-run ./build.sh --cuda." >&2
        FAILED=$((FAILED + 1))
    fi
fi

echo
if [[ ${FAILED} -eq 0 ]]; then
    echo "==> all backends PASS"
    exit 0
else
    echo "==> ${FAILED} backend(s) FAILED" >&2
    exit 1
fi
