#!/usr/bin/env bash
# snapshot_kernel_counters.sh — capture the kernel counters that reveal
# auto-NUMA-balance activity, THP splits, and load. Output: JSON to stdout.
#
# Use at START and END of each bench variant. Diff = "what the kernel did
# under the hood while my variant was running."
#
# Counters captured (all from /proc):
#   /proc/vmstat:
#     - pgmigrate_success, pgmigrate_fail  : pages the kernel moved
#     - numa_pte_updates, numa_huge_pte_updates : auto-balance scan work
#     - numa_hint_faults, numa_hint_faults_local : auto-balance "is this remote?" checks
#     - numa_pages_migrated                 : pages auto-balance moved
#     - thp_split_pmd, thp_split_page       : THP being broken up (placement precision lost)
#     - thp_collapse_alloc                  : the inverse (rare)
#   /proc/loadavg                           : last-1/5/15-minute avg, used as noise floor
#   /proc/<pid>/status (if --pid given):
#     - voluntary_ctxt_switches, nonvoluntary_ctxt_switches : process-level
#
# Usage:
#   snapshot_kernel_counters.sh                       # system-wide only
#   snapshot_kernel_counters.sh --pid <pid>           # plus process counters
#   snapshot_kernel_counters.sh --pid <pid> --out f.json

set -u

PID=""
OUT="-"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pid) PID="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        -h|--help)
            sed -n '/^# /,/^$/p' "$0" | head -30 | sed 's/^# //'
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

vmstat_get() {
    local key="$1"
    local val
    val=$(awk -v k="$key" '$1==k {print $2}' /proc/vmstat 2>/dev/null)
    if [[ -z "$val" ]]; then
        echo "null"
    else
        echo "$val"
    fi
}

VMSTAT_KEYS=(
    pgmigrate_success
    pgmigrate_fail
    numa_pte_updates
    numa_huge_pte_updates
    numa_hint_faults
    numa_hint_faults_local
    numa_pages_migrated
    thp_split_pmd
    thp_split_page
    thp_collapse_alloc
    pgfault
    pgmajfault
)

# Build JSON
{
    printf '{\n'
    printf '  "schema_version": "1",\n'
    printf '  "captured_at_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "monotonic_ns": %s,\n' "$(awk '{printf "%.0f", $1 * 1e9}' /proc/uptime)"

    # loadavg
    if [[ -r /proc/loadavg ]]; then
        read -r LA1 LA5 LA15 _PROC _LAST < /proc/loadavg
        printf '  "loadavg": { "1m": %s, "5m": %s, "15m": %s },\n' \
            "${LA1}" "${LA5}" "${LA15}"
    fi

    # vmstat block
    printf '  "vmstat": {\n'
    LAST_IDX=$(( ${#VMSTAT_KEYS[@]} - 1 ))
    for i in "${!VMSTAT_KEYS[@]}"; do
        k="${VMSTAT_KEYS[$i]}"
        v=$(vmstat_get "$k")
        if [[ $i -eq $LAST_IDX ]]; then
            printf '    "%s": %s\n' "$k" "$v"
        else
            printf '    "%s": %s,\n' "$k" "$v"
        fi
    done
    printf '  }'

    # per-process block
    if [[ -n "${PID}" ]]; then
        if [[ -r "/proc/${PID}/status" ]]; then
            VCS=$(awk '/^voluntary_ctxt_switches/ {print $2}'    "/proc/${PID}/status")
            NCS=$(awk '/^nonvoluntary_ctxt_switches/ {print $2}' "/proc/${PID}/status")
            RSS=$(awk '/^VmRSS/ {print $2}'                       "/proc/${PID}/status")
            printf ',\n  "process": {\n'
            printf '    "pid": %s,\n' "${PID}"
            printf '    "voluntary_ctxt_switches": %s,\n' "${VCS:-null}"
            printf '    "nonvoluntary_ctxt_switches": %s,\n' "${NCS:-null}"
            printf '    "vmrss_kb": %s\n' "${RSS:-null}"
            printf '  }'
        else
            printf ',\n  "process": { "pid": %s, "error": "no /proc/%s/status" }' \
                "${PID}" "${PID}"
        fi
    fi
    printf '\n}\n'
} > /tmp/.snap_kctr.$$

if [[ "${OUT}" == "-" ]]; then
    cat /tmp/.snap_kctr.$$
else
    mv /tmp/.snap_kctr.$$ "${OUT}"
fi
rm -f /tmp/.snap_kctr.$$
