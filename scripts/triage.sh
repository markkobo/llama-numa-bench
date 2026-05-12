#!/usr/bin/env bash
# triage.sh — run on a bench host after an interrupted Stage 0 run.
# Read-only; reports state without changing anything.

set -u

echo "===================== HOST & UPTIME ====================="
hostname
date -u --iso-8601=seconds
uptime
echo

echo "===================== RUNNING PROCESSES ====================="
# tmux / screen sessions
echo "--- tmux ---"
tmux ls 2>&1 || true
echo "--- screen ---"
screen -ls 2>&1 || true
echo
echo "--- llama-bench / bench.sh / perf processes ---"
pgrep -af "llama-bench|bench\.sh|perf stat|llama-cli" 2>&1 || echo "(none running)"
echo
echo "--- top memory consumers ---"
ps -eo pid,user,pcpu,pmem,rss,cmd --sort=-rss 2>/dev/null | head -10 || true
echo

echo "===================== DISK / MEMORY ====================="
df -h "${HOME}" / 2>/dev/null | grep -vE "^tmpfs|^devtmpfs" || true
echo
free -h
echo

echo "===================== LATEST BENCH RESULTS ====================="
# Find results dir under ~/llama.cpp/main/results or ~/llama-numa-bench/results
RESULTS_ROOT_CANDIDATES=(
    "${HOME}/llama.cpp/main/results"
    "${HOME}/llama-numa-bench/results"
)
RD=""
for root in "${RESULTS_ROOT_CANDIDATES[@]}"; do
    if [[ -d "$root" ]]; then
        latest=$(ls -td "$root"/baseline-* 2>/dev/null | head -1)
        if [[ -n "$latest" ]]; then
            RD="$latest"; break
        fi
    fi
done

if [[ -z "$RD" ]]; then
    echo "(no baseline-* results dir found in expected locations)"
else
    echo "--- latest results dir: $RD ---"
    ls -la "$RD"
    echo
    echo "--- env.txt (head) ---"
    head -20 "$RD/env.txt" 2>/dev/null || echo "(no env.txt)"
    echo
    echo "--- per-variant completion ---"
    for v in "$RD"/*/; do
        [[ -d "$v" ]] || continue
        name=$(basename "$v")
        bj="$v/bench.jsonl"
        if [[ -s "$bj" ]]; then
            lines=$(wc -l < "$bj")
            bytes=$(stat -c %s "$bj")
            mtime=$(stat -c %y "$bj" | cut -d. -f1)
            echo "  DONE   $name  ($lines jsonl lines, $bytes bytes, modified $mtime)"
        elif [[ -e "$bj" ]]; then
            mtime=$(stat -c %y "$bj" | cut -d. -f1)
            echo "  EMPTY  $name  (created $mtime, may be the variant that died)"
        else
            echo "  UNRUN  $name  (no bench.jsonl)"
        fi
    done
    echo
    echo "--- snapshot files present per variant ---"
    for v in "$RD"/*/; do
        [[ -d "$v" ]] || continue
        name=$(basename "$v")
        files=$(ls "$v"/*.json "$v"/*.jsonl 2>/dev/null | xargs -n1 basename 2>/dev/null | sort | tr '\n' ' ')
        echo "  $name : ${files:-(none)}"
    done
    echo
    echo "--- stderr tail of empty/incomplete variants ---"
    for v in "$RD"/*/; do
        [[ -d "$v" ]] || continue
        name=$(basename "$v")
        bj="$v/bench.jsonl"
        if [[ ! -s "$bj" && -s "$v/stderr.log" ]]; then
            echo
            echo "=== $name/stderr.log (last 40 lines) ==="
            tail -40 "$v/stderr.log"
        fi
    done
fi
echo

echo "===================== KERNEL OOM / ERRORS ====================="
# Try dmesg (may need sudo on some setups)
if dmesg -T 2>/dev/null | tail -50 | grep -qiE "killed process|oom|out of memory"; then
    echo "--- OOM/killed entries from dmesg ---"
    dmesg -T 2>/dev/null | grep -iE "killed process|oom|out of memory" | tail -10
else
    echo "(no OOM in dmesg — or dmesg requires sudo)"
fi
echo
echo "--- journalctl last hour, error+ priority ---"
journalctl --since "1 hour ago" -p err --no-pager 2>/dev/null | tail -20 || echo "(journalctl unavailable)"
echo

echo "===================== NUMA / THP CONTEXT ====================="
numactl --hardware 2>&1 | head -8 || echo "(no numactl)"
echo
echo "THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo unavailable)"
echo "numa_balancing: $(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo unavailable)"
echo

echo "===================== DONE ====================="
echo "If a variant is EMPTY (created but empty bench.jsonl), the bench died"
echo "during that variant. Look at its stderr.log above for the cause."
echo
echo "Resumability: re-running ./main/scripts/bench.sh with the same --model"
echo "skips variants whose bench.jsonl is non-empty. Delete an empty variant's"
echo "directory (rm -rf $RD/EMPTY_VARIANT_NAME) to force its re-run."
