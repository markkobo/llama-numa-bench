#!/usr/bin/env python3
"""
snapshot_thread_domains.py — per-thread NUMA placement snapshot.

For a running PID, walks /proc/<pid>/task/<tid>/ and reports per-thread:
  - tid, comm (thread name)
  - Cpus_allowed_list (the affinity mask actually in effect)
  - last running CPU (from /proc/<pid>/task/<tid>/stat field 39)
  - inferred NUMA domain (majority of allowed CPUs; "split" if multi-node)
  - voluntary + nonvoluntary context switches (cumulative since thread start —
    proxy for kernel-initiated migration pressure)

Use during Stage 0: take one snapshot per variant after warmup. Answers
"did `--numa isolate` actually isolate the workers?"

Reads only /proc/* and /sys/*. No code changes to llama.cpp. Stdlib only.

Output: JSON to stdout (one document; not JSONL).

Usage:
    snapshot_thread_domains.py --pid <pid>
    snapshot_thread_domains.py --pgrep llama-bench
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def read_node_cpulists():
    """Return dict {node_id: set(cpu_ids)} for all NUMA nodes."""
    out = {}
    base = Path("/sys/devices/system/node")
    if not base.is_dir():
        return out
    for d in sorted(base.glob("node[0-9]*")):
        m = re.match(r"node(\d+)$", d.name)
        if not m:
            continue
        node_id = int(m.group(1))
        try:
            cpulist = (d / "cpulist").read_text().strip()
        except OSError:
            continue
        cpus = set()
        for token in cpulist.split(","):
            if "-" in token:
                a, b = token.split("-")
                cpus.update(range(int(a), int(b) + 1))
            elif token:
                cpus.add(int(token))
        out[node_id] = cpus
    return out


def parse_cpu_list(s):
    """Parse '0-3,7,9-11' into a set."""
    cpus = set()
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-")
            cpus.update(range(int(a), int(b) + 1))
        else:
            cpus.add(int(token))
    return cpus


def thread_info(pid, tid, node_cpulists):
    """Return dict for one thread, or None if it vanished mid-read."""
    base = Path(f"/proc/{pid}/task/{tid}")
    try:
        status = (base / "status").read_text()
    except OSError:
        return None

    info = {"tid": tid}
    for line in status.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if k == "Name":
            info["comm"] = v
        elif k == "Cpus_allowed_list":
            info["allowed_cpu_list"] = v
            info["allowed_cpus"] = sorted(parse_cpu_list(v))
        elif k == "voluntary_ctxt_switches":
            info["voluntary_ctxt_switches"] = int(v)
        elif k == "nonvoluntary_ctxt_switches":
            info["nonvoluntary_ctxt_switches"] = int(v)
        elif k == "State":
            info["state"] = v.split()[0]
        elif k == "VmRSS":
            info["vmrss_kb"] = int(v.split()[0])
    # stat: last cpu is field 39 (0-indexed: 38). comm is in parens and may
    # contain spaces; parse by finding the final ')'.
    try:
        stat_raw = (base / "stat").read_text()
    except OSError:
        return info
    rp = stat_raw.rfind(")")
    fields = stat_raw[rp + 2:].split() if rp >= 0 else []
    # After comm: state, ppid, pgrp, ... -- field index (1-based after comm) =>
    # last cpu is original-stat field 39 = post-comm index 36 (0-based).
    if len(fields) > 36:
        try:
            info["last_cpu"] = int(fields[36])
        except ValueError:
            pass

    # Infer NUMA domain from allowed_cpus
    if node_cpulists and info.get("allowed_cpus") is not None:
        per_node = {}
        for nid, ncpus in node_cpulists.items():
            n = len(set(info["allowed_cpus"]) & ncpus)
            if n:
                per_node[nid] = n
        info["per_domain_cpus"] = per_node
        if per_node:
            # majority node
            best = max(per_node.items(), key=lambda kv: kv[1])
            info["inferred_domain"] = best[0]
            info["domain_pinning_strict"] = (len(per_node) == 1)
        else:
            info["inferred_domain"] = None
            info["domain_pinning_strict"] = False

        # which domain is last_cpu on?
        if "last_cpu" in info:
            for nid, ncpus in node_cpulists.items():
                if info["last_cpu"] in ncpus:
                    info["last_cpu_domain"] = nid
                    break

    return info


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pid", type=int)
    g.add_argument("--pgrep", help="pgrep pattern (uses pgrep -nf to find PID)")
    p.add_argument("--out", default="-", help="output file ('-' = stdout)")
    args = p.parse_args()

    if args.pid:
        pid = args.pid
    else:
        try:
            pid = int(subprocess.check_output(
                ["pgrep", "-nf", args.pgrep]).decode().strip().split("\n")[0])
        except subprocess.CalledProcessError:
            print(f"no process matched pgrep '{args.pgrep}'", file=sys.stderr)
            sys.exit(2)

    task_dir = Path(f"/proc/{pid}/task")
    if not task_dir.is_dir():
        print(f"no /proc/{pid}/task — pid gone or unprivileged", file=sys.stderr)
        sys.exit(2)

    node_cpulists = read_node_cpulists()

    threads = []
    for entry in sorted(task_dir.iterdir(), key=lambda e: int(e.name)):
        try:
            tid = int(entry.name)
        except ValueError:
            continue
        info = thread_info(pid, tid, node_cpulists)
        if info:
            threads.append(info)

    # Aggregate: count of threads per inferred domain
    domain_counts = {}
    strict_count = 0
    for t in threads:
        d = t.get("inferred_domain")
        if d is not None:
            domain_counts[d] = domain_counts.get(d, 0) + 1
        if t.get("domain_pinning_strict"):
            strict_count += 1

    out_doc = {
        "schema_version": "1",
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": pid,
        "n_threads": len(threads),
        "topology": {
            "n_domains": len(node_cpulists),
            "domain_cpu_counts": {str(k): len(v) for k, v in node_cpulists.items()},
        },
        "summary": {
            "threads_per_domain": {str(k): v for k, v in domain_counts.items()},
            "strictly_pinned_threads": strict_count,
            "split_pinned_threads": len(threads) - strict_count - sum(
                1 for t in threads if t.get("inferred_domain") is None
            ),
        },
        "threads": threads,
    }

    text = json.dumps(out_doc, indent=2)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
