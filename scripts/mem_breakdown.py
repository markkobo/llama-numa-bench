#!/usr/bin/env python3
"""
mem_breakdown.py — How much of a loaded GGUF model lives in mmap'd file regions
vs anonymous (heap/ggml) buffers? This decides where the mbind PoC needs to land.

Usage:
    mem_breakdown.py --bin ./build/bin/llama-bench --model mixtral.Q4_K_M.gguf
    mem_breakdown.py --bin ./build/bin/llama-bench --model mixtral.Q4_K_M.gguf --no-mmap
    mem_breakdown.py --bin ./build/bin/llama-bench --model m.gguf --bench-args="-p 512 -n 128 -r 1"

What it does:
  1. Spawns the binary loading the model.
  2. Polls /proc/<pid>/status:VmRSS until it stops growing (model loaded, paged in).
  3. SIGSTOPs the process to freeze state.
  4. Reads /proc/<pid>/smaps and categorizes every VMA:
       * "model_mmap"    — file-backed regions whose path matches --model
       * "anon"          — anonymous mappings (heap, ggml_aligned_malloc'd buffers,
                           KV cache, activation scratch)
       * "code_libs"     — file-backed regions for the binary + shared libs
       * "special"       — [heap] [stack] [vdso] [vvar] [vsyscall] etc.
  5. Reads /proc/<pid>/numa_maps (if readable) and tallies per-node page counts
     per category.
  6. SIGTERMs the process and prints a report.

Why this matters:
  llama.cpp mmap's GGUF weight files by default. mbind() applied inside
  ggml_backend_cpu_buffer_type_alloc_buffer() will pin anonymous CPU-backend
  buffers — NOT the mmap'd weights. If "model_mmap" dwarfs "anon" in the
  report, a buffer-allocator mbind PoC will pin only a small slice of the
  resident model. Decide before writing the PoC whether to:
    (a) test only with --no-mmap (model loaded as anon → buffer-allocator pin
        captures everything), or
    (b) add a separate mbind() pass over the mmap'd region after model load.

Author: scaffolding for markkobo/llama-numa-bench. Gitignored copy lives in
main/scripts/ on the llama.cpp checkout.
"""

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

MiB = 1024 * 1024
PAGE_SIZE = os.sysconf("SC_PAGESIZE")


# ---------- /proc parsing ----------

def read_vm_rss_kb(pid: int) -> int:
    """Return VmRSS in kB from /proc/<pid>/status, or -1 if process is gone."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        return -1
    return -1


SMAPS_HEADER_RE = re.compile(
    r"^([0-9a-f]+)-([0-9a-f]+) "       # start-end
    r"([rwxps-]{4}) "                  # perms
    r"([0-9a-f]+) "                    # offset
    r"([0-9a-f]+:[0-9a-f]+) "          # dev
    r"(\d+)"                           # inode
    r"(?:\s+(.*))?$"                   # pathname (optional)
)


def parse_smaps(pid: int):
    """Yield dicts of {start, end, perms, path, rss_kb, anon_kb, size_kb}
    for each VMA in /proc/<pid>/smaps."""
    path = f"/proc/{pid}/smaps"
    cur = None
    with open(path) as f:
        for line in f:
            m = SMAPS_HEADER_RE.match(line)
            if m:
                if cur is not None:
                    yield cur
                start_hex, end_hex, perms, _off, _dev, _ino, pathname = m.groups()
                cur = {
                    "start": int(start_hex, 16),
                    "end": int(end_hex, 16),
                    "perms": perms,
                    "path": (pathname or "").strip(),
                    "size_kb": 0,
                    "rss_kb": 0,
                    "anon_kb": 0,
                    "shared_kb": 0,
                    "private_kb": 0,
                }
                continue
            if cur is None:
                continue
            if ":" not in line:
                continue
            key, _, rest = line.partition(":")
            key = key.strip()
            rest = rest.strip()
            if rest.endswith(" kB"):
                try:
                    val = int(rest.split()[0])
                except ValueError:
                    continue
                if key == "Size":
                    cur["size_kb"] = val
                elif key == "Rss":
                    cur["rss_kb"] = val
                elif key == "Anonymous":
                    cur["anon_kb"] = val
                elif key == "Shared_Clean" or key == "Shared_Dirty":
                    cur["shared_kb"] += val
                elif key == "Private_Clean" or key == "Private_Dirty":
                    cur["private_kb"] += val
        if cur is not None:
            yield cur


NUMA_MAPS_RE = re.compile(r"^([0-9a-f]+) (\S+)\s+(.*)$")


def parse_numa_maps(pid: int):
    """Yield (start_addr, policy, kv_dict) per VMA from /proc/<pid>/numa_maps.
    kv_dict contains things like {'N0': pages, 'N1': pages, 'anon': pages,
    'file': filename, 'mapped': pages, 'kernelpagesize_kB': 4}."""
    path = f"/proc/{pid}/numa_maps"
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            m = NUMA_MAPS_RE.match(line)
            if not m:
                continue
            start_hex, policy, rest = m.groups()
            kv = {}
            for tok in rest.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    # numa_maps node tokens look like N0=12345
                    try:
                        kv[k] = int(v)
                    except ValueError:
                        kv[k] = v
                else:
                    kv[tok] = True
            yield int(start_hex, 16), policy, kv


# ---------- categorization ----------

SPECIAL_PATHS = {"[heap]", "[stack]", "[vdso]", "[vvar]", "[vsyscall]", "[uprobes]"}


def categorize(vma_path: str, model_path_abs: str, bin_path_abs: str) -> str:
    p = vma_path
    if not p:
        return "anon"
    if p.startswith("[anon:") or p.startswith("[stack:"):
        # named anon mapping (e.g. some allocators set names); still anon
        return "anon"
    if p in SPECIAL_PATHS:
        return "special"
    # absolute file path
    try:
        # if path is the model file
        if os.path.realpath(p) == os.path.realpath(model_path_abs):
            return "model_mmap"
    except OSError:
        pass
    # binary itself or any .so / .so.* — code or shared libs
    if p == bin_path_abs:
        return "code_libs"
    if "/lib" in p or p.endswith(".so") or ".so." in p:
        return "code_libs"
    # other file-backed (configs, locale, fonts, anything else)
    if p.startswith("/"):
        return "code_libs"
    return "anon"


# ---------- runner ----------

def wait_for_rss_stable(pid: int, settle_polls: int, poll_interval: float,
                       min_rss_mib: int, timeout_s: float) -> int:
    """Poll VmRSS until it doesn't grow for `settle_polls` consecutive polls
    AND exceeds min_rss_mib. Returns final VmRSS in kB. Returns -1 if the
    process dies or we time out."""
    deadline = time.monotonic() + timeout_s
    last = -1
    same_count = 0
    while time.monotonic() < deadline:
        rss_kb = read_vm_rss_kb(pid)
        if rss_kb < 0:
            return -1
        if rss_kb == last:
            same_count += 1
        else:
            same_count = 0
        if same_count >= settle_polls and rss_kb >= min_rss_mib * 1024:
            return rss_kb
        last = rss_kb
        time.sleep(poll_interval)
    return -1


def run_one(bin_path: str, model_path: str, no_mmap: bool, bench_args: str,
            settle_polls: int, poll_interval: float, min_rss_mib: int,
            timeout_s: float, verbose: bool) -> dict:
    bin_abs = str(Path(bin_path).resolve())
    model_abs = str(Path(model_path).resolve())

    args = [bin_abs, "-m", model_abs]
    if no_mmap:
        args.append("--no-mmap")
    if bench_args:
        args.extend(shlex.split(bench_args))

    label = "no-mmap" if no_mmap else "mmap"
    print(f"\n=== run [{label}] ===", flush=True)
    print(f"$ {' '.join(shlex.quote(a) for a in args)}", flush=True)

    # Quiet the child unless verbose
    stdout = None if verbose else subprocess.DEVNULL
    stderr = None if verbose else subprocess.DEVNULL

    proc = subprocess.Popen(args, stdout=stdout, stderr=stderr,
                            start_new_session=True)
    pid = proc.pid
    print(f"pid={pid}, waiting for RSS to stabilize "
          f"(>= {min_rss_mib} MiB, {settle_polls} polls × {poll_interval}s, "
          f"timeout {timeout_s}s)...",
          flush=True)

    try:
        rss_kb = wait_for_rss_stable(pid, settle_polls, poll_interval,
                                     min_rss_mib, timeout_s)
        if rss_kb < 0:
            print(f"  ERROR: process exited or timed out before reaching "
                  f"min RSS of {min_rss_mib} MiB.", flush=True)
            return {"label": label, "ok": False}

        print(f"  RSS stable at {rss_kb/1024:.1f} MiB; freezing process "
              f"with SIGSTOP for snapshot.", flush=True)
        os.kill(pid, signal.SIGSTOP)
        time.sleep(0.1)  # let kernel flush

        # snapshot
        smaps = list(parse_smaps(pid))
        numa = list(parse_numa_maps(pid))
        numa_by_start = {start: (policy, kv) for start, policy, kv in numa}

        # categorize and aggregate
        per_cat = defaultdict(lambda: {
            "count": 0, "rss_kb": 0, "size_kb": 0, "anon_kb": 0,
            "private_kb": 0, "shared_kb": 0,
        })
        per_cat_numa = defaultdict(lambda: defaultdict(int))  # cat -> {N0: pages, ...}
        top_anon = []
        top_model = []

        for vma in smaps:
            cat = categorize(vma["path"], model_abs, bin_abs)
            entry = per_cat[cat]
            entry["count"] += 1
            entry["rss_kb"] += vma["rss_kb"]
            entry["size_kb"] += vma["size_kb"]
            entry["anon_kb"] += vma["anon_kb"]
            entry["private_kb"] += vma["private_kb"]
            entry["shared_kb"] += vma["shared_kb"]

            # numa breakdown
            np = numa_by_start.get(vma["start"])
            if np is not None:
                _policy, kv = np
                for k, v in kv.items():
                    if k.startswith("N") and isinstance(v, int):
                        per_cat_numa[cat][k] += v

            # collect top contributors for the interesting categories
            if cat == "anon" and vma["rss_kb"] > 0:
                top_anon.append(vma)
            elif cat == "model_mmap" and vma["rss_kb"] > 0:
                top_model.append(vma)

        top_anon.sort(key=lambda v: v["rss_kb"], reverse=True)
        top_model.sort(key=lambda v: v["rss_kb"], reverse=True)

        # let it go
        os.kill(pid, signal.SIGCONT)
        time.sleep(0.05)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        # report
        total_rss_kb = sum(e["rss_kb"] for e in per_cat.values())
        print()
        print(f"--- breakdown [{label}] (total RSS {total_rss_kb/1024:.1f} MiB) ---")
        print(f"{'category':<14} {'#VMA':>6} {'RSS MiB':>12} {'%RSS':>7} "
              f"{'anon MiB':>10} {'priv MiB':>10}")
        for cat in ("model_mmap", "anon", "code_libs", "special"):
            e = per_cat.get(cat)
            if not e:
                continue
            pct = (e["rss_kb"] / total_rss_kb * 100) if total_rss_kb else 0
            print(f"{cat:<14} {e['count']:>6} "
                  f"{e['rss_kb']/1024:>12.1f} {pct:>6.1f}% "
                  f"{e['anon_kb']/1024:>10.1f} {e['private_kb']/1024:>10.1f}")

        if per_cat_numa:
            print()
            print(f"--- NUMA placement [{label}] (pages, page_size={PAGE_SIZE}) ---")
            all_nodes = sorted({k for cat in per_cat_numa.values() for k in cat})
            hdr = f"{'category':<14} " + " ".join(f"{n:>12}" for n in all_nodes)
            print(hdr)
            for cat in ("model_mmap", "anon", "code_libs", "special"):
                if cat not in per_cat_numa:
                    continue
                row = f"{cat:<14} " + " ".join(
                    f"{per_cat_numa[cat].get(n, 0):>12}" for n in all_nodes
                )
                print(row)
        else:
            print(f"\n(no numa_maps available — single-node system or "
                  f"insufficient perms)")

        # top contributors
        if top_model:
            print(f"\n--- top model_mmap VMAs [{label}] ---")
            for v in top_model[:5]:
                print(f"  {v['rss_kb']/1024:>10.1f} MiB  "
                      f"{v['perms']}  {v['path']}")
        if top_anon:
            print(f"\n--- top anon VMAs [{label}] ---")
            for v in top_anon[:8]:
                addr = f"0x{v['start']:x}-0x{v['end']:x}"
                size_mb = (v['end'] - v['start']) / MiB
                print(f"  rss {v['rss_kb']/1024:>10.1f} MiB / "
                      f"vsize {size_mb:>10.1f} MiB  {v['perms']}  {addr}")

        return {
            "label": label,
            "ok": True,
            "total_rss_mib": total_rss_kb / 1024,
            "per_cat": {cat: dict(e) for cat, e in per_cat.items()},
            "per_cat_numa": {cat: dict(d) for cat, d in per_cat_numa.items()},
        }
    except Exception as e:
        print(f"  ERROR: {e!r}", flush=True)
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise


# ---------- main ----------

def main():
    p = argparse.ArgumentParser(
        description="Measure mmap'd model vs anonymous buffer memory in a "
                    "running llama.cpp binary."
    )
    p.add_argument("--bin", required=True,
                   help="Path to llama-bench / llama-cli (the binary that "
                        "loads the model).")
    p.add_argument("--model", required=True,
                   help="Path to .gguf model file.")
    p.add_argument("--no-mmap", action="store_true",
                   help="Pass --no-mmap to the binary. Without this flag, "
                        "default mmap behavior is used.")
    p.add_argument("--both", action="store_true",
                   help="Run twice (mmap + no-mmap) and print both reports. "
                        "Overrides --no-mmap.")
    p.add_argument("--bench-args", default="-p 32 -n 8 -r 1",
                   help="Extra args appended to the binary. Default is a tiny "
                        "single-rep bench so the process stays alive long "
                        "enough to snapshot.")
    p.add_argument("--settle-polls", type=int, default=3,
                   help="How many consecutive identical RSS reads count as "
                        "stable.")
    p.add_argument("--poll-interval", type=float, default=0.5,
                   help="Seconds between RSS polls.")
    p.add_argument("--min-rss-mib", type=int, default=200,
                   help="Minimum RSS (MiB) before we accept 'loaded' — "
                        "filters out the brief pre-load window. Set lower "
                        "for small models.")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="Overall wait timeout in seconds.")
    p.add_argument("--verbose-child", action="store_true",
                   help="Let the binary's stdout/stderr through. Default "
                        "is suppressed.")
    args = p.parse_args()

    if not Path(args.bin).is_file():
        print(f"binary not found: {args.bin}", file=sys.stderr)
        sys.exit(2)
    if not Path(args.model).is_file():
        print(f"model not found: {args.model}", file=sys.stderr)
        sys.exit(2)

    runs = []
    if args.both:
        modes = [False, True]
    else:
        modes = [args.no_mmap]

    for no_mmap in modes:
        r = run_one(
            bin_path=args.bin,
            model_path=args.model,
            no_mmap=no_mmap,
            bench_args=args.bench_args,
            settle_polls=args.settle_polls,
            poll_interval=args.poll_interval,
            min_rss_mib=args.min_rss_mib,
            timeout_s=args.timeout,
            verbose=args.verbose_child,
        )
        runs.append(r)

    if len(runs) == 2 and all(r.get("ok") for r in runs):
        print("\n=== summary: mmap vs no-mmap ===")
        mmap_run = next(r for r in runs if r["label"] == "mmap")
        nommap_run = next(r for r in runs if r["label"] == "no-mmap")
        print(f"  mmap total RSS:    {mmap_run['total_rss_mib']:>10.1f} MiB")
        print(f"  no-mmap total RSS: {nommap_run['total_rss_mib']:>10.1f} MiB")
        for cat in ("model_mmap", "anon"):
            m = mmap_run["per_cat"].get(cat, {}).get("rss_kb", 0) / 1024
            n = nommap_run["per_cat"].get(cat, {}).get("rss_kb", 0) / 1024
            print(f"  {cat:<12} mmap={m:>10.1f} MiB  no-mmap={n:>10.1f} MiB  "
                  f"delta={n-m:+.1f} MiB")
        print()
        print("interpretation:")
        m_mmap = mmap_run["per_cat"].get("model_mmap", {}).get("rss_kb", 0)/1024
        m_anon = mmap_run["per_cat"].get("anon", {}).get("rss_kb", 0)/1024
        if m_mmap > 4 * m_anon:
            print(f"  default mmap: {m_mmap:.0f} MiB in model_mmap vs "
                  f"{m_anon:.0f} MiB in anon. A buffer-allocator mbind() "
                  f"would pin only {m_anon/(m_mmap+m_anon)*100:.1f}% of "
                  f"resident model memory. Either test with --no-mmap or "
                  f"add an mbind pass over the mmap'd region.")
        elif m_anon > m_mmap:
            print(f"  default mmap: anon ({m_anon:.0f} MiB) exceeds "
                  f"model_mmap ({m_mmap:.0f} MiB). Buffer-allocator mbind() "
                  f"is the right place to start.")
        else:
            print(f"  default mmap: roughly balanced "
                  f"(model_mmap={m_mmap:.0f} MiB, anon={m_anon:.0f} MiB). "
                  f"Buffer-allocator mbind() captures the activation/KV "
                  f"share; mmap region needs its own pass for weights.")


if __name__ == "__main__":
    main()
