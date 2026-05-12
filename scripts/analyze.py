#!/usr/bin/env python3
"""
analyze.py — Parse Stage 0 baseline output and emit a percentile comparison.

Reads results/baseline-<ts>/<variant>/bench.jsonl produced by bench.sh.
Each JSONL line is one llama-bench test run (pp512 or tg128), with
`samples_ns` = per-iteration wall-time nanoseconds.

Output:
  - One row per (variant, test) with p50/p95/p99/p99.9, mean, stddev, and
    samples count (in ms for readability).
  - A "delta vs baseline" column (negative = faster).
  - Optionally extracts a few perf-stat lines next to each variant.

Usage:
    analyze.py <results_dir>
    analyze.py --raw <results_dir>   # also print raw stats
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path


def percentile(sorted_vals, q):
    """Linear-interpolation percentile. q in [0, 100]."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (q / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def stats_from(samples_ns):
    """Return dict of stats in milliseconds."""
    if not samples_ns:
        return None
    ms = sorted(s / 1e6 for s in samples_ns)
    n = len(ms)
    mean = sum(ms) / n
    var = sum((x - mean) ** 2 for x in ms) / max(n - 1, 1)
    return {
        "n": n,
        "mean_ms": mean,
        "stddev_ms": math.sqrt(var),
        "min_ms": ms[0],
        "p50_ms": percentile(ms, 50),
        "p95_ms": percentile(ms, 95),
        "p99_ms": percentile(ms, 99),
        "p999_ms": percentile(ms, 99.9),
        "max_ms": ms[-1],
    }


def load_variant(variant_dir: Path):
    """Return {test_name: stats_dict} for one variant directory."""
    out = {}
    bench_file = variant_dir / "bench.jsonl"
    if not bench_file.is_file() or bench_file.stat().st_size == 0:
        return None
    with bench_file.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            # llama-bench's jsonl output can be wrapped in [ ... ] for one
            # of its formats; strip a trailing comma if present and the
            # outer brackets if a whole-file JSON array.
            line = line.rstrip(",")
            if line in ("[", "]"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARN: {bench_file}:{line_no} not valid JSON: {e}",
                      file=sys.stderr)
                continue
            test = row.get("test") or row.get("name") or "unknown"
            samples = row.get("samples_ns") or []
            if not samples:
                # Older builds may emit samples_ts (tokens/sec) only; skip
                continue
            stats = stats_from(samples)
            if stats is None:
                continue
            # Combine across multiple rows for the same test (e.g. multiple repsplits)
            if test in out:
                # naive merge: pool samples
                pooled = sorted(out[test]["_samples"] + samples)
                merged = stats_from(pooled)
                merged["_samples"] = pooled
                out[test] = merged
            else:
                stats["_samples"] = samples
                out[test] = stats
    # strip internal _samples before returning
    for s in out.values():
        s.pop("_samples", None)
    return out


PERF_LINE_RE = re.compile(
    r"^\s*([\d,\.]+)\s+([\w\-]+(?:[:-][\w-]+)*)\s*(?:#.*)?$"
)


def load_perf(variant_dir: Path):
    """Return {event_name: count_str} from perf.log; best-effort."""
    f = variant_dir / "perf.log"
    if not f.is_file():
        return {}
    out = {}
    for line in f.read_text(errors="replace").splitlines():
        m = PERF_LINE_RE.match(line)
        if not m:
            continue
        count, event = m.groups()
        # Skip lines that are clearly headers/derived (event names with units)
        if "/" in event or event in ("seconds", "time"):
            continue
        out[event] = count.replace(",", "")
    return out


def fmt(v, w=8, dp=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return f"{'-':>{w}}"
    if isinstance(v, (int, float)):
        return f"{v:>{w}.{dp}f}"
    return f"{v:>{w}}"


def main():
    p = argparse.ArgumentParser(description="Analyze Stage 0 baseline results.")
    p.add_argument("results_dir", help="Path to results/baseline-<timestamp>/")
    p.add_argument("--raw", action="store_true",
                   help="Also dump raw per-variant stats.")
    args = p.parse_args()

    rd = Path(args.results_dir)
    if not rd.is_dir():
        print(f"not a directory: {rd}", file=sys.stderr); sys.exit(2)

    env_file = rd / "env.txt"
    if env_file.is_file():
        print("--- environment ---")
        for line in env_file.read_text().splitlines()[:10]:
            print(f"  {line}")
        print()

    # discover variants
    variants = sorted(d for d in rd.iterdir() if d.is_dir())
    if not variants:
        print("no variant subdirs in results dir", file=sys.stderr); sys.exit(2)

    # load
    per_variant = {}
    per_variant_perf = {}
    for v in variants:
        s = load_variant(v)
        if s is None:
            print(f"  WARN: {v.name} has no bench.jsonl yet", file=sys.stderr)
            continue
        per_variant[v.name] = s
        per_variant_perf[v.name] = load_perf(v)

    if not per_variant:
        print("no parseable bench.jsonl found", file=sys.stderr); sys.exit(2)

    # union of test names
    all_tests = sorted({t for s in per_variant.values() for t in s})

    # use the first variant in 'baseline'-ish order as the reference
    BASELINE_PREF = ["baseline", "distribute", "isolate", "interleave", "bind0"]
    ref = None
    for name in BASELINE_PREF:
        if name in per_variant:
            ref = name; break
    if ref is None:
        ref = next(iter(per_variant))

    # Print per-test comparison
    for test in all_tests:
        print(f"=== {test}  (reference: {ref}) ===")
        hdr = (f"{'variant':<14} {'n':>6} {'mean':>9} {'p50':>9} "
               f"{'p95':>9} {'p99':>9} {'p99.9':>9} {'max':>9}  "
               f"{'Δp99 ms':>9}  {'Δp99 %':>8}")
        print(hdr)
        print("-" * len(hdr))
        ref_p99 = per_variant.get(ref, {}).get(test, {}).get("p99_ms")
        for vname in [ref] + [v for v in per_variant if v != ref]:
            stats = per_variant.get(vname, {}).get(test)
            if stats is None:
                print(f"{vname:<14}  (no data)")
                continue
            dp99_ms = ""
            dp99_pct = ""
            if ref_p99 is not None and stats["p99_ms"] is not None:
                d = stats["p99_ms"] - ref_p99
                dp99_ms = f"{d:+9.2f}"
                if ref_p99 > 0:
                    dp99_pct = f"{(d / ref_p99) * 100:+8.1f}"
            else:
                dp99_ms = "   -"
                dp99_pct = "   -"
            print(
                f"{vname:<14} "
                f"{stats['n']:>6d} "
                f"{stats['mean_ms']:>9.2f} "
                f"{stats['p50_ms']:>9.2f} "
                f"{stats['p95_ms']:>9.2f} "
                f"{stats['p99_ms']:>9.2f} "
                f"{stats['p999_ms']:>9.2f} "
                f"{stats['max_ms']:>9.2f}  "
                f"{dp99_ms}  {dp99_pct}"
            )
        print()

    # Perf-stat highlights
    interesting = [
        "node-load-misses", "node-loads", "cpu-migrations",
        "context-switches", "dTLB-load-misses",
        "LLC-load-misses", "instructions", "cycles",
    ]
    have_perf = any(per_variant_perf.values())
    if have_perf:
        print("=== perf stat highlights ===")
        hdr = f"{'variant':<14} " + " ".join(f"{e[:18]:>20}" for e in interesting)
        print(hdr)
        print("-" * len(hdr))
        for vname in per_variant:
            perf = per_variant_perf.get(vname, {})
            row = f"{vname:<14} "
            for e in interesting:
                val = perf.get(e, "-")
                # Compact: format big numbers with k/M/G suffix
                if val != "-":
                    try:
                        n = int(val)
                        for unit, div in [("G", 1e9), ("M", 1e6), ("k", 1e3)]:
                            if n >= div:
                                val = f"{n/div:.2f}{unit}"
                                break
                        else:
                            val = str(n)
                    except ValueError:
                        pass
                row += f"{val:>20} "
            print(row)
        print()
    else:
        print("(no perf.log found in any variant)")

    if args.raw:
        print("=== raw stats (JSON) ===")
        print(json.dumps(per_variant, indent=2, default=str))


if __name__ == "__main__":
    main()
