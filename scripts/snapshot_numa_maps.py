#!/usr/bin/env python3
"""
snapshot_numa_maps.py — per-VMA NUMA placement from /proc/<pid>/numa_maps.

Richer than smaps-based mem_breakdown.py because numa_maps tells us:
  - N<n>=<pages>  : pages residing on node n
  - anon=         : anonymous pages
  - file=<path>   : file backing
  - mapped=       : total mapped pages
  - migrated=     : pages migrated by kernel auto-balance
  - huge=         : THP backing pages
  - dirty=, swapcache=, active=, writeback=, mapmax=, kernelpagesize_kB=

Use during Stage 0: one snapshot per variant after warmup. Use along with
snapshot_thread_domains.py — those give "which thread on which socket?",
this gives "which MEMORY is on which socket?".

Categorizes VMAs into:
  - model_mmap    : the GGUF file
  - anon          : heap, ggml allocations, KV cache, scratch
  - code_libs     : binary + shared libs
  - special       : [heap], [stack], [vdso], [vvar], etc.

Output: JSON to stdout. Stdlib only.

Usage:
    snapshot_numa_maps.py --pid <pid> --model /path/to/model.gguf
    snapshot_numa_maps.py --pid <pid>   # category will mark gguf-by-extension
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


HEADER_RE = re.compile(r"^([0-9a-f]+)\s+(\S+)\s+(.*)$")
SPECIAL = {"[heap]", "[stack]", "[vdso]", "[vvar]", "[vsyscall]", "[uprobes]"}


def categorize(path, model_real):
    if not path:
        return "anon"
    if path in SPECIAL:
        return "special"
    if path.startswith("[anon:") or path.startswith("[stack:"):
        return "anon"
    if model_real and os.path.realpath(path) == model_real:
        return "model_mmap"
    if path.endswith(".gguf"):
        return "model_mmap"
    if "/lib" in path or path.endswith(".so") or ".so." in path:
        return "code_libs"
    if path.startswith("/"):
        return "code_libs"
    return "anon"


def parse_numa_maps(pid):
    path = Path(f"/proc/{pid}/numa_maps")
    vmas = []
    try:
        text = path.read_text()
    except OSError as e:
        raise SystemExit(f"cannot read {path}: {e}")

    for line in text.splitlines():
        m = HEADER_RE.match(line)
        if not m:
            continue
        addr_hex, policy, rest = m.groups()
        record = {
            "address": f"0x{addr_hex}",
            "policy": policy,
            "per_node": {},   # {node_id: pages}
        }
        path_part = None
        for tok in rest.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                if k.startswith("N"):
                    try:
                        record["per_node"][int(k[1:])] = int(v)
                    except ValueError:
                        pass
                elif k == "file":
                    path_part = v
                    record["file"] = v
                else:
                    try:
                        record[k] = int(v)
                    except ValueError:
                        record[k] = v
            else:
                # tokens with no '=' are flags/names ([heap], [stack], etc.)
                # The last bare token tends to be the pathname or special tag.
                path_part = tok
                if tok in SPECIAL:
                    record["file"] = tok
        record["_path"] = path_part or ""
        vmas.append(record)
    return vmas


def page_size():
    return os.sysconf("SC_PAGESIZE")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--model", default=None,
                   help="absolute path to the GGUF; helps categorize "
                        "model_mmap precisely")
    p.add_argument("--out", default="-")
    p.add_argument("--top-n", type=int, default=10,
                   help="how many top VMAs per category to include in detail")
    args = p.parse_args()

    model_real = os.path.realpath(args.model) if args.model else None

    vmas = parse_numa_maps(args.pid)
    ps = page_size()

    # Categorize and aggregate
    by_cat = {}
    for v in vmas:
        cat = categorize(v["_path"], model_real)
        v["category"] = cat
        v.pop("_path", None)
        b = by_cat.setdefault(cat, {
            "n_vmas": 0,
            "pages": 0,
            "per_node_pages": {},
            "migrated_pages": 0,
            "huge_pages": 0,
            "dirty_pages": 0,
            "vmas": [],
        })
        b["n_vmas"] += 1
        page_total = sum(v["per_node"].values())
        b["pages"] += page_total
        for nid, pg in v["per_node"].items():
            b["per_node_pages"][nid] = b["per_node_pages"].get(nid, 0) + pg
        for k in ("migrated", "huge", "dirty"):
            if k in v and isinstance(v[k], int):
                b[f"{k}_pages"] += v[k]
        b["vmas"].append(v)

    # Build summary (top-N largest VMAs per category for inspection)
    summary = {}
    for cat, b in by_cat.items():
        b["vmas"].sort(key=lambda x: -sum(x["per_node"].values()))
        top = b["vmas"][:args.top_n]
        summary[cat] = {
            "n_vmas": b["n_vmas"],
            "pages": b["pages"],
            "bytes": b["pages"] * ps,
            "per_node_pages": {str(k): v for k, v in b["per_node_pages"].items()},
            "per_node_bytes": {str(k): v * ps for k, v in b["per_node_pages"].items()},
            "migrated_pages": b["migrated_pages"],
            "huge_pages": b["huge_pages"],
            "dirty_pages": b["dirty_pages"],
            "top_vmas": top,
        }

    # Overall totals
    grand_pages = sum(s["pages"] for s in summary.values())
    grand_per_node = {}
    for s in summary.values():
        for k, v in s["per_node_pages"].items():
            grand_per_node[k] = grand_per_node.get(k, 0) + v

    out_doc = {
        "schema_version": "1",
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": args.pid,
        "model_path_resolved": model_real,
        "page_size_bytes": ps,
        "totals": {
            "pages": grand_pages,
            "bytes": grand_pages * ps,
            "per_node_pages": grand_per_node,
            "per_node_bytes": {k: v * ps for k, v in grand_per_node.items()},
        },
        "by_category": summary,
    }

    text = json.dumps(out_doc, indent=2)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
