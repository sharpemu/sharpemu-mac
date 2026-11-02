#!/usr/bin/env python3
# Copyright (C) 2026 SharpEmu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
"""Median and spread per metric for two sets of runs.

Usage: compare.py A_DIR B_DIR [--all] [--grep REGEX]
Each argument is a run directory (containing metrics.json) or a directory of run directories. Prints, per metric,
the median and min..max of A and B, the median change of B relative to A, and "*" when the two ranges do not overlap
(a difference larger than the run-to-run spread). With one run per arm every spread is a point: do not read an
"*" as significance.
"""
import glob
import json
import os
import re
import statistics
import sys

HEADLINE = [
    "cores_busy", "task_clock_s", "cores_busy_proc", "cores_busy_threads", "ctx_switches_per_s", "cpu_migrations_per_s",
    "futex_per_s", "sched_yield_per_s", "nanosleep_per_s", "clock_nanosleep_per_s", "ipc", "instructions_per_s",
    "host_other_cores", "wall_s",
]


def load(d):
    paths = [os.path.join(d, "metrics.json")] if os.path.exists(os.path.join(d, "metrics.json")) else sorted(glob.glob(os.path.join(d, "*", "metrics.json")))
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def scalars(m):
    out = {k: v for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    for k, v in m.get("guest_flat", {}).items():
        out[f"guest.{k}"] = v
    # Top thread CPU, ranked: robust against thread-id churn between runs.
    for i, t in enumerate(m.get("threads", [])[:3]):
        out[f"top_thread{i + 1}_cpu_s"] = t["cpu_s"]
    gc = m.get("gc") or {}
    for k in ("gc0", "gc1", "gc2", "pause_ms", "alloc_mb"):
        if k in gc:
            out[f"gc.{k}"] = gc[k]
    for nid, v in ((m.get("hle") or {}).get("per_nid_per_s") or {}).items():
        out[f"hle.{nid}.per_s"] = v
    return out


def fmt(v):
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.3g}"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_all = "--all" in sys.argv
    pat = None
    if "--grep" in sys.argv:
        pat = re.compile(sys.argv[sys.argv.index("--grep") + 1])
        args = [a for a in args if a != sys.argv[sys.argv.index("--grep") + 1]]
    if len(args) != 2:
        print(__doc__)
        sys.exit(2)
    A, B = load(args[0]), load(args[1])
    if not A or not B:
        print("no metrics.json found under", args[0] if not A else args[1])
        sys.exit(1)
    sa, sb = [scalars(m) for m in A], [scalars(m) for m in B]
    keys = [k for k in HEADLINE if any(k in s for s in sa + sb)]
    rest = sorted({k for s in sa + sb for k in s} - set(keys))
    if pat:
        keys = [k for k in keys if pat.search(k)]
        rest = [k for k in rest if pat.search(k)]
    elif not show_all:
        rest = [k for k in rest if k.startswith("guest.") and not re.search(r"\.(cfg|calib)\.|\.w=|measured_us|_n$", k)]
    print(f"A: {args[0]}  ({len(A)} runs)\nB: {args[1]}  ({len(B)} runs)")
    print(f"{'metric':52s} {'A median [min..max]':>32s} {'B median [min..max]':>32s} {'B/A':>8s}")
    for k in keys + rest:
        va = [s[k] for s in sa if k in s]
        vb = [s[k] for s in sb if k in s]
        if not va or not vb:
            continue
        ma, mb = statistics.median(va), statistics.median(vb)
        ratio = f"{mb / ma:.2f}x" if ma else "-"
        flag = "*" if (max(va) < min(vb) or max(vb) < min(va)) else " "
        print(f"{k:52s} {fmt(ma):>10s} [{fmt(min(va)):>8s}..{fmt(max(va)):>8s}] {fmt(mb):>10s} [{fmt(min(vb)):>8s}..{fmt(max(vb)):>8s}] {ratio:>8s} {flag}")


if __name__ == "__main__":
    main()
