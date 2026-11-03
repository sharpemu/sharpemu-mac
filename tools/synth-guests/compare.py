#!/usr/bin/env python3
# Copyright (C) 2026 SharpEmu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
"""Median and spread per metric for two sets of runs.

Usage: compare.py A_DIR B_DIR [--all] [--grep REGEX] [--max-other CORES] [--keep-invalid]
Each argument is a run directory (containing metrics.json) or a directory of run directories. Prints, per metric,
the median and min..max of A and B, the median change of B relative to A, the number of runs behind each side, and "*"
when the two ranges do not overlap (a difference larger than the run-to-run spread). With one run per arm every spread
is a point: do not read an "*" as significance.

Runs are dropped before comparing (and listed) when they failed (result_ok false), are marked invalid by the harness
(perf lag >= the guest's grace period, ...), or ran while the rest of the machine used more than --max-other cores
(default 0.5; host_other_cores in metrics.json). --keep-invalid keeps everything.

Which metrics are valid A/B metrics depends on the guest (the others are context, not evidence of an emulator change):
  scream   emu_cpu_s / emu_cores / emu_cpu_us_per_guard, scream.gtid p50, scream.lock/unlock percentiles, achieved
           gtid/s and mutex ops/s, futex/ctx rates. NOT cores_busy: pacing is a busy-wait, so total CPU is ~constant.
  jobpool  emu_cpu_s / emu_cpu_us_per_job, jobs/s, jobpool.parked_lat, jobpool.lat, frame percentiles. With NAP=1 (DS default)
           cores_busy is meaningful too; with NAP=0 and 13 spinning workers it saturates at the core count.
  agcbuild emu_cpu_us_per_draw, agcbuild.import.<name> ns p50/p95/p99, calls/s; agcbuild.verify bad must stay 0.
  wakeping, sleeplat, baseline: cores_busy / task_clock_s and the guest's latency percentiles.
"""
import glob
import json
import os
import re
import statistics
import sys

HEADLINE = [
    "emu_cpu_s", "emu_cores", "emu_cpu_us_per_job", "emu_cpu_us_per_guard", "cpu_us_per_job", "cpu_us_per_guard", "guest_busy_s",
    "cores_busy", "task_clock_s", "cores_busy_proc", "cores_busy_threads", "ctx_switches_per_s", "cpu_migrations_per_s",
    "futex_per_s", "sched_yield_per_s", "nanosleep_per_s", "clock_nanosleep_per_s", "ipc", "instructions_per_s",
    "host_other_cores", "wall_s",
]


VALID_METRICS = {
    "scream": "emu_cpu_s, emu_cores, emu_cpu_us_per_guard, scream.gtid/lock/unlock percentiles, achieved rates (NOT cores_busy)",
    "jobpool": "emu_cpu_s, emu_cpu_us_per_job, jobs/s, parked_lat, lat, frame (cores_busy with NAP=1; it saturates with NAP=0)",
    "agcbuild": "emu_cpu_us_per_draw, agcbuild.import.<name> ns percentiles, calls/s, agcbuild.verify bad (NOT cores_busy)",
    "wakeping": "cores_busy, task_clock_s, rtt percentiles, round trips/s",
    "sleeplat": "cores_busy, task_clock_s, per-phase elapsed percentiles and phase CPU",
    "baseline": "cores_busy, task_clock_s",
}


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
    max_other = 0.5
    if "--max-other" in sys.argv:
        max_other = float(sys.argv[sys.argv.index("--max-other") + 1])
        args.remove(sys.argv[sys.argv.index("--max-other") + 1])
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
    keep = "--keep-invalid" in sys.argv

    def usable(m):
        if keep:
            return True
        why = []
        if not m.get("result_ok"):
            why.append("result_ok=false")
        if m.get("valid") is False:
            why.append("invalid: " + "; ".join(m.get("invalid_reasons", [])))
        if m.get("host_other_cores", 0) > max_other:
            why.append(f"host_other_cores={m['host_other_cores']} > {max_other}")
        if why:
            print(f"dropping {m.get('out_dir')}: {', '.join(why)}")
        return not why

    nA, nB = len(A), len(B)
    A, B = [m for m in A if usable(m)], [m for m in B if usable(m)]
    if not A or not B:
        print(f"no usable runs left (A {len(A)}/{nA}, B {len(B)}/{nB}); rerun or pass --keep-invalid / a higher --max-other")
        sys.exit(1)
    guest = A[0].get("guest", "").replace(".bin", "")
    if guest in VALID_METRICS:
        print(f"valid A/B metrics for {guest}: {VALID_METRICS[guest]}")
    sa, sb = [scalars(m) for m in A], [scalars(m) for m in B]
    keys = [k for k in HEADLINE if any(k in s for s in sa + sb)]
    rest = sorted({k for s in sa + sb for k in s} - set(keys))
    if pat:
        keys = [k for k in keys if pat.search(k)]
        rest = [k for k in rest if pat.search(k)]
    elif not show_all:
        rest = [k for k in rest if k.startswith("guest.") and not re.search(r"\.(cfg|calib)\.|\.w=|measured_us|_n$", k)]
    print(f"A: {args[0]}  ({len(A)}/{nA} runs used)\nB: {args[1]}  ({len(B)}/{nB} runs used)")
    print(f"{'metric':52s} {'A median [min..max]':>32s} {'B median [min..max]':>32s} {'B/A':>8s}  nA/nB")
    for k in keys + rest:
        va = [s[k] for s in sa if k in s]
        vb = [s[k] for s in sb if k in s]
        if not va or not vb:
            continue
        ma, mb = statistics.median(va), statistics.median(vb)
        ratio = f"{mb / ma:.2f}x" if ma else "-"
        flag = "*" if (max(va) < min(vb) or max(vb) < min(va)) else " "
        print(f"{k:52s} {fmt(ma):>10s} [{fmt(min(va)):>8s}..{fmt(max(va)):>8s}] {fmt(mb):>10s} [{fmt(min(vb)):>8s}..{fmt(max(vb)):>8s}] {ratio:>8s} {flag} {len(va)}/{len(vb)}")


if __name__ == "__main__":
    main()
