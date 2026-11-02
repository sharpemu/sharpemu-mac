#!/usr/bin/env python3
# Copyright (C) 2026 SharpEmu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
"""One markdown table row per run directory: cores busy, top 3 threads, context switches / futex / sched_yield per
second and the guest's key counters.   Usage: summarize.py RUN_DIR [RUN_DIR ...]"""
import json
import os
import sys


def g(m, key, scale=1.0, fmt="{:.1f}"):
    v = m.get("guest_flat", {}).get(key)
    return "-" if v is None else fmt.format(v / scale)


def hle_top(m, n=4):
    per = (m.get("hle") or {}).get("per_nid_per_s") or {}
    if not per:
        return ""
    top = ", ".join(f"{k.split(':')[-1]}={v:,.0f}/s" for k, v in list(per.items())[:n])
    gc = m.get("gc") or {}
    return f"HLE: {top}; GC gen0/1/2={gc.get('gc0')}/{gc.get('gc1')}/{gc.get('gc2')} pause={gc.get('pause_ms')} ms alloc={gc.get('alloc_mb')} MB"


def key_counters(m):
    text = key_counters_guest(m)
    extra = hle_top(m)
    return f"{text} | {extra}" if extra else text


def key_counters_guest(m):
    f = m.get("guest_flat", {})
    name = m["guest"]
    ms = f.get("jobpool.total.measured_us") or f.get("scream.total.measured_us") or 0
    if name.startswith("jobpool"):
        s = ms / 1e6 or 1
        return (f"jobs/s={f.get('jobpool.total.jobs', 0) / s:.0f} polls/s={f.get('jobpool.total.polls', 0) / s:,.0f} "
                f"pause/s={f.get('jobpool.total.pauses', 0) / s:,.0f} pred/s={f.get('jobpool.total.pred_calls', 0) / s:,.0f} "
                f"lat p50/p95={g(m, 'jobpool.lat.lat_p50_ns', 1e6)}/{g(m, 'jobpool.lat.lat_p95_ns', 1e6)} ms "
                f"frame p50={g(m, 'jobpool.frame.frame_p50_ns', 1e6)} ms parked_lat p50/p95={g(m, 'jobpool.parked_lat.parked_lat_p50_ns', 1e3, '{:.0f}')}/{g(m, 'jobpool.parked_lat.parked_lat_p95_ns', 1e3, '{:.0f}')} us "
                f"cpu/job={m.get('cpu_us_per_job', '-')} emu cpu/job={m.get('emu_cpu_us_per_job', '-')} us")
    if name.startswith("scream"):
        return (f"gtid/s={g(m, 'scream.total.gtid_per_s', 1, '{:,.0f}')} mutex ops/s={g(m, 'scream.total.mutex_ops_per_s', 1, '{:,.0f}')} "
                f"gtid p50={g(m, 'scream.gtid.gtid_p50_ns', 1, '{:.0f}')} ns lock p50/p99={g(m, 'scream.lock.lock_p50_ns', 1, '{:.0f}')}/"
                f"{g(m, 'scream.lock.lock_p99_ns', 1000, '{:.0f}')}k ns")
    if name.startswith("wakeping"):
        parts = []
        for ph in ("evf", "cond"):
            n = f.get(f"wakeping.phase.{ph}.round_trips")
            us = f.get(f"wakeping.phase.{ph}.phase_us")
            if n:
                parts.append(f"{ph}: {n / (us / 1e6):,.0f} rt/s p50/p95={f.get(f'wakeping.phase.{ph}.rtt_p50_ns', 0) / 1000:.0f}/"
                             f"{f.get(f'wakeping.phase.{ph}.rtt_p95_ns', 0) / 1000:.0f} us")
        return "; ".join(parts)
    if name.startswith("sleeplat"):
        parts = []
        for ph in ("usleep1_b2b", "usleep50_b2b", "usleep1000_b2b", "usleep1000_gap", "yield0_b2b"):
            if f"sleeplat.phase.{ph}.elapsed_p50_ns" in f:
                parts.append(f"{ph} p50/p95={f[f'sleeplat.phase.{ph}.elapsed_p50_ns'] / 1000:.0f}/{f[f'sleeplat.phase.{ph}.elapsed_p95_ns'] / 1000:.0f} us")
        return "; ".join(parts)
    if name.startswith("baseline"):
        return f"usleep calls={f.get('baseline.total.usleep_calls')}"
    return ""


def main():
    print("| run | cores busy | emu cpu s (guest busy s) | other cores | top 3 threads (CPU s) | ctx/s | futex/s | yield/s | nanosleep/s | key guest counters |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in sys.argv[1:]:
        p = os.path.join(d, "metrics.json")
        if not os.path.exists(p):
            print(f"| {d} | (no metrics.json) |")
            continue
        m = json.load(open(p))
        top = ", ".join(f"{t['name']}={t['cpu_s']:.2f}" for t in m.get("threads", [])[:3])
        label = f"{m['guest'].replace('.bin', '')} {m['label']}"
        ok = "" if m.get("result_ok") and m.get("valid", True) else " **INVALID**"
        row = [label + ok, m.get("cores_busy", "-"), f"{m.get('emu_cpu_s', '-')} ({m.get('guest_busy_s', '-')})", m.get("host_other_cores", "-"), top,
               m.get("ctx_switches_per_s", "-"), m.get("futex_per_s", "-"), m.get("sched_yield_per_s", "-"),
               m.get("nanosleep_per_s", "-"), key_counters(m)]
        print("| " + " | ".join(str(x) for x in row) + " |")


if __name__ == "__main__":
    main()
