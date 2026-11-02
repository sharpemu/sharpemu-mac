#!/usr/bin/env python3
# Copyright (C) 2026 SharpEmu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
"""Classifies every `pause` and locked read-modify-write in a built guest the way SharpEmu's spin-nap patcher does
(GuestRedZonePatcher.SpinLoops.cs: ClosesShortBackwardLoop / InShortLockedLoop), from `objdump -d`, so a guest's
loop shapes can be checked without running it. The totals line up with the loader's
"spin-nap: pause_sites=.. non_loop=.. locked_sites=.. locked_non_loop=.." log line (the loader additionally drops
loop sites it cannot relocate, which this script does not model).

Usage: check-shapes.py guest.bin [--expect-jobpool]
  --expect-jobpool: exit 1 unless scan_slots has exactly one locked cmpxchg that is an eligible locked loop (<= 256 B,
  no call/ret/indirect jmp) and wait_for_jobs' pause is rejected because a call sits in its window.
"""
import re
import subprocess
import sys

MAX_SPIN = 96
MAX_LOCKED = 256


def disasm(path):
    out = subprocess.check_output(["objdump", "-d", "-M", "intel", "--no-show-raw-insn", path], text=True)
    funcs, cur = {}, None
    for line in out.splitlines():
        m = re.match(r"^([0-9a-f]+) <(.+)>:$", line)
        if m:
            cur = m.group(2)
            funcs[cur] = []
            continue
        m = re.match(r"^\s+([0-9a-f]+):\t(\S.*)$", line)
        if m and cur:
            text = m.group(2).strip()
            funcs[cur].append([int(m.group(1), 16), text])
    # Fill in NextIP (next instruction address; the last one gets +1, it never matters).
    res = {}
    for name, ins in funcs.items():
        rows = []
        for i, (addr, text) in enumerate(ins):
            nxt = ins[i + 1][0] if i + 1 < len(ins) else addr + 1
            mn = text.split()[0]
            if mn in ("lock", "rep", "repz", "repnz", "bnd", "notrack") and len(text.split()) > 1:
                mn = text.split()[0] + " " + text.split()[1]
            rows.append(dict(ip=addr, nxt=nxt, text=text, mn=mn))
        res[name] = rows
    return res


def target(row):
    m = re.match(r"^(?:j\w+|loop\w*)\s+([0-9a-f]+)\b", row["text"])
    return int(m.group(1), 16) if m else None


def is_call(r): return r["mn"] == "call"
def is_ret(r): return r["mn"] in ("ret", "retq")
def is_indirect_jmp(r): return r["mn"] == "jmp" and target(r) is None
def is_breaker(r):
    return is_call(r) or is_ret(r) or is_indirect_jmp(r) or r["mn"] in ("syscall", "sysenter", "int", "int3", "ud2", "hlt")
def is_direct_branch(r): return target(r) is not None


def has_mem(r): return "[" in r["text"]


def is_locked_rmw(r):
    if not has_mem(r):
        return False
    return r["text"].startswith("lock ") or r["mn"] == "xchg"


def closes_short_backward_loop(rows, idx, start):
    pause_addr = rows[idx]["ip"]
    pause_end = pause_addr + 2
    for r in rows[idx + 1:]:
        if r["nxt"] - pause_end > MAX_SPIN:
            return False
        t = target(r)
        if t is not None and t <= pause_addr and t >= start:
            return True
        if is_call(r) or is_ret(r) or is_indirect_jmp(r):
            return False
    return False


def in_short_locked_loop(rows, idx, start):
    site = rows[idx]
    for j in range(idx, len(rows)):
        r = rows[j]
        if r["nxt"] - site["ip"] > MAX_LOCKED:
            return False
        if j != idx and is_breaker(r):
            return False
        t = target(r)
        if t is not None and t <= site["ip"] and t >= start and r["nxt"] - t <= MAX_LOCKED:
            # PrecedingBodyIsClean: walk back from the site to the branch target without a breaker.
            for k in range(idx, -1, -1):
                q = rows[k]
                if q["ip"] < t:
                    break
                if k != idx and is_breaker(q):
                    break
                if q["ip"] == t:
                    return True
    return False


def main():
    path = sys.argv[1]
    funcs = disasm(path)
    pause_total = non_loop = locked_total = locked_non_loop = 0
    report = {}
    for name, rows in funcs.items():
        if not rows:
            continue
        start = rows[0]["ip"]
        for i, r in enumerate(rows):
            if r["mn"] == "pause":
                pause_total += 1
                ok = closes_short_backward_loop(rows, i, start)
                non_loop += 0 if ok else 1
                report.setdefault(name, []).append((r["ip"], "pause", "loop" if ok else "non_loop"))
            elif is_locked_rmw(r):
                locked_total += 1
                ok = in_short_locked_loop(rows, i, start)
                locked_non_loop += 0 if ok else 1
                report.setdefault(name, []).append((r["ip"], r["text"].split("\t")[0][:40], "locked_loop" if ok else "locked_non_loop"))
    for name, items in report.items():
        for ip, what, verdict in items:
            if what == "pause" or verdict == "locked_loop" or "--all" in sys.argv:
                print(f"{name:20s} {ip:#8x}  {what:28s} {verdict}")
    print(f"pause_sites={pause_total} non_loop={non_loop} locked_sites_all={locked_total} "
          f"locked_loop_sites={locked_total - locked_non_loop} locked_non_loop={locked_non_loop}")
    if "--expect-jobpool" in sys.argv:
        ok = True
        w = [x for x in report.get("wait_for_jobs", []) if x[1] == "pause"]
        if len(w) != 1 or w[0][2] != "non_loop":
            print("FAIL: wait_for_jobs pause must be non_loop (a call inside the window)")
            ok = False
        s = [x for x in report.get("scan_slots", []) if x[2] == "locked_loop"]
        if len(s) != 1 or "cmpxchg" not in s[0][1]:
            print("FAIL: scan_slots must have exactly one eligible locked loop, the lock cmpxchg")
            ok = False
        rows = funcs["scan_slots"]
        site = [i for i, r in enumerate(rows) if r["text"].startswith("lock cmpxchg")][0]
        t = [target(r) for r in rows[site:] if is_direct_branch(r) and target(r) <= rows[site]["ip"]][0]
        end = [r for r in rows[site:] if target(r) == t][0]["nxt"]
        print(f"scan_slots: lock cmpxchg at {rows[site]['ip']:#x}, backward jne to {t:#x}, loop body {end - t} bytes")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
