#!/usr/bin/env python3
# Copyright (C) 2026 SharpEmu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
"""Headless measurement harness for the synthetic guests (Linux only: /proc, perf).

Runs `SharpEmu <guest.bin>` once, watches its output for the guest's "@@ MARK start" / "@@ MARK end" lines
and measures host CPU for exactly that window:
  * perf stat (attached with --control, enabled at the start marker, disabled at the end marker):
    task-clock, context-switches, cpu-migrations, instructions, cycles and the futex / sched_yield / nanosleep /
    clock_nanosleep syscall tracepoints;
  * /proc/<pid>/task/*/{comm,schedstat,status} snapshots at both markers: per-thread CPU seconds and
    context-switch deltas;
  * "@@ MARK phase_start/phase_end" lines (sleeplat, wakeping) get their own whole-process CPU delta;
  * every other "@@ <section> k=v ..." line is parsed into metrics.json under "guest".
RUN_HLE=1 adds the emulator's per-import call counts (SHARPEMU_PROFILE_IMPORTS), RUN_GC=1 .NET GC counts / pause /
allocation (SHARPEMU_LOG_MEMORY). Both change what is being measured (they add work), so use them for counts, not CPU.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_BIN = os.path.join(REPO, "artifacts/profiles/linux/bin/Release/net10.0/linux-x64/SharpEmu")
CLK_TCK = os.sysconf("SC_CLK_TCK")

PERF_EVENTS = [
    ("task-clock", "task_clock"),
    ("context-switches", "ctx_switches"),
    ("cpu-migrations", "cpu_migrations"),
    ("instructions", "instructions"),
    ("cycles", "cycles"),
]
PERF_SYSCALLS = [
    ("syscalls:sys_enter_futex", "futex"),
    ("syscalls:sys_enter_sched_yield", "sched_yield"),
    ("syscalls:sys_enter_nanosleep", "nanosleep"),
    ("syscalls:sys_enter_clock_nanosleep", "clock_nanosleep"),
]


def to_num(s):
    for conv in (int, float):
        try:
            return conv(s)
        except ValueError:
            pass
    return s


def parse_kv(rest):
    out = {}
    for tok in rest.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = to_num(v)
    return out


def snap_threads(pid):
    """Per-thread comm, on-CPU ns (schedstat) and context switches."""
    threads = {}
    base = f"/proc/{pid}/task"
    try:
        tids = os.listdir(base)
    except OSError:
        return threads
    for t in tids:
        try:
            with open(f"{base}/{t}/comm") as f:
                comm = f.read().strip()
            with open(f"{base}/{t}/schedstat") as f:
                run_ns, wait_ns, slices = (int(x) for x in f.read().split())
            with open(f"{base}/{t}/status") as f:
                st = f.read()
            vol = int(re.search(r"^voluntary_ctxt_switches:\s+(\d+)", st, re.M).group(1))
            nonvol = int(re.search(r"^nonvoluntary_ctxt_switches:\s+(\d+)", st, re.M).group(1))
        except (OSError, AttributeError, ValueError):
            continue
        threads[int(t)] = dict(comm=comm, run_ns=run_ns, wait_ns=wait_ns, slices=slices, vol=vol, nonvol=nonvol)
    return threads


def proc_cpu_s(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / CLK_TCK  # utime + stime, includes exited threads
    except (OSError, IndexError, ValueError):
        return None


def sys_busy_s():
    """Total non-idle CPU seconds of the whole machine (/proc/stat, first line)."""
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    # user nice system idle iowait irq softirq steal
    return (v[0] + v[1] + v[2] + v[5] + v[6] + (v[7] if len(v) > 7 else 0)) / CLK_TCK


def wait_quiet(limit_s, max_cores):
    """Block until the machine, apart from nothing of ours, is quiet for a 2 s sample (or the limit passes).
    Returns the last measured 'other' core usage."""
    deadline = time.monotonic() + limit_s
    while True:
        a, t0 = sys_busy_s(), time.monotonic()
        time.sleep(2.0)
        cores = (sys_busy_s() - a) / (time.monotonic() - t0)
        if cores < max_cores or time.monotonic() > deadline:
            return cores


def total_run_s(snap):
    return sum(t["run_ns"] for t in snap.values()) / 1e9


class Run:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.lines = []
        self.marks = {}          # name -> (monotonic, snapshot)
        self.phases = []         # finished phases
        self.open_phases = {}
        self.guest_rows = []     # (section, dict)
        self.done = threading.Event()
        self.start = None        # (t, threads, proc_cpu)
        self.end = None
        self.perf = None
        self.perf_ctl = None
        self.perf_ack = None
        self.imports_reports = []
        self.mem_reports = []
        self.perf_enabled_at = self.perf_disabled_at = self.end_snap_at = None
        self.sys_start = self.sys_end = None
        self.finished_at = None
        self.idents = {}         # (comm, idx) -> dict(tid, delta_s) from the guests' ident handshake
        self.ident_base = None
        self.ident_failed = []
        self.pid = None

    # --- perf --------------------------------------------------------------------------------------------
    def perf_cmd(self, word):
        """Send enable/disable to perf stat and wait for its ack; returns the monotonic time of the ack (the moment
        the counters really changed state) or None."""
        if not self.perf_ctl:
            return None
        try:
            os.write(self.perf_ctl, (word + "\n").encode())
            os.read(self.perf_ack, 16)
        except OSError:
            return None
        return time.monotonic()

    def start_perf(self, pid, out_csv, events):
        ctl_r, ctl_w = os.pipe()
        ack_r, ack_w = os.pipe()
        cmd = ["perf", "stat", "-p", str(pid), "-x,", "-D", "-1", "--control", f"fd:{ctl_r},{ack_w}", "-o", out_csv]
        for ev in events:
            cmd += ["-e", ev]
        self.perf = subprocess.Popen(cmd, pass_fds=(ctl_r, ack_w), stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        os.close(ctl_r)
        os.close(ack_w)
        self.perf_ctl, self.perf_ack = ctl_w, ack_r
        time.sleep(0.3)
        if self.perf.poll() is not None:
            err = self.perf.stderr.read().decode(errors="replace")
            self.perf = None
            self.perf_ctl = None
            return err or "perf exited"
        return None

    # --- line handling -----------------------------------------------------------------------------------
    def on_line(self, line, t):
        self.lines.append((t, line))
        if line.startswith("@@ "):
            self.on_guest_line(line[3:].rstrip(), t)
        elif line.startswith("[IMPORTS]"):
            self.on_imports_line(line, t)
        elif "[MEM] dotnet" in line or line.startswith("[MEM]"):
            self.on_mem_line(line, t)
        if "SharpEmu execution completed" in line:
            self.finished_at = t

    def on_guest_line(self, body, t):
        m = re.match(r"MARK (\S+)(.*)", body)
        if m:
            kind, rest = m.group(1), parse_kv(m.group(2))
            if kind == "start":
                self.start = (t, snap_threads(self.pid), proc_cpu_s(self.pid))
                self.sys_start = sys_busy_s()
                self.perf_enabled_at = self.perf_cmd("enable")
            elif kind == "end":
                # Snapshot first: perf's ack can take tens of ms when the machine is saturated.
                self.end = (t, snap_threads(self.pid), proc_cpu_s(self.pid))
                self.sys_end = sys_busy_s()
                self.end_snap_at = time.monotonic()
                self.perf_disabled_at = self.perf_cmd("disable")
                self.end_rest = rest
            elif kind == "ident_begin":
                self.ident_base = snap_threads(self.pid)
            elif kind == "ident_end" and self.ident_base is not None:
                key = (rest.get("comm"), rest.get("idx"))
                burn = rest.get("cpu_ms", 80) / 1000.0
                s1 = snap_threads(self.pid)
                comm = str(key[0]).replace("_", " ")
                known = {v["tid"] for v in self.idents.values()}  # already identified: cannot be this one, whatever they just did
                deltas = sorted(((v["run_ns"] - self.ident_base.get(tid, {}).get("run_ns", 0)) / 1e9, tid)
                                for tid, v in s1.items() if v["comm"] == comm and tid not in known)[::-1]
                # Only this guest thread burned CPU since the previous snapshot (everything else is idle), so it is the biggest grower:
                # accept it when it grew by at least 60% of the burn and by 2x as much as the runner-up of the same name among the
                # not yet identified threads.
                if deltas and deltas[0][0] >= 0.6 * burn and (len(deltas) == 1 or deltas[0][0] >= 2 * deltas[1][0]):
                    self.idents[key] = dict(tid=deltas[0][1], delta_s=round(deltas[0][0], 4))
                else:
                    self.ident_failed.append(f"{comm}#{key[1]}: top deltas {[round(d, 3) for d, _ in deltas[:3]]} for burn {burn}")
                self.ident_base = s1
            elif kind == "phase_start":
                self.open_phases[rest.get("name")] = (t, snap_threads(self.pid))
            elif kind == "phase_end":
                name = rest.get("name")
                if name in self.open_phases:
                    t0, s0 = self.open_phases.pop(name)
                    s1 = snap_threads(self.pid)
                    cpu = sum(max(0, v["run_ns"] - s0.get(k, {}).get("run_ns", 0)) for k, v in s1.items()) / 1e9
                    wall = t - t0
                    self.phases.append(dict(name=name, wall_s=round(wall, 4), cpu_s=round(cpu, 4),
                                            cores_busy=round(cpu / wall, 3) if wall > 0 else None))
            return
        toks = body.split()
        if not toks:
            return
        self.guest_rows.append((toks[0], parse_kv(" ".join(toks[1:]))))

    def on_imports_line(self, line, t):
        m = re.match(r"\[IMPORTS\] t=([\d.]+)s calls=(\d+)", line)
        if m:
            self.imports_reports.append(dict(arrival=t, uptime_s=float(m.group(1)), calls=int(m.group(2)), nids={}))
            return
        m = re.match(r"\[IMPORTS\]\s+(\S+)\s+n=(\d+) total_ms=([\d.]+)", line)
        if m and self.imports_reports:
            self.imports_reports[-1]["nids"][m.group(1)] = int(m.group(2))

    def on_mem_line(self, line, t):
        m = re.search(r"gc0=(\d+) gc1=(\d+) gc2=(\d+) pause_ms=(\d+)\(\+\d+\) alloc_mb=(\d+)", line)
        if m:
            self.mem_reports.append(dict(arrival=t, gc0=int(m.group(1)), gc1=int(m.group(2)), gc2=int(m.group(3)),
                                         pause_ms=int(m.group(4)), alloc_mb=int(m.group(5))))


def read_perf_csv(path):
    out = {}
    if not os.path.exists(path):
        return out
    names = {ev: key for ev, key in PERF_EVENTS + PERF_SYSCALLS}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3:
                continue
            val, ev = parts[0], parts[2]
            key = names.get(ev)
            if key is None:
                continue
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = None  # <not counted> / <not supported>
    return out


def flatten_guest(rows):
    """section -> metrics.json "guest" tree plus a flat {dotted.name: number} map for compare.py."""
    tree, flat, counts = {}, {}, {}
    for sec, kv in rows:
        counts[sec] = counts.get(sec, 0) + 1
    for sec, kv in rows:
        tree.setdefault(sec, []).append(kv)
        key = None
        if counts[sec] == 1:
            key = sec
        elif "name" in kv:
            key = f"{sec}.{kv['name']}"
        if key:
            for k, v in kv.items():
                if isinstance(v, (int, float)) and k != "name":
                    flat[f"{key}.{k}"] = v
    return tree, flat


def gbusy_rows(rows):
    """[(host comm, idx, busy_s)] from the guests' "@@ gbusy" rows (work_ns + pace_ns = CPU burned on purpose)."""
    out = []
    for sec, kv in rows:
        if sec != "gbusy":
            continue
        comm = "SharpEmu Emulat" if kv.get("comm") == "main" else str(kv.get("comm", "")).replace("_", " ")
        out.append((comm, kv.get("idx", 0), (kv.get("work_ns", 0) + kv.get("pace_ns", 0)) / 1e9))
    return out


def diff_threads(s0, s1, wall, gbusy=None, idents=None):
    """Per-thread CPU over the window. A guest thread's busy time is matched to a host thread exactly: by the tid its ident handshake
    produced, or by name when only one host thread has that name (the entry thread). Anything else gets no guest_busy_s. A matched
    row whose guest_busy_s exceeds its CPU by more than 2% is untrusted: emu_cpu_s is null and the reason is recorded."""
    rows = []
    seen = {}
    for tid in sorted(s1):
        n = s1[tid]["comm"]
        seen[n] = seen.get(n, 0) + 1
    # tid -> busy_s
    busy_of, unmatched = {}, []
    by_tid = {v["tid"]: key for key, v in (idents or {}).items()}
    for comm, idx, busy in (gbusy or []):
        ident = (idents or {}).get((comm.replace(" ", "_"), idx))
        if ident:
            busy_of[ident["tid"]] = busy
        elif seen.get(comm) == 1:
            busy_of[next(t for t, v in s1.items() if v["comm"] == comm)] = busy
        else:
            unmatched.append((comm, idx, busy))
    gbusy_comms = {c for c, _, _ in (gbusy or [])}
    order = {}
    for tid, v in sorted(s1.items()):
        b = s0.get(tid)
        # A thread can reuse a tid only after exiting; compare comm to be safe.
        if b and b["comm"] != v["comm"]:
            b = None
        run = v["run_ns"] - (b["run_ns"] if b else 0)
        order[v["comm"]] = k = order.get(v["comm"], -1) + 1
        row = dict(tid=tid, name=v["comm"] if seen[v["comm"]] == 1 else f"{v['comm']}#{k}", cpu_s=round(run / 1e9, 4),
                   cores=round(run / 1e9 / wall, 3) if wall > 0 else None,
                   vol_ctx=v["vol"] - (b["vol"] if b else 0), nonvol_ctx=v["nonvol"] - (b["nonvol"] if b else 0),
                   new_in_window=b is None)
        if tid in by_tid:
            row["guest_idx"] = by_tid[tid][1]
        cpu = run / 1e9
        if tid in busy_of:
            busy = busy_of[tid]
            row["guest_busy_s"] = round(busy, 4)
            if busy > cpu * 1.02:
                row["emu_cpu_s"] = None
                row["emu_untrusted"] = f"guest_busy_s {busy:.3f} > cpu_s {cpu:.3f} x 1.02: thread match or accounting wrong"
            else:
                row["emu_cpu_s"] = round(max(0.0, cpu - busy), 4)
        elif v["comm"] in gbusy_comms:
            row["emu_cpu_s"] = None
            row["emu_untrusted"] = "guest thread not identified (no ident handshake for this tid)"
        rows.append(row)
    rows.sort(key=lambda r: -r["cpu_s"])
    return rows, unmatched


def inner_window(reports, run):
    """The reports to difference: the first one at or after the start marker and the last one at or before the end marker, so
    the interval lies entirely inside the measured window (rates are computed over exactly that interval)."""
    if not reports or run.start is None or run.end is None:
        return None
    inside = [r for r in reports if run.start[0] <= r["arrival"] <= run.end[0]]
    if len(inside) < 2:
        return None
    return inside[0], inside[-1]


def imports_delta(run):
    w = inner_window(run.imports_reports, run)
    if not w:
        return None
    a, b = w
    dt = b["uptime_s"] - a["uptime_s"]  # the emulator's own clock for both reports
    if dt <= 0:
        return None
    per_s = {}
    for nid, n in b["nids"].items():
        d = n - a["nids"].get(nid, 0)
        if d:
            per_s[nid] = round(d / dt, 1)
    return dict(window_s=round(dt, 2), calls_per_s=round((b["calls"] - a["calls"]) / dt, 1),
                per_nid_per_s=dict(sorted(per_s.items(), key=lambda kv: -kv[1])))


def gc_delta(run):
    w = inner_window(run.mem_reports, run)
    if not w:
        return None
    a, b = w
    dt = b["arrival"] - a["arrival"]  # [MEM] lines carry only whole-second uptime, so use the lines' arrival times
    if dt <= 0:
        return None
    return dict(window_s=round(dt, 2), gc0=b["gc0"] - a["gc0"], gc1=b["gc1"] - a["gc1"], gc2=b["gc2"] - a["gc2"],
                pause_ms=b["pause_ms"] - a["pause_ms"], alloc_mb=b["alloc_mb"] - a["alloc_mb"],
                alloc_mb_per_s=round((b["alloc_mb"] - a["alloc_mb"]) / dt, 2),
                note="deltas over the first..last 1 s report inside the measured window; alloc_mb has 1 MB resolution")


def run_once(args, out_dir, attempt):
    env = dict(os.environ)
    env.setdefault("DOTNET_ROOT", os.path.expanduser("~/.dotnet"))
    env["PATH"] = env["DOTNET_ROOT"] + os.pathsep + env.get("PATH", "")
    env.setdefault("SDL_VIDEODRIVER", "offscreen")
    env.setdefault("SDL_AUDIODRIVER", "dummy")
    if args.hle:
        env["SHARPEMU_PROFILE_IMPORTS"] = "1"
        env["SHARPEMU_PROFILE_IMPORTS_INTERVAL"] = "1"
        env["SHARPEMU_PROFILE_IMPORTS_TOP"] = "80"
    if args.gc:
        env["SHARPEMU_LOG_MEMORY"] = "1"
        env["SHARPEMU_LOG_MEMORY_SECONDS"] = "1"
    for kv in args.env:
        k, v = kv.split("=", 1)
        env[k] = v

    run = Run(args, env)
    cmd = [args.bin, f"--log-level={args.log_level}", "--no-log-file", os.path.abspath(args.guest)]
    if args.nice:  # optional: lower the emulator's priority so the harness/perf react promptly (both A/B arms alike)
        cmd = ["nice", "-n", str(args.nice)] + cmd
    proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    run.pid = proc.pid
    csv = os.path.join(out_dir, "perf.csv")
    perf_note = None
    if not args.no_perf:
        err = run.start_perf(proc.pid, csv, [e for e, _ in PERF_EVENTS + PERF_SYSCALLS])
        if err:  # e.g. tracepoints not readable: retry with the software/hardware events only
            perf_note = "syscall tracepoints unavailable: " + err.strip().splitlines()[-1]
            err2 = run.start_perf(proc.pid, csv, [e for e, _ in PERF_EVENTS])
            if err2:
                perf_note += "; perf unavailable: " + err2.strip().splitlines()[-1]

    def reader():
        for raw in proc.stdout:
            run.on_line(raw.decode(errors="replace").rstrip("\n"), time.monotonic())
        run.done.set()

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    t_launch = time.monotonic()
    timed_out = False
    while not run.done.is_set():
        run.done.wait(0.5)
        elapsed = time.monotonic() - t_launch
        limit = args.timeout if run.end is None else (run.end[0] - t_launch) + 60
        if run.finished_at is not None:  # the emulator reported its result but did not exit (e.g. after a guest trap)
            limit = min(limit, (run.finished_at - t_launch) + 20)
        if elapsed > limit:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            break
    th.join(5)
    rc = proc.wait()
    if run.perf:
        run.perf.send_signal(signal.SIGINT)
        try:
            run.perf.wait(10)
        except subprocess.TimeoutExpired:
            run.perf.kill()
    with open(os.path.join(out_dir, "guest.log"), "w") as f:
        for _, line in run.lines:
            f.write(line + "\n")
    return run, rc, timed_out, perf_note, csv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("guest")
    ap.add_argument("label", nargs="?", default="run")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--bin", default=os.environ.get("SHARPEMU_BIN", DEFAULT_BIN))
    ap.add_argument("--env", action="append", default=[], help="K=V passed to the emulator (repeatable)")
    ap.add_argument("--timeout", type=float, default=300, help="seconds until the marker phase starts (then +60 after end)")
    ap.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "info"))
    ap.add_argument("--hle", action="store_true", default=os.environ.get("RUN_HLE") == "1")
    ap.add_argument("--gc", action="store_true", default=os.environ.get("RUN_GC") == "1")
    ap.add_argument("--no-perf", action="store_true", default=os.environ.get("NO_PERF") == "1")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--nice", type=int, default=int(os.environ.get("SYNTH_NICE", "0")), help="nice level of the emulator (default 0)")
    ap.add_argument("--wait-quiet", type=float, default=float(os.environ.get("SYNTH_WAIT_QUIET", "60")),
                    help="before launching, wait up to this many seconds for other processes to use < --quiet-cores cores")
    ap.add_argument("--quiet-cores", type=float, default=float(os.environ.get("SYNTH_QUIET_CORES", "0.75")))
    args = ap.parse_args()

    runs_root = os.environ.get("SYNTH_RUNS", "/home/xnetcat/Projects/sharpemu/ds-cpu/synth-runs")
    gname = os.path.splitext(os.path.basename(args.guest))[0]
    out_dir = args.out_dir or os.path.join(runs_root, f"{time.strftime('%Y%m%d-%H%M%S')}-{gname}-{args.label}")
    os.makedirs(out_dir, exist_ok=True)

    startup_retries = 0
    quiet_before = None
    for attempt in range(args.retries + 1):
        if args.wait_quiet > 0:
            quiet_before = wait_quiet(args.wait_quiet, args.quiet_cores)
        run, rc, timed_out, perf_note, csv = run_once(args, out_dir, attempt)
        text = "\n".join(l for _, l in run.lines)
        # SharpEmu occasionally fails before the guest starts ("Unable to reserve an import stub region");
        # that is a loader flake, not a measurement, so retry.
        if run.start is None and "Unable to reserve an import stub region" in text and attempt < args.retries:
            startup_retries += 1
            continue
        break

    perf = read_perf_csv(csv)
    metrics = dict(guest=os.path.basename(args.guest), label=args.label, out_dir=out_dir, emulator=args.bin,
                   env=[kv for kv in args.env], run_hle=args.hle, run_gc=args.gc,
                   nice=args.nice, host_cpus=os.cpu_count(), startup_retries=startup_retries, exit_code=rc, timed_out=timed_out)
    metrics["result_ok"] = ("Result=ORBIS_GEN2_OK" in text) and ("@@ DONE" in text) and rc == 0
    if quiet_before is not None:
        metrics["other_cores_before_launch"] = round(quiet_before, 2)
    if perf_note:
        metrics["perf_note"] = perf_note
    try:
        metrics["git_head"] = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        pass

    tree, flat = flatten_guest(run.guest_rows)
    run.guest_flat = flat
    metrics["guest_results"] = tree
    metrics["guest_flat"] = flat

    patcher = [l for _, l in run.lines if "spin-nap" in l or "SPIN-NAP" in l or "red-zone" in l]
    metrics["patcher_log"] = patcher
    for l in patcher:
        m = re.search(r"spin-nap: (functions=.*)", l)
        if m:
            metrics["spin_nap"] = parse_kv(m.group(1).rstrip("."))

    if run.start and run.end:
        wall = run.end[0] - run.start[0]
        metrics["wall_s"] = round(wall, 4)
        metrics["guest_measured_s"] = round(run.end_rest.get("measured_us", 0) / 1e6, 4)
        # perf's window is enable-ack .. disable-ack (the ack follows the real state change); it differs from the marker
        # window by the control latency, so perf-derived rates divide by it.
        pw = wall
        if run.perf_enabled_at and run.perf_disabled_at:
            pw = run.perf_disabled_at - run.perf_enabled_at
            metrics["perf_window_s"] = round(pw, 4)
            metrics["perf_disable_lag_ms"] = round((run.perf_disabled_at - run.end[0]) * 1000, 1)
            metrics["end_snapshot_lag_ms"] = round((run.end_snap_at - run.end[0]) * 1000, 1)
            metrics["perf_enable_lag_ms"] = round((run.perf_enabled_at - run.start[0]) * 1000, 1)
        tc = perf.get("task_clock")
        if tc is not None:
            metrics["task_clock_s"] = round(tc / 1000.0, 4)
            metrics["cores_busy_perf"] = round(tc / 1000.0 / pw, 3)
        # Threads present at both snapshots plus threads created inside the window (their whole run time counts).
        s_cpu = sum(v["run_ns"] - (run.start[1].get(k, {}).get("run_ns", 0)) for k, v in run.end[1].items()) / 1e9
        metrics["thread_cpu_s"] = round(s_cpu, 4)
        metrics["cores_busy_threads"] = round(s_cpu / wall, 3)
        if run.start[2] is not None and run.end[2] is not None:
            metrics["proc_cpu_s"] = round(run.end[2] - run.start[2], 3)  # /proc stat ticks (10 ms), includes exited threads
            metrics["cores_busy_proc"] = round((run.end[2] - run.start[2]) / wall, 3)
        # Headline: perf task-clock over perf's own (ack-timed) window; /proc/pid/stat when perf is unavailable.
        if "cores_busy_perf" in metrics:
            metrics["cores_busy"], metrics["cores_busy_src"] = metrics["cores_busy_perf"], "perf task-clock"
        elif "cores_busy_proc" in metrics:
            metrics["cores_busy"], metrics["cores_busy_src"] = metrics["cores_busy_proc"], "/proc/pid/stat"
        if run.sys_start is not None and run.sys_end is not None and metrics.get("proc_cpu_s") is not None:
            other = ((run.sys_end - run.sys_start) - metrics["proc_cpu_s"]) / wall
            metrics["host_other_cores"] = round(max(0.0, other), 2)  # CPU used by everything that is not this emulator
        # emu_cpu: process CPU minus the CPU the guest burns on purpose (job bodies, guarded-call bodies, pacing busy-waits,
        # accounted by the guests with spin_cpu/spin_until and reported as "@@ gbusy" rows). What is left is the emulator's
        # cost of serving the guest (plus the guest's own polling code, which is the workload).
        gb = sum(kv.get("work_ns", 0) + kv.get("pace_ns", 0) for sec, kv in run.guest_rows if sec == "gbusy") / 1e9
        base = metrics.get("task_clock_s", metrics.get("proc_cpu_s"))
        metrics["guest_busy_s"] = round(gb, 4)
        if base is not None:
            metrics["emu_cpu_s"] = round(base - gb, 4)
            metrics["emu_cores"] = round((base - gb) / pw, 3)
            for key, label in (("jobpool.total.jobs", "job"), ("scream.total.guards", "guard")):
                n = run.guest_flat.get(key) if hasattr(run, "guest_flat") else None
                if n:
                    metrics[f"cpu_us_per_{label}"] = round(base * 1e6 / n, 2)
                    metrics[f"emu_cpu_us_per_{label}"] = round((base - gb) * 1e6 / n, 2)
        for key in ("ctx_switches", "cpu_migrations", "instructions", "cycles", "futex", "sched_yield", "nanosleep", "clock_nanosleep"):
            v = perf.get(key)
            if v is not None:
                metrics[key] = int(v)
                metrics[f"{key}_per_s"] = round(v / pw, 1)
        if perf.get("instructions") and perf.get("cycles"):
            metrics["ipc"] = round(perf["instructions"] / perf["cycles"], 3)
        gbusy = gbusy_rows(run.guest_rows)
        thr, unmatched = diff_threads(run.start[1], run.end[1], wall, gbusy, run.idents)
        metrics["idents"] = len(run.idents)
        if run.ident_failed:
            metrics["ident_failures"] = run.ident_failed
        if unmatched:
            metrics["gbusy_unmatched"] = [f"{c}#{i}" for c, i, _ in unmatched]
        # Per-thread emu_cpu_s must add up to the total: threads with a trusted row contribute cpu - busy, the others their cpu.
        trusted_sum = sum((t["emu_cpu_s"] if t.get("emu_cpu_s") is not None else t["cpu_s"]) for t in thr)
        metrics["thread_emu_cpu_sum_s"] = round(trusted_sum, 4)
        metrics["untrusted_thread_rows"] = sum(1 for t in thr if t.get("emu_untrusted"))
        if "emu_cpu_s" in metrics and metrics["emu_cpu_s"]:
            metrics["thread_emu_cpu_sum_vs_total_pct"] = round(100.0 * (trusted_sum - metrics["emu_cpu_s"]) / metrics["emu_cpu_s"], 2)
        metrics["threads"] = thr[:40]
        metrics["thread_count"] = len(run.end[1])
        metrics["top_threads"] = [f"{t['name']}[{t['tid']}]={t['cpu_s']:.2f}s" for t in thr[:3]]
    else:
        metrics["error"] = "no start/end marker seen (guest did not reach the measured phase)"
    if run.phases:
        metrics["phases"] = run.phases
        for p in run.phases:
            metrics.setdefault("guest_flat", {})[f"phase.{p['name']}.cores_busy"] = p["cores_busy"]
            metrics["guest_flat"][f"phase.{p['name']}.cpu_s"] = p["cpu_s"]
    reasons = []
    if not metrics["result_ok"]:
        reasons.append("run did not finish with Result=ORBIS_GEN2_OK")
    grace = run.end_rest.get("grace_ms") if run.end else None
    if grace is not None:
        metrics["grace_ms"] = grace
        for key in ("perf_disable_lag_ms", "end_snapshot_lag_ms"):
            if metrics.get(key) is not None and metrics[key] >= grace:
                reasons.append(f"{key}={metrics[key]} >= guest grace {grace} ms (counters include the guest shutting down)")
    if "error" in metrics:
        reasons.append(metrics["error"])
    metrics["valid"] = not reasons
    if reasons:
        metrics["invalid_reasons"] = reasons
    if args.hle:
        metrics["hle"] = imports_delta(run)
    if args.gc:
        metrics["gc"] = gc_delta(run)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1, sort_keys=True)

    # One-line summary for humans / ab.sh.
    print(f"[{args.label}] {metrics['guest']} ok={metrics['result_ok']} valid={metrics['valid']} emu_cpu_s={metrics.get('emu_cpu_s')} wall={metrics.get('wall_s')}s "
          f"cores={metrics.get('cores_busy')} other={metrics.get('host_other_cores')} ctx/s={metrics.get('ctx_switches_per_s')} futex/s={metrics.get('futex_per_s')} "
          f"yield/s={metrics.get('sched_yield_per_s')} top={metrics.get('top_threads')} dir={out_dir}")
    return 0 if metrics["result_ok"] and metrics["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
