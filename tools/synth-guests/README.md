<!--
Copyright (C) 2026 SharpEmu Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# Synthetic guests and CPU harness

Small freestanding PS5-style guest programs that reproduce the wait/sync patterns that make Demon's Souls burn host
CPU in SharpEmu, plus a headless harness that measures what a run costs the host. They are the A/B bench for CPU fixes:
same guest, two emulator settings (or two builds), compare.

The guests are plain ET_DYN ELFs (clang + ld.lld, OSABI 9 / ABI version 2, every import is a NID symbol name) and
run unchanged on any SharpEmu host, Linux or macOS. The harness (`run.sh`, `ab.sh`, `compare.py`) is Linux-only
(it uses `/proc` and `perf`).

Specs referenced below are the P1 static model of DS (`p1-static-model.md`, section numbers).

## What each guest models

| Guest | Models | Source of the shape |
|---|---|---|
| `jobpool.c` | 13 `BPE JobWorkerThread CPU<k>` workers polling a scheduler laid out at DS's offsets: 8 groups, 15 Chase-Lev deques each, 256-slot external (B) and parked (A) arrays with lock words, `count[g][w]`, mailboxes, `dlLock`. A "game thread" (main) releases and re-parks jobs each 60 Hz frame, submits ready jobs into B and waits like DS's non-worker `WaitForCounter` (pause + `usleep(1)` per 1000 polls). | section 0, 1, 3, 5(a), 5(c) |
| `scream.c` | The Scream guard: 2x `scePthreadGetthreadid` plus a global `scePthreadMutexLock/Unlock` when the caller is not the synth thread, plus NCA pool-mutex jobs (4 mutex ops each). | section 2, 5(b) |
| `wakeping.c` | Event-flag ping-pong (`bpe::SyncEvent` semantics: attr 0x21, bit 1, `AND\|CLEAR_ALL`, no timeout) and a `scePthreadCond` variant; round-trip latency. | section 3 (event flags), 5(c') |
| `sleeplat.c` | `sceKernelUsleep(N)` for N in 0,1,50,100,250,1000,4000 back to back and with 2 ms gaps, then `scePthreadYield` the same two ways; per-call elapsed p50/p95/max. | section 6 (nap policies), 5(d) |
| `baseline.c` | Warm-up, then `sceKernelUsleep(100000)` for the whole measured phase: the emulator's idle floor. | - |

The GPU staging-copy guest (e) is out of scope: Linux runs Vulkan, the Mac runs Metal.

### jobpool: the instruction shapes that matter

The emulator's behaviour depends on the shape of the idle loops (the spin-nap patcher, `GuestRedZonePatcher.SpinLoops.cs`),
so the two hot loops are written in assembly (top-level `__asm__` in `jobpool.c`) to copy DS's exactly:

* `scan_slots` = ChooseJobToRun's slot scan (0x86ee12..0x86eeb7): `cmp [ptr],0; je next; lock cmpxchg [lock],ecx (1->2);
  jne next; call pred; ... xchg [lock],eax`. The `lock cmpxchg` closes a 32-byte call-free backward loop, so it is a
  locked-op loop eligible for `SHARPEMU_SPIN_NAP_LOCKED=1`; the `xchg`s sit in the loop's calling path and are not.
* `wait_for_jobs` = WaitForJobs (0x86eac0..0x86ec64): the only `pause` is followed by `call rdtsc_call` inside its
  96-byte window, so the default patcher rejects it as `non_loop` (like 0x86eb39).

`check-shapes.py guest.bin --expect-jobpool` re-derives both facts from `objdump` (build.sh runs it). Idle workers
poll while any parked job keeps `count[g][w] > 0`; with `PARKED=0` they sit in the pause loop instead, and after 2 s
without a job take `dlLock` on every pass (CPU0 does this in the default config because `AFFINITY=0x1ffe` excludes it).

### Which metrics are valid A/B metrics

`cores_busy` (total CPU) is **not** a valid A/B metric for the two guests that burn CPU on purpose, because the guest's own
busy-work hides the emulator's share. Every deliberate busy-wait goes through `spin_cpu` / `spin_until` (`common.h`), which
count only CPU ticks (a rdtsc gap >= 20 us is the thread being descheduled, not CPU), and each guest prints its totals as
`@@ gbusy comm=.. idx=.. work_ns=.. pace_ns=..` rows. The harness subtracts them: `emu_cpu_s = task_clock_s - guest_busy_s`
(total, and per thread: `threads[].guest_busy_s` / `emu_cpu_s`; guest threads are matched to host tids exactly by a start-up handshake: each thread in turn burns 80 ms of CPU between `@@ MARK ident_*` lines while everything else is idle and the harness records which tid's CPU grew; a row that cannot be identified, or whose busy time exceeds its CPU by more than 2%, gets `emu_cpu_s: null` and an `emu_untrusted` reason), with
`emu_cores`, `emu_cpu_us_per_job` (jobpool) and `emu_cpu_us_per_guard` (scream) next to the raw `cores_busy`.

| Guest | Valid A/B metrics | Not valid |
|---|---|---|
| scream | `emu_cpu_s`, `emu_cores`, `emu_cpu_us_per_guard`; `scream.gtid` p50/p95, `scream.lock` / `scream.unlock` percentiles; achieved `gtid_per_s`, `mutex_ops_per_s`; futex and ctx-switch rates | `cores_busy` (PACE=1 busy-waits to each deadline, so total CPU stays ~100% per thread whatever the emulator does; fewer parked waits even raise it) |
| jobpool | `emu_cpu_s`, `emu_cpu_us_per_job`, `jobs/s`, `parked_lat`, `lat`, `frame` percentiles | `cores_busy` with 13 spinning workers (saturates at the core count); use `-DLIGHT=1` or few workers when you want it |
| wakeping, sleeplat, baseline | `cores_busy`, `task_clock_s`, latency percentiles, per-phase CPU (sleeplat's gapped phases include the guest's own 2 ms spin, which `emu_cpu_s` removes) | - |

Job bodies are CPU-time spins: under oversubscription a job takes longer wall-clock but the same CPU, like real work.

`-DLIGHT=1` is the unsaturated jobpool preset: `PRODUCER_HZ=4500` x `JOB_US=300` = ~1.35 cores of job work (default
15000 x 300 us = 4.5 cores, which together with 12 spinning pollers overloads an 8-core host and makes the closed
frame loop run behind). The pollers still spin (that is the behaviour under study), so `cores_busy` stays near the host size
until an emulator change stops them; compare `emu_cpu_us_per_job`.

### Machine size and oversubscription

13 workers + the game thread + emulator threads oversubscribe an 8-core host, as DS does on the Mac (more workers than
P-cores). Aggregate CPU therefore saturates around the core count, and per-worker CPU is below one core. To see "one core
per polling worker", run fewer workers than cores (`-DWORKERS=4`). The host may also be shared: every run records
`host_other_cores` (CPU used by everything that is not the emulator) and the harness waits up to `SYNTH_WAIT_QUIET`
seconds (default 60) for it to drop below 0.75 cores before launching. Check that field before trusting a number.

## Build

```
tools/synth-guests/build.sh                      # all guests -> artifacts/synth-guests/<guest>.bin
tools/synth-guests/build.sh jobpool -- -DWORKERS=13 -DPARKED=3 -DDURATION_S=20
tools/synth-guests/build.sh -o /tmp/g scream -- -DRATE=0
```

Needs clang, ld.lld, objdump, python3. The emulator build is described in the repo docs (profile `linux`:
`dotnet build src/SharpEmu.CLI -c Release -p:SharpEmuBuildProfile=linux`); the harness looks for it at
`artifacts/profiles/linux/bin/Release/net10.0/linux-x64/SharpEmu` (override with `SHARPEMU_BIN` or `--bin`).
Build fails if a guest has a non-NID undefined symbol (a compiler helper such as `__udivti3` would become an unresolved
import at run time) or if jobpool's loop shapes drift.

## Knobs (all `-D`; defaults in the source headers)

Common (`common.h`): `DURATION_S=20` measured seconds, `WARMUP_S=5` idle seconds before the measured phase (the
emulator's own JIT/tiering is still running for a few seconds after the guest starts), `GRACE_MS=500` (printed in the end marker; the harness invalidates a run whose perf/snapshot lag reaches it) workers keep
running after the end marker so the harness can snapshot them.

| Guest | Knobs |
|---|---|
| jobpool | `WORKERS=13` `PARKED=3` (0 = pure pause path) `PRODUCER_HZ=15000` ready jobs/s (250 per frame; <= 256 fit array B; `-DLIGHT=1` presets 4500) `JOB_US=300` `FRAME_HZ=60` `AFFINITY=0x1ffe` `PIN_WORKERS=1` (pass `1<<k` as thread affinity) `FORK=0` (children a job pushes on its own deque) `DL_CHECK=1` `GROUP=0` |
| scream | `THREADS=15` `RATE=193000` guards/s (0 = flat out) `OWNER_PCT=50` `NCA_HZ=3500` `NCA_WORK_US=100` `CS_NS=500` `PERIOD_US=2000` `PACE=1` (busy-work to the deadline; 0 = usleep) `CONTENDED=1` (0 = one mutex per thread) |
| wakeping | `MODE=2` both (each half of the duration), `0` evf only, `1` cond only `PERIOD_US=0` (sleep between round trips) |
| sleeplat | `SWEEP_N="0,1,50,100,250,1000,4000"` `GAP_US=2000` `YIELD=1` `ONLY_N=<n>` `ONLY_GAP=0\|1` |
| baseline | `SLEEP_US=100000` |

## Run one

```
tools/synth-guests/run.sh artifacts/synth-guests/jobpool.bin label [--env SHARPEMU_SPIN_NAP_LOCKED=1] [--bin DIR/SharpEmu]
```

Output goes to `$SYNTH_RUNS/<time>-<guest>-<label>/` (default `/home/xnetcat/Projects/sharpemu/ds-cpu/synth-runs`, not in
git): `guest.log` (everything the emulator printed), `perf.csv`, `metrics.json`. The harness starts counting at the
guest's `@@ MARK start` line and stops at `@@ MARK end`, so the emulator's startup JIT is excluded.

`metrics.json` (selected fields): `valid` / `invalid_reasons`, `emu_cpu_s`, `guest_busy_s`, `wall_s`, `cores_busy` (= `task_clock_s` / perf window; `cores_busy_proc` and
`cores_busy_threads` are cross-checks), `host_other_cores`, `threads` (per-thread CPU s, vcsw/nvcsw, `#k` = k-th thread of
that name by creation), `ctx_switches[_per_s]`, `cpu_migrations`, `futex`, `sched_yield`, `nanosleep`,
`clock_nanosleep` (+ `_per_s`), `instructions`, `cycles`, `ipc`, `phases` (CPU per `phase_start/end` for sleeplat/wakeping),
`spin_nap` and `patcher_log` (what the loader's spin-nap pass did), `guest_results` / `guest_flat` (the guest's own
counters), `startup_retries` (the loader sometimes fails with "Unable to reserve an import stub region"; the harness
reruns), `perf_window_s` / `perf_disable_lag_ms` (perf's enable/disable acks can lag on a saturated host).

Optional, they add work and so are for counts rather than CPU: `RUN_HLE=1` (`SHARPEMU_PROFILE_IMPORTS`: per-import calls/s in
`hle`), `RUN_GC=1` (`SHARPEMU_LOG_MEMORY`: GC counts, pause ms and allocation MB in `gc`, 1 s resolution).

Guest output protocol: `@@ <section> key=value ...` lines (`@@ MARK start|end|phase_start|phase_end`, `@@ jobpool.total`, ...)
written with one `_write(1, ...)` per line; the harness parses them.

## Everything at once

`run-all.sh [OUT_ROOT]` builds all guests plus the sanity variants (4 workers, pause-only, usleep-paced and flat-out Scream),
runs each once with `run.sh`, adds one `RUN_HLE=1 RUN_GC=1` Scream run, and prints `summarize.py`'s markdown table
(cores busy, top 3 threads, ctx/futex/yield per second, key guest counters). `check-shapes.py guest.bin` classifies every
`pause` / locked op in a guest the way the spin-nap patcher does.

## A/B

```
tools/synth-guests/ab.sh "-" "SHARPEMU_SPIN_NAP_LOCKED=1" artifacts/synth-guests/jobpool.bin 4
tools/synth-guests/ab.sh /path/to/build-A /path/to/build-B artifacts/synth-guests/scream.bin 6
python3 tools/synth-guests/compare.py <ab>/A <ab>/B [--all] [--grep REGEX]
```

`compare.py` drops failed / invalid runs and runs with `host_other_cores` > 0.5 (`--max-other`, `--keep-invalid`) and prints how many runs each side used.

`ab.sh` runs A B B A A B B A ... (N per arm, rounded up to even), one emulator at a time, then prints `compare.py`'s median and
min..max table (`*` marks non-overlapping ranges). An arm is an emulator build directory (a copy of the linux-x64 output),
an env spec, `-`, or `DIR:ENV...`.
