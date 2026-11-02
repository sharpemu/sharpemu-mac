#!/usr/bin/env bash
# Copyright (C) 2026 SharpEmu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Usage: run-all.sh [OUT_ROOT]
# Builds every guest at its defaults (DURATION_S=20, WARMUP_S=5) plus the variants used to sanity-check the
# harness, runs each once (one emulator at a time), and prints a summarize.py table. OUT_ROOT defaults to
# $SYNTH_RUNS/all-<time>.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=${SYNTH_RUNS:-/home/xnetcat/Projects/sharpemu/ds-cpu/synth-runs}
out=${1:-$root/all-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$out/bin"
export SYNTH_RUNS=$out SYNTH_WAIT_QUIET=${SYNTH_WAIT_QUIET:-180}
"$here/build.sh" -o "$out/bin" >/dev/null || exit 1
# Variants that sanity-check the harness and the patcher story (fewer workers than cores, so each poller can reach ~1 core).
"$here/build.sh" -o "$out/bin/w4" jobpool -- -DWORKERS=4 -DAFFINITY=0xe -DPRODUCER_HZ=0 -DNAP=0 >/dev/null || exit 1                   # 3 parked-job pollers + worker 0 in WaitForJobs
"$here/build.sh" -o "$out/bin/w4pause" jobpool -- -DWORKERS=4 -DAFFINITY=0xe -DPRODUCER_HZ=0 -DPARKED=0 -DDL_CHECK=0 -DNAP=0 >/dev/null || exit 1  # nothing but the WaitForJobs pause loop
"$here/build.sh" -o "$out/bin/light" jobpool -- -DLIGHT=1 >/dev/null || exit 1
"$here/build.sh" -o "$out/bin/light0" jobpool -- -DLIGHT=1 -DNAP=0 >/dev/null || exit 1                                      # spin path (SHARPEMU_SPIN_NAP_LOCKED checks)                                                 # unsaturated: ~1.35 cores of job work
"$here/build.sh" -o "$out/bin/nap0" jobpool -- -DNAP=0 >/dev/null || exit 1                                                   # DS with its idle nap switched off
"$here/build.sh" -o "$out/bin/pause" jobpool -- -DPARKED=0 -DPRODUCER_HZ=0 -DNAP=0 >/dev/null || exit 1                                # 13 workers, pure WaitForJobs path (+ dlLock after 2 s)
"$here/build.sh" -o "$out/bin/sleep" scream -- -DPACE=0 >/dev/null || exit 1                                                   # paced with usleep
"$here/build.sh" -o "$out/bin/flat" scream -- -DRATE=0 >/dev/null || exit 1                                                    # as fast as possible
runs=(
  "$out/bin/baseline.bin idle"
  "$out/bin/jobpool.bin default"
  "$out/bin/nap0/jobpool.bin nap-off"
  "$out/bin/agcbuild.bin default"
  "$out/bin/agcbuild.bin stubs-off --env SHARPEMU_AGC_PACKET_STUBS=0"
  "$out/bin/light/jobpool.bin light"
  "$out/bin/light0/jobpool.bin light-spin-nap-locked --env SHARPEMU_SPIN_NAP_LOCKED=1"
  "$out/bin/nap0/jobpool.bin spin-nap-locked --env SHARPEMU_SPIN_NAP_LOCKED=1"
  "$out/bin/w4/jobpool.bin workers4-parked"
  "$out/bin/w4/jobpool.bin workers4-parked-locked --env SHARPEMU_SPIN_NAP_LOCKED=1"
  "$out/bin/w4pause/jobpool.bin workers4-pause-only"
  "$out/bin/pause/jobpool.bin parked0-pause-path"
  "$out/bin/scream.bin default"
  "$out/bin/sleep/scream.bin pace-usleep"
  "$out/bin/flat/scream.bin flat-out"
  "$out/bin/wakeping.bin evf-and-cond"
  "$out/bin/sleeplat.bin sweep"
)
for r in "${runs[@]}"; do
  # shellcheck disable=SC2086
  "$here/run.sh" $r
done
# One run with the emulator's own per-import and GC accounting on (counts, not CPU: both add work).
RUN_HLE=1 RUN_GC=1 "$here/run.sh" "$out/bin/scream.bin" hle-gc
dirs=$(ls -d "$out"/*/ | grep -v '/bin/$')
echo
"$here/summarize.py" $dirs | tee "$out/summary.md"
