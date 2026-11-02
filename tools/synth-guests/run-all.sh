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
"$here/build.sh" -o "$out/bin/w4" jobpool -- -DWORKERS=4 -DAFFINITY=0xe -DPRODUCER_HZ=0 >/dev/null || exit 1       # 3 pollers + 1 idle: ~1 core each
"$here/build.sh" -o "$out/bin/pause" jobpool -- -DPARKED=0 -DPRODUCER_HZ=0 >/dev/null || exit 1                 # pure WaitForJobs path
"$here/build.sh" -o "$out/bin/flat" scream -- -DRATE=0 >/dev/null || exit 1                                      # as fast as possible
runs=(
  "$out/bin/baseline.bin idle"
  "$out/bin/jobpool.bin default"
  "$out/bin/jobpool.bin spin-nap-locked --env SHARPEMU_SPIN_NAP_LOCKED=1"
  "$out/bin/w4/jobpool.bin workers4-parked"
  "$out/bin/pause/jobpool.bin parked0-pause-path"
  "$out/bin/scream.bin default"
  "$out/bin/flat/scream.bin flat-out"
  "$out/bin/wakeping.bin evf-and-cond"
  "$out/bin/sleeplat.bin sweep"
)
for r in "${runs[@]}"; do
  # shellcheck disable=SC2086
  "$here/run.sh" $r
done
dirs=$(ls -d "$out"/*/ | grep -v '/bin/$')
echo
"$here/summarize.py" $dirs | tee "$out/summary.md"
