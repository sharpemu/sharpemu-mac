#!/usr/bin/env bash
# Copyright (C) 2026 SharpEmu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Usage: selftest-teardown.sh [SECONDS_INTO_RUN=15]
# Starts a harness run (a long baseline guest), sends SIGTERM to the harness mid-run, and checks, by the PIDs the harness recorded in
# pids.json (not by name), that neither the emulator, its process group nor perf survived. Takes the bench lock like any run.
set -u
here=$(cd "$(dirname "$0")" && pwd)
mid=${1:-15}
tmp=$(mktemp -d)
"$here/build.sh" -o "$tmp/bin" baseline -- -DDURATION_S=60 -DWARMUP_S=3 >/dev/null || exit 2
SYNTH_WAIT_QUIET=0 "$here/run.sh" "$tmp/bin/baseline.bin" teardown-test --out-dir "$tmp/run" > "$tmp/harness.out" 2>&1 &
h=$!
# wait for the harness to record the emulator and perf pids
for _ in $(seq 1 600); do
  [[ -s $tmp/run/pids.json ]] && grep -q '"perf": [0-9]' "$tmp/run/pids.json" && break
  sleep 0.5
done
[[ -s $tmp/run/pids.json ]] || { echo "FAIL: harness never recorded pids (see $tmp/harness.out)"; kill -TERM "$h" 2>/dev/null; exit 2; }
sleep "$mid"
read -r emu perf pgid < <(python3 -c "import json,sys; d=json.load(open('$tmp/run/pids.json')); print(d['emulator'], d['perf'] or 0, d['emulator_pgid'])")
echo "harness=$h emulator=$emu perf=$perf pgid=$pgid; alive before SIGTERM: emulator=$(kill -0 "$emu" 2>/dev/null && echo yes || echo no) perf=$(kill -0 "$perf" 2>/dev/null && echo yes || echo no)"
kill -TERM "$h"
wait "$h"; rc=$?
sleep 1
fail=0
kill -0 "$emu" 2>/dev/null && { echo "FAIL: emulator $emu still alive"; fail=1; }
[[ $perf -gt 0 ]] && kill -0 "$perf" 2>/dev/null && { echo "FAIL: perf $perf still alive"; fail=1; }
left=$(ps -o pid= -g "$pgid" 2>/dev/null | tr -d ' \n')
[[ -n $left ]] && { echo "FAIL: processes left in the emulator's group $pgid: $left"; fail=1; }
echo "harness exit code $rc (143 expected); $(tail -1 "$tmp/harness.out")"
[[ $fail -eq 0 ]] && echo "PASS: no process of the run remains" || exit 1
rm -rf "$tmp"
