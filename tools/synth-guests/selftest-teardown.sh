#!/usr/bin/env bash
# Copyright (C) 2026 SharpEmu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Usage: selftest-teardown.sh [SECONDS_INTO_RUN=8]
# Four harness runs of a long baseline guest, each interrupted mid-run in a different way, each verified by the PIDs the harness
# recorded in pids.json (not by name): the emulator, its process group and perf must all be gone afterwards.
#   term        one SIGTERM                                    (harness exits 143)
#   term-term   SIGTERM, then a second SIGTERM 0.3 s later, i.e. during the teardown   (143; the second must not abort the cleanup)
#   int         SIGINT                                         (130)
#   kill        SIGKILL: no teardown code runs, PR_SET_PDEATHSIG must take the emulator and perf down   (137)
# Each run takes the bench lock like any other, so it may queue behind other agents' runs first.
set -u
here=$(cd "$(dirname "$0")" && pwd)
mid=${1:-8}
tmp=$(mktemp -d)
"$here/build.sh" -o "$tmp/bin" baseline -- -DDURATION_S=60 -DWARMUP_S=3 >/dev/null || exit 2
overall=0

run_case() { # name expected_rc signal...
  local name=$1 want=$2; shift 2
  local d=$tmp/$name
  SYNTH_WAIT_QUIET=0 "$here/run.sh" "$tmp/bin/baseline.bin" "teardown-$name" --out-dir "$d" > "$tmp/$name.out" 2>&1 &
  local h=$!
  for _ in $(seq 1 7200); do  # up to an hour: the harness first queues for the bench lock
    [[ -s $d/pids.json ]] && grep -q '"perf": [0-9]' "$d/pids.json" && break
    sleep 0.5
  done
  [[ -s $d/pids.json ]] || { echo "[$name] FAIL: harness never recorded pids"; kill -TERM "$h" 2>/dev/null; overall=1; return; }
  sleep "$mid"
  local emu perf pgid
  read -r emu perf pgid < <(python3 -c "import json; d=json.load(open('$d/pids.json')); print(d['emulator'], d['perf'] or 0, d['emulator_pgid'])")
  echo "[$name] harness=$h emulator=$emu perf=$perf pgid=$pgid; before: emulator=$(kill -0 "$emu" 2>/dev/null && echo alive || echo dead) perf=$(kill -0 "$perf" 2>/dev/null && echo alive || echo dead)"
  local first=1 sig
  for sig in "$@"; do
    [[ $first -eq 0 ]] && sleep 0.3
    first=0
    kill "-$sig" "$h"
  done
  wait "$h"; local rc=$?
  sleep 2
  local fail=0
  kill -0 "$emu" 2>/dev/null && { echo "[$name] FAIL: emulator $emu still alive"; fail=1; }
  [[ $perf -gt 0 ]] && kill -0 "$perf" 2>/dev/null && { echo "[$name] FAIL: perf $perf still alive"; fail=1; }
  local left; left=$(ps -o pid= -g "$pgid" 2>/dev/null | tr -d ' \n')
  [[ -n $left ]] && { echo "[$name] FAIL: processes left in the emulator's group $pgid: $left"; fail=1; }
  [[ $rc -eq $want ]] || { echo "[$name] FAIL: harness exit code $rc, expected $want"; fail=1; }
  if [[ $fail -eq 0 ]]; then echo "[$name] PASS: exit $rc, nothing of the run remains"; else overall=1; fi
}

run_case term 143 TERM
run_case term-term 143 TERM TERM
run_case int 130 INT
run_case kill 137 KILL
[[ $overall -eq 0 ]] && echo "ALL PASS" || echo "SOME FAILED (outputs in $tmp)"
[[ $overall -eq 0 ]] && rm -rf "$tmp"
exit $overall
