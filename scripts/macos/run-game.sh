#!/usr/bin/env bash
# Copyright (C) 2026 SharpEmu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Runs one game on the macOS build for a bounded time and records evidence.
#
#   scripts/macos/run-game.sh <game|path/to/eboot.bin> <tag> <seconds> [keyplan] [ENV=VALUE ...]
#
#   game     ds | sarah | gta | poppy | quake | sh | superl | oregon | void (looked up under
#            $SHARPEMU_GAMES_DIR, which the shortcuts require), or a path to an eboot.bin or .zar
#   seconds  run length, capped at 150 (600 with SHARPEMU_GAME_ENTRY_RUN=1, 2400 with
#            SHARPEMU_GAME_SESSION_RUN=1 for 30-minute play sessions)
#   keyplan  comma list "t:key[:hold]" pressed through SHARPEMU_INPUT_FILE, e.g. "20:enter,30:down:1",
#            or "-" for none
#            SHARPEMU_INPUT_START_AFTER_LOG makes these times relative to the first
#            observed literal game-log marker, e.g. 'Quake Initialized'.
#            Several phases ("[@log=TEXT|flips=N|after=S[~timeout]|]steps;...") and named
#            segment markers ("t:mark:<name>") are described in input-plan.py.
#
# Only one game runs on the machine at a time: runs take a machine-wide lock directory
# ($SHARPEMU_GAME_LOCK) shared with other harnesses, and wait while any SharpEmu process is alive.
# The game is stopped with SIGTERM first, because SIGKILL with GPU work in flight has been
# linked to an IOGPU kernel panic; SIGKILL is used only if it is still alive 10 s later.
#
# Output: artifacts/runs/<timestamp>-<game>-<tag>/ with game.log, game.tlog (game.log lines
# prefixed with seconds since launch), result.txt, keys.log,
# footprint.log ("seconds footprint_mb rss_mb" per second), frame-times.txt (one line per guest
# flip, SHARPEMU_FRAME_TIMES), markers.tsv (epoch<TAB>name per plan marker), screenshots (every
# SHARPEMU_SHOT_EVERY seconds and one shot-mark-<name>.png per marker, when a window-id
# helper is available; SHARPEMU_HUD_WATCH=1 adds the `hud` marker, see hud-detect.py), run.env (launch_epoch is the wall clock at launch),
# guest.pid (the game's process id, written at launch) and window-id (its window, written by the first screenshot, for
# `screencapture -l<id> -V <seconds>` recordings). SHARPEMU_MEM_LIMIT_MB bounds the physical footprint.
#
# Memory evidence (memstat.py has the formats):
#   system-mem.log  always: every SHARPEMU_SYSMEM_EVERY (5) s the vm_stat deltas (pageins, pageouts,
#                   swapins, swapouts, compressions, decompressions), swap used and the memory pressure level.
#   memstat.log     SHARPEMU_MEMSTAT=1: every SHARPEMU_MEMSTAT_EVERY (15) s one JSON line from memstat
#                   (scripts/macos/memstat.c, built on demand): resident / dirty / compressed bytes with each VM
#                   object counted once (footprint charges every mapping of the guest's shared backing object),
#                   by category, next to the kernel's phys_footprint. Needs task_for_pid: runs through
#                   memstat (signed cs.debugger; the app needs get-task-allow, SHARPEMU_MAC_GET_TASK_ALLOW=1), or passwordless `sudo -n` when present.
#   memstat-zero-<t>.json  SHARPEMU_MEMSTAT_ZERO_AT="60,120": at the first sample at or after each second, one
#                   memstat --zero-scan line (how many resident guest_shm pages are entirely zero, plus a
#                   per-32-MiB-granule histogram); same sudo requirement, independent of SHARPEMU_MEMSTAT.
#   SHARPEMU_LOG_MEMORY=1 (read by the app) adds [MEM] lines to game.log, see MemoryTelemetry.cs.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GAMES_DIR="${SHARPEMU_GAMES_DIR:-}"
APP_DIR="${SHARPEMU_APP_DIR:-$REPO_ROOT/artifacts/macos/SharpEmu.app/Contents/MacOS}"
# Optional directory of a shared harness checkout: when set and present it holds the machine-wide
# game lock. SHARPEMU_WINID names a helper that prints the window id of a process; without it no
# screenshots are taken.
COMPAT_DIR="${SHARPEMU_COMPAT_DIR:-}"
if [[ -z "${SHARPEMU_GAME_LOCK:-}" ]]; then
  if [[ -n "$COMPAT_DIR" && -d "$COMPAT_DIR" ]]; then SHARPEMU_GAME_LOCK="$COMPAT_DIR/.gamelock"; else SHARPEMU_GAME_LOCK="$HOME/.sharpemu/gamelock"; fi
fi
WINID="${SHARPEMU_WINID:-}"
MEM_LIMIT_MB="${SHARPEMU_MEM_LIMIT_MB:-12288}"

usage() { sed -n '5,15p' "$0" >&2; exit 2; }
[[ $# -ge 3 ]] || usage

case "$1" in
  ds|sarah|gta|poppy|quake|sh|superl|oregon|void)
    [[ -n "$GAMES_DIR" ]] || { echo "set SHARPEMU_GAMES_DIR to the directory holding the games, or pass a path to an eboot.bin" >&2; exit 2; } ;;
esac
case "$1" in
  ds) EBOOT="$GAMES_DIR/demon souls.zar" ;;
  sarah) EBOOT="$GAMES_DIR/Dreaming Sarah/PPSA02929-app0/eboot.bin" ;;
  gta) EBOOT="$GAMES_DIR/GTA V/eboot.bin" ;;
  poppy) EBOOT="$GAMES_DIR/Poppy Playtime Chapter 1.zar" ;;
  quake) EBOOT="$GAMES_DIR/quake.zar" ;;
  sh) EBOOT="$GAMES_DIR/silent hill short message.zar" ;;
  superl) EBOOT="$GAMES_DIR/Superliminal.zar" ;;
  oregon) EBOOT="$GAMES_DIR/the oregon trail.zar" ;;
  void) EBOOT="$GAMES_DIR/void terrarium.zar" ;;
  *) EBOOT="$1" ;;
esac
GAME="$(basename "$1" .bin)"
TAG="$2"
SECONDS_LIMIT="$3"
PLAN="${4:--}"
INPUT_ANCHOR="${SHARPEMU_INPUT_START_AFTER_LOG:-}"
shift 3; [[ $# -gt 0 ]] && shift
# Harness settings may come as ENV=VALUE arguments (ds.sh passes its defaults that way); the last one wins.
ENV_ARGS=("$@")
setting() {
  local name="$1" value="${!1:-$2}" e
  for e in "${ENV_ARGS[@]+"${ENV_ARGS[@]}"}"; do [[ "$e" == "$name="* ]] && value="${e#*=}"; done
  printf '%s' "$value"
}

[[ -f "$EBOOT" ]] || { echo "eboot not found: $EBOOT" >&2; exit 2; }
[[ -x "$APP_DIR/SharpEmu" ]] || { echo "no SharpEmu in $APP_DIR (run scripts/macos/build-app.sh)" >&2; exit 2; }
MAX_SECONDS=150
[[ "${SHARPEMU_GAME_ENTRY_RUN:-0}" == "1" ]] && MAX_SECONDS=600
# Only an explicit session opt-in lifts the cap, and it takes the input scheduler's bound with it.
[[ "${SHARPEMU_GAME_SESSION_RUN:-0}" == "1" ]] && MAX_SECONDS=2400
[[ "$SECONDS_LIMIT" =~ ^[0-9]+$ ]] && (( 10#$SECONDS_LIMIT >= 1 && 10#$SECONDS_LIMIT <= MAX_SECONDS )) || {
  echo "refusing ${SECONDS_LIMIT}s (allowed 1..${MAX_SECONDS}; extended runs require SHARPEMU_GAME_ENTRY_RUN=1, or SHARPEMU_GAME_SESSION_RUN=1 for up to 2400)" >&2
  exit 2
}
python3 "$REPO_ROOT/scripts/macos/input-plan.py" --plan "$PLAN" --start-after-log "$INPUT_ANCHOR" --validate || exit 2

OUT="$REPO_ROOT/artifacts/runs/$(date +%Y%m%d-%H%M%S)-$GAME-$TAG"
mkdir -p "$OUT"
KEYS="$OUT/keys.state"
: > "$KEYS"
{
  echo "eboot=$EBOOT"
  echo "app=$APP_DIR"
  echo "commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD)$(git -C "$REPO_ROOT" diff --quiet || echo -dirty)"
  echo "commit_scope=checkout (use build-info.json for binary source provenance)"
  echo "keyplan=$PLAN"
  echo "input_start_after_log=$INPUT_ANCHOR"
  echo "entry_run=${SHARPEMU_GAME_ENTRY_RUN:-0}"
  echo "session_run=${SHARPEMU_GAME_SESSION_RUN:-0}"
  echo "seconds_limit=$SECONDS_LIMIT"
  for e in "$@"; do echo "$e"; done
} > "$OUT/run.env"

# Each checkout (main and every worktree) runs games on its own data root, artifacts/data-root
# as games/ds.sh does, so harness runs of many branches never write the user's real pipeline
# caches (every translator a branch builds adds its own stamp's files there) or saves. A title's
# pipeline cache and saves are cloned from the user's data root the first time a checkout runs it
# (cp -c: APFS clones, which take no space until written), so the first run starts warm.
# SHARPEMU_DATA_ROOT overrides the root.
USER_ROOT="$HOME/Library/Application Support/SharpEmu"
DATA_ROOT="${SHARPEMU_DATA_ROOT:-$REPO_ROOT/artifacts/data-root}"
case "$GAME" in
  ds) TITLE_ID=PPSA01342 ;; sarah) TITLE_ID=PPSA02929 ;; gta) TITLE_ID=PPSA04264 ;;
  poppy) TITLE_ID=PPSA18454 ;; quake) TITLE_ID=PPSA01880 ;; sh) TITLE_ID=PPSA10112 ;;
  superl) TITLE_ID=PPSA06084 ;; oregon) TITLE_ID=PPSA19244 ;; void) TITLE_ID=PPSA03061 ;;
  *) TITLE_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8-sig")).get("titleId", ""))' \
       "$(dirname "$EBOOT")/sce_sys/param.json" 2>/dev/null || true)" ;;
esac
if [[ "$TITLE_ID" =~ ^[A-Z0-9_-]+$ ]]; then
  for part in "user/pipeline_cache/$TITLE_ID" "user/savedata/$TITLE_ID"; do
    if [[ ! -e "$DATA_ROOT/$part" && -d "$USER_ROOT/$part" ]]; then
      mkdir -p "$(dirname "$DATA_ROOT/$part")"
      cp -cR "$USER_ROOT/$part" "$DATA_ROOT/$part" 2>/dev/null \
        || cp -R "$USER_ROOT/$part" "$DATA_ROOT/$part" 2>/dev/null || true
    fi
  done
fi
mkdir -p "$DATA_ROOT"
export SHARPEMU_DATA_ROOT="$DATA_ROOT"
echo "data_root=$DATA_ROOT title_id=$TITLE_ID" >> "$OUT/run.env"
# A preexisting bundle may predate provenance metadata. Never equate the current
# checkout with that binary: retain the exact payload hashes and label it unknown.
python3 - "$APP_DIR" "$OUT" <<'PY'
import hashlib, json, pathlib, sys
app, out = map(pathlib.Path, sys.argv[1:])
info = app.parent / "Resources" / "build-info.json"
record = {
    "buildInfo": json.loads(info.read_text()) if info.exists() else None,
    "binarySourceKnown": info.exists(),
    "payloads": [],
}
for p in sorted(app.glob("SharpEmu*")):
    if p.is_file():
        record["payloads"].append({"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
(out / "binary-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
if info.exists():
    (out / "build-info.json").write_bytes(info.read_bytes())
PY

# Machine-wide single-game lock; stale owners are cleared.
mkdir -p "$(dirname "$SHARPEMU_GAME_LOCK")"
LOCK_HELD=0
# Waiters queue first come, first served: a run started back to back with the last one would
# otherwise win the lock every time against one already polling. A ticket is named
# <epoch>-<pid>; tickets of exited waiters are dropped.
LOCK_QUEUE="$SHARPEMU_GAME_LOCK.queue"
mkdir -p "$LOCK_QUEUE"
LOCK_TICKET="$LOCK_QUEUE/$(date +%s)-$$"
touch "$LOCK_TICKET"
lock_turn() {
  local ticket pid
  for ticket in $(/bin/ls "$LOCK_QUEUE" 2>/dev/null | sort); do
    pid="${ticket##*-}"
    if ! ps -p "$pid" >/dev/null 2>&1; then rm -f "$LOCK_QUEUE/$ticket"; continue; fi
    [[ "$LOCK_QUEUE/$ticket" == "$LOCK_TICKET" ]]
    return
  done
  return 0
}
while :; do
  if lock_turn && mkdir "$SHARPEMU_GAME_LOCK" 2>/dev/null; then
    echo $$ > "$SHARPEMU_GAME_LOCK/pid"; LOCK_HELD=1; rm -f "$LOCK_TICKET"; break
  fi
  owner="$(cat "$SHARPEMU_GAME_LOCK/pid" 2>/dev/null || true)"
  if [[ -n "$owner" ]] && ! ps -p "$owner" >/dev/null 2>&1; then rm -rf "$SHARPEMU_GAME_LOCK"; continue; fi
  echo "waiting for game lock held by ${owner:-unknown}" >&2
  sleep 5
done
while pgrep -x SharpEmu >/dev/null; do echo "waiting for another SharpEmu to exit" >&2; sleep 5; done
# Other workloads on the machine slow the guest enough to change what a run shows (a 10x slower
# boot reaches no frames at all), so wait for the 1-minute load average to drop below
# SHARPEMU_MAX_LOAD (default: CPU count) for up to SHARPEMU_LOAD_WAIT_SECONDS (default 1800).
max_load="${SHARPEMU_MAX_LOAD:-$(sysctl -n hw.ncpu)}"
load_deadline=$(( $(date +%s) + ${SHARPEMU_LOAD_WAIT_SECONDS:-1800} ))
load_now() { sysctl -n vm.loadavg | awk '{print $2}'; }
while awk -v l="$(load_now)" -v m="$max_load" 'BEGIN { exit !(l > m) }' && (( $(date +%s) < load_deadline )); do
  echo "waiting for load $(load_now) to drop below $max_load" >&2
  sleep 15
done
echo "load_at_start=$(load_now) max_load=$max_load" >> "$OUT/run.env"

GPID=""
INPUT_PID=""
stop_game() {
  local pid="$1" i
  kill -0 "$pid" 2>/dev/null || return 0
  kill -TERM "$pid" 2>/dev/null
  for i in $(seq 1 40); do
    kill -0 "$pid" 2>/dev/null || { echo "stop=graceful after $((i / 4))s" >> "$OUT/result.txt"; return 0; }
    sleep 0.25
  done
  echo "stop=SIGKILL (alive 10s after SIGTERM)" >> "$OUT/result.txt"
  kill -KILL "$pid" 2>/dev/null
}
cleanup() {
  if [[ -n "$INPUT_PID" ]]; then
    kill -TERM "$INPUT_PID" 2>/dev/null || true
    wait "$INPUT_PID" 2>/dev/null || true
    INPUT_PID=""
  fi
  [[ -n "$GPID" ]] && stop_game "$GPID"
  GPID=""
  local sampler
  for sampler in "${SAMPLER_PIDS[@]+"${SAMPLER_PIDS[@]}"}"; do kill -TERM "$sampler" 2>/dev/null || true; done
  if (( LOCK_HELD )) && [[ "$(cat "$SHARPEMU_GAME_LOCK/pid" 2>/dev/null)" == "$$" ]]; then rm -rf "$SHARPEMU_GAME_LOCK"; fi
  LOCK_HELD=0
}
trap cleanup EXIT INT TERM

# Idle .NET build servers (MSBuild node-reuse processes, VBCSCompiler) from earlier builds pile up and
# hold gigabytes of swapped/compressed memory (161 idle MSBuild nodes once held ~8 GB) and skew
# every memory reading. They restart on demand, so a measurement run stops them first
# (SHARPEMU_KEEP_BUILD_SERVERS=1 skips this) and records the machine's memory state at launch.
if [[ "${SHARPEMU_KEEP_BUILD_SERVERS:-0}" != 1 ]]; then
  dotnet build-server shutdown > /dev/null 2>&1 || true
  pkill -f 'MSBuild.dll.*nodemode' 2> /dev/null || true
  pkill -f 'VBCSCompiler' 2> /dev/null || true
fi
{ sysctl vm.swapusage; vm_stat; } > "$OUT/system-at-launch.txt" 2>&1 || true

shot() {
  [[ -x "$WINID" ]] || return 0
  local id
  id="$("$WINID" "$GPID" 2>/dev/null)" && [[ -n "$id" ]] || return 0
  [[ -s "$OUT/window-id" ]] || echo "$id" > "$OUT/window-id"
  screencapture -x -o -l"$id" "$1" 2>/dev/null
}

(
  cd "$APP_DIR" || exit 1
  # Same gen0 budget the app bundle's Info.plist sets (GCgen0size is environment-only).
  export DOTNET_GCgen0size="${DOTNET_GCgen0size:-0x4000000}"
  for e in "$@"; do export "${e?}"; done
  # Per-flip timestamps for ds-metrics.py; an ENV=VALUE argument above may name another path.
  export SHARPEMU_FRAME_TIMES="${SHARPEMU_FRAME_TIMES:-$OUT/frame-times.txt}"
  export SHARPEMU_INPUT_FILE="$KEYS"
  export SHARPEMU_RUN_OUT="$OUT"
  exec ./SharpEmu ${SHARPEMU_ARGS:-} "$EBOOT"
) > "$OUT/game.log" 2>&1 &
GPID=$!
echo "$GPID" > "$OUT/guest.pid"
start=$(date +%s)
echo "launch_epoch=$(perl -MTime::HiRes=time -e 'printf "%.3f", time')" >> "$OUT/run.env"
# game.tlog: game.log with seconds since launch on each line, for load timelines.
perl -MTime::HiRes=time,sleep -e '
  my ($log, $pid) = @ARGV; my $t0 = time; $| = 1;
  sleep 0.05 until -e $log;
  open(my $in, "<", $log) or exit 1;
  my $drain = sub { while (defined(my $l = <$in>)) { printf "[+%8.3f] %s", time - $t0, $l } seek $in, 0, 1 };
  while (kill 0, $pid) { $drain->(); sleep 0.05 }
  $drain->();' "$OUT/game.log" "$GPID" > "$OUT/game.tlog" 2>/dev/null &
python3 "$REPO_ROOT/scripts/macos/input-plan.py" --plan "$PLAN" --keys "$KEYS" \
  --events "$OUT/keys.log" --game-log "$OUT/game.log" --start-after-log "$INPUT_ANCHOR" \
  --markers "$OUT/markers.tsv" --frame-times "$OUT/frame-times.txt" \
  --run-limit "$SECONDS_LIMIT" > "$OUT/input-plan.log" 2>&1 &
INPUT_PID=$!
next_shot=5
marks_shot=0
# SHARPEMU_HUD_WATCH=1: every 2 s look at the window for Demon's Souls' in-world HUD (hud-detect.py)
# and write the `hud` marker the first time it is on screen, the run's "controllable gameplay" signal.
hud_seen=0
hud_probe_at=-2
shot_every="${SHARPEMU_SHOT_EVERY:-10}"
peak_mb=0
peak_fp_mb=0
# SHARPEMU_FOOTPRINT_AT="60,120" records footprint -v and vmmap at the first sample at or after each second.
footprint_at=()
[[ -n "${SHARPEMU_FOOTPRINT_AT:-}" ]] && IFS=, read -r -a footprint_at <<< "$SHARPEMU_FOOTPRINT_AT"
# Load-time evidence at chosen seconds since launch (comma lists, each fires at the first
# sample at or after its second, in the background so the run keeps its pace):
#   SHARPEMU_SAMPLE_AT     native call stacks, `sample <pid> SHARPEMU_SAMPLE_SECONDS (5)`
#   SHARPEMU_TRACE_AT      managed sampled stacks of every thread for SHARPEMU_TRACE_SECONDS (10),
#                          dotnet-trace cpu-sampling -> trace-<t>.nettrace + .speedscope.json
#   SHARPEMU_GCDUMP_AT     managed heap object graph (what is live, by type and retained size),
#                          dotnet-gcdump collect -> gcdump-<t>.gcdump + gcdump-<t>.txt (report);
#                          it forces a full blocking GC in the game, so place it after measurements
#   SHARPEMU_FS_USAGE_AT   file-system calls with sizes and latencies, `sudo fs_usage -w -f filesys`
#                          for SHARPEMU_FS_USAGE_SECONDS (5); needs passwordless sudo
sample_at=(); trace_at=(); fs_usage_at=(); gcdump_at=()
[[ -n "${SHARPEMU_SAMPLE_AT:-}" ]] && IFS=, read -r -a sample_at <<< "$SHARPEMU_SAMPLE_AT"
[[ -n "${SHARPEMU_TRACE_AT:-}" ]] && IFS=, read -r -a trace_at <<< "$SHARPEMU_TRACE_AT"
[[ -n "${SHARPEMU_GCDUMP_AT:-}" ]] && IFS=, read -r -a gcdump_at <<< "$SHARPEMU_GCDUMP_AT"
[[ -n "${SHARPEMU_FS_USAGE_AT:-}" ]] && IFS=, read -r -a fs_usage_at <<< "$SHARPEMU_FS_USAGE_AT"
EVIDENCE_PIDS=()
SAMPLER_PIDS=()
python3 "$REPO_ROOT/scripts/macos/memstat.py" sysmem --pid "$GPID" --every "$(setting SHARPEMU_SYSMEM_EVERY 5)" \
  --out "$OUT/system-mem.log" > /dev/null 2>&1 &
SAMPLER_PIDS+=($!)
if [[ "$(setting SHARPEMU_MEMSTAT 0)" == 1 ]]; then
  if MEMSTAT_BIN="$("$REPO_ROOT/scripts/macos/build-memstat.sh" 2>"$OUT/memstat.err")"; then
    python3 "$REPO_ROOT/scripts/macos/memstat.py" sample --pid "$GPID" --every "$(setting SHARPEMU_MEMSTAT_EVERY 15)" \
      --out "$OUT/memstat.log" --bin "$MEMSTAT_BIN" >> "$OUT/memstat.err" 2>&1 &
    SAMPLER_PIDS+=($!)
  else
    echo "memstat disabled: could not build memstat or passwordless sudo is unavailable (see memstat.err)" >&2
  fi
fi
zero_at=()
if [[ -n "${SHARPEMU_MEMSTAT_ZERO_AT:-}" ]]; then
  if ZERO_BIN="$("$REPO_ROOT/scripts/macos/build-memstat.sh" 2>>"$OUT/memstat.err")"; then
    IFS=, read -r -a zero_at <<< "$SHARPEMU_MEMSTAT_ZERO_AT"
  else
    echo "memstat zero scans disabled: could not build memstat or passwordless sudo is unavailable (see memstat.err)" >&2
  fi
fi
status=""
while :; do
  now=$(( $(date +%s) - start ))
  if ! kill -0 "$GPID" 2>/dev/null; then wait "$GPID"; status="exited rc=$? after ${now}s"; GPID=""; break; fi
  (( now >= SECONDS_LIMIT )) && break
  # Physical footprint (what Activity Monitor shows) includes compressed pages that
  # ps -o rss leaves out; under memory pressure RSS understates a run several times over.
  read -r fp_mb rss_mb < <(python3 "$REPO_ROOT/scripts/macos/footprint.py" "$GPID" 2>/dev/null || echo "0 0")
  (( rss_mb > peak_mb )) && peak_mb=$rss_mb
  (( fp_mb > peak_fp_mb )) && peak_fp_mb=$fp_mb
  echo "$now $fp_mb $rss_mb" >> "$OUT/footprint.log"
  if (( ${#footprint_at[@]} > 0 )) && (( now >= footprint_at[0] )); then
    ( footprint -v "$GPID" > "$OUT/footprint-$now.txt" 2>&1
      vmmap "$GPID" > "$OUT/vmmap-$now.txt" 2>&1 ) &
    footprint_at=("${footprint_at[@]:1}")
  fi
  if (( ${#sample_at[@]} > 0 )) && (( now >= sample_at[0] )); then
    sample "$GPID" "${SHARPEMU_SAMPLE_SECONDS:-5}" -file "$OUT/sample-$(printf %04d "$now").txt" >/dev/null 2>&1 &
    EVIDENCE_PIDS+=($!)
    sample_at=("${sample_at[@]:1}")
  fi
  if (( ${#trace_at[@]} > 0 )) && (( now >= trace_at[0] )); then
    trace="$OUT/trace-$(printf %04d "$now")"
    ( DOTNET_ROOT="${DOTNET_ROOT:-$HOME/.dotnet}" "$HOME/.dotnet/tools/dotnet-trace" collect -p "$GPID" \
        --duration "00:00:$(printf %02d "${SHARPEMU_TRACE_SECONDS:-10}")" -o "$trace.nettrace" > "$trace.log" 2>&1
      DOTNET_ROOT="${DOTNET_ROOT:-$HOME/.dotnet}" "$HOME/.dotnet/tools/dotnet-trace" convert "$trace.nettrace" --format Speedscope -o "$trace" >> "$trace.log" 2>&1 ) &
    EVIDENCE_PIDS+=($!)
    trace_at=("${trace_at[@]:1}")
  fi
  if (( ${#gcdump_at[@]} > 0 )) && (( now >= gcdump_at[0] )); then
    gcdump="$OUT/gcdump-$(printf %04d "$now")"
    ( DOTNET_ROOT="${DOTNET_ROOT:-$HOME/.dotnet}" "$HOME/.dotnet/tools/dotnet-gcdump" collect -p "$GPID" -o "$gcdump.gcdump" > "$gcdump.log" 2>&1
      DOTNET_ROOT="${DOTNET_ROOT:-$HOME/.dotnet}" "$HOME/.dotnet/tools/dotnet-gcdump" report "$gcdump.gcdump" > "$gcdump.txt" 2>> "$gcdump.log" ) &
    EVIDENCE_PIDS+=($!)
    gcdump_at=("${gcdump_at[@]:1}")
  fi
  if (( ${#zero_at[@]} > 0 )) && (( now >= zero_at[0] )); then
    python3 "$REPO_ROOT/scripts/macos/memstat.py" zero --pid "$GPID" --bin "$ZERO_BIN" \
      --out "$OUT/memstat-zero-$(printf %04d "$now").json" >> "$OUT/memstat.err" 2>&1 &
    EVIDENCE_PIDS+=($!)
    zero_at=("${zero_at[@]:1}")
  fi
  if (( ${#fs_usage_at[@]} > 0 )) && (( now >= fs_usage_at[0] )); then
    sudo -n "$(command -v timeout)" "${SHARPEMU_FS_USAGE_SECONDS:-5}" fs_usage -w -f filesys "$GPID" \
      > "$OUT/fs_usage-$(printf %04d "$now").txt" 2>&1 &
    EVIDENCE_PIDS+=($!)
    fs_usage_at=("${fs_usage_at[@]:1}")
  fi
  if (( fp_mb > MEM_LIMIT_MB )); then status="memory limit footprint ${fp_mb}MB > ${MEM_LIMIT_MB}MB at ${now}s"; break; fi
  if [[ -f "$OUT/markers.tsv" ]]; then
    # One screenshot per plan marker, at the first sample after the marker was written.
    marks_now=$(wc -l < "$OUT/markers.tsv" | tr -d ' ')
    while (( marks_shot < marks_now )); do
      marks_shot=$(( marks_shot + 1 ))
      mark_name="$(sed -n "${marks_shot}p" "$OUT/markers.tsv" | cut -f2)"
      shot "$OUT/shot-mark-${mark_name}.png"
      # A plan marker named "stop" ends the run (SIGTERM, as at the time limit) once its screenshot is taken.
      [[ "$mark_name" == stop ]] && { status="stopped at the plan's stop marker after ${now}s"; break 2; }
    done
  fi
  if [[ "${SHARPEMU_HUD_WATCH:-0}" == 1 ]] && (( ! hud_seen )) && (( now - hud_probe_at >= 2 )); then
    hud_probe_at=$now
    shot "$OUT/hud-probe.png"
    hud_rc=0
    python3 "$REPO_ROOT/scripts/macos/hud-detect.py" "$OUT/hud-probe.png" > "$OUT/hud-probe.txt" 2> "$OUT/hud-probe.err" || hud_rc=$?
    # Every probe is logged (second, exit code, bar coverages or the error) so a missing marker can be diagnosed.
    echo "$now rc=$hud_rc $(cat "$OUT/hud-probe.txt" "$OUT/hud-probe.err" 2>/dev/null | tr '\n' ' ')" >> "$OUT/hud-detect.log"
    if (( hud_rc == 0 )); then
      hud_seen=1
      printf '%s\thud\n' "$(perl -MTime::HiRes=time -e 'printf "%.3f", time')" >> "$OUT/markers.tsv"
    fi
  fi
  if (( now >= next_shot )); then shot "$OUT/shot-$(printf %04d "$now").png"; next_shot=$(( now + shot_every )); fi
  if [[ -n "$INPUT_PID" ]] && ! kill -0 "$INPUT_PID" 2>/dev/null; then
    wait "$INPUT_PID"; input_rc=$?
    INPUT_PID=""
    if (( input_rc != 0 )); then status="input scheduler failed rc=$input_rc at ${now}s"; break; fi
  fi
  sleep 1
done
for p in "${EVIDENCE_PIDS[@]}"; do wait "$p" 2>/dev/null; done
if [[ -n "$GPID" ]]; then
  shot "$OUT/shot-final.png"
  status="${status:-alive at ${SECONDS_LIMIT}s}"
  # SHARPEMU_CAPTURE_STACKS=1 records managed (dotnet-stack, x64 runtime) and native (sample) stacks.
  if [[ "${SHARPEMU_CAPTURE_STACKS:-0}" == "1" ]]; then
    DOTNET_ROOT="${SHARPEMU_X64_DOTNET_ROOT:-$HOME/.dotnet-x64}" timeout 60 \
      "$HOME/.dotnet/tools/dotnet-stack" report -p "$GPID" > "$OUT/stacks.txt" 2>&1 || true
    sample "$GPID" 3 -file "$OUT/sample.txt" >/dev/null 2>&1 || true
  fi
fi
rm -f "$OUT/hud-probe.png" "$OUT/hud-probe.txt" "$OUT/hud-probe.err"
echo "$status" > "$OUT/result.txt"
echo "peak_footprint_mb=$peak_fp_mb" >> "$OUT/result.txt"
echo "peak_rss_mb=$peak_mb" >> "$OUT/result.txt"
cleanup
echo "$OUT"
cat "$OUT/result.txt"
