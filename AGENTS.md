<!--
SPDX-FileCopyrightText: 2026 SharpEmu Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# AGENTS.md — sharpemu-mac

## 1. What this is
macOS fork of the SharpEmu PS5 emulator: C#/.NET 10 (SDK `10.0.103`, `global.json`), Metal-only GPU backend, Avalonia GUI.
Guest x86-64 code runs natively, so the main process is `osx-x64` under Rosetta 2 (Rosetta ends after macOS 27); the
Metal GPU host is a separate native `osx-arm64` process (`src/SharpEmu.GpuHost`, default on, `SHARPEMU_GPU_PROCESS=0` off).
Apple silicon only. `origin` = github sharpemu/sharpemu-mac (PRs, CI, releases); `upstream` = sharpemu/sharpemu
(merged in, never rebased). Integration branch: `origin/main`; always `git fetch origin` and base new work on it.

## 2. Map
- `src/graphics/SharpEmu.Gpu.Metal/V2/` — Metal V2 backend: image cache, render host, guest memory windows, presenter.
- `src/SharpEmu.ShaderCompiler.Metal/` — RDNA2 (Gen5) → MSL translator (`Gen5MslTranslator.*.cs`, `Templates/`).
- `src/SharpEmu.ShaderCompiler/` — shared shader decoder, IR, resource tracking/`ResourceMaterializer`.
- `src/SharpEmu.Libs/Gpu/` — shared GPU engine: `GpuCommands` (PM4 interpreter), `Rendering`, `Pipelines`, `Images`, `Scheduling`.
- `src/SharpEmu.Libs/` (other dirs) — HLE libraries (Kernel, Agc, VideoOut, Audio…). `src/SharpEmu.HLE/` — guest memory, `PageGuard`.
- `src/SharpEmu.Core/` — loader, guest patchers, spin-nap (`Loader/`), CPU direct execution (`Cpu/`).
- `src/SharpEmu.GpuHost|GpuSplit|GpuRemote/` — two-process split. `src/platforms/macos/` — host memory, input, x64 execution.
- `src/SharpEmu.CLI/` — the one executable (CLI + GUI); `src/SharpEmu.GUI/` — Avalonia launcher.
- `src/graphics/SharpEmu.Gpu.Vulkan/` — upstream backend, not in the macOS build; semantic reference only.
- `tests/` — .NET suites (table in `docs/macos/testing.md`); `tests/scripts/` — Python build-tool tests.
- `tools/re/` — static analysis of PS5 ELFs (README); `tools/SharpEmu.Tools.ShaderDump/` — shader corpus replay.
- `scripts/macos/` — build, game harness, metrics; `scripts/mac/sync-upstream.sh` — upstream merge.
- `docs/macos/` — building, running, configuration, Metal backend, testing, troubleshooting. Read before guessing.
- `artifacts/` (git-ignored) — `bin/ obj/`, `macos/SharpEmu.app`, `runs/`, `reference/<game>/`, `re/`, `saves/ds-golden/`, `data-root/`.

## 3. Build, test, run
```sh
dotnet restore SharpEmu.slnx
dotnet build SharpEmu.slnx -c Release            # add -p:SharpEmuBuildProfile=<name> for a private bin/obj
dotnet test SharpEmu.slnx -c Release             # filter: --filter "FullyQualifiedName~<Name>"
SHARPEMU_TEST_REQUIRE_DEVICE=1 dotnet test tests/SharpEmu.Gpu.Metal.Tests -c Release
SHARPEMU_TEST_REQUIRE_DEVICE=1 dotnet test tests/SharpEmu.ShaderCompiler.Metal.Tests -c Release
python3 -m unittest discover -s tests/scripts -v
uvx reuse lint                                    # REUSE compliance, first CI step
scripts/macos/build-app.sh                        # -> artifacts/macos/SharpEmu.app (SHARPEMU_OUTPUT_DIR must be absolute)
```
- Without `SHARPEMU_TEST_REQUIRE_DEVICE=1` device tests skip and count as passed. Set it only for the Metal projects
  (on Libs it runs Vulkan device tests that fail on macOS by design).
- Parallel-safe: builds/tests in separate worktrees (each has its own `artifacts/`), as many as cores and memory
  allow. Not safe: any build or test while a game is being measured, two device-test runs at once (shared Metal
  device; flakes under load), two games.
- Durations: `ShaderCompiler.Tests` reruns ~4-13 s; the first game run after `build-app.sh` is cold (shader prewarm
  46-114 s), discard it. Full-build and full-suite times are not recorded.

## 4. Running games
- Static analysis, offline tests and batched changes come first; run games last, sparingly, to test a stated
  hypothesis or measure a finished batch. Static tools: `tools/re/` (flattened DS ELF at `artifacts/re/ds/eboot.elf`),
  `r2`, shader corpus replay (`SHARPEMU_SHADER_CORPUS_DIR=<dir>`, then
  `dotnet run -c Release --project tools/SharpEmu.Tools.ShaderDump -- corpus <dir> --metal`).
- One run: `SHARPEMU_GAMES_DIR=<games dir> scripts/macos/run-game.sh <alias|eboot> <tag> <seconds> [keyplan] [ENV=VALUE ...]`.
  Aliases: `ds` PPSA01342, `sh` PPSA10112 (Silent Hill The Short Message), `gta` PPSA04264, `poppy` PPSA18454,
  `quake` PPSA01880, `superl` PPSA06084, `oregon` PPSA19244, `void` PPSA03061, `sarah` PPSA02929.
  A `.zar` or eboot path works in place of an alias. Caps: 150 s; 600 with `SHARPEMU_GAME_ENTRY_RUN=1`;
  2400 with `SHARPEMU_GAME_SESSION_RUN=1`. Keyplan syntax: `scripts/macos/input-plan.py`.
- Lock: the script takes `~/.sharpemu/gamelock` (`SHARPEMU_GAME_LOCK`; `$SHARPEMU_COMPAT_DIR/.gamelock` if set)
  through a FIFO ticket queue in `<lock>.queue`, waits for no other SharpEmu process and for load below
  `SHARPEMU_MAX_LOAD` (default CPU count, up to 30 min). Never bypass it, never run two games, never SIGKILL a
  running game (linked to IOGPU kernel panics), never launch with `launchctl submit` (it relaunches forever).
- Output: `artifacts/runs/<timestamp>-<game>-<tag>/`: `game.log`, `game.tlog`, `frame-times.txt`, `footprint.log`,
  `system-mem.log`, `result.txt`, `run.env`, `guest.pid`, `shot-*.png` (only with `SHARPEMU_WINID` set
  to the game window id). Each checkout uses its own `artifacts/data-root`.
- Demon's Souls: `scripts/macos/games/ds.sh <tag> [seconds] [ingame|full|walk|title|boot] [ENV=VALUE ...]` → `metrics.md`
  per run (golden save `artifacts/saves/ds-golden`). `SHARPEMU_DS_*` plan knobs must be exported, not passed as args.
  Smoke set: `scripts/macos/regress.sh <tag>` (sarah quake void oregon). SH bench: `scripts/macos/perf-bench.sh`.
- Reference frames: `artifacts/reference/<game>/`, captured from real-hardware footage (yt-dlp is allowed).
- Reliable fps: load average <10 and ≥30% free memory before start; no builds, tests or other agents
  running; Activity Monitor closed. Same build/scenario/segment for before and after; ≥3 runs per arm, ABBA order or
  in-run toggles; DS noise is ±5%. Read fps from gameplay screenshots (menus render at 60). GTA: check Graphics Mode.

## 5. Project rules
- Fix root causes in generic emulation. No title-keyed or per-game hacks (title-gated patches already in
  `src/SharpEmu.Core/Loader/` are not a precedent). Why: CONTRIBUTING; hacks hide the real bug in other games.
- No silent fallbacks: refuse loudly (log + counter or fatal) instead of guessing. Why: guesses hid bugs for days.
- macOS build stays Metal/AppKit only: no Vulkan, MoltenVK, SDL (the output policy fails the build). Avalonia is fine.
- Rendering must be correct at any fps and match reference frames; a perf change that alters the image is a
  regression. Check full-size screenshots of every run before quoting numbers.
- Batch related changes, test once across the affected games plus the smoke set. Why: game runs are serialized.
- Defaults on: safe fixes need no switch; experimental ones ship default-on with a `SHARPEMU_*=0` off-switch.
- Verify NIDs/exports against the SDK and game imports; never invent an export. AnyPS5/Kyty are GPL: semantics only.
- Keep CI green: before push run `uvx reuse lint`, Release build, full test; new files need an SPDX header or a
  `REUSE.toml` entry; watch `gh run list -R sharpemu/sharpemu-mac --branch <branch>` after pushing.
- Never `pgrep -f <text>` in a command that contains that text; wait on PIDs or files instead.
- Disk: worktrees cost ~3 GB plus build output. Under 40 GB free (`df -h /System/Volumes/Data`) prune finished
  worktrees and their `artifacts/{bin,obj,profiles}` first.
- Do not switch the branch of a checkout another session may be using; work in your own worktree.

## 6. Waiting on long jobs
- Do not block an agent turn on a job longer than about 4 minutes. Start long jobs in the background with output to
  a log, then either return the PID and log path to the caller and end, or check at intervals of 4 minutes or less.
```sh
mkdir -p artifacts/agent-logs
nohup dotnet build SharpEmu.slnx -c Release > artifacts/agent-logs/build-<tag>.log 2>&1 & echo $! > artifacts/agent-logs/build-<tag>.pid
kill -0 "$(cat artifacts/agent-logs/build-<tag>.pid)" 2>/dev/null && echo running; tail -5 artifacts/agent-logs/build-<tag>.log
```
- Game runs can queue behind the lock for 30+ minutes: start `run-game.sh`/`ds.sh` the same way, report the tag, and
  let the caller read `artifacts/runs/*-<tag>/result.txt` and `metrics.md` later.
- CI: `gh run list -R sharpemu/sharpemu-mac --branch <branch> --limit 3`, then
  `gh run watch <run-id> -R sharpemu/sharpemu-mac --exit-status > artifacts/agent-logs/ci-<run-id>.log 2>&1 &`.

## 7. Investigation ledgers
- One file per topic, `ledger-<topic>.md`, kept outside the tracked tree so every worktree shares it. Your local
  agent notes say where they live on your machine; if you have none, agree a location before a long investigation.
- Read the topic's ledger before starting, especially **Ruled out**: do not re-test a ruled-out hypothesis without
  new evidence. Append before finishing; create `ledger-<topic>.md` with the same sections for a new topic.
- Sections: Status (2-3 lines, current truth), Confirmed, Ruled out, Open leads, Branches/commits, Log.
- Entry: `- YYYY-MM-DD <hypothesis or finding> → <verdict>; evidence: <numbers, run dir, test>; <branch> <sha>`.
  Add a dated one-liner to Log (newest first) and update Status if the truth changed.

## 8. Worktrees and merging
- Create, from the main checkout: `git fetch origin`, then `git worktree add ../<dir> -b <branch> origin/main`.
  Harness-created worktrees (`.claude/worktrees/`) also start at `origin/main`.
- Reset a reused worktree only when `git status --porcelain` is empty: `git fetch origin && git reset --hard origin/main`.
- Branch names: `fix/`, `perf/`, `feat/`, `diag/`, `chore/`, `followup/`, `test/<topic>`; upstream merges `sync/upstream-<sha>`
  via `scripts/mac/sync-upstream.sh`. Commit subjects start with a tag: `[GPU]`, `[Metal]`, `[Shader]`, `[Tests]`…
- A worktree app re-prewarms shaders against other builds; for side builds use `SHARPEMU_SHADER_PREWARM=0`.
- Merge verified work into main regularly through a PR to `sharpemu/sharpemu-mac` `main`, using the mandatory
  template (`.github/pull_request_template.md`). Before merging: REUSE lint, Release build, full test, device tests for
  backend changes, Python tests, and screenshot checks of the affected games plus `regress.sh`.
- Before removing a worktree: copy needed run dirs out, check that no local script references its path, then
  `git worktree remove <path>`.

## 9. Closing report
Every agent ends with this, even when stopped early:
```
Verdict: fixed | partial | not found
Branch + commit: <branch> <sha> (pushed? merged?)
Evidence: before/after numbers from the same scenario and build setup (run dirs), or test output, or frame comparison
Ruled out: <hypothesis → why>, also appended to the ledger
Next step: <one concrete action>
```
