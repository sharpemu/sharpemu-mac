<!--
Copyright (C) 2026 SharpEmu Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# macOS troubleshooting

Start with the log. Run the game with `--log-file` (see
[Running games](running-games.md#logs)) and read the lines tagged `[LOADER]` and
`[METAL-V2]`. When you report a problem, attach the log.

## SharpEmu does not start

**`Unsupported process architecture ...` (exit code 5).** The process is not
x86-64. On Apple Silicon use the `osx-x64` build and install Rosetta 2:

```sh
softwareupdate --install-rosetta
arch -x86_64 /usr/bin/true    # prints nothing when Rosetta works
```

**macOS refuses to open the app.** Release bundles are ad-hoc signed and not
notarized unless the publisher supplied a Developer ID and notarization
credentials ([Building](building.md#signing-and-entitlements)). If Gatekeeper blocks a
downloaded bundle, allow it in System Settings, Privacy and Security, or remove
the quarantine attribute:

```sh
xattr -dr com.apple.quarantine SharpEmu.app
```

**The app opens and the game does not.** Check that the game folder contains
`eboot.bin` and `sce_sys`, or that a `.zar` archive has `eboot.bin` at its root.
The CLI exits with code 2 when the `eboot.bin` path does not exist.

## Settings, saves and logs are not where I expect

Every log, trace and dump is under `user/logs/` in the data root, or under
`SHARPEMU_LOG_DIR` if you set it. The data root depends on how SharpEmu was started; see
[Running games](running-games.md#caches-and-data). From inside `SharpEmu.app` it is
`~/Library/Application Support/SharpEmu`. A copy of the app on an external drive
uses that folder too, unless a `portable.txt` file sits next to the bundle. The
GUI passes its data root to the game process, so both agree.

## The first run is slow, or the window shows a progress bar

The first run of a game compiles its shaders and builds GPU pipelines. While the
prewarm compiles what the first frames need, the guest is held and the window
shows the splash with a progress bar. Later runs reuse the caches in
`user/pipeline_cache/<title ID>/metal-v2/` and start much faster. To compile
everything for a title in one go, run it once with `--precompile-only`.

- To start without the hold, set `SHARPEMU_SHADER_PREWARM_WAIT=0`. Frames then
  wait for shader compiles as they arise.
- To discard the caches, delete the title's `metal-v2` folder. They are rebuilt
  on the next run.
- A cache from another macOS version, GPU or SharpEmu translator version is
  ignored or removed automatically.

## The game stops with exit code 4

The stall watchdog ends the process when the guest makes no import progress for
20 seconds, scaled up when the machine's load average is high. It logs an error
and a snapshot first, then exits. Set `SHARPEMU_STALL_WATCHDOG_SECONDS` to a
larger value, or to `0` to turn it off.

## Memory use and paging

Titles use a lot of memory. To see where it goes, set `SHARPEMU_LOG_MEMORY=1`:
the log then carries periodic `[MEM]` lines. The detiled image cache is sized
from the memory the host has free and shrinks under memory pressure. To bound it,
set `SHARPEMU_METAL_V2_IMAGE_BUDGET_MB` (at least 256). Close other large apps
before long sessions.

## Rendering problems

A black screen, missing objects or flicker can come from the translator, the
image cache or a missing feature. To narrow it down:

1. Look in the log for `[METAL-V2][WARN]` and `[METAL-V2][ERROR]` lines. Each
   warning is printed once. `A Metal submission failed` means the driver rejected
   a command buffer; the periodic `[METAL-V2][PERF]` line counts these as
   `failed=`.
2. `SHARPEMU_METAL_V2_TRACE=1` adds detail to the log.
3. `SHARPEMU_METAL_V2_DUMP_TARGETS=<dir>` (or `=1`) writes every render pass's color target
   and every presented frame as PNG, which shows where the picture is lost. It
   slows the game down.
4. `SHARPEMU_METAL_V2_IGNORE_TESTS=1` drops culling, depth and stencil tests, to
   tell a pass that drew nothing from one that was rejected.
5. `SHARPEMU_SHADER_MSL_DUMP_DIR=<dir>` (or `=1`) dumps translated shaders, with the guest
   instruction listing, for the hashes you name in `SHARPEMU_SHADER_MSL_DUMP_HASH`.
6. The code generation switches in [Configuration](configuration.md#shader-translation-and-caches)
   (for example `SHARPEMU_MSL_STRUCTURED=0`) undo one optimization each. If one
   fixes the picture, the cause is in that optimization.

Draws that use primitive shaders are dropped without a Metal 3 device. The log
says `Mesh render pipelines are unavailable on this device`.

## Slow performance

- Read the `[METAL-V2][PERF]` line. It reports GPU busy time, host waits and
  shader compile activity, among other counters, every two seconds.
- `[METAL-V2][SLOW-WAIT]` lines name host waits longer than 20 ms. With
  `SHARPEMU_METAL_V2_WAIT_TRACE=1` they also name the shader being prepared.
- `SHARPEMU_METAL_V2_PROFILE_SHADERS=1` charges GPU time to shader hashes. It
  changes timing, so use the ratios, not the absolute frame rate.
- Compile stalls during play mean the caches are cold. See above.

## Controls do not respond

Keyboard input is delivered only while the SharpEmu window has focus, and
Command chords never reach the game. See [Running games](running-games.md#input).
Controllers are read through the GameController framework, so the controller must
be paired and connected to macOS.

## The performance display is missing

The default display is the full on-screen panel. A saved **Title bar** mode shows
the summary in the window title instead, and a full-screen window has no title
bar. See [Performance display](../performance-overlay.md).
`SHARPEMU_OVERLAY=0` starts the display hidden, and F1 shows it.

## Stopping a game

Close the window or press Command+Q. If you must stop the process from a
terminal, send SIGTERM (`kill <pid>`) and give it several seconds. Killing a
process with SIGKILL while the GPU has work in flight has been linked to kernel
panics in the IOGPU driver.
