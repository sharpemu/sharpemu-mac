<!--
Copyright (C) 2026 SharpEmu Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# Running games on macOS

SharpEmu is an experimental emulator. Compatibility and performance vary by game
and by Mac. Use only games you dumped from a console you own.

## Starting SharpEmu

**GUI.** Open `SharpEmu.app`, or run the `SharpEmu` executable with no
arguments. Add your game folders in the library settings, then select a game and
press Play. Environment variables can be set globally or per game in the GUI; see
[Configuration](configuration.md).

**Command line.** Pass the game's `eboot.bin` (or a `.zar` archive):

```sh
SharpEmu.app/Contents/MacOS/SharpEmu "/path/to/game/eboot.bin" --log-file
```

The CLI usage line, as printed by the program:

```text
SharpEmu.CLI [--strict] [--precompile-only] [--shader-prewarm-hold]
  [--shader-cache=<on|off>] [--shader-learn=<on|off>] [--trace-imports[=N]]
  [--cpu-engine=<native>] [--console=<ps5>] [--console-language=<language>]
  [--log-level=<level>] [--log-file[=<path>]]
  [--window-mode=<windowed|borderless|exclusive>] [--resolution=<WIDTHxHEIGHT>]
  [--display=<N>] [--refresh-rate=<HZ>] [--scaling=<fit|cover|stretch|integer>]
  [--vsync=<on|off>] [--hdr=<auto|on|off>] [--overlay=<on|off>]
  [--overlay-mode=<full|minimal|titlebar>]
  [--overlay-corner=<topleft|topright|bottomright|bottomleft>]
  [--debug-server[=host:port]] <path-to-eboot.bin|path-to-game.zar>
```

The GUI passes the saved video options to the game process with these same
flags. `--shader-cache`, `--shader-learn`, `--precompile-only` and
`--shader-prewarm-hold` control the shader caches; see
[Shader caches](#caches-and-data).

The process exits with a non-zero code when it cannot start: 1 for bad
arguments, 2 when the `eboot.bin` file does not exist, 3 when the video options
cannot be applied, and 5 when the host process is not x86-64.

## Game layouts

SharpEmu loads a game from a dumped application folder: the folder that
contains `eboot.bin`, with `sce_sys/param.json` (title, title ID, version),
`sce_sys/icon0.png` and the game's data and modules beside it. `/app0` is
mounted at that folder.

The library scans each configured folder recursively, to a depth of 8, for
`eboot.bin` files and `*.zar` archives.

A `.zar` is a read-only archive of a whole game folder, compressed in 64 KiB
blocks (the ZArchive format). It must contain `eboot.bin` at its root. Guest
reads come from the archive; `eboot.bin`, the modules and `sce_sys` are also
unpacked to a cache folder at `user/zar-cache/<archive name>` under the data
root, which is rebuilt if the archive changes. The GUI can pack a game folder
into a `.zar` (`src/SharpEmu.GUI/ZarPackDialog.axaml.cs`).

Game patches are described in [Game patches](../patches.md).

Mods are described in [Game mods](../mods.md). On macOS a mod's file names are
matched the way the game volume matches the game's own: without regard to case
on a case-insensitive volume (the APFS default), exactly on a case-sensitive one.

**Create shortcut** in a game's context menu puts `<game name>.app` on the
Desktop: a small application bundle with the game's cover as its icon, whose
launcher starts SharpEmu with `--shortcut <game path>`. The shortcut holds the
absolute paths of SharpEmu and of the game, so it stops working when either is
moved; delete it and create it again.

## Input

Keyboard input reaches the game only while the SharpEmu window has focus.
Command chords are left to the menu bar and never reach the guest.

| PS5 control | Keys |
| --- | --- |
| D-pad | Arrow keys |
| Cross / Circle | Z or Return / X or Escape |
| Square / Triangle | C / V |
| L1 / R1 | Q / E |
| L2 / R2 (digital) | R / F |
| Options | Tab or Backspace |
| Left stick | W A S D |
| Right stick | I J K L |

Game controllers are read through the GameController framework. The code
for DualSense lightbar, haptics and adaptive triggers is in
`src/platforms/macos/SharpEmu.Platform.MacOS/Input/`.

Window shortcuts:

| Key | Action |
| --- | --- |
| F1 | Show or hide the [performance display](../performance-overlay.md) |
| F2 | Move the performance display to the next screen corner |
| F11 | Toggle native full screen |
| Command+Q | Close the window |

The `borderless` and `exclusive` window modes both use native macOS full screen.

## Performance display

Open **Options → Rendering** to choose the display mode. It is on by default, in
Title bar mode: the summary (FPS, CPU, GPU, elapsed time) is shown in the window
title. The on-screen Full and Minimal panels are not drawn by the Metal backend.
Details are in [Performance display](../performance-overlay.md).

## Logs

- With `--log-file` the CLI also mirrors the console to a file. Without a path
  it uses `user/logs/<title ID>-<yyyyMMdd-HHmmss>.log` under the data root.
- Everything diagnostic goes under `user/logs/` in the data root
  (`~/Library/Application Support/SharpEmu/user/logs` for the app bundle):
  run logs, `gui-crash.log`, and the subfolders `videoout`, `shader-dumps`,
  `msl-dumps`, `metal-targets`, `metal-bc-uploads`, `guest-images`,
  `shader-corpus`, `dstrace`, `game_logs/<title ID>` (the guest's `/devlog/app`
  and `/hostapp`), `capture_logs`, `image-snapshots` and `volume-uploads`.
  `SHARPEMU_LOG_DIR` moves the whole folder. The dump variables that take a
  path (for example `SHARPEMU_METAL_V2_DUMP_TARGETS`) also accept `1`, which
  writes to the default subfolder; an explicit path still goes where you name it.
- `--log-level` takes a level name such as `trace`, `debug`, `info`, `warn`,
  `error`.
- Log lines are tagged by subsystem. The Metal backend writes `[METAL-V2]`
  lines to the same stream; `[METAL-V2][PERF]` is a periodic counter line, printed
  about every two seconds. [Troubleshooting](troubleshooting.md) and
  [Configuration](configuration.md) list the switches that add detail.

## Caches and data

SharpEmu writes everything under one data root:

1. `SHARPEMU_DATA_ROOT`, if set, else `SHARPEMU_USER_DIR`, if set.
2. When running from inside `SharpEmu.app`: `~/Library/Application Support/SharpEmu`.
   Writing into a signed bundle would invalidate its signature. If a file
   `portable.txt` sits in the folder that contains `SharpEmu.app`, that folder is
   used instead.
3. Otherwise the directory of the executable, if it is writable or already has a
   `user` folder, else `~/Library/Application Support/SharpEmu`.

| Path under the data root | Content |
| --- | --- |
| `gui-settings.json` | GUI settings |
| `user/custom_configs/<title ID>.json` | Per-game settings |
| `user/custom_envs.json` | Global custom environment variables |
| `user/logs/` | Logs, traces, crash dumps and shader and frame dumps (`SHARPEMU_LOG_DIR` moves it) |
| `user/savedata/<title ID>/` | Save data (`SHARPEMU_SAVEDATA_DIR` moves it) |
| `user/patches/` | Game patches |
| `user/mods/<title ID>/` | Game mods, one folder each, and which are enabled |
| `user/redzone_cache/` | Red-zone patch plans, one per executable as loaded; safe to delete |
| `user/game-art/shortcut-splashes/` | Splash images shown while a shortcut starts its game |
| `user/zar-cache/` | Unpacked parts of `.zar` archives |
| `user/pipeline_cache/<title ID>/metal-v2/` | Metal shader and pipeline caches |
| `user/shader_seeds/<title ID>.seed` | Shader seed written by learn mode |

The Metal caches are keyed by the translator version, the macOS version and the
GPU, and are rebuilt when any of them changes. They are safe to delete. The first
run of a game compiles its shaders on the fly; later runs start faster. See
[Metal backend](metal-backend.md#shader-and-pipeline-caches) for what each cache
holds. macOS also keeps its own per-app Metal shader cache.

`--shader-cache=off` turns off the background compile of shaders the game has
not used yet. `--shader-learn=on` writes the shader seed when the session ends.
`--precompile-only` compiles every shader found for the title, holds the game
meanwhile, and exits when done; `--shader-prewarm-hold` holds the game until
that compile has finished and then lets it run.

## Requirements and limits

- Rosetta 2 on Apple Silicon (`softwareupdate --install-rosetta`); see
  [Building](building.md#why-the-host-process-is-x86-64).
- A bundle cannot update itself in place. The GUI reports new releases and opens
  the releases page.
- The macOS build has no Vulkan, MoltenVK or SDL. Settings and diagnostics that
  name Vulkan or RenderDoc, and Windows-only diagnostics such as crash dumps,
  do not apply.
- Games that draw with primitive (NGG) shaders need a Metal 3 device, because the
  backend runs them as Metal mesh pipelines. Without one it logs that mesh render
  pipelines are unavailable, and the shared draw executor drops those draws.
- A title can use a large amount of memory. The guest address space is backed by
  purgeable host memory objects, and the image cache shrinks under memory
  pressure, but a Mac with little free memory can still page heavily.
