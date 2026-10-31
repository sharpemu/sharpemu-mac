<!--
Copyright (C) 2026 SharpEmu Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# Performance display

Open **Options → Rendering**, globally or for one game.

- **Performance display:** start with statistics on or off.
- **Display mode:** Full shows all counters and the frame graph. Minimal shows
  guest FPS, process CPU usage, process GPU usage, and elapsed session time.
  Title bar shows the same minimal values in the window title.
- **Screen corner:** choose any of the four corners. This setting applies only
  to the on-screen panel.

F1 toggles the display for the current run. F2 cycles clockwise through the
screen corners. Neither key changes saved settings. The next run uses the saved
options again. On macOS, Command+F1 acts like F1; the Metal backend has no
separate HUD shortcut.

`SHARPEMU_OVERLAY=0` starts the display hidden, even when the saved option is on.
F1 can still show it. RenderDoc's F12 hint remains in the title when available.
The operating system does not show a title bar in borderless/fullscreen modes.

The launcher passes these options to the game process, including when a separate
emulator path is configured:

```text
--overlay=on|off
--overlay-mode=full|minimal|titlebar
--overlay-corner=topleft|topright|bottomright|bottomleft
```

## Usage values

The panel dimensions are separate constants in `PerfOverlay.cs`:
`PanelWidth` / `PanelHeight` for Full, and
`MinimalPanelWidth` / `MinimalPanelHeight` for Minimal. Both heights are fixed;
the renderer clips the panel only when the window is too small.

CPU usage is the emulator process's CPU time divided by wall time and the
number of logical processors. GPU usage is the busiest GPU engine used by the
emulator process, not GPU memory usage or total usage by all applications.

Windows supplies the GPU Engine performance counters. Collection runs off the
render thread and uses the existing statistics refresh interval. Rate counters
need two samples. Missing counters, unavailable samples, and other platforms
(except macOS, see [macOS](#macos)) show **N/A**, not zero. No GPU waits or presentation delays are added.

The full panel keeps cache memory counters and puts CPU/GPU usage on a separate
line above elapsed time and the Vulkan allocation counters.

The counter interpretation follows [Microsoft's GPU utilization description](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/).
Wildcard samples use [PdhGetFormattedCounterArray](https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw).

## macOS

The Metal backend does not draw the on-screen panels, so the display works in
**Title bar** mode only. Full and Minimal show nothing. A borderless or
full-screen window has no title bar and shows nothing either. The display is
on by default, in Title bar mode, with the top-right corner saved. F1 and
`SHARPEMU_OVERLAY=0` work as described above. F2 changes the saved corner only
for the on-screen panel, so it has no visible effect.

GPU usage on macOS is the utilization of the whole GPU, read from the
`Device Utilization %` statistic of the `IOAccelerator` service. macOS has no
per-process GPU engine counters, so the value includes other applications. It
shows **N/A** when the statistic cannot be read.

The Metal presenter draws Full and Minimal panels in the final present pass, at the
window's logical scale and selected corner. It refreshes the panel every 100 ms
even when the guest image is unchanged. FPS averages the last four statistics
windows so slow guest frame rates do not alternate between a positive value and
zero every second; four empty windows still report a stall. Title bar mode keeps
using the window title.
