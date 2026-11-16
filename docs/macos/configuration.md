<!--
Copyright (C) 2026 SharpEmu Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

# macOS configuration

This page lists the environment variables that matter on macOS and for the Metal
backend. [The general variable reference](../sharpemu-gui-undocumented-env-vars.md)
covers the rest; its Vulkan and RenderDoc entries do not apply on macOS.

## Setting a variable

- In the shell before you start SharpEmu: `SHARPEMU_METAL_V2_TRACE=1 SharpEmu ...`.
- In the GUI, globally or per game, in the custom environment list. An entry is
  `NAME` (value `1`) or `NAME=value`. The launcher passes only names that start
  with `SHARPEMU_` and have a non-empty value.

The emulator reads almost every variable once, at start-up. Restart the game
after changing one.

Switches that are on by default are turned off with `=0`, and switches that are
off by default are turned on with `=1`. The code compares the exact string, so
other values (`true`, `on`) do nothing unless the table says otherwise.

Unless noted, a variable below was checked against the source in
`src/graphics/SharpEmu.Gpu.Metal/V2/`. The Metal variables are tuning and
diagnostic switches. The defaults are what the project ships and tests; change
them for measurements or to isolate a problem.

## Data location and host

| Variable | Default | Effect |
| --- | --- | --- |
| `SHARPEMU_DATA_ROOT` | see [Running games](running-games.md#caches-and-data) | Absolute path of the data root. Wins over everything else. |
| `SHARPEMU_USER_DIR` | unset | Alternative data root, used when `SHARPEMU_DATA_ROOT` is unset. |
| `SHARPEMU_SAVEDATA_DIR` | `<data root>/user/savedata` | Save data directory. |
| `SHARPEMU_LOG_DIR` | `<data root>/user/logs` | Folder for every log, trace, crash dump and dump below. The dump variables that name a folder accept `1` for their default subfolder there. |
| `SHARPEMU_OVERLAY` | unset | `0` starts the [performance display](../performance-overlay.md) hidden. |
| `SHARPEMU_GUEST_BACKING` | chunked | `shm` backs guest memory with one POSIX shared-memory object instead of 2 MiB purgeable memory objects. The default allows memory the guest frees to return to the OS. |
| `SHARPEMU_GUEST_BACKING_SHAREABLE` | unset | `1` makes the guest backing shareable with another process, and keeps a shared table and log of guest views. This is groundwork for the two-process split (`SHARPEMU_GPU_PROCESS=1` sets it too). See [Guest memory for the two-process split](split-shared-memory.md). |
| `SHARPEMU_HLE_FAST_PATH` | on | `0` disables the native stubs for `pthread_self` and `pthread_getspecific`. They are also off while import or pthread tracing is active. |
| `SHARPEMU_STALL_WATCHDOG_SECONDS` | `20` | Seconds without import progress before the stall watchdog acts. `0` disables it. |

## Metal memory

| Variable | Default | Effect |
| --- | --- | --- |
| `SHARPEMU_METAL_V2_GRANULE_MB` | `32` | Size of the windows guest memory is cut into. A power of two from 1 to 1024. |
| `SHARPEMU_METAL_V2_WINDOW_MB` | `1024` | Largest a window grows to hold one binding. 512 to 4096. |
| `SHARPEMU_METAL_V2_IMAGE_BUDGET_MB` | computed | Fixed byte budget for the detiled image cache, at least 256. By default the budget follows the memory the host has free. |
| `SHARPEMU_METAL_V2_SINGLE_COPY` | on | `0` never releases guest memory that a clean texture's Metal copy duplicates. Requires the chunked guest backing. |
| `SHARPEMU_METAL_V2_STAGED_CACHE_MB` | `256` | Budget of the staged-range cache. |
| `SHARPEMU_METAL_V2_TRANSIENT_COMMIT_MB` | `256` | Transient staging a command buffer may hold before it is committed early. |
| `SHARPEMU_METAL_V2_RESIDENCY_SET` | off | `1` puts every guest window in an `MTLResidencySet`. |
| `SHARPEMU_METAL_V2_GUARD_WRITES` | off | `1` checks that GPU writes to guest memory stay inside the range their producer declared. Costs two 4 KiB reads per write. |
| `SHARPEMU_LOG_MEMORY` | off | `1` logs periodic `[MEM]` lines. `SHARPEMU_LOG_MEMORY_SECONDS` sets the interval (default 10). |

## Shader translation and caches

| Variable | Default | Effect |
| --- | --- | --- |
| `SHARPEMU_METAL_MESH_SHADERS` | on | `0` disables mesh pipelines for primitive-shader draws. |
| `SHARPEMU_METAL_MESHLETS` | on | `0` disables the meshlet cache. |
| `SHARPEMU_METAL_V2_BINARY_ARCHIVE` | on | `0` disables the on-disk library and pipeline archives. |
| `SHARPEMU_METAL_V2_BINARY_ARCHIVE_BUDGET_MB` | `8192` | Size bound of the archives on disk. |
| `SHARPEMU_METAL_V2_ASYNC_PIPELINES` | on | `0` builds pipeline states on the GPU worker thread. |
| `SHARPEMU_METAL_V2_ASYNC_LIBRARIES` | on | `0` compiles each shader library on the thread that needs it, not on the compile pool. |
| `SHARPEMU_METAL_V2_PIPELINE_LIMIT` | `8192` | Pipeline states kept before idle ones are released. `0` is unlimited. |
| `SHARPEMU_METAL_V2_MODULE_LIMIT` | `2048` | The same for shader modules. |
| `SHARPEMU_METAL_V2_SHADER_EVICTION` | on | `0` keeps every pipeline and module. |
| `SHARPEMU_SHADER_CACHE` | on | `off`, `0`, `false`, `no` or `disabled` turns off the background compile of shaders the game has not used yet. CLI: `--shader-cache=on\|off`. |
| `SHARPEMU_SHADER_LEARN` | off | `on` writes `user/shader_seeds/<title ID>.seed` when the session ends. CLI: `--shader-learn=on\|off`. |
| `SHARPEMU_SHADER_PREWARM` | on | `0` disables the prewarm and the background compile. |
| `SHARPEMU_SHADER_PREWARM_WAIT` | on | `0` stops the prewarm from holding the guest at its first frame. |
| `SHARPEMU_SHADER_PREWARM_HOLD` | unset | `complete` holds the guest until every recorded shader is compiled. CLI: `--shader-prewarm-hold`. |
| `SHARPEMU_SHADER_PRECOMPILE_ONLY` | off | `1` holds the guest, compiles everything, then exits. CLI: `--precompile-only`. |
| `SHARPEMU_SHADER_PRECOMPILE_JOBS` | `2` | Threads of the background compile, 1 to 16. |
| `SHARPEMU_SHADER_PRECOMPILE_BUDGET` | `3000` | New libraries one session may store. |
| `SHARPEMU_SHADER_SCAN` | `shaders` | Which game files the background compile scans: `shaders`, `all` or `off`. |
| `SHARPEMU_SHADER_PRECOMPILE_COMPUTE` | off | `1` includes compute programs in the background compile. |
| `SHARPEMU_PREWARM_PROGRESS` | on | `0` hides the progress bar drawn over the splash during the prewarm. |
| `SHARPEMU_SHADER_MAX_STEPS` | `100000` | Step limit per translated shader invocation, so a wrong loop stops instead of hanging the GPU. `0` removes it. |
| `SHARPEMU_SHADER_MAX_STEPS_GRAPHICS` | off | `1` also applies the limit to loops in structured vertex and pixel programs. |

The translator's code generation switches are read once per process. Each one
undoes an optimization to isolate a miscompile:

| Variable | Default | `0` (or the shown value) does this |
| --- | --- | --- |
| `SHARPEMU_MSL_STRUCTURED` | on | Emits a program counter dispatcher instead of structured control flow. |
| `SHARPEMU_MSL_REGALLOC` | on | Declares both full guest register files. |
| `SHARPEMU_MSL_REGINIT` | off | `1` zero-fills every declared register. |
| `SHARPEMU_MSL_EXECGUARD` | on | Keeps every per-lane EXEC guard. |
| `SHARPEMU_MSL_MOVREL` | on | Indexes the register file for relative moves. |
| `SHARPEMU_MSL_DWORDACCESS` | on | Assembles raw buffer dwords from bytes. |
| `SHARPEMU_MSL_CONSTANTTABLES` | on | Reads uniform tables through the device address space. |
| `SHARPEMU_MSL_LDSSIZE` | on | Declares all 32 KB of threadgroup memory. |
| `SHARPEMU_MSL_MASKBALLOT` | on | Materializes every wave ballot. |
| `SHARPEMU_MSL_SPILLCOPY` | on | Reads spilled scalar registers through the lane exchange. |
| `SHARPEMU_MSL_PERSISTENT_FENCES` | on | Leaves `S_WAITCNT` a no-op in persistent-thread kernels. |
| `SHARPEMU_MSL_PERSISTENT_RECORDS` | on | Accesses multi-dword records of persistent-thread kernels dword by dword. |
| `SHARPEMU_MSL_BYTEBIAS` | off | `1` keeps the per-buffer byte bias. |
| `SHARPEMU_LOD_STATS` | guarded | `always` compiles the LOD statistics counters into every pixel pipeline. |

## Diagnostics

| Variable | Effect |
| --- | --- |
| `SHARPEMU_METAL_V2_TRACE=1` | Verbose Metal host reporting, including the render-pass breakdown that follows the periodic counter line. |
| `SHARPEMU_METAL_V2_REPORT_UNBACKED=1` | Names ranges that had to be staged through the upload ring. Implied by the trace switch. |
| `SHARPEMU_METAL_V2_WAIT_TRACE=1` (or `SHARPEMU_METAL_V2_EOP_TRACE=1`) | Names, on each `[METAL-V2][SLOW-WAIT]` line, the shader whose resources were being prepared. |
| `SHARPEMU_METAL_V2_PROFILE_SHADERS=1` | Gives every draw and dispatch its own command buffer and prints GPU time per shader hash (`[METAL-V2][SHADER-PROF]`). It serializes the GPU, so only ratios are meaningful. |
| `SHARPEMU_METAL_V2_PROFILE_TOP=<n>` | Entries per profile report. Default 8. |
| `SHARPEMU_METAL_V2_PROFILE_PASSES=1` | Per-encoder GPU time (`[METAL-V2][PASS-PROF]`). |
| `SHARPEMU_PROFILE_RENDER=1` | Per-phase timing of the draw translation path on the GPU worker, reported at each flip. |
| `SHARPEMU_METAL_V2_DUMP_TARGETS=<dir>` | Writes each render pass's color target and each presented frame as PNG. Waits for the GPU after each dump. `1` writes to `metal-targets` under the logs folder. |
| `SHARPEMU_METAL_V2_DUMP_MAX=<n>` | Maximum dumps. Default 240. |
| `SHARPEMU_METAL_V2_DUMP_EVERY=<n>` | Dump one frame's targets every `n` flips. |
| `SHARPEMU_METAL_V2_DUMP_AFTER_SECONDS=<n>`, `SHARPEMU_METAL_V2_DUMP_AFTER_FLIPS=<n>` | Hold dumps back until the game has run that long, or flipped that many times. |
| `SHARPEMU_SHADER_MSL_DUMP_DIR=<dir>` | Writes each translated program as `<stage>-<hash>.metal` with a `<stage>-<hash>.gen5` listing of the guest instructions. `1` writes to `msl-dumps` under the logs folder. |
| `SHARPEMU_SHADER_MSL_DUMP_HASH=<hashes>` | Restricts the dump to a comma-separated list of shader hashes. |
| `SHARPEMU_METAL_V2_IGNORE_TESTS=1` | Drops culling, depth and stencil tests from every draw, to tell an empty pass from a rejected one. |
| `SHARPEMU_METAL_V2_CONSERVATIVE_READS=1` | Host reads of guest memory wait for all recorded GPU stores instead of only overlapping ones. |
| `SHARPEMU_METAL_V2_WAIT_ON_COMMAND_BUFFER=1` | Waits for a command buffer with `waitUntilCompleted` instead of the shared event. |
| `SHARPEMU_METAL_V2_POLL_COMPLETIONS=1` | Polls the shared event instead of using a completion handler. |
| `SHARPEMU_FRAME_TIMES=<path>` | Writes one timestamp line per guest flip. `1` writes `frame-times-<timestamp>.txt` under the logs folder. |
| `SHARPEMU_INPUT_FILE=<path>` | Test automation: the file lists the keys held right now; it is polled every 30 ms and works without window focus. |

The Metal source also reads many narrower switches that exist to bisect a
single optimization (for example `SHARPEMU_METAL_V2_FLUSH_COALESCE`,
`SHARPEMU_METAL_V2_FOLD_CLEARS` and `SHARPEMU_METAL_V2_KERNEL_FILLS`, all on by
default). They are listed in `MetalV2Diagnostics.cs` and next to the code that
reads them, and are not supported settings.
