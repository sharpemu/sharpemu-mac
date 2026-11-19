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
| `SHARPEMU_TITLE_SCENE_RESOLUTION` | unset | Demon's Souls only (known build). Caps the game's internal scene size below the display: `900p`, `720p`, `540p` (any `<height>p`, 16:9) or `WIDTHxHEIGHT`, at least 640x360. The game renders the scene at that size and upscales it itself; the output, display buffers and UI keep their sizes. Lowers image quality for frame time: on a base M4 at a 1080p display, about 78 ms per frame uncapped, 69 ms at `900p`, 63 ms at `720p`, 55 ms at `540p`. Below `540p` the picture breaks and nothing more is gained. |
| `SHARPEMU_STALL_WATCHDOG_SECONDS` | `20` | Seconds without import progress before the stall watchdog acts. `0` disables it. |
| `SHARPEMU_FLIP_AHEAD` | off | Experimental. `1` tells the game a GPU flip is done once the flip is recorded and the flip before it has really finished on the GPU, instead of when the frame is shown, so the game can build its next frame while the GPU draws this one (at most one flip ahead). Each such frame is copied in the GPU stream into a presenter-owned texture (up to 7 textures the size of the display buffer), so what is shown is unchanged; a flip whose frame cannot be copied keeps the usual timing, and the refusal is logged once. Pacing, interrupts and labels are unchanged. Counters: `[VIDEOOUT][FLIP-AHEAD]` and the `flip_copy*` fields of `[METAL-PRESENT]`. |
| `SHARPEMU_METAL_V2_PRESENT_SNAPSHOT` | on | `0` presents a flipped display buffer directly instead of a copy of it. By default each flip's frame is copied in the GPU stream, at the flip, into a presenter-owned texture (one blit per flip; up to 7 textures the size of the display buffer, 4 in use on a 60 fps title), so a game that starts drawing its next frame into the buffer still on screen cannot show a half-drawn (black) frame. A frame that cannot be copied is presented directly and the refusal is logged once. |

## Metal memory

| Variable | Default | Effect |
| --- | --- | --- |
| `SHARPEMU_METAL_V2_GRANULE_MB` | `32` | Size of the windows guest memory is cut into. A power of two from 1 to 1024. |
| `SHARPEMU_METAL_V2_WINDOW_MB` | `1024` | Largest a window grows to hold one binding. 512 to 4096. |
| `SHARPEMU_METAL_V2_IMAGE_BUDGET_MB` | computed | Fixed byte budget for the detiled image cache, at least 256. By default the budget follows the memory the host has free. |
| `SHARPEMU_METAL_V2_SINGLE_COPY` | on | `0` never releases guest memory that a clean texture's Metal copy duplicates. Requires the chunked guest backing. |
| `SHARPEMU_METAL_V2_WHOLE_STALE_RELOAD` | on | `0` restores the old refusal: a GPU-written image whose every byte an alias has since written keeps its older texels instead of reloading them, and writes them back over the alias's bytes. For A/B comparisons only. |
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
| `SHARPEMU_METAL_V2_GRAPHICS_WAIT_OFF_CACHE` | on | A draw that cannot be left out while its pipeline compiles waits for the compile outside the shared pipeline cache's lock, so other draws' programs keep resolving. `0` builds the pipeline under the lock, as before. Only with `SHARPEMU_METAL_V2_ASYNC_GRAPHICS` on. |
| `SHARPEMU_RESOLVE_SHARDS` | `1` | Threads that resolve draw and dispatch programs ahead of the GPU worker, 1 to 8. With more than one, each has its own queue and its own part of the resolve memo, and a draw always goes to the same one. `1` is the single resolver; `2` measured no frame-time gain in Demon's Souls for more resolver CPU. Logged once at start; per-thread counts and the shared-lock wait are in the `shards=(...)` field of the `[METAL-V2][PERF]` prefetch counters. |
| `SHARPEMU_PREFETCH_EXACT_EFFECTS` | on | A dispatch or draw resolved ahead of a dispatch the host consumed as a kernel fill or copy is no longer refused for that alone: its take refuses it only when the fill's or copy's logged write covers what its resolution read (the same check as across any other dispatch). DCC and constant fills, image clears, metadata clears, image copies and CPU-predicted dispatches still refuse every dispatch and draw resolved ahead of them. The prefetch counters report `dispatch_effects=(...)` by kind and `exact_effects=(mode,refused,taken,draws,draw_refused,draw_taken)`, the jobs the old rule would have refused that the check refused or took. `dispatch` keeps the old rule for draws only; `0` refuses on every effect again. |
| `SHARPEMU_RESOLVE_UNSHARED_COMPUTE` | on | With more than one resolve shard, a dispatch whose kept program still holds, or whose kept program only needs new user data on a fast plan with a permutation already made, is answered on its own thread without the shared lock. `0` sends every dispatch through the shared lock. The `shards=(...)` counters split the shared-lock takes and wait by reason (`shared_by=`, `shared_wait_by=`) and count the dispatches answered without it (`unshared_compute=`) and why a rebinding still took it (`rebind_shared_by=`). |
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
| `SHARPEMU_MSL_HALF_BRANCHES` | on | Tests the whole wave64 (an exchange between its two simdgroups) at every forward `S_CBRANCH_EXECZ` of a one-wave compute group, instead of letting each half skip a body that only touches its own lanes. |
| `SHARPEMU_MSL_SPILLCOPY` | on | Reads spilled scalar registers through the lane exchange. |
| `SHARPEMU_MSL_PERSISTENT_FENCES` | on | Leaves `S_WAITCNT` a no-op in persistent-thread kernels. |
| `SHARPEMU_MSL_PERSISTENT_RECORDS` | on | Accesses multi-dword records of persistent-thread kernels dword by dword. |
| `SHARPEMU_MSL_BYTEBIAS` | off | `1` keeps the per-buffer byte bias. |
| `SHARPEMU_MSL_FUSE_MAD` | on | Emits the guest's legacy multiply-adds (`V_MAD_F32`, `V_MAC_F32`) as one fused operation in compute programs and pixel programs without depth export. Faster (about 5 ms of an 82 ms Demon's Souls frame) but not bit-exact for MAD/MAC: the guest rounds the product first. `0` restores the exact two-op form. |
| `SHARPEMU_METAL_V2_BUFFER_WORD_CONSTANTS` | on | With portable buffers (`SHARPEMU_METAL_V2_PORTABLE_BUFFERS`, on), each buffer stride and format word a program reads at run time is also a Metal function constant. A draw draws through its pipeline as before and, once a background compile has made it, through a variant whose functions have the draw's words folded in (same pixels, faster vertex stage). Up to 16 variants a pipeline; mesh, rectangle-list and tessellation pipelines stay generic. A variant built once is stored in the title's binary archive (each specialized stage's library file, the pipeline in its segment), so later starts load it instead of compiling. The periodic counter line reports `buffer_words=[...]`, with where built variants came from (`from_file`, `from_library`: this process's own library, `from_msl`: the module's MSL compiled again, `msl_compiles`), `pipelines_compiled`, `files_rejected`, `unresident` and the builders' `build_ms` / `wait_ms`. `0` restores the previous MSL exactly. |
| `SHARPEMU_METAL_V2_BUFFER_WORD_COMPUTE` | on | The same variants for compute pipelines: a dispatch runs the generic pipeline until a background compile has made the one with its words folded in (same results). Off whenever `SHARPEMU_METAL_V2_BUFFER_WORD_CONSTANTS=0`. `0` keeps every compute dispatch on its generic pipeline while draws keep their variants, so the two can be measured apart; it does not change the MSL. The periodic counter line reports `buffer_words_compute=[...]` (`generic_off` counts dispatches the switch kept generic). |
| `SHARPEMU_LOD_STATS` | guarded | `always` compiles the LOD statistics counters into every pixel pipeline. |

## GPU command stream

| Variable | Default | Effect |
| --- | --- | --- |
| `SHARPEMU_METAL_V2_DEFERRED_LABEL_LANDINGS` | on | An end-of-pipe label that Metal defers to GPU completion still reports the submission it lands with, so another queue's soft wait (`SHARPEMU_END_OF_PIPE_SOFT_WAITS`) ends when that submission completes. `0` gives deferred labels no landing again: such a wait ends only at its cap. |
| `SHARPEMU_METAL_V2_DYNAMIC_STENCIL_OPS` | on | A draw's stencil operations and compare functions come from its own dynamic state, not from the draw its pipeline was first built for. `0` takes them from the pipeline again. |
| `SHARPEMU_POSTED_WORD_COPY` | on | A bounded word-copy kernel whose source is final and whose destination holds no image or surface metadata is recorded as a command-processor store of the copied words at its place in the stream, not as a dispatch. The counter line reports `word_copy=[posted=...,unsupported=...,image=...,...]`. `0` runs every such kernel as a dispatch. |

## Diagnostics

| Variable | Effect |
| --- | --- |
| `SHARPEMU_METAL_V2_TRACE=1` | Verbose Metal host reporting, including the render-pass breakdown that follows the periodic counter line. |
| `SHARPEMU_METAL_V2_REPORT_UNBACKED=1` | Names ranges that had to be staged through the upload ring. Implied by the trace switch. |
| `SHARPEMU_METAL_V2_WAIT_TRACE=1` (or `SHARPEMU_METAL_V2_EOP_TRACE=1`) | Names, on each `[METAL-V2][SLOW-WAIT]` line, the shader whose resources were being prepared. |
| `SHARPEMU_METAL_V2_PROFILE_SHADERS=1` | Gives every draw and dispatch its own command buffer and prints GPU time per shader hash (`[METAL-V2][SHADER-PROF]`). It serializes the GPU, so only ratios are meaningful. |
| `SHARPEMU_METAL_V2_PROFILE_TOP=<n>` | Entries per profile report. Default 8. |
| `SHARPEMU_METAL_V2_PROFILE_PASSES=1` | Per-encoder GPU time (`[METAL-V2][PASS-PROF]`). Blit encoders (`k=blit`), the tiler's compute encoders (`k=tiler-compute`) and the host's other compute encoders (`k=host-compute`) are recorded too, named for the transfer that opened them, so `gap_us` is time no encoder ran; the `sub` line sums GPU time per kind (`kind_us=`). |
| `SHARPEMU_DIAG_SKIP_ALL_DRAWS_AFTER_S=<seconds>` | Diagnostic. From that many seconds after the GPU process translated its first draw, every draw is dropped before it is resolved or recorded; dispatches, labels and flips go on. Shows the frame time a draw path of no cost would leave. The picture is wrong by design; logged once when it starts. |
| `SHARPEMU_PROFILE_RENDER=1` | Per-phase timing of the draw translation path on the GPU worker, reported at each flip. |
| `SHARPEMU_METAL_V2_DUMP_TARGETS=<dir>` | Writes each render pass's color target and each presented frame as PNG. Waits for the GPU after each dump. `1` writes to `metal-targets` under the logs folder. |
| `SHARPEMU_METAL_V2_DUMP_MAX=<n>` | Maximum dumps. Default 240. |
| `SHARPEMU_METAL_V2_DUMP_EVERY=<n>` | Dump one frame's targets every `n` flips. |
| `SHARPEMU_METAL_V2_DUMP_AFTER_SECONDS=<n>`, `SHARPEMU_METAL_V2_DUMP_AFTER_FLIPS=<n>` | Hold dumps back until the game has run that long, or flipped that many times. |
| `SHARPEMU_MSL_HALF_BRANCH_REPORT=1` | Prints, for each forward `S_CBRANCH_EXECZ` of a wave64 compute program, whether each half may take it alone and the first reason it may not (`[MSL][HALF-BRANCH]`). Works with the shader corpus replay. |
| `SHARPEMU_SHADER_MSL_DUMP_DIR=<dir>` | Writes each translated program as `<stage>-<hash>.metal` with a `<stage>-<hash>.gen5` listing of the guest instructions. `1` writes to `msl-dumps` under the logs folder. |
| `SHARPEMU_SHADER_MSL_DUMP_HASH=<hashes>` | Restricts the dump to a comma-separated list of shader hashes. |
| `SHARPEMU_METAL_V2_IGNORE_TESTS=1` | Drops culling, depth and stencil tests from every draw, to tell an empty pass from a rejected one. |
| `SHARPEMU_METAL_V2_CONSERVATIVE_READS=1` | Host reads of guest memory wait for all recorded GPU stores instead of only overlapping ones. |
| `SHARPEMU_METAL_V2_WAIT_ON_COMMAND_BUFFER=1` | Waits for a command buffer with `waitUntilCompleted` instead of the shared event. |
| `SHARPEMU_METAL_V2_POLL_COMPLETIONS=1` | Polls the shared event instead of using a completion handler. |
| `SHARPEMU_FRAME_TIMES=<path>` | Writes one timestamp line per guest flip. `1` writes `frame-times-<timestamp>.txt` under the logs folder. |
| `SHARPEMU_INPUT_FILE=<path>` | Test automation: the file lists the keys held right now; it is polled every 30 ms and works without window focus. |

### Finding what overwrote guest memory

These cost nothing until the game faults. When it does, the crash report names
the writes that covered the bytes you asked about.

| Variable | Effect |
| --- | --- |
| `SHARPEMU_FAULT_DUMP=<label>=<expr>:<len>,...` | At a guest fault, dumps `len` bytes at each expression. An expression is a register or number with `+`/`-` offsets, and `[x]` reads the pointer at `x` (`e0=[[rbp]-0x360]:0x1000`). A label starting `find` also searches all mapped guest memory for those bytes; one starting `save` writes them (up to 64 MiB) to a file instead. |
| `SHARPEMU_FAULT_SAVE_DIR=<dir>` | Where `save` labels write `fault-<label>-<address>.bin`. |
| `SHARPEMU_FAULT_ALIASES=1` | For each dumped range, prints its backing offset and every other guest view of the same backing. |
| `SHARPEMU_FAULT_FIND_ALL=1` | `find` labels list every hit, not only the first few. |
| `SHARPEMU_FAULT_DUMP_REPEAT_MS=<ms>` | Repeats the dump after that long, to show bytes still changing. |
| `SHARPEMU_GPU_WRITE_RING=1` | The GPU process keeps a ring of its writes to guest memory (image write-backs, fills, copies, shader stores of 16 bytes or more) with time, submission and producer, and prints the ones over each dumped range, by address and by backing. `SHARPEMU_GPU_WRITE_RING_ENTRIES=<n>` sets its size. It also records what a managed write found and waited for (`gpu:store-enter`, `gpu:store-return`, `gpu:store-landed`, each with 16 bytes sampled at +0x20000 of the write). |
| `SHARPEMU_HLE_WRITE_RING=1` | The same ring in the game process for file reads and `memcpy` made by the emulator's libraries, with the file and offset or the copy's source; a copy into a dumped range adds its source to the search. AMPR reads also record the sampled bytes as the fill starts and ends (`hle:ampr-fill-start`, `hle:ampr-fill-end`). `SHARPEMU_HLE_WRITE_RING_MIN=<bytes>` drops smaller writes: recording every small copy changes thread timing enough to hide a race. |
| `SHARPEMU_FAULT_RING_PRODUCER=<text>` | Adds the ranges of recorded writes whose producer contains the text (a file name, for example) to the search. |
| `SHARPEMU_WRITE_RING_TAIL=<n>` | Also prints the last `n` records of each ring. |

The Metal source also reads many narrower switches that exist to bisect a
single optimization (for example `SHARPEMU_METAL_V2_FLUSH_COALESCE`,
`SHARPEMU_METAL_V2_FOLD_CLEARS` and `SHARPEMU_METAL_V2_KERNEL_FILLS`, all on by
default). They are listed in `MetalV2Diagnostics.cs` and next to the code that
reads them, and are not supported settings.
