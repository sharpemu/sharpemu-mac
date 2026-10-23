<!-- Copyright (C) 2026 SharpEmu Emulator Project -->
<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

# Concurrent compute follow-up 2

Base: `ds-fps-concurrent-compute` at `997a8e1b0` (implementation `21d519f53`).
Worktree: `/Users/xnetcat/.t3/worktrees/sharpemu-mac/ds-fps-concurrent-compute2`.
No game was run for this change. The measurements below are analysis of the supplied runs.

## Cause

Recounting the last 3,000 encoder PASS-PROF records in each supplied game.log gives:

| Compute end reason | baseline2 | integ1-prof |
| --- | ---: | ---: |
| other | 727 | 2,138 |
| render | 401 | 175 |
| flush | 316 | 118 |
| transfer | 415 | 65 |

**All 2,138 integration compute records ending in `other` have `after=[label-copies ...]`; none have staged-writeback.**
The extra path is `PrepareBindings` -> `OrderComputeExternalWrites` ->
`OrderAfterDeferredCopies(0, ulong.MaxValue)` -> `EndEncoders(PassEndCause.Other)`.
Direct and indirect dispatch also called this blanket drain. It forced even unrelated held
CP writes to land, defeating the existing range-specific and labels-at-commit deferral.
The second blanket call, `OrderAfterStagedStores(0, ulong.MaxValue)`, could also end encoders
and needlessly publish staged allocations that following dispatches could have shared.
Neither RAW/WAR/WAW conflicts nor unknown access classes end an encoder in the dependency
tracker: both already call `ComputeScopeBarrier` (Metal `memoryBarrierWithScope:`).
There is no per-dispatch commit in this tracker.

The global drain in PrepareBindings ran only for compute, not render. The original range
capture also ran only in OrderComputeDispatch, not on graphics draws. Thus the reported
55.7 -> 115.0 ms/frame DrawResourcePreparation increase cannot be assigned to per-render-draw
range capture. `RenderExecutor.Resources.cs:382` opens this phase for graphics resources;
compute preparation is outside that phase. The compute binding preparation did separately
include the expensive blanket copy drain.
The integration report also records substantial machine-load/compilation noise; no new timing
claim is made from this patch alone.

## Changes

- Remove the three concurrent-only blanket drains. Both dispatch types retain the serial
  path's range-specific CP/staged-store ordering and the persistent-kernel label rule.
  Encoder/submission boundaries retain the existing GPU event ordering. Resource hazards
  and unknown footprints remain full buffer/texture barriers inside the current encoder.
- Resolve indirect arguments coherently in both modes through the same storage path, so
  disabling concurrency does not change materialization or transfer boundaries.
- Capture compute dependencies during BindResources using the descriptor, clamped length,
  and native buffer range already resolved there. Capture no graphics footprint. Reuse the
  binding footprint for dispatch and temporarily append auxiliary/indirect accesses.
- Retain footprint storage and preallocate a bounded 4,096-access epoch. Iterate spans
  without LINQ enumerators. Epoch capacity exhaustion inserts a barrier; oversized individual
  footprints conservatively mark the next epoch unknown. Neither case creates an encoder.
- Add cumulative `compute_dispatches`, `compute_barriers`, `compute_dependency_barriers`,
  and `compute_encoders` fields to the existing TBDR report. Total scope barriers include
  guest/shared-transfer barriers; dependency barriers are the subset selected by the tracker.
- Keep concurrency default-on and `SHARPEMU_METAL_V2_CONCURRENT_COMPUTE=0` as the disable switch.

## Source pointers

Paths in this paragraph are under `src/graphics/SharpEmu.Gpu.Metal/V2/`.
The removed blanket helper and its calls are at base commit `21d519f53` in
`MetalRenderHost.ComputeDependencies.cs:34`, `MetalRenderHost.Descriptors.cs:282`,
and `MetalRenderHost.Rendering.cs:1838,1883`.
Existing transfer boundaries are `MetalCommandStream.cs:1346` and
`MetalRenderHost.StagedStorage.cs:227`. The fixed compute-only capture gate is
`MetalRenderHost.Descriptors.cs:568`; capture/dispatch/barrier code is at
`MetalRenderHost.ComputeDependencies.cs:43,93,173`. Bounded epochs are in
`MetalComputeDependencies.cs:56`; counters are in `MetalRenderHost.cs:2415`.
Device parity is asserted by `tests/SharpEmu.Gpu.Metal.Tests/Scenes/ComputeSceneTests.Dependencies.cs:146`;
the allocation regression is at `tests/SharpEmu.Gpu.Metal.Tests/MetalComputeDependenciesTests.cs:171`.

## Validation

The negative device test temporarily reinstated **only** the blanket helper and its three
calls atop the new instrumentation/capture. Both cases failed on encoder counts:

| 64-dispatch stream | Serial compute / all encoders | Old concurrent compute / all encoders |
| --- | ---: | ---: |
| unrelated CP writes | 1 / 2 | 64 / 128 |
| some CP writes touch inputs | 8 / 17 | 64 / 128 |

The fixed focused run passes all 23 tests: compute counts are **1/1** and **8/8**,
all-encoder counts **2/2** and **17/17**, and output bytes match. The allocation test
measures **zero bytes for 10,000 warmed footprint/tracker iterations**. This does not
claim zero allocations for the entire pre-existing resource-binding pipeline.
Release solution build succeeds: **173 warnings, zero errors**, 2m34.55s.

Full GPU Metal suite: **1,434 passed, 7 existing explicit skips, zero failures** (1m09s).
Shared shader compiler: **2,134 passed, 1 existing skip, zero failures** (15.13s).
An earlier shared-suite invocation stalled after 607 passes and was terminated after four
minutes without progress. Its native sample shows several JIT/SRT threads blocked in
`tiny_malloc_should_clear` / `_os_unfair_lock_lock_slow`; managed diagnostic attachment
could not complete. A clean full retry with a 45-second hang timeout passed without a
source change. The interrupted log/TRX and native sample are retained beside the final logs.
Metal shader compiler: **3,162 passed, zero skips, zero failures** (26.63s).
All three required suites therefore finish with **6,730 passed, 8 skipped, zero failures**.
The seven GPU skips cover existing unsupported formats/device semantics and dynamic HZB
resource planning; the shared compiler skip needs the optional captured mesh fixture.
Additional device checks with `SHARPEMU_METAL_V2_PROFILE_PASSES=1` pass **4/4**, including
sampled concurrent/serial encoder counts and explicit guest barriers. With the disable
switch set to `0`, the same checks also pass **4/4**. Both runs retain matching byte
readback and compute/all-encoder counts. No game or gameplay/FPS claim is made.

Commands used `/Users/xnetcat/.dotnet-x64/dotnet` (SDK 10.0.103, x64/Rosetta):

```sh
dotnet build SharpEmu.slnx -c Release -m:2 -nr:false -p:UseSharedCompilation=false
# Each suite separately, sequentially; check /tmp/ds-fps-gamerun.lock before each command.
SHARPEMU_TEST_REQUIRE_DEVICE=1 SHARPEMU_LOG_MEMORY=1 dotnet test tests/<suite> \
  -c Release --no-build --no-restore -m:1 -nr:false
```

The first GPU suite used `-m:2`; the completed shared retry and Metal compiler suite used
`-m:1` with 45s/90s hang timeouts. No compiler/runtime feature was disabled for those runs.
Exact stdout and TRX files, the negative/focused/profiled/disabled logs, and
`tested-source.sha256` are retained under `artifacts/ds-fps/bench/concurrent-compute2/`.
The SHA-256 manifest matches every changed source/test file after final validation. The mandatory gamerun lock is checked before every build/test invocation; suites
run sequentially with `SHARPEMU_TEST_REQUIRE_DEVICE=1` for both Metal projects.
