<!-- Copyright (C) 2026 SharpEmu Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later -->

# b3-affine: cheap affine vertex-fetch preparation

Status: implementation committed, Release build and all four suites completed, all 13 added cases passed, and build outputs removed. Validation is **not all green**: ten documented/reproduced baseline failures plus one full-suite clock failure that passed the unchanged focused rerun. No game performance or rendering verification was performed in this task. No game launch, `.app` build, push, or review agent.

Branch: `ds-fps-b3-affine`, worktree `/Users/xnetcat/.t3/worktrees/sharpemu-mac/ds-fps-b3-affine`, base `46411e2df57257fd80f9c4e1c910293c809a97f4` (`ds-fps-v3-int2`). Implementation commit: **`e6c5f9700ebeb4af453cb5939b7cadd393c53a42`** (`[GPU] Cache affine vertex-fetch decisions before program lookup`). The 15 committed source/test files match the tested snapshot byte-for-byte; the build happened before this source commit. Binary hashes were saved before cleanup in [tested-binaries.json](../bench/b3-affine/tested-binaries.json).

## Change

`src/SharpEmu.Libs/Gpu/Pipelines/ShaderProgramCache.Affine.cs:22` retains the portable vertex analysis separately from the final static layout. The last-source fast path avoids a dictionary lookup for consecutive draws of the same source. `AffineFetches.Count == 0` is checked before resource eligibility or descriptor work. Resource write flags, workgroup eligibility, PC-to-resource lookup and lowering strategy are computed once at `ShaderProgramCache.cs:664` / `ShaderProgramCache.Affine.cs:112`.

`src/SharpEmu.Libs/Gpu/Pipelines/ShaderProgramCache.cs:405` selects the affine `VertexInputInfo` before building/probing the static key. It materializes only the selected plan and performs one permutation lookup. The recursive lookup/materialization is gone. Portable entries share immutable analysis across static variants, but each final key has its own permutation list; IEEE mode and other static compile options retain independent shader identities. Linked-call programs retain their previous portable behavior.

`ShaderProgramCache.Affine.cs:163` holds one bounded last decision per portable entry, including successful and rejected resolutions. Its key compares complete fetch bounds, capacity, all four freshly read descriptor words per candidate, effective masked scalar biases and their validity. It never treats a user-data array or a descriptor address as proof that its words stayed unchanged. Misses use bounded stack spans; result arrays are reused if unchanged. Hits reuse immutable attribute/buffer/PC arrays without lists, array copies, format decoding, PC lookup or mapping-range walks. Inline descriptors skip guest read scopes entirely; symbolic bias uses a value index instead of allocating a copied load record (`AffineVertexFetchResolver.cs:26`). A rejection leaves the caller's existing buffer metadata intact.

`src/SharpEmu.ShaderCompiler/Resources/RuntimeValueEvaluator.cs:706` reads only the candidate descriptors before selection. Inline constants/user-data descriptors allocate nothing. Guest-table descriptors retain evaluator storage on the entry, invalidate every computed value around each read, and preserve clean table-read rules, branch activity and host guest-word-read scopes; unrelated resources are left to the single final materialization. Unreadable descriptor results are not cached and retain portable failure handling.

`src/SharpEmu.Core/Memory/PhysicalVirtualMemory.cs:3083` exposes the existing mapping counter through `ICpuMemory`, `TrackedCpuMemory` and the Metal pipeline host. It declines a generation observation during concurrent map/unmap publication with a nonblocking read-lock attempt. Existing region insertion/removal paths now update that counter while publishing bookkeeping. The optional contract is at `src/SharpEmu.HLE/ICpuMemory.cs:57`, forwarded by `src/SharpEmu.Core/Cpu/TrackedCpuMemory.cs:163`, `src/SharpEmu.Libs/Gpu/Pipelines/IShaderPipelineHost.cs:58`, and `src/graphics/SharpEmu.Gpu.Metal/V2/MetalRenderHost.cs:2108`. Positive and negative mapping proofs are reused only while the generation agrees before/after the proof and on the next hit. Providers returning `null` keep per-resolution mapping checks. Decisions that never queried mapping do not read its generation on a hit.

## Switches and log counters

- Existing `SHARPEMU_METAL_V2_VERTEX_FETCH=0` disables affine native fetch selection. Default remains on for Metal.
- New `SHARPEMU_AFFINE_FETCH_CACHE=0` disables resolved-decision/mapping memoization; the nonrecursive single-materialization path remains. Default on; environment read once.
- `[METAL-V2][PERF]` includes `affine_cache_hits`, `affine_cache_misses`, `affine_cache_rejects` through the existing `VertexFetchCounters.TakeReport()` call. Counters reset per report. Hits/misses count eligible descriptor-resolved decisions; rejects count decisions yielding no replacement and static eligibility/unreadable refusals. No-affine programs remain visible as `vertex_fetch_refused=[...NoAffineIndex:...]` without reading descriptors.
- Existing `vertex_fetch_loads` counts selected loads on both misses and hits. Detailed `vertex_fetch_refused` reasons are generated on recomputation, rather than replaying every old refusal on a cache hit. Backend `vertex_fetch_fixed_draws` / `vertex_fetch_shader_draws` still count actual issued draws.

Look for many `NoAffineIndex` refusals taking the early path, successful `affine_cache_hits` with continued fixed-function draws, and rejects accompanied by the existing reason counters. A high miss rate with changing mesh descriptors/bounds is safe; this is a one-decision cache, not an unbounded draw-history cache.

## Validation

Release solution build: **0 errors / 235 warnings**, **101.71 s**. All four suites ran sequentially in separate FIFO acquisitions, using x64 .NET and shared restore packages. Tested source hashes are in [source-snapshot.json](../bench/b3-affine/source-snapshot.json); exact argv is in [commands.json](../bench/b3-affine/commands.json). Build log: [build.log](../bench/b3-affine/build.log). Tests used `--no-build --blame-hang-timeout 120s --blame-hang-dump-type mini`; ShaderCompiler uses the prescribed serial runsettings. Only the two Metal suites required a device. No blame hang or aborted run occurred.

- GPU Metal: **1,469 passed / 5 failed / 7 skipped**, 59 s ([log](../bench/b3-affine/metal.log)).
- Libs: **8,649 passed / 6 failed / 5 skipped**, 40 s ([log](../bench/b3-affine/libs.log)).
- ShaderCompiler: **2,158 passed / 0 failed / 1 skipped**, 9 s ([log](../bench/b3-affine/shader.log)).
- ShaderCompiler.Metal: **3,331 passed / 0 failed / 0 skipped**, 15 s ([log](../bench/b3-affine/msl.log)).

Full-suite aggregate: **15,607 passed / 11 failed / 13 skipped**. This is not an all-green validation. [validation-results.json](../bench/b3-affine/validation-results.json) records every failure and all 13 added-case outcomes; TRX files are under [results](../bench/b3-affine/results).

The six Libs failures match the documented base failures: four SSE4a Posix signal-recovery NullReferenceExceptions, import-result logging's delegate-signature mismatch, and Rosetta split-store expected 2 / actual 1. No assertions or existing test cases were changed to suppress them.

Four Metal failures reproduce in an independently built, **clean exact `46411e2df`** focused control from the GPU-wait task: lossy allocation (both 16,384 bytes) and three `MetalGuestViewTests` (`AScatteredRangePastTheRingCapIsReadWholeThroughAGuestView`, `ScatteredReadsLeaveTheStagedCacheAndTheRingEmpty`, `AnUnmapDropsTheViewAndTheNextBindReadsTheNewMapping`). The control records empty tracked diff SHA-256 and 13 passed / 4 failed. Evidence is copied here: [command/provenance](../bench/b3-affine/external-base-metal-command.json), [log](../bench/b3-affine/external-base-metal.log), [TRX](../bench/b3-affine/external-base-metal.trx). This task did not execute that external control. These guest-view failures extend the older brief's baseline list.

The fifth Metal failure was `MetalEndOfPipeOrderTests.AWaitPacketPollingALazyClockCommitsItsBuffer` (an Assert.True failure). Its fixture uses a fake backing and performs no affine shader preparation or physical-memory generation query. The unchanged class was rerun in a separate FIFO/device turn: **44 passed / 0 failed / 0 skipped**, 2 s, including that exact clock case. No source or assertions changed and all nine saved binary hashes still matched. [Focused log](../bench/b3-affine/eop-focused.log), [TRX](../bench/b3-affine/results/eop-focused.trx), [summary](../bench/b3-affine/eop-focused-result.json). The initial failure did not reproduce; a test-order/scheduling cause is an inference, not a proved diagnosis or a confirmed exact-base failure. The failed full-suite result above is retained.

Added coverage:

- Fresh draws reuse cached layout arrays and mapping proofs; changed descriptor words are read in place and reflected in bound addresses.
- All four descriptor words invalidate both successful and rejected decisions, with recovery when restored.
- Bounds, masked bias, capacity and mapping changes invalidate; unrelated/masked-off user-data bits do not.
- Unversioned providers recheck mapping; a changed mapping generation invalidates a cached negative proof.
- Portable static variants retain separate permutation identities.
- Descriptor subset reads match full materialization across changing branch activity and clean-reader availability; guest-table descriptor bytes and pointer changes are reread, unreadable bytes refuse and recover, replacement user-data arrays update clean branch evaluation, and warm table reads must allocate zero bytes.
- Real memory-provider generation changes with mapped membership and refuses observations during concurrent publication.

All added cases passed: `tests/SharpEmu.Libs.Tests/Gpu/Pipelines/AffineProgramDecisionTests.cs:45`, `ShaderProgramCacheTests.cs:28` / `:67`, `tests/SharpEmu.Libs.Tests/Memory/PhysicalVirtualMemoryTests.cs:19`, and `tests/SharpEmu.ShaderCompiler.Tests/Resources/BufferDescriptorEvaluationTests.cs:17` / `:45`.

Measured in the suite's x64 .NET test host after warm-up:

- Accepted affine decision, fresh inline descriptor, 20,000 calls: **2,101.2 ns / 32 B** uncached; **91.2 ns / 0 B** cached.
- Rejected bounds, fresh inline descriptor, 20,000 calls: **1,482.6 ns / 32 B** uncached; **81.1 ns / 0 B** cached.
- Full preparation of an empty vertex program with a fresh `VertexInputInfo` per draw, 20,000 calls: **3,068.8 ns / 192 B** feature off; **3,022.4 ns / 192 B** feature on. No measurable incremental no-affine cost in this small test; the baseline draw-input allocation remains.
- Fresh guest-table descriptor with its first word changed every call, 1,000 calls: **1,177.4 ns / 0 B**, using retained compiled evaluator storage. Failed-read recovery and changed descriptor-table pointers were also tested.

Cached inline decisions and warm ordinary table reads have explicit zero-allocation assertions. These are unit timings, not a production microbenchmark harness: fake mapping host for the decision timings, ordinary JIT/tiering, single samples, no confidence interval. They exclude real Metal mapping-generation lock cost and the game resource mix. The no-affine full-preparation absolute time must not be compared directly with the game's `ProgramPrepareLocked` figure; the feature-on/off comparison is the useful result.

## Cleanup and delivery

Deleted this worktree's `artifacts/bin` and `artifacts/obj`: **995,938,001 bytes / 1,271 files** inventoried before deletion. No `.packages` directories were found. [Cleanup plan](../bench/b3-affine/cleanup-plan.json), [completed cleanup](../bench/b3-affine/cleanup-results.json). Shared NuGet packages and other worktrees were left intact.

The report and retained command/provenance/log/TRX artifacts are also copied to `/Users/xnetcat/Projects/sharpemu/sharpemu-mac/artifacts/ds-fps/` under this task's names. No own .NET process, lock, reservation or queue entry remains. No game launch, `.app` build, push, or review agent occurred.

## Expected effect and limits

The supplied `v3a-run.md` measured `ProgramPrepareLocked` rising from 3.0 to 21.7 ms/frame at approximately 9,060 calls/frame. The target is to recover most of that 18.7 ms/frame CPU regression by eliminating repeated instruction/resource scans, repeated resolution and recursive preparation. The measured decision reductions correspond to 12.7–18.2 ms/frame only under the synthetic assumption that all 9,060 calls perform the benchmarked decision. That is a budget estimate, not a measured game saving; actual savings depend on no-affine frequency, guest-table read cost, hit rate and mapping churn. No additional GPU ms/frame saving is claimed: the same valid affine layouts and portable fallbacks remain selected, preserving the existing GPU benefit.

The cache is deliberately bounded to one decision per source entry. Interleaved bounds/descriptors can miss frequently; misses still avoid recursive materialization, invariant scans and temporary lists. Guest-table evaluator/activity storage is retained per eligible entry and reused; computed values are never retained across calls. This retains scratch capacity per plan. Rare nested evaluator operations can still use the existing transient evaluator path; the added allocation checks cover inline descriptors and ordinary guest-table reads. Mapping observations have the same lifetime boundary as `CanRead`: they do not pin a mapping against later guest mutation. Fresh descriptor reads retain the existing host ownership/finality rules.

Follow-up measurement: the orchestrator should compare `ProgramPrepareLocked`, cache hit/miss/reject counts, fixed-function/shader draw counts, allocation and screenshots on the integration build. Consider a small bounded associative decision cache only if measured misses justify its added key/storage cost.
