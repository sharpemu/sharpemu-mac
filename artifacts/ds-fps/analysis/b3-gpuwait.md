# b3-gpuwait

The candidate removes eligible compute-descriptor waits from the GPU worker by parking the consumer packet until its exact producer completes. The available warm pair shows host wait falling from 105.264 to 0 ms/flip, with idle rising from 10.045 to 109.331 ms/flip. **A reliable FPS improvement is not established.**

Implementation commit **`9f04fe46cf7634146c545bdb914d4bf158211586`** (`[GPU] Defer committed compute resource reads until producer completion`), branch `ds-fps-b3-gpuwait`, based on exact int2 **`46411e2df57257fd80f9c4e1c910293c809a97f4`**. No push. Candidate worktree: `/Users/xnetcat/.t3/worktrees/sharpemu-mac/ds-fps-b3-gpuwait`; separate pristine baseline worktree has the `-int2` suffix.

The Release build and all four suites completed; all six added cases passed. Eleven failures are baseline-confirmed or documented below. **The requested two warm runs per arm and warm phase-profile pair remain incomplete.** After repeated memory-guard failures the user authorized two further attempts, with a shorter 90-second window if needed, then instructed finishing with available evidence. Both further attempts were killed, including the 90-second exact baseline. All remaining game/profile controllers were cancelled; runtime work stopped at that limit.

## Change

Default-on `SHARPEMU_METAL_V2_ASYNC_RESOURCE_READS` (`=0` disables) suspends a compute dispatch's resource materialization when a four-byte read needs an already-submitted GPU producer. No consumer dispatch has been encoded. Its packet cursor stays fixed; one serial-specific callback on the existing completion queue wakes it while other queues may run. Retry prepares fresh resources. Completion retirement stays on the GPU worker and advances ring/copy/overlay state even when a command buffer stayed open. Timer retries and unrelated host publications do not wake these known serial dependencies.

Uncommitted producers, images, other consumers and non-dword reads retain the original waits. There is no new native Metal API, spin, forced GPU submission or synchronous GPU round trip. The normal asynchronous coalescer still commits work; yielding can change its boundaries. The original two-budget transient-memory bound remains in force.

Default-off `SHARPEMU_PROFILE_GPU_WAITS=1` records bounded call stacks and per-wait serial, committed/open/completed floors, read range/origin, materializing shader and flush/signal/drain timing. The original-behavior diagnostic app was built before the optimization; its exact source/provenance is retained in [diagnostic-source](../bench/b3-gpuwait/diagnostic-source/).

## Available warm skip40 evidence

These are **context**, not a completed two-per-arm comparison. The first two rows are the first qualified baseline and following qualified candidate; intermediate cold attempts are listed in the inventory. The extra candidate was designated extra before seeing its metrics because the preceding baseline slot failed its warmth gate. It is retained despite its lower FPS. All three had a correct inspected end screenshot, zero measured pipeline creations/library wait, and no competing build/test/game in process snapshots.

| Operation | Arm | Samples / span s | FPS | p50 ms | p95 ms | PERF flips / seconds |
|---|---|---:|---:|---:|---:|---:|
| `repeat-int2-1` | int2 | 137/79.558 | 1.722014 | 525.148 | 852.397 | 132/77.34 |
| `clean-opt-2` | candidate | 140/79.616 | 1.758430 | 509.648 | 918.542 | 133/76.54 |
| `repeat-opt-2-verified` | candidate extra | 93/79.405 | 1.171211 | 703.301 | 1810.823 | 89/77.34 |

| Complete PERF windows, per presented flip | int2 pair | candidate pair | candidate extra |
|---|---:|---:|---:|
| Host wait ms | 105.264 | 0.000 | 0.000 |
| Host waits | 1.000 | 0.000 | 0.000 |
| Worker idle ms | 10.045 | 109.331 | 181.067 |
| GPU busy ms | 228.458 | 217.076 | 335.009 |
| Timed command buffers | 20.765 | 21.684 | 23.303 |
| Kernel duration ms | 242.339 | 232.112 | 372.680 |
| Library wait ms | 0.000 | 0.000 | 0.000 |

The pair's FPS difference is 2.11%, with candidate p95 7.76% worse. Avoided host wait mostly reappears as idle. This is consistent with the consumer still depending on the producer when insufficient independent work is available; it does not prove why every idle interval occurred. Timed command buffers/flip were 20.765 and 21.684. This is consistent with changed coalescer boundaries; the counters do not prove the cause of every difference. The lower GPU duration in this pair does not establish removed GPU work. No pooled two-per-arm estimate or claimed speedup is reported.

The candidate pair recorded 268 suspensions, 269 wakes and 25,196.616 deferred milliseconds in its complete PERF windows; a wake can belong to a suspension before the selected report window. No refusal occurred in those windows. Deferred time overlaps/straddles windows and is not saved frame time. The extra candidate recorded 165 suspensions/wakes, zero host wait and 181.067 ms idle/flip.

## Final retries and stop boundary

Times below are 2026-10-24, Europe/Warsaw.

| Attempt | Standing window | Extra profiling | Actual result |
|---|---:|---|---|
| `stable-int2-1` before final user instruction |120s| Off | Guard killed PID 3786 at 13:17:27, HUD+28.012s; peak 12,757 MB; no end marker. |
| `stable-opt-1`, first further retry |120s| Off | Guard killed PID 34199 at 13:25:15, HUD+66.522s; peak 13,330 MB; two 5% free samples; no end marker. Last screenshot showed the correct knight/tunnel/foliage/HUD. |
| `retry90-int2-1`, second further retry |90s| Off | Guard killed PID 59394 at 13:31:34, HUD+62.347s; peak 12,801 MB; free 5% then 4%; no end marker. Last screenshot showed the correct world with Outpost Passage caption. |

The last two failures reach the user's explicit stop condition. The shorter 90-second configuration was prepared symmetrically for both arms; the candidate 90-second run was cancelled before execution because the limit was reached first. Neither killed run is promoted to a complete performance sample or planned end screenshot. User applications and foreign processes were left running; the required guard was unchanged.

Earlier 600-second standing attempts also failed under the guard: exact int2 at HUD+570 with 14,601 MB peak footprint, candidate at HUD+141 with 13,613 MB. A proposed 420-second protocol never ran. A third already-acquired 600-second baseline exploration was stopped explicitly before performance selection. The report preserves the entire attempt history rather than substituting cold or incomplete samples. Guard logs, commands, process snapshots, cancellation records and results are in the canonical evidence directory: `/Users/xnetcat/Projects/sharpemu/sharpemu-mac/artifacts/ds-fps/bench/b3-gpuwait/`.

There is **no warm render-phase profile per arm**. The only complete phase table retained below is the cold original-behavior diagnostic for attribution. It cannot establish steady worker savings or substitute for the requested A/B phase pair.

## Wait attribution (cold diagnostic; no FPS claim)

The initial diagnostic captured storage-buffer byte dependencies in CPU indirect-draw argument reads
and shader resource-table evaluation. It also captured four `Other` waits from the existing transient
memory budget (1,181.940 ms total across cold loading), and the final shutdown drain. The byte reads
were already targeted: some waited for a producer older than the latest submitted serial, while a newer
buffer remained open. EOP/labels already use GPU writes/completion callbacks; they are distinct from
the worker's synchronous byte-consumer waits.

For example, resource materialization of `0x28615BDDC8431AB6` read a dword at `0x70A837EE50` through
`ShaderProgramCache.ReadGuestWord -> RuntimeValueEvaluator.ReadRawWord -> ResourceMaterializer ->
RenderExecutor.Dispatch`. Three captured waits totalled 858.290 ms, of which 856.298 ms was signal wait
and 0.097 ms completion drain. Each targeted an older submitted producer. These cold figures are **not**
the requested steady in-world ms/frame. Complete call stacks, per-site totals and example serials:
[diagnostic-sites.md](../bench/b3-gpuwait/diagnostic-sites.md).

`CommandGpuWait` and `host_wait_ms` are two accountings of these worker waits; they must not be added.
A reduction in that phase will only count as a frame-time gain if it does not reappear as idle. The
candidate removes no GPU work, so expected direct GPU execution savings are zero; useful worker/frame
savings require runnable independent work while the resource producer completes.

## Repeat diagnostic attribution

The correct-scene repeat captured 20 synchronous byte waits in its skip40 diagnostic window.
The phase windows charge 175.402 ms/frame to CommandGpuWait (1.0 calls/frame); the distinct complete
PERF windows show 135.453 host-wait ms/frame (18 waits / 19 flips). Different window boundaries
explain the different totals. These remain cold figures, not the accepted comparison. Fifteen of the
20 diagnostic waits read the GPU-written storage-buffer table for shader `0x28615BDDC8431AB6`,
through a compute dispatch materialization. Those 15 cost 2,813.464 ms total, 2,806.404 ms in signal
wait and 1.363 ms draining completions. Every one targeted an older committed serial while a newer
buffer remained open. This is the candidate class. Full stacks and all loading/shutdown sites:
[diagnostic2-sites.md](../bench/b3-gpuwait/diagnostic2-sites.md).

## Source and ordering contract

All paths below are relative to the candidate worktree at implementation commit `9f04fe46c`.

| Change | Source |
|---|---|
| Eligibility, default-on switch, exact serial handoff, worker-side retirement | `src/graphics/SharpEmu.Gpu.Metal/V2/MetalRenderHost.ResourceReadRetry.cs:11`, `:34`, `:54` |
| Catch before dispatch encoding and register one existing-completion-queue callback per suspension | `src/graphics/SharpEmu.Gpu.Metal/V2/MetalCommandStream.cs:2065` |
| Keep the dispatch packet at its cursor with a GpuSerial condition | `src/SharpEmu.Libs/Gpu/GpuCommands/GpuCommandInterpreter.Draws.cs:258` |
| Targeted wake, completion-before-park race protection, exclusion from unrelated/timer/publication retries | `src/SharpEmu.Libs/Gpu/GpuCommands/CommandStreamQueue.cs:949`, `:980`, `:1149`, `:1320`, `:1369` |
| Default-off wait stacks and detailed per-wait serial/timing records | `src/graphics/SharpEmu.Gpu.Metal/V2/MetalRenderHost.GpuWait.cs:18` |
| Real GPU-produced descriptor behind a held native event, with enabled/disabled/uncommitted cases | `tests/SharpEmu.Gpu.Metal.Tests/Scenes/ComputeSceneTests.ResourceReadRetry.cs:21` |
| Queue progress, exact producer wake and wake-before-park regression cases | `tests/SharpEmu.Libs.Tests/Gpu/GpuCommands/CommandStreamResourceReadTests.cs:21`, `:55`, `:78` |

The producer must already be committed. The consumer packet cannot advance past its dispatch, and no
consumer dispatch was encoded before suspension. Other guest queues can run under the scheduler's
existing interleaving rules. Completion wakes only serial dependencies at or below that completed
serial. The worker retires completed transient memory/copies/overlays and clears the dword read cache
before resuming work, so the retry sees final GPU bytes. Long producers stay parked rather than being
retried by the fallback timer. The original two-budget transient-memory bound remains in force.

No speculative descriptor becomes a binding, no EOP publication is moved earlier, and the handler adds no GPU
signal or forced submission. Normal coalescer boundaries can change when a queue yields. Eligibility checks run only on an already-pending read; the
exception and callback are paid by a suspended materialization, rather than every descriptor word.

## Whole-run wait classification

The repeat's 69 captured waits span cold loading, the in-world window and shutdown. These totals are
for attribution only; they must not be divided by just the in-world frame count.

| Guest-visible reason | Waits | Total ms | Disposition |
|---|---:|---:|---|
| Compute materialization needs final GPU-written descriptor/table dword, producer submitted | 29 | 4,685.852 | Candidate defers this consumer packet until its exact producer completes. |
| Compute descriptor producer is still in the recording buffer | 8 | 405.605 | Retain commit and synchronous wait; suspension alone could strand its producer. |
| CPU resolves 16/20-byte indirect draw arguments | 25 | 186.222 | Retain existing targeted byte wait; this patch only handles retry-safe compute materialization. |
| Transient staging/ring budget before recording a third budget | 6 | 1,090.756 | Retain previous-budget serial wait and bounded memory. |
| Shutdown drains the final buffer | 1 | 0.174 | Retain outside measured in-world window. |

The dominant stack is `GpuCommandInterpreter.DispatchIndirect -> MetalRenderHost.DispatchDirect ->
RenderExecutor.Dispatch -> ShaderProgramCache -> ResourceMaterializer -> RuntimeValueEvaluator.ReadRawWord
-> ShaderProgramCache.ReadGuestWord -> MetalRenderHost.TryReadGuest -> WaitForCommittedSerial`.
For example, repeat site 12 reads `0x70A837F150+4` for reader `0x28615BDDC8431AB6`; the tracker classifies
its producer as StorageBuffer. Serial fields identify the precise submission whose shared-memory bytes
must be final. `last=` describes the last work encoded at observation and does not identify an older
producer shader. Complete per-site stacks, serial examples and timings are retained in the linked inventory.

Representative captured call paths for the retained classes are `DrawIndirectArguments -> RenderExecutor.DrawAutoWithCpuArguments -> TryReadGuest` (16 bytes), `DrawIndirectMulti -> RenderExecutor.DrawIndexedWithCpuArguments -> TryReadGuest` (20 bytes), and `RecordDraw/Dispatch -> BeginPreparation` (the inlined transient-budget wait). Uncommitted descriptor reads have the materialization stack above but a serial beyond `committed_before`; the original path commits before waiting. Full stack variants and serial/range examples are in the linked site inventory. The tracker conservatively covers writable storage ranges; the capture does not prove every stalled dword changed. The probe identifies the exact producer **command-buffer serial** and storage origin; it does not identify the shader/packet that previously wrote each range. `last=` must not be used to invent that attribution.

The capture observed no synchronous ReadImage, ImageCache, CommandProcessor or Synchronize wait.
EOP labels already write on the GPU or publish through the existing completion queue
(`MetalCommandStream.cs:267`); flips likewise use completion callbacks. Flip-slot and guest-label
suspensions may contribute idle time, but do not explain these synchronous CommandGpuWait records.
The Other waits map to `CommitIfTransientOverBudget` (`MetalRenderHost.cs:2012`), which permits two
transient budgets rather than imposing an ordinary frame-in-flight limit. The byte-wait implementation
already targets the producer serial in int2; the new behavior changes where the consumer waits.

## Validation

The final solution build at `9f04fe46c` completed in 92.21 seconds (MSBuild 1:31.05), with **0 errors and 237 warnings**:

```sh
~/.dotnet-x64/dotnet build SharpEmu.slnx -c Release -p:RestorePackagesPath=$HOME/.nuget/sharpemu-shared -nodeReuse:false -p:UseSharedCompilation=false
```

[Validated source and assembly hashes](../bench/b3-gpuwait/validated-build2-source.json) bind all eleven changed source/test files and eight assemblies. [Pre-cleanup verification](../bench/b3-gpuwait/pre-cleanup-source-and-test-hashes.json) found no differences after all runtime work. The new native enabled/committed case returned in 14.85 ms while its producer was held; both synchronous fallback cases stayed blocked until the fail-safe event opener, then produced the exact expected 64 words. These are correctness checks, not game FPS estimates.

All commands acquire their own FIFO turn, use x64 dotnet and the shared NuGet cache, disable node/compiler
reuse for builds, and keep suites sequential. Tests use `--no-build --blame-hang-timeout 120s
--blame-hang-dump-type mini`. Only the two Metal suites set `SHARPEMU_TEST_REQUIRE_DEVICE=1`;
ShaderCompiler uses the required serial runsettings.

| Suite | Passed | Failed | Skipped | Total | Actual status |
|---|---:|---:|---:|---:|---|
| SharpEmu.Gpu.Metal.Tests | 1473 | 4 | 7 | 1484 | Completed in 80.28 s process time; rc 1. All three new device theory cases passed. |
| SharpEmu.Libs.Tests | 8641 | 7 | 5 | 8653 | Completed in 48.17 s; all three new queue cases passed. Seven documented import-log/Rosetta/SSE4a failures. |
| SharpEmu.ShaderCompiler.Tests | 2155 | 0 | 1 | 2156 | Completed in 14.11 s; rc 0, serial settings. |
| SharpEmu.ShaderCompiler.Metal.Tests | 3331 | 0 | 0 | 3331 | Completed in 18.10 s; rc 0, real device required. |

The Metal failures are `MetalLossyTextureTests.ASampledImageIsExactByDefaultAndLossyWithTheFlagOn`
(allocated lossless=16384, lossy=16384), plus three `MetalGuestViewTests` assertions:
`ScatteredReadsLeaveTheStagedCacheAndTheRingEmpty`,
`AScatteredRangePastTheRingCapIsReadWholeThroughAGuestView`, and
`AnUnmapDropsTheViewAndTheNextBindReadsTheNewMapping`. The latter three exceed the batch's documented
single Metal baseline failure, but a fresh pristine exact-int2 build (86.70 s, rc 0) and focused real-device
run reproduced **all four identical failures**: 13 passed / 4 failed / 17 total in 6.38 s. Results:
[baseline-metal-validation.json](../bench/b3-gpuwait/baseline-metal-validation.json).

Raw TRX, complete diagnostics and source-bound commands are retained under
[bench/b3-gpuwait](../bench/b3-gpuwait/), with parsed results in
[test-validation.json](../bench/b3-gpuwait/test-validation.json). The initial build's two xUnit1031 warnings
in the new synchronous native test refer to joining its fail-safe gate-opening Task; the real device cases
completed and passed. No failing assertion was removed or weakened.

## Switches and counters

`SHARPEMU_METAL_V2_ASYNC_RESOURCE_READS=0` restores the synchronous behavior. It is on by default. `SHARPEMU_PROFILE_GPU_WAITS=1` enables detail only for diagnostic runs and is off for clean A/B.

| Counter | Meaning |
|---|---|
| `resource_read_suspend` | Compute materialization parked before encoding its dispatch. |
| `resource_read_wakes` | Existing completion callback notified the producer serial. A window can include a wake from an earlier window. |
| `resource_read_deferred_ms` | Time from suspension to callback; overlapping or cross-window time, not net savings. |
| `resource_read_refused_disabled` | Kill switch retained the original wait. |
| `resource_read_refused_scope` | Consumer/thread/size is outside the retry-safe compute dword scope. |
| `resource_read_refused_uncommitted` | Producer is still recording; retain commit and wait. |
| `resource_read_refused_image` | Image consumer retains its original path. |

In the complete final-candidate shader warm-up, the log recorded 1,088 suspensions and 1,088 wakes, 31 scope refusals, 5 uncommitted refusals, and no image/disabled refusals. The measured 137,733.636 deferred milliseconds are not a frame-time improvement. The incomplete warm comparison and missing profile pair leave the net effect unproven.

## Runtime method and binary provenance

Both signed apps were built locally with `SHARPEMU_MAC_GET_TASK_ALLOW=1 scripts/macos/build-app.sh`, x64 SDK 10.0.103. Each private data root was cloned from main with `cp -cR`. Int2 app build took 86.765 s; final candidate 80.430 s. Build metadata identifies exact 46411e2df and 9f04fe46c, both clean source; signing get-task-allow was verified. [Candidate provenance](../bench/b3-gpuwait/final-app-provenance.json), [baseline payload stability](../bench/b3-gpuwait/baseline-app-stability.json). No subsequent cache copying or relabelling across compiler identities was performed; the first runs and explicit extra warm-ups populated each app's own caches.

Every build, suite and game used one shared FIFO reservation and atomic lock, releasing immediately afterward. No build/test overlapped an owned game. Runtime used the supplied golden save, 1080p patch, precompile budget 0, overlay and unchanged OOM guard. The clean runs had no extra profiler. Ten-second process/cumulative-CPU snapshots checked active competition; orphan testhost 70067 was exempt only at its unchanged 13.38 s CPU total. Process sampling cannot exclude sub-ten-second interference.

Standard statistics use HUD+40..120, sample count divided by first-to-last-frame span, and percentiles of consecutive presented-flip intervals, matching the supplied helper. Complete timestamped phase/PERF windows must fit inside those bounds with a 0.1-second rounding margin and use their own actual frame counts. Warm acceptance required at most 5 pipeline creations and at most 1 ms library wait/flip; at least 75 seconds frame coverage and 60 seconds complete PERF; a correct inspected end screenshot and quiet process audit. Low FPS alone was never an exclusion. This library-wait gate is conservative: waiting for a cached library can occur without new shader compilation, so a failed warmth gate does not by itself prove compilation. The user-authorized 90-second retry would have 50 seconds after skip40, with corresponding 45-second frame/35-second report coverage; neither final retry completed. The full prospective method history is [protocol.md](../bench/b3-gpuwait/protocol.md).

Operation name, checkout and app build-info revision identify the arm. Raw run path tags can reflect reused OWN FIFO reservations. Only pending own controllers were restarted or cancelled; no shared queue entries were rewritten. Unused own reservations are left to the FIFO daemon's expiry policy. All retained benchmark links refer to the requested report in the main checkout; large raw evidence is local and is not committed with the report.

## Risks and expected savings

The patch removes no GPU execution work: expected direct GPU time saved is 0 ms/frame. Its potential worker benefit is the eligible synchronous descriptor wait (about 105 ms/frame in the earlier qualified baseline); only useful independent work during that dependency can reduce frame latency. Deferred duration is not measured savings and can overlap or straddle report windows. Lower CommandGpuWait with higher idle is still dependence on the same producer. No 60 FPS or fully overlapped CPU/GPU claim is made.

Retry can rematerialize resource state and alter normal coalescer boundaries. Compare command buffers/frame, total worker phases, idle, host wait, GPU busy and FPS/tails before integrating. The descriptor is final only after its exact producer completes; uncommitted producers and unsupported consumers intentionally keep their original waits. Native held-event tests cover exact output, no same-queue suffix advancement, independent queue progress, disabled fallback and uncommitted fallback. Pure queue tests cover completion-before-park and unrelated wake races. These tests and real rendering observations support the captured paths; they do not establish all-title parity.

Long-run host memory pressure remains an observed limitation on both arms. The existing transient two-budget bound is retained, but it does not bound all emulator/guest/device allocations. This task did not change that global policy.

Further ideas, not implemented: extend packet-level suspension to CPU indirect-draw argument consumers after proving retry safety; move eligible resource-table evaluation onto the GPU to eliminate the dependency instead of parking it; investigate the device-address-window/global-memory growth separately. Preserve the existing precise producer targeting and GPU-side EOP/label signalling.
## Cold diagnostic phase table (attribution only)

8 complete reports, 20 phase frames and 48.60 seconds; original wait behavior plus default-off detail probes. This cold table is not a warm int2/candidate comparison.

| Phase | ms/frame | calls/frame |
|---|---:|---:|
| DrawResourcePreparation | 760.571 | 6319.100 |
| Idle | 503.161 | 8.100 |
| ResourceMaterialization | 368.359 | 18085.100 |
| PipelineCreation | 295.712 | 9099.500 |
| CommandGpuWait | 175.401 | 1.000 |
| CommandStream | 45.122 | 134.150 |
| ProgramPrepareLocked | 26.002 | 9119.950 |
| ProgramTranslation | 20.673 | 3.300 |
| ProgramPreparation | 19.627 | 27370.150 |
| DrawTargetResolution | 16.799 | 6339.400 |
| GraphicsStateApply | 15.606 | 6319.100 |
| BindingTextureAcquire | 13.096 | 17887.450 |
| ArgumentBufferWrite | 12.571 | 15207.650 |
| DrawAttachmentPreparation | 12.370 | 6319.100 |
| ProgramSourceRead | 9.948 | 15247.000 |
| StorageBufferResolve | 9.367 | 51484.000 |
| ProgramPlanning | 8.997 | 3.250 |
| ResourceBinding | 8.531 | 15207.650 |
| VertexInputResolution | 8.350 | 6339.450 |
| PixelInputResolution | 7.365 | 6121.900 |
| CommandRegisterWrite | 7.364 | 16832.300 |
| ShaderHeaderLookup | 6.949 | 15247.000 |
| GuestWindowResolve | 6.887 | 52179.750 |
| ProgramCacheLookup | 6.760 | 18085.100 |
| DrawExecutor | 5.760 | 6343.400 |
| RenderPassBegin | 4.843 | 6319.100 |
| ImageViewLookup | 4.269 | 31203.600 |
| Unattributed | 4.067 | 0.000 |
| GpuWriteFlush | 4.031 | 58656.650 |
| ProgramPermutationLookup | 3.716 | 15241.850 |
| CommandPacketRead | 3.056 | 40251.000 |
| CommandLabelEncode | 3.014 | 61.400 |
| CommandDrawEntry | 2.827 | 6365.400 |
| MappedSizeClamp | 2.586 | 49224.900 |
| BindingWrittenInvalidate | 2.266 | 5555.200 |
| BindingBufferCheck | 2.037 | 39149.200 |
| BindingReport | 1.980 | 39149.200 |
| CommandFlush | 1.528 | 23.200 |
| BindingDeviceAddresses | 1.519 | 15207.650 |
| DccClearScan | 1.507 | 1777.600 |
| WriteNoteOverlays | 1.339 | 6217.000 |
| DrawDynamicStatePreparation | 1.185 | 6319.100 |
| DrawVertexBufferAcquisition | 1.007 | 6319.100 |
| DccRearm | 0.879 | 5917.950 |
| DrawIndexBufferAcquisition | 0.797 | 6319.100 |
| CommandCpBookkeeping | 0.743 | 1584.250 |
| BindingImageResolve | 0.692 | 17887.450 |
| DrawProgramResolution | 0.680 | 6339.450 |
| CommandEndOfPipe | 0.636 | 247.850 |
| ProgramLayoutRequest | 0.588 | 6.600 |
| BindingWrittenNote | 0.571 | 5555.200 |
| BindingWrittenBuffer | 0.501 | 5555.200 |
| WriteNoteStaged | 0.491 | 6217.000 |
| WriteNoteRanges | 0.436 | 6217.000 |
| ProgramResourceAssembly | 0.431 | 15241.850 |
| GraphicsResolveLockWait | 0.390 | 6339.450 |
| ProgramDecode | 0.378 | 1.100 |
| ProgramSynchronousResolution | 0.322 | 6339.450 |
| CommandProcessorWrite | 0.272 | 355.450 |
| ProgramSharedGateWait | 0.243 | 9119.950 |
| PendingGpuWriteQuery | 0.225 | 2634.200 |
| GeometrySnapshotValidation | 0.200 | 6365.400 |
| ComputeResolveLockWait | 0.174 | 2785.650 |
| PipelineSharedGateWait | 0.002 | 8.600 |


## Complete run inventory

FPS/p50/p95 use the original skip40 window for ordinary runs; warm-ups and failed sessions are not promoted into the comparison. Pipeline/library columns describe only complete PERF windows. A dash means no comparable summary. All raw data is retained, including the low warm candidate extra.

| Operation | Actual arm | Wrapper rc | FPS | p50/p95 ms | Pipelines / library ms per flip | Disposition |
|---|---|---:|---:|---:|---:|---|
| `warm-int2` | int2 | 3 | — | —/— | —/— | Warm-up; pre-HUD invalid nested pointer, rc134. |
| `diagnostic` | diagnostic (old wait) | 0 | 0.057 | 24360.4/30282.0 | 117/5097.93 | Original behavior with probes; cold/unresolved bright scene; attribution only. |
| `warm2-int2` | int2 | 0 | 0.070 | 14970.1/32427.5 | 81/6415.78 | Warm-up; cold and unresolved spawn. |
| `diagnostic2` | diagnostic (old wait) | 0 | 0.399 | 1412.0/9609.4 | 28/752.43 | Original behavior with probes; correct end scene; cold, attribution only. |
| `warm3-int2` | int2 | 0 | — | —/— | —/— | Extended warm-up 360 s; correct end scene; tail library wait 1,208.4 ms. |
| `warm-opt` | candidate | 0 | — | —/— | —/— | Final candidate warm-up 360 s; correct end scene; tail library wait 3,396.5 ms. |
| `clean-int2-1` | int2 | 0 | 1.819 | 469.8/704.6 | 1/12.12 | Correct end; cold library wait1709.5ms/141flips, rejected by prior warmth gate. |
| `repeat-int2-1` | int2 | 0 | 1.722 | 525.1/852.4 | 0/0.00 | Qualified short baseline, now context; not final selected block. |
| `clean-opt-1` | candidate | 0 | 1.773 | 484.7/893.6 | 25/7.79 | Correct end; cold 25 pipelines and library997.5ms/128flips; rejected. |
| `clean-int2-2` | int2 | 0 | 1.646 | 541.4/828.3 | 0/9.51 | Prospectively extra after cold B1; also cold 1,236.9 ms/130 flips. |
| `clean-opt-2` | candidate | 0 | 1.758 | 509.6/918.5 | 0/0.00 | Qualified short candidate, now context; not final selected block. |
| `block2-int2-2` | int2 | 3 | — | —/— | —/— | Exact baseline pre-HUD SIGSEGV RIP801784EDE, read8; not OOM. |
| `retry2-int2` | int2 | 3 | — | —/— | —/— | Exact baseline pre-HUD SIGBUS RIP8008468FF, read7199E10000; same payload hashes. |
| `retry3-int2` | int2 | 0 | 0.458 | 1957.6/4481.8 | 0/20.23 | Correct end, late HUD 568 s; cold 708 ms library/35flips. |
| `repeat-opt-2-verified` | candidate | 0 | 1.171 | 703.3/1810.8 | 0/0.00 | Designated extra BEFORE metrics; warm/quiet/correct but low 1.171 FPS, retained. |
| `retry4-int2` | int2 | 0 | 1.181 | 645.1/1866.4 | 38/41.22 | Correct end; cold 38 pipelines/3,379.9 ms library; short 71.954 s coverage. Own Raise 22 s before HUD recorded; no focus causal claim. |
| `settled-int2-1` | int2 | 3 | — | —/— | —/— | 600s protocol; memory guard killed at HUD+570; peak 14,601 MB; no end. |
| `settled-opt-1` | candidate | 3 | — | —/— | —/— | 600s protocol; memory guard killed at HUD+141; peak 13,613 MB; no end. |
| `settled-int2-2` | int2 | 3 | — | —/— | —/— | Already-acquired600s exploration cancelled after both guard failures; no end; excluded before metrics. |
| `stable-int2-1` | int2 | 3 | — | —/— | —/— | Required unchanged global memory guard KILL PID 3786 at 13:17:27 after two 5%free samples; no end marker. |
| `stable-opt-1` | candidate | 3 | — | —/— | —/— | First further retry after user instruction: unchanged memory guard killed PID 34199 after two <6% free samples. No extra profiling flags; no end marker. |
| `retry90-int2-1` | int2 | 3 | — | —/— | —/— | Second further retry failed under unchanged required memory guard, with 90 seconds standing/no extra profiler; user instructed finish after two failures. No end marker. |

[Machine-readable inventory with full run paths](../bench/b3-gpuwait/attempt-inventory.json).

## Cleanup and delivery

Implementation and this report are committed on `ds-fps-b3-gpuwait`; no push. Required cleanup completed successfully in its own FIFO turn in 4.069 seconds. It removed only the two owned worktrees' `artifacts/bin`, `artifacts/obj`, `artifacts/profiles/macos-bundle/bin` and `artifacts/profiles/macos-bundle/obj`. No `.packages` directory remained in either worktree. Signed apps, private data roots, logs, screenshots, TRX and source were retained. Every regular app file was hashed before and after deletion; all hashes matched.

| Owned app | Regular files verified unchanged |
|---|---:|
| `/Users/xnetcat/.t3/worktrees/sharpemu-mac/ds-fps-b3-gpuwait/artifacts/macos/SharpEmu.app` | 296 |
| `/Users/xnetcat/.t3/worktrees/sharpemu-mac/ds-fps-b3-gpuwait-int2/artifacts/macos/SharpEmu.app` | 296 |

[Cleanup manifest](../bench/b3-gpuwait/cleanup-manifest.json), [actual cleanup result](../bench/b3-gpuwait/cleanup-result.json). No further game/profile controller from this task remains; unused own reservations are handled by the existing FIFO expiry policy. Performance acceptance remains incomplete under the user-authorized retry limit.
