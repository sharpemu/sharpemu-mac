# Parallel Metal preparation and encoding experiment — final report

Branch: `ds-fps-arch-parallel`, from exact main
`bdc7ea28c8e7df0b8753966d8ba1881f60e5e764`.
Worktree: `/Users/xnetcat/.t3/worktrees/sharpemu-mac/ds-fps-arch-parallel`.
Phase 1 design/census commit: `587b282b1`; initial prototype: `e616f4a81`;
final implementation change: `118999b12b4023b7cfa56c978e1521e11272278b`.
The measured, signed prototype bundle is clean source
`c9598a26304da5db2d5a73c813986ac39864161c`; later commits contain evidence/reporting.

**Verdict: keep `SHARPEMU_METAL_V2_PARALLEL_ENCODE` default OFF.** One complete
K4 run shows no established speedup against two main controls, raises measured
peak footprint by 619–898 MiB, and the two subsequent revised K4 attempts were
killed by the unchanged OOM guard. No further games are queued. The user explicitly
requested reporting these available samples and stopping so PR validation can use
the shared queue; the original two-complete-runs-per-arm requirement is therefore
not satisfied for K4 and no longer being pursued in this task.

The delivered prototype batches **fully prepared native render draws** and encodes
contiguous chunks on four threads. Guest reads, resource materialization, argument
buffer writes, pipeline lookup and compute remain ordered on the original owner.
The full requested raw-register recorder plus parallel resource preparation is
**not implemented**. The dependency analysis below identifies the missing immutable
read-domain / writer-classification or transactional-rollback proof. Removing the
old split's prefix observation waits without that replacement would change guest
semantics. This is a negative native-encoding experiment, not a 60 fps solution.

## Completed game comparison

All three completed arms used the same private warm data root cloned from main
with `cp -cR`, the same save/idle Outpost Passage scene, 1080p, overlay enabled,
`SHARPEMU_SHADER_PRECOMPILE_BUDGET=0`, render/pass/census profiling off, and
`SHARPEMU_KEEP_BUILD_SERVERS=1`. The last option prevents the shared harness from
signalling another agent's build servers. Main was the frozen exact `bdc7ea28c`
bundle; optimized was the verified c9598a263 bundle with the flag set to 1 and
`SHARPEMU_METAL_V2_PARALLEL_WORKERS=4`. Each run used normal FIFO acquisition,
load-wait policy, the active OOM guard, HUD detection, and a full HUD+120-second
in-world window. No graphics quality or memory limit was reduced.

Accepted order was main `basecheck` → K4 `opt1` → main `base2`. Two K4 `opt2`
attempts followed but did not complete. Frame statistics discard the first 40
in-world seconds and stop at the end marker. FPS is presented entries divided by
first-to-last presentation span; the interval-count convention is also retained
in each JSON. p50/p95 are presentation intervals, not exclusive worker timings.

| Arm / run directory under `artifacts/runs/` | Entries / measured span | Skip40 fps | p50 ms | p95 ms | HUD after actual launch |
|---|---:|---:|---:|---:|---:|
| Main `20261024-211738-ds-arch-parallel-basecheck` | 117 /79.491110 s | 1.471863 | 552.483 | 1,752.232 | 289.800 s |
| K4 `20261024-221738-ds-arch-parallel-opt1` | 122 /79.703672 s | 1.530670 | 549.447 | 1,598.145 | 217.954 s |
| Main `20261024-222328-ds-arch-parallel-base2` | 127 /79.292967 s | 1.601655 | 526.479 | 1,486.331 | 179.990 s |

The two controls span 1.472–1.602fps. K4 is +4.00% against the first and −4.43%
against the second. Pooled control entries/span give 1.536678fps; the single K4
sample is 0.39% below that descriptive value. This is not a confidence estimate or
proof of equivalence: there is only one completed K4 sample and the main controls
already vary. No worker-time saving or GPU-time improvement was established.
All 8 accepted process-sampler windows in each completed run had zero detected
build/test activity. Other user/OS activity and paging remained present; these
are qualified observations on this host, not measurements on an otherwise empty
machine. Directory timestamps/wrapper durations can include load waits and must
not be substituted for actual game launch times.

All three end screenshots were opened and compared. They show the expected
knight/armour, vaulted tunnel, foliage, bright exit and the same HUD layout; no
structural rendering difference was observed. Idle animation and foliage differ,
so this does not establish bitwise game-image equivalence. End image SHA-256,
source, load and memory observations are in `basecheck-inspection.json`,
`opt1-inspection.json`, `base2-inspection.json`, and the consolidated
[`final-runtime-summary.json`](../bench/arch-parallel/final-runtime-summary.json).
Raw runs, screenshots, metrics, logs, TRXs and signed bundles remain local.

## Memory pressure and the two revised K4 guard kills

**Yes: K4 raised observed peak footprint against both main controls.** The
whole-run one-second `proc_pid_rusage` samples give:

| Arm | Outcome | Peak kernel footprint, MiB | Difference to main controls, MiB |
|---|---|---:|---:|
| Main basecheck | full window, graceful stop | 13,346 | — |
| Main base2 | full window, graceful stop | 13,067 | — |
| Revised K4 opt1 | full window, graceful stop | 13,965 | +619 /+898 |
| Revised K4 opt2, `20261024-222955` | guard kill | 14,153 | +807 /+1,086 |
| Revised K4 opt2 retry, `20261024-223948` | guard kill | 14,154 | +808 /+1,087 |

Raw field names say `footprint_mb`, but `scripts/macos/footprint.py:28` divides by
1,048,576: these are **MiB**, not decimal MB. Kernel footprint includes compressed
pages and shared-mapping charges; it is not unique resident RAM, nor should RSS,
Metal allocation and footprint be added. Kill-truncated runs do not bound how high
K4 would have grown without the guard.

The independently sampled skip40-window `task.phys_footprint` peaks were
13,351.6 and 13,052.8 MiB on main versus 13,891.5 MiB for completed K4: an increase
of 539.9–838.7 MiB in the comparable accepted windows. Unique resident+swapped
`graphics_iokit` objects in those windows were 520.6–590.6 MiB and 539.4–583.9 MiB
on main, versus 1,419.0–1,698.1 MiB for K4. Samples come from different instants and
kernel query APIs, so the one-second and object-walk peaks need not coincide.
The reported whole-run Metal-allocation peaks did not increase consistently
(main 14,881 /13,009 MiB, completed K4 13,770 MiB), so that separate counter is
not a substitute for kernel footprint or unique-object accounting.
This supports a native graphics-memory/pressure regression. It does not isolate a
leak or prove that every additional byte comes from the child encoder objects.

| Revised failure | Guard evidence, local Europe/Warsaw time | Progress | Result |
|---|---|---|---|
| `20261024-222955-ds-arch-parallel-opt2`, PID14903 | 22:35:18 free5%, swap8268MiB;22:35:20 free5%, swap8311MiB; kill22:35:20 | HUD173.15 s;~77 s in-world; no end marker | app137 /wrapper3 |
| `20261024-223948-ds-arch-parallel-opt2`, PID49153 | 22:43:47 free5%, swap8703MiB;22:43:49 free5%, swap8827MiB; kill22:43:49 | HUD195.56 s;~45 s in-world; no end marker | app137 /wrapper3 |

Both failures used the same revised c9598a263 source and K4 configuration as the
completed opt1 run. Actual-game process sampling found zero build/test alerts
(25 and23 windows respectively). Both last screenshots were opened: the expected
scene/HUD remained present, but neither is an end-window rendering comparison.
No incomplete-window FPS is accepted. These are external OOM-guard terminations,
not Metal exception crashes; they remain reliability findings rather than being
discarded as timing noise. Receipts and guard excerpts are preserved in
`bench/arch-parallel/attempts/opt2-oom-20261024-203520Z` and
`.../opt2-oom-20261024-204349Z`.

An earlier prototype at 751c2b9cf also failed the guard twice (14,418 and14,106 MiB;
archives `opt1-oom-20261024-184104Z` and`opt1-oom-20261024-191346Z`). Investigation
found that singleton batches unnecessarily created a child plus replacement tail.
Final implementation 118999b12 reuses the open serial child for those small batches;
a device regression proves 64 singleton passes create 64 children rather than 192
while matching serial pixels. That correction did **not** eliminate the revised
runtime memory-pressure failures. Owner/child autorelease pools and native retain/
release paths were inspected without identifying an unbalanced ownership path.
The remaining graphics charge has no established root cause. No more retries,
quality reductions, OOM-guard changes or user-app closures will be made.

## Implemented prototype, counters and ordering tests

`MetalRenderHost.ParallelEncode.cs` captures at most 64 fully prepared native draw
records, retains their native resources and partitions them into contiguous ranges.
It creates `MTLParallelRenderCommandEncoder` children in guest order before waking
K persistent workers. The owner executes range 0; K−1 helpers use native
UserInteractive QoS threads (managed-thread fallback), each with an autorelease
pool. One wake per helper and one batch join replace per-draw hand-offs; there is
no spin loop. Every chunk joins before record reuse, parent completion or exception
propagation. Batches of at least `min(8*K,64)` draws use K workers; smaller batches
encode on the owner into the existing open child. K defaults to 4 and clamps to 1–10.

Records copy dynamic state, buffer offsets, argument/push bindings, native
pipeline/depth state and draw arguments. Native handles stay retained through
encoding, while existing command-buffer/ring retirement retains GPU lifetimes.
Each child gets complete state and the captured union of resource residency;
previous-child/parent bindings are not inherited. Existing caches and image
ownership remain single-owner, so this prototype adds no cache-wide parallel lock
and does not expose mutable materialization state to helpers.

Eligibility rejects unbounded device-address stores, written/atomic buffers,
storage images, GDS, scratch, mesh/rectangle variants, unsupported depth/bias
behavior, tiny and profiled passes. Private mip-statistics min/add atomics are
commutative and reset/pack in ordered compute. Attachment effects retain child
creation order. A shader barrier ends the **entire parent pass**: child creation
order is not substituted for an interstage memory barrier. CP writes, DMA/fills,
labels/EOP, submissions, relay work, arbitrary guest-memory callbacks, CPU-visible
wait polls, compute and pass endings drain the prefix before their effects.
The existing synthetic occlusion policy is retained. Metal 3.2 APIs only; no new
title/hash checks or title-dependent rendering policy.

`[METAL-V2][PARALLEL]` proves application and refusal reasons. In the completed
K4 run,121 report intervals averaged:

| Counter | Per presented-frame report interval |
|---|---:|
| queued draws | 5,314.32 |
| K4 batches /small batches | 80.00 /60.59 |
| native children | 447.60 |
| refused draws | 1,018.31 |
| refused writer /shape /dynamic /pass | 206.23 /12.79 /434.18 /365.12 |
| additional barrier parent-pass ends | 24.28 |
| capture wall time | 21.710 ms |
| summed native worker wall time | 27.932 ms |
| owner join-region wall time | 37.074 ms |

Three native QoS helpers were active. Native worker time overlaps the join region;
these counters must not be added as disjoint CPU phases. The log explicitly says
`guest_prepare=ordered frozen_native_encode=1 compute=ordered`.

This leaves the dominant descriptor/table and materialization work serial. Base
phase reports put GraphicsStateApply around 29 ms/frame and ArgumentBufferWrite
around 9 ms/frame, with the latter still serial here. Perfect K4 scaling of 29 ms
could save at most 22 ms, about 5% of the supplied 430–470 ms worker budget, before
capture/retention/extra state/scheduling and extra pass ends. Capture plus the
owner join region alone measured ~58.8 ms/frame; those are added prototype regions,
not proof of a net 58.8 ms regression. No net gain was established by game FPS.

All 25 focused cases passed with `SHARPEMU_TEST_REQUIRE_DEVICE=1`:

- Contiguous partitions, real overlap, join before reuse, exception propagation
  after joining all chunks, and safe reuse following failure.
- Read-only eligibility and separate semantic/capacity/census cut classification.
- Device pixel comparison after 257 draws with changed scissors, colors and
  textures, batch rollover, and first-child completion deliberately delayed until
  the last child finishes. Actual parallel batch counts are asserted.
- Draw → barrier → compute write → EOP label → CPU wait → draw ordering, compared
  with serial execution, again requiring parallel batches.
- 64 singleton passes across two targets, exact pixels and exactly 64 native children.

## Final-source build, complete suites and signed bundle

Built/tested C# source SHA-256:
`86ed2d96539a0a6bb5de3f28a56402b57b83efd83123276b025b8ca6a3061fe0`.
Final full Release solution build succeeded: 234 warnings, 0 errors,98.39 s
MSBuild time (99.855 s wrapper). All four suites then ran sequentially, with
`--no-build --no-restore --blame-hang-timeout 120s --blame-hang-dump-type mini`.
ShaderCompiler used the BATCH3 serial runsettings; device gating was limited to
focused/Metal/MSL. Exact commands, source signatures, case results and TRX paths
are in the per-suite JSON summaries under `bench/arch-parallel/`.

| Revised-source suite | Passed | Failed | Skipped | Device required |
|---|---:|---:|---:|---|
| Focused ordering/device subset | 25 | 0 | 0 | yes |
| Metal | 1,428 | 1 | 7 | yes |
| Libs | 8,618 | 6 | 5 | no |
| ShaderCompiler | 2,134 | 0 | 1 | no |
| ShaderCompiler.Metal | 3,162 | 0 | 0 | yes |

The four full suites total 15,342 passed, 7 failed, 13 skipped; focused cases are a
subset, not additional unique full-suite coverage. The Metal failure is
`MetalLossyTextureTests.ASampledImageIsExactByDefaultAndLossyWithTheFlagOn`.
Libs failures are the four SSE4a POSIX signal bridge cases, import-result sampling
and Rosetta vector-store preservation. Exact names/messages are retained in
`known-base-failures.json` and the suite summaries. These are baseline failures,
not a claim that the suites are all green. No run aborted and no unexpected
failure was accepted by `benchmark_validation.require_ready`.

Earlier full Metal validation also failed three guest-view cases. They were
independently reproduced with the lossy-texture case in a freshly built exact-base
`git archive bdc7ea28c...`, with every archived file checked against its git blob:
17 filtered cases, 13 passed, 4 identical failures. All three guest-view cases
passed in final validation; the change does not modify their code, so no causal
fix is claimed. Base evidence is retained in `base-guest-views-*-summary.json`.

The signed prototype app was built with
`SHARPEMU_MAC_GET_TASK_ALLOW=1 scripts/macos/build-app.sh` at clean c9598a263,
osx-x64 ReadyToRun,74.042 s. Deep/strict codesign verification passed and
get-task-allow is true. `prototype-bundle.json` records 255 payload hashes;
`app-ready.json` binds the bundle to the tested source signature. The original
751c2b9cf bundle remains APFS-cloned separately for reproducing historical failures.

The final source build, focused/full tests, packaging and all post-GUI game runs
used the normal BATCH3 FIFO, one step per acquisition. An earlier source's
sequential build/unit-test runs used only the user's explicit `int3-user-gui`
exception; no game launched under that lock and the user's GUI was never signalled.
No further build/test is needed for this report-only change.

## Hard limits and proposed partition

The host has four performance and six efficiency logical CPUs (`sysctl
hw.perflevel0.logicalcpu hw.perflevel1.logicalcpu`: 4, 6). The emulator and .NET
worker are osx-x64 under Rosetta. K=4 is the first useful experiment; K=10 is not
ten performance cores, and the guest CPU threads and compiler already compete for
those cores.

Using the supplied 430-470 ms worker frame, 70 ms wait and 200 ms GPU frame, ideal
parallelization of *all* nonwaiting worker work gives:

| K | Ideal worker ms/frame, before scheduling/serial preparation | Throughput floor with unchanged GPU |
|---|---:|---:|
| 1 | 430-470 | 430-470 ms |
| 2 | 250-270 | 250-270 ms |
| 4 | 160-170 | at least 200 ms |
| 8 | 115-120 | at least 200 ms |
| 10 | 106-110 | at least 200 ms |

This simple model treats the wait as serial and assumes overlap elsewhere; it is
not a prediction of measured FPS. Even removing that wait entirely leaves the
~200 ms GPU service requirement: this architecture alone cannot deliver 16.67 ms.

The intended pipeline has four ownership domains:

1. The interpreter owns register banks and PM4 cursors. It appends raw scalar
   register changes, copied register-array changes, draw/dispatch arguments and
   guest addresses into a bounded arena. No resource evaluator or mutable bank
   reference crosses the handover. An arena slot lives until every chunk finishes.
2. A coordinator closes an epoch at semantic observations. It resolves ownership,
   pins backing identity and native resource lifetimes, and publishes an immutable
   read domain. It partitions only a certified epoch into contiguous index ranges.
3. K persistent workers own evaluator/materialization memo state, scratch arrays,
   argument allocation ranges, pipeline lookup scratch, native encoders and
   autorelease pools. Each worker processes its range in guest order. Publication
   and joining happen once per batch, never once per draw.
4. The coordinator publishes effects and completion records in guest order. For
   one render pass it creates MTLParallelRenderCommandEncoder children in range
   order before waking workers. For mixed compute/render it creates and enqueues
   ordinary command buffers in guest order before parallel encoding. No completion
   serial may signal before all preceding work; labels/interrupts/flip callbacks
   retain their current owner.

Apple specifies child encoder execution in creation order, regardless of finishing
order, and command-buffer execution in enqueue order. Neither contract makes
host resource preparation order-independent or replaces an explicit shader-store
barrier. References: [parallel render encoder](https://developer.apple.com/documentation/metal/mtlparallelrendercommandencoder),
[enqueue](https://developer.apple.com/documentation/metal/mtlcommandbuffer/enqueue()).
These are Metal 3.2-compatible APIs; no Metal 4 command allocator is needed.

## Which inputs can move, and under what proof

| Input | Current owner / source | Batch-time treatment required |
|---|---|---|
| Scalar register fields and register arrays | Interpreter; `RegisterBanks`, `AgcExports.CommandStreamTranslation` | Own value/delta copies. Preserve float bit patterns. Copying references is insufficient. |
| PM4 headers, indirect command buffers, branches, register tables and operands | `GpuCommandInterpreter`, `PacketBlocks` | They are *interpreter* memory observations. Drain an unclassified prefix before observing them. A cached packet block is valid only under a host token reflecting all earlier issued writes. |
| Shader headers, code, continuations, user-data pointers | `ShaderPipelineCache.PrepareSource`, `SynchronizeShaderCode` | Versioned immutable bytes/backing lease at the ordered prefix, not a VA-only cache key. Earlier GPU/CP writers must finish or be represented by the correct ordered overlay. |
| Descriptor words and recursively read tables | `ShaderProgramCache` materialization, host `TryReadGuestWord` | CPU-visible bytes can be read in parallel only from an immutable read domain whose writer prefix has been resolved. A descriptor can be produced by earlier GPU work in the same proposed batch. Split there; neither enqueue nor a render child repairs a stale CPU read. |
| Vertex/index/storage contents consumed only by Metal | Existing window/offset bindings and indirect draw buffers | Keep GPU reads on the GPU. Pin mapping/backing and native handles; preserve barriers and aliases. Do not copy GPU-produced indirect counts onto the CPU. |
| CPU-expanded indices, meshlets and CPU fallback indirect arguments | `RenderExecutor`, `MetalMeshletCache` | These are CPU observations too. Require the same prefix/finality proof as descriptors, or ordered fallback. |
| CP DMA, fills/copies, atomics, WRITE_DATA and pending labels | `CommandMemoryOperation`, `_cpOverlay`, `_labelOverlay` | Close epochs before effects. Atomic/consume return values must be observed in order. Apply overlays only for their proper issue prefix; never expose a future label to an earlier draw. |
| Labels, EOP/EOS, RELEASE/ACQUIRE, CPU-visible waits, GDS and reports | Interpreter + completion queue | Close/submit appropriate prefix. Preserve deferred-publication decisions, suspension/retry, timestamps and interrupts. A CPU wait cannot depend on an uncommitted batch. |
| Guest CPU writes during a frame | Raw/managed backing writes + page watches | Base packet tokens explicitly do not cover all CPU/raw stores. A generation check is not an immutable snapshot. A correct deferred-read design needs provider-level read leases/COW or an equally strong happens-before proof. Do not silently redefine races by reading the live table later. |
| Image aliases, GPU-owned image bytes, pending writebacks | `MetalImageCache`, `WaitForGuestBytes`, store publication | Preserve single authoritative ownership and publication before CPU stores. No independent per-worker image cache may own conflicting aliases. No worker may invoke a coordinator callback while holding a lock the coordinator needs. |

The central issue in a raw draw recorder is circular: until materialization, it
cannot know whether a draw writes a later PM4/table/code address through an
unbounded device pointer. It therefore cannot validate the next packet-block token
against a prefix whose effects have not yet been classified. The previous split's
unknown-writer guard protected this exact boundary. Removing its synchronization
without replacing its proof changes guest-visible behavior.

A second possible design is transactional recording: provisional packet and
operand reads are recorded with their issue prefix, effects are classified during
batch preparation, and a dependent suffix is rolled back and reinterpreted after
executing the prefix. This avoids requiring the interpreter to classify every
shader before recording. It requires a complete interpreter checkpoint (banks,
indirect-buffer frames/cursors, predication, deferred operands and queue state),
owned read observations and delayed host effects. Merely rejecting a stale
materialization result cannot repair already-interpreted stale PM4. This prototype
does not implement transactional interpretation or that rollback protocol.

## Cache and allocator ownership

The supplied v3-resmat trace found 96 real worker monitor waits totaling
1,008.298 ms / 30,191.49 ms = 3.340%, and 24 actual GC pauses totaling
1,643.214 ms = 5.443%. Its sampled Monitor/PollGC shares (26.40% / 52.93%)
are not corresponding CPU or blocked-time fractions; profiler safepoints biased
those samples. This design targets the repeated preparation work. Replacing it
with K callers of one big cache lock would retain serialization and add contention.
The report's separate phase captures (524.1 -> 466.0 ms/frame) are wall time, not
precise OS CPU self-time; they cannot simply be divided by K as a measured result.

`ShaderPipelineCache` has `_resolveGate` around source/input resolution and reused
user-data arrays, plus `_gate` around program preparation. Merely calling the same
cache from K workers serializes the expensive section. Split immutable decoded
programs, source identities and completed pipeline states from per-worker options,
input arrays and evaluator cursors. Publish cache misses once per key, outside the
read-mostly hot path; hold native handles through batch retirement. Cache misses
may fall back or join compilation, but must not retain a global preparation lock.

`ResourceMaterializationCache` owns mutable young/old dictionaries, byte budgets,
learned rebases, recent entries, spare recorders and comparison scratch. Give each
worker its own bounded cache and scratch; share only immutable plan/result entries
with independent validity evidence. Do not multiply the existing memory budget by
K accidentally. Cache membership, byte equality and ownership finality are separate
conditions. Thread-static evaluator scratch alone does not make the cache safe.

`MetalImageCache` uses a RegionLock for image lookup, aliases and ownership, and a
separate fence gate for CPU-store/writeback publication. Keep ownership decisions
on the coordinator, then give workers immutable image/view handles and retained
resources. Pure layout/view request calculation can move; ownership transitions,
uploads, writebacks, deferred clears and CPU-store requests cannot be run freely by
K independent hosts. The fault-side writer path must remain serviceable at batch
waits and compile waits.

`MetalGuestMemory` rings and `MetalStagingPool` slabs have mutable allocation and
retirement state. Reserve a disjoint slice/slab for each chunk once, use local bump
allocation, and retire it against the coordinator's final submission serial. A
chunk cannot recycle a range while another child or an earlier submission can
still read it. Keep the device-address table/window pin set fixed for the epoch.

Residency and bound-state caches belong to an encoder. Each child needs its own
complete initial pipeline, dynamic state, vertex streams, argument/push bindings,
window naming and indirect-resource residency. Parent/previous-child residency is
not inherited. Union residency once per chunk where possible; retain referenced
objects even if an image/pipeline cache evicts them while a batch is pending.

## Base-path in-game batch census

Measured on the base rendering path with only the optional passive counters,
clean source **888a006c20e20cdc37757cebb9f7f75e4a95aa30**. Signed x64 bundle
built with `SHARPEMU_MAC_GET_TASK_ALLOW=1 scripts/macos/build-app.sh` (73.451 s,
rc0); source metadata and payload hashes are in
`../bench/arch-parallel/census2-bundle.json`. The app and main's warm data root were
APFS-cloned (`cp -cR`). Overlay on, 1080p, shader precompile budget0, render/pass
profilers off. FIFO acquisition/release was enforced for both build and run.

Run: `artifacts/runs/20261024-134426-ds-arch-parallel-census2`, completed at the
script's stop marker, graceful exit. HUD at 232.415 s; measured HUD+40 through
HUD+120. **121 census reports /121 presented frames**, over 79.496 s between the
first/last selected presentation. Raw issued work averaged6,376.54 draws and
2,751.07 dispatches/frame; encoded work averaged6,330.91 draws and2,690.95
 dispatches/frame. Skipped/retried translation work explains why raw and encoded
censuses are distinct. `census2-summary.json` retains the full histograms, packet
counts, cut counts and bounds; `summarize.py RUN 40` reproduces it.

The end screenshot was opened and inspected: knight from behind in the vaulted
Outpost Passage tunnel, bright exit, foliage/ivy, armour detail, HUD and overlay;
no black/missing/garbled regions observed. The initial HUD screenshot also showed
the expected Outpost Passage entry animation. This diagnostic's 1.522 fps,
p50 485.327 ms and p95 1,827.172 ms are **not** an uninstrumented A/B control.
Process sampling found no competing build/test activity in the accepted window.
One pre-window alert was `triald`, falsely matched by an unanchored `ld$` pattern;
the sampler now matches exact build-tool names. Other OS/UI/inference load remains
recorded in `census2-system.jsonl` and is not assumed absent.

### Measured batch sizes (maximum64 work items)

`coarse` cuts at register-table reads in today's interface. `strict` additionally
closes an unclassified draw prefix before command/operand observations, as required
by the prior split's unknown-writer rule. `symbolic` records table operations for
later preparation while keeping the other candidate boundaries. It is an
optimistic opportunity measurement, **not** permission to defer reads.

| Policy | Graphics batches/frame | Mean draws/batch | Batch p50 /p95 | Singleton batches | Draw work in batches >=32 | Ideal equal-work K4 bound |
|---|---:|---:|---:|---:|---:|---:|
| strict | 6355.55 | 1.003 | 1 /1 | 99.83% | 0.00% | 1.003x |
| coarse | 1536.93 | 4.119 | 1 /16 | 58.18% | 29.24% | 2.544x |
| symbolic | 154.59 | 40.954 | 64 /64 | 17.87% | 94.21% | 3.905x |

| Policy | Compute batches/frame | Mean dispatches/batch | Batch p50 /p95 | All-work batches/frame |
|---|---:|---:|---:|---:|
| strict | 2751.07 | 1.000 | 1 /1 | 9106.62 |
| coarse | 822.16 | 3.273 | 1 /14 | 2359.08 |
| symbolic | 297.22 | 9.054 | 1 /64 | 451.81 |

The strict policy achieves 1.003x at K4 for graphics and 1.000x for compute: widening
that existing split without a new read/effect proof does not create useful chunks.
Keeping table operations symbolic exposes substantial potential: 94.21% of graphics
work is in batches of at least 32, with 3.905x ideal graphics and3.642x all-work K4
bounds. These bounds assume identical cost per item, perfect scheduling and no
coordinator, validation, memory, compilation or capture cost; they are not measured
CPU speedups. They also omit P/E-core asymmetry and guest-core competition.

### Candidate synchronization counts per presented frame

| Boundary | Observed candidate events | Nonempty coarse batches closed | Nonempty symbolic batches closed |
|---|---:|---:|---:|
| Register-table loads | 5500.56 | 2022.90 | 0.00 |
| Barrier /EVENT_WRITE class | 1054.24 | 262.36 | 266.47 |
| Label /WRITE_DATA class | 558.36 | 43.00 | 45.56 |
| Guest waits | 517.13 | 6.00 | 13.00 |
| Backend render-pass ends | 55.79 | 0.85 | 11.69 |
| Guest submissions | 78.41 | 2.03 | 2.03 |
| DMA /copy /atomic class | 40.73 | 0.00 | 0.00 |
| Control /indirect operand class | 0.50 | 0.00 | 0.00 |
| Draw /compute kind transitions | 61.77 | 0.00 | 0.11 |
| Host GPU/serial waits | 0.98 | 0.98 | 0.98 |
| Pending relay boundaries | 1.00 | 0.98 | 0.98 |
| Capacity64 (coarse events; symbolic110.99) | 19.98 | 19.98 | 110.99 |

Event counters include empty boundaries and different observers of one semantic
point; their sum is not a unique stall count. Nonempty-cut columns identify the
single reason that closed each nonempty batch. Frame cuts report 2.00/frame because
both the wrapped flip packet and reporting hook close; no nonempty frame cut
remained. Register-table count is 5,500.56/frame under both policies, but symbolic
records them instead of closing. Strict additionally observes 39,126.50 packet-read
boundaries/frame; 9,054.02 of those close a nonempty strict batch. None of these
simulated cuts is a measured thread wait.

For all work, the optimistic symbolic equal-item bounds are 1.940x (K2),3.642x (K4),
6.460x (K8) and7.284x (K10). Applying them illustratively to 360-400 ms of nonwaiting
worker work plus 70 ms serial wait gives about 256-276,169-180,126-132 and119-125
ms respectively, before actual serial work and overhead. With the unchanged
~200 ms GPU service time, K4 would already approach the GPU floor in this idealized
model. It is not a route to 60 fps by itself.

## Additional source findings for a complete raw recorder

`ShaderProgramCache.Decode` already owns `ShaderCodeCapture` ranges produced by
`RecordingCpuMemory` for prewarm/corpus replay. This is useful source data, but it
is not presently a positive immutability certificate. `TakeRanges` merges
repeated/overlapping reads and keeps the earlier overlapping bytes without
checking whether a later read returned different bytes. A certificate would have
to reject such conflicts, cover headers/fused registration/code identity, retain
backing identity and lifetime, and validate actual consumed bytes. A shader VA or
declared hash alone is insufficient.

A possible path beyond the prototype is a conservative unspecialized read-only
shader certificate plus exact backing ranges for attachment writes. The raw
recorder could then answer interpreter reads proven disjoint from its pending
attachment writes without classifying every draw synchronously. Unknown shader
calls, device-address stores, source changes, unmapped/released backing or alias
uncertainty would close the batch. Pure workers would still reject descriptor or
code reads overlapping earlier target writes; the coordinator would execute that
prefix and retry the dependent suffix in order. This needs actual source and
memory proof objects, not an unvalidated previous-draw hint.

`IGuestBackedSpace` exposes alias access and an alias snapshot identity. In
`SharedBackingViews`, alias access pins backing access and participates in copy
reservation exclusion; its existence does not establish frozen CPU bytes. The
mapped-read watch API explicitly says it excludes neither native/JIT stores nor
subsequent mapping effects. A new preparation domain must distinguish lifetime,
mapping identity, byte consistency and permission to publish GPU work. The
existing detached observations must not be promoted into that authority.


## Source anchors for the dependency decisions

These anchors refer to the base plus the passive census, before applying the
native-encoding experiment:

- `src/SharpEmu.Libs/Gpu/GpuCommands/GpuCommandInterpreter.PacketBlocks.cs:9`:
  token/prefetch contract; native and guest CPU stores are not observed and a block
  is bounded by its interpreter slice.
- `src/graphics/SharpEmu.Gpu.Metal/V2/MetalRenderHost.PacketBlocks.cs:15`:
  token covers CP overlays, classified GPU stores and image ownership, not raw CPU
  bytes or unclassified effects.
- `src/graphics/SharpEmu.Gpu.Metal/V2/MetalRenderHost.cs:1730`:
  CPU guest reads publish owning images or wait for their pending GPU serial.
- `src/SharpEmu.Libs/Gpu/Pipelines/ShaderPipelineCache.cs:215` and `:301`:
  shared mutable source/user-data resolution gate. Per-worker executor objects
  alone would still call this same gate.
- `src/SharpEmu.Libs/Gpu/Pipelines/ShaderProgramCache.cs:209`:
  mutable materialization cache belongs to the program cache; it is not a
  concurrent immutable shared plan directory.
- `src/SharpEmu.Libs/Gpu/Pipelines/ShaderPrewarmList.cs:34`:
  shader read captures merge overlapping ranges without a conflicting-byte test.
- `src/SharpEmu.Libs/Agc/AgcExports.CommandStream.cs:230` and `:344`:
  existing raw register snapshots/deferred draws are drained in order, rather than
  granting parallel host ownership.
- `src/graphics/SharpEmu.Gpu.Metal/V2/MetalGuestMemory.cs:1226` and `:1601`:
  ordered ring reservation and mutable submission residency. The frozen-native
  experiment retains these on the original owner.


## Implementation locations and evidence handoff

| File:line | Change |
|---|---|
| `src/SharpEmu.Libs/Gpu/GpuCommands/GpuCommandInterpreter.cs:37`, `:454` | Opt-in passive packet/operand census; static flag bypasses observer work when disabled. |
| `src/SharpEmu.Libs/Gpu/GpuCommands/ICommandBatchObserver.cs:7` | Diagnostic observer contract. |
| `src/graphics/SharpEmu.Gpu.Metal/V2/MetalBatchCensus.cs:11` | Strict/coarse/symbolic contiguous batch distributions, cut reasons and graphics/compute histograms. |
| `src/graphics/SharpEmu.Gpu.Metal/V2/MetalRenderHost.BatchCensus.cs:10` | Host observer, opt-in only. |
| `src/graphics/SharpEmu.Gpu.Metal/V2/MetalRenderHost.ParallelEncode.cs:17`, `:102`, `:192`, `:265` | Default-off switch, immutable native capture, ordered child creation/drain, complete child bindings. |
| `src/graphics/SharpEmu.Gpu.Metal/V2/MetalParallelBatchWorkers.cs:60`, `:88` | Persistent batch wake/join, exception-safe completion and contiguous partition. |
| `src/graphics/SharpEmu.Gpu.Metal/V2/MetalRenderHost.Rendering.cs:1021`, `:1145`, `:2262` | Parent/child pass creation and completion; barriers end the whole parallel parent. |
| `src/graphics/SharpEmu.Gpu.Metal/V2/MetalCommandStream.cs:58`, `:758`, `:1003`, `:1026`, `:1047`, `:1892` | Submission/CPU-visible effects drain queued work; PERF reporting. |
| `src/graphics/SharpEmu.Gpu.Metal/V2/MetalRenderHost.cs:1628`, `:1835`, `:2570` | Relay/read boundaries and worker disposal. |
| `src/graphics/SharpEmu.Gpu.Metal/V2/MetalRenderHost.Pipelines.cs:25`, `MetalRenderHost.DepthOnlyJoin.cs:60` | Owner-only eligibility memo; parallel parent excluded from depth-only joining. |
| `tests/SharpEmu.Gpu.Metal.Tests/MetalParallelBatchTests.cs:16`, `:31`, `:59` | Partition, overlap/lifetime and failure tests. |
| `tests/SharpEmu.Gpu.Metal.Tests/Scenes/ParallelEncodeSceneTests.cs:13`, `:48`, `:108` | Small-pass allocation, reverse-completion pixels, and barrier/compute/EOP/CPU-wait ordering. |

A first census attempt (`20261024-122606-ds-arch-parallel-census`, source `68edc6075`)
stopped at the pre-HUD watchdog, with no HUD/end marker. It contributes no in-world
histogram or FPS. The successful census above supersedes it. An earlier exact-main
control (`20261024-140403-ds-arch-parallel-base1`) completed at 1.409826fps but is
historical, not pooled into the recent comparison. Incomplete game runs and
failed-predecessor preflight refusals are not accepted samples.

Reproduction helpers `step.py`, `game.py`, `summarize.py`, `validate.py`,
`prototype-build.py`, `prototype-app.py`, `benchmark_validation.py` and
`source_manifest.py` remain in `bench/arch-parallel/`. Signed binaries and raw test/
game output are local; committed receipts record exact source, hashes, commands,
counts and screenshot observations. No more jobs are launched or queued: the
user's final instruction is to hand back the current negative result and stop.
User apps and other agents' processes, locks and reservations remain untouched.
Own `artifacts/bin `, `artifacts/obj` and isolated-base corresponding build outputs
were removed after validation/packaging; no `.packages` remains. Main checkout was
not edited. All changes are local to this branch/worktree, with no review or push.
