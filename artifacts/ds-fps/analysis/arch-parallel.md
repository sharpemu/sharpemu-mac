# Parallel Metal preparation and encoding experiment

Branch: `ds-fps-arch-parallel`. Base: `main` / `bdc7ea28c8e7df0b8753966d8ba1881f60e5e764`.
Worktree: `/Users/xnetcat/.t3/worktrees/sharpemu-mac/ds-fps-arch-parallel`.

Phase1 dependency analysis and the base-path in-world batch census are complete.
The opt-in native-encoding prototype builds and passes all 24 focused device/ordering
tests. All four full suites completed, with the base failures listed below. Game A/B
and the final verdict remain pending. No measured prototype speedup is claimed.
Guest preparation remains ordered in this prototype.

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

## Measurements and prototype

Measured on the base rendering path with only the optional passive counters,
clean source **888a006c20e20cdc37757cebb9f7f75e4a95aa30**. Signed x64 bundle
built with `SHARPEMU_MAC_GET_TASK_ALLOW=1 scripts/macos/build-app.sh` (73.451 s,
rc0); source metadata and payload hashes are in
`../bench/arch-parallel/census2-bundle.json`. The app and main's warm data root were
APFS-cloned (`cp -cR`). Overlay on, 1080p, shader precompile budget0, render/pass
profilers off. FIFO acquisition/release was enforced for both build and run.

Run: `artifacts/runs/20261024-134426-ds-arch-parallel-census2`, completed at the
script's stop marker, graceful exit. HUD at232.415 s; measured HUD+40 through
HUD+120. **121 census reports /121 presented frames**, over79.496 s between the
first/last selected presentation. Raw issued work averaged6,376.54 draws and
2,751.07 dispatches/frame; encoded work averaged6,330.91 draws and2,690.95
 dispatches/frame. Skipped/retried translation work explains why raw and encoded
censuses are distinct. `census2-summary.json` retains the full histograms, packet
counts, cut counts and bounds; `summarize.py RUN 40` reproduces it.

The end screenshot was opened and inspected: knight from behind in the vaulted
Outpost Passage tunnel, bright exit, foliage/ivy, armour detail, HUD and overlay;
no black/missing/garbled regions observed. The initial HUD screenshot also showed
the expected Outpost Passage entry animation. This diagnostic's1.522 fps,
p50 485.327 ms andp95 1,827.172 ms are **not** an uninstrumented A/B control.
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

The strict policy achieves1.003x atK4 for graphics and1.000x for compute: widening
that existing split without a new read/effect proof does not create useful chunks.
Keeping table operations symbolic exposes substantial potential:94.21% of graphics
work is in batches of at least32, with3.905x ideal graphics and3.642x all-work K4
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
single reason that closed each nonempty batch. Frame cuts report2.00/frame because
both the wrapped flip packet and reporting hook close; no nonempty frame cut
remained. Register-table count is5,500.56/frame under both policies, but symbolic
records them instead of closing. Strict additionally observes39,126.50 packet-read
boundaries/frame;9,054.02 of those close a nonempty strict batch. None of these
simulated cuts is a measured thread wait.

For all work, the optimistic symbolic equal-item bounds are1.940x (K2),3.642x (K4),
6.460x (K8) and7.284x (K10). Applying them illustratively to360-400 ms of nonwaiting
worker work plus70 ms serial wait gives about256-276,169-180,126-132 and119-125
ms respectively, before actual serial work and overhead. With the unchanged
~200 ms GPU service time, K4 would already approach the GPU floor in this idealized
model. It is not a route to60 fps by itself.

## Safe native-encoding experiment being prepared separately

The bounded prototype under preparation uses the immutable native-input boundary,
where all guest reads and image ownership decisions have already happened in guest
order. It deliberately does not claim to implement the raw-register recording /
parallel descriptor-materialization stage above. It can test whether parallel
native binding and draw encoding pays for its capture/retention cost, while leaving
that missing prerequisite explicit.

It admits ordinary vertex/fragment pipelines with no device-address stores,
written/atomic buffers, storage images or GDS. Attachment writes retain Metal's
child-creation order. Mip-statistics atomics are commutative; their pack/reset is a
separate ordered compute pass. Mesh/tessellation special paths, unsupported dynamic
state and profiled passes use the current path. The backend's existing synthetic occlusion policy is retained. A compute dispatch also uses the
current path. A shader barrier ends the entire parallel parent: an intrapass
barrier on one child is not treated as proof about another child's stores. Apple
explicitly says fences synchronize *different passes* and recommends intrapass
barriers for stages in one pass ([updateFence documentation](https://developer.apple.com/documentation/metal/mtlrendercommandencoder/updatefence(_:after:))).

Each frozen record owns copied dynamic state, buffer offsets, argument/push buffer
bindings, native pipeline/depth state and native draw arguments. Resources are
retained before the interpreter continues. Native children get full initial state
and their own residency declarations. A bounded 64-record batch is split into K
contiguous ranges when the batch contains at least `min(8*K, 64)` records; short batches encode
on the owner without waking workers. Relay work, submission boundaries, barriers,
transfers, compute and pass/command-buffer endings drain the prefix. Persistent
workers receive one chunk, finish it and join once per batch. A failed chunk still
joins all other chunks before any input storage can be reused.

This boundary leaves the dominant descriptor/table and materialization work
serial. Using the supplied ~200 ms prep/binding/encode aggregate as if all of it
moved would overstate its benefit. The base phase reports put GraphicsStateApply
around 29 ms/frame and ArgumentBufferWrite around 9 ms/frame, with the latter
still serial in this prototype. Perfect K=4 scaling of 29 ms saves at most 22 ms,
before capture, lifetime retention, extra child state and scheduling. That is only
about five percent of the 430-470 ms worker budget. Actual per-batch cut frequency
can make the gain much smaller or negative. The switch will stay default off
unless repeated warm measurements establish a gain and matching rendering.


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

The first diagnostic attempt (`20261024-122606-ds-arch-parallel-census`,
clean source `68edc60754acbaf0235209b27a891a87c0a7174e`) stopped at the
pre-HUD stall watchdog after 270 seconds. Its last flip was during loading at
66.49 seconds. It has no HUD/end marker and contributes no in-world histogram,
performance number or rendering comparison. A short native sample near shutdown
showed the GPU worker waiting for GC; it does not establish the cause of the
stall. The final PERF interval instead charged 202,889 ms of its 203 seconds to
worker idle, so it is not evidence of a 203-second worker GC pause. All 27 process-sampler windows had no detected competing build/test activity, and its last screenshot was black with the overlay title bar present. The second diagnostic adds a separate symbolic-table opportunity census
at clean source `888a006c2` (including separate graphics/compute histograms and nonempty-cut counts).


## Implemented prototype boundary and ordering validation

`MetalRenderHost.ParallelEncode.cs` captures at most 64 fully prepared native draws,
retains their native resources, and partitions them into contiguous chunks. It
creates `MTLParallelRenderCommandEncoder` children in guest order before waking
K persistent workers. Each child owns its encoder and declares the captured union
of resource residency. Every chunk joins before input reuse, parent completion or
exception propagation. Capacity batches with at least `min(8*K,64)` draws use K
workers; smaller batches run on the owner. The existing owner still interprets
guest memory, materializes resources, allocates argument/push storage and resolves
pipelines. Compute remains ordered. This is a working prototype of the native
encoding portion, **not the complete requested parallel resource-preparation
pipeline**. The dependencies above remain prerequisites for that larger change.

Eligibility rejects unbounded device-address stores, writable/atomic buffers,
storage images, GDS, scratch, mesh/rectangle launch variants and unsupported
dynamic depth/bias behavior. Mip-statistics min/add atomics are private and
commutative. Barriers end the whole parent pass: ordering child creation does not
replace an interstage memory barrier. CP writes, DMA/fills, labels/EOP, submission
changes, relay work, arbitrary guest-memory callbacks and CPU-visible wait polls
drain queued native work before their effects. Shader/image/materialization caches
and staging allocation remain owned by the original thread; they are not shared
unsafely or wrapped in new worker-wide locks.

`MetalParallelBatchTests` checks contiguous partitions, actual overlap, join-before-
reuse, failure propagation and reuse after failure, readonly eligibility and census
cut classification. `ParallelEncodeSceneTests` compares all pixels against the
serial path after 257 draws with changing scissors, colors and textures, while
deliberately holding the first child until the last child ends. A second device
scene checks draws, a cache barrier, compute buffer writes, an EOP label and a CPU
wait, then more draws. Both require actual parallel batches; a silent fallback
cannot satisfy the assertions. All 24 focused cases passed with
`SHARPEMU_TEST_REQUIRE_DEVICE=1`.

The user's later queue exception allows sequential builds and unit tests while
`int3-user-gui` holds the lock. It does not allow game launches. Obsolete own
build/test/package waiters were canceled; opt1/base2/opt2 game tags remain queued.
The GUI process and its lock were left untouched.


## Prototype build and complete suite results

Implementation commit: `e616f4a81`. Successfully built/tested source content
SHA-256: `3929f87e750903e303c11e9b166b7ec370e0cdcf75efb4863d2540fec0dd2f5a`.
The final Release solution build completed in 30.57 seconds with 197 warnings and
zero errors. Commands and per-case results are preserved under
`artifacts/ds-fps/bench/arch-parallel/`; `validate.py` records the exact command,
source signature, device flag and TRX path for each suite. All used `--no-build`,
`--blame-hang-timeout 120s --blame-hang-dump-type mini`; ShaderCompiler also used
the required serial runsettings. Suites ran sequentially under the user's GUI-lock
build/unit-test exception, without launching any game.

| Suite | Passed | Failed | Skipped | Device required |
|---|---:|---:|---:|---|
| Focused ordering | 24 | 0 | 0 | yes |
| Metal | 1,424 | 4 | 7 | yes |
| Libs | 8,618 | 6 | 5 | no |
| ShaderCompiler | 2,134 | 0 | 1 | no |
| ShaderCompiler.Metal | 3,162 | 0 | 0 | yes |

All six Libs failures have the exact names documented on base (four SSE4a signal
bridge cases, import-result sampling and Rosetta vector-store preservation).
The Metal failures were the lossy-texture case plus guest-view creation/rebind
cases `AScatteredRangePastTheRingCapIsReadWholeThroughAGuestView`,
`ScatteredReadsLeaveTheStagedCacheAndTheRingEmpty` and
`AnUnmapDropsTheViewAndTheNextBindReadsTheNewMapping`. These were **not assumed**
to be inherited: an isolated `git archive bdc7ea28c...` was built afresh, after
checking every archived file against the base git blob. Its device-required
17-case guest-view/lossy filter passed 13 and failed the identical four assertions.
See `base-validation.py`, `base-guest-views-build-summary.json` and
`base-guest-views-test-summary.json`. No new failure or aborted test run was
accepted. The 13 suite skips are preserved rather than counted as passing.


## A/B status and ready measurement bundle

The signed measurement bundle was built with
`SHARPEMU_MAC_GET_TASK_ALLOW=1 scripts/macos/build-app.sh`, osx-x64, ReadyToRun,
clean source `751c2b9cfd45a34209f097079a1c0a0f1dd8ea23`. `codesign --verify
--deep --strict` passed; get-task-allow is true. `prototype-bundle.json` records
the metadata and SHA-256 of all 255 managed/native/configuration payload files.
The flag remains **default off**.

The first warm control already completed under FIFO on exact base `bdc7ea28c`:
`20261024-140403-ds-arch-parallel-base1`. The HUD appeared 207.047 seconds after
actual launch. With `SHARPEMU_SHADER_PRECOMPILE_BUDGET=0`, overlay enabled, and
the first 40 in-world seconds discarded, it recorded 112 presented-frame entries
over 79.4424 seconds: 1.409826 fps, p50 481.783867 ms, p95 1665.824175 ms,
p99 1815.888167 ms. The FPS convention is entries / first-to-last span; the
interval-count convention is also preserved (1.397239 fps). There were zero
detected build/test process-load alerts in the accepted interval. End screenshot
inspection found the expected Outpost Passage scene, knight, foliage, tunnel,
lighting and overlay, matching the diagnostic scene.

The remaining alternating warm arms are `opt1`, `base2`, `opt2`, with the same
budget, resolution, own cloned data root, overlay, HUD+120-second stop, and
skip40 analysis. Their live FIFO entries are preserved behind the user's
`int3-user-gui` lock. They must not launch while that lock remains held.
No optimization FPS, A/B conclusion or rendering-equivalence claim is available
until those runs complete and their final screenshots have been inspected.


Completed validation outputs `artifacts/bin`, `artifacts/obj` and the isolated
base's corresponding directories were removed after saving the results. There
was no `.packages` directory. Signed game bundles, own data root and evidence
remain available to the queued runs. Their helper saves skip40 summaries after
each successful run; final screenshot inspection and A/B interpretation remain
required.


## First optimized attempt: excluded memory-pressure termination

The user GUI lock was released; `opt1` acquired FIFO at local19:13:31 but the
shared game harness continued waiting for the existing SharpEmu process to exit.
Actual launch was local20:34:37. The run directory timestamp therefore does not
represent launch time: `20261024-191332-ds-arch-parallel-opt1`. The HUD appeared
336.429 seconds after actual launch. At local20:41:02 and20:41:04 the OOM guard
reported 5% free memory twice and killed PID69844. The app exit was137; the wrapper
correctly returned3 because no end marker existed. This attempt is excluded from
A/B timing. Peak footprint was14,418MB and Metal allocation peaked at14,843MB;
these are whole-run peaks rather than matched steady-state differences to base.

The last screenshot (`shot-0376.png`) was opened and compared with base1's end
shot. Knight, tunnel, foliage, lighting and the large bottom-left HUD layout
matched; the Outpost Passage title was still visible after fewer completed
frames. This is not a completed end-window equivalence result. The trace confirms
three native QoS helpers and four-worker batches were active. Near the end, each
flip added about5,300 queued draws and80 parallel batches, with writer, shape,
dynamic-state and pass refusals reported separately. Memory pressure invalidates
using that incomplete attempt to predict normal speedup.

`base2` then rejected its failed predecessor before launching any game. The own
invalid `opt2` suffix waiter was canceled before acquisition, its queue line alone
was removed while another reservation held the queue stable, and fresh
opt1/base2/opt2 jobs were appended in that order. The failed logs/result/guard
evidence are preserved under `bench/arch-parallel/attempts/opt1-oom-20261024-184104Z`.


The second unchanged-binary opt1 attempt
(`20261024-210909-ds-arch-parallel-opt1`) also failed the OOM guard, PID11149,
local21:13:46 after two 5%-free samples. HUD was at246.51 seconds; it survived
only30 seconds in-world, so it never entered skip40. The wrapper returned3,
app137; peaks were14,106MB footprint and14,863MB Metal allocation. All27 process
sampler windows from actual launch through exit had zero build/test activity.
Its last screenshot (`shot-0265.png`) was opened: expected knight/tunnel/foliage
and HUD, with early fog/soul animation and title still visible. It is excluded
from FPS and final rendering-equivalence claims. The archive is
`attempts/opt1-oom-20261024-191346Z`.

Rather than repeatedly launch the failing optimized arm, an exact-main
`basecheck` control is queued under the same current host conditions and settings.
This distinguishes a general memory constraint from prototype-specific growth
before another optimized retry. Existing dependent base2/opt2 jobs retain their
failed-predecessor guards. Sampler logs now start fresh per attempt; older
attempt logs were archived first, and their actual launch timestamps bound
all reported load checks.


## Matched main control and small-batch allocation correction

`basecheck` completed under exact base `bdc7ea28c` with budget0, overlay1 and
build-server cleanup disabled, run`20261024-211738-ds-arch-parallel-basecheck`.
HUD was at289.800 seconds. Skip40 captured117 entries over79.4911 seconds:
1.471863fps, p50 552.483082ms, p95 1752.232075ms, p99 2163.524151ms. Its end
screenshot was opened and showed the expected knight/tunnel/foliage/HUD scene.
Main therefore completed under the current host conditions; the two optimized
terminations cannot simply be dismissed as a host-wide inability to run.

At nearby early-HUD memory samples, main had552MiB of unique resident+swapped
`graphics_iokit` objects versus1,767MiB for the optimized attempt; managed heaps
and Metal-reported allocations were similar. These are nearby scene samples,
not proof of the cause. Code inspection found a concrete avoidable difference:
the original small-batch path ended its current child, created another child for
the batch, then created a tail. About22,000 small batches had already occurred
by the second optimized attempt's termination.

The revised path encodes a small batch into the existing serial child and keeps
it open; full K-worker batches still use ordered distinct children and join
once. A new device test alternates64 singleton render passes, compares all
pixels to serial, and requires64 actual child creations rather than192. A
`native_children` counter records the allocation effect. This correction is
pending a fresh build, focused ordering tests, all four full suites and a new
signed bundle; old validation does not cover the revised source. The original
signed751c2b9cf bundle is preserved at`artifacts/parallel-initial-macos/SharpEmu.app`.
