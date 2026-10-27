<!-- Copyright (C) 2026 SharpEmu Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later -->

# Native occlusion and Silent Hill room measurements

Native Metal visibility counters replace the conservative always-visible query
results when `SHARPEMU_METAL_V2_NATIVE_OCCLUSION=1`. In a paired, same-binary
Silent Hill first-room comparison, median guest FPS increased from 4.485 to
13.43. This does not establish 30 FPS, full-game performance, or performance in
the user's original 2-FPS scene.

## Query behavior

Counting draws accumulate samples in distinct visibility intervals. A GPU prefix
sum preserves the counter value at each dump; ordered GPU copies publish these
snapshots with the guest ready bit. Empty intervals, sparse instance masks,
resets, multiple queues, pending predication, and query-memory reuse retain
stream ordering. GPU-produced values are never prematurely read into the CPU
label replay cache.

Each batch has 4,096 intervals. At most 256 batches are retained (16 MiB for
counts and snapshots). Between passes, exhaustion waits for retired storage or
commits and waits for the current buffer. Unfinished storage is not reused.
The initial experiment aborted on budget exhaustion; that failed run is excluded
from performance results and has a bounded-backpressure regression test.

The latest implementation initializes only exposed slots, caches visibility mode
and offset within each encoder, and skips a reduction only when no counting draw
or referenced snapshot requires it. A disabled pass preserves an existing
nonzero counter. Sparse dumps excluding block zero still publish ready-only
values without requiring a batch snapshot.

## Measured comparison

Apple M4; x64/Rosetta emulator with native arm64 GPU host; Release, self-contained
.NET 10; 1920×1080, scale 1, VSync enabled and existing rendering settings retained.
Each run uses a private copy of the same warm user/cache seed. Other applications
remain open. Host memory pressure and background load are not eliminated.
Only Silent Hill: The Short Message (PPSA10112) was launched.

The phone was closed after visual confirmation. The stationary view faces the
same first-room shelf and lamp. Samples are two-second guest-presentation reports
in the 375–410 second interval after launch. Host presentation Hz is a separate
counter and must not be substituted for guest FPS.

| Build/arm | Reports | Median guest FPS | Report range | Interpretation |
| --- | ---: | ---: | ---: | --- |
| v2, native queries disabled | 18 | 4.485 | 3.50–5.00 | Same-binary control |
| v2, native queries enabled, serial resolution | 18 | 13.43 | 10.45–14.44 | 2.994× paired median ratio |
| v3, command thread and parallel resolution | 18 | 11.455 | 9.46–11.95 | Slower than serial; different code and mode |
| v4, also queue CP writes | 18 | 11.725 | 9.46–13.00 | No established improvement |
| v5, also request native command-thread QoS | 17 | 11.98 | 10.95–13.00 | QoS applied; no established improvement |

These are observational comparisons, not confidence intervals. Only the first
two rows change one switch in the same executable. Their independently reported
draw and frame windows suggest roughly 9,492 versus 3,366 recorded draws per
frame; those ratios are approximate, not exact per-frame accounting. The five
successful runs completed their 480-second limit and controlled shutdown with
exit 0. A post-measurement camera turn in v5 changed the view and rendered new
frames; its approximately 23-FPS wall view is not the stationary benchmark.

## Experimental switches and correctness fixes

The following performance switches remain opt-in:

- `SHARPEMU_METAL_V2_NATIVE_OCCLUSION=1`: native sample counts and ordered snapshots.
- `SHARPEMU_METAL_V2_QUEUE_CP_WRITES=1`: asynchronously enqueue byte and eligible qword CP writes when the
  existing command thread is active; pending-range reads and prefetch gates keep
  consumers behind those writes.
- `SHARPEMU_METAL_V2_COMMAND_THREAD_QOS=1`: create the command producer using the
  existing native helper-thread QoS policy, with managed fallback.
- `SHARPEMU_METAL_V2_REUSE_PRESENTED_FRAME=1`: suppress repeated presentation of an
  unchanged frame while preserving movie/CPU-frame updates, output changes,
  restore handling and animated overlays. Gameplay acceptance remains incomplete.

Overlapping pending-label writes previously left stale cached qword answers.
The corrected shared queue invalidates conflicting cached widths and falls back
to ordered backend reads when a complete answer is unavailable. The same device
predication test failed in both synchronous and queued modes before the fix.

The command-thread coalescing path now waits for ordered producer-idle work or
normal budgets instead of committing merely because the producer's queue has
become empty while a removed slice is still executing. Test drains use a trailing
producer barrier, rather than mistaking queue removal for completion.

APR session teardown now stops admission, cancels queued commands and joins
executing workers before archive unmount. Admitted submission and result writes
hold lifetime leases without holding the global queue lock during guest access.
Stopped root-memory ownership also rejects newly created wrappers. Cached host
files remain alive while readers hold leases, including during cache eviction.
Address waits and chunked reads cooperate with cancellation.

## Validation and unresolved evidence

Latest source validation, before the v6 runtime publication:

- Full Libs: 8,971 passed, 0 failed, 5 existing Opus skips, 1m18s.
- Full Metal, default mode with required device: 1,606 passed, 0 failed,
  7 existing skips, 1m24s.
- Native-query device cases with command thread, QoS, parallel resolution and
  queued CP writes: 13 passed, 0 failed, 1s.
- APR suite: 57 passed, 0 failed, 753ms, including five new lifetime cases.
- Command-thread lifecycle tests: four passed with QoS requested and four with
  native QoS disabled. v5 full threaded Metal: 1,604 passed, 0 failed, 7 skips.

The first new admission test waited for a guest read that cached capture never
performed. It failed while 56 APR tests passed. The corrected test parks the
actual admitted submission-ID write; it does not weaken the shutdown assertion.

A presentation-reuse run aborted during startup, before any room measurement:
pixel shader `17174605ABA538FD` loaded a descriptor 72 bytes beyond a mapped guest
region. An earlier unrelated experimental run failed at the same shader offset
and a different mapping boundary. The mapping failure remains unresolved; no
causal attribution to presentation reuse is established. Failure-time user-data
logging now preserves the actual failing draw's inputs. The later archive-close
exception exposed the separately repaired APR ownership race.

An artifact-only single-thread compute copy experiment reduced CPU encoding time
but increased GPU time. At 128 tiny copies, CPU encoding changed from
62.5–65.2 µs to 9–10.8 µs while GPU time rose from 10.6–11.1 µs to 26.6–27.1 µs.
It is not integrated and is not an FPS improvement.

## CPU lookup follow-up

A second v7 diagnostic run used CPU phase timing only, disabled native automatic
sampling, and completed its 600-second bound cleanly. In the fixed 475–510-second
first-room interval, 17 presentation reports had a median of 12.0 FPS (9.96–13.5).
This is an instrumented diagnostic, not a new uninstrumented performance record.
Across 422 reported frames, deferred-label ordering consumed 5.38 ms per frame;
resource materialization consumed 6.79 ms and draw-resource preparation 5.79 ms.

The pending-copy list now has two bounded negative page filters: one for guest
addresses and one for native buffer/offset pairs. A missing page avoids scanning
the list; possible matches retain the exact overlap check. Large or wrapping
ranges force fallback, and draining the list clears both filters. The switch
`SHARPEMU_METAL_V2_LABEL_PAGE_FILTER=0` restores the original lookup. CPU profiling
reports query, rejection, scan, inspected-entry and peak-held-copy counts.

A synthetic ARM64 Release benchmark measured a binding in a gap between two
clusters of held copies. Four alternating trials, 100,000 lookups per trial,
tiering disabled; all measured lookups allocated zero bytes:

| Held copies | Original median | Filter median |
| --- | ---: | ---: |
| 16 | 8.27 ns | 3.81 ns |
| 256 | 142.30 ns | 3.84 ns |
| 4,096 | 1,944.69 ns | 3.78 ns |

A first-entry match costs about 2.2–2.4 ns more with the filter. Same-page gaps
still require the exact scan. Filter insertion and reset cost about 6 ns per
copy in the 4,096-copy case. These are synthetic lookup measurements, not FPS.

Sampler numeric-class preparation also changes from a per-sampler scan of all
sampled-image pairs to one pass over the pairs. Classification is recomputed
from the current resolved image formats on every preparation; mixed, invalid,
and unpaired cases retain their behavior.

Validation: 78 focused device/unit cases passed with the filter enabled, and
78 passed with it disabled and CPU profiling enabled. The full Metal suite
passed 1,617 cases with 7 existing skips. Five render-phase tests passed.
Subsequent opt-in materialization diagnostics passed 66 cache/profile cases,
including moved descriptors and in-place comparison. The v8 CPU-profiled run completed its 600-second bound cleanly. In the same
475–510-second room interval, 17 reports had a median of 13.44 FPS (12.49–13.5).
Deferred-label ordering fell from 5.38 to 0.89 ms per reported frame. The filter
rejected 5,347,658 queries beyond the coarse address-hull check; 625,835 exact
scans remained, with a peak of 16,620 held copies. This is a profiled combined
filter/sampler comparison; an uninstrumented v8 acceptance run remains pending.

## Reproducibility

All runtime publications retain their source snapshot, tracked binary diff,
untracked-source archive and payload hashes in the local diagnostic artifacts.
They share base revision `52f07c7b37570adaad7317b8196a86f9bb4d6548`; they are dirty
source snapshots, not separate clean commits. Tracked source-diff SHA-256:

| Publication | SHA-256 |
| --- | --- |
| v2 | `c436a870becdc70703e5b1c29c399cdf146ab0be11119daebdd96dd9c0b39587` |
| v3 | `138d695e9425b452c1b67625a6a377b8d7989a1bfa8e217d6d276d9047a8db54` |
| v4 | `f24dcf90c9d0d797fd7e0346573d7d685b0013f22c97d4d17cd829eb9104f24a` |
| v5 | `1d7372c56af1bb810d4fd75b2bdee2c6243f9bed3c87c9c2b4f974190b707656` |
| v6 | `6cc88fe55291141199cbec17e62ef80dcbf288fbf1b226049a65db5f187acc65` |
| v7 | `efdf79607bd5046d1ef71b74f0e04f9b39fc934fc964f3d8bb4916f9ea9ce0f4` |
| v8 | `2325d12d7defa5ab7fafcd05c425871163a2062e2eb3123ec6ea563217a30f3a` |

The v6 publication adds the query-work reductions and APR lifetime fix. Its
profiled run stopped cleanly at 497 seconds. Automatic native process sampling
and memory inspection started near 487 seconds, contaminating the planned
475–510-second interval; that interval is excluded from FPS comparisons. A
post-hoc diagnostic interval at 440–475 seconds attributes 12.23 ms per frame
to storage-buffer resolution, the largest exclusive CPU phase. Its sampled
GPU encoder durations sum to 16.98 ms per frame, with 17.01 ms of gaps; these
are not exclusive wall times and cannot be added to CPU phases. This identifies
the next investigation, not an uninstrumented performance improvement. Commercial assets, shader
captures, screenshots and raw game logs are not included in this repository.

### Unprofiled v8 retry limitation

The next unprofiled v8 run failed before gameplay at 59.977 seconds, so it adds
no FPS result. The shader's captured pointer again lay 232 bytes before the
end of the mapped region, with the requested descriptor read 72 bytes beyond
it. The complete mapping snapshot and actual user-data words were retained.
This recurring failure predates the label filter and native-query changes; its
cause remains unresolved. The best qualified unprofiled stationary-room result
remains the v2 measurement above.

### v10 diagnostic and register-map follow-up

The v10 CPU/materialization diagnostic completed its 600-second limit and
controlled shutdown with exit 0. No descriptor failure recurred; its cause is
still unresolved. The matching stationary room at 475–510 seconds produced
17 reports with median 12.95 FPS (10.46–13.0). The phone was closed at 401.6
seconds, later than the prior runs' approximately 380 seconds. Detailed profiling
and bounded pointer-origin history were enabled, so this is not an unprofiled
FPS acceptance result.

Across 458 reported frames, resource materialization cost 7.455 ms/frame.
Sampled snapshot preparation estimates approximately 1.559 ms/frame, versus
0.417 ms/frame for hit validation. The cumulative counter interval contained
1,744,075 normal hits, 197 recalled hits, 1,724,144 rebased hits, and 150,881
recorded ranges. Many hits therefore have no ranges to compare. Sampled and
nested timings are diagnostic estimates and must not be summed as exact
exclusive frame accounting.

The dense shader register map preserves written-zero presence, arbitrary
capture keys, reset behavior and AGC's shared map identity. Full Release Libs
validation passed 8,986 tests with zero failures and five existing Opus skips.
A native ARM64 interpreter ABBA comparison changed only `SharpEmu.Libs.dll`:

| PM4 case | Control ns/draw, two trials | Candidate ns/draw, two trials | Mean change |
| --- | --- | --- | ---: |
| 20 context registers, automatic draw | 615.0 / 639.4 | 592.7 / 615.1 | -3.71% |
| Same stream, individual packet reads | 616.3 / 658.6 | 642.4 / 627.0 | -0.43% |
| Indexed draw | 631.9 / 657.1 | 656.7 / 632.9 | +0.05% |
| Five context registers | 248.0 / 262.1 | 246.3 / 249.7 | -2.76% |
| 25 context registers | 723.7 / 763.7 | 748.3 / 742.3 | +0.22% |

All cases reported 0.1 B/draw. Each case used five seconds of warmup and five
150-ms samples, with tiering disabled. Host drift and two trials per arm limit
inference; these numbers establish no game-FPS gain. A separate register-batch
experiment showed no improvement and was reverted before this comparison.

The subsequent v10 command-thread/serial-resolution run stopped at 238 seconds
because the harness's free-space guard tripped. It produced no stationary-room
result and cannot establish a threading performance regression or deadlock.

### Additional candidate validation

Exact resident comparison now covers recalled and rebased ranges using each
range's existing address calculation. Refusal retains the original reader and
base-word probe path. Empty-range hits avoid the comparer. Successful head
lookup also avoids repeating the identical user-data check; other variants
retain their checks. The materialization suites passed 107 tests both normally
and with in-place verification enabled. A first failing test was corrected to
honor the existing immutable-user-data input contract; its probe assertion was
retained. The synthetic comparisons below do not establish an FPS improvement.

Queued qword writes use a conservative pending destination hull only when its
arithmetic is valid. Disabled and wrapping cases retain synchronous behavior.
Twenty-two focused device tests passed, including actual producer-thread
ordering, sparse destinations, indirect packets and predication.

`SHARPEMU_GPU_RESOURCE_ORIGIN=1` retains at most 128 historical PS s28/s29 writes
and reports their packet/table origins on materialization failure. Correlation
with captured inputs and a source-word reread are diagnostics, not proof of
current ownership or permission to bypass a failed descriptor read. Forty-five
focused provenance/interpreter tests passed with the recorder enabled.

### Resident-comparison benchmark follow-up

Native ARM64 ABBA runs changed only `SharpEmu.ShaderCompiler.dll`; the tool and
all other libraries were byte-identical between arms. Tiering was disabled,
with five seconds of warmup and five 150-ms samples per case. The first run
used a harness that converted a capturing reader delegate on every call:

| Case | Control ns/op (two trials) | Candidate ns/op (two trials) |
| --- | --- | --- |
| Full walk | 1503.8 / 1522.0 | 1490.4 / 1535.6 |
| Warm hit | 130.5 / 130.2 | 122.6 / 122.4 |
| Cold miss | 4066.4 / 4064.7 | 4209.8 / 4046.9 |
| Changed descriptors | 4841.3 / 4846.0 | 4903.3 / 4851.4 |
| Eight returning tables | 231.0 / 242.2 | 156.8 / 159.0 |
| Moving descriptor ring | 389.9 / 402.5 | 313.5 / 317.8 |
| Table-write control | 63.4 / 64.0 | 63.3 / 63.7 |

The tool now retains that delegate, matching the renderer. A repeat confirms
zero allocated bytes per warm hit and returning-table hit in **both** arms;
the earlier 64 B/op was harness overhead, not an emulator allocation saving.
Cold misses report approximately 6315–6326 B/op after this harness correction.
The corrected run's timing was unstable: unchanged full-walk and table-write
controls shifted by approximately 25% and 36% between arm means. Warm hits were
140.2/139.7 ns for control and 161.6/143.6 ns for candidate; returning-table hits
were 379.1/241.0 versus 253.5/202.7 ns. These repeats do not support a reliable
speedup estimate. The comparer uses fixed synthetic native memory, without
Metal ownership checks. Neither run establishes a game-FPS improvement.

### Graphics binding candidate and runtime exclusions

`SHARPEMU_METAL_V2_GRAPHICS_BINDING_REUSE=1` opts into skipping an identical
native argument/push buffer binding within a graphics encoder. Matching includes
physical vertex/mesh/fragment stage, selector, slot, buffer and offset. Encoder
reset clears the cache even if a native pointer is reused. Compute bindings,
resource acquisition, uploads and residency declarations retain their behavior.
Cumulative attempted/skipped counters are printed per physical stage.

Six focused tests passed with zero failures or skips: three memo tests, the
existing binding-reuse scene, and control/candidate device scenes with repeated
and changed textures on one target plus an encoder restart. No real MTL mesh
stage-switch device scene was added; physical-stage isolation has pure-test
coverage. Combined Release validation passed: Metal 1,632/0/7 with binding reuse enabled; Libs 8,992/0/5; ShaderCompiler 2,192/0/1; ShaderCompiler.Metal 3,588/0/0 (passed/failed/skipped). Device suites required a device, and Libs used serial run settings. These four suites total 16,404 passed, zero failed, 13 existing skips. The shutdown-import patch is not applied and is not included in these results. Game measurement of binding reuse remains pending.

The first v11 presentation-reuse run completed 600 seconds and controlled
shutdown, but missed phone dismissal. Its 9.975 FPS median at 475–510 seconds
is from a ceiling/phone view and is excluded from stationary-room comparisons.
Host presentation tracked guest FPS rather than repeatedly presenting at 120 Hz.
A second run stopped near 190 seconds when the disk guard detected less than
1 GiB free. A null-destination memcpy fault followed the shutdown request on the
guest RHI thread; it is a separate shutdown defect, not evidence of a gameplay
crash or a presentation-reuse regression. Neither run changes the best qualified
unprofiled room result of 13.43 FPS. The 30 FPS target remains unmet.
