# b3-fullwalk — resource materialization

**Implemented and measured; the all-walk 10× target is not consistently demonstrated.** The admitted batched path averages 5.4 µs. The valid exact-int2/candidate profiles show inclusive snapshot means **153.9 → 17.8 µs (8.65×)**, but total materialization means **157.9 → 36.1 µs (4.37×)**. Exclusive worker ResourceMaterialization is **243.8 → 195.9 ms/frame (47.9 ms/frame lower)**. The same-app kill-switch pair shows a larger 16.04× snapshot difference under different host conditions. The four alternating clean runs show **1.1575 → 1.1270 pooled FPS (−2.64%)**; no FPS gain is established. All seven accepted final-run end screenshots were inspected and show the intact scene. One extra profile attempt crashed before HUD and is explicitly excluded below.

Branch `ds-fps-b3-fullwalk`, exact parent `46411e2df57257fd80f9c4e1c910293c809a97f4` (`ds-fps-v3-int2`). Implementation commits: `f35f360587c19a0daabbdb1428dfb7d94b6e52ca` and final app/source **`5dc8a6c0b3cab0778e7e0f5741f14adf3f3a2f61`**. No push or review agents.

## Why full walks miss

The original diagnostic app was exact int2 plus default-off counters, before the optimization. Its saved source patch SHA-256 matches its build metadata: `60f0c5d2c1f2022b160a76608be5255e6f81845638aaeb6d76ce9b291173ce36`. Run `20261024-083459-ds-b3-fullwalk-diag-steady`, HUD+40..120, recorded 151,035 full misses: 96,323 absent keys, 54,712 stale same-key entries, zero unreadable/uncacheable misses. The matching preceding plan/base/state entries classified 95,981 as different-key, 157 first sight, and 185 evicted/unmatched. This run still compiled shaders; its counts diagnose misses, and its FPS is excluded from A/B.

The final diagnostic has finer same-key histograms. Run `20261024-130148-ds-b3-fullwalk-v2-warm`, HUD+40..240, produced these counts before an external memory guard later stopped the warm-up:

| Miss reason | Count | Share |
| --- | ---: | ---: |
| Key not found | 89,101 | 62.29% |
| Same-key entry stale | 53,938 | 37.71% |
| Resident unreadable / uncacheable | 0 / 0 | 0% |

Among not-found cases: 88,615 had a preceding entry for the same plan/base/compute state but a different key; 433 were first context sight; 53 were evicted/unmatched. These are observations of consecutive **successful full-walk entries**, not proof of a write once per game frame.

| Comparison with preceding entry | Walks | Changed words at common guest addresses, mean | Words at new guest addresses, mean |
| --- | ---: | ---: | ---: |
| Same key | 53,938 | 1.855 | 24.555 |
| Different key, same plan/base/state | 88,615 | 0.138 | 43.506 |

**38,908 / 53,938 same-key walks (72.13%) changed no words at common addresses but read words at new addresses.** The other 15,030 changed common-address words. Frequent `(changed, new, count)` same-key buckets: `(0,10,8892)`, `(0,14,7687)`, `(0,26,6990)`, `(0,37,4855)`, `(7,46,2459)`, `(0,30,2406)`, `(6,46,2345)`, `(10,0,1034)`, `(2,0,1029)`, `(1,0,959)`. New addresses still require fresh reads even when old addresses agree.

Logical same-key output differences average **10.137 table words, 5.182 buffer words, 0.803 image words, 0.00171 sampler words per full walk**. These overlap physical dependencies and must not be summed as unique guest bytes. Across all comparable entries, 3,568,708 guest words were compared at common addresses and 5,179,829 were at new addresses; 67,674 changed common-address words were table-only, 44,926 descriptor/mixed. Same-key contributions were 58,472 and 41,581 respectively.

Examples locating the rewrites (shader hashes are diagnostic identifiers only): vertex `9F3F16506CF5F9CD` table word 2 changed 5,698 times; buffer 1/2 word 0 and table 10/14 changed 5,544 times. Vertex `A948274AB1658049` buffer 1 word 0/2 and table 10/12 changed 5,496 times. Compute `4A329F97CC70A9DD` buffer 2/3 word 0 and table 10/16/20 changed 4,542 times. Buffer base/record changes and their dependent table words are frequent; the counters do not identify the CPU/GPU writer or establish per-frame periodicity.

Specialization stayed equal in **136,012** comparable walks and changed in **6,594** (4.62%); same-key split 48,206 equal / 5,732 changed, different-key split 87,753 / 862. Specialization change is an overlapping output observation, not a separate additive cache-miss reason. The resident comparer refused labels 1,158 times, but the ordinary reader can apply overlays: these are **not** 1,158 full misses. Ownership/alias/invalid/released refusals and true unreadable misses were zero in this window. Detail overflow was zero.

Evidence: `../bench/b3-fullwalk/diagnostic-source.patch`, `diag-build-info.json`, `diagnostic-steady-summary.json`, **`v2-diagnostic-prefix.json`**, `diagnostic-window.py`. Full per-shader/table/descriptor-word counters remain in the raw `game.tlog`.

## Making the walk cheaper

| Code location | Change and invariant |
| --- | --- |
| `src/SharpEmu.ShaderCompiler/Resources/BatchedResourceWalk.cs:59` | Decode a supported plain plan once into ordered scalar operations with integer operand/output indices, constants and offsets. Plan publication is synchronized in `ShaderResourcePlan.cs:49`. |
| `BatchedResourceWalk.cs:148` | Group contiguous reads with one handle/policy across total ALU only. Endpoint checks prove intermediate addresses; input accesses, different handles and potentially failing operations end a group. |
| `BatchedResourceWalk.cs:200` | Lease thread-local capacity and reset all values each use. Every guest memory instruction is evaluated fresh. Immutable output arrays retain independent lifetimes. |
| `BatchedResourceWalk.cs:259` | One resident span resolve/overlay/label check per admitted range, including single words. Refusal executes the original scalar reader at the original position; failure is final and does not replay synchronizing reads. |
| `BatchedResourceWalk.cs:336`, `src/graphics/SharpEmu.Gpu.Metal/V2/MetalRenderHost.Pipelines.cs:1950` | Range-clear proof skips per-word pending-data probes. A possible writer retains precise learning/probe order. Admitted resident data keeps the existing `RawReadPrefetch` behavior: read first, then learn pending device slots; it does not introduce a scalar wait. |
| `MetalRenderHost.Pipelines.cs:1959` | Reuse the existing completion-floor hint for pending-data queries. A zero with the older floor proves no pending writer; a possible writer is confirmed against the real serial. Existing serial-hint opt-out remains honored. |
| `src/SharpEmu.ShaderCompiler/Resources/ResourceMaterializationCache.cs:31`, `ResourceMaterializationCache.Slots.cs:472` | Typed span copies replace reflective `Array.Clone`; required immutable copies remain. |
| `src/SharpEmu.ShaderCompiler/Resources/IndirectSelectorValues.cs:29` | For conservatively bounded total selector sets, evaluate every runtime leaf fresh, then reuse only the pure set calculation if all leaf values agree. Immutable one-entry publication is thread safe. Unproved shapes and diagnostic capture keep the original path. |
| `ResourceMaterializationCache.FullWalk.cs:19` | Default-off comparison uses already-recorded bytes and outputs, adding no diagnostic guest read or wait. Same-key and changing-key differences are reported separately. |
| `ResourceMaterializationProfile.cs:70`, `src/SharpEmu.Libs/Gpu/Pipelines/ShaderProgramCache.cs:491` | Default-off inclusive timers distinguish materialization inside the render worker's existing profiling scopes from other calls. |

Structured/first-lane/specialization-memory plans retain the original evaluator. The optimization does not reuse prior guest payloads or add a new dependency/invalidation system. Existing ownership, labels, CP overlays, clean-read policy, specialization and descriptor finality remain authoritative. No shader translation, Metal API level or title-specific conditions changed.

Final diagnostic application counters: **115,754 batched walks / 143,039 full misses (80.92%)**, 1,542,921 resident groups for 5,960,878 words; zero refused spans, scalar reads or unsupported-shape fallbacks. Structured fallback: 27,285. Range-clear proofs skipped 1,915,824 pending-word probes; 332 admitted groups retained pending probes. Completion-hint queries: 1,282,952. Typed copies: 342,027. Selector reuse: only 76 hits, zero eligible misses; most selector shapes remain conservative, so this is not a broad selector-speedup claim.

## Switches and log checks

- **`SHARPEMU_RESOURCE_FULLWALK=0`** disables the decoded walk, typed copies, selector memo and pending-data completion hints. The optimization defaults on.
- **`SHARPEMU_RESOURCE_FULLWALK_DETAIL=1`** enables expensive miss/word diagnostics; default off. Periodic output uses the existing render report.
- `SHARPEMU_PROFILE_RENDER=1 SHARPEMU_PROFILE_RENDER_REPORT_S=5` emits exclusive worker phases. Adding `SHARPEMU_PROFILE_MATERIALIZATION=1` emits cumulative inclusive materialization timers. These timer systems overlap and must not be added together.
- `[PERF][RESOURCE_CACHE]`: `fullwalk_enabled`, `fullwalk_walks/groups/words`, `fullwalk_span_refused`, `fullwalk_scalar`, `fullwalk_fallback_structure/shape`, `fullwalk_pending_fallback/skipped/resident`, `fullwalk_selector_hits/misses`, `fullwalk_typed_copies`. `[METAL-V2][WAITS]`: `fullwalk_pending_hint_queries`.
- Detail: `[PERF][RESOURCE_FULLWALK_DETAIL]` and `_OVERFLOW`. Worker-only cumulative timers: `[PERF][RESOURCE_MATERIALIZATION_WORKER]`. Fine-grained resident/pending/scalar timers sample 1/64; their sampled time is not extrapolated into a frame total.

## Build and tests

Final source `5dc8a6c0b3` was built with:

```sh
~/.dotnet-x64/dotnet build SharpEmu.slnx -c Release \
  -p:RestorePackagesPath=$HOME/.nuget/sharpemu-shared \
  -nodeReuse:false -p:UseSharedCompilation=false
```

**0 errors, 226 warnings.** All four suites reran serially through FIFO acquisitions, x64 SDK 10.0.103, shared NuGet. Each used `dotnet test tests/<suite> -c Release --no-build --blame-hang-timeout 120s --blame-hang-dump-type mini`, a TRX logger and a private results directory. ShaderCompiler also used `--settings /Users/xnetcat/Projects/sharpemu/sharpemu-mac/artifacts/ds-fps/shader-serial.runsettings`. `SHARPEMU_TEST_REQUIRE_DEVICE=1` was scoped to Metal and ShaderCompiler.Metal only.

| Suite | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| SharpEmu.Gpu.Metal.Tests | 1,475 | 1 | 7 |
| SharpEmu.Libs.Tests | 8,639 | 6 | 5 |
| SharpEmu.ShaderCompiler.Tests | 2,173 | 0 | 1 |
| SharpEmu.ShaderCompiler.Metal.Tests | 3,331 | 0 | 0 |
| **Total** | **15,618** | **7** | **13** |

All **20 added cases** passed: fresh/moving/nested descriptor reads, output lifetime, ALU grouping and failure boundaries, isolated words, clean/resident refusal, pending resident prefetch equivalence, scalar probe order, fresh selector leaves after reuse and failure traversal, cache specialization changes, Metal completion and unbounded writers.

Known failures (unchanged fixture blobs match base; no assertions or filters changed):

- Metal `MetalLossyTextureTests.ASampledImageIsExactByDefaultAndLossyWithTheFlagOn`: expected lower lossy allocation; both 64×64 allocations are 16,384 bytes.
- Libs `ImportResultLoggingTests.RepeatedImportFailuresAreSampledPerImportAndResult`: incompatible delegate signature (`ArgumentException`).
- Libs `RosettaVectorStorePatchTests.RosettaSplitsStoresWithoutRedZoneUseAndPreservesAllBytes`: expected 2, actual 1.
- Libs `Sse4aPosixSignalRecoveryTests`: `ExtrqSigillRoundTripsXmmThroughTheBridge`, `InsertqSigillReadsSourceXmmThroughTheBridge`, `InsertInstructionPreservesOtherVectorRegisters`, `ExtractSignalPreservesLivePointerAndOtherVectorRegisters`: `NullReferenceException`.

These seven names/messages also failed in the earlier candidate suite run and are documented base failures. This task did **not** independently rerun exact-base failing-test controls. Evidence: `v2-suite-summary.json`, `v2-test-*/*.trx`, `v2-*-retry.log`, `unchanged-failing-test-blobs.json`, `v2-tested-source-manifest.json`. The tested manifest matches the final clean commit; `v2-tested-diagnostic-binary-check.json` checks absence of a transient reverted diagnostic edit from the tested compiler binary. No aborted/gated test attempt is counted as a pass.

## Runtime method and provenance

Both private roots were cloned from the main `artifacts/data-root` using **`cp -cR`**. The exact control app was built once in sibling worktree `/Users/xnetcat/.t3/worktrees/sharpemu-mac/ds-fps-b3-fullwalk-int2`, detached at clean `46411e2df`; the int1 app was not reused. The candidate app was rebuilt from clean `5dc8a6c0b3`. Both used **`SHARPEMU_MAC_GET_TASK_ALLOW=1 scripts/macos/build-app.sh`**, with `codesign` confirming get-task-allow. Build metadata is retained as `int2-build-info.json` / `v2-candidate-build-info.json` and per-run copies.

One build/test/run per shared FIFO acquisition: queue tag → matching `.next` → atomic exclusive `.lock` create → remove `.next` → run → release. No other agent's processes/lock entries were changed. The external memory guard remained enabled. Game command shape:

```sh
SHARPEMU_APP_DIR=<own-app>/Contents/MacOS SHARPEMU_OVERLAY=1 \
SHARPEMU_DS_INWORLD_SECONDS=120 \
bash scripts/macos/games/ds.sh <tag> 900 ingame \
  SHARPEMU_SHADER_PRECOMPILE_BUDGET=0 [profile flags]
```

Golden save, 1080p Performance mode, overlay on, stationary Outpost Passage. Each accepted run reaches HUD, idles 120 seconds, captures its end shot and stops at the harness marker. Measurements select **HUD+40 through end**. FPS is `(N−1)/(last−first flip timestamp)`; p50/p95 are empirical frame-interval percentiles in that window. `legacyFps=N/span` is also retained in JSON. Phase windows must lie wholly inside skip40; inclusive cumulative timers use first-to-last reports inside it. Their denominators/windows differ slightly from exclusive phases.

The int2 translator has a different shader namespace from base. Initial cold control/diagnostic runs, and a repeat still compiling heavily, were warm-ups and excluded. The final candidate's 300-second diagnostic warm-up reached approximately HUD+290 and was stopped by the external guard after two free-memory readings of 5%, ten seconds before planned end. Its intact last screenshot was inspected; last five performance reports had no compiles, inflight work or skipped compiling draws. That prefix is valid diagnostic/warming evidence **only**, never an accepted FPS arm. `v2-warmup-evidence.json` and `v2-warm-guard-evidence.log` preserve this failure explicitly.

All measured arms are retained after the warming protocol, including incidental streamed permutations. “Warm” does not mean guaranteed zero compilation; compile counts/time are reported below rather than filtering slow arms. Diagnostic-detail timing and managed tracing are excluded from clean FPS.

## Same-app enabled / disabled worker profiles

| Inclusive worker timer | Kill switch off µs/call | Enabled µs/call | Off/on ratio |
| --- | --- | --- | --- |
| Snapshot | 285.381 | 17.796 | 16.04× |
| Total | 295.714 | 36.138 | 8.18× |
| BatchedWalk | — | 5.417 | — |
| DescriptorWords | 70.748 | 33.636 | 2.10× |
| TableEvaluation | 206.691 | 24.531 | 8.43× |
| DeviceAddressRanges | 2.092 | 6.574 | 0.32× |
| Specialization | 7.411 | 9.802 | 0.76× |
| EntryBuild | 30.109 | 9.212 | 3.27× |

The pair has 108 / 60 wholly-contained exclusive phase frames (enabled / disabled). Inclusive snapshot counts normalize to 1,669.24 / 1,686.53 full calls per flip. Exclusive ResourceMaterialization is 195.930 / 416.142 ms/frame. Enabled profile had one incidental compile (17.9 ms); disabled had none. End screenshots were inspected and show the same intact arch, vegetation, armor and HUD, with expected animation/ghost differences.

Snapshot improvement in this pair is approximately 16.04×. Relative to the earlier report's 89.6 µs snapshot mean, 17.8 µs is approximately 5.0×, so the pair does not establish a universal 10× improvement across host conditions. Both runs have substantial swapping (median 3,233 / 3,324 MiB unique swapped memory); GC pause shares are 11.43% / 9.71%. The absolute disabled cost is much higher than the historical report. The exact-int2 profile below is the primary separate-build comparison.

## Alternating exact-int2 / candidate FPS

| Arm | Build | FPS | p50 ms | p95 ms | Flips | Compiles started/finished, time |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | int2-1 | 0.7660 | 1140.0 | 3933.7 | 61 | 0/0, 0.0 ms |
| B1 | candidate-1 | 1.5246 | 526.0 | 1286.2 | 122 | 1/0, 0.0 ms |
| A2 | int2-2 | 1.5504 | 493.1 | 1043.7 | 122 | 34/34, 184.2 ms |
| B2 | candidate-2 | 0.7254 | 1296.4 | 1761.1 | 58 | 0/0, 0.0 ms |

| Pooled arm | FPS | p50 ms | p95 ms | Intervals / span |
| --- | --- | --- | --- | --- |
| int2 | 1.1575 | 597.3 | 1660.0 | 181 / 156.375 s |
| candidate | 1.1270 | 589.0 | 1709.4 | 178 / 157.947 s |

Observed pooled change: **FPS -2.64%, p50 -1.39%, p95 +2.98%**. Both arms vary by about 2× between runs. This cohort does **not establish an FPS gain**. The slow arms were retained; shader activity does not explain the spread by itself. Pooling sums within-run intervals and durations; it never adds the gap between runs. These are two runs per arm, not a confidence bound or a causal attribution of the small pooled difference.

## Exact-int2 / candidate worker phase table

| Exclusive worker phase | Int2 ms/frame | Candidate ms/frame | Candidate minus int2 |
| --- | --- | --- | --- |
| ResourceMaterialization | 243.789 | 195.930 | -47.859 |
| DrawResourcePreparation | 61.393 | 62.099 | +0.706 |
| CommandStream | 48.587 | 48.397 | -0.190 |
| ProgramPrepareLocked | 49.601 | 25.152 | -24.449 |
| GraphicsStateApply | 20.315 | 16.684 | -3.630 |
| DrawTargetResolution | 16.621 | 14.274 | -2.347 |
| BindingTextureAcquire | 14.774 | 12.730 | -2.044 |
| DrawAttachmentPreparation | 13.958 | 12.173 | -1.785 |
| ArgumentBufferWrite | 13.673 | 10.404 | -3.270 |
| ProgramSourceRead | 11.536 | 10.158 | -1.378 |
| ResourceBinding | 9.219 | 8.096 | -1.123 |
| GuestWindowResolve | 7.672 | 6.334 | -1.338 |
| CommandGpuWait | 158.513 | 3.162 | -155.351 |
| Idle | 13.946 | 175.960 | +162.014 |
| Other measured phases | 170.259 | 112.070 | -58.189 |

Wholly contained phase windows: int2 15 reports / 91 frames; candidate 15 reports / 108 frames. Phase sums, including idle: 853.857 / 713.623 ms/frame.

The candidate profile arm is the earlier **`b3-fullwalk-v2-probe`** run. It used the same clean final app, warm data root, `SHARPEMU_PROFILE_RENDER=1 SHARPEMU_PROFILE_RENDER_REPORT_S=5 SHARPEMU_PROFILE_MATERIALIZATION=1`, detail off, HUD+120 stop and skip40 window as the exact-int2 profile. It ended normally and its end shot was inspected. The profiles are separated in time; the four clean FPS runs remain the alternating A1/B1/A2/B2 sequence above.

| Inclusive timer | Int2 global µs/call | Candidate global µs/call |
| --- | ---: | ---: |
| Total | 157.872 | 36.138 |
| Snapshot | 153.883 | 17.796 |
| DeviceAddressRanges | 1.006 | 6.574 |
| Specialization | 2.498 | 9.802 |
| EntryBuild | 4.223 | 9.212 |

The candidate worker-only means equal these global means in the selected window. Int2 lacks the added worker-context counters, so its inclusive figures are explicitly global; the exclusive worker phase table above remains the direct worker-budget measurement. Full calls per flip: int2 1,674.16 / candidate 1,669.24. The candidate's admitted `BatchedWalk` averages **5.417 µs**, 1,309.35 calls/flip. Remaining fallback descriptor/table evaluation averages 33.636 / 24.531 µs per corresponding call; these are a different subset from int2's all-plan descriptor/table means and must not be treated as identical populations.

The candidate profile counters recorded 153,519 batched walks / 195,627 misses (78.48%), 2,005,966 span groups / 7,756,522 words, zero span refusals or scalar fallbacks, 42,108 structured fallbacks, 2,471,624 skipped pending-word probes and 15 admitted groups retaining probes. The kill-switch profile has zero batched walks/hint queries, confirming the opt-out applies. Counter report windows can straddle skip40 edges; cumulative timer deltas use their own contained first/last reports.

Profile FPS (not included in the clean pool): int2 1.1819, candidate 1.4115. Profile compilation: int2 4 started / 3 finished, 149.4 ms; candidate 1/1, 17.9 ms. The candidate profile was earlier in time; the phase differences and the clean-run spread do not establish a stable FPS gain.

The additional later candidate profile attempt `20261024-150745-ds-b3-fullwalk-final-candidate-profile` **failed before HUD**, native exit 139 at ~138 seconds. The harness wrapper itself returned 0, but the normal-HUD/end-marker gate correctly rejected it. It has no scene/FPS result. Guest CPU thread `BPE JobWorkerThread CPU4` read address `0x10` at RIP `0x800B03037`. Nearby recorded-writeback release warnings (serial 10303 vs completed 10302, 250 ms) do not establish a cause. No independent base reproduction or causal attribution to this change is claimed. The failed raw run and `final-profile-crash.json` are retained. The already-completed enabled profile supplies the required candidate profile arm; no failed run was relabeled as successful.

| Clean arm | GC pause % | Median swapped MiB | Median footprint MiB |
| --- | ---: | ---: | ---: |
| b3-fullwalk-final-int2-1 | 9.16 | 3292 | 13349 |
| b3-fullwalk-final-candidate-1 | 13.12 | 2812 | 13488 |
| b3-fullwalk-final-int2-2 | 13.33 | 3101 | 13058 |
| b3-fullwalk-final-candidate-2 | 6.45 | 3288 | 13545 |

Memory/GC windows use available cumulative telemetry rows within skip40 and are slightly shorter than the FPS interval. Faster runs allocate more per second; these counters do not prove that GC or swapping caused any particular arm to be slow.

## Rendering and evidence

All **seven** accepted final-run end shots were opened and inspected: enabled profile (`v2-probe`), kill-switch profile (`v2-off`), A1/B1/A2/B2, and the exact-int2 profile. The arch, bright exterior, vegetation, armor materials and HUD remain intact; animation/ghost differences prevent a pixel-identical comparison. No obvious missing texture, geometry or lighting regression was observed. This is standing-scene visual evidence, not exhaustive rendering parity. `final-screenshot-inspection.json` records each actual path and observation. The crashed attempt has no end screenshot and is not included in this count.

| Tag | Raw run directory |
| --- | --- |
| b3-fullwalk-v2-probe | /Users/xnetcat/.t3/worktrees/sharpemu-mac/ds-fps-b3-fullwalk/artifacts/runs/20261024-141112-ds-b3-fullwalk-v2-probe |
| b3-fullwalk-v2-off | /Users/xnetcat/.t3/worktrees/sharpemu-mac/ds-fps-b3-fullwalk/artifacts/runs/20261024-141758-ds-b3-fullwalk-v2-off |
| b3-fullwalk-final-int2-1 | /Users/xnetcat/.t3/worktrees/sharpemu-mac/ds-fps-b3-fullwalk-int2/artifacts/runs/20261024-142619-ds-b3-fullwalk-final-int2-1 |
| b3-fullwalk-final-candidate-1 | /Users/xnetcat/.t3/worktrees/sharpemu-mac/ds-fps-b3-fullwalk/artifacts/runs/20261024-143642-ds-b3-fullwalk-final-candidate-1 |
| b3-fullwalk-final-int2-2 | /Users/xnetcat/.t3/worktrees/sharpemu-mac/ds-fps-b3-fullwalk-int2/artifacts/runs/20261024-144359-ds-b3-fullwalk-final-int2-2 |
| b3-fullwalk-final-candidate-2 | /Users/xnetcat/.t3/worktrees/sharpemu-mac/ds-fps-b3-fullwalk/artifacts/runs/20261024-145008-ds-b3-fullwalk-final-candidate-2 |
| b3-fullwalk-final-int2-profile | /Users/xnetcat/.t3/worktrees/sharpemu-mac/ds-fps-b3-fullwalk-int2/artifacts/runs/20261024-150004-ds-b3-fullwalk-final-int2-profile |

Raw run directories contain source/build provenance, `game.tlog`, `frame-times.txt`, markers, end screenshots, memory and harness result. Compact summaries and extraction scripts are under `../bench/b3-fullwalk/`: `final-results.json`, `final-memory-windows.json`, `final-screenshot-inspection.json`, `summarize.py`, `final-results.py`, `final-memory.py`. These local raw artifacts are retained rather than all committed.

The initial `f35f360` prototype was also measured: pooled clean FPS 1.6767 int2 / 1.7383 candidate (+3.67%), pooled p95 1,199.4 / 1,237.8 ms. Its inclusive snapshot mean did not improve (96.46 / 96.81 µs), and its profile candidate compiled 6.346 seconds of shaders. A subsequent safepoint-biased managed trace found selector evaluation and reflective array copies; its wall samples are not CPU self-time or contention measurements. The final revision also corrected the prototype's unnecessary synchronizing scalar fallback for admitted pending resident data. Prototype numbers, trace and source app are retained but are not substituted for final results.

## Expected savings, risks and remaining ideas

The direct observed worker-budget difference is **47.859 ms/frame lower exclusive ResourceMaterialization (19.63%)** in the time-separated profile comparison. The same-app enabled/disabled pair differs by **220.212 ms/frame** in that phase; its disabled snapshot cost is much larger than both the historical report and the exact-int2 profile. These values are observations, not a promise of a fixed saving. Fewer table/window checks, fewer native completion queries, predecoded scalar operations and cheaper typed copies explain the work removed; they do not explain every scheduling/wait/GC difference between runs.

The main limit is the remaining walk population and surrounding work: total inclusive materialization improves 4.37× in the primary profile comparison, while device-address-range, specialization and entry-build means increase in that sample. Snapshot improves 8.65× overall and the admitted scalar batch averages 5.4 µs; this does **not** demonstrate a 10× reduction for every full walk or a clean FPS improvement. `CommandGpuWait` decreases 155.4 ms/frame while idle increases 162.0 ms/frame between the profiles, so those phase changes must not be attributed wholesale to this patch. This branch is a measured worker optimization and diagnostic result, not a proven end-to-end FPS win.

Direct GPU shader-time savings are expected to be **zero**: the change removes worker-side evaluation/check/copy work. The CPU can feed the GPU sooner, but no GPU-duration reduction is inferred from FPS or worker phases. Pending-writer/label/clean refusals must still retain conservative base behavior.

Coverage is intentionally conservative. Structured and unsupported plans remain on the old evaluator; selector memo coverage was tiny in the diagnostic. Required independent descriptor/table/entry arrays still allocate, and sampled runs show meaningful GC and swapping. The evidence covers the standing scene, not a traversal or exhaustive visual proof. Future ideas: share repeated first-lane evaluator contexts within one walk after proving mask/clean-context identity; predecode more finite selector-set arithmetic; extend compiled coverage to structured plans while preserving branch/read/failure ordering. These are suggestions, not implemented results.

## Cleanup

Removed this worktree's `artifacts/bin` and `artifacts/obj`. The exact-int2 sibling's corresponding directories were already absent. A recursive inventory found no `.packages` directories in either owned worktree. `cleanup-manifest.json` records inspected paths and verifies absence afterward. No own game/queue helper remains active, and no own lock remains. Source/app metadata, raw runs, trace, screenshots and private data roots are retained. No push.
