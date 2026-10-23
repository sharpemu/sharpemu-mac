# Device binding words, follow-up 2

Branch: `ds-fps-binding-words2`, based on `377182a87` (`ds-fps-binding-words`). No game run, push, reviewers, or delegated work.

## Exact failure

The earlier hot MSL dump did not contain either reported consumer. Their original guest code was recovered read-only from the integration worktree's `artifacts/data-root/user/pipeline_cache/PPSA01342/shader-inventory-v1.bin` and decoded offline. No guest code is checked in.

Both provenance plan keys resolve to Compute, wave64, user-data base 0/count 2, excluded registers 0:

| Shader | Plan key | Pending memory | Consumer |
| --- | --- | --- | --- |
| `70E1A4C75F2F7ABB` | `07D4235BBC63CCCE` | indices 44..47, PC `19C`, components 0..3 | PC `258`, `ImageSampleLz`, DIM_3D |
| `4FC0AE1374C850CF` | `197362B3334EDE8A` | indices 157..160, PC `820`, components 0..3 | PC `834`, `ImageSampleLz`, DIM_2D |

The first shader loads a pointer pair at PC `190`: `s_load_dwordx2 s106:s107, [s0:s1 + 0x20]`. PC `19C` then executes `s_load_dwordx4 s20:s23, [s106:s107 + 0xC0]`. These four words are an S# sampler descriptor. PC `258` samples the image in `s0:s7` with the sampler in `s20:s23`. The second shader similarly loads its sampler into `s8:s11` at PC `820`.

With memory index 44 left on device, the original exact exception is:

```
shader resource tracking: hash=0x70E1A4C75F2F7ABB stage=Compute pc=0x00000258 memory (Read) SamplerHandle dword 0 is not a valid runtime value
```

The second shader fails the same check at PC `834`. `RuntimeValueValidator` correctly rejects the deferred scalar word as host-evaluable; `ResourceTracker.GetHandleSource` then requires a host sampler source. Existing runtime buffer/address lowering cannot handle that sampler. `ShaderProgramCache.TryPlan` swallowed the planning exception, leaving only `plan_failed`.

## Implementation

- Recognize device-loaded S# descriptors for plain `ImageSampleLz` with 2D/3D coordinates. Move the complete four-word sampler to the device when any component is learned, including single-word learning mode. Persist the expanded set actually used by the accepted plan.
- Keep a distinct, zero host sampler placeholder for layout bookkeeping. The device emitter consumes the actual SGPR words; no placeholder value supplies sampling state. It applies coordinate/address modes, point/linear filtering, mip selection/interpolation, LOD clamps, and border colours to the bound texture's real texels. Explicit zero LOD needs no derivatives or anisotropic footprint. Indexed custom borders retain the existing host path's transparent-black convention.
- Preserve image identities and all write-range guards. Other sampler instruction shapes retain the original planner check. Unsupported specialized image classes return to synchronized host sampling before work is encoded; the SPIR-V backend explicitly rejects device S# requests rather than consuming placeholders.
- `SHARPEMU_METAL_V2_WAIT_RESOURCE_WORDS=1` still disables device binding loads. No title hash, address, or per-title condition is used by the implementation.
- Include the actual planner failure check, PC and descriptor dword in refusal provenance, without transient scalar-graph allocation IDs.

Implementation pointers: `ResourceTracker.cs:1218` selects the narrow device sampler form; `ShaderResourcePlan.cs:143` and `DeviceBindingLoads.cs:567` expand the sampler word group; `Gen5MslTranslator.ImageSampling.cs:387` reads the SGPR S#; `Templates/device_sampler_prelude.msl:55` and `:98` perform texture-data sampling. `ShaderProgramCache.cs:557`/`:707` restore host plans for incompatible image classes, `:771` persists the actual expanded plan, and `:790` retains the failed planner check. Paths are below `src/SharpEmu.ShaderCompiler*/` or `src/SharpEmu.Libs/Gpu/Pipelines/`, respectively.

## Counters and persistence

`words_replans` reset on every report; a refused word's reason remains attached to subsequent reads. Thus the final integration report's zero replan counts and repeated `reason=plan_failed` were different time scopes. Keep those interval counters, add `words_replans_total`, and count subsequent refused-word reads separately as `words_refused_reads`. A test consumes and checks actual descriptor data across a report reset (`DeviceBindingLoads.cs:697`).

The first integration run was cold for `device-binding-loads-v2-6b3c1d72.txt`. The copied older stores were v1, and raw memory indices are intentionally compiler-build/schema scoped. The next run logged `keys=48 loads=452 refused=29`; the third logged `keys=74 loads=542 refused=59`. The v2 file exists with 121 records, including `07D4235BBC63CCCE - 44` and `197362B3334EDE8A - 157`. Persistence worked within the same compatible build. The seed message now prints its filename and `status=loaded` or `status=cold_build` so this distinction is visible (`src/graphics/SharpEmu.Gpu.Metal/V2/MetalDeviceBindingLoadsStore.cs:92`).

## Verification

Release build, ShaderCompiler and both Metal suites use `/Users/xnetcat/.dotnet-x64/dotnet`, `-c Release -m:2 -nr:false -p:UseSharedCompilation=false`, with `MSBUILDDISABLENODEREUSE=1`. The Metal gates additionally use `SHARPEMU_TEST_REQUIRE_DEVICE=1 SHARPEMU_LOG_MEMORY=1`. Libs uses its normal native-host `dotnet test` settings: its forced device gate requires unavailable Vulkan. The exploratory forced x64 Libs run is retained separately: 1939 missing-Vulkan prerequisites and seven CPU/loader failures outside the sampler path (four SSE4a fault-bridge tests, import-result logging, large red-zone groups and vector-store splitting); neither new binding regression failed. This is not an x64/Vulkan Libs validation claim. Each build/test command waits for `/tmp/ds-fps-gamerun.lock` to disappear; suites run sequentially. Work waited while `integ2-prof` held the lock.

- Release CLI build: passed, 0 errors, 19 warnings on the final rebuild (18 on the first full build).
- 60 device-data cases: 2D/3D, all address modes, point/linear filtering, border blending, nearest/trilinear mip clamps, large finite coordinates, queued/completed GPU producer. Values checked against independent ramp-texture results and native Metal samplers where a native mode exists. Additional cases cover the disable switch and exact unsupported planner check.
- Libs regression checks actual sampler snapshot words for both a first incompatible image class and a later float-to-integer image specialization.
- Existing guest-view tests also exposed a Rosetta fixture alignment bug: anonymous `mmap` could return a 4 KB-aligned base, but guest views round to 16 KB. The fixture now reserves and trims explicitly aligned guest and canonical spans; production guest-view behavior and all byte-content assertions are retained.
- An existing GPU test failed because 64x64 RGBA lossless and lossy textures both occupy one 16 KB allocation. Its fixture is enlarged to 64x1024; every compression, usage and strict smaller-allocation assertion is retained. The focused rerun passes.

Final successful results (all commands exited 0):

| Suite | Host/gate | Passed | Skipped | Failed |
| --- | --- | ---: | ---: | ---: |
| SharpEmu.ShaderCompiler.Tests | x64 | 2160 | 1 | 0 |
| SharpEmu.ShaderCompiler.Metal.Tests | x64, required Metal device | 3424 | 0 | 0 |
| SharpEmu.Gpu.Metal.Tests | x64, required Metal device | 1411 | 7 | 0 |
| SharpEmu.Libs.Tests | native arm64, normal Vulkan prerequisites | 8467 | 114 | 0 |

Compiler skip: absent captured fused program. GPU skips are existing HZB, stencil, line-width, packed-format and depth-precision cases. Libs skips follow its host/platform prerequisites; its optional Vulkan device paths are not verified. The first arm64 Libs test run reported all assertions passing but MSBuild child node 2 exited (`MSB4166`), so the clean command rerun used `--no-build --no-restore -m:1` and exited 0. The final Release rebuild and full native Metal suite also used one MSBuild worker.

Exact saved-code replay: both keys above match the integration log with user count 2/wave64; each of the four component seeds is accepted, expanding to indices 44..47 and 157..160 respectively. Evidence: `artifacts/ds-fps/diagnosis/probe-exact.log`.
 Logs/TRX: `artifacts/ds-fps/bench/binding-words2/`. Build outputs `artifacts/bin` and `artifacts/obj` are removed after verification.

This establishes plan construction and device texture-data correctness. It does not establish in-game wait elimination, rendering parity, or an FPS improvement; the game was not run as instructed.
