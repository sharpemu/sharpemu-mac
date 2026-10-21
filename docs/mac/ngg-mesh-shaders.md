<!-- Copyright (C) 2026 SharpEmu Emulator Project -->
<!-- SPDX-License-Identifier: GPL-2.0-or-later -->

# NGG primitive shaders as Metal mesh shaders

Status: accepted 2026-10-10. The translator and the Metal v2 render host both implement this
contract, and v2 enables mesh pipelines by default on a Metal 3 device.
`SHARPEMU_METAL_MESH_SHADERS=0` opts out and restores the per-vertex path for comparison.
The legacy Metal host (`SHARPEMU_METAL_LEGACY=1`) has no mesh pipeline path and never
reports the capability.

## Why

A Gen5 (RDNA2) primitive shader is the merged ES/GS hardware stage. It runs as a
**workgroup**: its waves write vertex data into LDS, synchronize with `S_BARRIER`, reserve
export space with `S_SENDMSG GS_ALLOC_REQ`, and then export positions (`pos0..pos4`),
parameters (`param0..param31`) and primitive connectivity (EXP target 20). A per-vertex
Metal vertex function has none of that: one logical lane, private LDS, no barrier, and a
vertex count fixed by the draw. That is why only *passthrough* primitive shaders ever
worked, and why the executor skipped the rest as "unsupported geometry".

`Gen5MslTranslator` now emits a real culling primitive shader as an MSL `[[mesh]]`
function. The mapping is one-to-one:

| Hardware | Metal |
| --- | --- |
| NGG subgroup | one threadgroup / one mesh grid element |
| wave lane | one thread, on a real simdgroup lane (EXEC and ballots stay exact) |
| LDS | `threadgroup` memory |
| `S_BARRIER` | a workgroup rendezvous in the dispatcher loop (see below) |
| `GS_ALLOC_REQ` | `mesh.set_primitive_count` |
| `EXP pos0` / `EXP param*` | `mesh.set_vertex(thread_index_in_threadgroup, …)` |
| `EXP` target 20 | `mesh.set_index` for this thread's primitive |

There is no `[[object]]` stage. The grid is driven from the host: one threadgroup per NGG
subgroup.

## Selecting the path

`Gen5MslTranslator.RequiresMeshEmission(request)` is true when **both** hold:

1. The request carries `ShaderCompileRequest.PrimitiveShader`. `ShaderPipelineCache` fills
   it only when the backend reports `SupportsPrimitiveShaderMeshPipelines`, the draw has no
   separate geometry program, and `VgtShaderStages.SelectsPrimitiveShaderOnly` accepts
   `VGT_SHADER_STAGES_EN` — the same reading `RenderExecutor` uses to admit the draw, so the
   two cannot disagree. A zero mask is the plain vertex/pixel path and is excluded.
2. `ShaderCompileRequest.RequiresPrimitiveShaderWorkgroup(program)` — the program uses
   LDS (any non-GDS `DS_*` other than the swizzle/bpermute pair) or `S_BARRIER`.

`GS_ALLOC_REQ` and the primitive export are deliberately **not** part of the test: every
NGG program issues them, passthrough ones included, and those already work on the
per-vertex path. Only cross-lane LDS and the barrier are things that path provably cannot
express.

The compiled result then has `Stage == Gen5MslStage.Mesh`, `EntryPoint == "gen5_ms"`, and a
non-null `MeshInfo`.

## Fragment barycentrics and per-primitive vertex values

The pixel shader's `V_INTERP_MOV` path receives three copies of each requested parameter
as per-primitive data. The mesh also hands over `float3 sharpemu_vertex_w`, containing the
three vertices' clip W values in primitive index order. Reciprocal-W guest exports are
converted to clip W before this metadata is stored. The output-size bound includes its
16-byte aligned storage.

Apple's [Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
sections 5.2.3.4 and 5.4, restricts `barycentric_coord` to `center_perspective` or
`center_no_perspective`. Ordinary user interpolants can still use sample and centroid
qualifiers. MSL 3.2 also rejects two barycentric inputs, even with different perspective
qualifiers; the native compiler tests cover this separately from the MSL 4.0
`vertex_value` path.

When a per-primitive pixel shader needs both modes, it receives one linear barycentric
input and reconstructs perspective weights as `p_i = (l_i / w_i) / sum(l_j / w_j)`.
Sample-only, centroid-only and mixed guest barycentric VGPRs use pixel-center weights.
This avoids invalid MSL but remains an approximation for multisampled edges; it does not
implement exact sample/centroid evaluation. The native tests compile both shader paths
and read back perspective and linear weights with unequal clip W values.

## What the host needs: `Gen5MslMeshInfo`

Everything needed to build the pipeline and launch the draw is on the compiled shader:

| Field | Use |
| --- | --- |
| `ThreadsPerThreadgroup` | `threadsPerMeshThreadgroup` of `drawMeshThreadgroups`, and `MTLMeshRenderPipelineDescriptor.maxTotalThreadsPerMeshThreadgroup`. Always a multiple of 32 (one guest wave32 per simdgroup), at most 256. Also mirrored in `Gen5MslShader.ThreadgroupSizeX`. |
| `WavesPerThreadgroup` | `ThreadsPerThreadgroup / 32`; how many merged waves one subgroup holds. |
| `MaxVertices`, `MaxPrimitives` | The `mesh<>` extents the function declares. The host must not launch a subgroup that needs more. |
| `VerticesPerPrimitive` | 1, 2 or 3, from `VGT_GS_OUT_PRIM_TYPE`. Indices per primitive. |
| `Topology` | `Point` / `Line` / `Triangle`, matching the declared `topology::*`. |
| `ThreadgroupMemoryBytes` | Guest LDS the mesh function declares. Counts against the mesh stage's threadgroup budget together with the output payload. |
| `DrawInfoBufferIndex` (2) | Where to bind the `Gen5MeshDrawInfo` block (below). |
| `IndexBufferIndex` (3) | Where to bind the draw's index buffer as a raw dword stream. |
| `ReadsIndexBuffer` | Currently always true: the slot must be bound even for an automatic draw. |

Buffer slots 0 and 1 keep their existing meaning for every stage:
`Gen5MslArgumentLayout.ResourcesBufferIndex` (0) is the stage argument buffer and
`PushDataBufferIndex` (1) is push data. Bind them to the **mesh** stage
(`setMeshBuffer:offset:atIndex:`), not the vertex stage.

## Pipeline creation

```objc
MTLMeshRenderPipelineDescriptor *d = [MTLMeshRenderPipelineDescriptor new];
d.meshFunction     = gen5_ms;                 // from the translated library
d.fragmentFunction = gen5_ps;                 // the existing pixel translation, unchanged
d.maxTotalThreadsPerMeshThreadgroup = mesh.ThreadsPerThreadgroup;
d.colorAttachments[i].pixelFormat = …;        // same as the vertex path
// no objectFunction, no payloadMemoryLength
id<MTLRenderPipelineState> p =
    [device newRenderPipelineStateWithMeshDescriptor:d options:0 reflection:nil error:&e];
```

**No `MTLVertexDescriptor`.** A mesh function has no `[[stage_in]]`, so the host binds no
vertex buffers for a mesh draw and `request.VertexInputs` is always empty on this path.

The ES part fetches its own attributes with `buffer_load_format` over a V# its user data
carries, and the resource plan has to keep that fetch. The fixed-function vertex-fetch
substitution would rewrite it into a host vertex attribute and drop the scalar loads behind
its V#, leaving the fetch with no buffer binding at all — the program then fails to compile
with "BufferLoadFormatXyzw: missing buffer-memory binding".
`ShaderCompileRequest.AllowsEmbeddedVertexFetch` is what refuses the substitution for a
workgroup primitive shader, and `ShaderProgramCache` consults it before detecting an
embedded fetch. One consequence for the host: such a program *does* have ordinary buffer
bindings in its argument buffer, where the vertex path had none.

The fragment interface is unchanged. The mesh vertex payload is declared field for field
like `Gen5VsOut`:

```c
struct Gen5MeshVertex {
    float4 sharpemu_position [[position]];
    float4 param0 [[user(locn0)]];
    …                                  // one per RequiredVertexOutputCount / exported param
};
```

so the paired pixel program built for the vertex path links against a mesh pipeline with
no changes. `Gen5MslShader.AttributeCount` is the param count, as for the vertex stage.

## Launching the draw

```objc
[encoder setRenderPipelineState:p];
[encoder setMeshBuffer:argumentBuffer offset:0 atIndex:0];
[encoder setMeshBuffer:pushData       offset:0 atIndex:1];
[encoder setMeshBuffer:drawInfo       offset:0 atIndex:mesh.DrawInfoBufferIndex];
[encoder setMeshBuffer:indexBuffer    offset:0 atIndex:mesh.IndexBufferIndex];
[encoder drawMeshThreadgroups:MTLSizeMake(subgroups, instances, 1)
        threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
          threadsPerMeshThreadgroup:MTLSizeMake(mesh.ThreadsPerThreadgroup, 1, 1)];
```

- `threadgroupsPerGrid.x` is the number of NGG subgroups:
  `ceil(primitiveCount / mesh.MaxPrimitives)`, and never fewer than
  `ceil(vertexCount / mesh.MaxVertices)`. `primitiveCount` is the draw's primitive count
  for its `VGT_PRIMITIVE_TYPE` (`indexCount / 3` for a triangle list, and so on).
- `threadgroupsPerGrid.y` is the instance count. The shader reads
  `threadgroup_position_in_grid.y` as the instance, so instancing does not need a second
  draw.
- `threadsPerObjectThreadgroup` is `(1,1,1)`; there is no object stage.
- Draw state the vertex path sets (viewport, scissor, depth/stencil, blend, cull mode)
  is unchanged. Cull mode should stay as the guest programs it: the primitive shader
  culls by marking primitives null, which this path lowers to degenerate triangles.

## `Gen5MeshDrawInfo` (buffer 2)

Sixteen dwords, 64 bytes — `Gen5MslMeshInfo.DrawInfoByteSize` /
`DrawInfoDwordCount`. It is what the geometry engine would have put in hardware registers;
the mesh prologue reconstructs the merged ES/GS launch state from it plus the thread's
position.

| Dword | Field | Value |
| --- | --- | --- |
| 0 | `vertices_per_subgroup` | ES vertices in one subgroup. `VGT_GS_ONCHIP_CNTL.ES_VERTS_PER_SUBGRP`, else `GE_MAX_OUTPUT_PER_SUBGROUP`. Clamped in-shader to `MaxVertices`. Without a GS half the host sends `PrimitivesPerSubgroup * vertices-per-primitive` (whole primitives, no vertex reuse) and a lane's vertex slot is `subgroup * vertices_per_subgroup + lane`. |
| 1 | `primitives_per_subgroup` | `VGT_GS_ONCHIP_CNTL.GS_PRIMS_PER_SUBGRP`, else `GE_MAX_OUTPUT_PER_SUBGROUP`. Clamped to `MaxPrimitives`. Without a GS half the host sends `min(MaxPrimitives, MaxVertices / vertices-per-primitive)`; the last subgroup is cut to the primitives `input_vertex_count` leaves. |
| 2 | `index_count` | Indices the draw has. Zero means an automatic (non-indexed) draw. |
| 3 | `index_type` | 0 none, 1 = 16-bit indices, 2 = 32-bit. |
| 4 | `first_index` | Index of the draw's first index, in elements. |
| 5 | `base_vertex` | Signed; added to each fetched index. |
| 6 | `instance_count` | Matches `threadgroupsPerGrid.y`. |
| 7 | `first_instance` | Added to `threadgroup_position_in_grid.y`. |
| 8 | `vertex_group_size` | `GE_CNTL.VERT_GRP_SIZE`. Informational today. |
| 9 | `primitive_group_size` | `GE_CNTL.PRIM_GRP_SIZE`. Informational today. |
| 10 | `user_vector_enable` | `GE_USER_VGPR_EN` bits 0-2. Informational today. |
| 11 | `primitive_amplification` | `GE_NGG_SUBGRP_CNTL`. Informational today. |
| 12 | `object_id` | `VGT_OBJECT_ID`. Informational today. |
| 13 | `gs_input_topology` | Fused (ES+GS) programs only: the draw's input topology, `Gen5MslMeshInputTopology` (0 triangle list, 1 triangle strip, 2 point list). |
| 14 | `gs_input_primitives` | Fused programs only: input primitives per subgroup, `Gen5MslMeshInfo.InputPrimitivesPerSubgroup`. The host launches one subgroup per this many. |
| 15 | `input_vertex_count` | Fused programs only: the draw's vertex or index count, which bounds the last subgroup. |
| 16 | `meshlet_count` | Programs without a GS half: 0 runs whole-triangle subgroups; otherwise subgroup n is meshlet n of the buffer at slot 4. |
| 17 | `meshlet_vertex_base` | Dword offset of the meshlet vertex list (raw index values; `base_vertex` is added in the shader). |
| 18 | `meshlet_prim_base` | Dword offset of the meshlet triangle list (`i0 | i1 << 8 | i2 << 16`, local indices). |
| 19 | reserved | Zero. |

Slot 4 (`sharpemu_meshlets`, programs without a GS half, always bound) holds `meshlet_count` uint4 headers `{vertexOffset, primOffset, firstTriangle, vertexCount | primCount << 16}` followed by the two lists. Slot 5 (`sharpemu_lds_pool`) is bound only when `Gen5MslMeshInfo.DeviceLds` is set: a 16 KB zeroed header of a probe counter and 2048 busy flags, then 2048 slots of 16 KB guest LDS.

The index buffer at slot 3 is read as a dword stream: a 16-bit index is the containing
dword shifted, so one binding serves both types. For an automatic draw the lane's vertex
index is its slot in the grid, and the slot must still be bound (any non-empty buffer).

Nothing in this block is part of the compiled program's cache key, so one pipeline serves
every draw that differs only in counts, index buffer or instancing. The subgroup *budget*
(dwords 0-1 and the output primitive type) **is** part of the key, via
`Gen5NggStageInfo`, because the `mesh<>` extents are compile-time constants.

## How the generated shader seeds the NGG system values

From `Gen5NggExports` (the one place these encodings live) and the thread's position:

- **`s3`, merged wave info** — ES vertex count of *this wave* in bits 0-7, GS primitive
  count in bits 8-15, GS wave id in 16-23, subgroup id in 24-27, wave count in 28-31.
  The guest rebuilds EXEC from the first pair, so this has to be right or the program
  either kills every lane or waits forever on lanes that do not exist.
- **`s2`, merged group info** — the workgroup's ES vertex count in bits 12-20 and input
  primitive count in bits 22-30. The ordered-append id below them stays zero.
- **`v0`/`v1`/`v4`** — GS vertex offsets, two 16-bit ES slots per register.
- **`v2`** — primitive id; **`v3`** — GS invocation id.
- **`v5`** — ES vertex index (after the index fetch and `base_vertex`); **`v8`** —
  instance index. These are the registers the per-vertex path already seeds.

A system scalar is written **only** when it is not also a user-data register for this
program (`UserDataBase > n`). For a graphics stage `UserDataBase` is 8
(`ShaderPipelineCache.VertexUserDataBase`), so `s0..s7` are system scalars in practice.

## The barrier, and why it is not where it looks

The translated program runs inside the `while (active) { switch (pc) }` dispatcher. Only
scalar branches move the pc, so the pc is **wave**-uniform — but two waves of the same
subgroup can sit at different pc values, and a `threadgroup_barrier` reached by a subset of
the threadgroup is undefined in Metal (in practice, a hung queue).

So `S_BARRIER` ends its basic block and parks the wave (`sharpemu_mesh_wait = true`), and
the dispatcher's own loop body carries the only barriers:

```c
for (;;) {
    if (active && !sharpemu_mesh_wait) { switch (pc) { … } /* step limit */ }

    // zero the counters, barrier, count live and parked threads, barrier
    if (live == 0u) break;
    if (parked == live) sharpemu_mesh_wait = false;
}
```

Every thread runs the loop the same number of times, so both barriers are convergent. A
parked wave resumes only once every thread that is still running has parked too, and a
thread that has ended stops being counted — which is also what the hardware does when a
wave ends while others wait on the barrier. `SHARPEMU_SHADER_MAX_STEPS` still bounds the
loop.

Cost is two threadgroup barriers and a few atomics per **block** execution, not per
instruction. A single-wave subgroup degenerates to nearly free.

## Culling

Metal cannot drop a mesh primitive in place. A primitive whose export set the
null-primitive bit (31), or that no lane exported at all, is emitted with three identical
indices: a zero-area triangle rasterizes nothing. `set_primitive_count` additionally
reflects what `GS_ALLOC_REQ` asked for, so slots past the allocation are never read.

## Open questions

These are the parts the synthetic fixtures pin but the guest has not confirmed:

1. **Primitive assembly for the GS vertex offsets.** A fused program (one with a GS half,
   `Gen5NggStageInfo.GeometryVerticesOut != 0`) gets its offsets assembled from the draw's
   input topology — list, strip (odd triangles swap their first two vertices) or point —
   each as the subgroup-local ES vertex shifted left by two, which is what the guest's GS
   half decodes (`v_bfe_u32 v, v0, 2, 14`, then its own ES stride), and the subgroup's ES
   vertex and GS primitive counts come from its share of the draw. Every other primitive
   shader is still seeded as if the
   subgroup were a triangle *list*: primitive *p* uses ES slots `3p, 3p+1, 3p+2`. Strips,
   fans and indexed draws with vertex reuse pack the subgroup differently, and
   `GE_CNTL`'s group sizes are what bound that packing. A program that reads these offsets
   rather than exporting connectivity it computed itself will be wrong.
2. **Passthrough primitive input.** In NGG passthrough mode the hardware hands the shader a
   pre-packed primitive in a VGPR instead of letting it build one. That register is not
   seeded, because such programs still take the per-vertex path (no LDS, no barrier). If
   one ever needs the mesh path, the packing it arrives in has to be settled first.
3. **`GE_NGG_SUBGRP_CNTL`.** The Void capture has 2 here with no geometry program, which
   reads like `PRIM_AMP_FACTOR = 2` — primitive amplification without a GS. It is
   transported as `PrimitiveAmplificationFactor` and otherwise unused; a program that
   really amplifies would need `MaxPrimitives` scaled by it.
4. **`VGT_GS_ONCHIP_CNTL` field offsets.** The register's `ES_VERTS_PER_SUBGRP` /
   `GS_PRIMS_PER_SUBGRP` field *names* and their role are public (Mesa's `G_028A44_*`
   macros); the exact bit positions are not, so `ShaderInterfaceRegisters`' decode is
   inherited rather than verified. The clamp to Metal's extents makes a wrong decode a
   performance or coverage problem, not a crash.
5. **Barrier uniformity across waves.** The rendezvous makes a non-uniform barrier safe
   rather than fatal, but it does not make it *correct*: if one wave takes a scalar branch
   that skips a barrier the others execute, the parked waves are released early. Mesa
   emits NGG barriers at top level precisely because the hardware requires workgroup
   uniformity, so real programs should not do this — but nothing here proves it.
6. **Position slots beyond `pos0`.** `pos1..pos4` (clip/cull distances, point size) are
   dropped, as on the per-vertex path. The mesh vertex payload would need the
   corresponding `[[clip_distance]]` / `[[point_size]]` members.

## Tests

`tests/SharpEmu.ShaderCompiler.Metal.Tests`:

- `NggMeshTranslationTests` — structure of the emitted MSL and of `Gen5MslMeshInfo`,
  including that passthrough programs and programs without the draw's primitive-shader
  state keep the per-vertex path. Runs everywhere.
- `ShaderProgramCacheTests.PrimitiveShaderWorkgroupKeepsItsOwnVertexFetch` (in
  `SharpEmu.Libs.Tests`) — the pipeline must not substitute the ES fetch for a workgroup
  primitive shader, and must still substitute it for the same program without the draw's
  primitive-shader state.
- `VgtShaderStagesTests` and the `MeshPipelines_*` rows of `RenderExecutorDrawTests` — which
  stage masks are primitive-shader draws, including that a plain vertex draw and a mask that
  selects tessellation or the legacy vertex stage are not.
- `NggMeshRuntimeTests` — needs a Metal device
  (`SHARPEMU_TEST_REQUIRE_DEVICE=1 arch -x86_64 ~/.dotnet-x64/dotnet test …`). The emitted
  MSL must compile with the OS Metal compiler, and a synthetic primitive shader must
  rasterize through a real mesh pipeline: it exchanges its vertex value across the
  subgroup's two waves through LDS, synchronizes, allocates one triangle, exports it, and
  the render target has to show that triangle with the exported parameter interpolated to
  its exact barycentric value. The null-primitive bit must leave the target empty. A second
  device row fetches the positions with `buffer_load_format_xyzw` over a V# in user data,
  so the same pixels only appear if the mesh stage's buffer binding reached the GPU.
- The preserved Void Terrarium capture (707 instructions, 19 `DS_WRITE_B128`, six
  `S_BARRIER`, `S_SENDMSG` message 9, `EXP` target 20) is translated and compiled when it
  is present on the machine, and skipped otherwise. No guest shader bytes are in the
  repository.
