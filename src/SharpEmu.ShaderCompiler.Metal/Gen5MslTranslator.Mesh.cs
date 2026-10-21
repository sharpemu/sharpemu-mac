// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal;

// The NGG primitive-shader path: a merged ES/GS program emitted as an MSL [[mesh]]
// function instead of a per-vertex vertex function.
//
// The hardware stage is a workgroup, not a vertex invocation: its waves exchange data
// through LDS, synchronize with S_BARRIER, ask for export space with
// S_SENDMSG GS_ALLOC_REQ and then export positions, parameters and primitive
// connectivity. None of that survives the per-vertex model, which is why the vertex path
// only ever worked for passthrough NGG programs. The mesh mapping is one-to-one:
//
//   NGG subgroup        -> one Metal threadgroup (one mesh grid element)
//   wave lane           -> one Metal thread; simdgroup lane, so EXEC/ballots stay real
//   LDS                 -> threadgroup memory
//   S_BARRIER           -> a workgroup rendezvous in the dispatcher loop (see below)
//   GS_ALLOC_REQ        -> set_primitive_count
//   EXP POS0/PARAMn     -> set_vertex at the exporting lane's index
//   EXP target 20       -> set_index for the exporting lane's primitive
//
// Barriers need care because the translated program runs inside the
// while(active){switch(pc)} dispatcher: only scalar branches move the pc, so the pc is
// wave-uniform, but two waves of the same subgroup can sit at different pc values. A
// threadgroup_barrier emitted at the barrier's own pc would then be reached by a subset
// of the threadgroup, which Metal leaves undefined (in practice a hung queue). Instead
// S_BARRIER ends its basic block and parks the wave, and the dispatcher's own loop body
// carries the only barriers: every thread runs the loop the same number of times, counts
// how many threads are still live and how many are parked, and releases the parked ones
// once every live thread has arrived. That is the hardware rule, including the part where
// a wave that has already ended stops holding the barrier.
public static partial class Gen5MslTranslator
{
    // Metal needs the mesh output extents as compile-time constants, and the whole output
    // must fit in the mesh stage's threadgroup budget, so the subgroup registers are
    // clamped into a range the device accepts.
    private const uint MeshMaxVertices = 256;
    private const uint MeshMaxPrimitives = 256;
    private const uint MeshMaxWaves = 8;
    private const uint MeshWaveLanes = 32;

    // Guest LDS for a primitive shader is sized from the stage's own allocation, not from
    // the compute path's full 32 KB: the mesh output payload shares the threadgroup
    // budget. The count stays a power of two because LDS addressing is mask-based.
    private const uint MeshLdsDwordFloor = 512;
    private const uint MeshLdsDwordCeiling = 4096;

    // A guest LDS above this does not stay in threadgroup memory: its occupancy cost
    // (a 15.6 KB allocation leaves a handful of threadgroups per core) outweighs the cost of
    // keeping the data in device memory, so it lives in a slot of a shared device pool.
    private const uint MeshDeviceLdsThresholdBytes = 12288;

    private const string MeshVertexStructName = "Gen5MeshVertex";
    private const string MeshPrimitiveStructName = "Gen5MeshPrimitive";
    private const string MeshDrawInfoStructName = "Gen5MeshDrawInfo";
    private const string MeshOutputTypeName = "Gen5MeshOutput";
    private const string MeshOutputName = "sharpemu_mesh";
    private const string MeshDrawInfoName = "sharpemu_draw";
    private const string MeshIndexBufferName = "sharpemu_indices";
    private const string MeshletBufferName = "sharpemu_meshlets";

    private const uint MeshDrawInfoBufferIndex = 2;
    private const uint MeshIndexBufferIndex = 3;

    /// <summary>True when this request must be emitted as a mesh function: the draw selected
    /// the NGG primitive-shader stage and the program is not a passthrough one, or its pixel
    /// program reads per-vertex values only a mesh primitive can carry on this host.</summary>
    public static bool RequiresMeshEmission(ShaderCompileRequest request) =>
        request.Stage == ShaderStage.Vertex &&
        request.PrimitiveShader is not null &&
        (request.MeshPerVertexParameterMask != 0 ||
         ShaderCompileRequest.RequiresPrimitiveShaderWorkgroup(request.Program));

    // The subgroup budget the geometry-engine registers describe, clamped to what a Metal
    // mesh threadgroup can carry.
    private static Gen5MslMeshInfo BuildMeshInfo(ShaderCompileRequest request, uint ldsDwords)
    {
        var ngg = request.PrimitiveShader ?? default;
        var fallback = ngg.MaxOutputPerSubgroup != 0 ? ngg.MaxOutputPerSubgroup : MeshWaveLanes;
        var vertices = Clamp(ngg.VerticesPerSubgroup != 0 ? ngg.VerticesPerSubgroup : fallback, 1, MeshMaxVertices);
        var primitives = Clamp(ngg.PrimitivesPerSubgroup != 0 ? ngg.PrimitivesPerSubgroup : fallback, 1, MeshMaxPrimitives);
        var perPrimitive = Gen5NggExports.VerticesPerPrimitive(ngg.OutputPrimitiveType);

        // A wave64 program with no GS half runs as a narrow wave64: one 32-lane simdgroup
        // that stands for the low half of the guest wave, so a subgroup (one lane per
        // vertex, one per primitive) has to fit in 32 lanes. The pipeline declares the full
        // 32 of each; how many a draw uses is per-draw state (draw-info words 0/1 for the
        // whole-triangle path, the meshlet header otherwise), never the declared extent.
        var deviceLds = ldsDwords * sizeof(uint) > MeshDeviceLdsThresholdBytes;
        var narrow = IsNarrowWave64(request);
        if (narrow)
        {
            vertices = MeshWaveLanes;
            primitives = MeshWaveLanes;
        }
        else if (request.MeshPerVertexParameterMask != 0)
        {
            // Three vertex values per handed parameter make each primitive heavy, so the
            // extents shrink until the declared outputs fit what the device accepts.
            var fit = MeshOutputBudgetBytes / ((vertices * MeshVertexBytesBound(request)) +
                (primitives * MeshPrimitiveBytesBound(request)));
            if (fit < 1.0)
            {
                vertices = Math.Max((uint)(vertices * fit), Math.Max(perPrimitive, 1u));
                primitives = Math.Max((uint)(primitives * fit), 1u);
            }
        }

        var waves = Clamp((Math.Max(vertices, primitives) + MeshWaveLanes - 1) / MeshWaveLanes, 1, MeshMaxWaves);
        var topology = ngg.OutputPrimitiveType switch
        {
            Gen5NggExports.OutputPrimitivePointList => Gen5MslMeshTopology.Point,
            Gen5NggExports.OutputPrimitiveLineStrip => Gen5MslMeshTopology.Line,
            _ => Gen5MslMeshTopology.Triangle,
        };

        return new Gen5MslMeshInfo(
            ThreadsPerThreadgroup: waves * MeshWaveLanes,
            WavesPerThreadgroup: waves,
            MaxVertices: vertices,
            MaxPrimitives: primitives,
            VerticesPerPrimitive: Gen5NggExports.VerticesPerPrimitive(ngg.OutputPrimitiveType),
            Topology: topology,
            ThreadgroupMemoryBytes: deviceLds ? 0 : ldsDwords * sizeof(uint),
            DrawInfoBufferIndex: MeshDrawInfoBufferIndex,
            IndexBufferIndex: MeshIndexBufferIndex,
            ReadsIndexBuffer: true,
            GeometryVerticesOut: ngg.GeometryVerticesOut,
            NarrowWave64: narrow,
            DeviceLds: deviceLds,
            Wave64: ngg.IsWave64,
            LdsBytes: ldsDwords * sizeof(uint));
    }

    // Apple GPUs refuse a mesh pipeline whose declared vertex and primitive outputs exceed
    // 32 KB ("Total mesh size exceeds the maximum mesh size allowed (32768)").
    private const double MeshOutputBudgetBytes = 32768;

    // The most one Gen5MeshVertex can take: the position, every param the program exports
    // or the pixel program reads, the host interpolants unpacked from them, and the point
    // size and clip distances rounded to 16.
    private static uint MeshVertexBytesBound(ShaderCompileRequest request)
    {
        var parameters = new HashSet<uint>();
        for (var location = 0u; location < (uint)Math.Max(request.RequiredVertexOutputCount, 0); location++)
        {
            parameters.Add(location);
        }

        foreach (var instruction in request.Program.Instructions)
        {
            if (instruction.Control is Gen5ExportControl { Target: >= 32 and < 64 } export)
            {
                parameters.Add(export.Target - 32);
            }
        }

        return 16u + (16u * (uint)(parameters.Count + request.VertexInterpolants.Count)) + 48u;
    }

    // Three float4 per handed parameter, clip W for its vertices, then layer/viewport.
    private static uint MeshPrimitiveBytesBound(ShaderCompileRequest request) =>
        (48u * (uint)System.Numerics.BitOperations.PopCount(request.MeshPerVertexParameterMask)) +
        (request.MeshPerVertexParameterMask != 0 ? 16u : 0u) + 16u;

    /// <summary>The [[user]] name of one vertex's value of a param a mesh primitive hands
    /// the pixel stage (<see cref="ShaderCompileRequest.MeshPerVertexParameterMask"/>);
    /// vertex 0..2 are P0, P1 and P2 of the guest's V_INTERP.</summary>
    internal static string PerVertexInputName(uint location, uint vertex) => $"locn{location}_v{vertex}";

    /// <summary>True when the program is a wave64 primitive shader without a GS half. The
    /// Metal wave is 32 lanes, so the guest's upper 32 lanes are modelled as architecturally
    /// inactive (see <see cref="CompilationContext.IsNarrowWave64"/>).</summary>
    private static bool IsNarrowWave64(ShaderCompileRequest request) =>
        request.PrimitiveShader is { IsWave64: true, GeometryVerticesOut: 0 };

    // The stage's LDS allocation, exactly, inside the mesh budget: the declared threadgroup
    // memory sets occupancy, so it is not rounded up. Addressing masks to the power of two
    // that covers it and clamps to the last dword (see LdsIndex). A program that asks for
    // nothing still gets the floor so a stray address stays in bounds.
    private static uint MeshLdsDwordCount(ShaderCompileRequest request)
    {
        var requested = request.PrimitiveShader is { LocalDataShareBytes: var bytes } && bytes != 0
            ? (bytes + sizeof(uint) - 1) / sizeof(uint)
            : request.LocalDataShareDwords;
        return requested == 0 ? MeshLdsDwordFloor : Math.Min(requested, MeshLdsDwordCeiling);
    }

    private static uint Clamp(uint value, uint low, uint high) => Math.Min(Math.Max(value, low), high);

    private sealed partial class CompilationContext
    {
        private readonly Gen5MslMeshInfo? _meshInfo;
        private readonly uint _meshLdsDwords;

        /// <summary>Threadgroup memory this compute dispatch declares, in dwords.</summary>
        private readonly uint _computeLdsDwords;

        /// <summary>Whether the wave64 ballot bridge has to share the LDS allocation.
        /// Only a dispatch that asked for the whole 32 KB leaves no room for its own
        /// slots.</summary>
        private bool AliasesWaveScratchIntoLds =>
            _stage == Gen5MslStage.Compute && _usesLds &&
            _computeLdsDwords + WaveScratchDwords > LdsDwordCount;


        /// <summary>A wave64 primitive shader with no GS half, run as one 32-lane simdgroup.
        /// Guest lanes 32..63 do not exist: a lane-mask write to an SGPR pair leaves the high
        /// dword 0, EXEC_HI is 0, V_READLANE / V_READFIRSTLANE of a lane >= 32 read 0,
        /// V_MBCNT_HI passes its addend through and V_WRITELANE to such a lane is a no-op.</summary>
        private bool IsNarrowWave64 => _meshInfo is { NarrowWave64: true };

        /// <summary>The guest LDS lives in a slot of the device pool: the program uses LDS and
        /// the exact allocation is above the threadgroup-memory threshold.</summary>
        private bool UsesDeviceLds => _usesLds && _meshInfo is { DeviceLds: true };

        /// <summary>Dwords of one device LDS slot; a power of two so the masked LDS index
        /// stays inside the slot.</summary>
        private uint DeviceLdsSlotDwords => (uint)System.Numerics.BitOperations.RoundUpToPowerOf2(_meshLdsDwords);

        /// <summary>The address space of a pointer to guest LDS.</summary>
        private string LdsAddressSpace => UsesDeviceLds ? "device" : "threadgroup";

        private string MeshWorkgroupBarrier => UsesDeviceLds
            ? "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);"
            : "threadgroup_barrier(mem_flags::mem_threadgroup);";

        private Gen5MslMeshInfo Mesh => _meshInfo ??
            throw new InvalidOperationException("the mesh interface is only built for the mesh stage");

        private static string MeshTopologyName(Gen5MslMeshTopology topology) => topology switch
        {
            Gen5MslMeshTopology.Point => "topology::point",
            Gen5MslMeshTopology.Line => "topology::line",
            _ => "topology::triangle",
        };

        // ---- module ----

        // The vertex payload, the launch inputs and the mesh output type. The vertex struct
        // deliberately matches Gen5VsOut field for field: the paired pixel program is
        // translated against that interface, so a mesh pipeline reuses it unchanged.
        private void EmitMeshTypes(StringBuilder source)
        {
            var mesh = Mesh;
            source.AppendLine($"struct {MeshVertexStructName}");
            source.AppendLine("{");
            // Invariant like the vertex stage's (see Gen5MslTranslator.EmitModule).
            source.AppendLine("    float4 sharpemu_position [[position, invariant]];");
            EmitAuxPositionFields(source, perVertexLayer: false);
            foreach (var location in _vertexOutputs)
            {
                source.AppendLine($"    float4 param{location} [[user(locn{location})]];");
            }

            EmitInterpolantOutputMembers(source);
            source.AppendLine("};");
            source.AppendLine();

            // The layer and viewport index are per-primitive in a Metal mesh, and so are the
            // three vertex values of a param the pixel program reads per vertex.
            var primitiveType = "void";
            if (MeshHasPrimitiveData)
            {
                primitiveType = MeshPrimitiveStructName;
                source.AppendLine($"struct {MeshPrimitiveStructName}");
                source.AppendLine("{");
                if (_writesLayer)
                {
                    source.AppendLine("    uint sharpemu_layer [[render_target_array_index]];");
                }

                if (_writesViewportIndex)
                {
                    source.AppendLine("    uint sharpemu_viewport [[viewport_array_index]];");
                }

                if (MeshPerVertexLocations.Any())
                {
                    source.AppendLine($"    float3 sharpemu_vertex_w [[user({VertexWInputName})]];");
                }

                foreach (var location in MeshPerVertexLocations)
                {
                    for (var vertex = 0u; vertex < 3u; vertex++)
                    {
                        source.AppendLine(
                            $"    float4 param{location}_v{vertex} [[user({PerVertexInputName(location, vertex)})]];");
                    }
                }

                source.AppendLine("};");
                source.AppendLine();
            }

            // What the geometry engine would hand the merged stage in hardware registers.
            // The host fills one block per draw; docs/mac/ngg-mesh-shaders.md is the
            // field-by-field contract and must stay in step with this struct.
            source.AppendLine($"struct {MeshDrawInfoStructName}");
            source.AppendLine("{");
            source.AppendLine("    uint vertices_per_subgroup;");
            source.AppendLine("    uint primitives_per_subgroup;");
            source.AppendLine("    uint index_count;");
            source.AppendLine("    uint index_type;");
            source.AppendLine("    uint first_index;");
            source.AppendLine("    int base_vertex;");
            source.AppendLine("    uint instance_count;");
            source.AppendLine("    uint first_instance;");
            source.AppendLine("    uint vertex_group_size;");
            source.AppendLine("    uint primitive_group_size;");
            source.AppendLine("    uint user_vector_enable;");
            source.AppendLine("    uint primitive_amplification;");
            source.AppendLine("    uint object_id;");
            source.AppendLine("    uint gs_input_topology;");
            source.AppendLine("    uint gs_input_primitives;");
            source.AppendLine("    uint input_vertex_count;");

            // Cached meshlets (see MetalMeshletCache): when meshlet_count is not zero, subgroup
            // n is meshlet n of the buffer bound at the meshlet slot instead of a run of the
            // draw's index stream. The vertex and primitive bases are dword offsets into it.
            source.AppendLine("    uint meshlet_count;");
            source.AppendLine("    uint meshlet_vertex_base;");
            source.AppendLine("    uint meshlet_prim_base;");
            source.AppendLine("    uint meshlet_reserved;");
            source.AppendLine("};");
            source.AppendLine();
            source.AppendLine(
                $"using {MeshOutputTypeName} = mesh<{MeshVertexStructName}, {primitiveType}, " +
                $"{mesh.MaxVertices}, {mesh.MaxPrimitives}, {MeshTopologyName(mesh.Topology)}>;");
            source.AppendLine();

            // Guest indices arrive as a raw dword stream; a 16-bit index buffer is read as
            // the containing dword and shifted, which keeps one binding for both types.
            source.AppendLine("static inline uint sharpemu_mesh_index(");
            source.AppendLine($"    device const uint* indices, constant {MeshDrawInfoStructName}& draw, uint slot)");
            source.AppendLine("{");
            source.AppendLine("    if (draw.index_type == 0u || slot >= draw.index_count)");
            source.AppendLine("    {");
            source.AppendLine("        return slot;");
            source.AppendLine("    }");
            source.AppendLine();
            source.AppendLine("    const uint offset = draw.first_index + slot;");
            source.AppendLine("    if (draw.index_type == 1u)");
            source.AppendLine("    {");
            source.AppendLine("        return (indices[offset >> 1u] >> ((offset & 1u) * 16u)) & 0xFFFFu;");
            source.AppendLine("    }");
            source.AppendLine();
            source.AppendLine("    return indices[offset];");
            source.AppendLine("}");
            source.AppendLine();
        }

        private void EmitMeshSignature(StringBuilder source)
        {
            var mesh = Mesh;
            source.AppendLine(
                $"[[mesh, max_total_threads_per_threadgroup({mesh.ThreadsPerThreadgroup})]]");
            source.AppendLine($"void {EntryPointName}(");
            source.AppendLine($"    {MeshOutputTypeName} {MeshOutputName},");
        }

        private void EmitMeshSystemParameters(StringBuilder source)
        {
            source.AppendLine(
                $"    constant {MeshDrawInfoStructName}& {MeshDrawInfoName} [[buffer({MeshDrawInfoBufferIndex})]],");
            source.AppendLine(
                $"    device const uint* {MeshIndexBufferName} [[buffer({MeshIndexBufferIndex})]],");
            if (UsesDeviceLds)
            {
                source.AppendLine(
                    $"    device atomic_uint* sharpemu_lds_pool [[buffer({Gen5MslMeshInfo.LdsPoolBufferIndex})]],");
            }

            if (!Mesh.HasGeometryHalf)
            {
                source.AppendLine(
                    $"    device const uint* {MeshletBufferName} [[buffer({Gen5MslMeshInfo.MeshletBufferIndex})]],");
            }

            source.AppendLine("    uint sharpemu_tg_index [[thread_index_in_threadgroup]],");
            source.AppendLine("    uint3 sharpemu_group_id [[threadgroup_position_in_grid]],");
            source.AppendLine("    uint sharpemu_lane [[thread_index_in_simdgroup]],");
        }

        // ---- body ----

        // The threadgroup state the mesh body needs on top of the shared register file:
        // the barrier rendezvous counters and the export allocation GS_ALLOC_REQ writes.
        // Threadgroup variables cannot carry initializers, so thread 0 seeds them and one
        // barrier publishes them before the dispatcher starts.
        private void EmitMeshDeclarations(StringBuilder source)
        {
            var mesh = Mesh;
            source.AppendLine($"    const uint sharpemu_mesh_wave = sharpemu_tg_index / {MeshWaveLanes}u;");
            source.AppendLine("    const uint sharpemu_mesh_subgroup = sharpemu_group_id.x;");
            source.AppendLine("    const uint sharpemu_mesh_instance = sharpemu_group_id.y;");
            source.AppendLine("    threadgroup atomic_uint sharpemu_mesh_live;");
            source.AppendLine("    threadgroup atomic_uint sharpemu_mesh_parked;");
            source.AppendLine("    threadgroup uint sharpemu_mesh_alloc[2];");
            source.AppendLine("    bool sharpemu_mesh_wait = false;");
            source.AppendLine($"    {MeshVertexStructName} sharpemu_out = {{}};");

            // A primitive no lane exported stays culled; a degenerate index triple is how
            // Metal expresses that, since a mesh primitive cannot be dropped in place.
            source.AppendLine("    uint3 sharpemu_mesh_prim = uint3(0u);");
            source.AppendLine("    bool sharpemu_mesh_culled = true;");
            if (MeshWritesPrimitiveOutputs)
            {
                // This lane's exported layer and viewport index, and every vertex's copy
                // for the epilogue: a primitive takes its provoking vertex's values.
                source.AppendLine("    uint sharpemu_mesh_layer = 0u;");
                source.AppendLine("    uint sharpemu_mesh_viewport = 0u;");
                source.AppendLine($"    threadgroup uint2 sharpemu_mesh_vertex_target[{mesh.MaxVertices}];");
            }

            // Every vertex's value of each param handed per vertex, for the epilogue to
            // gather by the primitive's indices.
            if (MeshPerVertexLocations.Any())
            {
                source.AppendLine($"    threadgroup float sharpemu_mesh_vertex_w[{mesh.MaxVertices}];");
            }
            foreach (var location in MeshPerVertexLocations)
            {
                source.AppendLine($"    threadgroup float4 sharpemu_mesh_param{location}[{mesh.MaxVertices}];");
            }

            if (mesh.HasGeometryHalf)
            {
                // A GS half takes this subgroup's share of the draw's input primitives, and
                // the ES half the vertices they are assembled from: a list's primitives own
                // three vertices each, a strip's share all but the first two.
                source.AppendLine($"    const uint sharpemu_gs_input = {MeshDrawInfoName}.gs_input_topology;");
                source.AppendLine(
                    $"    const uint sharpemu_gs_prim_start = sharpemu_mesh_subgroup * {MeshDrawInfoName}.gs_input_primitives;");
                source.AppendLine(
                    "    const uint sharpemu_gs_vertex_start = sharpemu_gs_input == " +
                    $"{(uint)Gen5MslMeshInputTopology.TriangleList}u ? sharpemu_gs_prim_start * 3u : sharpemu_gs_prim_start;");
            }

            if (!mesh.HasGeometryHalf)
            {
                // Whole subgroup uniform: the cached meshlet this subgroup is, or zeros.
                source.AppendLine($"    const bool sharpemu_meshlets_on = {MeshDrawInfoName}.meshlet_count != 0u;");
                source.AppendLine(
                    $"    const uint4 sharpemu_ml = sharpemu_meshlets_on ? ((device const uint4*){MeshletBufferName})[sharpemu_mesh_subgroup] : uint4(0u);");
            }

            source.AppendLine("    if (sharpemu_tg_index == 0u)");
            source.AppendLine("    {");
            if (mesh.HasGeometryHalf)
            {
                var list = (uint)Gen5MslMeshInputTopology.TriangleList;
                var strip = (uint)Gen5MslMeshInputTopology.TriangleStrip;
                source.AppendLine(
                    $"        const uint sharpemu_gs_per = {MeshDrawInfoName}.gs_input_primitives;");
                source.AppendLine(
                    $"        const uint sharpemu_gs_left = {MeshDrawInfoName}.input_vertex_count > sharpemu_gs_vertex_start");
                source.AppendLine(
                    $"            ? {MeshDrawInfoName}.input_vertex_count - sharpemu_gs_vertex_start : 0u;");
                source.AppendLine(
                    $"        const uint sharpemu_gs_verts = min(sharpemu_gs_left, sharpemu_gs_input == {list}u ? sharpemu_gs_per * 3u");
                source.AppendLine(
                    $"            : sharpemu_gs_input == {strip}u ? sharpemu_gs_per + 2u : sharpemu_gs_per);");
                source.AppendLine(
                    $"        const uint sharpemu_gs_prims = sharpemu_gs_input == {list}u ? sharpemu_gs_verts / 3u");
                source.AppendLine(
                    $"            : sharpemu_gs_input == {strip}u ? (sharpemu_gs_verts > 2u ? sharpemu_gs_verts - 2u : 0u)");
                source.AppendLine("            : sharpemu_gs_verts;");
                source.AppendLine($"        sharpemu_mesh_alloc[0] = min(sharpemu_gs_verts, {mesh.MaxVertices}u);");
                source.AppendLine($"        sharpemu_mesh_alloc[1] = min(sharpemu_gs_prims, {mesh.MaxPrimitives}u);");
            }
            else
            {
                // The last subgroup holds only the primitives the draw has left; a draw that
                // reports no vertex count (input_vertex_count == 0) is not bounded.
                var perPrimitive = Math.Max(mesh.VerticesPerPrimitive, 1u);
                source.AppendLine(
                    $"        const uint sharpemu_mesh_prims_total = {MeshDrawInfoName}.input_vertex_count / {perPrimitive}u;");
                source.AppendLine(
                    $"        const uint sharpemu_mesh_prims_first = sharpemu_mesh_subgroup * {MeshDrawInfoName}.primitives_per_subgroup;");
                source.AppendLine(
                    $"        const uint sharpemu_mesh_prims_left = {MeshDrawInfoName}.input_vertex_count == 0u ? 0xFFFFFFFFu");
                source.AppendLine(
                    "            : (sharpemu_mesh_prims_total > sharpemu_mesh_prims_first ? sharpemu_mesh_prims_total - sharpemu_mesh_prims_first : 0u);");
                source.AppendLine(
                    $"        const uint sharpemu_mesh_prims = min(min({MeshDrawInfoName}.primitives_per_subgroup, {mesh.MaxPrimitives}u), sharpemu_mesh_prims_left);");
                source.AppendLine(
                    $"        const uint sharpemu_mesh_verts = min({MeshDrawInfoName}.vertices_per_subgroup, {mesh.MaxVertices}u);");
                source.AppendLine(
                    $"        sharpemu_mesh_alloc[0] = sharpemu_mesh_prims_left < {MeshDrawInfoName}.primitives_per_subgroup");
                source.AppendLine(
                    $"            ? min(sharpemu_mesh_verts, sharpemu_mesh_prims * {perPrimitive}u) : sharpemu_mesh_verts;");
                source.AppendLine("        sharpemu_mesh_alloc[1] = sharpemu_mesh_prims;");
                source.AppendLine("        if (sharpemu_meshlets_on)");
                source.AppendLine("        {");
                source.AppendLine($"            sharpemu_mesh_alloc[0] = min(sharpemu_ml.w & 0xFFFFu, {mesh.MaxVertices}u);");
                source.AppendLine($"            sharpemu_mesh_alloc[1] = min(sharpemu_ml.w >> 16u, {mesh.MaxPrimitives}u);");
                source.AppendLine("        }");
            }

            if (UsesDeviceLds)
            {
                // Take a free slot of the shared pool. The probe counter spreads threadgroups
                // over the slots; a holder never waits on another threadgroup, and there are
                // far more slots than resident threadgroups, so the probe ends quickly. The
                // cap only exists so a corrupt pool degrades to a shared slot (reported)
                // instead of hanging the queue.
                var slots = Gen5MslMeshInfo.LdsPoolSlots;
                source.AppendLine("        for (uint sharpemu_probe = 0u;; sharpemu_probe++)");
                source.AppendLine("        {");
                source.AppendLine(
                    $"            const uint sharpemu_slot = atomic_fetch_add_explicit(&sharpemu_lds_pool[0], 1u, memory_order_relaxed) % {slots}u;");
                source.AppendLine("            uint sharpemu_expected = 0u;");
                source.AppendLine(
                    "            if (atomic_compare_exchange_weak_explicit(&sharpemu_lds_pool[1u + sharpemu_slot], &sharpemu_expected, 1u, memory_order_relaxed, memory_order_relaxed))");
                source.AppendLine("            {");
                source.AppendLine("                sharpemu_lds_slot = sharpemu_slot;");
                source.AppendLine("                break;");
                source.AppendLine("            }");
                source.AppendLine("            if (sharpemu_probe == 65536u)");
                source.AppendLine("            {");
                source.AppendLine("                sharpemu_lds_slot = sharpemu_slot;");
                source.AppendLine($"                {StepCapReportCode}");
                source.AppendLine("                break;");
                source.AppendLine("            }");
                source.AppendLine("        }");
            }

            source.AppendLine("    }");
            source.AppendLine();
            source.AppendLine($"    {MeshWorkgroupBarrier}");
            if (UsesDeviceLds)
            {
                source.AppendLine(
                    $"    device uint* sharpemu_lds = (device uint*)sharpemu_lds_pool + {Gen5MslMeshInfo.LdsPoolHeaderDwords}u + " +
                    $"sharpemu_lds_slot * {DeviceLdsSlotDwords}u;");
            }
        }

        // A system scalar is one the user data does not cover: below the user-data base, or
        // excluded from it - a fused program's user data starts at s0 so the geometry
        // user-data pointer can occupy s[0:1], and s2..s7 stay the launch state.
        private bool IsMeshSystemScalar(uint register) =>
            _request.UserDataBase > register ||
            (register < 64 && (_request.ExcludedUserDataRegisters & (1UL << (int)register)) != 0);

        // The merged ES/GS launch state. Hardware splits one subgroup across waves and
        // hands each wave its own counts plus packed per-lane inputs; the same values are
        // reconstructed here from the thread's position and the draw block.
        private void EmitMeshInitialState(StringBuilder source)
        {
            var mesh = Mesh;
            var lanes = $"{MeshWaveLanes}u";
            source.AppendLine($"    const uint sharpemu_mesh_wave_first = sharpemu_mesh_wave * {lanes};");
            source.AppendLine(
                "    const uint sharpemu_mesh_subgroup_verts = sharpemu_mesh_alloc[0];");
            source.AppendLine(
                "    const uint sharpemu_mesh_subgroup_prims = sharpemu_mesh_alloc[1];");
            source.AppendLine(
                "    const uint sharpemu_mesh_wave_verts = sharpemu_mesh_subgroup_verts > sharpemu_mesh_wave_first");
            source.AppendLine(
                $"        ? min(sharpemu_mesh_subgroup_verts - sharpemu_mesh_wave_first, {lanes}) : 0u;");
            source.AppendLine(
                "    const uint sharpemu_mesh_wave_prims = sharpemu_mesh_subgroup_prims > sharpemu_mesh_wave_first");
            source.AppendLine(
                $"        ? min(sharpemu_mesh_subgroup_prims - sharpemu_mesh_wave_first, {lanes}) : 0u;");

            // The lane's ES vertex and GS primitive inside the subgroup, and the guest
            // vertex the geometry engine's index fetch would have produced for it.
            if (mesh.HasGeometryHalf)
            {
                source.AppendLine(
                    "    const uint sharpemu_mesh_vertex_slot = sharpemu_gs_vertex_start + sharpemu_tg_index;");
                source.AppendLine(
                    "    const uint sharpemu_mesh_prim_slot = sharpemu_gs_prim_start + sharpemu_tg_index;");
            }
            else
            {
                // Every primitive owns its vertices, so a subgroup of P primitives owns
                // vertices_per_subgroup = P * vertices-per-primitive consecutive slots and
                // never splits a primitive; the host sizes both from the same numbers.
                source.AppendLine(
                    "    const uint sharpemu_mesh_vertex_slot = sharpemu_mesh_subgroup * " +
                    $"{MeshDrawInfoName}.vertices_per_subgroup + sharpemu_tg_index;");
                source.AppendLine(
                    "    const uint sharpemu_mesh_prim_slot = sharpemu_meshlets_on ? sharpemu_ml.z + sharpemu_tg_index : " +
                    $"sharpemu_mesh_subgroup * {MeshDrawInfoName}.primitives_per_subgroup + sharpemu_tg_index;");
            }
            var plainVertexIndex =
                $"(uint)((int)sharpemu_mesh_index({MeshIndexBufferName}, " +
                $"{MeshDrawInfoName}, sharpemu_mesh_vertex_slot) + {MeshDrawInfoName}.base_vertex)";
            if (!mesh.HasGeometryHalf)
            {
                // A meshlet lists its vertices' raw index values; the lane past the meshlet's
                // vertex count has none.
                source.AppendLine(
                    "    const uint sharpemu_mesh_vertex_index = sharpemu_meshlets_on");
                source.AppendLine(
                    "        ? (sharpemu_tg_index < (sharpemu_ml.w & 0xFFFFu)");
                source.AppendLine(
                    $"            ? (uint)((int){MeshletBufferName}[{MeshDrawInfoName}.meshlet_vertex_base + sharpemu_ml.x + sharpemu_tg_index] + {MeshDrawInfoName}.base_vertex)");
                source.AppendLine("            : 0u)");
                source.AppendLine($"        : {plainVertexIndex};");
            }
            else
            {
                source.AppendLine($"    const uint sharpemu_mesh_vertex_index = {plainVertexIndex};");
            }
            source.AppendLine(
                $"    const uint sharpemu_mesh_instance_index = {MeshDrawInfoName}.first_instance + sharpemu_mesh_instance;");
            source.AppendLine();

            // Merged wave info: this wave's ES vertex count in bits 0-7, its GS primitive
            // count in bits 8-15, the GS wave id in bits 16-23, this wave's subgroup id in
            // bits 24-27 and the workgroup's wave count in bits 28-31. The guest rebuilds
            // EXEC from the first pair, so a wrong value either kills every lane or leaves
            // a waterfall loop waiting on lanes that do not exist. Written only when the
            // register is not also user data for this program.
            if (IsMeshSystemScalar(Gen5NggExports.MergedWaveInfoRegister))
            {
                source.AppendLine(
                    $"    s[{Gen5NggExports.MergedWaveInfoRegister}] = sharpemu_mesh_wave_verts |");
                source.AppendLine(
                    $"        (sharpemu_mesh_wave_prims << {Gen5NggExports.MergedWavePrimitiveCountShift}u) |");
                source.AppendLine(
                    $"        (sharpemu_mesh_wave << {Gen5NggExports.MergedWaveGsWaveIdShift}u) |");
                source.AppendLine(
                    $"        (sharpemu_mesh_wave << {Gen5NggExports.MergedWaveSubgroupIdShift}u) |");
                source.AppendLine(
                    $"        ({mesh.WavesPerThreadgroup}u << {Gen5NggExports.MergedWaveSubgroupCountShift}u);");
            }

            // Merged group info: the workgroup's ES vertex count in bits 12-20 and its
            // input primitive count in bits 22-30. The ordered-append id below them stays
            // zero; no primitive shader seen so far consumes it.
            if (IsMeshSystemScalar(Gen5NggExports.MergedGroupInfoRegister))
            {
                source.AppendLine(
                    $"    s[{Gen5NggExports.MergedGroupInfoRegister}] =");
                source.AppendLine(
                    $"        ((sharpemu_mesh_subgroup_verts & 0x{Gen5NggExports.GroupCountMask:X}u) << " +
                    $"{Gen5NggExports.GroupVertexCountShift}u) |");
                source.AppendLine(
                    $"        ((sharpemu_mesh_subgroup_prims & 0x{Gen5NggExports.GroupCountMask:X}u) << " +
                    $"{Gen5NggExports.GroupPrimitiveCountShift}u);");
            }

            // The GS vector inputs pack two 16-bit ES vertex offsets per register
            // (v0 = vertices 0/1, v1 = 2/3, v4 = 4/5), with the primitive id in v2 and the
            // GS invocation id in v3. The ES/VS block follows the GS one, which is why the
            // vertex index lands in v5 and the instance index in v8 — the same registers
            // the per-vertex path seeds.
            if (mesh.HasGeometryHalf)
            {
                // A GS half's offsets name the ES vertices of the lane's input primitive, in
                // the draw's topology, each as the subgroup-local vertex index shifted left by
                // two: the guest extracts bits 2-15 of each half and scales by its own ES
                // stride. An odd strip triangle swaps its first two vertices to keep winding.
                var list = (uint)Gen5MslMeshInputTopology.TriangleList;
                var point = (uint)Gen5MslMeshInputTopology.PointList;
                var strip = (uint)Gen5MslMeshInputTopology.TriangleStrip;
                source.AppendLine(
                    $"    uint sharpemu_gs_v0 = sharpemu_gs_input == {list}u ? sharpemu_tg_index * 3u : sharpemu_tg_index;");
                source.AppendLine(
                    $"    uint sharpemu_gs_v1 = sharpemu_gs_input == {point}u ? 0u : sharpemu_gs_v0 + 1u;");
                source.AppendLine(
                    $"    const uint sharpemu_gs_v2 = sharpemu_gs_input == {point}u ? 0u : sharpemu_gs_v0 + 2u;");
                source.AppendLine(
                    $"    if (sharpemu_gs_input == {strip}u && ((sharpemu_gs_prim_start + sharpemu_tg_index) & 1u) != 0u)");
                source.AppendLine("    {");
                source.AppendLine("        const uint sharpemu_gs_swap = sharpemu_gs_v0;");
                source.AppendLine("        sharpemu_gs_v0 = sharpemu_gs_v1;");
                source.AppendLine("        sharpemu_gs_v1 = sharpemu_gs_swap;");
                source.AppendLine("    }");
                source.AppendLine(
                    $"    v[{Gen5NggExports.GsVertexOffset01Register}] = ((sharpemu_gs_v0 << 2u) & 0xFFFFu) |");
                source.AppendLine("        (((sharpemu_gs_v1 << 2u) & 0xFFFFu) << 16u);");
                source.AppendLine(
                    $"    v[{Gen5NggExports.GsVertexOffset23Register}] = (sharpemu_gs_v2 << 2u) & 0xFFFFu;");
            }
            else
            {
                // Each primitive owns vertices local_prim * N .. +N-1. The guest decodes a vertex
                // as bits 2-15 of each 16-bit field (v0 = vertices 0/1, v1 = vertex 2), so the
                // local index is shifted left by two, as in the GS half above.
                var perPrimitive = Math.Max(mesh.VerticesPerPrimitive, 1u);
                source.AppendLine(
                    $"    const uint sharpemu_mesh_prim_base = sharpemu_tg_index * {perPrimitive}u;");

                // A meshlet's triangle is three 8-bit local vertex indices; the same 16-bit
                // field encoding applies, so the guest decodes them identically.
                source.AppendLine(
                    "    const uint sharpemu_mesh_prim_packed = sharpemu_meshlets_on && sharpemu_tg_index < (sharpemu_ml.w >> 16u)");
                source.AppendLine(
                    $"        ? {MeshletBufferName}[{MeshDrawInfoName}.meshlet_prim_base + sharpemu_ml.y + sharpemu_tg_index] : 0u;");
                if (_request.PrimitiveShader is { Passthrough: true })
                {
                    // Passthrough mode: v0 is the primitive export itself, one 10-bit field
                    // per vertex, so the program's EXP prim v0 names the lane's own triangle.
                    var stride = Gen5NggExports.PrimitiveIndexStride;
                    source.AppendLine(
                        $"    v[{Gen5NggExports.GsVertexOffset01Register}] = sharpemu_meshlets_on");
                    source.AppendLine(
                        $"        ? ((sharpemu_mesh_prim_packed & 0xFFu) | (((sharpemu_mesh_prim_packed >> 8u) & 0xFFu) << {stride}u) |");
                    source.AppendLine(
                        $"           (((sharpemu_mesh_prim_packed >> 16u) & 0xFFu) << {stride * 2}u))");
                    source.AppendLine(
                        $"        : (sharpemu_mesh_prim_base | ((sharpemu_mesh_prim_base + 1u) << {stride}u) |");
                    source.AppendLine(
                        $"           ((sharpemu_mesh_prim_base + 2u) << {stride * 2}u));");
                    source.AppendLine($"    v[{Gen5NggExports.GsVertexOffset23Register}] = 0u;");
                }
                else
                {
                    source.AppendLine(
                        $"    v[{Gen5NggExports.GsVertexOffset01Register}] = sharpemu_meshlets_on");
                    source.AppendLine(
                        "        ? (((sharpemu_mesh_prim_packed & 0xFFu) << 2u) | (((sharpemu_mesh_prim_packed >> 8u) & 0xFFu) << 18u))");
                    source.AppendLine(
                        "        : (((sharpemu_mesh_prim_base << 2u) & 0xFFFFu) |");
                    source.AppendLine(
                        "           ((((sharpemu_mesh_prim_base + 1u) << 2u) & 0xFFFFu) << 16u));");
                    source.AppendLine(
                        $"    v[{Gen5NggExports.GsVertexOffset23Register}] = sharpemu_meshlets_on");
                    source.AppendLine(
                        "        ? (((sharpemu_mesh_prim_packed >> 16u) & 0xFFu) << 2u)");
                    source.AppendLine(
                        "        : (((sharpemu_mesh_prim_base + 2u) << 2u) & 0xFFFFu);");
                }
            }
            source.AppendLine($"    v[{Gen5NggExports.GsPrimitiveIdRegister}] = sharpemu_mesh_prim_slot;");
            source.AppendLine($"    v[{Gen5NggExports.GsInvocationIdRegister}] = 0u;");
            source.AppendLine($"    v[{Gen5NggExports.GsVertexOffset45Register}] = 0u;");
            source.AppendLine(
                $"    v[{Gen5NggExports.EsVertexIndexRegister}] = sharpemu_mesh_vertex_index;");
            source.AppendLine(
                $"    v[{Gen5NggExports.EsInstanceIndexRegister}] = sharpemu_mesh_instance_index;");
        }

        // The dispatcher loop. Unlike the other stages the loop body is the barrier point:
        // see the file header for why S_BARRIER cannot be emitted where it appears.
        //
        // A wave runs its blocks back to back until it parks at an S_BARRIER or ends, and
        // only then meets the rest of the workgroup. The pc is wave-uniform, so the inner
        // loop keeps a simdgroup converged, and nothing but S_BARRIER orders two waves in
        // hardware either. Meeting after every block instead cost two threadgroup barriers
        // and three atomics per block: SILENT HILL's cube-shadow primitive shader has no
        // S_BARRIER at all and still paid that on each of its blocks, which made its shadow
        // passes half of the title's GPU time.
        private void EmitMeshDispatcher(StringBuilder source, string body)
        {
            source.AppendLine("    for (;;)");
            source.AppendLine("    {");
            source.AppendLine("        while (active && !sharpemu_mesh_wait)");
            source.AppendLine("        {");
            source.AppendLine("            switch (pc)");
            source.AppendLine("            {");
            source.Append(body);
            source.AppendLine("            default:");
            source.AppendLine("                active = false;");
            source.AppendLine("                break;");
            source.AppendLine("            }");
            if (MaxDispatcherSteps > 0)
            {
                source.AppendLine($"            if (++steps >= {MaxDispatcherSteps}u)");
                source.AppendLine("            {");
                source.AppendLine("                active = false;");
                source.AppendLine($"                {StepCapReportCode}");
                source.AppendLine("            }");
            }

            source.AppendLine("        }");
            source.AppendLine();

            // Workgroup rendezvous: every thread reaches it every iteration, so it is the
            // one convergent point in the program. A parked wave resumes only once every
            // thread that is still running has parked too, and a thread that has ended
            // stops being counted, which is how S_BARRIER behaves in hardware.
            source.AppendLine("        if (sharpemu_tg_index == 0u)");
            source.AppendLine("        {");
            source.AppendLine("            atomic_store_explicit(&sharpemu_mesh_live, 0u, memory_order_relaxed);");
            source.AppendLine("            atomic_store_explicit(&sharpemu_mesh_parked, 0u, memory_order_relaxed);");
            source.AppendLine("        }");
            source.AppendLine();
            source.AppendLine($"        {MeshWorkgroupBarrier}");
            source.AppendLine("        if (active)");
            source.AppendLine("        {");
            source.AppendLine("            atomic_fetch_add_explicit(&sharpemu_mesh_live, 1u, memory_order_relaxed);");
            source.AppendLine("            if (sharpemu_mesh_wait)");
            source.AppendLine("            {");
            source.AppendLine("                atomic_fetch_add_explicit(&sharpemu_mesh_parked, 1u, memory_order_relaxed);");
            source.AppendLine("            }");
            source.AppendLine("        }");
            source.AppendLine();
            source.AppendLine("        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);");
            source.AppendLine(
                "        const uint sharpemu_mesh_live_count = atomic_load_explicit(&sharpemu_mesh_live, memory_order_relaxed);");
            source.AppendLine("        if (sharpemu_mesh_live_count == 0u)");
            source.AppendLine("        {");
            source.AppendLine("            break;");
            source.AppendLine("        }");
            source.AppendLine();
            source.AppendLine(
                "        if (atomic_load_explicit(&sharpemu_mesh_parked, memory_order_relaxed) == sharpemu_mesh_live_count)");
            source.AppendLine("        {");
            source.AppendLine("            sharpemu_mesh_wait = false;");
            source.AppendLine("        }");
            source.AppendLine("    }");
        }

        // Publish the subgroup's output. Exports accumulate into per-thread state during
        // the dispatcher, so the mesh writes happen once, after the whole workgroup has
        // finished and the allocation request is visible to every thread.
        private void EmitMeshEpilogue(StringBuilder source)
        {
            var mesh = Mesh;
            if (MeshHasPrimitiveData)
            {
                source.AppendLine($"    if (sharpemu_tg_index < {mesh.MaxVertices}u)");
                source.AppendLine("    {");
                if (MeshWritesPrimitiveOutputs)
                {
                    source.AppendLine(
                        "        sharpemu_mesh_vertex_target[sharpemu_tg_index] = " +
                        "uint2(sharpemu_mesh_layer, sharpemu_mesh_viewport);");
                }

                if (MeshPerVertexLocations.Any())
                {
                    var clipW = _request.ClipSpace.WReciprocal
                        ? "1.0f / sharpemu_out.sharpemu_position.w"
                        : "sharpemu_out.sharpemu_position.w";
                    source.AppendLine($"        sharpemu_mesh_vertex_w[sharpemu_tg_index] = {clipW};");
                }

                foreach (var location in MeshPerVertexLocations)
                {
                    var value = _vertexOutputs.Contains(location) ? $"sharpemu_out.param{location}" : "float4(0.0f)";
                    source.AppendLine($"        sharpemu_mesh_param{location}[sharpemu_tg_index] = {value};");
                }

                source.AppendLine("    }");
                source.AppendLine();
            }

            source.AppendLine($"    {MeshWorkgroupBarrier}");
            if (UsesDeviceLds)
            {
                // Every lane is past its last LDS access: hand the slot back. The fence
                // orders this threadgroup's LDS stores before the flag that lets the next
                // holder overwrite them.
                source.AppendLine("    if (sharpemu_tg_index == 0u)");
                source.AppendLine("    {");
                source.AppendLine(
                    "        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);");
                source.AppendLine(
                    "        atomic_store_explicit(&sharpemu_lds_pool[1u + sharpemu_lds_slot], 0u, memory_order_relaxed);");
                source.AppendLine("    }");
            }

            // Only the lanes inside this subgroup's allocation export: the counts are per draw
            // (the last subgroup, or a meshlet, holds fewer than the declared extent).
            source.AppendLine(
                $"    const uint sharpemu_mesh_out_verts = min(sharpemu_mesh_alloc[0], {mesh.MaxVertices}u);");
            source.AppendLine(
                $"    const uint sharpemu_mesh_out_prims = min(sharpemu_mesh_alloc[1], {mesh.MaxPrimitives}u);");
            source.AppendLine("    if (sharpemu_tg_index < sharpemu_mesh_out_verts)");
            source.AppendLine("    {");

            // A partial POS0 export keeps the components already written, so the position
            // reaches Metal's clip space once, when it is final; the plane distances are
            // taken from the guest's position first.
            foreach (var statement in HostPlaneStatements("sharpemu_out.sharpemu_position", "sharpemu_out"))
            {
                source.AppendLine($"        {statement}");
            }

            foreach (var statement in HostClipSpaceStatements("sharpemu_out.sharpemu_position"))
            {
                source.AppendLine($"        {statement}");
            }

            source.AppendLine($"        {MeshOutputName}.set_vertex(sharpemu_tg_index, sharpemu_out);");
            source.AppendLine("    }");
            source.AppendLine();
            source.AppendLine("    if (sharpemu_tg_index < sharpemu_mesh_out_prims)");
            source.AppendLine("    {");

            // A culled or never-exported primitive becomes degenerate: Metal has no
            // per-primitive discard, and repeating one index rasterizes nothing.
            source.AppendLine(
                "        const uint3 indices = sharpemu_mesh_culled ? uint3(0u) : sharpemu_mesh_prim;");
            var perPrimitive = Math.Max(mesh.VerticesPerPrimitive, 1u);
            source.AppendLine($"        const uint base = sharpemu_tg_index * {perPrimitive}u;");
            for (var component = 0u; component < perPrimitive; component++)
            {
                source.AppendLine(
                    $"        {MeshOutputName}.set_index(base + {component}u, indices[{Math.Min(component, 2u)}]);");
            }

            if (MeshHasPrimitiveData)
            {
                source.AppendLine($"        {MeshPrimitiveStructName} primitive = {{}};");
                if (MeshWritesPrimitiveOutputs)
                {
                    source.AppendLine(
                        $"        const uint2 target = sharpemu_mesh_vertex_target[min(indices[0], {mesh.MaxVertices - 1}u)];");
                }

                if (_writesLayer)
                {
                    source.AppendLine("        primitive.sharpemu_layer = target.x;");
                }

                if (_writesViewportIndex)
                {
                    source.AppendLine("        primitive.sharpemu_viewport = target.y;");
                }

                // P0, P1 and P2 of the guest's V_INTERP: the values of the vertices in the
                // order the primitive names them, which is also the order of Metal's
                // [[barycentric_coord]] components for a mesh triangle.
                if (MeshPerVertexLocations.Any())
                {
                    source.AppendLine("        primitive.sharpemu_vertex_w = float3(" +
                        string.Join(", ", Enumerable.Range(0, 3).Select(vertex =>
                            $"sharpemu_mesh_vertex_w[min(indices[{vertex}], {mesh.MaxVertices - 1}u)]")) + ");");
                }
                foreach (var location in MeshPerVertexLocations)
                {
                    for (var vertex = 0u; vertex < 3u; vertex++)
                    {
                        source.AppendLine(
                            $"        primitive.param{location}_v{vertex} = " +
                            $"sharpemu_mesh_param{location}[min(indices[{vertex}], {mesh.MaxVertices - 1}u)];");
                    }
                }

                source.AppendLine($"        {MeshOutputName}.set_primitive(sharpemu_tg_index, primitive);");
            }

            source.AppendLine("    }");
            source.AppendLine();
            source.AppendLine("    if (sharpemu_tg_index == 0u)");
            source.AppendLine("    {");
            source.AppendLine(
                $"        {MeshOutputName}.set_primitive_count(min(sharpemu_mesh_alloc[1], {mesh.MaxPrimitives}u));");
            source.AppendLine("    }");
        }

        // ---- instructions ----

        /// <summary>S_SENDMSG GS_ALLOC_REQ reserves the subgroup's export space. M0 carries
        /// the vertex count in bits 0-11 and the primitive count in bits 12-22; the request
        /// is wave-uniform and hardware takes it from the first wave, so the first thread
        /// of the workgroup publishes it for the epilogue.</summary>
        private void EmitMeshAllocationRequest()
        {
            var mesh = Mesh;
            var m0 = $"s[{M0Register}]";
            Line("if (sharpemu_tg_index == 0u)");
            Line("{");
            _indent++;
            Line($"sharpemu_mesh_alloc[0] = min((({m0} >> {Gen5NggExports.AllocationVertexCountShift}u) & " +
                $"0x{Gen5NggExports.AllocationVertexCountMask:X}u), {mesh.MaxVertices}u);");
            Line($"sharpemu_mesh_alloc[1] = min((({m0} >> {Gen5NggExports.AllocationPrimitiveCountShift}u) & " +
                $"0x{Gen5NggExports.AllocationPrimitiveCountMask:X}u), {mesh.MaxPrimitives}u);");
            _indent--;
            Line("}");
        }

        /// <summary>The mesh stage's EXP: positions and parameters land in this lane's
        /// vertex, target 20 decodes the packed primitive connectivity into this lane's
        /// primitive. A lane EXEC removed it keeps the previous value, like the other
        /// stages.</summary>
        private bool TryEmitMeshExport(
            Gen5ShaderInstruction instruction,
            Gen5ExportControl export)
        {
            if (export.Target == Gen5NggExports.PrimitiveTarget)
            {
                EmitMeshPrimitiveExport(instruction, export);
                return true;
            }

            if (export.Target is > Gen5NggExports.FirstPositionTarget and < Gen5NggExports.FirstPositionTarget + 4)
            {
                // POS1..POS3: point size, clip and cull distances land in this lane's
                // vertex; the layer and viewport index wait for the epilogue's primitive.
                EmitAuxPositionExport(instruction, export);
                return true;
            }

            string field;
            if (export.Target == Gen5NggExports.FirstPositionTarget)
            {
                field = "sharpemu_out.sharpemu_position";
            }
            else if (export.Target >= Gen5NggExports.FirstParameterTarget &&
                     export.Target < Gen5NggExports.FirstParameterTarget + Gen5NggExports.ParameterTargetCount &&
                     _vertexOutputs.Contains(export.Target - Gen5NggExports.FirstParameterTarget))
            {
                field = $"sharpemu_out.param{export.Target - Gen5NggExports.FirstParameterTarget}";
            }
            else
            {
                // MRTZ never reaches this stage.
                return true;
            }

            var values = new string[4];
            for (var component = 0; component < 4; component++)
            {
                if ((export.EnableMask & (1u << component)) == 0)
                {
                    values[component] = $"{field}[{component}]";
                    continue;
                }

                if (export.Compressed)
                {
                    var packed = $"v[{instruction.Sources[component >> 1].Value}]";
                    values[component] = $"(float)as_type<half2>({packed})[{component & 1}]";
                    continue;
                }

                values[component] = $"as_type<float>(v[{instruction.Sources[component].Value}])";
            }

            Line(ExecGuarded($"{field} = float4({values[0]}, {values[1]}, {values[2]}, {values[3]});"));
            if (export.Target == Gen5NggExports.FirstPositionTarget)
            {
                EmitInvalidPositionDistance();
            }
            else
            {
                EmitInterpolantExports(export.Target - Gen5NggExports.FirstParameterTarget, field);
            }

            return true;
        }

        // Target 20 exports one dword of primitive connectivity: a 9-bit subgroup-local
        // vertex index per 10-bit field (the field's top bit is that vertex's edge flag)
        // and a null-primitive marker in bit 31. The exporting lane owns one primitive.
        private void EmitMeshPrimitiveExport(
            Gen5ShaderInstruction instruction,
            Gen5ExportControl export)
        {
            var packed = (export.EnableMask & 1u) != 0
                ? $"v[{instruction.Sources[0].Value}]"
                : "0u";
            var value = Temp("uint", packed);
            BeginExecGuard();
            var perPrimitive = Math.Max(Mesh.VerticesPerPrimitive, 1u);
            for (var vertex = 0u; vertex < perPrimitive; vertex++)
            {
                var shift = vertex * Gen5NggExports.PrimitiveIndexStride;
                Line($"sharpemu_mesh_prim[{vertex}] = ({value} >> {shift}u) & " +
                    $"0x{Gen5NggExports.PrimitiveIndexMask:X}u;");
            }

            for (var vertex = perPrimitive; vertex < 3u; vertex++)
            {
                Line($"sharpemu_mesh_prim[{vertex}] = sharpemu_mesh_prim[{perPrimitive - 1}];");
            }

            Line($"sharpemu_mesh_culled = (({value} >> {Gen5NggExports.PrimitiveNullBit}u) & 1u) != 0u;");
            EndExecGuard();
        }
    }
}
