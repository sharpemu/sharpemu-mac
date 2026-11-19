// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Metal;

/// <summary>
/// How a draw of a primitive shader is launched: the subgroup count and the draw-info words
/// (Gen5MeshDrawInfo) of a draw of whole subgroups, and a
/// compute kernel that writes the very same words and the launch grid from a guest indirect
/// record on the GPU, so a draw whose counts a GPU culling pass just wrote need not wait for
/// them on the CPU.
/// </summary>
/// <remarks>
/// The kernel writes one launch slot per thread. A slot is <see cref="SlotBytes"/> bytes:
/// the draw-info block at 0 (or, for a non-indexed rectangle list, Gen5RectDraw), the grid at
/// <see cref="GridByteOffset"/> - MTLDrawMeshThreadgroupsIndirectArguments, or
/// MTLDrawPatchIndirectArguments for a rectangle list - and the parameters the host fills at
/// <see cref="ParametersByteOffset"/>. Buffer 0 holds the guest records, buffer 1 the slots.
/// </remarks>
public static class Gen5MslMeshLaunch
{
    /// <summary>Bytes of one launch slot; also its alignment.</summary>
    public const uint SlotBytes = 256;

    /// <summary>Byte offset of the launch grid in a slot, just past the draw-info block.</summary>
    public const uint GridByteOffset = Gen5MslMeshInfo.DrawInfoByteSize;

    /// <summary>Byte offset of the parameters in a slot.</summary>
    public const uint ParametersByteOffset = 128;

    /// <summary>Dwords of the parameters (see <see cref="WriteMeshParameters"/>).</summary>
    public const int ParameterDwords = 16;

    /// <summary>The kernel's function name in <see cref="KernelSource"/>.</summary>
    public const string KernelFunction = "sharpemu_mesh_indirect";

    /// <summary>
    /// The subgroups a draw of <paramref name="count"/> vertices or indices launches per
    /// instance: whole primitives per subgroup without a GS half, or the draw's input
    /// primitives split by <see cref="Gen5MslMeshInfo.InputPrimitivesPerSubgroup"/> with one.
    /// </summary>
    public static uint SubgroupCount(Gen5MslMeshInfo mesh, Gen5MslMeshInputTopology input, uint count)
    {
        if (mesh.Amplified)
        {
            // An amplified subgroup takes one input vertex: one workgroup per point.
            return count;
        }

        if (mesh.HasGeometryHalf)
        {
            // A GS half assembles its input primitives from the draw's own topology, so the
            // draw splits by input primitives.
            var inputPrimitivesPerSubgroup = mesh.InputPrimitivesPerSubgroup(input);
            var inputPrimitives = Gen5MslMeshInfo.InputPrimitiveCount(input, count);
            return (inputPrimitives + inputPrimitivesPerSubgroup - 1) / inputPrimitivesPerSubgroup;
        }

        // Without a GS half every primitive owns its vertices, so a subgroup is whole
        // primitives: the prologue maps slot -> index position from the same counts.
        var primitives = count / Math.Max(mesh.VerticesPerPrimitive, 1);
        var primitivesPerSubgroup = mesh.PrimitivesPerSubgroup;
        return (primitives + primitivesPerSubgroup - 1) / primitivesPerSubgroup;
    }

    /// <summary>
    /// Writes the draw-info words of a draw of whole subgroups (no cached meshlets: words
    /// 16-18 stay zero). <paramref name="words"/> is <see cref="Gen5MslMeshInfo.DrawInfoDwordCount"/>
    /// dwords the caller has cleared. <paramref name="indexType"/> is 1 for 16-bit and 2 for
    /// 32-bit indices; a non-indexed draw has no first index and no base vertex.
    /// </summary>
    public static void FillDrawInfo(
        Gen5MslMeshInfo mesh,
        Gen5MslMeshInputTopology input,
        Span<uint> words,
        uint count,
        uint instanceCount,
        bool indexed,
        uint indexType,
        uint firstIndex,
        int baseVertex,
        uint firstInstance)
    {
        words[0] = VerticesPerSubgroup(mesh);
        words[1] = PrimitivesPerSubgroup(mesh);
        words[2] = indexed ? count : 0;
        words[3] = indexed ? indexType : 0;
        words[4] = indexed ? firstIndex : 0;
        // An amplified draw's workgroup index is its point's vertex index, so a non-indexed
        // one counts on from the first vertex.
        words[5] = mesh.Amplified && !indexed ? firstIndex : unchecked((uint)baseVertex);
        words[6] = instanceCount;
        words[7] = firstInstance;
        // A GS half learns where each subgroup's share of input primitives starts and how
        // many vertices the draw has.
        words[13] = (uint)input;
        words[14] = mesh.HasGeometryHalf ? mesh.InputPrimitivesPerSubgroup(input) : 0;
        words[15] = count;
    }

    // A subgroup of a GS half has the program's whole budget; one without a GS half holds
    // whole primitives.
    private static uint VerticesPerSubgroup(Gen5MslMeshInfo mesh) =>
        mesh.HasGeometryHalf ? Math.Max(mesh.MaxVertices, 1) : mesh.VerticesPerSubgroup;

    private static uint PrimitivesPerSubgroup(Gen5MslMeshInfo mesh) =>
        mesh.HasGeometryHalf ? Math.Max(mesh.MaxPrimitives, 1) : mesh.PrimitivesPerSubgroup;

    /// <summary>
    /// The parameters of one mesh launch slot: <paramref name="argumentsOffset"/> is the byte
    /// offset of the guest record in buffer 0, (indexCount, instanceCount, firstIndex,
    /// vertexOffset, firstInstance) when <paramref name="indexed"/>, else (vertexCount,
    /// instanceCount, firstVertex, firstInstance). An indexed count is clamped to
    /// <paramref name="indexLimit"/>, as the bound index range clamps it on the CPU. A launch
    /// past <paramref name="maxThreadgroups"/> threadgroups is written as an empty grid.
    /// </summary>
    public static void WriteMeshParameters(
        Span<uint> parameters,
        Gen5MslMeshInfo mesh,
        Gen5MslMeshInputTopology input,
        ulong argumentsOffset,
        bool indexed,
        uint indexType,
        uint indexLimit,
        ulong maxThreadgroups)
    {
        parameters[..ParameterDwords].Clear();
        parameters[0] = unchecked((uint)argumentsOffset);
        parameters[1] = (uint)(argumentsOffset >> 32);
        parameters[2] = 0; // A mesh launch.
        parameters[3] = indexed ? 1u : 0u;
        parameters[4] = indexType;
        parameters[5] = indexLimit;
        parameters[6] = mesh.HasGeometryHalf ? 1u : 0u;
        parameters[7] = (uint)input;
        parameters[8] = VerticesPerSubgroup(mesh);
        parameters[9] = PrimitivesPerSubgroup(mesh);
        parameters[10] = Math.Max(mesh.VerticesPerPrimitive, 1);
        parameters[11] = mesh.HasGeometryHalf ? mesh.InputPrimitivesPerSubgroup(input) : 0;
        parameters[12] = unchecked((uint)maxThreadgroups);
        parameters[13] = (uint)(maxThreadgroups >> 32);
        parameters[14] = mesh.Amplified ? 1u : 0u;
    }

    /// <summary>
    /// The parameters of one non-indexed rectangle-list slot: the kernel writes Gen5RectDraw
    /// (vertex_base = base_vertex = firstVertex) and MTLDrawPatchIndirectArguments with one patch per
    /// rectangle, as the CPU draw does.
    /// </summary>
    public static void WriteRectangleParameters(Span<uint> parameters, ulong argumentsOffset, ulong maxThreadgroups)
    {
        parameters[..ParameterDwords].Clear();
        parameters[0] = unchecked((uint)argumentsOffset);
        parameters[1] = (uint)(argumentsOffset >> 32);
        parameters[2] = 1; // A rectangle list.
        parameters[12] = unchecked((uint)maxThreadgroups);
        parameters[13] = (uint)(maxThreadgroups >> 32);
    }

    /// <summary>
    /// One thread per launch slot. The arithmetic is <see cref="SubgroupCount"/> and
    /// <see cref="FillDrawInfo"/> in 32-bit unsigned integers, so it wraps where they do.
    /// </summary>
    public static readonly string KernelSource = $$"""
        #include <metal_stdlib>
        using namespace metal;

        struct sharpemu_mesh_launch_params
        {
            uint arguments_offset_low;
            uint arguments_offset_high;
            uint rectangle_list;
            uint indexed;
            uint index_type;
            uint index_limit;
            uint has_geometry_half;
            uint input_topology;
            uint vertices_per_subgroup;
            uint primitives_per_subgroup;
            uint vertices_per_primitive;
            uint input_primitives_per_subgroup;
            uint max_threadgroups_low;
            uint max_threadgroups_high;
            uint amplified;
        };

        kernel void {{KernelFunction}}(
            device const uchar* records [[buffer(0)]],
            device uint* slots [[buffer(1)]],
            uint slot_index [[thread_position_in_grid]])
        {
            device uint* block = slots + slot_index * {{SlotBytes / 4}}u;
            device uint* grid = block + {{GridByteOffset / 4}}u;
            const device sharpemu_mesh_launch_params& p =
                *reinterpret_cast<const device sharpemu_mesh_launch_params*>(block + {{ParametersByteOffset / 4}}u);
            const ulong offset = ulong(p.arguments_offset_low) | (ulong(p.arguments_offset_high) << 32);
            const device uint* arguments = reinterpret_cast<const device uint*>(records + offset);
            const ulong limit = ulong(p.max_threadgroups_low) | (ulong(p.max_threadgroups_high) << 32);

            // A non-indexed rectangle list: Gen5RectDraw (vertex_base, vertex_offset,
            // index_type, first_index, base_vertex), then (patchCount, instanceCount, patchStart,
            // baseInstance), one patch per {{Gen5MslRectangleList.VerticesPerRectangle}} vertices.
            if (p.rectangle_list != 0u)
            {
                const uint patches = arguments[1] == 0u ? 0u : arguments[0] / {{Gen5MslRectangleList.VerticesPerRectangle}}u;
                block[0] = arguments[2];
                block[1] = 0u;
                block[2] = 0u;
                block[3] = 0u;
                block[4] = arguments[2];
                grid[0] = ulong(patches) * ulong(arguments[1]) > limit ? 0u : patches;
                grid[1] = arguments[1];
                grid[2] = 0u;
                grid[3] = arguments[3];
                return;
            }

            const uint count = p.indexed != 0u ? min(arguments[0], p.index_limit) : arguments[0];
            const uint instances = arguments[1];
            const uint first_index = p.indexed != 0u ? arguments[2] : 0u;
            const uint base_vertex = p.indexed != 0u ? arguments[3] : 0u;
            const uint first_instance = p.indexed != 0u ? arguments[4] : arguments[3];

            uint subgroups;
            if (p.amplified != 0u)
            {
                subgroups = count;
            }
            else if (p.has_geometry_half != 0u)
            {
                uint input_primitives = count / 3u;
                if (p.input_topology == {{(uint)Gen5MslMeshInputTopology.TriangleStrip}}u)
                {
                    input_primitives = count > 2u ? count - 2u : 0u;
                }
                else if (p.input_topology == {{(uint)Gen5MslMeshInputTopology.PointList}}u)
                {
                    input_primitives = count;
                }

                subgroups = (input_primitives + p.input_primitives_per_subgroup - 1u) / p.input_primitives_per_subgroup;
            }
            else
            {
                const uint primitives = count / p.vertices_per_primitive;
                subgroups = (primitives + p.primitives_per_subgroup - 1u) / p.primitives_per_subgroup;
            }

            if (instances == 0u || ulong(subgroups) * ulong(instances) > limit)
            {
                subgroups = 0u;
            }

            block[0] = p.vertices_per_subgroup;
            block[1] = p.primitives_per_subgroup;
            block[2] = p.indexed != 0u ? count : 0u;
            block[3] = p.indexed != 0u ? p.index_type : 0u;
            block[4] = first_index;
            block[5] = p.amplified != 0u && p.indexed == 0u ? arguments[2] : base_vertex;
            block[6] = instances;
            block[7] = first_instance;
            for (uint word = 8u; word < 13u; word++)
            {
                block[word] = 0u;
            }

            block[13] = p.input_topology;
            block[14] = p.input_primitives_per_subgroup;
            block[15] = count;
            // No cached meshlets: the program runs whole subgroups.
            block[16] = 0u;
            block[17] = 0u;
            block[18] = 0u;
            block[19] = 0u;
            grid[0] = subgroups;
            grid[1] = subgroups == 0u ? 0u : instances;
            grid[2] = 1u;
        }
        """;
}
