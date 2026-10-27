// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;

namespace SharpEmu.ShaderCompiler.Metal;

// MSL-specific shader artifact types. These stay beside the MSL emitter (not in the
// backend-neutral SharpEmu.ShaderCompiler project): each codegen owns its own
// compiled-shader shape, mirroring Gen5SpirvShader on the Vulkan side.
public enum Gen5MslStage
{
    Vertex,
    Pixel,
    Compute,

    // An NGG primitive shader (the merged ES/GS hardware stage) emitted as an MSL
    // [[mesh]] function. One Metal threadgroup is one NGG subgroup.
    Mesh,
}

public enum Gen5MslMeshTopology
{
    Point,
    Line,
    Triangle,
}

// The draw topology a fused program's GS half assembles its input primitives from. The
// values are what the host writes into the draw-info block's gs_input_topology.
public enum Gen5MslMeshInputTopology : uint
{
    TriangleList = 0,
    TriangleStrip = 1,
    PointList = 2,
}

// Everything the render host needs to build an MTLMeshRenderPipelineDescriptor around a
// translated primitive shader and to launch it, plus the buffer slots the mesh entry reads
// its NGG system values from.
public sealed record Gen5MslMeshInfo(
    uint ThreadsPerThreadgroup,
    uint WavesPerThreadgroup,
    uint MaxVertices,
    uint MaxPrimitives,
    uint VerticesPerPrimitive,
    Gen5MslMeshTopology Topology,
    uint ThreadgroupMemoryBytes,
    uint DrawInfoBufferIndex,
    uint IndexBufferIndex,
    bool ReadsIndexBuffer,
    uint GeometryVerticesOut = 0,
    bool NarrowWave64 = false,
    bool DeviceLds = false,
    bool Wave64 = false,
    uint LdsBytes = 0,
    bool Amplified = false)
{
    /// <summary>Lanes in one guest wave of this program: an amplified wave64 subgroup runs
    /// its 64-lane waves as two bridged simdgroups each; everything else runs a guest wave
    /// as one simdgroup (<see cref="WaveLanes"/>).</summary>
    public uint GuestWaveLanes => Amplified && Wave64 ? 64u : WaveLanes;

    /// <summary>The mesh slot the device-memory LDS pool is bound at when
    /// <see cref="DeviceLds"/> is set.</summary>
    public const uint LdsPoolBufferIndex = 5;

    /// <summary>Slots in the device LDS pool; a threadgroup holds one for its lifetime.</summary>
    public const uint LdsPoolSlots = 2048;

    /// <summary>Dwords of the pool header: a probe counter and one busy flag per slot.</summary>
    public const uint LdsPoolHeaderDwords = 4096;

    /// <summary>Dwords of one pool slot, the most guest LDS a mesh stage declares.</summary>
    public const uint LdsPoolSlotDwords = 4096;

    /// <summary>Bytes of the whole pool: the zeroed header and every slot's data.</summary>
    public const ulong LdsPoolByteSize =
        (LdsPoolHeaderDwords + (ulong)LdsPoolSlots * LdsPoolSlotDwords) * sizeof(uint);

    /// <summary>Primitives one subgroup of a program without a GS half holds. Every
    /// primitive owns its vertices (no reuse), so a subgroup is whole primitives: as many as
    /// fit both the primitive extent and the vertex extent. The host splits the draw by this
    /// number and the mesh prologue maps a lane's slot from the same one.</summary>
    public uint PrimitivesPerSubgroup =>
        Math.Max(Math.Min(MaxPrimitives, MaxVertices / Math.Max(VerticesPerPrimitive, 1)), 1);

    /// <summary>Vertices one subgroup of a program without a GS half holds: three own
    /// vertices per triangle, so a subgroup never starts or ends mid-primitive.</summary>
    public uint VerticesPerSubgroup => PrimitivesPerSubgroup * Math.Max(VerticesPerPrimitive, 1);

    /// <summary>Whether the program has a GS half that assembles its input primitives
    /// from the draw's topology; see <see cref="InputPrimitivesPerSubgroup"/>.</summary>
    public bool HasGeometryHalf => GeometryVerticesOut != 0;

    /// <summary>
    /// The input primitives one subgroup of a fused (ES+GS) draw takes. Every lane is one
    /// ES vertex and one GS primitive, so the input vertices the topology needs must fit
    /// the vertex extent, and what the GS half can emit for them - up to
    /// <see cref="GeometryVerticesOut"/> vertices per primitive, as a strip - must fit the
    /// mesh output. The host splits the draw into subgroups of this size and the mesh
    /// prologue assembles the GS vertex offsets from the same number.
    /// </summary>
    public uint InputPrimitivesPerSubgroup(Gen5MslMeshInputTopology input)
    {
        // A wave64 GS half numbers its ES vertices, GS primitives and emitted vertices as
        // (merged-wave subgroup id << 6) + lane. A mesh wave is 32 lanes, so the second
        // wave's lanes land at 64 + lane while the offsets the prologue packs count on
        // from 32: a subgroup has to stay inside the first wave to line up.
        var vertices = Math.Min(MaxVertices, WaveLanes);
        var primitives = Math.Min(MaxPrimitives, WaveLanes);
        var byInput = input switch
        {
            Gen5MslMeshInputTopology.TriangleStrip => vertices > 2 ? vertices - 2 : 1,
            Gen5MslMeshInputTopology.PointList => vertices,
            _ => Math.Max(vertices / 3, 1),
        };
        var emitted = Math.Max(GeometryVerticesOut, 1);
        var byOutputVertices = Math.Max(vertices / emitted, 1);
        var byOutputPrimitives = Math.Max(primitives / (emitted > 2 ? emitted - 2 : 1), 1);
        return Math.Min(Math.Min(byInput, primitives), Math.Min(byOutputVertices, byOutputPrimitives));
    }

    /// <summary>Lanes in one mesh wave (one Apple simdgroup).</summary>
    public const uint WaveLanes = 32;

    /// <summary>The input primitives a fused draw of <paramref name="count"/> vertices or
    /// indices assembles.</summary>
    public static uint InputPrimitiveCount(Gen5MslMeshInputTopology input, uint count) => input switch
    {
        Gen5MslMeshInputTopology.TriangleStrip => count > 2 ? count - 2 : 0,
        Gen5MslMeshInputTopology.PointList => count,
        _ => count / 3,
    };

    /// <summary>Byte size of the Gen5MeshDrawInfo block the host fills at
    /// <see cref="DrawInfoBufferIndex"/>; see the generated struct of the same name.</summary>
    public const uint DrawInfoByteSize = 80;

    /// <summary>The mesh slot of the cached-meshlet buffer a program without a GS half reads
    /// when the draw-info block's meshlet_count is not zero; bound (to the null buffer when
    /// off) for every such draw.</summary>
    public const uint MeshletBufferIndex = 4;

    /// <summary>Dword count of that block, for a host that fills it as words.</summary>
    public const uint DrawInfoDwordCount = DrawInfoByteSize / 4;
}

// Compiled source, stage interface counts, and the argument-buffer layout.
public sealed record Gen5MslShader(
    string Source,
    string EntryPoint,
    Gen5MslStage Stage,
    uint AttributeCount,
    uint ThreadgroupSizeX = 1,
    uint ThreadgroupSizeY = 1,
    uint ThreadgroupSizeZ = 1,
    Gen5MslArgumentLayout? ArgumentLayout = null,
    Gen5MslMeshInfo? MeshInfo = null,
    bool WritesLayer = false,
    bool WritesViewportIndex = false)
{
    /// <summary>MSL 4.0 (macOS 26), not the backend's usual version, compiles this source:
    /// a pixel shader declares metal::vertex_value inputs, which only that version has.</summary>
    public const nuint PerVertexInputsLanguageVersion = (4 << 16) | 0;

    /// <summary>The source declares per-vertex pixel inputs; see <see cref="PerVertexInputsLanguageVersion"/>.</summary>
    public bool UsesPerVertexInputs { get; init; }

    /// <summary>The function constant that switches the GET_LOD_STATS counter sites on.</summary>
    public const string MipStatisticsFunctionConstantName = "sharpemu_fc_mip_stats";

    /// <summary>Its index (<c>[[function_constant(n)]]</c>).</summary>
    public const uint MipStatisticsFunctionConstantIndex = 0;

    /// <summary>The source declares <see cref="MipStatisticsFunctionConstantName"/>: its counter
    /// sites run only in a function the host specialized with the constant true, and the
    /// function must be made with a value for it (<c>newFunctionWithName:constantValues:error:</c>).</summary>
    public bool UsesMipStatisticsConstant { get; init; }

    /// <summary>The vertex indices <see cref="ConstantPositions"/> covers, from zero.</summary>
    public const int ConstantPositionVertexCount = 6;

    /// <summary>A vertex program whose clip position depends on nothing but the vertex
    /// index: the host clip-space position (x, y, z, w) of vertex indices 0 to
    /// <see cref="ConstantPositionVertexCount"/> - 1, four floats each. Null otherwise.</summary>
    public float[]? ConstantPositions { get; init; }

    /// <summary>A pixel program that provably never discards a fragment: every fragment
    /// it runs for writes its colour outputs.</summary>
    public bool CannotDiscard { get; init; }

    /// <summary>The [[color(n)]] attachments a pixel program returns a value for, one bit
    /// per n.</summary>
    public uint ColorOutputMask { get; init; }

    /// <summary>A compute kernel whose groups drain a shared queue and never read their own
    /// workgroup ID (see <see cref="Ir.Gen5PersistentThreadsAnalysis"/>): any number of groups
    /// computes the same result, so a host may launch only as many as it keeps resident.</summary>
    public bool PersistentThreads { get; init; }

    /// <summary>A tessellated draw's stage (<see cref="Gen5TessellationLaunch"/>): for the domain
    /// role the source is a post-tessellation vertex function, and the host's pipeline takes its
    /// partition, winding and maximum factor from here.</summary>
    public Gen5TessellationLaunch? Tessellation { get; init; }

    /// <summary>The words the flattened resource table bound to this program must have: the
    /// source reads the fixed slots below this without a bounds check, so a host binds a
    /// shorter table zero-padded to it (see <see cref="Gen5MslCodegenOptions.ConstantTableSlots"/>).</summary>
    public uint FlattenedTableMinimumWords { get; init; }
}
