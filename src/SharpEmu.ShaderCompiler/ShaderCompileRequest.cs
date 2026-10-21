// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;
using System.Linq;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler;

// One fixed-function vertex attribute a fetch instruction reads instead of its buffer.
public sealed record ShaderVertexInput(
    uint Pc,
    uint Location,
    uint FetchComponentCount,
    uint ComponentCount,
    uint NumberFormat,
    uint DestinationSelect,
    bool PerInstance,
    IReadOnlyList<uint> AliasPcs)
{
    public ShaderVertexInput(
        uint pc,
        uint location,
        uint componentCount,
        uint numberFormat,
        bool perInstance,
        IReadOnlyList<uint> aliasPcs)
        : this(
            pc,
            location,
            componentCount,
            componentCount,
            numberFormat,
            componentCount switch
            {
                1u => 4u,
                2u => 4u | (5u << 3),
                3u => 4u | (5u << 3) | (6u << 6),
                4u => 4u | (5u << 3) | (6u << 6) | (7u << 9),
                _ => 0u,
            },
            perInstance,
            aliasPcs)
    {
    }
}

/// <summary>The guest's depth clip planes a program emulates with clip distances, in the
/// guest's clip space: near at z = 0 (or z = -w for a [-w, w] depth) and far at z = w.</summary>
[Flags]
public enum ShaderDepthPlanes : byte
{
    None = 0,
    NearAtZero = 1,
    NearAtNegativeW = 2,
    Far = 4,
}

public readonly record struct ShaderClipSpaceTransform(
    bool Enabled,
    float ScaleX,
    float ScaleY,
    float OffsetX,
    float OffsetY,
    float HalfExtentX,
    float HalfExtentY)
{
    // The guest's z runs over [-w, w]. The Vulkan host expresses this as pipeline state
    // (depth clip control); a host with a fixed [0, w] clip volume remaps z in the program.
    public bool DepthNegativeOneToOne { get; init; }

    // The host viewport maps +y down the target, as Vulkan's NDC does. A host whose clip
    // space is y-up and which cannot set a negative viewport height negates y instead.
    public bool PositionYDown { get; init; }

    // The guest viewport is mirrored in x (a negative x scale) and the host viewport, which
    // takes no negative width, is not: the program negates x.
    public bool PositionXMirror { get; init; }

    // The guest depth planes the host does not clip (it clamps depth instead) and the program
    // clips with clip distances computed from the exported position.
    public ShaderDepthPlanes DepthPlanes { get; init; }

    // The enabled user clip planes (PA_CL_UCP_n_{X,Y,Z,W}, four floats each): the guest keeps
    // the side where dot(plane, position) >= 0 of its clip-space position. The program
    // clips with clip distances, or with UserClipPlanesCullOnly (UCP_CULL_ONLY_ENA) culls a
    // primitive whose vertices are all outside one plane.
    public float[]? UserClipPlanes { get; init; }
    public bool UserClipPlanesCullOnly { get; init; }

    // The exported x and y (PA_CL_VTE_CNTL.VTX_XY_FMT), or z (VTX_Z_FMT), are already
    // divided by w; the exported w is 1 / w (VTX_W0_FMT clear). The host divides by w and
    // interpolates with it, so the program multiplies the divided ones back and takes the
    // reciprocal of the reciprocal.
    public bool XyPreDivided { get; init; }
    public bool ZPreDivided { get; init; }
    public bool WReciprocal { get; init; }

    // The program puts the guest's depth on the host's viewport range itself:
    // z' = z * DepthScale + w * DepthOffset after the above. It replaces
    // DepthNegativeOneToOne, which is the remap's (0.5, 0.5) case.
    public bool DepthRemap { get; init; }
    public float DepthScale { get; init; }
    public float DepthOffset { get; init; }

    // Conventions only the program can apply: no Vulkan pipeline state carries them.
    public bool HasProgramOnlyConventions =>
        PositionXMirror || DepthPlanes != ShaderDepthPlanes.None || UserClipPlanes is { Length: > 0 } ||
        XyPreDivided || ZPreDivided || WReciprocal || DepthRemap;
}

// One bounded runtime V# table as the emitter sees it: a contiguous run of native buffer
// candidates plus the flattened key mapping that selects among them.
public readonly record struct BufferCandidateTableUse(
    uint FirstCandidate,
    uint CandidateCount,
    uint MappingOffset,
    uint SearchIterations);

public readonly record struct MeshShaderConfiguration(
    uint OutputVertexCapacity,
    uint OutputPrimitiveCapacity,
    uint ProvokingVertex,
    uint InputPrimitiveCountPerWorkgroup,
    uint InputVertexCountPerWorkgroup,
    uint LocalDataShareDwords,
    bool InputTriangleStrip = false,
    uint DeviceSubgroupLaneCount = 0,
    bool InputPointList = false,
    bool InputLineList = false)
{
    public static uint[] ParameterLocations(Gen5ShaderProgram program, int requiredOutputCount) =>
        program.Instructions.Select(instruction => instruction.Control).OfType<Gen5ExportControl>()
            .Where(export => export.Target is >= 32 and < 64).Select(export => export.Target - 32)
            .Concat(Enumerable.Range(0, Math.Max(requiredOutputCount, 0)).Select(location => (uint)location))
            .Distinct().Order().ToArray();

    public static uint[] ParameterLocations(ShaderCompileRequest request) =>
        request.MeshOutputLocationMask is { } mask
            ? Enumerable.Range(0, 32).Where(location => (mask & (1u << location)) != 0)
                .Select(location => (uint)location).ToArray()
            : ParameterLocations(request.Program, request.RequiredVertexOutputCount);

    public ulong OutputMemoryBytes(int parameterCount, uint vertexGranularity, uint primitiveGranularity)
    {
        if (vertexGranularity == 0 || primitiveGranularity == 0)
            throw new ArgumentOutOfRangeException(nameof(vertexGranularity));
        var vertices = ((ulong)OutputVertexCapacity + vertexGranularity - 1) / vertexGranularity * vertexGranularity;
        var primitives = ((ulong)OutputPrimitiveCapacity + primitiveGranularity - 1) / primitiveGranularity * primitiveGranularity;
        // Position and each parameter use one location. Layer and CullPrimitive use one each.
        // Primitive indices do not count toward output storage.
        return checked(((ulong)parameterCount + 1) * 16 * vertices + 32 * primitives);
    }
}

// Everything an emitter needs to compile one permutation of a program: the decoded
// program, its resource plan applied to one specialization, and the binding layout.
public sealed class ShaderCompileRequest
{
    public const uint UnboundedThreadCount = uint.MaxValue;
    public const int WrittenRangeDwordCount = 3;
    public bool TraceDeviceAddressFaults { get; init; }

    public ShaderCompileRequest(ShaderResourcePlan plan, SpecializedResourceInfo resources, BindingLayout bindings)
    {
        Program = plan.Graph.Program;
        Stage = plan.Stage;
        Hash = plan.Hash;
        Memory = plan.Memory;
        Resources = resources;
        Bindings = bindings;
        UserDataBase = plan.UserDataBase;
        UserDataCount = plan.UserDataCount;
        ExcludedUserDataRegisters = plan.Graph.ExcludedUserDataRegisters;
        UsesFlattenedTable = RequiresFlattenedTable(plan, resources);
        UsesGlobalDataShare = BindingLayout.UsesGlobalDataShare(Program);
        ReadsShaderBase = BindingLayout.ReadsShaderBase(Program);

        FlattenedSlotByMemoryIndex = new Dictionary<int, uint>(plan.FlattenedSlotByMemoryIndex);
        FlattenedTableReservedWords = (uint)plan.FlattenedTableReservedCount;
        IndirectKeyMemoryIndices = plan.IndirectImages.Select(access => access.Key.MemoryIndex).ToHashSet();
        IndirectOffsetKeyMemoryIndices = plan.IndirectImages.Where(access => access.KeyIsAddressOffset)
            .Select(access => access.Key.MemoryIndex).ToHashSet();
        IndirectRootByMemoryIndex = plan.IndirectImages.ToDictionary(access => access.MemoryIndex, access => access.Key.MemoryIndex);

        var writtenSlots = new Dictionary<int, uint>();
        foreach (var range in plan.DeviceAddressRanges)
        {
            if (!range.Written || !plan.WrittenRangeSlotByHandle.TryGetValue(range.Handle, out var slot))
            {
                continue;
            }

            foreach (var memoryIndex in range.MemoryIndices)
            {
                writtenSlots[memoryIndex] = slot;
            }
        }

        WrittenRangeSlotByMemoryIndex = writtenSlots;

        var candidateTables = new Dictionary<int, BufferCandidateTableUse>();
        for (var index = 0; index < plan.BufferCandidateTables.Count && index < resources.Info.BufferCandidateTables.Count; index++)
        {
            var info = resources.Info.BufferCandidateTables[index];
            var use = new BufferCandidateTableUse(info.FirstCandidate, info.CandidateCount, info.MappingOffset, info.SearchIterations);
            foreach (var memoryIndex in plan.BufferCandidateTables[index].MemoryIndices)
            {
                candidateTables[memoryIndex] = use;
            }
        }

        BufferCandidateTableByMemoryIndex = candidateTables;
    }

    // The flattened table is bound when host reads, written ranges, indirect mappings or
    // bounded buffer candidate mappings fill it.
    public static bool RequiresFlattenedTable(ShaderResourcePlan plan, SpecializedResourceInfo resources) =>
        plan.TableReads.Count != 0 || plan.WrittenRangeCount != 0 ||
        plan.BufferCandidateTables.Count != 0 ||
        resources.Info.Images.Any(image => image.IndirectSearchIterations != 0);

    public Gen5ShaderProgram Program { get; }
    public ShaderStage Stage { get; }
    public ulong Hash { get; }
    public MemoryAccessTable Memory { get; }
    public SpecializedResourceInfo Resources { get; }
    public BindingLayout Bindings { get; }
    public uint UserDataBase { get; }
    public uint UserDataCount { get; }
    public ulong ExcludedUserDataRegisters { get; }
    public bool UsesFlattenedTable { get; }
    public bool UsesGlobalDataShare { get; }
    public bool ReadsShaderBase { get; }

    // Host-flattened scalar reads: memory index → flattened table slot.
    public IReadOnlyDictionary<int, uint> FlattenedSlotByMemoryIndex { get; }

    // The materialized flattened table always holds at least this many words (the planned table
    // reads and written ranges), so a constant slot below it needs no bounds check.
    public uint FlattenedTableReservedWords { get; }

    // Scalar reads whose loaded dword selects an indirect image at a later instruction.
    public IReadOnlySet<int> IndirectKeyMemoryIndices { get; }

    public IReadOnlySet<int> IndirectOffsetKeyMemoryIndices { get; }

    // Indirect image accesses: memory index → the memory index of the key read.
    public IReadOnlyDictionary<int, int> IndirectRootByMemoryIndex { get; }

    // Bounded runtime V# accesses: memory index → the candidate run and its key mapping.
    public IReadOnlyDictionary<int, BufferCandidateTableUse> BufferCandidateTableByMemoryIndex { get; }

    // Written device-address accesses: memory index → the flattened slot of their range.
    public IReadOnlyDictionary<int, uint> WrittenRangeSlotByMemoryIndex { get; }

    public uint WaveSize { get; init; } = 32;
    // True when the pipeline runs on a 64-lane host subgroup, so one subgroup is one guest Wave64.
    public bool HostWave64Supported { get; init; }
    public bool HostComputeSubgroup32 { get; init; }
    public bool EnableExecGuardElision { get; init; } = true;
    public bool IeeeMode { get; init; }
    // RSRC1 FLOAT_MODE: bits 1:0 f32 round, 3:2 f16/f64 round, 5:4 f32 denormals, 7:6 f16/f64
    // denormals (per field, bit 0 keeps input denormals and bit 1 output ones). 0xC0, the mode
    // titles run with, flushes f32 denormals and keeps f16/f64 ones.
    public uint FloatMode { get; init; } = DefaultFloatMode;
    // RSRC1 DX10_CLAMP: a clamped NaN becomes 0 when set and stays NaN when clear.
    public bool Dx10Clamp { get; init; } = true;
    public const uint DefaultFloatMode = 0xC0;
    // SPI_PS_IN_CONTROL.PS_W32_EN: the guest pixel wave is wave32, so its VCC_HI is an
    // ordinary SGPR rather than the mask of lanes 32..63.
    public bool PixelGuestWave32 { get; init; }
    public uint ScratchDwords { get; init; }
    public bool EnableGraphicsSubgroupOperations { get; init; } = true;

    public bool MipStatisticsSubgroupReduction { get; init; }

    // The device supports 64-bit integer atomics on workgroup memory
    // (VkPhysicalDeviceFeatures.shaderSharedInt64Atomics). When set, the LDS
    // 64-bit atomics are emitted as real 64-bit atomics instead of a pair of
    // 32-bit ones, which is not atomic as a pair.
    public bool SupportsSharedInt64Atomics { get; init; }

    // The device supports 64-bit integer atomics on images (shaderImageInt64Atomics). Without
    // it a program that uses an atomic on a 64-bit texel is rejected at compile time.
    public bool SupportsImageInt64Atomics { get; init; }
    public bool ShaderSignedZeroInfNanPreserveFloat32Supported { get; init; }
    public Gen5ComputeSystemRegisters? ComputeSystemRegisters { get; init; }

    public IReadOnlyList<Gen5PixelOutputBinding> PixelOutputs { get; init; } = [];

    // DB_SHADER_CONTROL.Z_EXPORT_ENABLE with a depth aspect bound: the MRTZ export's
    // first component replaces the fragment depth.
    public bool PixelDepthExportEnable { get; init; }

    // SPI_SHADER_Z_FORMAT: where the MRTZ export carries depth, stencil and the sample mask.
    // Null when the registers name no format; the export's own layout (COMPR or not) decides.
    public Gen5ColorExportFormat? PixelDepthExportFormat { get; init; }

    // DB_SHADER_CONTROL.STENCIL_TEST_VAL_EXPORT_ENABLE with a stencil aspect bound: the MRTZ
    // export replaces the stencil reference value.
    public bool PixelStencilExportEnable { get; init; }

    // DB_SHADER_CONTROL.MASK_EXPORT_ENABLE: the MRTZ export's sample mask is ANDed with the
    // fragment's coverage.
    public bool PixelSampleMaskExportEnable { get; init; }
    public uint PixelInputEnable { get; init; }

    // SPI_BARYC_CNTL.FRONT_FACE_ALL_BITS: FRONT_FACE reads 0xFFFFFFFF/0 instead of +1.0/-1.0.
    public bool PixelFrontFaceAllBits { get; init; }
    public uint PixelCustomInterpolationMask { get; init; }

    // False when the device cannot read one vertex's value of a pixel input (PerVertexKHR).
    // MoltenVK advertises fragment shader barycentrics but cannot translate PerVertexKHR to MSL.
    public bool SupportsPerVertexPixelInputs { get; init; } = true;

    // Set when the paired vertex stage is a mesh function that hands the three vertex values
    // of every parameter in Gen5PixelInputMapping.PerVertexParameterMask to the pixel stage
    // as per-primitive data (MeshPerVertexParameterMask): the pixel program reads them there
    // instead of from vertex_value inputs.
    public bool PixelPerVertexInputsFromPrimitive { get; init; }

    // Set when the pixel program interpolates its per-vertex inputs only by hand, each into
    // one channel (Gen5ManualInterpolation finds the plan): the host interpolates those
    // channels as the paired vertex stage unpacks them (VertexInterpolants), the guest's
    // results are replaced by them, and the slots themselves become flat.
    public bool PixelManualInterpolation { get; init; }
    public uint PixelInputAddress { get; init; }
    public IReadOnlyList<uint>? PixelInputCntl { get; init; }

    // Set when the draw's VGT_SHADER_STAGES_EN selects the NGG primitive-shader path, so the
    // program is the merged ES/GS stage rather than a plain vertex shader. A backend that can
    // run the merged stage as a workgroup (the Metal mesh path) uses it; the ones that cannot
    // keep translating the program as a per-vertex shader, which only works for passthrough
    // NGG programs.
    public Gen5NggStageInfo? PrimitiveShader { get; init; }

    // The parameter exports the paired pixel program reads per vertex
    // (Gen5PixelInputMapping.PerVertexParameterMask) on a host without per-vertex pixel
    // inputs. A primitive shader with any is emitted as a mesh function whose primitives
    // carry the three vertex values of each of them; zero leaves the stage as it is.
    public uint MeshPerVertexParameterMask { get; init; }

    // The host interpolants the paired pixel program reads (Gen5Interpolant.Encoded, see
    // PixelManualInterpolation): the vertex stage writes each one, its param's channels
    // unpacked as that kind, whenever it exports the param.
    public IReadOnlyList<uint> VertexInterpolants { get; init; } = [];

    /// <summary>True when the program cannot be expressed per-vertex: its lanes exchange
    /// data through LDS or synchronize the workgroup, so it needs a real workgroup stage.
    ///
    /// GS_ALLOC_REQ and the primitive export are deliberately not part of the test. Every
    /// NGG program issues them, passthrough ones included, and those already run on the
    /// per-vertex path; cross-lane LDS and S_BARRIER are what that path provably cannot
    /// express, because it gives each invocation private LDS and one logical lane.</summary>
    public static bool RequiresPrimitiveShaderWorkgroup(Gen5ShaderProgram program)
    {
        foreach (var instruction in program.Instructions)
        {
            if (instruction.Opcode is "SBarrier" ||
                (instruction.Control is Gen5DataShareControl { Gds: false } &&
                    instruction.Opcode is not ("DsSwizzleB32" or "DsBpermuteB32")))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>False when the fixed-function vertex-fetch substitution must not be applied
    /// to this program.
    ///
    /// The substitution rewrites the program's own <c>buffer_load_format</c> fetches into
    /// host vertex attributes and drops the scalar loads that produced their V#, because a
    /// vertex stage gets those attributes from a vertex descriptor instead. A primitive
    /// shader compiled as the merged ES/GS workgroup stage has no such input: a mesh
    /// function has no <c>[[stage_in]]</c>, and the host binds no vertex buffers for it. Its
    /// ES part therefore has to keep fetching for itself, V# scalar loads included -
    /// otherwise the resource plan never sees those loads, the fetch has no buffer binding,
    /// and the program cannot be compiled. A passthrough primitive shader that hands its
    /// pixel program per-vertex values (<see cref="MeshPerVertexParameterMask"/>) is a mesh
    /// function too.</summary>
    public static bool AllowsEmbeddedVertexFetch(
        Gen5ShaderProgram program,
        Gen5NggStageInfo? primitiveShader,
        uint meshPerVertexParameterMask = 0) =>
        primitiveShader is null || (meshPerVertexParameterMask == 0 && !RequiresPrimitiveShaderWorkgroup(program));

    public int RequiredVertexOutputCount { get; init; }
    public uint? MeshOutputLocationMask { get; init; }
    public IReadOnlyList<ShaderVertexInput> VertexInputs { get; init; } = [];
    public uint PositionExportControl { get; init; }
    public bool SupportsClipDistance { get; init; } = true;
    public ShaderClipSpaceTransform ClipSpace { get; init; }
    public MeshShaderConfiguration? Mesh { get; init; }

    // The LDS the dispatch allocates (COMPUTE_PGM_RSRC2.LDS_SIZE), 0 when unknown.
    public uint LocalDataShareDwords { get; init; }

    public uint LocalSizeX { get; init; } = 1;
    public uint LocalSizeY { get; init; } = 1;
    public uint LocalSizeZ { get; init; } = 1;

    // Fixed bounds for standalone modules; runtime-limit layouts read the dispatch data.
    public uint ThreadCountX { get; init; } = UnboundedThreadCount;
    public uint ThreadCountY { get; init; } = UnboundedThreadCount;
    public uint ThreadCountZ { get; init; } = UnboundedThreadCount;
}
