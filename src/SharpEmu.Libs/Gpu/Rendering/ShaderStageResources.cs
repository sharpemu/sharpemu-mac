// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.ShaderCompiler.Resources;
using ResourceSnapshot = SharpEmu.ShaderCompiler.Resources.ResourceSnapshot;

namespace SharpEmu.Libs.Gpu.Rendering;

public enum ShaderStageKind
{
    Unknown,
    Vertex,
    Pixel,
    Compute,
    Mesh,
}

public enum ImageResourceClass : byte
{
    None,
    Sampled,
    Storage,
}

// How a compiled program uses one buffer resource.
public readonly record struct BufferResourceInfo(
    bool Read,
    bool Written,
    bool Atomic,
    bool Formatted,
    bool Scalar,
    uint MaxByteExtent,
    uint PackedStride);

// How a compiled program uses one image resource.
public readonly record struct ImageResourceInfo(ImageResourceClass Class, bool Written);

// The immutable part of a compiled program the executor and the hosts read.
public class ShaderProgramInfo
{
    public const int NoScalarRegister = -1;

    public ShaderStageKind Stage { get; init; }
    public ulong Hash { get; init; }
    public uint UserDataBase { get; init; }
    public uint UserDataCount { get; init; }
    public uint ParameterExportMask { get; init; }
    public uint? InputLocationMask { get; init; }
    // Four component bits per color target (MRT0 in the low nibble) the pixel program exports.
    public uint PixelColorExportMasks { get; init; } = uint.MaxValue;
    public int VertexOffsetScalarRegister { get; init; } = NoScalarRegister;
    public int InstanceOffsetScalarRegister { get; init; } = NoScalarRegister;
    public bool UsesDeviceAddresses { get; init; }

    // A store or an atomic through the device-address page table: the draw writes memory
    // no host-side range bounds, unlike a program that only reads through it.
    public bool WritesDeviceAddresses { get; init; }

    // The dword every invocation stores when the whole program is 'index, v_mov constant, one
    // buffer store, end' (AGC's constant fill kernels); null for any other program.
    public uint? ConstantStoreValue { get; init; }
    public bool HasBitwiseExclusiveOr { get; init; }
    public Pipelines.ConstantFill? ConstantFill { get; init; }
    public Pipelines.BoundedFill? BoundedFill { get; init; }
    public Pipelines.BoundedCopy? BoundedCopy { get; init; }
    public BufferResourceInfo[] Buffers { get; init; } = [];
    public ImageResourceInfo[] Images { get; init; } = [];
    public int SamplerCount { get; init; }
    public uint WaveSize { get; init; } = 64;
    public uint ScratchDwords { get; init; }

    // The components the vertex program fetches per attribute location; zero when it fetches none.
    public byte[] VertexFetchComponents { get; init; } = new byte[VertexInputInfo.MaxBuffers];

    // The specialized resources and the binding layout the module was compiled against.
    public SpecializedResourceInfo? Resources { get; init; }
    public BindingLayout? Bindings { get; init; }
}

// One shader stage bound to a draw or dispatch: its program, the resources it reads and its code base.
public readonly record struct ShaderStageResources(ShaderProgramInfo? Program, ResourceSnapshot Resources, ulong ShaderBase = 0)
{
    public bool IsValid => Program is not null;

    public DispatchThreadLimits? ThreadLimits { get; init; }

    public MeshDrawParameters? MeshDraw { get; init; }

    public void WriteDispatchThreadLimits(Span<uint> shaderData)
    {
        if (Program?.Bindings is not { UsesDispatchThreadLimits: true } layout) return;
        if (ThreadLimits is not { } limits || shaderData.Length != layout.ShaderDataDwordCount)
        {
            throw GpuFatal.Fatal("The compute dispatch has missing thread limits or invalid shader data.");
        }

        var offset = (int)layout.DispatchThreadLimitsDword;
        shaderData[offset] = limits.X;
        shaderData[offset + 1] = limits.Y;
        shaderData[offset + 2] = limits.Z;
    }

    public void WriteMeshDrawParameters(Span<uint> shaderData)
    {
        if (Program?.Bindings is not { UsesMeshDrawParameters: true } layout) return;
        if (MeshDraw is not { } draw || shaderData.Length != layout.ShaderDataDwordCount)
        {
            throw GpuFatal.Fatal("The mesh draw has missing parameters or invalid shader data.");
        }

        var offset = (int)layout.MeshDrawParametersDword;
        shaderData[offset] = draw.VertexCount;
        shaderData[offset + 1] = draw.FirstVertex;
        shaderData[offset + 2] = draw.FirstInstance;
        shaderData[offset + 3] = draw.IndexElementSize;
        shaderData[offset + 4] = (uint)draw.IndexAddress;
        shaderData[offset + 5] = (uint)(draw.IndexAddress >> 32);
    }
}

public readonly record struct DispatchThreadLimits(uint X, uint Y, uint Z);

public readonly record struct MeshDrawParameters(
    uint VertexCount,
    uint FirstVertex,
    uint FirstInstance,
    uint IndexElementSize,
    ulong IndexAddress);

// The vertex buffer words of one fetch slot as the vertex program declares them.
public readonly record struct VertexInputBuffer(ulong Address, uint Stride, uint RecordCount, bool PerInstance = false)
{
    public ulong Size => Stride != 0 ? (ulong)Stride * RecordCount : RecordCount;
}

// One attribute of the vertex tables: its buffer words, the registers it fills and its buffer slot.
public readonly record struct VertexAttributeResource(
    BufferDescriptorWords Descriptor,
    int RegisterStart,
    int RegisterCount,
    int AttributeId,
    uint FetchIndex,
    int BufferIndex,
    uint OffsetBytes);

// The viewport transform a vertex program applies when clipping is off, and the guest's
// clip-space conventions a host may have to apply in the program.
public readonly record struct ClipSpaceTransform(
    bool Enabled,
    float ScaleX,
    float ScaleY,
    float OffsetX,
    float OffsetY,
    float HalfExtentX,
    float HalfExtentY)
{
    // PA_CL_CLIP_CNTL.DX_CLIP_SPACE_DEF clear: the program's z runs over [-w, w].
    public bool DepthNegativeOneToOne { get; init; }

    // The host viewport maps +y down the target: the clip-disabled viewport always, else a
    // positive guest y scale. A host whose clip space is y-up negates the position's y then.
    public bool PositionYDown { get; init; }

    // A negative guest x scale: the host viewport keeps a positive width and the program
    // negates the position's x.
    public bool PositionXMirror { get; init; }

    // The guest depth planes the host clamps instead of clipping, which the program clips.
    public SharpEmu.ShaderCompiler.ShaderDepthPlanes DepthPlanes { get; init; }

    // PA_CL_VTE_CNTL.VTX_XY_FMT / VTX_Z_FMT: the exported x and y, or z, are already divided
    // by w. VTX_W0_FMT clear: the exported w is 1 / w.
    public bool XyPreDivided { get; init; }
    public bool ZPreDivided { get; init; }
    public bool WReciprocal { get; init; }

    // The enabled user clip planes (PA_CL_UCP_n, four floats each), clipped in the program;
    // UCP_CULL_ONLY_ENA culls with them instead.
    public float[]? UserClipPlanes { get; init; }
    public bool UserClipPlanesCullOnly { get; init; }

    // The program maps depth onto the host's viewport range: z' = z * DepthScale + w * DepthOffset.
    public bool DepthRemap { get; init; }
    public float DepthScale { get; init; }
    public float DepthOffset { get; init; }
}

public sealed class VertexInputInfo
{
    public const int MaxBuffers = 32;

    public VertexInputBuffer[] Buffers { get; init; } = [];
    public VertexAttributeResource[] Attributes { get; init; } = [];
    public bool FetchEmbedded { get; init; }
    public int FetchAttributeRegister { get; init; }
    public int FetchBufferRegister { get; init; }
    public bool IeeeMode { get; set; }
    public uint FloatMode { get; set; } = SharpEmu.ShaderCompiler.ShaderCompileRequest.DefaultFloatMode;
    public bool Dx10Clamp { get; set; } = true;
    public uint ScratchDwords { get; init; }
    public uint PositionExportControl { get; init; }
    public ClipSpaceTransform ClipSpace { get; init; }
    public ShaderStageResources Stage { get; set; }
}

public sealed record MeshDrawConfiguration
{
    public GuestGeometryConfiguration Geometry { get; init; } = new();
    public MeshExecutionLimits Execution { get; init; } = new();
}

public readonly record struct MeshExecutionLimits
{
    public uint DeviceSubgroupLaneCount { get; init; }
    public uint MaxGroupCountX { get; init; }
    public uint MaxGroupCountY { get; init; }
    public uint MaxGroupTotalCount { get; init; }
}

public readonly record struct GuestGeometryConfiguration
{
    public uint ThreadsPerGroup { get; init; }
    public bool InputTriangleStrip { get; init; }
    public bool InputPointList { get; init; }
    public bool InputLineList { get; init; }
    public uint InputPrimitiveCountPerWorkgroup { get; init; }
    public uint InputVertexCountPerWorkgroup { get; init; }
    public uint OutputVertexCapacity { get; init; }
    public uint OutputPrimitiveCapacity { get; init; }
    public uint ProvokingVertex { get; init; }
    public uint WaveSize { get; init; }
    public bool IeeeMode { get; init; }
    public uint ScratchDwords { get; init; }
    public uint LocalDataShareDwords { get; init; }
    public uint PositionExportControl { get; init; }
    public ClipSpaceTransform ClipSpace { get; init; }
}

public sealed class PixelInputInfo
{
    public const int InterpolatorCount = 32;
    public const int TargetCount = 8;
    public const uint NoPerspectiveCenterRegister = uint.MaxValue;

    public uint InputCount { get; init; }
    public uint SystemInputBase { get; init; }
    public uint CustomInterpolationMask { get; init; }
    public uint PerspectiveCenterRegister { get; init; } = NoPerspectiveCenterRegister;
    // Each instance still owns its default arrays, made on first read: the resolver sets all
    // three for every draw, and initializers allocated them only to be replaced.
    public uint[] InterpolatorSettings { get => _interpolatorSettings ??= new uint[InterpolatorCount]; init => _interpolatorSettings = value; }
    public byte[] TargetOutputModes { get => _targetOutputModes ??= new byte[TargetCount]; init => _targetOutputModes = value; }
    public ColorComponentMap[] TargetExportMappings { get => _targetExportMappings ??= new ColorComponentMap[TargetCount]; init => _targetExportMappings = value; }
    private uint[]? _interpolatorSettings;
    private byte[]? _targetOutputModes;
    private ColorComponentMap[]? _targetExportMappings;
    // SPI_PS_IN_CONTROL.PS_W32_EN (bit 15).
    public bool GuestWave32 { get; set; }
    // SPI_SHADER_COL_FORMAT of each EXP target.
    public byte[] TargetExportFormats { get => _targetExportFormats ??= new byte[TargetCount]; init => _targetExportFormats = value; }
    private byte[]? _targetExportFormats;
    public bool IeeeMode { get; set; }
    public uint FloatMode { get; set; } = SharpEmu.ShaderCompiler.ShaderCompileRequest.DefaultFloatMode;
    public bool Dx10Clamp { get; set; } = true;
    public uint ScratchDwords { get; init; }
    public bool PositionX { get; init; }
    public bool PositionY { get; init; }
    public bool PositionZ { get; init; }
    public bool PositionW { get; init; }
    public bool FrontFace { get; init; }
    // SPI_BARYC_CNTL.FRONT_FACE_ALL_BITS: FRONT_FACE is 0xFFFFFFFF/0 rather than +1.0/-1.0.
    public bool FrontFaceAllBits { get; init; }
    public bool NoPerspective { get; init; }
    public bool KillEnable { get; init; }
    public bool DepthExportEnable { get; init; }
    public bool StencilExportEnable { get; init; }
    public bool SampleMaskExportEnable { get; init; }
    // SPI_SHADER_Z_FORMAT; null when the registers name no format.
    public SharpEmu.ShaderCompiler.Gen5ColorExportFormat? DepthExportFormat { get; init; }
    public bool SampleShading { get; init; }
    public bool EarlyDepth { get; init; }
    public bool ExecuteOnNoop { get; init; }
    public ShaderStageResources Stage { get; set; }

    public bool PositionXY => PositionX && PositionY;
}

public sealed class ComputeInputInfo
{
    public uint ThreadsX { get; init; }
    public uint ThreadsY { get; init; }
    public uint ThreadsZ { get; init; }
    public bool DispatchThreadDimensions { get; init; }
    public uint DispatchThreadsX { get; set; }
    public uint DispatchThreadsY { get; set; }
    public uint DispatchThreadsZ { get; set; }
    public bool GroupIdX { get; init; }
    public bool GroupIdY { get; init; }
    public bool GroupIdZ { get; init; }
    public int ThreadIdCount { get; init; }
    public bool ThreadGroupSizeEnabled { get; init; }
    public uint WaveSize { get; init; } = 64;
    public uint LocalDataShareDwords { get; init; }
    public bool IeeeMode { get; set; }
    public uint FloatMode { get; set; } = SharpEmu.ShaderCompiler.ShaderCompileRequest.DefaultFloatMode;
    public bool Dx10Clamp { get; set; } = true;
    public uint ScratchDwords { get; init; }
    public bool NeedsLocalDataShareBarriers { get; init; }
    public int WorkgroupRegister { get; init; }
    public ShaderStageResources Stage { get; set; }
}

// The four buffer resource words as the guest writes them.
public readonly record struct BufferDescriptorWords(uint Word0, uint Word1, uint Word2, uint Word3)
{
    public const uint Format32UInt = 20;
    public const uint Format32x4UInt = 75;

    public static BufferDescriptorWords From(ReadOnlySpan<uint> words) => new(words[0], words[1], words[2], words[3]);

    public ulong Address => (Word0 | ((ulong)Word1 << 32)) & 0xFFFF_FFFF_FFFFul;
    public uint Stride => (Word1 >> 16) & 0x3FFF;
    public bool SwizzleEnabled => (Word1 >> 31) != 0;
    public uint RecordCount => Word2;
    public byte DestinationSelectX => (byte)(Word3 & 0x7);
    public byte DestinationSelectY => (byte)((Word3 >> 3) & 0x7);
    public byte DestinationSelectZ => (byte)((Word3 >> 6) & 0x7);
    public byte DestinationSelectW => (byte)((Word3 >> 9) & 0x7);
    public uint DestinationSelectXY => Word3 & 0x3F;
    public uint DestinationSelectXYZ => Word3 & 0x1FF;
    public uint DestinationSelectXYZW => Word3 & 0xFFF;
    public uint IndexStride => (Word3 >> 21) & 0x3;
    public bool AddThreadId => ((Word3 >> 23) & 0x1) != 0;
    public uint Format => (Word3 >> 12) & 0x7F;
    public uint OutOfBounds => (Word3 >> 28) & 0x3;
    public uint Type => (Word3 >> 30) & 0x3;

    public uint PackedStride =>
        Stride | ((SwizzleEnabled ? 1u : 0u) << 14) | (IndexStride << 16) | ((AddThreadId ? 1u : 0u) << 20);

    // The words with the 48-bit base replaced; the high address bits keep their other fields.
    public BufferDescriptorWords WithAddress(ulong address) =>
        this with { Word0 = (uint)address, Word1 = (Word1 & 0xFFFF_0000u) | (uint)((address >> 32) & 0xFFFFu) };

    public void CopyTo(Span<uint> words)
    {
        words[0] = Word0;
        words[1] = Word1;
        words[2] = Word2;
        words[3] = Word3;
    }

    // The byte footprint the records cover; null when it overflows.
    //
    // A swizzled V# (SWIZZLE_ENABLE with a nonzero stride, the rule the shader translators
    // apply too) counts num_records in records of stride bytes but does not lay them out
    // one after another: index_stride records share each row of dwords, one dword of each
    // in turn (RDNA2 ISA 8.1.5, a four-byte element on gfx10). Its footprint runs to one
    // past the last byte of record records - 1 at offset stride - 1, which is
    // sharpemu_swizzled_footprint in the Metal translator's buffer_offset_prelude.msl.
    public ulong? Footprint()
    {
        ulong records = RecordCount;
        ulong stride = Stride;
        if (SwizzleEnabled && stride != 0)
        {
            if (records == 0)
            {
                return 0;
            }

            // records < 2^32, stride < 2^14 and index_stride <= 64: no overflow.
            var indexStride = 8ul << (int)IndexStride;
            var last = records - 1;
            var end = stride - 1;
            return (((last / indexStride) * stride) + (end & ~3ul)) * indexStride + ((last % indexStride) * 4) + (end & 3) + 1;
        }

        if (stride != 0 && records > ulong.MaxValue / stride)
        {
            return null;
        }

        return stride == 0 ? records : records * stride;
    }
}
