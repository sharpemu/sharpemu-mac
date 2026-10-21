// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;
using System.Linq;

namespace SharpEmu.ShaderCompiler.Resources;

public enum ShaderStage : byte
{
    Unknown,
    Vertex,
    Pixel,
    Compute,
    Mesh,
}

public enum ImageNumericClass : byte
{
    Unsupported,
    Float,
    Uint,
    Sint,
    // A 64-bit unsigned view of a 64 bits per texel image; only image atomics use it.
    Uint64,
}

public enum ImageMipMode : byte
{
    None,
    DynamicStorage,
}

// The identity dword selection with every component in place.
public static class DescriptorConstants
{
    public const uint IdentityDestinationSelect = 4 | (5 << 3) | (6 << 6) | (7 << 9);
    public const uint IdentityImageSwizzle = 0xFAC;
    public const uint InvalidFormat = 0;
    public const uint NoIndex = uint.MaxValue;
}

// One buffer resource of a program and how the program uses it.
public sealed class BufferResource
{
    public uint Source { get; set; }
    public uint FirstUsePc { get; set; }
    public uint MaxByteExtent { get; set; }
    public uint PackedStride { get; set; }
    public uint DescriptorFormat { get; set; } = DescriptorConstants.InvalidFormat;
    public uint DescriptorSwizzle { get; set; } = DescriptorConstants.IdentityDestinationSelect;
    public uint ImageAlias { get; set; } = DescriptorConstants.NoIndex;
    public bool Read { get; set; }
    public bool Written { get; set; }
    public bool Atomic { get; set; }
    public bool Formatted { get; set; }
    public bool Scalar { get; set; }

    public BufferResource Clone() => (BufferResource)MemberwiseClone();
}

// One image resource: a descriptor source with its view class, dimension and use.
public sealed class ImageResource
{
    public uint Source { get; set; }
    public uint FirstUsePc { get; set; }
    public ImageResourceClass ResourceClass { get; set; }
    public ImageNumericClass NumericClass { get; set; } = ImageNumericClass.Unsupported;
    public ImageDimension Dimension { get; set; } = ImageDimension.Unknown;
    public ImageMipMode MipMode { get; set; }
    public uint MipCount { get; set; } = 1;
    public uint ConversionFormat { get; set; } = DescriptorConstants.InvalidFormat;
    public uint ShaderSwizzle { get; set; } = DescriptorConstants.IdentityImageSwizzle;
    public bool Read { get; set; }
    public bool Written { get; set; }
    public bool Atomic { get; set; }
    // An image atomic on 64-bit texels (DMASK 0x3, or 0xF for a compare-swap pair), such
    // as a visibility buffer packing depth and an ID into one 32_32_UINT texel.
    public bool Atomic64 { get; set; }
    public bool DepthCompare { get; set; }
    // Guest compare function (0..7) evaluated in the shader for a depth-compare
    // image whose format has no Vulkan depth equivalent; -1 when not emulated.
    public int EmulatedCompareFunction { get; set; } = -1;
    public bool Cube { get; set; }
    // Every access is a DIM_CUBE instruction: a table candidate that is not a cube cannot be read.
    public bool CubeAccess { get; set; }
    public bool R128 { get; set; }
    // Some access is a _PCK load or store; the materializer then records the guest
    // format in PackedFormat so the emitter can pack and unpack the texel's raw bits.
    public bool Packed { get; set; }
    public uint PackedFormat { get; set; } = DescriptorConstants.InvalidFormat;
    public uint IndirectRoot { get; set; } = DescriptorConstants.NoIndex;
    public uint IndirectMappingOffset { get; set; }
    public uint IndirectSearchIterations { get; set; }
    public List<uint> IndirectResources { get; set; } = [];
    // The bindless table this image is a class variant of (its slot in the stage's bindless
    // image tables), or NoIndex. The root and its variants list the variants in IndirectResources.
    public uint BindlessSlot { get; set; } = DescriptorConstants.NoIndex;

    public bool MipStatistics { get; set; }

    public ImageResource Clone()
    {
        var clone = (ImageResource)MemberwiseClone();
        clone.IndirectResources = [.. IndirectResources];
        return clone;
    }
}

public sealed class SamplerResource
{
    public uint Source { get; set; }
    public uint FirstUsePc { get; set; }
    public bool ForcePointFiltering { get; set; }
    public bool DepthCompare { get; set; }

    // The paired S#'s LOD_BIAS (word 2 bits 13:0), signed 6.8 fixed point, which the
    // emitter adds to every LOD the sampler computes or is given; zero for most samplers.
    public int LodBias { get; set; }

    // The axes (bit 0 X, 1 Y, 2 Z) whose S# clamp is MIRROR_ONCE_BORDER, which no host
    // address mode is: the host samples mirror-once-to-edge and the emitter substitutes the
    // S# border colour (BorderColorType) past the mirrored copy.
    public uint MirrorOnceBorderAxes { get; set; }

    public uint BorderColorType { get; set; }

    public SamplerResource Clone() => (SamplerResource)MemberwiseClone();
}

public sealed class SampledImagePair
{
    public uint Image { get; set; }
    public uint Sampler { get; set; }
    public uint FirstUsePc { get; set; }

    public SampledImagePair Clone() => (SampledImagePair)MemberwiseClone();
}

public enum StageInputKind : byte
{
    VertexIndex,
    InstanceIndex,
    FragCoord,
    FrontFacing,
    BaryCoordSmooth,
    BaryCoordNoPerspective,
    WorkgroupId,
    LocalInvocationId,
    LocalInvocationIndex,
    GlobalInvocationId,
    Parameter,
}

public enum StageOutputKind : byte
{
    Position,
    Parameter,
    Mrt,
    Depth,
    SampleMask,
    PointSize,
    ClipDistance,
    CullDistance,
    Layer,
}

public sealed record StageInput(StageInputKind Kind, uint Location, uint ComponentCount, string DebugName, bool PerVertex);

public sealed record StageOutput(StageOutputKind Kind, uint Index, uint Location, string DebugName);

// One bounded runtime V# table lowered to a contiguous run of native buffer candidates,
// selected through a key mapping in the flattened table.
public sealed class BufferCandidateTableInfo
{
    public uint FirstCandidate { get; set; }
    public uint CandidateCount { get; set; }
    public uint MappingOffset { get; set; }
    public uint SearchIterations { get; set; }

    public BufferCandidateTableInfo Clone() => (BufferCandidateTableInfo)MemberwiseClone();
}

// The dense resource tables of a program plus the facts the pipeline layout needs.
public sealed class ShaderResourceInfo
{
    public const int MaxBuffers = 32;
    public const int MaxIndirectImageCandidates = 64;
    public const int NoScalarRegister = -1;

    public List<BufferResource> Buffers { get; set; } = [];
    public List<ImageResource> Images { get; set; } = [];
    public List<SamplerResource> Samplers { get; set; } = [];
    public List<SampledImagePair> SampledPairs { get; set; } = [];
    public List<BufferCandidateTableInfo> BufferCandidateTables { get; set; } = [];
    public uint[] ImageBindings { get; set; } = [];
    public uint[] SamplerBindings { get; set; } = [];
    public uint GetCanonicalImageBinding(uint resource) => ImageBindings.Length == 0 ? resource : ImageBindings[resource];
    public uint SamplerBinding(uint resource) => SamplerBindings.Length == 0 ? resource : SamplerBindings[resource];
    public List<StageInput> Inputs { get; set; } = [];
    public List<StageOutput> Outputs { get; set; } = [];
    public byte[] VertexFetchComponents { get; set; } = new byte[32];
    public int VertexOffsetScalarRegister { get; set; } = NoScalarRegister;
    public int InstanceOffsetScalarRegister { get; set; } = NoScalarRegister;
    public bool HasBitwiseExclusiveOr { get; set; }
    public bool UsesDeviceAddresses { get; set; }

    /// <summary>
    /// The program stores through the device-address page table, so it can write any
    /// mapped guest page and no host-side range describes where. A backend that keeps
    /// guest memory as the GPU's memory has to treat the submission as writing
    /// everywhere; one that copies into staging resources does not.
    /// </summary>
    public bool WritesDeviceAddresses { get; set; }

    public ShaderResourceInfo Clone() => new()
    {
        Buffers = Buffers.Select(buffer => buffer.Clone()).ToList(),
        Images = Images.Select(image => image.Clone()).ToList(),
        Samplers = Samplers.Select(sampler => sampler.Clone()).ToList(),
        SampledPairs = SampledPairs.Select(pair => pair.Clone()).ToList(),
        BufferCandidateTables = BufferCandidateTables.Select(table => table.Clone()).ToList(),
        ImageBindings = [.. ImageBindings],
        SamplerBindings = [.. SamplerBindings],
        Inputs = [.. Inputs],
        Outputs = [.. Outputs],
        VertexFetchComponents = (byte[])VertexFetchComponents.Clone(),
        VertexOffsetScalarRegister = VertexOffsetScalarRegister,
        InstanceOffsetScalarRegister = InstanceOffsetScalarRegister,
        HasBitwiseExclusiveOr = HasBitwiseExclusiveOr,
        UsesDeviceAddresses = UsesDeviceAddresses,
        WritesDeviceAddresses = WritesDeviceAddresses,
    };
}

// A material-table key that selects one of several heap descriptors at run time.
public sealed record IndirectImageSelector(
    uint MaterialSource,
    uint HeapSource,
    uint SelectorStride,
    uint SelectorOffset,
    uint KeyArgument)
{
    public IndirectSelectorValues? SelectorValues { get; init; }
    public IReadOnlyList<DirectImageCandidate>? DirectCandidates { get; init; }
    public bool Dense { get; init; }
    public uint TableOffset { get; init; }
    public uint DynamicOffsetBase { get; init; }
    public uint KeyBound { get; init; }
    public WaveIndexedImageSelector? WaveIndexed { get; init; }
    // A dense table a V# describes, read through a per-entry handle table on the device
    // instead of a bounded set of bound candidates.
    public bool Bindless { get; init; }

    // The key is selected by a value the whole wave shares (a loop counter), not by a lane's value. Such an image
    // used to read as a null image, so a table the host cannot read stays null instead of failing the program.
    public bool SharedSelector { get; init; }

    // The key read's immediate offset. The hardware adds it after the 32-bit selector offset, without wrapping.
    public uint MaterialImmediate { get; init; }
}

public sealed record DirectImageCandidate(uint Offset, uint Source);

// A wave-uniform descriptor selector. The guest derives each descriptor key from
// a set bit in one scalar mask, through a compact global index table. Keeping this
// shape explicit lets the host materialize only those keys, rather than treating a
// lane value as an unknowable descriptor address.
public sealed record WaveIndexedImageSelector(uint MaskOffset, uint IndexTableOffset, uint IndexStride);

// The graph values one descriptor is assembled from, up to eight dwords.
public sealed class DescriptorSource
{
    public ScalarValue[] Dwords { get; init; } = [];
    public uint DwordCount => (uint)Dwords.Length;
    public IndirectImageSelector? IndirectImage { get; init; }
}

// One immediate-offset scalar read the host evaluates into the flattened table.
public sealed record ResourceTableRead(ScalarValue Value, uint FlatOffset);

// The dwords of one materialised descriptor.
public readonly record struct DescriptorWords(uint[] Dwords)
{
    public uint DwordCount => (uint)Dwords.Length;

    public static DescriptorWords Empty(uint dwordCount) => new(new uint[dwordCount]);

    public bool SameAs(DescriptorWords other) => Dwords.AsSpan().SequenceEqual(other.Dwords);
}
