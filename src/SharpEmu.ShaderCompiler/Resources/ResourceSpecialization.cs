// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;
using System.Linq;

namespace SharpEmu.ShaderCompiler.Resources;

public readonly record struct BufferSpecialization(uint PackedStride, uint DescriptorFormat, uint DescriptorSwizzle);

// The module-affecting layout of one bounded runtime V# table. Candidate descriptors
// themselves are appended to Buffers; this names their contiguous run and the flattened
// key mapping, so the switch body does not have to embed per-draw addresses.
public readonly record struct BufferCandidateTableSpecialization(
    uint FirstCandidate,
    uint CandidateCount,
    uint MappingOffset,
    uint SearchIterations);

// The module-affecting state of one guest S# that a host sampler state cannot carry.
// MirrorOnceBorderAxes has bit 0, 1, 2 set for a CLAMP_X, _Y, _Z of MIRROR_ONCE_BORDER;
// BorderColorType is the S# BORDER_COLOR_TYPE when any is set. ForceDegamma is the S# FORCE_DEGAMMA (word 0, bit 20).
// Reduction is the FILTER_MODE (word 0, bits 30:29: 1 minimum, 2 maximum) of a sampler that filters linearly, and
// ReductionAcrossLevels says its mip filter is linear too, so the reduction spans the two levels.
public readonly record struct SamplerSpecialization(
    int LodBias, uint MirrorOnceBorderAxes = 0, uint BorderColorType = 0, bool ForceDegamma = false,
    uint Reduction = 0, bool ReductionAcrossLevels = false);

public readonly record struct ImageSpecialization(
    ImageNumericClass NumericClass,
    ImageDimension Dimension,
    uint MipCount,
    uint ConversionFormat,
    uint ShaderSwizzle,
    uint IndirectRoot,
    uint IndirectMappingOffset,
    uint IndirectSearchIterations,
    bool Cube,
    int EmulatedCompareFunction = -1,
    uint PackedFormat = DescriptorConstants.InvalidFormat,
    uint BindlessTable = 0)
{
    // BindlessTable is the slot plus one, so a specialization recorded before the field
    // existed (a corpus record, a persisted cache) reads as not bindless.
    [System.Text.Json.Serialization.JsonIgnore]
    public uint BindlessSlot => BindlessTable == 0 ? DescriptorConstants.NoIndex : BindlessTable - 1;
}

// The module-affecting resource state of one draw. Addresses and descriptor payloads
// stay in the snapshot, so they never create a permutation.
public sealed class ResourceSpecialization : IEquatable<ResourceSpecialization>
{
    // The plan's base buffer count; Buffers beyond it are materialized candidates.
    public int BaseBufferCount { get; init; }
    // The lists start empty on first read rather than in the constructor: every draw builds one
    // of these with all of them set, and eager initializers allocated six lists it replaced.
    public List<BufferSpecialization> Buffers { get => _buffers ??= []; init => _buffers = value; }
    public List<ImageSpecialization> Images { get => _images ??= []; init => _images = value; }
    public List<BufferCandidateTableSpecialization> BufferCandidateTables { get => _bufferCandidateTables ??= []; init => _bufferCandidateTables = value; }
    public List<uint> ImageDescriptorGroups { get => _imageDescriptorGroups ??= []; init => _imageDescriptorGroups = value; }
    public List<uint> SamplerDescriptorGroups { get => _samplerDescriptorGroups ??= []; init => _samplerDescriptorGroups = value; }

    // One entry per plan sampler, or none when every entry would be the default, so the
    // common case keeps the specialization (and the cache keys built from it) it had
    // before samplers were specialized.
    public List<SamplerSpecialization> Samplers { get => _samplers ??= []; init => _samplers = value; }

    private List<BufferSpecialization>? _buffers;
    private List<ImageSpecialization>? _images;
    private List<BufferCandidateTableSpecialization>? _bufferCandidateTables;
    private List<uint>? _imageDescriptorGroups;
    private List<uint>? _samplerDescriptorGroups;
    private List<SamplerSpecialization>? _samplers;

    public bool Equals(ResourceSpecialization? other) =>
        other is not null && BaseBufferCount == other.BaseBufferCount && Buffers.SequenceEqual(other.Buffers) &&
        Images.SequenceEqual(other.Images) && BufferCandidateTables.SequenceEqual(other.BufferCandidateTables) &&
        ImageDescriptorGroups.SequenceEqual(other.ImageDescriptorGroups) &&
        SamplerDescriptorGroups.SequenceEqual(other.SamplerDescriptorGroups) &&
        Samplers.SequenceEqual(other.Samplers);

    public override bool Equals(object? obj) => Equals(obj as ResourceSpecialization);

    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(BaseBufferCount);
        foreach (var buffer in Buffers)
        {
            hash.Add(buffer);
        }

        foreach (var image in Images)
        {
            hash.Add(image);
        }

        foreach (var table in BufferCandidateTables)
        {
            hash.Add(table);
        }

        foreach (var group in ImageDescriptorGroups) hash.Add(group);
        foreach (var group in SamplerDescriptorGroups) hash.Add(group);
        foreach (var sampler in Samplers) hash.Add(sampler);
        return hash.ToHashCode();
    }

    public ResourceSpecialization Clone() => new()
    {
        BaseBufferCount = BaseBufferCount,
        Buffers = [.. Buffers],
        Images = [.. Images],
        BufferCandidateTables = [.. BufferCandidateTables],
        ImageDescriptorGroups = [.. ImageDescriptorGroups],
        SamplerDescriptorGroups = [.. SamplerDescriptorGroups],
        Samplers = [.. Samplers],
    };

    // The specialization of a plan before any draw: raw buffers and the tracked image classes.
    public static ResourceSpecialization Default(ShaderResourceInfo info) => new()
    {
        BaseBufferCount = info.Buffers.Count,
        Buffers = info.Buffers.Select(_ => new BufferSpecialization(0, DescriptorConstants.InvalidFormat, DescriptorConstants.IdentityDestinationSelect)).ToList(),
        Images = info.Images.Select(image => new ImageSpecialization(
            image.NumericClass == ImageNumericClass.Unsupported
                ? (image.Atomic64 ? ImageNumericClass.Uint64 : image.Atomic ? ImageNumericClass.Uint : ImageNumericClass.Float)
                : image.NumericClass,
            image.Dimension == ImageDimension.Unknown ? ImageDimension.Dim2D : image.Dimension,
            image.MipCount, image.ConversionFormat, image.ShaderSwizzle,
            image.IndirectRoot, image.IndirectMappingOffset, image.IndirectSearchIterations, image.Cube)).ToList(),
    };
}

// A plan's resource tables with one draw's specialization applied, which the emitter
// compiles against. Sampler overrides name the point-filtering duplicate an access uses.
public sealed class SpecializedResourceInfo
{
    public ShaderResourceInfo Info { get; init; } = new();
    public IReadOnlyDictionary<int, uint> SamplerByMemoryIndex { get; init; } = new Dictionary<int, uint>();
}
