// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Resources;

// The descriptor words and user data captured for one program at draw time, plus the
// flattened resource table and the device-address ranges the planner could bound.
public sealed class ResourceSnapshot
{
    public const uint NoMipStatistics = uint.MaxValue;
    public const uint MipStatisticsEnabled = 1u << 8;
    public const uint MipStatisticsEnableWordMask = 1u << 25;
    public const uint MipStatisticsSlotWordMask = 0xFF;

    public uint[][] Buffers { get; init; } = [];
    public uint[][] Images { get; init; } = [];
    public uint[][] Samplers { get; init; } = [];
    public uint[] FlattenedResourceTable { get; init; } = [];
    public uint[] UserData { get; init; } = [];
    public DeviceAddressRange[] DeviceAddressRanges { get; init; } = [];
    public BindlessImageTable[] BindlessImages { get; init; } = [];
    public uint[]? MipStatistics { get; init; }
    public object? Origin { get; set; }

    public static uint MipStatisticsOf(uint word5, uint word6) =>
        (word6 & MipStatisticsSlotWordMask) | ((word5 & MipStatisticsEnableWordMask) != 0 ? MipStatisticsEnabled : 0u);

    /// <summary>
    /// The same snapshot carrying this draw's user data.
    /// </summary>
    /// <remarks>
    /// A cached materialization is shared by every draw whose descriptors evaluate the same
    /// way, which includes draws that differ in user-data words the plan never reads — words
    /// a shader can still read as inline constants through its binding layout. Those come
    /// from the draw, not from the entry that was cached with it.
    /// </remarks>
    public ResourceSnapshot WithUserData(uint[] userData) =>
        ReferenceEquals(UserData, userData) || UserData.AsSpan().SequenceEqual(userData)
            ? this
            : new ResourceSnapshot
            {
                Buffers = Buffers,
                Images = Images,
                Samplers = Samplers,
                FlattenedResourceTable = FlattenedResourceTable,
                UserData = userData,
                DeviceAddressRanges = DeviceAddressRanges,
                BindlessImages = BindlessImages,
                MipStatistics = MipStatistics,
                // Origin stays unset: the copy's user data differs from its entry's, so it must
                // not take the entry's fast revalidation.
            };
}

// One guest image-descriptor table a V# describes, resolved for a draw: every distinct usable
// T# it holds with the class variant that reads it, and per entry the T# it names (NullEntry for
// a null, unusable or mismatched one). The host turns it into a device table of texture handles
// the shader indexes by entry; a key past the entries reads the null entry after them.
public sealed class BindlessImageTable
{
    public const int NullEntry = -1;

    /// <summary>The table's slot in the stage's bindless image tables.</summary>
    public required uint Slot { get; init; }

    /// <summary>The image resources of the table's class variants; the first is the root.</summary>
    public required uint[] VariantImages { get; init; }

    public required uint[][] Descriptors { get; init; }

    /// <summary>The variant (an index into VariantImages) each descriptor is read through.</summary>
    public required byte[] DescriptorVariants { get; init; }

    public required int[] Entries { get; init; }

    /// <summary>The guest address of entry zero, for diagnostics.</summary>
    public ulong Address { get; init; }
}

// One device-address handle's evaluated base and bounded extent for this draw.
public readonly record struct DeviceAddressRange(uint Handle, ulong Base, ulong Size, bool Planned, bool Written);
