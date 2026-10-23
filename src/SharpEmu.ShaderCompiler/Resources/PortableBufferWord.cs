// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Resources;

/// <summary>
/// Portable buffers: a program reads each buffer's V# stride, and the format and swizzle
/// word of a formatted buffer it only reads, from its shader data at run time instead of
/// baking them in. This is the generic/prewarm form. A materialized read-only formatted
/// descriptor can additionally select a narrow variant that fixes its stride and format.
/// </summary>
/// <remarks>
/// Two dwords per buffer at <see cref="BindingLayout.FirstBufferWordDword"/>: the stride
/// (V# word 1 bits 16-29) and V# word 3. The specialization keeps the flags in
/// PackedStride above bit 13 (SWIZZLE_ENABLE, INDEX_STRIDE, ADD_TID_ENABLE), which change
/// the addressing code, and the format of a written formatted buffer, which picks the
/// store's conversion. A process-wide switch like ResourceMaterializer.RecordMipStatistics:
/// the backend sets it before anything is translated, and every translation of the
/// process, a draw's or a precompile's, sees the same value.
/// </remarks>
public static class PortableBufferWord
{
    public const int DwordCount = 2;
    public const uint StrideMask = 0x3FFF;
    // A narrow read-format variant. Stored in the specialization/key, never in a V#.
    public const uint SpecializedFormat = 1u << 31;

    public static bool Enabled { get; set; }
    public static bool SpecializeReadFormats { get; set; }

    public static bool IsSpecialized(uint packedStride) => (packedStride & SpecializedFormat) != 0;

    public static bool ReadsFormatAtRuntime(BufferResource buffer) => buffer.Formatted && !buffer.Written;

    /// <summary>The run-time words of one V#: the stride and word 3.</summary>
    public static (uint Stride, uint DescriptorWord3) Pack(ReadOnlySpan<uint> descriptor) =>
        descriptor.Length < 4 ? default : ((descriptor[1] >> 16) & StrideMask, descriptor[3]);

    /// <summary>A buffer's specialization with what the program reads at run time left out.</summary>
    public static BufferSpecialization Normalize(BufferSpecialization specialization, BufferResource? buffer) =>
        IsSpecialized(specialization.PackedStride) ? specialization :
        buffer is not null && ReadsFormatAtRuntime(buffer)
            ? new BufferSpecialization(specialization.PackedStride & ~StrideMask, DescriptorConstants.InvalidFormat,
                DescriptorConstants.IdentityDestinationSelect)
            : specialization with { PackedStride = specialization.PackedStride & ~StrideMask };

    // Default/prewarm specializations remain portable. Only an actual materialized,
    // read-only formatted descriptor can select this variant; addresses and bounds
    // remain runtime data and never contribute to its identity.
    public static BufferSpecialization ForDescriptor(BufferSpecialization specialization, BufferResource buffer, bool? specializeReadFormats = null) =>
        (specializeReadFormats ?? SpecializeReadFormats) && ReadsFormatAtRuntime(buffer) &&
        specialization.DescriptorFormat != DescriptorConstants.InvalidFormat &&
        Gfx10UnifiedFormat.TryDecode(specialization.DescriptorFormat, out var dataFormat, out _) && dataFormat != 0
            ? specialization with { PackedStride = specialization.PackedStride | SpecializedFormat }
            : Normalize(specialization, buffer);
}
