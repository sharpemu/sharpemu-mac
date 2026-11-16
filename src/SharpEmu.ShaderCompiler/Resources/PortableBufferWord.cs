// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Resources;

/// <summary>
/// Portable buffers: a program reads each buffer's V# stride, and the format and swizzle
/// word of a formatted buffer it only reads, from its shader data at run time instead of
/// baking them in, so one translation serves every draw whatever its vertex layout - and a
/// program the game has not drawn yet can be compiled ahead of time with the same result.
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

    public static bool Enabled { get; set; }

    /// <summary>
    /// Each run-time word a portable program reads also becomes a Metal function constant
    /// (<c>[[function_constant(n)]]</c>) the program takes in place of the shader-data dword
    /// when a function is made with a value for it, so the library stays layout independent
    /// and a function specialized for one layout folds the word into its addressing and
    /// format decode. A function made without values reads the shader data as before.
    /// Process-wide like <see cref="Enabled"/>, and off until the host makes specialized
    /// functions.
    /// </summary>
    public static bool ConstantsEnabled { get; set; }

    /// <summary>The first function-constant index the buffer words take; 0 is the LOD statistics switch.</summary>
    public const uint FirstConstantIndex = 1;

    /// <summary>The prefix of a buffer word constant's name; <see cref="ConstantName"/> completes it.</summary>
    public const string ConstantNamePrefix = "sharpemu_fc_bw";

    /// <summary>The function-constant index of a buffer's word: 0 is the stride, 1 word 3.</summary>
    public static uint ConstantIndex(int binding, int word) =>
        (uint)word > 1 || binding < 0
            ? throw new ArgumentOutOfRangeException(word > 1 ? nameof(word) : nameof(binding))
            : FirstConstantIndex + ((uint)binding * DwordCount) + (uint)word;

    /// <summary>
    /// The name of the constant at <paramref name="index"/>, read in place of shader-data dword
    /// <paramref name="dword"/>: both are spelled in it, so a source declares what it reads.
    /// </summary>
    public static string ConstantName(uint index, uint dword) => $"{ConstantNamePrefix}{index}_sd{dword}";

    public static bool ReadsFormatAtRuntime(BufferResource buffer) => buffer.Formatted && !buffer.Written;

    /// <summary>The run-time words of one V#: the stride and word 3.</summary>
    public static (uint Stride, uint DescriptorWord3) Pack(ReadOnlySpan<uint> descriptor) =>
        descriptor.Length < 4 ? default : ((descriptor[1] >> 16) & StrideMask, descriptor[3]);

    /// <summary>A buffer's specialization with what the program reads at run time left out.</summary>
    public static BufferSpecialization Normalize(BufferSpecialization specialization, BufferResource? buffer) =>
        buffer is not null && ReadsFormatAtRuntime(buffer)
            ? new BufferSpecialization(specialization.PackedStride & ~StrideMask, DescriptorConstants.InvalidFormat,
                DescriptorConstants.IdentityDestinationSelect)
            : specialization with { PackedStride = specialization.PackedStride & ~StrideMask };
}
