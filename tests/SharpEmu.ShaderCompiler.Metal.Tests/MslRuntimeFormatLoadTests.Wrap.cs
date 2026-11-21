// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// The run-time word 3 forms of a formatted load whose selected components wrap past 2^32
// (upstream 7cf0ac568, BufferFormatWrapDeviceTests): the fetch helper, the called decoder, the
// inline decoder, the GLC (coherent, inline) form and the function-constant form each read
// [Y, Z, W, 0] of a 32_32_32_32 element at 2^32 - 4 + m as the buffer's bytes m, m + 4, m + 8.
public sealed partial class MslRuntimeFormatLoadTests
{
    private const uint WrapWord3 = (75u << 12) | 5u | (6u << 3) | (7u << 6);

    public static TheoryData<string, uint> WrapCases()
    {
        var cases = new TheoryData<string, uint>();
        foreach (var path in new[] { "fetch", "decoded", "inline", "glc", "constant" })
        {
            foreach (var misalignment in new uint[] { 0, 1, 3 })
            {
                cases.Add(path, misalignment);
            }
        }

        return cases;
    }

    [Theory]
    [MemberData(nameof(WrapCases))]
    public void SelectedComponentsPastTheTopOfTheByteSpaceWrap(string path, uint misalignment)
    {
        var options = path switch
        {
            "fetch" => new Gen5MslCodegenOptions(ForceRegisterInitialization: true) { FormatFetch = true },
            "decoded" => new Gen5MslCodegenOptions(RuntimeDwordFormatLoads: false, ForceRegisterInitialization: true)
            {
                DecodedFormatLoadThreshold = 1,
                FormatFetch = false,
            },
            "constant" => ConstantOptions(true),
            _ => new Gen5MslCodegenOptions(ForceRegisterInitialization: true)
            {
                DecodedFormatLoadThreshold = 0,
                FormatFetch = false,
            },
        };
        var (shader, layout) = Compile(Fixture(4, glc: path == "glc"), options);
        var marker = path switch
        {
            "fetch" => "sharpemu_format_fetch(b",
            "decoded" => "sharpemu_format_load_decoded(b",
            "constant" => "sharpemu_format_fetch_constant(b",
            "glc" => "sharpemu_load_word_coherent(",
            _ => "sharpemu_format_layout(",
        };
        Assert.Contains(marker, shader.Source);
        if (!RequireDevice()) return;

        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
        var data = Enumerable.Range(0, BufferBytes).Select(i => (byte)(i * 37 + 11)).ToArray();
        IReadOnlyList<(uint Index, uint Value)>? constants = path == "constant"
            ? [(Assert.Single(shader.BufferWordConstants).Index, WrapWord3)]
            : null;
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, data,
            Push(layout, WrapWord3, 0xFFFF_FFFCu + misalignment), shader.ArgumentLayout!, 1,
            out var result, out _, out var runError, constants: constants), runError);
        uint Bytes(uint first) => BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)first));
        Assert.Equal(
            [Bytes(misalignment), Bytes(misalignment + 4), Bytes(misalignment + 8), 0u],
            Enumerable.Range(0, 4).Select(index => BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(index * 4))).ToArray());
    }
}
