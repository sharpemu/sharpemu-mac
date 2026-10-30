// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

public sealed partial class MslRuntimeFormatLoadTests(ITestOutputHelper output)
{
    private const int BufferBytes = 128;

    // Independent read and write descriptors, as for a vertex stream and an output buffer.
    // s0 is the byte offset, supplied at execution rather than translation time.
    private static Gen5ComputeFixture Fixture(uint components, bool glc = false) => new(
        "runtime-format-load",
        [
            0x7E020200, // v_mov_b32 v1, s0
            0xE0001000 | ((components - 1) << 18) | (glc ? 1u << 14 : 0), 0x80020401,
            0xE0700000, 0x80030400,
            0xE0700004, 0x80030500,
            0xE0700008, 0x80030600,
            0xE070000C, 0x80030700,
            0xBF810000,
        ], 12, BufferBytes);

    private static (Gen5MslShader Shader, BindingLayout Layout) Compile(
        Gen5ComputeFixture fixture, bool fast, uint threads = 1) =>
        Compile(fixture, new Gen5MslCodegenOptions(RuntimeDwordFormatLoads: fast, ForceRegisterInitialization: true), threads);

    private static (Gen5MslShader Shader, BindingLayout Layout) Compile(
        Gen5ComputeFixture fixture, Gen5MslCodegenOptions options, uint threads = 1)
    {
        var program = Gen5ComputeFixtures.DecodeOrThrow(fixture);
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, 0x1234, 0, 16);
        var resources = ResourceMaterializer.ApplyTo(plan, ResourceSpecialization.Default(plan.Info));
        var layout = BindingLayout.Allocate(resources.Info,
            BindingLayout.CollectUserDataRegisters(program, 0, 16),
            false, false, false, portableBuffers: true);
        var request = new ShaderCompileRequest(plan, resources, layout)
        {
            LocalSizeX = threads,
            ComputeSystemRegisters = new Gen5ComputeSystemRegisters(16, null, null, null),
        };
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error, options), error);
        return (shader, layout);
    }

    private static byte[] Push(BindingLayout layout, uint word3, uint offset)
    {
        var push = new byte[PushData.ByteSize];
        for (var i = 0; i < layout.UserDataRegisters.Count; i++)
        {
            if (layout.UserDataRegisters[i] == 0)
                BinaryPrimitives.WriteUInt32LittleEndian(push.AsSpan(i * 4), offset);
        }
        for (uint i = 0; i < layout.MemoryOffsetCount; i++)
            BinaryPrimitives.WriteUInt32LittleEndian(push.AsSpan((int)(layout.FirstBufferWordDword() + i * 2 + 1) * 4), word3);
        return push;
    }

    private bool RequireDevice()
    {
        if (MetalNative.IsAvailable) return true;
        Assert.False(Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
            "A Metal device is required for runtime-format validation.");
        output.WriteLine("No Metal device; runtime-format execution skipped.");
        return false;
    }

    [Theory]
    [InlineData(1u, false)]
    [InlineData(2u, false)]
    [InlineData(3u, false)]
    [InlineData(4u, false)]
    [InlineData(4u, true)]
    public void ChangingDescriptorsAndBoundsMatchTheGeneralDecoder(uint components, bool glc)
    {
        var (fast, layout) = Compile(Fixture(components, glc), true);
        var (slow, _) = Compile(Fixture(components, glc), false);
        Assert.Contains($"sharpemu_load_words{(glc ? "_coherent" : "")}<{components}>", fast.Source);
        Assert.NotEqual(fast.Source, slow.Source);
        if (!RequireDevice()) return;
        Assert.True(MetalNative.TryCompileLibrary(fast.Source, out var fastLibrary, out var fastError), fastError);
        Assert.True(MetalNative.TryCompileLibrary(slow.Source, out var slowLibrary, out var slowError), slowError);
        // Same two compiled libraries, changing the descriptor each time. All 128
        // unified-format codes and nonidentity/constant selectors exercise fallback.
        var cases = Enumerable.Range(0, 128).Select(format => ((uint)format, 0xFACu, 32u))
            .Concat(from format in new uint[] { 20, 21, 22, 62, 63, 64, 72, 73, 74, 75, 76, 77 }
                    from swizzle in new uint[] { 0xFAC, 0x977, 0x204, 0xFFF }
                    from offset in new uint[] { 33, 112, 113, 124 }
                    select (format, swizzle, offset));
        foreach (var (format, swizzle, offset) in cases)
        {
            var data = Enumerable.Range(0, BufferBytes).Select(i => (byte)(i * 37 + 11)).ToArray();
            var push = Push(layout, (format << 12) | swizzle, offset);
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(slowLibrary, slow.EntryPoint, data, push,
                slow.ArgumentLayout!, 1, out var expected, out var slowRunError), slowRunError);
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(fastLibrary, fast.EntryPoint, data, push,
                fast.ArgumentLayout!, 1, out var actual, out var fastRunError), fastRunError);
            Assert.True(expected.AsSpan().SequenceEqual(actual),
                $"components={components} glc={glc} format={format} swizzle=0x{swizzle:X} offset={offset}");
            // Independent reference for all optimized formats, including missing components.
            var present = format switch { >= 20 and <= 22 => 1u, >= 62 and <= 64 => 2u,
                >= 72 and <= 74 => 3u, >= 75 and <= 77 => 4u, _ => 0u };
            if (present != 0 && swizzle == 0xFAC)
            {
                var valid = offset <= BufferBytes - Math.Min(components, present) * 4;
                var one = format is 22 or 64 or 74 or 77 ? 0x3F800000u : 1u;
                for (var c = 0; c < components; c++)
                    Assert.Equal(!valid ? 0u : c >= present ? (c == 3 ? one : 0u) :
                            BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset + c * 4)),
                        BinaryPrimitives.ReadUInt32LittleEndian(actual.AsSpan(c * 4)));
            }
        }
    }

    [Theory]
    [InlineData(1u)]
    [InlineData(2u)]
    [InlineData(3u)]
    [InlineData(4u)]
    public void TheCalledDecoderMatchesTheInlineOne(uint components)
    {
        // Without the whole-dword fast path every load takes the general decoder: inline below
        // the threshold, sharpemu_format_load_decoded at or above it.
        Gen5MslCodegenOptions Options(int threshold) =>
            new(RuntimeDwordFormatLoads: false, ForceRegisterInitialization: true) { DecodedFormatLoadThreshold = threshold };
        var (inline, layout) = Compile(Fixture(components), Options(0));
        var (called, _) = Compile(Fixture(components), Options(1));
        Assert.DoesNotContain("sharpemu_format_load_decoded", inline.Source);
        Assert.Contains($"sharpemu_format_load_decoded(b", called.Source);
        if (!RequireDevice()) return;
        Assert.True(MetalNative.TryCompileLibrary(inline.Source, out var inlineLibrary, out var inlineError), inlineError);
        Assert.True(MetalNative.TryCompileLibrary(called.Source, out var calledLibrary, out var calledError), calledError);
        // Every unified format with the identity, reversed, constant and repeated selectors,
        // at aligned, unaligned and end-straddling offsets.
        var cases = from format in Enumerable.Range(0, 128).Select(format => (uint)format)
                    from swizzle in new uint[] { 0xFAC, 0x977, 0x204, 0xFFF, 0x0D1 }
                    from offset in new uint[] { 0, 33, 113, 122, 124, 127 }
                    select (format, swizzle, offset);
        var data = Enumerable.Range(0, BufferBytes).Select(i => (byte)(i * 37 + 11)).ToArray();
        foreach (var (format, swizzle, offset) in cases)
        {
            var push = Push(layout, (format << 12) | swizzle, offset);
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(inlineLibrary, inline.EntryPoint, data, push,
                inline.ArgumentLayout!, 1, out var expected, out var inlineRunError), inlineRunError);
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(calledLibrary, called.EntryPoint, data, push,
                called.ArgumentLayout!, 1, out var actual, out var calledRunError), calledRunError);
            Assert.True(expected.AsSpan().SequenceEqual(actual),
                $"components={components} format={format} swizzle=0x{swizzle:X} offset={offset}");
        }
    }
}
