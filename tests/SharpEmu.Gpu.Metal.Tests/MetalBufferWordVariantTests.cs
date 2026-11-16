// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// Render pipelines specialized for a draw's buffer words (MetalRenderHost.BufferWordVariants):
/// the first draw of a layout draws through the generic pipeline while its variant builds, later
/// draws take the variant, which draws the same pixels; the variant follows only the words the
/// program declares, a pipeline keeps a bounded number of them, and a variant is listed in the
/// archive and the state table apart from the generic pipeline. The vertex program here is
/// written by hand in the translator's form (Gen5MslTranslator.BufferWordConstants): a stride
/// and a format word, each a function constant over its shader-data dword, which pick how the
/// packed vertex records are decoded. Device work is behind the exclusive device gate.
/// </summary>
public sealed unsafe class MetalBufferWordVariantTests(ITestOutputHelper output)
{
    private const uint Width = 16;
    private const uint Height = 16;
    private const ulong TargetBytes = Width * Height * 4;
    private const ulong FirstTarget = MetalPassRig.GuestBase + 0x10000;
    // The vertex records start past the 16 dwords of words at the head of the bound buffer.
    private const int RecordOffset = 64;
    private const int BufferBytes = 4096;
    private const uint Snorm16x2 = 1;
    private const uint Float32x2 = 2;
    private static readonly TimeSpan BuildTimeout = TimeSpan.FromSeconds(60);

    // Binding 0's stride (dword 0) and word 3 (dword 1); dword 2 is a word the program never reads.
    private static readonly Gen5MslShader.BufferWordConstant[] Declared =
    [
        new(PortableBufferWord.ConstantIndex(0, 0), 0),
        new(PortableBufferWord.ConstantIndex(0, 1), 1),
    ];

    private static readonly string Stride = PortableBufferWord.ConstantName(Declared[0].Index, Declared[0].Dword);
    private static readonly string Word3 = PortableBufferWord.ConstantName(Declared[1].Index, Declared[1].Dword);

    // Two triangles that cover part of the target, so a wrong stride or format moves pixels.
    private static readonly float[] Positions =
    [
        -1.0f, -1.0f, 0.5f, -1.0f, -1.0f, 0.25f,
        0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f,
    ];

    private static readonly string VertexSource =
        $$"""
        #include <metal_stdlib>
        using namespace metal;
        constant uint {{Stride}} [[function_constant({{Declared[0].Index}})]];
        constant bool {{Stride}}_set = is_function_constant_defined({{Stride}});
        constant uint {{Word3}} [[function_constant({{Declared[1].Index}})]];
        constant bool {{Word3}}_set = is_function_constant_defined({{Word3}});

        struct Gen5VsOut
        {
            float4 sharpemu_position [[position]];
            float4 color;
        };

        static inline uint load_u16(device const uchar* bytes, uint at)
        {
            return uint(bytes[at]) | (uint(bytes[at + 1u]) << 8);
        }

        static inline uint load_u32(device const uchar* bytes, uint at)
        {
            return load_u16(bytes, at) | (load_u16(bytes, at + 2u) << 16);
        }

        vertex Gen5VsOut gen5_vs(uint vertex_id [[vertex_id]], device const uint* data [[buffer(2)]])
        {
            uint stride = ({{Stride}}_set ? {{Stride}} : data[0]) & 0x3FFFu;
            uint word3 = {{Word3}}_set ? {{Word3}} : data[1];
            device const uchar* bytes = (device const uchar*)data + {{RecordOffset}}u;
            uint at = vertex_id * stride;
            float2 position;
            if ((word3 >> 12) == {{Snorm16x2}}u)
            {
                float2 packed = float2(float(short(load_u16(bytes, at))), float(short(load_u16(bytes, at + 2u))));
                position = max(packed / 32767.0f, float2(-1.0f));
            }
            else
            {
                position = float2(as_type<float>(load_u32(bytes, at)), as_type<float>(load_u32(bytes, at + 4u)));
            }

            Gen5VsOut result = {};
            result.sharpemu_position = float4(position, 0.5f, 1.0f);
            result.color = float4(float(word3 & 0xFu), float((word3 >> 4) & 0xFu), float((word3 >> 8) & 0xFu), 15.0f) / 15.0f;
            return result;
        }
        """;

    private const string PixelSource =
        """
        #include <metal_stdlib>
        using namespace metal;
        struct PixelIn
        {
            float4 position [[position]];
            float4 color;
        };

        fragment float4 gen5_ps(PixelIn in [[stage_in]])
        {
            return in.color;
        }
        """;

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);

    // The switch reads a static of another part of MetalRenderHost: computed on use, it cannot
    // come out false because the parts' static initializers ran in another order.
    [Fact]
    public void TheSwitchFollowsPortableBuffersAndItsVariable() =>
        Assert.Equal(
            MetalRenderHost.PortableBuffers &&
                Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_BUFFER_WORD_CONSTANTS") != "0",
            MetalRenderHost.BufferWordConstants);

    // A variant pipeline is listed under its pair, but is never written as the library file of
    // its modules: that file is what every later draw of the module loads.
    [Fact]
    public void AVariantPipelineIsNeverWrittenAsAModulesLibraryFile()
    {
        var vertex = new UInt128(0x1111, 0x2222);
        var pixel = new UInt128(0x1234, 0x5678);

        Assert.Equal((vertex, pixel), MetalBinaryArchive.LibraryFileKeys(vertex, pixel, specialized: false));
        Assert.Equal((default(UInt128), default(UInt128)), MetalBinaryArchive.LibraryFileKeys(vertex, pixel, specialized: true));
    }

    [Fact]
    public void AVariantIsListedUnderAnotherArchiveKeyThanTheGenericPipeline()
    {
        var vertex = new UInt128(0x1111, 0x2222);
        var pixel = new UInt128(0x1234, 0x5678);
        var first = new UInt128(1, 2);
        var second = new UInt128(3, 4);

        Assert.Equal(pixel, MetalRenderHost.BufferWordArchiveKey(pixel, default));
        Assert.NotEqual(pixel, MetalRenderHost.BufferWordArchiveKey(pixel, first));
        Assert.NotEqual(
            MetalRenderHost.BufferWordArchiveKey(pixel, first), MetalRenderHost.BufferWordArchiveKey(pixel, second));
        // A pipeline without a pixel program lists its variants apart too.
        Assert.NotEqual(default, MetalRenderHost.BufferWordArchiveKey(default, first));
        Assert.NotEqual(
            MetalBinaryArchive.PairKey(vertex, pixel),
            MetalBinaryArchive.PairKey(vertex, MetalRenderHost.BufferWordArchiveKey(pixel, first)));
        // With the LOD statistics counters as well.
        var counters = MetalBinaryArchive.LodStatisticsKey(pixel, counters: true);
        Assert.NotEqual(
            MetalBinaryArchive.PairKey(vertex, counters),
            MetalBinaryArchive.PairKey(vertex, MetalRenderHost.BufferWordArchiveKey(counters, first)));
    }

    [Fact]
    public void TheStateTableKeepsVariantsOfOtherValuesApart()
    {
        var description = Description(new ShaderProgram(1, 1), new ShaderProgram(2, 2));
        var source = new UInt128(7, 8);
        MetalPipelineStateKey Key(uint[]? words, int vertexWords = 2) =>
            MetalRenderHost.PipelineStateKeyOf(
                description, source, source, false, false, 0, bufferWords: words, vertexBufferWords: vertexWords);

        Assert.Equal(Key([12, 0x1000]), Key([12, 0x1000]));
        Assert.NotEqual(Key(null, 0), Key([12, 0x1000]));
        Assert.NotEqual(Key([12, 0x1000]), Key([12, 0x2000]));
        // The same values split between the stages another way are another pair of functions.
        Assert.NotEqual(Key([12, 0x1000], 2), Key([12, 0x1000], 1));
    }

    [Fact]
    public void AFunctionSpecializesWithTheMipSwitchAndBufferWords()
    {
        if (!RequireDevice()) return;
        var source =
            $$"""
            #include <metal_stdlib>
            using namespace metal;
            constant bool {{Gen5MslShader.MipStatisticsFunctionConstantName}} [[function_constant({{Gen5MslShader.MipStatisticsFunctionConstantIndex}})]];
            constant uint {{Stride}} [[function_constant({{Declared[0].Index}})]];
            constant bool {{Stride}}_set = is_function_constant_defined({{Stride}});
            constant uint {{Word3}} [[function_constant({{Declared[1].Index}})]];
            constant bool {{Word3}}_set = is_function_constant_defined({{Word3}});
            fragment float4 gen5_ps(float4 position [[position]], device atomic_uint* counters [[buffer(0)]], constant uint* data [[buffer(1)]])
            {
                if ({{Gen5MslShader.MipStatisticsFunctionConstantName}})
                {
                    atomic_fetch_add_explicit(counters, 1u, memory_order_relaxed);
                }

                uint stride = {{Stride}}_set ? {{Stride}} : data[0];
                uint word3 = {{Word3}}_set ? {{Word3}} : data[1];
                return position * float(stride) + float(word3);
            }
            """;
        var library = MetalRenderHost.TryCompileLibrary(Device.Value, source, usesPerVertexInputs: false, out var libraryError);
        Assert.True(library != 0, libraryError);
        var functions = new List<nint>();
        try
        {
            foreach (var (mip, values) in new (bool, uint[])[] { (false, []), (true, []), (true, [12, 0x1000]), (false, [8, 0x2000]) })
            {
                var function = MetalRenderHost.NewTranslatedFunction(library, "gen5_ps", mip, Declared, values, out var error);
                Assert.True(function != 0, MetalNative.DescribeError(error));
                functions.Add(function);
            }

            Assert.Equal(functions.Count, functions.Distinct().Count());
            Assert.Throws<ArgumentException>(
                () => MetalRenderHost.NewTranslatedFunction(library, "gen5_ps", false, Declared, [12], out _));
        }
        finally
        {
            foreach (var handle in functions.Append(library))
            {
                MetalNative.SendVoid(handle, MetalNative.Selector("release"));
            }
        }
    }

    [Fact]
    public void AnUnstoredLibraryThatDeclaresBufferWordsIsStored()
    {
        if (!RequireDevice()) return;
        var root = Path.Combine(Path.GetTempPath(), $"sharpemu-buffer-words-{Guid.NewGuid():N}");
        try
        {
            using var archive = MetalBinaryArchive.Open(root, Device.Value, "translator-a", 64L << 20);
            // As MetalUnstoredLibraries stores it: everything is read off the source text.
            var declared = Gen5MslShader.ParseBufferWordConstants(VertexSource);
            Assert.Equal(Declared, declared);
            var key = MetalBinaryArchive.LibraryKey(VertexSource, usesPerVertexInputs: false);
            var bytes = MetalStaticPrecompile.Store(
                archive,
                Device.Value,
                VertexSource,
                MetalUnstoredLibraries.EntryPointOf(VertexSource)!,
                usesPerVertexInputs: false,
                MetalUnstoredLibraries.UsesMipStatisticsConstant(VertexSource),
                declared,
                key,
                out var error);
            Assert.True(bytes > 0, error);
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }
    }

    [Fact]
    public void TheFirstDrawOfANewLayoutDrawsGenericWhileItsVariantBuilds()
    {
        if (!RequireDevice()) return;
        using var rig = new MetalPassRig(Device.Value, 1024 * 1024);
        using var gate = new ManualResetEventSlim(false);
        rig.Host.BeforeBufferWordVariantBuild = () => gate.Wait(BuildTimeout);
        var pipeline = Pipeline(rig, 0x7000_0000);
        var layout = Layout(Snorm16x2, 12, 0x0A1);
        var buffer = NewBuffer(layout);
        var stage = Stage(layout, unread: 0);
        try
        {
            var generic = DrawInto(rig, pipeline, FirstTarget, buffer, stage);
            var counters = rig.Host.BufferWordCounters;
            Assert.Equal((0L, 1L, 1L, 0L), (counters.DrawsVariant, counters.DrawsGeneric, counters.DrawsPending, counters.Built));
            AssertDrawn(generic, layout);

            gate.Set();
            Assert.True(rig.Host.WaitForBufferWordVariants(BuildTimeout));
            var specialized = DrawInto(rig, pipeline, FirstTarget + 0x1000, buffer, stage);
            counters = rig.Host.BufferWordCounters;
            Assert.Equal((1L, 1L, 1L), (counters.Built, counters.DrawsVariant, counters.DrawsGeneric));
            Assert.Equal(generic, specialized);
        }
        finally
        {
            gate.Set();
            MetalNative.SendVoid(buffer, MetalNative.Selector("release"));
        }
    }

    [Fact]
    public void TheGenericAndTheSpecializedPipelineDrawTheSamePixelsForPackedLayouts()
    {
        if (!RequireDevice()) return;
        using var rig = new MetalPassRig(Device.Value, 1024 * 1024);
        var pipeline = Pipeline(rig, 0x7100_0000);
        var target = FirstTarget;
        foreach (var layout in new[] { Layout(Snorm16x2, 12, 0x0A1), Layout(Float32x2, 8, 0x3F2), Layout(Snorm16x2, 4, 0xF0F) })
        {
            var buffer = NewBuffer(layout);
            var stage = Stage(layout, unread: 0);
            try
            {
                var before = rig.Host.BufferWordCounters;
                var generic = DrawInto(rig, pipeline, target, buffer, stage);
                Assert.True(rig.Host.WaitForBufferWordVariants(BuildTimeout));
                var specialized = DrawInto(rig, pipeline, target + 0x1000, buffer, stage);
                var after = rig.Host.BufferWordCounters;
                output.WriteLine($"stride={layout.Stride} word3=0x{layout.Word3:X} built={after.Built} variant_draws={after.DrawsVariant}");
                Assert.Equal(before.Built + 1, after.Built);
                Assert.Equal(before.DrawsVariant + 1, after.DrawsVariant);
                AssertDrawn(generic, layout);
                Assert.Equal(generic, specialized);
                target += 0x2000;
            }
            finally
            {
                MetalNative.SendVoid(buffer, MetalNative.Selector("release"));
            }
        }
    }

    [Fact]
    public void TheVariantFollowsOnlyTheWordsTheProgramReads()
    {
        if (!RequireDevice()) return;
        using var rig = new MetalPassRig(Device.Value, 1024 * 1024);
        var pipeline = Pipeline(rig, 0x7200_0000);
        var layout = Layout(Float32x2, 8, 0x123);
        var buffer = NewBuffer(layout);
        try
        {
            DrawInto(rig, pipeline, FirstTarget, buffer, Stage(layout, unread: 1));
            Assert.True(rig.Host.WaitForBufferWordVariants(BuildTimeout));
            Assert.Equal(1L, rig.Host.BufferWordCounters.Built);

            // Another value in a dword the program does not declare: the same variant.
            DrawInto(rig, pipeline, FirstTarget, buffer, Stage(layout, unread: 0xDEAD));
            Assert.True(rig.Host.WaitForBufferWordVariants(BuildTimeout));
            var counters = rig.Host.BufferWordCounters;
            Assert.Equal((1L, 1L), (counters.Built, counters.DrawsVariant));

            // Another value of a declared word: a variant of its own.
            var other = layout with { Word3 = 0x124 };
            DrawInto(rig, pipeline, FirstTarget, buffer, Stage(other, unread: 0xDEAD));
            Assert.True(rig.Host.WaitForBufferWordVariants(BuildTimeout));
            Assert.Equal(2L, rig.Host.BufferWordCounters.Built);
        }
        finally
        {
            MetalNative.SendVoid(buffer, MetalNative.Selector("release"));
        }
    }

    [Fact]
    public void LayoutsPastThePipelinesCapDrawGenericAndAreCounted()
    {
        if (!RequireDevice()) return;
        using var rig = new MetalPassRig(Device.Value, 1024 * 1024);
        using var gate = new ManualResetEventSlim(false);
        rig.Host.BeforeBufferWordVariantBuild = () => gate.Wait(BuildTimeout);
        var pipeline = Pipeline(rig, 0x7300_0000);
        var layout = Layout(Float32x2, 8, 0x100);
        var buffer = NewBuffer(layout);
        try
        {
            for (var index = 0; index <= MetalRenderHost.MaxBufferWordVariantsPerPipeline; index++)
            {
                // Each a format word of its own; the records decode the same.
                DrawInto(rig, pipeline, FirstTarget, buffer, Stage(layout with { Word3 = layout.Word3 + (uint)index }, unread: 0));
            }

            var counters = rig.Host.BufferWordCounters;
            Assert.Equal(MetalRenderHost.MaxBufferWordVariantsPerPipeline + 1L, counters.DrawsGeneric);
            Assert.Equal(1L, counters.DrawsCapped);
            Assert.Equal(MetalRenderHost.MaxBufferWordVariantsPerPipeline, rig.Host.BufferWordVariantsPending);

            gate.Set();
            Assert.True(rig.Host.WaitForBufferWordVariants(BuildTimeout));
            // The layout past the cap stays generic; the others draw their variants.
            DrawInto(rig, pipeline, FirstTarget, buffer, Stage(layout with { Word3 = layout.Word3 + 16 }, unread: 0));
            DrawInto(rig, pipeline, FirstTarget, buffer, Stage(layout, unread: 0));
            counters = rig.Host.BufferWordCounters;
            Assert.Equal((long)MetalRenderHost.MaxBufferWordVariantsPerPipeline, counters.Built);
            Assert.Equal(2L, counters.DrawsCapped);
            Assert.Equal(1L, counters.DrawsVariant);
        }
        finally
        {
            gate.Set();
            MetalNative.SendVoid(buffer, MetalNative.Selector("release"));
        }
    }

    private readonly record struct VertexLayout(uint Format, uint Stride, uint Word3Low)
    {
        public uint Word3 { get; init; } = (Format << 12) | Word3Low;
    }

    private static VertexLayout Layout(uint format, uint stride, uint swizzle) => new(format, stride, swizzle);

    // The words at the head of the buffer the generic function reads, then the records.
    private static nint NewBuffer(VertexLayout layout)
    {
        var buffer = MetalNative.SendNewBuffer(Device.Value, MetalNative.Selector("newBufferWithLength:options:"), BufferBytes, 0);
        Assert.NotEqual(0, buffer);
        var bytes = new Span<byte>((void*)MetalNative.Send(buffer, MetalNative.Selector("contents")), BufferBytes);
        bytes.Clear();
        var words = MemoryMarshal.Cast<byte, uint>(bytes);
        words[0] = layout.Stride;
        words[1] = layout.Word3;
        for (var vertex = 0; vertex < Positions.Length / 2; vertex++)
        {
            var record = bytes[(RecordOffset + vertex * (int)layout.Stride)..];
            for (var component = 0; component < 2; component++)
            {
                var value = Positions[vertex * 2 + component];
                if (layout.Format == Snorm16x2)
                {
                    MemoryMarshal.Write(record[(component * 2)..], (short)MathF.Round(value * 32767f));
                }
                else
                {
                    MemoryMarshal.Write(record[(component * 4)..], value);
                }
            }
        }

        return buffer;
    }

    // The vertex stage's shader data: the two declared words, then one the program never reads.
    // A format word written here and not in the buffer's head would make the generic draw differ.
    private static MetalRenderHost.MetalStageBindings Stage(VertexLayout layout, uint unread)
    {
        var stage = MetalPassRig.Stage();
        stage.ShaderData = [layout.Stride, layout.Word3, unread];
        return stage;
    }

    private static byte[] DrawInto(
        MetalPassRig rig, PipelineHandle pipeline, ulong address, nint buffer, MetalRenderHost.MetalStageBindings stage)
    {
        // The buffer's head holds the words the stage holds: rewrite them for this draw.
        var words = new Span<uint>((void*)MetalNative.Send(buffer, MetalNative.Selector("contents")), 2);
        words[0] = stage.ShaderData[0];
        words[1] = stage.ShaderData[1];
        var target = rig.MakeTarget(address, GpuFormat.R8G8B8A8Unorm, Width, Height);
        rig.Draw(pipeline, [target], bind: host =>
        {
            host.BindVertexBuffers([new BufferBinding((ulong)buffer, 0)], new VertexInputInfo());
            host.NoteBufferWordStages(stage, null);
        });
        rig.Host.FlushCommandBuffer();
        return rig.Read(address, TargetBytes);
    }

    // Some pixels carry the layout's colour and some do not: the triangles landed where they
    // were put, not over the whole target or nowhere.
    private static void AssertDrawn(byte[] pixels, VertexLayout layout)
    {
        var red = (byte)Math.Round((layout.Word3 & 0xF) * 255.0 / 15.0);
        var green = (byte)Math.Round(((layout.Word3 >> 4) & 0xF) * 255.0 / 15.0);
        var drawn = 0;
        for (var at = 0; at < pixels.Length; at += 4)
        {
            if (pixels[at] == red && pixels[at + 1] == green && pixels[at + 3] == 255)
            {
                drawn++;
            }
        }

        Assert.True(drawn > 0 && drawn < pixels.Length / 4, $"drawn={drawn} of {pixels.Length / 4}");
    }

    private static PipelineHandle Pipeline(MetalPassRig rig, ulong id)
    {
        var host = rig.Host;
        var vertex = new ShaderProgram(id, host.CreateShaderModule(
            new MetalCompiledGuestShader(new Gen5MslShader(VertexSource, "gen5_vs", Gen5MslStage.Vertex, 1)
            {
                BufferWordConstants = Declared,
            }),
            ShaderStage.Vertex, id, id));
        var pixel = new ShaderProgram(id + 1, host.CreateShaderModule(
            new MetalCompiledGuestShader(new Gen5MslShader(PixelSource, "gen5_ps", Gen5MslStage.Pixel, 1)),
            ShaderStage.Pixel, id + 1, id + 1));
        return host.CreateGraphicsPipeline(Description(vertex, pixel));
    }

    private static GraphicsPipelineDescription Description(ShaderProgram vertex, ShaderProgram pixel)
    {
        var parameters = new PipelineStaticParameters
        {
            ColorCount = 1,
            Topology = GpuPrimitiveTopology.TriangleList,
            Samples = 1,
            DepthClipEnable = true,
        };
        parameters.SetColorMask(0, 0xF);
        var rendering = new PipelineRenderingState { ColorCount = 1 };
        rendering.ColorFormats[0] = GpuFormat.R8G8B8A8Unorm;
        return new GraphicsPipelineDescription
        {
            Rendering = rendering,
            VertexInput = new PipelineVertexInputState(),
            VertexInfo = new VertexInputInfo(),
            VertexProgram = vertex,
            VertexStage = new ShaderProgramInfo { Stage = ShaderStageKind.Vertex, Hash = vertex.Id },
            PixelProgram = pixel,
            PixelStage = new ShaderProgramInfo { Stage = ShaderStageKind.Pixel, Hash = pixel.Id },
            StaticParameters = parameters,
        };
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Buffer-word variants are compiled and drawn on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }
}
