// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Diagnostics;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

public sealed partial class MslRuntimeFormatLoadTests
{
    private const uint VertexStride = 40;
    private const uint VertexCount = 1024;
    private const int OutputBase = 65536;

    // An interleaved vertex the way Demon's Souls lays its streams out: one V# per attribute
    // (s[0:27]), all with stride 40. (srsrc, offset, components, unified format, swizzle).
    private static readonly (uint Resource, uint Offset, uint Components, uint Format, uint Swizzle)[] VertexAttributes =
    [
        (0, 0, 3, 74, 0xFAC),  // position 32_32_32 FLOAT
        (1, 12, 4, 45, 0xFAC), // normal 10_10_10_2 SNORM
        (2, 16, 4, 51, 0xFAC), // tangent 2_10_10_10 SNORM
        (3, 20, 4, 56, 0xFAC), // colour 8_8_8_8 UNORM
        (4, 24, 2, 29, 0xFAC), // uv0 16_16 FLOAT
        (5, 28, 2, 24, 0xFAC), // uv1 16_16 SNORM
        (6, 32, 4, 65, 0xFAC), // weights 16_16_16_16 UNORM
    ];

    // Each lane fetches every attribute of vertex (lane + i) & 1023 for 256 iterations and sums
    // the words; the sum goes to s[28:31] past the vertices. s32 holds the group id.
    private static Gen5ComputeFixture VertexFetchFixture()
    {
        var words = new List<uint>
        {
            0x7E040300, // v_mov_b32 v2, v0
            0x7E020280, // v_mov_b32 v1, 0
            0xBEA103FF, 256, // s_mov_b32 s33, 256
        };
        var loop = words.Count;
        foreach (var (resource, offset, components, _, _) in VertexAttributes)
        {
            // buffer_load_format v[4:], v2, s[4r:4r+3], offset idxen
            words.AddRange([0xE0002000 | ((components - 1) << 18) | offset, 0x80000402 | (resource << 16)]);
            words.Add(0xBF8C0070);
            for (uint component = 0; component < components; component++)
            {
                words.Add(0x4A020304 + component); // v_add_nc_u32 v1, v(4+c), v1
            }
        }

        words.AddRange([
            0x4A040481, // v_add_nc_u32 v2, 1, v2
            0x360404FF, VertexCount - 1, // v_and_b32 v2, 1023, v2
            0x81A18121, // s_sub_i32 s33, s33, 1
            0xBF078021, // s_cmp_lg_u32 s33, 0
        ]);
        words.Add(0xBF850000 | (ushort)(short)(loop - words.Count - 1));
        words.AddRange([
            0x34140082, // v_lshlrev_b32 v10, 2, v0
            0x7E120220, // v_mov_b32 v9, s32 (group x)
            0x34121287, // v_lshlrev_b32 v9, 7, v9
            0x4A121509, // v_add_nc_u32 v9, v9, v10
            0x4A1212FF, OutputBase, // v_add_nc_u32 v9, 65536, v9
            0xE0701000, 0x80070109, // buffer_store_dword v1, v9, s[28:31] offen
            0xBF810000,
        ]);
        return new("vertex-fetch-benchmark", words.ToArray(), 28, OutputBase + 1024 * 128);
    }

    // portable: the run-time layout (today's default); baked: strides and formats in the source.
    private static (Gen5MslShader Shader, BindingLayout Layout, Dictionary<int, (uint Stride, uint Word3)> Words) CompileVertexFetch(
        bool portable, bool constants)
    {
        var fixture = VertexFetchFixture();
        var program = Gen5ComputeFixtures.DecodeOrThrow(fixture);
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, 0x1234, 0, 32);
        var specialization = ResourceSpecialization.Default(plan.Info);
        var words = new Dictionary<int, (uint, uint)>();
        // The attributes are fetched in order, each from its own V#, and the store comes last.
        var byFirstUse = Enumerable.Range(0, plan.Info.Buffers.Count)
            .OrderBy(binding => plan.Info.Buffers[binding].FirstUsePc).ToArray();
        Assert.Equal(VertexAttributes.Length + 1, byFirstUse.Length);
        for (var resource = 0; resource < VertexAttributes.Length; resource++)
        {
            var binding = byFirstUse[resource];
            var attribute = VertexAttributes[resource];
            Assert.True(plan.Info.Buffers[binding].Formatted);
            words[binding] = (VertexStride, (attribute.Format << 12) | attribute.Swizzle);
            if (!portable)
                specialization.Buffers[binding] = new BufferSpecialization(VertexStride, attribute.Format, attribute.Swizzle);
        }

        Assert.Equal(VertexAttributes.Length, words.Count);
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info,
            BindingLayout.CollectUserDataRegisters(program, 0, 32),
            false, false, false, portableBuffers: portable);
        var request = new ShaderCompileRequest(plan, resources, layout)
        {
            LocalSizeX = 32,
            ComputeSystemRegisters = new Gen5ComputeSystemRegisters(32, null, null, null),
        };
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error,
            new Gen5MslCodegenOptions(ForceRegisterInitialization: true) { BufferWordConstants = constants }), error);
        return (shader, layout, words);
    }

    // The shader data: in the push block when the layout fits it there, else the ShaderData table.
    private static (byte[] Push, IReadOnlyList<DeviceTable>? Tables) VertexPush(
        BindingLayout layout, IReadOnlyDictionary<int, (uint Stride, uint Word3)> words)
    {
        var push = new byte[PushData.ByteSize];
        var block = new uint[Math.Max(layout.ShaderDataDwordCount, 1)];
        if (layout.BufferWordCount != 0)
        {
            foreach (var (binding, (stride, word3)) in words)
            {
                var at = layout.FirstBufferWordDword() + (uint)binding * PortableBufferWord.DwordCount;
                block[at] = stride;
                block[at + 1] = word3;
            }
        }

        if (layout.UsesPushData)
        {
            for (var i = 0; i < block.Length && layout.PushDataStartDword + i < PushData.DwordCount; i++)
                BinaryPrimitives.WriteUInt32LittleEndian(push.AsSpan((int)(layout.PushDataStartDword + i) * 4), block[i]);
            return (push, null);
        }

        return (push, [MetalNative.CreateWordTable(DescriptorBindingKind.ShaderData, block)]);
    }

    [Fact]
    public void BenchmarkBufferWordConstants()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_GPU_BENCH") != "1") return;
        if (!RequireDevice()) return;
        var data = Enumerable.Range(0, VertexFetchFixture().StoreBackingBytes).Select(i => (byte)(i * 37 + 11)).ToArray();
        var portable = CompileVertexFetch(portable: true, constants: false);
        var baked = CompileVertexFetch(portable: false, constants: false);
        var constant = CompileVertexFetch(portable: true, constants: true);
        Assert.Equal(VertexAttributes.Length * 2, constant.Shader.BufferWordConstants.Count);
        Assert.Equal(portable.Layout.FirstBufferWordDword(), constant.Layout.FirstBufferWordDword());
        output.WriteLine($"push={portable.Layout.UsesPushData} shader_data_dwords={portable.Layout.ShaderDataDwordCount}");
        var right = VertexPush(portable.Layout, portable.Words);
        var wrong = VertexPush(constant.Layout, constant.Words.ToDictionary(pair => pair.Key, _ => (0u, 0u)));
        var values = constant.Words
            .SelectMany(pair => new[]
            {
                (PortableBufferWord.ConstantIndex(pair.Key, 0), pair.Value.Stride),
                (PortableBufferWord.ConstantIndex(pair.Key, 1), pair.Value.Word3),
            })
            .ToArray();
        var arms = new (string Name, Gen5MslShader Shader, (byte[] Push, IReadOnlyList<DeviceTable>? Tables) Data, (uint, uint)[]? Constants)[]
        {
            ("a-portable-runtime", portable.Shader, right, null),
            ("b-baked-literal", baked.Shader, VertexPush(baked.Layout, baked.Words), null),
            ("c-constants-unspecialized", constant.Shader, right, []),
            ("d-constants-specialized", constant.Shader, wrong, values),
        };
        var libraries = new nint[arms.Length];
        byte[]? expected = null;
        for (var arm = 0; arm < arms.Length; arm++)
        {
            var (name, shader, (push, tables), constants) = arms[arm];
            var source = shader.Source + $"\n// {Guid.NewGuid()}\n";
            var clock = Stopwatch.StartNew();
            Assert.True(MetalNative.TryCompileLibrary(source, out libraries[arm], out var compileError), compileError);
            var libraryMs = clock.Elapsed.TotalMilliseconds;
            clock.Restart();
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(libraries[arm], shader.EntryPoint, data, push, shader.ArgumentLayout!,
                32, out var result, out _, out var runError, threadgroupCount: 1024, tables: tables, constants: constants), runError);
            var pipelineMs = clock.Elapsed.TotalMilliseconds;
            output.WriteLine(FormattableString.Invariant(
                $"{name} library_ms={libraryMs:F0} pipeline_and_run_ms={pipelineMs:F0} source_bytes={source.Length}"));
            Assert.False(result.AsSpan(OutputBase, 128).SequenceEqual(data.AsSpan(OutputBase, 128)));
            expected ??= result;
            Assert.True(expected.AsSpan(OutputBase).SequenceEqual(result.AsSpan(OutputBase)), $"{name} output differs from arm a");
        }

        var best = new double[arms.Length];
        Array.Fill(best, double.MaxValue);
        for (var round = 0; round < 6; round++)
        {
            var order = Enumerable.Range(0, arms.Length).ToArray();
            if (round % 2 == 1) Array.Reverse(order);
            foreach (var arm in order)
            {
                var (name, shader, (push, tables), constants) = arms[arm];
                Assert.True(MetalNative.TryBenchmark(shader, data, push, 1024, 32, 4, out var result, out var error,
                    tables: tables, constants: constants, library: libraries[arm]), error);
                Assert.False(result.StepCapHit);
                best[arm] = Math.Min(best[arm], result.GpuMilliseconds);
                output.WriteLine(FormattableString.Invariant(
                    $"{name} round={round} gpu_ms={result.GpuMilliseconds:F4} max_threads={result.MaxThreadsPerThreadgroup}"));
            }
        }

        // 1024 groups x 32 lanes x 256 vertices x 4 dispatches.
        const double elements = 1024.0 * 32 * 256 * 4;
        for (var arm = 0; arm < arms.Length; arm++)
        {
            output.WriteLine(FormattableString.Invariant(
                $"best {arms[arm].Name} gpu_ms={best[arm]:F4} ns_per_vertex={best[arm] * 1e6 / elements:F4} vs_b={best[arm] / best[1]:F3}"));
        }
    }
}
