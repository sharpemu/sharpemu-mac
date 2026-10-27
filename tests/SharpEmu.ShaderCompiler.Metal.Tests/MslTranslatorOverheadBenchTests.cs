// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// GPU time of the per-instruction overheads the emitted MSL carried, before and after each
/// codegen switch, on the device (round-2 perf audit, workstream k21). Translated programs are
/// compiled both ways in one process and timed with the command buffer's clock, best of three
/// after a warm-up; where the overhead is a pattern rather than an opcode the kernel is the
/// pattern written out in MSL. Opt-in, because a timing is not a pass/fail property:
/// SHARPEMU_GPU_BENCH=1 with SHARPEMU_TEST_REQUIRE_DEVICE=1, and read the output
/// (dotnet test --filter MslTranslatorOverheadBench --logger "console;verbosity=detailed").
/// </summary>
public sealed class MslTranslatorOverheadBenchTests(ITestOutputHelper output)
{
    private const uint Groups = 4096;
    private const uint Threads = 64;
    private const uint Iterations = 4;
    private const int Rounds = 3;

    private static bool Enabled =>
        Environment.GetEnvironmentVariable("SHARPEMU_GPU_BENCH") == "1" &&
        Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1" &&
        MetalNative.IsAvailable;

    // 256 trips of S_BUFFER_LOAD_DWORDX4 and BUFFER_LOAD_DWORDX4, all eight dwords summed.
    private static readonly Gen5ComputeFixture Loads = new(
        "bench-loads",
        [
            0x34120084, // v_lshlrev_b32 v9, 4, v0
            0x361212FF, 0x000003F0, // v_and_b32 v9, 0x3f0, v9
            0x7E020280, // v_mov_b32 v1, 0
            0xBE8003FF, 0x00000100, // s_mov_b32 s0, 256
            0xBE810380, // s_mov_b32 s1, 0
            // loop:
            0xF4280304, 0x02000000, // s_buffer_load_dwordx4 s[12:15], s[8:11], s1
            0xE0381000, 0x80020409, // buffer_load_dwordx4 v[4:7], v9, s[8:11], 0 offen
            0xBF8C0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
            0x4A020304, // v_add_nc_u32 v1, v4, v1
            0x4A020305, // v_add_nc_u32 v1, v5, v1
            0x4A020306, // v_add_nc_u32 v1, v6, v1
            0x4A020307, // v_add_nc_u32 v1, v7, v1
            0x4A02020C, // v_add_nc_u32 v1, s12, v1
            0x4A02020D, // v_add_nc_u32 v1, s13, v1
            0x4A02020E, // v_add_nc_u32 v1, s14, v1
            0x4A02020F, // v_add_nc_u32 v1, s15, v1
            0x4A121290, // v_add_nc_u32 v9, 16, v9
            0x361212FF, 0x000003F0, // v_and_b32 v9, 0x3f0, v9
            0x80019001, // s_add_u32 s1, s1, 16
            0x8701FF01, 0x000003F0, // s_and_b32 s1, s1, 0x3f0
            0x81808100, // s_sub_i32 s0, s0, 1
            0xBF078000, // s_cmp_lg_u32 s0, 0
            0xBF85FFEA, // s_cbranch_scc1 loop
            0xE0700000, 0x80020100, // buffer_store_dword v1, off, s[8:11], 0
            0xBF810000, // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 4096);

    // 256 trips of twelve f32 min/max/med3 between registers and against 0, 1.0 and -1.0.
    private static readonly Gen5ComputeFixture MinMax = new(
        "bench-minmax",
        [
            0x7E020D00, // v_cvt_f32_u32 v1, v0
            0x100202FF, 0x3C23D70A, // v_mul_f32 v1, 0x3c23d70a, v1
            0x7E0402F0, // v_mov_b32 v2, 0.5
            0x7E100280, // v_mov_b32 v8, 0
            0xBE8003FF, 0x00000100, // s_mov_b32 s0, 256
            // loop:
            0x20060501, // v_max_f32 v3, v1, v2
            0x1E081103, // v_min_f32 v4, v3, v8
            0x200A0880, // v_max_f32 v5, 0, v4
            0x1E0C0AF2, // v_min_f32 v6, 1.0, v5
            0xD5570007, 0x03C90106, // v_med3_f32 v7, v6, 0, 1.0
            0x20060307, // v_max_f32 v3, v7, v1
            0x1E080503, // v_min_f32 v4, v3, v2
            0x200A0880, // v_max_f32 v5, 0, v4
            0x1E0C0AF2, // v_min_f32 v6, 1.0, v5
            0xD5570007, 0x03C9E706, // v_med3_f32 v7, v6, -1.0, 1.0
            0xD5540003, 0x02020307, // v_max3_f32 v3, v7, v1, 0
            0xD5510004, 0x03CA0503, // v_min3_f32 v4, v3, v2, 1.0
            0x06020304, // v_add_f32 v1, v4, v1
            0x08100F01, // v_sub_f32 v8, v1, v7
            0x81808100, // s_sub_i32 s0, s0, 1
            0xBF078000, // s_cmp_lg_u32 s0, 0
            0xBF85FFEB, // s_cbranch_scc1 loop
            0xE0700000, 0x80020100, // buffer_store_dword v1, off, s[8:11], 0
            0xBF810000, // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 16);

    // 256 trips of two each of V_RCP_F32, V_RSQ_F32 and V_SQRT_F32.
    private static readonly Gen5ComputeFixture Reciprocals = new(
        "bench-rcp",
        [
            0x7E020D00, // v_cvt_f32_u32 v1, v0
            0x060202F2, // v_add_f32 v1, 1.0, v1
            0xBE8003FF, 0x00000100, // s_mov_b32 s0, 256
            // loop:
            0x7E045501, // v_rcp_f32 v2, v1
            0x7E065D01, // v_rsq_f32 v3, v1
            0x7E086701, // v_sqrt_f32 v4, v1
            0x060A0702, // v_add_f32 v5, v2, v3
            0x060A0905, // v_add_f32 v5, v5, v4
            0x7E045505, // v_rcp_f32 v2, v5
            0x7E065D05, // v_rsq_f32 v3, v5
            0x7E086705, // v_sqrt_f32 v4, v5
            0x060A0702, // v_add_f32 v5, v2, v3
            0x06020905, // v_add_f32 v1, v5, v4
            0x81808100, // s_sub_i32 s0, s0, 1
            0xBF078000, // s_cmp_lg_u32 s0, 0
            0xBF85FFF3, // s_cbranch_scc1 loop
            0xE0700000, 0x80020100, // buffer_store_dword v1, off, s[8:11], 0
            0xBF810000, // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 16);

    // 1024 trips of V_MAD_F32 and V_MUL_F32 + V_ADD_F32 pairs a contraction could fuse.
    private static readonly Gen5ComputeFixture MultiplyAdds = new(
        "bench-mad",
        [
            0x7E020D00, // v_cvt_f32_u32 v1, v0
            0x7E0402FF, 0x3F7FF972, // v_mov_b32 v2, 0x3f7ff972
            0x7E0602FF, 0x3A83126F, // v_mov_b32 v3, 0x3a83126f
            0xBE8003FF, 0x00000400, // s_mov_b32 s0, 1024
            // loop:
            0xD5410001, 0x040E0501, // v_mad_f32 v1, v1, v2, v3
            0x10080501, // v_mul_f32 v4, v1, v2
            0x06080704, // v_add_f32 v4, v4, v3
            0xD5410001, 0x04060504, // v_mad_f32 v1, v4, v2, v1
            0x100A0301, // v_mul_f32 v5, v1, v1
            0x06020705, // v_add_f32 v1, v5, v3
            0x81808100, // s_sub_i32 s0, s0, 1
            0xBF078000, // s_cmp_lg_u32 s0, 0
            0xBF85FFF5, // s_cbranch_scc1 loop
            0xE0700000, 0x80020100, // buffer_store_dword v1, off, s[8:11], 0
            0xBF810000, // s_endpgm
        ],
        StoreScalarResourceBase: 8,
        StoreBackingBytes: 16);

    private static Gen5MslShader Compile(Gen5ComputeFixture fixture, Gen5MslCodegenOptions options)
    {
        var request = Gen5ComputeFixtures.CreateComputeRequest(fixture, localSizeX: Threads);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error, options), error);
        return shader;
    }

    private double Time(Gen5MslShader shader, string label, int bufferBytes, bool multipleOfWidth = false)
    {
        Assert.True(
            MetalNative.TryBenchmark(
                shader, new byte[bufferBytes], new byte[Resources.PushData.ByteSize], Groups, Threads, Iterations,
                out var result, out var error, groupSizeIsMultipleOfWidth: multipleOfWidth),
            error);
        output.WriteLine($"{label,-44} {result.GpuMilliseconds,9:F3} ms GPU   maxThreads {result.MaxThreadsPerThreadgroup}");
        return result.GpuMilliseconds;
    }

    private void Compare(Gen5ComputeFixture fixture, string name, Gen5MslCodegenOptions before, Gen5MslCodegenOptions after, int bufferBytes)
    {
        var old = Compile(fixture, before);
        var current = Compile(fixture, after);
        Assert.NotEqual(old.Source, current.Source);
        var (bestOld, bestNew) = (double.MaxValue, double.MaxValue);
        for (var round = 0; round < Rounds; round++)
        {
            bestOld = Math.Min(bestOld, Time(old, $"{name}, off", bufferBytes));
            bestNew = Math.Min(bestNew, Time(current, $"{name}, on", bufferBytes));
        }

        output.WriteLine($"{name}: {bestOld:F3} -> {bestNew:F3} ms ({(bestNew - bestOld) / bestOld * 100:+0.0;-0.0}%)");
    }

    private static readonly Gen5MslCodegenOptions Baseline = new(
        VectorDwordLoads: false, ThreadgroupSizeHint: false, NativeFloatMinMax: false, ConstantTableSlots: false,
        FastReciprocals: false, ContractPixelCompute: false);

    [Fact]
    public void VectorDwordLoads()
    {
        if (!Enabled) return;
        Compare(Loads, "s_buffer/buffer_load_dwordx4", Baseline, Baseline with { VectorDwordLoads = true }, 4096);
    }

    [Fact]
    public void NativeFloatMinMax()
    {
        if (!Enabled) return;
        Compare(MinMax, "f32 min/max/med3", Baseline, Baseline with { NativeFloatMinMax = true }, 16);
    }

    [Fact]
    public void FastReciprocals()
    {
        if (!Enabled) return;
        Compare(Reciprocals, "rcp/rsq/sqrt f32", Baseline, Baseline with { FastReciprocals = true }, 16);
    }

    [Fact]
    public void ContractPixelCompute()
    {
        if (!Enabled) return;
        Compare(MultiplyAdds, "mad/mul+add f32", Baseline, Baseline with { ContractPixelCompute = true }, 16);
    }

    [Fact]
    public void ThreadgroupSizeHintOnTranslatedPrograms()
    {
        if (!Enabled) return;
        foreach (var (fixture, name, bytes) in new[] { (Loads, "loads", 4096), (MinMax, "minmax", 16), (Reciprocals, "rcp", 16) })
        {
            var old = Compile(fixture, Baseline);
            var hinted = Compile(fixture, Baseline with { ThreadgroupSizeHint = true });
            Assert.Contains("[[max_total_threads_per_threadgroup(64)]]", hinted.Source, StringComparison.Ordinal);
            for (var round = 0; round < Rounds; round++)
            {
                Time(old, $"{name}, no hint", bytes);
                Time(hinted, $"{name}, max_total_threads 64", bytes);
                Time(hinted, $"{name}, + multiple of SIMD width", bytes, multipleOfWidth: true);
            }
        }
    }

    // A register-heavy kernel (96 floats live across a loop), the shape where the register
    // budget the hint buys matters.
    private static string RegisterHeavySource(bool hint)
    {
        const int live = 96;
        var source = new StringBuilder();
        source.AppendLine("#include <metal_stdlib>\nusing namespace metal;");
        source.AppendLine($"{(hint ? "[[max_total_threads_per_threadgroup(64)]] " : string.Empty)}kernel void bench_regs(device uint* b [[buffer(0)]], uint tid [[thread_position_in_grid]])");
        source.AppendLine("{");
        source.AppendLine($"    float r[{live}];");
        source.AppendLine($"    for (int i = 0; i < {live}; i++) r[i] = float(tid + (uint)i) * 0.001f;");
        source.AppendLine("    for (int k = 0; k < 32; k++)");
        source.AppendLine("    {");
        for (var i = 0; i < live; i++)
        {
            source.AppendLine($"        r[{i}] = fma(r[{i}], r[{(i + 1) % live}], r[{(i + 37) % live}]);");
        }

        source.AppendLine("    }");
        source.AppendLine("    float sum = 0.0f;");
        source.AppendLine($"    for (int i = 0; i < {live}; i++) sum += r[i];");
        source.AppendLine("    b[tid & 255u] = as_type<uint>(sum);");
        source.AppendLine("}");
        return source.ToString();
    }

    [Fact]
    public void ThreadgroupSizeHintOnARegisterHeavyKernel()
    {
        if (!Enabled) return;
        for (var round = 0; round < Rounds; round++)
        {
            foreach (var (hint, width) in new[] { (false, false), (true, false), (true, true) })
            {
                Assert.True(MetalNative.TryBenchmarkKernel(RegisterHeavySource(hint), "bench_regs", new byte[1024], Groups, Threads, Iterations,
                    out var ms, out var maxThreads, out var error, width), error);
                output.WriteLine($"96 live floats, hint {hint}, multiple of width {width}: {ms,9:F3} ms GPU   maxThreads {maxThreads}");
            }
        }
    }

    // The flattened-table pattern: 128 fixed-slot reads per trip, with and without the bounds
    // select the translator put on every one (the word count is read at run time, as the
    // argument buffer's is), through the constant address space the table lives in.
    private static string TableSource(bool select)
    {
        var source = new StringBuilder();
        source.AppendLine("#include <metal_stdlib>\nusing namespace metal;");
        source.AppendLine("kernel void bench_table(device uint* b [[buffer(0)]], constant uint* t [[buffer(1)]], uint tid [[thread_position_in_grid]])");
        source.AppendLine("{");
        source.AppendLine("    constant uint* table = t + 16;");
        source.AppendLine("    uint words = t[0];");
        source.AppendLine("    uint acc = tid;");
        source.AppendLine("    for (uint k = 0; k < 64u; k++)");
        source.AppendLine("    {");
        for (var slot = 0; slot < 128; slot++)
        {
            source.AppendLine(select
                ? $"        acc = (acc ^ ({slot}u < words ? table[{slot}u] : 0u)) * 0x9E3779B1u;"
                : $"        acc = (acc ^ table[{slot}u]) * 0x9E3779B1u;");
        }

        source.AppendLine("    }");
        source.AppendLine("    b[4096u + (tid & 255u)] = acc;");
        source.AppendLine("}");
        return source.ToString();
    }

    [Fact]
    public void FlattenedTableSelect()
    {
        if (!Enabled) return;
        var buffer = new byte[5120 * sizeof(uint)];
        BitConverter.TryWriteBytes(buffer, 200u);
        for (var round = 0; round < Rounds; round++)
        {
            foreach (var select in new[] { true, false })
            {
                Assert.True(MetalNative.TryBenchmarkKernel(TableSource(select), "bench_table", buffer, Groups, Threads, Iterations,
                    out var ms, out _, out var error), error);
                output.WriteLine($"128 table words x64, select {select}: {ms,9:F3} ms GPU");
            }
        }
    }
}
