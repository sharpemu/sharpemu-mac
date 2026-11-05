// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Text.RegularExpressions;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

public sealed class MslLaneObservationTests
{
    // A bounds check that narrows EXEC for good (lanes 0..47), then an inner if (lanes 0..19)
    // whose temporary v5 is written again before anything reads it once the if closes, and
    // whose v6 the other lanes read: v5's write runs unguarded, v6's keeps its guard.
    private static readonly uint[] NarrowedInnerIfProgram =
    [
        0xD7650000, 0x000100C1, // v_mbcnt_lo_u32_b32 v0, -1, 0
        0xD7660000, 0x000200C1, // v_mbcnt_hi_u32_b32 v0, -1, v0
        0x34080082,             // v_lshlrev_b32 v4, 2, v0
        0x7E0C02C0,             // v_mov_b32 v6, 64
        0x7DA800B0,             // v_cmpx_gt_u32 exec, 48, v0 (never restored)
        0x7D880094,             // v_cmp_gt_u32 vcc, 20, v0
        0xBE82246A,             // s_and_saveexec_b64 s[2:3], vcc
        0x4A0A0085,             // v_add_nc_u32 v5, 5, v0
        0x4A0C0B05,             // v_add_nc_u32 v6, v5, v5
        0xBEFE0402,             // s_mov_b64 exec, s[2:3]
        0x7E0A0281,             // v_mov_b32 v5, 1
        0x4A0C0D05,             // v_add_nc_u32 v6, v5, v6
        0xE0701000, 0x80020604, // buffer_store_dword v6, v4, s[8:11], 0 offen
        0xBF810000,             // s_endpgm
    ];

    [Fact]
    public void WritesNoMaskedOffLaneReadsBack_RunUnguarded_AndTheSwitchKeepsTheGuards()
    {
        var fixture = new Gen5ComputeFixture("lane-observation", NarrowedInnerIfProgram, 8, 64 * 4);
        var request = Gen5ComputeFixtures.CreateComputeRequest(fixture, 64, 64);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var exact, out var error,
            new Gen5MslCodegenOptions()), error);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var before, out error,
            new Gen5MslCodegenOptions { LaneExactUnobservedWrites = false }), error);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var guarded, out error,
            new Gen5MslCodegenOptions(ExecGuardElision: false)), error);

        Assert.True(UnguardedWrites(exact.Source) > UnguardedWrites(before.Source),
            $"exact {UnguardedWrites(exact.Source)} unguarded writes, before {UnguardedWrites(before.Source)}");

        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            return;
        }

        Assert.True(MetalNative.IsAvailable, "A Metal device is required.");
        var expected = Execute(guarded);
        Assert.Equal(expected, Execute(before));
        var actual = Execute(exact);
        Assert.Equal(expected, actual);
        for (var lane = 0; lane < 64; lane++)
        {
            var value = BinaryPrimitives.ReadUInt32LittleEndian(actual.AsSpan(lane * 4, 4));
            Assert.Equal(lane < 20 ? (uint)(2 * (lane + 5) + 1) : lane < 48 ? 65u : 0xCDCDCDCDu, value);
        }

        byte[] Execute(Gen5MslShader shader)
        {
            Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
            var bytes = Enumerable.Repeat((byte)0xCD, fixture.StoreBackingBytes).ToArray();
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, bytes,
                new byte[Resources.PushData.ByteSize], shader.ArgumentLayout!, 64,
                out var result, out var runError), runError);
            return result;
        }
    }

    // Vector register writes outside any "if (exec)" guard or run, in the entry body.
    private static int UnguardedWrites(string source)
    {
        var count = 0;
        var depth = 0;
        var pending = false;
        var guardDepths = new Stack<bool>();
        foreach (var raw in source.Split('\n'))
        {
            var line = raw.Trim();
            if (line.StartsWith("if (exec) {", StringComparison.Ordinal))
            {
                continue;
            }

            if (line == "if (exec)")
            {
                pending = true;
                continue;
            }

            if (line == "{")
            {
                guardDepths.Push(pending);
                depth += pending ? 1 : 0;
                pending = false;
                continue;
            }

            pending = false;
            if (line.StartsWith('}'))
            {
                if (guardDepths.Count > 0 && guardDepths.Pop())
                {
                    depth--;
                }

                continue;
            }

            if (depth == 0 && Regex.IsMatch(line, @"^v\d+ = "))
            {
                count++;
            }
        }

        return count;
    }
}
