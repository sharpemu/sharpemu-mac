// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

public sealed class MslStaticScratchSlotTests
{
    [Theory]
    [InlineData(7u, 32u, false)]
    [InlineData(7u, 64u, false)]
    [InlineData(46u, 32u, false)]
    [InlineData(46u, 64u, false)]
    [InlineData(46u, 1024u, false)]
    [InlineData(46u, 1024u, true)]
    public void RotatedSlots_PreserveAliasesAndMaskedStores(uint dwords, uint threads, bool relativeRead)
    {
        var request = Request(dwords, threads, relativeRead ? "relative-read" : "");
        var slow = Compile(request, false);
        var fast = Compile(request, true);
        Assert.Contains($"% {dwords}u]", slow.Source, StringComparison.Ordinal);
        Assert.DoesNotContain($"% {dwords}u]", fast.Source, StringComparison.Ordinal);
        Assert.Contains("sharpemu_scratch[0u]", fast.Source, StringComparison.Ordinal);
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1") return;
        Assert.True(MetalNative.IsAvailable, "A Metal device is required.");
        var expected = Execute(slow);
        var actual = Execute(fast);
        Assert.Equal(expected, actual);
        for (var lane = 0; lane < threads; lane++)
        {
            Assert.Equal(lane % 2 == 0 ? 17u : 11u,
                BinaryPrimitives.ReadUInt32LittleEndian(actual.AsSpan(lane * 8, 4)));
            Assert.Equal(13u, BinaryPrimitives.ReadUInt32LittleEndian(actual.AsSpan(lane * 8 + 4, 4)));
        }

        byte[] Execute(Gen5MslShader shader)
        {
            Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var error), error);
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint,
                new byte[threads * 8], new byte[PushData.ByteSize], shader.ArgumentLayout!, threads,
                out var result, out error), error);
            return result;
        }
    }

    [Theory]
    [InlineData("changed-base")]
    [InlineData("mixed-base")]
    [InlineData("unaligned")]
    [InlineData("negative")]
    [InlineData("dynamic")]
    [InlineData("return-to-access")]
    [InlineData("relative-read-base")]
    public void UnprovenAddressing_KeepsDynamicScratch(string variant)
    {
        var request = Request(46, 64, variant);
        var shader = Compile(request, true);
        Assert.Contains("% 46u]", shader.Source, StringComparison.Ordinal);
    }

    private static Gen5MslShader Compile(ShaderCompileRequest request, bool enabled)
    {
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error,
            new Gen5MslCodegenOptions { StaticScratchSlots = enabled }), error);
        return shader;
    }

    private static ShaderCompileRequest Request(uint dwords, uint threads, string variant = "")
    {
        uint[] relative = variant is "relative-read" or "relative-read-base"
            ? [0xBEFC0380, variant == "relative-read" ? 0x7E0C8700u : 0x7E008700u]
            : [];
        var fixture = new Gen5ComputeFixture("static-scratch",
        [
            .. relative,
            0x34080083,             // v_lshlrev_b32 v4, 3, v0 (local id * 8)
            0x7E04028B,             // v_mov_b32 v2, 11
            0xDC704000, 0x007D0200, // scratch_store_dword v0, v2, off
            0x7E04028D,             // v_mov_b32 v2, 13
            0xDC704004, 0x007D0200, // scratch_store_dword v0, v2, off offset:4
            0xBEFE03FF, 0x55555555, // s_mov_b32 exec_lo, alternating lanes
            0x7E040291,             // v_mov_b32 v2, 17
            0xDC704000 | (dwords * 4), 0x007D0200, // wrap to the first slot
            0xBEFE04C1,             // restore all lanes
            0xDC304000, 0x037D0000, // scratch_load_dword v3, v0, off
            0xDC304004, 0x057D0000, // scratch_load_dword v5, v0, off offset:4
            0x7E000280,             // v0 may change after the final scratch access
            0xE0701000, 0x80020304, // store v3 at local id * 8
            0xE0701004, 0x80020504, // store v5 at local id * 8 + 4
            0xBF810000,
        ], 8, checked((int)threads * 8));
        var program = Gen5ComputeFixtures.DecodeOrThrow(fixture);
        var instructions = program.Instructions.ToArray();
        var index = Array.FindIndex(instructions, i => i.Opcode.StartsWith("ScratchStore", StringComparison.Ordinal));
        var first = instructions[index];
        var control = (Gen5GlobalMemoryControl)first.Control!;
        if (variant == "changed-base")
        {
            instructions[0] = new Gen5ShaderInstruction(instructions[0].Pc, Gen5ShaderEncoding.Vop1,
                "VMovB32", [0u], [new Gen5Operand(Gen5OperandKind.EncodedConstant, 128)], [Gen5Operand.Vector(0)], null);
        }
        else if (variant == "return-to-access")
        {
            var branch = instructions[^1];
            var displacement = unchecked((ushort)(short)((int)(first.Pc - branch.Pc - 4) / 4));
            instructions[^1] = new Gen5ShaderInstruction(branch.Pc, Gen5ShaderEncoding.Sopp,
                "SBranch", [0xBF820000u | displacement], [], [], null);
        }
        else
        {
            control = variant switch
            {
                "mixed-base" => control with { VectorAddress = 1 },
                "unaligned" => control with { OffsetBytes = 2 },
                "negative" => control with { OffsetBytes = -4 },
                "dynamic" => control with { DynamicOffsetRegister = 0 },
                _ => control,
            };
            instructions[index] = first with { Control = control };
        }
        program = new Gen5ShaderProgram(program.Address, instructions);
        var data = new uint[16];
        data[8] = 0x200000;
        data[10] = threads * 8;
        data[11] = DescriptorConstants.IdentityDestinationSelect;
        return Gen5ComputeFixtures.RequestOrThrow(program, ShaderStage.Compute,
            localSizeX: threads, scratchDwords: dwords, runtimeInputs: new ResourceRuntimeInputs { UserData = data });
    }
}
