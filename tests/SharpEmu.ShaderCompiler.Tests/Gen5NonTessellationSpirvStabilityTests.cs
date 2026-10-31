// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Security.Cryptography;
using SharpEmu.ShaderCompiler.Resources;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests;

// Baselines captured on main at 669e54f4 before the tessellation port. Vertex, pixel and
// compute output, including both wave sizes and LDS paths, must remain byte-identical.
// The four pixel baselines are this fork's own pixel output (origin/main f205bde48), which
// differs from upstream's.
public sealed class Gen5NonTessellationSpirvStabilityTests
{
    [Theory]
    [InlineData("lds-readback", ShaderStage.Vertex, 64u, "C730F364D289ADF2")]
    [InlineData("lds-readback", ShaderStage.Vertex, 32u, "935715DD89E69C3B")]
    [InlineData("lds-readback", ShaderStage.Pixel, 64u, "E6FB8CD7093E3685")]
    [InlineData("lds-readback", ShaderStage.Pixel, 32u, "AD4D3A38453C3A73")]
    [InlineData("lds-readback", ShaderStage.Compute, 64u, "C9BC95825DCD2B6B")]
    [InlineData("lds-readback", ShaderStage.Compute, 32u, "8189688104D5FE93")]
    [InlineData("lds-staging-export", ShaderStage.Vertex, 64u, "C68BEA09038EA043")]
    [InlineData("lds-staging-export", ShaderStage.Vertex, 32u, "3AAFB4314FD18710")]
    [InlineData("lds-barrier", ShaderStage.Compute, 64u, "AEE924497E4633E2")]
    [InlineData("lds-barrier", ShaderStage.Compute, 32u, "B56973BC44D465FB")]
    [InlineData("divergent", ShaderStage.Vertex, 64u, "C9F0996EAD34F8B6")]
    [InlineData("divergent", ShaderStage.Pixel, 32u, "E284FBB5227436C3")]
    [InlineData("divergent", ShaderStage.Compute, 64u, "5CD8E2752F0745FE")]
    [InlineData("image-load", ShaderStage.Pixel, 64u, "96C588AB90CA0F72")]
    [InlineData("image-load", ShaderStage.Compute, 32u, "D5D535F158180FB0")]
    public void SpirvIsUnchanged(string name, ShaderStage stage, uint waveSize, string expected)
    {
        var (plan, resources, layout) = Prepare(Programs[name](), stage);
        var request = new ShaderCompileRequest(plan, resources, layout)
        {
            PixelOutputs = stage == ShaderStage.Pixel ? [new Gen5PixelOutputBinding(0, 0, Gen5PixelOutputKind.Float)] : [],
            WaveSize = waveSize,
            LocalSizeX = stage == ShaderStage.Compute ? 128u : 0u,
        };
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.Equal(expected, Convert.ToHexString(SHA256.HashData(shader.Spirv), 0, 8));
    }

    private static readonly Dictionary<string, Func<Gen5ShaderProgram>> Programs = new()
    {
        ["lds-readback"] = () => Gen5DataShareThreadWriteTests.CreateReadbackProgram(0xABCD0010, 0x100, true),
        // NGG-style staging: write two values at a lane-derived address, read them back, export.
        ["lds-staging-export"] = () => Program(
            Vop2(0, "VLshlrevB32", 10, Operand(3), Gen5Operand.Vector(7)),
            DataShare(4, "DsWriteB32", false, [Gen5Operand.Vector(10), Gen5Operand.Vector(5)], []),
            DataShare(12, "DsWriteB32", false, [Gen5Operand.Vector(10), Gen5Operand.Vector(6)], [], 4),
            DataShare(20, "DsReadB32", false, [Gen5Operand.Vector(10)], [11]),
            DataShare(28, "DsReadB32", false, [Gen5Operand.Vector(10)], [12], 4),
            new Gen5ShaderInstruction(36, Gen5ShaderEncoding.Exp, "Exp", [0u, 0u],
                [Gen5Operand.Vector(11), Gen5Operand.Vector(12), Gen5Operand.Vector(8), Gen5Operand.Source(242)], [],
                new Gen5ExportControl(12, 15, false, true, false)),
            EndProgram(44)),
        ["lds-barrier"] = () => Program(
            Vop2(0, "VLshlrevB32", 5, Operand(2), Gen5Operand.Vector(0)),
            Vop2(4, "VAddI32", 4, Operand(37), Gen5Operand.Vector(0)),
            DataShare(8, "DsWriteB32", false, [Gen5Operand.Vector(5), Gen5Operand.Vector(4)], []),
            new Gen5ShaderInstruction(16, Gen5ShaderEncoding.Sopp, "SBarrier", [0u], [], [], null),
            Vop2(20, "VXorB32", 6, Operand(64), Gen5Operand.Vector(0)),
            Vop2(24, "VLshlrevB32", 6, Operand(2), Gen5Operand.Vector(6)),
            DataShare(28, "DsReadB32", false, [Gen5Operand.Vector(6)], [4]),
            BufferAccess(36, "BufferStoreDword", 4, vectorData: 4, offsetEnabled: true, vectorAddress: 5),
            EndProgram(44)),
        ["divergent"] = () => Program(
            new Gen5ShaderInstruction(0, Gen5ShaderEncoding.Vop1, "VReadfirstlaneB32", [0u],
                [Gen5Operand.Vector(0)], [Gen5Operand.Scalar(20)], null),
            Sopc(4, "SCmpLtU32", Gen5Operand.Scalar(20), Operand(64)),
            Branch(8, "SCbranchScc0", 5),
            MoveScalar(12, 21, 111),
            Vop1(16, "VMovB32", 4, Operand(333)),
            Branch(20, "SBranch", 4),
            Nop(24), Nop(28),
            MoveScalar(32, 21, 222),
            Vop1(36, "VMovB32", 4, Operand(444)),
            Vop2(40, "VAddI32", 4, Gen5Operand.Scalar(21), Gen5Operand.Vector(4)),
            Vop2(44, "VLshlrevB32", 5, Operand(2), Gen5Operand.Vector(0)),
            BufferAccess(48, "BufferStoreDword", 4, vectorData: 4, offsetEnabled: true, vectorAddress: 5),
            EndProgram(56)),
        ["image-load"] = () => Program(
            Image(0, "ImageLoad", 4, dmask: 1, vectorAddress: 1),
            Vop2(8, "VLshlrevB32", 5, Operand(2), Gen5Operand.Vector(0)),
            BufferAccess(12, "BufferStoreDword", 12, vectorData: 4, offsetEnabled: true, vectorAddress: 5),
            EndProgram(20)),
    };
}
