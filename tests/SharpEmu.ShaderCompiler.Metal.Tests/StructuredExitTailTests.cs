// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// A pixel program whose early-out branches reach one S_ENDPGM tail from several places has two
/// exits and no join to nest around. The structured emitter writes the tail into each arm
/// rather than sending the shader to the PC dispatcher.
/// </summary>
public sealed class StructuredExitTailTests
{
    [Fact]
    public void SharedExitTailIsCopiedIntoEachEarlyOutArm()
    {
        // block 0: branch to the tail or fall into block 1; block 1: the same; block 2 and the tail end the program.
        var program = new Gen5ShaderProgram(0, [
            new(0, Gen5ShaderEncoding.Sopp, "SCbranchScc0", [0xBF82FFFF], [], [], new ShaderLinkedBranchControl(12)),
            new(4, Gen5ShaderEncoding.Sopp, "SCbranchScc0", [0xBF82FFFF], [], [], new ShaderLinkedBranchControl(12)),
            new(8, Gen5ShaderEncoding.Sopp, "SEndpgm", [0xBF810000], [], [], null),
            new(12, Gen5ShaderEncoding.Sopp, "SEndpgm", [0xBF810000], [], [], null)]);
        var request = Gen5ComputeFixtures.RequestOrThrow(program, ShaderStage.Pixel);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.DoesNotContain("switch (pc)", shader.Source, StringComparison.Ordinal);
    }
}
