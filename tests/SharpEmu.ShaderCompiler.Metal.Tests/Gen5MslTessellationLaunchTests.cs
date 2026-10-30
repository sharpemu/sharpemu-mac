// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text.RegularExpressions;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// A tessellated draw whose hull stage is a compute kernel (Gen5TessellationLaunch): the merged local/hull
// program starts as the hardware's hull stage would start it, one guest wave64 per threadgroup of whole
// patches, and the domain program is the host tessellator's post-tessellation vertex function. Grand Theft Auto V's palm
// trunks: 3 control points, 21 patches a group, 32 KB of the off-chip buffer and 16 factor bytes a patch.
public sealed class Gen5MslTessellationLaunchTests(ITestOutputHelper output)
{
    private static readonly uint[] EndProgram =
    [
        0xBF810000,             // s_endpgm
    ];

    [Fact]
    public void TheHullKernelStartsEachLaneOnItsPatchAndControlPoint()
    {
        var source = Compile(ShaderStage.Compute,
            new Gen5TessellationLaunch(Gen5TessellationRole.Hull, 21, 3, 0x8000, 16));

        Assert.Contains("constant uint* sharpemu_hull_launch [[buffer(2)]]", source, StringComparison.Ordinal);
        Assert.Contains("device const uint* sharpemu_hull_vertices [[buffer(3)]]", source, StringComparison.Ordinal);
        // The group's first patch, and only the patches the draw has in its last group.
        Assert.Contains("uint sharpemu_hull_first = sharpemu_group_id.x * 21u;", source, StringComparison.Ordinal);
        Assert.Contains("? min(21u, sharpemu_hull_launch[0] - sharpemu_hull_first) : 0u;", source, StringComparison.Ordinal);
        // Off-chip offset, merged wave info (local and hull counts), factor offset.
        Assert.Contains("s2 = sharpemu_group_id.x * 32768u;", source, StringComparison.Ordinal);
        Assert.Contains("s3 = sharpemu_hull_points | (sharpemu_hull_points << 8);", source, StringComparison.Ordinal);
        Assert.Contains("s4 = sharpemu_hull_first * 16u;", source, StringComparison.Ordinal);
        // Relative patch [7:0], control point [12:8]; the local half's vertex from the host's ids.
        Assert.Contains("v1 = (sharpemu_hull_lane / 3u) | ((sharpemu_hull_lane % 3u) << 8);", source, StringComparison.Ordinal);
        // Every instance's patches, instance after instance: the patch within its instance, its vertices, the instance.
        Assert.Contains("uint sharpemu_hull_instance_patch = sharpemu_hull_patch % sharpemu_hull_instance_patches;", source, StringComparison.Ordinal);
        Assert.Contains("v0 = sharpemu_hull_instance_patch;", source, StringComparison.Ordinal);
        Assert.Contains("? sharpemu_hull_vertices[sharpemu_hull_instance_patch * 3u + sharpemu_hull_lane % 3u] : 0u;", source, StringComparison.Ordinal);
        Assert.Contains("v5 = sharpemu_hull_launch[1] + sharpemu_hull_patch / sharpemu_hull_instance_patches;", source, StringComparison.Ordinal);
        // Not the ordinary compute launch.
        Assert.DoesNotContain("v0 = sharpemu_local_id.x;", source, StringComparison.Ordinal);
    }

    [Fact]
    public void TheDomainProgramIsAPostTessellationVertexFunction()
    {
        var source = Compile(ShaderStage.Vertex,
            new Gen5TessellationLaunch(Gen5TessellationRole.Domain, 21, 3, 0x8000, 16,
                Gen5TessellationPartition.FractionalOdd, OutputClockwise: true, MaxFactor: 15));

        // The tessellator's state is in the source, so domain programs differing only in it never share a library.
        Assert.Contains("// tessellator: partition=FractionalOdd output=cw max=15\n[[patch(triangle, 3)]]\nvertex Gen5VsOut",
            source.Replace("\r\n", "\n"), StringComparison.Ordinal);
        Assert.Contains("uint sharpemu_patch_id [[patch_id]],", source, StringComparison.Ordinal);
        Assert.Contains("float3 sharpemu_patch_position [[position_in_patch]],", source, StringComparison.Ordinal);
        Assert.DoesNotContain("[[vertex_id]]", source, StringComparison.Ordinal);
        // u and v, the patch within its hull group, the patch, and that group's off-chip offset.
        Assert.Contains("v5 = as_type<uint>(sharpemu_patch_position.x);", source, StringComparison.Ordinal);
        Assert.Contains("v6 = as_type<uint>(sharpemu_patch_position.y);", source, StringComparison.Ordinal);
        // The patch among every instance's patches the hull kernel ran over.
        Assert.Contains("constant uint* sharpemu_domain_launch [[buffer(2)]]", source, StringComparison.Ordinal);
        Assert.Contains(
            "uint sharpemu_domain_patch = (sharpemu_instance_id - sharpemu_domain_launch[1]) * sharpemu_domain_launch[0] + sharpemu_patch_id;",
            source, StringComparison.Ordinal);
        Assert.Contains("v7 = sharpemu_domain_patch % 21u;", source, StringComparison.Ordinal);
        Assert.Contains("v8 = sharpemu_patch_id;", source, StringComparison.Ordinal);
        Assert.Contains("s4 = (sharpemu_domain_patch / 21u) * 32768u;", source, StringComparison.Ordinal);
    }

    // Ghost of Tsushima's tessellated meshes: quad patches of 4 control points, 16 patches a group.
    [Fact]
    public void AQuadDomainProgramTakesTheSquaresUAndV()
    {
        var source = Compile(ShaderStage.Vertex,
            new Gen5TessellationLaunch(Gen5TessellationRole.Domain, 16, 4, 0x8000, 24,
                Gen5TessellationPartition.FractionalOdd, MaxFactor: 15, Quad: true));

        Assert.Contains("// tessellator: partition=FractionalOdd output=ccw max=15\n[[patch(quad, 4)]]\nvertex Gen5VsOut",
            source.Replace("\r\n", "\n"), StringComparison.Ordinal);
        Assert.Contains("float2 sharpemu_patch_position [[position_in_patch]],", source, StringComparison.Ordinal);
        Assert.Contains("v5 = as_type<uint>(sharpemu_patch_position.x);", source, StringComparison.Ordinal);
        Assert.Contains("v6 = as_type<uint>(sharpemu_patch_position.y);", source, StringComparison.Ordinal);
        Assert.Contains("v7 = sharpemu_domain_patch % 16u;", source, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(Gen5TessellationRole.Hull, false)]
    [InlineData(Gen5TessellationRole.Domain, false)]
    [InlineData(Gen5TessellationRole.Domain, true)]
    public void MetalCompilesBothStages(Gen5TessellationRole role, bool quad)
    {
        if (!MetalNative.IsAvailable)
        {
            Assert.False(
                OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
                "The required gate cannot compile the tessellation stages without a Metal device.");
            output.WriteLine("No Metal device on this host; the tessellation compile test skipped.");
            return;
        }

        var shader = CompileShader(role == Gen5TessellationRole.Hull ? ShaderStage.Compute : ShaderStage.Vertex,
            quad
                ? new Gen5TessellationLaunch(role, 16, 4, 0x8000, 24, Gen5TessellationPartition.FractionalOdd, MaxFactor: 15, Quad: true)
                : new Gen5TessellationLaunch(role, 21, 3, 0x8000, 16, Gen5TessellationPartition.FractionalOdd, OutputClockwise: true, MaxFactor: 15));
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out _, out var error), $"Metal rejected the {role} MSL: {error}\n{shader.Source}");
    }

    private static string Compile(ShaderStage stage, Gen5TessellationLaunch tessellation)
    {
        // Registers the program never indexes are emitted as locals (s2) rather than file entries (s[2]).
        return Regex.Replace(CompileShader(stage, tessellation).Source, @"\b([sv])\[(\d+)\]", "$1$2");
    }

    private static Gen5MslShader CompileShader(ShaderStage stage, Gen5TessellationLaunch tessellation)
    {
        var program = Gen5ComputeFixtures.DecodeOrThrow(new Gen5ComputeFixture("tessellation", EndProgram, 0, 0));
        var request = Gen5ComputeFixtures.RequestOrThrow(program, stage,
            waveLaneCount: 64, localSizeX: 64, tessellation: tessellation);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.Same(tessellation, shader.Tessellation);
        return shader;
    }
}
