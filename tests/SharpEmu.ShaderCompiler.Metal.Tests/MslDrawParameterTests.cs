// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// Draw parameters in a translated vertex program, as Gen5SpirvTranslator's vertex entry has
/// them (upstream 8ddf6601e, 99f9fe930): with SupportsDrawParameters the instance index counts
/// from the draw's base instance, and a user-data register an indirect packet names for the
/// base vertex (BaseVertexUserData) or the start instance (StartInstanceUserData) receives
/// that base, the vertex index then counting from the base vertex. The command processor
/// leaves such a register unwritten (GpuCommandInterpreter.DrawIndirectArguments), so without
/// the store the program would read whatever the register last held.
/// </summary>
public sealed class MslDrawParameterTests(ITestOutputHelper output)
{
    private const uint TargetSize = 8;
    private const uint Pos0 = 12;
    private const uint Param0 = 32;
    private const int BaseVertexUserData = 1;
    private const int StartInstanceUserData = 2;
    private const int BaseVertex = 100;
    private const uint BaseInstance = 7;

    // ---- structure ----

    [Fact]
    public void TheVertexEntryStoresBothBasesInTheNamedUserData()
    {
        var shader = Compile(ProbeProgram(), drawParameters: true, BaseVertexUserData, StartInstanceUserData);
        Assert.Contains("uint sharpemu_base_vertex [[base_vertex]]", shader.Source);
        Assert.Contains("uint sharpemu_base_instance [[base_instance]]", shader.Source);
        Assert.Contains("v5 = sharpemu_vertex_id - sharpemu_base_vertex;", shader.Source);
        Assert.Contains("v8 = sharpemu_instance_id - sharpemu_base_instance;", shader.Source);
        Assert.Contains($"s{BaseVertexUserData} = sharpemu_base_vertex;", shader.Source);
        Assert.Contains($"s{StartInstanceUserData} = sharpemu_base_instance;", shader.Source);
        // The stores land after the user data is loaded, or the load would undo them.
        var load = shader.Source.IndexOf($"    s{BaseVertexUserData} = ", StringComparison.Ordinal);
        var store = shader.Source.IndexOf($"    s{BaseVertexUserData} = sharpemu_base_vertex;", StringComparison.Ordinal);
        Assert.True(load >= 0 && store > load, "the user data load of s1 does not come first");
    }

    [Fact]
    public void WithoutAUserDataRegisterOnlyTheInstanceCountsFromItsBase()
    {
        var shader = Compile(ProbeProgram(), drawParameters: true, -1, -1);
        Assert.DoesNotContain("[[base_vertex]]", shader.Source);
        Assert.Contains("v5 = sharpemu_vertex_id;", shader.Source);
        Assert.Contains("v8 = sharpemu_instance_id - sharpemu_base_instance;", shader.Source);
        Assert.DoesNotContain("= sharpemu_base_instance;", shader.Source);
    }

    [Fact]
    public void WithoutDrawParametersTheIdsAreMetals()
    {
        var shader = Compile(ProbeProgram(), drawParameters: false, BaseVertexUserData, StartInstanceUserData);
        Assert.DoesNotContain("[[base_vertex]]", shader.Source);
        Assert.DoesNotContain("[[base_instance]]", shader.Source);
        Assert.Contains("v5 = sharpemu_vertex_id;", shader.Source);
        Assert.Contains("v8 = sharpemu_instance_id;", shader.Source);
    }

    [Fact]
    public void ARectangleListTakesItsBaseVertexFromTheDrawBlock()
    {
        var shader = Compile(ProbeProgram(), drawParameters: true, BaseVertexUserData, StartInstanceUserData);
        var source = Gen5MslRectangleList.BuildSource(shader);
        Assert.Contains("sharpemu_rect_base_vertex(sharpemu_rect_draw)", source);
        Assert.Contains("uint base_vertex;", source);
        Assert.DoesNotContain("[[base_vertex]]", source);
        Assert.Contains("[[base_instance]]", source);
        if (MetalNative.IsAvailable)
        {
            Assert.True(MetalNative.TryCompileLibrary(source, out _, out var error), $"Metal rejected the rectangle form: {error}\n{source}");
        }
    }

    [Fact]
    public void AMeshStageStoresTheDrawBlocksBases()
    {
        // A primitive shader the host runs as a mesh pipeline is a plain vertex program to
        // the command processor, so it gets the same user-data registers.
        var userDataBase = Gen5NggFixtures.VertexUserDataBase;
        var request = Gen5ComputeFixtures.RequestOrThrow(Gen5NggFixtures.CullingTriangle(), ShaderStage.Vertex,
            requiredVertexOutputCount: 1, primitiveShader: Gen5NggFixtures.StageInfo, userDataBase: userDataBase,
            supportsDrawParameters: true, baseVertexUserData: 4, startInstanceUserData: 5);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.Equal(Gen5MslStage.Mesh, shader.Stage);
        Assert.Contains($"s{userDataBase + 4} = (uint)sharpemu_draw.base_vertex;", shader.Source);
        Assert.Contains($"s{userDataBase + 5} = sharpemu_draw.first_instance;", shader.Source);
        Assert.Contains("const uint sharpemu_mesh_instance_index = sharpemu_mesh_instance;", shader.Source);
        Assert.DoesNotContain("sharpemu_mesh_vertex_slot) + sharpemu_draw.base_vertex", shader.Source);

        var plain = Gen5NggFixtures.CompileOrThrow(Gen5NggFixtures.CullingTriangle());
        Assert.Contains("sharpemu_mesh_vertex_slot) + sharpemu_draw.base_vertex", plain.Source);
        Assert.Contains("sharpemu_draw.first_instance + sharpemu_mesh_instance;", plain.Source);
        if (MetalNative.IsAvailable)
        {
            Assert.True(MetalNative.TryCompileLibrary(shader.Source, out _, out var compileError),
                $"Metal rejected the mesh stage: {compileError}");
        }
    }

    // ---- on the device ----

    public static TheoryData<string> Draws => new() { "auto", "indexed", "auto-indirect", "indexed-indirect" };

    [Theory]
    [MemberData(nameof(Draws))]
    public void TheProgramSeesTheDrawsBasesInItsUserData(string draw)
    {
        if (SkipWithoutMetalDevice($"the {draw} draw")) return;

        var pixel = Render(drawParameters: true, draw);
        output.WriteLine($"{draw}: s1={pixel[0]} s2={pixel[1]} v8={pixel[2]} w={pixel[3]}");
        // The triangle is only on screen when the vertex index counts from the base vertex,
        // and its colour is (s1, s2, v8): the base vertex, the start instance and instance 0.
        Assert.Equal([BaseVertex, BaseInstance, 0f, 1f], pixel);
    }

    [Fact]
    public void WithoutDrawParametersTheUserDataKeepsItsValue()
    {
        if (SkipWithoutMetalDevice("the draw without draw parameters")) return;

        // Metal's vertex id includes the base, which puts the triangle off screen.
        var pixel = Render(drawParameters: false, "indexed");
        Assert.Equal([0f, 0f, 0f, 0f], pixel);
    }

    private float[] Render(bool drawParameters, string draw)
    {
        var shader = Compile(ProbeProgram(), drawParameters, BaseVertexUserData, StartInstanceUserData);
        var source = shader.Source + FragmentSource;
        Assert.True(MetalNative.TryCompileLibrary(source, out var library, out var compileError), $"{compileError}\n{source}");
        ushort[] indices = [0, 1, 2];
        var encode = draw switch
        {
            "auto" => MetalNative.DrawAuto(BaseVertex, 3, BaseInstance),
            "indexed" => MetalNative.DrawIndexed(indices, BaseVertex, BaseInstance),
            "auto-indirect" => MetalNative.DrawAutoIndirect(BaseVertex, 3, BaseInstance),
            "indexed-indirect" => MetalNative.DrawIndexedIndirect(indices, BaseVertex, BaseInstance),
            _ => throw new ArgumentOutOfRangeException(nameof(draw)),
        };
        Assert.True(
            MetalNative.TryRenderVertices(library, shader.EntryPoint, "gen5_draw_parameter_fs", shader.ArgumentLayout?.ByteSize ?? 0,
                vertexCount: 3, TargetSize, TargetSize, layers: 1, viewports: null, out var pixels, out var renderError, encode),
            renderError);
        var center = ((TargetSize / 2) * TargetSize + (TargetSize / 2)) * 4;
        return pixels[(int)center..(int)(center + 4)];
    }

    private static Gen5MslShader Compile(Gen5ShaderProgram program, bool drawParameters, int baseVertexUserData, int startInstanceUserData)
    {
        var request = Gen5ComputeFixtures.RequestOrThrow(program, ShaderStage.Vertex, requiredVertexOutputCount: 1,
            supportsDrawParameters: drawParameters, baseVertexUserData: baseVertexUserData, startInstanceUserData: startInstanceUserData);
        if (!Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error))
        {
            throw new InvalidOperationException($"vertex emit failed: {error}");
        }

        return shader;
    }

    // A triangle over the whole target from vertex indices 0, 1 and 2 ((index & 1, index >> 1)
    // * 4 - 1), which indices from 100 put far above it; param 0 is (s1, s2, v8, 1).
    private static Gen5ShaderProgram ProbeProgram()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        var pc = 0u;
        void Add(string opcode, Gen5ShaderEncoding encoding, Gen5Operand[] sources,
            Gen5Operand[] destinations, Gen5InstructionControl? control = null)
        {
            instructions.Add(new Gen5ShaderInstruction(pc, encoding, opcode, [0u], sources, destinations, control));
            pc += sizeof(uint);
        }

        Add("VAndB32", Gen5ShaderEncoding.Vop2, [Inline(1), Vector(5)], [Vector(10)]);
        Add("VCvtF32U32", Gen5ShaderEncoding.Vop1, [Vector(10)], [Vector(11)]);
        Add("VMulF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(4.0f)), Vector(11)], [Vector(11)]);
        Add("VAddF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(-1.0f)), Vector(11)], [Vector(11)]);
        Add("VLshrrevB32", Gen5ShaderEncoding.Vop2, [Inline(1), Vector(5)], [Vector(12)]);
        Add("VCvtF32U32", Gen5ShaderEncoding.Vop1, [Vector(12)], [Vector(13)]);
        Add("VMulF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(4.0f)), Vector(13)], [Vector(13)]);
        Add("VAddF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(-1.0f)), Vector(13)], [Vector(13)]);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Inline(0)], [Vector(20)]);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Literal(Bits(1.0f))], [Vector(21)]);
        Add("VCvtF32U32", Gen5ShaderEncoding.Vop1, [Scalar(BaseVertexUserData)], [Vector(30)]);
        Add("VCvtF32U32", Gen5ShaderEncoding.Vop1, [Scalar(StartInstanceUserData)], [Vector(31)]);
        Add("VCvtF32U32", Gen5ShaderEncoding.Vop1, [Vector(8)], [Vector(32)]);
        Add("Exp", Gen5ShaderEncoding.Exp, [Vector(11), Vector(13), Vector(20), Vector(21)], [],
            new Gen5ExportControl(Pos0, 0xF, false, false, false));
        Add("Exp", Gen5ShaderEncoding.Exp, [Vector(30), Vector(31), Vector(32), Vector(21)], [],
            new Gen5ExportControl(Param0, 0xF, false, true, false));
        Add("SEndpgm", Gen5ShaderEncoding.Sopp, [], []);
        return new Gen5ShaderProgram(0, instructions);
    }

    private const string FragmentSource = """

        struct Gen5DrawParameterIn
        {
            float4 position [[position]];
            float4 param0 [[user(locn0)]];
        };

        fragment float4 gen5_draw_parameter_fs(Gen5DrawParameterIn in [[stage_in]])
        {
            return in.param0;
        }
        """;

    private bool SkipWithoutMetalDevice(string activity)
    {
        if (MetalNative.IsAvailable) return false;
        Assert.False(
            OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
            $"The required gate cannot run {activity} without a Metal device.");
        output.WriteLine($"No Metal device on this host; {activity} skipped.");
        return true;
    }

    private static uint Bits(float value) => BitConverter.SingleToUInt32Bits(value);

    private static Gen5Operand Literal(uint value) => new(Gen5OperandKind.LiteralConstant, value);

    private static Gen5Operand Inline(uint value) => new(Gen5OperandKind.EncodedConstant, 128 + value);

    private static Gen5Operand Vector(uint index) => Gen5Operand.Vector(index);

    private static Gen5Operand Scalar(int index) => new(Gen5OperandKind.ScalarRegister, (uint)index);
}
