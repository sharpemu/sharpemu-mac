// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// What the translator proves for the host's colour-load elision: the clip position of a
/// vertex program that makes it from the vertex index alone, and a pixel program whose
/// valid mask is the restored entry EXEC and so cannot discard.
/// </summary>
public sealed class Gen5MslLoadElisionTests
{
    private const uint Pos0 = 12;
    private const uint Exec = 126;

    // ---- constant positions ----

    [Fact]
    public void PrimitiveShaderQuadPositionsComeFromTheVertexIndex()
    {
        // Demon's Souls' present vertex program: EXEC is all-ones shifted by a count read
        // from s3, and x = 2 * (id & 1) - 1, y = 1 - 2 * (id >> 1).
        var shader = CompileVertex(PresentQuad());

        var positions = Assert.IsType<float[]>(shader.ConstantPositions);
        Assert.Equal(Gen5MslShader.ConstantPositionVertexCount * 4, positions.Length);
        Assert.Equal([-1f, 1f, 1f, 1f], positions[0..4]);
        Assert.Equal([1f, 1f, 1f, 1f], positions[4..8]);
        Assert.Equal([-1f, -1f, 1f, 1f], positions[8..12]);
        Assert.Equal([1f, -1f, 1f, 1f], positions[12..16]);
    }

    [Theory]
    [InlineData(ShaderDepthPlanes.Far)]
    [InlineData(ShaderDepthPlanes.NearAtZero | ShaderDepthPlanes.Far)]
    [InlineData(ShaderDepthPlanes.NearAtNegativeW | ShaderDepthPlanes.Far)]
    public void KnownPassingDepthClipPlanesKeepThePositionProof(ShaderDepthPlanes planes)
    {
        var shader = CompileVertex(PresentQuad(), new ShaderClipSpaceTransform { DepthPlanes = planes });
        Assert.Equal(CompileVertex(PresentQuad()).ConstantPositions, shader.ConstantPositions);
    }

    [Theory]
    [InlineData(ShaderDepthPlanes.Far, 2f)]
    [InlineData(ShaderDepthPlanes.NearAtZero, -0.5f)]
    [InlineData(ShaderDepthPlanes.NearAtNegativeW, -2f)]
    public void AFailingGuestDepthPlaneStillRefusesTheProof(ShaderDepthPlanes planes, float z)
    {
        var instructions = PresentQuad().Instructions.ToList();
        var at = instructions.FindIndex(static i => i.Control is Gen5ExportControl { Target: Pos0 });
        var sources = instructions[at].Sources.ToArray();
        sources[2] = Gen5Operand.Vector(6);
        instructions[at] = instructions[at] with { Sources = sources };
        instructions.Insert(at, Instruction(0, "VMovB32", Gen5ShaderEncoding.Vop1,
            [Literal(Bits(z))], [Gen5Operand.Vector(6)]));
        // The proof must test the guest z before a transform could move it back inside.
        var shader = CompileVertex(Renumber(instructions), new ShaderClipSpaceTransform
        {
            DepthPlanes = planes, DepthRemap = true, DepthScale = 0, DepthOffset = 0.5f,
        });
        Assert.Null(shader.ConstantPositions);
    }

    [Fact]
    public void HostClipSpaceTransformAppliesToConstantPositions()
    {
        var shader = CompileVertex(
            PresentQuad(),
            new ShaderClipSpaceTransform(false, 0, 0, 0, 0, 0, 0) { PositionYDown = true, DepthNegativeOneToOne = true });

        var positions = Assert.IsType<float[]>(shader.ConstantPositions);
        Assert.Equal([-1f, -1f, 1f, 1f], positions[0..4]);
    }

    [Fact]
    public void PositionReadFromMemoryIsNotConstant()
    {
        // The same program with x taken from a scalar load of user data.
        var instructions = PresentQuad().Instructions.ToList();
        var export = instructions.FindIndex(static instruction => instruction.Control is Gen5ExportControl { Target: Pos0 });
        instructions.Insert(export, Instruction(
            instructions[export].Pc, "VMovB32", Gen5ShaderEncoding.Vop1, [Gen5Operand.Scalar(20)], [Gen5Operand.Vector(3)]));
        var shader = CompileVertex(Renumber(instructions));

        Assert.Null(shader.ConstantPositions);
    }

    [Fact]
    public void BranchingVertexProgramIsNotEvaluated()
    {
        var instructions = PresentQuad().Instructions.ToList();
        instructions.Insert(0, new Gen5ShaderInstruction(
            0, Gen5ShaderEncoding.Sopp, "SCbranchScc0", [0xBF850000u], [], [], null));
        var shader = CompileVertex(Renumber(instructions));

        Assert.Null(shader.ConstantPositions);
    }

    [Fact]
    public void PixelProgramHasNoConstantPositions()
    {
        var shader = CompilePixel(RestoredEntryExecPixel(kill: false));

        Assert.Null(shader.ConstantPositions);
    }

    // ---- valid mask ----

    [Fact]
    public void RestoredEntryExecValidMaskCannotDiscard()
    {
        var shader = CompilePixel(RestoredEntryExecPixel(kill: false));

        Assert.True(shader.CannotDiscard);
        Assert.Contains("pixel_valid_mask_active = true;", shader.Source, StringComparison.Ordinal);
        Assert.DoesNotContain("pixel_valid_mask_active = exec;", shader.Source, StringComparison.Ordinal);
        Assert.Equal(1u, shader.ColorOutputMask);
    }

    [Fact]
    public void KillThroughTheSavedMaskKeepsTheDiscard()
    {
        // S_ANDN2_B64 on the saved copy is a real kill: the restored EXEC is no longer the
        // entry mask.
        var shader = CompilePixel(RestoredEntryExecPixel(kill: true));

        Assert.False(shader.CannotDiscard);
        Assert.Contains("pixel_valid_mask_active = exec;", shader.Source, StringComparison.Ordinal);
    }

    [Fact]
    public void ValidMaskWithoutRestoreKeepsTheDiscard()
    {
        // V_CMPX narrows EXEC after the restore: the export publishes the narrowed mask.
        var instructions = RestoredEntryExecPixel(kill: false).Instructions.ToList();
        var export = instructions.FindIndex(static instruction => instruction.Control is Gen5ExportControl);
        instructions.Insert(export, Instruction(
            instructions[export].Pc, "VCmpxGtF32", Gen5ShaderEncoding.Vopc,
            [Gen5Operand.Vector(0), Gen5Operand.Vector(1)], []));
        var shader = CompilePixel(Renumber(instructions));

        Assert.False(shader.CannotDiscard);
        Assert.Contains("pixel_valid_mask_active = exec;", shader.Source, StringComparison.Ordinal);
    }

    // ---- programs ----

    private static Gen5ShaderProgram PresentQuad()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        void Add(string opcode, Gen5ShaderEncoding encoding, Gen5Operand[] sources, Gen5Operand[] destinations,
            Gen5InstructionControl? control = null) =>
            instructions.Add(Instruction((uint)instructions.Count * 4, opcode, encoding, sources, destinations, control));

        var vop3 = new Gen5Vop3Control(0, 0, 0, false, 0, null);
        Add("SBfeU32", Gen5ShaderEncoding.Sop2, [Gen5Operand.Scalar(3), Literal(0x00080008)], [Gen5Operand.Scalar(107)]);
        Add("SLshlB32", Gen5ShaderEncoding.Sop2, [Gen5Operand.Scalar(107), Inline(12)], [Gen5Operand.Scalar(106)]);
        Add("SAndB32", Gen5ShaderEncoding.Sop2, [Gen5Operand.Scalar(3), Literal(0xFF)], [Gen5Operand.Scalar(0)]);
        Add("SOrB32", Gen5ShaderEncoding.Sop2, [Gen5Operand.Scalar(0), Gen5Operand.Scalar(106)], [Gen5Operand.Scalar(124)]);
        Add("SSubI32", Gen5ShaderEncoding.Sop2, [Inline(64), Gen5Operand.Scalar(107)], [Gen5Operand.Scalar(106)]);
        Add("SLshrB64", Gen5ShaderEncoding.Sop2, [NegativeInline(1), Gen5Operand.Scalar(106)],
            [Gen5Operand.Scalar(Exec), Gen5Operand.Scalar(Exec + 1)]);
        Add("SSubI32", Gen5ShaderEncoding.Sop2, [Inline(64), Gen5Operand.Scalar(0)], [Gen5Operand.Scalar(106)]);
        Add("SLshrB64", Gen5ShaderEncoding.Sop2, [NegativeInline(1), Gen5Operand.Scalar(106)],
            [Gen5Operand.Scalar(Exec), Gen5Operand.Scalar(Exec + 1)]);
        Add("VLshrrevB32", Gen5ShaderEncoding.Vop2, [Inline(1), Gen5Operand.Vector(5)], [Gen5Operand.Vector(4)]);
        Add("VAndB32", Gen5ShaderEncoding.Vop2, [Inline(1), Gen5Operand.Vector(5)], [Gen5Operand.Vector(2)]);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [new Gen5Operand(Gen5OperandKind.EncodedConstant, 242)], [Gen5Operand.Vector(0)]);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Inline(0)], [Gen5Operand.Vector(1)]);
        Add("VMulLoU32", Gen5ShaderEncoding.Vop3, [NegativeInline(2), Gen5Operand.Vector(4), Gen5Operand.Scalar(0)],
            [Gen5Operand.Vector(5)], vop3);
        Add("VLshlAddU32", Gen5ShaderEncoding.Vop3, [Gen5Operand.Vector(2), Inline(1), NegativeInline(1)],
            [Gen5Operand.Vector(3)], vop3);
        Add("VCvtF32I32", Gen5ShaderEncoding.Vop1, [Gen5Operand.Vector(2)], [Gen5Operand.Vector(2)]);
        Add("VCvtF32I32", Gen5ShaderEncoding.Vop1, [Gen5Operand.Vector(4)], [Gen5Operand.Vector(4)]);
        Add("VAddI32", Gen5ShaderEncoding.Vop2, [Inline(1), Gen5Operand.Vector(5)], [Gen5Operand.Vector(5)]);
        Add("VCvtF32I32", Gen5ShaderEncoding.Vop1, [Gen5Operand.Vector(3)], [Gen5Operand.Vector(3)]);
        Add("VCvtF32I32", Gen5ShaderEncoding.Vop1, [Gen5Operand.Vector(5)], [Gen5Operand.Vector(5)]);
        Add("Exp", Gen5ShaderEncoding.Exp,
            [Gen5Operand.Vector(3), Gen5Operand.Vector(5), Gen5Operand.Vector(0), Gen5Operand.Vector(0)], [],
            new Gen5ExportControl(Pos0, 0xF, false, true, false));
        Add("Exp", Gen5ShaderEncoding.Exp,
            [Gen5Operand.Vector(2), Gen5Operand.Vector(4), Gen5Operand.Vector(1), Gen5Operand.Vector(1)], [],
            new Gen5ExportControl(32, 0xF, false, false, false));
        Add("SEndpgm", Gen5ShaderEncoding.Sopp, [], []);
        return new Gen5ShaderProgram(0, instructions);
    }

    /// <summary>Save EXEC, widen it for whole-quad mode, put the copy back and export MRT0
    /// with VM — the shape of the present pixel program. With <paramref name="kill"/> the
    /// saved copy loses the lanes VCC names before it is restored.</summary>
    private static Gen5ShaderProgram RestoredEntryExecPixel(bool kill)
    {
        var instructions = new List<Gen5ShaderInstruction>();
        void Add(string opcode, Gen5ShaderEncoding encoding, Gen5Operand[] sources, Gen5Operand[] destinations,
            Gen5InstructionControl? control = null) =>
            instructions.Add(Instruction((uint)instructions.Count * 4, opcode, encoding, sources, destinations, control));

        Add("SMovB64", Gen5ShaderEncoding.Sop1, [Gen5Operand.Scalar(Exec)], [Gen5Operand.Scalar(34), Gen5Operand.Scalar(35)]);
        Add("SWqmB64", Gen5ShaderEncoding.Sop1, [Gen5Operand.Scalar(Exec)], [Gen5Operand.Scalar(Exec), Gen5Operand.Scalar(Exec + 1)]);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Literal(Bits(0.25f))], [Gen5Operand.Vector(0)]);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Literal(Bits(0.5f))], [Gen5Operand.Vector(1)]);
        if (kill)
        {
            Add("VCmpGtF32", Gen5ShaderEncoding.Vopc, [Gen5Operand.Vector(0), Gen5Operand.Vector(1)],
                [Gen5Operand.Scalar(106), Gen5Operand.Scalar(107)]);
            Add("SAndn2B64", Gen5ShaderEncoding.Sop2, [Gen5Operand.Scalar(34), Gen5Operand.Scalar(106)],
                [Gen5Operand.Scalar(34), Gen5Operand.Scalar(35)]);
        }

        Add("SMovB64", Gen5ShaderEncoding.Sop1, [Gen5Operand.Scalar(34)], [Gen5Operand.Scalar(Exec), Gen5Operand.Scalar(Exec + 1)]);
        Add("Exp", Gen5ShaderEncoding.Exp,
            [Gen5Operand.Vector(0), Gen5Operand.Vector(1), Gen5Operand.Vector(0), Gen5Operand.Vector(1)], [],
            new Gen5ExportControl(0, 0xF, false, true, true));
        Add("SEndpgm", Gen5ShaderEncoding.Sopp, [], []);
        return new Gen5ShaderProgram(0, instructions);
    }

    private static Gen5MslShader CompileVertex(Gen5ShaderProgram program, ShaderClipSpaceTransform clip = default)
    {
        var request = Gen5ComputeFixtures.RequestOrThrow(program, ShaderStage.Vertex, clipSpace: clip);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.Equal(Gen5MslStage.Vertex, shader.Stage);
        return shader;
    }

    private static Gen5MslShader CompilePixel(Gen5ShaderProgram program)
    {
        var request = Gen5ComputeFixtures.RequestOrThrow(program, ShaderStage.Pixel);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.Equal(Gen5MslStage.Pixel, shader.Stage);
        return shader;
    }

    private static Gen5ShaderProgram Renumber(List<Gen5ShaderInstruction> instructions) =>
        new(0, instructions.Select(static (instruction, index) => instruction with { Pc = (uint)index * 4 }).ToList());

    private static Gen5ShaderInstruction Instruction(
        uint pc, string opcode, Gen5ShaderEncoding encoding, Gen5Operand[] sources, Gen5Operand[] destinations,
        Gen5InstructionControl? control = null) =>
        new(pc, encoding, opcode, [0u], sources, destinations, control);

    private static uint Bits(float value) => BitConverter.SingleToUInt32Bits(value);

    private static Gen5Operand Literal(uint value) => new(Gen5OperandKind.LiteralConstant, value);

    private static Gen5Operand Inline(uint value) => new(Gen5OperandKind.EncodedConstant, 128 + value);

    private static Gen5Operand NegativeInline(uint value) => new(Gen5OperandKind.EncodedConstant, 192 + value);
}
