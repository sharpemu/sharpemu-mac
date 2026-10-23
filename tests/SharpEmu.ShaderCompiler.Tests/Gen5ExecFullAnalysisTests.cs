// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Ir;
using Xunit;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class Gen5ExecFullAnalysisTests
{
    private const uint Exec = 126;

    private readonly List<Gen5ShaderInstruction> _program = [];
    private uint _pc;

    private uint Add(string opcode, Gen5ShaderEncoding encoding, Gen5Operand[] destinations, Gen5Operand[] sources,
        uint word = 0, Gen5InstructionControl? control = null)
    {
        var pc = _pc;
        _program.Add(new Gen5ShaderInstruction(pc, encoding, opcode, [word], sources, destinations, control));
        _pc += 4;
        return pc;
    }

    private uint VectorAdd() =>
        Add("VAddF32", Gen5ShaderEncoding.Vop2, [Gen5Operand.Vector(1)], [Gen5Operand.Vector(2), Gen5Operand.Vector(3)]);

    private uint Scalar(string opcode, uint destination, params Gen5Operand[] sources) =>
        Add(opcode, Gen5ShaderEncoding.Sop1, [Gen5Operand.Scalar(destination)], sources);

    private uint Kill() =>
        Add("VCmpxLtF32", Gen5ShaderEncoding.Vopc, [], [Gen5Operand.Vector(0), Gen5Operand.Vector(1)]);

    private uint Branch(string opcode) => Add(opcode, Gen5ShaderEncoding.Sopp, [], [], 0xBF840000u);

    private void PointBranch(uint branchPc, uint targetPc)
    {
        var index = _program.FindIndex(instruction => instruction.Pc == branchPc);
        var original = _program[index];
        var offset = (ushort)(short)((int)(targetPc - branchPc - 4) / 4);
        _program[index] = original with { Words = [(original.Words[0] & 0xFFFF0000u) | offset] };
    }

    private IReadOnlySet<uint> Analyze() => Gen5ExecFullAnalysis.Analyze(new Gen5ShaderProgram(0x1000, _program), wave32: false);

    [Fact]
    public void StraightLineCode_RunsWithAFullExec()
    {
        var first = VectorAdd();
        var second = VectorAdd();

        var full = Analyze();
        Assert.Contains(first, full);
        Assert.Contains(second, full);
    }

    [Fact]
    public void AfterAVectorCompareIntoExec_ExecIsUnknown()
    {
        var before = VectorAdd();
        Kill();
        var after = VectorAdd();

        var full = Analyze();
        Assert.Contains(before, full);
        Assert.DoesNotContain(after, full);
    }

    [Theory]
    [InlineData(false, false, 40u)]
    [InlineData(false, false, 106u)]
    [InlineData(false, true, 40u)]
    [InlineData(false, true, 106u)]
    [InlineData(true, false, 40u)]
    [InlineData(true, false, 106u)]
    [InlineData(true, true, 40u)]
    [InlineData(true, true, 106u)]
    public void CompareIntoExecWithLegacyDestination_PreservesSavedMask(bool wave32, bool useVop3, uint savedMask)
    {
        Scalar("SMovB64", savedMask, Gen5Operand.Scalar(Exec));
        var before = VectorAdd();
        Add("VCmpxNltF32", useVop3 ? Gen5ShaderEncoding.Vop3 : Gen5ShaderEncoding.Vopc, [],
            [Gen5Operand.Vector(20), new Gen5Operand(Gen5OperandKind.EncodedConstant, 244)],
            control: CompareControl(useVop3, savedMask));
        var masked = VectorAdd();
        Scalar("SMovB64", Exec, Gen5Operand.Scalar(savedMask));
        var restored = VectorAdd();

        var full = Gen5ExecFullAnalysis.Analyze(new Gen5ShaderProgram(0x1000, _program), wave32);
        Assert.Contains(before, full);
        Assert.DoesNotContain(masked, full);
        Assert.Contains(restored, full);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void CompareIntoScalarDestination_InvalidatesSavedMask(bool wave32, bool useVop3)
    {
        Scalar("SMovB64", 106, Gen5Operand.Scalar(Exec));
        Add("VCmpNltF32", useVop3 ? Gen5ShaderEncoding.Vop3 : Gen5ShaderEncoding.Vopc, [],
            [Gen5Operand.Vector(20), new Gen5Operand(Gen5OperandKind.EncodedConstant, 244)],
            control: CompareControl(useVop3, 106));
        var beforeRestore = VectorAdd();
        Scalar("SMovB64", Exec, Gen5Operand.Scalar(106));
        var afterRestore = VectorAdd();

        var full = Gen5ExecFullAnalysis.Analyze(new Gen5ShaderProgram(0x1000, _program), wave32);
        Assert.Contains(beforeRestore, full);
        Assert.DoesNotContain(afterRestore, full);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ComplementarySavedRegionsReconvergeAtEntryMask(bool wave32)
    {
        Scalar("SMovB64", 20, Gen5Operand.Scalar(Exec));
        Add("VCmpEqU32", Gen5ShaderEncoding.Vopc, [], [Gen5Operand.Vector(0), Gen5Operand.Vector(1)]);
        Scalar("SAndB64", Exec, Gen5Operand.Scalar(20), Gen5Operand.Scalar(106));
        Scalar("SMovB64", 22, Gen5Operand.Scalar(Exec));
        Scalar("SAndn2B64", Exec, Gen5Operand.Scalar(20), Gen5Operand.Scalar(106));
        var masked = VectorAdd();
        Scalar("SOrB64", Exec, Gen5Operand.Scalar(Exec), Gen5Operand.Scalar(22));
        var joined = VectorAdd();
        var program = new Gen5ShaderProgram(0x1000, _program);
        var full = Gen5ExecFullAnalysis.Analyze(program, wave32, reconvergence: true);
        Assert.DoesNotContain(masked, full);
        Assert.Contains(joined, full);
        Assert.DoesNotContain(joined, Gen5ExecFullAnalysis.Analyze(program, wave32));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AJoinRequiresEntryOnBothIncomingPaths(bool restoresOtherPath)
    {
        Scalar("SMovB64", 20, Gen5Operand.Scalar(Exec));
        var branch = Branch("SCbranchScc0");
        Scalar("SAndSaveexecB64", 22, Gen5Operand.Scalar(106));
        Scalar("SMovB64", Exec, Gen5Operand.Scalar(20));
        var skip = Branch("SBranch");
        var other = Scalar("SMovB64", Exec, restoresOtherPath
            ? Gen5Operand.Scalar(20) : new Gen5Operand(Gen5OperandKind.EncodedConstant, 128));
        var joined = VectorAdd();
        PointBranch(branch, other);
        PointBranch(skip, joined);
        Assert.Equal(restoresOtherPath, Gen5ExecFullAnalysis.Analyze(
            new Gen5ShaderProgram(0x1000, _program), false, reconvergence: true).Contains(joined));
    }

    [Fact]
    public void ALoopCarriedMaskCannotAliasANewCompareAtTheSamePc()
    {
        Scalar("SMovB64", 20, Gen5Operand.Scalar(Exec));
        var header = Add("VCmpEqU32", Gen5ShaderEncoding.Vopc, [], [Gen5Operand.Vector(0), Gen5Operand.Vector(1)]);
        Scalar("SAndB64", Exec, Gen5Operand.Scalar(20), Gen5Operand.Scalar(106));
        Scalar("SOrB64", Exec, Gen5Operand.Scalar(Exec), Gen5Operand.Scalar(22));
        var observed = VectorAdd();
        Scalar("SAndn2B64", 22, Gen5Operand.Scalar(20), Gen5Operand.Scalar(106));
        var back = Branch("SCbranchScc0");
        PointBranch(back, header);
        Assert.DoesNotContain(observed, Gen5ExecFullAnalysis.Analyze(
            new Gen5ShaderProgram(0x1000, _program), false, reconvergence: true));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void AClampedHighHalfCannotRestoreLowExec(bool reconvergence, bool wave32)
    {
        Scalar("SMovB64", 20, Gen5Operand.Scalar(Exec));
        Scalar("SMovB32", Exec, Gen5Operand.Scalar(21));
        var masked = VectorAdd();
        Assert.DoesNotContain(masked, Gen5ExecFullAnalysis.Analyze(
            new Gen5ShaderProgram(0x1000, _program), wave32, reconvergence, clampExecHigh: true));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void LiteralDwordAndInlineMinusOneHaveDifferentHighHalves(bool wave32, bool inline)
    {
        Kill();
        Scalar("SMovB64", Exec, inline ? Gen5Operand.Source(193)
            : new Gen5Operand(Gen5OperandKind.LiteralConstant, uint.MaxValue));
        var after = VectorAdd();
        var full = Gen5ExecFullAnalysis.Analyze(new Gen5ShaderProgram(0x1000, _program), wave32, reconvergence: true);
        Assert.Equal(wave32 || inline, full.Contains(after));
    }

    private static Gen5InstructionControl CompareControl(bool useVop3, uint scalarDestination) =>
        useVop3
            ? new Gen5Vop3Control(0, 0, 0, false, 0, scalarDestination)
            : new Gen5SdwaControl(6, 0, 6, 6, false, false, 0, 0, 0, false, scalarDestination);

    [Fact]
    public void AfterAnSdwaVectorCompareIntoExec_ExecIsUnknown()
    {
        var before = VectorAdd();
        var pc = _pc;
        _program.Add(new Gen5ShaderInstruction(
            pc,
            Gen5ShaderEncoding.Vopc,
            "VCmpxNltF32",
            [0u],
            [Gen5Operand.Vector(0), Gen5Operand.Vector(1)],
            [],
            new Gen5SdwaControl(0, 0, 6, 6, false, false, 0, 0, 0, false, 106)));
        _pc += 4;
        var save = Scalar("SMovB64", 22, Gen5Operand.Scalar(Exec));
        VectorAdd();
        Scalar("SMovB64", Exec, Gen5Operand.Scalar(22));
        var after = VectorAdd();

        var full = Analyze();
        Assert.Contains(before, full);
        Assert.DoesNotContain(save, full);
        Assert.DoesNotContain(after, full);
    }

    [Fact]
    public void RestoringASavedFullExec_MakesItFullAgain()
    {
        Scalar("SMovB64", 40, Gen5Operand.Scalar(Exec));
        Kill();
        var masked = VectorAdd();
        Scalar("SMovB64", Exec, Gen5Operand.Scalar(40));
        var restored = VectorAdd();

        var full = Analyze();
        Assert.DoesNotContain(masked, full);
        Assert.Contains(restored, full);
    }

    [Fact]
    public void ACopyOverwrittenBeforeTheRestore_DoesNotCount()
    {
        Scalar("SMovB64", 40, Gen5Operand.Scalar(Exec));
        Kill();
        Scalar("SMovB32", 41, Gen5Operand.Scalar(3));
        Scalar("SMovB64", Exec, Gen5Operand.Scalar(40));
        var after = VectorAdd();

        Assert.DoesNotContain(after, Analyze());
    }

    [Fact]
    public void AJoinWithAPartialPath_IsUnknown()
    {
        var branch = Branch("SCbranchScc0");
        Kill();
        var join = VectorAdd();
        PointBranch(branch, join);

        Assert.DoesNotContain(join, Analyze());
    }

    [Fact]
    public void WqmOfAFullExec_StaysFull()
    {
        Scalar("SWqmB64", Exec, Gen5Operand.Scalar(Exec));
        var after = VectorAdd();

        Assert.Contains(after, Analyze());
    }
}
