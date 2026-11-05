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

    [Theory]
    [InlineData(false, false, false, false)]
    [InlineData(false, false, true, false)]
    [InlineData(false, true, false, true)]
    [InlineData(false, true, true, false)]
    [InlineData(true, false, false, true)]
    [InlineData(true, false, true, true)]
    [InlineData(true, true, false, true)]
    [InlineData(true, true, true, true)]
    public void ConstantExecRestore_RespectsWidthAndLiteralExtension(bool wave32, bool wide, bool literal, bool expected)
    {
        Kill();
        var mask = literal ? new Gen5Operand(Gen5OperandKind.LiteralConstant, uint.MaxValue)
            : new Gen5Operand(Gen5OperandKind.EncodedConstant, 193);
        Scalar(wide ? "SMovB64" : "SMovB32", Exec, mask);
        var after = VectorAdd();
        var full = Gen5ExecFullAnalysis.Analyze(new Gen5ShaderProgram(0x1000, _program), wave32);
        Assert.Equal(expected, full.Contains(after));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OrWithInlineMinusOne_RestoresExec(bool constantFirst)
    {
        Kill();
        var mask = new Gen5Operand(Gen5OperandKind.EncodedConstant, 193);
        Scalar("SOrB64", Exec, constantFirst ? mask : Gen5Operand.Scalar(Exec),
            constantFirst ? Gen5Operand.Scalar(Exec) : mask);
        Assert.Contains(VectorAdd(), Analyze());
    }

    [Theory]
    [InlineData(Gen5OperandKind.EncodedConstant, 192u)]
    [InlineData(Gen5OperandKind.EncodedConstant, 194u)]
    [InlineData(Gen5OperandKind.LiteralConstant, 0x7fffffffu)]
    [InlineData(Gen5OperandKind.LiteralConstant, 0xfffffffeu)]
    public void PartialConstantExecRestore_RemainsGuarded(Gen5OperandKind kind, uint value)
    {
        Kill();
        Scalar("SMovB64", Exec, new Gen5Operand(kind, value));
        Assert.DoesNotContain(VectorAdd(), Analyze());
    }

    [Fact]
    public void ConstantRestoreOnOnlyOneBranch_DoesNotProveJoinFull()
    {
        Kill();
        var branch = Branch("SCbranchScc0");
        Scalar("SMovB64", Exec, new Gen5Operand(Gen5OperandKind.EncodedConstant, 193));
        var join = VectorAdd();
        PointBranch(branch, join);
        Assert.DoesNotContain(join, Analyze());
    }

    [Fact]
    public void LiteralSavedHighWord_DoesNotProveWave32ExecFull()
    {
        Scalar("SMovB64", 40, new Gen5Operand(Gen5OperandKind.LiteralConstant, uint.MaxValue));
        Kill();
        Scalar("SMovB32", Exec, Gen5Operand.Scalar(41));
        var after = VectorAdd();
        Assert.DoesNotContain(after, Gen5ExecFullAnalysis.Analyze(new Gen5ShaderProgram(0x1000, _program), true));
    }

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

    private static Gen5Operand Lane(uint lane) => new(Gen5OperandKind.EncodedConstant, 128 + lane);

    private static readonly Gen5Vop3Control NoModifiers = new(0, 0, 0, false, 0, null);

    private uint WriteLane(uint vector, Gen5Operand value, Gen5Operand lane) =>
        Add("VWritelaneB32", Gen5ShaderEncoding.Vop3, [Gen5Operand.Vector(vector)], [value, lane], control: NoModifiers);

    private uint ReadLane(uint scalar, uint vector, Gen5Operand lane) =>
        Add("VReadlaneB32", Gen5ShaderEncoding.Vop3, [Gen5Operand.Scalar(scalar)], [Gen5Operand.Vector(vector), lane], control: NoModifiers);

    private IReadOnlySet<uint> AnalyzeLaneSpills(bool track) =>
        Gen5ExecFullAnalysis.Analyze(new Gen5ShaderProgram(0x1000, _program), wave32: false, trackLaneSpills: track);

    // The shape a lighting pixel shader takes under SGPR pressure: the if's saved EXEC is parked
    // in two lanes of v27, s[0:1] are reused, and the pair is read back just before the restore.
    private (uint Masked, uint Restored) SpilledSaveexecIf(Action? betweenParkAndReload = null)
    {
        Scalar("SAndSaveexecB64", 0, Gen5Operand.Scalar(106));
        WriteLane(27, Gen5Operand.Scalar(0), Lane(34));
        WriteLane(27, Gen5Operand.Scalar(1), Lane(35));
        Scalar("SMovB32", 1, Gen5Operand.Scalar(33));
        var masked = VectorAdd();
        betweenParkAndReload?.Invoke();
        ReadLane(0, 27, Lane(34));
        ReadLane(1, 27, Lane(35));
        Scalar("SMovB64", Exec, Gen5Operand.Scalar(0));
        return (masked, VectorAdd());
    }

    [Fact]
    public void ASavedFullExecReloadedFromVectorLanes_RestoresAFullExec()
    {
        var (masked, restored) = SpilledSaveexecIf();

        var tracked = AnalyzeLaneSpills(track: true);
        Assert.DoesNotContain(masked, tracked);
        Assert.Contains(restored, tracked);
        Assert.DoesNotContain(restored, AnalyzeLaneSpills(track: false));
        Assert.DoesNotContain(restored, Analyze());
    }

    [Fact]
    public void AVectorWriteToTheSpillRegister_DropsTheParkedCopy()
    {
        var (_, restored) = SpilledSaveexecIf(() =>
            Add("VMovB32", Gen5ShaderEncoding.Vop1, [Gen5Operand.Vector(27)], [Gen5Operand.Vector(3)]));

        Assert.DoesNotContain(restored, AnalyzeLaneSpills(track: true));
    }

    [Fact]
    public void AWideVectorWriteBelowTheSpillRegister_DropsTheParkedCopy()
    {
        // A four-dword load into v[24:27] reaches v27.
        var (_, restored) = SpilledSaveexecIf(() =>
            Add("BufferLoadDwordx4", Gen5ShaderEncoding.Mubuf, [Gen5Operand.Vector(24)], [Gen5Operand.Vector(4)],
                control: new Gen5BufferMemoryControl(4, 4, 24, 8, 0, IndexEnabled: false, OffsetEnabled: true, Glc: false, Slc: false)));

        Assert.DoesNotContain(restored, AnalyzeLaneSpills(track: true));
    }

    [Fact]
    public void ParkingAnotherValueInTheLane_DropsTheParkedCopy()
    {
        var (_, restored) = SpilledSaveexecIf(() => WriteLane(27, Gen5Operand.Scalar(5), Lane(35)));

        Assert.DoesNotContain(restored, AnalyzeLaneSpills(track: true));
    }

    [Fact]
    public void ADynamicLaneWrite_DropsEveryCopyInTheRegister()
    {
        var (_, restored) = SpilledSaveexecIf(() => WriteLane(27, Gen5Operand.Scalar(5), Gen5Operand.Scalar(7)));

        Assert.DoesNotContain(restored, AnalyzeLaneSpills(track: true));
    }

    [Fact]
    public void AReloadFromALaneOnlyOnePathParked_IsUnknown()
    {
        Scalar("SAndSaveexecB64", 0, Gen5Operand.Scalar(106));
        var skip = Branch("SCbranchScc0");
        WriteLane(27, Gen5Operand.Scalar(0), Lane(34));
        WriteLane(27, Gen5Operand.Scalar(1), Lane(35));
        var join = ReadLane(0, 27, Lane(34));
        PointBranch(skip, join);
        ReadLane(1, 27, Lane(35));
        Scalar("SMovB64", Exec, Gen5Operand.Scalar(0));
        var after = VectorAdd();

        Assert.DoesNotContain(after, AnalyzeLaneSpills(track: true));
    }

    [Fact]
    public void ParkingAPartialExec_DoesNotRestoreAFullOne()
    {
        Kill();
        WriteLane(27, Gen5Operand.Scalar(Exec), Lane(34));
        WriteLane(27, Gen5Operand.Scalar(Exec + 1), Lane(35));
        ReadLane(0, 27, Lane(34));
        ReadLane(1, 27, Lane(35));
        Scalar("SMovB64", Exec, Gen5Operand.Scalar(0));
        var after = VectorAdd();

        Assert.DoesNotContain(after, AnalyzeLaneSpills(track: true));
    }

    [Fact]
    public void ParkingTheFullExecItself_RestoresAFullOne()
    {
        WriteLane(27, Gen5Operand.Scalar(Exec), Lane(34));
        WriteLane(27, Gen5Operand.Scalar(Exec + 1), Lane(35));
        Kill();
        ReadLane(0, 27, Lane(34));
        ReadLane(1, 27, Lane(35));
        Scalar("SMovB64", Exec, Gen5Operand.Scalar(0));
        var after = VectorAdd();

        Assert.Contains(after, AnalyzeLaneSpills(track: true));
    }

    [Fact]
    public void WqmOfAFullExec_StaysFull()
    {
        Scalar("SWqmB64", Exec, Gen5Operand.Scalar(Exec));
        var after = VectorAdd();

        Assert.Contains(after, Analyze());
    }

    [Theory]
    [InlineData("SOrSaveexecB64", false)]
    [InlineData("SOrn2SaveexecB64", true)]
    public void AWholeWaveSection_RunsWithAFullExec_UntilTheSavedMaskComesBack(string entry, bool sourceIsExec)
    {
        Kill();
        Scalar(entry, 20, sourceIsExec ? Gen5Operand.Scalar(Exec) : new Gen5Operand(Gen5OperandKind.EncodedConstant, 193));
        var inside = VectorAdd();
        Scalar("SMovB64", Exec, Gen5Operand.Scalar(20));
        var after = VectorAdd();
        Add("SEndpgm", Gen5ShaderEncoding.Sopp, [], [], 0xBF810000u);

        var program = new Gen5ShaderProgram(0x1000, _program);
        var tracked = Gen5ExecFullAnalysis.Analyze(program, wave32: false, trackLaneSpills: true, trackWholeWave: true);
        Assert.Contains(inside, tracked);
        Assert.DoesNotContain(after, tracked);
        Assert.DoesNotContain(inside, Gen5ExecFullAnalysis.Analyze(program, wave32: false, trackLaneSpills: true));
    }

    [Fact]
    public void AnOrn2SaveexecOfAnotherMask_IsNotAWholeWaveSection()
    {
        Kill();
        Scalar("SOrn2SaveexecB64", 20, Gen5Operand.Scalar(30));
        var inside = VectorAdd();
        Add("SEndpgm", Gen5ShaderEncoding.Sopp, [], [], 0xBF810000u);

        Assert.DoesNotContain(inside, Gen5ExecFullAnalysis.Analyze(new Gen5ShaderProgram(0x1000, _program), wave32: false,
            trackLaneSpills: true, trackWholeWave: true));
    }
}
