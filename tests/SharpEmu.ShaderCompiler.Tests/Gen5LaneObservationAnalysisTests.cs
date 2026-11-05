// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Ir;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class Gen5LaneObservationAnalysisTests
{
    private const uint Exec = 126;

    private readonly List<Gen5ShaderInstruction> _instructions = [];
    private uint _pc;

    private uint Add(Gen5ShaderInstruction instruction)
    {
        var pc = _pc;
        _instructions.Add(instruction with { Pc = pc });
        _pc += 8;
        return pc;
    }

    private Gen5VectorObservationPlan Analyze()
    {
        Add(EndProgram(0));
        var plan = Gen5LaneObservationAnalysis.Analyze(Program([.. _instructions]), wave32: false, trackLaneSpills: true);
        Assert.NotNull(plan);
        return plan;
    }

    // The bounds check of a compute kernel: EXEC narrows once and is never restored.
    private void NarrowForGood() => Add(Vopc(0, "VCmpxLeU32", Operand(40), 1));

    private void SaveExec(uint register) => Add(Sop1(0, "SMovB64", register, Gen5Operand.Scalar(Exec)));

    private void NarrowInner() => Add(Vopc(0, "VCmpxLeU32", Operand(20), 2));

    private void RestoreExec(uint register) => Add(Sop1(0, "SMovB64", Exec, Gen5Operand.Scalar(register)));

    [Fact]
    public void AnInnerTemporaryAfterATopLevelNarrowing_IsNotObserved()
    {
        NarrowForGood();
        SaveExec(20);
        NarrowInner();
        var temporary = Add(Vop2(0, "VAddU32", 3, Operand(1), Gen5Operand.Vector(2)));
        RestoreExec(20);
        Add(MoveVector(0, 3, 9));
        Add(Vop2(0, "VAddU32", 4, Operand(1), Gen5Operand.Vector(3)));

        Assert.True(Analyze().UnobservedWrites(temporary).Contains(3));
        // The older analysis only kills a register under a provably full EXEC, which never
        // comes back after the bounds check.
        Assert.False(Gen5VectorObservationAnalysis.Analyze(Program([.. _instructions]), wave32: false)!
            .UnobservedWrites(temporary).Contains(3));
    }

    [Fact]
    public void AValueTheInnerIfLeavesForItsOtherLanes_IsObserved()
    {
        NarrowForGood();
        SaveExec(20);
        NarrowInner();
        var write = Add(Vop2(0, "VAddU32", 3, Operand(1), Gen5Operand.Vector(2)));
        RestoreExec(20);
        Add(Vop2(0, "VAddU32", 4, Operand(1), Gen5Operand.Vector(3)));

        Assert.False(Analyze().UnobservedWrites(write).Contains(3));
    }

    [Fact]
    public void ACrossLaneReadObservesTheLanesTheBoundsCheckTurnedOff()
    {
        NarrowForGood();
        var write = Add(Vop2(0, "VAddU32", 3, Operand(1), Gen5Operand.Vector(2)));
        Add(new Gen5ShaderInstruction(0, Gen5ShaderEncoding.Vop1, "VMovB32", [0u, 0u], [Gen5Operand.Vector(3)], [Gen5Operand.Vector(4)],
            new Gen5DppControl(0x111, false, false, 0, 0, 0xF, 0xF)));

        Assert.False(Analyze().UnobservedWrites(write).Contains(3));
    }

    [Fact]
    public void ARestoreOfTheSavedFullMaskWakesTheNarrowedLanes()
    {
        SaveExec(20);
        NarrowForGood();
        var write = Add(Vop2(0, "VAddU32", 3, Operand(1), Gen5Operand.Vector(2)));
        RestoreExec(20);
        Add(Vop2(0, "VAddU32", 4, Operand(1), Gen5Operand.Vector(3)));

        Assert.False(Analyze().UnobservedWrites(write).Contains(3));
    }

    private void EnterWholeWave(uint saved) =>
        Add(Sop1(0, "SOrSaveexecB64", saved, new Gen5Operand(Gen5OperandKind.EncodedConstant, 193)));

    private Gen5VectorObservationPlan Analyze(bool trackWholeWave)
    {
        Add(EndProgram(0));
        var plan = Gen5LaneObservationAnalysis.Analyze(Program([.. _instructions]), wave32: false, trackLaneSpills: true, trackWholeWave);
        Assert.NotNull(plan);
        return plan;
    }

    [Fact]
    public void AWholeWaveSection_WakesTheMaskedOffLanesOnlyForWhatItReads()
    {
        NarrowForGood();
        var write = Add(Vop2(0, "VAddU32", 6, Operand(5), Gen5Operand.Vector(0)));
        EnterWholeWave(24);
        Add(MoveVector(0, 7, 3));
        RestoreExec(24);
        Add(Vop2(0, "VAddU32", 6, Gen5Operand.Vector(7), Gen5Operand.Vector(6)));

        var instructions = _instructions.ToList();
        Assert.True(Analyze(trackWholeWave: true).UnobservedWrites(write).Contains(6));
        _instructions.Clear();
        _instructions.AddRange(instructions);
        Assert.False(Analyze(trackWholeWave: false).UnobservedWrites(write).Contains(6));
    }

    [Fact]
    public void ARegisterAWholeWaveSectionReads_IsObserved()
    {
        NarrowForGood();
        var write = Add(Vop2(0, "VAddU32", 6, Operand(5), Gen5Operand.Vector(0)));
        EnterWholeWave(24);
        Add(Vop2(0, "VAddU32", 7, Operand(1), Gen5Operand.Vector(6)));
        RestoreExec(24);
        Add(Vop2(0, "VAddU32", 8, Gen5Operand.Vector(7), Gen5Operand.Vector(6)));

        Assert.False(Analyze(trackWholeWave: true).UnobservedWrites(write).Contains(6));
    }

    [Fact]
    public void ASectionThatHandsBackAnotherMask_KeepsEveryLiveRegisterObserved()
    {
        NarrowForGood();
        var write = Add(Vop2(0, "VAddU32", 6, Operand(5), Gen5Operand.Vector(0)));
        EnterWholeWave(24);
        Add(MoveVector(0, 7, 3));
        RestoreExec(30);
        Add(Vop2(0, "VAddU32", 6, Gen5Operand.Vector(7), Gen5Operand.Vector(6)));

        Assert.False(Analyze(trackWholeWave: true).UnobservedWrites(write).Contains(6));
    }
}
