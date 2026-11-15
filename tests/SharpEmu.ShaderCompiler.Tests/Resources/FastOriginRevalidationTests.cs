// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

// A fast snapshot is revalidated with its own draw's inputs, never with whatever the shared inputs object held last.
public sealed class FastOriginRevalidationTests
{
    private static ShaderResourcePlan PointerPlan()
    {
        // s[0:1] points at a V# the program loads into s[4:7] and reads through.
        var plan = ShaderResourcePlan.Extract(Program(ScalarLoad(0, 0, 4, 4), BufferLoad(8, 4), EndProgram(16)), ShaderStage.Compute, Hash, 0, 4);
        Assert.True(ResourceMaterializer.CanMaterializeFast(plan));
        return plan;
    }

    [Fact]
    public void AKeptSnapshotIsStaleWhenItsOwnDescriptorChangedThoughTheLastDrawsAreEqual()
    {
        var plan = PointerPlan();
        var memory = new TestWordMemory { Base = 0x1000, Words = new uint[16] };
        uint[] descriptor = [0x8000, 0x10 << 16, 64, 0x24FAC];
        descriptor.CopyTo(memory.Words, 0);
        descriptor.CopyTo(memory.Words, 8);
        // One inputs object refilled per lookup, as ShaderProgramCache does.
        var inputs = new ResourceRuntimeInputs { ReadMemory = memory.Read };
        inputs.UserData = new uint[] { 0x1000, 0, 0, 0 };
        var kept = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.MaterializeFast(plan, inputs, ref kept, ref specialization, out _));

        // Another draw materialized last, through a pointer to an equal V#.
        inputs.UserData = new uint[] { 0x1020, 0, 0, 0 };
        var other = new ResourceSnapshot();
        Assert.True(ResourceMaterializer.MaterializeFast(plan, inputs, ref other, ref specialization, out _));
        Assert.True(ResourceMaterializer.RevalidateFast(kept));

        // The kept draw's own V# changes; the other draw's does not.
        memory.Words[0] = 0x9000;
        Assert.False(ResourceMaterializer.RevalidateFast(kept));
        Assert.True(ResourceMaterializer.RevalidateFast(other));
    }

    [Fact]
    public void AKeptSnapshotIsCurrentWhenOnlyTheLastDrawsDescriptorChanged()
    {
        var plan = PointerPlan();
        var memory = new TestWordMemory { Base = 0x1000, Words = new uint[16] };
        uint[] descriptor = [0x8000, 0x10 << 16, 64, 0x24FAC];
        descriptor.CopyTo(memory.Words, 0);
        descriptor.CopyTo(memory.Words, 8);
        var inputs = new ResourceRuntimeInputs { ReadMemory = memory.Read };
        inputs.UserData = new uint[] { 0x1000, 0, 0, 0 };
        var kept = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.MaterializeFast(plan, inputs, ref kept, ref specialization, out _));
        inputs.UserData = new uint[] { 0x1020, 0, 0, 0 };
        var other = new ResourceSnapshot();
        Assert.True(ResourceMaterializer.MaterializeFast(plan, inputs, ref other, ref specialization, out _));

        memory.Words[8] = 0x9000;
        Assert.True(ResourceMaterializer.RevalidateFast(kept));
        Assert.False(ResourceMaterializer.RevalidateFast(other));
    }

    [Fact]
    public void ExplicitInputsRevalidateWithTheirOwnUserDataAndShaderBase()
    {
        var plan = PointerPlan();
        var memory = new TestWordMemory { Base = 0x1000, Words = new uint[16] };
        uint[] descriptor = [0x8000, 0x10 << 16, 64, 0x24FAC];
        descriptor.CopyTo(memory.Words, 0);
        descriptor.CopyTo(memory.Words, 8);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.MaterializeFast(plan, Inputs([0x1000, 0, 0, 0], memory.Read), ref snapshot, ref specialization, out _));

        // Equal words through another pointer revalidate; a changed word or an unreadable one does not.
        Assert.True(ResourceMaterializer.Revalidate(plan, snapshot, Inputs([0x1020, 0, 0, 0], memory.Read)));
        memory.Words[9] = 0x20 << 16;
        Assert.False(ResourceMaterializer.Revalidate(plan, snapshot, Inputs([0x1020, 0, 0, 0], memory.Read)));
        Assert.True(ResourceMaterializer.Revalidate(plan, snapshot, Inputs([0x1000, 0, 0, 0], memory.Read)));
        Assert.False(ResourceMaterializer.Revalidate(plan, snapshot, Inputs([0x4000, 0, 0, 0], memory.Read)));
        Assert.False(ResourceMaterializer.LastRevalidateRefused);

        // A reader wrapper that splits the table phase is refused, never guessed.
        var phased = new ResourceRuntimeInputs { UserData = new uint[] { 0x1000, 0, 0, 0 }, ReadMemory = memory.Read, TablePhase = _ => { } };
        Assert.False(ResourceMaterializer.Revalidate(plan, snapshot, phased));
        Assert.True(ResourceMaterializer.LastRevalidateRefused);
    }
}
