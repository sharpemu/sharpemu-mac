// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.GpuCommands;

// Predication operations, operands that recorded work still owes, register-space waits,
// the acquire and the packets the interpreter consumes beyond the AGC builders.
public sealed class GpuCommandInterpreterPacketSemanticsTests
{
    private const ulong Label = StreamRunner.LabelAddress;
    private const ulong Data = StreamRunner.DataAddress;
    private const ulong Table = StreamRunner.TableAddress;
    private const ulong Ready = 1UL << 63;

    private static uint[] Instances(uint count) => StreamRunner.Packet(PacketOpcode.NumInstances, count);

    private static uint[] Predication(uint operation, uint drawIfVisible, uint hint, ulong address, bool continueSet = false) =>
        StreamRunner.Packet(
            PacketOpcode.SetPredication,
            (operation << 16) | (drawIfVisible << 8) | (hint << 12) | (continueSet ? 1u << 31 : 0),
            StreamRunner.Low(address), StreamRunner.High(address));

    private static uint[] ReleaseMemoryNative(uint eventType, uint dataSelection, uint interruptSelector, ulong destination, ulong value) =>
        StreamRunner.Packet(
            PacketOpcode.ReleaseMemory,
            eventType | (5u << 8),
            (interruptSelector << 24) | (dataSelection << 29),
            StreamRunner.Low(destination), StreamRunner.High(destination),
            StreamRunner.Low(value), StreamRunner.High(value),
            0);

    private static void WriteZPassPairs(RecordingCommandStreamHost host, ulong address, ulong begin, ulong end, int notReadyBlock = -1)
    {
        for (var block = 0; block < 16; block++)
        {
            host.WriteQword(address + (ulong)(block * 16), Ready | begin);
            host.WriteQword(address + (ulong)(block * 16) + 8, block == notReadyBlock ? end : Ready | end);
        }
    }

    [Fact]
    public void ZPassPredication_SumsTheDepthBlocksAndHonoursTheBoolean()
    {
        var runner = new StreamRunner();
        WriteZPassPairs(runner.Host, Data, 10, 10);

        // Nothing passed: a draw-if-visible predicate skips, a draw-if-not-visible one does not.
        runner.Run(Predication(1, 1, 0, Data));
        Assert.True(runner.Interpreter.PredicateSkip);
        runner.Run(Predication(1, 0, 0, Data));
        Assert.False(runner.Interpreter.PredicateSkip);

        runner.Host.WriteQword(Data + (3 * 16) + 8, Ready | 11);
        runner.Run(Predication(1, 1, 0, Data));
        Assert.False(runner.Interpreter.PredicateSkip);
        Assert.Empty(runner.Host.Calls);
    }

    [Fact]
    public void ZPassPredication_ContinueOrsIntoThePreviousResult()
    {
        var runner = new StreamRunner();
        WriteZPassPairs(runner.Host, Data, 10, 12);
        WriteZPassPairs(runner.Host, Data + 0x100, 10, 10);

        runner.Run(Predication(1, 1, 0, Data), Predication(1, 1, 0, Data + 0x100, continueSet: true));
        Assert.False(runner.Interpreter.PredicateSkip);

        runner.Run(Predication(1, 1, 0, Data + 0x100));
        Assert.True(runner.Interpreter.PredicateSkip);
    }

    [Fact]
    public void ZPassPredication_ResultsNotReadyParkOrRunUnpredicated()
    {
        var runner = new StreamRunner { };
        runner.Host.PendingGpuWrites = [];
        WriteZPassPairs(runner.Host, Data, 10, 10, notReadyBlock: 7);

        Assert.Equal(SubmissionProgress.Blocked, runner.Run(Predication(1, 1, 0, Data), Instances(3)));
        Assert.NotEqual(3u, runner.Interpreter.InstanceCount);

        // The dump lands and the same packet runs again from its start.
        runner.Host.WriteQword(Data + (7 * 16) + 8, Ready | 10);
        Assert.Equal(SubmissionProgress.Complete, runner.Run());
        Assert.True(runner.Interpreter.PredicateSkip);
        Assert.Equal(3u, runner.Interpreter.InstanceCount);

        WriteZPassPairs(runner.Host, Data, 10, 10, notReadyBlock: 2);
        Assert.Equal(SubmissionProgress.Complete, runner.Run(Predication(1, 1, 1, Data)));
        Assert.False(runner.Interpreter.PredicateSkip);
    }

    [Fact]
    public void PrimCountPredication_DrawsWhenNotVisibleOnOverflow()
    {
        var runner = new StreamRunner();
        runner.Host.WriteQword(Data, 10);
        runner.Host.WriteQword(Data + 8, 10);
        runner.Host.WriteQword(Data + 16, 15);
        runner.Host.WriteQword(Data + 24, 15);

        runner.Run(Predication(2, 0, 0, Data));
        Assert.True(runner.Interpreter.PredicateSkip);

        runner.Host.WriteQword(Data + 24, 17);
        runner.Run(Predication(2, 0, 0, Data));
        Assert.False(runner.Interpreter.PredicateSkip);
    }

    [Fact]
    public void VulkanPredication_ReadsADword()
    {
        var runner = new StreamRunner();
        runner.Host.WriteQword(Label, 0xFFFF_FFFF_0000_0000);

        runner.Run(Predication(4, 1, 0, Label));
        Assert.True(runner.Interpreter.PredicateSkip);

        runner.Host.WriteDword(Label, 1);
        runner.Run(Predication(4, 1, 0, Label));
        Assert.False(runner.Interpreter.PredicateSkip);
        Assert.Contains("predication operation is unknown", runner.RunExpectingFatal(Predication(5, 0, 0, Label)).Message);
    }

    [Fact]
    public void OwedPredicate_ParksInsteadOfDrainingTheGpu()
    {
        var runner = new StreamRunner();
        runner.Host.WriteQword(Label, 0);
        runner.Host.OwedOperandWrites.Add((Label, 8));

        Assert.Equal(SubmissionProgress.Blocked, runner.Run(Predication(3, 1, 0, Label), Instances(4)));
        Assert.Equal(new[] { $"commit_operand {Label:X} 8" }, runner.Host.Calls);

        runner.Host.OwedOperandWrites.Clear();
        runner.Host.WriteQword(Label, 1);
        Assert.Equal(SubmissionProgress.Complete, runner.Run());
        Assert.False(runner.Interpreter.PredicateSkip);
        Assert.Equal(4u, runner.Interpreter.InstanceCount);
        Assert.DoesNotContain("flush_and_wait", runner.Host.Calls);
    }

    [Fact]
    public void ConditionalExecute_SkipToTheEndOfTheBufferAndCachePolicy()
    {
        var runner = new StreamRunner();
        runner.Host.WriteDword(Label, 0);
        runner.Run(Instances(1));

        // The skipped dwords end exactly where the buffer does.
        var execute = StreamRunner.Packet(PacketOpcode.ConditionalExecute, StreamRunner.Low(Label), StreamRunner.High(Label), 2u << 25, 2);
        Assert.Equal(SubmissionProgress.Complete, runner.Run(execute, Instances(5)));
        Assert.Equal(1u, runner.Interpreter.InstanceCount);

        Assert.Contains("conditional-execute packet is not supported",
            runner.RunExpectingFatal(StreamRunner.Packet(PacketOpcode.ConditionalExecute, StreamRunner.Low(Label), StreamRunner.High(Label), 1, 2), Instances(5)).Message);
    }

    [Fact]
    public void ConditionalExecute_OwedOperandParksAndReruns()
    {
        var runner = new StreamRunner();
        runner.Host.WriteDword(Label, 0);
        runner.Host.OwedOperandWrites.Add((Label, 4));
        var execute = StreamRunner.Packet(PacketOpcode.ConditionalExecute, StreamRunner.Low(Label), StreamRunner.High(Label), 0, 2);

        Assert.Equal(SubmissionProgress.Blocked, runner.Run(execute, Instances(5), Instances(6)));
        runner.Host.OwedOperandWrites.Clear();
        runner.Host.WriteDword(Label, 1);
        Assert.Equal(SubmissionProgress.Complete, runner.Run());
        Assert.Equal(6u, runner.Interpreter.InstanceCount);
    }

    [Fact]
    public void ConditionalWrite_OwedCompareParksWithoutSynchronizing()
    {
        var runner = new StreamRunner();
        runner.Host.DefersEndOfPipeLabels = true;
        runner.Host.WriteDword(Data, 0);
        runner.Host.WriteDword(Label, 0);
        runner.Host.OwedOperandWrites.Add((Data, 4));
        var write = StreamRunner.Packet(
            PacketOpcode.ConditionalWrite, 3u | (1u << 4) | (1u << 8),
            StreamRunner.Low(Data), StreamRunner.High(Data), 7, 0xFF,
            StreamRunner.Low(Label), StreamRunner.High(Label), 0x11);

        Assert.Equal(SubmissionProgress.Blocked, runner.Run(write));
        Assert.DoesNotContain("synchronize", runner.Host.Calls);
        Assert.Equal(0u, runner.Host.ReadDword(Label));

        runner.Host.OwedOperandWrites.Clear();
        runner.Host.WriteDword(Data, 7);
        Assert.Equal(SubmissionProgress.Complete, runner.Run());
        Assert.Equal(0x11u, runner.Host.ReadDword(Label));
    }

    [Fact]
    public void Branch_OwedCompareParks()
    {
        var runner = new StreamRunner();
        runner.Host.WriteWords(Table, Instances(9));
        runner.Host.WriteQword(Label, 1);
        runner.Host.OwedOperandWrites.Add((Label, 8));
        var branch = StreamRunner.Packet(
            PacketOpcode.IndirectBuffer,
            1u | (3u << 8),
            StreamRunner.Low(Label), StreamRunner.High(Label),
            0xFFFF_FFFF, 0xFFFF_FFFF,
            1, 0,
            StreamRunner.Low(Table), StreamRunner.High(Table), 2,
            0, 0, 0);

        Assert.Equal(SubmissionProgress.Blocked, runner.Run(branch));
        runner.Host.OwedOperandWrites.Clear();
        Assert.Equal(SubmissionProgress.Complete, runner.Run());
        Assert.Equal(9u, runner.Interpreter.InstanceCount);
    }

    private static uint[] RegisterWait(uint function, uint register, uint reference, uint mask) =>
        StreamRunner.Packet(PacketOpcode.WaitRegisterMemory, function, register, 0, reference, mask, 0);

    [Fact]
    public void RegisterSpaceWait_ComparesTheRegisterThisQueueWrote()
    {
        var runner = new StreamRunner();
        var setIndexType = StreamRunner.Packet(PacketOpcode.SetUserConfigRegister, RegisterBankLayout.IndexTypeRegister, 1);

        Assert.Equal(SubmissionProgress.Complete, runner.Run(setIndexType, RegisterWait(3, 0xC000 + RegisterBankLayout.IndexTypeRegister, 1, 0xFFFF_FFFF), Instances(2)));
        Assert.Equal(2u, runner.Interpreter.InstanceCount);

        var setShader = StreamRunner.Packet(PacketOpcode.SetShaderRegister, 0x0C, 0x55);
        Assert.Equal(SubmissionProgress.Complete, runner.Run(setShader, RegisterWait(3, 0x2C0C, 0x55, 0xFF)));

        Assert.Contains("can never be satisfied", runner.RunExpectingFatal(RegisterWait(3, 0x2C0C, 0x56, 0xFF)).Message);
        Assert.Contains("never wrote", runner.RunExpectingFatal(RegisterWait(3, 0x2C0D, 0, 0)).Message);
        Assert.Contains("not modelled", runner.RunExpectingFatal(RegisterWait(3, 0xA100, 0, 0)).Message);
    }

    [Fact]
    public void ShaderRegisterWaitDistinguishesWrittenZeroFromClearedPresence()
    {
        var runner = new StreamRunner();
        var writeZero = StreamRunner.Packet(PacketOpcode.SetShaderRegister, 0x0C, 0);
        Assert.Equal(SubmissionProgress.Complete, runner.Run(writeZero, RegisterWait(3, 0x2C0C, 0, uint.MaxValue)));
        var alias = runner.Interpreter.Registers.Shader;
        runner.Interpreter.Registers.Reset();
        Assert.Same(alias, runner.Interpreter.Registers.Shader);
        Assert.Contains("never wrote", runner.RunExpectingFatal(RegisterWait(3, 0x2C0C, 0, uint.MaxValue)).Message);
    }

    [Fact]
    public void AcquireMemory_BarriersOnlyForCacheActions()
    {
        var runner = new StreamRunner();

        // gfx10 ACQUIRE_MEM: COHER_CNTL, size, size hi, base, base hi, poll, GCR_CNTL.
        runner.Run(StreamRunner.Packet(PacketOpcode.AcquireMemory, 0, 0x10, 0, 0x100, 0, 10, 0));
        Assert.Empty(runner.Host.Calls);
        Assert.Equal(1, runner.Host.ShaderWriteReleases);

        runner.Run(StreamRunner.Packet(PacketOpcode.AcquireMemory, 0, 0x10, 0, 0x100, 0, 10, 1u << 14));
        Assert.Equal(new[] { "barrier" }, runner.Host.Calls);

        runner.Host.Calls.Clear();
        runner.Run(StreamRunner.CustomPacket(PacketOpcode.Nop, PacketCustomCode.AcquireMemory, (1u << 31) | (1u << 25), 0x10, 0, 0x100, 0, 10, 0));
        Assert.Equal(new[] { "barrier" }, runner.Host.Calls);
        Assert.Equal(3, runner.Host.ShaderWriteReleases);
    }

    [Fact]
    public void SurfaceSyncAndContextControl_AreConsumed()
    {
        var runner = new StreamRunner();

        Assert.Equal(SubmissionProgress.Complete, runner.Run(
            StreamRunner.Packet(PacketOpcode.ContextControl, 0x8000_0001u, 0x8000_0001u),
            StreamRunner.Packet(PacketOpcode.SurfaceSync, 1u << 23, 0xFFFF_FFFF, 0, 10),
            Instances(7)));
        Assert.Equal(new[] { "barrier" }, runner.Host.Calls);
        Assert.Equal(1, runner.Host.ShaderWriteReleases);
        Assert.Equal(7u, runner.Interpreter.InstanceCount);
    }

    [Fact]
    public void IndexAttributesIndirect_LoadsTheRecordAtTheIndex()
    {
        var runner = new StreamRunner();
        runner.Host.WriteWords(Data + 16, [0x2000_0000, 0x1, 96, 1]);

        runner.Run(StreamRunner.Packet(PacketOpcode.IndexAttributesIndirect, StreamRunner.Low(Data), StreamRunner.High(Data), 1));

        Assert.Equal(0x1_2000_0000UL, runner.Interpreter.IndexBaseAddress);
        Assert.Equal(96u, runner.Interpreter.IndexBufferSize);
        Assert.Equal(1u, runner.Interpreter.IndexTypeAndSize);

        Assert.Contains("index-attributes packet is not supported",
            runner.RunExpectingFatal(StreamRunner.Packet(PacketOpcode.IndexAttributesIndirect, StreamRunner.Low(Data) | 1, StreamRunner.High(Data), 0)).Message);
    }

    [Fact]
    public void ConstantIndirectBuffer_RunsItsPackets()
    {
        var runner = new StreamRunner();
        runner.Host.WriteWords(Table, Instances(12));

        Assert.Equal(SubmissionProgress.Complete, runner.Run(
            StreamRunner.Packet(PacketOpcode.IndirectBufferConstant, StreamRunner.Low(Table), StreamRunner.High(Table), 2),
            Instances(1)));
        Assert.Equal(1u, runner.Interpreter.InstanceCount);

        runner.Run(StreamRunner.Packet(PacketOpcode.IndirectBufferConstant, StreamRunner.Low(Table), StreamRunner.High(Table), 2), Instances(1),
            StreamRunner.Packet(PacketOpcode.IndirectBufferConstant, StreamRunner.Low(Table), StreamRunner.High(Table), 2), Instances(3));
        Assert.Equal(3u, runner.Interpreter.InstanceCount);
    }

    [Fact]
    public void ReleaseMemory_DataSelectSevenAddsAndFourWritesTheClock()
    {
        var runner = new StreamRunner();
        runner.Host.WriteDword(Label, 5);

        runner.Run(ReleaseMemoryNative(0x28, 7, 0, Label, 3));
        Assert.Equal(8u, runner.Host.ReadDword(Label));
        Assert.Equal(EndOfPipeWriteKind.AtomicAdd32, runner.Host.EndOfPipeWrites[^1].Kind);

        runner.Run(ReleaseMemoryNative(0x28, 4, 0, Data, 0));
        Assert.NotEqual(0UL, runner.Host.ReadQword(Data));
        Assert.Equal(EndOfPipeWriteKind.ClockWrite, runner.Host.EndOfPipeWrites[^1].Kind);
    }

    [Fact]
    public void ReleaseMemory_DeferredAtomicAddIsLeftToTheHost()
    {
        var runner = new StreamRunner();
        runner.Host.DefersEndOfPipeLabels = true;
        runner.Host.WriteDword(Label, 5);

        runner.Run(ReleaseMemoryNative(0x28, 7, 2, Label, 3));

        Assert.Equal(5u, runner.Host.ReadDword(Label));
        // The add, then the interrupt the selector asks for once it has landed.
        Assert.Equal(2, runner.Host.EndOfPipeWrites.Count);
        Assert.Equal(EndOfPipeWriteKind.AtomicAdd32, runner.Host.EndOfPipeWrites[0].Kind);
        Assert.Equal(3UL, runner.Host.EndOfPipeWrites[0].Value);
        Assert.Equal(EndOfPipeWriteKind.InterruptOnly, runner.Host.EndOfPipeWrites[1].Kind);
    }

    [Fact]
    public void ReleaseMemory_GdsStoreRidesTheStreamOnAnOrderedHost()
    {
        var runner = new StreamRunner(queueId: 2);
        runner.Host.WritesGdsAtEndOfPipe = true;
        runner.Host.Gds[0] = 0x42;

        runner.Run(ReleaseMemoryNative(0x28, 5, 1, Label, 1u << 16));

        Assert.DoesNotContain("synchronize", runner.Host.Calls);
        Assert.DoesNotContain(runner.Host.Calls, call => call.StartsWith("read_gds", StringComparison.Ordinal));
        Assert.Contains("eop GdsWrite32", runner.Host.Calls);
    }
}
