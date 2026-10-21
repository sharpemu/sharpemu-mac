// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.GpuCommands;

public sealed class GpuCommandInterpreterPacketBlockTests
{
    private const ulong Command = StreamRunner.CommandAddress;
    private const ulong Label = StreamRunner.LabelAddress;
    private const ulong Nested = StreamRunner.DataAddress;

    private static StreamRunner Create(bool verify = false)
    {
        var host = new RecordingCommandStreamHost { SupportsCommandBlocks = true };
        var runner = new StreamRunner(0, host);
        runner.Interpreter.UsePacketBlocks = true;
        runner.Interpreter.VerifyPacketBlocks = verify;
        return runner;
    }

    private static uint[] Instances(uint count) => StreamRunner.Packet(PacketOpcode.NumInstances, count);

    private static uint[] WriteData(ulong destination, uint value) =>
        StreamRunner.Packet(PacketOpcode.WriteData, 5u << 8, StreamRunner.Low(destination), StreamRunner.High(destination), value);

    private static uint[] WaitEqual(ulong address, uint reference) =>
        StreamRunner.Packet(PacketOpcode.WaitRegisterMemory, 0x13u, StreamRunner.Low(address), StreamRunner.High(address), reference, 0xFFFF_FFFFu, 0);

    private static uint[] IndirectBuffer(ulong address, uint dwords) =>
        StreamRunner.Packet(PacketOpcode.IndirectBuffer, StreamRunner.Low(address), StreamRunner.High(address) & 0xFFFF, dwords);

    private static ulong PayloadOf(int dwordIndex) => Command + ((ulong)dwordIndex * sizeof(uint));

    [Fact]
    public void ManyPackets_CostOneHostRead()
    {
        var runner = Create();
        var packets = Enumerable.Range(1, 20).Select(index => Instances((uint)index)).ToArray();

        var progress = runner.Run(packets);

        Assert.Equal(SubmissionProgress.Complete, progress);
        Assert.Equal(20u, runner.Interpreter.InstanceCount);
        Assert.Single(runner.Host.BlockReads);
        Assert.Empty(runner.Host.GuestReads);
    }

    [Fact]
    public void ALongStream_RefillsPerBlockAndNeverReadsPastTheCursor()
    {
        var runner = Create();
        var packets = Enumerable.Range(1, 100).Select(index => Instances((uint)index)).ToArray();

        runner.Run(packets);

        Assert.Equal(100u, runner.Interpreter.InstanceCount);
        Assert.InRange(runner.Host.BlockReads.Count, 2, 3);
        Assert.All(runner.Host.BlockReads, read =>
            Assert.True(read.Address + (ulong)read.Size <= Command + 200 * sizeof(uint)));
        Assert.Empty(runner.Host.GuestReads);
    }

    [Fact]
    public void ACommandProcessorWriteIntoALaterPacketOfTheSameBlock_IsSeenByThatPacket()
    {
        var runner = Create();
        // Dwords: Instances 0-1, WriteData 2-6, Instances 7-8. The write patches the count at 8.
        var progress = runner.Run(Instances(1), WriteData(PayloadOf(8), 77), Instances(5));

        Assert.Equal(SubmissionProgress.Complete, progress);
        Assert.Equal(77u, runner.Interpreter.InstanceCount);
        Assert.True(runner.Host.BlockReads.Count >= 2);
    }

    [Fact]
    public void AHostVersionBumpMidBlock_MakesTheNextPacketReadTheOwedBytes()
    {
        var runner = Create();
        runner.Load(Instances(1), Instances(2), Instances(3));
        var packetsRun = 0;
        // What the real host does when a command-processor write lands in the buffer: the bytes
        // are owed (here: written) and the overlay's version moves.
        var patched = false;
        runner.Host.OnRunPending = () =>
        {
            if (++packetsRun == 2 && !patched)
            {
                patched = true;
                runner.Host.WriteDword(PayloadOf(5), 99);
                runner.Host.BlockVersion++;
            }
        };

        runner.Run();

        Assert.Equal(99u, runner.Interpreter.InstanceCount);
        Assert.Equal(2, runner.Host.BlockReads.Count);
    }

    [Fact]
    public void AGuestCpuWriteBetweenSlices_IsSeenByTheNextSlice()
    {
        var runner = Create();
        // Dwords: Instances 0-1, Wait 2-8, Instances 9-10. The wait is unmet, so the slice ends.
        var progress = runner.Run(Instances(1), WaitEqual(Label, 1), Instances(5));
        Assert.Equal(SubmissionProgress.Blocked, progress);
        Assert.Equal(1u, runner.Interpreter.InstanceCount);
        var readsAfterFirstSlice = runner.Host.BlockReads.Count;

        // The guest rewrites the packet it has not been parsed yet; nothing tells the host.
        runner.Host.WriteDword(PayloadOf(10), 9);
        runner.Host.WriteDword(Label, 1);
        progress = runner.Run();

        Assert.Equal(SubmissionProgress.Complete, progress);
        Assert.Equal(9u, runner.Interpreter.InstanceCount);
        Assert.True(runner.Host.BlockReads.Count > readsAfterFirstSlice);
    }

    [Fact]
    public void ABlockRefusedForCrossingAPage_FallsBackToPacketReads()
    {
        var runner = Create();
        runner.Host.RefusesBlock = (address, size) => (address >> 12) != ((address + (ulong)size - 1) >> 12);
        // Start 8 bytes under a page boundary, so every block from there on crosses it.
        var start = RecordingCommandStreamHost.MemoryBase + 0x2000 - 8;
        var packets = Enumerable.Range(1, 12).Select(index => Instances((uint)index)).ToArray();
        runner.Load(start, StreamRunner.Concat(packets));

        var progress = runner.Interpreter.Process(runner.Commands, start, runner.Dwords);

        Assert.Equal(SubmissionProgress.Complete, progress);
        Assert.Equal(12u, runner.Interpreter.InstanceCount);
        Assert.NotEmpty(runner.Host.GuestReads);
        // A refusal covers its region: the packets in it do not ask again.
        Assert.Single(runner.Host.BlockReads);
    }

    [Fact]
    public void ABlockWithAnUnmappedTail_FallsBackAndTheMappedPacketsStillRun()
    {
        var runner = Create();
        // 40 dwords fit in memory; the cursor claims 64. The stream suspends before the end of the mapped part.
        var start = RecordingCommandStreamHost.MemoryBase + (ulong)RecordingCommandStreamHost.MemorySize - 40 * sizeof(uint);
        var words = StreamRunner.Concat(Instances(3), Instances(4), WaitEqual(Label, 1));
        Assert.True(words.Length <= 16);
        runner.Host.WriteWords(start, words);

        var progress = runner.Interpreter.Process(new PacketCursorStack(), start, 64);

        Assert.Equal(SubmissionProgress.Blocked, progress);
        Assert.Equal(4u, runner.Interpreter.InstanceCount);
        Assert.Single(runner.Host.BlockReads);
        Assert.NotEmpty(runner.Host.GuestReads);
    }

    [Fact]
    public void AChainedIndirectBuffer_HasABlockPerDepthAndTheParentKeepsItsOwn()
    {
        var runner = Create();
        var nested = StreamRunner.Concat(Instances(10), Instances(11), Instances(12));
        runner.Host.WriteWords(Nested, nested);

        var progress = runner.Run(Instances(1), IndirectBuffer(Nested, (uint)nested.Length), Instances(2));

        Assert.Equal(SubmissionProgress.Complete, progress);
        Assert.Equal(2u, runner.Interpreter.InstanceCount);
        Assert.Equal(2, runner.Host.BlockReads.Count);
        Assert.Contains(runner.Host.BlockReads, read => read.Address == Nested);
        Assert.Contains(runner.Host.BlockReads, read => read.Address == Command);
        Assert.Equal(Nested, Assert.Single(runner.Host.GuestReads));
    }

    [Fact]
    public void AnIndirectBuffer_PatchedByTheParentBeforeItRuns_IsReadAfterThePatch()
    {
        var runner = Create();
        var nested = StreamRunner.Concat(Instances(10), Instances(11));
        runner.Host.WriteWords(Nested, nested);
        // The parent patches the nested buffer's second count, then calls it. Dwords: WriteData 0-4, IB 5-8.
        var progress = runner.Run(WriteData(Nested + 3 * sizeof(uint), 55), IndirectBuffer(Nested, (uint)nested.Length));

        Assert.Equal(SubmissionProgress.Complete, progress);
        Assert.Equal(55u, runner.Interpreter.InstanceCount);
    }

    [Fact]
    public void WithBlocksOff_EveryPacketIsReadFromTheHost()
    {
        var runner = Create();
        runner.Interpreter.UsePacketBlocks = false;

        runner.Run(Instances(1), Instances(2), Instances(3));

        Assert.Empty(runner.Host.BlockReads);
        Assert.Equal(3, runner.Host.GuestReads.Count);
    }

    [Fact]
    public void VerifyMode_LogsAndRecoversWhenTheBufferChangedBehindTheBlock()
    {
        var runner = Create(verify: true);
        runner.Load(Instances(1), Instances(2), Instances(3));
        var packetsRun = 0;
        runner.Host.OnRunPending = () =>
        {
            // A store nobody reported: no counter moves.
            if (++packetsRun == 2)
            {
                runner.Host.WriteDword(PayloadOf(5), 42);
            }
        };
        var before = PacketBlockReads.VerifyMismatches;
        var captured = new StringWriter();
        var error = Console.Error;
        Console.SetError(captured);
        try
        {
            runner.Run();
        }
        finally
        {
            Console.SetError(error);
        }

        Assert.True(PacketBlockReads.VerifyMismatches > before);
        Assert.Contains("[PERF][VERIFY] packet-block", captured.ToString());
        // The packet is read from the host once the block is found wrong.
        Assert.Equal(42u, runner.Interpreter.InstanceCount);
    }
}
