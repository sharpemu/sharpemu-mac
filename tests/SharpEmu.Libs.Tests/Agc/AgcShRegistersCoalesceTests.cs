// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.Libs.Agc;
using Xunit;

namespace SharpEmu.Libs.Tests.Agc;

// sceAgcCbSetShRegistersDirect with its runs coalesced into one reservation and one write must
// leave exactly the bytes, cursor, return value and full-callback calls of the run-by-run path.
[Collection(AgcCommandBufferChainCollection.Name)]
public sealed class AgcShRegistersCoalesceTests
{
    private const ulong BaseAddress = 0x2_7000_0000;
    private const ulong CommandBuffer = BaseAddress + 0x100;
    private const ulong Registers = BaseAddress + 0x200;
    private const ulong Packets = BaseAddress + 0x1000;
    private const ulong CallbackSpace = BaseAddress + 0x2000;
    private const int MemoryBytes = 0x4000;

    // The Demon's Souls shape: unsorted, a duplicated register, 7 runs of up to 4 registers.
    private static readonly (uint Offset, uint Value)[] DsShape =
    [
        (0x41, 0xA1), (0x10, 0x01), (0x12, 0x03), (0x80, 0xB0), (0x11, 0x02), (0x40, 0xA0),
        (0x09, 0x0F), (0x12, 0x33), (0x83, 0xB3), (0x81, 0xB1), (0x100, 0xC0), (0x82, 0xB2),
        (0x08, 0x0E), (0x30, 0xD0),
    ];

    // Total dwords for DsShape: 14 registers + 2 x 7 runs.
    private const uint DsShapeDwords = 14 + 2 * 7;

    [Theory]
    [InlineData(1024u, 0u, false)]               // plenty of room
    [InlineData(DsShapeDwords, 0u, false)]       // exactly the room every run needs
    [InlineData(DsShapeDwords + 4, 4u, false)]   // exactly, with reserved dwords
    [InlineData(DsShapeDwords - 1, 0u, false)]   // nearly full: the last run reaches the callback
    [InlineData(10u, 0u, false)]                 // the callback fires after the first runs
    [InlineData(DsShapeDwords - 1, 0u, true)]    // nearly full in a leaf: declines as before
    [InlineData(0u, 0u, false)]                  // full: the first run reaches the callback
    public void CoalescedRunsMatchTheRunByRunPathByteForByte(uint roomDwords, uint reservedDwords, bool leaf)
    {
        var perRun = Run(coalesce: false, DsShape, roomDwords, reservedDwords, leaf);
        var coalesced = Run(coalesce: true, DsShape, roomDwords, reservedDwords, leaf);
        Assert.Equal(perRun.Result, coalesced.Result);
        Assert.Equal(perRun.Rax, coalesced.Rax);
        Assert.Equal(perRun.Declined, coalesced.Declined);
        Assert.Equal(perRun.Callbacks, coalesced.Callbacks);
        Assert.Equal(perRun.Memory, coalesced.Memory);
        if (roomDwords >= DsShapeDwords + reservedDwords)
        {
            // Room for every run: one reservation from the cursor, no callback.
            Assert.Empty(coalesced.Callbacks);
            Assert.Equal(Packets, coalesced.Rax);
            Assert.Equal(Packets + DsShapeDwords * 4, BinaryPrimitives.ReadUInt64LittleEndian(coalesced.Memory.AsSpan((int)(CommandBuffer + 0x10 - BaseAddress))));
        }
        else if (!leaf)
        {
            Assert.NotEmpty(coalesced.Callbacks);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(32)]
    public void ContiguousAndSingleRegisterListsMatch(int registers)
    {
        var list = Enumerable.Range(0, registers).Select(index => ((uint)(0x200 - index), 0x1000u + (uint)index)).ToArray();
        var perRun = Run(coalesce: false, list, 1024, 0, false);
        var coalesced = Run(coalesce: true, list, 1024, 0, false);
        Assert.Equal(perRun.Rax, coalesced.Rax);
        Assert.Equal(perRun.Memory, coalesced.Memory);
        Assert.Empty(coalesced.Callbacks);
    }

    private sealed record Outcome(int Result, ulong Rax, bool Declined, uint[] Callbacks, byte[] Memory);

    private static Outcome Run(bool coalesce, (uint Offset, uint Value)[] registers, uint roomDwords, uint reservedDwords, bool leaf)
    {
        var memory = new FakeCpuMemory(BaseAddress, MemoryBytes);
        WriteUInt64(memory, CommandBuffer + 0x10, Packets);
        WriteUInt64(memory, CommandBuffer + 0x18, Packets + roomDwords * 4UL);
        WriteUInt64(memory, CommandBuffer + 0x20, 0x1234);
        WriteUInt64(memory, CommandBuffer + 0x28, 0x5678);
        WriteUInt32(memory, CommandBuffer + 0x30, reservedDwords);
        for (var index = 0; index < registers.Length; index++)
        {
            WriteUInt32(memory, Registers + (ulong)index * 8, registers[index].Offset);
            WriteUInt32(memory, Registers + (ulong)index * 8 + 4, registers[index].Value);
        }

        var context = new CpuContext(memory, Generation.Gen5)
        {
            [CpuRegister.Rdi] = CommandBuffer,
            [CpuRegister.Rsi] = Registers,
            [CpuRegister.Rdx] = (ulong)registers.Length,
        };
        context.DeclineGuestCallbacks = leaf;
        var scheduler = new SpaceCallbackScheduler(memory);
        var previousScheduler = GuestThreadExecution.Scheduler;
        var previousCoalesce = AgcExports.CoalesceShRegisterRuns;
        try
        {
            GuestThreadExecution.Scheduler = scheduler;
            AgcExports.CoalesceShRegisterRuns = coalesce;
            var result = AgcExports.CbSetShRegistersDirect(context);
            var bytes = new byte[MemoryBytes];
            Assert.True(memory.TryRead(BaseAddress, bytes));
            return new(result, context[CpuRegister.Rax], context.GuestCallbackDeclined, scheduler.Requests.ToArray(), bytes);
        }
        finally
        {
            AgcExports.CoalesceShRegisterRuns = previousCoalesce;
            GuestThreadExecution.Scheduler = previousScheduler;
        }
    }

    private static void WriteUInt32(FakeCpuMemory memory, ulong address, uint value)
    {
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        Assert.True(memory.TryWrite(address, bytes));
    }

    private static void WriteUInt64(FakeCpuMemory memory, ulong address, ulong value)
    {
        Span<byte> bytes = stackalloc byte[sizeof(ulong)];
        BinaryPrimitives.WriteUInt64LittleEndian(bytes, value);
        Assert.True(memory.TryWrite(address, bytes));
    }

    // The buffer's full callback: records the dwords asked for and hands over a fresh 0x400-byte
    // block each time, as a guest's chaining callback would.
    private sealed class SpaceCallbackScheduler(FakeCpuMemory memory) : IGuestThreadScheduler
    {
        public List<uint> Requests { get; } = [];
        public bool SupportsGuestContextTransfer => false;

        public bool TryCallGuestFunction(CpuContext callerContext, ulong entryPoint, ulong bufferAddress,
            ulong requiredDwords, ulong userData, ulong stackAddress, ulong stackSize, string reason,
            out ulong returnValue, out string? error)
        {
            Assert.Equal(0x1234UL, entryPoint);
            Assert.Equal(CommandBuffer, bufferAddress);
            Assert.Equal(0x5678UL, userData);
            var block = CallbackSpace + (ulong)Requests.Count * 0x400;
            Requests.Add((uint)requiredDwords);
            WriteUInt64(memory, CommandBuffer + 0x10, block);
            WriteUInt64(memory, CommandBuffer + 0x18, block + 0x400);
            returnValue = 0;
            error = null;
            return true;
        }

        public void RegisterGuestThreadContext(ulong threadHandle, CpuContext context) => throw new NotSupportedException();
        public bool TryStartThread(CpuContext creatorContext, GuestThreadStartRequest request, out string? error) => throw new NotSupportedException();
        public bool TryJoinThread(CpuContext callerContext, ulong threadHandle, out ulong returnValue, out string? error) => throw new NotSupportedException();
        public void Pump(CpuContext callerContext, string reason) => throw new NotSupportedException();
        public int WakeBlockedThreads(string wakeKey, int maxCount = int.MaxValue) => throw new NotSupportedException();
        public bool HasPendingGuestExceptionForCurrentThread() => false;
        public bool TrySetGuestThreadPriority(ulong guestThreadHandle, int guestPriority) => throw new NotSupportedException();
        public bool TrySetGuestThreadAffinity(ulong guestThreadHandle, ulong affinityMask) => throw new NotSupportedException();
        public IReadOnlyList<GuestThreadSnapshot> SnapshotThreads() => throw new NotSupportedException();
        public bool TryCallGuestFunction(CpuContext callerContext, ulong entryPoint, ulong firstArgument, ulong secondArgument,
            ulong stackAddress, ulong stackSize, string reason, out string? error) => throw new NotSupportedException();
        public bool TryCallGuestFunction(CpuContext callerContext, ulong entryPoint, ulong firstArgument, ulong secondArgument,
            ulong thirdArgument, ulong fourthArgument, ulong stackAddress, ulong stackSize, string reason,
            out ulong returnValue, out string? error) => throw new NotSupportedException();
        public bool TryCallGuestContinuation(CpuContext callerContext, GuestCpuContinuation continuation, string reason,
            out string? error) => throw new NotSupportedException();
        public bool TryRaiseGuestException(CpuContext callerContext, ulong threadHandle, ulong handler, int exceptionType,
            out string? error) => throw new NotSupportedException();
    }
}
