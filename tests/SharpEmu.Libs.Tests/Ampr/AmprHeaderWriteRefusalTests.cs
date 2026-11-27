// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using System.Collections;
using System.Reflection;
using SharpEmu.HLE;
using SharpEmu.Libs.Ampr;
using SharpEmu.Libs.Tests.Kernel;
using Xunit;

namespace SharpEmu.Libs.Tests.Ampr;

// Shares the kernel path state an AMPR command buffer reads through
// (TryPreindexApp0 -> ResolveGuestPath consults the global app0 root and mount
// table) with the kernel tests that rewrite it, so both run in one collection
// instead of racing in parallel.
[Collection(KernelMemoryCompatStateCollection.Name)]
public sealed class AmprHeaderWriteRefusalTests
{
    private const ulong Header = 0x7060EBB648;
    private const ulong OldBuffer = 0x7000000000;
    private const ulong NewBuffer = 0x75A0E34000;

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void SetBuffer_FirstRefusedFieldStopsWritesAndPreservesPriorHostState(int failedAttempt)
    {
        var memory = new HeaderMemory();
        var context = Context(memory, OldBuffer, 0x1000);
        Assert.Equal(0, AmprExports.CommandBufferSetBuffer(context));
        var state = States()[Header]!;
        memory.Writes.Clear();
        memory.FailAttempt = failedAttempt;
        context[CpuRegister.Rsi] = NewBuffer;
        context[CpuRegister.Rdx] = 0x4000;
        try
        {
            Assert.Equal((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT,
                AmprExports.CommandBufferSetBuffer(context));
            (ulong Address, int Count)[] expected = [(Header + 12, 4), (Header + 16, 8), (Header + 4, 4), (Header + 8, 4)];
            Assert.Equal(expected.Take(failedAttempt).ToArray(), memory.Writes.ToArray());
            Assert.Same(state, States()[Header]);
            Assert.Equal(OldBuffer, Field(state, "Buffer"));
            Assert.Equal(0x1000UL, Field(state, "Size"));
            Assert.Equal(0UL, Field(state, "WriteOffset"));
            Assert.Equal(0UL, Field(state, "CommandCount"));
        }
        finally { RemoveState(); }
    }

    [Fact]
    public void Constructor_RefusedWholeHeaderPreservesPriorHostState()
    {
        var memory = new HeaderMemory();
        var context = Context(memory, OldBuffer, 0x1000);
        Assert.Equal(0, AmprExports.CommandBufferSetBuffer(context));
        var state = States()[Header]!;
        memory.Writes.Clear(); memory.FailAttempt = 1;
        try
        {
            Assert.Equal((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT,
                AmprExports.CommandBufferConstructor(context));
            Assert.Equal(new[] { (Header, 24) }, memory.Writes.ToArray());
            Assert.Same(state, States()[Header]);
            Assert.Equal(OldBuffer, Field(state, "Buffer"));
        }
        finally { RemoveState(); }
    }

    [Fact]
    public void SetBuffer_SuccessWritesOnlyHeaderThenPublishesNewHostState()
    {
        var memory = new HeaderMemory();
        var context = Context(memory, NewBuffer, 0x4000);
        try
        {
            Assert.Equal(0, AmprExports.CommandBufferSetBuffer(context));
            Assert.Equal(new[] { (Header + 12, 4), (Header + 16, 8), (Header + 4, 4), (Header + 8, 4) }, memory.Writes.ToArray());
            Assert.Equal(NewBuffer, Field(States()[Header]!, "Buffer"));
            Assert.Equal(0x4000UL, Field(States()[Header]!, "Size"));
        }
        finally { RemoveState(); }
    }

    [Fact]
    public void SetBuffer_OversizedBufferDoesNotWriteOrPublishHostState()
    {
        var memory = new HeaderMemory();
        var context = Context(memory, NewBuffer, (ulong)uint.MaxValue + 1);
        Assert.Equal((int)OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT,
            AmprExports.CommandBufferSetBuffer(context));
        Assert.Empty(memory.Writes);
        Assert.False(States().Contains(Header));
    }

    private static CpuContext Context(HeaderMemory memory, ulong buffer, ulong size)
    {
        var context = new CpuContext(memory, Generation.Gen5);
        context[CpuRegister.Rdi] = Header;
        context[CpuRegister.Rsi] = buffer;
        context[CpuRegister.Rdx] = size;
        return context;
    }

    private static void RemoveState() => typeof(AmprExports)
        .GetMethod("RemoveCommandBufferState", BindingFlags.Static | BindingFlags.NonPublic)!
        .Invoke(null, [Header]);

    private static IDictionary States() => (IDictionary)typeof(AmprExports)
        .GetField("_commandBuffers", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
    private static ulong Field(object state, string name) =>
        (ulong)state.GetType().GetField(name)!.GetValue(state)!;

    private sealed class HeaderMemory : ICpuMemory
    {
        internal readonly List<(ulong Address, int Count)> Writes = [];
        internal int FailAttempt;
        // SetBuffer reads the header's own data pointer first to refuse a buffer that is still attached (upstream's busy
        // check); the header reads as unattached. Any other read (the supplied data buffer) is still refused.
        public bool TryRead(ulong address, Span<byte> destination)
        {
            if (address >= Header && address + (ulong)destination.Length <= Header + 24)
            {
                destination.Clear();
                return true;
            }

            throw new InvalidOperationException("Header writes must not read guest memory outside the header or the supplied data buffer.");
        }
        public bool TryWrite(ulong address, ReadOnlySpan<byte> source)
        {
            Writes.Add((address, source.Length));
            Assert.InRange(address, Header, Header + 23);
            Assert.True((ulong)source.Length <= Header + 24 - address);
            return FailAttempt == 0 || Writes.Count != FailAttempt;
        }
    }
}
