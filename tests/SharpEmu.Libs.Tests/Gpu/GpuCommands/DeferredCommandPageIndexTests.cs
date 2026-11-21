// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.GpuCommands;

// The command thread's queued stores, fills and copies indexed by guest page (upstream a7ef0c0a6,
// 9e1eed213, b13a98086 ported to DeferredCommandThread).
[Collection(CommandStreamQueueStateCollection.Name)]
public sealed class DeferredCommandPageIndexTests
{
    private const ulong Base = RecordingCommandStreamHost.MemoryBase;
    private static readonly Action Nothing = static () => { };

    private static DeferredCommandThread NewThread(RecordingCommandStreamHost host) =>
        new(new CommandStreamQueue(host), host.Memory, _ => { }, null, Nothing, recycleAsyncItems: true);

    [Fact]
    public void AReadOfAPageNoQueuedItemTouchesTakesNoLock()
    {
        var host = new RecordingCommandStreamHost();
        var thread = NewThread(host);
        thread.PostFill(Nothing, Base + 0x1000, 0x10, 0xAABBCCDD, ungated: true);
        thread.PostOwnedWrite(Base + 0x2000, [1, 2, 3, 4]);
        thread.PostLabel(Nothing, Base + 0x4000, 8);
        Span<byte> bytes = stackalloc byte[4];

        // Another page, and the page of a queued label: nothing the stream reads waits for either.
        Assert.False(thread.TryReadQueuedFill(Base + 0x3000, bytes));
        thread.WaitForPendingRange(Base + 0x3000, 4);
        Assert.True(thread.ReadOverlay(Base + 0x3000, bytes));
        Assert.False(thread.TryReadQueuedFill(Base + 0x4000, bytes));
        Assert.True(thread.ReadOverlay(Base + 0x4000, bytes));
        Assert.Equal(0, thread.LockedPendingReads);

        // The pages that hold queued items are still answered from them.
        Assert.True(thread.TryReadQueuedFill(Base + 0x1004, bytes));
        Assert.Equal(0xAABBCCDDu, BinaryPrimitives.ReadUInt32LittleEndian(bytes));
        Assert.True(thread.ReadOverlay(Base + 0x2000, bytes));
        Assert.Equal(new byte[] { 1, 2, 3, 4 }, bytes.ToArray());
        Assert.True(thread.LockedPendingReads > 0);

        // Once the worker ran them, their pages read without the lock again.
        Assert.Equal(3, thread.RunBatch(long.MaxValue, 64, Nothing));
        var locked = thread.LockedPendingReads;
        Assert.False(thread.TryReadQueuedFill(Base + 0x1004, bytes));
        Assert.True(thread.ReadOverlay(Base + 0x2000, bytes));
        Assert.Equal(new byte[] { 1, 2, 3, 4 }, bytes.ToArray());
        Assert.Equal(locked, thread.LockedPendingReads);
    }

    [Fact]
    public void TheLastQueuedItemOverTheBytesDecidesThemAcrossPageAndWideItems()
    {
        var host = new RecordingCommandStreamHost();
        var thread = NewThread(host);
        Span<byte> bytes = stackalloc byte[4];

        // A fill over 128 pages is listed as wide; a known-bytes write inside it on one page comes later.
        thread.PostFill(Nothing, Base, 0x80000, 0x11111111, ungated: true);
        thread.PostWrite(Nothing, Base + 0x5000, [0x22, 0x22, 0x22, 0x22], ungated: true);
        Assert.True(thread.TryReadQueuedFill(Base + 0x5000, bytes));
        Assert.Equal(0x22222222u, BinaryPrimitives.ReadUInt32LittleEndian(bytes));
        Assert.True(thread.TryReadQueuedFill(Base + 0x6000, bytes));
        Assert.Equal(0x11111111u, BinaryPrimitives.ReadUInt32LittleEndian(bytes));

        // A wide fill queued after the page's write decides the page again.
        thread.PostFill(Nothing, Base, 0x80000, 0x33333333, ungated: true);
        Assert.True(thread.TryReadQueuedFill(Base + 0x5000, bytes));
        Assert.Equal(0x33333333u, BinaryPrimitives.ReadUInt32LittleEndian(bytes));

        // A queued store over the bytes keeps the read from being answered by a range item.
        thread.PostOwnedWrite(Base + 0x5002, [9, 9]);
        Assert.False(thread.TryReadQueuedFill(Base + 0x5000, bytes));
        Assert.Equal(4, thread.RunBatch(long.MaxValue, 64, Nothing));
    }

    [Fact]
    public void QueuedStoresApplyInQueueOrderOnOnePageAndAcrossPages()
    {
        var host = new RecordingCommandStreamHost();
        var thread = NewThread(host);
        var owned = new byte[] { 1, 1, 1, 1, 1, 1, 1, 1 };
        thread.PostOwnedWrite(Base + 0x1FFC, owned);
        thread.PostOwnedWrite(Base + 0x1FFE, [2, 2, 2, 2]);
        thread.PostWrite(Base + 0x2000, [3]);
        // Owned bytes are queued as they are, not copied.
        owned[0] = 7;

        var page = new byte[4];
        Assert.True(thread.ReadOverlay(Base + 0x1FFC, page));
        Assert.Equal(new byte[] { 7, 1, 2, 2 }, page);
        var across = new byte[8];
        Assert.True(thread.ReadOverlay(Base + 0x1FFC, across));
        Assert.Equal(new byte[] { 7, 1, 2, 2, 3, 2, 1, 1 }, across);

        Assert.Equal(3, thread.RunBatch(long.MaxValue, 64, Nothing));
        Assert.True(host.Memory.TryRead(Base + 0x1FFC, across));
        Assert.Equal(new byte[] { 7, 1, 2, 2, 3, 2, 1, 1 }, across);
    }

    [Fact]
    public void AWideQueuedStoreInterleavesWithThePageStoresInQueueOrder()
    {
        var host = new RecordingCommandStreamHost();
        var thread = NewThread(host);
        thread.PostOwnedWrite(Base + 0x8000, [5, 5, 5, 5]);
        thread.PostOwnedWrite(Base, new byte[0x50000]);
        thread.PostOwnedWrite(Base + 0x8002, [6]);
        var bytes = new byte[4];
        Assert.True(thread.ReadOverlay(Base + 0x8000, bytes));
        Assert.Equal(new byte[] { 0, 0, 6, 0 }, bytes);

        // A wide range queued over a page makes the stream wait for it, as before the index.
        thread.PostRange(Nothing, Base, 0x50000);
        var ran = 0;
        thread.Post(() => ran++);
        Assert.Equal(5, thread.RunBatch(long.MaxValue, 64, Nothing));
        Assert.Equal(1, ran);
        thread.WaitForPendingRange(Base + 0x8000, 4);
        Assert.False(thread.HasWork);
    }
}
