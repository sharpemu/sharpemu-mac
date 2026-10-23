// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A resident read (a prefetch run, a grouped read, a cache hit's validation) over bytes a
/// recorded GPU store has yet to land is refused, and so is the in-place comparison, so the
/// caller reads the words one at a time: that read waits for the store, or the word is left
/// to the device. Neither returns the bytes the store is about to replace.
/// </summary>
public sealed unsafe partial class MetalClearFoldingTests
{
    private const ulong PendingTable = GuestBase + 0x100000;

    [Fact]
    public void ResidentReadsAndComparesOverAPendingStoreAreRefusedAndTheWordReadWaits()
    {
        if (!RequireDevice() || !MetalV2Diagnostics.ResidentReadPending) return;
        using var rig = new Rig(Device.Value);
        rig.AliasAccess = true;
        var host = rig.Host;
        for (var word = 0UL; word < 8; word++)
        {
            rig.WriteBacking32(PendingTable + word * sizeof(uint), 0xA000u + (uint)word);
        }

        // What a cache entry recorded last frame: the table as memory holds it.
        var recorded = new byte[8 * sizeof(uint)];
        Assert.True(host.TryReadResidentGuestBytes(PendingTable, recorded, clean: false));
        Assert.True(host.TryCompareResidentGuestBytes([PendingTable], [recorded.Length], [0], recorded));

        ulong serial;
        using (host.BeginPreparation())
        {
            var target = rig.Target(TargetAddress);
            var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
            host.BeginRendering(Rig.PassOver(attachment.View));
            serial = host.SubmissionSerial;
            // A dispatch in this submission stores words 2 and 3 of the table.
            host.NoteGuestMemoryWrite(PendingTable + 8, 8, MetalGpuWriteOrigin.StorageBuffer);
            var before = host.ResidentReadsPending;

            // A run over the pending words, and a single pending word, are refused.
            Assert.False(host.TryReadResidentGuestBytes(PendingTable, new byte[recorded.Length], clean: false));
            Assert.False(host.TryReadResidentGuestBytes(PendingTable + 12, new byte[sizeof(uint)], clean: false));
            // Words beside the store are read without waiting.
            var beside = new byte[2 * sizeof(uint)];
            Assert.True(host.TryReadResidentGuestBytes(PendingTable + 16, beside, clean: false));
            Assert.Equal(0xA004u, BitConverter.ToUInt32(beside));
            // Pending GPU bytes are not clean, even through resident bulk reads.
            Assert.False(host.TryReadResidentGuestBytes(PendingTable, new byte[recorded.Length], clean: true));
            // A hit's validation over the range is not a hit, though memory still equals the record.
            Assert.False(host.TryCompareResidentGuestBytes([PendingTable], [recorded.Length], [0], recorded));
            Assert.True(host.ResidentReadsPending > before);

            // The word read one at a time is one the device can be handed (DeviceBindingLoads).
            Assert.True(host.ResourceWordPending!(PendingTable + 8));
            Assert.False(host.ResourceWordPending!(PendingTable + 16));
            Assert.Equal(serial, host.SubmissionSerial);
            host.EndRendering();
        }

        // Or it is read, and the read waits for the store's submission.
        Assert.True(host.TryReadGuestWord(PendingTable + 8, out _));
        Assert.True(host.CompletedSerial >= serial);

        // Landed: the resident read and the comparison answer again.
        Assert.True(host.TryReadResidentGuestBytes(PendingTable, new byte[recorded.Length], clean: false));
        Assert.True(host.TryCompareResidentGuestBytes([PendingTable], [recorded.Length], [0], recorded));
    }
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ResidentReadsReturnTheActualGpuFillOnlyAfterCompletion(bool completeFirst)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        rig.AliasAccess = true;
        rig.WriteBacking32(PendingTable, 0x12345678);
        // A large fill uses the GPU, with storage ownership pending until completion.
        host.FillBufferInOrder(PendingTable, 0x2000, 0xA5A5A5A5);
        if (completeFirst) host.FlushAndWaitForGpu();
        var bytes = new byte[16];
        if (!completeFirst)
        {
            Assert.False(host.TryReadResidentGuestBytes(PendingTable, bytes, false));
            Assert.False(host.TryReadResidentGuestBytes(PendingTable, bytes, true));
            Assert.False(host.TryReadCleanGuestWord(PendingTable, out _));
            host.FlushAndWaitForGpu();
        }
        Assert.True(host.TryReadResidentGuestBytes(PendingTable, bytes, false));
        Assert.All(Enumerable.Range(0, 4), i => Assert.Equal(0xA5A5A5A5u, BitConverter.ToUInt32(bytes, i * 4)));
        Assert.True(host.TryCompareResidentGuestBytes([PendingTable], [16], [0], bytes));
        rig.WriteBacking32(PendingTable, 0xABCDEF01);
        Assert.False(host.TryCompareResidentGuestBytes([PendingTable], [16], [0], bytes));
        Assert.True(host.TryReadResidentGuestBytes(PendingTable, bytes, false));
        Assert.Equal(0xABCDEF01u, BitConverter.ToUInt32(bytes));
    }

}
