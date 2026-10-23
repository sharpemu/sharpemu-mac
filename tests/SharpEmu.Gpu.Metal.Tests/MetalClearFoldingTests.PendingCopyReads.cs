// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A host read of bytes a pending compute memcpy writes is answered from the copy's
/// source, without waiting, while the copy is the last write over them; any later write
/// makes the read wait as before.
/// </summary>
public sealed unsafe partial class MetalClearFoldingTests
{
    private const ulong CopySource = GuestBase + 0x30000;
    private const ulong CopyDestination = GuestBase + 0x31000;
    private const ulong CopyPeriod = 4;
    private const ulong CopyWords = 16;

    // The executor's order for a copy the host refuses: the refusal, then the dispatch's
    // storage binding of the destination, recorded in the open submission.
    private static ulong RecordPendingCopy(Rig rig)
    {
        var host = rig.Host;
        host.FinalGuestTablesEnabled = false;
        for (var word = 0UL; word < CopyPeriod; word++)
        {
            rig.WriteBacking32(CopySource + word * sizeof(uint), 0xA000u + (uint)word);
        }

        Assert.False(host.TryCopyWordsOnHost(CopyDestination, CopySource, CopyPeriod, CopyWords));
        host.NoteGuestMemoryWrite(CopyDestination, CopyWords * sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
        Assert.Equal(1, host.PendingCopyCount);
        return host.SubmissionSerial;
    }

    [Fact]
    public void AReadOfAPendingCopyDestinationIsAnsweredFromItsSourceWithoutWaiting()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.FinalGuestTablesEnabled = false;
        var serial = RecordPendingCopy(rig);

        Assert.True(host.TryReadGuestWord(CopyDestination + 6 * sizeof(uint), out var word));
        Assert.Equal(0xA002u, word);
        // A read across the wrap of the period takes the end and then the start of the source.
        Span<byte> wide = stackalloc byte[2 * sizeof(uint)];
        Assert.True(host.TryReadGuest(CopyDestination + 3 * sizeof(uint), wide));
        Assert.Equal(0x0000A000_0000A003UL, BitConverter.ToUInt64(wide));

        Assert.Equal((2L, 0L, 0L, 0L), host.CopyReadCounters);
        // Nothing was committed or waited for: the submission is still the one recording.
        Assert.Equal(serial, host.SubmissionSerial);
        Assert.True(host.CompletedSerial < serial);
    }

    [Fact]
    public void AWriteRecordedAfterThePendingCopyMakesTheReadWait()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.FinalGuestTablesEnabled = false;
        var serial = RecordPendingCopy(rig);

        host.NoteGuestMemoryWrite(CopyDestination + 0x10, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
        Assert.True(host.TryReadGuestWord(CopyDestination + 0x20, out var word));
        // Nothing ran on the GPU, so the waited read sees memory as it was.
        Assert.Equal(0x11111111u, word);
        Assert.Equal((0L, 0L, 1L, 0L), host.CopyReadCounters);
        Assert.True(host.CompletedSerial >= serial);
    }
    [Fact]
    public void ResidentCopyReadsUseFinalSourceBytesAndRejectPendingSourcesOrLaterOverwrites()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.FinalGuestTablesEnabled = false;
        RecordPendingCopy(rig);
        var bytes = new byte[16];
        Assert.True(host.TryReadResidentGuestBytes(CopyDestination, bytes, false));
        Assert.Equal(new uint[] { 0xA000, 0xA001, 0xA002, 0xA003 },
            Enumerable.Range(0, 4).Select(i => BitConverter.ToUInt32(bytes, i * 4)));
        rig.WriteBacking32(CopySource, 0xBEEF0001);
        Assert.True(host.TryReadResidentGuestBytes(CopyDestination, bytes, false));
        Assert.Equal(0xBEEF0001u, BitConverter.ToUInt32(bytes));
        host.NoteGuestMemoryWrite(CopySource, 4, MetalGpuWriteOrigin.StorageBuffer);
        Assert.False(host.TryReadResidentGuestBytes(CopyDestination, bytes, false));
        host.NoteGuestMemoryWrite(CopyDestination, 4, MetalGpuWriteOrigin.StorageBuffer);
        Assert.False(host.TryReadResidentGuestBytes(CopyDestination, bytes, false));
        host.FlushAndWaitForGpu();
        rig.WriteBacking32(CopyDestination, 0xCAFE0002);
        Assert.True(host.TryReadResidentGuestBytes(CopyDestination, bytes, false));
        Assert.Equal(0xCAFE0002u, BitConverter.ToUInt32(bytes));
    }

}
