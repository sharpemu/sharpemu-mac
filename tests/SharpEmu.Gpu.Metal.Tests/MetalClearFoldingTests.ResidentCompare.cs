// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A hit compares its recorded ranges where they lie. The comparison agrees with reading them
/// back through TryReadResidentGuestBytes when it says equal, and says false (the caller then
/// reads) for every case it does not model.
/// </summary>
public sealed unsafe partial class MetalClearFoldingTests
{
    private const ulong CompareBase = GuestBase + 0x80000;

    // Ranges that straddle a 256-byte line (0x3F0..0x410) and a page (0xFF8..0x1008), a range in
    // the same cluster after a gap, and one far enough off to be a cluster of its own.
    private static readonly (ulong Address, int Length)[] CompareRanges =
    [
        (CompareBase + 0x3F0, 0x20), (CompareBase + 0x480, 8), (CompareBase + 0xFF8, 16), (CompareBase + 0x1010, 4),
        (CompareBase + 0x40000, 32),
    ];

    private static void RecordRanges(Rig rig, out ulong[] addresses, out int[] lengths, out int[] offsets, out byte[] recorded)
    {
        for (var word = 0UL; word < 0x2000 / 4; word++)
        {
            rig.WriteBacking32(CompareBase + word * 4, 0x77000000u + (uint)word);
        }

        for (var word = 0UL; word < 8; word++)
        {
            rig.WriteBacking32(CompareBase + 0x40000 + word * 4, 0x78000000u + (uint)word);
        }

        addresses = [.. CompareRanges.Select(range => range.Address)];
        lengths = [.. CompareRanges.Select(range => range.Length)];
        offsets = new int[CompareRanges.Length];
        recorded = new byte[CompareRanges.Sum(range => range.Length)];
        var cursor = 0;
        for (var index = 0; index < CompareRanges.Length; index++)
        {
            offsets[index] = cursor;
            Assert.True(rig.Host.TryReadResidentGuestBytes(addresses[index], recorded.AsSpan(cursor, lengths[index]), clean: false));
            cursor += lengths[index];
        }
    }

    [Fact]
    public void RangesOverLinesPagesAndClustersCompareEqualInPlaceAndUnequalWhereAByteDiffers()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        RecordRanges(rig, out var addresses, out var lengths, out var offsets, out var recorded);
        rig.AliasAccess = true;
        Assert.True(rig.Host.TryCompareResidentGuestBytes(addresses, lengths, offsets, recorded));

        // A recorded byte that memory no longer holds: each range, its first and its last byte.
        for (var index = 0; index < addresses.Length; index++)
        {
            foreach (var position in new[] { 0, lengths[index] - 1 })
            {
                var changed = (byte[])recorded.Clone();
                changed[offsets[index] + position] ^= 0x5A;
                Assert.False(rig.Host.TryCompareResidentGuestBytes(addresses, lengths, offsets, changed));
            }
        }

        // A guest write between recording and comparing, over the page boundary.
        rig.WriteBacking32(CompareBase + 0xFFC, 0xDEADBEEF);
        Assert.False(rig.Host.TryCompareResidentGuestBytes(addresses, lengths, offsets, recorded));
        var reread = new byte[16];
        Assert.True(rig.Host.TryReadResidentGuestBytes(CompareBase + 0xFF8, reread, clean: false));
        Assert.False(reread.AsSpan().SequenceEqual(recorded.AsSpan(offsets[2], 16)));
    }

    [Fact]
    public void LabelsOnOtherPagesDoNotPreventComparingUnchangedRanges()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        RecordRanges(rig, out var addresses, out var lengths, out var offsets, out var recorded);
        rig.AliasAccess = true;
        var host = rig.Host;
        host.LabelOverlayEnabled = true;
        host.CompareLabelRangesEnabled = true;
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, CompareBase + 0x50000, 0x1234));
        Assert.Equal(1, host.PendingLabelOverlayEntries);
        Assert.True(host.TryCompareResidentGuestBytes(addresses, lengths, offsets, recorded));
        Assert.Equal(1, host.PendingLabelOverlayEntries); // The comparison retires no labels.

        host.CompareLabelRangesEnabled = false;
        Assert.False(host.TryCompareResidentGuestBytes(addresses, lengths, offsets, recorded));
        host.CompareLabelRangesEnabled = true;
        rig.WriteBacking32(addresses[1], 0xDEADBEEF);
        Assert.False(host.TryCompareResidentGuestBytes(addresses, lengths, offsets, recorded));
    }

    [Theory]
    [InlineData(0x480UL)] // Recorded bytes.
    [InlineData(0x4A0UL)] // A gap in the same cluster/page.
    public void ALabelThatMayOverlapStillUsesTheReader(ulong offset)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        RecordRanges(rig, out var addresses, out var lengths, out var offsets, out var recorded);
        rig.AliasAccess = true;
        var host = rig.Host;
        host.LabelOverlayEnabled = true;
        host.CompareLabelRangesEnabled = true;
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, CompareBase + offset, 0x1234));
        Assert.Equal(1, host.PendingLabelOverlayEntries);
        Assert.False(host.TryCompareResidentGuestBytes(addresses, lengths, offsets, recorded));
        Assert.Equal(1, host.PendingLabelOverlayEntries);
        if (offset == 0x480)
        {
            var current = new byte[8];
            Assert.True(host.TryReadResidentGuestBytes(addresses[1], current, clean: false));
            Assert.Equal(0x1234u, BitConverter.ToUInt32(current));
        }
    }

    [Fact]
    public void ACompareTheBackingWillNotAdmitOrAViewEdgeCrossesLeavesTheRangesToBeRead()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        RecordRanges(rig, out var addresses, out var lengths, out var offsets, out var recorded);

        // No alias access: the caller reads.
        Assert.False(rig.Host.TryCompareResidentGuestBytes(addresses, lengths, offsets, recorded));
        rig.AliasAccess = true;
        Assert.True(rig.Host.TryCompareResidentGuestBytes(addresses, lengths, offsets, recorded));

        // A page boundary that is also the edge of one view: the range over it is not one alias.
        rig.ViewBoundary = CompareBase + 0x1000;
        Assert.False(rig.Host.TryCompareResidentGuestBytes(addresses, lengths, offsets, recorded));
        var crossing = new byte[16];
        Assert.True(rig.Host.TryReadResidentGuestBytes(CompareBase + 0xFF8, crossing, clean: false));
        Assert.True(crossing.AsSpan().SequenceEqual(recorded.AsSpan(offsets[2], 16)));

        // Memory that is not there: neither compares nor reads.
        rig.ViewBoundary = 0;
        var end = GuestBase + Backing.Size;
        Assert.False(rig.Host.TryCompareResidentGuestBytes([end - 8], [16], [0], new byte[16]));
        Assert.False(rig.Host.TryReadResidentGuestBytes(end - 8, new byte[16], clean: false));

        // The last bytes of memory are compared like any others.
        var tail = new byte[16];
        Assert.True(rig.Host.TryReadResidentGuestBytes(end - 16, tail, clean: false));
        Assert.True(rig.Host.TryCompareResidentGuestBytes([end - 16], [16], [0], tail));
        tail[15] ^= 1;
        Assert.False(rig.Host.TryCompareResidentGuestBytes([end - 16], [16], [0], tail));
    }

    [Fact]
    public void OwedCommandProcessorBytesOverARangeLeaveTheRangesToBeRead()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        RecordRanges(rig, out var addresses, out var lengths, out var offsets, out var recorded);
        rig.AliasAccess = true;
        using (rig.Host.BeginPreparation())
        {
            var target = rig.Target(TargetAddress);
            var attachment = ((IRenderHost)rig.Host).AcquireColorAttachment(in target);
            rig.Host.BeginRendering(Rig.PassOver(attachment.View));
            // Owed bytes in the gap between two ranges of a cluster, not in a range: the cluster
            // is refused all the same, and the caller reads each range.
            rig.Host.RecordCommandProcessorWrite(CompareBase + 0x4A0, BitConverter.GetBytes(0xC0FFEE11u), "WRITE_DATA");
            Assert.Equal(1, rig.Host.DeferredLabelCopyCount);
            Assert.False(rig.Host.TryCompareResidentGuestBytes(addresses, lengths, offsets, recorded));
            // Owed bytes in a range read as written, so the read-back differs from the recorded bytes.
            rig.Host.RecordCommandProcessorWrite(CompareBase + 0x480, BitConverter.GetBytes(0xC0FFEE22u), "WRITE_DATA");
            Assert.False(rig.Host.TryCompareResidentGuestBytes(addresses, lengths, offsets, recorded));
            var read = new byte[8];
            Assert.True(rig.Host.TryReadResidentGuestBytes(CompareBase + 0x480, read, clean: false));
            Assert.Equal(0xC0FFEE22u, BitConverter.ToUInt32(read));
            rig.Host.EndRendering();
        }
    }

    [Fact]
    public void AGpuModifiedImageOverARangeNeverComparesEqualWhereTheReadRefuses()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.AliasAccess = true;
        var texture = rig.TextureOf(TargetAddress);
        rig.Clear(texture);
        var bytes = new byte[16];
        var read = rig.Host.TryReadResidentGuestBytes(TargetAddress, bytes, clean: false);
        var compared = rig.Host.TryCompareResidentGuestBytes([TargetAddress], [16], [0], bytes);
        Assert.True(read || !compared);
        output.WriteLine($"gpu-modified image: read={read} compared={compared}");
    }
}
