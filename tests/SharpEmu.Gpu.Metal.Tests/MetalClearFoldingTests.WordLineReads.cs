// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// Inside a materialization's word reads, a dword is served from a 256-byte line read once
/// when nothing pending covers any of the line; a line with owed command-processor bytes or
/// a pending GPU write leaves its words to the ordinary read, which sees the owed bytes and
/// waits where it always did.
/// </summary>
public sealed unsafe partial class MetalClearFoldingTests
{
    private const ulong LineTable = GuestBase + 0x50000;
    private const ulong PendingLine = GuestBase + 0x60000;

    [Fact]
    public void WordReadsOfACleanLineAreServedFromOneCheckedReadUntilAWriteIsRecorded()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.FinalGuestTablesEnabled = false;
        for (var word = 0UL; word < 16; word++)
        {
            rig.WriteBacking32(LineTable + word * sizeof(uint), 0xD000u + (uint)word);
        }

        // A recorded write elsewhere puts the table inside the write bounds, so the
        // ordinary read would run its full pending-write test for every word.
        host.NoteGuestMemoryWrite(GuestBase + 0x70000, 0x100, MetalGpuWriteOrigin.StorageBuffer);
        var before = host.WordLineCounters;
        host.BeginGuestWordReads();
        for (var word = 0UL; word < 16; word++)
        {
            Assert.True(host.TryReadGuestWord(LineTable + word * sizeof(uint), out var value));
            Assert.Equal(0xD000u + (uint)word, value);
        }

        Assert.Equal((16L, 16L, 1L, 0L), Delta(host.WordLineCounters, before));

        // A write recorded anywhere drops the line: the next word is checked and read again.
        rig.WriteBacking32(LineTable, 0xD0D0D0D0);
        host.NoteGuestMemoryWrite(GuestBase + 0x71000, 0x100, MetalGpuWriteOrigin.StorageBuffer);
        Assert.True(host.TryReadGuestWord(LineTable, out var reread));
        Assert.Equal(0xD0D0D0D0u, reread);
        Assert.Equal((17L, 17L, 2L, 0L), Delta(host.WordLineCounters, before));
        host.EndGuestWordReads();

        // Outside the bracket every word is read afresh.
        rig.WriteBacking32(LineTable + 4, 0xE0E0E0E0);
        Assert.True(host.TryReadGuestWord(LineTable + 4, out var fresh));
        Assert.Equal(0xE0E0E0E0u, fresh);
        Assert.Equal((18L, 17L, 2L, 0L), Delta(host.WordLineCounters, before));
    }

    [Fact]
    public void ALineWithOwedOrPendingBytesLeavesItsWordsToTheOrdinaryRead()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.FinalGuestTablesEnabled = false;
        rig.WriteBacking32(LineTable, 0x0BAD0BAD);
        rig.WriteBacking32(LineTable + 0x10, 0x12345678);
        rig.WriteBacking32(PendingLine + 8, 0x87654321);
        var before = host.WordLineCounters;
        ulong serial;
        using (host.BeginPreparation())
        {
            var target = rig.Target(TargetAddress);
            var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
            host.BeginRendering(Rig.PassOver(attachment.View));
            serial = host.SubmissionSerial;
            host.RecordCommandProcessorWrite(LineTable, BitConverter.GetBytes(0xC0FFEE11u), "WRITE_DATA");
            Assert.Equal(1, host.DeferredLabelCopyCount);
            host.NoteGuestMemoryWrite(PendingLine, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);

            host.BeginGuestWordReads();
            // The owed word reads as written; its neighbour in the line reads memory.
            Assert.True(host.TryReadGuestWord(LineTable, out var owed));
            Assert.Equal(0xC0FFEE11u, owed);
            Assert.True(host.TryReadGuestWord(LineTable + 0x10, out var neighbour));
            Assert.Equal(0x12345678u, neighbour);
            // A word beside a pending store in the same line is read without waiting.
            Assert.True(host.TryReadGuestWord(PendingLine + 8, out var beside));
            Assert.Equal(0x87654321u, beside);
            Assert.Equal((3L, 0L, 2L, 3L), Delta(host.WordLineCounters, before));
            Assert.Equal(serial, host.SubmissionSerial);
            Assert.True(host.RenderEncoderOpen);
            host.EndGuestWordReads();
            host.EndRendering();
        }

        // The pending store itself still waits for its submission.
        host.BeginGuestWordReads();
        Assert.True(host.TryReadGuestWord(PendingLine, out _));
        host.EndGuestWordReads();
        Assert.True(host.CompletedSerial >= serial);
    }

    private static bool WideLinesDisabled => Environment.GetEnvironmentVariable("SHARPEMU_RESOURCE_WIDE_LINES") == "0";

    [Fact]
    public void AWideLineServesTheWordsOf256BytesFromOneCheckedRead()
    {
        if (!RequireDevice() || WideLinesDisabled) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.FinalGuestTablesEnabled = false;
        for (var word = 0UL; word < 64; word++)
        {
            rig.WriteBacking32(LineTable + word * sizeof(uint), 0xC000u + (uint)word);
        }

        var before = host.WordLineCounters;
        host.BeginGuestWordReads();
        for (var word = 0UL; word < 64; word++)
        {
            Assert.True(host.TryReadGuestWord(LineTable + word * sizeof(uint), out var value));
            Assert.Equal(0xC000u + (uint)word, value);
        }

        // The next line is a second read.
        Assert.True(host.TryReadGuestWord(LineTable + 256, out _));
        host.EndGuestWordReads();
        Assert.Equal((65L, 65L, 2L, 0L), Delta(host.WordLineCounters, before));
    }

    [Fact]
    public void APendingWordAnywhereInAWideLineRefusesTheLineAndTheWordsReadAsBefore()
    {
        if (!RequireDevice() || WideLinesDisabled) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.FinalGuestTablesEnabled = false;
        rig.WriteBacking32(PendingLine + 8, 0x87654321);
        rig.WriteBacking32(PendingLine + 0xF0, 0x0BADF00D);
        var before = host.WordLineCounters;
        ulong serial;
        using (host.BeginPreparation())
        {
            var target = rig.Target(TargetAddress);
            var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
            host.BeginRendering(Rig.PassOver(attachment.View));
            serial = host.SubmissionSerial;
            // 0xF0 is in the line of +8 now, and beyond the 64 bytes it used to be read in.
            host.NoteGuestMemoryWrite(PendingLine + 0xF0, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
            host.BeginGuestWordReads();
            Assert.True(host.TryReadGuestWord(PendingLine + 8, out var beside));
            Assert.Equal(0x87654321u, beside);
            Assert.Equal((1L, 0L, 1L, 1L), Delta(host.WordLineCounters, before));
            Assert.Equal(serial, host.SubmissionSerial);
            Assert.True(host.RenderEncoderOpen);
            host.EndGuestWordReads();
            host.EndRendering();
        }
    }

    [Fact]
    public void WordsReadThroughLinesEqualWordsReadDirectlyAcrossOwedBytesAndTheEndOfMemory()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.FinalGuestTablesEnabled = false;
        const ulong region = LineTable;
        for (var word = 0UL; word < 192; word++)
        {
            rig.WriteBacking32(region + word * sizeof(uint), 0xAB000000u + (uint)word * 0x101);
        }

        // The last 16 bytes of the backing, and the 16 after it that are not memory.
        var end = GuestBase + Backing.Size;
        for (var word = 1UL; word <= 4; word++)
        {
            rig.WriteBacking32(end - word * sizeof(uint), 0xEE000000u + (uint)word);
        }

        using (host.BeginPreparation())
        {
            var target = rig.Target(TargetAddress);
            var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
            host.BeginRendering(Rig.PassOver(attachment.View));
            // Owed bytes in the middle of the region: the lines over them are refused.
            host.RecordCommandProcessorWrite(region + 0x1D4, BitConverter.GetBytes(0xC0FFEE11u), "WRITE_DATA");
            Assert.Equal(1, host.DeferredLabelCopyCount);

            var addresses = new List<ulong>();
            for (var word = 0UL; word < 192; word++) addresses.Add(region + word * sizeof(uint));
            for (var word = 0UL; word < 8; word++) addresses.Add(end - 16 + word * sizeof(uint));
            addresses.Add(region + 2);
            addresses.Add(region + 254);
            foreach (var wordLines in new[] { false, true })
            {
                if (wordLines) host.BeginGuestWordReads();
                foreach (var address in addresses)
                {
                    Span<byte> direct = stackalloc byte[sizeof(uint)];
                    var read = host.TryReadGuest(address, direct);
                    var viaWord = host.TryReadGuestWord(address, out var word);
                    Assert.Equal(read, viaWord);
                    if (read) Assert.Equal(BitConverter.ToUInt32(direct), word);
                }
                if (wordLines) host.EndGuestWordReads();
            }

            host.EndRendering();
        }
    }

    private static (long, long, long, long) Delta(
        (long Reads, long Hits, long Fills, long Refused) after,
        (long Reads, long Hits, long Fills, long Refused) before) =>
        (after.Reads - before.Reads, after.Hits - before.Hits, after.Fills - before.Fills, after.Refused - before.Refused);
}
