// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A command-processor write recorded inside an open render pass is copied at the next
/// encoder boundary. Until then the bytes are owed, and what runs at parse time - resource
/// tables, operands, packets - reads them as written, the way the hardware's command
/// processor sees its own writes: without ending the pass, and without waiting for a
/// submission the worker has not committed (which would wait on itself).
/// </summary>
public sealed unsafe partial class MetalClearFoldingTests
{
    private const ulong TableAddress = GuestBase + 0x20000;

    [Fact]
    public void AWriteDataInsideAnOpenPassIsReadAsWrittenWithoutEndingThePassOrWaiting()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        rig.WriteBacking32(TableAddress, 0x0BAD0BAD);
        using (host.BeginPreparation())
        {
            var target = rig.Target(TargetAddress);
            var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
            host.BeginRendering(Rig.PassOver(attachment.View));
            Assert.True(host.RenderEncoderOpen);
            var serial = host.SubmissionSerial;

            host.RecordCommandProcessorWrite(TableAddress, BitConverter.GetBytes(0xC0FFEE11u), "WRITE_DATA");
            Assert.Equal(1, host.DeferredLabelCopyCount);
            Assert.Equal(0x0BAD0BADu, rig.ReadBacking32(TableAddress));

            Span<byte> read = stackalloc byte[sizeof(uint)];
            Assert.True(host.TryReadGuest(TableAddress, read));
            Assert.Equal(0xC0FFEE11u, BitConverter.ToUInt32(read));
            Assert.True(host.TryReadGuestWord(TableAddress, out var word));
            Assert.Equal(0xC0FFEE11u, word);
            Assert.True(host.TryReadResidentGuestBytes(TableAddress, read, clean: true));
            Assert.Equal(0xC0FFEE11u, BitConverter.ToUInt32(read));
            Assert.True(host.TryReadCleanGuestWord(TableAddress, out var clean));
            Assert.Equal(0xC0FFEE11u, clean);
            Assert.True(host.TryReadGuestWaitOperand(TableAddress, read, commitPendingWrite: true));
            Assert.Equal(0xC0FFEE11u, BitConverter.ToUInt32(read));

            // A read reaching past the owed bytes takes the rest from memory.
            Span<byte> wide = stackalloc byte[2 * sizeof(uint)];
            rig.WriteBacking32(TableAddress + 4, 0x22222222);
            Assert.True(host.TryReadResidentGuestBytes(TableAddress, wide, clean: true));
            Assert.Equal(0x22222222_C0FFEE11UL, BitConverter.ToUInt64(wide));

            // Still the same pass, still the same unsubmitted buffer, the copy still owed.
            Assert.True(host.RenderEncoderOpen);
            Assert.Equal(serial, host.SubmissionSerial);
            Assert.Equal(1, host.DeferredLabelCopyCount);
            Assert.Equal(0x0BAD0BADu, rig.ReadBacking32(TableAddress));
            host.EndRendering();
        }

        host.FlushAndWaitForGpu();
        Assert.Equal(0xC0FFEE11u, rig.ReadBacking32(TableAddress));

        // Landed: memory answers again, including for what the guest writes over it.
        rig.WriteBacking32(TableAddress, 0x5EC0DD00);
        Span<byte> after = stackalloc byte[sizeof(uint)];
        Assert.True(host.TryReadGuest(TableAddress, after));
        Assert.Equal(0x5EC0DD00u, BitConverter.ToUInt32(after));
    }

    // A command-processor write recorded outside a pass (a posted word copy, in Demon's Souls the 3 KB table its draws
    // read) is a pending GPU write over its bytes. A draw resolved ahead of the worker reads them as written, as the worker
    // does, instead of abandoning; the worker's clean read takes them without waiting for the GPU. A GPU write recorded
    // after the resolution began over those bytes refuses it at the take.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void OwedBytesUnderAPendingWriteAreReadAsWrittenAheadOfTheWorkerAndOnIt(bool overlayReads)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.ConcurrentOverlayReads = overlayReads;
        rig.WriteBacking32(TableAddress, 0x0BAD0BAD);
        rig.WriteBacking32(TableAddress + 4, 0x0BAD0BAD);
        rig.WriteBacking32(TableAddress + 8, 0x22222222);
        host.RecordCommandProcessorWrite(TableAddress, BitConverter.GetBytes(0xC0FFEE22_C0FFEE11UL), "word copy");
        var serial = host.SubmissionSerial;
        Assert.Equal(0x0BAD0BADu, rig.ReadBacking32(TableAddress));

        var concurrent = (IConcurrentResolveHost)host;
        concurrent.BeginConcurrentResolve();
        object assumptions;
        try
        {
            if (!overlayReads)
            {
                // As before: the word's owner is the recorded write, which no pending copy explains.
                Assert.Throws<SharpEmu.Libs.Gpu.Scheduling.GpuFatal.AbandonedWorkException>(() => host.TryReadGuestWord(TableAddress, out _));
                Assert.Throws<SharpEmu.Libs.Gpu.Scheduling.GpuFatal.AbandonedWorkException>(() => host.TryReadCleanGuestWord(TableAddress, out _));
                return;
            }

            Assert.True(host.TryReadGuestWord(TableAddress, out var word));
            Assert.Equal(0xC0FFEE11u, word);
            // The rest of the line: the write's other word as written (from the line read with the first), and a word
            // no write owns as memory holds it.
            Assert.True(host.TryReadGuestWord(TableAddress + 4, out var second));
            Assert.Equal(0xC0FFEE22u, second);
            Assert.True(host.TryReadGuestWord(TableAddress + 8, out var unowned));
            Assert.Equal(0x22222222u, unowned);
            // A proof's compare of the range sees the bytes as written too, not as memory holds them.
            ulong[] addresses = [TableAddress];
            int[] lengths = [8];
            Assert.True(host.TryCompareGuestWords(addresses, lengths, BitConverter.GetBytes(0xC0FFEE22_C0FFEE11UL)));
            Assert.False(host.TryCompareGuestWords(addresses, lengths, BitConverter.GetBytes(0x0BAD0BAD_0BAD0BADUL)));
            Assert.True(host.TryReadCleanGuestWord(TableAddress, out var clean));
            Assert.Equal(0xC0FFEE11u, clean);
        }
        finally
        {
            assumptions = concurrent.EndConcurrentResolve();
            if (!overlayReads)
            {
                concurrent.ReleaseAssumptions(assumptions);
            }
        }

        try
        {
            // The owned answers are dropped, as for a copy read; the take stands while nothing is written over the word.
            Assert.DoesNotContain(((MetalRenderHost.ConcurrentAssumptions)assumptions).Answers, answer => answer.Owned);
            Assert.True(concurrent.IsStillCurrent(assumptions));

            // The worker reads the same bytes, without waiting for the write to land.
            Assert.True(host.TryReadCleanGuestWord(TableAddress, out var worker));
            Assert.Equal(0xC0FFEE11u, worker);
            Assert.True(host.CompletedSerial < serial, "the worker's clean read waited for the GPU");

            // A store recorded after the resolution began over the word is the draw's value instead: refused.
            host.NoteGuestMemoryWrite(TableAddress, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
            Assert.False(concurrent.IsStillCurrent(assumptions));
        }
        finally
        {
            concurrent.ReleaseAssumptions(assumptions);
        }

        host.FlushAndWaitForGpu();
        Assert.Equal(0xC0FFEE11u, rig.ReadBacking32(TableAddress));
        Assert.Equal(0xC0FFEE22u, rig.ReadBacking32(TableAddress + 4));
    }

    [Fact]
    public void AGpuWriteRecordedAfterTheOwedBytesIsWhatAReadWaitsFor()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.RecordCommandProcessorWrite(TableAddress, BitConverter.GetBytes(0x11111111u), "WRITE_DATA");

        // A fill too large to stage is a GPU fill recorded after the write: program order
        // says the read sees the fill, so the owed bytes no longer answer for it.
        host.FillBufferInOrder(TableAddress, 0x2000, 0x33333333);
        Span<byte> read = stackalloc byte[sizeof(uint)];
        Assert.True(host.TryReadGuest(TableAddress, read));
        Assert.Equal(0x33333333u, BitConverter.ToUInt32(read));
    }
}
