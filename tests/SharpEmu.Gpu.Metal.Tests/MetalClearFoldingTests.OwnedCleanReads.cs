// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A clean read of a word a recorded GPU write still owns answers the value the draw sees, after that write: the worker
/// reads it through the write's copy or waits for it, a resolving thread reads it through the copy or abandons. Neither
/// refuses (a resource branch's predicate then kept both paths, and the untaken path's descriptor, read from memory holding
/// floats, fataled as an invalid image in Demon's Souls) nor answers the bytes from before the write.
/// </summary>
public sealed unsafe partial class MetalClearFoldingTests
{
    private const ulong PredicateWord = GuestBase + 0x3A000;

    [Fact]
    public void ACleanReadUnderAPendingWriteAbandonsTheResolutionAndTheWorkerReadsTheLandedValue()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        rig.WriteBacking32(PredicateWord, 0);
        host.NoteGuestMemoryWrite(PredicateWord, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);

        var concurrent = (IConcurrentResolveHost)host;
        concurrent.BeginConcurrentResolve();
        try
        {
            Assert.Throws<GpuFatal.AbandonedWorkException>(() => host.TryReadCleanGuestWord(PredicateWord, out _));
        }
        finally
        {
            concurrent.ReleaseAssumptions(concurrent.EndConcurrentResolve());
        }

        // The worker waits for the write and reads what it left (the backing, here: the write is only recorded).
        var serial = host.SubmissionSerial;
        rig.WriteBacking32(PredicateWord, 0x1234);
        Assert.True(host.TryReadCleanGuestWord(PredicateWord, out var word));
        Assert.Equal(0x1234u, word);
        Assert.True(host.CompletedSerial >= serial);
    }

    [Fact]
    public void ACleanReadOfAPendingCopyDestinationIsAnsweredFromItsSourceOnBothThreads()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        var serial = RecordPendingCopy(rig);
        var concurrent = (IConcurrentResolveHost)host;
        concurrent.BeginConcurrentResolve();
        object assumptions;
        try
        {
            Assert.True(host.TryReadCleanGuestWord(CopyDestination + 6 * sizeof(uint), out var copied));
            Assert.Equal(0xA002u, copied);
        }
        finally
        {
            assumptions = concurrent.EndConcurrentResolve();
        }

        // The destination's owned answer is dropped (the copy writes it either way); the source's answers stand.
        Assert.DoesNotContain(((MetalRenderHost.ConcurrentAssumptions)assumptions).Answers, answer => answer.Owned);
        Assert.True(concurrent.IsStillCurrent(assumptions));
        concurrent.ReleaseAssumptions(assumptions);

        Assert.True(host.TryReadCleanGuestWord(CopyDestination + 6 * sizeof(uint), out var word));
        Assert.Equal(0xA002u, word);
        // Answered from the source without a wait.
        Assert.True(host.CompletedSerial < serial);
    }

    // A clean read, then a GPU write over it recorded by the worker while the resolution is still open (an item ahead of
    // the draw), then the write completes: the answer is clean again at the take, but the bytes the resolution read are
    // the ones from before the write. The take refuses it while the write is pending and after it completed.
    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public void ACleanReadIsNotCurrentOnceAWriteRecordedAfterItCompletes(bool resident, bool verifies)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        rig.WriteBacking32(PredicateWord, 0x1111_1111);

        var concurrent = (IConcurrentResolveHost)host;
        concurrent.BeginConcurrentResolve();
        object assumptions;
        try
        {
            uint word;
            if (resident)
            {
                Span<byte> bytes = stackalloc byte[sizeof(uint)];
                Assert.True(host.TryReadResidentGuestBytes(PredicateWord, bytes, clean: true));
                word = BitConverter.ToUInt32(bytes);
            }
            else
            {
                Assert.True(host.TryReadCleanGuestWord(PredicateWord, out word));
            }

            Assert.Equal(0x1111_1111u, word);
            Assert.False(host.ConcurrentReadsSuperseded());
            host.NoteGuestMemoryWrite(PredicateWord, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
            if (verifies)
            {
                // A verify in this resolution finds its reads straddling the write, and latches the take's refusal.
                Assert.True(host.ConcurrentReadsSuperseded());
            }
        }
        finally
        {
            assumptions = concurrent.EndConcurrentResolve();
        }

        try
        {
            Assert.False(((MetalRenderHost.ConcurrentAssumptions)assumptions).ReliesOnOwned, "relies on owned");
            Assert.Equal(verifies, ((MetalRenderHost.ConcurrentAssumptions)assumptions).Superseded);
            Assert.False(concurrent.IsStillCurrent(assumptions), "current while the write is pending");
            Assert.False(concurrent.IsStillCurrentAcrossDispatches(assumptions), "current across dispatches while the write is pending");

            var serial = host.FlushCommandBuffer();
            var deadline = Environment.TickCount64 + 10_000;
            while (host.CompletedSerial < serial && Environment.TickCount64 < deadline)
            {
                Thread.Sleep(1);
            }

            Assert.True(host.CompletedSerial >= serial);
            Assert.False(concurrent.IsStillCurrentAcrossDispatches(assumptions), "current across dispatches after the write completed");
            Assert.False(concurrent.IsStillCurrent(assumptions), "current after the write completed");
        }
        finally
        {
            concurrent.ReleaseAssumptions(assumptions);
        }
    }
}
