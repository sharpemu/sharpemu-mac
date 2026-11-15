// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A clean read a recorded GPU write still owns is refused on every thread (a resource branch's predicate then keeps both
/// paths), and a resolution that relied on that refusal is not current once the write has completed: the worker would
/// answer the read now and prune a path, so the take refuses the resolution even while no ownership version moved.
/// </summary>
public sealed unsafe partial class MetalClearFoldingTests
{
    private const ulong PredicateWord = GuestBase + 0x3A000;

    [Fact]
    public void ARefusedCleanReadIsNotCurrentOnceTheWriteThatOwnedItCompletes()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        rig.WriteBacking32(PredicateWord, 0);
        host.NoteGuestMemoryWrite(PredicateWord, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);

        // The worker refuses it as the resolving thread does.
        Assert.False(host.TryReadCleanGuestWord(PredicateWord, out _));

        var concurrent = (IConcurrentResolveHost)host;
        concurrent.BeginConcurrentResolve();
        object assumptions;
        try
        {
            Assert.False(host.TryReadCleanGuestWord(PredicateWord, out _));
        }
        finally
        {
            assumptions = concurrent.EndConcurrentResolve();
        }

        try
        {
            Assert.True(((MetalRenderHost.ConcurrentAssumptions)assumptions).ReliesOnOwned);
            Assert.True(concurrent.IsStillCurrent(assumptions));
            Assert.True(concurrent.IsStillCurrentAcrossDispatches(assumptions));

            // The GPU completes the write; nothing on the worker retires its record yet, so no ownership version moves.
            var serial = host.FlushCommandBuffer();
            var deadline = Environment.TickCount64 + 10_000;
            while (host.CompletedSerial < serial && Environment.TickCount64 < deadline)
            {
                Thread.Sleep(1);
            }

            Assert.True(host.CompletedSerial >= serial);
            Assert.False(concurrent.IsStillCurrent(assumptions));
            Assert.False(concurrent.IsStillCurrentAcrossDispatches(assumptions));
            Assert.True(host.TryReadCleanGuestWord(PredicateWord, out var word));
            Assert.Equal(0u, word);
        }
        finally
        {
            concurrent.ReleaseAssumptions(assumptions);
        }
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
