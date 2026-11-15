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
}
