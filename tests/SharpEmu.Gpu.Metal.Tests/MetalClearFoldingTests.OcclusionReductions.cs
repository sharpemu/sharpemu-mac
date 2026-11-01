// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A pass that dumped occlusion counts queues its prefix sum instead of encoding an encoder of its
/// own; the queued sums are encoded, in pass order, before the label copies that read them land.
/// </summary>
public sealed unsafe partial class MetalClearFoldingTests
{
    private const ulong OcclusionQueryA = GuestBase + 0x6C000;
    private const ulong OcclusionQueryB = GuestBase + 0x6C100;

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void DumpsOfTwoPassesLandTheirSnapshotsWhetherTheReductionsAreDeferredOrNot(bool deferReductions)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.NativeOcclusionEnabled = true;
        host.DeferOcclusionReductions = deferReductions;
        rig.WriteBacking32(OcclusionQueryA + 4, 0xDEADBEEF);
        rig.WriteBacking32(OcclusionQueryB + 4, 0xDEADBEEF);

        foreach (var query in new[] { OcclusionQueryA, OcclusionQueryB })
        {
            using (host.BeginPreparation())
            {
                var target = rig.Target(TargetAddress);
                var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
                host.BeginRendering(Rig.PassOver(attachment.View));
                Assert.True(host.DumpOcclusionCounter(0, query, 1, 16));
                host.EndRendering();
            }
        }

        // Both passes ended; with deferral their sums are still queued, without it they are encoded.
        Assert.Equal(deferReductions ? 0 : 2, host.NativeOcclusionTotals.Reductions);

        host.FlushAndWaitForGpu();
        Assert.Equal(2, host.NativeOcclusionTotals.Reductions);
        host.BeginGuestWordReads();
        Assert.True(host.TryReadGuestWord(OcclusionQueryA + 4, out var firstHigh));
        Assert.True(host.TryReadGuestWord(OcclusionQueryB + 4, out var secondHigh));
        host.EndGuestWordReads();
        // Bit 63 set, no sample counted: the snapshot of the empty passes, ready.
        Assert.Equal(0x80000000u, firstHigh);
        Assert.Equal(0x80000000u, secondHigh);
        Assert.True(host.TryReadOcclusionCounter(0, out var total));
        Assert.Equal(0UL, total);
    }
}
