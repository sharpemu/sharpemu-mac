// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// The counters of a recognised copy or fill kernel the host refuses: whether its result is
/// already in memory (would_elide_identical) or how much it would change (would_write, bytes,
/// pages), why the answer could not be taken, which kernel each fallback reason is for, and
/// which owner made the resolve thread abandon a read. With the switch off, nothing changes:
/// the kernel still runs as its dispatch and its copy is still armed for read-through.
/// </summary>
public sealed unsafe partial class MetalClearFoldingTests
{
    private const ulong KernelSource = GuestBase + 0x38000;
    private const ulong KernelDestination = GuestBase + 0x39000;
    private const ulong KernelPeriod = 4;
    private const ulong KernelWords = 16;

    private static void WriteKernelSource(Rig rig)
    {
        for (var word = 0UL; word < KernelPeriod; word++)
        {
            rig.WriteBacking32(KernelSource + word * sizeof(uint), 0xB000u + (uint)word);
        }
    }

    // The destination as the copy leaves it, so the kernel would change nothing.
    private static void WriteKernelResult(Rig rig)
    {
        for (var word = 0UL; word < KernelWords; word++)
        {
            rig.WriteBacking32(KernelDestination + word * sizeof(uint), 0xB000u + (uint)(word % KernelPeriod));
        }
    }

    [Fact]
    public void ARefusedCopyWhoseDestinationHoldsItsResultCountsAsIdenticalAndStillRuns()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        WriteKernelSource(rig);
        WriteKernelResult(rig);

        Assert.False(host.TryCopyWordsOnHost(KernelDestination, KernelSource, KernelPeriod, KernelWords));
        Assert.Equal((1L, 0L, 0L, 0L, 0L, 0L, 0L), host.HostKernelCopyOutcomes);
        // Still the dispatch: its storage binding records the copy for read-through.
        host.NoteGuestMemoryWrite(KernelDestination, KernelWords * sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
        Assert.Equal(1, host.PendingCopyCount);
        var churn = host.TakeKernelFillCounters();
        Assert.Contains("copy:no-image=1", churn);
        Assert.Contains(" host_kernel_copy=[fills=0,would_elide_identical=1,would_write=0,changed_bytes=0,changed_pages=0,", churn);
        Assert.Equal((0L, 0L, 0L, 0L, 0L, 0L, 0L), host.HostKernelCopyOutcomes);
    }

    [Fact]
    public void ARefusedCopyThatChangesWordsCountsTheBytesAndPagesItWouldWrite()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        WriteKernelSource(rig);
        WriteKernelResult(rig);
        // Two words differ, on one page.
        rig.WriteBacking32(KernelDestination + 5 * sizeof(uint), 0);
        rig.WriteBacking32(KernelDestination + 9 * sizeof(uint), 0);

        Assert.False(host.TryCopyWordsOnHost(KernelDestination, KernelSource, KernelPeriod, KernelWords));
        Assert.Equal((0L, 1L, 0L, 0L, 0L, 0L, 0L), host.HostKernelCopyOutcomes);
        Assert.Equal((8L, 1L), host.HostKernelCopyChanges);
    }

    [Fact]
    public void APendingWriteOverEitherSideIsCountedAsTheReasonNotCompared()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        WriteKernelSource(rig);
        WriteKernelResult(rig);

        host.NoteGuestMemoryWrite(KernelSource + 4, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
        Assert.False(host.TryCopyWordsOnHost(KernelDestination, KernelSource, KernelPeriod, KernelWords));
        host.ResetBindings();
        host.NoteGuestMemoryWrite(KernelDestination + 0x20, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
        Assert.False(host.TryCopyWordsOnHost(KernelDestination, KernelSource + 0x100, KernelPeriod, KernelWords));
        host.ResetBindings();
        // Too wide to compare at record time.
        Assert.False(host.TryCopyWordsOnHost(KernelDestination + 0x10000, KernelSource, KernelPeriod, 0x10001));

        Assert.Equal((0L, 0L, 1L, 1L, 0L, 1L, 0L), host.HostKernelCopyOutcomes);
    }

    [Fact]
    public void ARefusedFillIsComparedWithItsPatternAndItsReasonsNameTheFill()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        for (var word = 0UL; word < KernelWords; word++)
        {
            rig.WriteBacking32(KernelDestination + word * sizeof(uint), word % 2 == 0 ? 0x1u : 0x2u);
        }

        Assert.False(host.TryClearImageFromKernelFill(KernelDestination, KernelWords * sizeof(uint), [0x1u, 0x2u]));
        Assert.False(host.TryClearImageFromKernelFill(KernelDestination, KernelWords * sizeof(uint), [0x2u, 0x1u]));
        Assert.Equal((1L, 1L, 0L, 0L, 0L, 0L, 0L), host.HostKernelCopyOutcomes);
        Assert.Equal(((long)KernelWords * sizeof(uint), 1L), host.HostKernelCopyChanges);
        host.NoteKernelFillFallback("copy-shape");
        var churn = host.TakeKernelFillCounters();
        Assert.Contains("fill_fallback=[copy-shape=1,fill:no-image=2]", churn);
        Assert.Contains("fills=2,", churn);
    }

    [Fact]
    public void AnAbandonedResolutionIsLaidAtAPendingCopyOrAnOrdinaryWrite()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        WriteKernelSource(rig);
        Assert.False(host.TryCopyWordsOnHost(KernelDestination, KernelSource, KernelPeriod, KernelWords));
        host.NoteGuestMemoryWrite(KernelDestination, KernelWords * sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
        host.ResetBindings();
        host.NoteGuestMemoryWrite(KernelSource + 0x800, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);

        var concurrent = (IConcurrentResolveHost)host;
        concurrent.BeginConcurrentResolve();
        try
        {
            Assert.Throws<GpuFatal.AbandonedWorkException>(() => host.TryReadGuestWord(KernelDestination + 8, out _));
            Assert.Throws<GpuFatal.AbandonedWorkException>(() => host.TryReadGuestWord(KernelSource + 0x800, out _));
        }
        finally
        {
            concurrent.ReleaseAssumptions(concurrent.EndConcurrentResolve());
        }

        Assert.Equal("gpu_write=1,copy_dst=1,label=0,cp=0,image=0,code=0,queued=0,other=0", host.TakeAbandonCauses());
    }
}
