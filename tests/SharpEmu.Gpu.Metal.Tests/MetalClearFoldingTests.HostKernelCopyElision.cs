// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// SHARPEMU_METAL_V2_HOST_KERNEL_COPY=identical: a refused copy or fill kernel whose destination
/// already holds its result is consumed and writes nothing, so nothing records a GPU write there
/// and a later read does not wait. Any differing word, a source the GPU still writes, or a
/// destination an image owns leaves it the dispatch it was. Under verify the kernel still runs,
/// and once it completes the destination is checked against what consuming it would have left.
/// </summary>
public sealed unsafe partial class MetalClearFoldingTests
{
    private const ulong KernelBytes = KernelWords * sizeof(uint);

    [Fact]
    public void AnIdenticalCopyIsConsumedWithoutAWriteRecordAndALaterReadDoesNotWait()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.HostKernelCopyMode = MetalHostKernelCopyMode.Identical;
        WriteKernelSource(rig);
        WriteKernelResult(rig);
        var serial = host.SubmissionSerial;
        var waits = host.HostGpuWaits;

        Assert.True(host.TryCopyWordsOnHost(KernelDestination, KernelSource, KernelPeriod, KernelWords));
        Assert.Equal(1, host.KernelCopiesElided);
        Assert.Equal(0, host.PendingCopyCount);
        Assert.False(((IMetalImageCacheOwner)host).HasPendingGpuWrite(KernelDestination, KernelBytes));
        // What the executor does after a consumed dispatch; nothing was armed to record.
        host.ResetBindings();
        Assert.True(host.TryReadGuestWord(KernelDestination + 5 * sizeof(uint), out var word));
        Assert.Equal(0xB001u, word);
        Assert.Equal(waits, host.HostGpuWaits);
        Assert.Equal(serial, host.SubmissionSerial);
        Assert.Equal((0L, 0L), host.HostKernelCopyChanges);
    }

    [Fact]
    public void ACopyWithOneDifferingWordRunsAsItsDispatch()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.HostKernelCopyMode = MetalHostKernelCopyMode.Identical;
        WriteKernelSource(rig);
        WriteKernelResult(rig);
        rig.WriteBacking32(KernelDestination + 15 * sizeof(uint), 0);

        Assert.False(host.TryCopyWordsOnHost(KernelDestination, KernelSource, KernelPeriod, KernelWords));
        host.NoteGuestMemoryWrite(KernelDestination, KernelBytes, MetalGpuWriteOrigin.StorageBuffer);
        Assert.Equal(1, host.PendingCopyCount);
        Assert.Equal(0, host.KernelCopiesElided);
        Assert.Equal((0L, 1L, 0L, 0L, 0L, 0L, 0L, 0L), host.HostKernelCopyOutcomes);
    }

    [Fact]
    public void ACopyWhoseSourceTheGpuStillWritesRunsAsItsDispatch()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.HostKernelCopyMode = MetalHostKernelCopyMode.Identical;
        WriteKernelSource(rig);
        WriteKernelResult(rig);
        // An earlier dispatch stores into the source: what the copy reads is not in memory yet.
        host.NoteGuestMemoryWrite(KernelSource, KernelPeriod * sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);

        Assert.False(host.TryCopyWordsOnHost(KernelDestination, KernelSource, KernelPeriod, KernelWords));
        Assert.Equal(0, host.KernelCopiesElided);
        Assert.Equal((0L, 0L, 1L, 0L, 0L, 0L, 0L, 0L), host.HostKernelCopyOutcomes);
    }

    [Fact]
    public void ACopyIntoBytesAnImageOwnsRunsAsItsDispatch()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.HostKernelCopyMode = MetalHostKernelCopyMode.Identical;
        _ = rig.TextureOf(TargetAddress);
        WriteKernelSource(rig);
        // Part of the image, so no image path takes it; its bytes already hold the result.
        var destination = TargetAddress + 0x100;
        for (var word = 0UL; word < KernelWords; word++)
        {
            rig.WriteBacking32(destination + word * sizeof(uint), 0xB000u + (uint)(word % KernelPeriod));
        }

        Assert.False(host.TryCopyWordsOnHost(destination, KernelSource, KernelPeriod, KernelWords));
        Assert.Equal(0, host.KernelCopiesElided);
        Assert.Equal((0L, 0L, 0L, 0L, 1L, 0L, 0L, 0L), host.HostKernelCopyOutcomes);
    }

    [Fact]
    public void AnIdenticalFillIsConsumedAndADifferentOneRuns()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.HostKernelCopyMode = MetalHostKernelCopyMode.Identical;
        for (var word = 0UL; word < KernelWords; word++)
        {
            rig.WriteBacking32(KernelDestination + word * sizeof(uint), 0x7u);
        }

        Assert.True(host.TryClearImageFromKernelFill(KernelDestination, KernelBytes, [0x7u]));
        Assert.False(host.TryClearImageFromKernelFill(KernelDestination, KernelBytes, [0x8u]));
        Assert.Equal(1, host.KernelCopiesElided);
        Assert.False(((IMetalImageCacheOwner)host).HasPendingGpuWrite(KernelDestination, KernelBytes));
    }

    [Fact]
    public void VerifyRunsTheKernelAndChecksBothSidesOnceItCompletes()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.HostKernelCopyMode = MetalHostKernelCopyMode.Verify;
        WriteKernelSource(rig);
        WriteKernelResult(rig);

        // Left as found: both sides hold.
        RecordVerifiedCopy(host);
        host.FlushAndWaitForGpu();
        Assert.Equal(0, host.PendingKernelCopyChecks);
        Assert.Equal((1L, 0L, 0L, 0L), host.KernelCopyCheckCounters);
        Assert.Equal((1L, 0L, 0L), host.KernelCopyDestinationCheckCounters);

        // The destination changed after the check and before the GPU reached the kernel (a write
        // the guards missed): the blit ahead of the kernel sees it, and so does the source side
        // here, since no kernel ran to store over it.
        RecordVerifiedCopy(host);
        rig.WriteBacking32(KernelDestination + 3 * sizeof(uint), 0xDEADu);
        host.FlushAndWaitForGpu();
        Assert.Equal((2L, 1L, 0L, 0L), host.KernelCopyCheckCounters);
        Assert.Equal((2L, 1L, 0L), host.KernelCopyDestinationCheckCounters);

        // A later recorded store over the destination supersedes the source side only.
        rig.WriteBacking32(KernelDestination + 3 * sizeof(uint), 0xB003u);
        RecordVerifiedCopy(host);
        host.NoteGuestMemoryWrite(KernelDestination + 8, sizeof(uint), MetalGpuWriteOrigin.StorageBuffer);
        host.FlushAndWaitForGpu();
        Assert.Equal((2L, 1L, 1L, 0L), host.KernelCopyCheckCounters);
        Assert.Equal((3L, 1L, 0L), host.KernelCopyDestinationCheckCounters);
        Assert.Equal(0, host.KernelCopiesElided);
        Assert.Contains(
            "verify_checked=2,verify_mismatch=1,verify_dst_checked=3,verify_mismatch_dst=1,verify_dst_skipped=0,verify_superseded=1,verify_dropped=0]",
            host.TakeKernelFillCounters());
    }

    [Fact]
    public void VerifyCatchesAFillWhoseDestinationChangesBeforeTheGpuReachesIt()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.HostKernelCopyMode = MetalHostKernelCopyMode.Verify;
        for (var word = 0UL; word < KernelWords; word++)
        {
            rig.WriteBacking32(KernelDestination + word * sizeof(uint), 0x7u);
        }

        Assert.False(host.TryClearImageFromKernelFill(KernelDestination, KernelBytes, [0x7u]));
        host.NoteGuestMemoryWrite(KernelDestination, KernelBytes, MetalGpuWriteOrigin.StorageBuffer);
        host.ResetBindings();
        Assert.Equal(1, host.PendingKernelCopyChecks);
        rig.WriteBacking32(KernelDestination + 9 * sizeof(uint), 0x8u);
        host.FlushAndWaitForGpu();
        Assert.Equal((1L, 1L, 0L), host.KernelCopyDestinationCheckCounters);
    }

    // The executor's order under verify: the refusal (armed for its check), then the dispatch's
    // storage binding of the destination.
    private static void RecordVerifiedCopy(MetalRenderHost host)
    {
        Assert.False(host.TryCopyWordsOnHost(KernelDestination, KernelSource, KernelPeriod, KernelWords));
        host.NoteGuestMemoryWrite(KernelDestination, KernelBytes, MetalGpuWriteOrigin.StorageBuffer);
        host.ResetBindings();
        Assert.Equal(1, host.PendingKernelCopyChecks);
    }

    [Theory]
    [InlineData(null, "Off")]
    [InlineData("0", "Off")]
    [InlineData("identical", "Identical")]
    [InlineData("verify", "Verify")]
    [InlineData("write", "Off")]
    [InlineData("1", "Off")]
    public void TheSwitchTakesOnlyThePhaseAModes(string? text, string expected) =>
        Assert.Equal(expected, MetalV2Diagnostics.ParseHostKernelCopy(text).ToString());
}
