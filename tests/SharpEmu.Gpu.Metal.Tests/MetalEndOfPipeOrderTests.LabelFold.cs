// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// Held copies the CPU need not see before the work ahead of them land as a dispatch in the open compute encoder
// (MetalRenderHost.LabelScatter.cs); one the CPU must see keeps the shared event.
public sealed unsafe partial class MetalEndOfPipeOrderTests
{
    // A device-only label, then a store over its bytes, then a later label the next dispatch binds: each lands inside the
    // open compute encoder, in stream order, with no barrier.
    [Fact]
    public void DeviceOrderedHeldCopiesLandInTheOpenComputeEncoderInStreamOrder()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        Assert.True(host.LabelsAtCommit);
        Assert.True(host.LabelsInEncoder);
        const ulong second = GuestBase + 0x10000;
        _ = host.TakeLabelBarriers();
        _ = host.TakeLabelBarrierKinds();

        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 0x11));
        Assert.Equal(1, host.DeferredLabelCopyCount);
        var folded = host.LabelsFolded;

        // A fill over the label's bytes (a dispatch in the encoder) lands the label first, inside the encoder.
        host.FillBufferInOrder(WindowedLabel, 0x2000, 0x77);
        Assert.True(host.ComputeEncoderOpen);
        Assert.Equal(0, host.DeferredLabelCopyCount);
        Assert.Equal(folded + 1, host.LabelsFolded);

        // The next dispatch binds a label recorded after the fill: it lands ahead of it, still in the encoder.
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, second, 0x22));
        host.OrderAfterDeferredCopies(second, 4);
        Assert.True(host.ComputeEncoderOpen);
        Assert.Equal(0, host.DeferredLabelCopyCount);
        Assert.Equal(folded + 2, host.LabelsFolded);
        Assert.Equal(0L, host.TakeLabelBarriers());
        Assert.Equal(new long[] { 0, 0, 0, 2 }, host.TakeLabelBarrierKinds());

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(0x0000007700000077UL, rig.Backing.Read64(WindowedLabel));
        Assert.Equal(0x22u, rig.Backing.Read32(second));
    }

    // A held label the guest asked to be written back for the CPU is never folded: the batch it is in ends the encoder
    // and lands behind the shared event, with the device-only label held before it.
    [Fact]
    public void ABatchWithACpuVisibleLabelIsNotFoldedAndTakesTheSharedEvent()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        _ = host.TakeLabelBarriers();
        _ = host.TakeLabelBarrierKinds();

        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 1));
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.WriteBack64, 1, LaterLabel, 2));
        var folded = host.LabelsFolded;
        host.OrderAfterDeferredCopies(WindowedLabel, 8);
        Assert.False(host.ComputeEncoderOpen);
        Assert.Equal(0, host.DeferredLabelCopyCount);
        Assert.Equal(folded, host.LabelsFolded);
        Assert.Equal(1L, host.TakeLabelBarriers());
        Assert.Equal(new long[] { 1, 0, 0, 0 }, host.TakeLabelBarrierKinds());

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(1UL, rig.Backing.Read64(WindowedLabel));
        Assert.Equal(2UL, rig.Backing.Read64(LaterLabel));
    }

    // Without the encoder fence chain an encoder may start before the work ahead of it finished, so nothing is folded
    // (counted as no_fence); with the fold switched off nothing is folded either. Both land behind the device event.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TheFoldIsRefusedWithoutTheFenceChainAndWhenSwitchedOff(bool noFence)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        if (noFence)
        {
            host.EncoderOrdering = MetalRenderHost.EncoderOrderingMode.None;
        }
        else
        {
            host.LabelsInEncoder = false;
        }

        _ = host.TakeLabelInEncoderRefusals();
        _ = host.TakeLabelBarrierKinds();
        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 5));
        var folded = host.LabelsFolded;
        host.OrderAfterDeferredCopies(WindowedLabel, 8);
        Assert.False(host.ComputeEncoderOpen);
        Assert.Equal(folded, host.LabelsFolded);
        Assert.Equal(noFence ? 1L : 0L, host.TakeLabelInEncoderRefusals()[2]);
        Assert.Equal(new long[] { 0, 1, 0, 0 }, host.TakeLabelBarrierKinds());

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(5UL, rig.Backing.Read64(WindowedLabel));
    }

    // A command-processor write nothing causal came before (no read of a pending value, no flush or acquire on its
    // queue) is device-ordered: the dispatch that reads it lands it inside the open compute encoder, with no barrier.
    [Fact]
    public void APlainCommandProcessorWriteADispatchReadsLandsInTheOpenComputeEncoder()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        Assert.True(host.CpWritesDeviceOrdered);
        _ = host.TakeLabelBarriers();
        _ = host.TakeLabelBarrierKinds();

        _ = host.EnsureComputeEncoder();
        host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(0x55UL), "WRITE_DATA", causal: false);
        host.OrderAfterDeferredCopies(LaterLabel, 8, MetalRenderHost.LabelLandCause.BindRead);
        Assert.True(host.ComputeEncoderOpen);
        Assert.Equal(0, host.DeferredLabelCopyCount);
        Assert.Equal(0L, host.TakeLabelBarriers());
        var kinds = host.TakeLabelBarrierKinds();
        Assert.Equal(0L, kinds[0]);
        Assert.Equal(0L, kinds[1]);

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(0x55UL, rig.Backing.Read64(LaterLabel));
    }

    // A causal write (its queue's wait was met by a label not yet landed) stays CPU-visible: it is held, and every write
    // after it is held behind it, CPU-visible too, until they land together behind the shared event with the label -
    // in Demon's Souls' recycled command memory the store must not reach the guest before the label it waited on.
    [Fact]
    public void ACausalWriteAndTheWritesAfterItLandBehindTheSharedEventWithTheLabel()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        _ = host.TakeLabelBarriers();
        _ = host.TakeLabelBarrierKinds();

        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.WriteBack32, 1, WindowedLabel, 1));
        host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(0x55UL), "WRITE_DATA", causal: true);
        host.RecordCommandProcessorWrite(LaterLabel + 8, BitConverter.GetBytes(0x66UL), "WRITE_DATA", causal: false);
        Assert.Equal(3, host.DeferredLabelCopyCount);

        host.OrderAfterDeferredCopies(LaterLabel + 8, 8, MetalRenderHost.LabelLandCause.BindRead);
        Assert.False(host.ComputeEncoderOpen);
        Assert.Equal(0, host.DeferredLabelCopyCount);
        Assert.Equal(1L, host.TakeLabelBarriers());
        Assert.Equal(new long[] { 1, 0, 0, 0 }, host.TakeLabelBarrierKinds());

        // Landed, the next plain write is device-ordered again.
        _ = host.EnsureComputeEncoder();
        host.RecordCommandProcessorWrite(LaterLabel + 16, BitConverter.GetBytes(0x77UL), "WRITE_DATA", causal: false);
        host.OrderAfterDeferredCopies(LaterLabel + 16, 8, MetalRenderHost.LabelLandCause.BindRead);
        Assert.True(host.ComputeEncoderOpen);
        Assert.Equal(0L, host.TakeLabelBarriers());

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(1u, rig.Backing.Read32(WindowedLabel));
        Assert.Equal(0x55UL, rig.Backing.Read64(LaterLabel));
        Assert.Equal(0x66UL, rig.Backing.Read64(LaterLabel + 8));
        Assert.Equal(0x77UL, rig.Backing.Read64(LaterLabel + 16));
    }

    // A plain command-processor write recorded while a compute encoder is open is a dispatch in that encoder at once
    // (SHARPEMU_METAL_V2_CP_WRITES_IN_ENCODER), unless a held copy covers its bytes: Demon's Souls' recycled command
    // memory, where a label from an earlier frame is still owed over the bytes the write reuses. That one is held behind
    // the label and lands after it.
    [Fact]
    public void APlainWriteOverAHeldLabelIsHeldBehindItAndLandsAfterIt()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        Assert.True(host.CpWritesInEncoder);

        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, WindowedLabel, 0x0BAD));
        var inEncoder = host.CpWritesEncodedInEncoder;

        // Elsewhere, a plain write is a dispatch in the encoder: nothing is held for it.
        host.RecordCommandProcessorWrite(LaterLabel + 0x100, BitConverter.GetBytes(0x99UL), "WRITE_DATA", causal: false);
        Assert.Equal(1, host.DeferredLabelCopyCount);
        Assert.Equal(inEncoder + 1, host.CpWritesEncodedInEncoder);

        host.RecordCommandProcessorWrite(WindowedLabel, BitConverter.GetBytes(0xC0DEu), "WRITE_DATA", causal: false);
        Assert.Equal(2, host.DeferredLabelCopyCount);
        Assert.Equal(inEncoder + 1, host.CpWritesEncodedInEncoder);

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(0xC0DEu, rig.Backing.Read32(WindowedLabel));
        Assert.Equal(0x99UL, rig.Backing.Read64(LaterLabel + 0x100));
    }

    // A held command-processor write keeps every later one held behind it, so none overtakes it: a causal write, then a
    // plain one elsewhere and a plain one over the first write's bytes, land in that order.
    [Fact]
    public void APlainWriteAfterAHeldWriteIsHeldAndTheyLandInOrder()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;

        _ = host.EnsureComputeEncoder();
        var inEncoder = host.CpWritesEncodedInEncoder;
        host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(1UL), "WRITE_DATA", causal: true);
        host.RecordCommandProcessorWrite(LaterLabel + 0x40, BitConverter.GetBytes(2UL), "WRITE_DATA", causal: false);
        host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(3UL), "WRITE_DATA", causal: false);
        Assert.Equal(3, host.DeferredLabelCopyCount);
        Assert.Equal(inEncoder, host.CpWritesEncodedInEncoder);

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(3UL, rig.Backing.Read64(LaterLabel));
        Assert.Equal(2UL, rig.Backing.Read64(LaterLabel + 0x40));
    }

    // A dispatch reading a plain write needs no landing at all, and a device-only label inside a 1 GiB binding the
    // dispatch reads lands in the encoder: the encoder stays open. A CPU-visible label in that binding still ends it,
    // counted as a read binding forced by a CPU-visible label in a range of 1 GiB or more.
    [Fact]
    public void ABigReadBindingOverHeldLabelsEndsTheEncoderOnlyForACpuVisibleOne()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        const ulong gib = 1UL << 30;
        _ = host.TakeLabelLandCounts();
        _ = host.TakeLabelBarrierKinds();

        _ = host.EnsureComputeEncoder();
        host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(0x55UL), "WRITE_DATA", causal: false);
        host.OrderAfterDeferredCopies(LaterLabel, 8, MetalRenderHost.LabelLandCause.BindRead);
        Assert.Equal(0, host.DeferredLabelCopyCount);
        Assert.All(host.TakeLabelLandCounts(), count => Assert.Equal(0L, count));

        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 7));
        host.OrderAfterDeferredCopies(GuestBase, gib, MetalRenderHost.LabelLandCause.BindRead);
        Assert.True(host.ComputeEncoderOpen);
        Assert.Equal(0, host.DeferredLabelCopyCount);

        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.WriteBack64, 1, WindowedLabel + 8, 8));
        host.OrderAfterDeferredCopies(GuestBase, gib, MetalRenderHost.LabelLandCause.BindRead);
        Assert.False(host.ComputeEncoderOpen);
        var lands = host.TakeLabelLandCounts();
        Assert.Equal(2L, lands[(int)MetalRenderHost.LabelLandCause.BindRead - 1]);
        Assert.Equal(new long[] { 1, 0, 0, 1 }, host.TakeLabelBarrierKinds());

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(0x55UL, rig.Backing.Read64(LaterLabel));
        Assert.Equal(7UL, rig.Backing.Read64(WindowedLabel));
        Assert.Equal(8UL, rig.Backing.Read64(WindowedLabel + 8));
    }

    // A write-back label recorded after a GPU store is seen by the CPU only once the store is too: 500 times, the CPU
    // polls the label and reads the store's last word as soon as it sees it.
    [Fact]
    public void ACpuVisibleLabelIsNeverSeenBeforeTheStoreAheadOfIt()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        const ulong filled = GuestBase + 0x10000;
        const ulong size = 0x4000;
        var early = 0;
        for (var attempt = 1; attempt <= 500; attempt++)
        {
            _ = host.EnsureComputeEncoder();
            host.FillBufferInOrder(filled, size, (uint)attempt);
            host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.WriteBack64, 1, WindowedLabel, (ulong)attempt));
            // Forced while the encoder is open, as a dispatch binding the label would.
            host.OrderAfterDeferredCopies(WindowedLabel, 8);
            var serial = host.FlushCommandBuffer();
            var started = Stopwatch.GetTimestamp();
            while (rig.Backing.Read64(WindowedLabel) != (ulong)attempt)
            {
                Assert.True(Stopwatch.GetElapsedTime(started) < TimeSpan.FromSeconds(10), $"label {attempt} never landed");
                Thread.SpinWait(8);
            }

            if (rig.Backing.Read32(filled + size - sizeof(uint)) != (uint)attempt)
            {
                early++;
            }

            WaitForGpu(host, serial);
            WaitForCompletions(host);
        }

        Assert.Equal(0, early);
    }
}
