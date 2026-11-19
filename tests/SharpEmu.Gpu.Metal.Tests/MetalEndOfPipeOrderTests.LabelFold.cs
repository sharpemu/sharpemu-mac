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

    // A causal write the GPU path declines (an older label is owed by a completion) goes to the completion queue; a
    // plain write to the same bytes after it, with a compute encoder open, is declined by the same guard
    // (MayEncodeLabelOnGpu at the top of TryEncodeCommandProcessorWrite) before Step 3 or the fold can see it, so it
    // lands after the first, in order: the final bytes are the second write's.
    [Fact]
    public void APlainWriteAfterADeclinedCausalWriteLandsAfterIt()
    {
        if (!RequireDevice()) return;
        using var hold = new ManualResetEventSlim(false);
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        try
        {
            host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, WindowedLabel, 0xA0));
            var held = host.SubmissionSerial;
            host.Completions.Enqueue(held, () => hold.Wait(TimeSpan.FromSeconds(10)));
            // No window covers it, so this label is owed by a completion.
            host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, UnwindowedLabel, 1));
            var inEncoder = host.CpWritesEncodedInEncoder;

            host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(0x11UL), "WRITE_DATA", causal: true);
            _ = host.EnsureComputeEncoder();
            Assert.True(host.ComputeEncoderOpen);
            host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(0x22UL), "WRITE_DATA", causal: false);
            Assert.Equal(inEncoder, host.CpWritesEncodedInEncoder);
            Assert.Equal(0, host.DeferredLabelCopyCount);

            host.FlushCommandBuffer();
            WaitForGpu(host, host.LastWorkSerial);
            Assert.Equal(0UL, rig.Backing.Read64(LaterLabel));
            hold.Set();
            WaitForCompletions(host);
            Assert.Equal(0x22UL, rig.Backing.Read64(LaterLabel));
        }
        finally
        {
            hold.Set();
        }
    }

    // Stores a dispatch recorded into a staged copy are written back when the encoder ends, so a held copy is not folded
    // into the encoder ahead of them (counted as staged) and a plain write is held rather than encoded in it at once.
    [Fact]
    public void RecordedStagedStoresRefuseTheFoldAndHoldAPlainWrite()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        _ = host.TakeLabelInEncoderRefusals();
        _ = host.TakeLabelBarrierKinds();

        _ = host.EnsureComputeEncoder();
        host.StageRecordedStoresForTest(GuestBase + 0x10000, 0x100);
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 7));
        var folded = host.LabelsFolded;
        host.OrderAfterDeferredCopies(WindowedLabel, 8);
        Assert.False(host.ComputeEncoderOpen);
        Assert.Equal(folded, host.LabelsFolded);
        Assert.Equal(1L, host.TakeLabelInEncoderRefusals()[3]);
        Assert.Equal(new long[] { 0, 1, 0, 0 }, host.TakeLabelBarrierKinds());

        _ = host.EnsureComputeEncoder();
        host.StageRecordedStoresForTest(LaterLabel, 0x100);
        var inEncoder = host.CpWritesEncodedInEncoder;
        host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(0x55UL), "WRITE_DATA", causal: false);
        Assert.Equal(inEncoder, host.CpWritesEncodedInEncoder);

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(7UL, rig.Backing.Read64(WindowedLabel));
        Assert.Equal(0x55UL, rig.Backing.Read64(LaterLabel));
    }

    // Held copies a join asked to land, with no compute encoder open, land in the one the next dispatch opens, after its
    // fence wait.
    [Fact]
    public void HeldCopiesFoldIntoTheComputeEncoderTheyOpen()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        Assert.False(host.ComputeEncoderOpen);
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 0x31));
        Assert.Equal(1, host.DeferredLabelCopyCount);
        var folded = host.LabelsFolded;
        _ = host.TakeLabelBarrierKinds();

        host.RequestLabelLandingForTest();
        _ = host.EnsureComputeEncoder();
        Assert.True(host.ComputeEncoderOpen);
        Assert.Equal(0, host.DeferredLabelCopyCount);
        Assert.Equal(folded + 1, host.LabelsFolded);
        Assert.Equal(new long[] { 0, 0, 0, 1 }, host.TakeLabelBarrierKinds());

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(0x31UL, rig.Backing.Read64(WindowedLabel));
    }

    // A held copy that is not whole aligned words refuses the whole fold (counted as scatter): in the open encoder
    // (atOpen false) and in the one a dispatch opens (atOpen true, the decline fallback that ends it and opens
    // another). The old path lands the list in order behind the device event: the later label overwrites the unaligned
    // write's upper half, the unaligned write the earlier label's middle.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AnUnalignedHeldCopyRefusesTheWholeFoldAndTheListLandsInOrder(bool atOpen)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        _ = host.TakeLabelInEncoderRefusals();
        _ = host.TakeLabelBarrierKinds();
        if (!atOpen)
        {
            _ = host.EnsureComputeEncoder();
        }

        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, LaterLabel, 0x11111111_11111111UL));
        host.RecordCommandProcessorWrite(LaterLabel + 2, BitConverter.GetBytes(0xAABBCCDDu), "WRITE_DATA", causal: false);
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, LaterLabel + 4, 0x22222222));
        Assert.Equal(3, host.DeferredLabelCopyCount);
        var folded = host.LabelsFolded;
        if (atOpen)
        {
            host.RequestLabelLandingForTest();
            _ = host.EnsureComputeEncoder();
            Assert.True(host.ComputeEncoderOpen);
        }
        else
        {
            host.OrderAfterDeferredCopies(LaterLabel, 8);
            Assert.False(host.ComputeEncoderOpen);
        }

        Assert.Equal(0, host.DeferredLabelCopyCount);
        Assert.Equal(folded, host.LabelsFolded);
        Assert.Equal(1L, host.TakeLabelInEncoderRefusals()[1]);
        Assert.Equal(new long[] { 0, 1, 0, 0 }, host.TakeLabelBarrierKinds());

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(0x22222222_CCDD1111UL, rig.Backing.Read64(LaterLabel));
    }

    // Two held copies into the same words in one fold: the scatter keeps the later source per word.
    [Fact]
    public void TheLaterOfTwoHeldCopiesToTheSameWordWinsInsideOneFold()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, WindowedLabel, 0x11));
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 0x33333333_33333333UL));
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, WindowedLabel + 4, 0x22));
        var folded = host.LabelsFolded;
        host.OrderAfterDeferredCopies(WindowedLabel, 8);
        Assert.True(host.ComputeEncoderOpen);
        Assert.Equal(folded + 1, host.LabelsFolded);

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(0x00000022_33333333UL, rig.Backing.Read64(WindowedLabel));
    }

    // With SHARPEMU_METAL_V2_LABELS_IN_ENCODER, _CP_WRITE_DEVICE_ORDER and _CP_WRITES_IN_ENCODER all 0, a mixed sequence
    // lands as before the change: a batch with a command-processor write is CPU-visible and takes the shared event, a
    // device-only label batch the device event, each ending the encoder; nothing is folded or encoded in the encoder.
    [Fact]
    public void WithTheThreeSwitchesOffAMixedSequenceLandsAsBefore()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.LabelsInEncoder = false;
        host.CpWritesDeviceOrdered = false;
        host.CpWritesInEncoder = false;
        _ = host.TakeLabelBarriers();
        _ = host.TakeLabelBarrierKinds();
        var folded = host.LabelsFolded;
        var inEncoder = host.CpWritesEncodedInEncoder;

        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 1));
        host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(2UL), "WRITE_DATA", causal: false);
        host.RecordCommandProcessorWrite(WindowedLabel, BitConverter.GetBytes(4u), "WRITE_DATA", causal: false);
        Assert.Equal(3, host.DeferredLabelCopyCount);
        host.OrderAfterDeferredCopies(LaterLabel, 8, MetalRenderHost.LabelLandCause.BindRead);
        Assert.False(host.ComputeEncoderOpen);
        Assert.Equal(new long[] { 1, 0, 0, 0 }, host.TakeLabelBarrierKinds());

        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, LaterLabel + 0x10, 3));
        host.OrderAfterDeferredCopies(LaterLabel + 0x10, 4);
        Assert.False(host.ComputeEncoderOpen);
        Assert.Equal(new long[] { 0, 1, 0, 0 }, host.TakeLabelBarrierKinds());
        Assert.Equal(2L, host.TakeLabelBarriers());
        Assert.Equal(folded, host.LabelsFolded);
        Assert.Equal(inEncoder, host.CpWritesEncodedInEncoder);

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(4UL, rig.Backing.Read64(WindowedLabel));
        Assert.Equal(2UL, rig.Backing.Read64(LaterLabel));
        Assert.Equal(3u, rig.Backing.Read32(LaterLabel + 0x10));
    }
}
