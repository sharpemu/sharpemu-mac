// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// End-of-pipe writes of a stream land in the order they were recorded. A 32-bit label no
/// window covers is published by its submission's completion; a 64-bit label recorded
/// after it that the GPU could write itself used to land as soon as the GPU reached it,
/// before that completion ran, and a guest reusing memory on the newer signal then had
/// the older label land in what it had reused. The completion thread is held here so the
/// gap between the GPU finishing and the host publishing is as wide as it is in a slow
/// frame. Device work is behind the exclusive device gate.
/// </summary>
public sealed unsafe class MetalEndOfPipeOrderTests(ITestOutputHelper output)
{
    private const ulong GuestBase = 0x2_0000_0000;
    private const ulong WindowedLabel = GuestBase + 0x1000;
    private const ulong LaterLabel = GuestBase + 0x2000;

    // No window covers this page, so a label here can only be published by the host.
    private const ulong HoleOffset = 0x8000;
    private const ulong HoleSize = 0x1000;
    private const ulong UnwindowedLabel = GuestBase + HoleOffset + 0x40;

    [Theory]
    [InlineData(false, 0L, true)]
    [InlineData(false, 3L, true)]
    [InlineData(true, 0L, true)]
    [InlineData(true, 1L, false)]
    public void ALabelIsWrittenByTheGpuOnlyWhileNoOlderLabelIsOwed(bool ordered, long owed, bool expected) =>
        Assert.Equal(expected, MetalRenderHost.MayEncodeLabelOnGpu(ordered, owed));

    [Fact]
    public void ALaterGpuWrittenLabelWaitsBehindAnOwed32BitLabel()
    {
        if (!RequireDevice()) return;
        var (owedLanded, laterLanded, ordered) = RecordOwedThenLater(orderedEndOfPipe: true);

        // While the GPU has finished both and the completion is held, neither is visible:
        // the later label is not allowed to say more than the owed one.
        Assert.Equal(0UL, owedLanded.WhileHeld);
        Assert.Equal(0UL, laterLanded.WhileHeld);
        Assert.Equal(1L, ordered);

        Assert.Equal(1UL, owedLanded.AfterDrain);
        Assert.Equal(2UL, laterLanded.AfterDrain);
    }

    [Fact]
    public void TheOptOutLetsTheLaterLabelOvertakeTheOwedOne()
    {
        if (!RequireDevice()) return;
        var (owedLanded, laterLanded, ordered) = RecordOwedThenLater(orderedEndOfPipe: false);

        // SHARPEMU_METAL_V2_EOP_ORDERED=0: the GPU writes the 64-bit label at once, while
        // the 32-bit one recorded before it still waits for the held completion.
        Assert.Equal(0UL, owedLanded.WhileHeld);
        Assert.Equal(2UL, laterLanded.WhileHeld);
        Assert.Equal(0L, ordered);

        Assert.Equal(1UL, owedLanded.AfterDrain);
        Assert.Equal(2UL, laterLanded.AfterDrain);
    }

    [Fact]
    public void ACommandProcessorWriteWaitsBehindAnOwed32BitLabel()
    {
        if (!RequireDevice()) return;
        var (owedLanded, laterLanded, _) = RecordOwedThenLater(orderedEndOfPipe: true,
            later: host => host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(2UL), "WRITE_DATA"));

        Assert.Equal(0UL, owedLanded.WhileHeld);
        Assert.Equal(0UL, laterLanded.WhileHeld);
        Assert.Equal(1UL, owedLanded.AfterDrain);
        Assert.Equal(2UL, laterLanded.AfterDrain);
    }

    /// <summary>
    /// A WRITE_DATA or DMA fill recorded after a label is copied by the GPU behind it, not
    /// written as parsed: until the GPU reaches it the guest sees neither, so it can never
    /// see the store ahead of the label.
    /// </summary>
    [Fact]
    public void CommandProcessorWritesLandWithTheLabelBeforeThemAndNotAtParse()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        Assert.True(host.DefersCommandProcessorWrites);
        const ulong filled = GuestBase + 0x3000;

        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, WindowedLabel, 1));
        host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(0x55UL), "WRITE_DATA");
        host.FillBufferInOrder(filled, sizeof(uint), 0x77);

        Assert.Equal(0u, rig.Backing.Read32(WindowedLabel));
        Assert.Equal(0UL, rig.Backing.Read64(LaterLabel));
        Assert.Equal(0u, rig.Backing.Read32(filled));
        Assert.Equal((2L, 0L), host.TakeCommandProcessorWriteCounters());

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(1u, rig.Backing.Read32(WindowedLabel));
        Assert.Equal(0x55UL, rig.Backing.Read64(LaterLabel));
        Assert.Equal(0x77u, rig.Backing.Read32(filled));
    }

    [Fact]
    public void TheCommandProcessorWriteOptOutLeavesThemToTheParser()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        Assert.True(rig.Host.DefersCommandProcessorWrites);
        rig.Host.CommandProcessorWritesAtParse = true;
        Assert.False(rig.Host.DefersCommandProcessorWrites);
    }

    /// <summary>
    /// A label or a command-processor write recorded inside an open compute encoder waits for
    /// the encoder's end instead of ending it, and lands there: after what was recorded before
    /// the encoder and before what is recorded after it.
    /// </summary>
    [Fact]
    public void ACopyInsideAnOpenComputeEncoderLandsAtItsEndInOrder()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        // The encoder-boundary mode, where copies land at the next boundary rather than the commit.
        host.LabelsAtCommit = false;
        Assert.True(host.DefersCopiesInCompute);
        const ulong filled = GuestBase + 0x10000;

        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 1));
        host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(0x11UL), "WRITE_DATA");
        Assert.Equal(0, host.DeferredLabelCopyCount);
        _ = host.TakeDeferredLabelWrites();

        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 2));
        host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(0x55UL), "WRITE_DATA");
        Assert.True(host.ComputeEncoderOpen);
        Assert.Equal(2, host.DeferredLabelCopyCount);
        Assert.Equal(2L, host.TakeDeferredLabelWrites());

        // A transfer too large to stage ends the encoder; the copies are encoded at that boundary.
        host.FillBufferInOrder(filled, 0x2000, 0x77);
        Assert.False(host.ComputeEncoderOpen);
        Assert.Equal(0, host.DeferredLabelCopyCount);

        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 3));
        Assert.Equal(0, host.DeferredLabelCopyCount);

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(3UL, rig.Backing.Read64(WindowedLabel));
        Assert.Equal(0x55UL, rig.Backing.Read64(LaterLabel));
        Assert.Equal(0x77u, rig.Backing.Read32(filled));
    }

    // Held to the commit, label copies inside and outside an open encoder wait for it and land
    // behind one barrier, in stream order. A transfer over other bytes leaves them held; one
    // over their bytes lands them first (MetalClearFoldingTests.AGpuWriteRecordedAfterTheOwedBytes...).
    [Fact]
    public void LabelCopiesWaitForTheCommitAndLandBehindOneBarrierInOrder()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.LabelsAtCommit = true;
        const ulong filled = GuestBase + 0x10000;
        _ = host.TakeLabelBarriers();

        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 1));
        host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(0x11UL), "WRITE_DATA");
        var heldOutside = host.DeferredLabelCopyCount;
        Assert.True(heldOutside >= 2);

        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 2));
        host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(0x55UL), "WRITE_DATA");
        var held = host.DeferredLabelCopyCount;
        Assert.True(held > heldOutside);

        host.FillBufferInOrder(filled, 0x2000, 0x77);
        Assert.False(host.ComputeEncoderOpen);
        Assert.Equal(held, host.DeferredLabelCopyCount);

        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 3));
        Assert.True(host.DeferredLabelCopyCount > held);
        Assert.Equal(0L, host.TakeLabelBarriers());

        host.FlushCommandBuffer();
        Assert.Equal(0, host.DeferredLabelCopyCount);
        Assert.Equal(1L, host.TakeLabelBarriers());
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(3UL, rig.Backing.Read64(WindowedLabel));
        Assert.Equal(0x55UL, rig.Backing.Read64(LaterLabel));
        Assert.Equal(0x77u, rig.Backing.Read32(filled));
    }

    // A copy waiting for the open compute encoder to end lands after every dispatch recorded
    // into it. A dispatch binding its bytes - GTA V's bounds pass over the table its
    // WRITE_DATA has just initialised - ends the encoder first so the copy lands before it;
    // a dispatch over other bytes leaves the encoder open.
    [Fact]
    public void ADispatchBindingBytesACopyStillOwesEndsTheComputeEncoderFirst()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        Assert.True(host.DefersCopiesInCompute);

        _ = host.EnsureComputeEncoder();
        host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(0x55UL), "WRITE_DATA");
        Assert.Equal(1, host.DeferredLabelCopyCount);

        host.OrderAfterDeferredCopies(LaterLabel + 0x100, 0x40);
        Assert.True(host.ComputeEncoderOpen);
        Assert.Equal(1, host.DeferredLabelCopyCount);

        host.OrderAfterDeferredCopies(LaterLabel + 4, 0x40);
        Assert.False(host.ComputeEncoderOpen);
        Assert.Equal(0, host.DeferredLabelCopyCount);

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(0x55UL, rig.Backing.Read64(LaterLabel));
    }

    // The held copies' span only spares the scan: a binding between two of them that touches
    // neither keeps the encoder open, and once they land the span starts over, so bytes only
    // an earlier copy covered no longer count.
    [Fact]
    public void ABindingBetweenTwoHeldCopiesTouchesNeitherAndTheSpanStartsOverWhenTheyLand()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        Assert.True(host.DefersCopiesInCompute);

        _ = host.EnsureComputeEncoder();
        host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(0x55UL), "WRITE_DATA");
        host.RecordCommandProcessorWrite(LaterLabel + 0x1000, BitConverter.GetBytes(0x66UL), "WRITE_DATA");
        Assert.Equal(2, host.DeferredLabelCopyCount);

        host.OrderAfterDeferredCopies(LaterLabel + 0x800, 0x40);
        Assert.True(host.ComputeEncoderOpen);
        Assert.Equal(2, host.DeferredLabelCopyCount);

        host.OrderAfterDeferredCopies(LaterLabel + 0x1000, 8);
        Assert.False(host.ComputeEncoderOpen);
        Assert.Equal(0, host.DeferredLabelCopyCount);

        _ = host.EnsureComputeEncoder();
        host.RecordCommandProcessorWrite(LaterLabel + 0x1000, BitConverter.GetBytes(0x77UL), "WRITE_DATA");
        host.OrderAfterDeferredCopies(LaterLabel, 8);
        Assert.True(host.ComputeEncoderOpen);
        Assert.Equal(1, host.DeferredLabelCopyCount);

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(0x55UL, rig.Backing.Read64(LaterLabel));
        Assert.Equal(0x77UL, rig.Backing.Read64(LaterLabel + 0x1000));
    }

    [Fact]
    public void TheComputeCopyOptOutEndsTheComputeEncoderForTheCopy()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        // The encoder-boundary mode, where copies land at the next boundary rather than the commit.
        host.LabelsAtCommit = false;
        host.DefersCopiesInCompute = false;

        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 2));
        Assert.False(host.ComputeEncoderOpen);
        Assert.Equal(0, host.DeferredLabelCopyCount);
        Assert.Equal(0L, host.TakeDeferredLabelWrites());

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(2UL, rig.Backing.Read64(WindowedLabel));
    }

    // A tiled image upload and writeback the one-pass tiler encodes are dispatches alone, so
    // they are recorded into the open compute encoder instead of ending it, and land the same
    // bytes as the same round trip in encoders of their own.
    [Fact]
    public void AKernelOnlyImageTransferIsEncodedIntoTheOpenComputeEncoder()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        using var harness = new MetalTransferHarness(Device.Value);
        var host = rig.Host;
        Assert.True(host.TransfersShareCompute);

        const uint width = 192;
        const uint height = 96;
        var format = ImageDescription.RenderTargetTransferFormat(4);
        var size = TileGeometry.TextureTotalSize(format, width, height, 1, 1, GuestTileMode.RenderTarget, volume: false).Size;
        var description = ImageDescription.Create();
        description.Data = new GuestSpan(GuestBase + 0x40000, size);
        description.GuestFormat = GuestPixelFormat.Bits8_8_8_8UNorm;
        description.PixelFormat = GpuFormat.R8G8B8A8Unorm;
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(width, height, 1);
        description.Resources = new SubresourceCount(1, 1);
        description.Pitch = TileGeometry.TexturePitch(format, width, GuestTileMode.RenderTarget);
        description.BytesPerBlock = 4;
        description.TileMode = GuestTileMode.RenderTarget;
        var range = new SubresourceRange(0, 1, 0, 1);
        var guest = new byte[size];
        for (var index = 0; index < guest.Length; index++)
        {
            guest[index] = (byte)(index * 7 + 3);
        }

        var source = harness.Buffer(guest);
        var texture = harness.Texture(harness.Transfer.ShapeFor(description));

        // The same round trip in encoders of its own is the reference.
        var expected = harness.Buffer(size);
        harness.Begin();
        harness.Transfer.EncodeGuestToTexture(harness, description, range,
            new MetalGuestRange(source, 0, size, 0), default, texture, 0, size);
        harness.Transfer.EncodeTextureToGuest(harness, description, range, texture,
            new MetalGuestRange(expected, 0, size, 0), default, 0, size);
        harness.Submit();

        var shared = new MetalImageTransfer(Device.Value);
        try
        {
            var actual = harness.Buffer(size);
            var inEncoder = host.TransfersInEncoder;
            _ = host.EnsureComputeEncoder();
            var upload = ((IMetalImageCacheOwner)host).BeginTransfer(MetalTransferCause.Upload, texture);
            Assert.True(host.ComputeEncoderOpen);
            shared.EncodeGuestToTexture(upload, description, range,
                new MetalGuestRange(source, 0, size, 0), default, texture, 0, size);
            var writeback = ((IMetalImageCacheOwner)host).BeginTransfer(MetalTransferCause.Writeback, texture);
            shared.EncodeTextureToGuest(writeback, description, range, texture,
                new MetalGuestRange(actual, 0, size, 0), default, 0, size);
            Assert.True(host.ComputeEncoderOpen);
            Assert.Equal(inEncoder + 2, host.TransfersInEncoder);
            Assert.Equal(0u, shared.Tiler.StagingBufferCount);

            var serial = host.FlushCommandBuffer();
            WaitForGpu(host, serial);
            var landed = harness.Read(actual, 0, size);
            Assert.Contains(landed, value => value != 0);
            Assert.Equal(harness.Read(expected, 0, size), landed);
        }
        finally
        {
            shared.Dispose();
        }
    }

    [Fact]
    public void AFlushWithAnOpenComputeEncoderEncodesTheWaitingCopiesBeforeCommitting()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;

        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 2));
        host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(0x55UL), "WRITE_DATA");
        Assert.Equal(2, host.DeferredLabelCopyCount);

        var serial = host.FlushCommandBuffer();
        Assert.False(host.ComputeEncoderOpen);
        Assert.Equal(0, host.DeferredLabelCopyCount);

        WaitForGpu(host, serial);
        WaitForCompletions(host);
        Assert.Equal(2UL, rig.Backing.Read64(WindowedLabel));
        Assert.Equal(0x55UL, rig.Backing.Read64(LaterLabel));
    }

    // The windows are untracked, so Metal may start a label copy's encoder before the work
    // recorded ahead of it has finished. Each label copy, written at once or at an encoder
    // boundary, waits on an event the command buffer signals once that work is done.
    [Fact]
    public void ALabelCopyWaitsBehindABarrierForTheWorkRecordedBeforeIt()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        // The encoder-boundary mode, where copies land at the next boundary rather than the commit.
        host.LabelsAtCommit = false;
        const ulong filled = GuestBase + 0x10000;
        _ = host.TakeLabelBarriers();

        host.FillBufferInOrder(filled, 0x2000, 0x77);
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 7));
        Assert.Equal(0, host.DeferredLabelCopyCount);
        Assert.Equal(1L, host.TakeLabelBarriers());

        _ = host.EnsureComputeEncoder();
        host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(0x66UL), "WRITE_DATA");
        Assert.Equal(1, host.DeferredLabelCopyCount);
        Assert.Equal(0L, host.TakeLabelBarriers());

        var serial = host.FlushCommandBuffer();
        Assert.Equal(1L, host.TakeLabelBarriers());
        WaitForGpu(host, serial);
        WaitForCompletions(host);
        Assert.Equal(7UL, rig.Backing.Read64(WindowedLabel));
        Assert.Equal(0x66UL, rig.Backing.Read64(LaterLabel));
        Assert.Equal(0x77u, rig.Backing.Read32(filled));
    }

    // Finished is not visible: a device-only MTLEvent lets the copy land before the earlier
    // stores reach memory the CPU reads, and a store the guest makes on seeing the label is
    // then overwritten by them. The barrier signals an event the CPU can observe, which
    // publishes those stores first. Only an MTLSharedEvent answers signaledValue.
    [Fact]
    public void ALabelBarrierSignalsAnEventTheCpuCanObserve()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        // The encoder-boundary mode, where copies land at the next boundary rather than the commit.
        host.LabelsAtCommit = false;

        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 3));
        Assert.Equal(1L, host.TakeLabelBarriers());
        Assert.NotEqual(0, host.LabelBarrierEvent);
        Assert.True(MetalNative.SendBool(
            host.LabelBarrierEvent, MetalNative.Selector("respondsToSelector:"), MetalNative.Selector("signaledValue")));

        WaitForGpu(host, host.FlushCommandBuffer());
        WaitForCompletions(host);
        Assert.Equal(3UL, rig.Backing.Read64(WindowedLabel));
    }

    // A guest store waiting for a recorded writeback waits for its submission off the worker:
    // the worker itself refuses (it would wait for itself), a submission still being recorded
    // is committed through the relay, and a wait nobody commits for gives up at its timeout.
    [Fact]
    public void AWaitOffTheWorkerAsksTheWorkerToCommitTheSubmission()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        var owner = (IMetalImageCacheOwner)host;
        host.Relay.BindCurrentThread();

        host.FillBufferInOrder(GuestBase + 0x10000, 0x2000, 0x77);
        var serial = host.LastWorkSerial;
        Assert.True(host.CompletedSerial < serial);
        Assert.False(owner.WaitForSubmissionOffWorker(serial, 10_000));

        Assert.False(Task.Run(() => owner.WaitForSubmissionOffWorker(serial, 50)).Result);
        Assert.True(host.Relay.HasPendingCommands);

        var waiter = Task.Run(() => owner.WaitForSubmissionOffWorker(serial, 10_000));
        host.RunPendingOperations();
        Assert.True(waiter.Wait(TimeSpan.FromSeconds(10)));
        Assert.True(waiter.Result);
        Assert.True(host.CompletedSerial >= serial);
        WaitForCompletions(host);
        Assert.Equal(0x77u, rig.Backing.Read32(GuestBase + 0x10000));
    }

    [Theory]
    [InlineData(null, "Deferred")]
    [InlineData("", "Deferred")]
    [InlineData("completion", "Completion")]
    [InlineData("completion-strict", "CompletionStrict")]
    [InlineData("deferred", "Deferred")]
    [InlineData("parse", "Parse")]
    public void TheClockSamplingModeComesFromTheEnvironmentValue(string? value, string expected) =>
        Assert.Equal(expected, MetalRenderHost.ParseClockSampling(value).ToString());

    /// <summary>
    /// A clock label is the time the work before it finished, so it is sampled by that work's
    /// completion and not as parsed - Demon's Souls sizes its dynamic resolution from the gap
    /// between two of them. A label recorded after it must not become visible first.
    /// </summary>
    [Theory]
    [InlineData("Completion")]
    [InlineData("CompletionStrict")]
    public void AClockLabelCarriesItsCompletionTimeAndALaterLabelWaitsBehindIt(string mode)
    {
        if (!RequireDevice()) return;
        var sampling = Enum.Parse<EndOfPipeClockSampling>(mode);
        var (parsed, gpuDone, clock, later, counters, committed) = RecordClockThenLater(sampling);

        // The lazy default commits nothing for the clock or the label behind it; strict
        // commits the buffer at the clock.
        Assert.Equal(sampling == EndOfPipeClockSampling.CompletionStrict, committed);
        Assert.Equal(0UL, clock.WhileHeld);
        Assert.Equal(0UL, later.WhileHeld);
        Assert.True(clock.AfterDrain >= gpuDone,
            $"The clock 0x{clock.AfterDrain:X} predates the GPU finishing the work before it (0x{gpuDone:X}).");
        Assert.True(clock.AfterDrain > parsed);
        Assert.Equal(2UL, later.AfterDrain);
        Assert.Equal((1L, 1L), counters);
    }

    [Fact]
    public void TheDeferredClockLandsInOrderWithItsParseTimeValue()
    {
        if (!RequireDevice()) return;
        var (parsed, _, clock, later, counters, _) = RecordClockThenLater(EndOfPipeClockSampling.Deferred);

        // The GPU copied both as it reached them: no completion, the value as parsed.
        Assert.Equal(parsed, clock.WhileHeld);
        Assert.Equal(2UL, later.WhileHeld);
        Assert.Equal(parsed, clock.AfterDrain);
        Assert.Equal((0L, 0L), counters);
    }

    /// <summary>
    /// A clock that rides the next commit is still committed by a wait packet polling it, so
    /// a queue waiting on the stamp never waits for the worker to park.
    /// </summary>
    [Fact]
    public void AWaitPacketPollingALazyClockCommitsItsBuffer()
    {
        if (!RequireDevice()) return;
        const ulong clockLabel = GuestBase + 0x4000;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.ClockSampling = EndOfPipeClockSampling.Completion;

        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, WindowedLabel, 0xA0));
        var open = host.SubmissionSerial;
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.ClockWrite, 1, clockLabel, 1, ReferenceClock: true));
        Assert.Equal(open, host.SubmissionSerial);

        Span<byte> polled = stackalloc byte[sizeof(ulong)];
        Assert.True(host.TryReadGuestWaitOperand(clockLabel, polled, commitPendingWrite: false));
        Assert.Equal(open, host.SubmissionSerial);
        Assert.True(host.TryReadGuestWaitOperand(clockLabel, polled, commitPendingWrite: true));
        Assert.NotEqual(open, host.SubmissionSerial);

        WaitForGpu(host, open);
        WaitForCompletions(host);
        Assert.True(rig.Backing.Read64(clockLabel) > 1);
    }

    private (ulong Parsed, ulong GpuDone, (ulong WhileHeld, ulong AfterDrain) Clock,
        (ulong WhileHeld, ulong AfterDrain) Later, (long Completion, long Forced) Counters, bool Committed)
        RecordClockThenLater(EndOfPipeClockSampling sampling)
    {
        const ulong clockLabel = GuestBase + 0x4000;
        using var hold = new ManualResetEventSlim(false);
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.ClockSampling = sampling;
        try
        {
            // GPU work in the submission, held on the completion thread so the gap between the
            // GPU finishing and the host publishing is as wide as a slow frame makes it.
            host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, WindowedLabel, 0xA0));
            host.Completions.Enqueue(host.SubmissionSerial, () => hold.Wait(TimeSpan.FromSeconds(10)));

            var parsed = GpuReferenceClock.Read();
            var open = host.SubmissionSerial;
            host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.ClockWrite, 1, clockLabel, parsed, ReferenceClock: true));
            host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, LaterLabel, 2));
            var committed = host.SubmissionSerial != open;

            host.FlushCommandBuffer();
            WaitForGpu(host, host.LastWorkSerial);
            var gpuDone = GpuReferenceClock.Read();
            var clockWhileHeld = rig.Backing.Read64(clockLabel);
            var laterWhileHeld = rig.Backing.Read64(LaterLabel);

            hold.Set();
            WaitForCompletions(host);
            return (parsed, gpuDone, (clockWhileHeld, rig.Backing.Read64(clockLabel)),
                (laterWhileHeld, rig.Backing.Read64(LaterLabel)), host.TakeClockLabelCounters(), committed);
        }
        finally
        {
            hold.Set();
        }
    }

    private ((ulong WhileHeld, ulong AfterDrain) Owed, (ulong WhileHeld, ulong AfterDrain) Later, long Ordered)
        RecordOwedThenLater(bool orderedEndOfPipe, Action<MetalRenderHost>? later = null)
    {
        using var hold = new ManualResetEventSlim(false);
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.OrderedEndOfPipe = orderedEndOfPipe;
        try
        {
            // GPU work in the submission, so the owed label's completion has to wait for it.
            host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, WindowedLabel, 0xA0));
            var held = host.SubmissionSerial;
            host.Completions.Enqueue(held, () => hold.Wait(TimeSpan.FromSeconds(10)));

            host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, UnwindowedLabel, 1));
            if (later is null)
            {
                host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, LaterLabel, 2));
            }
            else
            {
                later(host);
            }

            host.FlushCommandBuffer();
            WaitForGpu(host, host.LastWorkSerial);

            var owedWhileHeld = rig.Backing.Read32(UnwindowedLabel);
            var laterWhileHeld = rig.Backing.Read64(LaterLabel);
            var ordered = host.TakeOrderedHostLabels();

            hold.Set();
            WaitForCompletions(host);
            return ((owedWhileHeld, rig.Backing.Read32(UnwindowedLabel)),
                (laterWhileHeld, rig.Backing.Read64(LaterLabel)), ordered);
        }
        finally
        {
            hold.Set();
        }
    }

    /// <summary>
    /// An end-of-pipe GDS store lands where its packet sits: the GDS words as the stream
    /// left them, written once the work before it is done, with no GPU drain at the packet.
    /// </summary>
    [Theory]
    [InlineData(WindowedLabel)]
    [InlineData(UnwindowedLabel)]
    public void AnEndOfPipeGdsStoreLandsWithItsSubmission(ulong destination)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        Assert.True(host.WritesGdsAtEndOfPipe);
        BinaryPrimitives.WriteUInt32LittleEndian(host.GuestMemory.GdsWords()[8..], 0xC0DE_0001);
        BinaryPrimitives.WriteUInt32LittleEndian(host.GuestMemory.GdsWords()[12..], 0xC0DE_0002);

        // Work ahead of it, so the store has something to wait for.
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, LaterLabel, 1));
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.GdsWrite32, 1, destination, GdsWordOffset: 2, GdsWordCount: 2));
        Assert.Equal(0UL, rig.Backing.Read64(destination));

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        WaitUntil(() => rig.Backing.Read64(destination) != 0);
        Assert.Equal(0xC0DE_0001u, rig.Backing.Read32(destination));
        Assert.Equal(0xC0DE_0002u, rig.Backing.Read32(destination + 4));
    }

    /// <summary>RELEASE_MEM data select 7: the add lands from the completion, once.</summary>
    [Fact]
    public void AnEndOfPipeAtomicAddLandsOnceWithItsSubmission()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        Assert.True(host.DefersEndOfPipeLabel(EndOfPipeWriteKind.AtomicAdd32));
        Assert.True(rig.Backing.TryWrite(LaterLabel, BitConverter.GetBytes(5u)));

        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, WindowedLabel, 1));
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.AtomicAdd32, 1, LaterLabel, 3));
        Assert.Equal(5u, rig.Backing.Read32(LaterLabel));

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        // The drain takes a completion off the list before it runs it.
        WaitUntil(() => rig.Backing.Read32(LaterLabel) != 5);
        Assert.Equal(8u, rig.Backing.Read32(LaterLabel));
    }

    /// <summary>
    /// A command-processor operand recorded work still owes is committed and answered as
    /// owed, so the interpreter parks instead of the worker waiting; once it lands the same
    /// question reads it now.
    /// </summary>
    [Fact]
    public void AnOwedOperandIsCommittedAndReportedUntilItLands()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;

        Assert.False(host.CommitPendingOperandWrite(LaterLabel, sizeof(ulong)));

        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 0x99));
        var recorded = host.SubmissionSerial;
        Assert.True(host.CommitPendingOperandWrite(WindowedLabel, sizeof(ulong)));
        // The commit took the label out of the buffer being recorded.
        Assert.NotEqual(recorded, host.SubmissionSerial);

        WaitForGpu(host, recorded);
        WaitForCompletions(host);
        Assert.False(host.CommitPendingOperandWrite(WindowedLabel, sizeof(ulong)));
        Assert.Equal(0x99UL, rig.Backing.Read64(WindowedLabel));
    }

    /// <summary>
    /// GET_LOD_STATS on the GPU: the report is packed from the counter bank where the
    /// packet sits, lands with the work before it rather than at parse, and a reset leaves
    /// the bank empty for the next window while a report without one keeps counting.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void AGetLodStatsReportIsPackedOnTheGpuWhereThePacketSits(bool reset)
    {
        if (!RequireDevice()) return;
        var previous = ResourceMaterializer.RecordMipStatistics;
        ResourceMaterializer.RecordMipStatistics = true;
        try
        {
            using var rig = new Rig(Device.Value);
            var host = rig.Host;
            const ulong report = GuestBase + 0x4000;
            const int entries = (int)MipStatistics.CounterWords;
            var bank = host.LodStatisticsBank;
            bank[3] = 2;
            bank[entries + 3] = 5;
            bank[2 * entries + 3] = 7;
            bank[9] = 0;
            bank[2 * entries + 9] = 0x2000000;
            Assert.True(rig.Backing.TryWrite(report + MipStatistics.ReportBytes, BitConverter.GetBytes(0xAAAAAAAAu)));

            Assert.True(((ICommandStreamHost)host).TryRecordLodStatistics(report, 0x1040, reset));
            Assert.Equal(0u, rig.Backing.Read32(report));

            host.FlushCommandBuffer();
            WaitForGpu(host, host.LastWorkSerial);
            WaitForCompletions(host);
            Assert.Equal(MipStatistics.FirstCounterOffset, rig.Backing.Read32(report));
            Assert.Equal(5u | (3u << 24), rig.Backing.Read32(report + 64 + 3 * 8));
            Assert.Equal(7u | (2u << 24), rig.Backing.Read32(report + 64 + 3 * 8 + 4));
            Assert.Equal(9u << 24, rig.Backing.Read32(report + 64 + 9 * 8));
            Assert.Equal(0xFFFFFFu, rig.Backing.Read32(report + 64 + 9 * 8 + 4));
            Assert.Equal(4u << 24, rig.Backing.Read32(report + 64 + 4 * 8));
            Assert.Equal(15u << 24, rig.Backing.Read32(report + 64 + 4 * 8 + 4));
            Assert.Equal(0xAAAAAAAAu, rig.Backing.Read32(report + MipStatistics.ReportBytes));

            bank = host.LodStatisticsBank;
            Assert.Equal(reset ? uint.MaxValue : 2u, bank[3]);
            Assert.Equal(reset ? 0u : 5u, bank[entries + 3]);
            Assert.Equal(reset ? 0u : 7u, bank[2 * entries + 3]);
            Assert.Equal(2u, bank[3 * entries]);
            Assert.Equal(1u, bank[3 * entries + 1]);
            Assert.Equal(1L, host.LodStatisticsReports);
        }
        finally
        {
            ResourceMaterializer.RecordMipStatistics = previous;
        }
    }

    /// <summary>
    /// A label the GPU has yet to land is read as written by the resource materializer
    /// (TryReadGuestWord) without a wait, while every read the guest can observe - a wait
    /// packet's operand, an ordinary read - still sees memory until the label lands.
    /// </summary>
    [Theory]
    [InlineData(EndOfPipeWriteKind.Write32, 0x1234UL, 0x1234u)]
    [InlineData(EndOfPipeWriteKind.Write64, 0x1122334455667788UL, 0x55667788u)]
    public void AnUnlandedLabelIsReadAsWrittenByTheMaterializerAndNotByTheGuest(
        EndOfPipeWriteKind kind, ulong value, uint low)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.RecordEndOfPipe(new EndOfPipeWrite(kind, 1, WindowedLabel, value));
        Assert.Equal(1, host.PendingLabelOverlayEntries);
        var completed = host.CompletedSerial;

        Assert.True(host.TryReadGuestWord(WindowedLabel, out var word));
        Assert.Equal(low, word);
        Assert.Equal(1L, host.TakeLabelOverlayHits());
        if (kind == EndOfPipeWriteKind.Write64)
        {
            Assert.True(host.TryReadGuestWord(WindowedLabel + 4, out var high));
            Assert.Equal(0x11223344u, high);
            Assert.Equal(1L, host.TakeLabelOverlayHits());
        }

        // Memory has not been written, and the materializer's read did not wait for it.
        Assert.Equal(0u, rig.Backing.Read32(WindowedLabel));
        Assert.Equal(completed, host.CompletedSerial);

        // What a wait packet or a poll reads is memory: the label has not landed.
        Span<byte> polled = stackalloc byte[sizeof(uint)];
        Assert.True(host.TryReadGuestWaitOperand(WindowedLabel, polled, commitPendingWrite: false));
        Assert.Equal(0u, BinaryPrimitives.ReadUInt32LittleEndian(polled));
        Assert.True(host.CommitPendingOperandWrite(WindowedLabel, sizeof(uint)));
        Assert.Equal(0L, host.TakeLabelOverlayHits());

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(low, rig.Backing.Read32(WindowedLabel));

        // Landed: memory answers, and the entry no longer does.
        Assert.True(host.TryReadGuestWord(WindowedLabel, out var landed));
        Assert.Equal(low, landed);
        Assert.Equal(0L, host.TakeLabelOverlayHits());
    }

    [Fact]
    public void ALaterWriteToTheSameBytesSupersedesTheOverlay()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;

        // The newest label in record order wins.
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, WindowedLabel, 1));
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, WindowedLabel, 2));
        Assert.Equal(1, host.PendingLabelOverlayEntries);
        Assert.True(host.TryReadGuestWord(WindowedLabel, out var word));
        Assert.Equal(2u, word);
        Assert.Equal(1L, host.TakeLabelOverlayHits());

        // A command-processor write after it is what a read sees from then on: the command
        // processor overlay answers, and the label overlay has stopped speaking for the bytes.
        host.RecordCommandProcessorWrite(WindowedLabel, BitConverter.GetBytes(7u), "WRITE_DATA");
        Assert.Equal(0, host.PendingLabelOverlayEntries);
        Assert.True(host.TryReadGuestWord(WindowedLabel, out word));
        Assert.Equal(7u, word);
        Assert.Equal(0L, host.TakeLabelOverlayHits());

        // And a label after a command-processor write is the newer of the two.
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, WindowedLabel, 9));
        Assert.True(host.TryReadGuestWord(WindowedLabel, out word));
        Assert.Equal(9u, word);
        Assert.Equal(1L, host.TakeLabelOverlayHits());

        // A host write or a write of unknown extent recorded after it drops it too.
        Assert.True(host.TryWriteGuest(WindowedLabel, BitConverter.GetBytes(0x55u)));
        Assert.Equal(0, host.PendingLabelOverlayEntries);
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, LaterLabel, 3));
        Assert.Equal(1, host.PendingLabelOverlayEntries);
        host.NoteUnboundedGuestMemoryWrite();
        Assert.Equal(0, host.PendingLabelOverlayEntries);
    }

    [Fact]
    public void AGuestCpuWriteOverTheLabelEndsTheOverlayAndTheReadWaits()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, WindowedLabel, 0x1234));
        Assert.Equal(1, host.PendingLabelOverlayEntries);

        // The guest wrote the bytes after the label was encoded, which the overlay cannot
        // see happen, only find out: memory no longer holds what it held at the encode.
        Assert.True(rig.Backing.TryWrite(WindowedLabel, BitConverter.GetBytes(0x5555u)));
        var completed = host.CompletedSerial;
        Assert.True(host.TryReadGuestWord(WindowedLabel, out var word));
        Assert.Equal(0L, host.TakeLabelOverlayHits());
        Assert.Equal(0, host.PendingLabelOverlayEntries);

        // The ordinary read ran: it waited for the submission, whose label then landed.
        Assert.True(host.CompletedSerial > completed);
        Assert.Equal(0x1234u, word);
    }

    [Fact]
    public void TheOverlayOptOutLeavesTheMaterializerWaitingForTheLabel()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.LabelOverlayEnabled = false;
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write32, 1, WindowedLabel, 0x1234));
        Assert.Equal(0, host.PendingLabelOverlayEntries);

        var completed = host.CompletedSerial;
        Assert.True(host.TryReadGuestWord(WindowedLabel, out var word));
        Assert.Equal(0x1234u, word);
        Assert.Equal(0L, host.TakeLabelOverlayHits());
        Assert.True(host.CompletedSerial > completed);
    }

    private static void WaitForGpu(MetalRenderHost host, ulong serial)
    {
        var deadline = Environment.TickCount64 + 10_000;
        while (host.CompletedSerial < serial)
        {
            Assert.True(Environment.TickCount64 < deadline, $"The GPU did not reach serial {serial}.");
            Thread.Sleep(1);
        }
    }

    private static void WaitUntil(Func<bool> condition)
    {
        var deadline = Environment.TickCount64 + 10_000;
        while (!condition())
        {
            Assert.True(Environment.TickCount64 < deadline, "The completion did not land.");
            Thread.Sleep(1);
        }
    }

    private static void WaitForCompletions(MetalRenderHost host)
    {
        var deadline = Environment.TickCount64 + 10_000;
        while (host.Completions.PendingCount != 0)
        {
            Assert.True(Environment.TickCount64 < deadline, "The held completions did not drain.");
            Thread.Sleep(1);
        }
    }

    private sealed class AddressSpace : IGuestAddressSpace
    {
        public ulong ProtectionPageSize => 4096;
        public bool TryProtect(ulong address, ulong size, GuestPageProtection protection) => true;
        public ulong AllocateAt(ulong desiredAddress, ulong size, bool executable = true, bool allowAlternative = true) => throw new NotSupportedException();
        public bool TryBackFixedRange(ulong address, ulong size, bool executable) => throw new NotSupportedException();
        public bool TryAllocateAtOrAbove(ulong desiredAddress, ulong size, bool executable, ulong alignment, out ulong actualAddress) => throw new NotSupportedException();
        public bool TryEnsureRangeCommitted(ulong address, ulong size) => throw new NotSupportedException();
        public bool TryAllocateGuestMemory(ulong size, ulong alignment, out ulong address) => throw new NotSupportedException();
        public bool TryFreeGuestMemory(ulong address) => throw new NotSupportedException();
    }

    // One page-aligned native block seen as guest memory at GuestBase, with one page the
    // GPU windows do not reach: the CPU can write it, the GPU has no address for it.
    private sealed class Backing : IGuestBackedSpace, ICpuMemory, IDisposable
    {
        public const ulong Size = 1024 * 1024;
        private readonly void* _block = NativeMemory.AlignedAlloc((nuint)Size, 16384);

        public Backing() => NativeMemory.Clear(_block, (nuint)Size);

        public void Dispose() => NativeMemory.AlignedFree(_block);

        private static bool Contains(ulong address, ulong length) =>
            address >= GuestBase && address - GuestBase + length <= Size;

        private static bool TouchesHole(ulong address, ulong length) =>
            address - GuestBase < HoleOffset + HoleSize && address - GuestBase + length > HoleOffset;

        public uint Read32(ulong address)
        {
            Span<byte> bytes = stackalloc byte[4];
            Assert.True(TryRead(address, bytes));
            return BinaryPrimitives.ReadUInt32LittleEndian(bytes);
        }

        public ulong Read64(ulong address)
        {
            Span<byte> bytes = stackalloc byte[8];
            Assert.True(TryRead(address, bytes));
            return BinaryPrimitives.ReadUInt64LittleEndian(bytes);
        }

        public bool TryHoldRange(ulong address, ulong size) => false;
        public bool TryHoldRangeAtOrAbove(ulong searchStart, ulong size, ulong alignment, out ulong address) { address = 0; return false; }
        public bool TryMapBacked(ulong address, ulong size, ulong backingOffset, GuestPageProtection protection, out HostViewFailure failure) { failure = HostViewFailure.None; return false; }
        public bool TryUnmapBacked(ulong address, ulong size) => false;
        public bool TryClearBacking(ulong offset, ulong size) => false;
        public bool IsBackedView(ulong address) => Contains(address, 1);
        public bool IsBackedRange(ulong address, ulong size) => Contains(address, size);
        public (ulong Base, ulong Size) BackingAliasRange => ((ulong)_block, Size);

        public bool TryResolveBackingAlias(ulong address, ulong size, out ulong alias)
        {
            alias = Contains(address, size) && !TouchesHole(address, size) ? (ulong)_block + (address - GuestBase) : 0;
            return alias != 0;
        }

        public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data) => TryWrite(address, data);
        public bool TryReadBacking(ulong address, Span<byte> data) => TryRead(address, data);

        public bool TryRead(ulong virtualAddress, Span<byte> destination)
        {
            if (!Contains(virtualAddress, (ulong)destination.Length)) return false;
            new ReadOnlySpan<byte>((byte*)_block + (virtualAddress - GuestBase), destination.Length).CopyTo(destination);
            return true;
        }

        public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source)
        {
            if (!Contains(virtualAddress, (ulong)source.Length)) return false;
            source.CopyTo(new Span<byte>((byte*)_block + (virtualAddress - GuestBase), source.Length));
            return true;
        }

        public bool CanRead(ulong address, ulong size) => Contains(address, size);
    }

    /// <summary>
    /// The synthetic occlusion dump's sixteen qword stores are recorded as one run. The same
    /// sixteen values land at the same places as sixteen separate writes would put them, the
    /// bytes between them are left alone, a parse-time read sees them before they land, and
    /// a label recorded before and one after keep their places around the run.
    /// </summary>
    [Fact]
    public void ABatchedQwordRunLandsTheSameBytesAsSeparateWritesAndInOrder()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        const ulong value = (1UL << 63) | 5;
        const ulong gap = 0xAAAAAAAAAAAAAAAAUL;
        const ulong batched = WindowedLabel + 0x1000;
        const ulong separate = WindowedLabel + 0x2000;
        for (var index = 0UL; index < 16; index++)
        {
            Assert.True(rig.Backing.TryWrite(batched + index * 16 + 8, BitConverter.GetBytes(gap)));
            Assert.True(rig.Backing.TryWrite(separate + index * 16 + 8, BitConverter.GetBytes(gap)));
        }

        // The first store of the run is at the label's address: the older label lands first,
        // then the run, then the newer label.
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, batched, 1));
        host.RecordCommandProcessorQwords(batched, 16, 16, value, "ZPASS_DUMP");
        Assert.Equal(1L, host.BatchedCommandProcessorRuns);
        Span<byte> read = stackalloc byte[sizeof(ulong)];
        for (var index = 0UL; index < 16; index++)
        {
            Assert.Equal(0UL, rig.Backing.Read64(batched + index * 16));
            Assert.True(host.TryReadGuest(batched + index * 16, read));
            Assert.Equal(value, BitConverter.ToUInt64(read));
            Assert.True(host.TryReadGuest(batched + index * 16 + 8, read));
            Assert.Equal(gap, BitConverter.ToUInt64(read));
        }

        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, batched + 16 * 5, 3));
        for (var index = 0UL; index < 16; index++)
        {
            host.RecordCommandProcessorWrite(separate + index * 16, BitConverter.GetBytes(value), "ZPASS_DUMP");
        }

        Assert.Equal((32L, 0L), host.TakeCommandProcessorWriteCounters());

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        for (var index = 0UL; index < 16; index++)
        {
            Assert.Equal(index == 5 ? 3UL : value, rig.Backing.Read64(batched + index * 16));
            Assert.Equal(value, rig.Backing.Read64(separate + index * 16));
            Assert.Equal(gap, rig.Backing.Read64(batched + index * 16 + 8));
            Assert.Equal(gap, rig.Backing.Read64(separate + index * 16 + 8));
        }
    }

    [Fact]
    public void ABatchedQwordRunInsideAnOpenComputeEncoderWaitsForItsEndAsOneRun()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        // The encoder-boundary mode, where copies land at the next boundary rather than the commit.
        host.LabelsAtCommit = false;
        const ulong dump = WindowedLabel + 0x1000;
        const ulong filled = GuestBase + 0x10000;

        _ = host.EnsureComputeEncoder();
        host.RecordCommandProcessorQwords(dump, 16, 16, 0x77, "ZPASS_DUMP");
        Assert.Equal(1L, host.BatchedCommandProcessorRuns);
        Assert.True(host.ComputeEncoderOpen);
        Assert.Equal(16, host.DeferredLabelCopyCount);
        Assert.Equal(16L, host.TakeDeferredLabelWrites());
        Assert.Equal(0UL, rig.Backing.Read64(dump));

        host.FillBufferInOrder(filled, 0x2000, 0x11);
        Assert.False(host.ComputeEncoderOpen);
        Assert.Equal(0, host.DeferredLabelCopyCount);

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        for (var index = 0UL; index < 16; index++)
        {
            Assert.Equal(0x77UL, rig.Backing.Read64(dump + index * 16));
        }
    }

    /// <summary>
    /// A run with a store no window covers is not batched: every store is recorded on its
    /// own, as before, and the ones the GPU cannot write are published by the completion.
    /// </summary>
    [Fact]
    public void AQwordRunWithAnUnwindowedStoreIsRecordedStoreByStore()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        const ulong start = GuestBase + HoleOffset - 32;

        host.RecordCommandProcessorQwords(start, 4, 16, 0x99, "ZPASS_DUMP");
        Assert.Equal((2L, 2L), host.TakeCommandProcessorWriteCounters());
        Assert.Equal(0L, host.BatchedCommandProcessorRuns);

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        for (var index = 0UL; index < 4; index++)
        {
            Assert.Equal(0x99UL, rig.Backing.Read64(start + index * 16));
        }
    }

    [Fact]
    public void TheBatchedRunVerifyFindsEveryStoreInTheOverlay()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        var before = Interlocked.Read(ref MetalRenderHost.CommandProcessorBatchVerifyMismatches);

        host.RecordCommandProcessorQwords(WindowedLabel + 0x1000, 16, 16, 0x1234, "ZPASS_DUMP");

        // Only meaningful with SHARPEMU_PERF_VERIFY=1; the invariant holds either way.
        Assert.Equal(before, Interlocked.Read(ref MetalRenderHost.CommandProcessorBatchVerifyMismatches));
        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
    }

    private sealed class Rig : IDisposable
    {
        private readonly nint _queue;
        private readonly AddressSpace _space = new();
        private readonly PageGuard _pages;

        public Rig(nint device)
        {
            _queue = MetalNative.Send(device, MetalNative.Selector("newCommandQueue"));
            _pages = new PageGuard(_space);
            Host = new MetalRenderHost(device, _queue, Backing, Backing, _pages, new MetalGpuWorkerRelay(static () => { }));
        }

        public Backing Backing { get; } = new();

        public MetalRenderHost Host { get; }

        public void Dispose()
        {
            Host.ImageCache.StopWatching();
            Host.Dispose();
            _pages.Dispose();
            Backing.Dispose();
            MetalNative.SendVoid(_queue, MetalNative.Selector("release"));
        }
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("End-of-pipe ordering runs on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
