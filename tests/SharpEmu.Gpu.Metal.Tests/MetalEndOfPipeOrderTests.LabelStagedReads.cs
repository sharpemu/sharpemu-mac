// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// A read-only compute binding over held label copies reads a view the dispatch fills with them, and the copies stay held
// to the commit (MetalRenderHost.LabelStagedReads.cs).
public sealed unsafe partial class MetalEndOfPipeOrderTests
{
    private const ulong StagedRange = GuestBase + 0x10000;
    private const ulong StagedLabel = StagedRange + 0x10;
    private const uint StagedWords = 16;

    // A CPU-visible label held over the bytes a dispatch reads: no barrier, the encoder stays open, the dispatch reads the
    // label's value over the rest of guest memory, the label is still held and lands at the commit behind the shared event.
    [Fact]
    public void AHeldCpuVisibleLabelIsReadThroughAStagedViewWithoutLandingIt()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        using var probe = new StagedProbe(Device.Value);
        var host = rig.Host;
        Assert.True(host.LabelStagedReads);
        FillStagedPattern(rig);
        TakeStagedCounters(host);

        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.WriteBack64, 1, StagedLabel, 0x1122334455667788));
        var view = host.ResolveStorageBuffer(StagedRange, StagedWords * 4, written: false, computeStage: true);
        Assert.True(host.ComputeEncoderOpen);
        Assert.Equal(1, host.DeferredLabelCopyCount);
        Assert.Equal(1, host.PendingLabelStagedReads);
        Assert.False(host.GuestMemory.IsInPlace(view));

        probe.Read(host, view, slot: 0);
        Assert.True(host.ComputeEncoderOpen);
        Assert.Equal(1, host.DeferredLabelCopyCount);
        Assert.Equal(0, host.PendingLabelStagedReads);
        Assert.Equal(0L, host.TakeLabelBarriers());
        Assert.All(host.TakeLabelLandCounts(), count => Assert.Equal(0L, count));
        var counts = host.TakeLabelStagedCounts();
        Assert.Equal(1L, counts[0]);
        Assert.Equal(1L, counts[1]);
        Assert.Equal(0L, counts[2]);

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        for (var word = 0; word < StagedWords; word++)
        {
            var expected = word switch { 4 => 0x55667788u, 5 => 0x11223344u, _ => 0xA0000000u + (uint)word };
            Assert.Equal(expected, probe.Word(0, word));
        }

        Assert.Equal(0x1122334455667788UL, rig.Backing.Read64(StagedLabel));
        var lands = host.TakeLabelLandCounts();
        Assert.Equal(1L, lands[(int)MetalRenderHost.LabelLandCause.Commit - 1]);
        Assert.Equal(0L, lands[(int)MetalRenderHost.LabelLandCause.BindRead - 1]);
        Assert.Equal(1L, host.TakeLabelBarrierKinds()[0]);
    }

    // A dispatch earlier in the same encoder that stores into the range (a fill beside the label) is seen through the view.
    [Fact]
    public void AStoreEarlierInTheEncoderIsSeenThroughTheStagedView()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        using var probe = new StagedProbe(Device.Value);
        var host = rig.Host;
        FillStagedPattern(rig);
        TakeStagedCounters(host);

        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.WriteBack32, 1, StagedLabel, 0x5A5A));
        // A dispatch ahead of the reader in the same encoder stores into the range beside the label.
        Assert.True(host.GuestMemory.TryResolveInPlace(StagedRange, StagedWords * 4, out var window));
        for (var word = 0u; word < 4; word++)
        {
            probe.StoreBehindTheHost(host, window with { Offset = window.Offset + word * 4 }, 0x77);
        }

        Assert.True(host.ComputeEncoderOpen);
        var view = host.ResolveStorageBuffer(StagedRange, StagedWords * 4, written: false, computeStage: true);
        probe.Read(host, view, slot: 0);
        Assert.True(host.ComputeEncoderOpen);
        Assert.Equal(1, host.DeferredLabelCopyCount);
        Assert.Equal(0L, host.TakeLabelBarriers());

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        for (var word = 0; word < StagedWords; word++)
        {
            var expected = word switch { < 4 => 0x77u, 4 => 0x5A5Au, _ => 0xA0000000u + (uint)word };
            Assert.Equal(expected, probe.Word(0, word));
        }
    }

    // A second label to the same bytes recorded after the view was filled makes the next reader fill it again: the first
    // reader saw the first value, the next one the second; the second lands last.
    [Fact]
    public void ALabelRecordedAfterTheViewWasFilledIsSeenByTheNextReader()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        using var probe = new StagedProbe(Device.Value);
        var host = rig.Host;
        FillStagedPattern(rig);
        TakeStagedCounters(host);

        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.WriteBack32, 1, StagedLabel, 1));
        var first = host.ResolveStorageBuffer(StagedRange, StagedWords * 4, written: false, computeStage: true);
        probe.Read(host, first, slot: 0);
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.WriteBack32, 1, StagedLabel, 2));
        var second = host.ResolveStorageBuffer(StagedRange, StagedWords * 4, written: false, computeStage: true);
        Assert.Equal(first, second);
        probe.Read(host, second, slot: 1);
        Assert.True(host.ComputeEncoderOpen);
        Assert.Equal(2, host.DeferredLabelCopyCount);
        Assert.Equal(0L, host.TakeLabelBarriers());
        var counts = host.TakeLabelStagedCounts();
        Assert.Equal(2L, counts[1]);
        Assert.Equal(0L, counts[2]);

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(1u, probe.Word(0, 4));
        Assert.Equal(2u, probe.Word(1, 4));
        Assert.Equal(0xA0000005u, probe.Word(1, 5));
        Assert.Equal(2u, rig.Backing.Read32(StagedLabel));
    }

    // A written binding over the held label still lands it, ending the encoder for the shared event.
    [Fact]
    public void AWrittenBindingOverAHeldLabelStillLandsIt()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        TakeStagedCounters(host);

        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.WriteBack64, 1, StagedLabel, 9));
        _ = host.ResolveStorageBuffer(StagedRange, StagedWords * 4, written: true, computeStage: true);
        Assert.False(host.ComputeEncoderOpen);
        Assert.Equal(0, host.DeferredLabelCopyCount);
        Assert.Equal(0, host.PendingLabelStagedReads);
        Assert.Equal(1L, host.TakeLabelBarriers());
        Assert.Equal(1L, host.TakeLabelLandCounts()[(int)MetalRenderHost.LabelLandCause.BindWritten - 1]);
        Assert.Equal(0L, host.TakeLabelStagedCounts()[0]);

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(9UL, rig.Backing.Read64(StagedLabel));
    }

    // Each reason a binding over held copies is not staged is counted, and the copies land as before. Not here: gpu_source
    // (an occlusion dump is held only inside an open occlusion pass) and alias (two guest addresses of one window).
    [Theory]
    [InlineData((int)MetalRenderHost.LabelStagedRefusal.Graphics)]
    [InlineData((int)MetalRenderHost.LabelStagedRefusal.NoFence)]
    [InlineData((int)MetalRenderHost.LabelStagedRefusal.Large)]
    [InlineData((int)MetalRenderHost.LabelStagedRefusal.Unaligned)]
    [InlineData((int)MetalRenderHost.LabelStagedRefusal.StagedStorage)]
    [InlineData((int)MetalRenderHost.LabelStagedRefusal.NotInPlace)]
    [InlineData((int)MetalRenderHost.LabelStagedRefusal.DispatchWrites)]
    public void EachRefusalIsCountedAndTheCopiesLand(int refusal)
    {
        var reason = (MetalRenderHost.LabelStagedRefusal)refusal;
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        TakeStagedCounters(host);
        var address = StagedRange;
        var size = (ulong)StagedWords * 4;
        var computeStage = true;
        _ = host.EnsureComputeEncoder();
        switch (reason)
        {
            case MetalRenderHost.LabelStagedRefusal.Graphics:
                computeStage = false;
                break;
            case MetalRenderHost.LabelStagedRefusal.NoFence:
                host.EncoderOrdering = MetalRenderHost.EncoderOrderingMode.None;
                break;
            case MetalRenderHost.LabelStagedRefusal.Large:
                size = MetalRenderHost.LabelStagedReadMaxBytes + 0x1000;
                break;
            case MetalRenderHost.LabelStagedRefusal.StagedStorage:
                _ = host.StageRecordedStoresForTest(address, size);
                break;
            case MetalRenderHost.LabelStagedRefusal.NotInPlace:
                // The range runs into the page no window covers.
                address = GuestBase + HoleOffset - 0x40;
                size = 0x80;
                break;
            case MetalRenderHost.LabelStagedRefusal.DispatchWrites:
                host.LabelStagedDispatchWriteForTest = (address + 0x20, 4);
                break;
        }

        if (reason == MetalRenderHost.LabelStagedRefusal.Unaligned)
        {
            host.RecordCommandProcessorWrite(address + 0x10, [1, 2, 3, 4, 5, 6], "WRITE_DATA", causal: true);
        }
        else
        {
            host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.WriteBack32, 1, address + 0x10, 0x42));
        }

        Assert.NotEqual(0, host.DeferredLabelCopyCount);
        _ = host.ResolveStorageBuffer(address, size, written: false, computeStage);
        var refused = host.TakeLabelStagedRefusals();
        for (var index = 0; index < refused.Length; index++)
        {
            Assert.Equal(index == (int)reason ? 1L : 0L, refused[index]);
        }

        Assert.Equal(0L, host.TakeLabelStagedCounts()[0]);
        Assert.Equal(0, host.PendingLabelStagedReads);
        Assert.Equal(0, host.DeferredLabelCopyCount);
        Assert.Equal(1L, host.TakeLabelLandCounts()[(int)MetalRenderHost.LabelLandCause.BindRead - 1]);
        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
    }

    // A view is reused (no fill) by a later dispatch only while nothing could have changed its bytes: a GPU write over it,
    // a new guest submission or another command buffer makes the next reader fill it again.
    [Fact]
    public void AViewIsReusedOnlyWhileItsBytesCannotHaveChanged()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        using var probe = new StagedProbe(Device.Value);
        var host = rig.Host;
        FillStagedPattern(rig);
        TakeStagedCounters(host);
        const ulong size = StagedWords * 4;

        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.WriteBack32, 1, StagedLabel, 3));
        probe.Read(host, host.ResolveStorageBuffer(StagedRange, size, written: false, computeStage: true), slot: 0);
        // Nothing since: reused.
        probe.Read(host, host.ResolveStorageBuffer(StagedRange, size, written: false, computeStage: true), slot: 1);
        Assert.Equal(new long[] { 2, 1, 1 }, host.TakeLabelStagedCounts()[..3]);

        // A label elsewhere, outside the view: still reused.
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.WriteBack32, 1, StagedRange + 0x800, 4));
        probe.Read(host, host.ResolveStorageBuffer(StagedRange, size, written: false, computeStage: true), slot: 2);
        Assert.Equal(new long[] { 1, 0, 1 }, host.TakeLabelStagedCounts()[..3]);

        // A GPU store over the view: filled again, and it sees the store.
        host.FillBufferInOrder(StagedRange, 4, 0x99);
        probe.Read(host, host.ResolveStorageBuffer(StagedRange, size, written: false, computeStage: true), slot: 3);
        Assert.Equal(new long[] { 1, 1, 0 }, host.TakeLabelStagedCounts()[..3]);

        // A new guest submission: filled again (in a new view).
        host.NoteLabelStagedReadsSubmission();
        probe.Read(host, host.ResolveStorageBuffer(StagedRange, size, written: false, computeStage: true), slot: 4);
        Assert.Equal(new long[] { 1, 1, 0 }, host.TakeLabelStagedCounts()[..3]);
        Assert.True(host.ComputeEncoderOpen);
        Assert.Equal(0L, host.TakeLabelBarriers());

        // Another command buffer: the commit lands the copies; a label held in the next one is staged anew.
        host.FlushCommandBuffer();
        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.WriteBack32, 1, StagedLabel, 5));
        probe.Read(host, host.ResolveStorageBuffer(StagedRange, size, written: false, computeStage: true), slot: 5);
        Assert.Equal(new long[] { 1, 1, 0 }, host.TakeLabelStagedCounts()[..3]);

        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        Assert.Equal(3u, probe.Word(0, 4));
        Assert.Equal(3u, probe.Word(1, 4));
        Assert.Equal(3u, probe.Word(2, 4));
        Assert.Equal(0xA0000000u, probe.Word(2, 0));
        Assert.Equal(0x99u, probe.Word(3, 0));
        Assert.Equal(3u, probe.Word(4, 4));
        Assert.Equal(5u, probe.Word(5, 4));
        Assert.Equal(0x99u, probe.Word(5, 0));
        Assert.Equal(5u, rig.Backing.Read32(StagedLabel));
    }

    // =verify: the view is compared with the bytes a landing leaves (and the landing is made), and a view found fresh
    // later is compared with the window it stands for. A store the host was never told of makes that check fail.
    [Fact]
    public void VerifyComparesTheViewWithTheLandingAndCatchesAStaleReuse()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        using var probe = new StagedProbe(Device.Value);
        var host = rig.Host;
        host.LabelStagedReadsVerify = true;
        FillStagedPattern(rig);
        TakeStagedCounters(host);
        const ulong size = StagedWords * 4;

        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.WriteBack64, 1, StagedLabel, 0xABCDEF));
        probe.Read(host, host.ResolveStorageBuffer(StagedRange, size, written: false, computeStage: true), slot: 0);
        // The check landed the label.
        Assert.Equal(0, host.DeferredLabelCopyCount);

        // Fresh with nothing held: the binding reads the window, and the dispatch checks the view against it.
        probe.Read(host, host.ResolveStorageBuffer(StagedRange, size, written: false, computeStage: true), slot: 1);
        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        var counts = host.TakeLabelStagedCounts();
        Assert.Equal(2L, counts[4]);
        Assert.Equal(0L, counts[5]);
        Assert.Null(host.TakeLabelStagedVerifyFailure());
        Assert.Equal(0xABCDEFu, probe.Word(0, 4));
        Assert.Equal(0xABCDEFu, probe.Word(1, 4));

        // The same again, with a store into the window the host does not know about between the two dispatches.
        _ = host.EnsureComputeEncoder();
        host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.WriteBack64, 1, StagedLabel, 0x123));
        probe.Read(host, host.ResolveStorageBuffer(StagedRange, size, written: false, computeStage: true), slot: 2);
        Assert.True(host.GuestMemory.TryResolveInPlace(StagedRange, size, out var window));
        probe.StoreBehindTheHost(host, window, 0xDEAD);
        probe.Read(host, host.ResolveStorageBuffer(StagedRange, size, written: false, computeStage: true), slot: 3);
        host.FlushCommandBuffer();
        WaitForGpu(host, host.LastWorkSerial);
        WaitForCompletions(host);
        counts = host.TakeLabelStagedCounts();
        Assert.Equal(2L, counts[4]);
        Assert.Equal(1L, counts[5]);
        Assert.Contains("its reuse reads", host.TakeLabelStagedVerifyFailure());
    }

    private static void FillStagedPattern(Rig rig)
    {
        Span<byte> words = stackalloc byte[(int)StagedWords * 4];
        for (var word = 0; word < StagedWords; word++)
        {
            BitConverter.TryWriteBytes(words[(word * 4)..], 0xA0000000u + (uint)word);
        }

        Assert.True(rig.Backing.TryWrite(StagedRange, words));
    }

    private static void TakeStagedCounters(MetalRenderHost host)
    {
        _ = host.TakeLabelBarriers();
        _ = host.TakeLabelBarrierKinds();
        _ = host.TakeLabelLandCounts();
        _ = host.TakeLabelStagedCounts();
        _ = host.TakeLabelStagedRefusals();
    }

    // A dispatch of its own in the host's compute encoder, encoded where a guest dispatch would be: after the host
    // filled the views its bindings got (EncodeStagedLabelReads), it copies the view's words out.
    private sealed class StagedProbe : IDisposable
    {
        private const int SlotWords = 256;
        private static readonly nint SelPipeline = MetalNative.Selector("setComputePipelineState:");
        private static readonly nint SelSetBuffer = MetalNative.Selector("setBuffer:offset:atIndex:");
        private static readonly nint SelSetBytes = MetalNative.Selector("setBytes:length:atIndex:");
        private static readonly nint SelDispatch = MetalNative.Selector("dispatchThreadgroups:threadsPerThreadgroup:");
        private static readonly nint SelBarrier = MetalNative.Selector("memoryBarrierWithScope:");
        private readonly nint _read;
        private readonly nint _store;
        private readonly nint _output;

        public StagedProbe(nint device)
        {
            _read = SeparatorBenchPipeline(device,
                "kernel void bench(device const uint* source [[buffer(0)]], device uint* destination [[buffer(1)]], " +
                "constant uint& count [[buffer(2)]], uint index [[thread_position_in_grid]]) { if (index < count) destination[index] = source[index]; }");
            _store = SeparatorBenchPipeline(device,
                "kernel void bench(device uint* words [[buffer(0)]], constant uint& value [[buffer(1)]], " +
                "uint index [[thread_position_in_grid]]) { if (index == 0u) words[0] = value; }");
            _output = MetalNative.SendNewBuffer(device, MetalNative.Selector("newBufferWithLength:options:"), 16 * SlotWords * 4, 0);
        }

        public void Read(MetalRenderHost host, MetalGuestRange view, int slot)
        {
            var encoder = host.EncodeStagedLabelReads(host.EnsureComputeEncoder());
            var count = StagedWords;
            MetalV2Native.SendMemoryBarrierScope(encoder, SelBarrier, 1);
            MetalNative.SendVoid(encoder, SelPipeline, _read);
            MetalNative.SendSetBuffer(encoder, SelSetBuffer, view.Buffer, (nuint)view.Offset, 0);
            MetalNative.SendSetBuffer(encoder, SelSetBuffer, _output, (nuint)(slot * SlotWords * 4), 1);
            MetalNative.SendSetBytes(encoder, SelSetBytes, (nint)(&count), sizeof(uint), 2);
            MetalNative.SendDispatch(encoder, SelDispatch,
                new MtlSize { Width = 1, Height = 1, Depth = 1 }, new MtlSize { Width = 64, Height = 1, Depth = 1 });
            MetalV2Native.SendMemoryBarrierScope(encoder, SelBarrier, 1);
        }

        public void StoreBehindTheHost(MetalRenderHost host, MetalGuestRange window, uint value)
        {
            var encoder = host.EnsureComputeEncoder();
            MetalV2Native.SendMemoryBarrierScope(encoder, SelBarrier, 1);
            MetalNative.SendVoid(encoder, SelPipeline, _store);
            MetalNative.SendSetBuffer(encoder, SelSetBuffer, window.Buffer, (nuint)window.Offset, 0);
            MetalNative.SendSetBytes(encoder, SelSetBytes, (nint)(&value), sizeof(uint), 1);
            MetalNative.SendDispatch(encoder, SelDispatch,
                new MtlSize { Width = 1, Height = 1, Depth = 1 }, new MtlSize { Width = 1, Height = 1, Depth = 1 });
            MetalV2Native.SendMemoryBarrierScope(encoder, SelBarrier, 1);
        }

        public uint Word(int slot, int index) =>
            ((uint*)MetalNative.Send(_output, MetalNative.Selector("contents")))[slot * SlotWords + index];

        public void Dispose()
        {
            var release = MetalNative.Selector("release");
            MetalNative.SendVoid(_read, release);
            MetalNative.SendVoid(_store, release);
            MetalNative.SendVoid(_output, release);
        }
    }
}
