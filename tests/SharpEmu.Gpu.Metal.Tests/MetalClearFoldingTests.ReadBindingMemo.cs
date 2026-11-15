// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A read-only binding of a range bound before is answered from the earlier binding (MetalReadBindingMemo) while
/// everything the skipped chain read still stands; each change below must send the binding down the full chain
/// again, and the binding it then gets is the full chain's.
/// </summary>
public sealed unsafe partial class MetalClearFoldingTests
{
    private const ulong MemoBuffer = GuestBase + 0x100000;
    private const ulong MemoBufferBytes = 0x100;

    private static BufferBinding BindRead(MetalRenderHost host, ulong address = MemoBuffer, ulong size = MemoBufferBytes) =>
        host.ObtainBuffer(address, size, isWritten: false);

    // Binds the range twice: the first binding is kept, the second is answered from it.
    private static BufferBinding BindTwiceAndHit(MetalRenderHost host, ulong address = MemoBuffer, ulong size = MemoBufferBytes)
    {
        var first = BindRead(host, address, size);
        var hits = host.ReadBindingMemoCounts.Hits;
        var second = BindRead(host, address, size);
        Assert.Equal(hits + 1, host.ReadBindingMemoCounts.Hits);
        Assert.Equal(first, second);
        return second;
    }

    private static void AssertMisses(MetalRenderHost host, Func<BufferBinding> bind, BufferBinding expected)
    {
        var (hits, misses, _, _) = host.ReadBindingMemoCounts;
        var binding = bind();
        Assert.Equal(hits, host.ReadBindingMemoCounts.Hits);
        Assert.Equal(misses + 1, host.ReadBindingMemoCounts.Misses);
        Assert.Equal(expected, binding);
    }

    private static Rig MemoRig()
    {
        var rig = new Rig(Device.Value);
        rig.Host.ReadBindingMemoEnabled = true;
        rig.Host.ReadBindingMemoVerify = true;
        return rig;
    }

    [Fact]
    public void ARepeatedReadOnlyBindingIsAnsweredFromTheOneBeforeAndAWrittenOneIsNot()
    {
        if (!RequireDevice()) return;
        using var rig = MemoRig();
        var host = rig.Host;
        var read = BindTwiceAndHit(host);
        Assert.Equal(1L, host.ReadBindingMemoCounts.Verified);

        var hits = host.ReadBindingMemoCounts.Hits;
        var written = host.ObtainBuffer(MemoBuffer, MemoBufferBytes, isWritten: true);
        Assert.Equal(read, written);
        Assert.Equal(hits, host.ReadBindingMemoCounts.Hits);

        host.ReadBindingMemoEnabled = false;
        Assert.Equal(read, BindRead(host));
        Assert.Equal(hits, host.ReadBindingMemoCounts.Hits);
    }

    [Fact]
    public void AReadOnlyBindingMissesOnceItsRangeIsRemapped()
    {
        if (!RequireDevice()) return;
        using var rig = MemoRig();
        var host = rig.Host;
        var before = BindTwiceAndHit(host);

        // W4: a remap elsewhere leaves it.
        rig.Remap = (MemoBuffer + 0x10000, MemoBuffer + 0x11000, 0x1000);
        AssertHits(host, () => BindRead(host), before);

        rig.Remap = (MemoBuffer, MemoBuffer + 0x1000, 0x1000);
        AssertMisses(host, () => BindRead(host), before with { Offset = before.Offset + 0x1000 });
    }

    [Fact]
    public void AReadOnlyBindingMissesWhileAStagedCopyIsBound()
    {
        if (!RequireDevice()) return;
        using var rig = MemoRig();
        var host = rig.Host;
        var before = BindTwiceAndHit(host);

        // A written range across a view edge has no window: it is staged, and the copy is listed.
        const ulong edge = GuestBase + 0x200000;
        rig.ViewBoundary = edge;
        Assert.True(host.ResolveStorageBuffer(edge - 0x80, 0x100, written: true).IsValid);
        AssertMisses(host, () => BindRead(host), before);
    }

    [Fact]
    public void AReadOnlyBindingMissesOnceAnImageOverItGoesGpuModified()
    {
        if (!RequireDevice()) return;
        using var rig = MemoRig();
        var host = rig.Host;
        var request = Rig.Request(TargetAddress, ImageRole.Texture);
        var image = host.ImageCache.FindImage(ref request);
        var before = BindTwiceAndHit(host, TargetAddress + 0x40, 0x40);

        host.ImageCache.MarkGpuWritten(image);
        Assert.True(host.ImageCache.HasGpuModifiedImageBytes(TargetAddress + 0x40, 0x40));
        AssertMisses(host, () => BindRead(host, TargetAddress + 0x40, 0x40), before);
        // While the image owns the bytes, the flush finds it each time: nothing is kept for the range.
        if (host.ImageCache.HasGpuModifiedImageBytes(TargetAddress + 0x40, 0x40))
        {
            AssertMisses(host, () => BindRead(host, TargetAddress + 0x40, 0x40), before);
        }
    }

    [Fact]
    public void AReadOnlyBindingMissesWhileADeferredLabelCopyIsHeld()
    {
        if (!RequireDevice()) return;
        using var rig = MemoRig();
        var host = rig.Host;
        using (host.BeginPreparation())
        {
            var target = rig.Target(TargetAddress);
            var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
            host.BeginRendering(Rig.PassOver(attachment.View));
            var before = BindTwiceAndHit(host);

            host.RecordCommandProcessorWrite(MemoBuffer + 0x10, BitConverter.GetBytes(0xC0FFEE11u), "WRITE_DATA");
            Assert.True(host.DeferredLabelCopyCount > 0, "the write was not held as a label copy");
            AssertMisses(host, () => BindRead(host), before);
            host.EndRendering();
        }

        host.FlushAndWaitForGpu();
    }

    // R1: the v1 memo (SHARPEMU_METAL_V2_READ_BINDING_MEMO_RANGES=0) stamps the submission: a new command buffer misses.
    [Fact]
    public void AV1ReadOnlyBindingMissesInANewCommandBuffer()
    {
        if (!RequireDevice()) return;
        using var rig = MemoRig();
        var host = rig.Host;
        host.ReadBindingMemoRanges = false;
        Assert.NotEqual(0, host.EnsureComputeEncoder());
        var before = BindTwiceAndHit(host);
        var serial = host.SubmissionSerial;

        host.FlushAndWaitForGpu();
        Assert.NotEqual(0, host.EnsureComputeEncoder());
        Assert.NotEqual(serial, host.SubmissionSerial);
        AssertMisses(host, () => BindRead(host), before);
        host.FlushAndWaitForGpu();
    }

    // Binds the range, which must miss for the reason the counter counts.
    private static BufferBinding AssertMissesBy(MetalRenderHost host, Func<MetalReadBindingMemoV2, long> cause, Func<BufferBinding> bind)
    {
        var memo = host.ReadBindingMemoV2;
        var (hits, misses, _, _) = host.ReadBindingMemoCounts;
        var counted = cause(memo);
        var binding = bind();
        Assert.Equal(hits, host.ReadBindingMemoCounts.Hits);
        Assert.Equal(misses + 1, host.ReadBindingMemoCounts.Misses);
        Assert.Equal(counted + 1, cause(memo));
        return binding;
    }

    private static BufferBinding AssertHits(MetalRenderHost host, Func<BufferBinding> bind, BufferBinding expected)
    {
        var hits = host.ReadBindingMemoCounts.Hits;
        var binding = bind();
        Assert.Equal(hits + 1, host.ReadBindingMemoCounts.Hits);
        Assert.Equal(expected, binding);
        return binding;
    }

    private static void NewCommandBuffer(MetalRenderHost host)
    {
        var serial = host.SubmissionSerial;
        host.FlushAndWaitForGpu();
        Assert.NotEqual(0, host.EnsureComputeEncoder());
        Assert.NotEqual(serial, host.SubmissionSerial);
    }

    // 1 MiB windows: MemoBuffer is the start of the second; a range across its end crosses into the third.
    private const ulong SecondWindowEnd = GuestBase + 0x200000;
    private const ulong ThirdWindowEnd = GuestBase + 0x300000;

    // W1: a new command buffer keeps the entry; the hit pins and touches its window for that command buffer (the
    // verify mode checks both on every hit), so a range crossing its edge cannot move the cut under it.
    [Fact]
    public void AReadOnlyBindingHitsInANewCommandBufferAndPinsItsWindowThere()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value, granuleMegabytes: 1);
        rig.Host.ReadBindingMemoEnabled = true;
        rig.Host.ReadBindingMemoVerify = true;
        var host = rig.Host;
        Assert.NotEqual(0, host.EnsureComputeEncoder());
        var before = BindTwiceAndHit(host);

        NewCommandBuffer(host);
        Assert.Equal(0, host.GuestMemory.TouchedWindows.Count);
        var newCommandBuffer = host.ReadBindingMemoV2.HitsNewCommandBuffer;
        AssertHits(host, () => BindRead(host), before);
        Assert.Equal(newCommandBuffer + 1, host.ReadBindingMemoV2.HitsNewCommandBuffer);
        Assert.Equal(1, host.GuestMemory.TouchedWindows.Count);

        var (moves, conflicts, _) = host.GuestMemory.WindowLayoutCounts;
        BindRead(host, SecondWindowEnd - 0x100, 0x200);
        Assert.Equal((moves, conflicts + 1), (host.GuestMemory.WindowLayoutCounts.Moves, host.GuestMemory.WindowLayoutCounts.Conflicts));
        AssertHits(host, () => BindRead(host), before);
        host.FlushAndWaitForGpu();
    }

    // W2: a range across the window's edge, bound in a command buffer that has not used the window, moves the cut and
    // retires the window: the entry misses. A cut moved between two other windows leaves it.
    [Fact]
    public void AReadOnlyBindingMissesOnceItsWindowIsRecutAndHitsWhenAnotherIs()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value, granuleMegabytes: 1);
        rig.Host.ReadBindingMemoEnabled = true;
        rig.Host.ReadBindingMemoVerify = true;
        var host = rig.Host;
        Assert.NotEqual(0, host.EnsureComputeEncoder());
        var before = BindTwiceAndHit(host);

        NewCommandBuffer(host);
        var partition = host.GuestMemory.PartitionVersion;
        var moves = host.GuestMemory.WindowLayoutCounts.Moves;
        BindRead(host, ThirdWindowEnd - 0x100, 0x200);
        Assert.Equal(moves + 1, host.GuestMemory.WindowLayoutCounts.Moves);
        Assert.NotEqual(partition, host.GuestMemory.PartitionVersion);
        AssertHits(host, () => BindRead(host), before);

        NewCommandBuffer(host);
        BindRead(host, SecondWindowEnd - 0x100, 0x200);
        Assert.Equal(moves + 2, host.GuestMemory.WindowLayoutCounts.Moves);
        var after = AssertMissesBy(host, static memo => memo.MissWindowRetired, () => BindRead(host));
        Assert.NotEqual(before.Handle, after.Handle);
        AssertHits(host, () => BindRead(host), after);
        host.FlushAndWaitForGpu();
    }

    // W3: a window idle for MetalGuestMemory.IdleWindowSerials submissions is cut back to granules: retired, so the
    // entry misses.
    [Fact]
    public void AReadOnlyBindingMissesOnceItsIdleWindowIsCutBack()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value, granuleMegabytes: 1);
        rig.Host.ReadBindingMemoEnabled = true;
        rig.Host.ReadBindingMemoVerify = true;
        var host = rig.Host;
        Assert.NotEqual(0, host.EnsureComputeEncoder());
        // Off the granule cut, so the cut back has something to do.
        BindRead(host, SecondWindowEnd - 0x100, 0x200);
        var before = BindTwiceAndHit(host);
        var partition = host.GuestMemory.PartitionVersion;

        for (var submission = 0; submission < (int)MetalGuestMemory.IdleWindowSerials + 160; submission++)
        {
            host.FlushAndWaitForGpu();
            Assert.NotEqual(0, host.EnsureComputeEncoder());
        }

        Assert.NotEqual(partition, host.GuestMemory.PartitionVersion);
        var after = AssertMissesBy(host, static memo => memo.MissWindowRetired, () => BindRead(host));
        AssertHits(host, () => BindRead(host), after);
        host.FlushAndWaitForGpu();
    }

    // C1: a frame's worth of distinct read-only ranges (Demon's Souls binds some 20k a frame) is held whole: binding
    // them all a second time hits nearly every one.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TwentyThousandDistinctReadOnlyRangesBoundTwiceHitOnTheSecondPass(bool perRange)
    {
        if (!RequireDevice()) return;
        using var rig = MemoRig();
        var host = rig.Host;
        host.ReadBindingMemoRanges = perRange;
        // Verifying each hit runs the whole chain again; this test is about capacity, the others about soundness.
        host.ReadBindingMemoVerify = false;
        const int ranges = 20000;
        for (var index = 0; index < ranges; index++)
        {
            BindRead(host, MemoBuffer + ((ulong)index * 0x80), 0x40);
        }

        var hits = host.ReadBindingMemoCounts.Hits;
        for (var index = 0; index < ranges; index++)
        {
            BindRead(host, MemoBuffer + ((ulong)index * 0x80), 0x40);
        }

        var secondPassHits = host.ReadBindingMemoCounts.Hits - hits;
        Assert.True(secondPassHits >= ranges * 99 / 100, $"second pass hits {secondPassHits} of {ranges}");
    }
}
