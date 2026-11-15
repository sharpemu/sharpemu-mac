// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

// Run sets on the cache: the cases the run-set suite above leaves open - a full set under a CPU read, a whole and a
// partial writeback of an image that is stale over two runs, and a whole-range device alias copy into an image that
// owns a published run. Each runs with run sets (or device publishing) on and off and must end bit-identical.
public sealed unsafe partial class MetalImageAliasTests
{
    // What guest memory holds when every alias's run came from the alias and the rest from the large texture.
    private static byte[] ExpectedAfterDraws((ulong Offset, uint Rows)[] draws)
    {
        var expected = MetalTileOracle.Pattern((int)LargeSize, 0x5A);
        byte seed = 0x11;
        foreach (var (offset, rows) in draws)
        {
            var size = (int)(rows * LargeSide * 4);
            MetalTileOracle.Pattern(size, seed).CopyTo(expected, (int)offset);
            seed += 0x22;
        }

        return expected;
    }

    // Five aliases apart from each other on a full set, then a CPU read: the fifth merges only by publishing the
    // image first, so guest memory holds each alias's bytes over its run and the large texture's elsewhere, and
    // nothing is merged over bytes only the texture held.
    [Fact]
    public void AFifthDisjointAliasOnAFullSetLeavesGuestMemoryAndTexturesExactlyAsWithRunSetsOff()
    {
        if (!RequireDevice()) return;
        (ulong Offset, uint Rows)[] draws = [(0, 4), (0x800, 4), (0x1000, 4), (0x1800, 4), (0x2000, 4)];
        var outcomes = OnAndOff(draws, cpuReadFirst: true);
        var expected = ExpectedAfterDraws(draws);
        Assert.Equal(MetalCachedImage.MaxRuns, (int)outcomes[0].Runs.SetsMax);
        Assert.True(outcomes[0].Runs.Merged >= 1);
        Assert.Equal(0, outcomes[0].Runs.Unpublished);
        foreach (var outcome in outcomes)
        {
            Assert.Equal(expected, outcome.GuestBeforeRead);
            Assert.Equal(expected, outcome.Guest);
            Assert.Equal(expected, outcome.Large);
        }
    }

    // A whole writeback (the host reads all of it) of a GPU-modified image that is stale over two runs: guest memory
    // over each run is the alias's newer bytes, never the large texture's older ones.
    [Fact]
    public void AWholeWritebackOverTwoStaleRunsNeverReplacesThemWithOlderTexels()
    {
        if (!RequireDevice()) return;
        (ulong Offset, uint Rows)[] draws = [(0x1000, AliasRows), (0x3000, AliasRows)];
        var outcomes = OnAndOff(draws, cpuReadFirst: true);
        var expected = ExpectedAfterDraws(draws);
        foreach (var outcome in outcomes)
        {
            Assert.Equal(expected, outcome.GuestBeforeRead);
            Assert.Equal(expected, outcome.Guest);
            Assert.Equal(expected, outcome.Large);
        }
    }

    // A run writeback: the host reads only a range over one stale run (and the bytes round it), and the range is
    // flushed alone. Whatever the flushed range holds over a run is the alias's, with run sets on and off alike, and
    // the whole of guest memory is identical between the two arms.
    private sealed record PartialFlushOutcome(byte[] Guest, byte[] Large, byte[] GuestFinal, string Text);

    private static PartialFlushOutcome PartialFlushOverTwoRuns(bool runSets, ulong flushOffset, ulong flushSize)
    {
        using var rig = new Rig(Device.Value);
        rig.Cache.AliasRunSetsEnabled = runSets;
        var (image, _, _) = GpuWrittenLarge(rig);
        var first = DrawAlias(rig, 0x1000, AliasRows, 0x11);
        var second = DrawAlias(rig, 0x3000, AliasRows, 0x33);
        var text = $"runs={image.StaleRunCount} gpu={image.IsGpuModified} first_gpu={first.IsGpuModified} " +
            $"second_gpu={second.IsGpuModified}";
        rig.Cache.FlushGpuWrites(ImageAddress + flushOffset, flushSize, MetalWritebackCause.HostRead);
        rig.Harness.Submit();
        rig.Harness.Begin();
        var guest = rig.ReadGuest(ImageAddress, LargeSize);
        rig.Acquire(ImageAddress, LargeSide, ImageRole.Texture);
        rig.Harness.Submit();
        rig.Harness.Begin();
        var large = ReadTexture(rig, image);
        rig.Cache.FlushGpuWrites(ImageAddress, LargeSize, MetalWritebackCause.HostRead);
        rig.Harness.Submit();
        rig.Harness.Begin();
        var final = rig.ReadGuest(ImageAddress, LargeSize);
        Assert.Equal(0, rig.Cache.AliasRunRefreshRefused);
        Assert.Equal(0, rig.Cache.AliasRunCounters.Unpublished);
        return new PartialFlushOutcome(guest, large, final, text + " " + rig.Cache.TakeAliasTransitionText());
    }

    [Theory]
    [InlineData(0x1000UL, 0x1000UL, "exactly the first run")]
    [InlineData(0x0800UL, 0x2000UL, "the first run and the bytes round it")]
    [InlineData(0x3000UL, 0x1000UL, "exactly the second run")]
    [InlineData(0x1000UL, 0x3000UL, "both runs and the gap")]
    public void ARunWritebackOverStaleRunsNeverReplacesThemWithOlderTexels(ulong flushOffset, ulong flushSize, string name)
    {
        if (!RequireDevice()) return;
        var on = PartialFlushOverTwoRuns(runSets: true, flushOffset, flushSize);
        var off = PartialFlushOverTwoRuns(runSets: false, flushOffset, flushSize);
        output.WriteLine($"{name}: on: {on.Text}");
        output.WriteLine($"{name}: off: {off.Text}");
        // Only the flushed range is promised right away (run sets on leaves the rest of the image GPU-modified,
        // off wrote it back whole at the hand-over); once everything is flushed the arms agree exactly.
        var expected = ExpectedAfterDraws([(0x1000, AliasRows), (0x3000, AliasRows)]);
        Assert.Equal(off.GuestFinal, on.GuestFinal);
        Assert.Equal(expected, on.GuestFinal);
        Assert.Equal(off.Large, on.Large);
        Assert.Equal(expected, on.Large);

        // Over each run the flushed range covers, guest memory holds the alias's bytes. The alias's drawn pattern is
        // the newest; the large texture's older pattern must not be there.
        var older = MetalTileOracle.Pattern((int)LargeSize, 0x5A);
        var runs = new[] { (Offset: 0x1000UL, Seed: (byte)0x11), (Offset: 0x3000UL, Seed: (byte)0x33) };
        foreach (var arm in new[] { on, off })
        {
            foreach (var (runStart, seed) in runs)
            {
                var from = Math.Max(runStart, flushOffset);
                var to = Math.Min(runStart + SmallSize, flushOffset + flushSize);
                if (to <= from)
                {
                    continue;
                }

                var drawn = MetalTileOracle.Pattern((int)SmallSize, seed);
                var guest = arm.Guest.AsSpan((int)from, (int)(to - from)).ToArray();
                var stale = MetalTileOracle.Pattern((int)LargeSize, 0xC3).AsSpan((int)from, (int)(to - from)).ToArray();
                output.WriteLine($"run 0x{runStart:X}: drawn={guest.SequenceEqual(drawn.AsSpan((int)(from - runStart), (int)(to - from)).ToArray())} " +
                    $"older={guest.SequenceEqual(older.AsSpan((int)from, (int)(to - from)).ToArray())} stale={guest.SequenceEqual(stale)}");
                Assert.Equal(drawn.AsSpan((int)(from - runStart), (int)(to - from)).ToArray(), guest);
                Assert.NotEqual(older.AsSpan((int)from, (int)(to - from)).ToArray(), guest);
            }
        }
    }

    // ---- a whole-range device alias copy into a GPU-modified target that owns a published run ----

    // Depth values a 32-bit depth plane keeps bit for bit (floats in [0, 1)), as guest bytes.
    private static byte[] DepthPattern(int size, int seed)
    {
        var bytes = new byte[size];
        for (var index = 0; index < size / 4; index++)
        {
            BitConverter.TryWriteBytes(bytes.AsSpan(index * 4), ((index * 31 + seed) % 1000) / 1000f);
        }

        return bytes;
    }

    private const uint DepthWidth = 128;
    private const uint DepthHeight = 64;

    // A 32-bit depth plane (D32) or its single-channel R32 alias over the same bytes, Depth tile mode: the pair the
    // device hands over (MetalImageTransfer.IsDepth32ToColorAlias), switched off by DeviceAliasPublishEnabled.
    private static ImageRequest Depth32Request(ImageRole role)
    {
        var depthRole = role == ImageRole.DepthTarget;
        Assert.True(TileGeometry.TryGetDepthSize(DepthWidth, DepthHeight, 0, GuestDepthFormat.Z32Float,
            GuestStencilFormat.Invalid, false, out _, out _, out var depthSize));
        var pitch = TileGeometry.DepthPitch(DepthWidth, 4);
        var description = ImageDescription.Create();
        description.Data = new GuestSpan(ImageAddress, depthSize.Size);
        var format = depthRole ? GpuFormat.D32Sfloat : GpuFormat.R32Sfloat;
        description.PixelFormat = format;
        description.GuestFormat = GuestPixelFormat.Bits32Float;
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(DepthWidth, DepthHeight, 1);
        description.Resources = new SubresourceCount(1, 1);
        description.Pitch = pitch;
        description.BytesPerBlock = 4;
        description.Samples = 1;
        description.TileMode = GuestTileMode.Depth;
        description.MipLayout[0] = new MipLevelLayout
        {
            Offset = 0, Size = depthSize.Size, Pitch = pitch, Height = DepthHeight,
        };
        var view = ImageViewDescription.Default with
        {
            Format = format,
            Type = GpuImageViewType.Type2D,
            Aspect = depthRole ? GpuImageAspect.DepthBit : GpuImageAspect.ColorBit,
            LayerCount = 1,
            Usage = depthRole ? GpuImageUsage.DepthStencilAttachmentBit : GpuImageUsage.SampledBit,
        };
        return new ImageRequest(description, view, role);
    }

    private sealed record DeviceAliasOutcome(
        byte[] GuestBeforeFlush, byte[] GuestAfterFlush, byte[] Target, long DevicePublishes, bool TargetGpuModified,
        int TargetPublishedRuns, bool PublishedOverCopy, byte[] Newer);

    // The D32 plane S is GPU-written; its R32 alias T (the target) is GPU-modified too and owns a published run (the
    // first half of it, as a partial writeback would leave). S is then written again - its texture gets new bytes,
    // guest memory still holds older ones - and T is sampled: the whole range is copied S -> T.
    private static DeviceAliasOutcome RunDeviceAliasIntoPublishedTarget(bool deviceOn, bool targetOwns = true)
    {
        using var rig = new Rig(Device.Value);
        rig.Cache.DepthTwins = false;
        rig.Cache.DeviceAliasPublishEnabled = deviceOn;
        var depthRequest = Depth32Request(ImageRole.DepthTarget);
        var textureRequest = Depth32Request(ImageRole.Texture);
        var size = depthRequest.Description.Data.Size;

        rig.WriteGuest(ImageAddress, DepthPattern((int)size, 0x5A));
        var depthId = rig.Cache.FindImage(ref depthRequest);
        rig.Cache.AcquireDepthTargetView(depthId, in depthRequest);
        var source = rig.Cache.GetImage(depthId);
        Assert.True(source.IsGpuModified);
        rig.Harness.Submit();
        rig.Harness.Begin();

        var targetId = rig.Cache.FindImage(ref textureRequest);
        Assert.NotEqual(depthId, targetId);
        rig.Cache.AcquireTextureView(targetId, in textureRequest);
        var target = rig.Cache.GetImage(targetId);
        rig.Harness.Submit();
        rig.Harness.Begin();

        // The target is GPU-modified and has published its first half.
        if (targetOwns)
        {
            target.MarkGpuModified();
            target.MarkPublished(0, size / 2);
            Assert.True(target.IsPublished(0, size / 2));
        }

        // The source's texture gets newer bytes than guest memory holds, and is the later writer.
        var newer = DepthPattern((int)size, 0xC3);
        rig.Harness.Transfer.EncodeGuestToTexture(
            rig.Harness, source.Description, new SubresourceRange(0, 1, 0, 1),
            new MetalGuestRange(rig.Harness.Buffer(newer), 0, size, 0), default, source.Texture, 0, size);
        rig.Harness.Submit();
        rig.Harness.Begin();
        // The texture's own texels in guest layout: bytes of the 64 KiB tile block that carry no texel come back as
        // zero, so the newer guest bytes cannot be compared whole.
        var newerBack = ReadTexture(rig, source);
        source.MarkGpuModified();
        rig.WriteGuest(ImageAddress, DepthPattern((int)size, 0x33));
        _ = rig.Cache.TakeDeviceAliasPublishes();

        rig.Cache.AcquireTextureView(targetId, in textureRequest);
        rig.Harness.Submit();
        rig.Harness.Begin();

        var publishes = rig.Cache.TakeDeviceAliasPublishes();
        var before = rig.ReadGuest(ImageAddress, size);
        var runs = target.PublishedRuns.Length;
        var publishedOverCopy = target.IsPublished(0, 1) || target.IsPublished(size - 1, size);
        var modified = target.IsGpuModified;
        rig.Cache.FlushGpuWrites(ImageAddress, size, MetalWritebackCause.HostRead);
        rig.Harness.Submit();
        rig.Harness.Begin();
        var after = rig.ReadGuest(ImageAddress, size);
        var targetBytes = ReadTexture(rig, target);
        return new DeviceAliasOutcome(before, after, targetBytes, publishes, modified, runs, publishedOverCopy, newerBack);
    }

    // Texel bytes of the source's newer content (its non-zero bytes), the part of a 64 KiB tile block that is written
    // back; the rest reads back as whatever the buffer held.
    private static int[] TexelIndices(byte[] newer) =>
        Enumerable.Range(0, newer.Length).Where(index => newer[index] != 0).ToArray();

    private static int Differing(int[] texels, byte[] expected, byte[] actual) =>
        texels.Count(index => actual[index] != expected[index]);

    // After the copy the target has no published run over the copied bytes, and guest memory after the flush holds
    // the source's newer texels - the same as the path where the device copy is off and the target owns nothing
    // (the baseline: with the copy off, a target that owns older texels is the next test).
    [Fact]
    public void ADeviceCopyIntoAGpuModifiedTargetLeavesNoPublishedRunOverTheCopiedBytes()
    {
        if (!RequireDevice()) return;
        var on = RunDeviceAliasIntoPublishedTarget(deviceOn: true);
        var baseline = RunDeviceAliasIntoPublishedTarget(deviceOn: false, targetOwns: false);
        output.WriteLine($"on: publishes={on.DevicePublishes} gpu={on.TargetGpuModified} runs={on.TargetPublishedRuns} " +
            $"| baseline: publishes={baseline.DevicePublishes} gpu={baseline.TargetGpuModified} runs={baseline.TargetPublishedRuns}");

        Assert.Equal(1, on.DevicePublishes);
        Assert.Equal(0, baseline.DevicePublishes);
        Assert.Equal(0, on.TargetPublishedRuns);
        Assert.False(on.PublishedOverCopy);
        Assert.True(on.TargetGpuModified);

        // The copy did not touch guest memory (the source still owes it its bytes).
        var size = on.GuestAfterFlush.Length;
        Assert.Equal(DepthPattern(size, 0x33), on.GuestBeforeFlush);
        var texels = TexelIndices(on.Newer);
        Assert.True(texels.Length > size / 8);
        foreach (var arm in new[] { on, baseline })
        {
            Assert.Equal(0, Differing(texels, on.Newer, arm.Newer));
            Assert.Equal(0, Differing(texels, on.Newer, arm.GuestAfterFlush));
            Assert.Equal(0, Differing(texels, on.Newer, arm.Target));
        }
    }

    // Looks wrong, pinned: with the device copy off, a target that is GPU-modified with older texels and a
    // published first half, sampled over a source that is GPU-modified with newer ones (a state the cache's own
    // transitions do not reach: taking GPU ownership clears the other image's), is not reloaded from the guest bytes
    // the source was just published into, and its flush writes its older texels over them. The device copy leaves
    // no such state (the test above).
    [Fact]
    public void WithTheDeviceCopyOffATargetOwningOlderTexelsOverwritesTheNewerGuestBytesAtTheFlush()
    {
        if (!RequireDevice()) return;
        var off = RunDeviceAliasIntoPublishedTarget(deviceOn: false);
        var texels = TexelIndices(off.Newer);
        Assert.Equal(0, off.DevicePublishes);
        // Published into guest memory by the sync, before the flush.
        Assert.Equal(0, Differing(texels, off.Newer, off.GuestBeforeFlush));
        // Then the target's older texels replace them.
        Assert.True(Differing(texels, off.Newer, off.GuestAfterFlush) > texels.Length / 2);
        Assert.Equal(0, Differing(texels, off.Target, off.GuestAfterFlush));
    }
}
