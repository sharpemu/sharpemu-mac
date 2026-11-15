// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// SHARPEMU_METAL_V2_UPLOAD_PRELUDE: an upload the CPU caused, needed while a render pass is open,
/// runs in a command buffer committed ahead of the open one instead of ending the pass. Every case
/// runs with the switch on and off and compares the sampled texture (read back on a queue of its
/// own) and the render target's pixels bit for bit; the counters say which arm moved what and why
/// the rest stayed. Device work is behind the exclusive device gate.
/// </summary>
public sealed class MetalUploadPreludeTests(ITestOutputHelper output)
{
    // The render target: RGBA8 64x8.
    private const uint Width = 64;
    private const uint Height = 8;
    private const ulong TargetBytes = Width * Height * 4;

    // The sampled textures: linear BC3 64x32 (a block-compressed texture is never a window view, so it
    // is uploaded, as Demon's Souls' BC1/BC4/BC7 textures are), 16 bytes a 4x4 block.
    private const uint TextureHeight = 32;
    private const ulong BlockRowBytes = Width / 4 * 16;
    private const ulong ImageBytes = BlockRowBytes * (TextureHeight / 4);
    private const uint TallHeight = 256;
    private const ulong TallBytes = BlockRowBytes * (TallHeight / 4);
    private const ulong TargetAddress = MetalPassRig.GuestBase + 0x10000;
    private const ulong FirstAddress = MetalPassRig.GuestBase + 0x20000;
    private const ulong SecondAddress = MetalPassRig.GuestBase + 0x30000;

    public enum Scene
    {
        // A CPU rewrite, the upload with the pass open, a second draw.
        Rewrite,
        // The same, then a GPU fill over the uploaded bytes recorded after the upload in the open buffer:
        // the prelude must have read them before it lands.
        RewriteThenGpuFill,
        // The texture is sampled in the open buffer before the CPU rewrites it.
        SampledEarlier,
        // A dispatch store over the source is recorded in the open buffer before the upload.
        StoreRecorded,
        // A GPU fill over part of the source is recorded in the open buffer before the upload.
        FillRecorded,
        // An alias drawn and written back over the source in the open buffer before the upload.
        AliasWrittenBack,
        // A command-processor label held over the source bytes.
        HeldLabel,
        // Two textures uploaded in one open pass, then the first rewritten and uploaded again.
        TwoTexturesThenSecondRewrite,
    }

    private sealed record Outcome(
        byte[] First, byte[] Second, byte[] Target, long Passes, long Eligible, long Hoisted, long Refused, long Failed,
        Dictionary<string, long> Blocked);

    [Fact]
    public void ACpuWrittenTextureUploadedInAnOpenPassDoesNotEndItAndHoldsTheNewBytes()
    {
        if (!RequireDevice()) return;
        var (on, off) = OnAndOff(Scene.Rewrite);
        Assert.Equal(1, on.Hoisted);
        Assert.Equal(0, on.Refused);
        Assert.Equal(1, on.Passes);
        Assert.Equal(1, off.Eligible);
        Assert.Equal(0, off.Hoisted);
        Assert.Equal(2, off.Passes);
        Assert.Equal(Pattern(ImageBytes, 0x40), on.First);
        AssertSame(on, off);
        AssertTargetHalves(on.Target);
    }

    [Fact]
    public void AGpuFillRecordedAfterTheHoistedUploadDoesNotReachIt()
    {
        if (!RequireDevice()) return;
        var (on, off) = OnAndOff(Scene.RewriteThenGpuFill);
        Assert.Equal(1, on.Hoisted);
        // The upload read the CPU's bytes, not the fill the open buffer recorded after it.
        Assert.Equal(Pattern(ImageBytes, 0x40), on.First);
        AssertSame(on, off);
    }

    [Fact]
    public void ATextureSampledEarlierInTheOpenBufferIsNotHoisted()
    {
        if (!RequireDevice()) return;
        var (on, off) = OnAndOff(Scene.SampledEarlier);
        Assert.Equal(0, on.Hoisted);
        Assert.Equal(1, on.Refused);
        Assert.Equal(1, on.Blocked["used"]);
        Assert.Equal(on.Passes, off.Passes);
        Assert.Equal(Pattern(ImageBytes, 0x40), on.First);
        AssertSame(on, off);
    }

    [Theory]
    [InlineData(Scene.StoreRecorded)]
    [InlineData(Scene.FillRecorded)]
    public void AGpuWriteOverTheSourceRecordedInTheOpenBufferIsNotHoisted(Scene scene)
    {
        if (!RequireDevice()) return;
        var (on, off) = OnAndOff(scene);
        Assert.Equal(0, on.Hoisted);
        Assert.Equal(1, on.Refused);
        Assert.Equal(1, on.Blocked["gpu_write"]);
        Assert.Equal(on.Passes, off.Passes);
        AssertSame(on, off);
    }

    [Fact]
    public void AnAliasWrittenBackOverTheSourceInTheOpenBufferIsNeverHoisted()
    {
        if (!RequireDevice()) return;
        var (on, off) = OnAndOff(Scene.AliasWrittenBack);
        Assert.Equal(0, on.Hoisted);
        Assert.Equal(1, on.Refused);
        Assert.Equal(1, on.Blocked["cause_alias"]);
        Assert.Equal(1, on.Blocked["gpu_write"]);
        Assert.Equal(on.Passes, off.Passes);
        AssertSame(on, off);
    }

    [Fact]
    public void AHeldLabelOverTheSourceLandsBeforeTheUploadReadsIt()
    {
        if (!RequireDevice()) return;
        var (on, off) = OnAndOff(Scene.HeldLabel);
        Assert.Equal(0, on.Hoisted);
        Assert.Equal(on.Passes, off.Passes);
        // The label landed before the upload read the bytes, in both arms.
        Assert.Equal(LabelValue, BitConverter.ToUInt64(on.First, (int)LabelOffset));
        AssertSame(on, off);
    }

    [Fact]
    public void TwoTexturesGoAheadOfOnePassAndASecondUploadOfOneStaysInTheBuffer()
    {
        if (!RequireDevice()) return;
        var (on, off) = OnAndOff(Scene.TwoTexturesThenSecondRewrite);
        Assert.Equal(2, on.Hoisted);
        Assert.Equal(1, on.Refused);
        Assert.Equal(1, on.Blocked["used"]);
        Assert.True(on.Passes < off.Passes, $"on={on.Passes} off={off.Passes}");
        Assert.Equal(Pattern(ImageBytes, 0x60), on.First);
        Assert.Equal(Pattern(ImageBytes, 0x50), on.Second);
        AssertSame(on, off);
    }

    private (Outcome On, Outcome Off) OnAndOff(Scene scene)
    {
        var on = Run(scene, prelude: true);
        var off = Run(scene, prelude: false);
        foreach (var (arm, outcome) in new[] { ("on", on), ("off", off) })
        {
            output.WriteLine(
                $"{scene} {arm}: passes={outcome.Passes} eligible={outcome.Eligible} hoisted={outcome.Hoisted} " +
                $"refused={outcome.Refused} failed={outcome.Failed} " +
                string.Join(' ', outcome.Blocked.Where(entry => entry.Value != 0).Select(entry => $"{entry.Key}={entry.Value}")));
        }

        Assert.Equal(0, on.Failed);
        return (on, off);
    }

    private static void AssertSame(Outcome on, Outcome off)
    {
        Assert.Equal(off.First, on.First);
        Assert.Equal(off.Second, on.Second);
        Assert.Equal(off.Target, on.Target);
    }

    // Red over the left half, green over the right.
    private static void AssertTargetHalves(byte[] pixels)
    {
        for (var x = 0u; x < Width; x++)
        {
            var texel = (int)(x * 4);
            Assert.Equal(x < Width / 2 ? (byte)0xFF : (byte)0x00, pixels[texel]);
            Assert.Equal(x < Width / 2 ? (byte)0x00 : (byte)0xFF, pixels[texel + 1]);
        }
    }

    private const ulong LabelOffset = 16;
    private const ulong LabelValue = 0x1122334455667788;

    private Outcome Run(Scene scene, bool prelude)
    {
        using var rig = new MetalPassRig(Device.Value, 1024 * 1024);
        var host = rig.Host;
        host.UploadPreludeEnabled = prelude;
        var cache = host.ImageCache;
        var store = (IGuestImageStore)cache;
        var formats = new[] { GpuFormat.R8G8B8A8Unorm };
        var red = rig.Pipeline(formats, ["1.0f, 0.0f, 0.0f, 1.0f"], -1f, 0f);
        var green = rig.Pipeline(formats, ["0.0f, 1.0f, 0.0f, 1.0f"], 0f, 1f);
        var target = rig.MakeTarget(TargetAddress, GpuFormat.R8G8B8A8Unorm, Width, Height);

        // Both textures cached and clean, their first uploads finished.
        rig.WriteRaw(FirstAddress, Pattern(ImageBytes, 0x10));
        rig.WriteRaw(SecondAddress, Pattern(ImageBytes, 0x20));
        var first = SampledRequest(FirstAddress, TextureHeight);
        var second = SampledRequest(SecondAddress, TextureHeight);
        var firstId = cache.FindImage(ref first);
        var secondId = cache.FindImage(ref second);
        Sample(host, firstId, first);
        Sample(host, secondId, second);
        host.FlushAndWaitForGpu();
        _ = host.TakePreludeCounters();
        var passes = host.RenderPassSerial;

        using (host.BeginPreparation())
        {
            if (scene == Scene.AliasWrittenBack)
            {
                // An R32Uint target over the texture's bytes: drawn, then written back for a host read, all
                // in the open buffer. The sampled texture is stale over it as an alias run.
                var integer = rig.Pipeline([GpuFormat.R32Uint], ["0x7u, 0u, 0u, 0u"]);
                var alias = rig.MakeTarget(FirstAddress, GpuFormat.R32Uint, Width, (uint)(ImageBytes / (Width * 4)));
                rig.Draw(integer, [alias]);
                cache.FlushGpuWrites(FirstAddress, ImageBytes, MetalWritebackCause.HostRead);
            }

            rig.Draw(red, [target]);
        }

        if (scene == Scene.SampledEarlier)
        {
            Sample(host, firstId, first);
        }

        if (scene != Scene.AliasWrittenBack)
        {
            CpuWrite(rig, store, FirstAddress, Pattern(ImageBytes, 0x40));
        }

        if (scene == Scene.HeldLabel)
        {
            // A label recorded over the source while the pass is open is held for the boundary; recording it
            // ends the pass, so the upload meets none, and the copy lands before the upload reads the bytes.
            // The prelude's own guard against a held copy over the source is read directly.
            host.RecordEndOfPipe(new EndOfPipeWrite(EndOfPipeWriteKind.Write64, 1, FirstAddress + LabelOffset, LabelValue));
            Assert.Equal(1, host.DeferredLabelCopyCount);
            var owner = (IMetalImageCacheOwner)host;
            Assert.True(owner.PreludeBlocks(0, FirstAddress, ImageBytes).HasFlag(MetalPreludeBlock.HeldCopy));
            Assert.False(owner.PreludeBlocks(0, SecondAddress, ImageBytes).HasFlag(MetalPreludeBlock.HeldCopy));
        }

        if (scene == Scene.StoreRecorded)
        {
            // A compute store into the source, as a dispatch of the open buffer records it.
            host.NoteGuestMemoryWrite(FirstAddress + 0x100, 0x40, MetalGpuWriteOrigin.StorageBuffer);
        }

        if (scene == Scene.FillRecorded)
        {
            host.FillGuestMemory(FirstAddress + 0x100, 0x100, 0xA5A5A5A5, allowHost: false);
            using (host.BeginPreparation())
            {
                rig.Draw(red, [target]);
            }
        }

        if (scene == Scene.TwoTexturesThenSecondRewrite)
        {
            CpuWrite(rig, store, SecondAddress, Pattern(ImageBytes, 0x50));
        }

        Sample(host, firstId, first);
        if (scene == Scene.TwoTexturesThenSecondRewrite)
        {
            Sample(host, secondId, second);
        }

        if (scene == Scene.RewriteThenGpuFill)
        {
            host.FillGuestMemory(FirstAddress + 0x100, 0x100, 0xA5A5A5A5, allowHost: false);
        }

        using (host.BeginPreparation())
        {
            rig.Draw(green, [target]);
        }

        if (scene == Scene.TwoTexturesThenSecondRewrite)
        {
            CpuWrite(rig, store, FirstAddress, Pattern(ImageBytes, 0x60));
            Sample(host, firstId, first);
            using (host.BeginPreparation())
            {
                rig.Draw(green, [target]);
            }
        }

        host.EndRendering();
        var drawnPasses = host.RenderPassSerial - passes;
        host.FlushAndWaitForGpu();
        var (eligible, hoisted, refused, failed, blocked) = host.TakePreludeCounters();
        var firstTexels = ReadTexture(cache.GetImage(firstId).Texture, TextureHeight);
        var secondTexels = ReadTexture(cache.GetImage(secondId).Texture, TextureHeight);
        var pixels = rig.Read(TargetAddress, TargetBytes);
        Assert.Equal(0, host.PendingUploadPreludes);
        return new Outcome(firstTexels, secondTexels, pixels, drawnPasses, eligible, hoisted, refused, failed, blocked);
    }

    // ---- Run sets: an image stale over several runs -------------------------------------------------------------

    private sealed record RunsOutcome(byte[] Texels, long Passes, long Hoisted, long Refused, Dictionary<string, long> Blocked);

    // Two CPU-written block rows of a 64-block-row texture are two stale runs: each goes ahead of the pass on its own.
    // With a third run an alias hand-over marked, nothing of the image goes ahead: its stale causes include
    // the alias, whatever run each cause marked.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AnImageStaleOverSeveralRunsIsHoistedPerRunOnlyWhenNoRunIsAnAliasRun(bool aliasRun)
    {
        if (!RequireDevice()) return;
        var on = RunSets(aliasRun, prelude: true);
        var off = RunSets(aliasRun, prelude: false);
        output.WriteLine($"alias_run={aliasRun} on: passes={on.Passes} hoisted={on.Hoisted} refused={on.Refused}");
        output.WriteLine($"alias_run={aliasRun} off: passes={off.Passes} hoisted={off.Hoisted} refused={off.Refused}");
        if (aliasRun)
        {
            Assert.Equal(0, on.Hoisted);
            Assert.Equal(1, on.Blocked["cause_alias"]);
        }
        else
        {
            Assert.Equal(2, on.Hoisted);
            Assert.Equal(0, on.Refused);
            Assert.Equal(1, on.Passes);
        }

        Assert.Equal(off.Texels, on.Texels);
    }

    private RunsOutcome RunSets(bool aliasRun, bool prelude)
    {
        using var rig = new MetalPassRig(Device.Value, 1024 * 1024);
        var host = rig.Host;
        host.UploadPreludeEnabled = prelude;
        var cache = host.ImageCache;
        var store = (IGuestImageStore)cache;
        var red = rig.Pipeline([GpuFormat.R8G8B8A8Unorm], ["1.0f, 0.0f, 0.0f, 1.0f"]);
        var target = rig.MakeTarget(TargetAddress, GpuFormat.R8G8B8A8Unorm, Width, Height);
        rig.WriteRaw(FirstAddress, Pattern(TallBytes, 0x10));
        var request = SampledRequest(FirstAddress, TallHeight);
        var id = cache.FindImage(ref request);
        Sample(host, id, request);
        host.FlushAndWaitForGpu();
        _ = host.TakePreludeCounters();
        var passes = host.RenderPassSerial;
        using (host.BeginPreparation())
        {
            rig.Draw(red, [target]);
        }

        const ulong row = BlockRowBytes;
        CpuWrite(rig, store, FirstAddress + (2 * row), Pattern(row, 0x70));
        CpuWrite(rig, store, FirstAddress + (40 * row), Pattern(row, 0x80));
        var image = cache.GetImage(id);
        if (aliasRun)
        {
            // Model only: guest memory is unchanged there, so the reload carries the same bytes.
            image.MarkCpuDirty(MetalImageDirtyCause.AliasWrite, 20 * row, 21 * row);
        }

        Assert.Equal(aliasRun ? 3 : 2, image.StaleRunCount);
        Sample(host, id, request);
        Assert.False(image.IsCpuDirty);
        host.EndRendering();
        var drawn = host.RenderPassSerial - passes;
        host.FlushAndWaitForGpu();
        var (_, hoisted, refused, _, blocked) = host.TakePreludeCounters();
        return new RunsOutcome(ReadTexture(image.Texture, TallHeight), drawn, hoisted, refused, blocked);
    }

    // ---- Lifetime --------------------------------------------------------------------------------------------------

    // What a prelude borrows is the open buffer's: staging lent to it is counted against that buffer and is
    // not lent again until the open buffer has completed, which is after the prelude.
    [Fact]
    public void APreludesStagingIsNotLentAgainBeforeTheOpenBufferCompletes()
    {
        if (!RequireDevice()) return;
        using var rig = new MetalPassRig(Device.Value, 1024 * 1024);
        var host = rig.Host;
        host.UploadPreludeEnabled = true;
        var owner = (IMetalImageCacheOwner)host;
        var red = rig.Pipeline([GpuFormat.R8G8B8A8Unorm], ["1.0f, 0.0f, 0.0f, 1.0f"]);
        var target = rig.MakeTarget(TargetAddress, GpuFormat.R8G8B8A8Unorm, Width, Height);
        using (host.BeginPreparation())
        {
            rig.Draw(red, [target]);
        }

        var passes = host.RenderPassSerial;
        var open = host.SubmissionSerial;
        var transient = host.RecordingTransientBytes;
        const ulong size = 64 * 1024;
        var spans = new List<MetalBufferSpan>();
        for (var index = 0; index < 2; index++)
        {
            var prelude = owner.BeginUploadPrelude();
            Assert.NotNull(prelude);
            Assert.Equal(open, prelude!.SubmissionSerial);
            Assert.NotEqual(host.CommandBuffer, prelude.CommandBuffer);
            spans.Add(host.StagingTiler!.Acquire(prelude, size));
            owner.EndUploadPrelude(prelude);
        }

        // The pass is still open and the buffer still the same; both spans are the open buffer's.
        Assert.Equal(passes, host.RenderPassSerial);
        Assert.Equal(open, host.SubmissionSerial);
        Assert.True(host.RecordingTransientBytes >= transient + (2 * size), $"transient={host.RecordingTransientBytes}");
        Assert.False(Overlap(spans[0], spans[1]));
        Assert.Equal(2, host.PendingUploadPreludes);

        // Trimming at what has completed so far frees neither: the open buffer has not even been committed.
        host.StagingTiler!.TrimStaging(host.CompletedSerial);
        var third = owner.BeginUploadPrelude()!;
        var again = host.StagingTiler.Acquire(third, size);
        owner.EndUploadPrelude(third);
        Assert.False(Overlap(again, spans[0]));
        Assert.False(Overlap(again, spans[1]));

        host.EndRendering();
        host.FlushAndWaitForGpu();
        Assert.True(host.CompletedSerial >= open);
        // The next buffer reads each prelude's status once the open serial has completed.
        _ = host.SubmissionSerial;
        Assert.Equal(0, host.PendingUploadPreludes);
        var (_, hoisted, _, failed, _) = host.TakePreludeCounters();
        Assert.Equal(3, hoisted);
        Assert.Equal(0, failed);
    }

    private static bool Overlap(MetalBufferSpan a, MetalBufferSpan b) =>
        a.Buffer == b.Buffer && a.Offset < b.Offset + b.Size && b.Offset < a.Offset + a.Size;

    // ---- Helpers ---------------------------------------------------------------------------------------------------

    private static void CpuWrite(MetalPassRig rig, IGuestImageStore store, ulong address, byte[] bytes)
    {
        rig.WriteRaw(address, bytes);
        Assert.True(store.MarkPreciseCpuWrite(address, (ulong)bytes.Length));
    }

    private static void Sample(MetalRenderHost host, ResourceSlotIdentifier id, ImageRequest request)
    {
        using (host.BeginPreparation())
        {
            Assert.NotEqual(0, host.ImageCache.AcquireTextureView(id, in request));
        }
    }

    private static byte[] Pattern(ulong size, byte seed)
    {
        var bytes = new byte[size];
        for (var index = 0; index < bytes.Length; index++)
        {
            bytes[index] = (byte)(seed + (index * 7) + (index >> 8));
        }

        return bytes;
    }

    private static ImageRequest SampledRequest(ulong address, uint height)
    {
        var bytes = BlockRowBytes * (height / 4);
        var description = ImageDescription.Create();
        description.Data = new GuestSpan(address, bytes);
        description.PixelFormat = GpuFormat.BC3UnormBlock;
        description.GuestFormat = GuestPixelFormat.Bc3UNorm;
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(Width, height, 1);
        description.Resources = new SubresourceCount(1, 1);
        description.Pitch = Width;
        description.BytesPerBlock = 16;
        description.Samples = 1;
        description.TileMode = GuestTileMode.Linear;
        description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = bytes, Pitch = Width, Height = height };
        var view = ImageViewDescription.Default with
        {
            Format = GpuFormat.BC3UnormBlock,
            Type = GpuImageViewType.Type2D,
            Aspect = GpuImageAspect.ColorBit,
            LayerCount = 1,
            Usage = GpuImageUsage.SampledBit,
        };
        return new ImageRequest(description, view, ImageRole.Texture);
    }

    // The texture's texels through a blit on a queue of its own, after the host's work has finished.
    private static byte[] ReadTexture(nint texture, uint height)
    {
        using var harness = new MetalTransferHarness(Device.Value);
        var size = BlockRowBytes * (height / 4);
        var buffer = harness.Buffer(size);
        harness.Begin();
        var blit = MetalNative.Send(harness.CommandBuffer, MetalNative.Selector("blitCommandEncoder"));
        MetalNative.SendCopyTextureToBufferOptions(
            blit,
            MetalNative.Selector(
                "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:" +
                "toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:"),
            texture, 0, 0, default, new MtlSize { Width = Width, Height = height, Depth = 1 },
            buffer, 0, (nuint)BlockRowBytes, (nuint)size, 0);
        MetalNative.SendVoid(blit, MetalNative.Selector("endEncoding"));
        harness.Submit();
        return harness.Read(buffer, 0, size);
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Upload preludes are encoded on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
