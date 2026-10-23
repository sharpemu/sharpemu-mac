// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// Single-copy textures (MetalSingleCopy) on a real chunked backing, a real guest view
/// mapping it, the windows over the GPU alias and a real image cache: a released chunk
/// is emptied only once its bytes are verified against the texture, and every way back to
/// it - a host read, a partial host write, a fill, a GPU binding, retirement, a fault on
/// several threads at once - gives the guest bytes back exactly. The page guard is a fake
/// (no real protection), so faults are driven through the handler the fault path calls.
/// </summary>
[System.Runtime.Versioning.SupportedOSPlatform("macos")]
public sealed unsafe partial class MetalSingleCopyTests(ITestOutputHelper output)
{
    private const ulong Chunk = 64 * 1024;
    private const ulong Mib = 1024 * 1024;
    private const ulong BackingSize = 8 * Mib;
    private const ulong GranuleBytes = Mib;
    private const ulong MaxWindowBytes = 4 * Mib;
    private const uint Side = 256;

    // Backing offset 1 MiB: its own granule window, with room beside it in that window.
    private const ulong ImageOffset = Mib;

    private static readonly nint SelBlit = MetalNative.Selector("blitCommandEncoder");
    private static readonly nint SelEnd = MetalNative.Selector("endEncoding");
    private static readonly nint SelCopy =
        MetalNative.Selector("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:");

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

    // The guest space as the cache and the windows see it: one view of the backing.
    private sealed class ViewsBacking(SharedBackingViews views) : IGuestBackedSpace, ICpuMemory
    {
        public bool TryHoldRange(ulong address, ulong size) => false;
        public bool TryHoldRangeAtOrAbove(ulong searchStart, ulong size, ulong alignment, out ulong address) { address = 0; return false; }
        public bool TryMapBacked(ulong address, ulong size, ulong backingOffset, GuestPageProtection protection, out HostViewFailure failure) { failure = HostViewFailure.None; return false; }
        public bool TryUnmapBacked(ulong address, ulong size) => false;
        public bool TryClearBacking(ulong offset, ulong size) => false;
        public ulong BackingChunkBytes => views.ChunkBytes;
        public GuestSingleCopy? SingleCopy => views.SingleCopy;
        public object? BackingAliasSnapshot => views.AliasSnapshot;
        public bool IsBackingAliasSnapshotCurrent(object snapshot) => views.IsAvailable && ReferenceEquals(snapshot, views.AliasSnapshot);
        public bool IsBackedView(ulong address) => views.Contains(address, 1);
        public bool IsBackedRange(ulong address, ulong size) => views.Contains(address, size);
        public (ulong Base, ulong Size) BackingAliasRange => (views.AliasBase, views.Size);
        public bool TryResolveBackingAlias(ulong address, ulong size, out ulong alias) => views.TryResolveAlias(address, size, out alias);
        public bool TryResolveBackingAliasSpan(ulong address, ulong size, out ulong alias) => views.TryResolveAliasSpan(address, size, out alias);
        public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data) => views.TryWriteBacking(address, data);
        public bool TryReadBacking(ulong address, Span<byte> data) => views.TryReadBacking(address, data);
        public bool TryRead(ulong virtualAddress, Span<byte> destination) => views.TryReadBacking(virtualAddress, destination);
        public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source) => views.TryWriteBacking(virtualAddress, source);
        public bool CanRead(ulong address, ulong size) => views.Contains(address, size);
    }

    private sealed class Owner(MetalTransferHarness harness) : IMetalImageCacheOwner
    {
        public nint Device => harness.Device;
        public ulong CurrentSerial => harness.SubmissionSerial;
        public ulong CompletedSerial => harness.CompletedSerial;
        public IMetalSubmissionContext BeginTransfer(MetalTransferCause cause) => harness;
        public bool HasPendingGpuWrite(ulong address, ulong size) => false;
        public void FillGuestRange(ulong address, ulong size, uint value) { }
        public void NoteGuestWrite(ulong address, ulong size) { }
        public void WaitForGpu() { }
        public void ClearTextureColor(nint texture, ulong pixelFormat, in SubresourceRange range, uint width, uint height, in GpuClearColor color) { }

        public void ClearTextureDepthStencil(nint texture, ulong pixelFormat, in SubresourceRange range, uint width, uint height,
            float depth, byte stencil, bool clearDepth, bool clearStencil)
        {
        }
    }

    private sealed class Rig : IDisposable
    {
        private readonly IHostViewMemory _host = SharpEmu.Host.MacOS.MacOSHost.CreateViewMemory();
        private readonly ulong _hole;
        private readonly ulong _holeSize = BackingSize + Chunk;
        private readonly PageGuard _pages;
        private readonly bool _savedSentinel = MetalSingleCopy.SentinelMode;
        private readonly bool _savedShadow = MetalSingleCopy.ShadowMode;
        private readonly bool _savedLinearViews = MetalLinearWindowView.Enabled;
        private long _now = 1;
        private ulong _serial;
        private ulong _completed;

        public Rig(nint device, bool sentinel = false, bool shadow = false)
        {
            MetalSingleCopy.SentinelMode = sentinel;
            MetalSingleCopy.ShadowMode = shadow;
            // A linear texture would be a window view, which reads guest memory in place and
            // so is never single-copy; the linear cases here are owned textures.
            MetalLinearWindowView.Enabled = false;
            var chunked = typeof(SharpEmu.Host.MacOS.MacOSHost).Assembly.GetType("SharpEmu.Host.MacOS.MacHostViews")!
                .GetField("ChunkedBackingForTests", BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Static)!;
            var saved = chunked.GetValue(null);
            chunked.SetValue(null, true);
            try
            {
                Views = new SharedBackingViews(_host, BackingSize);
            }
            finally
            {
                chunked.SetValue(null, saved);
            }

            Assert.True(Views.IsAvailable);
            Assert.Equal(Chunk, Views.ChunkBytes);
            Assert.NotNull(Views.SingleCopy);

            // A 64 KiB-aligned hole the view is mapped into, as the guest maps direct memory.
            ulong owned = 0;
            Assert.Equal(0, mach_vm_map(mach_task_self(), ref owned, _holeSize, Chunk - 1, 1, 0, 0, 0, 3, 7, 1));
            _hole = owned;
            Assert.True(_host.UnmapView(owned, _holeSize));
            Address = owned;
            Assert.True(Views.TryMapReservedRange(Address, BackingSize, 0, HostPageProtection.ReadWrite, out _));

            Backing = new ViewsBacking(Views);
            Harness = new MetalTransferHarness(device);
            Harness.Begin();
            _pages = new PageGuard(new AddressSpace());
            Memory = new MetalGuestMemory(device, Backing, Backing, GranuleBytes, MaxWindowBytes, singleCopy: true);
            Assert.NotNull(Memory.GpuAlias);
            Cache = new MetalImageCache(new Owner(Harness), Harness.Transfer, Memory, _pages, Backing);
            // Exact textures: a lossy one never round-trips, so it is never released.
            Cache.LossyTextures = false;
            Configure(retireFrames: 0, retireSeconds: 0);
            SingleCopy = new MetalSingleCopy(device, Memory, Backing, _pages, () => _completed, new MetalSingleCopy.Settings
            {
                Idle = TimeSpan.Zero,
                Grace = TimeSpan.Zero,
                ScanInterval = TimeSpan.Zero,
            });
            Cache.SingleCopy = SingleCopy;
            Memory.BeginSubmission(++_serial, _completed);
        }

        public SharedBackingViews Views { get; }
        public ViewsBacking Backing { get; }
        public ulong Address { get; }
        public MetalTransferHarness Harness { get; }
        public MetalGuestMemory Memory { get; }
        public MetalImageCache Cache { get; }
        public MetalSingleCopy SingleCopy { get; }
        public ulong ImageAddress => Address + ImageOffset;

        public void Configure(uint retireFrames, double retireSeconds) =>
            Cache.ConfigureMemoryPolicy(
                MetalImageBudgetSettings.Defaults(24UL * 1024 * Mib) with { FixedBudget = 64UL * 1024 * Mib },
                new MetalImageRetirementSettings { Frames = retireFrames, Seconds = retireSeconds },
                () => default,
                () => _now,
                () => 0);

        public void Advance(double seconds) => _now += (long)(seconds * Stopwatch.Frequency);

        // The command buffer being recorded runs; the windows' next submission begins and
        // completes at once (the GPU here has nothing else in flight).
        public void Submit()
        {
            Harness.Submit();
            Harness.Begin();
            Memory.BeginSubmission(++_serial, _completed);
            _completed = _serial;
        }

        public byte[] Raw(ulong address, ulong size) =>
            new ReadOnlySpan<byte>((void*)(Views.AliasBase + (address - Address)), (int)size).ToArray();

        public byte[] Read(ulong address, ulong size)
        {
            var bytes = new byte[size];
            Assert.True(Views.TryReadBacking(address, bytes));
            return bytes;
        }

        public int ChunkOf(ulong address) => (int)((address - Address) / Chunk);

        public ResourceSlotIdentifier Acquire(ImageRequest request)
        {
            var identifier = Cache.FindImage(ref request);
            Cache.AcquireTextureView(identifier, request);
            return identifier;
        }

        // Uploads the image from these guest bytes and runs the protocol until its chunks
        // are emptied: the scan sees it, arms it, the punch lands with the next submission,
        // and once that has completed the verify empties them.
        public MetalCachedImage Release(ImageRequest request, byte[] bytes, int expectedChunks)
        {
            Assert.True(Views.TryWriteBacking(request.Description.Data.Address, bytes));
            var identifier = Acquire(request);
            Submit();
            var image = Cache.GetImage(identifier);
            Assert.True(image.IsWatched);
            Assert.False(image.IsCpuDirty);
            Cache.Collect();
            Cache.Collect();
            Assert.NotNull(image.SingleCopyRecord);
            Submit();
            Cache.Collect();
            Assert.Equal((ulong)expectedChunks * Chunk, SingleCopy.Counts.EmptiedBytes);
            return image;
        }

        public void Dispose()
        {
            Cache.StopWatching();
            Cache.SingleCopy = null;
            SingleCopy.Dispose();
            Cache.Dispose();
            Memory.Dispose();
            _pages.Dispose();
            Harness.Dispose();
            Views.Dispose();
            if (Views.CleanupComplete)
            {
                _host.FreeHole(_hole, _holeSize);
            }

            MetalSingleCopy.SentinelMode = _savedSentinel;
            MetalSingleCopy.ShadowMode = _savedShadow;
            MetalLinearWindowView.Enabled = _savedLinearViews;
        }
    }

    private static ImageRequest Request(ulong address, GuestTileMode tileMode, uint width = Side, uint pitch = Side)
    {
        var description = ImageDescription.Create();
        var size = tileMode == GuestTileMode.Linear
            ? (ulong)pitch * Side * 4
            : TileGeometry.TextureTotalSize(GuestPixelFormat.Bits8_8_8_8UNorm, width, Side, 1, 1, tileMode, volume: false).Size;
        description.Data = new GuestSpan(address, size);
        description.PixelFormat = GpuFormat.R8G8B8A8Unorm;
        description.GuestFormat = GuestPixelFormat.Bits8_8_8_8UNorm;
        description.Type = GuestImageType.Color2D;
        description.Extent = new GpuExtent3D(width, Side, 1);
        description.Resources = new SubresourceCount(1, 1);
        description.Pitch = tileMode == GuestTileMode.Linear
            ? pitch
            : TileGeometry.TexturePitch(GuestPixelFormat.Bits8_8_8_8UNorm, width, tileMode);
        description.BytesPerBlock = 4;
        description.Samples = 1;
        description.TileMode = tileMode;
        description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = size, Pitch = description.Pitch, Height = Side };
        var view = ImageViewDescription.Default with
        {
            Format = GpuFormat.R8G8B8A8Unorm,
            Type = GpuImageViewType.Type2D,
            Aspect = GpuImageAspect.ColorBit,
            LayerCount = 1,
            Usage = GpuImageUsage.SampledBit,
        };
        return new ImageRequest(description, view, ImageRole.Texture);
    }

    private static byte[] Pattern(ulong size, int seed)
    {
        var bytes = new byte[size];
        new Random(seed).NextBytes(bytes);
        return bytes;
    }

    private static void Copy(MetalTransferHarness harness, nint source, ulong sourceOffset, nint destination, ulong size)
    {
        var blit = MetalNative.Send(harness.CommandBuffer, SelBlit);
        MetalNative.SendCopyBuffer(blit, SelCopy, source, (nuint)sourceOffset, destination, 0, (nuint)size);
        MetalNative.SendVoid(blit, SelEnd);
    }

    private static long Field(string describe, string name)
    {
        var token = describe.Split(' ').Single(part => part.StartsWith(name + "=", StringComparison.Ordinal));
        return (long)double.Parse(token[(name.Length + 1)..], System.Globalization.CultureInfo.InvariantCulture);
    }

    // Every tile mode the round-trip matrix classifies Exact: released, emptied (the guest
    // copy reads zero underneath), and a host read gets every byte back.
    [Theory]
    [InlineData(GuestTileMode.Linear)]
    [InlineData(GuestTileMode.Standard4KB)]
    [InlineData(GuestTileMode.Standard64KB)]
    public void AHostReadOfAReleasedChunkGetsTheExactBytesBack(GuestTileMode tileMode)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var request = Request(rig.ImageAddress, tileMode);
        var size = request.Description.Data.Size;
        using (var transfer = new MetalImageTransfer(Device.Value))
        {
            Assert.Equal(MetalRoundTripFidelity.Exact, transfer.ClassifyRoundTrip(request.Description).Fidelity);
        }

        var bytes = Pattern(size, (int)tileMode);
        rig.Release(request, bytes, (int)(size / Chunk));
        for (var chunk = rig.ChunkOf(rig.ImageAddress); chunk < rig.ChunkOf(rig.ImageAddress + size); chunk++)
        {
            Assert.True(rig.Views.SingleCopy!.IsReleased(chunk));
            Assert.True(rig.SingleCopy.IsEmptied(chunk));
            Assert.True(rig.Memory.IsSingleCopyHole(chunk));
        }

        Assert.True(rig.Raw(rig.ImageAddress, size).All(static value => value == 0));

        Assert.Equal(bytes, rig.Read(rig.ImageAddress, size));
        Assert.Equal(bytes, rig.Raw(rig.ImageAddress, size));
        Assert.Equal(0, rig.Views.SingleCopy!.ReleasedChunks);
        Assert.Equal(1, rig.SingleCopy.Counts.Restores);
        Assert.Equal(1, Field(rig.SingleCopy.Describe(), "restores_host_read"));
        output.WriteLine(rig.SingleCopy.Describe());
    }

    // A fill that covers a whole chunk (a file read into it) takes it over without a
    // retile; one that stops short gets the rest of its chunk back from the texture.
    [Fact]
    public void AFullOverwriteSkipsTheRestoreAndAShortOneRestoresTheRest()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var request = Request(rig.ImageAddress, GuestTileMode.Standard64KB);
        var size = request.Description.Data.Size;
        var bytes = Pattern(size, 7);
        rig.Release(request, bytes, (int)(size / Chunk));

        var incoming = Pattern(Chunk, 8);
        Assert.True(rig.Views.TryFillSingleView(rig.ImageAddress + Chunk, (int)Chunk, incoming,
            static (state, destination) => { state.CopyTo(destination); return state.Length; }, out var filled));
        Assert.Equal((int)Chunk, filled);
        Assert.Equal(0, rig.SingleCopy.Counts.Restores);
        Assert.Equal(1, rig.SingleCopy.Counts.Overwrites);
        Assert.Equal(incoming, rig.Raw(rig.ImageAddress + Chunk, Chunk));
        Assert.True(rig.Views.SingleCopy!.IsReleased(rig.ChunkOf(rig.ImageAddress)));
        Assert.False(rig.Views.SingleCopy!.IsReleased(rig.ChunkOf(rig.ImageAddress + Chunk)));

        // A read stops 1000 bytes into the third chunk: the rest of it is the texture's.
        const int Short = 1000;
        Assert.True(rig.Views.TryFillSingleView(rig.ImageAddress + 2 * Chunk, (int)Chunk, incoming,
            static (state, destination) => { state.AsSpan(0, Short).CopyTo(destination); return Short; }, out filled));
        Assert.Equal(Short, filled);
        var third = rig.Raw(rig.ImageAddress + 2 * Chunk, Chunk);
        Assert.Equal(incoming.AsSpan(0, Short).ToArray(), third.AsSpan(0, Short).ToArray());
        Assert.Equal(bytes.AsSpan((int)(2 * Chunk) + Short, (int)Chunk - Short).ToArray(), third.AsSpan(Short).ToArray());
        Assert.Equal(0, rig.SingleCopy.Counts.Restores);
    }

    // A store into part of a chunk restores it first, so the bytes it leaves are the guest's.
    [Fact]
    public void APartialHostWriteRestoresAndThenLands()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var request = Request(rig.ImageAddress, GuestTileMode.Linear);
        var size = request.Description.Data.Size;
        var bytes = Pattern(size, 11);
        rig.Release(request, bytes, (int)(size / Chunk));

        var patch = Pattern(16, 12);
        const int At = 70000;
        Assert.True(rig.Views.TryWriteBacking(rig.ImageAddress + At, patch));
        patch.CopyTo(bytes, At);
        Assert.Equal(bytes, rig.Raw(rig.ImageAddress, size));
        Assert.Equal(1, Field(rig.SingleCopy.Describe(), "restores_host_write"));
        Assert.Equal(0, rig.Views.SingleCopy!.ReleasedChunks);
    }

    // A binding over a released chunk restores it before any window names it: the GPU reads
    // the guest bytes through the window, in place.
    [Fact]
    public void AGpuBindingRestoresFirst()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var request = Request(rig.ImageAddress, GuestTileMode.Standard64KB);
        var size = request.Description.Data.Size;
        var bytes = Pattern(size, 21);
        rig.Release(request, bytes, (int)(size / Chunk));

        var range = rig.Memory.Resolve(rig.ImageAddress, size);
        Assert.True(rig.Memory.IsInPlace(range));
        Assert.Equal(1, Field(rig.SingleCopy.Describe(), "restores_gpu"));
        Assert.False(rig.Memory.IsSingleCopyHole(rig.ChunkOf(rig.ImageAddress)));
        var readback = rig.Harness.Buffer(size);
        Copy(rig.Harness, range.Buffer, range.Offset, readback, size);
        rig.Harness.Submit();
        rig.Harness.Begin();
        Assert.Equal(bytes, rig.Harness.Read(readback, 0, size));
    }

    // Retirement drops the image: its chunks get their bytes back in the backing first.
    [Fact]
    public void RetirementRestoresBeforeTheTextureGoes()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var request = Request(rig.ImageAddress, GuestTileMode.Standard4KB);
        var size = request.Description.Data.Size;
        var bytes = Pattern(size, 31);
        var image = rig.Release(request, bytes, (int)(size / Chunk));

        rig.Configure(retireFrames: 1, retireSeconds: 0.5);
        for (var frame = 0; frame < 8 && image.Registered; frame++)
        {
            rig.Cache.MarkFrame();
            for (var step = 0; step < 8; step++)
            {
                rig.Advance(0.25);
                rig.Submit();
                rig.Cache.Collect();
            }
        }

        Assert.False(image.Registered);
        Assert.Equal(0, rig.Views.SingleCopy!.ReleasedChunks);
        Assert.Equal(bytes, rig.Raw(rig.ImageAddress, size));
        Assert.Equal(1, Field(rig.SingleCopy.Describe(), "restores_teardown"));
    }

    // A run with bytes no element maps to (row padding) is released only when the verify
    // sees the guest bytes equal what the retile writes into zeroed staging.
    [Fact]
    public void AnElementsExactRunIsReleasedOnlyWhenTheVerifyAgrees()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        const uint Width = 200;
        var padded = Request(rig.ImageAddress, GuestTileMode.Linear, Width);
        var size = padded.Description.Data.Size;
        using (var transfer = new MetalImageTransfer(Device.Value))
        {
            Assert.Equal(MetalRoundTripFidelity.ElementsExact, transfer.ClassifyRoundTrip(padded.Description).Fidelity);
        }

        // Padding the retile does not write holds data: rejected, nothing emptied.
        var noisy = Pattern(size, 41);
        Assert.True(rig.Views.TryWriteBacking(rig.ImageAddress, noisy));
        var identifier = rig.Acquire(padded);
        rig.Submit();
        var image = rig.Cache.GetImage(identifier);
        rig.Cache.Collect();
        rig.Cache.Collect();
        rig.Submit();
        rig.Cache.Collect();
        Assert.Equal(1, rig.SingleCopy.Counts.VerifyRejects);
        Assert.Equal(0UL, rig.SingleCopy.Counts.EmptiedBytes);
        Assert.Equal(0, rig.Views.SingleCopy!.ReleasedChunks);
        Assert.Null(image.SingleCopyRecord);
        Assert.Equal(noisy, rig.Raw(rig.ImageAddress, size));

        // The same layout with zero padding, elsewhere: released, and restored exactly.
        var clean = Pattern(size, 42);
        for (var row = 0; row < Side; row++)
        {
            Array.Clear(clean, row * (int)Side * 4 + (int)Width * 4, (int)(Side - Width) * 4);
        }

        var other = Request(rig.ImageAddress + 2 * Mib, GuestTileMode.Linear, Width);
        rig.Release(other, clean, (int)(size / Chunk));
        Assert.Equal(clean, rig.Read(rig.ImageAddress + 2 * Mib, size));
    }

    // Guest threads fault (loads) across a released run at once while
    // host reads race them: one restore, every access retried, every byte the guest's.
    [Fact]
    public void ConcurrentFaultsRestoreOnce()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var request = Request(rig.ImageAddress, GuestTileMode.Standard64KB);
        var size = request.Description.Data.Size;
        var bytes = Pattern(size, 51);
        rig.Release(request, bytes, (int)(size / Chunk));

        const int Threads = 8;
        var start = new Barrier(Threads);
        var failures = new System.Collections.Concurrent.ConcurrentBag<string>();
        var threads = Enumerable.Range(0, Threads).Select(index => new Thread(() =>
        {
            start.SignalAndWait();
            var address = rig.ImageAddress + (ulong)index * (size / Threads);
            if (index % 3 == 2)
            {
                var read = new byte[4096];
                if (!rig.Views.TryReadBacking(address, read) || !read.AsSpan().SequenceEqual(bytes.AsSpan((int)(address - rig.ImageAddress), 4096)))
                {
                    failures.Add($"host read {index}");
                }
            }
            else if (!rig.SingleCopy.TryResolveFault(address + 0x123, write: false))
            {
                failures.Add($"fault {index}");
            }
        })).ToList();
        threads.ForEach(static thread => thread.Start());
        threads.ForEach(static thread => thread.Join());

        Assert.Empty(failures);
        Assert.Equal(1, rig.SingleCopy.Counts.Restores);
        Assert.Equal(bytes, rig.Raw(rig.ImageAddress, size));
        Assert.Equal(0, rig.Views.SingleCopy!.ReleasedChunks);
    }

    // Sentinels: a released chunk holds one in the canonical alias, the GPU alias dummy
    // another. A window that already holds the bytes beside the run reads the dummy over the
    // run - never the released chunk - and a binding over the run, restored while that
    // window is pinned, is staged with the guest's bytes until the next command buffer
    // renews the window.
    [Fact]
    public void NoWindowReadsAReleasedChunk()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value, sentinel: true);
        var request = Request(rig.ImageAddress, GuestTileMode.Standard64KB);
        var size = request.Description.Data.Size;
        var bytes = Pattern(size, 61);
        var beside = Pattern(size, 62);
        Assert.True(rig.Views.TryWriteBacking(rig.ImageAddress + size, beside));
        rig.Release(request, bytes, (int)(size / Chunk));
        Assert.True(rig.Raw(rig.ImageAddress, size).All(static value => value == MetalSingleCopy.ReleasedSentinel));

        // The bytes beside the run, then back into the run through the same window.
        var near = rig.Memory.Resolve(rig.ImageAddress + size, size);
        Assert.True(rig.Memory.IsInPlace(near));
        Assert.True(near.Offset >= size);
        var holes = rig.Harness.Buffer(2 * size);
        Copy(rig.Harness, near.Buffer, near.Offset - size, holes, 2 * size);
        var hits = rig.Memory.WindowLookupHits;
        Assert.Equal(near, rig.Memory.Resolve(rig.ImageAddress + size, size));
        Assert.Equal(hits + 1, rig.Memory.WindowLookupHits);
        var staged = rig.Memory.Resolve(rig.ImageAddress, size);
        Assert.False(rig.Memory.IsInPlace(staged));
        // The refill changed the trap epoch even for a cached neighbour window.
        rig.Memory.Resolve(rig.ImageAddress + size, size);
        Assert.Equal(hits + 1, rig.Memory.WindowLookupHits);
        Assert.True(rig.Memory.HasStaleSingleCopyWindows);
        var stagedReadback = rig.Harness.Buffer(size);
        Copy(rig.Harness, staged.Buffer, staged.Offset, stagedReadback, size);
        rig.Harness.Submit();
        rig.Harness.Begin();

        var seen = rig.Harness.Read(holes, 0, 2 * size);
        Assert.True(seen.AsSpan(0, (int)size).ToArray().All(static value => value == MetalSingleCopy.DummySentinel));
        Assert.Equal(beside, seen.AsSpan((int)size).ToArray());
        Assert.Equal(bytes, rig.Harness.Read(stagedReadback, 0, size));

        // The next command buffer renews the window: in place, the guest's bytes.
        rig.Submit();
        Assert.False(rig.Memory.HasStaleSingleCopyWindows);
        var range = rig.Memory.Resolve(rig.ImageAddress, size);
        Assert.True(rig.Memory.IsInPlace(range));
        var readback = rig.Harness.Buffer(size);
        Copy(rig.Harness, range.Buffer, range.Offset, readback, size);
        rig.Harness.Submit();
        rig.Harness.Begin();
        var gpu = rig.Harness.Read(readback, 0, size);
        Assert.Equal(bytes, gpu);
        Assert.Equal(bytes, rig.Raw(rig.ImageAddress, size));
        output.WriteLine(rig.SingleCopy.Describe());
    }

    // A device-address program can reach any mapped page past every window resolve: before
    // one is prepared, every released or punched chunk gets its bytes and its GPU alias pages
    // back (a window this command buffer used over one is left stale, so the host ends the
    // command buffer), and nothing is released for the rest of the session.
    [Fact]
    public void ADeviceAddressProgramGivesEverythingBackAndStopsReleases()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var request = Request(rig.ImageAddress, GuestTileMode.Standard64KB);
        var size = request.Description.Data.Size;
        var bytes = Pattern(size, 71);
        var image = rig.Release(request, bytes, (int)(size / Chunk));

        // A window beside the run is pinned by this command buffer.
        var near = rig.Memory.Resolve(rig.ImageAddress + size, size);
        Assert.True(rig.Memory.IsInPlace(near));

        rig.SingleCopy.NoteDeviceAddressUse(late: false);
        Assert.True(rig.SingleCopy.DeviceAddressSeen);
        Assert.Equal(0, rig.Views.SingleCopy!.ReleasedChunks);
        Assert.Equal(0, rig.Memory.SingleCopyHoleCount);
        Assert.Equal(bytes, rig.Raw(rig.ImageAddress, size));
        Assert.True(rig.Memory.HasStaleSingleCopyWindows);
        Assert.Equal(0, Field(rig.SingleCopy.Describe(), "device_address_stale"));

        // The next command buffer: every window over the run reads the guest's bytes in place.
        rig.Submit();
        Assert.False(rig.Memory.HasStaleSingleCopyWindows);
        var range = rig.Memory.Resolve(rig.ImageAddress, 2 * size);
        Assert.True(rig.Memory.IsInPlace(range));
        var readback = rig.Harness.Buffer(size);
        Copy(rig.Harness, range.Buffer, range.Offset, readback, size);
        rig.Harness.Submit();
        rig.Harness.Begin();
        Assert.Equal(bytes, rig.Harness.Read(readback, 0, size));

        // Nothing is armed again, and the in-preparation check finds nothing to count.
        for (var tick = 0; tick < 4; tick++)
        {
            rig.Submit();
            rig.Cache.Collect();
        }

        Assert.Null(image.SingleCopyRecord);
        Assert.Equal(0, rig.Views.SingleCopy!.ReleasedChunks);
        rig.SingleCopy.NoteDeviceAddressUse(late: true);
        Assert.Equal(0, Field(rig.SingleCopy.Describe(), "device_address_stale"));
        Assert.Equal(1, Field(rig.SingleCopy.Describe(), "device_address_restores"));
    }

    // A copy this large would run unlocked, without the hooks; its source holds a released
    // chunk, so it takes the locked path, which restores the bytes before copying them.
    [Fact]
    public void ALargeCopyOutOfAReleasedChunkCopiesTheGuestsBytes()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var request = Request(rig.ImageAddress, GuestTileMode.Standard64KB);
        var size = request.Description.Data.Size;
        Assert.True(size >= Chunk && size <= 3 * Mib);
        var bytes = Pattern(size, 29);
        rig.Release(request, bytes, (int)(size / Chunk));
        var destination = rig.Address + BackingSize - size;

        Assert.True(rig.Views.TryCopyBacking(destination, rig.ImageAddress, size));
        Assert.Equal(bytes, rig.Raw(destination, size));
        Assert.Equal(bytes, rig.Raw(rig.ImageAddress, size));
        Assert.Equal(0, rig.Views.SingleCopy!.ReleasedChunks);
    }

    // A lossy texture's texels are approximated, so its guest bytes are the only exact copy.
    [Fact]
    public void ALossyTextureIsNeverReleased()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Cache.LossyTextures = true;
        if (!rig.Cache.LossyTextures) return;
        var request = Request(rig.ImageAddress, GuestTileMode.Standard64KB);
        Assert.True(rig.Views.TryWriteBacking(rig.ImageAddress, Pattern(request.Description.Data.Size, 13)));
        var image = rig.Cache.GetImage(rig.Acquire(request));
        for (var tick = 0; tick < 4; tick++)
        {
            rig.Submit();
            rig.Cache.Collect();
        }

        Assert.True(image.Shape.Lossy);
        Assert.Null(image.SingleCopyRecord);
        Assert.Equal(0, rig.Views.SingleCopy!.ReleasedChunks);
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Single-copy textures run on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);

    [DllImport("libSystem.B.dylib")]
    private static extern uint mach_task_self();

    [DllImport("libSystem.B.dylib")]
    private static extern int mach_vm_map(uint task, ref ulong address, ulong size, ulong mask, int flags,
        uint memoryObject, ulong offset, int copy, int currentProtection, int maximumProtection, uint inheritance);
}
