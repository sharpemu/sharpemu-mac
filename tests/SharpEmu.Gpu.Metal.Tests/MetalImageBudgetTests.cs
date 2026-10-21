// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// The image cache's memory policy: the budget sized from host readings, retirement of
/// images idle for both a number of frames and a number of seconds, the working-set guards
/// both obey, real allocated-size accounting, and the writeback a GPU-written image gets
/// before its texture goes. The policy arithmetic runs anywhere; cache behaviour runs on the
/// device behind the exclusive device gate.
/// </summary>
public sealed unsafe partial class MetalImageBudgetTests(ITestOutputHelper output)
{
    private const ulong MiB = 1024 * 1024;
    private const ulong GiB = 1024 * MiB;
    private const ulong GuestBase = 0x2_0000_0000;
    private const uint Side = 64;
    private const ulong ImageSize = Side * Side * 4;

    // ---- Budget arithmetic -------------------------------------------------------------

    private static MetalImageBudgetSettings Settings24 => MetalImageBudgetSettings.Defaults(24 * GiB);

    [Fact]
    public void DefaultsForA24GiBMac()
    {
        var settings = Settings24;
        Assert.Equal(1 * GiB, settings.Floor);
        Assert.Equal(6 * GiB, settings.Ceiling);
        Assert.Equal(1536 * MiB, settings.Reserve);
        Assert.Null(settings.FixedBudget);

        // Small and large machines keep floor <= ceiling and the 8 GiB cap.
        Assert.Equal(2 * GiB, MetalImageBudgetSettings.Defaults(8 * GiB).Ceiling);
        Assert.Equal(8 * GiB, MetalImageBudgetSettings.Defaults(128 * GiB).Ceiling);
        Assert.Equal(1 * GiB, MetalImageBudgetSettings.Defaults(2 * GiB).Ceiling);
    }

    [Fact]
    public void AnIdleHostGivesTheCeiling()
    {
        var sample = new MetalHostMemorySample(24 * GiB, 16 * GiB, 2 * GiB, MetalMemoryPressure.Normal);
        Assert.Equal(6 * GiB, Settings24.Compute(in sample, 0, 0));
    }

    // The budget is the cache's own bytes against system-wide available memory. The process
    // footprint, which counts compressed pages and charges a shared guest page once per
    // mapping, has no say: Demon's Souls read 17-18 GiB of footprint on a 24 GiB Mac with
    // gigabytes of it compressed or aliased.
    [Fact]
    public void TheBudgetIsTheCacheAgainstAvailableMemoryNotTheFootprint()
    {
        var cache = 3 * GiB;
        var lean = new MetalHostMemorySample(24 * GiB, 4 * GiB, 8 * GiB, MetalMemoryPressure.Normal);
        var inflated = lean with { ProcessFootprintBytes = 30 * GiB };
        Assert.Equal(cache + 4 * GiB - 1536 * MiB, Settings24.Compute(in lean, cache, cache));
        Assert.Equal(Settings24.Compute(in lean, cache, cache), Settings24.Compute(in inflated, cache, cache));
    }

    [Fact]
    public void AvailableMemoryBelowTheReserveShrinksTheBudgetByTheShortfall()
    {
        var cache = 4 * GiB;
        // 1 GiB available against a 1.5 GiB reserve: give back 512 MiB.
        var sample = new MetalHostMemorySample(24 * GiB, 1 * GiB, 10 * GiB, MetalMemoryPressure.Normal);
        Assert.Equal(cache - 512 * MiB, Settings24.Compute(in sample, cache, cache));

        // 2 GiB available: grow by 512 MiB at most.
        sample = sample with { AvailableBytes = 2 * GiB };
        Assert.Equal(cache + 512 * MiB, Settings24.Compute(in sample, cache, cache));
    }

    // Under WARN the cache sheds what is outside its in-use set and no more. Halving the cache
    // was a ratchet: Demon's Souls uses every image each frame, so "used=1726MB budget=1123MB"
    // could never be met and the cache only churned.
    [Fact]
    public void WarnFloorsTheBudgetAtTheInUseSet()
    {
        var cache = 3 * GiB;
        var inUse = 2 * GiB;
        var warn = new MetalHostMemorySample(24 * GiB, 8 * GiB, 17 * GiB, MetalMemoryPressure.Warn);
        // Plenty available: down to the set plus an eighth of slack.
        Assert.Equal(inUse + inUse / 8, Settings24.Compute(in warn, cache, inUse));

        // Nothing available: the host budget (cache - 1.5 GiB shortfall) is under the set; the set holds.
        var tight = warn with { AvailableBytes = 0 };
        Assert.Equal(inUse, Settings24.Compute(in tight, cache, inUse));

        // A cache whose every image is in use keeps a budget it can meet.
        Assert.True(Settings24.Compute(in warn, cache, cache) >= cache);
        Assert.True(Settings24.Compute(in tight, cache, cache) >= cache);

        // Critical pressure still drops to the floor, and so does a guest that fills the machine.
        var critical = warn with { Pressure = MetalMemoryPressure.Critical };
        Assert.Equal(1 * GiB, Settings24.Compute(in critical, cache, inUse));
        var full = new MetalHostMemorySample(16 * GiB, 0, 15 * GiB, MetalMemoryPressure.Normal);
        Assert.Equal(1 * GiB, MetalImageBudgetSettings.Defaults(16 * GiB).Compute(in full, 512 * MiB, 512 * MiB));
    }

    [Fact]
    public void AFixedBudgetIgnoresTheHostAndNoReadingGivesTheCeiling()
    {
        var critical = new MetalHostMemorySample(24 * GiB, 0, 23 * GiB, MetalMemoryPressure.Critical);
        Assert.Equal(3 * GiB, (Settings24 with { FixedBudget = 3 * GiB }).Compute(in critical, 5 * GiB, 5 * GiB));
        Assert.Equal(6 * GiB, Settings24.Compute(default, 5 * GiB, 5 * GiB));
    }

    // ---- Idle clock --------------------------------------------------------------------

    [Fact]
    public void TheIdleClockErrsLate()
    {
        var second = Stopwatch.Frequency;
        var clock = new MetalImageIdleClock(frames: 3, seconds: 10, sampleInterval: second);
        Assert.Equal(0UL, clock.TickFramesAgo(3));
        Assert.Equal(0UL, clock.TickAtOrBefore(100 * second));

        // Frames begin at ticks 10, 20, 30, 40: three frames before the newest is the one at 10.
        foreach (var tick in new ulong[] { 10, 20, 30, 40 })
        {
            clock.MarkFrame(tick);
        }

        Assert.Equal(40UL, clock.TickFramesAgo(0));
        Assert.Equal(10UL, clock.TickFramesAgo(3));
        clock.MarkFrame(50);
        Assert.Equal(20UL, clock.TickFramesAgo(3));

        // Samples closer than the interval are skipped; a time between samples maps to the older one.
        clock.Sample(1 * second, 100);
        clock.Sample(1 * second + second / 2, 150);
        clock.Sample(3 * second, 300);
        Assert.Equal(0UL, clock.TickAtOrBefore(second - 1));
        Assert.Equal(100UL, clock.TickAtOrBefore(2 * second));
        Assert.Equal(300UL, clock.TickAtOrBefore(3 * second));
    }

    // ---- Host readings -----------------------------------------------------------------

    [Fact]
    public void TheHostReadingIsLive()
    {
        if (!OperatingSystem.IsMacOS())
        {
            return;
        }

        var sample = MetalHostMemory.Sample();
        output.WriteLine($"physical={sample.PhysicalBytes / MiB} available={sample.AvailableBytes / MiB} " +
            $"footprint={sample.ProcessFootprintBytes / MiB} pressure={sample.Pressure} source={MetalHostMemory.PressureSourceInstalled}");
        Assert.True(sample.PhysicalBytes >= 4 * GiB);
        Assert.True(sample.AvailableBytes > 0 && sample.AvailableBytes < sample.PhysicalBytes);
        Assert.True(sample.ProcessFootprintBytes > 0 && sample.ProcessFootprintBytes < sample.PhysicalBytes);
        Assert.Contains(sample.Pressure, new[] { MetalMemoryPressure.Normal, MetalMemoryPressure.Warn, MetalMemoryPressure.Critical });
        Assert.True(MetalHostMemory.PressureSourceInstalled);
    }

    // ---- Cache behaviour on the device -------------------------------------------------

    [Fact]
    public void TheBudgetCountsTheTexturesRealAllocation()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var image = rig.Cache.GetImage(rig.Acquire(GuestBase, ImageRole.Texture));
        output.WriteLine($"guest={ImageSize} allocated={image.AllocatedBytes}");
        Assert.True(image.AllocatedBytes >= ImageSize);
        Assert.Equal(image.AllocatedBytes, rig.Cache.TotalUsedMemory);
    }

    // An image is retired only once it is idle for both the frames and the seconds; the one the
    // title keeps using stays.
    [Fact]
    public void RetirementNeedsBothFramesAndSeconds()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Configure(retireFrames: 10, retireSeconds: 5);
        var idle = rig.Acquire(GuestBase, ImageRole.Texture);
        var busy = rig.Acquire(GuestBase + 0x10000, ImageRole.Texture);

        // Twenty frames inside one second: frames enough, seconds not.
        for (var frame = 0; frame < 20; frame++)
        {
            rig.Frame(seconds: 0.05, touch: GuestBase + 0x10000);
        }

        Assert.True(rig.Cache.TryGetImage(idle, out _));

        // Five more seconds: now both hold, and only the image nobody uses goes.
        rig.Frame(seconds: 5, touch: GuestBase + 0x10000);
        rig.Frame(seconds: 5, touch: GuestBase + 0x10000);
        Assert.False(rig.Cache.TryGetImage(idle, out _));
        Assert.True(rig.Cache.TryGetImage(busy, out _));
        Assert.Equal(1, rig.Cache.RetiredTotals.Images);
    }

    [Fact]
    public void RetirementWaitsForFramesAtALowFrameRate()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Configure(retireFrames: 10, retireSeconds: 5);
        var idle = rig.Acquire(GuestBase, ImageRole.Texture);

        // A title presenting every 4 s: 30 s pass in 8 frames, below the 10-frame age.
        for (var frame = 0; frame < 8; frame++)
        {
            rig.Frame(seconds: 4);
        }

        Assert.True(rig.Cache.TryGetImage(idle, out _));
        for (var frame = 0; frame < 4; frame++)
        {
            rig.Frame(seconds: 4);
        }

        Assert.False(rig.Cache.TryGetImage(idle, out _));
    }

    // An image the GPU never wrote that guest stores have already made stale holds nothing a
    // read could use without uploading it again; it goes once idle for the stale time, long
    // before the frame count retirement waits for, while a current image waits for that.
    [Fact]
    public void AStaleImageGoesBeforeTheAgePolicyAndACurrentOneDoesNot()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Configure(retireFrames: 1000, retireSeconds: 1000);
        var stale = rig.Acquire(GuestBase, ImageRole.Texture);
        var current = rig.Acquire(GuestBase + 0x10000, ImageRole.Texture);
        rig.Frame(seconds: 0.1);

        // A store the page watch catches: the image is behind guest memory now and unwatched.
        Assert.True(((IGuestImageStore)rig.Cache).MarkCpuWrite(GuestBase + 64, 4));
        Assert.True(rig.Cache.GetImage(stale).IsCpuDirty);
        Assert.False(rig.Cache.GetImage(stale).IsWatched);

        for (var frame = 0; frame < 3; frame++)
        {
            rig.Frame(seconds: 1);
        }

        Assert.False(rig.Cache.TryGetImage(stale, out _));
        Assert.True(rig.Cache.TryGetImage(current, out _));
    }

    // A GPU-written image is published into guest memory before it goes, and its texture lives
    // until the submission that carries the writeback completes.
    [Fact]
    public void ARetiredGpuWrittenImageIsWrittenBackBeforeItsTextureIsReleased()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Configure(retireFrames: 2, retireSeconds: 1);
        var written = MetalTileOracle.Pattern((int)ImageSize, 0x5A);
        rig.WriteGuest(GuestBase, written);
        var identifier = rig.Acquire(GuestBase, ImageRole.ColorTarget);
        var image = rig.Cache.GetImage(identifier);
        Assert.True(image.IsGpuModified);
        rig.Harness.Submit();
        rig.Harness.Begin();
        // What the texture holds is the "GPU's" result; guest memory now holds something else.
        var stale = MetalTileOracle.Pattern((int)ImageSize, 0xC3);
        rig.WriteGuest(GuestBase, stale);

        for (var frame = 0; frame < 4 && rig.Cache.TryGetImage(identifier, out _); frame++)
        {
            rig.Frame(seconds: 1, submit: false);
        }

        Assert.False(rig.Cache.TryGetImage(identifier, out _));
        Assert.Equal(0UL, rig.Cache.TotalUsedMemory);
        // The writeback is recorded, not yet run: the texture must still be alive.
        Assert.NotEqual(0, image.Texture);
        Assert.Equal(stale, rig.ReadGuest(GuestBase, ImageSize));

        rig.Harness.Submit();
        rig.Harness.Begin();
        Assert.Equal(written, rig.ReadGuest(GuestBase, ImageSize));
        rig.Cache.Collect();
        Assert.Equal(0, image.Texture);
    }

    // Budget eviction takes only images outside the working set and idle for the minimum time;
    // critical pressure drops the time requirement but never the frame guard.
    [Fact]
    public void BudgetEvictionKeepsTheWorkingSetAndRecentImages()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        // Retirement off; any non-empty cache is over a 1-byte budget.
        rig.Configure(retireFrames: 0, retireSeconds: 0, budget: Settings24 with { FixedBudget = 1 });
        var old = rig.Acquire(GuestBase, ImageRole.Texture);
        rig.Frame(seconds: 3);
        var recent = rig.Acquire(GuestBase + 0x10000, ImageRole.Texture);
        var current = rig.Acquire(GuestBase + 0x20000, ImageRole.Texture);
        for (var frame = 0; frame < 3; frame++)
        {
            rig.Frame(seconds: 0.3, touch: GuestBase + 0x20000);
        }

        // The recent image is two frames old, outside the frame guard, but idle under 2 s.
        Assert.False(rig.Cache.TryGetImage(old, out _));
        Assert.True(rig.Cache.TryGetImage(recent, out _));
        Assert.True(rig.Cache.TryGetImage(current, out _));
        Assert.Equal(1, rig.Cache.Counters.Evictions);

        // Critical pressure: the recent image goes, the one in this frame stays.
        rig.Configure(retireFrames: 0, retireSeconds: 0, budget: Settings24 with { Floor = 1, Ceiling = 1 },
            sample: () => new MetalHostMemorySample(24 * GiB, 0, 23 * GiB, MetalMemoryPressure.Critical));
        Assert.Equal(1UL, rig.Cache.MemoryBudgetBytes);
        rig.Frame(seconds: 0.3, touch: GuestBase + 0x20000);
        Assert.False(rig.Cache.TryGetImage(recent, out _));
        Assert.True(rig.Cache.TryGetImage(current, out _));
    }

    // Outside critical pressure an image used in the last InUseFrames presented frames is never
    // evicted, even outside the frame guard and idle past the minimum: here one used every
    // third frame of a title presenting once a second. Under WARN the budget sits at the
    // in-use set, so only what is outside it is over.
    [Fact]
    public void BudgetEvictionTakesOnlyImagesOutsideTheInUseSet()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Configure(retireFrames: 0, retireSeconds: 0, budget: Settings24 with { FixedBudget = 1 }, inUseFrames: 4);
        var once = rig.Acquire(GuestBase, ImageRole.Texture);
        var every = rig.Acquire(GuestBase + 0x10000, ImageRole.Texture);
        var third = rig.Acquire(GuestBase + 0x20000, ImageRole.Texture);
        for (var frame = 0; frame < 12; frame++)
        {
            if (frame % 3 == 0)
            {
                rig.Acquire(GuestBase + 0x20000, ImageRole.Texture);
            }

            rig.Frame(seconds: 1, touch: GuestBase + 0x10000);
        }

        Assert.False(rig.Cache.TryGetImage(once, out _));
        Assert.True(rig.Cache.TryGetImage(every, out _));
        Assert.True(rig.Cache.TryGetImage(third, out _));
        Assert.Equal(1, rig.Cache.Counters.Evictions);
        var inUse = rig.Cache.GetImage(every).CountedBytes + rig.Cache.GetImage(third).CountedBytes;
        Assert.Equal(inUse, rig.Cache.InUseBytes);

        // WARN: the budget is the in-use set plus an eighth, not half the cache.
        rig.Configure(retireFrames: 0, retireSeconds: 0, budget: Settings24 with { Floor = 1 },
            sample: () => new MetalHostMemorySample(24 * GiB, 8 * GiB, 10 * GiB, MetalMemoryPressure.Warn), inUseFrames: 4);
        Assert.Equal(inUse + inUse / 8, rig.Cache.MemoryBudgetBytes);
        for (var frame = 0; frame < 6; frame++)
        {
            if (frame % 3 == 0)
            {
                rig.Acquire(GuestBase + 0x20000, ImageRole.Texture);
            }

            rig.Frame(seconds: 1, touch: GuestBase + 0x10000);
        }

        Assert.True(rig.Cache.TryGetImage(every, out _));
        Assert.True(rig.Cache.TryGetImage(third, out _));
        Assert.Equal(1, rig.Cache.Counters.Evictions);
    }

    [Fact]
    public void ARecreatedEvictedImageCountsAsARefault()
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        rig.Configure(retireFrames: 2, retireSeconds: 1);
        var first = rig.Acquire(GuestBase, ImageRole.Texture);
        for (var frame = 0; frame < 4 && rig.Cache.TryGetImage(first, out _); frame++)
        {
            rig.Frame(seconds: 1);
        }

        Assert.False(rig.Cache.TryGetImage(first, out _));
        Assert.Equal(0, rig.Cache.Refaults);
        rig.Acquire(GuestBase, ImageRole.Texture);
        Assert.Equal(1, rig.Cache.Refaults);
    }

    // ---- Rig ---------------------------------------------------------------------------

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

    private sealed class Backing : IGuestBackedSpace, IDisposable
    {
        // Room for the mipped textures the BASE_LEVEL tests make next to the small ones.
        public const ulong Size = 4 * 1024 * 1024;
        private readonly void* _block = NativeMemory.AlignedAlloc((nuint)Size, 16384);

        public Backing() => NativeMemory.Clear(_block, (nuint)Size);

        public void Dispose() => NativeMemory.AlignedFree(_block);

        private bool Contains(ulong address, ulong length) => address >= GuestBase && address - GuestBase + length <= Size;

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
            alias = Contains(address, size) ? (ulong)_block + (address - GuestBase) : 0;
            return alias != 0;
        }

        public bool TryResolveBackingAliasSpan(ulong address, ulong size, out ulong alias) =>
            TryResolveBackingAlias(address, size, out alias);

        public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data)
        {
            if (!Contains(address, (ulong)data.Length)) return false;
            data.CopyTo(new Span<byte>((byte*)_block + (address - GuestBase), data.Length));
            return true;
        }

        public bool TryReadBacking(ulong address, Span<byte> data)
        {
            if (!Contains(address, (ulong)data.Length)) return false;
            new ReadOnlySpan<byte>((byte*)_block + (address - GuestBase), data.Length).CopyTo(data);
            return true;
        }
    }

    private sealed class NoMemory : ICpuMemory
    {
        public bool TryRead(ulong virtualAddress, Span<byte> destination) => false;
        public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source) => false;
        public bool CanRead(ulong address, ulong size) => false;
    }

    private sealed class Owner(MetalTransferHarness harness) : IMetalImageCacheOwner
    {
        public nint Device => harness.Device;
        public ulong CurrentSerial => harness.SubmissionSerial;
        public ulong CompletedSerial => CompletedOverride ?? harness.CompletedSerial;

        // A GPU running behind: submissions past this serial have not completed yet.
        public ulong? CompletedOverride { get; set; }

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

        // What a guest thread waiting for a recorded writeback does; a test supplies it.
        public Func<ulong, int, bool>? WaitOffWorker { get; set; }

        public bool WaitForSubmissionOffWorker(ulong serial, int timeoutMilliseconds) =>
            WaitOffWorker?.Invoke(serial, timeoutMilliseconds) ?? CompletedSerial >= serial;
    }

    private sealed class Rig : IDisposable
    {
        private readonly AddressSpace _space = new();
        private readonly PageGuard _pages;
        private readonly MetalGuestMemory _memory;
        private readonly Backing _backing = new();
        private readonly bool _savedLinearViews = MetalLinearWindowView.Enabled;
        private long _now = 1000 * Stopwatch.Frequency;

        public Rig(nint device, int writeWatchLimit = PageGuard.MaxWriteWatchesPerPage)
        {
            MetalLinearWindowView.Enabled = false;
            Harness = new MetalTransferHarness(device);
            Harness.Begin();
            _pages = NewPageGuard(_space, writeWatchLimit);
            _memory = new MetalGuestMemory(device, _backing, new NoMemory());
            CacheOwner = new Owner(Harness);
            Cache = new MetalImageCache(CacheOwner, Harness.Transfer, _memory, _pages, _backing);
            // Exact textures: lossy compression is not this suite's subject.
            Cache.LossyTextures = false;
            // No host readings and no retirement unless a test asks: a fixed budget nothing reaches.
            Configure(retireFrames: 0, retireSeconds: 0);
        }

        public MetalTransferHarness Harness { get; }
        public MetalImageCache Cache { get; }
        public Owner CacheOwner { get; }
        public PageGuard Pages => _pages;

        // inUseFrames 0 leaves budget eviction to the frame guard and the idle minimum.
        public void Configure(
            uint retireFrames, double retireSeconds,
            MetalImageBudgetSettings? budget = null, Func<MetalHostMemorySample>? sample = null, uint inUseFrames = 0)
        {
            budget ??= sample is null
                ? MetalImageBudgetSettings.Defaults(24 * GiB) with { FixedBudget = 64 * GiB }
                : MetalImageBudgetSettings.Defaults(24 * GiB);
            Cache.ConfigureMemoryPolicy(
                budget,
                new MetalImageRetirementSettings { Frames = retireFrames, Seconds = retireSeconds, InUseFrames = inUseFrames },
                sample ?? (() => default),
                () => _now,
                () => 0);
        }

        // One presented frame lasting this long: the touched image is used in it, the frame is
        // marked, and enough submissions complete that the tick guard is past.
        public void Frame(double seconds, ulong? touch = null, bool submit = true)
        {
            if (touch is { } address)
            {
                Acquire(address, ImageRole.Texture);
            }

            Cache.MarkFrame();
            var steps = 40;
            for (var step = 0; step < steps; step++)
            {
                _now += (long)(seconds * Stopwatch.Frequency / steps);
                if (submit)
                {
                    Harness.Submit();
                    Harness.Begin();
                }

                Cache.Collect();
            }
        }

        public void WriteGuest(ulong address, byte[] bytes) => Assert.True(_backing.TryWriteBacking(address, bytes));

        public byte[] ReadGuest(ulong address, ulong size)
        {
            var bytes = new byte[size];
            Assert.True(_backing.TryReadBacking(address, bytes));
            return bytes;
        }

        public ResourceSlotIdentifier Acquire(ulong address, ImageRole role)
        {
            var description = ImageDescription.Create();
            description.Data = new GuestSpan(address, ImageSize);
            description.PixelFormat = GpuFormat.R8G8B8A8Unorm;
            description.GuestFormat = GuestPixelFormat.Bits8_8_8_8UNorm;
            description.Type = GuestImageType.Color2D;
            description.Extent = new GpuExtent3D(Side, Side, 1);
            description.Resources = new SubresourceCount(1, 1);
            description.Pitch = Side;
            description.BytesPerBlock = 4;
            description.Samples = 1;
            description.TileMode = GuestTileMode.Linear;
            description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = ImageSize, Pitch = Side, Height = Side };
            var view = ImageViewDescription.Default with
            {
                Format = GpuFormat.R8G8B8A8Unorm,
                Type = GpuImageViewType.Type2D,
                Aspect = GpuImageAspect.ColorBit,
                LayerCount = 1,
                Usage = role == ImageRole.ColorTarget ? GpuImageUsage.ColorAttachmentBit : GpuImageUsage.SampledBit,
            };
            var request = new ImageRequest(description, view, role);
            var identifier = Cache.FindImage(ref request);
            if (role == ImageRole.ColorTarget)
            {
                Cache.AcquireColorTargetView(identifier, request);
            }
            else
            {
                Cache.AcquireTextureView(identifier, request);
            }

            return identifier;
        }

        public void Dispose()
        {
            Cache.StopWatching();
            Cache.Dispose();
            _pages.Dispose();
            Harness.Dispose();
            _backing.Dispose();
            MetalLinearWindowView.Enabled = _savedLinearViews;
        }
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Image cache memory policy runs on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
