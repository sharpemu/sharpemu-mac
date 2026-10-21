// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// The tiler's staging slabs: power-of-two classes held a 33 MiB detile in 64 MiB, and a
/// load burst's slabs stayed as shared memory (1226 MB during Demon's Souls' world load).
/// A slab now wastes at most an eighth, is released once idle, the free ones are capped,
/// idle ones are purgeable, and none is lent again before its command buffer completes.
/// </summary>
public sealed unsafe class MetalStagingPoolTests(ITestOutputHelper output)
{
    private const ulong MiB = 1024 * 1024;

    private static readonly nint SelNewSharedEvent = MetalNative.Selector("newSharedEvent");
    private static readonly nint SelSignaledValue = MetalNative.Selector("signaledValue");
    private static readonly nint SelSetSignaledValue = MetalNative.Selector("setSignaledValue:");
    private static readonly nint SelEncodeWait = MetalNative.Selector("encodeWaitForEvent:value:");
    private static readonly nint SelEncodeSignal = MetalNative.Selector("encodeSignalEvent:value:");
    private static readonly nint SelBlit = MetalNative.Selector("blitCommandEncoder");
    private static readonly nint SelFill = MetalNative.Selector("fillBuffer:range:value:");
    private static readonly nint SelEnd = MetalNative.Selector("endEncoding");
    private static readonly nint SelCommit = MetalNative.Selector("commit");
    private static readonly nint SelWait = MetalNative.Selector("waitUntilCompleted");
    private static readonly nint SelContents = MetalNative.Selector("contents");
    private static readonly nint SelRelease = MetalNative.Selector("release");
    private static readonly nint SelCommandBuffer = MetalNative.Selector("commandBuffer");
    private static readonly nint SelNewQueue = MetalNative.Selector("newCommandQueue");

    [Theory]
    [InlineData(1UL)]
    [InlineData(64 * 1024UL)]
    [InlineData(64 * 1024UL + 4)]
    [InlineData(1920UL * 1080 * 4)]
    [InlineData(33 * MiB)]
    [InlineData(64 * MiB + 1)]
    [InlineData(127 * MiB)]
    [InlineData(1000 * MiB + 12345)]
    public void ASlabWastesAtMostAnEighthOfTheRequest(ulong size)
    {
        var rounded = (size + 3) & ~3UL;
        var capacity = MetalStagingPool.SlabCapacity(size);
        Assert.True(capacity >= rounded);
        // Waste is under an eighth of the request, or under the 64 KiB page-run granule.
        Assert.True(capacity - rounded < Math.Max(MetalStagingPool.MinimumSlabBytes, rounded / 8 + 1),
            $"size={size} capacity={capacity}");
        // Every class is page aligned.
        Assert.Equal(0UL, capacity % MetalStagingPool.MinimumSlabBytes);
    }

    [Fact]
    public void AThirtyThreeMebibyteRequestTakesThirtySixNotSixtyFour()
    {
        Assert.Equal(36 * MiB, MetalStagingPool.SlabCapacity(33 * MiB));
        // A 1080p RGBA8 write-back (7.9 MiB) took 16 MiB.
        Assert.Equal(8 * MiB, MetalStagingPool.SlabCapacity(1920UL * 1080 * 4));
        Assert.Equal(18 * MiB, MetalStagingPool.SlabCapacity(17 * MiB));
    }

    [Fact]
    public void AFreeSlabOneStepLargerServesARequestAndNoneLargerDoes()
    {
        if (!RequireDevice()) return;
        using var pool = new MetalStagingPool(Device.Value);
        long now = 0;
        pool.Clock = () => now;

        var large = pool.Acquire(1, 0, 36 * MiB);
        // Completed: 32.5 MiB rounds to 36 MiB as well, so the same slab serves it.
        var reused = pool.Acquire(2, 1, 32 * MiB + 512 * 1024);
        Assert.Equal(large.Buffer, reused.Buffer);
        Assert.Equal(36 * MiB, pool.TotalBytes);

        // A 16 MiB request is not served by the free 36 MiB slab: it gets its own.
        var small = pool.Acquire(3, 2, 16 * MiB);
        Assert.NotEqual(large.Buffer, small.Buffer);
        Assert.Equal(52 * MiB, pool.TotalBytes);
    }

    [Fact]
    public void IdleSlabsAreReleasedByTimeOrSerialsPastAFloorKeepingOnlyASmallWarmSet()
    {
        if (!RequireDevice()) return;
        using var pool = new MetalStagingPool(Device.Value);
        long now = 0;
        pool.Clock = () => now;

        // A load burst: fifty 24 MiB slabs and a small one, all in flight in serial 1.
        for (var index = 0; index < 50; index++)
        {
            _ = pool.Acquire(1, 0, 24 * MiB);
        }

        _ = pool.Acquire(1, 0, 1 * MiB);
        Assert.True(pool.TotalBytes >= 1200 * MiB);
        Assert.Equal(1200 * MiB + 1 * MiB, pool.PeakBytes);

        // Completed but recent: the cap holds the free slabs to IdleCapBytes at once.
        now = 10;
        pool.Trim(1);
        Assert.True(pool.IdleBytes <= MetalStagingPool.IdleCapBytes, $"idle={pool.IdleBytes}");
        Assert.True(pool.TotalBytes <= MetalStagingPool.IdleCapBytes, $"total={pool.TotalBytes}");

        // A second later, with no submission past it, only the warm small slab is left.
        now = 10 + MetalStagingPool.IdleMilliseconds;
        pool.Trim(1);
        Assert.Equal(1 * MiB, pool.TotalBytes);
        Assert.Equal(1u, pool.SlabCount);

        // By serials: not while it has been idle less than the serial floor (a slow frame
        // commits many command buffers and stages through the same slabs each frame), then yes.
        _ = pool.Acquire(2, 1, 24 * MiB);
        pool.Trim(2);
        pool.Trim(2 + MetalStagingPool.IdleSerials);
        Assert.Equal(25 * MiB, pool.TotalBytes);
        now += MetalStagingPool.SerialExpiryMilliseconds;
        pool.Trim(3 + MetalStagingPool.IdleSerials);
        Assert.Equal(1 * MiB, pool.TotalBytes);
        output.WriteLine($"created={pool.SlabsCreated} released={pool.SlabsReleased} peak={pool.PeakBytes / MiB} MiB");
    }

    [Fact]
    public void AnIdleSlabTheOsPurgedIsReplacedWhenLentAgainAndWorks()
    {
        if (!RequireDevice()) return;
        using var pool = new MetalStagingPool(Device.Value);
        long now = 0;
        pool.Clock = () => now;

        var first = pool.Acquire(1, 0, 2 * MiB);
        // Seen free at 0, purgeable once idle for the volatile delay.
        pool.Trim(1);
        Assert.Equal(0UL, pool.VolatileBytes);
        now = MetalStagingPool.VolatileAfterMilliseconds;
        pool.Trim(1);
        Assert.Equal(2 * MiB, pool.VolatileBytes);
        // It is volatile: asking again (keep current) reports it so.
        Assert.Equal(MetalStagingPool.PurgeableVolatile, MetalStagingPool.SetPurgeableState(first.Buffer, 1));

        // An unpurged volatile slab comes back as itself.
        var again = pool.Acquire(2, 1, 2 * MiB);
        Assert.Equal(first.Buffer, again.Buffer);
        Assert.Equal(0L, pool.SlabsPurged);
        Assert.Equal(MetalStagingPool.PurgeableNonVolatile, MetalStagingPool.SetPurgeableState(again.Buffer, 1));

        // Volatile again, then emptied the way the OS empties it under pressure.
        pool.Trim(2);
        now += MetalStagingPool.VolatileAfterMilliseconds;
        pool.Trim(2);
        Assert.Equal(2 * MiB, pool.VolatileBytes);
        _ = MetalStagingPool.SetPurgeableState(again.Buffer, MetalStagingPool.PurgeableEmpty);
        var replaced = pool.Acquire(3, 2, 2 * MiB);
        Assert.Equal(1L, pool.SlabsPurged);
        Assert.Equal(2 * MiB, pool.TotalBytes);
        Assert.Equal(MetalStagingPool.PurgeableNonVolatile, MetalStagingPool.SetPurgeableState(replaced.Buffer, 1));

        // The replacement holds what the GPU writes into it.
        var queue = MetalNative.Send(Device.Value, SelNewQueue);
        var commandBuffer = MetalNative.Send(queue, SelCommandBuffer);
        var blit = MetalNative.Send(commandBuffer, SelBlit);
        MetalNative.SendFillBuffer(blit, SelFill, replaced.Buffer, new NsRange { Location = 0, Length = (nuint)replaced.Size }, 0x5A);
        MetalNative.SendVoid(blit, SelEnd);
        MetalNative.SendVoid(commandBuffer, SelCommit);
        MetalNative.SendVoid(commandBuffer, SelWait);
        var contents = (byte*)MetalNative.Send(replaced.Buffer, SelContents);
        Assert.Equal(0x5A, contents[0]);
        Assert.Equal(0x5A, contents[replaced.Size - 1]);
        MetalNative.SendVoid(queue, SelRelease);
    }

    // The slab of a command buffer the GPU has not finished is never lent again, however
    // old it looks by the clock, and it is not made purgeable; once the command buffer
    // completes, the next request gets it back with the GPU's bytes in it.
    [Fact]
    public void ASlabIsNotLentAgainOrMadePurgeableBeforeItsCommandBufferCompletes()
    {
        if (!RequireDevice()) return;
        using var pool = new MetalStagingPool(Device.Value);
        long now = 0;
        pool.Clock = () => now;
        var queue = MetalNative.Send(Device.Value, SelNewQueue);
        var gate = MetalNative.Send(Device.Value, SelNewSharedEvent);
        var completion = MetalNative.Send(Device.Value, SelNewSharedEvent);
        ulong Completed() => MetalV2Native.SendValue64(completion, SelSignaledValue);

        // Serial 1 waits on a gate the CPU opens later, then fills its slab.
        var commandBuffer = MetalNative.Send(queue, SelCommandBuffer);
        var slab = pool.Acquire(1, Completed(), 4 * MiB);
        MetalV2Native.SendEventValue(commandBuffer, SelEncodeWait, gate, 1);
        var blit = MetalNative.Send(commandBuffer, SelBlit);
        MetalNative.SendFillBuffer(blit, SelFill, slab.Buffer, new NsRange { Location = 0, Length = (nuint)slab.Size }, 0xC3);
        MetalNative.SendVoid(blit, SelEnd);
        MetalV2Native.SendEventValue(commandBuffer, SelEncodeSignal, completion, 1);
        MetalNative.SendVoid(commandBuffer, SelCommit);

        // Recording serial 2 while serial 1 is still pending, long after by the clock.
        now = 10 * MetalStagingPool.IdleMilliseconds;
        Assert.Equal(0UL, Completed());
        pool.Trim(Completed());
        Assert.Equal(0UL, pool.VolatileBytes);
        Assert.Equal(4 * MiB, pool.TotalBytes);
        var other = pool.Acquire(2, Completed(), 4 * MiB);
        Assert.NotEqual(slab.Buffer, other.Buffer);
        Assert.Equal(MetalStagingPool.PurgeableNonVolatile, MetalStagingPool.SetPurgeableState(slab.Buffer, 1));

        MetalNative.SendVoid(gate, SelSetSignaledValue, (nint)1);
        MetalNative.SendVoid(commandBuffer, SelWait);
        Assert.Equal(1UL, Completed());
        var back = pool.Acquire(3, Completed(), 4 * MiB);
        Assert.Equal(slab.Buffer, back.Buffer);
        var contents = (byte*)MetalNative.Send(back.Buffer, SelContents);
        Assert.Equal(0xC3, contents[0]);
        Assert.Equal(0xC3, contents[back.Size - 1]);

        MetalNative.SendVoid(gate, SelRelease);
        MetalNative.SendVoid(completion, SelRelease);
        MetalNative.SendVoid(queue, SelRelease);
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Staging slabs run on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
