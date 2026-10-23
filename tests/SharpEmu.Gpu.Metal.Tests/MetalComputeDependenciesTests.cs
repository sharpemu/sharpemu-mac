// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed class MetalComputeDependenciesTests
{
    private static MetalComputeFootprint Range(ulong start, ulong size, bool written)
    {
        var footprint = new MetalComputeFootprint();
        footprint.Guest(start, size, written);
        return footprint;
    }

    [Theory]
    [InlineData(true, false)] // RAW
    [InlineData(true, true)]  // WAW, including shared atomics
    [InlineData(false, true)] // WAR
    public void OverlappingAccessesRequireABarrier(bool firstWrite, bool secondWrite)
    {
        var tracker = new MetalComputeDependencies();
        tracker.Record(Range(0x1000, 64, firstWrite));
        Assert.True(tracker.NeedsBarrier(Range(0x1020, 64, secondWrite)));
    }

    [Theory]
    [InlineData(0x1000ul, 64ul, false)] // read/read
    [InlineData(0x1040ul, 64ul, true)]  // adjacent
    [InlineData(0x0FC0ul, 64ul, true)]  // adjacent below
    [InlineData(0x1000ul, 0ul, true)]   // empty
    public void IndependentAccessesStayInTheSameEpoch(ulong start, ulong size, bool written)
    {
        var tracker = new MetalComputeDependencies();
        tracker.Record(Range(0x1000, 64, false));
        Assert.False(tracker.NeedsBarrier(Range(start, size, written)));
    }

    [Theory]
    [InlineData(0x103Ful, 1ul)]
    [InlineData(0x0FFFul, 2ul)]
    [InlineData(0x0FFFul, 128ul)]
    public void AOneByteOverlapAndContainingRangesConflict(ulong start, ulong size)
    {
        var tracker = new MetalComputeDependencies();
        tracker.Record(Range(0x1000, 64, true));
        Assert.True(tracker.NeedsBarrier(Range(start, size, false)));
    }

    [Fact]
    public void IndependentDispatchDoesNotForgiveAnOlderDependency()
    {
        var tracker = new MetalComputeDependencies();
        tracker.Record(Range(0x1000, 64, true));
        var independent = Range(0x2000, 64, true);
        Assert.False(tracker.NeedsBarrier(independent));
        tracker.Record(independent);
        Assert.True(tracker.NeedsBarrier(Range(0x1000, 4, false)));
        tracker.Reset(); // only after the caller encodes a real scope barrier
        Assert.False(tracker.NeedsBarrier(Range(0x1000, 4, false)));
    }

    [Fact]
    public void ImageAliasesAndBufferViewsMeetInGuestCoordinates()
    {
        var tracker = new MetalComputeDependencies();
        var imageView = Range(0x1000, 4096, true);
        imageView.Accesses.Add(new(MetalComputeAddressSpace.Texture, 1, 0, 1, true));
        tracker.Record(imageView);
        var aliasView = Range(0x1800, 4096, false);
        aliasView.Accesses.Add(new(MetalComputeAddressSpace.Texture, 2, 0, 1, false));
        Assert.True(tracker.NeedsBarrier(aliasView));
        Assert.True(tracker.NeedsBarrier(Range(0x1FFF, 1, false)));
    }

    [Fact]
    public void NativeBufferAliasesConflictEvenWithDifferentGuestAddresses()
    {
        var tracker = new MetalComputeDependencies();
        var first = Range(0x1000, 64, true);
        first.Accesses.Add(new(MetalComputeAddressSpace.Buffer, 42, 128, 64, true));
        tracker.Record(first);
        var alias = Range(0x2000, 64, false);
        alias.Accesses.Add(new(MetalComputeAddressSpace.Buffer, 42, 160, 64, false));
        Assert.True(tracker.NeedsBarrier(alias));
    }

    [Fact]
    public void PhysicalBackingAliasesConflictAcrossDifferentBuffersAndGuestAddresses()
    {
        var tracker = new MetalComputeDependencies();
        var first = Range(0x1000, 64, true);
        first.Accesses.Add(new(MetalComputeAddressSpace.Backing, 0, 0x9000, 64, true));
        tracker.Record(first);
        var alias = Range(0x2000, 64, false);
        alias.Accesses.Add(new(MetalComputeAddressSpace.Backing, 0, 0x9020, 64, false));
        Assert.True(tracker.NeedsBarrier(alias));
    }

    [Fact]
    public void UnknownWritesOrderEverythingOnBothSides()
    {
        var tracker = new MetalComputeDependencies();
        tracker.Record(Range(0x1000, 4, false));
        var unknown = new MetalComputeFootprint { UnknownWrite = true };
        Assert.True(tracker.NeedsBarrier(unknown));
        tracker.Reset();
        tracker.Record(unknown);
        Assert.True(tracker.NeedsBarrier(Range(0x900000, 4, false)));
        Assert.True(tracker.NeedsBarrier(Range(0x900000, 4, true)));
    }

    [Fact]
    public void UnknownReadsOrderWritesButCanOverlapReads()
    {
        var tracker = new MetalComputeDependencies();
        tracker.Record(new MetalComputeFootprint { UnknownRead = true });
        Assert.False(tracker.NeedsBarrier(Range(0x1000, 4, false)));
        Assert.True(tracker.NeedsBarrier(Range(0x1000, 4, true)));
        tracker.Reset();
        tracker.Record(Range(0x1000, 4, true));
        Assert.True(tracker.NeedsBarrier(new MetalComputeFootprint { UnknownRead = true }));
    }

    [Fact]
    public void GpuProducedIndirectArgumentsAreAReadDependency()
    {
        var tracker = new MetalComputeDependencies();
        tracker.Record(Range(0x1000, 12, true));
        var indirectConsumer = Range(0x8000, 128, true);
        indirectConsumer.Guest(0x1000, 12, false);
        Assert.True(tracker.NeedsBarrier(indirectConsumer));
    }

    [Fact]
    public void OverflowingRangesBecomeUnknownInsteadOfWrapping()
    {
        var tracker = new MetalComputeDependencies();
        tracker.Record(Range(ulong.MaxValue - 2, 8, true));
        Assert.True(tracker.NeedsBarrier(Range(0x1000, 4, false)));
    }
}
