// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using System.Runtime.CompilerServices;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using Xunit;

namespace SharpEmu.Host.Memory.Tests;

/// <summary>
/// A guest allocation mapped in several calls is one contiguous run of backing
/// bytes. A range that crosses the edge between two of those mappings therefore has
/// one host address, and a caller that maps the backing no-copy has to be told so —
/// otherwise every such range is staged through a copy. Ordinary managed/pinned
/// bookkeeping only: nothing here maps host pages.
/// </summary>
public sealed class BackingAliasSpanTests
{
    private const ulong Page = 4096;

    [Fact]
    public void ARangeCrossingTwoAdjacentMappingsResolvesToOneAlias()
    {
        var host = new PinnedViews();
        using var views = new SharedBackingViews(host, 3 * Page);
        Assert.True(views.TryMapReservedRange(0x10_0000, Page, 0, HostPageProtection.ReadWrite, out _));
        Assert.True(views.TryMapReservedRange(0x10_1000, Page, Page, HostPageProtection.ReadWrite, out _));

        // The single-mapping test refuses it; the span test resolves it.
        Assert.False(views.TryResolveAlias(0x10_0000 + Page - 64, 128, out _));
        Assert.True(views.TryResolveAliasSpan(0x10_0000 + Page - 64, 128, out var alias));
        Assert.True(views.TryResolveAlias(0x10_0000, 1, out var start));
        Assert.Equal(start + Page - 64, alias);
    }

    // Two live mappings that share guest bytes hand the title two allocations over one
    // piece of memory; whichever writes second rewrites the other's data silently.
    [Fact]
    public void AMappingOverALiveOneIsReported()
    {
        var host = new PinnedViews();
        using var views = new SharedBackingViews(host, 4 * Page);
        Assert.True(views.TryMapReservedRange(0x10_0000, 2 * Page, 0, HostPageProtection.ReadWrite, out _));
        Assert.Equal(0, views.OverlappingViewRecords);

        // Starts inside the first mapping's second page.
        Assert.True(views.TryMapReservedRange(0x10_1000, Page, 2 * Page, HostPageProtection.ReadWrite, out _));
        Assert.Equal(1, views.OverlappingViewRecords);
    }

    // Two guest addresses over the same backing bytes are one piece of memory the title
    // sees twice: a write through one rewrites what the other holds.
    [Fact]
    public void AMappingOverBackingBytesALiveOneHoldsIsReported()
    {
        var host = new PinnedViews();
        using var views = new SharedBackingViews(host, 4 * Page);
        Assert.True(views.TryMapReservedRange(0x10_0000, 2 * Page, 0, HostPageProtection.ReadWrite, out _));
        Assert.Equal(0, views.AliasedBackingRecords);

        // A different guest address, the backing page the first mapping already holds.
        Assert.True(views.TryMapReservedRange(0x30_0000, Page, Page, HostPageProtection.ReadWrite, out _));
        Assert.Equal(1, views.AliasedBackingRecords);
        // And it is not counted as a guest-range overlap, which it is not.
        Assert.Equal(0, views.OverlappingViewRecords);
    }

    [Fact]
    public void MappingsOverDistinctBackingBytesAreNotReportedAsAliasing()
    {
        var host = new PinnedViews();
        using var views = new SharedBackingViews(host, 4 * Page);
        Assert.True(views.TryMapReservedRange(0x10_0000, Page, 0, HostPageProtection.ReadWrite, out _));
        Assert.True(views.TryMapReservedRange(0x30_0000, Page, Page, HostPageProtection.ReadWrite, out _));
        Assert.True(views.TryMapReservedRange(0x50_0000, 2 * Page, 2 * Page, HostPageProtection.ReadWrite, out _));
        Assert.Equal(0, views.AliasedBackingRecords);
    }

    [Fact]
    public void MappingsThatOnlyTouchAreNotReportedAsOverlapping()
    {
        var host = new PinnedViews();
        using var views = new SharedBackingViews(host, 4 * Page);
        Assert.True(views.TryMapReservedRange(0x10_0000, Page, 0, HostPageProtection.ReadWrite, out _));
        Assert.True(views.TryMapReservedRange(0x10_1000, Page, Page, HostPageProtection.ReadWrite, out _));
        Assert.True(views.TryMapReservedRange(0x10_2000, Page, 2 * Page, HostPageProtection.ReadWrite, out _));
        // And one below the first, ending exactly where it starts.
        Assert.True(views.TryMapReservedRange(0x0F_F000, Page, 3 * Page, HostPageProtection.ReadWrite, out _));
        Assert.Equal(0, views.OverlappingViewRecords);
    }

    [Fact]
    public void MappingsThatAreNotAdjacentInGuestSpaceDoNotResolveAsOneRange()
    {
        var host = new PinnedViews();
        using var views = new SharedBackingViews(host, 3 * Page);
        Assert.True(views.TryMapReservedRange(0x10_0000, Page, 0, HostPageProtection.ReadWrite, out _));
        // A page of guest address space between them.
        Assert.True(views.TryMapReservedRange(0x10_2000, Page, Page, HostPageProtection.ReadWrite, out _));
        Assert.False(views.TryResolveAliasSpan(0x10_0000 + Page - 64, 128, out _));
    }

    [Fact]
    public void MappingsThatAreNotAdjacentInTheBackingDoNotResolveAsOneRange()
    {
        var host = new PinnedViews();
        using var views = new SharedBackingViews(host, 3 * Page);
        Assert.True(views.TryMapReservedRange(0x10_0000, Page, 0, HostPageProtection.ReadWrite, out _));
        // Adjacent in guest space but a different part of the backing.
        Assert.True(views.TryMapReservedRange(0x10_1000, Page, 2 * Page, HostPageProtection.ReadWrite, out _));
        Assert.False(views.TryResolveAliasSpan(0x10_0000 + Page - 64, 128, out _));
    }

    [Fact]
    public void ARangeInsideOneMappingStillResolves()
    {
        var host = new PinnedViews();
        using var views = new SharedBackingViews(host, 3 * Page);
        Assert.True(views.TryMapReservedRange(0x10_0000, Page, 0, HostPageProtection.ReadWrite, out _));
        Assert.True(views.TryResolveAliasSpan(0x10_0000 + 16, 64, out var alias));
        Assert.True(views.TryResolveAlias(0x10_0000 + 16, 64, out var single));
        Assert.Equal(single, alias);
    }

    [Fact]
    public void ARangeRunningPastTheLastMappingDoesNotResolve()
    {
        var host = new PinnedViews();
        using var views = new SharedBackingViews(host, 3 * Page);
        Assert.True(views.TryMapReservedRange(0x10_0000, Page, 0, HostPageProtection.ReadWrite, out _));
        Assert.False(views.TryResolveAliasSpan(0x10_0000 + Page - 64, 128, out _));
        Assert.False(views.TryResolveAliasSpan(0x0F_F000, Page, out _));
    }

    // Enough of IHostViewMemory to keep the view bookkeeping honest; nothing maps.
    private sealed unsafe class PinnedViews : IHostViewMemory
    {
        private readonly byte[] _storage = GC.AllocateArray<byte>(5 * (int)Page, pinned: true);

        public ulong PageSize => Page;

        public ulong Granularity => Page;

        public bool TryCreateBacking(ulong size, out HostBackingObject? backing, out HostViewFailure failure)
        {
            var start = (ulong)Unsafe.AsPointer(ref _storage[0]);
            var aligned = (start + Page - 1) & ~(Page - 1);
            backing = new HostBackingObject(aligned, size, Page, _ =>
            {
                GC.KeepAlive(_storage);
                return new HostBackingReleaseResult(HostBackingReleaseStatus.Released);
            });
            failure = HostViewFailure.None;
            return true;
        }

        public ulong ReserveHole(ulong address, ulong size) => address;

        public bool SplitHole(ulong address, ulong size) => true;

        public bool JoinHoles(ulong address, ulong size) => true;

        public bool FreeHole(ulong address, ulong size) => true;

        public bool TryMapView(
            HostBackingObject backing,
            ulong address,
            ulong offset,
            ulong size,
            HostPageProtection protection,
            out HostViewFailure failure)
        {
            failure = HostViewFailure.None;
            return true;
        }

        public bool UnmapView(ulong address, ulong size) => true;

        public bool CommitPrivate(ulong address, ulong size, HostPageProtection protection) => true;

        public bool ReleasePrivate(ulong address, ulong size) => true;

        public bool ChangeAccess(ulong address, ulong size, HostPageProtection protection) => true;
        public IReadOnlyList<HostPageAccess> QueryAccess(ulong address, ulong size) => []; // access is never tracked here

        public bool FreeOwnedRange(ulong address, ulong size) => true;
    }
}
