// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
using System.Runtime.CompilerServices;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using Xunit;

namespace SharpEmu.Host.Memory.Tests;

/// <summary>
/// A guest `memcpy` of a contiguous guest range whose backing is not contiguous. The
/// copy is served by pairing the source's backing pieces against the destination's, and
/// the two partitions do not share boundaries: a copy that starts at a different offset
/// inside its page on each side crosses its pieces at different points in the payload.
/// Mispairing them writes the right bytes to the wrong backing address, which does not
/// fault — it rewrites whatever the guest put next door.
///
/// Every test here also checks the bytes either side of the destination, because an
/// off-by-one in the pairing shows up there first.
/// </summary>
public sealed class BackingCopyFragmentationTests
{
    private const ulong Page = 4096;
    private const int Pages = 16;

    private static readonly ulong SourceBase = 0x10_0000;
    private static readonly ulong DestinationBase = 0x20_0000;

    // Guest-contiguous, backing-fragmented on both sides, and the two sides are split at
    // different backing distances so no pairing is accidental.
    private static SharedBackingViews MapFragmented(PinnedViews host)
    {
        var views = new SharedBackingViews(host, Pages * Page);
        // Source: two guest pages whose backing pages are not neighbours.
        Assert.True(views.TryMapReservedRange(SourceBase, Page, 0 * Page, HostPageProtection.ReadWrite, out _));
        Assert.True(views.TryMapReservedRange(SourceBase + Page, Page, 5 * Page, HostPageProtection.ReadWrite, out _));
        Assert.True(views.TryMapReservedRange(SourceBase + (2 * Page), Page, 2 * Page, HostPageProtection.ReadWrite, out _));
        // Destination: likewise, and from a different part of the backing.
        Assert.True(views.TryMapReservedRange(DestinationBase, Page, 9 * Page, HostPageProtection.ReadWrite, out _));
        Assert.True(views.TryMapReservedRange(DestinationBase + Page, Page, 12 * Page, HostPageProtection.ReadWrite, out _));
        Assert.True(views.TryMapReservedRange(DestinationBase + (2 * Page), Page, 7 * Page, HostPageProtection.ReadWrite, out _));
        return views;
    }

    private static byte[] Pattern(int length, int seed)
    {
        var bytes = new byte[length];
        for (var index = 0; index < length; index++)
        {
            bytes[index] = (byte)((index * 31) + seed);
        }

        return bytes;
    }

    private static byte[] Read(SharedBackingViews views, ulong address, int length)
    {
        var bytes = new byte[length];
        if (length != 0)
        {
            Assert.True(views.TryReadBacking(address, bytes));
        }

        return bytes;
    }

    // sourceOffset and destinationOffset are deliberately different, so the payload
    // crosses the source's page edge at one data offset and the destination's at another.
    [Theory]
    [InlineData(0x800, 0x100, 0x1800)]   // both sides cross, at 0x800 and 0xF00
    [InlineData(0x100, 0x800, 0x1800)]   // the same, the other way round
    [InlineData(0xFF8, 0x008, 0x1010)]   // a source edge eight bytes in
    [InlineData(0x008, 0xFF8, 0x1010)]   // a destination edge eight bytes in
    [InlineData(0xFFF, 0x001, 0x1002)]   // one byte either side of each edge
    [InlineData(0x000, 0x000, 0x3000)]   // whole, aligned, every piece paired one to one
    [InlineData(0x7F0, 0x7F0, 0x0020)]   // the eight-byte tail straddling both edges alike
    public void ACopyAcrossFragmentedBackingLandsEveryByteWhereItBelongs(
        int sourceOffset, int destinationOffset, int length)
    {
        var host = new PinnedViews(Pages);
        using var views = MapFragmented(host);

        // Fill both guest ranges so the destination starts out holding something else.
        var payload = Pattern(length, seed: 7);
        var before = Pattern(3 * (int)Page, seed: 101);
        Assert.True(views.TryWriteBacking(DestinationBase, before));
        Assert.True(views.TryWriteBacking(SourceBase + (ulong)sourceOffset, payload));

        Assert.True(views.TryCopyBacking(
            DestinationBase + (ulong)destinationOffset, SourceBase + (ulong)sourceOffset, (ulong)length));

        Assert.Equal(payload, Read(views, DestinationBase + (ulong)destinationOffset, length));

        // Nothing outside the destination moved.
        Assert.Equal(
            before.AsSpan(0, destinationOffset).ToArray(),
            Read(views, DestinationBase, destinationOffset));
        var tail = (3 * (int)Page) - destinationOffset - length;
        Assert.Equal(
            before.AsSpan(destinationOffset + length, tail).ToArray(),
            Read(views, DestinationBase + (ulong)(destinationOffset + length), tail));
    }

    // The same guest range on both sides, shifted: libc memcpy is called with overlap by
    // real titles and the copy has to behave as memmove does, through the backing alias.
    [Theory]
    [InlineData(0x800, 0x808, 0x1000)]   // forward overlap across a page edge
    [InlineData(0x808, 0x800, 0x1000)]   // backward overlap across a page edge
    [InlineData(0x000, 0x008, 0x2000)]   // overlap spanning all three pieces
    [InlineData(0xFF8, 0x1000, 0x0010)]  // eight bytes, straddling the edge, overlapping
    public void AnOverlappingCopyInsideFragmentedBackingKeepsMemmoveSemantics(
        int sourceOffset, int destinationOffset, int length)
    {
        var host = new PinnedViews(Pages);
        using var views = MapFragmented(host);

        var seeded = Pattern(3 * (int)Page, seed: 23);
        Assert.True(views.TryWriteBacking(SourceBase, seeded));
        var expected = seeded.AsSpan(sourceOffset, length).ToArray();

        Assert.True(views.TryCopyBacking(
            SourceBase + (ulong)destinationOffset, SourceBase + (ulong)sourceOffset, (ulong)length));

        Assert.Equal(expected, Read(views, SourceBase + (ulong)destinationOffset, length));
    }

    // The fault that started this: a free-list link rewritten with float payload. The
    // copy's own bytes are fine; what matters is that nothing lands outside its range.
    [Fact]
    public void ACopyNeverWritesTheEightBytesPastItsEnd()
    {
        var host = new PinnedViews(Pages);
        using var views = MapFragmented(host);

        var guard = new byte[8];
        Assert.True(views.TryWriteBacking(DestinationBase, Pattern(3 * (int)Page, seed: 55)));
        for (var length = 1; length <= 0x40; length++)
        {
            // Finishing one byte before a backing piece edge is where an off-by-one shows.
            var destination = DestinationBase + Page - (ulong)length;
            var sentinel = Read(views, destination + (ulong)length, guard.Length);
            Assert.True(views.TryCopyBacking(destination, SourceBase + 0x40, (ulong)length));
            Assert.Equal(sentinel, Read(views, destination + (ulong)length, guard.Length));
        }
    }

    private sealed unsafe class PinnedViews : IHostViewMemory
    {
        private readonly byte[] _storage;

        public PinnedViews(int pages) =>
            _storage = GC.AllocateArray<byte>((pages + 2) * (int)Page, pinned: true);

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
