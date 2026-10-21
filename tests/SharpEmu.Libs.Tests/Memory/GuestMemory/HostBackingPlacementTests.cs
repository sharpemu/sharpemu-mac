// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.HLE.Host;
using Xunit;
using static SharpEmu.Libs.Tests.Memory.HostViews.HostViewTestSupport;

namespace SharpEmu.Libs.Tests.Memory.GuestMemory;

[Collection(GuestMemoryStateCollection.Name)]
public sealed unsafe class HostBackingPlacementTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FullMacBackingStaysOutsideGuestApplicationMemory(bool chunked)
    {
        if (!HostBackendUnderTest.SetChunkedBacking(chunked)) return;
        try
        {
            var host = HostViewMemory.Create();
            using var backing = new SharedBackingViews(host, GuestMemoryLayout.BackingBytes);
            Assert.True(backing.IsAvailable);
            Assert.True(backing.AliasBase >= 0x100_0000_0000,
                $"The private alias at 0x{backing.AliasBase:X} occupies guest application VA.");
            var view = ReserveFreeHole(host, host.Granularity);
            var offset = GuestMemoryLayout.BackingBytes - host.Granularity;
            try
            {
                Assert.True(backing.TryMapReservedRange(view, host.Granularity, offset,
                    HostPageProtection.ReadWrite, out var failure), failure.ToString());
                *(ulong*)(backing.AliasBase + offset) = Marker;
                Assert.Equal(Marker, *(ulong*)view);
                *(ulong*)(view + host.Granularity - 8) = ~Marker;
                Assert.Equal(~Marker, *(ulong*)(backing.AliasBase + GuestMemoryLayout.BackingBytes - 8));
            }
            finally
            {
                if (backing.Contains(view, host.Granularity)) Assert.True(backing.Unmap(view, host.Granularity, out _));
                Assert.True(host.FreeHole(view, host.Granularity));
            }
        }
        finally { HostBackendUnderTest.SetChunkedBacking(null); }
    }
}
