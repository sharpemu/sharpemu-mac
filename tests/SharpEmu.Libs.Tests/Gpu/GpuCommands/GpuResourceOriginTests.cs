// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.GpuCommands;

public sealed class GpuResourceOriginTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void RingRequiresPositiveCapacity(int capacity) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new GpuResourceOriginRing(capacity));

    [Fact]
    public void RingKeepsOnlyTheLast128WritesInSequenceOrder()
    {
        var ring = new GpuResourceOriginRing(128);
        Assert.Empty(ring.Snapshot());
        for (uint value = 0; value < 300; value++)
            ring.Add(new GpuResourceOriginWrite { Value = value });
        var writes = ring.Snapshot();
        Assert.Equal(128, writes.Length);
        Assert.Equal(Enumerable.Range(172, 128).Select(value => (uint)value), writes.Select(write => write.Value));
        Assert.Equal(Enumerable.Range(173, 128).Select(value => (long)value), writes.Select(write => write.Sequence));
        ring.Add(new GpuResourceOriginWrite { Value = 300 });
        Assert.Equal(172u, writes[0].Value); // A failure-report snapshot cannot be overwritten by producers.
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CaptureRetainsPacketSourceAndResultingPixelState(bool table)
    {
        var banks = new RegisterBanks(message => new InvalidOperationException(message));
        banks.Shader.Pixel.Address = 0x808B440000;
        banks.Shader.Pixel.UserScalars.Set(28, 0xE90AFF18, UserScalarKind.Unknown);
        banks.Shader.Pixel.UserScalars.Set(29, 0x80, UserScalarKind.Unknown);
        var packet = new PacketContext(0xC0047600, 0x8123400000, 3, 16, 32);
        var write = GpuResourceOrigin.Capture(2, 1234, packet, table, 0x8123400010,
            GpuResourceOrigin.PointerLow, 0xABCD0028, 0xE90AFF18, 2, 6, banks);
        banks.Shader.Pixel.UserScalars.Set(28, 0, UserScalarKind.Unknown);
        banks.Shader.Pixel.Address = 0;

        Assert.Equal(2, write.Queue);
        Assert.Equal(1234UL, write.Submit);
        Assert.Equal(packet.PacketAddress, write.PacketAddress);
        Assert.Equal(packet.Header, write.Header);
        Assert.Equal(table, write.Table);
        Assert.Equal(0x8123400010UL, write.SourceAddress);
        Assert.Equal(GpuResourceOrigin.PointerLow, write.Offset);
        Assert.Equal(0xABCD0028u, write.RawOffset);
        Assert.Equal(2u, write.Entry);
        Assert.Equal(6u, write.Count);
        Assert.Equal(0xE90AFF18u, write.Value);
        Assert.Equal(0x808B440000UL, write.ShaderAddress);
        Assert.Equal(0xE90AFF18u, write.PointerLow);
        Assert.Equal(0x80u, write.PointerHigh);
    }

    [Fact]
    public void ConcurrentProducersStillProduceABoundedOrderedSnapshot()
    {
        var ring = new GpuResourceOriginRing(128);
        Parallel.For(0, 1000, value => ring.Add(new GpuResourceOriginWrite { Value = (uint)value }));
        var writes = ring.Snapshot();
        Assert.Equal(128, writes.Length);
        Assert.Equal(Enumerable.Range(873, 128).Select(value => (long)value), writes.Select(write => write.Sequence));
    }
}
