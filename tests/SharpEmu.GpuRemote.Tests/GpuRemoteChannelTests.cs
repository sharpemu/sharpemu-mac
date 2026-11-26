// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;

namespace SharpEmu.GpuRemote.Tests;

/// <summary>Both ends of a region in one process: the protocol without a second process.</summary>
public sealed unsafe class GpuRemoteChannelTests
{
    private const long Second = 1_000_000_000;

    [MacOSFact]
    public void ClientAnswersTheHandshakePingsAndShutdown()
    {
        using var host = GpuRemoteChannel.Create(SharedMemoryRegion.CreateUniqueName("setest"));
        var hello = GpuPeerInfo.DescribeThisProcess();
        Assert.True(host.Outbound.TryWrite((uint)GpuRemoteMessage.Hello, in hello));

        using var client = GpuProcessClient.Attach(host.Name, parentProcessId: 0, TimeSpan.FromSeconds(5));
        Assert.Equal(Environment.ProcessId, client.Host.ProcessId);
        Assert.Equal(GpuRemoteChannel.PeerRunning, host.PeerState);
        var exitCode = -1;
        var loop = new Thread(() => exitCode = client.Run());
        loop.Start();

        Assert.True(host.Inbound.WaitForData(5 * Second));
        Assert.True(host.Inbound.TryRead(out var type, out var payload));
        Assert.Equal((uint)GpuRemoteMessage.HelloAck, type);
        Assert.Equal(GpuRemoteProtocol.Version, RingReader.Read<HelloMessage>(payload).ProtocolVersion);
        host.Inbound.Advance();

        var ping = new PingMessage { Sequence = 42, Timestamp = 7 };
        Assert.True(host.Outbound.TryWrite((uint)GpuRemoteMessage.Ping, in ping));
        Assert.True(host.Inbound.WaitForData(5 * Second));
        Assert.True(host.Inbound.TryRead(out type, out payload));
        Assert.Equal((uint)GpuRemoteMessage.Pong, type);
        Assert.Equal(42ul, RingReader.Read<PingMessage>(payload).Sequence);
        host.Inbound.Advance();

        var request = new byte[] { 9, 8, 7 };
        var response = new byte[16];
        Assert.Equal(
            SyncCallStatus.Completed,
            host.SyncCalls.Call((uint)GpuRemoteSyncOpcode.Ping, request, response, out var length, out var status, 5 * Second));
        Assert.Equal(3, length);
        Assert.Equal(0, status);
        Assert.Equal(request, response[..3]);

        Assert.True(host.Outbound.TryWrite((uint)GpuRemoteMessage.Shutdown, ReadOnlySpan<byte>.Empty));
        Assert.True(loop.Join(5_000));
        Assert.Equal(GpuProcessClient.ExitClean, exitCode);
    }

    [MacOSFact]
    public void ClientLoopEndsWhenTheHostIsMarkedGone()
    {
        using var host = GpuRemoteChannel.Create(SharedMemoryRegion.CreateUniqueName("setest"));
        var hello = GpuPeerInfo.DescribeThisProcess();
        Assert.True(host.Outbound.TryWrite((uint)GpuRemoteMessage.Hello, in hello));
        using var client = GpuProcessClient.Attach(host.Name, parentProcessId: 0, TimeSpan.FromSeconds(5));
        var exitCode = -1;
        var loop = new Thread(() => exitCode = client.Run());
        loop.Start();
        Thread.Sleep(50);

        // What the parent watch does when process A dies.
        client.Channel.MarkPeerExited();
        Assert.True(loop.Join(5_000));
        Assert.Equal(GpuProcessClient.ExitHostGone, exitCode);
    }

    [MacOSFact]
    public void AttachRefusesAnotherProtocolVersion()
    {
        using var host = GpuRemoteChannel.Create(SharedMemoryRegion.CreateUniqueName("setest"));
        using (var raw = SharedMemoryRegion.Open(host.Name))
        {
            ((RegionHeader*)raw.Pointer)->Version = GpuRemoteProtocol.Version + 1;
        }

        var exception = Assert.Throws<InvalidDataException>(() => GpuRemoteChannel.Attach(host.Name));
        Assert.Contains("protocol version", exception.Message);
    }

    [Theory]
    [InlineData(4u)]
    [InlineData(5u)]
    [InlineData(6u)]
    [InlineData(7u)]
    [InlineData(8u)]
    public void CombinedLayoutRefusesEveryIndependentRecoveryProtocol(uint version)
    {
        if (!OperatingSystem.IsMacOS()) return;
        using var host = GpuRemoteChannel.Create(SharedMemoryRegion.CreateUniqueName("setest"));
        using (var raw = SharedMemoryRegion.Open(host.Name))
            ((RegionHeader*)raw.Pointer)->Version = version;
        Assert.Equal(9u, GpuRemoteProtocol.Version);
        Assert.Throws<InvalidDataException>(() => GpuRemoteChannel.Attach(host.Name));
    }

    [MacOSFact]
    public void AttachRefusesSomethingThatIsNotARegion()
    {
        using var other = SharedMemoryRegion.Create(SharedMemoryRegion.CreateUniqueName("setest"), 4096);
        Assert.Throws<InvalidDataException>(() => GpuRemoteChannel.Attach(other.Name));
    }

    [Fact]
    public void LayoutKeepsEveryBlockOnItsOwnCacheLines()
    {
        var size = GpuRemoteChannel.RegionSize(new GpuRemoteLayout(), out var hostToGpu, out var gpuToHost, out var sync);
        Assert.Equal(0, hostToGpu % 128);
        Assert.Equal(0, gpuToHost % 128);
        Assert.Equal(0, sync % 128);
        Assert.True(hostToGpu >= RegionHeader.Size);
        Assert.True(gpuToHost >= hostToGpu + SpscRing.BlockSize(1 << 20));
        Assert.True(sync >= gpuToHost + SpscRing.BlockSize(256 << 10));
        Assert.True(size >= sync + SyncCallTable.BlockSize(SyncCallTable.MaxSlots, 512));
    }
}
