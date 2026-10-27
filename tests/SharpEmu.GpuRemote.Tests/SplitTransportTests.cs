// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using Xunit;

namespace SharpEmu.GpuRemote.Tests;

/// <summary>
/// What protocol 2 added for the split: the B → A call table, the application block with its
/// shared counters, one-call-at-a-time serving, and the client's host-service hook.
/// </summary>
public sealed unsafe class SplitTransportTests
{
    private const long Second = 1_000_000_000;

    [Fact]
    public void LayoutPlacesTheReverseTableAndTheApplicationBlockOnTheirOwnLines()
    {
        var layout = new GpuRemoteLayout { ApplicationBytes = 1000 };
        var size = GpuRemoteChannel.RegionSize(layout, out _, out _, out var sync, out var reverse, out var application);
        Assert.Equal(0, reverse % 128);
        Assert.Equal(0, application % 128);
        Assert.True(reverse >= sync + SyncCallTable.BlockSize(layout.SyncSlotCount, layout.SyncSlotPayloadBytes));
        Assert.True(application >= reverse + SyncCallTable.BlockSize(layout.ReverseSyncSlotCount, layout.ReverseSyncSlotPayloadBytes));
        Assert.True(size >= application + 1000);
    }

    [MacOSFact]
    public void BothViewsShareTheApplicationBlockAndTheReverseTable()
    {
        using var host = GpuRemoteChannel.Create(SharedMemoryRegion.CreateUniqueName("setest"), new GpuRemoteLayout { ApplicationBytes = 4096 });
        using var gpu = GpuRemoteChannel.Attach(host.Name);
        Assert.True(host.Application != null);
        Assert.Equal(4096, host.ApplicationBytes);
        Assert.Equal(4096, gpu.ApplicationBytes);
        host.Application[100] = 0x5A;
        Assert.Equal(0x5A, gpu.Application[100]);

        // B calls, A serves: the direction the split's video-out questions take.
        using var stop = new ManualResetEventSlim(false);
        var server = new Thread(() =>
        {
            while (!stop.IsSet)
            {
                if (host.ReverseSyncCalls.WaitForRequests(Second / 10))
                {
                    while (host.ReverseSyncCalls.ServeOne(static (opcode, request, response, out status) =>
                    {
                        status = (int)opcode;
                        var value = request[0];
                        response[0] = (byte)(value * 2);
                        return 1;
                    }))
                    {
                    }
                }
            }
        });
        server.Start();
        try
        {
            var response = new byte[4];
            Assert.Equal(SyncCallStatus.Completed, gpu.ReverseSyncCalls.Call(11, [21], response, out var length, out var status, 5 * Second));
            Assert.Equal(1, length);
            Assert.Equal(11, status);
            Assert.Equal(42, response[0]);
        }
        finally
        {
            stop.Set();
            host.ReverseSyncCalls.WakeAll();
            server.Join();
        }
    }

    [MacOSFact]
    public void ASharedCounterWakesItsWaiterAndNeverGoesBack()
    {
        using var host = GpuRemoteChannel.Create(SharedMemoryRegion.CreateUniqueName("setest"), new GpuRemoteLayout { ApplicationBytes = 4096 });
        using var gpu = GpuRemoteChannel.Attach(host.Name);
        var published = SharedCounter.InApplicationBlock(host, 256);
        var watched = SharedCounter.InApplicationBlock(gpu, 256);

        var reached = false;
        var waiter = new Thread(() => reached = watched.WaitAtLeast(5, 10 * Second, spinNanoseconds: 0));
        waiter.Start();
        Thread.Sleep(30);
        published.Publish(3);
        Thread.Sleep(30);
        Assert.True(waiter.IsAlive);
        published.Publish(7);
        Assert.True(waiter.Join(5_000));
        Assert.True(reached);

        published.Publish(2);
        Assert.Equal(7ul, watched.Value);
        Assert.False(watched.WaitAtLeast(8, Second / 50));
    }

    [MacOSFact]
    public void ACounterWaitEndsSoonAfterThePeerExits()
    {
        using var host = GpuRemoteChannel.Create(SharedMemoryRegion.CreateUniqueName("setest"), new GpuRemoteLayout { ApplicationBytes = 4096 });
        using var gpu = GpuRemoteChannel.Attach(host.Name);
        var watched = SharedCounter.InApplicationBlock(gpu, 0);
        var result = true;
        var watch = Stopwatch.StartNew();
        var waiter = new Thread(() => result = watched.WaitAtLeast(1, -1, spinNanoseconds: 0));
        waiter.Start();
        Thread.Sleep(30);
        gpu.MarkPeerExited();
        Assert.True(waiter.Join(5_000));
        Assert.False(result);
        Assert.True(watch.Elapsed < TimeSpan.FromSeconds(2));
    }

    [MacOSFact]
    public void ACounterOutsideTheApplicationBlockIsRefused()
    {
        using var host = GpuRemoteChannel.Create(SharedMemoryRegion.CreateUniqueName("setest"), new GpuRemoteLayout { ApplicationBytes = 256 });
        Assert.Throws<ArgumentOutOfRangeException>(() => SharedCounter.InApplicationBlock(host, 256));
        using var bare = GpuRemoteChannel.Create(SharedMemoryRegion.CreateUniqueName("setest"));
        Assert.True(bare.Application == null);
        Assert.Throws<ArgumentOutOfRangeException>(() => SharedCounter.InApplicationBlock(bare, 0));
    }

    [MacOSFact]
    public void ServeOneLetsAQuickCallPassASlowOne()
    {
        using var host = GpuRemoteChannel.Create(SharedMemoryRegion.CreateUniqueName("setest"));
        using var gpu = GpuRemoteChannel.Attach(host.Name);
        using var release = new ManualResetEventSlim(false);
        using var stop = new ManualResetEventSlim(false);
        SyncCallHandler handler = (opcode, request, response, out status) =>
        {
            if (opcode == 2)
            {
                release.Wait(10_000);
            }

            status = 0;
            return 0;
        };
        var servers = Enumerable.Range(0, 2).Select(_ => new Thread(() =>
        {
            while (!stop.IsSet)
            {
                if (gpu.SyncCalls.WaitForRequests(Second / 10))
                {
                    while (gpu.SyncCalls.ServeOne(handler))
                    {
                    }
                }
            }
        })).ToList();
        servers.ForEach(thread => thread.Start());
        try
        {
            var slow = new Thread(() => _ = host.SyncCalls.Call(2, [1], new byte[4], out _, out _, 10 * Second));
            slow.Start();
            Thread.Sleep(30);
            var watch = Stopwatch.StartNew();
            Assert.Equal(SyncCallStatus.Completed, host.SyncCalls.Call(1, [1], new byte[4], out _, out _, 5 * Second));
            Assert.True(watch.Elapsed < TimeSpan.FromSeconds(2), $"the quick call waited {watch.Elapsed}");
            release.Set();
            Assert.True(slow.Join(5_000));
        }
        finally
        {
            release.Set();
            stop.Set();
            gpu.SyncCalls.WakeAll();
            servers.ForEach(thread => thread.Join());
        }
    }

    [MacOSFact]
    public void TheClientHandsUnknownRecordsAndCallsToItsService()
    {
        using var host = GpuRemoteChannel.Create(SharedMemoryRegion.CreateUniqueName("setest"));
        var hello = GpuPeerInfo.DescribeThisProcess();
        Assert.True(host.Outbound.TryWrite((uint)GpuRemoteMessage.Hello, in hello));
        using var client = GpuProcessClient.Attach(host.Name, parentProcessId: 0, TimeSpan.FromSeconds(5));
        var service = new RecordingService();
        client.Service = service;
        var exitCode = -1;
        var loop = new Thread(() => exitCode = client.Run());
        loop.Start();

        var message = new ShaderHeaderMessage { CodeAddress = 0x1000, HeaderAddress = 0x2000 };
        Assert.True(host.Outbound.TryWrite((uint)GpuRemoteMessage.ShaderHeader, in message));
        var request = new QueryRequest { Which = 9 };
        var response = new byte[64];
        Assert.Equal(
            SyncCallStatus.Completed,
            host.SyncCalls.Call((uint)GpuRemoteSyncOpcode.Query, GpuSplitWire.Bytes(in request), response, out var length, out var status, 5 * Second));
        Assert.Equal(GpuSplitStatus.Ok, status);
        Assert.Equal(sizeof(QueryResponse), length);
        Assert.Equal(9, GpuSplitWire.Read<QueryResponse>(response).A);

        Assert.True(host.Outbound.TryWrite((uint)GpuRemoteMessage.Shutdown, ReadOnlySpan<byte>.Empty));
        Assert.True(loop.Join(5_000));
        Assert.Equal(GpuProcessClient.ExitClean, exitCode);
        Assert.Equal(new[] { (uint)GpuRemoteMessage.ShaderHeader }, service.Records);
        Assert.Equal(0x2000ul, service.LastHeader);
        Assert.True(service.Stopped);
    }

    [Fact]
    public void SplitPayloadsKeepTheirSizesOnEveryArchitecture()
    {
        // Explicit fields only: a size change here is a protocol change (bump the version).
        Assert.Equal(48, sizeof(VideoOptionsWire));
        Assert.Equal(72, sizeof(SplitStartMessage));
        Assert.Equal(32, sizeof(SubmitCommandStreamMessage));
        Assert.Equal(40, sizeof(FlipRequestMessage));
        Assert.Equal(40, sizeof(ProtectRequestMessage));
        Assert.Equal(40, sizeof(GamepadOutputMessage));
        Assert.Equal(48, sizeof(DisplayBufferResponse));
        Assert.Equal(64, sizeof(ReserveFlipResponse));
        Assert.Equal(32, sizeof(SpanRequest));
        Assert.True(sizeof(InputMessage) < 512);
    }

    [Fact]
    public void TextPayloadsRoundTripAndRefuseAShortRecord()
    {
        var notice = new VideoOutNoticeMessage { Kind = (uint)GpuSplitNotice.PresentationWindowClosed };
        var payload = GpuSplitWire.WithText(in notice, "Grand Theft Auto V", 8, out var length);
        Assert.Equal(8, length);
        Assert.Equal("Grand Th", GpuSplitWire.TextAfter<VideoOutNoticeMessage>(payload, length));
        Assert.Throws<InvalidDataException>(() => GpuSplitWire.TextAfter<VideoOutNoticeMessage>(payload, length + 1));
    }

    private sealed class RecordingService : IGpuHostService
    {
        public List<uint> Records { get; } = new();

        public ulong LastHeader { get; private set; }

        public bool Stopped { get; private set; }

        public void HandleRecord(uint type, ReadOnlySpan<byte> payload)
        {
            Records.Add(type);
            LastHeader = GpuSplitWire.Read<ShaderHeaderMessage>(payload).HeaderAddress;
        }

        public int HandleCall(uint opcode, ReadOnlySpan<byte> request, Span<byte> response, out int status)
        {
            var query = GpuSplitWire.Read<QueryRequest>(request);
            status = GpuSplitStatus.Ok;
            return GpuSplitWire.Write(response, new QueryResponse { A = query.Which });
        }

        public void Stop() => Stopped = true;
    }
}
