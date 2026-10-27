// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

public sealed class QueuedQwordWriteSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private static void Write(ScenePm4 stream, ulong address, params uint[] words) =>
        stream.Packet(ScenePm4.OpWriteData, [1u << 8, (uint)address, (uint)(address >> 32), .. words]);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ProducerQwordRewriteIsVisibleToTheFollowingIndirectPacket(bool queued)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.QueueCommandProcessorWrites = queued;
        var destination = rig.Memory.Allocate(16);
        var childStream = new ScenePm4();
        Write(childStream, destination, 0xBAD, 0xBAD);
        var child = rig.Memory.Upload<uint>(childStream.ToArray());
        using var producer = new ProducerHarness(rig);
        using var returned = new ManualResetEventSlim();
        producer.Enqueue(() =>
        {
            Assert.True(producer.Thread.IsProducerThread);
            rig.Host.RecordCommandProcessorQwords(child + 16, 1, ulong.MaxValue, 0x123400C0FFEEUL, "test-indirect");
            returned.Set();
        });
        var parent = new ScenePm4();
        parent.Packet(0x3F, (uint)child, (uint)(child >> 32), (uint)childStream.Count);
        rig.Enqueue(parent.ToArray());
        producer.Start();
        if (queued)
        {
            // No worker has run: returning here proves the actual producer queued the run.
            Assert.True(returned.Wait(TimeSpan.FromSeconds(5)));
            var stale = new byte[4];
            Assert.True(rig.Memory.TryRead(child + 16, stale));
            Assert.Equal(0xBADu, BitConverter.ToUInt32(stale));
        }
        else
        {
            Assert.True(SpinWait.SpinUntil(() => producer.Thread.TakeCounters().SyncRuns != 0, TimeSpan.FromSeconds(5)));
            Assert.False(returned.IsSet); // Disabled mode waits for the worker as before.
        }
        producer.Drain();
        Assert.True(returned.IsSet);
        Assert.Equal(new uint[] { 0xC0FFEE, 0x1234 }, rig.Read<uint>(destination, 2));
        Assert.Empty(rig.Fatals);
    }

    [Theory]
    [InlineData(false, 16UL)]
    [InlineData(true, 16UL)]
    [InlineData(false, 0UL)]
    [InlineData(true, 0UL)]
    public void ByteAndQwordWritesKeepPredicationOrderAndSparseGaps(bool queued, ulong stride)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.QueueCommandProcessorWrites = queued;
        var query = rig.Memory.Upload<ulong>(new ulong[] { 0, 0xAAAAAAAAAAAAAAAA, 0, 0xBBBBBBBBBBBBBBBB });
        var destination = rig.Memory.Allocate(16);
        using var producer = new ProducerHarness(rig);
        producer.Enqueue(() =>
        {
            rig.Host.RecordCommandProcessorWrite(query, BitConverter.GetBytes(0UL), "test-before");
            rig.Host.RecordCommandProcessorQwords(query, 2, stride, 1, "test-query");
            rig.Host.RecordCommandProcessorWrite(query + 4, BitConverter.GetBytes(2u), "test-after");
        });
        var stream = new ScenePm4();
        stream.SetPredication(3, true, false, false, query);
        stream.Packet(ScenePm4.OpWriteData, true, 1u << 8, (uint)destination, (uint)(destination >> 32), 0x55);
        stream.SetPredication(0, false, false, false, 0);
        rig.Enqueue(stream.ToArray());
        producer.Start();
        producer.Drain();
        Assert.Equal(0x55u, rig.Read<uint>(destination, 1)[0]);
        Assert.Equal(new ulong[] { 0x200000001, 0xAAAAAAAAAAAAAAAA, stride == 0 ? 0UL : 1UL, 0xBBBBBBBBBBBBBBBB },
            rig.Read<ulong>(query, 4));
        Assert.Empty(rig.Fatals);
    }

    [Fact]
    public void AReadInsideASparseGapWaitsForTheConservativePendingHull()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.QueueCommandProcessorWrites = true;
        var address = rig.Memory.Upload<ulong>(new ulong[] { 0, 0x1234, 0 });
        using var producer = new ProducerHarness(rig);
        using var queued = new ManualResetEventSlim();
        using var readReturned = new ManualResetEventSlim();
        ulong gap = 0;
        producer.Enqueue(() =>
        {
            rig.Host.RecordCommandProcessorQwords(address, 2, 16, 7, "test-sparse");
            queued.Set();
            var bytes = new byte[8];
            Assert.True(rig.Host.TryReadGuest(address + 8, bytes));
            gap = BitConverter.ToUInt64(bytes);
            readReturned.Set();
        });
        producer.Start();
        Assert.True(queued.Wait(TimeSpan.FromSeconds(5)));
        // Pending-range synchronization is installed before the producer advances.
        Assert.True(SpinWait.SpinUntil(() => producer.Thread.TakeCounters().SyncRuns != 0, TimeSpan.FromSeconds(5)));
        Assert.False(readReturned.IsSet);
        producer.Drain();
        Assert.Equal(0x1234UL, gap);
        Assert.Equal(new ulong[] { 7, 0x1234, 7 }, rig.Read<ulong>(address, 3));
        Assert.Empty(rig.Fatals);
    }

    [Fact]
    public void WrappingRunsRetainSynchronousStoreByStoreBehavior()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.QueueCommandProcessorWrites = true;
        var address = rig.Memory.Allocate(32) + 8;
        using var producer = new ProducerHarness(rig);
        producer.Enqueue(() => rig.Host.RecordCommandProcessorQwords(address, 2, ulong.MaxValue,
            0x0807060504030201, "test-wrap"));
        producer.Start();
        Assert.True(SpinWait.SpinUntil(() => producer.Thread.TakeCounters().SyncRuns != 0, TimeSpan.FromSeconds(5)));
        producer.Drain();
        Assert.Equal(new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 8 }, rig.Read(address - 1, 9));
        Assert.Empty(rig.Fatals);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void NonpositiveRunsDoNotQueueOrSynchronize(int count)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.QueueCommandProcessorWrites = true;
        using var producer = new ProducerHarness(rig);
        using var returned = new ManualResetEventSlim();
        producer.Enqueue(() =>
        {
            rig.Host.RecordCommandProcessorQwords(ulong.MaxValue, count, ulong.MaxValue, 1, "test-empty");
            returned.Set();
        });
        producer.Start();
        Assert.True(returned.Wait(TimeSpan.FromSeconds(5)));
        Assert.Equal(0L, producer.Thread.TakeCounters().Items);
        producer.Drain();
        Assert.Empty(rig.Fatals);
    }

    // Install a real producer on the host without changing process-wide environment flags.
    // These scenes issue no draws, so no translation sink or prefetch worker is required.
    private sealed class ProducerHarness : IDisposable
    {
        private static readonly FieldInfo CommandThread = typeof(MetalRenderHost).GetField("_commandThread",
            BindingFlags.Instance | BindingFlags.NonPublic)!;
        private readonly SceneRig _rig;
        private bool _started;
        internal DeferredCommandThread Thread { get; }

        internal ProducerHarness(SceneRig rig)
        {
            _rig = rig;
            Assert.Null(CommandThread.GetValue(rig.Host));
            Thread = new DeferredCommandThread(rig.Host.CommandStream, rig.Memory,
                _ => throw new InvalidOperationException("This test must not enqueue draws."), null, static () => { });
            CommandThread.SetValue(rig.Host, Thread);
        }

        internal void Enqueue(Action action) =>
            Assert.True(_rig.Host.CommandStream.TryEnqueueControlBarrier(action, static () => { }));

        internal void Start() { Thread.Start(); _started = true; }

        internal void Drain()
        {
            using var consumed = new ManualResetEventSlim();
            Enqueue(consumed.Set);
            var deadline = DateTime.UtcNow.AddSeconds(30);
            while (!consumed.IsSet || Thread.HasWork)
            {
                Thread.ThrowIfFailed();
                if (DateTime.UtcNow > deadline) throw new TimeoutException("The queued qword scene did not drain.");
                Thread.RunBatch(long.MaxValue, 64, static () => { });
                System.Threading.Thread.Yield();
            }
            _rig.Host.CommitCoalescedFlush();
            _rig.Host.FlushAndWaitForGpu();
            _rig.Host.PollGpuCompletions();
        }

        public void Dispose()
        {
            try
            {
                if (_started) Thread.Stop(() => Thread.RunBatch(long.MaxValue, 64, static () => { }));
            }
            finally { CommandThread.SetValue(_rig.Host, null); }
        }
    }
}
