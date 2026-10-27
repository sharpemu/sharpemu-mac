// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// Host reads of GPU-written guest bytes that no longer wait for more than they need: the first
/// read of a word waits for its writer's fence (eager write fences), a read ahead of the stream
/// gives up instead of waiting, and a packet-sized copy whose source the GPU still writes runs on
/// the GPU. Each holds the GPU on an event only the test signals, so a wait that included the held
/// work could not return before the test releases it.
/// </summary>
public sealed class GpuWordNoWaitSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private static readonly nint SelNewSharedEvent = MetalNative.Selector("newSharedEvent");
    private static readonly nint SelSetSignaledValue = MetalNative.Selector("setSignaledValue:");
    private static readonly nint SelEncodeWaitForEvent = MetalNative.Selector("encodeWaitForEvent:value:");
    private static readonly nint SelRelease = MetalNative.Selector("release");

    private const int WatchdogMilliseconds = 30_000;
    private const uint Words = 64;
    private const uint LaterGroups = 16;

    [Fact]
    public void TheFirstReadOfAWordAnEarlySubmissionWroteWaitsOnlyForItsWriter()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var host = rig.Host;
        Assert.True(host.WriteFences && host.EagerWriteFences, "eager write fences are on by default");
        var (word, later, writer, consumer) = WriterAndLater(rig);
        var plantedBefore = host.WriteFenceCounters.Planted;
        var waitsBefore = host.WriteFenceCounters;

        // No read has waited on these bytes before: the fence comes from the write alone.
        var gate = MetalNative.Send(DeviceOf(host), SelNewSharedEvent);
        Assert.NotEqual(0, gate);
        var released = 0;
        using var watchdog = Watchdog(gate, () => Interlocked.Exchange(ref released, 1));
        try
        {
            host.WriteFencePlantedHook = buffer => MetalV2Native.SendEventValue(buffer, SelEncodeWaitForEvent, gate, 1);
            RunBoth(rig, writer, consumer);
            host.WriteFencePlantedHook = null;
            Assert.True(host.WriteFenceCounters.Planted > plantedBefore, "a fence was planted after the writer's submission");

            Assert.Equal(5u ^ 0x5A5A, ReadWord(host, word + 5 * 4));
            Assert.Equal(0, Volatile.Read(ref released));
            // The later submission is still held: its bytes are the zeros the rig allocated.
            Assert.Equal(0u, BitConverter.ToUInt32(rig.Memory.ReadRaw(later + 4 * 9, 4)));
            var counters = host.WriteFenceCounters;
            Assert.True(counters.Waits + counters.Reads > waitsBefore.Waits + waitsBefore.Reads, $"fence counters {counters}");
        }
        finally
        {
            host.WriteFencePlantedHook = null;
            MetalNative.SendVoid(gate, SelSetSignaledValue, (nint)1);
        }

        rig.Drain();
        AssertStored(rig, later, Words * LaterGroups);
        watchdog.Dispose();
        MetalNative.SendVoid(gate, SelRelease);
        Assert.Empty(rig.Fatals);
    }

    [Fact]
    public void WithEagerFencesOffTheFirstReadWaitsForTheWholeBuffer()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var host = rig.Host;
        host.EagerWriteFences = false;
        var (word, later, writer, consumer) = WriterAndLater(rig);
        var plantedBefore = host.WriteFenceCounters.Planted;

        // The whole buffer is held from its start and released shortly after the read begins:
        // a read that waits for the whole buffer returns with the later submission's bytes written.
        var gate = MetalNative.Send(DeviceOf(host), SelNewSharedEvent);
        var released = 0;
        try
        {
            MetalV2Native.SendEventValue(host.CommandBuffer, SelEncodeWaitForEvent, gate, 1);
            RunBoth(rig, writer, consumer);
            Assert.Equal(plantedBefore, host.WriteFenceCounters.Planted);
            var signaller = new Thread(() =>
            {
                Thread.Sleep(100);
                Interlocked.Exchange(ref released, 1);
                MetalNative.SendVoid(gate, SelSetSignaledValue, (nint)1);
            }) { IsBackground = true };
            signaller.Start();
            Assert.Equal(5u ^ 0x5A5A, ReadWord(host, word + 5 * 4));
            Assert.Equal(1, Volatile.Read(ref released));
            Assert.Equal(9u ^ 0x5A5A, BitConverter.ToUInt32(rig.Memory.ReadRaw(later + 4 * 9, 4)));
            signaller.Join();
        }
        finally
        {
            MetalNative.SendVoid(gate, SelSetSignaledValue, (nint)1);
        }

        rig.Drain();
        MetalNative.SendVoid(gate, SelRelease);
        Assert.Empty(rig.Fatals);
    }

    [Fact]
    public void AReadAheadOfTheStreamGivesUpInsteadOfWaitingForTheGpu()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var host = rig.Host;
        var word = rig.Memory.Allocate(Words * 4, 256);
        var writer = rig.Dispatch(new ScenePm4(), ScenePrograms.StoreGlobalId, Descriptors.Buffer(word, 4, Words, format: 20), 1).ToArray();
        rig.Submit(writer);
        rig.Memory.Write(word, new byte[Words * 4]);

        var gate = MetalNative.Send(DeviceOf(host), SelNewSharedEvent);
        var released = 0;
        using var watchdog = Watchdog(gate, () => Interlocked.Exchange(ref released, 1));
        try
        {
            MetalV2Native.SendEventValue(host.CommandBuffer, SelEncodeWaitForEvent, gate, 1);
            rig.Enqueue(writer);
            rig.RecordPending();
            var waitsBefore = host.HostGpuWaits;

            Assert.Throws<GpuFatal.AbandonedWorkException>(() =>
                GpuFatal.Speculatively(() => ReadWord(host, word + 3 * 4)));
            Assert.Equal(waitsBefore, host.HostGpuWaits);
            Assert.Equal(0, Volatile.Read(ref released));
        }
        finally
        {
            MetalNative.SendVoid(gate, SelSetSignaledValue, (nint)1);
        }

        // The stream's own read waits as before and sees the written word.
        Assert.Equal(3u ^ 0x5A5A, ReadWord(host, word + 3 * 4));
        rig.Drain();
        watchdog.Dispose();
        MetalNative.SendVoid(gate, SelRelease);
        Assert.Empty(rig.Fatals);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void APacketSizedCopyOfAWordTheGpuStillWritesIsCopiedOnTheGpu(bool queuedBehindHeldWork)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var host = rig.Host;
        // Command-processor writes land as parsed, so a small DMA copy may take the host path.
        host.CommandProcessorWritesAtParse = true;
        var produced = rig.Memory.Allocate(Words * 4, 256);
        var copied = rig.Memory.Allocate(16, 256);
        var dispatch = rig.Dispatch(new ScenePm4(), ScenePrograms.StoreGlobalId, Descriptors.Buffer(produced, 4, Words, format: 20), 1);
        rig.Submit(dispatch.ToArray());
        rig.Memory.Write(produced, new byte[Words * 4]);

        // DMA_DATA, memory to memory, 8 bytes from produced + 6 * 4 to copied.
        var source = produced + 6 * 4;
        var stream = rig.Dispatch(new ScenePm4(), ScenePrograms.StoreGlobalId, Descriptors.Buffer(produced, 4, Words, format: 20), 1)
            .Packet(0x50, 0u, (uint)source, (uint)(source >> 32), (uint)copied, (uint)(copied >> 32), 8u)
            .ToArray();
        var gate = MetalNative.Send(DeviceOf(host), SelNewSharedEvent);
        var released = 0;
        using var watchdog = Watchdog(gate, () => Interlocked.Exchange(ref released, 1));
        try
        {
            if (queuedBehindHeldWork)
            {
                MetalV2Native.SendEventValue(host.CommandBuffer, SelEncodeWaitForEvent, gate, 1);
            }

            var waitsBefore = host.HostGpuWaits;
            rig.Enqueue(stream);
            rig.RecordPending();
            Assert.False(host.CommandStream.HasUnblockedPending);
            // Parsing the copy waited for nothing.
            Assert.Equal(waitsBefore, host.HostGpuWaits);
            Assert.Equal(0, Volatile.Read(ref released));
        }
        finally
        {
            MetalNative.SendVoid(gate, SelSetSignaledValue, (nint)1);
        }

        rig.Drain();
        var words = rig.Read<uint>(copied, 2);
        Assert.Equal(6u ^ 0x5A5A, words[0]);
        Assert.Equal(7u ^ 0x5A5A, words[1]);
        watchdog.Dispose();
        MetalNative.SendVoid(gate, SelRelease);
        Assert.Empty(rig.Fatals);
    }

    // A word buffer one group writes and, apart from it, a buffer a later, larger dispatch writes;
    // both programs are compiled first so neither submission parks on a pipeline.
    private static (ulong Word, ulong Later, uint[] Writer, uint[] Consumer) WriterAndLater(SceneRig rig)
    {
        var word = rig.Memory.Allocate(Words * 4, 256);
        var later = rig.Memory.Allocate(Words * 4 * LaterGroups, 0x10000);
        var writer = rig.Dispatch(new ScenePm4(), ScenePrograms.StoreGlobalId, Descriptors.Buffer(word, 4, Words, format: 20), 1).ToArray();
        var consumer = rig.Dispatch(new ScenePm4(), ScenePrograms.StoreGlobalId,
            Descriptors.Buffer(later, 4, Words * LaterGroups, format: 20), LaterGroups).ToArray();
        rig.Submit(writer);
        rig.Submit(consumer);
        rig.Memory.Write(word, new byte[Words * 4]);
        rig.Memory.Write(later, new byte[Words * 4 * LaterGroups]);
        return (word, later, writer, consumer);
    }

    private static void AssertStored(SceneRig rig, ulong address, uint count)
    {
        var words = rig.Read<uint>(address, (int)count);
        for (var index = 0u; index < words.Length; index++)
        {
            Assert.Equal(index ^ 0x5A5A, words[index]);
        }
    }

    // Both submissions run in one worker pass, so the coalescer keeps one command buffer open
    // across them and commits it when the stream runs dry.
    private static void RunBoth(SceneRig rig, uint[] first, uint[] second)
    {
        rig.Enqueue(first);
        rig.Enqueue(second);
        rig.RecordPending();
        Assert.False(rig.Host.CommandStream.HasPending);
    }

    private static IDisposable Watchdog(nint gate, Action fired)
    {
        var cancel = new CancellationTokenSource();
        var thread = new Thread(() =>
        {
            if (!cancel.Token.WaitHandle.WaitOne(WatchdogMilliseconds))
            {
                fired();
                MetalNative.SendVoid(gate, SelSetSignaledValue, (nint)1);
            }
        }) { IsBackground = true };
        thread.Start();
        return new WatchdogStop(cancel, thread);
    }

    private sealed class WatchdogStop(CancellationTokenSource cancel, Thread thread) : IDisposable
    {
        private int _stopped;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _stopped, 1) != 0)
            {
                return;
            }

            cancel.Cancel();
            thread.Join();
            cancel.Dispose();
        }
    }

    private static uint ReadWord(MetalRenderHost host, ulong address)
    {
        Span<byte> bytes = stackalloc byte[4];
        Assert.True(host.TryReadGuest(address, bytes));
        return BitConverter.ToUInt32(bytes);
    }

    private static nint DeviceOf(MetalRenderHost host) => ((IMetalImageCacheOwner)host).Device;
}
