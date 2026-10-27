// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// A host read of a word the GPU writes waits for the work that writes it and nothing recorded
/// after it, and a WAIT_REG_MEM on a label the GPU has yet to write resolves without a host
/// wait. Both hold the GPU on an event only the test signals, so a wait that included the
/// held work could not return: nothing here depends on how fast the GPU is.
/// </summary>
public sealed class GpuWordWaitSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private static readonly nint SelNewSharedEvent = MetalNative.Selector("newSharedEvent");
    private static readonly nint SelSetSignaledValue = MetalNative.Selector("setSignaledValue:");
    private static readonly nint SelEncodeWaitForEvent = MetalNative.Selector("encodeWaitForEvent:value:");
    private static readonly nint SelRelease = MetalNative.Selector("release");

    // A held GPU is released by this deadline whatever happens, so a wrong wait fails the test
    // instead of hanging the run.
    private const int WatchdogMilliseconds = 30_000;

    private const uint Words = 64;
    private const uint LaterGroups = 16;

    [Fact]
    public void AWordAnEarlySubmissionWroteIsReadWhileALaterSubmissionInTheSameBufferIsHeld()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var host = rig.Host;
        Assert.True(host.WriteFences, "write fences are on by default");
        // The learned fences: an eager one would answer the first read too (GpuWordNoWaitSceneTests).
        host.EagerWriteFences = false;
        var word = rig.Memory.Allocate(Words * 4, 256);
        // Apart, so the two submissions' write ranges stay two records rather than merging into one.
        var later = rig.Memory.Allocate(Words * 4 * LaterGroups, 0x10000);
        var writer = rig.Dispatch(new ScenePm4(), ScenePrograms.StoreGlobalId, Descriptors.Buffer(word, 4, Words, format: 20), 1).ToArray();
        var consumer = rig.Dispatch(new ScenePm4(), ScenePrograms.StoreGlobalId,
            Descriptors.Buffer(later, 4, Words * LaterGroups, format: 20), LaterGroups).ToArray();

        // Both programs are compiled first, so neither submission parks on a pipeline below.
        rig.Submit(writer);
        rig.Submit(consumer);

        // The first read of the word waits for the whole buffer, and teaches the host that the
        // CPU reads it. The GPU is held until shortly after the read starts, so the read does wait.
        var learnGate = MetalNative.Send(DeviceOf(host), SelNewSharedEvent);
        try
        {
            MetalV2Native.SendEventValue(host.CommandBuffer, SelEncodeWaitForEvent, learnGate, 1);
            RunBoth(rig, writer, consumer);
            var signaller = new Thread(() =>
            {
                Thread.Sleep(100);
                MetalNative.SendVoid(learnGate, SelSetSignaledValue, (nint)1);
            }) { IsBackground = true };
            signaller.Start();
            Assert.Equal(1u ^ 0x5A5A, ReadWord(host, word + 4));
        }
        finally
        {
            MetalNative.SendVoid(learnGate, SelSetSignaledValue, (nint)1);
        }

        rig.Drain();
        MetalNative.SendVoid(learnGate, SelRelease);
        rig.Memory.Write(word, new byte[Words * 4]);
        rig.Memory.Write(later, new byte[Words * 4 * LaterGroups]);
        var plantedBefore = host.WriteFenceCounters.Planted;

        // Now the later submission sits behind an event the GPU waits on, after the fence the
        // host plants between the two submissions.
        var gate = MetalNative.Send(DeviceOf(host), SelNewSharedEvent);
        Assert.NotEqual(0, gate);
        var released = 0;
        using var watchdog = Watchdog(gate, () => Interlocked.Exchange(ref released, 1));
        try
        {
            host.WriteFencePlantedHook = buffer => MetalV2Native.SendEventValue(buffer, SelEncodeWaitForEvent, gate, 1);
            RunBoth(rig, writer, consumer);
            host.WriteFencePlantedHook = null;
            Assert.Equal(plantedBefore + 1, host.WriteFenceCounters.Planted);

            var value = ReadWord(host, word + 4);
            Assert.Equal(0, Volatile.Read(ref released));
            Assert.Equal(1u ^ 0x5A5A, value);
            // The held submission has not run: its bytes are still the zeros written above.
            Assert.Equal(0u, BitConverter.ToUInt32(rig.Memory.ReadRaw(later + 4 * 5, 4)));
            // Answered through the fence: waited for, or already signalled when the read came.
            var counters = host.WriteFenceCounters;
            Assert.True(counters.Waits + counters.Reads >= 1, $"fence counters {counters}");
        }
        finally
        {
            host.WriteFencePlantedHook = null;
            MetalNative.SendVoid(gate, SelSetSignaledValue, (nint)1);
        }

        rig.Drain();
        var words = rig.Read<uint>(later, (int)(Words * LaterGroups));
        for (var index = 0u; index < words.Length; index++)
        {
            Assert.Equal(index ^ 0x5A5A, words[index]);
        }

        watchdog.Dispose();
        MetalNative.SendVoid(gate, SelRelease);
        Assert.Empty(rig.Fatals);
    }

    [Fact]
    public void WithWriteFencesOffNoFenceIsPlantedAndTheWordStillReadsWritten()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var host = rig.Host;
        host.WriteFences = false;
        var word = rig.Memory.Allocate(Words * 4, 256);
        var later = rig.Memory.Allocate(Words * 4 * LaterGroups, 256);
        var writer = rig.Dispatch(new ScenePm4(), ScenePrograms.StoreGlobalId, Descriptors.Buffer(word, 4, Words, format: 20), 1).ToArray();
        var consumer = rig.Dispatch(new ScenePm4(), ScenePrograms.StoreGlobalId,
            Descriptors.Buffer(later, 4, Words * LaterGroups, format: 20), LaterGroups).ToArray();
        rig.Submit(writer);
        rig.Submit(consumer);
        for (var round = 0; round < 2; round++)
        {
            rig.Memory.Write(word, new byte[Words * 4]);
            RunBoth(rig, writer, consumer);
            Assert.Equal(7u ^ 0x5A5A, ReadWord(host, word + 7 * 4));
            rig.Drain();
        }

        Assert.Equal(0, host.WriteFenceCounters.Planted);
        Assert.Empty(rig.Fatals);
    }

    [Fact]
    public void AWaitRegMemOnALabelAnotherQueueHasYetToWriteResolvesWithoutAHostWait()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var host = rig.Host;
        var produced = rig.Memory.Allocate(Words * 4, 256);
        var consumed = rig.Memory.Allocate(Words * 4, 256);
        var label = rig.Memory.Allocate(8, 8);
        const uint LabelValue = 7;

        // Graphics queue: a dispatch, then RELEASE_MEM writes the label at end of pipe (64-bit data).
        var producer = rig.Dispatch(new ScenePm4(), ScenePrograms.StoreGlobalId, Descriptors.Buffer(produced, 4, Words, format: 20), 1)
            .Packet(0x49, 0x2Fu | (5u << 8), 2u << 29, (uint)label, (uint)(label >> 32), LabelValue, 0, 0)
            .ToArray();
        // Compute queue: WAIT_REG_MEM (memory, equal) on the label, then a dispatch of its own.
        var waiter = rig.Dispatch(
                new ScenePm4().Packet(0x3C, 0x10u | 3u, (uint)label, (uint)(label >> 32), LabelValue, 0xFFFF_FFFFu, 0x10),
                ScenePrograms.StoreGlobalId, Descriptors.Buffer(consumed, 4, Words, format: 20), 1)
            .ToArray();

        var gate = MetalNative.Send(DeviceOf(host), SelNewSharedEvent);
        Assert.NotEqual(0, gate);
        var released = 0;
        using var watchdog = Watchdog(gate, () => Interlocked.Exchange(ref released, 1));
        try
        {
            // The GPU holds before the producer, so the label cannot land until the test says so.
            MetalV2Native.SendEventValue(host.CommandBuffer, SelEncodeWaitForEvent, gate, 1);
            var waitsBefore = host.HostGpuWaits;
            var suspendsBefore = SharpEmu.Libs.Gpu.GpuCommands.GpuCommandInterpreter.OperandSuspends;
            rig.Enqueue(producer);
            host.RunPendingCommands();
            host.RunCommandStreamSlices(64);
            rig.Enqueue(waiter, queue: 0x20);
            host.RunPendingCommands();
            host.RunCommandStreamSlices(64);

            Assert.Equal(0, Volatile.Read(ref released));
            // The wait met the pending label: the compute queue ran to its end without a host wait or a park.
            Assert.False(host.CommandStream.HasUnblockedPending);
            Assert.Equal(waitsBefore, host.HostGpuWaits);
            Assert.Equal(suspendsBefore, SharpEmu.Libs.Gpu.GpuCommands.GpuCommandInterpreter.OperandSuspends);
            // And nothing has landed yet.
            Assert.Equal(0UL, BitConverter.ToUInt64(rig.Memory.ReadRaw(label, 8)));
            Assert.Equal(0u, BitConverter.ToUInt32(rig.Memory.ReadRaw(consumed + 4, 4)));
        }
        finally
        {
            MetalNative.SendVoid(gate, SelSetSignaledValue, (nint)1);
        }

        rig.Drain();
        Assert.Equal((ulong)LabelValue, BitConverter.ToUInt64(rig.Memory.ReadRaw(label, 8)));
        var producedWords = rig.Read<uint>(produced, (int)Words);
        var consumedWords = rig.Read<uint>(consumed, (int)Words);
        for (var index = 0u; index < Words; index++)
        {
            Assert.Equal(index ^ 0x5A5A, producedWords[index]);
            Assert.Equal(index ^ 0x5A5A, consumedWords[index]);
        }

        watchdog.Dispose();
        MetalNative.SendVoid(gate, SelRelease);
        Assert.Empty(rig.Fatals);
    }

    // Both submissions run in one worker pass, so the coalescer keeps one command buffer open
    // across them and commits it when the stream runs dry.
    private static void RunBoth(SceneRig rig, uint[] first, uint[] second)
    {
        rig.Enqueue(first);
        rig.Enqueue(second);
        rig.Host.RunPendingCommands();
        rig.Host.RunCommandStreamSlices(64);
        Assert.False(rig.Host.CommandStream.HasPending);
    }

    // Releases a held GPU after WatchdogMilliseconds unless cancelled by disposal.
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

    // Disposing a token source does not cancel it: the watchdog is stopped and joined first, so it
    // never signals a gate the test has already released.
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
