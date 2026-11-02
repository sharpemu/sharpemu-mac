// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// Demon's Souls' masked table copy (dst[i] = src[i &amp; mask]) followed by a dispatch that reads the
/// table it writes, under SHARPEMU_METAL_V2_HOST_KERNEL_COPY=identical. A table that already holds the
/// copy's result is not copied at all: nothing records a GPU write over it, and the reader binds without
/// the host waiting, while the GPU is held on an event only the test signals. A table that differs is
/// copied on the GPU as before, and the reader sees the copied words.
/// </summary>
public sealed class HostKernelCopySceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private static readonly nint SelNewSharedEvent = MetalNative.Selector("newSharedEvent");
    private static readonly nint SelSetSignaledValue = MetalNative.Selector("setSignaledValue:");
    private static readonly nint SelEncodeWaitForEvent = MetalNative.Selector("encodeWaitForEvent:value:");
    private static readonly nint SelRelease = MetalNative.Selector("release");

    private const int WatchdogMilliseconds = 30_000;
    private const uint Words = 16;
    private const uint Mask = 3;
    private const uint BufferFormat = 20;
    private static readonly uint[] Pattern = [0xC0, 0xC1, 0xC2, 0xC3];

    // Demon's Souls 0xDC18751CB6791EF9: if (gid < cb[0]) dst[gid] = src[gid & cb[1]], with src in
    // s[0:3], dst in s[4:7], the constants in s[8:11] and the group id in s12.
    private static readonly uint[] MaskedCopy =
    [
        0xBEFC03FF, 0x94E6DD39, 0xBF960000, 0xBFA00003, 0xD7460000, 0x04010C0C, 0xF4201A84, 0xFA000000,
        0xBF8CC07F, 0x7DA8006A, 0xBF880009, 0xF4201A84, 0xFA000004, 0xBF8CC07F, 0x3602006A, 0xE0002000,
        0x80000101, 0xBF8C3F70, 0xE0102000, 0x80010100, 0xBF810000,
    ];

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ATableCopyWhoseResultIsAlreadyThereIsNotRunAndItsReaderDoesNotWait(bool identical)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var host = rig.Host;
        host.HostKernelCopyMode = MetalHostKernelCopyMode.Identical;
        var source = rig.Memory.Upload<uint>(Pattern);
        var table = rig.Memory.Allocate(Words * 4, 256);
        var constants = rig.Memory.Upload<uint>([Words, Mask]);
        var stored = rig.Memory.Allocate(16, 256);
        var copy = rig.Dispatch(new ScenePm4(), MaskedCopy,
            [.. Descriptors.Buffer(source, 4, (uint)Pattern.Length, BufferFormat), .. Descriptors.Buffer(table, 4, Words, BufferFormat),
                .. Descriptors.Buffer(constants, 4, 2, BufferFormat)], 1);
        // The reader takes record 1 of the table (words 4..7) through the pointer in s[0:1].
        var stream = rig.Dispatch(copy, ComputeScenePrograms.ScalarLoadImmediateOffset,
            [Descriptors.Lo(table), Descriptors.Hi(table), 0, 0, .. Descriptors.Buffer(stored, 4, 4, BufferFormat)], 1).ToArray();

        // Once through first, so neither program parks on its pipeline below.
        rig.Submit(stream);
        rig.Memory.Write<uint>(table, identical ? Expected() : new uint[Words]);
        rig.Memory.Write<uint>(stored, [~0u, ~0u, ~0u, ~0u]);
        var elidedBefore = host.KernelCopiesElided;

        var gate = MetalNative.Send(DeviceOf(host), SelNewSharedEvent);
        Assert.NotEqual(0, gate);
        var released = 0;
        using var watchdog = Watchdog(gate, () => Interlocked.Exchange(ref released, 1));
        try
        {
            MetalV2Native.SendEventValue(host.CommandBuffer, SelEncodeWaitForEvent, gate, 1);
            var waitsBefore = host.HostGpuWaits;
            rig.Enqueue(stream);
            rig.RecordPending();
            Assert.Equal(identical ? 1 : 0, host.KernelCopiesElided - elidedBefore);
            Assert.Equal(!identical, ((IMetalImageCacheOwner)host).HasPendingGpuWrite(table, Words * 4));
            if (identical)
            {
                Assert.Equal(waitsBefore, host.HostGpuWaits);
                Assert.Equal(0, Volatile.Read(ref released));
            }
        }
        finally
        {
            MetalNative.SendVoid(gate, SelSetSignaledValue, (nint)1);
        }

        rig.Drain();
        Assert.Equal(Expected(), rig.Read<uint>(table, (int)Words));
        Assert.Equal(Pattern, rig.Read<uint>(stored, 4));
        watchdog.Dispose();
        MetalNative.SendVoid(gate, SelRelease);
        Assert.Empty(rig.Fatals);
    }

    private static uint[] Expected() => Enumerable.Range(0, (int)Words).Select(index => Pattern[index & (int)Mask]).ToArray();

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

    private static nint DeviceOf(MetalRenderHost host) => ((IMetalImageCacheOwner)host).Device;
}
