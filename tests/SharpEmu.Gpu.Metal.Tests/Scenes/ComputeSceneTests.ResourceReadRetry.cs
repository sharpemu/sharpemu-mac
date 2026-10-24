// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

public sealed partial class ComputeSceneTests
{
    // The output descriptor is itself loaded from guest memory. Its record count
    // must be final on the CPU before the dispatch can bind it.
    private static readonly uint[] StoreThroughLoadedDescriptor =
        [0xF4080000, 0xFA000000, 0xBF8CC07F, .. ScenePrograms.StoreGlobalId];

    [Theory]
    [InlineData(true, true)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    public void AGpuProducedDescriptorIsReadAfterItsProducerAndOtherQueuesCanRun(bool enabled, bool committed)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var host = rig.Host;
        host.AsyncResourceReads = enabled;
        const uint count = 64;
        var output = Fill(rig, count, uint.MaxValue);
        var descriptor = rig.Memory.Upload<uint>(Vsharp(output, 4, count));
        var userData = new[] { Descriptors.Lo(descriptor), Descriptors.Hi(descriptor), 0u, 0u };
        // Warm this program and pipeline before holding the native producer.
        rig.Submit(rig.Dispatch(new ScenePm4(), StoreThroughLoadedDescriptor, userData, 1));
        Assert.Equal(Enumerable.Range(0, (int)count).Select(i => (uint)i ^ 0x5A5Au), rig.Read<uint>(output, (int)count));
        rig.Memory.Write<uint>(output, Enumerable.Repeat(uint.MaxValue, (int)count).ToArray());
        rig.Memory.Write32(descriptor + 8, 0);

        var device = ((IMetalSubmissionContext)host).Device;
        var gate = MetalNative.Send(device, MetalNative.Selector("newSharedEvent"));
        Assert.NotEqual(0, gate);
        using var release = new ManualResetEventSlim(false);
        // A failing synchronous path cannot leave the GPU waiting on an unopened event.
        var opener = Task.Run(() =>
        {
            release.Wait(TimeSpan.FromSeconds(3));
            using var pool = MetalAutoreleasePool.Enter();
            MetalNative.SendVoid(gate, MetalNative.Selector("setSignaledValue:"), (nint)1);
        });
        try
        {
            MetalV2Native.SendEventValue(host.CommandBuffer, MetalNative.Selector("encodeWaitForEvent:value:"), gate, 1);
            host.FillGuestMemory(descriptor + 8, sizeof(uint), count, allowHost: false);
            var producer = host.SubmissionSerial;
            if (committed) host.FlushCommandBuffer();
            var before = host.ResourceReadRetryCounters;
            var dispatch = rig.Dispatch(new ScenePm4(), StoreThroughLoadedDescriptor, userData, 1)
                .Packet(ScenePm4.OpNumInstances, 7);
            rig.Enqueue(dispatch.ToArray(), 0x20);
            // The graphics queue has no dependency on the descriptor.
            rig.Enqueue(new ScenePm4().Packet(ScenePm4.OpNumInstances, 5).ToArray());
            host.RunCommandStreamSlices(64);

            if (enabled && committed)
            {
                Assert.Equal(0UL, MetalV2Native.SendValue64(gate, MetalNative.Selector("signaledValue")));
                Assert.True(host.CompletedSerial < producer);
                Assert.True(host.ResourceReadRetryCounters.Suspended > before.Suspended);
                Assert.Equal(1, host.CommandStream.BlockedQueueCount);
                Assert.Equal(1u, host.CommandStream.GetInterpreter(1).InstanceCount);
                Assert.Equal(5u, host.CommandStream.GetInterpreter(0).InstanceCount);
                Assert.All(rig.Memory.ReadRaw(output, (int)count * 4), value => Assert.Equal(0xFF, value));
            }
            else
            {
                Assert.Equal(before.Suspended, host.ResourceReadRetryCounters.Suspended);
                Assert.True(enabled
                    ? host.ResourceReadRetryCounters.Uncommitted > before.Uncommitted
                    : host.ResourceReadRetryCounters.Disabled > before.Disabled);
            }

            release.Set();
            opener.GetAwaiter().GetResult();
            rig.Drain();
            Assert.Equal(7u, host.CommandStream.GetInterpreter(1).InstanceCount);
            Assert.Equal(Enumerable.Range(0, (int)count).Select(i => (uint)i ^ 0x5A5Au), rig.Read<uint>(output, (int)count));
            Assert.Empty(rig.Fatals);
        }
        finally
        {
            release.Set();
            opener.GetAwaiter().GetResult();
            host.FlushAndWaitForGpu();
            MetalNative.SendVoid(gate, MetalNative.Selector("release"));
        }
    }
}
