// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed unsafe partial class MetalEndOfPipeOrderTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CoalescedCpWritesOrderDependentKernelsAndPartialOverwritesInOneEncoder(bool concurrent)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.ConcurrentComputeEnabled = concurrent;
        host.LabelsInEncoder = true;
        var pipeline = SnapshotPipeline();
        try
        {
            var encoder = host.EnsureComputeEncoder();
            var barriers = host.ComputeScopeBarriers;
            // This kernel must finish reading zero before the first landing overwrites it.
            Snapshot(host, pipeline, LaterLabel, GuestBase + 0x10000);
            host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(0x1122334455667788UL), "WRITE_DATA");
            host.RecordCommandProcessorWrite(LaterLabel + 1, new byte[] { 0xAA, 0xBB, 0xCC }, "WRITE_DATA");
            host.OrderAfterDeferredCopies(LaterLabel, 8);
            Assert.Equal(encoder, host.EnsureComputeEncoder());
            Assert.Equal(0, host.DeferredLabelCopyCount);
            Snapshot(host, pipeline, LaterLabel, GuestBase + 0x10008);
            host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(0x99UL), "WRITE_DATA");
            host.OrderAfterDeferredCopies(LaterLabel, 8);
            Snapshot(host, pipeline, LaterLabel, GuestBase + 0x10010);
            Assert.Equal(encoder, host.EnsureComputeEncoder());
            Assert.Equal(3, host.InEncoderLabelWrites);
            Assert.Equal(2, host.LabelLandingBatches);
            Assert.Equal(1, host.LabelLandingsCoalesced);
            // These native snapshot/copy dispatches are outside the guest dependency
            // tracker. Each landing must still encode both sides of its ordering cut.
            Assert.Equal(barriers + 4, host.ComputeScopeBarriers);
            Assert.Equal(0, host.TakeLabelBarriers());
            // Recording has not published any CP bytes to a guest CPU reader.
            Assert.Equal(0UL, rig.Backing.Read64(LaterLabel));

            // A real EOP label still publishes earlier shader stores before CPU observation.
            host.RecordEndOfPipe(new(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 7));
            var serial = host.FlushCommandBuffer();
            Assert.Equal(1, host.TakeLabelBarriers());
            WaitUntil(() => rig.Backing.Read64(WindowedLabel) == 7);
            Assert.Equal(0UL, rig.Backing.Read64(GuestBase + 0x10000));
            Assert.Equal(0x11223344CCBBAA88UL, rig.Backing.Read64(GuestBase + 0x10008));
            Assert.Equal(0x99UL, rig.Backing.Read64(GuestBase + 0x10010));
            // The guest may recycle these bytes immediately on seeing the EOP label.
            Assert.True(rig.Backing.TryWrite(GuestBase + 0x10008, BitConverter.GetBytes(0xCAFEUL)));
            WaitForGpu(host, serial);
            WaitForCompletions(host);
            Assert.Equal(0xCAFEUL, rig.Backing.Read64(GuestBase + 0x10008));
        }
        finally { MetalNative.SendVoid(pipeline, MetalNative.Selector("release")); }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AMixedEopAndCpLandingRetainsTheCpuVisibilityBoundary(bool eopFirst)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.LabelsInEncoder = true;
        _ = host.EnsureComputeEncoder();
        void Eop() => host.RecordEndOfPipe(new(EndOfPipeWriteKind.Write64, 1, WindowedLabel, 5));
        if (eopFirst) Eop();
        host.RecordCommandProcessorWrite(LaterLabel, BitConverter.GetBytes(9UL), "WRITE_DATA");
        if (!eopFirst) Eop();
        host.OrderAfterDeferredCopies(LaterLabel, 8);
        Assert.False(host.ComputeEncoderOpen);
        Assert.Equal(0, host.InEncoderLabelWrites);
        Assert.Equal(1, host.TakeLabelBarriers());
        Assert.Equal(1, host.LabelLandingsCoalesced);
        Assert.Equal(0UL, rig.Backing.Read64(WindowedLabel));
        var serial = host.FlushCommandBuffer();
        WaitForGpu(host, serial);
        WaitForCompletions(host);
        Assert.Equal(5UL, rig.Backing.Read64(WindowedLabel));
        Assert.Equal(9UL, rig.Backing.Read64(LaterLabel));
    }

    private static nint SnapshotPipeline()
    {
        const string source = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void snapshot(const device ulong* src [[buffer(0)]], device ulong* dst [[buffer(1)]]) {
                dst[0] = src[0];
            }
            """;
        nint error = 0;
        var library = MetalNative.Send(Device.Value, MetalNative.Selector("newLibraryWithSource:options:error:"),
            MetalNative.NsString(source), 0, ref error);
        Assert.True(library != 0, MetalNative.DescribeError(error));
        var function = MetalNative.Send(library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString("snapshot"));
        MetalNative.SendVoid(library, MetalNative.Selector("release"));
        var pipeline = MetalNative.Send(Device.Value, MetalNative.Selector("newComputePipelineStateWithFunction:error:"), function, ref error);
        MetalNative.SendVoid(function, MetalNative.Selector("release"));
        Assert.True(pipeline != 0, MetalNative.DescribeError(error));
        return pipeline;
    }

    private static void Snapshot(MetalRenderHost host, nint pipeline, ulong from, ulong to)
    {
        var encoder = host.EnsureComputeEncoder();
        var src = host.GuestMemory.ResolveWritable(from, 8);
        var dst = host.GuestMemory.ResolveWritable(to, 8);
        MetalNative.SendVoid(encoder, MetalNative.Selector("setComputePipelineState:"), pipeline);
        MetalNative.SendSetBuffer(encoder, MetalNative.Selector("setBuffer:offset:atIndex:"), src.Buffer, (nuint)src.Offset, 0);
        MetalNative.SendSetBuffer(encoder, MetalNative.Selector("setBuffer:offset:atIndex:"), dst.Buffer, (nuint)dst.Offset, 1);
        var one = new MtlSize { Width = 1, Height = 1, Depth = 1 };
        MetalNative.SendDispatch(encoder, MetalNative.Selector("dispatchThreadgroups:threadsPerThreadgroup:"), one, one);
    }
}
