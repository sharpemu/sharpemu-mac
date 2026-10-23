// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

public sealed partial class ComputeSceneTests
{
    // dst[id] = src[id] + 1. The arithmetic prevents the native memcpy/fill
    // recognizers from replacing the dispatches this test must exercise.
    private static readonly uint[] IncrementBuffer =
    [
        0x8F098608,             // s_lshl_b32 s9, s8, 6
        0x4A020009,             // v_add_nc_u32_e32 v1, s9, v0
        0xE0302000, 0x80000201, // buffer_load_dword v2, v1, s[0:3], 0 idxen
        0xBF8C3F70,             // s_waitcnt vmcnt(0)
        0x4A040481,             // v_add_nc_u32_e32 v2, 1, v2
        0xE0702000, 0x80010201, // buffer_store_dword v2, v1, s[4:7], 0 idxen
        0xBF810000,             // s_endpgm
    ];

    [Theory]
    [InlineData("producer-consumer")]
    [InlineData("atomics")]
    [InlineData("indirect")]
    [InlineData("cp-write")]
    public void DependentDispatchesMatchSerialBytesWithoutGuestBarriers(string chain)
    {
        byte[]? Run(bool concurrent)
        {
            using var rig = CreateRig();
            if (rig is null) return null;
            rig.Host.ConcurrentComputeEnabled = concurrent;
            // Keep shader-code pages apart from outputs. Code synchronization otherwise
            // legitimately drains GPU writes to that page and hides the dependency cut.
            rig.Program(IncrementBuffer);
            rig.Program(ComputeScenePrograms.AtomicAdd);
            rig.Program(ComputeScenePrograms.WriteDispatchArgs);
            rig.Memory.Allocate(4096, 4096);
            var first = Fill(rig, 256, 0);
            var second = Fill(rig, 256, 0);
            var originalFirst = first;
            var originalSecond = second;
            var pm4 = new ScenePm4();
            if (chain == "producer-consumer")
            {
                rig.Memory.Write<uint>(first, Enumerable.Range(0, 256).Select(i => (uint)i * 7).ToArray());
                for (var index = 0; index < 32; index++)
                {
                    Setup(rig, pm4, IncrementBuffer, [.. Vsharp(first, 4, 256), .. Vsharp(second, 4, 256)], 64)
                        .DispatchDirect(4, 1, 1);
                    (first, second) = (second, first);
                }
            }
            else if (chain == "atomics")
            {
                for (var index = 0; index < 32; index++)
                    Setup(rig, pm4, ComputeScenePrograms.AtomicAdd, Vsharp(first, 4, 256), 64)
                        .DispatchDirect(3, 1, 1);
                Setup(rig, pm4, IncrementBuffer, [.. Vsharp(first, 4, 256), .. Vsharp(second, 4, 256)], 64)
                    .DispatchDirect(4, 1, 1);
                first = second;
            }
            else if (chain == "cp-write")
            {
                Setup(rig, pm4, IncrementBuffer, [.. Vsharp(first, 4, 256), .. Vsharp(second, 4, 256)], 64)
                    .DispatchDirect(4, 1, 1);
                pm4.Packet(ScenePm4.OpWriteData, 5u << 8, (uint)first, (uint)(first >> 32), 41u);
                Setup(rig, pm4, IncrementBuffer, [.. Vsharp(first, 4, 256), .. Vsharp(second, 4, 256)], 64)
                    .DispatchDirect(4, 1, 1);
                first = second;
            }
            else
            {
                var args = Fill(rig, 16, 0xAAAAAAAA);
                Setup(rig, pm4, ComputeScenePrograms.WriteDispatchArgs, [.. Vsharp(args, 4, 3), 3, 1, 1], 64)
                    .DispatchDirect(1, 1, 1);
                Setup(rig, pm4, ComputeScenePrograms.AtomicAdd, Vsharp(first, 4, 256), 64);
                SetBase(pm4, args, compute: true);
                pm4.Packet(ScenePm4.OpDispatchIndirect, 0u, 1u);
                Setup(rig, pm4, IncrementBuffer, [.. Vsharp(first, 4, 256), .. Vsharp(second, 4, 256)], 64)
                    .DispatchDirect(4, 1, 1);
                first = second;
            }

            // Cold pipeline suspension is a legitimate encoder boundary. Warm every
            // pipeline first, then reset the inputs and require cuts within one epoch.
            rig.Submit(pm4);
            rig.Memory.Write<uint>(originalFirst, chain == "producer-consumer"
                ? Enumerable.Range(0, 256).Select(i => (uint)i * 7).ToArray() : new uint[256]);
            rig.Memory.Write<uint>(originalSecond, new uint[256]);
            var barriersBefore = rig.Host.ComputeDependencyBarriers;
            rig.Submit(pm4);
            Assert.Equal(0, rig.Host.ComputeGuestBarriers);
            var barriers = rig.Host.ComputeDependencyBarriers - barriersBefore;
            if (chain == "cp-write") Assert.True(rig.Host.TakeDeferredLabelWrites() >= 2);
            else if (concurrent) Assert.True(barriers >= (chain == "indirect" ? 2 : 31),
                $"chain={chain} dependency barriers={barriers}");
            else Assert.Equal(0, rig.Host.ComputeDependencyBarriers);
            var values = rig.Read<uint>(first, 256);
            if (chain == "producer-consumer")
                Assert.Equal(Enumerable.Range(0, 256).Select(i => (uint)i * 7 + 32), values);
            else
            {
                Assert.Equal(chain == "atomics" ? 32u * 192 + 1 : chain == "cp-write" ? 42u : 193u, values[0]);
                Assert.All(values.Skip(1), value => Assert.Equal(1u, value));
            }

            return rig.Read<byte>(first, 1024);
        }

        var serial = Run(false);
        if (serial is null) return;
        Assert.Equal(serial, Run(true));
    }

    [Fact]
    public void IndependentDispatchesDoNotInsertDependencyBarriers()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.ConcurrentComputeEnabled = true;
        var pm4 = new ScenePm4();
        var outputs = new List<ulong>();
        for (var index = 0; index < 16; index++)
        {
            var input = Fill(rig, 64, (uint)index);
            var output = Fill(rig, 64, 0);
            outputs.Add(output);
            Setup(rig, pm4, IncrementBuffer, [.. Vsharp(input, 4, 64), .. Vsharp(output, 4, 64)], 64)
                .DispatchDirect(1, 1, 1);
        }

        rig.Submit(pm4);
        Assert.Equal(0, rig.Host.ComputeDependencyBarriers);
        Assert.Equal(15, rig.Host.ComputeIndependentDispatches);
        for (var index = 0; index < outputs.Count; index++)
            Assert.All(rig.Read<uint>(outputs[index], 64), value => Assert.Equal((uint)index + 1, value));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MixedDispatchStreamUsesTheSameEncodersAsSerial(bool touchingCopies)
    {
        (byte[] Bytes, long Encoders, long AllEncoders, long Barriers, long Independent)? Run(bool concurrent)
        {
            using var rig = CreateRig();
            if (rig is null) return null;
            rig.Host.ConcurrentComputeEnabled = concurrent;
            rig.Program(IncrementBuffer);
            rig.Memory.Allocate(4096, 4096);
            var buffers = Enumerable.Range(0, 4).Select(_ => Fill(rig, 64, 0)).ToArray();
            var label = Fill(rig, 64, 0);
            var expected = new uint[4];
            var pm4 = new ScenePm4();
            for (var index = 0; index < 64; index++)
            {
                // Two disjoint pairs can overlap; alternating direction in each pair
                // introduces RAW, WAR and WAW cuts. Unrelated CP writes must stay held.
                var source = index % 4;
                var destination = source ^ 1;
                var copyTarget = touchingCopies && index % 8 == 0 ? buffers[source] : label;
                pm4.Packet(ScenePm4.OpWriteData, 5u << 8,
                    (uint)copyTarget, (uint)(copyTarget >> 32), 41u);
                if (copyTarget != label) expected[source] = 41;
                expected[destination] = expected[source] + 1;
                Setup(rig, pm4, IncrementBuffer,
                        [.. Vsharp(buffers[source], 4, 64), .. Vsharp(buffers[destination], 4, 64)], 64)
                    .DispatchDirect(1, 1, 1);
            }

            rig.Submit(pm4); // warm the pipeline before counting real stream boundaries
            foreach (var buffer in buffers) rig.Memory.Write<uint>(buffer, new uint[64]);
            var encodersBefore = rig.Host.ComputeEncoders;
            var allBefore = MetalMemoryTelemetry.ComputeEncoders.Total + MetalMemoryTelemetry.BlitEncoders.Total +
                MetalMemoryTelemetry.RenderEncoders.Total;
            var dispatchesBefore = rig.Host.ComputeDispatches;
            var barriersBefore = rig.Host.ComputeDependencyBarriers;
            var independentBefore = rig.Host.ComputeIndependentDispatches;
            rig.Submit(pm4);
            var encoders = rig.Host.ComputeEncoders - encodersBefore;
            var allEncoders = MetalMemoryTelemetry.ComputeEncoders.Total + MetalMemoryTelemetry.BlitEncoders.Total +
                MetalMemoryTelemetry.RenderEncoders.Total - allBefore;
            Assert.Equal(64, rig.Host.ComputeDispatches - dispatchesBefore);
            Assert.True(encoders > 0);
            if (!concurrent) Assert.Equal(barriersBefore, rig.Host.ComputeDependencyBarriers);
            for (var index = 0; index < buffers.Length; index++)
                Assert.Equal(expected[index], rig.Read<uint>(buffers[index], 64)[0]);
            Assert.Equal(41u, rig.Read<uint>(label, 1)[0]);
            var bytes = buffers.SelectMany(buffer => rig.Read<byte>(buffer, 256)).ToArray();
            return (bytes, encoders, allEncoders, rig.Host.ComputeDependencyBarriers - barriersBefore,
                rig.Host.ComputeIndependentDispatches - independentBefore);
        }

        var serial = Run(false);
        if (serial is null) return;
        var concurrent = Run(true)!.Value;
        Output.WriteLine($"touchingCopies={touchingCopies} serial/concurrent compute encoders=" +
            $"{serial.Value.Encoders}/{concurrent.Encoders}, all encoders=" +
            $"{serial.Value.AllEncoders}/{concurrent.AllEncoders}, dependency barriers=" +
            $"{concurrent.Barriers}, independent={concurrent.Independent}");
        Assert.Equal(serial.Value.Bytes, concurrent.Bytes);
        Assert.Equal(serial.Value.Encoders, concurrent.Encoders);
        if (SharpEmu.Logging.MemoryTelemetry.Enabled)
            Assert.Equal(serial.Value.AllEncoders, concurrent.AllEncoders);
        if (!touchingCopies) Assert.Equal(1, concurrent.Encoders);
        Assert.True(concurrent.Barriers > 0);
        Assert.True(concurrent.Independent > 0);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EncoderDispatchTypeAndAllGuestBarriersArePreserved(bool concurrent)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        Assert.Equal(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_CONCURRENT_COMPUTE") != "0",
            rig.Host.TracksComputeDependencies);
        rig.Host.ConcurrentComputeEnabled = concurrent;
        var encoder = rig.Host.EnsureComputeEncoder();
        Assert.Equal(concurrent ? 1ul : 0ul,
            (ulong)MetalV2Native.SendValue(encoder, MetalNative.Selector("dispatchType")));
        if (MetalRenderHost.PassProfileEnabled)
        {
            Assert.NotNull(typeof(MetalRenderHost).GetField("_ppOpen",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(rig.Host));
        }

        for (var index = 0; index < 99; index++) rig.Host.EmitGlobalBarrier();
        Assert.Equal(99, rig.Host.ComputeGuestBarriers);
        rig.Host.FlushAndWaitForGpu();
    }
}
