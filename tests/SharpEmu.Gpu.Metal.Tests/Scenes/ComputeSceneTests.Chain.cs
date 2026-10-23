// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Xunit;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

public sealed partial class ComputeSceneTests
{
    [Fact]
    public void MixedEndOfPipeLabelsAndCpWritesKeepTheCompletionBoundary()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.ComputeChainEnabled = true;
        rig.Program(IncrementBuffer);
        rig.Memory.Allocate(4096, 4096);
        var data = Fill(rig, 64, 0);
        var output = Fill(rig, 64, 0);
        var label = Fill(rig, 2, 0);
        var pm4 = new ScenePm4();
        Setup(rig, pm4, IncrementBuffer, [.. Vsharp(data, 4, 64), .. Vsharp(output, 4, 64)], 64)
            .DispatchDirect(1, 1, 1);
        pm4.Packet(0x47, 0x504u, (uint)label, (uint)(label >> 32) | (2u << 29), 41u, 0u);
        pm4.Packet(ScenePm4.OpWriteData, 5u << 8, (uint)data, (uint)(data >> 32), 71u);
        Setup(rig, pm4, IncrementBuffer, [.. Vsharp(data, 4, 64), .. Vsharp(output, 4, 64)], 64)
            .DispatchDirect(1, 1, 1);
        rig.Submit(pm4);
        Assert.Equal(0, rig.Host.InEncoderLabelWrites);
        Assert.Equal(41ul, rig.Read<ulong>(label, 1)[0]);
        Assert.Equal(72u, rig.Read<uint>(output, 1)[0]);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void CommandProcessorConsumersKeepOneEncoderAndMatchSerialBytes(bool overlap, bool splitWorker)
    {
        (byte[] Bytes, long Encoders, long Copies)? Run(bool chain)
        {
            using var rig = CreateRig(splitWorker);
            if (rig is null) return null;
            rig.Host.ComputeChainEnabled = chain;
            rig.Host.ConcurrentComputeEnabled = chain;
            rig.Host.LabelsInEncoder = chain;
            rig.Program(IncrementBuffer);
            rig.Memory.Allocate(4096, 4096);
            var data = Fill(rig, 64, 0);
            var output = Fill(rig, 64, 0);
            var pm4 = new ScenePm4();
            // Open the encoder first, so all following CP landings have an encoder
            // to share. New user-data/argument snapshots accompany every dispatch.
            Setup(rig, pm4, IncrementBuffer, [.. Vsharp(data, 4, 64), .. Vsharp(output, 4, 64)], 64)
                .DispatchDirect(1, 1, 1);
            for (uint index = 0; index < 16; index++)
            {
                pm4.Packet(ScenePm4.OpWriteData, 5u << 8, (uint)data, (uint)(data >> 32), index + 71);
                if (overlap)
                    pm4.Packet(ScenePm4.OpWriteData, 5u << 8, (uint)data, (uint)(data >> 32), index + 91);
                Setup(rig, pm4, IncrementBuffer, [.. Vsharp(data, 4, 64), .. Vsharp(output, 4, 64)], 64)
                    .DispatchDirect(1, 1, 1);
            }
            rig.Submit(pm4);
            rig.Memory.Write<uint>(data, new uint[64]);
            rig.Memory.Write<uint>(output, new uint[64]);
            var encoders = rig.Host.ComputeEncoders;
            var copies = rig.Host.InEncoderLabelWrites;
            rig.Submit(pm4);
            var actual = rig.Read<uint>(output, 64);
            Assert.Equal(overlap ? 107u : 87u, actual[0]);
            Assert.All(actual.Skip(1), word => Assert.Equal(1u, word));
            return (rig.Read<byte>(output, 256), rig.Host.ComputeEncoders - encoders,
                rig.Host.InEncoderLabelWrites - copies);
        }
        var serial = Run(false);
        if (serial is null) return;
        var chain = Run(true)!.Value;
        Assert.Equal(serial.Value.Bytes, chain.Bytes);
        Assert.Equal(1, chain.Encoders);
        Assert.True(serial.Value.Encoders > chain.Encoders);
        Assert.Equal(overlap ? 32 : 16, chain.Copies);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BoundedDeviceAtomicRequestsMatchSerialWithAndWithoutDependencies(bool sharedCounter)
    {
        byte[]? Run(bool chain)
        {
            using var rig = CreateRig();
            if (rig is null) return null;
            rig.Host.ComputeChainEnabled = chain;
            rig.Host.ConcurrentComputeEnabled = chain;
            rig.Program(ComputeScenePrograms.GlobalAtomicAdd);
            rig.Memory.Allocate(4096, 4096);
            var counters = Enumerable.Range(0, 8).Select(_ => Fill(rig, 16, 0)).ToArray();
            var outputs = Enumerable.Range(0, 8).Select(_ => Fill(rig, 64, 0)).ToArray();
            var pm4 = new ScenePm4();
            for (var index = 0; index < 8; index++)
            {
                var counter = counters[sharedCounter ? 0 : index];
                Setup(rig, pm4, ComputeScenePrograms.GlobalAtomicAdd,
                    [Descriptors.Lo(counter), Descriptors.Hi(counter), 0, 0, .. Vsharp(outputs[index], 4, 64)], 64)
                    .DispatchDirect(1, 1, 1);
            }
            rig.Submit(pm4);
            foreach (var counter in counters) rig.Memory.Write<uint>(counter, new uint[16]);
            var batched = rig.Host.ComputeBatchedDispatches;
            var dependencies = rig.Host.ComputeDependencyBarriers;
            rig.Submit(pm4);
            if (chain)
            {
                var actualBatched = rig.Host.ComputeBatchedDispatches - batched;
                var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
                var diagnostics = rig.Host.GetType().GetMethod("ComputeBatchCounters", flags)!.Invoke(rig.Host, null) + " " +
                    rig.Host.GetType().GetMethod("ComputeChainCounters", flags)!.Invoke(rig.Host, null);
                Assert.True(actualBatched == (sharedCounter ? 0 : 8), $"batched={actualBatched} {diagnostics}");
                Assert.Equal(sharedCounter ? 7 : 0, rig.Host.ComputeDependencyBarriers - dependencies);
            }
            for (var index = 0; index < 8; index++)
                Assert.Equal(Enumerable.Range(sharedCounter ? index * 64 : 0, 64).Select(i => (uint)i),
                    rig.Read<uint>(outputs[index], 64).OrderBy(i => i));
            return counters.SelectMany(counter => rig.Read<byte>(counter, 64)).ToArray();
        }
        var serial = Run(false);
        if (serial is not null) Assert.Equal(serial, Run(true));
    }
}
