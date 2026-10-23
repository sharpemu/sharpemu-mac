// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

public sealed partial class ComputeSceneTests
{
    private static readonly uint[] BatchAppend =
    [
        0x7E040281,             // v_mov_b32 v2, 1
        0x7E080280,             // v_mov_b32 v4, 0
        0xE0C86000, 0x80000204, // buffer_atomic_add v2, v4, s[0:3], 0 idxen glc
        0xBF8C3F70,
        0x7E060208,             // v_mov_b32 v3, s8 (request's payload)
        0xE0702000, 0x80010302, // buffer_store_dword v3, v2, s[4:7], 0 idxen
        0xBF810000,
    ];

    // A wave64 mask read as scalar data must exchange both Apple simdgroup halves.
    private static readonly uint[] BatchWave =
    [
        0x8F0D860C,             // s_lshl_b32 s13, s12, 6 (group ID)
        0x4A02000D,             // v_add_nc_u32 v1, s13, v0
        0xD4C40002, 0x00017700, // v_cmp_gt_u32 s[2:3], v0, 59
        0x7E040203,             // v_mov_b32 v2, s3 (high half of ballot)
        0xE0702000, 0x80020201, // buffer_store_dword v2, v1, s[8:11], 0 idxen
        0xBF810000,
    ];

    [Theory]
    [InlineData("resources")]
    [InlineData("ids")]
    [InlineData("permuted")]
    [InlineData("partial")]
    [InlineData("lds")]
    [InlineData("wave64")]
    [InlineData("append")]
    public void SegmentedDispatchesMatchUnbatchedBytes(string kind)
    {
        byte[]? Run(bool batch)
        {
            using var rig = CreateRig();
            if (rig is null) return null;
            rig.Host.ConcurrentComputeEnabled = true;
            rig.Host.ComputeBatchEnabled = batch;
            var program = kind switch
            {
                "ids" or "partial" or "permuted" => ComputeScenePrograms.MultiDimXYZ,
                "lds" => ComputeScenePrograms.LdsReduce,
                "wave64" => BatchWave,
                "append" => BatchAppend,
                _ => IncrementBuffer,
            };
            rig.Program(program);
            rig.Memory.Allocate(4096, 4096);
            var pm4 = new ScenePm4();
            var outputs = new List<(ulong Address, uint Words, uint[] Expected)>();
            // Non-uniform request sizes, including one workgroup. All requests
            // have different descriptor/user-data snapshots and different storage.
            for (uint index = 0; index < 12; index++)
            {
                var gx = index % 4 + 1;
                var gy = kind is "ids" or "partial" or "permuted" ? index % 3 + 1 : 1;
                var gz = kind is "ids" or "partial" or "permuted" ? index % 2 + 1 : 1;
                if (kind is "ids" or "partial" or "permuted")
                {
                    uint tx = kind == "permuted" ? 2u : 4u, ty = kind == "permuted" ? 8u : 4u, tz = 4;
                    var totalX = gx * tx - (kind == "partial" ? index % 3 : 0);
                    var totalY = gy * ty - (kind == "partial" ? index % 2 : 0);
                    var totalZ = gz * tz;
                    var total = totalX * totalY * totalZ;
                    var result = Fill(rig, total * 2, 0xFFFFFFFF);
                    var expected = new uint[total * 2];
                    for (uint z = 0; z < totalZ; z++)
                    for (uint y = 0; y < totalY; y++)
                    for (uint x = 0; x < totalX; x++)
                    {
                        var at = ((z * totalY) + y) * totalX + x;
                        (expected[at * 2], expected[at * 2 + 1]) = MultiDimExpected(x, y, z, tx, ty, tz, true, true);
                    }
                    outputs.Add((result, total * 2, expected));
                    Setup(rig, pm4, program, [.. Vsharp(result, 8, total), totalX, totalY, tx, ty, tz],
                        tx, ty, tz, true, true, 2).DispatchDirect(
                            kind == "partial" ? totalX : gx, kind == "partial" ? totalY : gy,
                            kind == "partial" ? totalZ : gz, kind == "partial" ? 0x21u : 1u);
                }
                else if (kind == "lds")
                {
                    var input = Fill(rig, gx * 256, index + 7);
                    var result = Fill(rig, gx, 0xFFFFFFFF);
                    outputs.Add((result, gx, Enumerable.Repeat((index + 7) * 256, (int)gx).ToArray()));
                    Setup(rig, pm4, program, [.. Vsharp(input, 4, gx * 256), .. Vsharp(result, 4, gx)],
                        256, ldsBytes: 1024).DispatchDirect(gx, 1, 1);
                }
                else if (kind == "append")
                {
                    var counter = Fill(rig, 1, 0);
                    var result = Fill(rig, gx * 64, 0xFFFFFFFF);
                    outputs.Add((counter, 1, [gx * 64]));
                    outputs.Add((result, gx * 64, Enumerable.Repeat(index + 71, (int)gx * 64).ToArray()));
                    Setup(rig, pm4, program, [.. Vsharp(counter, 4, 1), .. Vsharp(result, 4, gx * 64), index + 71],
                        64).DispatchDirect(gx, 1, 1);
                }
                else
                {
                    var result = Fill(rig, gx * 64, 0xFFFFFFFF);
                    if (kind == "wave64")
                    {
                        outputs.Add((result, gx * 64, Enumerable.Repeat(0xF0000000u, (int)gx * 64).ToArray()));
                        Setup(rig, pm4, program, [0, 0, 0, 0, 0, 0, 0, 0, .. Vsharp(result, 4, gx * 64)],
                            64).DispatchDirect(gx, 1, 1);
                    }
                    else
                    {
                        var input = Fill(rig, gx * 64, index * 11);
                        outputs.Add((result, gx * 64, Enumerable.Repeat(index * 11 + 1, (int)gx * 64).ToArray()));
                        Setup(rig, pm4, program, [.. Vsharp(input, 4, gx * 64), .. Vsharp(result, 4, gx * 64)],
                            64).DispatchDirect(gx, 1, 1);
                    }
                }
            }
            rig.Submit(pm4);
            foreach (var result in outputs)
            {
                // Append counters must be reset too before the second stream.
                rig.Memory.Write<uint>(result.Address, new uint[result.Words]);
            }
            var batchedBefore = rig.Host.ComputeBatchedDispatches;
            var batchesBefore = rig.Host.ComputeBatches;
            rig.Submit(pm4); // warm stream: pipeline suspension cannot split the batch
            Assert.Equal(batch ? 12 : 0, rig.Host.ComputeBatchedDispatches - batchedBefore);
            Assert.Equal(batch ? 1 : 0, rig.Host.ComputeBatches - batchesBefore);
            foreach (var result in outputs)
                Assert.Equal(result.Expected, rig.Read<uint>(result.Address, (int)result.Words));
            return outputs.SelectMany(result => rig.Read<byte>(result.Address, (int)result.Words * 4)).ToArray();
        }
        var unbatched = Run(false);
        if (unbatched is not null) Assert.Equal(unbatched, Run(true));
    }

    [Theory]
    [InlineData("barrier")]
    [InlineData("acquire")]
    [InlineData("partial-flush")]
    [InlineData("cp-write")]
    [InlineData("dependency")]
    [InlineData("indirect")]
    [InlineData("initiator")]
    [InlineData("label")]
    [InlineData("transfer")]
    public void SegmentedDispatchesFlushAtGuestBoundaries(string boundary)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.ConcurrentComputeEnabled = true;
        rig.Host.ComputeBatchEnabled = true;
        rig.Program(IncrementBuffer);
        rig.Program(ComputeScenePrograms.WriteDispatchArgs);
        rig.Memory.Allocate(4096, 4096);
        var inputs = Enumerable.Range(0, 4).Select(i => Fill(rig, 64, (uint)i)).ToArray();
        var outputs = Enumerable.Range(0, 4).Select(_ => Fill(rig, 64, 0)).ToArray();
        var args = Fill(rig, 3, 0);
        var label = Fill(rig, 2048, 0);
        var pm4 = new ScenePm4();
        if (boundary == "indirect")
            Setup(rig, pm4, ComputeScenePrograms.WriteDispatchArgs, [.. Vsharp(args, 4, 3), 1, 1, 1], 64)
                .DispatchDirect(1, 1, 1);
        for (var index = 0; index < 4; index++)
        {
            if (index == 2)
            {
                if (boundary == "barrier") pm4.Packet(ScenePm4.OpAcquireMemory, 0x00800000u, 0u, 0u, 0u, 0u, 0u);
                if (boundary == "acquire") pm4.Packet(ScenePm4.OpAcquireMemory, 0u, 0u, 0u, 0u, 0u, 0u);
                if (boundary == "partial-flush") CsFlush(pm4);
                if (boundary == "label") pm4.Packet(0x47, 0x504u, (uint)label, (uint)(label >> 32) | (2u << 29), 41u, 0u);
                if (boundary == "transfer") pm4.Packet(0x50, 2u << 29, 41u, 0u, (uint)label, (uint)(label >> 32), 8192u);
                if (boundary == "cp-write") pm4.Packet(ScenePm4.OpWriteData, 5u << 8, (uint)label, (uint)(label >> 32), 41u);
            }
            var source = boundary == "dependency" && index == 2 ? outputs[0] : inputs[index];
            Setup(rig, pm4, IncrementBuffer, [.. Vsharp(source, 4, 64), .. Vsharp(outputs[index], 4, 64)], 64);
            if (boundary == "indirect" && index == 2)
            {
                SetBase(pm4, args, compute: true);
                pm4.Packet(ScenePm4.OpDispatchIndirect, 0u, 1u);
            }
            else pm4.DispatchDirect(1, 1, 1, boundary == "initiator" && index >= 2 ? 0x41u : 1u);
        }
        rig.Submit(pm4);
        var before = rig.Host.ComputeBatchedDispatches;
        rig.Submit(pm4);
        Assert.Equal(boundary == "indirect" ? 2 : 4, rig.Host.ComputeBatchedDispatches - before);
        var reason = boundary switch { "partial-flush" or "barrier" => "barrier", _ => boundary };
        Assert.True(rig.Host.ComputeBatchFlushes(reason) > 0, $"No {reason} flush recorded");
        for (var index = 0; index < 4; index++)
            Assert.All(rig.Read<uint>(outputs[index], 64), value => Assert.Equal(
                boundary == "dependency" && index == 2 ? 2u : (uint)index + 1, value));
    }
    [Fact]
    public void SegmentedDispatchesBoundTheTableAndKeepTheTail()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.ConcurrentComputeEnabled = true;
        rig.Host.ComputeBatchEnabled = true;
        rig.Program(IncrementBuffer);
        rig.Memory.Allocate(4096, 4096);
        var input = Fill(rig, 64, 17);
        var outputs = Enumerable.Range(0, 129).Select(_ => Fill(rig, 64, 0)).ToArray();
        var pm4 = new ScenePm4();
        foreach (var output in outputs)
            Setup(rig, pm4, IncrementBuffer, [.. Vsharp(input, 4, 64), .. Vsharp(output, 4, 64)], 64)
                .DispatchDirect(1, 1, 1);
        rig.Submit(pm4);
        var before = rig.Host.ComputeBatchedDispatches;
        var batches = rig.Host.ComputeBatches;
        rig.Submit(pm4);
        Assert.Equal(128, rig.Host.ComputeBatchedDispatches - before);
        Assert.Equal(1, rig.Host.ComputeBatches - batches);
        Assert.True(rig.Host.ComputeBatchFlushes("capacity") > 0);
        foreach (var output in outputs)
            Assert.All(rig.Read<uint>(output, 64), value => Assert.Equal(18u, value));
    }

    [Fact]
    public void SegmentedDispatchesSplitDifferentLdsSpecializations()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.ConcurrentComputeEnabled = true;
        rig.Host.ComputeBatchEnabled = true;
        rig.Program(ComputeScenePrograms.LdsReduce);
        rig.Memory.Allocate(4096, 4096);
        var input = Fill(rig, 256, 9);
        var outputs = Enumerable.Range(0, 4).Select(_ => Fill(rig, 1, 0)).ToArray();
        var pm4 = new ScenePm4();
        for (var index = 0; index < outputs.Length; index++)
            Setup(rig, pm4, ComputeScenePrograms.LdsReduce,
                [.. Vsharp(input, 4, 256), .. Vsharp(outputs[index], 4, 1)], 256,
                ldsBytes: index < 2 ? 1024u : 2048u).DispatchDirect(1, 1, 1);
        rig.Submit(pm4);
        var before = rig.Host.ComputeBatchedDispatches;
        var batches = rig.Host.ComputeBatches;
        rig.Submit(pm4);
        Assert.Equal(4, rig.Host.ComputeBatchedDispatches - before);
        Assert.Equal(2, rig.Host.ComputeBatches - batches);
        Assert.True(rig.Host.ComputeBatchFlushes("pipeline") > 0);
        foreach (var output in outputs) Assert.Equal(9u * 256, rig.Read<uint>(output, 1)[0]);
    }

}
