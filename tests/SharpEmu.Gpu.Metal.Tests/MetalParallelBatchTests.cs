// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.GpuCommands;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

public sealed class MetalParallelBatchTests
{
    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(63)] [InlineData(64)]
    [InlineData(65)] [InlineData(6331)] [InlineData(9000)]
    public void ContiguousChunksReconstructEveryGuestPositionExactlyOnce(int count)
    {
        foreach (var workers in new[] { 1, 2, 4, 10 })
        {
            var positions = new List<int>();
            for (var worker = 0; worker < workers; worker++)
            {
                var (start, end) = MetalParallelBatchWorkers.Chunk(count, workers, worker);
                positions.AddRange(Enumerable.Range(start, end - start));
            }
            Assert.Equal(Enumerable.Range(0, count), positions);
        }
    }

    [Fact]
    public void ChunksOverlapAndInputStorageIsNotReleasedBeforeEveryChunkFinishes()
    {
        using var entered = new CountdownEvent(4);
        using var release = new ManualResetEventSlim(false);
        var threads = new int[4];
        var output = new int[257];
        using var workers = new MetalParallelBatchWorkers(4, worker =>
        {
            threads[worker] = Environment.CurrentManagedThreadId;
            entered.Signal();
            Assert.True(release.Wait(TimeSpan.FromSeconds(10)));
            var (start, end) = MetalParallelBatchWorkers.Chunk(output.Length, 4, worker);
            for (var i = start; i < end; i++) output[i] = i + 1;
        });
        var execution = Task.Run(workers.Execute);
        try
        {
            Assert.True(entered.Wait(TimeSpan.FromSeconds(10)));
            Assert.Equal(4, threads.Distinct().Count());
            Assert.False(execution.IsCompleted);
            Assert.All(output, value => Assert.Equal(0, value));
        }
        finally { release.Set(); }
        Assert.True(execution.Wait(TimeSpan.FromSeconds(10)));
        Assert.Equal(Enumerable.Range(1, output.Length), output);
    }

    [Fact]
    public void AFailedChunkStillJoinsTheOtherChunksBeforeReturningItsStorage()
    {
        using var entered = new CountdownEvent(3);
        using var release = new ManualResetEventSlim(false);
        var completed = 0;
        using var workers = new MetalParallelBatchWorkers(4, worker =>
        {
            if (worker == 0) throw new InvalidOperationException("held batch failure");
            entered.Signal();
            Assert.True(release.Wait(TimeSpan.FromSeconds(10)));
            Interlocked.Increment(ref completed);
        });
        var execution = Task.Run(() => Assert.Throws<InvalidOperationException>(workers.Execute));
        try
        {
            Assert.True(entered.Wait(TimeSpan.FromSeconds(10)));
            Assert.False(execution.IsCompleted);
            Assert.Equal(0, Volatile.Read(ref completed));
        }
        finally { release.Set(); }
        Assert.True(execution.Wait(TimeSpan.FromSeconds(10)));
        Assert.Equal(3, completed);
        // A second generation still joins correctly after the first propagated a
        // failure; no completion signal from the old batch can retire this one.
        entered.Reset(3);
        Assert.Throws<InvalidOperationException>(workers.Execute);
        Assert.Equal(6, completed);
    }

    [Fact]
    public void UnboundedStoresAndStorageImagesCannotEnterAnAttachmentOnlyChunk()
    {
        Assert.True(MetalRenderHost.ParallelReadOnly(new ShaderProgramInfo()));
        Assert.False(MetalRenderHost.ParallelReadOnly(new ShaderProgramInfo { WritesDeviceAddresses = true }));
        Assert.False(MetalRenderHost.ParallelReadOnly(new ShaderProgramInfo
            { Buffers = [new BufferResourceInfo(true, false, true, false, false, 4, 4)] }));
        Assert.False(MetalRenderHost.ParallelReadOnly(new ShaderProgramInfo
            { Images = [new ImageResourceInfo(ImageResourceClass.Storage, false)] }));
    }

    [Fact]
    public void CapacityAndObservationCutsHaveSeparateNonemptyBatchDistributions()
    {
        var census = new MetalBatchCensus();
        for (var i = 0; i < 129; i++)
        {
            census.Work(compute: false);
            census.Encoded();
            census.Strict.Close(MetalBatchCensus.Cut.CommandRead);
        }
        census.Close(MetalBatchCensus.Cut.Frame);
        Assert.Equal(129, census.Strict.Batches);
        Assert.Equal(129, census.Strict.Histogram[1]);
        Assert.Equal(2, census.Coarse.Histogram[64]);
        Assert.Equal(1, census.Coarse.Histogram[1]);
        Assert.Equal(129, census.Coarse.Work);
        Assert.Equal(129, census.Strict.Work);
        Assert.Equal(2, census.Coarse.DrawHistogram[64]);
        Assert.All(census.Coarse.DispatchHistogram, count => Assert.Equal(0, count));
        Assert.Equal(2, census.Coarse.NonemptyCuts[(int)MetalBatchCensus.Cut.Capacity]);
    }

    [Fact]
    public void DrawAndDispatchDistributionsDoNotMixWorkKindsOrCountEmptyCutsAsBatches()
    {
        var census = new MetalBatchCensus();
        for (var i = 0; i < 3; i++) { census.Work(false); census.Encoded(); }
        for (var i = 0; i < 2; i++) { census.Work(true); census.Encoded(); }
        census.Close(MetalBatchCensus.Cut.Frame);
        census.Close(MetalBatchCensus.Cut.Frame);
        Assert.Equal(1, census.Coarse.DrawHistogram[3]);
        Assert.Equal(1, census.Coarse.DispatchHistogram[2]);
        Assert.Equal(2, census.Coarse.Batches);
        Assert.Equal(2, census.Coarse.Cuts[(int)MetalBatchCensus.Cut.Frame]);
        Assert.Equal(1, census.Coarse.NonemptyCuts[(int)MetalBatchCensus.Cut.Frame]);
    }

    [Theory]
    [InlineData(PacketOpcode.WaitRegisterMemory, (int)MetalBatchCensus.Cut.Wait)]
    [InlineData(PacketOpcode.ReleaseMemory, (int)MetalBatchCensus.Cut.Label)]
    [InlineData(PacketOpcode.DmaData, (int)MetalBatchCensus.Cut.Dma)]
    [InlineData(PacketOpcode.AcquireMemory, (int)MetalBatchCensus.Cut.Barrier)]
    [InlineData(PacketOpcode.SetShaderRegisterIndirect, (int)MetalBatchCensus.Cut.RegisterTable)]
    public void SemanticObservationsCutBeforeTheirEffects(uint opcode, int expected) =>
        Assert.Equal((MetalBatchCensus.Cut)expected, MetalBatchCensus.PacketCut(PacketHeader.Make(2, opcode)));

    [Theory]
    [InlineData(PacketCustomCode.ReleaseMemory, (int)MetalBatchCensus.Cut.Label)]
    [InlineData(PacketCustomCode.DmaData, (int)MetalBatchCensus.Cut.Dma)]
    [InlineData(PacketCustomCode.AcquireMemory, (int)MetalBatchCensus.Cut.Barrier)]
    [InlineData(PacketCustomCode.ShaderRegisterTable, (int)MetalBatchCensus.Cut.RegisterTable)]
    public void WrappedPacketsUseTheSameSemanticBoundaries(uint custom, int expected) =>
        Assert.Equal((MetalBatchCensus.Cut)expected,
            MetalBatchCensus.PacketCut(PacketHeader.Make(4, PacketOpcode.Nop, custom)));


    [Fact]
    public void SymbolicTableLoadsAreOpportunitiesRatherThanCertifiedReadPermissions()
    {
        var census = new MetalBatchCensus();
        for (var i = 0; i < 64; i++)
        {
            census.Work(false);
            census.Encoded();
            census.Packet(PacketHeader.Make(4, PacketOpcode.SetShaderRegisterIndirect));
        }
        census.Close(MetalBatchCensus.Cut.Frame);
        Assert.Equal(64, census.Coarse.Histogram[1]);
        Assert.Equal(1, census.Symbolic.Histogram[64]);
        Assert.Equal(0, census.Symbolic.Cuts[(int)MetalBatchCensus.Cut.RegisterTable]);
    }

}
