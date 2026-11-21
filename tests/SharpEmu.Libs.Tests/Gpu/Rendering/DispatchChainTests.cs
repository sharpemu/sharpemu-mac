// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Images;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.Libs.Tests.Gpu.Rendering.RenderExecutorFixtures;

namespace SharpEmu.Libs.Tests.Gpu.Rendering;

[Collection(SchedulingStateCollection.Name)]
public sealed class DispatchChainTests : IDisposable
{
    private const ulong BufferA = RecordingRenderHost.MemoryBase + 0x70_0000;
    private const ulong BufferB = RecordingRenderHost.MemoryBase + 0x71_0000;
    private readonly RecordingRenderHost _host = new() { DeferBarriers = true };
    private readonly FakePipelineProvider _pipelines = new();
    private readonly RenderExecutor _executor;
    private readonly FatalScope _fatal = new();

    public DispatchChainTests() => _executor = new RenderExecutor(_host, _pipelines);

    public void Dispose() => _fatal.Dispose();

    private static BufferResourceInfo Written => new(false, true, false, false, false, 4);

    private static BufferResourceInfo Read => new(true, false, false, false, false, 4);

    private static BufferResourceInfo AtomicOnly => new(true, true, true, false, false, 4, AtomicOnly: true);

    // Atomics beside a load or a store of the same buffer.
    private static BufferResourceInfo AtomicAndPlain => new(true, true, true, false, false, 4);

    private void Dispatch(ulong submitId, BufferResourceInfo[] buffers, uint[][] descriptors)
    {
        _pipelines.Compute = ComputeProgram(Stage(Program(ShaderStageKind.Compute, buffers: buffers), buffers: descriptors));
        _executor.Dispatch(submitId, Banks(), 1, 1, 1, 0x41);
    }

    // A dispatch the stream reached from the one before it through register writes alone (DispatchOverlap).
    private void DispatchMarked(ulong submitId, BufferResourceInfo[] buffers, uint[][] descriptors)
    {
        _pipelines.Compute = ComputeProgram(Stage(Program(ShaderStageKind.Compute, buffers: buffers), buffers: descriptors));
        _executor.Dispatch(submitId, Banks(), 1, 1, 1, 0x41 | RenderExecutor.DispatchMayOverlapBit);
    }

    private int HazardBarriers => _host.Calls.Count(call => call == "write_hazard_barrier");

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RuntimeCalleeAccessesKeepTheBarrierOnEitherSideOfTheChain(bool calleeFirst)
    {
        var caller = new ShaderProgramInfo
        {
            Stage = ShaderStageKind.Compute,
            Buffers = [Written],
            UsesDeviceAddresses = true,
            CallRecordCallees = [new CallRecordCallee([], [], 16)],
        };
        var hidden = new ShaderStageResources(caller, new ResourceSnapshot
        {
            Buffers = [BufferDescriptor(BufferB, 4, 16)],
            // A nonempty ordinary range list must not hide the missing callee accesses.
            DeviceAddressRanges = [new(1, BufferB, 64, true, true)],
        })
        {
            CallRecords = [(0x1000, BufferA)],
        };
        DispatchWithHiddenAccesses(hidden, calleeFirst);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BindlessImageAccessesKeepTheBarrierOnEitherSideOfTheChain(bool heapFirst)
    {
        // The heap may read an image written by a previous dispatch, or write an image
        // the next dispatch reads. Neither image is in the caller's explicit image list.
        var caller = new ShaderProgramInfo
        {
            Stage = ShaderStageKind.Compute,
            Buffers = [Written],
            Bindings = new BindingLayout { UsesBindlessHeap = true },
        };
        DispatchWithHiddenAccesses(Stage(caller, [BufferDescriptor(BufferB, 4, 16)]), heapFirst);
    }

    private void DispatchWithHiddenAccesses(ShaderStageResources hidden, bool hiddenFirst)
    {
        var ordinary = Stage(Program(ShaderStageKind.Compute, buffers: [Written]), [BufferDescriptor(BufferA, 4, 16)]);
        _pipelines.Compute = ComputeProgram(hiddenFirst ? hidden : ordinary);
        _executor.Dispatch(1, Banks(), 1, 1, 1, 0x41);
        _pipelines.Compute = ComputeProgram(hiddenFirst ? ordinary : hidden);
        _executor.Dispatch(2, Banks(), 1, 1, 1, 0x41);

        Assert.Equal(2, HazardBarriers);
    }

    [Fact]
    public void DisjointDispatchesShareOneHazardBarrierAndDeferTheirAccessBarriers()
    {
        Dispatch(1, [Written], [BufferDescriptor(BufferA, 4, 16)]);
        Dispatch(2, [Written], [BufferDescriptor(BufferB, 4, 16)]);

        Assert.Equal(1, HazardBarriers);
        Assert.Equal(2, _host.Calls.Count(call => call == "defer_access_barrier"));
        Assert.DoesNotContain("access_barrier", _host.Calls);
        var window = _host.Calls.IndexOf("checked_begin");
        Assert.True(window >= 0 && _host.Calls.IndexOf("dispatch 1 1 1") > window && _host.Calls.IndexOf("checked_end") > _host.Calls.IndexOf("dispatch 1 1 1"));
    }

    [Fact]
    public void AReadOfAChainWriteTakesTheHazardBarrier()
    {
        Dispatch(1, [Written], [BufferDescriptor(BufferA, 4, 16)]);
        Dispatch(2, [Read], [BufferDescriptor(BufferA + 0x20, 4, 4)]);

        Assert.Equal(2, HazardBarriers);
    }

    [Fact]
    public void AWriteOverAChainReadTakesTheHazardBarrier()
    {
        Dispatch(1, [Read, Written], [BufferDescriptor(BufferA, 4, 16), BufferDescriptor(BufferB, 4, 16)]);
        Dispatch(2, [Written], [BufferDescriptor(BufferA + 0x10, 4, 4)]);

        Assert.Equal(2, HazardBarriers);
    }

    [Fact]
    public void APromotionOutsideTheChainStartsANewOne()
    {
        Dispatch(1, [Written], [BufferDescriptor(BufferA, 4, 16)]);
        _host.Promotions++;
        Dispatch(2, [Written], [BufferDescriptor(BufferB, 4, 16)]);

        Assert.Equal(2, HazardBarriers);
    }

    // HostKernelCopySceneTests' reader: a marked dispatch reading what the marked dispatch before it wrote ran beside it
    // and read the bytes from before the write. It is ordered after it now, and counted; disjoint ones still overlap.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void AMarkedDispatchThatReadsWhatTheOneBeforeItWroteIsOrderedAfterIt(bool checkHazards)
    {
        var saved = DispatchChain.OverlapHazardsChecked;
        DispatchChain.OverlapHazardsChecked = checkHazards;
        try
        {
            var before = DispatchChainTally.OverlapsOrdered;
            DispatchMarked(1, [Written], [BufferDescriptor(BufferA, 4, 16)]);
            DispatchMarked(2, [Read], [BufferDescriptor(BufferA + 0x20, 4, 4)]);
            // A third, disjoint from both, still runs beside the second (the group restarts at the ordered one).
            DispatchMarked(3, [Written], [BufferDescriptor(BufferB, 4, 16)]);

            Assert.Equal(checkHazards ? [true, false, true] : new[] { true, true, true }, _host.DispatchOverlaps);
            Assert.Equal(before.Hazard + (checkHazards ? 1 : 0), DispatchChainTally.OverlapsOrdered.Hazard);
            Assert.Equal(before.Unknown, DispatchChainTally.OverlapsOrdered.Unknown);
        }
        finally
        {
            DispatchChain.OverlapHazardsChecked = saved;
        }
    }

    [Fact]
    public void AMarkedDispatchIsCheckedAgainstEveryDispatchSinceTheLastOrderedOne()
    {
        DispatchMarked(1, [Written], [BufferDescriptor(BufferA, 4, 16)]);
        DispatchMarked(2, [Written], [BufferDescriptor(BufferB, 4, 16)]);
        // Reads what the first wrote, two dispatches back, with no barrier between them either.
        DispatchMarked(3, [Read], [BufferDescriptor(BufferA + 0x10, 4, 4)]);

        Assert.Equal([true, true, false], _host.DispatchOverlaps);
    }

    // Two marked dispatches writing the same buffer bytes, neither reading them, leave whichever lands last on the
    // console too: they stay beside each other (SHARPEMU_DISPATCH_OVERLAP_WRITES_APART=0 orders them). A write over
    // bytes the group read is ordered either way.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void MarkedWritesOfTheSameBytesThatNeitherReadsStayBesideEachOther(bool apart)
    {
        var saved = DispatchChain.OverlapWritesApart;
        DispatchChain.OverlapWritesApart = apart;
        try
        {
            DispatchMarked(1, [Written], [BufferDescriptor(BufferA, 4, 16)]);
            DispatchMarked(2, [Written], [BufferDescriptor(BufferA + 0x10, 4, 4)]);
            DispatchMarked(3, [Read], [BufferDescriptor(BufferB, 4, 16)]);
            DispatchMarked(4, [Written], [BufferDescriptor(BufferB + 0x10, 4, 4)]);

            Assert.Equal([true, apart, true, false], _host.DispatchOverlaps);
        }
        finally
        {
            DispatchChain.OverlapWritesApart = saved;
        }
    }

    [Fact]
    public void AMarkedDispatchWhoseAccessesAreNotAllKnownIsOrdered()
    {
        var before = DispatchChainTally.OverlapsOrdered;
        DispatchMarked(1, [Written], [BufferDescriptor(BufferA, 4, 16)]);
        var heap = new ShaderProgramInfo
        {
            Stage = ShaderStageKind.Compute,
            Buffers = [Written],
            Bindings = new BindingLayout { UsesBindlessHeap = true },
        };
        _pipelines.Compute = ComputeProgram(Stage(heap, [BufferDescriptor(BufferB, 4, 16)]));
        _executor.Dispatch(2, Banks(), 1, 1, 1, 0x41 | RenderExecutor.DispatchMayOverlapBit);

        Assert.Equal([true, false], _host.DispatchOverlaps);
        Assert.Equal(before.Unknown + 1, DispatchChainTally.OverlapsOrdered.Unknown);
    }

    // A marked dispatch sampling an image beside one that wrote an image meets it only where their data does (the span
    // the request builders resolve); SHARPEMU_DISPATCH_OVERLAP_IMAGE_SPANS=0 makes any written image meet every other.
    [Theory]
    [InlineData(false, true, true)]
    [InlineData(true, true, false)]
    [InlineData(false, false, false)]
    public void AMarkedDispatchMeetsAnImageWrittenBesideItOnlyWhereTheirDataMeets(bool sameImage, bool spans, bool overlaps)
    {
        var saved = DispatchChain.OverlapImageSpans;
        DispatchChain.OverlapImageSpans = spans;
        try
        {
            const ulong written = 0x1_7000_0000;
            DispatchImage(1, written, store: true);
            DispatchImage(2, sameImage ? written : written + 0x100_0000, store: false);

            Assert.Equal([true, overlaps], _host.DispatchOverlaps);
        }
        finally
        {
            DispatchChain.OverlapImageSpans = saved;
        }
    }

    // Two marked dispatches storing to the same image, neither reading it (tile classes of one lighting pass), stay beside
    // each other; one that also loads the image it stores to is ordered after a store to it.
    [Theory]
    [InlineData(false, true, true)]
    [InlineData(true, true, false)]
    [InlineData(false, false, false)]
    public void MarkedStoresToOneImageThatNeitherReadsStayBesideEachOther(bool secondLoads, bool apart, bool overlaps)
    {
        var saved = DispatchChain.OverlapWritesApart;
        DispatchChain.OverlapWritesApart = apart;
        try
        {
            const ulong written = 0x1_7000_0000;
            DispatchImage(1, written, store: true);
            DispatchImage(2, written, store: true, load: secondLoads);

            Assert.Equal([true, overlaps], _host.DispatchOverlaps);
        }
        finally
        {
            DispatchChain.OverlapWritesApart = saved;
        }
    }

    private void DispatchImage(ulong submitId, ulong address, bool store, bool load = false)
    {
        var image = new ImageResource
        {
            Read = !store || load,
            ResourceClass = store ? SharpEmu.ShaderCompiler.Resources.ImageResourceClass.Storage : SharpEmu.ShaderCompiler.Resources.ImageResourceClass.Sampled,
            NumericClass = ImageNumericClass.Float,
            Dimension = ImageDimension.Dim2D,
            Written = store,
        };
        var program = new ShaderProgramInfo
        {
            Stage = ShaderStageKind.Compute,
            Images = [new ImageResourceInfo(store ? SharpEmu.Libs.Gpu.Rendering.ImageResourceClass.Storage : SharpEmu.Libs.Gpu.Rendering.ImageResourceClass.Sampled, store)],
            Resources = new SpecializedResourceInfo { Info = new ShaderResourceInfo { Images = [image] } },
        };
        var words = RegisterWords.Texture(address, GuestPixelFormat.Bits11_11_10Float, 256, 256, tile: GuestTileMode.RenderTarget);
        _pipelines.Compute = ComputeProgram(new ShaderStageResources(program, new ResourceSnapshot { Images = [words] }));
        _executor.Dispatch(submitId, Banks(), 1, 1, 1, 0x41 | RenderExecutor.DispatchMayOverlapBit);
    }

    // Two marked dispatches bumping the same counters, each through atomics alone, run beside each other as on the
    // console (SHARPEMU_DISPATCH_OVERLAP_ATOMICS_APART=0 orders them). A load of the counters after their atomics, or a
    // buffer the shader also loads or stores, is ordered either way.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void MarkedAtomicsOnTheSameBytesStayBesideEachOther(bool apart)
    {
        var saved = DispatchChain.OverlapAtomicsApart;
        DispatchChain.OverlapAtomicsApart = apart;
        try
        {
            DispatchMarked(1, [AtomicOnly], [BufferDescriptor(BufferA, 4, 16)]);
            DispatchMarked(2, [AtomicOnly], [BufferDescriptor(BufferA + 0x10, 4, 4)]);
            DispatchMarked(3, [Read], [BufferDescriptor(BufferA, 4, 16)]);
            DispatchMarked(4, [AtomicAndPlain], [BufferDescriptor(BufferB, 4, 16)]);
            DispatchMarked(5, [AtomicAndPlain], [BufferDescriptor(BufferB + 0x10, 4, 4)]);

            Assert.Equal([true, apart, false, true, false], _host.DispatchOverlaps);
        }
        finally
        {
            DispatchChain.OverlapAtomicsApart = saved;
        }
    }

    [Fact]
    public void AnUndeferredHostKeepsTheAccessBarrierAfterEveryDispatch()
    {
        _host.DeferBarriers = false;
        Dispatch(1, [Written], [BufferDescriptor(BufferA, 4, 16)]);
        Dispatch(2, [Written], [BufferDescriptor(BufferB, 4, 16)]);

        Assert.Equal(2, HazardBarriers);
        Assert.Equal(2, _host.Calls.Count(call => call == "access_barrier"));
    }
}
