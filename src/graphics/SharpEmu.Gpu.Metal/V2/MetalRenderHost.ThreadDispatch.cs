// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Indirect dispatches in thread units, converted to group counts on the GPU.
/// </summary>
/// <remarks>
/// <para>
/// With DISPATCH_INITIATOR.USE_THREAD_DIMENSIONS the guest record holds thread counts, which
/// Metal's indirect dispatch cannot take. Reading them on the CPU waits for the GPU work that
/// wrote them (GTA V's ray-traced passes size theirs from a compute pass the same frame, a
/// ~40 ms wait a frame). Instead a one-thread kernel on the dispatch's own (serial) compute
/// encoder reads the record and writes the group counts - rounded up per axis as
/// RenderExecutor.GroupsFromThreads, placed on the pipeline's physical axes - for
/// dispatchThreadgroupsWithIndirectBuffer, and the thread counts as the program's thread
/// limits, which mask the threads of the last group past them.
/// </para>
/// <para>
/// The limits are in the stage's push data or shader-data block, which CommitBindings gives
/// such a dispatch a region of its own: one shared by content with other bindings would be
/// changed under them. SHARPEMU_METAL_V2_THREAD_INDIRECT_GPU=0 reads the counts on the CPU.
/// </para>
/// </remarks>
internal sealed partial class MetalRenderHost
{
    internal static readonly bool ThreadIndirectOnGpu =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_THREAD_INDIRECT_GPU") != "0";

    private const string ThreadDispatchFunction = "sharpemu_thread_dispatch";

    // threads: the guest record's dword-aligned base; groups: three group counts; limits: the
    // three thread limits of the bound program, written when write_limits is set.
    private const string ThreadDispatchSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct sharpemu_thread_dispatch_params
        {
            uint local_size[3];
            uint physical_axis[3];
            uint byte_shift;
            uint write_limits;
        };

        kernel void sharpemu_thread_dispatch(device const uchar* threads [[buffer(0)]],
                                             device uint* groups [[buffer(1)]],
                                             device uint* limits [[buffer(2)]],
                                             constant sharpemu_thread_dispatch_params& p [[buffer(3)]])
        {
            uint counts[3];
            for (uint axis = 0; axis < 3; axis++)
            {
                uint at = p.byte_shift + axis * 4u;
                counts[axis] = uint(threads[at]) | (uint(threads[at + 1]) << 8) |
                    (uint(threads[at + 2]) << 16) | (uint(threads[at + 3]) << 24);
            }

            for (uint axis = 0; axis < 3; axis++)
            {
                uint size = max(p.local_size[axis], 1u);
                uint count = counts[axis];
                groups[p.physical_axis[axis]] = count == 0u ? 0u : (count + size - 1u) / size;
            }

            if (p.write_limits != 0u)
            {
                limits[0] = counts[0];
                limits[1] = counts[1];
                limits[2] = counts[2];
            }
        }
        """;

    private nint _threadDispatchPipeline;

    // Where CommitBindings put the bound compute program's thread limits for the GPU to write.
    private MetalGuestRange _threadLimitsTarget;

    private long _threadIndirectGpuDispatches;

    public bool ResolvesThreadIndirectDispatchOnGpu => ThreadIndirectOnGpu;

    public bool DispatchesThreadsIndirectOnGpu => ThreadIndirectOnGpu;

    /// <summary>Thread-unit indirect dispatches converted on the GPU; reported and reset.</summary>
    internal long TakeThreadIndirectDispatches() => Interlocked.Exchange(ref _threadIndirectGpuDispatches, 0);

    public unsafe bool TryDispatchThreadsIndirect(
        ulong argumentsAddress, uint localX, uint localY, uint localZ, ReadOnlySpan<int> axisOrder)
    {
        if (axisOrder.Length != 3)
        {
            return false;
        }

        var shift = argumentsAddress & 0x3;
        var alignedAddress = argumentsAddress - shift;
        var range = _guestMemory.Resolve(alignedAddress, shift + (3 * sizeof(uint)));
        if (!range.IsValid)
        {
            return false;
        }

        // The GPU reads the thread counts where the dispatch sits in the stream.
        OrderAfterDeferredCopies(argumentsAddress, 3 * sizeof(uint));
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        var encoder = EnsureComputeEncoder();
        var entry = _pendingComputePipeline ??
            throw Fatal("A thread-unit indirect dispatch has no bound Metal compute pipeline state.");
        var limits = _threadLimitsTarget;
        _threadLimitsTarget = default;

        var groups = _guestMemory.Allocate(4 * sizeof(uint), 16, out var groupSpan);
        groupSpan.Clear();
        var parameters = stackalloc uint[8];
        parameters[0] = localX;
        parameters[1] = localY;
        parameters[2] = localZ;
        parameters[3] = (uint)axisOrder[0];
        parameters[4] = (uint)axisOrder[1];
        parameters[5] = (uint)axisOrder[2];
        parameters[6] = (uint)shift;
        parameters[7] = limits.IsValid ? 1u : 0u;

        MetalNative.SendVoid(encoder, SelSetComputePipelineState, ThreadDispatchPipeline());
        _boundComputePipeline = _threadDispatchPipeline;
        MetalNative.SendSetBuffer(encoder, SelSetComputeBuffer, range.Buffer, (nuint)range.Offset, 0);
        MetalNative.SendSetBuffer(encoder, SelSetComputeBuffer, groups.Buffer, (nuint)groups.Offset, 1);
        MetalNative.SendSetBuffer(
            encoder,
            SelSetComputeBuffer,
            limits.IsValid ? limits.Buffer : groups.Buffer,
            (nuint)(limits.IsValid ? limits.Offset : groups.Offset),
            2);
        MetalNative.SendSetBytes(encoder, SelHullSetBytes, (nint)parameters, 8 * sizeof(uint), 3);
        MetalNative.SendDispatch(
            encoder,
            SelDispatchThreadgroups,
            new MtlSize { Width = 1, Height = 1, Depth = 1 },
            new MtlSize { Width = 1, Height = 1, Depth = 1 });

        // The encoder is serial: the dispatch below starts once the kernel's stores are done.
        // Its slots 0-3 are rebound by the stage bindings.
        EnsureComputePipelineState(encoder, entry);
        ApplyStageBindings(graphics: false);
        MetalV2Native.SendDispatchIndirect(
            encoder,
            SelDispatchIndirect,
            groups.Buffer,
            (nuint)groups.Offset,
            entry.ThreadsPerThreadgroup);
        CountDraw(started);
        Interlocked.Increment(ref _threadIndirectGpuDispatches);
        ProfileWork(
            entry,
            "dispatch-indirect-threads",
            WorkShape.Threads(entry.ThreadsPerThreadgroup));
        return true;
    }

    private nint ThreadDispatchPipeline()
    {
        if (_threadDispatchPipeline != 0)
        {
            return _threadDispatchPipeline;
        }

        var options = MetalNative.Send(MetalNative.Send(MetalNative.Class("MTLCompileOptions"), MetalNative.Selector("alloc")), MetalNative.Selector("init"));
        MetalNative.ApplyMslCompileOptions(options);
        nint error = 0;
        var library = MetalNative.Send(_device, MetalNative.Selector("newLibraryWithSource:options:error:"),
            MetalNative.NsString(ThreadDispatchSource), options, ref error);
        MetalNative.SendVoid(options, SelRelease);
        if (library == 0) throw Fatal($"The thread dispatch kernel did not compile: {MetalNative.DescribeError(error)}.");
        var function = MetalNative.Send(library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString(ThreadDispatchFunction));
        MetalNative.SendVoid(library, SelRelease);
        if (function == 0) throw Fatal("The thread dispatch function is missing.");
        _threadDispatchPipeline = MetalNative.Send(_device, MetalNative.Selector("newComputePipelineStateWithFunction:error:"), function, ref error);
        MetalNative.SendVoid(function, SelRelease);
        if (_threadDispatchPipeline == 0) throw Fatal($"The thread dispatch pipeline did not build: {MetalNative.DescribeError(error)}.");
        return _threadDispatchPipeline;
    }

    private void DisposeThreadDispatch()
    {
        if (_threadDispatchPipeline != 0) MetalNative.SendVoid(_threadDispatchPipeline, SelRelease);
        _threadDispatchPipeline = 0;
    }
}
