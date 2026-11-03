// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Held label copies landed by one compute dispatch instead of one blit each.
/// Silent Hill's first room holds about 19,000 eight-byte copies a frame (occlusion dumps,
/// end-of-pipe labels, command-processor writes), up to 16,887 at one boundary, and a
/// copyFromBuffer per copy was about a tenth of the GPU worker's time, most of it the driver
/// checking each blit against the ones before it.
/// The dispatch copies 32-bit words. A word several copies write takes the last copy's
/// source, which is what the blits left in it, so the order of the copies is kept without
/// running them in order. A batch whose words do not divide into 32-bit units, or in which a
/// buffer is both copied from and copied into, keeps the blits.
/// SHARPEMU_METAL_V2_LABEL_SCATTER=0 keeps the blits for every batch.
/// </summary>
internal sealed partial class MetalRenderHost
{
    internal bool LabelScatterEnabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_LABEL_SCATTER") != "0";

    // Fewer copies than this cost less as blits than a compute encoder does.
    private const int LabelScatterMinCopies = 16;
    // A command buffer's held copies read the result buffer of every render pass that dumped occlusion counts
    // into them (one per pass, up to NativeOcclusionBatchBudget of them), so a batch spans far more than a few
    // buffers; declined, every copy of it is a blit of its own.
    private const int LabelScatterMaxBuffers = 512;
    private const int LabelScatterThreads = 64;

    private const string LabelScatterSource = """
        #include <metal_stdlib>
        using namespace metal;
        struct sharpemu_label_word { device const uint* source; device uint* destination; };
        kernel void sharpemu_label_scatter(device const sharpemu_label_word* words [[buffer(0)]],
            constant uint& count [[buffer(1)]], uint index [[thread_position_in_grid]]) {
            if (index >= count) return;
            *words[index].destination = *words[index].source;
        }
        """;

    private nint _labelScatterPipeline;
    private readonly Dictionary<ulong, int> _labelScatterWords = [];
    private readonly nint[] _labelScatterBuffers = new nint[LabelScatterMaxBuffers];
    private readonly ulong[] _labelScatterAddresses = new ulong[LabelScatterMaxBuffers];
    private LabelScatterWord[] _labelScatterList = new LabelScatterWord[1024];

    // sharpemu_label_word: the device addresses of a source word and the word it lands in.
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    private struct LabelScatterWord
    {
        public ulong Source;
        public ulong Destination;
    }
    private long _labelScatterBatches, _labelScatterCopies, _labelScatterDeclined;

    internal (long Batches, long Copies, long Declined) LabelScatterTotals =>
        (_labelScatterBatches, _labelScatterCopies, _labelScatterDeclined);

    private unsafe bool TryEncodeLabelScatter(nint commandBuffer, ReadOnlySpan<DeferredLabelCopy> copies)
    {
        if (!LabelScatterEnabled || copies.Length < LabelScatterMinCopies)
        {
            return false;
        }

        // The buffers copied from come first, then those copied into; none may be both.
        var sources = 0;
        var sourceHint = 0;
        foreach (ref readonly var copy in copies)
        {
            if (copy.Bytes <= 0 || (copy.Bytes & 3) != 0 || ((copy.StagingOffset | copy.DestinationOffset) & 3) != 0 ||
                copy.StagingBuffer == 0 || copy.DestinationBuffer == 0 ||
                !NoteLabelScatterBuffer(copy.StagingBuffer, 0, ref sources, ref sourceHint))
            {
                _labelScatterDeclined++;
                return false;
            }
        }

        var buffers = sources;
        var destinationHint = sources;
        foreach (ref readonly var copy in copies)
        {
            if (FindLabelScatterBuffer(copy.DestinationBuffer, 0, sources, ref sourceHint) >= 0 ||
                !NoteLabelScatterBuffer(copy.DestinationBuffer, sources, ref buffers, ref destinationHint))
            {
                _labelScatterDeclined++;
                return false;
            }
        }

        for (var index = 0; index < buffers; index++)
        {
            _labelScatterAddresses[index] = MetalNative.SendGpuAddress(_labelScatterBuffers[index], SelLabelScatterGpuAddress);
            if (_labelScatterAddresses[index] == 0)
            {
                _labelScatterDeclined++;
                return false;
            }
        }

        // One entry per destination word, holding the source of the last copy into it.
        _labelScatterWords.Clear();
        var count = 0;
        sourceHint = 0;
        destinationHint = sources;
        foreach (ref readonly var copy in copies)
        {
            var source = _labelScatterAddresses[FindLabelScatterBuffer(copy.StagingBuffer, 0, sources, ref sourceHint)] + copy.StagingOffset;
            var destination = _labelScatterAddresses[FindLabelScatterBuffer(copy.DestinationBuffer, sources, buffers - sources, ref destinationHint)] +
                copy.DestinationOffset;
            for (var offset = 0UL; offset < (ulong)copy.Bytes; offset += sizeof(uint))
            {
                if (_labelScatterWords.TryGetValue(destination + offset, out var existing))
                {
                    _labelScatterList[existing].Source = source + offset;
                    continue;
                }

                if (count == _labelScatterList.Length)
                {
                    Array.Resize(ref _labelScatterList, count * 2);
                }

                _labelScatterWords.Add(destination + offset, count);
                _labelScatterList[count++] = new LabelScatterWord { Source = source + offset, Destination = destination + offset };
            }
        }

        var listBytes = (ulong)count * 16;
        var list = _guestMemory.Allocate(listBytes, 16, out var payload);
        if (!list.IsValid)
        {
            _labelScatterDeclined++;
            return false;
        }

        System.Runtime.InteropServices.MemoryMarshal.AsBytes(_labelScatterList.AsSpan(0, count)).CopyTo(payload);
        var pipeline = LabelScatterPipeline();
        EncodeLabelBarrier(commandBuffer, copies);
        var encoder = RetainEncoder(MetalNative.Send(commandBuffer, SelComputeEncoder));
        if (encoder == 0)
        {
            throw Fatal("A Metal compute encoder could not be created for held label copies.");
        }

        WaitForPreviousEncoder(encoder, render: false);
        MetalNative.SendVoid(encoder, SelOcclusionPipeline, pipeline);
        MetalNative.SendSetBuffer(encoder, SelOcclusionBuffer, list.Buffer, (nuint)list.Offset, 0);
        var words = (uint)count;
        MetalNative.SendSetBytes(encoder, SelOcclusionBytes, (nint)(&words), sizeof(uint), 1);
        fixed (nint* resources = _labelScatterBuffers)
        {
            if (sources != 0)
            {
                MetalNative.SendUseResources(encoder, SelUseResourcesUsage, resources, (nuint)sources, Mtl.ResourceUsageRead);
            }

            MetalNative.SendUseResources(encoder, SelUseResourcesUsage, resources + sources, (nuint)(buffers - sources),
                Mtl.ResourceUsageWrite);
        }

        MetalNative.SendDispatch(encoder, SelOcclusionDispatch,
            new MtlSize { Width = (nuint)((count + LabelScatterThreads - 1) / LabelScatterThreads), Height = 1, Depth = 1 },
            new MtlSize { Width = LabelScatterThreads, Height = 1, Depth = 1 });
        EndAndReleaseEncoder(encoder, render: false);
        _encodersInBuffer++;
        _labelScatterBatches++;
        _labelScatterCopies += copies.Length;
        return true;
    }

    // Consecutive copies mostly share their buffers (an occlusion dump's blocks, a pass's dumps), so the
    // buffer found last answers first.
    private int FindLabelScatterBuffer(nint buffer, int first, int count, ref int hint)
    {
        if ((uint)(hint - first) < (uint)count && _labelScatterBuffers[hint] == buffer)
        {
            return hint;
        }

        var index = Array.IndexOf(_labelScatterBuffers, buffer, first, count);
        if (index >= 0)
        {
            hint = index;
        }

        return index;
    }

    private bool NoteLabelScatterBuffer(nint buffer, int first, ref int count, ref int hint)
    {
        if (FindLabelScatterBuffer(buffer, first, count - first, ref hint) >= 0)
        {
            return true;
        }

        if (count == LabelScatterMaxBuffers)
        {
            return false;
        }

        _labelScatterBuffers[count++] = buffer;
        return true;
    }

    private static readonly nint SelLabelScatterGpuAddress = MetalNative.Selector("gpuAddress");

    private nint LabelScatterPipeline()
    {
        if (_labelScatterPipeline != 0)
        {
            return _labelScatterPipeline;
        }

        var options = MetalNative.Send(MetalNative.Send(MetalNative.Class("MTLCompileOptions"), MetalNative.Selector("alloc")), MetalNative.Selector("init"));
        MetalNative.ApplyMslCompileOptions(options);
        nint error = 0;
        var library = MetalNative.Send(_device, MetalNative.Selector("newLibraryWithSource:options:error:"),
            MetalNative.NsString(LabelScatterSource), options, ref error);
        MetalNative.SendVoid(options, SelRelease);
        if (library == 0) throw Fatal($"The label scatter kernel did not compile: {MetalNative.DescribeError(error)}.");
        var function = MetalNative.Send(library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString("sharpemu_label_scatter"));
        MetalNative.SendVoid(library, SelRelease);
        if (function == 0) throw Fatal("The label scatter function is missing.");
        _labelScatterPipeline = MetalNative.Send(_device, MetalNative.Selector("newComputePipelineStateWithFunction:error:"), function, ref error);
        MetalNative.SendVoid(function, SelRelease);
        if (_labelScatterPipeline == 0) throw Fatal($"The label scatter pipeline did not build: {MetalNative.DescribeError(error)}.");
        return _labelScatterPipeline;
    }

    private void DisposeLabelScatter()
    {
        if (_labelScatterPipeline != 0) MetalNative.SendVoid(_labelScatterPipeline, SelRelease);
        _labelScatterPipeline = 0;
    }
}
