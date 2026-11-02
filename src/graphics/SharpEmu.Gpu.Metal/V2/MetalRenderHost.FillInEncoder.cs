// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

// A guest fill recorded while a compute encoder is open, written by a dispatch in that encoder instead of a blit
// encoder of its own. Black Myth: Wukong fills a buffer between most of its dispatches (~37 of its ~100 compute
// encoders a frame ended at a fill), and each fill cost two encoder boundaries, each a fence wait the GPU sat idle
// through (~250 us). The encoder runs its dispatches in order (serial dispatch); the barriers on either side say so for
// the untracked guest windows, as for a transfer shared with the encoder (TryShareComputeEncoder).
internal sealed partial class MetalRenderHost
{
    // SHARPEMU_METAL_V2_FILL_IN_ENCODER=0 ends the open compute encoder for every guest fill, as before.
    internal bool FillsShareCompute { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_FILL_IN_ENCODER") != "0";

    private const int FillThreads = 256;

    private const string FillWordsSource = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void sharpemu_fill_words(device uint* words [[buffer(0)]], constant uint2& fill [[buffer(1)]],
            uint index [[thread_position_in_grid]]) {
            if (index < fill.y) words[index] = fill.x;
        }
        """;

    private nint _fillWordsPipeline;
    private long _fillsInEncoder;

    internal long FillsInEncoder => Interlocked.Read(ref _fillsInEncoder);

    // Records the fill into the open compute encoder; false leaves it to a blit encoder of its own. Not when what lands
    // as the encoder ends (a held label copy, a staged store written back) covers these bytes: that would land after
    // the fill, so the fill takes its place after the boundary.
    private bool TryFillInComputeEncoder(ulong address, ulong size, uint value, in MetalGuestRange range)
    {
        if (!FillsShareCompute || (size & 3) != 0 || (range.Offset & 3) != 0 || size / sizeof(uint) is 0 or > uint.MaxValue ||
            DeferredLabelCopiesTouch(address, size) ||
            (_stagedStorageCopies is { Count: > 0 } staged && staged.HasRecordedStoresOver(address, size)) ||
            ComputeEncoderForTransfer() is not (var encoder and not 0))
        {
            return false;
        }

        _transferCauses[(int)MetalTransferCause.GuestFill]++;
        EncodeFillWords(encoder, range.Buffer, range.Offset, size, value);
        _fillsInEncoder++;
        if (PassProfileEnabled)
        {
            PassProfileNoteWork(nameof(MetalTransferCause.GuestFill), "transfer");
        }

        return true;
    }

    // The open compute encoder, or with SHARPEMU_METAL_V2_COPY_IN_ENCODER=1 one opened for a transfer when no encoder is
    // open at all (as a blit encoder would be; the clears still waiting keep waiting, as for a blit). 0 otherwise.
    private nint ComputeEncoderForTransfer()
    {
        if (_computeEncoder != 0)
        {
            return _computeEncoder;
        }

        return CopiesShareCompute && _renderEncoder == 0 ? EnsureComputeEncoder(materializeClears: false) : 0;
    }

    // A fill of size bytes (a multiple of four) at buffer + offset, between memory barriers in the compute encoder.
    private unsafe void EncodeFillWords(nint encoder, nint buffer, ulong offset, ulong size, uint value)
    {
        var words = size / sizeof(uint);
        NoteRecordedWork();
        MetalV2Native.SendMemoryBarrierScope(encoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
        MetalNative.SendVoid(encoder, SelOcclusionPipeline, FillWordsPipeline());
        MetalNative.SendSetBuffer(encoder, SelOcclusionBuffer, buffer, (nuint)offset, 0);
        var fill = stackalloc uint[2];
        fill[0] = value;
        fill[1] = (uint)words;
        MetalNative.SendSetBytes(encoder, SelOcclusionBytes, (nint)fill, 2 * sizeof(uint), 1);
        MetalNative.SendDispatch(encoder, SelOcclusionDispatch,
            new MtlSize { Width = (nuint)((words + FillThreads - 1) / FillThreads), Height = 1, Depth = 1 },
            new MtlSize { Width = FillThreads, Height = 1, Depth = 1 });
        MetalV2Native.SendMemoryBarrierScope(encoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
        // The next guest dispatch binds its own pipeline again; its buffers are bound per dispatch.
        _boundComputePipeline = 0;
    }

    // SHARPEMU_METAL_V2_COPY_IN_ENCODER (on; =0 off) does the same for a guest copy (DMA_DATA memory to memory). Ghost of
    // Tsushima copies ~180 times a frame, mostly right after a dispatch; as blits that was a compute -> blit encoder
    // boundary each (the GPU idles ~50 us at one) and ~12% of the GPU timeline went to blit boundaries.
    internal bool CopiesShareCompute { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_COPY_IN_ENCODER") != "0";

    private const string CopyWordsSource = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void sharpemu_copy_words(device const uint* source [[buffer(0)]], device uint* destination [[buffer(1)]],
            constant uint& count [[buffer(2)]], uint index [[thread_position_in_grid]]) {
            if (index < count) destination[index] = source[index];
        }
        """;

    private nint _copyWordsPipeline;
    private long _copiesInEncoder;

    internal long CopiesInEncoder => Interlocked.Read(ref _copiesInEncoder);

    // Records the copy into the open compute encoder; false leaves it to a blit encoder of its own. Not for bytes that
    // overlap in one buffer (the dispatch does not order its threads), and not when what lands as the encoder ends
    // covers the source (the copy would read the bytes before it) or the destination (it would land over the copy).
    private bool TryCopyInComputeEncoder(
        ulong destination, ulong source, ulong size, in MetalGuestRange target, in MetalGuestRange origin)
    {
        if (!CopiesShareCompute || _renderEncoder != 0 ||
            !WordCopyFits(target.Buffer, target.Offset, origin.Buffer, origin.Offset, size) ||
            DeferredLabelCopiesTouch(destination, size) || DeferredLabelCopiesTouch(source, size) ||
            (_stagedStorageCopies is { Count: > 0 } staged &&
                (staged.HasRecordedStoresOver(destination, size) || staged.HasRecordedStoresOver(source, size))) ||
            ComputeEncoderForTransfer() is not (var encoder and not 0))
        {
            return false;
        }

        _transferCauses[(int)MetalTransferCause.GuestCopy]++;
        EncodeCopyWords(encoder, origin.Buffer, origin.Offset, target.Buffer, target.Offset, size);
        return true;
    }

    // Whether a word-copy dispatch can do this copy: whole aligned words, and no overlap in one buffer (the dispatch
    // does not order its threads).
    private static bool WordCopyFits(nint destination, ulong destinationOffset, nint source, ulong sourceOffset, ulong size) =>
        (size & 3) == 0 && (destinationOffset & 3) == 0 && (sourceOffset & 3) == 0 && size / sizeof(uint) is > 0 and <= uint.MaxValue &&
        !(destination == source && destinationOffset < sourceOffset + size && sourceOffset < destinationOffset + size);

    private unsafe void EncodeCopyWords(nint encoder, nint source, ulong sourceOffset, nint destination, ulong destinationOffset, ulong size)
    {
        var count = (uint)(size / sizeof(uint));
        NoteRecordedWork();
        MetalV2Native.SendMemoryBarrierScope(encoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
        MetalNative.SendVoid(encoder, SelOcclusionPipeline, CopyWordsPipeline());
        MetalNative.SendSetBuffer(encoder, SelOcclusionBuffer, source, (nuint)sourceOffset, 0);
        MetalNative.SendSetBuffer(encoder, SelOcclusionBuffer, destination, (nuint)destinationOffset, 1);
        MetalNative.SendSetBytes(encoder, SelOcclusionBytes, (nint)(&count), sizeof(uint), 2);
        MetalNative.SendDispatch(encoder, SelOcclusionDispatch,
            new MtlSize { Width = (nuint)((count + FillThreads - 1) / FillThreads), Height = 1, Depth = 1 },
            new MtlSize { Width = FillThreads, Height = 1, Depth = 1 });
        MetalV2Native.SendMemoryBarrierScope(encoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
        _boundComputePipeline = 0;
        _copiesInEncoder++;
        if (PassProfileEnabled)
        {
            PassProfileNoteWork(nameof(MetalTransferCause.GuestCopy), "transfer");
        }
    }

    private nint CopyWordsPipeline() =>
        _copyWordsPipeline != 0 ? _copyWordsPipeline : _copyWordsPipeline = BuildWordsPipeline(CopyWordsSource, "sharpemu_copy_words");

    private nint FillWordsPipeline()
    {
        if (_fillWordsPipeline != 0)
        {
            return _fillWordsPipeline;
        }

        var options = MetalNative.Send(MetalNative.Send(MetalNative.Class("MTLCompileOptions"), MetalNative.Selector("alloc")), MetalNative.Selector("init"));
        MetalNative.ApplyMslCompileOptions(options);
        nint error = 0;
        var library = MetalNative.Send(_device, MetalNative.Selector("newLibraryWithSource:options:error:"),
            MetalNative.NsString(FillWordsSource), options, ref error);
        MetalNative.SendVoid(options, SelRelease);
        if (library == 0) throw Fatal($"The fill kernel did not compile: {MetalNative.DescribeError(error)}.");
        var function = MetalNative.Send(library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString("sharpemu_fill_words"));
        MetalNative.SendVoid(library, SelRelease);
        if (function == 0) throw Fatal("The fill function is missing.");
        _fillWordsPipeline = MetalNative.Send(_device, MetalNative.Selector("newComputePipelineStateWithFunction:error:"), function, ref error);
        MetalNative.SendVoid(function, SelRelease);
        if (_fillWordsPipeline == 0) throw Fatal($"The fill pipeline did not build: {MetalNative.DescribeError(error)}.");
        return _fillWordsPipeline;
    }

    private nint BuildWordsPipeline(string source, string name)
    {
        var options = MetalNative.Send(MetalNative.Send(MetalNative.Class("MTLCompileOptions"), MetalNative.Selector("alloc")), MetalNative.Selector("init"));
        MetalNative.ApplyMslCompileOptions(options);
        nint error = 0;
        var library = MetalNative.Send(_device, MetalNative.Selector("newLibraryWithSource:options:error:"),
            MetalNative.NsString(source), options, ref error);
        MetalNative.SendVoid(options, SelRelease);
        if (library == 0) throw Fatal($"The {name} kernel did not compile: {MetalNative.DescribeError(error)}.");
        var function = MetalNative.Send(library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString(name));
        MetalNative.SendVoid(library, SelRelease);
        if (function == 0) throw Fatal($"The {name} function is missing.");
        var pipeline = MetalNative.Send(_device, MetalNative.Selector("newComputePipelineStateWithFunction:error:"), function, ref error);
        MetalNative.SendVoid(function, SelRelease);
        if (pipeline == 0) throw Fatal($"The {name} pipeline did not build: {MetalNative.DescribeError(error)}.");
        return pipeline;
    }

    private void DisposeFillWords()
    {
        if (_fillWordsPipeline != 0) MetalNative.SendVoid(_fillWordsPipeline, SelRelease);
        _fillWordsPipeline = 0;
        if (_copyWordsPipeline != 0) MetalNative.SendVoid(_copyWordsPipeline, SelRelease);
        _copyWordsPipeline = 0;
    }
}
