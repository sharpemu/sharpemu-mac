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
    private unsafe bool TryFillInComputeEncoder(ulong address, ulong size, uint value, in MetalGuestRange range)
    {
        var words = size / sizeof(uint);
        if (!FillsShareCompute || _computeEncoder == 0 || (size & 3) != 0 || (range.Offset & 3) != 0 ||
            words == 0 || words > uint.MaxValue ||
            DeferredLabelCopiesTouch(address, size) ||
            (_stagedStorageCopies is { Count: > 0 } staged && staged.HasRecordedStoresOver(address, size)))
        {
            return false;
        }

        _transferCauses[(int)MetalTransferCause.GuestFill]++;
        NoteRecordedWork();
        var encoder = _computeEncoder;
        MetalV2Native.SendMemoryBarrierScope(encoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
        MetalNative.SendVoid(encoder, SelOcclusionPipeline, FillWordsPipeline());
        MetalNative.SendSetBuffer(encoder, SelOcclusionBuffer, range.Buffer, (nuint)range.Offset, 0);
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
        _fillsInEncoder++;
        if (PassProfileEnabled)
        {
            PassProfileNoteWork(nameof(MetalTransferCause.GuestFill), "transfer");
        }

        return true;
    }

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

    private void DisposeFillWords()
    {
        if (_fillWordsPipeline != 0) MetalNative.SendVoid(_fillWordsPipeline, SelRelease);
        _fillWordsPipeline = 0;
    }
}
