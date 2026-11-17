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

        if (!TryBuildLabelScatter(copies, out var list, out var sources, out var buffers, out var count))
        {
            _labelScatterDeclined++;
            return false;
        }

        EncodeLabelBarrier(commandBuffer, copies);
        var encoder = RetainEncoder(PassProfileEnabled
            ? PassProfileOpenEncoder(commandBuffer, MetalProfiledEncoder.HostCompute, $"label-scatter n={copies.Length}")
            : MetalNative.Send(commandBuffer, SelComputeEncoder));
        if (encoder == 0)
        {
            throw Fatal("A Metal compute encoder could not be created for held label copies.");
        }

        WaitForPreviousEncoder(encoder, render: false);
        EncodeLabelScatterDispatch(encoder, in list, sources, buffers, count);
        EndAndReleaseEncoder(encoder, render: false);
        _encodersInBuffer++;
        _labelScatterBatches++;
        _labelScatterCopies += copies.Length;
        return true;
    }

    // The scatter's word list for these copies, in the ring: one entry per destination word, holding the source of
    // the last copy into it. False when the batch does not fit the scatter (a copy that is not whole aligned words, a
    // buffer both copied from and into, too many buffers, no device address, no ring room).
    private bool TryBuildLabelScatter(
        ReadOnlySpan<DeferredLabelCopy> copies, out MetalGuestRange list, out int sources, out int buffers, out int count)
    {
        list = default;
        count = 0;
        buffers = 0;
        // The buffers copied from come first, then those copied into; none may be both.
        sources = 0;
        var sourceHint = 0;
        foreach (ref readonly var copy in copies)
        {
            if (copy.Bytes <= 0 || (copy.Bytes & 3) != 0 || ((copy.StagingOffset | copy.DestinationOffset) & 3) != 0 ||
                copy.StagingBuffer == 0 || copy.DestinationBuffer == 0 ||
                !NoteLabelScatterBuffer(copy.StagingBuffer, 0, ref sources, ref sourceHint))
            {
                return false;
            }
        }

        buffers = sources;
        var destinationHint = sources;
        foreach (ref readonly var copy in copies)
        {
            if (FindLabelScatterBuffer(copy.DestinationBuffer, 0, sources, ref sourceHint) >= 0 ||
                !NoteLabelScatterBuffer(copy.DestinationBuffer, sources, ref buffers, ref destinationHint))
            {
                return false;
            }
        }

        for (var index = 0; index < buffers; index++)
        {
            _labelScatterAddresses[index] = MetalNative.SendGpuAddress(_labelScatterBuffers[index], SelLabelScatterGpuAddress);
            if (_labelScatterAddresses[index] == 0)
            {
                return false;
            }
        }

        _labelScatterWords.Clear();
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

        list = _guestMemory.Allocate((ulong)count * 16, 16, out var payload);
        if (!list.IsValid)
        {
            return false;
        }

        System.Runtime.InteropServices.MemoryMarshal.AsBytes(_labelScatterList.AsSpan(0, count)).CopyTo(payload);
        return true;
    }

    // The scatter dispatch over a list TryBuildLabelScatter made, into an encoder the caller ordered.
    private unsafe void EncodeLabelScatterDispatch(nint encoder, in MetalGuestRange list, int sources, int buffers, int count)
    {
        MetalNative.SendVoid(encoder, SelOcclusionPipeline, LabelScatterPipeline());
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
    }

    // Held copies none of which the CPU has to see before what came ahead of them (a device-only end-of-pipe label, a
    // command-processor write nothing waited on) land as one scatter dispatch in the open compute encoder instead of
    // ending it for a barrier and a copy encoder of their own. The encoder waited on the encoder fence when it began, so
    // it runs after every encoder before it; a memory barrier on either side of the dispatch orders it after the
    // dispatches already in it and before the ones that follow: the copies land after all the work recorded before them,
    // in device order, and before all the work after, which is what the device-only event and the copy encoder gave.
    // A batch with a CPU-visible copy keeps the shared event, which alone makes the earlier stores visible to the CPU
    // before the label (EncodeLabelBarrier). Demon's Souls ended ~55 compute encoders a frame this way, each a
    // ~120-170 us GPU gap.
    // SHARPEMU_METAL_V2_LABELS_IN_ENCODER=0 lands every batch behind its barrier and copy encoder again; =verify also
    // reads every folded word back at completion and counts the ones that do not hold the folded value (a word that
    // later work in the same submission writes again is reported too).
    private static readonly string? LabelsInEncoderMode = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_LABELS_IN_ENCODER");

    internal bool LabelsInEncoder { get; set; } = LabelsInEncoderMode != "0";

    private static readonly bool LabelsInEncoderVerify = LabelsInEncoderMode == "verify";

    private long _labelsFolded;
    private static long _labelFoldVerifyMismatches;

    /// <summary>Batches of held copies landed inside the open compute encoder.</summary>
    internal long LabelsFolded => Interlocked.Read(ref _labelsFolded);

    /// <summary>SHARPEMU_METAL_V2_LABELS_IN_ENCODER=verify: folded words that did not hold their value at completion.</summary>
    internal static long LabelFoldVerifyMismatches => Interlocked.Read(ref _labelFoldVerifyMismatches);

    // Whether these held copies may land inside a compute encoder. A CPU-visible copy is the normal case of not folding
    // and is not a refusal; the rest are counted (label_in_encoder_refused).
    private bool MayFoldHeldCopies(ReadOnlySpan<DeferredLabelCopy> copies)
    {
        if (!LabelsInEncoder || copies.IsEmpty || !LabelBarrierByCacheAction)
        {
            return false;
        }

        foreach (ref readonly var copy in copies)
        {
            if (copy.CpuVisible)
            {
                return false;
            }
        }

        if (EncoderOrdering != EncoderOrderingMode.Fence)
        {
            // Without the fence chain the encoder may start before the work ahead of it has finished.
            _labelInEncoderRefused[2]++;
            return false;
        }

        if (_oqPendingReductions.Count != 0)
        {
            // An occlusion prefix sum still owed is encoded with no encoder open, ahead of the copies that read it.
            _labelInEncoderRefused[0]++;
            return false;
        }

        if (_stagedStorageCopies is { Count: > 0 } staged && staged.HasRecordedStores)
        {
            // Staged stores are written back when the encoder ends; a held copy over them has to land after them.
            _labelInEncoderRefused[3]++;
            return false;
        }

        return true;
    }

    // Lands every held copy as one dispatch in the open compute encoder; false leaves them held (counted).
    private bool TryLandHeldCopiesInComputeEncoder()
    {
        var copies = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_deferredLabelCopies);
        if (_computeEncoder == 0 || _renderEncoder != 0 || _encodingDeferredLabels || !MayFoldHeldCopies(copies))
        {
            return false;
        }

        if (!TryBuildLabelScatter(copies, out var list, out var sources, out var buffers, out var count))
        {
            _labelInEncoderRefused[1]++;
            return false;
        }

        var cause = _labelLandCause == LabelLandCause.None ? LabelLandCause.Other : _labelLandCause;
        _landDeferredLabels = false;
        _labelLandCause = LabelLandCause.None;
        NoteLabelLanding(copies, cause);
        if (LabelsInEncoderVerify)
        {
            VerifyFoldedCopies(copies);
        }

        var encoder = _computeEncoder;
        NoteRecordedWork();
        MetalV2Native.SendMemoryBarrierScope(encoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
        EncodeLabelScatterDispatch(encoder, in list, sources, buffers, count);
        MetalV2Native.SendMemoryBarrierScope(encoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
        // The next guest dispatch binds its own pipeline again; its buffers are bound per dispatch.
        _boundComputePipeline = 0;
        _labelBarrierKinds[BarrierKindInEncoder]++;
        _labelsFolded++;
        var held = copies.Length;
        ClearHeldLabelCopies();
        if (PassProfileEnabled)
        {
            PassProfileNoteWork($"label-fold n={held}", "transfer");
        }

        return true;
    }

    // SHARPEMU_METAL_V2_CP_WRITES_IN_ENCODER (on; =0 off): a command-processor write that is not causal, recorded while a
    // compute encoder is open, is a scatter dispatch in that encoder at once instead of a held copy: it lands exactly where
    // it was recorded, after all the work before it in device order (the encoder's fence wait and a memory barrier), and a
    // dispatch that reads it then has nothing to land. It may land before an earlier held end-of-pipe label at other bytes,
    // which the console allows (the command processor does not wait for end-of-pipe). It is held as before when a held
    // copy covers its bytes (it must land after that copy: a label still owed into reused command memory), when any
    // command-processor write is held (it must not overtake it), when a staged store recorded over its bytes is still to
    // be written back at the encoder end, or when it is not whole aligned words.
    internal bool CpWritesInEncoder { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_CP_WRITES_IN_ENCODER") != "0";

    // Command-processor writes held now; a later one is held behind them.
    private int _heldCpWrites;
    private long _cpWritesEncodedInEncoder;

    /// <summary>Command-processor writes recorded as a dispatch in the open compute encoder.</summary>
    internal long CpWritesEncodedInEncoder => Interlocked.Read(ref _cpWritesEncodedInEncoder);

    private bool TryEncodeCommandProcessorWritesInEncoder(ReadOnlySpan<DeferredLabelCopy> copies)
    {
        if (!CpWritesInEncoder || _computeEncoder == 0 || _renderEncoder != 0 || _encodingDeferredLabels ||
            _heldCpWrites != 0 || EncoderOrdering != EncoderOrderingMode.Fence)
        {
            return false;
        }

        foreach (ref readonly var copy in copies)
        {
            if (copy.Kind != LabelCopyKind.CpWrite || copy.CpuVisible || (copy.Bytes & 3) != 0 ||
                ((copy.StagingOffset | copy.DestinationOffset) & 3) != 0 ||
                DeferredLabelCopiesTouch(copy.GuestAddress, (ulong)copy.Bytes) ||
                // The same bytes through another guest address (an alias of the window) are the same bytes.
                DeferredLabelCopiesTouch(new MetalGuestRange(copy.DestinationBuffer, copy.DestinationOffset, (ulong)copy.Bytes, 0)) ||
                (_stagedStorageCopies is { Count: > 0 } staged && staged.HasRecordedStoresOver(copy.GuestAddress, (ulong)copy.Bytes)))
            {
                return false;
            }
        }

        if (!TryBuildLabelScatter(copies, out var list, out var sources, out var buffers, out var count))
        {
            return false;
        }

        var encoder = _computeEncoder;
        NoteRecordedWork();
        MetalV2Native.SendMemoryBarrierScope(encoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
        EncodeLabelScatterDispatch(encoder, in list, sources, buffers, count);
        MetalV2Native.SendMemoryBarrierScope(encoder, SelComputeMemoryBarrier, Mtl.BarrierScopeBuffers | Mtl.BarrierScopeTextures);
        _boundComputePipeline = 0;
        _cpWritesEncodedInEncoder += copies.Length;
        _labelCpInEncoder += copies.Length;
        if (PassProfileEnabled)
        {
            PassProfileNoteWork("cp-write", "transfer");
        }

        return true;
    }

    // =verify: what each folded word holds once the submission completes, the last copy into a word winning.
    private unsafe void VerifyFoldedCopies(ReadOnlySpan<DeferredLabelCopy> copies)
    {
        var expected = new Dictionary<ulong, uint>();
        foreach (ref readonly var copy in copies)
        {
            // Only staging the host wrote (an end-of-pipe value, a command-processor write) holds its value now; a GDS or
            // occlusion source is produced by GPU work still ahead, so its words are not checked, and a word it lands
            // over is no longer what an earlier copy wrote.
            var staging = copy.Kind == LabelCopyKind.Other ? null : (byte*)MetalNative.Send(copy.StagingBuffer, SelLabelFoldContents);
            if (staging == null)
            {
                for (var offset = 0; offset < copy.Bytes; offset += sizeof(uint))
                {
                    expected.Remove(copy.GuestAddress + (ulong)offset);
                }

                continue;
            }

            for (var offset = 0; offset < copy.Bytes; offset += sizeof(uint))
            {
                expected[copy.GuestAddress + (ulong)offset] = *(uint*)(staging + copy.StagingOffset + (ulong)offset);
            }
        }

        _completions.Enqueue(SubmissionSerial, () =>
        {
            Span<byte> word = stackalloc byte[sizeof(uint)];
            foreach (var (address, value) in expected)
            {
                if (_memory.TryRead(address, word) && System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(word) != value)
                {
                    Interlocked.Increment(ref _labelFoldVerifyMismatches);
                    MetalV2Diagnostics.WarnOnce("label-fold-verify",
                        $"A label copy folded into a compute encoder did not hold its value at completion: address=0x{address:X16} " +
                        $"expected=0x{value:X8} found=0x{System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(word):X8}.");
                }
            }
        });
    }

    private static readonly nint SelLabelFoldContents = MetalNative.Selector("contents");

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
