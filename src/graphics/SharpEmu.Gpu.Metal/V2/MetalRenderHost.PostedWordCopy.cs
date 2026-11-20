// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;

namespace SharpEmu.Gpu.Metal.V2;

// A bounded word copy kernel (dst[i] = src[i % sourceWords]) posted as a command-processor store at its
// stream position instead of running as a dispatch (IRenderHost.TryPostWordCopy, upstream 61e9ec8f3 and
// 5d4937c46; the Vulkan host posts it in VulkanVideoPresenter.KernelWrites.cs PostWordCopy).
//
// Where it runs: the Metal translation resolves whole dispatches on the worker at their stream position
// (CommandStreamTranslation._defersWholeWork), so the executor asks here with everything recorded before
// the dispatch already recorded and nothing after it. The source is read now, which is what the kernel
// would read only if no recorded GPU work still writes it; the replicated bytes are then recorded as a
// command-processor write (TryEncodeCommandProcessorWrite): an ordered copy from staging where the stream
// has reached, after every write recorded before it, with parse-time reads answered from the
// command-processor overlay until it lands. A store the GPU cannot make (no window, no staging, a label
// still owed by a completion) would land only when the submission completes, after later GPU readers, so
// the copy is refused then and runs as the dispatch it is.
//
// Refusals, in the Vulkan host's order (KernelWritePolicy.Check, then the input read):
//   - unsupported: a null, unaligned, empty, overlapping or overflowing range, or one over 16 MiB
//     (GuestBufferCache.MaxHostCopyWords);
//   - thread: asked on the command thread rather than the worker;
//   - image / metadata: an image's bytes lie in the destination, or surface metadata does (a
//     store into either needs the image cache's bookkeeping, which the dispatch's storage binding does);
//   - input: the source is not final (a recorded GPU write, a GPU-modified image, pending
//     command-processor or label bytes), is surface metadata, or cannot be read;
//   - encode: the store cannot be recorded on the GPU at this point;
//   - host_kernel_copy: the Metal host's own kernel copy check takes it (an identical copy is consumed with
//     nothing written; under verify the copy runs as its dispatch).
//   - busy (the store class, below): recorded or submitted GPU work could still touch the destination.
// SHARPEMU_POSTED_WORD_COPY=0 (RenderExecutor) turns it off for every host.
//
// The store class (SHARPEMU_METAL_V2_POSTED_WORD_COPY_HOST_STORE, on; =0 posts every copy as a held command-processor
// write, as the port first did). Upstream's Vulkan host applies the copy as a host store at its stream position
// (VulkanVideoPresenter.KernelWrites.cs PostWordCopy -> CommandThread.PostOwnedWrite -> ApplyWrite, a plain
// _memory.TryWrite), which is sound there because every GPU read of guest bytes goes through a buffer-cache copy taken
// when the reading work was recorded (KernelWritePolicy.cs: "earlier readers have already taken their bytes"). Here the
// GPU reads guest memory in place through its windows and nothing tracks which ranges recorded work still reads, so a
// host store is only sound once no recorded or submitted work remains: then it is taken, written now and logged for the
// look-ahead resolutions (a write noted with no pending serial). Otherwise the copy is refused as busy and runs as the
// dispatch it is, which the host registers as a pending copy that readers read through (TryCopyWordsOnHost). Holding it
// as an ordered command-processor write instead costs more than that dispatch on this host: each one is a causal,
// CPU-visible write that holds every later command-processor write behind it, lands per word through the label scatter
// and stays in the pending-write overlay every recorded write scans until it lands (Demon's Souls: ~125 copies of ~3 KB
// a frame, 110 ms frames against 90 ms with them run as dispatches).
internal sealed partial class MetalRenderHost
{
    private const ulong MaxPostedCopyWords = 4 * 1024 * 1024;

    internal enum PostedWordCopyRefusal : byte
    {
        None,
        Unsupported,
        Thread,
        Image,
        Metadata,
        Input,
        Encode,
        HostKernelCopy,
        Busy,
    }

    private static readonly string[] PostedWordCopyRefusalNames = ["posted", "unsupported", "thread", "image", "metadata", "input", "encode", "host_kernel_copy", "busy"];

    private readonly long[] _postedWordCopies = new long[PostedWordCopyRefusalNames.Length];

    /// <summary>
    /// The image refusal asks for image bytes in the destination, not image pages (upstream 90d306cb1).
    /// SHARPEMU_METAL_V2_POSTED_WORD_COPY_EXACT_IMAGE=0 refuses any destination an image shares a page with.
    /// </summary>
    internal static bool PostedWordCopyExactImageBytes { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_POSTED_WORD_COPY_EXACT_IMAGE") != "0";

    /// <summary>
    /// The store class (see the file header): a copy is stored on the host when no GPU work can touch its destination and
    /// refused as busy otherwise. False posts every copy as a held command-processor write.
    /// </summary>
    internal static bool PostedWordCopyHostStore { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_POSTED_WORD_COPY_HOST_STORE") != "0";

    private long _postedWordCopyHostStores;

    /// <summary>Posted copies taken as host stores, since the host started.</summary>
    internal long PostedWordCopyHostStores => Interlocked.Read(ref _postedWordCopyHostStores);

    /// <summary>Posted word copies (index 0) and refusals by reason, since the host started.</summary>
    internal long PostedWordCopies(PostedWordCopyRefusal outcome) => Interlocked.Read(ref _postedWordCopies[(int)outcome]);

    public bool TryPostWordCopy(ulong destination, ulong source, ulong sourceWords, ulong words)
    {
        var outcome = PostWordCopy(destination, source, sourceWords, words);
        Interlocked.Increment(ref _postedWordCopies[(int)outcome]);
        return outcome == PostedWordCopyRefusal.None;
    }

    private PostedWordCopyRefusal PostWordCopy(ulong destination, ulong source, ulong sourceWords, ulong words)
    {
        if (destination == 0 || source == 0 || sourceWords == 0 || words == 0 || ((destination | source) & 3) != 0 ||
            words > MaxPostedCopyWords || sourceWords > MaxPostedCopyWords)
        {
            return PostedWordCopyRefusal.Unsupported;
        }

        var size = words * sizeof(uint);
        var sourceSize = Math.Min(sourceWords, words) * sizeof(uint);
        if (size > ulong.MaxValue - destination || sourceSize > ulong.MaxValue - source ||
            (source < destination + size && destination < source + sourceSize))
        {
            return PostedWordCopyRefusal.Unsupported;
        }

        if (OnCommandThread)
        {
            return PostedWordCopyRefusal.Thread;
        }

        // Exact (upstream 90d306cb1): only an image whose bytes the store meets refuses it, not one that merely shares
        // a page with it; the store's GPU copy touches no byte such an image holds.
        if (_imageCache.MayOwn(destination, size) &&
            (!PostedWordCopyExactImageBytes || _imageCache.HasImageBytes(destination, size)))
        {
            return PostedWordCopyRefusal.Image;
        }

        if (_imageCache.OverlapsSurfaceMetadata(destination, size))
        {
            return PostedWordCopyRefusal.Metadata;
        }

        // Draws deferred before the dispatch record their writes first, so the source check sees them.
        DrainDraws();

        // Checked before the identical-copy evaluation and the source read: a busy copy runs as its dispatch, whose own
        // host check (TryCopyWordsOnHost) makes that evaluation.
        var hostStore = PostedWordCopyHostStore;
        if (hostStore && !NoGpuWorkCanTouch(destination, size))
        {
            return PostedWordCopyRefusal.Busy;
        }

        // The host's own handling of a refused copy kernel comes first (SHARPEMU_METAL_V2_HOST_KERNEL_COPY): a
        // copy whose destination already holds its result is consumed there with nothing written, which a
        // posted store cannot beat, and under verify the copy runs so both checks see it.
        if (HostKernelCopyMode == MetalHostKernelCopyMode.Verify)
        {
            return PostedWordCopyRefusal.HostKernelCopy;
        }

        if (HostKernelCopyMode == MetalHostKernelCopyMode.Identical &&
            EvaluateKernelCopyCore(destination, words, source, sourceWords, default) == KernelCopyOutcome.Identical)
        {
            return PostedWordCopyRefusal.HostKernelCopy;
        }

        if (_imageCache.OverlapsSurfaceMetadata(source, sourceSize) ||
            !IsCopySourceFinal(source, sourceSize, CompletedSerial) ||
            (_labelOverlay.Count != 0 &&
             _labelOverlay.Coverage(source, sourceSize, _completedSerialReader ??= () => CompletedSerial) != MetalOverlayCoverage.None))
        {
            return PostedWordCopyRefusal.Input;
        }

        var data = GC.AllocateUninitializedArray<byte>((int)size);
        if (!_memory.TryRead(source, data.AsSpan(0, (int)sourceSize)))
        {
            return PostedWordCopyRefusal.Input;
        }

        for (var filled = (int)sourceSize; filled < data.Length;)
        {
            var count = Math.Min(filled, data.Length - filled);
            data.AsSpan(0, count).CopyTo(data.AsSpan(filled));
            filled += count;
        }

        if (SharpEmu.HLE.GpuMemory.FrameOverlapDiagnostic.Enabled)
        {
            SharpEmu.HLE.GpuMemory.FrameOverlapDiagnostic.NoteGpuWrite(destination, size, SharpEmu.HLE.GpuMemory.OverlapWrite.CpWrite);
        }

        if (hostStore)
        {
            StorePostedWordCopy(destination, data);
            Interlocked.Increment(ref _postedWordCopyHostStores);
            return PostedWordCopyRefusal.None;
        }

        if (!TryEncodeCommandProcessorWrite(destination, data, "word copy", causal: true))
        {
            return PostedWordCopyRefusal.Encode;
        }

        Interlocked.Increment(ref _gpuTimedCommandProcessorWrites);
        return PostedWordCopyRefusal.None;
    }

    // Whether no GPU work recorded or submitted can still read or write these bytes: no command buffer open (everything
    // recorded goes into one, including work encoded on it directly, which the encoder counts do not see), nothing held
    // to land, every submission complete, and no recorded write or GPU-modified image over them. Reads by recorded work
    // are not tracked on this host, so only an idle GPU proves it.
    private bool NoGpuWorkCanTouch(ulong address, ulong size) =>
        _commandBuffer == 0 && _deferredLabelCopies.Count == 0 && _computeEncoder == 0 && _renderEncoder == 0 &&
        CompletedSerial >= Volatile.Read(ref _submittedSerial) &&
        CanWriteGuestOnHost(address, size);

    // The copy's bytes stored now, as a host fill stores (TryFillGuestMemoryOnHost), and logged with no pending serial: a
    // look-ahead resolution that read these bytes before the store is refused at its take (LoggedWriteCoversRead). The
    // store comes first, so a resolution that begins after the log entry reads the stored bytes.
    private void StorePostedWordCopy(ulong destination, byte[] data)
    {
        var size = (ulong)data.Length;
        if (!_memory.TryWrite(destination, data))
        {
            throw Fatal($"A posted word copy could not be stored into readable guest memory: destination=0x{destination:X16} bytes={size}.");
        }

        SupersedeKernelCopyChecks(destination, size);
        _cpOverlay.Forget(destination, size);
        _labelOverlay.Forget(destination, size);
        _imageCache.InvalidateMemory(destination, size);
        _stagedStorageCopies?.Forget(destination, size);
        _stagedRanges?.OnHostWrite(destination, size);
        _gpuWrites.Note(destination, size, serial: 0, MetalGpuWriteOrigin.Copy);
    }

    /// <summary>The posted word copy fields of the CHURN line; resets them.</summary>
    private string TakePostedWordCopyCounters()
    {
        var text = new System.Text.StringBuilder(" word_copy=[");
        for (var index = 0; index < PostedWordCopyRefusalNames.Length; index++)
        {
            text.Append(index == 0 ? string.Empty : ",").Append(PostedWordCopyRefusalNames[index]).Append('=')
                .Append(Interlocked.Exchange(ref _postedWordCopies[index], 0));
        }

        return text.Append(",host_store=").Append(Interlocked.Exchange(ref _postedWordCopyHostStores, 0)).Append(']').ToString();
    }
}
