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
//   - image / metadata: an image may own the destination's pages, or surface metadata lies there (a
//     store into either needs the image cache's bookkeeping, which the dispatch's storage binding does);
//   - input: the source is not final (a recorded GPU write, a GPU-modified image, pending
//     command-processor or label bytes), is surface metadata, or cannot be read;
//   - encode: the store cannot be recorded on the GPU at this point;
//   - host_kernel_copy: the Metal host's own kernel copy check takes it (an identical copy is consumed with
//     nothing written; under verify the copy runs as its dispatch).
// SHARPEMU_POSTED_WORD_COPY=0 (RenderExecutor) turns it off for every host.
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
    }

    private static readonly string[] PostedWordCopyRefusalNames = ["posted", "unsupported", "thread", "image", "metadata", "input", "encode", "host_kernel_copy"];

    private readonly long[] _postedWordCopies = new long[PostedWordCopyRefusalNames.Length];

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

        if (_imageCache.MayOwn(destination, size))
        {
            return PostedWordCopyRefusal.Image;
        }

        if (_imageCache.OverlapsSurfaceMetadata(destination, size))
        {
            return PostedWordCopyRefusal.Metadata;
        }

        // Draws deferred before the dispatch record their writes first, so the source check sees them.
        DrainDraws();

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

        if (!TryEncodeCommandProcessorWrite(destination, data, "word copy", causal: true))
        {
            return PostedWordCopyRefusal.Encode;
        }

        Interlocked.Increment(ref _gpuTimedCommandProcessorWrites);
        return PostedWordCopyRefusal.None;
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

        return text.Append(']').ToString();
    }
}
