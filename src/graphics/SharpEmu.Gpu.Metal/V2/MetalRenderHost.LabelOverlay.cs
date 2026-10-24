// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

// An end-of-pipe label the GPU has been told to write but has not written yet is read as
// written by the resource materializer, and by nothing else.
//
// A descriptor word the shader's resources depend on can be the low dword of a label in a
// submission that is already committed. Reading it the ordinary way waits for that
// submission: on Silent Hill's title screen 54 such reads took 23.6 s, a median of 443 ms
// each, with the GPU idle behind them. The label's value is known on the CPU when the label
// is encoded, and the draw whose resources are being built executes after the label in queue
// order, so the value the materializer reads is the one memory holds by the time the draw
// runs. That holds for a timestamp or a counter as much as for a constant: what the label
// holds is fixed when it is encoded (a label that samples the clock when the work completes
// is never encoded this way), and a value the materializer reads early is the one in-order
// execution reads late.
//
// Nothing the guest can observe changes. Its waits (WAIT_REG_MEM, label waits, predicate
// polls, CPU reads) go through TryReadGuest and TryReadGuestWaitOperand, which do not
// consult this overlay, so the guest still sees a label only after the GPU has landed it.
// That is why this is not the command-processor overlay (_cpOverlay), which the guest-facing
// reads do consult.
//
// An entry stops answering for bytes a later write covers (NoteGuestMemoryWrite and the host
// write paths forget it, so a newer label over the same bytes wins), is dropped when its
// submission completes (memory holds it then), and is dropped when memory no longer holds
// what it held when the label was encoded, which is how a guest CPU write over the bytes
// shows (MetalPendingWriteOverlay.DropChangedBaselines); the read then waits as it always did.
internal sealed partial class MetalRenderHost
{
    private readonly MetalPendingWriteOverlay _labelOverlay = new(MetalV2Diagnostics.OverlayPageFilter);
    private long _labelOverlayHits;
    private readonly MetalPendingWriteOverlay.RawBytesReader _labelBaselineReader;
    private static readonly bool LabelReaderReuse =
        SharpEmu.ShaderCompiler.PerfSwitches.Enabled("SHARPEMU_METAL_V2_LABEL_READER_REUSE");
    private long _labelReaderCached;
    private long _labelReaderAllocated;

    // The memory object never changes. Forming this interface method group on every
    // check allocated a delegate even when the overlay's page filter refused the range.
    private MetalPendingWriteOverlay.RawBytesReader LabelBaselineReader()
    {
        if (LabelReaderReuse)
        {
            _labelReaderCached++;
            return _labelBaselineReader;
        }
        _labelReaderAllocated++;
        return _memory.TryRead;
    }

    private string TakeLabelReaderCounters() =>
        $" label_reader_cached={Interlocked.Exchange(ref _labelReaderCached, 0)}" +
        $" label_reader_allocated={Interlocked.Exchange(ref _labelReaderAllocated, 0)}";

    // SHARPEMU_METAL_V2_LABEL_OVERLAY=0 is read once; tests set it per host.
    internal bool LabelOverlayEnabled { get; set; } = MetalV2Diagnostics.LabelOverlay;

    internal int PendingLabelOverlayEntries => _labelOverlay.Count;

    /// <summary>Materializer reads answered from pending end-of-pipe label bytes; resets.</summary>
    public long TakeLabelOverlayHits() => Interlocked.Exchange(ref _labelOverlayHits, 0);

    // The label just encoded into submission <paramref name="serial"/> will hold these bytes.
    private void NoteEndOfPipeLabelPending(ulong address, ReadOnlySpan<byte> value, ulong serial)
    {
        if (!LabelOverlayEnabled)
        {
            return;
        }

        // What memory holds over the label now, to tell a later guest CPU write from the landing.
        Span<byte> baseline = stackalloc byte[value.Length];
        if (!_memory.TryRead(address, baseline))
        {
            return;
        }

        if (_labelOverlay.Count >= MaxOverlayEntriesBeforeRetire)
        {
            _labelOverlay.Retire(CompletedSerial);
        }

        _labelOverlay.Add(address, value, serial, baseline);
    }

    // One dword of a descriptor table: a pending label that covers all of it answers without
    // waiting. A partial cover is left to the ordinary read, which waits for the label.
    private bool TryReadPendingLabelWord(ulong address, out uint word)
    {
        word = 0;
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        if (!ApplyPendingLabelBytes(address, bytes, requireFull: true))
        {
            return false;
        }

        word = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes);
        return true;
    }

    // Pending label bytes over bytes the materializer has read from memory (its cache checks).
    private void ApplyPendingLabelBytes(ulong address, Span<byte> destination) =>
        ApplyPendingLabelBytes(address, destination, requireFull: false);

    private bool ApplyPendingLabelBytes(ulong address, Span<byte> destination, bool requireFull)
    {
        if (_labelOverlay.Count == 0 || destination.IsEmpty)
        {
            return false;
        }

        var size = (ulong)destination.Length;
        _labelOverlay.DropChangedBaselines(address, size, LabelBaselineReader());
        var coverage = _labelOverlay.Coverage(address, size, _completedSerialReader ??= () => CompletedSerial);
        if (coverage == MetalOverlayCoverage.None || (requireFull && coverage != MetalOverlayCoverage.Full))
        {
            return false;
        }

        // An image the GPU has drawn into since owns its bytes, and only its writeback can say what they are.
        if (_imageCache.MayOwn(address, size) && _imageCache.HasGpuModifiedImageBytes(address, size))
        {
            return false;
        }

        _labelOverlay.Apply(address, destination);
        Interlocked.Increment(ref _labelOverlayHits);
        return true;
    }
}
