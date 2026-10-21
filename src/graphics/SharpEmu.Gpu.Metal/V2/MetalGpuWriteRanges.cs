// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Which packet or host path asked the worker to wait for the GPU. Every wait costs
/// a frame's worth of latency on an idle GPU, so the counters are per reason: the
/// only way to remove one is to know who asks for it.
/// </summary>
internal enum MetalWaitReason
{
    /// <summary>A WAIT_REG_MEM, a predication packet or another explicit CP wait.</summary>
    CommandProcessor,

    /// <summary>A full GPU synchronization: the GDS paths and the shutdown drain.</summary>
    Synchronize,

    /// <summary>A host read of bytes a GPU-written image owns.</summary>
    ReadImage,

    /// <summary>A host read of bytes a recorded shader store or writeback covers.</summary>
    ReadRange,

    /// <summary>The image cache needed the GPU to finish before touching a texture.</summary>
    ImageCache,

    Other,
}

/// <summary>Which host path recorded a GPU write range.</summary>
internal enum MetalGpuWriteOrigin : byte
{
    StorageBuffer,
    WrittenBinding,

    /// <summary>A store through the device-address page table, at an evaluated range.</summary>
    DeviceAddress,
    Fill,
    Copy,
    ImageWriteback,

    /// <summary>An end-of-pipe label the GPU writes where its packet sits in the stream.</summary>
    EndOfPipeLabel,
    Unbounded,
}

internal static class MetalWaitReasons
{
    public const int Count = (int)MetalWaitReason.Other + 1;

    public static readonly string[] Names =
        ["cp", "sync", "read_image", "read_range", "image_cache", "other"];
}

internal static class MetalGpuWriteOrigins
{
    public const int Count = (int)MetalGpuWriteOrigin.Unbounded + 1;

    public static readonly string[] Names =
        ["storage", "binding", "device_address", "fill", "copy", "writeback", "eop_label", "unbounded"];
}

/// <summary>
/// The guest byte ranges recorded work may still be writing, each with the
/// submission serial that will finish it. The twin of the Vulkan
/// <c>GuestBufferCache._gpuModifiedRanges</c>: a CPU or command-processor read only
/// has to wait when it overlaps a range whose serial has not completed, and then
/// only for that serial.
///
/// Guest memory is the GPU's memory on this backend, so the alternative — one flag
/// saying "some shader stored somewhere" — serializes the command processor against
/// the GPU on every label poll, indirect-argument read and DMA source read.
///
/// Only the GPU worker thread touches this.
/// </summary>
internal sealed class MetalGpuWriteRanges
{
    // Above this the oldest records are folded together; a folded record covers more
    // bytes than were written, which can only make a read wait when it need not.
    private const int MaxRecords = 512;

    private struct Record
    {
        public ulong Start;
        public ulong End;
        public ulong Serial;
        // The order of the newest write the record covers, among every write recorded
        // here: which of two overlapping writes in one submission lands last.
        public ulong Sequence;
        public MetalGpuWriteOrigin Origin;
    }

    private Record[] _records = new Record[64];
    private int _count;
    // End-of-pipe labels are exact and never folded: they scatter 4-8 byte writes over
    // the whole command area, so a bounding range would block reads of unrelated memory.
    private Record[] _labels = new Record[64];
    private int _labelCount;

    // How many live labels lie on each 4 KiB page. Black Myth: Wukong records some 12,000
    // labels a second and keeps a couple of thousand live while the GPU runs behind, and
    // every pending-write query scanned all of them: ~8 us a query, 8% of the GPU worker.
    // A query whose pages hold no label skips the scan.
    private readonly Dictionary<ulong, int> _labelPages = [];
    private const int LabelPageShift = 12;

    // A query wider than this many pages scans the labels rather than looking each page up.
    private const ulong MaxLabelPageLookups = 64;
    // A write whose destination range is not known: every read waits for it.
    private ulong _unboundedSerial;
    private ulong _unboundedSequence;
    private ulong _retiredSerial;
    private ulong _sequence;
    private long _recorded;
    private long _folds;

    public int Count => _count + _labelCount;

    public long Recorded => _recorded;

    public long Folds => _folds;

    public bool IsEmpty => _count == 0 && _labelCount == 0 && _unboundedSerial == 0;

    /// <summary>
    /// Changes whenever a write is recorded or the records are cleared; retiring completed
    /// records leaves it alone, since that only ever makes a read wait for less.
    /// </summary>
    public ulong Version { get; private set; }

    /// <summary>
    /// Recorded work in <paramref name="serial"/> stores into this range. Returns the
    /// write's sequence (see <see cref="LatestSequence"/>), or zero when nothing was recorded.
    /// </summary>
    public ulong Note(
        ulong address,
        ulong size,
        ulong serial,
        MetalGpuWriteOrigin origin = MetalGpuWriteOrigin.StorageBuffer)
    {
        if (size == 0 || serial == 0 || serial <= _retiredSerial || address > ulong.MaxValue - size)
        {
            return 0;
        }

        _recorded++;
        Version++;
        var sequence = ++_sequence;
        var end = address + size;
        if (origin == MetalGpuWriteOrigin.EndOfPipeLabel)
        {
            NoteLabel(address, end, serial, sequence);
            return sequence;
        }

        // Draws and dispatches bind the same storage range over and over inside one
        // submission, so the newest record almost always absorbs the next one.
        for (var index = _count - 1; index >= 0 && index >= _count - 8; index--)
        {
            ref var candidate = ref _records[index];
            if (candidate.Serial != serial || candidate.End < address || candidate.Start > end)
            {
                continue;
            }

            candidate.Start = Math.Min(candidate.Start, address);
            candidate.End = Math.Max(candidate.End, end);
            candidate.Sequence = sequence;
            return sequence;
        }

        if (_count == _records.Length)
        {
            if (_records.Length < MaxRecords)
            {
                Array.Resize(ref _records, _records.Length * 2);
            }
            else
            {
                FoldOldest();
            }
        }

        _records[_count++] = new Record { Start = address, End = end, Serial = serial, Sequence = sequence, Origin = origin };
        _recordsByOrigin[(int)origin]++;
        if (address < _boundsStart)
        {
            _boundsStart = address;
        }

        if (end > _boundsEnd)
        {
            _boundsEnd = end;
        }

        return sequence;
    }

    private void NoteLabel(ulong address, ulong end, ulong serial, ulong sequence)
    {
        _recordsByOrigin[(int)MetalGpuWriteOrigin.EndOfPipeLabel]++;
        // The same label is rewritten by consecutive submissions; keep it exact and
        // carry the newest serial.
        for (var index = _labelCount - 1; index >= 0 && index >= _labelCount - 8; index--)
        {
            ref var candidate = ref _labels[index];
            if (candidate.Start == address && candidate.End == end)
            {
                candidate.Serial = Math.Max(candidate.Serial, serial);
                candidate.Sequence = sequence;
                return;
            }
        }

        if (_labelCount == _labels.Length)
        {
            Array.Resize(ref _labels, _labels.Length * 2);
        }

        _labels[_labelCount++] = new Record
        {
            Start = address,
            End = end,
            Serial = serial,
            Sequence = sequence,
            Origin = MetalGpuWriteOrigin.EndOfPipeLabel,
        };
        CountLabelPages(address, end, 1);
        if (address < _boundsStart)
        {
            _boundsStart = address;
        }

        if (end > _boundsEnd)
        {
            _boundsEnd = end;
        }
    }

    private readonly long[] _recordsByOrigin = new long[MetalGpuWriteOrigins.Count];

    /// <summary>How many records each host path contributed since startup.</summary>
    public long RecordsFrom(MetalGpuWriteOrigin origin) => _recordsByOrigin[(int)origin];

    /// <summary>The record the last <see cref="PendingSerial"/> call matched.</summary>
    public (ulong Start, ulong End, ulong Serial, MetalGpuWriteOrigin Origin) LastMatch { get; private set; }

    /// <summary>
    /// Recorded work in <paramref name="serial"/> may store anywhere in guest memory.
    /// Every later read waits for it, so this is the shape of the old global flag and
    /// is only recorded when a write really has no known destination.
    /// </summary>
    public void NoteUnbounded(ulong serial)
    {
        Version++;
        if (serial != 0 && serial > _retiredSerial && serial >= _unboundedSerial)
        {
            _unboundedSequence = ++_sequence;
        }

        if (serial > _unboundedSerial)
        {
            _unboundedSerial = serial;
            _recordsByOrigin[(int)MetalGpuWriteOrigin.Unbounded]++;
        }
    }

    /// <summary>
    /// The serial a read of this range must wait for, or zero when the bytes are not
    /// being written by work that has yet to complete. <paramref name="completed"/> is
    /// the serial the GPU has already signalled.
    /// </summary>
    // The span every live record lies inside. A command-stream read is almost never
    // anywhere near a range a shader wrote, and answering those in two comparisons
    // keeps the GPU-dirty test off the per-packet path: the full test reads the
    // shared event's value, which is a message send.
    private ulong _boundsStart = ulong.MaxValue;
    private ulong _boundsEnd;

    /// <summary>
    /// False when no recorded write can possibly cover these bytes. Cheap enough to
    /// call for every packet the command stream reads, and it touches nothing.
    /// </summary>
    public bool MayCover(ulong address, ulong size) =>
        _unboundedSerial != 0 ||
        ((_count != 0 || _labelCount != 0) && size != 0 && address < _boundsEnd && _boundsStart < address + size);

    public ulong PendingSerial(ulong address, ulong size, ulong completed)
    {
        Retire(completed);
        if (size == 0 || address > ulong.MaxValue - size)
        {
            return 0;
        }

        var end = address + size;
        var pending = _unboundedSerial > completed ? _unboundedSerial : 0;
        if (pending != 0)
        {
            LastMatch = (0, ulong.MaxValue, pending, MetalGpuWriteOrigin.Unbounded);
        }

        for (var index = 0; index < _count; index++)
        {
            ref var record = ref _records[index];
            if (record.Serial > pending && record.End > address && record.Start < end)
            {
                pending = record.Serial;
                LastMatch = (record.Start, record.End, record.Serial, record.Origin);
            }
        }

        var labels = LabelPagesMayCover(address, end) ? _labelCount : 0;
        for (var index = 0; index < labels; index++)
        {
            ref var label = ref _labels[index];
            if (label.Serial > pending && label.End > address && label.Start < end)
            {
                pending = label.Serial;
                LastMatch = (label.Start, label.End, label.Serial, label.Origin);
            }
        }

        return pending;
    }

    /// <summary>
    /// The sequence of the newest recorded write overlapping this range, or zero when no
    /// live record does. Every write gets a larger sequence than the ones recorded before
    /// it, and a merged or folded record carries the largest of its parts, so a write is
    /// the last to land over a range only when its sequence is this value. Call after
    /// <see cref="PendingSerial"/>, which retires what has completed.
    /// </summary>
    public ulong LatestSequence(ulong address, ulong size)
    {
        if (size == 0 || address > ulong.MaxValue - size)
        {
            return 0;
        }

        var end = address + size;
        var latest = _unboundedSerial != 0 ? _unboundedSequence : 0;
        for (var index = 0; index < _count; index++)
        {
            ref var record = ref _records[index];
            if (record.Sequence > latest && record.End > address && record.Start < end)
            {
                latest = record.Sequence;
            }
        }

        var labels = LabelPagesMayCover(address, end) ? _labelCount : 0;
        for (var index = 0; index < labels; index++)
        {
            ref var label = ref _labels[index];
            if (label.Sequence > latest && label.End > address && label.Start < end)
            {
                latest = label.Sequence;
            }
        }

        return latest;
    }

    private void CountLabelPages(ulong start, ulong end, int delta)
    {
        for (var page = start >> LabelPageShift; page <= (end - 1) >> LabelPageShift; page++)
        {
            var count = _labelPages.GetValueOrDefault(page) + delta;
            if (count > 0)
            {
                _labelPages[page] = count;
            }
            else
            {
                _labelPages.Remove(page);
            }
        }
    }

    // Whether a live label may lie in [address, end): a page lookup per page for a narrow
    // query, true (scan them) for a wide one.
    private bool LabelPagesMayCover(ulong address, ulong end)
    {
        if (_labelCount == 0)
        {
            return false;
        }

        var first = address >> LabelPageShift;
        var last = (end - 1) >> LabelPageShift;
        if (last - first >= MaxLabelPageLookups)
        {
            return true;
        }

        for (var page = first; page <= last; page++)
        {
            if (_labelPages.ContainsKey(page))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>Drops every record the GPU has finished.</summary>
    public void Retire(ulong completed)
    {
        if (completed <= _retiredSerial)
        {
            return;
        }

        _retiredSerial = completed;
        if (_unboundedSerial <= completed)
        {
            _unboundedSerial = 0;
        }

        var kept = 0;
        var start = ulong.MaxValue;
        var end = 0UL;
        for (var index = 0; index < _count; index++)
        {
            if (_records[index].Serial <= completed)
            {
                continue;
            }

            ref var record = ref _records[index];
            if (record.Start < start)
            {
                start = record.Start;
            }

            if (record.End > end)
            {
                end = record.End;
            }

            _records[kept++] = record;
        }

        _count = kept;
        kept = 0;
        for (var index = 0; index < _labelCount; index++)
        {
            ref var label = ref _labels[index];
            if (label.Serial <= completed)
            {
                CountLabelPages(label.Start, label.End, -1);
                continue;
            }

            start = Math.Min(start, label.Start);
            end = Math.Max(end, label.End);
            _labels[kept++] = label;
        }

        _labelCount = kept;
        _boundsStart = start;
        _boundsEnd = end;
    }

    /// <summary>Everything recorded has completed.</summary>
    public void Clear(ulong completed)
    {
        Version++;
        _count = 0;
        _labelCount = 0;
        _labelPages.Clear();
        _unboundedSerial = 0;
        _boundsStart = ulong.MaxValue;
        _boundsEnd = 0;
        if (completed > _retiredSerial)
        {
            _retiredSerial = completed;
        }
    }

    // Folds the oldest half of the records into at most a quarter as many, by address.
    // Only the closest neighbours merge (smallest gaps first), so a folded record never
    // spans unrelated memory when anything tighter is available. A merged record keeps
    // the NEWEST serial of its parts: a read overlapping any part waits at least as long
    // as it did before, which can only over-wait, never miss a write.
    private void FoldOldest()
    {
        _folds++;
        var half = _count / 2;
        var old = _records.AsSpan(0, half);
        old.Sort(static (a, b) => a.Start.CompareTo(b.Start));

        var target = Math.Max(1, half / 2);
        var merges = half - target;
        var gaps = new ulong[half - 1];
        for (var index = 1; index < half; index++)
        {
            // Overlapping neighbours have no gap; a record nested in an earlier, longer
            // one is handled by the running end below.
            gaps[index - 1] = old[index].Start > old[index - 1].End ? old[index].Start - old[index - 1].End : 0;
        }

        var sorted = (ulong[])gaps.Clone();
        Array.Sort(sorted);
        var threshold = merges <= 0 ? 0 : sorted[Math.Min(merges, sorted.Length) - 1];

        var folded = new List<Record>(target + 8) { old[0] };
        for (var index = 1; index < half; index++)
        {
            var current = old[index];
            var last = folded[^1];
            if (current.Start <= last.End || gaps[index - 1] <= threshold)
            {
                if (current.Serial > last.Serial)
                {
                    last.Serial = current.Serial;
                    last.Origin = current.Origin;
                }

                last.Sequence = Math.Max(last.Sequence, current.Sequence);

                last.End = Math.Max(last.End, current.End);
                folded[^1] = last;
            }
            else
            {
                folded.Add(current);
            }
        }

        var tail = _count - half;
        var kept = folded.Count;
        Array.Copy(_records, half, _records, kept, tail);
        for (var index = 0; index < kept; index++)
        {
            _records[index] = folded[index];
        }

        _count = kept + tail;
    }
}
