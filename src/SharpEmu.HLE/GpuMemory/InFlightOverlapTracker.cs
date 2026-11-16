// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace SharpEmu.HLE.GpuMemory;

/// <summary>What a guest range a command buffer reads or writes is used as (attribution only).</summary>
public enum OverlapTag : byte
{
    Other,
    Vertex,
    Index,
    Indirect,
    Constant,
    StorageRead,
    StorageWritten,
    Texture,
    DeviceAddress,
    Occlusion,
    LodStatistics,
}

/// <summary>
/// SHARPEMU_DIAG_FRAME_OVERLAP_ARM=inflight with SHARPEMU_FRAME_OVERLAP_PROBE=1 (diagnostic only): what the
/// guest touches while GPU work it has already been told is finished still runs. Each command buffer's
/// in-place reads and GPU writes are recorded under its serial. On a sampled frame, from the release of its
/// gate (the first publication at or past its present commit + 2) to the next frame's gate, every release
/// arms the buffers it covers that the GPU has not completed: their read pages get a write watch, their
/// written pages a read watch. A page is let go when the last buffer that armed it completes, or at its
/// first fault, which is attributed byte by byte: a store into a read range (the GPU may still read it), a
/// store into a written range, a load of a written range (the GPU has not produced it), or a store into the
/// command buffer of a submission admitted and not yet walked. Times are from the frame's gate.
/// </summary>
public sealed class InFlightOverlapTracker
{
    internal const ulong HugeRangeBytes = 256UL << 20;
    private const ulong PageBytes = TrackerLayout.PageBytes;
    private const int MaxRefsPerPage = 32;
    private const int MaxRetriesPerPage = 256;
    private const int OriginCount = (int)OverlapWrite.EopGpuValued + 1;
    private const int TagCount = (int)OverlapTag.LodStatistics + 1;
    private const int HitLines = 80;

    internal static readonly string[] TagNames =
        ["other", "vertex", "index", "indirect", "constant", "storage_read", "storage_written", "texture", "device_address", "occlusion", "lod_stats"];

    private static readonly string[] OriginNames =
        ["storage", "binding", "device_address", "fill", "copy", "writeback", "eop_label", "unbounded", "cp_write", "eop_dest", "eop_gpu_valued"];

    /// <summary>Where a buffer sits relative to the sampled frame's present commit.</summary>
    internal static readonly string[] EraNames = ["older", "f_first", "f_second", "f_last", "p+1", "p+2", "p+3", "p+4", "next"];

    internal sealed class Record
    {
        public ulong Serial;
        public bool Armed;
        public readonly List<(ulong Address, ulong Size, OverlapTag Tag)> Reads = new();
        public readonly List<(ulong Address, ulong Size, int Origin, OverlapTag Tag, ulong Program)> Writes = new();
        public readonly List<ulong> OwnedPages = new();
    }

    private sealed class PageState
    {
        public Window? Window;
        public ulong MaxSerial;
        public bool InRead, InWrite, Faulted, WriteWatched, ReadWatched, RefsCapped;
        public readonly List<(Record Record, int Index, bool Write)> Refs = new(2);
        public List<(ulong Address, ulong Bytes)>? CommandBuffers;
    }

    internal sealed class Window
    {
        public long Index;
        public ulong PresentSerial, PreviousPresentSerial, GateSerial, MaxArmedSerial;
        public long GateTicks, ArmTicks, FinalizedTicks;
        public bool Closed;
        public int ArmedPages, ReadArmedPages, CapSkipped, SkippedReadWatched, RecordsArmed, HugeRanges, RefsCapped, CommandBufferPages, Managed;
        public int RsHit, RsNeighbour, WsStoreHit, WsStoreNeighbour, WsLoadHit, WsLoadNeighbour, CbHit, CbNeighbour;
        public readonly long[] RsHitByTag = new long[TagCount];
        public readonly long[] RsHitByEra = new long[EraNames.Length];
        public readonly long[] WsLoadByOrigin = new long[OriginCount];
        public readonly long[] WsLoadByTag = new long[TagCount];
        public readonly long[] WsLoadByEra = new long[EraNames.Length];
        public readonly long[] WsStoreByOrigin = new long[OriginCount];
        public readonly HashSet<(ulong, ulong)> RsHitRanges = new();
        public readonly HashSet<(ulong, ulong)> RsPageRanges = new();
        public readonly HashSet<(ulong, ulong)> WsLoadRanges = new();
        public readonly List<double> RsHitMs = new(), WsLoadMs = new(), CbMs = new();
    }

    private readonly IFrameOverlapWatches _watches;
    private readonly Func<long> _clock;
    private readonly int _sampleEvery;
    private readonly int _maxArmedPages;
    private readonly bool _armReads;
    private readonly object _gate = new();

    private Record _open = new();
    private readonly List<Record> _inFlight = new();
    private ulong _completedSerial;
    private long _frames;
    private ulong _lastPresent, _presentBefore;
    private ulong _gateOpenedFor;
    private Window? _arming;
    private readonly List<Window> _windows = new();
    private readonly Dictionary<ulong, PageState> _pages = new();
    private readonly Dictionary<ulong, int> _retries = new();
    private readonly List<(ulong Address, ulong Bytes)> _admitted = new();
    private volatile bool _anyPages;
    private int _hitLines;
    private long _lateFaults;

    // Finalized windows since the last stats line.
    private readonly List<Window> _done = new();

    public InFlightOverlapTracker(IFrameOverlapWatches watches, Func<long> clock, int sampleEvery, int maxArmedPages, bool armReads)
    {
        _watches = watches;
        _clock = clock;
        _sampleEvery = Math.Max(1, sampleEvery);
        _maxArmedPages = Math.Max(1, maxArmedPages);
        _armReads = armReads;
    }

    // ---- recording (worker) ---------------------------------------------------------------

    public void MarkRead(ulong address, ulong size, OverlapTag tag)
    {
        if (size == 0 || size > ulong.MaxValue - address)
        {
            return;
        }

        lock (_gate)
        {
            _open.Reads.Add((address, size, tag));
        }
    }

    public void NoteWrite(ulong address, ulong size, OverlapWrite origin, OverlapTag tag, ulong program)
    {
        if (origin == OverlapWrite.Unbounded || size == 0 || size > ulong.MaxValue - address)
        {
            return;
        }

        lock (_gate)
        {
            _open.Writes.Add((address, size, (int)origin, tag, program));
        }
    }

    /// <summary>The worker committed <paramref name="serial"/>: what was recorded since the last commit is its buffer.</summary>
    public void OnCommit(ulong serial, bool present)
    {
        lock (_gate)
        {
            var record = _open;
            _open = new Record();
            record.Serial = serial;
            if (serial > _completedSerial && (record.Reads.Count != 0 || record.Writes.Count != 0))
            {
                _inFlight.Add(record);
            }

            if (present)
            {
                _presentBefore = _lastPresent;
                _lastPresent = serial;
                _frames++;
            }
        }
    }

    // ---- admission (command stream) -------------------------------------------------------

    public void OnAdmission(ulong address, ulong bytes)
    {
        if (bytes == 0)
        {
            return;
        }

        lock (_gate)
        {
            _admitted.Add((address, bytes));
            if (_arming is { } window)
            {
                ArmCommandBufferLocked(window, address, bytes);
            }
        }
    }

    public void OnSubmissionWalked(ulong address, ulong bytes)
    {
        lock (_gate)
        {
            var index = _admitted.IndexOf((address, bytes));
            if (index >= 0)
            {
                _admitted.RemoveAt(index);
            }

            var masks = new Dictionary<(ulong, bool), PageMask>();
            for (var page = address & ~(PageBytes - 1); page < address + bytes; page += PageBytes)
            {
                if (_pages.TryGetValue(page, out var state) && state.CommandBuffers is { } buffers)
                {
                    buffers.Remove((address, bytes));
                    if (buffers.Count == 0)
                    {
                        state.CommandBuffers = null;
                        if (state.MaxSerial <= _completedSerial)
                        {
                            Unwatch(page, state, masks);
                        }
                    }
                }
            }

            Apply(masks, add: false);
        }
    }

    // ---- release and completion (completion thread) ----------------------------------------

    /// <summary>
    /// The probe is about to let the guest see everything up to <paramref name="serial"/>. Runs before
    /// the publication, so the watches are in place before the guest can act on it.
    /// </summary>
    public void OnRelease(ulong serial)
    {
        var started = _clock();
        lock (_gate)
        {
            if (_lastPresent != 0 && _gateOpenedFor != _lastPresent && serial >= _lastPresent + 2)
            {
                // This frame's gate: the window of the previous sampled frame stops taking buffers.
                _gateOpenedFor = _lastPresent;
                if (_arming is { } previous)
                {
                    previous.Closed = true;
                    _arming = null;
                    FinalizeIfDoneLocked();
                }

                if (_frames % _sampleEvery == 0)
                {
                    var window = new Window
                    {
                        Index = _frames,
                        PresentSerial = _lastPresent,
                        PreviousPresentSerial = _presentBefore,
                        GateSerial = serial,
                        GateTicks = started,
                    };
                    _arming = window;
                    _windows.Add(window);
                    foreach (var (address, bytes) in _admitted)
                    {
                        ArmCommandBufferLocked(window, address, bytes);
                    }
                }
            }

            if (_arming is not { } arming)
            {
                return;
            }

            var masks = new Dictionary<(ulong, bool), PageMask>();
            foreach (var record in _inFlight)
            {
                if (record.Armed || record.Serial > serial || record.Serial <= _completedSerial)
                {
                    continue;
                }

                ArmRecordLocked(arming, record, masks);
            }

            Apply(masks, add: true);
            arming.ArmTicks += _clock() - started;
        }
    }

    public void OnCompleted(ulong serial)
    {
        lock (_gate)
        {
            if (serial <= _completedSerial)
            {
                return;
            }

            _completedSerial = serial;
            var masks = new Dictionary<(ulong, bool), PageMask>();
            for (var index = 0; index < _inFlight.Count;)
            {
                var record = _inFlight[index];
                if (record.Serial > serial)
                {
                    index++;
                    continue;
                }

                _inFlight.RemoveAt(index);
                foreach (var page in record.OwnedPages)
                {
                    if (_pages.TryGetValue(page, out var state) && state.MaxSerial == record.Serial && state.CommandBuffers is null)
                    {
                        Unwatch(page, state, masks);
                    }
                }
            }

            Apply(masks, add: false);
            FinalizeIfDoneLocked();
        }
    }

    // ---- arming -------------------------------------------------------------------------------

    private void ArmRecordLocked(Window window, Record record, Dictionary<(ulong, bool), PageMask> masks)
    {
        record.Armed = true;
        window.RecordsArmed++;
        window.MaxArmedSerial = Math.Max(window.MaxArmedSerial, record.Serial);
        for (var index = 0; index < record.Reads.Count; index++)
        {
            var read = record.Reads[index];
            AddRange(window, record, index, read.Address, read.Size, write: false, masks);
        }

        for (var index = 0; index < record.Writes.Count; index++)
        {
            var write = record.Writes[index];
            AddRange(window, record, index, write.Address, write.Size, write: true, masks);
        }
    }

    private void AddRange(Window window, Record record, int index, ulong address, ulong size, bool write, Dictionary<(ulong, bool), PageMask> masks)
    {
        if (size > HugeRangeBytes)
        {
            window.HugeRanges++;
            return;
        }

        var end = address + size;
        for (var page = address & ~(PageBytes - 1); page < end; page += PageBytes)
        {
            if (!_pages.TryGetValue(page, out var state))
            {
                if (window.ArmedPages >= _maxArmedPages)
                {
                    window.CapSkipped++;
                    continue;
                }

                state = new PageState();
                _pages[page] = state;
                _anyPages = true;
            }

            if (!ReferenceEquals(state.Window, window))
            {
                state.Window = window;
                state.Faulted = false;
                state.Refs.Clear();
                state.RefsCapped = false;
                state.InRead = state.InWrite = false;
            }

            if (state.Refs.Count < MaxRefsPerPage)
            {
                state.Refs.Add((record, index, write));
            }
            else if (!state.RefsCapped)
            {
                state.RefsCapped = true;
                window.RefsCapped++;
            }

            if (write) state.InWrite = true;
            else state.InRead = true;
            if (record.Serial > state.MaxSerial)
            {
                state.MaxSerial = record.Serial;
                record.OwnedPages.Add(page);
            }

            if (state.Faulted)
            {
                continue;
            }

            Watch(window, page, state, masks);
        }
    }

    private void Watch(Window window, ulong page, PageState state, Dictionary<(ulong, bool), PageMask> masks)
    {
        var wasWatched = state.WriteWatched || state.ReadWatched;
        if (state.InWrite && _armReads && !state.ReadWatched)
        {
            if (_watches.IsReadWatched(page))
            {
                if (!wasWatched)
                {
                    window.SkippedReadWatched++;
                }
            }
            else
            {
                state.ReadWatched = true;
                window.ReadArmedPages++;
                SetMask(masks, page, reads: true);
            }
        }

        if (!state.ReadWatched && !state.WriteWatched)
        {
            state.WriteWatched = true;
            SetMask(masks, page, reads: false);
        }

        if (!wasWatched)
        {
            window.ArmedPages++;
        }
    }

    private void ArmCommandBufferLocked(Window window, ulong address, ulong bytes)
    {
        var masks = new Dictionary<(ulong, bool), PageMask>();
        for (var page = address & ~(PageBytes - 1); page < address + bytes; page += PageBytes)
        {
            if (!_pages.TryGetValue(page, out var state))
            {
                state = new PageState { Window = window };
                _pages[page] = state;
                _anyPages = true;
            }

            (state.CommandBuffers ??= new()).Add((address, bytes));
            if (!state.WriteWatched && !state.ReadWatched && !(state.Faulted && ReferenceEquals(state.Window, window)))
            {
                if (!ReferenceEquals(state.Window, window))
                {
                    state.Window = window;
                    state.Refs.Clear();
                    state.InRead = state.InWrite = false;
                }

                state.Faulted = false;
                state.WriteWatched = true;
                window.CommandBufferPages++;
                window.ArmedPages++;
                SetMask(masks, page, reads: false);
            }
        }

        Apply(masks, add: true);
    }

    private void Unwatch(ulong page, PageState state, Dictionary<(ulong, bool), PageMask> masks)
    {
        if (state.WriteWatched)
        {
            SetMask(masks, page, reads: false);
            state.WriteWatched = false;
        }

        if (state.ReadWatched)
        {
            SetMask(masks, page, reads: true);
            state.ReadWatched = false;
        }
    }

    private static void SetMask(Dictionary<(ulong, bool), PageMask> masks, ulong page, bool reads)
    {
        var key = (page & ~(TrackerLayout.BlockBytes - 1), reads);
        masks.TryGetValue(key, out var mask);
        mask.Set((int)((page & (TrackerLayout.BlockBytes - 1)) / PageBytes));
        masks[key] = mask;
    }

    private void Apply(Dictionary<(ulong, bool), PageMask> masks, bool add)
    {
        foreach (var pair in masks)
        {
            var mask = pair.Value;
            if (add) _watches.Add(pair.Key.Item1, mask, pair.Key.Item2);
            else _watches.Remove(pair.Key.Item1, mask, pair.Key.Item2);
        }
    }

    // ---- faults -------------------------------------------------------------------------------

    public bool TryHandleFault(FaultKind kind, ulong address, out bool stillBlocked)
    {
        stillBlocked = false;
        if (!_anyPages)
        {
            return false;
        }

        var now = _clock();
        var page = address & ~(PageBytes - 1);
        lock (_gate)
        {
            if (!_pages.TryGetValue(page, out var state))
            {
                return false;
            }

            var ours = kind == FaultKind.Read ? state.ReadWatched : state.ReadWatched || state.WriteWatched;
            if (!ours)
            {
                // Let go already (another thread's fault, or the buffer completed) but the access met the
                // protection the watch had: once the release is applied it goes through, so it is retried.
                if (!_watches.StillBlocks(page, kind))
                {
                    var retries = _retries.GetValueOrDefault(page);
                    if (retries < MaxRetriesPerPage)
                    {
                        _retries[page] = retries + 1;
                        _lateFaults++;
                        return true;
                    }
                }

                return false;
            }

            if (state.Window is { } window)
            {
                Attribute(window, state, address, write: kind != FaultKind.Read, now);
            }

            state.Faulted = true;
            var masks = new Dictionary<(ulong, bool), PageMask>();
            Unwatch(page, state, masks);
            Apply(masks, add: false);
            stillBlocked = _watches.StillBlocks(page, kind);
            return true;
        }
    }

    /// <summary>A managed (HLE) write into an armed page: it does not fault, so it is only counted.</summary>
    public void NoteManagedWrite(ulong address, ulong size)
    {
        if (!_anyPages || size == 0 || size > ulong.MaxValue - address)
        {
            return;
        }

        lock (_gate)
        {
            var end = address + size;
            var pages = 0;
            for (var page = address & ~(PageBytes - 1); page < end && pages < 4096; page += PageBytes, pages++)
            {
                if (_pages.TryGetValue(page, out var state) && (state.WriteWatched || state.ReadWatched) && state.Window is { } window)
                {
                    window.Managed++;
                }
            }
        }
    }

    internal int Era(Window window, ulong serial)
    {
        if (serial > window.PresentSerial)
        {
            var after = serial - window.PresentSerial;
            return after <= 4 ? 3 + (int)after : 8;
        }

        if (serial <= window.PreviousPresentSerial || window.PresentSerial <= window.PreviousPresentSerial)
        {
            return 0;
        }

        var span = window.PresentSerial - window.PreviousPresentSerial;
        var position = (double)(serial - window.PreviousPresentSerial) / span;
        return position <= 1.0 / 3 ? 1 : position <= 2.0 / 3 ? 2 : 3;
    }

    private void Attribute(Window window, PageState state, ulong address, bool write, long now)
    {
        var ms = (now - window.GateTicks) * 1000.0 / Stopwatch.Frequency;
        if (write)
        {
            foreach (var reference in state.Refs)
            {
                if (reference.Write)
                {
                    continue;
                }

                var read = reference.Record.Reads[reference.Index];
                window.RsPageRanges.Add((read.Address, read.Size));
            }

            foreach (var reference in state.Refs)
            {
                if (reference.Write)
                {
                    continue;
                }

                var read = reference.Record.Reads[reference.Index];
                if (address >= read.Address && address - read.Address < read.Size)
                {
                    window.RsHit++;
                    window.RsHitByTag[(int)read.Tag]++;
                    var era = Era(window, reference.Record.Serial);
                    window.RsHitByEra[era]++;
                    window.RsHitRanges.Add((read.Address, read.Size));
                    window.RsHitMs.Add(ms);
                    Report(window, "store_rs", address, read.Address, read.Size, TagNames[(int)read.Tag], reference.Record.Serial, era, ms, 0);
                    return;
                }
            }

            foreach (var reference in state.Refs)
            {
                if (!reference.Write)
                {
                    continue;
                }

                var written = reference.Record.Writes[reference.Index];
                if (address >= written.Address && address - written.Address < written.Size)
                {
                    window.WsStoreHit++;
                    window.WsStoreByOrigin[written.Origin]++;
                    Report(window, "store_ws", address, written.Address, written.Size, OriginNames[written.Origin] + "/" + TagNames[(int)written.Tag],
                        reference.Record.Serial, Era(window, reference.Record.Serial), ms, written.Program);
                    return;
                }
            }

            if (state.CommandBuffers is { } buffers)
            {
                foreach (var (bufferAddress, bytes) in buffers)
                {
                    if (address >= bufferAddress && address - bufferAddress < bytes)
                    {
                        window.CbHit++;
                        window.CbMs.Add(ms);
                        Report(window, "store_cb", address, bufferAddress, bytes, "command_buffer", 0, 8, ms, 0);
                        return;
                    }
                }
            }

            if (state.InRead) window.RsNeighbour++;
            else if (state.InWrite) window.WsStoreNeighbour++;
            else window.CbNeighbour++;
            return;
        }

        foreach (var reference in state.Refs)
        {
            if (!reference.Write)
            {
                continue;
            }

            var written = reference.Record.Writes[reference.Index];
            if (address >= written.Address && address - written.Address < written.Size)
            {
                window.WsLoadHit++;
                window.WsLoadByOrigin[written.Origin]++;
                window.WsLoadByTag[(int)written.Tag]++;
                var era = Era(window, reference.Record.Serial);
                window.WsLoadByEra[era]++;
                window.WsLoadRanges.Add((written.Address, written.Size));
                window.WsLoadMs.Add(ms);
                Report(window, "load_ws", address, written.Address, written.Size, OriginNames[written.Origin] + "/" + TagNames[(int)written.Tag],
                    reference.Record.Serial, era, ms, written.Program);
                return;
            }
        }

        window.WsLoadNeighbour++;
    }

    private void Report(Window window, string what, ulong address, ulong rangeAddress, ulong rangeSize, string kind, ulong serial, int era, double ms, ulong program)
    {
        if (_hitLines >= HitLines)
        {
            return;
        }

        _hitLines++;
        var offset = serial == 0 ? "-" : ((long)serial - (long)window.PresentSerial).ToString(CultureInfo.InvariantCulture);
        Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"[METAL-V2][OVERLAP-INFLIGHT-HIT] frame={window.Index} {what} addr=0x{address:X} range=0x{rangeAddress:X}+0x{rangeSize:X} kind={kind} " +
            $"serial_vs_present={offset} era={EraNames[era]} ms_after_gate={ms:F2} program=0x{program:X16}"));
    }

    // ---- windows ------------------------------------------------------------------------------

    private void FinalizeIfDoneLocked()
    {
        for (var index = 0; index < _windows.Count;)
        {
            var window = _windows[index];
            if (!window.Closed || window.MaxArmedSerial > _completedSerial)
            {
                index++;
                continue;
            }

            _windows.RemoveAt(index);
            window.FinalizedTicks = _clock();
            _done.Add(window);
        }
    }

    // Pages of windows finalized over half a second ago that hold no watch: a fault that raced their
    // release has been retried by then, so they are forgotten (stats-window thread).
    private void PruneLocked(long now)
    {
        var stale = new List<ulong>();
        foreach (var pair in _pages)
        {
            var state = pair.Value;
            if (!state.ReadWatched && !state.WriteWatched && state.CommandBuffers is null &&
                (state.Window is null || (state.Window.FinalizedTicks != 0 && now - state.Window.FinalizedTicks > Stopwatch.Frequency / 2)))
            {
                stale.Add(pair.Key);
            }
        }

        foreach (var page in stale)
        {
            _pages.Remove(page);
            _retries.Remove(page);
        }

        _anyPages = _pages.Count != 0;
    }

    internal IReadOnlyList<Window> TakeFinished()
    {
        lock (_gate)
        {
            var finished = _done.ToArray();
            _done.Clear();
            return finished;
        }
    }

    /// <summary>One line per stats window: per sampled frame, mean (max) and time percentiles from the gate.</summary>
    public string? TakeWindowLine()
    {
        var windows = TakeFinished();
        long late;
        lock (_gate)
        {
            late = _lateFaults;
            PruneLocked(_clock());
        }

        if (windows.Count == 0)
        {
            return null;
        }

        var culture = CultureInfo.InvariantCulture;
        var n = windows.Count;
        string M(Func<Window, double> pick) => string.Create(culture, $"{windows.Average(pick):F1}/{windows.Max(pick):F0}");
        string Mb(Func<Window, IEnumerable<(ulong, ulong)>> pick) =>
            string.Create(culture, $"{windows.Average(w => pick(w).Sum(r => (double)r.Item2)) / (1 << 20):F3}");
        string Split(Func<Window, long[]> pick, string[] names)
        {
            var builder = new StringBuilder();
            for (var index = 0; index < names.Length; index++)
            {
                var total = windows.Sum(w => pick(w)[index]);
                if (total != 0)
                {
                    builder.Append(builder.Length == 0 ? string.Empty : ",").Append(names[index]).Append(':')
                        .Append(((double)total / n).ToString("F1", culture));
                }
            }

            return builder.ToString();
        }

        string P(Func<Window, List<double>> pick)
        {
            var values = windows.SelectMany(pick).ToList();
            return FrameOverlapTracker.Percentiles(values, culture);
        }

        string FirstP(Func<Window, List<double>> pick)
        {
            var values = windows.Where(w => pick(w).Count != 0).Select(w => pick(w).Min()).ToList();
            return FrameOverlapTracker.Percentiles(values, culture);
        }

        return string.Create(culture,
            $"[METAL-V2][OVERLAP-INFLIGHT] frames={n} armed_pages={M(w => w.ArmedPages)} read_armed={M(w => w.ReadArmedPages)} " +
            $"cb_pages={M(w => w.CommandBufferPages)} buffers_armed={M(w => w.RecordsArmed)} arm_ms={M(w => w.ArmTicks * 1000.0 / Stopwatch.Frequency)} " +
            $"cap_skipped={M(w => w.CapSkipped)} skipped_read_watched={M(w => w.SkippedReadWatched)} huge={M(w => w.HugeRanges)} refs_capped={M(w => w.RefsCapped)} " +
            $"| 1a store_rs_hit_pages={M(w => w.RsHit)} ranges={M(w => w.RsHitRanges.Count)} hit_range_mb={Mb(w => w.RsHitRanges)} " +
            $"page_range_mb={Mb(w => w.RsPageRanges)} nb_pages={M(w => w.RsNeighbour)} by_tag=[{Split(w => w.RsHitByTag, TagNames)}] " +
            $"by_era=[{Split(w => w.RsHitByEra, EraNames)}] first_ms={FirstP(w => w.RsHitMs)} all_ms={P(w => w.RsHitMs)} " +
            $"| 1b store_cb_hit_pages={M(w => w.CbHit)} nb={M(w => w.CbNeighbour)} first_ms={FirstP(w => w.CbMs)} " +
            $"| ws store_hit_pages={M(w => w.WsStoreHit)} nb={M(w => w.WsStoreNeighbour)} by_origin=[{Split(w => w.WsStoreByOrigin, OriginNames)}] " +
            $"| 2 load_ws_hit_pages={M(w => w.WsLoadHit)} ranges={M(w => w.WsLoadRanges.Count)} mb={Mb(w => w.WsLoadRanges)} nb_pages={M(w => w.WsLoadNeighbour)} " +
            $"by_origin=[{Split(w => w.WsLoadByOrigin, OriginNames)}] by_tag=[{Split(w => w.WsLoadByTag, TagNames)}] by_era=[{Split(w => w.WsLoadByEra, EraNames)}] " +
            $"first_ms={FirstP(w => w.WsLoadMs)} all_ms={P(w => w.WsLoadMs)} | managed={M(w => w.Managed)} late_faults={late}");
    }
}
