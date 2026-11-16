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
/// in-place reads and GPU writes are recorded under its serial. For a sampled frame F (every Kth), the
/// buffers committed from the present before F's to F's own present and its post-flip buffers are armed
/// as they commit, on a background thread, so the probe's publications are not delayed: read pages get a
/// write watch, written pages a read watch, and the command buffers of submissions admitted and not yet
/// walked a write watch. A buffer's pages are let go once it completes. F's window opens at its gate (the
/// first publication at or past its present + 2) and closes at the next frame's gate. A fault is
/// attributed byte by byte to the buffer whose range holds it and counted once per page: before that
/// buffer was published (pre-release: the page is armed again when it is), after it completed (not a
/// race), or between the two, the race the early publication opens: a store into a read range (1a), a
/// store into a written range, a load of a written range (2), a store into a command buffer not yet
/// walked (1b). Times are from F's gate.
/// </summary>
public sealed class InFlightOverlapTracker : IDisposable
{
    internal const ulong HugeRangeBytes = 256UL << 20;
    private const ulong PageBytes = TrackerLayout.PageBytes;
    private const int MaxRetriesPerPage = 256;
    private const int OriginCount = (int)OverlapWrite.EopGpuValued + 1;
    private const int TagCount = (int)OverlapTag.LodStatistics + 1;
    private const int HitLines = 120;

    private const byte InRead = 1, InWrite = 2, WriteWatched = 4, ReadWatched = 8, Faulted = 16, CommandBuffer = 32;

    internal static readonly string[] TagNames =
        ["other", "vertex", "index", "indirect", "constant", "storage_read", "storage_written", "texture", "device_address", "occlusion", "lod_stats"];

    private static readonly string[] OriginNames =
        ["storage", "binding", "device_address", "fill", "copy", "writeback", "eop_label", "unbounded", "cp_write", "eop_dest", "eop_gpu_valued"];

    /// <summary>Where a buffer sits relative to the sampled frame's present commit.</summary>
    internal static readonly string[] EraNames = ["older", "f_first", "f_second", "f_last", "p+1", "p+2", "p+3", "p+4", "next"];

    internal sealed class Record
    {
        public ulong Serial;
        public Window? Window;
        public bool Armed, Disarmed, Late;
        public readonly List<(ulong Address, ulong Size, OverlapTag Tag)> Reads = new();
        public readonly List<(ulong Address, ulong Size, int Origin, OverlapTag Tag, ulong Program)> Writes = new();
        public readonly List<ulong> OwnedPages = new();
        public (ulong Start, ulong End, int Index)[]? ReadIndex, WriteIndex;
        public ulong[]? ReadMaxEnd, WriteMaxEnd;
    }

    private struct PageState
    {
        public Window? Window;
        public ulong MaxSerial;
        public byte Flags;
    }

    internal sealed class Window
    {
        public long Index;
        public ulong PresentSerial, PreviousPresentSerial;
        public long GateTicks, ArmTicks, FinalizedTicks;
        public bool Open, Closed;
        public readonly List<Record> Records = new();
        public readonly List<ulong> PreReleasePages = new();
        public int ArmedPages, ReadArmedPages, CapSkipped, SkippedReadWatched, HugeRanges, CommandBufferPages, Managed, LateRecords, Rearmed;
        public int PreReleaseStores, PreReleaseLoads, AfterCompletion;
        public int RsHit, RsNeighbour, WsStoreHit, WsStoreNeighbour, WsLoadHit, WsLoadNeighbour, CbHit, CbNeighbour;
        public readonly long[] RsHitByTag = new long[TagCount];
        public readonly long[] RsHitByEra = new long[EraNames.Length];
        public readonly long[] PreStoreByTag = new long[TagCount];
        public readonly long[] WsLoadByOrigin = new long[OriginCount];
        public readonly long[] WsLoadByTag = new long[TagCount];
        public readonly long[] WsLoadByEra = new long[EraNames.Length];
        public readonly long[] WsStoreByOrigin = new long[OriginCount];
        public readonly HashSet<(ulong, ulong)> RsHitRanges = new();
        public readonly HashSet<(ulong, ulong)> WsLoadRanges = new();
        public readonly List<double> RsHitMs = new(), WsLoadMs = new(), CbMs = new();
    }

    private readonly IFrameOverlapWatches _watches;
    private readonly Func<long> _clock;
    private readonly int _sampleEvery;
    private readonly int _maxArmedPages;
    private readonly bool _armReads;
    private readonly object _gate = new();
    private readonly Queue<(bool Arm, Record? Record, ulong Completed)> _requests = new();
    private readonly Thread? _thread;
    private bool _stopping;

    private Record _open = new();
    private readonly List<Record> _inFlight = new();
    private ulong _completedSerial, _releasedSerial;
    private long _frames;
    private ulong _lastPresent;
    private ulong _gateOpenedFor;
    private readonly List<Window> _windows = new();
    private readonly Dictionary<ulong, PageState> _pages = new();
    private readonly Dictionary<ulong, int> _retries = new();
    private readonly List<(ulong Address, ulong Bytes)> _admitted = new();
    private volatile bool _anyPages;
    private int _hitLines;
    private long _lateFaults, _declined;
    private readonly List<Window> _done = new();

    public InFlightOverlapTracker(IFrameOverlapWatches watches, Func<long> clock, int sampleEvery, int maxArmedPages, bool armReads, bool startThread = true)
    {
        _watches = watches;
        _clock = clock;
        _sampleEvery = Math.Max(1, sampleEvery);
        _maxArmedPages = Math.Max(1, maxArmedPages);
        _armReads = armReads;
        if (startThread)
        {
            _thread = new Thread(Run) { IsBackground = true, Name = "SharpEmu in-flight overlap diagnostic" };
            _thread.Start();
        }
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

    private Window WindowFor(long frame)
    {
        foreach (var window in _windows)
        {
            if (window.Index == frame)
            {
                return window;
            }
        }

        var created = new Window { Index = frame };
        _windows.Add(created);
        return created;
    }

    /// <summary>The worker committed <paramref name="serial"/>: what was recorded since the last commit is its buffer.</summary>
    public void OnCommit(ulong serial, bool present)
    {
        Record? arm = null;
        lock (_gate)
        {
            var record = _open;
            _open = new Record();
            record.Serial = serial;
            // Frame F (F % K == 0) arms its own buffers (committed while F-1 presents were seen) and its post-flip ones.
            long? frame = _frames % _sampleEvery == 0 ? _frames : (_frames + 1) % _sampleEvery == 0 ? _frames + 1 : null;
            if (frame is { } index && index > 0 && serial > _completedSerial && (record.Reads.Count != 0 || record.Writes.Count != 0))
            {
                record.Window = WindowFor(index);
                record.Window.Records.Add(record);
                _inFlight.Add(record);
                arm = record;
            }

            if (present)
            {
                _frames++;
                if (_frames % _sampleEvery == 0)
                {
                    var window = WindowFor(_frames);
                    window.PresentSerial = serial;
                    window.PreviousPresentSerial = _lastPresent;
                }

                _lastPresent = serial;
            }
        }

        if (arm is not null)
        {
            Post(arm: true, arm, 0);
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
            foreach (var window in _windows)
            {
                if (window.Open && !window.Closed)
                {
                    ArmCommandBufferLocked(window, address, bytes);
                }
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
                ref var state = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrNullRef(_pages, page);
                if (System.Runtime.CompilerServices.Unsafe.IsNullRef(ref state) || (state.Flags & CommandBuffer) == 0)
                {
                    continue;
                }

                if (!StillAdmitted(page))
                {
                    state.Flags &= unchecked((byte)~CommandBuffer);
                    if (state.MaxSerial <= _completedSerial)
                    {
                        Unwatch(page, ref state, masks);
                    }
                }
            }

            Apply(masks, add: false);
        }
    }

    private bool StillAdmitted(ulong page)
    {
        foreach (var (address, bytes) in _admitted)
        {
            if (page < address + bytes && address < page + PageBytes)
            {
                return true;
            }
        }

        return false;
    }

    // ---- release and completion (completion thread) ----------------------------------------

    /// <summary>The probe is about to let the guest see everything up to <paramref name="serial"/>.</summary>
    public void OnRelease(ulong serial)
    {
        var now = _clock();
        lock (_gate)
        {
            _releasedSerial = Math.Max(_releasedSerial, serial);
            if (_lastPresent != 0 && _gateOpenedFor != _lastPresent && serial >= _lastPresent + 2)
            {
                _gateOpenedFor = _lastPresent;
                foreach (var window in _windows)
                {
                    if (window.Open && !window.Closed)
                    {
                        window.Closed = true;
                    }
                }

                if (_frames % _sampleEvery == 0)
                {
                    var window = WindowFor(_frames);
                    window.Open = true;
                    window.GateTicks = now;
                    foreach (var (address, bytes) in _admitted)
                    {
                        ArmCommandBufferLocked(window, address, bytes);
                    }
                }

                FinalizeIfDoneLocked();
            }

            // Pages a buffer's own guest touched before it was released are watched again now, and
            // released buffers the arming thread has not reached yet are counted.
            var masks = new Dictionary<(ulong, bool), PageMask>();
            foreach (var window in _windows)
            {
                if (window.PreReleasePages.Count != 0)
                {
                    for (var index = 0; index < window.PreReleasePages.Count;)
                    {
                        var page = window.PreReleasePages[index];
                        ref var state = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrNullRef(_pages, page);
                        if (System.Runtime.CompilerServices.Unsafe.IsNullRef(ref state) || state.MaxSerial <= _completedSerial)
                        {
                            window.PreReleasePages.RemoveAt(index);
                            continue;
                        }

                        if (state.MaxSerial > serial)
                        {
                            index++;
                            continue;
                        }

                        window.PreReleasePages.RemoveAt(index);
                        state.Flags &= unchecked((byte)~Faulted);
                        window.Rearmed++;
                        Watch(window, page, ref state, masks, count: false);
                    }
                }

                foreach (var record in window.Records)
                {
                    if (!record.Armed && !record.Late && record.Serial <= serial && record.Serial > _completedSerial)
                    {
                        record.Late = true;
                        window.LateRecords++;
                    }
                }
            }

            Apply(masks, add: true);
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
        }

        Post(arm: false, null, serial);
    }

    // ---- the arming thread ---------------------------------------------------------------------

    private void Post(bool arm, Record? record, ulong completed)
    {
        if (_thread is null)
        {
            Process(arm, record, completed);
            return;
        }

        lock (_requests)
        {
            _requests.Enqueue((arm, record, completed));
            Monitor.Pulse(_requests);
        }
    }

    private void Run()
    {
        while (true)
        {
            (bool Arm, Record? Record, ulong Completed) request;
            lock (_requests)
            {
                while (_requests.Count == 0 && !_stopping)
                {
                    Monitor.Wait(_requests);
                }

                if (_stopping)
                {
                    return;
                }

                request = _requests.Dequeue();
            }

            try
            {
                Process(request.Arm, request.Record, request.Completed);
            }
            catch (Exception exception)
            {
                Console.Error.WriteLine($"[METAL-V2][OVERLAP-INFLIGHT][ERROR] {(request.Arm ? "arm" : "disarm")}: {exception}");
            }
        }
    }

    private void Process(bool arm, Record? record, ulong completed)
    {
        if (arm)
        {
            var started = _clock();
            lock (_gate)
            {
                if (record!.Serial <= _completedSerial || record.Window is not { } window)
                {
                    return;
                }

                var masks = new Dictionary<(ulong, bool), PageMask>();
                ArmRecordLocked(window, record, masks);
                Apply(masks, add: true);
                window.ArmTicks += _clock() - started;
            }

            return;
        }

        lock (_gate)
        {
            var masks = new Dictionary<(ulong, bool), PageMask>();
            for (var index = 0; index < _inFlight.Count;)
            {
                var done = _inFlight[index];
                if (done.Serial > completed)
                {
                    index++;
                    continue;
                }

                _inFlight.RemoveAt(index);
                done.Disarmed = true;
                foreach (var page in done.OwnedPages)
                {
                    ref var state = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrNullRef(_pages, page);
                    if (!System.Runtime.CompilerServices.Unsafe.IsNullRef(ref state) && state.MaxSerial == done.Serial && (state.Flags & CommandBuffer) == 0)
                    {
                        Unwatch(page, ref state, masks);
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
        foreach (var read in record.Reads)
        {
            AddRange(window, record, read.Address, read.Size, write: false, masks);
        }

        foreach (var write in record.Writes)
        {
            AddRange(window, record, write.Address, write.Size, write: true, masks);
        }
    }

    private void AddRange(Window window, Record record, ulong address, ulong size, bool write, Dictionary<(ulong, bool), PageMask> masks)
    {
        if (size > HugeRangeBytes)
        {
            window.HugeRanges++;
            return;
        }

        var end = address + size;
        for (var page = address & ~(PageBytes - 1); page < end; page += PageBytes)
        {
            ref var state = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrNullRef(_pages, page);
            if (System.Runtime.CompilerServices.Unsafe.IsNullRef(ref state))
            {
                if (window.ArmedPages >= _maxArmedPages)
                {
                    window.CapSkipped++;
                    continue;
                }

                state = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(_pages, page, out _);
                _anyPages = true;
            }

            if (!ReferenceEquals(state.Window, window))
            {
                state.Window = window;
                state.Flags &= WriteWatched | ReadWatched | CommandBuffer;
            }

            state.Flags |= write ? InWrite : InRead;
            if (record.Serial > state.MaxSerial)
            {
                state.MaxSerial = record.Serial;
                record.OwnedPages.Add(page);
            }

            if ((state.Flags & Faulted) == 0)
            {
                Watch(window, page, ref state, masks, count: true);
            }
        }
    }

    private void Watch(Window window, ulong page, ref PageState state, Dictionary<(ulong, bool), PageMask> masks, bool count)
    {
        var wasWatched = (state.Flags & (WriteWatched | ReadWatched)) != 0;
        if ((state.Flags & InWrite) != 0 && _armReads && (state.Flags & ReadWatched) == 0)
        {
            if (_watches.IsReadWatched(page))
            {
                if (!wasWatched && count)
                {
                    window.SkippedReadWatched++;
                }
            }
            else
            {
                state.Flags |= ReadWatched;
                if (count)
                {
                    window.ReadArmedPages++;
                }

                SetMask(masks, page, reads: true);
            }
        }

        if ((state.Flags & (WriteWatched | ReadWatched)) == 0)
        {
            state.Flags |= WriteWatched;
            SetMask(masks, page, reads: false);
        }

        if (!wasWatched)
        {
            // A page watched again starts a new count of answered retries.
            _retries.Remove(page);
            if (count)
            {
                window.ArmedPages++;
            }
        }
    }

    private void ArmCommandBufferLocked(Window window, ulong address, ulong bytes)
    {
        var masks = new Dictionary<(ulong, bool), PageMask>();
        for (var page = address & ~(PageBytes - 1); page < address + bytes; page += PageBytes)
        {
            ref var state = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(_pages, page, out _);
            _anyPages = true;
            state.Flags |= CommandBuffer;
            if ((state.Flags & (WriteWatched | ReadWatched)) == 0)
            {
                if (!ReferenceEquals(state.Window, window))
                {
                    state.Window = window;
                    state.Flags &= CommandBuffer;
                }

                state.Flags = (byte)((state.Flags & ~Faulted) | WriteWatched);
                _retries.Remove(page);
                window.CommandBufferPages++;
                SetMask(masks, page, reads: false);
            }
        }

        Apply(masks, add: true);
    }

    private static void Unwatch(ulong page, ref PageState state, Dictionary<(ulong, bool), PageMask> masks)
    {
        if ((state.Flags & WriteWatched) != 0)
        {
            SetMask(masks, page, reads: false);
        }

        if ((state.Flags & ReadWatched) != 0)
        {
            SetMask(masks, page, reads: true);
        }

        state.Flags &= unchecked((byte)~(WriteWatched | ReadWatched));
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
            ref var state = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrNullRef(_pages, page);
            if (System.Runtime.CompilerServices.Unsafe.IsNullRef(ref state))
            {
                return false;
            }

            var ours = kind == FaultKind.Read ? (state.Flags & ReadWatched) != 0 : (state.Flags & (ReadWatched | WriteWatched)) != 0;
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

                    _declined++;
                    Console.Error.WriteLine($"[METAL-V2][OVERLAP-INFLIGHT][WARN] page 0x{page:X}: {MaxRetriesPerPage} retries answered since it was armed, declining the fault");
                }

                return false;
            }

            var preRelease = false;
            if (state.Window is { } window)
            {
                preRelease = Attribute(window, ref state, address, write: kind != FaultKind.Read, now);
                if (preRelease)
                {
                    window.PreReleasePages.Add(page);
                }
            }

            state.Flags |= Faulted;
            var masks = new Dictionary<(ulong, bool), PageMask>();
            Unwatch(page, ref state, masks);
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
                if (_pages.TryGetValue(page, out var state) && (state.Flags & (WriteWatched | ReadWatched)) != 0 && state.Window is { } window)
                {
                    window.Managed++;
                }
            }
        }
    }

    internal int Era(Window window, ulong serial)
    {
        if (window.PresentSerial == 0)
        {
            return 3;
        }

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

    private static void BuildIndex(Record record)
    {
        if (record.ReadIndex is not null)
        {
            return;
        }

        static ((ulong, ulong, int)[], ulong[]) Index(IEnumerable<(ulong Start, ulong End, int Index)> ranges)
        {
            var sorted = ranges.OrderBy(range => range.Start).ToArray();
            var maxEnd = new ulong[sorted.Length];
            ulong running = 0;
            for (var index = 0; index < sorted.Length; index++)
            {
                running = Math.Max(running, sorted[index].End);
                maxEnd[index] = running;
            }

            return (sorted, maxEnd);
        }

        (record.ReadIndex, record.ReadMaxEnd) = Index(record.Reads.Select((read, index) => (read.Address, read.Address + read.Size, index)));
        (record.WriteIndex, record.WriteMaxEnd) = Index(record.Writes.Select((write, index) => (write.Address, write.Address + write.Size, index)));
    }

    // The newest range of the record that holds the byte, or -1.
    private static int Find((ulong Start, ulong End, int Index)[] sorted, ulong[] maxEnd, ulong address)
    {
        int low = 0, high = sorted.Length - 1, last = -1;
        while (low <= high)
        {
            var middle = (low + high) >> 1;
            if (sorted[middle].Start <= address)
            {
                last = middle;
                low = middle + 1;
            }
            else
            {
                high = middle - 1;
            }
        }

        for (var index = last; index >= 0 && maxEnd[index] > address; index--)
        {
            if (sorted[index].End > address)
            {
                return sorted[index].Index;
            }
        }

        return -1;
    }

    // Returns true when the fault came before the buffer it hit was published (the page is armed again then).
    private bool Attribute(Window window, ref PageState state, ulong address, bool write, long now)
    {
        var ms = window.GateTicks == 0 ? double.NaN : (now - window.GateTicks) * 1000.0 / Stopwatch.Frequency;
        Record? readHit = null, writeHit = null;
        int readIndex = -1, writeIndex = -1;
        foreach (var record in window.Records)
        {
            if (!record.Armed)
            {
                continue;
            }

            BuildIndex(record);
            if (readHit is null || record.Serial > readHit.Serial)
            {
                var found = Find(record.ReadIndex!, record.ReadMaxEnd!, address);
                if (found >= 0 && (readHit is null || record.Serial < readHit.Serial || readHit.Serial <= _completedSerial))
                {
                    readHit = record;
                    readIndex = found;
                }
            }

            var foundWrite = Find(record.WriteIndex!, record.WriteMaxEnd!, address);
            if (foundWrite >= 0 && (writeHit is null || writeHit.Serial <= _completedSerial))
            {
                writeHit = record;
                writeIndex = foundWrite;
            }
        }

        var owner = write ? readHit ?? writeHit : writeHit;
        if (owner is null && write && (state.Flags & CommandBuffer) != 0)
        {
            foreach (var (bufferAddress, bytes) in _admitted)
            {
                if (address >= bufferAddress && address - bufferAddress < bytes)
                {
                    window.CbHit++;
                    window.CbMs.Add(ms);
                    Report(window, "store_cb", address, bufferAddress, bytes, "command_buffer", 0, 8, ms, 0);
                    return false;
                }
            }

            window.CbNeighbour++;
            return false;
        }

        var serial = owner?.Serial ?? state.MaxSerial;
        if (serial <= _completedSerial)
        {
            window.AfterCompletion++;
            return false;
        }

        if (serial > _releasedSerial)
        {
            if (write)
            {
                window.PreReleaseStores++;
                if (readHit is not null)
                {
                    window.PreStoreByTag[(int)readHit.Reads[readIndex].Tag]++;
                }
            }
            else
            {
                window.PreReleaseLoads++;
            }

            return true;
        }

        if (write)
        {
            if (readHit is not null)
            {
                var read = readHit.Reads[readIndex];
                window.RsHit++;
                window.RsHitByTag[(int)read.Tag]++;
                var era = Era(window, readHit.Serial);
                window.RsHitByEra[era]++;
                window.RsHitRanges.Add((read.Address, read.Size));
                window.RsHitMs.Add(ms);
                Report(window, "store_rs", address, read.Address, read.Size, TagNames[(int)read.Tag], readHit.Serial, era, ms, 0);
                return false;
            }

            if (writeHit is not null)
            {
                var written = writeHit.Writes[writeIndex];
                window.WsStoreHit++;
                window.WsStoreByOrigin[written.Origin]++;
                Report(window, "store_ws", address, written.Address, written.Size, OriginNames[written.Origin] + "/" + TagNames[(int)written.Tag],
                    writeHit.Serial, Era(window, writeHit.Serial), ms, written.Program);
                return false;
            }

            if ((state.Flags & InRead) != 0) window.RsNeighbour++;
            else window.WsStoreNeighbour++;
            return false;
        }

        if (writeHit is not null)
        {
            var written = writeHit.Writes[writeIndex];
            window.WsLoadHit++;
            window.WsLoadByOrigin[written.Origin]++;
            window.WsLoadByTag[(int)written.Tag]++;
            var era = Era(window, writeHit.Serial);
            window.WsLoadByEra[era]++;
            window.WsLoadRanges.Add((written.Address, written.Size));
            window.WsLoadMs.Add(ms);
            Report(window, "load_ws", address, written.Address, written.Size, OriginNames[written.Origin] + "/" + TagNames[(int)written.Tag],
                writeHit.Serial, era, ms, written.Program);
            return false;
        }

        window.WsLoadNeighbour++;
        return false;
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
            $"serial_vs_present={offset} era={EraNames[era]} ms_after_gate={ms:F2} completed_vs_present={(long)_completedSerial - (long)window.PresentSerial} program=0x{program:X16}"));
    }

    // ---- windows ------------------------------------------------------------------------------

    private void FinalizeIfDoneLocked()
    {
        for (var index = 0; index < _windows.Count;)
        {
            var window = _windows[index];
            if (!window.Closed || window.Records.Any(record => !record.Disarmed))
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
    // release has been retried by then, so they are forgotten.
    private void PruneLocked(long now)
    {
        var stale = new List<ulong>();
        foreach (var pair in _pages)
        {
            var state = pair.Value;
            if ((state.Flags & (ReadWatched | WriteWatched | CommandBuffer)) == 0 &&
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

    /// <summary>One line per stats window: per sampled frame, mean/max and time percentiles from the gate.</summary>
    public string? TakeWindowLine()
    {
        var windows = TakeFinished();
        long late, declined;
        lock (_gate)
        {
            late = _lateFaults;
            declined = _declined;
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

        string P(Func<Window, List<double>> pick) =>
            FrameOverlapTracker.Percentiles(windows.SelectMany(pick).Where(value => !double.IsNaN(value)).ToList(), culture);

        string FirstP(Func<Window, List<double>> pick) =>
            FrameOverlapTracker.Percentiles(windows.Where(w => pick(w).Count != 0).Select(w => pick(w).Min()).Where(value => !double.IsNaN(value)).ToList(), culture);

        return string.Create(culture,
            $"[METAL-V2][OVERLAP-INFLIGHT] frames={n} armed_pages={M(w => w.ArmedPages)} read_armed={M(w => w.ReadArmedPages)} " +
            $"cb_pages={M(w => w.CommandBufferPages)} buffers={M(w => w.Records.Count)} late_buffers={M(w => w.LateRecords)} " +
            $"arm_ms={M(w => w.ArmTicks * 1000.0 / Stopwatch.Frequency)} cap_skipped={M(w => w.CapSkipped)} skipped_read_watched={M(w => w.SkippedReadWatched)} " +
            $"huge={M(w => w.HugeRanges)} rearmed={M(w => w.Rearmed)} " +
            $"| pre_release stores={M(w => w.PreReleaseStores)} by_tag=[{Split(w => w.PreStoreByTag, TagNames)}] loads={M(w => w.PreReleaseLoads)} after_completion={M(w => w.AfterCompletion)} " +
            $"| 1a store_rs_hit_pages={M(w => w.RsHit)} ranges={M(w => w.RsHitRanges.Count)} hit_range_mb={Mb(w => w.RsHitRanges)} " +
            $"nb_pages={M(w => w.RsNeighbour)} by_tag=[{Split(w => w.RsHitByTag, TagNames)}] " +
            $"by_era=[{Split(w => w.RsHitByEra, EraNames)}] first_ms={FirstP(w => w.RsHitMs)} all_ms={P(w => w.RsHitMs)} " +
            $"| 1b store_cb_hit_pages={M(w => w.CbHit)} nb={M(w => w.CbNeighbour)} first_ms={FirstP(w => w.CbMs)} " +
            $"| ws store_hit_pages={M(w => w.WsStoreHit)} nb={M(w => w.WsStoreNeighbour)} by_origin=[{Split(w => w.WsStoreByOrigin, OriginNames)}] " +
            $"| 2 load_ws_hit_pages={M(w => w.WsLoadHit)} ranges={M(w => w.WsLoadRanges.Count)} mb={Mb(w => w.WsLoadRanges)} nb_pages={M(w => w.WsLoadNeighbour)} " +
            $"by_origin=[{Split(w => w.WsLoadByOrigin, OriginNames)}] by_tag=[{Split(w => w.WsLoadByTag, TagNames)}] by_era=[{Split(w => w.WsLoadByEra, EraNames)}] " +
            $"first_ms={FirstP(w => w.WsLoadMs)} all_ms={P(w => w.WsLoadMs)} | managed={M(w => w.Managed)} late_faults={late} declined={declined}");
    }

    public void Dispose()
    {
        lock (_requests)
        {
            _stopping = true;
            Monitor.PulseAll(_requests);
        }
    }
}
