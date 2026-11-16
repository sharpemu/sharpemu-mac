// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace SharpEmu.HLE.GpuMemory;

/// <summary>How the GPU work of a frame reads a guest range.</summary>
public enum OverlapRead
{
    /// <summary>A binding resolved in a window: the GPU reads (or writes) the guest bytes when it runs.</summary>
    InPlace,

    /// <summary>Copied into the upload ring at record time: the guest may rewrite it at once.</summary>
    Staged,

    /// <summary>The guest source of an image upload, read from its window when the blit runs.</summary>
    Upload,

    /// <summary>An evaluated range of a device-address program (the program may reach further).</summary>
    DeviceAddress,
}

/// <summary>
/// Who writes a guest range for the frame: the first eight are MetalGpuWriteOrigin in its order
/// (taken where each write is recorded, before any folding), then the command-processor and
/// end-of-pipe destinations as their packets are recorded.
/// </summary>
public enum OverlapWrite
{
    Storage,
    Binding,
    DeviceAddress,
    Fill,
    Copy,
    Writeback,
    EopLabel,
    Unbounded,
    CpWrite,
    EopDestination,
}

/// <summary>The page watches the diagnostic arms; a seam so the bookkeeping runs without a guard.</summary>
public interface IFrameOverlapWatches
{
    /// <summary>True when something already holds the page's one read-watch bit.</summary>
    bool IsReadWatched(ulong page);

    void Add(ulong blockBase, in PageMask pages, bool blockReads);

    void Remove(ulong blockBase, in PageMask pages, bool blockReads);

    /// <summary>True when the page still stops this access after the diagnostic let go of it.</summary>
    bool StillBlocks(ulong page, FaultKind kind);
}

/// <summary>The watches of a real page guard.</summary>
public sealed class PageGuardOverlapWatches(PageGuard guard) : IFrameOverlapWatches
{
    // Only a read watch takes read access away (PageGuard.PageCounts.GetAllowedAccess).
    public bool IsReadWatched(ulong page) => !guard.Allows(page, FaultKind.Read);

    public void Add(ulong blockBase, in PageMask pages, bool blockReads) => guard.AddWatchMask(blockBase, pages, blockReads);

    public void Remove(ulong blockBase, in PageMask pages, bool blockReads) => guard.RemoveWatchMask(blockBase, pages, blockReads);

    public bool StillBlocks(ulong page, FaultKind kind) => !guard.Allows(page, kind);
}

/// <summary>
/// SHARPEMU_DIAG_FRAME_OVERLAP=1 (diagnostic, measurement only): what would stand in the way of
/// letting the guest build frame N+1 while the GPU still runs frame N. Per frame it records the
/// guest pages N's GPU work reads in place and writes, the time from "N walked and committed" to
/// "N completed" (the overlap there is to win), the submissions still unfinished at N's boundary,
/// the packets an overlapped frame would have to refuse, and, on every Kth frame, which of those
/// pages the guest touches while it builds N+1: it arms page watches when N completes and drops
/// them at N+1's boundary, counts each page's first fault and lets it go at once. Off, every
/// entry point is a static readonly check.
/// </summary>
public static class FrameOverlapDiagnostic
{
    public static readonly bool Enabled =
        Environment.GetEnvironmentVariable("SHARPEMU_DIAG_FRAME_OVERLAP") == "1";

    /// <summary>SHARPEMU_DIAG_FRAME_OVERLAP_EVERY: arm the watches on every Kth frame (default 4).</summary>
    public static readonly int SampleEvery =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_DIAG_FRAME_OVERLAP_EVERY"), out var every) && every >= 1 ? every : 4;

    /// <summary>SHARPEMU_DIAG_FRAME_OVERLAP_MAX_PAGES: pages armed per sampled frame at most (default 65536 = 256 MiB).</summary>
    public static readonly int MaxArmedPages =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_DIAG_FRAME_OVERLAP_MAX_PAGES"), out var cap) && cap >= 1 ? cap : 65536;

    /// <summary>SHARPEMU_DIAG_FRAME_OVERLAP_READS=0: arm write watches only (no read traps on the write set).</summary>
    public static readonly bool ArmReads =
        Environment.GetEnvironmentVariable("SHARPEMU_DIAG_FRAME_OVERLAP_READS") != "0";

    private static FrameOverlapTracker? _tracker;

    /// <summary>
    /// The guest process of the two-process split: its read faults go to the GPU process, which
    /// holds the watches. Set only while the diagnostic is on; must not allocate or lock.
    /// </summary>
    public static Func<ulong, bool>? RemoteReadFault { get; set; }

    public static FrameOverlapTracker? Tracker => Volatile.Read(ref _tracker);

    /// <summary>The GPU host that records frames attaches its page guard (null detaches).</summary>
    public static void Attach(PageGuard? guard, bool readWatchesShared)
    {
        if (!Enabled)
        {
            return;
        }

        var previous = Interlocked.Exchange(ref _tracker, guard is null
            ? null
            : new FrameOverlapTracker(new PageGuardOverlapWatches(guard), Stopwatch.GetTimestamp, SampleEvery, MaxArmedPages,
                ArmReads && !readWatchesShared, startThread: true));
        previous?.Dispose();
        if (guard is not null)
        {
            Console.Error.WriteLine(
                $"[METAL-V2][OVERLAP-DIAG] on: every={SampleEvery} max_pages={MaxArmedPages} " +
                $"read_watches={(ArmReads && !readWatchesShared ? 1 : 0)}" +
                (ArmReads && readWatchesShared ? " (off: single-copy holds the read-watch bit)" : string.Empty));
        }
    }

    public static void MarkGpuRead(ulong address, ulong size, OverlapRead kind) => Tracker?.MarkRead(address, size, kind);

    public static void NoteGpuWrite(ulong address, ulong size, OverlapWrite origin) => Tracker?.NoteWrite(address, size, origin);

    public static void NoteDeviceAddressProgram(bool planned) => Tracker?.NoteDeviceAddressProgram(planned);

    /// <summary>A packet an overlapped frame could not publish early: GDS end-of-pipe store or a GPU-valued (clock) label.</summary>
    public static void NoteRefusal(bool gds) => Tracker?.NoteRefusal(gds);

    public static void NoteEndOfPipeAtomicAdd(ulong address) => Tracker?.NoteAtomicAdd(address);

    public static void OnAdmission(ulong address, ulong bytes) => Tracker?.OnAdmission(address, bytes);

    public static void OnBoundaryAdmitted(int unfinished, int blocked) => Tracker?.OnBoundaryAdmitted(unfinished, blocked);

    public static void OnBoundaryWalked() => Tracker?.OnBoundaryWalked();

    public static void OnFrameCommitted(ulong lastSerial) => Tracker?.OnFrameCommitted(lastSerial);

    public static void OnCompleted(ulong serial) => Tracker?.OnCompleted(serial);

    public static void NoteManagedWrite(ulong address, ulong size) => Tracker?.NoteManagedWrite(address, size);

    /// <summary>
    /// A fault on a page the diagnostic armed: counted and released. <paramref name="stillBlocked"/>
    /// says another watcher still stops the access, whose own path must then run as well.
    /// </summary>
    public static bool TryHandleFault(FaultKind kind, ulong address, out bool stillBlocked)
    {
        stillBlocked = false;
        return Tracker is { } tracker && tracker.TryHandleFault(kind, address, out stillBlocked);
    }

    public static string? TakeWindowLine() => Tracker?.TakeWindowLine();
}

/// <summary>The guest pages of one frame, one bit per 4 KiB page in 4 MiB blocks.</summary>
internal sealed class OverlapPageSet
{
    private const int BlockShift = 22;
    private const int PageShift = 12;
    private const int WordsPerBlock = TrackerLayout.PagesPerBlock / 64;
    private readonly Dictionary<ulong, ulong[]> _blocks = new();

    public long Count { get; private set; }

    public IEnumerable<KeyValuePair<ulong, ulong[]>> Blocks => _blocks;

    /// <summary>Sets the pages of the range; returns how many were new.</summary>
    public long Add(ulong address, ulong size)
    {
        if (size == 0 || size > ulong.MaxValue - address)
        {
            return 0;
        }

        long added = 0;
        var first = address >> PageShift;
        var last = (address + size - 1) >> PageShift;
        for (var page = first; page <= last; page++)
        {
            var block = page >> (BlockShift - PageShift);
            if (!_blocks.TryGetValue(block, out var words))
            {
                words = new ulong[WordsPerBlock];
                _blocks.Add(block, words);
            }

            var index = (int)(page & ((1UL << (BlockShift - PageShift)) - 1));
            var bit = 1UL << (index & 63);
            if ((words[index >> 6] & bit) == 0)
            {
                words[index >> 6] |= bit;
                added++;
            }
        }

        Count += added;
        return added;
    }

    /// <summary>The block's page bits (block = address &gt;&gt; 22), or null when none is set.</summary>
    public ulong[]? Words(ulong block) => _blocks.TryGetValue(block, out var words) ? words : null;

    public bool Contains(ulong address)
    {
        var page = address >> PageShift;
        if (!_blocks.TryGetValue(page >> (BlockShift - PageShift), out var words))
        {
            return false;
        }

        var index = (int)(page & ((1UL << (BlockShift - PageShift)) - 1));
        return (words[index >> 6] & (1UL << (index & 63))) != 0;
    }

    public bool ContainsAny(ulong address, ulong size)
    {
        if (size == 0 || size > ulong.MaxValue - address)
        {
            return false;
        }

        for (var page = address & ~0xFFFUL; page < address + size; page += 0x1000)
        {
            if (Contains(page))
            {
                return true;
            }
        }

        return false;
    }
}

/// <summary>The state of the diagnostic; separate from the static switch so a test can drive it.</summary>
public sealed class FrameOverlapTracker : IDisposable
{
    // A single range above this is counted, not walked page by page (a whole window or table).
    internal const ulong HugeRangeBytes = 256UL << 20;
    private const ulong PageBytes = TrackerLayout.PageBytes;
    private const int OriginCount = (int)OverlapWrite.EopDestination + 1;
    private static readonly string[] OriginNames =
        ["storage", "binding", "device_address", "fill", "copy", "writeback", "eop_label", "unbounded", "cp_write", "eop_dest"];

    // Page flags of an armed frame.
    private const byte InRead = 1;
    private const byte InWrite = 2;
    private const byte DeviceAddressOnly = 4;
    private const byte InCommandBuffer = 8;
    private const byte ReadArmed = 16;
    private const byte Armed = 32;
    private const byte ManagedSeen = 64;

    private readonly IFrameOverlapWatches _watches;
    private readonly Func<long> _clock;
    private readonly int _sampleEvery;
    private readonly int _maxArmedPages;
    private readonly bool _armReads;
    private readonly object _gate = new();
    private readonly object _watchGate = new();
    private readonly Queue<(bool Arm, Frame Frame)> _requests = new();
    private readonly Thread? _thread;
    private bool _stopping;

    // Recording side (worker).
    private Frame _recording;
    private long _recordedFrames;

    // Admission side (guest threads, under the command stream's lock).
    private Admission _admitting = new();
    private readonly Dictionary<long, Admission> _admissions = new();
    private long _admittedFrames;

    // Committed, not yet finalized, oldest first.
    private readonly List<Frame> _inFlight = new();
    private ulong _completedSerial;

    // The frame whose pages are armed.
    private Frame? _armed;

    private readonly Window _window = new();
    private readonly Dictionary<ulong, (ulong Size, long Hits)> _topDetach = new();
    private readonly Dictionary<ulong, (int Origin, long Hits)> _topWaits = new();
    private long _windows;

    public FrameOverlapTracker(IFrameOverlapWatches watches, Func<long> clock, int sampleEvery, int maxArmedPages, bool armReads, bool startThread)
    {
        _watches = watches;
        _clock = clock;
        _sampleEvery = Math.Max(1, sampleEvery);
        _maxArmedPages = Math.Max(1, maxArmedPages);
        _armReads = armReads;
        _recording = new Frame(0);
        if (startThread)
        {
            _thread = new Thread(Run) { IsBackground = true, Name = "SharpEmu overlap diagnostic" };
            _thread.Start();
        }
    }

    internal sealed class Frame(long index)
    {
        public readonly long Index = index;
        public readonly OverlapPageSet ReadInPlace = new();
        public readonly OverlapPageSet Staged = new();
        public readonly OverlapPageSet DeviceAddress = new();
        public readonly OverlapPageSet Written = new();
        public readonly OverlapPageSet[] ByOrigin = Enumerable.Range(0, OriginCount).Select(_ => new OverlapPageSet()).ToArray();
        public readonly long[] OriginBytes = new long[OriginCount];
        public readonly List<(ulong Address, ulong Size)> ReadRanges = new();
        public readonly List<ulong> Labels = new();
        public readonly List<ulong> AtomicTargets = new();
        public long InPlaceBytes, StagedBytes, UploadBytes, DeviceAddressBytes, HugeRanges;
        public int DaPlanned, DaUnplanned, GdsStores, GpuValued, Unbounded;
        public ulong LastSerial;
        public long CommittedTicks, CompletedTicks, FirstAdmitTicks, ArmStartTicks, ArmedTicks, FirstFaultTicks;
        public bool Sampled, Finalized;

        // Trap results (under the watch gate).
        public readonly Dictionary<ulong, byte> Pages = new();
        public readonly OverlapPageSet ReadWrittenPages = new();
        public int ArmedPages, ReadArmedPages, SkippedReadWatched, CapSkipped;
        public int ReadSetWritten, WriteSetRead, WriteSetWritten, LabelTouched, CommandBufferTouched, DeviceAddressOnlyWritten, ManagedRead, ManagedWrite;
    }

    private sealed class Admission
    {
        public readonly List<(ulong Address, ulong Bytes)> Submissions = new();
        public long BoundaryTicks, WalkedTicks;
        public int Unfinished, Blocked;
    }

    private sealed class Window
    {
        public int Frames, Sampled, CapFrames, UnfinishedFrames, BlockedFrames, RefusedFrames, TrapBeforeSubmitFrames, WriteSetReadFrames, LateArms;
        public readonly List<double> InPlacePages = new(), InPlaceMb = new(), StagedMb = new(), UploadMb = new(), DaPages = new(),
            WrittenPages = new(), LabelsInCb = new(), DrainMs = new(), OverlapMs = new(), CompleteToAdmitMs = new(), ArmMs = new(),
            ArmedPages = new(), RsWritten = new(), DetachMb = new(), WsRead = new(), WsWritten = new(), LabelTouched = new(), CbTouched = new(),
            DaOnly = new(), Managed = new(), FirstFaultMs = new();
        public readonly double[] OriginPages = new double[OriginCount];
        public long DaPlanned, DaUnplanned, Gds, GpuValued, Unbounded, AtomicInWrites, Huge, Skipped, Faults;
    }

    // ---- recording side -----------------------------------------------------------------

    public void MarkRead(ulong address, ulong size, OverlapRead kind)
    {
        if (size == 0 || size > ulong.MaxValue - address)
        {
            return;
        }

        lock (_gate)
        {
            var frame = _recording;
            if (size > HugeRangeBytes)
            {
                frame.HugeRanges++;
                return;
            }

            switch (kind)
            {
                case OverlapRead.Staged:
                    frame.StagedBytes += (long)size;
                    frame.Staged.Add(address, size);
                    break;
                case OverlapRead.DeviceAddress:
                    frame.DeviceAddressBytes += (long)size;
                    frame.DeviceAddress.Add(address, size);
                    break;
                default:
                    if (kind == OverlapRead.Upload)
                    {
                        frame.UploadBytes += (long)size;
                    }

                    frame.InPlaceBytes += (long)size;
                    frame.ReadInPlace.Add(address, size);
                    frame.ReadRanges.Add((address, size));
                    break;
            }
        }
    }

    public void NoteWrite(ulong address, ulong size, OverlapWrite origin)
    {
        lock (_gate)
        {
            var frame = _recording;
            if (origin == OverlapWrite.Unbounded)
            {
                frame.Unbounded++;
                return;
            }

            if (size == 0 || size > ulong.MaxValue - address)
            {
                return;
            }

            if (size > HugeRangeBytes)
            {
                frame.HugeRanges++;
                return;
            }

            frame.OriginBytes[(int)origin] += (long)size;
            frame.ByOrigin[(int)origin].Add(address, size);
            frame.Written.Add(address, size);
            if (origin is OverlapWrite.CpWrite or OverlapWrite.EopDestination)
            {
                frame.Labels.Add(address);
            }
        }
    }

    public void NoteDeviceAddressProgram(bool planned)
    {
        lock (_gate)
        {
            if (planned) _recording.DaPlanned++;
            else _recording.DaUnplanned++;
        }
    }

    public void NoteRefusal(bool gds)
    {
        lock (_gate)
        {
            if (gds) _recording.GdsStores++;
            else _recording.GpuValued++;
        }
    }

    public void NoteAtomicAdd(ulong address)
    {
        lock (_gate)
        {
            _recording.AtomicTargets.Add(address);
        }
    }

    /// <summary>The worker committed everything the frame recorded; <paramref name="lastSerial"/> finishes it.</summary>
    public void OnFrameCommitted(ulong lastSerial)
    {
        var now = _clock();
        lock (_gate)
        {
            var frame = _recording;
            frame.LastSerial = lastSerial;
            frame.CommittedTicks = now;
            frame.Sampled = frame.Index % _sampleEvery == 0;
            _inFlight.Add(frame);
            _recording = new Frame(++_recordedFrames);
            CompleteLocked(now);
        }
    }

    public void OnCompleted(ulong serial)
    {
        if (serial <= Volatile.Read(ref _completedSerial))
        {
            return;
        }

        var now = _clock();
        lock (_gate)
        {
            if (serial <= _completedSerial)
            {
                return;
            }

            _completedSerial = serial;
            CompleteLocked(now);
        }
    }

    private void CompleteLocked(long now)
    {
        foreach (var frame in _inFlight)
        {
            if (frame.CompletedTicks != 0 || frame.LastSerial > _completedSerial)
            {
                continue;
            }

            frame.CompletedTicks = now;
            if (frame.Sampled)
            {
                Post(arm: true, frame);
            }
        }
    }

    // ---- admission side -----------------------------------------------------------------

    public void OnAdmission(ulong address, ulong bytes)
    {
        var now = _clock();
        lock (_gate)
        {
            _admitting.Submissions.Add((address, bytes));
            foreach (var frame in _inFlight)
            {
                if (frame.CompletedTicks != 0 && frame.FirstAdmitTicks == 0)
                {
                    frame.FirstAdmitTicks = now;
                }
            }
        }
    }

    /// <summary>The guest ended a frame (its boundary was admitted) with these submissions still to walk.</summary>
    public void OnBoundaryAdmitted(int unfinished, int blocked)
    {
        var now = _clock();
        lock (_gate)
        {
            var admission = _admitting;
            admission.BoundaryTicks = now;
            admission.Unfinished = unfinished;
            admission.Blocked = blocked;
            _admissions[_admittedFrames] = admission;
            _admissions.Remove(_admittedFrames - 8);
            _admittedFrames++;
            _admitting = new Admission();

            // The frames that completed before this boundary: the guest's build of the next one is over.
            for (var index = 0; index < _inFlight.Count;)
            {
                var frame = _inFlight[index];
                if (frame.CompletedTicks != 0 && frame.CompletedTicks <= now && frame.Index < _admittedFrames - 1)
                {
                    _inFlight.RemoveAt(index);
                    Post(arm: false, frame);
                    continue;
                }

                index++;
            }
        }
    }

    /// <summary>The command stream reached the boundary: everything admitted before it has been walked.</summary>
    public void OnBoundaryWalked()
    {
        var now = _clock();
        lock (_gate)
        {
            // The oldest admitted boundary not yet walked.
            foreach (var pair in _admissions.OrderBy(pair => pair.Key))
            {
                if (pair.Value.WalkedTicks == 0)
                {
                    pair.Value.WalkedTicks = now;
                    break;
                }
            }
        }
    }

    // ---- arming -------------------------------------------------------------------------

    private void Post(bool arm, Frame frame)
    {
        if (_thread is null)
        {
            // Synchronous (tests): run it after the caller's lock is released by queueing and draining.
            _requests.Enqueue((arm, frame));
            return;
        }

        lock (_requests)
        {
            _requests.Enqueue((arm, frame));
            Monitor.Pulse(_requests);
        }
    }

    /// <summary>Runs the queued arm and disarm requests on the calling thread (tests, no thread).</summary>
    internal void DrainRequests()
    {
        while (true)
        {
            (bool Arm, Frame Frame) request;
            lock (_gate)
            {
                if (_requests.Count == 0)
                {
                    return;
                }

                request = _requests.Dequeue();
            }

            Process(request.Arm, request.Frame);
        }
    }

    private void Run()
    {
        while (true)
        {
            (bool Arm, Frame Frame) request;
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
                Process(request.Arm, request.Frame);
            }
            catch (Exception exception)
            {
                Console.Error.WriteLine($"[METAL-V2][OVERLAP-DIAG][ERROR] {(request.Arm ? "arm" : "disarm")} frame {request.Frame.Index}: {exception}");
            }
        }
    }

    private void Process(bool arm, Frame frame)
    {
        if (arm)
        {
            if (Volatile.Read(ref _armed) is { } previous)
            {
                // A frame armed past the next frame's completion: no boundary came between.
                Disarm(previous);
            }

            Arm(frame);
            return;
        }

        if (ReferenceEquals(Volatile.Read(ref _armed), frame))
        {
            Disarm(frame);
        }

        FinalizeFrame(frame);
    }

    internal void Arm(Frame frame)
    {
        frame.ArmStartTicks = _clock();
        Admission? admission;
        lock (_gate)
        {
            _admissions.TryGetValue(frame.Index, out admission);
        }

        var submissions = admission?.Submissions.ToArray() ?? [];
        var blocks = new SortedSet<ulong>();
        foreach (var set in new[] { frame.ReadInPlace, frame.DeviceAddress, frame.Written })
        {
            foreach (var pair in set.Blocks)
            {
                blocks.Add(pair.Key);
            }
        }

        lock (_watchGate)
        {
            Volatile.Write(ref _armed, frame);
            foreach (var block in blocks)
            {
                var blockBase = block << 22;
                var writeMask = default(PageMask);
                var readMask = default(PageMask);
                var anyWrite = false;
                var anyRead = false;
                var readWords = frame.ReadInPlace.Words(block);
                var writtenWords = frame.Written.Words(block);
                var deviceWords = frame.DeviceAddress.Words(block);
                for (var index = 0; index < TrackerLayout.PagesPerBlock; index++)
                {
                    var word = index >> 6;
                    var bit = 1UL << (index & 63);
                    byte flags = 0;
                    if (readWords is not null && (readWords[word] & bit) != 0) flags |= InRead;
                    if (writtenWords is not null && (writtenWords[word] & bit) != 0) flags |= InWrite;
                    if ((flags & InRead) == 0 && deviceWords is not null && (deviceWords[word] & bit) != 0) flags |= DeviceAddressOnly;
                    if (flags == 0)
                    {
                        continue;
                    }

                    var page = blockBase + (ulong)index * PageBytes;
                    if ((flags & InWrite) != 0 && InSubmissions(submissions, page))
                    {
                        flags |= InCommandBuffer;
                    }

                    if (frame.ArmedPages >= _maxArmedPages)
                    {
                        frame.CapSkipped++;
                        continue;
                    }

                    var reads = (flags & InWrite) != 0 && _armReads;
                    if (reads && _watches.IsReadWatched(page))
                    {
                        frame.SkippedReadWatched++;
                        reads = false;
                    }

                    if (reads)
                    {
                        readMask.Set(index);
                        anyRead = true;
                        flags |= ReadArmed;
                        frame.ReadArmedPages++;
                    }
                    else
                    {
                        writeMask.Set(index);
                        anyWrite = true;
                    }

                    frame.Pages[page] = (byte)(flags | Armed);
                    frame.ArmedPages++;
                }

                if (anyWrite) _watches.Add(blockBase, writeMask, blockReads: false);
                if (anyRead) _watches.Add(blockBase, readMask, blockReads: true);
            }

            frame.ArmedTicks = _clock();
        }

        if (frame.CapSkipped != 0)
        {
            Console.Error.WriteLine(
                $"[METAL-V2][OVERLAP-DIAG][WARN] frame {frame.Index}: page cap {_maxArmedPages} hit, {frame.CapSkipped} pages not armed " +
                "(SHARPEMU_DIAG_FRAME_OVERLAP_MAX_PAGES)");
        }
    }

    private static bool InSubmissions((ulong Address, ulong Bytes)[] submissions, ulong page)
    {
        foreach (var submission in submissions)
        {
            if (page < submission.Address + submission.Bytes && submission.Address < page + PageBytes)
            {
                return true;
            }
        }

        return false;
    }

    internal void Disarm(Frame frame)
    {
        lock (_watchGate)
        {
            var byBlock = new Dictionary<(ulong Block, bool Reads), PageMask>();
            foreach (var pair in frame.Pages)
            {
                if ((pair.Value & Armed) == 0)
                {
                    continue;
                }

                var key = (pair.Key & ~(TrackerLayout.BlockBytes - 1), (pair.Value & ReadArmed) != 0);
                byBlock.TryGetValue(key, out var mask);
                mask.Set((int)((pair.Key & (TrackerLayout.BlockBytes - 1)) / PageBytes));
                byBlock[key] = mask;
            }

            foreach (var pair in byBlock)
            {
                var mask = pair.Value;
                _watches.Remove(pair.Key.Block, mask, pair.Key.Reads);
            }

            foreach (var page in frame.Pages.Keys.ToArray())
            {
                frame.Pages[page] = (byte)(frame.Pages[page] & ~Armed);
            }

            if (ReferenceEquals(_armed, frame))
            {
                Volatile.Write(ref _armed, null);
            }
        }
    }

    public bool TryHandleFault(FaultKind kind, ulong address, out bool stillBlocked)
    {
        stillBlocked = false;
        if (Volatile.Read(ref _armed) is null)
        {
            return false;
        }

        var page = address & ~(PageBytes - 1);
        lock (_watchGate)
        {
            if (_armed is not { } frame || !frame.Pages.TryGetValue(page, out var flags) || (flags & Armed) == 0)
            {
                return false;
            }

            var reads = (flags & ReadArmed) != 0;
            var mask = default(PageMask);
            mask.Set((int)((page & (TrackerLayout.BlockBytes - 1)) / PageBytes));
            frame.Pages[page] = (byte)(flags & ~Armed);
            Count(frame, page, flags, write: kind != FaultKind.Read);
            _watches.Remove(page & ~(TrackerLayout.BlockBytes - 1), mask, reads);
            stillBlocked = _watches.StillBlocks(page, kind);
            return true;
        }
    }

    /// <summary>A managed (HLE) write the GPU process heard of: it does not fault, so it is counted here.</summary>
    public void NoteManagedWrite(ulong address, ulong size)
    {
        if (Volatile.Read(ref _armed) is null || size == 0 || size > ulong.MaxValue - address)
        {
            return;
        }

        lock (_watchGate)
        {
            if (_armed is not { } frame)
            {
                return;
            }

            var end = address + size;
            var pages = 0;
            for (var page = address & ~(PageBytes - 1); page < end && pages < 4096; page += PageBytes, pages++)
            {
                if (frame.Pages.TryGetValue(page, out var flags) && (flags & ManagedSeen) == 0)
                {
                    frame.Pages[page] = (byte)(flags | ManagedSeen);
                    if ((flags & InWrite) != 0) frame.ManagedWrite++;
                    else frame.ManagedRead++;
                }
            }
        }
    }

    private void Count(Frame frame, ulong page, byte flags, bool write)
    {
        if (frame.FirstFaultTicks == 0)
        {
            frame.FirstFaultTicks = _clock();
        }

        if ((flags & InWrite) != 0)
        {
            if (write) frame.WriteSetWritten++;
            else frame.WriteSetRead++;
            if (frame.ByOrigin[(int)OverlapWrite.CpWrite].Contains(page) || frame.ByOrigin[(int)OverlapWrite.EopDestination].Contains(page))
            {
                frame.LabelTouched++;
            }

            if ((flags & InCommandBuffer) != 0)
            {
                frame.CommandBufferTouched++;
            }

            lock (_topWaits)
            {
                var origin = 0;
                for (var index = 0; index < OriginCount; index++)
                {
                    if (frame.ByOrigin[index].Contains(page))
                    {
                        origin = index;
                        break;
                    }
                }

                _topWaits[page] = (origin, (_topWaits.TryGetValue(page, out var entry) ? entry.Hits : 0) + 1);
            }
        }

        if ((flags & InRead) != 0 && write)
        {
            frame.ReadSetWritten++;
            frame.ReadWrittenPages.Add(page, PageBytes);
        }

        if ((flags & DeviceAddressOnly) != 0 && write)
        {
            frame.DeviceAddressOnlyWritten++;
        }
    }

    // ---- per-frame results --------------------------------------------------------------

    internal void FinalizeFrame(Frame frame)
    {
        if (frame.Finalized)
        {
            return;
        }

        frame.Finalized = true;
        var frequency = (double)Stopwatch.Frequency;
        double Ms(long from, long to) => from == 0 || to == 0 ? double.NaN : (to - from) * 1000.0 / frequency;

        Admission? admission;
        lock (_gate)
        {
            _admissions.TryGetValue(frame.Index, out admission);
        }

        // Labels and command-processor writes into the frame's own submitted command buffers.
        var labelsInCb = 0;
        if (admission is not null)
        {
            foreach (var label in frame.Labels)
            {
                foreach (var submission in admission.Submissions)
                {
                    if (label >= submission.Address && label - submission.Address < submission.Bytes)
                    {
                        labelsInCb++;
                        break;
                    }
                }
            }
        }

        // Bytes of the bound ranges the guest wrote while it built the next frame: what would be detached.
        long detachBytes = 0;
        if (frame.ReadWrittenPages.Count != 0)
        {
            foreach (var range in frame.ReadRanges)
            {
                if (frame.ReadWrittenPages.ContainsAny(range.Address, range.Size))
                {
                    detachBytes += (long)range.Size;
                    lock (_topDetach)
                    {
                        _topDetach[range.Address] = (range.Size, (_topDetach.TryGetValue(range.Address, out var entry) ? entry.Hits : 0) + 1);
                    }
                }
            }
        }

        var atomicInWrites = 0;
        foreach (var target in frame.AtomicTargets)
        {
            if (frame.Written.Contains(target)) atomicInWrites++;
        }

        lock (_window)
        {
            var window = _window;
            window.Frames++;
            window.InPlacePages.Add(frame.ReadInPlace.Count);
            window.InPlaceMb.Add(frame.ReadInPlace.Count * 4096.0 / 1048576.0);
            window.StagedMb.Add(frame.StagedBytes / 1048576.0);
            window.UploadMb.Add(frame.UploadBytes / 1048576.0);
            window.DaPages.Add(frame.DeviceAddress.Count);
            window.WrittenPages.Add(frame.Written.Count);
            window.LabelsInCb.Add(labelsInCb);
            for (var index = 0; index < OriginCount; index++)
            {
                window.OriginPages[index] += frame.ByOrigin[index].Count;
            }

            window.DaPlanned += frame.DaPlanned;
            window.DaUnplanned += frame.DaUnplanned;
            window.Gds += frame.GdsStores;
            window.GpuValued += frame.GpuValued;
            window.Unbounded += frame.Unbounded;
            window.AtomicInWrites += atomicInWrites;
            window.Huge += frame.HugeRanges;
            if (frame.GdsStores + frame.GpuValued + frame.Unbounded + atomicInWrites != 0) window.RefusedFrames++;
            if (admission is not null)
            {
                if (admission.Unfinished != 0) window.UnfinishedFrames++;
                if (admission.Blocked != 0) window.BlockedFrames++;
                window.DrainMs.Add(Ms(admission.BoundaryTicks, admission.WalkedTicks));
            }

            window.OverlapMs.Add(Ms(frame.CommittedTicks, frame.CompletedTicks));
            window.CompleteToAdmitMs.Add(Ms(frame.CompletedTicks, frame.FirstAdmitTicks));
            if (frame.Sampled && frame.ArmedTicks != 0)
            {
                window.Sampled++;
                if (frame.CapSkipped != 0) window.CapFrames++;
                if (frame.FirstAdmitTicks != 0 && frame.ArmedTicks > frame.FirstAdmitTicks) window.LateArms++;
                window.Skipped += frame.SkippedReadWatched;
                window.ArmMs.Add(Ms(frame.ArmStartTicks, frame.ArmedTicks));
                window.ArmedPages.Add(frame.ArmedPages);
                window.RsWritten.Add(frame.ReadSetWritten);
                window.DetachMb.Add(detachBytes / 1048576.0);
                window.WsRead.Add(frame.WriteSetRead);
                window.WsWritten.Add(frame.WriteSetWritten);
                window.LabelTouched.Add(frame.LabelTouched);
                window.CbTouched.Add(frame.CommandBufferTouched);
                window.DaOnly.Add(frame.DeviceAddressOnlyWritten);
                window.Managed.Add(frame.ManagedRead + frame.ManagedWrite);
                window.Faults += frame.ReadSetWritten + frame.WriteSetRead + frame.WriteSetWritten;
                if (frame.WriteSetRead != 0) window.WriteSetReadFrames++;
                if (frame.FirstFaultTicks != 0)
                {
                    window.FirstFaultMs.Add(Ms(frame.CompletedTicks, frame.FirstFaultTicks));
                    if (frame.FirstAdmitTicks == 0 || frame.FirstFaultTicks < frame.FirstAdmitTicks) window.TrapBeforeSubmitFrames++;
                }
            }
        }
    }

    /// <summary>One window's line, and every 15th window the run's top ranges so far.</summary>
    public string TakeWindowLine()
    {
        var culture = CultureInfo.InvariantCulture;
        var builder = new StringBuilder("[METAL-V2][OVERLAP-DIAG] ");
        lock (_window)
        {
            var w = _window;
            string P(List<double> values) => Percentiles(values, culture);
            string Mx(List<double> values) => Mean(values, culture) + "/" + Max(values, culture);
            builder.Append(culture, $"frames={w.Frames} sampled={w.Sampled} every={_sampleEvery} ");
            builder.Append($"read: inplace_pages={P(w.InPlacePages)} inplace_mb={P(w.InPlaceMb)} staged_mb={P(w.StagedMb)} upload_mb={P(w.UploadMb)} da_pages={P(w.DaPages)} ");
            builder.Append(culture, $"da_enc planned={w.DaPlanned} unplanned={w.DaUnplanned} huge_ranges={w.Huge} | ");
            builder.Append($"write: pages={P(w.WrittenPages)}");
            for (var index = 0; index < OriginCount; index++)
            {
                builder.Append(' ').Append(OriginNames[index]).Append('=')
                    .Append(w.Frames == 0 ? "0" : (w.OriginPages[index] / w.Frames).ToString("F0", culture));
            }

            builder.Append($" labels_in_cb={P(w.LabelsInCb)} | ");
            builder.Append(culture, $"refuse: frames={w.RefusedFrames} gds={w.Gds} gpu_valued={w.GpuValued} unbounded={w.Unbounded} atomic_in_ws={w.AtomicInWrites} | ");
            builder.Append(culture, $"boundary: unfinished_frames={w.UnfinishedFrames} blocked_frames={w.BlockedFrames} drain_ms={P(w.DrainMs)} | ");
            builder.Append($"overlap_ms={P(w.OverlapMs)} complete_to_admit_ms={P(w.CompleteToAdmitMs)} | ");
            builder.Append(culture, $"trap: armed_pages={Mx(w.ArmedPages)} arm_ms={P(w.ArmMs)} late_arms={w.LateArms} cap_frames={w.CapFrames} skipped_read_watched={w.Skipped} ");
            builder.Append($"rs_written={Mx(w.RsWritten)} detach_mb={Mx(w.DetachMb)} ws_read={Mx(w.WsRead)} ws_written={Mx(w.WsWritten)} ");
            builder.Append($"label_touched={Mx(w.LabelTouched)} cb_touched={Mx(w.CbTouched)} da_only_written={Mx(w.DaOnly)} managed={Mx(w.Managed)} first_fault_ms={P(w.FirstFaultMs)} ");
            builder.Append(culture, $"trap_before_submit_frames={w.TrapBeforeSubmitFrames} ws_read_frames={w.WriteSetReadFrames}");
            _window.Frames = 0;
            ResetWindow();
        }

        if (++_windows % 15 == 0)
        {
            builder.Append('\n').Append(TopRanges());
        }

        return builder.ToString();
    }

    internal string TopRanges()
    {
        var builder = new StringBuilder("[METAL-V2][OVERLAP-DIAG-TOP] detach (bound ranges the guest wrote, run so far):");
        lock (_topDetach)
        {
            foreach (var pair in _topDetach.OrderByDescending(pair => pair.Value.Hits * (long)Math.Min(pair.Value.Size, long.MaxValue)).Take(10))
            {
                builder.Append(CultureInfo.InvariantCulture, $" 0x{pair.Key:X}+0x{pair.Value.Size:X}x{pair.Value.Hits}");
            }
        }

        builder.Append(" | wait (GPU-written pages the guest touched):");
        lock (_topWaits)
        {
            foreach (var pair in _topWaits.OrderByDescending(pair => pair.Value.Hits).Take(10))
            {
                builder.Append(CultureInfo.InvariantCulture, $" 0x{pair.Key:X}:{OriginNames[pair.Value.Origin]}x{pair.Value.Hits}");
            }
        }

        return builder.ToString();
    }

    private void ResetWindow()
    {
        var w = _window;
        w.Frames = w.Sampled = w.CapFrames = w.UnfinishedFrames = w.BlockedFrames = w.RefusedFrames = w.TrapBeforeSubmitFrames = 0;
        w.WriteSetReadFrames = w.LateArms = 0;
        w.DaPlanned = w.DaUnplanned = w.Gds = w.GpuValued = w.Unbounded = w.AtomicInWrites = w.Huge = w.Skipped = w.Faults = 0;
        Array.Clear(w.OriginPages);
        foreach (var list in new[] { w.InPlacePages, w.InPlaceMb, w.StagedMb, w.UploadMb, w.DaPages, w.WrittenPages, w.LabelsInCb,
                     w.DrainMs, w.OverlapMs, w.CompleteToAdmitMs, w.ArmMs, w.ArmedPages, w.RsWritten, w.DetachMb, w.WsRead,
                     w.WsWritten, w.LabelTouched, w.CbTouched, w.DaOnly, w.Managed, w.FirstFaultMs })
        {
            list.Clear();
        }
    }

    internal static string Percentiles(List<double> values, IFormatProvider culture)
    {
        var sorted = values.Where(value => !double.IsNaN(value)).OrderBy(value => value).ToArray();
        if (sorted.Length == 0)
        {
            return "-";
        }

        double At(double fraction) => sorted[Math.Min(sorted.Length - 1, (int)Math.Floor(fraction * sorted.Length))];
        return string.Create(culture, $"{At(0.5):0.#}/{At(0.9):0.#}");
    }

    private static string Mean(List<double> values, IFormatProvider culture) =>
        values.Count == 0 ? "-" : values.Average().ToString("0.#", culture);

    private static string Max(List<double> values, IFormatProvider culture) =>
        values.Count == 0 ? "-" : values.Max().ToString("0.#", culture);

    public void Dispose()
    {
        lock (_requests)
        {
            _stopping = true;
            Monitor.PulseAll(_requests);
        }

        if (Volatile.Read(ref _armed) is { } armed)
        {
            Disarm(armed);
        }
    }
}
