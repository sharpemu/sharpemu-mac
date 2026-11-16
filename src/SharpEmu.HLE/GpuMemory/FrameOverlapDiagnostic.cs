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

    /// <summary>An end-of-pipe destination whose value only the GPU knows: a clock sample, a GDS store, an atomic add.</summary>
    EopGpuValued,
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

    /// <summary>
    /// SHARPEMU_DIAG_FRAME_OVERLAP_ARM=commit: arm a sampled frame when its last serial is committed
    /// instead of when it completes. Arming ~200k pages takes longer than the guest's build of the
    /// next frame (r ovl2: 7-9 ms for 64k pages against a boundary ~7.5 ms after completion), so
    /// armed at completion the watches see only the build's tail; the guest waits on the GPU fence
    /// between commit and completion, which leaves the ~38 ms of the GPU tail to arm in.
    /// </summary>
    public static readonly bool ArmAtCommit =
        Environment.GetEnvironmentVariable("SHARPEMU_DIAG_FRAME_OVERLAP_ARM") == "commit";

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
                ArmReads && !readWatchesShared, startThread: true) { ArmAtCommit = ArmAtCommit });
        previous?.Dispose();
        if (guard is not null)
        {
            Console.Error.WriteLine(
                $"[METAL-V2][OVERLAP-DIAG] on: every={SampleEvery} max_pages={MaxArmedPages} " +
                $"read_watches={(ArmReads && !readWatchesShared ? 1 : 0)} arm_at={(ArmAtCommit ? "commit" : "completion")}" +
                (ArmReads && readWatchesShared ? " (off: single-copy holds the read-watch bit)" : string.Empty));
        }
    }

    public static void MarkGpuRead(ulong address, ulong size, OverlapRead kind, ulong program = 0) => Tracker?.MarkRead(address, size, kind, program);

    public static void NoteGpuWrite(ulong address, ulong size, OverlapWrite origin) =>
        Tracker?.NoteWrite(address, size, origin, origin is OverlapWrite.Storage or OverlapWrite.Binding or OverlapWrite.DeviceAddress ? TakeWriter() : 0);

    [ThreadStatic]
    private static ulong t_writer;

    /// <summary>The program whose store the next storage or device-address write note on this thread records (consumed by it).</summary>
    public static void SetWriter(ulong programHash) => t_writer = programHash;

    private static ulong TakeWriter()
    {
        var writer = t_writer;
        t_writer = 0;
        return writer;
    }

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
    private const int OriginCount = (int)OverlapWrite.EopGpuValued + 1;
    private static readonly string[] OriginNames =
        ["storage", "binding", "device_address", "fill", "copy", "writeback", "eop_label", "unbounded", "cp_write", "eop_dest", "eop_gpu_valued"];

    // Writes whose bytes are known when the packet is recorded: an overlapped frame lands them on
    // the CPU at publication, so a guest read of them is not a wait.
    private static bool KnownAtRecord(int origin) =>
        origin is (int)OverlapWrite.EopLabel or (int)OverlapWrite.CpWrite or (int)OverlapWrite.EopDestination;

    /// <summary>What a fault's address meets on its page (the fault reports the first byte touched; the access width is not decoded, so 1 byte is assumed).</summary>
    internal enum Hit
    {
        Neighbour,
        Label,
        GpuValued,
    }

    private sealed class PageReport
    {
        public long Faults;
        public readonly List<string> Ranges = new();
        public readonly List<string> Accesses = new();
    }

    private readonly Dictionary<ulong, PageReport> _pageReports = new();

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

    // The frame disarmed last: a fault that reached the guest's handler before the protection
    // the disarm gave back was applied there is answered once per page (retry), not declined.
    private Frame? _released;
    private long _lateFaults;

    // Retries answered per page: several guest threads can fault on one page at once (r ovl3: six job
    // workers reading one read-watched page; the first released it, the other five were declined and
    // the run died). Bounded so a page some other protection holds cannot loop forever.
    private const int MaxRetriesPerPage = 256;
    private readonly Dictionary<ulong, int> _retries = new();

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

    /// <summary>Arm a sampled frame at its commit rather than at its completion.</summary>
    public bool ArmAtCommit { get; init; }

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
        public readonly List<(ulong Address, ulong Size, ulong Program)> DeviceRanges = new();
        public readonly List<(ulong Address, ulong Size, int Origin, ulong Program)> WriteRanges = new();
        public readonly List<ulong> ReadHitAddresses = new();
        public int RsHit, RsNeighbour, WsReadLabel, WsReadGpu, WsReadNeighbour, WsWriteLabel, WsWriteGpu, WsWriteNeighbour, DaOnlyHit, DaOnlyNeighbour;
        public long FirstRealWaitTicks, FirstK6Ticks;
        public readonly List<ulong> Labels = new();
        public readonly List<ulong> AtomicTargets = new();
        public long InPlaceBytes, StagedBytes, UploadBytes, DeviceAddressBytes, HugeRanges;
        public int DaPlanned, DaUnplanned, GdsStores, GpuValued, Unbounded;
        public ulong LastSerial;
        public long CommittedTicks, CompletedTicks, FirstAdmitTicks, ArmStartTicks, ArmedTicks, FirstFaultTicks;
        public bool Sampled, Finalized, ArmPosted;

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
            DaOnly = new(), Managed = new(), FirstFaultMs = new(),
            RsHit = new(), RsNb = new(), WsRLabel = new(), WsRGpu = new(), WsRNb = new(), WsWLabel = new(), WsWGpu = new(), WsWNb = new(),
            DaHit = new(), DaNb = new(), RealTrap = new(), RealWait = new(), DetachHitMb = new(), K6Ms = new(), RealWaitMs = new(), OverlapUntilWait = new();
        public int ArmedBeforeComplete, K6Frames, K6BeforeSubmit, RealWaitBeforeSubmit;
        public readonly double[] OriginPages = new double[OriginCount];
        public long DaPlanned, DaUnplanned, Gds, GpuValued, Unbounded, AtomicInWrites, Huge, Skipped, Faults;
    }

    // ---- recording side -----------------------------------------------------------------

    public void MarkRead(ulong address, ulong size, OverlapRead kind, ulong program = 0)
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
                    frame.DeviceRanges.Add((address, size, program));
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

    public void NoteWrite(ulong address, ulong size, OverlapWrite origin, ulong program = 0)
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
            frame.WriteRanges.Add((address, size, (int)origin, program));
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
            if (frame.Sampled && ArmAtCommit)
            {
                frame.ArmPosted = true;
                Post(arm: true, frame);
            }

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
            if (frame.Sampled && !frame.ArmPosted)
            {
                frame.ArmPosted = true;
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
            _retries.Clear();
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

            Volatile.Write(ref _released, frame);
        }
    }

    public bool TryHandleFault(FaultKind kind, ulong address, out bool stillBlocked)
    {
        stillBlocked = false;
        if (Volatile.Read(ref _armed) is null && Volatile.Read(ref _released) is null)
        {
            return false;
        }

        var page = address & ~(PageBytes - 1);
        lock (_watchGate)
        {
            if (_armed is not { } frame || !frame.Pages.TryGetValue(page, out var flags) || (flags & Armed) == 0)
            {
                // Released already (by another thread's fault on the page, or by the disarm), but the
                // access faulted on the protection the watch had: once the release is applied the
                // access goes through, so it is retried.
                var owned = (_armed is { } current && current.Pages.ContainsKey(page)) ||
                    (_released is { } released && released.Pages.ContainsKey(page));
                if (owned && !_watches.StillBlocks(page, kind))
                {
                    var retries = _retries.GetValueOrDefault(page);
                    if (retries < MaxRetriesPerPage)
                    {
                        _retries[page] = retries + 1;
                        Interlocked.Increment(ref _lateFaults);
                        return true;
                    }

                    Console.Error.WriteLine($"[METAL-V2][OVERLAP-DIAG][WARN] page 0x{page:X}: {MaxRetriesPerPage} retries answered, declining the fault");
                }

                return false;
            }

            var reads = (flags & ReadArmed) != 0;
            var mask = default(PageMask);
            mask.Set((int)((page & (TrackerLayout.BlockBytes - 1)) / PageBytes));
            frame.Pages[page] = (byte)(flags & ~Armed);
            Count(frame, page, flags, write: kind != FaultKind.Read);
            Attribute(frame, address, flags, write: kind != FaultKind.Read);
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

    /// <summary>Which GPU ranges on the page the faulting address itself lies in.</summary>
    internal static Hit ClassifyWrite(List<(ulong Address, ulong Size, int Origin, ulong Program)> ranges, ulong address, out int origin, out ulong program)
    {
        var hit = Hit.Neighbour;
        origin = -1;
        program = 0;
        foreach (var range in ranges)
        {
            if (address < range.Address || address - range.Address >= range.Size)
            {
                continue;
            }

            if (!KnownAtRecord(range.Origin))
            {
                origin = range.Origin;
                program = range.Program;
                return Hit.GpuValued;
            }

            hit = Hit.Label;
            origin = range.Origin;
        }

        return hit;
    }

    private static bool Inside(List<(ulong Address, ulong Size)> ranges, ulong address)
    {
        foreach (var range in ranges)
        {
            if (address >= range.Address && address - range.Address < range.Size) return true;
        }

        return false;
    }

    private int _reported;

    // The first 60 real waits and device-address-only stores of the run, one line each: when, and before or
    // after the guest's first submission of the next frame (none yet at the fault means before).
    private void Report(string what, Frame frame, ulong address, bool write, int origin, ulong program, long now)
    {
        if (Interlocked.Increment(ref _reported) > 60)
        {
            return;
        }

        var frequency = (double)Stopwatch.Frequency;
        var sinceComplete = frame.CompletedTicks == 0 ? 0 : (now - frame.CompletedTicks) * 1000.0 / frequency;
        Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"[METAL-V2][OVERLAP-DIAG-HIT] {what} frame={frame.Index} {(write ? "write" : "read")} 0x{address:X} origin={(origin >= 0 ? OriginNames[origin] : "-")} " +
            $"prog=0x{program:X16} ms_after_complete={(frame.CompletedTicks == 0 ? "before" : sinceComplete.ToString("F2", CultureInfo.InvariantCulture))} " +
            $"ms_after_commit={(now - frame.CommittedTicks) * 1000.0 / frequency:F2} before_first_submit={(frame.FirstAdmitTicks == 0 || now < frame.FirstAdmitTicks ? 1 : 0)}"));
    }

    private void Attribute(Frame frame, ulong address, byte flags, bool write)
    {
        var now = _clock();
        var page = address & ~(PageBytes - 1);
        var what = new StringBuilder();
        if ((flags & InWrite) != 0)
        {
            var hit = ClassifyWrite(frame.WriteRanges, address, out var origin, out var program);
            if (hit == Hit.GpuValued)
            {
                if (write) frame.WsWriteGpu++; else frame.WsReadGpu++;
                if (frame.FirstRealWaitTicks == 0) frame.FirstRealWaitTicks = now;
                if (!write && frame.FirstK6Ticks == 0) frame.FirstK6Ticks = now;
                Report("K6", frame, address, write, origin, program, now);
            }
            else if (hit == Hit.Label)
            {
                if (write) frame.WsWriteLabel++; else frame.WsReadLabel++;
            }
            else
            {
                if (write) frame.WsWriteNeighbour++; else frame.WsReadNeighbour++;
            }

            what.Append(hit switch { Hit.GpuValued => "gpu_hit", Hit.Label => "label_hit", _ => "neighbour" });
            if (origin >= 0) what.Append(':').Append(OriginNames[origin]);
            if (program != 0) what.Append(CultureInfo.InvariantCulture, $":prog=0x{program:X16}");
        }

        if ((flags & InRead) != 0 && write)
        {
            var hit = Inside(frame.ReadRanges, address);
            if (hit) { frame.RsHit++; frame.ReadHitAddresses.Add(address); } else frame.RsNeighbour++;
            what.Append(what.Length != 0 ? "," : string.Empty).Append(hit ? "read_set_hit" : "read_set_neighbour");
        }

        if ((flags & DeviceAddressOnly) != 0 && write)
        {
            var hit = false;
            ulong program = 0;
            foreach (var range in frame.DeviceRanges)
            {
                if (address >= range.Address && address - range.Address < range.Size) { hit = true; program = range.Program; break; }
            }

            if (hit) frame.DaOnlyHit++; else frame.DaOnlyNeighbour++;
            Report(hit ? "K3-hit" : "K3-neighbour", frame, address, write, (int)OverlapWrite.DeviceAddress, program, now);
            what.Append(what.Length != 0 ? "," : string.Empty).Append(hit ? "da_only_hit" : "da_only_neighbour");
            if (program != 0) what.Append(CultureInfo.InvariantCulture, $":prog=0x{program:X16}");
        }

        lock (_pageReports)
        {
            if (!_pageReports.TryGetValue(page, out var report))
            {
                report = new PageReport();
                _pageReports[page] = report;
                foreach (var range in frame.WriteRanges)
                {
                    if (report.Ranges.Count < 8 && range.Address < page + PageBytes && page < range.Address + range.Size)
                        report.Ranges.Add(string.Create(CultureInfo.InvariantCulture,
                            $"{OriginNames[range.Origin]}:0x{range.Address:X}+0x{range.Size:X}{(range.Program != 0 ? $":prog=0x{range.Program:X16}" : string.Empty)}"));
                }

                foreach (var range in frame.ReadRanges)
                {
                    if (report.Ranges.Count < 8 && range.Address < page + PageBytes && page < range.Address + range.Size)
                        report.Ranges.Add(string.Create(CultureInfo.InvariantCulture, $"read:0x{range.Address:X}+0x{range.Size:X}"));
                }

                foreach (var range in frame.DeviceRanges)
                {
                    if (report.Ranges.Count < 8 && range.Address < page + PageBytes && page < range.Address + range.Size)
                        report.Ranges.Add(string.Create(CultureInfo.InvariantCulture, $"da_read:0x{range.Address:X}+0x{range.Size:X}:prog=0x{range.Program:X16}"));
                }
            }

            report.Faults++;
            var access = string.Create(CultureInfo.InvariantCulture, $"0x{address & (PageBytes - 1):X3}{(write ? "w" : "r")}={what}");
            if (report.Accesses.Count < 8 && !report.Accesses.Contains(access)) report.Accesses.Add(access);
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
        // Bound ranges that contain a written address itself (not just its page).
        long detachHitBytes = 0;
        foreach (var range in frame.ReadRanges)
        {
            foreach (var address in frame.ReadHitAddresses)
            {
                if (address >= range.Address && address - range.Address < range.Size)
                {
                    detachHitBytes += (long)range.Size;
                    break;
                }
            }
        }

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

                if (frame.ArmedTicks < frame.CompletedTicks) window.ArmedBeforeComplete++;
                window.RsHit.Add(frame.RsHit); window.RsNb.Add(frame.RsNeighbour);
                window.WsRLabel.Add(frame.WsReadLabel); window.WsRGpu.Add(frame.WsReadGpu); window.WsRNb.Add(frame.WsReadNeighbour);
                window.WsWLabel.Add(frame.WsWriteLabel); window.WsWGpu.Add(frame.WsWriteGpu); window.WsWNb.Add(frame.WsWriteNeighbour);
                window.DaHit.Add(frame.DaOnlyHit); window.DaNb.Add(frame.DaOnlyNeighbour);
                window.RealTrap.Add(frame.RsHit + frame.WsReadGpu + frame.WsWriteGpu + frame.DaOnlyHit);
                window.RealWait.Add(frame.WsReadGpu + frame.WsWriteGpu);
                window.DetachHitMb.Add(detachHitBytes / 1048576.0);
                var available = Ms(frame.CommittedTicks, frame.CompletedTicks);
                if (frame.FirstK6Ticks != 0)
                {
                    window.K6Frames++;
                    window.K6Ms.Add(Ms(frame.CompletedTicks, frame.FirstK6Ticks));
                    if (frame.FirstAdmitTicks == 0 || frame.FirstK6Ticks < frame.FirstAdmitTicks) window.K6BeforeSubmit++;
                }

                if (frame.FirstRealWaitTicks != 0)
                {
                    var wait = Ms(frame.CompletedTicks, frame.FirstRealWaitTicks);
                    window.RealWaitMs.Add(wait);
                    if (frame.FirstAdmitTicks == 0 || frame.FirstRealWaitTicks < frame.FirstAdmitTicks) window.RealWaitBeforeSubmit++;
                    // Build work done before the first real wait is what an overlap would hide.
                    window.OverlapUntilWait.Add(Math.Clamp(wait, 0, double.IsNaN(available) ? 0 : available));
                }
                else
                {
                    window.OverlapUntilWait.Add(available);
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
            builder.Append(culture, $"trap: armed_pages={Mx(w.ArmedPages)} arm_ms={P(w.ArmMs)} late_arms={w.LateArms} late_faults={Interlocked.Exchange(ref _lateFaults, 0)} cap_frames={w.CapFrames} skipped_read_watched={w.Skipped} ");
            builder.Append($"rs_written={Mx(w.RsWritten)} detach_mb={Mx(w.DetachMb)} ws_read={Mx(w.WsRead)} ws_written={Mx(w.WsWritten)} ");
            builder.Append($"label_touched={Mx(w.LabelTouched)} cb_touched={Mx(w.CbTouched)} da_only_written={Mx(w.DaOnly)} managed={Mx(w.Managed)} first_fault_ms={P(w.FirstFaultMs)} ");
            builder.Append(culture, $"trap_before_submit_frames={w.TrapBeforeSubmitFrames} ws_read_frames={w.WriteSetReadFrames} | ");
            builder.Append(culture, $"bytes: armed_before_complete={w.ArmedBeforeComplete} rs_hit={Mx(w.RsHit)} rs_nb={Mx(w.RsNb)} detach_hit_mb={Mx(w.DetachHitMb)} ");
            builder.Append($"ws_read label={Mx(w.WsRLabel)} gpu={Mx(w.WsRGpu)} nb={Mx(w.WsRNb)} ws_write label={Mx(w.WsWLabel)} gpu={Mx(w.WsWGpu)} nb={Mx(w.WsWNb)} ");
            builder.Append($"da_only hit={Mx(w.DaHit)} nb={Mx(w.DaNb)} real_trap_pages={Mx(w.RealTrap)} real_wait_pages={Mx(w.RealWait)} ");
            builder.Append(culture, $"k6_frames={w.K6Frames} k6_ms={P(w.K6Ms)} k6_before_submit={w.K6BeforeSubmit} ");
            builder.Append(culture, $"real_wait_ms={P(w.RealWaitMs)} real_wait_before_submit={w.RealWaitBeforeSubmit} overlap_until_wait_ms={P(w.OverlapUntilWait)}");
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
                builder.Append(CultureInfo.InvariantCulture, $" 0x{pair.Key:X}+0x{pair.Value.Size:X}*{pair.Value.Hits}");
            }
        }

        lock (_pageReports)
        {
            foreach (var pair in _pageReports.OrderByDescending(pair => pair.Value.Faults).Take(10))
            {
                builder.Append(CultureInfo.InvariantCulture, $"\n[METAL-V2][OVERLAP-DIAG-PAGE] 0x{pair.Key:X} faults={pair.Value.Faults} ranges=[{string.Join(' ', pair.Value.Ranges)}] accesses=[{string.Join(' ', pair.Value.Accesses)}]");
            }
        }

        builder.Append("\n[METAL-V2][OVERLAP-DIAG-TOP]");
        builder.Append(" | wait (GPU-written pages the guest touched):");
        lock (_topWaits)
        {
            foreach (var pair in _topWaits.OrderByDescending(pair => pair.Value.Hits).Take(10))
            {
                builder.Append(CultureInfo.InvariantCulture, $" 0x{pair.Key:X}:{OriginNames[pair.Value.Origin]}*{pair.Value.Hits}");
            }
        }

        return builder.ToString();
    }

    private void ResetWindow()
    {
        var w = _window;
        w.Frames = w.Sampled = w.CapFrames = w.UnfinishedFrames = w.BlockedFrames = w.RefusedFrames = w.TrapBeforeSubmitFrames = 0;
        w.WriteSetReadFrames = w.LateArms = 0;
        w.ArmedBeforeComplete = w.K6Frames = w.K6BeforeSubmit = w.RealWaitBeforeSubmit = 0;
        foreach (var list in new[] { w.RsHit, w.RsNb, w.WsRLabel, w.WsRGpu, w.WsRNb, w.WsWLabel, w.WsWGpu, w.WsWNb, w.DaHit, w.DaNb,
                     w.RealTrap, w.RealWait, w.DetachHitMb, w.K6Ms, w.RealWaitMs, w.OverlapUntilWait })
        {
            list.Clear();
        }

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
