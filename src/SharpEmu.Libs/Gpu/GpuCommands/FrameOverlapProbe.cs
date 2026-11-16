// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;

namespace SharpEmu.Libs.Gpu.GpuCommands;

/// <summary>Why a frame's publications were left to its real completion.</summary>
public enum FrameOverlapRefusal
{
    /// <summary>An end-of-pipe GDS store: the words are whatever the GPU left in the data share.</summary>
    GdsStore,

    /// <summary>A write whose value only the GPU (or its completion) can produce.</summary>
    GpuValued,

    /// <summary>A GPU write with no bounds: the frame's write set is everything.</summary>
    Unbounded,

    /// <summary>An end-of-pipe atomic add into memory the frame's GPU work writes.</summary>
    AtomicIntoWriteSet,

    /// <summary>The walk parked on a wait that only a completion of this stream releases.</summary>
    SuspendedWalk,

    /// <summary>A command-processor copy whose source the GPU still produces.</summary>
    CopyFromGpu,
}

/// <summary>
/// SHARPEMU_FRAME_OVERLAP_PROBE=1: an UNGUARDED measurement probe, default off, never to be
/// shipped. The host-side publications the stream queues for a GPU completion (host-landed
/// labels, end-of-pipe label slots, command-processor writes, end-of-pipe interrupts, copies of
/// those) land per command buffer: when the buffer that carries them is committed, in stream
/// order, instead of when the GPU completes it. The bound: a buffer's publications land early
/// only once every flip but the newest committed one has really completed on the GPU, so at most
/// one flip's worth of GPU work is outstanding behind what the guest has seen; a buffer held by
/// the bound lands when that flip completes. Nothing protects what unfinished GPU work still reads
/// or writes: rendering and guest memory may be corrupted. A buffer that queued a publication the
/// probe cannot make early (see <see cref="FrameOverlapRefusal"/>), and every buffer after it until
/// it completes, is published by its real completion, as without the switch.
/// </summary>
/// <remarks>
/// The probe only reorders work the host already queues. <c>enqueueAtSerial</c> runs an action
/// once the GPU signalled a serial, in serial order and, among equal serials, in the order queued
/// (the Metal completion queue); every publication then runs on that one thread, so a publication
/// that ran early is skipped by its real completion without a lock around the work itself.
/// </remarks>
public sealed class FrameOverlapProbe
{
    public static readonly bool Enabled = string.Equals(
        Environment.GetEnvironmentVariable("SHARPEMU_FRAME_OVERLAP_PROBE"), "1", StringComparison.Ordinal);

    // SHARPEMU_FRAME_OVERLAP_PROBE_WAITS=count: a park on owed bytes is counted (suspended_owed)
    // instead of refusing the buffer.
    public static readonly bool CountOwedWaitsOnly = string.Equals(
        Environment.GetEnvironmentVariable("SHARPEMU_FRAME_OVERLAP_PROBE_WAITS"), "count", StringComparison.Ordinal);

    // SHARPEMU_FRAME_OVERLAP_PROBE_AFTER_S=<seconds>: publications land early only that long after the probe
    // started (boot and menus publish at real completion); the other probe rules apply from the start.
    private static readonly double ArmAfterSeconds =
        double.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_FRAME_OVERLAP_PROBE_AFTER_S"),
            System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var seconds) && seconds > 0
            ? seconds
            : 0;

    public const string WarningLine =
        "unguarded frame-overlap probe: rendering and guest memory are not protected; measurement only";

    /// <summary>The probe the command stream reports into (one GPU host per process).</summary>
    public static FrameOverlapProbe? Active { get; set; }

    private const int RefusalCount = (int)FrameOverlapRefusal.CopyFromGpu + 1;
    private const int MaxRefusalReports = 24;

    private readonly Action<ulong, Action> _enqueueAtSerial;
    private readonly Func<ulong> _completedSerial;
    private readonly Func<long> _clock;
    private readonly object _gate = new();
    private Batch _open = new(0);
    // Committed batches whose real completion has not run yet, oldest first.
    private readonly List<Batch> _unfinished = new();
    // Serials of the committed buffers that carried a flip, newest last (the last two are kept).
    private ulong _lastFlipSerial;
    private ulong _flipBeforeLastSerial;
    private readonly long _armAt;
    private int _refusalReports;

    // Window counters, guarded by _gate.
    private long _commits;
    private long _batches;
    private long _batchesAtCommit;
    private long _batchesHeld;
    private long _batchesEarly;
    private long _batchesAtCompletion;
    private long _batchesRefused;
    private long _batchesBehindRefusal;
    private long _batchesUnarmed;
    private long _itemsEarly;
    private long _itemsHeld;
    private long _flips;
    private long _flipsEarly;
    private long _suspendedOther;
    private long _suspendedOwed;
    private long _queuedCopies;
    private int _flipsOutstandingMax;
    private readonly long[] _refusals = new long[RefusalCount];
    private readonly List<double> _earlyMs = new();
    private readonly List<double> _flipsOutstanding = new();

    public FrameOverlapProbe(Action<ulong, Action> enqueueAtSerial, Func<ulong> completedSerial, Func<long>? clock = null)
    {
        _enqueueAtSerial = enqueueAtSerial;
        _completedSerial = completedSerial;
        _clock = clock ?? Stopwatch.GetTimestamp;
        _armAt = ArmAfterSeconds > 0 ? _clock() + (long)(ArmAfterSeconds * Stopwatch.Frequency) : long.MinValue;
    }

    /// <summary>
    /// A publication the open buffer queued for its completion. The result is what the real
    /// completion runs: the publication unless it already ran early, then
    /// <paramref name="atRealCompletion"/>, which is never published early.
    /// </summary>
    public Action Track(Action publish, Action? atRealCompletion = null, ulong address = 0, ulong bytes = 0)
    {
        var item = new Item(publish, address, bytes);
        lock (_gate)
        {
            _open.Items.Add(item);
        }

        return atRealCompletion is null
            ? () => item.Run()
            : () =>
            {
                item.Run();
                atRealCompletion();
            };
    }

    /// <summary>The open buffer carries a flip.</summary>
    public void NoteFlip()
    {
        lock (_gate)
        {
            _open.HasFlip = true;
        }
    }

    /// <summary>A copy whose source is a queued publication was queued behind it.</summary>
    public void NoteQueuedCopy() => Interlocked.Increment(ref _queuedCopies);

    /// <summary>The first serial the open buffer's work can carry.</summary>
    public ulong OpenFrameFirstSerial
    {
        get
        {
            lock (_gate)
            {
                return _open.FirstSerial;
            }
        }
    }

    /// <summary>Whether a publication still queued (not yet run) writes any of these bytes.</summary>
    public bool Owes(ulong address, ulong size)
    {
        lock (_gate)
        {
            return OwesLocked(address, size);
        }
    }

    private bool OwesLocked(ulong address, ulong size)
    {
        if (Owes(_open, address, size))
        {
            return true;
        }

        foreach (var batch in _unfinished)
        {
            if (Owes(batch, address, size))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>The open buffer queued something the probe cannot publish early.</summary>
    public void Refuse(FrameOverlapRefusal reason, ulong address = 0)
    {
        lock (_gate)
        {
            _refusals[(int)reason]++;
            var first = _open.Refusal is null;
            _open.Refusal ??= reason;
            if (first && _refusalReports < MaxRefusalReports)
            {
                _refusalReports++;
                Console.Error.WriteLine(
                    $"[METAL-V2][OVERLAP-PROBE] refused buffer={_open.Index} reason={Name(reason)} address=0x{address:X16}; " +
                    "it and the buffers after it publish at their real completion");
            }
        }
    }

    /// <summary>
    /// The command walk parked a submission. A wait on bytes a queued, not yet published
    /// completion writes is released only by a completion: the open buffer is refused.
    /// </summary>
    public void NoteSuspendedWalk(in CommandWaitCondition wait)
    {
        if (wait.Kind == CommandWaitKind.GuestMemory && wait.Size != 0)
        {
            bool owed;
            lock (_gate)
            {
                owed = OwesLocked(wait.Address, wait.Size);
                if (!owed)
                {
                    _suspendedOther++;
                }
                else if (CountOwedWaitsOnly)
                {
                    _suspendedOwed++;
                }
            }

            if (owed && !CountOwedWaitsOnly)
            {
                Refuse(FrameOverlapRefusal.SuspendedWalk, wait.Address);
            }

            return;
        }

        lock (_gate)
        {
            _suspendedOther++;
        }
    }

    /// <summary>
    /// A command buffer was committed with <paramref name="serial"/>: what the stream queued since
    /// the previous commit is carried by it. It lands now if every flip but the newest committed one
    /// has completed, else when the flip before the newest completes (or at its own completion).
    /// </summary>
    public void OnCommit(ulong serial)
    {
        Batch batch;
        ulong gate;
        lock (_gate)
        {
            _commits++;
            batch = _open;
            _open = new Batch(batch.Index + 1) { FirstSerial = serial + 1 };
            if (batch.Items.Count == 0 && !batch.HasFlip)
            {
                return;
            }

            batch.Serial = serial;
            if (batch.HasFlip)
            {
                _flipBeforeLastSerial = _lastFlipSerial;
                _lastFlipSerial = serial;
                _flips++;
            }

            gate = _flipBeforeLastSerial;
            _unfinished.Add(batch);
            _batches++;
            if (gate > _completedSerial())
            {
                _batchesHeld++;
                _itemsHeld += batch.Items.Count;
            }
            else
            {
                _batchesAtCommit++;
            }
        }

        _enqueueAtSerial(gate, () => Publish(batch));
        _enqueueAtSerial(serial, () => Completed(batch));
    }

    // Runs once the bound allows it (completion thread).
    private void Publish(Batch batch)
    {
        lock (_gate)
        {
            if (batch.Refusal is not null)
            {
                _batchesRefused++;
                return;
            }

            foreach (var earlier in _unfinished)
            {
                if (earlier == batch)
                {
                    break;
                }

                if (earlier.Refusal is not null)
                {
                    // Landing now would overtake publications that wait for the GPU.
                    _batchesBehindRefusal++;
                    return;
                }
            }

            if (_clock() < _armAt)
            {
                _batchesUnarmed++;
                return;
            }
        }

        var early = _completedSerial() < batch.Serial;
        var ran = 0;
        foreach (var item in batch.Items)
        {
            if (item.Run())
            {
                ran++;
            }
        }

        lock (_gate)
        {
            if (!early)
            {
                _batchesAtCompletion++;
                return;
            }

            batch.PublishedEarly = true;
            batch.PublishedAt = _clock();
            _batchesEarly++;
            _itemsEarly += ran;
            _flipsEarly += batch.HasFlip ? 1 : 0;
            // The flips committed and not yet completed behind what the guest now sees.
            var outstanding = 0;
            foreach (var pending in _unfinished)
            {
                outstanding += pending.HasFlip ? 1 : 0;
            }

            _flipsOutstanding.Add(outstanding);
            _flipsOutstandingMax = Math.Max(_flipsOutstandingMax, outstanding);
        }
    }

    // Runs once the buffer has really completed (completion thread), after all its items.
    private void Completed(Batch batch)
    {
        var now = _clock();
        lock (_gate)
        {
            if (batch.PublishedEarly && ran(batch))
            {
                _earlyMs.Add((now - batch.PublishedAt) * 1000.0 / Stopwatch.Frequency);
            }

            // Every item has run: none is owed to a wait any more.
            batch.Items.Clear();
            _unfinished.Remove(batch);
        }

        static bool ran(Batch done) => done.Items.Count != 0;
    }

    private static bool Owes(Batch batch, ulong address, ulong size)
    {
        foreach (var item in batch.Items)
        {
            if (item.Bytes != 0 && !item.HasRun && address < item.Address + item.Bytes && item.Address < address + size)
            {
                return true;
            }
        }

        return false;
    }

    public static string Name(FrameOverlapRefusal reason) => reason switch
    {
        FrameOverlapRefusal.GdsStore => "gds_store",
        FrameOverlapRefusal.GpuValued => "gpu_valued",
        FrameOverlapRefusal.Unbounded => "unbounded",
        FrameOverlapRefusal.AtomicIntoWriteSet => "atomic_write_set",
        FrameOverlapRefusal.SuspendedWalk => "suspended_walk",
        FrameOverlapRefusal.CopyFromGpu => "copy_from_gpu",
        _ => reason.ToString(),
    };

    /// <summary>The window's counters as one line, then reset.</summary>
    public string TakeWindowLine()
    {
        lock (_gate)
        {
            var refusals = string.Join(' ', Enumerable.Range(0, RefusalCount).Select(index =>
                $"{Name((FrameOverlapRefusal)index)}={_refusals[index]}"));
            var line =
                $"[METAL-V2][OVERLAP-PROBE] commits={_commits} batches={_batches} at_commit={_batchesAtCommit} " +
                $"held_by_bound={_batchesHeld} items_held={_itemsHeld} published_early={_batchesEarly} " +
                $"published_at_completion={_batchesAtCompletion} refused={_batchesRefused} " +
                $"behind_refusal={_batchesBehindRefusal} unarmed={_batchesUnarmed} items_early={_itemsEarly} " +
                $"ms_early_p50={Percentile(_earlyMs, 0.5):F1} ms_early_p90={Percentile(_earlyMs, 0.9):F1} " +
                $"flips={_flips} flips_early={_flipsEarly} " +
                $"flips_outstanding_p50={Percentile(_flipsOutstanding, 0.5):F0} flips_outstanding_max={_flipsOutstandingMax} " +
                $"queued_copies={Interlocked.Exchange(ref _queuedCopies, 0)} " +
                $"suspended_other={_suspendedOther} suspended_owed={_suspendedOwed} refusals: {refusals}";
            _commits = _batches = _batchesAtCommit = _batchesHeld = _batchesEarly = _batchesAtCompletion = 0;
            _batchesRefused = _batchesBehindRefusal = _batchesUnarmed = _itemsEarly = _itemsHeld = 0;
            _flips = _flipsEarly = _suspendedOther = _suspendedOwed = 0;
            _flipsOutstandingMax = 0;
            Array.Clear(_refusals);
            _earlyMs.Clear();
            _flipsOutstanding.Clear();
            return line;
        }
    }

    private static double Percentile(List<double> values, double fraction)
    {
        if (values.Count == 0)
        {
            return 0;
        }

        values.Sort();
        return values[Math.Min(values.Count - 1, (int)(fraction * values.Count))];
    }

    private sealed class Batch(long index)
    {
        public long Index { get; } = index;
        public ulong FirstSerial { get; init; } = 1;
        public ulong Serial { get; set; }
        public List<Item> Items { get; } = new();
        public FrameOverlapRefusal? Refusal { get; set; }
        public bool HasFlip { get; set; }
        public bool PublishedEarly { get; set; }
        public long PublishedAt { get; set; }
    }

    private sealed class Item(Action publish, ulong address, ulong bytes)
    {
        private int _ran;

        public ulong Address { get; } = address;
        public ulong Bytes { get; } = bytes;
        public bool HasRun => Volatile.Read(ref _ran) != 0;

        // Runs the publication once, whoever gets there first; true when this call ran it.
        public bool Run()
        {
            if (Interlocked.Exchange(ref _ran, 1) != 0)
            {
                return false;
            }

            publish();
            return true;
        }
    }
}
