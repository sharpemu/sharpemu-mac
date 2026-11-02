// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using Iced.Intel;
using SharpEmu.HLE;
using static Iced.Intel.AssemblerRegisters;

namespace SharpEmu.Core.Loader;

/// <summary>
/// The host-side half of the generic spin-loop nap (see <see cref="GuestRedZonePatcher"/>'s spin
/// pass). Every patched guest <c>pause</c> jumps through one shared, position-independent native
/// probe this class emits. The probe keeps per-thread state in a direct-mapped table of 64-byte
/// slots keyed on the host thread pointer (one segment-relative load: <c>fs:[0]</c> on Linux, through
/// a based operand the guest-TLS scan leaves alone, <c>gs:[0]</c> on macOS, <c>gs:[0x30]</c> on
/// Windows; it can never fault), tracks how long the
/// thread has spun without interruption, and once that passes the idle threshold naps (a real
/// short host sleep). A gap between two <c>pause</c>es longer than the reset window restarts the
/// streak, so a lock backoff that resolves quickly never sleeps. <c>SHARPEMU_SPIN_NAP_SHARED_STATE=1</c>
/// restores the old keying (<c>rsp &gt;&gt; 26</c>, one bucket per 64&#160;MiB of guest stack, shared
/// by every thread whose stack falls in it). The probe saves and restores every general register
/// and the flags, and around a nap the x87/SSE state, so it is transparent to the guest: it runs in
/// the middle of guest code, not at a call boundary. A nap calls the host's sleep function directly
/// and never managed code, because the probe runs on the guest stack and managed code only runs on
/// the host stack (the import thunks and the fault bridge switch to it first).
/// </summary>
internal static unsafe class SpinNapRuntime
{
    public const string EnvironmentVariable = "SHARPEMU_SPIN_NAP";
    public const string IdleEnvironmentVariable = "SHARPEMU_SPIN_NAP_IDLE_US";
    public const string NapEnvironmentVariable = "SHARPEMU_SPIN_NAP_US";
    public const string GapEnvironmentVariable = "SHARPEMU_SPIN_NAP_GAP_US";
    public const string AbEnvironmentVariable = "SHARPEMU_SPIN_NAP_AB_S";
    public const string WideShapesEnvironmentVariable = "SHARPEMU_SPIN_NAP_SHAPES";
    public const string LockedEnvironmentVariable = "SHARPEMU_SPIN_NAP_LOCKED";
    public const string LockedSitesEnvironmentVariable = "SHARPEMU_SPIN_NAP_LOCKED_SITES";
    public const string LockedDensityEnvironmentVariable = "SHARPEMU_SPIN_NAP_LOCKED_DENSITY_US";
    public const string PauseDelayEnvironmentVariable = "SHARPEMU_PAUSE_DELAY_NS";
    public const string PauseStatsEnvironmentVariable = "SHARPEMU_PAUSE_DELAY_STATS";
    public const string SharedStateEnvironmentVariable = "SHARPEMU_SPIN_NAP_SHARED_STATE";

    internal const uint DefaultIdleMicroseconds = 500;
    internal const uint DefaultNapMicroseconds = 250;
    internal const uint DefaultGapMicroseconds = 50;
    internal const uint DefaultLockedDensityMicroseconds = 5;
    internal const int DefaultLockedSites = 4;
    internal const int MaxLockedSites = 8;

    // Experimental, off until a Mac A/B shows no fps loss. Under Rosetta a guest `pause` is a no-op
    // (about 0.26 ns: ARM `yield` does nothing on cores without SMT), so x86 spin backoffs that are
    // calibrated in pause counts give up far sooner than on the PS5 and fall through to their
    // sleep/syscall path far more often. PS5 candidate: Zen 2 PAUSE takes 65 cycles (uops.info,
    // measured throughput), 18.6 ns at the PS5's 3.5 GHz, rounded up.
    internal const uint DefaultPauseDelayNanoseconds = 0;
    internal const uint Zen2PauseDelayNanoseconds = 19;
    internal const uint MaxPauseDelayNanoseconds = 100_000;

    // Pause-probe counters (SHARPEMU_PAUSE_DELAY_STATS): 1024 slots x 64 bytes, entries(0)
    // delayedTicks(8). A slot is picked by a multiplicative hash of rsp >> 12, so threads count on
    // their own cache lines in either keying mode; a collision can only lose a count.
    private const int PauseStatsSlotBits = 10;
    private const int PauseStatsSlotCount = 1 << PauseStatsSlotBits;
    private const int PauseStatsSlotStride = 64;
    private const ulong PauseStatsHashMultiplier = 0x9E3779B97F4A7C15UL;

    // Per-thread state (the default). Both probes key a direct-mapped table on the host thread
    // pointer, hashed multiplicatively (key * golden >> (64 - bits)), so each guest thread's state
    // sits on cache lines no other live thread writes. The slot stores the full key: a collision
    // (another live thread hashing to the same slot, about n^2 / 2 / slots for n spinning threads)
    // only costs the two threads their streaks, as the old shared bucket did for everyone.
    //   pause:  16384 slots x 64 bytes = 1 MiB: key(0) epoch(8) last(16), rest of the line unused.
    //   locked: 4096 slots x 256 bytes = 1 MiB: the shared layout below, plus the thread's HLE
    //           generation at hleGen(48), bumped by NoteHleEntry on the same thread.
    private const ulong ThreadKeyHashMultiplier = 0x9E3779B97F4A7C15UL;
    private const int PrivateSlotBits = 14;
    private const int PrivateSlotCount = 1 << PrivateSlotBits;
    private const int PrivateSlotStride = 64;
    private const int PrivateLockedSlotBits = 12;
    private const int PrivateLockedSlotCount = 1 << PrivateLockedSlotBits;
    private const int LockedHleGenerationOffset = 48;

    // Locked-op probe table, shared keying: 1024 buckets x 256 bytes. entry: key(0) start(8) last(16)
    // hits(24) other(32) gen(40) | sites[8](64) | siteHits[8](128). The HLE generation array is one
    // qword per bucket, bumped by the import dispatcher on the calling guest thread's stack bucket.
    private const int LockedBucketCount = 1024;
    internal const int LockedBucketMask = LockedBucketCount - 1;
    private const int LockedBucketStride = 256;

    // Pause table, shared keying: 8192 buckets × 32 bytes = 256 KiB. A collision between two live
    // stacks only costs a streak reset (the key check fails), so the table never needs to be exact,
    // just large enough that collisions are rare. The bucket key is rsp >> StackKeyShift.
    private const int BucketCount = 8192;
    private const int BucketMask = BucketCount - 1;
    private const int BucketStride = 32; // key(8) epoch(8) last(8) pad(8)
    internal const int StackKeyShift = 26; // 64 MiB guest stacks

    // Native state the probes share (qword slots): the A/B epoch in TSC ticks (0 until the first
    // nap) and the nap counters the reporter logs.
    private const int ControlAbEpoch = 0;
    private const int ControlLockedNaps = 1;
    private const int ControlPauseNaps = 2;
    private const int ControlSlots = 3;

    // fxsave64 image of the x87/SSE state: 512 bytes, 16-byte aligned.
    private const int FxSaveBytes = 512;

    private static readonly object Gate = new();
    private static byte[]? _probeBytes;
    private static byte* _table;

    private static byte[]? _lockedProbeBytes;
    private static byte* _lockedTable;
    private static long* _lockedGenerations;
    private static bool _lockedSharedState;

    // The calling host thread's HLE generation in the private locked table, found once per thread.
    [ThreadStatic]
    private static long* t_hleGeneration;

    private static delegate* unmanaged<ulong> _readThreadKey;

    private static NapTarget? _napTarget;
    private static long* _control;
    private static Timer? _reporter;
    private static bool _abEpochReported;
    private static long _reportedLockedNaps;

    private static byte* _pauseStats;
    private static uint _pauseDelayNanoseconds;
    private static ulong _pauseTscFrequency;
    private static long _pauseReportTimestamp;
    private static PauseCounters _reportedPauseCounters;

    private readonly record struct PauseCounters(long Entries, long DelayedTicks, long Naps, long UsleepOneCalls, long LockedNaps);

    /// <summary>What a nap calls and with which argument, plus the A/B period in TSC ticks (0 = off).
    /// <paramref name="Control"/> is the shared native state (see ControlAbEpoch).</summary>
    private sealed record NapTarget(ulong SleepAddress, uint SleepArgument, int AbSeconds, long AbPeriodTicks, ulong Control);

    internal static bool IsEnabled =>
        !string.Equals(Environment.GetEnvironmentVariable(EnvironmentVariable), "0", StringComparison.Ordinal);

    /// <summary>On by default (<c>SHARPEMU_SPIN_NAP_SHAPES=0</c> turns it off): the spin pass also patches
    /// a <c>pause; jmp short</c> loop (taking the alignment nops after the jump) and loops in functions
    /// whose jump table base is hoisted out of the loop or bounds-checked with a 64-bit compare. Without
    /// it the pass leaves those pauses alone, so a pure spin-wait loop of that shape never naps.</summary>
    internal static bool WideShapesEnabled =>
        !string.Equals(Environment.GetEnvironmentVariable(WideShapesEnvironmentVariable), "0", StringComparison.Ordinal);

    /// <summary>Opt-in (<c>SHARPEMU_SPIN_NAP_LOCKED=1</c>): also instrument short backward loops
    /// that poll with <c>xchg mem</c> / <c>lock</c>-prefixed read-modify-write ops.</summary>
    internal static bool LockedEnabled =>
        string.Equals(Environment.GetEnvironmentVariable(LockedEnvironmentVariable), "1", StringComparison.Ordinal);

    internal static int LockedSiteLimit =>
        int.TryParse(Environment.GetEnvironmentVariable(LockedSitesEnvironmentVariable), out var value) &&
        value is >= 1 and <= MaxLockedSites
            ? value
            : DefaultLockedSites;

    /// <summary><c>SHARPEMU_SPIN_NAP_SHARED_STATE=1</c>: key both probes' state on <c>rsp &gt;&gt; 26</c> as
    /// before, so every thread whose stack falls in one 64&#160;MiB bucket shares one streak and one
    /// cache line. Off by default: each host thread gets its own slot.</summary>
    internal static bool SharedStateEnabled =>
        string.Equals(Environment.GetEnvironmentVariable(SharedStateEnvironmentVariable), "1", StringComparison.Ordinal);

    internal static uint IdleMicroseconds => ReadMicroseconds(IdleEnvironmentVariable, DefaultIdleMicroseconds);

    internal static uint NapMicroseconds => ReadMicroseconds(NapEnvironmentVariable, DefaultNapMicroseconds);

    internal static uint GapMicroseconds => ReadMicroseconds(GapEnvironmentVariable, DefaultGapMicroseconds);

    /// <summary><c>SHARPEMU_PAUSE_DELAY_NS</c>: how long each patched guest <c>pause</c> spins in the
    /// probe before the streak logic, so a backoff calibrated in pause counts lasts about as long as
    /// on the PS5. 0 (the default) leaves the probe exactly as it was.</summary>
    internal static uint PauseDelayNanoseconds =>
        uint.TryParse(Environment.GetEnvironmentVariable(PauseDelayEnvironmentVariable), out var value) &&
        value <= MaxPauseDelayNanoseconds
            ? value
            : DefaultPauseDelayNanoseconds;

    /// <summary><c>SHARPEMU_PAUSE_DELAY_STATS</c>: count probe entries, delayed time, pause-probe naps
    /// and <c>sceKernelUsleep(1)</c> calls and log them, with the locked-op nap count, once a second.
    /// On whenever a pause delay is set; <c>1</c> also counts with no delay (the A/B baseline), <c>0</c>
    /// turns counting off.</summary>
    internal static bool PauseStatsEnabled =>
        Environment.GetEnvironmentVariable(PauseStatsEnvironmentVariable) switch
        {
            "1" => true,
            "0" => false,
            _ => PauseDelayNanoseconds != 0,
        };

    /// <summary>Builds the shared probe's machine code once and returns it. The probe is
    /// position-independent (absolute imm64 operands, only self-relative branches), so the patcher
    /// copies it into each module's trampoline arena, which keeps it in <c>call rel32</c> reach of
    /// every site. <paramref name="tscFrequency"/> converts the microsecond thresholds to the host
    /// TSC ticks the probe's <c>rdtsc</c> measures. Returns an empty array only if allocation
    /// fails.</summary>
    internal static byte[] EnsureProbeBytes(ulong tscFrequency)
    {
        lock (Gate)
        {
            if (_probeBytes is not null)
            {
                return _probeBytes;
            }

            var sharedState = SharedStateEnabled;
            var table = AllocatePauseTable(sharedState);
            if (table == null || EnsureNapTargetLocked(tscFrequency) is not { } target)
            {
                return [];
            }

            Console.Error.WriteLine(sharedState
                ? $"[LOADER][INFO] spin nap: probe state shared per 64 MiB of guest stack ({SharedStateEnvironmentVariable}=1)"
                : "[LOADER][INFO] spin nap: probe state per host thread");
            var thresholdTicks = TicksFor(IdleMicroseconds, tscFrequency);
            var gapTicks = TicksFor(GapMicroseconds, tscFrequency);
            var delayNanoseconds = PauseDelayNanoseconds;
            var delayTicks = TicksForNanoseconds(delayNanoseconds, tscFrequency);
            var stats = PauseStatsEnabled
                ? (byte*)NativeMemory.AlignedAlloc((nuint)(PauseStatsSlotCount * PauseStatsSlotStride), PauseStatsSlotStride)
                : null;
            if (stats != null)
            {
                NativeMemory.Clear(stats, (nuint)(PauseStatsSlotCount * PauseStatsSlotStride));
                _pauseStats = stats;
                _pauseDelayNanoseconds = delayNanoseconds;
                _pauseTscFrequency = tscFrequency;
                SharpEmu.Libs.Kernel.KernelRuntimeCompatExports.CountUsleepOneCalls = true;
                EnsureReporterLocked();
            }

            if (delayTicks != 0)
            {
                Console.Error.WriteLine(
                    $"[LOADER][INFO] spin nap: pause delay {delayNanoseconds} ns = {delayTicks} TSC ticks at {tscFrequency} Hz (experimental)");
            }

            _table = table;
            _probeBytes = BuildProbe((ulong)table, thresholdTicks, gapTicks, target, delayTicks, (ulong)stats, sharedState);
            return _probeBytes;
        }
    }

    /// <summary>Builds the locked-op probe (see <see cref="BuildLockedProbe"/>) once. Same contract as
    /// <see cref="EnsureProbeBytes"/>. It naps a thread only when the thread is demonstrably polling:
    /// over the last idle-threshold window every inter-hit gap was under the gap threshold, the mean
    /// gap was under the density threshold, at least 90% of hits came from the first
    /// <see cref="LockedSiteLimit"/> distinct sites, and the thread made no HLE call in the window.</summary>
    internal static byte[] EnsureLockedProbeBytes(ulong tscFrequency)
    {
        lock (Gate)
        {
            if (_lockedProbeBytes is not null)
            {
                return _lockedProbeBytes;
            }

            var sharedState = SharedStateEnabled;
            var table = AllocateLockedTable(sharedState);
            var generations = sharedState ? (long*)NativeMemory.AllocZeroed((nuint)(LockedBucketCount * sizeof(long))) : null;
            if (table == null || (sharedState && generations == null) ||
                (!sharedState && EnsureThreadKeyReader() == null) ||
                EnsureNapTargetLocked(tscFrequency) is not { } target)
            {
                return [];
            }

            _lockedSharedState = sharedState;
            _lockedGenerations = generations;
            Interlocked.MemoryBarrier(); // NoteHleEntry reads the mode and generations after the table
            _lockedTable = table;
            _lockedProbeBytes = BuildLockedProbe(
                (ulong)table,
                (ulong)generations,
                TicksFor(IdleMicroseconds, tscFrequency),
                TicksFor(GapMicroseconds, tscFrequency),
                TicksFor(ReadMicroseconds(LockedDensityEnvironmentVariable, DefaultLockedDensityMicroseconds), tscFrequency),
                LockedSiteLimit,
                target,
                sharedState);
            return _lockedProbeBytes;
        }
    }

    // Resolves the host sleep a nap calls, allocates the shared native state and, when there is
    // something to report (A/B phases or locked-op naps), starts the reporter. Caller holds Gate.
    private static NapTarget? EnsureNapTargetLocked(ulong tscFrequency)
    {
        if (_napTarget is not null)
        {
            return _napTarget;
        }

        var control = (long*)NativeMemory.AllocZeroed((nuint)(ControlSlots * sizeof(long)));
        if (control == null || !TryResolveHostSleep(NapMicroseconds, out var sleepAddress, out var sleepArgument))
        {
            return null;
        }

        var abSeconds =
            int.TryParse(Environment.GetEnvironmentVariable(AbEnvironmentVariable), out var seconds) && seconds > 0 ? seconds : 0;
        var abPeriodTicks = (long)UInt128.Min((UInt128)(uint)abSeconds * tscFrequency, long.MaxValue);
        _control = control;
        _napTarget = new NapTarget(sleepAddress, sleepArgument, abSeconds, abPeriodTicks, (ulong)control);
        if (abPeriodTicks != 0 || LockedEnabled)
        {
            EnsureReporterLocked();
        }

        return _napTarget;
    }

    // Caller holds Gate.
    private static void EnsureReporterLocked() =>
        _reporter ??= new Timer(static _ => Report(), null, 100, 100);

    // usleep(us) on POSIX; Sleep(ms) on Windows, where Sleep counts whole timer periods (with
    // timeBeginPeriod(1) a sub-millisecond nap rounds to about 1 ms, which an idle worker can afford).
    private static bool TryResolveHostSleep(uint microseconds, out ulong address, out uint argument)
    {
        address = 0;
        argument = microseconds;
        nint library;
        nint export;
        if (OperatingSystem.IsWindows())
        {
            argument = Math.Max(1, (microseconds + 999) / 1000);
            if (!NativeLibrary.TryLoad("kernel32.dll", out library) ||
                !NativeLibrary.TryGetExport(library, "Sleep", out export))
            {
                return false;
            }
        }
        else if (!NativeLibrary.TryLoad("libc", typeof(SpinNapRuntime).Assembly, null, out library) ||
                 !NativeLibrary.TryGetExport(library, "usleep", out export))
        {
            return false;
        }

        address = (ulong)export;
        return true;
    }

    // The probes never call managed code, so the A/B epoch and the nap count are logged from here.
    private static void Report()
    {
        var target = _napTarget;
        var control = _control;
        if (target is null || control == null)
        {
            return;
        }

        var epoch = Volatile.Read(ref control[ControlAbEpoch]);
        if (target.AbPeriodTicks != 0 && epoch != 0 && !_abEpochReported)
        {
            _abEpochReported = true;
            Console.Error.WriteLine(
                $"[LOADER][INFO] spin nap A/B: epoch_ts={epoch} period_s={target.AbSeconds} (even phases nap)");
        }

        var naps = Volatile.Read(ref control[ControlLockedNaps]);
        if (naps != 0 && (_reportedLockedNaps == 0 || naps / 100_000 != _reportedLockedNaps / 100_000))
        {
            _reportedLockedNaps = naps;
            Console.Error.WriteLine($"[SPIN-NAP] locked-op naps={naps}");
        }

        ReportPauseCounters(control);
    }

    // Once a second while the pause counters move: rates over the window since the last line, then
    // running totals. delay_ns_per_entry shows what the delay loop really waited (the TSC's
    // granularity and the rdtsc cost round it up); delay_cores is the host CPU it spent.
    private static void ReportPauseCounters(long* control)
    {
        var stats = _pauseStats;
        if (stats == null)
        {
            return;
        }

        var now = Stopwatch.GetTimestamp();
        if (_pauseReportTimestamp == 0)
        {
            _pauseReportTimestamp = now;
            return;
        }

        var seconds = (now - _pauseReportTimestamp) / (double)Stopwatch.Frequency;
        if (seconds < 1.0)
        {
            return;
        }

        var (entries, delayedTicks) = ReadPauseStats(stats);
        var current = new PauseCounters(
            entries,
            delayedTicks,
            Volatile.Read(ref control[ControlPauseNaps]),
            SharpEmu.Libs.Kernel.KernelRuntimeCompatExports.UsleepOneCalls,
            Volatile.Read(ref control[ControlLockedNaps]));
        var previous = _reportedPauseCounters;
        _pauseReportTimestamp = now;
        _reportedPauseCounters = current;
        if (current == previous)
        {
            return;
        }

        var windowEntries = current.Entries - previous.Entries;
        var windowDelayedNs = TicksToNanoseconds(current.DelayedTicks - previous.DelayedTicks, _pauseTscFrequency);
        Console.Error.WriteLine(FormattableString.Invariant(
            $"[SPIN-NAP] pause-delay delay_ns={_pauseDelayNanoseconds} window_s={seconds:F2} entries_per_s={windowEntries / seconds:F0} ") +
            FormattableString.Invariant(
                $"delay_ns_per_entry={(windowEntries == 0 ? 0 : windowDelayedNs / windowEntries):F1} delay_cores={windowDelayedNs / 1e9 / seconds:F3} ") +
            FormattableString.Invariant(
                $"naps_per_s={(current.Naps - previous.Naps) / seconds:F0} usleep1_per_s={(current.UsleepOneCalls - previous.UsleepOneCalls) / seconds:F0} ") +
            FormattableString.Invariant(
                $"locked_naps_per_s={(current.LockedNaps - previous.LockedNaps) / seconds:F0} ") +
            FormattableString.Invariant(
                $"entries={current.Entries} delayed_ms={TicksToNanoseconds(current.DelayedTicks, _pauseTscFrequency) / 1e6:F1} naps={current.Naps} usleep1={current.UsleepOneCalls} locked_naps={current.LockedNaps}"));
    }

    /// <summary>Sums the pause-probe counter slots (racy reads of qwords the probes add to; fine for a
    /// rate).</summary>
    internal static (long Entries, long DelayedTicks) ReadPauseStats(byte* stats)
    {
        long entries = 0;
        long delayedTicks = 0;
        for (var slot = 0; slot < PauseStatsSlotCount; slot++)
        {
            var counters = (long*)(stats + slot * PauseStatsSlotStride);
            entries += Volatile.Read(ref counters[0]);
            delayedTicks += Volatile.Read(ref counters[1]);
        }

        return (entries, delayedTicks);
    }

    private static double TicksToNanoseconds(long ticks, ulong tscFrequency) =>
        tscFrequency == 0 ? 0 : ticks * 1e9 / tscFrequency;

    /// <summary>Called by the HLE import dispatcher, on the guest thread's own host thread, with an
    /// address on its guest stack: any HLE entry ends that thread's polling streak in the locked-op
    /// probe. A no-op until the locked probe exists. Per-thread state: bumps the generation in the
    /// calling thread's own slot (the same host-thread key the probe reads, so the stack address is
    /// not needed). Shared state: bumps the generation of the stack address's 64 MiB bucket.</summary>
    internal static void NoteHleEntry(nint guestStackAddress)
    {
        var generation = t_hleGeneration;
        if (generation != null)
        {
            (*generation)++;
            return;
        }

        var table = _lockedTable;
        if (table == null)
        {
            return;
        }

        if (!Volatile.Read(ref _lockedSharedState))
        {
            generation = (long*)(LockedSlot(table, sharedState: false, CurrentThreadKey()) + LockedHleGenerationOffset);
            t_hleGeneration = generation;
            (*generation)++;
            return;
        }

        _lockedGenerations[(int)(((ulong)guestStackAddress >> StackKeyShift) & LockedBucketMask)]++;
    }

    /// <summary>The value both probes key per-thread state on: the host thread pointer, read with the
    /// same instruction the probes emit (see <see cref="EmitThreadKey"/>).</summary>
    internal static ulong CurrentThreadKey()
    {
        var reader = EnsureThreadKeyReader();
        return reader == null ? 0 : reader();
    }

    // A tiny native function that runs EmitThreadKey into rax, so managed code computes exactly the
    // key the probes see. Emitted once.
    private static delegate* unmanaged<ulong> EnsureThreadKeyReader()
    {
        if (_readThreadKey != null)
        {
            return _readThreadKey;
        }

        lock (Gate)
        {
            if (_readThreadKey != null)
            {
                return _readThreadKey;
            }

            var a = new Assembler(64);
            EmitThreadKey(a, rax, eax);
            a.ret();
            using var stream = new MemoryStream();
            a.Assemble(new StreamCodeWriter(stream), 0);
            var code = stream.ToArray();
            var page = (byte*)HostMemory.Alloc(
                null, 4096, HostMemory.MEM_COMMIT | HostMemory.MEM_RESERVE, HostMemory.PAGE_EXECUTE_READWRITE);
            if (page == null)
            {
                return null;
            }

            code.CopyTo(new Span<byte>(page, code.Length));
            _ = HostMemory.Protect(page, 4096, HostMemory.PAGE_EXECUTE_READ, out _);
            HostMemory.FlushInstructionCache(page, (nuint)code.Length);
            _readThreadKey = (delegate* unmanaged<ulong>)page;
            return _readThreadKey;
        }
    }

    private static byte* AllocatePauseTable(bool sharedState) =>
        sharedState
            ? (byte*)NativeMemory.AllocZeroed((nuint)(BucketCount * BucketStride))
            : AllocateAlignedZeroed(PrivateSlotCount * PrivateSlotStride);

    private static byte* AllocateLockedTable(bool sharedState) =>
        sharedState
            ? (byte*)NativeMemory.AllocZeroed((nuint)(LockedBucketCount * LockedBucketStride))
            : AllocateAlignedZeroed(PrivateLockedSlotCount * LockedBucketStride);

    // Private slots must start on a cache line, which calloc does not promise for a large block.
    private static byte* AllocateAlignedZeroed(int bytes)
    {
        var block = (byte*)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        if (block != null)
        {
            NativeMemory.Clear(block, (nuint)bytes);
        }

        return block;
    }

    // Managed mirrors of the probes' slot addressing, for NoteHleEntry and the tests. `key` is the
    // host thread key (private) or rsp >> StackKeyShift (shared).
    private static int PrivateSlotIndex(ulong key, int bits) => (int)((key * ThreadKeyHashMultiplier) >> (64 - bits));

    internal static byte* PauseSlot(byte* table, bool sharedState, ulong key) =>
        sharedState
            ? table + (int)(key & BucketMask) * BucketStride
            : table + PrivateSlotIndex(key, PrivateSlotBits) * PrivateSlotStride;

    internal static byte* LockedSlot(byte* table, bool sharedState, ulong key) =>
        sharedState
            ? table + (int)(key & LockedBucketMask) * LockedBucketStride
            : table + PrivateSlotIndex(key, PrivateLockedSlotBits) * LockedBucketStride;

    // Test hook: a locked-op probe over a fresh table whose thresholds are zero, so it naps on
    // every hit after the first.
    internal static byte[] BuildNappingLockedProbeForTest(ulong tscFrequency, bool sharedState = false)
    {
        var table = AllocateLockedTable(sharedState);
        var generations = (long*)NativeMemory.AllocZeroed((nuint)(LockedBucketCount * sizeof(long)));
        return BuildRunnableLockedProbeForTest(
            tscFrequency, table, generations, sharedState, 0, int.MaxValue, int.MaxValue, MaxLockedSites);
    }

    // Test hook: a runnable locked-op probe over a caller's table (and, for shared state, generation
    // array) with a real nap target.
    internal static byte[] BuildRunnableLockedProbeForTest(
        ulong tscFrequency, byte* table, long* generations, bool sharedState, long thresholdTicks, long gapTicks,
        long densityTicks, int siteLimit)
    {
        lock (Gate)
        {
            var target = EnsureNapTargetLocked(tscFrequency) ?? throw new InvalidOperationException("No host sleep to nap with.");
            if (!sharedState && EnsureThreadKeyReader() == null)
            {
                throw new InvalidOperationException("No thread-key reader.");
            }

            return BuildLockedProbe(
                (ulong)table, (ulong)generations, thresholdTicks, gapTicks, densityTicks, siteLimit, target, sharedState);
        }
    }

    // Test hook: a runnable pause probe over a caller's table with a real nap target; naps are
    // counted (PauseNapCount) when statsAddress is set.
    internal static byte[] BuildRunnableProbeForTest(
        ulong tscFrequency, byte* table, bool sharedState, long thresholdTicks, long gapTicks, long delayTicks,
        ulong statsAddress)
    {
        lock (Gate)
        {
            var target = EnsureNapTargetLocked(tscFrequency) ?? throw new InvalidOperationException("No host sleep to nap with.");
            return BuildProbe((ulong)table, thresholdTicks, gapTicks, target, delayTicks, statsAddress, sharedState);
        }
    }

    // Test hooks: fresh tables in either layout, and fresh counters.
    internal static byte* AllocatePauseTableForTest(bool sharedState) => AllocatePauseTable(sharedState);

    internal static byte* AllocateLockedTableForTest(bool sharedState) => AllocateLockedTable(sharedState);

    internal static byte* AllocatePauseStatsForTest() => AllocateAlignedZeroed(PauseStatsSlotCount * PauseStatsSlotStride);

    /// <summary>The runtime's locked table, once the locked probe exists (test hook).</summary>
    internal static byte* RuntimeLockedTable => _lockedTable;

    internal static bool RuntimeLockedSharedState => _lockedSharedState;

    internal const int LockedHleGenerationOffsetForTest = LockedHleGenerationOffset;

    /// <summary>Naps the pause probes have counted so far (only probes built with counters).</summary>
    internal static long PauseNapCount => _control == null ? 0 : Volatile.Read(ref _control[ControlPauseNaps]);

    /// <summary>Naps the locked-op probes have taken so far, across modules.</summary>
    internal static long LockedNapCount => _control == null ? 0 : Volatile.Read(ref _control[ControlLockedNaps]);

    internal static long TicksFor(uint microseconds, ulong tscFrequency) =>
        (long)UInt128.Min((UInt128)microseconds * tscFrequency / 1_000_000, int.MaxValue);

    /// <summary>Rounds up, so a nonzero delay never becomes zero ticks on a coarse TSC.</summary>
    internal static long TicksForNanoseconds(uint nanoseconds, ulong tscFrequency) =>
        (long)UInt128.Min(((UInt128)nanoseconds * tscFrequency + 999_999_999) / 1_000_000_000, int.MaxValue);

    // Test hook: BuildProbe over caller-chosen addresses and thresholds, with a nap target that is
    // emitted but never resolved (only for comparing bytes; never run a probe whose nap can fire).
    internal static byte[] BuildProbeForTest(
        ulong tableAddress, long thresholdTicks, long gapTicks, ulong sleepAddress, uint sleepArgument, ulong control,
        long delayTicks, ulong statsAddress, bool sharedState) =>
        BuildProbe(
            tableAddress, thresholdTicks, gapTicks, new NapTarget(sleepAddress, sleepArgument, 0, 0, control), delayTicks,
            statsAddress, sharedState);

    // Test hook: BuildLockedProbe over caller-chosen addresses and thresholds (bytes only, like
    // BuildProbeForTest).
    internal static byte[] BuildLockedProbeForTest(
        ulong tableAddress, ulong generationsAddress, long thresholdTicks, long gapTicks, long densityTicks, int siteLimit,
        ulong sleepAddress, uint sleepArgument, ulong control, bool sharedState) =>
        BuildLockedProbe(
            tableAddress, generationsAddress, thresholdTicks, gapTicks, densityTicks, siteLimit,
            new NapTarget(sleepAddress, sleepArgument, 0, 0, control), sharedState);

    // Test hook: a runnable pause probe over a fresh streak table and fresh counters that never naps
    // (its idle threshold is the largest the probe takes), delaying each entry by delayNanoseconds.
    internal static byte[] BuildDelayingProbeForTest(ulong tscFrequency, uint delayNanoseconds, out nint stats)
    {
        var counters = AllocatePauseStatsForTest();
        stats = (nint)counters;
        return BuildRunnableProbeForTest(
            tscFrequency, AllocatePauseTable(sharedState: false), sharedState: false, int.MaxValue,
            TicksFor(DefaultGapMicroseconds, tscFrequency), TicksForNanoseconds(delayNanoseconds, tscFrequency),
            (ulong)counters);
    }

    // The probe, assembled position-independently (absolute imm64 operands, only self-relative
    // internal branches). Entered by `call` from a site trampoline whose rsp already sits below
    // the guest red zone; it preserves every register and flag. With sharedState, delayTicks == 0
    // and no statsAddress it is byte for byte the probe from before the pause delay existed.
    private static byte[] BuildProbe(
        ulong tableAddress, long thresholdTicks, long gapTicks, NapTarget target, long delayTicks, ulong statsAddress,
        bool sharedState)
    {
        var a = new Assembler(64);
        var newStreak = a.CreateLabel();
        var sameStreak = a.CreateLabel();
        var done = a.CreateLabel();

        a.pushfq();
        a.push(rax);
        a.push(rcx);
        a.push(rdx);
        a.push(rbx);
        a.push(rbp);
        a.push(rsi);
        a.push(rdi);
        a.push(r8);
        a.push(r9);
        a.push(r10);
        a.push(r11);
        a.push(r12);
        a.push(r13);
        a.push(r14);
        a.push(r15);

        // r11 = key: the host thread pointer, or (shared) rsp >> 26. rsp has shifted by a few KiB
        // since guest code ran, far less than the 64 MiB bucket, so the top bits are the guest
        // thread's.
        EmitStateKey(a, sharedState);

        // now = rdtsc (edx:eax) combined into r10.
        a.rdtsc();
        a.shl(rdx, 32);
        a.or(rdx, rax);
        a.mov(r10, rdx); // r10 = now
        EmitPauseDelay(a, delayTicks, statsAddress); // with a delay: r9 = now before it, r10 = after it

        if (sharedState)
        {
            // entry = table + (key & mask) * 32
            a.mov(rax, r11);
            a.and(rax, BucketMask);
            a.shl(rax, 5);
        }
        else
        {
            EmitPrivateSlotOffset(a, PrivateSlotBits, PrivateSlotStride); // rax = slot * 64
        }

        a.mov(rbx, tableAddress);
        a.add(rbx, rax); // rbx = &entry

        a.mov(rax, __qword_ptr[rbx]); // entry.key
        a.cmp(rax, r11);
        a.jne(newStreak);

        // same thread: if now - last > gap, the streak broke; restart the epoch. The gap ends where
        // this entry began, before its delay, so it is only the guest's own time between two pauses:
        // a delay as long as the gap must not read as a break on every entry.
        a.mov(rcx, delayTicks > 0 ? r9 : r10);
        a.sub(rcx, __qword_ptr[rbx + 16]); // now - last
        a.mov(rax, gapTicks);
        a.cmp(rcx, rax);
        a.jb(sameStreak);
        a.mov(__qword_ptr[rbx + 8], r10); // epoch = now

        a.Label(ref sameStreak);
        a.mov(__qword_ptr[rbx + 16], r10); // last = now
        a.mov(rcx, r10);
        a.sub(rcx, __qword_ptr[rbx + 8]); // now - epoch
        a.mov(rax, thresholdTicks);
        a.cmp(rcx, rax);
        a.jb(done);

        // Spun past the threshold: nap. The epoch stays, but `last` is this entry's pre-nap time, so
        // the thread's next pause sees the nap as a gap and starts a new streak: per-thread state
        // gives a 500 us spin / 250 us nap cycle (with the shared state, other threads' pauses kept
        // `last` fresh, so a bucket napped on every entry once anyone had spun past the threshold).
        EmitNap(a, target, napCounter: statsAddress != 0 ? ControlPauseNaps : -1, skip: done);
        a.jmp(done);

        a.Label(ref newStreak);
        a.mov(__qword_ptr[rbx], r11); // key
        a.mov(__qword_ptr[rbx + 8], r10); // epoch
        a.mov(__qword_ptr[rbx + 16], r10); // last

        a.Label(ref done);
        a.pop(r15);
        a.pop(r14);
        a.pop(r13);
        a.pop(r12);
        a.pop(r11);
        a.pop(r10);
        a.pop(r9);
        a.pop(r8);
        a.pop(rdi);
        a.pop(rsi);
        a.pop(rbp);
        a.pop(rbx);
        a.pop(rdx);
        a.pop(rcx);
        a.pop(rax);
        a.popfq();
        a.ret();

        using var stream = new MemoryStream();
        a.Assemble(new StreamCodeWriter(stream), 0);
        return stream.ToArray();
    }

    // The locked-op probe, assembled position-independently like BuildProbe. Per-thread window
    // (keyed on the host thread pointer, or rsp >> 26 with sharedState): a hit restarts the window
    // when the thread changed, the gap since the previous hit reached gapTicks, or the thread's HLE
    // generation moved. Otherwise it counts the hit against the first `siteLimit` distinct
    // call-site return addresses (the rest count as "other"). Once the window is thresholdTicks
    // old: nap if hits are dense (hits * densityTicks >= elapsed) and concentrated
    // (other * 10 <= hits); either way start a fresh window. Private state keeps the sites, the
    // density and the concentration per thread too: they live in the thread's slot. The HLE
    // generation is the slot's hleGen (private) or generationsAddress[bucket] (shared).
    private static byte[] BuildLockedProbe(
        ulong tableAddress, ulong generationsAddress, long thresholdTicks, long gapTicks, long densityTicks, int siteLimit,
        NapTarget target, bool sharedState)
    {
        var a = new Assembler(64);
        var reset = a.CreateLabel();
        var restart = a.CreateLabel();
        var clear = a.CreateLabel();
        var scan = a.CreateLabel();
        var claim = a.CreateLabel();
        var found = a.CreateLabel();
        var other = a.CreateLabel();
        var evaluate = a.CreateLabel();
        var done = a.CreateLabel();

        a.pushfq();
        a.push(rax);
        a.push(rcx);
        a.push(rdx);
        a.push(rbx);
        a.push(rbp);
        a.push(rsi);
        a.push(rdi);
        a.push(r8);
        a.push(r9);
        a.push(r10);
        a.push(r11);
        a.push(r12);
        a.push(r13);
        a.push(r14);
        a.push(r15);

        EmitStateKey(a, sharedState); // r11 = key
        a.rdtsc();
        a.shl(rdx, 32);
        a.or(rdx, rax);
        a.mov(r10, rdx); // r10 = now

        if (sharedState)
        {
            a.mov(rax, r11);
            a.and(rax, LockedBucketMask);
            a.mov(rcx, generationsAddress);
            a.mov(r9, __qword_ptr[rcx + rax * 8]); // r9 = this thread's HLE generation
            a.shl(rax, 8);
            a.mov(rbx, tableAddress);
            a.add(rbx, rax); // rbx = &entry
        }
        else
        {
            EmitPrivateSlotOffset(a, PrivateLockedSlotBits, LockedBucketStride); // rax = slot * 256
            a.mov(rbx, tableAddress);
            a.add(rbx, rax); // rbx = &entry
            a.mov(r9, __qword_ptr[rbx + LockedHleGenerationOffset]); // r9 = this thread's HLE generation
        }
        a.mov(r8, __qword_ptr[rsp + 128]); // r8 = site (return address into the site trampoline)

        a.cmp(__qword_ptr[rbx], r11);
        a.jne(reset);
        a.mov(rax, r10);
        a.sub(rax, __qword_ptr[rbx + 16]); // gap since the previous hit
        a.mov(__qword_ptr[rbx + 16], r10);
        a.mov(rcx, gapTicks);
        a.cmp(rax, rcx);
        a.jae(reset);
        a.cmp(r9, __qword_ptr[rbx + 40]);
        a.jne(reset);

        a.inc(__qword_ptr[rbx + 24]); // hits
        a.xor(ecx, ecx);
        a.Label(ref scan);
        a.cmp(rcx, siteLimit);
        a.jae(other);
        a.mov(rax, __qword_ptr[rbx + rcx * 8 + 64]);
        a.cmp(rax, r8);
        a.je(found);
        a.test(rax, rax);
        a.jz(claim);
        a.inc(rcx);
        a.jmp(scan);
        a.Label(ref claim);
        a.mov(__qword_ptr[rbx + rcx * 8 + 64], r8);
        a.Label(ref found);
        a.inc(__qword_ptr[rbx + rcx * 8 + 128]);
        a.jmp(evaluate);
        a.Label(ref other);
        a.inc(__qword_ptr[rbx + 32]);

        a.Label(ref evaluate);
        a.mov(rax, r10);
        a.sub(rax, __qword_ptr[rbx + 8]); // elapsed
        a.mov(rcx, thresholdTicks);
        a.cmp(rax, rcx);
        a.jb(done);
        a.mov(rcx, __qword_ptr[rbx + 24]);
        a.mov(rdx, densityTicks);
        a.imul(rcx, rdx);
        a.cmp(rcx, rax); // hits * density < elapsed: too sparse to be a poll
        a.jb(restart);
        a.mov(rcx, __qword_ptr[rbx + 32]);
        a.lea(rcx, __[rcx + rcx * 4]);
        a.add(rcx, rcx); // other * 10
        a.cmp(rcx, __qword_ptr[rbx + 24]);
        a.ja(restart); // under 90% of hits from the tracked sites

        EmitNap(a, target, ControlLockedNaps, skip: restart);

        a.Label(ref restart);
        a.mov(__qword_ptr[rbx + 8], r10);
        a.jmp(clear);

        a.Label(ref reset);
        a.mov(__qword_ptr[rbx], r11);
        a.mov(__qword_ptr[rbx + 8], r10);
        a.mov(__qword_ptr[rbx + 16], r10);
        a.mov(__qword_ptr[rbx + 40], r9);

        a.Label(ref clear);
        a.xor(eax, eax);
        a.mov(__qword_ptr[rbx + 24], rax);
        a.mov(__qword_ptr[rbx + 32], rax);
        for (var slot = 0; slot < MaxLockedSites; slot++)
        {
            a.mov(__qword_ptr[rbx + 64 + slot * 8], rax);
            a.mov(__qword_ptr[rbx + 128 + slot * 8], rax);
        }

        a.Label(ref done);
        a.pop(r15);
        a.pop(r14);
        a.pop(r13);
        a.pop(r12);
        a.pop(r11);
        a.pop(r10);
        a.pop(r9);
        a.pop(r8);
        a.pop(rdi);
        a.pop(rsi);
        a.pop(rbp);
        a.pop(rbx);
        a.pop(rdx);
        a.pop(rcx);
        a.pop(rax);
        a.popfq();
        a.ret();

        using var stream = new MemoryStream();
        a.Assemble(new StreamCodeWriter(stream), 0);
        return stream.ToArray();
    }

    // The pause delay and counters, emitted after the probe's first rdtsc (r10 = now) and before
    // the streak logic. The delay spins on rdtsc until delayTicks have passed since that read,
    // keeps that read in r9 and makes the end of the delay the probe's `now` (r10): the delay is
    // spin time, so it counts toward the idle threshold like the guest's own spinning; the streak
    // logic measures the gap since the previous entry's end from r9, so it covers only the guest's
    // loop body and a delay never looks like a break in the streak, however long. The elapsed
    // compare is unsigned, so a TSC that steps back (a migration) ends the wait instead of
    // stretching it. The overshoot is one loop pass, about one rdtsc (~8 ns under Rosetta) or the
    // TSC's granularity if coarser. No `pause` in the loop: under Rosetta it is the no-op this works
    // around, and on an x86 host (65-140 cycles) it would overshoot a ~19 ns delay several times over.
    // Uses rax, rcx, rdx, r8, r9; emits nothing when both are off.
    private static void EmitPauseDelay(Assembler a, long delayTicks, ulong statsAddress)
    {
        if (delayTicks > 0)
        {
            var spin = a.CreateLabel();
            a.mov(r9, r10); // the entry's start, for the gap
            a.Label(ref spin);
            a.rdtsc();
            a.shl(rdx, 32);
            a.or(rdx, rax);
            a.mov(rcx, rdx);
            a.sub(rcx, r10); // rcx = ticks waited so far
            a.cmp(rcx, (int)Math.Min(delayTicks, int.MaxValue));
            a.jb(spin);
        }

        if (statsAddress != 0)
        {
            // slot = stats + ((rsp >> 12) * golden >> (64 - bits)) * 64
            a.mov(rax, rsp);
            a.shr(rax, 12);
            a.mov(r8, PauseStatsHashMultiplier);
            a.imul(rax, r8);
            a.shr(rax, 64 - PauseStatsSlotBits);
            a.shl(rax, 6); // * PauseStatsSlotStride
            a.mov(r8, statsAddress);
            a.add(r8, rax);
            a.inc(__qword_ptr[r8]); // entries
            if (delayTicks > 0)
            {
                a.add(__qword_ptr[r8 + 8], rcx); // delayed ticks
            }
        }

        if (delayTicks > 0)
        {
            a.mov(r10, rdx); // now = end of the delay
        }
    }

    // r11 = the probe-state key: the host thread pointer (private) or rsp >> 26 (shared).
    private static void EmitStateKey(Assembler a, bool sharedState)
    {
        if (sharedState)
        {
            a.mov(r11, rsp);
            a.shr(r11, StackKeyShift);
        }
        else
        {
            EmitThreadKey(a, r11, r11d);
        }
    }

    // destination = the host thread pointer, one load from the thread's own control block through
    // the segment register the host OS points at it: fs:[0] on Linux (the x86-64 TLS ABI stores the
    // thread pointer there; guest fs accesses are emulated, so fs stays the host's), gs:[0] on macOS
    // (TSD slot 0, pthread_self; Rosetta translates gs-relative loads, the TlsGetValue stub relies
    // on it), gs:[0x30] on Windows (NT_TIB.Self). Unique per live thread, constant for its life, and
    // the same for every fiber the thread runs, so a fiber switch keeps the thread's streak.
    //
    // On Linux the load must not be `mov reg, fs:[0]` (no base, no index): the backend's guest-TLS
    // scan (DirectExecutionBackend.PatchDecodedTlsInstruction, at start and on every lazy commit of
    // executable memory) rewrites exactly that form, in every executable guest region, the spin-nap
    // arena included, into a call that returns the guest's TLS base. So it zeroes the register and
    // loads fs:[destination], a based form the scan skips. The xor clobbers flags, which the probes
    // have saved; the managed reader emits the same two instructions, so both see the same value.
    private static void EmitThreadKey(Assembler a, AssemblerRegister64 destination, AssemblerRegister32 destination32)
    {
        if (OperatingSystem.IsWindows())
        {
            a.mov(destination, __qword_ptr.gs[0x30]);
        }
        else if (OperatingSystem.IsMacOS())
        {
            a.mov(destination, __qword_ptr.gs[0]);
        }
        else
        {
            a.xor(destination32, destination32);
            a.mov(destination, __qword_ptr.fs[destination]);
        }
    }

    // rax = ((r11 * golden) >> (64 - bits)) * stride: Fibonacci hashing of the thread key, whose high
    // product bits depend on every key bit, so thread pointers a fixed stack size apart spread out.
    private static void EmitPrivateSlotOffset(Assembler a, int bits, int stride)
    {
        a.mov(rax, ThreadKeyHashMultiplier);
        a.imul(rax, r11);
        a.shr(rax, (byte)(64 - bits));
        a.shl(rax, (byte)BitOperations.Log2((uint)stride));
    }

    // A nap, emitted inline in a probe. On entry r10 = now (TSC) and the probe has already saved
    // every general register; rbx and r10 survive it. SHARPEMU_SPIN_NAP_AB_S=N (diagnostic): the
    // first nap anywhere sets the shared epoch, and naps are skipped in odd N-second periods from
    // it, so one run carries interleaved nap/spin phases over the same scenes. The host sleep is a
    // plain C call: x87/SSE state saved with fxsave64 (the guest is mid-function, so no vector
    // register is the callee's to clobber), DF clear, stack 16-byte aligned. With napCounter >= 0
    // the nap is counted in that control slot. A skipped nap jumps to `skip`; a taken one falls
    // through.
    private static void EmitNap(Assembler a, NapTarget target, int napCounter, Label skip)
    {
        a.mov(rsi, target.Control);
        if (target.AbPeriodTicks != 0)
        {
            var haveEpoch = a.CreateLabel();
            var napPhase = a.CreateLabel();
            a.mov(rdx, __qword_ptr[rsi + ControlAbEpoch * sizeof(long)]);
            a.test(rdx, rdx);
            a.jnz(haveEpoch);
            a.xor(eax, eax);
            a.@lock.cmpxchg(__qword_ptr[rsi + ControlAbEpoch * sizeof(long)], r10); // first nap: epoch = now
            a.mov(rdx, r10);
            a.test(rax, rax);
            a.cmovne(rdx, rax); // another thread set it first
            a.Label(ref haveEpoch);
            a.mov(rax, r10);
            a.sub(rax, rdx);
            a.jb(napPhase); // this core's TSC is a hair behind the epoch: still period 0
            a.xor(edx, edx);
            a.mov(rcx, (ulong)target.AbPeriodTicks);
            a.div(rcx);
            a.test(al, 1);
            a.jnz(skip); // odd period: spin
            a.Label(ref napPhase);
        }

        if (napCounter >= 0)
        {
            a.@lock.inc(__qword_ptr[rsi + napCounter * sizeof(long)]);
        }

        a.mov(r13, r10);
        a.mov(rbp, rsp);
        a.and(rsp, -16);
        a.sub(rsp, FxSaveBytes);
        a.fxsave64(__[rsp]);
        a.mov(r12, rsp);
        a.sub(rsp, 32); // Win64 shadow space; harmless on SysV
        a.cld();
        a.mov(edi, target.SleepArgument); // SysV first argument
        a.mov(ecx, target.SleepArgument); // Win64 first argument
        a.mov(rax, target.SleepAddress);
        a.call(rax);
        a.fxrstor64(__[r12]);
        a.mov(rsp, rbp);
        a.mov(r10, r13);
    }

    private static uint ReadMicroseconds(string variable, uint fallback) =>
        uint.TryParse(Environment.GetEnvironmentVariable(variable), out var value) && value is > 0 and <= 100_000
            ? value
            : fallback;
}
