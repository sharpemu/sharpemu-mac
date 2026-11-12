// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using SharpEmu.HLE.Host;

namespace SharpEmu.Libs.VideoOut.Overlay;

/// <summary>
/// The overlay's data plane: turns what the emulator reports (render-thread phases, GPU
/// command-buffer intervals, HLE call timing, per-thread CPU) into one immutable
/// <see cref="FrameRecord"/> per rendered frame, and serves snapshots to the overlay UI.
/// </summary>
/// <remarks>
/// <para>Hot paths only read plain static switches; everything else happens on a dedicated
/// low-priority builder thread that wakes once per frame. Nothing runs until a consumer asks
/// (<see cref="DataSource"/>), and the costly captures (render phases, timeline, HLE timing)
/// run only while a visible widget demands them. A consumer that stops capturing for two
/// seconds (overlay hidden) switches everything off again.</para>
/// <para>A "frame" is the render thread's frame: one <see cref="RenderPhaseProfile.RecordFrame"/>
/// call, which both backends make when a guest flip is published. Lanes are wall-clock: the GPU
/// lane shows what the GPU executed during the frame's interval, which on a pipelined title is
/// earlier frames' work.</para>
/// </remarks>
public static partial class OverlayTelemetry
{
    public const int HistoryCapacity = 300;
    internal const int HleNameBase = 1 << 16;
    internal const int DynamicNameBase = 1 << 24;

    // An HLE call at least this long is placed on the guest-main timeline lane.
    private static readonly long LongCallTicks = Stopwatch.Frequency / 10_000; // 100 µs
    // Set on a long call's import index (here and in the bridge's call ring) when more than half
    // of the call was blocked in host pacing; the timeline then draws it as a wait.
    internal const int MostlyBlockedCall = 1 << 30;
    private static readonly long IdleAfterTicks = Stopwatch.Frequency * 2;
    private const int MaxHleCostsPerFrame = 64;
    private const int MaxSegmentsPerLane = 512;

    // ---- switches read on hot paths (plain fields: a frame of staleness is harmless) ----
    internal static bool Active;
    internal static bool HleTimingActive;
    internal static bool TimelineActive;
    internal static bool PhasesActive;

    private static long _lastCaptureTimestamp;
    private static OverlayCaptureDemand _demand;
    private static Thread? _builder;
    private static readonly AutoResetEvent FrameSignal = new(false);

    // ---- render-thread frame handoff ----
    private static readonly object FrameGate = new();
    private static readonly Queue<RawFrame> PendingFrames = new();
    private static readonly Stack<RawFrame> FreeFrames = new();
    [ThreadStatic] private static ulong _osThreadId;

    // ---- GPU intervals (Stopwatch ticks) ----
    private static readonly object GpuGate = new();
    private static List<(long Start, long End)> _gpuIncoming = new(1024);
    private static List<(long Start, long End)> _gpuDraining = new(1024);
    private static double _machToStopwatchOffsetSeconds = double.NaN;

    // ---- counters ----
    private static long _draws;
    private static long _submits;
    private static long _dispatches;

    // ---- HLE timing ----
    private static readonly List<HleThreadSlots> HleThreads = [];
    [ThreadStatic] private static HleThreadSlots? _hleSlots;
    [ThreadStatic] private static bool _guestThreadNoted;
    private static readonly ConcurrentDictionary<ulong, byte> GuestThreadIds = new();
    private static Func<int, string?>? _hleNameResolver;

    // ---- names ----
    private static readonly object NameGate = new();
    private static readonly Dictionary<string, int> DynamicNameIds = new(StringComparer.Ordinal);
    private static readonly List<string> DynamicNames = [];
    private static readonly ConcurrentDictionary<int, string> HleNames = new();
    private static readonly ConcurrentDictionary<int, bool> HleIsWait = new();
    private static readonly string[] PhaseNames = Enum.GetNames<RenderPhaseProfile.Phase>();

    // ---- published state ----
    private static readonly object HistoryGate = new();
    private static readonly FrameRecord?[] History = new FrameRecord?[HistoryCapacity];
    private static long _historyCount;
    private static SystemStats _system = SystemStats.Empty;

    /// <summary>The emulator's snapshot source for the overlay UI.</summary>
    public static IOverlayDataSource DataSource { get; } = new Source();

    /// <summary>The colour class of each render phase (index by <see cref="RenderPhaseProfile.Phase"/>).</summary>
    internal static readonly CostKind[] PhaseKinds = BuildPhaseKinds();

    // =====================================================================================
    // Feeds
    // =====================================================================================

    /// <summary>One GPU command buffer's execution interval, in Metal's host-time seconds (GPUStartTime/GPUEndTime).</summary>
    public static void RecordGpuIntervalHostSeconds(double startSeconds, double endSeconds)
    {
        if (!Active || !(endSeconds > startSeconds))
        {
            return;
        }

        var offset = _machToStopwatchOffsetSeconds;
        if (double.IsNaN(offset))
        {
            offset = Stopwatch.GetTimestamp() / (double)Stopwatch.Frequency - MachClock.Seconds();
            _machToStopwatchOffsetSeconds = offset;
        }

        RecordGpuInterval(
            (long)((startSeconds + offset) * Stopwatch.Frequency),
            (long)((endSeconds + offset) * Stopwatch.Frequency));
    }

    /// <summary>One GPU execution interval in <see cref="Stopwatch"/> ticks (Vulkan timestamp queries mapped to host time).</summary>
    public static void RecordGpuInterval(long startTicks, long endTicks)
    {
        if (!Active || endTicks <= startTicks)
        {
            return;
        }

        lock (GpuGate)
        {
            if (_gpuIncoming.Count < 1 << 16)
            {
                _gpuIncoming.Add((startTicks, endTicks));
            }
        }
    }

    internal static void RecordDraw() => Interlocked.Increment(ref _draws);
    internal static void RecordSubmit() => Interlocked.Increment(ref _submits);
    public static void RecordDispatch() => Interlocked.Increment(ref _dispatches);

    /// <summary>Names HLE import slots (e.g. "libKernel:sceKernelWaitEqueue"); called lazily, once per slot.</summary>
    public static void SetHleNameResolver(Func<int, string?> resolver) => Volatile.Write(ref _hleNameResolver, resolver);

    /// <summary>Marks the calling host thread as one that runs guest code; cheap after the first call.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void NoteGuestThread()
    {
        if (!_guestThreadNoted)
        {
            NoteGuestThreadSlow();
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void NoteGuestThreadSlow()
    {
        _guestThreadNoted = true;
        GuestThreadIds.TryAdd(OsThreadId(), 0);
        if (Bridge.IsSplitGuest)
        {
            Bridge.StartGuestPublisher();
        }
    }

    /// <summary>
    /// The calling thread's blocked-time mark (<see cref="HostBlockedTime.Total"/>); the gateway reads it
    /// before a timed call and passes it to <see cref="RecordHleCall(int, long, long, long)"/>.
    /// </summary>
    public static long HleBlockedMark
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => HostBlockedTime.Total;
    }

    /// <summary>Charges one HLE call that never blocks in host pacing.</summary>
    internal static void RecordHleCall(int importIndex, long startTicks, long endTicks) =>
        RecordHleCall(importIndex, startTicks, endTicks, HostBlockedTime.Total);

    /// <summary>
    /// Charges one HLE call (Stopwatch ticks, gateway entry to exit) to the calling thread. The time the
    /// thread spent blocked in host pacing since <paramref name="blockedMark"/> (read with
    /// <see cref="HleBlockedMark"/> at entry) is kept apart from the body, so it counts as a wait.
    /// </summary>
    /// <remarks>Only call while <see cref="HleTimingActive"/>; the gateway checks the switch.</remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void RecordHleCall(int importIndex, long startTicks, long endTicks, long blockedMark)
    {
        var slots = _hleSlots ?? RegisterHleThread();
        var arrays = slots.Arrays;
        if ((uint)importIndex >= (uint)arrays.Ticks.Length)
        {
            if (importIndex < 0)
            {
                return;
            }

            arrays = slots.Grow(importIndex);
        }

        var elapsed = endTicks - startTicks;
        arrays.Ticks[importIndex] += elapsed;
        arrays.Counts[importIndex]++;
        // The mark only grows on this thread, so the difference is this call's blocking alone.
        var blocked = HostBlockedTime.Total - blockedMark;
        if (blocked != 0)
        {
            blocked = Math.Min(blocked, elapsed);
            arrays.Blocked[importIndex] += blocked;
        }

        if (elapsed >= LongCallTicks && TimelineActive)
        {
            slots.AppendLongCall(blocked * 2 > elapsed ? importIndex | MostlyBlockedCall : importIndex, startTicks, endTicks);
        }
    }

    public static bool IsHleTimingActive
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => HleTimingActive;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static HleThreadSlots RegisterHleThread()
    {
        var slots = new HleThreadSlots(OsThreadId());
        lock (HleThreads)
        {
            HleThreads.Add(slots);
        }

        _hleSlots = slots;
        return slots;
    }

    internal static void PublishSystemStats(SystemStats stats) => Volatile.Write(ref _system, stats);

    /// <summary>
    /// Ends the render thread's frame. Called from <see cref="RenderPhaseProfile.RecordFrame"/> with that
    /// frame's per-phase self ticks (null when phase capture is off) and the thread's timeline segments.
    /// </summary>
    internal static void EndRenderFrame(long now, long[]? phaseTicks, RenderSegmentBuffer? segments)
    {
        RawFrame? frame;
        lock (FrameGate)
        {
            frame = FreeFrames.Count > 0 ? FreeFrames.Pop() : PendingFrames.Count < 16 ? new RawFrame() : null;
        }

        if (frame is null)
        {
            // The builder is behind; drop this frame's detail but keep its boundary.
            segments?.Clear();
            return;
        }

        frame.EndTimestamp = now;
        frame.ThreadId = OsThreadId();
        frame.HasPhases = phaseTicks is not null;
        if (phaseTicks is not null)
        {
            Array.Copy(phaseTicks, frame.PhaseTicks, Math.Min(phaseTicks.Length, frame.PhaseTicks.Length));
        }

        frame.SegmentCount = 0;
        if (segments is not null)
        {
            frame.CopySegments(segments);
            segments.Clear();
        }

        lock (FrameGate)
        {
            PendingFrames.Enqueue(frame);
        }

        FrameSignal.Set();
    }

    // =====================================================================================
    // Names
    // =====================================================================================

    internal static int InternName(string name)
    {
        lock (NameGate)
        {
            if (!DynamicNameIds.TryGetValue(name, out var id))
            {
                id = DynamicNameBase + DynamicNames.Count;
                DynamicNames.Add(name);
                DynamicNameIds.Add(name, id);
            }

            return id;
        }
    }

    internal static string NameOf(int nameId)
    {
        if ((uint)nameId < (uint)PhaseNames.Length)
        {
            return PhaseNames[nameId];
        }

        if (nameId >= DynamicNameBase)
        {
            lock (NameGate)
            {
                var index = nameId - DynamicNameBase;
                return index < DynamicNames.Count ? DynamicNames[index] : "?";
            }
        }

        if (nameId >= HleNameBase)
        {
            return HleName(nameId - HleNameBase);
        }

        return "?";
    }

    private static string HleName(int importIndex)
    {
        if (HleNames.TryGetValue(importIndex, out var cached))
        {
            return cached;
        }

        string? name = null;
        try
        {
            // This process's import table, or in the GPU host of the split the guest's published names.
            name = Volatile.Read(ref _hleNameResolver)?.Invoke(importIndex) ?? Bridge.NameOf(importIndex);
        }
        catch (Exception exception) when (exception is IndexOutOfRangeException or ArgumentOutOfRangeException)
        {
        }

        if (string.IsNullOrEmpty(name))
        {
            // Not cached: the guest may publish the name a moment later.
            return $"import #{importIndex}";
        }

        HleNames[importIndex] = name;
        return name;
    }

    private static bool IsWaitImport(int importIndex)
    {
        if (HleIsWait.TryGetValue(importIndex, out var known))
        {
            return known;
        }

        // Only a resolved name is classified for good: the split guest publishes names a moment
        // after the first calls, and the placeholder would pin a wait as a cost.
        var name = HleName(importIndex);
        var wait = IsWaitName(name);
        if (HleNames.ContainsKey(importIndex))
        {
            HleIsWait[importIndex] = wait;
        }

        return wait;
    }

    /// <summary>True for exports that block the calling thread (waits, sleeps, yields, joins, mutex locks).</summary>
    internal static bool IsWaitName(string name)
    {
        var colon = name.LastIndexOf(':');
        var bare = colon >= 0 ? name.AsSpan(colon + 1) : name.AsSpan();
        return bare.Contains("Wait", StringComparison.OrdinalIgnoreCase) ||
            bare.Contains("sleep", StringComparison.OrdinalIgnoreCase) ||
            bare.Contains("Yield", StringComparison.OrdinalIgnoreCase) ||
            bare.Contains("Join", StringComparison.OrdinalIgnoreCase) ||
            bare.Contains("PollEvent", StringComparison.OrdinalIgnoreCase) ||
            bare.Contains("Nap", StringComparison.Ordinal) ||
            (bare.Contains("Mutex", StringComparison.OrdinalIgnoreCase) &&
                bare.Contains("Lock", StringComparison.OrdinalIgnoreCase) &&
                !bare.Contains("Unlock", StringComparison.OrdinalIgnoreCase));
    }

    private static CostKind[] BuildPhaseKinds()
    {
        var kinds = new CostKind[(int)RenderPhaseProfile.Phase.Count];
        for (var index = 0; index < kinds.Length; index++)
        {
            kinds[index] = ClassifyPhase(((RenderPhaseProfile.Phase)index).ToString());
        }

        return kinds;
    }

    internal static CostKind ClassifyPhase(string name)
    {
        if (name.StartsWith("Idle", StringComparison.Ordinal) ||
            name.Contains("Wait", StringComparison.Ordinal) ||
            name is "SubmissionCapacity" or "Acquire" or "QueuePresent")
        {
            return CostKind.Wait;
        }

        if (name is "ProgramCompile" or "PipelineCreation" or "GraphicsPipelineDriver" or "ComputePipelineDriver" or
            "ProgramTranslation" or "ProgramDecode" or "ProgramPlanning" or "ProgramLayoutRequest" or
            "ProgramSynchronousResolution" or "GraphicsProgramCache" or "ProgramPreparation" or "ProgramPrepareLocked")
        {
            return CostKind.Shader;
        }

        if (name is "Unattributed" or "CursorUpdate" or "GamepadPoll" or "MovieFramePolling" ||
            name.StartsWith("Window", StringComparison.Ordinal))
        {
            return CostKind.Other;
        }

        return CostKind.Render;
    }

    private static ulong OsThreadId()
    {
        var id = _osThreadId;
        if (id == 0)
        {
            id = OverlayThreadSampler.CurrentThreadId();
            _osThreadId = id;
        }

        return id;
    }

    // =====================================================================================
    // Consumer side
    // =====================================================================================

    private sealed class Source : IOverlayDataSource
    {
        public void SetCaptureDemand(OverlayCaptureDemand demand)
        {
            _demand = demand;
            Touch();
        }

        public OverlaySnapshot Capture()
        {
            Touch();
            FrameRecord[] frames;
            lock (HistoryGate)
            {
                var count = (int)Math.Min(_historyCount, HistoryCapacity);
                frames = new FrameRecord[count];
                var first = _historyCount - count;
                for (var index = 0; index < count; index++)
                {
                    frames[index] = History[(first + index) % HistoryCapacity]!;
                }
            }

            return new OverlaySnapshot(frames, Volatile.Read(ref _system), NameOf);
        }
    }

    private static void Touch()
    {
        Volatile.Write(ref _lastCaptureTimestamp, Stopwatch.GetTimestamp());
        ApplyDemand(_demand);
        if (_builder is null && !ManualBuildForTests)
        {
            lock (FrameGate)
            {
                if (_builder is null)
                {
                    var builder = new Thread(BuildLoop)
                    {
                        IsBackground = true,
                        Name = "SharpEmu overlay telemetry",
                        Priority = ThreadPriority.BelowNormal,
                    };
                    _builder = builder;
                    builder.Start();
                }
            }
        }
    }

    private static void ApplyDemand(OverlayCaptureDemand demand)
    {
        PhasesActive = (demand & (OverlayCaptureDemand.RenderPhases | OverlayCaptureDemand.Timeline)) != 0;
        TimelineActive = (demand & OverlayCaptureDemand.Timeline) != 0;
        HleTimingActive = (demand & (OverlayCaptureDemand.HleCalls | OverlayCaptureDemand.Timeline)) != 0;
        Active = true;
        RenderPhaseProfile.SetOverlayCapture(PhasesActive, TimelineActive);
    }

    private static void Deactivate()
    {
        Bridge.WriteDemand(OverlayCaptureDemand.None);
        Active = false;
        PhasesActive = false;
        TimelineActive = false;
        HleTimingActive = false;
        RenderPhaseProfile.SetOverlayCapture(false, false);
    }

    /// <summary>Test hook: no builder thread; tests call <see cref="BuildPendingForTests"/>.</summary>
    internal static bool ManualBuildForTests;

    /// <summary>Test hook: drops all history and state.</summary>
    internal static void ResetForTests()
    {
        lock (HistoryGate)
        {
            Array.Clear(History);
            _historyCount = 0;
        }

        lock (FrameGate)
        {
            PendingFrames.Clear();
        }

        Builder.Reset();
        Deactivate();
    }

    /// <summary>Test hook: builds every pending raw frame now, on the calling thread.</summary>
    internal static void BuildPendingForTests() => Builder.DrainPending(force: true);

    private static void BuildLoop()
    {
        var lastStats = 0L;
        var settling = false;
        while (true)
        {
            FrameSignal.WaitOne(settling ? 10 : 250);
            settling = false;
            try
            {
                var now = Stopwatch.GetTimestamp();
                if (Active && now - Volatile.Read(ref _lastCaptureTimestamp) > IdleAfterTicks)
                {
                    Deactivate();
                }

                if (!Active)
                {
                    lock (FrameGate)
                    {
                        while (PendingFrames.Count > 0) FreeFrames.Push(PendingFrames.Dequeue());
                    }

                    continue;
                }

                settling = Builder.DrainPending(force: false);
                if (now - lastStats >= Stopwatch.Frequency)
                {
                    lastStats = now;
                    PerfOverlay.RefreshStatsForTelemetry();
                }
            }
            catch (Exception exception)
            {
                // The overlay must never take the emulator down.
                Console.Error.WriteLine($"[OVERLAY][TELEMETRY] {exception.GetType().Name}: {exception.Message}");
            }
        }
    }

    // =====================================================================================
    // Frame assembly (builder thread only)
    // =====================================================================================

    private static class Builder
    {
        private static readonly long SettleTicks = Stopwatch.Frequency / 40; // 25 ms
        private static long _previousEnd;
        private static long _previousDraws;
        private static long _previousSubmits;
        private static long _previousDispatches;
        private static TimeSpan _previousGcPause = GC.GetTotalPauseDuration();
        private static readonly int[] PreviousCollections = new int[3];
        private static long _sequence;
        private static readonly List<(long Start, long End)> GpuCarry = new(256);
        private static readonly float[] RecentFrameMs = new float[60];
        private static int _recentCount;

        // Thread CPU.
        private static readonly OverlayThreadSampler? Sampler =
            OverlayThreadSampler.CreateForCurrentProcess(PerfOverlay.GuestProcessId);
        private static readonly List<OverlayThreadSampler.ThreadCpu> ThreadScratch = new(256);
        private static readonly Dictionary<ulong, ThreadTrack> Threads = [];
        private static long _lastThreadSample;
        private static ulong _guestMainId;
        // The guest worker with the most CPU in the last frame: its calls fill the workers lane.
        private static ulong _busiestWorkerId;

        // HLE.
        // Per import: whole-call ticks, the part of them blocked in host pacing, and calls.
        private static long[] _hleMainTicks = new long[512];
        private static long[] _hleMainBlocked = new long[512];
        private static int[] _hleMainCounts = new int[512];
        private static long[] _hleWorkerTicks = new long[512];
        private static long[] _hleWorkerBlocked = new long[512];
        private static int[] _hleWorkerCounts = new int[512];
        private static readonly List<int> HleTouched = new(256);
        private static readonly List<CostSample> CostScratch = new(256);
        // The last captured frame's render lane, standing for the frames between samples (RenderPhaseProfile.OverlayPhaseSample).
        private static readonly List<CostSample> SampledRenderCosts = new(256);
        private static readonly List<TimelineSegment> SampledRenderSegments = new(1024);
        private static readonly float[] SampledRenderKindMs = new float[(int)CostKind.Count];
        private static float _sampledRenderBusy, _sampledRenderWait;
        private static bool _hasSampledRender;
        private static readonly List<TimelineSegment> SegmentScratch = new(1024);
        private static readonly List<ThreadSample> ThreadSampleScratch = new(128);
        private static readonly List<(long Start, long End, int Import)> LongCallScratch = new(128);
        // The split guest's long calls not yet placed (they can straddle frames).
        private static readonly List<(long Start, long End, int Import, OverlayLane Lane)> RemoteCalls = new(1024);

        private sealed class ThreadTrack
        {
            public long LastCpu = -1;
            public double BusyMs;
            public double Ema;
            public int NameId;
            public string Name = string.Empty;
            public ThreadClass Class;
            public long SeenSample;
        }

        private enum ThreadClass : byte { Guest, Render, RenderHelper, ShaderHelper, Other }

        internal static void Reset()
        {
            _previousEnd = 0;
            _sequence = 0;
            _recentCount = 0;
            GpuCarry.Clear();
            Threads.Clear();
            _guestMainId = 0;
            _busiestWorkerId = 0;
            RemoteCalls.Clear();
            _hasSampledRender = false;
            SampledRenderCosts.Clear();
            SampledRenderSegments.Clear();
        }

        /// <summary>
        /// Builds pending frames. GPU completions reach the host a few milliseconds after the
        /// work ends, so the newest frame waits <see cref="SettleTicks"/> unless forced.
        /// </summary>
        internal static bool DrainPending(bool force = true)
        {
            while (true)
            {
                RawFrame? raw;
                lock (FrameGate)
                {
                    if (!PendingFrames.TryPeek(out raw))
                    {
                        return false;
                    }

                    if (!force && PendingFrames.Count == 1 &&
                        Stopwatch.GetTimestamp() - raw.EndTimestamp < SettleTicks)
                    {
                        return true;
                    }

                    PendingFrames.Dequeue();
                }

                try
                {
                    Bridge.TryOpen(PerfOverlay.GuestProcessId);
                    Bridge.WriteDemand(_demand);
                    Bridge.WriteFocus(_guestMainId, _busiestWorkerId);
                    Bridge.ReadGuestThreads();
                    var record = Build(raw);
                    if (record is not null)
                    {
                        lock (HistoryGate)
                        {
                            History[_historyCount % HistoryCapacity] = record;
                            _historyCount++;
                        }
                    }
                }
                finally
                {
                    lock (FrameGate)
                    {
                        FreeFrames.Push(raw);
                    }
                }
            }
        }

        private static FrameRecord? Build(RawFrame raw)
        {
            var end = raw.EndTimestamp;
            var start = _previousEnd;
            _previousEnd = end;
            if (start == 0 || end <= start)
            {
                // The first frame only opens the interval.
                SampleThreads(end, 0, raw.ThreadId);
                TakeHleDeltas();
                return null;
            }

            var frequency = (double)Stopwatch.Frequency;
            var frameMs = (float)((end - start) * 1000.0 / frequency);
            float ToMs(long ticks) => (float)(ticks * 1000.0 / frequency);
            float Relative(long timestamp) => Math.Clamp((float)((timestamp - start) * 1000.0 / frequency), 0f, frameMs);

            var kindMs = new float[(int)CostKind.Count];
            CostScratch.Clear();
            SegmentScratch.Clear();

            // ---- render thread ----
            float renderBusy = 0, renderWait = 0;
            var sampledRender = false;
            if (raw.HasPhases)
            {
                Array.Clear(SampledRenderKindMs);
                SampledRenderCosts.Clear();
                for (var phase = 0; phase < raw.PhaseTicks.Length; phase++)
                {
                    var ticks = raw.PhaseTicks[phase];
                    if (ticks <= 0) continue;
                    var ms = ToMs(ticks);
                    var kind = PhaseKinds[phase];
                    if (kind == CostKind.Wait) renderWait += ms;
                    else renderBusy += ms;
                    kindMs[(int)kind] += ms;
                    SampledRenderKindMs[(int)kind] += ms;
                    if (ms >= 0.005f)
                    {
                        var sample = new CostSample(phase, kind, OverlayLane.Render, ms, 1);
                        CostScratch.Add(sample);
                        SampledRenderCosts.Add(sample);
                    }
                }

                _sampledRenderBusy = renderBusy;
                _sampledRenderWait = renderWait;
                _hasSampledRender = true;
            }
            else if (_hasSampledRender && RenderPhaseProfile.OverlayPhasesSampled)
            {
                // A frame between two captured ones: the last captured frame's render breakdown stands for it,
                // scaled down to this frame when that one was longer.
                var total = _sampledRenderBusy + _sampledRenderWait;
                var scale = total > frameMs && total > 0 ? frameMs / total : 1f;
                for (var kind = 0; kind < SampledRenderKindMs.Length; kind++)
                {
                    kindMs[kind] += SampledRenderKindMs[kind] * scale;
                }

                foreach (var sample in SampledRenderCosts)
                {
                    CostScratch.Add(sample with { Milliseconds = sample.Milliseconds * scale });
                }

                renderBusy = _sampledRenderBusy * scale;
                renderWait = _sampledRenderWait * scale;
                sampledRender = true;
            }

            if (raw.HasPhases)
            {
                SampledRenderSegments.Clear();
            }

            for (var index = 0; index < raw.SegmentCount && index < MaxSegmentsPerLane; index++)
            {
                var segmentStart = Relative(raw.SegmentStart[index]);
                var segmentEnd = Relative(raw.SegmentEnd[index]);
                if (segmentEnd > segmentStart)
                {
                    var segment = new TimelineSegment(segmentStart, segmentEnd, OverlayLane.Render,
                        (CostKind)raw.SegmentKind[index], raw.SegmentPhase[index]);
                    SegmentScratch.Add(segment);
                    if (raw.HasPhases)
                    {
                        SampledRenderSegments.Add(segment);
                    }
                }
            }

            if (sampledRender)
            {
                // Positions are from the captured frame's start, clipped to this frame.
                foreach (var segment in SampledRenderSegments)
                {
                    if (segment.StartMs < frameMs)
                    {
                        SegmentScratch.Add(segment with { EndMs = Math.Min(segment.EndMs, frameMs) });
                    }
                }
            }

            // ---- GPU ----
            var gpuMs = CollectGpu(start, end, frameMs, Relative);
            kindMs[(int)CostKind.Gpu] += gpuMs;

            // ---- threads ----
            SampleThreads(end, frameMs, raw.ThreadId);
            float guestMain = 0, guestWorkers = 0, renderThreadCpu = 0;
            var activeWorkers = 0;
            var busiestWorker = 0f;
            ulong busiestWorkerId = 0;
            ThreadSampleScratch.Clear();
            foreach (var (id, track) in Threads)
            {
                if (track.SeenSample != _lastThreadSample || track.BusyMs <= 0.01)
                {
                    continue;
                }

                var busy = (float)track.BusyMs;
                var lane = track.Class == ThreadClass.Guest
                    ? id == _guestMainId ? OverlayLane.GuestMain : OverlayLane.GuestWorkers
                    : OverlayLane.Render;
                ThreadSampleScratch.Add(new ThreadSample(track.NameId, lane, busy));
                switch (track.Class)
                {
                    case ThreadClass.Guest when id == _guestMainId:
                        guestMain += busy;
                        break;
                    case ThreadClass.Guest:
                        guestWorkers += busy;
                        if (busy > 0.1f) activeWorkers++;
                        if (busy > busiestWorker)
                        {
                            busiestWorker = busy;
                            busiestWorkerId = id;
                        }

                        break;
                    case ThreadClass.Render:
                        renderThreadCpu += busy;
                        break;
                    case ThreadClass.RenderHelper:
                        kindMs[(int)CostKind.Render] += busy;
                        break;
                    case ThreadClass.ShaderHelper:
                        kindMs[(int)CostKind.Shader] += busy;
                        break;
                    default:
                        kindMs[(int)CostKind.Other] += busy;
                        break;
                }
            }

            _busiestWorkerId = busiestWorkerId;
            if (!raw.HasPhases && !sampledRender)
            {
                renderBusy = Math.Min(renderThreadCpu, frameMs);
                renderWait = Math.Max(0, frameMs - renderBusy);
                kindMs[(int)CostKind.Render] += renderBusy;
            }

            // ---- HLE ----
            var (hleBody, hleWait) = TakeHleDeltas();
            kindMs[(int)CostKind.Hle] += hleBody;
            kindMs[(int)CostKind.Wait] += hleWait;
            kindMs[(int)CostKind.Guest] += Math.Max(0, guestMain + guestWorkers - hleBody);
            if (HleTimingActive)
            {
                AddHleCosts();
            }

            AddGuestSegments(start, end, frameMs, guestMain, guestWorkers, activeWorkers, Relative);

            // ---- GC and counters ----
            var gcPause = GC.GetTotalPauseDuration();
            var gcPauseMs = (float)(gcPause - _previousGcPause).TotalMilliseconds;
            var (guestGcMs, guestCollections, guestGeneration) = Bridge.TakeGuestGc();
            _previousGcPause = gcPause;
            var collections = 0;
            var highestGeneration = -1;
            for (var generation = 0; generation < 3; generation++)
            {
                var count = GC.CollectionCount(generation);
                var delta = count - PreviousCollections[generation];
                PreviousCollections[generation] = count;
                if (delta > 0)
                {
                    highestGeneration = generation;
                }

                if (generation == 0) collections = delta; // gen0 count includes the higher ones
            }

            // In the split the guest's collections stop the guest threads; they matter more than the host's.
            gcPauseMs += guestGcMs;
            collections += guestCollections;
            highestGeneration = Math.Max(highestGeneration, guestGeneration);

            if (gcPauseMs > 0)
            {
                kindMs[(int)CostKind.Other] += gcPauseMs;
                CostScratch.Add(new CostSample(InternName("GC pause"), CostKind.Other, OverlayLane.GuestMain, gcPauseMs, Math.Max(1, collections)));
            }

            var draws = Interlocked.Read(ref _draws);
            var submits = Interlocked.Read(ref _submits);
            var dispatches = Interlocked.Read(ref _dispatches);

            var (hitch, reason) = DetectHitch(frameMs, gcPauseMs, highestGeneration, kindMs[(int)CostKind.Shader],
                gpuMs, guestMain, renderBusy);

            var record = new FrameRecord
            {
                Sequence = ++_sequence,
                EndTimestamp = end,
                FrameMs = frameMs,
                GuestFrameMs = frameMs,
                GuestMainMs = guestMain,
                GuestWorkersMs = guestWorkers,
                ActiveWorkers = activeWorkers,
                RenderBusyMs = renderBusy,
                RenderWaitMs = renderWait,
                GpuMs = gpuMs,
                KindMs = kindMs,
                Draws = (int)(draws - _previousDraws),
                Dispatches = (int)(dispatches - _previousDispatches),
                Submits = (int)(submits - _previousSubmits),
                GcPauseMs = gcPauseMs,
                GcCollections = collections,
                Hitch = hitch,
                HitchReason = reason,
                Segments = SegmentScratch.ToArray(),
                Costs = CostScratch.ToArray(),
                Threads = ThreadSampleScratch.ToArray(),
            };
            _previousDraws = draws;
            _previousSubmits = submits;
            _previousDispatches = dispatches;
            return record;
        }

        private static float CollectGpu(long start, long end, float frameMs, Func<long, float> relative)
        {
            lock (GpuGate)
            {
                (_gpuIncoming, _gpuDraining) = (_gpuDraining, _gpuIncoming);
            }

            GpuCarry.AddRange(_gpuDraining);
            _gpuDraining.Clear();
            if (GpuCarry.Count == 0)
            {
                return 0;
            }

            GpuCarry.Sort(static (left, right) => left.Start.CompareTo(right.Start));
            var unionTicks = 0L;
            long runStart = 0, runEnd = 0;
            var open = false;
            var gapTicks = Stopwatch.Frequency / 50_000; // 20 µs: merge for display
            var segments = 0;
            var keep = 0;
            for (var index = 0; index < GpuCarry.Count; index++)
            {
                var (intervalStart, intervalEnd) = GpuCarry[index];
                if (intervalEnd > end)
                {
                    // Still running past the frame's end: count its in-frame part and keep it.
                    GpuCarry[keep++] = (Math.Max(intervalStart, end), intervalEnd);
                }

                var clippedStart = Math.Max(intervalStart, start);
                var clippedEnd = Math.Min(intervalEnd, end);
                if (clippedEnd <= clippedStart)
                {
                    continue;
                }

                if (open && clippedStart <= runEnd + gapTicks)
                {
                    if (clippedEnd > runEnd)
                    {
                        unionTicks += clippedEnd - Math.Max(clippedStart, runEnd);
                        runEnd = clippedEnd;
                    }

                    continue;
                }

                if (open && segments < MaxSegmentsPerLane)
                {
                    SegmentScratch.Add(new TimelineSegment(relative(runStart), relative(runEnd), OverlayLane.Gpu, CostKind.Gpu, GpuNameId));
                    segments++;
                }

                runStart = clippedStart;
                runEnd = clippedEnd;
                unionTicks += clippedEnd - clippedStart;
                open = true;
            }

            if (open && segments < MaxSegmentsPerLane)
            {
                SegmentScratch.Add(new TimelineSegment(relative(runStart), relative(runEnd), OverlayLane.Gpu, CostKind.Gpu, GpuNameId));
            }

            GpuCarry.RemoveRange(keep, GpuCarry.Count - keep);
            return Math.Min(frameMs, (float)(unionTicks * 1000.0 / Stopwatch.Frequency));
        }

        private static readonly int GpuNameId = InternName("GPU command buffers");
        private static readonly int GuestCpuNameId = InternName("Guest CPU (not placed in time)");
        private static readonly int WorkerCpuNameId = InternName("Average worker CPU (not placed in time)");
        private static readonly int GuestRunNameId = InternName("Guest code");

        private static void SampleThreads(long now, float frameMs, ulong renderThreadId)
        {
            if (Sampler is null)
            {
                return;
            }

            var elapsedMs = _lastThreadSample == 0 ? 0 : (now - _lastThreadSample) * 1000.0 / Stopwatch.Frequency;
            if (_lastThreadSample != 0 && elapsedMs < 4)
            {
                // Frames closer than 4 ms: reuse the last rates instead of another full sample.
                foreach (var track in Threads.Values)
                {
                    if (track.SeenSample == _lastThreadSample) track.BusyMs = track.Ema * frameMs;
                }

                return;
            }

            if (!Sampler.Sample(ThreadScratch))
            {
                return;
            }

            var scale = elapsedMs > 0 && frameMs > 0 ? frameMs / elapsedMs : 0;
            double bestEma = -1;
            ulong bestId = 0;
            foreach (var thread in ThreadScratch)
            {
                if (!Threads.TryGetValue(thread.ThreadId, out var track))
                {
                    track = new ThreadTrack();
                    Threads[thread.ThreadId] = track;
                }

                if (!string.Equals(track.Name, thread.Name, StringComparison.Ordinal) || track.NameId == 0)
                {
                    track.Name = thread.Name;
                    track.NameId = InternName(string.IsNullOrEmpty(thread.Name) ? $"thread {thread.ThreadId}" : thread.Name);
                }

                track.Class = Classify(thread.ThreadId, thread.Name, renderThreadId, thread.Remote);
                var delta = track.LastCpu < 0 ? 0 : Math.Max(0, thread.CpuNanoseconds - track.LastCpu);
                track.LastCpu = thread.CpuNanoseconds;
                var busyMs = delta / 1_000_000.0 * scale;
                track.BusyMs = Math.Min(busyMs, frameMs);
                var rate = elapsedMs > 0 ? delta / 1_000_000.0 / elapsedMs : 0;
                track.Ema = track.SeenSample == 0 ? rate : track.Ema * 0.9 + rate * 0.1;
                track.SeenSample = now;
                if (track.Class == ThreadClass.Guest && track.Ema > bestEma)
                {
                    bestEma = track.Ema;
                    bestId = thread.ThreadId;
                }
            }

            _guestMainId = bestId;
            _lastThreadSample = now;
            if (Threads.Count > ThreadScratch.Count * 2 + 64)
            {
                foreach (var id in Threads.Where(pair => pair.Value.SeenSample != now).Select(pair => pair.Key).ToArray())
                {
                    Threads.Remove(id);
                }
            }
        }

        private static ThreadClass Classify(ulong id, string name, ulong renderThreadId, bool remote)
        {
            if (id == renderThreadId) return ThreadClass.Render;
            if (remote)
            {
                // The guest process of the split: its gateway threads and guest runners are guest;
                // the rest is its runtime and emulator housekeeping.
                return Bridge.GuestThreads.Contains(id) ||
                    name.StartsWith("GuestContinuation-", StringComparison.Ordinal) ||
                    (name.StartsWith("SharpEmu-", StringComparison.Ordinal) &&
                        name is not ("SharpEmu-StallWatchdog" or "SharpEmu-ReadyDispatch"))
                    ? ThreadClass.Guest
                    : ThreadClass.Other;
            }

            if (GuestThreadIds.ContainsKey(id) ||
                name.StartsWith("GuestContinuation-", StringComparison.Ordinal) ||
                (name.StartsWith("SharpEmu-", StringComparison.Ordinal) &&
                    name is not ("SharpEmu-StallWatchdog" or "SharpEmu-ReadyDispatch")))
            {
                return ThreadClass.Guest;
            }

            if (name.Contains("compiler", StringComparison.OrdinalIgnoreCase) ||
                name.Contains("shader", StringComparison.OrdinalIgnoreCase))
            {
                return ThreadClass.ShaderHelper;
            }

            return name.StartsWith("SharpEmu", StringComparison.Ordinal) ? ThreadClass.RenderHelper : ThreadClass.Other;
        }

        private static (float Body, float Wait) TakeHleDeltas()
        {
            HleThreadSlots[] threads;
            lock (HleThreads)
            {
                threads = HleThreads.ToArray();
            }

            // Drop what the previous frame accumulated but did not turn into costs.
            foreach (var stale in HleTouched)
            {
                ClearHle(stale);
            }

            HleTouched.Clear();
            float body = 0, wait = 0;
            var frequency = (double)Stopwatch.Frequency;
            if (Bridge.IsOpen)
            {
                // The split: the guest's gateway totals, split by the guest into the main thread
                // this process picked and every other guest thread.
                var ticksPerMs = frequency / 1000.0;
                Bridge.TakeImportDeltas((index, delta) =>
                {
                    var ms = delta.MainMs + delta.WorkerMs;
                    var blockedMs = delta.MainBlockedMs + delta.WorkerBlockedMs;
                    if (IsWaitImport(index))
                    {
                        wait += ms;
                    }
                    else
                    {
                        body += ms - blockedMs;
                        wait += blockedMs;
                    }

                    if (!HleTimingActive) return;
                    EnsureHleCapacity(index);
                    if (_hleMainTicks[index] == 0 && _hleWorkerTicks[index] == 0) HleTouched.Add(index);
                    _hleMainTicks[index] += (long)(delta.MainMs * ticksPerMs);
                    _hleMainBlocked[index] += (long)(delta.MainBlockedMs * ticksPerMs);
                    _hleMainCounts[index] += delta.MainCalls;
                    _hleWorkerTicks[index] += (long)(delta.WorkerMs * ticksPerMs);
                    _hleWorkerBlocked[index] += (long)(delta.WorkerBlockedMs * ticksPerMs);
                    _hleWorkerCounts[index] += delta.WorkerCalls;
                });
            }

            foreach (var slots in threads)
            {
                var arrays = slots.Arrays;
                slots.EnsurePrevious(arrays.Ticks.Length);
                var isMain = slots.ThreadId == _guestMainId;
                for (var index = 0; index < arrays.Ticks.Length; index++)
                {
                    var ticks = arrays.Ticks[index];
                    var deltaTicks = ticks - slots.PreviousTicks[index];
                    if (deltaTicks == 0) continue;
                    var count = arrays.Counts[index];
                    var deltaCount = count - slots.PreviousCounts[index];
                    var blocked = arrays.Blocked[index];
                    var deltaBlocked = blocked - slots.PreviousBlocked[index];
                    slots.PreviousTicks[index] = ticks;
                    slots.PreviousCounts[index] = count;
                    slots.PreviousBlocked[index] = blocked;
                    if (deltaTicks < 0) continue;
                    deltaBlocked = Math.Clamp(deltaBlocked, 0, deltaTicks);
                    var ms = (float)(deltaTicks * 1000.0 / frequency);
                    if (IsWaitImport(index))
                    {
                        wait += ms;
                    }
                    else
                    {
                        var blockedMs = (float)(deltaBlocked * 1000.0 / frequency);
                        body += ms - blockedMs;
                        wait += blockedMs;
                    }

                    if (!HleTimingActive) continue;
                    EnsureHleCapacity(index);
                    if (_hleMainTicks[index] == 0 && _hleWorkerTicks[index] == 0) HleTouched.Add(index);
                    if (isMain)
                    {
                        _hleMainTicks[index] += deltaTicks;
                        _hleMainBlocked[index] += deltaBlocked;
                        _hleMainCounts[index] += deltaCount;
                    }
                    else
                    {
                        _hleWorkerTicks[index] += deltaTicks;
                        _hleWorkerBlocked[index] += deltaBlocked;
                        _hleWorkerCounts[index] += deltaCount;
                    }
                }
            }

            return (body, wait);
        }

        private static void EnsureHleCapacity(int index)
        {
            if (index < _hleMainTicks.Length) return;
            var size = Math.Max(index + 1, _hleMainTicks.Length * 2);
            Array.Resize(ref _hleMainTicks, size);
            Array.Resize(ref _hleMainBlocked, size);
            Array.Resize(ref _hleMainCounts, size);
            Array.Resize(ref _hleWorkerTicks, size);
            Array.Resize(ref _hleWorkerBlocked, size);
            Array.Resize(ref _hleWorkerCounts, size);
        }

        private static void ClearHle(int index)
        {
            _hleMainTicks[index] = 0;
            _hleMainBlocked[index] = 0;
            _hleMainCounts[index] = 0;
            _hleWorkerTicks[index] = 0;
            _hleWorkerBlocked[index] = 0;
            _hleWorkerCounts[index] = 0;
        }

        /// <summary>
        /// One import's cost rows on one lane: a name-classified wait is a wait whole; otherwise the
        /// part blocked in host pacing is a wait row and the rest the HLE body row, under the same name.
        /// </summary>
        private static void AddHleCost(int nameId, bool waitImport, OverlayLane lane, long ticks, long blocked, int count,
            double frequency)
        {
            if (ticks <= 0)
            {
                return;
            }

            if (waitImport)
            {
                CostScratch.Add(new CostSample(nameId, CostKind.Wait, lane, (float)(ticks * 1000.0 / frequency), count));
                return;
            }

            blocked = Math.Clamp(blocked, 0, ticks);
            if (ticks > blocked)
            {
                CostScratch.Add(new CostSample(nameId, CostKind.Hle, lane, (float)((ticks - blocked) * 1000.0 / frequency), count));
            }

            if (blocked > 0)
            {
                CostScratch.Add(new CostSample(nameId, CostKind.Wait, lane, (float)(blocked * 1000.0 / frequency), count));
            }
        }

        private static void AddHleCosts()
        {
            var frequency = (double)Stopwatch.Frequency;
            var first = CostScratch.Count;
            foreach (var index in HleTouched)
            {
                var nameId = HleNameBase + index;
                var waitImport = IsWaitImport(index);
                AddHleCost(nameId, waitImport, OverlayLane.GuestMain, _hleMainTicks[index], _hleMainBlocked[index],
                    _hleMainCounts[index], frequency);
                AddHleCost(nameId, waitImport, OverlayLane.GuestWorkers, _hleWorkerTicks[index], _hleWorkerBlocked[index],
                    _hleWorkerCounts[index], frequency);
                ClearHle(index);
            }

            var added = CostScratch.Count - first;
            if (added > MaxHleCostsPerFrame)
            {
                CostScratch.Sort(first, added, Comparer<CostSample>.Create(static (left, right) => right.Milliseconds.CompareTo(left.Milliseconds)));
                CostScratch.RemoveRange(first + MaxHleCostsPerFrame, added - MaxHleCostsPerFrame);
            }
        }

        private static void AddGuestSegments(long start, long end, float frameMs, float guestMain, float guestWorkers,
            int activeWorkers, Func<long, float> relative)
        {
            var mainPlaced = false;
            var workerPlaced = false;
            if (TimelineActive)
            {
                if (Bridge.IsOpen)
                {
                    Bridge.TakeCalls(RemoteCalls);
                    mainPlaced = PlaceRemoteCalls(OverlayLane.GuestMain, start, end, frameMs, relative);
                    workerPlaced = PlaceRemoteCalls(OverlayLane.GuestWorkers, start, end, frameMs, relative);
                    // Calls that ended before this frame can no longer be placed; keep the ring of
                    // pending ones bounded if frames stop coming.
                    RemoteCalls.RemoveAll(call => call.End <= start);
                    if (RemoteCalls.Count > 8192)
                    {
                        RemoteCalls.RemoveRange(0, RemoteCalls.Count - 8192);
                    }
                }
                else
                {
                    mainPlaced = PlaceLocalCalls(_guestMainId, OverlayLane.GuestMain, start, end, frameMs, relative);
                    workerPlaced = PlaceLocalCalls(_busiestWorkerId, OverlayLane.GuestWorkers, start, end, frameMs, relative);
                }
            }

            if (!mainPlaced && guestMain > 0)
            {
                SegmentScratch.Add(new TimelineSegment(0, Math.Min(guestMain, frameMs), OverlayLane.GuestMain, CostKind.Guest, GuestCpuNameId));
            }

            if (!workerPlaced && guestWorkers > 0 && activeWorkers > 0)
            {
                SegmentScratch.Add(new TimelineSegment(0, Math.Min(guestWorkers / activeWorkers, frameMs),
                    OverlayLane.GuestWorkers, CostKind.Guest, WorkerCpuNameId));
            }
        }

        private static bool PlaceLocalCalls(ulong threadId, OverlayLane lane, long start, long end, float frameMs,
            Func<long, float> relative)
        {
            if (threadId == 0)
            {
                return false;
            }

            HleThreadSlots? thread = null;
            lock (HleThreads)
            {
                foreach (var slots in HleThreads)
                {
                    if (slots.ThreadId == threadId) { thread = slots; break; }
                }
            }

            if (thread is null)
            {
                return false;
            }

            LongCallScratch.Clear();
            thread.TakeLongCalls(start, end, LongCallScratch);
            return PlaceCalls(LongCallScratch, lane, frameMs, relative);
        }

        private static bool PlaceRemoteCalls(OverlayLane lane, long start, long end, float frameMs, Func<long, float> relative)
        {
            LongCallScratch.Clear();
            foreach (var call in RemoteCalls)
            {
                if (call.Lane == lane && call.End > start && call.Start < end)
                {
                    LongCallScratch.Add((call.Start, call.End, call.Import));
                }
            }

            LongCallScratch.Sort(static (left, right) => left.Start.CompareTo(right.Start));
            return PlaceCalls(LongCallScratch, lane, frameMs, relative);
        }

        /// <summary>
        /// Lays one thread's long HLE calls out in time and fills the gaps between them with guest
        /// code (a guest thread between two gateway calls is running guest code).
        /// </summary>
        private static bool PlaceCalls(List<(long Start, long End, int Import)> calls, OverlayLane lane, float frameMs,
            Func<long, float> relative)
        {
            if (calls.Count == 0)
            {
                return false;
            }

            var cursor = 0f;
            foreach (var (callStart, callEnd, flaggedImport) in calls)
            {
                var mostlyBlocked = (flaggedImport & MostlyBlockedCall) != 0;
                var import = flaggedImport & ~MostlyBlockedCall;
                var from = relative(callStart);
                var to = relative(callEnd);
                if (from > cursor)
                {
                    SegmentScratch.Add(new TimelineSegment(cursor, from, lane, CostKind.Guest, GuestRunNameId));
                }

                if (to > from)
                {
                    SegmentScratch.Add(new TimelineSegment(from, to, lane,
                        mostlyBlocked || IsWaitImport(import) ? CostKind.Wait : CostKind.Hle, HleNameBase + import));
                }

                cursor = Math.Max(cursor, to);
            }

            if (cursor < frameMs)
            {
                SegmentScratch.Add(new TimelineSegment(cursor, frameMs, lane, CostKind.Guest, GuestRunNameId));
            }

            return true;
        }

        private static (bool Hitch, string? Reason) DetectHitch(float frameMs, float gcPauseMs, int gcGeneration,
            float shaderMs, float gpuMs, float guestMain, float renderBusy)
        {
            var median = Median();
            RecentFrameMs[_recentCount % RecentFrameMs.Length] = frameMs;
            _recentCount++;
            if (median <= 0 || frameMs < 20 || frameMs < Math.Max(median * 1.6f, median + 6))
            {
                return (false, null);
            }

            var excess = frameMs - median;
            if (gcPauseMs >= excess * 0.4f) return (true, gcGeneration >= 0 ? $"GC gen{gcGeneration}" : "GC");
            if (shaderMs >= excess * 0.4f) return (true, "Shader compile");
            if (gpuMs >= frameMs * 0.9f) return (true, "GPU");
            if (guestMain >= frameMs * 0.8f) return (true, "Guest CPU");
            if (renderBusy >= frameMs * 0.8f) return (true, "Render thread");
            return (true, "Sync / wait");
        }

        private static float Median()
        {
            var count = Math.Min(_recentCount, RecentFrameMs.Length);
            if (count < 8) return 0;
            Span<float> sorted = stackalloc float[count];
            RecentFrameMs.AsSpan(0, count).CopyTo(sorted);
            sorted.Sort();
            return sorted[count / 2];
        }
    }

    // =====================================================================================
    // Storage types
    // =====================================================================================

    private sealed class RawFrame
    {
        public long EndTimestamp;
        public ulong ThreadId;
        public bool HasPhases;
        public readonly long[] PhaseTicks = new long[(int)RenderPhaseProfile.Phase.Count];
        public int SegmentCount;
        public long[] SegmentStart = new long[256];
        public long[] SegmentEnd = new long[256];
        public byte[] SegmentKind = new byte[256];
        public int[] SegmentPhase = new int[256];

        public void CopySegments(RenderSegmentBuffer source)
        {
            var count = source.Count;
            if (SegmentStart.Length < count)
            {
                var size = Math.Max(count, SegmentStart.Length * 2);
                SegmentStart = new long[size];
                SegmentEnd = new long[size];
                SegmentKind = new byte[size];
                SegmentPhase = new int[size];
            }

            Array.Copy(source.Start, SegmentStart, count);
            Array.Copy(source.End, SegmentEnd, count);
            Array.Copy(source.Kind, SegmentKind, count);
            Array.Copy(source.Phase, SegmentPhase, count);
            SegmentCount = count;
        }
    }

    private sealed class HleThreadSlots(ulong threadId)
    {
        internal sealed class Storage(int size)
        {
            public readonly long[] Ticks = new long[size];
            public readonly int[] Counts = new int[size];
            // The part of Ticks the thread spent blocked in host pacing.
            public readonly long[] Blocked = new long[size];
        }

        public readonly ulong ThreadId = threadId;
        public Storage Arrays = new(1024);
        // Builder-owned.
        public long[] PreviousTicks = [];
        public int[] PreviousCounts = [];
        public long[] PreviousBlocked = [];

        private readonly object _longCallGate = new();
        private readonly (long Start, long End, int Import)[] _longCalls = new (long, long, int)[256];
        private int _longCallCount;

        public Storage Grow(int index)
        {
            var current = Arrays;
            var next = new Storage(Math.Max(index + 1, current.Ticks.Length * 2));
            Array.Copy(current.Ticks, next.Ticks, current.Ticks.Length);
            Array.Copy(current.Counts, next.Counts, current.Counts.Length);
            Array.Copy(current.Blocked, next.Blocked, current.Blocked.Length);
            Volatile.Write(ref Arrays, next);
            return next;
        }

        public void EnsurePrevious(int length)
        {
            if (PreviousTicks.Length >= length) return;
            Array.Resize(ref PreviousTicks, length);
            Array.Resize(ref PreviousCounts, length);
            Array.Resize(ref PreviousBlocked, length);
        }

        public void AppendLongCall(int import, long start, long end)
        {
            lock (_longCallGate)
            {
                if (_longCallCount < _longCalls.Length)
                {
                    _longCalls[_longCallCount++] = (start, end, import);
                }
            }
        }

        public void TakeLongCalls(long frameStart, long frameEnd, List<(long Start, long End, int Import)> into)
        {
            lock (_longCallGate)
            {
                var keep = 0;
                for (var index = 0; index < _longCallCount; index++)
                {
                    var call = _longCalls[index];
                    if (call.End > frameEnd)
                    {
                        _longCalls[keep++] = call;
                    }

                    if (call.End > frameStart && call.Start < frameEnd)
                    {
                        into.Add(call);
                    }
                }

                _longCallCount = keep;
            }

            into.Sort(static (left, right) => left.Start.CompareTo(right.Start));
        }
    }
}

/// <summary>
/// The render thread's timeline for the current frame: consecutive phase changes coalesced by
/// <see cref="CostKind"/>. Owned by one thread; handed to the telemetry at the frame's end.
/// </summary>
internal sealed class RenderSegmentBuffer
{
    private const int Capacity = 2048;
    public readonly long[] Start = new long[Capacity];
    public readonly long[] End = new long[Capacity];
    public readonly byte[] Kind = new byte[Capacity];
    public readonly int[] Phase = new int[Capacity];
    private readonly long[] _longest = new long[Capacity];
    public int Count;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Append(int phase, long start, long end)
    {
        if (end <= start) return;
        var kind = (byte)OverlayTelemetry.PhaseKinds[phase];
        var last = Count - 1;
        if (last >= 0 && (Kind[last] == kind || Count == Capacity) && End[last] >= start - 1)
        {
            End[last] = end;
            if (end - start > _longest[last])
            {
                _longest[last] = end - start;
                Phase[last] = phase;
            }

            return;
        }

        if (Count == Capacity) return;
        Start[Count] = start;
        End[Count] = end;
        Kind[Count] = kind;
        Phase[Count] = phase;
        _longest[Count] = end - start;
        Count++;
    }

    public void Clear() => Count = 0;
}
