// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Diagnostics.Tracing;
using System.Globalization;

namespace SharpEmu.Logging;

/// <summary>
/// SHARPEMU_GC_PAUSES=1: every runtime garbage collection that paused this process for 10 ms or more
/// is logged as <c>[GC-PAUSE] pid=.. unix_us=.. gen=.. pause_ms=..</c>, on the wall clock the frame-time
/// log (SHARPEMU_FRAME_TIMES) names its origin in, so a long frame can be matched with the pause in it.
/// The last collection is polled every 50 ms; one that two polls do not separate is missed.
/// SHARPEMU_GC_PAUSES=2 does the same and additionally prints one <c>[GC-DETAIL]</c> line per collection
/// (see <see cref="GcDetailListener"/>); the =1 output and cost are unchanged.
/// </summary>
public static class GcPauseLog
{
    private const double ThresholdMilliseconds = 10;
    private static int _started;
    private static GcDetailListener? _detail;

    public static void StartFromEnvironment()
    {
        var mode = Environment.GetEnvironmentVariable("SHARPEMU_GC_PAUSES");
        if ((mode != "1" && mode != "2") || Interlocked.Exchange(ref _started, 1) != 0)
        {
            return;
        }

        if (mode == "2")
        {
            _detail = new GcDetailListener();
        }

        var thread = new Thread(Poll) { IsBackground = true, Name = "SharpEmu GC pause log" };
        thread.Start();
    }

    private static void Poll()
    {
        var pid = Environment.ProcessId;
        long lastIndex = -1;
        while (true)
        {
            Thread.Sleep(50);
            var info = GC.GetGCMemoryInfo(GCKind.Any);
            if (info.Index == lastIndex || info.Index == 0)
            {
                continue;
            }

            lastIndex = info.Index;
            var pause = 0.0;
            foreach (var duration in info.PauseDurations)
            {
                pause += duration.TotalMilliseconds;
            }

            if (pause >= ThresholdMilliseconds)
            {
                var unixMicroseconds = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1000;
                Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"[GC-PAUSE] pid={pid} unix_us={unixMicroseconds} gen={info.Generation} pause_ms={pause:F1} " +
                    $"heap_mb={info.HeapSizeBytes / (1024 * 1024)} index={info.Index}"));
            }
        }
    }
}

/// <summary>
/// One <c>[GC-DETAIL] pid=.. index=..</c> line per collection (only when suspend + GC time reaches the same
/// 10 ms threshold as [GC-PAUSE]; printed at GCRestartEEBegin because GCHeapStats follows GCEnd), from the runtime's GC events at Informational level (keyword 0x1; no
/// GCAllocationTick, so no Verbose cost). Payload fields are read by name. Fields (absent when no event gave them):
/// reason, type (GCStart_V2); suspend_ms, suspend_reason (GCSuspendEEBegin/End); gc_ms (GCStart to GCEnd);
/// mark_ms (first to last GCMarkWithType) and mark_{stack,fq,handles,older,sizedref,overflow,dependent}_kb
/// (promoted bytes summed over heaps); post_mark_ms (last mark event to GCEnd, i.e. plan/relocate/compact/sweep,
/// the runtime has no per-phase events); gen0/1/2/loh/poh_mb and promoted0/1/2_kb, pinned, handles (GCHeapStats_V2);
/// heaps, condemned, gen0_reduction, global_reason, mechanisms (hex bitmask), pause_mode, memory_load (percent),
/// desired_kb (FinalYoungestDesired, the gen0 budget) (GCGlobalHeapHistory_V*).
/// Events carry no shared index except GCStart/GCEnd/GCGlobalHeapHistory (Count), so one in-flight record
/// is assumed: a background GC with foreground GCs nested inside it is reported partially (partial=1).
/// </summary>
internal sealed class GcDetailListener : EventListener
{
    private const double ThresholdMilliseconds = 10;
    private static readonly string[] Reasons =
    {
        "AllocSmall", "Induced", "LowMemory", "Empty", "AllocLarge", "OutOfSpaceSOH", "OutOfSpaceLOH",
        "InducedNotForced", "Internal", "InducedLowMemory", "InducedCompacting", "LowMemoryHost", "PMFullGC",
        "LowMemoryHostBlocking",
    };
    private static readonly string[] Types = { "NonConcurrent", "Background", "Foreground" };
    private static readonly string[] SuspendReasons =
    {
        "Other", "GC", "AppDomainShutdown", "CodePitching", "Shutdown", "Debugger", "GCPrep", "DebuggerSweep",
    };
    private static readonly string[] PauseModes = { "Batch", "Interactive", "LowLatency", "SustainedLowLatency", "NoGC" };
    private static readonly string[] MarkNames =
    {
        "stack", "fq", "handles", "older", "sizedref", "overflow", "dependent",
    };

    private readonly object _lock = new();
    private readonly int _pid = Environment.ProcessId;
    private readonly long[] _markBytes = new long[MarkNames.Length];
    private int _errors;
    private int _errorsPrinted;

    // In-flight record. Long.MinValue / MinValue mean "not seen".
    private DateTime _end, _suspendBegin, _suspendEnd, _gcStart, _firstMark, _lastMark;
    private long _suspendReason = long.MinValue, _count = long.MinValue, _reason = long.MinValue, _type = long.MinValue;
    private int _markEvents;
    private bool _open, _partial;
    private readonly long[] _heap = new long[16];
    private bool _haveHeap;
    private readonly long[] _global = new long[8];
    private bool _haveGlobal;

    protected override void OnEventSourceCreated(EventSource eventSource)
    {
        if (eventSource.Name == "Microsoft-Windows-DotNETRuntime")
        {
            EnableEvents(eventSource, EventLevel.Informational, (EventKeywords)0x1);
        }
    }

    protected override void OnEventWritten(EventWrittenEventArgs data)
    {
        var name = data.EventName;
        if (name is null || !name.StartsWith("GC", StringComparison.Ordinal))
        {
            return;
        }

        try
        {
            lock (_lock)
            {
                Handle(name, data);
            }
        }
        catch
        {
            Interlocked.Increment(ref _errors);
        }
    }

    private void Handle(string name, EventWrittenEventArgs e)
    {
        if (name.StartsWith("GCSuspendEEBegin", StringComparison.Ordinal))
        {
            FlushEnded();
            if (_open && _gcStart != default)
            {
                // Suspension inside an in-flight (background) GC: keep that record.
                return;
            }

            Reset();
            _open = true;
            _suspendBegin = e.TimeStamp;
            _suspendReason = Get(e, "Reason");
        }
        else if (name.StartsWith("GCSuspendEEEnd", StringComparison.Ordinal))
        {
            _suspendEnd = e.TimeStamp;
        }
        else if (name.StartsWith("GCStart", StringComparison.Ordinal))
        {
            FlushEnded();
            if (_open && _gcStart != default)
            {
                _partial = true;
                Emit(default);
                Reset();
                _open = true;
            }

            _open = true;
            _gcStart = e.TimeStamp;
            _count = Get(e, "Count");
            _reason = Get(e, "Reason");
            _type = Get(e, "Type");
        }
        else if (name == "GCMarkWithType")
        {
            if (!_open)
            {
                return;
            }

            var type = Get(e, "Type");
            var bytes = Get(e, "Bytes");
            if (type >= 0 && type < _markBytes.Length && bytes > 0)
            {
                _markBytes[type] += bytes;
            }

            if (_markEvents++ == 0)
            {
                _firstMark = e.TimeStamp;
            }

            _lastMark = e.TimeStamp;
        }
        else if (name.StartsWith("GCHeapStats", StringComparison.Ordinal))
        {
            if (!_open)
            {
                return;
            }

            string[] fields =
            {
                "GenerationSize0", "TotalPromotedSize0", "GenerationSize1", "TotalPromotedSize1", "GenerationSize2",
                "TotalPromotedSize2", "GenerationSize3", "TotalPromotedSize3", "GenerationSize4", "PinnedObjectCount",
                "GCHandleCount", "FinalizationPromotedCount",
            };
            for (var i = 0; i < fields.Length; i++)
            {
                _heap[i] = Get(e, fields[i]);
            }

            _haveHeap = true;
        }
        else if (name.StartsWith("GCGlobalHeapHistory", StringComparison.Ordinal))
        {
            if (!_open)
            {
                return;
            }

            _global[0] = Get(e, "NumHeaps");
            _global[1] = Get(e, "CondemnedGeneration");
            _global[2] = Get(e, "Gen0ReductionCount");
            _global[3] = Get(e, "Reason");
            _global[4] = Get(e, "GlobalMechanisms");
            _global[5] = Get(e, "PauseMode");
            _global[6] = Get(e, "MemoryPressure");
            _global[7] = Get(e, "FinalYoungestDesired");
            _haveGlobal = true;
        }
        else if (name.StartsWith("GCEnd", StringComparison.Ordinal))
        {
            // GCHeapStats follows GCEnd; the record is printed at GCRestartEEBegin (or by the next GC).
            if (_open && _gcStart != default)
            {
                _end = e.TimeStamp;
            }
        }
        else if (name.StartsWith("GCRestartEEBegin", StringComparison.Ordinal))
        {
            FlushEnded();
        }
    }

    private void FlushEnded()
    {
        if (_open && _end != default)
        {
            Emit(_end);
            Reset();
        }
    }

    private void Reset()
    {
        _end = default;
        _open = false;
        _partial = false;
        _suspendBegin = _suspendEnd = _gcStart = _firstMark = _lastMark = default;
        _suspendReason = _count = _reason = _type = long.MinValue;
        _markEvents = 0;
        Array.Clear(_markBytes);
        _haveHeap = _haveGlobal = false;
    }

    private void Emit(DateTime end)
    {
        double suspendMs = _suspendBegin != default && _suspendEnd >= _suspendBegin
            ? (_suspendEnd - _suspendBegin).TotalMilliseconds : -1;
        double gcMs = end != default ? (end - _gcStart).TotalMilliseconds : -1;
        if (Math.Max(suspendMs, 0) + Math.Max(gcMs, 0) < ThresholdMilliseconds)
        {
            return;
        }

        var sb = new System.Text.StringBuilder(384);
        var ic = CultureInfo.InvariantCulture;
        if (_errors != _errorsPrinted)
        {
            _errorsPrinted = _errors;
            sb.Append("[GC-DETAIL] errors=").Append(_errorsPrinted.ToString(ic)).Append('\n');
        }

        sb.Append("[GC-DETAIL] pid=").Append(_pid).Append(" index=").Append(_count.ToString(ic));
        Name(sb, "reason", _reason, Reasons);
        Name(sb, "type", _type, Types);
        if (suspendMs >= 0)
        {
            sb.Append(ic, $" suspend_ms={suspendMs:F1}");
        }

        Name(sb, "suspend_reason", _suspendReason, SuspendReasons);
        if (gcMs >= 0)
        {
            sb.Append(ic, $" gc_ms={gcMs:F1}");
        }

        if (_markEvents > 0)
        {
            sb.Append(ic, $" mark_ms={(_lastMark - _firstMark).TotalMilliseconds:F1}");
            if (end != default)
            {
                sb.Append(ic, $" post_mark_ms={(end - _lastMark).TotalMilliseconds:F1}");
            }

            for (var i = 0; i < MarkNames.Length; i++)
            {
                sb.Append(ic, $" mark_{MarkNames[i]}_kb={_markBytes[i] / 1024}");
            }
        }

        if (_haveHeap)
        {
            Kv(sb, "gen0_mb", _heap[0], 20);
            Kv(sb, "gen1_mb", _heap[2], 20);
            Kv(sb, "gen2_mb", _heap[4], 20);
            Kv(sb, "loh_mb", _heap[6], 20);
            Kv(sb, "poh_mb", _heap[8], 20);
            Kv(sb, "promoted0_kb", _heap[1], 10);
            Kv(sb, "promoted1_kb", _heap[3], 10);
            Kv(sb, "promoted2_kb", _heap[5], 10);
            Kv(sb, "pinned", _heap[9], 0);
            Kv(sb, "handles", _heap[10], 0);
        }

        if (_haveGlobal)
        {
            Kv(sb, "heaps", _global[0], 0);
            Kv(sb, "condemned", _global[1], 0);
            Kv(sb, "gen0_reduction", _global[2], 0);
            Name(sb, "global_reason", _global[3], Reasons);
            if (_global[4] != long.MinValue)
            {
                sb.Append(" mechanisms=0x").Append(_global[4].ToString("x", ic));
            }

            Name(sb, "pause_mode", _global[5], PauseModes);
            Kv(sb, "memory_load", _global[6], 0);
            Kv(sb, "desired_kb", _global[7], 10);
        }

        if (_partial)
        {
            sb.Append(" partial=1");
        }

        Console.Error.WriteLine(sb.ToString());
    }

    private static void Kv(System.Text.StringBuilder sb, string key, long value, int shift)
    {
        if (value != long.MinValue)
        {
            sb.Append(' ').Append(key).Append('=').Append((value >> shift).ToString(CultureInfo.InvariantCulture));
        }
    }

    private static void Name(System.Text.StringBuilder sb, string key, long value, string[] names)
    {
        if (value == long.MinValue)
        {
            return;
        }

        sb.Append(' ').Append(key).Append('=');
        if (value >= 0 && value < names.Length)
        {
            sb.Append(names[value]);
        }
        else
        {
            sb.Append(value);
        }
    }

    /// <summary>Reads an integer payload field by name; long.MinValue when the field is absent.</summary>
    private static long Get(EventWrittenEventArgs e, string field)
    {
        var names = e.PayloadNames;
        var payload = e.Payload;
        if (names is null || payload is null)
        {
            return long.MinValue;
        }

        for (var i = 0; i < names.Count && i < payload.Count; i++)
        {
            if (names[i] == field)
            {
                return payload[i] switch
                {
                    ulong u => (long)u,
                    long l => l,
                    uint u => u,
                    int n => n,
                    ushort u => u,
                    byte b => b,
                    _ => long.MinValue,
                };
            }
        }

        return long.MinValue;
    }
}
