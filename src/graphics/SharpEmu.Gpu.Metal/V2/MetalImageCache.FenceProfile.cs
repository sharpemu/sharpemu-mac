// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalImageCache
{
    private static readonly bool FenceHoldProfile =
        Environment.GetEnvironmentVariable("SHARPEMU_PROFILE_FENCE_HOLDS") == "1";
    private readonly FenceHoldRecord[]? _fenceHoldRows = FenceHoldProfile ? new FenceHoldRecord[2048] : null;
    private long _fenceHoldSequence;
    private long _fenceHoldReported;
    private long _fenceHoldCalls;
    private long _fenceHoldTicks;
    private long _fenceHoldMaxTicks;

    private struct FenceHoldRecord
    {
        public long Sequence;
        public long Started;
        public long Ended;
        public int ManagedThread;
        public string? ThreadName;
        public string? Operation;
    }

    private FenceHoldScope MeasureFenceHold([CallerMemberName] string operation = "") =>
        FenceHoldProfile ? new(this, operation) : default;

    private readonly struct FenceHoldScope : IDisposable
    {
        private readonly MetalImageCache? _cache;
        private readonly string? _operation;
        private readonly long _started;

        public FenceHoldScope(MetalImageCache cache, string operation)
        {
            _cache = cache;
            _operation = operation;
            _started = Stopwatch.GetTimestamp();
        }

        public void Dispose()
        {
            if (_cache is not { } cache) return;
            var end = Stopwatch.GetTimestamp();
            var ticks = end - _started;
            // Called before releasing _fenceGate: one writer, with no allocation or logging
            // on a guest fault. Only the ordinary reporting thread formats these records.
            cache._fenceHoldCalls++;
            cache._fenceHoldTicks += ticks;
            cache._fenceHoldMaxTicks = Math.Max(cache._fenceHoldMaxTicks, ticks);
            if (ticks * 1_000_000.0 / Stopwatch.Frequency < 50) return;
            var sequence = cache._fenceHoldSequence + 1;
            ref var row = ref cache._fenceHoldRows![(int)(sequence & 2047)];
            Volatile.Write(ref row.Sequence, 0);
            row.Started = _started;
            row.Ended = end;
            row.ManagedThread = Environment.CurrentManagedThreadId;
            row.ThreadName = Thread.CurrentThread.Name;
            row.Operation = _operation;
            Volatile.Write(ref row.Sequence, sequence);
            Volatile.Write(ref cache._fenceHoldSequence, sequence);
        }
    }

    private string TakeFenceHoldReport()
    {
        if (_fenceHoldRows is not { } rows) return string.Empty;
        var last = Volatile.Read(ref _fenceHoldSequence);
        var first = Math.Max(_fenceHoldReported + 1, last - rows.Length + 1);
        var output = new StringBuilder(FormattableString.Invariant($"\n[METAL-V2][FENCE-HOLD-SUM] cumulative=1 calls={Interlocked.Read(ref _fenceHoldCalls)} hold_ms={Interlocked.Read(ref _fenceHoldTicks) * 1000.0 / Stopwatch.Frequency:F3} max_ms={Interlocked.Read(ref _fenceHoldMaxTicks) * 1000.0 / Stopwatch.Frequency:F3} overwritten={Math.Max(0, first - _fenceHoldReported - 1)} frequency={Stopwatch.Frequency}"));
        for (var sequence = first; sequence <= last; sequence++)
        {
            ref var slot = ref rows[(int)(sequence & 2047)];
            if (Volatile.Read(ref slot.Sequence) != sequence) continue;
            var row = slot;
            if (Volatile.Read(ref slot.Sequence) != sequence) continue;
            output.Append(FormattableString.Invariant($"\n[METAL-V2][FENCE-HOLD] operation={row.Operation} managed={row.ManagedThread} thread=[{row.ThreadName}] hold_ms={(row.Ended - row.Started) * 1000.0 / Stopwatch.Frequency:F3} start_tick={row.Started} end_tick={row.Ended}"));
        }
        _fenceHoldReported = last;
        return output.ToString();
    }
}
