// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.ExceptionServices;

namespace SharpEmu.Libs.Gpu.GpuCommands;

// One producer, one ordered consumer. Slots belong to A until Publish, then to B
// until Release. The two monotone indices are the release/acquire publication fence.
// The monitor is only for block empty/full transitions and acknowledgements, never
// for adding an item to a producer-owned block. Failure wakes both sides.
internal sealed class CommandIssueRing<T> where T : class
{
    internal sealed class Block
    {
        internal readonly T[] Items;
        internal int Count;
        internal long LastSequence;
        internal Block(int count, Func<T> create) => Items = Enumerable.Range(0, count).Select(_ => create()).ToArray();
    }

    private readonly Block[] _blocks;
    private readonly object _gate = new();
    private long _published;
    private long _released;
    private long _acknowledged;
    private bool _stopping;
    private ExceptionDispatchInfo? _failure;
    internal long FullWaitTicks;
    internal long EmptyWaitTicks;
    internal long AcknowledgementWaitTicks;
    internal long FullStalls;
    internal long SyncStalls;
    internal long Wakeups;
    internal long PublishedBlocks;
    internal long PublishedItems;

    internal CommandIssueRing(int blocks, int items, Func<T> create)
    {
        if (blocks < 2 || items < 1) throw new ArgumentOutOfRangeException(nameof(blocks));
        _blocks = Enumerable.Range(0, blocks).Select(_ => new Block(items, create)).ToArray();
    }

    internal Block Acquire()
    {
        ThrowIfFailed();
        if (_published - Volatile.Read(ref _released) >= _blocks.Length)
        {
            var started = Stopwatch.GetTimestamp();
            using var wait = VideoOut.RenderPhaseProfile.Measure(VideoOut.RenderPhaseProfile.Phase.WorkerAWaitsForB);
            Interlocked.Increment(ref FullStalls);
            lock (_gate)
            {
                while (_published - Volatile.Read(ref _released) >= _blocks.Length && !_stopping && _failure is null)
                    Monitor.Wait(_gate);
            }
            Interlocked.Add(ref FullWaitTicks, Stopwatch.GetTimestamp() - started);
            ThrowIfFailed();
        }
        if (_stopping) throw new OperationCanceledException("The issue ring is closed.");
        return _blocks[(int)(_published % _blocks.Length)];
    }

    internal void Publish(Block block)
    {
        ThrowIfFailed();
        if (block.Count == 0) return;
        lock (_gate)
        {
            var empty = _published == _released;
            Volatile.Write(ref _published, _published + 1);
            Interlocked.Increment(ref PublishedBlocks);
            Interlocked.Add(ref PublishedItems, block.Count);
            if (empty)
            {
                Interlocked.Increment(ref Wakeups);
                Monitor.PulseAll(_gate);
            }
        }
    }

    internal Block? Take(Action? idle = null)
    {
        for (;;)
        {
            ThrowIfFailed();
            if (Volatile.Read(ref _published) != _released)
                return _blocks[(int)(_released % _blocks.Length)];
            idle?.Invoke();
            var started = Stopwatch.GetTimestamp();
            lock (_gate)
            {
                if (_published == _released && !_stopping && _failure is null) Monitor.Wait(_gate);
                if (_published == _released && _stopping) return null;
            }
            Interlocked.Add(ref EmptyWaitTicks, Stopwatch.GetTimestamp() - started);
        }
    }

    internal void Release(Block block)
    {
        lock (_gate)
        {
            var full = _published - _released == _blocks.Length;
            Volatile.Write(ref _acknowledged, block.LastSequence);
            block.Count = 0;
            Volatile.Write(ref _released, _released + 1);
            // A may be waiting for a prefix even when the ring was not full.
            if (full || _syncWaiting) Monitor.PulseAll(_gate);
        }
    }

    private bool _syncWaiting;
    internal void WaitFor(long sequence)
    {
        ThrowIfFailed();
        if (Volatile.Read(ref _acknowledged) >= sequence) return;
        var started = Stopwatch.GetTimestamp();
        Interlocked.Increment(ref SyncStalls);
        lock (_gate)
        {
            _syncWaiting = true;
            try
            {
                while (_acknowledged < sequence && _failure is null && !_stopping) Monitor.Wait(_gate);
                ThrowIfFailed();
                if (_acknowledged < sequence) throw new OperationCanceledException("The issue prefix was not acknowledged.");
            }
            finally { _syncWaiting = false; }
        }
        Interlocked.Add(ref AcknowledgementWaitTicks, Stopwatch.GetTimestamp() - started);
    }

    internal void Wake() { lock (_gate) Monitor.PulseAll(_gate); }
    internal void Stop() { lock (_gate) { _stopping = true; Monitor.PulseAll(_gate); } }
    internal void Fail(Exception error)
    {
        lock (_gate) { _failure ??= ExceptionDispatchInfo.Capture(error); Monitor.PulseAll(_gate); }
    }
    internal void ThrowIfFailed() => Volatile.Read(ref _failure)?.Throw();
}
