// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.ExceptionServices;

namespace SharpEmu.Libs.Gpu.GpuCommands;

// One producer and one consumer. Indices publish ownership with release/acquire
// ordering. Events are only parking aids: publish/release never take a shared lock.
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
    private readonly ManualResetEventSlim _available = new(false, 0);
    private readonly ManualResetEventSlim _progress = new(false, 0);
    private long _published;
    private long _released;
    private long _acknowledged;
    private volatile bool _stopping;
    private int _consumerWaiting;
    private int _producerWaiting;
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

    internal long Acknowledged => Volatile.Read(ref _acknowledged);

    internal Block Acquire()
    {
        ThrowIfFailed();
        if (_published - Volatile.Read(ref _released) >= _blocks.Length)
        {
            var started = Stopwatch.GetTimestamp();
            using var wait = VideoOut.RenderPhaseProfile.Measure(VideoOut.RenderPhaseProfile.Phase.WorkerAWaitsForB);
            Interlocked.Increment(ref FullStalls);
            _waitingForSpace = true;
            try
            {
                while (_published - Volatile.Read(ref _released) >= _blocks.Length && !_stopping)
                {
                    ParkProducer();
                    ThrowIfFailed();
                }
            }
            finally { _waitingForSpace = false; }
            Interlocked.Add(ref FullWaitTicks, Stopwatch.GetTimestamp() - started);
        }
        if (_stopping) throw new OperationCanceledException("The issue ring is closed.");
        return _blocks[(int)(_published % _blocks.Length)];
    }

    internal void Publish(Block block)
    {
        ThrowIfFailed();
        if (block.Count == 0) return;
        // Only A changes _published and the diagnostics below.
        PublishedBlocks++;
        PublishedItems += block.Count;
        Volatile.Write(ref _published, _published + 1);
        if (Interlocked.CompareExchange(ref _consumerWaiting, 0, 1) == 1)
        {
            Interlocked.Increment(ref Wakeups);
            _available.Set();
        }
    }

    internal Block? Take(Action? idle = null)
    {
        for (;;)
        {
            ThrowIfFailed();
            if (Volatile.Read(ref _published) != _released)
                return _blocks[(int)(_released % _blocks.Length)];
            if (_stopping) return null;
            idle?.Invoke();
            var started = Stopwatch.GetTimestamp();
            _available.Reset();
            Interlocked.Exchange(ref _consumerWaiting, 1);
            try
            {
                // Reset, announce, recheck: a racing publisher either sees the
                // waiter or its release index is seen here. No lost empty wake.
                if (Volatile.Read(ref _published) == _released && !_stopping && Volatile.Read(ref _failure) is null)
                    _available.Wait();
            }
            finally { Volatile.Write(ref _consumerWaiting, 0); }
            Interlocked.Add(ref EmptyWaitTicks, Stopwatch.GetTimestamp() - started);
        }
    }

    internal void Release(Block block)
    {
        var sequence = block.LastSequence;
        block.Count = 0;
        Volatile.Write(ref _acknowledged, sequence);
        Volatile.Write(ref _released, _released + 1);
        if (Interlocked.CompareExchange(ref _producerWaiting, 0, 1) == 1) _progress.Set();
    }

    private long _waitSequence;
    private bool _waitingForSpace;

    // Called only by A. The index/acknowledgement recheck closes the reset/set race.
    private void ParkProducer()
    {
        _progress.Reset();
        Interlocked.Exchange(ref _producerWaiting, 1);
        try
        {
            var pending = _waitingForSpace
                ? _published - Volatile.Read(ref _released) >= _blocks.Length
                : Acknowledged < _waitSequence;
            if (pending && !_stopping && Volatile.Read(ref _failure) is null) _progress.Wait();
        }
        finally { Volatile.Write(ref _producerWaiting, 0); }
    }

    internal bool WaitFor(long sequence)
    {
        ThrowIfFailed();
        if (Acknowledged >= sequence) return false;
        var started = Stopwatch.GetTimestamp();
        Interlocked.Increment(ref SyncStalls);
        _waitSequence = sequence;
        while (Acknowledged < sequence && !_stopping)
        {
            ParkProducer();
            ThrowIfFailed();
        }
        ThrowIfFailed();
        if (Acknowledged < sequence) throw new OperationCanceledException("The issue prefix was not acknowledged.");
        Interlocked.Add(ref AcknowledgementWaitTicks, Stopwatch.GetTimestamp() - started);
        return true;
    }

    internal void Wake() => _available.Set();
    internal void Stop() { _stopping = true; _available.Set(); _progress.Set(); }
    internal void Fail(Exception error)
    {
        Interlocked.CompareExchange(ref _failure, ExceptionDispatchInfo.Capture(error), null);
        _available.Set();
        _progress.Set();
    }
    internal void ThrowIfFailed() => Volatile.Read(ref _failure)?.Throw();
}
