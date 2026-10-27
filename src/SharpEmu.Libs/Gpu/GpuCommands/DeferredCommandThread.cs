// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.HLE;
using SharpEmu.Libs.Agc;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Libs.Gpu.GpuCommands;

// The Vulkan presenter's command thread (VulkanVideoPresenter.CommandThread) for any backend: the
// command stream is interpreted on this thread and its draws, dispatches, stores and callbacks run
// on the backend's own worker in stream order, with ParallelProgramPrefetch resolving draws'
// programs ahead on a third. Work in progress for Metal V2; the Vulkan presenter still uses its
// own copy.
// The command stream is interpreted on its own thread, which hands its work to the render thread
// in stream order. Draws, dispatches, barriers, completions and the interpreter's own stores are
// queued; a callback that returns a result waits for the render thread to run everything before it.
public sealed class DeferredCommandThread : AgcExports.IDeferredWorkSink
{
    // The thread runs this far ahead of the render thread at most: enough to keep it busy, and
    // no more, since a GPU round trip (a readback, a label wait) waits for everything queued.
    // Measured in Astro Bot: 4096 items halved the frame rate of a crowd scene, 16 starved the
    // render thread, 32 to 64 kept it busy.
    private const int MaxOutstanding = 64;

    private enum Kind : byte
    {
        Deferred,
        Call,
        Write,
    }

    private sealed class Item
    {
        public Kind Kind;
        public bool IsDraw;
        public AgcExports.DeferredWork? Work;
        public Action? Action;
        public ulong Address;
        public byte[]? Data;
        // The guest range a queued fill or copy writes; zero for other items.
        public ulong RangeSize;
        // Resolution jobs submitted before this non-draw item; they finish before it runs.
        public long JobsBefore;
        public bool Sync;
        public volatile bool Done;
        public Exception? Failure;
    }

    private readonly CommandStreamQueue _queue;
    private readonly ICpuMemory _memory;
    private readonly Action<AgcExports.DeferredWork> _runDeferred;
    private readonly ParallelProgramPrefetch? _prefetch;
    // Wakes the thread that runs the queued items (the backend's render or GPU worker thread).
    private readonly Action _wakeConsumer;
    private readonly Thread _thread;
    private readonly bool _useHostQos;
    private SharpEmu.HLE.Host.HostHelperThread? _helperThread;
    private int _producerThreadId;
    private int _started;
    private readonly object _lock = new();
    private List<Item> _incoming = new();
    private List<Item> _draining = new();
    private int _drainIndex;
    private int _outstanding;
    // What the producer sleeps on: room in the queue, or its synchronous item.
    private volatile bool _waitingForSpace;
    private volatile bool _waitingForSync;
    private volatile bool _consumerIdle;
    private volatile bool _stopping;
    private volatile Exception? _failure;
    private long _enqueuedNonDraw;
    private long _executedNonDraw;

    // The interpreter's stores the render thread has not applied yet, oldest first; reads on
    // this thread see them over guest memory.
    private readonly object _writeLock = new();
    private readonly Queue<Item> _pendingWrites = new();
    private volatile int _pendingWriteCount;

    // The destinations of queued fills and copies; a read on this thread that overlaps one waits
    // for the render thread to run it.
    private readonly List<Item> _pendingRanges = new();
    private volatile int _pendingRangeCount;

    private static readonly long RetryTicks = CommandStreamQueue.AllBlockedRetryMilliseconds * Stopwatch.Frequency / 1000L;

    public DeferredCommandThread(
        CommandStreamQueue queue,
        ICpuMemory memory,
        Action<AgcExports.DeferredWork> runDeferred,
        ParallelProgramPrefetch? prefetch,
        Action wakeConsumer,
        bool useHostQos = false)
    {
        _wakeConsumer = wakeConsumer;
        _queue = queue;
        _memory = memory;
        _runDeferred = runDeferred;
        _prefetch = prefetch;
        _useHostQos = useHostQos;
        _thread = new Thread(Run) { IsBackground = true, Name = "SharpEmu command stream" };
    }

    public void Start()
    {
        if (Interlocked.Exchange(ref _started, 1) != 0)
            throw new ThreadStateException("The command thread has already started.");
        if (_useHostQos)
            _helperThread = SharpEmu.HLE.Host.HostHelperThread.Start("SharpEmu command stream", 16 * 1024 * 1024, Run);
        else
            _thread.Start();
    }

    /// <summary>Whether the producer was created with its requested native QoS class.</summary>
    public bool HasHostQos => _helperThread?.HasQos == true;

    // The producer's side since the last TakeCounters: synchronous calls, their wait, the wait for
    // room in the queue, and the items handed over.
    private long _syncRuns;
    private long _syncWaitTicks;
    private long _spaceWaitTicks;
    private long _itemsEnqueued;

    public (long SyncRuns, double SyncWaitMs, double SpaceWaitMs, long Items) TakeCounters() =>
        (Interlocked.Exchange(ref _syncRuns, 0),
            Interlocked.Exchange(ref _syncWaitTicks, 0) * 1000.0 / Stopwatch.Frequency,
            Interlocked.Exchange(ref _spaceWaitTicks, 0) * 1000.0 / Stopwatch.Frequency,
            Interlocked.Exchange(ref _itemsEnqueued, 0));

    /// <summary>
    /// Called on the command stream thread when nothing in the queue can run and it is about to wait:
    /// the backend may need to submit what it recorded so the work the stream waits on can land.
    /// </summary>
    public Action? ProducerIdle { get; set; }

    public bool IsProducerThread => Environment.CurrentManagedThreadId == Volatile.Read(ref _producerThreadId);

    public long Gate => _enqueuedNonDraw;

    public bool HasWork => Volatile.Read(ref _outstanding) != 0 || _failure is not null;

    // Read by the render loop before it waits, so a queued item wakes it.
    public bool ConsumerIdle
    {
        set => _consumerIdle = value;
    }

    public bool IsAlive => _helperThread is { } helper ? !helper.Join(TimeSpan.Zero) : _thread.IsAlive;

    // The producer side, on the command stream thread.

    public void EnqueueDeferred(AgcExports.DeferredWork work, bool isDraw) =>
        Enqueue(new Item { Kind = Kind.Deferred, Work = work, IsDraw = isDraw });

    public void Post(Action action) => Enqueue(new Item { Kind = Kind.Call, Action = action });

    public void Run(Action action)
    {
        var item = new Item { Kind = Kind.Call, Action = action, Sync = true };
        Interlocked.Increment(ref _syncRuns);
        Enqueue(item);
        var started = Stopwatch.GetTimestamp();
        WaitFor(item);
        Interlocked.Add(ref _syncWaitTicks, Stopwatch.GetTimestamp() - started);
    }

    public T Run<T>(Func<T> function)
    {
        T result = default!;
        Run(() => { result = function(); });
        return result;
    }

    public void PostWrite(ulong address, ReadOnlySpan<byte> data)
    {
        var item = new Item { Kind = Kind.Write, Address = address, Data = data.ToArray() };
        lock (_writeLock)
        {
            _pendingWrites.Enqueue(item);
            _pendingWriteCount = _pendingWrites.Count;
        }

        Enqueue(item);
    }

    // A fill or copy the render thread runs in order; reads of its destination wait for it.
    public void PostRange(Action action, ulong address, ulong size)
    {
        var item = new Item { Kind = Kind.Call, Action = action, Address = address, RangeSize = size };
        if (size != 0)
        {
            lock (_writeLock)
            {
                _pendingRanges.Add(item);
                _pendingRangeCount = _pendingRanges.Count;
            }
        }

        Enqueue(item);
    }

    // Runs every queued item when one of them is a fill or copy into this range.
    public void WaitForPendingRange(ulong address, ulong size)
    {
        if (_pendingRangeCount != 0 && OverlapsPendingRange(address, size))
        {
            Run(static () => { });
        }
    }

    private bool OverlapsPendingRange(ulong address, ulong size)
    {
        lock (_writeLock)
        {
            foreach (var pending in _pendingRanges)
            {
                if (address < pending.Address + pending.RangeSize && pending.Address < address + size)
                {
                    return true;
                }
            }

            return false;
        }
    }

    // Guest memory as the stream sees it: with its own stores applied, queued or not.
    public bool ReadOverlay(ulong address, Span<byte> destination)
    {
        WaitForPendingRange(address, (ulong)destination.Length);
        if (_pendingWriteCount == 0)
        {
            return _memory.TryRead(address, destination);
        }

        lock (_writeLock)
        {
            if (!_memory.TryRead(address, destination))
            {
                return false;
            }

            var end = address + (ulong)destination.Length;
            foreach (var write in _pendingWrites)
            {
                var writeEnd = write.Address + (ulong)write.Data!.Length;
                if (write.Address >= end || writeEnd <= address)
                {
                    continue;
                }

                var start = Math.Max(address, write.Address);
                var stop = Math.Min(end, writeEnd);
                write.Data.AsSpan((int)(start - write.Address), (int)(stop - start))
                    .CopyTo(destination[(int)(start - address)..]);
            }

            return true;
        }
    }

    private void Enqueue(Item item)
    {
        ThrowIfFailed();
        if (!item.IsDraw)
        {
            item.JobsBefore = _prefetch?.Submitted ?? 0;
            _enqueuedNonDraw++;
        }

        Interlocked.Increment(ref _itemsEnqueued);
        if (Volatile.Read(ref _outstanding) >= MaxOutstanding)
        {
            var started = Stopwatch.GetTimestamp();
            WaitForSpace();
            Interlocked.Add(ref _spaceWaitTicks, Stopwatch.GetTimestamp() - started);
        }

        lock (_lock)
        {
            _incoming.Add(item);
        }

        Interlocked.Increment(ref _outstanding);
        if (_consumerIdle || item.Sync)
        {
            _wakeConsumer();
        }
    }

    private void WaitForSpace()
    {
        lock (_lock)
        {
            // The flag is set before the check: the render thread reads it after it changes what is checked.
            for (;;)
            {
                _waitingForSpace = true;
                Interlocked.MemoryBarrier();
                if (Volatile.Read(ref _outstanding) < MaxOutstanding / 2 || _failure is not null)
                {
                    break;
                }

                Monitor.Wait(_lock, 10);
            }

            _waitingForSpace = false;
        }

        ThrowIfFailed();
    }

    private void WaitFor(Item item)
    {
        var spinner = new SpinWait();
        for (var spin = 0; spin < 100 && !item.Done; spin++)
        {
            spinner.SpinOnce(sleep1Threshold: -1);
        }

        if (!item.Done)
        {
            lock (_lock)
            {
                for (;;)
                {
                    _waitingForSync = true;
                    Interlocked.MemoryBarrier();
                    if (item.Done || _failure is not null)
                    {
                        break;
                    }

                    Monitor.Wait(_lock, 10);
                }

                _waitingForSync = false;
            }
        }

        if (item.Failure is { } failure)
        {
            throw new InvalidOperationException("A command stream callback failed on the render thread.", failure);
        }

        if (!item.Done)
        {
            ThrowIfFailed();
        }
    }

    // The consumer side, on the render thread. Runs queued items until the deadline or until
    // none is left; returns how many ran.
    public int RunBatch(long deadline, int maxItems, Action betweenItems)
    {
        if (_drainIndex == _draining.Count)
        {
            _draining.Clear();
            _drainIndex = 0;
            lock (_lock)
            {
                (_incoming, _draining) = (_draining, _incoming);
            }

            if (_draining.Count == 0)
            {
                ThrowIfFailed();
                return 0;
            }
        }

        var ran = 0;
        while (_drainIndex < _draining.Count)
        {
            var item = _draining[_drainIndex];
            _draining[_drainIndex++] = null!;
            Execute(item);
            ran++;
            var outstanding = Interlocked.Decrement(ref _outstanding);
            // One pulse per wait: the producer sets its flag again if it keeps waiting.
            var wakeForSync = item.Sync && _waitingForSync;
            var wakeForSpace = _waitingForSpace && outstanding < MaxOutstanding / 2;
            if (wakeForSync || wakeForSpace)
            {
                if (wakeForSync)
                {
                    _waitingForSync = false;
                }

                if (wakeForSpace)
                {
                    _waitingForSpace = false;
                }

                lock (_lock)
                {
                    Monitor.PulseAll(_lock);
                }
            }

            betweenItems();
            if (ran >= maxItems || ((ran & 31) == 0 && Stopwatch.GetTimestamp() >= deadline))
            {
                break;
            }
        }

        return ran;
    }

    private void Execute(Item item)
    {
        if (item.IsDraw)
        {
            _runDeferred(item.Work!);
            return;
        }

        _prefetch?.WaitFinished(item.JobsBefore);
        try
        {
            switch (item.Kind)
            {
                case Kind.Deferred:
                    _runDeferred(item.Work!);
                    break;
                case Kind.Call:
                    item.Action!();
                    break;
                case Kind.Write:
                    ApplyWrite(item);
                    break;
            }
        }
        catch (Exception exception) when (item.Sync)
        {
            item.Failure = exception;
        }
        catch (Exception exception)
        {
            _failure ??= exception;
            throw;
        }
        finally
        {
            _executedNonDraw++;
            if (item.RangeSize != 0)
            {
                lock (_writeLock)
                {
                    _pendingRanges.Remove(item);
                    _pendingRangeCount = _pendingRanges.Count;
                }
            }

            _prefetch?.AdvanceGate(_executedNonDraw);
            if (item.Sync)
            {
                item.Done = true;
            }
        }
    }

    private void ApplyWrite(Item item)
    {
        lock (_writeLock)
        {
            var written = _memory.TryWrite(item.Address, item.Data);
            if (!ReferenceEquals(_pendingWrites.Peek(), item))
            {
                throw GpuFatal.Fatal("The command stream stores ran out of order.");
            }

            _pendingWrites.Dequeue();
            _pendingWriteCount = _pendingWrites.Count;
            if (!written)
            {
                throw GpuFatal.Fatal(
                    $"The command stream cannot write guest memory: address=0x{item.Address:X16} size={item.Data!.Length}.");
            }
        }
    }

    public void ThrowIfFailed()
    {
        if (_failure is { } failure)
        {
            throw new InvalidOperationException("The command stream thread failed.", failure);
        }
    }

    // Ends the thread after its current slice; the render thread keeps running its items
    // meanwhile, since the slice may wait for one.
    public void Stop(Action runItems)
    {
        _stopping = true;
        _queue.Wake();
        var deadline = Stopwatch.GetTimestamp() + 5 * Stopwatch.Frequency;
        while (IsAlive && Stopwatch.GetTimestamp() < deadline)
        {
            runItems();
            if (_helperThread is { } helper) helper.Join(TimeSpan.FromMilliseconds(1));
            else _thread.Join(1);
        }

        runItems();
    }

    private void Run()
    {
        // A native helper can enter before Start has returned its wrapper. Publish
        // identity from the body, rather than depending on that wrapper assignment.
        Volatile.Write(ref _producerThreadId, Environment.CurrentManagedThreadId);
        try
        {
            var lastProgress = Stopwatch.GetTimestamp();
            while (!_stopping)
            {
                var result = _queue.ProcessOne();
                switch (result)
                {
                    case SliceResult.NoWork:
                        ProducerIdle?.Invoke();
                        _ = _queue.WaitForWork(8);
                        break;
                    case SliceResult.AllBlocked:
                        ProducerIdle?.Invoke();
                        WaitBlocked(ref lastProgress);
                        break;
                    case SliceResult.BlockedWithoutProgress:
                        if (!_queue.HasUnblockedPending)
                        {
                            ProducerIdle?.Invoke();
                            WaitBlocked(ref lastProgress);
                        }

                        break;
                    default:
                        lastProgress = Stopwatch.GetTimestamp();
                        break;
                }
            }
        }
        catch (Exception exception)
        {
            _failure ??= exception;
            _wakeConsumer();
        }
    }

    // Blocked heads retry when a GPU completion or a presentation unblocks them, or every
    // retry interval without progress.
    private void WaitBlocked(ref long lastProgress)
    {
        var remaining = RetryTicks - (Stopwatch.GetTimestamp() - lastProgress);
        if (remaining <= 0)
        {
            _queue.RetryBlocked();
            lastProgress = Stopwatch.GetTimestamp();
            return;
        }

        _ = _queue.WaitForRetryInterval((int)Math.Max(1, remaining * 1000 / Stopwatch.Frequency));
    }
}
