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
public sealed partial class DeferredCommandThread : AgcExports.IDeferredWorkSink
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
        // Orders GPU commands and changes nothing a program resolution reads: the draws after it
        // resolve without waiting for it, as they do behind a draw.
        public bool Ungated;
        public AgcExports.DeferredWork? Work;
        public Action? Action;
        public ulong Address;
        public byte[]? Data;
        // The guest range a queued fill or copy writes; zero for other items.
        public ulong RangeSize;
        // Resolution jobs submitted before this non-draw item; they finish before it runs.
        public long JobsBefore;
        // A range item that stores this word over its whole range (PostFill).
        public bool IsFill;
        public uint FillValue;
        // An end-of-pipe label: the GPU writes it once the work before it retires, so a read on the
        // stream's thread finds the bytes as they are until then (PostLabel).
        public bool IsLabel;
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
        bool useHostQos = false,
        bool? recycleAsyncItems = null)
    {
        _wakeConsumer = wakeConsumer;
        _queue = queue;
        _memory = memory;
        _runDeferred = runDeferred;
        _prefetch = prefetch;
        _useHostQos = useHostQos;
        if (recycleAsyncItems ?? RecycleAsyncItemsEnabled)
            _recycledItems = new Item?[MaxOutstanding];
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

    // SHARPEMU_CTHREAD_ITEM_SITES=1: the non-draw items by kind and source, in the PERF line.
    private static readonly bool CountItemSites =
        Environment.GetEnvironmentVariable("SHARPEMU_CTHREAD_ITEM_SITES") == "1";

    private readonly Dictionary<string, long> _itemSites = new();

    private void NoteItemSite(Item item)
    {
        var name = item.Kind switch
        {
            Kind.Deferred => item.Work?.GetType().Name ?? "deferred",
            Kind.Write => "write",
            _ => (item.Sync ? "run:" : item.RangeSize != 0 ? "range:" : item.Ungated ? "ordering:" : "post:") + (item.Action?.Method.Name ?? "?"),
        };
        lock (_itemSites)
        {
            System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(_itemSites, name, out _)++;
        }
    }

    public string TakeItemSites()
    {
        if (!CountItemSites)
        {
            return string.Empty;
        }

        lock (_itemSites)
        {
            var top = string.Join(",", _itemSites.Where(pair => !pair.Key.StartsWith("drain:")).OrderByDescending(pair => pair.Value).Take(12).Select(pair => $"{pair.Key}={pair.Value}"));
            var drains = string.Join(",", _itemSites.Where(pair => pair.Key.StartsWith("drain:")).OrderByDescending(pair => pair.Value).Take(8).Select(pair => $"{pair.Key}={pair.Value}"));
            _itemSites.Clear();
            return $" items_by=({top}) drains_by=({drains})";
        }
    }

    public bool HasWork => Volatile.Read(ref _outstanding) != 0 || _failure is not null;

    // Read by the render loop before it waits, so a queued item wakes it.
    public bool ConsumerIdle
    {
        set => _consumerIdle = value;
    }

    public bool IsAlive => _helperThread is { } helper ? !helper.Join(TimeSpan.Zero) : _thread.IsAlive;

    // The producer side, on the command stream thread.

    public void EnqueueDeferred(AgcExports.DeferredWork work, bool isDraw)
    {
        var item = RentAsyncItem(Kind.Deferred);
        item.Work = work;
        item.IsDraw = isDraw;
        Enqueue(item);
    }

    public void Post(Action action)
    {
        var item = RentAsyncItem(Kind.Call);
        item.Action = action;
        Enqueue(item);
    }

    // A host call that only orders GPU commands (a barrier, a pass end): the resolutions of the
    // draws after it are not held for it.
    public void PostOrdering(Action action)
    {
        var item = RentAsyncItem(Kind.Call);
        item.Action = action;
        item.Ungated = true;
        Enqueue(item);
    }

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
        var item = RentAsyncItem(Kind.Write);
        item.Address = address;
        item.Data = data.ToArray();
        lock (_writeLock)
        {
            _pendingWrites.Enqueue(item);
            _pendingWriteCount = _pendingWrites.Count;
        }

        Enqueue(item);
    }

    // A fill or copy the render thread runs in order; reads of its destination wait for it.
    // An ungated one writes nothing but its range: the draws after it resolve their programs
    // without waiting for it, and a resolution that reads the range while it is still queued is
    // abandoned instead (PendingRangeHolds).
    public void PostRange(Action action, ulong address, ulong size, bool ungated = false)
    {
        var item = RentAsyncItem(Kind.Call);
        item.Action = action;
        item.Address = address;
        item.RangeSize = size;
        item.Ungated = ungated && size != 0;
        item.JobsBefore = _prefetch?.Submitted ?? 0;
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

    // An end-of-pipe label write: it holds the range only for the program resolutions after it
    // (PendingRangeHolds); the stream's own reads do not wait for it, as the label lands later.
    public void PostLabel(Action action, ulong address, ulong size, bool ungated = false)
    {
        var item = RentAsyncItem(Kind.Call);
        item.IsLabel = size != 0;
        item.Action = action;
        item.Address = address;
        item.RangeSize = size;
        item.Ungated = ungated && size != 0;
        item.JobsBefore = _prefetch?.Submitted ?? 0;
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

    // A fill the worker stages as a command-processor write of one repeated word: a read of bytes it
    // covers, with nothing queued after it over them, is answered from the word without waiting.
    public void PostFill(Action action, ulong address, ulong size, uint value, bool ungated = false)
    {
        var item = RentAsyncItem(Kind.Call);
        item.IsFill = size != 0;
        item.FillValue = value;
        item.Action = action;
        item.Address = address;
        item.RangeSize = size;
        item.Ungated = ungated && size != 0;
        item.JobsBefore = _prefetch?.Submitted ?? 0;
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

    /// <summary>
    /// The bytes as the stream wrote them when the last queued range item over them is a fill that
    /// covers them all and no queued store overlaps them; false otherwise. Command stream thread.
    /// </summary>
    public bool TryReadQueuedFill(ulong address, Span<byte> destination)
    {
        if (Volatile.Read(ref _pendingRangeCount) == 0 || destination.IsEmpty)
        {
            return false;
        }

        var size = (ulong)destination.Length;
        lock (_writeLock)
        {
            Item? last = null;
            foreach (var pending in _pendingRanges)
            {
                if (!pending.IsLabel && address < pending.Address + pending.RangeSize && pending.Address < address + size)
                {
                    last = pending;
                }
            }

            if (last is not { IsFill: true } fill ||
                address < fill.Address || address + size > fill.Address + fill.RangeSize)
            {
                return false;
            }

            foreach (var write in _pendingWrites)
            {
                if (address < write.Address + (ulong)write.Data!.Length && write.Address < address + size)
                {
                    return false;
                }
            }

            Span<byte> word = stackalloc byte[sizeof(uint)];
            System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(word, fill.FillValue);
            var phase = (int)((address - fill.Address) % sizeof(uint));
            for (var index = 0; index < destination.Length; index++)
            {
                destination[index] = word[(phase + index) % sizeof(uint)];
            }

            return true;
        }
    }

    // Runs every queued item when one of them is a fill or copy into this range.
    public void WaitForPendingRange(ulong address, ulong size)
    {
        if (_pendingRangeCount != 0 && OverlapsPendingRange(address, size))
        {
            if (CountItemSites)
            {
                NoteDrainSite(address, size);
            }

            Run(static () => { });
        }
    }

    /// <summary>
    /// Whether an ungated range item queued before resolution job <paramref name="job"/> (its index
    /// among the submitted jobs) still has to write bytes in the range. Any thread.
    /// </summary>
    public bool PendingRangeHolds(long job, ulong address, ulong size)
    {
        if (Volatile.Read(ref _pendingRangeCount) == 0)
        {
            return false;
        }

        lock (_writeLock)
        {
            foreach (var pending in _pendingRanges)
            {
                if (pending.Ungated && pending.JobsBefore <= job &&
                    address < pending.Address + pending.RangeSize && pending.Address < address + size)
                {
                    return true;
                }
            }
        }

        return false;
    }

    // Which read drained the queue, and which queued item held the bytes (SHARPEMU_CTHREAD_ITEM_SITES=1).
    private void NoteDrainSite(ulong address, ulong size)
    {
        var holder = "?";
        lock (_writeLock)
        {
            foreach (var pending in _pendingRanges)
            {
                if (!pending.IsLabel && address < pending.Address + pending.RangeSize && pending.Address < address + size)
                {
                    holder = $"{pending.Action?.Method.Name ?? "?"}:fill={pending.IsFill}:+{pending.RangeSize}";
                }
            }
        }

        var frames = new System.Diagnostics.StackTrace(2, false).GetFrames();
        var callers = string.Join("<", frames.Take(4).Select(frame => frame.GetMethod()?.Name ?? "?"));
        var name = $"drain:{callers}@{holder}/{size}";
        lock (_itemSites)
        {
            System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(_itemSites, name, out _)++;
        }
    }

    private bool OverlapsPendingRange(ulong address, ulong size)
    {
        lock (_writeLock)
        {
            foreach (var pending in _pendingRanges)
            {
                if (!pending.IsLabel && address < pending.Address + pending.RangeSize && pending.Address < address + size)
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

    // SHARPEMU_CT_ITEM_KINDS=1 (diagnostic): every two seconds, the non-draw items the stream handed over, by kind and
    // by the deferred work's type or the posted action's method.
    private static readonly bool TraceItemKinds = Environment.GetEnvironmentVariable("SHARPEMU_CT_ITEM_KINDS") == "1";
    private readonly Dictionary<string, int> _itemKinds = new(StringComparer.Ordinal);
    private long _itemKindsReported;

    private void NoteItemKind(Item item)
    {
        var key = item.Kind switch
        {
            Kind.Deferred => $"deferred:{item.Work?.GetType().Name}",
            Kind.Call => $"call:{item.Action?.Method.DeclaringType?.Name}.{item.Action?.Method.Name}{(item.RangeSize != 0 ? ":range" : string.Empty)}",
            _ => $"write:{item.Data?.Length}",
        };
        _itemKinds[key] = _itemKinds.GetValueOrDefault(key) + 1;
        var now = Stopwatch.GetTimestamp();
        if (now - _itemKindsReported < 2 * Stopwatch.Frequency)
        {
            return;
        }

        _itemKindsReported = now;
        Console.Error.WriteLine("[GPU][CT-ITEMS] " + string.Join(" ",
            _itemKinds.OrderByDescending(static pair => pair.Value).Take(12).Select(static pair => $"{pair.Key}={pair.Value}")));
        _itemKinds.Clear();
    }

    private void Enqueue(Item item)
    {
        ThrowIfFailed();
        if (!item.IsDraw && !item.Ungated)
        {
            item.JobsBefore = _prefetch?.Submitted ?? 0;
            _enqueuedNonDraw++;
            if (TraceItemKinds)
            {
                NoteItemKind(item);
            }
        }

        if (CountItemSites && !item.IsDraw)
        {
            NoteItemSite(item);
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

            // Execute has removed pending write/range references, and the consumer has finished
            // reading the item. Synchronous callers still own their result, so never recycle them.
            if (!item.Sync)
                RecycleAsyncItem(item);

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

        // A write still waits for the resolutions before it: they read the bytes as they were.
        if (!item.Ungated || item.RangeSize != 0)
        {
            _prefetch?.WaitFinished(item.JobsBefore);
        }

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
            if (!item.Ungated)
            {
                _executedNonDraw++;
            }

            if (item.RangeSize != 0)
            {
                lock (_writeLock)
                {
                    _pendingRanges.Remove(item);
                    _pendingRangeCount = _pendingRanges.Count;
                }
            }

            if (!item.Ungated)
            {
                _prefetch?.AdvanceGate(_executedNonDraw);
            }

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
