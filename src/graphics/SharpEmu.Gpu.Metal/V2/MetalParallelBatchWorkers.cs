// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.ExceptionServices;
using SharpEmu.HLE.Host;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>One wake and one join per contiguous chunk; no per-draw scheduling.</summary>
internal sealed class MetalParallelBatchWorkers : IDisposable
{
    private readonly Thread?[] _threads;
    private readonly HostQosThread?[] _qosThreads;
    private readonly AutoResetEvent[] _ready;
    private readonly AutoResetEvent _finished = new(false);
    private int _remaining;
    private readonly ExceptionDispatchInfo?[] _failures;
    private readonly Action<int> _execute;
    private volatile bool _stopping;
    private bool _disposed;

    internal int Count => _threads.Length + 1;
    internal int NativeQosWorkers => _qosThreads.Count(thread => thread is not null);

    internal MetalParallelBatchWorkers(int count, Action<int> execute)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(count, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(count, 10);
        _execute = execute;
        _threads = new Thread?[count - 1];
        _qosThreads = new HostQosThread?[count - 1];
        _ready = new AutoResetEvent[count - 1];
        _failures = new ExceptionDispatchInfo?[count];
        for (var i = 0; i < _ready.Length; i++) _ready[i] = new AutoResetEvent(false);
        try
        {
            for (var i = 0; i < _threads.Length; i++)
            {
                var worker = i + 1;
                var name = $"SharpEmu Metal batch {worker}";
                _qosThreads[i] = HostQosThread.TryStart(name, HostThreadQos.UserInteractive,
                    4 * 1024 * 1024, () => Run(worker), out _);
                if (_qosThreads[i] is null)
                {
                    var thread = new Thread(() => Run(worker)) { IsBackground = true, Name = name };
                    thread.Start();
                    _threads[i] = thread;
                }
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    // The caller owns this pool and every input until Execute returns, including on
    // failure. All chunks finish before a failure is rethrown or storage is reused.
    internal void Execute()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Array.Clear(_failures);
        _remaining = _threads.Length;
        foreach (var ready in _ready) ready.Set();
        Invoke(0);
        if (_threads.Length != 0) _finished.WaitOne();
        foreach (var failure in _failures) failure?.Throw();
    }

    private void Invoke(int worker)
    {
        try { _execute(worker); }
        catch (Exception exception) { _failures[worker] = ExceptionDispatchInfo.Capture(exception); }
    }

    private void Run(int worker)
    {
        while (true)
        {
            _ready[worker - 1].WaitOne();
            if (_stopping) return;
            try { Invoke(worker); }
            finally { if (Interlocked.Decrement(ref _remaining) == 0) _finished.Set(); }
        }
    }

    internal static (int Start, int End) Chunk(int count, int workers, int worker)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        ArgumentOutOfRangeException.ThrowIfLessThan(workers, 1);
        if ((uint)worker >= (uint)workers) throw new ArgumentOutOfRangeException(nameof(worker));
        return ((int)((long)count * worker / workers), (int)((long)count * (worker + 1) / workers));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _stopping = true;
        foreach (var ready in _ready) ready.Set();
        foreach (var thread in _threads) thread?.Join();
        foreach (var thread in _qosThreads) thread?.Join(Timeout.Infinite);
        foreach (var ready in _ready) ready.Dispose();
        _finished.Dispose();
    }
}
