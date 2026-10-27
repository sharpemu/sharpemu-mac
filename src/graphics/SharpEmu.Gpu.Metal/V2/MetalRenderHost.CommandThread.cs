// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Agc;
using SharpEmu.Libs.Gpu.GpuCommands;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// SHARPEMU_METAL_V2_COMMAND_THREAD=1: the command stream is interpreted on a thread of its own
/// (DeferredCommandThread, the Vulkan presenter's command thread), and the GPU worker runs what it
/// hands over in stream order - the translation's deferred draws and dispatches, and the host calls
/// the stream makes. The worker no longer reads packets or writes registers: Grand Theft Auto V spent
/// about 5 us of its 23 us draw there. Program resolution stays on the prefetch thread
/// (ParallelResolve), so the stream runs on three threads as on Vulkan.
///
/// A host call made on the command thread either changes what only the worker owns and is queued
/// behind the work before it (Post), or needs an answer and waits for the worker to run everything
/// before it (Run). Command memory is read on the command thread itself unless something the worker
/// records may own the bytes. A deferred dispatch cannot suspend its packet, so the worker waits for
/// a compute pipeline's compile in this mode (SupportsPipelineSuspend).
/// </summary>
internal sealed partial class MetalRenderHost
{
    internal static readonly bool CommandThreadEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_COMMAND_THREAD") == "1";

    private static readonly bool CommandThreadQosEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_COMMAND_THREAD_QOS") == "1";

    // Items run in batches; completions and relay commands are taken between them.
    private const int CommandItemBatch = 256;

    private DeferredCommandThread? _commandThread;
    private readonly AutoResetEvent _workerWake = new(false);
    private Action? _betweenCommandItems;

    private bool OnCommandThread => _commandThread is { IsProducerThread: true };

    /// <summary>Whether the worker runs the command thread's items instead of interpreting slices.</summary>
    internal bool UsesCommandThread => CommandThreadEnabled && _translation is not null;

    /// <summary>Items are queued for the worker or the command thread failed.</summary>
    internal bool CommandThreadHasWork => _commandThread is { HasWork: true };

    /// <summary>Wakes the worker parked for the command thread's items (a relay command, a completion).</summary>
    internal void WakeWorker() => _workerWake.Set();

    // The worker's share of the stream for one step: the queued items until the budget ends.
    private void RunCommandThreadBatch()
    {
        if (_commandThread is null)
        {
            _commandThread = new DeferredCommandThread(CommandStream, _memory, RunDeferredWork, _translation!.Prefetch, WakeWorker,
                useHostQos: CommandThreadQosEnabled)
            {
                ProducerIdle = RequestIdleCommit,
            };
            _betweenCommandItems = RunRelayBetweenCommandItems;
            _translation.Sink = _commandThread;
            _commandThread.Start();
            if (CommandThreadQosEnabled)
                Console.Error.WriteLine($"[METAL-V2][QOS] command_stream requested=UserInteractive applied={_commandThread.HasHostQos}.");
        }

        var deadline = System.Diagnostics.Stopwatch.GetTimestamp() + System.Diagnostics.Stopwatch.Frequency / 100;
        int ran;
        using (SharpEmu.Libs.VideoOut.RenderPhaseProfile.Measure(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.CommandStream))
        {
            ran = _commandThread.RunBatch(deadline, CommandItemBatch, _betweenCommandItems!);
        }

        // The worker running out of items says nothing about the stream: the command thread is usually
        // still interpreting. The open buffer goes out when the stream has nothing runnable
        // (RequestIdleCommit), when it holds enough, or when its first deferred flush is overdue.
        // Held label copies go out with that commit too: flushing them after every batch committed the
        // buffer some 2500 times a second, a batch being a handful of items.
        if (ran != 0)
        {
            _lastProgressTicks = System.Diagnostics.Stopwatch.GetTimestamp();
            CommitCoalescedIfOverdue();
        }
    }

    private string FormatCommandThreadCounters()
    {
        if (_commandThread is not { } thread)
        {
            return "off";
        }

        var culture = System.Globalization.CultureInfo.InvariantCulture;
        var (runs, runWaitMs, spaceWaitMs, items) = thread.TakeCounters();
        return $"runs={runs} run_wait_ms={runWaitMs.ToString("F1", culture)} space_wait_ms={spaceWaitMs.ToString("F1", culture)} items={items} sites=({TakeSyncSites()})";
    }

    private int _idleCommitQueued;

    // On the command thread, about to wait with nothing runnable: what the worker recorded goes out,
    // as when the slices run dry, or a wait on a label in an unsubmitted buffer would never end.
    private void RequestIdleCommit()
    {
        if (Interlocked.Exchange(ref _idleCommitQueued, 1) != 0)
        {
            return;
        }

        _commandThread!.Post(() =>
        {
            Volatile.Write(ref _idleCommitQueued, 0);
            CommitCoalescedFlush();
            FlushPendingLabelWrites();
        });
    }

    private void RunDeferredWork(AgcExports.DeferredWork work) => _translation!.RunDeferred(work);

    // Commands other threads posted to the worker run between items, with the prefetch thread
    // held: they change the caches it reads.
    private void RunRelayBetweenCommandItems()
    {
        if (!_relay.HasPendingCommands)
        {
            return;
        }

        var prefetch = _translation?.Prefetch;
        prefetch?.Pause();
        try
        {
            _relay.RunPendingCommands();
        }
        finally
        {
            prefetch?.Resume();
        }
    }

    /// <summary>Parks the worker until the command thread queues an item, or the timeout.</summary>
    internal void WaitForCommandThreadWork(int milliseconds)
    {
        if (_commandThread is not { } thread)
        {
            _ = _workerWake.WaitOne(milliseconds);
            return;
        }

        // The flag is set before the check: the command thread reads it after it queues.
        thread.ConsumerIdle = true;
        try
        {
            if (!thread.HasWork)
            {
                _ = _workerWake.WaitOne(milliseconds);
            }
        }
        finally
        {
            thread.ConsumerIdle = false;
        }
    }

    /// <summary>
    /// A video-out change ordered behind the command streams (MetalV2Presenter's control barriers):
    /// the queue runs it where it interprets, and it belongs to the worker.
    /// </summary>
    internal void RunControlBarrier(Action work)
    {
        if (!RunOnWorker(work))
        {
            work();
        }
    }

    /// <summary>Ends the command thread, running its items until it has.</summary>
    internal void StopCommandThread()
    {
        if (_commandThread is not { } thread)
        {
            return;
        }

        thread.Stop(() => thread.RunBatch(long.MaxValue, int.MaxValue, _betweenCommandItems ?? (() => { })));
    }

    // Host work for the worker, queued behind the work before it when called on the command thread.
    private bool PostToWorker(Action action)
    {
        if (!OnCommandThread)
        {
            return false;
        }

        _commandThread!.Post(action);
        return true;
    }

    // Host work whose result the stream needs: on the command thread it waits for the worker.
    private bool RunOnWorker(Action action,
        [System.Runtime.CompilerServices.CallerMemberName] string member = "",
        [System.Runtime.CompilerServices.CallerLineNumber] int line = 0)
    {
        if (!OnCommandThread)
        {
            return false;
        }

        RunTagged(action, member, line);
        return true;
    }

    // A read the worker answers after everything queued before it.
    private bool ReadOnWorker(ulong address, Span<byte> destination, Func<ulong, byte[], bool> read,
        [System.Runtime.CompilerServices.CallerMemberName] string member = "",
        [System.Runtime.CompilerServices.CallerLineNumber] int line = 0)
    {
        var buffer = new byte[destination.Length];
        var result = false;
        RunTagged(() => result = read(address, buffer), member, line);
        buffer.CopyTo(destination);
        return result;
    }

    // Where the command thread waited for the worker, by caller, for the PERF line.
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, long> _syncSites = new();

    private void RunTagged(Action action,
        [System.Runtime.CompilerServices.CallerMemberName] string member = "",
        [System.Runtime.CompilerServices.CallerLineNumber] int line = 0)
    {
        _syncSites.AddOrUpdate($"{member}:{line}", 1, static (_, count) => count + 1);
        _commandThread!.Run(action);
    }

    private string TakeSyncSites()
    {
        var top = _syncSites.ToArray().OrderByDescending(pair => pair.Value).Take(4).ToArray();
        _syncSites.Clear();
        return string.Join(",", top.Select(pair => $"{pair.Key}={pair.Value}"));
    }

    // A polled word (WAIT_REG_MEM) on the command thread, read here: as the stream wrote it when a
    // command-processor write still owes it, else from memory. A label still in flight reads as it was,
    // the wait fails and parks, and lands with the commit the idle stream asks for (RequestIdleCommit)
    // and the completion that wakes the queue. Only bytes an image the GPU drew into owns, whose
    // writeback has to be asked for, go to the worker.
    private bool ReadWaitOperandOnCommandThread(ulong address, Span<byte> destination, bool commitPendingWrite)
    {
        var size = (ulong)destination.Length;
        _commandThread!.WaitForPendingRange(address, size);
        if (_imageCache.MayOwn(address, size) && _imageCache.HasGpuModifiedImageBytes(address, size))
        {
            return ReadOnWorker(address, destination, (at, bytes) => TryReadGuestWaitOperand(at, bytes, commitPendingWrite));
        }

        if (!_memory.TryRead(address, destination))
        {
            return false;
        }

        if (_cpOverlay.Count != 0 &&
            _cpOverlay.Coverage(address, size, _signaledSerialReader ??= SignaledSerial) != MetalOverlayCoverage.None)
        {
            _cpOverlay.Apply(address, destination);
        }

        return true;
    }

    // Command memory and polled words on the command thread: bytes a queued fill or copy writes wait
    // for it; bytes an image the GPU drew into, a pending GPU write or a command-processor or label
    // write owns are read by the worker; the rest are read here.
    private bool ReadOnCommandThread(ulong address, Span<byte> destination)
    {
        var size = (ulong)destination.Length;
        _commandThread!.WaitForPendingRange(address, size);
        if (IsOwnedConcurrent(ConcurrentQuery.Word, address, size))
        {
            return ReadOnWorker(address, destination, (at, bytes) => TryReadGuest(at, bytes));
        }

        return _memory.TryRead(address, destination);
    }
}
