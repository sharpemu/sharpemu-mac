// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.DsTrace;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Runs work other threads post to the GPU worker before the next submission; the
/// twin of the Vulkan <c>GpuWorkerRelay</c>. The relay never waits for the GPU — it
/// only orders managed work against the recording thread.
/// </summary>
internal sealed class MetalGpuWorkerRelay : IGpuQueueRelay
{
    [ThreadStatic]
    private static MetalGpuWorkerRelay? _boundWorker;

    private readonly object _gate = new();
    private readonly Queue<Action> _commands = new();
    private readonly Action _wake;
    private readonly Func<Action, Action, bool>? _tryEnqueueAfterPendingWork;
    private int _pendingCount;
    private bool _accepting = true;

    public MetalGpuWorkerRelay(Action wake, Func<Action, Action, bool>? tryEnqueueAfterPendingWork = null)
    {
        _wake = wake;
        _tryEnqueueAfterPendingWork = tryEnqueueAfterPendingWork;
    }

    public bool IsGpuQueueThread => _boundWorker == this;

    public bool HasPendingCommands => Volatile.Read(ref _pendingCount) != 0;

    public void BindCurrentThread() => _boundWorker = this;

    public bool TryRunAfterPendingWork(Action work) => TryRunOnGpuQueue(work);

    // An ordered video-out change must land behind the command streams already accepted.
    public bool TryRunAfterAcceptedCommandStreams(Action work)
    {
        ArgumentNullException.ThrowIfNull(work);
        if (IsGpuQueueThread || _tryEnqueueAfterPendingWork is null)
        {
            return TryRunOnGpuQueue(work);
        }

        using var done = new SemaphoreSlim(0);
        System.Runtime.ExceptionServices.ExceptionDispatchInfo? failure = null;
        if (!_tryEnqueueAfterPendingWork(
                () =>
                {
                    try
                    {
                        work();
                    }
                    catch (Exception exception)
                    {
                        failure = System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(exception);
                    }
                    finally
                    {
                        done.Release();
                    }
                },
                () =>
                {
                    failure = System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(
                        new OperationCanceledException(
                            "The command stream stopped before the ordered GPU worker action could run."));
                    done.Release();
                }))
        {
            return false;
        }

        done.Wait();
        failure?.Throw();
        return true;
    }

    public void Post(Action work)
    {
        if (!TryPost(work))
        {
            throw GpuFatal.Fatal("The Metal GPU worker relay is closed.");
        }
    }

    public bool TryPost(Action work)
    {
        if (IsGpuQueueThread)
        {
            work();
            return true;
        }

        lock (_gate)
        {
            if (!_accepting)
            {
                return false;
            }

            _commands.Enqueue(work);
            Interlocked.Increment(ref _pendingCount);
        }

        _wake();
        return true;
    }

    public void RunPendingCommands()
    {
        if (!IsGpuQueueThread)
        {
            throw GpuFatal.Fatal("Only the Metal GPU worker can run relay commands.");
        }

        while (HasPendingCommands)
        {
            Action command;
            lock (_gate)
            {
                command = _commands.Dequeue();
                Interlocked.Decrement(ref _pendingCount);
            }

            command();
        }
    }

    public void RunOnGpuQueue(Action work)
    {
        if (!TryRunOnGpuQueue(work))
        {
            throw GpuFatal.Fatal("The Metal GPU worker relay is closed.");
        }
    }

    public bool TryRunOnGpuQueue(Action work)
    {
        if (IsGpuQueueThread)
        {
            work();
            return true;
        }

        using var done = new SemaphoreSlim(0);
        if (!TryPost(() =>
            {
                work();
                done.Release();
            }))
        {
            return false;
        }

        done.Wait();
        return true;
    }

    // Reject new work from other threads; the worker still runs what it accepted.
    public void StopAcceptingWork()
    {
        lock (_gate)
        {
            _accepting = false;
        }
    }
}

/// <summary>
/// Completion of submitted command buffers, through one MTLSharedEvent with a
/// monotonically increasing value per submission. A single thread waits with
/// <c>waitUntilSignaledValue:timeoutMS:</c> and runs the end-of-pipe work of each
/// submission in order — no Objective-C blocks and no per-buffer callback object.
/// </summary>
internal sealed class MetalCompletionQueue : IDisposable
{
    private const ulong WaitTimeoutMilliseconds = 50;

    // The safety net behind a completion handler: a missed wakeup costs this much
    // latency instead of hanging. With the handler bound it is never reached.
    private const int HandlerWaitMilliseconds = 4;

    private static readonly nint SelSignaledValue = MetalNative.Selector("signaledValue");
    private static readonly nint SelWaitUntilSignaledValue = MetalNative.Selector("waitUntilSignaledValue:timeoutMS:");

    private static readonly bool PollCompletions =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_POLL_COMPLETIONS") == "1";

    private readonly nint _sharedEvent;
    private readonly object _gate = new();
    private readonly object _wakeGate = new();
    private readonly List<(ulong Serial, Action Work)> _pending = new();
    private readonly SharpEmu.HLE.Host.HostBackgroundThread _thread;
    private readonly Action _wakeHandler;
    private readonly bool _handlerBound;
    private ulong _publishedSerial;
    private ulong _drainedSerial;
    private ulong _runningPublicationSerial;
    internal static bool WaitForActivePublications { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_WAIT_ACTIVE_PUBLICATIONS") != "0";
    private long _gpuWaitTicks;
    private long _gpuWaitSamples;
    private long _gpuReadySamples;
    private long _wakeWaitTicks;
    private long _handlerWakeups;
    private volatile bool _stopping;
    private Action? _completionObserver;

    public MetalCompletionQueue(nint sharedEvent)
    {
        _sharedEvent = sharedEvent;
        _wakeHandler = OnGpuCompleted;
        _handlerBound = !PollCompletions && MetalCompletionSignal.TryBind(_wakeHandler);
        Console.Error.WriteLine(
            _handlerBound
                ? "[METAL-V2][INFO] Submission completions wake the host through a command-buffer handler."
                : "[METAL-V2][INFO] Submission completions are polled on the shared event.");
        // Native, so the UserInteractive class asked for in Run takes effect.
        _thread = SharpEmu.HLE.Host.HostBackgroundThread.Start(
            "SharpEmu Metal completion", SharpEmu.HLE.Host.HostThreadQos.UserInteractive, 1024 * 1024, Run);
    }

    /// <summary>Whether a committed command buffer should carry the wakeup handler.</summary>
    public bool WantsCompletionHandler => _handlerBound;

    public long HandlerWakeups => Interlocked.Read(ref _handlerWakeups);

    /// <summary>
    /// Runs on the Metal callback thread for every completed command buffer, after the
    /// shared event is read. Must not allocate or block for long: the command stream's
    /// worker uses it to leave a parked label wait at once.
    /// </summary>
    public void SetCompletionObserver(Action? observer) => Volatile.Write(ref _completionObserver, observer);

    // Metal finished a command buffer: everyone watching the shared event looks again.
    private void OnGpuCompleted()
    {
        Interlocked.Increment(ref _handlerWakeups);
        // The submission that woke us has completed: publish it for the hint readers.
        _ = SignaledValue;
        lock (_wakeGate)
        {
            Monitor.PulseAll(_wakeGate);
        }

        Volatile.Read(ref _completionObserver)?.Invoke();

        // A frame waiting for this submission can be promoted now; the pacer used to
        // find out by polling every millisecond.
        MetalV2Presenter.SignalFlipPacer();
    }

    public ulong SignaledValue
    {
        get
        {
            var signaled = MetalV2Native.SendValue64(_sharedEvent, SelSignaledValue);
            NoteSignaled(signaled);
            return signaled;
        }
    }

    private ulong _completedHint;

    /// <summary>
    /// The newest value of the shared event any reader has seen: never above the event's own,
    /// which only grows. Read without a message send.
    /// </summary>
    public ulong CompletedHint => Volatile.Read(ref _completedHint);

    public void NoteSignaled(ulong value)
    {
        var seen = Volatile.Read(ref _completedHint);
        while (value > seen)
        {
            var before = Interlocked.CompareExchange(ref _completedHint, value, seen);
            if (before == seen)
            {
                return;
            }

            seen = before;
        }
    }

    /// <summary>Runs the work once the submission with this serial completes.</summary>
    public void Enqueue(ulong serial, Action work)
    {
        lock (_gate)
        {
            _pending.Add((serial, work));
            Monitor.PulseAll(_gate);
        }
    }

    private long _lastSignalObserved;

    /// <summary>When the completion thread last saw the event reach a new value.</summary>
    public long LastSignalObservedTimestamp => Interlocked.Read(ref _lastSignalObserved);

    /// <summary>How many completions are still waiting for their submission.</summary>
    public int PendingCount
    {
        get
        {
            lock (_gate)
            {
                return _pending.Count;
            }
        }
    }

    /// <summary>A command buffer was committed; its completion work may now run.</summary>
    public void Publish(ulong serial)
    {
        lock (_gate)
        {
            if (serial > _publishedSerial)
            {
                _publishedSerial = serial;
            }

            Monitor.PulseAll(_gate);
        }
    }

    /// <summary>
    /// How this thread spent its time since the last read: how often a submission was
    /// already signalled when it looked, how long it waited on the event otherwise,
    /// and how long it waited to be told a submission had been committed.
    /// </summary>
    public (long Ready, long Waited, double WaitMilliseconds, double WakeMilliseconds) TakeWaitProfile()
    {
        var ready = Interlocked.Exchange(ref _gpuReadySamples, 0);
        var waited = Interlocked.Exchange(ref _gpuWaitSamples, 0);
        var waitTicks = Interlocked.Exchange(ref _gpuWaitTicks, 0);
        var wakeTicks = Interlocked.Exchange(ref _wakeWaitTicks, 0);
        var frequency = (double)System.Diagnostics.Stopwatch.Frequency;
        return (ready, waited, waitTicks * 1000.0 / frequency, wakeTicks * 1000.0 / frequency);
    }

    /// <summary>Blocks until the GPU finished the serial and its completions ran.</summary>
    public MetalCompletionWait WaitFor(ulong serial)
    {
        if (serial == 0)
        {
            return default;
        }

        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        // WaitForSignal returns as soon as the queue is stopping, so without the
        // same test here the wait becomes a hot spin that never ends: the worker
        // then misses its five-second shutdown deadline.
        var complainAfter = started + (System.Diagnostics.Stopwatch.Frequency * StallSeconds);
        while (SignaledValue < serial && !_stopping)
        {
            WaitForSignal(serial);
            if (System.Diagnostics.Stopwatch.GetTimestamp() < complainAfter)
            {
                continue;
            }

            // A serial nothing will ever signal stops this thread for good, and
            // the guest then waits on a submitDone that never returns. From
            // outside that is indistinguishable from a slow GPU, so say which
            // serial is missing and how far the event actually got.
            complainAfter = long.MaxValue;
            MetalV2Diagnostics.WarnOnce(
                "completion-stall",
                $"A completion has not arrived in {StallSeconds}s and the wait is still running: " +
                $"serial={serial} signalled={SignaledValue} published={Volatile.Read(ref _publishedSerial)} " +
                $"pending={PendingCount}.");
        }

        var signalled = System.Diagnostics.Stopwatch.GetTimestamp();
        lock (_gate)
        {
            // Drain removes a callback before executing it. A signaled GPU event is
            // insufficient while that callback is still publishing guest table bytes,
            // even if another callback already advanced _drainedSerial to this serial.
            while (!_stopping && (WaitForActivePublications
                ? HasWorkAtOrBefore(serial) || (_runningPublicationSerial != 0 && _runningPublicationSerial <= serial)
                : _drainedSerial < serial && HasWorkAtOrBefore(serial)))
            {
                Monitor.Wait(_gate, 2);
            }
        }

        return new MetalCompletionWait(
            signalled - started, System.Diagnostics.Stopwatch.GetTimestamp() - signalled);
    }

    // How long a completion may be outstanding before the wait says so once.
    private const int StallSeconds = 5;

    private bool HasWorkAtOrBefore(ulong serial)
    {
        foreach (var (pendingSerial, _) in _pending)
        {
            if (pendingSerial <= serial)
            {
                return true;
            }
        }

        return false;
    }

    private void Run(bool qosApplied)
    {
        // A completion carries a flip and an interrupt the guest is waiting on, so
        // the thread is a frame-deadline thread like the worker and the presenter
        // (UserInteractive, applied by the thread's creation), and ahead of the guest's workers.
        SharpEmu.HLE.Host.HostThreadPrecedence.TryBoostCurrentThread("Metal completion");
        if (DsTrace.On)
        {
            DsTrace.RegisterThread(3, 0, Thread.CurrentThread.Name ?? "MetalCompletion", qosApplied ? 1 : 0);
        }
        while (true)
        {
            ulong target;
            var wakeStarted = System.Diagnostics.Stopwatch.GetTimestamp();
            lock (_gate)
            {
                while (!_stopping && (_pending.Count == 0 || MinimumPendingSerial() > _publishedSerial))
                {
                    Monitor.Wait(_gate, 20);
                }

                if (_stopping && _pending.Count == 0)
                {
                    return;
                }

                target = MinimumPendingSerial();
                if (target == ulong.MaxValue || target > _publishedSerial)
                {
                    continue;
                }
            }

            Interlocked.Add(ref _wakeWaitTicks, System.Diagnostics.Stopwatch.GetTimestamp() - wakeStarted);
            // This thread has no pool, and the completions it runs call into Metal: each
            // pass frees what the event reads and the completion work autorelease.
            using var pool = MetalAutoreleasePool.Enter();
            if (SignaledValue < target)
            {
                var waitStarted = System.Diagnostics.Stopwatch.GetTimestamp();
                WaitForSignal(target);
                Interlocked.Add(ref _gpuWaitTicks, System.Diagnostics.Stopwatch.GetTimestamp() - waitStarted);
                Interlocked.Increment(ref _gpuWaitSamples);
            }
            else
            {
                Interlocked.Increment(ref _gpuReadySamples);
            }

            Interlocked.Exchange(ref _lastSignalObserved, System.Diagnostics.Stopwatch.GetTimestamp());
            Drain(SignaledValue);
        }
    }

    // The event carries a flip and the interrupts the guest is waiting on, so the
    // wait is latency-critical. waitUntilSignaledValue:timeoutMS: measured about
    // forty-five milliseconds per completion on this host — its full timeout — even
    // though the value had arrived: it does not wake this process on a GPU signal.
    // A command-buffer completion handler does wake it, so the wait is a short spin
    // and then a park that the handler ends; the timeout is only a safety net.
    private void WaitForSignal(ulong target)
    {
        var frequency = System.Diagnostics.Stopwatch.Frequency;
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        var spinDeadline = started + (frequency / 20000);
        if (!_handlerBound)
        {
            PollForSignal(target, started, started + (frequency / 2000), started + (frequency / 4));
            return;
        }

        while (SignaledValue < target)
        {
            if (System.Diagnostics.Stopwatch.GetTimestamp() < spinDeadline)
            {
                Thread.SpinWait(32);
                continue;
            }

            lock (_wakeGate)
            {
                // Inside the lock the handler cannot pulse between the test and the
                // wait, so a completion is never missed.
                if (SignaledValue >= target || _stopping)
                {
                    return;
                }

                _ = Monitor.Wait(_wakeGate, HandlerWaitMilliseconds);
            }
        }
    }

    /// <summary>
    /// Blocks any thread until the GPU has signalled the serial or the timeout passes. Only
    /// the signal: unlike <see cref="WaitFor"/> it does not wait for the completions to run,
    /// so it never waits on the completion thread.
    /// </summary>
    public bool WaitForSignalled(ulong target, int timeoutMilliseconds)
    {
        if (SignaledValue >= target)
        {
            return true;
        }

        var frequency = System.Diagnostics.Stopwatch.Frequency;
        var deadline = System.Diagnostics.Stopwatch.GetTimestamp() + frequency * timeoutMilliseconds / 1000;
        while (!_stopping)
        {
            lock (_wakeGate)
            {
                // Inside the lock the handler cannot pulse between the test and the wait.
                if (SignaledValue >= target)
                {
                    return true;
                }

                var remaining = deadline - System.Diagnostics.Stopwatch.GetTimestamp();
                if (remaining <= 0)
                {
                    return false;
                }

                // Polled as well as woken: without the completion handler nothing pulses.
                var milliseconds = (int)Math.Clamp(remaining * 1000 / frequency, 1, HandlerWaitMilliseconds);
                _ = Monitor.Wait(_wakeGate, milliseconds);
            }
        }

        return SignaledValue >= target;
    }

    // The original path, kept for A/B measurement under
    // SHARPEMU_METAL_V2_POLL_COMPLETIONS=1 and as the fallback when the completion
    // block could not be built.
    private void PollForSignal(ulong target, long started, long spinDeadline, long pollDeadline)
    {
        _ = started;
        while (SignaledValue < target)
        {
            var now = System.Diagnostics.Stopwatch.GetTimestamp();
            if (now < spinDeadline)
            {
                Thread.SpinWait(32);
                continue;
            }

            if (now >= pollDeadline)
            {
                // The submission is far later than any frame; let the thread park so
                // a stuck queue costs nothing, and re-check on the next round.
                _ = MetalV2Native.SendWaitUntilSignaledValue(
                    _sharedEvent, SelWaitUntilSignaledValue, target, WaitTimeoutMilliseconds);
                return;
            }

            Thread.Sleep(1);
        }
    }

    private ulong MinimumPendingSerial()
    {
        var minimum = ulong.MaxValue;
        foreach (var (serial, _) in _pending)
        {
            if (serial < minimum)
            {
                minimum = serial;
            }
        }

        return minimum;
    }

    // Runs completed work in submission order; a failure never stops the queue.
    private void Drain(ulong signaled)
    {
        while (true)
        {
            Action? work = null;
            ulong serial = 0;
            lock (_gate)
            {
                var index = -1;
                var best = ulong.MaxValue;
                for (var candidate = 0; candidate < _pending.Count; candidate++)
                {
                    if (_pending[candidate].Serial <= signaled && _pending[candidate].Serial < best)
                    {
                        best = _pending[candidate].Serial;
                        index = candidate;
                    }
                }

                if (index >= 0)
                {
                    (serial, work) = _pending[index];
                    _pending.RemoveAt(index);
                    _runningPublicationSerial = serial;
                }
                else
                {
                    if (signaled > _drainedSerial)
                    {
                        _drainedSerial = signaled;
                    }

                    Monitor.PulseAll(_gate);
                    return;
                }
            }

            try
            {
                work();
            }
            catch (Exception exception)
            {
                Console.Error.WriteLine(
                    $"[METAL-V2][ERROR] An end-of-pipe completion failed: serial={serial} {exception}");
            }

            lock (_gate)
            {
                _runningPublicationSerial = 0;
                if (serial > _drainedSerial)
                {
                    _drainedSerial = serial;
                }

                Monitor.PulseAll(_gate);
            }
        }
    }

    public void Dispose()
    {
        MetalCompletionSignal.Unbind(_wakeHandler);
        lock (_gate)
        {
            _stopping = true;
            Monitor.PulseAll(_gate);
        }

        lock (_wakeGate)
        {
            Monitor.PulseAll(_wakeGate);
        }

        _thread.Join(2000);
    }
}

/// <summary>How a host wait for one submission spent its time.</summary>
internal readonly record struct MetalCompletionWait(long SignalTicks, long DrainTicks);
