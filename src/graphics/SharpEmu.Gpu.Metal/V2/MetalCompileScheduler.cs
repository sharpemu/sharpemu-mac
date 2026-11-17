// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Orders the driver's compiler service between the game and the background prewarm. Every
/// front-end or back-end call reaches one MTLCompilerService, which runs them side by side,
/// so a runtime miss the GPU worker waits for competed with up to twelve prewarm compiles
/// (the compute and graphics prewarm's six threads and the library list's six) and took
/// seconds instead of a fraction of one. Demon's Souls' boot spent 80 s of a 99 s window
/// waiting for 34 such libraries (run 20261018-203322-ds-ds-memopt-check, game.log:1037).
///
/// A runtime compile counts from the moment the game asks for it (a module queued on
/// MetalLibraryCompilePool, a pipeline state made on the worker) until it is done. A
/// prewarm compile enters through <see cref="EnterPrewarm"/>, which lets it start only
/// while no runtime compile is pending and none finished within the quiet period (the
/// worker's misses come in bursts, and the gap between two of them used to be enough for
/// six prewarm compiles to start), and never more than the prewarm's own share at once. A
/// prewarm compile already in the driver runs on: a runtime miss for the same source
/// waits for it rather than starting a second one (MetalShaderPrewarm.WaitWhileCompiling,
/// MetalLibraryPrewarm.TryTake).
///
/// So that the prewarm is not starved forever while the game compiles continuously, a
/// prewarm thread that has waited for <see cref="StarvationLimit"/> may start one compile
/// whenever no other prewarm compile is in the driver: a trickle of one.
///
/// Buffer-word variants (MetalRenderHost.BufferWordVariants) compile through a lane of their
/// own (<see cref="EnterVariant"/>): the same rule against the game's compiles, but their own
/// slots, so the static precompile, which can hold every prewarm slot for many minutes, does
/// not starve them, and they take none of its slots either. A variant that loads from the
/// title's archive enters no lane at all.
/// </summary>
internal sealed class MetalCompileScheduler
{
    private readonly object _sync = new();
    private readonly int _prewarmSlots;
    private readonly long _quietTicks;
    private readonly long _starvationTicks;
    private int _runtimePending;
    private long _lastRuntimeDoneTicks;
    private int _prewarmInFlight;
    private readonly int _variantSlots;
    private int _variantInFlight;
    private int _waiters;
    private long _heldTicks;
    private long _trickled;
    private long _runtimeCompiles;

    public MetalCompileScheduler(int prewarmSlots, TimeSpan quiet, TimeSpan starvationLimit, int variantSlots = DefaultVariantSlots)
    {
        _prewarmSlots = Math.Max(1, prewarmSlots);
        _variantSlots = Math.Max(1, variantSlots);
        _quietTicks = (long)(quiet.TotalSeconds * Stopwatch.Frequency);
        _starvationTicks = (long)(starvationLimit.TotalSeconds * Stopwatch.Frequency);
        StarvationLimit = starvationLimit;
        _lastRuntimeDoneTicks = Stopwatch.GetTimestamp() - _quietTicks;
    }

    /// <summary>
    /// The process's scheduler: the prewarm as a whole keeps as many compiles in the driver
    /// as one prewarm's threads did, where the front end ran fastest (MetalShaderPrewarm.WorkerCount).
    /// </summary>
    public static MetalCompileScheduler Shared { get; } = new(
        MetalShaderPrewarm.WorkerCount, TimeSpan.FromMilliseconds(100), TimeSpan.FromSeconds(1));

    /// <summary>
    /// Variant compiles in the driver at once: the two the variant builders compiled with when
    /// they shared the prewarm's slots.
    /// </summary>
    public const int DefaultVariantSlots = 2;

    public TimeSpan StarvationLimit { get; }

    /// <summary>Variant compiles in the driver right now.</summary>
    public int VariantInFlight => Volatile.Read(ref _variantInFlight);

    /// <summary>Runtime compiles asked for and not yet done.</summary>
    public int RuntimePending => Volatile.Read(ref _runtimePending);

    /// <summary>Prewarm compiles in the driver right now.</summary>
    public int PrewarmInFlight => Volatile.Read(ref _prewarmInFlight);

    /// <summary>The game asked for a compile it will wait for.</summary>
    public void RuntimeStarted()
    {
        Interlocked.Increment(ref _runtimePending);
        Interlocked.Increment(ref _runtimeCompiles);
    }

    /// <summary>A compile <see cref="RuntimeStarted"/> announced is done (or failed).</summary>
    public void RuntimeDone()
    {
        Volatile.Write(ref _lastRuntimeDoneTicks, Stopwatch.GetTimestamp());
        if (Interlocked.Decrement(ref _runtimePending) == 0 && Volatile.Read(ref _waiters) > 0)
        {
            lock (_sync)
            {
                Monitor.PulseAll(_sync);
            }
        }
    }

    /// <summary>A runtime compile for the length of a using block.</summary>
    public RuntimeScope Runtime()
    {
        RuntimeStarted();
        return new RuntimeScope(this);
    }

    /// <summary>
    /// Waits until a prewarm compile may start (see the class summary) and holds one prewarm
    /// slot until the returned scope is disposed. A default scope when <paramref name="stopping"/>
    /// turned true while waiting; the caller then compiles nothing.
    /// </summary>
    public PrewarmScope EnterPrewarm(Func<bool> stopping) => Enter(stopping, variant: false);

    /// <summary>
    /// As <see cref="EnterPrewarm"/>, a slot of the buffer-word variants' own lane: the same
    /// wait for the game's compiles, but none for the prewarm's.
    /// </summary>
    public PrewarmScope EnterVariant(Func<bool> stopping) => Enter(stopping, variant: true);

    private PrewarmScope Enter(Func<bool> stopping, bool variant)
    {
        var started = Stopwatch.GetTimestamp();
        var waited = false;
        var slots = variant ? _variantSlots : _prewarmSlots;
        lock (_sync)
        {
            _waiters++;
            try
            {
                while (true)
                {
                    if (stopping())
                    {
                        return default;
                    }

                    var inFlight = variant ? _variantInFlight : _prewarmInFlight;
                    var now = Stopwatch.GetTimestamp();
                    var quiet = Volatile.Read(ref _runtimePending) == 0 &&
                                now - Volatile.Read(ref _lastRuntimeDoneTicks) >= _quietTicks;
                    if (quiet && inFlight < slots)
                    {
                        break;
                    }

                    if (!quiet && inFlight == 0 && now - started >= _starvationTicks)
                    {
                        Interlocked.Increment(ref _trickled);
                        break;
                    }

                    waited = true;
                    // Short enough to notice the quiet period end; a runtime compile that
                    // finishes pulses the waiters.
                    Monitor.Wait(_sync, 10);
                }

                if (variant)
                {
                    _variantInFlight++;
                }
                else
                {
                    _prewarmInFlight++;
                }
            }
            finally
            {
                _waiters--;
            }
        }

        if (waited)
        {
            Interlocked.Add(ref _heldTicks, Stopwatch.GetTimestamp() - started);
        }

        return new PrewarmScope(this, variant);
    }

    private void PrewarmDone(bool variant)
    {
        lock (_sync)
        {
            if (variant)
            {
                _variantInFlight--;
            }
            else
            {
                _prewarmInFlight--;
            }

            Monitor.PulseAll(_sync);
        }
    }

    /// <summary>
    /// Since the last call: how long prewarm threads were held back for the game, how many
    /// compiles the starvation limit let through, and how many runtime compiles were asked for.
    /// </summary>
    public (double HeldMilliseconds, long Trickled, long RuntimeCompiles) TakeCounters() =>
        (Interlocked.Exchange(ref _heldTicks, 0) * 1000.0 / Stopwatch.Frequency,
            Interlocked.Exchange(ref _trickled, 0),
            Interlocked.Exchange(ref _runtimeCompiles, 0));

    public readonly struct RuntimeScope(MetalCompileScheduler? scheduler) : IDisposable
    {
        public void Dispose() => scheduler?.RuntimeDone();
    }

    public readonly struct PrewarmScope(MetalCompileScheduler? scheduler, bool variant = false) : IDisposable
    {
        /// <summary>False when the prewarm stopped while waiting: nothing may be compiled.</summary>
        public bool Entered => scheduler is not null;

        public void Dispose() => scheduler?.PrewarmDone(variant);
    }

    [ThreadStatic]
    private static long t_prewarmWaitTicks;

    /// <summary>
    /// A runtime compile waited <paramref name="ticks"/> for a compile of the same source the
    /// prewarm had in the driver; the thread running the runtime compile collects it.
    /// </summary>
    public static void NotePrewarmWait(long ticks) => t_prewarmWaitTicks += ticks;

    /// <summary>What <see cref="NotePrewarmWait"/> collected on this thread since the last call.</summary>
    public static long TakeThreadPrewarmWait()
    {
        var ticks = t_prewarmWaitTicks;
        t_prewarmWaitTicks = 0;
        return ticks;
    }
}
