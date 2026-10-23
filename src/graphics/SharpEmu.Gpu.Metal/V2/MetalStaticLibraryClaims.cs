// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Who builds a library: the static precompile, which stores libraries the game has not
/// asked for yet, or the frame path, which needs one now. A precompile worker registers
/// the key of a library it will build (<see cref="Register"/>), usually well before it is
/// let into the driver (MetalCompileScheduler holds it while the game compiles), and moves
/// it to building once it is (<see cref="Entry.TryBegin"/>). The frame path asks
/// <see cref="Demand"/> before it compiles a library the archive lacks:
/// - registered and not begun: the frame path takes it, the worker drops it when it gets
///   there and compiles nothing;
/// - begun: the frame path waits for the worker's file (bounded) and loads it;
/// - not registered: nothing to wait for.
/// A library is therefore built once, by whoever needs it first, and a runtime need is
/// never queued behind precompile work that has not started.
/// </summary>
internal sealed class MetalStaticLibraryClaims
{
    private const int Queued = 0;
    private const int Building = 1;
    private const int Taken = 2;

    private readonly ConcurrentDictionary<UInt128, Entry> _entries = new();
    private long _taken;
    private long _waited;
    private long _timedOut;

    /// <summary>Libraries the frame path took before a precompile worker began them.</summary>
    public long TakenCount => Interlocked.Read(ref _taken);

    /// <summary>Libraries the frame path waited for a precompile worker's build of.</summary>
    public long WaitedCount => Interlocked.Read(ref _waited);

    /// <summary>Waits that gave up and left the frame path to compile.</summary>
    public long TimedOutCount => Interlocked.Read(ref _timedOut);

    /// <summary>Libraries registered and not yet released.</summary>
    public int Count => _entries.Count;

    /// <summary>Registers <paramref name="key"/>; null when a worker already has it.</summary>
    public Entry? Register(UInt128 key)
    {
        var entry = new Entry(this, key);
        return _entries.TryAdd(key, entry) ? entry : null;
    }

    /// <summary>
    /// The frame path needs the library of <paramref name="key"/>, which the archive does not
    /// have. True when it waited for a worker's build: load the archive again. False: compile
    /// it (a queued worker's registration is taken, so the worker leaves it).
    /// </summary>
    public bool Demand(UInt128 key, TimeSpan timeout)
    {
        if (!_entries.TryGetValue(key, out var entry))
        {
            return false;
        }

        if (Interlocked.CompareExchange(ref entry.State, Taken, Queued) == Queued)
        {
            Interlocked.Increment(ref _taken);
            return false;
        }

        if (Volatile.Read(ref entry.State) != Building)
        {
            return false;
        }

        if (entry.Done.Wait(timeout))
        {
            Interlocked.Increment(ref _waited);
            return true;
        }

        Interlocked.Increment(ref _timedOut);
        return false;
    }

    public sealed class Entry : IDisposable
    {
        internal int State;
        private readonly MetalStaticLibraryClaims _owner;
        private readonly UInt128 _key;
        private int _released;

        internal Entry(MetalStaticLibraryClaims owner, UInt128 key)
        {
            _owner = owner;
            _key = key;
        }

        internal ManualResetEventSlim Done { get; } = new(false);

        /// <summary>The worker may build it: false when the frame path took it first.</summary>
        public bool TryBegin() => Interlocked.CompareExchange(ref State, Building, Queued) == Queued;

        /// <summary>The worker is done (built, failed or dropped): wakes whoever waits.</summary>
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _released, 1) != 0)
            {
                return;
            }

            _owner._entries.TryRemove(_key, out _);
            // Not disposed: a frame-path wait may still be waking from it.
            Done.Set();
        }
    }
}
