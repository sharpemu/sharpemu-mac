// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.HLE.Host;

public sealed class HostBackingObject : IDisposable
{
    private readonly Action<HostBackingObject>? _release;
    private readonly Func<HostBackingObject, HostBackingReleaseResult>? _ownedRelease;
    private const long ClosingBit = 1L << 32;
    private static long _nextIdentity;
    private long _accessState;
    private bool _releaseInProgress;
    private bool _terminalReleaseFailure;
    private HostBackingReleaseResult _lastRelease;
    private bool _isDisposed;
    private HostManagedWriteIntervals? _managedWrites;
    // Alias accesses (SharedBackingViews' lock-free single-view reads and writes) count here,
    // each thread on its own cache line, instead of on _accessState, which every thread shares.
    // Null with SHARPEMU_ACCESS_STRIPES=0: alias accesses then take TryEnterAccess.
    private readonly HostAccessStripes? _aliasStripes = AccessStripesEnabled ? new HostAccessStripes() : null;
    internal const int AliasLane = 0;
    internal static readonly bool AccessStripesEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_ACCESS_STRIPES") != "0";
    internal HostAccessStripes? AliasStripes => _aliasStripes;

    // Optional ordinary bookkeeping only. No wait or callback during enrollment;
    // a concurrent first copy may allocate a discarded candidate, while warmed
    // ordinary writers only read the published reference and bounded scalar slots.
    internal void EnrollManagedWrites()
    {
        if (Volatile.Read(ref _managedWrites) is not null) return;
        var candidate = new HostManagedWriteIntervals(this);
        _ = Interlocked.CompareExchange(ref _managedWrites, candidate, null);
    }
    internal HostManagedWriteIntervals.Token BeginManagedWrite(ulong offset, ulong count) =>
        Volatile.Read(ref _managedWrites)?.Begin(offset, count) ?? default;
    internal HostManagedWriteIntervals.Token BeginManagedWriteOperation() =>
        Volatile.Read(ref _managedWrites)?.BeginOperation() ?? default;
    internal HostManagedWriteIntervals? ManagedWrites => Volatile.Read(ref _managedWrites);
    internal static void EndManagedWrite(HostManagedWriteIntervals.Token token, bool succeeded)
    {
        // Failure means stale/foreign bookkeeping, never repeat actual bytes.
        _ = token.Owner?.End(token, succeeded);
    }

    /// <summary>
    /// The lock the owning backend holds while it maps, unmaps or disposes this
    /// object, so a view operation cannot race the release callback.
    /// </summary>
    /// <remarks>Host-package contract.</remarks>
    public object Gate { get; } = new();

    /// <summary>
    /// The OS handle for the backing storage: a Mach memory entry on macOS, a
    /// section handle on Windows. Zero when the backend identifies the storage
    /// by <see cref="Descriptor"/> instead.
    /// </summary>
    /// <remarks>Host-package contract; written by the owning backend.</remarks>
    public nint Handle { get; init; }

    /// <summary>
    /// The file descriptor for the backing storage on the POSIX backends
    /// (memfd/shm), or -1 when the backend identifies it by
    /// <see cref="Handle"/> instead.
    /// </summary>
    /// <remarks>Host-package contract; written by the owning backend.</remarks>
    public int Descriptor { get; init; } = -1;

    /// <summary>
    /// Bytes per separately releasable chunk when the backend built the backing from many
    /// purgeable objects instead of one (macOS <c>SHARPEMU_GUEST_BACKING=chunked</c>);
    /// 0 for a single object, whose pages no in-process call can give back.
    /// </summary>
    /// <remarks>Host-package contract; written by the owning backend.</remarks>
    public ulong ChunkBytes { get; init; }

    /// <summary>
    /// Bytes per purgeable object behind the chunks, a multiple of <see cref="ChunkBytes"/>
    /// (the last object may be short); 0 when <see cref="ChunkBytes"/> is 0. A no-copy GPU
    /// buffer costs about the square of the objects it spans under Rosetta, so windows over
    /// the backing are sized in objects, not chunks.
    /// </summary>
    /// <remarks>Host-package contract; written by the owning backend.</remarks>
    public ulong ObjectBytes { get; init; }

    /// <summary>
    /// The backend's release of whole chunks, set together with <see cref="ChunkBytes"/>.
    /// Arguments are the backing, a chunk-aligned offset and a size that is a multiple of
    /// <see cref="ChunkBytes"/> or ends at <see cref="Size"/>. Afterwards every byte of the
    /// range reads zero through the alias and every view. The caller holds alias access.
    /// </summary>
    /// <remarks>Host-package contract; written by the owning backend.</remarks>
    public Func<HostBackingObject, ulong, ulong, HostBackingDiscard>? DiscardChunks { get; init; }

    /// <summary>
    /// The backend's second mapping of a chunked backing for GPU windows
    /// (<see cref="IHostBackingGpuAlias"/>), set together with <see cref="ChunkBytes"/>;
    /// null when the backend cannot make one. Each call makes a new mapping the caller owns.
    /// </summary>
    /// <remarks>Host-package contract; written by the owning backend.</remarks>
    public Func<HostBackingObject, IHostBackingGpuAlias?>? CreateGpuAlias { get; init; }

    /// <summary>
    /// What another process needs to map this backing (<see cref="IHostGuestMirrorMemory.TryMapMirror"/>);
    /// null unless the backend was asked to make it shareable. Its descriptor or handles belong to
    /// this backing and are released with it.
    /// </summary>
    /// <remarks>Host-package contract; written by the owning backend.</remarks>
    public GuestBackingExport? Export { get; init; }

    /// <summary>
    /// Creates a backing object for storage the caller has already reserved at
    /// <paramref name="aliasBase"/>.
    /// </summary>
    /// <param name="aliasBase">Host address of the reserved alias window.</param>
    /// <param name="size">Size of the backing storage in bytes.</param>
    /// <param name="release">
    /// Invoked once, under <see cref="Gate"/>, when the object is disposed; the
    /// owning backend releases the OS storage there.
    /// </param>
    /// <remarks>
    /// Host-package contract: the per-OS packages are the only code that creates
    /// backing storage, so they are the only callers.
    /// </remarks>
    public HostBackingObject(ulong aliasBase, ulong size, Action<HostBackingObject> release)
    {
        ArgumentNullException.ThrowIfNull(release);
        AliasBase = aliasBase;
        Size = size;
        _release = release;
        Identity = NextIdentity();
    }

    /// <summary>
    /// Provider-created alias/descriptor lifetime. This does not own guest virtual
    /// mappings or authorize write-fault recovery. The callback runs once after
    /// admission closes and all balanced ordinary accesses have exited.
    /// </summary>
    public HostBackingObject(ulong aliasBase, ulong size, ulong nativePageSize,
        Func<HostBackingObject, HostBackingReleaseResult> release)
    {
        ArgumentNullException.ThrowIfNull(release);
        if (!HostViewMemory.IsValidRange(aliasBase, size) || nativePageSize == 0 ||
            (nativePageSize & (nativePageSize - 1)) != 0 || aliasBase % nativePageSize != 0)
            throw new ArgumentOutOfRangeException(nameof(aliasBase));
        var remainder = size % nativePageSize;
        var padding = remainder == 0 ? 0 : nativePageSize - remainder;
        if (padding > ulong.MaxValue - size || size + padding > ulong.MaxValue - aliasBase)
            throw new ArgumentOutOfRangeException(nameof(size));
        AliasBase = aliasBase;
        Size = size;
        NativePageSize = nativePageSize;
        NativeAliasSize = size + padding;
        _ownedRelease = release;
        Identity = NextIdentity();
    }

    public long Identity { get; }
    public ulong NativePageSize { get; }
    public ulong NativeAliasSize { get; }
    public bool HasOwnedLifetime => _ownedRelease is not null;
    public bool IsClosing => HasOwnedLifetime && (Volatile.Read(ref _accessState) & ClosingBit) != 0;
    /// <summary>Count observation including briefly rejected entry bookkeeping; not native access authority.</summary>
    public int ActiveAccesses => HasOwnedLifetime ? (int)Volatile.Read(ref _accessState) + (int)AliasAccesses : 0;
    // Signed sum of the alias stripes; fatal when negative (an exit without an enter).
    private long AliasAccesses => _aliasStripes?.Sum(AliasLane) ?? 0;
    public HostBackingReleaseResult LastRelease { get { lock (Gate) return _lastRelease; } }

    /// <summary>
    /// Balanced ordinary caller contract. No signal safety, mapping pin, permission
    /// authority or content-write serialization is implied. Every success must exit
    /// exactly once in finally, after the final alias/descriptor access.
    /// </summary>
    public bool TryEnterAccess()
    {
        if (!HasOwnedLifetime) return !IsDisposed;
        if (!CanStartAccessEntry()) return false;
        return CompleteAccessEntry(IncrementAccessCount());
    }

    private bool CanStartAccessEntry() => (Volatile.Read(ref _accessState) & ClosingBit) == 0;
    private long IncrementAccessCount() => Interlocked.Increment(ref _accessState);
    private bool CompleteAccessEntry(long state)
    {
        // Increment's returned value linearizes admission against Close, not
        // the early veto. A caller preempted after the veto may increment after
        // Close: reject it without alias/descriptor access and undo its count.
        if ((state & ClosingBit) == 0 && (uint)state <= int.MaxValue) return true;
        Interlocked.Decrement(ref _accessState);
        return false;
    }

    public void ExitAccess()
    {
        if (!HasOwnedLifetime) return;
        var state = Interlocked.Decrement(ref _accessState);
        if ((uint)state > int.MaxValue)
        {
            Interlocked.Increment(ref _accessState);
            throw new InvalidOperationException("Unbalanced backing access exit.");
        }
    }

    /// <summary>
    /// <see cref="TryEnterAccess"/> for an alias access, counted on the calling thread's stripe.
    /// Same admission against <see cref="CloseAccess"/>: the stripe increment is a full fence
    /// before the closing re-check, and the drain sets the closing bit with a full fence before
    /// it sums the stripes, so either the entrant sees Close or the drain sees the entrant.
    /// Every success must <see cref="ExitAliasAccess"/> exactly once, after the final access.
    /// </summary>
    internal bool TryEnterAliasAccess()
    {
        if (_aliasStripes is not { } stripes) return TryEnterAccess();
        // A legacy (not owned) lifetime has no drain here; its stripes count for the views' drains.
        if (HasOwnedLifetime ? !CanStartAccessEntry() : IsDisposed) return false;
        var cell = stripes.Enter(AliasLane);
        if (HasOwnedLifetime ? CanStartAccessEntry() : !IsDisposed) return true;
        stripes.ExitCell(cell);
        return false;
    }

    internal void ExitAliasAccess()
    {
        if (_aliasStripes is not { } stripes) { ExitAccess(); return; }
        stripes.Exit(AliasLane);
    }

    /// <summary>Revokes new ordinary alias/descriptor accesses; never reopens admission.</summary>
    public void CloseAccess() { if (HasOwnedLifetime) Interlocked.Or(ref _accessState, ClosingBit); }

    private static long NextIdentity()
    {
        while (true)
        {
            var previous = Volatile.Read(ref _nextIdentity);
            if (previous == long.MaxValue) throw new InvalidOperationException("Backing identity exhausted.");
            if (Interlocked.CompareExchange(ref _nextIdentity, previous + 1, previous) == previous)
                return previous + 1;
        }
    }

    public ulong AliasBase { get; }

    public ulong Size { get; }

    /// <summary>Whether <see cref="Dispose"/> has already released the storage.</summary>
    /// <remarks>Host-package contract; read under <see cref="Gate"/>.</remarks>
    public bool IsDisposed => Volatile.Read(ref _isDisposed);

    // Legacy providers retain their existing release callback semantics.
    public void Dispose()
    {
        if (!HasOwnedLifetime)
        {
            lock (Gate)
            {
                if (IsDisposed) return;
                Volatile.Write(ref _isDisposed, true);
                _release!(this);
            }
            return;
        }
        if (!TryDispose(TimeSpan.FromSeconds(2), out var result))
            throw new HostBackingReleaseException(result);
    }

    /// <summary>
    /// Ordinary close/drain/revalidate. Caller must hold no access pin or mapping,
    /// protection, backing, or view gate. Only pre-native timeout/busy is retryable;
    /// native failure/exception keeps ownership quarantined without repeating cleanup.
    /// </summary>
    public bool TryDispose(TimeSpan timeout, out HostBackingReleaseResult result) =>
        TryDisposeCore(timeout, out result, afterDrain: null);

    // Null on the production path. The private ordinary-only phase seam lets
    // deterministic managed tests pause after zero observation; no access path,
    // signal callback, TLS, or public authorization object uses this callback.
    private bool TryDisposeCore(TimeSpan timeout, out HostBackingReleaseResult result, Action? afterDrain)
    {
        if (timeout < TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(timeout));
        if (HasOwnedLifetime && Monitor.IsEntered(Gate))
            throw new InvalidOperationException("Backing disposal cannot drain under its gate.");
        if (!HasOwnedLifetime) { Dispose(); result = new(HostBackingReleaseStatus.Released); return true; }
        lock (Gate)
        {
            if (IsDisposed) { result = _lastRelease; return true; }
            CloseAccess();
            if (_terminalReleaseFailure) { result = _lastRelease; return false; }
            if (_releaseInProgress) { result = new(HostBackingReleaseStatus.Busy); return false; }
            _releaseInProgress = true;
        }
        try
        {
            var start = System.Diagnostics.Stopwatch.GetTimestamp();
            while (ActiveAccesses != 0)
            {
                if (System.Diagnostics.Stopwatch.GetElapsedTime(start) >= timeout)
                { result = new(HostBackingReleaseStatus.TimedOut); return false; }
                Thread.Sleep(1); // ordinary side only; no access/provider gate held
            }
            afterDrain?.Invoke();
            lock (Gate)
            {
                // Close prevents fresh successful admissions. Zero observed
                // after Close proves prior admitted accesses exited. A delayed
                // veto-loser can now briefly increment, but will never touch
                // alias/descriptor storage; do not reject cleanup for that count.
                if (!_releaseInProgress || !IsClosing)
                    throw new InvalidOperationException("Backing release reservation changed.");
            }
            try
            {
                result = _ownedRelease!(this);
                if (result.Status is HostBackingReleaseStatus.NotAttempted or HostBackingReleaseStatus.Busy or HostBackingReleaseStatus.TimedOut ||
                    (uint)result.Status > (uint)HostBackingReleaseStatus.CallbackOutcomeUnknown ||
                    (result.Completed && result.NativeError != 0))
                    throw new InvalidOperationException("Invalid provider cleanup result.");
            }
            catch
            {
                lock (Gate)
                {
                    _terminalReleaseFailure = true;
                    _lastRelease = new(HostBackingReleaseStatus.CallbackOutcomeUnknown);
                }
                throw;
            }
            lock (Gate)
            {
                _lastRelease = result;
                _terminalReleaseFailure = !result.Completed;
                if (result.Completed) Volatile.Write(ref _isDisposed, true);
            }
            return result.Completed;
        }
        finally { lock (Gate) _releaseInProgress = false; }
    }
}

/// <summary>
/// What a <see cref="HostBackingObject.DiscardChunks"/> call did: the chunks it emptied
/// (their pages went back to the OS) and the bytes it had to zero-fill instead, because
/// their pages stayed resident (wired by a GPU command buffer, or touched while it ran).
/// </summary>
public readonly record struct HostBackingDiscard(ulong EmptiedBytes, ulong EmptiedChunks, ulong ZeroFilledBytes);
