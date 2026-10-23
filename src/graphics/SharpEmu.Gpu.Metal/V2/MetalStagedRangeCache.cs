// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>What the staged range cache needs from the device and the page tracker.</summary>
internal interface IStagedRangeBackend
{
    /// <summary>A shared buffer of at least this capacity, resident for the GPU, whose contents the host can write.</summary>
    (nint Handle, nint Contents, ulong GpuAddress) CreateBuffer(ulong capacity);

    void ReleaseBuffer(nint handle);

    /// <summary>Write-watches the guest pages of a range; every call is matched by one <see cref="RemoveWatch"/>.</summary>
    void AddWatch(ulong address, ulong size);

    void RemoveWatch(ulong address, ulong size);
}

/// <summary>
/// Guest ranges that no host window covers, kept staged between binds. Such a range
/// used to be copied into the upload ring on every bind, or on every command buffer for a
/// storage binding, which at Silent Hill's title was twelve gigabytes of copying a frame.
/// Here the copy lives in a buffer of its own for as long as nothing writes the bytes it
/// was taken from: a page watch catches the guest CPU's stores, and the host's own fills,
/// copies and recorded GPU writes name their range, so a bind after any of them stages
/// again.
/// </summary>
/// <remarks>
/// The worker thread binds and fills; the invalidation entry points run on any thread
/// that writes guest memory and take the same lock. A bind sees guest bytes as of the
/// time it is recorded: an entry is marked stale before the write that outdates it
/// returns, and a stale entry is staged again into a buffer no in-flight submission can
/// still read. An entry the shader has stored into (<see cref="Entry.Dirty"/>) is never
/// dropped before those stores are written back, since they are newer than anything
/// that wrote the range meanwhile; it is merely marked so it is staged again afterwards.
/// </remarks>
internal sealed class MetalStagedRangeCache
{
    public const ulong MinBytes = 256 * 1024;
    public const ulong MaxBytes = 64UL * 1024 * 1024;
    public const ulong ClassBytes = 1024 * 1024;
    // Every buffer of the pool is resident for every command buffer, so the budget is
    // memory the GPU keeps wired. Reads of ranges that are merely scattered over the
    // backing go through guest views (MetalGuestMemory.GuestViews.cs) and never reach the
    // cache; what does is written bindings and reads behind a write in the same command
    // buffer. At 1 GiB Demon's Souls filled 501 MB of it with read copies.
    public const ulong DefaultBudgetBytes = 256UL * 1024 * 1024;
    private const ulong WatchPageBytes = 4096;

    internal static readonly bool Enabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_STAGED_CACHE") != "0";

    public enum VerifyKind
    {
        Off,
        Full,
        Sample,
    }

    // SHARPEMU_METAL_V2_STAGED_CACHE_VERIFY=1 compares every hit whole; =sample compares
    // a slice of one hit in SampleEvery of each entry.
    internal static readonly VerifyKind VerifyMode =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_STAGED_CACHE_VERIFY") switch
        {
            "1" => VerifyKind.Full,
            "sample" => VerifyKind.Sample,
            _ => VerifyKind.Off,
        };

    public const uint SampleEvery = 64;
    public const int SampleEdgeBytes = 64 * 1024;
    public const int SamplePageBytes = 4096;
    public const int SamplePages = 4;

    public delegate void FreshReader(ulong offset, Span<byte> destination);

    /// <summary>
    /// The first offset at which the cached bytes differ from what <paramref name="read"/>
    /// returns now, or -1. A sample is the first and last 64 KB and four pseudo-random
    /// 4 KB pages chosen by <paramref name="seed"/>; otherwise the whole range.
    /// </summary>
    public static long FirstMismatch(ReadOnlySpan<byte> cached, FreshReader read, bool sample, uint seed)
    {
        var size = cached.Length;
        if (!sample || size <= 2 * SampleEdgeBytes + SamplePages * SamplePageBytes)
        {
            return Compare(cached, read, 0, size);
        }

        var best = long.MaxValue;
        var first = Compare(cached, read, 0, SampleEdgeBytes);
        if (first >= 0)
        {
            best = first;
        }

        var state = seed * 2654435761u + 0x9E3779B9u;
        for (var page = 0; page < SamplePages; page++)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            var slots = (uint)((size - SamplePageBytes) / SamplePageBytes + 1);
            var found = Compare(cached, read, (int)(state % slots) * SamplePageBytes, SamplePageBytes);
            if (found >= 0 && found < best)
            {
                best = found;
            }
        }

        var last = Compare(cached, read, size - SampleEdgeBytes, SampleEdgeBytes);
        if (last >= 0 && last < best)
        {
            best = last;
        }

        return best == long.MaxValue ? -1 : best;
    }

    private static long Compare(ReadOnlySpan<byte> cached, FreshReader read, int offset, int length)
    {
        var fresh = new byte[length];
        read((ulong)offset, fresh);
        var difference = cached.Slice(offset, length).CommonPrefixLength(fresh);
        return difference == length ? -1 : offset + difference;
    }

    private long _verifyChecks;
    private long _verifyMismatches;

    public void NoteVerify(bool mismatch)
    {
        Interlocked.Increment(ref _verifyChecks);
        if (mismatch)
        {
            Interlocked.Increment(ref _verifyMismatches);
        }
    }

    internal static ulong ConfiguredBudgetBytes =>
        ulong.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_STAGED_CACHE_MB"), out var megabytes) &&
        megabytes != 0
            ? megabytes * 1024 * 1024
            : DefaultBudgetBytes;

    public enum Reason
    {
        Fault,
        Precise,
        Host,
        Gpu,
        Unmap,
    }

    /// <summary>A buffer of the pool; its capacity is a multiple of <see cref="ClassBytes"/>.</summary>
    internal sealed class PoolBuffer
    {
        public nint Handle;
        public nint Contents;
        public ulong GpuAddress;
        public ulong Capacity;
    }

    internal sealed class Entry
    {
        public ulong Address;
        public ulong Size;

        // The staged bytes, and what guest memory held when they were staged (written entries only).
        public MetalGuestRange Staging;
        public MetalGuestRange Pristine;
        public PoolBuffer? StagingBuffer;
        public PoolBuffer? PristineBuffer;

        // Set from any thread that wrote the bytes; cleared only when the worker stages again.
        public int Stale;
        public ulong StagedSerial;
        public ulong LastUseSerial;
        public ulong WatchBegin;
        public ulong WatchEnd;

        // The shader may have stored into the copy, so guest memory is behind it.
        public uint HitCount;

        public bool Dirty;

        // The work that stores into it has been recorded. A copy bound for the draw being
        // prepared is written back only after that draw, never before it has run.
        public bool Armed;

        // The entry's own stores are being written back: that write is not news to it.
        public bool SelfWriteback;

        // Something wrote the range while its own stores were waiting for a writeback.
        public bool StaleOnClean;

        // The GPU fills the buffer, so its bytes are not visible to the host until the work completes.
        public bool Gathered;

        // Owned by the cache, as opposed to a ring copy the storage list keeps for one submission.
        public bool Cached;
        public bool InList;
        public bool Removed;
        internal uint QueryEpoch;

        public bool Overlaps(ulong address, ulong size) =>
            address < Address + Size && Address < address + size;
    }

    private readonly IStagedRangeBackend _backend;
    private readonly ulong _budget;
    private readonly object _gate = new();
    private readonly MetalStorePageCounts _storePages = new();
    internal bool StoreFastPathEnabled { get; set; } = MetalV2Diagnostics.StoreFastPath;
    private readonly Dictionary<(ulong Address, ulong Size), Entry> _entries = new();
    private readonly Dictionary<ulong, List<Entry>> _pages = new();
    private readonly long[] _everPages = new long[MetalImagePageOwners.PageCount / 64];
    private readonly List<(PoolBuffer Buffer, ulong RetireAfter)> _retired = new();
    private readonly Dictionary<ulong, Stack<PoolBuffer>> _free = new();
    private ulong _totalBytes;
    private ulong _completedSerial;
    private uint _queryEpoch;
    private int _entryCount;
    private bool _stopped;

    private long _hits;
    private long _misses;
    private long _stagedBytes;
    private long _evictions;
    private readonly long[] _invalidations = new long[5];

    public MetalStagedRangeCache(IStagedRangeBackend backend, ulong budgetBytes)
    {
        _backend = backend;
        _budget = budgetBytes;
    }

    public ulong CacheBytes
    {
        get
        {
            lock (_gate)
            {
                return _totalBytes;
            }
        }
    }

    public int EntryCount => Volatile.Read(ref _entryCount);

    /// <summary>
    /// Entries now, those with a live page watch, those holding stores not yet written back,
    /// those marked stale (unwatched, waiting for the next bind to stage them again), and
    /// those that would be used as they stand yet have no watch - which must always be zero.
    /// </summary>
    public (int Entries, int Watched, int Dirty, int Stale, int UnwatchedCurrent) Gauges()
    {
        lock (_gate)
        {
            int watched = 0, dirty = 0, stale = 0, unwatched = 0;
            foreach (var entry in _entries.Values)
            {
                var isStale = Volatile.Read(ref entry.Stale) != 0;
                watched += entry.WatchEnd != 0 ? 1 : 0;
                dirty += entry.Dirty ? 1 : 0;
                stale += isStale ? 1 : 0;
                unwatched += !isStale && !entry.Dirty && !entry.StaleOnClean && entry.WatchEnd == 0 ? 1 : 0;
            }

            return (_entries.Count, watched, dirty, stale, unwatched);
        }
    }

    public int FreeBufferCount
    {
        get
        {
            lock (_gate)
            {
                var count = 0;
                foreach (var stack in _free.Values)
                {
                    count += stack.Count;
                }

                return count;
            }
        }
    }

    public int RetiredCount
    {
        get
        {
            lock (_gate)
            {
                return _retired.Count;
            }
        }
    }

    /// <summary>False when no entry has ever covered a page of the range; answered without the lock.</summary>
    public bool MayCover(ulong address, ulong size) =>
        Volatile.Read(ref _entryCount) != 0 &&
        MetalImagePageOwners.TryGetPageRange(address, size, out var first, out var lastExclusive) &&
        (StoreFastPathEnabled ? _storePages.MayOverlap(address, size) : MayHavePages(first, lastExclusive));

    public static bool Eligible(ulong size) => size >= MinBytes && size <= MaxBytes;

    /// <summary>Counters for the periodic line; the per-window ones restart.</summary>
    public (long Hits, long Misses, long Bytes, long Fault, long Precise, long Host, long Gpu, long Unmap, long Evictions,
        long VerifyChecks, long VerifyMismatches) TakeCounters() =>
        (Interlocked.Exchange(ref _hits, 0), Interlocked.Exchange(ref _misses, 0),
            Interlocked.Exchange(ref _stagedBytes, 0),
            Interlocked.Exchange(ref _invalidations[(int)Reason.Fault], 0),
            Interlocked.Exchange(ref _invalidations[(int)Reason.Precise], 0),
            Interlocked.Exchange(ref _invalidations[(int)Reason.Host], 0),
            Interlocked.Exchange(ref _invalidations[(int)Reason.Gpu], 0),
            Interlocked.Exchange(ref _invalidations[(int)Reason.Unmap], 0),
            Interlocked.Exchange(ref _evictions, 0),
            Interlocked.Exchange(ref _verifyChecks, 0), Interlocked.Exchange(ref _verifyMismatches, 0));

    /// <summary>
    /// The entry a binding of the range uses, from the worker. True when it is current and
    /// needs no copy. False with an entry means the caller has to fill
    /// <see cref="Entry.Staging"/> (and <see cref="Entry.Pristine"/> when it is valid)
    /// before the buffer is recorded; false without one means the cache is full or
    /// stopped and the caller stages through the ring as before.
    /// </summary>
    public bool Acquire(
        ulong address, ulong size, bool written, ulong submissionSerial, ulong completedSerial, out Entry? entry)
    {
        lock (_gate)
        {
            _completedSerial = Math.Max(_completedSerial, completedSerial);
            Reap();
            entry = null;
            if (_stopped || !Eligible(size) ||
                !MetalImagePageOwners.TryGetPageRange(address, size, out _, out _))
            {
                return false;
            }

            if (_entries.TryGetValue((address, size), out var existing))
            {
                if (IsUsable(existing, written))
                {
                    existing.LastUseSerial = submissionSerial;
                    Interlocked.Increment(ref _hits);
                    entry = existing;
                    return true;
                }
            }

            var current = existing ?? new Entry { Address = address, Size = size, Cached = true };
            var needPristine = written;

            // Bytes an in-flight submission may still read are never rewritten: the old
            // buffers go back to the pool when that submission completes, and this entry
            // gets new ones.
            var inFlight = current.LastUseSerial > _completedSerial;
            if (inFlight)
            {
                RetireBuffers(current, current.LastUseSerial);
            }

            DropWatch(current);
            if (!needPristine && current.PristineBuffer is { } stalePristine)
            {
                // Its bytes are from before this staging, and nothing reads it unless a written binding does.
                PushFree(stalePristine);
                current.PristineBuffer = null;
                current.Pristine = default;
            }

            var capacity = Capacity(size);
            if (current.StagingBuffer is null)
            {
                if (!TakeBuffer(capacity, current, out var staging))
                {
                    Discard(current);
                    return false;
                }

                current.StagingBuffer = staging;
            }

            if (needPristine && current.PristineBuffer is null)
            {
                if (!TakeBuffer(capacity, current, out var pristine))
                {
                    Discard(current);
                    return false;
                }

                current.PristineBuffer = pristine;
            }

            current.Staging = new MetalGuestRange(
                current.StagingBuffer!.Handle, 0, size, current.StagingBuffer.GpuAddress);
            current.Pristine = current.PristineBuffer is { } p
                ? new MetalGuestRange(p.Handle, 0, size, p.GpuAddress)
                : default;
            current.Gathered = false;
            current.StaleOnClean = false;
            current.StagedSerial = submissionSerial;
            current.LastUseSerial = submissionSerial;

            // Marked current and watched before the caller reads the bytes: a write that
            // lands after this point stales the entry, even one that lands mid-copy.
            Volatile.Write(ref current.Stale, 0);
            if (existing is null)
            {
                Index(current);
            }

            current.WatchBegin = address;
            current.WatchEnd = address + size;
            _backend.AddWatch(address, size);

            Interlocked.Increment(ref _misses);
            Interlocked.Add(ref _stagedBytes, (long)size);
            entry = current;
            return false;
        }
    }

    private static bool IsUsable(Entry entry, bool written)
    {
        if (entry.Removed || (written && !entry.Pristine.IsValid))
        {
            return false;
        }

        // An entry with stores of its own is the newest word on its bytes.
        return entry.Dirty || (Volatile.Read(ref entry.Stale) == 0 && !entry.StaleOnClean);
    }

    /// <summary>The entry's stores reached guest memory (or it was dropped before they could).</summary>
    public void OnWrittenBack(Entry entry)
    {
        lock (_gate)
        {
            entry.Dirty = false;
            entry.SelfWriteback = false;
            if (entry.StaleOnClean)
            {
                entry.StaleOnClean = false;
                Volatile.Write(ref entry.Stale, 1);
                DropWatch(entry);
            }
        }
    }

    /// <summary>A submission records a use of the entry's buffers; they stay put until it completes.</summary>
    public void Touch(Entry entry, ulong submissionSerial)
    {
        lock (_gate)
        {
            entry.LastUseSerial = Math.Max(entry.LastUseSerial, submissionSerial);
        }
    }

    /// <summary>The entry's stores are being noted as GPU writes; they are not news to the entry.</summary>
    public void MarkSelfWriteback(Entry entry)
    {
        lock (_gate)
        {
            entry.SelfWriteback = true;
        }
    }

    /// <summary>A guest thread stored into a watched page. True when an entry shares it.</summary>
    public bool OnCpuWrite(ulong address, ulong size) =>
        Invalidate(address, size, Reason.Fault, pageGranular: true, origin: null, exact: false);

    /// <summary>A managed write of exactly these bytes; <paramref name="backed"/> writes land through the backing and never meet a watch.</summary>
    public bool OnPreciseCpuWrite(ulong address, ulong size, bool backed) =>
        Invalidate(address, size, Reason.Precise, pageGranular: !backed, origin: null, exact: false);

    /// <summary>The host wrote the range itself.</summary>
    public void OnHostWrite(ulong address, ulong size) =>
        Invalidate(address, size, Reason.Host, pageGranular: false, origin: null, exact: false);

    /// <summary>Recorded GPU work stores into the range.</summary>
    public void OnGpuWrite(ulong address, ulong size, MetalGpuWriteOrigin origin) =>
        Invalidate(address, size, Reason.Gpu, pageGranular: false, origin, exact: true);

    /// <summary>The guest unmapped the range: its entries go, stores and all, and their buffers return once the GPU is done.</summary>
    public void OnUnmap(ulong address, ulong size)
    {
        if (Volatile.Read(ref _entryCount) == 0 ||
            !MetalImagePageOwners.TryGetPageRange(address, size, out var first, out var lastExclusive))
        {
            return;
        }

        lock (_gate)
        {
            foreach (var entry in Collect(first, lastExclusive, address, size))
            {
                Interlocked.Increment(ref _invalidations[(int)Reason.Unmap]);
                RemoveEntry(entry);
            }
        }
    }

    /// <summary>Releases every page watch and takes no new ones, as the image cache does.</summary>
    public void StopWatching()
    {
        lock (_gate)
        {
            _stopped = true;
            foreach (var entry in _entries.Values)
            {
                DropWatch(entry);
            }
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            _stopped = true;
            foreach (var entry in _entries.Values)
            {
                DropWatch(entry);
                entry.Removed = true;
                _storePages.Remove(entry.Address, entry.Size);
                ReleaseBuffer(entry.StagingBuffer);
                ReleaseBuffer(entry.PristineBuffer);
            }

            _entries.Clear();
            _pages.Clear();
            Volatile.Write(ref _entryCount, 0);
            foreach (var (buffer, _) in _retired)
            {
                ReleaseBuffer(buffer);
            }

            _retired.Clear();
            foreach (var stack in _free.Values)
            {
                foreach (var buffer in stack)
                {
                    ReleaseBuffer(buffer);
                }
            }

            _free.Clear();
        }
    }

    private bool Invalidate(ulong address, ulong size, Reason reason, bool pageGranular, MetalGpuWriteOrigin? origin, bool exact)
    {
        if (Volatile.Read(ref _entryCount) == 0 || size == 0 ||
            !MetalImagePageOwners.TryGetPageRange(address, size, out var first, out var lastExclusive) ||
            !MayHavePages(first, lastExclusive))
        {
            return false;
        }

        if (StoreFastPathEnabled && !_storePages.MayOverlap(address, size))
        {
            MetalStoreFastPathCounters.Count(MetalStoreFastPathCounters.Kind.StagedFast);
            return false;
        }
        MetalStoreFastPathCounters.Count(MetalStoreFastPathCounters.Kind.StagedReject);

        var begin = address;
        var length = size;
        if (pageGranular)
        {
            begin = address & ~(WatchPageBytes - 1);
            length = ((address + size + WatchPageBytes - 1) & ~(WatchPageBytes - 1)) - begin;
        }

        var found = false;
        MetalStoreFastPathCounters.Count(MetalStoreFastPathCounters.Kind.StagedGate);
        lock (_gate)
        {
            foreach (var entry in Collect(first, lastExclusive, begin, length))
            {
                found = true;
                if (entry.SelfWriteback)
                {
                    continue;
                }

                if (entry.Dirty)
                {
                    // Its stores are newer than this write, so it stays; it is staged again once they are out.
                    if (exact && origin == MetalGpuWriteOrigin.StorageBuffer &&
                        entry.Address == address && entry.Size == size)
                    {
                        continue;
                    }

                    entry.StaleOnClean = true;
                    if (reason == Reason.Fault)
                    {
                        DropWatch(entry);
                    }

                    continue;
                }

                if (Interlocked.Exchange(ref entry.Stale, 1) == 0)
                {
                    Interlocked.Increment(ref _invalidations[(int)reason]);
                }

                DropWatch(entry);
            }
        }

        return found;
    }

    // Caller holds the lock. The entries whose bytes overlap [address, address + size) among those on the pages.
    private List<Entry> Collect(ulong first, ulong lastExclusive, ulong address, ulong size)
    {
        var result = new List<Entry>();
        var epoch = ++_queryEpoch;
        if (epoch == 0)
        {
            foreach (var entry in _entries.Values)
            {
                entry.QueryEpoch = 0;
            }

            epoch = ++_queryEpoch;
        }

        for (var page = first; page < lastExclusive; page++)
        {
            if (!_pages.TryGetValue(page, out var owners))
            {
                continue;
            }

            foreach (var entry in owners)
            {
                if (entry.QueryEpoch != epoch)
                {
                    entry.QueryEpoch = epoch;
                    if (entry.Overlaps(address, size))
                    {
                        result.Add(entry);
                    }
                }
            }
        }

        return result;
    }

    private bool MayHavePages(ulong first, ulong lastExclusive)
    {
        for (var page = first; page < lastExclusive; page++)
        {
            if ((Volatile.Read(ref _everPages[page / 64]) & (1L << (int)(page % 64))) != 0)
            {
                return true;
            }
        }

        return false;
    }

    private void Index(Entry entry)
    {
        _storePages.Add(entry.Address, entry.Size);
        _entries[(entry.Address, entry.Size)] = entry;
        Volatile.Write(ref _entryCount, _entries.Count);
        MetalImagePageOwners.TryGetPageRange(entry.Address, entry.Size, out var first, out var lastExclusive);
        for (var page = first; page < lastExclusive; page++)
        {
            if (!_pages.TryGetValue(page, out var owners))
            {
                _pages[page] = owners = new List<Entry>(2);
            }

            owners.Add(entry);
            Interlocked.Or(ref _everPages[page / 64], 1L << (int)(page % 64));
        }
    }

    private void Unindex(Entry entry)
    {
        _entries.Remove((entry.Address, entry.Size));
        Volatile.Write(ref _entryCount, _entries.Count);
        if (!MetalImagePageOwners.TryGetPageRange(entry.Address, entry.Size, out var first, out var lastExclusive))
        {
            return;
        }

        for (var page = first; page < lastExclusive; page++)
        {
            if (_pages.TryGetValue(page, out var owners))
            {
                owners.Remove(entry);
                if (owners.Count == 0)
                {
                    _pages.Remove(page);
                }
            }
        }
        _storePages.Remove(entry.Address, entry.Size);
    }

    // Drops an entry from the index and sends its buffers back once the GPU is done with them.
    private void RemoveEntry(Entry entry)
    {
        if (entry.Removed)
        {
            return;
        }

        entry.Removed = true;
        DropWatch(entry);
        Unindex(entry);
        RetireBuffers(entry, entry.LastUseSerial);
    }

    private void Discard(Entry entry)
    {
        if (_entries.ContainsKey((entry.Address, entry.Size)))
        {
            RemoveEntry(entry);
        }
        else
        {
            RetireBuffers(entry, entry.LastUseSerial);
        }
    }

    private void DropWatch(Entry entry)
    {
        if (entry.WatchEnd == 0)
        {
            return;
        }

        var begin = entry.WatchBegin;
        var length = entry.WatchEnd - begin;
        entry.WatchBegin = 0;
        entry.WatchEnd = 0;
        _backend.RemoveWatch(begin, length);
    }

    private void RetireBuffers(Entry entry, ulong retireAfter)
    {
        if (entry.StagingBuffer is { } staging)
        {
            _retired.Add((staging, retireAfter));
        }

        if (entry.PristineBuffer is { } pristine)
        {
            _retired.Add((pristine, retireAfter));
        }

        entry.StagingBuffer = null;
        entry.PristineBuffer = null;
        entry.Staging = default;
        entry.Pristine = default;
    }

    private void Reap()
    {
        var kept = 0;
        for (var index = 0; index < _retired.Count; index++)
        {
            var (buffer, retireAfter) = _retired[index];
            if (retireAfter <= _completedSerial)
            {
                PushFree(buffer);
            }
            else
            {
                _retired[kept++] = _retired[index];
            }
        }

        _retired.RemoveRange(kept, _retired.Count - kept);
    }

    private void PushFree(PoolBuffer buffer)
    {
        if (!_free.TryGetValue(buffer.Capacity, out var stack))
        {
            _free[buffer.Capacity] = stack = new Stack<PoolBuffer>();
        }

        stack.Push(buffer);
    }

    private static ulong Capacity(ulong size) => (size + ClassBytes - 1) / ClassBytes * ClassBytes;

    private bool TakeBuffer(ulong capacity, Entry requester, out PoolBuffer buffer)
    {
        if (_free.TryGetValue(capacity, out var stack) && stack.Count > 0)
        {
            buffer = stack.Pop();
            return true;
        }

        while (_totalBytes + capacity > _budget)
        {
            if (ReleaseOneFree())
            {
                continue;
            }

            if (!EvictOne(requester))
            {
                buffer = null!;
                return false;
            }
        }

        var created = _backend.CreateBuffer(capacity);
        if (created.Handle == 0)
        {
            buffer = null!;
            return false;
        }

        _totalBytes += capacity;
        buffer = new PoolBuffer
        {
            Handle = created.Handle,
            Contents = created.Contents,
            GpuAddress = created.GpuAddress,
            Capacity = capacity,
        };
        return true;
    }

    private bool ReleaseOneFree()
    {
        foreach (var stack in _free.Values)
        {
            if (stack.Count == 0)
            {
                continue;
            }

            ReleaseBuffer(stack.Pop());
            return true;
        }

        return false;
    }

    private void ReleaseBuffer(PoolBuffer? buffer)
    {
        if (buffer is null)
        {
            return;
        }

        _totalBytes -= buffer.Capacity;
        _backend.ReleaseBuffer(buffer.Handle);
    }

    // The least recently used entry the GPU is done with, as long as it has no stores waiting
    // to be written back. Its buffers are free at once: nothing in flight reads them.
    private bool EvictOne(Entry requester)
    {
        Entry? victim = null;
        foreach (var candidate in _entries.Values)
        {
            if (candidate == requester || candidate.Dirty || candidate.InList ||
                candidate.LastUseSerial > _completedSerial)
            {
                continue;
            }

            if (victim is null || candidate.LastUseSerial < victim.LastUseSerial)
            {
                victim = candidate;
            }
        }

        if (victim is null)
        {
            return false;
        }

        victim.Removed = true;
        DropWatch(victim);
        Unindex(victim);
        if (victim.StagingBuffer is { } staging)
        {
            PushFree(staging);
        }

        if (victim.PristineBuffer is { } pristine)
        {
            PushFree(pristine);
        }

        victim.StagingBuffer = null;
        victim.PristineBuffer = null;
        Interlocked.Increment(ref _evictions);
        return true;
    }
}

/// <summary>The device and page tracker behind the cache.</summary>
internal sealed class MetalStagedRangeBackend(MetalGuestMemory memory, SharpEmu.HLE.GpuMemory.PageGuard pages)
    : IStagedRangeBackend
{
    private static readonly nint SelRelease = MetalNative.Selector("release");

    public (nint Handle, nint Contents, ulong GpuAddress) CreateBuffer(ulong capacity)
    {
        var buffer = memory.CreateReportBuffer(capacity, "guest.staged", out var contents, out var gpuAddress);
        memory.AddResident(buffer);
        memory.CommitResidency();
        return (buffer, contents, gpuAddress);
    }

    public void ReleaseBuffer(nint handle)
    {
        memory.RemoveResident(handle);
        MetalNative.SendVoid(handle, SelRelease);
    }

    public void AddWatch(ulong address, ulong size) => pages.AddWatch(address, size, blockReads: false);

    public void RemoveWatch(ulong address, ulong size) => pages.RemoveWatch(address, size, blockReads: false);
}
