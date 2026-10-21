// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Guest memory as the GPU sees it. The guest's direct-memory backing is one contiguous
/// host alias, and <c>newBufferWithBytesNoCopy</c> windows over slices of it make every
/// vertex, index, constant, storage and indirect binding a (window, offset) pair: no guest
/// byte is copied for a range that is one contiguous run of the backing.
///
/// <para>
/// What a window costs is not its creation but its naming. A command buffer that names a
/// no-copy buffer (useResource, or a direct binding) makes the driver wire and fault in
/// every page of it for the life of the command buffer: measured on an M4, naming a 4 GiB
/// window over untouched memory for a kernel that wrote one word wired 4.3 GB and left all
/// 4.3 GB resident (zero pages the compressor then has to keep and hand back on the next
/// wire); the first wire costs ~25 ms/GiB, a resident one ~0.3 ms per submission however
/// many buffers carry it. With 1 GiB windows Demon's Souls named 8-9.5 GiB per command
/// buffer in-world for ~1.5-2.3 GiB of ranges it actually referenced, and paged constantly
/// on a 24 GB machine. A real PS5's GPU touches only the pages it accesses in unified GDDR6,
/// so the bytes wired here should track the bytes referenced.
/// </para>
///
/// <para>
/// The backing is therefore cut into windows of about a granule (<see cref="DefaultGranuleBytes"/>)
/// that do not overlap (<see cref="MetalWindowPartition"/>, which says why they must not):
/// <list type="bullet">
/// <item>a range inside one window binds it. Naming it wires about one granule;</item>
/// <item>a range that crosses a cut moves the cut out of it, so it lies in one window again;
/// no range is staged for straddling an edge in the steady state, and the windows beside the
/// cut grow or shrink by the range. When a window the command buffer being recorded already
/// used would have to change, the range is staged (read) or refused (write, which its caller
/// splits or writes another way) for this command buffer, and the cut moves as the next one
/// begins;</item>
/// <item>a range that is backed but not contiguous in the backing (pages mapped one by one
/// from scattered offsets), or that crosses a cut this command buffer may no longer move, is
/// read through a guest view: a no-copy buffer over the guest's
/// own mapping of it (MetalGuestMemory.GuestViews.cs), when nothing in the command buffer
/// being recorded writes those bytes through another address;</item>
/// <item>anything else no window covers is staged through the per-submission upload ring;
/// a binding is staged whole up to the largest window (<see cref="DefaultMaxWindowBytes"/>),
/// and one larger than that binds a window-sized prefix in place, as before.</item>
/// </list>
/// </para>
///
/// <para>
/// A window's buffer is created the first time something uses it and kept while the cut
/// stays; a buffer whose window a move or a re-cut replaced is released once the GPU has
/// completed the submission that replaced it. Windows unused for <see cref="IdleWindowSerials"/>
/// submissions are cut back to one per granule. A submission names exactly the windows its
/// own resolves touched (see <see cref="TouchedWindows"/>), plus, for an encoder that runs a
/// device-address program, the windows holding the title's reach set
/// (MetalRenderHost.DeviceAddressReach.cs).
/// </para>
/// </summary>
internal sealed partial class MetalGuestMemory : IDisposable, IMetalGranuleRuns
{
    // The granule is the size windows start at and are cut back to, and the unit the
    // device-address reach is learned in. Smaller wastes less around each referenced range
    // but makes more windows to name per encoder, more device-address table entries (a run
    // is split at every window and granule edge) and more reach feedback bits. Demon's Souls
    // in-world references ~1.5-2.3 GiB per 10 s through bindings of up to ~180 MiB, spread
    // over ~11 GiB of mapped spans: at 32 MiB a referenced range wires on average ~16 MiB of
    // neighbours (half a window each side), ~580 windows cover its 18 GiB backing, the
    // table grows by ~360 entries over the 1 GiB layout (11.2 GiB / 32 MiB, against the
    // 16384 it holds) and the reach feedback is 19 words. 64 MiB halves those counts and
    // doubles the waste per range; 16 MiB doubles them for ~8 MiB less per range.
    // SHARPEMU_METAL_V2_GRANULE_MB (a power of two, 1..1024) overrides it.
    internal const ulong DefaultGranuleBytes = 32UL * 1024 * 1024;

    // The largest a window grows to hold a range. A binding past it (Demon's Souls binds
    // whole heaps of 1.5-4 GB as raw storage) binds this much of its prefix in place, which
    // is what the 1 GiB windows gave it. SHARPEMU_METAL_V2_WINDOW_MB (512..4096), which used
    // to pick the window size, picks this cap now.
    internal const ulong DefaultMaxWindowBytes = 1UL * 1024 * 1024 * 1024;

    // Windows nothing used for this many submissions (completed) are cut back to one per
    // granule. Demon's Souls commits ~80 command buffers a second, so a cut moved for a
    // range stays where it is across ~10 s of the range not being used.
    internal const ulong IdleWindowSerials = 1024;

    // Over a chunked backing every purgeable object is its own VM map entry, and under Rosetta
    // (a 4 KiB map on the 16 KiB kernel) creating a no-copy buffer costs about the square of
    // the objects it spans: IOKit makes one memory entry per object, and each one re-simplifies
    // every map entry left in the range (vm_map_remap_extract -> vm_map_simplify_range, only
    // for sub-page maps). Measured on an M4: 2 ms for 512 objects, 30 ms for 2048, 187 ms for
    // 4096; natively 5.4 ms for 16384 (artifacts/memopt/exp-window). Windows stop growing at
    // this many objects, which at the backing's 2 MiB objects is the 1 GiB default window.
    internal const ulong MaxChunkedWindowEntries = 512;

    private static ulong ResolveGranuleBytes() =>
        ResolveGranuleBytes(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_GRANULE_MB"));

    internal static ulong ResolveGranuleBytes(string? configured)
    {
        if (ulong.TryParse(configured, out var megabytes) && megabytes >= 1 && megabytes <= 1024 &&
            System.Numerics.BitOperations.IsPow2(megabytes))
        {
            return megabytes * 1024 * 1024;
        }

        return DefaultGranuleBytes;
    }

    private static ulong ResolveMaxWindowBytes() =>
        ResolveMaxWindowBytes(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_WINDOW_MB"));

    internal static ulong ResolveMaxWindowBytes(string? configured)
    {
        if (ulong.TryParse(configured, out var megabytes) && megabytes >= 512 && megabytes <= 4096)
        {
            return megabytes * 1024 * 1024;
        }

        return DefaultMaxWindowBytes;
    }

    private const ulong MaxUploadBytes = 64UL * 1024 * 1024;

    // The largest binding staged whole: one host span holds it (SHARPEMU_METAL_V2_WINDOW_MB
    // may raise the largest window past what a span can).
    private const ulong MaxWholeStageBytes = 1UL << 30;
    private const ulong RingBufferBytes = 32UL * 1024 * 1024;
    private const ulong RingAlignment = 256;
    public const ulong GdsBytes = ManagedCommandStreamHost.GdsBytes;

    // Hazard tracking on a window makes Metal order and fence every command buffer that
    // touches it, which measured at tens of milliseconds per submission because every
    // submission touches a window. Untracked is the design's choice: the guest's own
    // barriers end the encoder, and encoders on one queue run in order.
    // SHARPEMU_METAL_V2_TRACKED=1 restores automatic tracking for comparison.
    // An MTLResidencySet would keep the windows mapped for the queue without any
    // per-encoder call — but a set holding the whole 16 GiB backing measured about seventy
    // milliseconds of scheduling latency on every submission, with the GPU itself idle:
    // Dreaming Sarah ran at five frames a second with it and at a hundred and twenty
    // without. SHARPEMU_METAL_V2_RESIDENCY_SET=1 restores the set for comparison (every
    // window joins it as it is created).
    private static readonly bool UseResidencySet = string.Equals(
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_RESIDENCY_SET"), "1", StringComparison.Ordinal);

    private static readonly bool StageEverything = string.Equals(
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_STAGE_ALL"), "1", StringComparison.Ordinal);

    private static readonly nuint HazardTrackingMode =
        string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_TRACKED"), "1", StringComparison.Ordinal)
            ? 0
            : Mtl.ResourceHazardTrackingModeUntracked;

    private static readonly nint SelNewBufferNoCopy =
        MetalNative.Selector("newBufferWithBytesNoCopy:length:options:deallocator:");
    private static readonly nint SelNewBuffer = MetalNative.Selector("newBufferWithLength:options:");
    private static readonly nint SelGpuAddress = MetalNative.Selector("gpuAddress");
    private static readonly nint SelContents = MetalNative.Selector("contents");
    private static readonly nint SelMaxBufferLength = MetalNative.Selector("maxBufferLength");
    private static readonly nint SelSetLabel = MetalNative.Selector("setLabel:");
    private static readonly nint SelRelease = MetalNative.Selector("release");
    private static readonly nint SelAddAllocation = MetalNative.Selector("addAllocation:");
    private static readonly nint SelCommit = MetalNative.Selector("commit");

    // A ring buffer of host-written bytes: valid until the submission that used it completes.
    private sealed class RingBuffer
    {
        public nint Buffer;
        public ulong GpuAddress;
        public nint Contents;
        public ulong Size;
        public ulong Cursor;
        public ulong LastUsedSerial;
        public long LastUsedAt;

        // The submission that last used it took it while an earlier one still held it.
        public bool Inherited;
    }

    private readonly nint _device;
    private readonly IGuestBackedSpace _backing;
    private readonly ICpuMemory _memory;
    private readonly ulong _aliasBase;
    private readonly ulong _aliasSize;
    private readonly int _granuleBits;
    private readonly ulong _granuleBytes;
    private readonly int _granuleCount;
    private readonly ulong _maxWindowBytes;

    // The cut of the backing into windows, and everything that reads or moves it, under
    // _windowLock. Buffers of windows a move replaced wait in _retired for the submission
    // that replaced them to complete; ranges a move had to wait for wait in _pendingPlacements
    // for the next submission.
    private readonly MetalWindowPartition _partition;
    private readonly object _windowLock = new();
    private readonly System.Collections.Concurrent.ConcurrentDictionary<nint, byte> _windowBuffers = new();
    private readonly List<(MetalWindowPartition.Window Window, ulong Serial)> _retired = new();
    private readonly List<MetalWindowPartition.Window> _retiredScratch = new();
    private readonly List<(ulong Start, ulong End)> _pendingPlacements = new();
    private readonly HashSet<(ulong Start, ulong End)> _pendingSet = new();
    private long _windowsCreated;
    private long _windowsReleased;
    private long _windowBytesMapped;
    private long _windowMoves;
    private long _windowConflicts;
    private ulong _normalizeSerial;

    // The largest range the ring stages whole; a bigger one binds a prefix instead.
    private readonly ulong _maxUploadBytes;
    private long _oversizedInPlace;
    private long _oversizedStaged;
    private readonly List<RingBuffer> _rings = new();
    private readonly nint _residencySet;
    private readonly nint _gdsBuffer;
    private readonly nint _nullBuffer;
    private readonly nint _nullContents;
    private ulong _submissionSerial;
    private ulong _completedSerial;
    private long _ringFallbackBytes;
    private long _ringBytes;
    private long _ringFallbacks;
    private bool _disposed;

    public MetalGuestMemory(nint device, IGuestBackedSpace backing, ICpuMemory memory)
        : this(device, backing, memory, ResolveGranuleBytes(), ResolveMaxWindowBytes())
    {
    }

    /// <summary>A layout with explicit granule and window cap, for tests on a small backing.</summary>
    internal MetalGuestMemory(
        nint device,
        IGuestBackedSpace backing,
        ICpuMemory memory,
        ulong granuleBytes,
        ulong maxWindowBytes,
        ulong maxUploadBytes = MaxUploadBytes,
        bool? singleCopy = null)
    {
        _device = device;
        _maxUploadBytes = maxUploadBytes;
        _backing = backing;
        _memory = memory;
        var (aliasBase, aliasSize) = backing.BackingAliasRange;
        if (aliasBase == 0 || aliasSize == 0)
        {
            throw GpuFatal.Fatal("Metal v2 needs the guest backing alias; the address space reports none.");
        }

        // Granules start at multiples of the granule from the alias base; a granule of at
        // least a 16 KB host page keeps every window page-aligned with its host pointer.
        if (!System.Numerics.BitOperations.IsPow2(granuleBytes) || granuleBytes < PairLockPageBytes)
        {
            throw GpuFatal.Fatal($"The guest window granule must be a power of two of at least 16 KB: 0x{granuleBytes:X}.");
        }

        _aliasBase = aliasBase;
        _aliasSize = aliasSize;
        _granuleBits = System.Numerics.BitOperations.TrailingZeroCount(granuleBytes);
        _granuleBytes = granuleBytes;
        _granuleCount = checked((int)((aliasSize + granuleBytes - 1) >> _granuleBits));
        var maxBufferLength = (ulong)MetalV2Native.SendValue(device, SelMaxBufferLength);
        var cap = maxBufferLength == 0 ? maxWindowBytes : Math.Min(maxWindowBytes, maxBufferLength);
        var chunkBytes = backing.BackingChunkBytes;
        var objectBytes = backing.BackingObjectBytes;
        if (objectBytes != 0 && MaxChunkedWindowEntries <= ulong.MaxValue / objectBytes)
        {
            cap = Math.Min(cap, MaxChunkedWindowEntries * objectBytes);
        }

        // Whole granules, and never less than one.
        _maxWindowBytes = Math.Max(granuleBytes, cap & ~(granuleBytes - 1));
        _partition = new MetalWindowPartition(aliasSize, granuleBytes, _maxWindowBytes, PairLockPageBytes);
        _touchedWindows = new MetalWindowSet(_partition.IdLimit + 64);
        _pinnedWindows = new MetalWindowSet(_partition.IdLimit + 64);

        InitializeSingleCopyWindows(backing, singleCopy);
        _residencySet = UseResidencySet ? CreateResidencySet(device) : 0;
        _gdsBuffer = CreateSharedBuffer(GdsBytes, "guest.gds", out _, out _);
        _nullBuffer = CreateSharedBuffer(256, "guest.null", out _nullContents, out _);
        // Both are reached through an address in an argument buffer, so the encoder
        // has to be told about them; both are small enough to always name.
        AddResident(_gdsBuffer);
        AddResident(_nullBuffer);
        unsafe
        {
            new Span<byte>((void*)_nullContents, 256).Clear();
        }

        CommitResidency();
        Console.Error.WriteLine(
            $"[METAL-V2][INFO] guest memory windows: granule=0x{granuleBytes:X} granules={_granuleCount} " +
            $"max_window=0x{_maxWindowBytes:X} backing_chunk=0x{chunkBytes:X} backing_object=0x{objectBytes:X} alias=0x{aliasBase:X16}+0x{aliasSize:X} " +
            $"residency_set={(_residencySet != 0 ? 1 : 0)} gpu_alias={(_gpuAlias is { } gpuAlias ? $"0x{gpuAlias.Base:X16}" : "none")} " +
            "(windows are created on first use)");
    }

    /// <summary>
    /// Records every window resolution while the slot trap is armed. The tiler and the
    /// transfer paths bind their buffers straight to an encoder rather than through a
    /// descriptor, so a submission that writes guest memory can resolve a window without
    /// any binding the descriptor path ever sees.
    /// </summary>
    internal MetalSlotTrap? Trap { get; set; }

    public nint Device => _device;

    public nint GdsBuffer => _gdsBuffer;

    public nint NullBuffer => _nullBuffer;

    public nint ResidencySet => _residencySet;

    public ulong RingFallbackBytes => (ulong)Volatile.Read(ref _ringFallbackBytes);

    public ulong RingFallbacks => (ulong)Volatile.Read(ref _ringFallbacks);

    /// <summary>How many ranges too large to stage bound a window prefix in place instead.</summary>
    public ulong OversizedInPlace => (ulong)Volatile.Read(ref _oversizedInPlace);

    /// <summary>How many bindings past the ring's cap were staged whole rather than cut to a prefix.</summary>
    public ulong OversizedStaged => (ulong)Volatile.Read(ref _oversizedStaged);

    // Shared memory the upload rings hold now, for the periodic counters line.
    public ulong RingBytes => (ulong)Volatile.Read(ref _ringBytes);

    /// <summary>The granule size: the unit a binding wires.</summary>
    public ulong GranuleBytes => _granuleBytes;

    /// <summary>How many granules cover the backing: the units the reach set is learned in.</summary>
    public int GranuleCount => _granuleCount;

    /// <summary>The largest window one binding may name.</summary>
    public ulong MaxWindowBytes => _maxWindowBytes;

    /// <summary>The backing alias size, which the granules cover.</summary>
    public ulong AliasBytes => _aliasSize;

    /// <summary>Window buffers alive now, created and released since startup, and the bytes the alive ones map.</summary>
    public (int Alive, long Created, long Released, ulong MappedBytes) WindowCounts
    {
        get
        {
            lock (_windowLock)
            {
                return ((int)(_windowsCreated - _windowsReleased), _windowsCreated, _windowsReleased,
                    (ulong)_windowBytesMapped);
            }
        }
    }

    /// <summary>
    /// Cuts moved for a range that crossed one, ranges that had to be staged or refused for
    /// one submission because the move would have changed a window it already used, and the
    /// windows the backing is cut into now.
    /// </summary>
    public (long Moves, long Conflicts, int Windows) WindowLayoutCounts
    {
        get
        {
            lock (_windowLock)
            {
                return (_windowMoves, _windowConflicts, _partition.Windows.Count);
            }
        }
    }

    /// <summary>Changes whenever a cut moves: what a table of window addresses is valid for.</summary>
    public long PartitionVersion
    {
        get
        {
            lock (_windowLock)
            {
                return _partition.Version;
            }
        }
    }

    /// <summary>
    /// Shared memory the rings handed to the submission being recorded hold. A ring
    /// returns to the pool only when the submission that last used it completes, so a
    /// submission that keeps allocating holds every ring it touched and grows the pool
    /// by a ring at a time; the host commits it at a budget. A ring the submission takes
    /// from the pool (new, or free again) counts whole; one an earlier submission still
    /// holds counts only what this one reserves in it, since the pool keeps that ring
    /// whatever this one does. Counting such a ring whole again was a commit storm: once
    /// bursts had left the pool a few 128 MB rings, a submission's first small uploads
    /// landed in two the previous one still held and passed the 256 MB budget, so every
    /// command buffer was committed after a handful of uploads (Black Myth: Wukong, in a
    /// run of five: ~4000 commits a 2 s window instead of ~40, each ending the render pass,
    /// every other one waiting for the GPU, and 4.5 fps falling to 2).
    /// </summary>
    public ulong RecordingBytes { get; private set; }

    /// <summary>
    /// The serial a resource allocated now is retired at. A new command buffer has used no
    /// window yet, so this is where cuts that had to wait move, and where idle windows are
    /// cut back to granules.
    /// </summary>
    public void BeginSubmission(ulong submissionSerial, ulong completedSerial)
    {
        _submissionSerial = submissionSerial;
        _completedSerial = Math.Max(_completedSerial, completedSerial);
        RecordingBytes = 0;
        TrimIdleRings();
        // Residency is per submission: a window is named by the submissions that resolve a
        // range through it, never by every submission after.
        _touchedWindows.Clear();
        _residentDirty = true;
        lock (_windowLock)
        {
            _pinnedWindows.Clear();
            foreach (var (start, end) in _pendingPlacements)
            {
                if (_partition.Place(start, end, _pinnedWindows, _retiredScratch, out _) == MetalWindowPartition.Placement.Moved)
                {
                    _windowMoves++;
                }
            }

            _pendingPlacements.Clear();
            _pendingSet.Clear();
            if (_completedSerial >= IdleWindowSerials && _submissionSerial - _normalizeSerial >= IdleWindowSerials / 8)
            {
                _normalizeSerial = _submissionSerial;
                _partition.Normalize(_completedSerial - IdleWindowSerials, _pinnedWindows, _retiredScratch);
            }

            ApplySingleCopyHolesLocked();
            RetireReplaced();
        }

        DropSingleCopyGuestViews();
        BeginGuestViewSubmission();
    }

    private readonly MetalWindowSet _touchedWindows;

    // The windows this submission reached in any way - its resolves, and the windows a
    // device-address encoder names - whose cuts may not move before it ends.
    private readonly MetalWindowSet _pinnedWindows;
    private bool _residentDirty = true;

    /// <summary>The buffer of one window, created if it has none; zero when no window has the id.</summary>
    public nint WindowBuffer(int id)
    {
        lock (_windowLock)
        {
            return _partition.ById(id) is { } window ? EnsureBuffer(window) : 0;
        }
    }

    /// <summary>The windows resolved since this submission began, one bit per window id.</summary>
    public MetalWindowSet TouchedWindows => _touchedWindows;

    /// <summary>
    /// Notes that this submission names these windows for a device-address program: their
    /// cuts stay until it ends, and they count as used.
    /// </summary>
    public void PinWindows(MetalWindowSet windows)
    {
        lock (_windowLock)
        {
            _pinnedWindows.UnionWith(windows);
            foreach (var id in windows)
            {
                if (_partition.ById(id) is { } window)
                {
                    window.LastUsedSerial = _submissionSerial;
                }
            }
        }
    }

    /// <summary>
    /// Adds the granules the windows of <paramref name="windowIds"/> overlap to
    /// <paramref name="granules"/>.
    /// </summary>
    public void AddCoveredGranules(MetalWindowSet windowIds, MetalWindowSet granules)
    {
        lock (_windowLock)
        {
            foreach (var id in windowIds)
            {
                if (_partition.ById(id) is { } window)
                {
                    for (var granule = window.Start >> _granuleBits; granule <= (window.End - 1) >> _granuleBits; granule++)
                    {
                        granules.Add((int)granule);
                    }
                }
            }
        }
    }

    /// <summary>
    /// The bytes the windows of a set map: what naming them wires. Windows do not overlap,
    /// so nothing is counted twice.
    /// </summary>
    public ulong WindowSetBytes(MetalWindowSet windowIds)
    {
        var bytes = 0UL;
        ForEachWindow(windowIds, (_, size) => bytes += size);
        return bytes;
    }

    /// <summary>The backing offset each window of a set starts at, and its size.</summary>
    public void ForEachWindow(MetalWindowSet windowIds, Action<ulong, ulong> visit)
    {
        lock (_windowLock)
        {
            foreach (var id in windowIds)
            {
                if (_partition.ById(id) is { } window)
                {
                    visit(window.Start, window.Size);
                }
            }
        }
    }

    /// <summary>The bytes of the backing the granules of the set cover.</summary>
    public ulong GranuleSetBytes(MetalWindowSet granules)
    {
        var bytes = 0UL;
        foreach (var granule in granules)
        {
            if (granule < _granuleCount)
            {
                bytes += GranuleBacking(granule).Size;
            }
        }

        return bytes;
    }

    /// <summary>The backing a granule covers: an offset into the alias and its length.</summary>
    public (ulong Offset, ulong Size) GranuleBacking(int granule)
    {
        var offset = (ulong)granule << _granuleBits;
        return (offset, Math.Min(_granuleBytes, _aliasSize - offset));
    }

    /// <summary>The granule a backing offset lies in.</summary>
    public int GranuleOfBacking(ulong offset) =>
        (int)Math.Min(offset >> _granuleBits, (ulong)(_granuleCount - 1));

    public void NoteCompleted(ulong completedSerial)
    {
        _completedSerial = Math.Max(_completedSerial, completedSerial);
        TrimIdleRings();
        lock (_windowLock)
        {
            ReleaseRetired();
        }

        ReleaseRetiredViews();
    }

    // A ring is created when a range is not one backed run, which a title can do in
    // bursts: a 4 GiB span of unbacked descriptors once cost tens of rings. Nothing freed
    // them again, so the pool was the process's high-water mark for the rest of the run.
    // A ring the GPU has been done with for this many serials or milliseconds is released,
    // except a few standard-size ones that keep a steady submission fed. Trimming ran only
    // at host waits and kept the first rings whatever their size, so a load's oversized
    // rings could stay; it now also runs at every new submission and keeps by size.
    private const int KeptRings = 4;
    private const ulong IdleRingSerials = 64;
    private const long IdleRingMilliseconds = 1000;
    private const long RingTrimIntervalMilliseconds = 50;
    private ulong _ringsTrimmedSerial = ulong.MaxValue;
    private long _ringsTrimmedAt;
    private long _ringPeakBytes;
    private long _ringWindowPeakBytes;

    /// <summary>Milliseconds, monotonic; tests replace it to age rings.</summary>
    internal Func<long> RingClock { get; set; } = static () => Environment.TickCount64;

    private void TrimIdleRings()
    {
        var now = RingClock();
        if (_completedSerial == _ringsTrimmedSerial && now - _ringsTrimmedAt < RingTrimIntervalMilliseconds)
        {
            return;
        }

        _ringsTrimmedSerial = _completedSerial;
        _ringsTrimmedAt = now;
        var kept = 0;
        for (var index = 0; index < _rings.Count; index++)
        {
            var ring = _rings[index];
            if (ring.LastUsedSerial > _completedSerial)
            {
                continue;
            }

            var idle = _completedSerial - ring.LastUsedSerial >= IdleRingSerials ||
                now - ring.LastUsedAt >= IdleRingMilliseconds;
            if (!idle)
            {
                continue;
            }

            if (ring.Size <= RingBufferBytes && kept < KeptRings)
            {
                kept++;
                continue;
            }

            RemoveResident(ring.Buffer);
            MetalNative.SendVoid(ring.Buffer, SelRelease);
            _rings.RemoveAt(index--);
            Interlocked.Add(ref _ringBytes, -(long)ring.Size);
            _ringsReleased++;
        }
    }

    /// <summary>The most shared memory the upload rings have held since startup.</summary>
    public ulong RingPeakBytes => (ulong)Volatile.Read(ref _ringPeakBytes);

    /// <summary>The most shared memory the upload rings have held since the last call.</summary>
    public ulong TakeWindowPeakRingBytes()
    {
        var now = Volatile.Read(ref _ringBytes);
        return (ulong)Math.Max(Interlocked.Exchange(ref _ringWindowPeakBytes, now), now);
    }

    private long _ringsReleased;

    // Caller holds _windowLock. Windows a move or a re-cut replaced keep their buffers until
    // the submission being recorded completes: earlier submissions still in flight may bind
    // them, and this one's device-address table may hold their addresses.
    private void RetireReplaced()
    {
        foreach (var window in _retiredScratch)
        {
            if (window.Buffer != 0)
            {
                _retired.Add((window, _submissionSerial));
            }
        }

        _retiredScratch.Clear();
    }

    // Caller holds _windowLock.
    private void ReleaseRetired()
    {
        for (var index = _retired.Count - 1; index >= 0; index--)
        {
            var (window, serial) = _retired[index];
            if (serial > _completedSerial)
            {
                continue;
            }

            ReleaseBuffer(window);
            _retired.RemoveAt(index);
        }
    }

    // Caller holds _windowLock.
    private void ReleaseBuffer(MetalWindowPartition.Window window)
    {
        _windowBuffers.TryRemove(window.Buffer, out _);
        if (_residencySet != 0)
        {
            RemoveResident(window.Buffer);
        }

        MetalNative.SendVoid(window.Buffer, SelRelease);
        window.Buffer = 0;
        _windowsReleased++;
        _windowBytesMapped -= (long)window.Size;
    }

    /// <summary>How many idle upload rings have been handed back since startup.</summary>
    public long RingsReleased => Volatile.Read(ref _ringsReleased);

    /// <summary>The upload rings alive right now.</summary>
    public int RingCount => _rings.Count;

    /// <summary>
    /// The GPU view of a guest byte range. A range that is one backed run binds its window
    /// in place; anything else is staged into the upload ring, so the bytes the GPU reads
    /// are the bytes the CPU can see right now. A range past the ring's cap is staged whole
    /// when <paramref name="whole"/> asks for it (a binding, which a prefix would cut short),
    /// otherwise only a prefix is handed back, which the caller has to check.
    /// </summary>
    public MetalGuestRange Resolve(ulong gpuVa, ulong size, bool whole = false)
    {
        if (size == 0 || gpuVa == 0 || size > ulong.MaxValue - gpuVa)
        {
            return default;
        }

        if (!StageEverything && TryResolveWindow(gpuVa, size, out var resolved))
        {
            MetalGpuReferenceCensus.Mark(gpuVa, size);
            return resolved;
        }

        return Upload(gpuVa, size, whole);
    }

    /// <summary>
    /// The window a binding of the range reads and writes in place; false when no one
    /// window covers it and it has to be staged (<see cref="Stage"/>).
    /// </summary>
    public bool TryResolveInPlace(ulong gpuVa, ulong size, out MetalGuestRange range)
    {
        range = default;
        if (StageEverything || !TryResolveWindow(gpuVa, size, out range))
        {
            return false;
        }

        MetalGpuReferenceCensus.Mark(gpuVa, size);
        return true;
    }

    /// <summary>True when the range is a window's own bytes - guest memory - rather than a ring copy.</summary>
    public bool IsInPlace(in MetalGuestRange range)
    {
        if (!range.IsValid || range.Buffer == 0)
        {
            return false;
        }

        foreach (var ring in _rings)
        {
            if (ring.Buffer == range.Buffer)
            {
                return false;
            }
        }

        return _windowBuffers.ContainsKey(range.Buffer);
    }

    /// <summary>A ring copy of the guest bytes the CPU sees now, whole up to the largest window.</summary>
    public MetalGuestRange Stage(ulong gpuVa, ulong size) =>
        size == 0 || gpuVa == 0 || size > ulong.MaxValue - gpuVa ? default : Upload(gpuVa, size, whole: true);

    /// <summary>
    /// True when the whole range lives in the backing. A managed write to such a range
    /// lands through the backing alias, never through the guest mapping a page watch
    /// protects.
    /// </summary>
    public bool IsBacked(ulong gpuVa, ulong size) =>
        size != 0 && size <= ulong.MaxValue - gpuVa && _backing.TryResolveBackingAliasSpan(gpuVa, size, out _);

    /// <summary>The window range of an address that must be written by the GPU; zero when it is not backed.</summary>
    public MetalGuestRange ResolveWritable(ulong gpuVa, ulong size)
    {
        if (!TryResolveWindow(gpuVa, size, out var resolved))
        {
            return default;
        }

        MetalGpuReferenceCensus.Mark(gpuVa, size);
        return resolved;
    }

    /// <summary>
    /// The longest prefix of a range that one window can cover, or an invalid range when
    /// not even the first byte is backed: the longest prefix that is one contiguous run of
    /// the backing, up to the largest window from its first granule. That is how an image
    /// too scattered to map in one piece is cut into the pieces that do map.
    /// </summary>
    public MetalGuestRange ResolveWritablePrefix(ulong gpuVa, ulong size)
    {
        var prefix = ResolveWindowPrefix(gpuVa, size, minimum: 1);
        if (prefix.IsValid)
        {
            MetalGpuReferenceCensus.Mark(gpuVa, prefix.Size);
        }

        return prefix;
    }

    // A prefix shorter than the minimum is not bound at all, so a caller that would stage
    // it instead does not name a window for nothing.
    private MetalGuestRange ResolveWindowPrefix(ulong gpuVa, ulong size, ulong minimum)
    {
        if (size == 0 || gpuVa == 0 || size > ulong.MaxValue - gpuVa)
        {
            return default;
        }

        if (TryResolveWindow(gpuVa, size, out var whole, warn: false))
        {
            return whole;
        }

        if (!_backing.TryResolveBackingAliasSpan(gpuVa, 1, out var alias) || !InAlias(alias, 1))
        {
            return default;
        }

        var offset = alias - _aliasBase;
        if (minimum > 1)
        {
            // A prefix that has to be large (one too large to stage) may move cuts to fit,
            // up to the largest window.
            var reach = Math.Min(size, Math.Min(_maxWindowBytes, _aliasSize - offset));
            var capped = ContiguousBackedPrefix(gpuVa, reach, alignment: 1);
            if (capped >= minimum && capped < size && TryResolveWindow(gpuVa, capped, out var large, warn: false))
            {
                return large;
            }
        }

        // Otherwise what the window holding the first byte holds, as the backing is cut now.
        ulong windowEnd;
        lock (_windowLock)
        {
            windowEnd = _partition.At(offset).End;
        }

        var length = ContiguousBackedPrefix(gpuVa, Math.Min(size, windowEnd - offset), alignment: 1);
        return length != 0 && length >= minimum && TryResolveWindow(gpuVa, length, out var prefix, warn: false)
            ? prefix
            : default;
    }

    /// <summary>
    /// The longest prefix of a range, in multiples of <paramref name="alignment"/>, that is
    /// one contiguous run of the backing. Whether a range resolves is monotone in its length,
    /// so it is a binary search; nothing is bound.
    /// </summary>
    private ulong ContiguousBackedPrefix(ulong gpuVa, ulong size, ulong alignment)
    {
        if (_backing.TryResolveBackingAliasSpan(gpuVa, size, out _))
        {
            return size;
        }

        var good = 0UL;
        var bad = size;
        while (bad - good > alignment)
        {
            var middle = (good + ((bad - good) / 2)) & ~(alignment - 1);
            if (middle <= good)
            {
                break;
            }

            if (_backing.TryResolveBackingAliasSpan(gpuVa, middle, out _))
            {
                good = middle;
            }
            else
            {
                bad = middle;
            }
        }

        return good;
    }

    /// <summary>
    /// For the device-address table: the longest run from <paramref name="gpuVa"/>, in whole
    /// pages and at most <paramref name="size"/>, that is contiguous in the backing and stays
    /// inside the granule its first byte lies in, with the backing offset of its first byte.
    /// The table cuts it further at window edges (<see cref="WindowAt"/>). Nothing is marked
    /// touched: the table is not a binding, and what a device-address encoder names is
    /// decided by the reach set.
    /// </summary>
    public bool TryResolveGranuleRun(
        ulong gpuVa, ulong size, ulong page, out ulong backingOffset, out ulong length, out int granule)
    {
        backingOffset = 0;
        length = 0;
        granule = -1;
        if (size < page || !_backing.TryResolveBackingAliasSpan(gpuVa, page, out var alias) || !InAlias(alias, page))
        {
            return false;
        }

        backingOffset = alias - _aliasBase;
        granule = (int)(backingOffset >> _granuleBits);
        var (granuleOffset, granuleSize) = GranuleBacking(granule);
        var clipped = Math.Min(size, granuleOffset + granuleSize - backingOffset) & ~(page - 1);
        length = clipped == 0 ? 0 : ContiguousBackedPrefix(gpuVa, clipped, page);
        return length != 0;
    }

    /// <summary>
    /// The window holding a backing offset as the backing is cut now: its id, the offset it
    /// ends at, and the GPU address of the offset in it. Its buffer is created if it has none.
    /// </summary>
    public (int Id, ulong End, ulong GpuAddress) WindowAt(ulong backingOffset)
    {
        lock (_windowLock)
        {
            var window = _partition.At(backingOffset);
            EnsureBuffer(window);
            return (window.Id, window.End, window.GpuAddress + (backingOffset - window.Start));
        }
    }

    // Caller holds _windowLock.
    private nint EnsureBuffer(MetalWindowPartition.Window window)
    {
        if (window.Buffer != 0)
        {
            return window.Buffer;
        }

        // Over the GPU alias when single-copy textures are on: released chunks are holes there.
        var pointer = _windowBase + window.Start;
        var buffer = MetalV2Native.SendNewBufferNoCopy(
            _device,
            SelNewBufferNoCopy,
            (nint)pointer,
            (nuint)window.Size,
            Mtl.ResourceStorageModeShared | HazardTrackingMode,
            0);
        if (buffer == 0)
        {
            throw GpuFatal.Fatal(
                $"A guest memory window could not be mapped: window={window.Id} backing=0x{window.Start:X}+0x{window.Size:X} " +
                $"pointer=0x{pointer:X16}.");
        }

        SetLabel(buffer, $"guest.window[0x{window.Start:X}+0x{window.Size:X}]");
        window.GpuAddress = MetalNative.SendGpuAddress(buffer, SelGpuAddress);
        // A guest qword's pair lock is picked by bits 3..13 of its GPU address
        // (pair_atomic_prelude.msl). Windows do not overlap, but a qword lives in different
        // windows over time as cuts move, so each must keep the host mapping's offset within
        // a 16 KB page for the lock to stay the same.
        if (((window.GpuAddress - pointer) & (PairLockPageBytes - 1)) != 0)
        {
            MetalV2Diagnostics.WarnOnce(
                "pair-lock-window-alignment",
                "A guest memory window's GPU address is not 16 KB congruent with its host pointer, so a qword " +
                "takes a different pair lock once a cut moves it into another window: " +
                $"window={window.Id} pointer=0x{pointer:X16} gpu=0x{window.GpuAddress:X16}.");
        }

        _windowBuffers[buffer] = 0;
        window.Buffer = buffer;
        _windowsCreated++;
        _windowBytesMapped += (long)window.Size;
        if (_residencySet != 0)
        {
            AddResident(buffer);
            CommitResidency();
        }

        return buffer;
    }

    private bool InAlias(ulong alias, ulong size) =>
        alias >= _aliasBase && alias - _aliasBase < _aliasSize && size <= _aliasSize - (alias - _aliasBase);

    /// <summary>
    /// Why a range has no window, for diagnostics. A caller that has to fall back wants to
    /// say whether the guest bytes are not one backed span at all or merely exceed the
    /// largest window, because the two have different answers.
    /// </summary>
    public string DescribeResolve(ulong gpuVa, ulong size)
    {
        if (size == 0 || gpuVa == 0 || size > ulong.MaxValue - gpuVa)
        {
            return "degenerate range";
        }

        if (!_backing.TryResolveBackingAlias(gpuVa, size, out var alias))
        {
            // Narrow down to the largest prefix that is one backed span, which says
            // whether the range is unbacked outright or merely not contiguous.
            var contiguous = 0UL;
            for (var probe = size / 2; probe >= 4096; probe /= 2)
            {
                if (_backing.TryResolveBackingAlias(gpuVa, contiguous + probe, out _))
                {
                    contiguous += probe;
                }
            }

            return $"not one backed span (contiguous prefix about 0x{contiguous:X} of 0x{size:X})";
        }

        if (!InAlias(alias, size))
        {
            return $"backed at alias 0x{alias:X16}, outside the mapped alias " +
                $"0x{_aliasBase:X16}+0x{_aliasSize:X}";
        }

        var offset = alias - _aliasBase;
        if (size > _maxWindowBytes)
        {
            return $"backed at offset 0x{offset:X}, larger than the largest window 0x{_maxWindowBytes:X}";
        }

        lock (_windowLock)
        {
            var window = _partition.At(offset);
            return offset + size <= window.End
                ? "resolvable"
                : $"resolvable once the window edge at 0x{window.End:X} moves out of it";
        }
    }

    // Why a range had to be staged into the ring instead of binding its window: the
    // start, the end, or the span between them. A sample per size class is enough to
    // name the allocation, and the ring bytes in the counters say how much it costs.
    private void ReportUnresolved(ulong gpuVa, ulong size)
    {
        if (!MetalV2Diagnostics.ReportUnbackedRanges)
        {
            return;
        }

        var startBacked = _backing.TryResolveBackingAlias(gpuVa, 1, out var startAlias);
        var endBacked = _backing.TryResolveBackingAlias(gpuVa + size - 1, 1, out var endAlias);
        var contiguous = startBacked && endBacked && endAlias - startAlias == size - 1;
        MetalV2Diagnostics.WarnOnce(
            $"unbacked-range:{size:X}:{(startBacked ? 1 : 0)}{(endBacked ? 1 : 0)}{(contiguous ? 1 : 0)}",
            $"A guest range does not resolve to a window and is staged into the ring: " +
            $"address=0x{gpuVa:X16} size=0x{size:X} start_backed={startBacked} end_backed={endBacked} " +
            $"alias_contiguous={contiguous} start_alias=0x{startAlias:X16} end_alias=0x{endAlias:X16}.");
    }

    private bool TryResolveWindow(ulong gpuVa, ulong size, out MetalGuestRange resolved, bool warn = true)
    {
        using var phaseScope = SharpEmu.Libs.VideoOut.RenderPhaseProfile.MeasureDetail(SharpEmu.Libs.VideoOut.RenderPhaseProfile.Phase.GuestWindowResolve);
        resolved = default;
        if (size == 0 || gpuVa == 0 || size > ulong.MaxValue - gpuVa)
        {
            return false;
        }

        // A range that crosses the edge between two mappings of one allocation is
        // still contiguous in the backing, so it still has one window address; the
        // single-mapping test alone sent those to the staging ring.
        if (!_backing.TryResolveBackingAliasSpan(gpuVa, size, out var alias) || !InAlias(alias, size))
        {
            if (warn)
            {
                ReportUnresolved(gpuVa, size);
            }

            return false;
        }

        var offset = alias - _aliasBase;
        // A released chunk gets its bytes back before any window names it (single-copy textures).
        RestoreSingleCopyRange(offset, size);
        MetalWindowPartition.Window window;
        lock (_windowLock)
        {
            FillSingleCopyHolesLocked(offset, size);
            switch (_partition.Place(offset, offset + size, _pinnedWindows, _retiredScratch, out window))
            {
                case MetalWindowPartition.Placement.Moved:
                    _windowMoves++;
                    RetireReplaced();
                    break;
                case MetalWindowPartition.Placement.Pinned:
                    // The cut has to move but this submission already used a window it would
                    // change: give it no window now, move the cut as the next one begins.
                    _windowConflicts++;
                    if (_pendingSet.Add((offset, offset + size)))
                    {
                        _pendingPlacements.Add((offset, offset + size));
                    }

                    MetalV2Diagnostics.InfoOnce(
                        "window-move-deferred",
                        "A guest range crosses a window edge whose windows this command buffer already used; it has " +
                        "no window until the next command buffer moves the edge (a byte may have one GPU address " +
                        $"per command buffer): address=0x{gpuVa:X16} size=0x{size:X}.");
                    return false;
                case MetalWindowPartition.Placement.TooLarge:
                    if (warn)
                    {
                        MetalV2Diagnostics.WarnOnce(
                            $"window-span:{size:X}",
                            $"A backed guest range is larger than the largest window can grow to hold it: " +
                            $"address=0x{gpuVa:X16} size=0x{size:X} alias=0x{alias:X16} " +
                            $"max_window=0x{_maxWindowBytes:X}.");
                    }

                    return false;
            }

            if (!TryUseSingleCopyWindowLocked(ref window, offset, size))
            {
                return false;
            }

            EnsureBuffer(window);
            window.LastUsedSerial = _submissionSerial;
            _pinnedWindows.Add(window.Id);
        }

        if (_touchedWindows.Add(window.Id))
        {
            _residentDirty = true;
        }

        var windowOffset = offset - window.Start;
        resolved = new MetalGuestRange(window.Buffer, windowOffset, size, window.GpuAddress + windowOffset);
        // Mappings are page-granular on both sides, so a page-aligned guest address has to
        // resolve to a page-aligned window offset. When it does not, some view record's
        // guest address and backing offset disagree below the page, and every range it
        // resolves lands that far from where the guest put it.
        if ((gpuVa & 0xFFF) != (windowOffset & 0xFFF))
        {
            MetalV2Diagnostics.WarnOnce(
                "unaligned-window-offset",
                $"A guest address resolved to a window offset that disagrees with it below the page: " +
                $"address=0x{gpuVa:X16} offset=0x{windowOffset:X} window={window.Id} delta=0x{(gpuVa ^ windowOffset) & 0xFFF:X}.");
        }

        Trap?.NoteBinding($"resolve[{window.Id}]", gpuVa, size, (ulong)window.Buffer, windowOffset, resolved.GpuAddress);
        return true;
    }

    /// <summary>Copies guest bytes into the ring; used for ranges no window covers.</summary>
    private MetalGuestRange Upload(ulong gpuVa, ulong size, bool whole)
    {
        if (size > _maxUploadBytes && whole && size <= Math.Min(_maxWindowBytes, MaxWholeStageBytes))
        {
            // A binding no window covers whole, because its pages are scattered over the
            // backing or a cut this command buffer already used crosses it, is staged whole:
            // a prefix would hand the shader fewer bytes than the guest bound, and past it
            // loads read zero and stores are dropped. One window could hold it, so it is not
            // a whole heap bound as raw storage (those still bind a prefix, below).
            Interlocked.Increment(ref _oversizedStaged);
            MetalV2Diagnostics.InfoOnce(
                $"oversized-staged:{size:X}",
                $"A guest range no single window covers is larger than the ring's cap and is staged whole: " +
                $"address=0x{gpuVa:X16} size=0x{size:X} ({DescribeResolve(gpuVa, size)}).");
        }
        else if (size > _maxUploadBytes)
        {
            // A range this large that no window covers is not data a draw reads whole;
            // bind its prefix rather than staging gigabytes per submission. When its
            // contiguous start holds at least a ring's worth, up to the largest window,
            // that prefix binds in place: it stays live with the GPU writes still in
            // flight and a shader's stores through it reach guest memory, neither of
            // which a staged copy of the first bytes can do. Demon's Souls binds whole
            // heaps (1.5-4 GB) this way.
            var prefix = ResolveWindowPrefix(gpuVa, size, minimum: _maxUploadBytes);
            if (prefix.IsValid && prefix.Size >= _maxUploadBytes)
            {
                MetalGpuReferenceCensus.Mark(gpuVa, prefix.Size);
                Interlocked.Increment(ref _oversizedInPlace);
                MetalV2Diagnostics.WarnOnce(
                    $"oversized-in-place:{size:X}",
                    $"A guest range no single window covers is too large to stage; binding its first " +
                    $"0x{prefix.Size:X} bytes in place: address=0x{gpuVa:X16} size=0x{size:X} " +
                    $"({DescribeResolve(gpuVa, size)}).");
                return prefix;
            }

            // It may well be backed and merely exceed the largest window, so the warning says which.
            MetalV2Diagnostics.WarnOnce(
                "oversized-upload",
                $"A guest range no single window covers is too large to stage; binding its first " +
                $"0x{_maxUploadBytes:X} bytes: address=0x{gpuVa:X16} size=0x{size:X} " +
                $"({DescribeResolve(gpuVa, size)}).");
            size = _maxUploadBytes;
        }

        var range = Allocate(size, RingAlignment, out var span);
        ReadForStaging(gpuVa, span);
        Interlocked.Add(ref _ringFallbackBytes, (long)size);
        Interlocked.Increment(ref _ringFallbacks);
        return range;
    }

    /// <summary>
    /// Fills a host span with the guest bytes the CPU sees now. A partially mapped range
    /// still binds: the readable prefix is what the guest shader would have seen, and the
    /// rest reads as zero.
    /// </summary>
    public void ReadForStaging(ulong gpuVa, Span<byte> span)
    {
        MetalGpuReferenceCensus.Mark(gpuVa, (ulong)span.Length, MetalRefKind.Staged);
        if (_memory.TryRead(gpuVa, span))
        {
            return;
        }

        span.Clear();
        for (var offset = 0; offset < span.Length; offset += 4096)
        {
            var chunk = Math.Min(4096, span.Length - offset);
            if (!_memory.TryRead(gpuVa + (ulong)offset, span.Slice(offset, chunk)))
            {
                break;
            }
        }
    }

    /// <summary>A second ring copy of bytes already staged in the ring; invalid when the range is not a ring range.</summary>
    public unsafe MetalGuestRange Duplicate(in MetalGuestRange staged)
    {
        nint source = 0;
        foreach (var ring in _rings)
        {
            if (ring.Buffer == staged.Buffer && staged.Offset + staged.Size <= ring.Size)
            {
                source = ring.Contents + (nint)staged.Offset;
                break;
            }
        }

        if (source == 0 || staged.Size == 0)
        {
            return default;
        }

        var copy = Allocate(staged.Size, RingAlignment, out var span);
        new ReadOnlySpan<byte>((void*)source, checked((int)staged.Size)).CopyTo(span);
        return copy;
    }

    /// <summary>Reserves ring bytes and hands back the host span to fill.</summary>
    public MetalGuestRange Allocate(ulong size, ulong alignment, out Span<byte> span)
    {
        if (size == 0)
        {
            throw GpuFatal.Fatal("A guest memory ring allocation needs a nonzero size.");
        }

        var ring = Reserve(size, Math.Max(alignment, 16), out var offset);
        if (ring.LastUsedSerial != _submissionSerial)
        {
            // The whole ring is this submission's until it completes, however little of
            // it the reservation took, unless an earlier submission still holds it.
            ring.Inherited = ring.LastUsedSerial > _completedSerial;
            RecordingBytes += ring.Inherited ? size : ring.Size;
        }
        else if (ring.Inherited)
        {
            RecordingBytes += size;
        }

        ring.LastUsedSerial = _submissionSerial;
        ring.LastUsedAt = RingClock();
        unsafe
        {
            span = new Span<byte>((void*)(ring.Contents + (nint)offset), checked((int)size));
        }

        return new MetalGuestRange(ring.Buffer, offset, size, ring.GpuAddress + offset);
    }

    private RingBuffer Reserve(ulong size, ulong alignment, out ulong offset)
    {
        // A range past the ring's cap (one staged whole) gets a ring of just its size: the
        // spare half doubling leaves for later allocations would be hundreds of megabytes.
        var capacity = size > MaxUploadBytes
            ? (size + alignment + RingAlignment - 1) & ~(RingAlignment - 1)
            : Math.Max(RingBufferBytes, (size + alignment) * 2);
        for (var index = 0; index < _rings.Count; index++)
        {
            var candidate = _rings[index];
            var aligned = (candidate.Cursor + alignment - 1) & ~(alignment - 1);
            if (aligned <= candidate.Size && size <= candidate.Size - aligned)
            {
                candidate.Cursor = aligned + size;
                offset = aligned;
                return candidate;
            }
        }

        // Recycle a ring the GPU is done with before growing the pool: the smallest that
        // fits, so an oversized ring left by a burst is not kept alive by small requests.
        RingBuffer? recycled = null;
        for (var index = 0; index < _rings.Count; index++)
        {
            var candidate = _rings[index];
            if (candidate.LastUsedSerial > _completedSerial || size > candidate.Size)
            {
                continue;
            }

            if (recycled is null || candidate.Size < recycled.Size)
            {
                recycled = candidate;
            }
        }

        if (recycled is not null)
        {
            recycled.Cursor = size;
            offset = 0;
            return recycled;
        }

        var buffer = CreateSharedBuffer(capacity, $"guest.ring[{_rings.Count}]", out var contents, out var gpuAddress);
        var ring = new RingBuffer
        {
            Buffer = buffer,
            Contents = contents,
            GpuAddress = gpuAddress,
            Size = capacity,
            Cursor = size,
        };
        _rings.Add(ring);
        var total = Interlocked.Add(ref _ringBytes, (long)capacity);
        if (total > Volatile.Read(ref _ringPeakBytes))
        {
            Volatile.Write(ref _ringPeakBytes, total);
        }

        if (total > Volatile.Read(ref _ringWindowPeakBytes))
        {
            Volatile.Write(ref _ringWindowPeakBytes, total);
        }
        AddResident(buffer);
        CommitResidency();
        offset = 0;
        return ring;
    }

    /// <summary>The host span of a ring range this submission allocated.</summary>
    public unsafe Span<byte> RingSpan(in MetalGuestRange range)
    {
        foreach (var ring in _rings)
        {
            if (ring.Buffer == range.Buffer)
            {
                return new Span<byte>((void*)(ring.Contents + (nint)range.Offset), checked((int)range.Size));
            }
        }

        return default;
    }

    public unsafe Span<byte> GdsWords()
    {
        var contents = MetalNative.Send(_gdsBuffer, SelContents);
        return new Span<byte>((void*)contents, (int)GdsBytes);
    }

    /// <summary>
    /// Reads guest bytes on the host, for diagnostics that want to say what a shader is
    /// about to read. Not a transfer path: the GPU reaches the same bytes through a
    /// window or the upload ring.
    /// </summary>
    public bool TryPeek(ulong gpuVa, Span<byte> destination) => _memory.TryRead(gpuVa, destination);

    /// <summary>
    /// A small shared buffer outside the per-submission ring, for words the GPU writes
    /// and the host reads back after a submission. The caller owns its lifetime and
    /// must name it with <c>useResource</c> wherever a shader reaches it.
    /// </summary>
    public nint CreateReportBuffer(ulong size, string label, out nint contents, out ulong gpuAddress) =>
        CreateSharedBuffer(size, label, out contents, out gpuAddress);

    /// <summary>A resident shared buffer for one cached meshlet list; see
    /// <see cref="MetalMeshletCache"/>.</summary>
    public nint CreateMeshletBuffer(ulong size, out nint contents)
    {
        var buffer = CreateSharedBuffer(size, "mesh.meshlets", out contents, out _);
        AddResident(buffer);
        CommitResidency();
        return buffer;
    }

    /// <summary>A small shared buffer kept resident for the session, for a shader to reach
    /// through an argument-buffer address.</summary>
    public nint CreateResidentSharedBuffer(ulong size, string label, out nint contents, out ulong gpuAddress)
    {
        var buffer = CreateSharedBuffer(size, label, out contents, out gpuAddress);
        AddResident(buffer);
        CommitResidency();
        return buffer;
    }

    public void ReleaseMeshletBuffer(nint buffer)
    {
        RemoveResident(buffer);
        MetalNative.SendVoid(buffer, MetalNative.Selector("release"));
    }

    private const ulong PairLockPageBytes = 0x4000;
    private nint _pairLockTable;
    private ulong _pairLockTableGpuAddress;
    private readonly object _pairLockTableLock = new();

    /// <summary>The GPU address of the pair-lock table every program shares (the
    /// <c>pair_locks</c> argument field, Templates/pair_atomic_prelude.msl): one zeroed
    /// sequence word per slot, resident like the windows. Created on first use and never
    /// reset; a lock is even again whenever no access holds it.</summary>
    public ulong PairLockTableGpuAddress
    {
        get
        {
            lock (_pairLockTableLock)
            {
                if (_pairLockTable == 0)
                {
                    const int bytes = SharpEmu.ShaderCompiler.Metal.Gen5MslArgumentLayout.PairLockTableWords * sizeof(uint);
                    var buffer = CreateSharedBuffer(bytes, "guest.pair-locks", out var contents, out var gpuAddress);
                    unsafe
                    {
                        new Span<byte>((void*)contents, bytes).Clear();
                    }

                    AddResident(buffer);
                    CommitResidency();
                    _pairLockTableGpuAddress = gpuAddress;
                    _pairLockTable = buffer;
                }

                return _pairLockTableGpuAddress;
            }
        }
    }

    private nint _meshLdsPool;
    private readonly object _meshLdsPoolLock = new();

    /// <summary>The device-memory LDS pool of mesh programs whose guest LDS is too large for
    /// threadgroup memory (see <c>Gen5MslMeshInfo.DeviceLds</c>): a zeroed header of a probe
    /// counter and one busy flag per slot, then one slot of data per resident threadgroup.
    /// Created on first use and never reset: a threadgroup frees its slot when it ends.</summary>
    public nint MeshLdsPool
    {
        get
        {
            lock (_meshLdsPoolLock)
            {
                if (_meshLdsPool == 0)
                {
                    var buffer = CreateSharedBuffer(
                        SharpEmu.ShaderCompiler.Metal.Gen5MslMeshInfo.LdsPoolByteSize, "mesh.lds-pool", out var contents, out _);
                    unsafe
                    {
                        new Span<byte>(
                            (void*)contents,
                            (int)(SharpEmu.ShaderCompiler.Metal.Gen5MslMeshInfo.LdsPoolHeaderDwords * sizeof(uint))).Clear();
                    }

                    AddResident(buffer);
                    CommitResidency();
                    _meshLdsPool = buffer;
                }

                return _meshLdsPool;
            }
        }
    }

    private nint CreateSharedBuffer(ulong size, string label, out nint contents, out ulong gpuAddress)
    {
        var buffer = MetalNative.SendNewBuffer(
            _device,
            SelNewBuffer,
            (nuint)size,
            Mtl.ResourceStorageModeShared | HazardTrackingMode);
        if (buffer == 0)
        {
            throw GpuFatal.Fatal($"A Metal buffer could not be created: label={label} size=0x{size:X}.");
        }

        SetLabel(buffer, label);
        contents = MetalNative.Send(buffer, SelContents);
        gpuAddress = MetalNative.SendGpuAddress(buffer, SelGpuAddress);
        return buffer;
    }

    private static void SetLabel(nint resource, string label)
    {
        var pool = MetalNative.objc_autoreleasePoolPush();
        try
        {
            MetalNative.SendVoid(resource, SelSetLabel, MetalNative.NsString(label));
        }
        finally
        {
            MetalNative.objc_autoreleasePoolPop(pool);
        }
    }

    // MTLResidencySet keeps the windows mapped for the queue without per-encoder calls.
    // macOS before 15 has no residency set; the fallback is useResource on each encoder.
    private static nint CreateResidencySet(nint device)
    {
        var selector = MetalNative.Selector("newResidencySetWithDescriptor:error:");
        if (!MetalNative.SendBool(device, MetalNative.Selector("respondsToSelector:"), selector))
        {
            Console.Error.WriteLine("[METAL-V2][WARN] MTLResidencySet is unavailable; windows bind per encoder.");
            return 0;
        }

        var pool = MetalNative.objc_autoreleasePoolPush();
        try
        {
            var descriptorClass = MetalNative.Class("MTLResidencySetDescriptor");
            var descriptor = MetalNative.Send(
                MetalNative.Send(descriptorClass, MetalNative.Selector("alloc")),
                MetalNative.Selector("init"));
            nint error = 0;
            var set = MetalNative.Send(device, selector, descriptor, ref error);
            MetalNative.SendVoid(descriptor, MetalNative.Selector("release"));
            if (set == 0)
            {
                Console.Error.WriteLine(
                    $"[METAL-V2][WARN] MTLResidencySet creation failed: {MetalNative.DescribeError(error)}");
            }

            return set;
        }
        finally
        {
            MetalNative.objc_autoreleasePoolPop(pool);
        }
    }

    public void AddResident(nint allocation)
    {
        if (_residencySet != 0 && allocation != 0)
        {
            MetalNative.SendVoid(_residencySet, SelAddAllocation, allocation);
            return;
        }

        NoteResident(allocation);
    }

    /// <summary>Drops a released allocation from whatever keeps it resident.</summary>
    public void RemoveResident(nint allocation)
    {
        if (allocation == 0)
        {
            return;
        }

        if (_residencySet != 0)
        {
            var selector = MetalNative.Selector("removeAllocation:");
            if (MetalNative.SendBool(_residencySet, MetalNative.Selector("respondsToSelector:"), selector))
            {
                MetalNative.SendVoid(_residencySet, selector, allocation);
                CommitResidency();
            }

            return;
        }

        for (var index = 0; index < _residentCount; index++)
        {
            if (_residentBuffers[index] != allocation)
            {
                continue;
            }

            _residentBuffers[index] = _residentBuffers[--_residentCount];
            _residentBuffers[_residentCount] = 0;
            _residentDirty = true;
            return;
        }
    }

    public void CommitResidency()
    {
        if (_residencySet == 0)
        {
            return;
        }

        // Commit publishes the allocation list; requesting residency once keeps the
        // windows mapped for the queue instead of letting Metal establish residency
        // around every command buffer. The pages stay lazily backed either way.
        MetalNative.SendVoid(_residencySet, SelCommit);
        MetalNative.SendVoid(_residencySet, MetalNative.Selector("requestResidency"));
    }

    /// <summary>Attaches the residency set to a queue, once per queue.</summary>
    public void AttachToQueue(nint queue)
    {
        if (_residencySet == 0 || queue == 0)
        {
            return;
        }

        var selector = MetalNative.Selector("addResidencySet:");
        if (MetalNative.SendBool(queue, MetalNative.Selector("respondsToSelector:"), selector))
        {
            MetalNative.SendVoid(queue, selector, _residencySet);
        }
    }

    /// <summary>
    /// The window and ring buffers an encoder has to be told about when there is no
    /// residency set: an argument buffer holds their addresses, not references, so
    /// Metal cannot see the dependency on its own. The windows are the ones this
    /// submission resolved a range through, and only those: naming a window wires it.
    /// </summary>
    public ReadOnlySpan<nint> ResidentBuffers
    {
        get
        {
            if (_residencySet != 0)
            {
                return default;
            }

            if (_residentDirty)
            {
                _residentDirty = false;
                _submissionResidentCount = 0;
                lock (_windowLock)
                {
                    foreach (var id in _touchedWindows)
                    {
                        if (_partition.ById(id) is { Buffer: not 0 } window)
                        {
                            Append(ref _submissionResidentBuffers, ref _submissionResidentCount, window.Buffer);
                        }
                    }
                }

                for (var index = 0; index < _residentCount; index++)
                {
                    Append(ref _submissionResidentBuffers, ref _submissionResidentCount, _residentBuffers[index]);
                }

                // Guest views, like windows, are named only by the submissions that bind them.
                foreach (var view in _touchedViews)
                {
                    Append(ref _submissionResidentBuffers, ref _submissionResidentCount, view.Buffer);
                }
            }

            return _submissionResidentBuffers.AsSpan(0, _submissionResidentCount);
        }
    }

    private static void Append(ref nint[] buffers, ref int count, nint buffer)
    {
        if (count == buffers.Length)
        {
            Array.Resize(ref buffers, buffers.Length * 2);
        }

        buffers[count++] = buffer;
    }

    private nint[] _residentBuffers = new nint[16];
    private int _residentCount;
    private nint[] _submissionResidentBuffers = new nint[16];
    private int _submissionResidentCount;

    private void NoteResident(nint buffer)
    {
        if (_residencySet != 0 || buffer == 0)
        {
            return;
        }

        for (var index = 0; index < _residentCount; index++)
        {
            if (_residentBuffers[index] == buffer)
            {
                return;
            }
        }

        if (_residentCount == _residentBuffers.Length)
        {
            Array.Resize(ref _residentBuffers, _residentBuffers.Length * 2);
        }

        _residentBuffers[_residentCount++] = buffer;
        _residentDirty = true;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        foreach (var ring in _rings)
        {
            MetalNative.SendVoid(ring.Buffer, SelRelease);
        }

        _rings.Clear();
        lock (_windowLock)
        {
            foreach (var window in _partition.Windows)
            {
                if (window.Buffer != 0)
                {
                    MetalNative.SendVoid(window.Buffer, SelRelease);
                    window.Buffer = 0;
                }
            }

            foreach (var (window, _) in _retired)
            {
                if (window.Buffer != 0)
                {
                    MetalNative.SendVoid(window.Buffer, SelRelease);
                    window.Buffer = 0;
                }
            }

            _retired.Clear();
            _windowBuffers.Clear();
            _gpuAlias?.Dispose();
        }

        DisposeGuestViews();

        if (_gdsBuffer != 0)
        {
            MetalNative.SendVoid(_gdsBuffer, SelRelease);
        }

        if (_nullBuffer != 0)
        {
            MetalNative.SendVoid(_nullBuffer, SelRelease);
        }

        if (_pairLockTable != 0)
        {
            MetalNative.SendVoid(_pairLockTable, SelRelease);
        }

        if (_residencySet != 0)
        {
            MetalNative.SendVoid(_residencySet, SelRelease);
        }
    }
}
