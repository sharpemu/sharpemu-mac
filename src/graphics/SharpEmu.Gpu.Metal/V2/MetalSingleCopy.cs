// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Single-copy textures (artifacts/memopt/single-copy-textures.md, stage 4): a clean
/// sampled texture that gives its guest bytes back exactly through a retile lets the 64 KiB
/// chunks of the chunked backing it alone covers be emptied, and gets them back the moment
/// anything else wants them. On wherever the backing is chunked; SHARPEMU_METAL_V2_SINGLE_COPY=0 turns it off.
///
/// <para>
/// A chunk goes Arming → Pending → Emptying → Emptied:
/// <list type="number">
/// <item>Arming (GPU worker, image lock held, from the cache's Collect): the chunk lies whole
/// inside the image's run, one view maps it, at one address, and no other image, staged
/// range, metadata watch or pending GPU write touches it. It is marked released in
/// <see cref="GuestSingleCopy"/>, so every host access through the canonical alias asks here
/// first.</item>
/// <item>Pending: a read+write page watch is on its guest address, so a guest access faults
/// here first, and the GPU alias gets a hole over it as the next submission begins
/// (MetalGuestMemory.SingleCopy); guest views over it are dropped then.</item>
/// <item>Emptying: once that submission has completed and the grace has passed (§4 a3: an
/// entry emptied soon after the GPU used it keeps its pages), the texture is retiled into
/// zeroed staging on a queue of its own and the chunk is emptied only if the guest bytes
/// equal the retile byte for byte, the image is still clean at the version it was armed at
/// and the chunk is still mapped the same way. Otherwise the chunk is given up (a verify
/// reject).</item>
/// <item>Emptied: its pages are back with the OS.</item>
/// </list>
/// </para>
///
/// <para>
/// A restore retiles the record's texture on the same queue, waits, copies every emptied
/// chunk of the record back, and only then clears the marks; the watch goes afterwards
/// (queued: the hooks can run with the PVM gate held, and removing a watch takes it). A
/// whole-chunk overwrite (a file read into the chunk, a clear, a free) skips the retile
/// and only drops the association. Changing the texture restores first (the
/// <see cref="MetalCachedImage.SingleCopyRecord"/> hook), and so does every release of
/// the image except an unmap, which keeps the texture for chunks no guest address maps
/// any more until the backing is freed (forgotten) or mapped again (restored).
/// </para>
///
/// <para>
/// Locks: M (<c>_m</c>) guards the records and chunks and is innermost but for the window
/// lock; W (<c>_w</c>) guards the watches this manager holds and is taken outside M and
/// PageGuard. A holder of M takes none of the views lock, the PVM gate, PageGuard, W or
/// the image lock; the GPU work runs under <c>_gpuLock</c> alone.
/// </para>
/// </summary>
internal sealed unsafe class MetalSingleCopy : IGuestSingleCopyHandler, IDisposable
{
    /// <summary>On wherever the backing is chunked; SHARPEMU_METAL_V2_SINGLE_COPY=0 turns it off.</summary>
    public static bool Enabled { get; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_SINGLE_COPY") != "0";

    /// <summary>Whether SHARPEMU_METAL_V2_SINGLE_COPY=1 asked for it explicitly (a missing GPU alias is then worth a warning).</summary>
    public static bool Requested { get; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_SINGLE_COPY") == "1";

    /// <summary>
    /// SHARPEMU_METAL_V2_SINGLE_COPY_SENTINEL=1 (debug): a released chunk is filled with
    /// <see cref="ReleasedSentinel"/> instead of emptied and the GPU alias dummy holds
    /// <see cref="DummySentinel"/>, so a read that was not restored first shows up as a
    /// sentinel rather than as zeros.
    /// </summary>
    internal static bool SentinelMode { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_SINGLE_COPY_SENTINEL") == "1";

    /// <summary>
    /// SHARPEMU_METAL_V2_SINGLE_COPY_SHADOW=1 (validation): arming, verify and the records run
    /// as normal, but nothing is punched out of the GPU alias or emptied. The chunks that would
    /// have been are flagged in the device-address table all the same, so the shader's trap
    /// counts the accesses that would have read a hole (da_shadow_hits) and lets them through,
    /// and sc_shadow_released_mb says what would have been released.
    /// </summary>
    internal static bool ShadowMode { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_SINGLE_COPY_SHADOW") == "1";

    /// <summary>
    /// SHARPEMU_METAL_V2_SINGLE_COPY_DA=1 (or the shadow mode): a device-address program pins and
    /// gives back only the chunks the ranges evaluated for it cover, and chunks stay released
    /// around it. Off by default until a Demon's Souls run shows it renders right and costs no
    /// frames: then the first device-address program takes the global fallback, as before.
    /// </summary>
    internal static bool DeviceAddressPins { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_SINGLE_COPY_DA") == "1" || ShadowMode;

    internal const byte DummySentinel = 0xD5;
    internal const byte ReleasedSentinel = 0xA7;

    // Why a chunk is pinned (telemetry): an exact evaluated range, a capped one, a trap hit;
    // and whether it was counted as one single copy wanted.
    private const byte PinExact = 1;
    private const byte PinCapped = 2;
    private const byte PinTrap = 4;
    private const byte PinCandidate = 0x80;

    private const ulong MiB = 1024 * 1024;
    private const ulong StatusCompleted = 4;

    private static readonly nint SelNewCommandQueue = MetalNative.Selector("newCommandQueue");
    private static readonly nint SelCommandBuffer = MetalNative.Selector("commandBuffer");
    private static readonly nint SelBlit = MetalNative.Selector("blitCommandEncoder");
    private static readonly nint SelFillBuffer = MetalNative.Selector("fillBuffer:range:value:");
    private static readonly nint SelEndEncoding = MetalNative.Selector("endEncoding");
    private static readonly nint SelCommit = MetalNative.Selector("commit");
    private static readonly nint SelWait = MetalNative.Selector("waitUntilCompleted");
    private static readonly nint SelStatus = MetalNative.Selector("status");
    private static readonly nint SelRetain = MetalNative.Selector("retain");
    private static readonly nint SelRelease = MetalNative.Selector("release");
    private static readonly nint SelContents = MetalNative.Selector("contents");
    private static readonly nint SelLength = MetalNative.Selector("length");
    private static readonly nint SelNewBuffer = MetalNative.Selector("newBufferWithLength:options:");

    /// <summary>Timings and caps; the defaults are the protocol's, tests shorten them.</summary>
    internal sealed class Settings
    {
        /// <summary>How long an image's content must have stayed the same before it is armed.</summary>
        public TimeSpan Idle = TimeSpan.FromSeconds(5);

        /// <summary>How long after the punch's submission completed a chunk waits to be emptied.</summary>
        public TimeSpan Grace = TimeSpan.FromSeconds(2);

        /// <summary>How often the images are scanned for candidates.</summary>
        public TimeSpan ScanInterval = TimeSpan.FromMilliseconds(250);

        /// <summary>A CPU restore keeps a chunk from being armed again this long, doubling per strike.</summary>
        public TimeSpan Cooldown = TimeSpan.FromSeconds(10);

        /// <summary>The largest run one image may release (its retile staging).</summary>
        public ulong MaxRunBytes = 256 * MiB;

        /// <summary>Bytes armed per scan.</summary>
        public ulong ArmBytesPerScan = 256 * MiB;

        /// <summary>Textures held for chunks no guest address maps.</summary>
        public ulong MaxOrphanBytes = 256 * MiB;
    }

    internal enum ChunkState : byte
    {
        Arming,
        Pending,
        Emptying,
        Emptied,
    }

    internal sealed class Chunk(int index, ulong offset, ulong address, ulong runOffset, Record record)
    {
        public readonly int Index = index;
        public readonly ulong Offset = offset;
        public readonly ulong RunOffset = runOffset;
        public readonly Record Record = record;

        // The guest address that maps the chunk; zero once the guest unmapped it.
        public ulong Address = address;
        public ChunkState State;
        public bool Cancelled;
        public ulong PunchSerial;
        public long SettledAt;
        public long EmptiedAt;
        public bool NeedsReEmpty;

        // Counted in the orphan bytes: the image went with the unmap and the record kept the texture.
        public bool Orphan;
    }

    /// <summary>A texture and the chunks whose guest copy it holds.</summary>
    internal sealed class Record
    {
        public MetalSingleCopy Owner = null!;
        public MetalCachedImage? Image;
        public nint Texture;
        public ImageDescription Description;
        public uint FirstLevel;
        public ulong RunStart;
        public ulong RunEnd;
        public long ContentVersion;
        public readonly List<Chunk> Chunks = new();

        // A retile, an empty or a restore of the record runs outside M.
        public bool Busy;
        public bool Gone;
    }

    private sealed class Overwrite(List<Chunk> chunks)
    {
        public readonly List<Chunk> Chunks = chunks;
    }

    // The retile's submission context: one command buffer at a time on the private queue.
    private sealed class RestoreContext(nint device) : IMetalSubmissionContext
    {
        public nint Device { get; } = device;

        public nint CommandBuffer { get; set; }

        public ulong SubmissionSerial { get; set; } = 1;

        public ulong CompletedSerial { get; set; }
    }

    private readonly nint _device;
    private readonly nint _queue;
    private readonly MetalImageTransfer _transfer;
    private readonly RestoreContext _context;
    private readonly GuestSingleCopy _state;
    private readonly MetalGuestMemory _memory;
    private readonly IGuestBackedSpace _backing;
    private readonly PageGuard _pages;
    private readonly Func<ulong> _completedSerial;
    private readonly Settings _settings;
    private readonly ulong _chunkBytes;
    private readonly ulong _aliasBase;

    private readonly object _m = new();
    private readonly Dictionary<int, Chunk> _chunks = new();
    private readonly Dictionary<ulong, Chunk> _byAddress = new();
    private readonly HashSet<Record> _records = new();
    private readonly Dictionary<int, (long Until, int Strikes)> _cooldown = new();
    private readonly HashSet<int> _noRelease = new();
    private readonly byte[] _everReleased;

    private readonly object _w = new();
    private readonly HashSet<ulong> _watched = new();
    private readonly ConcurrentQueue<ulong> _unwatch = new();

    private readonly object _gpuLock = new();

    // SHARPEMU_SINGLE_COPY_RETILE_POOL=0 allocates and releases a staging buffer per retile, as before.
    private static readonly bool PoolRetileStaging =
        Environment.GetEnvironmentVariable("SHARPEMU_SINGLE_COPY_RETILE_POOL") != "0";

    // Retile staging buffers kept for the next retile: a restore runs on the faulting guest thread, and a
    // fresh shared buffer costs the allocation and its first-touch page faults every time. A retile only
    // needs the bytes it fills and the texture writes, so any buffer at least as large will do.
    private const int RetilePoolMaxBuffers = 4;
    private const ulong RetilePoolMaxBytes = 256UL * 1024 * 1024;
    private const ulong RetileClassBytes = 1024 * 1024;
    private readonly List<(nint Buffer, ulong Capacity)> _retilePool = new();
    private ulong _retilePoolBytes;
    private long _retilePoolHits;
    private long _retilePoolMisses;

    /// <summary>Retile staging buffers taken from the pool, and those created, since the start.</summary>
    internal (long Hits, long Misses) RetilePoolCounts =>
        (Interlocked.Read(ref _retilePoolHits), Interlocked.Read(ref _retilePoolMisses));

    // The smallest pooled buffer that holds the run, or a new one of the run's size class.
    private nint TakeRetileStaging(ulong bytes)
    {
        if (PoolRetileStaging)
        {
            lock (_retilePool)
            {
                var best = -1;
                for (var index = 0; index < _retilePool.Count; index++)
                {
                    if (_retilePool[index].Capacity >= bytes &&
                        (best < 0 || _retilePool[index].Capacity < _retilePool[best].Capacity))
                    {
                        best = index;
                    }
                }

                if (best >= 0)
                {
                    var taken = _retilePool[best].Buffer;
                    _retilePoolBytes -= _retilePool[best].Capacity;
                    _retilePool.RemoveAt(best);
                    Interlocked.Increment(ref _retilePoolHits);
                    return taken;
                }
            }
        }

        Interlocked.Increment(ref _retilePoolMisses);
        var capacity = PoolRetileStaging
            ? (bytes + RetileClassBytes - 1) / RetileClassBytes * RetileClassBytes
            : bytes;
        return MetalNative.SendNewBuffer(_device, SelNewBuffer, (nuint)capacity, 0);
    }

    // Keeps the buffer for the next retile, or releases it when the pool is full.
    private void ReleaseRetileStaging(nint staging)
    {
        if (PoolRetileStaging && staging != 0)
        {
            var capacity = (ulong)MetalV2Native.SendValue(staging, SelLength);
            lock (_retilePool)
            {
                if (!_disposed && _retilePool.Count < RetilePoolMaxBuffers &&
                    _retilePoolBytes + capacity <= RetilePoolMaxBytes)
                {
                    _retilePool.Add((staging, capacity));
                    _retilePoolBytes += capacity;
                    return;
                }
            }
        }

        MetalNative.SendVoid(staging, SelRelease);
    }
    private long _lastScan;
    private bool _stopped;
    private bool _disposed;

    // Counters, under M unless interlocked.
    private long _emptiedBytes;
    private long _orphanBytes;
    private long _restoredBytes;
    private readonly long[] _restoresBy = new long[7];
    private long _restoreTicks;
    private long _verifyRejects;
    private long _discards;
    private long _reEmpties;
    private long _overwrites;
    private long _drops;
    private long _armed;
    private long _deviceAddressStale;
    private long _deviceAddressRestores;
    private bool _deviceAddressSeen;

    // Device-address programs (artifacts/memopt/single-copy-device-address.md (d)): the census
    // of what the host evaluated before each one, the chunks those ranges pinned, and the trap.
    private readonly bool _shadow;
    private readonly byte[] _pinFlags;
    private readonly HashSet<(ulong Base, ulong Size)> _pinnedRanges = new();
    private long _pinnedRangesVersion = long.MinValue;
    private long _daPrograms;
    private long _daSubmissions;
    private long _daRangesExact;
    private long _daRangesCapped;
    private long _daUnplannedHandles;
    private long _daHiddenWrites;
    private long _daInvisiblePrograms;
    private long _daFallbacks;
    private long _daLatePins;
    private long _daReleasedHits;
    private long _daShadowHits;

    // Under M: pinned chunks, by why, and those single copy wanted.
    private long _pinnedChunks;
    private long _pinnedExact;
    private long _pinnedCapped;
    private long _pinnedTrap;
    private long _pinnedCandidates;
    private long _pinnedCappedCandidates;
    private long _daReleasedHitChunks;
    private long _daShadowHitChunks;

    public MetalSingleCopy(
        nint device,
        MetalGuestMemory memory,
        IGuestBackedSpace backing,
        PageGuard pages,
        Func<ulong> completedSerial,
        Settings? settings = null)
    {
        _state = backing.SingleCopy ?? throw GpuFatal.Fatal("Single-copy textures need the chunked guest backing.");
        if (memory.GpuAlias is null)
        {
            throw GpuFatal.Fatal("Single-copy textures need the guest memory windows over the GPU alias.");
        }

        _device = device;
        _memory = memory;
        _backing = backing;
        _pages = pages;
        _completedSerial = completedSerial;
        _settings = settings ?? new Settings();
        _chunkBytes = _state.ChunkBytes;
        _aliasBase = _state.AliasBase;
        _everReleased = new byte[_state.ChunkCount];
        _pinFlags = new byte[_state.ChunkCount];
        _shadow = memory.IsSingleCopyShadow;
        _queue = MetalNative.Send(device, SelNewCommandQueue);
        if (_queue == 0)
        {
            throw GpuFatal.Fatal("Single-copy textures could not create their command queue.");
        }

        _transfer = new MetalImageTransfer(device);
        _context = new RestoreContext(device);
        _state.Handler = this;
        Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"[METAL-V2][INFO] single-copy textures: chunk=0x{_chunkBytes:X} idle={_settings.Idle.TotalSeconds:F1}s " +
            $"grace={_settings.Grace.TotalSeconds:F1}s max_run={_settings.MaxRunBytes / MiB}MB sentinel={(SentinelMode ? 1 : 0)} " +
            $"shadow={(_shadow ? 1 : 0)}"));
    }

    /// <summary>The title's kill switch (null: none); set, a device-address program takes the fallback.</summary>
    public MetalSingleCopyKillSwitch? KillSwitch { get; set; }

    /// <summary>The device-address table has little room left for trap cuts: nothing new is armed.</summary>
    public bool ArmingPaused
    {
        get => Volatile.Read(ref _armingPaused);
        set => Volatile.Write(ref _armingPaused, value);
    }

    private bool _armingPaused;

    /// <summary>Whether nothing is punched or emptied (SHARPEMU_METAL_V2_SINGLE_COPY_SHADOW=1).</summary>
    public bool IsShadow => _shadow;

    /// <summary>Whether the guest memory windows can serve single-copy textures (the GPU alias exists).</summary>
    public static bool CanServe(MetalGuestMemory memory, IGuestBackedSpace backing) =>
        memory.GpuAlias is not null && backing.SingleCopy is not null;

    // ---- Telemetry -------------------------------------------------------------------

    /// <summary>The "singlecopy" section of the [MEM] line.</summary>
    public string Describe()
    {
        const double Mb = 1048576.0;
        long emptied, orphan, restored, ticks, rejects, discards, reEmpties, overwrites, drops, armed;
        long pinned, pinnedExact, pinnedCapped, pinnedTrap, pinCandidates, cappedCandidates, releasedHitChunks, shadowHitChunks;
        int pending, records;
        long[] by;
        lock (_m)
        {
            pinned = _pinnedChunks;
            pinnedExact = _pinnedExact;
            pinnedCapped = _pinnedCapped;
            pinnedTrap = _pinnedTrap;
            pinCandidates = _pinnedCandidates;
            cappedCandidates = _pinnedCappedCandidates;
            releasedHitChunks = _daReleasedHitChunks;
            shadowHitChunks = _daShadowHitChunks;
            emptied = _emptiedBytes;
            orphan = _orphanBytes;
            restored = _restoredBytes;
            ticks = _restoreTicks;
            rejects = _verifyRejects;
            discards = _discards;
            reEmpties = _reEmpties;
            overwrites = _overwrites;
            drops = _drops;
            armed = _armed;
            by = (long[])_restoresBy.Clone();
            pending = _chunks.Values.Count(static chunk => chunk.State is ChunkState.Arming or ChunkState.Pending);
            records = _records.Count;
        }

        var window = _memory.SingleCopyWindowCounts;
        var chunkMb = _chunkBytes / Mb;
        return string.Create(CultureInfo.InvariantCulture,
            $"single_copy_released_mb={(_shadow ? 0 : emptied) / Mb:F1} sc_shadow={(_shadow ? 1 : 0)} " +
            $"sc_shadow_released_mb={(_shadow ? emptied : 0) / Mb:F1} " +
            $"da_programs={Interlocked.Read(ref _daPrograms)} da_submissions={Interlocked.Read(ref _daSubmissions)} " +
            $"da_ranges_exact={Interlocked.Read(ref _daRangesExact)} da_ranges_capped={Interlocked.Read(ref _daRangesCapped)} " +
            $"da_unplanned_handles={Interlocked.Read(ref _daUnplannedHandles)} da_hidden_writes={Interlocked.Read(ref _daHiddenWrites)} " +
            $"da_invisible_programs={Interlocked.Read(ref _daInvisiblePrograms)} da_fallbacks={Interlocked.Read(ref _daFallbacks)} " +
            $"da_kill_switch={(KillSwitch?.IsSet == true ? 1 : 0)} da_kill_switch_tripped={(KillSwitch?.Tripped == true ? 1 : 0)} " +
            $"da_pinned_mb={pinned * chunkMb:F1} da_planned_mb={pinnedExact * chunkMb:F1} da_capped_mb={pinnedCapped * chunkMb:F1} " +
            $"da_pin_candidate_mb={pinCandidates * chunkMb:F1} da_capped_candidate_mb={cappedCandidates * chunkMb:F1} " +
            $"da_trap_pinned_mb={pinnedTrap * chunkMb:F1} da_late_pins={Interlocked.Read(ref _daLatePins)} " +
            $"restores_da={by[(int)GuestSingleCopyAccess.DeviceAddress]} " +
            $"da_released_hits={Interlocked.Read(ref _daReleasedHits)} da_released_hit_chunks={releasedHitChunks} " +
            $"da_shadow_hits={Interlocked.Read(ref _daShadowHits)} da_shadow_hit_chunks={shadowHitChunks} " +
            $"arming_paused={(ArmingPaused ? 1 : 0)} " +
            $"restored_mb={restored / Mb:F1} restores={by.Sum()} " +
            $"restore_ms={ticks * 1000.0 / Stopwatch.Frequency:F1} verify_rejects={rejects} " +
            $"restores_fault={by[(int)GuestSingleCopyAccess.Fault]} restores_host_read={by[(int)GuestSingleCopyAccess.HostRead]} " +
            $"restores_host_write={by[(int)GuestSingleCopyAccess.HostWrite]} restores_map={by[(int)GuestSingleCopyAccess.Map]} " +
            $"restores_gpu={by[(int)GuestSingleCopyAccess.GpuBinding]} restores_teardown={by[(int)GuestSingleCopyAccess.Teardown]} " +
            $"single_copy_armed_chunks={armed} pending_empty={pending} records={records} discards={discards} " +
            $"re_empties={reEmpties} overwrites={overwrites} drops={drops} orphan_mb={orphan / Mb:F1} " +
            $"gpu_holes={window.Holes} hole_punches={window.Punches} hole_fills={window.Fills} " +
            $"window_renewals={window.Renewals} stale_window_refusals={window.StaleRefusals} " +
            $"device_address_stale={Interlocked.Read(ref _deviceAddressStale)} " +
            $"device_address_restores={Interlocked.Read(ref _deviceAddressRestores)} " +
            $"device_address_seen={(DeviceAddressSeen ? 1 : 0)}");
    }

    /// <summary>Released bytes now emptied, restores so far and verify rejects (tests).</summary>
    internal (ulong EmptiedBytes, long Restores, long VerifyRejects, long Overwrites, long Drops, int Chunks) Counts
    {
        get
        {
            lock (_m)
            {
                return ((ulong)_emptiedBytes, _restoresBy.Sum(), _verifyRejects, _overwrites, _drops, _chunks.Count);
            }
        }
    }

    internal bool IsEmptied(int chunk)
    {
        lock (_m)
        {
            return _chunks.TryGetValue(chunk, out var entry) && entry.State == ChunkState.Emptied;
        }
    }

    // ---- The tick: arm, empty, unwatch ------------------------------------------------

    /// <summary>From the image cache's Collect, on the GPU worker, with the image lock held. Returns whether it
    /// unwatched, took a punch, emptied or tried to arm anything: what guest memory backs may have changed
    /// (MetalImageCache.Collect counts it as an effect).</summary>
    public bool Tick(MetalImageCache cache, long now)
    {
        if (_disposed)
        {
            return false;
        }

        var changed = DrainUnwatches();
        if (_stopped)
        {
            return changed;
        }

        changed |= TakePunched(now);
        if (DeviceAddressSeen)
        {
            // Nothing new is armed; what was armed was given back when the program ran.
            return changed;
        }

        changed |= EmptyDue(cache, now);
        if (!ArmingPaused && now - _lastScan >= Ticks(_settings.ScanInterval))
        {
            _lastScan = now;
            changed |= ArmCandidates(cache, now);
        }

        return changed;
    }

    private static long Ticks(TimeSpan span) => (long)(span.TotalSeconds * Stopwatch.Frequency);

    // Whether it found a candidate to arm.
    private bool ArmCandidates(MetalImageCache cache, long now)
    {
        var budget = _settings.ArmBytesPerScan;
        var idle = Ticks(_settings.Idle);
        List<MetalCachedImage>? candidates = null;
        cache.ForEachImageLocked(image =>
        {
            // The version the image was last seen at, and since when.
            if (image.SingleCopySeenVersion != image.ContentVersion)
            {
                image.SingleCopySeenVersion = image.ContentVersion;
                image.SingleCopySeenAt = now;
                return;
            }

            if (image.SingleCopyRecord is null && now - image.SingleCopySeenAt >= idle && IsCandidateImage(image))
            {
                (candidates ??= new()).Add(image);
            }
        });

        if (candidates is null)
        {
            return false;
        }

        foreach (var image in candidates)
        {
            if (budget == 0)
            {
                break;
            }

            budget -= Math.Min(budget, TryArm(cache, image, now));
        }

        return true;
    }

    // The image-level rules; the chunk-level ones are in TryArm.
    private static bool IsCandidateImage(MetalCachedImage image)
    {
        ref readonly var description = ref image.Description;
        return image.Registered && image.Uses.Texture && !image.Uses.Storage && !image.Uses.RenderTarget &&
            !image.Uses.DepthTarget && !image.Uses.VideoOut && !image.IsCpuDirty && !image.IsGpuModified &&
            image.IsWatched && !image.IsWindowLinearView && image.Sparse is null && !image.Shape.Lossy &&
            image.Texture != 0 && !description.IsDepth && !description.HasStencil && description.Samples == 1 &&
            !description.HasMetadata;
    }

    // Arms the whole chunks of the image's run that it alone holds; the bytes armed.
    private ulong TryArm(MetalImageCache cache, MetalCachedImage image, long now)
    {
        var verdict = _transfer.ClassifyRoundTrip(image.Description, image.FirstLevel, image.Texture);
        if (verdict.Fidelity == MetalRoundTripFidelity.None || verdict.RunStart != image.ResidentStart ||
            verdict.RunEnd != image.ResidentEnd || verdict.RunEnd <= verdict.RunStart ||
            verdict.RunEnd - verdict.RunStart > _settings.MaxRunBytes || verdict.RunEnd - verdict.RunStart < _chunkBytes)
        {
            return 0;
        }

        var runAddress = image.Description.Data.Address + verdict.RunStart;
        var runEnd = image.Description.Data.Address + verdict.RunEnd;
        var found = new List<(int Chunk, ulong Offset, ulong Address)>();
        var address = runAddress;
        while (address < runEnd)
        {
            if (!_backing.TryResolveBackingAlias(address, 1, out var alias) || alias < _aliasBase)
            {
                address = (address | 0xFFF) + 1;
                continue;
            }

            var offset = alias - _aliasBase;
            var index = (int)(offset / _chunkBytes);
            var chunkOffset = (ulong)index * _chunkBytes;
            var chunkAddress = address - (offset - chunkOffset);
            var next = chunkAddress + _chunkBytes;
            if (chunkAddress >= runAddress && next <= runEnd && chunkOffset + _chunkBytes <= _state.Size &&
                IsReleasableChunk(cache, image, index, chunkOffset, chunkAddress, now))
            {
                found.Add((index, chunkOffset, chunkAddress));
            }

            address = next;
        }

        if (found.Count == 0)
        {
            return 0;
        }

        // 1. The record and its chunks, Arming.
        var record = new Record
        {
            Owner = this,
            Image = image,
            Texture = MetalNative.Send(image.Texture, SelRetain),
            Description = image.Description,
            FirstLevel = image.FirstLevel,
            RunStart = verdict.RunStart,
            RunEnd = verdict.RunEnd,
            ContentVersion = image.ContentVersion,
        };
        var chunks = new List<Chunk>(found.Count);
        lock (_m)
        {
            foreach (var (index, offset, chunkAddress) in found)
            {
                if (_chunks.ContainsKey(index))
                {
                    continue;
                }

                var chunk = new Chunk(index, offset, chunkAddress, chunkAddress - runAddress, record);
                _chunks[index] = chunk;
                _byAddress[chunkAddress] = chunk;
                record.Chunks.Add(chunk);
                chunks.Add(chunk);
            }

            _records.Add(record);
        }

        image.SingleCopyRecord = record;

        // 2. Marked released, under the views lock (a mapping in flight refuses).
        var marked = new List<Chunk>(chunks.Count);
        foreach (var chunk in chunks)
        {
            var ok = _state.TryMarkReleased(chunk.Index, chunk.Address);
            lock (_m)
            {
                if (chunk.Cancelled)
                {
                    // Restored while it was being marked: the mark it got is not wanted.
                    _state.ClearReleased(chunk.Index);
                }
                else if (!ok)
                {
                    ForgetChunkLocked(chunk);
                }
                else
                {
                    _everReleased[chunk.Index] = 1;
                    marked.Add(chunk);
                }
            }
        }

        // 3-4. Pending, with the read+write watch on its guest address.
        var pending = new List<Chunk>(marked.Count);
        lock (_w)
        {
            foreach (var chunk in marked)
            {
                lock (_m)
                {
                    if (chunk.Cancelled || chunk.State != ChunkState.Arming)
                    {
                        continue;
                    }

                    chunk.State = ChunkState.Pending;
                    _armed++;
                }

                if (_watched.Add(chunk.Address))
                {
                    _pages.AddWatch(chunk.Address, _chunkBytes, blockReads: true);
                }

                pending.Add(chunk);
            }
        }

        DropRecordIfEmpty(record);

        // 5. The hole in the GPU alias, as the next submission begins.
        foreach (var chunk in pending)
        {
            _memory.QueueSingleCopyPunch(chunk.Index, chunk.Address);
        }

        return (ulong)pending.Count * _chunkBytes;
    }

    private bool IsReleasableChunk(
        MetalImageCache cache, MetalCachedImage image, int index, ulong offset, ulong address, long now)
    {
        if (_state.IsReleased(index) || _state.ChunkViews(index) != 1 ||
            !_backing.TryResolveBackingAlias(address, _chunkBytes, out var alias) || alias != _aliasBase + offset)
        {
            return false;
        }

        // A page someone else already read-watches cannot take this watch too (one read
        // watch per page), and a page the guest cannot read is not one to arm.
        for (var page = address; page < address + _chunkBytes; page += 4096)
        {
            if (!_pages.Allows(page, FaultKind.Read))
            {
                return false;
            }
        }

        lock (_m)
        {
            if (_chunks.ContainsKey(index) || _noRelease.Contains(index) ||
                (_cooldown.TryGetValue(index, out var cooldown) && now < cooldown.Until))
            {
                return false;
            }

            if (_state.IsPinned(index))
            {
                // A device-address program may read it: counted once as a chunk single copy wanted.
                NotePinCandidateLocked(index);
                return false;
            }
        }

        return cache.IsSoleSingleCopyOwnerLocked(image, address, _chunkBytes);
    }

    // Whether it took any.
    private bool TakePunched(long now)
    {
        var took = false;
        while (_memory.TryTakeSingleCopyPunched(out var punched))
        {
            took = true;
            lock (_m)
            {
                if (_chunks.TryGetValue(punched.Chunk, out var chunk) && chunk.State == ChunkState.Pending &&
                    chunk.PunchSerial == 0)
                {
                    chunk.PunchSerial = punched.Serial;
                }
            }
        }

        return took;
    }

    // Empties every pending chunk whose punch has settled, a record at a time; whether any was due.
    private bool EmptyDue(MetalImageCache cache, long now)
    {
        var completed = _completedSerial();
        var grace = Ticks(_settings.Grace);
        List<Record>? due = null;
        List<Chunk>? reEmpty = null;
        lock (_m)
        {
            foreach (var chunk in _chunks.Values)
            {
                if (chunk.State == ChunkState.Pending && chunk.PunchSerial != 0 && completed >= chunk.PunchSerial)
                {
                    if (chunk.SettledAt == 0)
                    {
                        chunk.SettledAt = now;
                    }

                    if (now - chunk.SettledAt >= grace && !chunk.Record.Busy)
                    {
                        due ??= new();
                        if (!due.Contains(chunk.Record))
                        {
                            due.Add(chunk.Record);
                        }
                    }
                }
                else if (chunk.State == ChunkState.Emptied && chunk.NeedsReEmpty && now - chunk.EmptiedAt >= grace &&
                    !chunk.Record.Busy)
                {
                    (reEmpty ??= new()).Add(chunk);
                }
            }
        }

        if (due is not null)
        {
            foreach (var record in due)
            {
                EmptyRecord(record, now, grace);
            }
        }

        if (reEmpty is not null)
        {
            foreach (var chunk in reEmpty)
            {
                ReEmpty(chunk, now);
            }
        }

        return due is not null || reEmpty is not null;
    }

    private void EmptyRecord(Record record, long now, long grace)
    {
        // Still the texture it was armed from, clean, and the chunks still mapped the same way.
        var image = record.Image;
        var current = image is not null && image.Registered && image.SingleCopyRecord == record &&
            image.ContentVersion == record.ContentVersion && !image.IsCpuDirty && !image.IsGpuModified;
        var eligible = new List<Chunk>();
        lock (_m)
        {
            if (record.Gone || record.Busy)
            {
                return;
            }

            foreach (var chunk in record.Chunks)
            {
                if (chunk.State == ChunkState.Pending && chunk.PunchSerial != 0 && chunk.SettledAt != 0 &&
                    now - chunk.SettledAt >= grace)
                {
                    eligible.Add(chunk);
                }
            }
        }

        if (eligible.Count == 0)
        {
            return;
        }

        if (current)
        {
            foreach (var chunk in eligible)
            {
                if (chunk.Address == 0 || _state.ChunkViews(chunk.Index) != 1 || !_memory.IsSingleCopyHole(chunk.Index) ||
                    !_backing.TryResolveBackingAlias(chunk.Address, _chunkBytes, out var alias) ||
                    alias != _aliasBase + chunk.Offset)
                {
                    current = false;
                    break;
                }
            }
        }

        if (!current)
        {
            Interlocked.Increment(ref _verifyRejects);
            RestoreRecord(record, GuestSingleCopyAccess.Teardown, penalize: true, count: false);
            return;
        }

        lock (_m)
        {
            if (record.Gone || record.Busy)
            {
                return;
            }

            eligible.RemoveAll(static chunk => chunk.State != ChunkState.Pending);
            foreach (var chunk in eligible)
            {
                chunk.State = ChunkState.Emptying;
            }

            record.Busy = true;
        }

        var rejected = false;
        var emptiedCount = 0;
        try
        {
            var staging = Retile(record, out var contents);
            try
            {
                // Chunks next to each other in the backing are emptied in one call: the backing's
                // purgeable objects hold several chunks, and an object emptied whole is one EMPTY
                // where one emptied a chunk at a time is wired and unwired again for every chunk.
                var run = new List<Chunk>();
                foreach (var chunk in eligible)
                {
                    var expected = new ReadOnlySpan<byte>((byte*)contents + chunk.RunOffset, (int)_chunkBytes);
                    if (!_state.Holds(chunk.Offset, expected))
                    {
                        rejected = true;
                        chunk.State = ChunkState.Pending;
                        emptiedCount += EmptyRun(run, now);
                        continue;
                    }

                    if (_shadow)
                    {
                        // Verified, and counted as released, but the bytes stay: never in a run to empty.
                        chunk.EmptiedAt = now;
                        chunk.State = ChunkState.Emptied;
                        emptiedCount++;
                        continue;
                    }

                    if (SentinelMode)
                    {
                        _state.Fill(chunk.Offset, _chunkBytes, ReleasedSentinel);
                        chunk.EmptiedAt = now;
                        chunk.State = ChunkState.Emptied;
                        emptiedCount++;
                        continue;
                    }

                    if (run.Count != 0 && run[^1].Offset + _chunkBytes != chunk.Offset)
                    {
                        emptiedCount += EmptyRun(run, now);
                    }

                    run.Add(chunk);
                }

                emptiedCount += EmptyRun(run, now);
            }
            finally
            {
                ReleaseRetileStaging(staging);
            }
        }
        finally
        {
            lock (_m)
            {
                _emptiedBytes += (long)((ulong)emptiedCount * _chunkBytes);
                _discards += emptiedCount;
                record.Busy = false;
                Monitor.PulseAll(_m);
            }
        }

        if (rejected)
        {
            Interlocked.Increment(ref _verifyRejects);
            RestoreRecord(record, GuestSingleCopyAccess.Teardown, penalize: true, count: false);
        }
    }

    // Empties verified chunks that follow each other in the backing with one call and clears
    // the list; the chunks emptied. Some bytes zero-filled instead (a GPU use still wired them)
    // marks every chunk of the run for a later re-empty: which one they were is not reported.
    private int EmptyRun(List<Chunk> run, long now)
    {
        if (run.Count == 0)
        {
            return 0;
        }

        // The shadow mode never gives pages back, whatever path a chunk took here.
        var reEmpty = !_shadow && _state.Empty(run[0].Offset, (ulong)run.Count * _chunkBytes).ZeroFilledBytes != 0;
        foreach (var chunk in run)
        {
            chunk.NeedsReEmpty |= reEmpty;
            chunk.EmptiedAt = now;
            chunk.State = ChunkState.Emptied;
        }

        var count = run.Count;
        run.Clear();
        return count;
    }

    // A chunk that was still resident when it was emptied got zero-filled instead (it was
    // wired by a GPU use); now that the grace has passed it is emptied for real.
    private void ReEmpty(Chunk chunk, long now)
    {
        var record = chunk.Record;
        lock (_m)
        {
            if (record.Gone || record.Busy || chunk.State != ChunkState.Emptied || !_chunks.ContainsKey(chunk.Index))
            {
                return;
            }

            record.Busy = true;
        }

        try
        {
            chunk.NeedsReEmpty = !SentinelMode && !_shadow && _state.Empty(chunk.Offset, _chunkBytes).ZeroFilledBytes != 0;
            chunk.EmptiedAt = now;
        }
        finally
        {
            lock (_m)
            {
                _reEmpties++;
                record.Busy = false;
                Monitor.PulseAll(_m);
            }
        }
    }

    // Whether it dequeued any.
    private bool DrainUnwatches()
    {
        var any = false;
        while (_unwatch.TryDequeue(out var address))
        {
            any = true;
            lock (_w)
            {
                TryUnwatchLocked(address);
            }
        }

        return any;
    }

    // Caller holds W. Removes the watch at a chunk address once no released chunk is there.
    private bool TryUnwatchLocked(ulong address)
    {
        if (!_watched.Contains(address))
        {
            return false;
        }

        lock (_m)
        {
            if (_byAddress.ContainsKey(address))
            {
                return false;
            }
        }

        _watched.Remove(address);
        if (_backing.IsBackedRange(address, _chunkBytes))
        {
            _pages.RemoveWatch(address, _chunkBytes, blockReads: true);
        }

        return true;
    }

    // ---- The retile ------------------------------------------------------------------

    // The record's run, retiled from its texture into zeroed shared staging, waited for.
    // The caller releases the returned buffer.
    private nint Retile(Record record, out nint contents)
    {
        var runSize = record.RunEnd - record.RunStart;
        lock (_gpuLock)
        {
            using var pool = MetalAutoreleasePool.Enter();
            var staging = TakeRetileStaging((runSize + 3) & ~3UL);
            if (staging == 0)
            {
                throw GpuFatal.Fatal($"Single-copy textures could not allocate 0x{runSize:X} bytes of retile staging.");
            }

            var commandBuffer = MetalNative.Send(MetalNative.Send(_queue, SelCommandBuffer), SelRetain);
            var blit = MetalNative.Send(commandBuffer, SelBlit);
            MetalNative.SendFillBuffer(blit, SelFillBuffer, staging,
                new NsRange { Location = 0, Length = (nuint)((runSize + 3) & ~3UL) }, 0);
            MetalNative.SendVoid(blit, SelEndEncoding);
            _context.CommandBuffer = commandBuffer;
            var description = record.Description;
            var range = new SubresourceRange(
                record.FirstLevel, description.Resources.Levels - record.FirstLevel, 0, description.TransferLayers);
            _transfer.EncodeTextureToGuest(_context, in description, range, record.Texture,
                new MetalGuestRange(staging, 0, runSize, 0), default, record.RunStart, runSize, record.FirstLevel);
            MetalNative.SendVoid(commandBuffer, SelCommit);
            MetalNative.SendVoid(commandBuffer, SelWait);
            var status = (ulong)MetalV2Native.SendValue(commandBuffer, SelStatus);
            MetalNative.SendVoid(commandBuffer, SelRelease);
            _context.CommandBuffer = 0;
            _context.CompletedSerial = _context.SubmissionSerial;
            _context.SubmissionSerial++;
            _transfer.Tiler.TrimStaging(_context.CompletedSerial);
            if (status != StatusCompleted)
            {
                throw GpuFatal.Fatal(
                    $"A single-copy retile did not complete (status {status}): address=0x{description.Data.Address:X16} " +
                    $"run=0x{record.RunStart:X}..0x{record.RunEnd:X}.");
            }

            contents = MetalNative.Send(staging, SelContents);
            return staging;
        }
    }

    // ---- Restore ---------------------------------------------------------------------

    public void Restore(ulong offset, ulong size, GuestSingleCopyAccess access)
    {
        foreach (var record in RecordsIn(offset, size))
        {
            RestoreRecord(record, access, penalize: access != GuestSingleCopyAccess.Teardown, count: true);
        }
    }

    /// <summary>Whether the global fallback has run this session: nothing is released from then on.</summary>
    public bool DeviceAddressSeen => Volatile.Read(ref _deviceAddressSeen);

    /// <summary>A submission that runs device-address programs (the census's denominator).</summary>
    public void NoteDeviceAddressSubmission() => Interlocked.Increment(ref _daSubmissions);

    /// <summary>
    /// A device-address program is about to be prepared (<paramref name="late"/>: it is being
    /// prepared), with the ranges the host evaluated for its handles. Every chunk a planned range
    /// covers - exactly, or from its base to the cap for a handle with a run-time offset - is
    /// pinned (never released from now on) and, if released, given its bytes back now, so the
    /// program reads the guest's bytes; the host then ends the command buffer if a window it used
    /// still maps a hole. A program whose reach the host cannot bound - a handle nothing evaluated,
    /// or a store with no evaluated range - and every program once the title's kill switch is set
    /// take the global fallback (<see cref="NoteDeviceAddressUse"/>). An access past every range
    /// (beyond the cap, a device-loaded descriptor) is left to the shader's trap
    /// (<see cref="OnTrapHits"/>). Returns whether any chunk was given back; a late one is counted
    /// as stale (it was too late to end the command buffer).
    /// </summary>
    /// <param name="mappingVersion">Changes whenever guest mappings do; ranges seen before are pinned already.</param>
    /// <param name="mappedSize">The bytes from an address that are mapped, at most the size given (0: none).</param>
    public bool BeforeDeviceAddressProgram(
        bool writesDeviceAddresses,
        ReadOnlySpan<DeviceAddressRange> ranges,
        long mappingVersion,
        Func<ulong, ulong, ulong> mappedSize,
        bool late)
    {
        var unplanned = 0;
        var plannedWrite = false;
        foreach (var range in ranges)
        {
            if (!range.Planned)
            {
                unplanned++;
            }
            else if (range.Written)
            {
                plannedWrite = true;
            }
        }

        if (!late)
        {
            Interlocked.Increment(ref _daPrograms);
            var capped = 0;
            foreach (var range in ranges)
            {
                if (range.Planned && range.Size == DeviceAddressRangePlanner.MaxRangeBytes)
                {
                    capped++;
                }
            }

            Interlocked.Add(ref _daRangesCapped, capped);
            Interlocked.Add(ref _daRangesExact, ranges.Length - unplanned - capped);
            Interlocked.Add(ref _daUnplannedHandles, unplanned);
            if (writesDeviceAddresses && !plannedWrite)
            {
                Interlocked.Increment(ref _daHiddenWrites);
            }

            if (ranges.Length == 0)
            {
                Interlocked.Increment(ref _daInvisiblePrograms);
            }
        }

        string? fallback = !_shadow && !DeviceAddressPins ? "evaluated ranges are off: SHARPEMU_METAL_V2_SINGLE_COPY_DA=1 turns them on" :
            KillSwitch?.IsSet == true ? "the title's kill switch is set" :
            unplanned != 0 ? "a handle has no evaluated range" :
            writesDeviceAddresses && !plannedWrite ? "it stores where no evaluated range says" :
            DeviceAddressSeen ? "fallback already taken" : null;
        if (fallback is not null)
        {
            if (!late && !DeviceAddressSeen)
            {
                Interlocked.Increment(ref _daFallbacks);
            }

            return NoteDeviceAddressUse(late, fallback);
        }

        List<int>? restore = null;
        lock (_pinnedRanges)
        {
            if (_pinnedRangesVersion != mappingVersion || _pinnedRanges.Count > 1 << 20)
            {
                _pinnedRanges.Clear();
                _pinnedRangesVersion = mappingVersion;
            }
        }

        foreach (var range in ranges)
        {
            if (!range.Planned || range.Size == 0 || range.Base == 0)
            {
                continue;
            }

            lock (_pinnedRanges)
            {
                if (_pinnedRanges.Contains((range.Base, range.Size)))
                {
                    continue;
                }
            }

            var size = mappedSize(range.Base, range.Size);
            if (size != 0)
            {
                PinGuestRange(range.Base, size, range.Size == DeviceAddressRangePlanner.MaxRangeBytes ? PinCapped : PinExact, ref restore);
            }

            lock (_pinnedRanges)
            {
                _pinnedRanges.Add((range.Base, range.Size));
            }
        }

        if (restore is null)
        {
            return false;
        }

        GiveBack(restore);
        if (late)
        {
            Interlocked.Increment(ref _daLatePins);
            Interlocked.Increment(ref _deviceAddressStale);
        }

        return true;
    }

    /// <summary>
    /// The shader's trap saw device-address accesses to chunks the table flagged (accesses the
    /// evaluated ranges did not cover): <paramref name="hits"/> of them, to these backing pages.
    /// Those chunks are given their bytes back and pinned. In the shadow mode nothing was lost;
    /// otherwise each such access read zeros, which cannot be undone, so the title's kill switch
    /// is set for the next start.
    /// </summary>
    public void OnTrapHits(long hits, IReadOnlyList<ulong> pages)
    {
        if (hits == 0 && pages.Count == 0)
        {
            return;
        }

        if (_shadow)
        {
            Interlocked.Add(ref _daShadowHits, hits);
        }
        else
        {
            Interlocked.Add(ref _daReleasedHits, hits);
        }

        List<int>? restore = null;
        var chunks = new SortedSet<int>();
        foreach (var page in pages)
        {
            var chunk = (page << DeviceAddressPaging.PageBits) / _chunkBytes;
            if (chunk < (ulong)_state.ChunkCount)
            {
                chunks.Add((int)chunk);
            }
        }

        foreach (var chunk in chunks)
        {
            if (PinChunk(chunk, PinTrap))
            {
                (restore ??= new()).Add(chunk);
            }
        }

        if (restore is not null)
        {
            GiveBack(restore);
        }

        var listed = string.Join(',', chunks.Take(8).Select(chunk => $"0x{(ulong)chunk * _chunkBytes:X}"));
        if (_shadow)
        {
            MetalV2Diagnostics.WarnOnce(
                "single-copy-shadow-hit",
                $"A device-address program read a chunk single copy would have released (shadow mode, nothing lost): " +
                $"hits={hits} chunks={chunks.Count} offsets=[{listed}].");
            return;
        }

        MetalV2Diagnostics.WarnOnce(
            "single-copy-released-hit",
            $"A device-address program read a released single-copy chunk outside every evaluated range; those accesses " +
            $"read zeros. The chunks are back and pinned, and the title's kill switch is set: from the next start it " +
            $"releases nothing once a device-address program runs ({KillSwitch?.Path ?? "no profile directory"}): " +
            $"hits={hits} chunks={chunks.Count} offsets=[{listed}].");
        if (DeviceAddressPins)
        {
            // Without evaluated ranges the title takes the fallback anyway; nothing to switch off.
            KillSwitch?.Trip(string.Create(CultureInfo.InvariantCulture,
                $"released hit: hits={hits} chunks={chunks.Count} offsets=[{listed}]"));
        }
    }

    // Pins every chunk backing guest [address, address + size) (mapped), collecting the released ones.
    private void PinGuestRange(ulong address, ulong size, byte why, ref List<int>? restore)
    {
        var end = address + size;
        if (_backing.TryResolveBackingAliasSpan(address, size, out var span) && span >= _aliasBase)
        {
            PinBackingRange(span - _aliasBase, size, why, ref restore);
            return;
        }

        while (address < end)
        {
            var offset = 0UL;
            var length = 0UL;
            if (_backing.TryResolveBackingAlias(address, 1, out var alias) && alias >= _aliasBase)
            {
                offset = alias - _aliasBase;
                length = Math.Min((offset / _chunkBytes + 1) * _chunkBytes - offset, end - address);
                if (!_backing.TryResolveBackingAlias(address, length, out var whole) || whole != alias)
                {
                    // The mapping changes inside the chunk: a page at a time.
                    length = Math.Min(0x1000 - (address & 0xFFF), end - address);
                }
            }
            else
            {
                address = (address | 0xFFF) + 1;
                continue;
            }

            PinBackingRange(offset, length, why, ref restore);
            address += length;
        }
    }

    private void PinBackingRange(ulong offset, ulong size, byte why, ref List<int>? restore)
    {
        if (size == 0 || offset >= _state.Size)
        {
            return;
        }

        var first = (int)(offset / _chunkBytes);
        var last = (int)((Math.Min(offset + size, _state.Size) - 1) / _chunkBytes);
        for (var chunk = first; chunk <= last; chunk++)
        {
            if (PinChunk(chunk, why))
            {
                (restore ??= new()).Add(chunk);
            }
        }
    }

    // Pins a chunk; true when it is released (or being armed) and has to be given back.
    private bool PinChunk(int chunk, byte why)
    {
        var fresh = _state.Pin(chunk);
        bool armed;
        lock (_m)
        {
            armed = _chunks.ContainsKey(chunk);
            var flags = _pinFlags[chunk];
            if ((flags & why) == 0)
            {
                _pinFlags[chunk] = (byte)(flags | why);
                switch (why)
                {
                    case PinExact:
                        _pinnedExact++;
                        break;
                    case PinCapped:
                        _pinnedCapped++;
                        if ((flags & PinCandidate) != 0)
                        {
                            _pinnedCappedCandidates++;
                        }

                        break;
                    default:
                        _pinnedTrap++;
                        if (_shadow)
                        {
                            _daShadowHitChunks++;
                        }
                        else
                        {
                            _daReleasedHitChunks++;
                        }

                        break;
                }
            }

            if (fresh)
            {
                _pinnedChunks++;
            }

            if (armed || _everReleased[chunk] != 0)
            {
                NotePinCandidateLocked(chunk);
            }
        }

        return armed || _state.IsReleased(chunk);
    }

    // Caller holds M: a pinned chunk single copy wanted (armed, released once, or a candidate refused), counted once.
    private void NotePinCandidateLocked(int chunk)
    {
        var flags = _pinFlags[chunk];
        if ((flags & PinCandidate) != 0)
        {
            return;
        }

        _pinFlags[chunk] = (byte)(flags | PinCandidate);
        _pinnedCandidates++;
        if ((flags & PinCapped) != 0)
        {
            _pinnedCappedCandidates++;
        }
    }

    // The chunks a device-address program may read get their bytes and GPU alias pages back.
    private void GiveBack(List<int> chunks)
    {
        foreach (var chunk in chunks)
        {
            Restore((ulong)chunk * _chunkBytes, _chunkBytes, GuestSingleCopyAccess.DeviceAddress);
        }

        _memory.ApplySingleCopyFills();
    }

    /// <summary>
    /// The global fallback: a device-address program whose reach the host cannot bound is about
    /// to be prepared. It can reach any mapped guest page through the page table - the reach set
    /// only says which windows are named resident, and an access outside it still lands - so no
    /// chunk it could reach may be a hole while it runs, and none can be told apart from the
    /// rest: from now on nothing is armed, and every chunk released or punched so far gets its
    /// bytes and its GPU alias pages back before the preparation begins. <paramref name="late"/>
    /// is the in-preparation check; it finding anything to give back means a program got past
    /// the early one, and is counted. Returns whether anything was given back.
    /// </summary>
    public bool NoteDeviceAddressUse(bool late, string reason = "a device-address program ran")
    {
        if (!Volatile.Read(ref _deviceAddressSeen))
        {
            Volatile.Write(ref _deviceAddressSeen, true);
            Console.Error.WriteLine(
                $"[METAL-V2][INFO] single-copy textures: a device-address program ran; no chunk is released from now on ({reason}) " +
                $"(released_chunks={_state.ReleasedChunks} gpu_holes={_memory.SingleCopyHoleCount}).");
        }

        if (_state.ReleasedChunks == 0 && _memory.SingleCopyHoleCount == 0)
        {
            lock (_m)
            {
                if (_records.Count == 0)
                {
                    return false;
                }
            }
        }

        if (late)
        {
            Interlocked.Increment(ref _deviceAddressStale);
        }

        List<Record> records;
        lock (_m)
        {
            records = _records.ToList();
        }

        foreach (var record in records)
        {
            RestoreRecord(record, GuestSingleCopyAccess.GpuBinding, penalize: false, count: true);
        }

        Interlocked.Increment(ref _deviceAddressRestores);
        _memory.ApplySingleCopyFills();
        if (_memory.SingleCopyHoleCount != 0)
        {
            // Only a fill into a chunk still running (BeginOverwrite) keeps its hole: the guest
            // is writing those bytes while the program reads them.
            Interlocked.Increment(ref _deviceAddressStale);
        }

        return true;
    }

    private List<Record> RecordsIn(ulong offset, ulong size)
    {
        var records = new List<Record>();
        if (size == 0)
        {
            return records;
        }

        var first = (int)(offset / _chunkBytes);
        var last = (int)((Math.Min(offset + size, _state.Size) - 1) / _chunkBytes);
        lock (_m)
        {
            for (var index = first; index <= last; index++)
            {
                if (_chunks.TryGetValue(index, out var chunk) && !records.Contains(chunk.Record))
                {
                    records.Add(chunk.Record);
                }
            }
        }

        return records;
    }

    // Gives every chunk of the record its bytes back and drops the record. One retile for
    // all of them; a chunk not emptied yet has its bytes and is only forgotten.
    private void RestoreRecord(Record record, GuestSingleCopyAccess access, bool penalize, bool count)
    {
        List<Chunk> emptied;
        lock (_m)
        {
            while (record.Busy)
            {
                Monitor.Wait(_m);
            }

            if (record.Gone)
            {
                return;
            }

            record.Busy = true;
            emptied = record.Chunks.Where(static chunk => chunk.State == ChunkState.Emptied).ToList();
        }

        var started = Stopwatch.GetTimestamp();
        var restored = 0UL;
        try
        {
            if (_shadow)
            {
                // Nothing was emptied: the bytes are where they were.
                restored = (ulong)emptied.Count * _chunkBytes;
            }
            else if (emptied.Count != 0)
            {
                var staging = Retile(record, out var contents);
                try
                {
                    foreach (var chunk in emptied)
                    {
                        _state.WriteRestored(chunk.Offset,
                            new ReadOnlySpan<byte>((byte*)contents + chunk.RunOffset, (int)_chunkBytes));
                        restored += _chunkBytes;
                    }
                }
                finally
                {
                    ReleaseRetileStaging(staging);
                }
            }
        }
        finally
        {
            List<Chunk> forgotten;
            lock (_m)
            {
                forgotten = record.Chunks.ToList();
                var now = Stopwatch.GetTimestamp();
                foreach (var chunk in forgotten)
                {
                    if (penalize)
                    {
                        Penalize(chunk.Index, access, now);
                    }

                    ForgetChunkLocked(chunk);
                }

                RetireRecordLocked(record);
                if (count && emptied.Count != 0)
                {
                    _restoresBy[(int)access]++;
                    _restoredBytes += (long)restored;
                    _restoreTicks += Stopwatch.GetTimestamp() - started;
                }

                record.Busy = false;
                Monitor.PulseAll(_m);
            }

            AfterForget(forgotten);
            ReleaseTexture(record);
        }
    }

    // Caller holds M. A chunk restored for the GPU stays in place from now on; one the
    // CPU wanted back waits longer each time before it is armed again.
    private void Penalize(int index, GuestSingleCopyAccess access, long now)
    {
        if (access == GuestSingleCopyAccess.DeviceAddress)
        {
            // The chunk a device-address program may read is pinned; the record's others may
            // be armed again.
            return;
        }

        if (access == GuestSingleCopyAccess.GpuBinding)
        {
            _noRelease.Add(index);
            return;
        }

        var strikes = _cooldown.TryGetValue(index, out var entry) ? Math.Min(entry.Strikes + 1, 6) : 0;
        _cooldown[index] = (now + (Ticks(_settings.Cooldown) << strikes), strikes);
    }

    // Caller holds M: the chunk is no longer released (its bytes are in place or no longer matter).
    private void ForgetChunkLocked(Chunk chunk)
    {
        chunk.Cancelled = true;
        if (_chunks.TryGetValue(chunk.Index, out var current) && current == chunk)
        {
            _chunks.Remove(chunk.Index);
            _state.ClearReleased(chunk.Index);
            UncountLocked(chunk);
        }

        if (chunk.Address != 0 && _byAddress.TryGetValue(chunk.Address, out var mapped) && mapped == chunk)
        {
            _byAddress.Remove(chunk.Address);
        }

        chunk.Record.Chunks.Remove(chunk);
    }

    // Caller holds M: the chunk's bytes leave the released totals.
    private void UncountLocked(Chunk chunk)
    {
        if (chunk.State == ChunkState.Emptied)
        {
            _emptiedBytes -= (long)_chunkBytes;
        }

        if (chunk.Orphan)
        {
            chunk.Orphan = false;
            _orphanBytes -= (long)_chunkBytes;
        }
    }

    // Outside M: the hole is filled by the next submission (or the next binding over it),
    // and the watch goes once no thread holds the gate.
    private void AfterForget(List<Chunk> chunks)
    {
        foreach (var chunk in chunks)
        {
            _memory.QueueSingleCopyFill(chunk.Index);
            if (chunk.Address != 0)
            {
                _unwatch.Enqueue(chunk.Address);
            }
        }
    }

    // Caller holds M.
    private void RetireRecordLocked(Record record)
    {
        if (record.Gone)
        {
            return;
        }

        record.Gone = true;
        _records.Remove(record);
        if (record.Image is { } image && image.SingleCopyRecord == record)
        {
            image.SingleCopyRecord = null;
        }
    }

    private void DropRecordIfEmpty(Record record)
    {
        var drop = false;
        lock (_m)
        {
            if (!record.Gone && !record.Busy && record.Chunks.Count == 0)
            {
                RetireRecordLocked(record);
                drop = true;
            }
        }

        if (drop)
        {
            ReleaseTexture(record);
        }
    }

    private static void ReleaseTexture(Record record)
    {
        var texture = Interlocked.Exchange(ref record.Texture, 0);
        if (texture != 0)
        {
            MetalNative.SendVoid(texture, SelRelease);
        }
    }

    // ---- Overwrites and drops --------------------------------------------------------

    // Caller holds M: waits until no record with a released chunk in [first, last] is busy.
    private void WaitIdleLocked(int first, int last)
    {
        while (true)
        {
            var busy = false;
            for (var index = first; index <= last && !busy; index++)
            {
                busy = _chunks.TryGetValue(index, out var chunk) && chunk.Record.Busy;
            }

            if (!busy)
            {
                return;
            }

            Monitor.Wait(_m);
        }
    }

    public object? BeginOverwrite(ulong offset, ulong size)
    {
        if (size == 0)
        {
            return null;
        }

        var taken = new List<Chunk>();
        var first = (int)(offset / _chunkBytes);
        var last = (int)((Math.Min(offset + size, _state.Size) - 1) / _chunkBytes);
        lock (_m)
        {
            WaitIdleLocked(first, last);
            for (var index = first; index <= last; index++)
            {
                if (!_chunks.Remove(index, out var chunk))
                {
                    continue;
                }

                // Out of the released set now so the fill's own writes go through; the record
                // keeps the texture, busy, until the fill says how far it got.
                _state.ClearReleased(index);
                if (chunk.Address != 0 && _byAddress.TryGetValue(chunk.Address, out var mapped) && mapped == chunk)
                {
                    _byAddress.Remove(chunk.Address);
                }

                UncountLocked(chunk);
                chunk.Record.Busy = true;
                taken.Add(chunk);
            }
        }

        return taken.Count == 0 ? null : new Overwrite(taken);
    }

    public void EndOverwrite(object token, ulong offset, ulong written)
    {
        if (token is not Overwrite overwrite)
        {
            return;
        }

        var end = offset + written;
        foreach (var group in overwrite.Chunks.GroupBy(static chunk => chunk.Record))
        {
            var record = group.Key;
            var chunks = group.ToList();
            try
            {
                // An emptied chunk the fill stopped short of gets the rest of its bytes back.
                var short_ = _shadow
                    ? new List<Chunk>()
                    : chunks.Where(chunk => chunk.State == ChunkState.Emptied && end < chunk.Offset + _chunkBytes).ToList();
                if (short_.Count != 0)
                {
                    var staging = Retile(record, out var contents);
                    try
                    {
                        foreach (var chunk in short_)
                        {
                            var from = Math.Max(end, chunk.Offset);
                            var chunkEnd = chunk.Offset + _chunkBytes;
                            _state.WriteRestored(from, new ReadOnlySpan<byte>(
                                (byte*)contents + chunk.RunOffset + (from - chunk.Offset), (int)(chunkEnd - from)));
                        }
                    }
                    finally
                    {
                        ReleaseRetileStaging(staging);
                    }
                }
            }
            finally
            {
                var drop = false;
                lock (_m)
                {
                    foreach (var chunk in chunks)
                    {
                        chunk.Cancelled = true;
                        record.Chunks.Remove(chunk);
                        _overwrites++;
                    }

                    record.Busy = false;
                    if (record.Chunks.Count == 0)
                    {
                        RetireRecordLocked(record);
                        drop = true;
                    }

                    Monitor.PulseAll(_m);
                }

                AfterForget(chunks);
                if (drop)
                {
                    ReleaseTexture(record);
                }
            }
        }
    }

    public void Drop(ulong offset, ulong size)
    {
        if (size == 0)
        {
            return;
        }

        var forgotten = new List<Chunk>();
        var emptyRecords = new List<Record>();
        var first = (int)(offset / _chunkBytes);
        var last = (int)((Math.Min(offset + size, _state.Size) - 1) / _chunkBytes);
        lock (_m)
        {
            WaitIdleLocked(first, last);
            for (var index = first; index <= last; index++)
            {
                if (!_chunks.TryGetValue(index, out var chunk))
                {
                    continue;
                }

                ForgetChunkLocked(chunk);
                _drops++;
                forgotten.Add(chunk);
                if (chunk.Record.Chunks.Count == 0 && !chunk.Record.Gone)
                {
                    RetireRecordLocked(chunk.Record);
                    emptyRecords.Add(chunk.Record);
                }
            }
        }

        AfterForget(forgotten);
        foreach (var record in emptyRecords)
        {
            ReleaseTexture(record);
        }
    }

    // ---- Faults ----------------------------------------------------------------------

    public bool TryResolveFault(ulong address, bool write)
    {
        if (!_backing.TryResolveBackingAlias(address, 1, out var alias) || alias < _aliasBase)
        {
            return false;
        }

        var offset = alias - _aliasBase;
        var index = (int)(offset / _chunkBytes);
        if ((uint)index >= (uint)_everReleased.Length || Volatile.Read(ref _everReleased[index]) == 0)
        {
            return false;
        }

        var restored = false;
        if (_state.IsReleased(index))
        {
            Restore((ulong)index * _chunkBytes, _chunkBytes, GuestSingleCopyAccess.Fault);
            restored = true;
        }

        var chunkAddress = address - (offset - (ulong)index * _chunkBytes);
        bool unwatched;
        lock (_w)
        {
            unwatched = TryUnwatchLocked(chunkAddress);
        }

        // Another thread restored the chunk and took the watch while this access was faulting:
        // the page allows it now, so it is retried.
        return restored || unwatched || _pages.Allows(address, write ? FaultKind.Write : FaultKind.Read);
    }

    // ---- Image and mapping events ----------------------------------------------------

    /// <summary>
    /// The image's texture is about to change (a GPU write, an upload landing, a new
    /// texture): its released chunks get their bytes back from the texture as it is now.
    /// </summary>
    public void OnContentChanging(MetalCachedImage image)
    {
        if (image.SingleCopyRecord is { } record)
        {
            RestoreRecord(record, GuestSingleCopyAccess.Teardown, penalize: false, count: true);
        }
    }

    /// <summary>
    /// The cache drops the image (image lock held). Its chunks get their bytes back first,
    /// except after an unmap, when chunks no guest address maps any more keep the texture
    /// (up to a cap) until the backing is freed or mapped again.
    /// </summary>
    public void OnImageReleased(MetalCachedImage image, MetalImageReleaseCause cause)
    {
        if (image.SingleCopyRecord is not { } record)
        {
            return;
        }

        if (cause == MetalImageReleaseCause.Unmapped)
        {
            lock (_m)
            {
                while (record.Busy)
                {
                    Monitor.Wait(_m);
                }

                var chunks = record.Chunks.ToList();
                var orphanBytes = (long)((ulong)chunks.Count(static chunk => chunk.State == ChunkState.Emptied) * _chunkBytes);
                if (!record.Gone && chunks.Count != 0 && chunks.All(static chunk => chunk.Address == 0) &&
                    chunks.All(static chunk => chunk.State == ChunkState.Emptied) &&
                    (ulong)(_orphanBytes + orphanBytes) <= _settings.MaxOrphanBytes)
                {
                    _orphanBytes += orphanBytes;
                    foreach (var chunk in chunks)
                    {
                        chunk.Orphan = true;
                    }

                    record.Image = null;
                    image.SingleCopyRecord = null;
                    return;
                }
            }
        }

        RestoreRecord(record, GuestSingleCopyAccess.Teardown, penalize: false, count: true);
    }

    /// <summary>
    /// The guest is about to unmap [address, address + size) (GPU worker, before the unmap):
    /// the watches there go while the pages are still mapped, and the chunks become orphans.
    /// </summary>
    public void OnGuestUnmap(ulong address, ulong size)
    {
        // The guest addresses of the ranges pinned so far may name other chunks from now on.
        lock (_pinnedRanges)
        {
            _pinnedRanges.Clear();
        }

        if (size == 0 || _records.Count == 0 && _watched.Count == 0)
        {
            return;
        }

        var end = address + size;
        lock (_w)
        {
            var watched = _watched.Where(chunkAddress => chunkAddress < end && chunkAddress + _chunkBytes > address).ToList();
            lock (_m)
            {
                foreach (var chunkAddress in watched)
                {
                    if (_byAddress.Remove(chunkAddress, out var chunk))
                    {
                        chunk.Address = 0;
                    }
                }
            }

            foreach (var chunkAddress in watched)
            {
                _watched.Remove(chunkAddress);
                if (_backing.IsBackedRange(chunkAddress, _chunkBytes))
                {
                    _pages.RemoveWatch(chunkAddress, _chunkBytes, blockReads: true);
                }
            }
        }
    }

    /// <summary>A new image is registered over guest bytes: the chunks there are restored.</summary>
    public void OnImageRegistered(ulong address, ulong size) => RestoreAddressRangeFor(address, size, GuestSingleCopyAccess.Teardown);

    private void RestoreAddressRangeFor(ulong address, ulong size, GuestSingleCopyAccess access)
    {
        if (_state.ReleasedChunks == 0 || size == 0)
        {
            return;
        }

        if (_backing.TryResolveBackingAliasSpan(address, size, out var span) && span >= _aliasBase)
        {
            Restore(span - _aliasBase, size, access);
            return;
        }

        var end = address + size;
        while (address < end)
        {
            if (_backing.TryResolveBackingAlias(address, 1, out var alias) && alias >= _aliasBase)
            {
                var offset = alias - _aliasBase;
                var chunkEnd = (offset / _chunkBytes + 1) * _chunkBytes;
                var length = Math.Min(chunkEnd - offset, end - address);
                if (_state.AnyReleased(offset, length))
                {
                    Restore(offset, length, access);
                }

                address += length;
            }
            else
            {
                address = (address | 0xFFF) + 1;
            }
        }
    }

    /// <summary>
    /// The guest has left: every watch goes now (the page guard is freed next) and nothing
    /// new is armed. Released chunks stay restorable through the host hooks.
    /// </summary>
    public void StopWatching()
    {
        lock (_w)
        {
            _stopped = true;
            foreach (var address in _watched)
            {
                if (_backing.IsBackedRange(address, _chunkBytes))
                {
                    _pages.RemoveWatch(address, _chunkBytes, blockReads: true);
                }
            }

            _watched.Clear();
        }

        while (_unwatch.TryDequeue(out _))
        {
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _state.Handler = null;
        List<Record> records;
        lock (_m)
        {
            records = _records.ToList();
        }

        foreach (var record in records)
        {
            ReleaseTexture(record);
        }

        lock (_retilePool)
        {
            foreach (var (buffer, _) in _retilePool)
            {
                MetalNative.SendVoid(buffer, SelRelease);
            }

            _retilePool.Clear();
            _retilePoolBytes = 0;
        }

        lock (_gpuLock)
        {
            _transfer.Dispose();
            MetalNative.SendVoid(_queue, SelRelease);
        }
    }
}
