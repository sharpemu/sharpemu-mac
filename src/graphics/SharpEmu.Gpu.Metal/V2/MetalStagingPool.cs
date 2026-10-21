// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Numerics;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// The shared-storage slabs the tiler stages detiles, swaps, repacks and write-backs
/// through. A slab holds bytes only the GPU reads and writes, and only inside the command
/// buffer it was lent to, so it returns to the pool once that command buffer's serial has
/// completed and its contents never matter after that.
/// </summary>
/// <remarks>
/// <para>Slabs were power-of-two classes, so a 1080p RGBA8 write-back (8.3 MB) took a
/// 16 MiB slab and a 33 MiB detile took 64 MiB, and a free slab of one class never served
/// another. During Demon's Souls' world load the pool reached 1226 MB, and it was trimmed only
/// by submission count, which barely moves during a load. Now:</para>
/// <list type="bullet">
/// <item>a slab is the request rounded up to an eighth of its power of two (64 KiB at
/// least), so it wastes at most 12.5%, and a free slab up to one step larger is reused;</item>
/// <item>a free slab is released once it has been idle for <see cref="IdleMilliseconds"/>, or
/// for <see cref="IdleSerials"/> serials and at least <see cref="SerialExpiryMilliseconds"/>,
/// keeping only a small warm set of small slabs;</item>
/// <item>free slabs never hold more than <see cref="IdleCapBytes"/>: past it the least
/// recently used are released at once;</item>
/// <item>a free slab idle for a moment is made purgeable (volatile), so the OS can take its
/// pages under pressure without our freeing it; it is made nonvolatile again when lent, and
/// a slab the OS emptied is replaced by a new buffer of the same size.</item>
/// </list>
/// </remarks>
internal sealed class MetalStagingPool : IDisposable
{
    internal const ulong MinimumSlabBytes = 64 * 1024;

    // A free slab is released after this many completed serials or milliseconds idle.
    internal const ulong IdleSerials = 64;
    internal const long IdleMilliseconds = 1000;

    // The serial count releases a slab only once it has also been idle this long. A slow
    // frame commits many command buffers: Black Myth: Wukong's ~50 a frame at 4 fps passed 64
    // serials in about a frame, so the slabs each frame stages through were released and made
    // again every frame or two (~5 a second), each release a trip into the driver.
    internal const long SerialExpiryMilliseconds = 500;

    // A free slab is made purgeable after this many completed serials or milliseconds idle.
    internal const ulong VolatileAfterSerials = 8;
    internal const long VolatileAfterMilliseconds = 100;

    // Idle slabs no larger than WarmSlabBytes stay, up to WarmBytes, for the per-frame traffic.
    internal const ulong WarmSlabBytes = 4UL * 1024 * 1024;
    internal const ulong WarmBytes = 16UL * 1024 * 1024;

    // Free (completed, not lent) slabs never hold more than this. A frame of Black Myth: Wukong's
    // cinematic stages through some 500-1000 MB of slabs; with 256 MB kept, the rest were released
    // and made again every frame. Idle slabs are purgeable, so the OS can still take the pages.
    internal const ulong IdleCapBytes = 1024UL * 1024 * 1024;

    // A trim with no newly completed serial runs at most this often.
    private const long TrimIntervalMilliseconds = 50;

    // MTLPurgeableState.
    internal const nint PurgeableNonVolatile = 2;
    internal const nint PurgeableVolatile = 3;
    internal const nint PurgeableEmpty = 4;

    private static readonly nint SelNewBuffer = MetalNative.Selector("newBufferWithLength:options:");
    private static readonly nint SelRelease = MetalNative.Selector("release");
    private static readonly nint SelSetPurgeableState = MetalNative.Selector("setPurgeableState:");

    private sealed class Slab
    {
        public nint Buffer;
        public ulong Size;
        public ulong Serial;
        // When a trim first saw it free; idle time counts from here, not from the lend,
        // so a slab a long command buffer held is not released the moment it completes.
        public long FreeAt;
        public bool SeenFree;
        public bool Volatile;
    }

    private readonly nint _device;

    // Ordered by size, so a request finds the smallest free slab that fits.
    private readonly List<Slab> _slabs = new();
    private readonly List<Slab> _scratch = new();
    private ulong _trimmedCompleted = ulong.MaxValue;
    private long _trimmedAt;
    private ulong _windowPeakBytes;
    private bool _disposed;

    public MetalStagingPool(nint device)
    {
        _device = device;
    }

    /// <summary>Milliseconds, monotonic; tests replace it to age slabs.</summary>
    internal Func<long> Clock { get; set; } = static () => Environment.TickCount64;

    /// <summary>Shared memory every slab holds, lent or free.</summary>
    public ulong TotalBytes { get; private set; }

    public uint SlabCount { get; private set; }

    /// <summary>Free slabs at the last trim: what the pool holds for nobody.</summary>
    public ulong IdleBytes { get; private set; }

    /// <summary>Free slabs the OS may purge right now.</summary>
    public ulong VolatileBytes { get; private set; }

    /// <summary>The most <see cref="TotalBytes"/> has been since startup.</summary>
    public ulong PeakBytes { get; private set; }

    public long SlabsCreated { get; private set; }

    public long SlabsReleased { get; private set; }

    /// <summary>Slabs the OS emptied while they were purgeable, replaced when lent again.</summary>
    public long SlabsPurged { get; private set; }

    /// <summary>The submission the slabs in <see cref="RecordingBytes"/> were handed to.</summary>
    public ulong RecordingSerial { get; private set; }

    /// <summary>Shared memory the slabs handed to <see cref="RecordingSerial"/> hold.</summary>
    public ulong RecordingBytes { get; private set; }

    /// <summary>The most <see cref="TotalBytes"/> has been since the last call.</summary>
    public ulong TakeWindowPeakBytes()
    {
        var peak = Math.Max(_windowPeakBytes, TotalBytes);
        _windowPeakBytes = TotalBytes;
        return peak;
    }

    /// <summary>The slab a request of this many bytes is given when no free one fits.</summary>
    internal static ulong SlabCapacity(ulong size)
    {
        var rounded = (size + 3) & ~3UL;
        if (rounded <= MinimumSlabBytes)
        {
            return MinimumSlabBytes;
        }

        var floor = 1UL << (63 - BitOperations.LeadingZeroCount(rounded));
        var granule = Math.Max(MinimumSlabBytes, floor >> 3);
        return (rounded + granule - 1) & ~(granule - 1);
    }

    /// <summary>
    /// A slab of at least <paramref name="size"/> bytes for the command buffer that signals
    /// <paramref name="submissionSerial"/>. It is not lent again before
    /// <paramref name="completedSerial"/> has reached that serial.
    /// </summary>
    public MetalBufferSpan Acquire(ulong submissionSerial, ulong completedSerial, ulong size)
    {
        if (size == 0)
        {
            throw GpuFatal.Fatal("A tiler staging request has no size.");
        }

        var now = Clock();
        Trim(completedSerial, now);

        // A slab handed out during this command buffer must not be handed out again, so
        // its serial always lands past the completed one even when the two are equal.
        var held = Math.Max(submissionSerial, completedSerial + 1);
        if (held != RecordingSerial)
        {
            RecordingSerial = held;
            RecordingBytes = 0;
        }

        var rounded = (size + 3) & ~3UL;
        var capacity = SlabCapacity(size);
        var limit = capacity + capacity / 8;
        for (var index = LowerBound(rounded); index < _slabs.Count && _slabs[index].Size <= limit; index++)
        {
            var slab = _slabs[index];
            if (slab.Serial > completedSerial)
            {
                continue;
            }

            if (slab.Volatile)
            {
                MakeNonVolatile(slab);
            }

            Lend(slab, held);
            return new MetalBufferSpan(slab.Buffer, 0, size);
        }

        var created = new Slab { Buffer = NewBuffer(capacity), Size = capacity };
        _slabs.Insert(LowerBound(capacity), created);
        TotalBytes += capacity;
        SlabCount++;
        SlabsCreated++;
        PeakBytes = Math.Max(PeakBytes, TotalBytes);
        _windowPeakBytes = Math.Max(_windowPeakBytes, TotalBytes);
        Lend(created, held);
        return new MetalBufferSpan(created.Buffer, 0, size);
    }

    /// <summary>Releases, caps and makes purgeable the free slabs; cheap when nothing changed.</summary>
    public void Trim(ulong completedSerial) => Trim(completedSerial, Clock());

    private void Trim(ulong completed, long now)
    {
        if (completed == _trimmedCompleted && now - _trimmedAt < TrimIntervalMilliseconds)
        {
            return;
        }

        _trimmedCompleted = completed;
        _trimmedAt = now;
        _scratch.Clear();
        foreach (var slab in _slabs)
        {
            if (slab.Serial <= completed)
            {
                if (!slab.SeenFree)
                {
                    slab.SeenFree = true;
                    slab.FreeAt = now;
                }

                _scratch.Add(slab);
            }
        }

        // Most recently used first: the warm set and the cap keep the newest, release the oldest.
        _scratch.Sort(static (a, b) => b.Serial != a.Serial ? b.Serial.CompareTo(a.Serial) : b.FreeAt.CompareTo(a.FreeAt));
        ulong kept = 0;
        ulong warm = 0;
        ulong volatileBytes = 0;
        foreach (var slab in _scratch)
        {
            var idleSerials = completed - slab.Serial;
            var idleMilliseconds = now - slab.FreeAt;
            var expired = idleMilliseconds >= IdleMilliseconds ||
                (idleSerials >= IdleSerials && idleMilliseconds >= SerialExpiryMilliseconds);
            var keep = expired
                ? slab.Size <= WarmSlabBytes && warm + slab.Size <= WarmBytes && kept + slab.Size <= IdleCapBytes
                : kept + slab.Size <= IdleCapBytes;
            if (!keep)
            {
                ReleaseSlab(slab);
                continue;
            }

            kept += slab.Size;
            if (expired)
            {
                warm += slab.Size;
            }

            if (!slab.Volatile && (idleSerials >= VolatileAfterSerials || idleMilliseconds >= VolatileAfterMilliseconds))
            {
                SetPurgeableState(slab.Buffer, PurgeableVolatile);
                slab.Volatile = true;
            }

            if (slab.Volatile)
            {
                volatileBytes += slab.Size;
            }
        }

        _scratch.Clear();
        IdleBytes = kept;
        VolatileBytes = volatileBytes;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        foreach (var slab in _slabs)
        {
            Release(slab.Buffer);
        }

        _slabs.Clear();
        TotalBytes = 0;
        SlabCount = 0;
        IdleBytes = 0;
        VolatileBytes = 0;
    }

    private void Lend(Slab slab, ulong held)
    {
        if (_trimmedCompleted != ulong.MaxValue && slab.Serial <= _trimmedCompleted)
        {
            // It was counted free at the last trim.
            IdleBytes -= Math.Min(IdleBytes, slab.Size);
        }

        slab.Serial = held;
        slab.SeenFree = false;
        RecordingBytes += slab.Size;
    }

    private void MakeNonVolatile(Slab slab)
    {
        var prior = SetPurgeableState(slab.Buffer, PurgeableNonVolatile);
        slab.Volatile = false;
        VolatileBytes -= Math.Min(VolatileBytes, slab.Size);
        if (prior == PurgeableEmpty)
        {
            // The OS took its pages. The bytes never mattered, but an emptied buffer is
            // not one to bind again: replace it with a new one of the same size.
            Release(slab.Buffer);
            slab.Buffer = NewBuffer(slab.Size);
            SlabsPurged++;
        }
    }

    private void ReleaseSlab(Slab slab)
    {
        if (slab.Volatile)
        {
            VolatileBytes -= Math.Min(VolatileBytes, slab.Size);
        }

        Release(slab.Buffer);
        _slabs.Remove(slab);
        TotalBytes -= slab.Size;
        SlabCount--;
        SlabsReleased++;
    }

    // The first slab of at least `size` bytes.
    private int LowerBound(ulong size)
    {
        int low = 0, high = _slabs.Count;
        while (low < high)
        {
            var middle = (low + high) >>> 1;
            if (_slabs[middle].Size < size)
            {
                low = middle + 1;
            }
            else
            {
                high = middle;
            }
        }

        return low;
    }

    private nint NewBuffer(ulong capacity)
    {
        var buffer = MetalNative.SendNewBuffer(_device, SelNewBuffer, (nuint)capacity, 0);
        if (buffer == 0)
        {
            throw GpuFatal.Fatal($"The tiler could not allocate a staging buffer: size={capacity}.");
        }

        return buffer;
    }

    /// <summary>-[MTLResource setPurgeableState:]; returns the state before the call.</summary>
    internal static nint SetPurgeableState(nint resource, nint state) =>
        MetalNative.Send(resource, SelSetPurgeableState, state);

    private static void Release(nint handle)
    {
        if (handle != 0)
        {
            MetalNative.SendVoid(handle, SelRelease);
        }
    }
}
