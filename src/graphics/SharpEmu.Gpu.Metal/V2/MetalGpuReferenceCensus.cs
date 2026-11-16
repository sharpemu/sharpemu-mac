// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using System.Runtime.InteropServices;
using SharpEmu.Logging;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>Why the GPU path referenced a stretch of guest memory.</summary>
internal enum MetalRefKind
{
    /// <summary>A buffer binding (vertex, index, constant, storage, indirect) or a GPU-writable range resolved in a window.</summary>
    Binding,

    /// <summary>Bytes copied into the upload ring: staged ranges, staged storage, copies of ranges no window covers.</summary>
    Staged,

    /// <summary>The guest source of an image upload.</summary>
    ImageUpload,

    /// <summary>The guest destination of an image writeback.</summary>
    ImageWriteback,
}

/// <summary>
/// SHARPEMU_LOG_MEMORY=1: which 2 MiB blocks of guest memory the GPU path actually names. Whole
/// 1 GiB windows are made resident (wired and faulted in) when an encoder names them, so the
/// distinct MiB here, against the windows named, say how small the resident set could be with
/// finer windows. Every exact range is marked (a binding's own offset and size, an image's upload
/// source or writeback destination, a staged copy); the spans a device-address program may reach
/// but nothing resolved are only counted, as the device-address-only figure. Blocks are keyed by
/// guest address, 2 MiB aligned, in chunks of 64 so a multi-gigabyte binding is a few mask writes.
/// Off, every entry point is a static readonly check.
/// </summary>
internal static class MetalGpuReferenceCensus
{
    internal const int BlockBits = 21;
    private static readonly MetalGpuReferenceTracker Shared = new();

    [ThreadStatic]
    private static int _suppressed;

    [ThreadStatic]
    private static int _scope;

    // The frame-overlap diagnostic reads the same marks (SHARPEMU_DIAG_FRAME_OVERLAP=1).
    public static bool Enabled => MemoryTelemetry.Enabled || SharpEmu.HLE.GpuMemory.FrameOverlapDiagnostic.Enabled;

    /// <summary>Marks the blocks of a guest range; a scope (image upload or writeback) overrides the kind.</summary>
    public static void Mark(ulong address, ulong size, MetalRefKind kind = MetalRefKind.Binding)
    {
        if (!Enabled || size == 0 || _suppressed != 0 || size > ulong.MaxValue - address)
        {
            return;
        }

        var effective = _scope != 0 ? (MetalRefKind)(_scope - 1) : kind;
        if (MemoryTelemetry.Enabled)
        {
            Shared.Record(effective, address, size);
        }

        if (SharpEmu.HLE.GpuMemory.FrameOverlapDiagnostic.Enabled)
        {
            if (effective == MetalRefKind.ImageWriteback)
            {
                SharpEmu.HLE.GpuMemory.FrameOverlapDiagnostic.NoteGpuWrite(address, size, SharpEmu.HLE.GpuMemory.OverlapWrite.Writeback);
            }
            else
            {
                SharpEmu.HLE.GpuMemory.FrameOverlapDiagnostic.MarkGpuRead(address, size, effective switch
                {
                    MetalRefKind.Staged => SharpEmu.HLE.GpuMemory.OverlapRead.Staged,
                    MetalRefKind.ImageUpload => SharpEmu.HLE.GpuMemory.OverlapRead.Upload,
                    _ => SharpEmu.HLE.GpuMemory.OverlapRead.InPlace,
                });
            }
        }
    }

    /// <summary>While alive, every mark on this thread counts as <paramref name="kind"/> (image paths).</summary>
    public static Scoped Scope(MetalRefKind kind)
    {
        if (!Enabled)
        {
            return default;
        }

        var previous = _scope;
        _scope = (int)kind + 1;
        return new Scoped(previous + 1);
    }

    /// <summary>While alive, marks on this thread are ignored (a scan that resolves every mapped span).</summary>
    public static Suppressed Suppress()
    {
        if (!Enabled)
        {
            return default;
        }

        _suppressed++;
        return new Suppressed(true);
    }

    public readonly struct Scoped : IDisposable
    {
        private readonly int _previousPlusOne;

        internal Scoped(int previousPlusOne)
        {
            _previousPlusOne = previousPlusOne;
        }

        public void Dispose()
        {
            if (_previousPlusOne != 0)
            {
                _scope = _previousPlusOne - 1;
            }
        }
    }

    public readonly struct Suppressed : IDisposable
    {
        private readonly bool _active;

        internal Suppressed(bool active)
        {
            _active = active;
        }

        public void Dispose()
        {
            if (_active)
            {
                _suppressed--;
            }
        }
    }

    /// <summary>The spans the device-address table maps: reachable by any program, exactly named by none.</summary>
    public static void SetDeviceAddressSpans(IEnumerable<(ulong Address, ulong Size)> spans)
    {
        if (MemoryTelemetry.Enabled)
        {
            Shared.SetDeviceAddressSpans(spans);
        }
    }

    /// <summary>One report: distinct MiB this interval and since boot, per kind and together, and the device-address-only MiB.</summary>
    public static string TakeReport() => Shared.TakeReport();
}

/// <summary>The census state; separate from the static switch so a test can drive it.</summary>
internal sealed class MetalGpuReferenceTracker
{
    private const int KindCount = 4;
    private static readonly string[] KindNames = ["bind", "staged", "image_up", "image_wb"];

    private readonly object _gate = new();
    private readonly MetalBlockSet[] _interval = NewSets();
    private readonly MetalBlockSet[] _cumulative = NewSets();
    private volatile MetalBlockSet _deviceAddress = new();
    private long _marks;
    private ulong _largest;

    private static MetalBlockSet[] NewSets()
    {
        var sets = new MetalBlockSet[KindCount];
        for (var index = 0; index < sets.Length; index++)
        {
            sets[index] = new MetalBlockSet();
        }

        return sets;
    }

    public void Record(MetalRefKind kind, ulong address, ulong size)
    {
        lock (_gate)
        {
            _interval[(int)kind].AddRange(address, size);
            _cumulative[(int)kind].AddRange(address, size);
            _marks++;
            _largest = Math.Max(_largest, size);
        }
    }

    public void SetDeviceAddressSpans(IEnumerable<(ulong Address, ulong Size)> spans)
    {
        var set = new MetalBlockSet();
        foreach (var (address, size) in spans)
        {
            set.AddRange(address, size);
        }

        _deviceAddress = set;
    }

    public string TakeReport()
    {
        var culture = CultureInfo.InvariantCulture;
        long marks;
        ulong largest;
        MetalBlockSet intervalUnion = new(), cumulativeUnion = new();
        var perKindInterval = new long[KindCount];
        var perKindCumulative = new long[KindCount];
        lock (_gate)
        {
            for (var kind = 0; kind < KindCount; kind++)
            {
                perKindInterval[kind] = _interval[kind].Blocks;
                perKindCumulative[kind] = _cumulative[kind].Blocks;
                intervalUnion.UnionWith(_interval[kind]);
                cumulativeUnion.UnionWith(_cumulative[kind]);
                _interval[kind].Clear();
            }

            marks = _marks;
            largest = _largest;
            _marks = 0;
            _largest = 0;
        }

        var deviceAddress = _deviceAddress;
        var deviceAddressOnlyInterval = deviceAddress.CountNotIn(intervalUnion);
        var deviceAddressOnlyCumulative = deviceAddress.CountNotIn(cumulativeUnion);
        var kinds = string.Join(' ', Enumerable.Range(0, KindCount).Select(kind =>
            string.Create(culture, $"{KindNames[kind]}={Mib(perKindInterval[kind])}/{Mib(perKindCumulative[kind])}")));
        return string.Create(culture,
            $"interval_mib={Mib(intervalUnion.Blocks)} cumulative_mib={Mib(cumulativeUnion.Blocks)} " +
            $"da_only_mib={Mib(deviceAddressOnlyInterval)}/{Mib(deviceAddressOnlyCumulative)} " +
            $"da_table_mib={Mib(deviceAddress.Blocks)} per_kind_interval/cumulative_mib=[{kinds}] " +
            $"marks={marks} largest_mark_mib={largest >> 20} block_mib=2");
    }

    private static long Mib(long blocks) => blocks << (MetalGpuReferenceCensus.BlockBits - 20);
}

/// <summary>A set of 2 MiB blocks as 64-block chunk masks.</summary>
internal sealed class MetalBlockSet
{
    private readonly Dictionary<ulong, ulong> _chunks = new();

    public long Blocks { get; private set; }

    public void Clear()
    {
        _chunks.Clear();
        Blocks = 0;
    }

    public void AddRange(ulong address, ulong size)
    {
        var first = address >> MetalGpuReferenceCensus.BlockBits;
        var last = (address + size - 1) >> MetalGpuReferenceCensus.BlockBits;
        for (var chunk = first >> 6; chunk <= last >> 6; chunk++)
        {
            var low = chunk == first >> 6 ? (int)(first & 63) : 0;
            var high = chunk == last >> 6 ? (int)(last & 63) : 63;
            var width = high - low + 1;
            var mask = (width == 64 ? ulong.MaxValue : (1UL << width) - 1) << low;
            Or(chunk, mask);
        }
    }

    public void UnionWith(MetalBlockSet other)
    {
        foreach (var (chunk, mask) in other._chunks)
        {
            Or(chunk, mask);
        }
    }

    public long CountNotIn(MetalBlockSet other)
    {
        var count = 0L;
        foreach (var (chunk, mask) in _chunks)
        {
            other._chunks.TryGetValue(chunk, out var covered);
            count += System.Numerics.BitOperations.PopCount(mask & ~covered);
        }

        return count;
    }

    private void Or(ulong chunk, ulong mask)
    {
        ref var slot = ref CollectionsMarshal.GetValueRefOrAddDefault(_chunks, chunk, out _);
        var added = mask & ~slot;
        if (added != 0)
        {
            slot |= added;
            Blocks += System.Numerics.BitOperations.PopCount(added);
        }
    }
}
