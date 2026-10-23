// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

// A resource materialization reads its descriptor tables one dword at a time, and each
// dword read through TryReadGuest asks the pending-write overlay, the image cache and the
// recorded GPU writes (twice, with a message send for the completed serial each time)
// whether it may have to wait: measured at about 2.5 us a word with 250 live write
// records, which made the walk a third of the Metal worker in Demon's Souls character
// creation. Inside one materialization a word is instead served from a 256-byte line read
// once, after the whole line passed the same tests a word read makes:
//   - no pending command-processor bytes over it (overlay coverage None), and
//   - no GPU-modified image bytes and no pending recorded write over it,
// so every word in it would have been read straight from memory without waiting. A line
// that fails is remembered as refused and its words take the ordinary path, which waits
// exactly as before. The tests only get weaker over time (the GPU completes, entries
// land) except through a recorded write, an owed CP write, a host write that forgets owed
// bytes, or a wait: each of those drops every line. Lines live only between
// BeginGuestWordReads and EndGuestWordReads on the thread that began them.
internal sealed partial class MetalRenderHost : IResourceDependencySource
{
    ResourceDependencyEpoch IResourceDependencySource.DependencyEpoch => new(
        _gpuWrites.Version, _cpOverlay.Version, DeviceAddressMappingVersion(), _imageCache.Epochs.Binding, _labelOverlay.Version);

    // Kill switch retains the original protected-image-token enrollment policy.
    internal bool ResourceGenerationsEnabled { get; set; } = SharpEmu.ShaderCompiler.PerfSwitches.Enabled("SHARPEMU_RESOURCE_GENERATIONS");
    bool IResourceDependencySource.UsesDependencyByteValidation => ResourceGenerationsEnabled && _backing.BackingChunkBytes != 0;
    long IResourceDependencySource.CapturePolicyVersion => ResourceGenerationsEnabled ? 1 : 0;
    long IResourceDependencySource.CaptureRetryVersion => (long)CompletedSerialHint;
    bool IResourceDependencySource.TryValidateDependencyBytes(ReadOnlySpan<ulong> addresses, ReadOnlySpan<int> lengths,
        ReadOnlySpan<int> offsets, ReadOnlySpan<byte> expected) => TryCompareResidentGuestBytes(addresses, lengths, offsets, expected);
    private ResourceDependencyFailure _dependencyFailure;
    ResourceDependencyFailure IResourceDependencySource.CaptureFailure => _dependencyFailure;
    private readonly long[,] _dependencyCaptures = new long[(int)ResourceRangeClass.Released + 1, (int)ResourceCaptureFailure.Unsupported + 1];
    private byte[] _dependencyCapture = [];
    private readonly Dictionary<(ulong Address, int Size), WeakReference<WordDependency>> _wordDependencies = new();

    internal string TakeDependencyCaptureReport()
    {
        var counts = new List<string>();
        for (var range = 0; range < _dependencyCaptures.GetLength(0); range++)
            for (var reason = 0; reason < _dependencyCaptures.GetLength(1); reason++)
            {
                var count = Interlocked.Exchange(ref _dependencyCaptures[range, reason], 0);
                if (count != 0) counts.Add($"{(ResourceRangeClass)range}/{(ResourceCaptureFailure)reason}:{count}");
            }
        return $"resmat_capture=[{string.Join(',', counts)}]";
    }

    private bool DependencyRefused(ResourceRangeClass range, ResourceCaptureFailure reason)
    {
        _dependencyFailure = new(range, reason);
        Interlocked.Increment(ref _dependencyCaptures[(int)range, (int)reason]);
        return false;
    }

    bool IResourceDependencySource.TryCaptureDependency(ulong address, ReadOnlySpan<byte> expected, bool clean,
        out ResourceDependency? dependency)
    {
        dependency = null;
        var size = (ulong)expected.Length;
        if (expected.IsEmpty || expected.Length > 65536 || !new GuestSpan(address, size).IsValid)
            return DependencyRefused(ResourceRangeClass.Unmapped, ResourceCaptureFailure.Range);
        var range = _stagedRanges?.MayCover(address, size) == true ? ResourceRangeClass.Staged :
            _backing.IsBackedRange(address, size)
                ? (_backing.BackingChunkBytes != 0 ? ResourceRangeClass.ChunkedBacking : ResourceRangeClass.GuestBacking)
                : _memory.CanRead(address, size) ? ResourceRangeClass.NativeMapped : ResourceRangeClass.Unmapped;
        if (_labelOverlay.Coverage(address, size, _completedSerialReader ??= () => CompletedSerial) != MetalOverlayCoverage.None ||
            OverlayCoverage(address, size) != MetalOverlayCoverage.None)
            return DependencyRefused(ResourceRangeClass.Overlay, ResourceCaptureFailure.Overlay);
        if (DependencyRangeReleased(address, size))
            return DependencyRefused(ResourceRangeClass.Released, ResourceCaptureFailure.Released);
        if (_imageCache.HasGpuModifiedImageBytes(address, size))
            return DependencyRefused(ResourceRangeClass.ImageBacking, ResourceCaptureFailure.ImageBacking);
        if (_gpuWrites.MayCover(address, size) && PendingSerialHinted(address, size) != 0)
            return DependencyRefused(range, ResourceCaptureFailure.PendingWriter);
        if (_memory is not ICpuMappedReadWatchSource source)
            return DependencyRefused(range, ResourceCaptureFailure.PhysicalWatch);

        var epoch = ((IResourceDependencySource)this).DependencyEpoch;
        var key = (address, expected.Length);
        // Share the SAME original observation across stages, never re-certify an old
        // snapshot with a new token. Global packet epochs do not expire an original
        // physical watch: the exact-range guards above and Validate decide whether
        // it can be shared. This also avoids exhausting the finite watch ledger.
        // Native faults must re-arm BEFORE copying bytes.
        if (_wordDependencies.TryGetValue(key, out var weak) && weak.TryGetTarget(out var existing) &&
            existing.Matches(expected) && existing.Validate() == ResourceDependencyStatus.Current)
        {
            existing.Retain();
            dependency = existing;
            Interlocked.Increment(ref _dependencyCaptures[(int)range, 0]);
            return true;
        }

        GuestWriteGenerations.Token? generation = null;
        var protectedToken = default(GuestImageWriteTracker.ContentToken);
        if (ResourceGenerationsEnabled && GuestGpuMemoryHook.Current is { } manager && manager.Covers(address, size))
            generation = manager.WriteGenerations.Register(address, size);
        else if (!GuestImageWriteTracker.TryGetProtectedObservedContentToken(address, size, out protectedToken))
            return DependencyRefused(range, ResourceCaptureFailure.NativeWatch);

        // Physical watches exclude managed writes admitted before capture, including
        // alias writes. Generation/native tokens close first-fault and released-page races.
        if (_dependencyCapture.Length < expected.Length) _dependencyCapture = new byte[expected.Length];
        var bytes = _dependencyCapture.AsSpan(0, expected.Length);
        if (source.TryReadMappedCopyWithWriteWatch(address, size, address, bytes,
            out var observation) != CpuMappedCopyResult.Copied || observation is null)
            return DependencyRefused(range, ResourceCaptureFailure.PhysicalWatch);
        if (!observation.ExcludesNativeAliases)
        {
            source.ReleaseMappedCopyWriteWatch(observation);
            return DependencyRefused(range, ResourceCaptureFailure.NativeAlias);
        }
        if (!bytes.SequenceEqual(expected) || !source.IsMappedCopyWriteWatchCurrent(observation) ||
            !(generation?.IsCurrent ?? (GuestImageWriteTracker.TryGetProtectedObservedContentToken(address, size, out var after) && protectedToken == after)) ||
            ((IResourceDependencySource)this).DependencyEpoch != epoch || DependencyRangeReleased(address, size))
        {
            source.ReleaseMappedCopyWriteWatch(observation);
            return DependencyRefused(range, ResourceCaptureFailure.Changed);
        }
        var captured = new WordDependency(source, observation, generation, protectedToken, expected.ToArray());
        if (_wordDependencies.Count >= 2048) _wordDependencies.Clear();
        _wordDependencies[key] = new(captured);
        dependency = captured;
        Interlocked.Increment(ref _dependencyCaptures[(int)range, 0]);
        return true;
    }

    private bool DependencyRangeReleased(ulong address, ulong size)
    {
        if (_backing.SingleCopy is not { ReleasedChunks: > 0 } single) return false;
        var (aliasBase, _) = _backing.BackingAliasRange;
        // An uncertain resolution must decline; unrelated released chunks no longer
        // veto every descriptor in the process.
        return !_backing.TryResolveBackingAlias(address, size, out var alias) || alias < aliasBase ||
            single.AnyReleased(alias - aliasBase, size);
    }

    private sealed class WordDependency(ICpuMappedReadWatchSource source, CpuMappedCopyObservation observation,
        GuestWriteGenerations.Token? generation, GuestImageWriteTracker.ContentToken token, byte[] expected) : ResourceDependency
    {
        private int _references = 1;
        public override long RetainedBytes => 512L + (generation?.RetainedBytes ?? 0) + expected.Length + observation.Segments.Count * 128L;
        internal bool Matches(ReadOnlySpan<byte> bytes) => _references != 0 && bytes.SequenceEqual(expected);
        internal void Retain() => _references++;
        public override ResourceDependencyStatus Validate()
        {
            if (_references == 0) return ResourceDependencyStatus.Unavailable;
            if (!source.IsMappedCopyObservationCurrent(observation) || !observation.ExcludesNativeAliases) return ResourceDependencyStatus.Mapping;
            if (!source.IsMappedCopyWriteWatchCurrent(observation)) return ResourceDependencyStatus.CpuWrite;
            return (generation?.IsCurrent ?? (GuestImageWriteTracker.TryGetProtectedObservedContentToken(observation.SourceAddress,
                observation.SourceByteCount, out var current) && token == current))
                ? ResourceDependencyStatus.Current : ResourceDependencyStatus.CpuWrite;
        }
        public override void Dispose()
        {
            if (_references != 0 && --_references == 0) source.ReleaseMappedCopyWriteWatch(observation);
        }
    }

    // SHARPEMU_RESOURCE_WIDE_LINES=0 keeps the 64-byte lines of 16 slots. A line costs the same
    // tests however wide it is, and a descriptor table or pointer ring is read from a few lines.
    private static readonly bool WideLines = SharpEmu.ShaderCompiler.PerfSwitches.Enabled("SHARPEMU_RESOURCE_WIDE_LINES");
    private static readonly int WordLineBytes = WideLines ? 256 : 64;
    private static readonly int WordLineSlots = WideLines ? 32 : 16;

    private readonly ulong[] _wordLineAddress = new ulong[WordLineSlots];
    private readonly ulong[] _wordLineGeneration = new ulong[WordLineSlots];
    private readonly bool[] _wordLineReadable = new bool[WordLineSlots];
    private readonly byte[] _wordLineBytes = new byte[WordLineSlots * WordLineBytes];
    private ulong _wordLineGenerationCurrent = 1;
    private ulong _wordLineGpuVersion;
    private long _wordLineOverlayVersion;
    private int _wordLineThread;

    private long _wordReads;
    private long _wordLineHits;
    private long _wordLineFills;
    private long _wordLineRefused;
    private long _wordReadScopes;

    public void BeginGuestWordReads()
    {
        InvalidateWordLines();
        _wordLineGpuVersion = _gpuWrites.Version;
        _wordLineOverlayVersion = _cpOverlay.Version;
        _wordLineThread = Environment.CurrentManagedThreadId;
        _wordReadScopes++;
    }

    public void EndGuestWordReads()
    {
        _wordLineThread = 0;
        InvalidateWordLines();
    }

    private void InvalidateWordLines() => _wordLineGenerationCurrent++;

    private bool TryReadWordFromLine(ulong address, out uint word)
    {
        word = 0;
        var offset = (int)(address & (ulong)(WordLineBytes - 1));
        if (offset > WordLineBytes - sizeof(uint))
        {
            return false;
        }

        var gpuVersion = _gpuWrites.Version;
        var overlayVersion = _cpOverlay.Version;
        if (gpuVersion != _wordLineGpuVersion || overlayVersion != _wordLineOverlayVersion)
        {
            _wordLineGpuVersion = gpuVersion;
            _wordLineOverlayVersion = overlayVersion;
            InvalidateWordLines();
        }

        var line = address - (ulong)offset;
        var slot = (int)((line / (ulong)WordLineBytes) & (ulong)(WordLineSlots - 1));
        var bytes = _wordLineBytes.AsSpan(slot * WordLineBytes, WordLineBytes);
        if (_wordLineAddress[slot] != line || _wordLineGeneration[slot] != _wordLineGenerationCurrent)
        {
            _wordLineAddress[slot] = line;
            _wordLineGeneration[slot] = _wordLineGenerationCurrent;
            _wordLineReadable[slot] = TryFillWordLine(line, bytes);
            _wordLineFills++;
        }

        if (!_wordLineReadable[slot])
        {
            _wordLineRefused++;
            return false;
        }

        _wordLineHits++;
        word = BinaryPrimitives.ReadUInt32LittleEndian(bytes[offset..]);
        return true;
    }

    // True when every word of the line would be read straight from memory, without
    // overlay bytes and without a wait. Every test is an overlap test, so passing for the
    // line implies passing for each word in it.
    private bool TryFillWordLine(ulong line, Span<byte> bytes) =>
        TryReadUnownedBytes(line, (ulong)WordLineBytes, bytes);

    /// <summary>The word-line fields of the WAITS line; resets them.</summary>
    private string TakeWordLineCounters() =>
        $" word_reads={Interlocked.Exchange(ref _wordReads, 0)}" +
        $" word_read_scopes={Interlocked.Exchange(ref _wordReadScopes, 0)}" +
        $" word_line_hits={Interlocked.Exchange(ref _wordLineHits, 0)}" +
        $" word_line_fills={Interlocked.Exchange(ref _wordLineFills, 0)}" +
        $" word_line_refused={Interlocked.Exchange(ref _wordLineRefused, 0)}";

    internal (long Reads, long Hits, long Fills, long Refused) WordLineCounters =>
        (Interlocked.Read(ref _wordReads), Interlocked.Read(ref _wordLineHits),
            Interlocked.Read(ref _wordLineFills), Interlocked.Read(ref _wordLineRefused));
}
