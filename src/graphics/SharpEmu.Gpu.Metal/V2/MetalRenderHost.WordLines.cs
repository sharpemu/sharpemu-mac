// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
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
    // Keep the packet protocol independent of the write-version implementation. Until
    // a range has complete native protection AND a participating physical-writer watch,
    // equality of first-write counters cannot certify content. In particular, an
    // unprotected page may receive arbitrarily many stores after its first fault.
    ResourceDependencyEpoch IResourceDependencySource.DependencyEpoch => new(
        _gpuWrites.Version, _cpOverlay.Version, DeviceAddressMappingVersion(), _imageCache.Epochs.Binding, _labelOverlay.Version);

    private byte[] _dependencyCapture = [];
    private readonly Dictionary<(ulong Address, int Size), WeakReference<WordDependency>> _wordDependencies = new();

    bool IResourceDependencySource.TryCaptureDependency(ulong address, ReadOnlySpan<byte> expected, bool clean,
        out ResourceDependency? dependency)
    {
        dependency = null;
        if (expected.IsEmpty || expected.Length > 65536 || _labelOverlay.Count != 0 ||
            _backing.SingleCopy is { ReleasedChunks: > 0 } || _memory is not ICpuMappedReadWatchSource source ||
            !GuestImageWriteTracker.TryGetProtectedObservedContentToken(address, (ulong)expected.Length, out var before) ||
            _imageCache.HasGpuModifiedImageBytes(address, (ulong)expected.Length) ||
            OverlayCoverage(address, (ulong)expected.Length) != MetalOverlayCoverage.None ||
            (_gpuWrites.MayCover(address, (ulong)expected.Length) && PendingSerialHinted(address, (ulong)expected.Length) != 0))
            return false;

        var epoch = ((IResourceDependencySource)this).DependencyEpoch;
        var key = (address, expected.Length);
        // Share the SAME original observation across stages. Bytes are captured once
        // per interval/version, rather than re-read as each plan enrolls the interval.
        if (_wordDependencies.TryGetValue(key, out var weak) && weak.TryGetTarget(out var existing) &&
            existing.Matches(epoch, expected) && existing.Validate() == ResourceDependencyStatus.Current)
        {
            existing.Retain();
            dependency = existing;
            return true;
        }

        // Copy only on enrollment. The observation closes managed before/landed races
        // on physical aliases; protected tokens close native first-fault/rearm races.
        if (_dependencyCapture.Length < expected.Length) _dependencyCapture = new byte[expected.Length];
        var bytes = _dependencyCapture.AsSpan(0, expected.Length);
        if (source.TryReadMappedCopyWithWriteWatch(address, (ulong)bytes.Length, address, bytes,
            out var observation) != CpuMappedCopyResult.Copied || observation is null) return false;
        if (!bytes.SequenceEqual(expected) || !source.IsMappedCopyWriteWatchCurrent(observation) ||
            !GuestImageWriteTracker.TryGetProtectedObservedContentToken(address, (ulong)bytes.Length, out var after) ||
            before != after)
        {
            source.ReleaseMappedCopyWriteWatch(observation);
            return false;
        }
        var captured = new WordDependency(source, observation, before, epoch, expected.ToArray());
        if (_wordDependencies.Count >= 2048) _wordDependencies.Clear();
        _wordDependencies[key] = new(captured);
        dependency = captured;
        return true;
    }

    private sealed class WordDependency(ICpuMappedReadWatchSource source, CpuMappedCopyObservation observation,
        GuestImageWriteTracker.ContentToken token, ResourceDependencyEpoch epoch, byte[] expected) : ResourceDependency
    {
        // References are worker-owned, like the word-line scope and materialization cache.
        private int _references = 1;
        public override long RetainedBytes => 512L + expected.Length + observation.Segments.Count * 128L;
        internal bool Matches(ResourceDependencyEpoch current, ReadOnlySpan<byte> bytes) =>
            _references != 0 && epoch == current && bytes.SequenceEqual(expected);
        internal void Retain() => _references++;
        public override ResourceDependencyStatus Validate()
        {
            if (_references == 0) return ResourceDependencyStatus.Unavailable;
            if (!source.IsMappedCopyObservationCurrent(observation)) return ResourceDependencyStatus.Mapping;
            if (!source.IsMappedCopyWriteWatchCurrent(observation)) return ResourceDependencyStatus.CpuWrite;
            return GuestImageWriteTracker.TryGetProtectedObservedContentToken(observation.SourceAddress,
                observation.SourceByteCount, out var current) && token == current
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
