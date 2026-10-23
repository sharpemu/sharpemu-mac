// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

// Replaces only the ordinary path's pending-range query and first-byte read.
// Claim/clear/deferral decisions still run under the image lock. In particular this
// does not revive claim-first: a claim alone is never authority for metadata bytes.
internal sealed class MetalMetadataCodeCache : IDisposable
{
    private readonly record struct Entry(ResourceDependencyEpoch Epoch, ResourceDependency? Dependency, byte Code,
        GuestWriteGenerations.Token? SliceGeneration, long Policy, bool Unsupported);
    private readonly Dictionary<(ulong Address, ulong Size), Entry> _entries = new();
    internal long Hits { get; private set; }
    internal bool TryGet(IResourceDependencySource? source, ulong address, ulong size, out byte code)
    {
        code = 0;
        if (HostLookupCounters.Enabled && source is not null && source.DependencyReadsReady &&
            _entries.TryGetValue((address, size), out var entry) && entry.Epoch == source.DependencyEpoch &&
            (entry.SliceGeneration?.IsCurrent ?? true) &&
            entry.Dependency?.Validate() == ResourceDependencyStatus.Current)
        {
            code = entry.Code; Hits++;
            HostLookupCounters.Note(HostLookupCounters.Cache.Dcc, true);
            return true;
        }
        HostLookupCounters.Note(HostLookupCounters.Cache.Dcc, false);
        return false;
    }
    internal void Keep(IResourceDependencySource? source, ulong address, ulong size, byte code, ResourceDependencyEpoch epoch)
    {
        if (!HostLookupCounters.Enabled || source is null) return;
        var key = (address, size);
        if (epoch != source.DependencyEpoch) return;
        // A backing that cannot issue physical/native-alias proofs must use the
        // ordinary first-byte path. Do not retry enrollment on every repeated draw.
        if (_entries.TryGetValue(key, out var previous))
        {
            if (previous.Dependency is null && previous.Policy == source.CapturePolicyVersion &&
                (previous.Epoch == epoch || previous.Unsupported && previous.Epoch.Mapping == epoch.Mapping)) return;
            previous.Dependency?.Dispose(); _entries.Remove(key);
        }
        if (_entries.Count >= 1024) Dispose();
        var manager = GuestGpuMemoryHook.Current;
        var sliceGeneration = manager?.Covers(address, size) == true
            ? manager.WriteGenerations.Register(address, size) : null;
        Span<byte> expected = stackalloc byte[1]; expected[0] = code;
        source.TryCaptureDependency(address, expected, false, out var dependency);
        if (source.DependencyEpoch != epoch || !source.DependencyReadsReady)
        {
            dependency?.Dispose(); dependency = null;
        }
        if (dependency is null) HostLookupCounters.Note(HostLookupCounters.Cache.Dcc, false, refused: true);
        var unsupported = dependency is null && source.CaptureFailure.Reason is ResourceCaptureFailure.PhysicalWatch or
            ResourceCaptureFailure.NativeAlias or ResourceCaptureFailure.NativeWatch or ResourceCaptureFailure.Unsupported;
        _entries[key] = new(epoch, dependency, code, sliceGeneration, source.CapturePolicyVersion, unsupported);
    }
    public void Dispose()
    {
        foreach (var entry in _entries.Values) entry.Dependency?.Dispose();
        _entries.Clear();
    }
}
