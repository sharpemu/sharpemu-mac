// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Resources;

// An owned view of resmat's original immutable observation. It deliberately owns no
// physical watches: cache eviction cannot invalidate its lifetime, and validation
// uses the SAME nonwaiting, alias/publication-aware byte comparer as resmat3.
public sealed class ResourceReplayValidation
{
    public ShaderResourcePlan Plan { get; }
    private readonly IResourceDependencySource _source;
    private readonly ResourceDependencyEpoch _epoch;
    private readonly long _policy;
    private readonly ResourcePatchPlan _patch;
    private readonly ulong[] _used, _addresses;
    private readonly int[] _lengths, _offsets;
    private readonly byte[] _bytes;
    private readonly ResourceSnapshot _snapshot;
    private readonly Dictionary<ulong, ResourceSnapshot> _patched = new();
    private readonly int[] _dynamic;

    internal ResourceReplayValidation(IResourceDependencySource source, ResourceDependencyEpoch epoch,
        ShaderResourcePlan plan, ulong[] used, ulong[] addresses, int[] lengths, int[] offsets,
        byte[] bytes, ResourceSnapshot snapshot)
    {
        _source = source; _epoch = epoch; _policy = source.CapturePolicyVersion;
        Plan = plan; _patch = plan.PatchPlan; _used = used; _addresses = addresses; _lengths = lengths;
        _offsets = offsets; _bytes = bytes; _snapshot = snapshot;
        _dynamic = Enumerable.Range(0, snapshot.UserData.Length)
            .Where(word => !ResourceMaterializationCache.UsesWord(used, word) || _patch.IsPatchable(word)).ToArray();
    }

    public long RetainedBytes => 256L + _bytes.Length + _snapshot.UserData.Length * 4L +
        _addresses.Length * 16L;

    public void ObserveReads(ResourceReadBatchObserver observer, bool enabled)
    {
        if (enabled) observer(_addresses, _lengths, _offsets, _bytes);
    }

    public bool TryReplay(IResourceDependencySource source, IReadOnlyList<uint> userData,
        out ResourceSnapshot snapshot, out bool patched, out ResourceDependencyStatus refusal)
    {
        snapshot = _snapshot;
        patched = false;
        refusal = ResourceDependencyStatus.Unavailable;
        if (!ReferenceEquals(source, _source) || source.CapturePolicyVersion != _policy) return false;
        if (!source.DependencyReadsReady) { refusal = ResourceDependencyStatus.PendingWriter; return false; }
        var before = source.DependencyEpoch;
        if (before.Mapping != _epoch.Mapping) { refusal = ResourceDependencyStatus.Mapping; return false; }
        if (!_patch.Matches(_snapshot.UserData, userData, _used, out patched) ||
            (patched && !_patch.CanPatch(_snapshot, userData))) return false;

        // Changed global image/writer/overlay epochs are NOT a proof of changed
        // descriptor bytes. Every original range must pass current ownership and
        // byte checks. Resource handles/acquisitions are resolved again by the host.
        if (!_bytes.AsSpan().IsEmpty && !source.TryValidateDependencyBytes(_addresses, _lengths, _offsets, _bytes))
        {
            refusal = before.Images != _epoch.Images ? ResourceDependencyStatus.ImageIdentity :
                before.Labels != _epoch.Labels || before.Overlay != _epoch.Overlay ? ResourceDependencyStatus.Overlay :
                before.GpuWrites != _epoch.GpuWrites ? ResourceDependencyStatus.GpuWrite : ResourceDependencyStatus.CpuWrite;
            return false;
        }
        if (!source.DependencyReadsReady) { refusal = ResourceDependencyStatus.PendingWriter; return false; }
        if (source.DependencyEpoch != before) return false;
        if (patched)
        {
            var key = ResourcePatchPlan.Key(userData, _dynamic);
            if (_patched.TryGetValue(key, out var previous) && SameWords(previous.UserData, userData)) snapshot = previous;
            else
            {
                var words = new uint[userData.Count];
                for (var word = 0; word < words.Length; word++) words[word] = userData[word];
                snapshot = _patch.Apply(_snapshot, words);
                // Bounded ring memo. Published snapshots are never modified.
                if (_patched.Count < 32) _patched[key] = snapshot;
            }
        }
        refusal = ResourceDependencyStatus.Current;
        return true;
    }

    private static bool SameWords(uint[] left, IReadOnlyList<uint> right)
    {
        if (left.Length != right.Count) return false;
        for (var word = 0; word < left.Length; word++) if (left[word] != right[word]) return false;
        return true;
    }
}
