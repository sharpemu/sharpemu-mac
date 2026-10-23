// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Resources;

public sealed partial class ResourceMaterializationCache
{
    public static readonly bool DependencyBypassEnabled = PerfSwitches.Enabled("SHARPEMU_RESOURCE_BYPASS");
    public static readonly bool VerifyDependencyBypass = PerfSwitches.Verify("SHARPEMU_RESOURCE_BYPASS_VERIFY");
    private readonly bool _dependencyBypass;
    private readonly bool _verifyDependencyBypass;
    private readonly Dictionary<BypassKey, DependencyPacket> _packets = new();
    private readonly Dictionary<BypassKey, ResourceDependencyFailure> _uncertified = new();
    private readonly long _packetByteBudget;
    private readonly record struct BypassKey(ShaderResourcePlan Plan, ulong ShaderBase, ComputeSelectorState? Compute);
    private long _packetBytes;
    public long ByteValidatedCalls { get; private set; }
    public long BypassedCalls { get; private set; }
    public long PatchedCalls { get; private set; }
    public long FullPathCalls { get; private set; }
    internal static bool UsesWord(ulong[] mask, int index) => IsUsed(mask, index);

    private enum FullReason { Disabled, NoSource, Cold, Inputs, CpuWrite, GpuWrite, PendingWriter, Mapping, ImageIdentity, Overlay, Unavailable, Count }
    private static readonly long[] FullReasons = new long[(int)FullReason.Count];
    private const int CaptureFailureCount = (int)ResourceCaptureFailure.Unsupported + 1;
    private static readonly long[] RangeMisses = new long[((int)ResourceRangeClass.Released + 1) * CaptureFailureCount];
    private static void CountRangeMiss(ResourceDependencyFailure failure) =>
        Interlocked.Increment(ref RangeMisses[(int)failure.Range * CaptureFailureCount + (int)failure.Reason]);

    private static long _bypassCalls;
    private static long _patchedCalls;
    private static long _byteValidatedCalls;

    // Printed by Metal's periodic PERF report. These count calls which never entered
    // MaterializeCore, separately from the legacy cache's byte-validation hits.
    public static string TakeBypassReport()
    {
        var reasons = string.Join(',', Enumerable.Range(0, (int)FullReason.Count).Select(index =>
            $"{(FullReason)index}:{Interlocked.Exchange(ref FullReasons[index], 0)}"));
        var ranges = string.Join(',', Enumerable.Range(0, RangeMisses.Length).Select(index =>
        {
            var count = Interlocked.Exchange(ref RangeMisses[index], 0);
            return count == 0 ? null : $"{(ResourceRangeClass)(index / CaptureFailureCount)}/{(ResourceCaptureFailure)(index % CaptureFailureCount)}:{count}";
        }).Where(item => item is not null));
        return $"resmat_range_miss=[{ranges}] resmat_bypassed={Interlocked.Exchange(ref _bypassCalls, 0)} resmat_patched={Interlocked.Exchange(ref _patchedCalls, 0)} resmat_byte_validated={Interlocked.Exchange(ref _byteValidatedCalls, 0)} resmat_full=[{reasons}]";
    }

    private sealed class DependencyPacket
    {
        public required ResourcePatchPlan Plan;
        public required ResourceRuntimeInputs Readers;
        public required IResourceDependencySource Source;
        public required ResourceDependencyEpoch Epoch;
        public required ResourceDependency[] Dependencies;
        public required ResourceSnapshot Snapshot;
        public required ulong[] UsedUserData;
        public required int[] DynamicWords;
        public required ResourceSpecialization Specialization;
        public required ulong[] ReadAddresses;
        public required int[] ReadOffsets, ReadLengths;
        public required byte[] ReadBytes;
        public bool ByteValidated;
        public long CapturePolicy;
        public bool Disposed;
        public DependencyPacket? Next;
        // Published snapshots stay immutable. A bounded memo avoids allocating on repeated
        // rings; a new scalar value needs its own shell under ResourceSnapshot's owned ABI.
        public readonly Dictionary<ulong, ResourceSnapshot> Patches = new();
        public long Bytes;

        public void Dispose()
        {
            Disposed = true;
            foreach (var dependency in Dependencies) dependency.Dispose();
        }
    }

    private bool TryBypass(ShaderResourcePlan plan, ResourceRuntimeInputs inputs,
        ref ResourceSnapshot snapshot, ref ResourceSpecialization specialization, out FullReason reason)
    {
        reason = !_dependencyBypass ? FullReason.Disabled : inputs.DependencySource is null ? FullReason.NoSource : FullReason.Cold;
        if (!_dependencyBypass || inputs.DependencySource is not { } source || DeviceBindingLoads.Deferred) return false;
        var key = new BypassKey(plan, inputs.ShaderBase, inputs.ComputeState);
        if (!_packets.TryGetValue(key, out var head))
        {
            if (_uncertified.TryGetValue(key, out var failure))
            {
                reason = FullReason.Unavailable;
                CountRangeMiss(failure);
            }
            return false;
        }
        reason = FullReason.Inputs;
        var epoch = source.DependencyEpoch;
        for (var packet = head; packet is not null; packet = packet.Next)
        {
            if (!ReferenceEquals(packet.Source, source) ||
                packet.CapturePolicy != source.CapturePolicyVersion ||
                !ReferenceEquals(packet.Readers.ReadMemory, inputs.ReadMemory) ||
                !ReferenceEquals(packet.Readers.ReadCleanMemory, inputs.ReadCleanMemory) ||
                !ReferenceEquals(packet.Readers.DataWritePending, inputs.DataWritePending) ||
                !ReferenceEquals(packet.Readers.ResourceWordPending, inputs.ResourceWordPending) ||
                !ReferenceEquals(packet.Readers.ResourceWordWriter, inputs.ResourceWordWriter) ||
                packet.Readers.ReadsClean != inputs.ReadsClean ||
                !packet.Plan.Matches(packet.Snapshot.UserData, inputs.UserData, packet.UsedUserData, out var patched) ||
                (patched && !packet.Plan.CanPatch(packet.Snapshot, inputs.UserData))) continue;
            reason = EpochReason(packet.Epoch, epoch);
            // Global writers/overlays need not invalidate a byte packet when none
            // of its reads overlap them. Mapping and resource identity still veto.
            if (packet.ByteValidated && reason is FullReason.GpuWrite or FullReason.Overlay)
                reason = FullReason.Cold;
            if (reason != FullReason.Cold) continue;
            if (packet.ByteValidated && !source.TryValidateDependencyBytes(packet.ReadAddresses,
                packet.ReadLengths, packet.ReadOffsets, packet.ReadBytes))
            {
                reason = FullReason.Unavailable;
                continue;
            }
            foreach (var dependency in packet.Dependencies)
            {
                var status = dependency.Validate();
                if (status == ResourceDependencyStatus.Current) continue;
                reason = (FullReason)((int)FullReason.CpuWrite + (int)status - 1);
                break;
            }
            // A writer/overlay transition during validation cannot certify a packet.
            if (reason != FullReason.Cold || source.DependencyEpoch != epoch) continue;

            if (inputs.ObserveCachedReads is { } observeBatch)
                observeBatch(packet.ReadAddresses, packet.ReadLengths, packet.ReadOffsets, packet.ReadBytes);
            else if (inputs.ObserveCachedRead is { } observe)
            {
                for (var range = 0; range < packet.ReadAddresses.Length; range++)
                    observe(packet.ReadAddresses[range], packet.ReadBytes.AsSpan(packet.ReadOffsets[range], packet.ReadLengths[range]));
            }
            snapshot = packet.Snapshot;
            if (patched)
            {
                var patchKey = ResourcePatchPlan.Key(inputs.UserData, packet.DynamicWords);
                if (packet.Patches.TryGetValue(patchKey, out var previous) && SameUserData(previous.UserData, inputs.UserData))
                    snapshot = previous;
                else
                {
                    var words = new uint[inputs.UserData.Count];
                    for (var word = 0; word < words.Length; word++) words[word] = inputs.UserData[word];
                    snapshot = packet.Plan.Apply(packet.Snapshot, words);
                    var bytes = 256L + words.Length * sizeof(uint) + snapshot.Buffers.Length * IntPtr.Size + packet.Plan.Patches.Length * 40L;
                    if (packet.Patches.Count < 512 && _packetBytes <= _packetByteBudget - bytes)
                    {
                        // A hash is only a probe; exact roots above decide every hit.
                        packet.Patches[patchKey] = snapshot;
                        packet.Bytes += bytes;
                        _packetBytes += bytes;
                    }
                }
                PatchedCalls++;
                Interlocked.Increment(ref _patchedCalls);
            }
            else
            {
                BypassedCalls++;
                Interlocked.Increment(ref _bypassCalls);
            }
            specialization = packet.Specialization;
            if (packet.ByteValidated)
            {
                ByteValidatedCalls++;
                Interlocked.Increment(ref _byteValidatedCalls);
            }
            Hits++;
            if (_verifyDependencyBypass) VerifyBypass(plan, inputs, snapshot, specialization);
            return true;
        }
        return false;
    }

    private static FullReason EpochReason(ResourceDependencyEpoch previous, ResourceDependencyEpoch current) =>
        previous.Mapping != current.Mapping ? FullReason.Mapping :
        previous.Images != current.Images ? FullReason.ImageIdentity :
        previous.Overlay != current.Overlay || previous.Labels != current.Labels ? FullReason.Overlay :
        previous.GpuWrites != current.GpuWrites ? FullReason.GpuWrite : FullReason.Cold;

    private static void VerifyBypass(ShaderResourcePlan plan, ResourceRuntimeInputs inputs,
        ResourceSnapshot snapshot, ResourceSpecialization specialization)
    {
        var fresh = new ResourceSnapshot();
        var freshSpecialization = new ResourceSpecialization();
        if (!ResourceMaterializer.Materialize(plan, inputs, ref fresh, ref freshSpecialization, out var failure))
            throw new InvalidOperationException($"Resource bypass verification failed: {failure}");
        if (Difference(snapshot, specialization, fresh, freshSpecialization) is { } difference)
            throw new InvalidOperationException($"Resource bypass verification mismatch: {difference}");
    }

    private void CapturePacket(Entry entry, ResourceRuntimeInputs inputs, ResourceSnapshot? observed = null)
    {
        if (!_dependencyBypass || inputs.DependencySource is not { } source || entry.MovedReads || DeviceBindingLoads.Deferred) return;
        var epoch = source.DependencyEpoch;
        var policy = source.CapturePolicyVersion;
        // Cache refusal on the actual immutable read set, not just on a shader key.
        // Unsupported physical watches retry only after mapping/policy changes;
        // transient refusals retry in a new dependency epoch. Neither scans roots.
        if (entry.CaptureRefusal is { } refusal && ReferenceEquals(refusal.Source, source) && refusal.Policy == policy &&
            (refusal.Failure.Reason is ResourceCaptureFailure.PhysicalWatch or ResourceCaptureFailure.Unsupported
                ? refusal.Epoch.Mapping == epoch.Mapping : refusal.Epoch == epoch && refusal.Retry == source.CaptureRetryVersion)) return;
        for (var word = 0; word < entry.UserData.Length; word++)
            if (IsUsed(entry.UsedUserData, word) && entry.UserData[word] != inputs.UserData[word]) return;
        if (entry.CapturedPacket?.TryGetTarget(out var captured) == true && !captured.Disposed && captured.ByteValidated &&
            ReferenceEquals(captured.Source, source) && captured.CapturePolicy == policy &&
            ReferenceEquals(captured.Readers.ReadMemory, inputs.ReadMemory) &&
            ReferenceEquals(captured.Readers.ReadCleanMemory, inputs.ReadCleanMemory) &&
            ReferenceEquals(captured.Readers.DataWritePending, inputs.DataWritePending) &&
            ReferenceEquals(captured.Readers.ResourceWordPending, inputs.ResourceWordPending) &&
            ReferenceEquals(captured.Readers.ResourceWordWriter, inputs.ResourceWordWriter) &&
            captured.Readers.ReadsClean == inputs.ReadsClean &&
            captured.Epoch.Mapping == epoch.Mapping && captured.Epoch.Images == epoch.Images &&
            captured.Plan.Matches(captured.Snapshot.UserData, inputs.UserData, captured.UsedUserData, out var patched) &&
            (!patched || captured.Plan.CanPatch(captured.Snapshot, inputs.UserData))) return;
        var key = new BypassKey(entry.Plan, inputs.ShaderBase, inputs.ComputeState);
        var byteValidated = source.UsesDependencyByteValidation && entry.RangeAddresses.Length != 0;
        var size = entry.EstimatedBytes + (byteValidated ? 0 : entry.RangeAddresses.Length * 256L) +
            256L + entry.UserData.Length * sizeof(uint);
        if (size > _packetByteBudget) return;
        var result = observed ?? entry.Snapshot;
        ResourceDependency[] dependencies = [];
        var count = 0;
        try
        {
            // The evaluator may have read across an intervening writer. Do not
            // publish a result whose complete read set is no longer current.
            if (byteValidated && !source.TryValidateDependencyBytes(entry.RangeAddresses,
                entry.RangeLengths, entry.RangeOffsets, entry.Bytes))
            {
                var failure = new ResourceDependencyFailure(ResourceRangeClass.Unknown, ResourceCaptureFailure.Changed);
                entry.CaptureRefusal = new(source, epoch, policy, source.CaptureRetryVersion, failure);
                if (_uncertified.Count >= 2048) _uncertified.Clear();
                _uncertified[key] = failure;
                return;
            }
            for (var range = 0; !byteValidated && range < entry.RangeAddresses.Length; range++)
            {
                if (!source.TryCaptureDependency(entry.RangeAddresses[range],
                    entry.Bytes.AsSpan(entry.RangeOffsets[range], entry.RangeLengths[range]),
                    entry.RangeClean[range], out var dependency) || dependency is null)
                {
                    dependency?.Dispose();
                    if (_uncertified.Count >= 2048) _uncertified.Clear();
                    _uncertified[key] = source.CaptureFailure;
                    entry.CaptureRefusal = new(source, epoch, policy, source.CaptureRetryVersion, source.CaptureFailure);
                    return;
                }
                // Most unsupported ranges refuse at the first token. Keep that ordinary
                // fallback allocation-free instead of allocating an array before probing.
                if (dependencies.Length == 0) dependencies = new ResourceDependency[entry.RangeAddresses.Length];
                dependencies[count++] = dependency;
            }
            foreach (var dependency in dependencies)
            {
                if (dependency.Validate() != ResourceDependencyStatus.Current) return;
                size += Math.Max(0, dependency.RetainedBytes - 256);
            }
            if (size > _packetByteBudget) return;
            if (source.DependencyEpoch != epoch) return;

            // Independent byte/count bounds: a packet retains the snapshot and may outlive
            // its legacy cache entry. Clearing drops only optimization state.
            if (_packets.Count >= 2048 || _packetBytes > _packetByteBudget - size) ClearPackets();
            _uncertified.Remove(key);
            _packets.TryGetValue(key, out var previous);
            var roots = new uint[inputs.UserData.Count];
            for (var word = 0; word < roots.Length; word++) roots[word] = inputs.UserData[word];
            var patchPlan = entry.Plan.PatchPlan;
            var dynamicWords = new List<int>();
            for (var word = 0; word < roots.Length; word++)
                if (!IsUsed(entry.UsedUserData, word) || patchPlan.IsPatchable(word)) dynamicWords.Add(word);
            var packet = new DependencyPacket
            {
                Plan = patchPlan, ReadAddresses = entry.RangeAddresses, ReadOffsets = entry.RangeOffsets,
                ReadLengths = entry.RangeLengths, ReadBytes = entry.Bytes,
                UsedUserData = entry.UsedUserData,
                DynamicWords = dynamicWords.ToArray(),
                Source = source, Epoch = epoch, Dependencies = dependencies,
                ByteValidated = byteValidated, CapturePolicy = policy,
                Snapshot = PatchedSnapshot(result, result.Buffers,
                    result.FlattenedResourceTable, roots),
                Specialization = entry.Specialization,
                Readers = new ResourceRuntimeInputs
                {
                    ReadMemory = inputs.ReadMemory, ReadCleanMemory = inputs.ReadCleanMemory,
                    ReadsClean = inputs.ReadsClean, DataWritePending = inputs.DataWritePending,
                    ResourceWordPending = inputs.ResourceWordPending, ResourceWordWriter = inputs.ResourceWordWriter,
                },
                Next = previous, Bytes = size,
            };
            _packetBytes += packet.Bytes;
            _packets[key] = packet;
            entry.CapturedPacket = new(packet);
            entry.CaptureRefusal = null;
            count = 0; // Ownership moved to the packet.
            // Eight alternate descriptor/root sets per immutable plan, newest first.
            var tail = packet;
            for (var depth = 1; depth < 8 && tail.Next is not null; depth++) tail = tail.Next;
            var discarded = tail.Next;
            tail.Next = null;
            while (discarded is not null)
            {
                discarded.Dispose();
                _packetBytes -= discarded.Bytes;
                discarded = discarded.Next;
            }
        }
        finally
        {
            for (var index = 0; index < count; index++) dependencies[index].Dispose();
        }
    }

    private sealed record CaptureRefusal(IResourceDependencySource Source, ResourceDependencyEpoch Epoch,
        long Policy, long Retry, ResourceDependencyFailure Failure);

    private void ClearPackets()
    {
        foreach (var head in _packets.Values)
            for (var packet = head; packet is not null; packet = packet.Next) packet.Dispose();
        _packets.Clear();
        _packetBytes = 0;
    }
}
