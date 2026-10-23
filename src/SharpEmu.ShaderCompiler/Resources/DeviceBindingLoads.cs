// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;
using System.Linq;

namespace SharpEmu.ShaderCompiler.Resources;

/// <summary>
/// The scalar loads a resource depends on (binding reads), per shader, that the planner
/// leaves to the device because the GPU was still writing their dword when a bind read it.
/// </summary>
/// <remarks>
/// A binding read is a <see cref="ScalarValueKind.ScalarAddressWord"/> or
/// <see cref="ScalarValueKind.ScalarBufferWord"/> a descriptor, an address or a table index
/// is derived from. The host evaluates it per bind; when a dispatch recorded in the same
/// submission writes that dword, the read waits for the whole GPU backlog. Instead the read
/// returns unknown (<see cref="Defer"/>), the caller re-plans the shader with the load left
/// unflattened and unvalidated, so every value derived from it is computed on the device
/// (device descriptors, device addresses, device loads), and materializes again.
/// A re-plan is kept only when it costs nothing the host must know (<see cref="Accepts"/>):
/// image identities and unsupported sampler forms must remain host-evaluable; explicit
/// zero-LOD 2D/3D samplers can consume their S# on the device. A store whose range the
/// host can no longer evaluate would make every later read wait. Such a load is refused and
/// read (waited for) as before. SHARPEMU_METAL_V2_WAIT_RESOURCE_WORDS=1 turns
/// <see cref="Enabled"/> off.
/// A deferred word brings every other pending word read through the same table handle into
/// the same re-plan (<see cref="LearnTables"/>): one word per re-plan made a new program, and a
/// new pipeline compile, for each word a later bind found pending. A load learned stays learned
/// when some part of its plan still reads it on the host (<see cref="KeepOnHost"/>); it is
/// waited for there instead of planned back, which compiled the old program again. A key
/// rebuilt <see cref="RebuildCap"/> times in a run waits for its words from then on.
/// </remarks>
public static class DeviceBindingLoads
{
    private static readonly object Gate = new();
    private static readonly Dictionary<ulong, HashSet<int>> MemoryByHash = new();
    private static readonly HashSet<(ulong Hash, int Memory)> Refused = new();
    private static readonly Dictionary<(ulong Hash, int Memory), string> RefusalReasons = new();
    [ThreadStatic] private static string? _pendingReadReason;

    internal static string TakeReadReason()
    {
        var reason = _pendingReadReason ?? "host_required";
        _pendingReadReason = null;
        return reason;
    }

    internal static void SetRefusalReason(ulong key, int memory, string reason)
    {
        lock (Gate) RefusalReasons[(key, memory)] = reason;
    }

    private static string RefusalReason(ulong key, int memory)
    {
        lock (Gate) return RefusalReasons.GetValueOrDefault((key, memory), "persisted_refusal");
    }
    // Learned loads a plan that leaves them to the device still reads on the host: kept on the
    // device, and waited for instead of deferred (KeepOnHost).
    private static readonly HashSet<(ulong Hash, int Memory)> HostRead = new();
    // Re-plans that made a new program, per key, this run; a key at RebuildCap waits from then on.
    private static readonly Dictionary<ulong, int> Rebuilds = new();

    // Pending words handed to the device, and pending words read on the host (waited for),
    // by node kind: [0] ScalarAddressWord, [1] ScalarBufferWord.
    private static readonly long[] Unwaited = new long[2];
    private static readonly long[] Waited = new long[2];
    private static long _replans;
    private static long _refusedReplans;
    // Re-plans refused because planning failed with the loads on the device, and learned
    // loads kept on the device although read on the host (KeepOnHost).
    private static long _planFailed;
    private static long _planFailedReads;
    private static long _totalReplans, _totalRefused, _totalPlanFailed;
    private static long _keptOnHost;
    private static long _learned;
    private static long _seeded;

    /// <summary>
    /// A key's loads changed (one learned or refused): the key, every load it now leaves to the
    /// device and every load refused for it, so a store can keep them for the next start.
    /// </summary>
    public static event Action<ulong, int[], int[]>? Changed;

    /// <summary>Loads handed over by <see cref="Seed"/>, not counted as learned.</summary>
    public static long SeededLoads => Interlocked.Read(ref _seeded);

    /// <summary>Off: a pending binding word is read and waited for, as before.</summary>
    public static bool Enabled { get; set; } = true;

    /// <summary>
    /// A deferred word names the other pending words read through the same table handle too
    /// (<see cref="Request.TableWords"/>), so one re-plan learns them all.
    /// SHARPEMU_METAL_V2_REPLAN_TABLE=0: one word per re-plan, as before.
    /// </summary>
    public static bool LearnTables { get; set; } = PerfSwitches.Enabled("SHARPEMU_METAL_V2_REPLAN_TABLE");

    /// <summary>
    /// Re-plans that make a new program per key in a run before the key's pending words are
    /// waited for instead (<see cref="NoteRebuild"/>); 0 is no cap. SHARPEMU_METAL_V2_REPLAN_CAP
    /// sets it (=0 turns the cap off), 16 by default.
    /// </summary>
    public static int RebuildCap { get; set; } =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_REPLAN_CAP"), out var cap) && cap >= 0 ? cap : 16;

    // Hash is the plan's DeviceBindingKey: the shader hash, or the hash mixed with an embedded
    // vertex fetch (KeyFor). TableWords: the other binding loads read through the same table
    // handle whose words were pending too (LearnTables); null when none.
    public readonly record struct Request(ulong Hash, int MemoryIndex, ScalarValueKind Kind, IReadOnlyList<int>? TableWords = null,
        ulong ShaderHash = 0, uint Pc = 0, uint Component = 0, ulong Address = 0, BindingWordWriter Writer = default);

    /// <summary>
    /// The key a shader's binding loads are learned under. A memory index names a record of
    /// one memory table; a host-embedded vertex fetch drops its buffer loads and table loads
    /// from that table, so the same index names another load in a plan with another fetch,
    /// and a load learned (and accepted) for one embedding is not one for the other.
    /// </summary>
    public static ulong KeyFor(ulong hash, IReadOnlySet<uint>? fixedFunctionVertexLoads)
    {
        if (fixedFunctionVertexLoads is null || fixedFunctionVertexLoads.Count == 0)
        {
            return hash;
        }

        var key = hash ^ 0x9E3779B97F4A7C15UL;
        foreach (var pc in fixedFunctionVertexLoads.Order())
        {
            key = (key ^ pc) * 1099511628211UL;
        }

        return key == hash ? key + 1 : key;
    }

    // Index persistence is versioned by compiler build. Live keys also include every
    // option that changes scalar graph construction; PC/component accompany requests.
    public static ulong PlanKey(ulong hash, IReadOnlySet<uint>? fetch, ShaderStage stage,
        uint waveSize, ulong excluded, uint userBase, uint userCount)
    {
        var key = KeyFor(hash, fetch);
        foreach (var value in new ulong[] { (ulong)stage, waveSize, excluded, userBase, userCount })
            key = (key ^ value) * 1099511628211UL;
        return key;
    }

    [ThreadStatic]
    private static bool t_deferring;

    [ThreadStatic]
    private static Request? t_request;

    public static long LearnedLoads => Interlocked.Read(ref _learned);

    // The memory records of one shader the planner must leave to the device; null when none.
    public static IReadOnlySet<int>? For(ulong hash)
    {
        lock (Gate)
        {
            return MemoryByHash.TryGetValue(hash, out var set) && set.Count != 0 ? new HashSet<int>(set) : null;
        }
    }

    public static int CountFor(ulong hash)
    {
        lock (Gate)
        {
            return MemoryByHash.TryGetValue(hash, out var set) ? set.Count : 0;
        }
    }

    public static bool Contains(ulong hash, int memoryIndex)
    {
        lock (Gate)
        {
            return MemoryByHash.TryGetValue(hash, out var set) && set.Contains(memoryIndex);
        }
    }

    public static bool IsRefused(ulong hash, int memoryIndex)
    {
        lock (Gate)
        {
            return Refused.Contains((hash, memoryIndex));
        }
    }

    // True when the load is new to the shader. notify false leaves the store's line to a later
    // NotifyChanged, for a re-plan that may still take the load back (Unlearn).
    public static bool Learn(ulong hash, int memoryIndex, bool notify = true)
    {
        (int[] Learned, int[] Refused) state;
        lock (Gate)
        {
            if (Refused.Contains((hash, memoryIndex)))
            {
                return false;
            }

            if (!MemoryByHash.TryGetValue(hash, out var set))
            {
                MemoryByHash[hash] = set = [];
            }

            if (!set.Add(memoryIndex))
            {
                return false;
            }

            _learned++;
            if (!notify)
            {
                return true;
            }

            state = StateOf(hash);
        }

        Changed?.Invoke(hash, state.Learned, state.Refused);
        return true;
    }

    /// <summary>
    /// Takes back a load a re-plan learned (<see cref="Learn"/> with notify false) and did not
    /// keep, without refusing it: a later bind that finds its word pending may learn it alone.
    /// </summary>
    public static void Unlearn(ulong hash, int memoryIndex)
    {
        lock (Gate)
        {
            if (MemoryByHash.TryGetValue(hash, out var set) && set.Remove(memoryIndex))
            {
                _learned--;
            }
        }
    }

    /// <summary>Raises <see cref="Changed"/> with the key's loads as they are now.</summary>
    public static void NotifyChanged(ulong hash)
    {
        (int[] Learned, int[] Refused) state;
        lock (Gate)
        {
            state = StateOf(hash);
        }

        Changed?.Invoke(hash, state.Learned, state.Refused);
    }

    /// <summary>
    /// A learned load the plan that leaves it to the device still read on the host: it stays
    /// learned (the plan, and its program, stay as they are) and its word is waited for from
    /// now on instead of deferred. Refusing it planned the load back on the host, one more
    /// program and compile, and the next start learned it again.
    /// </summary>
    public static void KeepOnHost(ulong hash, int memoryIndex)
    {
        lock (Gate)
        {
            HostRead.Add((hash, memoryIndex));
        }

        Interlocked.Increment(ref _keptOnHost);
    }

    public static bool IsKeptOnHost(ulong hash, int memoryIndex)
    {
        lock (Gate)
        {
            return HostRead.Contains((hash, memoryIndex));
        }
    }

    /// <summary>
    /// Counts a re-plan of the key that made a new program. True once, when this one reached
    /// <see cref="RebuildCap"/>: the key's pending words are waited for from now on.
    /// </summary>
    public static bool NoteRebuild(ulong hash, out int rebuilds)
    {
        lock (Gate)
        {
            Rebuilds.TryGetValue(hash, out rebuilds);
            Rebuilds[hash] = ++rebuilds;
            return RebuildCap > 0 && rebuilds == RebuildCap;
        }
    }

    /// <summary>Whether the key reached <see cref="RebuildCap"/> this run.</summary>
    public static bool IsCapped(ulong hash)
    {
        lock (Gate)
        {
            return RebuildCap > 0 && Rebuilds.TryGetValue(hash, out var rebuilds) && rebuilds >= RebuildCap;
        }
    }

    /// <summary>A re-plan was refused because planning failed with the load on the device.</summary>
    public static void CountPlanFailure()
    {
        Interlocked.Increment(ref _planFailed);
        Interlocked.Increment(ref _totalPlanFailed);
    }

    // Whether a pending word of this load is waited for rather than deferred: refused, kept
    // on the host, or its key capped.
    private static bool WaitsFor(ulong hash, int memoryIndex)
    {
        lock (Gate)
        {
            return Refused.Contains((hash, memoryIndex)) || HostRead.Contains((hash, memoryIndex)) ||
                (RebuildCap > 0 && Rebuilds.TryGetValue(hash, out var rebuilds) && rebuilds >= RebuildCap);
        }
    }

    /// <summary>
    /// Hands over the loads an earlier start learned and refused for a key, before any shader
    /// is planned: the first plan leaves them to the device at once, instead of every start
    /// learning them again one re-plan (and one compile) at a time, in whatever order its
    /// GPU timing gives. They count as <see cref="SeededLoads"/> and raise no <see cref="Changed"/>.
    /// </summary>
    public static void Seed(ulong hash, IEnumerable<int> learned, IEnumerable<int> refused)
    {
        lock (Gate)
        {
            foreach (var index in refused)
            {
                if (index >= 0)
                {
                    Refused.Add((hash, index));
                }
            }

            foreach (var index in learned)
            {
                if (index < 0 || Refused.Contains((hash, index)))
                {
                    continue;
                }

                if (!MemoryByHash.TryGetValue(hash, out var set))
                {
                    MemoryByHash[hash] = set = [];
                }

                if (set.Add(index))
                {
                    _seeded++;
                }
            }
        }
    }

    /// <summary>Drops what a key learned and refused, as a new start without a store has it.</summary>
    public static void Forget(ulong hash)
    {
        lock (Gate)
        {
            MemoryByHash.Remove(hash);
            Refused.RemoveWhere(entry => entry.Hash == hash);
            HostRead.RemoveWhere(entry => entry.Hash == hash);
            Rebuilds.Remove(hash);
        }
    }

    /// <summary>
    /// Whether <paramref name="memory"/> is what a shader planned now under <paramref name="hash"/>
    /// leaves to the device: a program compiled for another set is one no bind asks for.
    /// </summary>
    public static bool IsCurrent(ulong hash, IReadOnlyCollection<int> memory)
    {
        lock (Gate)
        {
            if (!MemoryByHash.TryGetValue(hash, out var set) || set.Count == 0)
            {
                return memory.Count == 0;
            }

            return set.Count == memory.Count && memory.All(set.Contains);
        }
    }

    // Under Gate.
    private static (int[] Learned, int[] Refused) StateOf(ulong hash) =>
        (MemoryByHash.TryGetValue(hash, out var set) ? [.. set.Order()] : [],
            [.. Refused.Where(entry => entry.Hash == hash).Select(entry => entry.Memory).Order()]);

    // The re-plan with this load on the device was not kept: the load is read on the host
    // from now on, and never deferred again. notify false leaves the store's line to a later
    // NotifyChanged.
    public static void Refuse(ulong hash, int memoryIndex, bool notify = true)
    {
        (int[] Learned, int[] Refused) state = ([], []);
        lock (Gate)
        {
            if (MemoryByHash.TryGetValue(hash, out var set) && set.Remove(memoryIndex))
            {
                _learned--;
            }

            Refused.Add((hash, memoryIndex));
            if (notify)
            {
                state = StateOf(hash);
            }
        }

        if (notify)
        {
            Changed?.Invoke(hash, state.Learned, state.Refused);
        }

        Interlocked.Increment(ref _refusedReplans);
        Interlocked.Increment(ref _totalRefused);
    }

    public static void CountReplan(ScalarValueKind kind)
    {
        Interlocked.Increment(ref _replans);
        Interlocked.Increment(ref _totalReplans);
    }

    /// <summary>
    /// Opens one materialization attempt on this thread whose caller re-plans on a request
    /// (<see cref="EndAttempt"/>). Outside an attempt, or with <paramref name="defer"/> false,
    /// a pending binding word is read and counted as waited.
    /// </summary>
    public static void BeginAttempt(bool defer)
    {
        t_request = null;
        _pendingReadReason = null;
        t_deferring = defer && Enabled;
    }

    /// <summary>Whether the open attempt read a binding word as unknown.</summary>
    public static bool Deferred => t_request is not null;

    /// <summary>The first binding word the attempt deferred, if any.</summary>
    public static Request? EndAttempt()
    {
        var request = t_request;
        t_request = null;
        t_deferring = false;
        return request;
    }

    /// <summary>
    /// Asked before a host read of a binding word at <paramref name="address"/>: true when the
    /// GPU is still writing it and the read must return unknown instead of waiting.
    /// </summary>
    internal static bool Defer(ShaderResourcePlan plan, ResourceRuntimeInputs inputs, ScalarValue node, ulong address)
    {
        if (inputs.ReadsClean || inputs.ResourceWordPending is not { } pending ||
            node.Kind is not (ScalarValueKind.ScalarAddressWord or ScalarValueKind.ScalarBufferWord) ||
            (uint)node.MemoryIndex >= (uint)plan.Memory.Count || !pending(address))
        {
            return false;
        }

        var writer = inputs.ResourceWordWriter?.Invoke(address) ?? default;
        var reason = !Enabled ? "disabled" : IsRefused(plan.DeviceBindingKey, node.MemoryIndex) ? RefusalReason(plan.DeviceBindingKey, node.MemoryIndex) :
            IsKeptOnHost(plan.DeviceBindingKey, node.MemoryIndex) ? "host_read" : IsCapped(plan.DeviceBindingKey) ? "cap" :
            !t_deferring ? "fallback" : "deferred";
        _pendingReadReason = reason;
        BindingWaitProvenance.Word(plan, node, address, writer, reason);
        if (!t_deferring || WaitsFor(plan.DeviceBindingKey, node.MemoryIndex))
        {
            Interlocked.Increment(ref Waited[KindIndex(node.Kind)]);
            if (reason.StartsWith("plan_failed", StringComparison.Ordinal)) Interlocked.Increment(ref _planFailedReads);
            return false;
        }

        if (t_request is null) Interlocked.Increment(ref Unwaited[KindIndex(node.Kind)]);
        t_request ??= new Request(plan.DeviceBindingKey, node.MemoryIndex, node.Kind,
            LearnTables ? PendingTableWords(plan, node, address, pending) : null,
            plan.Hash, plan.Memory[node.MemoryIndex].Pc, plan.Memory[node.MemoryIndex].ComponentIndex, address, writer);
        return true;
    }

    /// <summary>
    /// The other binding loads of the plan read through the same table handle as
    /// <paramref name="node"/>, at an offset known relative to it, whose words are pending too:
    /// the words one GPU write lands together. One at a time, the bind that put the first on
    /// the device made one program, and each later bind that met the next word pending made
    /// one more, each a new pipeline compile.
    /// </summary>
    internal static List<int>? PendingTableWords(ShaderResourcePlan plan, ScalarValue node, ulong address, Func<ulong, bool> pending)
    {
        if (node.Operands.Length < 2)
        {
            return null;
        }

        List<int>? words = null;
        var handle = node.Operands[0];
        var offset = node.Operands[1];
        var immediate = (long)plan.Memory[node.MemoryIndex].Offset;
        foreach (var read in plan.TableReads)
        {
            Consider(read.Value);
        }

        foreach (var read in plan.DynamicReads)
        {
            Consider(read);
        }

        return words;

        void Consider(ScalarValue read)
        {
            if (read.Kind != node.Kind || read.MemoryIndex == node.MemoryIndex || (uint)read.MemoryIndex >= (uint)plan.Memory.Count ||
                read.Operands.Length < 2 || !SameValue(read.Operands[0], handle, depth: 4) ||
                plan.DeviceBindingMemory?.Contains(read.MemoryIndex) == true || words?.Contains(read.MemoryIndex) == true)
            {
                return;
            }

            // The same dynamic offset, or two constant ones: the distance between the words is known.
            var delta = (long)plan.Memory[read.MemoryIndex].Offset - immediate;
            if (!SameValue(read.Operands[1], offset, depth: 4))
            {
                if (!read.Operands[1].IsConstant || !offset.IsConstant)
                {
                    return;
                }

                delta += (long)(read.Operands[1].Payload & ~3UL) - (long)(offset.Payload & ~3UL);
            }

            var word = (long)(address & ~3UL) + (delta & ~3L);
            if (word < 0 || !pending((ulong)word))
            {
                return;
            }

            (words ??= []).Add(read.MemoryIndex);
        }
    }

    // The same node, or one of the same kind, payload and operands (a handle built twice from
    // the same user-data registers).
    private static bool SameValue(ScalarValue left, ScalarValue right, int depth)
    {
        if (ReferenceEquals(left, right))
        {
            return true;
        }

        if (depth == 0 || left.Kind != right.Kind || left.Operation != right.Operation || left.Payload != right.Payload ||
            left.Operands.Length != right.Operands.Length || left.Kind == ScalarValueKind.Phi)
        {
            return false;
        }

        for (var index = 0; index < left.Operands.Length; index++)
        {
            if (!SameValue(left.Operands[index], right.Operands[index], depth - 1))
            {
                return false;
            }
        }

        return true;
    }

    // One pending S# component moves the complete descriptor to the device. Leaving
    // siblings flattened would still read the producer's buffer on the host. Expand
    // only leaf loads belonging to the descriptor, not its host-evaluable table roots.
    internal static IReadOnlySet<int>? ExpandSamplerLoads(ScalarValueGraph graph, IReadOnlySet<int>? loads)
    {
        if (loads is null || loads.Count == 0) return loads;
        var expanded = new HashSet<int>(loads);
        foreach (var access in graph.Accesses)
        {
            if (access?.SamplerHandle is not { } sampler) continue;
            var leaves = new HashSet<int>();
            var pending = new Stack<ScalarValue>(sampler.Operands);
            var seen = new HashSet<ScalarValue>();
            while (pending.TryPop(out var value))
            {
                if (!seen.Add(value)) continue;
                if (value.Kind is ScalarValueKind.ScalarAddressWord or ScalarValueKind.ScalarBufferWord)
                    leaves.Add(value.MemoryIndex);
                else foreach (var operand in value.Operands) pending.Push(operand);
            }
            if (leaves.Overlaps(loads)) expanded.UnionWith(leaves);
        }
        return expanded;
    }

    /// <summary>
    /// Checks each access before replacing a plan: retain its tracked write range and
    /// every host-required image and sampler identity, including their descriptor words.
    /// </summary>
    public static bool Accepts(ShaderResourcePlan current, ShaderResourcePlan replanned) =>
        Accepts(current, replanned, out _);

    /// <summary><see cref="Accepts(ShaderResourcePlan, ShaderResourcePlan)"/>, naming the reason of a refusal.</summary>
    public static bool Accepts(ShaderResourcePlan current, ShaderResourcePlan replanned, out string? reason)
    {
        reason = null;
        if (current.Memory.Count != replanned.Memory.Count) { reason = "memory_shape"; return false; }
        for (var index = 0; index < current.Memory.Count; index++)
        {
            var old = current.Memory[index];
            var next = replanned.Memory[index];
            if (old.Pc != next.Pc || old.ComponentIndex != next.ComponentIndex || old.Kind != next.Kind || old.Access != next.Access)
            { reason = $"memory_shape:{old.Pc:X}:{old.ComponentIndex}"; return false; }
            if (old.Access is MemoryAccess.Write or MemoryAccess.Atomic)
            {
                var oldRange = current.DeviceAddressRanges.FirstOrDefault(range => range.MemoryIndices.Contains(index));
                var newRange = replanned.DeviceAddressRanges.FirstOrDefault(range => range.MemoryIndices.Contains(index));
                if (oldRange is not null && (newRange is null || (oldRange.Plannable && !newRange.Plannable) ||
                    oldRange.Bounded != newRange.Bounded || newRange.FirstByte > oldRange.FirstByte || newRange.EndByte < oldRange.EndByte))
                { reason = $"unranged_write:{old.Pc:X}:{old.ComponentIndex}"; return false; }
                // A runtime V# write has no bound host resource. Retain every existing
                // host-tracked write, including atomics, rather than comparing totals.
                if (!old.DeviceDescriptor && next.DeviceDescriptor ||
                    old.BufferDescriptor?.Provenance != BufferDescriptorProvenance.Runtime &&
                    next.BufferDescriptor?.Provenance == BufferDescriptorProvenance.Runtime ||
                    !Same(current, current.Accesses[index]?.Handle, replanned, replanned.Accesses[index]?.Handle) ||
                    !Same(current, current.Accesses[index]?.Active, replanned, replanned.Accesses[index]?.Active) ||
                    (old.Kind is MemoryResourceKind.Buffer or MemoryResourceKind.ScalarBuffer) &&
                    old.Resource < current.Info.Buffers.Count &&
                    (next.Resource >= replanned.Info.Buffers.Count ||
                     !SameDescriptor(current.Info.Buffers[(int)old.Resource].Source, replanned.Info.Buffers[(int)next.Resource].Source)))
                { reason = $"write_binding:{old.Pc:X}:{old.ComponentIndex}"; return false; }
            }
            if (old.Kind is MemoryResourceKind.Image or MemoryResourceKind.Sampler)
            {
                // Indirect candidate enumeration is host-required. This first implementation
                // keeps its exact existing plan rather than proving a new candidate universe.
                if (current.IndirectImages.Count != 0 || replanned.IndirectImages.Count != 0 ||
                    !Same(current, current.Accesses[index]?.Handle, replanned, replanned.Accesses[index]?.Handle) ||
                    !Same(current, current.Accesses[index]?.SamplerHandle, replanned, replanned.Accesses[index]?.SamplerHandle) ||
                    !SameSource(old.Kind, old.Resource, next.Resource) || !SameSampler(old.Sampler, next.Sampler))
                { reason = $"image_sampler_binding:{old.Pc:X}:{old.ComponentIndex}"; return false; }
            }
        }
        return true;

        bool SameSource(MemoryResourceKind kind, uint oldIndex, uint newIndex)
        {
            // Resource indices are dense image/buffer slots, whose sources may be renumbered.
            // Handles above compare identity; null substitution must also be excluded.
            if (kind == MemoryResourceKind.Image)
            {
                if (oldIndex != MemoryAccessInfo.NoResource && oldIndex < current.Info.Images.Count &&
                    newIndex != MemoryAccessInfo.NoResource && newIndex < replanned.Info.Images.Count)
                    return SameDescriptor(current.Info.Images[(int)oldIndex].Source, replanned.Info.Images[(int)newIndex].Source);
            }
            return kind != MemoryResourceKind.Image || oldIndex == newIndex && oldIndex == MemoryAccessInfo.NoResource;
        }
        bool SameSampler(uint x, uint y)
        {
            if (x == MemoryAccessInfo.NoResource || y == MemoryAccessInfo.NoResource) return x == y;
            return x < current.Info.Samplers.Count && y < replanned.Info.Samplers.Count &&
                (replanned.Info.Samplers[(int)y].Device ||
                 SameDescriptor(current.Info.Samplers[(int)x].Source, replanned.Info.Samplers[(int)y].Source));
        }
        bool SameDescriptor(uint x, uint y)
        {
            if (x >= current.DescriptorSources.Count || y >= replanned.DescriptorSources.Count) return false;
            var left = current.DescriptorSources[(int)x]; var right = replanned.DescriptorSources[(int)y];
            return left.Dwords.Length == right.Dwords.Length && left.Dwords.Zip(right.Dwords).All(pair => Same(current, pair.First, replanned, pair.Second));
        }
    }

    // Compare across plans after undoing flattening. Slot numbers alone are not identity.
    private static bool Same(ShaderResourcePlan a, ScalarValue? x, ShaderResourcePlan b, ScalarValue? y) =>
        Same(a, x, b, y, new HashSet<(ScalarValue, ScalarValue)>());

    private static bool Same(ShaderResourcePlan a, ScalarValue? x, ShaderResourcePlan b, ScalarValue? y,
        HashSet<(ScalarValue, ScalarValue)> seen)
    {
        if (x is null || y is null) return x is null && y is null;
        if (x.Kind == ScalarValueKind.ResourceTableWord && x.Payload < (ulong)a.TableReads.Count)
            x = a.TableReads[(int)x.Payload].Value;
        if (y.Kind == ScalarValueKind.ResourceTableWord && y.Payload < (ulong)b.TableReads.Count)
            y = b.TableReads[(int)y.Payload].Value;
        if (!seen.Add((x, y))) return true;
        if (x.Kind != y.Kind || x.Type != y.Type || x.Operation != y.Operation || x.Operands.Length != y.Operands.Length) return false;
        if (x.Kind is ScalarValueKind.ScalarAddressWord or ScalarValueKind.ScalarBufferWord)
        {
            var m = a.Memory[x.MemoryIndex]; var n = b.Memory[y.MemoryIndex];
            if (m.Pc != n.Pc || m.ComponentIndex != n.ComponentIndex || m.Offset != n.Offset) return false;
        }
        else if (x.Payload != y.Payload) return false;
        for (var i = 0; i < x.Operands.Length; i++)
            if (!Same(a, x.Operands[i], b, y.Operands[i], seen)) return false;
        return true;
    }

    public static string TakeReport() => FormattableString.Invariant(
        $" words_deferred=[addr={Interlocked.Exchange(ref Unwaited[0], 0)} buf={Interlocked.Exchange(ref Unwaited[1], 0)}]") +
        FormattableString.Invariant(
            $" words_waited=[addr={Interlocked.Exchange(ref Waited[0], 0)} buf={Interlocked.Exchange(ref Waited[1], 0)}]") +
        FormattableString.Invariant(
            $" words_replans=[ok={Interlocked.Exchange(ref _replans, 0)} refused={TakeRefused()} kept_on_host={Interlocked.Exchange(ref _keptOnHost, 0)}] words_replans_total=[ok={Interlocked.Read(ref _totalReplans)} refused={Interlocked.Read(ref _totalRefused)} plan_failed={Interlocked.Read(ref _totalPlanFailed)}] words_refused_reads=[plan_failed={Interlocked.Exchange(ref _planFailedReads, 0)}] binding_loads_device={LearnedLoads}");

    private static string TakeRefused() => FormattableString.Invariant(
        $"{Interlocked.Exchange(ref _refusedReplans, 0)} plan_failed={Interlocked.Exchange(ref _planFailed, 0)}");

    private static int KindIndex(ScalarValueKind kind) => kind == ScalarValueKind.ScalarBufferWord ? 1 : 0;
}
