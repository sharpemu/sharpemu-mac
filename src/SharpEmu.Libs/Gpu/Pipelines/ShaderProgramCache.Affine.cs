// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Libs.Gpu.Pipelines;

internal sealed partial class ShaderProgramCache
{
    // Portable analysis depends on code, graph inputs and position pruning, rather
    // than the native layout selected by the draw. The final static key is probed
    // only after this analysis has selected that layout.
    private readonly record struct AffineSourceKey(ulong Hash, uint Size, ulong Continuation,
        uint UserDataBase, int UserDataCount, ulong Excluded, bool PositionOnly);
    private readonly Dictionary<AffineSourceKey, ProgramSourceEntry> _affineSources = new();
    private AffineSourceKey _lastAffineSourceKey;
    private ProgramSourceEntry? _lastAffineSource;
    private static readonly bool AffineCacheEnabled = Environment.GetEnvironmentVariable("SHARPEMU_AFFINE_FETCH_CACHE") != "0";

    private ProgramSourceEntry? PrepareAffineVertexInput(ShaderSource source, StageCompileOptions options, ResourceRuntimeInputs inputs)
    {
        if (!AffineVertexFetchResolver.Enabled || source.Stage != ShaderStage.Vertex ||
            options.VertexInfo is not { Attributes.Length: 0, FetchEmbedded: false } info) return null;

        var key = new AffineSourceKey(source.Hash, source.CodeSize, source.ContinuationAddressOffset,
            source.UserDataBase, source.UserData.Length, source.ExcludedUserDataRegisters, options.PositionOnlyVertex);
        var entry = _lastAffineSourceKey == key ? _lastAffineSource : null;
        if (entry is null && !_affineSources.TryGetValue(key, out entry))
        {
            entry = CreateEntry(source, options.Detached());
            _affineSources.Add(key, entry);
        }
        var generation = DeviceDataLoads.Generation;
        if (_dataWritePending is not null && entry.DeviceDataGeneration != generation)
        {
            if (DeviceDataLoads.CountFor(source.Hash) != entry.DeviceDataLoadCount)
            {
                try
                {
                    entry = CreateEntry(source, options.Detached());
                    _affineSources[key] = entry;
                }
                catch (ShaderProgramRejectedException)
                {
                    // Keep the original flattened plan, as the ordinary replan path does.
                }
            }
            entry.DeviceDataGeneration = generation;
        }
        _lastAffineSourceKey = key;
        _lastAffineSource = entry;

        // Test the count first: no resource scans, eligibility checks or draw storage
        // for the overwhelming majority of vertex programs with no affine loads.
        if (entry.AffineFetches.Count == 0 || (entry.HasShaderCalls && _compiler.LinkedShaderCallsSupported))
        {
            VertexFetchCounters.Refuse(VertexFetchRefusal.NoAffineIndex);
            return entry;
        }
        var decision = entry.Affine!;
        if (!decision.Eligible || (options.PrimitiveShader is not null &&
            (options.MeshPerVertexParameterMask != 0 || decision.RequiresWorkgroup)))
        {
            VertexFetchCounters.Refuse(VertexFetchRefusal.Eligibility);
            VertexFetchCounters.AffineReject();
            return entry;
        }

        if (decision.NeedsGuestReadScope) _host.BeginGuestWordReads();
        bool read;
        try { read = RuntimeValueEvaluator.EvaluateBufferDescriptorWords(entry.Plan, decision.Sources, inputs, decision.FreshWords, ref decision.EvaluationState); }
        finally { if (decision.NeedsGuestReadScope) _host.EndGuestWordReads(); }
        if (!read)
        {
            // Do not cache an unreadable/transient descriptor. Portable materialization
            // owns its existing failure/retry handling.
            VertexFetchCounters.AffineMiss();
            VertexFetchCounters.AffineReject();
            return entry;
        }
        decision.Resolve(source, info, _host, AffineCacheEnabled);
        return entry;
    }
}

// One bounded last-decision cache per portable program. Storage is allocated at
// planning/miss time, never on a hit. The key contains descriptor values (not just
// their addresses), effective symbolic biases, complete draw bounds and capacity.
internal sealed class AffineProgramDecision
{
    private readonly record struct Candidate(AffineVertexFetchLoad Load, VertexFetchRefusal Refusal);
    private Candidate[] _candidates = [];
    public required uint[] Sources { get; init; }
    public required uint[] FreshWords { get; init; }
    public RuntimeValueEvaluator.BufferDescriptorEvaluationState? EvaluationState;
    private uint[] _words = [];
    private uint[] _biases = [];
    private bool[] _validBiases = [];
    private VertexFetchBounds _bounds;
    private uint _capacity;
    private long? _mappingGeneration;
    private bool _usedMapping, _valid;
    private VertexAttributeResource[] _attributes = [];
    private VertexInputBuffer[] _buffers = [];
    private uint[] _pcs = [];
    public bool Eligible { get; private init; }
    public bool RequiresWorkgroup { get; private init; }
    public bool NeedsGuestReadScope { get; private init; }

    public static AffineProgramDecision Create(ProgramSourceEntry entry)
    {
        var eligible = !entry.Plan.Info.WritesDeviceAddresses;
        foreach (var buffer in entry.Plan.Info.Buffers) eligible &= !buffer.Written;
        foreach (var image in entry.Plan.Info.Images) eligible &= !image.Written;
        var candidates = new Candidate[entry.AffineFetches.Count];
        var sources = new uint[candidates.Length];
        var needsGuestReadScope = entry.Plan.ResourceBranches.Count != 0;
        Array.Fill(sources, uint.MaxValue);
        for (var index = 0; index < candidates.Length; index++)
        {
            var load = entry.AffineFetches[index];
            var access = entry.Plan.Memory.Find(load.Pc);
            var refusal = VertexFetchRefusal.None;
            if (access is null || access.DeviceDescriptor || access.Resource >= entry.Plan.Info.Buffers.Count ||
                access.BufferDescriptor?.ChooseStrategy(false, !load.Raw, false) == BufferLoweringStrategy.BoundedCandidateTable)
                refusal = VertexFetchRefusal.Resource;
            else
            {
                var source = entry.Plan.Info.Buffers[(int)access.Resource].Source;
                if (entry.Plan.DescriptorSources[(int)source].DwordCount < 4) refusal = VertexFetchRefusal.Descriptor;
                else
                {
                    sources[index] = source;
                    foreach (var word in entry.Plan.DescriptorSources[(int)source].Dwords)
                        needsGuestReadScope |= word.Kind is not (ScalarValueKind.Constant or ScalarValueKind.UserData);
                }
            }
            candidates[index] = new(load, refusal);
        }
        return new AffineProgramDecision
        {
            _candidates = candidates, Sources = sources, FreshWords = new uint[candidates.Length * 4],
            _words = new uint[candidates.Length * 4], _biases = new uint[candidates.Length], _validBiases = new bool[candidates.Length],
            Eligible = eligible, RequiresWorkgroup = ShaderCompileRequest.RequiresPrimitiveShaderWorkgroup(entry.Program),
            NeedsGuestReadScope = needsGuestReadScope,
        };
    }

    private static bool TryBias(ShaderSource source, AffineVertexFetchLoad load, out uint bias)
    {
        bias = load.Index.Bias;
        var scalar = load.Index.BiasScalarRegister;
        if (scalar < 0) return true;
        var word = scalar - (int)source.UserDataBase;
        if ((uint)word >= source.UserData.Length ||
            (scalar < 64 && (source.ExcludedUserDataRegisters & (1ul << scalar)) != 0)) return false;
        bias = unchecked(bias + (source.UserData[word] & load.Index.BiasMask));
        return true;
    }

    public void Resolve(ShaderSource source, VertexInputInfo info, IShaderPipelineHost host, bool cacheEnabled)
    {
        var capacity = Math.Min(VertexInputInfo.MaxBuffers, host.MaxFixedFunctionVertexBuffers);
        var hit = cacheEnabled && _valid && _bounds == info.FetchBounds && _capacity == capacity &&
            FreshWords.AsSpan().SequenceEqual(_words);
        for (var index = 0; index < _candidates.Length; index++)
        {
            var valid = TryBias(source, _candidates[index].Load, out var bias);
            hit &= _validBiases[index] == valid && _biases[index] == bias;
            _validBiases[index] = valid;
            _biases[index] = bias;
        }
        long? generation = null;
        if (hit && _usedMapping)
        {
            generation = host.VertexFetchMappingGeneration;
            hit = generation.HasValue && generation == _mappingGeneration;
        }
        if (hit) VertexFetchCounters.AffineHit();
        else
        {
            VertexFetchCounters.AffineMiss();
            _valid = false;
            _bounds = info.FetchBounds;
            _capacity = capacity;
            _usedMapping = false;
            var limit = (int)Math.Min(capacity, (uint)_candidates.Length);
            Span<VertexAttributeResource> attributes = stackalloc VertexAttributeResource[limit];
            Span<VertexInputBuffer> buffers = stackalloc VertexInputBuffer[limit];
            Span<uint> pcs = stackalloc uint[limit];
            var count = 0;
            for (var index = 0; index < _candidates.Length; index++)
            {
                var candidate = _candidates[index];
                if (count >= capacity)
                { VertexFetchCounters.Refuse(VertexFetchRefusal.Capacity); continue; }
                if (candidate.Refusal != VertexFetchRefusal.None)
                { VertexFetchCounters.Refuse(candidate.Refusal); continue; }
                if (!_validBiases[index])
                { VertexFetchCounters.Refuse(VertexFetchRefusal.Index); continue; }
                var load = candidate.Load;
                var resolvedIndex = load.Index;
                if (resolvedIndex.BiasScalarRegister >= 0)
                    resolvedIndex = resolvedIndex with { Bias = _biases[index], BiasScalarRegister = -1 };
                if (!AffineVertexFetchResolver.TryResolve(load, resolvedIndex, BufferDescriptorWords.From(FreshWords.AsSpan(index * 4, 4)),
                    info.FetchBounds, out var buffer, out var attribute, out var refusal))
                { VertexFetchCounters.Refuse(refusal); continue; }
                if (!_usedMapping && cacheEnabled) generation = host.VertexFetchMappingGeneration;
                _usedMapping = true;
                if (!host.VertexFetchRangeIsMapped(buffer.Address, buffer.Size))
                { VertexFetchCounters.Refuse(VertexFetchRefusal.Mapping); continue; }
                attributes[count] = attribute with { AttributeId = count, BufferIndex = count };
                buffers[count] = buffer;
                pcs[count++] = load.Pc;
            }
            if (!attributes[..count].SequenceEqual(_attributes)) _attributes = attributes[..count].ToArray();
            if (!buffers[..count].SequenceEqual(_buffers)) _buffers = buffers[..count].ToArray();
            if (!pcs[..count].SequenceEqual(_pcs)) _pcs = pcs[..count].ToArray();
            FreshWords.CopyTo(_words, 0);
            _mappingGeneration = generation;
            // A mapping mutation during the proof must never publish a reusable answer.
            _valid = cacheEnabled && (!_usedMapping || (generation.HasValue && generation == host.VertexFetchMappingGeneration));
        }
        if (_pcs.Length == 0)
        {
            VertexFetchCounters.AffineReject();
            return;
        }
        for (var index = 0; index < _pcs.Length; index++) VertexFetchCounters.Replaced();
        info.Attributes = _attributes;
        info.Buffers = _buffers;
        info.AffineFetchPcs = _pcs;
    }
}
