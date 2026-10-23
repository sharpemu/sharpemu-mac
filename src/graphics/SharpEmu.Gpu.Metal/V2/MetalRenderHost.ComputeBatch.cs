// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Globalization;
using SharpEmu.ShaderCompiler.Metal;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost
{
    internal bool ComputeBatchEnabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_COMPUTE_BATCH") != "0";

    // Bound table size, search depth and time spent retaining an unencoded launch.
    private const int ComputeBatchCapacity = 128;
    private readonly record struct ComputeBatchRequest(ulong Resources, ulong Push, uint First, uint End, uint X, uint Y);
    private readonly ComputeBatchRequest[] _computeBatch = new ComputeBatchRequest[ComputeBatchCapacity];
    private int _computeBatchCount;
    private uint _computeBatchGroups;
    private uint _dispatchInitiator, _batchInitiator;
    private MetalPipelineEntry? _computeBatchPipeline;
    private nint _batchFirstArguments, _batchFirstPush;
    private ulong _batchFirstArgumentOffset, _batchFirstPushOffset;
    private bool _batchFirstHasArguments, _batchFirstHasPush;
    internal long ComputeBatchedDispatches { get; private set; }
    internal long ComputeBatches { get; private set; }
    internal long ComputeBatchRecompiles { get; private set; }
    internal long ComputeBatchFallbacks { get; private set; }
    internal long ComputeBatchFlushes(string reason) => _computeBatchFlushReasons.GetValueOrDefault(reason);
    private readonly Dictionary<string, long> _computeBatchFlushReasons = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long> _computeBatchRefusals = new(StringComparer.Ordinal);

    private bool CanBatchCompute(MetalPipelineEntry entry, uint x, uint y, uint z, out uint groups)
    {
        // No inference about commutative atomics: shared writes still conflict in
        // the existing alias-aware dependency tracker. Unknown addressing, shader
        // diagnostics, pair locks and persistent queues have no independence proof.
        var product = (ulong)x * y;
        groups = 0;
        var reason = !ComputeBatchEnabled ? "disabled" : !ConcurrentComputeEnabled ? "serial" :
            MetalShaderProfile.Enabled || DumpTargetsEnabled ? "diagnostic" :
            (_dispatchInitiator & ~0xA079u) != 0 ? "initiator" :
            x == 0 || y == 0 || z == 0 || product > uint.MaxValue || product * z > uint.MaxValue ? "grid" :
            entry.PersistentThreads ? "persistent" : entry.WritesStepReport ? "step-report" :
            entry.UsesPairLocks ? "pair-locks" : _computeFootprint.UnknownWrite ? "unknown-write" :
            _computeFootprint.UnknownRead ? "unknown-read" :
            (_preparedStageMask & (1 << StageCompute)) == 0 || !_stageBindings[StageCompute].Committed ? "bindings" :
            entry.ComputeDescription is not { } description ||
            LookupModule(description.Program).Shader.ComputeBatchEntryPoint is null ? "entry-point" : null;
        if (reason is not null)
        {
            _computeBatchRefusals[reason] = _computeBatchRefusals.GetValueOrDefault(reason) + 1;
            return false;
        }
        // Resolve when a second request can join, before retaining its bindings.
        // On failure the queued singleton flushes normally and this dispatch keeps
        // its current bindings. Singletons never need the optional pipeline.
        if (entry.BatchUnavailable || (_computeBatchCount != 0 && ComputeBatchState(entry) == 0))
        {
            ComputeBatchFallbacks += 1 + _computeBatchCount;
            return false;
        }
        groups = (uint)(product * z);
        return true;
    }

    private void QueueComputeBatch(MetalPipelineEntry entry, uint x, uint y, uint groups)
    {
        if (_computeBatchCount == 0)
        {
            _computeBatchPipeline = entry;
            _batchInitiator = _dispatchInitiator;
            _batchFirstArguments = _stageArgumentBuffers[StageCompute];
            _batchFirstArgumentOffset = _stageArgumentOffsets[StageCompute];
            _batchFirstHasArguments = _stageArgumentValid[StageCompute];
            _batchFirstPush = _pushBuffer;
            _batchFirstPushOffset = _pushOffset;
            _batchFirstHasPush = _hasPushData;
        }

        _computeBatch[_computeBatchCount++] = new(
            _stageArgumentValid[StageCompute] ? _stageArgumentAddresses[StageCompute] : 0,
            _hasPushData ? _pushAddress : 0, _computeBatchGroups, _computeBatchGroups + groups, x, y);
        _computeBatchGroups += groups;
        // Name each request's indirect references while its binding snapshot is
        // current, without rebinding argument/push slots. Residency lasts through
        // the encoder; transient snapshots last through submission retirement.
        ApplyStageBindings(graphics: false, bindComputeBuffers: false);
    }

    private nint ComputeBatchState(MetalPipelineEntry entry)
    {
        if (entry.BatchState != 0) return entry.BatchState;
        if (entry.BatchUnavailable) return 0;
        var module = RequireModule(entry.ComputeDescription!.Program);
        var name = module.Shader.ComputeBatchEntryPoint!;
        var key = MetalBinaryArchive.ComputeBatchKey(module.ArchiveKey);
        var library = MetalBinaryArchive.For(_device)?.TryLoadLibrary(key, requiredEntryPoint: name) ?? 0;
        nint function = 0;
        try
        {
            // A cold source library already has both kernels. A warm archive made
            // from the ordinary pipeline may contain only the ordinary kernel.
            function = MetalNative.Send(library != 0 ? library : module.Library,
                MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString(name));
            if (function == 0)
            {
                ComputeBatchRecompiles++;
                var shader = module.Compiled.AcquireShader();
                if (library != 0) MetalNative.SendVoid(library, SelRelease);
                library = 0;
                library = TryCompileLibrary(_device, shader.Source, shader.UsesPerVertexInputs,
                    out _, out _, askArchive: false);
                module.Compiled.ReleaseSource();
                if (library != 0)
                    function = MetalNative.Send(library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString(name));
            }
            if (function == 0)
            {
                entry.BatchUnavailable = true;
                return 0;
            }
            entry.BatchState = BuildComputePipelineState(function, entry.VertexHash, key);
            return entry.BatchState;
        }
        finally
        {
            if (function != 0) MetalNative.SendVoid(function, SelRelease);
            if (library != 0) MetalNative.SendVoid(library, SelRelease);
        }
    }

    private void FlushComputeBatch(string reason)
    {
        var count = _computeBatchCount;
        if (count == 0) return;
        var entry = _computeBatchPipeline!;
        var groups = _computeBatchGroups;
        // Clear before profiling/allocating: either can ask for a submission boundary.
        _computeBatchCount = 0;
        _computeBatchGroups = 0;
        _computeBatchPipeline = null;
        _computeBatchFlushReasons[reason] = _computeBatchFlushReasons.GetValueOrDefault(reason) + 1;
        if (_ppOpen is { } record)
        {
            record.BatchFlushReasons ??= new(StringComparer.Ordinal);
            record.BatchFlushReasons[reason] = record.BatchFlushReasons.GetValueOrDefault(reason) + 1;
        }
        var encoder = _computeEncoder;
        UsePipelineState(entry);
        if (count == 1)
        {
            EnsureComputePipelineState(encoder, entry);
            if (_batchFirstHasArguments)
                MetalNative.SendSetBuffer(encoder, SelSetComputeBuffer, _batchFirstArguments,
                    (nuint)_batchFirstArgumentOffset, Gen5MslArgumentLayout.ResourcesBufferIndex);
            if (_batchFirstHasPush)
                MetalNative.SendSetBuffer(encoder, SelSetComputeBuffer, _batchFirstPush,
                    (nuint)_batchFirstPushOffset, Gen5MslArgumentLayout.PushDataBufferIndex);
            var request = _computeBatch[0];
            MetalNative.SendDispatch(encoder, SelDispatchThreadgroups,
                new MtlSize { Width = request.X, Height = request.Y, Depth = groups / request.X / request.Y },
                entry.ThreadsPerThreadgroup);
        }
        else
        {
            var state = ComputeBatchState(entry);
            var table = _guestMemory.Allocate((uint)(Gen5MslArgumentLayout.ComputeBatchHeaderBytes +
                count * Gen5MslArgumentLayout.ComputeBatchRequestBytes), 256, out var bytes);
            bytes.Clear();
            BinaryPrimitives.WriteUInt32LittleEndian(bytes, (uint)count);
            for (var index = 0; index < count; index++)
            {
                var request = _computeBatch[index];
                var at = bytes.Slice(Gen5MslArgumentLayout.ComputeBatchHeaderBytes +
                    index * Gen5MslArgumentLayout.ComputeBatchRequestBytes);
                BinaryPrimitives.WriteUInt64LittleEndian(at, request.Resources);
                BinaryPrimitives.WriteUInt64LittleEndian(at[8..], request.Push);
                BinaryPrimitives.WriteUInt32LittleEndian(at[16..], request.First);
                BinaryPrimitives.WriteUInt32LittleEndian(at[20..], request.End);
                BinaryPrimitives.WriteUInt32LittleEndian(at[24..], request.X);
                BinaryPrimitives.WriteUInt32LittleEndian(at[28..], request.Y);
            }
            if (_boundComputePipeline != state)
            {
                MetalNative.SendVoid(encoder, SelSetComputePipelineState, state);
                _boundComputePipeline = state;
            }
            // One argument bind and one launch for the entire batch. Each record
            // points at the immutable argument and push allocations of its request.
            MetalNative.SendSetBuffer(encoder, SelSetComputeBuffer, table.Buffer, (nuint)table.Offset, 0);
            NameResidentBuffers(encoder, _guestMemory.ResidentBuffers, stages: 0);
            MetalNative.SendDispatch(encoder, SelDispatchThreadgroups,
                new MtlSize { Width = groups, Height = 1, Depth = 1 }, entry.ThreadsPerThreadgroup);
            ComputeBatchedDispatches += count;
            ComputeBatches++;
            if (_ppOpen is { } pass)
            {
                pass.BatchedDispatches += count;
                pass.Batches++;
            }
        }
        ProfileWork(entry, count == 1 ? "dispatch" : "dispatch-batch",
            WorkShape.Groups(groups, 1, 1, entry.ThreadsPerThreadgroup));
    }

    private string ComputeBatchCounters() =>
        $"compute_batched_dispatches={ComputeBatchedDispatches} compute_batches={ComputeBatches} " +
        $"compute_batch_recompiles={ComputeBatchRecompiles} compute_batch_fallbacks={ComputeBatchFallbacks} " +
        $"compute_batch_refused=[{string.Join(',', _computeBatchRefusals.Select(pair => $"{pair.Key}:{pair.Value}"))}] " +
        $"compute_avg_batch={(ComputeBatches == 0 ? 0 : (double)ComputeBatchedDispatches / ComputeBatches).ToString("F2", CultureInfo.InvariantCulture)} " +
        $"compute_batch_flush=[{string.Join(',', _computeBatchFlushReasons.Select(pair => $"{pair.Key}:{pair.Value}"))}] ";
}
