// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost : IRecipeReplaySink
{
    private static readonly bool RecipeReplayDefaultEnabled = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_RECIPE_REPLAY") != "0";
    internal bool RecipeReplayEnabled { get; set; } = RecipeReplayDefaultEnabled;
    internal bool CaptureReferenceCommands { get; set; }
    IRecipeReplaySink? IShaderPipelineHost.RecipeReplay => RecipeReplayEnabled ? this : null;
    private bool _replayIssue, _replayRelocated, _replayCompute;
    private int _replayIssuePrograms, _replayIssueMisses, _replayIssueBindings, _replayIssueBindingMisses, _replayIssuePipelines, _replayIssuePipelineMisses, _replayEmissions;
    private long _replayExact, _replayRelocation, _replayFallback, _replayNoEmission, _replayValidateTicks, _replaySavedTicks, _replayStages, _replayBindings, _replayBindingRefused, _replayArguments, _replayPipelines;
    private readonly long[] _replayReasons = new long[Enum.GetValues<ResourceDependencyStatus>().Length];
    private long _replayFrame;
    private long _replayArgumentPatches;
    private long _replayDrawExact, _replayDrawRelocation, _replayDrawFallback, _replayDispatchExact, _replayDispatchRelocation, _replayDispatchFallback;
    private MetalRecipeCommands? _replayCommands = RecipeReplayDefaultEnabled ? new() : null;
    private MetalRecipeCommands? _previousReplayCommands = RecipeReplayDefaultEnabled ? new() : null;
    private long _totalReplayStages, _totalReplayIssues, _totalReplayArguments;
    internal (long Stages, long Issues, long Arguments) ReplayCounters => (_totalReplayStages, _totalReplayIssues, _totalReplayArguments);
    private MetalRecipeOperation _replayOperation;
    private ulong _replayA, _replayB, _replayC, _replayD, _replayE;

    private readonly struct ReplayIssueScope(MetalRenderHost? host) : IDisposable
    {
        public void Dispose() => host?.ReplayEndIssue();
    }
    private ReplayIssueScope ReplayBeginIssue(bool compute)
    {
        if (!RecipeReplayEnabled && !CaptureReferenceCommands) return default;
        _replayCommands ??= new(); _previousReplayCommands ??= new();
        _replayIssue = true; _replayCompute = compute; _replayRelocated = false;
        _replayIssuePrograms = _replayIssueMisses = _replayIssueBindings = _replayIssueBindingMisses = 0;
        _replayIssuePipelines = _replayIssuePipelineMisses = _replayEmissions = 0;
        _replayOperation = MetalRecipeOperation.None;
        return new(this);
    }
    private void ReplayEndIssue()
    {
        _replayIssue = false;
        if (_replayEmissions == 0) { _replayNoEmission++; return; }
        if (_replayIssuePrograms != 0 && _replayIssueBindings == _replayIssuePrograms &&
            _replayIssuePipelines != 0 && _replayIssueMisses == 0 && _replayIssueBindingMisses == 0 && _replayIssuePipelineMisses == 0)
        {
            if (_replayRelocated) _replayRelocation++; else _replayExact++;
            if (_replayCompute) { if (_replayRelocated) _replayDispatchRelocation++; else _replayDispatchExact++; }
            else { if (_replayRelocated) _replayDrawRelocation++; else _replayDrawExact++; }
            _totalReplayIssues++;
        }
        else
        {
            _replayFallback++;
            if (_replayCompute) _replayDispatchFallback++; else _replayDrawFallback++;
        }
    }
    void IRecipeReplaySink.ProgramReplay(bool hit, bool relocated, ResourceDependencyStatus reason, long ticks, long saved)
    {
        _replayValidateTicks += ticks;
        if (hit)
        {
            _replayStages++; _replaySavedTicks += Math.Max(0, saved - ticks);
            _totalReplayStages++;
            if (_replayIssue) { _replayIssuePrograms++; _replayRelocated |= relocated; }
        }
        else
        {
            _replayReasons[(int)reason]++;
            if (_replayIssue) _replayIssueMisses++;
        }
    }
    void IRecipeReplaySink.PipelineReplay(bool hit)
    {
        if (hit) { _replayPipelines++; if (_replayIssue) _replayIssuePipelines++; }
        else if (_replayIssue) _replayIssuePipelineMisses++;
    }

    private void ReplayOperation(MetalRecipeOperation operation, ulong a = 0, ulong b = 0, ulong c = 0, ulong d = 0, ulong e = 0)
    {
        if (!RecipeReplayEnabled && !CaptureReferenceCommands && !IcbEnabled) return;
        _replayOperation = operation; _replayA = a; _replayB = b; _replayC = c; _replayD = d; _replayE = e;
    }
    private void ReplayRecordEmission()
    {
        if ((!RecipeReplayEnabled && !CaptureReferenceCommands) || !_replayIssue) return;
        _replayEmissions++;
        AppendRecipeCommand(_replayCommands!, _replayCompute);
    }
    private void AppendRecipeCommand(MetalRecipeCommands sequence, bool compute)
    {
        if (sequence.Commands.Count >= 65536 || sequence.Bindings.Count >= 1024 * 1024) { sequence.Dropped++; return; }
        var first = sequence.Bindings.Count;
        if (!compute)
            for (var i = 0; i < _vertexBufferCount; i++) sequence.Bindings.Add(new(0, i + VertexBufferSlotBase, _vertexBuffers[i], _vertexOffsets[i]));
        for (var stage = 0; stage < StageCount; stage++)
        {
            if (compute != (stage == StageCompute)) continue;
            if (_stageArgumentValid[stage]) sequence.Bindings.Add(new(stage, 0, _stageArgumentBuffers[stage], _stageArgumentOffsets[stage],
                Math.Max(ArgumentPlanOf(_stageBindings[stage].Program.Bindings!).Layout.ByteSize, 16u)));
            if (_hasPushData && (_preparedStageMask & (1 << stage)) != 0)
                sequence.Bindings.Add(new(stage, 1, _pushBuffer, _pushOffset, PushData.ByteSize));
        }
        var pipeline = compute ? _pendingComputePipeline : _boundGraphicsPipeline;
        var native = pipeline is not null && pipeline.Mesh is null && pipeline.RectangleStates is null &&
            (compute || FacePassCount(pipeline) == 1) && _replayOperation is MetalRecipeOperation.Draw or MetalRecipeOperation.DrawIndexed;
        sequence.Commands.Add(new(_submissionSerial, compute ? ComputeEncoders : _renderPassSerial,
            compute, pipeline?.State ?? 0, _replayOperation, _replayA, _replayB, _replayC, _replayD, _replayE,
            first, sequence.Bindings.Count - first, _indexBuffer, _indexOffset, _indexType, _dynamicState,
            pipeline?.PrimitiveType ?? 0, pipeline?.ThreadsPerThreadgroup ?? default, native));
    }
    internal MetalRecipeCommands CaptureReplayCommands()
    {
        if (!_relay.IsGpuQueueThread) throw new InvalidOperationException("Replay commands belong to the issue consumer.");
        var result = new MetalRecipeCommands();
        if (_replayCommands is { } source)
        {
            result.Commands.AddRange(source.Commands); result.Bindings.AddRange(source.Bindings); result.Dropped = source.Dropped;
        }
        return result;
    }
    private void ReplayEndFrame()
    {
        if (!RecipeReplayEnabled) return;
        ReportReplay("frame", ++_replayFrame);
        (_replayCommands, _previousReplayCommands) = (_previousReplayCommands, _replayCommands);
        _replayCommands!.Clear();
    }
    private void ReportReplay(string scope, long frame)
    {
        if (!RecipeReplayEnabled) return;
        var scale = 1000.0 / Stopwatch.Frequency;
        Console.Error.WriteLine(FormattableString.Invariant($"[METAL-V2][REPLAY] scope={scope} frame={frame} exact={_replayExact} relocation={_replayRelocation} fallback={_replayFallback} no_emission={_replayNoEmission} stages={_replayStages} bindings={_replayBindings} arguments={_replayArguments} pipelines={_replayPipelines} binding_refused={_replayBindingRefused} validate_ms={_replayValidateTicks * scale:F3} saved_estimate_ms={_replaySavedTicks * scale:F3} refusals=[{string.Join(',', _replayReasons.Select((count, i) => $"{(ResourceDependencyStatus)i}:{count}"))}] commands={_replayCommands?.Commands.Count ?? 0} dropped={_replayCommands?.Dropped ?? 0} arena_upload_bytes={_replayArena?.UploadedBytes ?? 0} arena_hits={_replayArena?.Hits ?? 0} arena_refused={_replayArena?.Refused ?? 0}"));
        if (scope != "frame") return; // Periodic samples must not truncate the per-frame denominator.
        Console.Error.WriteLine($"[METAL-V2][REPLAY] scope=frame_classes frame={frame} draw_exact={_replayDrawExact} draw_relocation={_replayDrawRelocation} draw_fallback={_replayDrawFallback} dispatch_exact={_replayDispatchExact} dispatch_relocation={_replayDispatchRelocation} dispatch_fallback={_replayDispatchFallback}");
        _replayDrawExact = _replayDrawRelocation = _replayDrawFallback = _replayDispatchExact = _replayDispatchRelocation = _replayDispatchFallback = 0;
        Console.Error.WriteLine($"[METAL-V2][REPLAY] scope=argument_patches frame={frame} patched={_replayArgumentPatches}");
        _replayArgumentPatches = 0;
        _replayExact = _replayRelocation = _replayFallback = _replayNoEmission = _replayStages = _replayBindings = _replayArguments = _replayPipelines = 0;
        _replayBindingRefused = _replayValidateTicks = _replaySavedTicks = 0;
        Array.Clear(_replayReasons);
    }
}
