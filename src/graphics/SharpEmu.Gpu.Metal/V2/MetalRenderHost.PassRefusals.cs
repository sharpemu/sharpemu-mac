// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost
{
    private enum JoinRefusal
    {
        Disabled, Pipeline, ColorCount, Size, Layers, Samples, Depth, Clear,
        Rectangle, Exports, Texture, Format, Subresource, Sampled, Bindless, Copies,
        Compiling, Launch, PendingClear, Refresh, Count,
    }

    private enum LoadRefusal
    {
        Disabled, Pipeline, Mesh, Positions, Discard, Exports, Blend, WriteMask, Culled, CoverageUnknown,
        Indirect, NoInstances, Restart, IndexCount, IndexRange, IndexRead, VertexRange,
        SampleMask, Size, DepthTest, StencilTest, Cull, Samples, Layers, Scissor,
        Coverage, Clear, Count,
    }

    private readonly long[] _joinRefusals = new long[(int)JoinRefusal.Count];
    private readonly long[] _loadRefusals = new long[(int)LoadRefusal.Count];
    private enum StoreDecision { ColorClear, DepthClear, StencilClear, NoFullClear, UnknownSuccessor, Disabled, Count }
    private readonly long[] _storeDecisions = new long[(int)StoreDecision.Count];
    private long[]? _passLoadRefusals;

    private void NoteStoreDecision(bool discard, StoreDecision clearedPlane, bool hasNext, long bytes)
    {
        var reason = !MetalV2Diagnostics.DeferStoreActions ? StoreDecision.Disabled
            : discard ? clearedPlane
            : hasNext || _clearedTexture != 0 ? StoreDecision.NoFullClear : StoreDecision.UnknownSuccessor;
        _storeDecisions[(int)reason]++;
        if (_ppOpen is { } pass)
        {
            pass.StoreDecisions ??= new long[(int)StoreDecision.Count];
            pass.StoreDecisions[(int)reason]++;
            if (discard) pass.DiscardedBytes += bytes;
        }
        if (discard) _discardedBytes += bytes;
    }

    internal string TakeStoreDecisions() => FormatRefusals<StoreDecision>(_storeDecisions, take: true);

    private bool RefuseJoin(JoinRefusal reason)
    {
        _joinRefusals[(int)reason]++;
        if (_ppOpen is { } pass)
        {
            pass.JoinRefusals ??= new long[(int)JoinRefusal.Count];
            pass.JoinRefusals[(int)reason]++;
        }
        return false;
    }

    private void RefuseLoad(LoadRefusal reason)
    {
        _loadRefusals[(int)reason]++;
        if (PassProfileEnabled)
        {
            _passLoadRefusals ??= new long[(int)LoadRefusal.Count];
            _passLoadRefusals[(int)reason]++;
        }
    }

    private LoadRefusal OverwriteRefusal(MetalPipelineEntry entry, int index)
    {
        if (entry.GraphicsDescription is not { } description) return LoadRefusal.Pipeline;
        if (description.PixelStage is null || !description.PixelProgram.IsValid) return LoadRefusal.Exports;
        var shader = LookupModule(description.PixelProgram).Shader;
        if (!shader.CannotDiscard) return LoadRefusal.Discard;
        if (description.StaticParameters.Samples > 1) return LoadRefusal.Samples;
        if ((shader.ColorOutputMask & (1u << index)) == 0) return LoadRefusal.Exports;
        var parameters = description.StaticParameters;
        var format = MetalV2Formats.PixelFormat(description.Rendering.ColorFormats[index]);
        return parameters.GetBlendEnable(index) && !parameters.GetBlendBypass(index) && !IsIntegerFormat(format)
            ? LoadRefusal.Blend : LoadRefusal.WriteMask;
    }

    // Join failures belong to the open pass that has to end; load failures belong
    // to the new pass. PERF takes interval totals without resetting either record.
    private static string FormatRefusals<T>(long[]? counts, bool take = false) where T : struct, Enum
    {
        if (counts is null) return "-";
        var items = new List<string>();
        for (var index = 0; index < counts.Length; index++)
        {
            var count = take ? Interlocked.Exchange(ref counts[index], 0) : counts[index];
            if (count != 0) items.Add($"{Enum.GetName(typeof(T), index)!.ToLowerInvariant()}:{count}");
        }
        return items.Count == 0 ? "-" : string.Join(',', items);
    }

    internal (string Joins, string Loads) TakePassRefusals() =>
        (FormatRefusals<JoinRefusal>(_joinRefusals, take: true),
         FormatRefusals<LoadRefusal>(_loadRefusals, take: true));
}
