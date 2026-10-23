// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Libs.Gpu.Pipelines;

internal sealed record PreparedProgramRecipe(ResourceReplayValidation Validation, ShaderSource Source,
    ShaderProgram Handle, ShaderProgramInfo Program, long CostTicks);

internal sealed partial class ShaderProgramCache
{
    private long _programRecipeBytes;
    private bool TryReplayProgram(ProgramSourceEntry entry, ShaderSource source, StageCompileOptions options,
        ref uint cursor, out ShaderProgram handle, out ShaderStageResources stage)
    {
        handle = default; stage = default;
        if (_spirvDumpEnabled || _host.RecipeReplay is not { } sink || _host is not IResourceDependencySource dependencies ||
            entry.HasShaderCalls || entry.Successor is not null) return false;
        var start = Stopwatch.GetTimestamp();
        var refusal = ResourceDependencyStatus.Unavailable;
        for (var index = entry.Recipes.Count - 1; index >= 0; index--)
        {
            var recipe = entry.Recipes[index];
            var layout = recipe.Program.Bindings!;
            // The real content identity and full static ProgramKey were checked by
            // the normal source/key lookup BEFORE reaching this fast path.
            if (recipe.Source.Identity != source.Identity || recipe.Source.Address != source.Address ||
                recipe.Source.Registered != source.Registered ||
                layout.PushDataStartDword != PushData.StartFor(cursor, layout.ShaderDataDwordCount)) continue;
            if (!recipe.Validation.TryReplay(dependencies, source.UserData, out var snapshot, out var patched, out refusal)) continue;
            stage = CreateStageResources(recipe.Program, snapshot, source, options);
            handle = recipe.Handle;
            // Keep the successful descriptor family first in the bounded probe
            // order; a stable draw must not validate 31 other read sets first.
            if (index != entry.Recipes.Count - 1)
            {
                entry.Recipes.RemoveAt(index); entry.Recipes.Add(recipe);
            }
            layout.AdvancePushData(ref cursor);
            // Retry bookkeeping must observe the same original reads. A suspended
            // compute attempt retains its own log rather than receiving an empty one.
            recipe.Validation.ObserveReads(_observeCachedReads, ReadLog is not null);
            _recipeDiagnostics?.Materialized(source, entry.Plan, snapshot);
            sink.ProgramReplay(true, patched, ResourceDependencyStatus.Current, Stopwatch.GetTimestamp() - start, recipe.CostTicks);
            return true;
        }
        sink.ProgramReplay(false, false, refusal, Stopwatch.GetTimestamp() - start, 0);
        return false;
    }

    private void KeepProgramRecipe(ProgramSourceEntry entry, ShaderSource source, ShaderProgram handle,
        ShaderStageResources stage, long cost)
    {
        if (_spirvDumpEnabled || _host.RecipeReplay is null || entry.HasShaderCalls || entry.Successor is not null ||
            _materializations?.ReplayValidation is not { } validation || _host is not IResourceDependencySource dependencies ||
            !ReferenceEquals(validation.Plan, entry.Plan) ||
            !validation.TryReplay(dependencies, source.UserData, out var observed, out _, out _) ||
            !SameSnapshot(observed, stage.Resources)) return;
        var bytes = 2048L + validation.RetainedBytes * 4;
        if (bytes > 32 * 1024 * 1024) return;
        if (_programRecipeBytes + bytes > 32 * 1024 * 1024)
        {
            foreach (var item in _programs.Values) item.Recipes.Clear();
            _programRecipeBytes = 0;
        }
        // Replacement is optimization state only. Its byte packet remains owned by
        // this recipe even after the materialization cache evicts its watch packet.
        if (entry.Recipes.Count >= 32) entry.Recipes.RemoveAt(0);
        entry.Recipes.Add(new(validation, source, handle, stage.Program!, cost));
        _programRecipeBytes += bytes;
    }

    private static bool SameSnapshot(ResourceSnapshot a, ResourceSnapshot b) =>
        SameDescriptors(a.Buffers, b.Buffers) && SameDescriptors(a.Images, b.Images) && SameDescriptors(a.Samplers, b.Samplers) &&
        a.UserData.AsSpan().SequenceEqual(b.UserData) && a.FlattenedResourceTable.AsSpan().SequenceEqual(b.FlattenedResourceTable) &&
        a.DeviceAddressRanges.AsSpan().SequenceEqual(b.DeviceAddressRanges) &&
        // Bindless candidates are immutable but have additional selection inputs.
        // Admit only the identical fully materialized table objects.
        a.BindlessImages.AsSpan().SequenceEqual(b.BindlessImages);

    private static bool SameDescriptors(uint[][] a, uint[][] b)
    {
        if (a.Length != b.Length) return false;
        for (var i = 0; i < a.Length; i++) if (!a[i].AsSpan().SequenceEqual(b[i])) return false;
        return true;
    }
}
