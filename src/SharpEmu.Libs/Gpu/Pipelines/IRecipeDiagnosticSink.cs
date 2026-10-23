// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Libs.Gpu.Pipelines;

// Optional observer on the materialization owner. No guest reads, validation, or replay
// authority is conferred by these observations. The sink must not retain mutable inputs.
public interface IRecipeDiagnosticSink
{
    void Materialized(ShaderSource source, ShaderResourcePlan plan, ResourceSnapshot snapshot);
    void MaterializationElapsed(long ticks);
}

internal readonly struct RecipeMaterializationTimer : IDisposable
{
    private readonly IRecipeDiagnosticSink? _sink;
    private readonly long _start;
    public RecipeMaterializationTimer(IRecipeDiagnosticSink? sink)
    {
        _sink = sink;
        _start = sink is null ? 0 : Stopwatch.GetTimestamp();
    }
    public void Dispose()
    {
        if (_sink is not null) _sink.MaterializationElapsed(Stopwatch.GetTimestamp() - _start);
    }
}
