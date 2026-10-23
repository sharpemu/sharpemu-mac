// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// How many shader modules (an MTLLibrary with its MTLFunction) and pipeline states the
/// render host keeps resident, and how long an unused one is kept. Measured on an M4
/// (MetalShaderObjectCostTests): a Demon's Souls library is ~48 KB of footprint, its function
/// ~1 KB, and a pipeline state ~10 KB of malloc for a small kernel (its binary, which the
/// state keeps while either it or the function lives). The defaults bound the two to about
/// 2048 x 48 KB + 8192 x 16 KB, ~224 MB, where they used to grow for as long as a title ran.
/// Only the counts bound them by default: a UE5 title (Silent Hill) that was cut on idle time
/// rebuilt 178 modules and 213 pipeline states on the GPU worker for ~35 MB of memory.
/// </summary>
/// <remarks>
/// A released object is made again on its next use through the path that made it: a module
/// translates its request again (MetalCompiledGuestShader.AcquireShader) and compiles the
/// library, a pipeline state is built from its description. Both compiles are hits in
/// macOS's Metal shader cache for anything this process compiled before (about a
/// millisecond instead of tens to hundreds), so the cost of a rebuilt module is mostly the
/// translation, synchronous on the GPU worker. Nothing used in the last
/// <see cref="MinimumIdleSeconds"/> is released even over a limit, so a working set larger
/// than the limit is kept rather than rebuilt every frame.
/// SHARPEMU_METAL_V2_PIPELINE_LIMIT / _MODULE_LIMIT (count, 0 = no limit),
/// SHARPEMU_METAL_V2_PIPELINE_IDLE_S / _MODULE_IDLE_S (seconds, 0 = kept however long idle, the default);
/// SHARPEMU_METAL_V2_SHADER_EVICTION=0 keeps everything, as before.
/// </remarks>
internal readonly record struct MetalShaderCacheLimits(
    int Pipelines,
    double PipelineIdleSeconds,
    int Modules,
    double ModuleIdleSeconds,
    double MinimumIdleSeconds)
{
    public const int DefaultPipelines = 8192;
    public const double DefaultPipelineIdleSeconds = 0;
    public const int DefaultModules = 2048;
    public const double DefaultModuleIdleSeconds = 0;
    public const double DefaultMinimumIdleSeconds = 10;

    // How long a prewarmed library no module took is held when the module idle time is off.
    public const double PrewarmLibraryIdleSeconds = 180;

    // What the telemetry's estimate counts per resident object (the measurements above).
    public const long EstimatedModuleBytes = 48 * 1024;
    public const long EstimatedPipelineBytes = 16 * 1024;

    public static readonly MetalShaderCacheLimits Default = new(
        DefaultPipelines, DefaultPipelineIdleSeconds, DefaultModules, DefaultModuleIdleSeconds, DefaultMinimumIdleSeconds);

    public static readonly MetalShaderCacheLimits Unbounded = new(0, 0, 0, 0, DefaultMinimumIdleSeconds);

    public static readonly MetalShaderCacheLimits Configured = FromEnvironment(Environment.GetEnvironmentVariable);

    public bool Enabled => Pipelines > 0 || PipelineIdleSeconds > 0 || Modules > 0 || ModuleIdleSeconds > 0;

    internal static MetalShaderCacheLimits FromEnvironment(Func<string, string?> read)
    {
        if (read("SHARPEMU_METAL_V2_SHADER_EVICTION") == "0")
        {
            return Unbounded;
        }

        return new MetalShaderCacheLimits(
            (int)Number(read("SHARPEMU_METAL_V2_PIPELINE_LIMIT"), DefaultPipelines),
            Number(read("SHARPEMU_METAL_V2_PIPELINE_IDLE_S"), DefaultPipelineIdleSeconds),
            (int)Number(read("SHARPEMU_METAL_V2_MODULE_LIMIT"), DefaultModules),
            Number(read("SHARPEMU_METAL_V2_MODULE_IDLE_S"), DefaultModuleIdleSeconds),
            DefaultMinimumIdleSeconds);
    }

    private static double Number(string? text, double fallback) =>
        double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var value) && value >= 0 && value <= int.MaxValue
            ? value
            : fallback;
}

// The bound on the module and pipeline-state objects (MetalShaderCacheLimits). The entries
// themselves stay: the executor holds pipeline handles and module handles for the life of
// the host, so only the native objects behind them are released and made again on use.
internal sealed partial class MetalRenderHost
{
    internal MetalShaderCacheLimits ShaderCacheLimits { get; set; } = MetalShaderCacheLimits.Configured;

    private long _residentPipelines;
    private long _residentModules;
    private long _pipelineEvictions;
    private long _moduleEvictions;
    private long _pipelineRebuilds;
    private long _moduleRebuilds;

    // When the command buffer being recorded was opened, the time every use in it is stamped with.
    private long _commandBufferTicks;
    private long _nextShaderTrimTicks;

    internal (long ResidentPipelines, long ResidentModules, long PipelineEvictions, long ModuleEvictions,
        long PipelineRebuilds, long ModuleRebuilds) ShaderCacheCounters =>
        (Interlocked.Read(ref _residentPipelines), Interlocked.Read(ref _residentModules),
            Interlocked.Read(ref _pipelineEvictions), Interlocked.Read(ref _moduleEvictions),
            Interlocked.Read(ref _pipelineRebuilds), Interlocked.Read(ref _moduleRebuilds));

    /// <summary>What the dedupe shared, and the distinct pipeline states and libraries alive, for tests.</summary>
    internal (long PipelineDedups, long ModuleSameSource, int SharedStates, int SharedLibraries) DedupeCounters =>
        (Interlocked.Read(ref _pipelineDedups), Interlocked.Read(ref _moduleSameSource), _states.Count, _libraries.Count);

    /// <summary>The newest command buffer that encoded a resident pipeline state, for tests.</summary>
    internal ulong NewestPipelineUseSerial =>
        _pipelineEntries.Values.Where(entry => entry.State != 0).Select(entry => entry.LastUsedSerial).DefaultIfEmpty().Max();

    // A pipeline's state is about to be encoded into the open command buffer: made again if
    // the bound released it, and stamped so it is kept until that command buffer is done.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void UsePipelineState(MetalPipelineEntry entry)
    {
        if (entry.State == 0)
        {
            RebuildPipelineState(entry);
        }

        entry.LastUsedSerial = _submissionSerial;
        entry.LastUsedTicks = _commandBufferTicks;
    }

    // With a new command buffer open (no encoder of an older one is), at most once a second.
    private void MaybeTrimShaderCache(ulong completed)
    {
        _commandBufferTicks = Stopwatch.GetTimestamp();
        if (_commandBufferTicks < _nextShaderTrimTicks || !ShaderCacheLimits.Enabled)
        {
            return;
        }

        _nextShaderTrimTicks = _commandBufferTicks + Stopwatch.Frequency;
        TrimShaderCache(completed, _commandBufferTicks, ShaderCacheLimits);
    }

    /// <summary>
    /// Releases the pipeline states and modules the limits no longer keep: idle past their
    /// idle time, or the oldest past the count limit, never one a command buffer that is not
    /// done yet encoded or one used in the last <see cref="MetalShaderCacheLimits.MinimumIdleSeconds"/>.
    /// On the GPU worker, between command buffers.
    /// </summary>
    internal (int Pipelines, int Modules) TrimShaderCache(ulong completed, long now, MetalShaderCacheLimits limits)
    {
        var pipelines = TrimPipelineStates(completed, now, limits);
        // A pipeline compiling on the pool reads its module's function there.
        var modules = PipelinesCompiling == 0 ? TrimModules(now, limits) : 0;
        if (limits.Enabled)
        {
            // A library the prewarm made and no module took is not a shader in use.
            var prewarmIdle = limits.ModuleIdleSeconds > 0 ? limits.ModuleIdleSeconds : MetalShaderCacheLimits.PrewarmLibraryIdleSeconds;
            MetalLibraryPrewarm.ReleaseIdle(now - SecondsToTicks(prewarmIdle));
        }

        return (pipelines, modules);
    }

    private static long SecondsToTicks(double seconds) => (long)(seconds * Stopwatch.Frequency);

    private int TrimPipelineStates(ulong completed, long now, MetalShaderCacheLimits limits)
    {
        var recent = now - SecondsToTicks(limits.MinimumIdleSeconds);
        var idle = limits.PipelineIdleSeconds > 0 ? now - SecondsToTicks(limits.PipelineIdleSeconds) : long.MinValue;
        var released = 0;
        List<MetalPipelineEntry>? oldest = null;
        foreach (var entry in _pipelineEntries.Values)
        {
            if (entry.State == 0 || entry.LastUsedSerial > completed || entry.LastUsedTicks > recent)
            {
                continue;
            }

            if (entry.LastUsedTicks <= idle)
            {
                ReleasePipelineState(entry);
                released++;
            }
            else if (limits.Pipelines > 0)
            {
                (oldest ??= []).Add(entry);
            }
        }

        var over = Interlocked.Read(ref _residentPipelines) - limits.Pipelines;
        if (limits.Pipelines > 0 && over > 0 && oldest is not null)
        {
            oldest.Sort(static (left, right) => left.LastUsedTicks.CompareTo(right.LastUsedTicks));
            for (var index = 0; index < oldest.Count && index < over; index++)
            {
                ReleasePipelineState(oldest[index]);
                released++;
            }

            // A count above the limit is left when everything over it was in use; it is
            // tried again at the next trim.
        }

        return released;
    }

    private int TrimModules(long now, MetalShaderCacheLimits limits)
    {
        var recent = now - SecondsToTicks(limits.MinimumIdleSeconds);
        var idle = limits.ModuleIdleSeconds > 0 ? now - SecondsToTicks(limits.ModuleIdleSeconds) : long.MinValue;
        List<MetalShaderModule>? expired = null;
        List<MetalShaderModule>? oldest = null;
        lock (_modules)
        {
            foreach (var module in _modules)
            {
                // A module still compiling on the pool is its compile's until it is done.
                if (Volatile.Read(ref module.Function) == 0 || module.BindingSuccessor || module.Compiling is { IsCompleted: false } ||
                    module.LastUsedTicks > recent)
                {
                    continue;
                }

                if (module.LastUsedTicks <= idle)
                {
                    (expired ??= []).Add(module);
                }
                else if (limits.Modules > 0)
                {
                    (oldest ??= []).Add(module);
                }
            }
        }

        var released = 0;
        foreach (var module in expired ?? [])
        {
            released += ReleaseModule(module) ? 1 : 0;
        }

        var over = Interlocked.Read(ref _residentModules) - limits.Modules;
        if (limits.Modules > 0 && over > 0 && oldest is not null)
        {
            oldest.Sort(static (left, right) => left.LastUsedTicks.CompareTo(right.LastUsedTicks));
            for (var index = 0; index < oldest.Count && over > 0; index++)
            {
                if (ReleaseModule(oldest[index]))
                {
                    released++;
                    over--;
                }
            }
        }

        return released;
    }

    private void ReleasePipelineState(MetalPipelineEntry entry)
    {
        ReleaseState(entry.State);
        ReleaseState(entry.BatchState);
        entry.BatchState = 0;
        if (entry.RectangleStates is { } rectangles)
        {
            // The first rectangle state is entry.State itself.
            for (var index = 1; index < rectangles.Length; index++)
            {
                if (rectangles[index] != 0)
                {
                    ReleaseState(rectangles[index]);
                }
            }

            Array.Clear(rectangles);
        }

        entry.State = 0;
        Interlocked.Decrement(ref _residentPipelines);
        Interlocked.Increment(ref _pipelineEvictions);
    }

    private bool ReleaseModule(MetalShaderModule module)
    {
        lock (module)
        {
            if (module.Function == 0)
            {
                return false;
            }

            MetalNative.SendVoid(module.Function, SelRelease);
            if (module.LodFunction != 0)
            {
                MetalNative.SendVoid(module.LodFunction, SelRelease);
                module.LodFunction = 0;
            }

            ReleaseLibrary(module.Library);
            if (module.RectangleFunction != 0)
            {
                MetalNative.SendVoid(module.RectangleFunction, SelRelease);
                ReleaseLibrary(module.RectangleLibrary);
            }

            Volatile.Write(ref module.Function, 0);
            module.Library = 0;
            module.RectangleFunction = 0;
            module.RectangleLibrary = 0;
        }

        Interlocked.Decrement(ref _residentModules);
        Interlocked.Increment(ref _moduleEvictions);
        return true;
    }

    // A module whose library the bound released, built again from its translation request.
    private void RebuildModule(MetalShaderModule module)
    {
        lock (module)
        {
            if (module.Function != 0)
            {
                return;
            }

            BuildModule(module, module.Stage);
        }

        Interlocked.Increment(ref _moduleRebuilds);
    }

    // The released state of a pipeline, made again from the description it was created from.
    private void RebuildPipelineState(MetalPipelineEntry entry)
    {
        if (entry.Graphics)
        {
            var description = entry.GraphicsDescription ??
                throw Fatal($"A released render pipeline has no description to build it again from: pipeline={entry.Id}.");
            var vertexModule = RequireModule(description.VertexProgram);
            var pixelModule = description.PixelStage is not null && description.PixelProgram.IsValid
                ? RequireModule(description.PixelProgram)
                : null;
            var rectangleList = entry.RectangleStates is not null;
            var state = CreateRenderPipelineState(
                description, vertexModule, pixelModule, entry.Mesh, rectangleList, Mtl.ControlPointIndexTypeNone, out _);
            if (rectangleList)
            {
                // The indexed variants are made again when a draw asks for them.
                entry.RectangleStates![0] = state;
            }

            entry.State = state;
        }
        else
        {
            var description = entry.ComputeDescription ??
                throw Fatal($"A released compute pipeline has no description to build it again from: pipeline={entry.Id}.");
            var module = RequireModule(description.Program);
            entry.State = BuildComputePipelineState(module.Function, description.Stage.Hash, module.ArchiveKey);
        }

        Interlocked.Increment(ref _residentPipelines);
        Interlocked.Increment(ref _pipelineRebuilds);
    }

    // The [MEM] metal section's and the PERF line's view of the bound.
    private string DescribeShaderCache()
    {
        const double Mb = 1048576.0;
        int modules;
        lock (_modules)
        {
            modules = _modules.Count;
        }

        var (pipelines, residentModules, pipelineEvictions, moduleEvictions, pipelineRebuilds, moduleRebuilds) =
            ShaderCacheCounters;
        var prewarmHeld = MetalLibraryPrewarm.HeldLibraries;
        var estimated = (residentModules + prewarmHeld) * MetalShaderCacheLimits.EstimatedModuleBytes +
            pipelines * MetalShaderCacheLimits.EstimatedPipelineBytes;
        return string.Create(CultureInfo.InvariantCulture,
            $"shader_modules={residentModules}/{modules} module_evictions={moduleEvictions} module_rebuilds={moduleRebuilds} " +
            $"pipeline_states={pipelines}/{_pipelineEntries.Count} pipeline_evictions={pipelineEvictions} " +
            $"pipeline_rebuilds={pipelineRebuilds} pipeline_dedup={Interlocked.Read(ref _pipelineDedups)} " +
            $"module_same_source={Interlocked.Read(ref _moduleSameSource)} depth_stencil_states={_depthStencilStates.Count} " +
            $"prewarm_libraries={prewarmHeld} shader_cache_est_mb={estimated / Mb:F0}");
    }
}
