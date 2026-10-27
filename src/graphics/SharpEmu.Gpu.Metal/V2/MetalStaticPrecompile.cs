// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.IO.Hashing;
using System.Runtime.InteropServices;
using System.Text;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.ShaderCache;
using SharpEmu.Libs.VideoOut;
using SharpEmu.Logging;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Metal;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// The static shader precompile: programs the game has not drawn or dispatched yet - those
/// it creates this session (sceAgcCreateShader, ShaderInventory) and those the scanner finds
/// in its files - translated under the draw states of recorded programs of the same pixel
/// signature (StaticShaderPlanner) and compiled into the title's binary archive as library
/// files (MetalBinaryArchive.StoreLibrary), which the runtime opens instead of running the
/// MSL front end. Nothing compiled is kept: each library is released once its file is
/// written, so the footprint is a few libraries in flight.
///
/// By default it runs behind the game: it starts once the guest is no longer held for its
/// first frame's programs, on <see cref="DefaultJobs"/> below-normal threads that give way
/// to the game's own compiles (MetalCompileScheduler), reads only the files that name
/// shaders and the executables, and stops after <see cref="DefaultBudget"/> new libraries a
/// start or when the archive is <see cref="ArchiveShare"/> full, so what the game compiled
/// itself is not evicted for guesses. The precompile modes (--precompile-only, hold
/// complete) scan every file and compile everything left, holding the guest, with the
/// prewarm's thread count. A finished unit is listed in static-precompile-&lt;stamp&gt;.done
/// and not translated again under the same translators.
///
/// SHARPEMU_SHADER_CACHE=off turns it off (and the capture of created programs);
/// SHARPEMU_SHADER_PRECOMPILE_JOBS, SHARPEMU_SHADER_PRECOMPILE_BUDGET and
/// SHARPEMU_SHADER_SCAN=shaders|all|off tune it, and SHARPEMU_SHADER_PRECOMPILE_COMPUTE=1
/// adds compute programs (off: see <see cref="IncludeComputes"/>); SHARPEMU_SHADER_LEARN=on writes the
/// title's seed (user/shader_seeds/&lt;TITLE&gt;.seed) when the session ends.
/// </summary>
internal static class MetalStaticPrecompile
{
    internal const string CacheVariable = "SHARPEMU_SHADER_CACHE";
    internal const string LearnVariable = "SHARPEMU_SHADER_LEARN";
    internal const string JobsVariable = "SHARPEMU_SHADER_PRECOMPILE_JOBS";
    internal const string BudgetVariable = "SHARPEMU_SHADER_PRECOMPILE_BUDGET";
    internal const string ScanVariable = "SHARPEMU_SHADER_SCAN";
    internal const string ComputeVariable = "SHARPEMU_SHADER_PRECOMPILE_COMPUTE";

    /// <summary>
    /// Compute programs only with SHARPEMU_SHADER_PRECOMPILE_COMPUTE=1: a compute's
    /// specialization before its first dispatch is not yet what the dispatch asks for (on
    /// Demon's Souls 2 of 9 sampled recorded computes derived to the same MSL; the others
    /// lacked a buffer's bounds check), and the compute prewarm list covers the ones the
    /// title dispatched.
    /// </summary>
    internal static readonly bool IncludeComputes = Environment.GetEnvironmentVariable(ComputeVariable) == "1";
    internal const int DefaultJobs = 2;

    /// <summary>
    /// New libraries a background start stores at most. A Demon's Souls library file with
    /// its binary slice averaged 151 KB (vertex) and 78 KB (pixel) over 399 derived programs
    /// (ShaderCompileBench mode=static), so this is some 300 MB of the archive's 1 GiB, and
    /// about 20 minutes of two below-normal threads (0.77 s a library on an M4).
    /// </summary>
    internal const int DefaultBudget = 3000;

    /// <summary>The share of the archive's budget past which the precompile stores nothing more.</summary>
    internal const double ArchiveShare = 0.6;

    private const int MaxLoggedFailures = 8;
    private const int MaxStubVertices = 64;
    private const nuint PixelFormatDepth32Float = 252;
    private const nuint PixelFormatDepth32FloatStencil8 = 260;
    private const string StubEntryPoint = "sharpemu_static_vs";

    private static readonly nint SelRelease = MetalNative.Selector("release");
    private static readonly nint SelAlloc = MetalNative.Selector("alloc");
    private static readonly nint SelInit = MetalNative.Selector("init");
    private static readonly nint SelNewFunctionWithName = MetalNative.Selector("newFunctionWithName:");

    private static readonly object _gate = new();
    private static readonly ConcurrentQueue<StaticShaderJob> _created = new();
    private static readonly ConcurrentDictionary<ulong, byte> _done = new();
    private static readonly ConcurrentDictionary<UInt128, byte> _inFlight = new();
    private static readonly ConcurrentDictionary<string, nint> _stubs = new(StringComparer.Ordinal);
    private static readonly ManualResetEventSlim _finished = new(false);
    private static StaticShaderPlanner? _planner;
    private static FileStream? _doneFile;
    private static List<StaticShaderJob> _plan = [];
    private static MetalCompileThread[] _threads = [];
    private static int _next = -1;
    private static bool _started;
    private static volatile bool _stopping;
    private static volatile bool _phase;
    private static volatile bool _planned;
    private static readonly CancellationTokenSource _scanCancel = new();
    private static string _titleId = string.Empty;
    private static string _seedPath = string.Empty;
    private static long _startBytes;
    private static int _budget;
    private static int _stored;
    private static long _storedBytes;
    private static int _present;
    private static int _translated;
    private static int _underivable;
    private static int _failed;
    private static int _jobsDone;
    private static int _createdJobs;
    private static long _startedTicks;

    /// <summary>Whether the precompile (and the capture of created programs) runs: SHARPEMU_SHADER_CACHE, on unless off.</summary>
    public static bool Enabled { get; } = ShaderCacheSwitch(Environment.GetEnvironmentVariable(CacheVariable)) &&
        Environment.GetEnvironmentVariable("SHARPEMU_SHADER_PREWARM") != "0";

    /// <summary>Whether the session's end writes the title's seed: SHARPEMU_SHADER_LEARN, off unless on.</summary>
    public static bool Learn { get; } = ShaderCacheSwitch(Environment.GetEnvironmentVariable(LearnVariable), fallback: false);

    private static MetalShaderPrecompile Precompile => MetalShaderPrecompile.Current;

    /// <summary>The counters of this session's precompile, for the log and tests.</summary>
    public static (int Jobs, int Done, int Stored, long StoredBytes, int Present, int Underivable, int Failed) Counters =>
        (Volatile.Read(ref _plan).Count + Volatile.Read(ref _createdJobs), Volatile.Read(ref _jobsDone), Volatile.Read(ref _stored),
            Interlocked.Read(ref _storedBytes), Volatile.Read(ref _present), Volatile.Read(ref _underivable), Volatile.Read(ref _failed));

    internal static bool ShaderCacheSwitch(string? value, bool fallback = true) =>
        value?.Trim().ToLowerInvariant() switch
        {
            "1" or "on" or "true" or "yes" or "enabled" => true,
            "0" or "off" or "false" or "no" or "disabled" => false,
            _ => fallback,
        };

    /// <summary>
    /// Starts the precompile once per session, off the caller's thread; the shader prewarm
    /// calls it when it opens the title's lists. Nothing happens without a binary archive.
    /// </summary>
    public static void Start(nint device, string metalDirectory, string stamp, IGuestGpuBackend compiler)
    {
        if (!Enabled)
        {
            StaticShaderPlanner.DisableCapture();
            return;
        }

        lock (_gate)
        {
            if (_started)
            {
                return;
            }

            _started = true;
        }

        // Utility unless the guest is held until it completes (MetalCompileThread).
        MetalCompileThread.Start(
            "SharpEmu static shader precompile",
            Precompile.HoldsUntilComplete ? SharpEmu.HLE.Host.HostThreadQos.UserInitiated : SharpEmu.HLE.Host.HostThreadQos.Utility,
            () =>
            {
                try
                {
                    Run(device, metalDirectory, stamp, compiler);
                }
                catch (Exception exception)
                {
                    Console.Error.WriteLine($"[LOADER][WARN] Shader precompile (static) stopped: {exception.Message}");
                    _finished.Set();
                }
            });
    }

    /// <summary>
    /// The precompile modes' last phase: waits for every unit, with the progress on the
    /// splash and in the log. Returns at once when the precompile is not running.
    /// </summary>
    public static void WaitForCompletion()
    {
        lock (_gate)
        {
            if (!_started)
            {
                return;
            }
        }

        // The plan is ready once the scan is done.
        while (!_finished.IsSet && !_planned && !_stopping)
        {
            _finished.Wait(100);
        }

        var remaining = Math.Max(0, Volatile.Read(ref _plan).Count - Math.Max(0, Volatile.Read(ref _next)));
        if (!_finished.IsSet)
        {
            MetalShaderPrewarm.Gate.SetTotal(remaining);
            MetalShaderPrewarm.Gate.SetRunning(true);
            Precompile.BeginPhase("static", remaining);
            _phase = true;
            _finished.Wait();
            _phase = false;
        }
    }

    /// <summary>A summary for the precompile's report.</summary>
    public static string Describe()
    {
        var (jobs, done, stored, bytes, present, underivable, failed) = Counters;
        return string.Create(CultureInfo.InvariantCulture,
            $"static_jobs={jobs} static_done={done} static_stored={stored} static_mb={bytes / 1048576.0:F1} " +
            $"static_present={present} static_underivable={underivable} static_failed={failed}");
    }

    /// <summary>Stops the workers, writes the seed in learn mode and closes the inventory.</summary>
    public static void Stop()
    {
        _stopping = true;
        _scanCancel.Cancel();
        foreach (var thread in Volatile.Read(ref _threads))
        {
            if (!thread.IsCurrentThread)
            {
                thread.Join(TimeSpan.FromSeconds(10));
            }
        }

        Volatile.Write(ref _threads, []);
        if (Interlocked.Exchange(ref _planner, null) is { } planner)
        {
            if (Learn && _seedPath.Length != 0)
            {
                var saved = planner.TrySaveSeed(_seedPath, _titleId, string.Empty);
                Console.Error.WriteLine($"[LOADER][INFO] Shader seed {(saved ? "written" : "not written")}: {_seedPath}");
            }

            planner.Dispose();
        }

        lock (_gate)
        {
            _doneFile?.Dispose();
            _doneFile = null;
        }

        foreach (var stub in _stubs.Values)
        {
            MetalNative.SendVoid(stub, SelRelease);
        }

        _stubs.Clear();
        _finished.Set();
    }

    private static void Run(nint device, string metalDirectory, string stamp, IGuestGpuBackend compiler)
    {
        var hold = Precompile.HoldsUntilComplete;
        if (MetalBinaryArchive.For(device) is not { } archive)
        {
            Console.Error.WriteLine("[LOADER][INFO] Shader precompile (static): off, the binary archive is off.");
            StaticShaderPlanner.DisableCapture();
            _finished.Set();
            return;
        }

        _startedTicks = Stopwatch.GetTimestamp();
        _titleId = MetalShaderPrewarm.SanitizeTitleId(VideoOutBackendServices.GetApplicationTitleId());
        var titleDirectory = Path.GetDirectoryName(metalDirectory)!;
        _seedPath = StaticShaderPlanner.SeedPath(SharpEmuPaths.UserPath(), _titleId);
        var records = Recorded();
        var planner = StaticShaderPlanner.Open(titleDirectory, records, _seedPath);
        if (planner is null)
        {
            StaticShaderPlanner.DisableCapture();
            _finished.Set();
            return;
        }

        Volatile.Write(ref _planner, planner);
        OpenDone(metalDirectory, stamp);
        planner.AttachCapture(static job =>
        {
            if (!_done.ContainsKey(job.Id) && (IncludeComputes || job.Kind != StaticShaderKind.Compute))
            {
                _created.Enqueue(job);
                Interlocked.Increment(ref _createdJobs);
            }
        });

        _startBytes = archive.DiskBytes();
        _budget = hold ? int.MaxValue : ReadInt(BudgetVariable, DefaultBudget, 0, int.MaxValue);
        var jobs = hold ? MetalShaderPrewarm.WorkerCount : ReadInt(JobsVariable, DefaultJobs, 1, 16);
        // The workers take the created programs while the scan runs; the plan follows it.
        var remaining = jobs;
        var threads = new MetalCompileThread[jobs];
        for (var index = 0; index < jobs; index++)
        {
            threads[index] = MetalCompileThread.Create(
                $"SharpEmu static shader precompile {index}",
                hold ? SharpEmu.HLE.Host.HostThreadQos.UserInitiated : SharpEmu.HLE.Host.HostThreadQos.Utility,
                () =>
                {
                    try
                    {
                        Work(planner, archive, device, compiler, hold);
                    }
                    finally
                    {
                        if (Interlocked.Decrement(ref remaining) == 0)
                        {
                            Finish(archive);
                        }
                    }
                });
        }

        Volatile.Write(ref _threads, threads);
        foreach (var thread in threads)
        {
            thread.Start();
        }

        // Behind the game's own start: its first frame's programs and the files it reads then
        // come first.
        while (!hold && !_stopping && MetalShaderPrewarm.Gate.IsHolding)
        {
            Thread.Sleep(100);
        }

        var scan = (Environment.GetEnvironmentVariable(ScanVariable) ?? (hold ? "all" : "shaders")).Trim().ToLowerInvariant();
        var root = Environment.GetEnvironmentVariable("SHARPEMU_APP0_DIR");
        if (!_stopping && scan != "off" && !string.IsNullOrWhiteSpace(root) && Directory.Exists(root))
        {
            Scan(planner, root, scan, hold);
        }

        var plan = planner.Plan(id => _done.ContainsKey(id));
        if (!IncludeComputes)
        {
            plan.RemoveAll(static job => job.Kind == StaticShaderKind.Compute);
        }

        Volatile.Write(ref _plan, plan);
        _planned = true;
        Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"[LOADER][INFO] Shader precompile (static): programs={planner.Programs} pairs={planner.Pairs} " +
            $"signatures={planner.Signatures} states={planner.States} jobs={plan.Count} done_before={_done.Count} " +
            $"created_jobs={Volatile.Read(ref _createdJobs)} budget={(hold ? "all" : _budget.ToString(CultureInfo.InvariantCulture))} " +
            $"threads={jobs} archive_mb={_startBytes / 1048576.0:F0} " +
            $"interpolant_variant={(planner.SecondInterpolantVariant ? 2 : 1)} votes={planner.FirstVariantVotes}/{planner.SecondVariantVotes}"));
    }

    private static void Scan(StaticShaderPlanner planner, string root, string scan, bool hold)
    {
        var started = Stopwatch.GetTimestamp();
        var lastReport = started;
        try
        {
            var result = planner.Scan(root, scan == "all", hold ? 2 : 1, (done, total) =>
            {
                if (Stopwatch.GetElapsedTime(Interlocked.Read(ref lastReport)) >= TimeSpan.FromSeconds(5))
                {
                    Interlocked.Exchange(ref lastReport, Stopwatch.GetTimestamp());
                    Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"[LOADER][INFO] Shader scan: {done / 1048576.0:F0}/{total / 1048576.0:F0} MB"));
                }
            }, _scanCancel.Token);
            Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"[LOADER][INFO] Shader scan done: mode={scan} files={result.FilesScanned} unchanged={result.FilesSkipped} " +
                $"new_programs={result.Programs} mb={result.BytesScanned / 1048576.0:F0} s={Stopwatch.GetElapsedTime(started).TotalSeconds:F1}"));
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException or OperationCanceledException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Shader scan stopped: {exception.Message}");
        }
    }

    private static IReadOnlyList<ShaderCorpusRecord> Recorded()
    {
        if (MetalShaderPrewarm.Graphics is not { } graphics)
        {
            return [];
        }

        var records = new List<ShaderCorpusRecord>(graphics.LoadedCount);
        foreach (var item in graphics.Loaded())
        {
            try
            {
                records.Add(GraphicsPrewarmList.Decode(item));
            }
            catch (Exception exception) when (exception is InvalidDataException or System.Text.Json.JsonException or NotSupportedException)
            {
                // A record this build cannot read only takes a state away.
            }
        }

        return records;
    }

    private static void Work(StaticShaderPlanner planner, MetalBinaryArchive archive, nint device, IGuestGpuBackend compiler, bool hold)
    {
        // The first frame's programs go first: the guest waits for those.
        while (!hold && !_stopping && MetalShaderPrewarm.Gate.IsHolding)
        {
            Thread.Sleep(100);
        }

        var lastReport = Stopwatch.GetTimestamp();
        while (!_stopping)
        {
            if (!hold && (Volatile.Read(ref _stored) >= _budget ||
                          _startBytes + Interlocked.Read(ref _storedBytes) >= archive.Budget * ArchiveShare))
            {
                break;
            }

            if (!_created.TryDequeue(out var job))
            {
                var plan = Volatile.Read(ref _plan);
                var index = Volatile.Read(ref _next) + 1;
                if (index >= plan.Count)
                {
                    // Created programs keep arriving while the game runs; the plan does not.
                    if (hold && _planned)
                    {
                        break;
                    }

                    Thread.Sleep(250);
                    continue;
                }

                if (Interlocked.CompareExchange(ref _next, index, index - 1) != index - 1)
                {
                    continue;
                }

                job = plan[index];
            }

            if (_done.ContainsKey(job.Id))
            {
                continue;
            }

            try
            {
                Process(planner, archive, device, compiler, job);
            }
            catch (Exception exception) when (!_stopping)
            {
                // A unit that throws is a guess that stores nothing; the session goes on.
                NoteFailure($"{job.Kind} 0x{job.Program:X16}: {exception.GetType().Name}: {exception.Message}");
            }
            catch (Exception)
            {
                // Stopping: the inventory may be closed under the unit.
                break;
            }

            MarkDone(job.Id);
            Interlocked.Increment(ref _jobsDone);
            if (_phase)
            {
                MetalShaderPrewarm.Gate.Advance();
                Precompile.Advance();
            }
            else if (Stopwatch.GetElapsedTime(lastReport) >= TimeSpan.FromSeconds(10))
            {
                lastReport = Stopwatch.GetTimestamp();
                Console.Error.WriteLine($"[LOADER][INFO] Shader precompile (static): {Describe()}");
            }
        }
    }

    private static void Process(StaticShaderPlanner planner, MetalBinaryArchive archive, nint device, IGuestGpuBackend compiler, in StaticShaderJob job)
    {
        if (!planner.TryBuild(job, out var build, out _))
        {
            Interlocked.Increment(ref _underivable);
            return;
        }

        foreach (var shader in Translate(build, compiler))
        {
            Interlocked.Increment(ref _translated);
            var key = MetalBinaryArchive.LibraryKey(shader.Source, shader.UsesPerVertexInputs);
            if (archive.HasLibrary(key) || !_inFlight.TryAdd(key, 0))
            {
                Interlocked.Increment(ref _present);
                continue;
            }

            try
            {
                // Waits while the game compiles, and holds a prewarm slot meanwhile.
                using var slot = MetalCompileScheduler.Shared.EnterPrewarm(static () => _stopping);
                if (!slot.Entered)
                {
                    return;
                }

                var bytes = Store(archive, device, shader, key, out var error);
                if (bytes > 0)
                {
                    Interlocked.Increment(ref _stored);
                    Interlocked.Add(ref _storedBytes, bytes);
                }
                else if (error is not null)
                {
                    NoteFailure(error);
                }
            }
            finally
            {
                _inFlight.TryRemove(key, out _);
            }
        }
    }

    /// <summary>The MSL a build translates to, through the request builders of recorded programs.</summary>
    internal static List<Gen5MslShader> Translate(StaticShaderBuild build, IGuestGpuBackend compiler)
    {
        var shaders = new List<Gen5MslShader>(2);
        try
        {
            if (build.Compute is { } compute)
            {
                if (ShaderPrewarmCompiler.TryCompilePrewarm(compute, build.ComputeCode!, compiler,
                        MetalRenderHost.PrewarmSharedInt64Atomics, MetalRenderHost.PrewarmExecGuardElision,
                        MetalRenderHost.PrewarmSignedZeroInfNanPreserve, out var compiled, out _, out _) &&
                    compiled is MetalCompiledGuestShader metal)
                {
                    shaders.Add(metal.Shader);
                }

                return shaders;
            }

            foreach (var record in (ReadOnlySpan<ShaderCorpusRecord?>)[build.Pixel, build.Vertex])
            {
                if (record is not null && ShaderCorpus.TryBuildRequest(record, out var request, out _) && request is not null &&
                    compiler.TryCompileProgram(request, out var compiled, out _) && compiled is MetalCompiledGuestShader metal)
                {
                    shaders.Add(metal.Shader);
                }
            }
        }
        catch (Exception exception) when (exception is InvalidOperationException or NotSupportedException or ArgumentException or
                                              IndexOutOfRangeException or KeyNotFoundException)
        {
            // An untranslatable guess is only a unit that stores nothing.
        }

        return shaders;
    }

    /// <summary>
    /// Compiles <paramref name="shader"/> and stores its library file under <paramref name="key"/>
    /// from a pipeline descriptor that is never made into a state: a compute pipeline, a
    /// vertex-only render pipeline, a fragment behind a stub vertex function with the
    /// fragment's inputs as its outputs, or a mesh pipeline. Everything is released before
    /// it returns. The file's bytes, or -1 (with the reason, or none when skipped).
    /// </summary>
    internal static long Store(MetalBinaryArchive archive, nint device, Gen5MslShader shader, UInt128 key, out string? error)
    {
        error = null;
        var source = shader.Source;
        var entry = shader.EntryPoint;
        var kind = entry switch
        {
            "gen5_cs" => MetalBinaryArchive.PipelineKind.Compute,
            "gen5_ms" => MetalBinaryArchive.PipelineKind.Mesh,
            _ => MetalBinaryArchive.PipelineKind.Render,
        };
        if (entry == "gen5_vs" && source.Contains("[[stage_in]]", StringComparison.Ordinal))
        {
            // Fetches through a vertex descriptor this precompile does not know.
            return -1;
        }

        string? stubSource = null;
        if (entry == "gen5_ps" && (stubSource = StubVertexSource(source)) is null)
        {
            error = "the fragment's inputs could not be read";
            return -1;
        }

        nint stub = 0;
        var ownsStub = false;
        if (stubSource is not null && (stub = StubVertex(device, stubSource, shader.UsesPerVertexInputs, out ownsStub, out error)) == 0)
        {
            return -1;
        }

        var pool = MetalNative.objc_autoreleasePoolPush();
        nint library = 0;
        nint function = 0;
        nint descriptor = 0;
        try
        {
            library = MetalRenderHost.TryCompileLibrary(device, source, shader.UsesPerVertexInputs, out var compileError, out _, askArchive: false);
            if (library == 0)
            {
                error = compileError;
                return -1;
            }

            nint functionError = 0;
            function = shader.UsesMipStatisticsConstant
                ? MetalRenderHost.NewMipStatisticsFunction(library, entry, enabled: false, out functionError)
                : MetalNative.Send(library, SelNewFunctionWithName, MetalNative.NsString(entry));
            if (function == 0)
            {
                error = $"the library has no entry point {entry}" +
                    (functionError != 0 ? $": {MetalNative.DescribeError(functionError)}" : string.Empty);
                return -1;
            }

            descriptor = Descriptor(kind, entry, function, stub, source);
            return descriptor == 0 ? -1 : archive.StoreLibrary(key, kind, descriptor);
        }
        finally
        {
            foreach (var handle in (ReadOnlySpan<nint>)[descriptor, function, library, ownsStub ? stub : 0])
            {
                if (handle != 0)
                {
                    MetalNative.SendVoid(handle, SelRelease);
                }
            }

            MetalNative.objc_autoreleasePoolPop(pool);
        }
    }

    private static nint Descriptor(MetalBinaryArchive.PipelineKind kind, string entry, nint function, nint stub, string source)
    {
        var className = kind switch
        {
            MetalBinaryArchive.PipelineKind.Compute => "MTLComputePipelineDescriptor",
            MetalBinaryArchive.PipelineKind.Mesh => "MTLMeshRenderPipelineDescriptor",
            _ => "MTLRenderPipelineDescriptor",
        };
        var descriptor = MetalNative.Send(MetalNative.Send(MetalNative.Class(className), SelAlloc), SelInit);
        if (descriptor == 0)
        {
            return 0;
        }

        switch (kind)
        {
            case MetalBinaryArchive.PipelineKind.Compute:
                MetalNative.SendVoid(descriptor, MetalNative.Selector("setComputeFunction:"), function);
                break;
            case MetalBinaryArchive.PipelineKind.Mesh:
                MetalNative.SendVoid(descriptor, MetalNative.Selector("setMeshFunction:"), function);
                break;
            default:
                if (entry == "gen5_ps")
                {
                    MetalNative.SendVoid(descriptor, MetalNative.Selector("setVertexFunction:"), stub);
                    MetalNative.SendVoid(descriptor, MetalNative.Selector("setFragmentFunction:"), function);
                    // The color attachments stay invalid (the writes are dropped); a depth or
                    // stencil export needs its attachment.
                    var stencil = source.Contains("[[stencil]]", StringComparison.Ordinal);
                    if (stencil || source.Contains("[[depth(", StringComparison.Ordinal))
                    {
                        var format = stencil ? PixelFormatDepth32FloatStencil8 : PixelFormatDepth32Float;
                        MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setDepthAttachmentPixelFormat:"), format);
                        if (stencil)
                        {
                            MetalNative.SendVoidNUInt(descriptor, MetalNative.Selector("setStencilAttachmentPixelFormat:"), format);
                        }
                    }
                }
                else
                {
                    MetalNative.SendVoid(descriptor, MetalNative.Selector("setVertexFunction:"), function);
                }

                break;
        }

        return descriptor;
    }

    /// <summary>
    /// A vertex function whose outputs are the fragment's interpolated inputs
    /// (struct Gen5PsIn's user(...) members, as float4, and the position); the fragment-only
    /// inputs (barycentrics, front facing, sample mask) are left out. Null when the source
    /// has no such struct.
    /// </summary>
    internal static string? StubVertexSource(string fragmentSource)
    {
        var start = fragmentSource.IndexOf("struct Gen5PsIn", StringComparison.Ordinal);
        var open = start < 0 ? -1 : fragmentSource.IndexOf('{', start);
        var close = open < 0 ? -1 : fragmentSource.IndexOf("};", open, StringComparison.Ordinal);
        if (close < 0)
        {
            return null;
        }

        var members = new StringBuilder();
        var position = false;
        foreach (var line in fragmentSource.AsSpan(open + 1, close - open - 1).ToString().Split('\n'))
        {
            var attributes = line.IndexOf("[[", StringComparison.Ordinal);
            var end = line.IndexOf("]]", StringComparison.Ordinal);
            if (attributes < 0 || end < attributes)
            {
                continue;
            }

            var declaration = line[..attributes].Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var first = line[(attributes + 2)..end].Split(',')[0].Trim();
            if (declaration.Length < 2)
            {
                continue;
            }

            if (first.StartsWith("user(", StringComparison.Ordinal))
            {
                members.Append("    float4 ").Append(declaration[^1]).Append(" [[").Append(first).Append("]];\n");
            }
            else if (first == "position")
            {
                members.Append("    float4 ").Append(declaration[^1]).Append(" [[position]];\n");
                position = true;
            }
        }

        if (!position)
        {
            members.Append("    float4 sharpemu_position [[position]];\n");
        }

        return "#include <metal_stdlib>\nusing namespace metal;\nstruct SharpEmuStaticVsOut\n{\n" + members +
            "};\nvertex SharpEmuStaticVsOut " + StubEntryPoint + "()\n{\n    return SharpEmuStaticVsOut{};\n}\n";
    }

    // The stub vertex function of a fragment input layout, compiled once and kept (a few
    // dozen layouts per title).
    private static nint StubVertex(nint device, string source, bool usesPerVertexInputs, out bool owned, out string? error)
    {
        error = null;
        owned = false;
        var cacheKey = (usesPerVertexInputs ? "1" : "0") + source;
        if (_stubs.TryGetValue(cacheKey, out var cached))
        {
            return cached;
        }

        var library = MetalRenderHost.TryCompileLibrary(device, source, usesPerVertexInputs, out var compileError, out _, askArchive: false);
        if (library == 0)
        {
            error = $"the stub vertex function did not compile: {compileError}";
            return 0;
        }

        var function = MetalNative.Send(library, SelNewFunctionWithName, MetalNative.NsString(StubEntryPoint));
        MetalNative.SendVoid(library, SelRelease);
        if (function == 0)
        {
            error = "the stub vertex library has no function";
            return 0;
        }

        // Past the cap (or raced by another thread) the caller releases it after its one use.
        owned = _stubs.Count >= MaxStubVertices || !_stubs.TryAdd(cacheKey, function);
        return function;
    }

    private static void Finish(MetalBinaryArchive archive)
    {
        Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"[LOADER][INFO] Shader precompile (static) done: {Describe()} translated={Volatile.Read(ref _translated)} " +
            $"s={Stopwatch.GetElapsedTime(_startedTicks).TotalSeconds:F1} archive_mb={archive.DiskBytes() / 1048576.0:F0}"));
        _finished.Set();
    }

    private static void NoteFailure(string error)
    {
        if (Interlocked.Increment(ref _failed) <= MaxLoggedFailures)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Shader precompile (static) failed: {error}");
        }
    }

    private static void OpenDone(string metalDirectory, string stamp)
    {
        var name = "static-precompile-" +
            XxHash3.HashToUInt64(MemoryMarshal.AsBytes(stamp.AsSpan())).ToString("x16", CultureInfo.InvariantCulture) + ".done";
        try
        {
            Directory.CreateDirectory(metalDirectory);
            foreach (var other in Directory.EnumerateFiles(metalDirectory, "static-precompile-*.done"))
            {
                if (!string.Equals(Path.GetFileName(other), name, StringComparison.Ordinal))
                {
                    File.Delete(other);
                }
            }

            var file = new FileStream(Path.Combine(metalDirectory, name), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read);
            var bytes = new byte[file.Length - file.Length % sizeof(ulong)];
            file.ReadExactly(bytes);
            foreach (var id in MemoryMarshal.Cast<byte, ulong>(bytes))
            {
                _done.TryAdd(id, 0);
            }

            file.SetLength(bytes.Length);
            file.Position = bytes.Length;
            lock (_gate)
            {
                _doneFile = file;
            }
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Shader precompile progress unavailable: {exception.Message}");
        }
    }

    private static void MarkDone(ulong id)
    {
        if (!_done.TryAdd(id, 0))
        {
            return;
        }

        lock (_gate)
        {
            if (_doneFile is not { } file)
            {
                return;
            }

            try
            {
                Span<byte> bytes = stackalloc byte[sizeof(ulong)];
                MemoryMarshal.Write(bytes, id);
                file.Write(bytes);
                file.Flush();
            }
            catch (IOException)
            {
                // Only the next start's skipping depends on it.
            }
        }
    }

    private static int ReadInt(string variable, int fallback, int minimum, int maximum) =>
        int.TryParse(Environment.GetEnvironmentVariable(variable), NumberStyles.None, CultureInfo.InvariantCulture, out var value)
            ? Math.Clamp(value, minimum, maximum)
            : fallback;
}
