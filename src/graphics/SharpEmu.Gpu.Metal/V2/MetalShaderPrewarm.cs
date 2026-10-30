// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO.Hashing;
using System.Runtime.InteropServices;
using SharpEmu.HLE;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.VideoOut;
using SharpEmu.Logging;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Metal;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Holds the guest at its first splash hide, GPU submission or flip until the shader
/// prewarm has been decided and has compiled what the guest needs first (every recorded
/// compute, or those of its first frame once a session has recorded them; see
/// MetalShaderPrewarm.HoldFirst), so the first frames do not wait on cold compiles. The
/// window keeps presenting the splash meanwhile. SHARPEMU_SHADER_PREWARM_WAIT=0 opts out.
/// The v2 twin of the Vulkan presenter's WaitForShaderPrewarm.
/// </summary>
internal sealed class MetalShaderPrewarmGate(bool disabled)
{
    private readonly object _sync = new();
    private bool _started;
    private bool _decided;
    private bool _running;
    private bool _cancelled;
    private volatile bool _released;
    private int _holdReported;
    private int _progress;
    private int _total;

    public int Progress => Volatile.Read(ref _progress);

    public int Total => Volatile.Read(ref _total);

    /// <summary>Whether the prewarm is compiling and holding the guest right now.</summary>
    public bool IsRunning
    {
        get
        {
            lock (_sync)
            {
                return _running && !_released;
            }
        }
    }

    /// <summary>The presenter started: a holder now waits for the prewarm to be decided.</summary>
    public void Begin()
    {
        lock (_sync)
        {
            _started = true;
        }
    }

    /// <summary>The prewarm is decided; running holds the guest until it is set false.</summary>
    public void SetRunning(bool running)
    {
        lock (_sync)
        {
            _decided = true;
            _running = running;
            _released = !running;
            Monitor.PulseAll(_sync);
        }
    }

    /// <summary>The session is closing: nothing is held any longer.</summary>
    public void Cancel()
    {
        lock (_sync)
        {
            _cancelled = true;
            _released = true;
            Monitor.PulseAll(_sync);
        }
    }

    public void SetTotal(int total)
    {
        Volatile.Write(ref _total, total);
        Volatile.Write(ref _progress, 0);
    }

    public int Advance() => Interlocked.Increment(ref _progress);

    /// <summary>Whether a holder would wait right now.</summary>
    public bool IsHolding
    {
        get
        {
            lock (_sync)
            {
                return HoldingLocked(static () => false);
            }
        }
    }

    public void Wait(string holder) => Wait(holder, static () => HostSessionControl.IsShutdownRequested);

    public void Wait(string holder, Func<bool> shutdownRequested)
    {
        if (disabled || _released)
        {
            return;
        }

        var started = Stopwatch.GetTimestamp();
        var reported = started;
        var reporter = false;
        lock (_sync)
        {
            while (HoldingLocked(shutdownRequested))
            {
                if (!reporter && _running && Interlocked.Exchange(ref _holdReported, 1) == 0)
                {
                    reporter = true;
                    Console.Error.WriteLine(
                        $"[LOADER][INFO] Shader prewarm: holding the game at its {holder} until it finishes.");
                }

                Monitor.Wait(_sync, 1000);
                if (reporter && Stopwatch.GetElapsedTime(reported) >= TimeSpan.FromSeconds(5))
                {
                    reported = Stopwatch.GetTimestamp();
                    var total = Volatile.Read(ref _total);
                    var done = Volatile.Read(ref _progress);
                    Console.Error.WriteLine(
                        $"[LOADER][INFO] Shader prewarm: {done}/{total} ({(total == 0 ? 0 : done * 100 / total)}%)");
                }
            }
        }

        if (reporter)
        {
            Console.Error.WriteLine(string.Create(
                System.Globalization.CultureInfo.InvariantCulture,
                $"[LOADER][INFO] Shader prewarm: game released after {Stopwatch.GetElapsedTime(started).TotalSeconds:F1}s."));
        }
    }

    private bool HoldingLocked(Func<bool> shutdownRequested) =>
        _started && (!_decided || _running) && !_cancelled && !shutdownRequested();
}

/// <summary>
/// Compiles the compute pipelines a previous run recorded (shader-prewarm-*.bin) on
/// background threads at startup: the MSL library, its function and the compute
/// pipeline state, which fills the driver's shader cache, so the first dispatch of
/// each shader no longer waits for the driver. Uses the shared list and record
/// infrastructure and the shared resource plan; the stamp names the translators, the
/// compiler type and the device and OS, and an interrupted prewarm resumes with the
/// pipelines it had not finished. SHARPEMU_SHADER_PREWARM=0 turns it off. The v2 twin
/// of the Vulkan presenter's ShaderPrewarm partial.
/// </summary>
internal static class MetalShaderPrewarm
{
    private const int MaxLoggedPrewarmFailures = 8;
    private const int PrewarmProgressBatch = 256;

    private static readonly nint SelNewFunctionWithName = MetalNative.Selector("newFunctionWithName:");
    private static readonly nint SelNewComputePipelineState =
        MetalNative.Selector("newComputePipelineStateWithFunction:error:");
    private static readonly nint SelRelease = MetalNative.Selector("release");
    private static readonly nint SelName = MetalNative.Selector("name");

    public static readonly MetalShaderPrewarmGate Gate =
        new(Environment.GetEnvironmentVariable("SHARPEMU_SHADER_PREWARM_WAIT") == "0");

    private static readonly object _startGate = new();
    private static bool _startRequested;
    private static ShaderPrewarmList? _list;
    private static GraphicsPrewarmList? _graphics;
    private static ConcurrentDictionary<ulong, byte>? _graphicsKeys;
    private static int _graphicsCompiled;
    private static int _graphicsFailed;
    private static bool _graphicsRan;
    private static ConcurrentDictionary<string, byte>? _prewarmedShaderIdentities;
    private static ConcurrentDictionary<string, byte>? _runtimeComputeIdentities;
    private static MetalCompileThread[] _threads = [];
    private static volatile bool _stopping;
    private static volatile bool _finished;
    private static int _compiled;
    private static int _failed;
    private static int _runtimeHits;
    private static int _deduplicated;
    private static int _archived;

    // SHARPEMU_SHADER_PREWARM_SKIP_ARCHIVED=0 compiles the computes the binary archive holds too.
    private static readonly bool SkipsArchived =
        Environment.GetEnvironmentVariable("SHARPEMU_SHADER_PREWARM_SKIP_ARCHIVED") != "0";
    private static readonly ConcurrentQueue<ulong> _completed = new();
    private static readonly ConcurrentDictionary<string, ManualResetEventSlim> _inFlight = new(StringComparer.Ordinal);

    // Set on a prewarm thread once it compiles at Utility (the computes the guest is not held
    // for): a runtime compile does not wait for one of its compiles, which a busy machine may
    // leave behind every thread above it (WaitWhileCompiling).
    [ThreadStatic]
    private static bool _onUtilityThread;

    private static readonly ConcurrentDictionary<ManualResetEventSlim, byte> _utilityCompiles = new();

    /// <summary>
    /// Compiles in flight at once. The MSL front end runs in the driver's compiler service
    /// and is CPU bound there. On a ten-core machine (four performance, six efficiency
    /// cores), 200 cold Demon's Souls computes took 14.5-15.4 s with six requests in flight,
    /// 16.8 s with five, 18.6-26.5 s with four and 44 s with two, and eight or more were
    /// slower again (ShaderCompileBench, machine otherwise idle).
    /// </summary>
    internal static int WorkerCount { get; } = Math.Clamp(Environment.ProcessorCount * 3 / 5, 1, 6);

    /// <summary>The list the shader cache records into; null while prewarm is off.</summary>
    public static ShaderPrewarmList? List => Volatile.Read(ref _list);

    /// <summary>The list the shader cache records its vertex and pixel programs into; null while prewarm is off.</summary>
    public static GraphicsPrewarmList? Graphics => Volatile.Read(ref _graphics);

    /// <summary>
    /// Decides the prewarm once per session, off the caller's thread: loading the list
    /// reads a file that can run to megabytes. The gate holds the guest until then.
    /// </summary>
    public static void Start(nint device)
    {
        lock (_startGate)
        {
            if (_startRequested)
            {
                return;
            }

            _startRequested = true;
        }

        TranslateAsTheGameDoes();
        Gate.Begin();
        MetalCompileThread.Start("SharpEmu Metal v2 shader prewarm", SharpEmu.HLE.Host.HostThreadQos.UserInitiated, () =>
        {
            try
            {
                if (!StartCore(device) && !CompletePrecompile())
                {
                    Gate.SetRunning(false);
                }
            }
            catch (Exception exception)
            {
                Console.Error.WriteLine($"[LOADER][WARN] Shader prewarm could not start: {exception.Message}");
                if (!CompletePrecompile())
                {
                    Gate.SetRunning(false);
                }
            }
        });
    }

    /// <summary>
    /// Sets the process-wide translation switches the command stream sets, before the
    /// prewarm translates anything: what the prewarm makes must be what the game's draws
    /// ask for (the pixel programs' mip-statistics recording, portable buffers).
    /// </summary>
    internal static void TranslateAsTheGameDoes() => MetalRenderHost.ApplyTranslationSwitches();

    private static bool StartCore(nint device)
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_SHADER_PREWARM") == "0")
        {
            return false;
        }

        var directory = SharpEmuPaths.UserPath(
            "pipeline_cache", SanitizeTitleId(VideoOutBackendServices.GetApplicationTitleId()), "metal-v2");
        var compiler = GuestGpu.Current;
        var stamp = Stamp(compiler, device);
        // Seeded before the compute prewarm or the game plans a shader.
        MetalDeviceDataLoadsStore.Open(directory, stamp);
        // Loaded before the gate lets the game go, so every module the game creates
        // finds the graphics libraries queued; they compile once the computes the guest is
        // held for are done.
        MetalLibraryPrewarm.Open(directory, device, stamp);
        var graphics = OpenGraphics(directory, stamp);
        // Programs the game has not drawn yet, behind everything above (MetalStaticPrecompile).
        MetalStaticPrecompile.Start(device, directory, stamp, compiler);
        if (ShaderPrewarmList.Open(directory) is not { } list)
        {
            BeginLibraries();
            return false;
        }

        _prewarmedShaderIdentities = new ConcurrentDictionary<string, byte>(StringComparer.Ordinal);
        _runtimeComputeIdentities = new ConcurrentDictionary<string, byte>(StringComparer.Ordinal);
        Volatile.Write(ref _list, list);
        if (list.LoadedComputeCount == 0 && graphics.Count == 0)
        {
            list.WriteStamp(stamp);
            _finished = true;
            BeginLibraries();
            return false;
        }

        var pending = PendingComputes(list, stamp, out var resumed);
        var started = Stopwatch.GetTimestamp();
        if (pending.Count == 0 && graphics.Count == 0)
        {
            Finish(list, stamp, started);
            return false;
        }

        var ordered = HoldFirst(pending, HoldsForEveryCompute ? null : list.ReadFirstFrameComputes(), out var heldComputes);
        var orderedGraphics = HoldFirstGraphics(
            graphics, HoldsForEveryCompute ? null : _graphics?.ReadFirstFrame(), out var heldGraphics);
        var held = heldComputes + heldGraphics;
        var total = pending.Count + graphics.Count;
        // What the guest is held for first, then what the game needs soonest: the graphics
        // libraries of its scenes, then the computes it was not held for.
        var work = new ConcurrentQueue<PrewarmWork>(
            orderedGraphics.Where(item => item.Held).Select(item => new PrewarmWork(item.Item, null, null, true))
                .Concat(ordered.Where(item => item.Held).Select(item => new PrewarmWork(null, item.Record, item.Code, true)))
                .Concat(orderedGraphics.Where(item => !item.Held).Select(item => new PrewarmWork(item.Item, null, null, false)))
                .Concat(ordered.Where(item => !item.Held).Select(item => new PrewarmWork(null, item.Record, item.Code, false))));
        var workerCount = WorkerCount;
        var remaining = workerCount;
        var done = 0;
        Gate.SetTotal(held);
        if (Precompile.HoldsUntilComplete)
        {
            Precompile.BeginPhase("programs", total);
        }

        Console.Error.WriteLine(
            $"[LOADER][INFO] Shader prewarm started: compute={pending.Count} graphics={graphics.Count} held={held} " +
            $"resumed={resumed} workers={workerCount}");
        Gate.SetRunning(held > 0);
        if (held == 0)
        {
            MetalLibraryPrewarm.Begin();
        }

        var threads = new MetalCompileThread[workerCount];
        for (var index = 0; index < workerCount; index++)
        {
            // UserInitiated while the guest is held for these compiles: the driver's compiler
            // service runs a request at the requester's class, and from below-normal threads
            // each compile took about a fifth longer even on an idle machine
            // (ShaderCompileBench, 200 cold Demon's Souls computes) and fell far behind under
            // load from other processes. The computes the guest is not held for compile at
            // Utility (see the loop below).
            threads[index] = MetalCompileThread.Create($"SharpEmu shader prewarm {index}", SharpEmu.HLE.Host.HostThreadQos.UserInitiated, () =>
            {
                var lowered = false;
                while (!_stopping && work.TryDequeue(out var item))
                {
                    if (item.Graphics is { } graphicsItem)
                    {
                        // A library the game takes waits for it, so these keep normal priority
                        // (see MetalLibraryPrewarm).
                        PrewarmGraphicsLibrary(graphicsItem, compiler);
                    }
                    else
                    {
                        if (!item.Held)
                        {
                            // The guest runs again: what is left compiles behind its threads.
                            if (!lowered)
                            {
                                lowered = true;
                                Thread.CurrentThread.Priority = ThreadPriority.BelowNormal;
                                _onUtilityThread = MetalCompileThread.ApplyToCurrent(SharpEmu.HLE.Host.HostThreadQos.Utility);
                            }

                            YieldToTheGame();
                        }

                        PrewarmComputePipeline(item.Compute!, item.Code!, compiler, device);
                        _completed.Enqueue(ShaderPrewarmList.Identity(item.Compute!));
                    }

                    if (item.Held && Precompile.HoldsUntilComplete)
                    {
                        Precompile.Advance();
                    }

                    if (item.Held && Gate.Advance() == held && held < total)
                    {
                        Gate.SetRunning(false);
                        // The graphics libraries of past sessions go ahead of the computes
                        // the guest was not held for (see YieldToTheGame).
                        MetalLibraryPrewarm.Begin();
                        Console.Error.WriteLine(string.Create(
                            System.Globalization.CultureInfo.InvariantCulture,
                            $"[LOADER][INFO] Shader prewarm: the {held} programs of the first frame are ready after " +
                            $"{Stopwatch.GetElapsedTime(started).TotalSeconds:F1}s; the other {total - held} compile while the game runs."));
                    }

                    if (Interlocked.Increment(ref done) % PrewarmProgressBatch == 0)
                    {
                        FlushProgress(list, stamp);
                    }
                }

                FlushProgress(list, stamp);
                if (Interlocked.Decrement(ref remaining) == 0)
                {
                    Finish(list, stamp, started);
                }
            });
        }

        Volatile.Write(ref _threads, threads);
        foreach (var thread in threads)
        {
            thread.Start();
        }

        return true;
    }

    /// <summary>SHARPEMU_METAL_V2_GRAPHICS_PREWARM=0 neither records graphics programs nor translates them again.</summary>
    private static readonly bool GraphicsPrewarmDisabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_GRAPHICS_PREWARM") == "0";

    private readonly record struct PrewarmWork(
        GraphicsPrewarmList.Item? Graphics, ComputePrewarmRecord? Compute, ShaderCodeCapture? Code, bool Held);

    /// <summary>
    /// The recorded graphics programs this start translates again and compiles the libraries
    /// of: all of them when the translator changed since a prewarm last finished, or when the
    /// library list no longer holds what that prewarm made of them (a list moved or pruned);
    /// none while the library list is whole, which compiles its own sources. The list is
    /// opened either way, so the session records into it.
    /// </summary>
    private static IReadOnlyList<GraphicsPrewarmList.Item> OpenGraphics(string directory, string stamp)
    {
        if (GraphicsPrewarmDisabled || !MetalLibraryPrewarm.Active || GraphicsPrewarmList.Open(directory) is not { } graphics)
        {
            return [];
        }

        Volatile.Write(ref _graphics, graphics);
        if (graphics.LoadedCount == 0)
        {
            return [];
        }

        if (graphics.ReadComplete() is { } complete &&
            complete.Stamp == stamp &&
            MetalLibraryPrewarm.ListedSources >= Math.Min(complete.Libraries, MetalLibraryPrewarm.MaxEntries))
        {
            return [];
        }

        _graphicsKeys = new ConcurrentDictionary<ulong, byte>();
        _graphicsRan = true;
        return graphics.Loaded();
    }

    /// <summary>
    /// The graphics programs in the order the prewarm translates them, and how many from the
    /// front the guest is held for: those the last session created before its first frame,
    /// then the rest in the order they were first used. With no first-frame list yet, every
    /// one is held for.
    /// </summary>
    internal static List<(GraphicsPrewarmList.Item Item, bool Held)> HoldFirstGraphics(
        IReadOnlyList<GraphicsPrewarmList.Item> items,
        IReadOnlySet<ulong>? firstFrame,
        out int held)
    {
        if (firstFrame is null)
        {
            held = items.Count;
            return items.Select(item => (item, true)).ToList();
        }

        var ordered = new List<(GraphicsPrewarmList.Item, bool)>(items.Count);
        ordered.AddRange(items.Where(item => firstFrame.Contains(item.Identity)).Select(item => (item, true)));
        held = ordered.Count;
        ordered.AddRange(items.Where(item => !firstFrame.Contains(item.Identity)).Select(item => (item, false)));
        return ordered;
    }

    // Translates one recorded graphics program and compiles its library into the library list's
    // slots, where the module the game creates for it takes it from.
    private static void PrewarmGraphicsLibrary(in GraphicsPrewarmList.Item item, IGuestGpuBackend compiler)
    {
        if (!ShaderPrewarmCompiler.TryCompileGraphicsPrewarm(item, compiler, out var compiled, out var error))
        {
            NoteGraphicsFailure(error);
            return;
        }

        if (compiled is not MetalCompiledGuestShader metal)
        {
            NoteGraphicsFailure($"the compiler produced {compiled?.GetType().Name ?? "nothing"}, not MSL");
            return;
        }

        var shader = metal.Shader;
        _graphicsKeys!.TryAdd(MetalLibraryPrewarm.Key(shader.Source, shader.UsesPerVertexInputs), 0);
        if (MetalLibraryPrewarm.Offer(shader.Source, shader.UsesPerVertexInputs))
        {
            Interlocked.Increment(ref _graphicsCompiled);
        }
    }

    private static void NoteGraphicsFailure(string error)
    {
        if (Interlocked.Increment(ref _graphicsFailed) <= MaxLoggedPrewarmFailures)
        {
            Console.Error.WriteLine($"[LOADER][WARN] Graphics prewarm failed: error={error}");
        }
    }

    /// <summary>
    /// The recorded computes this start compiles: every one, but those an interrupted
    /// prewarm under the same stamp already finished. A current stamp does not skip the
    /// rest. The libraries land in macOS's per-app Metal shader cache, which the OS caps
    /// and evicts by itself while other titles and builds under the same bundle fill it,
    /// so a finished prewarm says nothing about what the cache still holds. Skipping on
    /// the stamp left Demon's Souls, after an eviction, compiling every compute library
    /// cold on the GPU worker at its first dispatch, drawing nothing for about 100 s.
    /// A library the cache still holds compiles in milliseconds, and a warm prewarm
    /// finishes before the game reaches its splash hold.
    /// </summary>
    internal static List<(ComputePrewarmRecord Record, ShaderCodeCapture Code)> PendingComputes(
        ShaderPrewarmList list,
        string stamp,
        out int resumed)
    {
        var completed = list.ReadProgress(stamp);
        resumed = completed.Count;
        return list.LoadedComputes()
            .Where(item => !completed.Contains(ShaderPrewarmList.Identity(item.Record)))
            .ToList();
    }

    /// <summary>
    /// SHARPEMU_SHADER_PREWARM_HOLD=all holds the guest for every recorded compute, as
    /// before there was a first-frame list.
    /// </summary>
    private static readonly bool HoldsForEveryCompute =
        Environment.GetEnvironmentVariable("SHARPEMU_SHADER_PREWARM_HOLD") == "all" ||
        MetalShaderPrecompile.Current.HoldsUntilComplete;

    private static MetalShaderPrecompile Precompile => MetalShaderPrecompile.Current;

    /// <summary>
    /// Starts the library list behind the game, or, in the hold-until-done and precompile-only
    /// modes, compiles it to completion while the guest is still held (MetalShaderPrecompile).
    /// </summary>
    private static void BeginLibraries()
    {
        if (!Precompile.HoldsUntilComplete)
        {
            MetalLibraryPrewarm.Begin();
            return;
        }

        var sources = MetalLibraryPrewarm.ListedSources;
        if (sources == 0)
        {
            return;
        }

        Gate.SetTotal(sources);
        Gate.SetRunning(true);
        Precompile.BeginPhase("libraries", sources);
        MetalLibraryPrewarm.Precompile(() =>
        {
            Gate.Advance();
            Precompile.Advance();
        });
    }

    /// <summary>
    /// The precompile modes' end, once everything is compiled: the summary and, for
    /// precompile-only, the session's close. True when the guest stays held because the
    /// session is closing.
    /// </summary>
    private static bool CompletePrecompile()
    {
        if (Precompile.HoldsUntilComplete && !Precompile.Completed)
        {
            // The programs only found in the game's files come last.
            MetalStaticPrecompile.WaitForCompletion();
        }

        return Precompile.Complete(
            $"computes={Volatile.Read(ref _compiled)} compute_failed={Volatile.Read(ref _failed)} " +
            $"graphics_libraries={Volatile.Read(ref _graphicsCompiled)} graphics_failed={Volatile.Read(ref _graphicsFailed)} " +
            $"libraries_held={MetalLibraryPrewarm.HeldLibraries} {MetalStaticPrecompile.Describe()}");
    }

    /// <summary>
    /// The pending computes in the order the prewarm compiles them, and how many from the
    /// front the guest is held for: those the last session created before its first frame
    /// (in list order, which is the order a session first used them), then the rest. With no
    /// first-frame list yet, every one is held for. Demon's Souls records some 1400 computes
    /// across its scenes and creates about a hundred before its first frame; a cold start
    /// held it for all of them, 126 s at the splash screen (run 20261017-075343-ds-base-boot),
    /// while the rest can compile behind the logos and the intro.
    /// </summary>
    internal static List<(ComputePrewarmRecord Record, ShaderCodeCapture Code, bool Held)> HoldFirst(
        IReadOnlyList<(ComputePrewarmRecord Record, ShaderCodeCapture Code)> pending,
        IReadOnlySet<ulong>? firstFrame,
        out int held)
    {
        if (firstFrame is null)
        {
            held = pending.Count;
            return pending.Select(item => (item.Record, item.Code, true)).ToList();
        }

        var ordered = new List<(ComputePrewarmRecord, ShaderCodeCapture, bool)>(pending.Count);
        ordered.AddRange(pending
            .Where(item => firstFrame.Contains(ShaderPrewarmList.Identity(item.Record)))
            .Select(item => (item.Record, item.Code, true)));
        held = ordered.Count;
        ordered.AddRange(pending
            .Where(item => !firstFrame.Contains(ShaderPrewarmList.Identity(item.Record)))
            .Select(item => (item.Record, item.Code, false)));
        return ordered;
    }

    private static long _yieldedTicks;
    private static int _runtimeFirst;

    /// <summary>
    /// The game starts compiling a shader module of its own: from here until it is built,
    /// every prewarm compile not yet started waits (MetalCompileScheduler).
    /// </summary>
    public static void GameCompileStarted() => MetalCompileScheduler.Shared.RuntimeStarted();

    /// <summary>The game's shader module compile is done.</summary>
    public static void GameCompileDone() => MetalCompileScheduler.Shared.RuntimeDone();

    /// <summary>
    /// Holds a compute the guest is not held for while the graphics libraries of past
    /// sessions compile, so those go first; the game's own compiles hold every prewarm
    /// compile back through MetalCompileScheduler. A cold start that let the guest go after
    /// its first frame's computes had its GPU worker wait 120 s in one stretch for the
    /// logos' graphics libraries while 1500 computes compiled in the background
    /// (run 20261017-095415-ds-sc-cold2).
    /// </summary>
    private static void YieldToTheGame()
    {
        if (!MetalLibraryPrewarm.Running)
        {
            return;
        }

        var started = Stopwatch.GetTimestamp();
        while (!_stopping && MetalLibraryPrewarm.Running)
        {
            Thread.Sleep(2);
        }

        Interlocked.Add(ref _yieldedTicks, Stopwatch.GetTimestamp() - started);
    }

    /// <summary>The guest presented its first frame: what it created until now is what the next start holds it for.</summary>
    public static void NoteFirstGuestFrame()
    {
        List?.WriteFirstFrameComputes();
        Graphics?.WriteFirstFrame();
    }

    // SHARPEMU_SLOW_COMPILE_DIR=<dir>: the MSL of each compute a compile or a wait on one took seconds over, for compiling
    // it again offline.
    private static readonly string? SlowCompileDirectory = Environment.GetEnvironmentVariable("SHARPEMU_SLOW_COMPILE_DIR");

    private static void DumpSlowSource(MetalCompiledGuestShader shader)
    {
        if (SlowCompileDirectory is null || shader.Shader.Source.Length == 0)
        {
            return;
        }

        try
        {
            Directory.CreateDirectory(SlowCompileDirectory);
            File.WriteAllText(Path.Combine(SlowCompileDirectory, $"{Identity(shader)}.metal"), shader.Shader.Source);
        }
        catch (IOException)
        {
        }
    }

    /// <summary>
    /// Waits while the prewarm compiles this module's source, so a dispatch that comes for
    /// a compute the background prewarm has in hand takes its cached library instead of
    /// starting the same front-end compile a second time.
    /// </summary>
    public static void WaitWhileCompiling(MetalCompiledGuestShader shader)
    {
        if (_finished || _inFlight.IsEmpty)
        {
            return;
        }

        if (_inFlight.TryGetValue(Identity(shader), out var compiling) && !compiling.IsSet &&
            !_utilityCompiles.ContainsKey(compiling))
        {
            var started = Stopwatch.GetTimestamp();
            if (!compiling.Wait(10_000))
            {
                // A compile the game waits on for this long holds the frame (and every packet behind it).
                Console.Error.WriteLine(
                    $"[METAL-V2][PREWARM-WAIT] a compute's library waits >10 s on the prewarm compiling it: payload={Identity(shader)} " +
                    $"entry={shader.Shader.EntryPoint} source_chars={shader.Shader.Source.Length}");
                DumpSlowSource(shader);
                compiling.Wait();
            }

            MetalCompileScheduler.NotePrewarmWait(Stopwatch.GetTimestamp() - started);
        }
    }

    private static void FlushProgress(ShaderPrewarmList list, string stamp)
    {
        var batch = new List<ulong>();
        while (_completed.TryDequeue(out var identity))
        {
            batch.Add(identity);
        }

        list.AppendProgress(stamp, batch);
    }

    // The translators' code, not their build: every commit gives every assembly a new
    // module version id, and keying on it threw away the library list and the learned
    // device data loads at the first start after any commit (AssemblyCodeIdentity).
    private static string StampPart(Type type) => AssemblyCodeIdentity.Of(type.Assembly);

    // The translators decide the MSL; the device and the OS decide what the driver's
    // compiler makes of it.
    private static string Stamp(IGuestGpuBackend compiler, nint device) =>
        string.Join(
            '|',
            StampPart(typeof(Gen5ShaderTranslator)),
            StampPart(typeof(Gen5MslTranslator)),
            compiler.GetType().FullName,
            MslMathMode.Identity(MslMathMode.Default),
            MetalNative.ReadNsString(MetalNative.Send(device, SelName)),
            RuntimeInformation.OSDescription);

    /// <summary>The identity of a compiled module, for the runtime match count: the hash of
    /// its UTF-8 payload, which the shader keeps instead of the payload itself.</summary>
    public static string Identity(MetalCompiledGuestShader shader) =>
        shader.PayloadHash.ToString("X16", System.Globalization.CultureInfo.InvariantCulture);

    private static void PrewarmComputePipeline(
        ComputePrewarmRecord record,
        ShaderCodeCapture code,
        IGuestGpuBackend compiler,
        nint device)
    {
        if (!ShaderPrewarmCompiler.TryCompilePrewarm(
                record,
                code,
                compiler,
                MetalRenderHost.PrewarmSharedInt64Atomics,
                MetalRenderHost.PrewarmExecGuardElision,
                MetalRenderHost.PrewarmSignedZeroInfNanPreserve,
                out var compiled,
                out _,
                out var error))
        {
            NoteFailure(record, error);
            return;
        }

        if (compiled is not MetalCompiledGuestShader metal)
        {
            NoteFailure(record, $"the compiler produced {compiled?.GetType().Name ?? "nothing"}, not MSL");
            return;
        }

        // Records that differ only in what the plan never reads translate to the same
        // MSL; about half of Demon's Souls' 1389 recorded computes repeat another one's
        // source, and a cold start compiled each copy again, side by side with the first.
        var identity = Identity(metal);
        if (!_prewarmedShaderIdentities!.TryAdd(identity, 0))
        {
            Interlocked.Increment(ref _deduplicated);
            return;
        }

        // The title's binary archive holds this compute's library and its pipeline's binary
        // (MetalBinaryArchive): the dispatch opens both from there and asks the driver for
        // neither, so compiling them again here would only fill the driver's own cache, at
        // the cost of a front end and a back end per program (a second start of a title
        // compiled every recorded compute once more, behind the guest's first frame for the
        // held ones). The translation above still ran: its source names the file.
        if (SkipsArchived && MetalBinaryArchive.For(device) is { } archive &&
            archive.IsStored(MetalBinaryArchive.LibraryKey(metal.Shader.Source, metal.Shader.UsesPerVertexInputs)))
        {
            Interlocked.Increment(ref _archived);
            return;
        }

        // Waits while the game compiles (MetalCompileScheduler), and holds a prewarm slot
        // until this compile is done.
        using var prewarm = MetalCompileScheduler.Shared.EnterPrewarm(static () => _stopping);
        if (!prewarm.Entered)
        {
            return;
        }

        // The game created this module while the prewarm waited: its own compile fills
        // the cache, and a second one beside it would only compete with it.
        if (_runtimeComputeIdentities?.ContainsKey(identity) == true)
        {
            Interlocked.Increment(ref _runtimeFirst);
            return;
        }

        // Not disposed: a dispatch may still be waking from it.
        var compiling = new ManualResetEventSlim(false);
        if (_onUtilityThread)
        {
            _utilityCompiles[compiling] = 0;
        }

        _inFlight[identity] = compiling;
        var compileStarted = Stopwatch.GetTimestamp();
        try
        {
            CompileComputePipeline(record, metal, device);
        }
        finally
        {
            var compileSeconds = (Stopwatch.GetTimestamp() - compileStarted) / (double)Stopwatch.Frequency;
            if (compileSeconds > 5)
            {
                Console.Error.WriteLine(
                    $"[METAL-V2][PREWARM-SLOW] a compute compile took {compileSeconds:F1} s: payload={identity} " +
                    $"entry={metal.Shader.EntryPoint} source_chars={metal.Shader.Source.Length}");
                DumpSlowSource(metal);
            }

            _inFlight.TryRemove(identity, out _);
            _utilityCompiles.TryRemove(compiling, out _);
            compiling.Set();
        }
    }

    private static void CompileComputePipeline(ComputePrewarmRecord record, MetalCompiledGuestShader metal, nint device)
    {
        string error;
        var pool = MetalNative.objc_autoreleasePoolPush();
        nint library = 0;
        nint function = 0;
        nint state = 0;
        try
        {
            library = MetalRenderHost.TryCompileLibrary(
                device, metal.Shader.Source, metal.Shader.UsesPerVertexInputs, out error);
            if (library == 0)
            {
                NoteFailure(record, error);
                return;
            }

            function = MetalNative.Send(library, SelNewFunctionWithName, MetalNative.NsString(metal.Shader.EntryPoint));
            if (function == 0)
            {
                NoteFailure(record, $"the library has no entry point {metal.Shader.EntryPoint}");
                return;
            }

            nint pipelineError = 0;
            state = MetalNative.Send(device, SelNewComputePipelineState, function, ref pipelineError);
            if (state == 0)
            {
                NoteFailure(record, MetalNative.DescribeError(pipelineError));
                return;
            }

            Interlocked.Increment(ref _compiled);
        }
        catch (Exception exception)
        {
            NoteFailure(record, exception.Message);
        }
        finally
        {
            foreach (var handle in (ReadOnlySpan<nint>)[state, function, library])
            {
                if (handle != 0)
                {
                    MetalNative.SendVoid(handle, SelRelease);
                }
            }

            MetalNative.objc_autoreleasePoolPop(pool);
        }
    }

    private static void NoteFailure(ComputePrewarmRecord record, string error)
    {
        if (Interlocked.Increment(ref _failed) <= MaxLoggedPrewarmFailures)
        {
            Console.Error.WriteLine(
                $"[LOADER][WARN] Shader prewarm failed: cs=0x{record.Hash:X16} shader=0x{record.Address:X16} error={error}");
        }
    }

    private static void Finish(ShaderPrewarmList list, string stamp, long started)
    {
        var runtime = _runtimeComputeIdentities!;
        var prewarmed = _prewarmedShaderIdentities!;
        var matched = runtime.Keys.Count(prewarmed.ContainsKey);
        if (_stopping)
        {
            Gate.SetRunning(false);
            return;
        }

        list.WriteStamp(stamp);
        list.ClearProgress();
        if (_graphicsRan && _graphicsKeys is { } graphicsKeys)
        {
            _graphics?.WriteComplete(stamp, graphicsKeys.Count);
        }

        _finished = true;
        var summary =
            $"[LOADER][INFO] Shader prewarm done: compiled={Volatile.Read(ref _compiled)} " +
            $"failed={Volatile.Read(ref _failed)} same_source={Volatile.Read(ref _deduplicated)} " +
            $"archived={Volatile.Read(ref _archived)} " +
            $"runtime_first={Volatile.Read(ref _runtimeFirst)} " +
            $"graphics_libraries={Volatile.Read(ref _graphicsCompiled)} graphics_failed={Volatile.Read(ref _graphicsFailed)} " +
            $"ms={Stopwatch.GetElapsedTime(started).TotalMilliseconds:F0} " +
            $"yielded_ms={Interlocked.Read(ref _yieldedTicks) * 1000 / Stopwatch.Frequency} " +
            $"runtime_compute_modules={runtime.Count} matched={matched}";
        if (Precompile.HoldsUntilComplete)
        {
            // The library list compiles to completion before the guest goes, or the session closes.
            Console.Error.WriteLine(summary);
            BeginLibraries();
            if (!CompletePrecompile())
            {
                Gate.SetRunning(false);
            }

            return;
        }

        Gate.SetRunning(false);
        MetalLibraryPrewarm.Begin();
        Console.Error.WriteLine(summary);
    }

    /// <summary>A compute module the running title created, counted against the prewarmed ones.</summary>
    public static void NoteRuntimeComputeModule(MetalCompiledGuestShader shader)
    {
        if (_runtimeComputeIdentities is not { } runtime || _prewarmedShaderIdentities is not { } prewarmed)
        {
            return;
        }

        var identity = Identity(shader);
        runtime[identity] = 0;
        if (!_finished || !prewarmed.ContainsKey(identity))
        {
            return;
        }

        var hits = Interlocked.Increment(ref _runtimeHits);
        if (hits is 1 or 10 or 100 or 1000)
        {
            Console.Error.WriteLine(
                $"[LOADER][INFO] Shader prewarm hits: {hits} of runtime_compute_modules={runtime.Count}");
        }
    }

    /// <summary>Stops the workers and closes the list; the gate holds nobody afterwards.</summary>
    public static void Stop()
    {
        _stopping = true;
        Gate.Cancel();
        MetalStaticPrecompile.Stop();
        MetalLibraryPrewarm.Stop();
        MetalDeviceDataLoadsStore.Stop();
        foreach (var thread in Volatile.Read(ref _threads))
        {
            if (!thread.IsCurrentThread)
            {
                thread.Join(TimeSpan.FromSeconds(10));
            }
        }

        Volatile.Write(ref _threads, []);
        if (Interlocked.Exchange(ref _list, null) is { } list)
        {
            list.Dispose();
        }

        Interlocked.Exchange(ref _graphics, null)?.Dispose();
    }

    // The same per-title directory naming as the Vulkan pipeline cache.
    internal static string SanitizeTitleId(string? titleId)
    {
        if (string.IsNullOrWhiteSpace(titleId))
        {
            return "UNKNOWN";
        }

        var source = titleId.Trim();
        var sanitized = new char[source.Length];
        for (var index = 0; index < source.Length; index++)
        {
            var value = source[index];
            sanitized[index] = char.IsAsciiLetterOrDigit(value) || value is '-' or '_'
                ? char.ToUpperInvariant(value)
                : '_';
        }

        return new string(sanitized);
    }
}
