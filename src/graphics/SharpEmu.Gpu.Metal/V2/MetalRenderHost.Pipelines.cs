// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

// Render and compute pipeline states, the MSL libraries behind them, and the
// depth/stencil state objects the dynamic masks need.
internal sealed partial class MetalRenderHost
{
    // One pipeline the executor can bind, with the encoder state Metal keeps outside
    // the pipeline object (cull, winding, fill, depth clip, primitive type).
    private sealed class MetalPipelineEntry
    {
        public ulong Id;
        public nint State;
        public bool Graphics;

        /// <summary>See <see cref="Gen5MslShader.PersistentThreads"/>.</summary>
        public bool PersistentThreads;
        public nuint PrimitiveType;
        public nint CullMode;
        public nint Winding;
        public nint DepthClipMode;
        public nint FillMode;
        public MtlSize ThreadsPerThreadgroup;
        public bool CullsEverything;
        public PipelineStaticParameters? StaticParameters;
        public bool HasDepth;
        public bool HasStencil;

        // Set when the vertex program is an NGG primitive shader compiled as a Metal
        // mesh function; the draw then launches threadgroups instead of vertices.
        public Gen5MslMeshInfo? Mesh;

        // Set for a rectangle list: the pipeline state per control-point index type
        // (none, 16-bit, 32-bit; the indexed ones are built when a draw first uses them,
        // RectangleState), and the stride of each vertex buffer the patch control
        // points step through (zero for a per-instance or constant one), which a draw
        // with a vertex offset rebinds at that offset.
        public nint[]? RectangleStates;
        public uint[]? RectangleVertexStrides;

        // Set when the vertex program picks a viewport per primitive; the draw then
        // fills every viewport slot rather than only the first.
        public bool WritesViewportIndex;

        // What a pass begun for a draw of this pipeline needs to skip a colour load
        // (MetalPassLoads): the clip positions the vertex program makes from the vertex
        // index alone, and the colour attachments every fragment overwrites whole.
        public float[]? ConstantPositions;
        public uint OverwriteMask;

        // The shader hashes behind this pipeline, formatted once for the per-draw GPU
        // profiler; null unless SHARPEMU_METAL_V2_PROFILE_SHADERS=1.
        public string? Profile;

        // The vertex module's hash (the compute module's, for a compute pipeline), for
        // the tiny-pass diagnostics and the target dump's labels.
        public ulong VertexHash;

        // The vertex descriptor this pipeline was built with, for the draw-state report:
        // a stride, a step function or an attribute format that does not match what the
        // program fetches makes every vertex the same one, and the draw then rasterises
        // nothing at all while looking perfectly ordinary from the outside.
        public string? VertexLayout;

        // What the state is made again from once the shader cache bound released it
        // (State zero; MetalRenderHost.ShaderCache): the description the executor created
        // it from, which shares its parts with the executor's own pipeline key.
        public GraphicsPipelineDescription? GraphicsDescription;
        public ComputePipelineDescription? ComputeDescription;

        // The last command buffer that encoded the state and the time it was opened: the
        // state is only released once that command buffer is done and it has been idle.
        public ulong LastUsedSerial;
        public long LastUsedTicks;
    }

    // One compiled MSL module: its library, its function, and the shader it came from.
    private sealed class MetalShaderModule
    {
        public nint Library;
        public nint Function;
        // The function with the LOD statistics counters compiled in, for a program that
        // declares the function constant (Function is then the one without), built when a
        // pipeline of a title that uses LOD statistics first needs it.
        public nint LodFunction;
        public MetalCompiledGuestShader Compiled = null!;
        public ulong Hash;

        // The reflection data. Its source text is dropped once the library is built
        // (BuildModule); Compiled.AcquireShader has the text again.
        public Gen5MslShader Shader => Compiled.Shader;

        // The library compile on the module pool; Library and Function are set once it is
        // done, which RequireModule waits for, or runs itself while no pool thread has it.
        public Task? Compiling;
        public MetalLibraryCompilePool.Job? Job;

        // What the compile spent waiting for the prewarm's compile of the same source, and
        // whether prewarm compiles were in the driver when it started: the split of the
        // worker's library wait on the PERF line.
        public long PrewarmWaitTicks;
        public bool Contended;

        // The post-tessellation form a rectangle-list pipeline draws through, built the
        // first time one uses this vertex program.
        public nint RectangleLibrary;
        public nint RectangleFunction;

        // The binary archive keys of the library and of the rectangle-list library (zero
        // with no archive): the pipeline made from them stores them under these.
        public UInt128 ArchiveKey;
        public UInt128 RectangleArchiveKey;

        // The key of the MSL text each library was compiled from (MetalBinaryArchive.LibraryKey;
        // zero with the dedupe off): modules of different permutations that translated to the
        // same text share a library, and a pipeline state is told apart by it, not by module.
        public UInt128 SourceKey;
        public UInt128 RectangleSourceKey;

        // The stage it was built for, to build it again once the shader cache bound
        // released the library (Function zero) and a pipeline needs it.
        public ShaderStage Stage;

        // When a pipeline was last made from it: only pipeline creation reads the library
        // and the function; a pipeline state keeps what it needs of them.
        public long LastUsedTicks;
    }

    private static readonly nint SelSetAttachmentPixelFormat = MetalNative.Selector("setPixelFormat:");

    private readonly Dictionary<ulong, MetalPipelineEntry> _pipelineEntries = new();
    private readonly List<MetalShaderModule> _modules = new();
    private readonly Dictionary<ulong, nint> _depthStencilStates = new();
    private ulong _nextPipelineId;

    /// <summary>Runs where a module's library compile starts, on whichever thread runs it; tests hold it.</summary>
    internal Action? BeforeLibraryCompile { get; set; }

    public ulong CreateShaderModule(IGuestCompiledShader shader, ShaderStage stage, ulong hash, ulong programId)
    {
        _ = programId;
        if (shader is not MetalCompiledGuestShader metal)
        {
            throw Fatal($"The shader handle was not compiled by the Metal backend: type={shader.GetType().Name}.");
        }

        if (stage == ShaderStage.Compute)
        {
            MetalShaderPrewarm.NoteRuntimeComputeModule(metal);
        }

        var module = new MetalShaderModule
        {
            Compiled = metal,
            Hash = hash,
            Stage = stage,
            LastUsedTicks = System.Diagnostics.Stopwatch.GetTimestamp(),
        };

        // The library compiles on the module pool and the handle is returned at once: the
        // caller goes on translating the draw's other program, whose library then compiles
        // alongside, and only the pipeline state waits for both (RequireModule). A draw that
        // brought two new programs used to run one front end after the other on the GPU
        // worker, with the translation of the second in between.
        void build()
        {
            try
            {
                BeforeLibraryCompile?.Invoke();
                _ = MetalCompileScheduler.TakeThreadPrewarmWait();
                if (stage == ShaderStage.Compute)
                {
                    // The background prewarm may have this source in hand; its library is
                    // then a cache hit here once it is done.
                    MetalShaderPrewarm.WaitWhileCompiling(metal);
                }

                module.Contended = MetalCompileScheduler.Shared.PrewarmInFlight > 0;
                BuildModule(module, stage);
            }
            finally
            {
                Volatile.Write(ref module.PrewarmWaitTicks, MetalCompileScheduler.TakeThreadPrewarmWait());
                MetalShaderPrewarm.GameCompileDone();
            }
        }

        // From here until the module is built the background prewarm holds back.
        MetalShaderPrewarm.GameCompileStarted();
        if (MetalLibraryCompilePool.Enabled)
        {
            module.Job = MetalLibraryCompilePool.Shared.Run(build);
            module.Compiling = module.Job.Task;
        }
        else
        {
            build();
        }

        int index;
        // Module creation can reach the host from the shader cache's own threads,
        // while RequireModule runs on the GPU worker; the list is shared state.
        lock (_modules)
        {
            _modules.Add(module);
            index = _modules.Count;
        }

        Interlocked.Increment(ref _shaderCompilations);
        // Module zero never identifies a module, so the handle is the one-based index.
        return (ulong)index;
    }

    // The library and the function of a module; on the module pool or the caller's thread.
    // Everything that needs the source text has had it by the end (the translator's
    // dump, the permutation and compiled-shader dumps, the compute prewarm identity, the
    // library list's key and record), so the text is dropped there instead of being held
    // for the host's lifetime.
    private void BuildModule(MetalShaderModule module, ShaderStage stage)
    {
        var shader = module.Compiled.AcquireShader();
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        // The MSL front end runs here, and the driver's back end runs when a pipeline
        // state is made from the function. They are separate costs with separate
        // caches, so they are counted separately. A graphics library the library prewarm
        // already compiled is taken from it instead.
        var library = CompileLibrary(
            shader.Source, module.Hash, shader.UsesPerVertexInputs, prewarmable: stage != ShaderStage.Compute,
            out module.ArchiveKey, out module.SourceKey);
        Interlocked.Add(ref _libraryTicks, System.Diagnostics.Stopwatch.GetTimestamp() - started);
        var pool = MetalNative.objc_autoreleasePoolPush();
        try
        {
            nint specializationError = 0;
            var function = shader.UsesMipStatisticsConstant
                ? NewMipStatisticsFunction(library, shader.EntryPoint, enabled: false, out specializationError)
                : MetalNative.Send(
                    library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString(shader.EntryPoint));
            if (function == 0)
            {
                ReleaseLibrary(library);
                throw Fatal(
                    $"The MSL library has no entry point: entry={shader.EntryPoint} hash=0x{module.Hash:X16}" +
                    (specializationError != 0 ? $" {MetalNative.DescribeError(specializationError)}." : "."));
            }

            module.Library = library;
            Volatile.Write(ref module.Function, function);
            Interlocked.Increment(ref _residentModules);
        }
        finally
        {
            MetalNative.objc_autoreleasePoolPop(pool);
        }

        module.Compiled.ReleaseSource();

        MetalV2Diagnostics.Trace(
            $"compiled module hash=0x{module.Hash:X16} entry={shader.EntryPoint} " +
            $"ms={(System.Diagnostics.Stopwatch.GetTimestamp() - started) * 1000.0 / System.Diagnostics.Stopwatch.Frequency:F1}");
    }

    // The pixel function of a pipeline: the one with the LOD statistics counters compiled in
    // when the pipeline is for a title that uses them and the program has the sites.
    private nint PixelFunctionOf(MetalShaderModule module, bool lodStatistics)
    {
        if (!module.Shader.UsesMipStatisticsConstant)
        {
            return module.Function;
        }

        if (!lodStatistics)
        {
            // This pipeline has no counters: samples of draws through it never reach the bank.
            Volatile.Write(ref _lodStatsCompiledOut, true);
            return module.Function;
        }

        lock (module)
        {
            if (module.LodFunction != 0)
            {
                return module.LodFunction;
            }

            var function = NewMipStatisticsFunction(module.Library, module.Shader.EntryPoint, enabled: true, out var error);
            if (function == 0)
            {
                throw Fatal(
                    $"The pixel function with LOD statistics could not be made: hash=0x{module.Hash:X16} " +
                    $"{MetalNative.DescribeError(error)}.");
            }

            module.LodFunction = function;
            return function;
        }
    }

    private nint RequireRectangleFunction(MetalShaderModule module)
    {
        lock (module)
        {
            if (module.RectangleFunction != 0)
            {
                return module.RectangleFunction;
            }

            var library = CompileLibrary(
                Gen5MslRectangleList.BuildSource(module.Compiled.AcquireShader()), module.Hash, usesPerVertexInputs: false, prewarmable: true,
                out module.RectangleArchiveKey, out module.RectangleSourceKey);
            var function = MetalNative.Send(
                library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString(Gen5MslRectangleList.EntryPoint));
            if (function == 0)
            {
                ReleaseLibrary(library);
                throw Fatal(
                    $"The rectangle-list MSL library has no entry point: entry={Gen5MslRectangleList.EntryPoint} hash=0x{module.Hash:X16}.");
            }

            module.RectangleLibrary = library;
            module.RectangleFunction = function;
            return function;
        }
    }

    // Libraries by the key of their source text. Permutations of one program (a different
    // resource specialization, a different vertex stride) often translate to the same MSL; the
    // library of the first is then shared by the rest instead of compiled once each. Off with
    // SHARPEMU_METAL_V2_PIPELINE_DEDUPE=0.
    internal static readonly bool PipelineDedupe =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_PIPELINE_DEDUPE") != "0";

    private readonly MetalSharedHandleTable<UInt128> _libraries = new();
    private readonly Dictionary<UInt128, ManualResetEventSlim> _librariesBuilding = new();
    private readonly object _librariesGate = new();
    private long _moduleSameSource;
    private long _pipelineDedups;

    // The library of this source, compiled by the first caller and shared with the rest. Any
    // thread: modules compile on the module pool. The caller owns one reference to give back
    // through ReleaseLibrary.
    private nint CompileLibrary(
        string source, ulong hash, bool usesPerVertexInputs, bool prewarmable, out UInt128 archiveKey, out UInt128 sourceKey)
    {
        if (!PipelineDedupe)
        {
            sourceKey = default;
            return CompileLibraryUnshared(source, hash, usesPerVertexInputs, prewarmable, out archiveKey);
        }

        sourceKey = MetalBinaryArchive.LibraryKey(source, usesPerVertexInputs);
        while (true)
        {
            ManualResetEventSlim? building;
            var creator = false;
            lock (_librariesGate)
            {
                if (_libraries.TryAcquire(sourceKey, out var shared, out var sharedArchiveKey))
                {
                    Interlocked.Increment(ref _moduleSameSource);
                    archiveKey = sharedArchiveKey;
                    return shared;
                }

                if (!_librariesBuilding.TryGetValue(sourceKey, out building))
                {
                    building = new ManualResetEventSlim(false);
                    _librariesBuilding.Add(sourceKey, building);
                    creator = true;
                }
            }

            if (!creator)
            {
                // The same text is compiling on another thread; take its library once it is
                // there, or compile it here if that compile failed.
                building.Wait();
                continue;
            }

            try
            {
                var library = CompileLibraryUnshared(source, hash, usesPerVertexInputs, prewarmable, out archiveKey);
                lock (_librariesGate)
                {
                    if (!_libraries.Register(sourceKey, library, archiveKey, out var winner))
                    {
                        ReleaseLibrary(library);
                        library = winner;
                    }

                    return library;
                }
            }
            finally
            {
                lock (_librariesGate)
                {
                    _librariesBuilding.Remove(sourceKey);
                }

                building.Set();
            }
        }
    }

    // One owner of a shared library lets go; the library goes with the last.
    private void ReleaseLibrary(nint library)
    {
        if (_libraries.Release(library))
        {
            MetalNative.SendVoid(library, SelRelease);
        }
    }

    // Computes are left to the compute prewarm, which compiles them before the game runs.
    private nint CompileLibraryUnshared(string source, ulong hash, bool usesPerVertexInputs, bool prewarmable, out UInt128 archiveKey)
    {
        // The background prewarm gives way while the game compiles (MetalCompileScheduler).
        using var runtime = MetalCompileScheduler.Shared.Runtime();
        var listed = prewarmable && MetalLibraryPrewarm.Active;
        var key = listed ? MetalLibraryPrewarm.Key(source, usesPerVertexInputs) : 0;
        if (listed && MetalLibraryPrewarm.TryTake(key, out var prewarmed))
        {
            MetalLibraryPrewarm.NoteTaken(key);
            archiveKey = MetalBinaryArchive.For(_device) is not null
                ? MetalBinaryArchive.LibraryKey(source, usesPerVertexInputs)
                : default;
            return prewarmed;
        }

        var library = TryCompileLibrary(_device, source, usesPerVertexInputs, out var error, out archiveKey);
        if (library == 0)
        {
            throw Fatal($"An MSL library failed to compile: hash=0x{hash:X16} {error}");
        }

        if (listed)
        {
            MetalLibraryPrewarm.NoteCompiled(key, source, usesPerVertexInputs);
        }

        return library;
    }

    // The one MSL front-end call, shared with the shader prewarm so both ask the driver
    // with the same options and hit the same cache entries. Zero and a description on
    // failure; any thread.
    internal static nint TryCompileLibrary(nint device, string source, bool usesPerVertexInputs, out string error) =>
        TryCompileLibrary(device, source, usesPerVertexInputs, out error, out _);

    // The title's binary archive is asked first: a library stored there is its AIR, with no
    // front end to run (MetalBinaryArchive). The key comes back for storing it there later.
    // The static precompile, which knows the archive lacks it, does not ask.
    internal static nint TryCompileLibrary(
        nint device, string source, bool usesPerVertexInputs, out string error, out UInt128 archiveKey, bool askArchive = true)
    {
        error = string.Empty;
        archiveKey = default;
        if (askArchive && MetalBinaryArchive.For(device) is { } archive)
        {
            archiveKey = MetalBinaryArchive.LibraryKey(source, usesPerVertexInputs);
            var stored = archive.TryLoadLibrary(archiveKey);
            if (stored != 0)
            {
                return stored;
            }
        }

        var pool = MetalNative.objc_autoreleasePoolPush();
        try
        {
            var options = MetalNative.Send(
                MetalNative.Send(MetalNative.Class("MTLCompileOptions"), MetalNative.Selector("alloc")),
                MetalNative.Selector("init"));
            if (options == 0)
            {
                error = "Metal compile options could not be created.";
                return 0;
            }

            try
            {
                // Fast math off (the translated GCN float semantics do not survive
                // it) and an explicit MSL version, so texture atomics are available.
                MetalNative.ApplyMslCompileOptions(options);
                if (usesPerVertexInputs)
                {
                    MetalNative.SetMslLanguageVersion(options, Gen5MslShader.PerVertexInputsLanguageVersion);
                }

                nint compileError = 0;
                var library = MetalNative.Send(
                    device,
                    MetalNative.Selector("newLibraryWithSource:options:error:"),
                    MetalNative.NsString(source),
                    options,
                    ref compileError);
                if (library == 0)
                {
                    error = MetalNative.DescribeError(compileError);
                }

                return library;
            }
            finally
            {
                MetalNative.SendVoid(options, SelRelease);
            }
        }
        finally
        {
            MetalNative.objc_autoreleasePoolPop(pool);
        }
    }

    // The module of a program, as it is: its reflection is there from the start, its library
    // and function may still be compiling or released.
    private MetalShaderModule LookupModule(in ShaderProgram program)
    {
        lock (_modules)
        {
            var index = (int)program.Module - 1;
            if (index < 0 || index >= _modules.Count)
            {
                throw Fatal($"The shader module handle is unknown: module={program.Module} program={program.Id}.");
            }

            return _modules[index];
        }
    }

    // The module with its library and function, waiting for a compile still on the pool. A
    // pipeline compiler thread asks too (onWorker false); its waits are not the worker's.
    private MetalShaderModule RequireModule(in ShaderProgram program, bool onWorker = true)
    {
        var module = LookupModule(program);
        if (module.Compiling is { } compiling)
        {
            if (!compiling.IsCompleted && !onWorker)
            {
                if (module.Job is not { } queued || !MetalLibraryCompilePool.Shared.TryRunHere(queued))
                {
                    ((IAsyncResult)compiling).AsyncWaitHandle.WaitOne();
                }
            }
            else if (!compiling.IsCompleted)
            {
                var started = System.Diagnostics.Stopwatch.GetTimestamp();
                // A module still queued behind the pool's other compiles runs here and now:
                // the worker needs it before any of them.
                if (module.Job is { } job && MetalLibraryCompilePool.Shared.TryRunHere(job))
                {
                    Interlocked.Increment(ref _libraryInlineCompiles);
                }
                else
                {
                    ((IAsyncResult)compiling).AsyncWaitHandle.WaitOne();
                }

                NoteLibraryWait(module, System.Diagnostics.Stopwatch.GetTimestamp() - started);
            }

            // A failed compile is fatal where the module is first needed, as it was where
            // the module was made.
            compiling.GetAwaiter().GetResult();
        }

        module.LastUsedTicks = System.Diagnostics.Stopwatch.GetTimestamp();
        if (Volatile.Read(ref module.Function) == 0)
        {
            RebuildModule(module);
        }

        return module;
    }

    private void NoteLibraryWait(MetalShaderModule module, long waited)
    {
        Interlocked.Add(ref _libraryWaitTicks, waited);
        Interlocked.Increment(ref _libraryWaits);
        var onPrewarm = Math.Min(waited, Volatile.Read(ref module.PrewarmWaitTicks));
        Interlocked.Add(ref _libraryWaitPrewarmTicks, onPrewarm);
        if (module.Contended)
        {
            Interlocked.Add(ref _libraryWaitContendedTicks, waited - onPrewarm);
        }
    }

    // How long a library wait before a preparation sleeps between looks at the store requests.
    private const int LibraryWaitServiceMilliseconds = 1;

    /// <summary>
    /// The library compiles of a draw or dispatch that will wait for its pipeline, waited for
    /// before its preparation opens, serving guest stores that wait for a writeback meanwhile.
    /// A store's request is otherwise taken only between packets: inside the preparation the
    /// images the draw acquired are stamped with the open submission, which serving it would
    /// commit, so a request posted while the worker waited there for a cold compile waited for
    /// the whole compile, a second and more. Here nothing of the draw is recorded yet, as at the
    /// draw's entry where the relay runs anyway. Only the store requests are served, not the
    /// whole relay. A compile no pool thread has taken yet starts at once on a thread of its
    /// own, as RequireModule would run it on the worker, and the worker goes on serving.
    /// RequireModule then finds the modules built.
    /// </summary>
    public void AwaitProgramsBeforePreparation(ShaderProgram first, ShaderProgram second)
    {
        var firstCompiling = CompilingModule(first);
        var secondCompiling = CompilingModule(second);
        if (firstCompiling is null && secondCompiling is null)
        {
            return;
        }

        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        StartQueuedCompile(firstCompiling);
        StartQueuedCompile(secondCompiling);
        AwaitServingStores(firstCompiling);
        AwaitServingStores(secondCompiling);
        var waited = System.Diagnostics.Stopwatch.GetTimestamp() - started;
        if (firstCompiling is not null)
        {
            NoteLibraryWait(firstCompiling, waited);
        }

        if (secondCompiling is not null)
        {
            NoteLibraryWait(secondCompiling, firstCompiling is null ? waited : 0);
        }
    }

    private MetalShaderModule? CompilingModule(ShaderProgram program)
    {
        if (!program.IsValid)
        {
            return null;
        }

        var module = LookupModule(program);
        return module.Compiling is { IsCompleted: false } ? module : null;
    }

    private void StartQueuedCompile(MetalShaderModule? module)
    {
        if (module?.Job is not { IsQueued: true } job)
        {
            return;
        }

        new Thread(() =>
        {
            if (MetalLibraryCompilePool.Shared.TryRunHere(job))
            {
                Interlocked.Increment(ref _libraryInlineCompiles);
            }
        })
        {
            IsBackground = true,
            Name = "SharpEmu Metal v2 library compile (needed now)",
        }.Start();
    }

    private void AwaitServingStores(MetalShaderModule? module)
    {
        if (module?.Compiling is not { } compiling)
        {
            return;
        }

        var done = ((IAsyncResult)compiling).AsyncWaitHandle;
        do
        {
            _imageCache.ServiceStoreRequests();
        }
        while (!done.WaitOne(LibraryWaitServiceMilliseconds));
    }

    /// <summary>A rectangle list draws one patch per rectangle, so its count is needed up
    /// front; see <see cref="DrawRectangles"/>. An indexed one whose counts the GPU writes
    /// is built in the pass instead (<see cref="TakesIndirectIndexedRectangleListArguments"/>).</summary>
    public bool TakesIndirectRectangleListArguments => false;

    /// <summary>Metal has no fan primitive; the executor hands fans over as triangle lists.</summary>
    public bool DrawsTriangleFans => false;

    private MetalPipelineEntry RequirePipelineEntry(in PipelineHandle pipeline) =>
        _pipelineEntries.TryGetValue(pipeline.Pipeline, out var entry)
            ? entry
            : throw Fatal($"The pipeline handle is unknown: pipeline={pipeline.Pipeline}.");

    public PipelineHandle CreateGraphicsPipeline(GraphicsPipelineDescription description)
    {
        Interlocked.Increment(ref _pipelineWaitsBlocking);
        RequireDrawableDescription(description);
        // A draw that could be left out may have started this compile; it is waited for
        // rather than built a second time.
        if (_compiler is { } compiler && compiler.TryWait(GraphicsCompileKey(description), out var started))
        {
            return RegisterGraphicsPipeline(description, started!);
        }

        return RegisterGraphicsPipeline(description, BuildGraphicsPipelineStates(description, onWorker: true));
    }

    /// <summary>
    /// The pipeline of a draw the executor may leave out, or false while the compiler threads
    /// build it: the library compiles of its programs and the driver's back end both run there,
    /// several pipelines at once, and the draw is left out until a later one finds it done.
    /// </summary>
    public bool TryCreateGraphicsPipeline(GraphicsPipelineDescription description, out PipelineHandle handle)
    {
        if (!AsynchronousGraphics)
        {
            handle = CreateGraphicsPipeline(description);
            return true;
        }

        handle = default;
        RequireDrawableDescription(description);
        if (!Compiler.TryTake(
                GraphicsCompileKey(description),
                () => BuildGraphicsPipelineStates(description, onWorker: false),
                out var compiled))
        {
            Interlocked.Increment(ref _drawsSkippedCompiling);
            return false;
        }

        handle = RegisterGraphicsPipeline(description, compiled!);
        return true;
    }

    // What identifies a graphics compile: the parts of the description its states are built from.
    private static GraphicsPipelineKey GraphicsCompileKey(GraphicsPipelineDescription description) => new()
    {
        Rendering = description.Rendering,
        VertexProgramId = description.VertexProgram.Id,
        PixelProgramId = description.PixelInfo is not null ? description.PixelProgram.Id : 0,
        LodStatistics = description.LodStatistics,
        VertexInput = description.VertexInput,
        StaticParameters = description.StaticParameters,
    };

    private static Gen5MslMeshInfo? MeshOf(MetalShaderModule vertexModule) =>
        vertexModule.Shader.Stage == Gen5MslStage.Mesh ? vertexModule.Shader.MeshInfo : null;

    // A rectangle list draws through the program's post-tessellation form (see
    // Gen5MslRectangleList); the index type of its control points is part of the
    // pipeline state, so one state is built for each kind of draw.
    private static bool IsRectangleList(GraphicsPipelineDescription description, Gen5MslMeshInfo? mesh) =>
        mesh is null && description.StaticParameters.Topology == GpuPrimitiveTopology.PatchList;

    // Refused on the worker, before any compile starts.
    private void RequireDrawableDescription(GraphicsPipelineDescription description)
    {
        var rectangleList = IsRectangleList(description, MeshOf(LookupModule(description.VertexProgram)));
        if (rectangleList && description.VertexInput.BindingCount > RectangleRebaseFirstSlot - VertexBufferSlotBase)
        {
            throw Fatal(
                $"A rectangle list binds vertex buffers into the slots its draw block, index buffer and rebase draw use: " +
                $"buffers={description.VertexInput.BindingCount} vertex=0x{description.VertexStage.Hash:X16}.");
        }
    }

    // The render pipeline states of a description; on the worker, or on a compiler thread,
    // which then waits for (or runs) the library compiles of its modules itself. Touches only
    // the modules, the device, the archive and counters, all safe from any thread.
    private MetalPipelineCompiler.Result BuildGraphicsPipelineStates(GraphicsPipelineDescription description, bool onWorker)
    {
        var vertexModule = RequireModule(description.VertexProgram, onWorker);
        var pixelModule = description.PixelStage is not null && description.PixelProgram.IsValid
            ? RequireModule(description.PixelProgram, onWorker)
            : null;
        var mesh = MeshOf(vertexModule);
        var rectangleList = IsRectangleList(description, mesh);
        var state = CreateRenderPipelineState(
            description, vertexModule, pixelModule, mesh, rectangleList, Mtl.ControlPointIndexTypeNone, out var vertexLayout);
        nint[]? rectangleStates = rectangleList ? [state, 0, 0] : null;
        return new MetalPipelineCompiler.Result(state, rectangleStates, vertexLayout, ReleaseState);
    }

    // The pipeline entry for finished states. On the worker, so the entry table and the
    // identifier counter stay single threaded; reads only the modules' reflection.
    private PipelineHandle RegisterGraphicsPipeline(GraphicsPipelineDescription description, MetalPipelineCompiler.Result compiled)
    {
        var parameters = description.StaticParameters;
        var vertexModule = LookupModule(description.VertexProgram);
        var pixelModule = description.PixelStage is not null && description.PixelProgram.IsValid
            ? LookupModule(description.PixelProgram)
            : null;
        var mesh = MeshOf(vertexModule);
        var rectangleList = IsRectangleList(description, mesh);
        if (mesh is not null)
        {
            // One line per mesh pipeline: what the launch was sized from, for reading a
            // title's primitive shaders off its log.
            Console.Error.WriteLine(
                $"[METAL-V2][MESH-PIPE] vs=0x{vertexModule.Hash:X16} wave64={(mesh.Wave64 ? 1 : 0)} " +
                $"narrow={(mesh.NarrowWave64 ? 1 : 0)} gs={(mesh.HasGeometryHalf ? 1 : 0)} " +
                $"lds_bytes={mesh.LdsBytes} device_lds={(mesh.DeviceLds ? 1 : 0)} " +
                $"max_v={mesh.MaxVertices} max_p={mesh.MaxPrimitives} threads={mesh.ThreadsPerThreadgroup}");
        }

        Interlocked.Increment(ref _pipelineCreations);
        var entry = new MetalPipelineEntry
        {
            Id = ++_nextPipelineId,
            State = compiled.State,
            Graphics = true,
            PrimitiveType = MetalV2Formats.Topology(parameters.Topology),
            CullMode = parameters.CullFront && parameters.CullBack
                ? Mtl.CullModeNone
                : parameters.CullFront ? Mtl.CullModeFront : parameters.CullBack ? Mtl.CullModeBack : Mtl.CullModeNone,
            CullsEverything = parameters.CullFront && parameters.CullBack,
            Winding = parameters.FrontFaceClockwise ? Mtl.WindingClockwise : Mtl.WindingCounterClockwise,
            DepthClipMode = parameters.DepthClipEnable ? Mtl.DepthClipModeClip : Mtl.DepthClipModeClamp,
            FillMode = 0,
            StaticParameters = parameters,
            HasDepth = depthFormatOf(description) != MtlFormat.Invalid,
            HasStencil = description.Rendering.StencilFormat != GpuFormat.Undefined,
            Mesh = mesh,
            RectangleStates = compiled.RectangleStates,
            RectangleVertexStrides = rectangleList ? RectangleVertexStrides(description.VertexInput) : null,
            WritesViewportIndex = vertexModule.Shader.WritesViewportIndex,
            ConstantPositions = mesh is null ? vertexModule.Shader.ConstantPositions : null,
            OverwriteMask = OverwriteMaskOf(description, pixelModule),
            Profile = ProfileLabelFor(vertexModule, pixelModule, mesh),
            VertexHash = vertexModule.Hash,
            VertexLayout = compiled.VertexLayout,
            GraphicsDescription = description,
            LastUsedTicks = System.Diagnostics.Stopwatch.GetTimestamp(),
        };
        _pipelineEntries[entry.Id] = entry;
        Interlocked.Increment(ref _residentPipelines);
        return new PipelineHandle(entry.Id, 0, false);

        static ulong depthFormatOf(GraphicsPipelineDescription value) =>
            value.Rendering.DepthFormat != GpuFormat.Undefined
                ? MetalV2Formats.PixelFormat(value.Rendering.DepthFormat)
                : MtlFormat.Invalid;
    }

    /// <summary>The colour attachments every fragment of the pipeline writes whole: the
    /// pixel program returns a value for it and cannot discard, blending is off (as the
    /// pipeline state below decides it) and the write mask has every channel. One sample.</summary>
    private static uint OverwriteMaskOf(GraphicsPipelineDescription description, MetalShaderModule? pixelModule)
    {
        var parameters = description.StaticParameters;
        var count = (int)Math.Min(description.Rendering.ColorCount, 32u);
        Span<bool> blends = stackalloc bool[count];
        Span<nint> writeMasks = stackalloc nint[count];
        for (var index = 0; index < count; index++)
        {
            var format = MetalV2Formats.PixelFormat(description.Rendering.ColorFormats[index]);
            // The same test the pipeline state below uses to turn blending on.
            blends[index] = parameters.GetBlendEnable(index) && !parameters.GetBlendBypass(index) && !IsIntegerFormat(format);
            writeMasks[index] = MetalV2Formats.ColorWriteMask(parameters.GetColorMask(index));
        }

        return MetalPassLoads.OverwriteMask(
            pixelModule is not null && pixelModule.Shader.CannotDiscard,
            pixelModule?.Shader.ColorOutputMask ?? 0,
            parameters.Samples,
            blends,
            writeMasks);
    }

    private nint CreateRenderPipelineState(
        GraphicsPipelineDescription description,
        MetalShaderModule vertexModule,
        MetalShaderModule? pixelModule,
        Gen5MslMeshInfo? mesh,
        bool rectangleList,
        nuint controlPointIndexType,
        out string? vertexLayout)
    {
        var parameters = description.StaticParameters;
        var descriptorClass = mesh is null ? "MTLRenderPipelineDescriptor" : "MTLMeshRenderPipelineDescriptor";
        var descriptor = MetalNative.Send(
            MetalNative.Send(MetalNative.Class(descriptorClass), MetalNative.Selector("alloc")),
            MetalNative.Selector("init"));
        if (descriptor == 0)
        {
            throw Fatal($"A Metal render pipeline descriptor could not be created: class={descriptorClass}.");
        }

        nint state;
        vertexLayout = null;
        try
        {
            if (mesh is null)
            {
                MetalNative.SendVoid(
                    descriptor,
                    MetalNative.Selector("setVertexFunction:"),
                    rectangleList ? RequireRectangleFunction(vertexModule) : vertexModule.Function);
                MetalV2Native.SendVoidValue(
                    descriptor, MetalNative.Selector("setRasterSampleCount:"), Math.Max(parameters.Samples, 1));
                if (rectangleList)
                {
                    // Every patch is one rectangle tessellated at factor 1: the four corners as
                    // two triangles. Counter-clockwise output keeps the guest triangle's facing,
                    // since corners (0,0), (1,0) and (0,1) are its three vertices in order.
                    MetalV2Native.SendVoidValue(
                        descriptor, MetalNative.Selector("setTessellationFactorStepFunction:"), Mtl.TessellationFactorStepFunctionConstant);
                    MetalV2Native.SendVoidValue(
                        descriptor, MetalNative.Selector("setTessellationPartitionMode:"), Mtl.TessellationPartitionModeInteger);
                    MetalV2Native.SendVoidValue(
                        descriptor, MetalNative.Selector("setTessellationOutputWindingOrder:"), (nuint)Mtl.WindingCounterClockwise);
                    MetalV2Native.SendVoidValue(
                        descriptor, MetalNative.Selector("setTessellationControlPointIndexType:"), controlPointIndexType);
                }
            }
            else
            {
                // A mesh function has no stage_in and no vertex descriptor: the
                // primitive shader fetches its own attributes through the resource path.
                MetalNative.SendVoid(descriptor, MetalNative.Selector("setMeshFunction:"), vertexModule.Function);
                MetalV2Native.SendVoidValue(
                    descriptor,
                    MetalNative.Selector("setMaxTotalThreadsPerMeshThreadgroup:"),
                    Math.Max(mesh.ThreadsPerThreadgroup, 1));
                MetalV2Native.SendVoidValue(
                    descriptor, MetalNative.Selector("setRasterSampleCount:"), Math.Max(parameters.Samples, 1));
            }

            if (mesh is null && (vertexModule.Shader.WritesLayer || vertexModule.Shader.WritesViewportIndex))
            {
                // Layered and multi-viewport rendering need the primitive class up front.
                MetalV2Native.SendVoidValue(
                    descriptor,
                    MetalNative.Selector("setInputPrimitiveTopology:"),
                    MetalV2Formats.TopologyClass(parameters.Topology));
            }

            if (pixelModule is not null)
            {
                MetalNative.SendVoid(descriptor, MetalNative.Selector("setFragmentFunction:"), PixelFunctionOf(pixelModule, description.LodStatistics));
            }

            if (parameters.AlphaToCoverageEnable)
            {
                MetalNative.SendVoidBool(descriptor, MetalNative.Selector("setAlphaToCoverageEnabled:"), true);
            }

            var colorAttachments = MetalNative.Send(descriptor, SelColorAttachments);
            for (var index = 0; index < description.Rendering.ColorCount; index++)
            {
                var attachment = MetalNative.SendAtIndex(colorAttachments, SelObjectAtIndex, (nuint)index);
                var format = MetalV2Formats.PixelFormat(description.Rendering.ColorFormats[index]);
                MetalV2Native.SendVoidValue(attachment, SelSetAttachmentPixelFormat, (nuint)format);
                MetalV2Native.SendVoidValue(
                    attachment,
                    MetalNative.Selector("setWriteMask:"),
                    (nuint)MetalV2Formats.ColorWriteMask(parameters.GetColorMask(index)));
                if (!parameters.GetBlendEnable(index) || parameters.GetBlendBypass(index) || IsIntegerFormat(format))
                {
                    continue;
                }

                MetalNative.SendVoidBool(attachment, MetalNative.Selector("setBlendingEnabled:"), true);
                MetalV2Native.SendVoidValue(
                    attachment,
                    MetalNative.Selector("setSourceRGBBlendFactor:"),
                    (nuint)MetalV2Formats.BlendFactor(parameters.GetColorSourceBlend(index)));
                MetalV2Native.SendVoidValue(
                    attachment,
                    MetalNative.Selector("setDestinationRGBBlendFactor:"),
                    (nuint)MetalV2Formats.BlendFactor(parameters.GetColorDestinationBlend(index)));
                MetalV2Native.SendVoidValue(
                    attachment,
                    MetalNative.Selector("setRgbBlendOperation:"),
                    (nuint)MetalV2Formats.BlendOperation(parameters.GetColorBlendFunction(index)));
                var separate = parameters.GetSeparateAlphaBlend(index);
                MetalV2Native.SendVoidValue(
                    attachment,
                    MetalNative.Selector("setSourceAlphaBlendFactor:"),
                    (nuint)MetalV2Formats.BlendFactor(
                        separate ? parameters.GetAlphaSourceBlend(index) : parameters.GetColorSourceBlend(index)));
                MetalV2Native.SendVoidValue(
                    attachment,
                    MetalNative.Selector("setDestinationAlphaBlendFactor:"),
                    (nuint)MetalV2Formats.BlendFactor(
                        separate ? parameters.GetAlphaDestinationBlend(index) : parameters.GetColorDestinationBlend(index)));
                MetalV2Native.SendVoidValue(
                    attachment,
                    MetalNative.Selector("setAlphaBlendOperation:"),
                    (nuint)MetalV2Formats.BlendOperation(
                        separate ? parameters.GetAlphaBlendFunction(index) : parameters.GetColorBlendFunction(index)));
            }

            var depthFormat = description.Rendering.DepthFormat != GpuFormat.Undefined
                ? MetalV2Formats.PixelFormat(description.Rendering.DepthFormat)
                : MtlFormat.Invalid;
            var stencilFormat = description.Rendering.StencilFormat != GpuFormat.Undefined
                ? MetalV2Formats.PixelFormat(description.Rendering.StencilFormat)
                : MtlFormat.Invalid;
            if (depthFormat != MtlFormat.Invalid)
            {
                MetalV2Native.SendVoidValue(
                    descriptor, MetalNative.Selector("setDepthAttachmentPixelFormat:"), (nuint)depthFormat);
            }

            if (stencilFormat != MtlFormat.Invalid && MetalV2Formats.HasStencil(stencilFormat))
            {
                MetalV2Native.SendVoidValue(
                    descriptor, MetalNative.Selector("setStencilAttachmentPixelFormat:"), (nuint)stencilFormat);
            }

            if (mesh is null && description.VertexInput.AttributeCount != 0)
            {
                var vertexDescriptor = CreateVertexDescriptor(description, rectangleList, out vertexLayout);
                MetalNative.SendVoid(descriptor, MetalNative.Selector("setVertexDescriptor:"), vertexDescriptor);
                MetalNative.SendVoid(vertexDescriptor, SelRelease);
            }

            // A state already made from an identical descriptor is shared (the table owns it,
            // each pipeline entry is one of its owners).
            MetalPipelineStateKey? shareKey = null;
            if (PipelineDedupe && mesh is null)
            {
                shareKey = PipelineStateKeyOf(
                    description,
                    rectangleList ? vertexModule.RectangleSourceKey : vertexModule.SourceKey,
                    pixelModule?.SourceKey,
                    vertexModule.Shader.WritesLayer || vertexModule.Shader.WritesViewportIndex,
                    rectangleList,
                    controlPointIndexType,
                    description.LodStatistics && pixelModule is { Shader.UsesMipStatisticsConstant: true });
                if (_states.TryAcquire(shareKey, out var shared, out _))
                {
                    Interlocked.Increment(ref _pipelineDedups);
                    return shared;
                }
            }

            nint error = 0;
            using var runtime = MetalCompileScheduler.Shared.Runtime();
            var started = System.Diagnostics.Stopwatch.GetTimestamp();
            var archive = MetalBinaryArchive.For(_device);
            var vertexArchiveKey = rectangleList ? vertexModule.RectangleArchiveKey : vertexModule.ArchiveKey;
            // The same library makes two pipelines, with the counters and without, so the pair
            // the archive lists them under tells them apart.
            var pixelArchiveKey = pixelModule is null
                ? default
                : MetalBinaryArchive.LodStatisticsKey(
                    pixelModule.ArchiveKey, description.LodStatistics && pixelModule.Shader.UsesMipStatisticsConstant);
            var archives = archive?.AcquirePipelineArchives(vertexArchiveKey, pixelArchiveKey) ?? 0;
            state = 0;
            if (archives != 0)
            {
                // The title's archive first; only a miss compiles.
                MetalNative.SendVoid(descriptor, SelSetBinaryArchives, archives);
                state = NewRenderPipelineState(descriptor, mesh is not null, MetalBinaryArchive.FailOnBinaryArchiveMiss, ref error);
                MetalNative.SendVoid(descriptor, SelSetBinaryArchives, 0);
                MetalNative.SendVoid(archives, SelRelease);
                archive!.NotePipelineLookup(state != 0);
                error = 0;
            }
            else
            {
                // No segment holds the shader pair.
                archive?.NotePipelineLookup(false);
            }

            var missed = state == 0;
            if (missed)
            {
                state = NewRenderPipelineState(descriptor, mesh is not null, 0, ref error);
            }

            Interlocked.Add(ref _pipelineTicks, System.Diagnostics.Stopwatch.GetTimestamp() - started);
            if (state != 0 && archive is not null)
            {
                archive.NoteRenderPipeline(
                    descriptor,
                    mesh is null ? MetalBinaryArchive.PipelineKind.Render : MetalBinaryArchive.PipelineKind.Mesh,
                    vertexArchiveKey,
                    pixelArchiveKey,
                    missed);
            }

            if (state == 0)
            {
                throw Fatal(
                    $"A Metal render pipeline state could not be created: vertex=0x{description.VertexStage.Hash:X16} " +
                    $"pixel=0x{description.PixelStage?.Hash ?? 0:X16} colors={description.Rendering.ColorCount} " +
                    $"{MetalNative.DescribeError(error)}");
            }

            if (shareKey is not null && !_states.Register(shareKey, state, default, out var winner))
            {
                // Another thread made the same state first.
                MetalNative.SendVoid(state, SelRelease);
                state = winner;
            }
        }
        finally
        {
            MetalNative.SendVoid(descriptor, SelRelease);
        }

        return state;
    }

    // Render pipeline states by what their descriptor holds. Pipelines the shared cache keeps
    // apart for state Metal sets at encode time (cull, winding, depth clip, exact topology,
    // restart, stencil operations) still build the same descriptor, and so share one state.
    private readonly MetalSharedHandleTable<MetalPipelineStateKey> _states = new();

    // The state of a rectangle-list pipeline for an index type of its control points
    // (0 = non-indexed, 1 = 16-bit, 2 = 32-bit), made by the first draw that needs it. On the
    // GPU worker, the only thread that draws with or releases a pipeline's states.
    private nint RectangleState(MetalPipelineEntry entry, int variant)
    {
        var states = entry.RectangleStates!;
        if (states[variant] != 0)
        {
            return states[variant];
        }

        var description = entry.GraphicsDescription ??
            throw Fatal($"A rectangle-list pipeline has no description to build its indexed state from: pipeline={entry.Id}.");
        var vertexModule = RequireModule(description.VertexProgram);
        var pixelModule = description.PixelStage is not null && description.PixelProgram.IsValid
            ? RequireModule(description.PixelProgram)
            : null;
        var state = CreateRenderPipelineState(
            description, vertexModule, pixelModule, entry.Mesh, true,
            variant == 2 ? Mtl.ControlPointIndexTypeUInt32 : Mtl.ControlPointIndexTypeUInt16, out _);
        states[variant] = state;
        return state;
    }

    // One owner of a shared pipeline state lets go; the state goes with the last.
    private void ReleaseState(nint state)
    {
        if (_states.Release(state))
        {
            MetalNative.SendVoid(state, SelRelease);
        }
    }

    /// <summary>
    /// Everything CreateRenderPipelineState puts into a render descriptor as bytes: the source
    /// of the vertex and pixel functions, the sample count, the topology class when the program
    /// is layered, alpha to coverage, every attachment's format, write mask and effective
    /// blend, the depth and stencil formats, and the vertex layout. Keep in step with it: a
    /// field the descriptor reads that is not here would hand a pipeline another's state.
    /// </summary>
    internal static MetalPipelineStateKey PipelineStateKeyOf(
        GraphicsPipelineDescription description,
        UInt128 vertexSource,
        UInt128? pixelSource,
        bool layered,
        bool rectangleList,
        nuint controlPointIndexType,
        bool lodStatistics = false)
    {
        var parameters = description.StaticParameters;
        var key = new MetalPipelineStateKey.Builder()
            .Add(vertexSource)
            .Add(pixelSource is not null)
            .Add(pixelSource ?? default)
            .Add(lodStatistics)
            .Add(rectangleList)
            .Add(rectangleList ? controlPointIndexType : 0)
            .Add(Math.Max(parameters.Samples, 1))
            .Add(layered ? MetalV2Formats.TopologyClass(parameters.Topology) : ulong.MaxValue)
            .Add(parameters.AlphaToCoverageEnable)
            .Add(description.Rendering.ColorCount);
        for (var index = 0; index < description.Rendering.ColorCount; index++)
        {
            var format = MetalV2Formats.PixelFormat(description.Rendering.ColorFormats[index]);
            key.Add(format).Add((ulong)MetalV2Formats.ColorWriteMask(parameters.GetColorMask(index)));
            var blends = parameters.GetBlendEnable(index) && !parameters.GetBlendBypass(index) && !IsIntegerFormat(format);
            key.Add(blends);
            if (!blends)
            {
                continue;
            }

            var separate = parameters.GetSeparateAlphaBlend(index);
            key.Add((ulong)MetalV2Formats.BlendFactor(parameters.GetColorSourceBlend(index)))
                .Add((ulong)MetalV2Formats.BlendFactor(parameters.GetColorDestinationBlend(index)))
                .Add((ulong)MetalV2Formats.BlendOperation(parameters.GetColorBlendFunction(index)))
                .Add((ulong)MetalV2Formats.BlendFactor(
                    separate ? parameters.GetAlphaSourceBlend(index) : parameters.GetColorSourceBlend(index)))
                .Add((ulong)MetalV2Formats.BlendFactor(
                    separate ? parameters.GetAlphaDestinationBlend(index) : parameters.GetColorDestinationBlend(index)))
                .Add((ulong)MetalV2Formats.BlendOperation(
                    separate ? parameters.GetAlphaBlendFunction(index) : parameters.GetColorBlendFunction(index)));
        }

        var depthFormat = description.Rendering.DepthFormat != GpuFormat.Undefined
            ? MetalV2Formats.PixelFormat(description.Rendering.DepthFormat)
            : MtlFormat.Invalid;
        var stencilFormat = description.Rendering.StencilFormat != GpuFormat.Undefined
            ? MetalV2Formats.PixelFormat(description.Rendering.StencilFormat)
            : MtlFormat.Invalid;
        key.Add(depthFormat).Add(stencilFormat != MtlFormat.Invalid && MetalV2Formats.HasStencil(stencilFormat) ? stencilFormat : MtlFormat.Invalid);

        // The vertex descriptor is only set with attributes (CreateVertexDescriptor).
        var vertexInput = description.VertexInput;
        key.Add((ulong)vertexInput.AttributeCount);
        if (vertexInput.AttributeCount != 0)
        {
            for (var binding = 0; binding < vertexInput.BindingCount && binding < MaxVertexBufferSlots; binding++)
            {
                key.Add(vertexInput.Bindings[binding].Stride).Add(vertexInput.Bindings[binding].Instance);
            }

            for (var index = 0; index < vertexInput.AttributeCount; index++)
            {
                var binding = vertexInput.Attributes[index].Binding;
                if (binding >= MaxVertexBufferSlots)
                {
                    continue;
                }

                key.Add((ulong)index).Add((ulong)VertexFormatOf(description, index, out _))
                    .Add(vertexInput.Attributes[index].Offset).Add(binding);
            }
        }

        return key.Build();
    }

    private static readonly nint SelSetBinaryArchives = MetalNative.Selector("setBinaryArchives:");

    private nint NewRenderPipelineState(nint descriptor, bool mesh, nuint options, ref nint error) =>
        MetalNative.SendNewMeshPipeline(
            _device,
            MetalNative.Selector(mesh
                ? "newRenderPipelineStateWithMeshDescriptor:options:reflection:error:"
                : "newRenderPipelineStateWithDescriptor:options:reflection:error:"),
            descriptor,
            options,
            0,
            ref error);

    private static bool IsIntegerFormat(ulong format) => format is
        MtlFormat.R8Uint or MtlFormat.R8Sint or MtlFormat.Rg8Uint or MtlFormat.Rg8Sint or
        MtlFormat.R16Uint or MtlFormat.R16Sint or MtlFormat.Rg16Uint or MtlFormat.Rg16Sint or
        MtlFormat.R32Uint or MtlFormat.R32Sint or MtlFormat.Rg32Uint or MtlFormat.Rg32Sint or
        MtlFormat.Rgba8Uint or MtlFormat.Rgba8Sint or MtlFormat.Rgba16Uint or MtlFormat.Rgba16Sint or
        MtlFormat.Rgba32Uint or MtlFormat.Rgba32Sint or MtlFormat.Rgb10A2Uint;

    private static uint[] RectangleVertexStrides(PipelineVertexInputState vertexInput)
    {
        var strides = new uint[Math.Min((int)vertexInput.BindingCount, MaxVertexBufferSlots)];
        for (var binding = 0; binding < strides.Length; binding++)
        {
            strides[binding] = vertexInput.Bindings[binding].Instance ? 0 : vertexInput.Bindings[binding].Stride;
        }

        return strides;
    }

    // The Metal format of one vertex attribute, narrowed to the components the program fetches.
    private static MtlVertexFormat VertexFormatOf(GraphicsPipelineDescription description, int index, out uint usedComponents)
    {
        var resource = description.VertexInfo.Attributes[index];
        var compiledComponents = index < description.VertexStage.VertexFetchComponents.Length
            ? description.VertexStage.VertexFetchComponents[index]
            : (byte)0;
        usedComponents = compiledComponents > 0 ? compiledComponents : (uint)resource.RegisterCount;
        return MetalV2Formats.VertexFormat(resource.Descriptor, usedComponents);
    }

    // The vertex descriptor: one attribute per fetch, one layout per guest buffer. A
    // rectangle list steps its per-vertex buffers per patch control point, which is what
    // hands each corner of the post-tessellation function all three guest vertices.
    private nint CreateVertexDescriptor(GraphicsPipelineDescription description, bool rectangleList, out string? layoutReport)
    {
        var report = MetalV2Diagnostics.ReportVertexLayout ? new System.Text.StringBuilder() : null;
        var descriptor = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLVertexDescriptor"), MetalNative.Selector("alloc")),
            MetalNative.Selector("init"));
        if (descriptor == 0)
        {
            throw Fatal("A Metal vertex descriptor could not be created.");
        }

        var attributes = MetalNative.Send(descriptor, MetalNative.Selector("attributes"));
        var layouts = MetalNative.Send(descriptor, MetalNative.Selector("layouts"));
        var vertexInput = description.VertexInput;
        var info = description.VertexInfo;
        for (var binding = 0; binding < vertexInput.BindingCount && binding < MaxVertexBufferSlots; binding++)
        {
            var layout = MetalNative.SendAtIndex(layouts, SelObjectAtIndex, (nuint)(VertexBufferSlotBase + binding));
            var stride = vertexInput.Bindings[binding].Stride;
            // A zero stride means every vertex reads the same record; Metal expresses
            // that as a constant step function with a nonzero stride.
            MetalV2Native.SendVoidValue(layout, MetalNative.Selector("setStride:"), Math.Max(stride, 4u));
            MetalV2Native.SendVoidValue(
                layout,
                MetalNative.Selector("setStepFunction:"),
                (nuint)(stride == 0
                    ? 0
                    : vertexInput.Bindings[binding].Instance ? Mtl.StepFunctionPerInstance
                    : rectangleList ? Mtl.StepFunctionPerPatchControlPoint : Mtl.StepFunctionPerVertex));
            if (stride == 0)
            {
                MetalV2Native.SendVoidValue(layout, MetalNative.Selector("setStepRate:"), 0);
            }

            report?.Append(
                System.Globalization.CultureInfo.InvariantCulture,
                $" buf{VertexBufferSlotBase + binding}:stride={stride}{(vertexInput.Bindings[binding].Instance ? ":instance" : string.Empty)}");
        }

        for (var index = 0; index < vertexInput.AttributeCount; index++)
        {
            var binding = vertexInput.Attributes[index].Binding;
            if (binding >= MaxVertexBufferSlots)
            {
                continue;
            }

            var resource = info.Attributes[index];
            var format = VertexFormatOf(description, index, out var usedComponents);
            var attribute = MetalNative.SendAtIndex(attributes, SelObjectAtIndex, (nuint)index);
            MetalV2Native.SendVoidValue(attribute, MetalNative.Selector("setFormat:"), (nuint)format);
            MetalV2Native.SendVoidValue(
                attribute, MetalNative.Selector("setOffset:"), (nuint)vertexInput.Attributes[index].Offset);
            MetalV2Native.SendVoidValue(
                attribute, MetalNative.Selector("setBufferIndex:"), (nuint)(VertexBufferSlotBase + binding));
            report?.Append(
                System.Globalization.CultureInfo.InvariantCulture,
                $" a{index}:fmt={(int)format}:off={vertexInput.Attributes[index].Offset}" +
                $":buf={VertexBufferSlotBase + binding}:guest={resource.Descriptor.Format}:used={usedComponents}");
        }

        layoutReport = report?.ToString();
        return descriptor;
    }

    // Compute pipeline states are built off the command-stream thread, which stops a long
    // compile blocking every queue: the dispatch suspends and runs again once it is done.
    // Measured on Demon's Souls it removed the compile from the worker (pipeline_ms fell to
    // four milliseconds a window), and a retry no longer repeats the program preparation
    // and resource materialisation in front of the pipeline request (ComputeProgramRetry).
    // SHARPEMU_METAL_V2_ASYNC_PIPELINES=0 builds them on the worker again.
    private static readonly bool AsynchronousPipelines =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_ASYNC_PIPELINES") != "0";

    private static bool SynchronousPipelines => !AsynchronousPipelines;

    private MetalPipelineCompiler? _compiler;

    private MetalPipelineCompiler Compiler => _compiler ??= new MetalPipelineCompiler(
        () => _commandStream?.NotifyPipelineReady(),
        Math.Max(2, Environment.ProcessorCount / 2));

    /// <summary>Compiles that have been started, finished, shared and their time.</summary>
    public (long Started, long Finished, long Deduped, double Milliseconds) TakeCompilerCounters() =>
        _compiler?.TakeCounters() ?? (0, 0, 0, 0);

    public int PipelinesCompiling => _compiler?.InFlight ?? 0;

    public bool SupportsPipelineSuspend => !SynchronousPipelines;

    public void NotePipelineNotReady()
    {
        _pipelineNotReady = true;
        // The dispatch runs again when the pipeline is ready, and asks for its copy again.
        DisarmPendingCopy();
    }

    private bool _pipelineNotReady;

    bool SharpEmu.Libs.Gpu.GpuCommands.ICommandStreamHost.ConsumePipelineNotReady()
    {
        var pending = _pipelineNotReady;
        _pipelineNotReady = false;
        return pending;
    }

    /// <summary>
    /// The pipeline state for a dispatch, or false while the driver is still building
    /// it on the compile pool. The caller suspends the packet and runs it again when
    /// the compile reports in.
    /// </summary>
    public bool TryCreateComputePipeline(ComputePipelineDescription description, out PipelineHandle handle)
    {
        if (SynchronousPipelines)
        {
            handle = CreateComputePipeline(description);
            return true;
        }

        handle = default;
        var program = description.Program;
        var hash = description.Stage.Hash;
        // The library compile is waited for on the compiler thread too, not on the worker.
        if (!Compiler.TryTake(
                program.Id,
                () =>
                {
                    var module = RequireModule(program, onWorker: false);
                    return new MetalPipelineCompiler.Result(BuildComputePipelineState(module.Function, hash, module.ArchiveKey));
                },
                out var compiled,
                $"cs=0x{hash:X16}"))
        {
            return false;
        }

        handle = RegisterComputePipeline(description, LookupModule(program), compiled!.State);
        return true;
    }

    // Runs on the compile pool: MTLDevice is thread safe, and nothing here touches
    // the host's own state.
    private nint BuildComputePipelineState(nint function, ulong hash, UInt128 archiveKey)
    {
        nint error = 0;
        using var runtime = MetalCompileScheduler.Shared.Runtime();
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        var state = NewComputePipelineState(function, archiveKey, ref error);
        Interlocked.Add(ref _pipelineTicks, System.Diagnostics.Stopwatch.GetTimestamp() - started);
        if (state == 0)
        {
            throw Fatal(
                $"A Metal compute pipeline state could not be created: hash=0x{hash:X16} " +
                $"{MetalNative.DescribeError(error)}");
        }

        return state;
    }

    // A compute library's archive file holds its pipeline's binary too: that is asked
    // first, and a pipeline compiled here is stored with the library.
    private nint NewComputePipelineState(nint function, UInt128 archiveKey, ref nint error)
    {
        var archive = MetalBinaryArchive.For(_device);
        var stored = archive?.OpenLibraryArchive(archiveKey) ?? 0;
        if (stored != 0)
        {
            var pool = MetalNative.objc_autoreleasePoolPush();
            nint state = 0;
            try
            {
                var descriptor = MetalNative.Send(
                    MetalNative.Send(MetalNative.Class("MTLComputePipelineDescriptor"), MetalNative.Selector("alloc")),
                    MetalNative.Selector("init"));
                if (descriptor != 0)
                {
                    MetalNative.SendVoid(descriptor, MetalNative.Selector("setComputeFunction:"), function);
                    MetalNative.SendVoid(
                        descriptor,
                        SelSetBinaryArchives,
                        MetalNative.Send(MetalNative.Class("NSArray"), MetalNative.Selector("arrayWithObject:"), stored));
                    nint missError = 0;
                    state = MetalNative.SendNewMeshPipeline(
                        _device,
                        MetalNative.Selector("newComputePipelineStateWithDescriptor:options:reflection:error:"),
                        descriptor,
                        MetalBinaryArchive.FailOnBinaryArchiveMiss,
                        0,
                        ref missError);
                    MetalNative.SendVoid(descriptor, SelRelease);
                }
            }
            finally
            {
                MetalNative.SendVoid(stored, SelRelease);
                MetalNative.objc_autoreleasePoolPop(pool);
            }

            archive!.NotePipelineLookup(state != 0);
            if (state != 0)
            {
                return state;
            }
        }

        var compiled = MetalNative.Send(
            _device,
            MetalNative.Selector("newComputePipelineStateWithFunction:error:"),
            function,
            ref error);
        if (compiled != 0)
        {
            archive?.CaptureCompute(archiveKey, function);
        }

        return compiled;
    }

    public PipelineHandle CreateComputePipeline(ComputePipelineDescription description)
    {
        Interlocked.Increment(ref _pipelineWaitsBlocking);
        var module = RequireModule(description.Program);
        nint error = 0;
        // The driver turns the function into a GPU binary here, on the worker thread,
        // and the guest's dispatch packet waits for it. Measured on Demon's Souls'
        // load phase this is tens to hundreds of milliseconds per new shader.
        using var runtime = MetalCompileScheduler.Shared.Runtime();
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        var state = NewComputePipelineState(module.Function, module.ArchiveKey, ref error);
        Interlocked.Add(ref _pipelineTicks, System.Diagnostics.Stopwatch.GetTimestamp() - started);
        if (state == 0)
        {
            throw Fatal(
                $"A Metal compute pipeline state could not be created: hash=0x{description.Stage.Hash:X16} " +
                $"{MetalNative.DescribeError(error)}");
        }

        return RegisterComputePipeline(description, module, state);
    }

    // The pipeline entry for a finished compute state. Runs on the command stream, so
    // the entry table and the identifier counter stay single threaded.
    private PipelineHandle RegisterComputePipeline(
        ComputePipelineDescription description,
        MetalShaderModule module,
        nint state)
    {
        Interlocked.Increment(ref _pipelineCreations);
        var threads = new MtlSize
        {
            Width = Math.Max(module.Shader.ThreadgroupSizeX, 1),
            Height = Math.Max(module.Shader.ThreadgroupSizeY, 1),
            Depth = Math.Max(module.Shader.ThreadgroupSizeZ, 1),
        };
        if (threads.Width * threads.Height * threads.Depth == 1 && description.Input.ThreadsX != 0)
        {
            threads = new MtlSize
            {
                Width = Math.Max(description.Input.ThreadsX, 1),
                Height = Math.Max(description.Input.ThreadsY, 1),
                Depth = Math.Max(description.Input.ThreadsZ, 1),
            };
        }

        var entry = new MetalPipelineEntry
        {
            Id = ++_nextPipelineId,
            State = state,
            Graphics = false,
            PersistentThreads = module.Shader.PersistentThreads,
            ThreadsPerThreadgroup = threads,
            Profile = ProfileLabelFor(module, null, null),
            VertexHash = module.Hash,
            ComputeDescription = description,
            LastUsedTicks = System.Diagnostics.Stopwatch.GetTimestamp(),
        };
        _pipelineEntries[entry.Id] = entry;
        Interlocked.Increment(ref _residentPipelines);
        return new PipelineHandle(entry.Id, 0, false);
    }

    // The depth/stencil state object: Metal keeps the compare functions, the write
    // enable and the stencil masks in one immutable object, while the executor hands
    // the masks over as dynamic state, so the object is cached on the pair.
    private nint GetDepthStencilState(MetalPipelineEntry entry, in DynamicDrawState state)
    {
        var parameters = entry.StaticParameters;
        var stencilEnabled = state.StencilTestEnabled && entry.HasStencil && parameters is not null &&
            !MetalV2Diagnostics.IgnoreFragmentTests;
        var key = MixDepthStencilKey(entry, in state, stencilEnabled);
        if (_depthStencilStates.TryGetValue(key, out var found))
        {
            return found;
        }

        var descriptor = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLDepthStencilDescriptor"), MetalNative.Selector("alloc")),
            MetalNative.Selector("init"));
        if (descriptor == 0)
        {
            throw Fatal("A Metal depth/stencil descriptor could not be created.");
        }

        try
        {
            var depthCompare = entry.HasDepth && state.DepthTestEnabled && !MetalV2Diagnostics.IgnoreFragmentTests
                ? MetalV2Formats.CompareFunction(state.DepthCompare)
                : MetalV2Formats.CompareFunction(GpuCompareOp.Always);
            MetalV2Native.SendVoidValue(descriptor, MetalNative.Selector("setDepthCompareFunction:"), depthCompare);
            MetalNative.SendVoidBool(
                descriptor,
                MetalNative.Selector("setDepthWriteEnabled:"),
                entry.HasDepth && state.DepthWriteEnabled);
            if (stencilEnabled)
            {
                ConfigureStencil(
                    descriptor,
                    MetalNative.Selector("setFrontFaceStencil:"),
                    parameters!.StencilFront,
                    state.FrontStencil);
                ConfigureStencil(
                    descriptor,
                    MetalNative.Selector("setBackFaceStencil:"),
                    parameters.StencilBack,
                    state.BackStencil);
            }

            var created = MetalNative.Send(
                _device, MetalNative.Selector("newDepthStencilStateWithDescriptor:"), descriptor);
            if (created == 0)
            {
                throw Fatal("A Metal depth/stencil state could not be created.");
            }

            _depthStencilStates[key] = created;
            return created;
        }
        finally
        {
            MetalNative.SendVoid(descriptor, SelRelease);
        }
    }

    private void ConfigureStencil(
        nint descriptor,
        nint selector,
        in StencilOperations operations,
        in StencilMasks masks)
    {
        var stencil = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLStencilDescriptor"), MetalNative.Selector("alloc")),
            MetalNative.Selector("init"));
        if (stencil == 0)
        {
            throw Fatal("A Metal stencil descriptor could not be created.");
        }

        try
        {
            MetalV2Native.SendVoidValue(
                stencil,
                MetalNative.Selector("setStencilCompareFunction:"),
                MetalV2Formats.CompareFunction(operations.Compare));
            MetalV2Native.SendVoidValue(
                stencil,
                MetalNative.Selector("setStencilFailureOperation:"),
                MetalV2Formats.StencilOperation(operations.FailOperation));
            MetalV2Native.SendVoidValue(
                stencil,
                MetalNative.Selector("setDepthStencilPassOperation:"),
                MetalV2Formats.StencilOperation(operations.PassOperation));
            MetalV2Native.SendVoidValue(
                stencil,
                MetalNative.Selector("setDepthFailureOperation:"),
                MetalV2Formats.StencilOperation(operations.DepthFailOperation));
            MetalV2Native.SendVoidUInt(stencil, MetalNative.Selector("setReadMask:"), masks.CompareMask);
            MetalV2Native.SendVoidUInt(stencil, MetalNative.Selector("setWriteMask:"), masks.WriteMask);
            MetalNative.SendVoid(descriptor, selector, stencil);
        }
        finally
        {
            MetalNative.SendVoid(stencil, SelRelease);
        }
    }

    private static ulong MixDepthStencilKey(MetalPipelineEntry entry, in DynamicDrawState state, bool stencilEnabled)
    {
        var hash = 14695981039346656037UL;
        void Mix(ulong value) => hash = (hash ^ value) * 1099511628211UL;
        Mix(entry.HasDepth ? 1UL : 0UL);
        Mix(state.DepthTestEnabled ? 1UL : 0UL);
        Mix(state.DepthWriteEnabled ? 1UL : 0UL);
        Mix((ulong)state.DepthCompare);
        Mix(stencilEnabled ? 1UL : 0UL);
        if (stencilEnabled && entry.StaticParameters is { } parameters)
        {
            Mix((ulong)parameters.StencilFront.Compare);
            Mix((ulong)parameters.StencilFront.FailOperation);
            Mix((ulong)parameters.StencilFront.PassOperation);
            Mix((ulong)parameters.StencilFront.DepthFailOperation);
            Mix((ulong)parameters.StencilBack.Compare);
            Mix((ulong)parameters.StencilBack.FailOperation);
            Mix((ulong)parameters.StencilBack.PassOperation);
            Mix((ulong)parameters.StencilBack.DepthFailOperation);
            Mix(state.FrontStencil.CompareMask | ((ulong)state.FrontStencil.WriteMask << 32));
            Mix(state.BackStencil.CompareMask | ((ulong)state.BackStencil.WriteMask << 32));
        }

        return hash;
    }

    public bool TryResolveColorOutput(
        uint dataFormat,
        uint numberType,
        uint componentSwap,
        out Gen5PixelOutputKind outputKind,
        out Gen5ColorComponentMapping componentMapping)
    {
        if (MetalGuestFormats.TryDecodeRenderTargetFormat(dataFormat, numberType, componentSwap, out var format))
        {
            outputKind = format.OutputKind;
            componentMapping = format.ExportMapping;
            return true;
        }

        outputKind = default;
        componentMapping = default;
        return false;
    }

    public bool TryReadGuestWord(ulong address, out uint word)
    {
        _wordReads++;
        if (_labelOverlay.Count != 0 && TryReadPendingLabelWord(address, out word))
        {
            // A label the GPU has yet to land, read as written (MetalRenderHost.LabelOverlay).
            return true;
        }

        if (_wordLineThread == Environment.CurrentManagedThreadId && TryReadWordFromLine(address, out word))
        {
            return true;
        }

        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        if (!TryReadGuest(address, bytes))
        {
            word = 0;
            return false;
        }

        word = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes);
        return true;
    }

    public bool TryReadCleanGuestWord(ulong address, out uint word)
    {
        word = 0;
        if (_imageCache.HasGpuModifiedImageBytes(address, sizeof(uint)))
        {
            return false;
        }

        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        if (!_backing.TryReadBacking(address, bytes))
        {
            return false;
        }

        ApplyPendingCommandProcessorBytes(address, bytes);
        ApplyPendingLabelBytes(address, bytes);
        word = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes);
        return true;
    }

    public bool TryReadResidentGuestBytes(ulong address, Span<byte> destination, bool clean)
    {
        if (destination.IsEmpty)
        {
            return true;
        }

        if (_imageCache.HasGpuModifiedImageBytes(address, (ulong)destination.Length))
        {
            return false;
        }

        _ = clean;
        if (!_backing.TryReadBacking(address, destination))
        {
            return false;
        }

        ApplyPendingCommandProcessorBytes(address, destination);
        ApplyPendingLabelBytes(address, destination);
        return true;
    }

    // Ranges closer than this share one ownership and overlay test in TryCompareResidentGuestBytes.
    private const ulong CompareClusterGapBytes = 16 * 1024;

    // Labels elsewhere do not affect these bytes. Keep the ordinary reader for any
    // cluster whose pages may hold one, including page-filter collisions.
    internal bool CompareLabelRangesEnabled { get; set; } =
        SharpEmu.ShaderCompiler.PerfSwitches.Enabled("SHARPEMU_RESOURCE_COMPARE_LABEL_RANGES");
    private long _residentCompareCalls;
    private long _residentCompareHits;
    private long _residentCompareLabelSafe;
    private long _residentCompareLabelRefused;
    private long _residentCompareOwnershipRefused;
    private long _residentCompareAliasRefused;
    private long _residentCompareChanged;
    private long _residentCompareInvalid;
    private long _residentCompareReleased;

    private string TakeResidentCompareCounters() =>
        $" rescmp_calls={Interlocked.Exchange(ref _residentCompareCalls, 0)}" +
        $" rescmp_hits={Interlocked.Exchange(ref _residentCompareHits, 0)}" +
        $" rescmp_label_safe={Interlocked.Exchange(ref _residentCompareLabelSafe, 0)}" +
        $" rescmp_label_refused={Interlocked.Exchange(ref _residentCompareLabelRefused, 0)}" +
        $" rescmp_owner_refused={Interlocked.Exchange(ref _residentCompareOwnershipRefused, 0)}" +
        $" rescmp_alias_refused={Interlocked.Exchange(ref _residentCompareAliasRefused, 0)}" +
        $" rescmp_changed={Interlocked.Exchange(ref _residentCompareChanged, 0)}" +
        $" rescmp_invalid={Interlocked.Exchange(ref _residentCompareInvalid, 0)}" +
        $" rescmp_released={Interlocked.Exchange(ref _residentCompareReleased, 0)}";

    // TryReadResidentGuestBytes for each range, compared where it lies. The same refusals apply
    // (a GPU-modified image over a byte, a pending command-processor write), tested once for each
    // run of nearby ranges rather than for each range, and a refusal answers false for the
    // caller to read the ranges itself. So do the cases this does not model: possibly overlapping
    // label bytes (reading them retires entries), a released single-copy chunk (reading restores it),
    // a range that is not one backed view, and an alias access the backing will not admit.
    public unsafe bool TryCompareResidentGuestBytes(
        ReadOnlySpan<ulong> addresses, ReadOnlySpan<int> lengths, ReadOnlySpan<int> offsets, ReadOnlySpan<byte> recorded)
    {
        _residentCompareCalls++;
        var count = addresses.Length;
        if (count == 0)
        {
            return true;
        }

        if (lengths.Length != count || offsets.Length != count)
        {
            _residentCompareInvalid++;
            return false;
        }
        if (_backing.SingleCopy is { ReleasedChunks: > 0 })
        {
            _residentCompareReleased++;
            return false;
        }

        var start = 0;
        while (start < count)
        {
            var low = addresses[start];
            var high = low;
            var end = start;
            while (end < count)
            {
                var address = addresses[end];
                var length = lengths[end];
                var offset = offsets[end];
                if (length <= 0 || offset < 0 || offset > recorded.Length - length || ulong.MaxValue - address < (ulong)length)
                {
                    _residentCompareInvalid++;
                    return false;
                }

                if (end > start && (address < low || (address > high && address - high > CompareClusterGapBytes)))
                {
                    break;
                }

                high = Math.Max(high, address + (ulong)length);
                end++;
            }

            var size = high - low;
            if (_labelOverlay.Count != 0)
            {
                if (!CompareLabelRangesEnabled || _labelOverlay.MayOverlap(low, size))
                {
                    _residentCompareLabelRefused++;
                    return false;
                }
                _residentCompareLabelSafe++;
            }
            if (_imageCache.HasGpuModifiedImageBytes(low, size) ||
                (_cpOverlay.Count != 0 &&
                 _cpOverlay.Coverage(low, size, _completedSerialReader ??= () => CompletedSerial) != MetalOverlayCoverage.None))
            {
                _residentCompareOwnershipRefused++;
                return false;
            }

            start = end;
        }

        if (!_backing.TryEnterBackingAliasAccess())
        {
            _residentCompareAliasRefused++;
            return false;
        }

        try
        {
            for (var index = 0; index < count; index++)
            {
                var length = lengths[index];
                if (!_backing.TryResolveBackingAlias(addresses[index], (ulong)length, out var alias))
                {
                    _residentCompareAliasRefused++;
                    return false;
                }
                if (!new ReadOnlySpan<byte>((void*)alias, length).SequenceEqual(recorded.Slice(offsets[index], length)))
                {
                    _residentCompareChanged++;
                    return false;
                }
            }
        }
        finally
        {
            _backing.ExitBackingAliasAccess();
        }
        _residentCompareHits++;
        return true;
    }

    private Func<ulong, bool>? _dataWritePending;

    // A data-only table word the GPU may still be writing: the shader is re-planned to load
    // it on the device (DeviceDataLoads). Asked without waiting. A pending store with no
    // evaluated range covers every address, so it names no word in particular and teaches
    // nothing; only a recorded range that reaches this dword does.
    public Func<ulong, bool>? DataWritePending =>
        MetalV2Diagnostics.DataLoadsOnDevice ? _dataWritePending ??= IsDataWordPending : null;

    private bool IsDataWordPending(ulong address) =>
        _gpuWrites.MayCover(address, sizeof(uint)) &&
        _gpuWrites.PendingSerial(address, sizeof(uint), CompletedSerial) != 0 &&
        _gpuWrites.LastMatch.Origin != MetalGpuWriteOrigin.Unbounded;

    // Resource tables are read straight from the backing, so a WRITE_DATA that patched one
    // and has yet to land would otherwise be read as the table it replaced.
    private void ApplyPendingCommandProcessorBytes(ulong address, Span<byte> destination)
    {
        if (_cpOverlay.Count != 0 &&
            _cpOverlay.Coverage(address, (ulong)destination.Length, _completedSerialReader ??= () => CompletedSerial) !=
            MetalOverlayCoverage.None)
        {
            _cpOverlay.Apply(address, destination);
            Interlocked.Increment(ref _overlayReads);
        }
    }

    private void DisposePipelines()
    {
        // Let whatever is still compiling finish before the device goes away.
        _compiler?.Dispose();
        _compiler = null;
        // What this run added to the title's archive is written before its functions go.
        MetalBinaryArchive.For(_device)?.Flush(wait: true, TimeSpan.FromSeconds(10));
        foreach (var entry in _pipelineEntries.Values)
        {
            if (entry.State != 0)
            {
                ReleaseState(entry.State);
            }

            // The first rectangle state is entry.State itself.
            for (var index = 1; index < (entry.RectangleStates?.Length ?? 0); index++)
            {
                if (entry.RectangleStates![index] != 0)
                {
                    ReleaseState(entry.RectangleStates[index]);
                }
            }
        }

        _pipelineEntries.Clear();
        foreach (var state in _depthStencilStates.Values)
        {
            MetalNative.SendVoid(state, SelRelease);
        }

        _depthStencilStates.Clear();
        foreach (var module in _modules)
        {
            try
            {
                module.Compiling?.Wait();
            }
            catch (AggregateException)
            {
                // A compile that failed left nothing to release.
            }

            if (module.Function != 0)
            {
                MetalNative.SendVoid(module.Function, SelRelease);
            }

            if (module.LodFunction != 0)
            {
                MetalNative.SendVoid(module.LodFunction, SelRelease);
            }

            if (module.Library != 0)
            {
                ReleaseLibrary(module.Library);
            }

            if (module.RectangleFunction != 0)
            {
                MetalNative.SendVoid(module.RectangleFunction, SelRelease);
                ReleaseLibrary(module.RectangleLibrary);
            }
        }

        _modules.Clear();
        foreach (var sampler in _samplers.Values)
        {
            MetalNative.SendVoid(sampler.Handle, SelRelease);
        }

        _samplers.Clear();
    }
}
