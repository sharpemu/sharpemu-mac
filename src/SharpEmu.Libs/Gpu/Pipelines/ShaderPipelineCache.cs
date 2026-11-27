// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.IO.Hashing;
using System.Runtime.InteropServices;
using SharpEmu.HLE;
using SharpEmu.HLE.Diagnostics;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.VideoOut;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Libs.Gpu.Pipelines;

// The pipeline provider behind the executor: programs by identity, pipelines by their full static state.
internal sealed partial class ShaderPipelineCache : IShaderPipelineProvider
{
    internal bool SupportsParallelResolution => _registry.SupportsParallelResolution;
    internal long FrontendImageVersion => _registry.FrontendImageVersion;

    private const uint VertexUserDataBase = 8;
    private const uint MaxPixelInputs = 32;

    // The geometry stage programs its LDS allocation in 128-dword granules, like compute.
    private const uint LocalDataShareGranuleDwords = 128;

    private readonly CpuContext _context;
    private readonly IGuestGpuBackend _compiler;
    private readonly IShaderPipelineHost _host;
    private readonly ShaderHeaderRegistry _registry;
    private readonly ShaderProgramCache _programs;
    private readonly VertexTableMetadataCache _vertexTables = new();
    // The content hashes of the shaders that declare none, so a draw does not hash their code again.
    private readonly ShaderIdentityMemo _identities = new();
    // The last user data array of each graphics stage, kept under _resolveGate.
    private uint[]? _vertexUserData;
    private uint[]? _pixelUserData;
    private uint[]? _computeUserData;
    private uint[]? _hullUserData;
    private readonly List<Gen5PixelOutputBinding> _boundOutputs = new(ContextRegisters.ColorTargetCount);
    // What the last draw resolved, shared by a draw that resolves equal ones; kept under _resolveGate.
    // Nothing writes a resolved draw's arrays.
    private Gen5PixelOutputBinding[] _lastBoundOutputs = [];
    private readonly PixelInputArrays _pixelInputArrays = new();
    private readonly System.Collections.Concurrent.ConcurrentDictionary<GraphicsPipelineKey, PipelineHandle> _graphicsPipelines = new();
    // The pipelines the host is still compiling (IShaderPipelineHost.TryCreateGraphicsPipeline), each with the key and the
    // description that started its compile; under _gate.
    private readonly Dictionary<GraphicsPipelineKey, (GraphicsPipelineKey Key, GraphicsPipelineDescription Description)> _pendingGraphicsPipelines = new();
    private readonly System.Collections.Concurrent.ConcurrentDictionary<ComputePipelineKey, PipelineHandle> _computePipelines = new();
    private readonly Dictionary<Gen5ShaderProgram, bool> _rayTracingComputePrograms = new();
    private readonly object _gate = new();
    // The options of each stage's lookup, refilled under _gate (StageCompileOptions.Detached).
    private readonly StageCompileOptions _vertexOptions = StageCompileOptions.CreateReusable();
    private readonly StageCompileOptions _pixelOptions = StageCompileOptions.CreateReusable();
    private readonly StageCompileOptions _computeOptions = StageCompileOptions.CreateReusable();
    private readonly StageCompileOptions _hullOptions = StageCompileOptions.CreateReusable();
    // The pipeline maps are concurrent: a lookup takes no lock, so a cached draw does not wait for the
    // program preparation a resolving thread holds _gate for, and an addition holds _gate.
    private readonly bool _strictShaders = Environment.GetEnvironmentVariable("SHARPEMU_STRICT_COMPUTE") != "0";
    private readonly bool _skipRayTracing = Environment.GetEnvironmentVariable("SHARPEMU_SKIP_RT") == "1";
    private readonly HashSet<(ShaderStage Stage, ulong Hash, uint CodeSize)> _reportedShaderSkips = [];
    private readonly HashSet<(uint Stages, uint PrimitiveType, uint MaxOutput, uint MaxVerticesOut)> _reportedMeshSkips = [];

    public ShaderPipelineCache(CpuContext context, IShaderPipelineHost host, IGuestGpuBackend compiler, ShaderHeaderRegistry registry)
    {
        _context = context;
        _compiler = compiler;
        _host = host;
        _registry = registry;
        _programs = new ShaderProgramCache(context, compiler, host) { DeferLinking = true };
        _programs.TryComputePipeline = (program, stage, input) =>
        {
            input.Stage = stage;
            return TryCreateComputePipeline(input, program, out _);
        };
        CreateShards(ResolveShards.Default);
    }

    public bool SupportsPrimitiveShaderMeshPipelines => _compiler.SupportsPrimitiveShaderMeshPipelines;

    // The geometry-engine state one NGG subgroup runs under, for a draw the backend will
    // compile as the merged ES/GS workgroup stage. Null keeps the per-vertex path, which is
    // what every backend without mesh pipelines gets.
    private Gen5NggStageInfo? PrimitiveShaderStageInfo(
        VertexStageRegisters vertex,
        ShaderInterfaceRegisters shaderInterface,
        uint shaderStages) =>
        PrimitiveShaderStageInfo(SupportsPrimitiveShaderMeshPipelines, vertex, shaderInterface, shaderStages);

    internal static Gen5NggStageInfo? PrimitiveShaderStageInfo(
        bool supportsPrimitiveShaderMeshPipelines,
        VertexStageRegisters vertex,
        ShaderInterfaceRegisters shaderInterface,
        uint shaderStages)
    {
        // Exactly the draws the executor admits as mesh draws, or a program would be
        // compiled as the merged ES/GS stage with no mesh pipeline to run it - and an
        // ordinary vertex program that happens to touch LDS would have its workgroup sized
        // from geometry-engine registers the draw never programmed.
        if (!supportsPrimitiveShaderMeshPipelines || !VgtShaderStages.SelectsPrimitiveShaderOnly(shaderStages))
        {
            return null;
        }

        return new Gen5NggStageInfo(
            VerticesPerSubgroup: shaderInterface.ExportVerticesPerSubgroup,
            PrimitivesPerSubgroup: shaderInterface.GeometryPrimitivesPerSubgroup,
            MaxOutputPerSubgroup: shaderInterface.MaxOutputPerSubgroup,
            OutputPrimitiveType: shaderInterface.GeometryOutputPrimitiveType,
            PrimitiveAmplificationFactor: shaderInterface.PrimitiveShaderSubgroupControl,
            LocalDataShareBytes: vertex.GeometryResource2.LocalDataShareSize *
                LocalDataShareGranuleDwords * sizeof(uint),

            // GE_CNTL's group sizes live in the user-config bank, which this lookup does not
            // see; they are per-draw launch state and travel in the mesh draw-info block.
            VertexGroupSize: 0,
            PrimitiveGroupSize: 0)
        {
            IsWave64 = VgtShaderStages.IsGeometryWave64(shaderStages),
            Passthrough = VgtShaderStages.IsPrimitivePassthrough(shaderStages),
            AmplifiedVerticesOut = VgtShaderStages.AmplifiedVerticesOut(shaderStages, shaderInterface),
        };
    }

    // s2..s7 of the merged stage: the launch state the geometry engine loads, never user data.
    private const ulong MergedStageSystemScalars = 0xFCUL;

    // Every merged ES/GS wave starts with SPI_SHADER_USER_DATA_ADDR_LO/HI_GS in s[0:1]. The
    // GS half of a fused program reads its own user data through that pointer, so a fused
    // program compiled as the merged workgroup stage takes the pointer as user data: its
    // user data starts at s0, the pointer fills s[0:1], and s2..s7 stay the system scalars
    // the mesh prologue seeds. Any other program keeps its user data at s8.
    internal ShaderSource WithGeometryUserDataAddress(ShaderSource source, ulong geometryUserDataAddress)
    {
        if (!source.Registered.IsFused || source.UserDataBase != VertexUserDataBase ||
            !ShaderCompileRequest.RequiresPrimitiveShaderWorkgroup(_programs.Decode(source)))
        {
            return source;
        }

        var userData = new uint[VertexUserDataBase + source.UserData.Length];
        userData[0] = (uint)geometryUserDataAddress;
        userData[1] = (uint)(geometryUserDataAddress >> 32);
        source.UserData.CopyTo(userData, (int)VertexUserDataBase);
        return source with
        {
            UserData = userData,
            UserDataBase = 0,
            ExcludedUserDataRegisters = MergedStageSystemScalars,
        };
    }

    public ShaderProgramCache Programs => _programs;

    public bool MeshShadersSupported => _host.MeshShadersSupported;

    public int GraphicsPipelineCount => _graphicsPipelines.Count;

    public int ComputePipelineCount => _computePipelines.Count;

    // The user data of one stage: the count from its resource register, else the registers the guest wrote.
    private static uint[] UserData(UserScalarRegisters registers, uint declaredCount, bool probeWrittenRegisters, ulong shaderAddress, string label, ref uint[]? previous)
    {
        var count = declaredCount;
        if (count == 0 && probeWrittenRegisters)
        {
            count = registers.Count;
        }

        if (count > UserScalarRegisters.Capacity)
        {
            throw GpuFatal.Fatal($"The shader declares more user registers than the bank holds: label={label} shader=0x{shaderAddress:X16} count={count}.");
        }

        // Nothing mutates a source's user data, so a draw whose registers match the last one's shares its array.
        var words = ((ReadOnlySpan<uint>)registers.Values)[..(int)count];
        if (previous is not null && words.SequenceEqual(previous))
        {
            return previous;
        }

        return previous = words.ToArray();
    }

    // The same words as a span of the bank (the native tessellation hull, ShaderPipelineCache.Tessellation).
    private static ReadOnlySpan<uint> UserData(UserScalarRegisters registers, uint declaredCount, bool probeWrittenRegisters, ulong shaderAddress, string label)
    {
        var count = declaredCount;
        if (count == 0 && probeWrittenRegisters)
        {
            count = registers.Count;
        }

        if (count > UserScalarRegisters.Capacity)
        {
            throw GpuFatal.Fatal($"The shader declares more user registers than the bank holds: label={label} shader=0x{shaderAddress:X16} count={count}.");
        }

        return registers.ValueSpan((int)count);
    }

    private ShaderSource PrepareSource(ulong codeAddress, ShaderStage stage, string label, UserScalarRegisters registers, uint declaredCount, bool probeWrittenRegisters, uint userDataBase, ref uint[]? userDataMemo,
        ulong geometryPointer = 0)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramSourceRead);
        // A frontend reads a headerless shader with the draw's user data, so they come first.
        var userData = UserData(registers, declaredCount, probeWrittenRegisters, codeAddress, label, ref userDataMemo);
        var registered = _registry.Require(codeAddress, label, userData);
        var hash = HashOf(registered, codeAddress, label);
        if (registered.Frontend is { } frontend && userDataBase == VertexUserDataBase)
        {
            userDataBase = frontend.VertexUserDataBase;
        }

        if (stage == ShaderStage.Mesh && registered.IsFused)
        {
            // A fused program compiled as a mesh program takes the GS user-data pointer in
            // s[0:1] as user data, as WithGeometryUserDataAddress does for the primitive-shader
            // path: its user data starts at s0 and the registers follow at s8.
            var prefixed = new uint[VertexUserDataBase + userData.Length];
            prefixed[0] = (uint)geometryPointer;
            prefixed[1] = (uint)(geometryPointer >> 32);
            userData.CopyTo(prefixed, (int)VertexUserDataBase);
            return new ShaderSource(registered, hash, prefixed, 0, stage);
        }

        // A source is immutable, so a draw that resolves to the same one as the last draw of its
        // stage shares it instead of allocating another.
        ref var last = ref _lastSources[(int)stage];
        if (last is not null && ReferenceEquals(last.Registered, registered) && last.Hash == hash &&
            ReferenceEquals(last.UserData, userData) && last.UserDataBase == userDataBase && last.Stage == stage &&
            last.ExcludedUserDataRegisters == 0)
        {
            return last;
        }

        return last = new ShaderSource(registered, hash, userData, userDataBase, stage);
    }

    // The last source of each stage, kept under _resolveGate like the user data memos.
    private readonly ShaderSource?[] _lastSources = new ShaderSource?[Enum.GetValues<ShaderStage>().Length];

    // The identity of a shader's code: its code is synchronized first, then the declared hash or the
    // remembered content hash (ShaderIdentityMemo); a frontend's code is hashed by content in its domain.
    private ulong HashOf(RegisteredShader registered, ulong codeAddress, string label)
    {
        Span<(ulong Address, uint SizeBytes)> scratch = stackalloc (ulong, uint)[RegisteredShader.MaxCodeRanges];
        ReadOnlySpan<(ulong Address, uint SizeBytes)> ranges = registered.CodeRangeCount <= RegisteredShader.MaxCodeRanges
            ? scratch[..registered.CopyCodeRanges(scratch)]
            : registered.CodeRanges;
        foreach (var (address, sizeBytes) in ranges)
        {
            _host.SynchronizeShaderCode(address, sizeBytes);
        }

        return registered.Frontend is { } frontend
            ? ShaderIdentity.ComputeContent(_context.Memory, ranges, label) ^ frontend.CacheDomain
            : ShaderIdentity.Compute(_context.Memory, codeAddress, ranges, label, _identities);
    }

    // The identity of a header entry's code (ShaderPipelineCache.Tessellation).
    internal ulong IdentityOf(ShaderHeaderEntry header, ulong codeAddress, string label) => HashOf(header.Shader, codeAddress, label);

    // A source from user data already read (the native tessellation hull).
    private ShaderSource PrepareSource(ulong codeAddress, ShaderStage stage, string label, uint[] userData, uint userDataBase, out ShaderHeaderEntry header,
        ulong geometryPointer = 0)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramSourceRead);
        header = _registry.RequireEntry(codeAddress, label, userData);
        return SourceOf(header, IdentityOf(header, codeAddress, label), stage, userData, userDataBase, geometryPointer);
    }

    private static ShaderSource SourceOf(ShaderHeaderEntry header, ulong hash, ShaderStage stage, uint[] userData, uint userDataBase, ulong geometryPointer)
    {
        if (header.Shader.Frontend is { } frontend && userDataBase == VertexUserDataBase)
            userDataBase = frontend.VertexUserDataBase;
        if (stage == ShaderStage.Mesh && header.Shader.IsFused)
        {
            var withPointer = new uint[VertexUserDataBase + userData.Length];
            userData.CopyTo(withPointer.AsSpan((int)VertexUserDataBase));
            withPointer[0] = (uint)geometryPointer;
            withPointer[1] = (uint)(geometryPointer >> 32);
            return new ShaderSource(header.Shader, hash, withPointer, 0, stage);
        }

        return new ShaderSource(header.Shader, hash, userData, userDataBase, stage);
    }

    // Program resolution runs on the render thread and on a thread resolving draws ahead of it;
    // the program, decode and materialization caches it updates are used under this lock only.
    // A pipeline lookup takes no lock, so it does not wait for a resolution.
    // With resolver shards (ResolveShards), the shared tier only: what every shard's resolution shares (the program,
    // decode and materialization caches, the vertex tables, the user data memos and last sources a full resolution
    // fills, translation and links). A shard's memo hit or words rebuild does not take it; its miss does, inside the
    // shard's own gate (lock order: shard gate, then this; never a shard gate while holding this).
    private readonly object _resolveGate = new();

    // One resolver shard: a part of the resolve memo's keys (ResolveShards.DrawRoute) and the lock its memo is used
    // under. With one shard, its gate is _resolveGate itself and its memo _resolveMemo: everything as before.
    private sealed class ResolveShard
    {
        public required object Gate;
        public GraphicsResolveMemo? Memo;
    }

    private ResolveShard[] _shards = [];

    private void CreateShards(int count)
    {
        var shards = new ResolveShard[count];
        shards[0] = new ResolveShard { Gate = count == 1 ? _resolveGate : new object(), Memo = _resolveMemo };
        for (var index = 1; index < count; index++)
        {
            shards[index] = new ResolveShard { Gate = new object(), Memo = NewMemoLike(_resolveMemo) };
        }

        _shards = shards;
        // A memo hit's revalidation of a cached (not fast) materialization reads the materialization cache, which only
        // the shared tier may use.
        _programs.MaterializationGate = count == 1 ? null : _resolveGate;
    }

    private static GraphicsResolveMemo? NewMemoLike(GraphicsResolveMemo? memo) =>
        memo is null ? null : new GraphicsResolveMemo(memo.Mode, GraphicsResolveMemo.DefaultCapacity)
        {
            Rebuild = memo.Rebuild,
            HoldChurn = memo.HoldChurn,
            AdmitSeen = memo.AdmitSeen,
        };

    // The resolver shards (IShaderPipelineProvider.ResolveShardCount), ResolveShards.Default unless a test or bench sets
    // it before the first resolution.
    public int ResolveShardCount
    {
        get => _shards.Length;
        internal set
        {
            lock (_resolveGate)
            {
                CreateShards(Math.Clamp(value, 1, ResolveShards.Maximum));
            }
        }
    }

    public int ResolveShardOf(RegisterBanks banks, bool compute) =>
        _shards.Length == 1 ? 0
        : ResolveShards.ShardOf(compute ? ResolveShards.ComputeRoute(banks) : ResolveShards.DrawRoute(banks, RenderExecutor.PixelStageActive(banks)),
            _shards.Length);

    // The sharded paths' use of the shared tier and of their own gates, for the prefetch line (TakeShardReport): shared
    // gate takes and wait, shard gate wait (a shard's thread against the worker resolving a job of that shard), draws
    // resolved through a shard's memo, and nested resolutions that went to the shared tier whole.
    private static readonly ThreadCounter _sharedTakes = new();
    private static readonly ThreadCounter _sharedWaitTicks = new();
    private static readonly ThreadCounter _shardWaitTicks = new();
    private static readonly ThreadCounter _shardedDraws = new();
    private static readonly ThreadCounter _nestedShared = new();
    private static int _nestedReports;

    // Depth of sharded draw resolutions on this thread (any cache): a resolution inside one never waits for a shard gate.
    [ThreadStatic]
    private static int t_shardedDepth;

    // How this thread's last resolution was answered, for the prefetch's allocation sample by path (alloc_by_path=).
    internal enum ResolvePath : byte
    {
        Other,          // refused, abandoned, or a path not named here
        DrawHit,        // the shard memo's entry, revalidated
        DrawRebuilt,    // the entry rebuilt from its words revalidation
        DrawFull,       // a full graphics resolution (memo miss)
        ComputeKept,    // the kept compute program
        ComputeRebound, // the kept program rebound to new user data
        ComputeResolve, // a full compute resolution
        Count,
    }

    internal static readonly string[] ResolvePathNames = ["other", "draw_hit", "draw_rebuilt", "draw_full", "compute_kept", "compute_rebound", "compute_resolve"];

    [ThreadStatic]
    internal static ResolvePath t_resolvePath;

    // Why a resolution took the shared tier (shared_by= and shared_wait_by= in TakeShardReport).
    internal enum SharedReason
    {
        DrawNew,        // a draw whose key the shard's memo does not hold
        DrawMiss,       // a draw whose memo entry missed for another reason, or that cannot be kept (uncacheable)
        DrawUnsharded,  // a draw resolved without a shard's memo (memo off)
        Nested,         // a draw resolved inside another resolution on one thread (ResolveNestedShared)
        Verify,         // RESOLVE_MEMO=verify's full resolutions
        ComputeResolve, // a dispatch with no kept program of its shape: resolved in full
        ComputeRebind,  // a kept program's rebinding that the shared tier has to make (rebind_shared_by=)
        ComputeWorker,  // a dispatch resolved where the kernel may be predicted (the worker with predictions on)
        ComputeRetry,   // a dispatch that may suspend on its pipeline (ComputeProgramRetry)
        Other,          // tessellation, mesh and register-form resolutions, frontend fallback compiles
        Count,
    }

    private static readonly string[] SharedReasonNames =
        ["draw_new", "draw_miss", "draw_unsharded", "nested", "verify", "compute_resolve", "compute_rebind", "compute_worker", "compute_retry", "other"];

    private static readonly ThreadCounter[] _sharedTakesBy = Enumerable.Range(0, (int)SharedReason.Count).Select(_ => new ThreadCounter()).ToArray();
    private static readonly ThreadCounter[] _sharedWaitTicksBy = Enumerable.Range(0, (int)SharedReason.Count).Select(_ => new ThreadCounter()).ToArray();

    // Dispatches answered off the shared tier with shards (kept, rebound), and the rebindings it had to make, by why.
    private static readonly ThreadCounter _unsharedComputeKept = new();
    private static readonly ThreadCounter _unsharedComputeRebound = new();
    private static readonly ThreadCounter[] _rebindSharedBy =
        Enumerable.Range(0, (int)ShaderProgramCache.UnsharedRefusal.Count).Select(_ => new ThreadCounter()).ToArray();

    public static string TakeShardReport()
    {
        static string Milliseconds(long ticks) =>
            (ticks * 1000.0 / System.Diagnostics.Stopwatch.Frequency).ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
        var by = new System.Text.StringBuilder();
        var waitBy = new System.Text.StringBuilder();
        for (var index = 0; index < (int)SharedReason.Count; index++)
        {
            var separator = index == 0 ? "" : ",";
            by.Append(separator).Append(SharedReasonNames[index]).Append('=').Append(_sharedTakesBy[index].Take());
            waitBy.Append(separator).Append(SharedReasonNames[index]).Append('=').Append(Milliseconds(_sharedWaitTicksBy[index].Take()));
        }

        var rebind = new System.Text.StringBuilder();
        for (var index = 0; index < (int)ShaderProgramCache.UnsharedRefusal.Count; index++)
        {
            rebind.Append(index == 0 ? "" : ",").Append(ShaderProgramCache.UnsharedRefusalNames[index]).Append('=').Append(_rebindSharedBy[index].Take());
        }

        return FormattableString.Invariant($"shared_takes={_sharedTakes.Take()} shared_wait_ms={Milliseconds(_sharedWaitTicks.Take())} ") +
            FormattableString.Invariant($"shard_wait_ms={Milliseconds(_shardWaitTicks.Take())} sharded_draws={_shardedDraws.Take()} ") +
            FormattableString.Invariant($"nested_shared={_nestedShared.Take()} materialization_shared={ShaderProgramCache.TakeMaterializationGateTakes()} ") +
            FormattableString.Invariant($"shared_by=({by}) shared_wait_by=({waitBy}) ") +
            FormattableString.Invariant($"unshared_compute=(kept={_unsharedComputeKept.Take()},rebound={_unsharedComputeRebound.Take()}) rebind_shared_by=({rebind})");
    }

    // The shared tier's gate; timed and counted by reason with shards (with one shard, exactly the lock it always was).
    private void EnterShared(SharedReason reason)
    {
        if (_shards.Length == 1)
        {
            Monitor.Enter(_resolveGate);
            return;
        }

        _sharedTakes.Increment();
        _sharedTakesBy[(int)reason].Increment();
        if (Monitor.TryEnter(_resolveGate))
        {
            return;
        }

        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        Monitor.Enter(_resolveGate);
        var waited = System.Diagnostics.Stopwatch.GetTimestamp() - started;
        _sharedWaitTicks.Add(waited);
        _sharedWaitTicksBy[(int)reason].Add(waited);
    }

    // EnterShared for a block (using): the forms that held the gate with a lock statement.
    private SharedScope EnterSharedScope(SharedReason reason)
    {
        EnterShared(reason);
        return new SharedScope(_resolveGate);
    }

    private readonly ref struct SharedScope(object gate)
    {
        public void Dispose() => Monitor.Exit(gate);
    }

    // A full resolution under the shared tier (with one shard the caller already holds it: re-entered).
    private GraphicsPrograms ResolveFullShared(RegisterBanks banks, ReadOnlySpan<ColorComponentMap> targetExportMapping, in KeptGraphicsFlags flags,
        SharedReason reason)
    {
        EnterShared(reason);
        try
        {
            return ResolveFull(banks, targetExportMapping, flags);
        }
        finally
        {
            Monitor.Exit(_resolveGate);
        }
    }

    // A draw through its shard's memo, under the shard's gate; a miss takes the shared tier inside (ResolveThroughMemo).
    private GraphicsPrograms GetGraphicsProgramsSharded(RegisterBanks banks, ReadOnlySpan<ColorComponentMap> targetExportMapping, in KeptGraphicsFlags flags)
    {
        var shard = _shards[ResolveShards.ShardOf(ResolveShards.DrawRoute(banks, flags.PixelActive), _shards.Length)];
        using var waitScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.GraphicsResolveLockWait);
        if (!Monitor.TryEnter(shard.Gate))
        {
            var started = System.Diagnostics.Stopwatch.GetTimestamp();
            Monitor.Enter(shard.Gate);
            _shardWaitTicks.Add(System.Diagnostics.Stopwatch.GetTimestamp() - started);
        }

        t_shardedDepth++;
        try
        {
            using var workScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPreparation);
            _shardedDraws.Increment();
            return ResolveThroughMemo(shard.Memo!, banks, targetExportMapping, flags);
        }
        finally
        {
            t_shardedDepth--;
            Monitor.Exit(shard.Gate);
        }
    }

    // A draw resolved inside another resolution on this thread (a guest read that runs deferred work), with shards: it
    // must not wait for a shard gate while this thread holds the shared tier or another shard's gate, so it resolves in
    // full under the shared tier and leaves every memo alone. Counted and named once; the programs are the same.
    private GraphicsPrograms ResolveNestedShared(RegisterBanks banks, ReadOnlySpan<ColorComponentMap> targetExportMapping, in KeptGraphicsFlags flags)
    {
        _nestedShared.Increment();
        if (Interlocked.Increment(ref _nestedReports) == 1)
        {
            Console.Error.WriteLine("[GPU][PREFETCH] a draw resolved inside another resolution on one thread: resolved in full on the shared tier, " +
                $"not through a resolver shard's memo (vs=0x{banks.Shader.Vertex.ExportAddress:X16} ps=0x{banks.Shader.Pixel.Address:X16}; counted as nested_shared)");
        }

        _graphicsResolved.Increment();
        return ResolveFullShared(banks, targetExportMapping, flags, SharedReason.Nested);
    }

    // The draw's programs (IShaderPipelineProvider). A draw whose registers, user data, targets and resolved
    // inputs are those of the last draw this thread resolved keeps its programs after their resources revalidate
    // (TryKeepGraphicsPrograms); anything else resolves them again.
    public GraphicsPrograms GetGraphicsPrograms(
        RegisterBanks banks,
        ReadOnlySpan<ColorComponentMap> targetExportMapping,
        bool pixelActive,
        bool depthBound,
        bool depthAspectBound,
        bool stencilAspectBound,
        bool triangleList,
        int startInstanceUserData) =>
        GetGraphicsPrograms(banks, targetExportMapping, pixelActive, depthBound, depthAspectBound, stencilAspectBound, triangleList,
            startInstanceUserData, baseVertexUserData: -1);

    // baseVertexUserData: the vertex user data register that carries the draw's base vertex, or -1 (upstream's form).
    public GraphicsPrograms GetGraphicsPrograms(
        RegisterBanks banks,
        ReadOnlySpan<ColorComponentMap> targetExportMapping,
        bool pixelActive,
        bool depthBound,
        bool depthAspectBound,
        bool stencilAspectBound,
        bool triangleList,
        int startInstanceUserData,
        int baseVertexUserData)
    {
        var vertex = banks.Shader.Vertex;
        var pixel = banks.Shader.Pixel;
        var context = banks.Context;
        if (_shards.Length > 1 && _resolveMemo is { Mode: ResolveMemoMode.On or ResolveMemoMode.Verify })
        {
            var shardedFlags = new KeptGraphicsFlags(pixelActive, depthBound, depthAspectBound, stencilAspectBound, triangleList, startInstanceUserData,
                baseVertexUserData);
            return t_shardedDepth != 0 || Monitor.IsEntered(_resolveGate)
                ? ResolveNestedShared(banks, targetExportMapping, shardedFlags)
                : GetGraphicsProgramsSharded(banks, targetExportMapping, shardedFlags);
        }

        using var waitScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.GraphicsResolveLockWait);
        EnterShared(SharedReason.DrawUnsharded);
        try
        {
            using var workScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPreparation);
            var flags = new KeptGraphicsFlags(pixelActive, depthBound, depthAspectBound, stencilAspectBound, triangleList, startInstanceUserData,
                baseVertexUserData);
            if (_resolveMemo is { Mode: ResolveMemoMode.On or ResolveMemoMode.Verify } memo)
            {
                // The memo replaces the kept programs of the last draw: it holds them too.
                return ResolveThroughMemo(memo, banks, targetExportMapping, flags);
            }

            var stamps = new KeptGraphicsStamps(
                banks.Stamp(RegisterStamp.ShaderPrograms), banks.Stamp(RegisterStamp.VertexUserData), banks.Stamp(RegisterStamp.PixelUserData),
                banks.Stamp(RegisterStamp.ColorTargets, RegisterStamp.ShaderInterface), ShaderHeaderRegistry.Registrations);
            var shadowMiss = ShadowResolveMemo(banks, targetExportMapping, flags);
            ClaimKeptPrograms();
            var kept = _keptGraphics ??= new KeptGraphicsPrograms();
            if (TryKeepGraphicsPrograms(context, kept, stamps, flags, targetExportMapping, out var keptPrograms))
            {
                _graphicsKept.Increment();
                if (shadowMiss) _resolveMemo!.Store();
                return keptPrograms;
            }

            _graphicsResolved.Increment();
            kept.Programs = null;
            var programs = ResolveGraphicsProgramsCore(vertex, pixel, context.ShaderInterface, context, targetExportMapping, pixelActive, depthBound,
                depthAspectBound, stencilAspectBound, triangleList, mesh: null, startInstanceUserData: startInstanceUserData,
                nativeTessellation: true, baseVertexUserData: baseVertexUserData);
            KeepGraphicsPrograms(kept, programs, context, stamps, flags, targetExportMapping);
            if (shadowMiss) _resolveMemo!.Store();
            return programs;
        }
        finally
        {
            Monitor.Exit(_resolveGate);
        }
    }

    // Shader header registrations are in the stamp too: a header registered at a code address changes what a lookup there
    // finds, and it moves no content epoch.
    public ProgramStateStamp ProgramState =>
        new(_programs.ContentEpoch, DeviceDataLoads.Generation, _vertexTables.Generation, ShaderHeaderRegistry.Registrations);

    public string DescribeProgramState() =>
        $"content_epoch={_programs.ContentEpoch} device_data={DeviceDataLoads.Generation} " +
        $"registrations={ShaderHeaderRegistry.Registrations} tables={_vertexTables.Generation} linked={_programs.LinkedLookups}";

    // The resolve memo of this cache (GraphicsResolveMemo), or null with it off; SHARPEMU_DRAW_REUSE=0 turns it off too.
    private GraphicsResolveMemo? _resolveMemo = DrawReuseDisabled || GraphicsResolveMemo.DefaultMode == ResolveMemoMode.Off
        ? null
        : new GraphicsResolveMemo(GraphicsResolveMemo.DefaultMode, GraphicsResolveMemo.DefaultCapacity);

    // For tests: the memo's mode on this cache, with a fresh memo.
    internal ResolveMemoMode ResolveMemoMode
    {
        get => _resolveMemo?.Mode ?? ResolveMemoMode.Off;
        set
        {
            lock (_resolveGate)
            {
                _resolveMemo = value == ResolveMemoMode.Off ? null : new GraphicsResolveMemo(value, GraphicsResolveMemo.DefaultCapacity);
                CreateShards(_shards.Length);
            }
        }
    }

    // For tests: whether this cache's memo rebuilds a words miss from its revalidation (GraphicsResolveMemo.Rebuild).
    internal bool ResolveMemoRebuild
    {
        get => _resolveMemo?.Rebuild ?? false;
        set
        {
            lock (_resolveGate)
            {
                if (_resolveMemo is { } memo) memo.Rebuild = value;
                CreateShards(_shards.Length);
            }
        }
    }

    // For tests: whether this cache's memo keeps only rebuilds whose words repeat (GraphicsResolveMemo.HoldChurn).
    internal bool ResolveMemoHoldChurn
    {
        get => _resolveMemo?.HoldChurn ?? false;
        set
        {
            lock (_resolveGate)
            {
                if (_resolveMemo is { } memo) memo.HoldChurn = value;
                CreateShards(_shards.Length);
            }
        }
    }

    // For tests: whether this cache's memo keeps a new key's resolution only on its second miss (GraphicsResolveMemo.AdmitSeen).
    internal bool ResolveMemoAdmitSeen
    {
        get => _resolveMemo?.AdmitSeen ?? false;
        set
        {
            lock (_resolveGate)
            {
                if (_resolveMemo is { } memo) memo.AdmitSeen = value;
                CreateShards(_shards.Length);
            }
        }
    }

    // Shadow mode: counts whether the memo would have the draw's key; true when it has not and the key is to be stored
    // once the draw resolved.
    private bool ShadowResolveMemo(RegisterBanks banks, ReadOnlySpan<ColorComponentMap> targetExportMapping, in KeptGraphicsFlags flags)
    {
        if (_resolveMemo is not { Mode: ResolveMemoMode.Shadow } memo)
        {
            return false;
        }

        if (_registry.HasFrontend)
        {
            GraphicsResolveMemo.NoteUncacheable(ResolveMemoUncacheable.Frontend);
            return false;
        }

        if (!memo.Fill(banks, targetExportMapping, flags, new PixelOutputBlendBits(PixelOutputBlend(banks.Context))))
        {
            GraphicsResolveMemo.NoteUncacheable(ResolveMemoUncacheable.Key);
            return false;
        }

        if (memo.Find() is not null)
        {
            GraphicsResolveMemo.NoteShadowHit();
            return false;
        }

        GraphicsResolveMemo.NoteKeyMiss(memo.ClassifyMiss());
        return true;
    }

    // A draw through the memo (on or verify): the programs of an entry with its key whose guest reads still give the
    // same answer, else a full resolution kept for the draws that follow.
    private GraphicsPrograms ResolveThroughMemo(GraphicsResolveMemo memo, RegisterBanks banks, ReadOnlySpan<ColorComponentMap> targetExportMapping,
        in KeptGraphicsFlags flags)
    {
        var cacheable = false;
        var fillSerial = 0L;
        var sharedReason = SharedReason.DrawMiss;
        if (_registry.HasFrontend)
        {
            GraphicsResolveMemo.NoteUncacheable(ResolveMemoUncacheable.Frontend);
        }
        else if (_programs.TracesLookups)
        {
            GraphicsResolveMemo.NoteUncacheable(ResolveMemoUncacheable.Trace);
        }
        else if (!memo.Fill(banks, targetExportMapping, flags, new PixelOutputBlendBits(PixelOutputBlend(banks.Context))))
        {
            GraphicsResolveMemo.NoteUncacheable(ResolveMemoUncacheable.Key);
        }
        else
        {
            cacheable = true;
            fillSerial = memo.FillSerial;
            if (memo.Find() is { } entry)
            {
                if (TryTakeResolveMemo(memo, entry, flags.PixelActive, out var taken, out var fingerprint, out var rebuilt) is not { } miss)
                {
                    if (rebuilt)
                    {
                        // A words miss answered from its revalidation: counted as the miss it is, not resolved again.
                        GraphicsResolveMemo.NoteMiss(ResolveMemoMiss.Words);
                        GraphicsResolveMemo.NoteRebuilt();
                        t_resolvePath = ResolvePath.DrawRebuilt;
                    }
                    else
                    {
                        GraphicsResolveMemo.NoteHit();
                        t_resolvePath = ResolvePath.DrawHit;
                    }

                    return memo.Mode == ResolveMemoMode.Verify
                        ? VerifyResolveMemo(memo, entry, taken!, fingerprint, banks, targetExportMapping, flags)
                        : taken!;
                }

                GraphicsResolveMemo.NoteMiss(miss);
            }
            else
            {
                GraphicsResolveMemo.NoteMiss(ResolveMemoMiss.New);
                sharedReason = SharedReason.DrawNew;
                GraphicsResolveMemo.NoteKeyMiss(memo.ClassifyMiss());
                // GraphicsResolveMemo.DefaultAdmitSeen: a key's first miss resolves in full without being kept.
                if (memo.AdmitSeen && !memo.SeenBefore())
                {
                    GraphicsResolveMemo.NoteNotAdmitted();
                    cacheable = false;
                }
            }
        }

        _graphicsResolved.Increment();
        t_resolvePath = ResolvePath.DrawFull;
        // The miss: the full resolution and its store read and fill the shared tier (the last sources StoreResolveMemo
        // reads are the ones this resolution's PrepareSource left), so both run under it (re-entered with one shard).
        EnterShared(sharedReason);
        try
        {
            var deviceDataGeneration = DeviceDataLoads.Generation;
            var contentEpoch = _programs.ContentEpoch;
            var linkedLookups = _programs.LinkedLookups;
            var tablesGeneration = _vertexTables.Generation;
            var programs = ResolveFull(banks, targetExportMapping, flags);
            if (cacheable && memo.FillSerial != fillSerial)
            {
                // The memo holds another draw's key now: this result has no key to be stored under.
                GraphicsResolveMemo.NoteUncacheable(ResolveMemoUncacheable.Changed);
            }
            else if (cacheable)
            {
                StoreResolveMemo(memo, banks, programs, flags.PixelActive, deviceDataGeneration, contentEpoch, linkedLookups, tablesGeneration);
            }

            return programs;
        }
        finally
        {
            Monitor.Exit(_resolveGate);
        }
    }

    private GraphicsPrograms ResolveFull(RegisterBanks banks, ReadOnlySpan<ColorComponentMap> targetExportMapping, in KeptGraphicsFlags flags) =>
        ResolveGraphicsProgramsCore(banks.Shader.Vertex, banks.Shader.Pixel, banks.Context.ShaderInterface, banks.Context, targetExportMapping,
            flags.PixelActive, flags.DepthBound, flags.DepthAspectBound, flags.StencilAspectBound, flags.TriangleList, mesh: null,
            startInstanceUserData: flags.StartInstanceUserData, nativeTessellation: true, baseVertexUserData: flags.BaseVertexUserData);

    // Null when every guest read the entry's resolution made still gives the same answer: the program cache and the
    // device data loads unchanged, the same header record and code identity at each stage's address, the same vertex
    // table words and pixel input semantics, and each stage's descriptor words evaluating to its snapshot (the reads
    // the full resolution records are made here as well).
    // taken: the programs to hand out (the entry's, or rebuilt from them), with their verify fingerprint. held (verify's
    // re-check): the programs a take handed out, revalidated in place of the entry's. allowRebuild false (that re-check): a
    // words miss is a miss, and the entry is left as it is.
    private ResolveMemoMiss? TryTakeResolveMemo(GraphicsResolveMemo memo, GraphicsResolveMemo.Entry entry, bool pixelActive,
        out GraphicsPrograms? taken, out ulong fingerprint, out bool rebuilt, bool allowRebuild = true, GraphicsPrograms? held = null)
    {
        taken = null;
        fingerprint = 0;
        rebuilt = false;
        if (entry.ContentEpoch != _programs.ContentEpoch || entry.TablesGeneration != _vertexTables.Generation)
        {
            return ResolveMemoMiss.Epoch;
        }

        if (entry.DeviceDataGeneration != DeviceDataLoads.Generation)
        {
            return ResolveMemoMiss.Generation;
        }

        if (!ReferenceEquals(_registry.Require(entry.VertexAddress, "vertex"), entry.VertexShader))
        {
            return ResolveMemoMiss.Header;
        }

        if (HashOf(entry.VertexShader!, entry.VertexAddress, "vertex") != entry.VertexHash)
        {
            return ResolveMemoMiss.Identity;
        }

        if (pixelActive)
        {
            if (!ReferenceEquals(_registry.Require(entry.PixelAddress, "pixel"), entry.PixelShader))
            {
                return ResolveMemoMiss.Header;
            }

            if (HashOf(entry.PixelShader!, entry.PixelAddress, "pixel") != entry.PixelHash)
            {
                return ResolveMemoMiss.Identity;
            }
        }

        if (entry.TableWords is { } words && !VertexInputResolver.SameTableWords(_context, words, entry.AttributeTable, entry.BufferTable,
                _vertexTableScratch ??= new uint[VertexTableWords.MaxWords]))
        {
            return ResolveMemoMiss.Tables;
        }

        if (pixelActive &&
            PixelStageInputResolver.CustomInterpolationMask(_context, entry.PixelShader!, entry.PixelSemanticCount) != entry.PixelCustomMask)
        {
            return ResolveMemoMiss.Semantics;
        }

        // Each stage's descriptor words, pixel first as the full resolution reads them. With rebuild, a stage whose words
        // changed takes the snapshot its full resolution would materialize now from this same evaluation: everything else
        // that resolution reads is the entry's (checked above or in the key), so it would pick the entry's programs again
        // as long as the words keep their specialization (RevalidateOrRebuild refuses otherwise).
        var programs = held ?? entry.Programs!;
        var pixelStage = programs.PixelInput.Stage;
        var vertexStage = programs.VertexInput.Stage;
        var rebuild = allowRebuild && memo.Rebuild;
        var pixelRebuilt = false;
        var vertexRebuilt = false;
        if ((pixelActive && !RevalidateMemoStage(ref pixelStage, rebuild, ref pixelRebuilt)) || !RevalidateMemoStage(ref vertexStage, rebuild, ref vertexRebuilt))
        {
            return ResolveMemoMiss.Words;
        }

        var stageRebuilt = pixelRebuilt || vertexRebuilt;

        // A load learned while the words were read.
        if (entry.DeviceDataGeneration != DeviceDataLoads.Generation)
        {
            if (!stageRebuilt) return ResolveMemoMiss.Generation;
            GraphicsResolveMemo.NoteRebuildRefused(ResolveMemoRebuildRefusal.Generation);
            return ResolveMemoMiss.Words;
        }

        // With shards, a hit runs beside other shards' misses on the shared tier: a program entry replaced or the vertex
        // tables dropped while its words were read makes it a miss, as a load learned meanwhile does above. With one
        // shard nothing moves them while the gate is held.
        if (_shards.Length > 1 && (entry.ContentEpoch != _programs.ContentEpoch || entry.TablesGeneration != _vertexTables.Generation))
        {
            return ResolveMemoMiss.Epoch;
        }

        if (!stageRebuilt)
        {
            taken = programs;
            fingerprint = entry.Fingerprint;
            if (held is null)
            {
                entry.RebuiltLast = false;
            }

            return null;
        }

        taken = WithStages(programs, vertexRebuilt ? vertexStage : null, pixelRebuilt ? pixelStage : null);
        fingerprint = memo.Mode == ResolveMemoMode.Verify ? GraphicsProgramsComparer.Fingerprint(taken) : 0;
        rebuilt = true;
        // GraphicsResolveMemo.DefaultHoldChurn: a rebuild right after another is kept only when both evaluated the same words.
        var signature = 0UL;
        if (memo.HoldChurn && entry.RebuiltLast)
        {
            signature = RebuildSignature(taken, vertexRebuilt, pixelRebuilt);
            if (signature != entry.RebuildSignature)
            {
                entry.RebuildSignature = signature;
                GraphicsResolveMemo.NoteRebuildHeld();
                return null;
            }
        }

        entry.Programs = taken;
        entry.Fingerprint = fingerprint;
        entry.RebuildSignature = signature;
        entry.RebuiltLast = true;
        return null;
    }

    // What a rebuild evaluated, for telling a draw whose words keep moving from one whose words settled (HoldChurn): the
    // flattened tables of the rebuilt stages (a stage not rebuilt is the entry's own), which hold every descriptor word the
    // evaluation read from memory; the rest of a fast snapshot comes from the origin's own user data. A collision, or words
    // the tables miss, only keeps a correct rebuild.
    private static ulong RebuildSignature(GraphicsPrograms programs, bool vertexRebuilt, bool pixelRebuilt)
    {
        var hash = 0x9E3779B97F4A7C15UL ^ (vertexRebuilt ? 1UL : 0) ^ (pixelRebuilt ? 2UL : 0);
        if (vertexRebuilt) hash = SignTable(hash, programs.VertexInput.Stage.Resources);
        return pixelRebuilt ? SignTable(hash, programs.PixelInput.Stage.Resources) : hash;
    }

    private static ulong SignTable(ulong hash, ResourceSnapshot snapshot) =>
        XxHash3.HashToUInt64(MemoryMarshal.AsBytes(snapshot.FlattenedResourceTable.AsSpan()), (long)hash);

    // False when the stage's words no longer evaluate to its snapshot and it was not rebuilt (refusals counted).
    private bool RevalidateMemoStage(ref ShaderStageResources stage, bool rebuild, ref bool rebuilt)
    {
        _host.BeginGuestWordReads();
        try
        {
            switch (_programs.RevalidateOrRebuild(stage, rebuild, out var next))
            {
                case ResourceMaterializer.FastRevalidation.Current:
                    return true;
                case ResourceMaterializer.FastRevalidation.Rebuilt:
                    stage = next;
                    rebuilt = true;
                    return true;
                case ResourceMaterializer.FastRevalidation.NotFast:
                    GraphicsResolveMemo.NoteRebuildRefused(ResolveMemoRebuildRefusal.Origin);
                    return false;
                default:
                    if (rebuild)
                    {
                        GraphicsResolveMemo.NoteRebuildRefused(ResourceMaterializer.LastRebuildRefusal switch
                        {
                            ResourceMaterializer.RebuildRefusal.Evaluate => ResolveMemoRebuildRefusal.Evaluate,
                            ResourceMaterializer.RebuildRefusal.Failed => ResolveMemoRebuildRefusal.Failed,
                            ResourceMaterializer.RebuildRefusal.Specialization => ResolveMemoRebuildRefusal.Specialization,
                            _ => ResolveMemoRebuildRefusal.Refused,
                        });
                    }

                    return false;
            }
        }
        finally
        {
            _host.EndGuestWordReads();
        }
    }

    // The programs with these stage resources in place of their own; the input descriptions of a rebuilt stage are
    // copied, never written: the result held before may still be on its way to the worker. A stage not rebuilt keeps its
    // input description, shared as a hit shares the whole result.
    private static GraphicsPrograms WithStages(GraphicsPrograms programs, ShaderStageResources? vertexStage, ShaderStageResources? pixelStage)
    {
        var vertexInput = programs.VertexInput;
        if (vertexStage is { } vertex)
        {
            vertexInput = vertexInput.WithoutStage();
            vertexInput.Stage = vertex;
        }

        var pixelInput = programs.PixelInput;
        if (pixelStage is { } pixel)
        {
            pixelInput = pixelInput.WithoutStage();
            pixelInput.Stage = pixel;
        }

        return new GraphicsPrograms(vertexInput, pixelInput)
        {
            Tessellation = programs.Tessellation,
            Vertex = programs.Vertex,
            Pixel = programs.Pixel,
            MeshInput = programs.MeshInput,
            Available = programs.Available,
            SolidClear = programs.SolidClear,
            PositionStream = programs.PositionStream,
            DisableBlending = programs.DisableBlending,
            DropUnwrittenColorTargets = programs.DropUnwrittenColorTargets,
        };
    }

    // Keeps a full resolution under the filled key, with what a hit re-checks, when it can be repeated from them: one
    // vertex and pixel program pair with no tessellation, mesh or shader calls, nothing changed while it resolved, and
    // vertex tables read back to the inputs it decoded (the guest may have written them since).
    private void StoreResolveMemo(GraphicsResolveMemo memo, RegisterBanks banks, GraphicsPrograms programs, bool pixelActive,
        int deviceDataGeneration, long contentEpoch, long linkedLookups, long tablesGeneration)
    {
        var cause = !programs.Available ? ResolveMemoUncacheable.Unavailable
            : programs.MeshInput is not null ? ResolveMemoUncacheable.Mesh
            : programs.Tessellation is not null || programs.VertexInput.Tessellation is not null ? ResolveMemoUncacheable.Tessellation
            : _programs.LinkedLookups != linkedLookups || HasCalls(programs.VertexInput.Stage) || HasCalls(programs.PixelInput.Stage)
                ? ResolveMemoUncacheable.Calls
            : DeviceDataLoads.Generation != deviceDataGeneration || _programs.ContentEpoch != contentEpoch ||
              _vertexTables.Generation != tablesGeneration ? ResolveMemoUncacheable.Changed
            : programs.VertexInput.Stage.Program is null || programs.VertexInput.Stage.Resources.Origin is null ||
              (pixelActive && (programs.PixelInput.Stage.Program is null || programs.PixelInput.Stage.Resources.Origin is null))
                ? ResolveMemoUncacheable.Origin
            : (ResolveMemoUncacheable?)null;
        var vertexSource = _lastSources[(int)ShaderStage.Vertex];
        var pixelSource = pixelActive ? _lastSources[(int)ShaderStage.Pixel] : null;
        if (cause is null && (vertexSource is null || vertexSource.Address != banks.Shader.Vertex.ExportAddress ||
            (pixelActive && (pixelSource is null || pixelSource.Address != banks.Shader.Pixel.Address))))
        {
            cause = ResolveMemoUncacheable.Changed;
        }

        VertexTableWords? tableWords = null;
        ulong attributeTable = 0, bufferTable = 0;
        if (cause is null && !TryReadMemoTables(vertexSource!, programs.VertexInput, out tableWords, out attributeTable, out bufferTable))
        {
            cause = ResolveMemoUncacheable.Tables;
        }

        if (cause is { } refused)
        {
            GraphicsResolveMemo.NoteUncacheable(refused);
            return;
        }

        var entry = memo.Store();
        entry.Programs = programs;
        entry.VertexShader = vertexSource!.Registered;
        entry.VertexAddress = vertexSource.Address;
        entry.VertexHash = vertexSource.Hash;
        entry.TableWords = tableWords;
        entry.AttributeTable = attributeTable;
        entry.BufferTable = bufferTable;
        if (pixelSource is not null)
        {
            entry.PixelShader = pixelSource.Registered;
            entry.PixelAddress = pixelSource.Address;
            entry.PixelHash = pixelSource.Hash;
            entry.PixelSemanticCount = Math.Min(Math.Min(pixelSource.Registered.InputSemanticsCount, programs.PixelInput.InputCount),
                (uint)PixelInputInfo.InterpolatorCount);
            entry.PixelCustomMask = programs.PixelInput.CustomInterpolationMask;
        }

        entry.ContentEpoch = contentEpoch;
        entry.DeviceDataGeneration = deviceDataGeneration;
        entry.TablesGeneration = tablesGeneration;
        entry.Fingerprint = memo.Mode == ResolveMemoMode.Verify ? GraphicsProgramsComparer.Fingerprint(programs) : 0;
    }

    private static bool HasCalls(in ShaderStageResources stage) =>
        stage.CallArguments is { Length: > 0 } || stage.CallTargets is { Length: > 0 } || stage.CallRecords is { Length: > 0 } ||
        stage.CallRecordRanges is { Length: > 0 };

    // The vertex table words the draw's inputs decode from, read again now: false when they cannot be compared later
    // (static attributes, more inputs than a table holds, a table not read whole) or no longer decode to the draw's
    // attributes and buffers. Null words when the program reads no tables.
    private bool TryReadMemoTables(ShaderSource source, VertexInputInfo resolved, out VertexTableWords? words, out ulong attributeTable,
        out ulong bufferTable)
    {
        words = null;
        attributeTable = bufferTable = 0;
        if (!_vertexTables.TryGet(source.Registered, source.Hash, out var metadata) || metadata is null)
        {
            return false;
        }

        if (!VertexInputResolver.ReadsTables(metadata))
        {
            return resolved.Attributes.Length == 0 && resolved.Buffers.Length == 0;
        }

        if (metadata.InputSemantics.Length > VertexInputInfo.MaxBuffers)
        {
            return false;
        }

        foreach (var word in metadata.InputSemantics)
        {
            var semantic = new ShaderInputSemantic(word);
            if (semantic.StaticBufferIndex || semantic.StaticAttribute)
            {
                return false;
            }
        }

        attributeTable = VertexInputResolver.ReadTablePointer(source.UserData, metadata.VertexAttributeRegister);
        bufferTable = VertexInputResolver.ReadTablePointer(source.UserData, metadata.VertexBufferRegister);
        if (attributeTable == 0 || bufferTable == 0)
        {
            return false;
        }

        var decoded = VertexInputResolver.ResolveVertexInputs(_context, source.Registered, metadata, source.UserData, resolved.PositionExportControl,
            resolved.ClipSpace, out words);
        return words is { WholeSpans: true } &&
            decoded.Attributes.AsSpan().SequenceEqual(resolved.Attributes) && decoded.Buffers.AsSpan().SequenceEqual(resolved.Buffers);
    }

    // Verify: the hit's programs against a full resolution of the same draw, and against what they were when kept; the
    // programs to hand out. A resolution beside the guest and the worker reads guest memory a little later than the hit's
    // revalidation did, so a word written between the two shows as a difference no key or revalidation could have
    // caught: the hit was current when it was made, and the take check judges its recorded reads as it judges a full
    // resolution's. So a difference is first checked for that race: the hit's revalidation is made again (without a
    // rebuild), and a miss now means its words changed since (raced, the full resolution handed out); while it still
    // holds, the draw is resolved in full once more, and an answer equal to the hit means the difference came and went
    // (raced). Only a hit that still revalidates while a second full resolution still differs from it is a mismatch, and
    // a result written since it was kept always is; either stops emulation (on a speculative thread the work is
    // abandoned and counted, GpuFatal.VerifyFailed).
    // memoPrograms: what the take hands out (the entry's, or a rebuild kept or not), fingerprint: theirs when taken.
    private GraphicsPrograms VerifyResolveMemo(GraphicsResolveMemo memo, GraphicsResolveMemo.Entry entry, GraphicsPrograms memoPrograms,
        ulong fingerprint, RegisterBanks banks, ReadOnlySpan<ColorComponentMap> targetExportMapping, in KeptGraphicsFlags flags)
    {
        var full = ResolveFullShared(banks, targetExportMapping, flags, SharedReason.Verify);
        GraphicsResolveMemo.NoteVerified();
        var difference = GraphicsProgramsComparer.Difference(memoPrograms, full);
        if (difference is null)
        {
            if (GraphicsProgramsComparer.Fingerprint(memoPrograms) == fingerprint)
            {
                return memoPrograms;
            }

            throw ResolveMemoVerifyFailed(entry, "written_since_kept", "");
        }

        // Named before anything is read again: the words that differ, their addresses, and both permutation keys.
        var stages = GraphicsProgramsComparer.DescribeStages(memoPrograms, full);
        // Ahead of the worker, reads that straddle a GPU write recorded since the resolution began (the hit's proof
        // compared through the write, the full resolution read a line answered clean before it): the take refuses this
        // resolution (IShaderPipelineHost.ConcurrentReadsSuperseded latches it), whichever programs it hands out.
        if (_host.ConcurrentReadsSuperseded())
        {
            GraphicsResolveMemo.NoteVerifyRaced(ResolveMemoVerifyRace.Superseded);
            ReportResolveMemoRace(entry, difference, "recheck=superseded", stages);
            return full;
        }

        if (TryTakeResolveMemo(memo, entry, flags.PixelActive, out _, out _, out _, allowRebuild: false, held: memoPrograms) is { } miss)
        {
            GraphicsResolveMemo.NoteVerifyRaced(ResolveMemoVerifyRace.Changed);
            ReportResolveMemoRace(entry, difference, $"recheck=miss:{miss.ToString().ToLowerInvariant()}", stages);
            return full;
        }

        var again = ResolveFullShared(banks, targetExportMapping, flags, SharedReason.Verify);
        if (GraphicsProgramsComparer.Difference(memoPrograms, again) is not { } againDifference)
        {
            GraphicsResolveMemo.NoteVerifyRaced(ResolveMemoVerifyRace.CameAndWent);
            ReportResolveMemoRace(entry, difference, "recheck=current again=same_as_hit", stages);
            return memoPrograms;
        }

        throw ResolveMemoVerifyFailed(entry, difference,
            $" recheck=current again={againDifference} hit_vs_full=[{stages}] hit_vs_again=[{GraphicsProgramsComparer.DescribeStages(memoPrograms, again)}]");
    }

    private static Exception ResolveMemoVerifyFailed(GraphicsResolveMemo.Entry entry, string difference, string detail)
    {
        GraphicsResolveMemo.NoteVerifyFailed();
        var message = $"[GPU][RESOLVE_MEMO] verify mismatch field={difference} vs=0x{entry.VertexAddress:X16} ps=0x{entry.PixelAddress:X16} " +
            $"key_hash=0x{entry.Hash:X16}{detail}.";
        Console.Error.WriteLine(message);
        return GpuFatal.VerifyFailed(message);
    }

    // The first races of a run, with what differed: the words and the guest addresses they were read from name the writer.
    private static int _resolveMemoRaceReports;
    private const int MaxResolveMemoRaceReports = 32;

    private static void ReportResolveMemoRace(GraphicsResolveMemo.Entry entry, string difference, string how, string stages)
    {
        if (Interlocked.Increment(ref _resolveMemoRaceReports) > MaxResolveMemoRaceReports)
        {
            return;
        }

        var thread = Thread.CurrentThread;
        Console.Error.WriteLine($"[GPU][RESOLVE_MEMO] verify raced field={difference} {how} speculative={GpuFatal.IsSpeculative} " +
            $"thread={thread.Name ?? thread.ManagedThreadId.ToString(System.Globalization.CultureInfo.InvariantCulture)} " +
            $"vs=0x{entry.VertexAddress:X16} ps=0x{entry.PixelAddress:X16} key_hash=0x{entry.Hash:X16} hit_vs_full=[{stages}].");
    }

    // The interface's earlier forms, for callers holding only the four inputs.
    public GraphicsPrograms GetGraphicsPrograms(
        RegisterBanks banks,
        ReadOnlySpan<ColorComponentMap> targetExportMapping,
        bool pixelActive,
        bool depthBound) =>
        GetGraphicsPrograms(banks, targetExportMapping, pixelActive, depthBound, depthAspectBound: depthBound, stencilAspectBound: false,
            triangleList: false, startInstanceUserData: -1);

    public GraphicsPrograms GetGraphicsPrograms(
        RegisterBanks banks,
        ReadOnlySpan<ColorComponentMap> targetExportMapping,
        bool pixelActive,
        bool depthBound,
        int startInstanceUserData) =>
        GetGraphicsPrograms(banks, targetExportMapping, pixelActive, depthBound, depthAspectBound: depthBound, stencilAspectBound: false,
            triangleList: false, startInstanceUserData);

    // The register-bank form without the bank: resolves every time, as before draw reuse.
    public GraphicsPrograms GetGraphicsPrograms(
        VertexStageRegisters vertex,
        PixelStageRegisters pixel,
        ShaderInterfaceRegisters shaderInterface,
        ContextRegisters context,
        ReadOnlySpan<ColorComponentMap> targetExportMapping,
        bool pixelActive,
        bool depthBound,
        bool depthAspectBound,
        bool stencilAspectBound = false) =>
        GetGraphicsPrograms(vertex, pixel, shaderInterface, context, targetExportMapping, pixelActive, depthBound,
            depthAspectBound, stencilAspectBound, triangleList: false);

    public GraphicsPrograms GetGraphicsPrograms(
        VertexStageRegisters vertex,
        PixelStageRegisters pixel,
        ShaderInterfaceRegisters shaderInterface,
        ContextRegisters context,
        ReadOnlySpan<ColorComponentMap> targetExportMapping,
        bool pixelActive,
        bool depthBound,
        bool depthAspectBound,
        bool stencilAspectBound,
        bool triangleList) =>
        GetGraphicsProgramsCore(vertex, pixel, shaderInterface, context, targetExportMapping, pixelActive, depthBound,
            depthAspectBound, stencilAspectBound, triangleList, null);

    // The kept programs are per thread, but their handles belong to the host of the cache that resolved them:
    // a thread that resolves for another cache (a new render host in the same process) drops them first.
    private static long _nextKeptOwner;
    private readonly long _keptOwner = Interlocked.Increment(ref _nextKeptOwner);
    [ThreadStatic]
    private static long _keptOwnerOnThread;
    [ThreadStatic]
    private static KeptGraphicsPrograms? _keptGraphics;
    [ThreadStatic]
    private static Dictionary<ulong, List<KeptComputeProgram>>? _keptComputes;
    private const int KeptComputesPerShader = 4;

    private void ClaimKeptPrograms()
    {
        if (_keptOwnerOnThread != _keptOwner)
        {
            _keptOwnerOnThread = _keptOwner;
            _keptGraphics = null;
            _keptComputes = null;
        }
    }
    // A rebound compute program reads its code's identity again every this many dispatches.
    private const int IdentityRecheckPeriod = 64;
    [ThreadStatic]
    private static uint[]? _vertexTableScratch;

    private static readonly ThreadCounter _graphicsKept = new();
    private static readonly ThreadCounter _graphicsResolved = new();
    private static readonly ThreadCounter _computeKept = new();
    private static readonly ThreadCounter _computeRebound = new();
    private static readonly ThreadCounter _computeResolved = new();

    public static string TakeKeptProgramsReport() =>
        FormattableString.Invariant($"[PERF][KEPT_PROGRAMS] compute_kept={_computeKept.Take()} compute_rebound={_computeRebound.Take()} compute_resolved={_computeResolved.Take()}");

    /// <summary>
    /// How graphics resolutions were answered since the last call, for the backend's always-on counter line (the
    /// prefetch report): by the draw's own kept programs, by a full resolution, and what the resolve memo did.
    /// </summary>
    public static string TakeGraphicsResolveReport() =>
        FormattableString.Invariant($"graphics_kept={_graphicsKept.Take()} graphics_resolved={_graphicsResolved.Take()} ") +
        GraphicsResolveMemo.TakeReport() + " " + ResourceMaterializer.TakeReadProofReport();

    // Besides upstream's register stamps, the fork's resolution reads the host viewport transform (dynamic
    // viewport registers, compared resolved), the depth aspects and triangle-list assembly (per-vertex pixel
    // inputs), so those are part of the check too. Called under _resolveGate: revalidation uses the
    // program cache's materializations.
    private bool TryKeepGraphicsPrograms(ContextRegisters context, KeptGraphicsPrograms kept, in KeptGraphicsStamps stamps, in KeptGraphicsFlags flags,
        ReadOnlySpan<ColorComponentMap> targetExportMapping, out GraphicsPrograms programs)
    {
        programs = null!;
        if (DrawReuseDisabled || _registry.HasFrontend)
            return false;
        if (kept.Programs is not { } previous || kept.Stamps != stamps || kept.Flags != flags ||
            kept.ExportMappingCount != targetExportMapping.Length ||
            !targetExportMapping.SequenceEqual(kept.ExportMapping.AsSpan(0, kept.ExportMappingCount)))
        {
            return false;
        }

        if ((flags.PixelActive && kept.OutputBlend != PixelOutputBlend(context)) ||
            HostViewport(context).ClipSpace != previous.VertexInput.ClipSpace)
        {
            return false;
        }

        if (kept.TableWords is { } words && !SameVertexTables(kept, words))
        {
            return false;
        }

        if (!_programs.Revalidate(previous.VertexInput.Stage) || (flags.PixelActive && !_programs.Revalidate(previous.PixelInput.Stage)) ||
            (previous.Tessellation is { } tessellation && !_programs.Revalidate(tessellation.Hull.Input.Stage)))
        {
            return false;
        }

        programs = previous;
        return true;
    }

    private void KeepGraphicsPrograms(KeptGraphicsPrograms kept, GraphicsPrograms programs, ContextRegisters context, in KeptGraphicsStamps stamps,
        in KeptGraphicsFlags flags, ReadOnlySpan<ColorComponentMap> targetExportMapping)
    {
        kept.Programs = null;
        // A program with shader calls reads its call tables at every draw: its call records, targets and ready fallback can
        // change while its registers and resources stay the same (upstream's rule; the memo refuses it as Calls).
        if (DrawReuseDisabled || _registry.HasFrontend || !programs.Available || programs.MeshInput is not null ||
            programs.VertexInput.Stage.Program is null || HasCalls(programs.VertexInput.Stage) ||
            (flags.PixelActive && HasCalls(programs.PixelInput.Stage)) || !TryReadKeptTables(kept, programs))
        {
            return;
        }

        kept.Programs = programs;
        kept.Stamps = stamps;
        kept.Flags = flags;
        kept.OutputBlend = flags.PixelActive ? PixelOutputBlend(context) : 0;
        kept.ExportMappingCount = targetExportMapping.Length;
        targetExportMapping.CopyTo(kept.ExportMapping);
    }

    // The vertex table words the kept vertex inputs were decoded from, read again on a keep so a later draw
    // can compare them (VertexInputResolver.SameTableWords); false keeps nothing (static attributes, which
    // ReadTableWords refuses, or more attributes than a table holds).
    private bool TryReadKeptTables(KeptGraphicsPrograms kept, GraphicsPrograms programs)
    {
        kept.TableWords = null;
        kept.AttributeTable = 0;
        kept.BufferTable = 0;
        var source = _lastSources[(int)ShaderStage.Vertex];
        if (programs.Tessellation is not null || source is null)
        {
            // A native tessellation draw reads no vertex tables (its domain stage has none).
            return programs.Tessellation is not null;
        }

        if (!_vertexTables.TryGet(source.Registered, source.Hash, out var tables) || tables is null)
        {
            tables = VertexInputResolver.RequireTables(_context, source.Registered);
        }

        if (!VertexInputResolver.ReadsTables(tables))
        {
            return true;
        }

        if (tables.InputSemantics.Length > VertexInputInfo.MaxBuffers)
        {
            return false;
        }

        foreach (var word in tables.InputSemantics)
        {
            var semantic = new ShaderInputSemantic(word);
            if (semantic.StaticBufferIndex || semantic.StaticAttribute)
            {
                return false;
            }
        }

        var attributeTable = VertexInputResolver.ReadTablePointer(source.UserData, tables.VertexAttributeRegister);
        var bufferTable = VertexInputResolver.ReadTablePointer(source.UserData, tables.VertexBufferRegister);
        if (attributeTable == 0 || bufferTable == 0)
        {
            return false;
        }

        kept.TableWords = VertexInputResolver.ReadTableWords(_context, tables.InputSemantics, attributeTable, bufferTable, source.Address);
        kept.AttributeTable = attributeTable;
        kept.BufferTable = bufferTable;
        return kept.TableWords.WholeSpans;
    }

    private bool SameVertexTables(KeptGraphicsPrograms kept, VertexTableWords words) =>
        VertexInputResolver.SameTableWords(_context, words, kept.AttributeTable, kept.BufferTable,
            _vertexTableScratch ??= new uint[VertexTableWords.MaxWords]);

    // The blend state the pixel outputs are resolved from: the second blend source of target zero and the
    // targets that blend with the source alpha (NeedsDualSourceAlpha).
    private static uint PixelOutputBlend(ContextRegisters context)
    {
        var bits = context.BlendControls[0].UsesSecondarySource ? 1u : 0u;
        for (var slot = 0u; slot < ContextRegisters.ColorTargetCount; slot++)
        {
            if (BlendsWithSourceAlpha(context, slot))
            {
                bits |= 2u << (int)slot;
            }
        }

        return bits;
    }

    // SHARPEMU_DRAW_REUSE=0 resolves every draw's programs, as before draw reuse (A/B comparisons).
    private static readonly bool DrawReuseDisabled = Environment.GetEnvironmentVariable("SHARPEMU_DRAW_REUSE") == "0";

    // A tessellated draw's hull threadgroup: one guest wave64, as many whole patches as fit.
    private const uint HullWaveLanes = 64;

    // What a hull program hands each threadgroup's patches to the domain stage in, and the factor
    // ring's bytes per patch: GTA V's programs address a 32 KiB off-chip block per threadgroup
    // (patch constants grow down from 0x8000) and write four factor dwords per triangle patch; a quad
    // patch has six (four edges, two inside).
    public const uint TessellationOffchipBlockBytes = 0x8000;

    public static uint TessellationFactorBytesPerPatch(bool quad) => quad ? 24u : 16u;

    public static uint TessellationPatchesPerGroup(uint controlPoints) =>
        controlPoints == 0 ? 0 : HullWaveLanes / controlPoints;

    // The largest factor the guest's tessellator allows (VGT_HOS_MAX_TESS_LEVEL), rounded up: part of
    // the domain program's pipeline state, which the host sizes its own tessellator's limit from.
    public static uint TessellationMaxFactor(uint maxLevelBits)
    {
        var level = BitConverter.UInt32BitsToSingle(maxLevelBits);
        return float.IsFinite(level) ? (uint)Math.Clamp(MathF.Ceiling(level), 1f, 64f) : 64u;
    }

    // The programs of a tessellated draw (Gen5TessellationLaunch): the domain program as the host
    // tessellator's post-tessellation vertex function with the pixel program as for any draw, and
    // the merged local/hull program as a compute kernel the executor runs first. Never kept: the
    // hull kernel's launch is part of the lookup.
    public TessellationPrograms? GetTessellationPrograms(
        RegisterBanks banks,
        ReadOnlySpan<ColorComponentMap> targetExportMapping,
        bool pixelActive,
        bool depthBound,
        bool depthAspectBound,
        bool stencilAspectBound,
        uint controlPoints,
        bool quad) =>
        GetTessellationPrograms(banks.Shader.Vertex, banks.Shader.Pixel, banks.Context.ShaderInterface, banks.Context, targetExportMapping,
            pixelActive, depthBound, depthAspectBound, stencilAspectBound, controlPoints, quad);

    public TessellationPrograms? GetTessellationPrograms(
        VertexStageRegisters vertex,
        PixelStageRegisters pixel,
        ShaderInterfaceRegisters shaderInterface,
        ContextRegisters context,
        ReadOnlySpan<ColorComponentMap> targetExportMapping,
        bool pixelActive,
        bool depthBound,
        bool depthAspectBound,
        bool stencilAspectBound,
        uint controlPoints,
        bool quad)
    {
        // VGT_TF_PARAM: TYPE [1:0] (1 = triangles, 2 = quads), PARTITIONING [4:2], TOPOLOGY [7:5] (2 =
        // triangles wound clockwise, 3 = counter-clockwise; RenderExecutor.IsEmulatedTessellation takes no other).
        var parameter = shaderInterface.TessellationFactorParameter;
        var domain = new Gen5TessellationLaunch(
            Gen5TessellationRole.Domain, TessellationPatchesPerGroup(controlPoints), controlPoints,
            TessellationOffchipBlockBytes, TessellationFactorBytesPerPatch(quad),
            (Gen5TessellationPartition)((parameter >> 2) & 0x3),
            OutputClockwise: ((parameter >> 5) & 0x7) == 2,
            TessellationMaxFactor(shaderInterface.MaxTessellationLevel),
            quad);
        using var waitScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.GraphicsResolveLockWait);
        using (EnterSharedScope(SharedReason.Other))
        {
            using var workScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPreparation);
            var started = TraceTessellationPrograms ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            var graphics = ResolveGraphicsProgramsCore(vertex, pixel, shaderInterface, context, targetExportMapping, pixelActive,
                depthBound, depthAspectBound, stencilAspectBound, triangleList: true, mesh: null, tessellation: domain);
            var resolved = TraceTessellationPrograms ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            var hull = graphics.Available
                ? PrepareHullProgram(vertex, domain with { Role = Gen5TessellationRole.Hull })
                : new ComputeProgram { Available = false };
            if (TraceTessellationPrograms && _tracedTessellationPrograms++ < 200)
            {
                var frequency = (double)System.Diagnostics.Stopwatch.Frequency;
                Console.Error.WriteLine(
                    $"[GPU][TESS-PROGRAMS] graphics_ms={(resolved - started) * 1000 / frequency:F2} " +
                    $"hull_ms={(System.Diagnostics.Stopwatch.GetTimestamp() - resolved) * 1000 / frequency:F2}");
            }

            return new TessellationPrograms(graphics, hull);
        }
    }

    // SHARPEMU_TRACE_TESSELLATION_TIMES=1 also splits the program lookup of each tessellated draw.
    private static readonly bool TraceTessellationPrograms =
        Environment.GetEnvironmentVariable("SHARPEMU_TRACE_TESSELLATION_TIMES") == "1";
    private int _tracedTessellationPrograms;

    // The merged local/hull program as a compute kernel of one guest wave64. Its hull half reloads
    // its user data through SPI_SHADER_USER_DATA_ADDR_HS in s[0:1], so that pointer is user data, as
    // the GS pointer is for a fused export program (WithGeometryUserDataAddress); s2..s7 are the
    // system scalars the launch seeds, and the user registers follow at s8.
    private ComputeProgram PrepareHullProgram(VertexStageRegisters vertex, Gen5TessellationLaunch hull)
    {
        var source = PrepareSource(
            vertex.LocalAddress, ShaderStage.Compute, "hull", vertex.HullUserScalars, vertex.HullResource2.UserScalarCount,
            probeWrittenRegisters: true, VertexUserDataBase, ref _hullUserData);
        var userData = new uint[VertexUserDataBase + source.UserData.Length];
        userData[0] = (uint)vertex.HullUserDataAddress;
        userData[1] = (uint)(vertex.HullUserDataAddress >> 32);
        source.UserData.CopyTo(userData, (int)VertexUserDataBase);
        source = source with { UserData = userData, UserDataBase = 0, ExcludedUserDataRegisters = MergedStageSystemScalars };
        var input = new ComputeInputInfo
        {
            ThreadsX = HullWaveLanes,
            ThreadsY = 1,
            ThreadsZ = 1,
            WaveSize = HullWaveLanes,
            LocalDataShareDwords = Math.Max(1u, vertex.HullResource2.LocalDataShareSize) * LocalDataShareGranuleDwords,
            NeedsLocalDataShareBarriers = true,
            IeeeMode = vertex.HullResource1.IeeeMode,
            FloatMode = vertex.HullResource1.FloatMode,
            Dx10Clamp = vertex.HullResource1.Dx10Clamp,
        };
        lock (_gate)
        {
            var pushDataCursor = 0u;
            var options = _hullOptions;
            options.ComputeInfo = input;
            options.ComputeSystemRegisters = null;
            options.Tessellation = hull;
            if (!TryPrepareProgram(source, options, ref pushDataCursor, out var handle, out var stage))
            {
                return new ComputeProgram { Available = false };
            }

            input.Stage = stage;
            return new ComputeProgram { Program = handle, Input = input };
        }
    }

    // A primitive-shader draw compiled as a ShaderStage.Mesh program (hosts with mesh shaders,
    // MeshShadersSupported). The depth target's aspects are not passed in, so a bound depth
    // target stands for its depth aspect.
    public GraphicsPrograms GetMeshGraphicsPrograms(
        RegisterBanks banks,
        UserConfigRegisters userConfig,
        ReadOnlySpan<ColorComponentMap> targetExportMapping,
        bool pixelActive,
        bool depthBound) =>
        GetMeshGraphicsPrograms(banks, userConfig, targetExportMapping, pixelActive, depthBound, taskLaunch: false);

    // taskLaunch: an automatic indirect draw launched by the task stage (MeshTaskLaunch).
    public GraphicsPrograms GetMeshGraphicsPrograms(
        RegisterBanks banks,
        UserConfigRegisters userConfig,
        ReadOnlySpan<ColorComponentMap> targetExportMapping,
        bool pixelActive,
        bool depthBound,
        bool taskLaunch) =>
        GetMeshGraphicsPrograms(banks.Shader.Vertex, banks.Shader.Pixel, banks.Context.ShaderInterface, banks.Context, userConfig,
            targetExportMapping, pixelActive, depthBound, taskLaunch);

    public GraphicsPrograms GetMeshGraphicsPrograms(
        VertexStageRegisters vertex,
        PixelStageRegisters pixel,
        ShaderInterfaceRegisters shaderInterface,
        ContextRegisters context,
        UserConfigRegisters userConfig,
        ReadOnlySpan<ColorComponentMap> targetExportMapping,
        bool pixelActive,
        bool depthBound,
        bool taskLaunch = false)
    {
        var mesh = PrepareMeshInput(vertex, shaderInterface, context, userConfig, taskLaunch);
        if (MeshDrawTrace.Active)
        {
            var group = userConfig.GeometryEngineControl;
            MeshDrawTrace.Write("plan", $"accepted={mesh is not null} primitive={userConfig.PrimitiveType} outputPrimitive={shaderInterface.GeometryOutputPrimitiveType} maxOutput={shaderInterface.MaxOutputPerSubgroup} maxVerticesOut={shaderInterface.GeometryMaxVerticesOut} group={group.PrimitiveGroupSize}/{group.VertexGroupSize} lds={vertex.GeometryResource2.LocalDataShareSize * 128} configuration={mesh}");
        }
        if (mesh is null)
        {
            var key = (context.ShaderStages, userConfig.PrimitiveType,
                shaderInterface.MaxOutputPerSubgroup, shaderInterface.GeometryMaxVerticesOut);
            lock (_gate)
            {
                if (_reportedMeshSkips.Add(key))
                {
                    var group = userConfig.GeometryEngineControl;
                    Console.Error.WriteLine($"[GPU][WARN][DRAW_SKIPPED] The mesh draw has an unsupported configuration: " +
                        $"shader=0x{vertex.ExportAddress:X16} primitive={userConfig.PrimitiveType} " +
                        $"outputPrimitive={shaderInterface.GeometryOutputPrimitiveType} " +
                        $"maxOutput={shaderInterface.MaxOutputPerSubgroup} " +
                        $"maxVerticesOut={shaderInterface.GeometryMaxVerticesOut} " +
                        $"group={group.PrimitiveGroupSize}/{group.VertexGroupSize} " +
                        $"ldsDwords={(uint)vertex.GeometryResource2.LocalDataShareSize * 128}.");
                }
            }
            return new GraphicsPrograms { Available = false };
        }
        if (ResolveMemoMode is ResolveMemoMode.On or ResolveMemoMode.Verify)
        {
            GraphicsResolveMemo.NoteUncacheable(ResolveMemoUncacheable.Mesh);
        }

        return GetGraphicsProgramsCore(vertex, pixel, shaderInterface, context, targetExportMapping, pixelActive, depthBound,
            depthAspectBound: depthBound, stencilAspectBound: false, triangleList: false, mesh);
    }

    private GraphicsPrograms GetGraphicsProgramsCore(
        VertexStageRegisters vertex,
        PixelStageRegisters pixel,
        ShaderInterfaceRegisters shaderInterface,
        ContextRegisters context,
        ReadOnlySpan<ColorComponentMap> targetExportMapping,
        bool pixelActive,
        bool depthBound,
        bool depthAspectBound,
        bool stencilAspectBound,
        bool triangleList,
        MeshDrawConfiguration? mesh)
    {
        using var waitScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.GraphicsResolveLockWait);
        using (EnterSharedScope(SharedReason.Other))
        {
            using var workScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPreparation);
            return ResolveGraphicsProgramsCore(vertex, pixel, shaderInterface, context, targetExportMapping, pixelActive, depthBound,
                depthAspectBound, stencilAspectBound, triangleList, mesh);
        }
    }

    private GraphicsPrograms ResolveGraphicsProgramsCore(
        VertexStageRegisters vertex,
        PixelStageRegisters pixel,
        ShaderInterfaceRegisters shaderInterface,
        ContextRegisters context,
        ReadOnlySpan<ColorComponentMap> targetExportMapping,
        bool pixelActive,
        bool depthBound,
        bool depthAspectBound,
        bool stencilAspectBound,
        bool triangleList,
        MeshDrawConfiguration? mesh,
        Gen5TessellationLaunch? tessellation = null,
        int startInstanceUserData = -1,
        bool nativeTessellation = false,
        int baseVertexUserData = -1)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPreparation);
        // A merged local/hull + domain draw on a host with tessellation stages (upstream's native path): the
        // export program is the domain stage, which reads no vertex tables, and the hull runs as that host's
        // hull stage (PrepareTessellationHull). Metal emulates tessellation instead (GetTessellationPrograms).
        var tessellated = nativeTessellation && tessellation is null && mesh is null && _host.TessellationStagesSupported &&
            IsMergedTessellationMask(context.ShaderStages);
        Gen5TessellationInfo? tessellationInfo = null;
        if (tessellated)
        {
            if (!Gen5TessellationInfo.TryDecode(shaderInterface.TessellationFactorParameter, out var configuration, out var tessellationError))
                throw GpuFatal.Fatal(tessellationError);
            tessellationInfo = configuration;
        }

        var vertexStageKind = tessellated ? ShaderStage.TessellationEvaluation : mesh is null ? ShaderStage.Vertex : ShaderStage.Mesh;
        var vertexSource = PrepareSource(
            vertex.ExportAddress, vertexStageKind,
            tessellated ? "domain" : mesh is null ? "vertex" : "mesh", vertex.GeometryUserScalars, vertex.GeometryResource2.UserScalarCount,
            probeWrittenRegisters: true, VertexUserDataBase, ref _vertexUserData, vertex.GeometryUserDataAddress);
        var vertexInfo = tessellated ? new VertexInputInfo { PositionExportControl = shaderInterface.VertexOutputControl }
            : mesh is null ? PrepareVertexInput(vertexSource, shaderInterface, context) : new VertexInputInfo();
        vertexInfo.StartInstanceUserData = mesh is null && !tessellated ? startInstanceUserData : -1;
        vertexInfo.BaseVertexUserData = mesh is null && !tessellated ? baseVertexUserData : -1;
        vertexInfo.IeeeMode = vertex.GeometryResource1.IeeeMode;
        vertexInfo.FloatMode = vertex.GeometryResource1.FloatMode;
        vertexInfo.Dx10Clamp = vertex.GeometryResource1.Dx10Clamp;
        ShaderSource? pixelSource = null;
        PixelInputInfo? pixelInfo = null;
        Gen5ShaderProgram? decodedPixelProgram = null;
        Gen5PixelOutputBinding[] pixelOutputs = [];
        var dropUnwritten = false;
        var attributeCount = 0u;
        if (pixelActive)
        {
            pixelSource = PrepareSource(
                pixel.Address, ShaderStage.Pixel, "pixel", pixel.UserScalars, pixel.Resource2.UserScalarCount,
                probeWrittenRegisters: true, userDataBase: 0, ref _pixelUserData);
            if (RenderTrace.Enabled)
                RenderTrace.Write($"PixelDrawState shader=0x{pixelSource.Address:X16} hash=0x{pixelSource.Hash:X16} " +
                    $"user_data=[{string.Join(",", pixelSource.UserData.Select(word => $"{word:X8}"))}]");
            using var pixelProfile = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.PixelInputResolution);
            var pixelProgram = _programs.Decode(pixelSource);
            decodedPixelProgram = pixelProgram;
            var attributes = GetPixelAttributes(pixelProgram);
            attributeCount = attributes.Length == 0 ? 0 : attributes[^1] + 1;
            var inputCount = shaderInterface.PixelInputControl & 0x3Fu;
            if (inputCount == 0)
            {
                inputCount = Math.Min(attributeCount, MaxPixelInputs);
            }

            if (inputCount > MaxPixelInputs)
            {
                throw GpuFatal.Fatal($"The pixel program declares too many interpolators: shader=0x{pixel.Address:X16} count={inputCount}.");
            }

            // Each bound target is resolved once for the draw: the three questions below ask about the same
            // targets, and every resolution hands back a request of several hundred bytes.
            Span<uint> boundSlots = stackalloc uint[(int)ContextRegisters.ColorTargetCount];
            Span<GpuExtent2D> boundExtents = stackalloc GpuExtent2D[(int)ContextRegisters.ColorTargetCount];
            var boundCount = ResolveBoundSlots(context, boundSlots, boundExtents);
            boundSlots = boundSlots[..boundCount];
            boundExtents = boundExtents[..boundCount];
            dropUnwritten = depthBound ||
                LeavesSingleDualSourceTarget(context, targetExportMapping, pixelProgram.PixelColorExportMasks, boundSlots) ||
                UnwrittenTargetBoundsRenderArea(context, targetExportMapping, pixelProgram.PixelColorExportMasks, boundSlots, boundExtents);
            Span<byte> outputModes = stackalloc byte[PixelInputInfo.TargetCount];
            Span<ColorComponentMap> outputMappings = stackalloc ColorComponentMap[PixelInputInfo.TargetCount];
            pixelOutputs = ResolveBoundTargets(context, targetExportMapping, dropUnwritten ? pixelProgram.PixelColorExportMasks : null,
                boundSlots, outputModes, outputMappings);
            pixelInfo = PixelStageInputResolver.Resolve(_context, pixelSource.Registered, shaderInterface, outputModes, outputMappings, inputCount,
                depthAspectBound, stencilAspectBound, _pixelInputArrays, semantics: InputSemantics(pixelSource.Registered));
            // SPI_PS_INPUT_CNTL can map an input to any parameter export, beyond the input count;
            // the vertex program must declare every location the pixel program reads.
            attributeCount = Math.Max(attributeCount, ReadVertexOutputCount(attributes, pixelInfo));
            pixelInfo.IeeeMode = pixel.Resource1.IeeeMode;
            pixelInfo.FloatMode = pixel.Resource1.FloatMode;
            pixelInfo.Dx10Clamp = pixel.Resource1.Dx10Clamp;
            pixelInfo.GuestWave32 = (shaderInterface.PixelInputControl & (1u << 15)) != 0;
        }

        // A ShaderStage.Mesh draw replaces the vertex stage outright; the primitive-shader
        // workgroup path is for the vertex stage only.
        var primitiveShader = mesh is null && !tessellated ? PrimitiveShaderStageInfo(vertex, shaderInterface, context.ShaderStages) : null;
        if (primitiveShader is { } ngg && vertexSource.Registered.IsFused)
        {
            // The GS half of a fused program assembles its input primitives from LDS, so the
            // mesh prologue needs to know there is one and how much each primitive can emit.
            primitiveShader = ngg with { GeometryVerticesOut = Math.Max(shaderInterface.GeometryMaxVerticesOut, 1u) };
        }

        // A pixel program that reads the three vertex values of an input on a host that cannot
        // declare per-vertex pixel inputs gets host interpolants instead when it only
        // interpolates them by hand, or else them from the primitive shader, run as a mesh
        // function whose primitives carry them. The mesh path assembles a triangle list.
        var perVertexParameters = 0u;
        Gen5ManualInterpolationPlan? manualInterpolation = null;
        if (decodedPixelProgram is not null && pixelInfo is not null && pixelSource is not null &&
            !_host.PerVertexPixelInputsSupported)
        {
            perVertexParameters = ResolvePerVertexInputs(pixelSource, decodedPixelProgram, pixelInfo,
                shaderInterface.PixelInputAddress, attributeCount, primitiveShader is not null && triangleList,
                out manualInterpolation);
        }

        var vertexCompileSource = primitiveShader is null
            ? vertexSource
            : WithGeometryUserDataAddress(vertexSource, vertex.GeometryUserDataAddress);
        ShaderProgram vertexProgram;
        ShaderProgram pixelProgramHandle = default;
        ShaderStageResources vertexStage;
        ShaderStageResources pixelStage = default;
        TessellationDrawPrograms? tessellationPrograms = null;
        profileScope.SwitchPhase(RenderPhaseProfile.Phase.ProgramSharedGateWait);
        lock (_gate)
        {
            profileScope.SwitchPhase(RenderPhaseProfile.Phase.ProgramPreparation);
            using var lockedScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPrepareLocked);
            var pushDataCursor = 0u;
            if (pixelSource is not null)
            {
                // Each stage refills its own reusable options; a miss keeps a copy.
                var pixelOptions = _pixelOptions;
                pixelOptions.PixelInfo = pixelInfo;
                pixelOptions.PixelOutputs = pixelOutputs;
                pixelOptions.PixelInputEnable = shaderInterface.PixelInputEnable;
                pixelOptions.PixelInputAddress = shaderInterface.PixelInputAddress;
                pixelOptions.PerVertexInputsFromPrimitive = perVertexParameters != 0;
                pixelOptions.ManualInterpolation = manualInterpolation is not null;
                if (!TryPrepareProgram(
                    pixelSource,
                    pixelOptions,
                    ref pushDataCursor,
                    out pixelProgramHandle,
                    out pixelStage))
                    return new GraphicsPrograms { Available = false };
            }

            var vertexOptions = _vertexOptions;
            vertexOptions.VertexInfo = vertexInfo;
            vertexOptions.RequiredVertexOutputCount = (int)attributeCount;
            vertexOptions.PrimitiveShader = primitiveShader;
            vertexOptions.Tessellation = tessellation;
            vertexOptions.TessellationInfo = tessellationInfo;
            vertexOptions.TessellationHull = null;
            vertexOptions.MeshPerVertexParameterMask = perVertexParameters;
            vertexOptions.VertexInterpolants = manualInterpolation is null ? [] : EncodedInterpolantsOf(manualInterpolation);
            vertexOptions.MeshInfo = mesh;
            vertexOptions.MeshOutputLocationMask = mesh is null ? null : pixelSource is null ? 0u : pixelStage.Program?.InputLocationMask;
            if (!TryPrepareProgram(
                vertexCompileSource,
                vertexOptions,
                ref pushDataCursor,
                out vertexProgram,
                out vertexStage))
                return new GraphicsPrograms { Available = false };

            vertexInfo.Stage = vertexStage;
            if (tessellated)
            {
                tessellationPrograms = PrepareTessellationHull(vertex, shaderInterface, tessellationInfo!.Value);
                var bridge = _programs.GetTessellationBridge(vertexStage.Program!.Bindings!, tessellationInfo.Value.Domain);
                vertexInfo = new VertexInputInfo
                {
                    PositionExportControl = vertexInfo.PositionExportControl,
                    IeeeMode = vertexInfo.IeeeMode,
                    FloatMode = vertexInfo.FloatMode,
                    Dx10Clamp = vertexInfo.Dx10Clamp,
                    Stage = vertexStage,
                    Tessellation = new(bridge.Control, vertexProgram, tessellationPrograms.HullConfiguration.InputControlPoints),
                };
                vertexProgram = bridge.Vertex;
            }
        }

        if (pixelInfo is not null)
        {
            pixelInfo.Stage = pixelStage;
        }

        return new GraphicsPrograms
        {
            Vertex = vertexProgram,
            Pixel = pixelProgramHandle,
            VertexInput = vertexInfo,
            MeshInput = mesh,
            Tessellation = tessellationPrograms,
            PixelInput = pixelInfo ?? InactivePixelInput,
            DropUnwrittenColorTargets = dropUnwritten,
            PositionStream = mesh is null && !tessellated ? FindPositionStream(vertexInfo) : null,
        };
    }

    // The task stage's payload (SpirvMeshTask): three words, rounded up.
    private const uint MeshTaskPayloadBytes = 16;

    private MeshDrawConfiguration? PrepareMeshInput(VertexStageRegisters vertex,
        ShaderInterfaceRegisters shaderInterface, ContextRegisters context,
        UserConfigRegisters userConfig, bool taskLaunch = false)
    {
        var planned = MeshDrawPlanner.Create(vertex, shaderInterface, context, userConfig,
            _host.MeshLimits, _host.MeshSubgroupSize);
        if (planned is not { } configuration) return null;
        var registered = _registry.Require(vertex.ExportAddress, "mesh");
        return new MeshDrawConfiguration
        {
            Execution = configuration.Execution,
            Geometry = configuration.Geometry with
            {
                IeeeMode = vertex.GeometryResource1.IeeeMode,
                ScratchDwords = Math.Max(registered.ScratchDwords, registered.ContinuationScratchDwords),
                // The program's half of the viewport transform, as for a vertex program.
                ClipSpace = HostViewport(context).ClipSpace,
                // The task payload shares the mesh workgroup's payload-and-shared memory with the planned shared memory.
                TaskLaunched = taskLaunch &&
                    (configuration.Geometry.LocalDataShareDwords + configuration.Geometry.OutputVertexCapacity + 5) * 4 + MeshTaskPayloadBytes <=
                    _host.MeshLimits.MaxSharedMemoryBytes,
            },
        };
    }

    // The pixel input of a draw without a pixel stage. Nothing writes a resolved draw's inputs
    // but the resolver, which sets the stage of a pixel input it resolved, so draws share one.
    private static readonly PixelInputInfo InactivePixelInput = new()
    {
        InterpolatorSettings = new uint[PixelInputInfo.InterpolatorCount],
        TargetOutputModes = new byte[PixelInputInfo.TargetCount],
        TargetExportMappings = new ColorComponentMap[PixelInputInfo.TargetCount],
    };

    // A plan is cached per program and inputs (ResolvePerVertexInputs), and so are its encoded interpolants.
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Gen5ManualInterpolationPlan, uint[]> EncodedInterpolants = new();

    private static uint[] EncodedInterpolantsOf(Gen5ManualInterpolationPlan plan) =>
        EncodedInterpolants.GetValue(plan, static plan => plan.Interpolants.Select(interpolant => interpolant.Encoded).ToArray());

    // The vertex tables of the draw, the clip-space transform when clipping is off and the
    // draw's clip-space conventions.
    private VertexInputInfo PrepareVertexInput(ShaderSource source, ShaderInterfaceRegisters shaderInterface, ContextRegisters context)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.VertexInputResolution);
        // The program's half of the viewport transform the draw state sets the rest of.
        var clipSpace = HostViewport(context).ClipSpace;
        return VertexInputResolver.ResolveVertexInputs(_context, source.Registered, source.UserData,
            shaderInterface.VertexOutputControl, clipSpace, _vertexTables, source.Hash);
    }

    private HostViewportTransform HostViewport(ContextRegisters context)
    {
        var limits = _host.Limits;
        return HostViewportTransform.Resolve(context, limits.MaxViewportWidth, limits.MaxViewportHeight);
    }

    // A frontend's pixel shader carries its own input semantics (upstream's InputSemantics); an AGC
    // shader's are read from its header by the resolver, as before.
    private uint[]? InputSemantics(RegisteredShader shader)
    {
        if (shader.Frontend is null)
        {
            return null;
        }

        var words = new uint[Math.Min(shader.InputSemanticsCount, (uint)PixelInputInfo.InterpolatorCount)];
        return words.Length == 0 || _context.Memory.TryRead(shader.InputSemanticsAddress, MemoryMarshal.AsBytes(words.AsSpan())) ? words : null;
    }

    // The interpolated attributes a pixel program reads, ascending. A program never changes, and this
    // was rebuilt with LINQ on every draw.
    private sealed record PixelProgramAnalysis(uint[] Attributes, uint PerVertexAttributes);

    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Gen5ShaderProgram, PixelProgramAnalysis> InterpolatedAttributeCache = new();

    internal uint[] GetPixelAttributes(Gen5ShaderProgram program) => InterpolatedAttributesOf(program);

    internal static uint[] InterpolatedAttributesOf(Gen5ShaderProgram program) => PixelAnalysisOf(program).Attributes;

    private static PixelProgramAnalysis PixelAnalysisOf(Gen5ShaderProgram program) =>
        InterpolatedAttributeCache.GetValue(program, static program => new PixelProgramAnalysis(
            program.Instructions.Select(static instruction => instruction.Control)
                .OfType<Gen5InterpolationControl>()
                .Select(static control => control.Attribute)
                .Distinct().Order().ToArray(),
            // Empty controls map each input to itself: this is the instruction-only
            // attribute mask. Draw controls still map those attributes on every use.
            Gen5PixelInputMapping.PerVertexParameterMask(program, [])));

    // One past the highest parameter location the pixel program reads, resolved as its translator does.
    internal static uint ReadVertexOutputCount(ReadOnlySpan<uint> attributes, PixelInputInfo info)
    {
        if (attributes.Length == 0)
        {
            return 0;
        }

        Span<uint> controls = stackalloc uint[32];
        for (var index = 0u; index < (uint)controls.Length; index++)
        {
            controls[(int)index] = index < info.InputCount && index < (uint)info.InterpolatorSettings.Length
                ? info.InterpolatorSettings[index]
                : (uint)index;
        }

        // Attributes are distinct and below 32, so a program never has more than 32 of them.
        Span<uint> locations = attributes.Length <= 32 ? stackalloc uint[attributes.Length] : new uint[attributes.Length];
        Gen5PixelInputMapping.ResolveLocations(controls, attributes, locations);
        var highest = 0u;
        foreach (var location in locations)
        {
            highest = Math.Max(highest, location);
        }

        return highest + 1;
    }

    // The attribute list is ascending, so its last entry is the highest.
    internal static uint InterpolatedAttributeCount(Gen5ShaderProgram program)
    {
        var attributes = InterpolatedAttributesOf(program);
        return attributes.Length == 0 ? 0 : attributes[^1] + 1;
    }

    // The bound colour slots in order; each output mode names the kind the pixel program exports.
    // With a depth target bound, a slot the draw never writes is dropped here and from the
    // attachments alike (RenderExecutor.DropUnwrittenColorTargets): unwrittenExportMasks is the
    // pixel program's export mask then, null otherwise.
    private Gen5PixelOutputBinding[] ResolveBoundTargets(ContextRegisters context, ReadOnlySpan<ColorComponentMap> targetExportMapping,
        uint? unwrittenExportMasks, ReadOnlySpan<uint> boundSlots, Span<byte> outputModes, Span<ColorComponentMap> outputMappings)
    {
        var outputs = _boundOutputs;
        outputs.Clear();
        var location = 0u;
        var routing = PixelExportRouting.Routing.Of(context.ShaderInterface);
        foreach (var slot in boundSlots)
        {
            if (unwrittenExportMasks is { } exportMasks &&
                RenderExecutor.IsUnwrittenColorTarget(context, in routing, slot, targetExportMapping[(int)slot], exportMasks))
            {
                continue;
            }

            var words = context.ColorTargets[slot];
            if (!_host.TryResolveColorOutput((uint)words.Layout, (uint)words.NumberType, (uint)words.Order, out var kind, out var mapping))
            {
                throw GpuFatal.Fatal($"The color target format has no pixel output kind: slot={slot} layout={(uint)words.Layout} numberType={(uint)words.NumberType} order={(uint)words.Order}.");
            }

            // A bound slot that no export reaches keeps its contents: its color write mask is zero.
            var exportTarget = routing.ExportForSlot(slot);
            // The high nibble keys the compiled program by the export that feeds the slot.
            outputModes[(int)slot] = (byte)(((uint)kind + 1) | (uint)((exportTarget + 1) << 4));
            outputMappings[(int)slot] = new ColorComponentMap(mapping.Packed);
            // No EXP target reaches 8 and above, so an unfed slot is declared but never written.
            outputs.Add(new Gen5PixelOutputBinding(slot, location++, kind, mapping)
            {
                ExportTarget = exportTarget >= 0 ? (uint)exportTarget : ContextRegisters.ColorTargetCount + slot,
                ExportFormat = ExportFormatOf(context.ShaderInterface, exportTarget),
                ClampToUnit = ClampsFloatOutput(context, slot),
            });
        }

        if (context.BlendControls[0].UsesSecondarySource && (context.ColorTargets[0].Info & (1u << 16)) == 0 &&
            HasGuestSlotZero(outputs))
        {
            if (outputs.Count != 1 || outputs[0].Kind != Gen5PixelOutputKind.Float)
            {
                throw GpuFatal.Fatal("Dual-source blending requires one floating-point color output.");
            }

            // The second export feeds target zero's blend unit, not another attachment.
            outputs[0] = outputs[0] with { ExportTarget = 0, ExportFormat = ExportFormatOf(context.ShaderInterface, 0) };
            outputs.Add(outputs[0] with
            {
                GuestSlot = 1,
                ExportTarget = 1,
                BlendSourceIndex = 1,
                ExportFormat = ExportFormatOf(context.ShaderInterface, 1),
            });
            outputModes[0] = (byte)(((uint)outputs[0].Kind + 1) | 0x10u);
            outputModes[1] = DualSourceOutputMode;
        }

        if (System.Runtime.InteropServices.CollectionsMarshal.AsSpan(outputs).SequenceEqual(_lastBoundOutputs))
        {
            return _lastBoundOutputs;
        }

        return _lastBoundOutputs = outputs.ToArray();
    }

    private const byte DualSourceOutputMode = 0xFF;

    // CB_COLOR_INFO.BLEND_CLAMP on a FLOAT target clamps the colour to [0, 1]; normalized
    // targets clamp by their encoding. Without blending the program saturates its output.
    // With blending the CB also clamps the destination and the blend result, which the host's
    // fixed-function blend cannot do: documented deviation, only the source is clamped.
    private static bool ClampsFloatOutput(ContextRegisters context, uint slot)
    {
        const uint BlendClamp = 1u << 15, BlendBypass = 1u << 16, NumberTypeFloat = 7;
        ref readonly var words = ref context.ColorTargets[slot];
        if ((words.Info & BlendClamp) == 0 || (uint)words.NumberType != NumberTypeFloat)
        {
            return false;
        }

        if (context.BlendControls[slot].Enable && (words.Info & BlendBypass) == 0 &&
            !ShaderCompilerWarnings.WasReported("blend-clamp-blended-float"))
        {
            ShaderCompilerWarnings.WarnOnce(
                "blend-clamp-blended-float",
                $"BLEND_CLAMP on a blended float color target clamps the source only, not the blend result: " +
                $"slot={slot} layout={(uint)words.Layout}.");
        }

        return true;
    }

    // SPI_SHADER_COL_FORMAT of one export. With no format written for any export, the
    // routing pairs exports and slots by number (PixelExportRouting) and the translator reads
    // each export by its own layout.
    private static Gen5ColorExportFormat? ExportFormatOf(ShaderInterfaceRegisters registers, int exportTarget)
    {
        if (exportTarget < 0 || exportTarget >= registers.TargetOutputModes.Length)
        {
            return null;
        }

        foreach (var mode in registers.TargetOutputModes)
        {
            if ((mode & 0xF) != 0)
            {
                var format = registers.TargetOutputModes[exportTarget] & 0xFu;
                return format <= (uint)Gen5ColorExportFormat.Abgr32 ? (Gen5ColorExportFormat)format : null;
            }
        }

        return null;
    }

    private static bool HasGuestSlotZero(List<Gen5PixelOutputBinding> outputs)
    {
        foreach (var output in outputs)
        {
            if (output.GuestSlot == 0)
            {
                return true;
            }
        }

        return false;
    }

    // BLEND_SRC1_COLOR, BLEND_ONE_MINUS_SRC1_COLOR, BLEND_SRC1_ALPHA, BLEND_ONE_MINUS_SRC1_ALPHA.
    internal static bool UsesSecondBlendSource(in BlendRegisters blend)
    {
        return blend.UsesSecondarySource;
    }

    // Marks a pixel output mode whose program also writes its alpha to the dual-source output.
    internal const byte DualSourceAlphaOutputMode = 0x08;

    // The hardware blends in shader component order before the component swap lays the result out in memory,
    // but the host image keeps the memory layout and the program writes the swapped vector.
    // When that swap moves the shader's alpha out of host alpha (ABGR and ARGBorders),
    // SRC_ALPHA would read the wrong channel; the program exports the real alpha as the second source and
    // the blend uses SRC1_ALPHA instead.
    private static bool NeedsDualSourceAlpha(ContextRegisters context, uint slot, ColorComponentMap mapping) =>
        mapping.Map(3) != 3 && BlendsWithSourceAlpha(context, slot);

    // The mapping-independent half of NeedsDualSourceAlpha:
    // an enabled, not bypassed blend that reads the source alpha.
    private static bool BlendsWithSourceAlpha(ContextRegisters context, uint slot)
    {
        var blend = context.BlendControls[slot];
        if (!blend.Enable || ((context.ColorTargets[slot].Info >> 16) & 0x1) != 0)
        {
            return false;
        }

        var separate = blend.SeparateAlpha;
        var alphaSource = separate ? blend.AlphaSourceFactor : blend.ColorSourceFactor;
        var alphaDestination = separate ? blend.AlphaDestinationFactor : blend.ColorDestinationFactor;
        return UsesSourceAlpha(blend.ColorSourceFactor) || UsesSourceAlpha(blend.ColorDestinationFactor) ||
            UsesSourceAlpha(alphaSource) || UsesSourceAlpha(alphaDestination);
    }

    private static bool UsesSourceAlpha(byte factor) => factor is 4 or 5;

    // Dual-source blending allows a single attachment,
    // but post passes often keep another target bound that the program never writes.
    // Dropping such targets (as depth passes already do) is invisible to the guest and
    // lets the remaining target blend correctly. The executor drops the same targets through
    // GraphicsPrograms.DropUnwrittenColorTargets.
    private bool LeavesSingleDualSourceTarget(
        ContextRegisters context,
        ReadOnlySpan<ColorComponentMap> mappingBySlot,
        uint pixelColorExportMasks,
        ReadOnlySpan<uint> boundSlots)
    {
        // Most draws blend no target with source alpha.
        if (!_host.SupportsDualSourceBlend || !AnyTargetBlendsWithSourceAlpha(context))
        {
            return false;
        }

        if (boundSlots.Length < 2)
        {
            return false;
        }

        var written = 0;
        var writtenSlot = 0u;
        var routing = PixelExportRouting.Routing.Of(context.ShaderInterface);
        foreach (var slot in boundSlots)
        {
            if (!RenderExecutor.IsUnwrittenColorTarget(context, in routing, slot, mappingBySlot[(int)slot], pixelColorExportMasks))
            {
                written++;
                writtenSlot = slot;
            }
        }

        return written == 1 && NeedsDualSourceAlpha(context, writtenSlot, mappingBySlot[(int)writtenSlot]);
    }

    private static bool AnyTargetBlendsWithSourceAlpha(ContextRegisters context)
    {
        for (var slot = 0u; slot < ContextRegisters.ColorTargetCount; slot++)
        {
            if (BlendsWithSourceAlpha(context, slot))
            {
                return true;
            }
        }

        return false;
    }

    private static bool UnwrittenTargetBoundsRenderArea(
        ContextRegisters context,
        ReadOnlySpan<ColorComponentMap> mappingBySlot,
        uint pixelColorExportMasks,
        ReadOnlySpan<uint> boundSlots,
        ReadOnlySpan<GpuExtent2D> boundExtents)
    {
        if (boundSlots.Length < 2)
        {
            return false;
        }

        uint writtenWidth = 0, writtenHeight = 0;
        uint unwrittenWidth = uint.MaxValue, unwrittenHeight = uint.MaxValue;
        var routing = PixelExportRouting.Routing.Of(context.ShaderInterface);
        for (var index = 0; index < boundSlots.Length; index++)
        {
            var slot = boundSlots[index];
            var extent = boundExtents[index];
            if (RenderExecutor.IsUnwrittenColorTarget(context, in routing, slot, mappingBySlot[(int)slot], pixelColorExportMasks))
            {
                unwrittenWidth = Math.Min(unwrittenWidth, extent.Width);
                unwrittenHeight = Math.Min(unwrittenHeight, extent.Height);
            }
            else
            {
                writtenWidth = Math.Max(writtenWidth, extent.Width);
                writtenHeight = Math.Max(writtenHeight, extent.Height);
            }
        }

        return writtenWidth > unwrittenWidth || writtenHeight > unwrittenHeight;
    }

    // The slots with a color target, and their extents, before any are dropped as unwritten.
    // Each target is resolved here and nowhere else in the lookup.
    private static int ResolveBoundSlots(ContextRegisters context, Span<uint> slots, Span<GpuExtent2D> extents)
    {
        var count = 0;
        for (var slot = 0u; slot < ContextRegisters.ColorTargetCount; slot++)
        {
            if (slot != 0 && (context.RenderTargetMaskForSlot(slot) == 0 || context.ColorTargets[slot].BaseAddress == 0))
            {
                continue;
            }

            if (ColorTargetResolver.Resolve(context, slot, 0, ignoreTargetMask: false, out _) is not { } resolution)
            {
                continue;
            }

            slots[count] = slot;
            extents[count++] = resolution.Extent;
        }

        return count;
    }

    // SRC_ALPHA / ONE_MINUS_SRC_ALPHA read the dual-source alpha instead.
    private static byte ToDualSourceFactor(byte factor) => factor switch
    {
        4 => 17,
        5 => 18,
        _ => factor,
    };

    private static VertexPositionStream? FindPositionStream(VertexInputInfo info)
    {
        foreach (var attribute in info.Attributes)
        {
            if (Gfx10UnifiedFormat.TryDecode(attribute.Descriptor.Format, out var dataFormat, out var numberFormat) &&
                dataFormat == PositionDataFormat && numberFormat == PositionNumberFormat && attribute.BufferIndex >= 0)
            {
                var buffer = info.Buffers[attribute.BufferIndex];
                return new VertexPositionStream(buffer.Address, buffer.Stride, attribute.OffsetBytes);
            }
        }

        return null;
    }

    // SHARPEMU_COMPUTE_TICKETS: worker rebinds that direct dispatches take ahead of the command thread, or null.
    internal ComputeTickets? Tickets { get; set; }

    public ComputeProgram GetComputeProgram(RegisterBanks banks, uint dispatchInitiator, uint dimensionX, uint dimensionY, uint dimensionZ) =>
        GetComputeProgram(banks, dispatchInitiator, dimensionX, dimensionY, dimensionZ, indirect: false, retry: null);

    public ComputeProgram GetComputeProgram(RegisterBanks banks, uint dispatchInitiator, uint dimensionX, uint dimensionY, uint dimensionZ, bool indirect) =>
        GetComputeProgram(banks, dispatchInitiator, dimensionX, dimensionY, dimensionZ, indirect, retry: null);

    // A dispatch's program. With retry (a dispatch that may suspend on its pipeline) the program the retry kept is handed
    // back while its reads still hold, as GetComputeProgram(…, ComputeProgramRetry) does, and nothing is predicted: the
    // dispatch runs again. Without it, a dispatch whose registers and user data are those of a dispatch this thread
    // resolved keeps that program after its resources revalidate, else rebinds that program's entry to its user data,
    // else resolves; its writes are predicted on the CPU when the kernel is evaluable (ShaderPipelineCache.KernelEvaluation).
    // gpuThreadCounts: the thread counts are in GPU memory (zero here); the program reads its thread limits from shader data
    // the GPU writes (ComputeInputInfo.IndirectThreadCounts).
    public ComputeProgram GetComputeProgram(RegisterBanks banks, uint dispatchInitiator, uint dimensionX, uint dimensionY, uint dimensionZ, bool indirect,
        bool gpuThreadCounts) =>
        GetComputeProgram(banks, dispatchInitiator, dimensionX, dimensionY, dimensionZ, indirect, retry: null, gpuThreadCounts);

    // Each public form below has exactly the interface member's signature: IShaderPipelineProvider gives these members
    // default bodies, so a form with one more (even optional) parameter would not implement its member and the default
    // would answer for it (InterfaceDefaultShadowTests).
    public ComputeProgram GetComputeProgram(RegisterBanks banks, uint dispatchInitiator, uint dimensionX, uint dimensionY, uint dimensionZ, bool indirect,
        ComputeProgramRetry? retry) =>
        GetComputeProgram(banks, dispatchInitiator, dimensionX, dimensionY, dimensionZ, indirect, retry, gpuThreadCounts: false);

    private ComputeProgram GetComputeProgram(RegisterBanks banks, uint dispatchInitiator, uint dimensionX, uint dimensionY, uint dimensionZ, bool indirect,
        ComputeProgramRetry? retry, bool gpuThreadCounts) =>
        GetComputeProgramKept(banks, dispatchInitiator, dimensionX, dimensionY, dimensionZ, indirect | gpuThreadCounts, retry, ahead: false, out _,
            gpuThreadCounts)!;

    // The same resolution on a thread ahead of the dispatch (ParallelProgramPrefetch): nothing is consumed or submitted
    // (GetComputeProgramCore's ahead form), and a program whose writes the dispatch would predict on the CPU, or that a
    // frontend may turn into an image copy, or that is unavailable, is refused, so the dispatch resolves it as before.
    // The kept programs it reads and updates are this thread's own.
    public ComputeProgram? ResolveComputeProgramAhead(RegisterBanks banks, uint dispatchInitiator, uint dimensionX, uint dimensionY,
        uint dimensionZ, bool indirect, out ComputePrefetchRefusal refusal) =>
        // A dispatch whose thread counts the GPU writes is never resolved ahead (RenderExecutor.Compute).
        GetComputeProgramKept(banks, dispatchInitiator, dimensionX, dimensionY, dimensionZ, indirect, retry: null, ahead: true,
            out refusal, gpuThreadCounts: false);

    private ComputeProgram? GetComputeProgramKept(RegisterBanks banks, uint dispatchInitiator, uint dimensionX, uint dimensionY, uint dimensionZ,
        bool indirect, ComputeProgramRetry? retry, bool ahead, out ComputePrefetchRefusal refusal, bool gpuThreadCounts)
    {
        refusal = ComputePrefetchRefusal.None;
        var compute = banks.Shader.Compute;
        var sharedReason = retry is not null ? SharedReason.ComputeRetry : SharedReason.ComputeResolve;
        // With shards, a dispatch whose kept program holds or rebinds off the shared tier never takes it (the kept programs
        // are this thread's own); the worker's form only where it predicts nothing (WithPrediction returns the program).
        if (_shards.Length > 1 && UnsharedCompute && retry is null && !Monitor.IsEntered(_resolveGate))
        {
            if (!ahead && PredictsKernels)
            {
                sharedReason = SharedReason.ComputeWorker;
            }
            else if (TryGetComputeProgramUnshared(compute, dispatchInitiator, dimensionX, dimensionY, dimensionZ, indirect, ahead, gpuThreadCounts,
                         out var unshared, out refusal, ref sharedReason))
            {
                return unshared;
            }
        }

        using var waitScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ComputeResolveLockWait);
        EnterShared(sharedReason);
        try
        {
            using var workScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPreparation);
            if (retry is not null)
            {
                return GetComputeProgramCore(compute, dispatchInitiator, dimensionX, dimensionY, dimensionZ, retry, out _, out _, out _,
                    gpuThreadCounts: gpuThreadCounts);
            }

            if (ahead && _registry.HasFrontend)
            {
                refusal = ComputePrefetchRefusal.Frontend;
                return null;
            }

            var registrations = ShaderHeaderRegistry.Registrations;
            ClaimKeptPrograms();
            var all = _keptComputes ??= new Dictionary<ulong, List<KeptComputeProgram>>();
            if (!all.TryGetValue(compute.Address, out var keptList))
            {
                keptList = new List<KeptComputeProgram>(KeptComputesPerShader);
                all.Add(compute.Address, keptList);
            }

            KeptComputeProgram? kept = null;
            foreach (var candidate in keptList)
            {
                if (candidate.Registrations == registrations && candidate.DispatchInitiator == dispatchInitiator &&
                    candidate.GpuThreadCounts == gpuThreadCounts && SameComputeShape(candidate.Registers, compute))
                {
                    kept = candidate;
                    break;
                }
            }

            if (kept is not null && !DrawReuseDisabled && !_registry.HasFrontend)
            {
                if (kept.Program is { } previous && kept.Registers is { } previousRegisters && kept.DimensionX == dimensionX &&
                    kept.DimensionY == dimensionY && kept.DimensionZ == dimensionZ &&
                    SameUserScalars(previousRegisters.UserScalars, compute.UserScalars, compute.UserScalarCount) && _programs.Revalidate(previous.Input.Stage))
                {
                    _computeKept.Increment();
                    t_resolvePath = ResolvePath.ComputeKept;
                    return ahead
                        ? Ahead(previous, kept.Kernel, kept.Source, out refusal)
                        : WithPrediction(previous, kept.Kernel, kept.Source, kept.SystemRegisters, compute, dimensionX, dimensionY, dimensionZ, indirect);
                }

                if (kept.Entry is { } entry && !kept.RecheckDue && ++kept.Rebinds % IdentityRecheckPeriod != 0)
                {
                    _computeRebound.Increment();
                    t_resolvePath = ResolvePath.ComputeRebound;
                    // A direct dispatch takes the rebind a compute ticket worker made ahead of it, while the ticket still holds.
                    var rebound = !ahead && !indirect && Tickets is { } tickets &&
                        tickets.TryTake(kept, compute, dispatchInitiator, dimensionX, dimensionY, dimensionZ) is { } ticket
                        ? TakeComputeTicket(tickets, ticket, kept, entry, compute, dispatchInitiator, dimensionX, dimensionY, dimensionZ)
                        : RebindComputeProgram(kept, entry, compute, dispatchInitiator, dimensionX, dimensionY, dimensionZ, ahead);
                    KeepComputeProgram(kept, rebound, compute, dimensionX, dimensionY, dimensionZ);
                    return ahead
                        ? Ahead(rebound, kept.Kernel, kept.Source, out refusal)
                        : WithPrediction(rebound, kept.Kernel, kept.Source, kept.SystemRegisters, compute, dimensionX, dimensionY, dimensionZ, indirect);
                }
            }

            _computeResolved.Increment();
            t_resolvePath = ResolvePath.ComputeResolve;
            var program = GetComputeProgramCore(compute, dispatchInitiator, dimensionX, dimensionY, dimensionZ, retry: null,
                out var source, out var systemRegisters, out var decoded, ahead, gpuThreadCounts);
            if (program.ImageCopy is not null)
                return program;
            if (kept is null)
            {
                if (keptList.Count == KeptComputesPerShader)
                {
                    keptList.RemoveAt(0);
                }

                kept = new KeptComputeProgram { Registrations = registrations, DispatchInitiator = dispatchInitiator, GpuThreadCounts = gpuThreadCounts };
                keptList.Add(kept);
            }

            var usable = program.Available && !program.Consumed && source is not null;
            kept.Source = source;
            kept.RecheckDue = false;
            kept.SystemRegisters = systemRegisters;
            kept.Kernel = decoded is not null && usable ? KernelFor(decoded) : null;
            kept.Entry = usable ? ShaderProgramCache.LastEntry : null;
            KeepComputeProgram(kept, program, compute, dimensionX, dimensionY, dimensionZ);
            return ahead
                ? Ahead(program, kept.Kernel, source, out refusal)
                : WithPrediction(program, kept.Kernel, source, systemRegisters, compute, dimensionX, dimensionY, dimensionZ, indirect);
        }
        finally
        {
            Monitor.Exit(_resolveGate);
        }
    }

    // GetComputeProgramKept's kept and rebound answers without the shared tier (shards only): true with the answer, false
    // with why the shared tier must make it (reason), having changed nothing the shared path reads differently.
    private bool TryGetComputeProgramUnshared(ComputeStageRegisters compute, uint dispatchInitiator, uint dimensionX, uint dimensionY, uint dimensionZ,
        bool indirect, bool ahead, bool gpuThreadCounts, out ComputeProgram? result, out ComputePrefetchRefusal refusal, ref SharedReason reason)
    {
        result = null;
        refusal = ComputePrefetchRefusal.None;
        if (ahead && _registry.HasFrontend)
        {
            refusal = ComputePrefetchRefusal.Frontend;
            return true;
        }

        if (DrawReuseDisabled || _registry.HasFrontend)
        {
            return false;
        }

        var registrations = ShaderHeaderRegistry.Registrations;
        ClaimKeptPrograms();
        if (_keptComputes is not { } all || !all.TryGetValue(compute.Address, out var keptList))
        {
            return false;
        }

        KeptComputeProgram? kept = null;
        foreach (var candidate in keptList)
        {
            if (candidate.Registrations == registrations && candidate.DispatchInitiator == dispatchInitiator &&
                candidate.GpuThreadCounts == gpuThreadCounts && SameComputeShape(candidate.Registers, compute))
            {
                kept = candidate;
                break;
            }
        }

        if (kept is null)
        {
            return false;
        }

        if (kept.Program is { } previous && kept.Registers is { } previousRegisters && kept.DimensionX == dimensionX &&
            kept.DimensionY == dimensionY && kept.DimensionZ == dimensionZ &&
            SameUserScalars(previousRegisters.UserScalars, compute.UserScalars, compute.UserScalarCount) && _programs.Revalidate(previous.Input.Stage))
        {
            _computeKept.Increment();
            t_resolvePath = ResolvePath.ComputeKept;
            _unsharedComputeKept.Increment();
            result = ahead
                ? Ahead(previous, kept.Kernel, kept.Source, out refusal)
                : WithPrediction(previous, kept.Kernel, kept.Source, kept.SystemRegisters, compute, dimensionX, dimensionY, dimensionZ, indirect);
            return true;
        }

        if (kept.Entry is not { } entry)
        {
            return false;
        }

        reason = SharedReason.ComputeRebind;
        if (kept.RecheckDue || (kept.Rebinds + 1) % IdentityRecheckPeriod == 0)
        {
            _rebindSharedBy[(int)ShaderProgramCache.UnsharedRefusal.Recheck].Increment();
            return false;
        }

        var userData = UserData(compute.UserScalars, compute.UserScalarCount, probeWrittenRegisters: false, compute.Address, "compute", ref t_computeUserData);
        var source = kept.Source! with { UserData = userData };
        var input = ComputeStageInputResolver.Resolve(compute, source.Registered, dispatchInitiator, !_host.ComputeWave64Supported, dimensionX, dimensionY, dimensionZ,
            kept.GpuThreadCounts);
        var options = t_computeOptions ??= StageCompileOptions.CreateReusable();
        options.ComputeInfo = input;
        options.ComputeSystemRegisters = kept.SystemRegisters;
        options.ComputeDispatch = CountedDispatch(input, kept.SystemRegisters, dimensionX, dimensionY, dimensionZ, kept.GpuThreadCounts);
        options.Tessellation = null;
        var pushDataCursor = ComputePushDataCursor(input);
        bool completed;
        ShaderProgram handle;
        ShaderStageResources stage;
        ShaderProgramCache.UnsharedRefusal refused;
        using (RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPreparation))
        {
            completed = _programs.TryCompleteProgramUnshared(entry, source, options, ref pushDataCursor, out handle, out stage, out refused);
        }

        options.ComputeInfo = null;
        if (!completed)
        {
            _rebindSharedBy[(int)refused].Increment();
            return false;
        }

        // What GetComputeProgramKept's rebinding leaves behind, with the same program.
        kept.Rebinds++;
        kept.Source = source;
        input.Stage = stage;
        var rebound = new ComputeProgram { Program = handle, Input = input };
        _computeRebound.Increment();
        t_resolvePath = ResolvePath.ComputeRebound;
        _unsharedComputeRebound.Increment();
        KeepComputeProgram(kept, rebound, compute, dimensionX, dimensionY, dimensionZ);
        result = ahead
            ? Ahead(rebound, kept.Kernel, kept.Source, out refusal)
            : WithPrediction(rebound, kept.Kernel, kept.Source, kept.SystemRegisters, compute, dimensionX, dimensionY, dimensionZ, indirect);
        return true;
    }

    // SHARPEMU_RESOLVE_UNSHARED_COMPUTE=0: with shards, every dispatch takes the shared tier as with one resolver.
    internal static bool UnsharedCompute { get; set; } = Environment.GetEnvironmentVariable("SHARPEMU_RESOLVE_UNSHARED_COMPUTE") != "0";

    // This thread's user data memo and lookup options for unshared rebindings (the cache's own are the shared tier's).
    [ThreadStatic]
    private static uint[]? t_computeUserData;
    [ThreadStatic]
    private static StageCompileOptions? t_computeOptions;

    // A program resolved ahead stands for the dispatch's own only where the dispatch would take it unchanged:
    // WithPrediction returns it as it is, and no frontend or consumption had a say.
    private ComputeProgram? Ahead(ComputeProgram program, KernelDemand? kernel, ShaderSource? source, out ComputePrefetchRefusal refusal)
    {
        refusal = !program.Available || program.Consumed || program.ImageCopy is not null ? ComputePrefetchRefusal.Unavailable
            : source?.Registered.Frontend is not null ? ComputePrefetchRefusal.Frontend
            : WouldPredict(program, kernel, source) ? ComputePrefetchRefusal.Prediction
            : ComputePrefetchRefusal.None;
        return refusal == ComputePrefetchRefusal.None ? program : null;
    }

    private void KeepComputeProgram(KeptComputeProgram kept, ComputeProgram program, ComputeStageRegisters compute, uint dimensionX, uint dimensionY, uint dimensionZ)
    {
        // A program with shader calls reads its call tables at every dispatch (its entry rebinds instead); calls a backend
        // without linking skips (LinkedShaderCallsSupported) read none.
        kept.Program = program.Available && !program.Consumed && !(kept.Entry is { } keptEntry && _programs.ReadsCallTables(keptEntry)) ? program : null;
        if (kept.Registers is { } registers && SameComputeShape(registers, compute))
        {
            registers.UserScalars.CopyFrom(compute.UserScalars, compute.UserScalarCount);
        }
        else
        {
            kept.Registers = compute.Copy();
        }

        kept.DimensionX = dimensionX;
        kept.DimensionY = dimensionY;
        kept.DimensionZ = dimensionZ;
    }

    // The kept program's entry with this dispatch's user data: its code is not read or hashed again.
    private ComputeProgram RebindComputeProgram(KeptComputeProgram kept, ProgramSourceEntry entry, ComputeStageRegisters compute, uint dispatchInitiator,
        uint dimensionX, uint dimensionY, uint dimensionZ, bool ahead)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPreparation);
        var userData = UserData(compute.UserScalars, compute.UserScalarCount, probeWrittenRegisters: false, compute.Address, "compute", ref _computeUserData);
        var source = kept.Source! with { UserData = userData };
        kept.Source = source;
        var input = ComputeStageInputResolver.Resolve(compute, source.Registered, dispatchInitiator, !_host.ComputeWave64Supported, dimensionX, dimensionY, dimensionZ,
            kept.GpuThreadCounts);
        ShaderProgram handle;
        ShaderStageResources stage;
        lock (_gate)
        {
            var pushDataCursor = ComputePushDataCursor(input);
            var options = _computeOptions;
            options.ComputeInfo = input;
            options.ComputeSystemRegisters = kept.SystemRegisters;
            options.ComputeDispatch = CountedDispatch(input, kept.SystemRegisters, dimensionX, dimensionY, dimensionZ, kept.GpuThreadCounts);
            options.Tessellation = null;
            var started = System.Diagnostics.Stopwatch.GetTimestamp();
            var cursor = pushDataCursor;
            var idleRetries = 0;
            while (!_programs.TryGetProgramForEntry(entry, source, options, _strictShaders, ref pushDataCursor, out handle, out stage, out var rejection))
            {
                if (rejection.Length != 0 || ahead || !HoldPendingPreparation(source, started, ref idleRetries))
                {
                    ReportShaderSkip(source, rejection);
                    return new ComputeProgram { Available = false };
                }

                pushDataCursor = cursor;
            }
        }

        input.Stage = stage;
        return new ComputeProgram { Program = handle, Input = input };
    }

    private static ShaderSource RebindSource(ShaderSource kept, ComputeStageRegisters compute) =>
        kept with { UserData = UserData(compute.UserScalars, compute.UserScalarCount, probeWrittenRegisters: false, compute.Address, "compute").ToArray() };

    // The rebind of a kept program's entry for new user data and dimensions. It touches no kept state, so a ticket worker runs
    // it as well (reads recorded between BeginWorkerResolve and EndWorkerResolve).
    private bool TryRebindProgram(ProgramSourceEntry entry, ShaderSource source, Gen5ComputeSystemRegisters systemRegisters, bool gpuThreadCounts,
        ComputeStageRegisters compute, uint dispatchInitiator, uint dimensionX, uint dimensionY, uint dimensionZ, out ComputeProgram program,
        out string rejection)
    {
        var input = ComputeStageInputResolver.Resolve(compute, source.Registered, dispatchInitiator, !_host.ComputeWave64Supported, dimensionX, dimensionY, dimensionZ,
            gpuThreadCounts);
        var pushDataCursor = ComputePushDataCursor(input);
        var options = new StageCompileOptions
        {
            ComputeInfo = input, ComputeSystemRegisters = systemRegisters,
            ComputeDispatch = CountedDispatch(input, systemRegisters, dimensionX, dimensionY, dimensionZ, gpuThreadCounts),
        };
        ShaderProgram handle;
        ShaderStageResources stage;
        bool found;
        // Ticket and deferred-dispatch workers rebind here too: the program cache takes its lookups under the cache gate,
        // as RebindComputeProgram does.
        lock (_gate)
        {
            found = _programs.TryGetProgramForEntry(entry, source, options, _strictShaders, ref pushDataCursor, out handle, out stage, out rejection);
        }

        if (!found)
        {
            program = null!;
            return false;
        }

        input.Stage = stage;
        program = new ComputeProgram { Program = handle, Input = input };
        return true;
    }

    // SHARPEMU_COMPUTE_TICKETS, on the command thread: the kept program a scanned dispatch would rebind. Programs with shader
    // calls or without a fast materialization rebind on the command thread only.
    internal bool PrepareComputeTicket(ComputeTickets.Ticket ticket)
    {
        var compute = ticket.Compute;
        var registrations = ShaderHeaderRegistry.Registrations;
        KeptComputeProgram? kept = null;
        // The kept programs are this thread's (the resolving command thread's).
        ClaimKeptPrograms();
        if (!_registry.HasFrontend && _keptComputes is { } all && all.TryGetValue(compute.Address, out var keptList))
        {
            foreach (var candidate in keptList)
            {
                if (candidate.Registrations == registrations && candidate.DispatchInitiator == ticket.Initiator &&
                    candidate.GpuThreadCounts == ticket.GpuThreadCounts && SameComputeShape(candidate.Registers, compute))
                {
                    kept = candidate;
                    break;
                }
            }
        }

        if (kept is null || kept.Entry is not { } entry || kept.Source is not { } source || entry.HasShaderCalls ||
            !SharpEmu.ShaderCompiler.Resources.ResourceMaterializer.CanMaterializeFast(entry.Plan))
        {
            return false;
        }

        ticket.Kept = kept;
        ticket.Entry = entry;
        ticket.BaseSource = source;
        ticket.SystemRegisters = kept.SystemRegisters;
        ticket.GpuThreadCounts = kept.GpuThreadCounts;
        ticket.Registrations = registrations;
        return true;
    }

    // On a ticket worker, between BeginWorkerResolve and EndWorkerResolve.
    internal void RunComputeTicket(ComputeTickets.Ticket ticket)
    {
        var source = RebindSource(ticket.BaseSource!, ticket.Compute);
        if (!TryRebindProgram(ticket.Entry!, source, ticket.SystemRegisters, ticket.GpuThreadCounts, ticket.Compute, ticket.Initiator,
            ticket.GroupsX, ticket.GroupsY, ticket.GroupsZ, out var program, out _))
        {
            ticket.Refused = true;
            return;
        }

        ticket.Program = program;
        ticket.Source = source;
    }

    // On the command thread: the kept program a direct dispatch would rebind, when a deferred dispatch worker may rebind it
    // (as a ticket worker could), or null.
    internal KeptComputeProgram? FindDeferrableKept(ComputeStageRegisters compute, uint dispatchInitiator)
    {
        ClaimKeptPrograms();
        if (_registry.HasFrontend || _keptComputes is not { } all || !all.TryGetValue(compute.Address, out var keptList))
        {
            return null;
        }

        var registrations = ShaderHeaderRegistry.Registrations;
        KeptComputeProgram? kept = null;
        foreach (var candidate in keptList)
        {
            if (candidate.Registrations == registrations && candidate.DispatchInitiator == dispatchInitiator && !candidate.GpuThreadCounts &&
                SameComputeShape(candidate.Registers, compute))
            {
                kept = candidate;
                break;
            }
        }

        return kept is { Entry: { HasShaderCalls: false } entry, Source: not null, Kernel: null } &&
            SharpEmu.ShaderCompiler.Resources.ResourceMaterializer.CanMaterializeFast(entry.Plan)
            ? kept
            : null;
    }

    // On the command thread: takes a direct dispatch for a deferred dispatch worker. The kept state is left as the rebind
    // path leaves it, but for the program, which the worker resolves: the program's next dispatch on the command thread
    // rebinds. An identity recheck turn moves to that dispatch, unless a recheck already waits for one.
    internal bool ClaimDeferral(KeptComputeProgram kept, ComputeStageRegisters compute, uint dimensionX, uint dimensionY, uint dimensionZ,
        uint dispatchInitiator, DeferredDispatchInputs inputs)
    {
        if ((kept.Rebinds + 1) % IdentityRecheckPeriod == 0)
        {
            if (kept.RecheckDue)
            {
                return false;
            }

            kept.RecheckDue = true;
        }

        kept.Rebinds++;
        _computeRebound.Increment();
        inputs.Entry = kept.Entry;
        inputs.BaseSource = kept.Source;
        inputs.SystemRegisters = kept.SystemRegisters;
        inputs.GpuThreadCounts = kept.GpuThreadCounts;
        inputs.GroupsX = dimensionX;
        inputs.GroupsY = dimensionY;
        inputs.GroupsZ = dimensionZ;
        inputs.Initiator = dispatchInitiator;
        compute.CopyTo(inputs.Compute);
        kept.Program = null;
        if (kept.Registers is { } registers && SameComputeShape(registers, compute))
        {
            registers.UserScalars.CopyFrom(compute.UserScalars, compute.UserScalarCount);
        }
        else
        {
            kept.Registers = compute.Copy();
        }

        kept.DimensionX = dimensionX;
        kept.DimensionY = dimensionY;
        kept.DimensionZ = dimensionZ;
        return true;
    }

    // A deferred member's rebind: on a worker between BeginWorkerResolve and EndWorkerResolve, or on the render thread.
    internal bool RunDeferredRebind(DeferredDispatchInputs inputs, out ComputeProgram program)
    {
        var source = RebindSource(inputs.BaseSource!, inputs.Compute);
        return TryRebindProgram(inputs.Entry!, source, inputs.SystemRegisters, inputs.GpuThreadCounts, inputs.Compute, inputs.Initiator,
            inputs.GroupsX, inputs.GroupsY, inputs.GroupsZ, out program, out _);
    }

    internal static bool TicketMatches(ComputeTickets.Ticket ticket, KeptComputeProgram kept, ComputeStageRegisters compute, uint dispatchInitiator,
        uint dimensionX, uint dimensionY, uint dimensionZ) =>
        ReferenceEquals(ticket.Kept, kept) && ReferenceEquals(ticket.Entry, kept.Entry) && ticket.Registrations == ShaderHeaderRegistry.Registrations &&
        ticket.Initiator == dispatchInitiator && ticket.GroupsX == dimensionX && ticket.GroupsY == dimensionY && ticket.GroupsZ == dimensionZ &&
        ticket.GpuThreadCounts == kept.GpuThreadCounts && ticket.SystemRegisters == kept.SystemRegisters &&
        kept.Source is { } source && ticket.BaseSource is { } baseSource && (baseSource with { UserData = source.UserData }) == source &&
        SameComputeShape(ticket.Compute, compute) && SameUserScalars(ticket.Compute.UserScalars, compute.UserScalars, compute.UserScalarCount);

    private ComputeProgram TakeComputeTicket(ComputeTickets tickets, ComputeTickets.Ticket ticket, KeptComputeProgram kept, ProgramSourceEntry entry,
        ComputeStageRegisters compute, uint dispatchInitiator, uint dimensionX, uint dimensionY, uint dimensionZ)
    {
        var program = ticket.Program!;
        var source = ticket.Source!;
        tickets.Recycle(ticket);
        if (!tickets.Verify)
        {
            kept.Source = source;
            return program;
        }

        var serial = RebindComputeProgram(kept, entry, compute, dispatchInitiator, dimensionX, dimensionY, dimensionZ, ahead: false);
        tickets.NoteVerify(SameRebind(program, source, serial, kept.Source!));
        return serial;
    }

    internal static bool SameRebind(ComputeProgram ticket, ShaderSource ticketSource, ComputeProgram serial, ShaderSource serialSource)
    {
        if (ticket.Available != serial.Available || ticket.Program != serial.Program || !ticketSource.UserData.AsSpan().SequenceEqual(serialSource.UserData))
        {
            return false;
        }

        var left = ticket.Input.Stage;
        var right = serial.Input.Stage;
        if (left.Program?.Hash != right.Program?.Hash)
        {
            return false;
        }

        var a = left.Resources;
        var b = right.Resources;
        if (a is null || b is null)
        {
            return ReferenceEquals(a, b);
        }

        return SameWords(a.Buffers, b.Buffers) && SameWords(a.Images, b.Images) && SameWords(a.Samplers, b.Samplers) &&
            a.FlattenedResourceTable.AsSpan().SequenceEqual(b.FlattenedResourceTable) && a.UserData.AsSpan().SequenceEqual(b.UserData) &&
            a.DeviceAddressRanges.AsSpan().SequenceEqual(b.DeviceAddressRanges);
    }

    private static bool SameWords(uint[][] left, uint[][] right)
    {
        if (left.Length != right.Length)
        {
            return false;
        }

        for (var index = 0; index < left.Length; index++)
        {
            if (!left[index].AsSpan().SequenceEqual(right[index]))
            {
                return false;
            }
        }

        return true;
    }

    private static bool SameUserScalars(UserScalarRegisters kept, UserScalarRegisters current, int count) =>
        kept.ValueSpan(count).SequenceEqual(current.ValueSpan(count)) &&
        kept.KindSpan(count).SequenceEqual(current.KindSpan(count));

    private static bool SameComputeShape(ComputeStageRegisters? kept, ComputeStageRegisters current) =>
        kept is not null &&
        kept.Address == current.Address && kept.ProgramChecksum == current.ProgramChecksum &&
        kept.ThreadsX == current.ThreadsX && kept.ThreadsY == current.ThreadsY && kept.ThreadsZ == current.ThreadsZ &&
        kept.VectorRegisterCount == current.VectorRegisterCount && kept.Priority == current.Priority && kept.FloatMode == current.FloatMode &&
        kept.Dx10Clamp == current.Dx10Clamp && kept.DebugMode == current.DebugMode && kept.IeeeMode == current.IeeeMode &&
        kept.RequireForwardProgress == current.RequireForwardProgress && kept.HalfPrecisionOverflow == current.HalfPrecisionOverflow &&
        kept.ThreadgroupConfiguration == current.ThreadgroupConfiguration && kept.WaveSize == current.WaveSize && kept.ScratchEnable == current.ScratchEnable &&
        kept.UserScalarCount == current.UserScalarCount && kept.ThreadGroupIdXEnable == current.ThreadGroupIdXEnable &&
        kept.ThreadGroupIdYEnable == current.ThreadGroupIdYEnable && kept.ThreadGroupIdZEnable == current.ThreadGroupIdZEnable &&
        kept.ThreadGroupSizeEnable == current.ThreadGroupSizeEnable && kept.ThreadIdComponentCount == current.ThreadIdComponentCount &&
        kept.LocalDataShareSize == current.LocalDataShareSize && kept.SharedVectorRegisters == current.SharedVectorRegisters;

    // A frontend image copy's fallback program, compiled when the executor cannot copy: outside any resolution.
    private ComputeProgram CompileComputeProgram(ShaderSource source, ComputeInputInfo input, Gen5ComputeSystemRegisters systemRegisters,
        DispatchBufferFootprint.Dispatch? dispatch)
    {
        using (EnterSharedScope(SharedReason.Other))
        {
            using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPreparation);
            ShaderProgram handle;
            ShaderStageResources stage;
            lock (_gate)
            {
                var pushDataCursor = ComputePushDataCursor(input);
                if (!TryPrepareProgram(
                    source,
                    new StageCompileOptions { ComputeInfo = input, ComputeSystemRegisters = systemRegisters, ComputeDispatch = dispatch },
                    ref pushDataCursor,
                    out handle,
                    out stage))
                {
                    return new ComputeProgram { Available = false };
                }
            }

            input.Stage = stage;
            return new ComputeProgram { Program = handle, Input = input };
        }
    }

    public ComputeProgram GetComputeProgram(
        ComputeStageRegisters compute,
        ShaderInterfaceRegisters shaderInterface,
        uint dispatchInitiator,
        uint dimensionX,
        uint dimensionY,
        uint dimensionZ)
    {
        using var waitScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ComputeResolveLockWait);
        using (EnterSharedScope(SharedReason.Other))
        {
            using var workScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPreparation);
            // Predicted on the CPU like the bank form (upstream's register form does the same).
            var program = GetComputeProgramCore(compute, dispatchInitiator, dimensionX, dimensionY, dimensionZ, retry: null,
                out var source, out var systemRegisters, out var decoded);
            if (program.ImageCopy is not null)
                return program;
            var kernel = decoded is not null && program.Available && !program.Consumed && source is not null ? KernelFor(decoded) : null;
            return WithPrediction(program, kernel, source, systemRegisters, compute, dimensionX, dimensionY, dimensionZ, indirect: false);
        }
    }

    // The program of a dispatch the stream has not reached (IShaderPipelineProvider), for a
    // pipeline asked for ahead of it: the same preparation, which the dispatch then finds cached.
    public ComputeProgram PrepareComputeProgramAhead(
        ComputeStageRegisters compute,
        uint dispatchInitiator,
        uint dimensionX,
        uint dimensionY,
        uint dimensionZ)
    {
        using var waitScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ComputeResolveLockWait);
        using (EnterSharedScope(SharedReason.Other))
        {
            using var workScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPreparation);
            return GetComputeProgramCore(compute, dispatchInitiator, dimensionX, dimensionY, dimensionZ, retry: null, out _, out _, out _, ahead: true);
        }
    }

    // The same for a dispatch that may suspend on its pipeline: the program the retry kept from
    // the last attempt at it is handed back while it still holds (ComputeProgramRetry);
    // otherwise the program is prepared and kept in it for the next attempt.
    public ComputeProgram GetComputeProgram(
        ComputeStageRegisters compute,
        ShaderInterfaceRegisters shaderInterface,
        uint dispatchInitiator,
        uint dimensionX,
        uint dimensionY,
        uint dimensionZ,
        ComputeProgramRetry retry)
    {
        using var waitScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ComputeResolveLockWait);
        using (EnterSharedScope(SharedReason.ComputeRetry))
        {
            using var workScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPreparation);
            return GetComputeProgramCore(compute, dispatchInitiator, dimensionX, dimensionY, dimensionZ, retry, out _, out _, out _);
        }
    }

    private ComputeProgram GetComputeProgramCore(
        ComputeStageRegisters compute,
        uint dispatchInitiator,
        uint dimensionX,
        uint dimensionY,
        uint dimensionZ,
        ComputeProgramRetry? retry,
        out ShaderSource? resolvedSource,
        out Gen5ComputeSystemRegisters resolvedSystemRegisters,
        out Gen5ShaderProgram? decoded,
        bool ahead = false,
        bool gpuThreadCounts = false)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPreparation);
        resolvedSource = null;
        resolvedSystemRegisters = default;
        decoded = null;
        var source = PrepareSource(compute.Address, ShaderStage.Compute, "compute", compute.UserScalars, compute.UserScalarCount, probeWrittenRegisters: false, userDataBase: 0, ref _computeUserData);
        if (retry is not null)
        {
            // The source is read again (the code hash and the user data), the rest only checked.
            if (retry.Matches(source, dispatchInitiator, dimensionX, dimensionY, dimensionZ) &&
                retry.Reads.Holds(_host.TryReadResidentGuestBytes))
            {
                retry.Reused = true;
                return retry.Program!;
            }

            retry.Reset();
        }

        var program = _programs.Decode(source);
        resolvedSource = source;
        decoded = program;
        if (_skipRayTracing)
        {
            profileScope.SwitchPhase(RenderPhaseProfile.Phase.ProgramSharedGateWait);
            lock (_gate)
            {
                profileScope.SwitchPhase(RenderPhaseProfile.Phase.ProgramPreparation);
                if (!_rayTracingComputePrograms.TryGetValue(program, out var usesRayTracing))
                {
                    usesRayTracing = program.Instructions.Any(instruction =>
                        instruction.Opcode is "ImageBvhIntersectRay" or "ImageBvh64IntersectRay");
                    _rayTracingComputePrograms.Add(program, usesRayTracing);
                    if (usesRayTracing)
                        Console.Error.WriteLine($"[GPU][WARN][RAY_TRACING_SKIPPED] Compute dispatch skipped: hash=0x{source.Hash:X16}. " +
                            "The shader contains BVH ray-intersection instructions. Ray-traced effects can be missing or incorrect. " +
                            "Set SHARPEMU_SKIP_RT=0 before launch to disable this skip.");
                }
                if (usesRayTracing) return new ComputeProgram { Available = false };
            }
        }
        var input = ComputeStageInputResolver.Resolve(compute, source.Registered, dispatchInitiator, !_host.ComputeWave64Supported, dimensionX, dimensionY, dimensionZ,
            gpuThreadCounts);
        var systemRegisters = DecodeComputeSystemRegisters(compute);
        resolvedSystemRegisters = systemRegisters;
        var dispatch = CountedDispatch(input, systemRegisters, dimensionX, dimensionY, dimensionZ, gpuThreadCounts);
        if (!ahead && retry is null && source.Registered.Frontend is { } frontend &&
            frontend.TryResolveImageCopy(_context, source.Registered, source.UserData, input, systemRegisters, dimensionX, dimensionY, dimensionZ,
                out var copySource, out var copyDestination))
        {
            return new ComputeProgram
            {
                Input = input,
                ImageCopy = new(copySource, copyDestination, () => CompileComputeProgram(source, input, systemRegisters, dispatch)),
            };
        }

        if (ahead && IsMaskedDwordCopyKernel(program))
        {
            // Submitting it is running it; the dispatch does that itself when the stream gets there.
            return new ComputeProgram { Available = false };
        }

        // Thread counts in GPU memory are not known here: the masked copy's CPU replacement cannot size it.
        if (!input.IndirectThreadCounts && TrySubmitMaskedDwordCopyKernel(program, source, systemRegisters, input, out var description))
        {
            if (RenderTrace.Enabled)
            {
                RenderTrace.Write($"Consumed a masked dword copy dispatch: shader=0x{compute.Address:X16} hash=0x{source.Hash:X16} {description}");
            }

            return new ComputeProgram { Consumed = true };
        }

        ShaderProgram handle;
        ShaderStageResources stage;
        profileScope.SwitchPhase(RenderPhaseProfile.Phase.ProgramSharedGateWait);
        lock (_gate)
        {
            profileScope.SwitchPhase(RenderPhaseProfile.Phase.ProgramPreparation);
            using var lockedScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPrepareLocked);
            var pushDataCursor = ComputePushDataCursor(input);
            var computeOptions = _computeOptions;
            computeOptions.ComputeInfo = input;
            computeOptions.ComputeSystemRegisters = systemRegisters;
            computeOptions.ComputeDispatch = dispatch;
            computeOptions.Tessellation = null;
            _programs.ReadLog = retry?.Reads;
            try
            {
                if (!TryPrepareProgram(
                    source,
                    computeOptions,
                    ref pushDataCursor,
                    out handle,
                    out stage,
                    holdPending: !ahead))
                {
                    return new ComputeProgram { Available = false };
                }
            }
            finally
            {
                _programs.ReadLog = null;
            }
        }

        input.Stage = stage;
        var prepared = new ComputeProgram { Program = handle, Input = input };
        retry?.Keep(prepared, source, dispatchInitiator, dimensionX, dimensionY, dimensionZ);
        return prepared;
    }

    // Thread limits the GPU writes need shader data in a buffer: a cursor at the end of the push range keeps it out.
    private static uint ComputePushDataCursor(ComputeInputInfo input) => input.IndirectThreadCounts ? PushData.DwordCount : 0u;

    // The dispatch whose GPU-loaded record counts a footprint proof bounds (DeviceRecordCountPlan); none when the thread
    // counts are in GPU memory or the initiator counts threads.
    private static DispatchBufferFootprint.Dispatch? CountedDispatch(ComputeInputInfo input, Gen5ComputeSystemRegisters systemRegisters,
        uint dimensionX, uint dimensionY, uint dimensionZ, bool gpuThreadCounts) =>
        DeviceRecordCountPlan.Enabled && !gpuThreadCounts && !input.DispatchThreadDimensions
            ? new(dimensionX, dimensionY, dimensionZ, Math.Max(input.ThreadsX, 1), Math.Max(input.ThreadsY, 1), Math.Max(input.ThreadsZ, 1),
                input.WaveSize, input.ThreadIdCount, systemRegisters)
            : null;

    private bool TryPrepareProgram(ShaderSource source, StageCompileOptions options, ref uint pushDataCursor,
        out ShaderProgram program, out ShaderStageResources stage, bool holdPending = false)
    {
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        var cursor = pushDataCursor;
        var idleRetries = 0;
        while (true)
        {
            pushDataCursor = cursor;
            if (_programs.TryGetProgram(source, options, _strictShaders, ref pushDataCursor, out program, out stage, out var rejection))
                return true;
            if (rejection.Length != 0 || !holdPending)
            {
                ReportShaderSkip(source, rejection);
                return false;
            }

            if (!HoldPendingPreparation(source, started, ref idleRetries))
                return false;
        }
    }

    // A program with shader calls is linked and translated on a worker (DeferLinking) and reports pending meanwhile.
    // The dispatch that asked for it waits for that work, as a linked dispatch waits for its module
    // (RenderExecutor.SkipWhileLinkedModuleCompiles), for at most LinkedPreparationWait: a skipped dispatch that set up
    // later work is lost for good. A program whose preparation outlasts the bound is not waited for again, and every
    // dispatch skipped for it is counted (LinkedPreparationSkips) and reported once per program.
    internal long LinkedPreparationWait = RenderExecutor.LinkedCompileWaitTicks;
    private readonly HashSet<ulong> _slowPreparations = new();
    private long _linkedPreparationSkips;
    public long LinkedPreparationSkips => Interlocked.Read(ref _linkedPreparationSkips);

    private bool HoldPendingPreparation(ShaderSource source, long started, ref int idleRetries)
    {
        var left = LinkedPreparationWait - (System.Diagnostics.Stopwatch.GetTimestamp() - started);
        if (left > 0 && !_slowPreparations.Contains(source.Hash))
        {
            if (_programs.PendingPreparation(source.Hash) is { } task)
            {
                idleRetries = 0;
                try
                {
                    task.Wait(TimeSpan.FromSeconds((double)left / System.Diagnostics.Stopwatch.Frequency));
                }
                catch (AggregateException)
                {
                    // A failed link or translation surfaces at the next request, which refuses the program.
                }

                return true;
            }

            // Pending with no worker: the next request starts one (a per-record link); a program that stays so is refused.
            if (++idleRetries <= 2)
                return true;
        }

        Interlocked.Increment(ref _linkedPreparationSkips);
        if (_slowPreparations.Add(source.Hash))
        {
            Console.Error.WriteLine(
                $"[GPU][WARN][LINKED_PENDING] stage={source.Label} hash=0x{source.Hash:X16}: the linked program was not prepared within " +
                $"{LinkedPreparationWait * 1000 / System.Diagnostics.Stopwatch.Frequency} ms (SHARPEMU_LINKED_COMPILE_WAIT_MS); its dispatches " +
                "are skipped until it is ready and counted in LinkedPreparationSkips. Images can be incorrect.");
        }

        return false;
    }

    private void ReportShaderSkip(ShaderSource source, string rejection)
    {
        if (rejection.Length == 0)
        {
            return;
        }

        bool reported;
        lock (_reportedShaderSkips)
        {
            reported = _reportedShaderSkips.Add((source.Stage, source.Hash, source.CodeSize));
        }

        if (reported)
        {
            var tag = source.Stage == ShaderStage.Compute ? "COMPUTE_SKIPPED" : "DRAW_SKIPPED";
            Console.Error.WriteLine($"[GPU][WARN][{tag}] {rejection} " +
                "The operation was not executed. Images and FPS can be incorrect. Set SHARPEMU_STRICT_COMPUTE=1 to stop on this failure.");
        }
    }

    // The system registers that follow the user data, in the order the resource register enables them.
    public static Gen5ComputeSystemRegisters DecodeComputeSystemRegisters(ComputeStageRegisters compute)
    {
        var nextRegister = (uint)compute.UserScalarCount;
        uint? workGroupX = null;
        uint? workGroupY = null;
        uint? workGroupZ = null;
        uint? threadGroupSize = null;
        if (compute.ThreadGroupIdXEnable)
        {
            workGroupX = nextRegister++;
        }

        if (compute.ThreadGroupIdYEnable)
        {
            workGroupY = nextRegister++;
        }

        if (compute.ThreadGroupIdZEnable)
        {
            workGroupZ = nextRegister++;
        }

        if (compute.ThreadGroupSizeEnable)
        {
            threadGroupSize = nextRegister;
        }

        return new Gen5ComputeSystemRegisters(workGroupX, workGroupY, workGroupZ, threadGroupSize);
    }

    public PipelineHandle CreateGraphicsPipeline(
        ReadOnlySpan<ColorTargetState> colors,
        in DepthAttachmentState depth,
        VertexInputInfo vertexInput,
        PixelInputInfo? pixelInput,
        ContextRegisters context,
        in RenderingState rendering,
        GpuPrimitiveTopology topology,
        bool primitiveRestartEnabled,
        bool disableBlending,
        ShaderProgram vertexProgram,
        ShaderProgram pixelProgram)
    {
        if (!_host.WaitsForGraphicsPipelines)
        {
            _ = CreateGraphicsPipeline(colors, in depth, vertexInput, pixelInput, context, in rendering, topology,
            primitiveRestartEnabled, disableBlending, vertexProgram, pixelProgram, wait: true, out var created, out _);
            return created;
        }

        // A host that waits off the cache (WaitsForGraphicsPipelines): the draw starts the compile and waits for it outside
        // the cache's lock, so programs and pipelines of other draws resolve meanwhile.
        PipelineHandle handle;
        while (!CreateGraphicsPipeline(colors, in depth, vertexInput, pixelInput, context, in rendering, topology,
            primitiveRestartEnabled, disableBlending, vertexProgram, pixelProgram, wait: false, out handle, out var compiling))
        {
            if (compiling is null)
            {
                Thread.Sleep(1);
                continue;
            }

            _host.WaitForGraphicsPipeline(compiling);
        }

        return handle;
    }

    public bool TryFindGraphicsPipeline(
        ReadOnlySpan<ColorTargetState> colors,
        in DepthAttachmentState depth,
        VertexInputInfo vertexInput,
        PixelInputInfo? pixelInput,
        ContextRegisters context,
        in RenderingState rendering,
        GpuPrimitiveTopology topology,
        bool primitiveRestartEnabled,
        bool disableBlending,
        ShaderProgram vertexProgram,
        ShaderProgram pixelProgram,
        out PipelineHandle handle) =>
        CreateGraphicsPipeline(colors, in depth, vertexInput, pixelInput, context, in rendering, topology,
            primitiveRestartEnabled, disableBlending, vertexProgram, pixelProgram, wait: null, out handle, out _);

    // The pipeline of a draw the executor may leave out while it compiles: false while the
    // host is still building it (the build carries on), and nothing is cached until it is
    // handed over, so a later draw asks again and finds it then.
    public bool TryCreateGraphicsPipeline(
        ReadOnlySpan<ColorTargetState> colors,
        in DepthAttachmentState depth,
        VertexInputInfo vertexInput,
        PixelInputInfo? pixelInput,
        ContextRegisters context,
        in RenderingState rendering,
        GpuPrimitiveTopology topology,
        bool primitiveRestartEnabled,
        bool disableBlending,
        ShaderProgram vertexProgram,
        ShaderProgram pixelProgram,
        out PipelineHandle handle) =>
        CreateGraphicsPipeline(colors, in depth, vertexInput, pixelInput, context, in rendering, topology,
            primitiveRestartEnabled, disableBlending, vertexProgram, pixelProgram, wait: false, out handle, out _);

    // wait: true blocks in the host's compile, false starts it and answers false while it runs (compiling: the description
    // of that compile), null only looks the pipeline up.
    private bool CreateGraphicsPipeline(
        ReadOnlySpan<ColorTargetState> colors,
        in DepthAttachmentState depth,
        VertexInputInfo vertexInput,
        PixelInputInfo? pixelInput,
        ContextRegisters context,
        in RenderingState rendering,
        GpuPrimitiveTopology topology,
        bool primitiveRestartEnabled,
        bool disableBlending,
        ShaderProgram vertexProgram,
        ShaderProgram pixelProgram,
        bool? wait,
        out PipelineHandle handle,
        out GraphicsPipelineDescription? compiling)
    {
        compiling = null;
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.PipelineCreation);
        // The lookup fills objects this thread reuses for every draw; only a new pipeline gets its own.
        var lookup = _graphicsLookup ??= new GraphicsPipelineKey
        {
            Rendering = new PipelineRenderingState(),
            VertexInput = new PipelineVertexInputState(),
            StaticParameters = new PipelineStaticParameters(),
        };
        lookup.Rendering.Reset();
        lookup.VertexInput.Reset();
        lookup.StaticParameters.Reset();
        FillGraphicsState(colors, in depth, vertexInput, pixelInput, context, in rendering, topology, primitiveRestartEnabled, disableBlending,
            vertexProgram, pixelProgram, _host.NoAttachmentSampleCounts, lookup.Rendering, lookup.StaticParameters, lookup.VertexInput,
            _host.DepthBoundsInPipelineState);
        lookup.VertexProgramId = vertexProgram.Id;
        lookup.PixelProgramId = pixelInput is not null ? pixelProgram.Id : 0;
        var lodStatistics = pixelInput is not null && _host.PixelPipelinesFollowLodStatistics && MipStatistics.TitleUsesCounters;
        lookup.LodStatistics = lodStatistics;
        lookup.Tessellation = vertexInput.Tessellation;
        if (_graphicsPipelines.TryGetValue(lookup, out handle))
        {
            return true;
        }

        profileScope.SwitchPhase(RenderPhaseProfile.Phase.PipelineSharedGateWait);
        lock (_gate)
        {
            profileScope.SwitchPhase(RenderPhaseProfile.Phase.PipelineCreation);
            if (_graphicsPipelines.TryGetValue(lookup, out handle))
            {
                return true;
            }

            // A lookup only (TryFindGraphicsPipeline): nothing is built or started.
            if (wait is null)
            {
                return false;
            }

            // A pipeline the host is still compiling is asked for with the description that started its compile.
            if (_pendingGraphicsPipelines.TryGetValue(lookup, out var pending))
            {
                if (wait == true)
                {
                    handle = _host.CreateGraphicsPipeline(pending.Description);
                }
                else if (!_host.TryCreateGraphicsPipeline(pending.Description, out handle))
                {
                    compiling = pending.Description;
                    return false;
                }

                _pendingGraphicsPipelines.Remove(pending.Key);
                AddGraphicsPipeline(pending.Key, pending.Description, handle);
                return true;
            }

            var description = BuildGraphicsDescription(
                colors, in depth, vertexInput, pixelInput, context, in rendering, topology, primitiveRestartEnabled, disableBlending,
                vertexProgram, pixelProgram, _host.NoAttachmentSampleCounts, _host.DepthBoundsInPipelineState, lodStatistics);
            var key = KeyOf(description);
            if (RenderTrace.Enabled && RenderTrace.Pipeline())
            {
                RenderTrace.Write(
                    $"PipelineCache create graphics vertex=0x{vertexProgram.Id:X16} pixel=0x{key.PixelProgramId:X16} colors={key.Rendering.ColorCount} " +
                    $"depth={description.StaticParameters.WithDepth} samples={description.StaticParameters.Samples}");
            }

            if (wait == true)
            {
                handle = _host.CreateGraphicsPipeline(description);
            }
            else if (!_host.TryCreateGraphicsPipeline(description, out handle))
            {
                _pendingGraphicsPipelines.Add(key, (key, description));
                compiling = description;
                return false;
            }

            AddGraphicsPipeline(key, description, handle);
            return true;
        }
    }

    // Caller holds _gate.
    private void AddGraphicsPipeline(GraphicsPipelineKey key, GraphicsPipelineDescription description, PipelineHandle handle)
    {
        _graphicsPipelines.TryAdd(key, handle);
        // The upstream graphics records of a host without its own graphics prewarm (Metal keeps GraphicsPrewarmList).
        if (_host.GraphicsPrewarm is null && _host.ShaderPrewarm is { } prewarm && description.Tessellation is null &&
            description.VertexStage.Stage == ShaderStageKind.Vertex)
        {
            prewarm.RecordGraphics(description);
        }

        ShaderCacheCounters.CountGraphicsPipeline();
    }

    [ThreadStatic]
    private static GraphicsPipelineKey? _graphicsLookup;

    // The full static state of one graphics pipeline from the draw's targets, registers and programs.
    internal static GraphicsPipelineDescription BuildGraphicsDescription(
        ReadOnlySpan<ColorTargetState> colors,
        in DepthAttachmentState depth,
        VertexInputInfo vertexInput,
        PixelInputInfo? pixelInput,
        ContextRegisters context,
        in RenderingState rendering,
        GpuPrimitiveTopology topology,
        bool primitiveRestartEnabled,
        bool disableBlending,
        ShaderProgram vertexProgram,
        ShaderProgram pixelProgram,
        GpuSampleCount noAttachmentSampleCounts,
        bool depthBoundsInPipelineState = true,
        bool lodStatistics = false)
    {
        var renderingState = new PipelineRenderingState();
        var parameters = new PipelineStaticParameters();
        var vertexState = new PipelineVertexInputState();
        var (vertexStage, pixelStage) = FillGraphicsState(colors, in depth, vertexInput, pixelInput, context, in rendering, topology,
            primitiveRestartEnabled, disableBlending, vertexProgram, pixelProgram, noAttachmentSampleCounts, renderingState, parameters, vertexState,
            depthBoundsInPipelineState);
        return new GraphicsPipelineDescription
        {
            Tessellation = vertexInput.Tessellation,
            Rendering = renderingState,
            VertexInput = vertexState,
            VertexInfo = vertexInput,
            VertexProgram = vertexProgram,
            VertexStage = vertexStage,
            PixelInfo = pixelInput,
            PixelProgram = pixelProgram,
            PixelStage = pixelStage,
            StaticParameters = parameters,
            LodStatistics = lodStatistics,
        };
    }

    // Writes the pipeline state into objects in their initial state; returns the stage programs.
    private static (ShaderProgramInfo VertexStage, ShaderProgramInfo? PixelStage) FillGraphicsState(
        ReadOnlySpan<ColorTargetState> colors,
        in DepthAttachmentState depth,
        VertexInputInfo vertexInput,
        PixelInputInfo? pixelInput,
        ContextRegisters context,
        in RenderingState rendering,
        GpuPrimitiveTopology topology,
        bool primitiveRestartEnabled,
        bool disableBlending,
        ShaderProgram vertexProgram,
        ShaderProgram pixelProgram,
        GpuSampleCount noAttachmentSampleCounts,
        PipelineRenderingState renderingState,
        PipelineStaticParameters parameters,
        PipelineVertexInputState vertexState,
        bool depthBoundsInPipelineState = true)
    {
        if (colors.Length > PipelineStaticParameters.ColorAttachmentCount)
        {
            throw GpuFatal.Fatal($"The draw binds more color attachments than a pipeline holds: count={colors.Length}.");
        }

        if (!vertexProgram.IsValid)
        {
            throw GpuFatal.Fatal("The draw has no vertex program.");
        }

        var pixelActive = pixelInput is not null;
        if (pixelActive && !pixelProgram.IsValid)
        {
            throw GpuFatal.Fatal("The draw has an active pixel stage without a pixel program.");
        }

        var vertexStage = vertexInput.Stage.Program ?? throw GpuFatal.Fatal("The vertex stage has no program.");
        var pixelStage = pixelActive ? pixelInput!.Stage.Program ?? throw GpuFatal.Fatal("The pixel stage has no program.") : null;
        var colorCount = (uint)colors.Length;
        parameters.ColorCount = colorCount;
        renderingState.ColorCount = colorCount;
        var routing = PixelExportRouting.Routing.Of(context.ShaderInterface);
        for (var index = 0; index < colors.Length; index++)
        {
            ref readonly var color = ref colors[index];
            var format = rendering.ColorAttachments[index].Format;
            if (!color.Image.IsValid || format == GpuFormat.Undefined)
            {
                throw GpuFatal.Fatal($"A color attachment has no image or format: slot={color.Slot} format={format}.");
            }

            renderingState.ColorFormats[index] = format;
            // A target the pixel program never exports keeps its contents, as on hardware; the
            // host output would otherwise write an undefined value (e.g. depth-only passes that
            // leave a color target bound and export only to the null target).
            var exportTarget = color.Slot == 0 && context.BlendControls[0].UsesSecondarySource &&
                (context.ColorTargets[0].Info & (1u << 16)) == 0
                ? 0 : routing.ExportForSlot(color.Slot);
            var exported = pixelStage is not null &&
                (exportTarget >= 0 && ((pixelStage.PixelColorExportMasks >> (exportTarget * 4)) & 0xFu) != 0);
            var colorMask = exported ? color.Resolution.ExportMapping.ApplyMask(context.ColorWriteMaskForSlot(color.Slot)) : 0;
            parameters.SetColorMask(index, colorMask);
            if (RenderTrace.Enabled && RenderTrace.Pipeline())
            {
                RenderTrace.Write(
                    $"PipelineCache output slot={color.Slot} export={exportTarget} " +
                    $"guestMask=0x{context.ColorWriteMaskForSlot(color.Slot):X} " +
                    $"exported={(exported ? 1 : 0)} shaderMask=0x{pixelStage?.PixelColorExportMasks ?? 0:X8} " +
                    $"mapping=0x{color.Resolution.ExportMapping.Packed:X2} hostMask=0x{colorMask:X} format={(int)format}");
            }
        }

        var withDepth = depth.HasTarget;
        if (withDepth)
        {
            renderingState.DepthFormat = rendering.DepthFormat;
            renderingState.StencilFormat = rendering.StencilFormat;
        }

        var samples = rendering.Samples;
        if (colorCount == 0 && !withDepth)
        {
            samples = 1u << context.AntialiasingConfig.SampleCountLog2;
            var flag = ImageDescription.VulkanSampleCount(samples);
            if (flag == 0 || (noAttachmentSampleCounts & flag) == 0)
            {
                throw GpuFatal.Fatal($"The device cannot rasterize without attachments at the draw's sample count: samples={samples}.");
            }
        }

        if (samples == 0 || ImageDescription.VulkanSampleCount(samples) == 0)
        {
            throw GpuFatal.Fatal($"The draw's sample count is invalid: samples={samples}.");
        }

        var depthState = withDepth ? depth.Target.State : default;
        // A patch list without tessellation stages carries legacy rectangles, which are never
        // culled. Tessellated patches rasterize as ordinary triangles and keep the guest's culling.
        var rectangleList = topology == GpuPrimitiveTopology.PatchList && vertexInput.Tessellation is null;
        var mode = context.RasterMode;
        parameters.NegativeOneToOne = !context.Clip.DirectXClipSpace;
        parameters.DepthClipEnable = HostViewportTransform.DepthClipOf(context);
        parameters.Topology = topology;
        parameters.PrimitiveRestartEnable = primitiveRestartEnabled;
        parameters.Samples = samples;
        parameters.SampleShadingEnable = pixelActive && samples > 1 && pixelInput!.SampleShading;
        parameters.WithDepth = withDepth;
        parameters.DepthBoundsTestEnable = depthState.DepthBoundsTestEnabled;
        parameters.DepthMinBounds = depthState.DepthMinBounds;
        parameters.DepthMaxBounds = depthState.DepthMaxBounds;
        parameters.StencilTestEnable = depthState.StencilTestEnabled;
        parameters.StencilFront = withDepth ? depthState.FrontOperations : StencilOperations.Default;
        parameters.StencilBack = withDepth ? depthState.BackOperations : StencilOperations.Default;
        parameters.CullBack = !rectangleList && mode.CullBack;
        parameters.CullFront = !rectangleList && mode.CullFront;
        parameters.FrontFaceClockwise = mode.FrontFaceClockwise;
        parameters.AlphaToCoverageEnable = AlphaToCoverageEnabled(colors, context, pixelActive);
        for (var index = 0; index < colors.Length; index++)
        {
            var slot = colors[index].Slot;
            var blend = context.BlendControls[slot];
            parameters.SetColorSourceBlend(index, (byte)blend.ColorSourceFactor);
            parameters.SetColorBlendFunction(index, (byte)blend.ColorFunction);
            parameters.SetColorDestinationBlend(index, (byte)blend.ColorDestinationFactor);
            parameters.SetAlphaSourceBlend(index, (byte)blend.AlphaSourceFactor);
            parameters.SetAlphaBlendFunction(index, (byte)blend.AlphaFunction);
            parameters.SetAlphaDestinationBlend(index, (byte)blend.AlphaDestinationFactor);
            parameters.SetSeparateAlphaBlend(index, blend.SeparateAlpha);
            parameters.SetBlendEnable(index, blend.Enable && !disableBlending);
            parameters.SetBlendBypass(index, ((context.ColorTargets[slot].Info >> 16) & 0x1) != 0);
        }

        NormalizeStaticParameters(parameters, depthBoundsInPipelineState);
        FillVertexInputState(vertexInput, vertexState);
        return (vertexStage, pixelStage);
    }

    // Zeroes the state no backend reads, so draws that differ only there share one pipeline:
    // the depth bounds when the test is off (or the host keeps them out of its pipeline), the
    // stencil operations when the stencil test is off, and the blend factors, functions and
    // flags of an attachment that does not blend (disabled, bypassed, or writing no channel).
    // Every field is either ignored by the host pipeline when its switch is off or, for the
    // separate-alpha factors, replaced by the colour ones.
    internal static void NormalizeStaticParameters(PipelineStaticParameters parameters, bool depthBoundsInPipelineState)
    {
        if (!depthBoundsInPipelineState || !parameters.DepthBoundsTestEnable)
        {
            parameters.DepthBoundsTestEnable = false;
            parameters.DepthMinBounds = 0f;
            parameters.DepthMaxBounds = 0f;
        }

        if (!parameters.StencilTestEnable)
        {
            parameters.StencilFront = StencilOperations.Default;
            parameters.StencilBack = StencilOperations.Default;
        }

        for (var index = 0; index < (int)parameters.ColorCount; index++)
        {
            var blends = parameters.GetBlendEnable(index) && !parameters.GetBlendBypass(index) && parameters.GetColorMask(index) != 0;
            if (!blends)
            {
                parameters.SetBlendEnable(index, false);
                parameters.SetBlendBypass(index, false);
                parameters.SetSeparateAlphaBlend(index, false);
                parameters.SetColorSourceBlend(index, 0);
                parameters.SetColorDestinationBlend(index, 0);
                parameters.SetColorBlendFunction(index, 0);
            }

            if (!parameters.GetSeparateAlphaBlend(index))
            {
                parameters.SetAlphaSourceBlend(index, 0);
                parameters.SetAlphaDestinationBlend(index, 0);
                parameters.SetAlphaBlendFunction(index, 0);
            }
        }
    }

    // DB_ALPHA_TO_MASK turns the alpha of MRT0's export into sample coverage unless the pixel
    // program opts out (DB_SHADER_CONTROL.ALPHA_TO_MASK_DISABLE). The host takes the alpha of
    // its first color output, so that output has to be the one export 0 feeds; otherwise the
    // draw runs without alpha to coverage (documented deviation, warned once).
    private static bool AlphaToCoverageEnabled(ReadOnlySpan<ColorTargetState> colors, ContextRegisters context, bool pixelActive)
    {
        if (!pixelActive || (context.AlphaToMask & 1u) == 0 || context.ShaderInterface.DepthShaderControl.AlphaToMaskDisable)
        {
            return false;
        }

        if (colors.Length == 0 || PixelExportRouting.ExportForSlot(context.ShaderInterface, colors[0].Slot) != 0)
        {
            ShaderCompilerWarnings.WarnOnce(
                "alpha-to-coverage-without-mrt0",
                $"Alpha to coverage without MRT0's export at the first color attachment runs without coverage: " +
                $"attachments={colors.Length} firstSlot={(colors.Length == 0 ? -1 : (int)colors[0].Slot)} " +
                $"dbAlphaToMask=0x{context.AlphaToMask:X8}.");
            return false;
        }

        return true;
    }

    internal static GraphicsPipelineKey KeyOf(GraphicsPipelineDescription description) => new()
    {
        Rendering = description.Rendering,
        VertexProgramId = description.VertexProgram.Id,
        PixelProgramId = description.PixelInfo is not null ? description.PixelProgram.Id : 0,
        LodStatistics = description.LodStatistics,
        Tessellation = description.Tessellation,
        VertexInput = description.VertexInput,
        StaticParameters = description.StaticParameters,
    };

    private static void FillVertexInputState(VertexInputInfo info, PipelineVertexInputState state)
    {
        if (info.Buffers.Length > VertexInputInfo.MaxBuffers || info.Attributes.Length > VertexInputInfo.MaxBuffers)
        {
            throw GpuFatal.Fatal($"The vertex input is too large: buffers={info.Buffers.Length} attributes={info.Attributes.Length}.");
        }

        state.BindingCount = (byte)info.Buffers.Length;
        state.AttributeCount = (byte)info.Attributes.Length;
        for (var binding = 0; binding < info.Buffers.Length; binding++)
        {
            var buffer = info.Buffers[binding];
            state.Bindings[binding] = new PipelineVertexBinding(buffer.Stride, buffer.PerInstance);
        }

        for (var index = 0; index < info.Attributes.Length; index++)
        {
            var attribute = info.Attributes[index];
            if (attribute.BufferIndex < 0 || attribute.BufferIndex >= info.Buffers.Length)
            {
                throw GpuFatal.Fatal($"The vertex attribute names a buffer outside the input: attribute={index} buffer={attribute.BufferIndex} buffers={info.Buffers.Length}.");
            }

            state.Attributes[index] = new PipelineVertexAttribute(attribute.OffsetBytes, (byte)attribute.BufferIndex);
        }
    }

    public PipelineHandle CreateComputePipeline(ComputeInputInfo input, ShaderProgram program)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.PipelineCreation);
        if (!program.IsValid)
        {
            throw GpuFatal.Fatal("The dispatch has no compute program.");
        }

        var stage = input.Stage.Program ?? throw GpuFatal.Fatal("The compute stage has no program.");
        var key = new ComputePipelineKey(program.Id);
        if (_computePipelines.TryGetValue(key, out var existing))
        {
            return existing;
        }

        profileScope.SwitchPhase(RenderPhaseProfile.Phase.PipelineSharedGateWait);
        lock (_gate)
        {
            profileScope.SwitchPhase(RenderPhaseProfile.Phase.PipelineCreation);
            if (_computePipelines.TryGetValue(key, out var cached))
            {
                return cached;
            }

            if (RenderTrace.Enabled && RenderTrace.Pipeline())
            {
                RenderTrace.Write($"PipelineCache create compute program=0x{program.Id:X16} hash=0x{stage.Hash:X16}");
            }

            var created = _host.CreateComputePipeline(new ComputePipelineDescription { Input = input, Program = program, Stage = stage });
            _computePipelines[key] = created;

            ShaderCacheCounters.CountComputePipeline();
            return created;
        }
    }

    public bool TryCreateComputePipeline(ComputeInputInfo input, ShaderProgram program, out PipelineHandle handle)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.PipelineCreation);
        handle = default;
        if (!program.IsValid)
        {
            throw GpuFatal.Fatal("The dispatch has no compute program.");
        }

        var stage = input.Stage.Program ?? throw GpuFatal.Fatal("The compute stage has no program.");
        var key = new ComputePipelineKey(program.Id);
        if (_computePipelines.TryGetValue(key, out var existing))
        {
            handle = existing;
            return true;
        }

        profileScope.SwitchPhase(RenderPhaseProfile.Phase.PipelineSharedGateWait);
        lock (_gate)
        {
            profileScope.SwitchPhase(RenderPhaseProfile.Phase.PipelineCreation);
            if (_computePipelines.TryGetValue(key, out var cached))
            {
                handle = cached;
                return true;
            }

            if (!_host.TryCreateComputePipeline(
                    new ComputePipelineDescription { Input = input, Program = program, Stage = stage },
                    out var created))
            {
                return false;
            }

            _computePipelines[key] = created;

            ShaderCacheCounters.CountComputePipeline();
            handle = created;
            return true;
        }
    }

    public Task? GetComputePipelineCompletion(ShaderProgram program)
    {
        lock (_gate)
        {
            return _host.GetComputePipelineCompletion(program);
        }
    }

    public bool IsSlowLinkedCompile(ulong hash) => _host.ShaderPrewarm?.IsSlowLinkedCompile(hash) ?? false;

    public void RecordLinkedCompile(ulong hash, bool slow, long milliseconds) =>
        _host.ShaderPrewarm?.RecordLinkedCompile(hash, slow, milliseconds);
}

// The register stamps a kept draw was resolved under (RegisterBanks.Stamp) and the shader registrations then.
internal readonly record struct KeptGraphicsStamps(ulong Programs, ulong VertexUserData, ulong PixelUserData, ulong Interface, long Registrations);

// The executor's inputs a kept draw was resolved with.
internal readonly record struct KeptGraphicsFlags(bool PixelActive, bool DepthBound, bool DepthAspectBound, bool StencilAspectBound,
    bool TriangleList, int StartInstanceUserData, int BaseVertexUserData = -1);

// The programs of the last draw a thread resolved, with what they were resolved from (ShaderPipelineCache.TryKeepGraphicsPrograms).
internal sealed class KeptGraphicsPrograms
{
    public GraphicsPrograms? Programs;
    public KeptGraphicsStamps Stamps;
    public KeptGraphicsFlags Flags;
    public uint OutputBlend;
    public int ExportMappingCount;
    public readonly ColorComponentMap[] ExportMapping = new ColorComponentMap[RenderingState.ColorAttachmentCapacity];
    public VertexTableWords? TableWords;
    public ulong AttributeTable;
    public ulong BufferTable;
}

// The last dispatches of one compute shader a thread resolved (ShaderPipelineCache.GetComputeProgram).
internal sealed class KeptComputeProgram
{
    public ComputeProgram? Program;
    public ComputeStageRegisters? Registers;
    public ShaderSource? Source;
    public Gen5ComputeSystemRegisters SystemRegisters;
    public ProgramSourceEntry? Entry;
    public ShaderPipelineCache.KernelDemand? Kernel;
    public long Rebinds;
    // The identity recheck turn came while one of the program's dispatches was deferred: its next dispatch on the command
    // thread resolves it from scratch.
    public bool RecheckDue;
    public long Registrations;
    public uint DispatchInitiator;
    public bool GpuThreadCounts;
    public uint DimensionX;
    public uint DimensionY;
    public uint DimensionZ;
}
