// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

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
    private readonly object _resolveGate = new();

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
        int startInstanceUserData)
    {
        var vertex = banks.Shader.Vertex;
        var pixel = banks.Shader.Pixel;
        var context = banks.Context;
        using var waitScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.GraphicsResolveLockWait);
        lock (_resolveGate)
        {
            using var workScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPreparation);
            var stamps = new KeptGraphicsStamps(
                banks.Stamp(RegisterStamp.ShaderPrograms), banks.Stamp(RegisterStamp.VertexUserData), banks.Stamp(RegisterStamp.PixelUserData),
                banks.Stamp(RegisterStamp.ColorTargets, RegisterStamp.ShaderInterface), ShaderHeaderRegistry.Registrations);
            var flags = new KeptGraphicsFlags(pixelActive, depthBound, depthAspectBound, stencilAspectBound, triangleList, startInstanceUserData);
            var kept = _keptGraphics ??= new KeptGraphicsPrograms();
            if (TryKeepGraphicsPrograms(context, kept, stamps, flags, targetExportMapping, out var keptPrograms))
            {
                _graphicsKept.Increment();
                return keptPrograms;
            }

            _graphicsResolved.Increment();
            kept.Programs = null;
            var programs = ResolveGraphicsProgramsCore(vertex, pixel, context.ShaderInterface, context, targetExportMapping, pixelActive, depthBound,
                depthAspectBound, stencilAspectBound, triangleList, mesh: null, startInstanceUserData: startInstanceUserData,
                nativeTessellation: true);
            KeepGraphicsPrograms(kept, programs, context, stamps, flags, targetExportMapping);
            return programs;
        }
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

    [ThreadStatic]
    private static KeptGraphicsPrograms? _keptGraphics;
    [ThreadStatic]
    private static Dictionary<ulong, List<KeptComputeProgram>>? _keptComputes;
    private const int KeptComputesPerShader = 4;
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
        FormattableString.Invariant($"[PERF][KEPT_PROGRAMS] graphics_kept={_graphicsKept.Take()} graphics_resolved={_graphicsResolved.Take()} ") +
        FormattableString.Invariant($"compute_kept={_computeKept.Take()} compute_rebound={_computeRebound.Take()} compute_resolved={_computeResolved.Take()}");

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
        if (DrawReuseDisabled || _registry.HasFrontend || !programs.Available || programs.MeshInput is not null ||
            programs.VertexInput.Stage.Program is null || !TryReadKeptTables(kept, programs))
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
        lock (_resolveGate)
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
        GetMeshGraphicsPrograms(banks.Shader.Vertex, banks.Shader.Pixel, banks.Context.ShaderInterface, banks.Context, userConfig,
            targetExportMapping, pixelActive, depthBound);

    public GraphicsPrograms GetMeshGraphicsPrograms(
        VertexStageRegisters vertex,
        PixelStageRegisters pixel,
        ShaderInterfaceRegisters shaderInterface,
        ContextRegisters context,
        UserConfigRegisters userConfig,
        ReadOnlySpan<ColorComponentMap> targetExportMapping,
        bool pixelActive,
        bool depthBound)
    {
        var mesh = PrepareMeshInput(vertex, shaderInterface, context, userConfig);
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
        lock (_resolveGate)
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
        bool nativeTessellation = false)
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

    private MeshDrawConfiguration? PrepareMeshInput(VertexStageRegisters vertex,
        ShaderInterfaceRegisters shaderInterface, ContextRegisters context,
        UserConfigRegisters userConfig)
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
        foreach (var slot in boundSlots)
        {
            if (unwrittenExportMasks is { } exportMasks &&
                RenderExecutor.IsUnwrittenColorTarget(context, slot, targetExportMapping[(int)slot], exportMasks))
            {
                continue;
            }

            var words = context.ColorTargets[slot];
            if (!_host.TryResolveColorOutput((uint)words.Layout, (uint)words.NumberType, (uint)words.Order, out var kind, out var mapping))
            {
                throw GpuFatal.Fatal($"The color target format has no pixel output kind: slot={slot} layout={(uint)words.Layout} numberType={(uint)words.NumberType} order={(uint)words.Order}.");
            }

            // A bound slot that no export reaches keeps its contents: its color write mask is zero.
            var exportTarget = PixelExportRouting.ExportForSlot(context.ShaderInterface, slot);
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
        foreach (var slot in boundSlots)
        {
            if (!RenderExecutor.IsUnwrittenColorTarget(context, slot, mappingBySlot[(int)slot], pixelColorExportMasks))
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
        for (var index = 0; index < boundSlots.Length; index++)
        {
            var slot = boundSlots[index];
            var extent = boundExtents[index];
            if (RenderExecutor.IsUnwrittenColorTarget(context, slot, mappingBySlot[(int)slot], pixelColorExportMasks))
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

    public ComputeProgram GetComputeProgram(RegisterBanks banks, uint dispatchInitiator, uint dimensionX, uint dimensionY, uint dimensionZ) =>
        GetComputeProgram(banks, dispatchInitiator, dimensionX, dimensionY, dimensionZ, indirect: false, retry: null);

    public ComputeProgram GetComputeProgram(RegisterBanks banks, uint dispatchInitiator, uint dimensionX, uint dimensionY, uint dimensionZ, bool indirect) =>
        GetComputeProgram(banks, dispatchInitiator, dimensionX, dimensionY, dimensionZ, indirect, retry: null);

    // A dispatch's program. With retry (a dispatch that may suspend on its pipeline) the program the retry kept is handed
    // back while its reads still hold, as GetComputeProgram(…, ComputeProgramRetry) does, and nothing is predicted: the
    // dispatch runs again. Without it, a dispatch whose registers and user data are those of a dispatch this thread
    // resolved keeps that program after its resources revalidate, else rebinds that program's entry to its user data,
    // else resolves; its writes are predicted on the CPU when the kernel is evaluable (ShaderPipelineCache.KernelEvaluation).
    public ComputeProgram GetComputeProgram(RegisterBanks banks, uint dispatchInitiator, uint dimensionX, uint dimensionY, uint dimensionZ, bool indirect,
        ComputeProgramRetry? retry)
    {
        var compute = banks.Shader.Compute;
        using var waitScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ComputeResolveLockWait);
        lock (_resolveGate)
        {
            using var workScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPreparation);
            if (retry is not null)
            {
                return GetComputeProgramCore(compute, dispatchInitiator, dimensionX, dimensionY, dimensionZ, retry, out _, out _, out _);
            }

            var registrations = ShaderHeaderRegistry.Registrations;
            var all = _keptComputes ??= new Dictionary<ulong, List<KeptComputeProgram>>();
            if (!all.TryGetValue(compute.Address, out var keptList))
            {
                keptList = new List<KeptComputeProgram>(KeptComputesPerShader);
                all.Add(compute.Address, keptList);
            }

            KeptComputeProgram? kept = null;
            foreach (var candidate in keptList)
            {
                if (candidate.Registrations == registrations && candidate.DispatchInitiator == dispatchInitiator && SameComputeShape(candidate.Registers, compute))
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
                    return WithPrediction(previous, kept.Kernel, kept.Source, kept.SystemRegisters, compute, dimensionX, dimensionY, dimensionZ, indirect);
                }

                if (kept.Entry is { } entry && ++kept.Rebinds % IdentityRecheckPeriod != 0)
                {
                    _computeRebound.Increment();
                    var rebound = RebindComputeProgram(kept, entry, compute, dispatchInitiator, dimensionX, dimensionY, dimensionZ);
                    KeepComputeProgram(kept, rebound, compute, dimensionX, dimensionY, dimensionZ);
                    return WithPrediction(rebound, kept.Kernel, kept.Source, kept.SystemRegisters, compute, dimensionX, dimensionY, dimensionZ, indirect);
                }
            }

            _computeResolved.Increment();
            var program = GetComputeProgramCore(compute, dispatchInitiator, dimensionX, dimensionY, dimensionZ, retry: null,
                out var source, out var systemRegisters, out var decoded);
            if (program.ImageCopy is not null)
                return program;
            if (kept is null)
            {
                if (keptList.Count == KeptComputesPerShader)
                {
                    keptList.RemoveAt(0);
                }

                kept = new KeptComputeProgram { Registrations = registrations, DispatchInitiator = dispatchInitiator };
                keptList.Add(kept);
            }

            var usable = program.Available && !program.Consumed && source is not null;
            kept.Source = source;
            kept.SystemRegisters = systemRegisters;
            kept.Kernel = decoded is not null && usable ? KernelFor(decoded) : null;
            kept.Entry = usable ? ShaderProgramCache.LastEntry : null;
            KeepComputeProgram(kept, program, compute, dimensionX, dimensionY, dimensionZ);
            return WithPrediction(program, kept.Kernel, source, systemRegisters, compute, dimensionX, dimensionY, dimensionZ, indirect);
        }
    }

    private static void KeepComputeProgram(KeptComputeProgram kept, ComputeProgram program, ComputeStageRegisters compute, uint dimensionX, uint dimensionY, uint dimensionZ)
    {
        kept.Program = program.Available && !program.Consumed ? program : null;
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
        uint dimensionX, uint dimensionY, uint dimensionZ)
    {
        using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPreparation);
        var userData = UserData(compute.UserScalars, compute.UserScalarCount, probeWrittenRegisters: false, compute.Address, "compute", ref _computeUserData);
        var source = kept.Source! with { UserData = userData };
        kept.Source = source;
        var input = ComputeStageInputResolver.Resolve(compute, source.Registered, dispatchInitiator, !_host.ComputeWave64Supported, dimensionX, dimensionY, dimensionZ);
        ShaderProgram handle;
        ShaderStageResources stage;
        lock (_gate)
        {
            var pushDataCursor = 0u;
            var options = _computeOptions;
            options.ComputeInfo = input;
            options.ComputeSystemRegisters = kept.SystemRegisters;
            options.Tessellation = null;
            if (!_programs.TryGetProgramForEntry(entry, source, options, _strictShaders, ref pushDataCursor, out handle, out stage, out var rejection))
            {
                ReportShaderSkip(source, rejection);
                return new ComputeProgram { Available = false };
            }
        }

        input.Stage = stage;
        return new ComputeProgram { Program = handle, Input = input };
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
    private ComputeProgram CompileComputeProgram(ShaderSource source, ComputeInputInfo input, Gen5ComputeSystemRegisters systemRegisters)
    {
        lock (_resolveGate)
        {
            using var profileScope = RenderPhaseProfile.MeasureDetail(RenderPhaseProfile.Phase.ProgramPreparation);
            ShaderProgram handle;
            ShaderStageResources stage;
            lock (_gate)
            {
                var pushDataCursor = 0u;
                if (!TryPrepareProgram(
                    source,
                    new StageCompileOptions { ComputeInfo = input, ComputeSystemRegisters = systemRegisters },
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
        lock (_resolveGate)
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
        lock (_resolveGate)
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
        lock (_resolveGate)
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
        bool ahead = false)
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
        var input = ComputeStageInputResolver.Resolve(compute, source.Registered, dispatchInitiator, !_host.ComputeWave64Supported, dimensionX, dimensionY, dimensionZ);
        var systemRegisters = DecodeComputeSystemRegisters(compute);
        resolvedSystemRegisters = systemRegisters;
        if (!ahead && retry is null && source.Registered.Frontend is { } frontend &&
            frontend.TryResolveImageCopy(_context, source.Registered, source.UserData, input, systemRegisters, dimensionX, dimensionY, dimensionZ,
                out var copySource, out var copyDestination))
        {
            return new ComputeProgram
            {
                Input = input,
                ImageCopy = new(copySource, copyDestination, () => CompileComputeProgram(source, input, systemRegisters)),
            };
        }

        if (ahead && IsMaskedDwordCopyKernel(program))
        {
            // Submitting it is running it; the dispatch does that itself when the stream gets there.
            return new ComputeProgram { Available = false };
        }

        if (TrySubmitMaskedDwordCopyKernel(program, source, systemRegisters, input, out var description))
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
            var pushDataCursor = 0u;
            var computeOptions = _computeOptions;
            computeOptions.ComputeInfo = input;
            computeOptions.ComputeSystemRegisters = systemRegisters;
            computeOptions.Tessellation = null;
            _programs.ReadLog = retry?.Reads;
            try
            {
                if (!TryPrepareProgram(
                    source,
                    computeOptions,
                    ref pushDataCursor,
                    out handle,
                    out stage))
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

    private bool TryPrepareProgram(ShaderSource source, StageCompileOptions options, ref uint pushDataCursor,
        out ShaderProgram program, out ShaderStageResources stage)
    {
        if (_programs.TryGetProgram(source, options, _strictShaders, ref pushDataCursor, out program, out stage, out var rejection))
            return true;

        ReportShaderSkip(source, rejection);
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
        _ = CreateGraphicsPipeline(colors, in depth, vertexInput, pixelInput, context, in rendering, topology,
            primitiveRestartEnabled, disableBlending, vertexProgram, pixelProgram, wait: true, out var created);
        return created;
    }

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
            primitiveRestartEnabled, disableBlending, vertexProgram, pixelProgram, wait: false, out handle);

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
        bool wait,
        out PipelineHandle handle)
    {
        handle = default;
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

        var description = BuildGraphicsDescription(
            colors, in depth, vertexInput, pixelInput, context, in rendering, topology, primitiveRestartEnabled, disableBlending,
            vertexProgram, pixelProgram, _host.NoAttachmentSampleCounts, _host.DepthBoundsInPipelineState, lodStatistics);
        var key = KeyOf(description);
        profileScope.SwitchPhase(RenderPhaseProfile.Phase.PipelineSharedGateWait);
        lock (_gate)
        {
            profileScope.SwitchPhase(RenderPhaseProfile.Phase.PipelineCreation);
            if (_graphicsPipelines.TryGetValue(key, out handle))
            {
                return true;
            }

            if (RenderTrace.Enabled && RenderTrace.Pipeline())
            {
                RenderTrace.Write(
                    $"PipelineCache create graphics vertex=0x{vertexProgram.Id:X16} pixel=0x{key.PixelProgramId:X16} colors={key.Rendering.ColorCount} " +
                    $"depth={description.StaticParameters.WithDepth} samples={description.StaticParameters.Samples}");

            }

            PipelineHandle created;
            if (wait)
            {
                created = _host.CreateGraphicsPipeline(description);
            }
            else if (!_host.TryCreateGraphicsPipeline(description, out created))
            {
                return false;
            }

            _graphicsPipelines[key] = created;

            // The upstream graphics records of a host without its own graphics prewarm (Metal keeps GraphicsPrewarmList).
            if (_host.GraphicsPrewarm is null && _host.ShaderPrewarm is { } prewarm && description.Tessellation is null &&
                description.VertexStage.Stage == ShaderStageKind.Vertex)
            {
                prewarm.RecordGraphics(description);
            }

            ShaderCacheCounters.CountGraphicsPipeline();
            handle = created;
            return true;
        }
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
                ? 0 : PixelExportRouting.ExportForSlot(context.ShaderInterface, color.Slot);
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
}

// The register stamps a kept draw was resolved under (RegisterBanks.Stamp) and the shader registrations then.
internal readonly record struct KeptGraphicsStamps(ulong Programs, ulong VertexUserData, ulong PixelUserData, ulong Interface, long Registrations);

// The executor's inputs a kept draw was resolved with.
internal readonly record struct KeptGraphicsFlags(bool PixelActive, bool DepthBound, bool DepthAspectBound, bool StencilAspectBound,
    bool TriangleList, int StartInstanceUserData);

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
    public long Registrations;
    public uint DispatchInitiator;
    public uint DimensionX;
    public uint DimensionY;
    public uint DimensionZ;
}
