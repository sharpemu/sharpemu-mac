// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Globalization;
using System.IO.Hashing;
using System.Runtime.InteropServices;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// Render pipelines specialized for a draw's buffer words (PortableBufferWord.ConstantsEnabled).
/// A portable program reads each V#'s stride and word 3 through a function constant that, when
/// the function is made without a value, falls back to the shader-data dword. The pipeline the
/// executor creates is that generic one. Before each draw the host reads the words the draw's
/// shader data holds at the dwords the vertex and pixel programs declare constants for, and
/// draws through a pipeline whose functions were made with exactly those values once it is
/// built: the same result, with the stride and format folded into the code (Demon's Souls'
/// vertex stage: 12.16 ms a frame portable, 7.62 ms baked).
/// </summary>
/// <remarks>
/// The first draw of a layout draws through the generic pipeline, as today, and queues the
/// variant on a background builder that compiles at prewarm priority (MetalCompileScheduler);
/// later draws take the variant once it is ready. Nothing is skipped or waited for. A pipeline
/// keeps at most <see cref="MaxBufferWordVariantsPerPipeline"/> variants and a module at most
/// <see cref="MaxBufferWordSpecializationsPerModule"/> specialized functions; past either the
/// draw stays on the generic pipeline and is counted. Mesh, rectangle-list and tessellation
/// pipelines stay generic, and so does the pixel stage of a pipeline with remapped colour
/// outputs. SHARPEMU_METAL_V2_BUFFER_WORD_CONSTANTS=0 turns the constants off in translation.
///
/// A compute pipeline does the same for its one program (Demon's Souls: 579 of 636 compute
/// programs declare buffer words): CommitBindings(Compute) notes the stage and the entry it
/// was committed for, BindPipeline(Compute) chooses the state from that stage's shader data
/// once per dispatch, and every path that then binds the program's state (direct, indirect,
/// thread-unit indirect) binds the chosen one through EnsureComputePipelineState. Its own
/// counters go on the PERF line as buffer_words_compute=[...].
/// SHARPEMU_METAL_V2_BUFFER_WORD_COMPUTE=0 keeps compute pipelines generic (the source is the
/// same either way).
/// </remarks>
internal sealed partial class MetalRenderHost
{
    /// <summary>
    /// The buffer words of a portable program are function constants
    /// (PortableBufferWord.ConstantsEnabled), on with portable buffers unless
    /// SHARPEMU_METAL_V2_BUFFER_WORD_CONSTANTS=0, which restores today's MSL exactly.
    /// </summary>
    internal static bool BufferWordConstants => PortableBuffers && BufferWordConstantsRequested;

    // Read on its own: PortableBuffers is a static of another part of this class, and the order
    // the parts' static initializers run in is not defined.
    private static readonly bool BufferWordConstantsRequested =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_BUFFER_WORD_CONSTANTS") != "0";

    /// <summary>
    /// Compute pipelines take variants too, unless SHARPEMU_METAL_V2_BUFFER_WORD_COMPUTE=0 (or
    /// the constants are off): then every dispatch runs the generic pipeline, while render
    /// pipelines keep theirs. It changes no translated source, only which state a dispatch binds.
    /// </summary>
    internal static bool BufferWordComputeVariants => BufferWordConstants && BufferWordComputeRequested;

    private static readonly bool BufferWordComputeRequested =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_BUFFER_WORD_COMPUTE") != "0";

    /// <summary>Whether this host's dispatches take compute variants; tests turn it off.</summary>
    internal bool ComputeBufferWordVariants { get; set; } = BufferWordComputeVariants;

    /// <summary>The translation switches that decide the MSL, for the stamps of what is stored by them.</summary>
    internal static string TranslationSwitchStamp =>
        $"portable={(PortableBuffers ? 1 : 0)},buffer_word_constants={(BufferWordConstants ? 1 : 0)}";

    /// <summary>Variants one pipeline entry keeps; a draw of another layout past them draws generic.</summary>
    internal const int MaxBufferWordVariantsPerPipeline = 16;

    /// <summary>
    /// Specialized functions one module makes; past them its variants stay generic. Some 1,260
    /// specializations of one library in one process logged "AGX: exceeded compiled variants
    /// footprint limit".
    /// </summary>
    internal const int MaxBufferWordSpecializationsPerModule = 64;

    private static readonly nint SelBufferWordRetain = MetalNative.Selector("retain");
    private const int BufferWordVariantSlots = 32;
    private const nuint MetalDataTypeUInt = 33;
    private const int BufferWordVariantPending = 0;
    private const int BufferWordVariantReady = 1;
    private const int BufferWordVariantFailed = 2;
    private const int BufferWordVariantCapped = 3;
    private const int BufferWordVariantAbandoned = 4;

    /// <summary>
    /// The function <paramref name="entry"/> of a translated library. A source that declares
    /// function constants (the LOD statistics switch when <paramref name="mipStatistics"/> is
    /// set, buffer words when <paramref name="declared"/> is not empty) is made with
    /// <c>newFunctionWithName:constantValues:error:</c>, with the values given and none for the
    /// buffer words when <paramref name="values"/> is empty: a function of such a source made
    /// with the plain selector fails pipeline validation and aborts the process. The caller
    /// owns the function, or zero with <paramref name="error"/>.
    /// </summary>
    internal static unsafe nint NewTranslatedFunction(
        nint library,
        string entry,
        bool? mipStatistics,
        ReadOnlySpan<Gen5MslShader.BufferWordConstant> declared,
        ReadOnlySpan<uint> values,
        out nint error)
    {
        error = 0;
        if (mipStatistics is null && declared.IsEmpty)
        {
            return MetalNative.Send(library, MetalNative.Selector("newFunctionWithName:"), MetalNative.NsString(entry));
        }

        if (!values.IsEmpty && values.Length != declared.Length)
        {
            throw new ArgumentException(
                $"Buffer-word values do not match the constants declared: values={values.Length} declared={declared.Length}.",
                nameof(values));
        }

        var constants = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLFunctionConstantValues"), MetalNative.Selector("alloc")),
            MetalNative.Selector("init"));
        try
        {
            var setValue = MetalNative.Selector("setConstantValue:type:atIndex:");
            var mip = mipStatistics == true ? (byte)1 : (byte)0;
            if (mipStatistics is not null)
            {
                MetalNative.SendSetConstantValue(
                    constants, setValue, (nint)(&mip), MetalDataTypeBool, Gen5MslShader.MipStatisticsFunctionConstantIndex);
            }

            fixed (uint* words = values)
            {
                for (var index = 0; index < values.Length; index++)
                {
                    MetalNative.SendSetConstantValue(constants, setValue, (nint)(words + index), MetalDataTypeUInt, declared[index].Index);
                }

                return MetalNative.Send(
                    library,
                    MetalNative.Selector("newFunctionWithName:constantValues:error:"),
                    MetalNative.NsString(entry),
                    constants,
                    ref error);
            }
        }
        finally
        {
            MetalNative.SendVoid(constants, MetalNative.Selector("release"));
        }
    }

    /// <summary>As <see cref="NewTranslatedFunction(nint, string, bool?, ReadOnlySpan{Gen5MslShader.BufferWordConstant}, ReadOnlySpan{uint}, out nint)"/>,
    /// the generic function of a translated shader.</summary>
    internal static nint NewTranslatedFunction(nint library, Gen5MslShader shader, bool mipStatistics, out nint error) =>
        NewTranslatedFunction(
            library,
            shader.EntryPoint,
            shader.UsesMipStatisticsConstant ? mipStatistics : null,
            BufferWordConstantsOf(shader),
            [],
            out error);

    private static ReadOnlySpan<Gen5MslShader.BufferWordConstant> BufferWordConstantsOf(Gen5MslShader shader) =>
        shader.BufferWordConstants switch
        {
            Gen5MslShader.BufferWordConstant[] array => array,
            List<Gen5MslShader.BufferWordConstant> list => CollectionsMarshal.AsSpan(list),
            var other => other.ToArray(),
        };

    /// <summary>
    /// The archive key a variant's shader pair is listed under: the generic pair's pixel key
    /// mixed with the variant's values (<see cref="BufferWordVariant.ArchiveKey"/>), so the
    /// same libraries list their generic pipeline and each variant apart.
    /// </summary>
    internal static UInt128 BufferWordArchiveKey(UInt128 pixelPairKey, UInt128 values) =>
        values == default ? pixelPairKey : pixelPairKey ^ values ^ new UInt128(0x4255464645525F57UL, 0x4F5244535F564152UL);

    /// <summary>
    /// The archive key a compute variant's library file (library and pipeline binary,
    /// MetalBinaryArchive.CaptureCompute) is stored under: the module's library key mixed with
    /// the variant's values, apart from the generic pipeline's file and from every other
    /// variant's. None (zero) when the module has no key: the values alone would name the
    /// same file for every program.
    /// </summary>
    internal static UInt128 ComputeBufferWordArchiveKey(UInt128 libraryKey, UInt128 values) =>
        libraryKey == default || values == default
            ? libraryKey
            : libraryKey ^ values ^ new UInt128(0x434F4D505554455FUL, 0x42575F5641524941UL);

    // The constants of a pipeline's programs, its variants, and a scratch row for the values a
    // draw holds. On the GPU worker but for the builder's reads of Vertex, Pixel and a
    // variant it was handed. A compute pipeline's program is its Vertex; its Pixel is empty.
    private sealed class BufferWordVariants(Gen5MslShader.BufferWordConstant[] vertex, Gen5MslShader.BufferWordConstant[] pixel)
    {
        public readonly Gen5MslShader.BufferWordConstant[] Vertex = vertex;
        public readonly Gen5MslShader.BufferWordConstant[] Pixel = pixel;
        public readonly uint[] Scratch = new uint[vertex.Length + pixel.Length];
        public readonly BufferWordVariant?[] Table = new BufferWordVariant?[BufferWordVariantSlots];
        public int Count;
    }

    // One variant: the values (vertex constants first), their hash, the archive key, and the
    // state once built. Status moves from pending exactly once, by compare-exchange: to a
    // final state on the builder, or to abandoned on the worker when the entry lets go first.
    private sealed class BufferWordVariant(uint[] values, ulong hash, UInt128 archiveKey)
    {
        public readonly uint[] Values = values;
        public readonly ulong Hash = hash;
        public readonly UInt128 ArchiveKey = archiveKey;
        public nint State;
        public int Status;
    }

    // The specialized functions a variant pipeline is made from; zero takes the generic one.
    private sealed record BufferWordPipeline(nint VertexFunction, nint PixelFunction, uint[] Values, int VertexCount, UInt128 ArchiveKey);

    // A module's specialized function, by its values (and the LOD statistics switch of a pixel function).
    private sealed class SpecializationKey(bool? mipStatistics, uint[] values) : IEquatable<SpecializationKey>
    {
        private readonly bool? _mipStatistics = mipStatistics;
        private readonly uint[] _values = values;

        public bool Equals(SpecializationKey? other) =>
            other is not null && other._mipStatistics == _mipStatistics && other._values.AsSpan().SequenceEqual(_values);

        public override bool Equals(object? obj) => Equals(obj as SpecializationKey);

        public override int GetHashCode()
        {
            var hash = new HashCode();
            hash.Add(_mipStatistics);
            foreach (var value in _values)
            {
                hash.Add(value);
            }

            return hash.ToHashCode();
        }
    }

    // The stages of the draw being recorded, from CommitBindings, whose shader data the
    // variant lookup reads. On the GPU worker.
    private MetalStageBindings? _bufferWordVertexStage;
    private MetalStageBindings? _bufferWordPixelStage;

    // The compute stage CommitBindings(Compute) committed and the entry it committed it for,
    // whose shader data BindPipeline(Compute) chooses the dispatch's state from; and that
    // choice (zero: the generic state), which EnsureComputePipelineState binds for
    // _pendingComputePipeline. On the GPU worker.
    private MetalStageBindings? _bufferWordComputeStage;
    private MetalPipelineEntry? _bufferWordComputeEntry;
    private nint _pendingComputeState;

    private readonly BlockingCollection<(MetalPipelineEntry Entry, BufferWordVariants Variants, BufferWordVariant Variant)> _bufferWordQueue = new();
    private MetalCompileThread[]? _bufferWordBuilders;
    private volatile bool _bufferWordStopping;

    // The counters of one pipeline kind: the builder's (Interlocked), and the worker's per draw
    // or dispatch of a program that declares buffer words, taken by the PERF line.
    private sealed class BufferWordTally
    {
        public int Pending;
        public long Built;
        public long Failed;
        public long Capped;
        public long Variant;
        public long Generic;
        public long GenericPending;
        public long GenericCapped;
        public long GenericKind;
        public long GenericUnreadable;
        public long GenericFailed;
        public long GenericOff;
    }

    private readonly BufferWordTally _bufferWordRender = new();
    private readonly BufferWordTally _bufferWordCompute = new();

    private BufferWordTally TallyOf(MetalPipelineEntry entry) => entry.Graphics ? _bufferWordRender : _bufferWordCompute;

    /// <summary>Runs on the builder before it compiles a variant; tests hold it.</summary>
    internal Action? BeforeBufferWordVariantBuild { get; set; }

    /// <summary>Variants queued or compiling, render and compute.</summary>
    internal int BufferWordVariantsPending =>
        Volatile.Read(ref _bufferWordRender.Pending) + Volatile.Read(ref _bufferWordCompute.Pending);

    /// <summary>The render variant counters since the host started, for tests.</summary>
    internal (long Built, long Failed, long Capped, long DrawsVariant, long DrawsGeneric, long DrawsPending, long DrawsCapped) BufferWordCounters =>
        CountersOf(_bufferWordRender);

    /// <summary>The compute variant counters since the host started (dispatches for draws), for tests.</summary>
    internal (long Built, long Failed, long Capped, long DrawsVariant, long DrawsGeneric, long DrawsPending, long DrawsCapped) BufferWordComputeCounters =>
        CountersOf(_bufferWordCompute);

    /// <summary>Compute dispatches of declaring programs that ran generic because the compute switch was off, for tests.</summary>
    internal long BufferWordComputeDispatchesOff => _bufferWordCompute.GenericOff;

    private static (long, long, long, long, long, long, long) CountersOf(BufferWordTally tally) =>
        (Interlocked.Read(ref tally.Built), Interlocked.Read(ref tally.Failed), Interlocked.Read(ref tally.Capped),
            tally.Variant, tally.Generic, tally.GenericPending, tally.GenericCapped);

    /// <summary>Waits until no variant is queued or compiling, for tests.</summary>
    internal bool WaitForBufferWordVariants(TimeSpan timeout)
    {
        var deadline = System.Diagnostics.Stopwatch.GetTimestamp() + (long)(timeout.TotalSeconds * System.Diagnostics.Stopwatch.Frequency);
        while (BufferWordVariantsPending != 0)
        {
            if (System.Diagnostics.Stopwatch.GetTimestamp() > deadline)
            {
                return false;
            }

            Thread.Sleep(1);
        }

        return true;
    }

    /// <summary>The stages of the draw being recorded; CommitBindings sets them, tests too.</summary>
    internal void NoteBufferWordStages(MetalStageBindings? vertex, MetalStageBindings? pixel)
    {
        _bufferWordVertexStage = vertex;
        _bufferWordPixelStage = pixel;
    }

    private void NoteBufferWordStages(ReadOnlySpan<IPreparedBindings> stages)
    {
        MetalStageBindings? vertex = null;
        MetalStageBindings? pixel = null;
        foreach (var prepared in stages)
        {
            var stage = (MetalStageBindings)prepared;
            switch (StageIndexOf(stage.Program.Stage))
            {
                case StageVertex:
                    vertex = stage;
                    break;
                case StageFragment:
                    pixel = stage;
                    break;
            }
        }

        NoteBufferWordStages(vertex, pixel);
    }

    /// <summary>The compute stage CommitBindings(Compute) committed for <paramref name="entry"/>; ResetBindings clears it.</summary>
    private void NoteBufferWordComputeStage(MetalPipelineEntry? entry, ReadOnlySpan<IPreparedBindings> stages)
    {
        _bufferWordComputeEntry = entry;
        _bufferWordComputeStage = stages.Length == 1 ? (MetalStageBindings)stages[0] : null;
    }

    // The variants of a new compute pipeline entry: null when its program declares no buffer words.
    private static BufferWordVariants? ComputeBufferWordVariantsFor(MetalShaderModule module) =>
        module.Shader.BufferWordConstants.Count == 0 ? null : new BufferWordVariants([.. module.Shader.BufferWordConstants], []);

    // The variants of a new pipeline entry: null when neither program declares buffer words,
    // or when its kind stays generic (mesh, rectangle list, tessellation).
    private static BufferWordVariants? BufferWordVariantsFor(
        GraphicsPipelineDescription description,
        MetalShaderModule vertexModule,
        MetalShaderModule? pixelModule,
        Gen5MslMeshInfo? mesh,
        bool rectangleList)
    {
        if (mesh is not null || rectangleList || vertexModule.Shader.Tessellation is not null)
        {
            return null;
        }

        var vertex = vertexModule.Shader.BufferWordConstants;
        // A remapped pixel function is a library of its own (RequireRemappedPixelFunction).
        var pixel = pixelModule is not null && description.ColorRemap == 0
            ? pixelModule.Shader.BufferWordConstants
            : [];
        return vertex.Count + pixel.Count == 0 ? null : new BufferWordVariants([.. vertex], [.. pixel]);
    }

    private static bool DeclaresBufferWords(MetalShaderModule vertexModule, MetalShaderModule? pixelModule) =>
        vertexModule.Shader.BufferWordConstants.Count != 0 || pixelModule is { Shader.BufferWordConstants.Count: not 0 };

    /// <summary>
    /// The render pipeline state a draw of <paramref name="entry"/> draws with: the variant
    /// made for the words the draw's shader data holds once it is ready, else the generic
    /// state, with the reason counted. On the GPU worker, once per draw: a hash of the
    /// declared words and a probe of a small open-addressed table; a new layout allocates
    /// its variant and queues its build.
    /// </summary>
    private nint GraphicsStateOf(MetalPipelineEntry entry)
    {
        if (!entry.DeclaresBufferWords)
        {
            return entry.State;
        }

        var tally = _bufferWordRender;
        if (entry.BufferWordVariants is not { } variants)
        {
            tally.GenericKind++;
            tally.Generic++;
            return entry.State;
        }

        var state = VariantStateOf(entry, variants, _bufferWordVertexStage, _bufferWordPixelStage, tally);
        return state != 0 ? state : entry.State;
    }

    /// <summary>
    /// The variant state a dispatch of the compute <paramref name="entry"/> binds, chosen from
    /// the shader data of the stage CommitBindings(Compute) committed for that same entry, or
    /// zero for the generic state, with the reason counted. On the GPU worker, once per
    /// dispatch (BindPipeline).
    /// </summary>
    private nint ComputeStateOf(MetalPipelineEntry entry)
    {
        if (!entry.DeclaresBufferWords)
        {
            return 0;
        }

        var tally = _bufferWordCompute;
        if (!ComputeBufferWordVariants)
        {
            tally.GenericOff++;
            tally.Generic++;
            return 0;
        }

        if (entry.BufferWordVariants is not { } variants)
        {
            tally.GenericKind++;
            tally.Generic++;
            return 0;
        }

        if (!ReferenceEquals(_bufferWordComputeEntry, entry))
        {
            // The committed stage is not this pipeline's (or none was): its words are not this dispatch's.
            tally.GenericUnreadable++;
            tally.Generic++;
            return 0;
        }

        return VariantStateOf(entry, variants, _bufferWordComputeStage, null, tally);
    }

    // The ready variant for the words the stages hold, or zero with the reason the generic
    // state is used counted: a hash of the declared words and a probe of a small
    // open-addressed table; a new layout allocates its variant and queues its build.
    private nint VariantStateOf(
        MetalPipelineEntry entry, BufferWordVariants variants, MetalStageBindings? first, MetalStageBindings? second,
        BufferWordTally tally)
    {
        var values = variants.Scratch;
        if (!TryReadBufferWords(first, variants.Vertex, values.AsSpan(0, variants.Vertex.Length)) ||
            !TryReadBufferWords(second, variants.Pixel, values.AsSpan(variants.Vertex.Length)))
        {
            tally.GenericUnreadable++;
            tally.Generic++;
            return 0;
        }

        var hash = XxHash3.HashToUInt64(MemoryMarshal.AsBytes(values.AsSpan()));
        var table = variants.Table;
        var slot = (int)hash & (BufferWordVariantSlots - 1);
        while (table[slot] is { } candidate)
        {
            if (candidate.Hash == hash && candidate.Values.AsSpan().SequenceEqual(values))
            {
                switch (Volatile.Read(ref candidate.Status))
                {
                    case BufferWordVariantReady:
                        tally.Variant++;
                        return candidate.State;
                    case BufferWordVariantPending:
                        tally.GenericPending++;
                        break;
                    case BufferWordVariantCapped:
                        tally.GenericCapped++;
                        break;
                    default:
                        tally.GenericFailed++;
                        break;
                }

                tally.Generic++;
                return 0;
            }

            slot = (slot + 1) & (BufferWordVariantSlots - 1);
        }

        tally.Generic++;
        if (variants.Count >= MaxBufferWordVariantsPerPipeline)
        {
            tally.GenericCapped++;
            MetalV2Diagnostics.WarnOnce(
                entry.Graphics ? "buffer-word-variant-cap" : "buffer-word-compute-variant-cap",
                $"A pipeline {(entry.Graphics ? "drew" : "dispatched")} more buffer-word layouts than it keeps variants for; " +
                $"the rest run generic: pipeline={entry.Id} {(entry.Graphics ? "vertex" : "compute")}=0x{entry.VertexHash:X16} " +
                $"variants={variants.Count}.");
            return 0;
        }

        var copy = values.ToArray();
        var variant = new BufferWordVariant(copy, hash, VariantArchiveKey(copy, variants.Vertex.Length));
        table[slot] = variant;
        variants.Count++;
        tally.GenericPending++;
        QueueBufferWordVariant(entry, variants, variant);
        return 0;
    }

    // The words a stage's shader data holds at the dwords its program declares constants for.
    private static bool TryReadBufferWords(
        MetalStageBindings? stage, Gen5MslShader.BufferWordConstant[] declared, Span<uint> values)
    {
        if (declared.Length == 0)
        {
            return true;
        }

        if (stage is null)
        {
            return false;
        }

        var data = stage.ShaderData;
        for (var index = 0; index < declared.Length; index++)
        {
            var dword = declared[index].Dword;
            if (dword >= (uint)data.Length)
            {
                return false;
            }

            values[index] = data[dword];
        }

        return true;
    }

    private static UInt128 VariantArchiveKey(uint[] values, int vertexCount)
    {
        var key = XxHash128.HashToUInt128(MemoryMarshal.AsBytes(values.AsSpan()), vertexCount);
        return key == default ? UInt128.One : key;
    }

    private void QueueBufferWordVariant(MetalPipelineEntry entry, BufferWordVariants variants, BufferWordVariant variant)
    {
        if (_bufferWordBuilders is null)
        {
            // Two builders: a variant waits behind the game's compiles anyway (EnterPrewarm).
            _bufferWordBuilders = new MetalCompileThread[2];
            for (var index = 0; index < _bufferWordBuilders.Length; index++)
            {
                _bufferWordBuilders[index] = MetalCompileThread.Start(
                    "SharpEmu Metal v2 buffer-word variants", SharpEmu.HLE.Host.HostThreadQos.Utility, RunBufferWordBuilder);
            }
        }

        var tally = TallyOf(entry);
        Interlocked.Increment(ref tally.Pending);
        if (_bufferWordStopping || !_bufferWordQueue.TryAdd((entry, variants, variant)))
        {
            Interlocked.CompareExchange(ref variant.Status, BufferWordVariantFailed, BufferWordVariantPending);
            Interlocked.Decrement(ref tally.Pending);
        }
    }

    private void RunBufferWordBuilder()
    {
        try
        {
            foreach (var (entry, variants, variant) in _bufferWordQueue.GetConsumingEnumerable())
            {
                // Ahead of need: a pipeline Metal refuses for a variant must not end the
                // process (the generic pipeline draws the same), so a fatal inside the build
                // is thrown to BuildBufferWordVariant, which counts the variant as failed.
                SharpEmu.Libs.Gpu.Scheduling.GpuFatal.Speculatively(() =>
                {
                    BuildBufferWordVariant(entry, variants, variant);
                    return 0;
                });
            }
        }
        catch (ObjectDisposedException)
        {
            // The host went away.
        }
    }

    private void BuildBufferWordVariant(MetalPipelineEntry entry, BufferWordVariants variants, BufferWordVariant variant)
    {
        var status = BufferWordVariantFailed;
        nint state = 0;
        nint vertexFunction = 0;
        nint pixelFunction = 0;
        try
        {
            if (_bufferWordStopping || Volatile.Read(ref variant.Status) != BufferWordVariantPending)
            {
                status = BufferWordVariantAbandoned;
                return;
            }

            BeforeBufferWordVariantBuild?.Invoke();
            using var slot = MetalCompileScheduler.Shared.EnterPrewarm(() => _bufferWordStopping);
            if (!slot.Entered)
            {
                status = BufferWordVariantAbandoned;
                return;
            }

            // Made from now on: the bound keeps a module it made a pipeline from in the last
            // seconds (TrimModules), as it does for the compiler threads' builds.
            var now = System.Diagnostics.Stopwatch.GetTimestamp();
            if (!entry.Graphics)
            {
                var compute = entry.ComputeDescription ??
                    throw new InvalidOperationException($"A compute buffer-word variant has no description: pipeline={entry.Id}.");
                var module = RequireModule(compute.Program, onWorker: false);
                module.LastUsedTicks = now;
                // Made as the module's own function is: with the LOD statistics switch off.
                var computeMip = module.Shader.UsesMipStatisticsConstant ? false : (bool?)null;
                // Its own library file in the archive (library and pipeline binary), made from
                // it again when a past session stored it.
                var archiveKey = ComputeBufferWordArchiveKey(module.ArchiveKey, variant.ArchiveKey);
                if ((vertexFunction = RetainSpecializedFunction(module, computeMip, variants.Vertex, variant.Values, archiveKey)) == 0)
                {
                    status = BufferWordVariantCapped;
                    return;
                }

                // No empty-kernel stand-in when the compiler fails: the generic state runs instead.
                state = BuildComputePipelineState(vertexFunction, compute.Stage.Hash, archiveKey, variant: true);
                status = BufferWordVariantReady;
                return;
            }

            var description = entry.GraphicsDescription ??
                throw new InvalidOperationException($"A buffer-word variant has no description: pipeline={entry.Id}.");
            var vertexModule = RequireModule(description.VertexProgram, onWorker: false);
            var pixelModule = description.PixelStage is not null && description.PixelProgram.IsValid
                ? RequireModule(description.PixelProgram, onWorker: false)
                : null;
            vertexModule.LastUsedTicks = now;
            if (pixelModule is not null)
            {
                pixelModule.LastUsedTicks = now;
            }

            var vertexCount = variants.Vertex.Length;
            if (vertexCount != 0 &&
                (vertexFunction = RetainSpecializedFunction(vertexModule, null, variants.Vertex, variant.Values[..vertexCount])) == 0)
            {
                status = BufferWordVariantCapped;
                return;
            }

            if (variants.Pixel.Length != 0)
            {
                var mip = pixelModule!.Shader.UsesMipStatisticsConstant ? description.LodStatistics : (bool?)null;
                if ((pixelFunction = RetainSpecializedFunction(pixelModule, mip, variants.Pixel, variant.Values[vertexCount..])) == 0)
                {
                    status = BufferWordVariantCapped;
                    return;
                }
            }

            state = CreateRenderPipelineState(
                description, vertexModule, pixelModule, null, false, Mtl.ControlPointIndexTypeNone, out _,
                new BufferWordPipeline(vertexFunction, pixelFunction, variant.Values, vertexCount, variant.ArchiveKey));
            status = BufferWordVariantReady;
        }
        catch (Exception exception)
        {
            // The draws keep the generic pipeline, which gives the same result; the counter
            // and this line say the variant is missing.
            if (_bufferWordStopping)
            {
                status = BufferWordVariantAbandoned;
                return;
            }

            Console.Error.WriteLine(
                $"[METAL-V2][WARN] A buffer-word variant could not be built; its " +
                $"{(entry.Graphics ? "draws" : "dispatches")} stay generic: pipeline={entry.Id} " +
                $"{(entry.Graphics ? "vertex" : "compute")}=0x{entry.VertexHash:X16} {exception.Message}");
        }
        finally
        {
            foreach (var function in (ReadOnlySpan<nint>)[vertexFunction, pixelFunction])
            {
                if (function != 0)
                {
                    MetalNative.SendVoid(function, SelRelease);
                }
            }

            var tally = TallyOf(entry);
            PublishBufferWordVariant(variant, status, state, tally);
            Interlocked.Decrement(ref tally.Pending);
        }
    }

    private void PublishBufferWordVariant(BufferWordVariant variant, int status, nint state, BufferWordTally tally)
    {
        if (status == BufferWordVariantReady)
        {
            variant.State = state;
        }

        var published = status != BufferWordVariantAbandoned &&
            Interlocked.CompareExchange(ref variant.Status, status, BufferWordVariantPending) == BufferWordVariantPending;
        if (!published)
        {
            // The entry let go of it first (or the host is stopping): nothing will draw with it.
            if (state != 0)
            {
                ReleaseState(state);
            }

            Interlocked.CompareExchange(ref variant.Status, BufferWordVariantAbandoned, BufferWordVariantPending);
            return;
        }

        if (status == BufferWordVariantReady)
        {
            Interlocked.Increment(ref tally.Built);
        }
        else if (status == BufferWordVariantCapped)
        {
            Interlocked.Increment(ref tally.Capped);
        }
        else
        {
            Interlocked.Increment(ref tally.Failed);
        }
    }

    /// <summary>
    /// The function of <paramref name="module"/> specialized with <paramref name="values"/>,
    /// made once and kept with the module like its LOD statistics function, retained for the
    /// caller; zero once the module has made <see cref="MaxBufferWordSpecializationsPerModule"/>.
    /// The function is made outside the module's lock: the worker takes it to release an idle
    /// module, and must not wait for a specialization meanwhile.
    /// </summary>
    private nint RetainSpecializedFunction(
        MetalShaderModule module, bool? mipStatistics, Gen5MslShader.BufferWordConstant[] declared, uint[] values,
        UInt128 storedVariant = default)
    {
        var key = new SpecializationKey(mipStatistics, values);
        lock (module)
        {
            RequireResidentModule(module);
            if (module.SpecializedFunctions?.TryGetValue(key, out var known) == true)
            {
                MetalNative.Send(known, SelBufferWordRetain);
                return known;
            }

            if ((module.SpecializedFunctions?.Count ?? 0) >= MaxBufferWordSpecializationsPerModule)
            {
                return 0;
            }
        }

        // A compute variant a past session stored (storedVariant): its file's library is the
        // function as specialized with these values, and its pipeline binary is in the same
        // file, so neither the front end nor the back end runs. Otherwise a library that still
        // has the constants.
        var library = storedVariant != default && MetalBinaryArchive.For(_device) is { } archive && archive.IsStored(storedVariant)
            ? archive.TryLoadLibrary(storedVariant)
            : 0;
        if (library == 0)
        {
            library = RetainSpecializableLibrary(module);
        }

        nint function;
        nint error;
        try
        {
            function = NewTranslatedFunction(library, module.Shader.EntryPoint, mipStatistics, declared, values, out error);
        }
        finally
        {
            MetalNative.SendVoid(library, SelRelease);
        }

        if (function == 0)
        {
            throw new InvalidOperationException(
                $"The specialized function could not be made: hash=0x{module.Hash:X16} {MetalNative.DescribeError(error)}.");
        }

        lock (module)
        {
            if (module.Library == 0 || module.Function == 0)
            {
                MetalNative.SendVoid(function, SelRelease);
                throw new InvalidOperationException(
                    $"The module's library was released while it was specialized: hash=0x{module.Hash:X16}.");
            }

            module.SpecializedFunctions ??= new();
            if (module.SpecializedFunctions.TryGetValue(key, out var winner))
            {
                // Another builder made the same one first.
                MetalNative.SendVoid(function, SelRelease);
                function = winner;
            }
            else
            {
                module.SpecializedFunctions.Add(key, function);
            }

            MetalNative.Send(function, SelBufferWordRetain);
            return function;
        }

        static void RequireResidentModule(MetalShaderModule module)
        {
            if (module.Library == 0 || module.Function == 0)
            {
                throw new InvalidOperationException($"The module's library was released: hash=0x{module.Hash:X16}.");
            }
        }
    }

    /// <summary>
    /// A library of <paramref name="module"/> whose functions can still be specialized,
    /// retained for the caller. The module's own library is not one when it came from the
    /// title's archive (TryCompileLibrary asks the archive first, so on every start after the
    /// first): a library file holds the AIR of the function the pipeline that stored it was
    /// made with, its constants already resolved, and a function made from it with buffer
    /// words reads the shader data as the generic one does (measured: the values are ignored).
    /// So the module's MSL is compiled once more, without asking the archive, on the variant
    /// builder, and kept with the module.
    /// </summary>
    private nint RetainSpecializableLibrary(MetalShaderModule module)
    {
        lock (module)
        {
            if (module.SpecializationLibrary != 0)
            {
                MetalNative.Send(module.SpecializationLibrary, SelBufferWordRetain);
                return module.SpecializationLibrary;
            }
        }

        var shader = module.Compiled.AcquireShader();
        var library = TryCompileLibrary(_device, shader.Source, shader.UsesPerVertexInputs, out var error, out _, askArchive: false);
        if (library == 0)
        {
            throw new InvalidOperationException(
                $"The module's library could not be compiled for specialization: hash=0x{module.Hash:X16} {error}");
        }

        lock (module)
        {
            if (module.Library == 0 || module.Function == 0)
            {
                MetalNative.SendVoid(library, SelRelease);
                throw new InvalidOperationException($"The module's library was released while it was specialized: hash=0x{module.Hash:X16}.");
            }

            if (module.SpecializationLibrary != 0)
            {
                // Another builder made it first.
                MetalNative.SendVoid(library, SelRelease);
            }
            else
            {
                module.SpecializationLibrary = library;
            }

            MetalNative.Send(module.SpecializationLibrary, SelBufferWordRetain);
            return module.SpecializationLibrary;
        }
    }

    // Under the module's lock.
    private static void ReleaseSpecializedFunctions(MetalShaderModule module)
    {
        if (module.SpecializationLibrary != 0)
        {
            MetalNative.SendVoid(module.SpecializationLibrary, SelRelease);
            module.SpecializationLibrary = 0;
        }

        if (module.SpecializedFunctions is not { } functions)
        {
            return;
        }

        foreach (var function in functions.Values)
        {
            MetalNative.SendVoid(function, SelRelease);
        }

        module.SpecializedFunctions = null;
    }

    // The variants of an entry whose states go (the bound released it, or the host is going).
    // On the GPU worker. One still compiling is abandoned: its builder releases what it made.
    private void ReleaseBufferWordVariants(MetalPipelineEntry entry)
    {
        if (entry.BufferWordVariants is not { Count: not 0 } variants)
        {
            return;
        }

        if (ReferenceEquals(entry, _pendingComputePipeline))
        {
            // The bound dispatch's choice goes with them: it runs the generic state, made again.
            _pendingComputeState = 0;
        }

        var table = variants.Table;
        for (var index = 0; index < table.Length; index++)
        {
            if (table[index] is not { } variant)
            {
                continue;
            }

            if (Interlocked.CompareExchange(ref variant.Status, BufferWordVariantAbandoned, BufferWordVariantPending) !=
                    BufferWordVariantPending &&
                Volatile.Read(ref variant.Status) == BufferWordVariantReady)
            {
                ReleaseState(variant.State);
            }

            table[index] = null;
        }

        variants.Count = 0;
    }

    // Before the pipelines are released at disposal: the builders finish what they hold.
    private void StopBufferWordBuilders()
    {
        _bufferWordStopping = true;
        _bufferWordQueue.CompleteAdding();
        foreach (var builder in _bufferWordBuilders ?? [])
        {
            if (!builder.IsCurrentThread)
            {
                builder.Join(TimeSpan.FromSeconds(30));
            }
        }
    }

    // The PERF line's view: variants built, queued or compiling, and refused at a module's cap
    // or failed; then draws of programs that declare buffer words on a variant, on the generic
    // pipeline, and of those why: variant not ready, a pipeline's cap, a kind that stays
    // generic, words not readable, or a variant that failed or was capped.
    private string FormatBufferWordCounters() => FormatBufferWordCounters(_bufferWordRender, "draws");

    // The same for compute pipelines, per dispatch, and the dispatches the compute switch
    // (SHARPEMU_METAL_V2_BUFFER_WORD_COMPUTE=0) kept generic.
    private string FormatBufferWordComputeCounters() =>
        FormatBufferWordCounters(_bufferWordCompute, "dispatches") +
        string.Create(CultureInfo.InvariantCulture, $" generic_off={Interlocked.Exchange(ref _bufferWordCompute.GenericOff, 0)}");

    private static string FormatBufferWordCounters(BufferWordTally tally, string work) =>
        string.Create(CultureInfo.InvariantCulture,
            $"built={Interlocked.Read(ref tally.Built)} pending={Volatile.Read(ref tally.Pending)} " +
            $"capped={Interlocked.Read(ref tally.Capped)} failed={Interlocked.Read(ref tally.Failed)} " +
            $"{work}_variant={Interlocked.Exchange(ref tally.Variant, 0)} " +
            $"{work}_generic={Interlocked.Exchange(ref tally.Generic, 0)} " +
            $"generic_pending={Interlocked.Exchange(ref tally.GenericPending, 0)} " +
            $"generic_capped={Interlocked.Exchange(ref tally.GenericCapped, 0)} " +
            $"generic_kind={Interlocked.Exchange(ref tally.GenericKind, 0)} " +
            $"generic_unreadable={Interlocked.Exchange(ref tally.GenericUnreadable, 0)} " +
            $"generic_failed={Interlocked.Exchange(ref tally.GenericFailed, 0)}");
}
