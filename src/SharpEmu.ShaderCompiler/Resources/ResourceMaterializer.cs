// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace SharpEmu.ShaderCompiler.Resources;

// Resolves a plan against one draw: descriptor sources, indirect image tables and the
// specialization. Both outputs change only when the whole materialisation succeeds.
public static partial class ResourceMaterializer
{
    private const ulong AddressMask = 0x0000_FFFF_FFFF_FFFFul;
    private const ulong MaxIndirectImageProbes = 65536;

    /// <summary>The most entries a V#-described image table is read through a device handle table with.</summary>
    public const ulong MaxBindlessImageEntries = 1 << 20;

    /// <summary>The most class variants one bindless access switches over.</summary>
    public const int MaxBindlessImageVariants = 8;
    // The smallest and largest layouts of a table selected by a loop counter; see FitSharedImageTables.
    private const int MinSharedCandidates = 8;
    private const int MaxSharedCandidates = 256;
    private const int MinSharedKeyCapacity = 32;
    private const int MaxSharedKeyCapacity = 1024;

    // Written to standard error like every specialization refusal; the host turns it
    // into its fatal.
    public static Action<string> SpecializationFailed { get; set; } = message => Console.Error.WriteLine($"shader resource specialization failed: {message}");

    private sealed class IndirectImageTable
    {
        public uint Resource;
        public List<uint> Keys = [];
        public List<uint> Candidates = [];
        public List<DescriptorWords> Descriptors = [];
        public uint[] MaterialDescriptor = [];
        public uint[] HeapDescriptor = [];
        public uint SelectorStride;
        public uint SelectorOffset;
        public bool Shared;
        public int KeyCapacity;
        public IndirectSelectorDiagnostic? SelectorDiagnostic;
    }

    // A counted loop's counter against its bound, ordered as the loop's comparison orders them.
    internal static bool CounterWithinLimit(uint counter, uint bound, bool inclusive, bool signed) =>
        signed
            ? inclusive ? (int)counter <= (int)bound : (int)counter < (int)bound
            : inclusive ? counter <= bound : counter < bound;

    // Direct candidates in record order over a table whose end the guest does not state (DirectRecordsMayEnd):
    // the records the host can read are a prefix, found by a binary search, and read together; the first record
    // past the end of the guest's table and every later one read as null.
    private static bool TryReadRecordsUpToTheEnd(ShaderResourcePlan plan, uint[] sources, ResourceRuntimeInputs inputs,
        out List<DescriptorWords> descriptors)
    {
        bool Readable(int index) =>
            RuntimeValueEvaluator.EvaluateSources(plan, [sources[index]], inputs, [], evaluateTable: false, out _, out _);

        var readable = 0;
        if (sources.Length != 0 && Readable(0))
        {
            var low = 1;
            var high = sources.Length;
            while (low < high)
            {
                var middle = low + (high - low) / 2;
                if (Readable(middle)) low = middle + 1;
                else high = middle;
            }
            readable = low;
        }

        descriptors = [];
        if (readable != 0 &&
            !RuntimeValueEvaluator.EvaluateSources(plan, sources[..readable], inputs, [], evaluateTable: false, out descriptors, out _))
        {
            // A hole inside the table: one at a time.
            descriptors = sources[..readable].Select(source =>
                RuntimeValueEvaluator.EvaluateSources(plan, [source], inputs, [], evaluateTable: false, out var one, out _)
                    ? one[0] : DescriptorWords.Empty(8)).ToList();
        }

        for (var index = readable; index < sources.Length; index++)
            descriptors.Add(DescriptorWords.Empty(8));
        return true;
    }

    private sealed class MaterializedSnapshot
    {
        public uint[][] Buffers = [];
        public uint[][] Images = [];
        public uint[][] Samplers = [];
        public uint[] FlattenedTable = [];
        public uint[] UserData = [];
        public List<IndirectImageTable> IndirectImages = [];
        public List<BufferCandidateTable> BufferCandidateTables = [];
        public List<BindlessImageEntries> BindlessImages = [];
        public BindlessImageTable[] BindlessTables = [];
    }

    // A V#-described image table read for one draw: the distinct usable descriptors it holds
    // and, per entry, the one it names (NullEntry for a null or unusable one).
    private sealed class BindlessImageEntries
    {
        public uint Resource;
        public ulong Address;
        public uint DynamicOffsetBase;
        public uint RecordStride = 32;
        public List<uint[]> Descriptors = [];
        public int[] Entries = [];
        // Entries read as null before classification, by reason.
        public Dictionary<string, int> Nulls = [];
        // The memo of the table these entries came from, when its words were all read.
        public BindlessTableMemo? Memo;
    }

    // The last bindless table read at an address, kept while its guest words stay the same. Sorting
    // tens of thousands of descriptors into distinct ones and classes costs tens of milliseconds a
    // draw, and the host reuses a table's device handles only for the same BindlessImageTable object.
    private sealed class BindlessTableMemo
    {
        public required uint[] Words;
        public required List<uint[]> Descriptors;
        public required int[] Entries;
        public required Dictionary<string, int> Nulls;
        // The classifications of the accesses that read the table, by what each read of its access besides the
        // words: the resource, dimension and swizzle. A few shaders often share one table.
        public readonly List<((uint Resource, ImageDimension Dimension, uint Swizzle) Access, BindlessClassification Classification)> Classes = [];
    }

    // Diagnostics: bindless table memo hits, words that changed, keys not kept, and tables not wholly read.
    public static long BindlessMemoHits, BindlessMemoChanged, BindlessMemoAbsent, BindlessMemoUnread, BindlessMemoClassHits;

    // A table of records (RecordStride above 32) is kept whatever its size: each of a material's textures is a table
    // of its own, and the host would make every draw's device handles anew.
    private const int BindlessMemoMinEntries = 256;
    private const int BindlessMemoCapacity = 512;
    // The words the kept tables hold together (128 MiB): a few descriptor heaps of 65536 entries fill it.
    private const long BindlessMemoWordBudget = 32L << 20;
    private static long BindlessMemoWords;
    private static readonly Dictionary<(ulong Address, ulong Entries, bool R128, uint DynamicOffsetBase, uint RecordStride), BindlessTableMemo> BindlessMemos = [];

    // SHARPEMU_TRACE_BINDLESS_TABLES=1: each V#-described table's reach - the V#, its offset and record stride, the key
    // bound and the entries enumerated - so a table cut short can be told from one the guest made small.
    private static readonly bool TraceBindlessTables =
        Environment.GetEnvironmentVariable("SHARPEMU_TRACE_BINDLESS_TABLES") == "1";
    private static readonly Lock BindlessMemoGate = new();

    // One bounded runtime V# table resolved for this draw: the distinct candidate descriptors
    // and the run-time offset -> candidate mapping the emitter searches.
    private sealed class BufferCandidateTable
    {
        public int MemoryIndex;
        public List<uint> Keys = [];
        public List<uint> Candidates = [];
        public List<DescriptorWords> Descriptors = [];
    }

    public static bool Materialize(
        ShaderResourcePlan plan,
        ResourceRuntimeInputs inputs,
        ref ResourceSnapshot snapshot,
        ref ResourceSpecialization specialization,
        Action<IndirectImageFailure>? captureIndirectImageFailure = null)
        => Materialize(plan, inputs, ref snapshot, ref specialization, out _, captureIndirectImageFailure);

    public static bool Materialize(
        ShaderResourcePlan plan,
        ResourceRuntimeInputs inputs,
        ref ResourceSnapshot snapshot,
        ref ResourceSpecialization specialization,
        out ResourceMaterializationFailure failure,
        Action<IndirectImageFailure>? captureIndirectImageFailure = null)
    {
        using var totalProfile = ResourceMaterializationProfile.Measure(ResourceMaterializationProfile.Phase.Total);
        if (!MaterializeSnapshot(plan, inputs, captureIndirectImageFailure is not null, out var materialized, out failure))
        {
            return false;
        }

        // The written ranges follow the table reads so every store can check its own.
        DeviceAddressRange[] ranges;
        using (ResourceMaterializationProfile.Measure(ResourceMaterializationProfile.Phase.DeviceAddressRanges))
            ranges = DeviceAddressRangePlanner.Evaluate(plan, inputs);
        foreach (var range in ranges)
        {
            if (!plan.WrittenRangeSlotByHandle.TryGetValue(range.Handle, out var slot))
            {
                continue;
            }

            var offset = (int)slot;
            materialized.FlattenedTable[offset] = (uint)range.Base;
            materialized.FlattenedTable[offset + 1] = (uint)(range.Base >> 32);
            materialized.FlattenedTable[offset + 2] = (uint)Math.Min(range.Size, uint.MaxValue);
        }

        var guard = SpecializationGuards ? Volatile.Read(ref plan.SpecializationGuard) : null;
        MaterializedSnapshot nextSnapshot;
        ResourceSpecialization nextSpecialization;
        if (guard is not null && MatchesGuard(plan, materialized, guard))
        {
            using var guardProfile = ResourceMaterializationProfile.Measure(ResourceMaterializationProfile.Phase.SpecializationGuardHit);
            ApplyGuard(materialized, guard);
            nextSnapshot = materialized;
            nextSpecialization = guard.Specialization.Clone();
        }
        else
        {
            // A guard is only kept when guards are on; the Metal backend turns them off, and
            // capturing one anyway copied every descriptor of every miss for nothing.
            var candidate = SpecializationGuards ? CaptureGuard(plan, materialized) : null;
            var originalSamplers = materialized.Samplers;
            if (!BuildSpecialization(plan, materialized, out nextSnapshot, out nextSpecialization, out failure, captureIndirectImageFailure))
                return false;
            if (SpecializationGuards && candidate is not null && MatchesGuard(plan, materialized, candidate, checkSamplers: false))
            {
                candidate.SamplerSources = materialized.Samplers.Select(words => Array.FindIndex(originalSamplers,
                    original => ReferenceEquals(original, words))).ToArray();
                if (candidate.SamplerSources.All(index => index >= 0))
                {
                    candidate.Specialization = nextSpecialization.Clone();
                    Volatile.Write(ref plan.SpecializationGuard, candidate);
                }
            }
        }

        using var outputProfile = ResourceMaterializationProfile.Measure(ResourceMaterializationProfile.Phase.OutputAssembly);
        snapshot = new ResourceSnapshot
        {
            Buffers = nextSnapshot.Buffers,
            Images = nextSnapshot.Images,
            Samplers = nextSnapshot.Samplers,
            FlattenedResourceTable = nextSnapshot.FlattenedTable,
            UserData = nextSnapshot.UserData,
            DeviceAddressRanges = ranges,
            BindlessImages = nextSnapshot.BindlessTables,
        };
        specialization = nextSpecialization;
        failure = ResourceMaterializationFailure.None;
        return true;
    }

    // Evaluates only the flattened table, laid out as a full materialization lays it out
    // before specialization; the written device-address slots are left zero for the caller.
    public static bool TryEvaluateTable(ShaderResourcePlan plan, ResourceRuntimeInputs inputs, out uint[] table)
    {
        var additionalTableWords = checked(plan.WrittenRangeCount * ShaderResourcePlan.WrittenRangeDwordCount);
        // Only this table-only caller can omit descriptor branch activity.
        if (RuntimeValueEvaluator.HasResourceStructure(plan) && plan.CompiledTableTransaction is { } transaction)
        {
            ResourceMaterializationCache.RecordEvaluationPath(plan, "compiled_table");
            return transaction.Evaluate(inputs, additionalTableWords, out _, out table);
        }
        return RuntimeValueEvaluator.EvaluateSources(plan, [], inputs, plan.CleanFlatSlots, evaluateTable: true,
            out _, out table, out _, additionalTableWords: additionalTableWords);
    }

    // ---- snapshot ----

    private static bool MaterializeSnapshot(ShaderResourcePlan plan, ResourceRuntimeInputs inputs,
        bool captureSelectorDiagnostic, out MaterializedSnapshot snapshot, out ResourceMaterializationFailure failure)
    {
        using var snapshotProfile = ResourceMaterializationProfile.Measure(ResourceMaterializationProfile.Phase.Snapshot);
        failure = ResourceMaterializationFailure.Other;
        snapshot = new MaterializedSnapshot();
        if (plan.RequiresSpecializationMemory && inputs.ReadCleanMemory is null)
        {
            return false;
        }

        if (!RuntimeValueEvaluator.EvaluateSources(plan, plan.MaterializationSources, inputs, plan.CleanFlatSlots,
            evaluateTable: true, out var values, out var table, out var activeSources,
            additionalTableWords: checked(plan.WrittenRangeCount * ShaderResourcePlan.WrittenRangeDwordCount)))
        {
            SpecializationFailed(DiagnoseSnapshotEvaluationFailure(plan, inputs, activeSources));
            return false;
        }

        var cursor = 0;
        snapshot.Buffers = new uint[plan.Info.Buffers.Count][];
        for (var index = 0; index < snapshot.Buffers.Length; index++)
            snapshot.Buffers[index] = values[cursor++].Dwords;
        snapshot.FlattenedTable = table;
        snapshot.Images = new uint[plan.Info.Images.Count][];
        for (var imageIndex = 0; imageIndex < plan.Info.Images.Count; imageIndex++)
        {
            var image = plan.Info.Images[imageIndex];
            var source = plan.DescriptorSources[(int)image.Source];
            if (source.IndirectImage is { } indirect)
            {
                if (activeSources.Length != 0 && !activeSources[image.Source])
                {
                    snapshot.Images[imageIndex] = new uint[8];
                    continue;
                }
                var cleanInputs = inputs.WithReader(inputs.ReadCleanMemory);
                if (indirect.DirectCandidates is { } directCandidates)
                {
                    // A counted loop reads only the records below its bound; the others stay null and unread. A bound
                    // the host cannot read leaves every record null, as the access read before it had candidates.
                    var reached = directCandidates.Select(_ => true).ToArray();
                    if (indirect.DirectLimitSource is { } limitSource)
                    {
                        var readable = RuntimeValueEvaluator.EvaluateSources(plan, [limitSource], cleanInputs, [], evaluateTable: false,
                            out var bounds, out _);
                        var bound = readable ? bounds[0].Dwords[0] : 0u;
                        if (readable && directCandidates[^1].Counter is { } lastCounter &&
                            CounterWithinLimit(lastCounter + 1, bound, indirect.DirectLimitInclusive, indirect.DirectLimitSigned) &&
                            Interlocked.Increment(ref LoopBoundWarnings) <= 8)
                        {
                            Console.Error.WriteLine(
                                $"[SHADER][WARN] A counted loop reads records past its {directCandidates.Count} candidates: hash=0x{plan.Hash:X16} " +
                                $"image={imageIndex} bound={bound}; the later records read the first record's image.");
                        }
                        for (var candidateIndex = 0; candidateIndex < reached.Length; candidateIndex++)
                        {
                            reached[candidateIndex] = directCandidates[candidateIndex].Counter is not { } counter ||
                                readable && CounterWithinLimit(counter, bound, indirect.DirectLimitInclusive, indirect.DirectLimitSigned);
                        }
                    }

                    var reachedSources = directCandidates.Where((_, candidateIndex) => reached[candidateIndex])
                        .Select(candidate => candidate.Source).ToArray();
                    List<DescriptorWords> reachedDescriptors;
                    if (indirect.DirectRecordsMayEnd)
                    {
                        if (!TryReadRecordsUpToTheEnd(plan, reachedSources, cleanInputs, out reachedDescriptors))
                            return false;
                    }
                    else if (!RuntimeValueEvaluator.EvaluateSources(plan, reachedSources, cleanInputs, [], evaluateTable: false,
                        out reachedDescriptors, out _))
                    {
                        return false;
                    }
                    var directTable = new IndirectImageTable { Resource = (uint)imageIndex };
                    var nextReached = 0;
                    for (var candidateIndex = 0; candidateIndex < directCandidates.Count; candidateIndex++)
                    {
                        var descriptor = reached[candidateIndex] ? reachedDescriptors[nextReached++] : DescriptorWords.Empty(8);
                        if (!UsableTableCandidate(descriptor.Dwords, image))
                            descriptor = DescriptorWords.Empty(8);
                        var existing = directTable.Descriptors.FindIndex(candidate => candidate.SameAs(descriptor));
                        if (existing < 0)
                        {
                            existing = directTable.Descriptors.Count;
                            directTable.Descriptors.Add(descriptor);
                        }
                        directTable.Keys.Add(directCandidates[candidateIndex].Offset);
                        directTable.Candidates.Add((uint)existing);
                    }
                    snapshot.Images[imageIndex] = directTable.Descriptors[(int)directTable.Candidates[0]].Dwords;
                    if (directTable.Descriptors.Count > 1) snapshot.IndirectImages.Add(directTable);
                    continue;
                }
                if (indirect.WaveIndexed is { } waveIndexed)
                {
                    if (!RuntimeValueEvaluator.EvaluateSources(plan, [indirect.HeapSource], cleanInputs, [], evaluateTable: false,
                        out var waveSources, out _))
                        return false;
                    if (!MaterializeWaveIndexedImage(indirect, waveIndexed, waveSources[0], image, image.R128, inputs, out var waveTable, out failure))
                        return false;
                    snapshot.Images[imageIndex] = waveTable.Descriptors[(int)waveTable.Candidates[0]].Dwords;
                    if (waveTable.Descriptors.Count > 1)
                    {
                        waveTable.Resource = (uint)imageIndex;
                        snapshot.IndirectImages.Add(waveTable);
                    }
                    continue;
                }
                if (indirect.Dense)
                {
                    if (!RuntimeValueEvaluator.EvaluateSources(plan, [indirect.HeapSource], cleanInputs, [], evaluateTable: false,
                        out var denseSources, out _))
                        return false;
                    if (indirect.Bindless)
                    {
                        if (!MaterializeBindlessImage(indirect, denseSources[0], image.R128, inputs, out var bindless, out failure))
                            return false;
                        // The root binds a null texture of its class; the table names everything else.
                        bindless.Resource = (uint)imageIndex;
                        snapshot.Images[imageIndex] = new uint[8];
                        snapshot.BindlessImages.Add(bindless);
                        continue;
                    }

                    if (!MaterializeDenseIndirectImage(indirect, denseSources[0], image, image.R128, inputs, out var denseTable, out failure))
                        return false;
                    // The first descriptor is the root: the first probed entry, or the null one a V# table leads with.
                    snapshot.Images[imageIndex] = denseTable.Descriptors[0].Dwords;
                    if (denseTable.Descriptors.Count > 1)
                    {
                        denseTable.Resource = (uint)imageIndex;
                        snapshot.IndirectImages.Add(denseTable);
                    }
                    continue;
                }
                IndirectImageTable indirectTable;
                if (!RuntimeValueEvaluator.EvaluateSources(plan, [indirect.MaterialSource, indirect.HeapSource], cleanInputs, [], evaluateTable: false, out var tables, out _) ||
                    !MaterializeIndirectImage(plan, indirect, tables[0], tables[1], image, inputs, captureSelectorDiagnostic, out indirectTable, out failure))
                {
                    if (!indirect.SharedSelector)
                    {
                        return false;
                    }

                    // The table is unreadable now: keep the null image this access read as before it had a table.
                    snapshot.Images[imageIndex] = new uint[8];
                    failure = ResourceMaterializationFailure.None;
                    continue;
                }

                snapshot.Images[imageIndex] = indirectTable.Descriptors[(int)indirectTable.Candidates[0]].Dwords;
                if (indirectTable.Descriptors.Count > 1)
                {
                    indirectTable.Resource = (uint)imageIndex;
                    indirectTable.Shared = indirect.SharedSelector;
                    snapshot.IndirectImages.Add(indirectTable);
                }
            }
            else
            {
                var descriptor = values[cursor++];
                if (!ValidImageDescriptor(descriptor.Dwords, image.R128))
                {
                    TraceInvalidDescriptor(plan, imageIndex, descriptor.Dwords);
                    TraceZeroDescriptor(plan, imageIndex, image.Source, descriptor.Dwords, inputs,
                        activeSources.Length == 0 || activeSources[image.Source]);
                    descriptor = DescriptorWords.Empty(descriptor.DwordCount);
                }

                snapshot.Images[imageIndex] = descriptor.Dwords;
            }
        }

        FitSharedImageTables(plan, snapshot);
        snapshot.Samplers = new uint[plan.Info.Samplers.Count][];
        for (var index = 0; index < snapshot.Samplers.Length; index++)
            snapshot.Samplers[index] = values[cursor++].Dwords;

        // Bounded runtime V# tables: the whole table is read and validated before it is
        // published, so a single unreadable candidate leaves the previous snapshot intact.
        foreach (var candidatePlan in plan.BufferCandidateTables)
        {
            if (candidatePlan.SourceSrtResource == DescriptorConstants.NoIndex ||
                !RuntimeValueEvaluator.EvaluateSources(plan, [candidatePlan.SourceSrtResource], inputs, [], evaluateTable: false,
                    out var srtSources, out _) || srtSources.Count == 0)
            {
                return false;
            }

            if (!MaterializeBufferCandidateTable(plan, candidatePlan, srtSources[0], inputs, out var candidateTable))
            {
                return false;
            }

            snapshot.BufferCandidateTables.Add(candidateTable);
        }

        // One draw's user data is built for that draw and never written again, so the snapshot
        // can hold the array it was given; a copy only matters once the snapshot is cached,
        // and a cached one is handed out carrying the user data of the draw that reuses it.
        snapshot.UserData = inputs.UserData as uint[] ?? inputs.UserData.ToArray();
        return true;
    }

    // Reads every candidate the loop guard can select from the guest SRT, validates it and
    // records the run-time probe key (its base-address low dword) -> candidate mapping.
    // Distinct keys are required so the mapping is unambiguous; identical descriptors share
    // one candidate.
    private static bool MaterializeBufferCandidateTable(
        ShaderResourcePlan plan,
        BufferCandidateTablePlan table,
        DescriptorWords srt,
        ResourceRuntimeInputs inputs,
        out BufferCandidateTable result)
    {
        result = new BufferCandidateTable { MemoryIndex = table.MemoryIndices.Count != 0 ? table.MemoryIndices[0] : -1 };
        if (srt.DwordCount != 4)
        {
            return false;
        }

        int count;
        if (table.IsStaticallyBounded)
        {
            count = table.Count;
        }
        else
        {
            if (table.Limit is null)
            {
                return false;
            }

            using var scratch = RuntimeEvaluationScratch.Rent();
            var evaluator = new RuntimeValueEvaluator(scratch, plan, inputs);
            if (!evaluator.Evaluate(table.Limit, out var limit) || !table.TryResolveCount(limit, out count))
            {
                return false;
            }
        }

        if (count <= 0 || count > table.Cap)
        {
            return false;
        }

        var keyToCandidate = new Dictionary<uint, uint>();
        for (var index = 0; index < count; index++)
        {
            var offset = unchecked(table.MinOffset + (uint)index * table.CandidateSpacing);
            var words = new uint[4];
            for (uint dword = 0; dword < 4; dword++)
            {
                if (!ReadScalarBufferWord(srt.Dwords, offset, dword * sizeof(uint), inputs, out words[dword]))
                {
                    return false;
                }
            }

            // A descriptor whose reserved bits are set is unbound and reads as zero.
            if ((words[3] >> 30) != 0)
            {
                Array.Clear(words);
            }

            var descriptor = new DescriptorWords(words);
            if (keyToCandidate.TryGetValue(words[0], out var existing))
            {
                // Two different descriptors sharing one probe key cannot be told apart at
                // run time; reject precisely instead of guessing.
                if (!result.Descriptors[(int)existing].SameAs(descriptor))
                {
                    return false;
                }

                continue;
            }

            var candidate = (uint)result.Descriptors.Count;
            result.Descriptors.Add(descriptor);
            keyToCandidate[words[0]] = candidate;
        }

        result.Keys = [.. keyToCandidate.Keys];
        result.Candidates = result.Keys.Select(key => keyToCandidate[key]).ToList();
        return true;
    }

    private static string DiagnoseSnapshotEvaluationFailure(
        ShaderResourcePlan plan,
        ResourceRuntimeInputs inputs,
        IReadOnlyList<bool> activeSources)
    {
        var cleanEvaluator = new RuntimeValueEvaluator(
            plan,
            inputs.WithReader(inputs.ReadCleanMemory));
        var evaluator = new RuntimeValueEvaluator(
            plan,
            inputs,
            plan.CleanFlatSlots,
            cleanEvaluator);

        foreach (var sourceIndex in plan.MaterializationSources)
        {
            if (sourceIndex >= plan.DescriptorSources.Count)
            {
                return $"descriptor source {sourceIndex} is outside the source table";
            }

            if (activeSources.Count != 0 && !activeSources[(int)sourceIndex])
            {
                continue;
            }

            var source = plan.DescriptorSources[(int)sourceIndex];
            for (var dword = 0; dword < source.Dwords.Length; dword++)
            {
                if (!evaluator.Evaluate(source.Dwords[dword], out _))
                {
                    return $"descriptor source {sourceIndex} dword {dword} cannot be evaluated: " +
                        DescribeEvaluationValue(plan, inputs, evaluator, source.Dwords[dword]);
                }
            }
        }

        foreach (var read in plan.TableReads)
        {
            var clean = read.FlatOffset < plan.CleanFlatSlots.Count &&
                plan.CleanFlatSlots[(int)read.FlatOffset] != 0;
            var selected = clean ? cleanEvaluator : evaluator;
            if (read.FlatOffset >= plan.TableReads.Count || !selected.Evaluate(read.Value, out _))
            {
                return $"resource table read {read.FlatOffset} cannot be evaluated: " +
                    DescribeEvaluationValue(plan, inputs, selected, read.Value);
            }
        }

        return "descriptor snapshot evaluation failed without an isolated source";
    }

    private static string DescribeEvaluationValue(
        ShaderResourcePlan plan,
        ResourceRuntimeInputs inputs,
        RuntimeValueEvaluator evaluator,
        ScalarValue value,
        int depth = 0)
    {
        if (depth >= 6)
        {
            return value.ToString();
        }

        if (value.Kind == ScalarValueKind.ResourceTableWord)
        {
            var slot = (int)value.Payload;
            if ((uint)slot < plan.TableReads.Count)
            {
                var read = plan.TableReads[slot];
                return $"table-slot={slot} flat={read.FlatOffset} value=" +
                    DescribeEvaluationValue(plan, inputs, evaluator, read.Value, depth + 1);
            }

            return $"table-slot={slot} (outside table)";
        }

        if (value.Kind is ScalarValueKind.ScalarAddressWord or ScalarValueKind.ScalarBufferWord &&
            value.MemoryIndex >= 0 && value.MemoryIndex < plan.Memory.Count)
        {
            var memory = plan.Memory[value.MemoryIndex];
            var detail = $"{value.Kind} memory={value.MemoryIndex} pc=0x{memory.Pc:X8} " +
                $"opcode={memory.Opcode} offset={memory.Offset} operands=[{string.Join(", ", value.Operands.Select(operand =>
                    DescribeEvaluationValue(plan, inputs, evaluator, operand, depth + 1)))}]";
            if (value.Operands.Length >= 2 && value.Operands[0].Operands.Length >= 2 &&
                evaluator.EvaluateWide(value.Operands[0].Operands[0], out var low) &&
                evaluator.EvaluateWide(value.Operands[0].Operands[1], out var high) &&
                evaluator.EvaluateWide(value.Operands[1], out var dynamicOffset))
            {
                var baseAddress = ((high << 32) | (uint)low) & AddressMask;
                var relative = (long)(int)memory.Offset + (uint)dynamicOffset;
                if (relative >= 0 && baseAddress <= AddressMask - (ulong)relative)
                {
                    var address = (baseAddress + (ulong)relative) & ~3ul;
                    var regular = inputs.ReadMemory is not null && inputs.ReadMemory(address, out _);
                    var clean = inputs.ReadCleanMemory is not null && inputs.ReadCleanMemory(address, out _);
                    detail += $" base=0x{baseAddress:X16} dynamic=0x{dynamicOffset:X} " +
                        $"address=0x{address:X16} readable={regular} clean={clean}";
                }
            }

            return detail;
        }

        if (value.Kind is ScalarValueKind.AddressHandle or ScalarValueKind.BufferHandle or
            ScalarValueKind.ImageHandle or ScalarValueKind.SamplerHandle)
        {
            return $"{value.Kind}[{string.Join(", ", value.Operands.Select(operand =>
                DescribeEvaluationValue(plan, inputs, evaluator, operand, depth + 1)))}]";
        }

        if (value.Kind is ScalarValueKind.Phi or ScalarValueKind.Select or ScalarValueKind.Operation)
        {
            return $"{value} operands=[{string.Join(", ", value.Operands.Select(operand =>
                DescribeEvaluationValue(plan, inputs, evaluator, operand, depth + 1)))}]";
        }

        return value.ToString();
    }

    private static readonly HashSet<uint> NulledSampledFormats = new();

    // Whether a compare access reads a mipmapped image, and so compares in the shader (see BuildSpecialization).
    // SHARPEMU_TRACE_MIPMAPPED_COMPARE=1 names each such program and image once.
    private static readonly bool TracesMipmappedCompare = Environment.GetEnvironmentVariable("SHARPEMU_TRACE_MIPMAPPED_COMPARE") == "1";
    private static readonly HashSet<(ulong, int)> TracedMipmappedCompares = new();

    private static bool MipmappedCompare(ulong hash, int image, ReadOnlySpan<uint> descriptor)
    {
        if (!GuestImageFormat.IsMipmapped(descriptor))
        {
            return false;
        }

        if (TracesMipmappedCompare)
        {
            lock (TracedMipmappedCompares)
            {
                if (TracedMipmappedCompares.Add((hash, image)))
                {
                    Console.Error.WriteLine($"[SHADER][MIPMAPPED-COMPARE] hash=0x{hash:X16} image={image} words={string.Join(",", descriptor.ToArray().Select(word => $"{word:X8}"))}");
                }
            }
        }

        return true;
    }

    private static void ReportNulledSampledFormat(uint format)
    {
        lock (NulledSampledFormats)
        {
            if (NulledSampledFormats.Add(format))
            {
                Console.Error.WriteLine($"[GPU][WARN] A sampled image descriptor uses unsupported format {format}; it is bound as a null texture.");
            }
        }
    }

    private static bool NullImageDescriptor(ReadOnlySpan<uint> descriptor) =>
        descriptor[0] == 0 && (descriptor[1] & 0xFF) == 0;

    // Word 2 bit 31 is RESOURCE_LEVEL, which the hardware wants set: descriptors a title
    // builds itself (Demon's Souls' reflection-probe cubes) carry it, the driver's do not.
    private static bool ReservedImageBitsClear(ReadOnlySpan<uint> descriptor)
    {
        ReadOnlySpan<uint> reserved =
        [
            0x00000000u, 0x20000000u, 0x70003000u, 0x00000000u,
            0xe000e000u, 0xf9000000u, 0x00007b00u, 0x00000000u,
        ];
        for (var dword = 0; dword < reserved.Length; dword++)
        {
            if ((descriptor[dword] & reserved[dword]) != 0)
                return false;
        }
        return true;
    }

    // SHARPEMU_TRACE_NULL_DESCRIPTORS=1: an image whose descriptor holds something but reads as null
    // because the host rejects it (type, format or slice range), once per program and image, with its words.
    private static readonly bool TraceNullDescriptors =
        Environment.GetEnvironmentVariable("SHARPEMU_TRACE_NULL_DESCRIPTORS") == "1";

    private static readonly HashSet<(ulong, int)> TracedInvalidDescriptors = [];
    private static int LoopBoundWarnings;
    private static readonly HashSet<ulong> TracedZeroNeighbourhoods = [];

    private static void TraceInvalidDescriptor(ShaderResourcePlan plan, int imageIndex, uint[] words)
    {
        if (!TraceNullDescriptors || words.All(word => word == 0))
        {
            return;
        }

        lock (TracedInvalidDescriptors)
        {
            if (!TracedInvalidDescriptors.Add((plan.Hash, imageIndex)))
            {
                return;
            }
        }

        Console.Error.WriteLine(
            $"[SHADER][INVALID-DESCRIPTOR] hash=0x{plan.Hash:X16} stage={plan.Stage} image={imageIndex} " +
            $"type={GuestImageFormat.ImageTypeOf(words)} format={GuestImageFormat.FormatOf(words)} words=" +
            string.Join(",", words.Select(word => word.ToString("X8", System.Globalization.CultureInfo.InvariantCulture))));
    }

    // SHARPEMU_TRACE_NULL_DESCRIPTORS=1 as well: an image whose descriptor is all zero, once per program and image,
    // with the guest addresses its descriptor was read from (none when it is user data).
    private static void TraceZeroDescriptor(ShaderResourcePlan plan, int imageIndex, uint source, uint[] words,
        ResourceRuntimeInputs inputs, bool active)
    {
        if (!TraceNullDescriptors || words.Any(word => word != 0))
        {
            return;
        }

        lock (TracedInvalidDescriptors)
        {
            if (!TracedInvalidDescriptors.Add((plan.Hash, imageIndex)))
            {
                return;
            }
        }

        var addresses = new List<ulong>();
        var reader = inputs.ReadMemory;
        bool Recording(ulong address, out uint word)
        {
            addresses.Add(address);
            word = 0;
            return reader is not null && reader(address, out word);
        }

        var again = RuntimeValueEvaluator.EvaluateSources(plan, [source], inputs.WithReader(Recording), [], evaluateTable: false,
            out var reevaluated, out _);
        var clean = inputs.ReadCleanMemory is not null && RuntimeValueEvaluator.EvaluateSources(plan, [source],
            inputs.WithReader(inputs.ReadCleanMemory), [], evaluateTable: false, out var cleanEvaluated, out _)
            ? string.Join(",", cleanEvaluated[0].Dwords.Select(word => word.ToString("X8", System.Globalization.CultureInfo.InvariantCulture)))
            : "unreadable";
        var dwordKinds = string.Join(",", plan.DescriptorSources[(int)source].Dwords.Select(dword => dword.Kind.ToString()).Distinct());
        Console.Error.WriteLine(
            $"[SHADER][ZERO-DESCRIPTOR] hash=0x{plan.Hash:X16} stage={plan.Stage} image={imageIndex} active={active} " +
            $"branches={plan.ResourceBranches.Count} kinds={dwordKinds} again=" +
            (again ? string.Join(",", reevaluated[0].Dwords.Select(word => word.ToString("X8", System.Globalization.CultureInfo.InvariantCulture))) : "failed") +
            $" clean={clean} reads=" +
            (addresses.Count == 0 ? "none" : string.Join(",", addresses.Distinct().Take(12).Select(address => $"0x{address:X}"))));
        if (addresses.Count == 0 || reader is null)
            return;

        // The descriptor's neighbourhood (a table of default textures, say) once per address, and whether the GPU
        // has writes pending over the descriptor and the pointer that named it.
        var descriptorAddress = addresses[^1] & ~0x1Ful;
        lock (TracedZeroNeighbourhoods)
        {
            if (!TracedZeroNeighbourhoods.Add(descriptorAddress))
                return;
        }

        var neighbourhood = new List<string>();
        for (var offset = -0x80L; offset < 0x80; offset += sizeof(uint))
        {
            var address = (ulong)((long)descriptorAddress + offset);
            neighbourhood.Add(reader(address, out var word) ? word.ToString("X8", System.Globalization.CultureInfo.InvariantCulture) : "--------");
        }

        Console.Error.WriteLine(
            $"[SHADER][ZERO-DESCRIPTOR] around=0x{descriptorAddress - 0x80:X} pending_descriptor={inputs.DataWritePending?.Invoke(descriptorAddress)} " +
            $"pending_pointer={inputs.DataWritePending?.Invoke(addresses[0])} words={string.Join(" ", neighbourhood)}");
    }

    private static bool ValidImageDescriptor(ReadOnlySpan<uint> descriptor, bool r128)
    {
        var type = GuestImageFormat.ImageTypeOf(descriptor);
        var format = GuestImageFormat.FormatOf(descriptor);
        if (type < GuestImageFormat.ImageType1D || format == GuestImageFormat.Invalid || format > GuestImageFormat.MaxFormat)
        {
            return false;
        }

        if (r128 && type is not (GuestImageFormat.ImageType1D or GuestImageFormat.ImageType2D or GuestImageFormat.ImageType2DMsaa))
        {
            return false;
        }

        if (type is GuestImageFormat.ImageTypeCube or
            GuestImageFormat.ImageType1DArray or
            GuestImageFormat.ImageType2DArray or
            GuestImageFormat.ImageType2DMsaaArray)
        {
            var lastSlice = descriptor[4] & 0x1FFF;
            var baseSlice = (descriptor[4] >> 16) & 0x1FFF;
            if (baseSlice > lastSlice)
            {
                return false;
            }

        }

        if (type is GuestImageFormat.ImageType2DMsaa or GuestImageFormat.ImageType2DMsaaArray)
        {
            var baseLevel = (descriptor[3] >> 12) & 0xF;
            var fragments = (descriptor[3] >> 16) & 0xF;
            var maxMip = (descriptor[5] >> 4) & 0xF;
            return baseLevel == 0 && fragments is >= 1 and <= 3 && (r128 || maxMip == fragments);
        }

        return true;
    }

    // The instruction samples one kind of image. A descriptor of another kind (the default heap entry, or a key read from some
    // other field of the record) is not something it can sample, so a shared selector reads it as null.
    private static bool SamplesAs(ReadOnlySpan<uint> descriptor, ImageResource image)
    {
        var dimension = DescriptorDimension(descriptor, image.Dimension);
        return dimension == image.Dimension ||
            image.Dimension is ImageDimension.Dim2D or ImageDimension.Dim2DArray &&
            dimension is ImageDimension.Dim2D or ImageDimension.Dim2DArray;
    }

    private static bool CompatibleCandidates(ImageSpecialization image, ImageSpecialization exemplar, bool separateSampledDimensions)
    {
        // Each sampled candidate has its own binding and coordinate width.
        var compatibleDimensions = image.Dimension == exemplar.Dimension ||
            separateSampledDimensions && !image.Cube && !exemplar.Cube &&
            image.Dimension is ImageDimension.Dim2D or ImageDimension.Dim2DArray &&
            exemplar.Dimension is ImageDimension.Dim2D or ImageDimension.Dim2DArray;
        return image.NumericClass == exemplar.NumericClass && compatibleDimensions &&
            image.MipCount == exemplar.MipCount && image.ConversionFormat == exemplar.ConversionFormat &&
            image.ShaderSwizzle == exemplar.ShaderSwizzle && image.Cube == exemplar.Cube;
    }

    private static bool UsableImageCandidate(ReadOnlySpan<uint> candidate, bool r128) =>
        !NullImageDescriptor(candidate) && ValidImageDescriptor(candidate, r128) && ReservedImageBitsClear(candidate) &&
        GuestImageFormat.SampledNumericClass(GuestImageFormat.FormatOf(candidate)) != ImageNumericClass.Unsupported;

    // A table candidate an access selects at run time. A DIM_CUBE access addresses a face,
    // which only a cube has: a table also holding 2D placeholders for unused slots (Demon's
    // Souls' reflection-probe heap) leaves those unreadable through it, so they bind null.
    private static bool UsableTableCandidate(ReadOnlySpan<uint> candidate, ImageResource image) =>
        UsableImageCandidate(candidate, image.R128) &&
        (!image.CubeAccess || GuestImageFormat.ImageTypeOf(candidate) == GuestImageFormat.ImageTypeCube);

    // A call record's image the host can give a float view without a conversion in the shader: a valid sampled T# of a
    // float format. A freed T# (0xFEFEFEFE words) fails the reserved-bit check.
    public static bool IsFloatHeapImage(ReadOnlySpan<uint> words)
    {
        if (words.Length < 8 || !UsableImageCandidate(words, r128: false))
        {
            return false;
        }

        var format = GuestImageFormat.FormatOf(words);
        return GuestImageFormat.SampledNumericClass(format) == ImageNumericClass.Float &&
            ImageConversionFormat(format) == GuestImageFormat.Invalid;
    }

    private static ulong ScalarBufferSize(ReadOnlySpan<uint> descriptor)
    {
        var stride = (descriptor[1] >> 16) & 0x3FFF;
        return stride == 0 ? descriptor[2] : (ulong)stride * descriptor[2];
    }

    private static bool ReadScalarBufferWord(ReadOnlySpan<uint> descriptor, uint dynamicOffset, uint immediateOffset, ResourceRuntimeInputs inputs, out uint word)
    {
        word = 0;
        var byteOffset = (ulong)dynamicOffset + immediateOffset;
        var aligned = byteOffset & ~3ul;
        var size = ScalarBufferSize(descriptor);
        if (aligned > size || size - aligned < sizeof(uint))
        {
            return true;
        }

        var baseAddress = ((descriptor[0] | ((ulong)descriptor[1] << 32)) & AddressMask) & ~3ul;
        if (aligned > AddressMask - baseAddress)
        {
            return false;
        }

        return inputs.ReadCleanMemory is not null && inputs.ReadCleanMemory(baseAddress + aligned, out word);
    }

    // Enumerates every material key that can pass the table's bounds and reads the
    // heap descriptor each selects; stale or invalid descriptors become null.
    private static bool MaterializeIndirectImage(
        ShaderResourcePlan plan,
        IndirectImageSelector indirect,
        DescriptorWords material,
        DescriptorWords heap,
        ImageResource image,
        ResourceRuntimeInputs inputs,
        bool captureSelectorDiagnostic,
        out IndirectImageTable result,
        out ResourceMaterializationFailure failure)
    {
        failure = ResourceMaterializationFailure.Other;
        result = new IndirectImageTable();
        if (material.DwordCount != 4 || heap.DwordCount != 4)
        {
            return false;
        }

        var materialStride = (material.Dwords[1] >> 16) & 0x3FFF;
        if (materialStride != indirect.SelectorStride)
        {
            return false;
        }

        var period = 1ul << 32;
        var step = (ulong)BigInteger.GreatestCommonDivisor(indirect.SelectorStride, period);
        var residue = indirect.SelectorOffset % step;
        var size = ScalarBufferSize(material.Dwords);
        var limit = Math.Min(uint.MaxValue, size + 3);
        var probeCount = residue <= limit ? (limit - residue) / step + 1 : 0;
        uint[]? provenOffsets = null;
        var diagnostic = captureSelectorDiagnostic ? new IndirectSelectorDiagnostic() : null;
        if (indirect.SelectorValues is { } selectorValues)
        {
            var evaluated = selectorValues.TryEvaluate(plan, inputs, out var selectors, diagnostic);
            if (evaluated)
                provenOffsets = selectors.Select(selector => unchecked(selector * indirect.SelectorStride + indirect.SelectorOffset)).Distinct().ToArray();
            if (diagnostic is not null)
            {
                diagnostic.SelectionMode = evaluated ? "bounded" : "full_domain_evaluation_failed";
                diagnostic.SelectorIndices = selectors;
                diagnostic.ProvenOffsets = provenOffsets ?? [];
            }
        }
        var records = (ulong)material.Dwords[2];
        if (provenOffsets is null && (indirect.SharedSelector ? records : probeCount) > MaxIndirectImageProbes)
        {
            return false;
        }

        var keys = new List<uint> { 0 };
        var seen = new HashSet<uint> { 0 };
        // A loop counter walks the table record by record: the keys are the ones its records hold, not every dword
        // the stride lattice reaches. Anything past the last record reads as zero, which is key 0.
        var offsets = provenOffsets is not null ? provenOffsets.Select(offset => (ulong)offset) :
            indirect.SharedSelector ?
                Enumerable.Range(0, (int)records).Select(index => (ulong)unchecked(indirect.SelectorOffset + (uint)index * materialStride)) :
                Enumerable.Range(0, (int)probeCount).Select(index => residue + (ulong)index * step);
        foreach (var offset in offsets)
        {
            if (!ReadScalarBufferWord(material.Dwords, (uint)offset, indirect.MaterialImmediate, inputs, out var key))
            {
                return false;
            }

            if (seen.Add(key))
            {
                keys.Add(key);
                diagnostic?.KeyProbes.Add(new((uint)offset, key));
            }

        }

        var probed = new List<uint[]>(keys.Count);
        foreach (var key in keys)
        {
            var candidate = new uint[8];
            var heapOffset = key << 5;
            for (uint dword = 0; dword < 8; dword++)
            {
                if (!ReadScalarBufferWord(heap.Dwords, heapOffset, dword * sizeof(uint), inputs, out candidate[dword]))
                {
                    return false;
                }
            }

            if (!UsableTableCandidate(candidate, image) || (indirect.SharedSelector && !SamplesAs(candidate, image)))
            {
                Array.Clear(candidate);
            }
            probed.Add(candidate);
        }

        if (!FinishIndirectImage(
                probed,
                keys,
                out var table,
                out failure,
                keepFirstCandidates: indirect.SharedSelector))
        {
            return false;
        }

        table.MaterialDescriptor = material.Dwords;
        table.HeapDescriptor = heap.Dwords;
        table.SelectorStride = indirect.SelectorStride;
        table.SelectorOffset = indirect.SelectorOffset;
        table.SelectorDiagnostic = diagnostic;
        result = table;
        return true;
    }

    private static bool MaterializeDenseIndirectImage(
        IndirectImageSelector indirect,
        DescriptorWords heap,
        ImageResource image,
        bool r128,
        ResourceRuntimeInputs inputs,
        out IndirectImageTable result,
        out ResourceMaterializationFailure failure)
    {
        failure = ResourceMaterializationFailure.Other;
        result = new IndirectImageTable();
        if (heap.DwordCount != 2 || indirect.KeyBound == 0 || indirect.KeyBound > MaxIndirectImageProbes || inputs.ReadCleanMemory is null)
            return false;

        var baseAddress = (((ulong)heap.Dwords[1] << 32) | heap.Dwords[0]) & AddressMask;
        var probed = new List<uint[]>((int)indirect.KeyBound);
        for (uint key = 0; key < indirect.KeyBound; key++)
        {
            var candidate = new uint[8];
            var entry = (ulong)indirect.TableOffset + ((ulong)key << 5);
            for (uint dword = 0; dword < 8; dword++)
            {
                var relative = entry + dword * sizeof(uint);
                if (relative > AddressMask || baseAddress > AddressMask - relative ||
                    !inputs.ReadCleanMemory(baseAddress + relative, out candidate[dword]))
                    return false;
            }

            inputs.TraceImageCandidate?.Invoke(image.Source, key, baseAddress, baseAddress + entry, candidate);
            if (!UsableTableCandidate(candidate, image))
                Array.Clear(candidate);
            probed.Add(candidate);
        }

        return FinishIndirectImage(
            probed,
            Enumerable.Range(0, (int)indirect.KeyBound)
                .Select(index => unchecked(indirect.DynamicOffsetBase + ((uint)index << 5))),
            out result,
            out failure);
    }

    // A V#-described table a sampled access reads through a device table of texture handles:
    // every entry the V# reaches, each distinct usable descriptor once. Range checking is the
    // scalar load's: a dword past the V#'s size reads zero, so a partly covered entry keeps
    // only the dwords inside it, and a key past the last entry reads the null descriptor.
    private static bool MaterializeBindlessImage(
        IndirectImageSelector indirect,
        DescriptorWords heap,
        bool r128,
        ResourceRuntimeInputs inputs,
        out BindlessImageEntries result,
        out ResourceMaterializationFailure failure)
    {
        failure = ResourceMaterializationFailure.Other;
        result = new BindlessImageEntries();
        if (heap.DwordCount != 4 || inputs.ReadCleanMemory is null)
            return false;

        var words = heap.Dwords;
        var baseAddress = (((ulong)(words[1] & 0xFFFFu) << 32) | words[0]) & AddressMask;
        var stride = (words[1] >> 16) & 0x3FFFu;
        var size = stride == 0 ? words[2] : (ulong)stride * words[2];
        var recordStride = indirect.RecordStride;
        var entries = size > indirect.TableOffset ? (size - indirect.TableOffset + recordStride - 1) / recordStride : 0;
        if (indirect.KeyBound != 0)
            entries = Math.Min(entries, indirect.KeyBound);
        if (TraceBindlessTables)
        {
            Console.Error.WriteLine(
                $"[GPU][BINDLESS-TABLE] V#=[{string.Join(",", words.Select(word => word.ToString("X8")))}] size=0x{size:X} " +
                $"tableOffset=0x{indirect.TableOffset:X} recordStride={recordStride} keyBound={indirect.KeyBound} entries={entries}");
        }

        if (baseAddress > AddressMask - indirect.TableOffset)
            return false;
        var tableAddress = baseAddress + indirect.TableOffset;

        result.Address = tableAddress;
        result.DynamicOffsetBase = indirect.DynamicOffsetBase;
        result.RecordStride = recordStride;
        if (entries > MaxBindlessImageEntries)
        {
            // A V# reaching further than any device table (an unbounded one, say) names no
            // entries the host can enumerate: every key reads the null descriptor.
            result.Nulls[$"lie in a table larger than the {MaxBindlessImageEntries} entries a device handle table holds"] =
                (int)Math.Min(entries, int.MaxValue);
            return true;
        }

        // Every dword the scalar load can return: those whose four bytes lie inside the V#.
        var readableWords = entries == 0 ? 0 : (int)Math.Min((size - indirect.TableOffset) / sizeof(uint), int.MaxValue);
        var table = new uint[entries * 8];
        bool[]? unreadable = null;
        if (recordStride != 32)
        {
            unreadable = ReadRecordDescriptors(tableAddress, entries, recordStride, readableWords, table, inputs);
            readableWords = 0;
        }
        readableWords = Math.Min(readableWords, table.Length);
        if (readableWords != 0)
        {
            var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(table.AsSpan(0, readableWords));
            if (inputs.ReadResidentMemory is null || !inputs.ReadResidentMemory(tableAddress, bytes, true))
            {
                // Not resident as one run: the same dwords, one clean read each. An entry with
                // a dword the guest has not mapped, or the GPU has written, reads as null.
                unreadable = new bool[entries];
                for (var word = 0; word < readableWords; word++)
                {
                    if (!unreadable[word >> 3] &&
                        !TryReadCleanWord(tableAddress, (ulong)word * sizeof(uint), inputs, out table[word]))
                        unreadable[word >> 3] = true;
                }
            }
        }

        var memoKey = (tableAddress, entries, r128, indirect.DynamicOffsetBase, recordStride);
        var memoKept = entries >= BindlessMemoMinEntries || recordStride != 32;
        var memoized = unreadable is null && memoKept;
        if (unreadable is not null && memoKept) BindlessMemoUnread++;
        if (memoized)
        {
            lock (BindlessMemoGate)
            {
                var kept = BindlessMemos.TryGetValue(memoKey, out var memo);
                if (!kept) BindlessMemoAbsent++;
                else if (!memo!.Words.AsSpan().SequenceEqual(table)) BindlessMemoChanged++;
                else BindlessMemoHits++;
                if (kept && memo!.Words.AsSpan().SequenceEqual(table))
                {
                    result.Descriptors = memo.Descriptors;
                    result.Entries = memo.Entries;
                    result.Nulls = new Dictionary<string, int>(memo.Nulls);
                    result.Memo = memo;
                    return true;
                }
            }
        }

        var distinct = new Dictionary<ImageDescriptorKey, int>();
        var unreadableEntries = 0;
        result.Entries = new int[entries];
        for (var entry = 0; entry < result.Entries.Length; entry++)
        {
            var candidate = table.AsSpan(entry * 8, 8);
            if (unreadable?[entry] == true)
            {
                unreadableEntries++;
                result.Entries[entry] = BindlessImageTable.NullEntry;
                continue;
            }

            if (!UsableImageCandidate(candidate, r128))
            {
                result.Entries[entry] = BindlessImageTable.NullEntry;
                continue;
            }

            var key = new ImageDescriptorKey(candidate);
            if (!distinct.TryGetValue(key, out var index))
            {
                index = result.Descriptors.Count;
                distinct.Add(key, index);
                result.Descriptors.Add(candidate.ToArray());
            }

            result.Entries[entry] = index;
        }

        if (unreadableEntries != 0)
            result.Nulls["could not be read from guest memory"] = unreadableEntries;
        if (memoized)
        {
            var memo = new BindlessTableMemo
            {
                Words = table,
                Descriptors = result.Descriptors,
                Entries = result.Entries,
                Nulls = new Dictionary<string, int>(result.Nulls),
            };
            lock (BindlessMemoGate)
            {
                if (BindlessMemos.Remove(memoKey, out var replaced))
                    BindlessMemoWords -= replaced.Words.Length;
                while (BindlessMemos.Count != 0 &&
                    (BindlessMemos.Count >= BindlessMemoCapacity || BindlessMemoWords + table.Length > BindlessMemoWordBudget))
                {
                    var oldest = BindlessMemos.Keys.First();
                    BindlessMemoWords -= BindlessMemos[oldest].Words.Length;
                    BindlessMemos.Remove(oldest);
                }

                BindlessMemos[memoKey] = memo;
                BindlessMemoWords += table.Length;
            }

            result.Memo = memo;
        }

        return true;
    }

    // The descriptors of a V# table of records, one at the same offset in each: the records' bytes in one resident
    // read where they can be, one clean read a dword otherwise. A dword past the V# reads as zero, as the scalar
    // load returns it; the entries with a dword the guest has not mapped, or the GPU has written, come back marked.
    private static bool[]? ReadRecordDescriptors(ulong tableAddress, ulong entries, uint recordStride, int readableWords,
        uint[] table, ResourceRuntimeInputs inputs)
    {
        if (entries == 0)
            return null;

        var spanWords = (int)Math.Min(((entries - 1) * recordStride + 32) / sizeof(uint), int.MaxValue);
        var span = new uint[spanWords];
        var spanReadable = Math.Min(readableWords, spanWords);
        bool[]? unreadable = null;
        var resident = spanReadable != 0 && inputs.ReadResidentMemory is not null &&
            inputs.ReadResidentMemory(tableAddress, System.Runtime.InteropServices.MemoryMarshal.AsBytes(span.AsSpan(0, spanReadable)), true);
        for (var entry = 0; entry < (int)entries; entry++)
        {
            var first = (int)((ulong)entry * recordStride / sizeof(uint));
            for (var word = 0; word < 8; word++)
            {
                var at = first + word;
                if (at >= spanReadable)
                    continue;
                if (resident)
                {
                    table[entry * 8 + word] = span[at];
                }
                else if (!TryReadCleanWord(tableAddress, (ulong)at * sizeof(uint), inputs, out table[entry * 8 + word]))
                {
                    unreadable ??= new bool[entries];
                    unreadable[entry] = true;
                    break;
                }
            }
        }

        return unreadable;
    }

    private readonly record struct ImageDescriptorKey(ulong W01, ulong W23, ulong W45, ulong W67)
    {
        public ImageDescriptorKey(ReadOnlySpan<uint> words)
            : this(words[0] | ((ulong)words[1] << 32), words[2] | ((ulong)words[3] << 32),
                words[4] | ((ulong)words[5] << 32), words[6] | ((ulong)words[7] << 32))
        {
        }
    }

    private static bool MaterializeWaveIndexedImage(
        IndirectImageSelector indirect,
        WaveIndexedImageSelector wave,
        DescriptorWords heap,
        ImageResource image,
        bool r128,
        ResourceRuntimeInputs inputs,
        out IndirectImageTable result,
        out ResourceMaterializationFailure failure)
    {
        failure = ResourceMaterializationFailure.Other;
        result = new IndirectImageTable();
        if (heap.DwordCount != 2 || inputs.ReadCleanMemory is null || wave.IndexStride == 0)
            return false;

        var baseAddress = (((ulong)heap.Dwords[1] << 32) | heap.Dwords[0]) & AddressMask;
        if (!TryReadCleanWord(baseAddress, wave.MaskOffset, inputs, out var activeMask))
            return false;

        var keys = new SortedSet<uint>();
        for (uint bit = 0; bit < 32; bit++)
        {
            if ((activeMask & (1u << (int)bit)) == 0)
                continue;
            var indexOffset = (ulong)wave.IndexTableOffset + (ulong)bit * wave.IndexStride;
            if (!TryReadCleanWord(baseAddress, indexOffset, inputs, out var key))
                return false;
            // VCmpxLeI32 0, key suppresses signed-negative indices before ReadFirstLane.
            if ((key & 0x8000_0000u) == 0)
                keys.Add(key);
        }

        var probed = new List<uint[]>(Math.Max(1, keys.Count));
        var offsets = new List<uint>(Math.Max(1, keys.Count));
        foreach (var key in keys)
        {
            var candidate = new uint[8];
            var descriptorOffset = (ulong)indirect.TableOffset + ((ulong)key << 5);
            for (uint dword = 0; dword < candidate.Length; dword++)
            {
                if (!TryReadCleanWord(baseAddress, descriptorOffset + dword * sizeof(uint), inputs, out candidate[dword]))
                    return false;
            }

            if (!UsableTableCandidate(candidate, image))
                Array.Clear(candidate);
            probed.Add(candidate);
            offsets.Add(unchecked(indirect.DynamicOffsetBase + (key << 5)));
        }

        if (probed.Count == 0)
        {
            probed.Add(new uint[8]);
            offsets.Add(0);
        }

        return FinishIndirectImage(probed, offsets, out result, out failure);
    }

    private static bool TryReadCleanWord(ulong baseAddress, ulong offset, ResourceRuntimeInputs inputs, out uint word)
    {
        word = 0;
        return offset <= AddressMask && baseAddress <= AddressMask - offset &&
            inputs.ReadCleanMemory is not null && inputs.ReadCleanMemory(baseAddress + offset, out word);
    }

    private static bool FinishIndirectImage(
        IReadOnlyList<uint[]> probed,
        IEnumerable<uint> keys,
        out IndirectImageTable result,
        out ResourceMaterializationFailure failure,
        bool keepFirstCandidates = false)
    {
        failure = ResourceMaterializationFailure.Other;
        result = new IndirectImageTable();
        using var keyEnumerator = keys.GetEnumerator();
        foreach (var candidate in probed)
        {
            keyEnumerator.MoveNext();
            var words = new DescriptorWords(candidate);
            var found = result.Descriptors.FindIndex(existing => existing.SameAs(words));
            if (found < 0)
            {
                // A table selected by a loop counter may name up to its largest layout (see FitSharedImageTables).
                if (result.Descriptors.Count >= (keepFirstCandidates ? MaxSharedCandidates + 1 : ShaderResourceInfo.MaxIndirectImageCandidates))
                {
                    if (keepFirstCandidates)
                    {
                        // The key is left out of the mapping and reads the first candidate, like any key the table lacks.
                        continue;
                    }

                    var distinct = probed.Select(words => string.Join(',', words)).Distinct().Count();
                    SpecializationFailed($"indirect image table of {probed.Count} entries holds about {distinct} distinct descriptors, " +
                        $"more than the {ShaderResourceInfo.MaxIndirectImageCandidates} candidates an access selects among");
                    failure = ResourceMaterializationFailure.ImageCapacityExceeded;
                    return false;
                }
                found = result.Descriptors.Count;
                result.Descriptors.Add(words);
            }
            result.Keys.Add(keyEnumerator.Current);
            result.Candidates.Add((uint)found);
        }
        return true;
    }

    // A table selected by a loop counter keeps every volume its records name: a volume left out reads the first candidate,
    // and its light goes missing wherever it covers the view. The plan's loop-selected tables share one layout, the most
    // candidates and keys any of them has needed so far, rounded up to powers of two and never shrinking: a view that
    // names another set of volumes does not compile another pipeline. A table that names more than the largest layout
    // keeps the volumes of its first records, and one that names fewer pads with null images; a key whose candidate was
    // dropped is left out of the mapping and reads the table's first candidate, like any key the table lacks.
    private static void FitSharedImageTables(ShaderResourcePlan plan, MaterializedSnapshot snapshot)
    {
        var shared = new List<int>();
        for (var index = 0; index < plan.Info.Images.Count; index++)
        {
            if (plan.DescriptorSources[(int)plan.Info.Images[index].Source].IndirectImage is { SharedSelector: true })
            {
                shared.Add(index);
            }
        }

        if (shared.Count == 0)
        {
            return;
        }

        var neededCandidates = 0;
        var neededKeys = 0;
        foreach (var table in snapshot.IndirectImages.Where(table => table.Shared))
        {
            neededCandidates = Math.Max(neededCandidates, table.Descriptors.Count - 1);
            neededKeys = Math.Max(neededKeys, table.Keys.Count);
        }

        var candidateCapacity = Grow(ref plan.SharedImageCandidates, SizeClass(neededCandidates, MinSharedCandidates, MaxSharedCandidates));
        var keyCapacity = Grow(ref plan.SharedImageKeys, SizeClass(neededKeys, MinSharedKeyCapacity, MaxSharedKeyCapacity));
        foreach (var resource in shared)
        {
            var table = snapshot.IndirectImages.Find(entry => entry.Resource == resource);
            if (table is null)
            {
                continue;
            }

            var extra = candidateCapacity;
            // Whatever reads the first candidate is the same as a key the mapping lacks, so only the others are kept.
            table.KeyCapacity = keyCapacity;
            var keys = new List<uint> { 0 };
            var candidates = new List<uint> { 0 };
            for (var index = 0; index < table.Keys.Count && keys.Count < keyCapacity; index++)
            {
                var candidate = table.Candidates[index];
                if (candidate != 0 && candidate <= extra)
                {
                    keys.Add(table.Keys[index]);
                    candidates.Add(candidate);
                }
            }

            table.Keys = keys;
            table.Candidates = candidates;
            if (table.Descriptors.Count > extra + 1)
            {
                table.Descriptors.RemoveRange(extra + 1, table.Descriptors.Count - extra - 1);
            }

            while (table.Descriptors.Count < extra + 1)
            {
                table.Descriptors.Add(DescriptorWords.Empty(8));
            }
        }
    }

    // The power of two that holds count, between minimum and maximum.
    private static int SizeClass(int count, int minimum, int maximum) =>
        Math.Clamp((int)BitOperations.RoundUpToPowerOf2((uint)Math.Max(count, 1)), minimum, maximum);

    // Raises a plan's layout to size and returns the layout in force, which never shrinks.
    private static int Grow(ref int capacity, int size)
    {
        var current = Volatile.Read(ref capacity);
        while (current < size)
        {
            var seen = Interlocked.CompareExchange(ref capacity, size, current);
            if (seen == current)
            {
                return size;
            }

            current = seen;
        }

        return current;
    }

    // The words a table's key mapping reserves: a loop-selected table reserves its size class, so neither its search
    // depth nor the offsets of the tables after it move with each key.
    private static int MappingCapacity(IndirectImageTable table) =>
        table.Shared ? Math.Max(table.KeyCapacity, table.Keys.Count) : table.Keys.Count;

    // ---- specialization ----

    private static ImageDimension DescriptorDimension(ReadOnlySpan<uint> descriptor, ImageDimension requested)
    {
        var isArray = requested is ImageDimension.Dim1DArray or ImageDimension.Dim2DArray or ImageDimension.Dim2DMsaaArray;
        return GuestImageFormat.ImageTypeOf(descriptor) switch
        {
            GuestImageFormat.ImageType1D => ImageDimension.Dim1D,
            GuestImageFormat.ImageType1DArray => isArray ? ImageDimension.Dim1DArray : ImageDimension.Dim1D,
            GuestImageFormat.ImageType3D => ImageDimension.Dim3D,
            GuestImageFormat.ImageTypeCube => ImageDimension.Dim2DArray,
            GuestImageFormat.ImageType2DArray => isArray ? ImageDimension.Dim2DArray : ImageDimension.Dim2D,
            GuestImageFormat.ImageType2DMsaaArray => isArray ? ImageDimension.Dim2DMsaaArray : ImageDimension.Dim2DMsaa,
            GuestImageFormat.ImageType2D => ImageDimension.Dim2D,
            GuestImageFormat.ImageType2DMsaa => ImageDimension.Dim2DMsaa,
            _ => ImageDimension.Unknown,
        };
    }

    private static uint ImageConversionFormat(uint format) =>
        GuestImageFormat.Remap(format) != format ? format : GuestImageFormat.Invalid;

    // True when DST_SEL reads each of the format's components from itself (X from X, ...).
    private static bool PackedComponentsUnswizzled(uint format, uint dstSelect)
    {
        if (!Gfx10UnifiedFormat.TryDecode(GuestImageFormat.Remap(format), out var dataFormat, out _))
        {
            return true;
        }

        var count = (int)Gfx10UnifiedFormat.ComponentCount(dataFormat);
        for (var component = 0; component < count; component++)
        {
            if (((dstSelect >> (3 * component)) & 7) != 4 + (uint)component)
            {
                return false;
            }
        }

        return true;
    }

    private static bool RequiresPointSampler(ImageNumericClass numericClass, uint conversionFormat) =>
        numericClass == ImageNumericClass.Sint || conversionFormat != GuestImageFormat.Invalid;

    // The depth compare function (sampler word 0, bits 12..14) of the sampler paired
    // with an image; an image sampled with comparison always has one.
    private static int SamplerCompareFunction(ShaderResourceInfo info, MaterializedSnapshot snapshot, uint image)
    {
        foreach (var pair in info.SampledPairs)
        {
            if (pair.Image == image && pair.Sampler < snapshot.Samplers.Length && snapshot.Samplers[pair.Sampler].Length != 0)
            {
                return (int)((snapshot.Samplers[pair.Sampler][0] >> 12) & 0x7);
            }
        }

        return 0;
    }

    // Each plan sampler's specialization, or an empty list when all are the default. The
    // LOD_BIAS is word 2 bits 13:0, signed 6.8. LOD_BIAS_SEC (bits 19:14) is scaled by the
    // T#'s PERF_MOD in a way the ISA does not specify; it is left out, a documented deviation
    // reported once, rather than stopping a title over a bias the ISA cannot pin down.
    private static bool SpecializeSamplers(ShaderResourceInfo info, MaterializedSnapshot snapshot, out List<SamplerSpecialization> samplers)
    {
        samplers = [];
        Span<SamplerSpecialization> specialized = info.Samplers.Count <= 64
            ? stackalloc SamplerSpecialization[info.Samplers.Count]
            : new SamplerSpecialization[info.Samplers.Count];
        var any = false;
        for (var index = 0; index < info.Samplers.Count && index < snapshot.Samplers.Length; index++)
        {
            var words = snapshot.Samplers[index];
            if (words.Length < 4)
            {
                continue;
            }

            var secondary = (words[2] >> 14) & 0x3F;
            if (secondary != 0)
            {
                ShaderCompilerWarnings.WarnOnce(
                    "lod-bias-sec",
                    $"A sampler sets LOD_BIAS_SEC 0x{secondary:X2}, whose PERF_MOD scaling is not modelled; it is ignored.");
            }

            const uint MirrorOnceBorder = 7;
            var axes = (words[0] & 0x7) == MirrorOnceBorder ? 1u : 0u;
            axes |= ((words[0] >> 3) & 0x7) == MirrorOnceBorder ? 2u : 0u;
            axes |= ((words[0] >> 6) & 0x7) == MirrorOnceBorder ? 4u : 0u;
            // A min/max filter mode reduces only a footprint of several texels: a point filter reads one.
            var reduction = (words[0] >> 29) & 0x3;
            var linear = ((words[2] >> 20) & 0x1) != 0 || ((words[2] >> 22) & 0x1) != 0;
            reduction = reduction is 1 or 2 && linear ? reduction : 0;
            specialized[index] = new SamplerSpecialization(
                (int)(words[2] << 18) >> 18, axes, axes != 0 ? words[3] >> 30 : 0, (words[0] & (1u << 20)) != 0,
                reduction, reduction != 0 && ((words[2] >> 26) & 0x3) == 2);
            any |= specialized[index] != default;
        }

        if (any)
        {
            samplers.AddRange(specialized);
        }

        return true;
    }

    private static uint StorageMipCount(ImageResource image, ReadOnlySpan<uint> descriptor)
    {
        if (image.MipMode != ImageMipMode.DynamicStorage || NullImageDescriptor(descriptor))
        {
            return 1;
        }

        var baseLevel = (descriptor[3] >> 12) & 0xF;
        var last = (descriptor[3] >> 16) & 0xF;
        return baseLevel <= last ? last - baseLevel + 1 : 0;
    }

    private static bool Fail(string message)
    {
        SpecializationFailed(message);
        return false;
    }

    private static bool BuildSpecialization(
        ShaderResourcePlan plan,
        MaterializedSnapshot snapshot,
        out MaterializedSnapshot specializedSnapshot,
        out ResourceSpecialization specialization,
        out ResourceMaterializationFailure failure,
        Action<IndirectImageFailure>? captureIndirectImageFailure)
    {
        using var specializationProfile = ResourceMaterializationProfile.Measure(ResourceMaterializationProfile.Phase.Specialization);
        failure = ResourceMaterializationFailure.Other;
        specializedSnapshot = snapshot;
        // Read only on success, which assigns it at the end.
        specialization = null!;
        var info = plan.Info;
        var imageCount = info.Images.Count;
        var mappingWordCount = 0;
        foreach (var table in snapshot.IndirectImages)
        {
            if (table.Resource >= info.Images.Count || table.Descriptors.Count < 2)
            {
                return Fail("indirect image table has an invalid root or candidate count");
            }

            imageCount = checked(imageCount + table.Descriptors.Count - 1);
            mappingWordCount = checked(mappingWordCount + 1 + MappingCapacity(table) * 2);
        }

        foreach (var table in snapshot.BufferCandidateTables)
        {
            mappingWordCount = checked(mappingWordCount + 1 + table.Keys.Count * 2);
        }

        var bindlessTables = new List<BindlessClassification>(snapshot.BindlessImages.Count);
        foreach (var bindless in snapshot.BindlessImages)
        {
            if (bindless.Resource >= info.Images.Count)
            {
                return Fail("bindless image table has an invalid root");
            }

            var classified = ClassifyBindlessImages(plan, snapshot, bindless);
            bindlessTables.Add(classified);
            imageCount = checked(imageCount + classified.Variants.Count - 1);
            mappingWordCount = checked(mappingWordCount + BindlessMappingWords);
        }

        // Each draw owns these arrays. Only indirect candidates require a larger table.
        Array.Resize(ref snapshot.Images, imageCount);
        var mappingCursor = snapshot.FlattenedTable.Length;
        Array.Resize(ref snapshot.FlattenedTable, checked(mappingCursor + mappingWordCount));
        var imageCursor = info.Images.Count;
        var images = new List<ImageSpecialization>(imageCount);
        foreach (var image in info.Images)
        {
            images.Add(new ImageSpecialization(
                image.NumericClass, image.Dimension, image.MipCount, image.ConversionFormat, image.ShaderSwizzle,
                image.IndirectRoot, image.IndirectMappingOffset, image.IndirectSearchIterations, image.Cube));
        }

        foreach (var table in snapshot.IndirectImages)
        {
            var rootImage = images[(int)table.Resource];
            for (var candidate = 1; candidate < table.Descriptors.Count; candidate++)
            {
                images.Add(rootImage with { IndirectRoot = table.Resource });
                snapshot.Images[imageCursor++] = table.Descriptors[candidate].Dwords;
            }

            var mappingOffset = (uint)mappingCursor;
            images[(int)table.Resource] = rootImage with
            {
                IndirectRoot = table.Resource,
                IndirectMappingOffset = mappingOffset,
                IndirectSearchIterations = (uint)BitOperations.Log2((uint)MappingCapacity(table)) + 1,
            };
            mappingCursor += 1 + MappingCapacity(table) * 2;
            var order = Enumerable.Range(0, table.Keys.Count).OrderBy(index => table.Keys[index]).ToArray();
            snapshot.FlattenedTable[(int)mappingOffset] = (uint)table.Keys.Count;
            for (var entry = 0; entry < order.Length; entry++)
            {
                var source = order[entry];
                var offset = (int)mappingOffset + 1 + entry * 2;
                snapshot.FlattenedTable[offset] = table.Keys[source];
                snapshot.FlattenedTable[offset + 1] = table.Candidates[source];
            }

            snapshot.Images[(int)table.Resource] = table.Descriptors[0].Dwords;
        }

        // A bindless table's variants follow the indirect candidates; each binds a null texture
        // of its class, and the flattened table carries the entry count and the key base.
        var bindlessOutputs = new BindlessImageTable[bindlessTables.Count];
        for (var slot = 0; slot < bindlessTables.Count; slot++)
        {
            var classified = bindlessTables[slot];
            var mappingOffset = (uint)mappingCursor;
            mappingCursor += BindlessMappingWords;
            snapshot.FlattenedTable[(int)mappingOffset] = (uint)classified.Entries.Length;
            snapshot.FlattenedTable[(int)mappingOffset + 1] = classified.DynamicOffsetBase;
            snapshot.FlattenedTable[(int)mappingOffset + 2] = classified.RecordStride;
            var variantImages = new uint[classified.Variants.Count];
            for (var variant = 0; variant < classified.Variants.Count; variant++)
            {
                var specialized = classified.Variants[variant] with
                {
                    IndirectRoot = classified.Resource,
                    IndirectMappingOffset = mappingOffset,
                    IndirectSearchIterations = 1,
                    BindlessTable = (uint)slot + 1,
                };
                if (variant == 0)
                {
                    variantImages[0] = classified.Resource;
                    images[(int)classified.Resource] = specialized;
                    snapshot.Images[(int)classified.Resource] = new uint[8];
                    continue;
                }

                variantImages[variant] = (uint)imageCursor;
                images.Add(specialized);
                snapshot.Images[imageCursor++] = new uint[8];
            }

            // The same table again is the same object, so the host keeps its device handles.
            BindlessImageTable? output = null;
            if (classified.Memoized)
            {
                lock (BindlessMemoGate)
                {
                    foreach (var previous in classified.Outputs)
                    {
                        if (previous.Slot == (uint)slot && ReferenceEquals(previous.Descriptors, classified.Descriptors) &&
                            previous.VariantImages.AsSpan().SequenceEqual(variantImages))
                        {
                            output = previous;
                            break;
                        }
                    }
                }
            }

            if (output is null)
            {
                output = new BindlessImageTable
                {
                    Slot = (uint)slot,
                    VariantImages = variantImages,
                    Descriptors = classified.Descriptors,
                    DescriptorVariants = classified.DescriptorVariants,
                    Entries = classified.Entries,
                    Address = classified.Address,
                };
                if (classified.Memoized)
                {
                    lock (BindlessMemoGate)
                    {
                        if (classified.Outputs.Count >= BindlessOutputsKept)
                            classified.Outputs.RemoveAt(0);
                        classified.Outputs.Add(output);
                    }
                }
            }

            bindlessOutputs[slot] = output;
        }

        snapshot.BindlessTables = bindlessOutputs;

        var buffers = new List<BufferSpecialization>(info.Buffers.Count);
        for (var index = 0; index < info.Buffers.Count; index++)
        {
            var descriptor = snapshot.Buffers[index];
            if (descriptor.Length != 4)
            {
                return Fail($"buffer descriptor {index} has invalid width");
            }

            var words = descriptor;
            if ((words[3] >> 30) != 0)
            {
                Array.Clear(words);
            }

            var stride = (words[1] >> 16) & 0x3FFF;
            var swizzleEnabled = (words[1] >> 31) != 0;
            var indexStride = (words[3] >> 21) & 0x3;
            var addThreadId = (words[3] >> 23) & 0x1;
            var packedStride = stride | ((swizzleEnabled ? 1u : 0u) << 14) | (indexStride << 16) | (addThreadId << 20);
            var swizzle = stride != 0 && ((packedStride >> 14) & 1) != 0;
            if (stride == 0)
            {
                packedStride &= ~((1u << 14) | (3u << 16));
            }
            else if (!swizzle)
            {
                packedStride &= ~(3u << 16);
            }

            var formatted = info.Buffers[index].Formatted;
            var specialized = new BufferSpecialization(
                packedStride,
                formatted ? (words[3] >> 12) & 0x7F : DescriptorConstants.InvalidFormat,
                formatted ? words[3] & 0xFFF : DescriptorConstants.IdentityDestinationSelect);
            buffers.Add(PortableBufferWord.Enabled ? PortableBufferWord.Normalize(specialized, info.Buffers[index]) : specialized);
        }

        for (var index = 0; index < images.Count; index++)
        {
            var descriptor = snapshot.Images[index];
            var image = images[index];
            if (image.BindlessSlot != DescriptorConstants.NoIndex)
            {
                // Classified with its table.
                continue;
            }

            var baseIndex = index < info.Images.Count ? (uint)index : image.IndirectRoot;
            if (baseIndex >= info.Images.Count)
            {
                return Fail($"image resource {index} has an invalid root");
            }

            var baseImage = info.Images[(int)baseIndex];
            if (baseImage.ResourceClass == ImageResourceClass.None || (baseImage.Atomic && baseImage.ResourceClass != ImageResourceClass.Storage))
            {
                return Fail($"image resource {index} has an invalid class");
            }

            var mipCount = StorageMipCount(baseImage, descriptor);
            if (mipCount == 0)
            {
                return Fail($"storage image descriptor {index} has an invalid mip range");
            }

            image = image with { MipCount = mipCount };
            if (index < info.Images.Count && plan.DescriptorSources[(int)baseImage.Source].CallRecord is { } callRecord)
            {
                // A call record's image is resolved for each record by the host: the instruction names its class.
                images[index] = image with
                {
                    NumericClass = ImageNumericClass.Float,
                    Dimension = baseImage.Dimension,
                    Cube = callRecord.Cube,
                };
                continue;
            }

            if (NullImageDescriptor(descriptor))
            {
                images[index] = image with
                {
                    // A null-bound atomic keeps its declared numeric class
                    // (Uint or, since float image atomics were added, Float)
                    // instead of assuming every atomic image is Uint.
                    NumericClass = baseImage.Atomic64
                        ? ImageNumericClass.Uint64
                        : baseImage.Atomic ? baseImage.NumericClass : ImageNumericClass.Float,
                    Dimension = ImageDimension.Dim2D,
                    Cube = false,
                };
                continue;
            }

            var dimension = DescriptorDimension(descriptor, baseImage.Dimension);
            if (dimension == ImageDimension.Unknown)
            {
                return Fail(
                    $"image descriptor {index} has unsupported type {GuestImageFormat.ImageTypeOf(descriptor)}: " +
                    string.Join(",", descriptor.Select(word => $"{word:x8}")));
            }

            var format = GuestImageFormat.FormatOf(descriptor);
            // Integer image atomics (Add/Smax/And/...) only ever operate on a
            // 32-bit uint UAV. The float image atomics (Fmin/Fmax/Fcmpswap,
            // MIMG op 0x1D-0x1F) are lowered as a compare-and-swap loop on the
            // raw bit pattern (Gen5SpirvTranslator) and legitimately target a
            // 32-bit float-format UAV instead - reject only formats that are
            // neither.
            // A 64-bit image atomic works on whole 32_32_UINT texels, and only on those.
            var atomicFormatSupported = !baseImage.Atomic || (baseImage.Atomic64
                ? format == GuestImageFormat.Format32x2Uint
                : format == GuestImageFormat.Format32Uint ||
                    GuestImageFormat.SampledNumericClass(format) == ImageNumericClass.Float);
            if (!atomicFormatSupported)
            {
                return Fail($"atomic image descriptor {index} uses unsupported format {format}");
            }

            var storage = baseImage.ResourceClass == ImageResourceClass.Storage;
            var conversionFormat = ImageConversionFormat(format);
            var shaderSwizzle = storage || conversionFormat != GuestImageFormat.Invalid ? descriptor[3] & 0xFFF : image.ShaderSwizzle;
            // A sampled view applies the T# DST_SEL, which would permute a _PCK access's raw
            // components; storage and converted views are unswizzled.
            if (baseImage.Packed && !storage && conversionFormat == GuestImageFormat.Invalid &&
                !PackedComponentsUnswizzled(format, descriptor[3] & 0xFFF))
            {
                return Fail($"packed image descriptor {index} swizzles its format's components (DST_SEL 0x{descriptor[3] & 0xFFF:X3})");
            }

            var rawSintStorage = storage && format == GuestImageFormat.Format32Sint && baseImage.Written && !baseImage.Read && !baseImage.Atomic;
            var numericClass = GuestImageFormat.SampledNumericClass(format);
            if (storage)
            {
                if (numericClass == ImageNumericClass.Unsupported)
                {
                    return Fail($"storage image descriptor {index} uses unsupported format {format}");
                }

                // A write-only 32-bit sint image is stored as raw bits through its uint view;
                // every other sint storage image keeps its signed class (and *_SINT view).
                // Atomics use the uint view; float min/max compare-exchange its bits.
                if (rawSintStorage || (baseImage.Atomic && numericClass == ImageNumericClass.Float))
                {
                    numericClass = ImageNumericClass.Uint;
                }

                if (baseImage.Atomic64)
                {
                    numericClass = ImageNumericClass.Uint64;
                }
            }
            else if (numericClass == ImageNumericClass.Unsupported)
            {
                if (baseImage.Packed)
                {
                    // The nulled view would hand a _PCK access a dummy texture's bits.
                    return Fail($"packed image descriptor {index} uses unsupported format {format}");
                }

                ReportNulledSampledFormat(format);
                Array.Clear(descriptor);
                images[index] = image with
                {
                    NumericClass = ImageNumericClass.Float,
                    Dimension = ImageDimension.Dim2D,
                    Cube = false,
                };
                continue;
            }
            else if (baseImage.DepthCompare && numericClass != ImageNumericClass.Float)
            {
                return Fail($"sampled image descriptor {index} uses unsupported format {format}");
            }

            images[index] = image with
            {
                Dimension = dimension,
                Cube = GuestImageFormat.ImageTypeOf(descriptor) == GuestImageFormat.ImageTypeCube,
                ConversionFormat = conversionFormat,
                ShaderSwizzle = shaderSwizzle,
                NumericClass = numericClass,
                // Only 2D, 2D-array and cube views of a single-level depth-equivalent format get a host
                // depth binding; every other compare access tests texels with the S# function in the shader.
                // A depth target is one level, so a mipmapped surface's levels were written as colour, and
                // a host depth twin of it never holds them (a mipmapped depth surface is not transferred):
                // Ghost of Tsushima's terrain culler compare-gathers its R32F depth pyramid at the level of
                // each triangle's size, and every test past level 0 read empty and culled the triangle.
                EmulatedCompareFunction = !storage && baseImage.DepthCompare &&
                    (!GuestImageFormat.HasDepthEquivalent(format) || MipmappedCompare(plan.Hash, index, descriptor) ||
                     dimension is not (ImageDimension.Dim2D or ImageDimension.Dim2DArray))
                    ? SamplerCompareFunction(info, snapshot, baseIndex)
                    : -1,
                PackedFormat = baseImage.Packed ? format : DescriptorConstants.InvalidFormat,
            };
        }

        for (var rootIndex = 0; rootIndex < images.Count; rootIndex++)
        {
            var root = images[rootIndex];
            if (root.IndirectRoot != rootIndex || root.BindlessSlot != DescriptorConstants.NoIndex)
            {
                continue;
            }

            var keyCount = root.IndirectMappingOffset < snapshot.FlattenedTable.Length ? snapshot.FlattenedTable[(int)root.IndirectMappingOffset] : 0;
            if (root.IndirectSearchIterations == 0 || keyCount < 2 ||
                (ulong)root.IndirectMappingOffset + 1 + (ulong)keyCount * 2 > (ulong)snapshot.FlattenedTable.Length)
            {
                return Fail("indirect image specialization has an invalid key mapping");
            }

            var exemplar = DescriptorConstants.NoIndex;
            var resourceCount = 0;
            for (var resource = 0; resource < images.Count; resource++)
            {
                if (images[resource].IndirectRoot != rootIndex)
                {
                    continue;
                }

                resourceCount++;
                if (exemplar == DescriptorConstants.NoIndex && !NullImageDescriptor(snapshot.Images[resource]))
                {
                    exemplar = (uint)resource;
                }
            }

            if (resourceCount < 2 || exemplar == DescriptorConstants.NoIndex)
            {
                return Fail("indirect image specialization has no typed candidate");
            }

            var imageClass = images[(int)exemplar];
            var separateSampledDimensions = info.Images[rootIndex].ResourceClass == ImageResourceClass.Sampled;
            var sharedSelector = plan.DescriptorSources[(int)info.Images[rootIndex].Source].IndirectImage is { SharedSelector: true };
            for (var candidate = 0; candidate < images.Count; candidate++)
            {
                var image = images[candidate];
                if (image.IndirectRoot != rootIndex)
                {
                    continue;
                }

                // A shared selector read such a table as null before it had one: a candidate of another class reads as null too.
                if (sharedSelector && !NullImageDescriptor(snapshot.Images[candidate]) &&
                    !CompatibleCandidates(image, imageClass, separateSampledDimensions))
                {
                    Array.Clear(snapshot.Images[candidate]);
                }

                if (NullImageDescriptor(snapshot.Images[candidate]))
                {
                    image = image with
                    {
                        NumericClass = imageClass.NumericClass,
                        Dimension = imageClass.Dimension,
                        MipCount = imageClass.MipCount,
                        ConversionFormat = imageClass.ConversionFormat,
                        ShaderSwizzle = imageClass.ShaderSwizzle,
                        Cube = imageClass.Cube,
                        PackedFormat = imageClass.PackedFormat,
                    };
                    images[candidate] = image;
                }

                if (!CompatibleCandidates(image, imageClass, separateSampledDimensions))
                {
                    failure = ResourceMaterializationFailure.IncompatibleImageCandidates;
                    if (captureIndirectImageFailure is not null)
                    {
                        var table = snapshot.IndirectImages.Single(table => table.Resource == rootIndex);
                        captureIndirectImageFailure(new IndirectImageFailure(
                            info.Images[rootIndex].FirstUsePc, (uint)rootIndex, exemplar, (uint)candidate,
                            table.SelectorStride, table.SelectorOffset,
                            [.. table.MaterialDescriptor], [.. table.HeapDescriptor],
                            [.. table.Keys], [.. table.Candidates],
                            table.Descriptors.Select(words => words.Dwords.ToArray()).ToArray(),
                            snapshot.Images.Select(words => words.ToArray()).ToArray(),
                            [.. images], [.. snapshot.UserData])
                        {
                            SelectorDiagnostic = table.SelectorDiagnostic,
                            ScalarReads = plan.TableReads.Select(read =>
                            {
                                var access = plan.Memory[read.Value.MemoryIndex];
                                return new MaterializedScalarRead(access.Pc, access.ComponentIndex, read.FlatOffset,
                                    snapshot.FlattenedTable[(int)read.FlatOffset]);
                            }).ToArray(),
                        });
                    }
                    return Fail($"indirect image table at pc 0x{info.Images[rootIndex].FirstUsePc:x8} has incompatible candidates: exemplar={exemplar} candidate={candidate}");
                }
            }
        }

        if (!BuildSamplerPlan(info, images, out var samplerPlan))
        {
            return Fail("specialized sampler layout exceeds its resource limit");
        }

        Array.Resize(ref snapshot.Samplers, checked((int)samplerPlan.SamplerCount));
        for (var index = 0; index < info.Samplers.Count; index++)
        {
            var target = samplerPlan.PointSampler[index];
            if (target != DescriptorConstants.NoIndex && target >= info.Samplers.Count)
            {
                snapshot.Samplers[target] = snapshot.Samplers[index];
            }
        }

        // ApplyTo appends a depth-compare copy of every sampler shared by ordinary and
        // depth-reference sampling, after the point samplers; the snapshot needs the same words there.
        var compareUsage = new byte[snapshot.Samplers.Length];
        foreach (var pair in info.SampledPairs)
        {
            var image = info.Images[(int)pair.Image];
            var specialized = images[(int)pair.Image];
            var sampler = RequiresPointSampler(specialized.NumericClass, specialized.ConversionFormat)
                ? samplerPlan.PointSampler[pair.Sampler]
                : pair.Sampler;
            var depthCompare = image.DepthCompare && specialized.EmulatedCompareFunction < 0;
            compareUsage[sampler] |= depthCompare ? (byte)2 : (byte)1;
        }

        for (var index = 0; index < snapshot.Samplers.Length && index < compareUsage.Length; index++)
        {
            if (compareUsage[index] == 3)
            {
                Array.Resize(ref snapshot.Samplers, snapshot.Samplers.Length + 1);
                snapshot.Samplers[^1] = snapshot.Samplers[index];
            }
        }


        // Bounded runtime V# candidates are appended after the plan's buffers; each draw
        // owns their words and the run-time key mapping sits in the flattened table.
        var baseBufferCount = buffers.Count;
        var bufferCursor = snapshot.Buffers.Length;
        var candidateTables = new List<BufferCandidateTableSpecialization>(snapshot.BufferCandidateTables.Count);
        foreach (var table in snapshot.BufferCandidateTables)
        {
            if (table.Descriptors.Count == 0 || table.Keys.Count == 0)
            {
                return Fail("buffer candidate table has no candidates");
            }

            Array.Resize(ref snapshot.Buffers, bufferCursor + table.Descriptors.Count);
            for (var candidate = 0; candidate < table.Descriptors.Count; candidate++)
            {
                var words = table.Descriptors[candidate].Dwords;
                if ((words[3] >> 30) != 0)
                {
                    Array.Clear(words);
                }

                snapshot.Buffers[bufferCursor + candidate] = words;
                var stride = (words[1] >> 16) & 0x3FFF;
                var swizzleEnabled = (words[1] >> 31) != 0;
                var indexStride = (words[3] >> 21) & 0x3;
                var addThreadId = (words[3] >> 23) & 0x1;
                var packedStride = stride | ((swizzleEnabled ? 1u : 0u) << 14) | (indexStride << 16) | (addThreadId << 20);
                var swizzle = stride != 0 && ((packedStride >> 14) & 1) != 0;
                if (stride == 0)
                {
                    packedStride &= ~((1u << 14) | (3u << 16));
                }
                else if (!swizzle)
                {
                    packedStride &= ~(3u << 16);
                }

                var specialized = new BufferSpecialization(packedStride, (words[3] >> 12) & 0x7F, words[3] & 0xFFF);
                buffers.Add(PortableBufferWord.Enabled ? PortableBufferWord.Normalize(specialized, null) : specialized);
            }

            var mappingOffset = (uint)mappingCursor;
            var order = Enumerable.Range(0, table.Keys.Count).OrderBy(index => table.Keys[index]).ToArray();
            snapshot.FlattenedTable[(int)mappingOffset] = (uint)table.Keys.Count;
            for (var entry = 0; entry < order.Length; entry++)
            {
                var source = order[entry];
                var offset = (int)mappingOffset + 1 + entry * 2;
                snapshot.FlattenedTable[offset] = table.Keys[source];
                snapshot.FlattenedTable[offset + 1] = table.Candidates[source];
            }

            mappingCursor += 1 + table.Keys.Count * 2;
            candidateTables.Add(new BufferCandidateTableSpecialization(
                (uint)bufferCursor, (uint)table.Descriptors.Count, mappingOffset,
                (uint)BitOperations.Log2((uint)table.Keys.Count) + 1));
            bufferCursor += table.Descriptors.Count;
        }

        if (!SpecializeSamplers(info, snapshot, out var samplers))
        {
            return false;
        }

        specialization = new ResourceSpecialization
        {
            BaseBufferCount = baseBufferCount,
            Buffers = buffers,
            Images = images,
            BufferCandidateTables = candidateTables,
            ImageDescriptorGroups = DescriptorBindingAliases.Group(snapshot.Images, snapshot.Images.Length, images),
            SamplerDescriptorGroups = DescriptorBindingAliases.Group(snapshot.Samplers, Math.Min(info.Samplers.Count, snapshot.Samplers.Length)),
            Samplers = samplers,
        };
        specializedSnapshot = snapshot;
        return true;
    }

    // The flattened words a bindless table takes: its entry count and the key of entry zero.
    // The table's entry count, the key of its first entry and the bytes from one entry to the next.
    private const int BindlessMappingWords = 3;

    // The placements of one memoized table whose output tables are kept for reuse.
    private const int BindlessOutputsKept = 8;

    private sealed class BindlessClassification
    {
        public uint Resource;
        public uint DynamicOffsetBase;
        public uint RecordStride = 32;
        public ulong Address;
        public List<ImageSpecialization> Variants = [];
        public uint[][] Descriptors = [];
        public byte[] DescriptorVariants = [];
        public int[] Entries = [];
        // A memoized classification is shared by every draw that reads the same table words the same way;
        // an output table is reused when a draw places it the same. Programs that share a table (Ghost of
        // Tsushima's material programs, one table a texture) place it differently, so a few are kept.
        public bool Memoized;
        public readonly List<BindlessImageTable> Outputs = [];
    }

    private static readonly HashSet<(ulong Plan, uint Resource, string Reason)> ReportedBindlessNulls = [];

    // A descriptor the access cannot read reads as null, as it would through an unbound
    // descriptor; the first time a program meets each kind it is said so.
    private static void ReportBindlessNulls(ShaderResourcePlan plan, BindlessImageEntries table, string reason, int entries)
    {
        lock (ReportedBindlessNulls)
        {
            if (!ReportedBindlessNulls.Add((plan.Hash, table.Resource, reason)))
                return;
        }

        Console.Error.WriteLine(
            $"[GPU][WARN] Bindless image table at 0x{table.Address:X} (hash=0x{plan.Hash:X16} image={table.Resource} " +
            $"pc=0x{plan.Info.Images[(int)table.Resource].FirstUsePc:X}): {entries} entries {reason}; they read as null.");
    }

    // Sorts a bindless table's descriptors into the classes the access reads them through.
    // The class most entries use is variant 0 and the root; the others are variants a
    // per-entry id selects. A class the access cannot read through (another coordinate
    // shape, or one needing a different sampler) and classes past the variant limit become
    // null entries, said so once per program.
    private static BindlessClassification ClassifyBindlessImages(
        ShaderResourcePlan plan, MaterializedSnapshot snapshot, BindlessImageEntries table)
    {
        var info = plan.Info;
        var root = (int)table.Resource;
        var baseImage = info.Images[root];
        // A depth-compare access also classifies by this draw's sampler; anything else only by the plan and the words.
        var memo = baseImage.DepthCompare ? null : table.Memo;
        var access = (table.Resource, baseImage.Dimension, baseImage.ShaderSwizzle);
        if (memo is not null)
        {
            lock (BindlessMemoGate)
            {
                foreach (var known in memo.Classes)
                {
                    if (known.Access == access)
                    {
                        BindlessMemoClassHits++;
                        return known.Classification;
                    }
                }
            }
        }

        var classification = ClassifyBindlessImagesCore(plan, snapshot, table);
        if (memo is not null)
        {
            classification.Memoized = true;
            lock (BindlessMemoGate)
            {
                if (memo.Classes.Count >= 4)
                    memo.Classes.RemoveAt(0);
                memo.Classes.Add((access, classification));
            }
        }

        return classification;
    }

    private static BindlessClassification ClassifyBindlessImagesCore(
        ShaderResourcePlan plan, MaterializedSnapshot snapshot, BindlessImageEntries table)
    {
        var info = plan.Info;
        var root = (int)table.Resource;
        var baseImage = info.Images[root];
        var accessDimension = baseImage.Dimension == ImageDimension.Unknown ? ImageDimension.Dim2D : baseImage.Dimension;
        var rootSpecialization = new ImageSpecialization(
            ImageNumericClass.Float, ImageDimension.Dim2D, 1, GuestImageFormat.Invalid, baseImage.ShaderSwizzle,
            DescriptorConstants.NoIndex, 0, 0, false);

        var uses = new int[table.Descriptors.Count];
        foreach (var entry in table.Entries)
        {
            if (entry >= 0)
                uses[entry]++;
        }

        var classes = new List<ImageSpecialization>();
        var classOf = new int[table.Descriptors.Count];
        var rejected = new Dictionary<string, int>(table.Nulls);
        for (var index = 0; index < table.Descriptors.Count; index++)
        {
            var descriptor = table.Descriptors[index];
            var dimension = DescriptorDimension(descriptor, accessDimension);
            var format = GuestImageFormat.FormatOf(descriptor);
            var numericClass = GuestImageFormat.SampledNumericClass(format);
            var cube = GuestImageFormat.ImageTypeOf(descriptor) == GuestImageFormat.ImageTypeCube;
            string? reason = null;
            if (dimension == ImageDimension.Unknown)
                reason = $"hold image type {GuestImageFormat.ImageTypeOf(descriptor)}";
            else if (dimension != accessDimension &&
                     (cube || dimension is not (ImageDimension.Dim2D or ImageDimension.Dim2DArray) ||
                      accessDimension is not (ImageDimension.Dim2D or ImageDimension.Dim2DArray)))
                reason = $"are {(cube ? "cube" : dimension.ToString())} images a {accessDimension} access does not address";
            else if (baseImage.DepthCompare && numericClass != ImageNumericClass.Float)
                reason = $"use format {format}, which depth comparison cannot read";

            if (reason is not null)
            {
                classOf[index] = -1;
                rejected[reason] = rejected.GetValueOrDefault(reason) + uses[index];
                continue;
            }

            var conversionFormat = ImageConversionFormat(format);
            var specialization = rootSpecialization with
            {
                NumericClass = numericClass,
                Dimension = dimension,
                ConversionFormat = conversionFormat,
                ShaderSwizzle = conversionFormat != GuestImageFormat.Invalid ? descriptor[3] & 0xFFF : baseImage.ShaderSwizzle,
                Cube = cube,
                EmulatedCompareFunction = baseImage.DepthCompare &&
                    (!GuestImageFormat.HasDepthEquivalent(format) || GuestImageFormat.IsMipmapped(descriptor))
                    ? SamplerCompareFunction(info, snapshot, (uint)root)
                    : -1,
            };
            var found = classes.IndexOf(specialization);
            if (found < 0)
            {
                found = classes.Count;
                classes.Add(specialization);
            }

            classOf[index] = found;
        }

        // Weigh each class by the entries that name it.
        var weights = new int[classes.Count];
        for (var index = 0; index < classOf.Length; index++)
        {
            if (classOf[index] >= 0)
                weights[classOf[index]] += uses[index];
        }

        var order = Enumerable.Range(0, classes.Count).OrderByDescending(index => weights[index]).ThenBy(index => index).ToList();
        var variants = new List<ImageSpecialization>();
        var variantOfClass = new int[classes.Count];
        Array.Fill(variantOfClass, -1);
        if (order.Count == 0)
        {
            // Nothing usable: the access reads the null texture an unbound descriptor binds.
            variants.Add(rootSpecialization);
        }

        foreach (var index in order)
        {
            var candidate = classes[index];
            if (variants.Count != 0)
            {
                var first = variants[0];
                string? reason = null;
                if (RequiresPointSampler(candidate.NumericClass, candidate.ConversionFormat) !=
                    RequiresPointSampler(first.NumericClass, first.ConversionFormat) ||
                    candidate.EmulatedCompareFunction != first.EmulatedCompareFunction)
                    reason = "need a different sampler than the table's most used class";
                else if (variants.Count == MaxBindlessImageVariants)
                    reason = $"are of a class past the {MaxBindlessImageVariants} one access switches over";
                if (reason is not null)
                {
                    rejected[reason] = rejected.GetValueOrDefault(reason) + weights[index];
                    continue;
                }
            }

            variantOfClass[index] = variants.Count;
            variants.Add(candidate);
        }

        foreach (var (reason, count) in rejected)
        {
            if (count != 0)
                ReportBindlessNulls(plan, table, reason, count);
        }

        // Keep the descriptors some variant reads, in their table order.
        var kept = new int[table.Descriptors.Count];
        var descriptors = new List<uint[]>();
        var descriptorVariants = new List<byte>();
        for (var index = 0; index < kept.Length; index++)
        {
            var variant = classOf[index] >= 0 ? variantOfClass[classOf[index]] : -1;
            if (variant < 0)
            {
                kept[index] = BindlessImageTable.NullEntry;
                continue;
            }

            kept[index] = descriptors.Count;
            descriptors.Add(table.Descriptors[index]);
            descriptorVariants.Add((byte)variant);
        }

        var entries = new int[table.Entries.Length];
        for (var entry = 0; entry < entries.Length; entry++)
        {
            var descriptor = table.Entries[entry];
            entries[entry] = descriptor >= 0 ? kept[descriptor] : BindlessImageTable.NullEntry;
        }

        return new BindlessClassification
        {
            Resource = table.Resource,
            DynamicOffsetBase = table.DynamicOffsetBase,
            RecordStride = table.RecordStride,
            Address = table.Address,
            Variants = variants,
            Descriptors = [.. descriptors],
            DescriptorVariants = [.. descriptorVariants],
            Entries = entries,
        };
    }

    private sealed class SamplerPlan
    {
        public uint[] PointSampler = [];
        public uint SamplerCount;
    }

    // A sampler that some pair uses with a point-only image needs a point-filtering
    // copy; when every pair does, the sampler itself switches.
    private static bool BuildSamplerPlan(ShaderResourceInfo info, IReadOnlyList<ImageSpecialization> images, out SamplerPlan plan)
    {
        plan = new SamplerPlan { PointSampler = new uint[info.Samplers.Count] };

        Array.Fill(plan.PointSampler, DescriptorConstants.NoIndex);
        plan.SamplerCount = (uint)info.Samplers.Count;
        var usage = new byte[info.Samplers.Count];
        foreach (var pair in info.SampledPairs)
        {
            if (pair.Image >= images.Count || pair.Sampler >= info.Samplers.Count)
            {
                return false;
            }

            var image = images[(int)pair.Image];
            usage[pair.Sampler] |= RequiresPointSampler(image.NumericClass, image.ConversionFormat) ? (byte)2 : (byte)1;
        }

        for (var index = 0; index < info.Samplers.Count; index++)
        {
            if ((usage[index] & 2) == 0)
            {
                continue;
            }

            if ((usage[index] & 1) == 0)
            {
                plan.PointSampler[index] = (uint)index;
            }
            else
            {
                plan.PointSampler[index] = plan.SamplerCount++;
            }
        }

        return true;
    }

    // Applies a specialization to the plan's tables: buffer strides and formats, image
    // classes and indirect candidates, point samplers and the sampler each access uses.
    public static bool RecordMipStatistics { get; set; }

    /// <summary>
    /// Process-wide: plans evaluate through compiled resource transactions
    /// (CompiledResourceTransaction). The Metal backend turns it off and keeps the
    /// evaluator, whose table reads also learn GPU-written data words (DeviceDataLoads).
    /// </summary>
    public static bool CompiledTransactions { get; set; } = true;

    /// <summary>
    /// Process-wide: a materialisation whose descriptors match the plan's published guard
    /// reuses its specialization instead of building one. The Metal backend turns it off.
    /// </summary>
    public static bool SpecializationGuards { get; set; } = true;

    public static SpecializedResourceInfo ApplyTo(ShaderResourcePlan plan, ResourceSpecialization specialization)
    {
        var source = plan.Info;
        if (source.Buffers.Count != specialization.BaseBufferCount || source.Images.Count > specialization.Images.Count)
        {
            throw new ResourcePlanException(
                $"shader resource specialization does not match the plan: hash=0x{plan.Hash:X16} stage={plan.Stage} " +
                $"buffers={source.Buffers.Count}/{specialization.BaseBufferCount} images={source.Images.Count}/{specialization.Images.Count}");
        }

        var info = source.Clone();
        // A record made before portable buffers, or without them, still names its strides
        // and read formats: what portable buffers read at run time is left out here too.
        var portable = PortableBufferWord.Enabled;
        for (var index = 0; index < info.Buffers.Count; index++)
        {
            var specialized = portable
                ? PortableBufferWord.Normalize(specialization.Buffers[index], source.Buffers[index])
                : specialization.Buffers[index];
            info.Buffers[index].PackedStride = specialized.PackedStride;
            info.Buffers[index].DescriptorFormat = specialized.DescriptorFormat;
            info.Buffers[index].DescriptorSwizzle = specialized.DescriptorSwizzle;
        }

        // Candidate buffers follow the plan's buffers, one native binding each.
        for (var index = source.Buffers.Count; index < specialization.Buffers.Count; index++)
        {
            var specialized = portable ? PortableBufferWord.Normalize(specialization.Buffers[index], null) : specialization.Buffers[index];
            info.Buffers.Add(new BufferResource
            {
                Source = DescriptorConstants.NoIndex,
                Read = true,
                PackedStride = specialized.PackedStride,
                DescriptorFormat = specialized.DescriptorFormat,
                DescriptorSwizzle = specialized.DescriptorSwizzle,
            });
        }

        for (var index = 0; index < specialization.BufferCandidateTables.Count; index++)
        {
            var table = specialization.BufferCandidateTables[index];
            info.BufferCandidateTables.Add(new BufferCandidateTableInfo
            {
                FirstCandidate = table.FirstCandidate,
                CandidateCount = table.CandidateCount,
                MappingOffset = table.MappingOffset,
                SearchIterations = table.SearchIterations,
            });
        }

        for (var index = 0; index < specialization.Images.Count; index++)
        {
            var specialized = specialization.Images[index];
            if (index >= info.Images.Count)
            {
                if (specialized.IndirectRoot >= source.Images.Count)
                {
                    throw new ResourcePlanException($"shader resource specialization names an invalid indirect root: hash=0x{plan.Hash:X16} image={index}");
                }

                info.Images.Add(source.Images[(int)specialized.IndirectRoot].Clone());
            }

            var image = info.Images[index];
            image.NumericClass = specialized.NumericClass;
            image.Dimension = specialized.Dimension;
            image.MipCount = specialized.MipCount;
            image.ConversionFormat = specialized.ConversionFormat;
            image.ShaderSwizzle = specialized.ShaderSwizzle;
            image.IndirectRoot = specialized.IndirectRoot;
            image.IndirectMappingOffset = specialized.IndirectMappingOffset;
            image.IndirectSearchIterations = specialized.IndirectSearchIterations;
            image.Cube = specialized.Cube;
            image.EmulatedCompareFunction = specialized.EmulatedCompareFunction;
            image.PackedFormat = specialized.PackedFormat;
            image.BindlessSlot = specialized.BindlessSlot;
            // Eligibility only, from the plan: whether a sample records is the bound T#'s
            // MIP_STATS_ENABLE, which the shader reads at run time from the image's field
            // (MipStatistics.ImageField), so it is not part of the specialization.
            image.MipStatistics = RecordMipStatistics && plan.Stage == ShaderStage.Pixel &&
                image.ResourceClass != ImageResourceClass.Storage && image.BindlessSlot == DescriptorConstants.NoIndex;
            if (specialized.EmulatedCompareFunction >= 0)
            {
                // The shader compares instead; the host keeps a color view and a plain sampler.
                image.DepthCompare = false;
            }

            image.IndirectResources = [];
        }

        for (var index = 0; index < info.Images.Count; index++)
        {
            var root = info.Images[index].IndirectRoot;
            if (root != DescriptorConstants.NoIndex)
            {
                info.Images[(int)root].IndirectResources.Add((uint)index);
            }
        }

        if (!BuildSamplerPlan(source, specialization.Images, out var samplerPlan))
        {
            throw new ResourcePlanException($"shader resource specialization exceeds the sampler limit: hash=0x{plan.Hash:X16}");
        }

        if (specialization.Samplers.Count != 0)
        {
            if (specialization.Samplers.Count != source.Samplers.Count)
            {
                throw new ResourcePlanException($"shader resource specialization does not match the plan's samplers: hash=0x{plan.Hash:X16}");
            }

            for (var index = 0; index < source.Samplers.Count; index++)
            {
                info.Samplers[index].LodBias = specialization.Samplers[index].LodBias;
                info.Samplers[index].MirrorOnceBorderAxes = specialization.Samplers[index].MirrorOnceBorderAxes;
                info.Samplers[index].BorderColorType = specialization.Samplers[index].BorderColorType;
                info.Samplers[index].ForceDegamma = specialization.Samplers[index].ForceDegamma;
                info.Samplers[index].Reduction = specialization.Samplers[index].Reduction;
                info.Samplers[index].ReductionAcrossLevels = specialization.Samplers[index].ReductionAcrossLevels;
            }
        }

        for (var index = 0; index < source.Samplers.Count; index++)
        {
            var target = samplerPlan.PointSampler[index];
            if (target == DescriptorConstants.NoIndex)
            {
                continue;
            }

            if (target == index)
            {
                info.Samplers[index].ForcePointFiltering = true;
            }
            else
            {
                var sampler = info.Samplers[index].Clone();
                sampler.ForcePointFiltering = true;
                info.Samplers.Add(sampler);
            }
        }

        foreach (var pair in info.SampledPairs)
        {
            var image = info.Images[(int)pair.Image];
            if (RequiresPointSampler(image.NumericClass, image.ConversionFormat))
            {
                pair.Sampler = samplerPlan.PointSampler[pair.Sampler];
            }
        }

        // A guest sampler can be shared by ordinary sampling and depth-reference
        // sampling. Vulkan bakes compareEnable into VkSampler, so those uses cannot
        // share one host sampler. Split only the mixed cases; compare-only samplers
        // can use their existing slot.
        var compareUsage = new byte[info.Samplers.Count];
        foreach (var pair in info.SampledPairs)
        {
            var image = info.Images[(int)pair.Image];
            compareUsage[pair.Sampler] |= image.DepthCompare ? (byte)2 : (byte)1;
        }

        var compareSampler = new uint[info.Samplers.Count];
        Array.Fill(compareSampler, DescriptorConstants.NoIndex);
        for (var index = 0; index < compareUsage.Length; index++)
        {
            if ((compareUsage[index] & 2) == 0)
            {
                continue;
            }

            if ((compareUsage[index] & 1) == 0)
            {
                info.Samplers[index].DepthCompare = true;
                compareSampler[index] = (uint)index;
                continue;
            }

            var sampler = info.Samplers[index].Clone();
            sampler.DepthCompare = true;
            compareSampler[index] = (uint)info.Samplers.Count;
            info.Samplers.Add(sampler);
        }

        foreach (var pair in info.SampledPairs)
        {
            if (info.Images[(int)pair.Image].DepthCompare)
            {
                pair.Sampler = compareSampler[pair.Sampler];
            }
        }

        var samplerByMemory = new Dictionary<int, uint>();
        for (var index = 0; index < plan.Memory.Count; index++)
        {
            var memory = plan.Memory[index];
            if (memory.Kind != MemoryResourceKind.Image || memory.Resource >= info.Images.Count)
            {
                continue;
            }

            var image = info.Images[(int)memory.Resource];
            if (!memory.NeedsSampler || memory.Sampler >= source.Samplers.Count)
            {
                continue;
            }

            var sampler = memory.Sampler;
            if (RequiresPointSampler(image.NumericClass, image.ConversionFormat))
            {
                sampler = samplerPlan.PointSampler[sampler];
            }

            if (image.DepthCompare)
            {
                sampler = compareSampler[sampler];
            }

            if (sampler != memory.Sampler)
            {
                samplerByMemory[index] = sampler;
            }
        }

        DescriptorBindingAliases.Apply(info, source, specialization);
        return new SpecializedResourceInfo { Info = info, SamplerByMemoryIndex = samplerByMemory };
    }
}
