// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;

namespace SharpEmu.ShaderCompiler.Resources;

internal sealed partial class MaterializationProgram
{
    private const ulong AddressMask = 0x0000_FFFF_FFFF_FFFFul;
    private const int MinimumRunWords = 2;
    private const int MaximumRunBytes = 64 * 1024;

    private enum StepKind : byte
    {
        UserData,
        ShaderBase,
        Select,
        Operation,
    }

    private readonly record struct Step(StepKind Kind, int Target, int First, int Second, int Third, ScalarOperation Operation, int[]? Operands);

    private readonly record struct ReadStep(int Target, ScalarValueKind Kind, int Low, int High, int Records, int Word3, int Offset, uint ConstantOffset,
        long Immediate, bool Absolute);

    private sealed class Level
    {
        public ReadStep[] Reads = [];
        public Step[] Steps = [];
        // Set when every read loads a word at a constant offset from one base (FixedReads).
        public FixedReads? Fixed;
        // The address order this level's reads took last (ReadOrder.Sort), kept per program: one per thread and level
        // index was overwritten by every other program the thread evaluated (a draw's pixel and vertex plans in turn),
        // so it rarely fit and the level sorted again. Never written once published; threads may race to replace it.
        public int[]? Order;
    }

    /// <summary>
    /// A level whose reads all load scalar words at constant offsets from the same base registers, as a material
    /// table's T#s and S#s do: their address order and the runs ReadLevel groups them into follow from the offsets
    /// alone, so they are worked out once here rather than sorted and grouped for every draw. A draw takes them only
    /// when its base is non-null and every offset resolves from it; any other draw takes ReadLevel's general walk.
    /// </summary>
    private sealed class FixedReads
    {
        public required int Low;
        public required int High;
        // Read indices in (offset, index) order, with their offsets, and each run's end in that order.
        public required int[] Order;
        public required long[] Offsets;
        public required int[] RunEnds;
        public required int[] RunWords;
        // The reads with the lowest and highest offset, whose resolution bounds every other's.
        public required int First;
        public required int Last;

        private static long OffsetOf(in ReadStep read) => (read.Immediate & ~3L) + (long)(read.ConstantOffset & ~3u);

        public static FixedReads? TryBuild(ReadStep[] reads)
        {
            if (reads.Length == 0) return null;
            foreach (var read in reads)
            {
                if (read.Kind == ScalarValueKind.ScalarBufferWord || read.Offset >= 0 || read.Low != reads[0].Low || read.High != reads[0].High)
                    return null;
            }

            var order = Enumerable.Range(0, reads.Length).OrderBy(index => OffsetOf(reads[index])).ThenBy(index => index).ToArray();
            var offsets = order.Select(index => OffsetOf(reads[index])).ToArray();
            var ends = new List<int>();
            var words = new List<int>();
            var start = 0;
            while (start < order.Length)
            {
                var end = start + 1;
                var count = 1;
                while (end < order.Length && offsets[end] - offsets[end - 1] <= sizeof(uint) && offsets[end] - offsets[start] < MaximumRunBytes)
                {
                    if (offsets[end] != offsets[end - 1]) count++;
                    end++;
                }

                ends.Add(end);
                words.Add(count);
                start = end;
            }

            return new FixedReads
            {
                Low = reads[0].Low, High = reads[0].High, Order = order, Offsets = offsets, RunEnds = [.. ends], RunWords = [.. words],
                First = order[0], Last = order[^1],
            };
        }
    }

    private readonly int _registerCount;
    private readonly ulong[] _constants;
    private readonly bool[] _constantValid;
    private readonly Level[] _levels;
    private readonly int[][] _sourceWords;
    private readonly int[] _tableWords;
    private readonly uint[] _tableOffsets;
    private readonly int[] _rangeLow;
    private readonly int[] _rangeHigh;

    [ThreadStatic]
    private static Scratch? _scratch;

    internal sealed class Scratch
    {
        public ulong[] Values = new ulong[256];
        public bool[] Valid = new bool[256];
        public ulong[] Addresses = new ulong[64];
        public int[] Targets = new int[64];
        public ulong[] NullBases = new ulong[64];
        public bool[] Absolute = new bool[64];
        public byte[] Bytes = new byte[1024];
        public readonly ReadOrder Order = new();
        public int Count;

        // While an evaluation records what it reads (ReadProof): each word read from memory, and each address whose
        // read found none.
        public bool Recording;
        public ulong[] ReadAddresses = new ulong[64];
        public uint[] ReadWords = new uint[64];
        public int ReadCount;
        // Where ReadProof.Build merges the reads' address-ordered runs.
        public ulong[] MergeAddresses = new ulong[64];
        public uint[] MergeWords = new uint[64];
        public readonly List<ulong> Unreadable = [];

        public void NoteRead(ulong address, uint word)
        {
            if (ReadCount == ReadAddresses.Length)
            {
                Array.Resize(ref ReadAddresses, ReadCount * 2);
                Array.Resize(ref ReadWords, ReadCount * 2);
            }

            ReadAddresses[ReadCount] = address;
            ReadWords[ReadCount] = word;
            ReadCount++;
        }

        // Active-source arrays by length: TryRunInto reads its array before the scratch goes back to the thread.
        private readonly bool[]?[] _activeSources = new bool[]?[65];

        public bool[] ActiveSources(int count) => count < _activeSources.Length ? _activeSources[count] ??= new bool[count] : new bool[count];

        public void Ensure(int registers)
        {
            if (Values.Length < registers)
            {
                Values = new ulong[Math.Max(registers, Values.Length * 2)];
                Valid = new bool[Values.Length];
            }
        }

        public void Add(ulong address, int target, ulong baseAddress, bool absolute)
        {
            if (Count == Addresses.Length)
            {
                Array.Resize(ref Addresses, Count * 2);
                Array.Resize(ref Targets, Count * 2);
                Array.Resize(ref NullBases, Count * 2);
                Array.Resize(ref Absolute, Count * 2);
            }

            Addresses[Count] = address;
            Targets[Count] = target;
            NullBases[Count] = baseAddress;
            Absolute[Count] = absolute;
            Count++;
        }
    }

    private MaterializationProgram(int registerCount, ulong[] constants, bool[] constantValid, Level[] levels, int[][] sourceWords, int[] tableWords,
        uint[] tableOffsets, int[] rangeLow, int[] rangeHigh)
    {
        _registerCount = registerCount;
        _constants = constants;
        _constantValid = constantValid;
        _levels = levels;
        foreach (var level in levels)
        {
            level.Fixed = FixedReads.TryBuild(level.Reads);
        }

        _sourceWords = sourceWords;
        _tableWords = tableWords;
        _tableOffsets = tableOffsets;
        _rangeLow = rangeLow;
        _rangeHigh = rangeHigh;
    }

    internal static MaterializationProgram? Build(ShaderResourcePlan plan, out string reason)
    {
        // DeviceAddressRangePlanner.Evaluate appends ranges for stores through V#s a program loads
        // itself (function buffer accesses, device descriptor tables); the program evaluates only
        // the planned handle ranges, so such plans take the full materialization.
        reason = plan.IndirectImages.Count != 0 ? "indirect_images" : plan.BufferCandidateTables.Count != 0 ? "candidate_tables" :
            plan.DeviceDescriptorTables.Count != 0 ||
            plan.Graph.Program.FunctionBufferAccesses.Count != 0 ? "device_descriptor_tables" : "";
        if (reason.Length != 0)
        {
            return null;
        }

        foreach (var clean in plan.CleanFlatSlots)
        {
            if (clean != 0)
            {
                reason = "clean_slots";
                return null;
            }
        }

        foreach (var source in plan.DescriptorSources)
        {
            if (source.IndirectImage is not null)
            {
                reason = "indirect_source";
                return null;
            }
        }

        var builder = new Builder(plan);
        var program = Build(plan, builder);
        reason = program is null ? builder.Reason ?? "layout" : "";
        return program;
    }

    private static MaterializationProgram? Build(ShaderResourcePlan plan, Builder builder)
    {
        var sourceWords = new int[plan.MaterializationSources.Count][];
        for (var index = 0; index < sourceWords.Length; index++)
        {
            var source = plan.MaterializationSources[index];
            if (source >= plan.DescriptorSources.Count)
            {
                return null;
            }

            var dwords = plan.DescriptorSources[(int)source].Dwords;
            sourceWords[index] = new int[dwords.Length];
            for (var word = 0; word < dwords.Length; word++)
            {
                if (!builder.Register(dwords[word], out sourceWords[index][word]))
                {
                    return null;
                }
            }
        }

        var tableWords = new int[plan.TableReads.Count];
        var tableOffsets = new uint[plan.TableReads.Count];
        for (var index = 0; index < tableWords.Length; index++)
        {
            var read = plan.TableReads[index];
            if (read.FlatOffset >= plan.TableReads.Count || !builder.Register(read.Value, out tableWords[index]))
            {
                return null;
            }

            tableOffsets[index] = read.FlatOffset;
        }

        var rangeLow = new int[plan.DeviceAddressRanges.Count];
        var rangeHigh = new int[plan.DeviceAddressRanges.Count];
        for (var index = 0; index < rangeLow.Length; index++)
        {
            var range = plan.DeviceAddressRanges[index];
            if (!range.Plannable)
            {
                rangeLow[index] = -1;
                rangeHigh[index] = -1;
                continue;
            }

            if (!builder.Register(range.BaseLow, out rangeLow[index]) || !builder.Register(range.BaseHigh, out rangeHigh[index]))
            {
                return null;
            }
        }

        return builder.Finish(sourceWords, tableWords, tableOffsets, rangeLow, rangeHigh);
    }

    private sealed class Builder(ShaderResourcePlan plan)
    {
        private readonly Dictionary<ScalarValue, int> _registers = new(ReferenceEqualityComparer.Instance);
        private readonly HashSet<ScalarValue> _visiting = new(ReferenceEqualityComparer.Instance);
        private readonly List<ulong> _constants = [];
        private readonly List<bool> _constantValid = [];
        private readonly List<int> _registerLevels = [];
        private readonly List<(int Level, Step Step)> _steps = [];
        private readonly List<(int Level, ReadStep Read)> _reads = [];

        public string? Reason;

        private bool Refuse(string reason)
        {
            Reason ??= reason;
            return false;
        }

        private int NewRegister(int level, ulong constant = 0, bool constantValid = false)
        {
            _constants.Add(constant);
            _constantValid.Add(constantValid);
            _registerLevels.Add(level);
            return _constants.Count - 1;
        }

        private int LevelOf(int register) => _registerLevels[register];

        // Every register as a node a branch selection evaluates on demand: its constant, step or read.
        public BranchNode[] Nodes()
        {
            var nodes = new BranchNode[_constants.Count];
            for (var register = 0; register < nodes.Length; register++)
            {
                nodes[register] = new BranchNode(BranchNodeKind.Constant, _constants[register], default, default);
            }

            foreach (var (_, step) in _steps)
            {
                nodes[step.Target] = new BranchNode(BranchNodeKind.Step, 0, step, default);
            }

            foreach (var (_, read) in _reads)
            {
                nodes[read.Target] = new BranchNode(BranchNodeKind.Read, 0, default, read);
            }

            return nodes;
        }

        public bool Register(ScalarValue value, out int register)
        {
            if (_registers.TryGetValue(value, out register))
            {
                return true;
            }

            if (!_visiting.Add(value))
            {
                return Refuse("cycle");
            }

            try
            {
                if (!Lower(value, out register))
                {
                    return false;
                }
            }
            finally
            {
                _visiting.Remove(value);
            }

            _registers[value] = register;
            return true;
        }

        private bool Lower(ScalarValue value, out int register)
        {
            register = -1;
            switch (value.Kind)
            {
                case ScalarValueKind.Constant:
                    register = NewRegister(0, value.Payload, constantValid: true);
                    return true;
                case ScalarValueKind.MemoryAperture:
                    register = NewRegister(0, Gen5InlineConstants.DecodeAperture64((uint)value.Payload) >> 32, constantValid: true);
                    return true;
                case ScalarValueKind.UserData:
                    if (value.UserDataRegister < plan.UserDataBase)
                    {
                        return Refuse("user_data_base");
                    }

                    register = NewRegister(0);
                    _steps.Add((0, new Step(StepKind.UserData, register, (int)(value.UserDataRegister - plan.UserDataBase), 0, 0, default, null)));
                    return true;
                case ScalarValueKind.ShaderBase:
                    register = NewRegister(0);
                    _steps.Add((0, new Step(StepKind.ShaderBase, register, 0, 0, 0, default, null)));
                    return true;
                case ScalarValueKind.Phi:
                    return plan.Graph.ResolveInvariantPhi(value) is { } invariant ? Register(invariant, out register) : Refuse("phi");
                case ScalarValueKind.ResourceTableWord:
                    return value.Payload < (ulong)plan.TableReads.Count && Register(plan.TableReads[(int)value.Payload].Value, out register);
                case ScalarValueKind.Select:
                {
                    if (value.Operands.Length != 3 || !Register(value.Operands[0], out var condition) || !Register(value.Operands[1], out var whenTrue) ||
                        !Register(value.Operands[2], out var whenFalse))
                    {
                        return false;
                    }

                    var level = Math.Max(LevelOf(condition), Math.Max(LevelOf(whenTrue), LevelOf(whenFalse)));
                    register = NewRegister(level);
                    _steps.Add((level, new Step(StepKind.Select, register, condition, whenTrue, whenFalse, default, null)));
                    return true;
                }
                case ScalarValueKind.Operation:
                {
                    if (!RuntimeValueValidator.IsUniformOperation(value.Operation))
                    {
                        return Refuse("operation_" + value.Operation);
                    }

                    var operands = new int[value.Operands.Length];
                    var level = 0;
                    for (var index = 0; index < operands.Length; index++)
                    {
                        if (!Register(value.Operands[index], out operands[index]))
                        {
                            return false;
                        }

                        level = Math.Max(level, LevelOf(operands[index]));
                    }

                    register = NewRegister(level);
                    _steps.Add((level, new Step(StepKind.Operation, register, 0, 0, 0, value.Operation, operands)));
                    return true;
                }
                case ScalarValueKind.ScalarAddressWord:
                case ScalarValueKind.ScalarBufferWord:
                {
                    var buffer = value.Kind == ScalarValueKind.ScalarBufferWord;
                    if ((uint)value.MemoryIndex >= (uint)plan.Memory.Count || value.Operands.Length < 2)
                    {
                        return Refuse("memory_index");
                    }

                    var handle = value.Operands[0];
                    if (handle.Operands.Length < 2 || (buffer && handle.Operands.Length != 4))
                    {
                        return Refuse("memory_handle");
                    }

                    var records = -1;
                    var word3 = -1;
                    if (!Register(handle.Operands[0], out var low) || !Register(handle.Operands[1], out var high) ||
                        (buffer && (!Register(handle.Operands[2], out records) || !Register(handle.Operands[3], out word3))))
                    {
                        return false;
                    }

                    var offsetValue = value.Operands[1];
                    var offset = -1;
                    if (!offsetValue.IsConstant && !Register(offsetValue, out offset))
                    {
                        return false;
                    }

                    var level = Math.Max(LevelOf(low), LevelOf(high));
                    if (buffer)
                    {
                        level = Math.Max(level, Math.Max(LevelOf(records), LevelOf(word3)));
                    }

                    if (offset >= 0)
                    {
                        level = Math.Max(level, LevelOf(offset));
                    }

                    level++;
                    register = NewRegister(level);
                    _reads.Add((level, new ReadStep(register, value.Kind, low, high, records, word3, offset, offsetValue.IsConstant ? (uint)offsetValue.Payload : 0,
                        (int)plan.Memory[value.MemoryIndex].Offset, RuntimeValueEvaluator.IsAbsoluteAddress(value.Kind, handle))));
                    return true;
                }
                default:
                    return Refuse(value.Kind.ToString());
            }
        }

        public MaterializationProgram Finish(int[][] sourceWords, int[] tableWords, uint[] tableOffsets, int[] rangeLow, int[] rangeHigh)
        {
            var levelCount = 1;
            foreach (var level in _registerLevels)
            {
                levelCount = Math.Max(levelCount, level + 1);
            }

            var reads = new List<ReadStep>[levelCount];
            var steps = new List<Step>[levelCount];
            for (var level = 0; level < levelCount; level++)
            {
                reads[level] = [];
                steps[level] = [];
            }

            foreach (var (level, read) in _reads)
            {
                reads[level].Add(read);
            }

            foreach (var (level, step) in _steps)
            {
                steps[level].Add(step);
            }

            var levels = new Level[levelCount];
            for (var level = 0; level < levelCount; level++)
            {
                ReadStep[] levelReads = [.. reads[level]];
                if (StaticReadOrder)
                {
                    Array.Sort(levelReads, CompareReadOrder);
                }

                levels[level] = new Level { Reads = levelReads, Steps = [.. steps[level]] };
            }

            return new MaterializationProgram(_constants.Count, [.. _constants], [.. _constantValid], levels, sourceWords, tableWords, tableOffsets,
                rangeLow, rangeHigh);
        }
    }

    internal bool TryRun(ShaderResourcePlan plan, ResourceRuntimeInputs inputs, int additionalTableWords, out List<DescriptorWords> sources,
        out uint[] table, out DeviceAddressRange[] ranges, out bool[] activeSources)
    {
        activeSources = plan.ResourceBranches.Count != 0 ? SelectActiveSources(plan, inputs, reuse: null) : [];
        var scratch = _scratch ?? new Scratch();
        _scratch = null;
        try
        {
            return Run(plan, inputs, additionalTableWords, scratch, activeSources, out sources, out table, out ranges);
        }
        finally
        {
            _scratch = scratch;
        }
    }

    // The words of one evaluation, held by the thread that runs it and reused by the next.
    internal sealed class Output
    {
        public uint[] Words = new uint[256];
        public int[] Offsets = new int[33];
        public uint[] Table = new uint[64];
        public int TableLength;
        public DeviceAddressRange[] Ranges = new DeviceAddressRange[8];
        public int RangeCount;

        public Span<uint> Source(int index) => Words.AsSpan(Offsets[index], Offsets[index + 1] - Offsets[index]);

        public Span<uint> TableWords => Table.AsSpan(0, TableLength);

        public ReadOnlySpan<DeviceAddressRange> RangeValues => Ranges.AsSpan(0, RangeCount);
    }

    internal bool TryRunInto(ShaderResourcePlan plan, ResourceRuntimeInputs inputs, int additionalTableWords, Output output) =>
        TryRunInto(plan, inputs, additionalTableWords, output, record: false, previous: null, out _);

    // With record, also what the evaluation read (null when one address read as two values): previous itself when it
    // holds exactly that, or a proof sharing its ranges when only their bytes differ.
    internal bool TryRunInto(ShaderResourcePlan plan, ResourceRuntimeInputs inputs, int additionalTableWords, Output output, bool record,
        ReadProof? previous, out ReadProof? proof)
    {
        proof = null;
        var scratch = _scratch ?? new Scratch();
        _scratch = null;
        try
        {
            // A recorded proof keeps its active-source array (ReadProof.ActiveSources), so it must not be the scratch's
            // reused one. Selected before recording starts: the proof covers the evaluation's reads only.
            var activeSources = plan.ResourceBranches.Count != 0 ? SelectActiveSources(plan, inputs, reuse: record ? null : scratch) : [];
            scratch.Recording = record;
            scratch.ReadCount = 0;
            scratch.Unreadable.Clear();
            Evaluate(inputs, scratch);
            scratch.Recording = false;
            if (!Emit(plan, additionalTableWords, scratch, activeSources, output))
            {
                return false;
            }

            if (record)
            {
                proof = ReadProof.Build(this, scratch, activeSources, previous);
            }

            return true;
        }
        finally
        {
            scratch.Recording = false;
            _scratch = scratch;
        }
    }

    /// <summary>
    /// What one evaluation of a program read, as the proof that another evaluation with the same user data, shader base,
    /// compute state and readers emits the same words. Every read's address is a function of the registers of the levels
    /// before it, and those of the first read level of the user data, shader base and constants alone; so while those
    /// are the same, every word read from memory still holds the value it was read as, and every read that found no
    /// memory finds none again, every read is at the same address with the same outcome, level by level, and every
    /// register comes out the same. The words emitted are a function of the registers and the active sources
    /// (<see cref="ActiveSources"/>, which the caller evaluates again). A caller compares the words
    /// (<see cref="Addresses"/>, <see cref="Lengths"/>, <see cref="Bytes"/>) through the ownership tests a word read
    /// makes, recorded as that read records them, and reads <see cref="Unreadable"/> again. Immutable once built.
    /// </summary>
    internal sealed class ReadProof
    {
        // Words of one range lie in one 256-byte line, the unit the concurrent ownership tests answer for.
        internal const int LineBytes = 256;

        public required MaterializationProgram Program;
        // The words read from memory, in address order, as runs of consecutive words within a line, and their bytes, run
        // after run.
        public required ulong[] Addresses;
        public required int[] Lengths;
        public required byte[] Bytes;
        // Addresses whose read found no memory.
        public required ulong[] Unreadable;
        // The sources the plan's branches left active (empty without branches).
        public required bool[] ActiveSources;

        internal static ReadProof? Build(MaterializationProgram program, Scratch scratch, bool[] activeSources, ReadProof? previous = null)
        {
            var count = scratch.ReadCount;
            var addresses = scratch.ReadAddresses;
            var words = scratch.ReadWords;
            SortReads(scratch, count);
            var unique = 0;
            var ranges = 0;
            for (var index = 0; index < count; index++)
            {
                if (unique != 0 && addresses[unique - 1] == addresses[index])
                {
                    // One address read as two values in the same evaluation (the guest wrote it meanwhile): no proof.
                    if (words[unique - 1] != words[index])
                    {
                        return null;
                    }

                    continue;
                }

                if (unique == 0 || !Continues(addresses[unique - 1], addresses[index]))
                {
                    ranges++;
                }

                addresses[unique] = addresses[index];
                words[unique] = words[index];
                unique++;
            }

            // A revalidation that read in full mostly reads what the proof before it did (its compare was refused) or the
            // same words with other values (a words miss): kept, or its ranges shared, rather than allocated again.
            var sameRanges = previous is not null && ReferenceEquals(previous.Program, program) && previous.Addresses.Length == ranges &&
                previous.Bytes.Length == unique * sizeof(uint) && SameRanges(previous, addresses, unique);
            var sameBytes = sameRanges && SameWords(previous!.Bytes, words, unique);
            var sameRest = previous is not null && previous.ActiveSources.AsSpan().SequenceEqual(activeSources) &&
                previous.Unreadable.AsSpan().SequenceEqual(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(scratch.Unreadable));
            if (sameBytes && sameRest)
            {
                return previous;
            }

            ulong[] starts;
            int[] lengths;
            if (sameRanges)
            {
                starts = previous!.Addresses;
                lengths = previous.Lengths;
            }
            else
            {
                starts = new ulong[ranges];
                lengths = new int[ranges];
                var range = -1;
                for (var index = 0; index < unique; index++)
                {
                    if (index == 0 || !Continues(addresses[index - 1], addresses[index]))
                    {
                        starts[++range] = addresses[index];
                    }

                    lengths[range] += sizeof(uint);
                }
            }

            byte[] bytes;
            if (sameBytes)
            {
                bytes = previous!.Bytes;
            }
            else
            {
                bytes = new byte[unique * sizeof(uint)];
                System.Runtime.InteropServices.MemoryMarshal.AsBytes(words.AsSpan(0, unique)).CopyTo(bytes);
                if (!BitConverter.IsLittleEndian)
                {
                    for (var index = 0; index < unique; index++)
                    {
                        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(index * sizeof(uint)), words[index]);
                    }
                }
            }

            return new ReadProof
            {
                Program = program, Addresses = starts, Lengths = lengths, Bytes = bytes,
                Unreadable = scratch.Unreadable.Count == 0 ? [] : sameRest ? previous!.Unreadable : [.. scratch.Unreadable],
                ActiveSources = activeSources,
            };
        }

        // Most runs of recorded reads to merge rather than sort: each level records its reads in address order.
        private const int MaxMergedRuns = 16;

        // The recorded reads in address order. A level's reads are recorded in address order (ReadLevel sorts them, a
        // fixed plan reads its runs in offset order), so the reads are a few ascending runs, merged here one into the
        // ordered prefix before it; more runs than that are sorted. Equal addresses may end in any order: Build refuses
        // them when their words differ and keeps one when they are the same.
        private static void SortReads(Scratch scratch, int count)
        {
            var addresses = scratch.ReadAddresses;
            var words = scratch.ReadWords;
            var runs = 1;
            for (var index = 1; index < count; index++)
            {
                if (addresses[index] < addresses[index - 1] && ++runs > MaxMergedRuns)
                {
                    Array.Sort(addresses, words, 0, count);
                    return;
                }
            }

            if (runs == 1)
            {
                return;
            }

            if (scratch.MergeAddresses.Length < count)
            {
                scratch.MergeAddresses = new ulong[Math.Max(count, scratch.MergeAddresses.Length * 2)];
                scratch.MergeWords = new uint[scratch.MergeAddresses.Length];
            }

            var mergedAddresses = scratch.MergeAddresses;
            var mergedWords = scratch.MergeWords;
            var ordered = 1;
            while (ordered < count && addresses[ordered] >= addresses[ordered - 1]) ordered++;
            while (ordered < count)
            {
                var end = ordered + 1;
                while (end < count && addresses[end] >= addresses[end - 1]) end++;
                // Merge [0, ordered) and [ordered, end) into the merge buffers, then back.
                int left = 0, right = ordered, at = 0;
                while (left < ordered && right < end)
                {
                    if (addresses[right] < addresses[left])
                    {
                        mergedAddresses[at] = addresses[right];
                        mergedWords[at++] = words[right++];
                    }
                    else
                    {
                        mergedAddresses[at] = addresses[left];
                        mergedWords[at++] = words[left++];
                    }
                }

                while (left < ordered)
                {
                    mergedAddresses[at] = addresses[left];
                    mergedWords[at++] = words[left++];
                }

                // The rest of the right run is already in place after the merged part.
                Array.Copy(mergedAddresses, addresses, at);
                Array.Copy(mergedWords, words, at);
                ordered = end;
            }
        }

        // Whether the unique sorted addresses form exactly the proof's ranges.
        private static bool SameRanges(ReadProof proof, ulong[] addresses, int unique)
        {
            var index = 0;
            for (var range = 0; range < proof.Addresses.Length; range++)
            {
                var start = proof.Addresses[range];
                var words = proof.Lengths[range] / sizeof(uint);
                if (index + words > unique || addresses[index] != start || (index != 0 && Continues(addresses[index - 1], start)))
                {
                    return false;
                }

                for (var word = 1; word < words; word++)
                {
                    if (addresses[index + word] != start + (ulong)(word * sizeof(uint)))
                    {
                        return false;
                    }
                }

                index += words;
            }

            return index == unique;
        }

        private static bool SameWords(byte[] bytes, uint[] words, int unique)
        {
            for (var index = 0; index < unique; index++)
            {
                if (System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(index * sizeof(uint))) != words[index])
                {
                    return false;
                }
            }

            return true;
        }

        // Whether a word at next extends the run whose last word is at previous: the next word, in the same line.
        private static bool Continues(ulong previous, ulong next) =>
            next - previous == sizeof(uint) && (next & ~(ulong)(LineBytes - 1)) == (previous & ~(ulong)(LineBytes - 1));
    }

    private void Evaluate(ResourceRuntimeInputs inputs, Scratch scratch)
    {
        scratch.Ensure(_registerCount);
        var values = scratch.Values;
        var valid = scratch.Valid;
        Array.Copy(_constants, values, _registerCount);
        Array.Copy(_constantValid, valid, _registerCount);
        Span<ulong> operands = stackalloc ulong[8];
        for (var levelIndex = 0; levelIndex < _levels.Length; levelIndex++)
        {
            var level = _levels[levelIndex];
            if (level.Reads.Length != 0 && (level.Fixed is not { } fixedReads || !TryReadFixed(level.Reads, fixedReads, inputs, scratch)))
            {
                ReadLevel(level, inputs, scratch);
            }

            foreach (var step in level.Steps)
            {
                Execute(step, inputs, values, valid, operands);
            }
        }
    }

    private bool Emit(ShaderResourcePlan plan, int additionalTableWords, Scratch scratch, bool[] activeSources, Output output)
    {
        var values = scratch.Values;
        var valid = scratch.Valid;
        var total = 0;
        foreach (var registers in _sourceWords)
        {
            total += registers.Length;
        }

        if (output.Words.Length < total)
        {
            output.Words = new uint[Math.Max(total, output.Words.Length * 2)];
        }

        if (output.Offsets.Length < _sourceWords.Length + 1)
        {
            output.Offsets = new int[_sourceWords.Length + 1];
        }

        var cursor = 0;
        for (var source = 0; source < _sourceWords.Length; source++)
        {
            var registers = _sourceWords[source];
            output.Offsets[source] = cursor;
            var inactive = activeSources.Length != 0 && !activeSources[plan.MaterializationSources[source]];
            for (var word = 0; word < registers.Length; word++)
            {
                var register = registers[word];
                if (inactive)
                {
                    output.Words[cursor++] = 0;
                    continue;
                }

                if (!valid[register])
                {
                    return false;
                }

                output.Words[cursor++] = (uint)values[register];
            }
        }

        output.Offsets[_sourceWords.Length] = cursor;
        var tableLength = checked(_tableWords.Length + additionalTableWords);
        if (output.Table.Length < tableLength)
        {
            output.Table = new uint[Math.Max(tableLength, output.Table.Length * 2)];
        }

        Array.Clear(output.Table, 0, tableLength);
        output.TableLength = tableLength;
        for (var index = 0; index < _tableWords.Length; index++)
        {
            var register = _tableWords[index];
            if (!valid[register])
            {
                return false;
            }

            output.Table[_tableOffsets[index]] = (uint)values[register];
        }

        var planned = plan.DeviceAddressRanges;
        if (output.Ranges.Length < planned.Count)
        {
            output.Ranges = new DeviceAddressRange[Math.Max(planned.Count, output.Ranges.Length * 2)];
        }

        output.RangeCount = planned.Count;
        for (var index = 0; index < planned.Count; index++)
        {
            output.Ranges[index] = RangeOf(planned[index], index, values, valid);
        }

        return true;
    }

    private DeviceAddressRange RangeOf(DeviceAddressRangePlan range, int index, ulong[] values, bool[] valid)
    {
        var low = _rangeLow[index];
        var high = _rangeHigh[index];
        var plannedRange = low >= 0 && valid[low] && valid[high];
        return DeviceAddressRangePlanner.EvaluateRange(range, plannedRange,
            plannedRange ? (uint)values[low] : 0, plannedRange ? (uint)values[high] : 0);
    }

    private bool Run(ShaderResourcePlan plan, ResourceRuntimeInputs inputs, int additionalTableWords, Scratch scratch, bool[] activeSources,
        out List<DescriptorWords> sources, out uint[] table, out DeviceAddressRange[] ranges)
    {
        sources = null!;
        table = null!;
        ranges = null!;
        Evaluate(inputs, scratch);
        var values = scratch.Values;
        var valid = scratch.Valid;
        var evaluated = new List<DescriptorWords>(_sourceWords.Length);
        for (var source = 0; source < _sourceWords.Length; source++)
        {
            var registers = _sourceWords[source];
            var words = new uint[registers.Length];
            if (activeSources.Length != 0 && !activeSources[plan.MaterializationSources[source]])
            {
                evaluated.Add(new DescriptorWords(words));
                continue;
            }

            for (var word = 0; word < words.Length; word++)
            {
                var register = registers[word];
                if (!valid[register])
                {
                    return false;
                }

                words[word] = (uint)values[register];
            }

            evaluated.Add(new DescriptorWords(words));
        }

        var flattened = new uint[checked(_tableWords.Length + additionalTableWords)];
        for (var index = 0; index < _tableWords.Length; index++)
        {
            var register = _tableWords[index];
            if (!valid[register])
            {
                return false;
            }

            flattened[_tableOffsets[index]] = (uint)values[register];
        }

        var planned = plan.DeviceAddressRanges;
        var result = new DeviceAddressRange[planned.Count];
        for (var index = 0; index < result.Length; index++)
        {
            result[index] = RangeOf(planned[index], index, values, valid);
        }

        sources = evaluated;
        table = flattened;
        ranges = result;
        return true;
    }

    private static void Execute(in Step step, ResourceRuntimeInputs inputs, ulong[] values, bool[] valid, Span<ulong> operands)
    {
        switch (step.Kind)
        {
            case StepKind.UserData:
                if ((uint)step.First < (uint)inputs.UserData.Count)
                {
                    values[step.Target] = inputs.UserData[step.First];
                    valid[step.Target] = true;
                }
                else
                {
                    valid[step.Target] = false;
                }

                return;
            case StepKind.ShaderBase:
                values[step.Target] = inputs.ShaderBase;
                valid[step.Target] = true;
                return;
            case StepKind.Select:
                valid[step.Target] = valid[step.First] && valid[step.Second] && valid[step.Third];
                values[step.Target] = values[step.First] != 0 ? values[step.Second] : values[step.Third];
                return;
            case StepKind.Operation:
            {
                var registers = step.Operands!;
                var count = registers.Length;
                var span = count <= operands.Length ? operands[..count] : new ulong[count];
                for (var index = 0; index < count; index++)
                {
                    if (!valid[registers[index]])
                    {
                        valid[step.Target] = false;
                        return;
                    }

                    span[index] = values[registers[index]];
                }

                valid[step.Target] = ScalarOperationSemantics.TryEvaluate(step.Operation, span, out var result);
                values[step.Target] = result;
                return;
            }
        }
    }

    // A level's reads are put in a fixed order of base, offset and immediate once, when the program is built, so their
    // addresses mostly arrive in order; ReadLevel still puts them in address order (ReadOrder) before it finds runs.
    // SHARPEMU_STATIC_READ_ORDER=0 keeps the order they were added in.
    private static readonly bool StaticReadOrder = Environment.GetEnvironmentVariable("SHARPEMU_STATIC_READ_ORDER") != "0";

    private static int CompareReadOrder(ReadStep left, ReadStep right)
    {
        var order = left.Low.CompareTo(right.Low);
        if (order == 0) order = left.High.CompareTo(right.High);
        if (order == 0) order = left.Offset.CompareTo(right.Offset);
        if (order == 0) order = left.Records.CompareTo(right.Records);
        if (order == 0) order = left.Word3.CompareTo(right.Word3);
        if (order == 0) order = left.Kind.CompareTo(right.Kind);
        if (order == 0) order = (left.Immediate + left.ConstantOffset).CompareTo(right.Immediate + right.ConstantOffset);
        return order != 0 ? order : left.Target.CompareTo(right.Target);
    }

    // ReadLevel for a FixedReads level, with the same words, runs and fallbacks; false, reading nothing, when the base is
    // not valid, is null, or does not resolve every offset, which ReadLevel handles.
    private static bool TryReadFixed(ReadStep[] reads, FixedReads plan, ResourceRuntimeInputs inputs, Scratch scratch)
    {
        var values = scratch.Values;
        var valid = scratch.Valid;
        if (!valid[plan.Low] || !valid[plan.High])
        {
            return false;
        }

        var high = values[plan.High];
        var baseAddress = ((high << 32) | (uint)values[plan.Low]) & AddressMask;
        ref readonly var first = ref reads[plan.First];
        ref readonly var last = ref reads[plan.Last];
        if ((baseAddress & ~3ul) == 0 ||
            RuntimeValueEvaluator.ResolveRawAddress(first.Kind, baseAddress, high, 0, first.Immediate, first.ConstantOffset, out var lowest) !=
            RuntimeValueEvaluator.RawAddress.Read ||
            RuntimeValueEvaluator.ResolveRawAddress(last.Kind, baseAddress, high, 0, last.Immediate, last.ConstantOffset, out _) !=
            RuntimeValueEvaluator.RawAddress.Read)
        {
            return false;
        }

        // Every address is the lowest one plus its offset past the lowest offset.
        var origin = plan.Offsets[0];
        var order = plan.Order;
        var offsets = plan.Offsets;
        var resident = inputs.ReadResidentMemory;
        var start = 0;
        for (var run = 0; run < plan.RunEnds.Length; run++)
        {
            var end = plan.RunEnds[run];
            var runFirst = lowest + (ulong)(offsets[start] - origin);
            if (plan.RunWords[run] >= MinimumRunWords && resident is not null && inputs.ReadMemory is not null)
            {
                var length = checked((int)(offsets[end - 1] - offsets[start]) + sizeof(uint));
                if (scratch.Bytes.Length < length)
                {
                    scratch.Bytes = new byte[Math.Max(length, scratch.Bytes.Length * 2)];
                }

                var bytes = scratch.Bytes.AsSpan(0, length);
                if (resident(runFirst, bytes, inputs.ReadsClean))
                {
                    for (var index = start; index < end; index++)
                    {
                        var target = reads[order[index]].Target;
                        var word = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes[(int)(offsets[index] - offsets[start])..]);
                        values[target] = word;
                        valid[target] = true;
                        if (scratch.Recording)
                        {
                            scratch.NoteRead(runFirst + (ulong)(offsets[index] - offsets[start]), word);
                        }
                    }

                    start = end;
                    continue;
                }
            }

            for (var index = start; index < end; index++)
            {
                ref readonly var read = ref reads[order[index]];
                ReadWord(inputs, lowest + (ulong)(offsets[index] - origin), baseAddress, read.Absolute, read.Target, scratch);
            }

            start = end;
        }

        return true;
    }

    private static void ReadLevel(Level level, ResourceRuntimeInputs inputs, Scratch scratch)
    {
        var reads = level.Reads;
        var values = scratch.Values;
        var valid = scratch.Valid;
        scratch.Count = 0;
        foreach (var read in reads)
        {
            valid[read.Target] = false;
            var buffer = read.Kind == ScalarValueKind.ScalarBufferWord;
            if (!valid[read.Low] || !valid[read.High] || (buffer && (!valid[read.Records] || !valid[read.Word3])) || (read.Offset >= 0 && !valid[read.Offset]))
            {
                continue;
            }

            var high = values[read.High];
            var records = buffer ? values[read.Records] : 0;
            var offset = read.Offset >= 0 ? (uint)values[read.Offset] : read.ConstantOffset;
            var baseAddress = ((high << 32) | (uint)values[read.Low]) & AddressMask;
            switch (RuntimeValueEvaluator.ResolveRawAddress(read.Kind, baseAddress, high, records, read.Immediate, offset, out var address))
            {
                case RuntimeValueEvaluator.RawAddress.Failed:
                    continue;
                case RuntimeValueEvaluator.RawAddress.Zero:
                    values[read.Target] = 0;
                    valid[read.Target] = true;
                    continue;
            }

            scratch.Add(address, read.Target, baseAddress, read.Absolute);
        }

        var count = scratch.Count;
        if (count == 0)
        {
            return;
        }

        var addresses = scratch.Addresses;
        var targets = scratch.Targets;
        var bases = scratch.NullBases;
        var absolute = scratch.Absolute;
        scratch.Order.Sort(addresses, targets, bases, absolute, count, ref level.Order);

        var resident = inputs.ReadResidentMemory;
        var start = 0;
        while (start < count)
        {
            var end = start + 1;
            var words = 1;
            while (end < count && addresses[end] - addresses[end - 1] <= sizeof(uint) && addresses[end] - addresses[start] < MaximumRunBytes &&
                   (bases[end] & ~3ul) != 0 && (bases[start] & ~3ul) != 0)
            {
                if (addresses[end] != addresses[end - 1])
                {
                    words++;
                }

                end++;
            }

            if (words >= MinimumRunWords && resident is not null && inputs.ReadMemory is not null && ReadRun(resident, inputs.ReadsClean, scratch, start, end))
            {
                start = end;
                continue;
            }

            for (var index = start; index < end; index++)
            {
                ReadWord(inputs, addresses[index], bases[index], absolute[index], targets[index], scratch);
            }

            start = end;
        }
    }

    /// <summary>
    /// Puts a level's reads in address order, equal addresses in the order they were added (a stable sort). A
    /// table of descriptors interleaves its words across the reads, so they rarely arrive in order: one sort of
    /// packed (address, position) keys replaces moving every out-of-order read past each read before it. A level's
    /// reads mostly come back in the order they took last time, so that order (the caller's, one per program level)
    /// is tried first: the keys are unique, so an order that leaves them increasing is the sorted one.
    /// </summary>
    internal sealed class ReadOrder
    {
        private ulong[] _keys = new ulong[64];
        private ulong[] _addresses = new ulong[64];
        private int[] _targets = new int[64];
        private ulong[] _bases = new ulong[64];
        private bool[] _absolute = new bool[64];
        private int[] _order = new int[64];

        public void Sort(ulong[] addresses, int[] targets, ulong[] bases, bool[] absolute, int count)
        {
            int[]? none = null;
            Sort(addresses, targets, bases, absolute, count, ref none, remember: false);
        }

        // With the order these reads took last time in remembered, replaced by this sort's order when it does not fit.
        public void Sort(ulong[] addresses, int[] targets, ulong[] bases, bool[] absolute, int count, ref int[]? remembered) =>
            Sort(addresses, targets, bases, absolute, count, ref remembered, remember: true);

        private void Sort(ulong[] addresses, int[] targets, ulong[] bases, bool[] absolute, int count, ref int[]? remembered, bool remember)
        {
            var ordered = true;
            var packable = count <= 1 << 16;
            for (var index = 0; index < count; index++)
            {
                ordered &= index == 0 || addresses[index - 1] <= addresses[index];
                packable &= addresses[index] <= AddressMask;
            }

            if (ordered)
            {
                return;
            }

            if (!packable)
            {
                InsertionSort(addresses, targets, bases, absolute, count);
                return;
            }

            if (_keys.Length < count)
            {
                var length = Math.Max(count, _keys.Length * 2);
                _keys = new ulong[length];
                _addresses = new ulong[length];
                _targets = new int[length];
                _bases = new ulong[length];
                _absolute = new bool[length];
                _order = new int[length];
            }

            var known = remember ? Volatile.Read(ref remembered) : null;
            if (known is not null && known.Length == count && Orders(known, addresses, count))
            {
                Permute(known, addresses, targets, bases, absolute, count);
                return;
            }

            var keys = _keys.AsSpan(0, count);
            for (var index = 0; index < count; index++)
            {
                keys[index] = (addresses[index] << 16) | (uint)index;
            }

            keys.Sort();
            // A new array to remember: another thread may be checking the one published before.
            var order = remember ? new int[count] : _order;
            for (var index = 0; index < count; index++)
            {
                order[index] = (int)(keys[index] & 0xFFFF);
            }

            if (remember)
            {
                Volatile.Write(ref remembered, order);
            }

            Permute(order, addresses, targets, bases, absolute, count);
        }

        // Whether taking the reads in this order leaves their (address, position) keys strictly increasing.
        private static bool Orders(int[] order, ulong[] addresses, int count)
        {
            var previous = (addresses[order[0]] << 16) | (uint)order[0];
            for (var index = 1; index < count; index++)
            {
                var from = order[index];
                var key = (addresses[from] << 16) | (uint)from;
                if (key <= previous)
                {
                    return false;
                }

                previous = key;
            }

            return true;
        }

        private void Permute(int[] order, ulong[] addresses, int[] targets, ulong[] bases, bool[] absolute, int count)
        {
            Array.Copy(addresses, _addresses, count);
            Array.Copy(targets, _targets, count);
            Array.Copy(bases, _bases, count);
            Array.Copy(absolute, _absolute, count);
            for (var index = 0; index < count; index++)
            {
                var from = order[index];
                addresses[index] = _addresses[from];
                targets[index] = _targets[from];
                bases[index] = _bases[from];
                absolute[index] = _absolute[from];
            }
        }

        internal static void InsertionSort(ulong[] addresses, int[] targets, ulong[] bases, bool[] absolute, int count)
        {
            for (var index = 1; index < count; index++)
            {
                var address = addresses[index];
                var target = targets[index];
                var baseAddress = bases[index];
                var isAbsolute = absolute[index];
                var position = index - 1;
                while (position >= 0 && addresses[position] > address)
                {
                    addresses[position + 1] = addresses[position];
                    targets[position + 1] = targets[position];
                    bases[position + 1] = bases[position];
                    absolute[position + 1] = absolute[position];
                    position--;
                }

                addresses[position + 1] = address;
                targets[position + 1] = target;
                bases[position + 1] = baseAddress;
                absolute[position + 1] = isAbsolute;
            }
        }
    }

    private static bool ReadRun(ResidentGuestBytesReader reader, bool clean, Scratch scratch, int start, int end)
    {
        var addresses = scratch.Addresses;
        var first = addresses[start];
        var length = checked((int)(addresses[end - 1] - first) + sizeof(uint));
        if (scratch.Bytes.Length < length)
        {
            scratch.Bytes = new byte[Math.Max(length, scratch.Bytes.Length * 2)];
        }

        var bytes = scratch.Bytes.AsSpan(0, length);
        if (!reader(first, bytes, clean))
        {
            return false;
        }

        for (var index = start; index < end; index++)
        {
            var target = scratch.Targets[index];
            var word = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes[(int)(addresses[index] - first)..]);
            scratch.Values[target] = word;
            scratch.Valid[target] = true;
            if (scratch.Recording)
            {
                scratch.NoteRead(addresses[index], word);
            }
        }

        return true;
    }

    private static void ReadWord(ResourceRuntimeInputs inputs, ulong address, ulong baseAddress, bool absolute, int target, Scratch scratch)
    {
        var values = scratch.Values;
        var valid = scratch.Valid;
        if (inputs.ReadMemory is not null && inputs.ReadMemory(address, out var word))
        {
            values[target] = word;
            valid[target] = true;
            if (scratch.Recording)
            {
                scratch.NoteRead(address, word);
            }

            return;
        }

        values[target] = 0;
        valid[target] = (baseAddress & ~3ul) == 0 || (absolute && inputs.ReadMemory is not null && !inputs.ReadsClean);
        if (scratch.Recording)
        {
            scratch.Unreadable.Add(address);
        }
    }
}
