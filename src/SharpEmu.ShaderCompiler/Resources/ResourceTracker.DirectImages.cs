// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Resources;

public sealed partial class ResourceTracker
{
    // The records a counted loop's direct image selects among. A loop whose bound is read per draw
    // and runs further reads its later records' images as the first one's.
    internal const int MaxLoopDirectCandidates = 64;

    // The values a lane-read selector may take below the guard that bounds it (Ghost of Tsushima's material
    // records: an 8-bit material id below 255), and the records of a V#'s table a selector the host cannot know
    // picks without one (a workgroup's record in a tiled job); a record past the end of the table reads as null.
    // (A V#'s table of records sampled only is a bindless one: TryMakeDenseIndirectImage.)
    internal const int MaxLaneDirectCandidates = 256;
    internal const int MaxRecordDirectCandidates = 256;

    // SHARPEMU_TRACE_SIGNED_LOOP_IMAGES=1 (diagnostic): each program whose counted loop selects images under a signed bound.
    private static readonly bool TraceSignedLoopImages = Environment.GetEnvironmentVariable("SHARPEMU_TRACE_SIGNED_LOOP_IMAGES") == "1";

    private bool TryMakeDirectImage(ScalarValue handle, out IndirectImagePlan plan)
    {
        plan = null!;
        if (handle.Kind != ScalarValueKind.ImageHandle || handle.Operands.Length != 8)
            return false;

        // The descriptor is read through an address (s_load) or a V# (s_buffer_load), all eight dwords alike. A loop that
        // reloads it every iteration carries the registers through phis that only merge the read with themselves (Ghost
        // of Tsushima's menu wind, which walks a table of four textures per emitter): the read is the descriptor.
        var reads = handle.Operands.Select(operand => operand.Kind == ScalarValueKind.Phi ? Invariant(operand) : operand).ToArray();
        var readKind = reads[0].Kind;
        var memoryKind = readKind == ScalarValueKind.ScalarBufferWord ? MemoryResourceKind.ScalarBuffer : MemoryResourceKind.ScalarAddress;
        var memoryIndices = new int[8];
        var canSuppressMemoryReads = true;
        for (var component = 0; component < reads.Length; component++)
        {
            var read = reads[component];
            if (read.Kind is not (ScalarValueKind.ScalarAddressWord or ScalarValueKind.ScalarBufferWord) || read.Kind != readKind ||
                read.MemoryIndex < 0 || read.MemoryIndex >= _plan.Memory.Count || !MemoryIndexBelongsTo(read.MemoryIndex, read) ||
                !(UsesOnly(read, [handle]) || UsedByDescriptorsAndLoopPhis(read, handle))) return false;
            var memory = _plan.Memory[read.MemoryIndex];
            if (memory.Kind != memoryKind || memory.DataBits != 32 || memory.DataDwords != 1)
                return false;
            canSuppressMemoryReads &= HasOnlyImageConsumers(memory, handle);
            memoryIndices[component] = read.MemoryIndex;
        }

        // The key is the load's dynamic offset, whichever of its dwords the descriptor starts at (the second
        // descriptor of an s_buffer_load_dwordx16 starts at the ninth).
        var keyRead = reads[0];
        var keyMemory = _plan.Memory[keyRead.MemoryIndex];
        var keyInstruction = _graph.Program.Instructions.First(instruction => instruction.Pc == keyMemory.Pc);
        if (keyInstruction.Control is not Gen5ScalarMemoryControl { DynamicOffsetRegister: not null })
            return false;
        var block = _graph.ControlFlow.Blocks.First(candidate => keyMemory.Pc >= candidate.StartPc && keyMemory.Pc < candidate.EndPc);
        if (memoryIndices.Any(index => _plan.Memory[index].Pc < block.StartPc || _plan.Memory[index].Pc >= block.EndPc))
            return false;

        // The record offset is built from one selector: a bit scan of a lane mask (s_ff1), the counter of a
        // loop over a table of records that each hold a descriptor (Ghost of Tsushima's terrain height patches),
        // a value read from the wave's lanes behind a guard that bounds it (its material records), or any value the
        // host cannot know (a workgroup id's bits, a list entry it indexes) picking a record of a V#'s table.
        var pending = new Stack<ScalarValue>(reads.Select(read => read.Operands[1]));
        var visited = new HashSet<ScalarValue>();
        ScalarValue? selector = null;
        while (pending.TryPop(out var value))
        {
            if (!visited.Add(value)) continue;
            if (value.Kind == ScalarValueKind.Phi && _graph.ResolveInvariantPhi(value) is { } invariant)
            {
                pending.Push(invariant);
                continue;
            }
            if ((value.Kind == ScalarValueKind.Operation && value.Operation == ScalarOperation.FindLowestBit32) ||
                value.Kind == ScalarValueKind.Phi || IsLaneReadSelector(value) || IsOpaqueSelector(value))
            {
                if (selector is not null && !ReferenceEquals(selector, value)) return false;
                selector = value;
                continue;
            }
            if (!value.IsConstant && value.Kind != ScalarValueKind.Operation) return false;
            foreach (var operand in value.Operands) pending.Push(operand);
        }
        if (selector is null) return false;

        IEnumerable<uint> results;
        var unboundedLoop = false;
        ScalarValue? limit = null;
        var inclusive = false;
        var signedLimit = false;
        var laneRead = IsLaneReadSelector(selector);
        var opaque = !laneRead && selector.Kind != ScalarValueKind.Phi &&
            !(selector.Kind == ScalarValueKind.Operation && selector.Operation == ScalarOperation.FindLowestBit32);
        if (laneRead || opaque)
        {
            // Every value below the guard, or every record of a V#'s table; the records past its end read as null.
            if (!laneRead || !TryLaneSelectorBound(selector, _graph.ControlFlow.BlockOf(keyMemory.Pc), out var bound))
            {
                if (readKind != ScalarValueKind.ScalarBufferWord) return false;
                bound = MaxRecordDirectCandidates;
            }
            results = Enumerable.Range(0, (int)bound).Select(index => (uint)index);
        }
        else if (selector.Kind == ScalarValueKind.Phi)
        {
            // Every counter value the loop can reach, up to the cap; a record past the loop's bound is never
            // read (the materializer compares each candidate's counter with the bound it evaluates per draw).
            if (!BufferCandidateTablePlanner.TryProveInductionBound(_plan, selector, out var initial, out var step,
                    out limit, out inclusive, out signedLimit, exitGuards: true, signedGuards: true) ||
                !initial.IsConstant || limit is null)
            {
                // A nested loop's counter whose bound is not tied to its phi: its first counter values, the records
                // past the readable end of the table null. The shader only looks up the counters it reaches.
                if (!BufferCandidateTablePlanner.TryProveInductionStep(selector, out initial, out step) || !initial.IsConstant)
                    return false;
                limit = null;
                inclusive = false;
                signedLimit = false;
                unboundedLoop = true;
            }
            var first = initial.ConstantU32;
            // Under a signed bound the candidates' counters must stay non-negative, where both orders agree.
            if (signedLimit && (first > int.MaxValue || step > (int.MaxValue - first) / (MaxLoopDirectCandidates + 1)))
                return false;
            if (signedLimit && TraceSignedLoopImages)
                Console.Error.WriteLine($"[SHADER][SIGNED-LOOP-IMAGE] hash=0x{_plan.Hash:X16} stage={_plan.Stage} first={first} step={step} " +
                    $"limit={(limit.IsConstant ? limit.ConstantU32.ToString() : "runtime")} inclusive={inclusive}");
            var counters = Enumerable.Range(0, MaxLoopDirectCandidates + 1).Select(index => first + (uint)index * step);
            if (limit is { IsConstant: true })
            {
                var bound = limit.ConstantU32;
                var inside = inclusive;
                var signed = signedLimit;
                var reached = counters.TakeWhile(counter => ResourceMaterializer.CounterWithinLimit(counter, bound, inside, signed))
                    .ToArray();
                if (reached.Length > MaxLoopDirectCandidates) return false;
                results = reached;
                limit = null;
            }
            else
            {
                results = counters.Take(MaxLoopDirectCandidates);
            }
        }
        else
        {
            // Include the all-zero input result unless the scan's incoming edge proves it cannot occur.
            results = Enumerable.Range(0, 32).Select(index => (uint)index);
            if (!_graph.HasNonZeroBitScanInput(selector)) results = results.Append(uint.MaxValue);
        }

        var candidates = new List<DirectImageCandidate>();
        var sources = new List<DescriptorSource>();
        var keys = new HashSet<uint>();
        // The table's address a loop carries unchanged is the address before the loop.
        var carried = LoopCarriedInvariants(reads.Select(read => read.Operands[0]), selector);
        foreach (var result in results)
        {
            var replacements = new Dictionary<ScalarValue, ScalarValue>(carried) { [selector] = _graph.Constant(result) };
            var memo = new Dictionary<ScalarValue, ScalarValue>();
            var key = _graph.Substitute(keyRead.Operands[1], replacements, memo);
            if (!key.IsConstant || key.Type != ScalarValueType.U32 || !keys.Add(key.ConstantU32)) return false;
            var source = new DescriptorSource
            {
                Dwords = reads.Select(read => _graph.Substitute(read, replacements, memo)).ToArray(),
            };
            if (!ValidateSource(source, out _)) return false;
            sources.Add(source);
            candidates.Add(new DirectImageCandidate(key.ConstantU32, 0, selector.Kind == ScalarValueKind.Phi ? result : null));
        }
        if (candidates.Count == 0) return false;

        for (var index = 0; index < candidates.Count; index++)
            candidates[index] = candidates[index] with { Source = InternSource(sources[index]) };
        uint? limitSource = null;
        if (limit is not null)
        {
            var bound = new DescriptorSource { Dwords = [limit] };
            if (!ValidateSource(bound, out _)) return false;
            limitSource = InternSource(bound);
        }
        var imageSource = new DescriptorSource
        {
            Dwords = sources[0].Dwords,
            IndirectImage = new IndirectImageSelector(0, 0, 0, 0, 0)
            {
                DirectCandidates = candidates,
                DirectLimitSource = limitSource,
                DirectLimitInclusive = inclusive,
                DirectLimitSigned = signedLimit,
                DirectRecordsMayEnd = laneRead || opaque || unboundedLoop,
            },
        };
        plan = new IndirectImagePlan
        {
            Handle = handle,
            Source = InternSource(imageSource),
            Key = keyRead,
            KeyIsAddressOffset = true,
            SuppressMemoryReads = canSuppressMemoryReads,
            Memory = memoryIndices,
            Reads = reads,
        };
        return true;
    }

    // The phis under these values that only merge a value with itself, each mapped to that value, so a substitution
    // rebuilds the value without them; the selector's own phi is left alone.
    private Dictionary<ScalarValue, ScalarValue> LoopCarriedInvariants(IEnumerable<ScalarValue> roots, ScalarValue selector)
    {
        var carried = new Dictionary<ScalarValue, ScalarValue>();
        var pending = new Stack<ScalarValue>(roots);
        var visited = new HashSet<ScalarValue>();
        while (pending.TryPop(out var value))
        {
            if (!visited.Add(value) || ReferenceEquals(value, selector)) continue;
            if (value.Kind == ScalarValueKind.Phi)
            {
                if (_graph.ResolveInvariantPhi(value) is { } invariant && !ReferenceEquals(invariant, value))
                {
                    carried[value] = invariant;
                    pending.Push(invariant);
                }
                continue;
            }
            foreach (var operand in value.Operands) pending.Push(operand);
        }
        return carried;
    }

    // A descriptor read through a pointer that a record of a V#'s table holds, the record picked by a value the host
    // cannot know: Ghost of Tsushima's foliage culling, whose waves each test their batch against the depth pyramid
    // of the view their record points at. The key is the pointer read's dynamic offset, the record's offset in the
    // table; each record's candidate is the descriptor its pointer reaches, and a record past the table's end is null.
    private bool TryMakeChainedDirectImage(ScalarValue handle, out IndirectImagePlan plan)
    {
        plan = null!;
        if (!ChainedDirectImages || handle.Kind != ScalarValueKind.ImageHandle || handle.Operands.Length != 8)
            return false;

        var reads = handle.Operands;
        var address = reads[0].Kind == ScalarValueKind.ScalarAddressWord ? reads[0].Operands[0] : null;
        if (address is not { Kind: ScalarValueKind.AddressHandle, Operands.Length: 2 })
            return false;
        var memoryIndices = new int[8];
        var canSuppressMemoryReads = true;
        for (var component = 0; component < reads.Length; component++)
        {
            var read = reads[component];
            if (read.Kind != ScalarValueKind.ScalarAddressWord || !ReferenceEquals(read.Operands[0], address) ||
                !read.Operands[1].IsConstant || read.MemoryIndex < 0 || read.MemoryIndex >= _plan.Memory.Count ||
                !MemoryIndexBelongsTo(read.MemoryIndex, read) || !UsesOnly(read, [handle])) return false;
            var memory = _plan.Memory[read.MemoryIndex];
            if (memory.Kind != MemoryResourceKind.ScalarAddress || memory.DataBits != 32 || memory.DataDwords != 1)
                return false;
            canSuppressMemoryReads &= HasOnlyImageConsumers(memory, handle);
            memoryIndices[component] = read.MemoryIndex;
        }

        // The pointer's two dwords are one record's, read through a V# at the offset of one selector's record (and
        // carried unchanged through the loops between that read and the descriptor's).
        var pointer = address.Operands.Select(Invariant).ToArray();
        if (pointer.Any(dword => dword.Kind != ScalarValueKind.ScalarBufferWord || dword.MemoryIndex < 0 ||
                dword.MemoryIndex >= _plan.Memory.Count) ||
            !_graph.Equivalent(pointer[0].Operands[0], pointer[1].Operands[0]))
            return false;
        var keyRead = pointer[0];
        var keyMemory = _plan.Memory[keyRead.MemoryIndex];
        var keyInstruction = _graph.Program.Instructions.First(instruction => instruction.Pc == keyMemory.Pc);
        if (keyInstruction.Control is not Gen5ScalarMemoryControl { DynamicOffsetRegister: not null, ImmediateOffsetBytes: 0 } ||
            keyMemory.ComponentIndex != 0)
            return false;

        var pending = new Stack<ScalarValue>(pointer.Select(dword => dword.Operands[1]));
        var visited = new HashSet<ScalarValue>();
        ScalarValue? selector = null;
        while (pending.TryPop(out var value))
        {
            if (!visited.Add(value)) continue;
            if (IsLaneReadSelector(value) || IsOpaqueSelector(value))
            {
                if (selector is not null && !ReferenceEquals(selector, value)) return false;
                selector = value;
                continue;
            }
            if (!value.IsConstant && value.Kind != ScalarValueKind.Operation) return false;
            foreach (var operand in value.Operands) pending.Push(operand);
        }
        if (selector is null) return false;

        var candidates = new List<DirectImageCandidate>();
        var sources = new List<DescriptorSource>();
        var keys = new HashSet<uint>();
        for (var record = 0u; record < MaxRecordDirectCandidates; record++)
        {
            var replacements = new Dictionary<ScalarValue, ScalarValue> { [selector] = _graph.Constant(record) };
            var memo = new Dictionary<ScalarValue, ScalarValue>();
            var key = _graph.Substitute(keyRead.Operands[1], replacements, memo);
            if (!key.IsConstant || key.Type != ScalarValueType.U32 || !keys.Add(key.ConstantU32)) return false;
            for (var dword = 0; dword < pointer.Length; dword++)
                replacements[address.Operands[dword]] = _graph.Substitute(pointer[dword], replacements, memo);
            var source = new DescriptorSource
            {
                Dwords = reads.Select(read => _graph.Substitute(read, replacements, [])).ToArray(),
            };
            if (!ValidateSource(source, out _)) return false;
            sources.Add(source);
            candidates.Add(new DirectImageCandidate(key.ConstantU32, 0));
        }

        for (var index = 0; index < candidates.Count; index++)
            candidates[index] = candidates[index] with { Source = InternSource(sources[index]) };
        var imageSource = new DescriptorSource
        {
            Dwords = sources[0].Dwords,
            IndirectImage = new IndirectImageSelector(0, 0, 0, 0, 0)
            {
                DirectCandidates = candidates,
                DirectRecordsMayEnd = true,
            },
        };
        plan = new IndirectImagePlan
        {
            Handle = handle,
            Source = InternSource(imageSource),
            Key = keyRead,
            KeyIsAddressOffset = true,
            SuppressMemoryReads = canSuppressMemoryReads,
            Memory = memoryIndices,
            Reads = reads,
        };
        if (TraceChainedImages)
            Console.Error.WriteLine($"[SHADER][CHAINED-IMAGE] hash=0x{_plan.Hash:X16} stage={_plan.Stage} key_pc=0x{keyMemory.Pc:X} " +
                $"descriptor_pc=0x{_plan.Memory[memoryIndices[0]].Pc:X} candidates={candidates.Count} suppress={canSuppressMemoryReads}");
        return true;
    }

    // SHARPEMU_CHAINED_DIRECT_IMAGES=0 (diagnostic): leaves descriptors read through a record's pointer to the null
    // fallback, as before; SHARPEMU_TRACE_CHAINED_IMAGES=1 names each program that selects one per record.
    private static readonly bool ChainedDirectImages = Environment.GetEnvironmentVariable("SHARPEMU_CHAINED_DIRECT_IMAGES") != "0";
    private static readonly bool TraceChainedImages = Environment.GetEnvironmentVariable("SHARPEMU_TRACE_CHAINED_IMAGES") == "1";

    // A value the host cannot know that a record offset is built from: one derived from a workgroup id (a tiled job's
    // per-tile records), from a lane's value (v_readlane), or read from memory at an offset built from one; not the
    // offset arithmetic itself (an add, a multiply or a shift by a constant) around it.
    private bool IsOpaqueSelector(ScalarValue value) =>
        !value.IsConstant && !IsOffsetArithmetic(value) && (DependsOnWaveValue(value) || IsListEntryRead(value));

    // A descriptor a loop also carries on to later samples (cs7B97365C keeps each record's texture in its registers for
    // the next pass over the tile, and samples it again past inner loops that leave those registers alone, through phis
    // that only carry the read): each sample takes it from the read, directly or through those phis, and the read stays
    // in the shader for the other samples and the carried copy (HasOnlyImageConsumers sees the register live past this one).
    // SHARPEMU_SHARED_DIRECT_READS=0 (diagnostic) only takes the read for a sample that uses it directly, as before.
    private static readonly bool SharedDirectReads = Environment.GetEnvironmentVariable("SHARPEMU_SHARED_DIRECT_READS") != "0";

    private bool UsedByDescriptorsAndLoopPhis(ScalarValue read, ScalarValue handle) =>
        _uses.TryGetValue(read, out var uses) && uses.Count != 0 &&
        (SharedDirectReads || uses.Any(user => ReferenceEquals(user, handle))) &&
        uses.All(user => user.Kind is ScalarValueKind.ImageHandle or ScalarValueKind.SamplerHandle or ScalarValueKind.Phi);

    // A list entry a loop walks: memory read at an offset built from a loop counter that starts at a value the host
    // cannot know (Ghost of Tsushima's clustered decals, cs7B97365C: each tile walks its own run of a list of record
    // indices, from a start read out of the tile's header, and takes each record's texture from a table of 536-byte
    // records). The counter is a phi, which DependsOnWaveValue does not look through.
    private bool IsListEntryRead(ScalarValue value) =>
        value.Kind is ScalarValueKind.ScalarBufferWord or ScalarValueKind.ScalarAddressWord &&
        value.Operands.Skip(1).Any(operand => DependsOnWaveValueThroughPhis(operand, []));

    private bool DependsOnWaveValueThroughPhis(ScalarValue value, HashSet<ScalarValue> visiting)
    {
        if (value.Kind != ScalarValueKind.Phi)
            return DependsOnWaveValue(value) || (value.Kind == ScalarValueKind.Operation &&
                value.Operands.Any(operand => operand.Kind == ScalarValueKind.Phi && DependsOnWaveValueThroughPhis(operand, visiting)));
        return visiting.Add(value) && value.Operands.Any(operand => DependsOnWaveValueThroughPhis(operand, visiting));
    }

    private static bool IsOffsetArithmetic(ScalarValue value) =>
        value is { Kind: ScalarValueKind.Operation, Operation: ScalarOperation.IAdd32 or ScalarOperation.IMul32 or ScalarOperation.ShiftLeft32 } &&
        value.Operands.Any(operand => operand.IsConstant);

    private readonly Dictionary<ScalarValue, bool> _dependsOnWaveValue = [];

    private bool DependsOnWaveValue(ScalarValue value)
    {
        if (_dependsOnWaveValue.TryGetValue(value, out var known))
            return known;
        _dependsOnWaveValue[value] = false;
        var result = value.Kind == ScalarValueKind.Undefined
            ? _graph.TryGetUndefinedOrigin(value, out var origin) &&
                origin.Opcode is ScalarValueGraph.SystemRegisterOrigin or "VReadlaneB32"
            : value.Kind != ScalarValueKind.Phi && value.Operands.Any(DependsOnWaveValue);
        _dependsOnWaveValue[value] = result;
        return result;
    }

    // A value the shader read from its wave's lanes (v_readlane), or the smallest or largest of such values: the
    // material id a waterfall loop handles this pass, which the host cannot know but which the wave shares.
    private bool IsLaneReadSelector(ScalarValue value)
    {
        bool LaneRead(ScalarValue candidate) =>
            candidate.Kind == ScalarValueKind.Undefined && _graph.TryGetUndefinedOrigin(candidate, out var origin) &&
            origin.Opcode == "VReadlaneB32";
        return LaneRead(value) ||
            value is { Kind: ScalarValueKind.Operation, Operation: ScalarOperation.UMin32 or ScalarOperation.UMax32 } &&
            value.Operands.All(LaneRead);
    }

    // The bound a guard puts on a lane-read selector before the block that reads the record: a branch that
    // leaves when the selector is at or above it (s_cmpk_ge_u32 then s_cbranch_scc1), in the block whose lane
    // reads make the selector, so no path reaches the block without the guard having tested that very value.
    private bool TryLaneSelectorBound(ScalarValue selector, int readBlock, out uint bound)
    {
        bound = 0;
        var flow = _graph.ControlFlow;
        if (readBlock < 0) return false;
        var laneReads = selector.Kind == ScalarValueKind.Undefined ? [selector] : selector.Operands;
        var found = false;
        foreach (var (pc, condition) in _graph.BranchConditions)
        {
            var test = condition;
            var negated = test is { Kind: ScalarValueKind.Operation, Operation: ScalarOperation.LogicalNot };
            if (negated) test = test.Operands[0];
            if (test.Kind != ScalarValueKind.Operation || test.Operands.Length != 2 || !test.Operands[1].IsConstant ||
                !ReferenceEquals(test.Operands[0], selector) && !_graph.Equivalent(test.Operands[0], selector))
                continue;
            var constant = test.Operands[1].ConstantU32;
            // The largest value the selector keeps past the guard, plus one.
            var limit = (test.Operation, negated) switch
            {
                (ScalarOperation.UGreaterThanEqual32, false) or (ScalarOperation.ULessThan32, true) => (ulong)constant,
                (ScalarOperation.UGreaterThan32, false) or (ScalarOperation.ULessThanEqual32, true) => constant + 1ul,
                _ => ulong.MaxValue,
            };
            if (limit == 0 || limit > MaxLaneDirectCandidates) continue;

            var guard = flow.BlockOf(pc);
            if (guard < 0 || guard == readBlock) continue;
            var range = flow.Blocks[guard];
            if (laneReads.Any(read => !_graph.TryGetUndefinedOrigin(read, out var origin) ||
                    origin.Pc < range.StartPc || origin.Pc >= pc)) continue;
            var fallThrough = flow.Successors[guard].Where(successor => flow.Blocks[successor].StartPc == range.EndPc).ToArray();
            var taken = flow.Successors[guard].Where(successor => flow.Blocks[successor].StartPc != range.EndPc).ToArray();
            if (fallThrough.Length != 1 || taken.Length != 1) continue;

            // Neither the program's start nor the branch's target reaches the record without passing the guard again.
            var pending = new Stack<int>();
            var seen = new HashSet<int> { guard };
            if (guard != 0) pending.Push(0);
            pending.Push(taken[0]);
            var reaches = false;
            while (pending.TryPop(out var block))
            {
                if (!seen.Add(block)) continue;
                if (block == readBlock)
                {
                    reaches = true;
                    break;
                }
                foreach (var successor in flow.Successors[block]) pending.Push(successor);
            }
            if (reaches) continue;

            bound = found ? Math.Min(bound, (uint)limit) : (uint)limit;
            found = true;
        }
        return found;
    }

    // A sampler read from the record a counted loop selects (beside the image a direct table selects) has no
    // candidates of its own: it reads as the first record's, which the loop reads first, rather than as null. One
    // read from the record a value the host cannot know selects reads as the first record's as well.
    // A descriptor read through a pointer the record holds keeps the pointer: the selector beneath the pointer's
    // read is what takes the first record's value, not the read or the handle it builds. Only when the host cannot
    // read that pointer does the read itself stand for the first record, as it did before.
    private bool TryFirstIterationSource(DescriptorSource source, out DescriptorSource first)
    {
        // A word a loop carries unchanged through phis that only merge it with itself is its read (cs7B97365C samples each
        // decal again past inner loops that keep the record's sampler in its registers): take that read's first record,
        // as for the sample that uses the read directly.
        source = new DescriptorSource
        {
            Dwords = source.Dwords.Select(dword => dword.Kind == ScalarValueKind.Phi ? Invariant(dword) : dword).ToArray(),
        };
        if (ChainedDirectImages && TryFirstIterationSource(source, throughReads: true, out first))
        {
            var through = first;
            if (TraceChainedImages && (!TryFirstIterationSource(source, throughReads: false, out var before) ||
                    before.Dwords.Where((dword, index) => !_graph.Equivalent(dword, through.Dwords[index])).Any()))
                Console.Error.WriteLine($"[SHADER][CHAINED-SAMPLER] hash=0x{_plan.Hash:X16} stage={_plan.Stage} first record through reads");
            return true;
        }

        return TryFirstIterationSource(source, throughReads: false, out first);
    }

    private static bool IsReadOrHandle(ScalarValue value) => value.Kind is ScalarValueKind.ScalarBufferWord or
        ScalarValueKind.ScalarAddressWord or ScalarValueKind.AddressHandle or ScalarValueKind.BufferHandle;

    private bool TryFirstIterationSource(DescriptorSource source, bool throughReads, out DescriptorSource first)
    {
        first = source;
        var replacements = new Dictionary<ScalarValue, ScalarValue>();
        var invariantPhis = new List<ScalarValue>();
        var pending = new Stack<ScalarValue>(source.Dwords);
        var visited = new HashSet<ScalarValue>();
        while (pending.TryPop(out var value))
        {
            if (!visited.Add(value)) continue;
            // The descriptor's own words are the reads, not what selects their record.
            if (IsOpaqueSelector(value) && value.Type == ScalarValueType.U32 && !source.Dwords.Contains(value) &&
                !(throughReads && IsReadOrHandle(value)))
            {
                replacements[value] = _graph.Constant(0u);
                continue;
            }
            if (value.Kind == ScalarValueKind.Phi)
            {
                // A pointer a loop carries unchanged is the read before the loop, which the selector beneath picks.
                if (_graph.ResolveInvariantPhi(value) is { } invariant)
                {
                    if (throughReads)
                    {
                        invariantPhis.Add(value);
                        pending.Push(invariant);
                    }
                    continue;
                }
                if (!BufferCandidateTablePlanner.TryProveInductionBound(_plan, value, out var initial, out _, out _, out _,
                        exitGuards: true) || !initial.IsConstant) return false;
                replacements[value] = initial;
                continue;
            }
            foreach (var operand in value.Operands) pending.Push(operand);
        }
        if (replacements.Count == 0) return false;

        foreach (var phi in invariantPhis)
            replacements[phi] = _graph.Substitute(_graph.ResolveInvariantPhi(phi)!, replacements, []);
        var memo = new Dictionary<ScalarValue, ScalarValue>();
        first = new DescriptorSource { Dwords = source.Dwords.Select(dword => _graph.Substitute(dword, replacements, memo)).ToArray() };
        return ValidateSource(first, out _);
    }

    // A walk that reaches this many instructions before the register is written again keeps the load.
    private const int MaxConsumerWalkInstructions = 4096;

    private Dictionary<uint, int>? _instructionIndexByPc;

    // The index of the first instruction at or after pc, in the program's ascending pc order.
    private int FirstInstructionAt(uint pc)
    {
        var instructions = _graph.Program.Instructions;
        if (_instructionIndexByPc is null)
        {
            _instructionIndexByPc = new Dictionary<uint, int>(instructions.Count);
            for (var index = 0; index < instructions.Count; index++)
            {
                _instructionIndexByPc.TryAdd(instructions[index].Pc, index);
            }
        }

        if (_instructionIndexByPc.TryGetValue(pc, out var found))
        {
            return found;
        }

        var low = 0;
        var high = instructions.Count;
        while (low < high)
        {
            var middle = low + ((high - low) >> 1);
            if (instructions[middle].Pc < pc) low = middle + 1;
            else high = middle;
        }

        return low;
    }

    // Resource-graph uses omit ordinary shader arithmetic; check those reads before removing a load.
    private bool HasOnlyImageConsumers(MemoryAccessInfo memory, ScalarValue handle)
    {
        var instructions = _graph.Program.Instructions;
        var loadIndex = FirstInstructionAt(memory.Pc);
        if (loadIndex >= instructions.Count || instructions[loadIndex].Pc != memory.Pc) return false;
        var load = instructions[loadIndex];
        if (memory.ComponentIndex >= load.Destinations.Count) return false;
        var destination = load.Destinations[(int)memory.ComponentIndex];
        if (destination.Kind != Gen5OperandKind.ScalarRegister || destination.Value >= 106) return false;
        var flow = _graph.ControlFlow;
        var initialBlock = flow.BlockOf(load.Pc);
        if (initialBlock < 0) return false;
        var pending = new Queue<(int Block, uint Start)>();
        var visited = new HashSet<(int Block, uint Start)>();
        var walked = 0;
        pending.Enqueue((initialBlock, load.Pc + (uint)load.Words.Count * sizeof(uint)));
        while (pending.TryDequeue(out var position))
        {
            if (!visited.Add(position)) continue;
            var block = flow.Blocks[position.Block];
            var overwritten = false;
            for (var index = FirstInstructionAt(position.Start); index < instructions.Count && instructions[index].Pc < block.EndPc; index++)
            {
                var instruction = instructions[index];
                if (++walked > MaxConsumerWalkInstructions) return false;
                if (instruction.Opcode.Contains("rel", StringComparison.OrdinalIgnoreCase) ||
                    instruction.Opcode.Contains("GprIdx", StringComparison.Ordinal)) return false;
                var allowedImage = instruction.Control is Gen5ImageControl &&
                    _plan.Memory.TryGetIndex(instruction.Pc, 0, out var imageIndex) &&
                    ReferenceEquals(_plan.Accesses[imageIndex]?.Handle, handle);
                if (!allowedImage && (ReadsScalar(instruction, destination.Value) ||
                    (instruction.Encoding == Gen5ShaderEncoding.Sopk && instruction.Opcode is "SAddkI32" or "SMulkI32" &&
                        instruction.Destinations.Contains(destination)))) return false;
                overwritten = instruction.Destinations.Any(target => target.Kind == Gen5OperandKind.ScalarRegister &&
                    (target == destination || (instruction.DestinationWidth > 1 && target.Value + 1 == destination.Value))) ||
                    instruction.Control is Gen5Vop3Control { ScalarDestination: { } scalarDestination } &&
                        destination.Value >= scalarDestination && destination.Value - scalarDestination < 2 ||
                    instruction.Control is Gen5SdwaControl { ScalarDestination: { } compareDestination } &&
                        destination.Value >= compareDestination && destination.Value - compareDestination < 2;
                if (overwritten) break;
            }
            if (!overwritten)
                foreach (var successor in flow.Successors[position.Block])
                    pending.Enqueue((successor, flow.Blocks[successor].StartPc));
        }
        return true;
    }

    private static bool ReadsScalar(Gen5ShaderInstruction instruction, uint register)
    {
        bool InRange(uint first, uint width) => register >= first && register - first < width;
        var width = instruction.Opcode.Contains("64", StringComparison.Ordinal) ? 2u : 1u;
        if (instruction.Sources.Any(source => source.Kind == Gen5OperandKind.ScalarRegister && InRange(source.Value, width)))
            return true;
        return instruction.Control switch
        {
            Gen5ImageControl image => InRange(image.ScalarResource, 8) || InRange(image.ScalarSampler, 4),
            Gen5BufferMemoryControl buffer => InRange(buffer.ScalarResource, 4),
            Gen5GlobalMemoryControl global => InRange(global.ScalarAddress, 2),
            Gen5ScalarMemoryControl scalar =>
                instruction.Sources.Count > 0 && InRange(instruction.Sources[0].Value,
                    instruction.Opcode.StartsWith("SBuffer", StringComparison.Ordinal) ? 4u : 2u) ||
                scalar.DynamicOffsetRegister == register,
            _ => false,
        };
    }
}
