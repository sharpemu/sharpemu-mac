// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal;

// Shared call bodies, the MSL form of upstream's ShareLinkedCallBodies (Gen5SpirvTranslator.SharedCalls.cs). A program
// linked per record (Gen5ShaderCallLinker without run-time records) holds one copy of a callee for each record of a call
// site's table: the copies are the same instructions and differ only in the record's argument, and so in the descriptors
// the body loads through it (the plan's memory records at each copy's pcs). Inlined, the MSL source and the Metal compile
// grow with records x callee size. Here the module keeps the last copy of each callee at each site: every record's test
// and entry stay, the entry stores the record's slot (sharpemu_call_slot), and the body of every other copy becomes a
// branch into the kept one, forward, so the program gains no back edge. An instruction of the kept body whose memory
// records differ between the records is emitted once for each distinct set, selected by the slot, with the memory
// records of that record's copy; every other instruction is emitted once. The resource plan still holds every copy, so
// bindings do not change.
//
// What is not ported: upstream's site lookup (one test per callee instead of per record) and its DontInline functions
// that let one body serve several sites (compute, wave32). MSL has no goto and no lambdas, and a translated body reads
// and writes the kernel's own locals (the registers this program touches, EXEC, VCC, SCC, the lane and LDS state), so a
// body outlined into a device function would take that whole state by reference; Metal inlines device functions anyway.
// A site's records share one body, which is what grows with the table.
//
// SHARPEMU_METAL_SHARED_CALL_BODIES=0 (Gen5MslCodegenOptions.SharedCallBodies) emits every copy inline, as before.
public static partial class Gen5MslTranslator
{
    internal sealed class SharedCallBodies
    {
        // Each sharing record's entry pc and its slot in its body's records.
        public required Dictionary<uint, uint> SlotByEntryPc { get; init; }

        // The kept copy of each shared body: its pc range and every record's body start, in slot order.
        public required List<(uint Start, uint End, uint[] Bases)> Bodies { get; init; }

        // The pc of an instruction of a kept body in each record's copy, by slot.
        public bool TryGetRecordPcs(uint pc, out uint[] pcs)
        {
            var low = 0;
            var high = Bodies.Count - 1;
            while (low <= high)
            {
                var middle = low + ((high - low) >> 1);
                var body = Bodies[middle];
                if (pc < body.Start)
                {
                    high = middle - 1;
                }
                else if (pc >= body.End)
                {
                    low = middle + 1;
                }
                else
                {
                    pcs = new uint[body.Bases.Length];
                    for (var slot = 0; slot < pcs.Length; slot++)
                    {
                        pcs[slot] = body.Bases[slot] + (pc - body.Start);
                    }

                    return true;
                }
            }

            pcs = [];
            return false;
        }
    }

    // The request with each site's copies of one callee sharing the last copy's body, or the request itself when no two
    // copies can share.
    internal static ShaderCompileRequest ShareLinkedCallBodies(ShaderCompileRequest request, out SharedCallBodies? shared)
    {
        shared = null;
        var instructions = request.Program.Instructions;
        if (!instructions.Any(static instruction => instruction.Control is ShaderCallEntryControl))
        {
            return request;
        }

        // A body runs from its entry to the next record's test or to the site's fault.
        var bodies = new List<(int Entry, int Start, int End, int Site, ulong Target)>();
        var site = 0;
        ulong target = 0;
        for (var index = 0; index < instructions.Count; index++)
        {
            var control = instructions[index].Control;
            if (control is ShaderCallMatchControl match)
            {
                target = match.Address;
            }
            else if (control is ShaderCallFaultControl)
            {
                site++;
            }
            else if (control is ShaderCallEntryControl)
            {
                var end = index + 1;
                while (end < instructions.Count && instructions[end].Control is not (ShaderCallMatchControl or ShaderCallFaultControl))
                {
                    end++;
                }

                if (end < instructions.Count && end > index + 1)
                {
                    bodies.Add((index, index + 1, end, site, target));
                }

                index = end - 1;
            }
        }

        var slotByEntryPc = new Dictionary<uint, uint>();
        var shares = new List<(uint Start, uint End, uint[] Bases)>();
        var replaced = new Dictionary<int, (int End, uint Target)>();
        foreach (var group in bodies.GroupBy(static body => (body.Site, body.Target)))
        {
            var kept = group.Last();
            var members = group.Where(member => member == kept || SameBody(instructions, kept.Start, kept.End, member.Start, member.End)).ToList();
            if (members.Count < 2)
            {
                continue;
            }

            var start = instructions[kept.Start].Pc;
            for (var slot = 0; slot < members.Count; slot++)
            {
                var member = members[slot];
                slotByEntryPc.Add(instructions[member.Entry].Pc, (uint)slot);
                if (member != kept)
                {
                    replaced.Add(member.Start, (member.End, start));
                }
            }

            shares.Add((start, instructions[kept.End].Pc, members.Select(member => instructions[member.Start].Pc).ToArray()));
        }

        if (shares.Count == 0)
        {
            return request;
        }

        var program = new List<Gen5ShaderInstruction>(instructions.Count);
        for (var index = 0; index < instructions.Count; index++)
        {
            if (replaced.TryGetValue(index, out var copy))
            {
                program.Add(new Gen5ShaderInstruction(instructions[index].Pc, Gen5ShaderEncoding.Sopp, "SBranch", [0xBF820000], [], [],
                    new ShaderLinkedBranchControl(copy.Target)));
                index = copy.End - 1;
                continue;
            }

            program.Add(instructions[index]);
        }

        shares.Sort(static (left, right) => left.Start.CompareTo(right.Start));
        shared = new SharedCallBodies { SlotByEntryPc = slotByEntryPc, Bodies = shares };
        return request.WithProgram(request.Program with { Instructions = program });
    }

    // The same instructions at the same offsets; a branch inside the body keeps its offset, one leaving it its target.
    private static bool SameBody(IReadOnlyList<Gen5ShaderInstruction> instructions, int firstStart, int firstEnd, int copyStart, int copyEnd)
    {
        if (firstEnd - firstStart != copyEnd - copyStart)
        {
            return false;
        }

        var firstPc = instructions[firstStart].Pc;
        var firstEndPc = instructions[firstEnd].Pc;
        var copyPc = instructions[copyStart].Pc;
        for (var offset = 0; offset < firstEnd - firstStart; offset++)
        {
            var left = instructions[firstStart + offset];
            var right = instructions[copyStart + offset];
            if (left.Pc - firstPc != right.Pc - copyPc || left.Opcode != right.Opcode || left.Encoding != right.Encoding ||
                left.AddressOffset != right.AddressOffset || !left.Words.SequenceEqual(right.Words) ||
                !left.Sources.SequenceEqual(right.Sources) || !left.Destinations.SequenceEqual(right.Destinations))
            {
                return false;
            }

            if (left.Control is ShaderLinkedBranchControl leftBranch && right.Control is ShaderLinkedBranchControl rightBranch)
            {
                var inside = leftBranch.TargetPc >= firstPc && leftBranch.TargetPc < firstEndPc;
                if (inside
                    ? rightBranch.TargetPc - copyPc != leftBranch.TargetPc - firstPc
                    : rightBranch.TargetPc != leftBranch.TargetPc)
                {
                    return false;
                }
            }
            else if (!Equals(left.Control, right.Control))
            {
                return false;
            }
        }

        return true;
    }

    private sealed partial class CompilationContext
    {
        private const string CallSlotName = "sharpemu_call_slot";

        // While a record's variant of a shared body's instruction is emitted: the pc of that instruction in the record's
        // copy, whose memory records it reads (MemoryPc).
        private uint? _recordMemoryPc;

        // While a shared body's instruction is emitted for its records: the emission does not divide it again.
        private bool _emittingCallRecord;

        // A shared body's instruction that reads memory records, emitted for its records (TryEmitSharedCallInstruction).
        private bool IsSharedCallMemoryInstruction(Gen5ShaderInstruction instruction, out uint[] pcs)
        {
            pcs = [];
            return _sharedCalls is { } shared && !_emittingCallRecord && _request.Memory.TryGetIndex(instruction.Pc, 0, out _) &&
                shared.TryGetRecordPcs(instruction.Pc, out pcs);
        }

        // The pc whose memory records the instruction at pc reads: its record's copy while a record variant is emitted.
        private uint MemoryPc(uint pc) => _recordMemoryPc ?? pc;

        private void EmitCallSlot(Gen5ShaderInstruction entry)
        {
            if (_sharedCalls is { } shared && shared.SlotByEntryPc.TryGetValue(entry.Pc, out var slot))
            {
                Line($"{CallSlotName} = {slot.ToString(CultureInfo.InvariantCulture)}u;");
            }
        }

        // An instruction of a shared body that reads memory records: once when every record reads the same ones, else once
        // per distinct set, under a test of the slot, with the memory records of the set's first record.
        private bool TryEmitSharedCallInstruction(Gen5ShaderInstruction instruction, uint[] pcs, out string error)
        {
            error = string.Empty;
            var signatures = pcs.Select(CallRecordSignature).ToArray();
            _emittingCallRecord = true;
            try
            {
                return signatures.All(signature => signature == signatures[0])
                    ? TryEmitInstruction(instruction, out error)
                    : TryEmitRecordVariants(instruction, pcs, signatures, out error);
            }
            finally
            {
                _emittingCallRecord = false;
            }
        }

        private bool TryEmitRecordVariants(Gen5ShaderInstruction instruction, uint[] pcs, string[] signatures, out string error)
        {
            error = string.Empty;

            var first = true;
            foreach (var group in Enumerable.Range(0, pcs.Length).GroupBy(slot => signatures[slot]))
            {
                var condition = string.Join(" || ", group.Select(slot => $"{CallSlotName} == {slot.ToString(CultureInfo.InvariantCulture)}u"));
                Line($"{(first ? "" : "else ")}if ({condition})");
                Line("{");
                _indent++;
                _recordMemoryPc = pcs[group.First()];
                try
                {
                    if (!TryEmitInstruction(instruction, out error))
                    {
                        return false;
                    }
                }
                finally
                {
                    _recordMemoryPc = null;
                    _indent--;
                }

                Line("}");
                first = false;
            }

            return true;
        }

        // What decides the code of a memory instruction at pc: its memory records and the resources they bind (as
        // upstream's CallRecordSignature). Two records with the same signature emit the same code.
        private string CallRecordSignature(uint pc)
        {
            var request = _request;
            var text = new System.Text.StringBuilder();
            for (uint component = 0; request.Memory.TryGetIndex(pc, component, out var index); component++)
            {
                var entry = request.Memory[index];
                text.Append(entry.Kind).Append(',').Append(entry.Access).Append(',').Append(entry.Offset).Append(',').Append(entry.Dmask)
                    .Append(',').Append(entry.DataDwords).Append(',').Append(entry.DataBits).Append(',').Append(entry.ComponentCount)
                    .Append(',').Append(entry.DataFormat).Append(',').Append(entry.NumberFormat).Append(',').Append(entry.ImageSampleFlags)
                    .Append(',').Append(entry.ImageDimension).Append(',').Append(entry.ImageClass).Append(',').Append(entry.NeedsSampler)
                    .Append(',').Append(entry.AddressIsFull).Append(',').Append(entry.DataSigned).Append(',').Append(entry.Typed)
                    .Append(',').Append(entry.Formatted).Append(',').Append(entry.ImageHasMip).Append(',').Append(entry.ImageR128)
                    .Append(',').Append(entry.Glc).Append(',').Append(entry.Slc).Append(',').Append(entry.IndexEnabled)
                    .Append(',').Append(entry.OffsetEnabled).Append(',').Append(entry.PlanningOnly).Append(',').Append(entry.DeviceDescriptor)
                    .Append(',').Append(entry.AddressSpace).Append(',').Append(entry.BufferDescriptor?.Provenance)
                    .Append(',').Append(entry.BufferDescriptor?.StaticResource)
                    // The resource itself: this translator binds each buffer and image by its own index.
                    .Append(",res=").Append(entry.Resource);
                var sampler = request.Resources.SamplerByMemoryIndex.TryGetValue(index, out var mappedSampler) ? mappedSampler : entry.Sampler;
                text.Append(",smp=").Append(sampler);
                if (request.CallRecordImageByMemoryIndex.TryGetValue(index, out var recordImage))
                    text.Append(",record=").Append(recordImage.Field).Append(',').Append(recordImage.Slot);
                if (request.CallRecordSamplerByMemoryIndex.TryGetValue(index, out var recordSampler))
                    text.Append(",recordsmp=").Append(recordSampler.Field).Append(',').Append(recordSampler.Slot);
                if (request.FlattenedSlotByMemoryIndex.TryGetValue(index, out var flattened)) text.Append(",flat=").Append(flattened);
                // A key read stores its value for a later image access of the same record, so each record keeps its own.
                if (request.IndirectKeyMemoryIndices.Contains(index)) text.Append(",key=").Append(index);
                if (request.IndirectOffsetKeyMemoryIndices.Contains(index)) text.Append(",offsetkey=").Append(index);
                if (request.IndirectRootByMemoryIndex.TryGetValue(index, out var root)) text.Append(",root=").Append(root);
                if (request.BufferCandidateTableByMemoryIndex.TryGetValue(index, out var table)) text.Append(",table=").Append(table);
                if (request.WrittenRangeSlotByMemoryIndex.TryGetValue(index, out var written)) text.Append(",written=").Append(written);
                text.Append(';');
            }

            return text.ToString();
        }
    }
}
