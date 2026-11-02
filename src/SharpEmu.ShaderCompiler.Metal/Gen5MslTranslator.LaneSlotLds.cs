// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal;

// Lane-slot LDS for the graphics stages.
//
// A pixel or vertex invocation has no workgroup to share LDS with, so it gets a private
// array of PrivateLdsDwordCount dwords: 8 KB per fragment, on the stack. The guest compiler
// reaches for LDS in those stages to spill VGPRs: DS_WRITE_ADDTID_B32 and
// DS_READ_ADDTID_B32 with M0 set to a constant just before, so each lane only ever touches
// its own dword of every 64-lane row. When every LDS access in a program is one of those,
// each distinct M0 + offset is one slot of the invocation's own, and the array shrinks to
// one dword per slot with constant indices, which the Metal compiler keeps in registers.
// Black Myth: Wukong's heaviest pixel shader spills five VGPRs that way and was about 40%
// of the GPU frame with the 8 KB array.
public static partial class Gen5MslTranslator
{
    // SHARPEMU_MSL_LANE_SLOT_LDS=0 keeps the full per-invocation array for every program.
    private static readonly bool LaneSlotLds =
        Environment.GetEnvironmentVariable("SHARPEMU_MSL_LANE_SLOT_LDS") != "0";

    private sealed partial class CompilationContext
    {
        /// <summary>The slot each ADDTID access uses, by PC, when the program's LDS is
        /// lane slots; null when it keeps the per-invocation array.</summary>
        private Dictionary<uint, int>? _ldsLaneSlots;

        private int _ldsLaneSlotCount;

        private void AssignLaneSlotLds(IReadOnlyList<ShaderBlock> blocks)
        {
            if (!LaneSlotLds || !_usesLds || _stage is Gen5MslStage.Compute or Gen5MslStage.Mesh)
            {
                return;
            }

            foreach (var memory in _request.Memory.Entries)
            {
                if (memory.AddressSpace is FlatAddressSpace.Shared or FlatAddressSpace.SharedOrPrivate)
                {
                    return;
                }
            }

            var instructions = _request.Program.Instructions;
            if (M0AtBlockEntries(blocks, instructions) is not { } entries)
            {
                return;
            }

            var slotsByAddress = new Dictionary<uint, int>();
            var slotsByPc = new Dictionary<uint, int>();
            for (var blockIndex = 0; blockIndex < blocks.Count; blockIndex++)
            {
                var block = blocks[blockIndex];
                uint? m0 = entries[blockIndex];
                for (var index = block.StartIndex; index < block.EndIndex; index++)
                {
                    var instruction = instructions[index];
                    if (instruction.Control is Gen5DataShareControl { Gds: false } control &&
                        instruction.Opcode is not ("DsSwizzleB32" or "DsBpermuteB32"))
                    {
                        if (instruction.Opcode is not ("DsWriteAddtidB32" or "DsReadAddtidB32") ||
                            m0 is not { } known ||
                            instruction.Sources is not [{ Kind: Gen5OperandKind.ScalarRegister, Value: M0Register }, ..])
                        {
                            return;
                        }

                        var address = (known & 0xFFFFu) + control.SingleOffsetBytes;
                        if (!slotsByAddress.TryGetValue(address, out var slot))
                        {
                            slot = slotsByAddress.Count;
                            slotsByAddress.Add(address, slot);
                        }

                        slotsByPc[instruction.Pc] = slot;
                    }

                    if (WritesM0(instruction))
                    {
                        m0 = ConstantM0(instruction);
                    }
                }
            }

            if (slotsByPc.Count == 0)
            {
                return;
            }

            _ldsLaneSlots = slotsByPc;
            _ldsLaneSlotCount = slotsByAddress.Count;
        }

        /// <summary>
        /// M0 on entry to each block where every path into it set the same constant, else null; null for the whole
        /// program when a block leaves by a jump this cannot follow. The guest compiler sets M0 once before a run of
        /// ADDTID spills and keeps it across the branches between them (Wukong's 8 KB-array pixel programs set
        /// it in one block and spill in the next), so a constant written in the same block is not enough.
        /// </summary>
        private static uint?[]? M0AtBlockEntries(IReadOnlyList<ShaderBlock> blocks, IReadOnlyList<Gen5ShaderInstruction> instructions)
        {
            var entries = new uint?[blocks.Count];
            var reached = new bool[blocks.Count];
            var queued = new bool[blocks.Count];
            var work = new Queue<int>();
            if (blocks.Count == 0)
            {
                return entries;
            }

            reached[0] = true;
            work.Enqueue(0);
            queued[0] = true;
            Span<int> successors = stackalloc int[2];
            while (work.TryDequeue(out var blockIndex))
            {
                queued[blockIndex] = false;
                var block = blocks[blockIndex];
                var m0 = entries[blockIndex];
                for (var index = block.StartIndex; index < block.EndIndex; index++)
                {
                    if (WritesM0(instructions[index]))
                    {
                        m0 = ConstantM0(instructions[index]);
                    }
                }

                var last = instructions[block.EndIndex - 1];
                var count = 0;
                if (last.Opcode is "SSetpcB64" or "SSwappcB64" or "SRfeB64" or "SCbranchJoin" or "SCbranchGFork" or "SCbranchIFork")
                {
                    return null;
                }

                if (last.Opcode == "SEndpgm")
                {
                }
                else if (last.Opcode == "SBranch" || last.Opcode.StartsWith("SCbranch", StringComparison.Ordinal))
                {
                    if (!IsExitBranchTarget(instructions, last))
                    {
                        if (!TryGetBranchTargetBlock(blocks, last, out var target))
                        {
                            return null;
                        }

                        successors[count++] = target;
                    }

                    if (last.Opcode != "SBranch" && blockIndex + 1 < blocks.Count)
                    {
                        successors[count++] = blockIndex + 1;
                    }
                }
                else if (blockIndex + 1 < blocks.Count)
                {
                    successors[count++] = blockIndex + 1;
                }

                foreach (var successor in successors[..count])
                {
                    var merged = reached[successor] ? (entries[successor] == m0 ? m0 : null) : m0;
                    if (reached[successor] && merged == entries[successor])
                    {
                        continue;
                    }

                    reached[successor] = true;
                    entries[successor] = merged;
                    if (!queued[successor])
                    {
                        queued[successor] = true;
                        work.Enqueue(successor);
                    }
                }
            }

            return entries;
        }

        private static bool WritesM0(Gen5ShaderInstruction instruction)
        {
            foreach (var destination in instruction.Destinations)
            {
                if (destination is { Kind: Gen5OperandKind.ScalarRegister, Value: M0Register })
                {
                    return true;
                }
            }

            return false;
        }

        private static uint? ConstantM0(Gen5ShaderInstruction instruction)
        {
            if (instruction.Destinations.Count != 1)
            {
                return null;
            }

            switch (instruction.Opcode)
            {
                case "SMovkI32":
                    return unchecked((uint)(short)(instruction.Words[0] & 0xFFFF));
                case "SMovB32" when instruction.Sources.Count == 1:
                    var source = instruction.Sources[0];
                    if (source.Kind == Gen5OperandKind.LiteralConstant)
                    {
                        return source.Value;
                    }

                    return source.Kind == Gen5OperandKind.EncodedConstant &&
                        Gen5InlineConstants.TryDecode(source.Value, out var constant)
                            ? constant
                            : null;
                default:
                    return null;
            }
        }

        /// <summary>The array element an ADDTID access names in lane-slot mode, or null
        /// when the program keeps the per-invocation array.</summary>
        private string? LaneSlotLdsElement(Gen5ShaderInstruction instruction) =>
            _ldsLaneSlots is { } slots && slots.TryGetValue(instruction.Pc, out var slot)
                ? $"sharpemu_lds[{slot}u]"
                : null;
    }
}
