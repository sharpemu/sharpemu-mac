// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;

namespace SharpEmu.ShaderCompiler.Metal;

// Wave32 row reductions (ShaderCompileRequest.SubgroupArithmetic), ported from
// Gen5SpirvTranslator.WaveReductions with the same pattern and the same proofs:
//   S_AND_B32 mask, x, entry            entry holds a subset of the entry EXEC (IsEntryExecSubset)
//   S_ORN2_SAVEEXEC_B32 saved, x        EXEC = x | ~EXEC, every lane (x holds EXEC)
//   V_CNDMASK_B32 v, identity, value, mask   (VOP3)
//   V_<op> v, v, v                      DPP row_shr:1, 2, 4 and 8, full masks, FI and BOUND_CTRL clear
//   [S_MOV_B32 exec_lo, saved]
//   V_READLANE_B32 s, v, 15 and/or V_READLANE_B32 s, v, 31
// On the PS5 every lane of the wave exists and V_CNDMASK gives each lane outside mask the identity, so lane 15 (31)
// holds the operation over row 0 (1). A fragment simdgroup holds only the fragments (and helpers) the rasterizer
// packed into it: a lane that never ran has no register to read, and a row can have none at all. Neither the
// nearest-running-lane read of a pixel V_READLANE (sharpemu_present_lane) nor the scan over the running lanes gives
// the PS5's value then. Each read is replaced by a SIMD-group reduction (MSL 6.10.2: simd_sum, simd_min, simd_max,
// simd_and, simd_or, simd_xor over the active threads, kernel and fragment functions, Apple7 and later) of the
// V_CNDMASK result over its row, the other rows taking the identity. mask selects only lanes of the entry EXEC,
// which are running threads, so a lane missing from the simdgroup takes the identity on the PS5 too and leaving it
// out changes nothing. The scan itself is emitted unchanged.
public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        private readonly record struct WaveReduction(int CndmaskIndex, uint Vector, string Operation, bool Signed, uint Identity, int Row);

        // VCC_LO: SGPRs from here on (VCC, trap registers, M0, NULL, EXEC) are written by instructions that do not name them.
        private const uint FirstSpecialScalarRegister = VccLoRegister;

        // The V_CNDMASK results kept for the reads after them, by instruction index.
        private readonly Dictionary<int, string> _waveReductionValues = new();

        // Guest lane = simdgroup lane, one guest wave32 per simdgroup: a pixel stage with real lanes, or a wave32
        // compute program. Vertex runs one lane per thread; a wave64 or a mesh wave is not one simdgroup.
        private bool WaveReductionsApply =>
            _request.SubgroupArithmetic && _waveLaneCount == 32 && HasRealWaveLanes && !IsWave64 && !IsNarrowWave64 &&
            _stage is Gen5MslStage.Pixel or Gen5MslStage.Compute;

        // The reductions of a block by instruction index: the V_CNDMASK (Row -1) and each replaced V_READLANE. Null when the
        // block has none or the stage cannot compute them.
        private Dictionary<int, WaveReduction>? FindWaveReductions(ShaderBlock block)
        {
            if (!WaveReductionsApply)
            {
                return null;
            }

            Dictionary<int, WaveReduction>? found = null;
            for (var index = block.StartIndex; index < block.EndIndex; index++)
            {
                TryMatchWaveReduction(block, index, ref found);
            }

            return found;
        }

        // After the instruction at index was emitted: keeps a V_CNDMASK result, or replaces a row read.
        private void ApplyWaveReduction(Dictionary<int, WaveReduction> reductions, int index)
        {
            if (!reductions.TryGetValue(index, out var reduction))
            {
                return;
            }

            if (reduction.Row < 0)
            {
                _waveReductionValues[reduction.CndmaskIndex] = Temp("uint", VectorExpression(reduction.Vector));
                return;
            }

            if (!_waveReductionValues.TryGetValue(reduction.CndmaskIndex, out var value))
            {
                return;
            }

            var contribution = Temp(
                "uint",
                $"select({FormatUInt(reduction.Identity)}, {value}, (sharpemu_lane >> 4u) == {reduction.Row}u)");
            var reduced = reduction.Signed
                ? $"as_type<uint>({reduction.Operation}(as_type<int>({contribution})))"
                : $"{reduction.Operation}({contribution})";
            StoreScalar(_request.Program.Instructions[index].Destinations[0].Value, Temp("uint", reduced));
        }

        // index names the S_ORN2_SAVEEXEC_B32 of a reduction.
        private void TryMatchWaveReduction(ShaderBlock block, int index, ref Dictionary<int, WaveReduction>? found)
        {
            var instructions = _request.Program.Instructions;
            var saveexec = instructions[index];
            if (saveexec.Opcode != "SOrn2SaveexecB32" ||
                saveexec.Destinations is not [{ Kind: Gen5OperandKind.ScalarRegister } saved] ||
                saveexec.Sources is not [{ Kind: Gen5OperandKind.ScalarRegister } execSource] ||
                !HoldsExec(block, index, execSource.Value))
            {
                return;
            }

            var at = index + 1;
            while (at < block.EndIndex && instructions[at].Opcode is "SWaitcnt" or "SNop") at++;
            // The VOP3 form, whose mask is a named SGPR: the VOP2 form reads VCC, which compares and carries write unnamed.
            if (at >= block.EndIndex ||
                instructions[at] is not { Opcode: "VCndmaskB32" } cndmask ||
                cndmask.Destinations is not [{ Kind: Gen5OperandKind.VectorRegister } scanned] ||
                cndmask.Sources is not [var identitySource, _, { Kind: Gen5OperandKind.ScalarRegister } mask] ||
                cndmask.Control is not Gen5Vop3Control { AbsoluteMask: 0, NegateMask: 0, OutputModifier: 0, Clamp: false } ||
                !TryGetReductionConstant(identitySource, out var identity) ||
                !SelectsEntryLanes(block, at, mask.Value))
            {
                return;
            }

            var cndmaskIndex = at;
            var vector = scanned.Value;
            string? scan = null;
            foreach (var shift in new uint[] { 1, 2, 4, 8 })
            {
                if (++at >= block.EndIndex ||
                    instructions[at] is not { Control: Gen5DppControl dpp } step ||
                    dpp is not { FetchInactive: false, BoundControl: false, AbsoluteMask: 0, NegateMask: 0, BankMask: 15, RowMask: 15 } ||
                    dpp.Control != 0x110 + shift ||
                    step.Destinations is not [{ Kind: Gen5OperandKind.VectorRegister } written] || written.Value != vector ||
                    step.Sources is not [{ Kind: Gen5OperandKind.VectorRegister } left, { Kind: Gen5OperandKind.VectorRegister } right] ||
                    left.Value != vector || right.Value != vector ||
                    (scan ??= step.Opcode) != step.Opcode)
                {
                    return;
                }
            }

            var (operation, signed, expectedIdentity) = scan switch
            {
                "VMaxU32" => ("simd_max", false, 0u),
                "VMinU32" => ("simd_min", false, uint.MaxValue),
                "VMaxI32" => ("simd_max", true, 0x8000_0000u),
                "VMinI32" => ("simd_min", true, 0x7FFF_FFFFu),
                "VOrB32" => ("simd_or", false, 0u),
                "VAndB32" => ("simd_and", false, uint.MaxValue),
                "VXorB32" => ("simd_xor", false, 0u),
                // V_ADD_NC_U32 decodes as VAddI32 in its VOP2 form, the only one DPP has (VOP3 0x125 is VAddU32): both
                // are the same wrapping add. The SPIR-V matcher names VAddU32 only, so it never sees a decoded DPP add.
                "VAddU32" or "VAddI32" => ("simd_sum", false, 0u),
                _ => ((string?)null, false, 1u),
            };
            if (operation is null || identity != expectedIdentity)
            {
                return;
            }

            at++;
            if (at < block.EndIndex && instructions[at] is { Opcode: "SMovB32" } restore &&
                restore.Destinations is [{ Kind: Gen5OperandKind.ScalarRegister, Value: ExecLoRegister }] &&
                restore.Sources is [{ Kind: Gen5OperandKind.ScalarRegister } restored] && restored.Value == saved.Value)
            {
                at++;
            }

            var reads = new List<(int Index, int Row)>();
            for (; at < block.EndIndex && reads.Count < 2; at++)
            {
                if (instructions[at] is not { Opcode: "VReadlaneB32" } readlane ||
                    readlane.Destinations is not [{ Kind: Gen5OperandKind.ScalarRegister }] ||
                    readlane.Sources.Count < 2 ||
                    readlane.Sources[0] is not { Kind: Gen5OperandKind.VectorRegister } source || source.Value != vector ||
                    !TryGetFixedReductionLane(readlane.Sources[1], out var lane) || lane is not (15 or 31) ||
                    reads.Exists(read => read.Row == (int)(lane >> 4)))
                {
                    break;
                }

                reads.Add((at, (int)(lane >> 4)));
            }

            if (reads.Count == 0)
            {
                return;
            }

            found ??= new Dictionary<int, WaveReduction>();
            found[cndmaskIndex] = new WaveReduction(cndmaskIndex, vector, operation, signed, identity, -1);
            foreach (var (readIndex, row) in reads)
            {
                found[readIndex] = new WaveReduction(cndmaskIndex, vector, operation, signed, identity, row);
            }
        }

        // Whether the scalar register holds EXEC when the instruction at index runs: it is EXEC itself, or the last
        // instruction of the block that may write EXEC before it is S_MOV_B32 exec_lo, register and nothing between them
        // writes the register.
        private bool HoldsExec(ShaderBlock block, int index, uint register)
        {
            if (register == ExecLoRegister)
            {
                return true;
            }

            if (register >= FirstSpecialScalarRegister)
            {
                return false;
            }

            var instructions = _request.Program.Instructions;
            for (var at = index - 1; at >= block.StartIndex; at--)
            {
                var instruction = instructions[at];
                if (instruction is { Opcode: "SMovB32", Sources: [{ Kind: Gen5OperandKind.ScalarRegister } source] } &&
                    instruction.Destinations is [{ Kind: Gen5OperandKind.ScalarRegister, Value: ExecLoRegister }] &&
                    source.Value == register)
                {
                    return true;
                }

                if (MayWriteExec(instruction) || MayWriteReductionScalar(instruction, register))
                {
                    return false;
                }
            }

            return false;
        }

        // Whether the SGPR mask that V_CNDMASK reads at cndmaskIndex selects only lanes of the wave's entry EXEC: its last
        // write in the block is S_AND_B32 mask, x, entry (in either order), entry holding a subset of the entry EXEC.
        private bool SelectsEntryLanes(ShaderBlock block, int cndmaskIndex, uint mask)
        {
            if (mask >= FirstSpecialScalarRegister)
            {
                return false;
            }

            var instructions = _request.Program.Instructions;
            for (var at = cndmaskIndex - 1; at >= block.StartIndex; at--)
            {
                var instruction = instructions[at];
                if (MayWriteReductionScalar(instruction, mask))
                {
                    return instruction is { Opcode: "SAndB32", Sources: [var left, var right] } &&
                        (IsEntryExecSubset(left, at) || IsEntryExecSubset(right, at));
                }
            }

            return false;
        }

        // Whether the operand is an SGPR that holds a subset of the wave's entry EXEC when the instruction at useIndex runs.
        // Its one copy S_MOV_B32 register, exec_lo comes before useIndex, in the entry block (which no branch reaches) and
        // before any EXEC write; every other write of it is S_AND_B32 or S_ANDN2_B32 of itself, which keeps a subset.
        private bool IsEntryExecSubset(Gen5Operand operand, int useIndex)
        {
            if (operand.Kind != Gen5OperandKind.ScalarRegister || operand.Value >= FirstSpecialScalarRegister)
            {
                return false;
            }

            var register = operand.Value;
            var instructions = _request.Program.Instructions;
            var copy = -1;
            var inEntryBlock = true;
            var execWritten = false;
            for (var at = 0; at < instructions.Count; at++)
            {
                var instruction = instructions[at];
                if (instruction is { Opcode: "SMovB32", Sources: [{ Kind: Gen5OperandKind.ScalarRegister, Value: ExecLoRegister }] } &&
                    instruction.Destinations is [{ Kind: Gen5OperandKind.ScalarRegister } destination] &&
                    destination.Value == register)
                {
                    if (copy >= 0 || !inEntryBlock || execWritten)
                    {
                        return false;
                    }

                    copy = at;
                }
                else if (MayWriteReductionScalar(instruction, register) && !KeepsSubset(instruction, register))
                {
                    return false;
                }

                execWritten |= MayWriteExec(instruction);
                inEntryBlock &= !IsBranch(instruction.Opcode) && instruction.Opcode is not ("SEndpgm" or "SBarrier") &&
                    instruction.Control is not ShaderCallEntryControl;
            }

            if (copy < 0 || copy >= useIndex)
            {
                return false;
            }

            // A branch into the entry block would run the copy again, after EXEC may have changed.
            for (var at = 0; at < instructions.Count; at++)
            {
                if (!IsBranch(instructions[at].Opcode) || !TryGetBranchTargetPc(instructions[at], out var target))
                {
                    continue;
                }

                for (var entry = 0; entry <= copy; entry++)
                {
                    if (instructions[entry].Pc == target)
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        // S_AND_B32 register, register, x (either order) and S_ANDN2_B32 register, register, x keep a subset of register.
        private static bool KeepsSubset(Gen5ShaderInstruction instruction, uint register) =>
            instruction.Destinations is [{ Kind: Gen5OperandKind.ScalarRegister } destination] &&
            destination.Value == register &&
            instruction switch
            {
                { Opcode: "SAndB32", Sources: [var left, var right] } => IsScalarOperand(left, register) || IsScalarOperand(right, register),
                { Opcode: "SAndn2B32", Sources: [var left, _] } => IsScalarOperand(left, register),
                _ => false,
            };

        private static bool IsScalarOperand(Gen5Operand operand, uint register) =>
            operand.Kind == Gen5OperandKind.ScalarRegister && operand.Value == register;

        // MayWriteScalar (an over-approximation: full-width destinations, a vector instruction's lane-mask pair and VCC,
        // anything relative-addressed), and, as the SPIR-V matcher has it, a linked call's instructions, which set
        // registers they do not name.
        private static bool MayWriteReductionScalar(Gen5ShaderInstruction instruction, uint register) =>
            instruction.Control is ShaderCallEntryControl or ShaderCallMatchControl or ShaderCallGroupControl or ShaderCallFaultControl ||
            MayWriteScalar(instruction, register);

        private static bool TryGetReductionConstant(Gen5Operand operand, out uint value)
        {
            value = operand.Value;
            return operand.Kind == Gen5OperandKind.LiteralConstant ||
                   operand.Kind == Gen5OperandKind.EncodedConstant && Gen5InlineConstants.TryDecode(operand.Value, out value);
        }

        // A V_READLANE lane select that is a constant (Gen5SpirvTranslator.TryGetFixedLane), in a wave32.
        private static bool TryGetFixedReductionLane(Gen5Operand operand, out uint lane)
        {
            lane = operand.Value;
            if (operand.Kind == Gen5OperandKind.EncodedConstant && operand.Value is >= 128 and <= 192)
                lane -= 128;
            else if (operand.Kind == Gen5OperandKind.EncodedConstant && operand.Value is >= 193 and <= 208)
                lane = unchecked(192u - operand.Value);
            else if (operand.Kind != Gen5OperandKind.LiteralConstant)
                return false;
            lane &= 31u;
            return true;
        }
    }
}
