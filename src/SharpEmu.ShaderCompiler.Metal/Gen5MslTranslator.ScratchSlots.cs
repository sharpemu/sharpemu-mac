// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal;

public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        private bool _staticScratchSlots;

        // Renaming a private allocation by a per-invocation cyclic permutation is
        // unobservable. For base v0 = local_id.x and dword-aligned constant offsets,
        // ((v0 + offset) >> 2) % N = (v0 / 4 + offset / 4) % N, provided the add
        // cannot wrap. Removing the common rotation leaves constant array indices,
        // allowing Metal to scalarize guest register spills. No other access may
        // observe the allocation, and v0 must retain its entry value at every access.
        private bool CanUseStaticScratchSlots()
        {
            bool Refuse(string reason)
            {
                if (_options.StaticScratchSlots && Environment.GetEnvironmentVariable("SHARPEMU_MSL_STATIC_SCRATCH_REPORT") == "1")
                    Console.Error.WriteLine($"[MSL-SCRATCH] hash={_request.Hash:X16} refused={reason}");
                return false;
            }

            if (!_options.StaticScratchSlots || _stage != Gen5MslStage.Compute ||
                _request.Tessellation is not null || _scratchDwordCount == 0 || _localSizeX == 0 ||
                _request.Memory.Entries.Any(static m =>
                    m.AddressSpace is FlatAddressSpace.Private or FlatAddressSpace.SharedOrPrivate))
            {
                return Refuse("stage-or-aperture");
            }

            var instructions = _request.Program.Instructions;
            var last = -1;
            for (var index = 0; index < instructions.Count; index++)
            {
                var instruction = instructions[index];
                if (!instruction.Opcode.StartsWith("Scratch", StringComparison.Ordinal)) continue;
                if (instruction.Opcode is not ("ScratchLoadDword" or "ScratchLoadDwordx2" or
                    "ScratchLoadDwordx3" or "ScratchLoadDwordx4" or "ScratchStoreDword" or
                    "ScratchStoreDwordx2" or "ScratchStoreDwordx3" or "ScratchStoreDwordx4") ||
                    instruction.Control is not Gen5GlobalMemoryControl control ||
                    control.ScalarAddress < 125 || control.VectorAddress != 0 ||
                    control.DynamicOffsetRegister is not null || control.AddThreadId ||
                    control.OffsetBytes < 0 || (control.OffsetBytes & 3) != 0 ||
                    control.DwordCount is < 1 or > 4 ||
                    (ulong)control.OffsetBytes + (control.DwordCount - 1) * 4 + _localSizeX - 1 > uint.MaxValue)
                {
                    return Refuse($"address pc={instruction.Pc:X} op={instruction.Opcode} control={instruction.Control}");
                }
                last = index;
            }
            if (last < 0) return false;

            for (var index = 0; index < instructions.Count; index++)
            {
                var instruction = instructions[index];
                // V_MOVRELS reads a relative source but writes its explicit destination;
                // the destination check below still rejects a write to v0.
                // Unknown/indirect control flow or register writes cannot prove the
                // lifetime of the base. A later ordinary write is fine only when no
                // path from that suffix can return to an access.
                if (instruction.Opcode.Contains("Raw", StringComparison.Ordinal) ||
                    (instruction.Opcode.StartsWith("VMovrel", StringComparison.Ordinal) && instruction.Opcode != "VMovrelsB32") ||
                    instruction.Opcode is "VSwaprelB32" or "SSetpcB64" or "SSwappcB64" or "SRfeB64" or
                        "SCbranchJoin" or "SCbranchGFork" or "SCbranchIFork") return Refuse($"dynamic pc={instruction.Pc:X} op={instruction.Opcode}");
                if (index <= last && instruction.Destinations.Any(static d =>
                    d.Kind == Gen5OperandKind.VectorRegister && d.Value == 0)) return Refuse($"base-write pc={instruction.Pc:X} op={instruction.Opcode}");
                if (index > last && Ir.Gen5IrBranchResolver.Instance.TryGetBranchTarget(instruction, out var target) &&
                    target <= instructions[last].Pc) return Refuse($"backedge pc={instruction.Pc:X} target={target:X} last={instructions[last].Pc:X}");
            }
            return true;
        }

        private string ScratchAccessWord(Gen5GlobalMemoryControl control, string byteAddress, uint offsetBytes = 0) =>
            _staticScratchSlots
                ? $"sharpemu_scratch[{((uint)control.OffsetBytes / 4 + offsetBytes / 4) % _scratchDwordCount}u]"
                : ScratchWord(byteAddress, offsetBytes);
    }
}
