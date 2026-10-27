// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal;

public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        // One hardware wave only: every lane reaches the shuffle even when guest EXEC
        // is partial. Other workgroups cannot address this group's LDS. Refuse aliases
        // whose writes do not pass through the DS instruction emitter.
        private bool UseMeshLdsForwarding => _options.MeshLdsForwarding &&
            _stage == Gen5MslStage.Mesh && Mesh.ThreadsPerThreadgroup == 32 &&
            !_request.Memory.Entries.Any(entry =>
                entry.AddressSpace is FlatAddressSpace.Shared or FlatAddressSpace.SharedOrPrivate) &&
            !_request.Program.Instructions.Any(instruction => instruction.Control is Gen5BufferMemoryControl { Lds: true });

        private void RememberMeshLdsWrite(Gen5ShaderInstruction instruction, string address, uint offset)
        {
            if (!UseMeshLdsForwarding || instruction.Opcode != "DsWriteB128" ||
                (offset & 3u) != 0 || _meshLdsDwords < 128 || offset > _meshLdsDwords * 4u - 512u) return;

            // Prove the active writers own disjoint records, before forwarding anything.
            // A shader using another layout still performs its original memory accesses.
            var affine = Temp("bool", $"simd_all(!exec || {address} == sharpemu_lane * 16u)");
            Line($"sharpemu_lds_forward_valid = exec && {affine};");
            Line($"sharpemu_lds_forward_base = {offset}u;");
            Line($"sharpemu_lds_forward_values = uint4({RawSource(instruction, 1)}, {RawSource(instruction, 2)}, " +
                $"{RawSource(instruction, 3)}, {RawSource(instruction, 4)});");
        }

        private bool TryForwardMeshLdsRead(Gen5ShaderInstruction instruction, string address, uint offset)
        {
            if (!UseMeshLdsForwarding || instruction.Opcode != "DsReadB128") return false;
            var distance = Temp("uint", $"({address} + {offset}u) - sharpemu_lds_forward_base");
            var lane = Temp("uint", $"({distance} >> 4u) & 31u");
            var valid = Temp("bool", $"simd_shuffle((uint)sharpemu_lds_forward_valid, {lane}) != 0u && " +
                $"{distance} < 512u && ({distance} & 15u) == 0u");
            var values = Temp("uint4", $"simd_shuffle(sharpemu_lds_forward_values, {lane})");
            for (uint component = 0; component < 4; component++)
                StoreVector(instruction.Destinations[(int)component].Value,
                    $"{valid} ? {values}[{component}] : sharpemu_lds[{LdsIndex(address, offset + component * 4)}]");
            return true;
        }
    }
}
