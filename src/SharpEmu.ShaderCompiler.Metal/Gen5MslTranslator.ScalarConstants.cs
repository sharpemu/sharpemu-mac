// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal;

public static partial class Gen5MslTranslator
{
    private sealed partial class CompilationContext
    {
        private bool IsScalarConstantBuffer(int binding)
        {
            if (!_options.ScalarConstantBuffers || !_options.RawDwordAccess || _stage != Gen5MslStage.Mesh)
                return false;

            // SMEM coherence is carried by the instruction control, not consistently
            // copied into MemoryAccessInfo. Do not infer it from the resource table alone.
            if (_request.Program.Instructions.Any(instruction => instruction.Control is
                Gen5ScalarMemoryControl { Glc: true } or Gen5ScalarMemoryControl { Dlc: true }))
                return false;

            // A store through another descriptor, a flat address or an image can alias
            // this binding. Refuse the entire shader when any device resource is written.
            // LDS and private scratch cannot alias a host-bound buffer.
            if (_request.Memory.Entries.Any(entry => entry.Access is MemoryAccess.Write or MemoryAccess.Atomic &&
                entry.Kind is not (MemoryResourceKind.LocalDataShare or MemoryResourceKind.Scratch)))
                return false;

            var uses = _request.Memory.Entries.Where(entry => !entry.PlanningOnly &&
                entry.Resource == (uint)binding && entry.Kind is
                    (MemoryResourceKind.ScalarBuffer or MemoryResourceKind.Buffer or MemoryResourceKind.Global or MemoryResourceKind.Flat)).ToArray();
            return uses.Length != 0 && uses.All(entry => entry.Kind == MemoryResourceKind.ScalarBuffer &&
                entry.Access == MemoryAccess.Read && !entry.DeviceDescriptor && !entry.Glc && !entry.Slc);
        }
    }
}
