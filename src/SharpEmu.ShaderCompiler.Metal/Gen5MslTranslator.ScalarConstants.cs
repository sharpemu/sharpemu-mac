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
            if (!_options.ScalarConstantBuffers || !_options.RawDwordAccess ||
                _stage switch
                {
                    Gen5MslStage.Mesh => false,
                    Gen5MslStage.Vertex or Gen5MslStage.Pixel => !_options.ScalarConstantBuffersGraphicsStages,
                    _ => !_options.ScalarConstantBuffersAllStages,
                })
                return false;

            // SMEM coherence is carried by the instruction control, not consistently
            // copied into MemoryAccessInfo. Do not infer it from the resource table alone.
            if (_request.Program.Instructions.Any(instruction => instruction.Control is
                Gen5ScalarMemoryControl { Glc: true } or Gen5ScalarMemoryControl { Dlc: true }))
                return false;

            // A compute shader is held to stricter rules than a graphics one: its waves can hand
            // each other data through device memory within the dispatch, so a store through a
            // buffer, global or flat address refuses it. Image stores are let through: a scalar
            // load does not read a texel back. A compute shader that drops its scalar cache
            // (S_DCACHE_INV, S_GL1_INV) expects to see memory change under it, from this dispatch
            // or another one, so it is refused whether or not it writes.
            if (_stage == Gen5MslStage.Compute &&
                (_request.Program.Instructions.Any(instruction => instruction.Opcode is "SDcacheInv" or "SGl1Inv") ||
                 _request.Memory.Entries.Any(entry => entry.Access is MemoryAccess.Write or MemoryAccess.Atomic &&
                     entry.Kind is not (MemoryResourceKind.LocalDataShare or MemoryResourceKind.Scratch or MemoryResourceKind.Image))))
                return false;

            // A store through another descriptor, a flat address or an image can alias
            // this binding, but a scalar load without GLC/DLC does not see such a store on the
            // guest either. ScalarConstantBuffersBesideWrites off refuses the entire shader when
            // any device resource is written. LDS and private scratch cannot alias a host-bound buffer.
            // A shader that invalidates its scalar cache (S_DCACHE_INV, S_GL1_INV) may read its own
            // stores back through scalar loads, which a constant binding would not see.
            if (_request.Memory.Entries.Any(entry => entry.Access is MemoryAccess.Write or MemoryAccess.Atomic &&
                    entry.Kind is not (MemoryResourceKind.LocalDataShare or MemoryResourceKind.Scratch)) &&
                (!_options.ScalarConstantBuffersBesideWrites ||
                 _request.Program.Instructions.Any(instruction => instruction.Opcode is "SDcacheInv" or "SGl1Inv")))
                return false;

            var uses = _request.Memory.Entries.Where(entry => !entry.PlanningOnly &&
                entry.Resource == (uint)binding && entry.Kind is
                    (MemoryResourceKind.ScalarBuffer or MemoryResourceKind.Buffer or MemoryResourceKind.Global or MemoryResourceKind.Flat)).ToArray();
            return uses.Length != 0 && uses.All(entry => entry.Kind == MemoryResourceKind.ScalarBuffer &&
                entry.Access == MemoryAccess.Read && !entry.DeviceDescriptor && !entry.Glc && !entry.Slc);
        }
    }
}
