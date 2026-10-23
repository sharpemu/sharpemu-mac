// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Resources;

public static class VertexOutputPruning
{
    public static bool Enabled { get; set; }

    // Caller proves no parameter exports are consumed. Preserve PCs, control flow,
    // side effects, and every position (including clip/cull/layer/viewport) export.
    // The native compiler can eliminate the now-unused pure arithmetic chains.
    public static Gen5ShaderProgram PositionOnly(Gen5ShaderProgram program) => new(program.Address,
        program.Instructions.Select(instruction =>
            instruction.Control is Gen5ExportControl { Target: >= 32 and < 64 }
                ? instruction with { Encoding = Gen5ShaderEncoding.Sopp, Opcode = "SNop", Sources = [], Destinations = [], Control = null }
                : instruction).ToArray());
}
