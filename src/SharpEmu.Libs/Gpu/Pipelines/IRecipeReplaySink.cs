// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Libs.Gpu.Pipelines;

// Present only on an opted-in backend. Decisions remain on the ordered consumer.
public interface IRecipeReplaySink
{
    void ProgramReplay(bool hit, bool relocated, ResourceDependencyStatus reason, long validateTicks, long savedTicks);
    void PipelineReplay(bool hit);
}
