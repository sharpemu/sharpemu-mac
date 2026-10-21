// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Libs.Gpu.Pipelines;

/// <summary>Compiles a recorded shader through the same shared resource plan as live submissions.</summary>
public static class ShaderPrewarmCompiler
{
    public static bool TryCompilePrewarm(ComputePrewarmRecord record, ShaderCodeCapture code, IGuestGpuBackend compiler,
        bool sharedInt64Atomics, bool execGuardElision, bool signedZeroInfNanPreserve, out IGuestCompiledShader? compiled, out BindingLayout? layout, out string error,
        bool hostWave64Supported = false, bool hostComputeSubgroup32 = false, bool imageInt64Atomics = false) =>
        ShaderProgramCache.TryCompilePrewarm(record, code, compiler, sharedInt64Atomics, execGuardElision, signedZeroInfNanPreserve, out compiled, out layout, out error,
            hostWave64Supported, hostComputeSubgroup32, imageInt64Atomics);

    /// <summary>
    /// Translates a recorded graphics program through the shared plan, layout and request builders
    /// of live draws, over the recorded guest code; the backend turns the request into a module.
    /// </summary>
    public static bool TryCompileGraphicsPrewarm(in GraphicsPrewarmList.Item item, IGuestGpuBackend compiler,
        out IGuestCompiledShader? compiled, out string error)
    {
        compiled = null;
        try
        {
            if (!ShaderCorpus.TryBuildRequest(GraphicsPrewarmList.Decode(item), out var request, out error))
            {
                return false;
            }

            if (request is null)
            {
                error = "the record has no specialization";
                return false;
            }

            return compiler.TryCompileProgram(request, out compiled, out error) && compiled is not null;
        }
        catch (Exception exception)
        {
            error = exception.Message;
            return false;
        }
    }
}
