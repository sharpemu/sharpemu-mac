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
        bool hostWave64Supported = false, bool hostComputeSubgroup32 = false, bool imageInt64Atomics = false, bool tableBlocks = false,
        Func<ulong, IShaderFrontend?>? frontendOf = null, bool runtimeImageSwizzles = false,
        StableImageDescriptorLimits stableImageLimits = default) =>
        ShaderProgramCache.TryCompilePrewarm(record, code, compiler, sharedInt64Atomics, execGuardElision, signedZeroInfNanPreserve, out compiled, out layout, out error,
            hostWave64Supported, hostComputeSubgroup32, imageInt64Atomics, stableImageLimits: stableImageLimits, tableBlocks: tableBlocks,
            frontendOf: frontendOf, runtimeImageSwizzles: runtimeImageSwizzles);

    /// <summary>
    /// Translates a recorded graphics program through the shared plan, layout and request builders
    /// of live draws, over the recorded guest code; the backend turns the request into a module.
    /// </summary>
    public static bool TryCompileGraphicsPrewarm(in GraphicsPrewarmList.Item item, IGuestGpuBackend compiler,
        out IGuestCompiledShader? compiled, out string error) =>
        TryCompileGraphicsPrewarm(item, compiler, null, out compiled, out error, out _);

    /// <summary>
    /// As above; with <paramref name="imageLimits"/> (the stable image budget the host answers now), a record whose
    /// session answered another budget is not translated (<paramref name="stale"/>): its specialization follows the
    /// other image binding policy, which this host's sessions no longer ask for.
    /// </summary>
    public static bool TryCompileGraphicsPrewarm(in GraphicsPrewarmList.Item item, IGuestGpuBackend compiler,
        StableImageDescriptorLimits? imageLimits, out IGuestCompiledShader? compiled, out string error, out bool stale)
    {
        compiled = null;
        stale = false;
        try
        {
            var record = GraphicsPrewarmList.Decode(item);
            if (imageLimits is { } limits &&
                (record.Host.GraphicsImageLimits != limits || record.Host.ComputeImageLimits != limits))
            {
                stale = true;
                error = $"recorded under another stable image budget ({record.Host.GraphicsImageLimits}, now {limits})";
                return false;
            }

            if (!ShaderCorpus.TryBuildRequest(record, out var request, out error))
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
