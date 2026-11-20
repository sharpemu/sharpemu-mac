// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.VideoOut;
using SharpEmu.Logging;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// The game's two shader disk caches, beside the rest of its Metal shader data
/// (pipeline_cache/&lt;title&gt;/metal-v2, where the prewarm list and the binary archive live): the
/// translations (TranslationDiskCache, <c>translations/</c>), so a later session takes a program's MSL
/// and reflection from disk instead of translating it again, and the scalar graphs
/// (ShaderAnalysisCache, <c>analysis/</c>). Opened by the process that translates, where the
/// presenter starts: the GPU process with the split on, the one process with it off.
/// SHARPEMU_TRANSLATION_CACHE=0 turns the translations off; SHARPEMU_SHADER_ANALYSIS_CACHE_PATH
/// names another folder for the graphs ("0" turns them off).
/// </summary>
internal static class MetalTranslationCache
{
    /// <summary>
    /// What the payload of a translation depends on beyond what TranslationDiskCache keys itself
    /// (the builds of SharpEmu.ShaderCompiler and SharpEmu.Libs, the request, the SHARPEMU_
    /// switches): the prewarm stamp (the translators' code, the backend, the math mode, the
    /// host's translation switches, the device and the OS), the build of the MSL translator (any
    /// rebuild, as Vulkan keys on its SPIR-V translator's), the build of this assembly (the shader
    /// type the payload is read into) and the payload's format.
    /// </summary>
    internal static string Stamp(IGuestGpuBackend compiler, nint device) =>
        string.Join(
            '|',
            MetalShaderPrewarm.Stamp(compiler, device),
            typeof(Gen5MslTranslator).Module.ModuleVersionId.ToString("N"),
            typeof(MetalTranslationCache).Module.ModuleVersionId.ToString("N"),
            "payload=" + Gen5MslShaderPayload.FormatVersion.ToString(CultureInfo.InvariantCulture));

    /// <summary>Points both caches at the running game's folders.</summary>
    public static void Start(nint device)
    {
        var directory = SharpEmuPaths.UserPath(
            "pipeline_cache", MetalShaderPrewarm.SanitizeTitleId(VideoOutBackendServices.GetApplicationTitleId()), "metal-v2");
        ShaderAnalysisCache.UseGameFolder(Path.Combine(directory, "analysis"));
        TranslationDiskCache.UseGameFolder(Path.Combine(directory, "translations"), Stamp(GuestGpu.Current, device));
    }
}
