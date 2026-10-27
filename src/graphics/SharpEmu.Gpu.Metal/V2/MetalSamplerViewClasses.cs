// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Images;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

// Recomputed from the resolved views each preparation: guest descriptors and the image
// cache may select a different numeric format even when the shader program is unchanged.
internal static class MetalSamplerViewClasses
{
    internal const byte AllInteger = 1;
    private const byte AnyNonInteger = 2;

    internal static void Resolve(List<SampledImagePair> pairs, ReadOnlySpan<ImageRequest> images, Span<byte> classes)
    {
        classes.Clear();
        if (classes.IsEmpty) return;
        foreach (var pair in pairs)
        {
            // Invalid image entries do not establish a pairing, matching the per-sampler walk.
            // Pairs for samplers outside this draw's table never participate in that walk.
            if (pair.Sampler >= classes.Length || pair.Image >= images.Length) continue;
            ref var classification = ref classes[(int)pair.Sampler];
            if (classification == AnyNonInteger) continue;
            classification = ViewFormatRules.IsIntegerFormat(images[(int)pair.Image].View.Format)
                ? AllInteger : AnyNonInteger;
        }
    }
}
