// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace SharpEmu.Gpu.Metal.V2;

// SHARPEMU_METAL_V2_USAGE_CENSUS=1 (diagnostic, with SHARPEMU_LOG_MEMORY=1): the images of 1 MiB or more used in
// the last second, grouped by pixel format, size, the compression-breaking usages they hold (w = shader write,
// v = pixel-format view, a = atomic, - = none: losslessly compressible) and what they were used as. An Apple GPU
// keeps a texture compressed only without those usages, and reading or loading one without compression costs
// 2.5-5x the bandwidth on an M4 (RGBA16F/RGBA8), so this says which of a frame's surfaces pay that.
internal sealed partial class MetalImageCache
{
    internal static readonly bool UsageCensusEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_USAGE_CENSUS") == "1";

    public string DescribeUsageCensus()
    {
        if (!UsageCensusEnabled)
        {
            return string.Empty;
        }

        var groups = new Dictionary<string, (int Count, ulong Bytes)>();
        var now = Stopwatch.GetTimestamp();
        using (var held = _lock.Hold())
        {
            _slots.ForEach((_, image) =>
            {
                if (image.Texture == 0 || image.IsWindowLinearView || image.AllocatedBytes < (1UL << 20) ||
                    now - image.LastAccessTimestamp > Stopwatch.Frequency)
                {
                    return;
                }

                var usage = image.Shape.Usage;
                var bits = ((usage & Mtl.TextureUsageShaderWrite) != 0 ? "w" : "") +
                    ((usage & Mtl.TextureUsagePixelFormatView) != 0 ? "v" : "") +
                    ((usage & (ulong)MtlTextureUsage.ShaderAtomic) != 0 ? "a" : "");
                var uses = image.Uses;
                var role = (uses.DepthTarget ? "D" : "") + (uses.RenderTarget ? "R" : "") + (uses.Storage ? "S" : "") +
                    (uses.VideoOut ? "V" : "");
                var key = $"{(MtlPixelFormat)image.Shape.PixelFormat}:{image.Shape.Width}x{image.Shape.Height}" +
                    $"{(image.Shape.ArrayLength > 1 ? "x" + image.Shape.ArrayLength : "")}:{(bits.Length == 0 ? "-" : bits)}:{(role.Length == 0 ? "T" : role)}";
                groups.TryGetValue(key, out var group);
                groups[key] = (group.Count + 1, group.Bytes + image.AllocatedBytes);
            });
        }

        var text = new StringBuilder(" usage_census=[");
        foreach (var (key, group) in groups.OrderByDescending(pair => pair.Value.Bytes).Take(24))
        {
            text.Append(' ').Append(key).Append('*').Append(group.Count).Append('/')
                .Append((group.Bytes / 1048576.0).ToString("F0", CultureInfo.InvariantCulture));
        }

        return text.Append(" ]").ToString();
    }
}
