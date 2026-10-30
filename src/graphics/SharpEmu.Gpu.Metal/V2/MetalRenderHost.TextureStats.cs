// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// SHARPEMU_METAL_V2_DUMP_TEXTURE_STATS=&lt;address&gt;[,...] (hex, diagnostic): on each dump-trigger frame, the min, mean
/// and max of RGB per array layer (the first TextureStatsLayers) and level of every cached texture at those guest
/// addresses, read on the GPU, so what a shader samples can be set beside a decode of the guest bytes
/// (SHARPEMU_METAL_V2_DUMP_GUEST_RANGES) - a block-compressed texture the dumper cannot write as a picture included.
/// Flushes and waits.
/// </summary>
internal sealed partial class MetalRenderHost
{
    private const int TextureStatsLayers = 12;
    private const string TextureStatsArrayFunction = "sharpemu_texture_stats_array";
    private const string TextureStatsPlainFunction = "sharpemu_texture_stats_2d";

    private const string TextureStatsSource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void sharpemu_texture_stats_array(texture2d_array<float, access::read> t [[texture(0)]],
            device float4* o [[buffer(0)]], constant uint4& a [[buffer(1)]], uint2 g [[thread_position_in_grid]])
        {
            if (g.x < a.x && g.y < a.y) o[g.y * a.x + g.x] = t.read(g, a.z, a.w);
        }

        kernel void sharpemu_texture_stats_2d(texture2d<float, access::read> t [[texture(0)]],
            device float4* o [[buffer(0)]], constant uint4& a [[buffer(1)]], uint2 g [[thread_position_in_grid]])
        {
            if (g.x < a.x && g.y < a.y) o[g.y * a.x + g.x] = t.read(g, a.w);
        }
        """;

    private static readonly ulong[] _textureStatsAddresses =
        (Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DUMP_TEXTURE_STATS") ?? string.Empty)
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(text => ulong.TryParse(text.Replace("0x", string.Empty), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var address) ? address : 0)
            .Where(address => address != 0)
            .ToArray();

    private nint _textureStatsArrayPipeline;
    private nint _textureStatsPlainPipeline;

    private unsafe void DumpTextureStats()
    {
        foreach (var address in _textureStatsAddresses)
        {
            foreach (var (texture, described) in _imageCache.RetainTexturesAt(address))
            {
                try
                {
                    DumpTextureStats(texture, $"0x{address:X} {described}");
                }
                finally
                {
                    MetalNative.SendVoid(texture, MetalNative.Selector("release"));
                }
            }
        }
    }

    private unsafe void DumpTextureStats(nint texture, string label)
    {
        var type = (ulong)MetalV2Native.SendValue(texture, SelTextureType);
        var arrayed = type == 3; // MTLTextureType2DArray
        if (!arrayed && type != 2)
        {
            Console.Error.WriteLine($"[METAL-V2][TEXTURE-STATS] {label}: texture type {type} is not read");
            return;
        }

        var width = (uint)MetalV2Native.SendValue(texture, SelWidth);
        var height = (uint)MetalV2Native.SendValue(texture, SelHeight);
        var levels = (uint)MetalV2Native.SendValue(texture, MetalNative.Selector("mipmapLevelCount"));
        var layers = arrayed ? (uint)Math.Min(MetalV2Native.SendValue(texture, MetalNative.Selector("arrayLength")), (ulong)TextureStatsLayers) : 1u;
        ulong texels = 0;
        for (var level = 0; level < levels; level++)
        {
            texels += (ulong)Math.Max(width >> level, 1u) * Math.Max(height >> level, 1u);
        }

        var readback = _guestMemory.CreateReportBuffer(texels * layers * 16, "SharpEmu TextureStats", out var contents, out _);
        try
        {
            EndEncoders(PassEndCause.Other);
            var encoder = MetalNative.Send(EnsureCommandBuffer(), SelLodStatsComputeEncoder);
            if (encoder == 0)
            {
                return;
            }

            WaitForPreviousEncoder(encoder, render: false);
            MetalNative.SendVoid(encoder, SelLodStatsSetPipeline, TextureStatsPipeline(arrayed));
            MetalNative.SendSetAtIndex(encoder, MetalNative.Selector("setTexture:atIndex:"), texture, 0);
            ulong offset = 0;
            for (uint layer = 0; layer < layers; layer++)
            {
                for (uint level = 0; level < levels; level++)
                {
                    var w = Math.Max(width >> (int)level, 1u);
                    var h = Math.Max(height >> (int)level, 1u);
                    var arguments = stackalloc uint[4] { w, h, layer, level };
                    MetalNative.SendSetBuffer(encoder, SelLodStatsSetBuffer, readback, (nuint)offset, 0);
                    MetalNative.SendSetBytes(encoder, SelLodStatsSetBytes, (nint)arguments, 16, 1);
                    MetalNative.SendDispatch(encoder, SelLodStatsDispatch,
                        new MtlSize { Width = (w + 7) / 8, Height = (h + 7) / 8, Depth = 1 },
                        new MtlSize { Width = 8, Height = 8, Depth = 1 });
                    offset += (ulong)w * h * 16;
                }
            }

            UpdateEncoderFence(encoder, render: false);
            MetalNative.SendVoid(encoder, SelLodStatsEndEncoding);
            FlushAndWaitForGpu();

            var values = new ReadOnlySpan<float>((void*)contents, (int)(texels * layers * 4));
            var at = 0;
            for (uint layer = 0; layer < layers; layer++)
            {
                var line = new System.Text.StringBuilder($"[METAL-V2][TEXTURE-STATS] {label} layer {layer}:");
                for (uint level = 0; level < levels; level++)
                {
                    var count = (int)(Math.Max(width >> (int)level, 1u) * Math.Max(height >> (int)level, 1u));
                    float min = float.MaxValue, max = 0, sum = 0;
                    for (var texel = 0; texel < count; texel++, at += 4)
                    {
                        var value = Math.Max(values[at], Math.Max(values[at + 1], values[at + 2]));
                        min = Math.Min(min, value);
                        max = Math.Max(max, value);
                        sum += (values[at] + values[at + 1] + values[at + 2]) / 3;
                    }

                    line.Append(CultureInfo.InvariantCulture, $" m{level} {min:0.000}/{sum / count:0.000}/{max:0.00}");
                }

                Console.Error.WriteLine(line.ToString());
            }
        }
        finally
        {
            MetalNative.SendVoid(readback, MetalNative.Selector("release"));
        }
    }

    private nint TextureStatsPipeline(bool arrayed)
    {
        ref var pipeline = ref arrayed ? ref _textureStatsArrayPipeline : ref _textureStatsPlainPipeline;
        if (pipeline != 0)
        {
            return pipeline;
        }

        var options = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLCompileOptions"), MetalNative.Selector("alloc")), MetalNative.Selector("init"));
        MetalNative.ApplyMslCompileOptions(options);
        nint error = 0;
        var library = MetalNative.Send(
            _device, MetalNative.Selector("newLibraryWithSource:options:error:"), MetalNative.NsString(TextureStatsSource), options, ref error);
        MetalNative.SendVoid(options, MetalNative.Selector("release"));
        if (library == 0)
        {
            throw Fatal($"The texture statistics kernel did not compile: {MetalNative.DescribeError(error)}.");
        }

        var function = MetalNative.Send(library, MetalNative.Selector("newFunctionWithName:"),
            MetalNative.NsString(arrayed ? TextureStatsArrayFunction : TextureStatsPlainFunction));
        MetalNative.SendVoid(library, MetalNative.Selector("release"));
        error = 0;
        pipeline = MetalNative.Send(_device, MetalNative.Selector("newComputePipelineStateWithFunction:error:"), function, ref error);
        MetalNative.SendVoid(function, MetalNative.Selector("release"));
        if (pipeline == 0)
        {
            throw Fatal($"The texture statistics pipeline did not build: {MetalNative.DescribeError(error)}.");
        }

        return pipeline;
    }
}
