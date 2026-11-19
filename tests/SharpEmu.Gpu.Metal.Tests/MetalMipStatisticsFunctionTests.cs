// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.ShaderCompiler.Metal;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A pixel program whose GET_LOD_STATS sites sit behind a function constant is made into a
/// function with a value for it: both values specialize, and the archive key of the pipeline
/// with the counters differs from the one without.
/// </summary>
public sealed class MetalMipStatisticsFunctionTests
{
    private static readonly string Source =
        $$"""
        #include <metal_stdlib>
        using namespace metal;
        constant bool {{Gen5MslShader.MipStatisticsFunctionConstantName}} [[function_constant({{Gen5MslShader.MipStatisticsFunctionConstantIndex}})]];
        fragment float4 gen5_ps(float4 position [[position]], device atomic_uint* counters [[buffer(0)]])
        {
            if ({{Gen5MslShader.MipStatisticsFunctionConstantName}})
            {
                atomic_fetch_add_explicit(counters, 1u, memory_order_relaxed);
            }

            return position;
        }
        """;

    [Fact]
    public void TheFunctionSpecializesWithTheConstantOnAndOff()
    {
        var device = MetalNative.MTLCreateSystemDefaultDevice();
        var library = MetalRenderHost.TryCompileLibrary(device, Source, usesPerVertexInputs: false, out var libraryError);
        Assert.True(library != 0, libraryError);
        nint off = 0;
        nint on = 0;
        try
        {
            off = MetalRenderHost.NewTranslatedFunction(library, "gen5_ps", false, [], [], out var offError);
            on = MetalRenderHost.NewTranslatedFunction(library, "gen5_ps", true, [], [], out var onError);
            Assert.True(off != 0, MetalNative.DescribeError(offError));
            Assert.True(on != 0, MetalNative.DescribeError(onError));
            Assert.NotEqual(off, on);
        }
        finally
        {
            foreach (var handle in (ReadOnlySpan<nint>)[off, on, library])
            {
                if (handle != 0)
                {
                    MetalNative.SendVoid(handle, MetalNative.Selector("release"));
                }
            }
        }
    }

    [Fact]
    public void ThePipelineWithCountersIsListedUnderAnotherArchiveKey()
    {
        var key = new UInt128(0x1234, 0x5678);

        Assert.Equal(key, MetalBinaryArchive.LodStatisticsKey(key, counters: false));
        Assert.NotEqual(key, MetalBinaryArchive.LodStatisticsKey(key, counters: true));
        Assert.Equal(default, MetalBinaryArchive.LodStatisticsKey(default, counters: true));
        Assert.NotEqual(
            MetalBinaryArchive.PairKey(default, key),
            MetalBinaryArchive.PairKey(default, MetalBinaryArchive.LodStatisticsKey(key, counters: true)));
    }
}
