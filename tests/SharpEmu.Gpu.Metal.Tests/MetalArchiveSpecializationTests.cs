// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.ShaderCompiler.Metal;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// What buffer-word variants (MetalRenderHost.BufferWordVariants) rely on in Metal's binary
/// archives: a library file the archive wrote from a pipeline holds that pipeline's function
/// with its constants resolved, so a library loaded from the generic pipeline's file cannot be
/// specialized (the values given are ignored; logged, not asserted), while a library compiled
/// from the MSL can, and a variant's own file gives back the specialized function and finds
/// its pipeline. A kernel that writes its word times three tells them apart.
/// </summary>
public sealed unsafe class MetalArchiveSpecializationTests(ITestOutputHelper output)
{
    private const string Source =
        """
        #include <metal_stdlib>
        using namespace metal;
        constant uint c1 [[function_constant(1)]];
        constant bool c1_set = is_function_constant_defined(c1);
        kernel void gen5_cs(device uint* d [[buffer(0)]], uint id [[thread_position_in_grid]])
        {
            d[id] = (c1_set ? c1 : d[0]) * 3u + id;
        }
        """;

    [Fact]
    public void ASpecializedFunctionComesFromTheSourceOrItsOwnFileNotTheGenericFile()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1") return;
        var device = MetalNative.MTLCreateSystemDefaultDevice();
        var root = Path.Combine(Path.GetTempPath(), $"p3probe-{Guid.NewGuid():N}");
        Gen5MslShader.BufferWordConstant[] declared = [new(1, 0)];
        var keyGeneric = new UInt128(11, 1);
        var keyVariant = new UInt128(11, 2);
        try
        {
            var lib1 = MetalRenderHost.TryCompileLibrary(device, Source, false, out var e1, out _, askArchive: false);
            Assert.True(lib1 != 0, e1);
            var g1 = MetalRenderHost.NewTranslatedFunction(lib1, "gen5_cs", null, declared, [], out _);
            var v1 = MetalRenderHost.NewTranslatedFunction(lib1, "gen5_cs", null, declared, [7u], out _);
            Assert.True(g1 != 0 && v1 != 0);
            using (var archive = MetalBinaryArchive.Open(root, device, "translator-a", 64L << 20))
            {
                archive.CaptureCompute(keyGeneric, g1);
                archive.CaptureCompute(keyVariant, v1);
                archive.Flush(wait: true);
            }

            using (var archive = MetalBinaryArchive.Open(root, device, "translator-a", 64L << 20))
            {
                var lib2 = MetalRenderHost.TryCompileLibrary(device, Source, false, out _, out _, askArchive: false);
                var lib3 = archive.TryLoadLibrary(keyGeneric);
                var lib4 = archive.TryLoadLibrary(keyVariant);
                foreach (var (name, lib) in new[] { ("source", lib2), ("generic-file", lib3), ("variant-file", lib4), ("same-lib", lib1) })
                {
                    Assert.True(lib != 0, name);
                    var g = MetalRenderHost.NewTranslatedFunction(lib, "gen5_cs", null, declared, [], out _);
                    var v = MetalRenderHost.NewTranslatedFunction(lib, "gen5_cs", null, declared, [7u], out _);
                    var results = (GenericHit: Hit(device, archive, keyGeneric, g), VariantHit: Hit(device, archive, keyVariant, v),
                        Generic: Run(device, g), Variant: Run(device, v));
                    output.WriteLine($"{name}: {results} (5*3=15 read from the data, 7*3=21 specialized)");
                    if (name == "generic-file")
                    {
                        // Metal's behaviour today, which the design works around: not asserted.
                        continue;
                    }

                    // The specialized function computes with the value given, and the archive has its pipeline.
                    Assert.Equal((21u, true), (results.Variant, results.VariantHit));
                    if (name != "variant-file")
                    {
                        Assert.Equal((15u, true), (results.Generic, results.GenericHit));
                    }
                }
            }
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, true);
        }
    }

    private static unsafe uint Run(nint device, nint function)
    {
        nint error = 0;
        var state = MetalNative.Send(device, MetalNative.Selector("newComputePipelineStateWithFunction:error:"), function, ref error);
        if (state == 0) return 0xDEAD;
        var queue = MetalNative.Send(device, MetalNative.Selector("newCommandQueue"));
        var buffer = MetalNative.SendNewBuffer(device, MetalNative.Selector("newBufferWithLength:options:"), 64, 0);
        var words = (uint*)MetalNative.Send(buffer, MetalNative.Selector("contents"));
        words[0] = 5;
        var commands = MetalNative.Send(queue, MetalNative.Selector("commandBuffer"));
        var encoder = MetalNative.Send(commands, MetalNative.Selector("computeCommandEncoder"));
        MetalNative.SendVoid(encoder, MetalNative.Selector("setComputePipelineState:"), state);
        MetalNative.SendSetBuffer(encoder, MetalNative.Selector("setBuffer:offset:atIndex:"), buffer, 0, 0);
        MetalNative.SendDispatch(encoder, MetalNative.Selector("dispatchThreadgroups:threadsPerThreadgroup:"),
            new MtlSize { Width = 1, Height = 1, Depth = 1 }, new MtlSize { Width = 1, Height = 1, Depth = 1 });
        MetalNative.SendVoid(encoder, MetalNative.Selector("endEncoding"));
        MetalNative.SendVoid(commands, MetalNative.Selector("commit"));
        MetalNative.SendVoid(commands, MetalNative.Selector("waitUntilCompleted"));
        return words[0];
    }

    private static bool Hit(nint device, MetalBinaryArchive archive, UInt128 key, nint function)
    {
        var stored = archive.OpenLibraryArchive(key);
        if (stored == 0) return false;
        var descriptor = MetalNative.Send(
            MetalNative.Send(MetalNative.Class("MTLComputePipelineDescriptor"), MetalNative.Selector("alloc")),
            MetalNative.Selector("init"));
        MetalNative.SendVoid(descriptor, MetalNative.Selector("setComputeFunction:"), function);
        MetalNative.SendVoid(descriptor, MetalNative.Selector("setBinaryArchives:"),
            MetalNative.Send(MetalNative.Class("NSArray"), MetalNative.Selector("arrayWithObject:"), stored));
        nint error = 0;
        var state = MetalNative.SendNewMeshPipeline(
            device, MetalNative.Selector("newComputePipelineStateWithDescriptor:options:reflection:error:"),
            descriptor, MetalBinaryArchive.FailOnBinaryArchiveMiss, 0, ref error);
        return state != 0;
    }
}
