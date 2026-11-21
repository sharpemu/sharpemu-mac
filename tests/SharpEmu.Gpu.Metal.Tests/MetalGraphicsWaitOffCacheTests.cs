// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Pipelines;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A draw that must have its pipeline waits for it outside the pipeline cache's lock
/// (IShaderPipelineHost.WaitsForGraphicsPipelines): the cache starts the compile with TryCreateGraphicsPipeline,
/// waits in WaitForGraphicsPipeline, and asks again with the description that started it. A second draw with an
/// equal description shares the compile, and the re-ask finds it done (the sync's notes-g3 "Needs a decision 2").
/// Device work is behind the exclusive device gate.
/// </summary>
public sealed class MetalGraphicsWaitOffCacheTests(ITestOutputHelper output)
{
    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ADrawThatWaitsSharesTheCompileAndTheReaskTakesIt(bool compilerThreadRuns)
    {
        if (!RequireDevice()) return;
        // Declared before the rig, so it is disposed after it: a compiler thread may still reach this gate until the
        // rig's host has joined its threads, and waiting on a disposed gate throws on that thread and ends the process.
        using var gate = new ManualResetEventSlim(false);
        using var rig = new MetalPassRig(Device.Value, 1024 * 1024);
        var host = rig.Host;
        host.AsynchronousGraphics = true;
        Assert.True(((IShaderPipelineHost)host).WaitsForGraphicsPipelines);
        host.PipelineCompiler.Gate = gate;
        _ = host.PipelineCompiler.TakeCounters();

        var first = rig.PipelineDescription([GpuFormat.R8G8B8A8Unorm], ["float4(1.0f, 0.0f, 0.0f, 1.0f)"]);
        // Another draw's description: other objects, the same pipeline.
        var second = new GraphicsPipelineDescription
        {
            Rendering = first.Rendering,
            VertexInput = first.VertexInput,
            VertexInfo = first.VertexInfo,
            VertexProgram = first.VertexProgram,
            VertexStage = first.VertexStage,
            PixelProgram = first.PixelProgram,
            PixelStage = first.PixelStage,
            StaticParameters = first.StaticParameters,
        };

        // The cache's first ask starts the compile; the second draw's shares it.
        Assert.False(host.TryCreateGraphicsPipeline(first, out _));
        Assert.False(host.TryCreateGraphicsPipeline(second, out _));
        var (started, _, deduped, _) = host.PipelineCompiler.TakeCounters();
        Assert.Equal(1L, started);
        Assert.Equal(1L, deduped);

        if (compilerThreadRuns)
        {
            gate.Set();
        }

        // Outside the cache's lock: built here when no compiler thread took it, waited for when one did.
        ((IShaderPipelineHost)host).WaitForGraphicsPipeline(second);
        Assert.True(host.TryCreateGraphicsPipeline(first, out var handle));
        Assert.NotEqual(0L, (long)handle.Pipeline);
        Assert.Equal(1L, host.GraphicsWaitsOffCacheCount);
        var (skipped, waits) = host.CompilingDrawCounters;
        Assert.Equal(1L, skipped);
        Assert.Equal(1L, waits);
        gate.Set();
    }

    [Fact]
    public void WithoutTheCompilerThreadsTheCacheKeepsTheBlockingBuild()
    {
        if (!RequireDevice()) return;
        using var rig = new MetalPassRig(Device.Value, 1024 * 1024);
        var host = rig.Host;
        host.AsynchronousGraphics = false;
        Assert.False(((IShaderPipelineHost)host).WaitsForGraphicsPipelines);
        host.AsynchronousGraphics = true;
        host.GraphicsWaitsOffCache = false;
        Assert.False(((IShaderPipelineHost)host).WaitsForGraphicsPipelines);
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Pipeline compiles run on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }
}
