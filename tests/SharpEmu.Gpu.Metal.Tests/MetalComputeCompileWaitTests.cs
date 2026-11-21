// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Metal.Tests;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// The compile TryCreateComputePipeline leaves running is handed out (IShaderPipelineHost.GetComputePipelineCompletion),
/// as upstream's host hands out its in-flight compute compile: a dispatch that may not be left out and a linked module's
/// bounded wait then wake when it finishes instead of polling every millisecond. The task never faults; the re-ask takes
/// the pipeline. SHARPEMU_METAL_COMPUTE_COMPILE_WAIT=0 hands out nothing. Device work is behind the exclusive device gate.
/// </summary>
public sealed class MetalComputeCompileWaitTests(ITestOutputHelper output)
{
    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);

    [Fact]
    public void AComputeCompileInFlightHandsOutItsCompletionAndTheReaskTakesThePipeline()
    {
        if (!RequireDevice()) return;
        // Declared before the rig, so it is disposed after it: a compiler thread may still reach this gate until the
        // rig's host has joined its threads, and waiting on a disposed gate throws on that thread and ends the process.
        using var gate = new ManualResetEventSlim(false);
        using var rig = new MetalPassRig(Device.Value, 1024 * 1024);
        var host = rig.Host;
        IShaderPipelineHost pipelines = host;
        var description = Description(host, 0x5EED);
        Assert.Null(pipelines.GetComputePipelineCompletion(description.Program));

        host.PipelineCompiler.Gate = gate;
        Assert.False(host.TryCreateComputePipeline(description, out _));
        var completion = pipelines.GetComputePipelineCompletion(description.Program);
        Assert.NotNull(completion);
        Assert.False(completion!.IsCompleted);
        // Another program's compile is not this one's.
        Assert.Null(pipelines.GetComputePipelineCompletion(new ShaderProgram(description.Program.Id + 1, 0)));

        gate.Set();
        Assert.True(completion.Wait(TimeSpan.FromSeconds(60)), "The compute compile never finished.");
        Assert.False(completion.IsFaulted);
        Assert.True(host.TryCreateComputePipeline(description, out var handle));
        Assert.NotEqual(0L, (long)handle.Pipeline);
        Assert.Null(pipelines.GetComputePipelineCompletion(description.Program));
    }

    [Fact]
    public void WithTheSwitchOffNoCompletionIsHandedOut()
    {
        if (!RequireDevice()) return;
        using var gate = new ManualResetEventSlim(false);
        using var rig = new MetalPassRig(Device.Value, 1024 * 1024);
        var host = rig.Host;
        IShaderPipelineHost pipelines = host;
        var description = Description(host, 0x5EEE);
        host.PipelineCompiler.Gate = gate;
        Assert.False(host.TryCreateComputePipeline(description, out _));
        var previous = MetalRenderHost.ComputeCompileWait;
        try
        {
            MetalRenderHost.ComputeCompileWait = false;
            Assert.Null(pipelines.GetComputePipelineCompletion(description.Program));
        }
        finally
        {
            MetalRenderHost.ComputeCompileWait = previous;
            gate.Set();
        }
    }

    private static ComputePipelineDescription Description(MetalRenderHost host, ulong hash)
    {
        var request = Gen5ComputeFixtures.CreateComputeRequest(Gen5ComputeFixtures.ExecStore);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var compiled, out var error), error);
        var shader = new MetalCompiledGuestShader(compiled, request);
        var program = new ShaderProgram(hash, host.CreateShaderModule(shader, ShaderStage.Compute, hash, hash));
        return new ComputePipelineDescription
        {
            Input = new ComputeInputInfo { ThreadsX = 32, ThreadsY = 1, ThreadsZ = 1 },
            Program = program,
            Stage = new ShaderProgramInfo { Stage = ShaderStageKind.Compute, Hash = hash },
        };
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
