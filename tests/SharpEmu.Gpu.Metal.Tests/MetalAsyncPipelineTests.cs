// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using SharpEmu.Gpu.Metal.Tests.Scenes;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE.GpuMemory;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// Pipelines built off the worker: a compile still running is left out and shared, a result
/// is taken by its own key whatever order compiles end in, a request that may not be left
/// out builds a queued compile itself, an untaken result is dropped, teardown does not wait
/// for queued compiles, and a compile in flight keeps its modules through the shader cache
/// bound. Which targets count as rebuilt every frame, and that a draw into any other target
/// waits for its pipeline - unless, in wide mode, passes drew into it every frame - serving the
/// guest stores that wait for a writeback meanwhile. Scenes are behind the exclusive device gate.
/// </summary>
public sealed class MetalAsyncPipelineTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private static readonly TimeSpan Patience = TimeSpan.FromSeconds(10);
    private static readonly MetalShaderCacheLimits ReleaseEverything = new(0, 1, 0, 1, 0);

    private static long AnHourFromNow => Stopwatch.GetTimestamp() + 3600 * Stopwatch.Frequency;

    private static MetalPipelineCompiler.Result Named(string name) => new(0, null, name);

    [Fact]
    public void ACompileInFlightIsLeftOutAndSharedThenTakenOnce()
    {
        var woken = 0;
        using var gate = new ManualResetEventSlim(false);
        using var compiler = new MetalPipelineCompiler(() => Interlocked.Increment(ref woken), 2) { Gate = gate };
        var builds = 0;
        MetalPipelineCompiler.Result Build()
        {
            Interlocked.Increment(ref builds);
            return Named("a");
        }

        Assert.False(compiler.TryTake("a", Build, out _));
        Assert.False(compiler.TryTake("a", Build, out _));
        Assert.Equal(1, compiler.InFlight);

        gate.Set();
        Assert.True(SpinWait.SpinUntil(() => Volatile.Read(ref woken) == 1, Patience));
        Assert.True(compiler.TryTake("a", Build, out var result));
        Assert.Equal("a", result!.VertexLayout);
        Assert.Equal(1, builds);
        Assert.Equal(0, compiler.InFlight);
        var (started, finished, deduped, _) = compiler.TakeCounters();
        Assert.Equal((1L, 1L, 1L), (started, finished, deduped));
    }

    [Fact]
    public void CompilesFinishedOutOfOrderAreEachTakenByTheirOwnKey()
    {
        var woken = 0;
        using var slow = new ManualResetEventSlim(false);
        using var compiler = new MetalPipelineCompiler(() => Interlocked.Increment(ref woken), 2);

        Assert.False(compiler.TryTake("first", () => { slow.Wait(); return Named("first"); }, out _));
        Assert.False(compiler.TryTake("second", () => Named("second"), out _));
        Assert.True(SpinWait.SpinUntil(() => Volatile.Read(ref woken) == 1, Patience));

        Assert.False(compiler.TryTake("first", () => Named("wrong"), out _));
        Assert.True(compiler.TryTake("second", () => Named("wrong"), out var second));
        Assert.Equal("second", second!.VertexLayout);

        slow.Set();
        Assert.True(SpinWait.SpinUntil(() => Volatile.Read(ref woken) == 2, Patience));
        Assert.True(compiler.TryTake("first", () => Named("wrong"), out var first));
        Assert.Equal("first", first!.VertexLayout);
    }

    [Fact]
    public void ARequestThatMayNotBeLeftOutBuildsAQueuedCompileItself()
    {
        using var gate = new ManualResetEventSlim(false);
        using var compiler = new MetalPipelineCompiler(static () => { }, 1) { Gate = gate };
        var builds = 0;
        var builtOn = 0;
        MetalPipelineCompiler.Result Build()
        {
            Interlocked.Increment(ref builds);
            builtOn = Environment.CurrentManagedThreadId;
            return Named("a");
        }

        Assert.False(compiler.TryWait("a", out _));
        Assert.False(compiler.TryTake("a", Build, out _));
        // The compiler thread is held at its gate: waiting for it would never end.
        Assert.True(compiler.TryWait("a", out var result));
        Assert.Equal("a", result!.VertexLayout);
        Assert.Equal(Environment.CurrentManagedThreadId, builtOn);
        Assert.Equal(0, compiler.InFlight);

        // The compiler thread finds it claimed and builds nothing.
        gate.Set();
        Assert.False(compiler.TryTake("b", () => Named("b"), out _));
        Assert.True(SpinWait.SpinUntil(() => compiler.InFlight == 0, Patience));
        Assert.Equal(1, builds);
    }

    [Fact]
    public void AResultNobodyTookIsDroppedAndCompiledAgainWhenAskedFor()
    {
        var woken = 0;
        using var compiler = new MetalPipelineCompiler(() => Interlocked.Increment(ref woken), 1);
        var builds = 0;
        MetalPipelineCompiler.Result Build()
        {
            Interlocked.Increment(ref builds);
            return Named("a");
        }

        Assert.False(compiler.TryTake("a", Build, out _));
        Assert.True(SpinWait.SpinUntil(() => Volatile.Read(ref woken) == 1, Patience));

        // Finished after this moment: kept.
        Assert.Equal(0, compiler.DropAbandoned(Stopwatch.GetTimestamp() - Stopwatch.Frequency));
        Assert.Equal(1, compiler.DropAbandoned(Stopwatch.GetTimestamp() + 1));
        Assert.Equal(1, compiler.Abandoned);

        Assert.False(compiler.TryTake("a", Build, out _));
        Assert.True(SpinWait.SpinUntil(() => Volatile.Read(ref woken) == 2, Patience));
        Assert.True(compiler.TryTake("a", Build, out _));
        Assert.Equal(2, builds);
    }

    [Fact]
    public void TeardownWithCompilesQueuedSkipsThemAndReturns()
    {
        using var gate = new ManualResetEventSlim(false);
        var compiler = new MetalPipelineCompiler(static () => { }, 1) { Gate = gate };
        var builds = 0;
        for (var index = 0; index < 4; index++)
        {
            Assert.False(compiler.TryTake(index, () => { Interlocked.Increment(ref builds); return Named("queued"); }, out _));
        }

        Assert.True(Task.Run(compiler.Dispose).Wait(Patience));
        Assert.Equal(0, builds);
        Assert.Equal(0, compiler.InFlight);
        // Asked for after teardown: built where it is asked for.
        Assert.True(compiler.TryTake(0, () => Named("late"), out var late));
        Assert.Equal("late", late!.VertexLayout);
    }

    [Fact]
    public void ATargetIsRebuiltEveryFrameOnlyWhenEachOfTheLastFramesRebuiltIt()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var host = rig.Host;
        host.AsynchronousGraphics = true;
        var everyFrame = new MetalRenderHost.TargetSubresource(0x1000, 0, 0);
        var missedOne = new MetalRenderHost.TargetSubresource(0x1000, 1, 0);
        for (var frame = 0; frame < MetalRenderHost.RebuiltFrames; frame++)
        {
            Assert.False(host.RebuiltEveryFrame(everyFrame));
            host.NoteFlipForCompilingDraws();
            host.NoteTargetRebuilt(everyFrame);
            if (frame != 1)
            {
                host.NoteTargetRebuilt(missedOne);
            }
        }

        Assert.True(host.RebuiltEveryFrame(everyFrame));
        Assert.False(host.RebuiltEveryFrame(missedOne));
        // A new frame: not until this one rebuilds it too.
        host.NoteFlipForCompilingDraws();
        Assert.False(host.RebuiltEveryFrame(everyFrame));
        host.NoteTargetRebuilt(everyFrame);
        Assert.True(host.RebuiltEveryFrame(everyFrame));
        // Left alone for as many frames as the rule looks back, it is forgotten.
        for (var frame = 0; frame < MetalRenderHost.RebuiltFrames; frame++)
        {
            host.NoteFlipForCompilingDraws();
        }

        host.NoteTargetRebuilt(missedOne);
        Assert.False(host.RebuiltEveryFrame(everyFrame));
    }

    [Fact]
    public void ADrawIntoATargetNotRebuiltEveryFrameWaitsForItsPipeline()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        const uint size = 16;
        rig.Host.AsynchronousGraphics = true;
        using var gate = new ManualResetEventSlim(false);
        rig.Host.PipelineCompiler.Gate = gate;
        var target = rig.ColorTarget(size, size);
        var state = new SceneState(size, size).ColorTarget(0, target, size, size);

        rig.Submit(rig.DrawMesh(new ScenePm4(), state, SceneDraws.Quad(0.5f, 1, 0, 0)));

        var pixels = rig.ReadImage(target, size, size);
        Assert.Equal(0xFF0000FFu, SceneDraws.Pixel(pixels, size, size / 2, size / 2));
        var (skipped, waits) = rig.Host.CompilingDrawCounters;
        Assert.Equal(0, skipped);
        Assert.True(waits >= 1, $"waits={waits}");
        Assert.Empty(rig.Fatals);
        gate.Set();
    }

    // InterpolatedColor behind an s_nop: a program of its own, whose library compiles anew.
    private static readonly uint[] PaddedInterpolatedColor = [0xBF800000, .. ScenePrograms.InterpolatedColor];

    // With the pool busy, the compile is still queued when the draw needs it, and runs off the worker.
    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void AGuestStoreIsServedWhileADrawWaitsForItsLibraryCompile(bool poolBusy, bool deferred)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        rig.Host.DeferLibraryCompiles = deferred;
        using var poolFree = new ManualResetEventSlim(false);
        try
        {
            StoreWhileADrawWaitsForItsLibraryCompile(rig, poolBusy, poolFree);
        }
        finally
        {
            poolFree.Set();
        }
    }

    private static void StoreWhileADrawWaitsForItsLibraryCompile(SceneRig rig, bool poolBusy, ManualResetEventSlim poolFree)
    {
        const uint size = 16;
        // Every draw waits for its pipeline.
        rig.Host.AsynchronousGraphics = false;
        rig.Host.ImageCache.StorePublishTakeTimeoutMilliseconds = 2000;
        var written = rig.ColorTarget(size, size);
        rig.Submit(rig.DrawMesh(new ScenePm4(), new SceneState(size, size).ColorTarget(0, written, size, size),
            SceneDraws.Quad(0.5f, 1, 0, 0)));
        // Only the texture holds the red the GPU drew.
        Assert.Equal(0u, BitConverter.ToUInt32(rig.Memory.ReadRaw(written, 4)));

        // The next draw's pixel program compiles while a guest thread stores into the first
        // target; the compile is held until the store has returned.
        using var gate = new ManualResetEventSlim(false);
        using var held = new ManualResetEventSlim(false);
        var stored = false;
        var compileHeld = false;
        byte[]? landedOver = null;
        Exception? failure = null;
        var store = new Thread(() =>
        {
            try
            {
                if (!held.Wait(Patience))
                {
                    return;
                }

                stored = ((IGuestImageStore)rig.Host.ImageCache).MarkCpuWrite(written, 8);
                landedOver = rig.Memory.ReadRaw(written, 4);
                compileHeld = !gate.IsSet;
            }
            catch (Exception exception)
            {
                failure = exception;
            }
            finally
            {
                gate.Set();
            }
        })
        {
            IsBackground = true,
        };
        store.Start();
        rig.Host.BeforeLibraryCompile = () =>
        {
            held.Set();
            gate.Wait();
        };
        if (poolBusy)
        {
            using var busy = new CountdownEvent(MetalLibraryCompilePool.ThreadCount);
            for (var index = 0; index < MetalLibraryCompilePool.ThreadCount; index++)
            {
                MetalLibraryCompilePool.Shared.Run(() =>
                {
                    busy.Signal();
                    poolFree.Wait();
                });
            }

            Assert.True(busy.Wait(Patience), "The pool's threads did not all take a job.");
        }

        var other = rig.ColorTarget(size, size);
        rig.Submit(rig.DrawMesh(new ScenePm4(), new SceneState(size, size).ColorTarget(0, other, size, size),
            SceneDraws.Quad(0.5f, 0, 1, 0), pixelProgram: PaddedInterpolatedColor));
        store.Join();
        rig.Host.BeforeLibraryCompile = null;

        Assert.Null(failure);
        Assert.True(held.IsSet, "The second draw compiled no library.");
        Assert.True(stored);
        Assert.True(compileHeld, "The store returned only once the compile was let go.");
        // The writeback landed before the store returned, while the draw still waited.
        Assert.Equal(0xFF0000FFu, BitConverter.ToUInt32(landedOver!));
        var counts = rig.Host.ImageCache.StorePublishCounts;
        Assert.Equal((1L, 0L), (counts.Writebacks, counts.Fallbacks));
        // And the draw that waited was drawn.
        Assert.Equal(0xFF00FF00u, SceneDraws.Pixel(rig.ReadImage(other, size, size), size, size / 2, size / 2));
        Assert.Empty(rig.Fatals);
    }

    // Wide mode (SHARPEMU_METAL_V2_ASYNC_GRAPHICS=wide): a target passes drew into in each of the last
    // frames, never cleared, takes a draw left out while its pipeline compiles; a later frame draws it.
    [Fact]
    public void InWideModeADrawIntoATargetDrawnEveryFrameIsLeftOutWhileItsPipelineCompiles()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        const uint size = 16;
        const uint red = 0xFF0000FFu, yellow = 0xFF00FFFFu;
        rig.Host.AsynchronousGraphics = true;
        rig.Host.AsynchronousWide = true;
        var target = rig.ColorTarget(size, size);
        SceneState Target() => new SceneState(size, size).ColorTarget(0, target, size, size);
        var additive = Target();
        additive.Context[Cx.CbBlend0Control] = 1u | (1u << 8) | (1u << 16) | (1u << 30);
        uint Frame(bool green)
        {
            rig.Host.NoteFlipForCompilingDraws();
            var pm4 = rig.DrawMesh(new ScenePm4(), Target(), SceneDraws.Quad(0.5f, 1, 0, 0));
            rig.Submit(green ? rig.DrawMesh(pm4, additive, SceneDraws.Quad(0.5f, 0, 1, 0)) : pm4);
            return SceneDraws.Pixel(rig.ReadImage(target, size, size), size, size / 2, size / 2);
        }

        // Red alone, one pipeline, in as many frames as the rule looks back.
        for (var frame = 0; frame < MetalRenderHost.RebuiltFrames; frame++)
        {
            Assert.Equal(red, Frame(green: false));
        }

        using var gate = new ManualResetEventSlim(false);
        rig.Host.PipelineCompiler.Gate = gate;
        Assert.Equal(red, Frame(green: true));
        Assert.Equal(1, rig.Host.CompilingDrawCounters.Skipped);

        gate.Set();
        Assert.True(SpinWait.SpinUntil(() => Frame(green: true) == yellow, Patience));
        Assert.Empty(rig.Fatals);
    }

    // Wide mode counts a depth target whose depth every frame clears, its stencil loaded (Grand Theft Auto V's);
    // outside wide mode nothing is noted.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void InWideModeADepthTargetIsCountedWhenEachFrameClearsItsDepth(bool wide)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var host = rig.Host;
        host.AsynchronousGraphics = true;
        host.AsynchronousWide = wide;
        var depth = new MetalRenderHost.TargetSubresource(0x2000, 0, 0);
        var missedOne = new MetalRenderHost.TargetSubresource(0x2000, 0, 1);
        for (var frame = 0; frame < MetalRenderHost.RebuiltFrames; frame++)
        {
            host.NoteFlipForCompilingDraws();
            host.NoteDepthCleared(depth);
            if (frame != 1)
            {
                host.NoteDepthCleared(missedOne);
            }
        }

        Assert.Equal(wide, host.DepthClearedEveryFrame(depth));
        Assert.False(host.DepthClearedEveryFrame(missedOne));
        Assert.False(host.RebuiltEveryFrame(depth));
    }

    // Without wide mode the same target, never cleared, waits for the pipeline.
    [Fact]
    public void WithoutWideModeATargetOnlyDrawnEveryFrameWaitsForThePipeline()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        const uint size = 16;
        rig.Host.AsynchronousGraphics = true;
        rig.Host.AsynchronousWide = false;
        var target = rig.ColorTarget(size, size);
        SceneState Target() => new SceneState(size, size).ColorTarget(0, target, size, size);
        for (var frame = 0; frame < MetalRenderHost.RebuiltFrames; frame++)
        {
            rig.Host.NoteFlipForCompilingDraws();
            rig.Submit(rig.DrawMesh(new ScenePm4(), Target(), SceneDraws.Quad(0.5f, 1, 0, 0)));
        }

        using var gate = new ManualResetEventSlim(false);
        rig.Host.PipelineCompiler.Gate = gate;
        var additive = Target();
        additive.Context[Cx.CbBlend0Control] = 1u | (1u << 8) | (1u << 16) | (1u << 30);
        rig.Host.NoteFlipForCompilingDraws();
        var pm4 = rig.DrawMesh(new ScenePm4(), Target(), SceneDraws.Quad(0.5f, 1, 0, 0));
        rig.Submit(rig.DrawMesh(pm4, additive, SceneDraws.Quad(0.5f, 0, 1, 0)));

        Assert.Equal(0xFF00FFFFu, SceneDraws.Pixel(rig.ReadImage(target, size, size), size, size / 2, size / 2));
        Assert.Equal(0, rig.Host.CompilingDrawCounters.Skipped);
        Assert.Empty(rig.Fatals);
        gate.Set();
    }

    [Fact]
    public void ADispatchSuspendedOnItsPipelineKeepsItsModulesThroughATrimAndStoresOnceReady()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        if (!rig.Host.SupportsPipelineSuspend)
        {
            Output.WriteLine("SHARPEMU_METAL_V2_ASYNC_PIPELINES=0: dispatches build their pipelines on the worker.");
            return;
        }

        rig.Host.ShaderCacheLimits = MetalShaderCacheLimits.Unbounded;
        const uint groups = 4;
        using var gate = new ManualResetEventSlim(false);
        rig.Host.PipelineCompiler.Gate = gate;
        var buffer = rig.Memory.Allocate(groups * 64 * 4, 256);
        rig.Enqueue(rig.Dispatch(new ScenePm4(), ScenePrograms.StoreGlobalId, Descriptors.Buffer(buffer, 4, groups * 64, format: 20), groups).ToArray());
        rig.Host.RunPendingCommands();
        rig.Host.RunCommandStreamSlices(64);

        Assert.True(rig.Host.CommandStream.HasPending);
        Assert.Equal(1, rig.Host.PipelinesCompiling);
        var (_, modules) = rig.Host.TrimShaderCache(rig.Host.CompletedSerial, AnHourFromNow, ReleaseEverything);
        Assert.Equal(0, modules);

        gate.Set();
        rig.Drain();
        var stored = rig.Read<uint>(buffer, (int)(groups * 64));
        for (var index = 0u; index < stored.Length; index++)
        {
            Assert.Equal(index ^ 0x5A5Au, stored[index]);
        }

        Assert.Equal(0, rig.Host.PipelinesCompiling);
        Assert.Empty(rig.Fatals);
    }
}
