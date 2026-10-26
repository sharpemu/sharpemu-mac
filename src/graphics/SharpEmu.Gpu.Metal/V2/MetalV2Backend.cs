// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.Versioning;
using SharpEmu.HLE;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Metal;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// The v2 backend at the guest-GPU seam. Deliberately minimal, like
/// <c>VulkanGuestGpuBackend</c>: MSL codegen plus the command stream, with no
/// snapshot or translated-draw surface — the render host serves the shared
/// RenderExecutor directly.
/// </summary>
[SupportedOSPlatform("macos")]
internal sealed class MetalV2Backend : IGuestGpuBackend
{
    private static readonly IGuestCompiledShader DepthOnlyFragmentShader =
        new MetalCompiledGuestShader(new Gen5MslShader(
            MslFixedShaders.CreateDepthOnlyFragment(),
            "depth_only_fs",
            Gen5MslStage.Pixel,
            AttributeCount: 0));

    public string BackendName => "Metal v2";

    /// <summary>
    /// v2 builds MTLMeshRenderPipelineDescriptor pipelines and launches
    /// drawMeshThreadgroups, so the merged ES/GS stage runs as what it is.
    /// On by default on a Metal 3 device: without it
    /// the executor drops every culling primitive-shader draw, which is most of what
    /// a UE4/UE5 title renders. SHARPEMU_METAL_MESH_SHADERS=0 opts out and restores
    /// the per-vertex path for comparison.
    /// </summary>
    public bool SupportsPrimitiveShaderMeshPipelines { get; } =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_MESH_SHADERS"), "0", StringComparison.Ordinal) &&
        MetalV2Presenter.SupportsMeshPipelines();

    public bool TryCompileProgram(ShaderCompileRequest request, out IGuestCompiledShader? shader, out string error)
    {
        shader = null;
        if (!Gen5MslTranslator.TryCompileProgram(request, out var compiled, out error))
        {
            return false;
        }

        // The request is kept so the render host can drop the source once the library is
        // built and translate it again if it is ever needed (MetalCompiledGuestShader).
        shader = new MetalCompiledGuestShader(compiled, request);
        return true;
    }

    public IGuestCompiledShader GetDepthOnlyFragmentShader() => DepthOnlyFragmentShader;

    public void EnsureStarted(uint width, uint height) => MetalV2Presenter.EnsureStarted(width, height);

    public void HideSplashScreen()
    {
        MetalShaderPrewarm.Gate.Wait("splash screen");
        MetalV2Presenter.HideSplashScreen();
    }

    public void Submit(byte[] bgraFrame, uint width, uint height) =>
        MetalV2Presenter.Submit(bgraFrame, width, height);

    // v2 has no fixed-function guest-draw path; the executor renders every draw.
    public void SubmitGuestDraw(GuestDrawKind drawKind, uint width, uint height)
    {
        _ = drawKind;
        _ = width;
        _ = height;
    }

    public bool TrySubmitGuestImage(
        int videoOutHandle,
        int displayBufferIndex,
        ulong address,
        uint width,
        uint height,
        uint pitchInPixel,
        ulong flipRequestId)
    {
        _ = address;
        _ = width;
        _ = height;
        _ = pitchInPixel;
        MetalShaderPrewarm.Gate.Wait("first flip");
        if (!MetalV2Presenter.IsStarted)
        {
            return false;
        }

        Interlocked.Increment(ref _flipSubmissions);
        // The flip queues behind the graphics submissions before it.
        var stream = MetalV2Presenter.Host.CommandStream;
        if (!stream.TryEnqueueFlipPreparation(videoOutHandle, displayBufferIndex, flipRequestId))
        {
            Interlocked.Increment(ref _flipsRefused);
            return false;
        }

        stream.Wake();
        return true;
    }

    public void SubmitCommandStream(
        ICpuMemory memory,
        uint queue,
        ulong address,
        uint dwordCount,
        ulong submissionId,
        object? geometrySnapshots)
    {
        MetalV2Presenter.EnsureStarted(1280, 720);
        MetalShaderPrewarm.Gate.Wait("first GPU submission");
        var host = MetalV2Presenter.EnsureHost();
        if (queue == 0)
        {
            host.CommandStream.EnqueueGraphics(address, dwordCount, submissionId, geometrySnapshots);
        }
        else
        {
            host.CommandStream.EnqueueCompute(queue, address, dwordCount, submissionId, geometrySnapshots);
        }

        host.CommandStream.Wake();
        _ = memory;
    }

    public IdleOutcome SubmitDone(ICpuMemory memory)
    {
        _ = memory;
        if (HostSessionControl.IsShutdownRequested)
        {
            // The last frame boundary of a session is where the worker's teardown is
            // ordered before the runtime releases the guest page guard.
            MetalV2Presenter.StopWorker();
            return IdleOutcome.Cancelled;
        }

        if (!MetalV2Presenter.IsStarted)
        {
            return IdleOutcome.Completed;
        }

        // The guest's own view of the backend: how long its frame boundary blocks.
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        var outcome = MetalV2Presenter.Host.CommandStream.Done();
        ReportGuestWait(System.Diagnostics.Stopwatch.GetTimestamp() - started);
        return outcome;
    }

    public IdleOutcome BeginSubmitDone(ICpuMemory memory, out Func<IdleOutcome>? wait)
    {
        wait = null;
        if (HostSessionControl.IsShutdownRequested || !MetalV2Presenter.IsStarted)
        {
            return SubmitDone(memory);
        }

        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        var commandStream = MetalV2Presenter.Host.CommandStream;
        var outcome = commandStream.BeginDone(out var ticket);
        if (outcome != IdleOutcome.Completed || ticket == 0)
        {
            ReportGuestWait(System.Diagnostics.Stopwatch.GetTimestamp() - started);
            return outcome;
        }

        wait = () =>
        {
            var waited = commandStream.WaitDone(ticket);
            ReportGuestWait(System.Diagnostics.Stopwatch.GetTimestamp() - started);
            return waited;
        };
        return outcome;
    }

    private static long _flipSubmissions;
    private static long _flipsRefused;
    private static long _doneCalls;
    private static long _doneTicks;
    private static long _doneStatsTimestamp;

    private static void ReportGuestWait(long ticks)
    {
        Interlocked.Increment(ref _doneCalls);
        Interlocked.Add(ref _doneTicks, ticks);
        var now = System.Diagnostics.Stopwatch.GetTimestamp();
        var previous = Volatile.Read(ref _doneStatsTimestamp);
        if (previous == 0)
        {
            Interlocked.CompareExchange(ref _doneStatsTimestamp, now, 0);
            return;
        }

        if (now - previous < 2 * System.Diagnostics.Stopwatch.Frequency ||
            Interlocked.CompareExchange(ref _doneStatsTimestamp, now, previous) != previous)
        {
            return;
        }

        var calls = Interlocked.Exchange(ref _doneCalls, 0);
        var waited = Interlocked.Exchange(ref _doneTicks, 0);
        var seconds = (now - previous) / (double)System.Diagnostics.Stopwatch.Frequency;
        var culture = System.Globalization.CultureInfo.InvariantCulture;
        var waitMs = (waited * 1000.0 / System.Diagnostics.Stopwatch.Frequency).ToString("F1", culture);
        var flips = Interlocked.Exchange(ref _flipSubmissions, 0);
        var refused = Interlocked.Exchange(ref _flipsRefused, 0);
        var captured = Interlocked.Exchange(ref MetalRenderHost.FlipsCaptured, 0);
        var noBuffer = Interlocked.Exchange(ref MetalRenderHost.FlipsWithoutBuffer, 0);
        var discarded = Interlocked.Exchange(ref MetalRenderHost.FlipsDiscarded, 0);
        var published = Interlocked.Exchange(ref MetalRenderHost.FlipsPublished, 0);
        Console.Error.WriteLine(
            $"[METAL-V2][GUEST] seconds={seconds.ToString("F2", culture)} submit_done={calls} " +
            $"submit_done_wait_ms={waitMs} flip_submissions={flips} flips_refused={refused} " +
            $"flips_captured={captured} flips_no_buffer={noBuffer} flips_discarded={discarded} " +
            $"flips_published={published}");
    }

    public void RunAfterPendingCommandStreams(Action work)
    {
        ArgumentNullException.ThrowIfNull(work);
        if (MetalV2Presenter.IsStarted &&
            MetalV2Presenter.Host.Relay.TryRunAfterAcceptedCommandStreams(work))
        {
            return;
        }

        work();
    }

    public void RegisterKnownDisplayBuffer(ulong address, uint guestFormat)
    {
        // The image cache finds a display surface by its guest range like any other
        // image, so no address needs registering up front.
        _ = address;
        _ = guestFormat;
    }

    public bool IsGpuGuestImageAvailable(ulong address, uint format, uint numberType)
    {
        _ = address;
        _ = format;
        _ = numberType;
        return false;
    }

    public void CountShaderCompilation()
    {
        if (MetalV2Presenter.IsStarted)
        {
            MetalV2Presenter.Host.CountShaderCompilation();
        }
    }

    public (long Draws, double DrawMs, long Pipelines, long ShaderCompilations) ReadAndResetPerfCounters() =>
        MetalV2Presenter.IsStarted ? MetalV2Presenter.Host.ReadAndResetPerfCounters() : (0, 0, 0, 0);

    public void RequestClose() => MetalV2Presenter.RequestClose();
}
