// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE;
using SharpEmu.Libs.Agc;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.ShaderCompiler;

namespace SharpEmu.Libs.Gpu;

/// <summary>
/// Test seam at the backend boundary: guest command streams run inline on the submitting
/// thread through <see cref="AgcExports.HeadlessCommandStream"/>, so command-stream tests are
/// deterministic and independent of the host OS's backend. Every other member forwards to the
/// host backend, which is created only when one of those members is first used.
/// </summary>
internal sealed class HeadlessCommandStreamBackend : IGuestGpuBackend
{
    private readonly Func<ICpuMemory, AgcExports.HeadlessCommandStream> _streams;
    private readonly Lazy<IGuestGpuBackend> _host;

    public HeadlessCommandStreamBackend(
        Func<ICpuMemory, AgcExports.HeadlessCommandStream> streams,
        Func<IGuestGpuBackend> createHost)
    {
        _streams = streams;
        _host = new Lazy<IGuestGpuBackend>(createHost);
    }

    public string BackendName => GuestGpu.HostBackendName;

    public void SubmitCommandStream(ICpuMemory memory, uint queue, ulong address, uint dwordCount, ulong submissionId, object? geometrySnapshots) =>
        _streams(memory).Submit(queue, address, dwordCount, submissionId, geometrySnapshots);

    public IdleOutcome SubmitDone(ICpuMemory memory) => _streams(memory).Done();

    // The inline stream runs the boundary on the submitting thread: nothing is left to wait for.
    public IdleOutcome BeginSubmitDone(ICpuMemory memory, out Func<IdleOutcome>? wait)
    {
        wait = null;
        return SubmitDone(memory);
    }

    public void RunAfterPendingCommandStreams(Action work)
    {
        ArgumentNullException.ThrowIfNull(work);
        work();
    }

    public void EnsureStarted(uint width, uint height) => _host.Value.EnsureStarted(width, height);

    public bool TryCompileProgram(ShaderCompileRequest request, out IGuestCompiledShader? shader, out string error) =>
        _host.Value.TryCompileProgram(request, out shader, out error);

    public bool SupportsPrimitiveShaderMeshPipelines => _host.Value.SupportsPrimitiveShaderMeshPipelines;

    // The inline stream is fed the submit-time capture as before, so command-stream tests keep exercising it.
    public bool ConsumesSubmittedGeometrySnapshots => true;

    public IGuestCompiledShader GetDepthOnlyFragmentShader() => _host.Value.GetDepthOnlyFragmentShader();

    public void HideSplashScreen() => _host.Value.HideSplashScreen();

    public void Submit(byte[] bgraFrame, uint width, uint height) => _host.Value.Submit(bgraFrame, width, height);

    public void SubmitGuestDraw(GuestDrawKind drawKind, uint width, uint height) =>
        _host.Value.SubmitGuestDraw(drawKind, width, height);

    public bool TrySubmitGuestImage(
        int videoOutHandle,
        int displayBufferIndex,
        ulong address,
        uint width,
        uint height,
        uint pitchInPixel,
        ulong flipRequestId) =>
        _host.Value.TrySubmitGuestImage(videoOutHandle, displayBufferIndex, address, width, height, pitchInPixel, flipRequestId);

    public void RegisterKnownDisplayBuffer(ulong address, uint guestFormat) =>
        _host.Value.RegisterKnownDisplayBuffer(address, guestFormat);

    public bool IsGpuGuestImageAvailable(ulong address, uint format, uint numberType) =>
        _host.Value.IsGpuGuestImageAvailable(address, format, numberType);

    public void CountShaderCompilation() => _host.Value.CountShaderCompilation();

    public (long Draws, double DrawMs, long Pipelines, long ShaderCompilations) ReadAndResetPerfCounters() =>
        _host.Value.ReadAndResetPerfCounters();

    public void RequestClose()
    {
        if (_host.IsValueCreated)
        {
            _host.Value.RequestClose();
        }
    }
}
