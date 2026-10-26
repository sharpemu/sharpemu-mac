// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.Versioning;
using SharpEmu.GpuRemote;
using SharpEmu.HLE;
using SharpEmu.Libs.Gpu;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.ShaderCompiler;

namespace SharpEmu.GpuSplit;

/// <summary>
/// The guest process's <see cref="IGuestGpuBackend"/>: the AGC and video-out exports call it as
/// they call the Metal backend in one process, and it sends each call to the GPU process
/// (docs/macos/split-protocol.md, "Messages").
/// </summary>
/// <remarks>
/// <para>One-way calls are records, in ring order: submissions (with the view log the GPU
/// process's replica must have replayed first), export flips, the control calls. A flip is taken
/// as accepted when it is sent, as the one-process call only enqueues it; a refusal comes back as
/// a flip event. The frame boundary, the ordered video-out changes and the counters are calls; the
/// frame boundary and the brackets wait in the GPU process until it handled every record sent
/// before them.</para>
///
/// <para>What only the GPU process runs - shader compilation, its capabilities - is not asked
/// here: the command streams that need it run there. What the Metal v2 backend answers with a
/// constant (no guest-image availability, no display-buffer registration) is answered here with
/// the same constant.</para>
/// </remarks>
[SupportedOSPlatform("macos")]
public sealed class RemoteGpuBackend : IGuestGpuBackend
{
    // Under a ring record's payload cap with room for the header.
    private const int HostFramePieceBytes = 256 * 1024;

    private readonly SplitGpuSession _session;
    private int _closeSent;

    internal RemoteGpuBackend(SplitGpuSession session)
    {
        _session = session;
    }

    public string BackendName => "Metal v2 (GPU process)";

    public void EnsureStarted(uint width, uint height) =>
        SendControl(GpuSplitControl.EnsureStarted, width, height);

    public bool TryCompileProgram(ShaderCompileRequest request, out IGuestCompiledShader? shader, out string error)
    {
        shader = null;
        error = "Shaders are compiled in the GPU process.";
        return false;
    }

    public IGuestCompiledShader GetDepthOnlyFragmentShader() =>
        throw new NotSupportedException("Shaders live in the GPU process.");

    public void HideSplashScreen() => SendControl(GpuSplitControl.HideSplashScreen);

    public void Submit(byte[] bgraFrame, uint width, uint height)
    {
        ArgumentNullException.ThrowIfNull(bgraFrame);
        if (!_session.IsAlive || bgraFrame.Length == 0)
        {
            return;
        }

        // The pieces go back to back: another writer cannot put a record between them.
        var writer = _session.Writer;
        lock (writer.Gate)
        {
            for (var offset = 0; offset < bgraFrame.Length; offset += HostFramePieceBytes)
            {
                var length = Math.Min(HostFramePieceBytes, bgraFrame.Length - offset);
                var header = new HostFrameMessage
                {
                    Width = width,
                    Height = height,
                    TotalBytes = (uint)bgraFrame.Length,
                    Offset = (uint)offset,
                };
                if (!writer.Write(GpuRemoteMessage.HostFrame, in header, bgraFrame.AsSpan(offset, length)))
                {
                    return;
                }
            }
        }
    }

    // Metal v2 has no fixed-function draw path either.
    public void SubmitGuestDraw(GuestDrawKind drawKind, uint width, uint height)
    {
    }

    public bool TrySubmitGuestImage(int videoOutHandle, int displayBufferIndex, ulong address, uint width, uint height,
        uint pitchInPixel, ulong flipRequestId)
    {
        if (!_session.IsAlive)
        {
            return false;
        }

        var message = new FlipRequestMessage
        {
            VideoOutHandle = videoOutHandle,
            DisplayBufferIndex = displayBufferIndex,
            Address = address,
            Width = width,
            Height = height,
            PitchInPixel = pitchInPixel,
            HdrRequested = Libs.VideoOut.VideoOutBackendServices.IsHdrOutputRequested ? 1u : 0u,
            FlipRequestId = flipRequestId,
        };
        return _session.Writer.Write(GpuRemoteMessage.FlipRequest, in message);
    }

    public void SubmitCommandStream(ICpuMemory memory, uint queue, ulong address, uint dwordCount, ulong submissionId, object? geometrySnapshots)
    {
        var message = new SubmitCommandStreamMessage
        {
            Queue = queue,
            DwordCount = dwordCount,
            Address = address,
            SubmissionId = submissionId,
            ViewLogSequence = _session.ViewLogSequence,
        };
        _ = _session.Writer.Write(GpuRemoteMessage.SubmitCommandStream, in message);
    }

    public IdleOutcome SubmitDone(ICpuMemory memory)
    {
        if (HostSessionControl.IsShutdownRequested)
        {
            // The last frame boundary of a session: the GPU process tears its worker down.
            RequestClose();
            return IdleOutcome.Cancelled;
        }

        if (!_session.IsAlive)
        {
            return IdleOutcome.Cancelled;
        }

        return _session.Call(GpuRemoteSyncOpcode.SubmitDone, new BarrierRequest { Barrier = _session.Writer.Written },
                out IdleOutcomeResponse response, SplitGpuSession.LongCallNanoseconds)
            ? (IdleOutcome)response.Outcome
            : IdleOutcome.Failed;
    }

    // The boundary goes on the ring behind the submissions already sent, so a submission another
    // thread sends after it is ordered after it in the GPU process too. Only the wait is a call.
    public IdleOutcome BeginSubmitDone(ICpuMemory memory, out Func<IdleOutcome>? wait)
    {
        wait = null;
        if (HostSessionControl.IsShutdownRequested || !_session.IsAlive)
        {
            return SubmitDone(memory);
        }

        var message = new FrameBoundaryMessage { Ticket = (ulong)Interlocked.Increment(ref _frameBoundaryTickets) };
        if (!_session.Writer.Write(GpuRemoteMessage.FrameBoundary, in message))
        {
            return _session.IsAlive ? IdleOutcome.Failed : IdleOutcome.Cancelled;
        }

        var request = new FrameBoundaryWaitRequest { Barrier = _session.Writer.Written, Ticket = message.Ticket };
        wait = () =>
            !_session.IsAlive ? IdleOutcome.Cancelled :
            _session.Call(GpuRemoteSyncOpcode.WaitFrameBoundary, request,
                out IdleOutcomeResponse response, SplitGpuSession.LongCallNanoseconds)
                ? (IdleOutcome)response.Outcome
                : IdleOutcome.Failed;
        return IdleOutcome.Completed;
    }

    private long _frameBoundaryTickets;

    public void RunAfterPendingCommandStreams(Action work)
    {
        ArgumentNullException.ThrowIfNull(work);
        if (!_session.Relay.TryRun(GpuSplitBracket.AfterPendingCommandStreams, work))
        {
            work();
        }
    }

    public void RegisterKnownDisplayBuffer(ulong address, uint guestFormat)
    {
    }

    public bool IsGpuGuestImageAvailable(ulong address, uint format, uint numberType) => false;

    public void CountShaderCompilation() => SendControl(GpuSplitControl.CountShaderCompilation);

    public (long Draws, double DrawMs, long Pipelines, long ShaderCompilations) ReadAndResetPerfCounters() =>
        _session.Call(GpuRemoteSyncOpcode.Query, new QueryRequest { Which = (uint)GpuSplitQuery.PerfCounters },
            out QueryResponse response, SplitGpuSession.CallNanoseconds)
            ? (response.A, response.D, response.B, response.C)
            : (0, 0, 0, 0);

    public void RequestClose()
    {
        if (Interlocked.Exchange(ref _closeSent, 1) == 0)
        {
            SendControl(GpuSplitControl.RequestClose);
        }
    }

    /// <summary>Whether the GPU process's presenter drives HDR output (the presenter seam's question).</summary>
    public bool QueryHdrOutputSupport() =>
        _session.Call(GpuRemoteSyncOpcode.Query, new QueryRequest { Which = (uint)GpuSplitQuery.HdrOutputSupport },
            out QueryResponse response, SplitGpuSession.CallNanoseconds) && response.A != 0;

    private void SendControl(GpuSplitControl kind, uint width = 0, uint height = 0)
    {
        if (_session.IsAlive)
        {
            _ = _session.Writer.Write(GpuRemoteMessage.Control, new ControlMessage { Kind = (uint)kind, Width = width, Height = height });
        }
    }
}
