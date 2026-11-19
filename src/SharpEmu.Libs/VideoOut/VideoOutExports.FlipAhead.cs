// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;

namespace SharpEmu.Libs.VideoOut;

// SHARPEMU_FLIP_AHEAD=1 (experimental, default off). A GPU-queued flip's guest-visible
// completion (flip count, process time, flip arg, current buffer, the previous buffer's
// label, the pending count and the flip events) is published once the GPU host has recorded
// the flip, committed the buffer carrying it and handed the presenter a copy of its frame
// (NoteFlipRecorded), and the GPU-queued flip before it on the handle has really completed
// on the GPU and been published. The guest is then at most one flip ahead of the GPU. The
// request itself stays in the table for the pacer, which presents or drops it as before, and
// its real GPU completion still releases the GPU slot and sets the ready timestamp. Without
// the notice (switch off, or a flip whose frame was not copied) nothing here runs.
public static partial class VideoOutExports
{
    // SHARPEMU_FRAME_OVERLAP_PROBE=1 (the unguarded measurement probe) turns it on too.
    internal static bool FlipAheadEnabled { get; set; } = string.Equals(
        Environment.GetEnvironmentVariable("SHARPEMU_FLIP_AHEAD"), "1", StringComparison.Ordinal) ||
        SharpEmu.Libs.Gpu.GpuCommands.FrameOverlapProbe.Enabled;

    // The last GPU-queued request reserved on each handle. Guarded by the state gate.
    private static readonly Dictionary<int, FlipRequest> _lastGpuQueuedFlip = new();

    private static long _flipAheadAtRecord;
    private static long _flipAheadAtPrevious;
    private static long _flipAheadLateNotices;
    private static long _flipAheadLeadTicks;
    private static long _flipAheadLeadCount;
    private static long _flipAheadReportedAt;

    // Caller holds the state gate.
    private static void NoteGpuQueuedFlipLocked(FlipRequest request)
    {
        if (_lastGpuQueuedFlip.TryGetValue(request.Handle, out var previous) &&
            previous.Outcome != FlipOutcome.Cancelled &&
            !(previous.State == FlipRequestState.Completed && previous.GpuCompleted && previous.EventsTriggered))
        {
            request.AheadPrevious = previous;
        }

        _lastGpuQueuedFlip[request.Handle] = request;
    }

    /// <summary>
    /// The GPU host recorded this flip, committed the command buffer that carries it and
    /// captured its frame as a copy the guest cannot overwrite. A notice for a request that is
    /// already past that point (its real completion came first, it was cancelled or published)
    /// is ignored, and the flip keeps today's timing.
    /// </summary>
    internal static void NoteFlipRecorded(ulong requestId)
    {
        if (!FlipAheadEnabled)
        {
            return;
        }

        int handle;
        lock (_stateGate)
        {
            if (!_flipRequests.TryGetValue(requestId, out var request) || !request.GpuQueued ||
                request.State != FlipRequestState.Reserved)
            {
                Interlocked.Increment(ref _flipAheadLateNotices);
                return;
            }

            request.AheadRecorded = true;
            handle = request.Handle;
        }

        DrainFlipAhead(handle, requestId);
    }

    // Caller holds the state gate. The real completion of a flip published ahead.
    private static void CompleteAheadFlipOnGpuLocked(FlipRequest request)
    {
        ReleaseGpuQueueSlotLocked(request);
        request.GpuCompleted = true;
        Interlocked.Add(ref _flipAheadLeadTicks, Math.Max(0, Stopwatch.GetTimestamp() - request.AheadPublishedAt));
        Interlocked.Increment(ref _flipAheadLeadCount);
        RemoveIfFinishedLocked(request);
        Monitor.PulseAll(_stateGate);
    }

    // Outside the state gate. The flip's counters were published: trigger its events, then let
    // the flip after it go ahead if it was waiting for this one.
    private static void TriggerPublishedFlip(FlipRequest request)
    {
        TriggerFlipEvents(request);
        if (!FlipAheadEnabled)
        {
            return;
        }

        lock (_stateGate)
        {
            request.EventsTriggered = true;
        }

        DrainFlipAhead(request.Handle, noticed: 0);
    }

    // Outside the state gate. Publishes, oldest first, every flip on the handle that may go
    // ahead now. noticed is the request whose recorded notice is being handled, if any.
    private static void DrainFlipAhead(int handle, ulong noticed)
    {
        if (!FlipAheadEnabled)
        {
            return;
        }

        while (true)
        {
            FlipRequest? next;
            lock (_stateGate)
            {
                next = FindFlipAheadCandidateLocked(handle);
                if (next is not null)
                {
                    // Set first: a dropped frame's request must outlive the publication until its
                    // real completion releases the GPU slot (RemoveIfFinishedLocked).
                    next.AheadPublishedAt = Stopwatch.GetTimestamp();
                    PublishFlipLocked(next);
                    Interlocked.Increment(ref next.RequestId == noticed ? ref _flipAheadAtRecord : ref _flipAheadAtPrevious);
                }
            }

            if (next is null)
            {
                break;
            }

            TriggerFlipEvents(next);
            lock (_stateGate)
            {
                next.EventsTriggered = true;
            }
        }

        ReportFlipAhead();
    }

    // Caller holds the state gate. Only the oldest unpublished request on the handle may be
    // published, so the guest sees flips in order; an unpublished CPU flip before it blocks it.
    private static FlipRequest? FindFlipAheadCandidateLocked(int handle)
    {
        FlipRequest? oldest = null;
        foreach (var request in _flipRequests.Values)
        {
            if (request.Handle == handle && request.State != FlipRequestState.Completed &&
                (oldest is null || request.RequestId < oldest.RequestId))
            {
                oldest = request;
            }
        }

        if (oldest is null || !oldest.AheadRecorded || !oldest.GpuQueued || oldest.State != FlipRequestState.Reserved)
        {
            return null;
        }

        var previous = oldest.AheadPrevious;
        if (previous is not null && previous.Outcome != FlipOutcome.Cancelled &&
            !(previous.State == FlipRequestState.Completed && previous.GpuCompleted && previous.EventsTriggered))
        {
            return null;
        }

        return oldest;
    }

    // At most one line every two seconds, and only while flips go ahead.
    private static void ReportFlipAhead()
    {
        var now = Stopwatch.GetTimestamp();
        var last = Interlocked.Read(ref _flipAheadReportedAt);
        if (now - last < 2 * Stopwatch.Frequency ||
            Interlocked.CompareExchange(ref _flipAheadReportedAt, now, last) != last)
        {
            return;
        }

        var atRecord = Interlocked.Exchange(ref _flipAheadAtRecord, 0);
        var atPrevious = Interlocked.Exchange(ref _flipAheadAtPrevious, 0);
        var late = Interlocked.Exchange(ref _flipAheadLateNotices, 0);
        var leadTicks = Interlocked.Exchange(ref _flipAheadLeadTicks, 0);
        var leadCount = Interlocked.Exchange(ref _flipAheadLeadCount, 0);
        if (last == 0 || atRecord + atPrevious + late == 0)
        {
            return;
        }

        var leadMs = leadCount == 0 ? 0 : leadTicks * 1000.0 / Stopwatch.Frequency / leadCount;
        Console.Error.WriteLine(
            $"[VIDEOOUT][FLIP-AHEAD] ahead_at_record={atRecord} ahead_at_previous_completion={atPrevious} " +
            $"late_notices={late} lead_ms_mean={leadMs.ToString("F1", System.Globalization.CultureInfo.InvariantCulture)} " +
            $"lead_samples={leadCount} seconds={((now - last) / (double)Stopwatch.Frequency).ToString("F1", System.Globalization.CultureInfo.InvariantCulture)}");
    }

    internal readonly record struct FlipAheadCounters(long AtRecord, long AtPrevious, long LateNotices, long LeadSamples);

    internal static FlipAheadCounters PeekFlipAheadCountersForTests() => new(
        Interlocked.Read(ref _flipAheadAtRecord),
        Interlocked.Read(ref _flipAheadAtPrevious),
        Interlocked.Read(ref _flipAheadLateNotices),
        Interlocked.Read(ref _flipAheadLeadCount));

    internal static void ResetFlipAheadForTests(bool enabled)
    {
        lock (_stateGate)
        {
            _lastGpuQueuedFlip.Clear();
        }

        FlipAheadEnabled = enabled;
        // Keep the report from consuming the counters while a test reads them.
        Interlocked.Exchange(ref _flipAheadReportedAt, Stopwatch.GetTimestamp() + 3600 * Stopwatch.Frequency);
        Interlocked.Exchange(ref _flipAheadAtRecord, 0);
        Interlocked.Exchange(ref _flipAheadAtPrevious, 0);
        Interlocked.Exchange(ref _flipAheadLateNotices, 0);
        Interlocked.Exchange(ref _flipAheadLeadTicks, 0);
        Interlocked.Exchange(ref _flipAheadLeadCount, 0);
    }

    internal static int GetGpuQueueCountForTests(int handle)
    {
        lock (_stateGate)
        {
            return _ports.TryGetValue(handle, out var port) ? port.GpuQueueCount : -1;
        }
    }
}
