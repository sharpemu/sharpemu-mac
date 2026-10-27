// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;

namespace SharpEmu.Libs.VideoOut;

/// <summary>
/// Self-time accounting for the render thread, enabled with
/// SHARPEMU_PROFILE_RENDER=1 or the unified performance profile. The existing
/// videoout counters report how much work was done (draws, pipelines, SPIR-V)
/// but not where the render thread's second went, which is the number that
/// decides whether a low frame rate is the emulator recording commands, the
/// GPU executing them, or neither.
///
/// Scopes nest: entering a phase suspends the enclosing one and resumes it on
/// dispose, so a <see cref="Phase.QueueSubmit"/> inside
/// <see cref="Phase.Flush"/> is never counted twice.
///
/// Backend contract: the phases a graphics backend package measures, plus the
/// report provider seam, are public here; everything the shared engine charges
/// for itself stays internal.
/// </summary>
public static class RenderPhaseProfile
{
    // Backend-owned cache counters append themselves to the per-window report. The shared
    // profile cannot name them: the resource caches they measure live in the graphics backend
    // package, which the shared engine does not reference.
    private static readonly List<Func<string>> ReportProviders = [];

    /// <summary>
    /// Appends a backend-owned counter block to the per-window report.
    /// </summary>
    /// <remarks>
    /// Backend contract: the resource caches these counters measure live in the
    /// graphics backend package, which the shared engine does not reference, so
    /// the backend hands in the formatter instead.
    /// </remarks>
    public static void RegisterReportProvider(Func<string> provider)
    {
        ArgumentNullException.ThrowIfNull(provider);
        lock (ReportProviders)
        {
            if (!ReportProviders.Contains(provider))
            {
                ReportProviders.Add(provider);
            }
        }
    }

    /// <summary>The render-thread phases the profile charges time to.</summary>
    public enum Phase
    {
        /// <summary>Outside any measured phase — loop overhead.</summary>
        Unattributed = 0,
        /// <summary>Parked because no command stream or newer flip exists.</summary>
        Idle,
        IdleBlockedCommands,
        IdlePendingPresentation,
        IdleNoQueuedWork,
        WindowEventPolling,
        WindowEventHandling,
        /// <summary>Blocked on the frame slot's fence: the GPU is behind.</summary>
        FrameSlotWait,
        /// <summary>Reaping completed guest submissions (fence polls).</summary>
        Collect,
        /// <summary>Selecting the presentation to show this iteration.</summary>
        TakePresentation,
        /// <summary>Running one slice of the guest command stream.</summary>
        CommandStream,
        CommandMemorySync,
        CommandMemoryRead,
        CommandGpuWait,
        CommandMemoryTransfer,
        CommandEndOfPipe,
        CommandDrawTranslation,
        CommandDispatchTranslation,
        GeometrySnapshotValidation,
        CommandDrawStateCreation,
        VertexProgramSetup,
        VertexProgramEvaluation,
        VertexMetadataResolution,
        PixelProgramSetup,
        PixelProgramEvaluation,
        GraphicsProgramCache,
        GraphicsBindingDescription,
        StageResourceDescription,
        ResourceSnapshotCreation,
        VertexInputDescription,
        DrawTargetResolution,
        DrawAttachmentPreparation,
        DrawDynamicStatePreparation,
        DrawDynamicStateRecording,
        DrawRenderingSetup,
        DrawExecutor,
        DrawProgramResolution,
        DrawResourcePreparation,
        DrawVertexBufferAcquisition,
        DrawIndexBufferAcquisition,
        BufferMappedRangeValidation,
        BufferAcquisitionChecks,
        BufferCacheLookup,
        BufferDirtySynchronization,
        BufferDirtySyncWritten,
        BufferDirtySyncTexel,
        BufferDirtySyncUpload,
        BufferStreamUpload,
        BufferStagingUpload,
        DrawVertexShaderSetup,
        DrawVertexEvaluation,
        DrawBindingAssembly,
        ProgramPreparation,
        // The part of program preparation that holds the pipeline cache's own lock.
        ProgramPrepareLocked,
        ProgramSourceRead,
        VertexInputResolution,
        PixelInputResolution,
        ProgramCacheLookup,
        ProgramPermutationLookup,
        ProgramResourceAssembly,
        ResourceMaterialization,
        ProgramCompile,
        PipelineCreation,
        DescriptorPreparation,
        DescriptorCommit,
        Draw,
        /// <summary>Closing and submitting the batched guest command buffer.</summary>
        Flush,
        /// <summary>vkQueueSubmit itself.</summary>
        QueueSubmit,
        /// <summary>vkAcquireNextImageKHR.</summary>
        Acquire,
        /// <summary>Recording + submitting the presentation command buffer.</summary>
        Present,
        /// <summary>vkQueuePresentKHR.</summary>
        QueuePresent,
        BufferFaults,
        ImageReadback,
        ImageCollect,
        BufferCollect,
        ImageLookup,
        ImageAcquire,
        ImageCreate,
        ImageDelete,
        ImageRefresh,
        ImageUpload,
        ImageDownload,
        ImageOverlap,
        ImageTracking,
        ImageTiling,
        ImageTransitions,
        DrawResources,
        BufferResources,
        DescriptorSetup,
        PipelineSetup,
        GraphicsPipelineDriver,
        ComputePipelineDriver,
        DrawRecording,
        ResourceDestroy,
        ImageVersions,
        QueueRelay,
        WindowLoop,
        WindowEvents,
        CursorUpdate,
        GamepadPoll,
        WindowState,
        WindowDelay,
        QueueContext,
        PresentationPreparation,
        GpuCompletionWait,
        SubmissionCapacity,
        CompletedSubmissionCleanup,
        MovieFramePolling,
        ProgramPrefetchWait,
        ProgramPrefetchValidation,
        ProgramPrefetchOrderWait,
        ProgramPrefetchDrainWait,
        ProgramPrefetchPauseWait,
        ProgramSynchronousResolution,
        GraphicsResolveLockWait,
        ComputeResolveLockWait,
        ProgramSharedGateWait,
        PipelineSharedGateWait,
        DeferredDrawTranslation,
        DeferredDispatchTranslation,
        ComputePipelineWait,
        ProgramDecode,
        ProgramPlanning,
        ProgramLayoutRequest,
        ProgramTranslation,
        DccClearScan,
        GuestWindowResolve,
        CommandProcessorWrite,
        OcclusionDump,
        ShaderHeaderLookup,
        ImageViewLookup,
        GraphicsStateApply,
        ArgumentBufferWrite,
        ResourceBinding,
        RenderPassBegin,
        PendingGpuWriteQuery,
        DccRearm,
        GpuWriteFlush,
        MappedSizeClamp,
        StorageBufferResolve,
        StorageDeferredLabels,
        StorageSharedLookup,
        StorageStoreOrder,
        // The command stream's own time on the Metal worker, split by what it was doing.
        // Exclusive: nested scopes (draw translation, waits, label encoding) are carved out.
        CommandPacketRead,
        CommandRegisterWrite,
        CommandDrawEntry,
        CommandFlush,
        CommandLabelEncode,
        CommandCpBookkeeping,
        // Parts of MetalRenderHost.BindResources (ResourceBinding keeps what is left).
        BindingImageResolve,
        BindingTextureAcquire,
        BindingBufferCheck,
        BindingWrittenBuffer,
        BindingWrittenInvalidate,
        BindingWrittenNote,
        WriteNoteOverlays,
        WriteNoteStaged,
        WriteNoteRanges,
        BindingReport,
        BindingDeviceAddresses,
        Count,
    }

    public static readonly bool Enabled =
        string.Equals(
            Environment.GetEnvironmentVariable("SHARPEMU_PROFILE_RENDER"),
            "1",
            StringComparison.Ordinal) ||
        string.Equals(
            Environment.GetEnvironmentVariable("SHARPEMU_PROFILE_PERFORMANCE"),
            "1",
            StringComparison.Ordinal);

    // SHARPEMU_PROFILE_SLOW_FRAME_MS=<n> (with the profile on): a frame whose flip comes n ms or
    // more after the previous one prints its own phase breakdown at once, so a hitch can be told
    // apart - the render thread busy (and where), or idle waiting for the guest.
    private static readonly long SlowFrameTicks =
        double.TryParse(
            Environment.GetEnvironmentVariable("SHARPEMU_PROFILE_SLOW_FRAME_MS"),
            System.Globalization.CultureInfo.InvariantCulture,
            out var slowMilliseconds) && slowMilliseconds > 0
            ? (long)(slowMilliseconds * Stopwatch.Frequency / 1000.0)
            : 0;

    internal static readonly bool FrameTraceEnabled =
        Enabled &&
        (SlowFrameTicks > 0 || string.Equals(
            Environment.GetEnvironmentVariable("SHARPEMU_PROFILE_PERFORMANCE_FRAME_TRACE"),
            "1",
            StringComparison.Ordinal));

    private static readonly double _reportSeconds =
        double.TryParse(
            Environment.GetEnvironmentVariable("SHARPEMU_PROFILE_RENDER_REPORT_S"),
            System.Globalization.CultureInfo.InvariantCulture,
            out var seconds) && seconds > 0
            ? seconds
            : 5.0;

    private static readonly long[] _ticks = new long[(int)Phase.Count];
    private static readonly long[] _entries = new long[(int)Phase.Count];
    private static long _frames;
    private const int FrameTraceCapacity = 4096;
    private static long[] _framePhaseTicks => FrameTraceStorage.PhaseTicks;
    private static long[] _frameTrace => FrameTraceStorage.Records;
    private static class FrameTraceStorage
    {
        internal static readonly long[] PhaseTicks = new long[(int)Phase.Count];
        internal static readonly long[] Records = new long[FrameTraceCapacity * ((int)Phase.Count + 4)];
    }
    private static long _frameTraceCount;
    private static long _previousFrameTimestamp;
    private static long _lastSubmissionTimestamp;
    private static long _submissionCount;

    /// <summary>Marks the arrival of a guest submission in the current frame trace row.</summary>
    /// <remarks>Backend contract: only the backend sees the submission arrive.</remarks>
    public static void RecordSubmissionArrival()
    {
        if (!FrameTraceEnabled) return;
        Interlocked.Exchange(ref _lastSubmissionTimestamp, Stopwatch.GetTimestamp());
        Interlocked.Increment(ref _submissionCount);
    }

    // Keep recent presentation intervals in memory; format them only after the window stops.
    /// <summary>Flushes the per-frame trace rows collected since the last call.</summary>
    /// <remarks>Backend contract: the backend decides where a frame ends.</remarks>
    public static void WriteFrameTrace()
    {
        if (!FrameTraceEnabled) return;
        var stride = (int)Phase.Count + 4;
        var first = Math.Max(0, _frameTraceCount - FrameTraceCapacity);
        Console.Error.WriteLine($"[PERF][FRAME_TRACE] frequency={Stopwatch.Frequency} retained={_frameTraceCount - first} overwritten={first}");
        for (var sequence = first; sequence < _frameTraceCount; sequence++)
        {
            var offset = (int)(sequence % FrameTraceCapacity) * stride;
            var parts = new List<string>();
            for (var phase = 0; phase < (int)Phase.Count; phase++)
            {
                var ticks = _frameTrace[offset + 4 + phase];
                if (ticks != 0) parts.Add($"{(Phase)phase}={ticks * 1000.0 / Stopwatch.Frequency:F3}");
            }
            Console.Error.WriteLine($"[PERF][FRAME] sequence={sequence} timestamp={_frameTrace[offset]} gap_ms={_frameTrace[offset + 1] * 1000.0 / Stopwatch.Frequency:F3} last_submission={_frameTrace[offset + 2]} submissions_total={_frameTrace[offset + 3]} {string.Join(" ", parts)}");
        }
        _frameTraceCount = 0;
        _previousFrameTimestamp = 0;
    }
    internal enum CommandReadKind { Header, Payload, RegisterTable, Operand32, Operand64, Other, Block, Count }
    private static readonly long[] _commandReadCalls = new long[(int)CommandReadKind.Count];
    private static readonly long[] _commandReadBytes = new long[(int)CommandReadKind.Count];

    internal static void RecordCommandRead(CommandReadKind kind, int bytes)
    {
        if (!Enabled || _scopeDepth == 0)
        {
            return;
        }

        _commandReadCalls[(int)kind]++;
        _commandReadBytes[(int)kind] += bytes;
    }
    /// <summary>Whether <see cref="RecordImageUpload"/> would record anything.</summary>
    /// <remarks>
    /// Backend contract: lets a backend skip gathering upload timings when the
    /// profile is off or the call is not on the render thread.
    /// </remarks>
    public static bool ImageUploadDetailsEnabled => Enabled && _scopeDepth > 0;

    private readonly record struct ImageUploadKey(ulong Address, uint Width, uint Height, uint Depth,
        uint Layers, uint Levels, uint Format, uint TileMode, string Reason,
        string UploadPath, ulong WriteAddress, ulong WriteSize);

    internal sealed class ImageUploadStatistics
    {
        public long Count;
        public ulong SourceBytes;
        public long WatchTicks;
        public long SourceTicks;
        public long RecordTicks;

        public void Add(ulong sourceBytes, long watchTicks, long sourceTicks, long recordTicks)
        {
            Count++;
            SourceBytes += sourceBytes;
            WatchTicks += watchTicks;
            SourceTicks += sourceTicks;
            RecordTicks += recordTicks;
        }
    }

    private static readonly Dictionary<ImageUploadKey, ImageUploadStatistics> _imageUploads = new();
    private static ImageUploadStatistics _otherImageUploads = new();

    /// <summary>Records the cost of one guest image upload, grouped by image.</summary>
    /// <remarks>Backend contract: the backend owns the image cache that performs the upload.</remarks>
    public static void RecordImageUpload(in SharpEmu.Libs.Gpu.Images.ImageDescription description,
        string reason, long watchTicks, long sourceTicks, long recordTicks,
        string uploadPath = "unknown", ulong writeAddress = 0, ulong writeSize = 0)
    {
        if (!ImageUploadDetailsEnabled)
        {
            return;
        }

        var key = new ImageUploadKey(description.Data.Address, description.Extent.Width,
            description.Extent.Height, description.Extent.Depth, description.Resources.Layers,
            description.Resources.Levels, (uint)description.PixelFormat, (uint)description.TileMode, reason,
            uploadPath, writeAddress, writeSize);
        if (!_imageUploads.TryGetValue(key, out var statistics))
        {
            // Bound diagnostic memory when a frame creates many distinct images.
            if (_imageUploads.Count >= 256)
            {
                _otherImageUploads.Add(description.Data.Size, watchTicks, sourceTicks, recordTicks);
                return;
            }

            statistics = new ImageUploadStatistics();
            _imageUploads.Add(key, statistics);
        }

        statistics.Add(description.Data.Size, watchTicks, sourceTicks, recordTicks);
    }

    private static string FormatImageUploadStatistics(ImageUploadStatistics statistics) =>
        $"uploads={statistics.Count} source_bytes={statistics.SourceBytes} " +
        $"watch_ms={statistics.WatchTicks * 1000.0 / Stopwatch.Frequency:F2} " +
        $"source_ms={statistics.SourceTicks * 1000.0 / Stopwatch.Frequency:F2} " +
        $"record_ms={statistics.RecordTicks * 1000.0 / Stopwatch.Frequency:F2}";

    private static void ReportImageUploads()
    {
        var ranked = _imageUploads.OrderByDescending(static pair =>
            pair.Value.WatchTicks + pair.Value.SourceTicks + pair.Value.RecordTicks).ToArray();
        foreach (var (key, statistics) in ranked.Take(8))
        {
            Console.Error.WriteLine($"[PERF][IMAGE_UPLOAD] address=0x{key.Address:X16} " +
                $"size={key.Width}x{key.Height}x{key.Depth} layers={key.Layers} levels={key.Levels} " +
                $"format={key.Format} tile={key.TileMode} reason={key.Reason} path={key.UploadPath} " +
                $"write_address=0x{key.WriteAddress:X16} write_bytes={key.WriteSize} {FormatImageUploadStatistics(statistics)}");
        }

        foreach (var (_, statistics) in ranked.Skip(8))
        {
            _otherImageUploads.Count += statistics.Count;
            _otherImageUploads.SourceBytes += statistics.SourceBytes;
            _otherImageUploads.WatchTicks += statistics.WatchTicks;
            _otherImageUploads.SourceTicks += statistics.SourceTicks;
            _otherImageUploads.RecordTicks += statistics.RecordTicks;
        }

        if (_otherImageUploads.Count > 0)
        {
            Console.Error.WriteLine($"[PERF][IMAGE_UPLOAD] other=1 {FormatImageUploadStatistics(_otherImageUploads)}");
        }

        _imageUploads.Clear();
        _otherImageUploads = new ImageUploadStatistics();
    }
    private static long _windowStart = Stopwatch.GetTimestamp();
    // The render loop is single-threaded, so plain fields are enough and keep
    // the per-scope cost to two timestamp reads.
    [ThreadStatic] private static Phase _current;
    [ThreadStatic] private static long _lastTimestamp;
    [ThreadStatic] private static int _scopeDepth;

    /// <summary>The scope returned by <see cref="Measure"/>; disposing it charges the elapsed time.</summary>
    public readonly ref struct Scope
    {
        private readonly Phase _previous;
        private readonly bool _active;

        internal Scope(Phase previous)
        {
            _previous = previous;
            _active = true;
        }

        public void Dispose()
        {
            if (!_active)
            {
                return;
            }

            Charge(_previous);
            _scopeDepth--;
        }

        // Switch sequential work in this scope. Nested scopes must finish first.
        public void SwitchPhase(Phase phase)
        {
            if (!_active)
            {
                return;
            }

            Charge(phase);
            _entries[(int)phase]++;
        }
    }

    public static Scope Measure(Phase phase)
    {
        if (!Enabled)
        {
            return default;
        }

        var previous = Charge(phase);
        _scopeDepth++;
        _entries[(int)phase]++;
        return new Scope(previous);
    }

    // Cache calls on other threads must not enter the render-thread counters.
    // Check the switch first to avoid thread-local storage allocation when profiling is disabled.
    /// <summary>
    /// Like <see cref="Measure"/>, but only inside an enclosing scope, so cache
    /// work on other threads does not enter the render-thread counters.
    /// </summary>
    /// <remarks>Backend contract: the measurement entry point backends use.</remarks>
    public static Scope MeasureDetail(Phase phase) =>
        Enabled && _scopeDepth > 0 ? Measure(phase) : default;

    internal static bool DetailMeasurementsEnabled => Enabled && _scopeDepth > 0;

    /// <summary>The ticks charged to a phase in the current report window; for tests.</summary>
    internal static long PhaseTicks(Phase phase) => Volatile.Read(ref _ticks[(int)phase]);

    /// <summary>
    /// Closes out the running phase and switches to <paramref name="next"/>,
    /// returning the phase that was running.
    /// </summary>
    private static Phase Charge(Phase next)
    {
        var now = Stopwatch.GetTimestamp();
        var previous = _current;
        if (_lastTimestamp != 0)
        {
            _ticks[(int)previous] += now - _lastTimestamp;
            if (FrameTraceEnabled)
            {
                _framePhaseTicks[(int)previous] += now - _lastTimestamp;
            }
        }

        _lastTimestamp = now;
        _current = next;
        return previous;
    }

    /// <summary>
    /// Drives the report from the render thread without counting a frame.
    /// </summary>
    /// <remarks>
    /// A title that presents nothing is the one whose breakdown is wanted most — where the
    /// thread's second goes is the whole question when no frame comes out — and a report
    /// driven only by presented frames never runs there. Callers pass through the same
    /// window check, so this costs a timestamp per call.
    /// </remarks>
    public static void RecordProgress()
    {
        if (!Enabled) return;
        ReportIfWindowElapsed();
    }

    /// <summary>Called once per presented frame; also drives the report.</summary>
    public static void RecordFrame()
    {
        if (!Enabled)
        {
            return;
        }

        _frames++;
        Charge(_current);
        var now = Stopwatch.GetTimestamp();
        if (FrameTraceEnabled)
        {
            var traceOffset = (int)(_frameTraceCount % FrameTraceCapacity) * ((int)Phase.Count + 4);
            _frameTrace[traceOffset] = now;
            _frameTrace[traceOffset + 1] = _previousFrameTimestamp == 0 ? 0 : now - _previousFrameTimestamp;
            _frameTrace[traceOffset + 2] = Interlocked.Read(ref _lastSubmissionTimestamp);
            _frameTrace[traceOffset + 3] = Interlocked.Read(ref _submissionCount);
            if (SlowFrameTicks > 0 && _previousFrameTimestamp != 0 && now - _previousFrameTimestamp >= SlowFrameTicks)
            {
                WriteSlowFrame(now - _previousFrameTimestamp);
            }

            Array.Copy(_framePhaseTicks, 0, _frameTrace, traceOffset + 4, (int)Phase.Count);
            Array.Clear(_framePhaseTicks);
            _previousFrameTimestamp = now;
            _frameTraceCount++;
        }

        ReportWindow(now);
    }

    private static void WriteSlowFrame(long gapTicks)
    {
        var phases = Enumerable.Range(0, (int)Phase.Count)
            .Where(phase => _framePhaseTicks[phase] > 0)
            .OrderByDescending(phase => _framePhaseTicks[phase])
            .Take(12)
            .Select(phase => $"{(Phase)phase}={_framePhaseTicks[phase] * 1000.0 / Stopwatch.Frequency:F1}");
        Console.Error.WriteLine(
            $"[PERF][SLOW_FRAME] unix_ms={DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()} " +
            $"gap_ms={gapTicks * 1000.0 / Stopwatch.Frequency:F1} {string.Join(" ", phases)}");
    }

    private static void ReportIfWindowElapsed()
    {
        Charge(_current);
        ReportWindow(Stopwatch.GetTimestamp());
    }

    private static void ReportWindow(long now)
    {
        var elapsedTicks = now - _windowStart;
        if (elapsedTicks < _reportSeconds * Stopwatch.Frequency)
        {
            return;
        }

        _windowStart = now;
        var seconds = elapsedTicks / (double)Stopwatch.Frequency;
        var frames = _frames;
        _frames = 0;

        var parts = new List<(Phase Phase, double Percent, long Entries, double Milliseconds)>((int)Phase.Count);
        var accounted = 0L;
        for (var index = 0; index < (int)Phase.Count; index++)
        {
            var phaseTicks = _ticks[index];
            _ticks[index] = 0;
            var entries = _entries[index];
            _entries[index] = 0;
            accounted += phaseTicks;
            if (phaseTicks <= 0)
            {
                continue;
            }

            parts.Add(((Phase)index, phaseTicks * 100.0 / elapsedTicks, entries,
                phaseTicks * 1000.0 / Stopwatch.Frequency));
        }

        parts.Sort(static (left, right) => right.Percent.CompareTo(left.Percent));
        Console.Error.WriteLine(
            $"[PERF][RENDER] {seconds:F1}s fps={frames / seconds:F1} " +
            $"covered={accounted * 100.0 / elapsedTicks:F0}% " +
            string.Join(
                " ",
                parts.Select(part =>
                    $"{part.Phase}={part.Percent:F1}%" +
                    (part.Entries > 0 ? $"/n{part.Entries}" : string.Empty))));

        Console.Error.WriteLine(
            $"[PERF][RENDER_MS] window_s={seconds:F1} frames={frames} " +
            string.Join(" ", parts.Select(part => $"{part.Phase}={part.Milliseconds:F2}ms/n{part.Entries}")));

        GpuWaitSiteProfile.Report();
        ComputeCompileProfile.Report();
        ReportImageUploads();
        BufferUploadProfile.Report();
        Console.Error.WriteLine(SharpEmu.ShaderCompiler.Resources.ResourceMaterializationCache.TakeReport());
        Console.Error.WriteLine(SharpEmu.ShaderCompiler.Resources.ResourcePlanBuildCounters.TakeReport());
        foreach (var provider in ReportProviders)
        {
            Console.Error.WriteLine(provider());
        }
        SharpEmu.ShaderCompiler.Resources.ResourceMaterializationProfile.WriteReport();
        DescriptorStallProfile.WriteReport();
        SharpEmu.Libs.Diagnostics.AgcRegisterPacketProfile.WriteReport();
        SharpEmu.HLE.GpuMemory.GpuMemoryAccessProfile.WriteReport();
        WindowPollProfile.Report();
        SharpEmu.HLE.GuestMemory.GuestMemoryProfile.WriteReport();
        var commandReads = new List<string>();
        for (var index = 0; index < (int)CommandReadKind.Count; index++)
        {
            if (_commandReadCalls[index] != 0)
            {
                commandReads.Add($"{(CommandReadKind)index}={_commandReadCalls[index]}/bytes{_commandReadBytes[index]}");
            }
            _commandReadCalls[index] = 0;
            _commandReadBytes[index] = 0;
        }
        if (commandReads.Count != 0)
        {
            Console.Error.WriteLine($"[PERF][COMMAND_READS] window_s={seconds:F1} {string.Join(" ", commandReads)}");
        }
    }

}
