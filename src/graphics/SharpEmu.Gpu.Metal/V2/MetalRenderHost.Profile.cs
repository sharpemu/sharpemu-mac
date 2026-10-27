// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using SharpEmu.ShaderCompiler.Metal;

namespace SharpEmu.Gpu.Metal.V2;

// Per-draw and per-dispatch GPU timing, behind SHARPEMU_METAL_V2_PROFILE_SHADERS=1.
//
// Apple silicon reports no support for MTLCounterSamplingPointAtDrawBoundary or
// AtDispatchBoundary (only AtStageBoundary, which times a whole encoder), so a draw
// cannot be timed inside a shared encoder. The profiler therefore takes the documented
// fallback: it commits the command buffer after every draw or dispatch, waits for it,
// and reads GPUEndTime - GPUStartTime, which is that one item's GPU execution. Only the
// profiling mode pays for it; nothing here runs when the switch is off.
internal sealed partial class MetalRenderHost
{
    // Diagnostic only: stage-boundary counters cannot time individual dispatches
    // inside one compute encoder. Split after each dispatch, without committing or
    // waiting, to identify its cost. This changes scheduling and is not a frame-time
    // benchmark; normal pass profiling leaves encoder boundaries untouched.
    private static readonly bool SplitComputeProfile =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_PROFILE_SPLIT_COMPUTE") == "1";

    private static readonly nint SelCommandBufferStatus = MetalNative.Selector("status");

    // The pipeline entry's profile label, resolved once per pipeline so a draw only
    // concatenates its own shape onto it.
    private static string ProfileLabelFor(MetalShaderModule vertex, MetalShaderModule? pixel, Gen5MslMeshInfo? mesh)
    {
        var stage = mesh is not null ? "mesh" : vertex.Shader.Stage.ToString().ToLowerInvariant();
        var label = $"stage={stage} vs=0x{vertex.Hash:X16}";
        if (pixel is not null)
        {
            label += $" ps=0x{pixel.Hash:X16}";
        }

        if (mesh is not null)
        {
            label += $" mesh_threads={mesh.ThreadsPerThreadgroup} max_vertices={mesh.MaxVertices}";
        }

        return label;
    }

    /// <summary>
    /// Closes the command buffer this draw or dispatch was encoded into, waits for it,
    /// and charges its GPU execution time to the bound pipeline's shaders. Called at the
    /// end of every encode path; a no-op unless the profiling switch is set.
    /// </summary>
    private void ProfileWork(MetalPipelineEntry entry, string kind, WorkShape shape)
    {
        // Remembered even when the profiler is off: it costs two field writes and it is
        // what names the shaders in a slow command buffer's SLOW-WAIT line.
        _lastWorkKind = kind;
        _lastWorkLabel = entry.Profile;
        var split = PassProfileEnabled && SplitComputeProfile && kind.StartsWith("dispatch", StringComparison.Ordinal);
        if (PassProfileEnabled)
        {
            PassProfileNoteWork(split ? $"{entry.Profile} {shape}" : entry.Profile, kind);
        }
        if (DumpTargetsEnabled && kind.StartsWith("draw", StringComparison.Ordinal))
        {
            NoteDrawState(kind, shape);
        }
        else if (DumpTargetsEnabled)
        {
            NoteDispatchState(entry, kind, shape);
        }

        if (split)
        {
            PassProfileNoteEnd("profile-dispatch");
            EndComputeEncoder();
        }

        ProfileCommandBuffer(kind, entry.Profile ?? "stage=?", shape);
    }

    // What a draw or dispatch looked like, formatted only when a profiler or the target dump
    // asks for it: the counts are kept raw so a draw that nobody reads builds no string.
    private readonly struct WorkShape
    {
        private enum Form : byte { Text, Vertices, Indices, Groups, Threads }

        private readonly Form _form;
        private readonly string? _text;
        private readonly uint _first;
        private readonly uint _second;
        private readonly uint _third;
        private readonly MtlSize _threads;

        private WorkShape(Form form, string? text, uint first, uint second, uint third, MtlSize threads)
        {
            _form = form;
            _text = text;
            _first = first;
            _second = second;
            _third = third;
            _threads = threads;
        }

        public static implicit operator WorkShape(string text) => new(Form.Text, text, 0, 0, 0, default);

        public static WorkShape Vertices(uint vertices, uint instances) =>
            new(Form.Vertices, null, vertices, instances, 0, default);

        public static WorkShape Indices(uint indices, uint instances) =>
            new(Form.Indices, null, indices, instances, 0, default);

        public static WorkShape Groups(uint x, uint y, uint z, MtlSize threads) =>
            new(Form.Groups, null, x, y, z, threads);

        public static WorkShape Threads(MtlSize threads) => new(Form.Threads, null, 0, 0, 0, threads);

        public override string ToString() => _form switch
        {
            Form.Vertices => $"vertices={_first} instances={_second}",
            Form.Indices => $"indices={_first} instances={_second}",
            Form.Groups => $"groups={_first}x{_second}x{_third} " +
                $"threadgroup={_threads.Width}x{_threads.Height}x{_threads.Depth}",
            Form.Threads => $"threadgroup={_threads.Width}x{_threads.Height}x{_threads.Depth}",
            _ => _text ?? string.Empty,
        };
    }

    private string _lastWorkKind = string.Empty;
    private string? _lastWorkLabel;

    /// <summary>The last draw or dispatch encoded, for a diagnostic that found a slow
    /// command buffer and has to say what was in it.</summary>
    internal string LastWorkDescription =>
        _lastWorkLabel is null ? "none" : $"{_lastWorkKind} {_lastWorkLabel}";

    /// <summary>
    /// What an image upload or writeback cost on the GPU. The cache calls this once a
    /// transfer is encoded and its encoders are closed, so the draw that follows is not
    /// charged for the tiler run that preceded it.
    /// </summary>
    void IMetalSubmissionContext.ProfileEncoded(string label)
    {
        // Remembered unconditionally, like _lastWorkLabel: it is what names the producer
        // in a write-guard violation and in a slow command buffer's report.
        _lastTransferLabel = label;
        if (!_transferInSharedEncoder)
        {
            // A transfer recorded into the open compute encoder was counted as work of that
            // pass (EndSharedCompute); only one in encoders of its own sits between passes.
            PassProfileNoteBetween(label);
        }

        _slotTrap?.NoteBinding($"transfer {label}", 0, 0, 0, 0, 0);
        ProfileCommandBuffer("transfer", label, string.Empty);
    }

    private string? _lastTransferLabel;

    // The label is read by the shader and pass profilers, and by the write guard and slot trap
    // that name the last transfer (LastTransferDescription is read only with the guard on).
    bool IMetalSubmissionContext.WantsEncodedLabels =>
        MetalShaderProfile.Enabled || PassProfileEnabled || _guard is not null || _slotTrap is not null;

    /// <summary>The last image transfer, fill or copy encoded, for a diagnostic that has
    /// to name which one wrote a guest byte.</summary>
    internal string LastTransferDescription => _lastTransferLabel ?? "none";

    private void ProfileCommandBuffer(string kind, string label, WorkShape shape)
    {
        if (!MetalShaderProfile.Enabled)
        {
            return;
        }

        MetalV2Diagnostics.InfoOnce(
            "shader-profile",
            "SHARPEMU_METAL_V2_PROFILE_SHADERS=1: one command buffer per draw and dispatch, " +
            "committed and waited; frame rate in this mode is meaningless, per-shader GPU time is not.");
        var transfers = Interlocked.Exchange(ref _profileTransfers, 0);
        FlushAndWaitForGpu();
        // FlushCommandBuffer parks the committed buffer in _previousCommandBuffer; the
        // next flush is what reports and releases it, so reading it here is safe.
        var buffer = _previousCommandBuffer;
        if (buffer == 0)
        {
            return;
        }

        // The shared event is signalled from inside the command buffer, so the wait
        // above returns before the driver has moved the buffer to Completed and its
        // GPU timestamps are not readable yet. The profiler wants the real status, and
        // this buffer holds one draw, so waiting on the buffer itself costs nothing.
        MetalNative.SendVoid(buffer, SelWaitUntilCompleted);
        var key = $"{kind} {label}";
        var shapeText = shape.ToString();
        var detail = transfers == 0 ? shapeText : $"{shapeText} transfers={transfers}";
        var status = MetalV2Native.SendValue(buffer, SelCommandBufferStatus);
        if (status != CommandBufferStatusCompleted)
        {
            var error = MetalNative.Send(buffer, MetalNative.Selector("error"));
            MetalShaderProfile.NoteFailure(
                key, detail, status, error == 0 ? "no error object" : MetalNative.DescribeError(error));
            return;
        }

        var start = MetalNative.SendDouble(buffer, SelGpuStartTime);
        var end = MetalNative.SendDouble(buffer, SelGpuEndTime);
        MetalShaderProfile.Note(key, detail, Math.Max(end - start, 0.0));
    }

    private long _profileTransfers;

    /// <summary>Counts image uploads/writebacks encoded into the draw's command buffer,
    /// so a slow sample can be told apart from a slow tiler run sharing the buffer.</summary>
    private void NoteProfileTransfer()
    {
        if (MetalShaderProfile.Enabled)
        {
            Interlocked.Increment(ref _profileTransfers);
        }
    }
}

// The aggregate behind the per-draw samples: totals per (kind, shaders) key, printed
// every two seconds and once more at exit. Keys are bounded by the pipeline count.
internal static class MetalShaderProfile
{
    private sealed class Bucket
    {
        public long Count;
        public double TotalSeconds;
        public double MaxSeconds;
        public string Shape = string.Empty;
        public string WorstShape = string.Empty;
    }

    private static readonly Dictionary<string, Bucket> _buckets = new(StringComparer.Ordinal);
    private static readonly Dictionary<string, long> _failures = new(StringComparer.Ordinal);
    private static readonly object _gate = new();
    private static long _reportTimestamp;
    private static bool _exitHooked;
    private static int _failureDetails;

    public static readonly bool Enabled = string.Equals(
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_PROFILE_SHADERS"), "1", StringComparison.Ordinal);

    /// <summary>How many entries one report prints (SHARPEMU_METAL_V2_PROFILE_TOP).</summary>
    private static readonly int _topCount =
        int.TryParse(
            Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_PROFILE_TOP"),
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out var top) && top > 0
            ? top
            : 8;

    public static void Note(string key, string shape, double seconds)
    {
        lock (_gate)
        {
            if (!_buckets.TryGetValue(key, out var bucket))
            {
                bucket = new Bucket();
                _buckets[key] = bucket;
                HookExitLocked();
            }

            bucket.Count++;
            bucket.TotalSeconds += seconds;
            bucket.Shape = shape;
            if (seconds > bucket.MaxSeconds)
            {
                bucket.MaxSeconds = seconds;
                bucket.WorstShape = shape;
            }

            ReportIfDueLocked();
        }
    }

    /// <summary>
    /// A draw whose own command buffer did not complete. The first such draws are
    /// printed in full with the driver's error: a per-draw split turns "a submission
    /// failed somewhere" into the exact shader and shape that failed.
    /// </summary>
    public static void NoteFailure(string key, string shape, nuint status, string error)
    {
        lock (_gate)
        {
            var failureKey = $"{key} status={status}";
            _failures[failureKey] = _failures.TryGetValue(failureKey, out var count) ? count + 1 : 1;
            HookExitLocked();
            if (_failureDetails < 32)
            {
                _failureDetails++;
                Console.Error.WriteLine(string.Create(
                    CultureInfo.InvariantCulture,
                    $"[METAL-V2][SHADER-PROF] failed {key} [{shape}] status={status} {error}"));
            }

            ReportIfDueLocked();
        }
    }

    private static void HookExitLocked()
    {
        if (_exitHooked)
        {
            return;
        }

        _exitHooked = true;
        AppDomain.CurrentDomain.ProcessExit += (_, _) => Report("exit");
    }

    private static void ReportIfDueLocked()
    {
        var now = System.Diagnostics.Stopwatch.GetTimestamp();
        if (_reportTimestamp == 0)
        {
            _reportTimestamp = now;
        }
        else if (now - _reportTimestamp >= 2 * System.Diagnostics.Stopwatch.Frequency)
        {
            _reportTimestamp = now;
            ReportLocked("interval");
        }
    }

    public static void Report(string reason)
    {
        lock (_gate)
        {
            ReportLocked(reason);
        }
    }

    private static void ReportLocked(string reason)
    {
        if (_buckets.Count == 0 && _failures.Count == 0)
        {
            return;
        }

        var culture = CultureInfo.InvariantCulture;
        var ordered = _buckets.OrderByDescending(pair => pair.Value.TotalSeconds).Take(_topCount);
        var rank = 0;
        var total = 0.0;
        var samples = 0L;
        foreach (var bucket in _buckets.Values)
        {
            total += bucket.TotalSeconds;
            samples += bucket.Count;
        }

        Console.Error.WriteLine(string.Create(
            culture,
            $"[METAL-V2][SHADER-PROF] {reason} keys={_buckets.Count} samples={samples} " +
            $"total_gpu_ms={total * 1000.0:F1}"));
        foreach (var (key, bucket) in ordered)
        {
            rank++;
            Console.Error.WriteLine(string.Create(
                culture,
                $"[METAL-V2][SHADER-PROF] #{rank} {key} n={bucket.Count} " +
                $"total_ms={bucket.TotalSeconds * 1000.0:F1} " +
                $"avg_ms={bucket.TotalSeconds * 1000.0 / bucket.Count:F2} " +
                $"max_ms={bucket.MaxSeconds * 1000.0:F2} worst=[{bucket.WorstShape}] last=[{bucket.Shape}]"));
        }

        foreach (var (failureKey, count) in _failures)
        {
            Console.Error.WriteLine(string.Create(
                culture, $"[METAL-V2][SHADER-PROF] incomplete {failureKey} n={count}"));
        }
    }
}
