// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.VideoOut.Overlay;

// The contract between the overlay's data plane (OverlayTelemetry, which the emulator feeds
// from the render thread, the GPU completion handlers, the HLE gateway and a thread-CPU
// sampler) and its UI (layout, widgets, rasterizer). The UI only ever sees an immutable
// OverlaySnapshot; it never reads live counters.

/// <summary>A horizontal lane of the frame timeline; also the "thread" a cost belongs to.</summary>
public enum OverlayLane : byte
{
    /// <summary>The busiest guest thread (usually the game's main/render-submit thread).</summary>
    GuestMain,
    /// <summary>Every other guest thread, aggregated.</summary>
    GuestWorkers,
    /// <summary>The emulator's render / command-translation thread.</summary>
    Render,
    /// <summary>Host GPU execution.</summary>
    Gpu,
    Count,
}

/// <summary>What kind of work a cost or segment is; picks its colour.</summary>
public enum CostKind : byte
{
    /// <summary>Guest code running (CPU).</summary>
    Guest,
    /// <summary>Emulated library (HLE) call body, excluding blocking waits.</summary>
    Hle,
    /// <summary>Emulator render-thread work (command translation, draws, resources, present).</summary>
    Render,
    /// <summary>Shader translation / compilation and pipeline creation.</summary>
    Shader,
    /// <summary>Host GPU execution.</summary>
    Gpu,
    /// <summary>Blocked: guest waits (cond/sema/equeue/sleep) and render-thread waits (fences, slots, idle).</summary>
    Wait,
    /// <summary>GC pauses, window/event handling, unattributed time.</summary>
    Other,
    Count,
}

/// <summary>One interval on the frame timeline, in milliseconds relative to the frame's start.</summary>
public readonly record struct TimelineSegment(float StartMs, float EndMs, OverlayLane Lane, CostKind Kind, int NameId);

/// <summary>One named cost accumulated over a frame. Milliseconds are self (exclusive) time.</summary>
public readonly record struct CostSample(int NameId, CostKind Kind, OverlayLane Lane, float Milliseconds, int Count);

/// <summary>CPU time of one host thread over a frame or a window.</summary>
public readonly record struct ThreadSample(int NameId, OverlayLane Lane, float BusyMs);

/// <summary>Everything measured about one presented frame. Immutable once published.</summary>
public sealed class FrameRecord
{
    public long Sequence { get; init; }
    /// <summary>Stopwatch timestamp of the frame's end (its host present).</summary>
    public long EndTimestamp { get; init; }
    /// <summary>Present-to-present interval. The timeline spans [0, FrameMs].</summary>
    public float FrameMs { get; init; }
    /// <summary>Guest flip-to-flip interval for the guest frame shown, or 0 when unknown.</summary>
    public float GuestFrameMs { get; init; }

    /// <summary>CPU (busy) time per lane over the frame interval; GPU is the union of GPU intervals.</summary>
    public float GuestMainMs { get; init; }
    public float GuestWorkersMs { get; init; }
    /// <summary>Number of worker threads with non-zero CPU time this frame.</summary>
    public int ActiveWorkers { get; init; }
    public float RenderBusyMs { get; init; }
    /// <summary>Render-thread time spent blocked (fence/slot/GPU waits) or idle for want of work.</summary>
    public float RenderWaitMs { get; init; }
    public float GpuMs { get; init; }

    /// <summary>Per-kind totals across all lanes (index by CostKind); self time, so they do not double count.</summary>
    public float[] KindMs { get; init; } = new float[(int)CostKind.Count];

    public int Draws { get; init; }
    public int Dispatches { get; init; }
    public int Submits { get; init; }
    public float GcPauseMs { get; init; }
    public int GcCollections { get; init; }
    /// <summary>True when the frame took clearly longer than its neighbours (see OverlayTelemetry).</summary>
    public bool Hitch { get; init; }
    /// <summary>Short reason for a hitch ("GC gen2", "shader compile", "GPU", …) or null.</summary>
    public string? HitchReason { get; init; }

    /// <summary>Intervals for the timeline widget; may be empty when timeline capture is off.</summary>
    public TimelineSegment[] Segments { get; init; } = [];
    /// <summary>Named costs for the top-costs widget (render phases, HLE exports, waits); unsorted.</summary>
    public CostSample[] Costs { get; init; } = [];
    /// <summary>CPU per host thread, for the CPU widget.</summary>
    public ThreadSample[] Threads { get; init; } = [];
}

/// <summary>Slow-moving machine and process figures, refreshed about once a second.</summary>
public sealed class SystemStats
{
    public static readonly SystemStats Empty = new();

    /// <summary>Guest frames per second (averaged over a few seconds).</summary>
    public double Fps { get; init; }
    /// <summary>Host presents per second (may exceed Fps when the host redraws).</summary>
    public double PresentFps { get; init; }
    /// <summary>Display refresh / frame-rate target used for the budget line, e.g. 30 or 60.</summary>
    public double TargetFps { get; init; } = 60;
    /// <summary>Whole-emulator CPU as a share of machine capacity, 0..100, NaN when unknown.</summary>
    public double CpuPercent { get; init; } = double.NaN;
    /// <summary>The same expressed in busy cores (CpuPercent * LogicalCores / 100).</summary>
    public double CpuCores { get; init; } = double.NaN;
    public int LogicalCores { get; init; } = Environment.ProcessorCount;
    /// <summary>GPU busy share, 0..100, NaN when the platform has no source.</summary>
    public double GpuPercent { get; init; } = double.NaN;
    /// <summary>Where GpuPercent comes from ("Metal intervals", "PDH", "n/a"), so it can be labelled honestly.</summary>
    public string GpuSource { get; init; } = "n/a";
    /// <summary>Process memory (macOS phys_footprint, Windows private working set, Linux RSS); null when unknown.</summary>
    public ulong? ProcessMemoryBytes { get; init; }
    public string ProcessMemoryKind { get; init; } = "n/a";
    public ulong? GpuAllocatedBytes { get; init; }
    public ulong GuestBufferCacheBytes { get; init; }
    public ulong GuestImageCacheBytes { get; init; }
    public double AllocatedMbPerSecond { get; init; }
    public int Gen0 { get; init; }
    public int Gen1 { get; init; }
    public int Gen2 { get; init; }
    public uint RenderWidth { get; init; }
    public uint RenderHeight { get; init; }
    public long ElapsedSeconds { get; init; }
    public double DrawsPerSecond { get; init; }
    /// <summary>"Metal", "Vulkan", …</summary>
    public string Backend { get; init; } = string.Empty;
    /// <summary>"macOS", "Windows", "Linux".</summary>
    public string Os { get; init; } = string.Empty;
    /// <summary>Game title or id, when known.</summary>
    public string Title { get; init; } = string.Empty;
}

/// <summary>An immutable view of recent frames plus system stats, captured for one overlay redraw.</summary>
public sealed class OverlaySnapshot
{
    public static readonly OverlaySnapshot Empty = new([], SystemStats.Empty, static _ => string.Empty);

    private readonly Func<int, string> _names;

    public OverlaySnapshot(IReadOnlyList<FrameRecord> frames, SystemStats system, Func<int, string> names)
    {
        Frames = frames;
        System = system;
        _names = names;
    }

    /// <summary>Recent frames, oldest first. At most OverlayTelemetry.HistoryCapacity.</summary>
    public IReadOnlyList<FrameRecord> Frames { get; }
    public SystemStats System { get; }
    public FrameRecord? Latest => Frames.Count > 0 ? Frames[^1] : null;

    /// <summary>The display name of a NameId (render phase, HLE export, thread).</summary>
    public string NameOf(int nameId) => _names(nameId);
}

/// <summary>Source of snapshots; OverlayTelemetry in the emulator, a simulator in tests.</summary>
public interface IOverlayDataSource
{
    /// <summary>Which optional (costly) captures the visible widgets need.</summary>
    void SetCaptureDemand(OverlayCaptureDemand demand);
    OverlaySnapshot Capture();
}

/// <summary>Optional captures; each costs CPU in the emulator, so they run only while a widget shows them.</summary>
[Flags]
public enum OverlayCaptureDemand
{
    None = 0,
    /// <summary>Per-frame render-phase totals (budget, split, top costs).</summary>
    RenderPhases = 1,
    /// <summary>Timeline segments (thread timeline widget).</summary>
    Timeline = 2,
    /// <summary>Per-export HLE timing (top costs with HLE/waits).</summary>
    HleCalls = 4,
    /// <summary>Per-thread CPU sampling (CPU widget, guest lanes).</summary>
    ThreadCpu = 8,
}

/// <summary>
/// The composited overlay image handed to a graphics backend: premultiplied BGRA8 at the
/// surface's pixel size, plus the regions that hold widgets. Backends upload and alpha-blend
/// only the regions (source One, destination OneMinusSourceAlpha). Pixels outside the regions
/// are undefined.
/// </summary>
public sealed class OverlayFrame
{
    public OverlayFrame(int width, int height)
    {
        Width = width;
        Height = height;
        Pixels = new byte[checked(width * height * 4)];
    }

    public int Width { get; }
    public int Height { get; }
    /// <summary>Row-major premultiplied BGRA8, stride Width * 4.</summary>
    public byte[] Pixels { get; }
    /// <summary>Pixel rectangles to composite this present, clipped to the surface.</summary>
    public List<OverlayRectangle> Regions { get; } = [];
    /// <summary>Increments whenever Pixels change; a backend may skip the upload when unchanged.</summary>
    public long Version { get; internal set; }
}
