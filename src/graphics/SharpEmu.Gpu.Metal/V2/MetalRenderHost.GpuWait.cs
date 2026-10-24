// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost
{
    private static readonly bool ProfileGpuWaits =
        Environment.GetEnvironmentVariable("SHARPEMU_PROFILE_GPU_WAITS") == "1";

    private Dictionary<string, int>? _gpuWaitSites;

    private readonly record struct GpuWaitDetail(
        int Site, MetalWaitReason Reason, bool Partial, ulong Completed, ulong Committed,
        ulong Recording, ulong Present, ulong Address, ulong Size, ulong Writer, MetalGpuWriteOrigin Origin);

    // Only the optional diagnostic takes stacks or allocates. No per-draw state is added.
    private GpuWaitDetail CaptureGpuWait(MetalWaitReason reason, bool partial)
    {
        if (!ProfileGpuWaits) return default;
        var stack = new System.Diagnostics.StackTrace(2, false).ToString();
        var sites = _gpuWaitSites ??= new(StringComparer.Ordinal);
        if (!sites.TryGetValue(stack, out var site) && sites.Count < 64)
        {
            site = sites.Count + 1;
            sites.Add(stack, site);
            Console.Error.WriteLine($"[METAL-V2][GPUWAIT-SITE] id={site}{Environment.NewLine}{stack}");
        }

        var read = reason is MetalWaitReason.ReadRange or MetalWaitReason.ReadImage;
        return new(site, reason, partial, CompletedSerial, _previousSerial,
            _commandBuffer != 0 ? _submissionSerial : 0, _perfPresentId,
            read ? _lastReadAddress : 0, read ? _lastReadSize : 0,
            reason == MetalWaitReason.ReadRange ? _gpuWrites.LastMatch.Serial : 0,
            reason == MetalWaitReason.ReadRange ? _gpuWrites.LastMatch.Origin : default);
    }

    private void ReportGpuWait(in GpuWaitDetail detail, ulong serial, long started,
        long committedAt, in MetalCompletionWait wait)
    {
        if (!ProfileGpuWaits) return;
        var ms = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        Console.Error.WriteLine(FormattableString.Invariant(
            $"[METAL-V2][GPUWAIT] site={detail.Site} reason={detail.Reason} partial={(detail.Partial ? 1 : 0)} serial={serial} completed_before={detail.Completed} committed_before={detail.Committed} recording_before={detail.Recording} present={detail.Present} read=0x{detail.Address:X}+0x{detail.Size:X} writer={detail.Writer} origin={(detail.Reason == MetalWaitReason.ReadRange ? detail.Origin.ToString() : "none")} reader=0x{SharpEmu.ShaderCompiler.Resources.MaterializingShader.Hash:X16} total_ms={(System.Diagnostics.Stopwatch.GetTimestamp() - started) * ms:F3} flush_ms={(committedAt - started) * ms:F3} signal_ms={wait.SignalTicks * ms:F3} drain_ms={wait.DrainTicks * ms:F3} last={LastWorkDescription}"));
    }
}
