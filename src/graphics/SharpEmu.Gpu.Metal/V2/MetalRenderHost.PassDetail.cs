// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text.Json;
using SharpEmu.Libs.Gpu.Rendering;

namespace SharpEmu.Gpu.Metal.V2;

// Lossless metadata for the existing timestamp profiler. No encoder, draw, load or
// store is changed. Set PROFILE_PASSES=1 and PROFILE_DETAIL to a JSONL path.
internal sealed partial class MetalRenderHost
{
    private static readonly string? PassDetailPath =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_PROFILE_DETAIL");
    private bool _passDetailFailed;

    private sealed class PassAttachment
    {
        public required string Slot { get; init; }
        public required string View { get; init; }
        public required string Format { get; init; }
        public uint Width { get; init; }
        public uint Height { get; init; }
        public uint Layers { get; init; }
        public uint Samples { get; init; }
        public uint Level { get; init; }
        public uint Slice { get; init; }
        public long LogicalBytes { get; init; }
        public required string Load { get; init; }
        public string Store { get; set; } = "store";
    }

    private void WritePassDetail<T>(T value)
    {
        if (PassDetailPath is null || _passDetailFailed)
        {
            return;
        }

        try
        {
            File.AppendAllText(PassDetailPath, JsonSerializer.Serialize(value) + "\n");
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            _passDetailFailed = true;
            Console.Error.WriteLine($"[METAL-V2][PASS-DETAIL] {exception.Message}");
        }
    }

    private void DescribePassAttachments(PassRecord record, in RenderingState state)
    {
        var attachments = record.Attachments = new List<PassAttachment>();
        for (var index = 0; index < state.ColorAttachmentCount; index++)
        {
            ref readonly var color = ref state.ColorAttachments[index];
            if (color.View.Handle == 0 || index >= MaxColorAttachments)
            {
                continue;
            }

            var clears = color.IsClear || _overrideClear[index];
            attachments.Add(new PassAttachment
            {
                Slot = $"c{index}", View = color.View.Handle.ToString("X"), Format = color.Format.ToString(),
                Width = _attachmentWidths[index], Height = _attachmentHeights[index],
                Layers = Math.Max(1u, state.Layers), Samples = Math.Max(1u, state.Samples),
                Level = _attachmentLevels[index], Slice = _attachmentSlices[index],
                LogicalBytes = AttachmentBytes(in state, _attachmentTexelBytes[index]) * Math.Max(1u, state.Layers),
                Load = clears ? "clear" : _elidedLoads[index] ? "dontCare" : "load",
            });
        }

        ref readonly var depth = ref state.DepthStencilAttachment;
        foreach (var stencil in new[] { false, true })
        {
            if (depth.View.Handle == 0 || !(stencil ? depth.HasStencil : depth.HasDepth))
            {
                continue;
            }

            attachments.Add(new PassAttachment
            {
                Slot = stencil ? "stencil" : "depth", View = depth.View.Handle.ToString("X"),
                Format = stencil ? "Stencil8" : depth.Format.ToString(),
                Width = _depthAttachmentWidth, Height = _depthAttachmentHeight,
                Layers = Math.Max(1u, state.Layers), Samples = Math.Max(1u, state.Samples),
                Level = _depthAttachmentLevel, Slice = _depthAttachmentSlice,
                LogicalBytes = AttachmentBytes(in state, stencil ? 1u : _depthTexelBytes) * Math.Max(1u, state.Layers),
                Load = (stencil ? depth.StencilClear : depth.DepthClear) ? "clear" : "load",
            });
        }
    }

    private void DetailStoreAction(string slot, bool discard)
    {
        var attachment = _ppOpen?.Attachments?.Find(item => item.Slot == slot);
        if (attachment is not null)
        {
            attachment.Store = discard ? "dontCare" : "store";
        }
    }
}
