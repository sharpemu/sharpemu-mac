// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;

namespace SharpEmu.Gpu.Metal.V2;

// SHARPEMU_METAL_V2_UPLOAD_TRACE=1: which images are uploaded again and again, and why. The CHURN line counts
// uploads by cause only; a texture the guest's CPU rewrites every frame, or one an unrelated write keeps
// invalidating, costs an upload and often an ended render pass each time, and only its address says which it is.
// One [METAL-V2][UPLOAD-TRACE] line per report interval: the images with the most uploaded bytes whose cause is
// not an alias run (those have their own trace), with count, bytes, cause, format, extent, levels and address.
internal sealed partial class MetalImageCache
{
    private static readonly bool UploadTraceEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_UPLOAD_TRACE") == "1";

    private readonly Dictionary<(ulong Address, ulong Size, MetalImageDirtyCause Cause, bool Run), (long Count, long Bytes, string Shape)>
        _uploadTrace = [];

    private void NoteUploadForTrace(MetalCachedImage image, MetalImageDirtyCause cause, ulong bytes, bool run)
    {
        if (!UploadTraceEnabled || cause == MetalImageDirtyCause.AliasWrite)
        {
            return;
        }

        ref readonly var description = ref image.Description;
        var key = (description.Data.Address, description.Data.Size, cause, run);
        lock (_uploadTrace)
        {
            if (_uploadTrace.TryGetValue(key, out var seen))
            {
                _uploadTrace[key] = (seen.Count + 1, seen.Bytes + (long)bytes, seen.Shape);
            }
            else
            {
                _uploadTrace[key] = (1, (long)bytes,
                    $"{description.PixelFormat} {description.Extent.Width}x{description.Extent.Height} " +
                    $"levels={description.Resources.Levels} layers={description.TransferLayers} tile={description.TileMode}");
            }
        }
    }

    /// <summary>Writes the interval's upload trace line, if the trace is on and anything was uploaded; resets it.</summary>
    internal void WriteUploadTrace()
    {
        if (!UploadTraceEnabled)
        {
            return;
        }

        var text = new StringBuilder();
        lock (_uploadTrace)
        {
            if (_uploadTrace.Count == 0)
            {
                return;
            }

            long uploads = 0, total = 0;
            foreach (var entry in _uploadTrace.Values)
            {
                uploads += entry.Count;
                total += entry.Bytes;
            }

            text.Append($"[METAL-V2][UPLOAD-TRACE] images={_uploadTrace.Count} uploads={uploads} mb={total / (1024.0 * 1024.0):F1} top=[");
            var shown = 0;
            foreach (var (key, entry) in _uploadTrace.OrderByDescending(pair => pair.Value.Bytes).Take(12))
            {
                text.Append(shown++ == 0 ? string.Empty : "; ");
                text.Append($"n={entry.Count} kb={entry.Bytes / 1024} cause={key.Cause}{(key.Run ? " run" : string.Empty)} " +
                    $"{entry.Shape} addr=0x{key.Address:X}+0x{key.Size:X}");
            }

            text.Append(']');
            _uploadTrace.Clear();
        }

        Console.Error.WriteLine(text.ToString());
    }
}
