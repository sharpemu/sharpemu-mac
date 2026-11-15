// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

// Why an alias of a GPU-written image was published whole rather than over the run the
// two images share. None is the narrow publication itself.
internal enum MetalAliasFallbackReason : byte
{
    None,
    SubRangeDisabled,
    EmptyRun,
    Stencil,
    // The other image lies inside the requester, so the run is all of it.
    RunIsWholeOther,
    RunPastHalf,
    OtherDirtyOutsideRun,
    NoTexture,
    CoveringRunRefused,
    RunUnmapped,
    // The transfer refused the run and retiled nothing.
    RetileRefused,
}

internal sealed partial class MetalImageCache
{
    private const int AliasFallbackReasons = (int)MetalAliasFallbackReason.RetileRefused + 1;
    // SHARPEMU_METAL_V2_ALIAS_TRACE=1 logs the first alias declines and publications of
    // every report window with both image descriptions; =depth logs only pairs with a depth
    // image in them, ten times as many, which a frame of other aliases would otherwise crowd out.
    private static readonly string? AliasTraceMode = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_ALIAS_TRACE");
    private static readonly bool AliasTraceEnabled = AliasTraceMode is "1" or "depth";
    private static readonly bool AliasTraceDepthOnly = AliasTraceMode == "depth";
    private static readonly int AliasTraceLinesPerWindow = AliasTraceDepthOnly ? 400 : 40;

    private static bool AliasTraceSkips(MetalCachedImage image, MetalCachedImage other) =>
        AliasTraceDepthOnly && !image.Description.IsDepth && !other.Description.IsDepth;

    // Write uses first, then read uses, one slot per reason.
    private readonly long[] _aliasFallbacks = new long[2 * AliasFallbackReasons];
    private int _aliasTraceDeclines;
    private int _aliasTracePublications;

    /// <summary>
    /// The whole alias publications since the last call, by use and reason, as
    /// <c>[write:Reason=n,read:Reason=n]</c>; starts a new trace window.
    /// </summary>
    public string TakeAliasFallbacks()
    {
        Interlocked.Exchange(ref _aliasTraceDeclines, 0);
        Interlocked.Exchange(ref _aliasTracePublications, 0);
        var text = new StringBuilder("[");
        for (var slot = 0; slot < _aliasFallbacks.Length; slot++)
        {
            var count = Interlocked.Exchange(ref _aliasFallbacks[slot], 0);
            if (count == 0)
            {
                continue;
            }

            if (text.Length > 1)
            {
                text.Append(',');
            }

            text.Append(slot < AliasFallbackReasons ? "write:" : "read:")
                .Append((MetalAliasFallbackReason)(slot % AliasFallbackReasons))
                .Append('=').Append(count);
        }

        return text.Append(']').ToString();
    }

    // Counts a whole publication and, with the trace on, logs it. Caller holds the lock.
    private void NoteAliasFallback(
        bool write, MetalAliasFallbackReason reason, MetalCachedImage image, MetalCachedImage other,
        bool otherGpuModified, ulong runStart, ulong runEnd)
    {
        _aliasFallbacks[(write ? 0 : AliasFallbackReasons) + (int)reason]++;
        if (!AliasTraceEnabled || AliasTraceSkips(image, other) ||
            Interlocked.Increment(ref _aliasTraceDeclines) > AliasTraceLinesPerWindow)
        {
            return;
        }

        var because = write ? $"reason={reason}" : $"reason=ReadUse detail={reason}";
        Console.Error.WriteLine(
            $"[METAL-V2][ALIAS-TRACE] decline use={(write ? "write" : "read")} {because} " +
            AliasPairText(image, other, otherGpuModified, runStart, runEnd) + AliasUseText());
    }

    // Logs an alias publication with the trace on. Caller holds the lock.
    private void NoteAliasPublication(
        string kind, MetalCachedImage image, MetalCachedImage other, bool otherGpuModified,
        ulong runStart, ulong runEnd)
    {
        if (!AliasTraceEnabled || AliasTraceSkips(image, other) ||
            Interlocked.Increment(ref _aliasTracePublications) > AliasTraceLinesPerWindow)
        {
            return;
        }

        Console.Error.WriteLine(
            $"[METAL-V2][ALIAS-TRACE] publish kind={kind} " +
            AliasPairText(image, other, otherGpuModified, runStart, runEnd) + AliasUseText());
    }

    // The requester's use (rt-overwrite is a colour target whose pass is proven to overwrite it; rt-load one whose
    // pass loads or is not proven), and the guest bytes written back for this pair so far in this sync step.
    private string _aliasSyncRole = "other";
    private long _aliasTraceMark;

    private string AliasUseText() => $" role={_aliasSyncRole} moved_wb=0x{_writebackBytes - _aliasTraceMark:X}";

    // gpu_mod is the requester's and the other image's state when the alias was found,
    // before any writeback cleared it.
    private static string AliasPairText(
        MetalCachedImage image, MetalCachedImage other, bool otherGpuModified, ulong runStart, ulong runEnd)
    {
        var requester = image.Description.Data.Size < other.Description.Data.Size ? "small"
            : image.Description.Data.Size > other.Description.Data.Size ? "large" : "same";
        var dirty = other.IsCpuDirty ? $"0x{other.DirtyStart:X}+0x{other.DirtyEnd - other.DirtyStart:X}" : "none";
        return $"requester={requester} req=<{AliasImageText(in image.Description)}> " +
            $"other=<{AliasImageText(in other.Description)}> " +
            $"run=0x{runStart:X}+0x{(runEnd > runStart ? runEnd - runStart : 0):X} other_dirty={dirty} " +
            $"gpu_mod={image.IsGpuModified}/{otherGpuModified} seq={image.GpuWriteSequence}/{other.GpuWriteSequence}";
    }

    private static string AliasImageText(in ImageDescription description) =>
        $"{description.PixelFormat} {description.Extent.Width}x{description.Extent.Height} " +
        $"layers={description.Resources.Layers} mips={description.Resources.Levels} pitch={description.Pitch} " +
        $"tile={description.TileMode} addr=0x{description.Data.Address:X} bytes=0x{description.Data.Size:X}";
}
