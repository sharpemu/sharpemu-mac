// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

// A GPU-written image whose stale runs cover every byte it holds: guest memory (once the aliases that wrote them
// have published, as SynchronizeAliases does before every acquire's refresh) holds all of its bytes, and the texture
// owns none. TryRefreshDirtyRun used to refuse such a run for widening to the whole image, and RefreshFromGuestCore
// then refused the whole upload because the image is GPU-written (alias_run_refresh_refused), so the texture kept
// its older texels - sampled, and retiled back over the alias's bytes at its writeback. Now the run is the whole
// resident range, reloaded through the run path, and the image stops being GPU-modified; a writeback of such an
// image writes nothing. Not for a stencil-carrying image (its stencil plane is not in the runs: the refusal stays,
// logged and counted), nor past TransferPieceBytes (refused as whole-large), nor with sub-range transfers off.
// SHARPEMU_METAL_V2_WHOLE_STALE_RELOAD=0 restores the refusal, for an A/B.
internal sealed partial class MetalImageCache
{
    private static readonly bool WholeStaleReloadDefault =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_WHOLE_STALE_RELOAD") != "0";

    /// <summary>Whether a GPU-written image stale over all of its bytes reloads them whole.</summary>
    internal bool WholeStaleReloadEnabled { get; set; } = WholeStaleReloadDefault;

    private long _wholeStaleReloads, _wholeStaleWritebacksSkipped;

    /// <summary>Test hook: whole-stale reloads, and writebacks of whole-stale images that wrote nothing.</summary>
    internal (long Reloads, long WritebacksSkipped) WholeStaleCounters =>
        (Interlocked.Read(ref _wholeStaleReloads), Interlocked.Read(ref _wholeStaleWritebacksSkipped));

    // Whether the stale runs, clipped to [start, end), leave no byte of it out.
    private static bool RunsCover(ReadOnlySpan<(ulong Start, ulong End)> runs, ulong start, ulong end)
    {
        var at = start;
        foreach (var run in runs)
        {
            if (run.Start > at)
            {
                return false;
            }

            at = Math.Max(at, run.End);
            if (at >= end)
            {
                return true;
            }
        }

        return at >= end;
    }

    private bool IsWholeStale(MetalCachedImage image, ReadOnlySpan<(ulong Start, ulong End)> runs, ulong start, ulong end) =>
        WholeStaleReloadEnabled && image.IsGpuModified && image.FirstLevel == 0 && !image.Description.HasStencil &&
        end > start && RunsCover(runs, start, end);

    // The texture now equals guest memory over every byte: it owes guest memory nothing, so a later release or alias
    // publication must not retile it back.
    private void NoteWholeStaleReloaded(MetalCachedImage image)
    {
        image.ClearGpuModified();
        Interlocked.Increment(ref _wholeStaleReloads);
    }

    // A writeback of a GPU-written image stale over all of its bytes: guest memory is newer everywhere, so there is
    // nothing to write. The image stays stale (its next use reloads it) and stops being GPU-modified.
    private bool TrySkipWholeStaleWriteback(MetalCachedImage image)
    {
        if (!image.IsCpuDirty || ImageDescription.IsEmptyRange(image.Description.Data) ||
            !IsWholeStale(image, image.StaleRuns, 0, image.Description.Data.Size))
        {
            return false;
        }

        image.ClearGpuModified();
        Interlocked.Increment(ref _wholeStaleWritebacksSkipped);
        return true;
    }

    private string TakeWholeStaleText() =>
        $"whole_stale=[reloaded:{Interlocked.Exchange(ref _wholeStaleReloads, 0)}," +
        $"writeback-skipped:{Interlocked.Exchange(ref _wholeStaleWritebacksSkipped, 0)}]";
}
