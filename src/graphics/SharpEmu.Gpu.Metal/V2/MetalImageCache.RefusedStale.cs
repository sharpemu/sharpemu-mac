// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;

namespace SharpEmu.Gpu.Metal.V2;

// Why TryRefreshDirtyRun gave up on a run, for alias_run_refresh_refused_by.
internal enum MetalRunRefusal : byte
{
    None,
    SubRangeOff,
    Stencil,
    NoResidentRun,
    OverHalf,
    NoCoveringRun,
    SourceInvalid,
    NotMoved,
}

// Diagnostics only: what becomes of the texels a GPU-written image keeps when RefreshFromGuestCore refuses its
// stale run (alias_run_refresh_refused). The refusal marks the image; the mark records every later read of the
// texture (sampled, storage, a render pass that loads it, a writeback that retiles it into guest memory) and ends
// at a whole rewrite (a clear, an elided load, a kernel or deferred pass that overwrites it), a whole reload or a
// release. A rewrite of one subresource of a multi-subresource image is counted on its own and keeps the mark.
internal sealed partial class MetalImageCache
{
    private static readonly string[] RunRefusalNames =
        ["none", "subrange-off", "stencil", "no-run", "over-half", "no-cover", "source", "not-moved"];

    private sealed class RefusedMark
    {
        public ulong Start, End;
        public string Runs = string.Empty;
        public MetalRunRefusal Reason;
        public bool Read;
    }

    private enum RefusedRead
    {
        Sampled,
        SampledConservative,
        Storage,
        AttachmentLoad,
        Writeback,
        Transfer,
    }

    // Set by TryRefreshDirtyRun before each false return; read by the refusal that follows it.
    private MetalRunRefusal _runRefusal;
    private readonly long[] _runRefusedBy = new long[RunRefusalNames.Length];
    private readonly Dictionary<MetalCachedImage, RefusedMark> _refusedMarks = new(ReferenceEqualityComparer.Instance);
    private int _refusedMarkCount;
    private readonly long[] _refusedReads = new long[6];
    private long _refusedDeadWhole, _refusedReadThenWhole, _refusedPartialWrite, _refusedReloaded, _refusedReleased;

    private static bool SingleSubresource(MetalCachedImage image) =>
        image.Description.Resources.Levels <= 1 && image.Description.TransferLayers <= 1 && !image.Description.IsVolume;

    private MetalRunRefusal NoteRunRefusal(MetalCachedImage image)
    {
        var reason = _runRefusal;
        _runRefusal = MetalRunRefusal.None;
        Interlocked.Increment(ref _runRefusedBy[(int)reason]);
        var builder = new StringBuilder();
        foreach (var (start, end) in image.StaleRuns)
        {
            builder.Append(builder.Length == 0 ? "" : ",").Append($"0x{start:X}..0x{end:X}");
        }

        lock (_refusedMarks)
        {
            if (!_refusedMarks.TryGetValue(image, out var mark))
            {
                mark = new RefusedMark();
                _refusedMarks[image] = mark;
                _refusedMarkCount = _refusedMarks.Count;
            }

            mark.Start = image.DirtyStart;
            mark.End = image.DirtyEnd;
            mark.Runs = builder.ToString();
            mark.Reason = reason;
        }

        return reason;
    }

    // A read of a marked image's texture over [start, end) of its bytes (whole image when end is 0).
    private void NoteRefusedRead(MetalCachedImage image, RefusedRead role, string detail, ulong start = 0, ulong end = 0)
    {
        if (Volatile.Read(ref _refusedMarkCount) == 0)
        {
            return;
        }

        RefusedMark? mark;
        lock (_refusedMarks)
        {
            if (!_refusedMarks.TryGetValue(image, out mark) || (end > start && (end <= mark.Start || start >= mark.End)))
            {
                return;
            }

            mark.Read = true;
        }

        Interlocked.Increment(ref _refusedReads[(int)role]);
        ref readonly var description = ref image.Description;
        MetalV2Diagnostics.WarnOnce(
            $"refused-stale-read:{role}:{detail}:{description.PixelFormat}:{description.Extent.Width}x{description.Extent.Height}",
            $"A GPU-written image whose stale run was refused a reload is read ({role} {detail}): " +
            $"{description.PixelFormat} {description.Extent.Width}x{description.Extent.Height} " +
            $"levels={description.Resources.Levels} layers={description.TransferLayers} " +
            $"0x{description.Data.Address:X16}+0x{description.Data.Size:X} reason={RunRefusalNames[(int)mark.Reason]} " +
            $"stale=[{mark.Runs}].");
    }

    // An alias-run writeback condition that reads the run out of the texture: noted, never changes the outcome.
    private bool NoteRefusedReadThrough(MetalCachedImage image, ulong start, ulong end)
    {
        NoteRefusedRead(image, RefusedRead.Writeback, "alias-run", start, end);
        return true;
    }

    // A run reload over [start, end): the mark ends when it covers every byte the refusal left stale.
    private void NoteRefusedReloadedRun(MetalCachedImage image, ulong start, ulong end)
    {
        if (Volatile.Read(ref _refusedMarkCount) == 0)
        {
            return;
        }

        bool covers;
        lock (_refusedMarks)
        {
            covers = _refusedMarks.TryGetValue(image, out var mark) && start <= mark.Start && end >= mark.End;
        }

        if (covers)
        {
            NoteRefusedEnded(image, released: false);
        }
    }

    // A sampled bind: exact for a one-subresource image, counted conservatively (as reading the stale run) otherwise.
    private void NoteRefusedSampled(MetalCachedImage image)
    {
        if (Volatile.Read(ref _refusedMarkCount) != 0)
        {
            NoteRefusedRead(
                image, SingleSubresource(image) ? RefusedRead.Sampled : RefusedRead.SampledConservative, "bind");
        }
    }

    // The GPU rewrote the image: whole (every texel of every subresource), or only part of it (the mark stays).
    private void NoteRefusedRewrite(MetalCachedImage image, bool whole)
    {
        if (Volatile.Read(ref _refusedMarkCount) == 0)
        {
            return;
        }

        lock (_refusedMarks)
        {
            if (!_refusedMarks.TryGetValue(image, out var mark))
            {
                return;
            }

            if (!whole)
            {
                _refusedPartialWrite++;
                return;
            }

            if (mark.Read)
            {
                _refusedReadThenWhole++;
            }
            else
            {
                _refusedDeadWhole++;
            }

            _refusedMarks.Remove(image);
            _refusedMarkCount = _refusedMarks.Count;
        }
    }

    // The texture took the guest's bytes again over all of the resident image, or the image went.
    private void NoteRefusedEnded(MetalCachedImage image, bool released)
    {
        if (Volatile.Read(ref _refusedMarkCount) == 0)
        {
            return;
        }

        lock (_refusedMarks)
        {
            if (!_refusedMarks.Remove(image))
            {
                return;
            }

            if (released)
            {
                _refusedReleased++;
            }
            else
            {
                _refusedReloaded++;
            }

            _refusedMarkCount = _refusedMarks.Count;
        }
    }

    /// <summary>
    /// A render pass opened over an image texture's subresource with its load action final: Load reads whatever
    /// the texture holds; Clear or an elided load (DontCare under a full-cover draw) rewrites that subresource.
    /// </summary>
    internal void NoteRefusedAttachmentOpened(nint texture, uint level, uint layer, bool loads, bool depth)
    {
        if (texture == 0 || Volatile.Read(ref _refusedMarkCount) == 0)
        {
            return;
        }

        MetalCachedImage? image = null;
        lock (_refusedMarks)
        {
            foreach (var marked in _refusedMarks.Keys)
            {
                if (marked.Texture == texture)
                {
                    image = marked;
                    break;
                }
            }
        }

        if (image is null)
        {
            return;
        }

        if (loads)
        {
            NoteRefusedRead(image, RefusedRead.AttachmentLoad, $"{(depth ? "depth" : "color")}-l{level}-s{layer}");
        }
        else
        {
            NoteRefusedRewrite(image, whole: SingleSubresource(image));
        }
    }

    private string TakeRefusedStaleText()
    {
        var by = new StringBuilder();
        for (var index = 1; index < _runRefusedBy.Length; index++)
        {
            by.Append(index == 1 ? "" : ",").Append(RunRefusalNames[index]).Append(':')
                .Append(Interlocked.Exchange(ref _runRefusedBy[index], 0));
        }

        var none = Interlocked.Exchange(ref _runRefusedBy[0], 0);
        if (none != 0)
        {
            by.Append(",unset:").Append(none);
        }

        long deadWhole, readThenWhole, partial, reloaded, released;
        int outstanding;
        lock (_refusedMarks)
        {
            // An image disposed without passing ReleaseImage leaves no texture behind: its mark goes as released.
            List<MetalCachedImage>? gone = null;
            foreach (var marked in _refusedMarks.Keys)
            {
                if (marked.Texture == 0)
                {
                    (gone ??= new List<MetalCachedImage>()).Add(marked);
                }
            }

            if (gone is not null)
            {
                foreach (var marked in gone)
                {
                    _refusedMarks.Remove(marked);
                    _refusedReleased++;
                }
            }

            _refusedMarkCount = _refusedMarks.Count;
            deadWhole = _refusedDeadWhole;
            readThenWhole = _refusedReadThenWhole;
            partial = _refusedPartialWrite;
            reloaded = _refusedReloaded;
            released = _refusedReleased;
            _refusedDeadWhole = _refusedReadThenWhole = _refusedPartialWrite = _refusedReloaded = _refusedReleased = 0;
            outstanding = _refusedMarks.Count;
        }

        return $"alias_run_refresh_refused_by=[{by}] " +
            $"refused_stale_reads=[sampled:{Interlocked.Exchange(ref _refusedReads[(int)RefusedRead.Sampled], 0)}," +
            $"sampled-conservative:{Interlocked.Exchange(ref _refusedReads[(int)RefusedRead.SampledConservative], 0)}," +
            $"storage:{Interlocked.Exchange(ref _refusedReads[(int)RefusedRead.Storage], 0)}," +
            $"load:{Interlocked.Exchange(ref _refusedReads[(int)RefusedRead.AttachmentLoad], 0)}," +
            $"writeback:{Interlocked.Exchange(ref _refusedReads[(int)RefusedRead.Writeback], 0)}," +
            $"transfer:{Interlocked.Exchange(ref _refusedReads[(int)RefusedRead.Transfer], 0)}] " +
            $"refused_marks=[dead-whole:{deadWhole},read-then-whole:{readThenWhole},partial-write:{partial}," +
            $"reloaded:{reloaded},released:{released},outstanding:{outstanding}]";
    }
}
