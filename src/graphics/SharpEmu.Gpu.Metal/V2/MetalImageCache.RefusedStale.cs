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
// stale run (alias_run_refresh_refused). The refusal marks the image (afresh at every refusal); the mark records
// every later read of the texture that reaches a layer of the refused run no pass has rewritten since (sampled,
// storage, a render pass that loads it, a writeback that retiles it into guest memory, a copy or present source),
// split by whether any GPU write of uncertain extent (a storage bind, a resolve, a partial clear) came between -
// "clean" reads are certain to see the kept texels. The mark ends at a whole rewrite (every layer cleared or
// overwritten by a pass, a whole clear or kernel fill), a whole reload or a release.
internal sealed partial class MetalImageCache
{
    private static readonly string[] RunRefusalNames =
        ["none", "subrange-off", "stencil", "no-run", "over-half", "no-cover", "source", "not-moved"];

    private static readonly string[] RefusedReadNames =
        ["sampled", "sampled-conservative", "storage", "load", "writeback", "transfer"];

    private sealed class RefusedMark
    {
        public ulong Start, End;
        public string Runs = string.Empty;
        public MetalRunRefusal Reason;
        public bool Read;
        // A GPU write of unknown extent since the refusal: later reads may see its texels, not the kept ones.
        public bool Written;
        // Layers (of a one-level image of at most 64 layers) a pass cleared or overwrote whole since the refusal.
        public ulong LayersRewritten;
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
    // [role * 2 + (written ? 1 : 0)]
    private readonly long[] _refusedReads = new long[RefusedReadNames.Length * 2];
    private readonly Dictionary<string, long> _refusedReadsByImage = new(StringComparer.Ordinal);
    private long _refusedDeadWhole, _refusedReadThenWhole, _refusedPartialWrite, _refusedReloaded, _refusedReleased;

    private static bool SingleSubresource(MetalCachedImage image) =>
        image.Description.Resources.Levels <= 1 && image.Description.TransferLayers <= 1 && !image.Description.IsVolume;

    // Layers are tracked one by one only for a one-level, non-volume image of at most 64 of them.
    private static bool TracksLayers(MetalCachedImage image) =>
        image.Description.Resources.Levels <= 1 && !image.Description.IsVolume && image.Description.TransferLayers <= 64;

    private static ulong LayerMask(uint baseLayer, uint count)
    {
        if (count == 0 || baseLayer >= 64)
        {
            return 0;
        }

        count = Math.Min(count, 64 - baseLayer);
        var bits = count >= 64 ? ulong.MaxValue : (1UL << (int)count) - 1;
        return bits << (int)baseLayer;
    }

    private static ulong AllLayers(MetalCachedImage image) => LayerMask(0, Math.Max(image.Description.TransferLayers, 1));

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

            // Every refusal finds the run stale again: what passes rewrote before it is stale once more.
            mark.Start = image.DirtyStart;
            mark.End = image.DirtyEnd;
            mark.Runs = builder.ToString();
            mark.Reason = reason;
            mark.Written = false;
            mark.LayersRewritten = 0;
        }

        return reason;
    }

    // The layers of the refused run a read of [start, end) of the image's bytes (all of them when end is 0) reaches.
    private static ulong StaleLayersOf(MetalCachedImage image, RefusedMark mark, ulong start, ulong end)
    {
        if (end <= start)
        {
            start = 0;
            end = image.Description.Data.Size;
        }

        start = Math.Max(start, mark.Start);
        end = Math.Min(end, mark.End);
        if (end <= start)
        {
            return 0;
        }

        if (!TracksLayers(image))
        {
            return ulong.MaxValue;
        }

        var layers = Math.Max(image.Description.TransferLayers, 1);
        var layerBytes = image.Description.Data.Size / layers;
        if (layerBytes == 0)
        {
            return ulong.MaxValue;
        }

        var first = (uint)Math.Min(start / layerBytes, layers - 1);
        var last = (uint)Math.Min((end - 1) / layerBytes, layers - 1);
        return LayerMask(first, last - first + 1) & ~mark.LayersRewritten;
    }

    // A read of a marked image's texture over [start, end) of its bytes (all of them when end is 0), or over the
    // layers in layerMask when it is not 0.
    private void NoteRefusedRead(
        MetalCachedImage image, RefusedRead role, string detail, ulong start = 0, ulong end = 0, ulong layerMask = 0)
    {
        if (Volatile.Read(ref _refusedMarkCount) == 0)
        {
            return;
        }

        RefusedMark? mark;
        bool written;
        lock (_refusedMarks)
        {
            if (!_refusedMarks.TryGetValue(image, out mark))
            {
                return;
            }

            var stale = StaleLayersOf(image, mark, start, end);
            if (layerMask != 0 && TracksLayers(image))
            {
                stale &= layerMask;
            }

            if (stale == 0)
            {
                return;
            }

            mark.Read = true;
            written = mark.Written;
            ref readonly var shape = ref image.Description;
            var key = $"{shape.PixelFormat}/{shape.Extent.Width}x{shape.Extent.Height}x{shape.TransferLayers}" +
                $"@{shape.Data.Address:X}:{RefusedReadNames[(int)role]}{(written ? "/after-write" : "")}";
            _refusedReadsByImage[key] = _refusedReadsByImage.GetValueOrDefault(key) + 1;
        }

        Interlocked.Increment(ref _refusedReads[(int)role * 2 + (written ? 1 : 0)]);
        ref readonly var description = ref image.Description;
        MetalV2Diagnostics.WarnOnce(
            $"refused-stale-read:{role}:{written}:{detail}:{description.PixelFormat}:{description.Extent.Width}x{description.Extent.Height}",
            $"A GPU-written image whose stale run was refused a reload is read ({role} {detail}" +
            $"{(written ? ", after a GPU write of unknown extent" : ", no GPU write since")}): " +
            $"{description.PixelFormat} {description.Extent.Width}x{description.Extent.Height} " +
            $"levels={description.Resources.Levels} layers={description.TransferLayers} " +
            $"0x{description.Data.Address:X16}+0x{description.Data.Size:X} reason={RunRefusalNames[(int)mark.Reason]} " +
            $"stale=[{mark.Runs}] rewritten-layers=0x{mark.LayersRewritten:X}.");
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

    // A sampled bind of a view's layers: exact by layer for a one-level image, conservative (the whole run) otherwise.
    private void NoteRefusedSampled(MetalCachedImage image, uint baseLayer, uint layerCount)
    {
        if (Volatile.Read(ref _refusedMarkCount) != 0)
        {
            NoteRefusedRead(
                image, TracksLayers(image) ? RefusedRead.Sampled : RefusedRead.SampledConservative, "bind",
                layerMask: LayerMask(baseLayer, Math.Max(layerCount, 1)));
        }
    }

    // The GPU rewrote the image: whole (every texel of every subresource) ends the mark; a write of unknown or partial
    // extent keeps it and makes later reads "after-write".
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
                mark.Written = true;
                return;
            }

            EndRewrittenLocked(image, mark);
        }
    }

    private void EndRewrittenLocked(MetalCachedImage image, RefusedMark mark)
    {
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

            if (image is null)
            {
                return;
            }

            if (!loads)
            {
                var mark = _refusedMarks[image];
                if (TracksLayers(image) && level == 0)
                {
                    mark.LayersRewritten |= LayerMask(layer, 1);
                    if ((mark.LayersRewritten & AllLayers(image)) == AllLayers(image))
                    {
                        EndRewrittenLocked(image, mark);
                    }
                    else
                    {
                        _refusedPartialWrite++;
                    }
                }
                else
                {
                    _refusedPartialWrite++;
                    mark.Written = true;
                }

                return;
            }
        }

        NoteRefusedRead(
            image, RefusedRead.AttachmentLoad, $"{(depth ? "depth" : "color")}-l{level}-s{layer}",
            layerMask: LayerMask(layer, 1));
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

        var reads = new StringBuilder();
        for (var role = 0; role < RefusedReadNames.Length; role++)
        {
            reads.Append(role == 0 ? "" : ",").Append(RefusedReadNames[role]).Append(':')
                .Append(Interlocked.Exchange(ref _refusedReads[role * 2], 0)).Append('/')
                .Append(Interlocked.Exchange(ref _refusedReads[role * 2 + 1], 0));
        }

        long deadWhole, readThenWhole, partial, reloaded, released;
        int outstanding;
        string byImage;
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
            byImage = string.Join(',', _refusedReadsByImage.OrderBy(p => p.Key, StringComparer.Ordinal)
                .Select(p => $"{p.Key}={p.Value}"));
            _refusedReadsByImage.Clear();
        }

        return $"alias_run_refresh_refused_by=[{by}] refused_stale_reads(clean/after-write)=[{reads}] " +
            $"refused_marks=[dead-whole:{deadWhole},read-then-whole:{readThenWhole},partial-write:{partial}," +
            $"reloaded:{reloaded},released:{released},outstanding:{outstanding}] refused_reads_by_image=[{byImage}]";
    }
}
