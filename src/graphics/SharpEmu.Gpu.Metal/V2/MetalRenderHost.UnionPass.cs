// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Rendering;

namespace SharpEmu.Gpu.Metal.V2;

// A pass opened with the attachments the draws after its first one will need.
//
// Demon's Souls' G-buffer alternates draws into five targets with draws into the same five and a
// sixth, over the same depth, several times a frame. A draw into fewer of the open pass's
// targets joins it (SubsetJoin.cs), but a draw into more cannot: Metal fixes a pass's attachments
// when it begins, so each five-to-six step ended the pass, stored all six 1080p targets and
// loaded them again (about 1.75 ms of GPU time a frame). The step is the same from frame to
// frame, so the set that followed a pass's set last time is remembered, and a pass begun for the
// smaller set is opened with the larger one: the extra attachments are loaded and stored as they
// are, the draws into the smaller set are drawn with a pipeline that writes nothing to them
// (the subset join's variant), and the draws into the larger set continue the pass.
//
// Only an exact prefix is remembered: the larger set holds the smaller one's attachments at the
// same indices, subresources and formats, the same depth, area, layers and samples, and clears
// none of them. The extra attachments must still be the textures of live images that are not
// windows onto guest memory (a store there would write guest bytes the CPU may have changed),
// and the first draw must be able to join such a pass (no sampled extra, no sample mask, a
// variant compiled). A pass opened this way that no draw into the larger set continues is a
// miss; two misses in a row forget the set, and so does one that ended the pass early.
//
// Until a draw into the larger set arrives the union pass stands in for the smaller set's own
// pass, so a draw into exactly that set continues it as it would have continued that pass
// (ContinueUnionPass), drawn with the same variant: the subset join's extra conditions are for
// draws that would have begun a pass of their own. Demon's Souls' G-buffer draws reach memory
// through device addresses while labels are held to the commit, which the subset join refuses
// ("copies"), so every union pass ended at the second draw of the smaller set: two misses, the
// set forgotten and learnt again, never a hit, and two passes more than without it each time.
// (The subset join now checks such a draw against the ranges binding ordered, so the draws into
// the smaller set after the hit join as well: DepthOnlyJoin.cs, SHARPEMU_METAL_V2_JOIN_DEVICE_RANGES.)
// The TBDR line's union_miss=[...] says how the passes that missed ended; one a commit ended
// names the commit (flush-stream-capped, flush-labels, ...).
// SHARPEMU_METAL_V2_UNION_PASSES=0 turns it off (and the mesh draws it lets join, too).
internal sealed partial class MetalRenderHost
{
    private const int MaxUnionPredictions = 64;

    /// <summary>Whether passes open with a remembered superset (SHARPEMU_METAL_V2_UNION_PASSES).</summary>
    internal bool UnionPasses { get; set; } = MetalV2Diagnostics.UnionPasses;

    // The image behind each acquired colour attachment, for the liveness check of a remembered one.
    private readonly ResourceSlotIdentifier[] _attachmentImages = new ResourceSlotIdentifier[MaxColorAttachments];

    // One attachment as the arrays above record it.
    private struct UnionAttachment
    {
        public RenderingAttachment Attachment;
        public uint Level;
        public uint Slice;
        public uint DepthPlane;
        public uint TexelBytes;
        public uint Width;
        public uint Height;
        public nint Texture;
        public uint RealLevel;
        public uint RealLayer;
        public ResourceSlotIdentifier Image;
    }

    // A smaller set and the attachments the set that followed it added.
    private sealed class UnionPrediction
    {
        public RenderingState Subset;
        public readonly uint[] Levels = new uint[MaxColorAttachments];
        public readonly uint[] Slices = new uint[MaxColorAttachments];
        public readonly uint[] DepthPlanes = new uint[MaxColorAttachments];
        public uint DepthLevel;
        public uint DepthSlice;
        public readonly UnionAttachment[] Extras = new UnionAttachment[MaxColorAttachments];
        public int ExtraCount;
        public int Misses;
    }

    private readonly Dictionary<int, UnionPrediction> _unionPredictions = new();

    // The arrays' own values at the extra indices while a union pass is made.
    private readonly UnionAttachment[] _unionSaved = new UnionAttachment[MaxColorAttachments];

    // The union pass that is open, whether a draw into its whole set continued it, and the
    // number of attachments of the draw that began it.
    private UnionPrediction? _openUnion;
    private bool _openUnionHit;

    // How the open union pass is ending (the pass-end cause), and why the draw now beginning
    // could not continue it in the smaller set, if it could not.
    private string? _unionEndCause;
    private string? _unionConflict;

    private long _unionLearned;
    private long _unionOpened;
    private long _unionHits;
    private long _unionMisses;
    private long _unionRefused;
    private long _meshSubsetJoins;
    private long _unionContinued;
    private readonly Dictionary<string, long> _unionMissCauses = new();

    internal (long Learned, long Opened, long Hits, long Misses, long Refused, long MeshJoins) TakeUnionPasses() =>
        (Interlocked.Exchange(ref _unionLearned, 0), Interlocked.Exchange(ref _unionOpened, 0),
         Interlocked.Exchange(ref _unionHits, 0), Interlocked.Exchange(ref _unionMisses, 0),
         Interlocked.Exchange(ref _unionRefused, 0), Interlocked.Exchange(ref _meshSubsetJoins, 0));

    /// <summary>Draws into the smaller set that continued a union pass before a draw into the larger one.</summary>
    internal long TakeUnionContinued() => Interlocked.Exchange(ref _unionContinued, 0);

    /// <summary>How the union passes that missed ended: cause:count, most first.</summary>
    internal string TakeUnionMissCauses()
    {
        if (_unionMissCauses.Count == 0)
        {
            return string.Empty;
        }

        var text = string.Join(",", _unionMissCauses.OrderByDescending(pair => pair.Value).Select(pair => $"{pair.Key}:{pair.Value}"));
        _unionMissCauses.Clear();
        return text;
    }

    internal string TakeUnionPassText()
    {
        var (learned, opened, hits, misses, refused, meshJoins) = TakeUnionPasses();
        return $"union=[learned:{learned},opened:{opened},hits:{hits},misses:{misses},refused:{refused},mesh_joins:{meshJoins}] " +
            $"union_continued={TakeUnionContinued()} union_miss=[{TakeUnionMissCauses()}]";
    }

    internal int UnionPredictionCount => _unionPredictions.Count;

    private static int UnionKey(
        in RenderingState state, ReadOnlySpan<uint> levels, ReadOnlySpan<uint> slices, ReadOnlySpan<uint> planes,
        uint depthLevel, uint depthSlice)
    {
        var hash = new HashCode();
        hash.Add(state.Width);
        hash.Add(state.Height);
        hash.Add(state.Layers);
        hash.Add(state.Samples);
        hash.Add(state.ColorAttachmentCount);
        hash.Add(state.DepthStencilAttachment.View.Handle);
        hash.Add(depthLevel);
        hash.Add(depthSlice);
        for (var index = 0; index < (int)state.ColorAttachmentCount; index++)
        {
            hash.Add(state.ColorAttachments[index].View.Handle);
            hash.Add(state.ColorAttachments[index].Format);
            hash.Add(levels[index]);
            hash.Add(slices[index]);
            hash.Add(planes[index]);
        }

        return hash.ToHashCode();
    }

    private static bool SameUnionSubset(
        UnionPrediction prediction, in RenderingState state, ReadOnlySpan<uint> levels, ReadOnlySpan<uint> slices,
        ReadOnlySpan<uint> planes, uint depthLevel, uint depthSlice)
    {
        ref readonly var subset = ref prediction.Subset;
        if (subset.ColorAttachmentCount != state.ColorAttachmentCount || subset.Width != state.Width ||
            subset.Height != state.Height || subset.Layers != state.Layers || subset.Samples != state.Samples ||
            !SameDepth(in subset.DepthStencilAttachment, in state.DepthStencilAttachment) ||
            prediction.DepthLevel != depthLevel || prediction.DepthSlice != depthSlice)
        {
            return false;
        }

        for (var index = 0; index < (int)state.ColorAttachmentCount; index++)
        {
            if (subset.ColorAttachments[index].View.Handle != state.ColorAttachments[index].View.Handle ||
                subset.ColorAttachments[index].Format != state.ColorAttachments[index].Format ||
                prediction.Levels[index] != levels[index] || prediction.Slices[index] != slices[index] ||
                prediction.DepthPlanes[index] != planes[index])
            {
                return false;
            }
        }

        return true;
    }

    private static bool SameDepth(in RenderingAttachment left, in RenderingAttachment right) =>
        left.View.Handle == right.View.Handle && left.Format == right.Format &&
        left.HasDepth == right.HasDepth && left.HasStencil == right.HasStencil;

    /// <summary>
    /// Remembers the attachments a draw adds to the open pass's set, when it adds them after the
    /// open ones (the open set a prefix of its own) and clears none: the next pass opened for the
    /// open set is then opened with this one. Runs before the open pass ends.
    /// </summary>
    private void LearnUnionPass(in RenderingState next)
    {
        if (!UnionPasses || _openUnion is not null || !_renderingActive || _renderEncoder == 0)
        {
            return;
        }

        ref readonly var open = ref _renderingState;
        var openCount = (int)open.ColorAttachmentCount;
        var nextCount = (int)next.ColorAttachmentCount;
        if (openCount == 0 || nextCount <= openCount || nextCount > MaxColorAttachments ||
            _passAttachmentCount != openCount || _attachmentCount != nextCount || _passScratchMask != 0 ||
            next.Width != open.Width || next.Height != open.Height || next.Layers != open.Layers ||
            next.Samples != open.Samples)
        {
            return;
        }

        ref readonly var depth = ref next.DepthStencilAttachment;
        if (depth.DepthClear || depth.StencilClear || !SameDepth(in depth, in open.DepthStencilAttachment) ||
            (depth.View.Handle != 0 && (_depthAttachmentLevel != _passDepthLevel || _depthAttachmentSlice != _passDepthSlice)))
        {
            return;
        }

        for (var index = 0; index < nextCount; index++)
        {
            ref readonly var attachment = ref next.ColorAttachments[index];
            if (attachment.IsClear || attachment.View.Handle == 0)
            {
                return;
            }

            if (index < openCount)
            {
                ref readonly var opened = ref open.ColorAttachments[index];
                if (opened.View.Handle != attachment.View.Handle || opened.Format != attachment.Format ||
                    _attachmentLevels[index] != _passLevels[index] || _attachmentSlices[index] != _passSlices[index] ||
                    _attachmentDepthPlanes[index] != _passDepthPlanes[index])
                {
                    return;
                }

                continue;
            }

            if (!UnionExtraUsable(_attachmentImages[index], _attachmentTextures[index]))
            {
                return;
            }

            for (var other = 0; other < index; other++)
            {
                if (next.ColorAttachments[other].View.Handle == attachment.View.Handle ||
                    _attachmentTextures[other] == _attachmentTextures[index])
                {
                    return;
                }
            }
        }

        var key = UnionKey(in open, _passLevels, _passSlices, _passDepthPlanes, _passDepthLevel, _passDepthSlice);
        if (!_unionPredictions.TryGetValue(key, out var prediction) ||
            !SameUnionSubset(prediction, in open, _passLevels, _passSlices, _passDepthPlanes, _passDepthLevel, _passDepthSlice))
        {
            if (_unionPredictions.Count >= MaxUnionPredictions)
            {
                _unionPredictions.Clear();
            }

            prediction = new UnionPrediction();
            _unionPredictions[key] = prediction;
        }

        prediction.Subset = open;
        Array.Copy(_passLevels, prediction.Levels, MaxColorAttachments);
        Array.Copy(_passSlices, prediction.Slices, MaxColorAttachments);
        Array.Copy(_passDepthPlanes, prediction.DepthPlanes, MaxColorAttachments);
        prediction.DepthLevel = _passDepthLevel;
        prediction.DepthSlice = _passDepthSlice;
        prediction.ExtraCount = nextCount - openCount;
        prediction.Misses = 0;
        for (var index = openCount; index < nextCount; index++)
        {
            prediction.Extras[index - openCount] = CaptureUnionAttachment(in next.ColorAttachments[index], index);
        }

        _unionLearned++;
    }

    // A remembered attachment is used only while it is still the texture of a live image that owns it.
    private bool UnionExtraUsable(ResourceSlotIdentifier image, nint texture) =>
        texture != 0 && _imageCache.TryGetImage(image, out var cached) && cached.Texture == texture &&
        !cached.IsWindowLinearView;

    private UnionAttachment CaptureUnionAttachment(in RenderingAttachment attachment, int index) => new()
    {
        Attachment = attachment,
        Level = _attachmentLevels[index],
        Slice = _attachmentSlices[index],
        DepthPlane = _attachmentDepthPlanes[index],
        TexelBytes = _attachmentTexelBytes[index],
        Width = _attachmentWidths[index],
        Height = _attachmentHeights[index],
        Texture = _attachmentTextures[index],
        RealLevel = _attachmentRealLevels[index],
        RealLayer = _attachmentRealLayers[index],
        Image = _attachmentImages[index],
    };

    private void InstallUnionAttachment(in UnionAttachment attachment, int index)
    {
        _attachmentLevels[index] = attachment.Level;
        _attachmentSlices[index] = attachment.Slice;
        _attachmentDepthPlanes[index] = attachment.DepthPlane;
        _attachmentTexelBytes[index] = attachment.TexelBytes;
        _attachmentWidths[index] = attachment.Width;
        _attachmentHeights[index] = attachment.Height;
        _attachmentTextures[index] = attachment.Texture;
        _attachmentRealLevels[index] = attachment.RealLevel;
        _attachmentRealLayers[index] = attachment.RealLayer;
        _attachmentImages[index] = attachment.Image;
    }

    /// <summary>
    /// The pass to open for a draw into <paramref name="state"/>: its own set, or the superset
    /// remembered for it with the variant of the draw's pipeline that writes none of the extra
    /// attachments. Null when the draw opens its own set.
    /// </summary>
    private UnionPrediction? PlanUnionPass(
        in RenderingState state, MetalPipelineEntry? upcoming, uint scratchMask, bool launch,
        out RenderingState union, out MetalPipelineEntry? variant)
    {
        union = default;
        variant = null;
        var count = (int)state.ColorAttachmentCount;
        if (!UnionPasses || _unionPredictions.Count == 0 || upcoming is null || launch || scratchMask != 0 ||
            count == 0 || _attachmentCount != count)
        {
            return null;
        }

        var key = UnionKey(in state, _attachmentLevels, _attachmentSlices, _attachmentDepthPlanes,
            _depthAttachmentLevel, _depthAttachmentSlice);
        if (!_unionPredictions.TryGetValue(key, out var prediction) ||
            !SameUnionSubset(prediction, in state, _attachmentLevels, _attachmentSlices, _attachmentDepthPlanes,
                _depthAttachmentLevel, _depthAttachmentSlice) ||
            count + prediction.ExtraCount > MaxColorAttachments)
        {
            return null;
        }

        for (var extra = 0; extra < prediction.ExtraCount; extra++)
        {
            ref readonly var attachment = ref prediction.Extras[extra];
            if (!UnionExtraUsable(attachment.Image, attachment.Texture))
            {
                _unionPredictions.Remove(key);
                _unionRefused++;
                return null;
            }

            for (var index = 0; index < count; index++)
            {
                if (state.ColorAttachments[index].View.Handle == attachment.Attachment.View.Handle ||
                    _attachmentTextures[index] == attachment.Texture)
                {
                    _unionRefused++;
                    return null;
                }
            }
        }

        union = state;
        union.ColorAttachmentCount = (uint)(count + prediction.ExtraCount);
        for (var extra = 0; extra < prediction.ExtraCount; extra++)
        {
            union.ColorAttachments[count + extra] = prediction.Extras[extra].Attachment;
        }

        if (UnionDrawRefusal(prediction, in union, count, upcoming, out variant) is not null)
        {
            _unionRefused++;
            return null;
        }

        return prediction;
    }

    /// <summary>
    /// Why a draw into the first <paramref name="count"/> attachments of <paramref name="union"/>
    /// cannot be drawn in a union pass of that set, or null with the variant of its pipeline that
    /// writes none of the extra attachments.
    /// </summary>
    private string? UnionDrawRefusal(
        UnionPrediction prediction, in RenderingState union, int count, MetalPipelineEntry upcoming,
        out MetalPipelineEntry? variant)
    {
        variant = null;
        for (var extra = 0; extra < prediction.ExtraCount; extra++)
        {
            // Drawn into while the pass holds it, a sampled extra would be read before the pass stores it.
            ref readonly var attachment = ref prediction.Extras[extra];
            if (StagesSampleTexture((nint)attachment.Attachment.View.Handle, attachment.Texture))
            {
                return "sampled";
            }
        }

        if (upcoming.RectangleStates is not null || upcoming.GraphicsDescription is not { } description ||
            description.Rendering.ColorCount != count || description.StaticParameters.Samples != union.Samples ||
            description.ColorRemap != 0)
        {
            return "pipeline";
        }

        if (description.PixelInfo is { SampleMaskExportEnable: true })
        {
            return "coverage";
        }

        if (JoinExcludedPixelPrograms.Count != 0 && description.PixelStage is { } pixelStage &&
            JoinExcludedPixelPrograms.Contains(pixelStage.Hash))
        {
            return "excluded";
        }

        var placement = 0u;
        for (var index = 0; index < count; index++)
        {
            placement |= (uint)(index + 1) << (index * 4);
        }

        variant = SubsetJoinVariant(upcoming, description, in union, 0, placement);
        return variant is null ? "compiling" : null;
    }

    /// <summary>
    /// Whether a draw continues the open union pass in the smaller set it was opened for, before
    /// any draw into the larger set: the draw would have continued that set's own pass, so what
    /// such a continuation needs (CanContinuePass) is what it needs here, and a pipeline that
    /// writes none of the extra attachments. A draw into that set that cannot have one ends the
    /// pass (_unionConflict says why).
    /// </summary>
    private bool ContinueUnionPass(in RenderingState state, MetalPipelineEntry? upcoming)
    {
        if (_openUnion is not { } union || _openUnionHit || !_renderingActive || _renderEncoder == 0)
        {
            return false;
        }

        ref readonly var open = ref _renderingState;
        var count = (int)union.Subset.ColorAttachmentCount;
        if (state.ColorAttachmentCount != (uint)count || _attachmentCount != count ||
            _passAttachmentCount != count + union.ExtraCount ||
            state.Width != open.Width || state.Height != open.Height ||
            state.Layers != open.Layers || state.Samples != open.Samples)
        {
            return false;
        }

        for (var index = 0; index < count; index++)
        {
            ref readonly var next = ref state.ColorAttachments[index];
            ref readonly var opened = ref open.ColorAttachments[index];
            if (next.IsClear || next.View.Handle != opened.View.Handle || next.Format != opened.Format ||
                _attachmentLevels[index] != _passLevels[index] || _attachmentSlices[index] != _passSlices[index] ||
                _attachmentDepthPlanes[index] != _passDepthPlanes[index])
            {
                return false;
            }
        }

        ref readonly var nextDepth = ref state.DepthStencilAttachment;
        ref readonly var openDepth = ref open.DepthStencilAttachment;
        if (nextDepth.DepthClear || nextDepth.StencilClear || nextDepth.View.Handle != openDepth.View.Handle ||
            nextDepth.HasDepth != openDepth.HasDepth || nextDepth.HasStencil != openDepth.HasStencil ||
            (nextDepth.View.Handle != 0 &&
                (_depthAttachmentLevel != _passDepthLevel || _depthAttachmentSlice != _passDepthSlice)))
        {
            return false;
        }

        if (upcoming is null)
        {
            _unionConflict = "pipeline";
            return false;
        }

        _unionConflict = UnionDrawRefusal(union, in open, count, upcoming, out var variant);
        if (_unionConflict is not null)
        {
            return false;
        }

        _joinedPipeline = upcoming;
        _joinedVariant = variant;
        _unionContinued++;
        return true;
    }

    // The cause the pass-end counters give the open pass (PassProfileNoteEnd), kept for a union pass.
    private void NoteUnionPassEnd(string cause)
    {
        if (_openUnion is not null && _renderingActive)
        {
            // A commit says which one it was: a capped hold, labels owed to a guest, a host wait...
            _unionEndCause ??= cause == "flush" && _passEndFlushReason is { } reason ? "flush-" + reason : cause;
        }
    }

    // Whether a texture the next draw binds is this one or a view of it. A program that
    // reaches images through bindless tables could reach any.
    private bool StagesSampleTexture(nint view, nint root)
    {
        foreach (var stage in _joinStages)
        {
            if (stage is null)
            {
                continue;
            }

            if (stage.Program.Bindings is { BindlessImageTableCount: > 0 })
            {
                return true;
            }

            for (var index = 0; index < stage.ImageCount; index++)
            {
                if (IsTextureOrViewOf(stage.Textures[index], view, root))
                {
                    return true;
                }

                if (stage.MipTextures[index] is { } mips)
                {
                    foreach (var mip in mips)
                    {
                        if (IsTextureOrViewOf(mip, view, root))
                        {
                            return true;
                        }
                    }
                }
            }
        }

        return false;
    }

    private static bool IsTextureOrViewOf(nint texture, nint view, nint root)
    {
        if (texture == 0)
        {
            return false;
        }

        if (texture == view || texture == root)
        {
            return true;
        }

        var parent = MetalNative.Send(texture, SelParentTexture);
        return parent != 0 && (parent == view || parent == root);
    }

    // A union pass ends: whether any draw into its whole set continued it decides whether the
    // set stays remembered.
    private void FinishUnionPass()
    {
        if (_openUnion is not { } union)
        {
            return;
        }

        if (_openUnionHit)
        {
            union.Misses = 0;
            _unionHits++;
        }
        else
        {
            // A pass a draw into the smaller set had to end was split where its own pass would
            // not have been: one such miss forgets the set.
            var conflict = _unionConflict is not null && _unionEndCause == "target";
            var cause = conflict ? "conflict-" + _unionConflict : _unionEndCause ?? "end";
            ref var count = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(_unionMissCauses, cause, out _);
            count++;
            _unionMisses++;
            if (++union.Misses >= 2 || conflict)
            {
                foreach (var (key, value) in _unionPredictions)
                {
                    if (ReferenceEquals(value, union))
                    {
                        _unionPredictions.Remove(key);
                        break;
                    }
                }
            }
        }

        _openUnion = null;
        _openUnionHit = false;
        _unionEndCause = null;
    }
}
