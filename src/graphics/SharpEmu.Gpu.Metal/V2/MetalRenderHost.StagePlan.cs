// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.CompilerServices;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

/// <summary>
/// A stage's image and sampler lookups answered from the last preparation of the same snapshot arrays (the stage
/// binding plan). The resource materializer hands every draw whose descriptors evaluate the same way the same
/// Images and Samplers arrays (Demon's Souls: nine in ten stage bindings), and PrepareBindings looked each of their
/// T#s up in the texture memo (a hash and a whole-key compare per sampled image) and each S# up in the sampler
/// cache again. The plan keeps what those lookups returned: per image the texture memo entry the T#'s key maps to,
/// per sampler the handle for its words and integer-view class.
///
/// What a plan's answers depend on, and how each is checked:
/// <list type="bullet">
/// <item>The stage, the program (its image shapes, sampler resources and sampled pairs) and the T# and S# words:
/// a plan answers only for the same stage, the same <see cref="ShaderProgramInfo"/> and the same Images and Samplers
/// arrays by reference. A published snapshot's arrays are never written (the materializer builds new ones per
/// evaluation and shares them whole); =verify recomputes every key from the words to catch one that was.</item>
/// <item>Which memo entry a key maps to: an entry changes in place only for its own key (RememberTexture), so
/// the entry stays the one a lookup returns until the memo is cleared (its generation, checked per plan).</item>
/// <item>Whether the entry's stamp holds: checked for every use exactly as on a lookup hit, and a binding whose stamp
/// fails is resolved and memoised as on a lookup miss. A memoisable image whose key had no entry when the plan was
/// made is looked up each time, and the plan takes the entry once a lookup finds one.</item>
/// <item>A host movie plane: a stage with one takes the full path (FindHostMovieBindings runs for every preparation).</item>
/// <item>The sampler handles: the sampler cache is only ever cleared whole (its generation, checked per plan).</item>
/// <item>A sampler's integer-view class: resolved again for every preparation from this draw's image requests; a
/// planned handle answers only for the class it was made for, and the plan takes the other one when it moves.</item>
/// </list>
/// Buffers are not planned: what a buffer binding answers depends on live state (mapping, staged copies, GPU-modified
/// images, the window), which MetalReadBindingMemo already stamps per range. SHARPEMU_METAL_V2_STAGE_PLAN=0 looks
/// everything up; =verify also checks every planned answer against a fresh lookup and raises a fatal error when one
/// differs.
/// </summary>
internal sealed partial class MetalRenderHost
{
    private static readonly string? StagePlanSetting = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_STAGE_PLAN");

    internal bool StagePlanEnabled { get; set; } = StagePlanSetting != "0";

    internal bool StagePlanVerify { get; set; } =
        StagePlanSetting == "verify" || Environment.GetEnvironmentVariable("SHARPEMU_PERF_VERIFY") == "1";

    private sealed class StagePlan
    {
        public int Stage;
        public ShaderProgramInfo? Program;
        public uint[][]? Images;
        public uint[][]? Samplers;
        public long MemoGeneration;
        public long SamplerGeneration;
        // Per image the memo entry its key maps to; null for a binding the memo does not take, or a key that had none.
        public TextureMemoEntry?[] Entries = [];
        public bool[] SamplerInteger = [];
        public nint[] SamplerHandles = [];
        public ulong[] SamplerIds = [];
    }

    // Two ways per set, the more recently used first.
    private const int StagePlanSets = 1024;
    private readonly StagePlan?[] _stagePlans = new StagePlan?[StagePlanSets * 2];
    private TextureMemoEntry?[] _stagePlanEntries = [];
    private long _textureMemoGeneration;
    private long _samplerGeneration;

    private long _stagePlanHits, _stagePlanAbsent, _stagePlanMemoCleared, _stagePlanSamplersCleared, _stagePlanMovie;
    private long _stagePlanStores, _stagePlanEvictions, _stagePlanStaleImages, _stagePlanImageLookups;
    private long _stagePlanSamplerClass, _stagePlanVerified;

    internal (long Hits, long Absent, long MemoCleared, long Stores, long StaleImages, long ImageLookups, long Verified) StagePlanCounts =>
        (_stagePlanHits, _stagePlanAbsent, _stagePlanMemoCleared, _stagePlanStores, _stagePlanStaleImages, _stagePlanImageLookups,
            _stagePlanVerified);

    private static int StagePlanSet(int stage, uint[][] images, uint[][] samplers) =>
        (int)((((uint)RuntimeHelpers.GetHashCode(images) * 0x9E3779B1u) ^ ((uint)RuntimeHelpers.GetHashCode(samplers) * 0x85EBCA77u) ^
            ((uint)stage * 0xC2B2AE3Du)) >> 22) & (StagePlanSets - 1);

    private static bool PlanFor([System.Diagnostics.CodeAnalysis.NotNullWhen(true)] StagePlan? plan, int stage, ShaderProgramInfo program,
        ResourceSnapshot snapshot) =>
        plan is not null && plan.Stage == stage && ReferenceEquals(plan.Program, program) &&
        ReferenceEquals(plan.Images, snapshot.Images) && ReferenceEquals(plan.Samplers, snapshot.Samplers);

    /// <summary>
    /// The plan for this stage's arrays when one stands; otherwise null, with <paramref name="store"/> set when the
    /// full path's answers should be kept as the plan.
    /// </summary>
    private StagePlan? FindStagePlan(int stage, ShaderProgramInfo program, ShaderResourceInfo info, ResourceSnapshot snapshot,
        MetalHostMovieSelector.Bindings movie, out bool store)
    {
        store = false;
        if (!StagePlanEnabled || info.Images.Count + info.Samplers.Count == 0)
        {
            return null;
        }

        if (movie != MetalHostMovieSelector.Bindings.None)
        {
            _stagePlanMovie++;
            return null;
        }

        store = true;
        var way = StagePlanSet(stage, snapshot.Images, snapshot.Samplers) * 2;
        var plan = _stagePlans[way];
        if (!PlanFor(plan, stage, program, snapshot))
        {
            plan = _stagePlans[way + 1];
            if (!PlanFor(plan, stage, program, snapshot))
            {
                _stagePlanAbsent++;
                return null;
            }

            (_stagePlans[way], _stagePlans[way + 1]) = (plan, _stagePlans[way]);
        }

        if (plan.MemoGeneration != _textureMemoGeneration)
        {
            _stagePlanMemoCleared++;
            return null;
        }

        if (plan.SamplerGeneration != _samplerGeneration)
        {
            _stagePlanSamplersCleared++;
            return null;
        }

        store = false;
        _stagePlanHits++;
        return plan;
    }

    // The full path's answers for this stage's arrays, from the memo entries its lookups found (_stagePlanEntries)
    // and the samplers it resolved. Reuses the evicted plan's object: a stage whose arrays never repeat allocates nothing.
    private void StoreStagePlan(int stage, ShaderProgramInfo program, ShaderResourceInfo info, ResourceSnapshot snapshot,
        MetalStageBindings prepared)
    {
        var way = StagePlanSet(stage, snapshot.Images, snapshot.Samplers) * 2;
        StagePlan plan;
        if (PlanFor(_stagePlans[way], stage, program, snapshot))
        {
            plan = _stagePlans[way]!;
        }
        else if (PlanFor(_stagePlans[way + 1], stage, program, snapshot))
        {
            plan = _stagePlans[way + 1]!;
            (_stagePlans[way], _stagePlans[way + 1]) = (plan, _stagePlans[way]);
        }
        else
        {
            // The less recently used way's object takes the new plan, in front.
            plan = _stagePlans[way + 1] ?? new StagePlan();
            if (plan.Images is not null)
            {
                _stagePlanEvictions++;
            }

            _stagePlans[way + 1] = _stagePlans[way];
            _stagePlans[way] = plan;
        }

        var images = info.Images.Count;
        var samplers = info.Samplers.Count;
        if (plan.Entries.Length < images)
        {
            plan.Entries = new TextureMemoEntry?[images];
        }

        if (plan.SamplerHandles.Length < samplers)
        {
            plan.SamplerInteger = new bool[samplers];
            plan.SamplerHandles = new nint[samplers];
            plan.SamplerIds = new ulong[samplers];
        }

        Array.Copy(_stagePlanEntries, plan.Entries, images);
        for (var index = 0; index < samplers; index++)
        {
            plan.SamplerInteger[index] = prepared.SamplerViewClasses[index] == MetalSamplerViewClasses.AllInteger;
            plan.SamplerHandles[index] = prepared.Samplers[index];
            plan.SamplerIds[index] = prepared.SamplerIds[index];
        }

        plan.Stage = stage;
        plan.Program = program;
        plan.Images = snapshot.Images;
        plan.Samplers = snapshot.Samplers;
        plan.MemoGeneration = _textureMemoGeneration;
        plan.SamplerGeneration = _samplerGeneration;
        _stagePlanStores++;
    }

    // =verify: the entry a fresh lookup of the image's key returns is the planned one.
    private void VerifyPlannedImage(MetalStageBindings prepared, ImageResource image, uint[] words, int index, TextureMemoEntry planned)
    {
        TextureMemoEntry? found = null;
        if (!TextureMemoApplies(image, words) ||
            !_textureMemo.TryGetValue(new TextureMemoKey(words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7],
                ShapeOf(image)), out found) || !ReferenceEquals(found, planned))
        {
            throw Fatal(
                $"The stage plan's texture memo entry differs from a lookup: hash=0x{prepared.Program.Hash:X16} image={index} " +
                $"planned=0x{planned.Key.W0:X8}/{planned.Image} found={(found is null ? "none" : $"0x{found.Key.W0:X8}/{found.Image}")}.");
        }

        _stagePlanVerified++;
    }

    // =verify: the sampler a fresh lookup of the S# and class returns is the planned one.
    private void VerifyPlannedSampler(MetalStageBindings prepared, SamplerResource sampler, uint[] words, int index, bool integer,
        nint planned, ulong plannedId)
    {
        var descriptor = NativeSamplerWords(sampler, words);
        var found = GetSampler(in descriptor, integer, out var foundId);
        if (found != planned || foundId != plannedId)
        {
            throw Fatal(
                $"The stage plan's sampler differs from a lookup: hash=0x{prepared.Program.Hash:X16} sampler={index} " +
                $"integer={integer} planned=0x{planned:X}/{plannedId} found=0x{found:X}/{foundId}.");
        }

        _stagePlanVerified++;
    }

    private string TakeStagePlanCounters()
    {
        var text = $"stage_plan=(hits={_stagePlanHits} absent={_stagePlanAbsent} memo_cleared={_stagePlanMemoCleared} " +
            $"samplers_cleared={_stagePlanSamplersCleared} movie={_stagePlanMovie} stores={_stagePlanStores} " +
            $"evictions={_stagePlanEvictions} stale_images={_stagePlanStaleImages} image_lookups={_stagePlanImageLookups} " +
            $"sampler_class={_stagePlanSamplerClass} verified={_stagePlanVerified})";
        _stagePlanHits = _stagePlanAbsent = _stagePlanMemoCleared = _stagePlanSamplersCleared = _stagePlanMovie = 0;
        _stagePlanStores = _stagePlanEvictions = _stagePlanStaleImages = _stagePlanImageLookups = 0;
        _stagePlanSamplerClass = _stagePlanVerified = 0;
        return text;
    }
}
