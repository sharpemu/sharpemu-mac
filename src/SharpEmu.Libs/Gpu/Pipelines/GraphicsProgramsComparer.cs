// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Libs.Gpu.Pipelines;

// Whether two graphics resolutions hand the executor the same programs, inputs and resources: every field and array of
// GraphicsPrograms, its vertex and pixel inputs and each stage's program, shader base and snapshot words. The resolve
// memo's verify mode and its tests compare a hit with a full resolution through it.
internal static class GraphicsProgramsComparer
{
    // The first field that differs, or null when none does.
    public static string? Difference(GraphicsPrograms memo, GraphicsPrograms full)
    {
        if (memo.Available != full.Available) return "available";
        if (memo.Vertex != full.Vertex) return "vertex_program";
        if (memo.Pixel != full.Pixel) return "pixel_program";
        if (!ReferenceEquals(memo.MeshInput, full.MeshInput) && memo.MeshInput != full.MeshInput) return "mesh_input";
        if (memo.Tessellation is not null || full.Tessellation is not null) return "tessellation";
        if (memo.SolidClear != full.SolidClear) return "solid_clear";
        if (memo.DisableBlending != full.DisableBlending) return "disable_blending";
        if (memo.DropUnwrittenColorTargets != full.DropUnwrittenColorTargets) return "drop_unwritten";
        if (memo.PositionStream != full.PositionStream) return "position_stream";
        return VertexDifference(memo.VertexInput, full.VertexInput) ?? PixelDifference(memo.PixelInput, full.PixelInput);
    }

    // Every group that differs, for a fatal that has to name more than the first field: the top-level fields one by one,
    // then the first difference within the vertex input and within the pixel input. Empty when none does.
    public static string Differences(GraphicsPrograms memo, GraphicsPrograms full)
    {
        var differences = new List<string>();
        if (memo.Available != full.Available) differences.Add("available");
        if (memo.Vertex != full.Vertex) differences.Add("vertex_program");
        if (memo.Pixel != full.Pixel) differences.Add("pixel_program");
        if (!ReferenceEquals(memo.MeshInput, full.MeshInput) && memo.MeshInput != full.MeshInput) differences.Add("mesh_input");
        if (memo.Tessellation is not null || full.Tessellation is not null) differences.Add("tessellation");
        if (memo.SolidClear != full.SolidClear) differences.Add("solid_clear");
        if (memo.DisableBlending != full.DisableBlending) differences.Add("disable_blending");
        if (memo.DropUnwrittenColorTargets != full.DropUnwrittenColorTargets) differences.Add("drop_unwritten");
        if (memo.PositionStream != full.PositionStream) differences.Add("position_stream");
        if (VertexDifference(memo.VertexInput, full.VertexInput) is { } vertex) differences.Add(vertex);
        if (PixelDifference(memo.PixelInput, full.PixelInput) is { } pixel) differences.Add(pixel);
        return string.Join(",", differences);
    }

    // The identities of a resolution's programs and stages, for the same fatal: each program's id and module, and each
    // stage's compiled program hash, shader base and resource counts.
    public static string Describe(GraphicsPrograms programs)
    {
        static string Stage(in ShaderStageResources stage) => stage.Program is not { } program
            ? "none"
            : $"hash=0x{program.Hash:X16} base=0x{stage.ShaderBase:X} buffers={stage.Resources.Buffers.Length} " +
              $"images={stage.Resources.Images.Length} samplers={stage.Resources.Samplers.Length} table={stage.Resources.FlattenedResourceTable.Length}";

        return $"available={programs.Available} vs=0x{programs.Vertex.Id:X}/0x{programs.Vertex.Module:X} " +
            $"ps=0x{programs.Pixel.Id:X}/0x{programs.Pixel.Module:X} drop_unwritten={programs.DropUnwrittenColorTargets} " +
            $"vertex_stage=[{Stage(programs.VertexInput.Stage)}] pixel_stage=[{Stage(programs.PixelInput.Stage)}] " +
            $"pixel_inputs={programs.PixelInput.InputCount} custom=0x{programs.PixelInput.CustomInterpolationMask:X}";
    }

    // For a fatal or a race report: per stage, how the two resolutions' permutation keys (specializations) and snapshot
    // words differ, whatever their programs are; each differing flattened-table word with both values and the guest
    // address it is loaded from (evaluated now with the snapshot's own draw values). Tells inputs that differ (some
    // writer) from equal inputs that picked another permutation.
    public static string DescribeStages(GraphicsPrograms left, GraphicsPrograms right) =>
        $"vertex=[{DescribeStage(left.VertexInput.Stage, right.VertexInput.Stage)}] pixel=[{DescribeStage(left.PixelInput.Stage, right.PixelInput.Stage)}]";

    private static string DescribeStage(in ShaderStageResources left, in ShaderStageResources right)
    {
        if (left.Program is null || right.Program is null)
        {
            return left.Program is null && right.Program is null ? "none" : "one_side_none";
        }

        var leftSpecialization = ResourceMaterializer.SpecializationOf(left.Resources);
        var rightSpecialization = ResourceMaterializer.SpecializationOf(right.Resources);
        var specialization = leftSpecialization is null || rightSpecialization is null ? "unknown"
            : ReferenceEquals(leftSpecialization, rightSpecialization) ? "same_object"
            : leftSpecialization.DifferenceFrom(rightSpecialization) ?? "equal";
        return $"same_program={ReferenceEquals(left.Program, right.Program)} specialization={specialization} " +
            $"words={DescribeWords(left.Resources, right.Resources)}";
    }

    private static string DescribeWords(ResourceSnapshot left, ResourceSnapshot right)
    {
        if (ReferenceEquals(left, right)) return "same_snapshot";
        var differences = new List<string>();
        List<(ulong Address, uint Word)>? reads = null;
        var readsTaken = false;
        // Where a differing word of the right side is read from now: the addresses an evaluation of right's program
        // reads that value at (a descriptor's words come from memory, not from a table slot with a known address).
        string ReadAt(uint[] a, uint[] b)
        {
            if (!readsTaken)
            {
                readsTaken = true;
                reads = ResourceMaterializer.RecordWordReads(right);
            }

            if (reads is null) return " read_at=unknown";
            var found = new List<string>();
            for (var word = 0; word < b.Length && found.Count < 4; word++)
            {
                if (word < a.Length && a[word] == b[word]) continue;
                foreach (var read in reads)
                {
                    if (read.Word != b[word]) continue;
                    found.Add($"w{word}@0x{read.Address:X}");
                    break;
                }
            }

            return found.Count == 0 ? " read_at=none" : " read_at=" + string.Join(",", found);
        }

        void Jagged(string name, uint[][] a, uint[][] b)
        {
            if (a.Length != b.Length)
            {
                differences.Add($"{name}.count {a.Length} vs {b.Length}");
                return;
            }

            for (var index = 0; index < a.Length; index++)
            {
                if (a[index].AsSpan().SequenceEqual(b[index])) continue;
                differences.Add($"{name}[{index}] {Words(a[index])} vs {Words(b[index])}{ReadAt(a[index], b[index])}");
            }
        }

        Jagged("buffers", left.Buffers, right.Buffers);
        Jagged("images", left.Images, right.Images);
        Jagged("samplers", left.Samplers, right.Samplers);
        var (leftTable, rightTable) = (left.FlattenedResourceTable, right.FlattenedResourceTable);
        if (leftTable.Length != rightTable.Length)
        {
            differences.Add($"table.count {leftTable.Length} vs {rightTable.Length}");
        }
        else
        {
            var listed = 0;
            for (var slot = 0; slot < leftTable.Length; slot++)
            {
                if (leftTable[slot] == rightTable[slot]) continue;
                if (++listed > 12)
                {
                    differences.Add("table.more");
                    break;
                }

                var address = ResourceMaterializer.TableWordAddress(left, slot) ?? ResourceMaterializer.TableWordAddress(right, slot);
                differences.Add($"table[{slot}]@{(address is { } at ? $"0x{at:X}" : "none")} 0x{leftTable[slot]:X8} vs 0x{rightTable[slot]:X8}");
            }
        }

        if (!left.UserData.AsSpan().SequenceEqual(right.UserData)) differences.Add("user_data");
        if (!left.DeviceAddressRanges.AsSpan().SequenceEqual(right.DeviceAddressRanges)) differences.Add("ranges");
        if (!SameArray(left.MipStatistics, right.MipStatistics)) differences.Add("mip_statistics");
        if (left.BindlessImages.Length != right.BindlessImages.Length) differences.Add("bindless.count");
        return differences.Count == 0 ? "equal" : string.Join(";", differences);
    }

    private static string Words(uint[] words) => string.Join(",", words.Select(word => word.ToString("X8", System.Globalization.CultureInfo.InvariantCulture)));

    private static string? VertexDifference(VertexInputInfo memo, VertexInputInfo full)
    {
        if (memo.Tessellation is not null || full.Tessellation is not null) return "vertex.tessellation";
        if (!memo.Buffers.AsSpan().SequenceEqual(full.Buffers)) return "vertex.buffers";
        if (!memo.Attributes.AsSpan().SequenceEqual(full.Attributes)) return "vertex.attributes";
        if (memo.FetchEmbedded != full.FetchEmbedded || memo.FetchAttributeRegister != full.FetchAttributeRegister ||
            memo.FetchBufferRegister != full.FetchBufferRegister) return "vertex.fetch";
        if (memo.IeeeMode != full.IeeeMode || memo.FloatMode != full.FloatMode || memo.Dx10Clamp != full.Dx10Clamp) return "vertex.modes";
        if (memo.ScratchDwords != full.ScratchDwords) return "vertex.scratch";
        if (memo.PositionExportControl != full.PositionExportControl) return "vertex.position_export";
        if (!SameClipSpace(memo.ClipSpace, full.ClipSpace)) return "vertex.clip_space";
        if (memo.StartInstanceUserData != full.StartInstanceUserData) return "vertex.start_instance";
        return StageDifference("vertex", memo.Stage, full.Stage);
    }

    private static bool SameClipSpace(in ClipSpaceTransform memo, in ClipSpaceTransform full) =>
        (memo with { UserClipPlanes = null }) == (full with { UserClipPlanes = null }) &&
        (memo.UserClipPlanes ?? []).AsSpan().SequenceEqual(full.UserClipPlanes ?? []) &&
        (memo.UserClipPlanes is null) == (full.UserClipPlanes is null);

    private static string? PixelDifference(PixelInputInfo memo, PixelInputInfo full)
    {
        if (memo.InputCount != full.InputCount || memo.SystemInputBase != full.SystemInputBase ||
            memo.CustomInterpolationMask != full.CustomInterpolationMask || memo.PerspectiveCenterRegister != full.PerspectiveCenterRegister)
            return "pixel.inputs";
        if (!memo.InterpolatorSettings.AsSpan().SequenceEqual(full.InterpolatorSettings)) return "pixel.interpolators";
        if (!memo.TargetOutputModes.AsSpan().SequenceEqual(full.TargetOutputModes)) return "pixel.output_modes";
        if (!memo.TargetExportMappings.AsSpan().SequenceEqual(full.TargetExportMappings)) return "pixel.export_mappings";
        if (!memo.TargetExportFormats.AsSpan().SequenceEqual(full.TargetExportFormats)) return "pixel.export_formats";
        if (memo.GuestWave32 != full.GuestWave32 || memo.IeeeMode != full.IeeeMode || memo.FloatMode != full.FloatMode ||
            memo.Dx10Clamp != full.Dx10Clamp) return "pixel.modes";
        if (memo.ScratchDwords != full.ScratchDwords) return "pixel.scratch";
        if (memo.PositionX != full.PositionX || memo.PositionY != full.PositionY || memo.PositionZ != full.PositionZ ||
            memo.PositionW != full.PositionW || memo.FrontFace != full.FrontFace || memo.FrontFaceAllBits != full.FrontFaceAllBits ||
            memo.NoPerspective != full.NoPerspective || memo.SampleShading != full.SampleShading) return "pixel.system_inputs";
        if (memo.KillEnable != full.KillEnable || memo.DepthExportEnable != full.DepthExportEnable ||
            memo.StencilExportEnable != full.StencilExportEnable || memo.SampleMaskExportEnable != full.SampleMaskExportEnable ||
            memo.DepthExportFormat != full.DepthExportFormat || memo.EarlyDepth != full.EarlyDepth || memo.ExecuteOnNoop != full.ExecuteOnNoop)
            return "pixel.depth_control";
        return StageDifference("pixel", memo.Stage, full.Stage);
    }

    // A dispatch's two resolutions (ParallelProgramPrefetch's compute verify): what the executor branches on, the
    // program, every launch input and the stage as Difference compares a draw's. Null when they hand it the same.
    public static string? ComputeDifference(ComputeProgram ahead, ComputeProgram worker)
    {
        if (ahead.Available != worker.Available) return "available";
        if (ahead.Consumed != worker.Consumed) return "consumed";
        if ((ahead.ImageCopy is null) != (worker.ImageCopy is null)) return "image_copy";
        if ((ahead.KnownWrite is null) != (worker.KnownWrite is null)) return "known_write";
        if (ahead.Program != worker.Program) return "compute_program";
        var (a, w) = (ahead.Input, worker.Input);
        if (ReferenceEquals(a, w)) return null;
        if (a.ThreadsX != w.ThreadsX || a.ThreadsY != w.ThreadsY || a.ThreadsZ != w.ThreadsZ) return "compute.threads";
        if (a.DispatchThreadDimensions != w.DispatchThreadDimensions || a.DispatchThreadsX != w.DispatchThreadsX ||
            a.DispatchThreadsY != w.DispatchThreadsY || a.DispatchThreadsZ != w.DispatchThreadsZ) return "compute.dispatch_threads";
        if (a.GroupIdX != w.GroupIdX || a.GroupIdY != w.GroupIdY || a.GroupIdZ != w.GroupIdZ || a.ThreadIdCount != w.ThreadIdCount ||
            a.ThreadGroupSizeEnabled != w.ThreadGroupSizeEnabled || a.WorkgroupRegister != w.WorkgroupRegister) return "compute.ids";
        if (a.WaveSize != w.WaveSize || a.LocalDataShareDwords != w.LocalDataShareDwords || a.ScratchDwords != w.ScratchDwords ||
            a.NeedsLocalDataShareBarriers != w.NeedsLocalDataShareBarriers) return "compute.launch";
        if (a.IeeeMode != w.IeeeMode || a.FloatMode != w.FloatMode || a.Dx10Clamp != w.Dx10Clamp) return "compute.float_mode";
        return StageDifference("compute", a.Stage, w.Stage);
    }

    // For the compute verify's fatal: each side's program and stage, and how their stages differ (DescribeStages).
    public static string DescribeCompute(ComputeProgram ahead, ComputeProgram worker)
    {
        static string Side(ComputeProgram program) => program.Input.Stage.Program is not { } stage
            ? $"available={program.Available} cs=0x{program.Program.Id:X}/0x{program.Program.Module:X} stage=none"
            : $"available={program.Available} cs=0x{program.Program.Id:X}/0x{program.Program.Module:X} hash=0x{stage.Hash:X16} " +
              $"base=0x{program.Input.Stage.ShaderBase:X} buffers={program.Input.Stage.Resources.Buffers.Length} " +
              $"images={program.Input.Stage.Resources.Images.Length} table={program.Input.Stage.Resources.FlattenedResourceTable.Length}";

        return $"ahead=[{Side(ahead)}] worker=[{Side(worker)}] stages=[{DescribeStage(ahead.Input.Stage, worker.Input.Stage)}]";
    }

    private static string? StageDifference(string stage, in ShaderStageResources memo, in ShaderStageResources full)
    {
        if (!ReferenceEquals(memo.Program, full.Program)) return stage + ".program";
        if (memo.ShaderBase != full.ShaderBase) return stage + ".shader_base";
        if (memo.ThreadLimits != full.ThreadLimits || memo.MeshDraw != full.MeshDraw) return stage + ".launch";
        if (!SameArray(memo.TessellationData, full.TessellationData)) return stage + ".tessellation_data";
        if (!SameArray(memo.CallArguments, full.CallArguments) || !SameArray(memo.CallTargets, full.CallTargets) ||
            !SameArray(memo.CallRecords, full.CallRecords) || !SameArray(memo.CallRecordRanges, full.CallRecordRanges))
            return stage + ".calls";
        if (memo.Program is null) return null;
        return SnapshotDifference(stage, memo.Resources, full.Resources);
    }

    private static string? SnapshotDifference(string stage, ResourceSnapshot memo, ResourceSnapshot full)
    {
        if (ReferenceEquals(memo, full)) return null;
        if (!SameJagged(memo.Buffers, full.Buffers)) return stage + ".buffers";
        if (!SameJagged(memo.Images, full.Images)) return stage + ".images";
        if (!SameJagged(memo.Samplers, full.Samplers)) return stage + ".samplers";
        if (!memo.FlattenedResourceTable.AsSpan().SequenceEqual(full.FlattenedResourceTable)) return stage + ".table";
        if (!memo.UserData.AsSpan().SequenceEqual(full.UserData)) return stage + ".user_data";
        if (!memo.DeviceAddressRanges.AsSpan().SequenceEqual(full.DeviceAddressRanges)) return stage + ".ranges";
        if (!SameArray(memo.MipStatistics, full.MipStatistics)) return stage + ".mip_statistics";
        if (memo.BindlessImages.Length != full.BindlessImages.Length) return stage + ".bindless";
        for (var index = 0; index < memo.BindlessImages.Length; index++)
        {
            var (a, b) = (memo.BindlessImages[index], full.BindlessImages[index]);
            if (!ReferenceEquals(a, b) && (a.Slot != b.Slot || a.Address != b.Address || !a.VariantImages.AsSpan().SequenceEqual(b.VariantImages) ||
                !SameJagged(a.Descriptors, b.Descriptors) || !a.DescriptorVariants.AsSpan().SequenceEqual(b.DescriptorVariants) ||
                !a.Entries.AsSpan().SequenceEqual(b.Entries)))
                return stage + ".bindless";
        }

        return null;
    }

    private static bool SameArray<T>(T[]? memo, T[]? full) =>
        ReferenceEquals(memo, full) || (memo is not null && full is not null && memo.AsSpan().SequenceEqual(full));

    private static bool SameJagged(uint[][] memo, uint[][] full)
    {
        if (memo.Length != full.Length) return false;
        for (var index = 0; index < memo.Length; index++)
        {
            if (!memo[index].AsSpan().SequenceEqual(full[index])) return false;
        }

        return true;
    }

    // A hash of everything Difference compares, taken when a result is kept and again before it is handed out, so a
    // write into a shared result shows.
    public static ulong Fingerprint(GraphicsPrograms programs)
    {
        var hash = 0xCBF29CE484222325UL;
        void Mix(ulong value) => hash = (hash ^ value) * 0x100000001B3UL;
        void MixWords(ReadOnlySpan<uint> words)
        {
            Mix((ulong)words.Length);
            foreach (var word in words) Mix(word);
        }

        void MixStage(in ShaderStageResources stage)
        {
            Mix((ulong)System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(stage.Program));
            Mix(stage.ShaderBase);
            if (stage.Program is null) return;
            var snapshot = stage.Resources;
            foreach (var words in snapshot.Buffers) MixWords(words);
            foreach (var words in snapshot.Images) MixWords(words);
            foreach (var words in snapshot.Samplers) MixWords(words);
            MixWords(snapshot.FlattenedResourceTable);
            MixWords(snapshot.UserData);
            MixWords(snapshot.MipStatistics ?? []);
            foreach (var range in snapshot.DeviceAddressRanges) Mix((ulong)range.GetHashCode());
        }

        Mix(programs.Vertex.Id);
        Mix(programs.Pixel.Id);
        Mix(programs.DropUnwrittenColorTargets ? 1u : 0u);
        var vertex = programs.VertexInput;
        foreach (var buffer in vertex.Buffers) Mix((ulong)buffer.GetHashCode());
        foreach (var attribute in vertex.Attributes) Mix((ulong)attribute.GetHashCode());
        Mix((ulong)(vertex.ClipSpace with { UserClipPlanes = null }).GetHashCode());
        Mix((uint)vertex.StartInstanceUserData);
        Mix((vertex.IeeeMode ? 1u : 0u) | (vertex.Dx10Clamp ? 2u : 0u) | ((ulong)vertex.FloatMode << 8));
        MixStage(vertex.Stage);
        var pixel = programs.PixelInput;
        Mix(pixel.InputCount);
        Mix(pixel.CustomInterpolationMask);
        MixWords(pixel.InterpolatorSettings);
        foreach (var mode in pixel.TargetOutputModes) Mix(mode);
        foreach (var mapping in pixel.TargetExportMappings) Mix(mapping.Packed);
        foreach (var format in pixel.TargetExportFormats) Mix(format);
        Mix((pixel.IeeeMode ? 1u : 0u) | (pixel.Dx10Clamp ? 2u : 0u) | (pixel.GuestWave32 ? 4u : 0u) | ((ulong)pixel.FloatMode << 8));
        MixStage(pixel.Stage);
        return hash;
    }
}
