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
