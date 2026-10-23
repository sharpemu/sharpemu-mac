// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;
using SharpEmu.Libs.Gpu.Scheduling;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost
{
    // Selection inputs only. Acquiring images, buffers, write effects and residency
    // still happens at this draw's ordered position in BindResources.
    internal sealed class PreparedStageRecipe
    {
        internal required ShaderStageResources Stage;
        internal required ResourceSlotIdentifier[] Images;
        internal required ImageRequest[] Requests;
        internal required bool[] Storage;
        internal required nint[] Samplers;
        internal required ulong[] SamplerIds;
        internal required uint[] ShaderData;
        internal long ImageSet, Shape, Mapping;
        internal long MovieEpoch;
        internal uint MovieWidth, MovieHeight;
        internal bool HadMovieFrame;
        internal MetalRecipeArena.Record? Arguments;
        internal long ArgumentGeneration;
        internal ulong ArgumentSerial;
        internal bool SerialLocal, UsesDeviceAddresses;
        internal ulong[] Addresses = [], TextureIds = [], Bindless = [];
        internal uint[] Lengths = [];
        internal ulong[]?[] Mips = [];
        internal nint[] UsedResources = [];
        internal nuint[] UsedUsage = [];
        internal uint[] ArgumentShaderData = [];
        internal uint[] ArgumentFlattenedTable = [];
        internal MetalRecipeArena.Slab[] ReferencedSlabs = [];
    }
    // ResourcePatchPlan preserves these immutable arrays when it patches roots.
    // User/scalar/buffer address churn must not create another selection recipe.
    private readonly record struct PreparedStageKey(ShaderProgramInfo Program, uint[][] Images, uint[][] Samplers);
    private static PreparedStageKey PreparedKey(ShaderStageResources stage) => new(stage.Program!, stage.Resources.Images, stage.Resources.Samplers);
    private readonly Dictionary<PreparedStageKey, PreparedStageRecipe> _preparedRecipes = new();
    private long _preparedRecipeBytes;
    private MetalRecipeArena? _replayArena;
    private byte[] _argumentReplayScratch = [];
    private bool _replayTransientPointer;
    private long _argumentBuildGeneration;
    internal long ArgumentDataPatches { get; private set; }
    private bool _recordingArgumentUses;
    private readonly List<(nint Resource, nuint Usage)> _argumentUses = new();

    private bool TryReplayBindings(ShaderStageResources stage, out MetalStageBindings prepared)
    {
        prepared = _stageBindings[StageIndexOf(stage.Program!.Stage)];
        if (!RecipeReplayEnabled || !_preparedRecipes.TryGetValue(PreparedKey(stage), out var recipe)) return false;
        // A failed movie-pair search depends on these descriptors and frame shape,
        // not frame pixels or the last remembered pair. The bridge keeps its last
        // frame after a cinematic; its mere existence must not veto in-world reuse.
        if (recipe.Images.Length != 0 && (recipe.HadMovieFrame != _hostMovie.HasFrame || recipe.MovieEpoch != _hostMovie.PathEpoch ||
            recipe.MovieWidth != _hostMovie.FrameWidth || recipe.MovieHeight != _hostMovie.FrameHeight)) return false;
        var info = stage.Program!.Resources!.Info;
        if (stage.Resources.Buffers.Length != info.Buffers.Count || stage.Resources.Images.Length != info.Images.Count ||
            stage.Resources.Samplers.Length != info.Samplers.Count) return false;
        if (recipe.Mapping != DeviceAddressMappingVersion() || recipe.Images.Length != 0 &&
            (recipe.ImageSet != _imageCache.ImageSetVersion || recipe.Shape != _imageCache.Epochs.Shape))
        {
            _replayBindingRefused++; return false;
        }
        prepared.StageResources = stage; prepared.Program = stage.Program!; prepared.Committed = false;
        prepared.EnsureCapacity(recipe.Images.Length, recipe.Samplers.Length, stage.Resources.Buffers.Length, recipe.ShaderData.Length);
        recipe.Images.CopyTo(prepared.Images, 0); recipe.Requests.CopyTo(prepared.Requests, 0);
        recipe.Storage.CopyTo(prepared.IsStorage, 0); recipe.Samplers.CopyTo(prepared.Samplers, 0);
        recipe.SamplerIds.CopyTo(prepared.SamplerIds, 0); recipe.ShaderData.CopyTo(prepared.ShaderData, 0);
        var layout = prepared.Program.Bindings!;
        for (var i = 0; i < layout.UserDataRegisters.Count; i++)
        {
            var register = layout.UserDataRegisters[i];
            var at = (int)(register - prepared.Program.UserDataBase);
            if (register < prepared.Program.UserDataBase || at >= stage.Resources.UserData.Length)
                throw Fatal("A replay user register is outside the stage's current user data.");
            prepared.ShaderData[i] = stage.Resources.UserData[at];
        }
        if (layout.UsesShaderBase)
        {
            prepared.ShaderData[layout.ShaderBaseDword] = (uint)stage.ShaderBase;
            prepared.ShaderData[layout.ShaderBaseDword + 1] = (uint)(stage.ShaderBase >> 32);
        }
        if (layout.BufferWordCount != 0)
            for (var i = 0; i < Math.Min(stage.Resources.Buffers.Length, (int)layout.MemoryOffsetCount); i++)
            {
                var (stride, word3) = PortableBufferWord.Pack(stage.Resources.Buffers[i]);
                var at = (int)layout.FirstBufferWordDword() + i * PortableBufferWord.DwordCount;
                prepared.ShaderData[at] = stride; prepared.ShaderData[at + 1] = word3;
            }
        stage.WriteDispatchThreadLimits(prepared.ShaderData.AsSpan(0, (int)layout.ShaderDataDwordCount));
        prepared.MovieLuma = prepared.MovieChroma = -1;
        prepared.Replay = recipe;
        _preparedStageMask |= 1 << StageIndexOf(stage.Program!.Stage);
        _replayBindings++;
        if (_replayIssue) _replayIssueBindings++;
        return true;
    }

    private void KeepBindingRecipe(MetalStageBindings prepared)
    {
        if (!RecipeReplayEnabled || prepared.MovieLuma >= 0 || prepared.MovieChroma >= 0) return;
        var bytes = 512L + prepared.ImageCount * 128L + prepared.SamplerCount * 32L +
            prepared.BufferCount * 64L + prepared.ShaderData.Length * 4L +
            prepared.StageResources.Resources.FlattenedResourceTable.Length * 4L +
            prepared.StageResources.Resources.UserData.Length * 4L;
        foreach (var table in prepared.StageResources.Resources.BindlessImages)
            bytes += 256L + table.Descriptors.Length * 64L + table.Entries.Length * 4L;
        if (bytes > 16 * 1024 * 1024) return;
        if (_preparedRecipes.Count >= 8192 || _preparedRecipeBytes + bytes > 16 * 1024 * 1024)
        {
            _preparedRecipes.Clear(); _preparedRecipeBytes = 0;
        }
        var recipe = new PreparedStageRecipe
        {
            Stage = prepared.StageResources,
            Images = prepared.Images.AsSpan(0, prepared.ImageCount).ToArray(),
            Requests = prepared.Requests.AsSpan(0, prepared.ImageCount).ToArray(),
            Storage = prepared.IsStorage.AsSpan(0, prepared.ImageCount).ToArray(),
            Samplers = prepared.Samplers.AsSpan(0, prepared.SamplerCount).ToArray(),
            SamplerIds = prepared.SamplerIds.AsSpan(0, prepared.SamplerCount).ToArray(),
            ShaderData = prepared.ShaderData.AsSpan(0, (int)prepared.Program.Bindings!.ShaderDataDwordCount).ToArray(),
            ImageSet = _imageCache.ImageSetVersion, Shape = _imageCache.Epochs.Shape, Mapping = DeviceAddressMappingVersion(),
            HadMovieFrame = _hostMovie.HasFrame, MovieEpoch = _hostMovie.PathEpoch,
            MovieWidth = _hostMovie.FrameWidth, MovieHeight = _hostMovie.FrameHeight,
        };
        _preparedRecipes[PreparedKey(prepared.StageResources)] = recipe;
        _preparedRecipeBytes += bytes; prepared.Replay = recipe;
    }

    private bool TryReplayArguments(MetalStageBindings stage, int stageIndex)
    {
        if (!RecipeReplayEnabled || PpDummyTex || stage.Replay is not { Arguments: { } record } recipe ||
            _replayArena is not { } arena || recipe.ArgumentGeneration != arena.Generation ||
            recipe.Addresses.Length != stage.BufferCount || recipe.TextureIds.Length != stage.ImageCount ||
            !recipe.Bindless.AsSpan().SequenceEqual(stage.BindlessTableAddresses)) return false;
        for (var i = 0; i < stage.BufferCount; i++)
            if (recipe.Addresses[i] != stage.Buffers[i].GpuAddress || recipe.Lengths[i] != stage.BufferLengths[i]) return false;
        for (var i = 0; i < stage.ImageCount; i++)
            if (recipe.TextureIds[i] != stage.TextureIds[i] ||
                !((ReadOnlySpan<ulong>)(recipe.Mips[i] ?? [])).SequenceEqual(stage.MipTextureIds[i] ?? [])) return false;
        var changedData = (!ReferenceEquals(recipe.ArgumentFlattenedTable, stage.StageResources.Resources.FlattenedResourceTable) &&
            !recipe.ArgumentFlattenedTable.AsSpan().SequenceEqual(stage.StageResources.Resources.FlattenedResourceTable)) ||
            !recipe.ArgumentShaderData.AsSpan().SequenceEqual(stage.ShaderData.AsSpan(0, (int)stage.Program.Bindings!.ShaderDataDwordCount));
        if (changedData && !HostLookupCounters.Enabled) return false;
        // An indirect pointer can reference a DIFFERENT slab from the bound
        // argument record. Keep every such slab alive through this issue too.
        foreach (var slab in recipe.ReferencedSlabs) slab.LastSerial = Math.Max(slab.LastSerial, _submissionSerial);
        if ((changedData || recipe.SerialLocal && recipe.ArgumentSerial != _submissionSerial) &&
            !TryPatchArgumentPointers(stage, recipe, ref record, changedData)) return false;
        if (changedData)
        {
            ArgumentDataPatches++;
            HostLookupCounters.Note(HostLookupCounters.Cache.ArgumentPatch, true);
        }
        record.Use(_submissionSerial);
        var range = record.Range;
        _stageArgumentAddresses[stageIndex] = range.GpuAddress;
        _stageArgumentBuffers[stageIndex] = range.Buffer;
        _stageArgumentOffsets[stageIndex] = range.Offset;
        _stageArgumentValid[stageIndex] = true;
        _stageUsesDeviceAddresses[stageIndex] |= recipe.UsesDeviceAddresses;
        NoteUsedResource(range.Buffer, Mtl.ResourceUsageRead);
        for (var i = 0; i < recipe.UsedResources.Length; i++)
            NoteUsedResource(recipe.UsedResources[i], recipe.UsedUsage[i]);
        _replayArguments++;
        _totalReplayArguments++;
        return true;
    }

    private unsafe bool TryPatchArgumentPointers(MetalStageBindings stage, PreparedStageRecipe recipe,
        ref MetalRecipeArena.Record record, bool changedData = false)
    {
        var layout = ArgumentPlanOf(stage.Program.Bindings!).Layout;
        // Mip-statistics banks have their own retirement/declaration path.
        if (layout.Fields.Any(field => field.Kind == DescriptorBindingKind.MipStatistics)) return false;
        if (_argumentReplayScratch.Length < record.Length) _argumentReplayScratch = new byte[record.Length];
        var bytes = _argumentReplayScratch.AsSpan(0, record.Length);
        _replayTransientPointer = false;
        new ReadOnlySpan<byte>((byte*)record.Owner.Contents + record.Offset, record.Length).CopyTo(bytes);
        foreach (var field in layout.Fields)
        {
            var destination = bytes.Slice((int)field.ByteOffset, (int)field.ByteSize);
            if (field.FieldKind == MslArgumentFieldKind.StepReport)
                System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(destination,
                    StepReportAddress(stage.Program.Hash, StepReportStageName(StageIndexOf(stage.Program.Stage))));
            else if (field.FieldKind == MslArgumentFieldKind.Pointer &&
                (field.Kind is DescriptorBindingKind.DeviceAddressPageTable or DescriptorBindingKind.FaultBuffer ||
                 changedData && field.Kind is DescriptorBindingKind.ShaderData or DescriptorBindingKind.FlattenedResourceTable))
                System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(destination, PointerFieldValue(stage, field.Kind));
            else if (field.FieldKind == MslArgumentFieldKind.Word &&
                (field.Kind == DescriptorBindingKind.DeviceAddressPageTable || changedData &&
                 field.Kind is DescriptorBindingKind.ShaderData or DescriptorBindingKind.FlattenedResourceTable))
                System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(destination, WordFieldValue(stage, field.Kind));
        }
        if (_replayTransientPointer) return false;
        var generation = _replayArena!.Generation;
        if (!_replayArena.Intern(bytes, _submissionSerial, CompletedSerialHint, out var current)) return false;
        record = current;
        // Changed scalar/table pointers belong only to this immutable result. Keep
        // the original static template and its leases; each later issue patches its
        // own current data, so scalar churn cannot accumulate stale slab references.
        if (!changedData)
        {
            recipe.Arguments = generation == _replayArena.Generation ? current : null;
            recipe.ArgumentGeneration = _replayArena.Generation;
            recipe.ArgumentSerial = _submissionSerial;
        }
        _replayArgumentPatches++;
        return true;
    }

    private void KeepArgumentRecipe(MetalStageBindings stage, int stageIndex, Gen5MslArgumentLayout layout,
        ReadOnlySpan<byte> bytes, int usedStart, ref MetalGuestRange range)
    {
        var arena = _replayArena ??= new(_device);
        if (_replayTransientPointer || !arena.Intern(bytes, _submissionSerial, CompletedSerialHint, out var record)) return;
        range = record.Range; NoteUsedResource(range.Buffer, Mtl.ResourceUsageRead);
        if (stage.Replay is not { } recipe || _argumentBuildGeneration != arena.Generation) return;
        recipe.Arguments = record; recipe.ArgumentGeneration = arena.Generation;
        recipe.ArgumentSerial = _submissionSerial;
        recipe.SerialLocal = layout.Fields.Any(field => field.FieldKind is MslArgumentFieldKind.StepReport or MslArgumentFieldKind.BindlessImageTables ||
            field.Kind is DescriptorBindingKind.DeviceAddressPageTable or DescriptorBindingKind.FaultBuffer or DescriptorBindingKind.MipStatistics);
        recipe.UsesDeviceAddresses = _stageUsesDeviceAddresses[stageIndex];
        recipe.ArgumentFlattenedTable = stage.StageResources.Resources.FlattenedResourceTable;
        var shaderDwords = (int)stage.Program.Bindings!.ShaderDataDwordCount;
        if (recipe.ArgumentShaderData.Length != shaderDwords) recipe.ArgumentShaderData = new uint[shaderDwords];
        stage.ShaderData.AsSpan(0, shaderDwords).CopyTo(recipe.ArgumentShaderData);
        if (recipe.Addresses.Length != stage.BufferCount)
        {
            recipe.Addresses = new ulong[stage.BufferCount]; recipe.Lengths = new uint[stage.BufferCount];
        }
        for (var i = 0; i < stage.BufferCount; i++) recipe.Addresses[i] = stage.Buffers[i].GpuAddress;
        stage.BufferLengths.AsSpan(0, stage.BufferCount).CopyTo(recipe.Lengths);
        if (recipe.TextureIds.Length != stage.ImageCount)
        {
            recipe.TextureIds = new ulong[stage.ImageCount]; recipe.Mips = new ulong[]?[stage.ImageCount];
        }
        stage.TextureIds.AsSpan(0, stage.ImageCount).CopyTo(recipe.TextureIds);
        for (var i = 0; i < stage.ImageCount; i++)
        {
            var mips = stage.MipTextureIds[i];
            if (mips is null) recipe.Mips[i] = null;
            else
            {
                if (recipe.Mips[i]?.Length != mips.Length) recipe.Mips[i] = new ulong[mips.Length];
                mips.AsSpan().CopyTo(recipe.Mips[i]);
            }
        }
        if (recipe.Bindless.Length != stage.BindlessTableAddresses.Length) recipe.Bindless = new ulong[stage.BindlessTableAddresses.Length];
        stage.BindlessTableAddresses.CopyTo(recipe.Bindless, 0);
        // This stage's declarations, including calls deduped from a prior stage.
        // Retaining another stage's textures would outlive its identity proof.
        if (recipe.UsedResources.Length != _argumentUses.Count)
        {
            recipe.UsedResources = new nint[_argumentUses.Count]; recipe.UsedUsage = new nuint[_argumentUses.Count];
        }
        for (var i = 0; i < _argumentUses.Count; i++)
        {
            recipe.UsedResources[i] = _argumentUses[i].Resource; recipe.UsedUsage[i] = _argumentUses[i].Usage;
        }
        recipe.ReferencedSlabs = arena.References(recipe.UsedResources);
    }

    private ulong UploadReplayDwords(ReadOnlySpan<uint> words)
    {
        var arena = _replayArena ??= new(_device);
        var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(words);
        if (!bytes.IsEmpty && arena.Intern(bytes, _submissionSerial, CompletedSerialHint, out var record))
        {
            NoteUsedResource(record.Range.Buffer, Mtl.ResourceUsageRead);
            return record.Range.GpuAddress;
        }
        _replayTransientPointer = true;
        return UploadDwords(words);
    }
}
