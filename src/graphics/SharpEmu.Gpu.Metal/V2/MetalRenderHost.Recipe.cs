// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.CompilerServices;
using SharpEmu.HLE;
using SharpEmu.HLE.GuestMemory;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.Gpu.Metal.V2;

// All instance state is B-owned (the inline issue consumer when split=0). Only the
// static switch is read outside B. Capture never reads guest bytes, enrolls watches,
// resolves resources, changes encoders, or retains arrays supplied by the executor.
internal sealed partial class MetalRenderHost : IRecipeDiagnosticSink
{
    private static readonly bool RecipeClassifierEnabled =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_RECIPE_CLASSIFIER") == "1";
    private readonly MetalRecipeClassifier? _recipes = RecipeClassifierEnabled ? new() : null;
    IRecipeDiagnosticSink? IShaderPipelineHost.RecipeDiagnostics => _recipes is null ? null : this;
    internal RecipeObservation[] CaptureRecipeIssues()
    {
        if (!_relay.IsGpuQueueThread) throw new InvalidOperationException("Recipe capture requires the issue consumer.");
        return _recipes?.CaptureCurrentFrame() ?? [];
    }
    private RecipeHash _recipeStructure, _recipeValues, _recipeLocations, _recipeDependencies, _recipePass;
    private long _recipeMaterializationTicks;
    private int _recipeMaterializations, _recipeEmissions, _recipeUnknown, _recipeUnobserved, _recipeBindings, _recipeObservedStages;
    private bool _recipeIncomplete;
    private bool _recipeActive, _recipeCompute;
    private long _recipeSequence;
    private int _recipeQueue;
    private readonly ConditionalWeakTable<object, Identity>? _recipeIdentities = RecipeClassifierEnabled ? new() : null;
    private readonly ConditionalWeakTable<ShaderResourcePlan, Roots>? _recipeRoots = RecipeClassifierEnabled ? new() : null;
    private ulong _recipeNextIdentity;
    private readonly record struct Root(int Low, int High, bool Buffer);
    private sealed record Identity(ulong Value);
    private sealed record Roots(Root[] Values);

    private ulong RecipeIdentity(object value)
    {
        if (_recipeIdentities!.TryGetValue(value, out var id)) return id.Value;
        _recipeIdentities.Add(value, id = new(++_recipeNextIdentity));
        return id.Value;
    }
    private void RecipeBegin(bool compute, int queue)
    {
        var start = Stopwatch.GetTimestamp(); var bytes = GC.GetAllocatedBytesForCurrentThread();
        _recipeActive = true; _recipeCompute = compute; _recipeQueue = queue; _recipeSequence++;
        _recipeStructure = _recipeValues = _recipeLocations = _recipeDependencies = _recipePass = default;
        _recipeMaterializationTicks = 0;
        _recipeMaterializations = _recipeEmissions = _recipeUnknown = _recipeUnobserved = _recipeBindings = _recipeObservedStages = 0;
        _recipeIncomplete = false;
        _recipeStructure.Add(compute ? 2UL : 1UL); _recipeStructure.Add(unchecked((ulong)queue));
        // Global versions are deliberately conservative: unrelated writers can veto an
        // observed repeat. value_repeat reports the upper bound with these vetoes removed.
        var epoch = ((IResourceDependencySource)this).DependencyEpoch;
        _recipeDependencies.Add(epoch.GpuWrites); _recipeDependencies.Add(unchecked((ulong)epoch.Overlay));
        _recipeDependencies.Add(unchecked((ulong)epoch.Mapping)); _recipeDependencies.Add(unchecked((ulong)epoch.Images));
        _recipeDependencies.Add(unchecked((ulong)epoch.Labels));
        if (_backing.BackingAliasSnapshot is { } mappings) _recipeDependencies.Add(RecipeIdentity(mappings));
        _recipes!.NoteCost(start, bytes);
    }
    private void RecipeDrawArguments(bool indexed, in DrawIndexedArguments index, in DrawAutoArguments auto)
    {
        var start = Stopwatch.GetTimestamp(); var bytes = GC.GetAllocatedBytesForCurrentThread();
        _recipeStructure.Add(indexed ? 1UL : 0UL);
        if (indexed)
        {
            _recipeStructure.Add(index.Opcode); _recipeValues.Add(index.IndexCount); _recipeValues.Add(index.InstanceCount);
            _recipeValues.Add(index.IndexTypeAndSize); _recipeValues.Add(unchecked((ulong)index.BaseVertex));
            _recipeValues.Add(index.FirstInstance); _recipeValues.Add((ulong)index.OffsetSource);
            _recipeValues.Add(index.UnboundedIndexBuffer ? 1UL : 0); _recipeValues.Add(index.MustExecute ? 1UL : 0);
            RecipeAddress(index.IndexAddress, Math.Max(1UL, (ulong)index.IndexCount * ((index.IndexTypeAndSize & 1) != 0 ? 4UL : 2UL)), contents: true);
            _recipeStructure.Add(index.IndirectArgumentsAddress != 0 ? 1UL : 0);
            if (index.IndirectArgumentsAddress != 0) RecipeAddress(index.IndirectArgumentsAddress, 20, contents: true);
        }
        else
        {
            _recipeStructure.Add(auto.Opcode); _recipeValues.Add(auto.VertexCount); _recipeValues.Add(auto.InstanceCount);
            _recipeValues.Add(auto.FirstVertex); _recipeValues.Add(auto.FirstInstance);
            _recipeValues.Add((ulong)auto.OffsetSource); _recipeValues.Add(auto.MustExecute ? 1UL : 0);
            _recipeStructure.Add(auto.IndirectArgumentsAddress != 0 ? 1UL : 0);
            if (auto.IndirectArgumentsAddress != 0) RecipeAddress(auto.IndirectArgumentsAddress, 16, contents: true);
        }
        _recipes!.NoteCost(start, bytes);
    }
    private void RecipeDispatchArguments(uint x, uint y, uint z, uint initiator, ulong indirect)
    {
        var start = Stopwatch.GetTimestamp(); var bytes = GC.GetAllocatedBytesForCurrentThread();
        _recipeValues.Add(x); _recipeValues.Add(y); _recipeValues.Add(z); _recipeValues.Add(initiator);
        _recipeStructure.Add(indirect != 0 ? 1UL : 0);
        if (indirect != 0) RecipeAddress(indirect, 12, contents: true);
        _recipes!.NoteCost(start, bytes);
    }
    // A mapped view is the finest allocation identity the existing mapping owner
    // publishes. Offset is explicit subtraction, never an address-bit heuristic.
    // Unknown/noncontiguous mappings are kept verbatim and cannot yield relocation.
    private void RecipeAddress(ulong address, ulong size, bool contents = false)
    {
        if (address == 0) { _recipeValues.Add(0); return; }
        var allocation = new RecipeHash(); ulong offset = 0;
        var known = _backing.BackingAliasSnapshot is ViewRecord[] views &&
            RecipeAddressExpression.TryResolve(views, address, size, out allocation, out offset);
        if (known)
        {
            _recipeValues.Add(1); _recipeValues.Add(allocation.A); _recipeValues.Add(allocation.B);
            _recipeLocations.Add(offset);
        }
        if (!known) { _recipeValues.Add(2); _recipeValues.Add(address); _recipeUnknown++; }
        _recipeValues.Add(size);
        if (!contents) return;
        // Observe existing protection only. Never enroll/rearm a token for diagnostics.
        if (GuestImageWriteTracker.TryGetProtectedObservedContentToken(address, size, out var token))
        {
            _recipeDependencies.Add(1);
            _recipeDependencies.Add(unchecked((ulong)token.ProtectedIdentity));
            _recipeDependencies.Add(unchecked((ulong)token.ProtectedGeneration));
            _recipeDependencies.Add(unchecked((ulong)token.WatchIdentity));
            _recipeDependencies.Add(unchecked((ulong)token.WatchGeneration));
        }
        else { _recipeDependencies.Add(0); _recipeUnobserved++; }
    }
    void IRecipeDiagnosticSink.MaterializationElapsed(long ticks)
    {
        if (!_recipeActive) return;
        _recipeMaterializationTicks += ticks; _recipeMaterializations++;
    }
    void IRecipeDiagnosticSink.Materialized(ShaderSource source, ShaderResourcePlan plan, ResourceSnapshot snapshot)
    {
        if (!_recipeActive) return;
        var start = Stopwatch.GetTimestamp(); var bytes = GC.GetAllocatedBytesForCurrentThread();
        _recipeObservedStages |= source.Stage switch
        {
            ShaderStage.Vertex or ShaderStage.Mesh => 1,
            ShaderStage.Pixel => 2,
            ShaderStage.Compute => 4,
            _ => 0,
        };
        _recipeStructure.Add((ulong)source.Stage); _recipeStructure.Add(source.Identity);
        _recipeStructure.Add(RecipeIdentity(plan));
        RecipeAddress(source.Address, Math.Max(1UL, source.CodeSize));
        if (!_recipeRoots!.TryGetValue(plan, out var rootPlan))
        {
            var found = new List<Root>();
            // Only graph-proven address handles and direct V# bases are relocatable.
            // Other user words, including inline constants, remain semantic inputs.
            void Add(ScalarValue low, ScalarValue high, bool buffer)
            {
                if (low.Kind != ScalarValueKind.UserData || high.Kind != ScalarValueKind.UserData) return;
                var lo = (int)(low.UserDataRegister - source.UserDataBase);
                var hi = (int)(high.UserDataRegister - source.UserDataBase);
                if (lo < 0 || lo >= snapshot.UserData.Length || hi < 0 || hi != lo + 1 || hi >= snapshot.UserData.Length || found.Any(r => r.Low == lo && r.High == hi && r.Buffer == buffer)) return;
                found.Add(new(lo, hi, buffer));
            }
            foreach (var node in plan.Graph.Values)
                if (node.Kind == ScalarValueKind.AddressHandle && node.Operands.Length == 2) Add(node.Operands[0], node.Operands[1], false);
            foreach (var buffer in plan.Info.Buffers)
            {
                var words = plan.DescriptorSources[(int)buffer.Source].Dwords;
                if (words.Length == 4) Add(words[0], words[1], true);
            }
            rootPlan = new(found.ToArray());
            _recipeRoots.Add(plan, rootPlan);
        }
        var roots = rootPlan.Values;
        // Capture every proven address expression, including overlapping pairs.
        // A scalar can be the high half of one root and the low half of another.
        foreach (var root in roots)
        {
            var high = snapshot.UserData[root.High];
            if (root.Buffer)
            {
                var descriptor = new BufferDescriptorWords(snapshot.UserData[root.Low], high, 0, 0);
                RecipeAddress(descriptor.Address, 1);
                _recipeValues.Add(descriptor.WithAddress(0).Word1);
            }
            else RecipeAddress(snapshot.UserData[root.Low] | ((ulong)high << 32), 1);
        }
        for (var word = 0; word < snapshot.UserData.Length; word++)
        {
            var addressPart = false;
            foreach (var root in roots)
                if (root.Low == word || root.High == word) { addressPart = true; break; }
            if (!addressPart) _recipeValues.Add(snapshot.UserData[word]);
        }
        _recipeValues.Add((ulong)snapshot.FlattenedResourceTable.Length);
        foreach (var word in snapshot.FlattenedResourceTable) _recipeValues.Add(word);
        foreach (var range in snapshot.DeviceAddressRanges)
        {
            _recipeValues.Add(range.Handle); _recipeValues.Add(range.Planned ? 1UL : 0); _recipeValues.Add(range.Written ? 1UL : 0);
            RecipeAddress(range.Base, range.Size, contents: true);
        }
        foreach (var table in snapshot.BindlessImages)
        {
            _recipeValues.Add(table.Slot); RecipeAddress(table.Address, 1);
            foreach (var variant in table.VariantImages) _recipeValues.Add(variant);
            foreach (var descriptor in table.Descriptors) foreach (var word in descriptor) _recipeValues.Add(word);
            foreach (var variant in table.DescriptorVariants) _recipeValues.Add(variant);
            foreach (var entry in table.Entries) _recipeValues.Add(unchecked((ulong)entry));
        }
        _recipes!.NoteCost(start, bytes);
    }
    private void RecipeBindings(MetalStageBindings stage)
    {
        if (!_recipeActive) return;
        var start = Stopwatch.GetTimestamp(); var bytes = GC.GetAllocatedBytesForCurrentThread();
        _recipeBindings++;
        _recipeStructure.Add(RecipeIdentity(stage.Program)); // Actual specialized program + ABI/layout.
        _recipeStructure.Add((ulong)stage.BufferCount); _recipeStructure.Add((ulong)stage.ImageCount); _recipeStructure.Add((ulong)stage.SamplerCount);
        var snapshot = stage.StageResources.Resources;
        // Scalars actually exposed by the shader ABI remain semantic inputs even
        // when the same SGPR is also a descriptor/address root. No ALU-use guess.
        var inlineWords = stage.Program.Bindings!.UserDataRegisters.Count;
        _recipeValues.Add((ulong)inlineWords);
        foreach (var word in stage.ShaderData.AsSpan(0, inlineWords)) _recipeValues.Add(word);
        if ((_recipeObservedStages & (1 << StageIndexOf(stage.Program.Stage))) == 0)
        {
            // A suspended dispatch can reuse its previously prepared program. Capture
            // its current stored values, but never credit a partial plan/dependency
            // observation as a complete recipe match.
            _recipeIncomplete = true;
            foreach (var word in snapshot.UserData) _recipeValues.Add(word);
            foreach (var word in snapshot.FlattenedResourceTable) _recipeValues.Add(word);
            foreach (var range in snapshot.DeviceAddressRanges)
            {
                _recipeValues.Add(range.Handle); _recipeValues.Add(range.Planned ? 1UL : 0); _recipeValues.Add(range.Written ? 1UL : 0);
                RecipeAddress(range.Base, range.Size, contents: true);
            }
        }
        for (var i = 0; i < stage.BufferCount; i++)
        {
            var words = snapshot.Buffers[i]; _recipeValues.Add((ulong)words.Length);
            if (words.Length < 4) continue;
            var descriptor = BufferDescriptorWords.From(words);
            var shape = descriptor.WithAddress(0);
            _recipeValues.Add(shape.Word1); _recipeValues.Add(shape.Word2); _recipeValues.Add(shape.Word3);
            RecipeAddress(descriptor.Address, Math.Max(1UL, descriptor.Footprint() ?? 0), contents: true);
            var range = stage.Buffers[i];
            _recipeValues.Add(unchecked((ulong)range.Buffer)); _recipeValues.Add(stage.BufferLengths[i]);
            _recipeLocations.Add(range.Offset);
        }
        for (var i = 0; i < stage.ImageCount; i++)
        {
            _recipeValues.Add(stage.Images[i].Index); _recipeValues.Add(stage.Images[i].Generation);
            _recipeValues.Add(stage.TextureIds[i]); _recipeValues.Add(stage.TextureLevels[i]);
            foreach (var word in snapshot.Images[i]) _recipeValues.Add(word);
            if (stage.Images[i].IsValid)
            {
                var image = _imageCache.GetImage(stage.Images[i]);
                _recipeDependencies.Add(unchecked((ulong)image.ContentVersion));
                _recipeDependencies.Add(unchecked((ulong)image.GpuWriteSequence));
            }
            if (stage.MipTextureIds[i] is { } mips) foreach (var id in mips) _recipeValues.Add(id);
        }
        foreach (var table in snapshot.BindlessImages)
        {
            if (!_bindlessTables.TryGetValue(table, out var device)) { _recipeUnobserved++; continue; }
            _recipeValues.Add(table.Slot); _recipeValues.Add((ulong)device.ResourceIds.Length);
            for (var i = 0; i < device.ResourceIds.Length; i++)
            {
                _recipeValues.Add(device.ResourceIds[i]);
                _recipeValues.Add(device.Images[i].Index); _recipeValues.Add(device.Images[i].Generation);
                if (!device.Images[i].IsValid) continue;
                var image = _imageCache.GetImage(device.Images[i]);
                _recipeDependencies.Add(unchecked((ulong)image.ContentVersion));
                _recipeDependencies.Add(unchecked((ulong)image.GpuWriteSequence));
            }
        }
        for (var i = 0; i < stage.SamplerCount; i++)
        {
            // Samplers are immutable and not evicted; resource ID is their lifetime identity.
            _recipeValues.Add(stage.SamplerIds[i]);
            foreach (var word in snapshot.Samplers[i]) _recipeValues.Add(word);
        }
        _recipes!.NoteCost(start, bytes);
    }
    private void RecipeVertexBuffers(ReadOnlySpan<BufferBinding> bindings, VertexInputInfo input)
    {
        if (!_recipeActive) return;
        var start = Stopwatch.GetTimestamp(); var bytes = GC.GetAllocatedBytesForCurrentThread();
        _recipeValues.Add((ulong)bindings.Length);
        for (var i = 0; i < bindings.Length; i++)
        {
            _recipeValues.Add(bindings[i].Handle); _recipeLocations.Add(bindings[i].Offset);
            if (i >= input.Buffers.Length) continue;
            var buffer = input.Buffers[i];
            _recipeValues.Add(buffer.Stride); _recipeValues.Add(buffer.RecordCount);
            _recipeValues.Add(buffer.PerInstance ? 1UL : 0); _recipeValues.Add(buffer.Divisor);
            RecipeAddress(buffer.Address, Math.Max(1UL, buffer.Size), contents: true);
        }
        _recipes!.NoteCost(start, bytes);
    }
    private void RecipePipeline(in PipelineHandle pipeline)
    {
        if (!_recipeActive) return;
        var start = Stopwatch.GetTimestamp(); var bytes = GC.GetAllocatedBytesForCurrentThread();
        _recipeStructure.Add(pipeline.Pipeline); _recipeStructure.Add(pipeline.Layout);
        _recipeStructure.Add(pipeline.UsesPushDescriptors ? 1UL : 0);
        _recipes!.NoteCost(start, bytes);
    }
    private void RecipeBufferLocation(ulong kind, BufferBinding binding, ulong type = 0)
    {
        if (!_recipeActive) return;
        var start = Stopwatch.GetTimestamp(); var bytes = GC.GetAllocatedBytesForCurrentThread();
        _recipeStructure.Add(kind); _recipeValues.Add(binding.Handle); _recipeValues.Add(type);
        _recipeLocations.Add(binding.Offset);
        _recipes!.NoteCost(start, bytes);
    }
    private void RecipeEmission(ulong kind, ulong a, ulong b = 0, ulong c = 0, ulong d = 0, ulong e = 0)
    {
        if (!_recipeActive) return;
        var start = Stopwatch.GetTimestamp(); var bytes = GC.GetAllocatedBytesForCurrentThread();
        _recipeStructure.Add(kind); _recipeValues.Add(a); _recipeValues.Add(b);
        _recipeValues.Add(c); _recipeValues.Add(d); _recipeValues.Add(e);
        _recipes!.NoteCost(start, bytes);
    }
    private void RecipeEnd()
    {
        var start = Stopwatch.GetTimestamp(); var bytes = GC.GetAllocatedBytesForCurrentThread();
        _recipeActive = false;
        _recipeValues.Add((ulong)_recipeEmissions); _recipeValues.Add((ulong)_recipeBindings);
        if (_recipeBindings == 0 || _recipeIncomplete) _recipeValues.Add(unchecked((ulong)_recipeSequence)); // Unprepared/fast-path issue cannot claim replay.
        if (_recipeCompute) { _recipePass.Add(2); }
        var signature = new RecipeSignature(_recipePass, _recipeStructure, _recipeValues, _recipeLocations, _recipeDependencies);
        _recipes!.Record(new(signature, _recipeMaterializationTicks, _recipeMaterializations, _recipeEmissions,
            _recipeUnknown, _recipeUnobserved, _recipeCompute ? ComputeEncoders : _renderPassSerial,
            _recipes.FrameId, _recipeSequence, _recipeQueue, _recipeCompute, _recipeBindings != 0 && !_recipeIncomplete));
        _recipes.NoteCost(start, bytes);
    }
    private void RecipeTargets(in RenderingState state)
    {
        if (!_recipeActive) return;
        var start = Stopwatch.GetTimestamp(); var bytes = GC.GetAllocatedBytesForCurrentThread();
        _recipePass = default;
        _recipePass.Add(1); _recipePass.Add(state.Width); _recipePass.Add(state.Height); _recipePass.Add(state.Samples); _recipePass.Add(state.Layers);
        _recipePass.Add(state.ColorAttachmentCount);
        void Attachment(in RenderingAttachment attachment, uint level, uint slice, uint plane)
        {
            _recipePass.Add(attachment.View.Handle); _recipePass.Add((ulong)attachment.Format);
            _recipePass.Add(level); _recipePass.Add(slice); _recipePass.Add(plane);
            _recipeValues.Add((ulong)attachment.Layout); _recipeValues.Add(attachment.IsClear ? 1UL : 0);
            _recipeValues.Add(attachment.ClearWord0); _recipeValues.Add(attachment.ClearWord1); _recipeValues.Add(attachment.ClearWord2); _recipeValues.Add(attachment.ClearWord3);
            _recipeValues.Add(attachment.HasDepth ? 1UL : 0); _recipeValues.Add(attachment.HasStencil ? 1UL : 0);
            _recipeValues.Add(attachment.DepthClear ? 1UL : 0); _recipeValues.Add(attachment.StencilClear ? 1UL : 0);
        }
        for (var i = 0; i < state.ColorAttachmentCount; i++) Attachment(state.ColorAttachments[i], _attachmentLevels[i], _attachmentSlices[i], _attachmentDepthPlanes[i]);
        Attachment(state.DepthStencilAttachment, _depthAttachmentLevel, _depthAttachmentSlice, 0);
        _recipeValues.Add(state.RasterizesWithoutAttachments ? 1UL : 0);
        _recipes!.NoteCost(start, bytes);
    }
    private void RecipeDynamic(in DynamicDrawState s)
    {
        if (!_recipeActive) return;
        var start = Stopwatch.GetTimestamp(); var bytes = GC.GetAllocatedBytesForCurrentThread();
        void F(float v) => _recipeValues.Add(BitConverter.SingleToUInt32Bits(v));
        void B(bool v) => _recipeValues.Add(v ? 1UL : 0);
        F(s.ViewportX); F(s.ViewportY); F(s.ViewportWidth); F(s.ViewportHeight); F(s.ViewportMinDepth); F(s.ViewportMaxDepth);
        _recipeValues.Add(unchecked((ulong)s.Scissor.Left)); _recipeValues.Add(unchecked((ulong)s.Scissor.Top));
        _recipeValues.Add(unchecked((ulong)s.Scissor.Right)); _recipeValues.Add(unchecked((ulong)s.Scissor.Bottom));
        F(s.LineWidth); F(s.BlendRed); F(s.BlendGreen); F(s.BlendBlue); F(s.BlendAlpha);
        B(s.DepthTestEnabled); B(s.DepthWriteEnabled); _recipeValues.Add((ulong)s.DepthCompare);
        B(s.DepthBiasEnabled); F(s.DepthBiasConstantFactor); F(s.DepthBiasClamp); F(s.DepthBiasSlopeFactor);
        B(s.DepthBoundsTestEnabled); F(s.DepthMinBounds); F(s.DepthMaxBounds); B(s.StencilTestEnabled);
        _recipeValues.Add(s.FrontStencil.CompareMask); _recipeValues.Add(s.FrontStencil.WriteMask); _recipeValues.Add(s.FrontStencil.Reference);
        _recipeValues.Add(s.BackStencil.CompareMask); _recipeValues.Add(s.BackStencil.WriteMask); _recipeValues.Add(s.BackStencil.Reference);
        _recipeValues.Add(s.ColorWriteCount); _recipeValues.Add(s.ColorWriteEnableMask);
        B(s.DepthBiasPerFace); F(s.DepthBiasBackConstantFactor); F(s.DepthBiasBackSlopeFactor);
        F(s.DepthBiasPointLineConstantFactor); F(s.DepthBiasPointLineSlopeFactor); _recipeValues.Add(s.DepthCountControl);
        _recipes!.NoteCost(start, bytes);
    }
}
