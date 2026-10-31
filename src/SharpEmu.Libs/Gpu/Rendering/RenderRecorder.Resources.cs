// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Libs.Gpu.Rendering;

// The recording half of the fork's draw path: pipelines left out while they compile, deferred
// fragment-stage stores of tile-based hosts, and the emission of converted draws.
internal sealed partial class RenderRecorder
{
    // A draw stored to memory from its fragment stage and, because the host defers it, the
    // render pass it drew in (_fragmentWritesPass, the host's pass serial) was left open.
    private bool _fragmentWritesPending;
    private long _fragmentWritesPass;

    // A draw stored to memory from its vertex stage through a device address or a storage image,
    // which no barrier after the draw orders, in the pass with this serial.
    private bool _vertexStoresUnordered;
    private long _vertexStoresPass;

    /// <summary>
    /// Whether the open render pass holds shader stores no barrier has ordered yet: fragment stores
    /// whose pass end is deferred, or vertex stores the draw's own barrier does not cover. A guest
    /// cache flush or invalidate in the pass has nothing else to make visible to later draws.
    /// </summary>
    internal bool HasUnorderedStoresInOpenPass =>
        (_fragmentWritesPending && _host.RenderPassSerial == _fragmentWritesPass) ||
        (_vertexStoresUnordered && _host.RenderPassSerial == _vertexStoresPass);

    // The buffer ranges those deferred stores may have written, or untracked when one of the
    // storing draws wrote through a device address or a storage image.
    private readonly List<(ulong Start, ulong End)> _fragmentWriteRanges = new();
    private bool _fragmentWritesUntracked;

    /// <summary>
    /// A consumer of earlier shader stores can follow: the guest flushed or invalidated its
    /// caches. Ends the render pass that holds deferred fragment-stage stores, if any.
    /// </summary>
    internal void ReleaseShaderWrites() => ReleaseFragmentWrites();

    // Ends the pass a deferred fragment store was made in and records the barrier the storing
    // draw would have recorded. A pass the host has ended since needs neither: the encoder
    // boundary orders its stores, and a barrier now would land in whatever pass is open.
    private void ReleaseFragmentWrites()
    {
        if (!_fragmentWritesPending)
        {
            return;
        }

        _fragmentWritesPending = false;
        if (_host.RenderPassSerial != _fragmentWritesPass)
        {
            return;
        }

        _host.EndRendering("release");
        _host.ShaderWriteBarrier(GpuPipelineStage.FragmentShaderBit);
    }

    // Whether a deferred fragment store may have written the range: always when a storing
    // draw wrote memory no buffer range names.
    private bool FragmentWritesMayCover(ulong address, ulong size)
    {
        if (!_fragmentWritesPending)
        {
            return false;
        }

        if (_fragmentWritesUntracked || _host.RenderPassSerial != _fragmentWritesPass)
        {
            return true;
        }

        var end = address + size;
        foreach (var (start, stop) in _fragmentWriteRanges)
        {
            if (start < end && address < stop)
            {
                return true;
            }
        }

        return false;
    }

    // Notes what a draw whose fragment stores are deferred may write.
    private void NoteFragmentWrites(ShaderStageResources stage)
    {
        if (!_fragmentWritesPending || _host.RenderPassSerial != _fragmentWritesPass)
        {
            _fragmentWriteRanges.Clear();
            _fragmentWritesUntracked = false;
        }

        var program = stage.Program!;
        if (program.WritesDeviceAddresses || RenderExecutor.WritesStorageImage(program) || _fragmentWriteRanges.Count >= MaxFragmentWriteRanges)
        {
            _fragmentWritesUntracked = true;
            return;
        }

        var descriptors = stage.Resources.Buffers;
        for (var i = 0; i < program.Buffers.Length; i++)
        {
            if (!program.Buffers[i].Written || descriptors[i].Length < 4)
            {
                continue;
            }

            var descriptor = BufferDescriptorWords.From(descriptors[i]);
            if (descriptor.Footprint() is not { } footprint || descriptor.Address > ulong.MaxValue - footprint)
            {
                _fragmentWritesUntracked = true;
                return;
            }

            if (descriptor.Address != 0 && footprint != 0)
            {
                _fragmentWriteRanges.Add((descriptor.Address, descriptor.Address + footprint));
            }
        }
    }

    private const int MaxFragmentWriteRanges = 64;

    // The bytes one fetch of a descriptor format reads (formats 113 and 121 are the two the
    // vertex input accepts outside the unified table).
    private uint ConstantAttributeByteSize(in BufferDescriptorWords descriptor)
    {
        if (descriptor.Format == 113) return 16;
        if (descriptor.Format == 121) return 4;
        if (SharpEmu.ShaderCompiler.Gfx10UnifiedFormat.TryDecode(descriptor.Format, out var dataFormat, out _))
        {
            var bytes = SharpEmu.ShaderCompiler.Gfx10UnifiedFormat.GetAccessByteSize(dataFormat, 4);
            if (bytes != 0) return bytes;
        }

        throw _host.Fatal($"The vertex buffer format is unknown: format={descriptor.Format}.");
    }

    /// <summary>
    /// Whether this draw may be left out of the frame while its pipeline compiles, rather than
    /// the command stream waiting for the compile. Leaving a draw out is only safe when the frame
    /// it belongs to is the only thing that misses it: the next frame draws it again into
    /// targets that are rebuilt from nothing every frame. So, when in doubt, it is not:
    /// <list type="bullet">
    /// <item>a program that writes memory (storage buffers or images, atomics, device-address
    /// stores), uses the GDS (ordered counts, append and consume), or writes mip statistics:
    /// what it writes outlives the frame and later work reads it. Mip statistics only when the
    /// host leaves those out too (IRenderHost.LeavesOutMipStatisticsWhileCompiling): they are
    /// the texture streamer's feedback, gathered over frames, and a draw missing from a few
    /// frames only reports its object's mips that much later;</item>
    /// <item>a predicated draw, or one inside an occlusion query: the guest decides on the
    /// count it produces (the interpreter's DrawsMustExecute);</item>
    /// <item>an indirect mesh draw, whose launch was prepared before the pipeline;</item>
    /// <item>a draw whose attachments carry a clear (a fast clear consumed at their
    /// acquire, a load clear): leaving it out would lose the clear;</item>
    /// <item>a draw that writes depth or stencil, unless the host leaves those out too
    /// (IRenderHost.LeavesOutDepthWritesWhileCompiling): the depth target is what the rest of
    /// the frame tests and samples, and a missing occluder there shows in every later pass of
    /// the frame - but of that frame alone when the host's check below finds the depth target
    /// rebuilt every frame as well;</item>
    /// <item>and the host's own say on the targets (IRenderHost.MayLeaveOutWhileCompiling),
    /// which leaves out only draws into targets it saw cleared or overwritten whole in this
    /// frame and in the frames before it, so a look-up table, a shadow or virtual-texture
    /// cache, a capture or a history buffer is never one of them.</item>
    /// </list>
    /// This executor has no stream-out; a host that adds it treats such a draw as writing memory.
    /// </summary>
    private bool MayLeaveOutWhileCompiling(
        ResolvedDraw record, ref DrawState state, ShaderStageResources vertexStage, ShaderStageResources? pixelStage, bool indirectMesh)
    {
        if (!MayLeaveOutBeforeAcquiring(record, ref state, vertexStage, pixelStage, indirectMesh))
        {
            return false;
        }

        ref readonly var rendering = ref state.Rendering;
        for (var index = 0; index < rendering.ColorAttachmentCount; index++)
        {
            if (rendering.ColorAttachments[index].IsClear)
            {
                _host.NoteLeaveOutRefused("color-clear");
                return false;
            }
        }

        if (state.Depth.HasTarget &&
            (rendering.DepthStencilAttachment.DepthClear || rendering.DepthStencilAttachment.StencilClear))
        {
            _host.NoteLeaveOutRefused("depth-clear");
            return false;
        }

        return _host.MayLeaveOutWhileCompiling(in rendering);
    }

    // The part of MayLeaveOutWhileCompiling the draw knows before it acquires anything. False
    // here is false there: acquiring only adds reasons to keep the draw (a clear it consumes, a
    // metadata clear that makes the depth load a write, the host's say on the targets).
    private bool MayLeaveOutBeforeAcquiring(
        ResolvedDraw record, ref DrawState state, ShaderStageResources vertexStage, ShaderStageResources? pixelStage, bool indirectMesh)
    {
        if (!_host.LeavesOutDrawsWhileCompiling)
        {
            return false;
        }

        if (record.MustExecute || indirectMesh)
        {
            _host.NoteLeaveOutRefused("must-execute");
            return false;
        }

        var mipStatistics = _host.LeavesOutMipStatisticsWhileCompiling;
        if (RenderExecutor.WriteOutsideTargets(vertexStage, mipStatistics) is { } vertexWrite)
        {
            _host.NoteLeaveOutRefused(vertexWrite);
            return false;
        }

        if (pixelStage is { } pixel && RenderExecutor.WriteOutsideTargets(pixel, mipStatistics) is { } pixelWrite)
        {
            _host.NoteLeaveOutRefused(pixelWrite);
            return false;
        }

        if (state.Depth.HasTarget)
        {
            var depth = state.Depth.Target;
            if (state.Depth.LoadClear || depth.State.StencilClearEnabled)
            {
                _host.NoteLeaveOutRefused("depth-clear");
                return false;
            }

            if ((!state.Depth.LoadState.IsReadOnly(depth.Target.Format) || depth.State.DepthWriteEnabled) &&
                !_host.LeavesOutDepthWritesWhileCompiling)
            {
                _host.NoteLeaveOutRefused("depth-write");
                return false;
            }
        }

        return true;
    }

    // The vertices of the first host draw call EmitDraw makes; a zero count for an indexed or
    // indirect draw, whose vertices are not known before it runs.
    private static (uint Count, uint First) UpcomingVertices(
        GuestPrimitiveType primitiveType, in DrawCall draw, in DrawEmission emission)
    {
        if (emission.Indexed || emission.IndirectArgumentsAddress != 0 || emission.PatchControlPoints != 0)
        {
            return (0, 0);
        }

        return primitiveType switch
        {
            GuestPrimitiveType.QuadListLegacy => (4, emission.FirstVertex),
            _ => (draw.Count, emission.FirstVertex),
        };
    }

    private void EmitDraw(GuestPrimitiveType primitiveType, VertexInputInfo vertexInput, in DrawCall draw, in DrawEmission emission)
    {
        if (emission.TriangleListIndices != 0)
        {
            _host.DrawIndexed(emission.TriangleListIndices, draw.InstanceCount, 0, emission.VertexOffset, emission.FirstInstance);
            return;
        }

        if (emission.PatchControlPoints != 0)
        {
            _host.DrawPatches(emission.PatchControlPoints, draw.Count, draw.InstanceCount, emission.FirstInstance);
            return;
        }

        switch (primitiveType)
        {
            case GuestPrimitiveType.PointList:
            case GuestPrimitiveType.LineList:
            case GuestPrimitiveType.LineStrip:
            case GuestPrimitiveType.TriangleList:
            case GuestPrimitiveType.TriangleFan:
            case GuestPrimitiveType.TriangleStrip:
            case GuestPrimitiveType.Polygon:
            case GuestPrimitiveType.RectangleList:
            case GuestPrimitiveType.RectangleListLegacy:
                if (emission.Indexed)
                {
                    _host.DrawIndexed(draw.Count, draw.InstanceCount, 0, emission.VertexOffset, emission.FirstInstance);
                }
                else
                {
                    _host.Draw(draw.Count, draw.InstanceCount, emission.FirstVertex, emission.FirstInstance);
                }

                break;
            case GuestPrimitiveType.QuadListLegacy:
                if ((draw.Count & 0x3) != 0)
                {
                    throw _host.Fatal($"A legacy quad list count is not a multiple of four: count={draw.Count}.");
                }

                for (var i = 0u; i < draw.Count; i += 4)
                {
                    if (emission.Indexed)
                    {
                        _host.DrawIndexed(4, draw.InstanceCount, i, emission.VertexOffset, emission.FirstInstance);
                    }
                    else
                    {
                        _host.Draw(4, draw.InstanceCount, i + emission.FirstVertex, emission.FirstInstance);
                    }
                }

                break;
            default:
                throw _host.Fatal($"The primitive type is unknown: primitiveType={(uint)primitiveType}.");
        }
    }
}
