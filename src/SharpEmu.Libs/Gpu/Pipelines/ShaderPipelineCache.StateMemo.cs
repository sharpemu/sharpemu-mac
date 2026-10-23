// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;

namespace SharpEmu.Libs.Gpu.Pipelines;

internal sealed partial class ShaderPipelineCache
{
    // Value snapshots, never references to the mutable register bank. Snapshot/restore and
    // direct register writers therefore obey the same invalidation rules.
    private sealed class PipelineStateMemo
    {
        public readonly GraphicsPipelineKey Key = new()
        {
            Rendering = new(), VertexInput = new(), StaticParameters = new(),
        };
        private readonly ColorGroup[] _colors = new ColorGroup[8];
        private int _colorCount = -1;
        private ColorRegisters _colorRegisters;
        private RasterGroup _raster;
        private PipelineHandle _ready;
        private bool _initialized;

        private readonly record struct ColorRegisters(uint Mask, uint ShaderMask, ulong ExportModes,
            uint AlphaToMask, bool AlphaDisable, uint ExportMask, bool PixelActive, bool DisableBlending,
            bool SecondarySource, uint Target0Info);
        private readonly record struct ColorGroup(uint Slot, GpuFormat Format, bool Valid, byte Mapping, ulong Blend, uint Info);
        private readonly record struct RasterGroup(bool WithDepth, GpuFormat Depth, GpuFormat Stencil,
            bool Bounds, uint Min, uint Max, bool StencilTest, StencilOperations Front, StencilOperations Back,
            bool DirectX, bool DepthClip, bool CullFront, bool CullBack, bool Clockwise,
            GpuPrimitiveTopology Topology, bool Restart, uint Samples, uint AttachmentlessSamples, bool SampleShading);

        private static ulong BlendBits(BlendRegisters blend) =>
            blend.ColorSourceFactor | ((ulong)blend.ColorFunction << 8) | ((ulong)blend.ColorDestinationFactor << 16) |
            ((ulong)blend.AlphaSourceFactor << 24) | ((ulong)blend.AlphaFunction << 32) |
            ((ulong)blend.AlphaDestinationFactor << 40) | (blend.SeparateAlpha ? 1UL << 48 : 0) |
            (blend.Enable ? 1UL << 49 : 0);

        public bool Update(ReadOnlySpan<ColorTargetState> colors, in DepthAttachmentState depth,
            VertexInputInfo vertex, PixelInputInfo? pixel, ContextRegisters context, in RenderingState rendering,
            GpuPrimitiveTopology topology, bool restart, bool disableBlending, ShaderProgram vs, ShaderProgram ps,
            GpuSampleCount noAttachmentSamples, bool depthBounds, bool lodStatistics, out PipelineHandle ready)
        {
            ulong exportModes = 0;
            for (var i = 0; i < context.ShaderInterface.TargetOutputModes.Length; i++)
                exportModes |= (ulong)context.ShaderInterface.TargetOutputModes[i] << (i * 8);
            var colorRegisters = new ColorRegisters(context.RenderTargetMask, context.ShaderInterface.ColorShaderMask,
                exportModes, context.AlphaToMask, context.ShaderInterface.DepthShaderControl.AlphaToMaskDisable,
                pixel?.Stage.Program?.PixelColorExportMasks ?? 0, pixel is not null, disableBlending,
                context.BlendControls[0].UsesSecondarySource, context.ColorTargets[0].Info);
            var colorsChanged = !_initialized || _colorCount != colors.Length || _colorRegisters != colorRegisters;
            for (var i = 0; i < colors.Length; i++)
            {
                ref readonly var color = ref colors[i];
                var group = new ColorGroup(color.Slot, rendering.ColorAttachments[i].Format, color.Image.IsValid,
                    color.Resolution.ExportMapping.Packed, BlendBits(context.BlendControls[color.Slot]), context.ColorTargets[color.Slot].Info);
                colorsChanged |= _colors[i] != group;
                _colors[i] = group;
            }
            var state = depth.HasTarget ? depth.Target.State : default;
            var mode = context.RasterMode;
            var raster = new RasterGroup(depth.HasTarget, rendering.DepthFormat, rendering.StencilFormat,
                state.DepthBoundsTestEnabled, BitConverter.SingleToUInt32Bits(state.DepthMinBounds),
                BitConverter.SingleToUInt32Bits(state.DepthMaxBounds), state.StencilTestEnabled,
                state.FrontOperations, state.BackOperations, context.Clip.DirectXClipSpace,
                HostViewportTransform.DepthClipOf(context), mode.CullFront, mode.CullBack, mode.FrontFaceClockwise,
                topology, restart, rendering.Samples, context.AntialiasingConfig.SampleCountLog2, pixel?.SampleShading ?? false);
            var rasterChanged = colorsChanged || !_initialized || _raster != raster;
            // Compare only layout fields: changing a vertex buffer's address is a binding
            // change and must not rebuild its pipeline key.
            var vertexChanged = !_initialized || !VertexLayoutMatches(vertex, Key.VertexInput);
            var programsChanged = Key.VertexProgramId != vs.Id || Key.PixelProgramId != (pixel is null ? 0 : ps.Id) || Key.LodStatistics != lodStatistics;
            if (colorsChanged)
            {
                Key.Rendering.Reset();
                Key.StaticParameters.Reset();
            }
            if (vertexChanged) Key.VertexInput.Reset();
            if (colorsChanged || rasterChanged || vertexChanged)
                FillGraphicsState(colors, depth, vertex, pixel, context, rendering, topology, restart, disableBlending,
                    vs, ps, noAttachmentSamples, Key.Rendering, Key.StaticParameters, Key.VertexInput, depthBounds,
                    colorsChanged, rasterChanged, vertexChanged);
            Key.VertexProgramId = vs.Id;
            Key.PixelProgramId = pixel is null ? 0 : ps.Id;
            Key.LodStatistics = lodStatistics;
            _colorCount = colors.Length;
            _colorRegisters = colorRegisters;
            _raster = raster;
            _initialized = true;
            if (colorsChanged || rasterChanged || vertexChanged || programsChanged) _ready = default;
            ready = _ready;
            return _ready.Pipeline != 0;
        }

        public void Publish(PipelineHandle ready) => _ready = ready;

        private static bool VertexLayoutMatches(VertexInputInfo info, PipelineVertexInputState key)
        {
            if (info.Buffers.Length != key.BindingCount || info.Attributes.Length != key.AttributeCount) return false;
            for (var i = 0; i < info.Buffers.Length; i++)
                if (key.Bindings[i] != new PipelineVertexBinding(info.Buffers[i].Stride, info.Buffers[i].PerInstance)) return false;
            for (var i = 0; i < info.Attributes.Length; i++)
                if (info.Attributes[i].BufferIndex < 0 || info.Attributes[i].BufferIndex >= info.Buffers.Length || key.Attributes[i] != new PipelineVertexAttribute(info.Attributes[i].OffsetBytes, (byte)info.Attributes[i].BufferIndex)) return false;
            return true;
        }
    }

    [ThreadStatic]
    private static ShaderPipelineCache? _stateMemoOwner;
    [ThreadStatic]
    private static PipelineStateMemo? _stateMemo;
    private readonly Dictionary<(ulong Vertex, ulong Pixel), PipelineStateMemo> _recipePipelineStates = new();
}
