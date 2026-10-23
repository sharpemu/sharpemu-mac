// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Runtime.InteropServices;
using SharpEmu.Libs.Gpu.Rendering;

namespace SharpEmu.Gpu.Metal.V2;

internal sealed partial class MetalRenderHost
{
    internal bool IcbEnabled { get; set; } = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_ICB") != "0";
    private enum IcbRefusal { UnsupportedDevice, Mesh, Rectangle, MultiFace, IndirectArguments, DiagnosticRepeat, CapacityOrAllocation, ArgumentPatchUnavailable }
    private readonly long[] _icbRefusals = new long[Enum.GetValues<IcbRefusal>().Length];
    private readonly MetalRecipeCommands _icbRun = new();
    private MetalIndirectCommands? _icbCache;
    private bool? _icbSupported;
    private bool _icbEncodingDraw;
    private IcbParent _icbParent;
    private long _icbPass = -1, _icbFrame, _icbPasses, _icbDraws, _icbBuilt, _icbReused, _icbRuns, _icbSplits, _icbInherited;
    private long _icbBuildTicks, _icbReplayTicks, _icbSavedTicks;
    private long _totalIcbDraws, _totalIcbBuilt, _totalIcbReused, _totalIcbRuns;
    internal (long Draws, long Built, long Reused, long Runs) IcbCounters =>
        (_totalIcbDraws, _totalIcbBuilt, _totalIcbReused, _totalIcbRuns);

    // Everything that remains parent encoder state under Metal 3.2. In particular,
    // equal DynamicDrawState alone does not prove equal static stencil operations,
    // culling, viewport-index behavior or point/triangle depth bias.
    private readonly record struct IcbParent(DynamicDrawState Dynamic, nint DepthStencil,
        nint Cull, nint Winding, nint Clip, nint Fill, bool Culled, bool Viewports, bool Triangles);

    private void IcbBeforeApply(MetalPipelineEntry entry, nint depthStencil)
    {
        _icbEncodingDraw = false;
        IcbRefusal? refusal = null;
        if (IcbEnabled)
        {
            _icbSupported ??= OperatingSystem.IsMacOSVersionAtLeast(13) &&
                MetalV2Native.SendValue(_device, MetalNative.Selector("argumentBuffersSupport")) >= 1 && // Tier 2 GPU-mutability
                (MetalNative.SendBool(_device, MetalNative.Selector("supportsFamily:"), 1003) ||
                 MetalNative.SendBool(_device, MetalNative.Selector("supportsFamily:"), 2002));
            if (!_icbSupported.Value) refusal = IcbRefusal.UnsupportedDevice;
            else if (_icbPatchUnavailable) refusal = IcbRefusal.ArgumentPatchUnavailable;
            else if (_replayOperation is not (MetalRecipeOperation.Draw or MetalRecipeOperation.DrawIndexed)) refusal = IcbRefusal.IndirectArguments;
            else if (entry.Mesh is not null) refusal = IcbRefusal.Mesh;
            else if (entry.RectangleStates is not null) refusal = IcbRefusal.Rectangle;
            else if (FacePassCount(entry) != 1) refusal = IcbRefusal.MultiFace;
            else if (_tinyPass && PpRepeat > 1) refusal = IcbRefusal.DiagnosticRepeat;
            else _icbEncodingDraw = true;
        }
        if (!_icbEncodingDraw)
        {
            FlushIcbRun(timedByDraw: true);
            if (refusal is { } reason) _icbRefusals[(int)reason]++;
            return;
        }
        var parent = new IcbParent(_dynamicState, depthStencil, entry.CullMode, entry.Winding,
            entry.DepthClipMode, entry.FillMode, entry.CullsEverything, entry.WritesViewportIndex, RasterizesTriangles(entry));
        if (_icbRun.Commands.Count != 0 && (parent != _icbParent || _icbRun.Commands.Count >= 2048))
        {
            _icbSplits++;
            FlushIcbRun(timedByDraw: true);
        }
        _icbParent = parent;
        // An ICB draw can share the logical pipeline shadow without a native bind.
        // It still keeps the PSO alive in this submission under the ordinary cache.
        UsePipelineState(entry);
    }

    private void QueueIcbDraw() => AppendRecipeCommand(_icbRun, compute: false);

    private void FlushIcbRun(bool timedByDraw = false)
    {
        if (_icbRun.Commands.Count == 0) return;
        var started = Stopwatch.GetTimestamp();
        var commands = CollectionsMarshal.AsSpan(_icbRun.Commands);
        var bindings = CollectionsMarshal.AsSpan(_icbRun.Bindings);
        _icbCache ??= new(_device);
        var cached = _icbCache.Get(_icbRun, _submissionSerial, _renderPassSerial, CompletedSerialHint, out var reused);
        if (cached is not null && !reused) { _icbBuilt++; _totalIcbBuilt++; }
        if (cached is null)
        {
            _icbRefusals[(int)IcbRefusal.CapacityOrAllocation] += commands.Length;
            EncodeIcbRunDirect(commands, bindings);
        }
        else
        {
            const nuint stages = Mtl.RenderStageVertex | Mtl.RenderStageFragment;
            // ICB command arguments are indirect references, even when their ordinary
            // direct-encoder counterparts would have made them resident implicitly.
            foreach (var binding in bindings) NameIcbBuffer(binding.Buffer, stages);
            foreach (var command in commands)
                if (command.Operation == MetalRecipeOperation.DrawIndexed) NameIcbBuffer(command.IndexBuffer, stages);
            NameIcbBuffer(cached.Buffer, stages);
            if (!PatchIcbArguments(cached, bindings))
            {
                _icbRefusals[(int)IcbRefusal.ArgumentPatchUnavailable] += commands.Length;
                EncodeIcbRunDirect(commands, bindings);
            }
            else
            {
                var first = commands[0];
                if (cached.InheritBuffers)
                {
                    MetalIndirectCommands.Bind(_renderEncoder, bindings.Slice(first.FirstBinding, first.BindingCount));
                    _icbInherited++;
                }
                MetalV2Native.SendExecuteIndirectCommands(_renderEncoder,
                    MetalNative.Selector("executeCommandsInBuffer:withRange:"), cached.Buffer,
                    new NsRange { Location = 0, Length = (nuint)commands.Length });
                if (_icbPass != _renderPassSerial) { _icbPass = _renderPassSerial; _icbPasses++; }
                _icbDraws += commands.Length; _totalIcbDraws += commands.Length;
                _icbRuns++; _totalIcbRuns++;
                if (reused) { _icbReused++; _totalIcbReused++; _icbSavedTicks += cached.EncodeTicks; }
            }
        }
        var ticks = Stopwatch.GetTimestamp() - started;
        if (reused) _icbReplayTicks += ticks; else _icbBuildTicks += ticks;
        // A flush inside ApplyGraphicsState is already inside CountDraw's timer.
        // Pass-end/barrier/flip flushes need their native CPU cost charged here.
        if (!timedByDraw) Interlocked.Add(ref _drawTicks, ticks);
        _icbRun.Clear();
        // ICB execution is not a direct bind. Force the next ordinary draw to set its
        // real pipeline/vertex slots; retain the valid dynamic/residency shadows.
        _boundGraphicsPipeline = null;
        _boundRenderState = 0;
        Array.Fill(_boundVertexBuffers, (nint)(-1));
    }

    private void EncodeIcbRunDirect(ReadOnlySpan<MetalRecipeCommand> commands, ReadOnlySpan<MetalRecipeBuffer> bindings)
    {
        // Exact current records, in order, with the same parent state. Allocation
        // pressure never drops a draw or splits a render pass.
        foreach (var command in commands)
        {
            MetalNative.SendVoid(_renderEncoder, SelSetRenderPipelineState, command.Pipeline);
            MetalIndirectCommands.Bind(_renderEncoder, bindings.Slice(command.FirstBinding, command.BindingCount));
            MetalIndirectCommands.EncodeDraw(_renderEncoder, command);
        }
    }

    private void NameIcbBuffer(nint buffer, nuint stages)
    {
        if (buffer != 0 && TryNoteResident(buffer, Mtl.ResourceUsageRead, stages))
            MetalNative.SendUseResource(_renderEncoder, SelUseResourceStages, buffer, Mtl.ResourceUsageRead, stages);
    }

    private void ReportIcbFrame()
    {
        FlushIcbRun();
        if (!IcbEnabled) return;
        var scale = 1000.0 / Stopwatch.Frequency;
        Console.Error.WriteLine(FormattableString.Invariant($"[METAL-V2][ICB] scope=frame frame={++_icbFrame} passes={_icbPasses} draws={_icbDraws} runs={_icbRuns} built={_icbBuilt} reused={_icbReused} inherited_bindings={_icbInherited} state_splits={_icbSplits} patch_records={_icbPatchRecords} patch_bytes={_icbPatchBytes} build_ms={_icbBuildTicks * scale:F3} replay_ms={_icbReplayTicks * scale:F3} encode_saved_estimate_ms={(_icbSavedTicks - _icbReplayTicks - _icbBuildTicks) * scale:F3} refusals=[{string.Join(',', _icbRefusals.Select((count, i) => $"{(IcbRefusal)i}:{count}"))}]"));
        _icbPass = -1;
        _icbPasses = _icbDraws = _icbRuns = _icbBuilt = _icbReused = _icbInherited = _icbSplits = 0;
        _icbBuildTicks = _icbReplayTicks = _icbSavedTicks = _icbPatchRecords = _icbPatchBytes = 0;
        Array.Clear(_icbRefusals);
    }
}
