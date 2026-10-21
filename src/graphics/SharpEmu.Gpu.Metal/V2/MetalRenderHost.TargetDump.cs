// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Gpu.Metal.V2;

// SHARPEMU_METAL_V2_DUMP_TARGETS=<dir>: writes the colour target of every render pass,
// and the texture every flip presents, as a PNG. A title that presents black frames has
// either drawn nothing, drawn somewhere else, or drawn something the flip never reaches,
// and those look identical from outside: the only way to tell is to look at the pixels
// at each stage.
//
// It reads back through a blit into a shared buffer and waits for the GPU, so it is as
// intrusive as the shader profiler and equally opt-in.
internal sealed partial class MetalRenderHost
{
    private static readonly string? _dumpTargetsDirectory =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DUMP_TARGETS") is { Length: > 0 } value
            ? value
            : null;

    // Enough to see a title screen appear and settle without filling a disk.
    // SHARPEMU_METAL_V2_DUMP_MAX raises it for a title whose interesting frames
    // come minutes in, where the default budget is spent before the menu ends.
    private static readonly int _maxTargetDumps =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DUMP_MAX"), out var max) && max > 0
            ? max
            : 240;

    // SHARPEMU_METAL_V2_DUMP_AFTER_SECONDS=<n> holds every dump back until the
    // title has been running that long, so the budget lands on the level rather
    // than on the loading screen in front of it.
    private static readonly double _dumpAfterSeconds =
        double.TryParse(
            Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DUMP_AFTER_SECONDS"),
            System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture,
            out var after) && after > 0
            ? after
            : 0;

    private static readonly long _dumpStartTimestamp = System.Diagnostics.Stopwatch.GetTimestamp();

    // SHARPEMU_METAL_V2_DUMP_AFTER_FLIPS=<n> does the same by flip count, for a
    // title whose phases are easier to count than to time.
    private static readonly long _dumpAfterFlips =
        long.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DUMP_AFTER_FLIPS"), out var afterFlips) && afterFlips > 0
            ? afterFlips
            : 0;

    private static bool DumpWindowOpen =>
        (_dumpAfterSeconds <= 0 ||
            (System.Diagnostics.Stopwatch.GetTimestamp() - _dumpStartTimestamp) /
            (double)System.Diagnostics.Stopwatch.Frequency >= _dumpAfterSeconds);

    // One frame's worth of targets every so many flips. Dumping the first N occurrences
    // of each texture instead spends the whole budget on the first seconds, which for a
    // title that loads for minutes is exactly the part that has nothing in it.
    private static readonly int _dumpEveryFlips =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DUMP_EVERY"), out var every) && every > 0
            ? every
            : 30;

    private long _dumpFlipIndex;
    private bool _dumpThisFlip;

    private int _targetDumps;
    private readonly Dictionary<nint, int> _dumpsByTexture = new();
    private bool _dumpDirectoryReady;
    private readonly List<(nint Texture, string Label)> _pendingDumps = new();
    private long _passDrawCount;

    private string? _passLastProfile;

    /// <summary>
    /// Draws recorded into the pass now open, and the last pipeline to draw into it, so a
    /// dump can say whether the target it shows was drawn into or only cleared, and name
    /// the shader to read when it was drawn into and still came out wrong.
    /// </summary>
    private void NotePassDraw(string? profile)
    {
        _passDrawCount++;
        if (profile is not null)
        {
            _passLastProfile = profile;
        }
    }

    // A target that never changes although draws reached it was culled, clipped,
    // depth-rejected, masked out or drawn somewhere else, and all five look the same in
    // the target itself. SHARPEMU_METAL_V2_DUMP_DRAWS=<n> writes the state the first n
    // draws of a dumped frame rasterise with, which tells the five apart.
    private static readonly int _maxDrawStateLines =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DUMP_DRAWS"), out var draws) && draws > 0
            ? draws
            : 0;

    private int _drawStateLines;

    // SHARPEMU_METAL_V2_DUMP_DRAWS_MATCH=<text>: report only draws whose pipeline profile
    // ("vs=0x... ps=0x...") contains the text, so the budget reaches a pass at the end of
    // the frame instead of being spent on the shadow maps at its start.
    private static readonly string? _drawStateMatch =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DUMP_DRAWS_MATCH") is { Length: > 0 } match
            ? match
            : null;

    // One sampled input per draw fills the budget long before the passes that matter
    // end; SHARPEMU_METAL_V2_DUMP_INPUTS=0 spends it all on the targets instead.
    private static readonly bool _dumpInputs =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DUMP_INPUTS"), "0", StringComparison.Ordinal);

    private static readonly nint SelBufferContents = MetalNative.Selector("contents");

    /// <summary>The first bytes of a Metal buffer, as hex, for a draw-state report.</summary>
    private static unsafe string PreviewBuffer(nint buffer, ulong offset, int count)
    {
        if (buffer == 0)
        {
            return "none";
        }

        var contents = MetalNative.Send(buffer, SelBufferContents);
        if (contents == 0)
        {
            return "private";
        }

        return Convert.ToHexString(new ReadOnlySpan<byte>((byte*)contents + offset, count));
    }

    /// <summary>Reports what the draw just recorded rasterises with.</summary>
    private void NoteDrawState(string kind, WorkShape shape)
    {
        if (!DumpTargetsEnabled || !_dumpThisFlip || _drawStateLines >= _maxDrawStateLines)
        {
            return;
        }

        if (_drawStateMatch is not null &&
            !(_boundGraphicsPipeline?.Profile?.Contains(_drawStateMatch, StringComparison.OrdinalIgnoreCase) ?? false))
        {
            return;
        }

        _drawStateLines++;
        var entry = _boundGraphicsPipeline;
        ref readonly var state = ref _dynamicState;
        ref readonly var depth = ref _renderingState.DepthStencilAttachment;
        // The write mask Metal was given, which the executor derives from the pixel
        // program's export mask: a pixel stage whose exports went unrecognised produces a
        // pipeline that writes no colour at all while still writing depth.
        var masks = new System.Text.StringBuilder();
        if (entry?.StaticParameters is { } statics)
        {
            for (var index = 0; index < statics.ColorCount; index++)
            {
                masks.Append(index == 0 ? string.Empty : ",").Append(statics.GetColorMask(index).ToString("X", CultureInfo.InvariantCulture));
            }
        }

        // The first bytes of the streams the draw fetches from. A draw that rasterises
        // nothing at all with every test switched off read either nothing or rubbish.
        var streams = new System.Text.StringBuilder();
        for (var index = 0; index < _vertexBufferCount; index++)
        {
            streams.Append(index == 0 ? "vb" : " vb").Append(index).Append('@')
                .Append(_vertexOffsets[index].ToString("X", CultureInfo.InvariantCulture)).Append('=')
                .Append(PreviewBuffer(_vertexBuffers[index], _vertexOffsets[index], 24));
        }

        streams.Append(" ib@").Append(_indexOffset.ToString("X", CultureInfo.InvariantCulture)).Append('=')
            .Append(PreviewBuffer(_indexBuffer, _indexOffset, 12));

        var cull = entry is null ? "?" :
            entry.CullsEverything ? "both" :
            entry.CullMode == Mtl.CullModeFront ? "front" :
            entry.CullMode == Mtl.CullModeBack ? "back" : "none";
        Console.Error.WriteLine(string.Format(
            CultureInfo.InvariantCulture,
            "[METAL-V2][DRAW] #{0} draw={1} {25} {26} prim={27} streams={28} targets={2} " +
            "vp=({3},{4} {5}x{6} z{7}..{8}) " +
            "scissor=({9},{10})-({11},{12}) cull={13} winding={14} clip={15} " +
            "depth={16} test={17} write={18} cmp={19} clear={20} stencil={21} " +
            "colormask={22:X} colorcount={23} hostmasks=[{29}] ps={24}",
            _drawStateLines, _passDrawCount, _renderingState.ColorAttachmentCount,
            state.ViewportX, state.ViewportY, state.ViewportWidth, state.ViewportHeight,
            state.ViewportMinDepth, state.ViewportMaxDepth,
            state.Scissor.Left, state.Scissor.Top, state.Scissor.Right, state.Scissor.Bottom,
            cull, entry?.Winding == Mtl.WindingClockwise ? "cw" : "ccw", entry?.DepthClipMode,
            depth.HasDepth ? "yes" : "no", state.DepthTestEnabled, state.DepthWriteEnabled,
            state.DepthCompare,
            depth.DepthClear ? BitConverter.UInt32BitsToSingle(depth.ClearWord0) : float.NaN,
            state.StencilTestEnabled, state.ColorWriteEnableMask, state.ColorWriteCount,
            entry?.Profile ?? "-",
            kind, shape.ToString(), entry?.PrimitiveType, _vertexBufferCount, masks));
        Console.Error.WriteLine($"[METAL-V2][DRAW]   #{_drawStateLines} {streams}");
        // What the pixel stage samples, by guest address: a composite that shows stale
        // or unrelated pictures read the wrong surface, which only the addresses tell.
        var fragment = _stageBindings[StageFragment];
        var images = new System.Text.StringBuilder();
        for (var index = 0; index < fragment.ImageCount && index < fragment.Textures.Length; index++)
        {
            var texture = fragment.Textures[index];
            images.Append(" t").Append(index).Append('=');
            if (_imageCache.TryGetImage(fragment.Images[index], out var image))
            {
                ref readonly var d = ref image.Description;
                images.Append(CultureInfo.InvariantCulture, $"{d.PixelFormat}/{d.TileMode}/{d.Extent.Width}x{d.Extent.Height}@{d.Data.Address:X}+{d.Data.Size:X}");
            }
            else
            {
                images.Append("uncached");
            }

            if (texture != 0)
            {
                images.Append(CultureInfo.InvariantCulture,
                    $"[host {MetalV2Native.SendValue(texture, SelWidth)}x{MetalV2Native.SendValue(texture, SelHeight)} 0x{texture:X}]");
            }
        }

        Console.Error.WriteLine($"[METAL-V2][DRAW]   #{_drawStateLines} ps_images={fragment.ImageCount}{images}");
        if (entry?.VertexLayout is { Length: > 0 } layout)
        {
            Console.Error.WriteLine($"[METAL-V2][DRAW]   #{_drawStateLines} layout{layout}");
        }
    }

    /// <summary>
    /// Remembers a texture to dump at the next safe point. EndRendering runs inside the
    /// flush, and a dump has to flush, so nothing may be read back from there.
    /// </summary>
    private void NotePendingDump(nint texture, string label)
    {
        if (texture == 0 || !_dumpThisFlip || _pendingDumps.Count >= 512)
        {
            return;
        }

        var shaders = _passLastProfile is null
            ? string.Empty
            : "-" + _passLastProfile.Replace("stage=", string.Empty).Replace(" ", "-").Replace("=0x", "");
        var full = $"{label}-draws{_passDrawCount}{shaders}";
        if (TrySnapshotTexture(texture, full))
        {
            return;
        }

        _pendingDumps.Add((texture, full));
    }

    // SHARPEMU_METAL_V2_DUMP_SNAPSHOT=1: copy a target the moment the pass (or compute
    // encoder) that wrote it ends, rather than reading it back at the flip. Without it
    // every dump of a texture shows the frame's last state of that texture, so a target
    // written by five passes reads the same five times and the pass that broke it is
    // invisible. Each copy is held in its own readback buffer until the flip drains
    // them, so a sampled frame costs its targets' size in memory: keep DUMP_EVERY high.
    private static readonly bool _dumpSnapshots = string.Equals(
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DUMP_SNAPSHOT"), "1", StringComparison.Ordinal);

    private const long MaxSnapshotBytes = 3L << 30;

    // SHARPEMU_METAL_V2_DUMP_SNAPSHOT_MATCH=<a,b,...>: snapshot only targets whose label
    // contains one of the texts ("-cs" for compute, a format, an address), so a frame's
    // budget reaches the pass that matters instead of every geometry pass before it.
    private static readonly string[]? _snapshotMatch =
        Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DUMP_SNAPSHOT_MATCH") is { Length: > 0 } snapshotMatch
            ? snapshotMatch.Split(',', StringSplitOptions.RemoveEmptyEntries)
            : null;

    private readonly List<(nint Buffer, nint Contents, uint Width, uint Height, ulong Format, nint Texture, string Label)>
        _snapshots = new();

    private readonly List<(nint Texture, string Label)> _computeStorageNoted = new();
    private long _snapshotBytes;
    private int _snapshotSerial;

    /// <summary>
    /// Copies <paramref name="texture"/> into a readback buffer on the command buffer
    /// being recorded. Only between encoders: a blit cannot open while another encoder
    /// is recording. Returns false when the caller should fall back to a flip-time dump.
    /// </summary>
    private readonly Dictionary<nint, string> _targetDescriptions = new();

    /// <summary>
    /// Under snapshots, remembers which guest surface an attachment view is, so a
    /// snapshot can name it: two passes whose targets are different textures over the
    /// same guest bytes are an alias the cache has to keep coherent.
    /// </summary>
    private void NoteTargetDescription(
        nint texture, in ImageDescription description, uint layer, uint level, string extra = "")
    {
        if (!_dumpSnapshots || !_dumpThisFlip || texture == 0)
        {
            return;
        }

        _targetDescriptions[texture] = string.Create(CultureInfo.InvariantCulture,
            $"{description.PixelFormat}-{description.TileMode}-{description.Extent.Width}x{description.Extent.Height}" +
            $"-L{level}A{layer}-at{description.Data.Address:X}+{description.Data.Size:X}{extra}");
    }

    private bool TrySnapshotTexture(nint texture, string label)
    {
        if (_dumpSnapshots && _targetDescriptions.TryGetValue(texture, out var described))
        {
            label = $"{label}-{described}";
        }

        if (!_dumpSnapshots || _renderEncoder != 0 || _computeEncoder != 0)
        {
            return false;
        }

        if (_snapshotMatch is not null && !Array.Exists(_snapshotMatch, token => label.Contains(token, StringComparison.Ordinal)))
        {
            return true;
        }

        var width = (uint)MetalV2Native.SendValue(texture, SelWidth);
        var height = (uint)MetalV2Native.SendValue(texture, SelHeight);
        var pixelFormat = MetalV2Native.SendValue(texture, SelPixelFormat);
        if (width == 0 || height == 0 ||
            _targetDumps + _snapshots.Count >= _maxTargetDumps ||
            !TryDescribeDumpFormat(pixelFormat, out var sourceBytes, out _))
        {
            return true;
        }

        var rowBytes = (ulong)sourceBytes * width;
        var total = rowBytes * height;
        if (_snapshotBytes + (long)total > MaxSnapshotBytes)
        {
            MetalV2Diagnostics.WarnOnce(
                "dump-snapshot-budget",
                "Target snapshots stopped for this frame: the readback budget is spent.");
            return true;
        }

        var readback = _guestMemory.CreateReportBuffer(total, "SharpEmu TargetSnapshot", out var contents, out _);
        var blit = MetalNative.Send(EnsureCommandBuffer(), SelBlitEncoder);
        MetalMemoryTelemetry.BlitEncoders.Add();
        if (blit == 0)
        {
            MetalNative.SendVoid(readback, MetalNative.Selector("release"));
            return true;
        }

        MetalNative.SendCopyTextureToBufferOptions(
            blit, SelDumpCopyTextureToBuffer, texture, 0, 0, default,
            new MtlSize { Width = width, Height = height, Depth = 1 },
            readback, 0, (nuint)rowBytes, (nuint)total,
            (nuint)(pixelFormat == MtlFormat.Depth32FloatStencil8
                ? MtlBlitOption.DepthFromDepthStencil
                : MtlBlitOption.None));
        MetalNative.SendVoid(blit, SelEndEncoding);
        _snapshots.Add((readback, contents, width, height, pixelFormat, texture,
            $"s{_snapshotSerial++:D4}-{label}"));
        _snapshotBytes += (long)total;
        return true;
    }

    private const string UnnamedComputeLabel = "cs?";

    /// <summary>
    /// A dispatch was just recorded: names the program behind the storage images its
    /// bindings noted, and under snapshots says which program ran with which grid, so a
    /// compute pass that wrote nothing can be told from one that never ran.
    /// </summary>
    private void NoteDispatchState(MetalPipelineEntry entry, string kind, WorkShape shape)
    {
        if (!_dumpThisFlip || !_dumpSnapshots)
        {
            return;
        }

        var name = $"cs{entry.VertexHash:X16}";
        for (var index = 0; index < _computeStorageNoted.Count; index++)
        {
            var (texture, label) = _computeStorageNoted[index];
            if (label.StartsWith(UnnamedComputeLabel, StringComparison.Ordinal))
            {
                _computeStorageNoted[index] = (texture, name + label[UnnamedComputeLabel.Length..]);
            }
        }

        Console.Error.WriteLine($"[METAL-V2][DUMP] {kind} {name} {shape} seq={_snapshotSerial}");
    }

    /// <summary>
    /// The compute encoder just ended: copy every storage image its dispatches wrote,
    /// labelled with the last pipeline and grid, so the per-encoder state is on record.
    /// </summary>
    private void SnapshotComputeStorage()
    {
        if (_computeStorageNoted.Count == 0)
        {
            return;
        }

        var noted = _computeStorageNoted.ToArray();
        _computeStorageNoted.Clear();
        var seen = new HashSet<nint>();
        foreach (var (texture, label) in noted)
        {
            if (seen.Add(texture))
            {
                TrySnapshotTexture(texture, label);
            }
        }
    }

    /// <summary>Writes the snapshots the GPU has finished copying.</summary>
    private void DrainSnapshots()
    {
        if (_snapshots.Count == 0)
        {
            return;
        }

        EndEncoders(PassEndCause.Other);
        FlushAndWaitForGpu();
        var snapshots = _snapshots.ToArray();
        _snapshots.Clear();
        _snapshotBytes = 0;
        foreach (var (buffer, contents, width, height, format, texture, label) in snapshots)
        {
            try
            {
                if (TryDescribeDumpFormat(format, out var sourceBytes, out var convert))
                {
                    var source = DumpSpan(contents, (int)((ulong)sourceBytes * width * height));
                    var rgba = new byte[(int)(4UL * width * height)];
                    convert(source, rgba);
                    _dumpsByTexture.TryGetValue(texture, out var seen);
                    _dumpsByTexture[texture] = seen + 1;
                    _targetDumps++;
                    WriteDumpPng(rgba, width, height, label, seen, texture);
                }
            }
            catch (Exception exception)
            {
                MetalV2Diagnostics.WarnOnce("dump-snapshot-failed", $"A target snapshot could not be written: {exception.Message}");
            }
            finally
            {
                MetalNative.SendVoid(buffer, MetalNative.Selector("release"));
            }
        }
    }

    /// <summary>
    /// A texture a stage is about to sample. The target a pass writes only says where the
    /// wrongness ends up; the inputs say where it came from, and the label carries the
    /// guest description — format, tile mode, extent, bytes — because a surface read with
    /// the wrong element size or swizzle is wrong in exactly those fields.
    /// </summary>
    private void NotePendingInputDump(
        nint texture, SharpEmu.Libs.Gpu.Buffers.ResourceSlotIdentifier identifier, int index, ImageRole role)
    {
        if (!DumpTargetsEnabled || !_dumpThisFlip || !_dumpInputs || texture == 0)
        {
            return;
        }

        if (!_imageCache.TryGetImage(identifier, out var image))
        {
            if (_dumpSnapshots)
            {
                Console.Error.WriteLine(
                    $"[METAL-V2][DUMP] input in{index}-uncached-{identifier} tex=0x{texture:X} " +
                    $"{MetalV2Native.SendValue(texture, SelWidth)}x{MetalV2Native.SendValue(texture, SelHeight)} " +
                    $"fmt={MetalV2Native.SendValue(texture, SelPixelFormat)} seq={_snapshotSerial}");
            }

            return;
        }

        ref readonly var description = ref image.Description;
        var label =
            $"in{index}-{(role == ImageRole.StorageImage ? "storage" : "sampled")}-{description.PixelFormat}" +
            $"-{description.Type}-{description.TileMode}" +
            $"-{description.Extent.Width}x{description.Extent.Height}x{description.Extent.Depth}" +
            $"-L{description.Resources.Levels}A{description.Resources.Layers}" +
            $"-bpb{description.BytesPerBlock}-pitch{description.Pitch}-bytes{description.Data.Size:X}" +
            $"-at{description.Data.Address:X}";
        // A compute encoder's inputs are copied when it ends, like its storage images:
        // nothing inside it writes them, so that copy is what its dispatches read.
        if (_dumpSnapshots && _computeEncoder != 0)
        {
            // The program is named once the dispatch is recorded (NoteDispatchState):
            // the bindings are applied before the pipeline is known here.
            _computeStorageNoted.Add((texture, $"{UnnamedComputeLabel}-{label}"));
            return;
        }

        if (_dumpSnapshots)
        {
            // Inputs are described in the log only; their pixels at the flip would be
            // the misleading end-of-frame state this mode exists to avoid.
            Console.Error.WriteLine($"[METAL-V2][DUMP] input {label} tex=0x{texture:X} seq={_snapshotSerial}");
            return;
        }

        NotePendingDump(texture, label);
    }

    /// <summary>
    /// Reads back everything remembered since the last drain, and decides whether the
    /// frame about to be recorded is one of the sampled ones.
    /// </summary>
    private void DrainPendingDumps()
    {
        if (!DumpTargetsEnabled)
        {
            return;
        }

        DrainSnapshots();
        var flip = _dumpFlipIndex++;
        _dumpThisFlip = flip >= _dumpAfterFlips && DumpWindowOpen &&
            (flip - _dumpAfterFlips) % _dumpEveryFlips == 0;
        if (_pendingDumps.Count == 0)
        {
            return;
        }

        var pending = _pendingDumps.ToArray();
        _pendingDumps.Clear();
        foreach (var (texture, label) in pending)
        {
            DumpTexture(texture, label);
        }
    }

    private static bool DumpTargetsEnabled => _dumpTargetsDirectory is not null;

    private static readonly nint SelDumpCopyTextureToBuffer =
        MetalNative.Selector("copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:");

    private static readonly nint SelPixelFormat = MetalNative.Selector("pixelFormat");
    private static readonly nint SelWidth = MetalNative.Selector("width");
    private static readonly nint SelHeight = MetalNative.Selector("height");

    /// <summary>Dumps a texture, taking its shape from the texture itself.</summary>
    private void DumpTexture(nint texture, string label)
    {
        if (!DumpTargetsEnabled || texture == 0)
        {
            return;
        }

        DumpTexture(
            texture,
            (uint)MetalV2Native.SendValue(texture, SelWidth),
            (uint)MetalV2Native.SendValue(texture, SelHeight),
            MetalV2Native.SendValue(texture, SelPixelFormat),
            label);
    }

    /// <summary>
    /// Reads a colour texture back and writes it as a PNG. Flushes and waits, so it only
    /// ever runs under the dump env var.
    /// </summary>
    private void DumpTexture(nint texture, uint width, uint height, ulong pixelFormat, string label)
    {
        if (!DumpTargetsEnabled || texture == 0 || width == 0 || height == 0 || _targetDumps >= _maxTargetDumps)
        {
            return;
        }

        _dumpsByTexture.TryGetValue(texture, out var seen);
        if (!TryDescribeDumpFormat(pixelFormat, out var sourceBytes, out var convert))
        {
            MetalV2Diagnostics.WarnOnce(
                $"dump-format-{pixelFormat}",
                $"A render target cannot be dumped because its format is not one the dumper reads: " +
                $"format={pixelFormat} {width}x{height}.");
            return;
        }

        try
        {
            var rowBytes = (ulong)sourceBytes * width;
            var total = rowBytes * height;
            var readback = _guestMemory.CreateReportBuffer(total, "SharpEmu TargetDump", out var contents, out _);
            try
            {
                EndEncoders(PassEndCause.Other);
                var blit = MetalNative.Send(EnsureCommandBuffer(), SelBlitEncoder);
                MetalMemoryTelemetry.BlitEncoders.Add();
                if (blit == 0)
                {
                    return;
                }

                MetalNative.SendCopyTextureToBufferOptions(
                    blit, SelDumpCopyTextureToBuffer, texture, 0, 0, default,
                    new MtlSize { Width = width, Height = height, Depth = 1 },
                    readback, 0, (nuint)rowBytes, (nuint)total,
                    (nuint)(pixelFormat == MtlFormat.Depth32FloatStencil8
                        ? MtlBlitOption.DepthFromDepthStencil
                        : MtlBlitOption.None));
                MetalNative.SendVoid(blit, SelEndEncoding);
                FlushAndWaitForGpu();

                var source = DumpSpan(contents, (int)total);
                var rgba = new byte[(int)(4UL * width * height)];
                convert(source, rgba);
                if (label.StartsWith("flip", StringComparison.Ordinal))
                {
                    // The presenter's layer is opaque, so the screen never shows a flip's alpha;
                    // a viewer that composites the guest's (often zero) alpha would show a
                    // correct frame as white or black.
                    for (var alpha = 3; alpha < rgba.Length; alpha += 4)
                    {
                        rgba[alpha] = 0xFF;
                    }
                }
                _dumpsByTexture[texture] = seen + 1;
                _targetDumps++;
                WriteDumpPng(rgba, width, height, label, seen, texture);
            }
            finally
            {
                MetalNative.SendVoid(readback, MetalNative.Selector("release"));
            }
        }
        catch (Exception exception)
        {
            MetalV2Diagnostics.WarnOnce(
                "dump-failed",
                $"A render target could not be dumped: {exception.Message}");
        }
    }

    private static unsafe ReadOnlySpan<byte> DumpSpan(nint contents, int length) =>
        new((void*)contents, length);

    private void WriteDumpPng(byte[] rgba, uint width, uint height, string label, int index, nint texture)
    {
        if (!_dumpDirectoryReady)
        {
            Directory.CreateDirectory(_dumpTargetsDirectory!);
            _dumpDirectoryReady = true;
        }

        var safe = new string(label.Select(c => char.IsLetterOrDigit(c) || c is '-' or '_' ? c : '-').ToArray());
        var name = string.Format(
            CultureInfo.InvariantCulture, "{0:D4}-{1}-{2}x{3}-{4}.png", _targetDumps, safe, width, height, index);
        var path = Path.Combine(_dumpTargetsDirectory!, name);
        File.WriteAllBytes(path, EncodePng(rgba, width, height));

        // A target that was only cleared and a target that was drawn into with zeros look
        // the same in a mean, so report the extremes and the first pixel that is not
        // black as well: "everything is exactly zero" and "everything is very dark" are
        // different bugs.
        ulong sum = 0;
        var opaque = 0;
        byte max = 0;
        var nonBlack = -1;
        for (var i = 0; i < rgba.Length; i += 4)
        {
            var r = rgba[i];
            var g = rgba[i + 1];
            var b = rgba[i + 2];
            sum += (ulong)(r + g + b);
            max = Math.Max(max, Math.Max(r, Math.Max(g, b)));
            if (rgba[i + 3] != 0)
            {
                opaque++;
            }

            if (nonBlack < 0 && (r | g | b) != 0)
            {
                nonBlack = i;
            }
        }

        var pixels = rgba.Length / 4;
        var sample = nonBlack < 0
            ? "none"
            : string.Format(
                CultureInfo.InvariantCulture, "#{0}=({1},{2},{3},{4})", nonBlack / 4,
                rgba[nonBlack], rgba[nonBlack + 1], rgba[nonBlack + 2], rgba[nonBlack + 3]);
        Console.Error.WriteLine(string.Format(
            CultureInfo.InvariantCulture,
            "[METAL-V2][DUMP] {0} tex=0x{1:X} {2}x{3} mean_rgb={4:F4} max_rgb={5} opaque={6}/{7} " +
            "first_lit={8} -> {9}",
            label, texture, width, height, sum / (double)(pixels * 3), max, opaque, pixels, sample, name));
    }

    internal static bool TryDescribeDumpFormat(
        ulong pixelFormat, out int sourceBytes, out Action<ReadOnlySpan<byte>, byte[]> convert)
    {
        switch (pixelFormat)
        {
            case MtlFormat.Rgba8Unorm:
            case MtlFormat.Rgba8UnormSrgb:
                sourceBytes = 4;
                convert = static (source, target) => source.CopyTo(target);
                return true;
            case MtlFormat.Bgra8Unorm:
            case MtlFormat.Bgra8UnormSrgb:
                sourceBytes = 4;
                convert = static (source, target) =>
                {
                    for (var i = 0; i + 3 < source.Length; i += 4)
                    {
                        target[i] = source[i + 2];
                        target[i + 1] = source[i + 1];
                        target[i + 2] = source[i];
                        target[i + 3] = source[i + 3];
                    }
                };
                return true;
            case MtlFormat.Rgb10A2Unorm:
            case MtlFormat.Bgr10A2Unorm:
                sourceBytes = 4;
                var blueFirst = pixelFormat == MtlFormat.Bgr10A2Unorm;
                convert = (source, target) =>
                {
                    for (var i = 0; i + 3 < source.Length; i += 4)
                    {
                        var packed = BinaryPrimitives.ReadUInt32LittleEndian(source[i..]);
                        var c0 = (byte)(((packed >> 0) & 0x3FF) >> 2);
                        var c1 = (byte)(((packed >> 10) & 0x3FF) >> 2);
                        var c2 = (byte)(((packed >> 20) & 0x3FF) >> 2);
                        target[i] = blueFirst ? c2 : c0;
                        target[i + 1] = c1;
                        target[i + 2] = blueFirst ? c0 : c2;
                        target[i + 3] = (byte)(((packed >> 30) & 3) * 85);
                    }
                };
                return true;
            case MtlFormat.Rgba16Float:
                sourceBytes = 8;
                convert = static (source, target) =>
                {
                    for (int i = 0, o = 0; i + 7 < source.Length; i += 8, o += 4)
                    {
                        for (var c = 0; c < 4; c++)
                        {
                            var half = (Half)BitConverter.UInt16BitsToHalf(
                                BinaryPrimitives.ReadUInt16LittleEndian(source[(i + (c * 2))..]));
                            var v = (float)half;
                            target[o + c] = (byte)Math.Clamp(v * 255f, 0f, 255f);
                        }
                    }
                };
                return true;
            // Depth, so a pass whose colour never changes can be told apart from a pass
            // whose depth test rejected everything. The stencil half of a combined format
            // stays behind: the blit asks for the depth plane alone.
            case MtlFormat.Depth32Float:
            case MtlFormat.Depth32FloatStencil8:
                sourceBytes = 4;
                convert = static (source, target) =>
                {
                    for (int i = 0, o = 0; i + 3 < source.Length; i += 4, o += 4)
                    {
                        var depth = BitConverter.UInt32BitsToSingle(
                            BinaryPrimitives.ReadUInt32LittleEndian(source[i..]));
                        var level = (byte)Math.Clamp(depth * 255f, 0f, 255f);
                        target[o] = level;
                        target[o + 1] = level;
                        target[o + 2] = level;
                        target[o + 3] = 255;
                    }
                };
                return true;
            // A depth plane sampled as a colour alias reads as the depth it holds.
            case MtlFormat.R32Float:
                sourceBytes = 4;
                convert = static (source, target) =>
                {
                    for (int i = 0, o = 0; i + 3 < source.Length; i += 4, o += 4)
                    {
                        var value = BitConverter.UInt32BitsToSingle(
                            BinaryPrimitives.ReadUInt32LittleEndian(source[i..]));
                        var level = (byte)Math.Clamp(value * 255f, 0f, 255f);
                        target[o] = level;
                        target[o + 1] = level;
                        target[o + 2] = level;
                        target[o + 3] = 255;
                    }
                };
                return true;
            case MtlFormat.R16Float:
                sourceBytes = 2;
                convert = static (source, target) =>
                {
                    for (int i = 0, o = 0; i + 1 < source.Length; i += 2, o += 4)
                    {
                        var value = (float)BitConverter.UInt16BitsToHalf(
                            BinaryPrimitives.ReadUInt16LittleEndian(source[i..]));
                        var level = (byte)Math.Clamp(value * 255f, 0f, 255f);
                        target[o] = level;
                        target[o + 1] = level;
                        target[o + 2] = level;
                        target[o + 3] = 255;
                    }
                };
                return true;
            // A stencil plane read as R8Uint shows its byte as a level, like R8Unorm.
            case MtlFormat.R8Unorm:
            case MtlFormat.R8Uint:
                sourceBytes = 1;
                convert = static (source, target) =>
                {
                    for (int i = 0, o = 0; i < source.Length; i++, o += 4)
                    {
                        target[o] = source[i];
                        target[o + 1] = source[i];
                        target[o + 2] = source[i];
                        target[o + 3] = 255;
                    }
                };
                return true;
            case MtlFormat.Rg11B10Float:
                sourceBytes = 4;
                convert = static (source, target) =>
                {
                    for (int i = 0, o = 0; i + 3 < source.Length; i += 4, o += 4)
                    {
                        var packed = BinaryPrimitives.ReadUInt32LittleEndian(source[i..]);
                        target[o] = SmallFloatLevel(packed & 0x7FF, 6);
                        target[o + 1] = SmallFloatLevel((packed >> 11) & 0x7FF, 6);
                        target[o + 2] = SmallFloatLevel((packed >> 22) & 0x3FF, 5);
                        target[o + 3] = 255;
                    }
                };
                return true;
            case MtlFormat.Depth16Unorm:
                sourceBytes = 2;
                convert = static (source, target) =>
                {
                    for (int i = 0, o = 0; i + 1 < source.Length; i += 2, o += 4)
                    {
                        var level = (byte)(BinaryPrimitives.ReadUInt16LittleEndian(source[i..]) >> 8);
                        target[o] = level;
                        target[o + 1] = level;
                        target[o + 2] = level;
                        target[o + 3] = 255;
                    }
                };
                return true;
            // A packed G-buffer word: its four bytes as RGB plus the top byte as a level,
            // opaque, so "never written" (all zero) stays distinguishable from content.
            case MtlFormat.R32Uint:
                sourceBytes = 4;
                convert = static (source, target) =>
                {
                    for (int i = 0; i + 3 < source.Length; i += 4)
                    {
                        target[i] = source[i];
                        target[i + 1] = source[i + 1];
                        target[i + 2] = source[i + 2];
                        target[i + 3] = 255;
                    }
                };
                return true;
            // Two half floats (velocity, normals): |x| and |y| scaled so motion of a few
            // pixels is visible; blue stays empty.
            case MtlFormat.Rg16Float:
                sourceBytes = 4;
                convert = static (source, target) =>
                {
                    for (int i = 0, o = 0; i + 3 < source.Length; i += 4, o += 4)
                    {
                        for (var c = 0; c < 2; c++)
                        {
                            var v = (float)BitConverter.UInt16BitsToHalf(
                                BinaryPrimitives.ReadUInt16LittleEndian(source[(i + (c * 2))..]));
                            target[o + c] = (byte)Math.Clamp(MathF.Abs(v) * 255f, 0f, 255f);
                        }

                        target[o + 2] = 0;
                        target[o + 3] = 255;
                    }
                };
                return true;
            case MtlFormat.Rgba16Unorm:
                sourceBytes = 8;
                convert = static (source, target) =>
                {
                    for (int i = 0, o = 0; i + 7 < source.Length; i += 8, o += 4)
                    {
                        for (var c = 0; c < 4; c++)
                        {
                            target[o + c] = (byte)(BinaryPrimitives.ReadUInt16LittleEndian(
                                source[(i + (c * 2))..]) >> 8);
                        }
                    }
                };
                return true;
            default:
                sourceBytes = 0;
                convert = static (_, _) => { };
                return false;
        }
    }

    // An unsigned small float (11 or 10 bits: 5-bit exponent, bias 15) as an 8-bit level.
    private static byte SmallFloatLevel(uint bits, int mantissaBits)
    {
        var exponent = (int)(bits >> mantissaBits);
        var mantissa = bits & ((1u << mantissaBits) - 1);
        var value = exponent == 0
            ? mantissa / (float)(1 << mantissaBits) * MathF.Pow(2, -14)
            : (1 + mantissa / (float)(1 << mantissaBits)) * MathF.Pow(2, exponent - 15);
        return (byte)Math.Clamp(value * 255f, 0f, 255f);
    }

    // A minimal PNG: one IDAT of zlib-deflated scanlines, each with filter byte zero.
    internal static byte[] EncodePng(byte[] rgba, uint width, uint height)
    {
        var raw = new byte[(int)((4UL * width + 1) * height)];
        for (var y = 0; y < height; y++)
        {
            var source = (int)(4UL * width * (ulong)y);
            var target = (int)((4UL * width + 1) * (ulong)y);
            raw[target] = 0;
            Array.Copy(rgba, source, raw, target + 1, (int)(4 * width));
        }

        using var deflated = new MemoryStream();
        using (var zlib = new ZLibStream(deflated, CompressionLevel.Fastest, leaveOpen: true))
        {
            zlib.Write(raw, 0, raw.Length);
        }

        using var png = new MemoryStream();
        png.Write([0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        var header = new byte[13];
        BinaryPrimitives.WriteUInt32BigEndian(header, width);
        BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(4), height);
        header[8] = 8;  // bit depth
        header[9] = 6;  // truecolour with alpha
        WriteChunk(png, "IHDR", header);
        WriteChunk(png, "IDAT", deflated.ToArray());
        WriteChunk(png, "IEND", []);
        return png.ToArray();
    }

    private static void WriteChunk(Stream stream, string type, byte[] data)
    {
        Span<byte> length = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(length, (uint)data.Length);
        stream.Write(length);
        var typeBytes = System.Text.Encoding.ASCII.GetBytes(type);
        stream.Write(typeBytes);
        stream.Write(data);
        var crc = Crc32(typeBytes, data);
        Span<byte> crcBytes = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(crcBytes, crc);
        stream.Write(crcBytes);
    }

    private static readonly uint[] _crcTable = BuildCrcTable();

    private static uint[] BuildCrcTable()
    {
        var table = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            var c = n;
            for (var k = 0; k < 8; k++)
            {
                c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            }

            table[n] = c;
        }

        return table;
    }

    private static uint Crc32(byte[] type, byte[] data)
    {
        var c = 0xFFFFFFFFu;
        foreach (var b in type)
        {
            c = _crcTable[(c ^ b) & 0xFF] ^ (c >> 8);
        }

        foreach (var b in data)
        {
            c = _crcTable[(c ^ b) & 0xFF] ^ (c >> 8);
        }

        return c ^ 0xFFFFFFFFu;
    }
}
