// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.HLE.Host;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A draw into the depth alone, between two draws into colour targets over that depth, stays
/// in the open pass: the pixels are those of three passes, drawn in one. Nothing joins while
/// the draw samples a colour attachment of the pass, while a clear waits, or while a command
/// processor copy owes bytes the draw binds. Device work is behind the exclusive device gate.
/// </summary>
public sealed unsafe class MetalDepthOnlyJoinTests(ITestOutputHelper output)
{
    private const ulong GuestBase = 0x2_0000_0000;
    private const ulong TargetAddress = GuestBase + 0x4000;
    private const ulong OtherAddress = GuestBase + 0x8000;
    private const ulong LabelAddress = GuestBase + 0x1000;
    private const uint Width = 64;
    private const uint Height = 8;
    private const ulong ImageBytes = Width * Height * 4;
    private const nint PixelFormatRgba8Unorm = 70;
    private const nuint PixelFormatDepth32Float = 252;
    private const byte GuestPattern = 0x11;

    // Red over everything at depth 0.5; the depth-only draw lowers the left half to 0.25;
    // green over everything at 0.4 passes a LESS test on the right half only.
    private const float RedDepth = 0.5f;
    private const float LoweredDepth = 0.25f;
    private const float GreenDepth = 0.4f;

    private sealed record Outcome(
        byte[] Pixels, long Passes, long Joins, long JoinsSampled, long JoinsWaitingOnCopies, int PendingClears,
        int DeferredCopies);

    private enum Scene
    {
        // Nothing the depth-only draw binds stands in the way.
        Free,
        // The depth-only draw samples the colour attachment of the pass.
        SamplesAttachment,
        // A clear for another texture, asked for while the pass is open, waits when the depth-only draw begins.
        PendingClear,
        // A deferred copy lands in a buffer the depth-only draw binds.
        BindsCopyDestination,
        // A deferred copy waits, in bytes the draw does not bind.
        BindsOtherBytes,
    }

    [Fact]
    public void ADepthOnlyDrawBetweenTwoColourDrawsJoinsTheirPass()
    {
        if (!RequireDevice()) return;
        var joined = Run(Scene.Free);
        Assert.Equal(1L, joined.Joins);
        Assert.Equal(1L, joined.Passes);
        AssertRedLeftGreenRight(joined.Pixels);
    }

    [Fact]
    public void TheJoinedPixelsAreThoseOfThreeSeparatePasses()
    {
        if (!RequireDevice()) return;
        var joined = Run(Scene.Free);
        // Sampling the attachment keeps the depth-only draw out of the pass: three passes.
        var separate = Run(Scene.SamplesAttachment);
        Assert.Equal(0L, separate.Joins);
        Assert.Equal(1L, separate.JoinsSampled);
        Assert.Equal(3L, separate.Passes);
        Assert.Equal(separate.Pixels, joined.Pixels);
        AssertRedLeftGreenRight(separate.Pixels);
    }

    [Fact]
    public void ADepthOnlyDrawDoesNotJoinWhileAClearWaits()
    {
        if (!RequireDevice()) return;
        var outcome = Run(Scene.PendingClear);
        Assert.Equal(1, outcome.PendingClears);
        Assert.Equal(0L, outcome.Joins);
        AssertRedLeftGreenRight(outcome.Pixels);
    }

    [Fact]
    public void ADepthOnlyDrawDoesNotJoinWhileACopyOwesBytesItBinds()
    {
        if (!RequireDevice()) return;
        var outcome = Run(Scene.BindsCopyDestination);
        Assert.Equal(0L, outcome.Joins);
        Assert.Equal(1L, outcome.JoinsWaitingOnCopies);
        Assert.Equal(3L, outcome.Passes);
        // The boundary the draw's own pass began encoded the copy.
        Assert.Equal(0, outcome.DeferredCopies);
        AssertRedLeftGreenRight(outcome.Pixels);
    }

    [Fact]
    public void ADepthOnlyDrawStillJoinsWhileACopyOwesOtherBytes()
    {
        if (!RequireDevice()) return;
        var outcome = Run(Scene.BindsOtherBytes);
        Assert.Equal(1L, outcome.Joins);
        Assert.Equal(0L, outcome.JoinsWaitingOnCopies);
        Assert.Equal(1L, outcome.Passes);
        Assert.Equal(1, outcome.DeferredCopies);
        AssertRedLeftGreenRight(outcome.Pixels);
    }

    [Fact]
    public void TheJoinedDescriptionCopiesThePassFormatsAndWritesNoColour()
    {
        var parameters = new PipelineStaticParameters { ColorCount = 0, Topology = GpuPrimitiveTopology.TriangleList, Samples = 1 };
        var rendering = new PipelineRenderingState
        {
            ColorCount = 0,
            DepthFormat = GpuFormat.D32SfloatS8Uint,
            StencilFormat = GpuFormat.D32SfloatS8Uint,
        };
        var description = new GraphicsPipelineDescription
        {
            Rendering = rendering,
            VertexInput = new PipelineVertexInputState(),
            VertexInfo = new VertexInputInfo(),
            VertexProgram = new ShaderProgram(7, 7),
            VertexStage = new ShaderProgramInfo { Stage = ShaderStageKind.Vertex, Hash = 7 },
            StaticParameters = parameters,
        };
        var open = new RenderingState { Width = Width, Height = Height, Layers = 1, ColorAttachmentCount = 2, Samples = 1 };
        open.ColorAttachments[0] = new RenderingAttachment(
            new GpuImageHandle(1), GpuImageLayout.ColorAttachmentOptimal, GpuFormat.R8G8B8A8Unorm,
            0, 0, 0, 0, false, false, false, false, false);
        open.ColorAttachments[1] = new RenderingAttachment(
            new GpuImageHandle(2), GpuImageLayout.ColorAttachmentOptimal, GpuFormat.R16G16B16A16Sfloat,
            0, 0, 0, 0, false, false, false, false, false);

        var joined = MetalRenderHost.JoinedDescription(description, in open);

        Assert.Equal(2u, joined.Rendering.ColorCount);
        Assert.Equal(GpuFormat.R8G8B8A8Unorm, joined.Rendering.ColorFormats[0]);
        Assert.Equal(GpuFormat.R16G16B16A16Sfloat, joined.Rendering.ColorFormats[1]);
        Assert.Equal(GpuFormat.D32SfloatS8Uint, joined.Rendering.DepthFormat);
        Assert.Equal(GpuFormat.D32SfloatS8Uint, joined.Rendering.StencilFormat);
        Assert.Equal(2u, joined.StaticParameters.ColorCount);
        for (var index = 0; index < 2; index++)
        {
            Assert.Equal(0u, joined.StaticParameters.GetColorMask(index));
            Assert.False(joined.StaticParameters.GetBlendEnable(index));
        }

        Assert.Equal(description.VertexProgram, joined.VertexProgram);
        Assert.Equal(GpuPrimitiveTopology.TriangleList, joined.StaticParameters.Topology);
    }

    [Fact]
    public void SubsetDescriptionPreservesTheDrawMasksAndBlends()
    {
        var parameters = new PipelineStaticParameters { ColorCount = 5, Samples = 1 };
        var rendering = new PipelineRenderingState { ColorCount = 5 };
        var open = new RenderingState { ColorAttachmentCount = 6 };
        for (var index = 0; index < 6; index++)
        {
            parameters.SetColorMask(index, (uint)index + 1);
            parameters.SetBlendEnable(index, true);
            rendering.ColorFormats[index] = GpuFormat.R8G8B8A8Unorm;
            open.ColorAttachments[index] = new RenderingAttachment(new GpuImageHandle((ulong)index + 1),
                GpuImageLayout.ColorAttachmentOptimal, GpuFormat.R8G8B8A8Unorm,
                0, 0, 0, 0, false, false, false, false, false);
        }
        var description = new GraphicsPipelineDescription
        {
            Rendering = rendering, StaticParameters = parameters, VertexProgram = new ShaderProgram(1, 1),
            VertexInput = new PipelineVertexInputState(), VertexInfo = new VertexInputInfo(),
            VertexStage = new ShaderProgramInfo { Stage = ShaderStageKind.Vertex },
        };
        var joined = MetalRenderHost.JoinedDescription(description, in open);
        for (var index = 0; index < 5; index++)
        {
            Assert.Equal((uint)index + 1, joined.StaticParameters.GetColorMask(index));
            Assert.True(joined.StaticParameters.GetBlendEnable(index));
        }
        Assert.Equal(0u, joined.StaticParameters.GetColorMask(5));
        Assert.False(joined.StaticParameters.GetBlendEnable(5));
        Assert.Equal(6u, joined.Rendering.ColorCount);
        Assert.Equal(GpuFormat.R8G8B8A8Unorm, joined.Rendering.ColorFormats[5]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SixFiveSixTargetsPreserveEveryAttachment(bool sampleOmittedAttachment)
    {
        if (!RequireDevice()) return;
        var joined = RunSubset(forceEnd: false, sampleOmittedAttachment);
        var separate = RunSubset(forceEnd: true, sampleOmittedAttachment: false);
        Assert.Equal(3L, separate.Passes);
        var joins = !sampleOmittedAttachment && MetalV2Diagnostics.JoinSubsetDraws;
        Assert.Equal(joins ? 1L : 3L, joined.Passes);
        Assert.Equal(joins ? 2L : 0L, joined.Joins);
        // The raw native depth used by this rig has no image-cache traffic metadata.
        Assert.Equal(joins ? (5 + 6) * (long)ImageBytes : 0L, joined.Bytes);
        for (var index = 0; index < 6; index++)
        {
            Assert.Equal(separate.Pixels[index], joined.Pixels[index]);
            for (var y = 0u; y < Height; y++)
            for (var x = 0u; x < Width; x++)
            {
                var offset = (int)(y * Width + x) * 4;
                var expected = x >= Width / 2 ? new byte[] { 0, 0, 255, 255 }
                    : index == 5 ? new byte[] { 255, 0, 0, 255 } : new byte[] { 0, 255, 0, 255 };
                Assert.Equal(expected, joined.Pixels[index].AsSpan(offset, 4).ToArray());
            }
        }
    }

    [Fact]
    public void ReturningFromASubsetLandsCopiesBeforeTheDrawBindsTheirBytes()
    {
        if (!RequireDevice()) return;
        var result = RunSubset(forceEnd: false, sampleOmittedAttachment: false, copyOnReturn: true);
        var separate = RunSubset(forceEnd: true, sampleOmittedAttachment: false);
        Assert.Equal(MetalV2Diagnostics.JoinSubsetDraws ? 2L : 3L, result.Passes);
        Assert.Equal(MetalV2Diagnostics.JoinSubsetDraws ? 1L : 0L, result.Joins);
        for (var index = 0; index < 6; index++) Assert.Equal(separate.Pixels[index], result.Pixels[index]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LoggedGbufferShapeJoinsVertexAndMeshSubsetsByteExactly(bool meshSubset)
    {
        if (!RequireDevice()) return;
        var joined = RunGbuffer(forceEnd: false, meshSubset);
        var separate = RunGbuffer(forceEnd: true, meshSubset);
        Assert.Equal(3L, separate.Passes);
        Assert.Equal(MetalV2Diagnostics.JoinSubsetDraws ? 1L : 3L, joined.Passes);
        Assert.Equal(MetalV2Diagnostics.JoinSubsetDraws ? 2L : 0L, joined.Joins);
        for (var slot = 0; slot < 6; slot++)
        {
            Assert.Equal(separate.Pixels[slot], joined.Pixels[slot]);
            Assert.DoesNotContain(GuestPattern, joined.Pixels[slot]);
        }
    }

    private (byte[][] Pixels, long Passes, long Joins) RunGbuffer(bool forceEnd, bool meshSubset)
    {
        // PASS-PROF f=1170/s=14206: four R32Uint, RG16F, RGBA16F,
        // D32S8 with stencil loaded, all 1920x1080 at one sample. The fifth
        // pass's 6->5 transition starts with a mesh draw.
        using var rig = new Rig(Device.Value, gbuffer: true);
        var host = rig.Host;
        host.AsynchronousGraphics = false;
        var full = rig.Pipeline(31, -1, 0.5f, "float4(1, 0, 0, 1)", colors: 6);
        var subset = rig.Pipeline(32, -1, 0.5f, "float4(0, 1, 0, 1)", right: 0, colors: 5, mesh: meshSubset);
        var finish = rig.Pipeline(33, 0, 0.5f, "float4(0, 0, 1, 1)", colors: 6);
        const ulong stride = 16 * 1024 * 1024;
        var first = host.RenderPassSerial;
        using (host.BeginPreparation())
        {
            var targets = Enumerable.Range(0, 6).Select(i => rig.Target(TargetAddress + (ulong)i * stride, i)).ToArray();
            rig.DrawTargets(full, targets, depthClear: true);
            if (forceEnd) host.EndRendering();
            rig.DrawTargets(subset, targets[..5]);
            if (forceEnd) host.EndRendering();
            rig.DrawTargets(finish, targets);
            host.EndRendering();
        }
        var passes = host.RenderPassSerial - first;
        var (joins, _) = host.TakeSubsetJoins();
        host.FlushCommandBuffer();
        var pixels = Enumerable.Range(0, 6).Select(i => rig.Read(TargetAddress + (ulong)i * stride, i)).ToArray();
        output.WriteLine($"gbuffer mesh={meshSubset} forceEnd={forceEnd}: passes={passes} joins={joins}");
        return (pixels, passes, joins);
    }

    private (byte[][] Pixels, long Passes, long Joins, long Bytes) RunSubset(bool forceEnd, bool sampleOmittedAttachment, bool copyOnReturn = false)
    {
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.AsynchronousGraphics = false;
        var red = rig.Pipeline(11, -1, 0.5f, "float4(1, 0, 0, 1)", colors: 6);
        var green = rig.Pipeline(12, -1, 0.5f, "float4(0, 1, 0, 1)", right: 0, colors: 5);
        var blue = rig.Pipeline(13, 0, 0.5f, "float4(0, 0, 1, 1)", colors: 6);
        var firstPass = host.RenderPassSerial;
        using (host.BeginPreparation())
        {
            var targets = Enumerable.Range(0, 6).Select(i => rig.Target(TargetAddress + (ulong)i * 0x4000)).ToArray();
            rig.DrawTargets(red, targets, depthClear: true);
            var omitted = rig.LastColour;
            if (forceEnd) host.EndRendering();
            rig.DrawTargets(green, targets[..5], sample: sampleOmittedAttachment ? omitted : default);
            // Repeated draws with the same subset do not count as another avoided pass.
            if (!sampleOmittedAttachment) rig.DrawTargets(green, targets[..5]);
            if (forceEnd) host.EndRendering();
            if (copyOnReturn) host.RecordCommandProcessorWrite(LabelAddress, BitConverter.GetBytes(0x55UL), "WRITE_DATA");
            rig.DrawTargets(blue, targets, bindsCopy: copyOnReturn);
            // This rig names the synthetic binding for the join guard. Without a join,
            // ordinary pass changes keep these copies deferred to the commit.
            if (copyOnReturn) Assert.Equal(MetalV2Diagnostics.JoinSubsetDraws ? 0 : 1, host.DeferredLabelCopyCount);
            host.EndRendering();
        }
        var passes = host.RenderPassSerial - firstPass;
        var (joins, bytes) = host.TakeSubsetJoins();
        var refusals = host.TakePassRefusals();
        if (MetalV2Diagnostics.JoinSubsetDraws && sampleOmittedAttachment)
            Assert.Contains("sampled:1", refusals.Joins);
        if (MetalV2Diagnostics.JoinSubsetDraws && copyOnReturn)
            Assert.Equal("copies:1", refusals.Joins);
        host.FlushCommandBuffer();
        if (copyOnReturn) Assert.Equal(0, host.DeferredLabelCopyCount);
        var pixels = Enumerable.Range(0, 6).Select(i => rig.Read(TargetAddress + (ulong)i * 0x4000)).ToArray();
        output.WriteLine($"subset forceEnd={forceEnd} sample={sampleOmittedAttachment}: passes={passes} joins={joins} avoided={bytes}");
        return (pixels, passes, joins, bytes);
    }

    [Theory]
    [InlineData(false, false, true, false)]
    [InlineData(false, true, false, false)]
    [InlineData(true, false, false, false)]
    [InlineData(false, false, false, true)]
    public void IndexedCoverageElidesOnlyAProvenFullOverwrite(bool indirect, bool degenerate, bool elides, bool sampleMask)
    {
        if (!RequireDevice()) return;
        using var rig = new Rig(Device.Value);
        var pipeline = rig.Pipeline(21, -1, 0.5f, "float4(1, 0, 0, 1)", depth: false, sampleMask: sampleMask);
        using (rig.Host.BeginPreparation())
        {
            rig.DrawIndexed(pipeline, indirect, degenerate);
            rig.Host.EndRendering();
        }
        var (attachments, bytes) = rig.Host.TakeElidedLoads();
        elides &= MetalV2Diagnostics.ElideLoads && MetalV2Diagnostics.ElideIndexedLoads;
        Assert.Equal(elides ? 1L : 0L, attachments);
        Assert.Equal(elides ? (long)ImageBytes : 0L, bytes);
        var refusals = rig.Host.TakePassRefusals();
        if (MetalV2Diagnostics.ElideLoads && MetalV2Diagnostics.ElideIndexedLoads)
        {
            if (sampleMask) Assert.Contains("samplemask:1", refusals.Loads);
            else if (indirect) Assert.Contains("coverageunknown:1", refusals.Loads);
            else if (degenerate) Assert.Contains("coverage:1", refusals.Loads);
            else Assert.Equal("-", refusals.Loads);
        }
        Assert.Equal(("-", "-"), rig.Host.TakePassRefusals());
        rig.Host.FlushCommandBuffer();
        var pixels = rig.Read(TargetAddress);
        if (!degenerate)
        {
            for (var i = 0; i < pixels.Length; i += 4)
                Assert.Equal(new byte[] { 255, 0, 0, 255 }, pixels.AsSpan(i, 4).ToArray());
        }
        else
        {
            Assert.Contains(GuestPattern, pixels);
        }
    }

    private Outcome Run(Scene scene)
    {
        using var rig = new Rig(Device.Value);
        var host = rig.Host;
        host.AsynchronousGraphics = false;
        var red = rig.Pipeline(1, -1f, RedDepth, "float4(1.0f, 0.0f, 0.0f, 1.0f)");
        var lowered = rig.Pipeline(2, -1f, LoweredDepth, null, right: 0f);
        var green = rig.Pipeline(3, -1f, GreenDepth, "float4(0.0f, 1.0f, 0.0f, 1.0f)");
        _ = host.TakeDepthOnlyJoins();
        var firstPass = host.RenderPassSerial;
        var pending = 0;
        var deferred = 0;
        using (host.BeginPreparation())
        {
            var target = rig.Target(TargetAddress);
            rig.Draw(red, in target, depthClear: true, GpuCompareOp.Always);
            var colourView = rig.LastColour;

            if (scene == Scene.PendingClear)
            {
                var other = rig.Target(OtherAddress);
                var attachment = ((IRenderHost)host).AcquireColorAttachment(in other);
                var color = default(GpuClearColor);
                color.Float32_0 = 1f;
                ((IMetalImageCacheOwner)host).ClearTextureColor(
                    (nint)attachment.View.Handle, (ulong)PixelFormatRgba8Unorm, new SubresourceRange(0, 1, 0, 1),
                    Width, Height, in color);
                Assert.Equal(1, host.PendingClearCount);
            }

            if (scene is Scene.BindsCopyDestination or Scene.BindsOtherBytes)
            {
                host.RecordCommandProcessorWrite(LabelAddress, BitConverter.GetBytes(0x55UL), "WRITE_DATA");
                Assert.Equal(1, host.DeferredLabelCopyCount);
            }

            host.ResetBindings();
            var stage = scene switch
            {
                Scene.SamplesAttachment => Rig.Stage(textures: [(nint)colourView.Handle]),
                Scene.BindsCopyDestination => Rig.Stage(buffers: [host.GuestMemory.ResolveWritable(LabelAddress, 64)]),
                Scene.BindsOtherBytes => Rig.Stage(buffers: [host.GuestMemory.ResolveWritable(LabelAddress + 0x400, 64)]),
                _ => null,
            };
            host.NoteJoinStages(stage is null ? ReadOnlySpan<IPreparedBindings>.Empty : new IPreparedBindings[] { stage });
            pending = host.PendingClearCount;
            rig.DrawDepthOnly(lowered);
            deferred = host.DeferredLabelCopyCount;
            rig.Draw(green, in target, depthClear: false, GpuCompareOp.Less);
            host.EndRendering();
        }

        var passes = host.RenderPassSerial - firstPass;
        var (joins, sampled, copies) = host.TakeDepthOnlyJoins();
        host.FlushCommandBuffer();
        var pixels = rig.Read(TargetAddress);
        output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"{scene}: passes={passes} joins={joins} sampled={sampled} copies={copies} pending={pending} deferred={deferred}"));
        return new Outcome(pixels, passes, joins, sampled, copies, pending, deferred);
    }

    private static void AssertRedLeftGreenRight(byte[] pixels)
    {
        for (var y = 0u; y < Height; y++)
        {
            for (var x = 0u; x < Width; x++)
            {
                var texel = (int)(((y * Width) + x) * 4);
                var left = x < Width / 2;
                Assert.True(
                    pixels[texel] == (left ? 0xFF : 0x00) && pixels[texel + 1] == (left ? 0x00 : 0xFF) &&
                    pixels[texel + 2] == 0 && pixels[texel + 3] == 0xFF,
                    $"texel ({x}, {y}) is {pixels[texel]:X2}{pixels[texel + 1]:X2}{pixels[texel + 2]:X2}{pixels[texel + 3]:X2}");
            }
        }
    }

    private sealed class AddressSpace : IGuestAddressSpace
    {
        public ulong ProtectionPageSize => 4096;
        public bool TryProtect(ulong address, ulong size, GuestPageProtection protection) => true;
        public ulong AllocateAt(ulong desiredAddress, ulong size, bool executable = true, bool allowAlternative = true) => throw new NotSupportedException();
        public bool TryBackFixedRange(ulong address, ulong size, bool executable) => throw new NotSupportedException();
        public bool TryAllocateAtOrAbove(ulong desiredAddress, ulong size, bool executable, ulong alignment, out ulong actualAddress) => throw new NotSupportedException();
        public bool TryEnsureRangeCommitted(ulong address, ulong size) => throw new NotSupportedException();
        public bool TryAllocateGuestMemory(ulong size, ulong alignment, out ulong address) => throw new NotSupportedException();
        public bool TryFreeGuestMemory(ulong address) => throw new NotSupportedException();
    }

    // One page-aligned native block seen as guest memory at GuestBase, read by the CPU
    // and by the GPU through the same bytes.
    private sealed class Backing : IGuestBackedSpace, ICpuMemory, IDisposable
    {
        private readonly ulong Size;
        private readonly void* _block;

        public Backing(ulong size = 4 * 1024 * 1024)
        {
            Size = size;
            _block = NativeMemory.AlignedAlloc((nuint)Size, 16384);
            NativeMemory.Fill(_block, (nuint)Size, GuestPattern);
        }

        public void Dispose() => NativeMemory.AlignedFree(_block);

        private bool Contains(ulong address, ulong length) => address >= GuestBase && address - GuestBase + length <= Size;

        public bool TryHoldRange(ulong address, ulong size) => false;
        public bool TryHoldRangeAtOrAbove(ulong searchStart, ulong size, ulong alignment, out ulong address) { address = 0; return false; }
        public bool TryMapBacked(ulong address, ulong size, ulong backingOffset, GuestPageProtection protection, out HostViewFailure failure) { failure = HostViewFailure.None; return false; }
        public bool TryUnmapBacked(ulong address, ulong size) => false;
        public bool TryClearBacking(ulong offset, ulong size) => false;
        public bool IsBackedView(ulong address) => Contains(address, 1);
        public bool IsBackedRange(ulong address, ulong size) => Contains(address, size);
        public (ulong Base, ulong Size) BackingAliasRange => ((ulong)_block, Size);

        public bool TryResolveBackingAlias(ulong address, ulong size, out ulong alias)
        {
            alias = Contains(address, size) ? (ulong)_block + (address - GuestBase) : 0;
            return alias != 0;
        }

        public bool TryWriteBacking(ulong address, ReadOnlySpan<byte> data) => TryWrite(address, data);

        public bool TryReadBacking(ulong address, Span<byte> data) => TryRead(address, data);

        public bool TryRead(ulong virtualAddress, Span<byte> destination)
        {
            if (!Contains(virtualAddress, (ulong)destination.Length)) return false;
            new ReadOnlySpan<byte>((byte*)_block + (virtualAddress - GuestBase), destination.Length).CopyTo(destination);
            return true;
        }

        public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source)
        {
            if (!Contains(virtualAddress, (ulong)source.Length)) return false;
            source.CopyTo(new Span<byte>((byte*)_block + (virtualAddress - GuestBase), source.Length));
            return true;
        }

        public bool CanRead(ulong address, ulong size) => Contains(address, size);
    }

    private sealed class Rig : IDisposable
    {
        private readonly nint _queue;
        private readonly AddressSpace _space = new();
        private readonly PageGuard _pages;
        private readonly Backing _backing;
        private readonly nint _depth;
        private readonly bool _gbuffer;
        private uint Width => _gbuffer ? 1920u : MetalDepthOnlyJoinTests.Width;
        private uint Height => _gbuffer ? 1080u : MetalDepthOnlyJoinTests.Height;
        private ulong ImageBytes => (ulong)Width * Height * 4;
        private GpuFormat DepthFormat => _gbuffer ? GpuFormat.D32SfloatS8Uint : GpuFormat.D32Sfloat;
        private GpuFormat ColorFormat(int slot) => !_gbuffer ? GpuFormat.R8G8B8A8Unorm
            : slot < 4 ? GpuFormat.R32Uint : slot == 4 ? GpuFormat.R16G16Sfloat : GpuFormat.R16G16B16A16Sfloat;

        public Rig(nint device, bool gbuffer = false)
        {
            _gbuffer = gbuffer;
            _backing = new Backing(gbuffer ? 128UL * 1024 * 1024 : 4UL * 1024 * 1024);
            _queue = MetalNative.Send(device, MetalNative.Selector("newCommandQueue"));
            _pages = new PageGuard(_space);
            Host = new MetalRenderHost(device, _queue, _backing, _backing, _pages, new MetalGpuWorkerRelay(static () => { }));
            var descriptor = MetalNative.SendTextureDescriptor(
                MetalNative.Class("MTLTextureDescriptor"),
                MetalNative.Selector("texture2DDescriptorWithPixelFormat:width:height:mipmapped:"),
                _gbuffer ? 260u : PixelFormatDepth32Float, Width, Height, mipmapped: false);
            MetalNative.Send(descriptor, MetalNative.Selector("setUsage:"), (nint)4);
            MetalNative.Send(descriptor, MetalNative.Selector("setStorageMode:"), (nint)2);
            _depth = MetalNative.Send(device, MetalNative.Selector("newTextureWithDescriptor:"), descriptor);
            Assert.NotEqual(0, _depth);
        }

        public MetalRenderHost Host { get; }

        public GpuImageHandle LastColour { get; private set; }

        public ImageRequest Request(ulong address, int slot = 0)
        {
            var format = ColorFormat(slot);
            var texelBytes = format == GpuFormat.R16G16B16A16Sfloat ? 8u : 4u;
            var imageBytes = (ulong)Width * Height * texelBytes;
            var description = ImageDescription.Create();
            description.Data = new GuestSpan(address, imageBytes);
            description.PixelFormat = format;
            description.GuestFormat = format switch
            {
                GpuFormat.R32Uint => GuestPixelFormat.Bits32UInt,
                GpuFormat.R16G16Sfloat => GuestPixelFormat.Bits16_16Float,
                GpuFormat.R16G16B16A16Sfloat => GuestPixelFormat.Bits16_16_16_16Float,
                _ => GuestPixelFormat.Bits8_8_8_8UNorm,
            };
            description.Type = GuestImageType.Color2D;
            description.Extent = new GpuExtent3D(Width, Height, 1);
            description.Resources = new SubresourceCount(1, 1);
            description.Pitch = Width;
            description.BytesPerBlock = texelBytes;
            description.Samples = 1;
            description.TileMode = GuestTileMode.Linear;
            description.MipLayout[0] = new MipLevelLayout { Offset = 0, Size = imageBytes, Pitch = Width, Height = Height };
            var view = ImageViewDescription.Default with
            {
                Format = format,
                Type = GpuImageViewType.Type2D,
                Aspect = GpuImageAspect.ColorBit,
                LayerCount = 1,
                Usage = GpuImageUsage.ColorAttachmentBit,
            };
            return new ImageRequest(description, view, ImageRole.ColorTarget);
        }

        public ColorTargetState Target(ulong address, int slot = 0)
        {
            var request = Request(address, slot);
            var image = Host.ImageCache.FindImage(ref request);
            var resolution = new ColorTargetResolution(
                request, address, request.Description.Data.Size, new GpuExtent2D(Width, Height), 0, 0, 1, default, false, false, default);
            return new ColorTargetState(in resolution, 0, image);
        }

        public byte[] Read(ulong address, int slot = 0)
        {
            var bytes = new byte[Request(address, slot).Description.Data.Size];
            Assert.True(Host.TryReadGuest(address, bytes));
            return bytes;
        }

        // A stage that binds what the join looks at: textures and buffers.
        public static MetalRenderHost.MetalStageBindings Stage(nint[]? textures = null, MetalGuestRange[]? buffers = null) =>
            new()
            {
                Program = new ShaderProgramInfo { Stage = ShaderStageKind.Vertex, Bindings = new BindingLayout() },
                Textures = textures ?? [],
                MipTextures = new nint[]?[textures?.Length ?? 0],
                ImageCount = textures?.Length ?? 0,
                Buffers = buffers ?? [],
                BufferCount = buffers?.Length ?? 0,
            };

        private RenderingAttachment DepthAttachment(bool clear) => new(
            new GpuImageHandle((ulong)_depth), GpuImageLayout.DepthStencilAttachmentOptimal, DepthFormat,
            BitConverter.SingleToUInt32Bits(1f), 0, 0, 0, false, true, clear, _gbuffer, clear && _gbuffer);

        // The way an executor begins a draw: bindings reset, the target acquired, the pass
        // begun for the draw, then its pipeline and state.
        public void Draw(PipelineHandle pipeline, in ColorTargetState target, bool depthClear, GpuCompareOp depthTest)
        {
            var host = Host;
            host.ResetBindings();
            var attachment = ((IRenderHost)host).AcquireColorAttachment(in target);
            LastColour = attachment.View;
            var state = new RenderingState { Width = Width, Height = Height, Layers = 1, ColorAttachmentCount = 1, Samples = 1 };
            state.ColorAttachments[0] = new RenderingAttachment(
                attachment.View, GpuImageLayout.ColorAttachmentOptimal, GpuFormat.R8G8B8A8Unorm,
                0, 0, 0, 0, false, false, false, false, false);
            state.DepthStencilAttachment = DepthAttachment(depthClear);
            host.SetDynamicState(Dynamic(depthTest, colorWrites: 1));
            host.NoteUpcomingDraw(in pipeline, 6, 0);
            host.BeginRendering(in state);
            host.BindPipeline(GpuBindPoint.Graphics, pipeline);
            host.Draw(6, 1, 0, 0);
        }

        public void DrawTargets(PipelineHandle pipeline, ColorTargetState[] targets, bool depthClear = false,
            GpuImageHandle sample = default, bool bindsCopy = false)
        {
            Host.ResetBindings();
            var state = new RenderingState { Width = Width, Height = Height, Layers = 1,
                ColorAttachmentCount = (uint)targets.Length, Samples = 1 };
            for (var index = 0; index < targets.Length; index++)
            {
                var attachment = ((IRenderHost)Host).AcquireColorAttachment(in targets[index]);
                LastColour = attachment.View;
                state.ColorAttachments[index] = new RenderingAttachment(attachment.View,
                    GpuImageLayout.ColorAttachmentOptimal, ColorFormat(index),
                    0, 0, 0, 0, false, false, false, false, false);
            }
            state.DepthStencilAttachment = DepthAttachment(depthClear);
            var stage = sample.Handle != 0 ? Stage(textures: [(nint)sample.Handle])
                : bindsCopy ? Stage(buffers: [Host.GuestMemory.ResolveWritable(LabelAddress, 64)]) : null;
            Host.NoteJoinStages(stage is null ? ReadOnlySpan<IPreparedBindings>.Empty : new IPreparedBindings[] { stage });
            Host.SetDynamicState(Dynamic(GpuCompareOp.Always, (uint)((1 << targets.Length) - 1)));
            Host.NoteUpcomingDraw(in pipeline, 6, 0);
            Host.BeginRendering(in state);
            Host.BindPipeline(GpuBindPoint.Graphics, pipeline);
            Host.Draw(6, 1, 0, 0);
        }

        public void DrawIndexed(PipelineHandle pipeline, bool indirect, bool degenerate)
        {
            uint[] indices = degenerate ? [0, 1, 2, 0, 1, 2] : [3, 4, 5, 0, 1, 2];
            Host.ResetBindings();
            var indexBinding = Host.UploadTransient(MemoryMarshal.AsBytes(indices.AsSpan()), 4);
            Host.BindIndexBuffer(indexBinding, GpuIndexType.Uint32);
            var target = Target(TargetAddress);
            var attachment = ((IRenderHost)Host).AcquireColorAttachment(in target);
            var state = new RenderingState { Width = Width, Height = Height, Layers = 1, ColorAttachmentCount = 1, Samples = 1 };
            state.ColorAttachments[0] = new RenderingAttachment(attachment.View,
                GpuImageLayout.ColorAttachmentOptimal, GpuFormat.R8G8B8A8Unorm,
                0, 0, 0, 0, false, false, false, false, false);
            Host.SetDynamicState(Dynamic(GpuCompareOp.Always, 1));
            BufferBinding arguments = default;
            if (indirect)
            {
                uint[] counts = [6, 1, 0, 0, 0];
                arguments = Host.UploadTransient(MemoryMarshal.AsBytes(counts.AsSpan()), 4);
                Host.NoteUpcomingDraw(in pipeline, 0, 0);
            }
            else
            {
                Host.NoteUpcomingIndexedDraw(in pipeline, indices);
            }
            Host.BeginRendering(in state);
            Host.BindPipeline(GpuBindPoint.Graphics, pipeline);
            if (indirect) ((IRenderHost)Host).DrawIndexedIndirect(arguments);
            else ((IRenderHost)Host).DrawIndexed(6, 1, 0, 0, 0);
        }

        public void DrawDepthOnly(PipelineHandle pipeline)
        {
            var host = Host;
            var state = new RenderingState { Width = Width, Height = Height, Layers = 1, ColorAttachmentCount = 0, Samples = 1 };
            state.DepthStencilAttachment = DepthAttachment(clear: false);
            host.SetDynamicState(Dynamic(GpuCompareOp.Always, colorWrites: 0));
            host.NoteUpcomingDraw(in pipeline, 6, 0);
            host.BeginRendering(in state);
            host.BindPipeline(GpuBindPoint.Graphics, pipeline);
            host.Draw(6, 1, 0, 0);
        }

        private DynamicDrawState Dynamic(GpuCompareOp depthTest, uint colorWrites) => new(
            0, 0, Width, Height, 0, 1, new ScissorRectangle(0, 0, (int)Width, (int)Height), 1,
            0, 0, 0, 0, true, true, depthTest, false, 0, 0, 0, false, 0, 1, false, default, default, colorWrites,
            colorWrites == 0 ? (byte)0 : (byte)0xF);

        // A quad from x = left to x = right (the right edge by default) at depth z. A pipeline
        // with a colour expression writes it to the target; one without has no pixel program
        // and no colour target.
        public PipelineHandle Pipeline(ulong id, float left, float z, string? colour, float right = 1f, uint colors = 1, bool depth = true, bool sampleMask = false, bool mesh = false)
        {
            var vertexSource = $$"""
                #include <metal_stdlib>
                using namespace metal;
                struct Gen5VsOut
                {
                    float4 sharpemu_position [[position]];
                };

                vertex Gen5VsOut gen5_vs(uint sharpemu_vertex_id [[vertex_id]], uint sharpemu_instance_id [[instance_id]])
                {
                    const float2 corners[6] = {
                        float2({{F(left)}}, -1.0f), float2({{F(right)}}, -1.0f), float2({{F(left)}}, 1.0f),
                        float2({{F(left)}}, 1.0f), float2({{F(right)}}, -1.0f), float2({{F(right)}}, 1.0f) };
                    Gen5VsOut sharpemu_out = {};
                    sharpemu_out.sharpemu_position = float4(corners[sharpemu_vertex_id], {{F(z)}}, 1.0f);
                    return sharpemu_out;
                }
                """;
            if (mesh)
            {
                vertexSource = $$"""
                    #include <metal_stdlib>
                    using namespace metal;
                    struct Gen5VsOut { float4 sharpemu_position [[position]]; };
                    using Output = metal::mesh<Gen5VsOut, void, 6, 2, topology::triangle>;
                    [[mesh, max_total_threads_per_threadgroup(32)]]
                    void gen5_ms(Output out, uint lane [[thread_index_in_threadgroup]])
                    {
                        if (lane != 0) return;
                        const float2 corners[6] = {
                            float2({{F(left)}}, -1), float2({{F(right)}}, -1), float2({{F(left)}}, 1),
                            float2({{F(left)}}, 1), float2({{F(right)}}, -1), float2({{F(right)}}, 1) };
                        for (uint i = 0; i < 6; i++) {
                            Gen5VsOut v = {float4(corners[i], {{F(z)}}, 1)};
                            out.set_vertex(i, v);
                            out.set_index(i, i);
                        }
                        out.set_primitive_count(2);
                    }
                    """;
            }
            var vertex = new ShaderProgram(id, Host.CreateShaderModule(
                new MetalCompiledGuestShader(new Gen5MslShader(vertexSource, mesh ? "gen5_ms" : "gen5_vs", mesh ? Gen5MslStage.Mesh : Gen5MslStage.Vertex, 1,
                    MeshInfo: mesh ? new Gen5MslMeshInfo(32, 1, 6, 2, 3, Gen5MslMeshTopology.Triangle, 0, 2, 3, false) : null)
                {
                    ConstantPositions = [left, -1, z, 1, right, -1, z, 1, left, 1, z, 1,
                        left, 1, z, 1, right, -1, z, 1, right, 1, z, 1],
                }),
                ShaderStage.Vertex, id, id));
            var parameters = new PipelineStaticParameters
            {
                ColorCount = colour is null ? 0u : colors,
                Topology = GpuPrimitiveTopology.TriangleList,
                Samples = 1,
                DepthClipEnable = true,
            };
            var rendering = new PipelineRenderingState { ColorCount = colour is null ? 0u : colors, DepthFormat = depth ? DepthFormat : GpuFormat.Undefined, StencilFormat = depth && _gbuffer ? DepthFormat : GpuFormat.Undefined };
            ShaderProgram pixel = default;
            ShaderProgramInfo? pixelStage = null;
            if (colour is not null)
            {
                var fields = string.Join("\n", Enumerable.Range(0, (int)colors).Select(i => $"{(ColorFormat(i) == GpuFormat.R32Uint ? "uint4" : "float4")} c{i} [[color({i})]];"));
                var assignments = string.Join("\n", Enumerable.Range(0, (int)colors).Select(i => $"result.c{i} = {(ColorFormat(i) == GpuFormat.R32Uint ? $"uint4({colour})" : colour)};"));
                var pixelSource = $$"""
                    #include <metal_stdlib>
                    using namespace metal;
                    struct PixelIn
                    {
                        float4 position [[position]];
                    };

                    struct PixelOut { {{fields}} };
                    fragment PixelOut fill_fs(PixelIn in [[stage_in]])
                    {
                        PixelOut result;
                        {{assignments}}
                        return result;
                    }
                    """;
                for (var index = 0; index < colors; index++)
                {
                    parameters.SetColorMask(index, 0xF);
                    rendering.ColorFormats[index] = ColorFormat(index);
                }
                pixel = new ShaderProgram(id + 100, Host.CreateShaderModule(
                    new MetalCompiledGuestShader(new Gen5MslShader(pixelSource, "fill_fs", Gen5MslStage.Pixel, 1)
                    {
                        CannotDiscard = true,
                        ColorOutputMask = (1u << (int)colors) - 1,
                    }),
                    ShaderStage.Pixel, id + 100, id + 100));
                pixelStage = new ShaderProgramInfo { Stage = ShaderStageKind.Pixel, Hash = id + 100 };
            }

            var description = new GraphicsPipelineDescription
            {
                Rendering = rendering,
                VertexInput = new PipelineVertexInputState(),
                VertexInfo = new VertexInputInfo(),
                VertexProgram = vertex,
                VertexStage = new ShaderProgramInfo { Stage = ShaderStageKind.Vertex, Hash = id },
                PixelProgram = pixel,
                PixelInfo = new PixelInputInfo { SampleMaskExportEnable = sampleMask },
                PixelStage = pixelStage,
                StaticParameters = parameters,
            };
            return Host.CreateGraphicsPipeline(description);

            static string F(float value) => value.ToString("F4", CultureInfo.InvariantCulture) + "f";
        }

        public void Dispose()
        {
            Host.ImageCache.StopWatching();
            Host.Dispose();
            MetalNative.SendVoid(_depth, MetalNative.Selector("release"));
            _pages.Dispose();
            _backing.Dispose();
            MetalNative.SendVoid(_queue, MetalNative.Selector("release"));
        }
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Depth-only joins are encoded on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
