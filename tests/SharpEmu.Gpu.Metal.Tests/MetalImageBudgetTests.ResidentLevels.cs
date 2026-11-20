// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Scheduling;
using Xunit;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// Upstream's resident-mip textures (ImageDescription.FirstLevel / HostExtent, ImageRequestBuilders.PartialMipChains,
/// ResidentImageBacking) on the Metal cache, through the real request builder: a streamed texture whose finer levels
/// the guest has not mapped is held from its first resident level and samples the right texels at each of them, grows
/// when more levels become resident, takes the whole chain once it is all mapped, and reads its guest bytes with the
/// Vulkan caches' hole policy. The base-level memory policy is switched off where a test is about the resident levels
/// alone, so only the port can keep the unmapped levels out (MetalImageCache.ResidentLevels.cs).
/// Mirrors upstream's ImageRequestBuildersTests.Texture_StreamedChainHoldsOnlyItsResidentLevels on the image cache.
/// </summary>
public sealed unsafe partial class MetalImageBudgetTests
{
    private const ulong StreamedAddress = GuestBase + 0x100000;
    private const ulong ReferenceAddress = GuestBase + 0x200000;

    // ---- The capability ----------------------------------------------------------------

    // The backend that composes the Metal cache asks the request builders for resident-levels textures, and
    // SHARPEMU_METAL_V2_PARTIAL_MIP_CHAINS=0 keeps them on whole chains.
    [Fact]
    public void TheMetalBackendTurnsResidentLevelTexturesOnUnlessSwitchedOff()
    {
        var saved = (ImageRequestBuilders.PartialMipChains, ImageRequestBuilders.NativeColorClearMetadata,
            SharpEmu.ShaderCompiler.Resources.ResourceMaterializer.CompiledTransactions,
            SharpEmu.ShaderCompiler.Resources.ResourceMaterializer.SpecializationGuards,
            SharpEmu.ShaderCompiler.Resources.ResourceTracker.UnvalidatedAddressWordsAreMemory, MetalImageCache.PartialMipChains);
        try
        {
            MetalImageCache.PartialMipChains = true;
            ImageRequestBuilders.PartialMipChains = false;
            MetalGpuBackend.ConfigureSharedGpuCode();
            Assert.True(ImageRequestBuilders.PartialMipChains);

            MetalImageCache.PartialMipChains = false;
            MetalGpuBackend.ConfigureSharedGpuCode();
            Assert.False(ImageRequestBuilders.PartialMipChains);
        }
        finally
        {
            (ImageRequestBuilders.PartialMipChains, ImageRequestBuilders.NativeColorClearMetadata,
                SharpEmu.ShaderCompiler.Resources.ResourceMaterializer.CompiledTransactions,
                SharpEmu.ShaderCompiler.Resources.ResourceMaterializer.SpecializationGuards,
                SharpEmu.ShaderCompiler.Resources.ResourceTracker.UnvalidatedAddressWordsAreMemory, MetalImageCache.PartialMipChains) = saved;
        }
    }

    // ---- On the device -----------------------------------------------------------------

    // A texture whose levels 0 and 1 are not mapped is described from level 2: a 64x64 texture of seven levels, whose
    // view level k is guest level 2 + k, with the texels a whole chain of the same bytes has there.
    [Fact]
    public void AStreamedTextureIsHeldFromItsFirstResidentLevelAndSamplesEveryLevel()
    {
        if (!RequireDevice()) return;
        using var scope = new ResidentLevelsScope(baseLevelCut: false);
        using var rig = new Rig(Device.Value);
        var reference = AcquireReference(rig, seed: 0x3D);
        StreamIn(rig, StreamedAddress, residentFrom: 2, seed: 0x3D);

        var request = StreamedRequest(StreamedAddress, baseLevel: 2);
        Assert.Equal(2u, request.Description.FirstLevel);
        Assert.Equal(MipLevels - 2, request.Description.Resources.Levels);
        Assert.Equal(MipSide >> 2, request.Description.HostExtent.Width);
        Assert.Equal(0u, request.View.BaseLevel);

        var identifier = rig.Cache.FindImage(ref request);
        // The request comes back as the whole chain it is part of, viewed from guest level 2.
        Assert.Equal(0u, request.Description.FirstLevel);
        Assert.Equal(MipLevels, request.Description.Resources.Levels);
        Assert.Equal(2u, request.View.BaseLevel);
        var view = rig.Cache.AcquireTextureView(identifier, request);
        rig.Harness.Submit();
        rig.Harness.Begin();

        var image = rig.Cache.GetImage(identifier);
        Assert.Equal(2u, image.FirstLevel);
        Assert.Equal(MipSide >> 2, image.Shape.Width);
        Assert.Equal(MipSide >> 2, image.Shape.Height);
        Assert.Equal(MipLevels - 2, image.Shape.MipLevels);
        Assert.False(image.IsCpuDirty);
        Assert.Equal(1L, rig.Cache.ResidentLevelCounters.Requests);
        Assert.True(rig.Cache.ResidentLevelCounters.Floors >= 1);
        Assert.Equal(image.AllocatedBytes + reference.AllocatedBytes, rig.Cache.TotalUsedMemory);
        Assert.Equal(MipLevels - 2, (uint)MetalV2Native.SendValue(view, MetalNative.Selector("mipmapLevelCount")));
        for (uint level = 0; level < MipLevels - 2; level++)
        {
            Assert.Equal(ReadLevel(rig, reference.Texture, level + 2), ReadLevel(rig, view, level));
        }
    }

    // The streamer maps a finer level and lowers the T#'s base: the same image grows to hold it, with the new level's
    // texels and the old levels' unchanged; once the whole chain is mapped a texture bound past mip 0 holds it all.
    [Fact]
    public void ATextureWhoseFirstResidentLevelDropsGrowsWithTheNewLevelAndKeepsTheOld()
    {
        if (!RequireDevice()) return;
        using var scope = new ResidentLevelsScope(baseLevelCut: false);
        using var rig = new Rig(Device.Value);
        var reference = AcquireReference(rig, seed: 0x52);
        StreamIn(rig, StreamedAddress, residentFrom: 3, seed: 0x52);

        var (identifier, _) = AcquireStreamed(rig, StreamedAddress, baseLevel: 3);
        var image = rig.Cache.GetImage(identifier);
        Assert.Equal(3u, image.FirstLevel);
        var before = image.AllocatedBytes;

        // Level 2 is mapped now (its bytes were laid out with the rest).
        StreamIn(rig, StreamedAddress, residentFrom: 2, seed: 0x52, write: false);
        var (grownId, grownView) = AcquireStreamed(rig, StreamedAddress, baseLevel: 2);
        Assert.Equal(identifier, grownId);
        Assert.Equal(2u, image.FirstLevel);
        Assert.Equal(MipLevels - 2, image.Shape.MipLevels);
        Assert.Equal(1, rig.Cache.BaseLevelCounters.Grows);
        Assert.True(image.AllocatedBytes > before);
        // The old texture is retired, not counted alongside.
        Assert.Equal(image.AllocatedBytes + reference.AllocatedBytes, rig.Cache.TotalUsedMemory);
        for (uint level = 0; level < MipLevels - 2; level++)
        {
            Assert.Equal(ReadLevel(rig, reference.Texture, level + 2), ReadLevel(rig, grownView, level));
        }

        // Everything mapped: a binding from level 1 takes the whole chain, as upstream's lookup does.
        rig.GuestBacking.Unmapped.Clear();
        var (wholeId, wholeView) = AcquireStreamed(rig, StreamedAddress, baseLevel: 1);
        Assert.Equal(identifier, wholeId);
        Assert.Equal(0u, image.FirstLevel);
        Assert.Equal(MipLevels, image.Shape.MipLevels);
        Assert.Equal(ReadLevel(rig, reference.Texture, 1), ReadLevel(rig, wholeView, 0));
        Assert.Equal(ReadLevel(rig, reference.Texture, 2), ReadLevel(rig, wholeView, 1));
    }

    // An unmapped page inside the resident levels is not a level to invent: the upload stops as upstream's does, and
    // with SHARPEMU_IMAGE_BACKING_HOLES=1 it reads the page as zero and uploads the rest. With the port switched off the
    // read is the binding's readable prefix again.
    [Fact]
    public void AHoleInTheResidentLevelsStopsTheUploadUnlessHolesAreAllowed()
    {
        if (!RequireDevice()) return;
        using var scope = new ResidentLevelsScope(baseLevelCut: false);
        using var rig = new Rig(Device.Value);
        var reference = AcquireReference(rig, seed: 0x64);
        StreamIn(rig, StreamedAddress, residentFrom: 2, seed: 0x64);
        var whole = StreamedRequest(StreamedAddress, baseLevel: 0).Description;
        var holeOffset = (whole.MipLayout[2].Offset + whole.MipLayout[2].TailOffset) & ~0xFFFUL;
        rig.GuestBacking.Unmapped.Add((StreamedAddress + holeOffset, 0x1000));

        var request = StreamedRequest(StreamedAddress, baseLevel: 2);
        var identifier = rig.Cache.FindImage(ref request);
        // A fatal ends the process; on a speculative thread it abandons the work instead, which a test can see.
        var refused = Assert.Throws<GpuFatal.AbandonedWorkException>(
            () => GpuFatal.Speculatively(() => rig.Cache.AcquireTextureView(identifier, request)));
        Assert.Contains("Could not read the mapped guest image backing", refused.Message);
        var image = rig.Cache.GetImage(identifier);
        Assert.Equal(2u, image.FirstLevel);
        Assert.True(image.IsCpuDirty);

        MetalGuestMemory.ImageBackingHoles = true;
        var view = rig.Cache.AcquireTextureView(identifier, request);
        rig.Harness.Submit();
        rig.Harness.Begin();
        Assert.False(image.IsCpuDirty);
        // The page held the start of level 2: zeros there, every later level as the guest has it.
        Assert.NotEqual(ReadLevel(rig, reference.Texture, 2), ReadLevel(rig, view, 0));
        for (uint level = 1; level < MipLevels - 2; level++)
        {
            Assert.Equal(ReadLevel(rig, reference.Texture, level + 2), ReadLevel(rig, view, level));
        }

        MetalGuestMemory.ImageBackingHoles = false;
        MetalImageCache.PartialMipChains = false;
        _ = ((IGuestImageStore)rig.Cache).MarkCpuWrite(StreamedAddress + holeOffset, 8);
        Assert.True(image.IsCpuDirty);
        _ = rig.Cache.AcquireTextureView(identifier, request);
        rig.Harness.Submit();
        rig.Harness.Begin();
        Assert.False(image.IsCpuDirty);
    }

    // The same hole inside a PRT aperture as the split's GPU process sees it: its backing has no kernel to ask (as this
    // rig's has not), so the guest process's mirrored aperture (PartiallyResidentRanges) is what lets the upload read
    // the unmapped page as zero, without SHARPEMU_IMAGE_BACKING_HOLES, as the kernel's PRT read does in one process.
    [Fact]
    public void AHoleInsideAMirroredPrtApertureReadsAsZeroWithoutTheHolePolicy()
    {
        if (!RequireDevice()) return;
        using var scope = new ResidentLevelsScope(baseLevelCut: false);
        using var rig = new Rig(Device.Value);
        var reference = AcquireReference(rig, seed: 0x66);
        StreamIn(rig, StreamedAddress, residentFrom: 2, seed: 0x66);
        var whole = StreamedRequest(StreamedAddress, baseLevel: 0).Description;
        var holeOffset = (whole.MipLayout[2].Offset + whole.MipLayout[2].TailOffset) & ~0xFFFUL;
        rig.GuestBacking.Unmapped.Add((StreamedAddress + holeOffset, 0x1000));
        PartiallyResidentRanges.ResetMirror();
        try
        {
            _ = PartiallyResidentRanges.Mirror(new PartiallyResidentRanges.Range(1, StreamedAddress, whole.Data.Size, LazilyBacked: true));
            var request = StreamedRequest(StreamedAddress, baseLevel: 2);
            var identifier = rig.Cache.FindImage(ref request);
            var before = ResidentImageBacking.MirroredPrtReads;
            var view = rig.Cache.AcquireTextureView(identifier, request);
            rig.Harness.Submit();
            rig.Harness.Begin();
            var image = rig.Cache.GetImage(identifier);
            Assert.False(image.IsCpuDirty);
            Assert.True(ResidentImageBacking.MirroredPrtReads > before);
            Assert.NotEqual(ReadLevel(rig, reference.Texture, 2), ReadLevel(rig, view, 0));
            for (uint level = 1; level < MipLevels - 2; level++)
            {
                Assert.Equal(ReadLevel(rig, reference.Texture, level + 2), ReadLevel(rig, view, level));
            }
        }
        finally
        {
            PartiallyResidentRanges.ResetMirror();
        }
    }

    // A block-compressed texture resident from a level that is not block-aligned (250x250 of a 1000x1000 BC1 chain) is
    // held from that level: the memory policy would start at the 500x500 level, which the guest has not mapped.
    [Fact]
    public void ABlockCompressedTextureIsHeldFromAnUnalignedResidentLevel()
    {
        if (!RequireDevice()) return;
        using var scope = new ResidentLevelsScope(baseLevelCut: true);
        using var rig = new Rig(Device.Value);
        const uint side = 1000;
        const uint levels = 10;
        var referenceRequest = StreamedRequest(ReferenceAddress, baseLevel: 0, GuestPixelFormat.Bc1UNorm, side, levels);
        rig.WriteGuest(ReferenceAddress, MetalTileOracle.Pattern((int)referenceRequest.Description.Data.Size, 0x7A));
        var referenceId = rig.Cache.FindImage(ref referenceRequest);
        _ = rig.Cache.AcquireTextureView(referenceId, referenceRequest);
        var reference = rig.Cache.GetImage(referenceId);

        var whole = StreamedRequest(StreamedAddress, baseLevel: 0, GuestPixelFormat.Bc1UNorm, side, levels).Description;
        Assert.Equal(1u, MetalImageCache.DesiredFirstLevel(in whole, 2));
        rig.WriteGuest(StreamedAddress, MetalTileOracle.Pattern((int)whole.Data.Size, 0x7A));
        Unmap(rig, StreamedAddress, in whole, residentFrom: 2);
        // Level 1 lies past the mapped prefix, so only a texture from level 2 can be filled.
        Assert.True(rig.Harness.Transfer.LevelBytes(in whole, 1).End > rig.GuestBacking.Unmapped[0].Start - StreamedAddress);

        var request = StreamedRequest(StreamedAddress, baseLevel: 2, GuestPixelFormat.Bc1UNorm, side, levels);
        Assert.Equal(2u, request.Description.FirstLevel);
        var identifier = rig.Cache.FindImage(ref request);
        var view = rig.Cache.AcquireTextureView(identifier, request);
        rig.Harness.Submit();
        rig.Harness.Begin();
        var image = rig.Cache.GetImage(identifier);
        Assert.Equal(2u, image.FirstLevel);
        Assert.Equal(side >> 2, image.Shape.Width);
        Assert.Equal(levels - 2, image.Shape.MipLevels);
        Assert.False(image.IsCpuDirty);
        for (uint level = 0; level < levels - 2; level++)
        {
            Assert.Equal(ReadBlockLevel(rig, reference.Texture, side, level + 2), ReadBlockLevel(rig, view, side >> 2, level));
        }
    }

    // ---- Helpers -----------------------------------------------------------------------

    // Resident-levels requests from the builder, the port on, and the base-level memory policy as the test asks;
    // everything as it was afterwards.
    private sealed class ResidentLevelsScope : IDisposable
    {
        private readonly bool _builders = ImageRequestBuilders.PartialMipChains;
        private readonly bool _cache = MetalImageCache.PartialMipChains;
        private readonly bool _baseLevel = MetalImageCache.BaseLevelAllocation;
        private readonly bool _holes = MetalGuestMemory.ImageBackingHoles;

        public ResidentLevelsScope(bool baseLevelCut)
        {
            ImageRequestBuilders.PartialMipChains = true;
            MetalImageCache.PartialMipChains = true;
            MetalImageCache.BaseLevelAllocation = baseLevelCut;
            MetalGuestMemory.ImageBackingHoles = false;
        }

        public void Dispose()
        {
            ImageRequestBuilders.PartialMipChains = _builders;
            MetalImageCache.PartialMipChains = _cache;
            MetalImageCache.BaseLevelAllocation = _baseLevel;
            MetalGuestMemory.ImageBackingHoles = _holes;
        }
    }

    // The T# of a 64 KiB-swizzled 2D texture, by default the 256x256 RGBA8 nine-level chain of the base-level tests.
    private static ImageRequest StreamedRequest(
        ulong address, uint baseLevel, GuestPixelFormat format = GuestPixelFormat.Bits8_8_8_8UNorm,
        uint side = MipSide, uint levels = MipLevels)
    {
        var words = new uint[8];
        var encoded = address >> 8;
        words[0] = (uint)encoded;
        words[1] = (uint)((encoded >> 32) & 0xFF) | ((uint)format << 20) | (((side - 1) & 0x3) << 30);
        words[2] = ((side - 1) >> 2) | ((side - 1) << 14);
        words[3] = 0xFAC | (baseLevel << 12) | ((levels - 1) << 16) |
            ((uint)GuestTileMode.Standard64KB << 20) | ((uint)GuestImageType.Color2D << 28);
        words[5] = (levels - 1) << 4;
        return ImageRequestBuilders.Texture(
            words, new ShaderImageShape(false, false, false, false, TextureNumericClass.Float)).Request;
    }

    // Lays the chain's bytes out (a pattern of the seed) and unmaps the levels before residentFrom.
    private static void StreamIn(Rig rig, ulong address, uint residentFrom, uint seed, bool write = true)
    {
        var whole = StreamedRequest(address, baseLevel: 0).Description;
        if (write)
        {
            rig.WriteGuest(address, MetalTileOracle.Pattern((int)whole.Data.Size, seed));
        }

        Unmap(rig, address, in whole, residentFrom);
    }

    // A tiled chain lays the small levels first: the levels from residentFrom are a prefix, the rest is unmapped.
    private static void Unmap(Rig rig, ulong address, in ImageDescription whole, uint residentFrom)
    {
        Assert.True(address + whole.Data.Size <= ReferenceAddress, $"the chain (0x{whole.Data.Size:X}) reaches the reference");
        var (start, end) = rig.Harness.Transfer.LevelBytes(in whole, residentFrom);
        Assert.Equal(0UL, start);
        var mappedEnd = (end + 0xFFF) & ~0xFFFUL;
        Assert.True(mappedEnd < whole.Data.Size);
        rig.GuestBacking.Unmapped.Clear();
        rig.GuestBacking.Unmapped.Add((address + mappedEnd, whole.Data.Size - mappedEnd));
    }

    // The whole chain of the same bytes, at an address nothing unmaps.
    private static MetalCachedImage AcquireReference(Rig rig, uint seed)
    {
        var request = StreamedRequest(ReferenceAddress, baseLevel: 0);
        Assert.True(ReferenceAddress + request.Description.Data.Size <= GuestBase + Backing.Size);
        rig.WriteGuest(ReferenceAddress, MetalTileOracle.Pattern((int)request.Description.Data.Size, seed));
        var identifier = rig.Cache.FindImage(ref request);
        _ = rig.Cache.AcquireTextureView(identifier, request);
        rig.Harness.Submit();
        rig.Harness.Begin();
        var image = rig.Cache.GetImage(identifier);
        Assert.Equal(0u, image.FirstLevel);
        return image;
    }

    private static (ResourceSlotIdentifier Image, nint View) AcquireStreamed(Rig rig, ulong address, uint baseLevel)
    {
        var request = StreamedRequest(address, baseLevel);
        var identifier = rig.Cache.FindImage(ref request);
        var view = rig.Cache.AcquireTextureView(identifier, request);
        rig.Harness.Submit();
        rig.Harness.Begin();
        return (identifier, view);
    }

    // One level of a BC1 texture whose level 0 is `side` texels wide, as its 8-byte blocks.
    private static byte[] ReadBlockLevel(Rig rig, nint texture, uint side, uint level)
    {
        var texels = Math.Max(side >> (int)level, 1);
        var blocks = (texels + 3) / 4;
        var rowBytes = (ulong)blocks * 8;
        var buffer = rig.Harness.Buffer(rowBytes * blocks);
        var blit = rig.Harness.Tiler.BeginBlit(rig.Harness);
        MetalNative.SendCopyTextureToBufferOptions(blit, SelCopyLevelToBuffer,
            texture, 0, level, default, new MtlSize { Width = texels, Height = texels, Depth = 1 },
            buffer, 0, (nuint)rowBytes, (nuint)(rowBytes * blocks), (nuint)MtlBlitOption.None);
        MetalTiler.EndEncoding(blit);
        rig.Harness.Submit();
        rig.Harness.Begin();
        return rig.Harness.Read(buffer, 0, rowBytes * blocks);
    }
}
