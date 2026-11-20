// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.Tests.Scenes;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using ImageResourceClass = SharpEmu.ShaderCompiler.Resources.ImageResourceClass;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A program translated with runtime image swizzles reads each written storage image's store channel map from its
/// shader data. The Metal host writes the maps from the bound T#s with the rest of the stage's shader data, on the
/// worker and in a binding packet alike, so one module stores through any destination select.
/// </summary>
public sealed class MetalStoreChannelMapTests
{
    private const uint Rgba8 = 56;

    [Fact]
    public void TheHostReportsRuntimeImageSwizzles()
    {
        Assert.True(MetalRenderHost.PrewarmRuntimeImageSwizzles);
    }

    [Fact]
    public void TheShaderDataCarriesEachWrittenImagesStoreChannelMap()
    {
        // Two written storage images take slots 0 and 1; the sampled one between them takes none.
        ImageResource[] images = [Storage(), Sampled(), Storage()];
        var info = new ShaderResourceInfo { Images = [.. images] };
        var resources = new SpecializedResourceInfo { Info = info };
        var layout = new BindingLayout
        {
            UserDataRegisters = [0],
            MemoryOffsetDword = 8,
            StorageImageSwizzleCount = 2,
        };
        var program = new ShaderProgramInfo
        {
            Stage = ShaderStageKind.Compute, Resources = resources, Bindings = layout, UserDataBase = 0,
        };
        const uint ZyxW = 0xF2E;
        const uint X001 = 0x204;
        var snapshot = new ResourceSnapshot
        {
            Images = [Texture(0x10_0000, ZyxW), Texture(0x20_0000, 0xFAC), Texture(0x30_0000, X001)],
            Samplers = [],
            UserData = [7],
        };

        var shaderData = new uint[layout.ShaderDataDwordCount];
        Assert.True(MetalRenderHost.WriteStageShaderData(shaderData, program, layout, info, snapshot, 0));
        // Bits 12-13 of each map carry the image's component width code (1: every component 8 bits for RGBA8), which
        // the store narrows integer channels by before Metal's saturating integer texture write.
        var width = SharpEmu.ShaderCompiler.Metal.Gen5MslTranslator.StoreComponentWidthCode(GuestImageFormat.FormatOf(snapshot.Images[0]));
        Assert.Equal(1u, width);
        var expected = (RuntimeImageSwizzle.StoreChannelMap(ZyxW) | (width << SharpEmu.ShaderCompiler.Metal.Gen5MslTranslator.StoreWidthCodeShift)) |
            ((RuntimeImageSwizzle.StoreChannelMap(X001) | (width << SharpEmu.ShaderCompiler.Metal.Gen5MslTranslator.StoreWidthCodeShift)) << 16);
        Assert.Equal(expected, shaderData[layout.StorageImageSwizzlesDword]);
        Assert.NotEqual(RuntimeImageSwizzle.StoreChannelMap(ZyxW), RuntimeImageSwizzle.StoreChannelMap(X001));
        Assert.Equal(7u, shaderData[0]);

        var packet = MetalRenderHost.BuildBindingPacket(program, resources, layout, snapshot, 0)!;
        Assert.True(packet.HasShaderData);
        Assert.Equal(shaderData, packet.ShaderData[..packet.ShaderDataCount]);
    }

    // A written integer storage image with no store channel map (the select baked into the module, or an indirect
    // table's candidate) takes a width slot, which the host fills from that image's own bound T#: the MSL store narrows
    // its integer channels by it instead of saturating (Gen5MslTranslator.StoreWidths.cs).
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TheShaderDataCarriesTheStoreWidthOfEachIntegerImageWithoutAMap(bool runtimeSwizzles)
    {
        // Image 0: a plain integer storage image (a map with run-time selects, a width slot without). Image 1: a float
        // one (a map with run-time selects, no width slot). Images 2 and 3: an indirect table's root and candidate (width slots either way).
        ImageResource[] images = [Storage(ImageNumericClass.Uint), Storage(), Storage(ImageNumericClass.Uint, root: 2),
            Storage(ImageNumericClass.Sint, root: 2)];
        var info = new ShaderResourceInfo { Images = [.. images], UsesRuntimeImageSwizzles = runtimeSwizzles };
        var resources = new SpecializedResourceInfo { Info = info };
        var layout = new BindingLayout
        {
            UserDataRegisters = [0],
            MemoryOffsetDword = 8,
            StorageImageSwizzleCount = runtimeSwizzles ? 2u : 0u,
            StorageImageWidthCount = ImageStoreWidth.SlotCount(info.Images, runtimeSwizzles),
        };
        Assert.Equal(runtimeSwizzles ? 2u : 3u, layout.StorageImageWidthCount);
        var program = new ShaderProgramInfo
        {
            Stage = ShaderStageKind.Compute, Resources = resources, Bindings = layout, UserDataBase = 0,
        };
        var snapshot = new ResourceSnapshot
        {
            Images = [Texture(0x10_0000, 0xFAC, Format16Uint), Texture(0x20_0000, 0xFAC), Texture(0x30_0000, 0xFAC, Format8Uint),
                Texture(0x40_0000, 0xFAC, Format10x3x2Uint)],
            Samplers = [],
            UserData = [7],
        };

        var shaderData = new uint[layout.ShaderDataDwordCount];
        Assert.True(MetalRenderHost.WriteStageShaderData(shaderData, program, layout, info, snapshot, 0));
        // Width codes: 2 (16-bit), 1 (8-bit), 3 (10_10_10_2), 2 bits per slot.
        var expected = runtimeSwizzles ? 1u | (3u << 2) : 2u | (1u << 2) | (3u << 4);
        Assert.Equal(expected, shaderData[layout.StorageImageWidthsDword]);
        if (runtimeSwizzles)
        {
            // Images 0 (16-bit) and 1 (RGBA8) carry their codes above their identity maps instead.
            Assert.Equal((2u << SharpEmu.ShaderCompiler.Metal.Gen5MslTranslator.StoreWidthCodeShift) |
                (1u << (SharpEmu.ShaderCompiler.Metal.Gen5MslTranslator.StoreWidthCodeShift + 16)), shaderData[layout.StorageImageSwizzlesDword]);
        }

        var packet = MetalRenderHost.BuildBindingPacket(program, resources, layout, snapshot, 0)!;
        Assert.Equal(shaderData, packet.ShaderData[..packet.ShaderDataCount]);
    }

    private const uint Format8Uint = 5;
    private const uint Format16Uint = 11;
    private const uint Format10x3x2Uint = 54;

    private static uint[] Texture(ulong address, uint select, uint format)
    {
        var words = Descriptors.Texture(address, format, 16, 16);
        words[3] = (words[3] & ~0xFFFu) | select;
        return words;
    }

    private static ImageResource Storage(ImageNumericClass numericClass, uint root = DescriptorConstants.NoIndex) => new()
    {
        ResourceClass = ImageResourceClass.Storage, NumericClass = numericClass, Dimension = ImageDimension.Dim2D,
        Written = true, IndirectRoot = root,
    };

    private static uint[] Texture(ulong address, uint select)
    {
        var words = Descriptors.Texture(address, Rgba8, 16, 16);
        words[3] = (words[3] & ~0xFFFu) | select;
        return words;
    }

    private static ImageResource Sampled() => new()
    {
        ResourceClass = ImageResourceClass.Sampled, NumericClass = ImageNumericClass.Float, Dimension = ImageDimension.Dim2D,
    };

    private static ImageResource Storage() => new()
    {
        ResourceClass = ImageResourceClass.Storage, NumericClass = ImageNumericClass.Float, Dimension = ImageDimension.Dim2D,
        Written = true,
    };
}
