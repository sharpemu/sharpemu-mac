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
        var expected = RuntimeImageSwizzle.StoreChannelMap(ZyxW) | (RuntimeImageSwizzle.StoreChannelMap(X001) << 16);
        Assert.Equal(expected, shaderData[layout.StorageImageSwizzlesDword]);
        Assert.NotEqual(RuntimeImageSwizzle.StoreChannelMap(ZyxW), RuntimeImageSwizzle.StoreChannelMap(X001));
        Assert.Equal(7u, shaderData[0]);

        var packet = MetalRenderHost.BuildBindingPacket(program, resources, layout, snapshot, 0)!;
        Assert.True(packet.HasShaderData);
        Assert.Equal(shaderData, packet.ShaderData[..packet.ShaderDataCount]);
    }

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
