// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.Tests.Scenes;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using ImageResourceClass = SharpEmu.ShaderCompiler.Resources.ImageResourceClass;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A binding packet (MetalBindingPacket) is what PrepareBindings decodes from a stage's program and snapshot alone,
/// decoded on the resolving thread. The worker takes it only for the program, layout, snapshot and shader base it was
/// built for, and what it carries is what the worker would have decoded itself; whatever the build could not decode
/// is left to the worker.
/// </summary>
public sealed class MetalBindingPacketTests
{
    private const uint Rgba8 = 56;

    private static ImageResource Sampled() => new()
    {
        ResourceClass = ImageResourceClass.Sampled, NumericClass = ImageNumericClass.Float, Dimension = ImageDimension.Dim2D,
    };

    private static ImageResource Storage() => new()
    {
        ResourceClass = ImageResourceClass.Storage, NumericClass = ImageNumericClass.Float, Dimension = ImageDimension.Dim2D,
    };

    private static (ShaderProgramInfo Program, SpecializedResourceInfo Resources, BindingLayout Layout) Program(
        ImageResource[] images, SamplerResource[] samplers, uint[] userRegisters, uint shaderBaseDword = BindingLayout.NoShaderBase)
    {
        var info = new ShaderResourceInfo { Images = [.. images], Samplers = [.. samplers] };
        var resources = new SpecializedResourceInfo { Info = info };
        var layout = new BindingLayout
        {
            UserDataRegisters = userRegisters,
            ShaderBaseDword = shaderBaseDword,
            // The shader data runs to the memory offsets: eight dwords here.
            MemoryOffsetDword = 8,
        };
        return (new ShaderProgramInfo { Stage = ShaderStageKind.Compute, Resources = resources, Bindings = layout, UserDataBase = 0 },
            resources, layout);
    }

    private static ResourceSnapshot Snapshot(uint[][] images, uint[][] samplers, uint[] userData) =>
        new() { Images = images, Samplers = samplers, UserData = userData };

    private static MetalBindingPacket Build(ShaderProgramInfo program, ResourceSnapshot snapshot, ulong shaderBase = 0) =>
        MetalRenderHost.BuildBindingPacket(program, program.Resources!, program.Bindings!, snapshot, shaderBase)!;

    [Fact]
    public void APacketStandsOnlyForTheProgramLayoutSnapshotAndShaderBaseItWasBuiltFor()
    {
        var (program, _, layout) = Program([], [], []);
        var snapshot = Snapshot([], [], []);
        var packet = Build(program, snapshot, shaderBase: 0x1000);
        Assert.Equal(MetalRenderHost.PacketRefusal.Count, MetalRenderHost.ClassifyPacket(packet, program, snapshot, 0x1000));

        Assert.Equal(MetalRenderHost.PacketRefusal.None, MetalRenderHost.ClassifyPacket(null, program, snapshot, 0x1000));
        // A packet left on another snapshot (its copy) is not this snapshot's.
        Assert.Equal(MetalRenderHost.PacketRefusal.None, MetalRenderHost.ClassifyPacket(packet, program, Snapshot([], [], []), 0x1000));
        var (other, _, _) = Program([], [], []);
        Assert.Equal(MetalRenderHost.PacketRefusal.Program, MetalRenderHost.ClassifyPacket(packet, other, snapshot, 0x1000));
        var relaid = new MetalBindingPacket(program, new BindingLayout(), snapshot, 0x1000);
        Assert.Equal(MetalRenderHost.PacketRefusal.Program, MetalRenderHost.ClassifyPacket(relaid, program, snapshot, 0x1000));
        Assert.Equal(MetalRenderHost.PacketRefusal.ShaderBase, MetalRenderHost.ClassifyPacket(packet, program, snapshot, 0x2000));
        Assert.Same(layout, packet.Layout);
    }

    [Fact]
    public void ASnapshotThatDoesNotMatchItsProgramGetsNoPacket()
    {
        var (program, resources, layout) = Program([Sampled()], [], []);
        Assert.Null(MetalRenderHost.BuildBindingPacket(program, resources, layout, Snapshot([], [], []), 0));
    }

    [Fact]
    public void AStorageImageCarriesTheRequestTheWorkerWouldBuildAndASampledImageItsMemoKey()
    {
        var storage = Storage();
        var sampled = Sampled();
        var (program, _, _) = Program([storage, sampled], [], []);
        var storageWords = Descriptors.Texture(0x10_0000, Rgba8, 16, 16);
        var sampledWords = Descriptors.Texture(0x20_0000, Rgba8, 8, 8);
        var packet = Build(program, Snapshot([storageWords, sampledWords], [], []));

        Assert.True(packet.HasResolution(0));
        Assert.False(packet.HasMemoKey(0));
        Assert.Equal(ImageRequestBuilders.Texture(storageWords, ShapeOf(storage)), packet.Resolutions![0]);
        Assert.True(packet.HasMemoKey(1));
        Assert.False(packet.HasResolution(1));
        Assert.Equal(new MetalRenderHost.TextureMemoKey(sampledWords[0], sampledWords[1], sampledWords[2], sampledWords[3],
            sampledWords[4], sampledWords[5], sampledWords[6], sampledWords[7], ShapeOf(sampled)), packet.MemoKeys![1]);
    }

    // The worker refuses these itself (TryTextureMemo, ResolveImageBinding) and fails where it always failed.
    [Fact]
    public void ShortDescriptorsAreLeftToTheWorker()
    {
        var (program, _, _) = Program([Storage(), Sampled(), Sampled()], [new SamplerResource()], []);
        var packet = Build(program, Snapshot([[1, 2, 3], [1, 2, 3, 4, 5], new uint[8]], [[1, 2]], []));
        Assert.False(packet.HasResolution(0));
        Assert.False(packet.HasMemoKey(1));
        Assert.True(packet.HasMemoKey(2));
        Assert.False(packet.HasSampler(0));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public void ASamplerCarriesTheWordsTheHostSamplesWith(bool depthCompare, bool forcePoint)
    {
        var sampler = new SamplerResource { DepthCompare = depthCompare, ForcePointFiltering = forcePoint };
        var (program, _, _) = Program([], [sampler], []);
        var words = Descriptors.Sampler(depthCompare: CompareFunc.Less, magFilter: 1, minFilter: 1, mipFilter: 2);
        var packet = Build(program, Snapshot([], [words], []));
        Assert.True(packet.HasSampler(0));
        var expected = MetalRenderHost.NativeSamplerWords(sampler, words);
        for (var index = 0; index < 4; index++)
        {
            Assert.Equal(expected[index], packet.Samplers![0][index]);
        }

        Assert.Equal(depthCompare, ((packet.Samplers![0][0] >> 12) & 0x7) != 0);
    }

    [Fact]
    public void TheShaderDataIsWhatTheWorkerWritesForTheStage()
    {
        var (program, _, layout) = Program([], [], [2, 0, 5], shaderBaseDword: 4);
        var snapshot = Snapshot([], [], [10, 11, 12, 13, 14, 15]);
        var packet = Build(program, snapshot, shaderBase: 0x1234_5678_9ABCUL);
        var expected = new uint[layout.ShaderDataDwordCount];
        Assert.True(MetalRenderHost.WriteStageShaderData(expected, program, layout, program.Resources!.Info, snapshot, 0x1234_5678_9ABCUL));
        Assert.Equal(expected, packet.ShaderData);
        Assert.Equal([12u, 10u, 15u], packet.ShaderData![..3]);
        Assert.Equal([0x5678_9ABCu, 0x1234u], packet.ShaderData![4..6]);
    }

    // A user register past the draw's user data: no shader data, so the worker writes it itself and stops there.
    [Fact]
    public void AUserRegisterOutsideTheUserDataLeavesTheShaderDataToTheWorker()
    {
        var (program, _, _) = Program([], [], [0, 7]);
        var packet = Build(program, Snapshot([], [], [1, 2, 3]));
        Assert.Null(packet.ShaderData);
    }

    // The masks hold 64 entries: a larger stage carries its shader data alone.
    [Fact]
    public void AStageOfMoreThanSixtyFourImagesCarriesOnlyItsShaderData()
    {
        var images = Enumerable.Range(0, 65).Select(_ => Sampled()).ToArray();
        var (program, _, _) = Program(images, [], [0]);
        var packet = Build(program, Snapshot([.. images.Select(_ => new uint[8])], [], [7]));
        Assert.Null(packet.MemoKeys);
        Assert.False(packet.HasMemoKey(0));
        Assert.False(packet.HasMemoKey(64));
        Assert.Equal(7u, packet.ShaderData![0]);
    }

    private static ShaderImageShape ShapeOf(ImageResource image) => MetalRenderHost.ShapeOf(image);
}
