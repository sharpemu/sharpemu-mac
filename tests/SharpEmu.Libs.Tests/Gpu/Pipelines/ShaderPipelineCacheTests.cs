// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using SharpEmu.ShaderCompiler;
using Silk.NET.Vulkan;
using Xunit;
using static SharpEmu.Libs.Tests.Gpu.Rendering.RenderExecutorFixtures;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

// The graphics pipeline description folds the draw's static state byte for byte; the keys fold all of it.
[Collection(SchedulingStateCollection.Name)]
public sealed class ShaderPipelineCacheTests : IDisposable
{
    private const uint Format32x4Float = 77;

    private readonly FatalScope _fatal = new();

    public void Dispose() => _fatal.Dispose();

    [Fact]
    public void PixelAttributesReuseDecodedProgramButKeepDrawMappingsLive()
    {
        var guest = new PipelineTestGuest();
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);
        Gen5ShaderInstruction Move(uint pc, uint attribute) =>
            new(pc, Gen5ShaderEncoding.Vintrp, "VInterpMovF32",
                [1], [Gen5Operand.Vector(1)], [Gen5Operand.Vector(0)], new Gen5InterpolationControl(attribute, 0));
        var program = new Gen5ShaderProgram(0x1000, [Move(0, 2), Move(4, 1), Move(8, 2)]);
        var attributes = cache.GetPixelAttributes(program);

        Assert.Equal(new uint[] { 1, 2 }, attributes);
        Assert.Same(attributes, cache.GetPixelAttributes(program));
        Assert.Equal(3u, ShaderPipelineCache.ReadVertexOutputCount(attributes, new PixelInputInfo()));
        Assert.Equal(8u, ShaderPipelineCache.ReadVertexOutputCount(attributes,
            new PixelInputInfo { InputCount = 3, InterpolatorSettings = [0, 5, 7] }));
        Assert.Equal(new uint[] { 4 }, cache.GetPixelAttributes(new Gen5ShaderProgram(0x1000, [Move(0, 4)])));
        Assert.Empty(cache.GetPixelAttributes(new Gen5ShaderProgram(0x2000, [])));
        Assert.Equal(0u, ShaderPipelineCache.ReadVertexOutputCount([], new PixelInputInfo()));
    }

    [Fact]
    public void IeeeModeChangesEveryStageKey()
    {
        var ordinary = new List<uint>();
        var ieee = new List<uint>();
        StageStaticKey.Build(new VertexInputInfo(), 0, ordinary);
        StageStaticKey.Build(new VertexInputInfo { IeeeMode = true }, 0, ieee);
        Assert.False(ordinary.SequenceEqual(ieee));
        StageStaticKey.Build(new PixelInputInfo(), ordinary);
        StageStaticKey.Build(new PixelInputInfo { IeeeMode = true }, ieee);
        Assert.False(ordinary.SequenceEqual(ieee));
        StageStaticKey.Build(new ComputeInputInfo(), ordinary);
        StageStaticKey.Build(new ComputeInputInfo { IeeeMode = true }, ieee);
        Assert.False(ordinary.SequenceEqual(ieee));
        StageStaticKey.Build(new MeshDrawConfiguration(), 0, ordinary);
        StageStaticKey.Build(new MeshDrawConfiguration
        {
            Geometry = new GuestGeometryConfiguration { IeeeMode = true },
        }, 0, ieee);
        Assert.False(ordinary.SequenceEqual(ieee));
    }

    [Fact]
    public void FloatModeAndDx10ClampChangeEveryStageKey()
    {
        var ordinary = new List<uint>();
        var mode = new List<uint>();
        var clamp = new List<uint>();
        StageStaticKey.Build(new VertexInputInfo(), 0, ordinary);
        StageStaticKey.Build(new VertexInputInfo { FloatMode = 0xF0 }, 0, mode);
        StageStaticKey.Build(new VertexInputInfo { Dx10Clamp = false }, 0, clamp);
        Assert.False(ordinary.SequenceEqual(mode));
        Assert.False(ordinary.SequenceEqual(clamp));
        StageStaticKey.Build(new PixelInputInfo(), ordinary);
        StageStaticKey.Build(new PixelInputInfo { FloatMode = 0xF0 }, mode);
        StageStaticKey.Build(new PixelInputInfo { Dx10Clamp = false }, clamp);
        Assert.False(ordinary.SequenceEqual(mode));
        Assert.False(ordinary.SequenceEqual(clamp));
        StageStaticKey.Build(new PixelInputInfo { GuestWave32 = true }, mode);
        Assert.False(ordinary.SequenceEqual(mode));
        StageStaticKey.Build(new ComputeInputInfo(), ordinary);
        StageStaticKey.Build(new ComputeInputInfo { FloatMode = 0xF0 }, mode);
        StageStaticKey.Build(new ComputeInputInfo { Dx10Clamp = false }, clamp);
        Assert.False(ordinary.SequenceEqual(mode));
        Assert.False(ordinary.SequenceEqual(clamp));
    }

    [Theory]
    [InlineData(0x00u, null, null)]
    [InlineData(0x34u, Gen5ColorExportFormat.Fp16Abgr, Gen5ColorExportFormat.AR32)]
    [InlineData(0x90u, Gen5ColorExportFormat.Zero, Gen5ColorExportFormat.Abgr32)]
    public void ColorExportFormatsReachTheirBindings(uint columnFormat, Gen5ColorExportFormat? first, Gen5ColorExportFormat? second)
    {
        var guest = new PipelineTestGuest();
        var vertexAddress = PipelineTestGuest.MemoryBase + 0x1000;
        var pixelAddress = PipelineTestGuest.MemoryBase + 0x2000;
        guest.RegisterProgram(vertexAddress, PipelineTestGuest.MemoryBase + 0x8000, PipelineTestGuest.EndProgram,
            userDataAddress: PipelineTestGuest.MemoryBase + 0x9000);
        guest.RegisterProgram(pixelAddress, PipelineTestGuest.MemoryBase + 0x8100, PipelineTestGuest.EndProgram);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);
        var banks = Banks();
        banks.Shader.Vertex.ExportAddress = vertexAddress;
        banks.Shader.Pixel.Address = pixelAddress;
        // SPI_SHADER_COL_FORMAT: one nibble an export. The dual-source second binding takes
        // export 1's format.
        for (var index = 0; index < banks.Context.ShaderInterface.TargetOutputModes.Length; index++)
        {
            banks.Context.ShaderInterface.TargetOutputModes[index] = (byte)((columnFormat >> (index * 4)) & 0xF);
        }

        banks.Context.BlendControls[0] = new BlendRegisters { Enable = true, ColorDestinationFactor = 15 };
        var mappings = Enumerable.Repeat(ColorComponentMap.Identity, 8).ToArray();
        Assert.True(cache.GetGraphicsPrograms(banks.Shader.Vertex, banks.Shader.Pixel, banks.Context.ShaderInterface,
            banks.Context, mappings, pixelActive: true, depthBound: false, depthAspectBound: false).Available);
        var outputs = guest.Compiler.Requests.Last(request =>
            request.Stage == SharpEmu.ShaderCompiler.Resources.ShaderStage.Pixel).PixelOutputs;
        Assert.Equal(2, outputs.Count);
        Assert.Equal(first, outputs[0].ExportFormat);
        Assert.Equal(second, outputs[1].ExportFormat);
    }

    [Theory]
    [InlineData(false, false, 0)]
    [InlineData(true, false, 0)]
    [InlineData(true, true, 0)]
    [InlineData(false, false, 3)]
    [InlineData(true, false, 3)]
    [InlineData(true, true, 3)]
    public void SecondaryBlendExportDoesNotRequireAnotherAttachment(bool enabled, bool bypass, int packedExport)
    {
        var guest = new PipelineTestGuest();
        var vertexAddress = PipelineTestGuest.MemoryBase + 0x1000;
        var pixelAddress = PipelineTestGuest.MemoryBase + 0x2000;
        guest.RegisterProgram(vertexAddress, PipelineTestGuest.MemoryBase + 0x8000, PipelineTestGuest.EndProgram,
            userDataAddress: PipelineTestGuest.MemoryBase + 0x9000);
        guest.RegisterProgram(pixelAddress, PipelineTestGuest.MemoryBase + 0x8100, PipelineTestGuest.EndProgram);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);
        var banks = Banks();
        Array.Clear(banks.Context.ShaderInterface.TargetOutputModes);
        banks.Context.ShaderInterface.TargetOutputModes[packedExport] = 1;
        banks.Context.ShaderInterface.ColorShaderMask = 15;
        banks.Shader.Vertex.ExportAddress = vertexAddress;
        banks.Shader.Pixel.Address = pixelAddress;
        banks.Context.BlendControls[0] = new BlendRegisters { Enable = enabled, ColorDestinationFactor = 15 };
        if (bypass)
        {
            banks.Context.ColorTargets[0] = banks.Context.ColorTargets[0] with
            {
                Info = banks.Context.ColorTargets[0].Info | (1u << 16),
            };
        }
        var mappings = Enumerable.Repeat(ColorComponentMap.Identity, 8).ToArray();
        GraphicsPrograms Prepare() => cache.GetGraphicsPrograms(banks.Shader.Vertex, banks.Shader.Pixel,
            banks.Context.ShaderInterface, banks.Context, mappings, pixelActive: true, depthBound: false, depthAspectBound: false);
        Assert.True(Prepare().Available);
        var outputs = guest.Compiler.Requests[0].PixelOutputs;
        Assert.Equal(enabled && !bypass ? 2 : 1, outputs.Count);
        Assert.Equal(enabled && !bypass ? 0u : (uint)packedExport, outputs[0].ExportTarget);
        if (enabled && !bypass)
        {
            Assert.Equal(0u, outputs[1].HostLocation);
            Assert.Equal(1u, outputs[1].BlendSourceIndex);
            Assert.Equal(1u, outputs[1].ExportTarget);
            banks.Context.BlendControls[0].Enable = false;
            Assert.True(Prepare().Available);
            Assert.Single(guest.Compiler.Requests.Last(request => request.Stage == SharpEmu.ShaderCompiler.Resources.ShaderStage.Pixel).PixelOutputs);
        }
    }

    // SPI_SHADER_COL_FORMAT says what a compressed export carries. The pixel output names the format of the
    // export that feeds it, and a draw with another format compiles another program.
    [Fact]
    public void PixelOutputsCarryTheExportFormatOfTheirTarget()
    {
        var guest = new PipelineTestGuest();
        var vertexAddress = PipelineTestGuest.MemoryBase + 0x1000;
        var pixelAddress = PipelineTestGuest.MemoryBase + 0x2000;
        guest.RegisterProgram(vertexAddress, PipelineTestGuest.MemoryBase + 0x8000, PipelineTestGuest.EndProgram,
            userDataAddress: PipelineTestGuest.MemoryBase + 0x9000);
        guest.RegisterProgram(pixelAddress, PipelineTestGuest.MemoryBase + 0x8100, PipelineTestGuest.EndProgram);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);
        var banks = Banks();
        Array.Clear(banks.Context.ShaderInterface.TargetOutputModes);
        banks.Context.ShaderInterface.TargetOutputModes[0] = (byte)Gen5ColorExportFormat.Unorm16Abgr;
        banks.Context.ShaderInterface.ColorShaderMask = 15;
        banks.Shader.Vertex.ExportAddress = vertexAddress;
        banks.Shader.Pixel.Address = pixelAddress;
        var mappings = Enumerable.Repeat(ColorComponentMap.Identity, 8).ToArray();
        GraphicsPrograms Prepare() => cache.GetGraphicsPrograms(banks.Shader.Vertex, banks.Shader.Pixel,
            banks.Context.ShaderInterface, banks.Context, mappings, pixelActive: true, depthBound: false, depthAspectBound: false);
        Gen5PixelOutputBinding LastOutput() => Assert.Single(guest.Compiler.Requests
            .Last(request => request.Stage == SharpEmu.ShaderCompiler.Resources.ShaderStage.Pixel).PixelOutputs);

        Assert.True(Prepare().Available);
        Assert.Equal(Gen5ColorExportFormat.Unorm16Abgr, LastOutput().ExportFormat);
        var compiled = guest.Compiler.Requests.Count;
        banks.Context.ShaderInterface.TargetOutputModes[0] = (byte)Gen5ColorExportFormat.Fp16Abgr;
        Assert.True(Prepare().Available);
        Assert.True(guest.Compiler.Requests.Count > compiled);
        Assert.Equal(Gen5ColorExportFormat.Fp16Abgr, LastOutput().ExportFormat);
    }

    [Fact]
    public void FusedMeshPointerUpdatesWithoutCompilingAnotherProgram()
    {
        var guest = new PipelineTestGuest();
        var address = PipelineTestGuest.MemoryBase + 0x1000;
        var header = PipelineTestGuest.MemoryBase + 0x8000;
        var continuation = address + 0x100;
        guest.RegisterProgram(address, header, [0xBE940400, 0xBE960308, 0xBE970303, 0xBF810000]);
        guest.RegisterProgram(continuation, header + 0x100, PipelineTestGuest.EndProgram);
        var registry = new ShaderHeaderRegistry(guest.Context, _ => header,
            _ => new FusedProgramParts(continuation, header + 0x100));
        guest.Host.MeshShadersSupported = true;
        guest.Host.MeshLimits = new MeshShaderLimits(128, 256, 256, 32768, 65535, 65535, 65535,
            128, 32768, 32768, 32, 32, 128);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, registry);
        var banks = Banks();
        banks.Shader.Vertex.ExportAddress = address;
        banks.Shader.Vertex.GeometryUserDataAddress = 0x2012345678;
        banks.Shader.Vertex.GeometryUserScalars.Set(0, 123, UserScalarKind.Unknown);
        banks.Context.ShaderStages = 0x00400020;
        banks.UserConfig.PrimitiveType = 4;
        banks.Context.ShaderInterface.MaxOutputPerSubgroup = 32;
        banks.Context.ShaderInterface.GeometryMaxVerticesOut = 3;
        banks.Context.ShaderInterface.GeometryOutputPrimitiveType = 2;
        banks.UserConfig.GeometryEngineControl.PrimitiveGroupSize = 8;
        banks.UserConfig.GeometryEngineControl.VertexGroupSize = 24;
        GraphicsPrograms Prepare() => cache.GetMeshGraphicsPrograms(banks.Shader.Vertex, banks.Shader.Pixel,
            banks.Context.ShaderInterface, banks.Context, banks.UserConfig, [], pixelActive: false, depthBound: false);

        var first = Prepare().VertexInput.Stage;
        Assert.Equal(0u, first.Program!.UserDataBase);
        Assert.Equal(new uint[] { 0, 1, 8 }, first.Program.Bindings!.UserDataRegisters);
        Assert.Equal(0x12345678u, first.Resources.UserData[0]);
        Assert.Equal(0x20u, first.Resources.UserData[1]);
        Assert.Equal(123u, first.Resources.UserData[8]);
        banks.Shader.Vertex.GeometryUserDataAddress = 0x3198765432;
        var second = Prepare().VertexInput.Stage;
        Assert.Equal(0x98765432u, second.Resources.UserData[0]);
        Assert.Equal(0x31u, second.Resources.UserData[1]);
        Assert.Equal(1, guest.Compiler.Compilations);
    }

    [Fact]
    public void MeshDescriptorsUseTheMeshPipelineStage()
    {
        Assert.Equal(PipelineStageFlags.MeshShaderBitExt,
            SharpEmu.Gpu.Vulkan.Pipelines.DescriptorWriter.PipelineStageFlag(ShaderStageFlags.MeshBitExt));
        Assert.Equal(PipelineStageFlags.MeshShaderBitExt | PipelineStageFlags.FragmentShaderBit,
            SharpEmu.Gpu.Vulkan.Pipelines.DescriptorWriter.PipelineStageFlag(ShaderStageFlags.MeshBitExt | ShaderStageFlags.FragmentBit));
    }

    [Theory]
    [InlineData(false, "")]
    [InlineData(true, "")]
    [InlineData(false, "workgroup")]
    [InlineData(false, "output")]
    [InlineData(false, "payload-output")]
    [InlineData(false, "exact-output")]
    [InlineData(false, "components")]
    [InlineData(false, "", true)]
    [InlineData(true, "", true)]
    public void MeshProgramChecksDeviceLimitsAndRetainsPixelStage(bool triangleStrip, string limit, bool depthBound = false)
    {
        var guest = new PipelineTestGuest();
        var vertexAddress = PipelineTestGuest.MemoryBase + 0x1000;
        var pixelAddress = PipelineTestGuest.MemoryBase + 0x2000;
        guest.RegisterProgram(vertexAddress, PipelineTestGuest.MemoryBase + 0x8000, PipelineTestGuest.EndProgram);
        guest.RegisterProgram(pixelAddress, PipelineTestGuest.MemoryBase + 0x8100, PipelineTestGuest.EndProgram);
        guest.Host.MeshShadersSupported = true;
        guest.Host.MeshLimits = new MeshShaderLimits(128, 256, 256, 32768, 65535, 65535, 65535,
            128, 32768, 32768, 32, 32, 128);
        guest.Host.MeshLimits = limit switch
        {
            "workgroup" => guest.Host.MeshLimits with { MaxWorkGroupSizeX = 31 },
            "components" => guest.Host.MeshLimits with { MaxOutputComponents = 8 },
            "output" => guest.Host.MeshLimits with { MaxOutputMemoryBytes = 1535 },
            "payload-output" => guest.Host.MeshLimits with { MaxPayloadAndOutputMemoryBytes = 1535 },
            "exact-output" => guest.Host.MeshLimits with { MaxOutputMemoryBytes = 1536, MaxPayloadAndOutputMemoryBytes = 1536 },
            _ => guest.Host.MeshLimits,
        };
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);
        var banks = Banks();
        banks.Shader.Vertex.ExportAddress = vertexAddress;
        banks.Shader.Vertex.GeometryAddress = 0;
        banks.Shader.Pixel.Address = pixelAddress;
        banks.Context.ShaderStages = triangleStrip ? 0x20u : 0x00400020u;
        banks.UserConfig.PrimitiveType = triangleStrip ? 6u : 4u;
        banks.Context.ShaderInterface.MaxOutputPerSubgroup = triangleStrip ? 192u : 32u;
        banks.Context.ShaderInterface.GeometryMaxVerticesOut = 3;
        banks.Context.ShaderInterface.GeometryOutputPrimitiveType = 2;
        banks.UserConfig.GeometryEngineControl.PrimitiveGroupSize = 8;
        banks.UserConfig.GeometryEngineControl.VertexGroupSize = 24;
        var mapping = Enumerable.Repeat(ColorComponentMap.Identity, 8).ToArray();

        GraphicsPrograms Prepare() => cache.GetMeshGraphicsPrograms(banks.Shader.Vertex, banks.Shader.Pixel,
            banks.Context.ShaderInterface, banks.Context, banks.UserConfig, mapping, pixelActive: true, depthBound);
        if (limit is "components")
        {
            var error = Assert.Throws<SchedulerFatalException>(() => Prepare());
            Assert.Contains("device component limit", error.Message);
            return;
        }
        if (limit is "output" or "payload-output")
        {
            var error = Assert.Throws<SchedulerFatalException>(() => Prepare());
            Assert.Contains("mesh output exceeds device memory limits", error.Message);
            return;
        }
        var programs = Prepare();
        if (limit == "workgroup")
        {
            Assert.False(programs.Available);
            Assert.Equal(0, guest.Compiler.Compilations);
            return;
        }

        Assert.True(programs.Available);
        Assert.NotNull(programs.MeshInput);
        Assert.Equal(triangleStrip ? 192u : 32u, programs.MeshInput.Geometry.ThreadsPerGroup);
        Assert.Equal(triangleStrip ? 10u : 24u, programs.MeshInput.Geometry.InputVertexCountPerWorkgroup);
        Assert.Equal(triangleStrip, programs.MeshInput.Geometry.InputTriangleStrip);
        Assert.Equal(ShaderStageKind.Mesh, programs.VertexInput.Stage.Program!.Stage);
        Assert.Equal(ShaderStageKind.Pixel, programs.PixelInput.Stage.Program!.Stage);
        Assert.Equal(2, guest.Compiler.Compilations);
        Assert.Equal(depthBound ? 0 : 1, guest.Compiler.Requests[0].PixelOutputs.Count);
        Assert.Null(programs.SolidClear);
    }

    // Forwards each pipeline request to the static builder and keeps the description.
    private sealed class DescribingProvider : IShaderPipelineProvider
    {
        public GraphicsPrograms Graphics { get; set; } = Programs();

        public List<GraphicsPipelineDescription> Descriptions { get; } = new();

        public GraphicsPrograms GetGraphicsPrograms(VertexStageRegisters vertex, PixelStageRegisters pixel, ShaderInterfaceRegisters shaderInterface, ContextRegisters context, ReadOnlySpan<ColorComponentMap> targetExportMapping, bool pixelActive, bool depthBound, bool depthAspectBound, bool stencilAspectBound = false) => Graphics;

        public PipelineHandle CreateGraphicsPipeline(ReadOnlySpan<ColorTargetState> colors, in DepthAttachmentState depth, VertexInputInfo vertexInput, PixelInputInfo? pixelInput, ContextRegisters context, in RenderingState rendering, GpuPrimitiveTopology topology, bool primitiveRestartEnabled, bool disableBlending, ShaderProgram vertexProgram, ShaderProgram pixelProgram)
        {
            Descriptions.Add(ShaderPipelineCache.BuildGraphicsDescription(
                colors, in depth, vertexInput, pixelInput, context, in rendering, topology, primitiveRestartEnabled, disableBlending,
                vertexProgram, pixelProgram, GpuSampleCount.Count1Bit | GpuSampleCount.Count4Bit));
            return new PipelineHandle(0xA1, 0xB1, false);
        }

        public ComputeProgram GetComputeProgram(ComputeStageRegisters compute, ShaderInterfaceRegisters shaderInterface, uint dispatchInitiator, uint dimensionX, uint dimensionY, uint dimensionZ) =>
            throw new InvalidOperationException("The describing provider has no compute program.");

        public PipelineHandle CreateComputePipeline(ComputeInputInfo input, ShaderProgram program) =>
            throw new InvalidOperationException("The describing provider has no compute pipeline.");
    }

    private static GraphicsPipelineDescription Describe(Action<RegisterBanks>? configure = null, GraphicsPrograms? programs = null, uint primitiveType = PrimitiveTriangleList, bool withDepth = false)
    {
        var host = new RecordingRenderHost();
        var provider = new DescribingProvider();
        if (programs is not null)
        {
            provider.Graphics = programs;
        }

        var banks = Banks(primitiveType, withDepth);
        configure?.Invoke(banks);
        new RenderExecutor(host, provider).DrawAuto(1, banks, Auto(3));
        return Assert.Single(provider.Descriptions);
    }

    private static GraphicsPipelineDescription With(GraphicsPipelineDescription description, PipelineStaticParameters parameters) => new()
    {
        Rendering = description.Rendering,
        VertexInput = description.VertexInput,
        VertexInfo = description.VertexInfo,
        VertexProgram = description.VertexProgram,
        VertexStage = description.VertexStage,
        PixelInfo = description.PixelInfo,
        PixelProgram = description.PixelProgram,
        PixelStage = description.PixelStage,
        StaticParameters = parameters,
    };

    [Fact]
    public void StaticParameters_FoldTheBlendCullAndTopologyRegisters()
    {
        var description = Describe(banks =>
        {
            banks.Context.BlendControls[0] = new BlendRegisters
            {
                Enable = true, ColorSourceFactor = 4, ColorDestinationFactor = 5, ColorFunction = 1,
                AlphaSourceFactor = 2, AlphaDestinationFactor = 3, AlphaFunction = 2, SeparateAlpha = true,
            };
            banks.Context.RasterMode.CullBack = true;
            banks.Context.RasterMode.FrontFaceClockwise = true;
        });

        var parameters = description.StaticParameters;
        Assert.Equal(GpuPrimitiveTopology.TriangleList, parameters.Topology);
        Assert.False(parameters.PrimitiveRestartEnable);
        Assert.Equal(1u, parameters.Samples);
        Assert.False(parameters.SampleShadingEnable);
        Assert.Equal(1u, parameters.ColorCount);
        Assert.NotEqual(0u, parameters.GetColorMask(0));
        Assert.True(parameters.GetBlendEnable(0));
        Assert.False(parameters.GetBlendBypass(0));
        Assert.Equal((4, 5, 1), ((int)parameters.GetColorSourceBlend(0), (int)parameters.GetColorDestinationBlend(0), (int)parameters.GetColorBlendFunction(0)));
        Assert.Equal((2, 3, 2), ((int)parameters.GetAlphaSourceBlend(0), (int)parameters.GetAlphaDestinationBlend(0), (int)parameters.GetAlphaBlendFunction(0)));
        Assert.True(parameters.GetSeparateAlphaBlend(0));
        Assert.True(parameters.CullBack);
        Assert.False(parameters.CullFront);
        Assert.True(parameters.FrontFaceClockwise);
        Assert.False(parameters.WithDepth);
        Assert.False(parameters.StencilTestEnable);
        Assert.Equal(GpuFormat.Undefined, description.Rendering.DepthFormat);
        Assert.NotEqual(GpuFormat.Undefined, description.Rendering.ColorFormats[0]);
        Assert.Equal(1u, description.Rendering.ColorCount);
    }

    [Fact]
    public void DisableBlending_ClearsTheBlendEnableAndTheFactorsNoBackendReadsWithoutIt()
    {
        var programs = Programs();
        var description = Describe(
            banks => banks.Context.BlendControls[0] = new BlendRegisters { Enable = true, ColorSourceFactor = 4, ColorDestinationFactor = 5 },
            new GraphicsPrograms { Vertex = programs.Vertex, Pixel = programs.Pixel, VertexInput = programs.VertexInput, PixelInput = programs.PixelInput, DisableBlending = true });

        Assert.False(description.StaticParameters.GetBlendEnable(0));
        Assert.Equal(0, description.StaticParameters.GetColorSourceBlend(0));
        Assert.Equal(0, description.StaticParameters.GetColorDestinationBlend(0));
    }

    [Fact]
    public void RectangleList_DisablesCullingAndKeepsThePatchTopology()
    {
        var description = Describe(banks => banks.Context.RasterMode.CullBack = true, primitiveType: 7);

        Assert.Equal(GpuPrimitiveTopology.PatchList, description.StaticParameters.Topology);
        Assert.False(description.StaticParameters.CullBack);
    }

    [Theory]
    [InlineData(true, 15, 0, 1, 0, true)]
    [InlineData(true, 1, 16, 1, 0, true)]
    [InlineData(true, 1, 0, 17, 0, true)]
    [InlineData(true, 1, 0, 1, 18, true)]
    [InlineData(true, 4, 5, 1, 0, false)]
    [InlineData(false, 15, 16, 17, 18, false)]
    public void SecondBlendSource_IsUsedOnlyByEnabledSrc1Factors(bool enable, byte colorSource, byte colorDestination,
        byte alphaSource, byte alphaDestination, bool expected)
    {
        var blend = new BlendRegisters
        {
            Enable = enable, ColorSourceFactor = colorSource, ColorDestinationFactor = colorDestination,
            AlphaSourceFactor = alphaSource, AlphaDestinationFactor = alphaDestination, SeparateAlpha = true,
        };

        Assert.Equal(expected, blend.UsesSecondarySource);
    }

    [Fact]
    public void SecondBlendSourceIgnoresDormantAlphaFactors()
    {
        var blend = new BlendRegisters
        {
            Enable = true, ColorSourceFactor = 1, AlphaSourceFactor = 17, AlphaDestinationFactor = 18,
            SeparateAlpha = false,
        };
        Assert.False(blend.UsesSecondarySource);
    }

    [Fact]
    public void DepthTarget_FoldsTheDepthAndStencilState()
    {
        var description = Describe(withDepth: true);

        Assert.True(description.StaticParameters.WithDepth);
        Assert.NotEqual(GpuFormat.Undefined, description.Rendering.DepthFormat);
        Assert.Equal(StencilOperations.Default, description.StaticParameters.StencilFront);
    }

    [Fact]
    public void VertexInputState_FoldsEveryBindingAndAttribute()
    {
        var defaults = Programs();
        var vertexInput = new VertexInputInfo
        {
            Buffers = [new VertexInputBuffer(VertexBase, 32, 3), new VertexInputBuffer(VertexBase + 0x1000, 16, 3, PerInstance: true)],
            Attributes =
            [
                new VertexAttributeResource(new BufferDescriptorWords(unchecked((uint)VertexBase), (uint)(VertexBase >> 32) | (32u << 16), 3, Format32x4Float << 12), 0, 4, 0, 0, 0, 0),
                new VertexAttributeResource(new BufferDescriptorWords(unchecked((uint)VertexBase), (uint)(VertexBase >> 32) | (32u << 16), 3, Format32x4Float << 12), 4, 2, 1, 0, 0, 16),
                new VertexAttributeResource(new BufferDescriptorWords(unchecked((uint)VertexBase), (uint)(VertexBase >> 32) | (16u << 16), 3, Format32x4Float << 12), 6, 4, 2, 1, 1, 0),
            ],
            Stage = defaults.VertexInput.Stage,
        };
        var programs = new GraphicsPrograms { Vertex = defaults.Vertex, Pixel = defaults.Pixel, VertexInput = vertexInput, PixelInput = defaults.PixelInput };

        var state = Describe(programs: programs).VertexInput;

        Assert.Equal(2, state.BindingCount);
        Assert.Equal(3, state.AttributeCount);
        Assert.Equal(new PipelineVertexBinding(32, false), state.Bindings[0]);
        Assert.Equal(new PipelineVertexBinding(16, true), state.Bindings[1]);
        Assert.Equal(new PipelineVertexAttribute(16, 0), state.Attributes[1]);
        Assert.Equal(new PipelineVertexAttribute(0, 1), state.Attributes[2]);
    }

    [Fact]
    public void GraphicsPipelineKey_FoldsAll167StaticBytes()
    {
        var description = Describe();
        var key = ShaderPipelineCache.KeyOf(description);
        Assert.Equal(key, ShaderPipelineCache.KeyOf(With(description, PipelineStaticParameters.FromBytes(description.StaticParameters.Bytes))));
        Assert.Equal(167, description.StaticParameters.Bytes.Length);

        for (var index = 0; index < description.StaticParameters.Bytes.Length; index++)
        {
            var bytes = description.StaticParameters.Bytes.ToArray();
            bytes[index] ^= 0x01;
            var flipped = ShaderPipelineCache.KeyOf(With(description, PipelineStaticParameters.FromBytes(bytes)));
            Assert.False(key.Equals(flipped), $"byte {index} did not change the key");
        }
    }

    [Fact]
    public void GraphicsPipelineKey_FoldsTheProgramsTheRenderingAndTheVertexInput()
    {
        var description = Describe();
        var key = ShaderPipelineCache.KeyOf(description);

        var otherVertex = new GraphicsPipelineDescription
        {
            Rendering = description.Rendering, VertexInput = description.VertexInput, VertexInfo = description.VertexInfo,
            VertexProgram = new ShaderProgram(0x99), VertexStage = description.VertexStage, PixelInfo = description.PixelInfo,
            PixelProgram = description.PixelProgram, PixelStage = description.PixelStage, StaticParameters = description.StaticParameters,
        };
        Assert.NotEqual(key, ShaderPipelineCache.KeyOf(otherVertex));

        var otherRendering = new PipelineRenderingState { ColorCount = 1 };
        otherRendering.ColorFormats[0] = GpuFormat.R16G16B16A16Sfloat;
        var renderingChanged = new GraphicsPipelineDescription
        {
            Rendering = otherRendering, VertexInput = description.VertexInput, VertexInfo = description.VertexInfo,
            VertexProgram = description.VertexProgram, VertexStage = description.VertexStage, PixelInfo = description.PixelInfo,
            PixelProgram = description.PixelProgram, PixelStage = description.PixelStage, StaticParameters = description.StaticParameters,
        };
        Assert.NotEqual(key, ShaderPipelineCache.KeyOf(renderingChanged));
        Assert.Equal(key, ShaderPipelineCache.KeyOf(With(description, description.StaticParameters)));
    }

    [Fact]
    public void ComputePipelineKey_IsTheProgramId()
    {
        Assert.Equal(new ComputePipelineKey(5), new ComputePipelineKey(5));
        Assert.NotEqual(new ComputePipelineKey(5), new ComputePipelineKey(6));

        var guest = new PipelineTestGuest();
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);
        var input = ComputeProgram().Input;
        var first = cache.CreateComputePipeline(input, new ShaderProgram(7, 1));
        var again = cache.CreateComputePipeline(input, new ShaderProgram(7, 1));
        var other = cache.CreateComputePipeline(input, new ShaderProgram(8, 2));

        Assert.Equal(first, again);
        Assert.NotEqual(first, other);
        Assert.Equal(2, guest.Host.ComputePipelines.Count);
        Assert.Equal(2, cache.ComputePipelineCount);
    }

    [Fact]
    public void TryCreateComputePipeline_ReportsTheHostCompileAndCachesTheResult()
    {
        var guest = new PipelineTestGuest();
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);
        var input = ComputeProgram().Input;
        var program = new ShaderProgram(7, 1);
        guest.Host.PendingComputeCompiles[program.Id] = 2;

        Assert.False(cache.TryCreateComputePipeline(input, program, out _));
        Assert.False(cache.TryCreateComputePipeline(input, program, out _));
        Assert.Empty(guest.Host.ComputePipelines);
        Assert.Equal(0, cache.ComputePipelineCount);

        Assert.True(cache.TryCreateComputePipeline(input, program, out var ready));
        Assert.Single(guest.Host.ComputePipelines);

        // The cached pipeline is served without asking the host again.
        Assert.True(cache.TryCreateComputePipeline(input, program, out var cached));
        Assert.Equal(ready, cached);
        Assert.Single(guest.Host.ComputePipelines);
        Assert.Equal(1, cache.ComputePipelineCount);
    }

    [Fact]
    public void GraphicsPipelines_AreCachedByTheirKey()
    {
        var guest = new PipelineTestGuest();
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);
        var host = new RecordingRenderHost();
        var executor = new RenderExecutor(host, new CachingProvider(cache));

        executor.DrawAuto(1, Banks(), Auto(3));
        executor.DrawAuto(2, Banks(), Auto(3));
        executor.DrawAuto(3, Banks(PrimitiveTriangleStrip), Auto(4));
        // The lookup key is reused across draws: after another pipeline, the first is found again.
        executor.DrawAuto(4, Banks(), Auto(3));
        executor.DrawAuto(5, Banks(PrimitiveTriangleStrip), Auto(4));

        Assert.Equal(2, guest.Host.GraphicsPipelines.Count);
        Assert.Equal(2, cache.GraphicsPipelineCount);
    }

    [Fact]
    public void TryCreateGraphicsPipeline_CachesNothingWhileTheHostCompiles()
    {
        var guest = new PipelineTestGuest();
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);
        var host = new RecordingRenderHost { LeavesOutDrawsWhileCompiling = true };
        var executor = new RenderExecutor(host, new CachingProvider(cache));
        guest.Host.PendingGraphicsCompiles = 2;

        executor.DrawAuto(1, Banks(), Auto(3));
        executor.DrawAuto(2, Banks(), Auto(3));
        // Left out twice, and nothing kept that a later draw could take for the pipeline.
        Assert.DoesNotContain(host.Calls, call => call.StartsWith("draw ", StringComparison.Ordinal));
        Assert.Empty(guest.Host.GraphicsPipelines);
        Assert.Equal(0, cache.GraphicsPipelineCount);

        executor.DrawAuto(3, Banks(), Auto(3));
        executor.DrawAuto(4, Banks(), Auto(3));
        Assert.Equal(2, host.Calls.Count(call => call.StartsWith("draw ", StringComparison.Ordinal)));
        Assert.Single(guest.Host.GraphicsPipelines);
        Assert.Equal(1, cache.GraphicsPipelineCount);
        // The cached pipeline is served without asking the host again.
        Assert.Equal(3, guest.Host.GraphicsCompileRequests);
    }

    [Fact]
    public void ComputeRetry_TakesTheKeptProgramBackOnlyWhileWhatItReadHolds()
    {
        const ulong code = PipelineTestGuest.MemoryBase + 0x1000;
        const ulong header = PipelineTestGuest.MemoryBase + 0x8000;
        const ulong table = PipelineTestGuest.MemoryBase + 0x9000;
        var guest = new PipelineTestGuest();
        guest.Host.ResidentReadsEnabled = true;
        guest.RegisterProgram(code, header, PipelineTestGuest.EndProgram);
        guest.WriteWords(table, 0x1234);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);
        var registers = new ComputeStageRegisters { Address = code, ThreadsX = 1, ThreadsY = 1, ThreadsZ = 1, UserScalarCount = 8 };
        var retry = new ComputeProgramRetry();

        var first = cache.GetComputeProgram(registers, new ShaderInterfaceRegisters(), 0x8001, 1, 1, 1, retry);
        Assert.False(retry.Reused);
        Assert.True(retry.Reads.Complete);
        // What a materialisation reading the table would have logged.
        retry.Reads.Word(table, read: true, 0x1234);

        Assert.Same(first, cache.GetComputeProgram(registers, new ShaderInterfaceRegisters(), 0x8001, 1, 1, 1, retry));
        Assert.True(retry.Reused);

        // Other dimensions are another dispatch.
        Assert.NotSame(first, cache.GetComputeProgram(registers, new ShaderInterfaceRegisters(), 0x8001, 2, 1, 1, retry));
        Assert.False(retry.Reused);

        // A word the preparation read changed since: it is prepared again.
        var again = cache.GetComputeProgram(registers, new ShaderInterfaceRegisters(), 0x8001, 1, 1, 1, retry);
        retry.Reads.Word(table, read: true, 0x1234);
        guest.WriteWords(table, 0x5678);
        Assert.NotSame(again, cache.GetComputeProgram(registers, new ShaderInterfaceRegisters(), 0x8001, 1, 1, 1, retry));
        Assert.False(retry.Reused);

        // So is one whose read the host cannot answer without waiting.
        var kept = cache.GetComputeProgram(registers, new ShaderInterfaceRegisters(), 0x8001, 1, 1, 1, retry);
        Assert.True(retry.Reused);
        retry.Reads.Word(table, read: true, 0x5678);
        guest.Host.ResidentReadsEnabled = false;
        Assert.NotSame(kept, cache.GetComputeProgram(registers, new ShaderInterfaceRegisters(), 0x8001, 1, 1, 1, retry));
        Assert.False(retry.Reused);
    }

    [Fact]
    public void GuestReadLog_HoldsOnlyWhatStillReadsTheSame()
    {
        var bytes = new Dictionary<ulong, byte>();
        bool Read(ulong address, Span<byte> destination, bool clean)
        {
            for (var index = 0; index < destination.Length; index++)
            {
                if (!bytes.TryGetValue(address + (ulong)index, out destination[index]))
                {
                    return false;
                }
            }

            return true;
        }

        for (var address = 0x100UL; address < 0x110; address++)
        {
            bytes[address] = (byte)address;
        }

        var log = new GuestReadLog();
        Assert.True(log.Holds(Read));
        log.Word(0x100, read: true, 0x03020100);
        log.Bytes(0x104, read: true, [4, 5, 6, 7, 8]);
        Assert.True(log.Holds(Read));
        bytes[0x106] = 0xFF;
        Assert.False(log.Holds(Read));
        bytes[0x106] = 6;
        Assert.True(log.Holds(Read));

        // A read that failed, or more than the bound, is never taken as holding.
        log.Word(0x200, read: false, 0);
        Assert.False(log.Complete);
        Assert.False(log.Holds(Read));
        log.Clear();
        log.Bytes(0x100, read: true, new byte[64 * 1024 + 1]);
        Assert.False(log.Holds(Read));
    }

    // Programs from the fixtures, pipelines from the real cache over the fake host.
    private sealed class CachingProvider(ShaderPipelineCache cache) : IShaderPipelineProvider
    {
        private readonly GraphicsPrograms _programs = Programs();

        public GraphicsPrograms GetGraphicsPrograms(VertexStageRegisters vertex, PixelStageRegisters pixel, ShaderInterfaceRegisters shaderInterface, ContextRegisters context, ReadOnlySpan<ColorComponentMap> targetExportMapping, bool pixelActive, bool depthBound, bool depthAspectBound, bool stencilAspectBound = false) => _programs;

        public PipelineHandle CreateGraphicsPipeline(ReadOnlySpan<ColorTargetState> colors, in DepthAttachmentState depth, VertexInputInfo vertexInput, PixelInputInfo? pixelInput, ContextRegisters context, in RenderingState rendering, GpuPrimitiveTopology topology, bool primitiveRestartEnabled, bool disableBlending, ShaderProgram vertexProgram, ShaderProgram pixelProgram) =>
            cache.CreateGraphicsPipeline(colors, in depth, vertexInput, pixelInput, context, in rendering, topology, primitiveRestartEnabled, disableBlending, vertexProgram, pixelProgram);

        public bool TryCreateGraphicsPipeline(ReadOnlySpan<ColorTargetState> colors, in DepthAttachmentState depth, VertexInputInfo vertexInput, PixelInputInfo? pixelInput, ContextRegisters context, in RenderingState rendering, GpuPrimitiveTopology topology, bool primitiveRestartEnabled, bool disableBlending, ShaderProgram vertexProgram, ShaderProgram pixelProgram, out PipelineHandle handle) =>
            cache.TryCreateGraphicsPipeline(colors, in depth, vertexInput, pixelInput, context, in rendering, topology, primitiveRestartEnabled, disableBlending, vertexProgram, pixelProgram, out handle);

        public ComputeProgram GetComputeProgram(ComputeStageRegisters compute, ShaderInterfaceRegisters shaderInterface, uint dispatchInitiator, uint dimensionX, uint dimensionY, uint dimensionZ) =>
            throw new InvalidOperationException("The caching provider has no compute program.");

        public PipelineHandle CreateComputePipeline(ComputeInputInfo input, ShaderProgram program) => cache.CreateComputePipeline(input, program);
    }

    [Fact]
    public void NoAttachments_TakeTheSampleCountFromTheAntialiasingConfig()
    {
        var banks = Banks();
        banks.Context.AntialiasingConfig.SampleCountLog2 = 2;
        var programs = Programs();
        var rendering = new RenderingState { Samples = 1 };

        var description = ShaderPipelineCache.BuildGraphicsDescription(
            [], default, programs.VertexInput, programs.PixelInput, banks.Context, in rendering, GpuPrimitiveTopology.TriangleList, false, false,
            programs.Vertex, programs.Pixel, GpuSampleCount.Count1Bit | GpuSampleCount.Count4Bit);

        Assert.Equal(4u, description.StaticParameters.Samples);
        Assert.Equal(0u, description.StaticParameters.ColorCount);
    }

    [Theory]
    [InlineData(0x0u, 0x0u)]
    [InlineData(0xFu, 0xFu)]
    [InlineData(0xFu, 0xFu, true, false)]
    [InlineData(0xF000u, 0x0u, true, false)]
    [InlineData(0xFu, 0x0u, true, true)]
    [InlineData(0xF000u, 0xFu, true, true)]
    public void ColorTarget_TheProgramNeverExports_IsNotWritten(uint exportMasks, uint expectedMask,
        bool dualSource = false, bool bypass = false)
    {
        // Astro Bot's depth-only pattern pass keeps a color target bound but exports only to
        // the null target; writing the undefined host output drew a rectangle over the movie.
        var banks = Banks();
        banks.Context.RenderTargetMask = 0xF;
        if (dualSource)
        {
            Array.Clear(banks.Context.ShaderInterface.TargetOutputModes);
            banks.Context.ShaderInterface.TargetOutputModes[3] = 1;
            banks.Context.ShaderInterface.ColorShaderMask = 15;
            banks.Context.BlendControls[0] = new BlendRegisters { Enable = true, ColorSourceFactor = 15 };
            if (bypass)
            {
                banks.Context.ColorTargets[0] = banks.Context.ColorTargets[0] with
                {
                    Info = banks.Context.ColorTargets[0].Info | (1u << 16),
                };
            }
        }
        var programs = Programs(pixelStage: Stage(new ShaderProgramInfo { Stage = ShaderStageKind.Pixel, PixelColorExportMasks = exportMasks }));
        var resolution = new ColorTargetResolution(
            default, 0x1000, 0x10000, new GpuExtent2D(64, 64), 0, 0, 1, ColorComponentMap.Identity, false, false, default);
        ColorTargetState[] colors = [new(in resolution, 0, new SharpEmu.Libs.Gpu.Buffers.ResourceSlotIdentifier(1, 1))];
        var rendering = new RenderingState { Samples = 1, ColorAttachmentCount = 1 };
        rendering.ColorAttachments[0] = new RenderingAttachment(
            default, GpuImageLayout.ColorAttachmentOptimal, GpuFormat.R8G8B8A8Unorm, 0, 0, 0, 0, false, false, false, false, false);

        var description = ShaderPipelineCache.BuildGraphicsDescription(
            colors, default, programs.VertexInput, programs.PixelInput, banks.Context, in rendering, GpuPrimitiveTopology.TriangleList, false, false,
            programs.Vertex, programs.Pixel, GpuSampleCount.Count1Bit);

        Assert.Equal(expectedMask, description.StaticParameters.GetColorMask(0));
    }

    private static PipelineStaticParameters NormalizableParameters(uint colorCount = 1)
    {
        var parameters = new PipelineStaticParameters { ColorCount = colorCount };
        for (var index = 0; index < colorCount; index++)
        {
            parameters.SetColorMask(index, 0xF);
        }

        return parameters;
    }

    [Theory]
    [InlineData(true, false, 0.25f, 0.75f, false, 0f, 0f)]
    [InlineData(true, true, 0.25f, 0.75f, true, 0.25f, 0.75f)]
    [InlineData(false, true, 0.25f, 0.75f, false, 0f, 0f)]
    [InlineData(false, false, 0.25f, 0.75f, false, 0f, 0f)]
    public void DepthBounds_AreKeyedOnlyWhileTheHostPipelineReadsThem(
        bool inPipelineState, bool enabled, float min, float max, bool expectedEnable, float expectedMin, float expectedMax)
    {
        // Vulkan with the feature (inPipelineState) bakes the bounds into the pipeline; Metal
        // sets them on the encoder, so a draw's bounds never split its pipeline.
        var parameters = NormalizableParameters();
        parameters.DepthBoundsTestEnable = enabled;
        parameters.DepthMinBounds = min;
        parameters.DepthMaxBounds = max;

        ShaderPipelineCache.NormalizeStaticParameters(parameters, inPipelineState);

        Assert.Equal(expectedEnable, parameters.DepthBoundsTestEnable);
        Assert.Equal(expectedMin, parameters.DepthMinBounds);
        Assert.Equal(expectedMax, parameters.DepthMaxBounds);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void StencilOperations_AreKeyedOnlyWhileTheStencilTestIsOn(bool enabled)
    {
        var operations = new StencilOperations(GpuStencilOp.Replace, GpuStencilOp.IncrementAndClamp, GpuStencilOp.Invert, GpuCompareOp.Less);
        var parameters = NormalizableParameters();
        parameters.StencilTestEnable = enabled;
        parameters.StencilFront = operations;
        parameters.StencilBack = operations;

        ShaderPipelineCache.NormalizeStaticParameters(parameters, true);

        var expected = enabled ? operations : StencilOperations.Default;
        Assert.Equal(expected, parameters.StencilFront);
        Assert.Equal(expected, parameters.StencilBack);
    }

    [Theory]
    [InlineData(true, false, 0xFu, false)]
    [InlineData(false, false, 0xFu, true)]
    [InlineData(true, true, 0xFu, true)]
    [InlineData(true, false, 0x0u, true)]
    public void BlendState_IsKeyedOnlyForAnAttachmentThatBlends(bool enable, bool bypass, uint mask, bool zeroed)
    {
        var parameters = NormalizableParameters(2);
        for (var index = 0; index < 2; index++)
        {
            parameters.SetBlendEnable(index, enable);
            parameters.SetBlendBypass(index, bypass);
            parameters.SetColorMask(index, mask);
            parameters.SetSeparateAlphaBlend(index, true);
            parameters.SetColorSourceBlend(index, 5);
            parameters.SetColorDestinationBlend(index, 6);
            parameters.SetColorBlendFunction(index, 1);
            parameters.SetAlphaSourceBlend(index, 7);
            parameters.SetAlphaDestinationBlend(index, 8);
            parameters.SetAlphaBlendFunction(index, 2);
        }

        ShaderPipelineCache.NormalizeStaticParameters(parameters, true);

        for (var index = 0; index < 2; index++)
        {
            Assert.Equal(mask, parameters.GetColorMask(index));
            Assert.Equal(zeroed ? (byte)0 : (byte)5, parameters.GetColorSourceBlend(index));
            Assert.Equal(zeroed ? (byte)0 : (byte)6, parameters.GetColorDestinationBlend(index));
            Assert.Equal(zeroed ? (byte)0 : (byte)1, parameters.GetColorBlendFunction(index));
            Assert.Equal(zeroed ? (byte)0 : (byte)7, parameters.GetAlphaSourceBlend(index));
            Assert.Equal(zeroed ? (byte)0 : (byte)8, parameters.GetAlphaDestinationBlend(index));
            Assert.Equal(zeroed ? (byte)0 : (byte)2, parameters.GetAlphaBlendFunction(index));
            Assert.Equal(!zeroed, parameters.GetBlendEnable(index));
            Assert.Equal(!zeroed, parameters.GetSeparateAlphaBlend(index));
            Assert.Equal(bypass && !zeroed, parameters.GetBlendBypass(index));
        }
    }

    [Fact]
    public void BlendState_TheAlphaFactorsOfAColorOnlyBlendAreNotKeyed()
    {
        var parameters = NormalizableParameters();
        parameters.SetBlendEnable(0, true);
        parameters.SetSeparateAlphaBlend(0, false);
        parameters.SetColorSourceBlend(0, 5);
        parameters.SetAlphaSourceBlend(0, 7);
        parameters.SetAlphaDestinationBlend(0, 8);
        parameters.SetAlphaBlendFunction(0, 2);

        ShaderPipelineCache.NormalizeStaticParameters(parameters, true);

        Assert.Equal((byte)5, parameters.GetColorSourceBlend(0));
        Assert.Equal((byte)0, parameters.GetAlphaSourceBlend(0));
        Assert.Equal((byte)0, parameters.GetAlphaDestinationBlend(0));
        Assert.Equal((byte)0, parameters.GetAlphaBlendFunction(0));
    }

    [Fact]
    public void NormalizedParameters_ThatDifferOnlyInIgnoredStateShareAKey()
    {
        PipelineStaticParameters Draw(float min, byte factor, GpuStencilOp stencil, bool depthBounds = false, bool blend = false)
        {
            var parameters = NormalizableParameters();
            parameters.DepthBoundsTestEnable = depthBounds;
            parameters.DepthMinBounds = min;
            parameters.DepthMaxBounds = min + 0.5f;
            parameters.SetBlendEnable(0, blend);
            parameters.SetColorSourceBlend(0, factor);
            parameters.StencilFront = StencilOperations.Default with { FailOperation = stencil };
            ShaderPipelineCache.NormalizeStaticParameters(parameters, true);
            return parameters;
        }

        Assert.Equal(Draw(0.1f, 4, GpuStencilOp.Keep), Draw(0.3f, 9, GpuStencilOp.Replace));
        // What is switched on still tells draws apart.
        Assert.NotEqual(Draw(0.1f, 4, GpuStencilOp.Keep, depthBounds: true), Draw(0.3f, 4, GpuStencilOp.Keep, depthBounds: true));
        Assert.NotEqual(Draw(0.1f, 4, GpuStencilOp.Keep, blend: true), Draw(0.1f, 9, GpuStencilOp.Keep, blend: true));
    }

    [Fact]
    public void GraphicsDescription_OfDrawsThatDifferOnlyInDisabledBlendFactors_IsOneKey()
    {
        PipelineStaticParameters Parameters(bool enable, byte sourceFactor)
        {
            var banks = Banks();
            banks.Context.RenderTargetMask = 0xF;
            banks.Context.BlendControls[0] = new BlendRegisters { Enable = enable, ColorSourceFactor = sourceFactor };
            var programs = Programs(pixelStage: Stage(new ShaderProgramInfo { Stage = ShaderStageKind.Pixel, PixelColorExportMasks = 0xF }));
            var resolution = new ColorTargetResolution(
                default, 0x1000, 0x10000, new GpuExtent2D(64, 64), 0, 0, 1, ColorComponentMap.Identity, false, false, default);
            ColorTargetState[] colors = [new(in resolution, 0, new SharpEmu.Libs.Gpu.Buffers.ResourceSlotIdentifier(1, 1))];
            var rendering = new RenderingState { Samples = 1, ColorAttachmentCount = 1 };
            rendering.ColorAttachments[0] = new RenderingAttachment(
                default, GpuImageLayout.ColorAttachmentOptimal, GpuFormat.R8G8B8A8Unorm, 0, 0, 0, 0, false, false, false, false, false);
            return ShaderPipelineCache.BuildGraphicsDescription(
                colors, default, programs.VertexInput, programs.PixelInput, banks.Context, in rendering, GpuPrimitiveTopology.TriangleList, false, false,
                programs.Vertex, programs.Pixel, GpuSampleCount.Count1Bit).StaticParameters;
        }

        Assert.Equal(Parameters(false, 5), Parameters(false, 9));
        Assert.NotEqual(Parameters(true, 5), Parameters(true, 9));
        Assert.NotEqual(Parameters(true, 5), Parameters(false, 5));
    }

    [Fact]
    public void ColorTarget_WithoutAPixelStage_IsNotWritten()
    {
        var banks = Banks();
        banks.Context.RenderTargetMask = 0xF;
        var programs = Programs();
        var resolution = new ColorTargetResolution(
            default, 0x1000, 0x10000, new GpuExtent2D(64, 64), 0, 0, 1, ColorComponentMap.Identity, false, false, default);
        ColorTargetState[] colors = [new(in resolution, 0, new SharpEmu.Libs.Gpu.Buffers.ResourceSlotIdentifier(1, 1))];
        var rendering = new RenderingState { Samples = 1, ColorAttachmentCount = 1 };
        rendering.ColorAttachments[0] = new RenderingAttachment(
            default, GpuImageLayout.ColorAttachmentOptimal, GpuFormat.R8G8B8A8Unorm, 0, 0, 0, 0, false, false, false, false, false);

        var description = ShaderPipelineCache.BuildGraphicsDescription(
            colors, default, programs.VertexInput, null, banks.Context, in rendering, GpuPrimitiveTopology.TriangleList, false, false,
            programs.Vertex, programs.Pixel, GpuSampleCount.Count1Bit);

        Assert.Equal(0u, description.StaticParameters.GetColorMask(0));
    }

    [Fact]
    public void NoAttachments_AtAnUnsupportedSampleCount_AreFatal()
    {
        var banks = Banks();
        banks.Context.AntialiasingConfig.SampleCountLog2 = 3;
        var programs = Programs();
        var rendering = new RenderingState { Samples = 1 };

        var fatal = Assert.Throws<SchedulerFatalException>(() => ShaderPipelineCache.BuildGraphicsDescription(
            [], default, programs.VertexInput, programs.PixelInput, banks.Context, in rendering, GpuPrimitiveTopology.TriangleList, false, false,
            programs.Vertex, programs.Pixel, GpuSampleCount.Count1Bit | GpuSampleCount.Count4Bit));

        Assert.Contains("samples=8", fatal.Message);
    }

    [Theory]
    [InlineData(true, 2, true)]
    [InlineData(false, 2, false)]
    [InlineData(true, 0, false)]
    public void SampleShading_IsDerivedFromThePixelInputsAndTheSampleCount(bool pixelSampleShading, byte sampleCountLog2, bool expected)
    {
        var banks = Banks();
        banks.Context.AntialiasingConfig.SampleCountLog2 = sampleCountLog2;
        var programs = Programs();
        var pixelInput = new PixelInputInfo { InputCount = 1, SampleShading = pixelSampleShading, Stage = programs.PixelInput.Stage };
        var rendering = new RenderingState { Samples = 1 };

        var description = ShaderPipelineCache.BuildGraphicsDescription(
            [], default, programs.VertexInput, pixelInput, banks.Context, in rendering, GpuPrimitiveTopology.TriangleList, false, false,
            programs.Vertex, programs.Pixel, GpuSampleCount.Count1Bit | GpuSampleCount.Count4Bit);

        Assert.Equal(expected, description.StaticParameters.SampleShadingEnable);
    }

    // A geometry-stage draw on a backend with mesh pipelines: the merged program at the
    // export address, the GS program address register and the GS user-data address written.
    private static RegisterBanks GeometryStageBanks(ulong exportAddress, ulong geometryUserDataAddress)
    {
        var banks = Banks();
        banks.Context.ShaderStages = 0x00C02030;
        banks.Context.ShaderInterface.MaxOutputPerSubgroup = 0x40;
        banks.Context.ShaderInterface.GeometryMaxVerticesOut = 3;
        banks.Context.ShaderInterface.GeometryOutputPrimitiveType = 2;
        banks.Shader.Vertex.ExportAddress = exportAddress;
        banks.Shader.Vertex.GeometryAddress = exportAddress;
        banks.Shader.Vertex.GeometryUserDataAddress = geometryUserDataAddress;
        return banks;
    }

    private static GraphicsPrograms PrepareVertexOnly(ShaderPipelineCache cache, RegisterBanks banks) =>
        cache.GetGraphicsPrograms(banks.Shader.Vertex, banks.Shader.Pixel, banks.Context.ShaderInterface, banks.Context,
            Enumerable.Repeat(ColorComponentMap.Identity, 8).ToArray(), pixelActive: false, depthBound: false, depthAspectBound: false);

    [Fact]
    public void FusedGeometryProgram_ReadsItsUserDataThroughTheGeometryUserDataAddress()
    {
        var guest = new PipelineTestGuest();
        guest.Compiler.SupportsPrimitiveShaderMeshPipelines = true;
        var entry = PipelineTestGuest.MemoryBase + 0x1000;
        var continuation = PipelineTestGuest.MemoryBase + 0x1100;
        var userDataTable = PipelineTestGuest.MemoryBase + 0x9000;
        var data = PipelineTestGuest.MemoryBase + 0xA000;

        // The front half jumps to the back half; the back half loads its V# through s[0:1],
        // fetches with it and synchronizes the subgroup, so it is a workgroup program.
        guest.RegisterFusedProgram(
            entry, PipelineTestGuest.MemoryBase + 0x8000, [0xBF800000, 0xBE802000],
            continuation, PipelineTestGuest.MemoryBase + 0x8100,
            [0xF4080200, 0xFA000000, 0xE00C2000, 0x80020401, 0xBF8A0000, 0xBF810000],
            entryUserDataAddress: PipelineTestGuest.MemoryBase + 0xB000);
        guest.WriteWords(userDataTable, PipelineTestGuest.BufferDescriptor(data, 16, 4, Format32x4Float));
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);

        var programs = PrepareVertexOnly(cache, GeometryStageBanks(entry, userDataTable));

        Assert.True(programs.Available);
        var request = Assert.Single(guest.Compiler.Requests);
        Assert.NotNull(request.PrimitiveShader);

        // The user data starts at s0 with the pointer, and s2..s7 stay system scalars.
        Assert.Equal(0u, request.UserDataBase);
        Assert.Equal(0xFCUL, request.ExcludedUserDataRegisters);
        var userData = programs.VertexInput.Stage.Resources.UserData;
        Assert.Equal((uint)userDataTable, userData[0]);
        Assert.Equal((uint)(userDataTable >> 32), userData[1]);

        // The V# behind the pointer was planned, so the fetch has a buffer binding.
        Assert.NotEmpty(request.Resources.Info.Buffers);
    }

    [Fact]
    public void UnfusedPrimitiveShader_KeepsItsUserDataAtS8()
    {
        var guest = new PipelineTestGuest();
        guest.Compiler.SupportsPrimitiveShaderMeshPipelines = true;
        var program = PipelineTestGuest.MemoryBase + 0x1000;
        guest.RegisterProgram(program, PipelineTestGuest.MemoryBase + 0x8000, [0xBF8A0000, 0xBF810000],
            userDataAddress: PipelineTestGuest.MemoryBase + 0xB000);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);

        Assert.True(PrepareVertexOnly(cache, GeometryStageBanks(program, PipelineTestGuest.MemoryBase + 0x9000)).Available);

        // The geometry program address no longer keeps the draw off the merged stage.
        var request = Assert.Single(guest.Compiler.Requests);
        Assert.NotNull(request.PrimitiveShader);
        Assert.Equal(8u, request.UserDataBase);
        Assert.Equal(0UL, request.ExcludedUserDataRegisters);
    }

    [Fact]
    public void GeometryStageProgram_KeepsThePerVertexStateWithoutMeshPipelines()
    {
        var guest = new PipelineTestGuest();
        var program = PipelineTestGuest.MemoryBase + 0x1000;
        guest.RegisterProgram(program, PipelineTestGuest.MemoryBase + 0x8000, [0xBF8A0000, 0xBF810000],
            userDataAddress: PipelineTestGuest.MemoryBase + 0xB000);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);

        Assert.True(PrepareVertexOnly(cache, GeometryStageBanks(program, PipelineTestGuest.MemoryBase + 0x9000)).Available);

        var request = Assert.Single(guest.Compiler.Requests);
        Assert.Null(request.PrimitiveShader);
        Assert.Equal(8u, request.UserDataBase);
    }

    [Fact]
    public void MissingPrograms_AreFatal()
    {
        var banks = Banks();
        var programs = Programs();
        var rendering = new RenderingState { Samples = 1 };

        var noVertex = Assert.Throws<SchedulerFatalException>(() => ShaderPipelineCache.BuildGraphicsDescription(
            [], default, programs.VertexInput, programs.PixelInput, banks.Context, in rendering, GpuPrimitiveTopology.TriangleList, false, false,
            default, programs.Pixel, GpuSampleCount.Count1Bit));
        Assert.Contains("no vertex program", noVertex.Message);

        var noPixel = Assert.Throws<SchedulerFatalException>(() => ShaderPipelineCache.BuildGraphicsDescription(
            [], default, programs.VertexInput, programs.PixelInput, banks.Context, in rendering, GpuPrimitiveTopology.TriangleList, false, false,
            programs.Vertex, default, GpuSampleCount.Count1Bit));
        Assert.Contains("without a pixel program", noPixel.Message);
    }
}
