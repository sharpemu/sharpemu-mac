// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

// Stable image bindings as the Metal host answers them (MetalStableImageLimits: one per-stage budget for compute and
// graphics, textures 1M, samplers 996 on Apple7/8, buffers unlimited), through the shared paths the Metal prewarm
// takes: a Metal-shaped host compiles one MSL specialization where compact bindings give one per alias partition, and
// what a live session records is found again by the prewarm under the same budget, with the switch on (the budget) and
// off (no budget). A record made under the other budget is not translated as if current.
[Collection(SchedulingStateCollection.Name)]
public sealed class MetalStableImagePrewarmTests : IDisposable
{
    private const ulong CodeAddress = PipelineTestGuest.MemoryBase + 0x1_0000;
    private const ulong HeaderAddress = PipelineTestGuest.MemoryBase + 0x8000;
    private const ulong BufferAddress = PipelineTestGuest.MemoryBase + 0x4_0000;
    private const uint ImageType2D = 9;

    // The Metal host's answer on an Apple7/Apple8 device, SHARPEMU_METAL_STABLE_IMAGE_BINDINGS on.
    internal static readonly StableImageDescriptorLimits MetalLimits = new(1_000_000, 1_000_000, 996, uint.MaxValue, 1_000_000);

    // MRT0 = (T#0.r, T#0.g, T#1.r, T#1.g): T#0 s[0:7], T#1 s[8:15], S# s[16:19] at PARAM0.xy (s20 is the primitive mask).
    private static readonly uint[] TwoSlotProgram =
    [
        0xBEFC0314,                // s_mov_b32 m0, s20
        0xC8080000,                // v_interp_p1_f32_e32 v2, v0, attr0.x
        0xC8090001,                // v_interp_p2_f32_e32 v2, v1, attr0.x
        0xC80C0100,                // v_interp_p1_f32_e32 v3, v0, attr0.y
        0xC80D0101,                // v_interp_p2_f32_e32 v3, v1, attr0.y
        0xF0800F08, 0x00800402,    // image_sample v[4:7], v[2:3], s[0:7], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xF0800F08, 0x00820802,    // image_sample v[8:11], v[2:3], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C3F70,                // s_waitcnt vmcnt(0)
        0xF800180F, 0x09080504,    // exp mrt0 v4, v5, v8, v9 done vm
        0xBF810000,                // s_endpgm
    ];

    private readonly string _directory = Path.Combine(Path.GetTempPath(), "SharpEmuTests", Guid.NewGuid().ToString("N"));
    private readonly FatalScope _fatal = new();

    public void Dispose()
    {
        _fatal.Dispose();
        if (Directory.Exists(_directory))
        {
            Directory.Delete(_directory, recursive: true);
        }
    }

    private static byte[] Msl(ShaderCompileRequest request)
    {
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        return Encoding.UTF8.GetBytes(shader.Source);
    }

    // T# slot 0 at A, slot 1 at A (one descriptor in both slots) or at B, then an S# of zeros and the primitive mask.
    private static uint[] UserData(bool sameTexture)
    {
        var userData = new uint[21];
        WriteTexture(userData, 0, 0x20);
        WriteTexture(userData, 8, sameTexture ? 0x20u : 0x40u);
        return userData;
    }

    private static void WriteTexture(uint[] userData, int at, uint baseAddress)
    {
        userData[at] = baseAddress;
        userData[at + 1] = 71u << 20;                                               // 8_8_8_8 unorm
        userData[at + 2] = (63u << 14) | 63u;                                       // 64 x 64
        userData[at + 3] = (ImageType2D << 28) | Gen5ShaderTranslator.IdentityImageDstSelect;
    }

    // A live session's pixel draws, each partition in turn; the MSL of every specialization it compiled.
    private List<string> DrawPixel(StableImageDescriptorLimits limits, params bool[] sameTexture)
    {
        var drawn = new List<string>();
        var guest = new PipelineTestGuest(request =>
        {
            var msl = Msl(request);
            drawn.Add(Encoding.UTF8.GetString(msl));
            return msl;
        });
        guest.Host.GraphicsImageDescriptorLimits = limits;
        guest.Host.ComputeImageDescriptorLimits = limits;
        using (var list = GraphicsPrewarmList.Open(_directory))
        {
            Assert.NotNull(list);
            guest.Host.GraphicsPrewarm = list;
            guest.RegisterProgram(CodeAddress, HeaderAddress, TwoSlotProgram);
            foreach (var same in sameTexture)
            {
                var cursor = 0u;
                guest.Programs.GetOrCompile(guest.Source(CodeAddress, ShaderStage.Pixel, UserData(same)),
                    new StageCompileOptions { PixelInfo = new PixelInputInfo() }, ref cursor, out _);
            }
        }

        return drawn;
    }

    // Two slots holding one texture, then two textures: one specialization within the Metal budget, two without.
    [Theory]
    [InlineData(true, 1)]
    [InlineData(false, 2)]
    public void AMetalShapedHostCompilesOneSpecializationForEveryAliasPartition(bool stable, int compiles)
    {
        var drawn = DrawPixel(stable ? MetalLimits : default, true, false);
        Assert.Equal(compiles, drawn.Count);
    }

    // The graphics prewarm replays each recorded specialization to the MSL the draw compiled, under the budget the
    // session answered (the switch on: the Metal budget; off: none).
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TheGraphicsPrewarmFindsWhatTheSessionRecorded(bool stable)
    {
        var limits = stable ? MetalLimits : default;
        var drawn = DrawPixel(limits, true, false);
        using var list = GraphicsPrewarmList.Open(_directory);
        Assert.NotNull(list);
        var items = list.Loaded();
        Assert.Equal(drawn.Count, items.Count);
        var replayed = new List<string>();
        foreach (var item in items)
        {
            var record = GraphicsPrewarmList.Decode(item);
            Assert.Equal(limits, record.Host.GraphicsImageLimits);
            Assert.Equal(stable, record.Specialization!.StableImageBindings);
            var compiler = new FakeShaderCompiler(Msl);
            Assert.True(ShaderPrewarmCompiler.TryCompileGraphicsPrewarm(item, compiler, limits, out var compiled, out var error,
                out var stale), error);
            Assert.False(stale);
            replayed.Add(Encoding.UTF8.GetString(((FakeCompiledShader)compiled!).Spirv));
        }

        Assert.Equal(drawn.Order(StringComparer.Ordinal), replayed.Order(StringComparer.Ordinal));
    }

    // A record the session made under the other setting is skipped as stale, never translated.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TheGraphicsPrewarmSkipsARecordMadeUnderTheOtherBudget(bool stable)
    {
        _ = DrawPixel(stable ? MetalLimits : default, true);
        using var list = GraphicsPrewarmList.Open(_directory);
        Assert.NotNull(list);
        var item = Assert.Single(list.Loaded());
        var compiler = new FakeShaderCompiler(Msl);
        Assert.False(ShaderPrewarmCompiler.TryCompileGraphicsPrewarm(item, compiler, stable ? default : MetalLimits,
            out _, out var error, out var stale));
        Assert.True(stale, error);
        Assert.Empty(compiler.Requests);
    }

    // The static precompile predicts the plan's binding policy, so its specialization applies to the plan.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void APredictedSpecializationFollowsThePlansBindingPolicy(bool stable)
    {
        _ = DrawPixel(stable ? MetalLimits : default, true);
        using var list = GraphicsPrewarmList.Open(_directory);
        Assert.NotNull(list);
        var record = GraphicsPrewarmList.Decode(Assert.Single(list.Loaded())) with { Specialization = null };
        Assert.True(ShaderProgramCache.TryPredictSpecialization(record, out var predicted, out var error), error);
        Assert.Equal(stable, predicted!.StableImageBindings);
        Assert.True(ShaderCorpus.TryBuildRequest(record with { Specialization = predicted }, out var request, out error), error);
        Assert.NotNull(request);
    }

    // The compute prewarm call the Metal prewarm makes (ShaderPrewarmCompiler.TryCompilePrewarm with the host's budget)
    // finds the specialization a live dispatch recorded, and refuses one of the other policy.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TheComputePrewarmFindsWhatTheSessionRecorded(bool stable)
    {
        var limits = stable ? MetalLimits : default;
        byte[] runtime;
        using (var list = ShaderPrewarmList.Open(_directory))
        {
            Assert.NotNull(list);
            var guest = new PipelineTestGuest(Msl);
            guest.Host.ShaderPrewarm = list;
            guest.Host.ComputeImageDescriptorLimits = limits;
            guest.Host.GraphicsImageDescriptorLimits = limits;
            guest.RegisterProgram(CodeAddress, HeaderAddress, ImageBindingTestProgram.Code());
            ImageBindingTestProgram.WriteDescriptors(guest, BufferAddress, ImageBindingTestProgram.Partitions[0]);
            var cursor = 0u;
            guest.Programs.GetOrCompile(guest.Source(CodeAddress, ShaderStage.Compute,
                    [unchecked((uint)BufferAddress), (uint)(BufferAddress >> 32)]),
                PipelineTestGuest.ComputeOptions(), ref cursor, out _);
            runtime = Assert.Single(guest.Compiler.Shaders).Spirv;
        }

        using var reloaded = ShaderPrewarmList.Open(_directory);
        Assert.NotNull(reloaded);
        var (record, code) = Assert.Single(reloaded.LoadedComputes());
        Assert.Equal(stable, record.Specialization.StableImageBindings);
        var compiler = new FakeShaderCompiler(Msl);
        Assert.True(ShaderPrewarmCompiler.TryCompilePrewarm(record, code, compiler, false, true, false,
            out var compiled, out _, out var error, stableImageLimits: limits), error);
        Assert.Equal(runtime, Assert.IsType<FakeCompiledShader>(compiled).Spirv);
        var other = new FakeShaderCompiler(Msl);
        Assert.False(ShaderPrewarmCompiler.TryCompilePrewarm(record, code, other, false, true, false,
            out _, out _, out error, stableImageLimits: stable ? default : MetalLimits));
        Assert.Contains("image binding policy", error);
        Assert.Empty(other.Requests);
    }

    // The Metal compute prewarm (MetalShaderPrewarm.TryCompileCompute, then NoteComputeRefusal) counts a record made under
    // the other image binding policy as stale beside graphics_stale, never as a failure, and translates nothing for it.
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TheMetalComputePrewarmCountsARecordOfTheOtherPolicyAsStale(bool stable)
    {
        var limits = stable ? MetalLimits : default;
        using (var list = ShaderPrewarmList.Open(_directory))
        {
            Assert.NotNull(list);
            var guest = new PipelineTestGuest(Msl);
            guest.Host.ShaderPrewarm = list;
            guest.Host.ComputeImageDescriptorLimits = limits;
            guest.Host.GraphicsImageDescriptorLimits = limits;
            guest.RegisterProgram(CodeAddress, HeaderAddress, ImageBindingTestProgram.Code());
            ImageBindingTestProgram.WriteDescriptors(guest, BufferAddress, ImageBindingTestProgram.Partitions[0]);
            var cursor = 0u;
            guest.Programs.GetOrCompile(guest.Source(CodeAddress, ShaderStage.Compute,
                    [unchecked((uint)BufferAddress), (uint)(BufferAddress >> 32)]),
                PipelineTestGuest.ComputeOptions(), ref cursor, out _);
        }

        using var reloaded = ShaderPrewarmList.Open(_directory);
        Assert.NotNull(reloaded);
        var (record, code) = Assert.Single(reloaded.LoadedComputes());
        var other = new FakeShaderCompiler(Msl);
        Assert.False(SharpEmu.Gpu.Metal.V2.MetalShaderPrewarm.TryCompileCompute(record, code, other, stable ? default : MetalLimits,
            out _, out var error));
        Assert.Empty(other.Requests);
        var stale = SharpEmu.Gpu.Metal.V2.MetalShaderPrewarm.ComputeStale;
        var failed = SharpEmu.Gpu.Metal.V2.MetalShaderPrewarm.ComputeFailed;
        SharpEmu.Gpu.Metal.V2.MetalShaderPrewarm.NoteComputeRefusal(record, error);
        Assert.Equal(stale + 1, SharpEmu.Gpu.Metal.V2.MetalShaderPrewarm.ComputeStale);
        Assert.Equal(failed, SharpEmu.Gpu.Metal.V2.MetalShaderPrewarm.ComputeFailed);

        // Any other refusal is still a failure.
        SharpEmu.Gpu.Metal.V2.MetalShaderPrewarm.NoteComputeRefusal(record, "the record has no specialization");
        Assert.Equal(stale + 1, SharpEmu.Gpu.Metal.V2.MetalShaderPrewarm.ComputeStale);
        Assert.Equal(failed + 1, SharpEmu.Gpu.Metal.V2.MetalShaderPrewarm.ComputeFailed);
    }
}
