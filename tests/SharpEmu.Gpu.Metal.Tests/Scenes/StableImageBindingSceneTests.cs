// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections;
using System.Reflection;
using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// Stable image bindings on the Metal host (MetalStableImageLimits, upstream 7cf0ac568): a pixel program that samples
/// two image slots is drawn with both slots holding one texture, then two different textures in both orders. Within the
/// device's budget the program binds one texture per slot, so all three draws use one specialization (one pixel module)
/// and every pixel reads its own slot's texture; with SHARPEMU_METAL_STABLE_IMAGE_BINDINGS=0 the equal slots alias one
/// argument-buffer element and the program is specialized once per alias partition, with the same pixels.
/// </summary>
public sealed class StableImageBindingSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Rgba8 = 56;
    private const uint Size = 8;

    /// <summary>
    /// MRT0 = (T#0.r, T#0.g, T#1.r, T#1.g): T#0 in s[0:7] and T#1 in s[8:15] sampled through the S# in s[16:19] at the
    /// interpolated PARAM0.xy (twenty user SGPRs, so the primitive mask is s20).
    /// </summary>
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

    [Theory]
    [InlineData(true, 1)]
    [InlineData(false, 2)]
    public void OneProgramSamplesEachSlotsOwnTextureWhetherTheSlotsHoldOneTextureOrTwo(bool stable, int pixelModules)
    {
        var saved = MetalStableImageLimits.Enabled;
        MetalStableImageLimits.Enabled = stable;
        try
        {
            using var rig = CreateRig();
            if (rig is null) return;
            Assert.Equal(stable, ((SharpEmu.Libs.Gpu.Pipelines.IShaderPipelineHost)rig.Host).GraphicsImageDescriptorLimits.Resources != 0);
            var red = SolidTexture(rig, 0xFF0000FF);
            var green = SolidTexture(rig, 0xFF00FF00);
            var sampler = Descriptors.Sampler();
            var pm4 = new ScenePm4();
            (uint[] Slot0, uint[] Slot1, float[] Expected)[] draws =
            [
                (red, red, [1, 0, 1, 0]),
                (red, green, [1, 0, 0, 1]),
                (green, red, [0, 1, 1, 0]),
                (green, green, [0, 1, 0, 1]),
            ];
            var targets = new List<ulong>();
            foreach (var (slot0, slot1, _) in draws)
            {
                var target = rig.ColorTarget(Size, Size, 16, 0xFF);
                targets.Add(target);
                var state = new SceneState(Size, Size).ColorTarget(0, target, Size, Size, format: 14, numberType: 7);
                rig.DrawMesh(pm4, state, Quad(), TwoSlotProgram, [.. slot0, .. slot1, .. sampler]);
            }

            rig.Submit(pm4);
            Assert.Empty(rig.Fatals);
            for (var index = 0; index < draws.Length; index++)
            {
                var image = MemoryMarshal.Cast<byte, float>(rig.ReadImage(targets[index], Size, Size, 16)).ToArray();
                for (var pixel = 0; pixel < Size * Size; pixel++)
                {
                    for (var channel = 0; channel < 4; channel++)
                    {
                        Assert.True(MathF.Abs(image[(pixel * 4) + channel] - draws[index].Expected[channel]) < 0.002f,
                            $"draw {index} pixel {pixel} channel {channel}: {image[(pixel * 4) + channel]}, " +
                            $"expected {draws[index].Expected[channel]}");
                    }
                }
            }

            Assert.Equal(pixelModules, PixelModules(rig.Host));
        }
        finally
        {
            MetalStableImageLimits.Enabled = saved;
        }
    }

    // The live host answers one budget for compute and graphics, the one the prewarm passes (MetalStableImageLimits.Of
    // of the same device), and the prewarm stamp (so the translation cache stamp) holds it: entries made with the switch
    // on are not taken with it off.
    [Fact]
    public void TheHostAndThePrewarmAnswerOneBudgetAndTheStampHoldsIt()
    {
        var saved = MetalStableImageLimits.Enabled;
        try
        {
            using var rig = CreateRig();
            if (rig is null) return;
            var device = (nint)typeof(MetalRenderHost).GetField("_device", BindingFlags.Instance | BindingFlags.NonPublic)!
                .GetValue(rig.Host)!;
            var host = (SharpEmu.Libs.Gpu.Pipelines.IShaderPipelineHost)rig.Host;
            var compiler = new MetalV2Backend();
            MetalStableImageLimits.Enabled = true;
            var on = MetalShaderPrewarm.Stamp(compiler, device);
            var budget = MetalStableImageLimits.Of(device);
            Output.WriteLine(MetalStableImageLimits.StampPart(budget));
            Assert.Equal(MetalStableImageLimits.TexturesPerStage, budget.SampledImages);
            Assert.InRange(budget.Samplers, MetalStableImageLimits.SamplersPerStageApple7, MetalStableImageLimits.SamplersPerStageApple9);
            Assert.Equal(budget, host.ComputeImageDescriptorLimits);
            Assert.Equal(budget, host.GraphicsImageDescriptorLimits);
            MetalStableImageLimits.Enabled = false;
            Assert.Equal(default, host.ComputeImageDescriptorLimits);
            Assert.Equal(default, host.GraphicsImageDescriptorLimits);
            Assert.NotEqual(on, MetalShaderPrewarm.Stamp(compiler, device));
        }
        finally
        {
            MetalStableImageLimits.Enabled = saved;
        }
    }

    // A linear RGBA8 texture of one colour.
    private static uint[] SolidTexture(SceneRig rig, uint abgr)
    {
        var rows = new byte[Size * Size * 4];
        for (var texel = 0; texel < Size * Size; texel++)
        {
            BitConverter.TryWriteBytes(rows.AsSpan(texel * 4), abgr);
        }

        return Descriptors.Texture(rig.LinearImage(Size, Size, 4, rows), Rgba8, Size, Size);
    }

    // The whole target, PARAM0 = (0.5, 0.5, 0, 1) everywhere.
    private static SceneVertex[] Quad()
    {
        static SceneVertex V(float x, float y) => SceneVertex.At(x, y, 0.5f, 0.5f, 0.5f, 0f, 1f);
        return [V(-1, -1), V(1, -1), V(-1, 1), V(-1, 1), V(1, -1), V(1, 1)];
    }

    // The pixel modules the host created: one per specialization the program cache compiled.
    private static int PixelModules(MetalRenderHost host)
    {
        var modules = (IList)typeof(MetalRenderHost).GetField("_modules", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(host)!;
        lock (modules)
        {
            return modules.Cast<object>().Count(module =>
                (ShaderStage)module.GetType().GetField("Stage")!.GetValue(module)! == ShaderStage.Pixel);
        }
    }
}
