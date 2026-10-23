// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Pipelines;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

public sealed class VertexFetchSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Size = 32;
    private static long Counter(MetalRenderHost host, string name) =>
        (long)typeof(MetalRenderHost).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(host)!;

    [Fact]
    public void InstanceBiasChangesAcrossReusedPipelinePreservePixels()
    {
        var previous = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_VERTEX_FETCH");
        try
        {
            (uint[][] Pixels, long Fixed)? Replay(bool native)
            {
                Environment.SetEnvironmentVariable("SHARPEMU_METAL_V2_VERTEX_FETCH", native ? "1" : "0");
                using var rig = CreateRig();
                if (rig is null) return null;
                uint[] code =
                [
                    0xE0382000, 0x80020005, // buffer_load_dwordx4 v[0:3], v5, s[8:11], 0 idxen
                    0x4A081010,             // v_add_nc_u32 v4, s16, v8
                    0xE0382000, 0x80030A04, // buffer_load_dwordx4 v[10:13], v4, s[12:15], 0 idxen
                    0xBF8C3F70,
                    0xF80008CF, 0x03020100,
                    0xF800020F, 0x0D0C0B0A,
                    0xBF810000,
                ];
                var meshUser = rig.UploadMesh(SceneDraws.Quad(0.5f, 1, 1, 1));
                var colors = rig.Memory.Upload<float>([1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]);
                var targets = new[] { rig.ColorTarget(Size, Size), rig.ColorTarget(Size, Size) };
                var pm4 = new ScenePm4();
                for (var i = 0; i < 2; i++)
                {
                    uint[] user = [.. meshUser.Take(4), .. Descriptors.Buffer(colors, 16, 4), (uint)(1 - i)];
                    new SceneState(Size, Size).ColorTarget(0, targets[i], Size, Size)
                        .VertexShader(rig.Program(code), user).PixelShader(rig.Program(ScenePrograms.InterpolatedColor), []).Emit(pm4);
                    pm4.NumInstances(3).DrawAuto(6);
                }
                rig.Submit(pm4);
                Assert.Empty(rig.Fatals);
                return (targets.Select(target => rig.ReadImage<uint>(target, Size, Size)).ToArray(),
                    Counter(rig.Host, "_fixedFunctionFetchDraws"));
            }
            var shader = Replay(false);
            if (shader is null) return;
            var native = Replay(true)!.Value;
            Assert.Equal(shader.Value.Pixels[0], native.Pixels[0]);
            Assert.Equal(shader.Value.Pixels[1], native.Pixels[1]);
            Assert.All(native.Pixels[0], pixel => Assert.Equal(0xFFFF0000u, pixel));
            Assert.All(native.Pixels[1], pixel => Assert.Equal(0xFF00FF00u, pixel));
            Assert.Equal(0, shader.Value.Fixed);
            Assert.True(native.Fixed >= 2);
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPEMU_METAL_V2_VERTEX_FETCH", previous);
            MetalRenderHost.ApplyTranslationSwitches();
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NativeFetchAndPositionOnlyDepthPrepassMatchShaderPixelsAndDepth(bool indexed)
    {
        var fetchSwitch = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_VERTEX_FETCH");
        var positionSwitch = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_POSITION_ONLY_VERTEX");
        try
        {
            (uint[] Pixels, float[] Depth, long Fixed, long Position)? Replay(bool native)
            {
                Environment.SetEnvironmentVariable("SHARPEMU_METAL_V2_VERTEX_FETCH", native ? "1" : "0");
                Environment.SetEnvironmentVariable("SHARPEMU_METAL_V2_POSITION_ONLY_VERTEX", native ? "1" : "0");
                using var rig = CreateRig();
                if (rig is null) return null;
                var d = new DepthRig(rig, Size, Size, zFormat: 3);
                var pm4 = d.Pm4;
                var mesh = DepthRig.Plane(0.25f, 0.125f, 0.25f, 0.25f, 0.5f, 1f);
                // Full 16-bit domain is allocated: indexed native fetch is proven
                // without reading index contents or trusting a vertex-count hint.
                var positions = rig.Memory.Allocate(65536 * 16, 256);
                var parameters = rig.Memory.Allocate(65536 * 16, 256);
                for (var i = 0; i < mesh.Length; i++)
                {
                    var v = mesh[i];
                    rig.Memory.Write(positions + (ulong)(i * 16), System.Runtime.InteropServices.MemoryMarshal.AsBytes(new float[] { v.X, v.Y, v.Z, v.W }.AsSpan()));
                    rig.Memory.Write(parameters + (ulong)(i * 16), System.Runtime.InteropServices.MemoryMarshal.AsBytes(new float[] { v.R, v.G, v.B, v.A }.AsSpan()));
                }
                uint[] user = [.. Descriptors.Buffer(positions, 16, 65536), .. Descriptors.Buffer(parameters, 16, 65536)];
                var vertex = rig.Program(ScenePrograms.FetchVertex);
                uint[] nullPixel = [0xF8001890, 0u, 0xBF810000]; // exp null off, off, off, off done vm
                void Draw(SceneState state, uint[] pixel)
                {
                    state.VertexShader(vertex, user).PixelShader(rig.Program(pixel), []).Emit(pm4);
                    pm4.NumInstances(1);
                    if (indexed)
                    {
                        var indices = rig.Memory.Upload<ushort>([0, 1, 2, 3, 4, 5]);
                        pm4.IndexType(0).DrawIndexed(indices, 6, 6);
                    }
                    else pm4.DrawAuto(6);
                }
                var prepass = d.State().DepthControl(true, true, CompareFunc.Always);
                prepass.Context[Cx.CbTargetMask] = 0;
                prepass.Context[Cx.CbShaderMask] = 0;
                Draw(prepass, nullPixel);
                // EQUAL makes even a one-bit position/depth mismatch visible.
                Draw(d.State().DepthControl(true, false, CompareFunc.Equal), ScenePrograms.InterpolatedColor);
                var depth = d.QueueDepthReadback();
                var pixels = d.QueueColorReadback();
                rig.Submit(pm4);
                Assert.Empty(rig.Fatals);
                return (pixels(), depth(), Counter(rig.Host, "_fixedFunctionFetchDraws"), Counter(rig.Host, "_positionOnlyVertexDraws"));
            }
            var shader = Replay(false);
            if (shader is null) return;
            var native = Replay(true)!.Value;
            Assert.Equal(shader.Value.Pixels, native.Pixels);
            Assert.Equal(shader.Value.Depth.Select(BitConverter.SingleToUInt32Bits), native.Depth.Select(BitConverter.SingleToUInt32Bits));
            Assert.All(native.Pixels, pixel => Assert.NotEqual(0u, pixel));
            Assert.Contains(native.Depth, z => z > 0.25f && z < 0.625f);
            Assert.Equal(0, shader.Value.Fixed);
            Assert.Equal(0, shader.Value.Position);
            Assert.True(native.Fixed >= 2, $"fixed draws={native.Fixed}");
            Assert.True(native.Position >= 1, $"position-only draws={native.Position}");
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPEMU_METAL_V2_VERTEX_FETCH", fetchSwitch);
            Environment.SetEnvironmentVariable("SHARPEMU_METAL_V2_POSITION_ONLY_VERTEX", positionSwitch);
            MetalRenderHost.ApplyTranslationSwitches();
        }
    }
}
