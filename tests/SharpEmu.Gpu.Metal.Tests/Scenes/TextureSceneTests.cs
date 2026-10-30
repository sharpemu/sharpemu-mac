// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Numerics;
using System.Runtime.InteropServices;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// Texturing scenes: every texel the shader reads is written by the test into guest memory
/// and every expected value is computed here from the hardware rules (RDNA2 ISA, addrlib
/// linear layout, BC and packed-float format specifications). Results are rendered to an
/// RGBA32F target so filtering and format decode are compared as floats.
/// </summary>
public sealed class TextureSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Rgba8 = 56;

    // ---- mip chain colours (RGBA8, one solid colour per level) ----
    private static readonly uint[] LevelColours =
    [
        0xFF0000FF, 0xFF00FF00, 0xFFFF0000, 0xFF00FFFF, 0xFFFF00FF, 0xFFFFFF00, 0xFFFFFFFF,
    ];

    private static float[] Unpack(uint abgr) =>
        [(abgr & 0xFF) / 255f, ((abgr >> 8) & 0xFF) / 255f, ((abgr >> 16) & 0xFF) / 255f, ((abgr >> 24) & 0xFF) / 255f];

    private static float[] Lerp(float[] a, float[] b, float t) =>
        [a[0] + ((b[0] - a[0]) * t), a[1] + ((b[1] - a[1]) * t), a[2] + ((b[2] - a[2]) * t), a[3] + ((b[3] - a[3]) * t)];

    /// <summary>
    /// A linear RGBA8 mip chain laid out as addrlib's gfx10 linear surface does: levels from the
    /// smallest to the base, each row padded to 64 texels (256 bytes), no height padding.
    /// </summary>
    private static ulong MipChain(SceneRig rig, uint size, out uint levels)
    {
        levels = (uint)BitOperations.Log2(size) + 1;
        var offsets = new ulong[levels];
        ulong total = 0;
        for (var level = (int)levels - 1; level >= 0; level--)
        {
            offsets[level] = total;
            var dim = Math.Max(size >> level, 1);
            total += (ulong)((dim + 63) / 64 * 64) * dim * 4;
        }

        var bytes = new byte[total];
        for (var level = 0; level < levels; level++)
        {
            var dim = Math.Max(size >> level, 1);
            var pitch = (dim + 63) / 64 * 64;
            for (var y = 0; y < dim; y++)
            {
                for (var x = 0; x < dim; x++)
                {
                    BitConverter.TryWriteBytes(bytes.AsSpan((int)(offsets[level] + ((ulong)((y * pitch) + x) * 4))), LevelColours[level]);
                }
            }
        }

        var address = rig.Memory.Allocate(total, 0x10000);
        rig.Memory.Write(address, bytes);
        return address;
    }

    // ---- draw plumbing ----

    private readonly record struct Pending(ulong Target, uint Width, uint Height);

    private static SceneVertex[] PixelQuad(uint targetWidth, uint targetHeight, int x0, int y0, int x1, int y1,
        Vector4 topLeft, Vector4 topRight, Vector4 bottomLeft, Vector4 bottomRight)
    {
        var left = -1f + (2f * x0 / targetWidth);
        var right = -1f + (2f * x1 / targetWidth);
        var top = 1f - (2f * y0 / targetHeight);
        var bottom = 1f - (2f * y1 / targetHeight);
        static SceneVertex V(float x, float y, Vector4 p) => SceneVertex.At(x, y, 0.5f, p.X, p.Y, p.Z, p.W);
        return
        [
            V(left, bottom, bottomLeft), V(right, bottom, bottomRight), V(left, top, topLeft),
            V(left, top, topLeft), V(right, bottom, bottomRight), V(right, top, topRight),
        ];
    }

    /// <summary>PARAM0 = (u, v, z, 1) mapped linearly over the pixel rectangle.</summary>
    private static SceneVertex[] UvQuad(uint targetWidth, uint targetHeight, int x0, int y0, int x1, int y1,
        float u0, float v0, float u1, float v1, float z = 0f) =>
        PixelQuad(targetWidth, targetHeight, x0, y0, x1, y1,
            new Vector4(u0, v0, z, 1), new Vector4(u1, v0, z, 1), new Vector4(u0, v1, z, 1), new Vector4(u1, v1, z, 1));

    private static Pending Draw(SceneRig rig, ScenePm4 pm4, uint width, uint height, uint[] program, uint[] texture, uint[] sampler,
        SceneVertex[] quad, float c0 = 0, float c1 = 0, float c2 = 0, float c3 = 0)
    {
        var target = rig.ColorTarget(width, height, 16, 0xFF);
        var state = new SceneState(width, height).ColorTarget(0, target, width, height, format: 14, numberType: 7);
        uint[] userData = [.. texture, .. sampler, SceneState.F(c0), SceneState.F(c1), SceneState.F(c2), SceneState.F(c3)];
        rig.DrawMesh(pm4, state, quad, program, userData);
        return new Pending(target, width, height);
    }

    private static float[] Read(SceneRig rig, Pending pending) =>
        MemoryMarshal.Cast<byte, float>(rig.ReadImage(pending.Target, pending.Width, pending.Height, 16)).ToArray();

    private static float[] Pixel(float[] image, uint width, int x, int y) => image.AsSpan((int)(((y * width) + x) * 4), 4).ToArray();

    private sealed class Check
    {
        private readonly List<string> _errors = [];
        public int Count { get; private set; }

        public void Colour(string what, float[] actual, float[] expected, float tolerance)
        {
            Count++;
            for (var channel = 0; channel < 4; channel++)
            {
                if (!(MathF.Abs(actual[channel] - expected[channel]) <= tolerance + (1e-5f * MathF.Abs(expected[channel]))))
                {
                    if (_errors.Count < 12)
                    {
                        _errors.Add($"{what}: got ({actual[0]:G6},{actual[1]:G6},{actual[2]:G6},{actual[3]:G6}) expected ({expected[0]:G6},{expected[1]:G6},{expected[2]:G6},{expected[3]:G6}) tol {tolerance}");
                    }
                    else
                    {
                        _errors.Add(string.Empty);
                    }

                    return;
                }
            }
        }

        public void Done()
        {
            var failures = _errors.Count;
            Assert.True(failures == 0, $"{failures} of {Count} checks failed:\n{string.Join("\n", _errors.Where(e => e.Length > 0))}");
        }
    }

    /// <summary>Every pixel of the n x n region at the target's top left must equal <paramref name="expected"/>.</summary>
    private static void Region(Check check, string what, float[] image, uint width, int n, float[] expected, float tolerance)
    {
        for (var y = 0; y < n; y++)
        {
            for (var x = 0; x < n; x++)
            {
                check.Colour($"{what} px({x},{y})", Pixel(image, width, x, y), expected, tolerance);
            }
        }
    }

    /// <summary>
    /// The chain's T#. Word 5 MAX_MIP stays at the surface's last level (6) whatever the view's LAST_LEVEL is: the emulator derives
    /// the chain layout from it.
    /// </summary>
    private static uint[] MipTexture(ulong address, uint baseLevel = 0, uint lastLevel = 6) =>
        Descriptors.Texture(address, Rgba8, 64, 64, baseLevel: baseLevel, lastLevel: lastLevel, maxMip: 6);

    // ======================================================================
    // 1. Mip selection
    // ======================================================================

    [Fact]
    public void ExplicitLodSelectsEachMipLevelWithPointMipFilter()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var chain = MipChain(rig, 64, out _);
        var pm4 = new ScenePm4();
        var draws = new List<Pending>();
        for (var lod = 0; lod <= 6; lod++)
        {
            draws.Add(Draw(rig, pm4, 16, 16, SampleL, MipTexture(chain), Descriptors.Sampler(mipFilter: 1),
                UvQuad(16, 16, 0, 0, 16, 16, 0, 0, 1, 1), lod));
        }

        rig.Submit(pm4);
        var check = new Check();
        for (var lod = 0; lod <= 6; lod++)
        {
            Region(check, $"image_sample_l lod {lod}", Read(rig, draws[lod]), 16, 16, Unpack(LevelColours[lod]), 0.003f);
        }

        check.Done();
    }

    [Fact]
    public void ExplicitLodRoundsForPointMipFilterAndBlendsForLinear()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var chain = MipChain(rig, 64, out _);
        var pm4 = new ScenePm4();
        var cases = new List<(string Name, Pending Draw, float[] Expected, float Tolerance)>();
        // Point mip filter: nearest level (lod rounded to the nearest integer).
        foreach (var (lod, level) in new[] { (1.3f, 1), (1.7f, 2), (3.2f, 3), (5.8f, 6) })
        {
            cases.Add(($"point lod {lod}", Draw(rig, pm4, 16, 16, SampleL, MipTexture(chain), Descriptors.Sampler(mipFilter: 1),
                UvQuad(16, 16, 0, 0, 16, 16, 0, 0, 1, 1), lod), Unpack(LevelColours[level]), 0.003f));
        }

        // Linear mip filter: a blend of the two neighbouring levels by the fractional lod (hardware weights are quantised).
        foreach (var lod in new[] { 0.25f, 1.5f, 2.75f, 4.4f })
        {
            var level = (int)MathF.Floor(lod);
            cases.Add(($"linear lod {lod}", Draw(rig, pm4, 16, 16, SampleL, MipTexture(chain), Descriptors.Sampler(mipFilter: 2, minFilter: 1, magFilter: 1),
                UvQuad(16, 16, 0, 0, 16, 16, 0, 0, 1, 1), lod),
                Lerp(Unpack(LevelColours[level]), Unpack(LevelColours[level + 1]), lod - level), 0.03f));
        }

        rig.Submit(pm4);
        var check = new Check();
        foreach (var (name, draw, expected, tolerance) in cases)
        {
            Region(check, name, Read(rig, draw), 16, 16, expected, tolerance);
        }

        check.Done();
    }

    // FORCE_DEGAMMA converts the filtered colour from sRGB to linear: halfway between the red and the green level, a
    // 0.5 of each decodes to 0.214 (decoding the texels before the filter would leave 0.5, as 0 and 1 decode to themselves).
    [Fact]
    public void ForceDegammaSamplerDecodesTheFilteredColour()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var chain = MipChain(rig, 64, out _);
        var pm4 = new ScenePm4();
        var decoded = Draw(rig, pm4, 16, 16, SampleL, MipTexture(chain),
            Descriptors.Sampler(mipFilter: 2, minFilter: 1, magFilter: 1, forceDegamma: true), UvQuad(16, 16, 0, 0, 16, 16, 0, 0, 1, 1), 0.5f);
        var plain = Draw(rig, pm4, 16, 16, SampleL, MipTexture(chain),
            Descriptors.Sampler(mipFilter: 2, minFilter: 1, magFilter: 1), UvQuad(16, 16, 0, 0, 16, 16, 0, 0, 1, 1), 0.5f);
        rig.Submit(pm4);
        var half = MathF.Pow((0.5f + 0.055f) / 1.055f, 2.4f);
        var check = new Check();
        Region(check, "force-degamma sampler", Read(rig, decoded), 16, 16, [half, half, 0f, 1f], 0.03f);
        Region(check, "ordinary sampler", Read(rig, plain), 16, 16, [0.5f, 0.5f, 0f, 1f], 0.03f);
        check.Done();
    }

    // A min/max FILTER_MODE sampler (S# word 0, bits 30:29) returns per component the minimum or maximum of the texels its
    // bilinear filter weighs. Ghost of Tsushima's volumetric fog reads the farthest depth of a tile so; a blend of the sky
    // with the ground in front of it stopped the froxel march short of the ground, and the froxels it skipped showed as
    // white rims along every silhouette.
    [Theory]
    [InlineData(1u, "min")]
    [InlineData(2u, "max")]
    public void AReductionSamplerReturnsTheMinimumOrMaximumOfItsFootprint(uint filterMode, string name)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var texels = new byte[8 * 8 * 4];
        for (var j = 0; j < 8; j++)
        {
            for (var i = 0; i < 8; i++)
            {
                BitConverter.TryWriteBytes(texels.AsSpan(((j * 8) + i) * 4), GatherTexel(i, j));
            }
        }

        var image = rig.LinearImage(8, 8, 4, texels);
        uint[] texture = Descriptors.Texture(image, Rgba8, 8, 8);
        // u * 8 - 0.5 = x + 0.25: the footprint of pixel x is texels x and x + 1, each weighing.
        var pm4 = new ScenePm4();
        var reduced = Draw(rig, pm4, 16, 16, SampleL, texture, Descriptors.Sampler(magFilter: 1, minFilter: 1, filterMode: filterMode),
            UvQuad(16, 16, 0, 0, 8, 8, 0.25f / 8, 0.25f / 8, 1 + (0.25f / 8), 1 + (0.25f / 8)));
        rig.Submit(pm4);
        var pixels = Read(rig, reduced);
        var check = new Check();
        float[] C(int i, int j) => Unpack(GatherTexel(Math.Min(i, 7), Math.Min(j, 7)));
        for (var y = 0; y < 8; y++)
        {
            for (var x = 0; x < 8; x++)
            {
                float[][] footprint = [C(x, y), C(x + 1, y), C(x, y + 1), C(x + 1, y + 1)];
                var expected = Enumerable.Range(0, 4)
                    .Select(component => filterMode == 1 ? footprint.Min(texel => texel[component]) : footprint.Max(texel => texel[component]))
                    .ToArray();
                check.Colour($"{name} px({x},{y})", Pixel(pixels, 16, x, y), expected, 0.003f);
            }
        }

        check.Done();
    }

    // With a linear mip filter the reduction spans both levels: halfway between the red and the green level the minimum
    // is black and the maximum yellow, where a blend gives half of each.
    [Theory]
    [InlineData(1u, 0f, 0f)]
    [InlineData(2u, 1f, 1f)]
    public void AReductionSamplerWithALinearMipFilterReducesBothLevels(uint filterMode, float red, float green)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var chain = MipChain(rig, 64, out _);
        var pm4 = new ScenePm4();
        var reduced = Draw(rig, pm4, 16, 16, SampleL, MipTexture(chain),
            Descriptors.Sampler(mipFilter: 2, minFilter: 1, magFilter: 1, filterMode: filterMode), UvQuad(16, 16, 0, 0, 16, 16, 0, 0, 1, 1), 0.5f);
        var level = Draw(rig, pm4, 16, 16, SampleL, MipTexture(chain),
            Descriptors.Sampler(mipFilter: 2, minFilter: 1, magFilter: 1, filterMode: filterMode), UvQuad(16, 16, 0, 0, 16, 16, 0, 0, 1, 1), 1f);
        rig.Submit(pm4);
        var check = new Check();
        Region(check, "between levels", Read(rig, reduced), 16, 16, [red, green, 0f, 1f], 0.003f);
        Region(check, "on a level", Read(rig, level), 16, 16, Unpack(LevelColours[1]), 0.003f);
        check.Done();
    }

    [Fact]
    public void ScreenDerivativesPickTheLevelFromTheQuadSize()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var chain = MipChain(rig, 64, out _);
        var pm4 = new ScenePm4();
        var draws = new List<Pending>();
        // An n x n pixel quad over the whole 64x64 texture minifies by 64/n: lod = log2(64/n).
        for (var lod = 0; lod <= 6; lod++)
        {
            var n = 64 >> lod;
            draws.Add(Draw(rig, pm4, 64, 64, Sample, MipTexture(chain), Descriptors.Sampler(mipFilter: 1),
                UvQuad(64, 64, 0, 0, n, n, 0, 0, 1, 1)));
        }

        rig.Submit(pm4);
        var check = new Check();
        for (var lod = 0; lod <= 6; lod++)
        {
            Region(check, $"implicit lod {lod} (quad {64 >> lod}px)", Read(rig, draws[lod]), 64, 64 >> lod, Unpack(LevelColours[lod]), 0.003f);
        }

        check.Done();
    }

    [Fact]
    public void ScreenDerivativesBlendLevelsWithLinearMipFilter()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var chain = MipChain(rig, 64, out _);
        var pm4 = new ScenePm4();
        var cases = new List<(int N, Pending Draw)>();
        foreach (var n in new[] { 48, 24, 12 })
        {
            cases.Add((n, Draw(rig, pm4, 64, 64, Sample, MipTexture(chain), Descriptors.Sampler(mipFilter: 2, minFilter: 1, magFilter: 1),
                UvQuad(64, 64, 0, 0, n, n, 0, 0, 1, 1))));
        }

        rig.Submit(pm4);
        var check = new Check();
        foreach (var (n, draw) in cases)
        {
            var lod = MathF.Log2(64f / n);
            var level = (int)MathF.Floor(lod);
            // Hardware derives lod from a quantised approximation, so the blend weight tolerance is 0.08.
            Region(check, $"quad {n}px lod {lod:F3}", Read(rig, draw), 64, n,
                Lerp(Unpack(LevelColours[level]), Unpack(LevelColours[level + 1]), lod - level), 0.08f);
        }

        check.Done();
    }

    [Fact]
    public void InstructionLodBiasShiftsTheDerivativeLevel()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var chain = MipChain(rig, 64, out _);
        var pm4 = new ScenePm4();
        var cases = new List<(float Bias, int N, int Level, Pending Draw)>();
        // The 16px quad is at lod 2; the 64px quad at lod 0.
        foreach (var (bias, n, level) in new[] { (1f, 16, 3), (-1f, 16, 1), (2f, 16, 4), (3f, 64, 3), (-2f, 64, 0) })
        {
            cases.Add((bias, n, level, Draw(rig, pm4, 64, 64, SampleB, MipTexture(chain), Descriptors.Sampler(mipFilter: 1),
                UvQuad(64, 64, 0, 0, n, n, 0, 0, 1, 1), bias)));
        }

        rig.Submit(pm4);
        var check = new Check();
        foreach (var (bias, n, level, draw) in cases)
        {
            Region(check, $"image_sample_b bias {bias} quad {n}px", Read(rig, draw), 64, n, Unpack(LevelColours[level]), 0.003f);
        }

        check.Done();
    }

    [Fact]
    public void SamplerLodBiasShiftsTheLevel()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var chain = MipChain(rig, 64, out _);
        var pm4 = new ScenePm4();
        var cases = new List<(string Name, int Level, Pending Draw)>();
        // Derivative lod 2 (16px quad) plus S# bias.
        foreach (var (bias, level) in new[] { (1f, 3), (-1f, 1), (2.0f, 4) })
        {
            cases.Add(($"implicit lod 2 + S# bias {bias}", level, Draw(rig, pm4, 64, 64, Sample, MipTexture(chain), Descriptors.Sampler(mipFilter: 1, lodBias: bias),
                UvQuad(64, 64, 0, 0, 16, 16, 0, 0, 1, 1))));
        }

        // Explicit lod 1 plus S# bias 2.
        cases.Add(("explicit lod 1 + S# bias 2", 3, Draw(rig, pm4, 16, 16, SampleL, MipTexture(chain), Descriptors.Sampler(mipFilter: 1, lodBias: 2f),
            UvQuad(16, 16, 0, 0, 16, 16, 0, 0, 1, 1), 1f)));
        // Gradient lod 1 (2 texels per pixel) plus S# bias 2.
        cases.Add(("gradient lod 1 + S# bias 2", 3, Draw(rig, pm4, 16, 16, SampleD, MipTexture(chain), Descriptors.Sampler(mipFilter: 1, lodBias: 2f),
            UvQuad(16, 16, 0, 0, 16, 16, 0, 0, 1, 1), 2f / 64, 0, 0, 2f / 64)));
        rig.Submit(pm4);
        var check = new Check();
        foreach (var (name, level, draw) in cases)
        {
            var width = draw.Width;
            Region(check, name, Read(rig, draw), width, 16, Unpack(LevelColours[level]), 0.003f);
        }

        check.Done();
    }

    [Fact]
    public void SamplerMinAndMaxLodClampTheLevel()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var chain = MipChain(rig, 64, out _);
        var pm4 = new ScenePm4();
        var cases = new List<(string Name, int Level, Pending Draw)>();
        cases.Add(("implicit lod 2, MIN_LOD 3", 3, Draw(rig, pm4, 64, 64, Sample, MipTexture(chain), Descriptors.Sampler(mipFilter: 1, minLod: 3),
            UvQuad(64, 64, 0, 0, 16, 16, 0, 0, 1, 1))));
        cases.Add(("implicit lod 2, MAX_LOD 1", 1, Draw(rig, pm4, 64, 64, Sample, MipTexture(chain), Descriptors.Sampler(mipFilter: 1, maxLod: 1),
            UvQuad(64, 64, 0, 0, 16, 16, 0, 0, 1, 1))));
        cases.Add(("explicit lod 0, MIN_LOD 4", 4, Draw(rig, pm4, 16, 16, SampleL, MipTexture(chain), Descriptors.Sampler(mipFilter: 1, minLod: 4),
            UvQuad(16, 16, 0, 0, 16, 16, 0, 0, 1, 1), 0f)));
        cases.Add(("explicit lod 6, MAX_LOD 3", 3, Draw(rig, pm4, 16, 16, SampleL, MipTexture(chain), Descriptors.Sampler(mipFilter: 1, maxLod: 3),
            UvQuad(16, 16, 0, 0, 16, 16, 0, 0, 1, 1), 6f)));
        rig.Submit(pm4);
        var check = new Check();
        foreach (var (name, level, draw) in cases)
        {
            Region(check, name, Read(rig, draw), draw.Width, 16, Unpack(LevelColours[level]), 0.003f);
        }

        check.Done();
    }

    [Fact]
    public void TextureBaseAndLastLevelWindowTheChain()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var chain = MipChain(rig, 64, out _);
        var pm4 = new ScenePm4();
        var cases = new List<(string Name, int Level, Pending Draw)>();
        // BASE_LEVEL 1, LAST_LEVEL 4: lod is relative to the base level and clamped to the last level.
        foreach (var (lod, level) in new[] { (0f, 1), (2f, 3), (6f, 4) })
        {
            cases.Add(($"BASE 1 LAST 4, lod {lod}", level, Draw(rig, pm4, 16, 16, SampleL, MipTexture(chain, 1, 4), Descriptors.Sampler(mipFilter: 1),
                UvQuad(16, 16, 0, 0, 16, 16, 0, 0, 1, 1), lod)));
        }

        // Derivative driven: the 64px quad would be lod 0 -> the base level (1).
        cases.Add(("BASE 2 LAST 6, implicit quad 64px", 2, Draw(rig, pm4, 64, 64, Sample, MipTexture(chain, 2, 6), Descriptors.Sampler(mipFilter: 1),
            UvQuad(64, 64, 0, 0, 16, 16, 0, 0, 1, 1))));
        rig.Submit(pm4);
        var check = new Check();
        foreach (var (name, level, draw) in cases)
        {
            Region(check, name, Read(rig, draw), draw.Width, 16, Unpack(LevelColours[level]), 0.003f);
        }

        check.Done();
    }

    [Fact]
    public void ExplicitDerivativesDriveTheLevel()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var chain = MipChain(rig, 64, out _);
        var pm4 = new ScenePm4();
        var cases = new List<(string Name, int Level, Pending Draw)>();
        // Derivatives are in normalised coordinates per pixel; the texture is 64 texels wide, so
        // d = k/64 texels is lod log2(k). The larger of the two axis gradients wins (no anisotropy).
        foreach (var (name, dudx, dvdx, dudy, dvdy, level) in new (string, float, float, float, float, int)[]
        {
            ("1 texel/px", 1f / 64, 0, 0, 1f / 64, 0),
            ("2 texel/px", 2f / 64, 0, 0, 2f / 64, 1),
            ("8 texel/px", 8f / 64, 0, 0, 8f / 64, 3),
            ("x 4, y 16", 4f / 64, 0, 0, 16f / 64, 4),
            ("rotated 8", 0, 8f / 64, 8f / 64, 0, 3),
        })
        {
            cases.Add((name, level, Draw(rig, pm4, 16, 16, SampleD, MipTexture(chain), Descriptors.Sampler(mipFilter: 1),
                UvQuad(16, 16, 0, 0, 16, 16, 0, 0, 1, 1), dudx, dvdx, dudy, dvdy)));
        }

        rig.Submit(pm4);
        var check = new Check();
        foreach (var (name, level, draw) in cases)
        {
            Region(check, $"image_sample_d {name}", Read(rig, draw), 16, 16, Unpack(LevelColours[level]), 0.003f);
        }

        check.Done();
    }

    [Fact]
    public void AnisotropicFilteringStaysBetweenIsotropicAndFullyAnisotropicLevels()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var chain = MipChain(rig, 64, out _);
        var pm4 = new ScenePm4();
        // 32 x 4 pixel quad over the whole texture: 2 texels/px in x, 16 texels/px in y (ratio 8).
        var off = Draw(rig, pm4, 64, 64, Sample, MipTexture(chain), Descriptors.Sampler(mipFilter: 1, minFilter: 3, magFilter: 3, maxAniso: 0),
            UvQuad(64, 64, 0, 0, 32, 4, 0, 0, 1, 1));
        var ratio8 = Draw(rig, pm4, 64, 64, Sample, MipTexture(chain), Descriptors.Sampler(mipFilter: 1, minFilter: 3, magFilter: 3, maxAniso: 3),
            UvQuad(64, 64, 0, 0, 32, 4, 0, 0, 1, 1));
        rig.Submit(pm4);
        var check = new Check();
        var offImage = Read(rig, off);
        var anisoImage = Read(rig, ratio8);
        for (var y = 0; y < 4; y++)
        {
            for (var x = 0; x < 32; x++)
            {
                // Isotropic: lod = log2(16) = 4. Fully anisotropic (ratio 8): lod = log2(16/8) = 1. Every tap lands on a solid level, so the
                // result must be one of the solid colours of levels 1..4 (nothing from the base level or the coarse tail).
                check.Colour($"aniso off px({x},{y})", Pixel(offImage, 64, x, y), Unpack(LevelColours[4]), 0.003f);
                var actual = Pixel(anisoImage, 64, x, y);
                var matches = Enumerable.Range(1, 4).Any(level =>
                    Unpack(LevelColours[level]).Zip(actual, (e, a) => MathF.Abs(e - a) <= 0.03f).All(ok => ok));
                check.Colour($"aniso ratio 8 px({x},{y}) in levels 1..4", matches ? Unpack(LevelColours[1]) : actual, matches ? Unpack(LevelColours[1]) : [-1, -1, -1, -1], 0.003f);
            }
        }

        check.Done();
    }

    // ======================================================================
    // 2. Cube maps
    // ======================================================================

    private static uint CubeTexel(int face, int x, int y) =>
        0xFF000000u | (uint)((100 + (y * 40)) << 16) | (uint)((20 + (x * 60)) << 8) | (uint)(20 + (face * 40));

    /// <summary>The ISA's cube face, face s/t in [0,1] for a direction (V_CUBEID/SC/TC/MA tables), or null near a face or texel boundary.</summary>
    private static (int Face, int X, int Y)? CubeOracle(Vector3 d, int size)
    {
        var ax = MathF.Abs(d.X);
        var ay = MathF.Abs(d.Y);
        var az = MathF.Abs(d.Z);
        var sorted = new[] { ax, ay, az }.OrderByDescending(v => v).ToArray();
        if ((sorted[0] - sorted[1]) / sorted[0] < 0.03f)
        {
            return null;
        }

        int face;
        float sc, tc, ma;
        if (az >= ax && az >= ay)
        {
            face = d.Z < 0 ? 5 : 4;
            sc = d.Z < 0 ? -d.X : d.X;
            tc = -d.Y;
            ma = az;
        }
        else if (ay >= ax)
        {
            face = d.Y < 0 ? 3 : 2;
            sc = d.X;
            tc = d.Y < 0 ? -d.Z : d.Z;
            ma = ay;
        }
        else
        {
            face = d.X < 0 ? 1 : 0;
            sc = d.X < 0 ? d.Z : -d.Z;
            tc = -d.Y;
            ma = ax;
        }

        var s = ((sc / ma) + 1f) * 0.5f * size;
        var t = ((tc / ma) + 1f) * 0.5f * size;
        if (MathF.Abs(s - MathF.Round(s)) < 0.04f || MathF.Abs(t - MathF.Round(t)) < 0.04f)
        {
            return null;
        }

        return (face, Math.Clamp((int)MathF.Floor(s), 0, size - 1), Math.Clamp((int)MathF.Floor(t), 0, size - 1));
    }

    [Fact]
    public void CubeMapDirectionsSelectTheRightFaceAndTexel()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        const int size = 4;
        const int sliceBytes = 256 * size;
        var bytes = new byte[6 * sliceBytes];
        for (var face = 0; face < 6; face++)
        {
            for (var y = 0; y < size; y++)
            {
                for (var x = 0; x < size; x++)
                {
                    BitConverter.TryWriteBytes(bytes.AsSpan((face * sliceBytes) + (y * 256) + (x * 4)), CubeTexel(face, x, y));
                }
            }
        }

        var cube = rig.Memory.Allocate((ulong)bytes.Length, 0x10000);
        rig.Memory.Write(cube, bytes);
        uint[] texture = Descriptors.Texture(cube, Rgba8, size, size, type: 11, depthOrLastLayer: 5);

        var sweeps = new List<(Vector3 From, Vector3 To)>
        {
            (new(1, -0.8f, -0.7f), new(1, 0.8f, 0.7f)),
            (new(-1, -0.7f, 0.8f), new(-1, 0.7f, -0.8f)),
            (new(-0.8f, 1, -0.7f), new(0.8f, 1, 0.7f)),
            (new(0.7f, -1, -0.8f), new(-0.7f, -1, 0.8f)),
            (new(-0.7f, 0.8f, 1), new(0.7f, -0.8f, 1)),
            (new(0.8f, 0.7f, -1), new(-0.8f, -0.7f, -1)),
            // Sweeps that cross face edges and corners.
            (new(1, 0.3f, 0.1f), new(0.1f, 1, 0.2f)),
            (new(1, 0.2f, 0.4f), new(0.3f, -0.1f, 1)),
            (new(-1, -0.4f, 0.3f), new(-0.2f, -1, 0.1f)),
            (new(0.9f, 0.95f, 1), new(-0.95f, -0.9f, -1)),
            (new(-1, 0.9f, -0.95f), new(1, -0.95f, 0.9f)),
            (new(0.2f, 1, -0.6f), new(0.1f, -0.6f, -1)),
        };
        var pm4 = new ScenePm4();
        var draws = new List<Pending>();
        foreach (var (from, to) in sweeps)
        {
            draws.Add(Draw(rig, pm4, 32, 4, SampleCube, texture, Descriptors.Sampler(),
                PixelQuad(32, 4, 0, 0, 32, 4, new Vector4(from, 1), new Vector4(to, 1), new Vector4(from, 1), new Vector4(to, 1))));
        }

        rig.Submit(pm4);
        var check = new Check();
        var skipped = 0;
        for (var sweep = 0; sweep < sweeps.Count; sweep++)
        {
            var image = Read(rig, draws[sweep]);
            for (var x = 0; x < 32; x++)
            {
                var d = Vector3.Lerp(sweeps[sweep].From, sweeps[sweep].To, (x + 0.5f) / 32f);
                var expected = CubeOracle(d, size);
                if (expected is null)
                {
                    skipped++;
                    continue;
                }

                var (face, tx, ty) = expected.Value;
                for (var y = 0; y < 4; y++)
                {
                    check.Colour($"sweep {sweep} dir ({d.X:F3},{d.Y:F3},{d.Z:F3}) face {face} texel ({tx},{ty}) px({x},{y})",
                        Pixel(image, 32, x, y), Unpack(CubeTexel(face, tx, ty)), 0.003f);
                }
            }
        }

        Output.WriteLine($"cube: {check.Count} checks, {skipped} ambiguous pixels skipped");
        Assert.True(check.Count > 600);
        check.Done();
    }

    // ======================================================================
    // 3. Arrays and 3D
    // ======================================================================

    private static uint SliceTexel(int slice, int x, int y) =>
        0xFF000000u | (uint)((20 + (y * 60)) << 16) | (uint)((20 + (x * 60)) << 8) | (uint)(20 + (slice * 60));

    private static ulong SliceStack(SceneRig rig, int slices)
    {
        const int sliceBytes = 256 * 4;
        var bytes = new byte[slices * sliceBytes];
        for (var slice = 0; slice < slices; slice++)
        {
            for (var y = 0; y < 4; y++)
            {
                for (var x = 0; x < 4; x++)
                {
                    BitConverter.TryWriteBytes(bytes.AsSpan((slice * sliceBytes) + (y * 256) + (x * 4)), SliceTexel(slice, x, y));
                }
            }
        }

        var address = rig.Memory.Allocate((ulong)bytes.Length, 0x10000);
        rig.Memory.Write(address, bytes);
        return address;
    }

    [Fact]
    public void ArraySliceIndexSelectsTheSliceAndRoundsToNearest()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var stack = SliceStack(rig, 4);
        uint[] texture = Descriptors.Texture(stack, Rgba8, 4, 4, type: 13, depthOrLastLayer: 3);
        var pm4 = new ScenePm4();
        var cases = new List<(float Z, int Slice, Pending Draw)>();
        foreach (var (z, slice) in new[] { (0f, 0), (1f, 1), (2f, 2), (3f, 3), (1.4f, 1), (1.6f, 2) })
        {
            cases.Add((z, slice, Draw(rig, pm4, 16, 16, SampleArray, texture, Descriptors.Sampler(), UvQuad(16, 16, 0, 0, 16, 16, 0, 0, 1, 1, z))));
        }

        rig.Submit(pm4);
        var check = new Check();
        foreach (var (z, slice, draw) in cases)
        {
            var image = Read(rig, draw);
            for (var y = 0; y < 16; y++)
            {
                for (var x = 0; x < 16; x++)
                {
                    check.Colour($"array z={z} px({x},{y})", Pixel(image, 16, x, y), Unpack(SliceTexel(slice, x / 4, y / 4)), 0.003f);
                }
            }
        }

        check.Done();
    }

    [Fact]
    public void VolumeTextureRCoordinateSelectsThePointFilteredSlice()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var stack = SliceStack(rig, 4);
        uint[] texture = Descriptors.Texture(stack, Rgba8, 4, 4, type: 10, depthOrLastLayer: 3);
        var pm4 = new ScenePm4();
        var cases = new List<(int Slice, Pending Draw)>();
        for (var slice = 0; slice < 4; slice++)
        {
            cases.Add((slice, Draw(rig, pm4, 16, 16, Sample3D, texture, Descriptors.Sampler(), UvQuad(16, 16, 0, 0, 16, 16, 0, 0, 1, 1, (slice + 0.5f) / 4f))));
        }

        rig.Submit(pm4);
        var check = new Check();
        foreach (var (slice, draw) in cases)
        {
            var image = Read(rig, draw);
            for (var y = 0; y < 16; y++)
            {
                for (var x = 0; x < 16; x++)
                {
                    check.Colour($"volume slice {slice} px({x},{y})", Pixel(image, 16, x, y), Unpack(SliceTexel(slice, x / 4, y / 4)), 0.003f);
                }
            }
        }

        check.Done();
    }

    // ======================================================================
    // 4. Gathers
    // ======================================================================

    private static uint GatherTexel(int i, int j) =>
        (uint)(i * 32 + 16) | ((uint)(j * 32 + 16) << 8) | ((uint)(((i * 8) + j) * 4 + 1) << 16) | ((uint)(255 - (((i * 8) + j) * 4) - 1) << 24);

    [Theory]
    [InlineData(0, "R")]
    [InlineData(1, "G")]
    [InlineData(2, "B")]
    [InlineData(3, "A")]
    public void Gather4ReturnsTheSelectedComponentOfTheBilinearFootprint(int component, string name)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var texels = new byte[8 * 8 * 4];
        for (var j = 0; j < 8; j++)
        {
            for (var i = 0; i < 8; i++)
            {
                BitConverter.TryWriteBytes(texels.AsSpan(((j * 8) + i) * 4), GatherTexel(i, j));
            }
        }

        var image = rig.LinearImage(8, 8, 4, texels);
        uint[] texture = Descriptors.Texture(image, Rgba8, 8, 8);
        // u = (x + 0.75) / 8 => u * 8 - 0.5 = x + 0.25: footprint starts at texel x, clear of the texel-centre tie.
        var program = new[] { GatherR, GatherG, GatherB, GatherA }[component];
        var pm4 = new ScenePm4();
        var draw = Draw(rig, pm4, 16, 16, program, texture, Descriptors.Sampler(),
            UvQuad(16, 16, 0, 0, 8, 8, 0.25f / 8, 0.25f / 8, 1 + (0.25f / 8), 1 + (0.25f / 8)));
        rig.Submit(pm4);
        var pixels = Read(rig, draw);
        var check = new Check();
        float C(int i, int j) => Unpack(GatherTexel(Math.Min(i, 7), Math.Min(j, 7)))[component];
        for (var y = 0; y < 8; y++)
        {
            for (var x = 0; x < 8; x++)
            {
                // Documented order: x = (i0,j1), y = (i1,j1), z = (i1,j0), w = (i0,j0) with j growing downwards; out-of-range clamps.
                float[] expected = [C(x, y + 1), C(x + 1, y + 1), C(x + 1, y), C(x, y)];
                check.Colour($"gather4 {name} px({x},{y})", Pixel(pixels, 16, x, y), expected, 0.003f);
            }
        }

        check.Done();
    }

    // ======================================================================
    // 5. Formats
    // ======================================================================

    private sealed record FormatCase(uint Format, uint TexWidth, uint TexHeight, uint ElementBytes, uint ElementsX, uint ElementsY,
        byte[] Data, float[][] Expected, float Tolerance);

    private static float SrgbToLinear(float c) => c <= 0.04045f ? c / 12.92f : MathF.Pow((c + 0.055f) / 1.055f, 2.4f);

    private static float MiniFloat(uint bits, int mantissaBits)
    {
        var exponent = (int)(bits >> mantissaBits) & 31;
        var mantissa = bits & ((1u << mantissaBits) - 1);
        var scale = (float)(1 << mantissaBits);
        return exponent == 0 ? mantissa / scale * MathF.Pow(2, -14) : (1 + (mantissa / scale)) * MathF.Pow(2, exponent - 15);
    }

    private static FormatCase Uncompressed(uint format, uint bytesPerTexel, Func<Random, (byte[] Bytes, float[] Expected)> texel, float tolerance)
    {
        var random = new Random(7 + (int)format);
        var data = new byte[16 * bytesPerTexel];
        var expected = new float[16][];
        for (var index = 0; index < 16; index++)
        {
            var (bytes, value) = texel(random);
            bytes.CopyTo(data, index * (int)bytesPerTexel);
            expected[index] = value;
        }

        return new FormatCase(format, 4, 4, bytesPerTexel, 4, 4, data, expected, tolerance);
    }

    private static byte[] U32(uint value) => BitConverter.GetBytes(value);

    private static FormatCase BuildFormat(string name)
    {
        switch (name)
        {
            case "Rgba8Unorm":
                return Uncompressed(56, 4, r => { var v = (uint)r.NextInt64(); return (U32(v), Unpack(v)); }, 0.001f);
            case "Rgba8Srgb":
                return Uncompressed(130, 4, r =>
                {
                    var v = (uint)r.NextInt64();
                    var c = Unpack(v);
                    return (U32(v), [SrgbToLinear(c[0]), SrgbToLinear(c[1]), SrgbToLinear(c[2]), c[3]]);
                }, 0.003f);
            case "Rgb10A2Unorm":
                // IMG_FORMAT 50, 2_10_10_10_UNORM (mesa gfx10-rsrc.json, RDNA2 ISA table 47; PAL X10Y10Z10W2_Unorm): names list
                // the fields from the most significant bit down, so R is the low ten bits and A the top two. 44 is 10_10_10_2.
                return Uncompressed(50, 4, r =>
                {
                    var v = (uint)r.NextInt64();
                    return (U32(v), [(v & 1023) / 1023f, ((v >> 10) & 1023) / 1023f, ((v >> 20) & 1023) / 1023f, (v >> 30) / 3f]);
                }, 0.001f);
            case "R11G11B10Float":
                // IMG_FORMAT 36, 10_11_11_FLOAT (R11G11B10F; mesa maps PIPE_FORMAT_R11G11B10_FLOAT to it, PAL X11Y11Z10_Float):
                // R is bits 0..10, G bits 11..21, B bits 22..31.
                return Uncompressed(36, 4, r =>
                {
                    var rbits = (uint)r.Next(8, 22) << 6 | (uint)r.Next(64);
                    var gbits = (uint)r.Next(8, 22) << 6 | (uint)r.Next(64);
                    var bbits = (uint)r.Next(8, 22) << 5 | (uint)r.Next(32);
                    var v = rbits | (gbits << 11) | (bbits << 22);
                    return (U32(v), [MiniFloat(rbits, 6), MiniFloat(gbits, 6), MiniFloat(bbits, 5), 1f]);
                }, 1e-6f);
            case "Float11_11_10":
                // IMG_FORMAT 43, 11_11_10_FLOAT (PAL X10Y11Z11_Float): R is the low ten bits (5-bit mantissa), G bits 10..20,
                // B bits 21..31.
                return Uncompressed(43, 4, r =>
                {
                    var rbits = (uint)r.Next(8, 22) << 5 | (uint)r.Next(32);
                    var gbits = (uint)r.Next(8, 22) << 6 | (uint)r.Next(64);
                    var bbits = (uint)r.Next(8, 22) << 6 | (uint)r.Next(64);
                    var v = rbits | (gbits << 10) | (bbits << 21);
                    return (U32(v), [MiniFloat(rbits, 5), MiniFloat(gbits, 6), MiniFloat(bbits, 6), 1f]);
                }, 1e-6f);
            case "Rgba16Float":
                return Uncompressed(71, 8, r =>
                {
                    var values = Enumerable.Range(0, 4).Select(_ => (Half)((r.NextSingle() * 8f) - 4f)).ToArray();
                    var bytes = new byte[8];
                    for (var i = 0; i < 4; i++) { BitConverter.TryWriteBytes(bytes.AsSpan(i * 2), values[i]); }
                    return (bytes, values.Select(v => (float)v).ToArray());
                }, 1e-6f);
            case "Rg16Snorm":
                return Uncompressed(24, 4, r =>
                {
                    var a = (short)r.Next(short.MinValue, short.MaxValue + 1);
                    var b = (short)r.Next(short.MinValue, short.MaxValue + 1);
                    var bytes = new byte[4];
                    BitConverter.TryWriteBytes(bytes.AsSpan(0), a);
                    BitConverter.TryWriteBytes(bytes.AsSpan(2), b);
                    return (bytes, [Math.Max(a / 32767f, -1f), Math.Max(b / 32767f, -1f), 0f, 1f]);
                }, 2e-5f);
            case "R32Float":
                return Uncompressed(22, 4, r =>
                {
                    var v = (r.NextSingle() * 200f) - 100f;
                    return (BitConverter.GetBytes(v), [v, 0f, 0f, 1f]);
                }, 0f);
            case "R8Unorm":
                return Uncompressed(1, 1, r => { var v = (byte)r.Next(256); return ([v], [v / 255f, 0f, 0f, 1f]); }, 0.001f);
            case "Bc1":
                return BuildBc1();
            case "Bc3":
                return BuildBc3();
            case "Bc4":
                return BuildBc45(175, 1);
            case "Bc5":
                return BuildBc45(177, 2);
            case "Bc7Mode6":
                return BuildBc7Mode6();
            default:
                throw new ArgumentException(name);
        }
    }

    private static float[] Expand565(ushort value)
    {
        var r = (value >> 11) & 31;
        var g = (value >> 5) & 63;
        var b = value & 31;
        return [((r << 3) | (r >> 2)) / 255f, ((g << 2) | (g >> 4)) / 255f, ((b << 3) | (b >> 2)) / 255f];
    }

    /// <summary>Colour palette of a BC1 / BC3 colour block per the BC1 specification; index 3 of the three-colour mode is transparent black.</summary>
    private static float[][] Bc1Palette(ushort c0, ushort c1, bool forceFourColour)
    {
        var a = Expand565(c0);
        var b = Expand565(c1);
        float[] Mix(float wa, float wb) => [(a[0] * wa) + (b[0] * wb), (a[1] * wa) + (b[1] * wb), (a[2] * wa) + (b[2] * wb), 1f];
        if (c0 > c1 || forceFourColour)
        {
            return [[a[0], a[1], a[2], 1f], [b[0], b[1], b[2], 1f], Mix(2f / 3, 1f / 3), Mix(1f / 3, 2f / 3)];
        }

        return [[a[0], a[1], a[2], 1f], [b[0], b[1], b[2], 1f], Mix(0.5f, 0.5f), [0f, 0f, 0f, 0f]];
    }

    private static byte[] Bc1Block(ushort c0, ushort c1, int[] indices)
    {
        var bytes = new byte[8];
        BitConverter.TryWriteBytes(bytes.AsSpan(0), c0);
        BitConverter.TryWriteBytes(bytes.AsSpan(2), c1);
        uint bits = 0;
        for (var texel = 0; texel < 16; texel++)
        {
            bits |= (uint)indices[texel] << (texel * 2);
        }

        BitConverter.TryWriteBytes(bytes.AsSpan(4), bits);
        return bytes;
    }

    private static int[] PatternIndices(int seed, int range)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, 16).Select(i => i < range ? i : random.Next(range)).ToArray();
    }

    /// <summary>Assembles 2x2 blocks (8x8 texels) into the block-row data the image upload and expected texel arrays use.</summary>
    private static FormatCase BlockCase(uint format, uint blockBytes, byte[][] blocks, float[][][] blockTexels, float tolerance)
    {
        var data = new byte[4 * blockBytes];
        var expected = new float[64][];
        for (var block = 0; block < 4; block++)
        {
            blocks[block].CopyTo(data, block * (int)blockBytes);
            for (var texel = 0; texel < 16; texel++)
            {
                var x = ((block % 2) * 4) + (texel % 4);
                var y = ((block / 2) * 4) + (texel / 4);
                expected[(y * 8) + x] = blockTexels[block][texel];
            }
        }

        return new FormatCase(format, 8, 8, blockBytes, 2, 2, data, expected, tolerance);
    }

    private static FormatCase BuildBc1()
    {
        // 4-colour block, 3-colour block with transparent index 3, 4-colour block, and c0 == c1 (three-colour mode).
        ushort[][] endpoints = [[0xF800, 0x001F], [0x07E0, 0xFFE0], [0x8410, 0x4208], [0x1234, 0x1234]];
        var blocks = new byte[4][];
        var texels = new float[4][][];
        for (var block = 0; block < 4; block++)
        {
            var indices = PatternIndices(100 + block, 4);
            blocks[block] = Bc1Block(endpoints[block][0], endpoints[block][1], indices);
            var palette = Bc1Palette(endpoints[block][0], endpoints[block][1], false);
            texels[block] = indices.Select(i => palette[i]).ToArray();
        }

        // Block 0 (0xF800 > 0x001F) and 2 are four-colour; 1 (0x07E0 < 0xFFE0) and 3 (equal) are the 1-bit-alpha mode.
        return BlockCase(169, 8, blocks, texels, 0.012f);
    }

    private static (byte[] Block, float[] Values) Bc4Block(byte a0, byte a1, int[] indices)
    {
        var bytes = new byte[8];
        bytes[0] = a0;
        bytes[1] = a1;
        ulong bits = 0;
        for (var texel = 0; texel < 16; texel++)
        {
            bits |= (ulong)(uint)indices[texel] << (texel * 3);
        }

        for (var i = 0; i < 6; i++)
        {
            bytes[2 + i] = (byte)(bits >> (i * 8));
        }

        var palette = new float[8];
        palette[0] = a0 / 255f;
        palette[1] = a1 / 255f;
        if (a0 > a1)
        {
            for (var k = 2; k < 8; k++)
            {
                palette[k] = (((8 - k) * a0) + ((k - 1) * a1)) / 7f / 255f;
            }
        }
        else
        {
            for (var k = 2; k < 6; k++)
            {
                palette[k] = (((6 - k) * a0) + ((k - 1) * a1)) / 5f / 255f;
            }

            palette[6] = 0f;
            palette[7] = 1f;
        }

        return (bytes, indices.Select(i => palette[i]).ToArray());
    }

    private static FormatCase BuildBc3()
    {
        var blocks = new byte[4][];
        var texels = new float[4][][];
        // Alpha: 8-value mode (a0 > a1) and 6-value mode (a0 <= a1, with the explicit 0 and 1 entries).
        (byte, byte)[] alphas = [(250, 20), (30, 200), (10, 240), (180, 60)];
        ushort[][] colours = [[0xF800, 0x001F], [0xFFE0, 0x07E0], [0xFFFF, 0x0000], [0x7BEF, 0x39E7]];
        for (var block = 0; block < 4; block++)
        {
            var alphaIndices = PatternIndices(200 + block, 8);
            var (alphaBlock, alpha) = Bc4Block(alphas[block].Item1, alphas[block].Item2, alphaIndices);
            var colourIndices = PatternIndices(300 + block, 4);
            var palette = Bc1Palette(colours[block][0], colours[block][1], true);
            blocks[block] = [.. alphaBlock, .. Bc1Block(colours[block][0], colours[block][1], colourIndices)];
            texels[block] = Enumerable.Range(0, 16).Select(t =>
            {
                var c = palette[colourIndices[t]];
                return new[] { c[0], c[1], c[2], alpha[t] };
            }).ToArray();
        }

        return BlockCase(173, 16, blocks, texels, 0.012f);
    }

    private static FormatCase BuildBc45(uint format, int channels)
    {
        var blocks = new byte[4][];
        var texels = new float[4][][];
        (byte, byte)[] endpoints = [(240, 16), (40, 220), (255, 0), (7, 7)];
        (byte, byte)[] endpointsG = [(10, 250), (230, 30), (90, 180), (120, 60)];
        for (var block = 0; block < 4; block++)
        {
            var indicesR = PatternIndices(400 + block, 8);
            var (redBlock, red) = Bc4Block(endpoints[block].Item1, endpoints[block].Item2, indicesR);
            if (channels == 1)
            {
                blocks[block] = redBlock;
                texels[block] = red.Select(r => new[] { r, 0f, 0f, 1f }).ToArray();
            }
            else
            {
                var indicesG = PatternIndices(500 + block, 8);
                var (greenBlock, green) = Bc4Block(endpointsG[block].Item1, endpointsG[block].Item2, indicesG);
                blocks[block] = [.. redBlock, .. greenBlock];
                texels[block] = Enumerable.Range(0, 16).Select(t => new[] { red[t], green[t], 0f, 1f }).ToArray();
            }
        }

        return BlockCase(format, (uint)(8 * channels), blocks, texels, 0.008f);
    }

    private static readonly int[] Bc7Weights4 = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

    private static FormatCase BuildBc7Mode6()
    {
        var blocks = new byte[4][];
        var texels = new float[4][][];
        for (var block = 0; block < 4; block++)
        {
            var random = new Random(600 + block);
            var endpoint0 = new int[4];
            var endpoint1 = new int[4];
            for (var channel = 0; channel < 4; channel++)
            {
                endpoint0[channel] = random.Next(128);
                endpoint1[channel] = random.Next(128);
            }

            var p0 = random.Next(2);
            var p1 = random.Next(2);
            // The anchor texel 0 stores 3 index bits (its implied top bit is 0), so its index is below 8.
            var indices = Enumerable.Range(0, 16).Select(t => t == 0 ? random.Next(8) : random.Next(16)).ToArray();

            var bits = new List<bool>();
            void Put(int value, int count) { for (var i = 0; i < count; i++) { bits.Add(((value >> i) & 1) != 0); } }
            Put(1 << 6, 7); // mode 6: six zero bits then a one
            for (var channel = 0; channel < 4; channel++) { Put(endpoint0[channel], 7); Put(endpoint1[channel], 7); }
            Put(p0, 1);
            Put(p1, 1);
            for (var t = 0; t < 16; t++) { Put(indices[t], t == 0 ? 3 : 4); }
            Assert.Equal(128, bits.Count);
            var bytes = new byte[16];
            for (var i = 0; i < 128; i++) { if (bits[i]) { bytes[i / 8] |= (byte)(1 << (i % 8)); } }
            blocks[block] = bytes;

            var e0 = endpoint0.Select(v => (v << 1) | p0).ToArray();
            var e1 = endpoint1.Select(v => (v << 1) | p1).ToArray();
            texels[block] = indices.Select(index =>
            {
                var weight = Bc7Weights4[index];
                return Enumerable.Range(0, 4).Select(c => ((((64 - weight) * e0[c]) + (weight * e1[c]) + 32) >> 6) / 255f).ToArray();
            }).ToArray();
        }

        return BlockCase(181, 16, blocks, texels, 0.008f);
    }

    [Theory]
    [InlineData("Rgba8Unorm")]
    [InlineData("Rgba8Srgb")]
    [InlineData("Rgb10A2Unorm")]
    [InlineData("R11G11B10Float")]
    [InlineData("Float11_11_10", Skip = "IMG_FORMAT 43 (11_11_10_FLOAT, R in the low ten bits) has no Metal pixel format; the emulator binds it as a null texture with a one-time warning (ResourceMaterializer, the unsupported-format fallback)")]
    [InlineData("Rgba16Float")]
    [InlineData("Rg16Snorm")]
    [InlineData("R32Float")]
    [InlineData("R8Unorm")]
    [InlineData("Bc1")]
    [InlineData("Bc3")]
    [InlineData("Bc4")]
    [InlineData("Bc5")]
    [InlineData("Bc7Mode6")]
    public void PointSampledFormatsDecodeToTheAnalyticTexels(string name)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var format = BuildFormat(name);
        var image = rig.LinearImage(format.ElementsX, format.ElementsY, format.ElementBytes, format.Data);
        uint[] texture = Descriptors.Texture(image, format.Format, format.TexWidth, format.TexHeight);
        var pm4 = new ScenePm4();
        var draw = Draw(rig, pm4, 16, 16, Sample, texture, Descriptors.Sampler(), UvQuad(16, 16, 0, 0, 16, 16, 0, 0, 1, 1));
        rig.Submit(pm4);
        foreach (var fatal in rig.Fatals) { Output.WriteLine($"GPU fatal: {fatal}"); }
        var pixels = Read(rig, draw);
        var check = new Check();
        var scaleX = 16 / (int)format.TexWidth;
        var scaleY = 16 / (int)format.TexHeight;
        for (var y = 0; y < 16; y++)
        {
            for (var x = 0; x < 16; x++)
            {
                var tx = x / scaleX;
                var ty = y / scaleY;
                check.Colour($"{name} texel ({tx},{ty})", Pixel(pixels, 16, x, y), format.Expected[(ty * (int)format.TexWidth) + tx], format.Tolerance);
            }
        }

        check.Done();
    }

    // ======================================================================
    // 7. Address modes
    // ======================================================================

    private static uint AddressTexel(int x, int y) => 0xFF000000u | 0x00640000u | (uint)((30 + (y * 60)) << 8) | (uint)(30 + (x * 60));

    /// <summary>The texel index a mode maps texel coordinate t to, or -1 for the border (t is a texel-centre index, so ties never occur).</summary>
    private static int AddressIndex(uint mode, int t)
    {
        const int n = 4;
        switch (mode)
        {
            case 0: return ((t % n) + n) % n;
            case 1:
            {
                var m = ((t % (2 * n)) + (2 * n)) % (2 * n);
                return m < n ? m : (2 * n) - 1 - m;
            }
            case 2: return Math.Clamp(t, 0, n - 1);
            case 3: // mirror once, clamp to the last texel
            {
                var m = t < 0 ? -t - 1 : t;
                return Math.Min(m, n - 1);
            }
            case 6: return t is >= 0 and < n ? t : -1;
            case 7: // mirror once, border beyond
            {
                var m = t < 0 ? -t - 1 : t;
                return m < n ? m : -1;
            }
            default: throw new ArgumentOutOfRangeException(nameof(mode));
        }
    }

    [Theory]
    [InlineData(0u, 0u)]
    [InlineData(1u, 0u)]
    [InlineData(2u, 0u)]
    [InlineData(3u, 0u)]
    [InlineData(6u, 0u)]
    [InlineData(6u, 1u)]
    [InlineData(6u, 2u)]
    [InlineData(7u, 1u)]
    public void CoordinatesOutsideTheTextureFollowTheSamplerAddressMode(uint mode, uint borderType)
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var texels = new byte[4 * 4 * 4];
        for (var y = 0; y < 4; y++)
        {
            for (var x = 0; x < 4; x++)
            {
                BitConverter.TryWriteBytes(texels.AsSpan(((y * 4) + x) * 4), AddressTexel(x, y));
            }
        }

        var image = rig.LinearImage(4, 4, 4, texels);
        uint[] texture = Descriptors.Texture(image, Rgba8, 4, 4);
        var pm4 = new ScenePm4();
        // u, v from -2 to 2 over 16 pixels: pixel i samples the centre of texel coordinate i - 8.
        var draw = Draw(rig, pm4, 16, 16, Sample, texture,
            Descriptors.Sampler(clampX: mode, clampY: mode, borderColorType: borderType), UvQuad(16, 16, 0, 0, 16, 16, -2, -2, 2, 2));
        rig.Submit(pm4);
        var pixels = Read(rig, draw);
        float[] border = borderType switch { 0 => [0, 0, 0, 0], 1 => [0, 0, 0, 1], _ => [1, 1, 1, 1] };
        var check = new Check();
        for (var y = 0; y < 16; y++)
        {
            for (var x = 0; x < 16; x++)
            {
                var ix = AddressIndex(mode, x - 8);
                var iy = AddressIndex(mode, y - 8);
                var expected = ix < 0 || iy < 0 ? border : Unpack(AddressTexel(ix, iy));
                check.Colour($"mode {mode} border {borderType} texel coord ({x - 8},{y - 8})", Pixel(pixels, 16, x, y), expected, 0.003f);
            }
        }

        check.Done();
    }

    [Fact]
    public void UnnormalisedCoordinatesAddressTexelsDirectly()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var texels = new byte[16 * 16 * 4];
        for (var y = 0; y < 16; y++)
        {
            for (var x = 0; x < 16; x++)
            {
                BitConverter.TryWriteBytes(texels.AsSpan(((y * 16) + x) * 4), 0xFF000000u | (uint)(y * 16 << 8) | (uint)(x * 16));
            }
        }

        var image = rig.LinearImage(16, 16, 4, texels);
        uint[] texture = Descriptors.Texture(image, Rgba8, 16, 16);
        var pm4 = new ScenePm4();
        // u, v in texel units: 0..16 over the 16 pixels, so pixel (x, y) reads texel (x, y).
        var draw = Draw(rig, pm4, 16, 16, Sample, texture, Descriptors.Sampler(forceUnnormalized: true), UvQuad(16, 16, 0, 0, 16, 16, 0, 0, 16, 16));
        rig.Submit(pm4);
        var pixels = Read(rig, draw);
        var check = new Check();
        for (var y = 0; y < 16; y++)
        {
            for (var x = 0; x < 16; x++)
            {
                check.Colour($"unnormalised texel ({x},{y})", Pixel(pixels, 16, x, y), [x * 16 / 255f, y * 16 / 255f, 0f, 1f], 0.003f);
            }
        }

        check.Done();
    }

    // Pixel programs: image T# s[0:7], S# s[8:11], four constants s[12:15], primitive mask s16. PARAM0.xyz is interpolated into v2..v4;
    // the image result lands in v[8:11] and is exported to MRT0 (an RGBA32F target). Assembled with llvm-mc -mcpu=gfx1030.
    private static readonly uint[] Sample =
    [
        0xBEFC0310,                // s_mov_b32 m0, s16
        0xC8080000,                // v_interp_p1_f32_e32 v2, v0, attr0.x
        0xC8090001,                // v_interp_p2_f32_e32 v2, v1, attr0.x
        0xC80C0100,                // v_interp_p1_f32_e32 v3, v0, attr0.y
        0xC80D0101,                // v_interp_p2_f32_e32 v3, v1, attr0.y
        0xC8100200,                // v_interp_p1_f32_e32 v4, v0, attr0.z
        0xC8110201,                // v_interp_p2_f32_e32 v4, v1, attr0.z
        0xF0800F08, 0x00400802,    // image_sample v[8:11], v[2:3], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C3F70,                // s_waitcnt vmcnt(0)
        0xF800180F, 0x0B0A0908,    // exp mrt0 v8, v9, v10, v11 done vm
        0xBF810000,                // s_endpgm
    ];

    private static readonly uint[] SampleL =
    [
        0xBEFC0310,                // s_mov_b32 m0, s16
        0xC8080000,                // v_interp_p1_f32_e32 v2, v0, attr0.x
        0xC8090001,                // v_interp_p2_f32_e32 v2, v1, attr0.x
        0xC80C0100,                // v_interp_p1_f32_e32 v3, v0, attr0.y
        0xC80D0101,                // v_interp_p2_f32_e32 v3, v1, attr0.y
        0xC8100200,                // v_interp_p1_f32_e32 v4, v0, attr0.z
        0xC8110201,                // v_interp_p2_f32_e32 v4, v1, attr0.z
        0x7E08020C,                // v_mov_b32_e32 v4, s12
        0xF0900F08, 0x00400802,    // image_sample_l v[8:11], v[2:4], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C3F70,                // s_waitcnt vmcnt(0)
        0xF800180F, 0x0B0A0908,    // exp mrt0 v8, v9, v10, v11 done vm
        0xBF810000,                // s_endpgm
    ];

    private static readonly uint[] SampleB =
    [
        0xBEFC0310,                // s_mov_b32 m0, s16
        0xC8080000,                // v_interp_p1_f32_e32 v2, v0, attr0.x
        0xC8090001,                // v_interp_p2_f32_e32 v2, v1, attr0.x
        0xC80C0100,                // v_interp_p1_f32_e32 v3, v0, attr0.y
        0xC80D0101,                // v_interp_p2_f32_e32 v3, v1, attr0.y
        0xC8100200,                // v_interp_p1_f32_e32 v4, v0, attr0.z
        0xC8110201,                // v_interp_p2_f32_e32 v4, v1, attr0.z
        0x7E0A0302,                // v_mov_b32_e32 v5, v2
        0x7E0C0303,                // v_mov_b32_e32 v6, v3
        0x7E04020C,                // v_mov_b32_e32 v2, s12
        0x7E060305,                // v_mov_b32_e32 v3, v5
        0x7E080306,                // v_mov_b32_e32 v4, v6
        0xF0940F08, 0x00400802,    // image_sample_b v[8:11], v[2:4], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C3F70,                // s_waitcnt vmcnt(0)
        0xF800180F, 0x0B0A0908,    // exp mrt0 v8, v9, v10, v11 done vm
        0xBF810000,                // s_endpgm
    ];

    private static readonly uint[] SampleD =
    [
        0xBEFC0310,                // s_mov_b32 m0, s16
        0xC8080000,                // v_interp_p1_f32_e32 v2, v0, attr0.x
        0xC8090001,                // v_interp_p2_f32_e32 v2, v1, attr0.x
        0xC80C0100,                // v_interp_p1_f32_e32 v3, v0, attr0.y
        0xC80D0101,                // v_interp_p2_f32_e32 v3, v1, attr0.y
        0xC8100200,                // v_interp_p1_f32_e32 v4, v0, attr0.z
        0xC8110201,                // v_interp_p2_f32_e32 v4, v1, attr0.z
        0x7E0C0302,                // v_mov_b32_e32 v6, v2
        0x7E0E0303,                // v_mov_b32_e32 v7, v3
        0x7E04020C,                // v_mov_b32_e32 v2, s12
        0x7E06020D,                // v_mov_b32_e32 v3, s13
        0x7E08020E,                // v_mov_b32_e32 v4, s14
        0x7E0A020F,                // v_mov_b32_e32 v5, s15
        0xF0880F08, 0x00400802,    // image_sample_d v[8:11], v[2:7], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C3F70,                // s_waitcnt vmcnt(0)
        0xF800180F, 0x0B0A0908,    // exp mrt0 v8, v9, v10, v11 done vm
        0xBF810000,                // s_endpgm
    ];

    private static readonly uint[] Sample3D =
    [
        0xBEFC0310,                // s_mov_b32 m0, s16
        0xC8080000,                // v_interp_p1_f32_e32 v2, v0, attr0.x
        0xC8090001,                // v_interp_p2_f32_e32 v2, v1, attr0.x
        0xC80C0100,                // v_interp_p1_f32_e32 v3, v0, attr0.y
        0xC80D0101,                // v_interp_p2_f32_e32 v3, v1, attr0.y
        0xC8100200,                // v_interp_p1_f32_e32 v4, v0, attr0.z
        0xC8110201,                // v_interp_p2_f32_e32 v4, v1, attr0.z
        0xF0800F10, 0x00400802,    // image_sample v[8:11], v[2:4], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_3D
        0xBF8C3F70,                // s_waitcnt vmcnt(0)
        0xF800180F, 0x0B0A0908,    // exp mrt0 v8, v9, v10, v11 done vm
        0xBF810000,                // s_endpgm
    ];

    private static readonly uint[] SampleArray =
    [
        0xBEFC0310,                // s_mov_b32 m0, s16
        0xC8080000,                // v_interp_p1_f32_e32 v2, v0, attr0.x
        0xC8090001,                // v_interp_p2_f32_e32 v2, v1, attr0.x
        0xC80C0100,                // v_interp_p1_f32_e32 v3, v0, attr0.y
        0xC80D0101,                // v_interp_p2_f32_e32 v3, v1, attr0.y
        0xC8100200,                // v_interp_p1_f32_e32 v4, v0, attr0.z
        0xC8110201,                // v_interp_p2_f32_e32 v4, v1, attr0.z
        0xF0800F28, 0x00400802,    // image_sample v[8:11], v[2:4], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY
        0xBF8C3F70,                // s_waitcnt vmcnt(0)
        0xF800180F, 0x0B0A0908,    // exp mrt0 v8, v9, v10, v11 done vm
        0xBF810000,                // s_endpgm
    ];

    private static readonly uint[] GatherR =
    [
        0xBEFC0310,                // s_mov_b32 m0, s16
        0xC8080000,                // v_interp_p1_f32_e32 v2, v0, attr0.x
        0xC8090001,                // v_interp_p2_f32_e32 v2, v1, attr0.x
        0xC80C0100,                // v_interp_p1_f32_e32 v3, v0, attr0.y
        0xC80D0101,                // v_interp_p2_f32_e32 v3, v1, attr0.y
        0xC8100200,                // v_interp_p1_f32_e32 v4, v0, attr0.z
        0xC8110201,                // v_interp_p2_f32_e32 v4, v1, attr0.z
        0xF1000108, 0x00400802,    // image_gather4 v[8:11], v[2:3], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D
        0xBF8C3F70,                // s_waitcnt vmcnt(0)
        0xF800180F, 0x0B0A0908,    // exp mrt0 v8, v9, v10, v11 done vm
        0xBF810000,                // s_endpgm
    ];

    private static readonly uint[] GatherG =
    [
        0xBEFC0310,                // s_mov_b32 m0, s16
        0xC8080000,                // v_interp_p1_f32_e32 v2, v0, attr0.x
        0xC8090001,                // v_interp_p2_f32_e32 v2, v1, attr0.x
        0xC80C0100,                // v_interp_p1_f32_e32 v3, v0, attr0.y
        0xC80D0101,                // v_interp_p2_f32_e32 v3, v1, attr0.y
        0xC8100200,                // v_interp_p1_f32_e32 v4, v0, attr0.z
        0xC8110201,                // v_interp_p2_f32_e32 v4, v1, attr0.z
        0xF1000208, 0x00400802,    // image_gather4 v[8:11], v[2:3], s[0:7], s[8:11] dmask:0x2 dim:SQ_RSRC_IMG_2D
        0xBF8C3F70,                // s_waitcnt vmcnt(0)
        0xF800180F, 0x0B0A0908,    // exp mrt0 v8, v9, v10, v11 done vm
        0xBF810000,                // s_endpgm
    ];

    private static readonly uint[] GatherB =
    [
        0xBEFC0310,                // s_mov_b32 m0, s16
        0xC8080000,                // v_interp_p1_f32_e32 v2, v0, attr0.x
        0xC8090001,                // v_interp_p2_f32_e32 v2, v1, attr0.x
        0xC80C0100,                // v_interp_p1_f32_e32 v3, v0, attr0.y
        0xC80D0101,                // v_interp_p2_f32_e32 v3, v1, attr0.y
        0xC8100200,                // v_interp_p1_f32_e32 v4, v0, attr0.z
        0xC8110201,                // v_interp_p2_f32_e32 v4, v1, attr0.z
        0xF1000408, 0x00400802,    // image_gather4 v[8:11], v[2:3], s[0:7], s[8:11] dmask:0x4 dim:SQ_RSRC_IMG_2D
        0xBF8C3F70,                // s_waitcnt vmcnt(0)
        0xF800180F, 0x0B0A0908,    // exp mrt0 v8, v9, v10, v11 done vm
        0xBF810000,                // s_endpgm
    ];

    private static readonly uint[] GatherA =
    [
        0xBEFC0310,                // s_mov_b32 m0, s16
        0xC8080000,                // v_interp_p1_f32_e32 v2, v0, attr0.x
        0xC8090001,                // v_interp_p2_f32_e32 v2, v1, attr0.x
        0xC80C0100,                // v_interp_p1_f32_e32 v3, v0, attr0.y
        0xC80D0101,                // v_interp_p2_f32_e32 v3, v1, attr0.y
        0xC8100200,                // v_interp_p1_f32_e32 v4, v0, attr0.z
        0xC8110201,                // v_interp_p2_f32_e32 v4, v1, attr0.z
        0xF1000808, 0x00400802,    // image_gather4 v[8:11], v[2:3], s[0:7], s[8:11] dmask:0x8 dim:SQ_RSRC_IMG_2D
        0xBF8C3F70,                // s_waitcnt vmcnt(0)
        0xF800180F, 0x0B0A0908,    // exp mrt0 v8, v9, v10, v11 done vm
        0xBF810000,                // s_endpgm
    ];

    private static readonly uint[] SampleCube =
    [
        0xBEFC0310,                // s_mov_b32 m0, s16
        0xC8080000,                // v_interp_p1_f32_e32 v2, v0, attr0.x
        0xC8090001,                // v_interp_p2_f32_e32 v2, v1, attr0.x
        0xC80C0100,                // v_interp_p1_f32_e32 v3, v0, attr0.y
        0xC80D0101,                // v_interp_p2_f32_e32 v3, v1, attr0.y
        0xC8100200,                // v_interp_p1_f32_e32 v4, v0, attr0.z
        0xC8110201,                // v_interp_p2_f32_e32 v4, v1, attr0.z
        0xD5440005, 0x04120702,    // v_cubeid_f32 v5, v2, v3, v4
        0xD5450006, 0x04120702,    // v_cubesc_f32 v6, v2, v3, v4
        0xD5460007, 0x04120702,    // v_cubetc_f32 v7, v2, v3, v4
        0xD547000C, 0x04120702,    // v_cubema_f32 v12, v2, v3, v4
        0xD5AA010C, 0x0000010C,    // v_rcp_f32_e64 v12, |v12|
        0x7E1A02FF, 0x3FC00000,    // v_mov_b32_e32 v13, 0x3fc00000
        0x561A1906,                // v_fmac_f32_e32 v13, v6, v12
        0x7E1C02FF, 0x3FC00000,    // v_mov_b32_e32 v14, 0x3fc00000
        0x561C1907,                // v_fmac_f32_e32 v14, v7, v12
        0x7E04030D,                // v_mov_b32_e32 v2, v13
        0x7E06030E,                // v_mov_b32_e32 v3, v14
        0x7E080305,                // v_mov_b32_e32 v4, v5
        0xF0800F18, 0x00400802,    // image_sample v[8:11], v[2:4], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_CUBE
        0xBF8C3F70,                // s_waitcnt vmcnt(0)
        0xF800180F, 0x0B0A0908,    // exp mrt0 v8, v9, v10, v11 done vm
        0xBF810000,                // s_endpgm
    ];

}
