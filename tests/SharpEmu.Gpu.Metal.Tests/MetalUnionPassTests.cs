// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.Pipelines;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A pass whose attachment set was last followed by a larger one opens with the larger one:
/// draws into one target alternating with draws into it and a second come out in one pass,
/// with the pixels of the passes they were drawn in before. A pass opened that way that no
/// draw into the larger set continues keeps the extra target as it was, and two such passes
/// in a row forget the set. Nothing opens that way while the first draw samples the extra
/// target. Device work is behind the exclusive device gate.
/// </summary>
public sealed class MetalUnionPassTests(ITestOutputHelper output)
{
    private const uint Width = 64;
    private const uint Height = 8;
    private const ulong AddressA = MetalPassRig.GuestBase + 0x10000;
    private const ulong AddressB = MetalPassRig.GuestBase + 0x20000;

    private sealed record Outcome(byte[] A, byte[] B, long[] PassesPerFrame, long Opened, long Hits, long Misses, long Refused, int Predictions);

    private enum Frame
    {
        // A alone, A and B, A alone, A and B.
        Alternating,
        // A alone only.
        SmallerOnly,
        // As Alternating, but the first draw samples B.
        AlternatingSamplingB,
    }

    [Fact]
    public void AlternatingSetsComeOutInOnePassWithTheSamePixels()
    {
        if (!RequireDevice()) return;
        var merged = Run(union: true, Frame.Alternating, Frame.Alternating);
        var separate = Run(union: false, Frame.Alternating, Frame.Alternating);

        // The first frame learns the step from A to A and B; the second opens with both.
        Assert.Equal(2L, merged.PassesPerFrame[0]);
        Assert.Equal(1L, merged.PassesPerFrame[1]);
        Assert.Equal(1L, merged.Opened);
        Assert.Equal(1L, merged.Hits);
        Assert.Equal(2L, separate.PassesPerFrame[1]);
        Assert.Equal(0L, separate.Opened);
        Assert.Equal(separate.A, merged.A);
        Assert.Equal(separate.B, merged.B);
        AssertQuarters(merged.A, [Red, Green, White, Blue]);
        AssertQuarters(merged.B, [Pattern, Blue, Pattern, Yellow]);
    }

    [Fact]
    public void APassThatNeverNeedsTheExtraTargetKeepsItAndIsForgottenAfterTwoMisses()
    {
        if (!RequireDevice()) return;
        var merged = Run(union: true, Frame.Alternating, Frame.SmallerOnly, Frame.SmallerOnly, Frame.SmallerOnly);
        var separate = Run(union: false, Frame.Alternating, Frame.SmallerOnly, Frame.SmallerOnly, Frame.SmallerOnly);

        Assert.Equal(2L, merged.Opened);
        Assert.Equal(0L, merged.Hits);
        Assert.Equal(2L, merged.Misses);
        Assert.Equal(0, merged.Predictions);
        Assert.Equal(separate.A, merged.A);
        // B holds what the first frame drew, through the two passes that only loaded and stored it.
        Assert.Equal(separate.B, merged.B);
        AssertQuarters(merged.B, [Pattern, Blue, Pattern, Yellow]);
    }

    [Fact]
    public void AFirstDrawThatSamplesTheExtraTargetOpensItsOwnSet()
    {
        if (!RequireDevice()) return;
        var refused = Run(union: true, Frame.Alternating, Frame.AlternatingSamplingB);
        var separate = Run(union: false, Frame.Alternating, Frame.AlternatingSamplingB);

        Assert.Equal(0L, refused.Opened);
        Assert.Equal(1L, refused.Refused);
        Assert.Equal(2L, refused.PassesPerFrame[1]);
        Assert.Equal(separate.A, refused.A);
        Assert.Equal(separate.B, refused.B);
    }

    private static readonly byte[] Red = [0xFF, 0x00, 0x00, 0xFF];
    private static readonly byte[] Green = [0x00, 0xFF, 0x00, 0xFF];
    private static readonly byte[] Blue = [0x00, 0x00, 0xFF, 0xFF];
    private static readonly byte[] White = [0xFF, 0xFF, 0xFF, 0xFF];
    private static readonly byte[] Yellow = [0xFF, 0xFF, 0x00, 0xFF];
    private static readonly byte[] Pattern = [MetalPassRig.GuestPattern, MetalPassRig.GuestPattern, MetalPassRig.GuestPattern, MetalPassRig.GuestPattern];

    private Outcome Run(bool union, params Frame[] frames)
    {
        using var rig = new MetalPassRig(Device.Value, 1024 * 1024);
        var host = rig.Host;
        host.UnionPasses = union;
        var one = new[] { GpuFormat.R8G8B8A8Unorm };
        var two = new[] { GpuFormat.R8G8B8A8Unorm, GpuFormat.R8G8B8A8Unorm };
        var s1 = rig.Pipeline(one, ["1.0f, 0.0f, 0.0f, 1.0f"], -1f, -0.5f);
        var u1 = rig.Pipeline(two, ["0.0f, 1.0f, 0.0f, 1.0f", "0.0f, 0.0f, 1.0f, 1.0f"], -0.5f, 0f);
        var s2 = rig.Pipeline(one, ["1.0f, 1.0f, 1.0f, 1.0f"], 0f, 0.5f);
        var u2 = rig.Pipeline(two, ["0.0f, 0.0f, 1.0f, 1.0f", "1.0f, 1.0f, 0.0f, 1.0f"], 0.5f, 1f);
        var a = rig.MakeTarget(AddressA, GpuFormat.R8G8B8A8Unorm, Width, Height);
        var b = rig.MakeTarget(AddressB, GpuFormat.R8G8B8A8Unorm, Width, Height);
        _ = host.TakeUnionPasses();
        var passes = new long[frames.Length];
        for (var index = 0; index < frames.Length; index++)
        {
            var first = host.RenderPassSerial;
            using (host.BeginPreparation())
            {
                MetalRenderHost.MetalStageBindings? stage = null;
                if (frames[index] == Frame.AlternatingSamplingB)
                {
                    var targetB = b.State;
                    var view = ((SharpEmu.Libs.Gpu.Rendering.IRenderHost)host).AcquireColorAttachment(in targetB).View;
                    stage = MetalPassRig.Stage(textures: [(nint)view.Handle]);
                }

                rig.Draw(s1, [a], stage);
                if (frames[index] != Frame.SmallerOnly)
                {
                    rig.Draw(u1, [a, b]);
                    rig.Draw(s2, [a]);
                    rig.Draw(u2, [a, b]);
                }

                host.EndRendering();
            }

            passes[index] = host.RenderPassSerial - first;
        }

        var (_, opened, hits, misses, refused, _) = host.TakeUnionPasses();
        var predictions = host.UnionPredictionCount;
        host.FlushCommandBuffer();
        var pixelsA = rig.Read(AddressA, Width * Height * 4);
        var pixelsB = rig.Read(AddressB, Width * Height * 4);
        output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"union={union} frames={string.Join(',', frames)} passes={string.Join(',', passes)} opened={opened} hits={hits} misses={misses} refused={refused} predictions={predictions}"));
        return new Outcome(pixelsA, pixelsB, passes, opened, hits, misses, refused, predictions);
    }

    private static void AssertQuarters(byte[] pixels, byte[][] quarters)
    {
        for (var y = 0u; y < Height; y++)
        {
            for (var x = 0u; x < Width; x++)
            {
                var texel = (int)(((y * Width) + x) * 4);
                var expected = quarters[x / (Width / 4)];
                Assert.True(
                    pixels.AsSpan(texel, 4).SequenceEqual(expected),
                    $"texel ({x}, {y}) is {Convert.ToHexString(pixels, texel, 4)}, expected {Convert.ToHexString(expected)}");
            }
        }
    }

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Union passes are encoded on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
