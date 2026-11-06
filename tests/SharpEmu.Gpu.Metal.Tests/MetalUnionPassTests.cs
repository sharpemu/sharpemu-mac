// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands;
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

    private sealed record Outcome(
        byte[] A, byte[] B, long[] PassesPerFrame, long Opened, long Hits, long Misses, long Refused, int Predictions, string MissCauses);

    private enum Frame
    {
        // A alone, A and B, A alone, A and B.
        Alternating,
        // A alone only.
        SmallerOnly,
        // As Alternating, but the first draw samples B.
        AlternatingSamplingB,
        // A alone, A alone again with a draw that samples B, then A and B.
        SecondSmallerSamplesB,
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

    // Demon's Souls' G-buffer as its pass profile shows it at 1920x1080: five targets (four
    // R32Uint, one R16G16Sfloat) and the same five with an R16G16B16A16Sfloat, the same views
    // every frame. A run of draws into the five - mesh and vertex, more than one program, with
    // an end-of-pipe label held to the commit and programs that reach memory through device
    // addresses - comes before the draw into the six. With doubleBuffered the frames
    // alternate between two such sets at different addresses.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ADemonsSoulsGBufferRunContinuesItsUnionPassWithALabelHeld(bool doubleBuffered)
    {
        if (!RequireDevice()) return;
        const int frames = 6;
        var merged = RunDemonsSouls(union: true, doubleBuffered, frames);
        var separate = RunDemonsSouls(union: false, doubleBuffered, frames);

        var learning = doubleBuffered ? 2 : 1;
        for (var frame = 0; frame < frames; frame++)
        {
            Assert.Equal(2L, separate.PassesPerFrame[frame]);
            Assert.Equal(frame < learning ? 2L : 1L, merged.PassesPerFrame[frame]);
        }

        Assert.Equal(frames - learning, merged.Opened);
        Assert.Equal(frames - learning, merged.Hits);
        Assert.Equal(0L, merged.Misses);
        Assert.Equal(0L, separate.Opened);
        Assert.Equal(separate.Targets.Length, merged.Targets.Length);
        for (var index = 0; index < separate.Targets.Length; index++)
        {
            Assert.True(separate.Targets[index].AsSpan().SequenceEqual(merged.Targets[index]), $"target {index} differs");
        }
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
        Assert.Equal("end:2", merged.MissCauses);
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

    [Fact]
    public void ADrawIntoTheSmallerSetThatSamplesTheExtraTargetEndsTheUnionPassAndForgetsTheSet()
    {
        if (!RequireDevice()) return;
        var merged = Run(union: true, Frame.Alternating, Frame.SecondSmallerSamplesB, Frame.Alternating);
        var separate = Run(union: false, Frame.Alternating, Frame.SecondSmallerSamplesB, Frame.Alternating);

        // The second frame's union pass ends at the draw that samples B, a split its own pass
        // would not have had: the set is forgotten at once, and learnt again by the step that follows.
        Assert.Equal(3L, merged.PassesPerFrame[1]);
        Assert.Equal(2L, separate.PassesPerFrame[1]);
        Assert.Equal(1L, merged.PassesPerFrame[2]);
        Assert.Equal(2L, merged.Opened);
        Assert.Equal(1L, merged.Hits);
        Assert.Equal(1L, merged.Misses);
        Assert.Equal("conflict-sampled:1", merged.MissCauses);
        Assert.Equal(separate.A, merged.A);
        Assert.Equal(separate.B, merged.B);
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
                if (frames[index] is Frame.AlternatingSamplingB or Frame.SecondSmallerSamplesB)
                {
                    var targetB = b.State;
                    var view = ((SharpEmu.Libs.Gpu.Rendering.IRenderHost)host).AcquireColorAttachment(in targetB).View;
                    stage = MetalPassRig.Stage(textures: [(nint)view.Handle]);
                }

                if (frames[index] == Frame.SecondSmallerSamplesB)
                {
                    rig.Draw(s1, [a]);
                    rig.Draw(s2, [a], stage);
                    rig.Draw(u1, [a, b]);
                }
                else
                {
                    rig.Draw(s1, [a], stage);
                }

                if (frames[index] is not (Frame.SmallerOnly or Frame.SecondSmallerSamplesB))
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
        var missCauses = host.TakeUnionMissCauses();
        host.FlushCommandBuffer();
        var pixelsA = rig.Read(AddressA, Width * Height * 4);
        var pixelsB = rig.Read(AddressB, Width * Height * 4);
        output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"union={union} frames={string.Join(',', frames)} passes={string.Join(',', passes)} opened={opened} hits={hits} misses={misses} refused={refused} predictions={predictions} miss=[{missCauses}]"));
        return new Outcome(pixelsA, pixelsB, passes, opened, hits, misses, refused, predictions, missCauses);
    }

    private sealed record DsOutcome(byte[][] Targets, long[] PassesPerFrame, long Opened, long Hits, long Misses, string Counters);

    private const uint DsWidth = 1920;
    private const uint DsHeight = 1080;
    private const ulong DsLabel = MetalPassRig.GuestBase + 0x8000;
    private const ulong DsTargets = MetalPassRig.GuestBase + 0x10_0000;

    private static readonly GpuFormat[] DsFive =
    [
        GpuFormat.R32Uint, GpuFormat.R32Uint, GpuFormat.R32Uint, GpuFormat.R32Uint, GpuFormat.R16G16Sfloat,
    ];

    private static readonly GpuFormat[] DsSix = [.. DsFive, GpuFormat.R16G16B16A16Sfloat];

    private DsOutcome RunDemonsSouls(bool union, bool doubleBuffered, int frames)
    {
        using var rig = new MetalPassRig(Device.Value, 192UL * 1024 * 1024);
        var host = rig.Host;
        host.UnionPasses = union;
        // Each draw covers the screen and leaves a quarter of it, a different quarter for each,
        // so the pixels depend on the order the draws landed in.
        var first = rig.Pipeline(DsFive, DsOutputs(DsFive, 1), mesh: true, discardWhen: DsQuarter(0));
        var second = rig.Pipeline(DsFive, DsOutputs(DsFive, 2), discardWhen: DsQuarter(1));
        var third = rig.Pipeline(DsFive, DsOutputs(DsFive, 3), mesh: true, discardWhen: DsQuarter(2));
        var six = rig.Pipeline(DsSix, DsOutputs(DsSix, 4), discardWhen: DsQuarter(3));
        var address = DsTargets;
        var sets = new MetalPassRig.Target[doubleBuffered ? 2 : 1][];
        for (var set = 0; set < sets.Length; set++)
        {
            sets[set] = new MetalPassRig.Target[DsSix.Length];
            for (var index = 0; index < DsSix.Length; index++)
            {
                sets[set][index] = rig.MakeTarget(address, DsSix[index], DsWidth, DsHeight);
                address += (sets[set][index].Bytes + 0xFFFF) & ~0xFFFFUL;
            }
        }

        var deviceAddresses = MetalPassRig.Stage(deviceAddresses: true);
        _ = host.TakeUnionPasses();
        var passes = new long[frames];
        for (var frame = 0; frame < frames; frame++)
        {
            var targets = sets[frame % sets.Length];
            var five = targets.AsSpan(0, DsFive.Length).ToArray();
            var firstPass = host.RenderPassSerial;
            using (host.BeginPreparation())
            {
                rig.Draw(first, five);
                // Held to the commit, as the labels Demon's Souls writes between its draws are: a
                // draw whose program reaches memory through device addresses may read their bytes.
                host.RecordCommandProcessorWrite(DsLabel, BitConverter.GetBytes((ulong)frame + 1), "WRITE_DATA");
                rig.Draw(second, five, deviceAddresses);
                host.RecordCommandProcessorWrite(DsLabel + 8, BitConverter.GetBytes((ulong)frame + 1), "WRITE_DATA");
                rig.Draw(third, five, deviceAddresses);
                rig.Draw(six, targets);
                host.EndRendering();
            }

            ((ICommandStreamHost)host).Flush();
            passes[frame] = host.RenderPassSerial - firstPass;
        }

        var (_, opened, hits, misses, _, _) = host.TakeUnionPasses();
        var counters = string.Create(CultureInfo.InvariantCulture, $"continued={host.TakeUnionContinued()} miss=[{host.TakeUnionMissCauses()}]");
        host.FlushCommandBuffer();
        var read = new List<byte[]>();
        foreach (var set in sets)
        {
            foreach (var target in set)
            {
                read.Add(rig.Read(target.Address, target.Bytes));
            }
        }

        output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"union={union} double_buffered={doubleBuffered} passes={string.Join(',', passes)} opened={opened} hits={hits} misses={misses} {counters}"));
        return new DsOutcome([.. read], passes, opened, hits, misses, counters);
    }

    private static string[] DsOutputs(GpuFormat[] formats, int seed)
    {
        var outputs = new string[formats.Length];
        for (var index = 0; index < formats.Length; index++)
        {
            var value = (seed * 8) + index;
            outputs[index] = MetalPassRig.IsInteger(formats[index])
                ? string.Create(CultureInfo.InvariantCulture, $"rig_hash(in.position, {value}u), {value}u, 0u, 1u")
                : string.Create(CultureInfo.InvariantCulture,
                    $"float(rig_hash(in.position, {value}u) & 255u) / 255.0f, {value}.0f / 64.0f, 0.5f, 1.0f");
        }

        return outputs;
    }

    private static string DsQuarter(int quarter) =>
        string.Create(CultureInfo.InvariantCulture, $"rig_hash(in.position, 977u) % 4u == {quarter}u");

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
