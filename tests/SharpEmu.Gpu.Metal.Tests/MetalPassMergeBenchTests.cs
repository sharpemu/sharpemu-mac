// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.Gpu.Abstractions;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests;

/// <summary>
/// A synthetic Demon's Souls-shaped frame at 1920x1080, replayed with the pass-merging work
/// off and on. Runs of three draws into five attachments (four R32Uint, one R16G16Sfloat; mesh,
/// vertex, mesh) alternate with draws that add a sixth (R16G16B16A16Sfloat), five times a
/// frame; the second and third draw of each run reach memory through device addresses and
/// follow a command-processor write held to the commit, as Demon's Souls' labels are; two
/// guest fills land between draws; sixteen Stream submissions end with the stream busy, one of
/// each segment's between its second and third draw, so the coalescing limit asks for a commit
/// mid-frame; two full-screen single-draw passes follow. The capped configurations hold such a
/// commit for no time at all, as Demon's Souls' G-buffer outlasts the hold limit, so the commit
/// ends the pass unless the GPU still runs earlier work. Each optimization
/// is measured alone and all together, and the targets and the filled bytes of all-on are
/// compared with all-off byte for byte. The targets hold incompressible noise, so a split costs
/// its full load and store here (about 8 µs/MB on an M4) where Demon's Souls' measured ~3.5
/// µs/MB: read the saving per split, not the absolute milliseconds. No depth target: Demon's
/// Souls' passes also carry a D32S8 target, which each split it avoids also loads and stores.
/// Device work is behind the exclusive device gate; the timed run also needs
/// SHARPEMU_BENCH_PASSES=1.
/// </summary>
public sealed class MetalPassMergeBenchTests(ITestOutputHelper output)
{
    private const uint Width = 1920;
    private const uint Height = 1080;
    private const int Segments = 5;
    private const int FlushesPerSegment = 3;
    private const ulong FillBytes = 0xC000;
    private const ulong FillBase = MetalPassRig.GuestBase + 0x10_0000;
    private const ulong LabelBase = MetalPassRig.GuestBase + 0x8000;
    private const ulong ConstantsBase = MetalPassRig.GuestBase + 0x9000;
    private const ulong TargetBase = MetalPassRig.GuestBase + 0x100_0000;

    private static readonly GpuFormat[] Shared =
    [
        GpuFormat.R32Uint, GpuFormat.R32Uint, GpuFormat.R32Uint, GpuFormat.R32Uint, GpuFormat.R16G16Sfloat,
    ];

    private static readonly GpuFormat[] Union = [.. Shared, GpuFormat.R16G16B16A16Sfloat];
    private static readonly GpuFormat[] Post = [GpuFormat.R16G16B16A16Sfloat];

    private readonly record struct Config(
        string Name, bool Union, bool Hold, bool Fills, bool Elide, bool Ranges = false, bool Busy = false, bool Capped = false);

    private static readonly Config Off = new("off", false, false, false, false);
    private static readonly Config On = new("all-on", true, true, true, true, true, true);

    // What 79c942699 had on: every pass merge but the join's device-range check and the busy-GPU hold.
    private static readonly Config Before = new("r15", true, true, true, true);
    private static readonly Config CappedBefore = new("capped:r15", true, true, true, true, false, false, true);
    private static readonly Config CappedOn = On with { Name = "capped:all-on", Capped = true };

    private static readonly Config[] Configs =
    [
        Off,
        new("union", true, false, false, false),
        new("flush-hold", false, true, false, false),
        new("defer-fills", false, false, true, false),
        new("elide-loads", false, false, false, true),
        // Each split the others leave in place hides the one a lone switch removes, so each is
        // also measured as what all-on loses without it.
        new("join-ranges", false, false, false, false, true),
        new("on-no-union", false, true, true, true, true, true),
        new("on-no-hold", true, false, true, true, true, false),
        new("on-no-fills", true, true, false, true, true, true),
        new("on-no-elide", true, true, true, false, true, true),
        new("on-no-ranges", true, true, true, true, false, true),
        Before,
        On,
        CappedBefore,
        new("capped:no-busy", true, true, true, true, true, false, true),
        new("capped:no-ranges", true, true, true, true, false, true, true),
        CappedOn,
    ];

    private sealed record Outcome(double[] GpuMs, double PassesPerFrame, byte[][] Bytes, string Counters);

    [Fact]
    public void ADsShapedFrameComesOutTheSameWithEveryPassMergeOn()
    {
        if (!RequireDevice()) return;
        var off = Run(Off, warmup: 1, frames: 1);
        var before = Run(Before, warmup: 1, frames: 1);
        var on = Run(On, warmup: 1, frames: 1);

        output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"passes/frame off={off.PassesPerFrame} before={before.PassesPerFrame} on={on.PassesPerFrame} {on.Counters}"));
        Assert.True(on.PassesPerFrame < before.PassesPerFrame);
        Assert.True(before.PassesPerFrame < off.PassesPerFrame);
        AssertSameBytes(off, before);
        AssertSameBytes(off, on);
    }

    // Demon's Souls' commits capped mid-pass, with the GPU held busy so the result does not
    // depend on how fast it ran the frame's earlier buffers: the busy-GPU hold keeps the union
    // pass to its six-target draw, and the join keeps the five-target draws after it.
    [Fact]
    public void ADsShapedFrameWithCappedCommitsComesOutTheSameWithFewerPasses()
    {
        if (!RequireDevice()) return;
        var off = Run(Off, warmup: 1, frames: 1);
        var before = Run(CappedBefore, warmup: 1, frames: 1, gpuBusy: true);
        var noBusy = Run(CappedOn with { Busy = false }, warmup: 1, frames: 1, gpuBusy: true);
        var on = Run(CappedOn, warmup: 1, frames: 1, gpuBusy: true);
        var idle = Run(CappedOn, warmup: 1, frames: 1, gpuBusy: false);

        output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"passes/frame off={off.PassesPerFrame} capped-before={before.PassesPerFrame} no-busy={noBusy.PassesPerFrame} " +
            $"on={on.PassesPerFrame} idle-gpu={idle.PassesPerFrame}\n before: {before.Counters}\n on: {on.Counters}"));
        Assert.Contains("end_flush_by=[stream-capped:", before.Counters, StringComparison.Ordinal);
        Assert.True(noBusy.PassesPerFrame < before.PassesPerFrame);
        Assert.True(on.PassesPerFrame < noBusy.PassesPerFrame);
        // A GPU with nothing left to run gets the commit at the limit, as before.
        Assert.Equal(noBusy.PassesPerFrame, idle.PassesPerFrame);
        AssertSameBytes(off, before);
        AssertSameBytes(off, noBusy);
        AssertSameBytes(off, on);
        AssertSameBytes(off, idle);
    }

    private static void AssertSameBytes(Outcome expected, Outcome actual)
    {
        Assert.Equal(expected.Bytes.Length, actual.Bytes.Length);
        for (var index = 0; index < expected.Bytes.Length; index++)
        {
            Assert.True(expected.Bytes[index].AsSpan().SequenceEqual(actual.Bytes[index]), $"target {index} differs");
        }
    }

    [Fact]
    public void BenchADsShapedFrame()
    {
        if (!RequireDevice()) return;
        if (Environment.GetEnvironmentVariable("SHARPEMU_BENCH_PASSES") != "1")
        {
            output.WriteLine("The timed run needs SHARPEMU_BENCH_PASSES=1.");
            return;
        }

        var frames = int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_BENCH_FRAMES"), out var parsed) ? parsed : 40;
        var results = new List<(Config Config, Outcome Outcome)>();
        // Three times through, interleaved, so drift in the GPU's clocks lands on every
        // configuration; the median frame of each is reported.
        for (var round = 0; round < 3; round++)
        {
            foreach (var config in Configs)
            {
                results.Add((config, Run(config, warmup: 4, frames: frames, keepBytes: false)));
            }
        }

        var baseline = Median(results.Where(result => result.Config == Off).SelectMany(result => result.Outcome.GpuMs));
        foreach (var group in results.GroupBy(result => result.Config.Name))
        {
            var median = Median(group.SelectMany(result => result.Outcome.GpuMs));
            var rounds = group.Select(result => Median(result.Outcome.GpuMs).ToString("F3", CultureInfo.InvariantCulture));
            output.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"{group.Key,-12} gpu_ms/frame median={median:F3} (rounds {string.Join("/", rounds)}) " +
                $"saved={baseline - median:F3} passes/frame={group.First().Outcome.PassesPerFrame:F1} {group.Last().Outcome.Counters}"));
        }
    }

    private Outcome Run(Config config, int warmup, int frames, bool keepBytes = true, bool? gpuBusy = null)
    {
        using var rig = new MetalPassRig(Device.Value, 160UL * 1024 * 1024);
        var host = rig.Host;
        host.UnionPasses = config.Union;
        host.HoldFlushForPass = config.Hold;
        host.DeferPassFills = config.Fills;
        host.JoinChecksDeviceRanges = config.Ranges;
        host.HoldFlushWhileGpuBusy = config.Busy;
        host.AssumeGpuBusy = gpuBusy;
        host.AssumeStreamBusy = true;
        if (config.Capped)
        {
            host.HoldFlushTicks = 0;
        }

        var shared = new PipelineHandle[Segments];
        var sharedVertex = new PipelineHandle[Segments];
        var sharedMesh = new PipelineHandle[Segments];
        var union = new PipelineHandle[Segments];
        for (var segment = 0; segment < Segments; segment++)
        {
            // Every draw covers the whole screen, as Demon's Souls' G-buffer geometry reaches
            // every tile (a tile no draw touches is neither loaded nor stored, so narrow draws
            // would hide the cost being measured), and writes a fifth of its pixels, a
            // different fifth for each draw, so the final pixels depend on the order every
            // draw landed in.
            shared[segment] = rig.Pipeline(Shared, Outputs(Shared, segment * 16 + 1), mesh: true,
                discardWhen: Pattern(segment));
            sharedVertex[segment] = rig.Pipeline(Shared, Outputs(Shared, segment * 16 + 3),
                discardWhen: Pattern(segment + 1));
            sharedMesh[segment] = rig.Pipeline(Shared, Outputs(Shared, segment * 16 + 5), mesh: true,
                discardWhen: Pattern(segment + 3));
            union[segment] = rig.Pipeline(Union, Outputs(Union, segment * 16 + 8),
                discardWhen: Pattern(segment + 2));
        }

        var tonemap = rig.Pipeline(Post, Outputs(Post, 101), knownPositions: config.Elide);
        var present = rig.Pipeline(Post, Outputs(Post, 117), knownPositions: config.Elide);

        var address = TargetBase;
        var targets = new MetalPassRig.Target[Union.Length];
        for (var index = 0; index < Union.Length; index++)
        {
            targets[index] = rig.MakeTarget(address, Union[index], Width, Height);
            address += Align(targets[index].Bytes);
        }

        var tonemapTarget = rig.MakeTarget(address, Post[0], Width, Height);
        address += Align(tonemapTarget.Bytes);
        var presentTarget = rig.MakeTarget(address, Post[0], Width, Height);

        var five = targets.AsSpan(0, Shared.Length).ToArray();
        // The draws' own constants, apart from the labels held ahead of them.
        var deviceAddresses = MetalPassRig.Stage(deviceAddresses: true, deviceRanges: [(ConstantsBase, 0x1000)]);
        _ = host.TakeGpuBusySecondsWhenIdle();
        _ = host.TakeUnionPasses();
        _ = host.TakeHeldCommits();
        _ = host.TakeBusyHeldCommits();
        _ = host.TakePassEndFlushReasons();
        _ = host.TakeDeviceRangeJoinChecks();
        _ = host.TakeDeferredFills();
        _ = host.TakeElidedLoads();
        var gpuMs = new List<double>();
        long passes = 0;
        for (var frame = 0; frame < warmup + frames; frame++)
        {
            var firstPass = host.RenderPassSerial;
            var fill = 0;
            for (var segment = 0; segment < Segments; segment++)
            {
                using (host.BeginPreparation())
                {
                    rig.Draw(shared[segment], five);
                    host.RecordCommandProcessorWrite(LabelBase, BitConverter.GetBytes((ulong)frame), "WRITE_DATA");
                    rig.Draw(sharedVertex[segment], five, deviceAddresses);
                }

                // A submission ends between the five-target draws, before the six-target one.
                ((ICommandStreamHost)host).Flush();
                using (host.BeginPreparation())
                {
                    host.RecordCommandProcessorWrite(LabelBase + 8, BitConverter.GetBytes((ulong)frame), "WRITE_DATA");
                    rig.Draw(sharedMesh[segment], five, deviceAddresses);
                    if (segment is 1 or 3)
                    {
                        host.FillGuestMemory(FillBase + (ulong)fill * 0x1_0000, FillBytes, 0xA1B2C3D4u + (uint)fill + (uint)frame, allowHost: false);
                        fill++;
                    }

                    rig.Draw(union[segment], targets);
                }

                for (var flush = 1; flush < FlushesPerSegment; flush++)
                {
                    ((ICommandStreamHost)host).Flush();
                }
            }

            using (host.BeginPreparation())
            {
                rig.Draw(tonemap, [tonemapTarget]);
                rig.Draw(present, [presentTarget]);
                host.EndRendering();
            }

            ((ICommandStreamHost)host).Flush();
            var seconds = host.TakeGpuBusySecondsWhenIdle();
            if (frame >= warmup)
            {
                gpuMs.Add(seconds * 1000.0);
                passes += host.RenderPassSerial - firstPass;
            }
        }

        var counters = string.Create(CultureInfo.InvariantCulture,
            $"union={host.TakeUnionPasses()} union_continued={host.TakeUnionContinued()} union_miss=[{host.TakeUnionMissCauses()}] " +
            $"flush_hold={host.TakeHeldCommits()} busy={host.TakeBusyHeldCommits()} end_flush_by=[{host.TakePassEndFlushReasons()}] " +
            $"join_device_ranges={host.TakeDeviceRangeJoinChecks()} fills_deferred={host.TakeDeferredFills()} load_elided={host.TakeElidedLoads()}");
        var bytes = Array.Empty<byte[]>();
        if (keepBytes)
        {
            var read = new List<byte[]>();
            foreach (var target in targets.Append(tonemapTarget).Append(presentTarget))
            {
                read.Add(rig.Read(target.Address, target.Bytes));
            }

            read.Add(rig.Read(FillBase, 0x2_0000));
            bytes = [.. read];
        }

        return new Outcome([.. gpuMs], (double)passes / frames, bytes, counters);
    }

    // Per-pixel noise from a well-mixed hash (cheap to shade), so the targets do not compress:
    // Apple's lossless framebuffer compression makes loading and storing a flat or smoothly
    // varying target nearly free, and Demon's Souls' G-buffer is neither.
    private static string[] Outputs(GpuFormat[] formats, int seed)
    {
        var outputs = new string[formats.Length];
        for (var index = 0; index < formats.Length; index++)
        {
            var value = (seed + index) * 4;
            outputs[index] = MetalPassRig.IsInteger(formats[index])
                ? string.Create(CultureInfo.InvariantCulture,
                    $"rig_hash(in.position, {value}u), rig_hash(in.position, {value + 1}u), " +
                    $"rig_hash(in.position, {value + 2}u), rig_hash(in.position, {value + 3}u)")
                : string.Create(CultureInfo.InvariantCulture,
                    $"float(rig_hash(in.position, {value}u) & 1023u) / 1023.0f, " +
                    $"float(rig_hash(in.position, {value + 1}u) & 1023u) / 1023.0f, " +
                    $"float(rig_hash(in.position, {value + 2}u) & 1023u) / 1023.0f, " +
                    $"float(rig_hash(in.position, {value + 3}u) & 1023u) / 1023.0f");
        }

        return outputs;
    }

    private static string Pattern(int fifth) =>
        string.Create(CultureInfo.InvariantCulture, $"rig_hash(in.position, 977u) % 5u != {fifth % Segments}u");

    private static double Median(IEnumerable<double> values)
    {
        var sorted = values.Order().ToArray();
        return sorted.Length == 0 ? 0 : sorted[sorted.Length / 2];
    }

    private static ulong Align(ulong bytes) => (bytes + 0xFFFF) & ~0xFFFFUL;

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("The pass-merge frame is replayed on the device only behind the exclusive device gate.");
            return false;
        }

        Assert.True(OperatingSystem.IsMacOS());
        Assert.NotEqual(0, Device.Value);
        return true;
    }

    private static readonly Lazy<nint> Device = new(MetalNative.MTLCreateSystemDefaultDevice);
}
