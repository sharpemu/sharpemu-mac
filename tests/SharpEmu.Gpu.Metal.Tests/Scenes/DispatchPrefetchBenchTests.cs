// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using SharpEmu.Gpu.Metal.V2;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// Runs of dispatches through the whole path (interpreter, command thread, resolver, worker, GPU), the way Demon's
/// Souls issues ~1000 a frame: each dispatch reads a texture through its user data and its destination image and
/// sizes through a table of its own in guest memory, so its program resolution materializes a table every time.
/// SHARPEMU_METAL_V2_DISPATCH_PREFETCH=0 / _CROSS=0 in the environment compare the worker-resolved path.
/// </summary>
public sealed class DispatchPrefetchBenchTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Edge = 16;
    private const int Images = 64;

    private static readonly MethodInfo PrefetchCounters =
        typeof(MetalRenderHost).GetMethod("FormatPrefetchCounters", BindingFlags.Instance | BindingFlags.NonPublic)!;

    // SHARPEMU_DISPATCH_PREFETCH_BENCH_OUTPUT=<path>: microseconds per dispatch for a run of _DISPATCHES (256)
    // dispatches, _ROUNDS (40) times after _WARM (10), with an FNV checksum of every destination image so runs of
    // the two paths can be compared for identical results. _FLUSH_EVERY=<n> puts a CS partial flush after every
    // n-th dispatch (0, none, by default), which holds a prefetched dispatch behind it.
    [Fact]
    public void MeasureDispatchRunsWhenRequested()
    {
        var path = Environment.GetEnvironmentVariable("SHARPEMU_DISPATCH_PREFETCH_BENCH_OUTPUT");
        if (path is null) return;
        Environment.SetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE", "1");
        using var rig = CreateRig();
        Assert.NotNull(rig);
        int Setting(string name, int fallback) =>
            int.TryParse(Environment.GetEnvironmentVariable($"SHARPEMU_DISPATCH_PREFETCH_BENCH_{name}"), out var value) ? value : fallback;
        var dispatches = Setting("DISPATCHES", 256);
        var rounds = Setting("ROUNDS", 40);
        var warm = Setting("WARM", 10);
        var flushEvery = Setting("FLUSH_EVERY", 0);

        var images = Enumerable.Range(0, Images).Select(_ => rig.ColorTarget(Edge, Edge, 4)).ToArray();
        for (var index = 0; index < Images; index++)
        {
            rig.Memory.Write<uint>(images[index], Enumerable.Range(0, (int)(Edge * Edge)).Select(texel => (uint)(texel * 7 + index)).ToArray());
        }

        var pm4 = new ScenePm4();
        for (var dispatch = 0; dispatch < dispatches; dispatch++)
        {
            var source = images[dispatch % Images];
            var destination = images[(dispatch + 1) % Images];
            var table = rig.Memory.Upload<uint>(
                [.. Descriptors.Texture(destination, 22, Edge, Edge), Edge, Edge, Edge / 2, Edge / 2], 16);
            var state = new SceneState(1, 1);
            state.Context.Clear();
            state.Uconfig.Clear();
            state.ComputeShader(rig.Program(ComputeScenePrograms.HzbReduceMax), 8, 8, 1,
                [.. Descriptors.Texture(source, 22, Edge, Edge), Descriptors.Lo(table), Descriptors.Hi(table)], true, true, false, 1, 0);
            state.Emit(pm4);
            pm4.DispatchDirect(1, 1, 1);
            if (flushEvery > 0 && dispatch % flushEvery == flushEvery - 1)
            {
                pm4.Event(ScenePm4.EventCsPartialFlush);
            }
        }

        var words = pm4.ToArray();
        var stream = rig.Memory.Upload<uint>(words, 256);
        ulong submission = 1000;
        var microseconds = new List<double>();
        _ = PrefetchCounters.Invoke(rig.Host, null);
        string counters = string.Empty;
        for (var round = 0; round < warm + rounds; round++)
        {
            if (round == warm)
            {
                _ = PrefetchCounters.Invoke(rig.Host, null);
            }

            var started = Stopwatch.GetTimestamp();
            rig.Host.CommandStream.EnqueueGraphics(stream, (uint)words.Length, submission++, null);
            rig.Drain();
            var elapsed = Stopwatch.GetTimestamp() - started;
            if (round >= warm)
            {
                microseconds.Add(elapsed * 1e6 / Stopwatch.Frequency / dispatches);
            }
        }

        counters = (string)PrefetchCounters.Invoke(rig.Host, null)!;
        var checksum = 0xCBF29CE484222325UL;
        foreach (var image in images)
        {
            foreach (var texel in rig.Read<uint>(image, (int)(Edge * Edge)))
            {
                checksum = (checksum ^ texel) * 0x100000001B3UL;
            }
        }

        microseconds.Sort();
        var median = microseconds[microseconds.Count / 2];
        Output.WriteLine($"dispatch run: median {median:F2} us/dispatch, p25 {microseconds[microseconds.Count / 4]:F2}, min {microseconds[0]:F2}; {counters}");
        Assert.Empty(rig.Fatals);
        File.WriteAllText(path, JsonSerializer.Serialize(new
        {
            arch = RuntimeInformation.ProcessArchitecture.ToString(),
            prefetch = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DISPATCH_PREFETCH") ?? "default",
            cross = Environment.GetEnvironmentVariable("SHARPEMU_METAL_V2_DISPATCH_PREFETCH_CROSS") ?? "default",
            dispatches,
            rounds,
            flushEvery,
            medianMicrosecondsPerDispatch = median,
            p25MicrosecondsPerDispatch = microseconds[microseconds.Count / 4],
            minMicrosecondsPerDispatch = microseconds[0],
            checksum = $"{checksum:X16}",
            counters,
        }, new JsonSerializerOptions { WriteIndented = true }));
    }
}
