// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using SharpEmu.Gpu.Metal.V2;
using SharpEmu.Libs.VideoOut;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// The worker's per-draw recording cost (RenderExecutor.RecordDraw and everything below it): the same
/// three-texture, depth-tested draw repeated, then with one binding or the pixel program alternating.
/// </summary>
public sealed class DrawRecordBenchTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Rgba8 = 56;
    // SHARPEMU_DRAW_RECORD_BENCH_TEXTURE=<edge> samples edge x edge linear images (8 by default; a title's are megabytes).
    private static readonly uint TextureSize =
        uint.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_DRAW_RECORD_BENCH_TEXTURE"), out var edge) ? edge : 8;
    private const uint TargetSize = 64;
    // SHARPEMU_DRAW_RECORD_BENCH_DRAWS=<n> draws per submission (256 by default).
    private static readonly int DrawsPerSubmit =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_DRAW_RECORD_BENCH_DRAWS"), out var drawCount) && drawCount > 0 ? drawCount : 256;

    // SHARPEMU_DRAW_RECORD_BENCH_FILL=images=760,windows=300,overlay=400,writes=400,dccfills=500 pre-fills the host's
    // state to a title's size before the draws are measured: cached images over distinct 1 MiB-spaced ranges, guest
    // windows the draws' vertex buffers are spread over (needs SHARPEMU_SCENE_MEMORY_MB of about 33 per window),
    // pending command-processor writes, GPU write ranges, and recorded DCC fills.
    private static readonly Dictionary<string, int> Fill =
        (Environment.GetEnvironmentVariable("SHARPEMU_DRAW_RECORD_BENCH_FILL") ?? "")
        .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
        .Select(item => item.Split('='))
        .Where(parts => parts.Length == 2 && int.TryParse(parts[1], out _))
        .ToDictionary(parts => parts[0], parts => int.Parse(parts[1]));

    // Three images sampled through one sampler and summed (T# s[0:7], s[8:15], s[16:23], S# s[24:27]; the
    // primitive mask is s28), the way a material pass reads several textures.
    internal static readonly uint[] SampledThree =
    [
        0xBEFC031C,                // s_mov_b32 m0, s28
        0xC8080000,                // v_interp_p1_f32_e32 v2, v0, attr0.x
        0xC8090001,                // v_interp_p2_f32_e32 v2, v1, attr0.x
        0xC80C0100,                // v_interp_p1_f32_e32 v3, v0, attr0.y
        0xC80D0101,                // v_interp_p2_f32_e32 v3, v1, attr0.y
        0xF0800F08, 0x00C00402,    // image_sample v[4:7], v[2:3], s[0:7], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xF0800F08, 0x00C20802,    // image_sample v[8:11], v[2:3], s[8:15], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xF0800F08, 0x00C40C02,    // image_sample v[12:15], v[2:3], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C3F70,                // s_waitcnt vmcnt(0)
        0x06080908, 0x060A0B09, 0x060C0D0A, 0x060E0F0B,    // v_add_f32 v4..v7, v8..v11, v4..v7
        0x06080D08, 0x060A0F09, 0x060C110A, 0x060E130B,    // v_add_f32 v4..v7, v12..v15, v4..v7
        0xF800180F, 0x07060504,    // exp mrt0 v4, v5, v6, v7 done vm
        0xBF810000,                // s_endpgm
    ];

    // The same program under another hash: k s_nop instructions in front of it.
    private static uint[] SampledThreeNops(int count) => [.. Enumerable.Repeat(0xBF800000u, count), .. SampledThree];

    private static readonly Func<RenderPhaseProfile.Phase, long> PhaseTicks =
        typeof(RenderPhaseProfile).GetMethod("PhaseTicks", BindingFlags.Static | BindingFlags.NonPublic)!
            .CreateDelegate<Func<RenderPhaseProfile.Phase, long>>();

    // CLOCK_THREAD_CPUTIME_ID: the recording thread's own CPU time, which a busy machine does not stretch the way it does wall time.
    [DllImport("libSystem.dylib")]
    private static extern ulong clock_gettime_nsec_np(int clockId);

    private static ulong ThreadCpuNanoseconds() => clock_gettime_nsec_np(16);

    // ReportCounters zeroes the host's render-pass count every two seconds, which is what keeps its first-pass diagnostics
    // (strings built per pass while the count is under 4096) running in the emulator; the bench does the same per round.
    private static readonly System.Reflection.FieldInfo RenderPasses =
        typeof(MetalRenderHost).GetField("_renderPasses", BindingFlags.Instance | BindingFlags.NonPublic)!;

    private static ulong SolidTexture(SceneRig rig, uint abgr)
    {
        var pitchTexels = SceneDraws.LinearPitchBytes(TextureSize, 4) / 4;
        var texels = new uint[pitchTexels * TextureSize];
        Array.Fill(texels, abgr);
        var address = rig.Memory.Allocate((ulong)texels.Length * 4, 0x10000);
        rig.Memory.Write(address, MemoryMarshal.AsBytes(texels.AsSpan()));
        return address;
    }

    // Same: one state, 256 draws. AlternatingTexture: texture 0 flips every draw. AlternatingProgram: sixteen
    // pixel programs in turn. PassChurn: two colour targets, a new render pass every four draws. Mixed: all three.
    // Production: Mixed with an occlusion dump after every draw and an end-of-pipe label after every eighth, as a
    // title's frame has them.
    private enum Variation { Same, AlternatingTexture, AlternatingProgram, PassChurn, Mixed, Production }

    private static readonly MethodInfo PrefetchCounters =
        typeof(MetalRenderHost).GetMethod("FormatPrefetchCounters", BindingFlags.Instance | BindingFlags.NonPublic)!;

    private static readonly FieldInfo CpOverlayField =
        typeof(MetalRenderHost).GetField("_cpOverlay", BindingFlags.Instance | BindingFlags.NonPublic)!;
    private static readonly FieldInfo GpuWritesField =
        typeof(MetalRenderHost).GetField("_gpuWrites", BindingFlags.Instance | BindingFlags.NonPublic)!;
    private static readonly MethodInfo RecordDccFill =
        typeof(MetalImageCache).GetMethod("RecordDccFillLocked", BindingFlags.Instance | BindingFlags.NonPublic)!;

    // SHARPEMU_DRAW_RECORD_BENCH_OUTPUT=<path>: microseconds per recorded draw for each variation, and
    // with SHARPEMU_PROFILE_RENDER=1 the self time per render phase. Draws record on this thread
    // (no command thread), so the number is the worker's CPU for packets plus RecordDraw. _ROUNDS and _WARM set
    // the measured and warm-up submissions, _VARIATION keeps one case (a native sampler wants a long one).
    [Fact]
    public void MeasureDrawRecordWhenRequested()
    {
        var path = Environment.GetEnvironmentVariable("SHARPEMU_DRAW_RECORD_BENCH_OUTPUT");
        if (path is null) return;
        Environment.SetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE", "1");
        using var rig = CreateRig();
        Assert.NotNull(rig);
        var colors = new[] { rig.ColorTarget(TargetSize, TargetSize, 16, 0), rig.ColorTarget(TargetSize, TargetSize, 16, 0) };
        var depth = rig.Memory.Allocate(TargetSize * TargetSize * 4 + 0x10000, 0x10000);
        var textures = Enumerable.Range(0, 10).Select(index => SolidTexture(rig, 0x80000040u + (uint)index * 0x00101000u)).ToArray();
        var meshData = rig.UploadMesh(SceneDraws.TexturedQuad());
        // One mesh per guest window (32 MiB granules): draw n fetches from window n modulo the count.
        var windowMeshes = Enumerable.Range(0, Fill.GetValueOrDefault("windows")).Select(_ =>
        {
            rig.Memory.Allocate(1, 32UL * 1024 * 1024);
            return rig.UploadMesh(SceneDraws.TexturedQuad());
        }).ToArray();
        var queries = rig.Memory.Allocate((ulong)DrawsPerSubmit * 256, 256);
        var labels = rig.Memory.Allocate((ulong)DrawsPerSubmit * 16, 256);
        var vertexProgram = rig.Program(ScenePrograms.FetchVertex);
        var programs = Enumerable.Range(0, 16).Select(count => rig.Program(SampledThreeNops(count))).ToArray();
        var rounds = int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_DRAW_RECORD_BENCH_ROUNDS"), out var configured) ? configured : 60;
        var warm = int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_DRAW_RECORD_BENCH_WARM"), out var warmed) ? warmed : 400;
        var only = Environment.GetEnvironmentVariable("SHARPEMU_DRAW_RECORD_BENCH_VARIATION");
        var phases = Enum.GetValues<RenderPhaseProfile.Phase>().Where(phase => phase != RenderPhaseProfile.Phase.Count).ToArray();

        uint[] Stream(Variation variation)
        {
            var mixed = variation is Variation.Mixed or Variation.Production;
            var textureChanges = variation is Variation.AlternatingTexture || mixed;
            var programChanges = variation is Variation.AlternatingProgram || mixed;
            var passChanges = variation is Variation.PassChurn || mixed;
            var passLength = mixed ? 8 : 4;
            var pm4 = new ScenePm4();
            for (var draw = 0; draw < DrawsPerSubmit; draw++)
            {
                var texture = textures[textureChanges ? (mixed ? draw % 7 : draw & 1) : 0];
                var program = programs[programChanges ? (mixed ? draw % 5 : draw) % programs.Length : 0];
                var color = colors[passChanges ? draw / passLength & 1 : 0];
                var mesh = windowMeshes.Length != 0 ? windowMeshes[draw % windowMeshes.Length] : meshData;
                uint[] pixelData =
                [
                    .. Descriptors.Texture(texture, Rgba8, TextureSize, TextureSize),
                    .. Descriptors.Texture(textures[8], Rgba8, TextureSize, TextureSize),
                    .. Descriptors.Texture(textures[9], Rgba8, TextureSize, TextureSize),
                    .. Descriptors.Sampler(),
                ];
                if (draw == 0 || (passChanges && draw % passLength == 0))
                {
                    var full = new SceneState(TargetSize, TargetSize)
                        .ColorTarget(0, color, TargetSize, TargetSize, format: 14, numberType: 7)
                        .DepthTarget(depth, TargetSize, TargetSize, zFormat: 3)
                        .DepthControl(test: true, write: true, CompareFunc.Always)
                        .VertexShader(vertexProgram, mesh)
                        .PixelShader(program, pixelData);
                    if (variation == Variation.Production) full.Context[Cx.DbCountControl] = 1u << 8;
                    full.Emit(pm4);
                }
                else if (textureChanges || programChanges || windowMeshes.Length != 0)
                {
                    // A guest rewrites the pixel stage's registers when its material changes, nothing else.
                    var changed = new SceneState(TargetSize, TargetSize);
                    changed.Context.Clear();
                    changed.Uconfig.Clear();
                    changed.Shader.Clear();
                    if (windowMeshes.Length != 0) changed.VertexShader(vertexProgram, mesh);
                    changed.PixelShader(program, pixelData).Emit(pm4);
                }

                pm4.NumInstances(1);
                pm4.DrawAuto(6);
                if (variation != Variation.Production) continue;
                pm4.EventWithAddress(ScenePm4.EventPixelPipeStatDump, 1, queries + (ulong)draw * 256);
                if (draw % 8 == 7) pm4.Packet(0x49, 0x28u | (5u << 8), 1u << 29, (uint)(labels + (ulong)draw * 16), (uint)((labels + (ulong)draw * 16) >> 32), (uint)draw, 0, 0);
            }

            return pm4.ToArray();
        }

        ulong submissionId = 1000;
        if (Fill.GetValueOrDefault("images") is > 0 and var imageCount)
        {
            // Sampled once so each becomes a cached image; 1 MiB apart, which is how a title's few-megabyte targets sit.
            var fillTextures = Enumerable.Range(0, imageCount).Select(index =>
            {
                rig.Memory.Allocate(1, 1UL << 20);
                return SolidTexture(rig, 0x80000040u + (uint)index * 0x1010u);
            }).ToArray();
            var fillStream = new ScenePm4();
            for (var index = 0; index < imageCount; index += 3)
            {
                uint[] pixelData =
                [
                    .. Descriptors.Texture(fillTextures[index], Rgba8, TextureSize, TextureSize),
                    .. Descriptors.Texture(fillTextures[(index + 1) % imageCount], Rgba8, TextureSize, TextureSize),
                    .. Descriptors.Texture(fillTextures[(index + 2) % imageCount], Rgba8, TextureSize, TextureSize),
                    .. Descriptors.Sampler(),
                ];
                var state = new SceneState(TargetSize, TargetSize);
                if (index == 0)
                {
                    state.ColorTarget(0, colors[0], TargetSize, TargetSize, format: 14, numberType: 7)
                        .DepthTarget(depth, TargetSize, TargetSize, zFormat: 3)
                        .DepthControl(test: true, write: true, CompareFunc.Always)
                        .VertexShader(vertexProgram, meshData);
                }
                else
                {
                    state.Context.Clear();
                    state.Uconfig.Clear();
                    state.Shader.Clear();
                }

                state.PixelShader(programs[0], pixelData).Emit(fillStream);
                fillStream.NumInstances(1);
                fillStream.DrawAuto(6);
            }

            for (var pass = 0; pass < 2; pass++) rig.Submit(fillStream);
            Output.WriteLine($"fill: {imageCount} images, host reports {rig.Host.ImageCache.ImageCount} cached");
        }

        // 16 MiB each for the overlay entries, the write ranges and the DCC fills, away from every address a draw uses.
        var fillRegion = rig.Memory.Allocate(3UL << 24, 1UL << 20);
        const ulong NeverCompleted = ulong.MaxValue / 2;
        var overlayCount = Fill.GetValueOrDefault("overlay");
        if (overlayCount > 0)
        {
            var overlay = (MetalPendingWriteOverlay)CpOverlayField.GetValue(rig.Host)!;
            Span<byte> value = stackalloc byte[8];
            for (var index = 0; index < overlayCount; index++) overlay.Add(fillRegion + (ulong)index * 256, value, NeverCompleted);
        }

        var writeCount = Fill.GetValueOrDefault("writes");
        if (writeCount > 0)
        {
            var writes = (MetalGpuWriteRanges)GpuWritesField.GetValue(rig.Host)!;
            for (var index = 0; index < writeCount; index++)
                writes.Note(fillRegion + (1UL << 24) + (ulong)index * 0x2000, 0x1000, NeverCompleted + (ulong)index);
        }

        for (var index = 0; index < Fill.GetValueOrDefault("dccfills"); index++)
            RecordDccFill.Invoke(rig.Host.ImageCache, [fillRegion + (2UL << 24) + (ulong)index * 0x2000, 0x1000UL, (byte)0, true]);

        var results = new List<object>();
        var profiling = RenderPhaseProfile.Enabled;
        foreach (var variation in Enum.GetValues<Variation>().Where(variation => only is null || variation.ToString() == only))
        {
            rig.Host.NativeOcclusionEnabled = variation == Variation.Production;
            var words = Stream(variation);
            // The stream is placed once: every round submits the same packets.
            var streamAddress = rig.Memory.Upload<uint>(words, 256);
            var microseconds = new List<double>();
            var cpuMicroseconds = new List<double>();
            var phaseTotals = new Dictionary<string, double>();
            var hostDrawMilliseconds = 0.0;
            long allocatedBytes = 0;
            var gen0Collections = 0;
            // The first variation also pays for the JIT's tier-up of the shared recording path.
            var warmRounds = results.Count == 0 ? warm * 2 : warm;
            for (var round = 0; round < warmRounds + rounds; round++)
            {
                if (round == warmRounds) _ = PrefetchCounters.Invoke(rig.Host, null);
                RenderPasses.SetValue(rig.Host, 0L);
                var before = profiling ? phases.Select(phase => PhaseTicks(phase)).ToArray() : null;
                rig.Host.CommandStream.EnqueueGraphics(streamAddress, (uint)words.Length, ++submissionId, null);
                long started;
                ulong cpuStarted;
                var allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
                var collectionsBefore = GC.CollectionCount(0);
                using (profiling ? RenderPhaseProfile.Measure(RenderPhaseProfile.Phase.CommandStream) : default)
                {
                    cpuStarted = ThreadCpuNanoseconds();
                    started = Stopwatch.GetTimestamp();
                    rig.RecordPending();
                }

                var elapsed = Stopwatch.GetTimestamp() - started;
                var cpuElapsed = ThreadCpuNanoseconds() - cpuStarted;
                var allocated = GC.GetAllocatedBytesForCurrentThread() - allocatedBefore;
                var collections = GC.CollectionCount(0) - collectionsBefore;
                var recordedBeforeDrain = rig.Host.ReadAndResetPerfCounters().Draws;
                var drainStarted = Stopwatch.GetTimestamp();
                rig.Drain();
                var drained = Stopwatch.GetTimestamp() - drainStarted;
                if (round == warmRounds + 3) Output.WriteLine($"{variation}: {recordedBeforeDrain} draws recorded by RecordPending, drain {drained * 1e6 / Stopwatch.Frequency:F0} us");
                // The warm-up rounds fill the pipeline and image caches and let the JIT reach its optimized code.
                var counters = rig.Host.ReadAndResetPerfCounters();
                if (round < warmRounds) continue;
                Assert.Equal(DrawsPerSubmit, recordedBeforeDrain + counters.Draws);
                microseconds.Add(elapsed * 1e6 / Stopwatch.Frequency / DrawsPerSubmit);
                cpuMicroseconds.Add(cpuElapsed / 1e3 / DrawsPerSubmit);
                allocatedBytes += allocated;
                gen0Collections += collections;
                hostDrawMilliseconds += counters.DrawMs;
                if (before is null) continue;
                for (var index = 0; index < phases.Length; index++)
                {
                    var delta = PhaseTicks(phases[index]) - before[index];
                    if (delta == 0) continue;
                    var key = phases[index].ToString();
                    phaseTotals[key] = phaseTotals.GetValueOrDefault(key) + delta * 1e6 / Stopwatch.Frequency / DrawsPerSubmit;
                }
            }

            var prefetchCounters = (string)PrefetchCounters.Invoke(rig.Host, null)!;
            // FNV over both colour targets after the variation's last round, to compare the paths' results.
            var checksum = 0xCBF29CE484222325UL;
            foreach (var color in colors)
            {
                foreach (var word in rig.Read<uint>(color, (int)(TargetSize * TargetSize * 4)))
                {
                    checksum = (checksum ^ word) * 0x100000001B3UL;
                }
            }

            microseconds.Sort();
            cpuMicroseconds.Sort();
            var median = microseconds[microseconds.Count / 2];
            Output.WriteLine($"{variation}: wall median {median:F2} us/draw, p25 {microseconds[microseconds.Count / 4]:F2}, min {microseconds[0]:F2}; " +
                $"thread cpu median {cpuMicroseconds[cpuMicroseconds.Count / 2]:F2}, p25 {cpuMicroseconds[cpuMicroseconds.Count / 4]:F2}; " +
                $"{allocatedBytes / (double)(rounds * DrawsPerSubmit):F0} B/draw");
            results.Add(new
            {
                variation = variation.ToString(),
                draws = DrawsPerSubmit,
                rounds,
                medianMicrosecondsPerDraw = median,
                minMicrosecondsPerDraw = microseconds[0],
                p25MicrosecondsPerDraw = microseconds[microseconds.Count / 4],
                cpuMedianMicrosecondsPerDraw = cpuMicroseconds[cpuMicroseconds.Count / 2],
                cpuP25MicrosecondsPerDraw = cpuMicroseconds[cpuMicroseconds.Count / 4],
                maxMicrosecondsPerDraw = microseconds[^1],
                allocatedBytesPerDraw = allocatedBytes / (double)(rounds * DrawsPerSubmit),
                gen0CollectionsPerThousandDraws = gen0Collections * 1000.0 / (rounds * DrawsPerSubmit),
                hostDrawMicrosecondsPerDraw = hostDrawMilliseconds * 1000.0 / (rounds * DrawsPerSubmit),
                checksum = $"{checksum:X16}",
                counters = prefetchCounters,
                phaseMicrosecondsPerDraw = phaseTotals.Where(pair => pair.Value / rounds >= 0.005)
                    .OrderByDescending(pair => pair.Value).ToDictionary(pair => pair.Key, pair => Math.Round(pair.Value / rounds, 3)),
            });
        }

        Assert.Empty(rig.Fatals);
        File.WriteAllText(path, JsonSerializer.Serialize(
            new { arch = RuntimeInformation.ProcessArchitecture.ToString(), profiling, fill = Fill, results },
            new JsonSerializerOptions { WriteIndented = true }));
    }
}
