// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Gpu.Metal.V2;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// Draw and dispatch replay on the Metal host (MetalRenderHost.Replay.cs): a draw or dispatch that repeats the one
/// before it in all but its user data binds what that one bound, patched with its own user data, and renders or
/// writes exactly what the full path does. What moves under a sealed item between two items refuses the replay by
/// name, and the item takes the full path.
/// </summary>
public sealed class ReplaySceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Width = 16;
    private const uint Rgba8 = 56;
    private const uint TextureSize = 8;

    // MRT0 = the four user-data words s[0:3] as floats.
    private static readonly uint[] ConstantColor =
    [
        0x7E040200,                // v_mov_b32_e32 v2, s0
        0x7E060201,                // v_mov_b32_e32 v3, s1
        0x7E080202,                // v_mov_b32_e32 v4, s2
        0x7E0A0203,                // v_mov_b32_e32 v5, s3
        0xF800180F, 0x05040302,    // exp mrt0 v2, v3, v4, v5 done vm
        0xBF810000,                // s_endpgm
    ];

    // MRT0 = the 2D image T# s[0:7] sampled through S# s[8:11] at PARAM0.xy (sixteen user SGPRs: m0 is s16).
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

    // 64-thread groups; thread g writes the user-data word s4 to record g of the V# in s[0:3] (group id X is s5).
    private static readonly uint[] StoreUserWord =
    [
        0x8F068605,                // s_lshl_b32 s6, s5, 6
        0x4A020006,                // v_add_nc_u32_e32 v1, s6, v0
        0x7E040204,                // v_mov_b32_e32 v2, s4
        0xE0702000, 0x80000201,    // buffer_store_dword v2, v1, s[0:3], 0 idxen
        0xBF810000,                // s_endpgm
    ];

    private static readonly SceneVertex[] Quad =
    [
        SceneVertex.At(-1, -1, 0.5f, 0, 1, 0, 1), SceneVertex.At(1, -1, 0.5f, 1, 1, 0, 1),
        SceneVertex.At(-1, 1, 0.5f, 0, 0, 0, 1), SceneVertex.At(-1, 1, 0.5f, 0, 0, 0, 1),
        SceneVertex.At(1, -1, 0.5f, 1, 1, 0, 1), SceneVertex.At(1, 1, 0.5f, 1, 0, 0, 1),
    ];

    // One target and one mesh; each draw is a stripe of the target, its programs and user data emitted again.
    private sealed class Stripes
    {
        private readonly SceneRig _rig;
        private readonly uint[] _mesh;
        private readonly ulong _vertex;
        private readonly SceneState _state;

        public Stripes(SceneRig rig)
        {
            _rig = rig;
            _mesh = rig.UploadMesh(Quad);
            _vertex = rig.Program(ScenePrograms.FetchVertex);
            Target = rig.ColorTarget(Width, Width, 16, 0xFF);
            _state = new SceneState(Width, Width).ColorTarget(0, Target, Width, Width, format: 14, numberType: 7);
        }

        public ulong Target { get; }

        public void Draw(ScenePm4 pm4, int left, int right, uint[] pixelProgram, uint[] pixelUserData)
        {
            _state.ScreenScissor(left, 0, right, (int)Width);
            _state.VertexShader(_vertex, _mesh);
            _state.PixelShader(_rig.Program(pixelProgram), pixelUserData);
            _state.Emit(pm4);
            pm4.NumInstances(1);
            pm4.DrawAuto((uint)Quad.Length);
        }

        public float[] Pixel(int x)
        {
            var pixels = MemoryMarshal.Cast<byte, float>(_rig.ReadImage(Target, Width, Width, 16));
            return pixels.Slice((int)((8 * Width + (uint)x) * 4), 4).ToArray();
        }

        public byte[] Image() => _rig.ReadImage(Target, Width, Width, 16);
    }

    private static uint[] Colour(float r, float g, float b) => [SceneState.F(r), SceneState.F(g), SceneState.F(b), SceneState.F(1)];

    // A tiled texture (64 KiB swizzle): a cached copy, not a view of the guest window, so a CPU write must reach it
    // through the image cache. One colour throughout, so its bytes are the same in any tiling.
    private const uint TiledMode = 27;
    private const uint TextureBytes = 0x10000;

    private static uint[] TextureUserData(ulong texture) =>
        [.. Descriptors.Texture(texture, Rgba8, TextureSize, TextureSize, tileMode: TiledMode), .. Descriptors.Sampler(), 0, 0, 0, 0];

    private static void FillTexture(SceneRig rig, ulong address, uint abgr)
    {
        var texels = new uint[TextureBytes / 4];
        Array.Fill(texels, abgr);
        rig.Memory.Write(address, MemoryMarshal.AsBytes(texels.AsSpan()));
    }

    private static ulong SolidTexture(SceneRig rig, uint abgr)
    {
        var address = rig.Memory.Allocate(TextureBytes, 0x10000);
        FillTexture(rig, address, abgr);
        return address;
    }

    private static void AssertColour(string what, float[] actual, float r, float g, float b)
    {
        float[] expected = [r, g, b, 1];
        for (var channel = 0; channel < 4; channel++)
        {
            Assert.True(
                MathF.Abs(actual[channel] - expected[channel]) <= 0.01f,
                $"{what}: channel {channel} is {actual[channel]}, expected {expected[channel]}");
        }
    }

    private static void AssertAbgr(string what, float[] actual, uint abgr) =>
        AssertColour(what, actual, (abgr & 0xFF) / 255f, ((abgr >> 8) & 0xFF) / 255f, ((abgr >> 16) & 0xFF) / 255f);

    private void Report(SceneRig rig, string what)
    {
        var (draws, dispatches, rebound) = rig.Host.ReplayCounts;
        Output.WriteLine($"{what}: draws={draws} dispatches={dispatches} rebound={rebound} " +
            $"draw_misses=[{string.Join(',', Names().Select(name => $"{name}:{rig.Host.ReplayCount(false, name)}"))}] " +
            $"dispatch_misses=[{string.Join(',', Names().Select(name => $"{name}:{rig.Host.ReplayCount(true, name)}"))}]");
        static string[] Names() =>
        [
            "unsealed", "serial", "scope", "stages", "images", "buffers", "vertex", "patch", "commit", "epoch_moved",
            "not_in_place", "argument_field", "unstamped", "vertex_map", "written", "traces", "memo_off",
        ];
    }

    private static readonly (float R, float G, float B)[] StripeColours =
    [
        (1, 0, 0), (1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), (0.25f, 0.5f, 0.75f),
    ];

    private byte[] DrawStripes(bool replay, out long replays)
    {
        using var rig = CreateRig()!;
        rig.Host.DrawReplayEnabled = replay;
        rig.Host.DispatchReplayEnabled = replay;
        var stripes = new Stripes(rig);
        var pm4 = new ScenePm4();
        for (var stripe = 0; stripe < StripeColours.Length; stripe++)
        {
            var (r, g, b) = StripeColours[stripe];
            stripes.Draw(pm4, stripe * 2, stripe * 2 + 2, ConstantColor, Colour(r, g, b));
        }

        rig.Submit(pm4);
        for (var stripe = 0; stripe < StripeColours.Length; stripe++)
        {
            var (r, g, b) = StripeColours[stripe];
            AssertColour($"replay={replay} stripe {stripe}", stripes.Pixel(stripe * 2 + 1), r, g, b);
        }

        Report(rig, $"replay={replay}");
        replays = rig.Host.ReplayCounts.Draws;
        Assert.Empty(rig.Fatals);
        return stripes.Image();
    }

    // Draws that differ only in their user data and scissor replay the one before them, each with its own user data,
    // and the target is byte for byte what the full path renders.
    [Fact]
    public void RepeatedDrawsReplayWithTheirOwnUserDataAndRenderAsWithout()
    {
        if (CreateRig() is not { } probe) return;
        probe.Dispose();
        var with = DrawStripes(replay: true, out var replays);
        var without = DrawStripes(replay: false, out var none);
        Assert.True(replays > 0, "no draw was replayed");
        Assert.Equal(0, none);
        Assert.Equal(without, with);
    }

    // A cached (tiled) texture the sealed draw sampled is written by the CPU between two draws: its binding epoch moves,
    // the replay is refused (images) and the draw acquires the texture again, sampling the new texels. A replay here
    // would bind the copy as it was, never refreshed.
    [Fact]
    public void ATextureWrittenBetweenDrawsRefusesTheReplayAndIsSampledNew()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        const uint red = 0xFF0000FF;
        const uint green = 0xFF00FF00;
        var texture = SolidTexture(rig, red);
        var stripes = new Stripes(rig);
        var calls = 0;
        rig.Host.ReplayProbe = dispatch =>
        {
            if (!dispatch && ++calls == 2)
            {
                var before = rig.Host.ImageCache.Epochs.Binding;
                FillTexture(rig, texture, green);
                rig.Host.ImageCache.InvalidateMemory(texture, TextureBytes);
                Output.WriteLine($"epoch {before} -> {rig.Host.ImageCache.Epochs.Binding}");
            }
        };
        var pm4 = new ScenePm4();
        for (var stripe = 0; stripe < 4; stripe++)
        {
            stripes.Draw(pm4, stripe * 4, stripe * 4 + 4, Sample, TextureUserData(texture));
        }

        rig.Submit(pm4);
        Report(rig, "texture written");
        // Stripes 0 and 1 are not looked at: the CPU rewrote the texels under GPU work that was not finished.
        AssertAbgr("stripe 2 (refused, full)", stripes.Pixel(10), green);
        AssertAbgr("stripe 3", stripes.Pixel(14), green);
        Assert.True(calls >= 2, $"the host was asked for {calls} replays");
        Assert.True(rig.Host.ReplayCount(false, "images") >= 1, "the moved epoch did not refuse the replay");
        Assert.True(rig.Host.ReplayCounts.Draws >= 1, "the draw before the refused one was not replayed");
        Assert.Empty(rig.Fatals);
    }

    // A read-only binding of the sealed draw no longer answered by the read-binding memo refuses the replay (buffers):
    // nothing then says the full chain would bind the same range.
    [Fact]
    public void AReadBindingTheMemoNoLongerAnswersRefusesTheReplay()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var stripes = new Stripes(rig);
        var calls = 0;
        rig.Host.ReplayProbe = dispatch =>
        {
            if (!dispatch && ++calls == 2)
            {
                rig.Host.ReadBindingMemoEnabled = false;
            }
        };
        var pm4 = new ScenePm4();
        for (var stripe = 0; stripe < 4; stripe++)
        {
            stripes.Draw(pm4, stripe * 4, stripe * 4 + 4, ConstantColor, Colour(0, stripe / 4f, 1));
        }

        rig.Submit(pm4);
        Report(rig, "memo off");
        for (var stripe = 0; stripe < 4; stripe++)
        {
            AssertColour($"stripe {stripe}", stripes.Pixel(stripe * 4 + 2), 0, stripe / 4f, 1);
        }

        Assert.Equal(1, rig.Host.ReplayCounts.Draws);
        Assert.True(rig.Host.ReplayCount(false, "buffers") >= 1, "a read binding the memo did not answer was replayed");
        Assert.True(rig.Host.ReplayCount(false, "memo_off") >= 1, "a draw was captured with the memo off");
        Assert.Empty(rig.Fatals);
    }

    // The command buffer goes out between two draws (a host read waiting for the GPU): the argument and push ranges
    // the sealed draw bound belong to the old one, so the replay is refused (serial) and the draw binds in full.
    [Fact]
    public void ACommandBufferCommittedBetweenDrawsRefusesTheReplay()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        var stripes = new Stripes(rig);
        var calls = 0;
        rig.Host.ReplayProbe = dispatch =>
        {
            if (!dispatch && ++calls == 2)
            {
                rig.Host.FlushAndWaitForGpu();
            }
        };
        var pm4 = new ScenePm4();
        for (var stripe = 0; stripe < 4; stripe++)
        {
            stripes.Draw(pm4, stripe * 4, stripe * 4 + 4, ConstantColor, Colour(stripe / 4f, 1, 0.5f));
        }

        rig.Submit(pm4);
        Report(rig, "flushed");
        for (var stripe = 0; stripe < 4; stripe++)
        {
            AssertColour($"stripe {stripe}", stripes.Pixel(stripe * 4 + 2), stripe / 4f, 1, 0.5f);
        }

        Assert.True(rig.Host.ReplayCount(false, "serial") >= 1, "no replay was refused across a committed command buffer");
        Assert.True(rig.Host.ReplayCounts.Draws >= 1, "the draw before the refused one was not replayed");
        Assert.Empty(rig.Fatals);
    }

    private uint[] StoreWords(bool replay, Action<SceneRig>? probe, out long replays, out long imagesMisses)
    {
        using var rig = CreateRig()!;
        rig.Host.DrawReplayEnabled = replay;
        rig.Host.DispatchReplayEnabled = replay;
        var calls = 0;
        if (probe is not null)
        {
            rig.Host.ReplayProbe = dispatch =>
            {
                if (dispatch && ++calls == 2)
                {
                    probe(rig);
                }
            };
        }

        var first = rig.Memory.Allocate(64 * 4, 0x10000);
        var second = rig.Memory.Allocate(64 * 4, 0x10000);
        var pm4 = new ScenePm4();
        // Four dispatches into the first buffer, then three into the second: the last word of each run is what lands.
        uint[] words = [1, 2, 3, 4];
        foreach (var word in words)
        {
            rig.Dispatch(pm4, StoreUserWord, [.. Descriptors.Buffer(first, 4, 64), word], 1);
        }

        foreach (var word in words[..3])
        {
            rig.Dispatch(pm4, StoreUserWord, [.. Descriptors.Buffer(second, 4, 64), word + 10], 1);
        }

        rig.Submit(pm4);
        Report(rig, $"dispatch replay={replay}");
        replays = rig.Host.ReplayCounts.Dispatches;
        imagesMisses = rig.Host.ReplayCount(true, "images");
        Assert.Empty(rig.Fatals);
        return [.. rig.Read<uint>(first, 64), .. rig.Read<uint>(second, 64)];
    }

    // Dispatches that differ only in their user data replay the one before them with their own shader data, and take
    // their written buffer's chain again: the bytes written are those of the full path.
    [Fact]
    public void RepeatedDispatchesReplayWithTheirOwnUserDataAndWriteAsWithout()
    {
        if (CreateRig() is not { } probe) return;
        probe.Dispose();
        var with = StoreWords(replay: true, probe: null, out var replays, out _);
        var without = StoreWords(replay: false, probe: null, out var none, out _);
        Assert.All(with[..64], word => Assert.Equal(4u, word));
        Assert.All(with[64..], word => Assert.Equal(13u, word));
        Assert.True(replays > 0, "no dispatch was replayed");
        Assert.Equal(0, none);
        Assert.Equal(without, with);
    }

    // An image-cache binding epoch that moves between two dispatches refuses the replay (images); the dispatch binds
    // in full and writes what it would have.
    [Fact]
    public void AnEpochMovedBetweenDispatchesRefusesTheReplay()
    {
        if (CreateRig() is not { } probe) return;
        probe.Dispose();
        var with = StoreWords(replay: true, rig => rig.Host.ImageCache.Epochs.NoteShapeChanged(), out var replays, out var imagesMisses);
        Assert.True(imagesMisses >= 1, "the moved epoch did not refuse the replay");
        Assert.True(replays >= 1, "no dispatch was replayed around the refused one");
        Assert.All(with[..64], word => Assert.Equal(4u, word));
        Assert.All(with[64..], word => Assert.Equal(13u, word));
    }
}
