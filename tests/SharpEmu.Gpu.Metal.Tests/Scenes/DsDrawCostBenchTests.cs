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
/// The worker's per-draw and per-dispatch recording cost on draws and dispatches shaped like Demon's Souls' (their
/// binding counts taken from its translated programs): the vertex stage fetches four vertex buffers whose V#s it
/// loads from a table, plus an object-constant V# in user data and a table one; the pixel stage samples twelve 2D
/// textures and two 2D arrays through five samplers, every T# and S# loaded from a material table, and reads a
/// constant buffer; draws export to five or six attachments. Seven in ten draws are indexed, two indexed indirect,
/// one auto. A submission is twelve passes over three target sets (alternately five and six attachments). Dispatches
/// load five sampled T#s, two storage T#s, two S#s and four V#s from a table of their own; each stores two images and
/// a buffer of its own and samples what the dispatch sixteen before stored, a CS partial flush after every sixteenth.
/// </summary>
public sealed class DsDrawCostBenchTests(ITestOutputHelper output) : SceneTestBase(output)
{
    private const uint Rgba8 = 56;
    private const uint Rgba32F = 77;
    private const uint TextureSize = 8;
    private const uint TargetSize = 64;
    private const uint StorageEdge = 16;
    private const int Passes = 12;
    private const int TargetSets = 3;
    private const int Meshes = 24;
    private const int Materials = 48;
    private const int SampledTextures = 32;
    private const int ArrayTextures = 4;
    private const int StorageImages = 16;

    // s[8:9] the vertex table (four vertex-buffer V#s, then the table's constant V#), s[12:15] the object constants.
    // Assembled with llvm-mc -triple=amdgcn -mcpu=gfx1030 -show-encoding.
    internal static readonly uint[] TableVertex =
    [
        0xF4100404, 0xFA000000,    // s_load_dwordx16 s[16:31], s[8:9], 0x0
        0xF4080804, 0xFA000040,    // s_load_dwordx4 s[32:35], s[8:9], 0x40
        0xBF8CC07F,                // s_waitcnt lgkmcnt(0)
        0xE0382000, 0x80040005,    // buffer_load_dwordx4 v[0:3], v5, s[16:19], 0 idxen
        0xE0382000, 0x80050A05,    // buffer_load_dwordx4 v[10:13], v5, s[20:23], 0 idxen
        0xE0382000, 0x80060E05,    // buffer_load_dwordx4 v[14:17], v5, s[24:27], 0 idxen
        0xE0382000, 0x80071205,    // buffer_load_dwordx4 v[18:21], v5, s[28:31], 0 idxen
        0xF4280910, 0xFA000000,    // s_buffer_load_dwordx4 s[36:39], s[32:35], 0x0
        0xF4280A06, 0xFA000010,    // s_buffer_load_dwordx4 s[40:43], s[12:15], 0x10
        0xBF8C0070,                // s_waitcnt vmcnt(0) lgkmcnt(0)
        0x06141D0A,                // v_add_f32_e32 v10, v10, v14
        0x06161F0B,                // v_add_f32_e32 v11, v11, v15
        0x0618250C,                // v_add_f32_e32 v12, v12, v18
        0x061A270D,                // v_add_f32_e32 v13, v13, v19
        0x06000024,                // v_add_f32_e32 v0, s36, v0
        0x06020225,                // v_add_f32_e32 v1, s37, v1
        0x06040428,                // v_add_f32_e32 v2, s40, v2
        0xF80008CF, 0x03020100,    // exp pos0, v0, v1, v2, v3 done
        0xF800020F, 0x0D0C0B0A,    // exp param0, v10, v11, v12, v13
        0xBF810000,                // s_endpgm
    ];

    // s[0:1] the material table (T#s 0-13 at 0x0, the last two 2D arrays; S#s 0-4 at 0x1C0), s[4:7] the constants
    // V#; eight user SGPRs, so the primitive mask is s8. Fourteen samples summed and scaled by constant 0.
    private static uint[] TablePixel(int targets)
    {
        uint[] head =
        [
            0xBEFC0308,                // s_mov_b32 m0, s8
            0xC8080000,                // v_interp_p1_f32_e32 v2, v0, attr0.x
            0xC8090001,                // v_interp_p2_f32_e32 v2, v1, attr0.x
            0xC80C0100,                // v_interp_p1_f32_e32 v3, v0, attr0.y
            0xC80D0101,                // v_interp_p2_f32_e32 v3, v1, attr0.y
            0xF4100B00, 0xFA0001C0,    // s_load_dwordx16 s[44:59], s[0:1], 0x1c0
            0xF4081000, 0xFA000200,    // s_load_dwordx4 s[64:67], s[0:1], 0x200
            0xF4281102, 0xFA000000,    // s_buffer_load_dwordx4 s[68:71], s[4:7], 0x0
            0x7E280280,                // v_mov_b32_e32 v20, 0
            0x7E2A0280,                // v_mov_b32_e32 v21, 0
            0x7E2C0280,                // v_mov_b32_e32 v22, 0
            0x7E2E0280,                // v_mov_b32_e32 v23, 0
            0x7E300302,                // v_mov_b32_e32 v24, v2
            0x7E320303,                // v_mov_b32_e32 v25, v3
            0x7E340280,                // v_mov_b32_e32 v26, 0
        ];
        var code = new List<uint>(head);
        for (var group = 0; group < 4; group++)
        {
            var count = Math.Min(4, 14 - group * 4);
            for (var slot = 0; slot < count; slot++)
            {
                // s_load_dwordx8 s[12 + 8 * slot : ...], s[0:1], texture * 32
                var texture = group * 4 + slot;
                code.Add(0xF40C0000u | ((uint)(12 + 8 * slot) << 6));
                code.Add(0xFA000000u | (uint)(texture * 32));
            }

            code.Add(0xBF8CC07F);    // s_waitcnt lgkmcnt(0)
            for (var slot = 0; slot < count; slot++)
            {
                // image_sample v[4 + 4 * slot : ...], coordinates, T# s[12 + 8 * slot], S# texture % 5 (s[44 + 4 * n], the fifth s[64:67])
                var texture = group * 4 + slot;
                var sampler = texture % 5;
                var samplerBase = sampler < 4 ? 44 + 4 * sampler : 64;
                var array = texture >= 12;
                code.Add(array ? 0xF0800F28u : 0xF0800F08u);
                code.Add((uint)(array ? 24 : 2) | ((uint)(4 + 4 * slot) << 8) | ((uint)(12 + 8 * slot) / 4 << 16) |
                    ((uint)samplerBase / 4 << 21));
            }

            code.Add(0xBF8C3F70);    // s_waitcnt vmcnt(0)
            for (var slot = 0; slot < count; slot++)
            {
                for (var channel = 0u; channel < 4; channel++)
                {
                    // v_add_f32_e32 v(20 + c), v(4 + 4 * slot + c), v(20 + c)
                    code.Add(0x06000000u | ((20 + channel) << 17) | ((20 + channel) << 9) | (256 + 4 + 4 * (uint)slot + channel));
                }
            }
        }

        for (var channel = 0u; channel < 4; channel++)
        {
            code.Add(0x10000000u | ((20 + channel) << 17) | ((20 + channel) << 9) | 68);    // v_mul_f32_e32 v(20 + c), s68, v(20 + c)
        }

        uint[] sources = [0x17161514, 0x07060504, 0x0B0A0908, 0x0F0E0D0C, 0x07060504, 0x17161514];
        for (var target = 0; target < targets; target++)
        {
            // exp mrt<n> (the last done vm)
            code.Add(0xF800000Fu | ((uint)target << 4) | (target == targets - 1 ? 0x1800u : 0u));
            code.Add(sources[target]);
        }

        code.Add(0xBF810000);    // s_endpgm
        return [.. code];
    }

    // s[0:1] the dispatch's table: sampled T#s 0-4 at 0x0, storage T#s at 0xA0 and 0xC0, S#s at 0xE0, V#s at 0x100
    // (three read, the fourth written); group ids s2/s3, 8x8 threads.
    internal static readonly uint[] TableCompute =
    [
        0xF4100100, 0xFA000000,    // s_load_dwordx16 s[4:19], s[0:1], 0x0
        0xF4100500, 0xFA000040,    // s_load_dwordx16 s[20:35], s[0:1], 0x40
        0xF40C0900, 0xFA000080,    // s_load_dwordx8 s[36:43], s[0:1], 0x80
        0xF4100B00, 0xFA0000A0,    // s_load_dwordx16 s[44:59], s[0:1], 0xa0
        0xF40C0F00, 0xFA0000E0,    // s_load_dwordx8 s[60:67], s[0:1], 0xe0
        0xF4101100, 0xFA000100,    // s_load_dwordx16 s[68:83], s[0:1], 0x100
        0x8F548302,                // s_lshl_b32 s84, s2, 3
        0x8F558303,                // s_lshl_b32 s85, s3, 3
        0x4A040054,                // v_add_nc_u32_e32 v2, s84, v0
        0x4A060255,                // v_add_nc_u32_e32 v3, s85, v1
        0x7E080D02,                // v_cvt_f32_u32_e32 v4, v2
        0x7E0A0D03,                // v_cvt_f32_u32_e32 v5, v3
        0x060808F0,                // v_add_f32_e32 v4, 0.5, v4
        0x060A0AF0,                // v_add_f32_e32 v5, 0.5, v5
        0x100808FF, 0x3D800000,    // v_mul_f32_e32 v4, 0x3d800000, v4
        0x100A0AFF, 0x3D800000,    // v_mul_f32_e32 v5, 0x3d800000, v5
        0xD746001A, 0x04090903,    // v_lshl_add_u32 v26, v3, 4, v2
        0xBF8CC07F,                // s_waitcnt lgkmcnt(0)
        0xF09C0F08, 0x01E10604,    // image_sample_lz v[6:9], v[4:5], s[4:11], s[60:63] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xF09C0F08, 0x02030A04,    // image_sample_lz v[10:13], v[4:5], s[12:19], s[64:67] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xF09C0F08, 0x01E50E04,    // image_sample_lz v[14:17], v[4:5], s[20:27], s[60:63] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xF09C0F08, 0x02071204,    // image_sample_lz v[18:21], v[4:5], s[28:35], s[64:67] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xF09C0F08, 0x01E91604,    // image_sample_lz v[22:25], v[4:5], s[36:43], s[60:63] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xE0302000, 0x80111B1A,    // buffer_load_dword v27, v26, s[68:71], 0 idxen
        0xE0302000, 0x80121C1A,    // buffer_load_dword v28, v26, s[72:75], 0 idxen
        0xE0302000, 0x80131D1A,    // buffer_load_dword v29, v26, s[76:79], 0 idxen
        0xBF8C3F70,                // s_waitcnt vmcnt(0)
        0x060C1506,                // v_add_f32_e32 v6, v6, v10
        0x060E1707,                // v_add_f32_e32 v7, v7, v11
        0x06101908,                // v_add_f32_e32 v8, v8, v12
        0x06121B09,                // v_add_f32_e32 v9, v9, v13
        0x060C1D06,                // v_add_f32_e32 v6, v6, v14
        0x060E1F07,                // v_add_f32_e32 v7, v7, v15
        0x06102108,                // v_add_f32_e32 v8, v8, v16
        0x06122309,                // v_add_f32_e32 v9, v9, v17
        0x060C2506,                // v_add_f32_e32 v6, v6, v18
        0x060E2707,                // v_add_f32_e32 v7, v7, v19
        0x06102908,                // v_add_f32_e32 v8, v8, v20
        0x06122B09,                // v_add_f32_e32 v9, v9, v21
        0x100C0CF0,                // v_mul_f32_e32 v6, 0.5, v6
        0x100E0EF0,                // v_mul_f32_e32 v7, 0.5, v7
        0x06102D08,                // v_add_f32_e32 v8, v8, v22
        0x06122F09,                // v_add_f32_e32 v9, v9, v23
        0x4A36391B,                // v_add_nc_u32_e32 v27, v27, v28
        0x4A363B1B,                // v_add_nc_u32_e32 v27, v27, v29
        0xF0201F08, 0x000B0602,    // image_store v[6:9], v[2:3], s[44:51] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
        0xF0201F08, 0x000D1602,    // image_store v[22:25], v[2:3], s[52:59] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
        0xE0702000, 0x80141B1A,    // buffer_store_dword v27, v26, s[80:83], 0 idxen
        0xBF810000,                // s_endpgm
    ];

    private static uint[] WithNops(uint[] program, int count) => [.. Enumerable.Repeat(0xBF800000u, count), .. program];

    private static readonly Func<RenderPhaseProfile.Phase, long> PhaseTicks =
        typeof(RenderPhaseProfile).GetMethod("PhaseTicks", BindingFlags.Static | BindingFlags.NonPublic)!
            .CreateDelegate<Func<RenderPhaseProfile.Phase, long>>();

    private static readonly FieldInfo RenderPasses =
        typeof(MetalRenderHost).GetField("_renderPasses", BindingFlags.Instance | BindingFlags.NonPublic)!;

    private static readonly MethodInfo PrefetchCounters =
        typeof(MetalRenderHost).GetMethod("FormatPrefetchCounters", BindingFlags.Instance | BindingFlags.NonPublic)!;

    [DllImport("libSystem.dylib")]
    private static extern ulong clock_gettime_nsec_np(int clockId);

    // CLOCK_THREAD_CPUTIME_ID: the recording thread's own CPU time.
    private static ulong ThreadCpuNanoseconds() => clock_gettime_nsec_np(16);

    private enum Variation { Draws, Dispatches }

    private static ulong Texels(SceneRig rig, uint width, uint height, uint layers, uint bytesPerTexel, Func<uint, uint, uint, uint> word)
    {
        var pitch = SceneDraws.LinearPitchBytes(width, bytesPerTexel) / 4;
        var words = new uint[pitch * height * layers];
        var wordsPerTexel = Math.Max(1, bytesPerTexel / 4);
        for (var layer = 0u; layer < layers; layer++)
        for (var y = 0u; y < height; y++)
        for (var x = 0u; x < width * wordsPerTexel; x++)
            words[(layer * height + y) * pitch + x] = word(layer, y, x);
        var address = rig.Memory.Allocate((ulong)words.Length * 4, 0x10000);
        rig.Memory.Write(address, MemoryMarshal.AsBytes(words.AsSpan()));
        return address;
    }

    // SHARPEMU_DS_DRAW_BENCH_OUTPUT=<path>: microseconds per draw and per dispatch (wall and this thread's CPU, which
    // records: no command thread), bytes allocated, gen0 collections, an FNV checksum of every target and storage
    // image, and with SHARPEMU_PROFILE_RENDER=1 the self time per render phase. _DRAWS (1024) draws and _DISPATCHES
    // (256) dispatches per submission, _ROUNDS (40) after _WARM (150), _VARIATION keeps one.
    [Fact]
    public void MeasureDsShapedRecordingWhenRequested()
    {
        var path = Environment.GetEnvironmentVariable("SHARPEMU_DS_DRAW_BENCH_OUTPUT");
        if (path is null) return;
        Environment.SetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE", "1");
        int Setting(string name, int fallback) =>
            int.TryParse(Environment.GetEnvironmentVariable($"SHARPEMU_DS_DRAW_BENCH_{name}"), out var value) && value > 0 ? value : fallback;
        var drawCount = Setting("DRAWS", 1024);
        var dispatchCount = Setting("DISPATCHES", 256);
        var rounds = Setting("ROUNDS", 40);
        var warm = Setting("WARM", 150);
        var only = Environment.GetEnvironmentVariable("SHARPEMU_DS_DRAW_BENCH_VARIATION");
        using var rig = CreateRig();
        Assert.NotNull(rig);

        // Three target sets of six RGBA32F attachments and a depth target each.
        var targets = Enumerable.Range(0, TargetSets).Select(_ => Enumerable.Range(0, 6).Select(_ => rig.ColorTarget(TargetSize, TargetSize, 16)).ToArray()).ToArray();
        var depths = Enumerable.Range(0, TargetSets).Select(_ => rig.Memory.Allocate(TargetSize * TargetSize * 4 + 0x10000, 0x10000)).ToArray();
        var textures = Enumerable.Range(0, SampledTextures).Select(index =>
            Texels(rig, TextureSize, TextureSize, 1, 4, (_, y, x) => 0x40000000u + (uint)index * 0x00030507u + y * 0x0100u + x * 0x10u)).ToArray();
        var arrays = Enumerable.Range(0, ArrayTextures).Select(index =>
            Texels(rig, TextureSize, TextureSize, 2, 4, (layer, y, x) => 0x20101010u + (uint)index * 0x00050301u + layer * 0x40u + y + x * 0x100u)).ToArray();
        uint[][] samplers =
        [
            Descriptors.Sampler(),
            Descriptors.Sampler(0, 0, 0, magFilter: 1, minFilter: 1),
            Descriptors.Sampler(1, 1, 1),
            Descriptors.Sampler(2, 2, 2, magFilter: 1, minFilter: 1),
            Descriptors.Sampler(0, 2, 2),
        ];
        var pixelConstants = rig.Memory.Upload<float>([0.0625f, 0, 0, 0, 0, 0, 0, 0], 256);
        var materials = Enumerable.Range(0, Materials).Select(material =>
        {
            var table = new List<uint>();
            for (var texture = 0; texture < 12; texture++)
                table.AddRange(Descriptors.Texture(textures[(material * 7 + texture * 3) % SampledTextures], Rgba8, TextureSize, TextureSize));
            for (var texture = 0; texture < 2; texture++)
                table.AddRange(Descriptors.Texture(arrays[(material + texture) % ArrayTextures], Rgba8, TextureSize, TextureSize, type: 13, depthOrLastLayer: 1));
            for (var sampler = 0; sampler < 5; sampler++) table.AddRange(samplers[(sampler + material) % samplers.Length]);
            return rig.Memory.Upload<uint>([.. table], 256);
        }).ToArray();

        // Each mesh a quad over one cell of a 6x4 grid, four 16-byte-record vertex buffers and a constants buffer;
        // indices 0-5 and an indexed-indirect argument record.
        var objectConstants = rig.Memory.Upload<float>(new float[16], 256);
        var meshes = Enumerable.Range(0, Meshes).Select(mesh =>
        {
            var (column, row) = (mesh % 6, mesh / 6);
            var (left, right) = (-1f + column / 3f, -1f + (column + 1) / 3f);
            var (top, bottom) = (1f - row / 2f, 1f - (row + 1) / 2f);
            float[] positions = [left, bottom, 0.5f, 1, right, bottom, 0.5f, 1, left, top, 0.5f, 1, left, top, 0.5f, 1, right, bottom, 0.5f, 1, right, top, 0.5f, 1];
            float[] uvs = [0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1];
            var position = rig.Memory.Upload<float>(positions);
            var uv = rig.Memory.Upload<float>(uvs);
            var normal = rig.Memory.Upload<float>(new float[24]);
            var color = rig.Memory.Upload<float>(Enumerable.Range(0, 24).Select(index => index % 4 == 3 ? 0f : mesh / 512f).ToArray());
            var constants = rig.Memory.Upload<float>(new float[8], 256);
            var table = rig.Memory.Upload<uint>(
            [
                .. Descriptors.Buffer(position, 16, 6), .. Descriptors.Buffer(uv, 16, 6), .. Descriptors.Buffer(normal, 16, 6),
                .. Descriptors.Buffer(color, 16, 6), .. Descriptors.Buffer(constants, 16, 2),
            ], 256);
            var indices = rig.Memory.Upload<ushort>([0, 1, 2, 3, 4, 5, 0, 0], 256);
            var arguments = rig.Memory.Upload<uint>([6, 1, 0, 0, 0, 0, 0, 0], 256);
            return (table, indices, arguments);
        }).ToArray();

        var vertexPrograms = Enumerable.Range(0, 3).Select(count => rig.Program(WithNops(TableVertex, count))).ToArray();
        var pixelPrograms = new[] { 5, 6 }.Select(attachments =>
            Enumerable.Range(0, 4).Select(count => rig.Program(WithNops(TablePixel(attachments), count))).ToArray()).ToArray();

        uint[] DrawStream()
        {
            var pm4 = new ScenePm4();
            var passLength = (drawCount + Passes - 1) / Passes;
            for (var draw = 0; draw < drawCount; draw++)
            {
                var pass = draw / passLength;
                var attachments = pass % 2 == 0 ? 5 : 6;
                var set = pass % TargetSets;
                var mesh = meshes[(draw * 5 + pass) % Meshes];
                var material = materials[(draw * 11 + pass * 3) % Materials];
                uint[] vertexData = [Descriptors.Lo(mesh.table), Descriptors.Hi(mesh.table), 0, 0, .. Descriptors.Buffer(objectConstants, 16, 4)];
                uint[] pixelData = [Descriptors.Lo(material), Descriptors.Hi(material), 0, 0, .. Descriptors.Buffer(pixelConstants, 16, 2)];
                var vertexProgram = vertexPrograms[(draw / 3) % vertexPrograms.Length];
                var pixelProgram = pixelPrograms[attachments - 5][(draw / 2) % 4];
                SceneState state;
                if (draw % passLength == 0)
                {
                    state = new SceneState(TargetSize, TargetSize)
                        .DepthTarget(depths[set], TargetSize, TargetSize, zFormat: 3)
                        .DepthControl(test: true, write: true, CompareFunc.LessEqual);
                    for (var slot = 0u; slot < attachments; slot++)
                        state.ColorTarget(slot, targets[set][slot], TargetSize, TargetSize, format: 14, numberType: 7);
                    var mask = (1u << (attachments * 4)) - 1;
                    state.Context[Cx.CbTargetMask] = mask;
                    state.Context[Cx.CbShaderMask] = mask;
                    state.Context[Cx.SpiShaderColFormat] = 0x99999999u & mask;
                    pm4.Event(ScenePm4.EventCsPartialFlush);
                }
                else
                {
                    // A material change rewrites both stages' user data, a program change their programs.
                    state = new SceneState(TargetSize, TargetSize);
                    state.Context.Clear();
                    state.Uconfig.Clear();
                    state.Shader.Clear();
                }

                state.VertexShader(vertexProgram, vertexData).PixelShader(pixelProgram, pixelData).Emit(pm4);
                pm4.NumInstances(1);
                switch (draw % 10)
                {
                    case < 7:
                        pm4.IndexType(0);
                        pm4.DrawIndexed(mesh.indices, 6, 6);
                        break;
                    case < 9:
                        pm4.Raw(ScenePm4.Header(4, ScenePm4.OpSetBase), 1u, (uint)mesh.arguments, (uint)(mesh.arguments >> 32));
                        pm4.IndexType(0).Packet(ScenePm4.OpIndexBase, (uint)mesh.indices, (uint)(mesh.indices >> 32)).Packet(ScenePm4.OpIndexBufferSize, 6u);
                        pm4.Packet(ScenePm4.OpDrawIndexIndirect, 0u, 0u, 0u, 0u);
                        break;
                    default:
                        pm4.DrawAuto(6);
                        break;
                }
            }

            return pm4.ToArray();
        }

        // RGBA32F 16x16 storage images: the first sixteen dispatches sample sources, the others what dispatch n - 16
        // stored; every dispatch stores two images and a buffer no other writes, so the results do not depend on order.
        var sources = Enumerable.Range(0, StorageImages).Select(index =>
            Texels(rig, StorageEdge, StorageEdge, 1, 16, (_, y, x) => BitConverter.SingleToUInt32Bits((index + y + x) / 64f))).ToArray();
        var storage = Enumerable.Range(0, dispatchCount).Select(_ => Texels(rig, StorageEdge, StorageEdge, 1, 16, (_, _, _) => 0)).ToArray();
        var scratch = Enumerable.Range(0, dispatchCount).Select(_ => Texels(rig, StorageEdge, StorageEdge, 1, 16, (_, _, _) => 0)).ToArray();
        var readBuffers = Enumerable.Range(0, 8).Select(index =>
            rig.Memory.Upload<uint>(Enumerable.Range(0, (int)(StorageEdge * StorageEdge)).Select(word => (uint)(word * 3 + index)).ToArray(), 256)).ToArray();
        var writeBuffers = Enumerable.Range(0, dispatchCount).Select(_ => rig.Memory.Upload<uint>(new uint[StorageEdge * StorageEdge], 256)).ToArray();
        var computePrograms = Enumerable.Range(0, 4).Select(count => rig.Program(WithNops(TableCompute, count))).ToArray();

        uint[] DispatchStream()
        {
            var pm4 = new ScenePm4();
            for (var dispatch = 0; dispatch < dispatchCount; dispatch++)
            {
                var table = new List<uint>();
                table.AddRange(Descriptors.Texture(dispatch < StorageImages ? sources[dispatch] : storage[dispatch - StorageImages], Rgba32F, StorageEdge, StorageEdge));
                for (var texture = 1; texture < 5; texture++)
                    table.AddRange(Descriptors.Texture(textures[(dispatch * 3 + texture) % SampledTextures], Rgba8, TextureSize, TextureSize));
                table.AddRange(Descriptors.Texture(storage[dispatch], Rgba32F, StorageEdge, StorageEdge));
                table.AddRange(Descriptors.Texture(scratch[dispatch], Rgba32F, StorageEdge, StorageEdge));
                table.AddRange(samplers[dispatch % samplers.Length]);
                table.AddRange(samplers[(dispatch + 1) % samplers.Length]);
                for (var buffer = 0; buffer < 3; buffer++)
                    table.AddRange(Descriptors.Buffer(readBuffers[(dispatch + buffer) % readBuffers.Length], 4, StorageEdge * StorageEdge, 20));
                table.AddRange(Descriptors.Buffer(writeBuffers[dispatch], 4, StorageEdge * StorageEdge, 20));
                var address = rig.Memory.Upload<uint>([.. table], 256);
                var state = new SceneState(1, 1);
                state.Context.Clear();
                state.Uconfig.Clear();
                state.ComputeShader(computePrograms[(dispatch / 4) % computePrograms.Length], 8, 8, 1,
                    [Descriptors.Lo(address), Descriptors.Hi(address)], true, true, false, 1, 0);
                state.Emit(pm4);
                pm4.DispatchDirect(StorageEdge / 8, StorageEdge / 8, 1);
                if (dispatch % StorageImages == StorageImages - 1) pm4.Event(ScenePm4.EventCsPartialFlush);
            }

            return pm4.ToArray();
        }

        var phases = Enum.GetValues<RenderPhaseProfile.Phase>().Where(phase => phase != RenderPhaseProfile.Phase.Count).ToArray();
        var profiling = RenderPhaseProfile.Enabled;
        var results = new List<object>();
        ulong submissionId = 1000;
        foreach (var variation in Enum.GetValues<Variation>().Where(variation => only is null || variation.ToString() == only))
        {
            var words = variation == Variation.Draws ? DrawStream() : DispatchStream();
            var items = variation == Variation.Draws ? drawCount : dispatchCount;
            var streamAddress = rig.Memory.Upload<uint>(words, 256);
            var wall = new List<double>();
            var cpu = new List<double>();
            var phaseTotals = new Dictionary<string, double>();
            long allocated = 0;
            long allAllocated = 0;
            var collections = 0;
            for (var round = 0; round < warm + rounds; round++)
            {
                if (round == warm) _ = PrefetchCounters.Invoke(rig.Host, null);
                RenderPasses.SetValue(rig.Host, 0L);
                var before = profiling ? phases.Select(phase => PhaseTicks(phase)).ToArray() : null;
                rig.Host.CommandStream.EnqueueGraphics(streamAddress, (uint)words.Length, ++submissionId, null);
                var allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
                var allBefore = GC.GetTotalAllocatedBytes(precise: false);
                var collectionsBefore = GC.CollectionCount(0);
                var cpuStarted = ThreadCpuNanoseconds();
                var started = Stopwatch.GetTimestamp();
                rig.RecordPending();
                var elapsed = Stopwatch.GetTimestamp() - started;
                var cpuElapsed = ThreadCpuNanoseconds() - cpuStarted;
                var roundAllocated = GC.GetAllocatedBytesForCurrentThread() - allocatedBefore;
                var roundAll = GC.GetTotalAllocatedBytes(precise: false) - allBefore;
                var roundCollections = GC.CollectionCount(0) - collectionsBefore;
                rig.Drain();
                var counters = rig.Host.ReadAndResetPerfCounters();
                if (round < warm) continue;
                wall.Add(elapsed * 1e6 / Stopwatch.Frequency / items);
                cpu.Add(cpuElapsed / 1e3 / items);
                allocated += roundAllocated;
                allAllocated += roundAll;
                collections += roundCollections;
                if (before is null) continue;
                for (var index = 0; index < phases.Length; index++)
                {
                    var delta = PhaseTicks(phases[index]) - before[index];
                    if (delta == 0) continue;
                    var key = phases[index].ToString();
                    phaseTotals[key] = phaseTotals.GetValueOrDefault(key) + delta * 1e6 / Stopwatch.Frequency / items;
                }
            }

            var prefetch = (string)PrefetchCounters.Invoke(rig.Host, null)!;
            var checksum = 0xCBF29CE484222325UL;
            void Fold(ulong address, int count)
            {
                foreach (var word in rig.Read<uint>(address, count)) checksum = (checksum ^ word) * 0x100000001B3UL;
            }

            if (variation == Variation.Draws)
            {
                foreach (var set in targets)
                foreach (var target in set)
                    Fold(target, (int)(SceneDraws.LinearPitchBytes(TargetSize, 16) / 4 * TargetSize));
            }
            else
            {
                foreach (var image in storage.Concat(scratch)) Fold(image, (int)(SceneDraws.LinearPitchBytes(StorageEdge, 16) / 4 * StorageEdge));
                foreach (var buffer in writeBuffers) Fold(buffer, (int)(StorageEdge * StorageEdge));
            }

            wall.Sort();
            cpu.Sort();
            var unit = variation == Variation.Draws ? "draw" : "dispatch";
            Output.WriteLine($"{variation}: wall median {wall[wall.Count / 2]:F2} us/{unit}, p25 {wall[wall.Count / 4]:F2}; " +
                $"thread cpu median {cpu[cpu.Count / 2]:F2}, p25 {cpu[cpu.Count / 4]:F2}; {allocated / (double)(rounds * items):F0} B/{unit} (all threads {allAllocated / (double)(rounds * items):F0}), " +
                $"gen0 {collections}; checksum {checksum:X16}");
            results.Add(new
            {
                variation = variation.ToString(),
                items,
                rounds,
                wallMedianMicroseconds = wall[wall.Count / 2],
                wallP25Microseconds = wall[wall.Count / 4],
                wallMinMicroseconds = wall[0],
                cpuMedianMicroseconds = cpu[cpu.Count / 2],
                cpuP25Microseconds = cpu[cpu.Count / 4],
                allocatedBytesPerItem = allocated / (double)(rounds * items),
                allThreadsAllocatedBytesPerItem = allAllocated / (double)(rounds * items),
                gen0Collections = collections,
                gen0CollectionsPerThousandItems = collections * 1000.0 / (rounds * items),
                checksum = $"{checksum:X16}",
                counters = prefetch,
                phaseMicroseconds = phaseTotals.Where(pair => pair.Value / rounds >= 0.005)
                    .OrderByDescending(pair => pair.Value).ToDictionary(pair => pair.Key, pair => Math.Round(pair.Value / rounds, 3)),
            });
        }

        Assert.Empty(rig.Fatals);
        File.WriteAllText(path, JsonSerializer.Serialize(
            new { arch = RuntimeInformation.ProcessArchitecture.ToString(), profiling, gc = System.Runtime.GCSettings.IsServerGC ? "server" : "workstation", results },
            new JsonSerializerOptions { WriteIndented = true }));
    }
}
