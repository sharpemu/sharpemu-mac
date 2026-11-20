// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Gpu.Metal.V2;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.Gpu.Metal.Tests.Scenes;

/// <summary>
/// The bindless heap through the whole host (MetalRenderHost.BindlessHeap.cs): a compute program reads a T# from a table
/// V# at a record each lane loads, so planning makes it a heap record. The first dispatch reads null and marks the
/// records; a later one reads the textures the host resolved from the marked records. The elements stay resolved while
/// draws read the view, a rewritten record gets its new texture once the view is compared again, and a heap that grows
/// past its run keeps the elements it had.
/// </summary>
public sealed class BindlessHeapSceneTests(ITestOutputHelper output) : SceneTestBase(output)
{
    // MetalBindlessImageTests' waterfall (wave64): per lane, the key (<< 4) at +0x14000 + lane * 4 of the V# at block + 0x20,
    // s_buffer_load_dwordx8 of the T# at key * 32 through the table V# at block + 0x10, image_sample_l at
    // (0.5, 0.5, layer 1, lod 0) dim:2D_ARRAY with the S# at block + 0x30, the texel stored at lane * 16.
    private static readonly uint[] Program =
    [
        0xF4080200, 0xFA000010, 0xF4080600, 0xFA000020, 0xF4080700, 0xFA000030, 0xBF8CC07F, 0x34020082,
        0x4A0202FF, 0x00014000, 0x34060084, 0xE0301000, 0x80060201, 0xBF8C3F70, 0x2C160484, 0x7E0802F0,
        0x7E0A02F0, 0x7E0C02F2, 0x7E0E0280, 0xBE86047E, 0x7E260281, 0x7EC8050B, 0xBE94047E, 0x7DA41664,
        0x8F648564, 0x7E260280, 0xF42C0304, 0xC8000000, 0xBF8CC07F, 0xF0900F28, 0x00E31404, 0xBEFE0414,
        0x7DAA2680, 0xBF89FFF3, 0xBEFE0406, 0xBF8C3F70, 0xE0781000, 0x80061403, 0xBF810000,
    ];

    // The T# from the block (s_load_dwordx8 s[12:19] at +0x40), the S# from the table at key * 16
    // (s_lshl_b32 s100, s100, 4; s_buffer_load_dwordx4 s[28:31], s[8:11], s100): a heap sampler record.
    private static readonly uint[] SamplerProgram = Patch(Program, (4, 0xF40C0300), (5, 0xFA000040), (24, 0x8F648464), (26, 0xF4280704), (27, 0xC8000000));

    private static uint[] Patch(uint[] words, params (int Index, uint Word)[] patches)
    {
        var patched = words.ToArray();
        foreach (var (index, word) in patches)
            patched[index] = word;
        return patched;
    }

    private const int Lanes = 64;
    private const uint KeyOffset = 0x14000;
    private const uint RecordBytes = 32;
    private const uint Rgba8 = 56;
    private const uint Type2DArray = 13;
    // Far enough apart that no texture's range (a linear layer is padded) reaches the next one.
    private const int TextureBytes = 0x10000;

    private sealed class Heap
    {
        public ulong Block;
        public ulong Table;
        public ulong Output;
        public ulong[] Textures = [];
    }

    // Texture t is one colour in every texel of both layers, so the read does not depend on the layer layout.
    private static uint Colour(int texture) => 0xFF000000u | ((uint)(7 + 3 * texture) << 16) | ((uint)(200 - texture) << 8) | (uint)(16 * texture + 1);

    private static float[] Expected(int texture)
    {
        var colour = Colour(texture);
        return [(colour & 0xFF) / 255f, ((colour >> 8) & 0xFF) / 255f, ((colour >> 16) & 0xFF) / 255f, (colour >> 24) / 255f];
    }

    private static Heap Build(SceneRig rig, int textures, int records, int tableRecords)
    {
        var heap = new Heap
        {
            Block = rig.Memory.Allocate(0x100),
            Table = rig.Memory.Allocate((ulong)tableRecords * RecordBytes),
            Output = rig.Memory.Allocate(KeyOffset + Lanes * 4),
            Textures = new ulong[textures],
        };

        for (var texture = 0; texture < textures; texture++)
        {
            heap.Textures[texture] = rig.Memory.Allocate(TextureBytes, TextureBytes);
            rig.Memory.Write<uint>(heap.Textures[texture], Enumerable.Repeat(Colour(texture), TextureBytes / 4).ToArray());
        }

        rig.Memory.Write<byte>(heap.Table, new byte[tableRecords * RecordBytes]);
        for (var record = 0; record < tableRecords; record++)
        {
            WriteRecord(rig, heap, record, record % textures);
        }

        SetRecords(rig, heap, records);
        rig.Memory.Write<uint>(heap.Block + 0x20, [Descriptors.Lo(heap.Output), Descriptors.Hi(heap.Output) & 0xFFFF, KeyOffset + Lanes * 4, 0x00038FAC]);
        rig.Memory.Write<uint>(heap.Block + 0x30, Descriptors.Sampler());
        return heap;
    }

    // The table V#: records * 32 bytes of T#s.
    private static void SetRecords(SceneRig rig, Heap heap, int records) =>
        rig.Memory.Write<uint>(heap.Block + 0x10,
            [Descriptors.Lo(heap.Table), Descriptors.Hi(heap.Table) & 0xFFFF, (uint)records * RecordBytes, 0x00038FAC]);

    private static void WriteRecord(SceneRig rig, Heap heap, int record, int texture) =>
        rig.Memory.Write<uint>(heap.Table + (ulong)record * RecordBytes,
            Descriptors.Texture(heap.Textures[texture], Rgba8, 1, 1, type: Type2DArray, depthOrLastLayer: 1));

    // One dispatch of a wave whose lane l reads record keys[l]; the four result words of each lane, as floats.
    private static float[][] Pass(SceneRig rig, Heap heap, int[] keys, uint[]? program = null)
    {
        rig.Memory.Write<uint>(heap.Output, new uint[Lanes * 4]);
        // The program reads each key shifted left by four (v_lshrrev_b32 v11, 4, v2).
        rig.Memory.Write<uint>(heap.Output + KeyOffset, keys.Select(key => (uint)key << 4).ToArray());
        rig.Submit(rig.Dispatch(new ScenePm4(), program ?? Program, [Descriptors.Lo(heap.Block), Descriptors.Hi(heap.Block)], 1, threadsX: Lanes));
        var words = rig.Read<float>(heap.Output, Lanes * 4);
        return Enumerable.Range(0, Lanes).Select(lane => words.AsSpan(lane * 4, 4).ToArray()).ToArray();
    }

    private static void AssertTexel(float[] actual, float[] expected, string what)
    {
        for (var channel = 0; channel < 4; channel++)
        {
            Assert.True(MathF.Abs(actual[channel] - expected[channel]) <= 0.5f / 255f,
                $"{what}: got ({string.Join(", ", actual)}) expected ({string.Join(", ", expected)})");
        }
    }

    private static readonly float[] Zero = [0, 0, 0, 0];

    [Fact]
    public void TheFirstDispatchMarksAndALaterOneReadsTheTextureOfEachRecord()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        Assert.True(rig.Host.SupportsBindlessHeap);
        const int textures = 8;
        var heap = Build(rig, textures, records: 16, tableRecords: 16);
        var keys = Enumerable.Range(0, Lanes).Select(lane => lane * 5 % 16).ToArray();

        // Nothing is resolved before the marks of a completed dispatch come back: every lane reads the null element.
        var first = Pass(rig, heap, keys);
        for (var lane = 0; lane < Lanes; lane++)
        {
            AssertTexel(first[lane], Zero, $"first pass lane {lane} key {keys[lane]}");
        }

        Assert.Equal(0, rig.Host.HeapResolvedEntries);
        var second = Pass(rig, heap, keys);
        for (var lane = 0; lane < Lanes; lane++)
        {
            AssertTexel(second[lane], Expected(keys[lane] % textures), $"second pass lane {lane} key {keys[lane]}");
        }

        // Each of the 16 records was resolved (a record whose image a later record's lookup changed is resolved again
        // once), and a dispatch that reads the same records resolves nothing more.
        var resolved = rig.Host.HeapResolvedEntries;
        Assert.InRange(resolved, 16, 32);
        _ = Pass(rig, heap, keys);
        Assert.Equal(resolved, rig.Host.HeapResolvedEntries);
        // A key past the view reads null, and marks nothing.
        var past = Pass(rig, heap, Enumerable.Range(0, Lanes).Select(lane => lane % 2 == 0 ? 16 + lane : lane % 16).ToArray());
        for (var lane = 0; lane < Lanes; lane++)
        {
            AssertTexel(past[lane], lane % 2 == 0 ? Zero : Expected(lane % 16 % textures), $"past pass lane {lane}");
        }
    }

    [Fact]
    public void ResolvedElementsStayAcrossDispatchesAndARewrittenRecordTakesItsNewTexture()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        const int textures = 4;
        var heap = Build(rig, textures, records: 8, tableRecords: 8);
        var lowKeys = Enumerable.Range(0, Lanes).Select(lane => lane % 4).ToArray();
        var highKeys = Enumerable.Range(0, Lanes).Select(lane => 4 + lane % 4).ToArray();
        _ = Pass(rig, heap, lowKeys);
        _ = Pass(rig, heap, highKeys);
        var resolved = rig.Host.HeapResolvedEntries;

        // Each set keeps its elements while the other is read: no null read, nothing resolved again.
        for (var round = 0; round < 3; round++)
        {
            foreach (var keys in new[] { lowKeys, highKeys })
            {
                var result = Pass(rig, heap, keys);
                for (var lane = 0; lane < Lanes; lane++)
                {
                    AssertTexel(result[lane], Expected(keys[lane] % textures), $"round {round} lane {lane} key {keys[lane]}");
                }
            }
        }

        Assert.Equal(resolved + 4, rig.Host.HeapResolvedEntries);

        // Record 1 now names texture 3: the view's records are compared again after a few submissions, and the lane that
        // reads record 1 reads texture 3 from then on.
        WriteRecord(rig, heap, 1, 3);
        float[][] last = [];
        for (var pass = 0; pass < 20; pass++)
        {
            last = Pass(rig, heap, lowKeys);
        }

        for (var lane = 0; lane < Lanes; lane++)
        {
            AssertTexel(last[lane], Expected(lowKeys[lane] == 1 ? 3 : lowKeys[lane] % textures), $"rewritten lane {lane} key {lowKeys[lane]}");
        }
    }

    [Fact]
    public void AHeapThatGrowsPastItsRunKeepsTheElementsItHad()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        const int textures = 8;
        var heap = Build(rig, textures, records: 8, tableRecords: 32);
        var oldKeys = Enumerable.Range(0, Lanes).Select(lane => lane % 8).ToArray();
        _ = Pass(rig, heap, oldKeys);
        _ = Pass(rig, heap, oldKeys);

        // The guest streams more records into the heap: the view outgrows its run and takes a larger one. The grown V# is
        // in a new resource block (the scene's memory has no write tracking, so the old block would be read as before).
        var block = rig.Memory.Allocate(0x100);
        rig.Memory.Write(block, rig.Memory.ReadRaw(heap.Block, 0x100));
        heap.Block = block;
        SetRecords(rig, heap, 32);
        var keys = Enumerable.Range(0, Lanes).Select(lane => lane % 2 == 0 ? lane % 8 : 8 + lane % 24).ToArray();
        var grown = Pass(rig, heap, keys);
        for (var lane = 0; lane < Lanes; lane++)
        {
            // The records the old run held read at once; the new ones read null until their marks come back.
            AssertTexel(grown[lane], keys[lane] < 8 ? Expected(keys[lane] % textures) : Zero, $"grown pass lane {lane} key {keys[lane]}");
        }

        var later = Pass(rig, heap, keys);
        for (var lane = 0; lane < Lanes; lane++)
        {
            AssertTexel(later[lane], Expected(keys[lane] % textures), $"later pass lane {lane} key {keys[lane]}");
        }
    }

    [Fact]
    public void AHeapSamplerRecordFiltersWithTheSamplerItNames()
    {
        using var rig = CreateRig();
        if (rig is null) return;
        // One 2x1 texture (texel 0 colour 0, texel 1 colour 1, every row of both layers) and a table of S#s: even records
        // point-sample, odd ones filter. At u = 0.5 a point sample reads texel 1 and a bilinear one their average.
        var heap = Build(rig, textures: 1, records: 1, tableRecords: 1);
        var texture = rig.Memory.Allocate(TextureBytes, TextureBytes);
        var rows = new uint[TextureBytes / 4];
        for (var row = 0; row < rows.Length; row += 64)
        {
            rows[row] = Colour(0);
            rows[row + 1] = Colour(1);
        }

        rig.Memory.Write<uint>(texture, rows);
        rig.Memory.Write<uint>(heap.Block + 0x40, Descriptors.Texture(texture, Rgba8, 2, 1, type: Type2DArray, depthOrLastLayer: 1));
        const int samplers = 8;
        var table = rig.Memory.Allocate(samplers * 16);
        for (var record = 0; record < samplers; record++)
        {
            var filter = (uint)(record & 1);
            rig.Memory.Write<uint>(table + (ulong)record * 16, Descriptors.Sampler(magFilter: filter, minFilter: filter));
        }

        rig.Memory.Write<uint>(heap.Block + 0x10, [Descriptors.Lo(table), Descriptors.Hi(table) & 0xFFFF, samplers * 16, 0x00038FAC]);
        var keys = Enumerable.Range(0, Lanes).Select(lane => lane * 3 % samplers).ToArray();
        _ = Pass(rig, heap, keys, SamplerProgram);
        var result = Pass(rig, heap, keys, SamplerProgram);
        var point = Expected(1);
        var linear = Expected(0).Zip(Expected(1), (a, b) => (a + b) / 2).ToArray();
        for (var lane = 0; lane < Lanes; lane++)
        {
            AssertTexel(result[lane], keys[lane] % 2 == 0 ? point : linear, $"lane {lane} sampler record {keys[lane]}");
        }
    }
}
