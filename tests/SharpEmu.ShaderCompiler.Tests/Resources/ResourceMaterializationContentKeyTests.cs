// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

// Draws that bind their descriptors through a table pointer in user data: the pointer moves from
// draw to draw (a per-frame descriptor ring, or several copies of one table), the descriptors it
// points at often do not. Every result must equal a full walk of the same draw.
public sealed class ResourceMaterializationContentKeyTests
{
    private const ulong MemoryBase = 0x10_0000;
    private const int MemoryWords = 0x40000;

    // textures T# and buffers V#, all loaded through the table pointer s[0:1]; the shader samples
    // every texture with one sampler (also from the table) and loads from every buffer.
    private static ShaderResourcePlan TablePlan(int textures, int buffers, int[]? textureOffsets = null, int samplerOffset = -1,
        int[]? bufferOffsets = null)
    {
        textureOffsets ??= Enumerable.Range(0, textures).Select(index => index * 32).ToArray();
        samplerOffset = samplerOffset >= 0 ? samplerOffset : textures * 32;
        bufferOffsets ??= Enumerable.Range(0, buffers).Select(index => samplerOffset + 16 + index * 16).ToArray();
        var program = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        for (var index = 0; index < textures; index++, pc += 8)
            program.Add(ScalarLoad(pc, 0, 8 + 8 * (uint)index, 8, textureOffsets[index]));
        program.Add(ScalarLoad(pc, 0, 72, 4, samplerOffset));
        pc += 8;
        for (var index = 0; index < buffers; index++, pc += 8)
            program.Add(ScalarLoad(pc, 0, 80 + 4 * (uint)index, 4, bufferOffsets[index]));
        for (var index = 0; index < textures; index++, pc += 8)
            program.Add(Image(pc, "ImageSample", 8 + 8 * (uint)index, 72));
        for (var index = 0; index < buffers; index++, pc += 8)
            program.Add(BufferLoad(pc, 80 + 4 * (uint)index));
        program.Add(EndProgram(pc));
        return Extract(Program([.. program]), userDataCount: 2, stage: ShaderStage.Pixel);
    }

    private static uint[] Texture(uint address) => [address >> 8, 20u << 20, 0, 0xFACu | (9u << 28), 0, 0, 0, 0];

    private static uint[] Buffer(uint address, uint size = 256) => [address, 0, size, 0x2_0000];

    // Material m's descriptors, written at table.
    private static void WriteTable(TestWordMemory memory, ulong table, int material, int textures, int buffers, int[]? textureOffsets = null,
        int samplerOffset = -1, int[]? bufferOffsets = null)
    {
        textureOffsets ??= Enumerable.Range(0, textures).Select(index => index * 32).ToArray();
        samplerOffset = samplerOffset >= 0 ? samplerOffset : textures * 32;
        bufferOffsets ??= Enumerable.Range(0, buffers).Select(index => samplerOffset + 16 + index * 16).ToArray();
        for (var index = 0; index < textures; index++)
            Write(memory, table + (ulong)textureOffsets[index], Texture(0x0100_0000u + (uint)(material * 16 + index) * 0x1_0000u));
        Write(memory, table + (ulong)samplerOffset, [0, 0, 0, 0]);
        for (var index = 0; index < buffers; index++)
            Write(memory, table + (ulong)bufferOffsets[index], Buffer(0x0200_0000u + (uint)(material * 8 + index) * 0x1000u));
    }

    private static void Write(TestWordMemory memory, ulong address, uint[] words)
    {
        for (var index = 0; index < words.Length; index++)
            memory.At(address + (ulong)index * 4) = words[index];
    }

    private static TestWordMemory NewMemory() => new() { Base = MemoryBase, Words = new uint[MemoryWords] };

    private static uint[] Pointer(ulong table) => [(uint)table, (uint)(table >> 32)];

    private static bool Run(ResourceMaterializationCache cache, ShaderResourcePlan plan, TestWordMemory memory, uint[] userData,
        out ResourceSnapshot snapshot, out ResourceSpecialization specialization)
    {
        snapshot = new ResourceSnapshot();
        specialization = new ResourceSpecialization();
        return cache.Materialize(plan, Inputs(userData, memory.Read, memory.Read), (address, destination, _) =>
        {
            for (var offset = 0; offset < destination.Length; offset += 4)
            {
                if (!memory.Read(address + (ulong)offset, out var word))
                    return false;
                BitConverter.TryWriteBytes(destination[offset..], word);
            }

            return true;
        }, ref snapshot, ref specialization, out _);
    }

    private static void AssertFullWalk(ShaderResourcePlan plan, TestWordMemory memory, uint[] userData, ResourceSnapshot snapshot,
        ResourceSpecialization specialization)
    {
        var expected = new ResourceSnapshot();
        var expectedSpecialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs(userData, memory.Read, memory.Read), ref expected, ref expectedSpecialization));
        Assert.Equal(expected.Buffers.Select(words => words.ToArray()), snapshot.Buffers.Select(words => words.ToArray()));
        Assert.Equal(expected.Images.Select(words => words.ToArray()), snapshot.Images.Select(words => words.ToArray()));
        Assert.Equal(expected.Samplers.Select(words => words.ToArray()), snapshot.Samplers.Select(words => words.ToArray()));
        Assert.Equal(expected.FlattenedResourceTable, snapshot.FlattenedResourceTable);
        Assert.Equal(expected.DeviceAddressRanges, snapshot.DeviceAddressRanges);
        Assert.Equal(userData, snapshot.UserData);
        Assert.Equal(expectedSpecialization, specialization);
    }

    [Fact]
    public void IdenticalDescriptorsAtDifferentTableAddressesHit()
    {
        var plan = TablePlan(textures: 4, buffers: 2);
        var memory = NewMemory();
        var tables = Enumerable.Range(0, 8).Select(index => MemoryBase + 0x1000 + (ulong)index * 0x300).ToArray();
        foreach (var table in tables)
            WriteTable(memory, table, material: 3, textures: 4, buffers: 2);
        var cache = new ResourceMaterializationCache();
        for (var draw = 0; draw < 50; draw++)
        {
            var userData = Pointer(tables[draw % tables.Length]);
            Assert.True(Run(cache, plan, memory, userData, out var snapshot, out var specialization));
            AssertFullWalk(plan, memory, userData, snapshot, specialization);
        }

        // One miss to build, three to confirm the pointer; every later draw is a hit.
        Assert.Equal((46, 4), (cache.Hits, cache.Misses));
    }

    [Fact]
    public void ObjectsWhoseTablesMoveEveryFrameHitOnceThePointerIsConfirmed()
    {
        const int Materials = 6;
        var plan = TablePlan(textures: 4, buffers: 2);
        var memory = NewMemory();
        var cache = new ResourceMaterializationCache();
        for (var frame = 0; frame < 12; frame++)
        {
            for (var material = 0; material < Materials; material++)
            {
                // A ring that drifts, so no material lies where it lay before.
                var table = MemoryBase + 0x1000 + (ulong)((frame * 7 + material) % 300) * 0x200;
                WriteTable(memory, table, material, textures: 4, buffers: 2);
                var userData = Pointer(table);
                Assert.True(Run(cache, plan, memory, userData, out var snapshot, out var specialization));
                AssertFullWalk(plan, memory, userData, snapshot, specialization);
            }
        }

        // The first frame builds, the second confirms the pointer, and from the third every object
        // finds its own entry wherever its table lies.
        Assert.True(cache.Hits >= 9 * Materials, $"hits={cache.Hits} misses={cache.Misses}");
    }

    [Fact]
    public void AChangedDescriptorWordIsNeverReused()
    {
        var plan = TablePlan(textures: 4, buffers: 2);
        var memory = NewMemory();
        var tables = Enumerable.Range(0, 8).Select(index => MemoryBase + 0x1000 + (ulong)index * 0x300).ToArray();
        foreach (var table in tables)
            WriteTable(memory, table, material: 3, textures: 4, buffers: 2);
        var cache = new ResourceMaterializationCache();
        for (var draw = 0; draw < 24; draw++)
            Assert.True(Run(cache, plan, memory, Pointer(tables[draw % tables.Length]), out _, out _));
        var hits = cache.Hits;

        // A texture rewritten in place at an address the cache has seen.
        Write(memory, tables[5] + 32, Texture(0x0700_0000));
        Assert.True(Run(cache, plan, memory, Pointer(tables[5]), out var rewritten, out var rewrittenSpecialization));
        AssertFullWalk(plan, memory, Pointer(tables[5]), rewritten, rewrittenSpecialization);
        Assert.Equal(hits, cache.Hits);

        // A buffer resized behind a pointer the cache has never seen, whose other words match.
        var moved = MemoryBase + 0x8000;
        WriteTable(memory, moved, material: 3, textures: 4, buffers: 2);
        Write(memory, moved + 4 * 32 + 16 + 8, [512]);
        Assert.True(Run(cache, plan, memory, Pointer(moved), out var resized, out var resizedSpecialization));
        AssertFullWalk(plan, memory, Pointer(moved), resized, resizedSpecialization);
        Assert.Equal(512u, resized.Buffers[0][2]);

        // The untouched tables still hit, and still give their own descriptors.
        Assert.True(Run(cache, plan, memory, Pointer(tables[2]), out var untouched, out var untouchedSpecialization));
        AssertFullWalk(plan, memory, Pointer(tables[2]), untouched, untouchedSpecialization);
        Assert.Equal(hits + 1, cache.Hits);
    }

    // A hit through a pointer seen before guesses its key from the print the pointer gave last time.
    // Descriptors rewritten behind the pointer since then must never hit the entry of the old ones,
    // and when they now equal another object's, the draw finds that object's entry instead.
    [Fact]
    public void ARecalledPrintNeverHitsDescriptorsRewrittenBehindThePointer()
    {
        var plan = TablePlan(textures: 4, buffers: 2);
        var memory = NewMemory();
        var tables = Enumerable.Range(0, 8).Select(index => MemoryBase + 0x1000 + (ulong)index * 0x300).ToArray();
        foreach (var table in tables)
            WriteTable(memory, table, material: 3, textures: 4, buffers: 2);
        var cache = new ResourceMaterializationCache();
        for (var draw = 0; draw < 32; draw++)
            Assert.True(Run(cache, plan, memory, Pointer(tables[draw % tables.Length]), out _, out _));
        var hits = cache.Hits;
        var misses = cache.Misses;

        void Draw(ulong table)
        {
            Assert.True(Run(cache, plan, memory, Pointer(table), out var snapshot, out var specialization));
            AssertFullWalk(plan, memory, Pointer(table), snapshot, specialization);
        }

        // Every descriptor behind tables[5] rewritten: the recalled print points at material 3's entry.
        WriteTable(memory, tables[5], material: 7, textures: 4, buffers: 2);
        Draw(tables[5]);
        Assert.Equal((hits, misses + 1), (cache.Hits, cache.Misses));

        // tables[6] now holds what tables[5] does: its recalled print is stale, its real one finds
        // the entry tables[5] just built.
        WriteTable(memory, tables[6], material: 7, textures: 4, buffers: 2);
        Draw(tables[6]);
        Assert.Equal((hits + 1, misses + 1), (cache.Hits, cache.Misses));

        // One word rewritten behind a pointer whose print was just read.
        Write(memory, tables[6] + 32 + 4, [21u << 20]);
        Draw(tables[6]);
        Assert.Equal((hits + 1, misses + 2), (cache.Hits, cache.Misses));

        // Pointers left alone keep hitting their own descriptors.
        Draw(tables[5]);
        Draw(tables[2]);
        Assert.Equal((hits + 3, misses + 2), (cache.Hits, cache.Misses));
    }

    [Fact]
    public void ARecalledPrintChecksReadabilityAndReplacementMemoryAtTheSameAddress()
    {
        var plan = TablePlan(textures: 4, buffers: 2);
        var memory = NewMemory();
        var tables = Enumerable.Range(0, 8).Select(index => MemoryBase + 0x1000 + (ulong)index * 0x300).ToArray();
        foreach (var table in tables)
            WriteTable(memory, table, material: 3, textures: 4, buffers: 2);
        var cache = new ResourceMaterializationCache();
        for (var draw = 0; draw < 32; draw++)
            Assert.True(Run(cache, plan, memory, Pointer(tables[draw % tables.Length]), out _, out _));

        var hits = cache.Hits;
        // The table pointer and its last print survive, but a descriptor word is no longer readable.
        memory.FailAddress = tables[5] + 4;
        var expected = new ResourceSnapshot();
        var expectedSpecialization = new ResourceSpecialization();
        var expectedResult = ResourceMaterializer.Materialize(plan,
            Inputs(Pointer(tables[5]), memory.Read, memory.Read), ref expected, ref expectedSpecialization);
        Assert.Equal(expectedResult, Run(cache, plan, memory, Pointer(tables[5]), out _, out _));
        Assert.Equal(hits, cache.Hits);

        // A new mapping at the same guest address must use its own contents, including buffer bounds.
        var replacement = NewMemory();
        foreach (var table in tables)
            WriteTable(replacement, table, material: 9, textures: 4, buffers: 2);
        Write(replacement, tables[5] + 4 * 32 + 16 + 8, [1024u]);
        Assert.True(Run(cache, plan, replacement, Pointer(tables[5]), out var snapshot, out var specialization));
        AssertFullWalk(plan, replacement, Pointer(tables[5]), snapshot, specialization);
        Assert.Equal(1024u, snapshot.Buffers[0][2]);
        Assert.Equal(hits, cache.Hits);
    }

    // A hit through a moved pointer gives each pointer its own snapshot; draws that come back to a
    // pointer share the one made for it, and nothing a later draw does changes an earlier one's.
    [Fact]
    public void HitSnapshotsAreSharedPerPointerAndNeverWritten()
    {
        var plan = TablePlan(textures: 2, buffers: 1);
        var memory = NewMemory();
        var tables = Enumerable.Range(0, 8).Select(index => MemoryBase + 0x1000 + (ulong)index * 0x300).ToArray();
        foreach (var table in tables)
            WriteTable(memory, table, material: 1, textures: 2, buffers: 1);
        var userData = tables.Select(Pointer).ToArray();
        var cache = new ResourceMaterializationCache();
        ResidentGuestBytesReader resident = (address, destination, _) =>
        {
            for (var offset = 0; offset < destination.Length; offset += 4)
            {
                if (!memory.Read(address + (ulong)offset, out var word))
                    return false;
                BitConverter.TryWriteBytes(destination[offset..], word);
            }

            return true;
        };
        var inputs = userData.Select(words => Inputs(words, memory.Read, memory.Read)).ToArray();
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        var first = new ResourceSnapshot[tables.Length];
        for (var draw = 0; draw < 4 * tables.Length; draw++)
        {
            Assert.True(cache.Materialize(plan, inputs[draw % tables.Length], resident, ref snapshot, ref specialization, out _));
            if (draw >= 2 * tables.Length && draw < 3 * tables.Length)
                first[draw % tables.Length] = snapshot;
            if (draw >= 3 * tables.Length)
                Assert.Same(first[draw % tables.Length], snapshot);
        }

        for (var index = 0; index < tables.Length; index++)
        {
            Assert.Equal(userData[index], first[index].UserData);
            AssertFullWalk(plan, memory, userData[index], first[index], specialization);
        }

        // Steady state: a cycle of hits allocates nothing.
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var draw = 0; draw < tables.Length; draw++)
            Assert.True(cache.Materialize(plan, inputs[draw], resident, ref snapshot, ref specialization, out _));
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    // Random layouts, random materials at random and reused table addresses, in-place rewrites and
    // tables that move: whatever the cache answers, hit, refresh or miss, is what a full walk gives.
    [Fact]
    public void EveryAnswerEqualsAFullWalkOverRandomLayouts()
    {
        var random = new Random(0x5EED);
        for (var layout = 0; layout < 24; layout++)
        {
            var textures = random.Next(1, 7);
            var buffers = random.Next(0, 4);
            // Descriptor slots in a shuffled order with gaps; textures need 32 bytes, the rest 16.
            var slots = Enumerable.Range(0, textures + 1 + buffers + random.Next(0, 4)).OrderBy(_ => random.Next()).ToArray();
            var textureOffsets = slots.Take(textures).Select(slot => slot * 32).ToArray();
            var samplerOffset = slots[textures] * 32;
            var bufferOffsets = slots.Skip(textures + 1).Take(buffers).Select(slot => slot * 32 + 16 * random.Next(0, 2)).ToArray();
            var plan = TablePlan(textures, buffers, textureOffsets, samplerOffset, bufferOffsets);
            var memory = NewMemory();
            var cache = new ResourceMaterializationCache();
            var materials = random.Next(1, 6);
            var addresses = Enumerable.Range(0, 12).Select(index => MemoryBase + 0x1000 + (ulong)index * 0x400).ToArray();
            var tableOf = new int[addresses.Length];
            Array.Fill(tableOf, -1);
            for (var draw = 0; draw < 160; draw++)
            {
                var slot = random.Next(addresses.Length);
                var table = addresses[slot];
                var action = random.Next(10);
                if (tableOf[slot] < 0 || action < 3)
                {
                    tableOf[slot] = random.Next(materials);
                    WriteTable(memory, table, tableOf[slot], textures, buffers, textureOffsets, samplerOffset, bufferOffsets);
                }
                else if (action == 3)
                {
                    // One descriptor taken from another material, in place.
                    var texture = random.Next(textures);
                    Write(memory, table + (ulong)textureOffsets[texture], Texture(0x0300_0000u + (uint)random.Next(64) * 0x1_0000u));
                    tableOf[slot] = materials + draw;
                }
                else if (action == 4 && buffers != 0)
                {
                    Write(memory, table + (ulong)bufferOffsets[random.Next(buffers)] + 8, [(uint)random.Next(1, 4) * 256]);
                    tableOf[slot] = materials + draw;
                }

                var userData = Pointer(table);
                Assert.True(Run(cache, plan, memory, userData, out var snapshot, out var specialization));
                AssertFullWalk(plan, memory, userData, snapshot, specialization);
            }

            Assert.True(cache.Hits > 0, $"layout {layout}: hits={cache.Hits} misses={cache.Misses}");
        }
    }
}
