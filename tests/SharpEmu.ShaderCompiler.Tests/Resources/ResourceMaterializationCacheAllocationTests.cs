// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

// What the cache allocates per draw once it has seen the draws: a hit allocates nothing beyond
// the per-draw parts of the snapshot it returns, and an entry rebuilt every frame shares what did
// not change. Measured around Materialize alone; the inputs are made before.
public sealed class ResourceMaterializationCacheAllocationTests(ITestOutputHelper output)
{
    private sealed class Draws(ShaderResourcePlan plan, TestWordMemory memory)
    {
        private readonly ResourceMaterializationCache _cache = new();
        private readonly ResidentGuestBytesReader _resident = (address, destination, _) =>
        {
            for (var offset = 0; offset < destination.Length; offset += 4)
            {
                if (!memory.Read(address + (ulong)offset, out var word))
                    return false;
                BitConverter.TryWriteBytes(destination[offset..], word);
            }

            return true;
        };

        private ResourceSnapshot _snapshot = new();
        private ResourceSpecialization _specialization = new();

        public ResourceMaterializationCache Cache => _cache;

        public ResourceSnapshot Snapshot => _snapshot;

        public ResourceRuntimeInputs InputsOf(uint[] userData) => Inputs(userData, memory.Read, memory.Read);

        // The bytes this thread allocated over the draw.
        public long Run(ResourceRuntimeInputs inputs)
        {
            var before = GC.GetAllocatedBytesForCurrentThread();
            Assert.True(_cache.Materialize(plan, inputs, _resident, ref _snapshot, ref _specialization, out _));
            return GC.GetAllocatedBytesForCurrentThread() - before;
        }
    }

    // A descriptor table at s[0:1] holds a pointer to a per-frame ring at +8; the ring holds two
    // constants the shader reads and, at +0x60, the V# of the buffer it loads.
    private static ShaderResourcePlan RingPointerPlan() =>
        Extract(Program(ScalarLoad(0, 0, 4, 2, immediateOffset: 8), ScalarLoad(4, 4, 8, 4, immediateOffset: 0x60),
            ScalarLoad(8, 4, 12, 2), Vop1(12, "VMovB32", 0, Gen5Operand.Scalar(12)), Vop1(16, "VMovB32", 1, Gen5Operand.Scalar(13)),
            BufferLoad(20, 8), EndProgram(24)), userDataCount: 2);

    private static void WriteRing(TestWordMemory memory, uint ring, uint first, uint second)
    {
        memory.At(0x1008) = ring;
        memory.At(0x100C) = 0;
        memory.At(ring) = first;
        memory.At(ring + 4) = second;
        memory.At(ring + 0x60) = 0x18000;
        memory.At(ring + 0x64) = 0;
        memory.At(ring + 0x68) = 256;
        memory.At(ring + 0x6C) = 0x2_0000;
    }

    [Fact]
    public void AnUnchangedDrawHitsWithoutAllocating()
    {
        var memory = new TestWordMemory { Base = 0x1000, Words = new uint[0x8000] };
        var draws = new Draws(RingPointerPlan(), memory);
        WriteRing(memory, 0x10000, 11, 22);
        var inputs = draws.InputsOf([0x1000, 0]);
        draws.Run(inputs);
        draws.Run(inputs);
        Assert.Equal(0, draws.Run(inputs));
        Assert.Equal(2, draws.Cache.Hits);
    }

    [Fact]
    public void ARepeatedDrawWithOtherUnreadWordsHitsWithoutAllocating()
    {
        // The plan reads s[0:1]; s[2] is a word only the shader's own ALU reads.
        var plan = Extract(Program(ScalarLoad(0, 0, 4, 2, immediateOffset: 8), ScalarLoad(4, 4, 8, 4, immediateOffset: 0x60),
            ScalarLoad(8, 4, 12, 2), Vop1(12, "VMovB32", 0, Gen5Operand.Scalar(12)), Vop1(16, "VMovB32", 1, Gen5Operand.Scalar(13)),
            BufferLoad(20, 8), EndProgram(24)), userDataCount: 3);
        var memory = new TestWordMemory { Base = 0x1000, Words = new uint[0x8000] };
        var draws = new Draws(plan, memory);
        WriteRing(memory, 0x10000, 11, 22);
        var built = draws.InputsOf([0x1000, 0, 1]);
        var other = draws.InputsOf([0x1000, 0, 2]);
        draws.Run(built);
        draws.Run(other);
        var first = draws.Snapshot;
        Assert.Equal([0x1000u, 0, 2], first.UserData);
        draws.Run(built);
        Assert.Equal(0, draws.Run(other));
        Assert.Same(first, draws.Snapshot);
        Assert.Equal(3, draws.Cache.Hits);
    }

    [Fact]
    public void ARepeatedRebasedDrawHitsWithoutAllocating()
    {
        // A V# taken verbatim from s[4:7]; its base word moves with every draw and is rebased.
        var plan = Extract(Program(BufferLoad(0, 4), EndProgram(8)), userDataCount: 8);
        var draws = new Draws(plan, new TestWordMemory());
        var inputs = Enumerable.Range(0, 16)
            .Select(draw => draws.InputsOf([0, 0, 0, 0, 0x10000u + (uint)draw * 0x100u, 0, 256, 0x2_0000])).ToArray();
        foreach (var draw in inputs)
            draws.Run(draw);
        var hits = draws.Cache.Hits;
        var patched = draws.Run(inputs[^1]);
        Assert.Equal(hits + 1, draws.Cache.Hits);
        Assert.Equal(0x10F00u, draws.Snapshot.Buffers[0][0]);
        Assert.Equal(0, patched);

        // A draw with a base not seen before patches a copy: one snapshot, the buffer list and the V#. The snapshot's
        // binding packet slot (ResourceSnapshot.BindingPacket) adds 8 bytes, so 168.
        var fresh = draws.Run(draws.InputsOf([0, 0, 0, 0, 0x20000, 0, 256, 0x2_0000]));
        output.WriteLine($"fresh rebased hit: {fresh} bytes");
        Assert.Equal(0x20000u, draws.Snapshot.Buffers[0][0]);
        Assert.True(fresh <= 168, $"bytes={fresh}");
    }

    [Fact]
    public void ConstantsRewrittenInPlaceRefreshTheirSlots()
    {
        var (plan, _, _) = Prepare(FlattenedReadReuseTests.RepeatedReadProgram(4), userDataCount: 9);
        var memory = new TestWordMemory { Words = [11, 22, 33, 44] };
        var draws = new Draws(plan, memory);
        var inputs = draws.InputsOf([0x1000, 0, 0, 0, 0, 0, 0, 0, 0]);
        var perFrame = new List<long>();
        for (var frame = 0; frame < 16; frame++)
        {
            memory.Words[2] = 100 + (uint)frame;
            perFrame.Add(draws.Run(inputs) + draws.Run(inputs));
        }

        output.WriteLine($"per frame (bytes): {string.Join(' ', perFrame)} slot_refreshes={draws.Cache.SlotRefreshes} hits={draws.Cache.Hits}");
        Assert.True(draws.Cache.SlotRefreshes >= 14);
        Assert.Equal(100u + 15, draws.Snapshot.FlattenedResourceTable[2]);
    }

    [Fact]
    public void ConstantsBehindARebasedPointerRefreshTheTable()
    {
        // s[4:5] is the base of a V# and the address of four constants the table reads; it moves with
        // every draw, and the constants there differ every time.
        var plan = Extract(Program(ScalarLoad(0, 4, 16), Vop1(8, "VMovB32", 0, Gen5Operand.Scalar(16)), BufferStore(12, 4), EndProgram(20)),
            userDataCount: 8);
        var memory = new TestWordMemory { Base = 0x10000, Words = Enumerable.Range(0, 0x1000).Select(index => (uint)index * 7).ToArray() };
        var draws = new Draws(plan, memory);
        var inputs = Enumerable.Range(0, 16)
            .Select(draw => draws.InputsOf([0, 0, 0, 0, 0x10000u + (uint)draw * 0x100u, 0, 256, 0x2_0000])).ToArray();
        var perDraw = inputs.Select(draws.Run).ToList();
        output.WriteLine($"per draw (bytes): {string.Join(' ', perDraw)} refreshes={draws.Cache.TableRefreshes} slot_refreshes={draws.Cache.SlotRefreshes}");
        Assert.True(draws.Cache.TableRefreshes - draws.Cache.SlotRefreshes >= 10);
    }

    [Fact]
    public void ABaseRewrittenEveryDrawPatchesOneSnapshot()
    {
        // The V# comes from a descriptor table at s[0:1], rewritten every draw with a fresh base.
        var plan = Extract(Program(ScalarLoad(0, 0, 8, 4), ScalarBufferLoad(4, 8, 12, 4), Vop1(8, "VMovB32", 0, Gen5Operand.Scalar(12)),
            BufferLoad(12, 8), EndProgram(16)), userDataCount: 2);
        var memory = new TestWordMemory { Base = 0x1000, Words = new uint[0x8000] };
        var draws = new Draws(plan, memory);
        var inputs = draws.InputsOf([0x1000, 0]);
        var perDraw = new List<long>();
        for (var draw = 0; draw < 16; draw++)
        {
            var bufferBase = 0x10000u + (uint)draw * 0x100u;
            memory.At(0x1000) = bufferBase;
            memory.At(0x1004) = 0;
            memory.At(0x1008) = 256;
            memory.At(0x100C) = 0x2_0000;
            for (var word = 0u; word < 4; word++)
                memory.At(bufferBase + word * 4) = (uint)draw * 10 + word;
            perDraw.Add(draws.Run(inputs));
            Assert.Equal(bufferBase, draws.Snapshot.Buffers[0][0]);
        }

        output.WriteLine($"per draw (bytes): {string.Join(' ', perDraw)} hits={draws.Cache.Hits} refreshes={draws.Cache.TableRefreshes}");
        Assert.True(draws.Cache.Hits >= 10);
    }

    [Fact]
    public void ARingThatMovesEveryFrameSharesWhatItKeeps()
    {
        var memory = new TestWordMemory { Base = 0x1000, Words = new uint[0x8000] };
        var draws = new Draws(RingPointerPlan(), memory);
        var inputs = draws.InputsOf([0x1000, 0]);
        var perFrame = new List<long>();
        for (var frame = 0; frame < 16; frame++)
        {
            // The title writes the same constants into the next frame's ring.
            WriteRing(memory, 0x10000 + (uint)(frame % 8) * 0x400, 11, 22);
            var bytes = draws.Run(inputs);
            bytes += draws.Run(inputs);
            bytes += draws.Run(inputs);
            perFrame.Add(bytes);
        }

        output.WriteLine($"per frame (bytes): {string.Join(' ', perFrame)} slot_refreshes={draws.Cache.SlotRefreshes} hits={draws.Cache.Hits}");
        Assert.True(draws.Cache.SlotRefreshes >= 14);
        // Only what moved is allocated again: the entry, its ring address and the joined read of it.
        Assert.True(perFrame[^1] <= 1024, $"bytes={perFrame[^1]}");
    }
}
