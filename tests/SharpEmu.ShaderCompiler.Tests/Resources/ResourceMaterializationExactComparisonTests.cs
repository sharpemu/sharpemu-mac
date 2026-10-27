// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed class ResourceMaterializationExactComparisonTests
{
    private sealed class Draws
    {
        private readonly ShaderResourcePlan _plan;
        private readonly ResourceRuntimeInputs _inputs;
        private readonly ResidentGuestBytesReader _resident;
        private ResourceSnapshot _snapshot = new();
        private ResourceSpecialization _specialization = new();
        public readonly TestWordMemory Memory = new() { Base = 0x1000, Words = new uint[0x20000] };
        public readonly ResourceMaterializationCache Cache = new();
        public ResourceSnapshot Snapshot => _snapshot;
        public int Compares, ResidentReads, ProbeReads;
        public bool RefuseComparison, ForceEqual, ObserveOrder, SawReversedOrder;

        public Draws(ShaderResourcePlan plan)
        {
            _plan = plan;
            _resident = Resident;
            _inputs = new ResourceRuntimeInputs
            {
                ReadMemory = Memory.Read, ReadCleanMemory = Memory.Read, CompareResidentMemory = Compare,
            };
        }

        public bool Run(uint[] userData)
        {
            _inputs.UserData = userData;
            return Cache.Materialize(_plan, _inputs, _resident, ref _snapshot, ref _specialization, out _);
        }

        public void AssertFullWalk(uint[] userData)
        {
            var expected = new ResourceSnapshot();
            var specialization = new ResourceSpecialization();
            Assert.True(ResourceMaterializer.Materialize(_plan, Inputs(userData, Memory.Read, Memory.Read),
                ref expected, ref specialization));
            Assert.Equal(expected.Buffers.Select(x => x.ToArray()), _snapshot.Buffers.Select(x => x.ToArray()));
            Assert.Equal(expected.FlattenedResourceTable, _snapshot.FlattenedResourceTable);
            Assert.Equal(specialization, _specialization);
        }

        private bool Compare(ReadOnlySpan<ulong> addresses, ReadOnlySpan<int> lengths, ReadOnlySpan<int> offsets,
            ReadOnlySpan<byte> bytes)
        {
            Compares++;
            if (RefuseComparison) return false;
            if (ObserveOrder)
                for (var index = 1; index < addresses.Length; index++)
                    SawReversedOrder |= addresses[index] < addresses[index - 1];
            if (ForceEqual) return true;
            for (var range = 0; range < addresses.Length; range++)
                for (var offset = 0; offset < lengths[range]; offset += 4)
                    if (!Memory.Read(addresses[range] + (ulong)offset, out var word) ||
                        word != BitConverter.ToUInt32(bytes.Slice(offsets[range] + offset, 4))) return false;
            return true;
        }

        private bool Resident(ulong address, Span<byte> destination, bool clean)
        {
            ResidentReads++;
            if (destination.Length == sizeof(uint)) ProbeReads++;
            for (var offset = 0; offset < destination.Length; offset += 4)
            {
                if (!Memory.Read(address + (ulong)offset, out var word)) return false;
                BitConverter.TryWriteBytes(destination[offset..], word);
            }
            return true;
        }

        public void Write(ulong address, params uint[] values)
        {
            for (var index = 0; index < values.Length; index++) Memory.At(address + (ulong)index * 4) = values[index];
        }
    }

    private static ShaderResourcePlan TablePlan()
    {
        var program = new List<Gen5ShaderInstruction>();
        for (uint index = 0; index < 6; index++)
            program.Add(ScalarLoad(index * 8, 0, 8 + index * 4, 4, (int)index * 16));
        for (uint index = 0; index < 6; index++) program.Add(BufferLoad(48 + index * 8, 8 + index * 4));
        program.Add(EndProgram(96));
        return Extract(Program([.. program]), userDataCount: 2);
    }

    [Fact]
    public void ReturningTablePointersCompareWithoutCopiesAndRefusalsStillRead()
    {
        if (!ResourceMaterializationCache.InPlaceValidateEnabled) return;
        var draws = new Draws(TablePlan());
        uint[][] pointers = Enumerable.Range(0, 8).Select(i => new uint[] { 0x10000u + (uint)i * 0x300, 0 }).ToArray();
        foreach (var pointer in pointers)
            for (uint slot = 0; slot < 6; slot++) draws.Write(pointer[0] + slot * 16, 0x50000 + slot * 0x1000, 0, 256, 0x2_0000);
        for (var draw = 0; draw < 40; draw++) Assert.True(draws.Run(pointers[draw % pointers.Length]));
        var hits = draws.Cache.Hits;
        var compares = draws.Compares;
        var reads = draws.ResidentReads;
        Assert.True(draws.Run(pointers[5]));
        Assert.Equal(hits + 1, draws.Cache.Hits);
        Assert.True(draws.Compares > compares);
        if (!ResourceMaterializationCache.InPlaceValidateVerify) Assert.Equal(reads, draws.ResidentReads);
        draws.AssertFullWalk(pointers[5]);

        draws.RefuseComparison = true;
        reads = draws.ResidentReads;
        hits = draws.Cache.Hits;
        Assert.True(draws.Run(pointers[5]));
        Assert.Equal(hits + 1, draws.Cache.Hits);
        Assert.True(draws.ResidentReads > reads);
        draws.AssertFullWalk(pointers[5]);

        draws.RefuseComparison = false;
        draws.Memory.At(pointers[5][0] + 8) = 512;
        Assert.True(draws.Run(pointers[5]));
        draws.AssertFullWalk(pointers[5]);
        Assert.Equal(512u, draws.Snapshot.Buffers[0][2]);
        draws.Memory.FailAddress = pointers[5][0] + 4;
        hits = draws.Cache.Hits;
        Assert.False(draws.Run(pointers[5]));
        Assert.Equal(hits, draws.Cache.Hits);
    }

    [Fact]
    public void IndependentlyMovedPointersCompareExactRangesEvenWhenTheirOrderReverses()
    {
        if (!ResourceMaterializationCache.InPlaceValidateEnabled) return;
        var plan = Extract(Program(ScalarLoad(0, 0, 8, 4), ScalarLoad(8, 2, 12, 4), ScalarLoad(16, 4, 16, 4),
            BufferLoad(24, 8), BufferLoad(32, 12), BufferLoad(40, 16), EndProgram(48)), userDataCount: 6);
        var draws = new Draws(plan) { ObserveOrder = true };
        draws.Write(0x7000, 0x80000, 0, 128, 0x2_0000);
        for (uint draw = 0; draw < 32; draw++)
        {
            uint[] pointers = [0x10000 + draw * 0x1000, 0, 0x30800 - draw * 0x1000, 0, 0x7000, 0];
            draws.Write(pointers[0], 0x60000, 0, 256, 0x2_0000);
            draws.Write(pointers[2], 0x70000, 0, 512, 0x2_0000);
            Assert.True(draws.Run(pointers));
            draws.AssertFullWalk(pointers);
        }
        Assert.True(draws.Cache.Hits > 0);
        Assert.True(draws.Compares > 0);
        Assert.True(draws.SawReversedOrder);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MemoryLoadedBaseChangesStillReachTheOriginalProbeAndRefreshPath(bool movingTable)
    {
        var plan = Extract(Program(ScalarLoad(0, 0, 8, 4), ScalarBufferLoad(4, 8, 12, 4),
            Vop1(8, "VMovB32", 0, Gen5Operand.Scalar(12)), BufferLoad(12, 8), EndProgram(16)), userDataCount: 2);
        var draws = new Draws(plan);
        uint[] pointer = [0x1000, 0];
        for (uint draw = 0; draw < 20; draw++)
        {
            // First confirm the raw table pointer independently, then change the base
            // loaded from that table. Entries share immutable input arrays like ShaderSource.
            var valueDraw = movingTable && draw < 8 ? 0 : draw;
            var address = 0x10000 + valueDraw * 0x100;
            pointer = [movingTable ? 0x2000 + draw * 0x100 : 0x1000, 0];
            draws.Write(pointer[0], address, 0, 256, 0x2_0000);
            draws.Write(address, valueDraw * 10, valueDraw * 10 + 1, valueDraw * 10 + 2, valueDraw * 10 + 3);
            Assert.True(draws.Run(pointer), $"draw={draw} movingTable={movingTable} table=0x{pointer[0]:X} base=0x{address:X}");
            draws.AssertFullWalk(pointer);
            if (movingTable && draw == 7)
                Assert.True(draws.Cache.Hits > 0,
                    $"Raw table pointer did not warm: hits={draws.Cache.Hits} misses={draws.Cache.Misses} compares={draws.Compares} probes={draws.ProbeReads}");
        }
        draws.Write(pointer[0] + 8, 512);
        Assert.True(draws.Run(pointer), $"Resized final table failed: movingTable={movingTable} table=0x{pointer[0]:X}");
        draws.AssertFullWalk(pointer);
        Assert.Equal(512u, draws.Snapshot.Buffers[0][2]);
        if (movingTable && ResourceMaterializationCache.InPlaceValidateEnabled)
            Assert.True(draws.ProbeReads > 0,
                $"Expected a scalar base-word probe after learned raw-pointer movement: hits={draws.Cache.Hits} misses={draws.Cache.Misses} compares={draws.Compares} residentReads={draws.ResidentReads} probes={draws.ProbeReads}");
    }

    [Fact]
    public void WarmRebasedComparisonDoesNotAllocateAndEmptyRangesDoNotCallTheComparer()
    {
        if (!ResourceMaterializationCache.InPlaceValidateEnabled) return;
        var plan = Extract(Program(ScalarLoad(0, 0, 8, 4), BufferLoad(8, 8), EndProgram(16)), userDataCount: 2);
        var draws = new Draws(plan);
        uint[] final = [];
        for (uint draw = 0; draw < 20; draw++)
        {
            final = [0x10000 + draw * 0x100, 0];
            draws.Write(final[0], 0x60000, 0, 256, 0x2_0000);
            Assert.True(draws.Run(final));
        }
        for (var warmup = 0; warmup < 20; warmup++) Assert.True(draws.Run(final));
        Assert.True(draws.Compares > 0);
        var before = GC.GetAllocatedBytesForCurrentThread();
        var result = draws.Run(final);
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(result);
        Assert.Equal(0, allocated);

        var empty = new Draws(Extract(Program(BufferLoad(0, 4), EndProgram(8)), userDataCount: 8));
        for (uint draw = 0; draw < 20; draw++)
            Assert.True(empty.Run([0, 0, 0, 0, 0x10000 + draw * 0x100, 0, 256, 0x2_0000]));
        Assert.True(empty.Cache.Hits > 0);
        Assert.Equal(0, empty.Compares);
        Assert.Equal(0, empty.ResidentReads);
    }

    [Fact]
    public void VerificationReadsInsteadOfTrustingAnIncorrectSuccessfulComparison()
    {
        if (!ResourceMaterializationCache.InPlaceValidateEnabled || !ResourceMaterializationCache.InPlaceValidateVerify) return;
        var plan = Extract(Program(ScalarLoad(0, 0, 8, 4), BufferLoad(8, 8), EndProgram(16)), userDataCount: 2);
        var draws = new Draws(plan);
        for (uint draw = 0; draw < 12; draw++)
        {
            uint[] pointer = [0x10000 + draw * 0x100, 0];
            draws.Write(pointer[0], 0x60000, 0, 256, 0x2_0000);
            Assert.True(draws.Run(pointer));
        }
        draws.ForceEqual = true;
        uint[] changed = [0x10B00, 0];
        draws.Memory.At(changed[0] + 8) = 512;
        var reads = draws.ResidentReads;
        Assert.True(draws.Run(changed));
        Assert.True(draws.ResidentReads > reads);
        draws.AssertFullWalk(changed);
        Assert.Equal(512u, draws.Snapshot.Buffers[0][2]);
    }
}
