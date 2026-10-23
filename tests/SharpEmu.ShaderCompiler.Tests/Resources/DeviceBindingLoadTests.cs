// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed partial class DeviceBindingLoadTests
{
    // The registry is process-wide, so every test owns hashes no other test plans.
    private const ulong BaseHash = 0xB1D0_10AD_5EED_0000;
    private const ulong TableBase = 0x1000;

    // s[4:7] is a V# loaded from the address in s[0:1] (ScalarAddressWord binding reads);
    // the shader loads through it.
    private static Gen5ShaderProgram AddressBoundBufferLoad() => Program(
        ScalarLoad(0, 0, 4, 4), BufferLoad(8, 4), EndProgram(16));

    // s8 is a table index loaded through the V# in s[0:3] (a ScalarBufferWord binding read);
    // s[12:15] is the V# at that index, which the shader loads through.
    private static Gen5ShaderProgram IndexedBufferLoad() => Program(
        ScalarBufferLoad(0, 0, destination: 8),
        ScalarBufferLoad(8, 0, destination: 12, count: 4, dynamicOffsetRegister: 8),
        BufferLoad(16, 12),
        EndProgram(24));

    // The same V# as AddressBoundBufferLoad, stored through.
    private static Gen5ShaderProgram AddressBoundBufferStore() => Program(
        ScalarLoad(0, 0, 4, 4), BufferStore(8, 4), EndProgram(16));

    // An image descriptor loaded from the address in s[0:1].
    private static Gen5ShaderProgram AddressBoundImageLoad() => Program(
        ScalarLoad(0, 0, 8, 8), Image(8, "ImageLoad", 8), EndProgram(16));

    // A V# the shader reads from its registers: a device descriptor, or a runtime one where
    // unvalidated address words are kept host-evaluated (ResourceTracker).
    private static bool OnDevice(MemoryAccessInfo memory) =>
        memory.DeviceDescriptor || memory.BufferDescriptor?.Provenance == BufferDescriptorProvenance.Runtime;

    private static ShaderResourcePlan Plan(Gen5ShaderProgram program, ulong hash) =>
        ShaderResourcePlan.Extract(program, ShaderStage.Compute, hash, 0, 8);

    private static ulong Key(ulong hash) => DeviceBindingLoads.PlanKey(hash, null, ShaderStage.Compute, 64, 0, 0, 8);

    private sealed class PendingMemory
    {
        public required TestWordMemory Memory { get; init; }
        public HashSet<ulong> Pending { get; } = [];
        public List<ulong> ReadAddresses { get; } = [];

        public bool Read(ulong address, out uint word)
        {
            ReadAddresses.Add(address);
            return Memory.Read(address, out word);
        }

        public ResourceRuntimeInputs Inputs(uint[] userData) => new()
        {
            UserData = userData,
            ReadMemory = Read,
            ReadCleanMemory = Memory.Read,
            ResourceWordPending = Pending.Contains,
        };
    }

    // One materialization the way the program cache runs it: the attempt names the first
    // binding word it read as unknown.
    private static (bool Materialized, DeviceBindingLoads.Request? Deferred, ResourceSnapshot Snapshot) Attempt(
        ShaderResourcePlan plan, ResourceRuntimeInputs inputs, bool defer = true)
    {
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        DeviceBindingLoads.BeginAttempt(defer);
        bool materialized;
        DeviceBindingLoads.Request? deferred;
        try
        {
            materialized = ResourceMaterializer.Materialize(plan, inputs, ref snapshot, ref specialization);
        }
        finally
        {
            deferred = DeviceBindingLoads.EndAttempt();
        }

        return (materialized, deferred, snapshot);
    }

    [Fact]
    public void LandedAddressWord_IsReadOnTheHost()
    {
        var hash = BaseHash + 1;
        var plan = Plan(AddressBoundBufferLoad(), hash);
        var memory = new PendingMemory { Memory = new TestWordMemory { Base = TableBase, Words = [0x2000, 0, 256, 0] } };

        var (materialized, deferred, snapshot) = Attempt(plan, memory.Inputs([(uint)TableBase, 0]));

        Assert.True(materialized);
        Assert.Null(deferred);
        Assert.Contains(TableBase, memory.ReadAddresses);
        Assert.Equal(0x2000u, snapshot.Buffers[0][0]);
        Assert.Equal(0, DeviceBindingLoads.CountFor(Key(hash)));
    }

    [Fact]
    public void PendingAddressWord_IsUnknown_AndTheReplanLeavesTheDescriptorToTheDevice()
    {
        var hash = BaseHash + 2;
        var program = AddressBoundBufferLoad();
        var plan = Plan(program, hash);
        Assert.True(plan.Memory.TryGetIndex(0, 0, out var wordMemory));
        Assert.True(plan.Memory[wordMemory].PlanningOnly);
        var memory = new PendingMemory { Memory = new TestWordMemory { Base = TableBase, Words = [0x2000, 0, 256, 0] } };
        memory.Pending.Add(TableBase);

        var (_, deferred, _) = Attempt(plan, memory.Inputs([(uint)TableBase, 0]));

        // The pending word is never read (so never waited for), and is named for a re-plan.
        Assert.DoesNotContain(TableBase, memory.ReadAddresses);
        var request = Assert.IsType<DeviceBindingLoads.Request>(deferred);
        Assert.Equal(Key(hash), request.Hash);
        Assert.Equal(wordMemory, request.MemoryIndex);
        Assert.Equal(ScalarValueKind.ScalarAddressWord, request.Kind);

        Assert.True(DeviceBindingLoads.Learn(Key(hash), request.MemoryIndex));
        var replanned = Plan(program, hash);
        Assert.True(DeviceBindingLoads.Accepts(plan, replanned));
        Assert.False(replanned.FlattenedSlotByMemoryIndex.ContainsKey(wordMemory));
        Assert.False(replanned.Memory[wordMemory].PlanningOnly);
        Assert.Empty(replanned.Info.Buffers);
        Assert.True(replanned.Memory.TryGetIndex(8, 0, out var loadMemory));
        Assert.True(OnDevice(replanned.Memory[loadMemory]));

        // The re-planned shader binds while the word is still pending without reading it.
        memory.ReadAddresses.Clear();
        var (materialized, again, _) = Attempt(replanned, memory.Inputs([(uint)TableBase, 0]));
        Assert.True(materialized);
        Assert.Null(again);
        Assert.DoesNotContain(TableBase, memory.ReadAddresses);
    }

    [Fact]
    public void PendingTableIndex_IsUnknown_AndTheReplanLeavesTheIndexedDescriptorToTheDevice()
    {
        var hash = BaseHash + 3;
        var program = IndexedBufferLoad();
        var plan = Plan(program, hash);
        Assert.True(plan.Memory.TryGetIndex(0, 0, out var indexMemory));
        Assert.True(plan.FlattenedSlotByMemoryIndex.ContainsKey(indexMemory));
        // The table: index 16 at offset 0, the V# at byte 16.
        var memory = new PendingMemory
        {
            Memory = new TestWordMemory { Base = TableBase, Words = [16, 0, 0, 0, 0x2000, 0, 256, 0] },
        };
        uint[] userData = [(uint)TableBase, 0, 32, 0];

        var landed = Attempt(plan, memory.Inputs(userData));
        Assert.True(landed.Materialized);
        Assert.Null(landed.Deferred);
        // The indexed V# is read on the device either way; the host reads the index for it.
        Assert.Contains(TableBase, memory.ReadAddresses);
        Assert.True(plan.Memory.TryGetIndex(16, 0, out var vectorLoad));
        Assert.True(OnDevice(plan.Memory[vectorLoad]));

        memory.ReadAddresses.Clear();
        memory.Pending.Add(TableBase);
        var (_, deferred, _) = Attempt(plan, memory.Inputs(userData));
        Assert.DoesNotContain(TableBase, memory.ReadAddresses);
        var request = Assert.IsType<DeviceBindingLoads.Request>(deferred);
        Assert.Equal(indexMemory, request.MemoryIndex);
        Assert.Equal(ScalarValueKind.ScalarBufferWord, request.Kind);

        Assert.True(DeviceBindingLoads.Learn(Key(hash), request.MemoryIndex));
        var replanned = Plan(program, hash);
        Assert.True(DeviceBindingLoads.Accepts(plan, replanned));
        Assert.False(replanned.FlattenedSlotByMemoryIndex.ContainsKey(indexMemory));
        Assert.True(replanned.Memory.TryGetIndex(16, 0, out var loadMemory));
        Assert.True(OnDevice(replanned.Memory[loadMemory]));

        memory.ReadAddresses.Clear();
        var (materialized, again, _) = Attempt(replanned, memory.Inputs(userData));
        Assert.True(materialized);
        Assert.Null(again);
        Assert.DoesNotContain(TableBase, memory.ReadAddresses);
    }

    [Fact]
    public void ReplanThatStoresWithoutAHostRange_IsNotAccepted()
    {
        var hash = BaseHash + 4;
        var program = AddressBoundBufferStore();
        var plan = Plan(program, hash);
        Assert.True(plan.Memory.TryGetIndex(0, 0, out var wordMemory));
        Assert.True(DeviceBindingLoads.Learn(Key(hash), wordMemory));
        var replanned = Plan(program, hash);
        Assert.False(DeviceBindingLoads.Accepts(plan, replanned));

        // Refused: the word is read on the host again, counted as waited, never deferred.
        DeviceBindingLoads.Refuse(Key(hash), wordMemory);
        Assert.False(DeviceBindingLoads.Contains(Key(hash), wordMemory));
        Assert.Null(DeviceBindingLoads.For(Key(hash)));
        var memory = new PendingMemory { Memory = new TestWordMemory { Base = TableBase, Words = [0x2000, 0, 256, 0] } };
        memory.Pending.Add(TableBase);
        var (materialized, deferred, _) = Attempt(Plan(program, hash), memory.Inputs([(uint)TableBase, 0]));
        Assert.True(materialized);
        Assert.Null(deferred);
        Assert.Contains(TableBase, memory.ReadAddresses);
    }

    [Fact]
    public void ImageDescriptorFromAPendingWord_CannotBePlannedOnTheDevice()
    {
        var hash = BaseHash + 5;
        var program = AddressBoundImageLoad();
        var plan = Plan(program, hash);
        Assert.Single(plan.Info.Images);
        Assert.True(plan.Memory.TryGetIndex(0, 0, out var wordMemory));
        Assert.True(DeviceBindingLoads.Learn(Key(hash), wordMemory));
        Assert.Throws<ResourcePlanException>(() => Plan(program, hash));
        DeviceBindingLoads.Refuse(Key(hash), wordMemory);
        Assert.Single(Plan(program, hash).Info.Images);
    }

    [Fact]
    public void OutsideAnAttempt_APendingWordIsReadAndWaitedFor()
    {
        var hash = BaseHash + 6;
        var plan = Plan(AddressBoundBufferLoad(), hash);
        var memory = new PendingMemory { Memory = new TestWordMemory { Base = TableBase, Words = [0x2000, 0, 256, 0] } };
        memory.Pending.Add(TableBase);

        var (materialized, deferred, snapshot) = Attempt(plan, memory.Inputs([(uint)TableBase, 0]), defer: false);

        Assert.True(materialized);
        Assert.Null(deferred);
        Assert.Contains(TableBase, memory.ReadAddresses);
        Assert.Equal(0x2000u, snapshot.Buffers[0][0]);
    }

    // Two V#s loaded from the table s[0:1] points at, at bytes 0 and 16; the shader loads
    // through both. One GPU write lands the whole table.
    private static Gen5ShaderProgram TwoBufferTable() => Program(
        ScalarLoad(0, 0, 4, 4), ScalarLoad(8, 0, 8, 4, immediateOffset: 16), BufferLoad(16, 4), BufferLoad(24, 8), EndProgram(32));

    // Demon's Souls' logo movies: each bind that met a pending word learned that one load, one
    // re-plan and one pipeline compile per word. The deferred word names every other pending
    // word read through the same table, and one re-plan takes them all.
    [Fact]
    public void PendingTableWords_AreAllNamed_AndOneReplanLeavesTheWholeTableToTheDevice()
    {
        var hash = BaseHash + 7;
        var program = TwoBufferTable();
        var plan = Plan(program, hash);
        Assert.Equal(2, plan.Info.Buffers.Count);
        Assert.True(plan.Memory.TryGetIndex(0, 0, out var first));
        Assert.True(plan.Memory.TryGetIndex(8, 0, out var second));
        var memory = new PendingMemory
        {
            Memory = new TestWordMemory { Base = TableBase, Words = [0x2000, 0, 256, 0, 0x3000, 0, 256, 0] },
        };
        for (var offset = 0UL; offset < 32; offset += 4)
        {
            memory.Pending.Add(TableBase + offset);
        }

        var (_, deferred, _) = Attempt(plan, memory.Inputs([(uint)TableBase, 0]));

        var request = Assert.IsType<DeviceBindingLoads.Request>(deferred);
        Assert.Equal(first, request.MemoryIndex);
        Assert.NotNull(request.TableWords);
        Assert.Contains(second, request.TableWords!);
        Assert.DoesNotContain(first, request.TableWords!);
        Assert.DoesNotContain(TableBase, memory.ReadAddresses);

        Assert.True(DeviceBindingLoads.Learn(Key(hash), request.MemoryIndex));
        foreach (var word in request.TableWords!)
        {
            DeviceBindingLoads.Learn(Key(hash), word);
        }

        var replanned = Plan(program, hash);
        Assert.True(DeviceBindingLoads.Accepts(plan, replanned));
        Assert.Empty(replanned.Info.Buffers);

        // The next bind meets the table pending again and reads nothing of it: no second re-plan.
        memory.ReadAddresses.Clear();
        var (materialized, again, _) = Attempt(replanned, memory.Inputs([(uint)TableBase, 0]));
        Assert.True(materialized);
        Assert.Null(again);
        Assert.DoesNotContain(memory.ReadAddresses, address => address >= TableBase && address < TableBase + 32);
    }

    // A word of the same table that has landed is not named: it stays read on the host.
    [Fact]
    public void LandedTableWords_AreNotNamed()
    {
        var hash = BaseHash + 8;
        var plan = Plan(TwoBufferTable(), hash);
        var memory = new PendingMemory
        {
            Memory = new TestWordMemory { Base = TableBase, Words = [0x2000, 0, 256, 0, 0x3000, 0, 256, 0] },
        };
        memory.Pending.Add(TableBase);

        var (_, deferred, _) = Attempt(plan, memory.Inputs([(uint)TableBase, 0]));

        var request = Assert.IsType<DeviceBindingLoads.Request>(deferred);
        Assert.True(request.TableWords is null || request.TableWords.Count == 0);
    }

    // A load learned stays learned when its plan still reads it on the host: its word is
    // waited for there, and the key neither refuses it nor plans it back (which compiled one
    // more program, and the next start learned it again: keys flipped 117-126 times).
    [Fact]
    public void ALoadKeptOnTheHost_StaysLearned_AndItsWordIsWaitedFor()
    {
        var hash = BaseHash + 9;
        var program = AddressBoundBufferLoad();
        var plan = Plan(program, hash);
        Assert.True(plan.Memory.TryGetIndex(0, 0, out var wordMemory));
        Assert.True(DeviceBindingLoads.Learn(Key(hash), wordMemory));

        DeviceBindingLoads.KeepOnHost(Key(hash), wordMemory);

        Assert.True(DeviceBindingLoads.Contains(Key(hash), wordMemory));
        Assert.False(DeviceBindingLoads.IsRefused(Key(hash), wordMemory));
        Assert.True(DeviceBindingLoads.IsKeptOnHost(Key(hash), wordMemory));
        Assert.True(DeviceBindingLoads.IsCurrent(Key(hash), [wordMemory]));
        // A plan that reads it on the host waits for it instead of naming it for a re-plan.
        var memory = new PendingMemory { Memory = new TestWordMemory { Base = TableBase, Words = [0x2000, 0, 256, 0] } };
        memory.Pending.Add(TableBase);
        var (materialized, deferred, _) = Attempt(plan, memory.Inputs([(uint)TableBase, 0]));
        Assert.True(materialized);
        Assert.Null(deferred);
        Assert.Contains(TableBase, memory.ReadAddresses);
        // Learning it again changes nothing.
        Assert.False(DeviceBindingLoads.Learn(Key(hash), wordMemory));
        Assert.True(DeviceBindingLoads.Contains(Key(hash), wordMemory));
    }

    // A key re-planned RebuildCap times in a run waits for its pending words from then on.
    [Fact]
    public void AKeyAtItsRebuildCap_WaitsForItsPendingWords()
    {
        var hash = BaseHash + 10;
        var plan = Plan(AddressBoundBufferLoad(), hash);
        var memory = new PendingMemory { Memory = new TestWordMemory { Base = TableBase, Words = [0x2000, 0, 256, 0] } };
        memory.Pending.Add(TableBase);
        var cap = DeviceBindingLoads.RebuildCap;
        Assert.True(cap > 1);
        for (var rebuild = 1; rebuild < cap; rebuild++)
        {
            Assert.False(DeviceBindingLoads.NoteRebuild(Key(hash), out var rebuilds));
            Assert.Equal(rebuild, rebuilds);
        }

        Assert.False(DeviceBindingLoads.IsCapped(Key(hash)));
        Assert.NotNull(Attempt(plan, memory.Inputs([(uint)TableBase, 0])).Deferred);

        // The rebuild that reaches the cap says so once.
        Assert.True(DeviceBindingLoads.NoteRebuild(Key(hash), out _));
        Assert.False(DeviceBindingLoads.NoteRebuild(Key(hash), out _));
        Assert.True(DeviceBindingLoads.IsCapped(Key(hash)));

        memory.ReadAddresses.Clear();
        var (materialized, deferred, _) = Attempt(plan, memory.Inputs([(uint)TableBase, 0]));
        Assert.True(materialized);
        Assert.Null(deferred);
        Assert.Contains(TableBase, memory.ReadAddresses);
    }

    // A plan with no embedded fetch learns under its hash; an embedded fetch makes another
    // key, the same for the same fetch loads in any order and another for other loads.
    [Fact]
    public void KeyFor_TellsPlansApartByTheirEmbeddedFetch()
    {
        const ulong hash = BaseHash + 0x70;
        Assert.Equal(hash, DeviceBindingLoads.KeyFor(hash, null));
        Assert.Equal(hash, DeviceBindingLoads.KeyFor(hash, new HashSet<uint>()));

        var fetch = DeviceBindingLoads.KeyFor(hash, new HashSet<uint> { 8, 24 });
        Assert.NotEqual(hash, fetch);
        Assert.Equal(fetch, DeviceBindingLoads.KeyFor(hash, new SortedSet<uint>(Comparer<uint>.Create((a, b) => b.CompareTo(a))) { 8, 24 }));
        Assert.NotEqual(fetch, DeviceBindingLoads.KeyFor(hash, new HashSet<uint> { 8 }));
        Assert.NotEqual(fetch, DeviceBindingLoads.KeyFor(hash + 1, new HashSet<uint> { 8, 24 }));

        var program = AddressBoundBufferLoad();
        Assert.Equal(Key(hash), Plan(program, hash).DeviceBindingKey);
        var embedded = ShaderResourcePlan.Extract(program, ShaderStage.Compute, hash, 0, 8,
            fixedFunctionVertexLoads: new HashSet<uint> { 8 });
        Assert.Equal(DeviceBindingLoads.PlanKey(hash, new HashSet<uint> { 8 }, ShaderStage.Compute, 64, 0, 0, 8), embedded.DeviceBindingKey);
        Assert.Equal(embedded.DeviceBindingKey, embedded.DeviceDataKey);
    }
}
