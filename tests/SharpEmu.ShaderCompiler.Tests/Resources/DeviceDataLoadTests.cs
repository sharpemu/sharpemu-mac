// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed class DeviceDataLoadTests
{
    // The registry is process-wide, so this test owns a hash no other test plans.
    private const ulong LearnedHash = 0xD47A_10AD_5EED_0001;
    private const ulong TableBase = 0x1000;
    private const int DataOffset = 32;

    // s[8:11] is a V# loaded from the table (a binding read); s16 is a dword loaded from
    // the same table that the shader only stores (a data read).
    private static Gen5ShaderProgram BindingAndDataProgram() => Program(
        ScalarLoad(0, 0, destination: 8, count: 4),
        ScalarLoad(8, 0, destination: 16, immediateOffset: DataOffset),
        Vop1(16, "VMovB32", 4, Gen5Operand.Scalar(16)),
        BufferStore(20, 8),
        EndProgram(28));

    [Fact]
    public void LearnedDataLoad_IsLeftToTheDevice_WhileBindingReadsStayFlattened()
    {
        var program = BindingAndDataProgram();
        var flattened = ShaderResourcePlan.Extract(program, ShaderStage.Compute, LearnedHash, 0, 2);
        Assert.True(flattened.Memory.TryGetIndex(8, 0, out var dataMemory));
        Assert.Equal(5, flattened.TableReads.Count);
        Assert.Equal(4, flattened.DataSlotStart);
        var dataSlot = Assert.Single(flattened.DeviceDataSlots);
        Assert.Equal([dataMemory], dataSlot.Value);
        Assert.True(flattened.Memory[dataMemory].PlanningOnly);

        // A bind whose data dword is pending a GPU write learns it, and still reads it.
        var memory = new TestWordMemory { Base = TableBase, Words = new uint[16] };
        memory.Words[DataOffset / sizeof(uint)] = 0x1234;
        var inputs = new ResourceRuntimeInputs
        {
            UserData = [(uint)TableBase, 0],
            ReadMemory = memory.Read,
            DataWritePending = address => address == TableBase + DataOffset,
        };
        Assert.True(RuntimeValueEvaluator.FlattenResourceTable(flattened, inputs, out var table));
        Assert.Equal(0x1234u, table[dataSlot.Key]);
        Assert.Equal(1, DeviceDataLoads.CountFor(flattened.DeviceDataKey));
        Assert.True(DeviceDataLoads.Contains(flattened.DeviceDataKey, dataMemory));

        var replanned = ShaderResourcePlan.Extract(program, ShaderStage.Compute, LearnedHash, 0, 2);
        Assert.Equal(4, replanned.TableReads.Count);
        Assert.Empty(replanned.DeviceDataSlots);
        Assert.False(replanned.Memory[dataMemory].PlanningOnly);
        Assert.False(replanned.FlattenedSlotByMemoryIndex.ContainsKey(dataMemory));
        for (uint component = 0; component < 4; component++)
        {
            Assert.True(replanned.Memory.TryGetIndex(0, component, out var bindingMemory));
            Assert.True(replanned.Memory[bindingMemory].PlanningOnly);
            Assert.Equal(component, replanned.FlattenedSlotByMemoryIndex[bindingMemory]);
        }

        var range = Assert.Single(replanned.DeviceAddressRanges);
        Assert.True(range.Plannable && range.Bounded);
        Assert.Equal(DataOffset, range.FirstByte);

        var resources = ResourceMaterializer.ApplyTo(replanned, ResourceSpecialization.Default(replanned.Info));
        var layout = BindingLayout.Allocate(
            resources.Info,
            BindingLayout.CollectUserDataRegisters(program, 0, 2, replanned.Graph.ExcludedUserDataRegisters),
            BindingLayout.UsesGlobalDataShare(program),
            ShaderCompileRequest.RequiresFlattenedTable(replanned, resources),
            BindingLayout.ReadsShaderBase(program),
            0);
        var request = new ShaderCompileRequest(replanned, resources, layout);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var metal, out var error), error);
        Assert.Contains("sharpemu_load_device_dword", metal.Source[metal.Source.IndexOf("kernel void ", StringComparison.Ordinal)..]);
    }

    [Fact]
    public void BindingReads_AreNeverLearned()
    {
        var program = BindingAndDataProgram();
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, LearnedHash + 1, 0, 2);
        var memory = new TestWordMemory { Base = TableBase, Words = new uint[16] };
        var inputs = new ResourceRuntimeInputs
        {
            UserData = [(uint)TableBase, 0],
            ReadMemory = memory.Read,
            // Every word pending, the V# words included.
            DataWritePending = _ => true,
        };
        Assert.True(RuntimeValueEvaluator.FlattenResourceTable(plan, inputs, out _));
        Assert.Equal(1, DeviceDataLoads.CountFor(plan.DeviceDataKey));
        Assert.Equal(5 - 1, ShaderResourcePlan.Extract(program, ShaderStage.Compute, LearnedHash + 1, 0, 2).TableReads.Count);
    }

    // The compiled transactions read the table themselves, so the learning probe is made
    // for them before they read (and wait for) the word.
    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    public void CompiledPaths_LearnAPendingDataWord(int offset)
    {
        var hash = LearnedHash + 0x10 + (ulong)offset;
        var program = BindingAndDataProgram();
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, hash, 0, 2);
        plan.CompileEvaluatorNow();
        var dataSlot = Assert.Single(plan.DeviceDataSlots);
        var memory = new TestWordMemory { Base = TableBase, Words = new uint[16] };
        memory.Words[DataOffset / sizeof(uint)] = 0x4321;
        var pendingProbes = 0;
        var inputs = new ResourceRuntimeInputs
        {
            UserData = [(uint)TableBase, 0],
            ReadMemory = memory.Read,
            DataWritePending = address =>
            {
                pendingProbes++;
                return address == TableBase + DataOffset;
            },
        };

        uint[] table;
        if (offset == 2)
        {
            Assert.NotNull(plan.CompiledTableTransaction);
            Assert.True(RuntimeValueEvaluator.EvaluateSources(
                plan, [], inputs, plan.CleanFlatSlots, true, out _, out table));
        }
        else
        {
            Assert.NotNull(plan.CompiledTransaction);
            Assert.True(RuntimeValueEvaluator.EvaluateSources(
                plan, plan.MaterializationSources, inputs, plan.CleanFlatSlots, true, out _, out table));
        }

        Assert.Equal(0x4321u, table[dataSlot.Key]);
        Assert.Equal(1, pendingProbes);
        Assert.Equal(1, DeviceDataLoads.CountFor(plan.DeviceDataKey));
        Assert.True(DeviceDataLoads.Contains(plan.DeviceDataKey, dataSlot.Value[0]));

        // Learned: the next bind of the same plan makes no probe at all.
        pendingProbes = 0;
        Assert.True(RuntimeValueEvaluator.EvaluateSources(
            plan, offset == 2 ? [] : plan.MaterializationSources, inputs, plan.CleanFlatSlots, true, out _, out _));
        Assert.Equal(0, pendingProbes);
    }

    [Fact]
    public void CompiledPaths_LearnNothingWhenNoDataWordIsPending()
    {
        var hash = LearnedHash + 0x20;
        var plan = ShaderResourcePlan.Extract(BindingAndDataProgram(), ShaderStage.Compute, hash, 0, 2);
        plan.CompileEvaluatorNow();
        Assert.NotNull(plan.CompiledTableTransaction);
        var memory = new TestWordMemory { Base = TableBase, Words = new uint[16] };
        var inputs = new ResourceRuntimeInputs
        {
            UserData = [(uint)TableBase, 0],
            ReadMemory = memory.Read,
            DataWritePending = _ => false,
        };
        Assert.True(RuntimeValueEvaluator.EvaluateSources(plan, [], inputs, plan.CleanFlatSlots, true, out _, out _));
        Assert.Equal(0, DeviceDataLoads.CountFor(plan.DeviceDataKey));

        // A host with no pending-write probe pays nothing and learns nothing.
        Assert.True(RuntimeValueEvaluator.EvaluateSources(
            plan, [], new ResourceRuntimeInputs { UserData = [(uint)TableBase, 0], ReadMemory = memory.Read },
            plan.CleanFlatSlots, true, out _, out _));
        Assert.Equal(0, DeviceDataLoads.CountFor(plan.DeviceDataKey));
    }

    // s[8:11] is a V#; pc 8 is a buffer load a host may embed as a vertex fetch; s16 and s17
    // are data dwords the shader only stores.
    private const int SecondDataOffset = 48;

    private static Gen5ShaderProgram FetchAndTwoDataProgram() => Program(
        ScalarLoad(0, 0, destination: 8, count: 4),
        BufferLoad(8, 8),
        ScalarLoad(16, 0, destination: 16, immediateOffset: DataOffset),
        ScalarLoad(24, 0, destination: 17, immediateOffset: SecondDataOffset),
        Vop1(32, "VMovB32", 4, Gen5Operand.Scalar(16)),
        Vop1(40, "VMovB32", 5, Gen5Operand.Scalar(17)),
        BufferStore(48, 8, dwords: 2),
        EndProgram(56));

    // An embedded fetch drops its buffer load from the memory table, so the second data load
    // of the embedded plan has the index of the first one in the plain plan. A load learned
    // for one embedding is kept apart from the other: the plain plan still flattens both.
    [Fact]
    public void PlansOfOneHashWithAnotherEmbeddedFetch_LearnSeparateLoads()
    {
        const ulong hash = LearnedHash + 0x30;
        var program = FetchAndTwoDataProgram();
        var plain = ShaderResourcePlan.Extract(program, ShaderStage.Compute, hash, 0, 2);
        var embedded = ShaderResourcePlan.Extract(program, ShaderStage.Compute, hash, 0, 2,
            fixedFunctionVertexLoads: new HashSet<uint> { 8 });
        Assert.Equal(DeviceBindingLoads.PlanKey(hash, null, ShaderStage.Compute, 64, 0, 0, 2), plain.DeviceDataKey);
        Assert.NotEqual(hash, embedded.DeviceDataKey);
        Assert.True(plain.Memory.TryGetIndex(16, 0, out var plainFirst));
        Assert.True(embedded.Memory.TryGetIndex(16, 0, out var embeddedFirst));
        Assert.True(embedded.Memory.TryGetIndex(24, 0, out var embeddedSecond));
        Assert.Equal(plainFirst, embeddedSecond);
        Assert.Equal(2, plain.DeviceDataSlots.Count);
        Assert.Equal(2, embedded.DeviceDataSlots.Count);

        // The embedded plan's bind learns its second data load.
        var memory = new TestWordMemory { Base = TableBase, Words = new uint[16] };
        var inputs = new ResourceRuntimeInputs
        {
            UserData = [(uint)TableBase, 0],
            ReadMemory = memory.Read,
            DataWritePending = address => address == TableBase + SecondDataOffset,
        };
        Assert.True(RuntimeValueEvaluator.FlattenResourceTable(embedded, inputs, out _));
        Assert.Equal(1, DeviceDataLoads.CountFor(embedded.DeviceDataKey));
        Assert.True(DeviceDataLoads.Contains(embedded.DeviceDataKey, embeddedSecond));
        Assert.Equal(0, DeviceDataLoads.CountFor(plain.DeviceDataKey));

        var plainAgain = ShaderResourcePlan.Extract(program, ShaderStage.Compute, hash, 0, 2);
        Assert.Equal(2, plainAgain.DeviceDataSlots.Count);
        Assert.True(plainAgain.FlattenedSlotByMemoryIndex.ContainsKey(plainFirst));

        var embeddedAgain = ShaderResourcePlan.Extract(program, ShaderStage.Compute, hash, 0, 2,
            fixedFunctionVertexLoads: new HashSet<uint> { 8 });
        var kept = Assert.Single(embeddedAgain.DeviceDataSlots);
        Assert.Equal([embeddedFirst], kept.Value);
        Assert.False(embeddedAgain.FlattenedSlotByMemoryIndex.ContainsKey(embeddedSecond));
    }
}
