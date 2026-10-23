// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed partial class DeviceBindingLoadTests
{
    [Theory]
    [InlineData(false, 4u)]
    [InlineData(true, 4u)]
    [InlineData(false, 8u)]
    [InlineData(true, 8u)]
    public void BulkAndCompiledReadsDeferWithoutPublishingAStaleDescriptor(bool compiled, uint width)
    {
        var program = width == 4 ? AddressBoundBufferLoad() : AddressBoundImageLoad();
        var plan = Plan(program, BaseHash + 0x100 + width + (compiled ? 16u : 0u));
        if (compiled) plan.CompileEvaluatorNow();
        var words = new uint[] { 0x2000, 0, 256, 0, 1, 2, 3, 4 };
        var memory = new TestWordMemory { Base = TableBase, Words = words };
        var snapshot = new ResourceSnapshot { UserData = [0xDEAD] };
        var specialization = new ResourceSpecialization();
        var before = snapshot;
        var reads = 0;
        var inputs = new ResourceRuntimeInputs
        {
            UserData = [(uint)TableBase, 0],
            ReadMemory = (ulong at, out uint value) => { reads++; return memory.Read(at, out value); },
            // Even a wrapper that mistakenly offers old bytes cannot bypass deferral.
            ReadResidentMemory = (ulong at, Span<byte> data, bool clean) =>
            {
                data.Clear(); return true;
            },
            ResourceWordPending = at => at >= TableBase && at < TableBase + width * 4,
        };
        DeviceBindingLoads.BeginAttempt(true);
        try
        {
            Assert.False(ResourceMaterializer.Materialize(plan, inputs, ref snapshot, ref specialization, out var failure));
            Assert.Equal(ResourceMaterializationFailure.DeferredPendingLoad, failure);
            Assert.Same(before, snapshot);
            Assert.Equal(0, reads);
        }
        finally { Assert.NotNull(DeviceBindingLoads.EndAttempt()); }
    }

    [Theory]
    [InlineData(0u, 1u)]
    [InlineData(1u, 0u)]
    public void HostFallbackReadsUpdatedPointerLowAndHighDwords(uint oldHigh, uint newHigh)
    {
        var plan = Plan(AddressBoundBufferLoad(), BaseHash + 0x200 + oldHigh);
        var memory = new PendingMemory { Memory = new TestWordMemory { Base = TableBase, Words = [0x2000, oldHigh, 256, 0] } };
        memory.Pending.Add(TableBase + 4);
        var (ok, request, _) = Attempt(plan, memory.Inputs([(uint)TableBase, 0]));
        Assert.False(ok);
        Assert.Equal(1u, request!.Value.Component);
        memory.Memory.Words[0] = 0x3000;
        memory.Memory.Words[1] = newHigh;
        var (fallback, _, snapshot) = Attempt(plan, memory.Inputs([(uint)TableBase, 0]), defer: false);
        Assert.True(fallback);
        Assert.Equal(new uint[] { 0x3000, newHigh, 256, 0 }, snapshot.Buffers[0]);
    }

    [Fact]
    public void AWriteCannotBecomeUnknownEvenWhenAnotherWriteWasAlreadyUnknown()
    {
        var program = Program(ScalarLoad(0, 0, 4, 4), BufferStore(8, 4),
            GlobalAccess(16, "FlatStoreDword", 0), EndProgram(24));
        var plan = Plan(program, BaseHash + 0x300);
        Assert.True(plan.Memory.TryGetIndex(0, 0, out var load));
        var next = ShaderResourcePlan.Extract(program, ShaderStage.Compute, plan.Hash, 0, 8, deviceBindingMemory: new HashSet<int> { load });
        Assert.False(DeviceBindingLoads.Accepts(plan, next, out var reason));
        Assert.Contains("write", reason);
    }

    [Fact]
    public void AllGraphShapingOptionsHaveDistinctLearnedLoadKeys()
    {
        var hash = BaseHash + 0x400;
        var key = DeviceBindingLoads.PlanKey(hash, null, ShaderStage.Compute, 64, 0, 0, 8);
        Assert.NotEqual(key, DeviceBindingLoads.PlanKey(hash, null, ShaderStage.Compute, 32, 0, 0, 8));
        Assert.NotEqual(key, DeviceBindingLoads.PlanKey(hash, null, ShaderStage.Vertex, 64, 0, 0, 8));
        Assert.NotEqual(key, DeviceBindingLoads.PlanKey(hash, null, ShaderStage.Compute, 64, 4, 0, 8));
        Assert.NotEqual(key, DeviceBindingLoads.PlanKey(hash, null, ShaderStage.Compute, 64, 0, 4, 8));
        Assert.NotEqual(key, DeviceBindingLoads.PlanKey(hash, null, ShaderStage.Compute, 64, 0, 0, 4));
    }
    [Fact]
    public void AWaitReportKeepsItsConsumerWriterAndSynchronizedValue()
    {
        _ = BindingWaitProvenance.TakeReport();
        var plan = Plan(AddressBoundBufferLoad(), BaseHash + 0x500);
        var read = plan.TableReads[0].Value;
        MaterializingShader.Hash = plan.Hash;
        MaterializingShader.BeginRead(plan, read);
        try
        {
            BindingWaitProvenance.Wait(42, "StorageBuffer", TableBase, 1234, 7);
            BindingWaitProvenance.Consumed(0xFEDCBA98);
            var report = BindingWaitProvenance.TakeReport();
            Assert.Contains($"shader={plan.Hash:X16}", report);
            Assert.Contains("pc=0 component=0", report);
            Assert.Contains("writer=42/StorageBuffer", report);
            Assert.Contains("waits=1", report);
            Assert.Contains("last_value=FEDCBA98 frame=7", report);
        }
        finally { MaterializingShader.EndRead(); MaterializingShader.Hash = 0; }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void WritesAndAtomicsWhoseDescriptorWouldBecomeUnknownKeepTheHostReference(bool atomic)
    {
        var program = Program(ScalarLoad(0, 0, 4, 4), atomic ? BufferAtomicAdd(8, 4) : BufferStore(8, 4), EndProgram(16));
        var plan = Plan(program, BaseHash + 0x600 + (atomic ? 1u : 0u));
        plan.Memory.TryGetIndex(0, 0, out var index);
        try
        {
            var next = ShaderResourcePlan.Extract(program, ShaderStage.Compute, plan.Hash, 0, 8, deviceBindingMemory: new HashSet<int> { index });
            Assert.False(DeviceBindingLoads.Accepts(plan, next));
        }
        catch (ResourcePlanException) { /* A host-required atomic V# can reject during tracking. */ }
        var memory = new PendingMemory { Memory = new TestWordMemory { Base = TableBase, Words = [0x3000, 1, 64, 0] } };
        memory.Pending.Add(TableBase);
        var (ok, deferred, snapshot) = Attempt(plan, memory.Inputs([(uint)TableBase, 0]), false);
        Assert.True(ok);
        Assert.Null(deferred);
        Assert.Equal(new uint[] { 0x3000, 1, 64, 0 }, snapshot.Buffers[0]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OnlyAReadOffsetCoveredByACompleteConstantByteBufferIsDataOnly(bool dynamicDescriptor)
    {
        var program = Program(MoveScalar(0, 8, 0x2000), MoveScalar(4, 9, 0), dynamicDescriptor ? MoveScalarRegister(8, 10, 4) : MoveScalar(8, 10, 64), MoveScalar(12, 11, 0),
            ScalarLoad(16, 0, 16), Vop1(24, "VMovB32", 0, Gen5Operand.Scalar(16)),
            BufferAccess(28, "BufferLoadDword", 8, offsetEnabled: true), EndProgram(36));
        var plan = Plan(program, BaseHash + 0x700 + (dynamicDescriptor ? 1u : 0u));
        var memory = new TestWordMemory { Base = TableBase, Words = [4] };
        Assert.True(RuntimeValueEvaluator.FlattenResourceTable(plan, new ResourceRuntimeInputs
        { UserData = [(uint)TableBase, 0, 0, 0, 64, 0, 0, 0], ReadMemory = memory.Read }, out var table));
        Assert.Contains(4u, table);
        if (dynamicDescriptor) Assert.Empty(plan.DeviceDataSlots);
        else Assert.Single(plan.DeviceDataSlots);
    }

    [Fact]
    public void ARepeatedPlanFailureCountsTheHostReadAndKeepsCumulativeAttempts()
    {
        var hash = BaseHash + 0x700;
        var plan = Plan(AddressBoundBufferLoad(), hash);
        var read = plan.TableReads[0].Value;
        var request = new DeviceBindingLoads.Request(plan.DeviceBindingKey, read.MemoryIndex, read.Kind,
            ShaderHash: hash, Pc: plan.Memory[read.MemoryIndex].Pc);
        var memory = new PendingMemory { Memory = new TestWordMemory { Base = TableBase, Words = [0x2000, 0, 256, 0] } };
        memory.Pending.Add(TableBase);
        try
        {
            DeviceBindingLoads.Refuse(plan.DeviceBindingKey, read.MemoryIndex);
            DeviceBindingLoads.CountPlanFailure();
            BindingWaitProvenance.Decision(request, "plan_failed:SamplerHandle_dword_0");
            var attempted = DeviceBindingLoads.TakeReport();
            Assert.Contains("words_replans_total=[", attempted);
            var (materialized, deferred, snapshot) = Attempt(plan, memory.Inputs([(uint)TableBase, 0]), defer: false);
            Assert.True(materialized);
            Assert.Null(deferred);
            Assert.Equal(new uint[] { 0x2000, 0, 256, 0 }, Assert.Single(snapshot.Buffers));
            var nextInterval = DeviceBindingLoads.TakeReport();
            Assert.Contains("words_replans=[ok=0 refused=0 plan_failed=0", nextInterval);
            Assert.Contains("words_refused_reads=[plan_failed=1]", nextInterval);
            Assert.Contains("plan_failed:SamplerHandle_dword_0", BindingWaitProvenance.TakeReport());
        }
        finally { DeviceBindingLoads.Forget(plan.DeviceBindingKey); }
    }

}
