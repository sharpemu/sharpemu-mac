// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;
using System.Linq;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

// Ghost of Tsushima's terrain vertex program walks a table of height patches. The table's first dword
// is the patch count; each 376-byte record holds the patch's T# at +4 and its S# at +260, read with
// s_load at the counter times the stride. The loop tests exec first, then leaves when the counter
// reaches the count through s_cselect_b64 vcc, exec, 0 and s_cbranch_vccz. Each patch's heightmap used
// to read as a null image, so the terrain lost its height.
public sealed class RecordLoopImageTests
{
    private const uint Table = 0x1000;
    private const uint Stride = 0x178;
    private const uint ImageOffset = 4;
    private const uint SamplerOffset = 260;

    // s[4:5] is the table; s9 the counter, s10 the count.
    private static Gen5ShaderProgram PatchLoop()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        void Add(Gen5ShaderInstruction instruction)
        {
            instructions.Add(instruction);
            pc += (uint)instruction.Words.Count * sizeof(uint);
        }

        Add(ScalarLoad(pc, 4, destination: 10));
        Add(MoveScalar(pc, 9, 0));
        var headerPc = pc;
        Add(Vopc(pc, "VCmpxGeU32", Operand(0), 20));
        var execExit = instructions.Count;
        Add(Branch(pc, "SCbranchExecz", 0));
        Add(Sopc(pc, "SCmpLtU32", Gen5Operand.Scalar(9), Gen5Operand.Scalar(10)));
        Add(Sop2(pc, "SCselectB64", 0, Gen5Operand.Scalar(126), Operand(0)));
        Add(Sop1(pc, "SMovB64", 106, Gen5Operand.Scalar(0)));
        Add(Sop1(pc, "SMovB64", 126, Gen5Operand.Scalar(0)));
        var countExit = instructions.Count;
        Add(Branch(pc, "SCbranchVccz", 0));
        Add(Sop2(pc, "SMulI32", 12, Gen5Operand.Scalar(9), Operand(Stride)));
        Add(Sop2(pc, "SAddU32", 13, Gen5Operand.Scalar(12), Operand(ImageOffset)));
        Add(ScalarLoad(pc, 4, destination: 16, count: 8, dynamicOffsetRegister: 13));
        Add(ScalarLoad(pc, 4, destination: 24, count: 4, immediateOffset: (int)SamplerOffset, dynamicOffsetRegister: 12));
        Add(Image(pc, "ImageSampleLz", 16, 24, dimension: 1, dmask: 1));
        Add(Sop2(pc, "SAddU32", 9, Gen5Operand.Scalar(9), Operand(1)));
        Add(Branch(pc, "SBranch", (short)(((int)headerPc - (int)(pc + 4)) / 4)));
        var exitPc = pc;
        foreach (var index in new[] { execExit, countExit })
        {
            var branch = instructions[index];
            instructions[index] = Branch(branch.Pc, branch.Opcode, (short)(((int)exitPc - (int)(branch.Pc + 4)) / 4));
        }

        Add(EndProgram(pc));
        return Program([.. instructions]);
    }

    private static uint[] Patch(uint record)
    {
        var descriptor = ResourceTrackerTests.ImageDescriptor();
        descriptor[0] += 0x100 * (record + 1);
        return descriptor;
    }

    // A table of `count` patches; reads of records at or past `readable` fail.
    private static (uint[] UserData, GuestWordReader Read, List<ulong> Reads) Patches(uint count, uint readable)
    {
        var memory = new TestWordMemory { Base = Table, Words = new uint[(int)(Stride * 8 / 4)] };
        memory.At(Table) = count;
        for (uint record = 0; record < 8; record++)
        {
            ResourceTrackerTests.WriteImage(memory, Table + record * Stride + ImageOffset, Patch(record));
            memory.At(Table + record * Stride + SamplerOffset) = 0x11 + record;
        }

        var reads = new List<ulong>();
        bool Read(ulong address, out uint word)
        {
            reads.Add(address);
            word = 0;
            return address < Table + 4 || address < Table + readable * Stride ? memory.Read(address, out word) : false;
        }

        return ([0, 0, 0, 0, Table, 0], Read, reads);
    }

    private static ResourceSnapshot Materialize(ShaderResourcePlan plan, uint count, uint readable)
    {
        var (userData, read, _) = Patches(count, readable);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs(userData, readMemory: read, readCleanMemory: read),
            ref snapshot, ref specialization));
        return snapshot;
    }

    [Fact]
    public void PatchImage_SelectsAmongTheRecordsTheLoopReaches()
    {
        var plan = Extract(PatchLoop(), stage: ShaderStage.Vertex);

        var access = Assert.Single(plan.IndirectImages);
        Assert.True(access.KeyIsAddressOffset);
        var selector = plan.DescriptorSources[(int)Assert.Single(plan.Info.Images).Source].IndirectImage!;
        Assert.NotNull(selector.DirectLimitSource);
        Assert.False(selector.DirectLimitInclusive);
        Assert.Equal(ResourceTracker.MaxLoopDirectCandidates, selector.DirectCandidates!.Count);
        Assert.Equal(
            [(ImageOffset, 0u), (Stride + ImageOffset, 1u), (2 * Stride + ImageOffset, 2u)],
            selector.DirectCandidates.Take(3).Select(candidate => (candidate.Offset, candidate.Counter!.Value)).ToArray());
    }

    [Fact]
    public void PatchImage_ReadsOnlyTheRecordsBelowTheCount()
    {
        var plan = Extract(PatchLoop(), stage: ShaderStage.Vertex);

        // Every record past the third fails to read: the count keeps the host from reaching them.
        var snapshot = Materialize(plan, count: 3, readable: 3);

        Assert.Equal(4, snapshot.Images.Length);
        foreach (var record in new uint[] { 0, 1, 2 })
        {
            Assert.Contains(snapshot.Images, image => image.SequenceEqual(Patch(record)));
        }

        Assert.Contains(snapshot.Images, image => image.All(word => word == 0));
    }

    [Fact]
    public void PatchSampler_ReadsTheFirstRecords()
    {
        var plan = Extract(PatchLoop(), stage: ShaderStage.Vertex);

        var snapshot = Materialize(plan, count: 3, readable: 3);

        Assert.Equal(0x11u, Assert.Single(snapshot.Samplers)[0]);
    }

    [Fact]
    public void PatchImage_WithoutPatchesIsNull()
    {
        var plan = Extract(PatchLoop(), stage: ShaderStage.Vertex);

        var snapshot = Materialize(plan, count: 0, readable: 1);

        Assert.All(Assert.Single(snapshot.Images), word => Assert.Equal(0u, word));
    }

    [Fact]
    public void PatchImage_Compiles()
    {
        var program = PatchLoop();
        var plan = Extract(program);
        var (userData, read, _) = Patches(3, 3);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs(userData, readMemory: read, readCleanMemory: read),
            ref snapshot, ref specialization));
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(program, 0, 64),
            false, ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false);
        var request = new ShaderCompileRequest(plan, resources, layout) { LocalSizeX = 1, ThreadCountX = 1 };

        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out _, out var error), error);
    }
}
