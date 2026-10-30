// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;
using System.Linq;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

// Ghost of Tsushima's material pixel programs handle the 8-bit material ids of a wave one at a time: the
// smallest id left (v_readlane of lanes 31 and 63 after a DPP min, then s_min_u32) leaves the loop at 255
// (s_cmpk_ge_u32 then s_cbranch_scc1), and otherwise selects a 340-byte material record whose textures are
// read at the id times the stride: through the V# in s[4:7] with s_buffer_load, or from an address with s_load.
// The host cannot know the id, so every texture of every material read as a null image.
public sealed class MaterialWaterfallImageTests
{
    private const uint Table = 0x2000;
    private const uint Stride = 0x154;
    private const uint ImageOffset = 64;
    private const uint Records = 4;

    // v2 holds each lane's material id and s[40:43] the sampler. With secondHalf the texture is the second
    // descriptor an s_buffer_load_dwordx16 reads; with addressed the table is the address in s[4:5].
    private static Gen5ShaderProgram MaterialLoop(bool guarded = true, bool secondHalf = false, bool addressed = false)
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        void Add(Gen5ShaderInstruction instruction)
        {
            instructions.Add(instruction);
            pc += (uint)instruction.Words.Count * sizeof(uint);
        }

        var headerPc = pc;
        Add(ReadLane(pc, 10, 2, 31));
        Add(ReadLane(pc, 11, 2, 63));
        Add(Sop2(pc, "SMinU32", 12, Gen5Operand.Scalar(10), Gen5Operand.Scalar(11)));
        var exit = -1;
        if (guarded)
        {
            Add(new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Sopk, "SCmpkGeU32", [0u],
                [new Gen5Operand(Gen5OperandKind.LiteralConstant, 255)], [Gen5Operand.Scalar(12)], null));
            exit = instructions.Count;
            Add(Branch(pc, "SCbranchScc1", 0));
        }

        Add(Sop2(pc, "SMulI32", 13, Gen5Operand.Scalar(12), Operand(Stride)));
        if (addressed)
        {
            Add(ScalarLoad(pc, 4, destination: 16, count: 8, immediateOffset: (int)ImageOffset, dynamicOffsetRegister: 13));
            Add(Image(pc, "ImageSampleLz", 16, 40, dimension: 1, dmask: 1));
        }
        else if (secondHalf)
        {
            Add(ScalarBufferLoad(pc, 4, destination: 16, count: 16, immediateOffset: (int)ImageOffset - 32, dynamicOffsetRegister: 13));
            Add(Image(pc, "ImageSampleLz", 24, 40, dimension: 1, dmask: 1));
        }
        else
        {
            Add(ScalarBufferLoad(pc, 4, destination: 16, count: 8, immediateOffset: (int)ImageOffset, dynamicOffsetRegister: 13));
            Add(Image(pc, "ImageSampleLz", 16, 40, dimension: 1, dmask: 1));
        }

        Add(MoveVector(pc, 2, 255));
        Add(Branch(pc, "SBranch", (short)(((int)headerPc - (int)(pc + 4)) / 4)));
        if (exit >= 0)
        {
            var branch = instructions[exit];
            instructions[exit] = Branch(branch.Pc, branch.Opcode, (short)(((int)pc - (int)(branch.Pc + 4)) / 4));
        }

        Add(EndProgram(pc));
        return Program([.. instructions]);
    }

    private static uint[] Material(uint record)
    {
        var descriptor = ResourceTrackerTests.ImageDescriptor();
        descriptor[0] += 0x100 * (record + 1);
        return descriptor;
    }

    // A table of Records materials, its V# sized to them; reads past them fail.
    private static (ResourceSnapshot Snapshot, ResourceSpecialization Specialization) Materialize(ShaderResourcePlan plan)
    {
        var memory = new TestWordMemory { Base = Table, Words = new uint[(int)(Stride * Records / 4)] };
        for (uint record = 0; record < Records; record++)
        {
            ResourceTrackerTests.WriteImage(memory, Table + record * Stride + ImageOffset, Material(record));
        }

        bool Read(ulong address, out uint word) => memory.Read(address, out word);
        var userData = new uint[44];
        userData[4] = Table;
        userData[6] = Stride * Records;
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs(userData, readMemory: Read, readCleanMemory: Read),
            ref snapshot, ref specialization));
        return (snapshot, specialization);
    }

    // Through the V#: a table of records the device reads through a handle table, the key over the record size.
    [Theory]
    [InlineData(true, false)]
    [InlineData(true, true)]
    [InlineData(false, false)]
    public void MaterialTable_IsABindlessTableOfRecords(bool guarded, bool secondHalf)
    {
        var plan = Extract(MaterialLoop(guarded, secondHalf), stage: ShaderStage.Pixel);

        var selector = plan.DescriptorSources[(int)Assert.Single(plan.Info.Images).Source].IndirectImage!;
        Assert.True(selector.Dense && selector.Bindless);
        Assert.Equal(Stride, selector.RecordStride);
        Assert.Equal(ImageOffset, selector.TableOffset);
        Assert.Equal(0u, selector.DynamicOffsetBase);
        Assert.True(Assert.Single(plan.IndirectImages).KeyIsAddressOffset);

        var (snapshot, _) = Materialize(plan);
        var table = Assert.Single(snapshot.BindlessImages);
        Assert.Equal((ulong)(Table + ImageOffset), table.Address);
        Assert.Equal([0, 1, 2, 3], table.Entries);
        Assert.Equal(Enumerable.Range(0, (int)Records).Select(record => Material((uint)record)), table.Descriptors);
    }

    [Fact]
    public void MaterialTable_TheShaderDividesTheKeyByTheRecordSize()
    {
        var program = MaterialLoop();
        var plan = Extract(program, stage: ShaderStage.Pixel);
        var (snapshot, specialization) = Materialize(plan);
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(program, 0, 44),
            false, ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false);
        var request = new ShaderCompileRequest(plan, resources, layout);

        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var metal, out var error), error);
        Assert.Contains("bindless_images[0u]", metal!.Source);
        Assert.Contains("== 32u ?", metal.Source);
        // The mapping holds the entry count, the first key and the record size.
        Assert.Contains(Stride, snapshot.FlattenedResourceTable);
    }

    // From an address: the ids below the guard are candidates, and those past the readable table read as null.
    [Fact]
    public void AddressedMaterialImage_SelectsAmongEveryIdBelowTheGuard()
    {
        var plan = Extract(MaterialLoop(addressed: true), stage: ShaderStage.Pixel);

        var selector = plan.DescriptorSources[(int)Assert.Single(plan.Info.Images).Source].IndirectImage!;
        Assert.True(selector.DirectRecordsMayEnd);
        Assert.Equal(255, selector.DirectCandidates!.Count);
        Assert.Equal([0u, Stride, 2 * Stride], selector.DirectCandidates.Take(3).Select(candidate => candidate.Offset).ToArray());

        var (snapshot, _) = Materialize(plan);
        foreach (var record in Enumerable.Range(0, (int)Records))
        {
            Assert.Contains(snapshot.Images, image => image.SequenceEqual(Material((uint)record)));
        }
    }

    [Fact]
    public void AddressedMaterialImage_WithoutTheGuardStaysUnselected()
    {
        var plan = Extract(MaterialLoop(guarded: false, addressed: true), stage: ShaderStage.Pixel);

        Assert.Empty(plan.IndirectImages);
    }
}
