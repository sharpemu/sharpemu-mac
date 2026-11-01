// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;
using System.Linq;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

// Ghost of Tsushima's cloud noise volumes are built by compute programs that sum octaves of noise: a C for loop
// over an int count (s_cmp_lt_i32 then s_cbranch_scc0 out of the loop) samples the volume whose T# is the counter's
// 32-byte record of a table. Only unsigned bounds were proved, so every octave read as a null image, the volumes
// stayed zero and the menu's storm clouds had no density.
public sealed class SignedLoopImageTests
{
    private const uint Table = 0x1000;
    private const uint Stride = 32;
    private const uint CountOffset = 0x100;
    private const uint SamplerOffset = 0xE0;
    private const uint ImageType3D = 10;

    // s[4:5] is the table; s9 the counter, s10 the count.
    private static Gen5ShaderProgram OctaveLoop()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        void Add(Gen5ShaderInstruction instruction)
        {
            instructions.Add(instruction);
            pc += (uint)instruction.Words.Count * sizeof(uint);
        }

        Add(ScalarLoad(pc, 4, destination: 10, immediateOffset: (int)CountOffset));
        Add(MoveScalar(pc, 9, 0));
        var headerPc = pc;
        Add(Sopc(pc, "SCmpLtI32", Gen5Operand.Scalar(9), Gen5Operand.Scalar(10)));
        var exit = instructions.Count;
        Add(Branch(pc, "SCbranchScc0", 0));
        Add(Sop2(pc, "SLshlB32", 12, Gen5Operand.Scalar(9), Operand(5)));
        Add(ScalarLoad(pc, 4, destination: 16, count: 4, immediateOffset: (int)SamplerOffset));
        Add(ScalarLoad(pc, 4, destination: 20, count: 8, dynamicOffsetRegister: 12));
        Add(Image(pc, "ImageSampleLz", 20, 16, dimension: 2, dmask: 1));
        Add(Sop2(pc, "SAddI32", 9, Gen5Operand.Scalar(9), Operand(1)));
        Add(Branch(pc, "SBranch", (short)(((int)headerPc - (int)(pc + 4)) / 4)));
        var exitPc = pc;
        var branch = instructions[exit];
        instructions[exit] = Branch(branch.Pc, branch.Opcode, (short)(((int)exitPc - (int)(branch.Pc + 4)) / 4));
        Add(EndProgram(pc));
        return Program([.. instructions]);
    }

    private static uint[] Octave(uint record, uint type = ImageType3D)
    {
        var descriptor = ResourceTrackerTests.ImageDescriptor();
        descriptor[0] += 0x100 * (record + 1);
        descriptor[3] = (descriptor[3] & 0x0FFFFFFFu) | (type << 28);
        return descriptor;
    }

    // A table of eight records whose count reads `count`; reads of records at or past `readable` fail.
    private static (uint[] UserData, GuestWordReader Read) Octaves(uint count, uint readable, uint secondType = ImageType3D)
    {
        var memory = new TestWordMemory { Base = Table, Words = new uint[0x200 / 4] };
        memory.At(Table + CountOffset) = count;
        memory.At(Table + SamplerOffset) = 0x11;
        for (uint record = 0; record < 7; record++)
        {
            ResourceTrackerTests.WriteImage(memory, Table + record * Stride, Octave(record, record == 1 ? secondType : ImageType3D));
        }

        bool Read(ulong address, out uint word)
        {
            word = 0;
            var record = (address - Table) / Stride;
            return address >= Table + 0xE0 || record < readable ? memory.Read(address, out word) : false;
        }

        return ([0, 0, 0, 0, Table, 0], Read);
    }

    private static ResourceSnapshot Materialize(ShaderResourcePlan plan, uint count, uint readable, uint secondType = ImageType3D)
    {
        var (userData, read) = Octaves(count, readable, secondType);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs(userData, readMemory: read, readCleanMemory: read),
            ref snapshot, ref specialization));
        return snapshot;
    }

    [Fact]
    public void OctaveImage_IsSelectedByASignedCountedLoop()
    {
        var plan = Extract(OctaveLoop());

        Assert.Single(plan.IndirectImages);
        var selector = plan.DescriptorSources[(int)Assert.Single(plan.Info.Images).Source].IndirectImage!;
        Assert.NotNull(selector.DirectLimitSource);
        Assert.True(selector.DirectLimitSigned);
        Assert.False(selector.DirectLimitInclusive);
        Assert.Equal(
            [(0u, 0u), (Stride, 1u), (2 * Stride, 2u)],
            selector.DirectCandidates!.Take(3).Select(candidate => (candidate.Offset, candidate.Counter!.Value)).ToArray());
    }

    [Fact]
    public void OctaveImage_ReadsTheOctavesBelowTheCount()
    {
        var plan = Extract(OctaveLoop());

        var snapshot = Materialize(plan, count: 3, readable: 3);

        foreach (var record in new uint[] { 0, 1, 2 })
        {
            Assert.Contains(snapshot.Images, image => image.SequenceEqual(Octave(record)));
        }
    }

    [Fact]
    public void OctaveImage_UnderANegativeCountReadsNoRecord()
    {
        var plan = Extract(OctaveLoop());

        // The loop never runs for a count below zero, which an unsigned comparison would take for a huge one.
        var snapshot = Materialize(plan, count: 0xFFFFFFFF, readable: 0);

        Assert.All(snapshot.Images, image => Assert.All(image, word => Assert.Equal(0u, word)));
    }

    [Fact]
    public void OctaveImage_AFlatRecordAmongVolumesIsIncompatible()
    {
        var plan = Extract(OctaveLoop());
        var (userData, read) = Octaves(3, 3, secondType: 9);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();

        // A 2D T# cannot back the volume the program samples: Metal has no 3D view of a flat texture.
        Assert.False(ResourceMaterializer.Materialize(plan, Inputs(userData, readMemory: read, readCleanMemory: read),
            ref snapshot, ref specialization, out var failure));
        Assert.Equal(ResourceMaterializationFailure.IncompatibleImageCandidates, failure);
    }

    [Fact]
    public void OctaveImage_Compiles()
    {
        var program = OctaveLoop();
        var plan = Extract(program);
        var (userData, read) = Octaves(3, 3);
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
