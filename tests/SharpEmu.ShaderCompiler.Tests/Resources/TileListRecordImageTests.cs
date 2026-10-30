// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;
using System.Linq;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

// Ghost of Tsushima's clustered decal pass (cs7B97365C) gives each 8x8 tile a run of a list of record indices: the
// tile's header holds where its run starts and how long it is, the loop reads one index per pass, and the record it
// names (536 bytes) holds the decal's texture at +0x80. The loop also carries the last texture on in its registers to
// a sample after the loop, and samples each texture again past an inner loop that leaves its registers alone. The index
// read sits behind the loop counter, a phi, so the texture used to read as null; the second sample takes the same read
// through the inner loop's phis, which only carry it.
public sealed class TileListRecordImageTests
{
    private const uint Records = 0x1000;
    private const uint List = 0x8000;
    private const uint Headers = 0x9000;
    private const uint Stride = 0x218;
    private const uint ImageOffset = 0x80;
    private const uint SamplerOffset = 0x100;

    // s[0:3] the record table's V#, s[4:7] the list's, s[8:11] the tile headers'; with 12 user registers the
    // workgroup id is s12. s16 counts, s17 is the end of the tile's run; s40 counts the inner loop.
    private static Gen5ShaderProgram DecalTile(bool innerLoop = false)
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        void Add(Gen5ShaderInstruction instruction)
        {
            instructions.Add(instruction);
            pc += (uint)instruction.Words.Count * sizeof(uint);
        }

        Add(Sop2(pc, "SLshlB32", 13, Gen5Operand.Scalar(12), Operand(3)));
        Add(ScalarBufferLoad(pc, 8, destination: 14, count: 2, dynamicOffsetRegister: 13));
        Add(MoveScalarRegister(pc, 16, 14));
        Add(Sop2(pc, "SAddU32", 17, Gen5Operand.Scalar(14), Gen5Operand.Scalar(15)));
        // A tile with no decals samples the first record's texture after the loop.
        Add(ScalarBufferLoad(pc, 0, destination: 24, count: 8, immediateOffset: (int)ImageOffset));
        Add(ScalarBufferLoad(pc, 0, destination: 32, count: 4, immediateOffset: (int)SamplerOffset));
        var headerPc = pc;
        Add(Sopc(pc, "SCmpLtU32", Gen5Operand.Scalar(16), Gen5Operand.Scalar(17)));
        var exit = instructions.Count;
        Add(Branch(pc, "SCbranchScc0", 0));
        Add(Sop2(pc, "SLshlB32", 18, Gen5Operand.Scalar(16), Operand(2)));
        Add(ScalarBufferLoad(pc, 4, destination: 19, dynamicOffsetRegister: 18));
        Add(Sop2(pc, "SMulI32", 20, Gen5Operand.Scalar(19), Operand(Stride)));
        Add(Sop2(pc, "SAddU32", 21, Gen5Operand.Scalar(20), Operand(ImageOffset)));
        Add(ScalarBufferLoad(pc, 0, destination: 24, count: 8, dynamicOffsetRegister: 21));
        Add(ScalarBufferLoad(pc, 0, destination: 32, count: 4, immediateOffset: (int)SamplerOffset, dynamicOffsetRegister: 20));
        Add(Image(pc, "ImageSampleLz", 24, 32, dimension: 1, dmask: 1));
        if (innerLoop)
        {
            Add(MoveScalar(pc, 40, 0));
            var innerPc = pc;
            Add(Sopc(pc, "SCmpLtU32", Gen5Operand.Scalar(40), Operand(4)));
            var innerExit = instructions.Count;
            Add(Branch(pc, "SCbranchScc0", 0));
            // The inner loop samples the record's texture too (cs7B97365C's parallax steps), so its header merges
            // the texture and sampler registers with themselves.
            Add(Image(pc, "ImageSampleLz", 24, 32, dimension: 1, dmask: 1));
            Add(Sop2(pc, "SAddU32", 40, Gen5Operand.Scalar(40), Operand(1)));
            Add(Branch(pc, "SBranch", (short)(((int)innerPc - (int)(pc + 4)) / 4)));
            var inner = instructions[innerExit];
            instructions[innerExit] = Branch(inner.Pc, inner.Opcode, (short)(((int)pc - (int)(inner.Pc + 4)) / 4));
            Add(Image(pc, "ImageSampleLz", 24, 32, dimension: 1, dmask: 1));
        }

        Add(Sop2(pc, "SAddU32", 16, Gen5Operand.Scalar(16), Operand(1)));
        Add(Branch(pc, "SBranch", (short)(((int)headerPc - (int)(pc + 4)) / 4)));
        var exitPc = pc;
        var branch = instructions[exit];
        instructions[exit] = Branch(branch.Pc, branch.Opcode, (short)(((int)exitPc - (int)(branch.Pc + 4)) / 4));
        Add(Image(pc, "ImageSampleLz", 24, 32, dimension: 1, dmask: 1));
        Add(EndProgram(pc));
        return Program([.. instructions]);
    }

    private static uint[] Decal(uint record)
    {
        var descriptor = ResourceTrackerTests.ImageDescriptor();
        descriptor[0] += 0x100 * (record + 1);
        return descriptor;
    }

    private static uint[] BufferDescriptor(uint address, uint bytes) => [address, 0, bytes, 0];

    private static ResourceSnapshot Materialize(ShaderResourcePlan plan)
    {
        var memory = new TestWordMemory { Base = Records, Words = new uint[(int)((Headers + 0x100 - Records) / 4)] };
        for (uint record = 0; record < 4; record++)
        {
            ResourceTrackerTests.WriteImage(memory, Records + record * Stride + ImageOffset, Decal(record));
            memory.At(Records + record * Stride + SamplerOffset) = 0x11 + record;
        }

        bool Read(ulong address, out uint word) => memory.Read(address, out word);
        var userData = new uint[12];
        BufferDescriptor(Records, 4 * Stride).CopyTo(userData, 0);
        BufferDescriptor(List, 0x100).CopyTo(userData, 4);
        BufferDescriptor(Headers, 0x100).CopyTo(userData, 8);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs(userData, readMemory: Read, readCleanMemory: Read),
            ref snapshot, ref specialization));
        return snapshot;
    }

    [Fact]
    public void TheTextureOfTheRecordAListEntryNames_SelectsAmongTheTablesRecords()
    {
        var plan = Extract(DecalTile(), userDataCount: 12);

        var selectors = plan.Info.Images
            .Select(image => plan.DescriptorSources[(int)image.Source].IndirectImage)
            .Where(selector => selector?.DirectCandidates is not null)
            .ToArray();
        var selector = Assert.Single(selectors)!;
        Assert.True(selector.DirectRecordsMayEnd);
        Assert.Equal([ImageOffset, Stride + ImageOffset, 2 * Stride + ImageOffset],
            selector.DirectCandidates!.Take(3).Select(candidate => candidate.Offset).ToArray());

        var snapshot = Materialize(plan);
        for (uint record = 0; record < 4; record++)
            Assert.Contains(snapshot.Images, bound => bound.SequenceEqual(Decal(record)));
    }

    [Fact]
    public void ASecondSamplePastAnInnerLoop_SelectsTheSameRecords()
    {
        var plan = Extract(DecalTile(innerLoop: true), userDataCount: 12);

        var selectors = plan.Info.Images
            .Select(image => plan.DescriptorSources[(int)image.Source].IndirectImage)
            .Where(selector => selector?.DirectCandidates is not null)
            .ToArray();
        Assert.Equal(2, selectors.Length);
        foreach (var selector in selectors)
        {
            Assert.Equal([ImageOffset, Stride + ImageOffset, 2 * Stride + ImageOffset],
                selector!.DirectCandidates!.Take(3).Select(candidate => candidate.Offset).ToArray());
        }

        var snapshot = Materialize(plan);
        for (uint record = 0; record < 4; record++)
            Assert.Contains(snapshot.Images, bound => bound.SequenceEqual(Decal(record)));
    }
}
