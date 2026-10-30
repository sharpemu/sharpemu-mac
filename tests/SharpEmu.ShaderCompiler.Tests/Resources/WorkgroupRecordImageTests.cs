// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;
using System.Linq;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

// Ghost of Tsushima's tiled compute jobs pick a 376-byte record per workgroup (the workgroup id over 16) through a V#
// and write the texture its record names: the record offset is built from a workgroup id the host cannot know, so the
// texture they write, and the one and the sampler they read, used to be null and the job wrote nowhere.
public sealed class WorkgroupRecordImageTests
{
    private const uint Table = 0x3000;
    private const uint Stride = 0x178;
    private const uint OutputOffset = 0x100;
    private const uint SamplerOffset = 0x120;
    private const uint Records = 3;

    // s[4:7] is the record table's V#; with 8 user registers the workgroup id is s8.
    private static Gen5ShaderProgram TiledJob()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        void Add(Gen5ShaderInstruction instruction)
        {
            instructions.Add(instruction);
            pc += (uint)instruction.Words.Count * sizeof(uint);
        }

        Add(Sop2(pc, "SLshrB32", 10, Gen5Operand.Scalar(8), Operand(4)));
        Add(Sop2(pc, "SMulI32", 11, Gen5Operand.Scalar(10), Operand(Stride)));
        Add(Sop2(pc, "SAddI32", 12, Gen5Operand.Scalar(11), Operand(OutputOffset)));
        Add(ScalarBufferLoad(pc, 4, destination: 16, count: 8, dynamicOffsetRegister: 11));
        Add(ScalarBufferLoad(pc, 4, destination: 24, count: 8, dynamicOffsetRegister: 12));
        Add(ScalarBufferLoad(pc, 4, destination: 32, count: 4, immediateOffset: (int)SamplerOffset, dynamicOffsetRegister: 11));
        Add(Image(pc, "ImageSampleLz", 16, 32, dimension: 1, dmask: 1));
        Add(Image(pc, "ImageStore", 24, dimension: 1, dmask: 0xF));
        Add(EndProgram(pc));
        return Program([.. instructions]);
    }

    private static uint[] RecordImage(uint record, uint salt)
    {
        var descriptor = ResourceTrackerTests.ImageDescriptor();
        descriptor[0] += 0x100 * (record + 1) + salt;
        return descriptor;
    }

    private static ResourceSnapshot Materialize(ShaderResourcePlan plan)
    {
        var memory = new TestWordMemory { Base = Table, Words = new uint[(int)(Stride * Records / 4)] };
        for (uint record = 0; record < Records; record++)
        {
            ResourceTrackerTests.WriteImage(memory, Table + record * Stride, RecordImage(record, 0));
            ResourceTrackerTests.WriteImage(memory, Table + record * Stride + OutputOffset, RecordImage(record, 0x10));
            memory.At(Table + record * Stride + SamplerOffset) = 0x11 + record;
        }

        bool Read(ulong address, out uint word) => memory.Read(address, out word);
        var userData = new uint[8];
        userData[4] = Table;
        userData[6] = Stride * Records;
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs(userData, readMemory: Read, readCleanMemory: Read),
            ref snapshot, ref specialization));
        return snapshot;
    }

    [Fact]
    public void TheWrittenTexture_SelectsAmongTheTablesRecords()
    {
        var plan = Extract(TiledJob(), userDataCount: 8);

        var written = plan.Info.Images.Single(image => plan.DescriptorSources[(int)image.Source].IndirectImage is { Bindless: false });
        var selector = plan.DescriptorSources[(int)written.Source].IndirectImage!;
        Assert.True(selector.DirectRecordsMayEnd);
        Assert.Equal(ResourceTracker.MaxRecordDirectCandidates, selector.DirectCandidates!.Count);
        Assert.Equal([OutputOffset, Stride + OutputOffset], selector.DirectCandidates.Take(2).Select(candidate => candidate.Offset).ToArray());

        var snapshot = Materialize(plan);
        foreach (var record in Enumerable.Range(0, (int)Records))
        {
            Assert.Contains(snapshot.Images, image => image.SequenceEqual(RecordImage((uint)record, 0x10)));
        }
    }

    // The sampled texture is a bindless table of the records; their sampler reads as the first record's.
    [Fact]
    public void TheReadTextureIsABindlessTableAndItsSamplerTheFirstRecords()
    {
        var plan = Extract(TiledJob(), userDataCount: 8);

        Assert.Contains(plan.Info.Images, image =>
            plan.DescriptorSources[(int)image.Source].IndirectImage is { Bindless: true, RecordStride: Stride });
        var snapshot = Materialize(plan);
        Assert.Equal(0x11u, Assert.Single(snapshot.Samplers)[0]);
    }
}
