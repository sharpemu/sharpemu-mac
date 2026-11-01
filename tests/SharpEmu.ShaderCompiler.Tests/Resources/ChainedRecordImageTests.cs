// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;
using System.Linq;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

// Ghost of Tsushima's foliage culling picks a 504-byte batch record per wave and tests its instances against the
// depth pyramid of the view the record points at: the texture and its sampler sit in that view object, read through
// the pointer the record holds. The record is picked by a value the host cannot know, so the texture used to be
// null, every instance failed the occlusion test and the menu's far pampas field drew only in its first frame.
public sealed class ChainedRecordImageTests
{
    private const uint Table = 0x3000;
    private const uint Stride = 0x1F8;
    private const uint Objects = 0x4000;
    private const uint ObjectBytes = 0x400;
    private const uint DescriptorOffset = 0x30C;
    private const uint SamplerOffset = 0x334;
    private const uint Records = 3;

    // s[4:7] is the record table's V#; with 8 user registers the workgroup id is s8.
    private static Gen5ShaderProgram CullingJob()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        void Add(Gen5ShaderInstruction instruction)
        {
            instructions.Add(instruction);
            pc += (uint)instruction.Words.Count * sizeof(uint);
        }

        Add(Sop2(pc, "SMulI32", 11, Gen5Operand.Scalar(8), Operand(Stride)));
        Add(ScalarBufferLoad(pc, 4, destination: 16, count: 2, dynamicOffsetRegister: 11));
        Add(ScalarLoad(pc, 16, destination: 20, count: 8, immediateOffset: (int)DescriptorOffset));
        Add(ScalarLoad(pc, 16, destination: 28, count: 4, immediateOffset: (int)SamplerOffset));
        Add(Image(pc, "ImageSampleLz", 20, 28, dimension: 1, dmask: 1));
        Add(EndProgram(pc));
        return Program([.. instructions]);
    }

    private static uint[] ViewImage(uint view)
    {
        var descriptor = ResourceTrackerTests.ImageDescriptor();
        descriptor[0] += 0x100 * (view + 1);
        return descriptor;
    }

    // Records 0 and 1 point at view 0, record 2 at view 1: records share views the way the game's batches do.
    private static uint ViewOf(uint record) => record < 2 ? 0u : 1u;

    private static ResourceSnapshot Materialize(ShaderResourcePlan plan)
    {
        var memory = new TestWordMemory { Base = Table, Words = new uint[(int)((Objects + 2 * ObjectBytes - Table) / 4)] };
        for (uint record = 0; record < Records; record++)
        {
            memory.At(Table + record * Stride) = Objects + ViewOf(record) * ObjectBytes;
        }

        for (uint view = 0; view < 2; view++)
        {
            ResourceTrackerTests.WriteImage(memory, Objects + view * ObjectBytes + DescriptorOffset, ViewImage(view));
            memory.At(Objects + view * ObjectBytes + SamplerOffset) = 0x11 + view;
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
    public void TheTextureBehindARecordsPointer_SelectsAmongTheTablesRecords()
    {
        var plan = Extract(CullingJob(), userDataCount: 8);

        var image = Assert.Single(plan.Info.Images);
        var selector = plan.DescriptorSources[(int)image.Source].IndirectImage;
        Assert.NotNull(selector);
        Assert.True(selector!.DirectRecordsMayEnd);
        Assert.Equal(ResourceTracker.MaxRecordDirectCandidates, selector.DirectCandidates!.Count);
        Assert.Equal([0u, Stride, 2 * Stride], selector.DirectCandidates.Take(3).Select(candidate => candidate.Offset).ToArray());

        var snapshot = Materialize(plan);
        Assert.Contains(snapshot.Images, bound => bound.SequenceEqual(ViewImage(0)));
        Assert.Contains(snapshot.Images, bound => bound.SequenceEqual(ViewImage(1)));
    }

    // The sampler beside it reads as the first record's view's, through the pointer, rather than as null.
    [Fact]
    public void ItsSamplerReadsTheFirstRecordsView()
    {
        var snapshot = Materialize(Extract(CullingJob(), userDataCount: 8));

        Assert.Equal(0x11u, Assert.Single(snapshot.Samplers)[0]);
    }
}
