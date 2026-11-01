// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;
using System.Linq;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

// A buffer store through a V# a counted loop loads from a pointer table (Ghost of Tsushima's
// light binning) is recorded against the buffers the loop can select, read for each draw,
// instead of all of guest memory; anything the draw cannot bound stays unplanned.
public sealed class DeviceDescriptorTableTests
{
    private const ulong Table = 0x1000;
    private const int CountOffset = 136;

    // s3 = 0; header: s10 = count at [s0:s1 + 136]; while (s3 < s10) { s[4:7] = [s0:s1 + (s3 << 4) + 16]; store; s3++ }
    private static Gen5ShaderProgram BinningProgram(uint guardRegister = 3)
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        void Add(Gen5ShaderInstruction instruction)
        {
            instructions.Add(instruction);
            pc += (uint)instruction.Words.Count * sizeof(uint);
        }

        Add(MoveScalar(pc, 3, 0));
        var headerPc = pc;
        Add(ScalarLoad(pc, 0, destination: 10, immediateOffset: CountOffset));
        Add(Sopc(pc, "SCmpLtU32", Gen5Operand.Scalar(guardRegister), Gen5Operand.Scalar(10)));
        var exitBranch = instructions.Count;
        Add(Branch(pc, "SCbranchScc0", 0));
        Add(Sop2(pc, "SLshl4AddU32", 12, Gen5Operand.Scalar(3), Operand(16)));
        Add(ScalarLoad(pc, 0, destination: 4, count: 4, dynamicOffsetRegister: 12));
        Add(BufferStore(pc, 4));
        Add(Sop2(pc, "SAddU32", 3, Gen5Operand.Scalar(3), Operand(1)));
        Add(Branch(pc, "SBranch", (short)(((int)headerPc - (int)(pc + 4)) / 4)));
        var exitPc = pc;
        var branch = instructions[exitBranch];
        instructions[exitBranch] = Branch(branch.Pc, "SCbranchScc0", (short)(((int)exitPc - (int)(branch.Pc + 4)) / 4));
        Add(EndProgram(pc));
        return Program([.. instructions]);
    }

    private static TestWordMemory TableMemory(uint count)
    {
        var memory = new TestWordMemory { Base = Table, Words = new uint[64] };
        memory.At(Table + CountOffset) = count;
        void Descriptor(int entry, uint address, uint stride, uint records)
        {
            var at = Table + 16 + (ulong)entry * 16;
            memory.At(at) = address;
            memory.At(at + 4) = stride << 16;
            memory.At(at + 8) = records;
            memory.At(at + 12) = 0;
        }

        Descriptor(0, 0x10_0000, 4, 256);
        Descriptor(1, 0x20_0000, 0, 0x100);
        Descriptor(2, 0x30_0000, 8, 16);
        return memory;
    }

    private static DeviceAddressRange[] Written(DeviceAddressRange[] ranges) => ranges.Where(range => range.Written).ToArray();

    [Fact]
    public void LoopIndexedTable_IsPlanned()
    {
        var plan = Extract(BinningProgram());

        var table = Assert.Single(plan.DeviceDescriptorTables);
        Assert.False(plan.UnplannedDeviceDescriptorWrites);
        Assert.Equal(16u, table.Stride);
        Assert.Equal(16u, table.BaseOffset);
        Assert.Equal(0u, table.Initial);
        Assert.Equal(1u, table.Step);
        Assert.False(table.LimitInclusive);
        Assert.True(plan.Info.WritesDeviceAddresses);
    }

    [Fact]
    public void EveryDescriptorTheLoopSelects_IsAWrittenRange()
    {
        var plan = Extract(BinningProgram());
        var memory = TableMemory(3);

        var written = Written(DeviceAddressRangePlanner.Evaluate(plan, Inputs([(uint)Table, 0], memory.Read)));

        Assert.All(written, range => Assert.True(range.Planned));
        Assert.Equal(
            [(0x10_0000ul, 1024ul), (0x20_0000ul, 0x100ul), (0x30_0000ul, 128ul)],
            written.Select(range => (range.Base, range.Size)).ToArray());
        Assert.Equal(written.Length, written.Select(range => range.Handle).Distinct().Count());
    }

    [Fact]
    public void LoopThatRunsNoIteration_WritesNothing()
    {
        var plan = Extract(BinningProgram());

        Assert.Empty(Written(DeviceAddressRangePlanner.Evaluate(plan, Inputs([(uint)Table, 0], TableMemory(0).Read))));
    }

    [Fact]
    public void LoopPastTheCap_StaysUnplanned()
    {
        var plan = Extract(BinningProgram());
        var memory = TableMemory(DeviceAddressRangePlanner.MaxDescriptorTableEntries + 1);

        var written = Assert.Single(Written(DeviceAddressRangePlanner.Evaluate(plan, Inputs([(uint)Table, 0], memory.Read))));
        Assert.False(written.Planned);
    }

    [Fact]
    public void UnreadableDescriptor_StaysUnplanned()
    {
        var plan = Extract(BinningProgram());
        var memory = TableMemory(3);
        memory.FailAddress = Table + 16 + 2 * 16 + 8;

        var written = Assert.Single(Written(DeviceAddressRangePlanner.Evaluate(plan, Inputs([(uint)Table, 0], memory.Read))));
        Assert.False(written.Planned);
    }

    [Fact]
    public void SwizzledDescriptor_ReachesItsLastWholeGroup()
    {
        var plan = Extract(BinningProgram());
        var memory = TableMemory(3);
        memory.At(Table + 16 + 2 * 16 + 4) |= 0x8000_0000u;

        var written = Written(DeviceAddressRangePlanner.Evaluate(plan, Inputs([(uint)Table, 0], memory.Read)));

        Assert.All(written, range => Assert.True(range.Planned));
        Assert.Equal((0x30_0000ul, 64ul * 8), (written[^1].Base, written[^1].Size));
    }

    [Fact]
    public void LoopTheGuardDoesNotBound_StaysUnplanned()
    {
        var plan = Extract(BinningProgram(guardRegister: 7));

        Assert.Empty(plan.DeviceDescriptorTables);
        Assert.True(plan.UnplannedDeviceDescriptorWrites);
    }

    // vcc = lane 0; v1 = vcc ? 32 : 48; s12 = readfirstlane v1; s[4:7] = [s0:s1 + s12]; store.
    private static Gen5ShaderProgram SelectorProgram() => Program(
        Vopc(0, "VCmpEqU32", Operand(0), 0),
        MoveVector(4, 2, 48),
        Vop2(8, "VCndmaskB32", 1, Operand(32), Gen5Operand.Vector(2)),
        ReadFirstLane(12, 12, 1),
        ScalarLoad(16, 0, destination: 4, count: 4, dynamicOffsetRegister: 12),
        BufferStore(24, 4),
        EndProgram(32));

    [Fact]
    public void OffsetOfAFewComputedValues_ReadsEachDescriptor()
    {
        var plan = Extract(SelectorProgram());

        var table = Assert.Single(plan.DeviceDescriptorTables);
        Assert.Equal([32u, 48u], table.Offsets);
        var memory = TableMemory(0);
        var written = Written(DeviceAddressRangePlanner.Evaluate(plan, Inputs([(uint)Table, 0], memory.Read)));
        Assert.Equal(
            [(0x20_0000ul, 0x100ul), (0x30_0000ul, 128ul)],
            written.Select(range => (range.Base, range.Size)).ToArray());
    }

    private const ulong Records = 0x2000;
    private const uint RecordBytes = 504;
    private const int RecordField = 0x190;

    // Ghost of Tsushima's Hi-Z culler: s[8:11] = [s0:s1] (a buffer of 504-byte records); s12 = a lane's
    // record index (v_readlane); s13 = s12 * 504; s[4:7] = s_buffer_load_dwordx4 s[8:11], s13 offset:0x190; store.
    private static Gen5ShaderProgram RecordProgram() => Program(
        ScalarLoad(0, 0, destination: 8, count: 4),
        ReadLane(8, 12, 1, 0),
        Sop2(16, "SMulI32", 13, Gen5Operand.Scalar(12), Operand(RecordBytes)),
        ScalarBufferLoad(24, 8, destination: 4, count: 4, immediateOffset: RecordField, dynamicOffsetRegister: 13),
        BufferStore(32, 4),
        EndProgram(40));

    private static TestWordMemory RecordMemory(uint records)
    {
        var memory = new TestWordMemory { Base = Table, Words = new uint[0x800] };
        memory.At(Table) = (uint)Records;
        memory.At(Table + 8) = records * RecordBytes;
        void Descriptor(uint record, uint address, uint stride, uint count)
        {
            var at = Records + record * RecordBytes + RecordField;
            memory.At(at) = address;
            memory.At(at + 4) = stride << 16;
            memory.At(at + 8) = count;
        }

        Descriptor(0, 0x10_0000, 4, 256);
        Descriptor(1, 0x10_0400, 4, 256);
        Descriptor(2, 0x30_0000, 0, 0x80);
        return memory;
    }

    [Fact]
    public void RecordFieldAtALostIndex_ReadsEveryRecord()
    {
        var plan = Extract(RecordProgram());

        var table = Assert.Single(plan.DeviceDescriptorTables);
        Assert.False(plan.UnplannedDeviceDescriptorWrites);
        Assert.NotNull(table.RecordTable);
        Assert.Equal(RecordBytes, table.RecordStride);
        Assert.Equal(RecordField, table.FirstImmediate);
        var written = Written(DeviceAddressRangePlanner.Evaluate(plan, Inputs([(uint)Table, 0], RecordMemory(3).Read)));
        Assert.All(written, range => Assert.True(range.Planned));
        // The first two records' buffers touch, so they are one range.
        Assert.Equal(
            [(0x10_0000ul, 0x800ul), (0x30_0000ul, 0x80ul)],
            written.Select(range => (range.Base, range.Size)).ToArray());
    }

    [Fact]
    public void RecordTablePastTheCap_StaysUnplanned()
    {
        var plan = Extract(RecordProgram());
        var memory = RecordMemory(3);
        memory.At(Table + 8) = (DeviceAddressRangePlanner.MaxRecordTableEntries + 1) * RecordBytes;

        var written = Assert.Single(Written(DeviceAddressRangePlanner.Evaluate(plan, Inputs([(uint)Table, 0], memory.Read))));
        Assert.False(written.Planned);
    }

    // s10 = user data word 2 (a bin index the draw fixes); s12 = s10 * 32; s[4:7] = [s0:s1 + s12]; store.
    private static Gen5ShaderProgram DrawIndexedProgram() => Program(
        Sop2(0, "SMulI32", 12, Gen5Operand.Scalar(2), Operand(32)),
        ScalarLoad(4, 0, destination: 4, count: 4, dynamicOffsetRegister: 12),
        BufferStore(12, 4),
        EndProgram(20));

    [Fact]
    public void DescriptorTheDrawFixes_IsOneWrittenRange()
    {
        var plan = Extract(DrawIndexedProgram());
        if (plan.DeviceDescriptorTables.Count == 0)
        {
            // The tracker bound this V# as an ordinary buffer: nothing goes through the page table.
            Assert.False(plan.Info.WritesDeviceAddresses);
            return;
        }

        var table = Assert.Single(plan.DeviceDescriptorTables);
        Assert.True(table.Direct);
        var memory = TableMemory(0);
        var written = Assert.Single(Written(DeviceAddressRangePlanner.Evaluate(plan, Inputs([(uint)Table, 0, 1], memory.Read))));
        Assert.True(written.Planned);
        Assert.Equal((0x20_0000ul, 0x100ul), (written.Base, written.Size));
    }
}
