// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// Synthetic NGG primitive shaders, built as IR so the fixtures stay readable and no guest
/// shader bytes enter the repository. Each one is a merged ES/GS program: it exchanges a
/// value through LDS across the subgroup's two waves, synchronizes with S_BARRIER, asks for
/// export space with S_SENDMSG GS_ALLOC_REQ and then exports POS0 plus the packed primitive.
/// </summary>
internal static class Gen5NggFixtures
{
    // The merged stage's user data starts at s8 (ShaderPipelineCache.VertexUserDataBase), so
    // s0..s7 are the system scalars the geometry engine loads.
    public const uint VertexUserDataBase = 8;

    public const uint VertexCount = 64;
    public const uint PrimitiveCount = 1;

    // Two waves of 32, so the LDS exchange below genuinely crosses a wave boundary.
    public static Gen5NggStageInfo StageInfo { get; } = new(
        VerticesPerSubgroup: VertexCount,
        PrimitivesPerSubgroup: PrimitiveCount,
        MaxOutputPerSubgroup: VertexCount,
        OutputPrimitiveType: Gen5NggExports.OutputPrimitiveTriangleStrip,
        PrimitiveAmplificationFactor: 1,
        LocalDataShareBytes: 1024,
        VertexGroupSize: VertexCount,
        PrimitiveGroupSize: PrimitiveCount);

    /// <summary>The packed connectivity a triangle over subgroup vertices 0/1/2 exports:
    /// one 9-bit index per 10-bit field.</summary>
    public const uint TriangleConnectivity =
        (0u << (0 * Gen5NggExports.PrimitiveIndexStride)) |
        (1u << (1 * Gen5NggExports.PrimitiveIndexStride)) |
        (2u << (2 * Gen5NggExports.PrimitiveIndexStride));

    /// <summary>M0 for GS_ALLOC_REQ: the vertex count in bits 0-9, the primitive count in
    /// bits 12-22.</summary>
    public const uint AllocationRequest =
        VertexCount | (PrimitiveCount << Gen5NggExports.AllocationPrimitiveCountShift);

    private const uint SendMessageWord = 0xBF90_0000u | Gen5NggExports.GsAllocRequestMessage;
    private const uint EndProgramWord = 0xBF81_0000u;

    /// <summary>A culling primitive shader: LDS round trip across both waves, a barrier, the
    /// allocation request, a position export built from the exchanged value and the packed
    /// primitive export. Rasterizes one screen-space triangle through the center.</summary>
    public static Gen5ShaderProgram CullingTriangle(bool nullPrimitive = false, bool bufferFetchPosition = false)
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;

        void Add(string opcode, Gen5ShaderEncoding encoding, Gen5Operand[] sources,
            Gen5Operand[] destinations, Gen5InstructionControl? control = null, uint word = 0)
        {
            instructions.Add(new Gen5ShaderInstruction(
                pc, encoding, opcode, [word], sources, destinations, control));
            pc += sizeof(uint);
        }

        // value = vertex index & 31 -> each wave writes its own lane number.
        Add("VAndB32", Gen5ShaderEncoding.Vop2,
            [Literal(31), Vector(Gen5NggExports.EsVertexIndexRegister)], [Vector(10)]);
        Add("VLshlrevB32", Gen5ShaderEncoding.Vop2,
            [Inline(2), Vector(Gen5NggExports.EsVertexIndexRegister)], [Vector(11)]);
        Add("DsWriteB32", Gen5ShaderEncoding.Ds,
            [Vector(11), Vector(10)], [], new Gen5DataShareControl(0, 0, false));
        Add("SBarrier", Gen5ShaderEncoding.Sopp, [], []);

        // Read the slot the *other* wave wrote: (index ^ 32) & 31 == index & 31, so the
        // value only survives if threadgroup LDS and the barrier both work.
        Add("VXorB32", Gen5ShaderEncoding.Vop2,
            [Literal(32), Vector(Gen5NggExports.EsVertexIndexRegister)], [Vector(12)]);
        Add("VLshlrevB32", Gen5ShaderEncoding.Vop2, [Inline(2), Vector(12)], [Vector(13)]);
        Add("DsReadB32", Gen5ShaderEncoding.Ds,
            [Vector(13)], [Vector(14)], new Gen5DataShareControl(0, 0, false));

        Add("VCvtF32U32", Gen5ShaderEncoding.Vop1, [Vector(14)], [Vector(15)]);
        if (bufferFetchPosition)
        {
            // buffer_load_format_xyzw v[24:27], v14, s[8:11], 0 idxen - the ES vertex fetch.
            Add("BufferLoadFormatXyzw", Gen5ShaderEncoding.Mubuf,
                [Vector(14), Scalar(8), Inline(0)],
                [Vector(24), Vector(25), Vector(26), Vector(27)],
                new Gen5BufferMemoryControl(
                    DwordCount: 4,
                    VectorAddress: 14,
                    VectorData: 24,
                    ScalarResource: 8,
                    OffsetBytes: 0,
                    IndexEnabled: true,
                    OffsetEnabled: false,
                    Glc: false,
                    Slc: false));
        }

        // x = 0.8 * value - 0.8 over values 0/1/2.
        Add("VMulF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(0.8f)), Vector(15)], [Vector(16)]);
        Add("VAddF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(-0.8f)), Vector(16)], [Vector(16)]);

        // y = 1.6 * (value & 1) - 0.8, which puts vertex 1 at the top of the triangle.
        Add("VAndB32", Gen5ShaderEncoding.Vop2, [Inline(1), Vector(14)], [Vector(17)]);
        Add("VCvtF32U32", Gen5ShaderEncoding.Vop1, [Vector(17)], [Vector(18)]);
        Add("VMulF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(1.6f)), Vector(18)], [Vector(19)]);
        Add("VAddF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(-0.8f)), Vector(19)], [Vector(19)]);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Inline(0)], [Vector(20)]);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Literal(Bits(1.0f))], [Vector(21)]);

        // The allocation request has to precede the exports, as it does in hardware.
        Add("SMovB32", Gen5ShaderEncoding.Sop1, [Literal(AllocationRequest)], [Scalar(124)]);
        Add("SSendmsg", Gen5ShaderEncoding.Sopp, [], [], word: SendMessageWord);
        Add("Exp", Gen5ShaderEncoding.Exp,
            bufferFetchPosition
                ? [Vector(24), Vector(25), Vector(26), Vector(27)]
                : [Vector(16), Vector(19), Vector(20), Vector(21)],
            [],
            new Gen5ExportControl(Gen5NggExports.FirstPositionTarget, 0xF, false, false, false));

        // One parameter export so the mesh vertex payload carries a user location the
        // paired pixel program can read.
        Add("Exp", Gen5ShaderEncoding.Exp,
            [Vector(15), Vector(20), Vector(20), Vector(21)], [],
            new Gen5ExportControl(Gen5NggExports.FirstParameterTarget, 0xF, false, false, false));

        var connectivity = nullPrimitive
            ? TriangleConnectivity | (1u << Gen5NggExports.PrimitiveNullBit)
            : TriangleConnectivity;
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Literal(connectivity)], [Vector(22)]);
        Add("Exp", Gen5ShaderEncoding.Exp,
            [Vector(22), Vector(22), Vector(22), Vector(22)], [],
            new Gen5ExportControl(Gen5NggExports.PrimitiveTarget, 0x1, false, true, false));
        Add("SEndpgm", Gen5ShaderEncoding.Sopp, [], [], word: EndProgramWord);
        return new Gen5ShaderProgram(0, instructions);
    }

    private const uint BranchWord = 0xBF82_0000u;
    private const uint BranchScc0Word = 0xBF84_0000u;
    private const uint BranchScc1Word = 0xBF85_0000u;

    private static uint Branch(uint opcodeWord, uint fromPc, uint toPc) =>
        opcodeWord | (ushort)(short)(((int)toPc - (int)(fromPc + 4)) / 4);

    // The two waves take opposite exits after their LDS exchange. One keeps the
    // original parameter; the other overwrites it. Both must reach the mesh epilogue.
    public static Gen5ShaderProgram ConditionalMeshExit(bool firstWaveExitsEarly)
    {
        var instructions = CullingTriangle().Instructions.SkipLast(1).ToList();
        var pc = instructions[^1].Pc + 4;
        instructions.Add(new(pc, Gen5ShaderEncoding.Sop2, "SAndB32", [0u],
            [Scalar(Gen5NggExports.MergedWaveInfoRegister), Literal(1u << 24)], [Scalar(30)], null));
        pc += 4;
        instructions.Add(new(pc, Gen5ShaderEncoding.Sopc, "SCmpEqU32", [0u],
            [Scalar(30), Literal(firstWaveExitsEarly ? 0u : 1u << 24)], [], null));
        pc += 4;
        instructions.Add(new(pc, Gen5ShaderEncoding.Sopp, "SCbranchScc1",
            [Branch(BranchScc1Word, pc, pc + 16)], [], [], null));
        pc += 4;
        instructions.Add(new(pc, Gen5ShaderEncoding.Vop1, "VMovB32", [0u],
            [Literal(Bits(0.25f))], [Vector(15)], null));
        pc += 4;
        instructions.Add(new(pc, Gen5ShaderEncoding.Exp, "Exp", [0u],
            [Vector(15), Vector(15), Vector(15), Vector(15)], [],
            new Gen5ExportControl(Gen5NggExports.FirstParameterTarget, 0xF, false, false, false)));
        pc += 4;
        instructions.Add(new(pc, Gen5ShaderEncoding.Sopp, "SEndpgm", [EndProgramWord], [], [], null));
        pc += 4;
        instructions.Add(new(pc, Gen5ShaderEncoding.Sopp, "SEndpgm", [EndProgramWord], [], [], null));
        return new Gen5ShaderProgram(0, instructions);
    }

    public static Gen5ShaderProgram MaskedMeshAlu()
    {
        var instructions = MeshExitLoop(3).Instructions.SkipLast(1).ToList();
        var pc = instructions[^1].Pc + 4;
        void Add(string opcode, Gen5ShaderEncoding encoding, Gen5Operand[] sources,
            Gen5Operand[] destinations, Gen5InstructionControl? control = null)
        {
            instructions.Add(new(pc, encoding, opcode, [0u], sources, destinations, control));
            pc += 4;
        }
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Literal(Bits(0.125f))], [Vector(11)]);
        Add("SMovB32", Gen5ShaderEncoding.Sop1, [Literal(5)], [Scalar(126)]);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Literal(Bits(0.75f))], [Vector(12)]);
        Add("VAddF32", Gen5ShaderEncoding.Vop2, [Vector(12), Vector(12)], [Vector(11)]);
        Add("SMovB64", Gen5ShaderEncoding.Sop1, [new Gen5Operand(Gen5OperandKind.EncodedConstant, 193)], [Scalar(126)]);
        Add("Exp", Gen5ShaderEncoding.Exp, [Vector(11), Vector(11), Vector(11), Vector(11)], [],
            new Gen5ExportControl(Gen5NggExports.FirstParameterTarget, 0xF, false, false, false));
        instructions.Add(new(pc, Gen5ShaderEncoding.Sopp, "SEndpgm", [EndProgramWord], [], [], null));
        return new Gen5ShaderProgram(0, instructions);
    }

    public static Gen5ShaderProgram MeshExitLoop(uint iterations)
    {
        // One wave: read back its own LDS rather than the absent second wave.
        var instructions = CullingTriangle().Instructions.SkipLast(1).Select(instruction =>
            instruction.Opcode == "VXorB32"
                ? instruction with { Sources = [Inline(0), Vector(Gen5NggExports.EsVertexIndexRegister)] }
                : instruction).ToList();
        var pc = instructions[^1].Pc + 4;
        void Add(string opcode, Gen5ShaderEncoding encoding, Gen5Operand[] sources,
            Gen5Operand[] destinations, Gen5InstructionControl? control = null, uint word = 0)
        {
            instructions.Add(new(pc, encoding, opcode, [word], sources, destinations, control));
            pc += 4;
        }
        Add("SMovB32", Gen5ShaderEncoding.Sop1, [Literal(iterations)], [Scalar(31)]);
        var loop = pc;
        Add("SSubU32", Gen5ShaderEncoding.Sop2, [Scalar(31), Inline(1)], [Scalar(31)]);
        Add("SCmpEqU32", Gen5ShaderEncoding.Sopc, [Scalar(31), Inline(0)], []);
        Add("SCbranchScc0", Gen5ShaderEncoding.Sopp, [], [], word: Branch(BranchScc0Word, pc, loop));
        // Only a normal loop exit executes this export. A capped loop preserves the
        // preceding parameter, making an incorrect fall-through visible in the image.
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Literal(Bits(0.25f))], [Vector(15)]);
        Add("Exp", Gen5ShaderEncoding.Exp, [Vector(15), Vector(15), Vector(15), Vector(15)], [],
            new Gen5ExportControl(Gen5NggExports.FirstParameterTarget, 0xF, false, false, false));
        Add("SEndpgm", Gen5ShaderEncoding.Sopp, [], [], word: EndProgramWord);
        return new Gen5ShaderProgram(0, instructions);
    }

    private static void AppendTail(List<Gen5ShaderInstruction> instructions, ref uint pc)
    {
        var at = pc;
        void Add(string opcode, Gen5ShaderEncoding encoding, Gen5Operand[] sources,
            Gen5Operand[] destinations, Gen5InstructionControl? control = null, uint word = 0)
        {
            instructions.Add(new Gen5ShaderInstruction(
                at, encoding, opcode, [word], sources, destinations, control));
            at += sizeof(uint);
        }

        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Literal(Bits(1.0f))], [Vector(21)]);
        Add("SMovB32", Gen5ShaderEncoding.Sop1, [Literal(AllocationRequest)], [Scalar(124)]);
        Add("SSendmsg", Gen5ShaderEncoding.Sopp, [], [], word: SendMessageWord);
        Add("Exp", Gen5ShaderEncoding.Exp,
            [Vector(21), Vector(21), Vector(21), Vector(21)], [],
            new Gen5ExportControl(Gen5NggExports.FirstPositionTarget, 0xF, false, false, false));
        Add("SEndpgm", Gen5ShaderEncoding.Sopp, [], [], word: EndProgramWord);
        pc = at;
    }

    /// <summary>An S_BARRIER that only the threads which do not take a scalar branch reach.
    /// No barrier built this way is workgroup-uniform.</summary>
    public static Gen5ShaderProgram BarrierInDivergentBranch()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        instructions.Add(new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Ds, "DsWriteB32", [0],
            [Vector(11), Vector(10)], [], new Gen5DataShareControl(0, 0, false)));
        pc += 4;
        instructions.Add(new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Sopp, "SCbranchScc1",
            [Branch(BranchScc1Word, pc, pc + 8)], [], [], null));
        pc += 4;
        instructions.Add(new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Sopp, "SBarrier", [0], [], [], null));
        pc += 4;
        AppendTail(instructions, ref pc);
        return new Gen5ShaderProgram(0, instructions);
    }

    /// <summary>A loop whose body holds the barrier and which leaves through two different
    /// blocks, a shape the structured assembler cannot nest.</summary>
    public static Gen5ShaderProgram BarrierLoopWithTwoExits()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        instructions.Add(new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Ds, "DsWriteB32", [0],
            [Vector(11), Vector(10)], [], new Gen5DataShareControl(0, 0, false)));
        pc += 4;

        // 4: header; leaves to the exit at 20 or runs the body.
        instructions.Add(new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Sopp, "SCbranchScc1",
            [Branch(BranchScc1Word, pc, 20)], [], [], null));
        pc += 4;

        // 8: the barrier; 12: the back edge to the header.
        instructions.Add(new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Sopp, "SBarrier", [0], [], [], null));
        pc += 4;
        instructions.Add(new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Sopp, "SCbranchScc0",
            [Branch(BranchScc0Word, pc, 4)], [], [], null));
        pc += 4;

        // 16: the second exit jumps over the first to the tail at 24.
        instructions.Add(new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Sopp, "SBranch",
            [Branch(BranchWord, pc, 24)], [], [], null));
        pc += 4;

        // 20: the first exit falls into the tail.
        instructions.Add(new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Sop1, "SMovB32", [0],
            [Inline(0)], [Scalar(20)], null));
        pc += 4;
        AppendTail(instructions, ref pc);
        return new Gen5ShaderProgram(0, instructions);
    }

    /// <summary>The same two-exit loop, then a barrier after it: the loop holds no barrier,
    /// so every thread reaches the barrier once whatever the number of waves.</summary>
    public static Gen5ShaderProgram BarrierAfterLoopWithTwoExits()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        instructions.Add(new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Ds, "DsWriteB32", [0],
            [Vector(11), Vector(10)], [], new Gen5DataShareControl(0, 0, false)));
        pc += 4;
        instructions.Add(new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Sopp, "SCbranchScc1",
            [Branch(BranchScc1Word, pc, 20)], [], [], null));
        pc += 4;
        instructions.Add(new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Sop1, "SMovB32", [0],
            [Inline(0)], [Scalar(20)], null));
        pc += 4;
        instructions.Add(new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Sopp, "SCbranchScc0",
            [Branch(BranchScc0Word, pc, 4)], [], [], null));
        pc += 4;
        instructions.Add(new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Sopp, "SBranch",
            [Branch(BranchWord, pc, 24)], [], [], null));
        pc += 4;
        instructions.Add(new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Sop1, "SMovB32", [0],
            [Inline(0)], [Scalar(20)], null));
        pc += 4;
        instructions.Add(new Gen5ShaderInstruction(pc, Gen5ShaderEncoding.Sopp, "SBarrier", [0], [], [], null));
        pc += 4;
        AppendTail(instructions, ref pc);
        return new Gen5ShaderProgram(0, instructions);
    }

    /// <summary>Every visible vertex samples the same LDS record, so its red output has
    /// an exact CPU oracle. Mutations cover masked writers, aliasing stores/atomics,
    /// another region, non-affine writers and wrapped/unaligned read addresses.</summary>
    public static Gen5ShaderProgram MeshLdsForwardingProbe(uint readLane, int mutation, uint baseAddress = 0, bool allComponents = false)
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        void Add(string op, Gen5ShaderEncoding encoding, Gen5Operand[] sources,
            Gen5Operand[] destinations, Gen5InstructionControl? control = null)
        {
            instructions.Add(new Gen5ShaderInstruction(pc, encoding, op, [0], sources, destinations, control));
            pc += 4;
        }
        static Gen5DataShareControl Offset(uint value) => new(value & 255u, value >> 8, false);
        var all = new Gen5Operand(Gen5OperandKind.EncodedConstant, 193);
        void Restore() => Add("SMovB64", Gen5ShaderEncoding.Sop1, [all], [Scalar(126), Scalar(127)]);
        Restore();
        Add("VMbcntLoU32B32", Gen5ShaderEncoding.Vop3, [all, Inline(0)], [Vector(40)]);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Vector(40)], [Vector(44)]);
        if (mutation == 3)
            Add("VXorB32", Gen5ShaderEncoding.Vop2, [Inline(1), Vector(44)], [Vector(44)]);
        Add("VLshlrevB32", Gen5ShaderEncoding.Vop2, [Inline(4), Vector(44)], [Vector(44)]);
        if (baseAddress != 0)
            Add("VAddI32", Gen5ShaderEncoding.Vop2, [Literal(baseAddress), Vector(44)], [Vector(44)]);
        for (uint i = 1; i < 4; i++) Add("VMovB32", Gen5ShaderEncoding.Vop1, [Inline(i)], [Vector(40 + i)]);
        Add("DsWriteB128", Gen5ShaderEncoding.Ds, [Vector(44), Vector(40), Vector(41), Vector(42), Vector(43)], [], Offset(128));
        if (mutation == 1)
        {
            Add("SMovB64", Gen5ShaderEncoding.Sop1, [Literal(255)], [Scalar(126), Scalar(127)]);
            Add("VAddI32", Gen5ShaderEncoding.Vop2, [Literal(100), Vector(40)], [Vector(40)]);
            Add("DsWriteB128", Gen5ShaderEncoding.Ds, [Vector(44), Vector(40), Vector(41), Vector(42), Vector(43)], [], Offset(128));
            Restore();
        }
        if (mutation == 4)
            Add("DsWriteB128", Gen5ShaderEncoding.Ds, [Vector(44), Vector(40), Vector(41), Vector(42), Vector(43)], [], Offset(640));
        if (mutation is 2 or 5 or 6 or 10)
        {
            Add("SMovB64", Gen5ShaderEncoding.Sop1, [Inline(2)], [Scalar(126), Scalar(127)]);
            Add("VMovB32", Gen5ShaderEncoding.Vop1, [Literal(baseAddress + readLane * 16 + (mutation == 10 ? 128u : 0u))], [Vector(45)]);
            Add("VMovB32", Gen5ShaderEncoding.Vop1, [Literal(mutation is 2 or 10 ? 999u : mutation == 5 ? 127u : 1u)], [Vector(46)]);
            Add(mutation is 2 or 10 ? "DsWriteB32" : mutation == 5 ? "DsWriteB8" : "DsAddU32",
                Gen5ShaderEncoding.Ds, [Vector(45), Vector(46)], [], Offset(mutation == 10 ? 0u : 128u));
            Restore();
        }
        Add("SBarrier", Gen5ShaderEncoding.Sopp, [], []);
        var readAddress = baseAddress + readLane * 16 + (mutation == 7 ? 1024u : mutation == 8 ? 1u : 0u);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Literal(readAddress)], [Vector(45)]);
        Add("DsReadB128", Gen5ShaderEncoding.Ds, [Vector(45)], [Vector(50), Vector(51), Vector(52), Vector(53)], Offset(128));
        if (allComponents)
        {
            for (uint i = 1; i < 4; i++)
            {
                Add("VLshlrevB32", Gen5ShaderEncoding.Vop2, [Inline(i * 5), Vector(50 + i)], [Vector(55)]);
                Add("VAddI32", Gen5ShaderEncoding.Vop2, [Vector(55), Vector(50)], [Vector(50)]);
            }
        }
        Add("VCvtF32U32", Gen5ShaderEncoding.Vop1, [Vector(50)], [Vector(54)]);
        foreach (var instruction in GeometryQuad(false, 0).Instructions)
        {
            var next = instruction with { Pc = pc };
            if (next.Control is Gen5ExportControl { Target: Gen5NggExports.FirstParameterTarget })
                next = next with { Sources = new[] { Vector(54), Vector(20), Vector(20), Vector(21) } };
            instructions.Add(next);
            pc += 4;
        }
        return new Gen5ShaderProgram(0, instructions);
    }

    /// <summary>A fused geometry program emits a quad from one input triangle. With a
    /// nonzero first vertex its output lives in a later guest wave. The lane exchange
    /// and LDS round trip exercise both halves of wave64 and the workgroup barrier.</summary>
    public static Gen5ShaderProgram GeometryQuad(bool wave64, uint firstVertex)
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        void Add(string opcode, Gen5ShaderEncoding encoding, Gen5Operand[] sources,
            Gen5Operand[] destinations, Gen5InstructionControl? control = null, uint word = 0)
        {
            instructions.Add(new Gen5ShaderInstruction(pc, encoding, opcode, [word], sources, destinations, control));
            pc += sizeof(uint);
        }
        var minusOne = new Gen5Operand(Gen5OperandKind.EncodedConstant, 193);
        Add("VMbcntLoU32B32", Gen5ShaderEncoding.Vop3, [minusOne, Inline(0)], [Vector(10)]);
        if (wave64)
            Add("VMbcntHiU32B32", Gen5ShaderEncoding.Vop3, [minusOne, Vector(10)], [Vector(10)]);
        Add("SBfeU32", Gen5ShaderEncoding.Sop2, [Scalar(3), Literal(0x00040018)], [Scalar(20)]);
        Add("VLshlAddU32", Gen5ShaderEncoding.Vop3, [Scalar(20), Inline(wave64 ? 6u : 5u), Vector(10)], [Vector(11)]);
        Add("VLshlrevB32", Gen5ShaderEncoding.Vop2, [Inline(2), Vector(11)], [Vector(12)]);
        Add("DsWriteB32", Gen5ShaderEncoding.Ds, [Vector(12), Vector(11)], [], new Gen5DataShareControl(0, 0, false));
        Add("SBarrier", Gen5ShaderEncoding.Sopp, [], []);
        Add("DsReadB32", Gen5ShaderEncoding.Ds, [Vector(12)], [Vector(11)], new Gen5DataShareControl(0, 0, false));
        Add("VReadlaneB32", Gen5ShaderEncoding.Vop3, [Vector(10), Inline(wave64 ? 63u : 31u)], [Scalar(21)]);
        Add("SAddI32", Gen5ShaderEncoding.Sop2, [Scalar(21), Literal(unchecked(0u - (wave64 ? 63u : 31u)))], [Scalar(21)]);
        Add("VAddI32", Gen5ShaderEncoding.Vop2, [Literal(unchecked(0u - firstVertex)), Vector(11)], [Vector(14)]);
        Add("VAddI32", Gen5ShaderEncoding.Vop2, [Scalar(21), Vector(14)], [Vector(14)]);
        Add("VAndB32", Gen5ShaderEncoding.Vop2, [Inline(1), Vector(14)], [Vector(15)]);
        Add("VCvtF32U32", Gen5ShaderEncoding.Vop1, [Vector(15)], [Vector(16)]);
        Add("VMulF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(1.6f)), Vector(16)], [Vector(16)]);
        Add("VAddF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(-0.8f)), Vector(16)], [Vector(16)]);
        Add("VLshrrevB32", Gen5ShaderEncoding.Vop2, [Inline(1), Vector(14)], [Vector(17)]);
        Add("VCvtF32U32", Gen5ShaderEncoding.Vop1, [Vector(17)], [Vector(18)]);
        Add("VMulF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(1.6f)), Vector(18)], [Vector(19)]);
        Add("VAddF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(-0.8f)), Vector(19)], [Vector(19)]);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Inline(0)], [Vector(20)]);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Literal(Bits(1.0f))], [Vector(21)]);
        Add("SMovB32", Gen5ShaderEncoding.Sop1,
            [Literal((firstVertex + 4) | (2u << Gen5NggExports.AllocationPrimitiveCountShift))], [Scalar(124)]);
        Add("SSendmsg", Gen5ShaderEncoding.Sopp, [], [], word: SendMessageWord);
        Add("Exp", Gen5ShaderEncoding.Exp, [Vector(16), Vector(19), Vector(20), Vector(21)], [],
            new Gen5ExportControl(Gen5NggExports.FirstPositionTarget, 0xF, false, false, false));
        Add("Exp", Gen5ShaderEncoding.Exp, [Vector(20), Vector(20), Vector(20), Vector(21)], [],
            new Gen5ExportControl(Gen5NggExports.FirstParameterTarget, 0xF, false, false, false));
        // Primitive 0 joins vertices 0,1,2; primitive 1 joins 1,3,2.
        var a = firstVertex | ((firstVertex + 1) << 10) | ((firstVertex + 2) << 20);
        var b = (firstVertex + 1) | ((firstVertex + 3) << 10) | ((firstVertex + 2) << 20);
        Add("VCmpEqU32", Gen5ShaderEncoding.Vop3, [Inline(1), Vector(11)], [Scalar(30)]);
        Add("VCndmaskB32", Gen5ShaderEncoding.Vop3, [Literal(a), Literal(b), Scalar(30)], [Vector(22)]);
        Add("Exp", Gen5ShaderEncoding.Exp, [Vector(22), Vector(22), Vector(22), Vector(22)], [],
            new Gen5ExportControl(Gen5NggExports.PrimitiveTarget, 0x1, false, true, false));
        Add("SEndpgm", Gen5ShaderEncoding.Sopp, [], [], word: EndProgramWord);
        return new Gen5ShaderProgram(0, instructions);
    }

    /// <summary>The stage state of a wave64 primitive shader with no GS half.</summary>
    public static Gen5NggStageInfo Wave64StageInfo { get; } = StageInfo with
    {
        PrimitivesPerSubgroup = 64,
        IsWave64 = true,
    };

    /// <summary>A subgroup of 10 triangles over 30 vertices, the narrow-wave64 launch.</summary>
    public const uint Wave64SubgroupAllocation = 30u | (10u << Gen5NggExports.AllocationPrimitiveCountShift);

    /// <summary>The compiler-generated culling template's shape, reduced: no barrier, no LDS,
    /// every primitive owning three vertices. A vertex's position comes from its fetched
    /// index (x = 0.8 * index - 0.8, y = 1.6 * (index &amp; 1) - 0.8), and a primitive decodes
    /// its three local vertices from v0/v1 the way the guest does (bits 2-15 of each 16-bit
    /// field) before exporting them.</summary>
    public static Gen5ShaderProgram Wave64Triangles()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;

        void Add(string opcode, Gen5ShaderEncoding encoding, Gen5Operand[] sources,
            Gen5Operand[] destinations, Gen5InstructionControl? control = null, uint word = 0)
        {
            instructions.Add(new Gen5ShaderInstruction(
                pc, encoding, opcode, [word], sources, destinations, control));
            pc += sizeof(uint);
        }

        Add("DsWriteB32", Gen5ShaderEncoding.Ds,
            [Vector(11), Vector(10)], [], new Gen5DataShareControl(0, 0, false));
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Vector(Gen5NggExports.EsVertexIndexRegister)], [Vector(14)]);
        Add("VCvtF32U32", Gen5ShaderEncoding.Vop1, [Vector(14)], [Vector(15)]);
        Add("VMulF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(0.8f)), Vector(15)], [Vector(16)]);
        Add("VAddF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(-0.8f)), Vector(16)], [Vector(16)]);
        Add("VAndB32", Gen5ShaderEncoding.Vop2, [Inline(1), Vector(14)], [Vector(17)]);
        Add("VCvtF32U32", Gen5ShaderEncoding.Vop1, [Vector(17)], [Vector(18)]);
        Add("VMulF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(1.6f)), Vector(18)], [Vector(19)]);
        Add("VAddF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(-0.8f)), Vector(19)], [Vector(19)]);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Inline(0)], [Vector(20)]);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Literal(Bits(1.0f))], [Vector(21)]);
        Add("SMovB32", Gen5ShaderEncoding.Sop1, [Literal(Wave64SubgroupAllocation)], [Scalar(124)]);
        Add("SSendmsg", Gen5ShaderEncoding.Sopp, [], [], word: SendMessageWord);
        Add("Exp", Gen5ShaderEncoding.Exp,
            [Vector(16), Vector(19), Vector(20), Vector(21)], [],
            new Gen5ExportControl(Gen5NggExports.FirstPositionTarget, 0xF, false, false, false));
        Add("Exp", Gen5ShaderEncoding.Exp,
            [Vector(15), Vector(20), Vector(20), Vector(21)], [],
            new Gen5ExportControl(Gen5NggExports.FirstParameterTarget, 0xF, false, false, false));

        // i0 = bfe(v0, 2, 14), i1 = v0 >> 18, i2 = bfe(v1, 2, 14), packed one per 10-bit field.
        Add("VLshrrevB32", Gen5ShaderEncoding.Vop2, [Inline(2), Vector(Gen5NggExports.GsVertexOffset01Register)], [Vector(30)]);
        Add("VAndB32", Gen5ShaderEncoding.Vop2, [Literal(0x3FFF), Vector(30)], [Vector(30)]);
        Add("VLshrrevB32", Gen5ShaderEncoding.Vop2, [Literal(18), Vector(Gen5NggExports.GsVertexOffset01Register)], [Vector(31)]);
        Add("VLshlrevB32", Gen5ShaderEncoding.Vop2, [Literal(Gen5NggExports.PrimitiveIndexStride), Vector(31)], [Vector(31)]);
        Add("VLshrrevB32", Gen5ShaderEncoding.Vop2, [Inline(2), Vector(Gen5NggExports.GsVertexOffset23Register)], [Vector(32)]);
        Add("VAndB32", Gen5ShaderEncoding.Vop2, [Literal(0x3FFF), Vector(32)], [Vector(32)]);
        Add("VLshlrevB32", Gen5ShaderEncoding.Vop2, [Literal(2 * Gen5NggExports.PrimitiveIndexStride), Vector(32)], [Vector(32)]);
        Add("VOrB32", Gen5ShaderEncoding.Vop2, [Vector(31), Vector(30)], [Vector(22)]);
        Add("VOrB32", Gen5ShaderEncoding.Vop2, [Vector(32), Vector(22)], [Vector(22)]);
        Add("Exp", Gen5ShaderEncoding.Exp,
            [Vector(22), Vector(22), Vector(22), Vector(22)], [],
            new Gen5ExportControl(Gen5NggExports.PrimitiveTarget, 0x1, false, true, false));
        Add("SEndpgm", Gen5ShaderEncoding.Sopp, [], [], word: EndProgramWord);
        return new Gen5ShaderProgram(0, instructions);
    }

    /// <summary>A lane mask written to an SGPR pair whose high dword held stale data, then
    /// counted with s_bcnt1_i32_b64: the culling templates' survivor count.</summary>
    public static Gen5ShaderProgram Wave64MaskPairProbe()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;

        void Add(string opcode, Gen5ShaderEncoding encoding, Gen5Operand[] sources,
            Gen5Operand[] destinations, Gen5InstructionControl? control = null, uint word = 0)
        {
            instructions.Add(new Gen5ShaderInstruction(
                pc, encoding, opcode, [word], sources, destinations, control));
            pc += sizeof(uint);
        }

        Add("DsWriteB32", Gen5ShaderEncoding.Ds,
            [Vector(11), Vector(10)], [], new Gen5DataShareControl(0, 0, false));
        Add("SMovB32", Gen5ShaderEncoding.Sop1, [Literal(0xFFFF_FFFFu)], [Scalar(1)]);
        Add("VCmpNeU32", Gen5ShaderEncoding.Vop3, [Inline(0), Vector(8)], [Scalar(0)]);
        Add("SBcnt1I32B64", Gen5ShaderEncoding.Sop1, [Scalar(0)], [Scalar(10)]);
        Add("SMovB32", Gen5ShaderEncoding.Sop1, [Literal(Wave64SubgroupAllocation)], [Scalar(124)]);
        Add("SSendmsg", Gen5ShaderEncoding.Sopp, [], [], word: SendMessageWord);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Scalar(10)], [Vector(21)]);
        Add("Exp", Gen5ShaderEncoding.Exp,
            [Vector(21), Vector(21), Vector(21), Vector(21)], [],
            new Gen5ExportControl(Gen5NggExports.FirstPositionTarget, 0xF, false, false, false));
        Add("SEndpgm", Gen5ShaderEncoding.Sopp, [], [], word: EndProgramWord);
        return new Gen5ShaderProgram(0, instructions);
    }

    /// <summary>The cross-lane operations whose wave64 behaviour the narrow lowering pins:
    /// v_readlane of lane 63, v_mbcnt_hi, and an all-ones write to EXEC_HI.</summary>
    public static Gen5ShaderProgram Wave64LaneProbe()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;

        void Add(string opcode, Gen5ShaderEncoding encoding, Gen5Operand[] sources,
            Gen5Operand[] destinations, Gen5InstructionControl? control = null, uint word = 0)
        {
            instructions.Add(new Gen5ShaderInstruction(
                pc, encoding, opcode, [word], sources, destinations, control));
            pc += sizeof(uint);
        }

        Add("DsWriteB32", Gen5ShaderEncoding.Ds,
            [Vector(11), Vector(10)], [], new Gen5DataShareControl(0, 0, false));
        Add("VReadlaneB32", Gen5ShaderEncoding.Vop3,
            [Vector(Gen5NggExports.EsVertexIndexRegister), Literal(63)], [Scalar(20)]);
        Add("VMbcntHiU32B32", Gen5ShaderEncoding.Vop3, [Scalar(20), Vector(11)], [Vector(12)]);
        Add("SMovB32", Gen5ShaderEncoding.Sop1, [Literal(0xFFFF_FFFFu)], [Scalar(127)]);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Literal(Bits(1.0f))], [Vector(21)]);
        Add("SMovB32", Gen5ShaderEncoding.Sop1, [Literal(Wave64SubgroupAllocation)], [Scalar(124)]);
        Add("SSendmsg", Gen5ShaderEncoding.Sopp, [], [], word: SendMessageWord);
        Add("Exp", Gen5ShaderEncoding.Exp,
            [Vector(12), Vector(21), Vector(21), Vector(21)], [],
            new Gen5ExportControl(Gen5NggExports.FirstPositionTarget, 0xF, false, false, false));
        Add("SEndpgm", Gen5ShaderEncoding.Sopp, [], [], word: EndProgramWord);
        return new Gen5ShaderProgram(0, instructions);
    }

    /// <summary>A primitive shader with no LDS and no barrier: the passthrough shape that
    /// must keep using the per-vertex path.</summary>
    public static Gen5ShaderProgram Passthrough()
    {
        var instructions = new List<Gen5ShaderInstruction>
        {
            new(0, Gen5ShaderEncoding.Vop1, "VMovB32", [0],
                [Literal(Bits(1.0f))], [Vector(21)], null),
            new(4, Gen5ShaderEncoding.Exp, "Exp", [0],
                [Vector(21), Vector(21), Vector(21), Vector(21)], [],
                new Gen5ExportControl(Gen5NggExports.FirstPositionTarget, 0xF, false, false, false)),
            new(8, Gen5ShaderEncoding.Sopp, "SSendmsg", [SendMessageWord], [], [], null),
            new(12, Gen5ShaderEncoding.Sopp, "SEndpgm", [EndProgramWord], [], [], null),
        };
        return new Gen5ShaderProgram(0, instructions);
    }

    /// <summary>NGG passthrough mode (VGT_SHADER_STAGES_EN.PRIMGEN_PASSTHRU_EN).</summary>
    public static Gen5NggStageInfo PassthroughStageInfo { get; } = StageInfo with { Passthrough = true };

    /// <summary>A passthrough-mode primitive shader as the guest compiler emits one: no LDS,
    /// no barrier, and the primitive exported straight from v0, which the geometry engine
    /// packs. Vertex n of the draw sits at the CullingTriangle corner n and exports
    /// param0.x = n.</summary>
    public static Gen5ShaderProgram PassthroughTriangle()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;

        void Add(string opcode, Gen5ShaderEncoding encoding, Gen5Operand[] sources,
            Gen5Operand[] destinations, Gen5InstructionControl? control = null, uint word = 0)
        {
            instructions.Add(new Gen5ShaderInstruction(
                pc, encoding, opcode, [word], sources, destinations, control));
            pc += sizeof(uint);
        }

        Add("SMovB32", Gen5ShaderEncoding.Sop1, [Literal(AllocationRequest)], [Scalar(124)]);
        Add("SSendmsg", Gen5ShaderEncoding.Sopp, [], [], word: SendMessageWord);
        Add("Exp", Gen5ShaderEncoding.Exp,
            [Vector(0), Vector(0), Vector(0), Vector(0)], [],
            new Gen5ExportControl(Gen5NggExports.PrimitiveTarget, 0x1, false, true, false));

        Add("VCvtF32U32", Gen5ShaderEncoding.Vop1, [Vector(Gen5NggExports.EsVertexIndexRegister)], [Vector(15)]);
        Add("VMulF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(0.8f)), Vector(15)], [Vector(16)]);
        Add("VAddF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(-0.8f)), Vector(16)], [Vector(16)]);
        Add("VAndB32", Gen5ShaderEncoding.Vop2, [Inline(1), Vector(Gen5NggExports.EsVertexIndexRegister)], [Vector(17)]);
        Add("VCvtF32U32", Gen5ShaderEncoding.Vop1, [Vector(17)], [Vector(18)]);
        Add("VMulF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(1.6f)), Vector(18)], [Vector(19)]);
        Add("VAddF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(-0.8f)), Vector(19)], [Vector(19)]);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Inline(0)], [Vector(20)]);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Literal(Bits(1.0f))], [Vector(21)]);
        Add("Exp", Gen5ShaderEncoding.Exp,
            [Vector(16), Vector(19), Vector(20), Vector(21)], [],
            new Gen5ExportControl(Gen5NggExports.FirstPositionTarget, 0xF, false, false, false));
        Add("Exp", Gen5ShaderEncoding.Exp,
            [Vector(15), Vector(20), Vector(20), Vector(21)], [],
            new Gen5ExportControl(Gen5NggExports.FirstParameterTarget, 0xF, false, true, false));
        Add("SEndpgm", Gen5ShaderEncoding.Sopp, [], [], word: EndProgramWord);
        return new Gen5ShaderProgram(0, instructions);
    }

    /// <summary>The same program with its vertex positions fetched the way the ES part of a
    /// real primitive shader does: <c>buffer_load_format_xyzw</c> indexed by the vertex
    /// index, over a V# the geometry stage's user data carries. That fetch is exactly what
    /// the fixed-function vertex-fetch substitution used to swallow, leaving the mesh stage
    /// with no buffer binding at all.</summary>
    public static Gen5ShaderProgram BufferFetchTriangle() => CullingTriangle(bufferFetchPosition: true);

    // Vertex positions live at the start of the fetched buffer, one float4 per vertex.
    public const uint PositionStrideBytes = 16;
    public const uint PositionRecordCount = 4;
    private const uint PositionBaseAddress = 0x20_0000;

    // 32_32_32_32_FLOAT with the identity destination select, as a V# carries it.
    private const uint PositionDescriptorFormat = 77;

    /// <summary>The geometry stage's user data for <see cref="BufferFetchTriangle"/>: the V#
    /// at s[8:11], which is user-data words 0..3 when the stage's user data starts at s8.</summary>
    public static uint[] PositionUserData()
    {
        var userData = new uint[16];
        userData[0] = PositionBaseAddress;
        userData[1] = PositionStrideBytes << 16;
        userData[2] = PositionRecordCount;
        userData[3] = (PositionDescriptorFormat << 12) | DescriptorConstants.IdentityDestinationSelect;
        return userData;
    }

    public static ShaderCompileRequest Request(
        Gen5ShaderProgram program,
        Gen5NggStageInfo? stageInfo = null,
        int requiredVertexOutputCount = 1,
        ResourceRuntimeInputs? runtimeInputs = null,
        uint positionExportControl = 0,
        ShaderClipSpaceTransform clipSpace = default,
        uint meshPerVertexParameterMask = 0) =>
        Gen5ComputeFixtures.RequestOrThrow(
            program,
            ShaderStage.Vertex,
            requiredVertexOutputCount: requiredVertexOutputCount,
            runtimeInputs: runtimeInputs,
            primitiveShader: stageInfo ?? StageInfo,
            userDataBase: VertexUserDataBase,
            positionExportControl: positionExportControl,
            clipSpace: clipSpace,
            meshPerVertexParameterMask: meshPerVertexParameterMask);

    public static Gen5MslShader CompileOrThrow(
        Gen5ShaderProgram program,
        Gen5NggStageInfo? stageInfo = null,
        int requiredVertexOutputCount = 1,
        ResourceRuntimeInputs? runtimeInputs = null,
        uint positionExportControl = 0,
        ShaderClipSpaceTransform clipSpace = default,
        uint meshPerVertexParameterMask = 0)
    {
        var request = Request(program, stageInfo, requiredVertexOutputCount, runtimeInputs, positionExportControl, clipSpace,
            meshPerVertexParameterMask);
        if (!Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error))
        {
            throw new InvalidOperationException($"NGG mesh emit failed: {error}");
        }

        return shader;
    }

    /// <summary>An amplified wave64 subgroup (Prospero's mesh shader): one ES vertex in, up
    /// to 128 vertices and 128 primitives out, so two 64-lane waves.</summary>
    public static Gen5NggStageInfo AmplifiedStageInfo { get; } = StageInfo with
    {
        VerticesPerSubgroup = 1,
        PrimitivesPerSubgroup = 0,
        MaxOutputPerSubgroup = 128,
        PrimitiveAmplificationFactor = 128,
        IsWave64 = true,
        AmplifiedVerticesOut = 128,
    };

    /// <summary>The thread the amplified fixture's triangle starts at by default: lane 36 of
    /// the second wave, in the upper half of that wave.</summary>
    public const uint AmplifiedFirstVertex = 100;

    /// <summary>The shape of Black Myth: Wukong's Nanite rasterizer, reduced. A thread numbers
    /// itself (merged-wave subgroup id &lt;&lt; 6) + V_MBCNT over both halves, asks for 103 vertices
    /// and one primitive, and exports its own vertex: x = 0.8 * value - 0.8 and
    /// y = 1.6 * (value &amp; 1) - 0.8, value = thread - 100 + (workgroup ^ 1) * 64, the workgroup
    /// being its ES vertex index. Primitive 0 joins threads 100/101/102, so only workgroup 1
    /// puts its triangle through the screen center, and only if the lanes, the second wave
    /// and the workgroup index all line up.</summary>
    /// <para>With <paramref name="readLane"/> every thread also adds
    /// V_READLANE(lane, readLane) - readLane, which is zero only when the read reaches the
    /// named lane: for a triangle in the low half of a wave and a lane in its high half,
    /// that is the bridged exchange between the wave's two simdgroups.</para>
    public static Gen5ShaderProgram AmplifiedWave64Triangle(uint firstVertex = AmplifiedFirstVertex, uint? readLane = null)
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;

        void Add(string opcode, Gen5ShaderEncoding encoding, Gen5Operand[] sources,
            Gen5Operand[] destinations, Gen5InstructionControl? control = null, uint word = 0)
        {
            instructions.Add(new Gen5ShaderInstruction(
                pc, encoding, opcode, [word], sources, destinations, control));
            pc += sizeof(uint);
        }

        var minusOne = new Gen5Operand(Gen5OperandKind.EncodedConstant, 193);
        Add("VMbcntLoU32B32", Gen5ShaderEncoding.Vop3, [minusOne, Inline(0)], [Vector(0)]);
        Add("SBfeU32", Gen5ShaderEncoding.Sop2, [Scalar(3), Literal(0x00040018)], [Scalar(20)]);
        Add("VMbcntHiU32B32", Gen5ShaderEncoding.Vop3, [minusOne, Vector(0)], [Vector(0)]);
        Add("VLshlAddU32", Gen5ShaderEncoding.Vop3, [Scalar(20), Inline(6), Vector(0)], [Vector(4)]);

        Add("VXorB32", Gen5ShaderEncoding.Vop2, [Inline(1), Vector(Gen5NggExports.EsVertexIndexRegister)], [Vector(30)]);
        Add("VLshlrevB32", Gen5ShaderEncoding.Vop2, [Inline(6), Vector(30)], [Vector(30)]);
        Add("VAddI32", Gen5ShaderEncoding.Vop2, [Literal(unchecked((uint)-(int)firstVertex)), Vector(4)], [Vector(14)]);
        Add("VAddI32", Gen5ShaderEncoding.Vop2, [Vector(30), Vector(14)], [Vector(14)]);
        if (readLane is { } lane)
        {
            Add("VReadlaneB32", Gen5ShaderEncoding.Vop3, [Vector(0), Literal(lane)], [Scalar(21)]);
            Add("SAddI32", Gen5ShaderEncoding.Sop2, [Scalar(21), Literal(unchecked((uint)-(int)lane))], [Scalar(21)]);
            Add("VAddI32", Gen5ShaderEncoding.Vop2, [Scalar(21), Vector(14)], [Vector(14)]);
        }

        Add("VCvtF32I32", Gen5ShaderEncoding.Vop1, [Vector(14)], [Vector(15)]);
        Add("VMulF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(0.8f)), Vector(15)], [Vector(16)]);
        Add("VAddF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(-0.8f)), Vector(16)], [Vector(16)]);
        Add("VAndB32", Gen5ShaderEncoding.Vop2, [Inline(1), Vector(14)], [Vector(17)]);
        Add("VCvtF32U32", Gen5ShaderEncoding.Vop1, [Vector(17)], [Vector(18)]);
        Add("VMulF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(1.6f)), Vector(18)], [Vector(19)]);
        Add("VAddF32", Gen5ShaderEncoding.Vop2, [Literal(Bits(-0.8f)), Vector(19)], [Vector(19)]);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Inline(0)], [Vector(20)]);
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Literal(Bits(1.0f))], [Vector(21)]);

        Add("SMovB32", Gen5ShaderEncoding.Sop1,
            [Literal((firstVertex + 3) | (1u << Gen5NggExports.AllocationPrimitiveCountShift))], [Scalar(124)]);
        Add("SSendmsg", Gen5ShaderEncoding.Sopp, [], [], word: SendMessageWord);
        Add("Exp", Gen5ShaderEncoding.Exp,
            [Vector(16), Vector(19), Vector(20), Vector(21)], [],
            new Gen5ExportControl(Gen5NggExports.FirstPositionTarget, 0xF, false, false, false));
        Add("Exp", Gen5ShaderEncoding.Exp,
            [Vector(15), Vector(20), Vector(20), Vector(21)], [],
            new Gen5ExportControl(Gen5NggExports.FirstParameterTarget, 0xF, false, false, false));
        var connectivity =
            (firstVertex << (0 * Gen5NggExports.PrimitiveIndexStride)) |
            ((firstVertex + 1) << (1 * Gen5NggExports.PrimitiveIndexStride)) |
            ((firstVertex + 2) << (2 * Gen5NggExports.PrimitiveIndexStride));
        Add("VMovB32", Gen5ShaderEncoding.Vop1, [Literal(connectivity)], [Vector(22)]);
        Add("Exp", Gen5ShaderEncoding.Exp,
            [Vector(22), Vector(22), Vector(22), Vector(22)], [],
            new Gen5ExportControl(Gen5NggExports.PrimitiveTarget, 0x1, false, true, false));
        Add("SEndpgm", Gen5ShaderEncoding.Sopp, [], [], word: EndProgramWord);
        return new Gen5ShaderProgram(0, instructions);
    }

    private static uint Bits(float value) => BitConverter.SingleToUInt32Bits(value);

    private static Gen5Operand Literal(uint value) => new(Gen5OperandKind.LiteralConstant, value);

    private static Gen5Operand Inline(uint value) =>
        new(Gen5OperandKind.EncodedConstant, 128 + value);

    private static Gen5Operand Vector(uint index) => Gen5Operand.Vector(index);

    private static Gen5Operand Scalar(uint index) => Gen5Operand.Scalar(index);
}
