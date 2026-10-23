// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed class EmbeddedVertexFetchDetectorTests
{
    private const int AttributeTable = 4;
    private const int BufferTable = 6;

    [Fact]
    public void DepthPrologRecognizesPositionAndSdwaInstanceMatrixReads()
    {
        // The actual instruction shape/PCs of dumped VS 0x8B94AE10B80DDFCD.
        // The instance add uses the low 16 bits of user s10, even when EXEC
        // was narrowed by the preceding NGG vertex/primitive split.
        var add = Vop2(0x48, "VAddI32", 4, Gen5Operand.Vector(8), Gen5Operand.Scalar(10)) with
        { Control = new Gen5SdwaControl(6, 0, 6, 4, false, false, 0, 0, 0, false, null) };
        var program = Program(
            MoveScalar(0, 126, 1),
            add,
            ScalarLoad(0x50, 8, 106, 2, 16),
            ScalarLoad(0x58, 8, 16, 4),
            ScalarLoad(0x64, 106, 20, 4, 16),
            ScalarLoad(0x6C, 16, 24, 4, 16),
            BufferAccess(0x88, "BufferLoadDwordx4", 20, 32, 4, 10, true, vectorAddress: 4),
            BufferAccess(0x90, "BufferLoadDwordx4", 20, 16, 4, 0, true, vectorAddress: 4),
            BufferAccess(0x98, "BufferLoadDwordx4", 20, 0, 4, 6, true, vectorAddress: 4),
            BufferAccess(0xA0, "BufferLoadFormatXyz", 24, 0, 3, 14, true, vectorAddress: 5),
            EndProgram(0x130));
        var plan = EmbeddedVertexFetchDetector.Detect(program, -2, -2, 8, 16, 32);
        Assert.Equal(new uint[] { 0x88, 0x90, 0x98, 0xA0 }, plan.AffineLoads.Select(l => l.Pc));
        foreach (var matrix in plan.AffineLoads.Take(3))
        {
            Assert.True(matrix.Raw);
            Assert.Equal(new AffineVertexIndex(VertexFetchIndex.Instance, BiasScalarRegister: 10, BiasMask: 0xFFFF), matrix.Index);
        }
        var position = plan.AffineLoads[3];
        Assert.Equal(0xA0u, position.Pc);
        Assert.Equal(new AffineVertexIndex(VertexFetchIndex.Vertex), position.Index);
        Assert.Equal(3u, position.Components);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AffineIndicesUseTheSharedConstantAndLaneSpillDescriptorProof(bool spill)
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        uint[] words = [0x200000, 16u << 16, 8, 77u << 12 | 0xFACu];
        for (uint i = 0; i < 4; i++)
        {
            instructions.Add(MoveScalar(pc, 16 + i, words[i])); pc += 4;
            if (!spill) continue;
            instructions.Add(WriteLane(pc, 31, 16 + i, i)); pc += 8;
            instructions.Add(ReadLane(pc, 24 + i, 31, i)); pc += 8;
        }
        instructions.Add(Vop2(pc, "VLshrrevB32", 9, Operand(1), Gen5Operand.Vector(8))); pc += 4;
        instructions.Add(Vop2(pc, "VAddNcU32", 9, Operand(3), Gen5Operand.Vector(9))); pc += 4;
        instructions.Add(BufferAccess(pc, "BufferLoadFormatXyzw", spill ? 24u : 16u, dwords: 4, indexEnabled: true, vectorAddress: 9));
        instructions.Add(EndProgram(pc + 8));
        var program = Program(instructions.ToArray());
        var fetch = Assert.Single(EmbeddedVertexFetchDetector.Detect(program, -2, -2, 8, 16, 32).AffineLoads);
        Assert.Equal(new AffineVertexIndex(VertexFetchIndex.Instance, 2, 3), fetch.Index);
        var resources = ShaderResourcePlan.Extract(program, ShaderStage.Vertex, Hash, 8, 16, waveSize: 32);
        Assert.False(resources.Memory.Find(fetch.Pc)!.DeviceDescriptor);
        Assert.Single(resources.Info.Buffers);
    }

    [Fact]
    public void PositionOnlyPruningPreservesEveryNonParameterInstructionAndPc()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        foreach (var target in new uint[] { 12, 13, 14, 15, 20, 32, 63 })
            instructions.Add(new((uint)instructions.Count * 8, Gen5ShaderEncoding.Exp, "Exp", [0u, 0u],
                [Gen5Operand.Vector(0), Gen5Operand.Vector(1), Gen5Operand.Vector(2), Gen5Operand.Vector(3)], [],
                new Gen5ExportControl(target, 15, false, true, false)));
        instructions.Add(BufferAccess(56, "BufferStoreDword", 16, indexEnabled: true, vectorAddress: 5));
        instructions.Add(EndProgram(64));
        var program = Program(instructions.ToArray());
        var pruned = VertexOutputPruning.PositionOnly(program);
        Assert.Equal(program.Instructions.Select(i => i.Pc), pruned.Instructions.Select(i => i.Pc));
        Assert.Equal(program.Instructions.Where(i => i.Control is not Gen5ExportControl { Target: >= 32 and < 64 }),
            pruned.Instructions.Where(i => i.Opcode != "SNop"));
        Assert.Equal(0u, pruned.ParameterExportMask);
    }

    [Fact]
    public void MaskedWritesAndBranchesDoNotInventAffineIndices()
    {
        var program = Program(
            MoveScalar(0, 126, 0),
            Vop1(4, "VMovB32", 9, Gen5Operand.Vector(5)),
            MoveScalar(8, 126, uint.MaxValue), // widens EXEC after a masked definition
            BufferAccess(12, "BufferLoadFormatX", 16, indexEnabled: true, vectorAddress: 9),
            Branch(16, "SBranch", 0),
            BufferAccess(20, "BufferLoadFormatX", 16, indexEnabled: true, vectorAddress: 5),
            EndProgram(28));
        Assert.Empty(EmbeddedVertexFetchDetector.Detect(program, -2, -2, 8, 16, 32).AffineLoads);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GenericDescriptorLoadsArePrunedOnlyWhenEveryResultIsUnused(bool shared)
    {
        var program = Program(
            ScalarLoad(0, 8, 16, 4),
            ScalarLoad(8, 16, 24, 4, 16),
            BufferAccess(16, "BufferLoadFormatXyz", 24, dwords: 3, vectorData: 14, indexEnabled: true, vectorAddress: 5),
            shared ? BufferAccess(24, "BufferLoadDword", 24, indexEnabled: true, vectorAddress: 5) : Nop(24),
            EndProgram(32));
        var plan = new EmbeddedVertexFetchPlan();
        plan.Loads.Add(new(16, 0, 3, []));
        var prepared = plan.RemoveUnusedScalarLoads(program);
        Assert.Equal(shared ? "SLoadDwordx4" : "SNop", prepared.Instructions[0].Opcode);
        Assert.Equal(shared ? "SLoadDwordx4" : "SNop", prepared.Instructions[1].Opcode);
        Assert.Equal(program.Instructions.Select(instruction => instruction.Pc), prepared.Instructions.Select(instruction => instruction.Pc));
    }

    private static Gen5ShaderInstruction FetchLoad(uint pc, uint resourceRegister, uint dwords) =>
        BufferAccess(pc, "BufferLoadFormat" + dwords switch { 1 => "X", 2 => "Xy", 3 => "Xyz", _ => "Xyzw" }, resourceRegister,
            dwords: dwords, vectorData: 4, indexEnabled: true, vectorAddress: 1);

    private static Gen5ShaderInstruction IndexSelect(uint pc) =>
        Vop2(pc, "VCndmaskB32", 1, Gen5Operand.Vector(8), Gen5Operand.Vector(5));

    [Theory]
    [InlineData(124u)]
    [InlineData(126u)]
    [InlineData(127u)]
    public void SpecialScalarWritesRemainInTheProgramWithoutEnteringFetchTracking(uint specialRegister)
    {
        var instructions = new List<Gen5ShaderInstruction>
        {
            ScalarLoad(0, AttributeTable, destination: 8, count: 4),
            MoveScalar(8, specialRegister, 16),
            MoveScalarRegister(12, specialRegister, 8),
            new(16, Gen5ShaderEncoding.Sopk, "SMovkI32", [0u],
                [new(Gen5OperandKind.LiteralConstant, 16)], [Gen5Operand.Scalar(specialRegister)], null),
        };
        uint instructionAddress = 20;
        foreach (var opcode in new[] { "SAndB32", "SLshlB32", "SBfeU32", "SAddI32", "SAddU32" })
        {
            instructions.Add(Sop2(instructionAddress, opcode, specialRegister, Gen5Operand.Scalar(8), Operand(1)));
            instructions.Add(Sop2(instructionAddress + 4, opcode, specialRegister, Operand(16), Operand(1)));
            instructionAddress += 8;
        }

        instructions.Add(ScalarLoad(instructionAddress, BufferTable, destination: 12, count: 4));
        instructions.Add(IndexSelect(instructionAddress + 8));
        instructions.Add(FetchLoad(instructionAddress + 12, 12, 4));
        instructions.Add(EndProgram(instructionAddress + 20));
        var program = Program(instructions.ToArray());
        var plan = EmbeddedVertexFetchDetector.Detect(program, AttributeTable, BufferTable, 0, 8, 32);

        var load = Assert.Single(plan.Loads);
        Assert.Equal(0, load.AttributeId);
        Assert.Equal([instructionAddress], load.PrologLoads);
        var prepared = plan.RemoveReplacedTableLoads(program);
        Assert.Equal(instructions.Take(instructions.Count - 4), prepared.Instructions.Take(instructions.Count - 4));
    }

    [Theory]
    [InlineData(105u)]
    [InlineData(106u)]
    [InlineData(107u)]
    public void UpperGeneralAndConditionRegistersStillCarryFetchOffsets(uint scalarRegister)
    {
        var program = Program(
            MoveScalar(0, scalarRegister, 16),
            ScalarLoad(4, BufferTable, destination: 12, count: 4, dynamicOffsetRegister: scalarRegister),
            IndexSelect(12),
            FetchLoad(16, 12, 4),
            EndProgram(24));

        var plan = EmbeddedVertexFetchDetector.Detect(program, AttributeTable, BufferTable, 0, 8, 32);
        Assert.Equal(1, Assert.Single(plan.Loads).AttributeId);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ReplacedFetchTablesDoNotRequireDeviceAddressesButOtherReadsStillDo(bool additionalRead)
    {
        var program = Program(
            ScalarLoad(0, AttributeTable, destination: 8, count: 4, immediateOffset: 8),
            Sop2(8, "SLshlB32", 9, Gen5Operand.Scalar(10), Operand(4)),
            ScalarLoad(12, BufferTable, destination: 12, count: 4, dynamicOffsetRegister: 9),
            IndexSelect(20),
            FetchLoad(24, 12, 4),
            additionalRead ? ScalarLoad(32, BufferTable, destination: 20, count: 1, dynamicOffsetRegister: 2) : EndProgram(32),
            EndProgram(40));
        var fetch = EmbeddedVertexFetchDetector.Detect(program, AttributeTable, BufferTable, 0, 8, 32);
        Assert.Single(fetch.Loads);
        Assert.True(Extract(program, userDataCount: 8, stage: ShaderStage.Vertex).Info.UsesDeviceAddresses);

        var prepared = fetch.RemoveReplacedTableLoads(program);
        var plan = ShaderResourcePlan.Extract(prepared, ShaderStage.Vertex, Hash, 0, 8,
            fetch.Loads.Select(load => load.Pc).ToHashSet());
        Assert.Equal(additionalRead, plan.Info.UsesDeviceAddresses);
        Assert.Empty(plan.Info.Buffers);
        Assert.Empty(plan.TableReads);
        Assert.Equal("SNop", prepared.Instructions[0].Opcode);
        Assert.Equal("SNop", prepared.Instructions[2].Opcode);
        Assert.Null(plan.Memory.Find(24));
        Assert.Equal(program.Instructions.Select(instruction => instruction.Pc), prepared.Instructions.Select(instruction => instruction.Pc));

        var resources = ResourceMaterializer.ApplyTo(plan, ResourceSpecialization.Default(plan.Info));
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(prepared, 0, 8), false, false, false, 0);
        var request = new ShaderCompileRequest(plan, resources, layout)
        {
            VertexInputs = [new ShaderVertexInput(24, 0, 4, 7, false, [])],
        };
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out var spirv, out var spirvError), spirvError);
        Assert.Equal(additionalRead ? 5348u : 0u, new SpirvModuleInspector(spirv.Spirv).AddressingModel);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var metal, out var metalError), metalError);
        Assert.DoesNotContain("s[12] = sharpemu_load_device_dword", metal.Source);
    }

    [Fact]
    public void BufferTableWalk_YieldsOneLoadPerAttributeBuffer()
    {
        var program = Program(
            Vop2(0, "VAddI32", 0, Gen5Operand.Scalar(16), Gen5Operand.Vector(0)),
            Vop2(4, "VAddI32", 3, Gen5Operand.Scalar(17), Gen5Operand.Vector(3)),
            ScalarLoad(8, BufferTable, destination: 12, count: 4, immediateOffset: 16),
            IndexSelect(16),
            FetchLoad(20, 12, 3),
            EndProgram(28));

        var plan = EmbeddedVertexFetchDetector.Detect(program, AttributeTable, BufferTable, userDataBase: 0, userDataCount: 20, waveSize: 32);

        var load = Assert.Single(plan.Loads);
        Assert.Equal(20u, load.Pc);
        Assert.Equal(1, load.AttributeId);
        Assert.Equal(3u, load.Components);
        Assert.Equal([8u], load.PrologLoads);
        Assert.Equal(16, plan.VertexOffsetScalarRegister);
        Assert.Equal(17, plan.InstanceOffsetScalarRegister);
    }

    [Fact]
    public void AttributeTableWalk_CarriesTheAttributeIdThroughTheBufferLoad()
    {
        var program = Program(
            ScalarLoad(0, AttributeTable, destination: 8, count: 4, immediateOffset: 8),
            Sop2(8, "SLshlB32", 9, Gen5Operand.Scalar(10), Operand(4)),
            ScalarLoad(12, BufferTable, destination: 12, count: 4, dynamicOffsetRegister: 9),
            IndexSelect(20),
            FetchLoad(24, 12, 4),
            EndProgram(32));

        var plan = EmbeddedVertexFetchDetector.Detect(program, AttributeTable, BufferTable, userDataBase: 0, userDataCount: 8, waveSize: 32);

        var load = Assert.Single(plan.Loads);
        Assert.Equal(4, load.AttributeId);
        Assert.Equal([0u, 12u], load.PrologLoads);
        Assert.Equal(ShaderResourceInfo.NoScalarRegister, plan.VertexOffsetScalarRegister);
    }

    [Fact]
    public void ConflictingOffsetRegisters_LeaveTheOffsetUnresolved()
    {
        var program = Program(
            Vop2(0, "VAddI32", 0, Gen5Operand.Scalar(16), Gen5Operand.Vector(0)),
            Vop2(4, "VAddI32", 0, Gen5Operand.Scalar(17), Gen5Operand.Vector(0)),
            ScalarLoad(8, BufferTable, destination: 12, count: 4),
            IndexSelect(16),
            FetchLoad(20, 12, 1),
            EndProgram(28));

        var plan = EmbeddedVertexFetchDetector.Detect(program, AttributeTable, BufferTable, userDataBase: 0, userDataCount: 20, waveSize: 32);

        Assert.Single(plan.Loads);
        Assert.Equal(ShaderResourceInfo.NoScalarRegister, plan.VertexOffsetScalarRegister);
    }

    [Fact]
    public void ScalarOffsetAccumulatedThroughSad_UsesTheUserDataAtEightLayout()
    {
        var program = Program(
            Vop3(0, "VSadU32", 5, Gen5Operand.Scalar(10), Operand(0), Gen5Operand.Vector(5)),
            ScalarLoad(8, BufferTable, destination: 12, count: 4),
            IndexSelect(16),
            FetchLoad(20, 12, 2),
            EndProgram(28));

        var plan = EmbeddedVertexFetchDetector.Detect(program, AttributeTable, BufferTable, userDataBase: 8, userDataCount: 16, waveSize: 32);

        Assert.Single(plan.Loads);
        Assert.Equal(10, plan.VertexOffsetScalarRegister);
    }

    [Fact]
    public void LaneSavedTableRecord_SurvivesTheRoundTrip()
    {
        var program = Program(
            ScalarLoad(0, BufferTable, destination: 12, count: 4),
            WriteLane(8, vectorRegister: 20, scalarRegister: 12, lane: 3),
            MoveScalar(16, 12, 0),
            ReadLane(20, scalarRegister: 12, vectorRegister: 20, lane: 3),
            IndexSelect(28),
            FetchLoad(32, 12, 4),
            EndProgram(40));

        var plan = EmbeddedVertexFetchDetector.Detect(program, AttributeTable, BufferTable, userDataBase: 0, userDataCount: 8, waveSize: 32);

        var load = Assert.Single(plan.Loads);
        Assert.Equal(0, load.AttributeId);
    }

    [Fact]
    public void LoadWithoutTheIndexSelect_IsNotAFetch()
    {
        var program = Program(
            ScalarLoad(0, BufferTable, destination: 12, count: 4),
            FetchLoad(8, 12, 4),
            EndProgram(16));

        var plan = EmbeddedVertexFetchDetector.Detect(program, AttributeTable, BufferTable, userDataBase: 0, userDataCount: 8, waveSize: 32);

        Assert.Empty(plan.Loads);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LaneSavedTableRecord_RequiresSaveAndRestoreInTheSameBlock(bool branchSkipsSave)
    {
        var program = Program(
            ScalarLoad(0, BufferTable, destination: 12, count: 4),
            Branch(8, "SCbranchScc0", branchSkipsSave ? (short)3 : (short)0),
            WriteLane(12, vectorRegister: 20, scalarRegister: 12, lane: 3),
            MoveScalar(20, 12, 0),
            ReadLane(24, scalarRegister: 12, vectorRegister: 20, lane: 3),
            IndexSelect(32),
            FetchLoad(36, 12, 4),
            EndProgram(44));

        var plan = EmbeddedVertexFetchDetector.Detect(program, AttributeTable, BufferTable, 0, 8, 32);

        if (branchSkipsSave)
        {
            Assert.Empty(plan.Loads);
        }
        else
        {
            Assert.Equal(0, Assert.Single(plan.Loads).AttributeId);
        }
    }

    [Theory]
    [InlineData("SSetpcB64", false)]
    [InlineData("SSwappcB64", false)]
    [InlineData("SRfeB64", false)]
    [InlineData("SGetpcB64", true)]
    public void UnresolvedTransfersPreventFetchReplacementButGetpcDoesNot(
        string opcode, bool shouldTrackSavedLane)
    {
        var program = Program(
            ScalarLoad(0, BufferTable, destination: 12, count: 4),
            WriteLane(8, vectorRegister: 20, scalarRegister: 12, lane: 3),
            MoveScalar(16, 12, 0),
            new Gen5ShaderInstruction(20, Gen5ShaderEncoding.Sop1, opcode, [0u],
                [Gen5Operand.Scalar(0)], [], null),
            ReadLane(24, scalarRegister: 12, vectorRegister: 20, lane: 3),
            IndexSelect(32),
            FetchLoad(36, 12, 4),
            ScalarLoad(44, BufferTable, destination: 16, count: 4),
            IndexSelect(52),
            FetchLoad(56, 16, 4),
            EndProgram(64));

        var plan = EmbeddedVertexFetchDetector.Detect(program, AttributeTable, BufferTable, 0, 8, 32);

        uint[] expectedFetchAddresses = shouldTrackSavedLane ? [36u, 56u] : [];
        Assert.Equal(expectedFetchAddresses, plan.Loads.Select(load => load.Pc).ToArray());
    }

    [Theory]
    [InlineData("SCbranchScc0", false)]
    [InlineData("SCbranchExecz", false)]
    [InlineData("SCbranchScc0", true)]
    public void JoinRequiresTheSameScalarOnBothPaths(string branch, bool sameValue)
    {
        var program = Program(
            ScalarLoad(0, BufferTable, destination: 12, count: 4),
            Branch(8, branch, 2),
            sameValue ? MoveScalarRegister(12, 12, 12) : MoveScalar(12, 12, 0),
            Branch(16, "SBranch", 0),
            WriteLane(20, vectorRegister: 20, scalarRegister: 12, lane: 3),
            MoveScalar(28, 12, 0),
            ReadLane(32, scalarRegister: 12, vectorRegister: 20, lane: 3),
            IndexSelect(40), FetchLoad(44, 12, 4), EndProgram(52));
        var plan = EmbeddedVertexFetchDetector.Detect(program, AttributeTable, BufferTable, 0, 8, 32);
        Assert.Equal(sameValue ? 1 : 0, plan.Loads.Count);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SavedLaneAcrossLoopRequiresAnUnchangedBackEdge(bool clobber)
    {
        var program = Program(
            ScalarLoad(0, BufferTable, destination: 12, count: 4),
            WriteLane(8, vectorRegister: 20, scalarRegister: 12, lane: 3),
            ReadLane(16, scalarRegister: 12, vectorRegister: 20, lane: 3),
            IndexSelect(24), FetchLoad(28, 12, 4),
            MoveScalar(36, 12, 0),
            clobber ? WriteLane(40, vectorRegister: 20, scalarRegister: 12, lane: 3)
                : new Gen5ShaderInstruction(40, Gen5ShaderEncoding.Sopp, "SNop", [0u, 0u], [], [], null),
            Branch(48, "SCbranchScc1", -9), EndProgram(52));
        var plan = EmbeddedVertexFetchDetector.Detect(program, AttributeTable, BufferTable, 0, 8, 32);
        Assert.Equal(clobber ? 0 : 1, plan.Loads.Count);
    }

    [Fact]
    public void UnreachableTableLoadsCannotSupplyAFetch()
    {
        var program = Program(Branch(0, "SBranch", 2),
            ScalarLoad(4, BufferTable, destination: 12, count: 4),
            IndexSelect(12), FetchLoad(16, 12, 4), EndProgram(24));
        Assert.Empty(EmbeddedVertexFetchDetector.Detect(program, AttributeTable, BufferTable, 0, 8, 32).Loads);
    }

    [Fact]
    public void VectorIndexMustBeValidOnEveryIncomingPath()
    {
        var program = Program(ScalarLoad(0, BufferTable, destination: 12, count: 4),
            IndexSelect(8), Branch(12, "SCbranchScc0", 1),
            Vop2(16, "VAddI32", 1, Operand(0), Operand(1)),
            FetchLoad(20, 12, 4), EndProgram(28));
        Assert.Empty(EmbeddedVertexFetchDetector.Detect(program, AttributeTable, BufferTable, 0, 8, 32).Loads);
    }
}
