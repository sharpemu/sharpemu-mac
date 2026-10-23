// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Metal;
using SharpEmu.ShaderCompiler.Resources;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

public sealed class DirectImageTableTests
{
    [Theory]
    [InlineData(9u, 0u, true)]
    [InlineData(13u, 0x00020003u, true)]
    [InlineData(13u, 0x00040003u, false)]
    public void DenseCandidateTracePreservesSourceAddressesAndRawWords(uint type, uint sliceWord, bool valid)
    {
        uint[] descriptor = [0x017D5C00, 0xC4700000, 0x010DC1DF,
            0x01B00FAC | (type << 28), sliceWord, 0x00500004, 0x00000400, 0x00005204];
        bool ReadCandidate(ulong address, out uint word)
        {
            word = 0;
            const ulong start = 0x1000 + 344;
            if (address < start || address >= start + 32 * 32)
                return false;
            word = descriptor[(int)((address - start) % 32 / 4)];
            return true;
        }

        var plan = ShaderResourcePlan.Extract(CreateGuardedProgram(), ShaderStage.Compute, Hash, 0, 2);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        var observedKeys = new List<uint>();
        var inputs = new ResourceRuntimeInputs
        {
            UserData = [0x1000, 0],
            ReadCleanMemory = ReadCandidate,
            TraceImageCandidate = (source, key, heap, address, words) =>
            {
                Assert.Equal(plan.Info.Images[0].Source, source);
                Assert.Equal(0x1000ul, heap);
                Assert.Equal(heap + 344 + key * 32, address);
                Assert.Equal(descriptor, words.ToArray());
                observedKeys.Add(key);
            },
        };
        Assert.True(ResourceMaterializer.Materialize(plan,
            inputs.WithReader(ReadCandidate), ref snapshot, ref specialization));
        Assert.Equal(Enumerable.Range(0, 32).Select(key => (uint)key), observedKeys);
        Assert.Equal(valid ? descriptor : new uint[8], Assert.Single(snapshot.Images));
    }

    internal static Gen5ShaderProgram CreateWaveIndexedDescriptorProgram()
    {
        return Program(
            ScalarLoad(0, 0, 16, immediateOffset: 0x80),
            Sop1(8, "SFF1I32B32", 18, Gen5Operand.Scalar(16)),
            Sop2(12, "SMulI32", 106, Gen5Operand.Scalar(18), new Gen5Operand(Gen5OperandKind.LiteralConstant, 0x90)),
            MoveVectorFromScalar(16, 34, 18),
            Vop2(20, "VLshlrevB32", 15, Operand(4), Gen5Operand.Vector(34)),
            Vop3(24, "VLshlAddU32", 16, Gen5Operand.Vector(15), Operand(3), Gen5Operand.Vector(15)),
            Sop1(32, "SBitset0B32", 16, Gen5Operand.Scalar(18)),
            Vop2(36, "VAddI32", 15, new Gen5Operand(Gen5OperandKind.LiteralConstant, 0x40), Gen5Operand.Vector(16)),
            GlobalMemory(40, "GlobalLoadDword", 0, 15, 22, 0),
            ReadFirstLane(48, 106, 22),
            Sop2(52, "SLshlB32", 106, Gen5Operand.Scalar(106), Operand(5)),
            ScalarLoad(56, 0, 4, 8, immediateOffset: 0x100, dynamicOffsetRegister: 106),
            Image(64, "ImageLoad", 4, dmask: 1, vectorAddress: 1),
            EndProgram(72));
    }

    internal static Gen5ShaderProgram CreateWaveIndexedReadLaneProgram(bool selfAddressed, bool restoreExec = true)
    {
        var instructions = new List<Gen5ShaderInstruction>
        {
            ScalarLoad(0, 0, 16, immediateOffset: 0x80),
            Sop1(8, "SFF1I32B32", 18, Gen5Operand.Scalar(16)),
            Sop2(12, "SMulI32", 19, Gen5Operand.Scalar(18), new Gen5Operand(Gen5OperandKind.LiteralConstant, 0x90)),
            MoveVectorFromScalar(16, 34, 18),
            Vop2(20, "VLshlrevB32", 15, Operand(4), Gen5Operand.Vector(34)),
            Vop3(24, "VLshlAddU32", selfAddressed ? 15u : 16u, Gen5Operand.Vector(15), Operand(3), Gen5Operand.Vector(15)),
            Sop2(32, "SLshlB32", 20, Operand(1), Gen5Operand.Scalar(18)),
            Sop2(36, "SXorB32", 16, Gen5Operand.Scalar(20), Gen5Operand.Scalar(16)),
            Vop2(40, "VAddI32", 15, new Gen5Operand(Gen5OperandKind.LiteralConstant, 0x40), Gen5Operand.Vector(selfAddressed ? 15u : 16u)),
            GlobalMemory(44, "GlobalLoadDword", 0, 15, 22, 0),
            Sop1(52, "SMovB64", 12, Gen5Operand.Scalar(126)),
            Sop1(56, "SFF1I32B64", 24, Gen5Operand.Scalar(12)),
            new(60, Gen5ShaderEncoding.Vop3, "VReadlaneB32", [0u, 0u],
                [Gen5Operand.Vector(22), Gen5Operand.Scalar(24), Gen5Operand.Scalar(24)], [Gen5Operand.Scalar(106)], null),
            new(68, Gen5ShaderEncoding.Vop3, "VCmpEqU32", [0u, 0u],
                [Gen5Operand.Scalar(106), Gen5Operand.Vector(22)], [Gen5Operand.Scalar(14)], new Gen5Vop3Control(0, 0, 0, false, 0, 14)),
            Sop1(76, "SAndSaveexecB64", 28, Gen5Operand.Scalar(14)),
            Branch(80, "SCbranchExecz", 8),
            Sop2(84, "SLshlB32", 106, Gen5Operand.Scalar(106), Operand(5)),
            Sop2(88, "SAddI32", 107, Gen5Operand.Scalar(106), Operand(16)),
            ScalarLoad(92, 0, 4, 4, immediateOffset: 0x100, dynamicOffsetRegister: 106),
            ScalarLoad(100, 0, 8, 4, immediateOffset: 0x100, dynamicOffsetRegister: 107),
            Image(108, "ImageLoad", 4, dmask: 1, vectorAddress: 1),
            Sop2(116, "SAndn2B64", 12, Gen5Operand.Scalar(12), Gen5Operand.Scalar(14)),
            restoreExec ? Sop1(120, "SMovB64", 126, Gen5Operand.Scalar(28)) : Nop(120),
            Branch(124, "SCbranchScc1", -18),
            EndProgram(128),
        };
        return Program([.. instructions]);
    }

    public static Gen5ShaderProgram CreateProgram(uint mask = 1, bool split = true, bool bitScan = true)
    {
        var instructions = new List<Gen5ShaderInstruction>
        {
            Vop2(0, "VAddU32", 12, Operand(mask), Gen5Operand.Vector(0)),
            ReadFirstLane(4, 18, 12),
            Sop1(8, bitScan ? "SFF1I32B32" : "SMovB32", 19, Gen5Operand.Scalar(18)),
            Sop2(12, "SLshlB32", 106, Gen5Operand.Scalar(19), Operand(5)),
            new(16, Gen5ShaderEncoding.Sopk, "SAddkI32", [0xB7EA0158], [new(Gen5OperandKind.LiteralConstant, 344)], [Gen5Operand.Scalar(106)], null),
            Sop2(20, "SAddI32", 107, Gen5Operand.Scalar(106), Operand(16)),
            ScalarLoad(24, 0, 4, split ? 4u : 8u, dynamicOffsetRegister: 106),
        };
        if (split) instructions.Add(ScalarLoad(32, 0, 8, 4, dynamicOffsetRegister: 107));
        instructions.AddRange([
            MoveScalar(40, 106, 999),
            MoveVector(44, 1, 0), MoveVector(48, 2, 0),
            Image(52, "ImageLoad", 4, dmask: 1, vectorAddress: 1),
            MoveScalar(60, 20, 0), MoveScalar(64, 21, 0), MoveScalar(68, 22, 64), MoveScalar(72, 23, 0),
            BufferAccess(76, "BufferStoreDword", 20, vectorData: 4), EndProgram(84),
        ]);
        return Program([.. instructions]);
    }

    public static bool ReadDescriptor(ulong address, out uint word)
    {
        word = 0;
        if (address < 0x1000 + 312 || address >= 0x1000 + 344 + 32 * 32) return false;
        var offset = address - 0x1000 - 312;
        var record = offset / 32;
        var component = offset % 32 / 4;
        word = component switch
        {
            0 => (record & 1) == 0 ? 0x2000u : 0x1000u,
            1 => 20u << 20,
            3 => 0xFACu | (9u << 28),
            _ => 0,
        };
        return true;
    }

    public static Gen5ShaderProgram CreateGuardedProgram(uint mask = 1, bool split = true)
    {
        var program = CreateProgram(mask, split);
        return program with
        {
            Instructions = program.Instructions.Where(instruction => instruction.Pc < 8)
                .Concat([
                    Sopc(8, "SCmpLgU32", Operand(0), Gen5Operand.Scalar(18)),
                    Branch(12, "SCbranchScc0", 19),
                ])
                .Concat(program.Instructions.Where(instruction => instruction.Pc >= 8)
                    .Select(instruction => instruction with { Pc = instruction.Pc + 8 })).ToArray(),
        };
    }

    public static (ShaderResourcePlan Plan, ResourceSnapshot Snapshot, ShaderCompileRequest Request) PrepareDirect(uint mask = 1, bool split = true, bool guarded = false)
    {
        var program = guarded ? CreateGuardedProgram(mask, split) : CreateProgram(mask, split);
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: ReadDescriptor), ref snapshot, ref specialization));
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(program, 0, 2),
            false, ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false);
        return (plan, snapshot, new ShaderCompileRequest(plan, resources, layout) { LocalSizeX = 1, ThreadCountX = 1 });
    }

    [Fact]
    public void WaveIndexedDescriptorTableMaterializesOnlyActiveMaskKeys()
    {
        var plan = ShaderResourcePlan.Extract(CreateWaveIndexedDescriptorProgram(), ShaderStage.Compute, Hash, 0, 2);
        var selector = plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage!;
        Assert.True(selector.Dense);
        Assert.Equal(0u, selector.KeyBound);
        Assert.Equal(new WaveIndexedImageSelector(0x80, 0x40, 0x90), selector.WaveIndexed);

        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: ReadWaveIndexedMemory), ref snapshot, ref specialization));
        Assert.Equal(2, snapshot.Images.Length);
    }

    [Fact]
    public void CandidateWithAnUnsupportedFormatIsMaterializedAsNull()
    {
        var plan = ShaderResourcePlan.Extract(CreateWaveIndexedDescriptorProgram(), ShaderStage.Compute, Hash, 0, 2);
        bool Read(ulong address, out uint word)
        {
            var success = ReadWaveIndexedMemory(address, out word);
            if (address == 0x1000 + 0x100 + 5 * 32 + 4) word = 139u << 20;
            return success;
        }

        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: Read), ref snapshot, ref specialization, out var failure), $"materialize {failure}");
        Assert.Equal(2, snapshot.Images.Length);
        Assert.Single(snapshot.Images, image => image.All(word => word == 0));
    }

    // Word 2 bit 31 is RESOURCE_LEVEL, a defined field, not a reserved bit: a candidate with
    // it set (Demon's Souls' reflection-probe cubes) stays a real descriptor.
    [Fact]
    public void CandidateWithTheResourceLevelBitIsKept()
    {
        var plan = ShaderResourcePlan.Extract(CreateWaveIndexedDescriptorProgram(), ShaderStage.Compute, Hash, 0, 2);
        bool Read(ulong address, out uint word)
        {
            var success = ReadWaveIndexedMemory(address, out word);
            if (address == 0x1000 + 0x100 + 5 * 32 + 8) word = 0x8000_0000u;
            return success;
        }

        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: Read), ref snapshot, ref specialization, out var failure), $"materialize {failure}");
        Assert.Equal(2, snapshot.Images.Length);
        Assert.DoesNotContain(snapshot.Images, image => image.All(word => word == 0));
        Assert.Single(snapshot.Images, image => image[2] == 0x8000_0000u);
    }

    // A DIM_CUBE access through a table holding a cube and a 2D placeholder: the placeholder
    // cannot be read as a cube, so it binds null instead of failing the specialization.
    [Fact]
    public void CubeAccessNullsANonCubeCandidate()
    {
        var program = CreateWaveIndexedDescriptorProgram();
        program = program with
        {
            Instructions = program.Instructions.Select(instruction => instruction.Pc == 64
                ? Image(64, "ImageLoad", 4, dmask: 1, vectorAddress: 1, dimension: 3)
                : instruction).ToArray(),
        };
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        Assert.True(plan.Info.Images[0].CubeAccess);
        bool Read(ulong address, out uint word)
        {
            var success = ReadWaveIndexedMemory(address, out word);
            if (address == 0x1000 + 0x100 + 2 * 32 + 12) word = 0xFACu | (11u << 28);
            if (address == 0x1000 + 0x100 + 2 * 32 + 16) word = 5;
            return success;
        }

        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: Read), ref snapshot, ref specialization, out var failure), $"materialize {failure}");
        Assert.Equal(2, snapshot.Images.Length);
        Assert.Single(snapshot.Images, image => image.All(word => word == 0));
        Assert.Single(snapshot.Images, image => image[3] >> 28 == 11);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ReadLaneWaterfallSelectsWaveIndexedDescriptors(bool selfAddressed)
    {
        var program = CreateWaveIndexedReadLaneProgram(selfAddressed);
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        var selector = plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage!;
        Assert.True(selector.Dense, "dense");
        Assert.Equal(0x100u, selector.TableOffset);
        Assert.Equal(new WaveIndexedImageSelector(0x80, 0x40, 0x90), selector.WaveIndexed);
        Assert.True(Assert.Single(plan.IndirectImages).KeyIsAddressOffset, "address-offset key");

        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: ReadWaveIndexedMemory), ref snapshot, ref specialization, out var failure), $"materialize {failure}");
        Assert.Equal(2, snapshot.Images.Length);
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(program, 0, 2),
            false, ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false);
        var request = new ShaderCompileRequest(plan, resources, layout) { LocalSizeX = 64, ThreadCountX = 64 };
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var error), error);
    }

    [Fact]
    public void ReadLaneWithoutRestoredExecutionIsNotWaveIndexed()
    {
        Assert.Throws<ResourcePlanException>(() =>
            ShaderResourcePlan.Extract(CreateWaveIndexedReadLaneProgram(selfAddressed: true, restoreExec: false), ShaderStage.Compute, Hash, 0, 2));
    }

    private static bool ReadWaveIndexedMemory(ulong address, out uint word)
    {
        word = 0;
        if (address == 0x1000 + 0x80)
        {
            word = (1u << 1) | (1u << 4);
            return true;
        }

        if (address == 0x1000 + 0x40 + 0x90 || address == 0x1000 + 0x40 + 4 * 0x90)
        {
            word = address == 0x1000 + 0x40 + 0x90 ? 2u : 5u;
            return true;
        }

        if (address < 0x1000 + 0x100 || address >= 0x1000 + 0x100 + 6 * 32)
            return false;
        var relative = address - 0x1000 - 0x100;
        var record = relative / 32;
        if (record is not (2 or 5))
            return false;
        word = (relative % 32 / 4) switch
        {
            0 => (record & 1) == 0 ? 0x2000u : 0x1000u,
            1 => 20u << 20,
            3 => 0xFACu | (9u << 28),
            _ => 0,
        };
        return true;
    }

    public static (ResourceSnapshot Snapshot, ShaderCompileRequest Request) PrepareMixedDimensions(uint mask, bool arrayFirst)
    {
        var program = CreateGuardedProgram(mask);
        program = program with
        {
            Instructions = program.Instructions.Select(instruction => instruction.Pc switch
            {
                48 => MoveVector(48, 3, 1),
                60 => Image(60, "ImageLoad", 4, dimension: 5, dmask: 1, vectorAddress: 1),
                _ => instruction,
            }).ToArray(),
        };
        bool Read(ulong address, out uint word)
        {
            word = 0;
            if (address < 0x1000 + 344 || address >= 0x1000 + 344 + 32 * 32) return false;
            var offset = address - 0x1000 - 344;
            var record = offset / 32;
            if (record is not (0 or 31)) return true;
            var array = (record == 0) == arrayFirst;
            word = (offset % 32 / 4) switch
            {
                0 => array ? 0x1000u : 0x2000u,
                1 => 20u << 20,
                3 => 0xFACu | ((array ? 13u : 9u) << 28),
                4 => array ? 1u : 0u,
                _ => 0,
            };
            return true;
        }
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: Read), ref snapshot, ref specialization));
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(program, 0, 2),
            false, ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false);
        return (snapshot, new ShaderCompileRequest(plan, resources, layout) { LocalSizeX = 1, ThreadCountX = 1 });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MixedDimensionsRetainEveryCandidateAndUseSeparateBindings(bool arrayFirst)
    {
        var (snapshot, request) = PrepareMixedDimensions(1, arrayFirst);
        Assert.Equal(3, snapshot.Images.Length);
        var images = request.Resources.Info.Images;
        Assert.Equal(3, images[0].IndirectResources.Count);
        Assert.Contains(images, image => image.Dimension == ImageDimension.Dim2D);
        Assert.Contains(images, image => image.Dimension == ImageDimension.Dim2DArray);
        var mapping = (int)images[0].IndirectMappingOffset;
        Assert.Equal(32u, snapshot.FlattenedResourceTable[mapping]);
        Assert.Equal(344u + 31 * 32, snapshot.FlattenedResourceTable[mapping + 63]);
        Assert.Equal(2u, snapshot.FlattenedResourceTable[mapping + 64]);
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var error), error);

        // Both backends declare one image class per dimension, so a candidate list that
        // mixes 2D and 2D-array descriptors translates across the two binding classes.
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var metal, out error), error);
        Assert.Contains("_2d [[id(", metal.Source);
        Assert.Contains("_2d_array [[id(", metal.Source);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DirectTableRetainsAllBitScanResultsAndCompiles(bool split)
    {
        var (plan, snapshot, request) = PrepareDirect(split: split);
        Assert.Single(plan.IndirectImages);
        Assert.True(plan.IndirectImages[0].KeyIsAddressOffset);
        Assert.False(plan.Info.UsesDeviceAddresses);
        Assert.Empty(plan.DynamicReads);
        var selector = plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage!;
        Assert.Equal(33, selector.DirectCandidates!.Count);
        Assert.Contains(selector.DirectCandidates, candidate => candidate.Offset == 312);
        Assert.Contains(selector.DirectCandidates, candidate => candidate.Offset == 344 + 31 * 32);
        Assert.Equal(2, snapshot.Images.Length);
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var spirvError), spirvError);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out _, out var metalError), metalError);
    }

    [Theory]
    [InlineData(32)]
    [InlineData(33)]
    public void DirectImageCapacityPreservesTheLimitAndPublishedState(int distinctCount)
    {
        var (plan, snapshot, _) = PrepareDirect();
        var specialization = new ResourceSpecialization();
        var previousSnapshot = snapshot;
        var previousSpecialization = specialization;
        bool Read(ulong address, out uint word)
        {
            var success = ReadDescriptor(address, out word);
            if (success && (address - 0x1000 - 312) % 32 == 0)
                word = 0x2000 + (uint)(((address - 0x1000 - 312) / 32) % (ulong)distinctCount);
            return success;
        }

        var success = ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: Read),
            ref snapshot, ref specialization, out var failure);
        Assert.Equal(distinctCount <= ShaderResourceInfo.MaxIndirectImageCandidates, success);
        if (success)
        {
            Assert.Equal(ResourceMaterializationFailure.None, failure);
            Assert.Equal(distinctCount, snapshot.Images.Length);
        }
        else
        {
            Assert.Equal(ResourceMaterializationFailure.ImageCapacityExceeded, failure);
            Assert.Same(previousSnapshot, snapshot);
            Assert.Same(previousSpecialization, specialization);
        }
    }

    [Fact]
    public void UnboundedDirectTableIsStillRejected()
    {
        Assert.Throws<ResourcePlanException>(() => ShaderResourcePlan.Extract(CreateProgram(bitScan: false), ShaderStage.Compute, Hash, 0, 2));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void NonZeroGuardExcludesTheUnreachableDescriptor(bool split, bool equality)
    {
        var program = CreateGuardedProgram(split: split);
        if (equality)
            program = program with
            {
                Instructions = program.Instructions.Select(instruction => instruction.Pc switch
                {
                    8 => Sopc(8, "SCmpEqI32", Gen5Operand.Scalar(18), Operand(0)),
                    12 => Branch(12, "SCbranchScc1", 19),
                    _ => instruction,
                }).ToArray(),
            };
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        var selector = plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage!;
        Assert.True(selector.Dense);
        Assert.Equal(32u, selector.KeyBound);
        Assert.Equal(344u, selector.TableOffset);
        Assert.True(Assert.Single(plan.IndirectImages).KeyIsAddressOffset);
        bool Read(ulong address, out uint word)
        {
            Assert.True(address >= 0x1000 + 344);
            return ReadDescriptor(address, out word);
        }
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: Read), ref snapshot, ref specialization));
        var (_, _, request) = PrepareDirect(split: split, guarded: true);
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var spirvError), spirvError);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out _, out var metalError), metalError);
    }

    [Theory]
    [InlineData(600u, 8u)]
    [InlineData(444u, 3u)]
    [InlineData(312u, 32u)]
    [InlineData(1400u, 32u)]
    public void DenseTableEndsBeforeASamplerLoadedFromTheSameBase(uint samplerOffset, uint expectedBound)
    {
        var program = CreateGuardedProgram();
        program = program with
        {
            Instructions = program.Instructions.Where(instruction => instruction.Pc != 52).Select(instruction => instruction.Pc switch
            {
                48 => ScalarLoad(48, 0, 12, 4, immediateOffset: (int)samplerOffset),
                60 => Image(60, "ImageSample", 4, samplerRegister: 12, dmask: 1, vectorAddress: 1),
                _ => instruction,
            }).ToArray(),
        };
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        var selector = plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage!;
        Assert.True(selector.Dense);
        Assert.Equal(344u, selector.TableOffset);
        Assert.Equal(expectedBound, selector.KeyBound);

        var highest = 0ul;
        bool Read(ulong address, out uint word)
        {
            if (address >= 0x1000 + samplerOffset && address < 0x1000 + samplerOffset + 16)
            {
                word = 0;
                return true;
            }

            if (address >= 0x1000 + 344) highest = Math.Max(highest, address);
            return ReadDescriptor(address, out word);
        }
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], Read, Read), ref snapshot, ref specialization, out var failure), $"materialize {failure}");
        Assert.True(highest < 0x1000 + 344 + expectedBound * 32, $"read 0x{highest:X} beyond the table");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LoopMustRecheckTheBitScanInput(bool bypassGuard)
    {
        var program = CreateGuardedProgram();
        program = program with
        {
            Instructions = program.Instructions.Where(instruction => instruction.Pc < 92)
                .Select(instruction => instruction.Pc == 12 ? Branch(12, "SCbranchScc0", 21) : instruction)
                .Concat([
                    ReadFirstLane(92, 18, 13),
                    Branch(96, "SBranch", bypassGuard ? (short)-21 : (short)-23),
                    EndProgram(100),
                ]).ToArray(),
        };
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        var selector = plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage!;
        if (bypassGuard)
        {
            Assert.Equal(33, selector.DirectCandidates!.Count);
            Assert.Contains(selector.DirectCandidates, candidate => candidate.Offset == 312);
        }
        else
        {
            Assert.True(selector.Dense);
            Assert.Equal(32u, selector.KeyBound);
            Assert.Equal(344u, selector.TableOffset);
        }
    }

    [Theory]
    [InlineData("opposite-branch")]
    [InlineData("different-register")]
    [InlineData("guard-bypass")]
    [InlineData("condition-write")]
    [InlineData("input-write")]
    public void UnprovenGuardRetainsTheZeroInputDescriptor(string variation)
    {
        var program = CreateGuardedProgram();
        var instructions = program.Instructions.ToList();
        switch (variation)
        {
            case "opposite-branch":
                instructions[instructions.FindIndex(instruction => instruction.Pc == 12)] = Branch(12, "SCbranchScc1", 19);
                break;
            case "different-register":
                instructions[instructions.FindIndex(instruction => instruction.Pc == 8)] =
                    Sopc(8, "SCmpLgU32", Operand(0), Gen5Operand.Scalar(17));
                break;
            case "guard-bypass":
                instructions[0] = Branch(0, "SCbranchScc1", 3);
                break;
            case "condition-write":
                instructions = instructions.Select(instruction => instruction.Pc >= 12
                    ? instruction with { Pc = instruction.Pc + 4 } : instruction).ToList();
                instructions.Add(Sopc(12, "SCmpEqU32", Operand(0), Operand(0)));
                break;
            case "input-write":
                instructions = instructions.Select(instruction => instruction.Pc >= 16
                    ? instruction with { Pc = instruction.Pc + 4 } : instruction).ToList();
                instructions[instructions.FindIndex(instruction => instruction.Pc == 12)] = Branch(12, "SCbranchScc0", 20);
                instructions.Add(ReadFirstLane(16, 18, 13));
                break;
        }
        program = program with { Instructions = instructions.OrderBy(instruction => instruction.Pc).ToArray() };
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        var candidates = plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage!.DirectCandidates!;
        Assert.Equal(33, candidates.Count);
        Assert.Contains(candidates, candidate => candidate.Offset == 312);
    }

    [Fact]
    public void DescriptorWordsUsedByArithmeticAreNotRemoved()
    {
        var program = CreateProgram();
        program = program with
        {
            Instructions = program.Instructions.Select(instruction => instruction.Pc == 40
                ? Vop1(40, "VMovB32", 15, Gen5Operand.Scalar(4)) : instruction).ToArray(),
        };
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        Assert.Single(plan.IndirectImages);
        Assert.True(plan.Info.UsesDeviceAddresses);
        Assert.True(plan.Memory.TryGetIndex(24, 0, out var memoryIndex));
        Assert.False(plan.Memory[memoryIndex].PlanningOnly);
    }

    [Fact]
    public void DirectTableSlotWithAnUnsupportedFormatIsMaterializedAsNull()
    {
        var program = CreateProgram();
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        var unsupported = true;
        bool Read(ulong address, out uint word)
        {
            var success = ReadDescriptor(address, out word);
            if (unsupported && address == 0x1000 + 344 + 3 * 32 + 4) word = 139u << 20;
            return success;
        }

        var clean = new ResourceSnapshot();
        var cleanSpecialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: ReadDescriptor), ref clean, ref cleanSpecialization));
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: Read), ref snapshot, ref specialization, out var failure), $"materialize {failure}");
        Assert.Equal(clean.Images.Length + 1, snapshot.Images.Length);
        Assert.Single(snapshot.Images, image => image.All(word => word == 0));
        unsupported = false;
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: Read), ref snapshot, ref specialization, out failure), $"materialize {failure}");
        Assert.Equal(clean.Images.Length, snapshot.Images.Length);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void FailedReadOrIncompatibleCandidateDoesNotPublish(bool incompatible, bool guarded)
    {
        var (plan, original, _) = PrepareDirect(guarded: guarded);
        var snapshot = original;
        var specialization = new ResourceSpecialization();
        var originalSpecialization = specialization;
        bool Read(ulong address, out uint word)
        {
            var success = ReadDescriptor(address, out word);
            if (address == 0x1000 + 344 + 12)
            {
                if (!incompatible) return false;
                word = (word & 0x0FFFFFFF) | (11u << 28);
            }
            return success;
        }
        Assert.False(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readCleanMemory: Read), ref snapshot, ref specialization,
            out var failure));
        Assert.Equal(incompatible ? ResourceMaterializationFailure.IncompatibleImageCandidates : ResourceMaterializationFailure.UnreadableGuestPointer, failure);
        Assert.Same(original, snapshot);
        Assert.Same(originalSpecialization, specialization);
    }

    // GTA V's ray-traced compute pass (0x9FBCE50E7D68468F) samples one texture per lane through a
    // waterfall loop: the lane's index is read first-lane, scaled by the T# size and used as the
    // dynamic offset of an S_BUFFER_LOAD_DWORDX8 through a V# that describes the descriptor table.
    // The index comes from vector memory, so only the V#'s record count bounds the table.
    internal static Gen5ShaderProgram CreateBufferHeapWaterfallProgram(
        string keyLoad = "BufferLoadDword", string imageOpcode = "ImageSampleL") => Program(
        ScalarLoad(0, 0, 8, 4, immediateOffset: 0x10),
        ScalarLoad(8, 0, 16, 4, immediateOffset: 0x20),
        BufferAccess(16, keyLoad, 16, vectorData: 2),
        Vop2(24, "VLshrrevB32", 11, Operand(4), Gen5Operand.Vector(2)),
        MoveScalar(28, 24, 0), MoveScalar(32, 25, 0xFFF000), MoveScalar(36, 26, 0x6500000), MoveScalar(40, 27, 0),
        Sop1(44, "SMovB64", 6, Gen5Operand.Scalar(126)),
        MoveVector(48, 19, 1),
        ReadFirstLane(52, 106, 11),
        Sop1(56, "SMovB64", 20, Gen5Operand.Scalar(126)),
        Vopc(60, "VCmpxEqU32", Gen5Operand.Scalar(106), 11),
        Sop2(64, "SLshlB32", 106, Gen5Operand.Scalar(106), Operand(5)),
        MoveVector(68, 19, 0),
        ScalarBufferLoad(72, 8, 12, 8, dynamicOffsetRegister: 106),
        Image(80, imageOpcode, 12, samplerRegister: 24, vectorAddress: 0),
        Sop1(88, "SMovB64", 126, Gen5Operand.Scalar(20)),
        Vopc(92, "VCmpxNeU32", Operand(0), 19),
        Branch(96, "SCbranchExecnz", -12),
        Sop1(100, "SMovB64", 126, Gen5Operand.Scalar(6)),
        EndProgram(104));

    // The descriptor table holds `records` bytes at 0x2000: entry 3 repeats entry 0, and a read
    // past the V#'s reach fails, so the host must not probe beyond it.
    private static GuestWordReader BufferHeapMemory(uint records) => (ulong address, out uint word) =>
    {
        word = 0;
        switch (address)
        {
            case 0x1010: word = 0x2000; return true;
            case 0x1014: return true;
            case 0x1018: word = records; return true;
            case 0x101C: return true;
            case 0x1020: word = 0x4000; return true;
            case 0x1024: return true;
            case 0x1028: word = 256; return true;
            case 0x102C: return true;
        }

        if (address < 0x2000 || address + 4 > 0x2000ul + records)
            return false;
        var relative = address - 0x2000;
        var entry = relative / 32 % 3;
        word = (relative % 32 / 4) switch
        {
            0 => 0x3000u + (uint)entry * 0x100u,
            1 => 20u << 20,
            3 => 0xFACu | (9u << 28),
            _ => 0,
        };
        return true;
    };

    [Fact]
    public void BufferHeapWaterfallReadsEveryEntryThroughAHandleTable()
    {
        var program = CreateBufferHeapWaterfallProgram();
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        var selector = plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage!;
        Assert.True(selector.Dense, "dense");
        Assert.True(selector.Bindless, "bindless");
        Assert.Equal(4u, plan.DescriptorSources[(int)selector.HeapSource].DwordCount);
        Assert.Equal(0u, selector.KeyBound);
        Assert.True(Assert.Single(plan.IndirectImages).KeyIsAddressOffset, "address-offset key");

        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        var memory = BufferHeapMemory(4 * 32);
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readMemory: memory, readCleanMemory: memory),
            ref snapshot, ref specialization, out var failure), $"materialize {failure}");
        // Entry 3 repeats entry 0, so three textures serve four entries; the root binds null.
        var table = Assert.Single(snapshot.BindlessImages);
        Assert.Equal([0, 1, 2, 0], table.Entries);
        Assert.Equal(3, table.Descriptors.Length);
        Assert.Equal([0u], table.VariantImages);
        Assert.Equal(0x2000ul, table.Address);
        Assert.All(table.DescriptorVariants, variant => Assert.Equal(0, variant));
        Assert.All(Assert.Single(snapshot.Images), word => Assert.Equal(0u, word));
        Assert.Equal(0u, Assert.Single(specialization.Images).BindlessSlot);

        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(program, 0, 2),
            false, ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false);
        Assert.Equal(1u, layout.BindlessImageTableCount);
        var request = new ShaderCompileRequest(plan, resources, layout) { LocalSizeX = 64, ThreadCountX = 64 };
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var metal, out var metalError), metalError);
        Assert.Contains("bindless_images[0u]", metal!.Source);
        // Only the Metal backend binds a handle table; SPIR-V reads the null root, as it did before.
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var spirvError), spirvError);
    }

    [Fact]
    public void BufferHeapWithoutRecordsReadsOnlyTheNullEntry()
    {
        var plan = ShaderResourcePlan.Extract(CreateBufferHeapWaterfallProgram(), ShaderStage.Compute, Hash, 0, 2);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        var memory = BufferHeapMemory(0);
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readMemory: memory, readCleanMemory: memory),
            ref snapshot, ref specialization, out var failure), $"materialize {failure}");
        var table = Assert.Single(snapshot.BindlessImages);
        Assert.Empty(table.Entries);
        Assert.Empty(table.Descriptors);
        Assert.All(snapshot.Images, image => Assert.All(image, word => Assert.Equal(0u, word)));
    }

    // A key loaded with a typed buffer load is the shape a dynamic descriptor used to resolve to
    // null (ResourceTracker.GetHandleSource). Stores and atomics through a V# table still do:
    // their plan, snapshot and both backends are what they were before sampled reads went bindless.
    [Fact]
    public void StorageThroughAVSharpTableKeepsTheNullDescriptor()
    {
        var program = CreateBufferHeapWaterfallProgram("BufferLoadFormatX", "ImageStore");
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        var image = Assert.Single(plan.Info.Images);
        Assert.Null(plan.DescriptorSources[(int)image.Source].IndirectImage);
        Assert.Empty(plan.IndirectImages);

        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        var memory = BufferHeapMemory(4 * 32);
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readMemory: memory, readCleanMemory: memory),
            ref snapshot, ref specialization, out var failure), $"materialize {failure}");
        Assert.Empty(snapshot.BindlessImages);
        Assert.All(Assert.Single(snapshot.Images), word => Assert.Equal(0u, word));

        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(program, 0, 2),
            false, ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false);
        Assert.Equal(0u, layout.BindlessImageTableCount);
        var request = new ShaderCompileRequest(plan, resources, layout) { LocalSizeX = 64, ThreadCountX = 64 };
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out _, out var metalError), metalError);
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out _, out var spirvError), spirvError);
    }

    // The same typed-load key under a sample used to read null; it now reads the table's textures.
    [Fact]
    public void SampleThroughAVSharpTableFormerlyNullReadsTheTable()
    {
        var program = CreateBufferHeapWaterfallProgram("BufferLoadFormatX");
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, Hash, 0, 2);
        Assert.True(plan.DescriptorSources[(int)plan.Info.Images[0].Source].IndirectImage!.Bindless, "bindless");

        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        var memory = BufferHeapMemory(4 * 32);
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readMemory: memory, readCleanMemory: memory),
            ref snapshot, ref specialization, out var failure), $"materialize {failure}");
        Assert.Equal([0, 1, 2, 0], Assert.Single(snapshot.BindlessImages).Entries);
    }

    // A V# reaching guest memory the host cannot read, or further than a device table holds,
    // reads null where it cannot be enumerated instead of failing the draw.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BufferHeapEntriesTheHostCannotReadAreNull(bool unbounded)
    {
        var plan = ShaderResourcePlan.Extract(CreateBufferHeapWaterfallProgram(), ShaderStage.Compute, Hash, 0, 2);
        var inner = BufferHeapMemory(4 * 32);
        GuestWordReader memory = (ulong address, out uint word) =>
        {
            if (address == 0x1018)
            {
                word = unbounded ? uint.MaxValue : 4 * 32;
                return true;
            }

            // Entry 1 lies on a page the guest has not mapped.
            if (address is >= 0x2020 and < 0x2040)
            {
                word = 0;
                return false;
            }

            return inner(address, out word);
        };

        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, Inputs([0x1000, 0], readMemory: memory, readCleanMemory: memory),
            ref snapshot, ref specialization, out var failure), $"materialize {failure}");
        var table = Assert.Single(snapshot.BindlessImages);
        if (unbounded)
        {
            Assert.Empty(table.Entries);
            Assert.Empty(table.Descriptors);
        }
        else
        {
            Assert.Equal([0, BindlessImageTable.NullEntry, 1, 0], table.Entries);
            Assert.Equal(2, table.Descriptors.Length);
        }
    }
}
