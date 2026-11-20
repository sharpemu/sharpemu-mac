// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

/// <summary>
/// The wave32 row reductions of Gen5MslTranslator.WaveReductions (ShaderCompileRequest.SubgroupArithmetic), as the
/// SPIR-V translator has them (WaveReductionDeviceTests in Libs.Tests): s80 copies the entry EXEC and drops lanes 19
/// and up, lanes outside s0 = EXEC &amp; s80 take the identity, a DPP row_shr 1/2/4/8 scan runs, and lanes 15 and/or 31
/// are read. On the PS5 every lane of the wave exists, so each read is the operation over its row's lanes in s0.
/// A simdgroup with fewer threads than the wave (a partial threadgroup, a fragment simdgroup holding one quad) has
/// rows with some or no running lane; the reads must still give the PS5's value.
/// </summary>
public sealed class Gen5MslWaveReductionTests(ITestOutputHelper output)
{
    public static readonly string[] Operations =
        ["VMaxU32", "VMinU32", "VMaxI32", "VMinI32", "VOrB32", "VAndB32", "VXorB32", "VAddU32"];

    public enum Reads { Both, Fifteen, ThirtyOne, BothReversed }

    private const uint Unread = 0xEEEE;
    private const uint TargetSize = 8;

    public static IEnumerable<object[]> OperationsAndThreads() =>
        from operation in Operations from threads in new uint[] { 32, 20, 12 } select new object[] { operation, threads };

    public static IEnumerable<object[]> OperationData() => Operations.Select(operation => new object[] { operation });

    // A threadgroup of fewer than 32 threads leaves lanes missing from the simdgroup: 20 threads leave row 1 with lanes
    // 16-19 (16-18 in the mask), 12 threads leave row 1 with none.
    [Theory]
    [MemberData(nameof(OperationsAndThreads))]
    public void Wave32RowScan_ReadsEachRowOverTheLanesThatExist(string operation, uint threads)
    {
        if (SkipWithoutMetalDevice("the wave reduction compute test")) return;

        var present = threads == 32 ? uint.MaxValue : (1u << (int)threads) - 1u;
        foreach (var reads in Enum.GetValues<Reads>())
        {
            var actual = RunCompute(operation, reads, threads, subgroupArithmetic: true);
            var (expected15, expected31) = Expected(operation, reads, present);
            for (var lane = 0; lane < threads; lane++)
            {
                Assert.True(expected15 == actual[lane].S1,
                    $"{operation} {reads} threads={threads} lane {lane}: lane 15 read 0x{actual[lane].S1:X8}, expected 0x{expected15:X8}");
                Assert.True(expected31 == actual[lane].S2,
                    $"{operation} {reads} threads={threads} lane {lane}: lane 31 read 0x{actual[lane].S2:X8}, expected 0x{expected31:X8}");
            }

            // A full simdgroup gives the same values without the reduction: the old lowering was right there.
            if (threads == 32)
            {
                Assert.Equal(actual, RunCompute(operation, reads, threads, subgroupArithmetic: false));
            }
        }
    }

    // A triangle covering only the center of pixel (1,1): its quad runs, the rest of the fragment simdgroup does not,
    // so row 1 has no running lane (the case a nearest-running-lane read of lane 31 got wrong: it read row 0's scan).
    [Theory]
    [MemberData(nameof(OperationData))]
    public void Wave32RowScan_InAPixelSimdgroupWithAnEmptyRow(string operation)
    {
        if (SkipWithoutMetalDevice("the wave reduction pixel test")) return;

        foreach (var reads in Enum.GetValues<Reads>())
        {
            var on = Texel(Render(operation, reads, subgroupArithmetic: true), 1, 1);
            var off = Texel(Render(operation, reads, subgroupArithmetic: false), 1, 1);
            var lane = on.B & 31u;
            var present = on.A;
            output.WriteLine($"{operation} {reads}: pixel (1,1) ran on lane {lane}, running lanes 0x{present:X8}");
            Assert.Equal(present, off.A);
            Assert.NotEqual(0u, present);
            // One quad: its lanes sit in one row, so the other row has no running lane.
            var emptyRow = (present & 0xFFFFu) == 0 ? 0 : 1;
            Assert.Equal(0u, present & (0xFFFFu << (16 * emptyRow)));

            var (expected15, expected31) = Expected(operation, reads, present);
            Assert.Equal(expected15, on.R);
            Assert.Equal(expected31, on.G);
            Identity(operation, out var identity);
            if (reads != (emptyRow == 0 ? Reads.ThirtyOne : Reads.Fifteen))
            {
                Assert.Equal(identity, emptyRow == 0 ? on.R : on.G);
            }

            if (emptyRow == 1)
            {
                // Row 0 has its running lanes at the bottom of the row: the old nearest-running-lane read of lane 15
                // held the row's scan and agrees; lane 31 read row 0's scan where the PS5 reads the identity.
                if (reads != Reads.ThirtyOne)
                {
                    Assert.Equal(on.R, off.R);
                }

                if (reads != Reads.Fifteen)
                {
                    Assert.Equal(Fold(operation, 0, present), off.G);
                }
            }
        }
    }

    // Every operation becomes its SIMD-group reduction; the signed ones compare as int.
    [Theory]
    [InlineData("VMaxU32", "simd_max(t", false, "0u")]
    [InlineData("VMinU32", "simd_min(t", false, "0xFFFFFFFFu")]
    [InlineData("VMaxI32", "simd_max(as_type<int>", true, "0x80000000u")]
    [InlineData("VMinI32", "simd_min(as_type<int>", true, "0x7FFFFFFFu")]
    [InlineData("VOrB32", "simd_or(t", false, "0u")]
    [InlineData("VAndB32", "simd_and(t", false, "0xFFFFFFFFu")]
    [InlineData("VXorB32", "simd_xor(t", false, "0u")]
    [InlineData("VAddU32", "simd_sum(t", false, "0u")]
    public void EachOperationIsItsSimdgroupReduction(string operation, string call, bool signed, string identity)
    {
        foreach (var stage in new[] { ShaderStage.Compute, ShaderStage.Pixel })
        {
            var source = Translate(Decode(Program(operation, Reads.Both, stage == ShaderStage.Pixel)), stage, true);
            Assert.Equal(2, Occurrences(source, call));
            Assert.Equal(signed ? 2 : 0, Occurrences(source, "as_type<uint>(simd_"));
            Assert.Contains($"select({identity}, ", source, StringComparison.Ordinal);
            Assert.Contains("(sharpemu_lane >> 4u) == 0u)", source, StringComparison.Ordinal);
            Assert.Contains("(sharpemu_lane >> 4u) == 1u)", source, StringComparison.Ordinal);
        }
    }

    // Only the exact scan becomes a reduction (the SPIR-V matcher's cases, Wave32RowScan_OnlyTheExactScanIsReduced).
    [Theory]
    [InlineData("Exact", 2)]
    [InlineData("NoSubgroupArithmetic", 0)]
    [InlineData("IdentityChanged", 0)]
    [InlineData("ShiftOrder", 0)]
    [InlineData("BoundControl", 0)]
    [InlineData("OtherLane", 0)]
    [InlineData("ExecNotSaved", 0)]
    [InlineData("ScanSourceChanged", 0)]
    [InlineData("MaskAllLanes", 0)]
    [InlineData("MaskNotFromEntryCopy", 0)]
    [InlineData("MaskInVcc", 0)]
    [InlineData("EntryCopyOverwritten", 0)]
    [InlineData("EntryCopyAfterExecWrite", 0)]
    [InlineData("EntryCopyInOddRegister", 2)]
    [InlineData("EntryCopyPairWrite", 0)]
    [InlineData("MaskInOddRegister", 2)]
    [InlineData("MaskPairWrite", 0)]
    [InlineData("SavedSourceKept", 2)]
    [InlineData("SavedSourcePairWrite", 0)]
    [InlineData("OneRead", 1)]
    [InlineData("SameRowTwice", 1)]
    [InlineData("ReadAfterOtherInstruction", 0)]
    [InlineData("WaitBeforeCndmask", 2)]
    public void Wave32RowScan_OnlyTheExactScanIsReduced(string variant, int reductions)
    {
        var instructions = Decode(Program("VMaxU32", Reads.Both, pixel: false)).Instructions.ToList();
        int Find(string opcode, int skip = 0) => instructions.FindIndex(skip, instruction => instruction.Opcode == opcode);
        if (variant is "EntryCopyInOddRegister" or "EntryCopyPairWrite")
        {
            Rename(instructions, 80, 81, _ => true);
        }

        if (variant is "MaskInOddRegister" or "MaskPairWrite")
        {
            Rename(instructions, 0, 1, instruction => instruction.Opcode is "SAndB32" or "VCndmaskB32");
        }

        var cndmask = Find("VCndmaskB32");
        var maskWrite = Find("SAndB32");
        var read = Find("VReadlaneB32");
        switch (variant)
        {
            case "IdentityChanged":
                instructions[cndmask] = instructions[cndmask] with { Sources = [Operand(5), .. instructions[cndmask].Sources.Skip(1)] };
                break;
            case "ShiftOrder":
                (instructions[cndmask + 1], instructions[cndmask + 2]) = (instructions[cndmask + 2], instructions[cndmask + 1]);
                break;
            case "BoundControl":
                instructions[cndmask + 3] = instructions[cndmask + 3] with { Control = new Gen5DppControl(0x114, false, true, 0, 0, 15, 15) };
                break;
            case "OtherLane":
                instructions[read] = ReadLane(1, 2, 14);
                break;
            case "ExecNotSaved":
                var saveexec = Find("SOrn2SaveexecB32");
                instructions[saveexec] = Sop1("SOrn2SaveexecB32", 20, Gen5Operand.Scalar(80));
                break;
            case "ScanSourceChanged":
                instructions[cndmask + 2] = instructions[cndmask + 2] with { Sources = [Gen5Operand.Vector(1), Gen5Operand.Vector(2)] };
                break;
            case "MaskAllLanes":
                // A lane missing from the simdgroup would take value, not the identity, on the PS5.
                instructions[maskWrite] = MoveScalar(0, uint.MaxValue);
                break;
            case "MaskNotFromEntryCopy":
                instructions[maskWrite] = Sop2("SAndB32", 0, Gen5Operand.Scalar(126), Gen5Operand.Scalar(81));
                instructions.Insert(0, MoveScalar(81, 0x7FFFF));
                break;
            case "MaskInVcc":
                instructions[cndmask] = Vop2("VCndmaskB32", 2, Operand(0), Gen5Operand.Vector(1));
                break;
            case "EntryCopyOverwritten":
                instructions.Insert(maskWrite, Sop2("SOrB32", 80, Gen5Operand.Scalar(80), Operand(1u << 25)));
                break;
            case "EntryCopyAfterExecWrite":
                instructions.Insert(0, Sop2("SAndB32", 126, Gen5Operand.Scalar(126), Operand(uint.MaxValue)));
                break;
            case "EntryCopyPairWrite":
                // S_MOV_B64 s[80:81] names s80 only.
                instructions.Insert(maskWrite, Sop1("SMovB64", 80, Gen5Operand.Scalar(8)));
                break;
            case "MaskPairWrite":
                // S_MOV_B64 s[0:1] names s0 only.
                instructions.Insert(maskWrite + 1, Sop1("SMovB64", 0, Gen5Operand.Scalar(8)));
                break;
            case "SavedSourceKept":
            case "SavedSourcePairWrite":
                // S_MOV_B32 exec_lo, s3 makes s3 hold EXEC; S_MOV_B64 s[2:3] names s2 only.
                var savedAt = Find("SOrn2SaveexecB32");
                instructions[savedAt] = Sop1("SOrn2SaveexecB32", 20, Gen5Operand.Scalar(3));
                instructions.InsertRange(savedAt, variant == "SavedSourceKept"
                    ? [MoveScalarRegister(3, 126), MoveScalarRegister(126, 3)]
                    : [MoveScalarRegister(3, 126), MoveScalarRegister(126, 3), Sop1("SMovB64", 2, Gen5Operand.Scalar(8))]);
                break;
            case "OneRead":
                instructions[read + 1] = MoveScalar(2, Unread);
                break;
            case "SameRowTwice":
                // The second read of row 0 is left to the scan.
                instructions[read + 1] = ReadLane(2, 2, 15);
                break;
            case "ReadAfterOtherInstruction":
                instructions.Insert(read, MoveScalar(3, 0));
                break;
            case "WaitBeforeCndmask":
                instructions.Insert(cndmask, Sopp("SWaitcnt"));
                break;
        }

        var source = Translate(Placed(instructions), ShaderStage.Compute, variant != "NoSubgroupArithmetic");
        Assert.Equal(reductions, Occurrences(source, "simd_max("));
    }

    // V_ADD_NC_U32 decodes as VAddI32 in its VOP2 (DPP) form and as VAddU32 in VOP3; the scan is the same add.
    [Theory]
    [InlineData("VAddI32", 2)]
    [InlineData("VAddU32", 2)]
    [InlineData("VSubU32", 0)]
    public void TheScanAddMatchesUnderEitherDecodedName(string opcode, int reductions)
    {
        var instructions = Decode(Program("VAddU32", Reads.Both, pixel: false)).Instructions.ToList();
        Assert.Equal(4, instructions.Count(instruction => instruction is { Opcode: "VAddI32", Control: Gen5DppControl }));
        for (var index = 0; index < instructions.Count; index++)
        {
            if (instructions[index].Control is Gen5DppControl)
            {
                instructions[index] = instructions[index] with { Opcode = opcode };
            }
        }

        Assert.Equal(reductions, Occurrences(Translate(Placed(instructions), ShaderStage.Compute, true), "simd_sum("));
    }

    // Without the request flag, in a stage whose simdgroup is not one guest wave32, or for a shader without the scan,
    // the translation is the one it always was.
    [Fact]
    public void ShadersWithoutTheScanOrTheFlagTranslateAsBefore()
    {
        var compute = Decode(Program("VAddU32", Reads.Both, pixel: false));
        var pixel = Decode(Program("VAddU32", Reads.Both, pixel: true));
        Assert.DoesNotContain("simd_sum(", Translate(compute, ShaderStage.Compute, false), StringComparison.Ordinal);
        Assert.DoesNotContain("simd_sum(", Translate(pixel, ShaderStage.Pixel, false), StringComparison.Ordinal);

        // Single-lane pixel stage (graphics subgroups off) and a wave64 compute program.
        Assert.Equal(
            Translate(pixel, ShaderStage.Pixel, false, enableGraphicsSubgroupOperations: false),
            Translate(pixel, ShaderStage.Pixel, true, enableGraphicsSubgroupOperations: false));
        Assert.Equal(
            Translate(compute, ShaderStage.Compute, false, waveLaneCount: 64, localSizeX: 64),
            Translate(compute, ShaderStage.Compute, true, waveLaneCount: 64, localSizeX: 64));

        // Programs without the pattern: byte for byte the same with the flag.
        foreach (var fixture in Gen5ComputeFixtures.All)
        {
            var program = Gen5ComputeFixtures.DecodeOrThrow(fixture);
            Assert.Equal(Translate(program, ShaderStage.Compute, false), Translate(program, ShaderStage.Compute, true));
        }

        var pixelFixture = Gen5ComputeFixtures.DecodeOrThrow(new Gen5ComputeFixture("pixel", Gen5ComputeFixtures.PixelWords, 0, 0));
        Assert.Equal(Translate(pixelFixture, ShaderStage.Pixel, false), Translate(pixelFixture, ShaderStage.Pixel, true));
    }

    // The two-row pattern compiles in a fragment function with the partial-lane lowering on as well: the reduction
    // reads no other lane, so it does not depend on how absent lanes are read.
    [Fact]
    public void TheReductionDoesNotDependOnThePartialLaneLowering()
    {
        var program = Decode(Program("VXorB32", Reads.Both, pixel: true));
        var partial = Translate(program, ShaderStage.Pixel, true, graphicsPartialSubgroupLanes: true);
        Assert.Equal(2, Occurrences(partial, "simd_xor("));
        if (SkipWithoutMetalDevice("the partial-lane reduction compile")) return;

        Assert.True(MetalNative.TryCompileLibrary(partial + VertexSource, out _, out var error), $"{error}\n{partial}");
    }

    // ---- the guest program ----

    // Each lane's value: 8 * lane + 100 (MAX_U, ADD), 3 * lane + 7 (XOR: the XOR of one quad's is not zero),
    // 1023 - 8 * lane (MIN_U), 8 * lane - 100 (MAX_I, MIN_I), 1 << lane (OR), ~(1 << lane) (AND).
    private static uint Value(string operation, uint lane) => operation switch
    {
        "VXorB32" => 3 * lane + 7,
        "VMinU32" => 1023 - 8 * lane,
        "VMaxI32" or "VMinI32" => unchecked(8 * lane - 100),
        "VOrB32" => 1u << (int)lane,
        "VAndB32" => ~(1u << (int)lane),
        _ => 8 * lane + 100,
    };

    private static uint[] Identity(string operation, out uint identity)
    {
        identity = operation switch
        {
            "VMinU32" or "VAndB32" => uint.MaxValue,
            "VMaxI32" => 0x8000_0000u,
            "VMinI32" => 0x7FFF_FFFFu,
            _ => 0u,
        };
        // v_cndmask_b32_e64 v2, identity, v1, s0
        return identity switch
        {
            0u => [0xD5010002, 0x00020280],
            uint.MaxValue => [0xD5010002, 0x000202C1],
            _ => [0xD5010002, 0x000202FF, identity],
        };
    }

    private static uint Combine(string operation, uint left, uint right) => operation switch
    {
        "VMaxU32" => Math.Max(left, right),
        "VMinU32" => Math.Min(left, right),
        "VMaxI32" => (uint)Math.Max((int)left, (int)right),
        "VMinI32" => (uint)Math.Min((int)left, (int)right),
        "VOrB32" => left | right,
        "VAndB32" => left & right,
        "VXorB32" => left ^ right,
        _ => left + right,
    };

    // The operation over the lanes of a row that run (present) and are in the scan mask (lanes 0 to 18).
    private static uint Fold(string operation, uint row, uint present)
    {
        Identity(operation, out var result);
        for (var lane = row * 16; lane < row * 16 + 16; lane++)
        {
            if (((present >> (int)lane) & 1u) != 0 && lane <= 18) result = Combine(operation, result, Value(operation, lane));
        }

        return result;
    }

    private static (uint Fifteen, uint ThirtyOne) Expected(string operation, Reads reads, uint present) =>
    (
        reads == Reads.ThirtyOne ? Unread : Fold(operation, 0, present),
        reads == Reads.Fifteen ? Unread : Fold(operation, 1, present)
    );

    // DPP V_<op> v2, v2, v2 row_shr:shift, full masks, FI and BOUND_CTRL clear.
    private static uint[] Step(string operation, uint shift)
    {
        uint opcode = operation switch
        {
            "VMaxU32" => 0x14,
            "VMinU32" => 0x13,
            "VMaxI32" => 0x12,
            "VMinI32" => 0x11,
            "VOrB32" => 0x1C,
            "VAndB32" => 0x1B,
            "VXorB32" => 0x1D,
            "VAddU32" => 0x25,
            _ => throw new ArgumentOutOfRangeException(nameof(operation)),
        };
        return [0x000404FA | (opcode << 25), 0xFF000002 | ((0x110 + shift) << 8)];
    }

    private static uint[] Program(string operation, Reads reads, bool pixel)
    {
        var words = new List<uint>
        {
            0xD7650000, 0x020100C1, // v_mbcnt_lo_u32_b32 v0, -1, 0
            0xBE89037E,             // s_mov_b32 s9, exec_lo
            0xBED0037E,             // s_mov_b32 s80, exec_lo
            0x8A50FF50, 0xFFF80000, // s_andn2_b32 s80, s80, 0xfff80000
        };
        words.AddRange(operation switch
        {
            // v_lshlrev_b32 v1, 3, v0; v_xor_b32 v1, 0x3ff, v1
            "VMinU32" => [0x34020083, 0x3A0202FF, 0x000003FF],
            // v_lshlrev_b32 v1, 3, v0; v_add_nc_u32 v1, -100, v1
            "VMaxI32" or "VMinI32" => [0x34020083, 0x4A0202FF, 0xFFFFFF9C],
            // v_mad_u32_u24 v1, v0, 3, 7
            "VXorB32" => [0xD5430001, 0x021D0700],
            // v_lshlrev_b32_e64 v1, v0, 1
            "VOrB32" => [0xD51A0001, 0x02010300],
            // v_lshlrev_b32_e64 v1, v0, 1; v_not_b32 v1, v1
            "VAndB32" => [0xD51A0001, 0x02010300, 0x7E026F01],
            // v_lshlrev_b32 v1, 3, v0; v_add_nc_u32 v1, 100, v1
            _ => [0x34020083, 0x4A0202FF, 0x00000064],
        });
        words.Add(0x8700507E); // s_and_b32 s0, exec_lo, s80
        words.Add(0xBE94407E); // s_orn2_saveexec_b32 s20, exec_lo
        words.AddRange(Identity(operation, out _));
        foreach (var shift in new uint[] { 1, 2, 4, 8 })
        {
            words.AddRange(Step(operation, shift));
        }

        words.Add(0xBEFE0314); // s_mov_b32 exec_lo, s20
        uint[] read15 = [0xD7600001, 0x02011F02]; // v_readlane_b32 s1, v2, 15
        uint[] read31 = [0xD7600002, 0x02013F02]; // v_readlane_b32 s2, v2, 31
        uint[] unread1 = [0xBE8103FF, Unread];    // s_mov_b32 s1, 0xeeee
        uint[] unread2 = [0xBE8203FF, Unread];    // s_mov_b32 s2, 0xeeee
        words.AddRange(reads switch
        {
            Reads.Both => [.. read15, .. read31],
            Reads.BothReversed => [.. read31, .. read15],
            Reads.Fifteen => [.. read15, .. unread2],
            _ => [.. read31, .. unread1],
        });
        words.AddRange([0x7E080201, 0x7E0A0202]); // v_mov_b32 v4, s1; v_mov_b32 v5, s2
        words.AddRange(pixel
            ?
            [
                0x7E180209,             // v_mov_b32 v12, s9
                0xF800180F, 0x0C000504, // exp mrt0 v4, v5, v0, v12 done vm
            ]
            :
            [
                0xE0702000, 0x80010400, // buffer_store_dword v4, v0, s[4:7], 0 idxen
                0xE0702004, 0x80010500, // buffer_store_dword v5, v0, s[4:7], 0 idxen offset:4
            ]);
        words.Add(0xBF810000); // s_endpgm
        return [.. words];
    }

    private static Gen5ShaderProgram Decode(uint[] words) =>
        Gen5ComputeFixtures.DecodeOrThrow(new Gen5ComputeFixture("wave-reduction", words, 0, 0));

    private static ResourceRuntimeInputs StoreInputs(uint threads)
    {
        var userData = new uint[16];
        userData[4] = 0x200000;
        userData[5] = 8u << 16;
        userData[6] = threads;
        userData[7] = DescriptorConstants.IdentityDestinationSelect;
        return new ResourceRuntimeInputs { UserData = userData };
    }

    private static ShaderCompileRequest Request(Gen5ShaderProgram program, ShaderStage stage, bool subgroupArithmetic,
        uint waveLaneCount = 32, uint localSizeX = 32, bool enableGraphicsSubgroupOperations = true, bool? graphicsPartialSubgroupLanes = null) =>
        Gen5ComputeFixtures.RequestOrThrow(program, stage, waveLaneCount: waveLaneCount, localSizeX: localSizeX,
            pixelOutputs: stage == ShaderStage.Pixel ? [new Gen5PixelOutputBinding(0, 0, Gen5PixelOutputKind.Uint)] : null,
            runtimeInputs: stage == ShaderStage.Compute && program.Instructions.Any(instruction => instruction.Control is Gen5BufferMemoryControl)
                ? StoreInputs(localSizeX) : null,
            enableGraphicsSubgroupOperations: enableGraphicsSubgroupOperations,
            pixelGuestWave32: true,
            graphicsPartialSubgroupLanes: graphicsPartialSubgroupLanes,
            subgroupArithmetic: subgroupArithmetic);

    private static string Translate(Gen5ShaderProgram program, ShaderStage stage, bool subgroupArithmetic,
        uint waveLaneCount = 32, uint localSizeX = 32, bool enableGraphicsSubgroupOperations = true, bool? graphicsPartialSubgroupLanes = false)
    {
        var request = Request(program, stage, subgroupArithmetic, waveLaneCount, localSizeX, enableGraphicsSubgroupOperations,
            graphicsPartialSubgroupLanes);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        return shader.Source;
    }

    private (uint S1, uint S2)[] RunCompute(string operation, Reads reads, uint threads, bool subgroupArithmetic)
    {
        var request = Request(Decode(Program(operation, reads, pixel: false)), ShaderStage.Compute, subgroupArithmetic,
            localSizeX: threads, graphicsPartialSubgroupLanes: false);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), $"{compileError}\n{shader.Source}");
        Assert.True(
            MetalNative.TryExecuteWithArgumentBuffer(
                library,
                shader.EntryPoint,
                new byte[threads * 8],
                new byte[Resources.PushData.ByteSize],
                shader.ArgumentLayout!,
                threads,
                out var bytes,
                out var runError),
            runError);
        var lanes = new (uint, uint)[threads];
        for (var lane = 0; lane < threads; lane++)
        {
            lanes[lane] = (BitConverter.ToUInt32(bytes, lane * 8), BitConverter.ToUInt32(bytes, lane * 8 + 4));
        }

        return lanes;
    }

    private const string VertexSource = """

        struct SharpEmuWaveReductionVertex
        {
            float4 position [[position]];
        };

        vertex SharpEmuWaveReductionVertex sharpemu_wave_reduction_vs(
            uint id [[vertex_id]],
            constant float2* positions [[buffer(0)]])
        {
            SharpEmuWaveReductionVertex vertex_out;
            vertex_out.position = float4(positions[id], 0.0f, 1.0f);
            return vertex_out;
        }
        """;

    private static byte[] Render(string operation, Reads reads, bool subgroupArithmetic)
    {
        var request = Request(Decode(Program(operation, reads, pixel: true)), ShaderStage.Pixel, subgroupArithmetic,
            graphicsPartialSubgroupLanes: false);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.True(
            MetalNative.TryCompileLibrary(shader.Source + VertexSource, out var library, out error),
            $"Metal rejected the wave reduction MSL: {error}\n{shader.Source}");

        // Covers only the center of pixel (1,1) of the 8x8 target.
        var triangle = new[] { -0.8f, 0.8f, -0.4f, 0.8f, -0.8f, 0.4f };
        var positions = new byte[triangle.Length * sizeof(float)];
        Buffer.BlockCopy(triangle, 0, positions, 0, positions.Length);
        var resources = new byte[Math.Max(shader.ArgumentLayout?.ByteSize ?? 0, 256)];
        Assert.True(
            MetalNative.TryRenderTriangles(
                library,
                "sharpemu_wave_reduction_vs",
                shader.EntryPoint,
                [(positions, 0)],
                [(resources, (uint)Gen5MslArgumentLayout.ResourcesBufferIndex)],
                3,
                TargetSize,
                TargetSize,
                MetalNative.Rgba32UintPixelFormat,
                out var texels,
                out error),
            error);
        return texels;
    }

    private static (uint R, uint G, uint B, uint A) Texel(byte[] texels, uint x, uint y)
    {
        var offset = (int)(((y * TargetSize) + x) * 16);
        return (
            BitConverter.ToUInt32(texels, offset),
            BitConverter.ToUInt32(texels, offset + 4),
            BitConverter.ToUInt32(texels, offset + 8),
            BitConverter.ToUInt32(texels, offset + 12));
    }

    // ---- hand-built instructions for the matcher cases (shapes of Libs.Tests' ResourceTestProgram) ----

    private static Gen5Operand Operand(uint value) =>
        value <= 64 ? Gen5Operand.Source(128 + value) : new Gen5Operand(Gen5OperandKind.LiteralConstant, value);

    private static Gen5ShaderInstruction MoveScalar(uint destination, uint value) =>
        new(0, Gen5ShaderEncoding.Sop1, "SMovB32", [0u], [Operand(value)], [Gen5Operand.Scalar(destination)], null);

    private static Gen5ShaderInstruction MoveScalarRegister(uint destination, uint source) =>
        new(0, Gen5ShaderEncoding.Sop1, "SMovB32", [0u], [Gen5Operand.Scalar(source)], [Gen5Operand.Scalar(destination)], null);

    private static Gen5ShaderInstruction Sop1(string opcode, uint destination, Gen5Operand source) =>
        new(0, Gen5ShaderEncoding.Sop1, opcode, [0u], [source], [Gen5Operand.Scalar(destination)], null);

    private static Gen5ShaderInstruction Sop2(string opcode, uint destination, Gen5Operand source0, Gen5Operand source1) =>
        new(0, Gen5ShaderEncoding.Sop2, opcode, [0u], [source0, source1], [Gen5Operand.Scalar(destination)], null);

    private static Gen5ShaderInstruction Sopp(string opcode) =>
        new(0, Gen5ShaderEncoding.Sopp, opcode, [0xBF8C0000u], [], [], null);

    private static Gen5ShaderInstruction Vop2(string opcode, uint destination, Gen5Operand source0, Gen5Operand source1) =>
        new(0, Gen5ShaderEncoding.Vop2, opcode, [0u], [source0, source1], [Gen5Operand.Vector(destination)], null);

    private static Gen5ShaderInstruction ReadLane(uint scalarRegister, uint vectorRegister, uint lane) =>
        new(0, Gen5ShaderEncoding.Vop3, "VReadlaneB32", [0u, 0u],
            [Gen5Operand.Vector(vectorRegister), Gen5Operand.Source(128 + lane), Gen5Operand.Scalar(0)], [Gen5Operand.Scalar(scalarRegister)], null);

    // Renames an SGPR in the operands of the instructions the filter picks.
    private static void Rename(List<Gen5ShaderInstruction> instructions, uint from, uint to, Func<Gen5ShaderInstruction, bool> filter)
    {
        Gen5Operand Map(Gen5Operand operand) =>
            operand.Kind == Gen5OperandKind.ScalarRegister && operand.Value == from ? Gen5Operand.Scalar(to) : operand;
        for (var index = 0; index < instructions.Count; index++)
        {
            if (filter(instructions[index]))
            {
                instructions[index] = instructions[index] with
                {
                    Sources = instructions[index].Sources.Select(Map).ToArray(),
                    Destinations = instructions[index].Destinations.Select(Map).ToArray(),
                };
            }
        }
    }

    // Straight-line code: each instruction at its own pc.
    private static Gen5ShaderProgram Placed(List<Gen5ShaderInstruction> instructions)
    {
        var placed = new List<Gen5ShaderInstruction>();
        uint pc = 0;
        foreach (var instruction in instructions)
        {
            placed.Add(instruction with { Pc = pc });
            pc += 8;
        }

        return new Gen5ShaderProgram(0, placed);
    }

    private static int Occurrences(string source, string text) =>
        (source.Length - source.Replace(text, string.Empty, StringComparison.Ordinal).Length) / text.Length;

    private bool SkipWithoutMetalDevice(string activity)
    {
        if (MetalNative.IsAvailable) return false;
        Assert.False(
            OperatingSystem.IsMacOS() && Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1",
            $"The required gate cannot run {activity} without a Metal device.");
        output.WriteLine($"No Metal device on this host; {activity} skipped.");
        return true;
    }
}
