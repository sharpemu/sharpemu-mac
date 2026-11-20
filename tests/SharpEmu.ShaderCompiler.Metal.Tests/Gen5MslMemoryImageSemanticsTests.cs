// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Text.RegularExpressions;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// Memory, image and cross-lane semantics the translator used to get wrong silently: operand
// overlap on device-descriptor scalar loads, LOD clamp, A16 address packing, UNRM, wave64
// bpermute, swizzled buffers and S_SETREG on MODE.
public sealed class Gen5MslMemoryImageSemanticsTests(ITestOutputHelper output)
{
    private const uint SEndpgm = 0xBF810000;

    // ---- S_BUFFER_LOAD through a device-resolved V# whose destination overlaps it ----

    [Fact]
    public void WideScalarBufferLoadSnapshotsTheDescriptorBeforeWritingAnyDestination()
    {
        // s_buffer_load_dwordx4 s[12:15], s[12:15], 0 on a descriptor the draw cannot bind.
        var program = new Gen5ShaderProgram(0x2000, [
            ReadFirstLane(0, 12, 0),
            MoveScalar(4, 13, 0),
            MoveScalar(8, 14, 16),
            MoveScalar(12, 15, 0),
            ScalarBufferLoad(16, resourceRegister: 12, destination: 12, count: 4),
            new Gen5ShaderInstruction(24, Gen5ShaderEncoding.Sopp, "SEndpgm", [SEndpgm], [], [], null),
        ]);
        var request = Gen5ComputeFixtures.RequestOrThrow(
            program, ShaderStage.Compute, localSizeX: 1,
            runtimeInputs: new ResourceRuntimeInputs { UserData = new uint[16] });
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);

        var source = shader!.Source;
        // The load's own first destination store is the last write of s12 (the setup moves precede it).
        var firstStore = Regex.Matches(source, @"(?m)^\s*s12 = ").Last();
        var afterStores = source[firstStore.Index..];
        // Every read of the descriptor words (s12..s14) has to precede the first destination store:
        // after it they hold loaded data, not the V#.
        var staleReads = Regex.Matches(afterStores, @"\bs1[234]\b(?!\s*=[^=])")
            .Select(match => afterStores.Substring(Math.Max(0, match.Index - 40), 60))
            .ToArray();
        Assert.True(staleReads.Length == 0, "descriptor re-read after a destination store: " + string.Join(" | ", staleReads));
    }

    // ---- S_SETREG_B32 ----

    [Theory]
    [InlineData(0x0A01u, false)] // hwreg(MODE, 8, 2): DX10_CLAMP and IEEE, honoured via launch state
    [InlineData(0x0201u, false)] // hwreg(MODE, 8, 1)
    [InlineData(0x1901u, true)]  // hwreg(MODE, 4, 4): FP_DENORM, not honoured anywhere in this translator
    [InlineData(0x0003u, true)]  // hwreg(TRAPSTS): no effect on results
    public void SetregOfModeIeeeOrClampIsRefusedAndOtherFieldsRemainNoOps(uint simm16, bool accepted)
    {
        var fixture = new Gen5ComputeFixture("setreg",
            [0xB9840000u | simm16, SEndpgm], 8, 16); // s_setreg_b32 hwreg(...), s4
        var program = Gen5ComputeFixtures.DecodeOrThrow(fixture);
        Assert.Contains(program.Instructions, static instruction => instruction.Opcode == "SSetregB32");
        var request = Gen5ComputeFixtures.RequestOrThrow(program, ShaderStage.Compute);
        var compiled = Gen5MslTranslator.TryCompileProgram(request, out _, out var error);
        Assert.Equal(accepted, compiled);
        if (!accepted)
        {
            Assert.Contains("MODE.IEEE/DX10_CLAMP", error);
        }
    }

    // ---- buffer swizzle / add_tid ----

    // MUBUF addressing (RDNA2 ISA 8.1.5): index = vindex + (ADD_TID_ENABLE ? lane : 0) and
    // offset = voffset + inst_offset, with the SGPR offset added outside. A swizzled V#
    // places each dword on its own; on gfx10 the element is one dword.
    private static uint ReferenceBufferAddress(uint soffset, uint index, uint offset, uint stride, uint indexStride) =>
        indexStride == 0
            ? soffset + offset + (index * stride)
            : soffset + ((((index / indexStride) * stride) + (offset & ~3u)) * indexStride) + ((index % indexStride) * 4) + (offset & 3);

    private const uint SwizzleStride = 32;
    private const uint SwizzleSoffset = 16;     // the MUBUF soffset field's inline constant 16
    private const uint SwizzleVectorOffset = 8;
    private const uint SwizzleInstOffset = 4;
    private const int SwizzleBufferBytes = 4096;

    // The V# in s[8:11]: stride 32, 64 records, SWIZZLE_ENABLE in word 1 bit 31, INDEX_STRIDE
    // in word 3 bits 21-22 and ADD_TID_ENABLE in word 3 bit 23.
    private static uint[] SwizzleUserData(bool swizzle, uint indexStrideCode, bool addTid)
    {
        var userData = new uint[16];
        userData[8] = 0x200000;
        userData[9] = (SwizzleStride << 16) | (swizzle ? 1u << 31 : 0u);
        userData[10] = 64;
        userData[11] = DescriptorConstants.IdentityDestinationSelect | (indexStrideCode << 21) | (addTid ? 1u << 23 : 0u);
        // A linear V# in s[12:15] (stride 12) for the load test's write-back.
        userData[12] = 0x200000;
        userData[13] = 12u << 16;
        userData[14] = 64;
        userData[15] = DescriptorConstants.IdentityDestinationSelect;
        return userData;
    }

    private static uint ExpectedIndex(uint lane, bool addTid) => lane + 3 + (addTid ? lane : 0);

    private static uint IndexStrideOf(bool swizzle, uint indexStrideCode) => swizzle ? 8u << (int)indexStrideCode : 0u;

    private static Gen5MslShader CompileSwizzle(string name, uint[] words, bool swizzle, uint indexStrideCode, bool addTid)
    {
        var fixture = new Gen5ComputeFixture(name, words, 8, SwizzleBufferBytes);
        var request = Gen5ComputeFixtures.RequestOrThrow(Gen5ComputeFixtures.DecodeOrThrow(fixture), ShaderStage.Compute,
            runtimeInputs: new ResourceRuntimeInputs { UserData = SwizzleUserData(swizzle, indexStrideCode, addTid) });
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        return shader!;
    }

    // Each of 32 lanes stores four dwords through buffer_store_dwordx4 idxen offen with
    // index lane + 3 and offset 8 + 4: every dword must land where the ISA formula puts
    // it, and nothing else may change.
    [Theory]
    [InlineData(true, 0u, false)]
    [InlineData(true, 1u, true)]
    [InlineData(true, 3u, true)]
    [InlineData(true, 2u, false)]
    [InlineData(false, 0u, true)]   // ADD_TID_ENABLE on a linear V#
    public void ASwizzledBufferStorePlacesEachDwordAtItsIsaAddress(bool swizzle, uint indexStrideCode, bool addTid)
    {
        var shader = CompileSwizzle(
            "swizzled-store",
            [
                0xD7650000, 0x000100C1, // v_mbcnt_lo_u32_b32 v0, -1, 0      -> v0 = lane
                0x4A040083,             // v_add_nc_u32 v2, 3, v0           -> vindex = lane + 3
                0x7E060288,             // v_mov_b32 v3, 8                  -> voffset
                0x380800FF, 0xA0000000, // v_or_b32 v4, 0xA0000000, v0
                0x380A00FF, 0xB0000000, // v_or_b32 v5, 0xB0000000, v0
                0x380C00FF, 0xC0000000, // v_or_b32 v6, 0xC0000000, v0
                0x380E00FF, 0xD0000000, // v_or_b32 v7, 0xD0000000, v0
                0xE0783004, 0x90020402, // buffer_store_dwordx4 v[4:7], v[2:3], s[8:11], 16 idxen offen offset:4
                SEndpgm,
            ],
            swizzle, indexStrideCode, addTid);
        Assert.Equal(swizzle, shader.Source.Contains("sharpemu_buffer_offset(", StringComparison.Ordinal));
        if (!RequireDevice()) return;

        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[SwizzleBufferBytes],
            new byte[PushData.ByteSize], shader.ArgumentLayout!, 32, out var result, out var runError), runError);

        var expected = new byte[SwizzleBufferBytes];
        for (uint lane = 0; lane < 32; lane++)
        {
            for (uint dword = 0; dword < 4; dword++)
            {
                var address = ReferenceBufferAddress(
                    SwizzleSoffset,
                    ExpectedIndex(lane, addTid),
                    SwizzleVectorOffset + SwizzleInstOffset + (dword * 4),
                    SwizzleStride,
                    IndexStrideOf(swizzle, indexStrideCode));
                BinaryPrimitives.WriteUInt32LittleEndian(expected.AsSpan((int)address), (0xA0000000u + (dword * 0x10000000u)) | lane);
            }
        }

        Assert.Equal(expected, result);
    }

    // Each lane loads three dwords through buffer_load_dwordx3 from the same placement and
    // writes them back linearly through a second V#: each must be the dword at its ISA address.
    [Theory]
    [InlineData(true, 1u, true)]
    [InlineData(true, 0u, false)]
    [InlineData(false, 0u, true)]
    public void ASwizzledBufferLoadReadsEachDwordFromItsIsaAddress(bool swizzle, uint indexStrideCode, bool addTid)
    {
        const uint writeBack = 0x800;
        var shader = CompileSwizzle(
            "swizzled-load",
            [
                0xD7650000, 0x000100C1, // v_mbcnt_lo_u32_b32 v0, -1, 0
                0x4A040083,             // v_add_nc_u32 v2, 3, v0
                0x7E060288,             // v_mov_b32 v3, 8
                0xBE9003FF, writeBack,  // s_mov_b32 s16, 0x800
                0xE03C3004, 0x90020402, // buffer_load_dwordx3 v[4:6], v[2:3], s[8:11], 16 idxen offen offset:4
                0xE07C2000, 0x10030400, // buffer_store_dwordx3 v[4:6], v0, s[12:15], s16 idxen
                SEndpgm,
            ],
            swizzle, indexStrideCode, addTid);
        if (!RequireDevice()) return;

        var initial = new byte[SwizzleBufferBytes];
        for (var word = 0; word < writeBack / 4; word++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(initial.AsSpan(word * 4), (uint)word * 0x01010101u ^ 0x5A000000u);
        }

        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, initial,
            new byte[PushData.ByteSize], shader.ArgumentLayout!, 32, out var result, out var runError), runError);

        for (uint lane = 0; lane < 32; lane++)
        {
            for (uint dword = 0; dword < 3; dword++)
            {
                var address = ReferenceBufferAddress(
                    SwizzleSoffset,
                    ExpectedIndex(lane, addTid),
                    SwizzleVectorOffset + SwizzleInstOffset + (dword * 4),
                    SwizzleStride,
                    IndexStrideOf(swizzle, indexStrideCode));
                Assert.Equal(
                    BinaryPrimitives.ReadUInt32LittleEndian(initial.AsSpan((int)address)),
                    BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan((int)(writeBack + (lane * 12) + (dword * 4)))));
            }
        }
    }

    // ADD_TID_ENABLE adds the lane in the wave; a stage that models a single lane has none
    // to add, so the access fails translation instead of addressing lane 0 for every lane.
    [Fact]
    public void AnAddTidBufferInAStageWithoutWaveLanesFailsTranslation()
    {
        var request = Gen5ComputeFixtures.RequestOrThrow(
            Gen5ComputeFixtures.DecodeOrThrow(new Gen5ComputeFixture("add-tid-vertex", [0xE0303000, 0x80020401, SEndpgm], 8, 64)),
            ShaderStage.Vertex,
            runtimeInputs: new ResourceRuntimeInputs { UserData = SwizzleUserData(false, 0, addTid: true) });
        Assert.False(Gen5MslTranslator.TryCompileProgram(request, out _, out var error));
        Assert.Contains("adds the thread id", error);
    }

    // The placement and footprint helpers every swizzled access goes through, run on the
    // device against the ISA formula: odd offsets (offset_lsb), each index stride, a zero
    // (linear) index stride, and indices that cross index_stride groups. The footprint is
    // one past the furthest byte an in-range access (index < records, offset < stride)
    // reaches, found by brute force here.
    [Fact]
    public void TheBufferOffsetHelpersMatchTheIsaFormula()
    {
        var shader = CompileSwizzle(
            "swizzled-helpers",
            [0xE0303000, 0x80020401, SEndpgm], // buffer_load_dword v4, v1, s[8:11], 0 offen
            swizzle: true, indexStrideCode: 0, addTid: false);
        Assert.Contains("sharpemu_swizzled_footprint(", shader.Source, StringComparison.Ordinal);
        if (!RequireDevice()) return;

        // index, offset, stride, index stride, records
        uint[][] cases =
        [
            [0, 0, 32, 8, 1], [7, 4, 32, 8, 8], [8, 4, 32, 8, 9], [9, 13, 32, 16, 10], [37, 30, 48, 32, 37],
            [100, 0, 0x3F90, 8, 101], [63, 0x3F8C, 0x3F90, 64, 64], [65, 7, 16, 64, 66], [3, 21, 24, 16, 3],
            [12345, 4, 12, 0, 0], [5, 3, 0, 8, 5], [200, 60, 64, 32, 250],
        ];
        const string kernel = @"
kernel void buffer_offset_test(device uint* b [[buffer(0)]], uint t [[thread_position_in_threadgroup]])
{
    uint i = t * 8u;
    b[i + 5u] = sharpemu_buffer_offset(b[i], b[i + 1u], b[i + 2u], b[i + 3u]);
    ulong footprint = b[i + 3u] == 0u ? 0ul : sharpemu_swizzled_footprint(b[i + 4u], b[i + 2u], b[i + 3u]);
    b[i + 6u] = (uint)footprint;
    b[i + 7u] = (uint)(footprint >> 32);
}
";
        Assert.True(MetalNative.TryCompileLibrary(shader.Source + kernel, out var library, out var compileError), compileError);
        var contents = new byte[cases.Length * 32];
        for (var index = 0; index < cases.Length; index++)
        {
            for (var field = 0; field < 5; field++)
            {
                BinaryPrimitives.WriteUInt32LittleEndian(contents.AsSpan((index * 32) + (field * 4)), cases[index][field]);
            }
        }

        Assert.True(MetalNative.TryExecuteOnBuffer(library, "buffer_offset_test", contents, (uint)cases.Length, out var result, out var error), error);
        for (var index = 0; index < cases.Length; index++)
        {
            var (vindex, offset, stride, indexStride, records) = (cases[index][0], cases[index][1], cases[index][2], cases[index][3], cases[index][4]);
            var row = result.AsSpan(index * 32);
            Assert.Equal(ReferenceBufferAddress(0, vindex, offset, stride, indexStride), BinaryPrimitives.ReadUInt32LittleEndian(row[20..]));

            ulong footprint = 0;
            if (indexStride != 0 && stride != 0)
            {
                for (uint record = 0; record < records; record++)
                {
                    for (uint at = 0; at < stride; at++)
                    {
                        footprint = Math.Max(footprint, (ulong)ReferenceBufferAddress(0, record, at, stride, indexStride) + 1);
                    }
                }
            }

            Assert.Equal(footprint, BinaryPrimitives.ReadUInt64LittleEndian(row[24..]));
        }
    }

    // ---- DS_BPERMUTE_B32 under wave64 ----

    [Fact]
    public void Wave64BpermuteShufflesWithinTheSimdgroupAndStaysInItsOwnHalf()
    {
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(Wave64Bpermute, waveLaneCount: 64, localSizeX: 64);
        // simd_shuffle takes a lane within the 32-wide simdgroup; 32..63 is out of range.
        Assert.Matches(@"simd_shuffle\([^\n]*\(ushort\)\(t\d+ & 31u\)\)", shader.Source);
        if (!RequireDevice()) return;

        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[256],
            new byte[PushData.ByteSize], shader.ArgumentLayout!, 64, out var result, out var runError), runError);
        // Every lane reads byte offset 12 = index 3 of its own half: lane 3 below, lane 35 above.
        for (var lane = 0; lane < 64; lane++)
        {
            Assert.Equal((uint)((lane & ~31) | 3), BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(lane * 4)));
        }
    }

    // ---- S_MEMTIME / S_MEMREALTIME and SMEM cache maintenance ----

    [Theory]
    [InlineData(0x24u, true)] // s_memtime
    [InlineData(0x25u, true)] // s_memrealtime
    [InlineData(0x24u, false)]
    [InlineData(0x25u, false)]
    public void ShaderClockReadsAreNonZeroAndAdvanceAcrossACacheInvalidate(uint opcode, bool wideStep)
    {
        // MSL has no shader clock, so the 64-bit counter advances on every read: a later
        // read is larger, never the hardcoded zero a missing clock would leave. The wide step
        // (default) moves it 2^32 + 1 per read, as upstream's SPIR-V fallback moves its high
        // dword: a wait on the 64-bit difference ends at the next read, one on the low dword
        // still sees it move by one. SHARPEMU_METAL_SHADER_CLOCK_STEP=0 steps by one.
        var fixture = new Gen5ComputeFixture($"smem-clock-{opcode:X2}-{wideStep}",
        [
            0xF4000000u | opcode << 18 | 4u << 6, 0u, // s_memtime s[4:5]
            0xF4800000u, 0u,                          // s_dcache_inv
            0xF47C0000u, 0u,                          // s_gl1_inv
            0xF4000000u | opcode << 18 | 6u << 6, 0u, // s_memtime s[6:7]
            0xBF8C0000,                               // s_waitcnt 0
            0x7E060204,                               // v_mov_b32 v3, s4
            0x7E080206,                               // v_mov_b32 v4, s6
            0x7E0A0207,                               // v_mov_b32 v5, s7
            0x7E0C0205,                               // v_mov_b32 v6, s5
            0xE0700000, 0x80020300,                   // buffer_store_dword v3, off, s[8:11], 0
            0xE0700004, 0x80020400,                   // buffer_store_dword v4, off, s[8:11], 0 offset:4
            0xE0700008, 0x80020500,                   // buffer_store_dword v5, off, s[8:11], 0 offset:8
            0xE070000C, 0x80020600,                   // buffer_store_dword v6, off, s[8:11], 0 offset:12
            SEndpgm,
        ], 8, 16);
        var before = Gen5MslTranslator.StandInClockPrograms;
        var shader = Gen5ComputeFixtures.CompileRequestOrThrow(fixture,
            options: Gen5MslCodegenOptions.FromEnvironment with { ShaderClockWideStep = wideStep });
        Assert.Contains("sharpemu_shader_clock", shader.Source);
        Assert.Equal(wideStep, shader.Source.Contains("sharpemu_shader_clock += 0x100000001ul;", StringComparison.Ordinal));
        Assert.True(Gen5MslTranslator.StandInClockPrograms > before);
        if (!RequireDevice()) return;

        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError),
            $"{compileError}\n{shader.Source}");
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[16],
            new byte[PushData.ByteSize], shader.ArgumentLayout!, 1, out var result, out var runError), runError);
        var first = BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(0));
        var second = BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(4));
        var secondHigh = BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(8));
        var firstHigh = BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(12));
        Assert.NotEqual(0u, first);
        Assert.True(second > first, $"{second} <= {first}");
        Assert.Equal(1u, second - first);
        // The 64-bit difference of the two reads: 2^32 + 1 with the wide step, which ends any wait for fewer
        // than 2^32 ticks; 1 without it.
        var difference = (((ulong)secondHigh << 32) | second) - (((ulong)firstHigh << 32) | first);
        Assert.Equal(wideStep ? 0x1_0000_0001ul : 1ul, difference);
    }

    // ---- image sampling: LOD clamp, A16, UNRM ----

    [Fact]
    public void SampleClHonoursTheLodClamp()
    {
        // image_sample_cl on a 2D texture: v0/v1 = coordinates, v2 = min-LOD clamp of 1.
        if (!CompileImage("ImageSampleCl", [F(.75f), F(.25f), F(1f)], out var shader)) return;
        // Mip 1 texel (1, 0): the clamp lifts the level that the explicit level(0) would have picked.
        AssertTexel(RunImage(shader), mip: 1, x: 1, y: 0);
    }

    [Theory]
    [InlineData("ImageSampleCl")]
    [InlineData("ImageSampleDCl")]
    [InlineData("ImageSampleBCl")]
    [InlineData("ImageSampleCCl")]
    public void ClampedSamplesCompileInAFragmentFunctionWithMinLodClamp(string opcode)
    {
        Assert.True(Gen5ImageFixtures.TryCompile(opcode, Gen5ImageFixtures.ImageType2D, out var shader, out var error,
            stage: ShaderStage.Pixel), error);
        Assert.Contains("metal::min_lod_clamp(", shader!.Source);
        if (!RequireDevice()) return;
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out _, out var compileError), compileError + shader.Source);
    }

    [Fact]
    public void A16SampleBiasOccupiesAFullDwordBeforeThePackedCoordinates()
    {
        // image_sample_b A16: v0 = bias half in the low 16 bits, v1 = pack(u, v). The compute stage
        // samples level 0, so the coordinates alone select the texel; reading u from v0's upper
        // half (the old packing) lands on a different one.
        if (!CompileImage("ImageSampleB", [0x00003C00u, PackHalves(.75f, .25f)], out var shader, a16: true)) return;
        AssertTexel(RunImage(shader), mip: 0, x: 3, y: 1);
    }

    [Fact]
    public void A16SampleDerivativesAreFullDwordsAndOnlyCoordinatesArePacked()
    {
        // image_sample_d A16: v0..v3 = f32 dsdh, dtdh, dsdv, dtdv, v4 = pack(u, v). 0.5 uv/pixel
        // over a 4x4 texture is a two-texel footprint, so the level is 1.
        if (!CompileImage("ImageSampleD", [F(.5f), F(0f), F(0f), F(.5f), PackHalves(.75f, .25f)], out var shader, a16: true)) return;
        AssertTexel(RunImage(shader), mip: 1, x: 1, y: 0);
    }

    [Fact]
    public void UnormSampleAddressesTexelsOfMipZero()
    {
        // image_sample ... unorm at (1.5, 2.5) is texel (1, 2), not normalized coordinates.
        if (!CompileImage("ImageSample", [F(1.5f), F(2.5f)], out var shader, unorm: true)) return;
        AssertTexel(RunImage(shader), mip: 0, x: 1, y: 2);
    }

    [Fact]
    public void UnormCubeSampleFailsTranslation()
    {
        Assert.False(Gen5ImageFixtures.TryCompile("ImageSample", Gen5ImageFixtures.ImageTypeCube, out _, out var error,
            unorm: true));
        Assert.Contains("unnormalized-coordinate", error);
    }

    // ---- helpers ----

    private static Gen5ComputeFixture Wave64Bpermute => new("wave64-bpermute",
    [
        0xD7650000, 0x000100C1, // v_mbcnt_lo_u32_b32 v0, -1, 0
        0xD7660000, 0x000200C1, // v_mbcnt_hi_u32_b32 v0, -1, v0        lane 0..63
        0x7E04028C,             // v_mov_b32 v2, 12                      byte index 3
        0xD8000000u | 0xB3u << 18, 3u << 24 | 0u << 8 | 2u, // ds_bpermute_b32 v3, v2, v0
        0xBF8C0000,             // s_waitcnt 0
        0x34080082,             // v_lshlrev_b32 v4, 2, v0
        0xE0701000, 0x80020304, // buffer_store_dword v3, v4, s[8:11], 0 offen
        SEndpgm,
    ], 8, 256);

    private bool CompileImage(string opcode, uint[] address, out Gen5MslShader shader,
        bool a16 = false, bool unorm = false)
    {
        Assert.True(Gen5ImageFixtures.TryCompile(opcode, Gen5ImageFixtures.ImageType2D, out var compiled, out var error,
            addressValues: address, storeResults: true, a16: a16, unorm: unorm), error);
        shader = compiled!;
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Texture execution requires the exclusive Metal device gate.");
            return false;
        }

        Assert.True(MetalNative.IsAvailable, "The required gate needs a Metal device.");
        return true;
    }

    // A 4x4 two-mip 2D texture whose texels encode mip, x and y.
    private static byte[] RunImage(Gen5MslShader shader)
    {
        var texture = MetalNative.CreateFloatTexture(2, 4, 4, 1, 1, [Texels(0), Texels(1)]);
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var error), error + shader.Source);
        Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[16],
            new byte[PushData.ByteSize], shader.ArgumentLayout!, 1, out var result, out error,
            texture, MetalNative.CreatePointSampler()), error);
        return result;
    }

    private static void AssertTexel(byte[] result, uint mip, uint x, uint y)
    {
        var expected = mip * 10000 + y * 10 + x;
        for (var component = 0; component < 4; component++)
        {
            Assert.Equal(F(expected + component), BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(component * 4)));
        }
    }

    private static byte[] Texels(uint mip)
    {
        var size = 4u >> (int)mip;
        var bytes = new byte[size * size * 16];
        var offset = 0;
        for (uint y = 0; y < size; y++)
        for (uint x = 0; x < size; x++)
        for (uint component = 0; component < 4; component++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(offset), F(mip * 10000 + y * 10 + x + component));
            offset += 4;
        }

        return bytes;
    }

    private static uint F(float value) => BitConverter.SingleToUInt32Bits(value);

    private static uint PackHalves(float low, float high) =>
        BitConverter.HalfToUInt16Bits((Half)low) | ((uint)BitConverter.HalfToUInt16Bits((Half)high) << 16);

    private bool RequireDevice()
    {
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Execution requires the exclusive Metal device gate.");
            return false;
        }

        Assert.True(MetalNative.IsAvailable, "The required gate needs a Metal device.");
        return true;
    }

    private static Gen5ShaderInstruction ReadFirstLane(uint pc, uint scalarDestination, uint vectorSource) =>
        new(pc, Gen5ShaderEncoding.Vop1, "VReadfirstlaneB32", [0u],
            [Gen5Operand.Vector(vectorSource)], [Gen5Operand.Scalar(scalarDestination)], null);

    private static Gen5ShaderInstruction MoveScalar(uint pc, uint destination, uint value) =>
        new(pc, Gen5ShaderEncoding.Sop1, "SMovB32", [0u],
            [Gen5Operand.Source(128 + value)], [Gen5Operand.Scalar(destination)], null);

    private static Gen5ShaderInstruction ScalarBufferLoad(uint pc, uint resourceRegister, uint destination, uint count) =>
        new(pc, Gen5ShaderEncoding.Smem, $"SBufferLoadDwordx{count}", [0u, 0u],
            [Gen5Operand.Scalar(resourceRegister), Gen5Operand.Source(125)],
            [.. Enumerable.Range((int)destination, (int)count).Select(index => Gen5Operand.Scalar((uint)index))],
            new Gen5ScalarMemoryControl(count, 0, null));
}
