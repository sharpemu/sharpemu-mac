// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// Two rewrites that share one proof across the dwords of a vector access:
//  - DeviceSpanAccess: a GLOBAL/FLAT load or store of 2..4 dwords resolves the guest
//    address once (sharpemu_resolve_device_span) instead of walking the page table per dword;
//  - RawDwordRuns: a raw BUFFER load or store of 2..4 dwords is one bounds proof
//    (sharpemu_load_dword_run / sharpemu_store_dword_run).
// Each is run against the per-dword form (the option off) and against the value the ISA
// gives, including a span that leaves its 16 KB page and a run that ends past the binding.
public sealed class Gen5MslMemoryRunTests(ITestOutputHelper output)
{
    private const int Lanes = 16;
    private const int BufferBytes = 1024;
    private const int OutputOffset = 512;
    private const ulong GuestBase = 0x4000_0000;
    private const int GuestBytes = 1 << 14;

    private static bool DeviceRequired => Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") == "1";

    private static readonly Gen5MslCodegenOptions On = new();
    private static readonly Gen5MslCodegenOptions Off = new(DeviceSpanAccess: false, RawDwordRuns: false);

    // global_load_dwordx2/x4/x3 and global_store_dwordx2/x4/x3 (dword count, opcode).
    public static TheoryData<uint, uint> GlobalLoads => new() { { 2u, 13u }, { 4u, 14u }, { 3u, 15u } };
    public static TheoryData<uint, uint> GlobalStores => new() { { 2u, 29u }, { 4u, 30u }, { 3u, 31u } };
    // buffer_load_dwordx2/x4/x3 and buffer_store_dwordx2/x4/x3.
    public static TheoryData<uint, uint> BufferLoads => new() { { 2u, 13u }, { 4u, 14u }, { 3u, 15u } };
    public static TheoryData<uint, uint> BufferStores => new() { { 2u, 29u }, { 4u, 30u }, { 3u, 31u } };

    [Theory]
    [MemberData(nameof(GlobalLoads))]
    public void AWideGlobalLoadResolvesItsSpanOnce(uint dwords, uint opcode)
    {
        var fixture = GlobalLoadProgram(opcode);
        var on = Compile(fixture, On, out _);
        var off = Compile(fixture, Off, out _);
        Assert.Equal(1, CountOccurrences(on.Source, "sharpemu_resolve_device_span(sharpemu_ranges"));
        Assert.Equal(0, CountOccurrences(off.Source, "sharpemu_resolve_device_span(sharpemu_ranges"));
        // The one walk serves the whole record; each dword keeps a per-dword fallback for a
        // span that crosses its page or is unmapped.
        Assert.Equal((int)dwords, CountOccurrences(on.Source, "sharpemu_load_device_dword(sharpemu_ranges"));
        Assert.Equal((int)dwords, CountOccurrences(off.Source, "sharpemu_load_device_dword(sharpemu_ranges"));
        Assert.Contains($"sharpemu_resolve_device_span(sharpemu_ranges, sharpemu_range_count, sharpemu_faults, sharpemu_fault_words, sharpemu_range_cache, ", on.Source, StringComparison.Ordinal);
        Assert.Contains($"{dwords * 4}u, ", on.Source, StringComparison.Ordinal);
    }

    [Theory]
    [MemberData(nameof(GlobalStores))]
    public void AWideGlobalStoreResolvesItsSpanOnce(uint dwords, uint opcode)
    {
        var fixture = GlobalStoreProgram(opcode);
        var on = Compile(fixture, On, out _);
        var off = Compile(fixture, Off, out _);
        Assert.Equal(1, CountOccurrences(on.Source, "sharpemu_resolve_device_span(sharpemu_ranges"));
        Assert.Equal((int)dwords, CountOccurrences(on.Source, "sharpemu_store_device_dword(sharpemu_ranges"));
        Assert.Equal((int)dwords, CountOccurrences(off.Source, "sharpemu_store_device_dword(sharpemu_ranges"));
    }

    [Theory]
    [MemberData(nameof(BufferLoads))]
    public void AWideBufferLoadIsOneRun(uint dwords, uint opcode)
    {
        var fixture = BufferLoadProgram(opcode);
        var on = Compile(fixture, On, out _);
        var off = Compile(fixture, Off, out _);
        Assert.Equal(1, CountOccurrences(on.Source, $"sharpemu_load_dword_run<{dwords}>(b0,"));
        Assert.Equal(0, CountOccurrences(off.Source, "sharpemu_load_dword_run<"));
        // The wide load's own per-dword calls are gone; the narrow ones around it are not.
        Assert.Equal(CountOccurrences(off.Source, "sharpemu_load_dword(b0,") - (int)dwords, CountOccurrences(on.Source, "sharpemu_load_dword(b0,"));
    }

    [Theory]
    [MemberData(nameof(BufferStores))]
    public void AWideBufferStoreIsOneRun(uint dwords, uint opcode)
    {
        var fixture = BufferStoreProgram(opcode, 1020);
        var on = Compile(fixture, On, out _);
        var off = Compile(fixture, Off, out _);
        Assert.Equal(1, CountOccurrences(on.Source, $"sharpemu_store_dword_run<{dwords}>(b0,"));
        Assert.Equal((int)dwords, CountOccurrences(off.Source, "sharpemu_store_dword(b0,"));
        Assert.Equal(0, CountOccurrences(on.Source, "sharpemu_store_dword(b0,"));
    }

    [Fact]
    public void TheSwitchesGoBackToPerDwordAccesses()
    {
        var load = Compile(BufferLoadProgram(14), new Gen5MslCodegenOptions(RawDwordRuns: false), out _);
        Assert.DoesNotContain("sharpemu_load_dword_run<", load.Source, StringComparison.Ordinal);
        var global = Compile(GlobalLoadProgram(14), new Gen5MslCodegenOptions(DeviceSpanAccess: false), out _);
        Assert.DoesNotContain("sharpemu_resolve_device_span(sharpemu_ranges", global.Source, StringComparison.Ordinal);
    }

    // s_buffer_load_dwordx2/x4/x8/x16 of a bound buffer (SMEM op, runs of up to four dwords).
    [Theory]
    [InlineData(9u, 1, "2")]
    [InlineData(0xAu, 1, "4")]
    [InlineData(0xBu, 2, "4")]
    [InlineData(0xCu, 4, "4")]
    public void AWideScalarBufferLoadIsRunsOfUpToFourDwords(uint opcode, int runs, string width)
    {
        var fixture = ScalarBufferLoadProgram(opcode, 0x10);
        var on = Compile(fixture, On, out _);
        var off = Compile(fixture, Off, out _);
        Assert.Equal(runs, CountOccurrences(on.Source, $"sharpemu_load_dword_run<{width}>(b0,"));
        Assert.Equal(0, CountOccurrences(off.Source, "sharpemu_load_dword_run<"));
    }

    [Fact]
    public void AScalarBufferLoadRunKeepsTheValidDwordsOfARunPastTheEnd()
    {
        if (!RequireDevice())
        {
            return;
        }

        var input = new byte[BufferBytes];
        for (var word = 0; word < BufferBytes / 4; word++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(input.AsSpan(word * 4), 0xC000_0000u + (uint)word);
        }

        // Inside the buffer, then two dwords inside and two past the end.
        foreach (var (offset, first) in new[] { (0x10u, 4), (1016u, 254) })
        {
            foreach (var options in new[] { On, Off })
            {
                var result = ExecuteBound(ScalarBufferLoadProgram(0xAu, offset), options, input);
                for (var lane = 0; lane < Lanes; lane++)
                {
                    for (var dword = 0; dword < 4; dword++)
                    {
                        var source = first + dword;
                        var expected = source < BufferBytes / 4 ? 0xC000_0000u + (uint)source : 0u;
                        var actual = Word(result, (OutputOffset + lane * 16) / 4 + dword);
                        Assert.True(expected == actual, $"{(options == On ? "on" : "off")} offset {offset}: lane {lane} dword {dword} read 0x{actual:X8}, expected 0x{expected:X8}");
                    }
                }
            }
        }
    }

    [Fact]
    public void ADwordLoadStaysSingle()
    {
        var shader = Compile(BufferLoadProgram(12), On, out _);
        Assert.DoesNotContain("sharpemu_load_dword_run<", shader.Source, StringComparison.Ordinal);
    }

    [Theory]
    [MemberData(nameof(GlobalLoads))]
    public void AGlobalLoadReadsItsSpanAndFallsBackAcrossThePage(uint dwords, uint opcode)
    {
        if (!RequireDevice())
        {
            return;
        }

        var fixture = GlobalLoadProgram(opcode);
        var guest = GuestPattern();
        foreach (var options in new[] { On, Off })
        {
            var result = ExecuteGlobal(fixture, options, guest, out _);
            for (var lane = 0; lane < Lanes; lane++)
            {
                for (var dword = 0; dword < 4; dword++)
                {
                    var address = GlobalBase + (ulong)(lane * 16 + dword * 4);
                    var inPage = dword < dwords && address + 4 <= GuestBase + GuestBytes;
                    var expected = inPage ? Word(guest, (int)(address - GuestBase) / 4) : 0u;
                    var actual = Word(result, (OutputOffset + lane * 16) / 4 + dword);
                    if (dword >= dwords)
                    {
                        // A register the load does not write keeps its initial value in both forms.
                        continue;
                    }

                    Assert.True(expected == actual, $"{(options == On ? "on" : "off")}: lane {lane} dword {dword} read 0x{actual:X8}, expected 0x{expected:X8}");
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(BufferLoads))]
    public void ABufferRunLoadKeepsTheValidDwordsOfARunPastTheEnd(uint dwords, uint opcode)
    {
        if (!RequireDevice())
        {
            return;
        }

        var fixture = BufferLoadProgram(opcode);
        var input = new byte[BufferBytes];
        for (var word = 0; word < BufferBytes / 4; word++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(input.AsSpan(word * 4), 0xC000_0000u + (uint)word);
        }

        foreach (var options in new[] { On, Off })
        {
            var result = ExecuteBound(fixture, options, input);
            for (var lane = 0; lane < Lanes; lane++)
            {
                for (var dword = 0; dword < dwords; dword++)
                {
                    var source = 252 + lane + dword;
                    var expected = source < BufferBytes / 4 ? 0xC000_0000u + (uint)source : 0u;
                    var actual = Word(result, (OutputOffset + lane * 16) / 4 + dword);
                    Assert.True(expected == actual, $"{(options == On ? "on" : "off")}: lane {lane} dword {dword} read 0x{actual:X8}, expected 0x{expected:X8}");
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(BufferStores))]
    public void ABufferRunStoreWritesTheValidDwordsOfARunPastTheEnd(uint dwords, uint opcode)
    {
        if (!RequireDevice())
        {
            return;
        }

        // Every lane stores the same four constants at byte 1020: only dword 255 is inside.
        var fixture = BufferStoreProgram(opcode, 1020);
        var input = new byte[BufferBytes];
        for (var word = 0; word < BufferBytes / 4; word++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(input.AsSpan(word * 4), 0xC000_0000u + (uint)word);
        }

        foreach (var options in new[] { On, Off })
        {
            var result = ExecuteBound(fixture, options, input);
            Assert.Equal(0x1111_1111u, Word(result, 255));
            for (var word = 0; word < 255; word++)
            {
                Assert.Equal(0xC000_0000u + (uint)word, Word(result, word));
            }
        }

        // And a run inside the binding writes all of its dwords.
        var inside = BufferStoreProgram(opcode, 512);
        foreach (var options in new[] { On, Off })
        {
            var result = ExecuteBound(inside, options, input);
            for (var dword = 0; dword < 4; dword++)
            {
                var expected = dword < dwords ? 0x1111_1111u * (uint)(dword + 1) : 0xC000_0000u + 128u + (uint)dword;
                Assert.Equal(expected, Word(result, 128 + dword));
            }
        }
    }

    // ---- programs ----

    // The lane's records sit at GlobalBase + 16 * lane; GlobalBase leaves 120 bytes of the
    // one mapped page, so lane 7's record straddles its end and every later lane is past it.
    private static ulong GlobalBase => GuestBase + GuestBytes - 120;

    private static uint StoredValue(int lane, int dword) => 0xA000_0000u + (uint)(dword * 0x100 + lane);

    private static Gen5ComputeFixture GlobalLoadProgram(uint opcode) => new("global-wide-load",
    [
        0x34020084,                                  // v_lshlrev_b32 v1, 4, v0
        0x4A1202FF, (uint)GlobalBase,                // v_add_nc_u32 v9, GlobalBase, v1
        0x7E140280,                                  // v_mov_b32 v10, 0
        0xDC008000u | opcode << 18, 0x047D0009,      // global_load_dwordxN v[4:], v[9:10], off
        0xBF8C3F70,                                  // s_waitcnt vmcnt(0)
        0x34060084,                                  // v_lshlrev_b32 v3, 4, v0
        0xE0781200, 0x80020403,                      // buffer_store_dwordx4 v[4:7], v3, s[8:11], 0 offen offset:512
        0xBF810000,                                  // s_endpgm
    ], 8, BufferBytes);

    private static Gen5ComputeFixture GlobalStoreProgram(uint opcode) => new("global-wide-store",
    [
        0x34020084,                                  // v_lshlrev_b32 v1, 4, v0
        0x4A1202FF, (uint)GlobalBase,                // v_add_nc_u32 v9, GlobalBase, v1
        0x7E140280,                                  // v_mov_b32 v10, 0
        0x4A0802FF, 0xA000_0000,                     // v_add_nc_u32 v4, 0xA0000000, v0
        0x4A0A02FF, 0xA000_0100,                     // v_add_nc_u32 v5, 0xA0000100, v0
        0x4A0C02FF, 0xA000_0200,                     // v_add_nc_u32 v6, 0xA0000200, v0
        0x4A0E02FF, 0xA000_0300,                     // v_add_nc_u32 v7, 0xA0000300, v0
        0xDC008000u | opcode << 18, 0x007D0409,      // global_store_dwordxN v[9:10], v[4:], off
        0xBF8C3F70,                                  // s_waitcnt
        0xBF810000,                                  // s_endpgm
    ], 8, BufferBytes);

    // Lane L loads the run at byte 1008 + 4 * L of the bound buffer (a V# of 1024 bytes),
    // so lane 0..3 are inside and lane 4.. run past the end by one dword, two, ... or wholly.
    private static Gen5ComputeFixture BufferLoadProgram(uint opcode) => new("buffer-wide-load",
    [
        0x34020082,                                  // v_lshlrev_b32 v1, 2, v0
        0x4A0202FF, 1008,                            // v_add_nc_u32 v1, 1008, v1
        0xE0001000u | opcode << 18, 0x80020401,      // buffer_load_dwordN v[4:], v1, s[8:11], 0 offen
        0xBF8C3F70,                                  // s_waitcnt vmcnt(0)
        0x34060084,                                  // v_lshlrev_b32 v3, 4, v0
        0xE0781200, 0x80020403,                      // buffer_store_dwordx4 v[4:7], v3, s[8:11], 0 offen offset:512
        0xBF810000,                                  // s_endpgm
    ], 8, BufferBytes);

    // s_buffer_load_dwordxN s[12:], s[8:11], offset; the first four loaded dwords are stored
    // at 512 + 16 * lane (an x2 load leaves the other two registers at their initial value).
    private static Gen5ComputeFixture ScalarBufferLoadProgram(uint opcode, uint offset) => new("scalar-buffer-wide-load",
    [
        0xF4000000u | opcode << 18 | 12u << 6 | 4u, 0xFA000000u | offset,  // s_buffer_load_dwordxN s[12:], s[8:11], offset
        0xBF8CC07F,                                  // s_waitcnt lgkmcnt(0)
        0x7E08020C,                                  // v_mov_b32 v4, s12
        0x7E0A020D,                                  // v_mov_b32 v5, s13
        0x7E0C020E,                                  // v_mov_b32 v6, s14
        0x7E0E020F,                                  // v_mov_b32 v7, s15
        0x34060084,                                  // v_lshlrev_b32 v3, 4, v0
        0xE0781200, 0x80020403,                      // buffer_store_dwordx4 v[4:7], v3, s[8:11], 0 offen offset:512
        0xBF810000,                                  // s_endpgm
    ], 8, BufferBytes);

    // Every lane stores the constants 0x11111111 .. 0x44444444 at the same byte address.
    private static Gen5ComputeFixture BufferStoreProgram(uint opcode, uint address) => new("buffer-wide-store",
    [
        0x7E0802FF, 0x1111_1111,                     // v_mov_b32 v4, 0x11111111
        0x7E0A02FF, 0x2222_2222,                     // v_mov_b32 v5, 0x22222222
        0x7E0C02FF, 0x3333_3333,                     // v_mov_b32 v6, 0x33333333
        0x7E0E02FF, 0x4444_4444,                     // v_mov_b32 v7, 0x44444444
        0x7E0602FF, address,                         // v_mov_b32 v3, address
        0xE0001000u | opcode << 18, 0x80020403,      // buffer_store_dwordN v[4:], v3, s[8:11], 0 offen
        0xBF810000,                                  // s_endpgm
    ], 8, BufferBytes);

    // ---- harness ----

    private static byte[] GuestPattern()
    {
        var guest = new byte[GuestBytes];
        for (var word = 0; word < GuestBytes / 4; word++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(guest.AsSpan(word * 4), 0xB000_0000u + (uint)word);
        }

        return guest;
    }

    private static Gen5MslShader Compile(Gen5ComputeFixture fixture, Gen5MslCodegenOptions options, out ShaderCompileRequest request)
    {
        var userData = new uint[16];
        userData[8] = 0x200000;
        userData[10] = BufferBytes;
        userData[11] = DescriptorConstants.IdentityDestinationSelect;
        request = Gen5ComputeFixtures.RequestOrThrow(Gen5ComputeFixtures.DecodeOrThrow(fixture), ShaderStage.Compute,
            localSizeX: Lanes, runtimeInputs: new ResourceRuntimeInputs { UserData = userData });
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error, options), error);
        if (MetalNative.IsAvailable)
        {
            Assert.True(MetalNative.TryCompileLibrary(shader!.Source, out var library, out var compileError), compileError + "\n" + shader.Source);
            if (library != 0) MetalNative.ReleaseCollisionLibrary(library);
        }

        return shader!;
    }

    private static byte[] ExecuteBound(Gen5ComputeFixture fixture, Gen5MslCodegenOptions options, byte[] initial)
    {
        var shader = Compile(fixture, options, out _);
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
        try
        {
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, initial,
                new byte[PushData.ByteSize], shader.ArgumentLayout!, Lanes, out var result, out _, out var runError), runError);
            return result;
        }
        finally
        {
            MetalNative.ReleaseCollisionLibrary(library);
        }
    }

    private static byte[] ExecuteGlobal(Gen5ComputeFixture fixture, Gen5MslCodegenOptions options, byte[] guest, out byte[] guestAfter)
    {
        var shader = Compile(fixture, options, out _);
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
        try
        {
            var (tables, _) = MetalNative.CreateGuestMapping(GuestBase, guest, 0x1000);
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[BufferBytes],
                new byte[PushData.ByteSize], shader.ArgumentLayout!, Lanes, out var result, out _, out var runError, tables: tables), runError);
            var memory = MetalNative.ReadWords(tables[0].Residents[0], GuestBytes / 4);
            guestAfter = new byte[GuestBytes];
            Buffer.BlockCopy(memory, 0, guestAfter, 0, GuestBytes);
            return result;
        }
        finally
        {
            MetalNative.ReleaseCollisionLibrary(library);
        }
    }

    private static uint Word(byte[] bytes, int index) => BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(index * 4));

    private bool RequireDevice()
    {
        if (!DeviceRequired)
        {
            output.WriteLine("Execution requires the exclusive Metal device gate.");
            return false;
        }

        Assert.True(MetalNative.IsAvailable, "The required execution gate needs a Metal device.");
        return true;
    }

    private static int CountOccurrences(string text, string value)
    {
        var count = 0;
        for (var index = text.IndexOf(value, StringComparison.Ordinal); index >= 0; index = text.IndexOf(value, index + value.Length, StringComparison.Ordinal))
        {
            count++;
        }

        return count;
    }
}
