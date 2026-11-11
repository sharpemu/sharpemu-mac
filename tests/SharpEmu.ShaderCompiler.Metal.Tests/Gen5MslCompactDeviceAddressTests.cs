// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// CompactMemoryAccessPolicy (#1138) marks a compute program that reads many dwords through device addresses
// UsesDeviceAddressFallback, and its binding layout then holds DescriptorBindingKind.DeviceAddressFallback. Before
// Metal named that kind, the argument layout took it for an image class and the translation threw.
public sealed class Gen5MslCompactDeviceAddressTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(CompactMemoryAccessPolicy.DefaultThreshold - 1)]
    [InlineData(CompactMemoryAccessPolicy.DefaultThreshold)]
    [InlineData(2000)]
    public void ComputeProgramWithManyDeviceAddressDwordsCompiles(int dwords)
    {
        var fixture = CopyFixture(dwords);
        var plan = ShaderResourcePlan.Extract(Gen5ComputeFixtures.DecodeOrThrow(fixture), ShaderStage.Compute, 0, 0, 16);
        Assert.Equal(dwords, CompactMemoryAccessPolicy.CountDeviceAddressDwords(plan));
        var selected = CompactMemoryAccessPolicy.Prefers(plan, ShaderStage.Compute);
        Assert.Equal(dwords >= CompactMemoryAccessPolicy.DefaultThreshold, selected);
        var request = Gen5ComputeFixtures.CreateComputeRequest(fixture);
        Assert.Equal(selected, request.Resources.Info.UsesDeviceAddressFallback);
        Assert.Equal(selected, request.Bindings.Find(DescriptorBindingKind.DeviceAddressFallback) is not null);

        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.Contains("address_ranges", shader.Source);
        // MSL reads no fallback address: the binding has no argument field. It moves the single-dword helpers out
        // of line; the range lookup a multi-dword record shares stays inline.
        Assert.DoesNotContain(shader.ArgumentLayout!.Fields, field => field.Kind == DescriptorBindingKind.DeviceAddressFallback);
        foreach (var helper in new[] { "uint sharpemu_load_device_dword(", "void sharpemu_store_device_dword(" })
            Assert.Contains((selected ? "__attribute__((noinline)) static " : "static inline ") + helper, shader.Source);
        Assert.Contains("static inline device uint* sharpemu_resolve_device_address(", shader.Source);
        if (MetalNative.IsAvailable)
            Assert.True(MetalNative.TryCompileLibrary(shader.Source, out _, out error), error);
    }

    // The guest range: two pages at GuestBase (range 0); the page after it is a hole; then two one-page ranges (1 and 2)
    // that both alias the first page of the guest buffer (MetalNative.CreateGuestMapping with two trailing ranges).
    private const ulong GuestBase = 0x4000_0000;
    private const int GuestBytes = 0x8000;
    private const ulong PageBytes = 0x4000;
    private const ulong InputBase = GuestBase + 0x100;                     // s[16:17], range 0, first page
    private const ulong OutputBase = GuestBase + PageBytes + 0x200;        // s[18:19], range 0, second page
    private const ulong RangeEnd = GuestBase + GuestBytes - 4;             // s[20:21], last dword of range 0; +4 is the hole
    private const ulong CrossBase = GuestBase + 3 * PageBytes + 0x3FFC;    // s[22:23], last dword of range 1; +4 is range 2
    private const ulong RangeOne = GuestBase + 3 * PageBytes;              // s[24:25], range 1

    // (destination vgpr, base sgpr, offset): each load of the program, in order. The range alternates between loads, so
    // each out-of-line call starts from a range cache the previous call left on another range.
    private static readonly (uint Register, uint Base, uint Offset)[] Loads =
    [
        (2, 16, 0),   // range 0
        (3, 22, 0),   // range 1, its last dword
        (4, 16, 4),   // range 0 again
        (5, 22, 4),   // the next dword: range 2
        (6, 20, 0),   // the last dword of range 0
        (7, 20, 4),   // the hole after it: reads zero
        (8, 22, 8),   // range 2
    ];

    // (source vgpr, base sgpr, offset): each store, after the loads. None writes a word a load reads, so lanes that
    // run the program at once agree on every value.
    private static readonly (uint Register, uint Base, uint Offset)[] Stores =
    [
        (2, 18, 0), (3, 18, 4), (4, 18, 8), (5, 18, 12), (6, 18, 16), (7, 18, 20), (8, 18, 24),
        (2, 20, 4),     // the hole: dropped
        (4, 22, 0x40),  // range 2
        (6, 24, 0x80),  // range 1
    ];

    // The same program and range table, compiled with the single-dword helpers static inline (no DeviceAddressFallback
    // binding) and out of line (the binding present), writes the same guest memory, and that memory is the one the
    // CPU model below writes: zero for the hole, a dropped store there, and range 1 and 2 aliasing the first page.
    [Fact]
    public void OutOfLineDeviceAddressHelpersWriteWhatTheInlineOnesWrite()
    {
        var fixture = new Gen5ComputeFixture("device-address-ranges", RangeProgram(), 0, 0);
        var (inlineRequest, inline) = Compile(fixture, outOfLine: false);
        var (outOfLineRequest, outOfLine) = Compile(fixture, outOfLine: true);
        Assert.NotEmpty(inlineRequest.WrittenRangeSlotByMemoryIndex);
        foreach (var helper in new[] { "uint sharpemu_load_device_dword(", "void sharpemu_store_device_dword(" })
        {
            Assert.Contains("static inline " + helper, inline.Source);
            Assert.Contains("__attribute__((noinline)) static " + helper, outOfLine.Source);
        }

        Assert.Contains("sharpemu_load_device_dword(sharpemu_ranges", inline.Source);
        Assert.Contains("sharpemu_store_device_dword(sharpemu_ranges", inline.Source);
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Execution requires the exclusive Metal device gate.");
            return;
        }

        Assert.True(MetalNative.IsAvailable, "The required execution gate needs a Metal device.");
        var guest = new byte[GuestBytes];
        for (var word = 0; word < GuestBytes / 4; word++)
            BinaryPrimitives.WriteUInt32LittleEndian(guest.AsSpan(word * 4), 0x9E37_0000u ^ ((uint)word * 0x0101_0101u));

        var expected = Model(guest);
        var inlineWords = Execute(inlineRequest, inline, guest);
        var outOfLineWords = Execute(outOfLineRequest, outOfLine, guest);
        for (var word = 0; word < expected.Length; word++)
        {
            if (inlineWords[word] != outOfLineWords[word] || inlineWords[word] != expected[word])
                output.WriteLine($"word 0x{word * 4:X4}: cpu 0x{expected[word]:X8} inline 0x{inlineWords[word]:X8} out-of-line 0x{outOfLineWords[word]:X8}");
        }

        Assert.Equal(inlineWords, outOfLineWords);
        Assert.Equal(expected, inlineWords);
        // The program did write: every store but the dropped one changed a word.
        Assert.Equal(Stores.Length - 1, Enumerable.Range(0, expected.Length).Count(word => expected[word] != BinaryPrimitives.ReadUInt32LittleEndian(guest.AsSpan(word * 4))));
    }

    private static (ShaderCompileRequest Request, Gen5MslShader Shader) Compile(Gen5ComputeFixture fixture, bool outOfLine)
    {
        var program = Gen5ComputeFixtures.DecodeOrThrow(fixture);
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, 0x1138, 0, 16);
        Assert.True(plan.Info.UsesDeviceAddresses);
        Assert.False(plan.Info.UsesDeviceAddressFallback);
        plan.Info.UsesDeviceAddressFallback = outOfLine;
        var resources = ResourceMaterializer.ApplyTo(plan, ResourceSpecialization.Default(plan.Info));
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(program, 0, 16, 0),
            BindingLayout.UsesGlobalDataShare(program), ShaderCompileRequest.RequiresFlattenedTable(plan, resources),
            BindingLayout.ReadsShaderBase(program));
        Assert.Equal(outOfLine, layout.Find(DescriptorBindingKind.DeviceAddressFallback) is not null);
        var request = new ShaderCompileRequest(plan, resources, layout) { WaveSize = 32, LocalSizeX = 32 };
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        return (request, shader);
    }

    private static uint[] Execute(ShaderCompileRequest request, Gen5MslShader shader, byte[] guest)
    {
        var (mapping, _) = MetalNative.CreateGuestMapping(GuestBase, guest, 0x1000, trailingRanges: 2);
        Assert.Equal(3u, mapping[0].Count);
        // A store is made only inside the written range the host names in the flattened table: every store's names
        // all five pages, the hole included, so the range table alone decides what a store reaches.
        var slots = request.WrittenRangeSlotByMemoryIndex.Values;
        var flattened = new uint[Math.Max(Math.Max(request.FlattenedTableReservedWords, shader.FlattenedTableMinimumWords), slots.Max() + 3)];
        foreach (var slot in slots)
        {
            flattened[slot] = unchecked((uint)GuestBase);
            flattened[slot + 1] = (uint)(GuestBase >> 32);
            flattened[slot + 2] = (uint)(5 * PageBytes);
        }

        var bytes = new byte[flattened.Length * sizeof(uint)];
        Buffer.BlockCopy(flattened, 0, bytes, 0, bytes.Length);
        var tables = mapping.Append(MetalNative.CreateTable(DescriptorBindingKind.FlattenedResourceTable, bytes, (uint)flattened.Length)).ToArray();
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out var compileError), compileError);
        try
        {
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[64],
                new byte[PushData.ByteSize], shader.ArgumentLayout!, 32, out _, out _, out var runError, tables: tables), runError);
            return MetalNative.ReadWords(mapping[0].Residents[0], GuestBytes / 4);
        }
        finally
        {
            MetalNative.ReleaseCollisionLibrary(library);
        }
    }

    // The guest memory after the program, on the CPU: an address of range 0 is its own offset into the buffer, one of
    // range 1 or 2 an offset into the first page, one in the hole reads zero and drops its store.
    private static uint[] Model(byte[] guest)
    {
        var memory = new uint[GuestBytes / 4];
        for (var word = 0; word < memory.Length; word++)
            memory[word] = BinaryPrimitives.ReadUInt32LittleEndian(guest.AsSpan(word * 4));

        int? Index(ulong address)
        {
            if (address >= GuestBase && address < GuestBase + GuestBytes)
                return (int)((address - GuestBase) / 4);
            if (address >= GuestBase + 3 * PageBytes && address < GuestBase + 5 * PageBytes)
                return (int)(((address - GuestBase) % PageBytes) / 4);
            return null;
        }

        var registers = new Dictionary<uint, uint>();
        foreach (var (register, baseRegister, offset) in Loads)
            registers[register] = Index(BaseAddress(baseRegister) + offset) is { } index ? memory[index] : 0u;
        foreach (var (register, baseRegister, offset) in Stores)
        {
            if (Index(BaseAddress(baseRegister) + offset) is { } index)
                memory[index] = registers[register];
        }

        return memory;
    }

    private static ulong BaseAddress(uint register) => register switch
    {
        16 => InputBase,
        18 => OutputBase,
        20 => RangeEnd,
        22 => CrossBase,
        24 => RangeOne,
        _ => throw new ArgumentOutOfRangeException(nameof(register)),
    };

    // s_mov_b32 of each base pair, v1 = 0, then the loads and stores (GLOBAL_LOAD_DWORD / GLOBAL_STORE_DWORD, saddr + v1).
    private static uint[] RangeProgram()
    {
        var words = new List<uint>();
        foreach (var register in new uint[] { 16, 18, 20, 22, 24 })
        {
            var address = BaseAddress(register);
            words.AddRange([0xBE80_03FFu | (register << 16), unchecked((uint)address), 0xBE80_03FFu | ((register + 1) << 16), (uint)(address >> 32)]);
        }

        words.Add(0x7E02_0280u);
        foreach (var (register, baseRegister, offset) in Loads)
            words.AddRange([GlobalWord(12, offset), (register << 24) | (baseRegister << 16) | 1u]);
        foreach (var (register, baseRegister, offset) in Stores)
            words.AddRange([GlobalWord(28, offset), (baseRegister << 16) | (register << 8) | 1u]);
        words.Add(0xBF81_0000u);
        return [.. words];
    }

    // Loads dwords - dwords / 2 dwords through GLOBAL_LOAD_DWORD and stores dwords / 2 of them through
    // GLOBAL_STORE_DWORD (s[16:17] and s[18:19] + v1): dwords device-address dwords in all.
    internal static Gen5ComputeFixture CopyFixture(int dwords)
    {
        var words = new List<uint>();
        var copies = dwords / 2;
        for (var index = 0; index < dwords - copies; index++)
        {
            var register = 8u + (uint)(index % 32);
            var offset = (uint)(index * 4) & 0x7FCu;
            words.Add(GlobalWord(12, offset));
            words.Add((register << 24) | (16u << 16) | 1u);
            if (index < copies)
            {
                words.Add(GlobalWord(28, offset));
                words.Add((18u << 16) | (register << 8) | 1u);
            }
        }

        words.Add(0xBF81_0000u);
        return new Gen5ComputeFixture($"device-address-copy-{dwords}", [.. words], 0, 0);
    }

    private static uint GlobalWord(uint opcode, uint offset) => 0xDC00_8000u | (opcode << 18) | offset;
}
