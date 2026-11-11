// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

// CompactMemoryAccessPolicy (#1138) marks a compute program that reads many dwords through device addresses
// UsesDeviceAddressFallback, and its binding layout then holds DescriptorBindingKind.DeviceAddressFallback. Before
// Metal named that kind, the argument layout took it for an image class and the translation threw.
public sealed class Gen5MslCompactDeviceAddressTests
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
