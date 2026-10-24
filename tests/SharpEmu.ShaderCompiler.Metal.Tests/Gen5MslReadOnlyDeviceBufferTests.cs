// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

public sealed class Gen5MslReadOnlyDeviceBufferTests
{
    private const ulong GuestBase = 0x40000000;

    [Theory]
    [InlineData(false, false, 0, 0x12345678u, 8)]
    [InlineData(true, false, 0, 0x12345678u, 8)]
    [InlineData(false, false, 4, 0x87654321u, 8)]
    [InlineData(true, false, 4, 0x87654321u, 8)]
    [InlineData(false, false, 8, 0u, 8)]
    [InlineData(true, false, 8, 0u, 8)]
    [InlineData(false, true, 0, 0x12345678u, 8)]
    [InlineData(true, true, 0, 0x12345678u, 8)]
    [InlineData(false, true, 4, 0x87654321u, 8)]
    [InlineData(true, true, 4, 0x87654321u, 8)]
    [InlineData(false, true, 8, 0u, 8)]
    [InlineData(true, true, 8, 0u, 8)]
    [InlineData(false, true, 4, 0u, 4)]
    [InlineData(true, true, 4, 0u, 4)]
    public void CapturedDescriptorsReadTheSameBytesAndRespectTheirEnd(bool enabled, bool vector, int offset, uint expected, int byteLimit)
    {
        var program = new Gen5ShaderProgram(0x2000, [
            vector ? new(0, Gen5ShaderEncoding.Mubuf, "BufferLoadDword", [0u, 0u],
                [Gen5Operand.Vector(0), Gen5Operand.Scalar(8), Gen5Operand.Source(125)], [Gen5Operand.Vector(1)],
                new Gen5BufferMemoryControl(1, 0, 1, 8, offset, false, false, false, false)) :
            new(0, Gen5ShaderEncoding.Smem, "SBufferLoadDword", [0u, 0u],
                [Gen5Operand.Scalar(8), Gen5Operand.Source(125)], [Gen5Operand.Scalar(20)],
                new Gen5ScalarMemoryControl(1, offset, null)),
            new(8, Gen5ShaderEncoding.Vop1, "VMovB32", [0u],
                [vector ? Gen5Operand.Vector(1) : Gen5Operand.Scalar(20)], [Gen5Operand.Vector(1)], null),
            new(12, Gen5ShaderEncoding.Vop1, "VMovB32", [0u], [Gen5Operand.Source(128)], [Gen5Operand.Vector(2)], null),
            new(16, Gen5ShaderEncoding.Ds, "DsWriteB32", [0u],
                [Gen5Operand.Vector(2), Gen5Operand.Vector(1)], [], new Gen5DataShareControl(0, 0, true)),
            new(20, Gen5ShaderEncoding.Sopp, "SEndpgm", [0xBF810000u], [], [], null),
        ]);
        var words = new uint[16];
        words[8] = (uint)GuestBase;
        words[10] = 8;
        words[11] = DescriptorConstants.IdentityDestinationSelect | (20u << 12) | (3u << 28);
        var request = Gen5ComputeFixtures.RequestOrThrow(program, ShaderStage.Compute, localSizeX: 1,
            runtimeInputs: new ResourceRuntimeInputs { UserData = words }, readOnlyDeviceBuffers: enabled);
        Assert.Equal(enabled, request.Resources.Info.ReadOnlyDeviceBuffers);
        Assert.Equal(enabled, request.Bindings.Find(DescriptorBindingKind.Buffers) is null);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        if (enabled)
        {
            Assert.DoesNotContain("sharpemu_resources.buffers[", shader.Source, StringComparison.Ordinal);
            Assert.Contains("sharpemu_load_device_dword(sharpemu_ranges", shader.Source, StringComparison.Ordinal);
        }
        if (!MetalNative.IsAvailable)
        {
            Assert.NotEqual("1", Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE"));
            return;
        }
        var data = new byte[1 << 14];
        BinaryPrimitives.WriteUInt32LittleEndian(data, 0x12345678u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(4), 0x87654321u);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(8), 0xDEADBEEFu);
        var (mapping, _) = MetalNative.CreateGuestMapping(GuestBase, data, 8);
        var output = MetalNative.CreateTable(DescriptorBindingKind.GlobalDataShare, new byte[16], 4);
        var push = new byte[PushData.ByteSize];
        for (var i = 0; i < request.Bindings.UserDataRegisters.Count; i++)
            BinaryPrimitives.WriteUInt32LittleEndian(push.AsSpan((int)(request.Bindings.PushDataStartDword + i) * 4),
                words[request.Bindings.UserDataRegisters[i]]);
        if (enabled)
            BinaryPrimitives.WriteUInt32LittleEndian(push.AsSpan((int)(request.Bindings.PushDataStartDword + request.Bindings.FirstBufferWordDword()) * 4), (uint)byteLimit);
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out error), error);
        try
        {
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, data[..byteLimit], push,
                shader.ArgumentLayout!, 1, out _, out _, out error, tables: [.. mapping, output]), error);
            Assert.Equal(expected, MetalNative.ReadWords(output.Buffer, 1)[0]);
        }
        finally { MetalNative.ReleaseCollisionLibrary(library); }
    }

    [Fact]
    public void AStageWithABufferStoreRetainsItsNativeBindings()
    {
        var program = Gen5ComputeFixtures.DecodeOrThrow(Gen5ComputeFixtures.ExecStore);
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, 0xAB1, 0, 16, readOnlyDeviceBuffers: true);
        Assert.NotEmpty(plan.Info.Buffers);
        Assert.False(plan.Info.ReadOnlyDeviceBuffers);
        Assert.Contains(plan.Info.Buffers, buffer => buffer.Written);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ADescriptorLoadedFromAPointerTableUsesTheCapturedWords(bool enabled)
    {
        var program = new Gen5ShaderProgram(0x2000, [
            new(0, Gen5ShaderEncoding.Smem, "SLoadDwordx4", [0u, 0u],
                [Gen5Operand.Scalar(0), Gen5Operand.Source(125)],
                [Gen5Operand.Scalar(8), Gen5Operand.Scalar(9), Gen5Operand.Scalar(10), Gen5Operand.Scalar(11)],
                new Gen5ScalarMemoryControl(4, 0, null)),
            new(8, Gen5ShaderEncoding.Smem, "SBufferLoadDword", [0u, 0u],
                [Gen5Operand.Scalar(8), Gen5Operand.Source(125)], [Gen5Operand.Scalar(20)],
                new Gen5ScalarMemoryControl(1, 0, null)),
            new(16, Gen5ShaderEncoding.Vop1, "VMovB32", [0u], [Gen5Operand.Scalar(20)], [Gen5Operand.Vector(1)], null),
            new(20, Gen5ShaderEncoding.Vop1, "VMovB32", [0u], [Gen5Operand.Source(128)], [Gen5Operand.Vector(2)], null),
            new(24, Gen5ShaderEncoding.Ds, "DsWriteB32", [0u],
                [Gen5Operand.Vector(2), Gen5Operand.Vector(1)], [], new Gen5DataShareControl(0, 0, true)),
            new(28, Gen5ShaderEncoding.Sopp, "SEndpgm", [0xBF810000u], [], [], null),
        ]);
        var data = new byte[1 << 14];
        BinaryPrimitives.WriteUInt32LittleEndian(data, 0x12345678u);
        uint[] descriptor = [(uint)GuestBase, 0, 8, DescriptorConstants.IdentityDestinationSelect | (20u << 12) | (3u << 28)];
        MemoryMarshal.AsBytes(descriptor.AsSpan()).CopyTo(data.AsSpan(0x100));
        var words = new uint[16];
        words[0] = (uint)GuestBase + 0x100;
        bool Read(ulong address, out uint word)
        {
            word = 0;
            if (address < GuestBase || address - GuestBase > (ulong)data.Length - 4) return false;
            word = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)(address - GuestBase)));
            return true;
        }
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, 0xB1ADCAFE, 0, 16, readOnlyDeviceBuffers: enabled);
        var inputs = new ResourceRuntimeInputs { UserData = words, ReadMemory = Read };
        var specialization = ResourceSpecialization.Default(plan.Info);
        ResourceSnapshot snapshot = null!;
        Assert.True(ResourceMaterializer.Materialize(plan, inputs, ref snapshot, ref specialization));
        Assert.Equal(descriptor, snapshot.Buffers[0]);
        Assert.NotEmpty(snapshot.FlattenedResourceTable);
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(program, 0, 16),
            true, ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false, 0);
        var request = new ShaderCompileRequest(plan, resources, layout) { LocalSizeX = 1, WaveSize = 32 };
        Assert.Equal(enabled, resources.Info.ReadOnlyDeviceBuffers);
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        if (!MetalNative.IsAvailable)
        {
            Assert.NotEqual("1", Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE"));
            return;
        }
        // The later live table no longer describes the captured issue. Neither path may
        // observe it: the descriptor comes from the immutable flattened table instead.
        data.AsSpan(0x100, 16).Clear();
        var (mapping, _) = MetalNative.CreateGuestMapping(GuestBase, data, 8);
        var flat = MetalNative.CreateTable(DescriptorBindingKind.FlattenedResourceTable,
            MemoryMarshal.AsBytes(snapshot.FlattenedResourceTable.AsSpan()).ToArray(), (uint)snapshot.FlattenedResourceTable.Length);
        var output = MetalNative.CreateTable(DescriptorBindingKind.GlobalDataShare, new byte[16], 4);
        var push = new byte[PushData.ByteSize];
        for (var i = 0; i < layout.UserDataRegisters.Count; i++)
            BinaryPrimitives.WriteUInt32LittleEndian(push.AsSpan((int)(layout.PushDataStartDword + i) * 4), words[layout.UserDataRegisters[i]]);
        if (enabled)
            BinaryPrimitives.WriteUInt32LittleEndian(push.AsSpan((int)(layout.PushDataStartDword + layout.FirstBufferWordDword()) * 4), 8);
        Assert.True(MetalNative.TryCompileLibrary(shader.Source, out var library, out error), error);
        try
        {
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, data[..8], push,
                shader.ArgumentLayout!, 1, out _, out _, out error, tables: [.. mapping, flat, output]), error);
            Assert.Equal(0x12345678u, MetalNative.ReadWords(output.Buffer, 1)[0]);
        }
        finally { MetalNative.ReleaseCollisionLibrary(library); }
    }
}
