// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;
using Xunit.Abstractions;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

[CollectionDefinition("Device binding compatibility", DisableParallelization = true)]
public sealed class DeviceBindingCompatibilityCollection { }

[Collection("Device binding compatibility")]
public sealed class DeviceBindingLoadTests(ITestOutputHelper output)
{
    private const ulong GuestBase = 0x1_4000_0000;
    private const uint Raw = DescriptorConstants.IdentityDestinationSelect | (20u << 12) | (3u << 28);

    // A pointer pair and a complete V# are fetched on the device. The host binds only
    // the output V#; neither the old pointer nor any old descriptor byte is flattened.
    [Theory]
    [InlineData(0, 0x12345678u, false)]
    [InlineData(0, 0x12345678u, true)]
    [InlineData(1, 0u, false)] // null V#
    [InlineData(1, 0u, true)]
    [InlineData(2, 0u, false)] // unmapped V# address
    [InlineData(2, 0u, true)]
    [InlineData(3, 0u, false)] // mapped address, zero record count
    [InlineData(3, 0u, true)]
    public void DependentPointerAndDescriptorLoadsReadTheDeviceBytes(int mode, uint expected, bool completeWriterFirst)
    {
        var program = new Gen5ShaderProgram(0x100000, [
            Load(0, 0, 4, 2), Load(8, 4, 12, 4),
            Buffer(16, false, 12), Buffer(24, true, 8),
            new Gen5ShaderInstruction(32, Gen5ShaderEncoding.Sopp, "SEndpgm", [0xBF810000], [], [], null),
        ]);
        var hash = 0xB1D10000u + (uint)mode;
        var wasUnvalidated = ResourceTracker.UnvalidatedAddressWordsAreMemory;
        ResourceTracker.UnvalidatedAddressWordsAreMemory = false;
        ShaderResourcePlan old;
        ShaderResourcePlan plan;
        try
        {
            old = ShaderResourcePlan.Extract(program, ShaderStage.Compute, hash, 0, 12);
            var rootLoads = old.Memory.Entries.Select((memory, index) => (memory, index))
                .Where(pair => pair.memory.Pc == 0).Select(pair => pair.index).ToHashSet();
            plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, hash, 0, 12, deviceBindingMemory: rootLoads);
        }
        finally { ResourceTracker.UnvalidatedAddressWordsAreMemory = wasUnvalidated; }
        Assert.True(DeviceBindingLoads.Accepts(old, plan));
        var user = new uint[12];
        user[0] = unchecked((uint)GuestBase); user[1] = (uint)(GuestBase >> 32);
        user[8] = 0x200000; user[10] = 64; user[11] = Raw;
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, new ResourceRuntimeInputs
        {
            UserData = user,
            ReadMemory = (ulong address, out uint value) => throw new InvalidOperationException($"host read of device binding word {address:X}"),
        }, ref snapshot, ref specialization));
        Assert.Single(snapshot.Buffers);
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(program, 0, 12), false,
            ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false, 0);
        var request = new ShaderCompileRequest(plan, resources, layout) { WaveSize = 32, LocalSizeX = 1 };
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1")
        {
            output.WriteLine("Device output comparison requires SHARPEMU_TEST_REQUIRE_DEVICE=1.");
            return;
        }
        Assert.True(MetalNative.IsAvailable);
        var bytes = new byte[1 << DeviceAddressPaging.PageBits];
        BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(1024), GuestBase + 128);
        var dataAddress = mode == 1 ? 0 : GuestBase + (mode == 2 ? (ulong)bytes.Length : 512);
        var descriptor = new uint[] { (uint)dataAddress, (uint)(dataAddress >> 32), mode == 3 ? 0u : 64u, Raw };
        for (var i = 0; i < 4; i++) BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(1040 + i * 4), descriptor[i]);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(512), 0x12345678);
        var (tables, _) = MetalNative.CreateGuestMapping(GuestBase, bytes, 4096);
        void WriteBindings(nint command)
        {
            var memory = tables[0].Residents[0];
            MetalNative.CopyBindingWords(command, memory, 1024, 0, 8);
            MetalNative.CopyBindingWords(command, memory, 1040, 128, 16);
        }
        if (completeWriterFirst) MetalNative.CompleteBindingWriter(WriteBindings);
        var push = new byte[PushData.ByteSize];
        for (var i = 0; i < layout.UserDataRegisters.Count; i++)
            BinaryPrimitives.WriteUInt32LittleEndian(push.AsSpan((int)(layout.PushDataStartDword + i) * 4), user[layout.UserDataRegisters[i]]);
        Assert.True(MetalNative.TryCompileLibrary(shader!.Source, out var library, out var compileError), compileError);
        try
        {
            var initial = new byte[64];
            BinaryPrimitives.WriteUInt32LittleEndian(initial.AsSpan(16), 0xDEADBEEF);
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, initial, push, shader.ArgumentLayout!, 1,
                out var result, out _, out var runError, tables: tables, beforeCompute: completeWriterFirst ? null : WriteBindings), runError);
            Assert.Equal(expected, BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(16)));
            Assert.Equal(initial.AsSpan(0, 16).ToArray(), result.AsSpan(0, 16).ToArray());
        }
        finally { MetalNative.ReleaseCollisionLibrary(library); }
    }

    private static Gen5ShaderInstruction Load(uint pc, uint source, uint destination, uint count) =>
        new(pc, Gen5ShaderEncoding.Smem, $"SLoadDwordx{count}", [0u, 0u], [Gen5Operand.Scalar(source), Gen5Operand.Source(125)],
            Enumerable.Range((int)destination, (int)count).Select(index => Gen5Operand.Scalar((uint)index)).ToArray(),
            new Gen5ScalarMemoryControl(count, 0, null));

    private static Gen5ShaderInstruction Buffer(uint pc, bool store, uint resource) =>
        new(pc, Gen5ShaderEncoding.Mubuf, store ? "BufferStoreDword" : "BufferLoadDword", [0u, 0u],
            store ? [Gen5Operand.Vector(0), Gen5Operand.Scalar(resource), Gen5Operand.Source(125), Gen5Operand.Vector(4)]
                : [Gen5Operand.Vector(0), Gen5Operand.Scalar(resource), Gen5Operand.Source(125)],
            store ? [] : [Gen5Operand.Vector(4)], new Gen5BufferMemoryControl(1, 0, 4, resource, store ? 16 : 0, false, false, false, false));
}
