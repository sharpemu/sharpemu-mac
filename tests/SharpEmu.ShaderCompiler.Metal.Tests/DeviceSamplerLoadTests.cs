// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.ShaderCompiler.Resources;
using Xunit;

namespace SharpEmu.ShaderCompiler.Metal.Tests;

[Collection("Device binding compatibility")]
public sealed class DeviceSamplerLoadTests
{
    private const ulong GuestBase = 0x1_4000_0000;
    private static uint F(float value) => BitConverter.SingleToUInt32Bits(value);

    public static IEnumerable<object[]> Samples()
    {
        foreach (var volume in new[] { false, true })
        foreach (var completeWriterFirst in new[] { false, true })
        {
            // Independent results for a ramp texture: x + 10*y + 100*z.
            var scale = volume ? 111f : 11f;
            yield return [volume, 0u, 1.375f, 0u, 0u, 0u, 0u, scale, completeWriterFirst]; // repeat
            yield return [volume, 1u, 1.375f, 0u, 0u, 0u, 0u, 2 * scale, completeWriterFirst]; // mirror repeat
            yield return [volume, 2u, 1.375f, 0u, 0u, 0u, 0u, 3 * scale, completeWriterFirst]; // edge
            yield return [volume, 3u, -.375f, 0u, 0u, 0u, 0u, scale, completeWriterFirst]; // mirror once edge
            yield return [volume, 4u, -.5f, 0u, 2u, 0u, 0u, 0f, completeWriterFirst]; // half border
            yield return [volume, 5u, -1.5f, 0u, 2u, 0u, 0u, 1f, completeWriterFirst]; // mirror half border
            yield return [volume, 6u, -.25f, 0u, 2u, 0u, 0u, 1f, completeWriterFirst]; // border
            yield return [volume, 7u, -1.25f, 0u, 2u, 0u, 0u, 1f, completeWriterFirst]; // mirror once border
            yield return [volume, 4u, -.5f, 1u, 2u, 0u, 0u, volume ? .875f : .75f, completeWriterFirst];
            yield return [volume, 2u, .5f, 1u, 0u, 0u, 0u, 1.5f * scale, completeWriterFirst]; // linear
            yield return [volume, 2u, .5f, 1u, 0u, 256u, 1u, 1000f + .5f * scale, completeWriterFirst]; // mip clamp
            yield return [volume, 2u, .5f, 1u, 0u, 128u, 2u, 500f + scale, completeWriterFirst]; // trilinear
            yield return [volume, 2u, 1e20f, 0u, 0u, 0u, 0u, 3 * scale, completeWriterFirst];
            yield return [volume, 3u, -1e20f, 0u, 0u, 0u, 0u, 3 * scale, completeWriterFirst];
            yield return [volume, 6u, 1e20f, 0u, 2u, 0u, 0u, 1f, completeWriterFirst];
        }
    }

    [Theory]
    [MemberData(nameof(Samples))]
    public void PendingSamplerWordsAreConsumedOnDevice(bool volume, uint mode, float coordinate,
        uint filter, uint border, uint minLod, uint mipFilter, float expected, bool completeWriterFirst)
    {
        var words = new uint[] { mode | mode << 3 | mode << 6, minLod | 0xF00u << 12,
            filter << 20 | filter << 22 | mipFilter << 26, border << 30 };
        var (program, user) = Program(volume, coordinate);
        const ulong hash = 0xB1D5_AA00;
        var old = ShaderResourcePlan.Extract(program, ShaderStage.Compute, hash, 0, 16);
        var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, hash, 0, 16,
            deviceBindingMemory: new HashSet<int> { 0 });
        Assert.Equal(new[] { 0, 1, 2, 3 }, plan.DeviceBindingMemory!.Order());
        Assert.True(DeviceBindingLoads.Accepts(old, plan, out var reason), reason);
        Assert.True(Assert.Single(plan.Info.Samplers).Device);
        Assert.DoesNotContain(plan.TableReads, read => read.Value.MemoryIndex is >= 0 and <= 3);
        var snapshot = new ResourceSnapshot();
        var specialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(plan, new ResourceRuntimeInputs
        {
            UserData = user,
            ReadMemory = (ulong address, out uint value) => throw new InvalidOperationException($"host read {address:X}"),
            ResourceWordPending = _ => true,
        }, ref snapshot, ref specialization));
        var reference = new ResourceSnapshot();
        var referenceSpecialization = new ResourceSpecialization();
        Assert.True(ResourceMaterializer.Materialize(old, new ResourceRuntimeInputs
        {
            UserData = user,
            ReadMemory = (ulong address, out uint value) =>
            {
                value = words[(int)((address - GuestBase) / 4)];
                return true;
            },
        }, ref reference, ref referenceSpecialization));
        Assert.Equal(words, Assert.Single(reference.Samplers));
        var resources = ResourceMaterializer.ApplyTo(plan, specialization);
        var layout = BindingLayout.Allocate(resources.Info, BindingLayout.CollectUserDataRegisters(program, 0, 16), false,
            ShaderCompileRequest.RequiresFlattenedTable(plan, resources), false);
        var request = new ShaderCompileRequest(plan, resources, layout) { WaveSize = 32, LocalSizeX = 1 };
        Assert.True(Gen5MslTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Assert.True(MetalNative.TryCompileLibrary(shader!.Source, out var library, out error), error);
        try
        {
            if (Environment.GetEnvironmentVariable("SHARPEMU_TEST_REQUIRE_DEVICE") != "1") return;
            var bytes = new byte[1 << DeviceAddressPaging.PageBits];
            for (var i = 0; i < words.Length; i++) BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(1024 + i * 4), words[i]);
            // The original descriptor is zero. A queued GPU writer replaces it before
            // sampling, proving we use the device bytes, not the host placeholder.
            var (tables, _) = MetalNative.CreateGuestMapping(GuestBase, bytes, 4096);
            var texels = Enumerable.Range(0, 2).Select(mip => Texels(volume, mip)).ToArray();
            var texture = MetalNative.CreateFloatTexture(volume ? 7u : 2u, 4, 4, 1, volume ? 4u : 1u, texels);
            var push = new byte[PushData.ByteSize];
            for (var i = 0; i < layout.UserDataRegisters.Count; i++)
                BinaryPrimitives.WriteUInt32LittleEndian(push.AsSpan((int)(layout.PushDataStartDword + i) * 4), user[layout.UserDataRegisters[i]]);
            void WriteSampler(nint command) => MetalNative.CopyBindingWords(command, tables[0].Residents[0], 1024, 0, 16);
            if (completeWriterFirst) MetalNative.CompleteBindingWriter(WriteSampler);
            Assert.True(MetalNative.TryExecuteWithArgumentBuffer(library, shader.EntryPoint, new byte[64], push,
                shader.ArgumentLayout!, 1, out var result, out _, out error, texture, MetalNative.CreatePointSampler(), tables: tables,
                beforeCompute: completeWriterFirst ? null : WriteSampler), error);
            for (var i = 0; i < 4; i++)
                Assert.Equal(expected, BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(i * 4))), .001f);
            Assert.Equal(new byte[48], result.AsSpan(16).ToArray());
            // Cross-check the supported native address modes with Metal's immutable
            // sampler path. Half-border modes have no exact native representation.
            if (mode is not (4 or 5))
            {
                Assert.True(Gen5ImageFixtures.TryCompile("ImageSampleLz", volume ? 10u : 9u, out var nativeShader, out error,
                    addressValues: [F(coordinate), F(coordinate), F(coordinate)], storeResults: true, samplerWords: words), error);
                Assert.True(MetalNative.TryCompileLibrary(nativeShader!.Source, out var nativeLibrary, out error), error);
                try
                {
                    Assert.True(MetalNative.TryExecuteWithArgumentBuffer(nativeLibrary, nativeShader.EntryPoint, new byte[16],
                        new byte[PushData.ByteSize], nativeShader.ArgumentLayout!, 1, out var nativeResult, out error,
                        texture, MetalNative.CreateReferenceSampler(words)), error);
                    for (var i = 0; i < 4; i++)
                        Assert.Equal(BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(nativeResult.AsSpan(i * 4))),
                            BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(result.AsSpan(i * 4))), .001f);
                }
                finally { MetalNative.ReleaseCollisionLibrary(nativeLibrary); }
            }
        }
        finally { MetalNative.ReleaseCollisionLibrary(library); }
    }

    [Fact]
    public void UnsupportedSamplingInstructionsKeepThePlannerCheck()
    {
        var (program, _) = Program(false, .5f);
        program = program with { Instructions = program.Instructions.Select(instruction => instruction.Opcode == "ImageSampleLz"
            ? instruction with { Opcode = "ImageSampleL" } : instruction).ToArray() };
        var failure = Assert.Throws<ResourcePlanException>(() => ShaderResourcePlan.Extract(program, ShaderStage.Compute, 0xB1D5_AA01, 0, 16,
            deviceBindingMemory: new HashSet<int> { 0 }));
        Assert.Contains("pc=0x00000020", failure.Message);
        Assert.Contains("SamplerHandle dword 0 is not a valid runtime value", failure.Message);
    }

    [Fact]
    public void DisableSwitchMaterializesTheActualSamplerOnTheHost()
    {
        var enabled = DeviceBindingLoads.Enabled;
        try
        {
            DeviceBindingLoads.Enabled = false;
            var (program, user) = Program(true, .5f);
            var plan = ShaderResourcePlan.Extract(program, ShaderStage.Compute, 0xB1D5_AA02, 0, 16,
                deviceBindingMemory: new HashSet<int> { 0 });
            Assert.Null(plan.DeviceBindingMemory);
            Assert.False(Assert.Single(plan.Info.Samplers).Device);
            uint[] words = [0x08007492, 0x00FFF000, 0x00100000, 0x80000000];
            var snapshot = new ResourceSnapshot();
            var specialization = new ResourceSpecialization();
            Assert.True(ResourceMaterializer.Materialize(plan, new ResourceRuntimeInputs
            {
                UserData = user,
                ReadMemory = (ulong address, out uint value) => { value = words[(address - GuestBase) / 4]; return true; },
            }, ref snapshot, ref specialization));
            Assert.Equal(words, Assert.Single(snapshot.Samplers));
        }
        finally { DeviceBindingLoads.Enabled = enabled; }
    }

    private static (Gen5ShaderProgram, uint[]) Program(bool volume, float coordinate)
    {
        var instructions = new List<Gen5ShaderInstruction>
        {
            new(0, Gen5ShaderEncoding.Smem, "SLoadDwordx4", [0u, 0u], [Gen5Operand.Scalar(0), Gen5Operand.Source(125)],
                [Gen5Operand.Scalar(16), Gen5Operand.Scalar(17), Gen5Operand.Scalar(18), Gen5Operand.Scalar(19)], new Gen5ScalarMemoryControl(4, 0, null)),
        };
        for (uint i = 0; i < 3; i++)
            instructions.Add(new(8 + i * 8, Gen5ShaderEncoding.Vop1, "VMovB32", [], [Gen5Operand.Source(255, F(coordinate))], [Gen5Operand.Vector(i)], null));
        instructions.Add(new(32, Gen5ShaderEncoding.Mimg, "ImageSampleLz", [],
            [Gen5Operand.Vector(0), Gen5Operand.Scalar(8), Gen5Operand.Scalar(16)],
            [Gen5Operand.Vector(4), Gen5Operand.Vector(5), Gen5Operand.Vector(6), Gen5Operand.Vector(7)],
            new Gen5ImageControl(15, 0, [0, 1, 2], 4, 8, 16, volume ? 2u : 1u, false, false, false, false, false)));
        instructions.Add(new(40, Gen5ShaderEncoding.Mubuf, "BufferStoreDwordx4", [0u, 0u],
            [Gen5Operand.Vector(8), Gen5Operand.Scalar(4), Gen5Operand.Source(125), Gen5Operand.Vector(4), Gen5Operand.Vector(5), Gen5Operand.Vector(6), Gen5Operand.Vector(7)], [],
            new Gen5BufferMemoryControl(4, 8, 4, 4, 0, false, false, false, false)));
        instructions.Add(new(48, Gen5ShaderEncoding.Sopp, "SEndpgm", [0xBF810000], [], [], null));
        var user = new uint[16];
        user[0] = unchecked((uint)GuestBase); user[1] = (uint)(GuestBase >> 32);
        user[4] = 0x200000; user[6] = 64; user[7] = 0x00038FAC;
        user[8] = 0x20; user[9] = 71u << 20; user[11] = (volume ? 10u : 9u) << 28 | DescriptorConstants.IdentityImageSwizzle;
        return (new Gen5ShaderProgram(0x100000, instructions), user);
    }

    private static byte[] Texels(bool volume, int mip)
    {
        var size = 4 >> mip;
        var bytes = new byte[size * size * (volume ? size : 1) * 16];
        for (var z = 0; z < (volume ? size : 1); z++)
        for (var y = 0; y < size; y++)
        for (var x = 0; x < size; x++)
        for (var c = 0; c < 4; c++)
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(((z * size * size + y * size + x) * 4 + c) * 4), F(mip * 1000 + z * 100 + y * 10 + x));
        return bytes;
    }
}
